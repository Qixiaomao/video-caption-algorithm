#!/usr/bin/env python3
"""Validation-only, no-training prefix interventions with paired diagnostics."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys

# This diagnostic must use the existing local checkpoint/model cache.
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import torch.nn.functional as F
import transformers

from core.preprocessing.frame_loader import load_video_tensor
from scripts.diagnose_visual_path import load_model, resolve_frames_dir, sha256_file, tensor_stats
from src.models.caption_model import build_caption_supervision

VAL_PATH = ROOT / "data/processed/msvd_full/val/annotations.json"
VAL_SHA256 = "bcda3ba0583704a54297e4da4cc9acbcc5ec1ebffd6ed436a6ba3ee1b8a03c8b"
CORE = ("native", "repeat", "shuffled", "zero")
SCALES = {"scale_0": 0.0, "scale_025": 0.25, "scale_05": 0.5,
          "scale_1": 1.0, "scale_2": 2.0}
FULL = CORE + ("mean",) + tuple(SCALES)


def validate_validation_records(path, limit, *, expected_path=VAL_PATH,
                                expected_sha=VAL_SHA256, expected_count=100):
    path = Path(path).resolve()
    if path != Path(expected_path).resolve():
        raise ValueError("Only the pinned full-MSVD validation annotation path is allowed")
    if sha256_file(path) != expected_sha:
        raise ValueError("Validation annotation SHA-256 mismatch")
    records = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(records, list) or len(records) != expected_count:
        raise ValueError("Unexpected validation record count")
    if not 2 <= limit <= len(records):
        raise ValueError("limit must be between 2 and the validation record count")
    ids = [str(r["video_id"]) for r in records]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate validation video IDs")
    selected = records[:limit]
    for record in selected:
        refs = record.get("captions")
        if not isinstance(refs, list) or not refs or any(
            not isinstance(c, str) or not c.strip() for c in refs
        ):
            raise ValueError(f"Invalid references for {record['video_id']}")
        directory = resolve_frames_dir(record, path).resolve()
        # Fail closed: selected frames must belong to the validation tree too.
        if not directory.is_relative_to(path.parent):
            raise ValueError(f"Frames outside validation tree: {directory}")
        if len(list(directory.glob("frame_*.jpg"))) != 8:
            raise ValueError(f"Expected 8 frames, cannot silently skip: {directory}")
    return selected


def donor_permutation(count, seed):
    """A seeded single cycle: all donors are unique and never self-paired."""
    if count < 2:
        raise ValueError("At least two videos required")
    order = torch.randperm(count, generator=torch.Generator().manual_seed(seed))
    donors = torch.empty(count, dtype=torch.long)
    donors[order] = order.roll(1)
    return donors


def intervene(prefixes, name, donors):
    if name in ("native", "repeat"):
        return prefixes.clone()
    if name == "shuffled":
        return prefixes[donors.to(prefixes.device)].clone()
    if name == "zero":
        return torch.zeros_like(prefixes)
    if name == "mean":
        return prefixes.mean(0, keepdim=True).expand_as(prefixes).clone()
    if name in SCALES:
        return prefixes * SCALES[name]
    raise ValueError(f"Unknown intervention: {name}")


def describe(values):
    x = torch.as_tensor(values, dtype=torch.float64).flatten()
    if not x.numel() or not torch.isfinite(x).all():
        raise ValueError("Expected nonempty finite metric values")
    return {"mean": x.mean().item(), "median": x.quantile(0.5).item(),
            "std": x.std(unbiased=False).item(), "min": x.min().item(),
            "max": x.max().item()}


def js_divergence(left, right):
    """Jensen-Shannon divergence in nats, invariant to logit row offsets."""
    log_p = left.double().log_softmax(-1)
    log_q = right.double().log_softmax(-1)
    log_m = torch.logaddexp(log_p, log_q) - math.log(2)
    return (0.5 * ((log_p.exp() * (log_p - log_m)).sum(-1)
                   + (log_q.exp() * (log_q - log_m)).sum(-1))).clamp_min(0)


def paired_tensor_metrics(native, changed):
    a = native.double().reshape(native.shape[0], -1)
    b = changed.double().reshape(changed.shape[0], -1)
    delta = (a - b).norm(dim=-1)
    return {"cosine": describe(F.cosine_similarity(a, b, dim=-1)),
            "l2": describe(delta),
            "relative_l2_to_native": describe(delta / a.norm(dim=-1).clamp_min(1e-12))}


def paired_bootstrap(deltas, seed=123, resamples=10000):
    x = torch.as_tensor(deltas, dtype=torch.float64)
    if x.ndim != 1 or len(x) < 2 or not torch.isfinite(x).all() or resamples < 100:
        raise ValueError("Bootstrap requires >=2 finite video deltas and >=100 resamples")
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randint(len(x), (resamples, len(x)), generator=generator)
    means = x[indices].mean(-1)
    return {"mean": x.mean().item(), "ci95_low": means.quantile(0.025).item(),
            "ci95_high": means.quantile(0.975).item(), "resamples": resamples,
            "seed": seed, "resampling_unit": "video",
            "donor_assignment": "fixed; CI does not include rematching uncertainty"}


def caption_nll(logits, labels, prefix_len):
    """Predict caption token 0 from last prefix; include EOS, exclude -100 pads."""
    count = labels.shape[1]
    if prefix_len < 1 or logits.shape[1] < prefix_len + count - 1:
        raise ValueError("Insufficient logits for shifted caption supervision")
    aligned = logits[:, prefix_len - 1:prefix_len - 1 + count, :]
    losses = F.cross_entropy(aligned.float().transpose(1, 2), labels,
                             ignore_index=-100, reduction="none")
    valid = labels.ne(-100)
    if (valid.sum(-1) == 0).any():
        raise ValueError("Empty supervised caption")
    return losses, valid


def tokenize_references(tokenizer, captions, max_tokens):
    if max_tokens < 2:
        raise ValueError("max_tokens must leave room for text and EOS")
    result = []
    for caption in captions:
        original = tokenizer.encode(caption, add_special_tokens=False)
        if not original or tokenizer.eos_token_id in original:
            raise ValueError("Empty token sequence or embedded EOS in caption")
        result.append({"text": caption, "original_token_count": len(original),
                       "truncated": len(original) > max_tokens - 1,
                       "ids": original[:max_tokens - 1] + [tokenizer.eos_token_id]})
    return result


@torch.inference_mode()
def reference_nll(lm, tokenizer, prefix, references, microbatch):
    results = []
    for start in range(0, len(references), microbatch):
        group = references[start:start + microbatch]
        length = max(len(r["ids"]) for r in group)
        ids = torch.full((len(group), length), tokenizer.eos_token_id,
                         dtype=torch.long, device=prefix.device)
        for i, ref in enumerate(group):
            ids[i, :len(ref["ids"])] = torch.tensor(ref["ids"], device=prefix.device)
        text_mask, labels = build_caption_supervision(ids, tokenizer.eos_token_id)
        batch_prefix = prefix.expand(len(group), -1, -1)
        embeds = torch.cat([batch_prefix, lm.transformer.wte(ids)], dim=1)
        mask = torch.cat([torch.ones(batch_prefix.shape[:2], dtype=torch.long,
                                     device=prefix.device), text_mask], dim=1)
        logits = lm(inputs_embeds=embeds, attention_mask=mask, use_cache=False).logits
        losses, valid = caption_nll(logits, labels, prefix.shape[1])
        for i, ref in enumerate(group):
            tokens = losses[i][valid[i]].cpu().tolist()
            results.append({"nll": sum(tokens) / len(tokens), "first_token_nll": tokens[0],
                            "after_first_nll": sum(tokens[1:]) / len(tokens[1:]),
                            "token_count_including_eos": len(tokens), "token_nll": tokens})
    return results


@torch.inference_mode()
def probe_layers(lm, prefix):
    """Capture every block before final LN, final LN, and embedding boundary."""
    layers = {}
    handles = []

    def capture(name):
        def hook(module, args, output):
            state = output[0] if isinstance(output, tuple) else output
            layers[name] = state.detach().cpu().clone()
        return hook

    try:
        handles.append(lm.transformer.drop.register_forward_hook(capture("embedding_plus_position")))
        for i, block in enumerate(lm.transformer.h):
            handles.append(block.register_forward_hook(capture(f"block_{i + 1:02d}_pre_final_ln")))
        handles.append(lm.transformer.ln_f.register_forward_hook(capture("final_ln")))
        out = lm(inputs_embeds=prefix,
                 attention_mask=torch.ones(prefix.shape[:2], dtype=torch.long, device=prefix.device),
                 use_cache=False)
        return out.logits[:, -1].detach().cpu(), layers
    finally:
        for handle in handles:
            handle.remove()


def decode_options(max_new_tokens):
    return {"max_new_tokens": max_new_tokens, "min_new_tokens": 0,
            "num_beams": 1, "do_sample": False, "temperature": 1.0,
            "top_p": 0.9, "no_repeat_ngram_size": 3, "repetition_penalty": 1.0}


@torch.inference_mode()
def generate_prefix(lm, tokenizer, prefix, max_new_tokens):
    output = lm.generate(
        inputs_embeds=prefix,
        attention_mask=torch.ones(prefix.shape[:2], dtype=torch.long, device=prefix.device),
        pad_token_id=tokenizer.eos_token_id, eos_token_id=tokenizer.eos_token_id,
        **decode_options(max_new_tokens),
    )
    return [text.strip() for text in tokenizer.batch_decode(output, skip_special_tokens=True)]


def normalize_caption(text):
    return " ".join(text.lower().split())


def caption_summary(captions, native):
    if not captions or len(captions) != len(native):
        raise ValueError("Caption comparison requires equal, nonempty video sets")
    normalized = [normalize_caption(t) for t in captions]
    counts = Counter(normalized)
    return {"nonempty_rate": sum(bool(t) for t in normalized) / len(captions),
            "unique_rate": len(counts) / len(captions),
            "largest_template_rate": max(counts.values()) / len(captions),
            "changed_rate": sum(a != normalize_caption(b) for a, b in zip(normalized, native)) / len(captions),
            "top_templates": [{"caption": t, "count": n} for t, n in counts.most_common(10)]}


def summarize_condition(native, current, prefixes, native_prefixes, seed, resamples, smoke):
    a, b = native["logits"], current["logits"]
    jsd = js_divergence(a, b)
    deltas = [y["nll"] - x["nll"] for x, y in zip(native["videos"], current["videos"])]
    post_first = [y["after_first_nll"] - x["after_first_nll"]
                  for x, y in zip(native["videos"], current["videos"])]
    summary = {
        "prefix_distribution": tensor_stats(prefixes),
        "prefix_change": paired_tensor_metrics(native_prefixes, prefixes),
        "first_token": {"jsd_nats": describe(jsd), "jsd_per_video": jsd.tolist(),
                        "top1_flip_rate": float((a.argmax(-1) != b.argmax(-1)).float().mean()),
                        "raw_logit_max_abs_diff": float((a - b).abs().max()),
                        "vocab_centered_logits": paired_tensor_metrics(
                            a - a.mean(-1, keepdim=True), b - b.mean(-1, keepdim=True))},
        "caption": caption_summary([v["caption"] for v in current["videos"]],
                                   [v["caption"] for v in native["videos"]]),
        "paired_nll": {"delta_per_video": deltas,
                       "bootstrap": paired_bootstrap(deltas, seed, resamples),
                       "after_first_delta_per_video": post_first,
                       "after_first_bootstrap": paired_bootstrap(post_first, seed, resamples),
                       "interpretation_enabled": not smoke},
        "layers": {name: {
            "all_prefix_positions": paired_tensor_metrics(native["layers"][name], state),
            "last_prefix_position": paired_tensor_metrics(native["layers"][name][:, -1], state[:, -1])
        } for name, state in current["layers"].items()},
        "videos": current["videos"],
    }
    summary["heuristic_flags_not_acceptance_gates"] = {
        "top1_flip_ge_20_percent": summary["first_token"]["top1_flip_rate"] >= 0.2,
        "caption_change_ge_30_percent": summary["caption"]["changed_rate"] >= 0.3,
        "nll_ci_lower_gt_zero": None if smoke else summary["paired_nll"]["bootstrap"]["ci95_low"] > 0,
    }
    return summary


def check_identity(native, current, tolerance=1e-5):
    if (native["logits"] - current["logits"]).abs().max().item() > tolerance:
        raise RuntimeError("Identity-control logits are not reproducible")
    for a, b in zip(native["videos"], current["videos"]):
        if a["caption"] != b["caption"] or abs(a["nll"] - b["nll"]) > tolerance:
            raise RuntimeError("Identity-control caption/NLL is not reproducible")
    for name, state in native.get("layers", {}).items():
        if (state - current["layers"][name]).abs().max().item() > tolerance:
            raise RuntimeError(f"Identity-control hidden state changed at {name}")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--expected-checkpoint-sha256", required=True)
    parser.add_argument("--annotations", type=Path, default=VAL_PATH)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=32)
    parser.add_argument("--preset", choices=("core", "full"), default="full")
    parser.add_argument("--references-per-video", type=int, default=0, help="0=all references; smoke may use 1 or 2")
    parser.add_argument("--reference-batch-size", type=int, default=4)
    parser.add_argument("--max-text-tokens", type=int, default=40, help="Includes appended EOS; truncations are logged")
    parser.add_argument("--max-new-tokens", type=int, default=24)
    parser.add_argument("--bootstrap-resamples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args(argv)
    if (args.reference_batch_size < 1 or args.max_text_tokens < 2 or args.max_new_tokens < 1
            or args.bootstrap_resamples < 100 or args.threads < 1):
        parser.error("Invalid batch, token, bootstrap or thread limit")
    if args.smoke_test:
        if not 2 <= args.limit <= 4 or args.references_per_video not in (1, 2):
            parser.error("Smoke requires limit 2..4 and references-per-video 1..2")
    elif args.references_per_video != 0:
        parser.error("Full diagnostic must use all references (references-per-video=0)")
    return args


def main(argv=None):
    args = parse_args(argv)
    if args.output.exists():
        raise FileExistsError(f"Will not overwrite report: {args.output}")
    source_files = ["scripts/diagnose_prefix_sensitivity.py", "scripts/diagnose_visual_path.py",
                    "src/models/text_decoder.py", "src/models/caption_model.py",
                    "core/preprocessing/frame_loader.py"]
    initial_source_hashes = {f: sha256_file(ROOT / f) for f in source_files}
    records = validate_validation_records(args.annotations, args.limit)
    checkpoint_hash = sha256_file(args.checkpoint)
    if checkpoint_hash != args.expected_checkpoint_sha256.lower():
        raise ValueError("Checkpoint SHA-256 mismatch; model was not loaded")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    if str(args.device).startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(args.device)
    model, checkpoint = load_model(args.checkpoint, args.device)
    # Actual compatible tensors, not just a permissive loader return.
    model.load_state_dict(checkpoint.get("model_state", checkpoint), strict=True)
    model.requires_grad_(False).eval()
    checkpoint_metadata = {k: checkpoint.get(k) for k in (
        "checkpoint_version", "progress", "training_contract", "dataset_contract", "args")}
    del checkpoint
    lm, tokenizer = model.decoder.model, model.decoder.tokenizer
    if model.decoder.cond_mode != "prefix":
        raise ValueError("This protocol requires prefix conditioning")

    prefixes, references = [], []
    first_emb = None
    for i, record in enumerate(records):
        with torch.inference_mode():
            video = load_video_tensor(resolve_frames_dir(record, args.annotations), 8, 224, args.device)
            emb = model.proj(model.encoder(video))
            prefixes.append(model.decoder.build_visual_prefix(emb).cpu())
            if i == 0:
                first_emb = emb.clone()
        captions = record["captions"]
        if args.references_per_video:
            captions = captions[:args.references_per_video]
        references.append(tokenize_references(tokenizer, captions, args.max_text_tokens))
        print(f"[extract] {i + 1}/{len(records)} {record['video_id']}", flush=True)
    prefixes = torch.cat(prefixes, 0)
    if not torch.isfinite(prefixes).all():
        raise RuntimeError("Nonfinite prefix")
    donors = donor_permutation(len(records), args.seed)
    conditions = FULL if args.preset == "full" else CORE
    baseline = None
    reports = {}
    for name in conditions:
        altered = intervene(prefixes, name, donors)
        current = {"logits": [], "layers": {}, "videos": []}
        for i, record in enumerate(records):
            prefix = altered[i:i + 1].to(args.device)
            logits, layers = probe_layers(lm, prefix)
            current["logits"].append(logits)
            for layer_name, state in layers.items():
                current["layers"].setdefault(layer_name, []).append(state)
            captions = generate_prefix(lm, tokenizer, prefix, args.max_new_tokens)
            nll = reference_nll(lm, tokenizer, prefix, references[i], args.reference_batch_size)
            current["videos"].append({"video_id": record["video_id"], "caption": captions[0],
                "nll": sum(r["nll"] for r in nll) / len(nll),
                "after_first_nll": sum(r["after_first_nll"] for r in nll) / len(nll),
                "reference_count": len(nll), "references": nll,
                "first_token_id": logits.argmax(-1).item(),
                "first_token": tokenizer.decode([logits.argmax(-1).item()])})
            print(f"[{name}] {i + 1}/{len(records)}", flush=True)
        current["logits"] = torch.cat(current["logits"], 0)
        current["layers"] = {k: torch.cat(v, 0) for k, v in current["layers"].items()}
        if name == "native":
            baseline = current
            wrapper = model.decoder.generate(first_emb, prompt="", max_new_tokens=args.max_new_tokens,
                min_new_tokens=0, num_beams=1, temperature=1.0,
                no_repeat_ngram_size=3, repetition_penalty=1.0)
            if wrapper[0] != baseline["videos"][0]["caption"]:
                raise RuntimeError("Direct-prefix generation disagrees with existing wrapper")
        if name in ("repeat", "scale_1"):
            check_identity(baseline, current)
        if name == "shuffled":
            expected = baseline["logits"][donors]
            if (expected - current["logits"]).abs().max().item() > 1e-5 or any(
                current["videos"][i]["caption"] != baseline["videos"][int(j)]["caption"]
                for i, j in enumerate(donors)
            ):
                raise RuntimeError("Shuffled-prefix output does not follow its donor")
        reports[name] = summarize_condition(baseline, current, altered, prefixes,
            args.seed, args.bootstrap_resamples, args.smoke_test)

    final_hash = sha256_file(args.checkpoint)
    if final_hash != checkpoint_hash:
        raise RuntimeError("Checkpoint changed on disk during diagnostic")
    if sha256_file(args.annotations) != VAL_SHA256:
        raise RuntimeError("Validation annotations changed during diagnostic")
    if {f: sha256_file(ROOT / f) for f in source_files} != initial_source_hashes:
        raise RuntimeError("Diagnostic source changed during run; refusing ambiguous provenance")
    try:
        git_head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        git_status = subprocess.check_output(["git", "status", "--short"], cwd=ROOT, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        git_head, git_status = None, None
    result = {
        "protocol": "prefix-sensitivity-v1", "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "paper_ready": False, "smoke_test": args.smoke_test,
        "checkpoint": str(args.checkpoint.resolve()), "checkpoint_sha256": checkpoint_hash,
        "checkpoint_unchanged_after_run": True, "checkpoint_metadata": checkpoint_metadata,
        "annotations": str(args.annotations.resolve()), "annotation_sha256": VAL_SHA256,
        "sample_count": len(records), "video_ids": [r["video_id"] for r in records],
        "donor_indices": donors.tolist(),
        "donor_video_ids": [records[int(i)]["video_id"] for i in donors],
        "reference_policy": "first N only in smoke; otherwise all annotated references, equal reference then equal video weighting",
        "references": references, "truncated_reference_count": sum(r["truncated"] for group in references for r in group),
        "mean_prefix_scope": "fixed selected validation subset, never microbatch-local",
        "decode": {"prompt": "", "synthetic_bos": False, **decode_options(args.max_new_tokens)},
        "sanity": {"native_matches_wrapper_first_video": True, "identity_controls_passed": True,
                   "shuffle_tracks_donor": True, "weights_modified": False},
        "conditions": reports,
        "limits": ["Zero prefix retains prefix positions and attention; it is not a no-prefix LM baseline.",
                   "First-token 'a' agreement and raw-logit cosine/rank alone do not establish collapse.",
                   "Layer probes cover prefix-only inputs, not later autoregressive decoding states.",
                   "Layer-relative L2 can change with normalization; it does not isolate a causal failure layer.",
                   "Fixed-donor paired bootstrap is descriptive; no donor-rematching uncertainty is included.",
                   "Scale and zero interventions are potentially out of distribution; shuffled is the main semantic control."],
        "runtime": {"python": sys.version, "torch": torch.__version__, "transformers": transformers.__version__,
                    "device": args.device, "dtype": str(next(lm.parameters()).dtype),
                    "seed": args.seed, "video_batch_size": 1, "reference_batch_size": args.reference_batch_size,
                    "max_text_tokens": args.max_text_tokens, "git_head": git_head, "git_status": git_status,
                    "source_sha256": initial_source_hashes,
                    "source_unchanged_during_run": True},
    }
    if str(args.device).startswith("cuda"):
        result["runtime"].update(gpu=torch.cuda.get_device_name(args.device),
            peak_allocated_mb=torch.cuda.max_memory_allocated(args.device) / 2**20,
            peak_reserved_mb=torch.cuda.max_memory_reserved(args.device) / 2**20)
    payload = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        handle.write(payload)
    print(json.dumps({"output": str(args.output.resolve()), "sample_count": len(records),
                      "conditions": list(reports), "sanity": result["sanity"]}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
