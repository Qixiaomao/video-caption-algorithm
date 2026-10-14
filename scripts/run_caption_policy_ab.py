#!/usr/bin/env python3
"""Run the preregistered fixed-caption versus multi-reference trial."""
from __future__ import annotations

import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import argparse
from collections import Counter
from datetime import datetime, timezone
import gc
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from transformers.utils.hub import cached_file

from scripts.diagnose_prefix_sensitivity import donor_permutation, reference_nll
from scripts.run_overfit16 import (
    BASE_GPT2_SHA256,
    SELECTED,
    TRAIN,
    TRAIN_SHA256,
    component_hashes,
    configure_trainable,
    generate_one,
    normalize_caption,
    state_sha,
    tensor_sha,
)
from src.data.data_loader import build_dataloader
from src.models.caption_model import VideoCaptionModel, build_caption_supervision
from src.training.checkpoint import (
    build_dataset_contract,
    capture_rng_state,
    save_training_checkpoint,
    sha256_file,
)
from src.cli.train_caption_mapper import set_seed

MILESTONES = (0, 50, 100, 250, 500)
SOURCE_FILES = (
    "experiments/caption_policy_ab_protocol_20260905.md",
    "scripts/run_caption_policy_ab.py",
    "scripts/run_overfit16.py",
    "scripts/diagnose_prefix_sensitivity.py",
    "src/data/data_loader.py",
    "src/models/caption_model.py",
    "src/models/text_decoder.py",
    "src/models/video_encoder.py",
    "src/training/checkpoint.py",
)


def write_json(path: Path, payload: object) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)


def reference_seed(video_id: str) -> int:
    digest = hashlib.sha256(f"caption-policy-v1:{video_id}".encode()).digest()
    return int.from_bytes(digest[:8], "big") % (2**63 - 1)


def make_reference_orders(records: dict[str, dict]) -> dict[str, list[int]]:
    orders = {}
    for video_id, _ in SELECTED:
        count = len(records[video_id]["captions"])
        generator = torch.Generator().manual_seed(reference_seed(video_id))
        orders[video_id] = torch.randperm(count, generator=generator).tolist()
    return orders


def make_batch_schedule(epochs: int) -> list[list[list[int]]]:
    generator = torch.Generator().manual_seed(456)
    return [torch.randperm(16, generator=generator).reshape(2, 8).tolist()
            for _ in range(epochs)]


def tokenize_caption(tokenizer, text: str, max_len: int = 64) -> torch.Tensor:
    ids = tokenizer.encode(text, add_special_tokens=False)
    if not ids or len(ids) >= max_len or tokenizer.eos_token_id in ids:
        raise ValueError("Caption is empty, contains EOS, or would be truncated")
    output = torch.full((max_len,), tokenizer.eos_token_id, dtype=torch.long)
    output[:len(ids)] = torch.tensor(ids, dtype=torch.long)
    return output


def tokenize_references(tokenizer, captions: list[str]) -> list[dict]:
    result = []
    for text in captions:
        ids = tokenizer.encode(text, add_special_tokens=False)
        if not ids or len(ids) >= 64 or tokenizer.eos_token_id in ids:
            raise ValueError("A stored reference is empty, contains EOS, or exceeds 63 tokens")
        result.append({"text": text, "ids": ids + [tokenizer.eos_token_id]})
    return result


@torch.inference_mode()
def raw_backbone_features(encoder, video: torch.Tensor) -> torch.Tensor:
    if not isinstance(encoder.dropout, torch.nn.Identity) or encoder.cfg.l2norm:
        raise RuntimeError("Cache protocol requires identity encoder dropout and no L2 normalization")
    if video.ndim != 5:
        raise ValueError("Expected [B,T,C,H,W] video tensor")
    batch, frames, channels, height, width = video.shape
    value = video.reshape(batch * frames, channels, height, width)
    if encoder.cfg.prefer_channels_last and value.device.type == "cuda":
        value = value.contiguous(memory_format=torch.channels_last)
    with encoder._sdpa_context(value):
        encoded = encoder._forward_backbone_impl(value)
    return encoder._spatial_temporal_pool(encoded, batch, frames)


@torch.inference_mode()
def cache_backbone(model: VideoCaptionModel, loader, device: str) -> tuple[torch.Tensor, list[str], dict]:
    model.eval()
    features = []
    ids = []
    differences = []
    tensor_hashes = []
    for batch in loader:
        video = batch["video"].to(device)
        direct = model.encoder(video)
        raw = raw_backbone_features(model.encoder, video)
        cached_path = model.encoder.proj(raw)
        differences.append(float((direct - cached_path).abs().max()))
        features.append(raw.cpu())
        ids.extend(batch["video_id"])
        tensor_hashes.append(tensor_sha(batch["video"]))
    features = torch.cat(features, 0)
    if ids != [video_id for video_id, _ in SELECTED]:
        raise RuntimeError("Cached feature order differs from the pinned subset")
    maximum = max(differences)
    if maximum > 1e-6 or not torch.isfinite(features).all():
        raise RuntimeError(f"Frozen-backbone cache equivalence failed: {maximum}")
    return features, ids, {"max_abs_error": maximum, "pass": True,
                           "raw_feature_sha256": tensor_sha(features),
                           "video_batch_sha256": tensor_hashes}


def cached_caption_loss(model: VideoCaptionModel, raw_features: torch.Tensor,
                        caption_ids: torch.Tensor) -> torch.Tensor:
    video_embedding = model.proj(model.encoder.proj(raw_features))
    mask, labels = build_caption_supervision(caption_ids, model.decoder.tokenizer.pad_token_id)
    return model.decoder(video_embedding, caption_ids, mask, labels)["loss"]


def token_f1(left: str, right: str) -> float:
    a = normalize_caption(left).split()
    b = normalize_caption(right).split()
    if not a or not b:
        return 0.0
    common = sum((Counter(a) & Counter(b)).values())
    precision = common / len(a)
    recall = common / len(b)
    return 2 * precision * recall / (precision + recall) if common else 0.0


@torch.inference_mode()
def evaluate_arm(model: VideoCaptionModel, raw_features: torch.Tensor, video_ids: list[str],
                 canonical: dict[str, str], references: dict[str, list[dict]],
                 donors: torch.Tensor, step: int, include_all_references: bool,
                 device: str) -> dict:
    model.eval()
    features = raw_features.to(device)
    embeddings = model.proj(model.encoder.proj(features))
    prefixes = model.decoder.build_visual_prefix(embeddings)
    tokenizer = model.decoder.tokenizer
    native_rows = []
    shuffled_rows = []
    for index, video_id in enumerate(video_ids):
        native_prefix = prefixes[index:index + 1]
        donor_index = int(donors[index])
        wrong_prefix = prefixes[donor_index:donor_index + 1]
        canonical_ref = [{"text": canonical[video_id],
                          "ids": tokenizer.encode(canonical[video_id], add_special_tokens=False)
                                 + [tokenizer.eos_token_id]}]
        native_nll = reference_nll(model.decoder.model, tokenizer, native_prefix,
                                   canonical_ref, microbatch=1)[0]
        wrong_nll = reference_nll(model.decoder.model, tokenizer, wrong_prefix,
                                  canonical_ref, microbatch=1)[0]
        native_text, native_eos, native_ids = generate_one(model.decoder.model, tokenizer, native_prefix)
        wrong_text, wrong_eos, wrong_ids = generate_one(model.decoder.model, tokenizer, wrong_prefix)
        reference_texts = [item["text"] for item in references[video_id]]
        normalized_refs = {normalize_caption(text) for text in reference_texts}
        native_rows.append({"video_id": video_id, "target": canonical[video_id],
                            "caption": native_text, "generated_ids": native_ids,
                            "eos_terminated": native_eos,
                            "exact_any_reference": normalize_caption(native_text) in normalized_refs,
                            "max_reference_token_f1": max(token_f1(native_text, text)
                                                          for text in reference_texts),
                            **native_nll})
        shuffled_rows.append({"video_id": video_id, "donor_video_id": video_ids[donor_index],
                              "caption": wrong_text, "generated_ids": wrong_ids,
                              "eos_terminated": wrong_eos, **wrong_nll})

    normalized = [normalize_caption(row["caption"]) for row in native_rows]
    normalized_wrong = [normalize_caption(row["caption"]) for row in shuffled_rows]
    targets = [normalize_caption(row["target"]) for row in native_rows]
    counts = Counter(normalized)
    deltas = [wrong["nll"] - native["nll"] for native, wrong in zip(native_rows, shuffled_rows)]
    metrics = {
        "canonical_native_nll": sum(row["nll"] for row in native_rows) / 16,
        "canonical_shuffled_minus_native_nll": sum(deltas) / 16,
        "canonical_positive_mismatch_videos": sum(value > 0 for value in deltas),
        "exact_canonical_count": sum(left == right for left, right in zip(normalized, targets)),
        "exact_any_reference_count": sum(row["exact_any_reference"] for row in native_rows),
        "mean_max_reference_token_f1": sum(row["max_reference_token_f1"] for row in native_rows) / 16,
        "nonempty_count": sum(bool(text) for text in normalized),
        "unique_count": len(counts),
        "largest_template_count": max(counts.values()),
        "eos_count": sum(row["eos_terminated"] for row in native_rows),
        "native_vs_shuffled_caption_change_count": sum(
            left != right for left, right in zip(normalized, normalized_wrong)),
        "top_templates": [{"caption": text, "count": count} for text, count in counts.most_common()],
    }
    result = {"step": step, "metrics": metrics, "native": native_rows,
              "shuffled": shuffled_rows, "canonical_delta_per_video": deltas,
              "prefix_sha256": tensor_sha(prefixes)}
    if include_all_references:
        per_video = []
        for index, video_id in enumerate(video_ids):
            native = reference_nll(model.decoder.model, tokenizer, prefixes[index:index + 1],
                                   references[video_id], microbatch=8)
            donor_index = int(donors[index])
            wrong = reference_nll(model.decoder.model, tokenizer,
                                  prefixes[donor_index:donor_index + 1],
                                  references[video_id], microbatch=8)
            native_mean = sum(item["nll"] for item in native) / len(native)
            wrong_mean = sum(item["nll"] for item in wrong) / len(wrong)
            per_video.append({"video_id": video_id, "reference_count": len(native),
                              "native_nll": native_mean, "shuffled_nll": wrong_mean,
                              "delta": wrong_mean - native_mean})
        native_mean = sum(item["native_nll"] for item in per_video) / 16
        mean_delta = sum(item["delta"] for item in per_video) / 16
        all_reference = {"native_nll": native_mean,
                         "shuffled_minus_native_nll": mean_delta,
                         "positive_mismatch_videos": sum(item["delta"] > 0 for item in per_video),
                         "per_video": per_video,
                         "reference_count": sum(item["reference_count"] for item in per_video),
                         "weighting": "reference mean within video, then equal video mean"}
        gate = {
            "native_nll_le_2_50": native_mean <= 2.50,
            "shuffled_delta_ge_1_00": mean_delta >= 1.00,
            "positive_mismatch_ge_14": all_reference["positive_mismatch_videos"] >= 14,
            "caption_changes_ge_12": metrics["native_vs_shuffled_caption_change_count"] >= 12,
            "unique_ge_12": metrics["unique_count"] >= 12,
            "eos_ge_15": metrics["eos_count"] >= 15,
        }
        gate["pass"] = all(gate.values())
        all_reference["long_run_readiness_gate"] = gate
        result["all_references"] = all_reference
    return result


def trace_evidence(batch_indices: list[int], video_ids: list[str], features: torch.Tensor,
                   captions: torch.Tensor, step: int, epoch: int, policy: str,
                   target_texts: list[str]) -> dict:
    rng = capture_rng_state()
    return {"step": step, "epoch": epoch, "policy": policy,
            "indices": batch_indices,
            "video_ids": [video_ids[index] for index in batch_indices],
            "target_texts": target_texts,
            "caption_ids_sha256": tensor_sha(captions),
            "visual_features_sha256": tensor_sha(features),
            "python_rng_sha256": hashlib.sha256(repr(rng["python"]).encode()).hexdigest(),
            "torch_cpu_rng_sha256": tensor_sha(rng["torch_cpu"]),
            "torch_cuda_rng_sha256": [tensor_sha(value) for value in rng["torch_cuda"] or []]}


def compare_traces(left: Path, right: Path, expected_steps: int) -> dict:
    a = [json.loads(line) for line in left.read_text(encoding="utf-8").splitlines()]
    b = [json.loads(line) for line in right.read_text(encoding="utf-8").splitlines()]
    if len(a) != expected_steps or len(b) != expected_steps:
        raise RuntimeError("Trace length mismatch")
    treatment_keys = {"policy", "target_texts", "caption_ids_sha256"}
    differing_captions = 0
    for step, (first, second) in enumerate(zip(a, b), 1):
        common_first = {key: value for key, value in first.items() if key not in treatment_keys}
        common_second = {key: value for key, value in second.items() if key not in treatment_keys}
        if common_first != common_second:
            keys = [key for key in common_first if common_first.get(key) != common_second.get(key)]
            raise RuntimeError(f"Non-treatment trace mismatch at step {step}: {keys}")
        differing_captions += first["caption_ids_sha256"] != second["caption_ids_sha256"]
    if differing_captions < expected_steps * 0.9:
        raise RuntimeError("Caption treatment was absent from too many steps")
    return {"matched_steps": expected_steps, "all_non_treatment_inputs_and_rng_identical": True,
            "caption_treatment_different_steps": differing_captions,
            "left_sha256": sha256_file(left), "right_sha256": sha256_file(right)}


def run_arm(label: str, common_path: Path, raw_features: torch.Tensor, video_ids: list[str],
            canonical: dict[str, str], records: dict[str, dict], references: dict[str, list[dict]],
            reference_orders: dict[str, list[int]], batch_schedule: list[list[list[int]]],
            donors: torch.Tensor, directory: Path, run_steps: int, source_path: str,
            subset_contract: dict, expected_initial: dict) -> dict:
    directory.mkdir()
    set_seed(123)
    model = VideoCaptionModel(gpt2_name=source_path, freeze_vit=True,
                              vit_enable_torch_compile=False)
    common = torch.load(common_path, map_location="cpu", weights_only=True)
    model.load_state_dict(common["model_state"], strict=True)
    if component_hashes(model) != expected_initial:
        raise RuntimeError("Arm initialization differs from common state")
    model = model.to("cuda")
    named = configure_trainable(model)
    optimizer = torch.optim.AdamW([parameter for _, parameter in named], lr=3e-4, weight_decay=0.01)
    caption_cache = {text: tokenize_caption(model.decoder.tokenizer, text)
                     for record in records.values() for text in record["captions"]}
    milestones = (0, 2) if run_steps == 2 else MILESTONES
    evaluations = []
    set_seed(123)
    evaluations.append(evaluate_arm(model, raw_features, video_ids, canonical, references,
                                    donors, 0, False, "cuda"))
    set_seed(123)
    trace_path = directory / "trace.jsonl"
    metric_path = directory / "train_metrics.jsonl"
    seen = Counter()
    caption_indices_seen = {video_id: Counter() for video_id in video_ids}
    step = 0
    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    with trace_path.open("x", encoding="utf-8") as trace, metric_path.open("x", encoding="utf-8") as metrics:
        for epoch, epoch_batches in enumerate(batch_schedule):
            for batch_indices in epoch_batches:
                if step >= run_steps:
                    break
                step += 1
                texts = []
                for index in batch_indices:
                    video_id = video_ids[index]
                    seen[video_id] += 1
                    if label == "A_fixed":
                        text = canonical[video_id]
                        caption_index = records[video_id]["captions"].index(text)
                    else:
                        order = reference_orders[video_id]
                        caption_index = order[epoch % len(order)]
                        text = records[video_id]["captions"][caption_index]
                    caption_indices_seen[video_id][caption_index] += 1
                    texts.append(text)
                batch_features = raw_features[batch_indices]
                captions = torch.stack([caption_cache[text] for text in texts])
                evidence = trace_evidence(batch_indices, video_ids, batch_features, captions,
                                          step, epoch, label, texts)
                trace.write(json.dumps(evidence, ensure_ascii=False, sort_keys=True) + "\n")
                trace.flush()
                model.train()
                model.encoder.backbone.eval()
                optimizer.zero_grad(set_to_none=True)
                loss = cached_caption_loss(model, batch_features.to("cuda"), captions.to("cuda"))
                if not torch.isfinite(loss):
                    raise RuntimeError(f"Nonfinite loss at {label} step {step}")
                loss.backward()
                if any(parameter.grad is None or not torch.isfinite(parameter.grad).all()
                       for _, parameter in named):
                    raise RuntimeError(f"Missing/nonfinite gradient at {label} step {step}")
                optimizer.step()
                value = float(loss.detach())
                metrics.write(json.dumps({"step": step, "epoch": epoch, "loss": value}) + "\n")
                metrics.flush()
                if step == 1 or step % 25 == 0:
                    print(f"[{label}] step={step}/{run_steps} loss={value:.6f} elapsed={time.monotonic()-started:.1f}s", flush=True)
                if step in milestones:
                    evaluation = evaluate_arm(model, raw_features, video_ids, canonical, references,
                                              donors, step, run_steps == 500 and step == 500, "cuda")
                    evaluations.append(evaluation)
                    compact = dict(evaluation["metrics"])
                    compact.pop("top_templates")
                    if "all_references" in evaluation:
                        compact["all_references"] = {key: value for key, value in evaluation["all_references"].items()
                                                     if key != "per_video"}
                    print(json.dumps({"label": label, "step": step, **compact}, ensure_ascii=False), flush=True)
            if step >= run_steps:
                break
    final_hashes = component_hashes(model)
    if final_hashes["decoder"] != expected_initial["decoder"] or final_hashes["backbone"] != expected_initial["backbone"]:
        raise RuntimeError("A frozen component changed")
    if final_hashes["projection"] == expected_initial["projection"] or final_hashes["mapper"] == expected_initial["mapper"]:
        raise RuntimeError("A trainable component did not change")
    if set(seen) != set(video_ids) or min(seen.values()) < 1:
        raise RuntimeError("Not every video was trained")
    model.eval()
    direct_diffs = []
    subset_loader = build_dataloader(str(directory.parent / "subset_annotations.json"), model.decoder.tokenizer,
                                     batch_size=8, max_len=64, num_frame=8, image_size=224,
                                     shuffle=False, num_wokers=0, random_caption=False)
    offset = 0
    with torch.inference_mode():
        for batch in subset_loader:
            count = len(batch["video_id"])
            direct = model.encoder(batch["video"].to("cuda"))
            cached = model.encoder.proj(raw_features[offset:offset + count].to("cuda"))
            direct_diffs.append(float((direct - cached).abs().max()))
            offset += count
    if max(direct_diffs) > 1e-6:
        raise RuntimeError("Post-training cached/direct visual equivalence failed")
    checkpoint = directory / "last.pt"
    contract = {"objective_version": 3, "caption_objective": "causal_lm_cross_entropy",
                "caption_policy": label, "alignment_objective": "none", "alignment_weight": 0.0,
                "physical_batch_size": 8, "protocol": "caption-policy-ab-v1",
                "frozen_gpt2": True, "frozen_vit_backbone": True,
                "cached_frozen_backbone": True, "legacy_visual_weights_used": False}
    save_training_checkpoint(checkpoint, model=model, optimizer=optimizer, scheduler=None,
        progress={"step": step, "epoch": math.ceil(step / 2), "epoch_complete": step % 2 == 0,
                  "best_val": 0.0},
        args={"gpt2_name": source_path, "vit_name": "vit_base_patch16_224", "cond_mode": "prefix",
              "prefix_len": 4, "freeze_vit": True, "seed": 123, "batch_size": 8,
              "max_len": 64, "lr": 3e-4, "steps": run_steps, "caption_policy": label},
        training_contract=contract, dataset_contract=subset_contract,
        loader_state={"batch_schedule_sha256": state_sha({"schedule": torch.tensor(batch_schedule)}),
                      "num_workers": 0})
    result = {"label": label, "steps": step, "evaluations": evaluations,
              "initial_component_hashes": expected_initial, "final_component_hashes": final_hashes,
              "frozen_weights_unchanged": True, "trainable_modules_changed": True,
              "seen_count_per_video": dict(seen),
              "caption_coverage": {video_id: {"unique_indices": len(counts),
                                               "minimum_visits": min(counts.values()),
                                               "maximum_visits": max(counts.values())}
                                   for video_id, counts in caption_indices_seen.items()},
              "post_train_cache_equivalence_max_abs_error": max(direct_diffs),
              "checkpoint": str(checkpoint.resolve()), "checkpoint_sha256": sha256_file(checkpoint),
              "trace_sha256": sha256_file(trace_path), "elapsed_seconds": time.monotonic() - started,
              "peak_allocated_mb": torch.cuda.max_memory_allocated() / 2**20,
              "peak_reserved_mb": torch.cuda.max_memory_reserved() / 2**20}
    write_json(directory / "result.json", result)
    del model, optimizer
    gc.collect()
    torch.cuda.empty_cache()
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--steps", type=int, choices=(2, 500), default=500)
    args = parser.parse_args()
    if args.output_root.exists():
        raise FileExistsError("Refusing to overwrite an existing trial")
    if not torch.cuda.is_available():
        raise RuntimeError("This registered trial requires the local CUDA GPU")
    args.output_root.mkdir(parents=True)
    if sha256_file(TRAIN) != TRAIN_SHA256:
        raise RuntimeError("Pinned MSVD training annotations changed")
    source_hashes = {name: sha256_file(ROOT / name) for name in SOURCE_FILES}
    all_records_list = json.loads(TRAIN.read_text(encoding="utf-8"))
    all_records = {str(record["video_id"]): record for record in all_records_list}
    subset = []
    canonical = dict(SELECTED)
    for video_id, target in SELECTED:
        record = all_records[video_id]
        if target not in record["captions"]:
            raise RuntimeError("Canonical target missing from source record")
        item = {key: value for key, value in record.items() if not key.startswith("_")}
        item["captions"] = [target]
        subset.append(item)
    subset_path = args.output_root / "subset_annotations.json"
    write_json(subset_path, subset)
    subset_contract = build_dataset_contract({"overfit16": str(subset_path)})
    full_contract = build_dataset_contract({"train": str(TRAIN)})
    base_model_file = Path(cached_file("gpt2", "model.safetensors", local_files_only=True))
    if sha256_file(base_model_file) != BASE_GPT2_SHA256:
        raise RuntimeError("Original GPT-2 model source changed")
    source_path = str(base_model_file.parent)
    set_seed(123)
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    common_model = VideoCaptionModel(gpt2_name=source_path, freeze_vit=True,
                                     vit_enable_torch_compile=False)
    initial = component_hashes(common_model)
    common_state = {key: value.detach().cpu().clone() for key, value in common_model.state_dict().items()}
    common_path = args.output_root / "common_initial.pt"
    torch.save({"model_state": common_state}, common_path)
    common_model = common_model.to("cuda")
    loader = build_dataloader(str(subset_path), common_model.decoder.tokenizer, batch_size=8,
                              max_len=64, num_frame=8, image_size=224, shuffle=False,
                              num_wokers=0, random_caption=False)
    raw_features, video_ids, cache_audit = cache_backbone(common_model, loader, "cuda")
    references = {video_id: tokenize_references(common_model.decoder.tokenizer,
                                                all_records[video_id]["captions"])
                  for video_id in video_ids}
    if sum(len(value) for value in references.values()) != 668:
        raise RuntimeError("Unexpected selected reference count")
    reference_orders = make_reference_orders(all_records)
    run_steps = args.steps
    epochs = math.ceil(run_steps / 2)
    batch_schedule = make_batch_schedule(epochs)
    donors = donor_permutation(16, seed=321)
    schedule_path = args.output_root / "schedule.json"
    schedule_payload = {"batch_seed": 456, "donor_seed": 321,
                        "batch_schedule": batch_schedule, "donor_indices": donors.tolist(),
                        "reference_seeds": {video_id: reference_seed(video_id) for video_id in video_ids},
                        "reference_orders": reference_orders}
    write_json(schedule_path, schedule_payload)
    preflight = {"protocol": "caption-policy-ab-v1", "paper_ready": False,
        "smoke_test": run_steps == 2, "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": {"seed": 123, "batch_seed": 456, "donor_seed": 321, "steps": run_steps,
                   "epochs": epochs, "batch_size": 8, "frames": 8, "image_size": 224,
                   "max_len": 64, "lr": 3e-4, "weight_decay": 0.01,
                   "milestones": [0, 2] if run_steps == 2 else list(MILESTONES)},
        "arms": {"A_fixed": "one pinned canonical caption per video",
                 "B_multi": "deterministic balanced cycles over every stored reference"},
        "selected_reference_count": 668,
        "reference_count_per_video": {video_id: len(references[video_id]) for video_id in video_ids},
        "cache_audit": cache_audit, "initial_component_hashes": initial,
        "common_initial_sha256": sha256_file(common_path),
        "schedule_sha256": sha256_file(schedule_path),
        "full_dataset_contract": full_contract, "subset_dataset_contract": subset_contract,
        "source_sha256": source_hashes, "model_source": {"path": source_path,
                                                           "sha256": BASE_GPT2_SHA256},
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "runtime": {"python": sys.version, "torch": torch.__version__,
                    "gpu": torch.cuda.get_device_name(), "deterministic_algorithms": True, "tf32": False}}
    write_json(args.output_root / "preflight.json", preflight)
    del common_model, loader, common_state
    gc.collect()
    torch.cuda.empty_cache()
    arms = {}
    for label in ("A_fixed", "B_multi"):
        arms[label] = run_arm(label, common_path, raw_features, video_ids, canonical,
                              {video_id: all_records[video_id] for video_id in video_ids},
                              references, reference_orders, batch_schedule, donors,
                              args.output_root / label, run_steps, source_path,
                              subset_contract, initial)
    matching = compare_traces(args.output_root / "A_fixed/trace.jsonl",
                              args.output_root / "B_multi/trace.jsonl", run_steps)
    if arms["A_fixed"]["evaluations"][0] != arms["B_multi"]["evaluations"][0]:
        raise RuntimeError("Step-zero arm endpoints differ")
    if sha256_file(TRAIN) != TRAIN_SHA256 or source_hashes != {
            name: sha256_file(ROOT / name) for name in SOURCE_FILES}:
        raise RuntimeError("Source or data changed during the trial")
    final_multi = arms["B_multi"]["evaluations"][-1]
    decision = "smoke_only"
    if run_steps == 500:
        decision = ("proceed_full_msvd_balanced_multi_reference"
                    if final_multi["all_references"]["long_run_readiness_gate"]["pass"]
                    else "do_not_run_old_random_multi_reference_policy")
    result = {"protocol": "caption-policy-ab-v1", "paper_ready": False,
              "smoke_test": run_steps == 2, "decision": decision,
              "matching": matching, "arms": arms, "source_and_data_unchanged": True,
              "artifacts": {"preflight": str((args.output_root / "preflight.json").resolve()),
                            "schedule": str(schedule_path.resolve()),
                            "common_initial": str(common_path.resolve())},
              "limits": ["One seed, one sixteen-video subset, and a frozen decoder.",
                         "Backbone features are cached only after direct-path equivalence validation.",
                         "Passing is permission to test full MSVD, not a guarantee of validation quality.",
                         "Token-F1 is a triage statistic and is not a benchmark metric."]}
    write_json(args.output_root / "result.json", result)
    print(json.dumps({"output": str(args.output_root.resolve()), "decision": decision,
                      "matching": matching,
                      "final": {label: arm["evaluations"][-1].get("all_references", arm["evaluations"][-1]["metrics"])
                                for label, arm in arms.items()}}, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
