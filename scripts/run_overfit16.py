#!/usr/bin/env python3
"""Run the preregistered sixteen-video memorization diagnostic."""
from __future__ import annotations

import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from transformers.utils.hub import cached_file

from scripts.diagnose_prefix_sensitivity import donor_permutation, reference_nll
from src.cli.train_caption_mapper import compute_loss_local, set_seed
from src.data.data_loader import build_dataloader
from src.models.caption_model import VideoCaptionModel
from src.training.checkpoint import (
    build_dataset_contract,
    capture_rng_state,
    save_training_checkpoint,
    sha256_file,
)

TRAIN = ROOT / "data/processed/msvd_full/train/annotations.json"
TRAIN_SHA256 = "ced443e28c5ad2ddd0780af55a820b5ec7da6462c092e1329951dc9c7598c7c5"
BASE_GPT2_SHA256 = "248dfc3911869ec493c76e65bf2fcf7f615828b0254c12b473182f0f81d3a707"
SELECTED = (
    ("WTf5EgVY5uU_98_104", "a woman breaks an egg"),
    ("WeOU0Iba1Xg_1_30", "a man galloping on his horse"),
    ("0wutCy2ZGOQ_4_10", "several boys are playing football in a fenced in area"),
    ("IWhrWLOAin0_1_4", "a large dog gets something out of a refrigerator"),
    ("O9cOSO9L8Zs_1_16", "a man is swinging on a rope"),
    ("02Z-kuB3IaM_2_13", "a black boar is running in the woods"),
    ("UbmZAe5u5FI_36_40", "a woman is squeezing the liquid out of cucumber slices with her hands over a glass bowl"),
    ("W8l_ezoU8Lc_55_70", "a man is playing a guitar"),
    ("1sffYOXq4Iw_23_49", "a person cutting a potatoes"),
    ("Dgf0VHMEtNs_57_66", "four people in surgical masks are operating on a man"),
    ("5JLi7L1Kwp8_23_30", "a cat is laying down"),
    ("ZbzDGXEwtGc_6_15", "a plane flying in the sky"),
    ("R8FDJgVW3Vc_0_4", "a woman is tapping her nails"),
    ("Td29-SApubM_48_57", "a polar bear pup looks into the mirror"),
    ("TsagxA4DKuc_28_41", "a man is sweeping a room"),
    ("9WpJxn-2h1o_122_150", "an old man is sitting in a chair reading the newspaper"),
)
MILESTONES = (0, 50, 100, 250, 500)
SOURCE_FILES = (
    "experiments/overfit16_protocol_20260904.md",
    "scripts/run_overfit16.py",
    "scripts/diagnose_prefix_sensitivity.py",
    "src/cli/train_caption_mapper.py",
    "src/data/data_loader.py",
    "src/models/caption_model.py",
    "src/models/text_decoder.py",
    "src/models/video_encoder.py",
    "src/training/checkpoint.py",
)


def tensor_sha(tensor: torch.Tensor) -> str:
    value = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str((str(value.dtype), tuple(value.shape))).encode())
    digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def state_sha(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for key in sorted(state):
        digest.update(key.encode())
        digest.update(tensor_sha(state[key]).encode())
    return digest.hexdigest()


def component_hashes(model: VideoCaptionModel) -> dict[str, str]:
    state = model.state_dict()
    return {
        "decoder": state_sha({k: v for k, v in state.items() if k.startswith("decoder.model.")}),
        "backbone": state_sha({k: v for k, v in state.items()
                               if k.startswith("encoder.") and not k.startswith("encoder.proj.")}),
        "projection": state_sha({k: v for k, v in state.items() if k.startswith("encoder.proj.")}),
        "mapper": state_sha({k: v for k, v in state.items() if k.startswith("decoder.mapper.")}),
    }


def normalize_caption(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", text.lower()))


def build_subset(output_root: Path) -> tuple[Path, list[dict]]:
    if sha256_file(TRAIN) != TRAIN_SHA256:
        raise RuntimeError("Pinned full training annotation hash changed")
    records = json.loads(TRAIN.read_text(encoding="utf-8"))
    by_id = {str(record["video_id"]): record for record in records}
    if len(by_id) != len(records):
        raise RuntimeError("Duplicate video IDs in full training annotations")
    subset = []
    for video_id, target in SELECTED:
        record = by_id.get(video_id)
        if record is None or target not in record.get("captions", []):
            raise RuntimeError(f"Pinned pair missing from source annotations: {video_id}")
        frames_dir = Path(record.get("frames_dir", ""))
        if not frames_dir.is_dir() or not any(frames_dir.glob("frame_*.jpg")):
            raise RuntimeError(f"Pinned video lacks extracted frames: {video_id}")
        item = {k: v for k, v in record.items() if not k.startswith("_")}
        item["captions"] = [target]
        subset.append(item)
    normalized = [normalize_caption(caption) for _, caption in SELECTED]
    if len(subset) != 16 or len(set(normalized)) != 16 or any(not text for text in normalized):
        raise RuntimeError("Overfit subset must contain sixteen unique nonempty targets")
    path = output_root / "subset_annotations.json"
    with path.open("x", encoding="utf-8") as handle:
        json.dump(subset, handle, ensure_ascii=False, indent=2)
    return path, subset


def configure_trainable(model: VideoCaptionModel) -> list[tuple[str, torch.nn.Parameter]]:
    model.requires_grad_(False)
    model.encoder.proj.requires_grad_(True)
    model.decoder.mapper.requires_grad_(True)
    named = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
    if not named or any(not name.startswith(("encoder.proj.", "decoder.mapper.")) for name, _ in named):
        raise RuntimeError("Unexpected trainable parameter group")
    return named


def batch_evidence(batch: dict, step: int, epoch: int) -> dict:
    rng = capture_rng_state()
    return {
        "step": step,
        "epoch": epoch,
        "video_ids": list(batch["video_id"]),
        "caption_ids_sha256": tensor_sha(batch["caption_ids"]),
        "video_tensor_sha256": tensor_sha(batch["video"]),
        "python_rng_sha256": hashlib.sha256(repr(rng["python"]).encode()).hexdigest(),
        "torch_cpu_rng_sha256": tensor_sha(rng["torch_cpu"]),
        "torch_cuda_rng_sha256": [tensor_sha(item) for item in rng["torch_cuda"] or []],
    }


@torch.inference_mode()
def generate_one(lm, tokenizer, prefix: torch.Tensor, max_new_tokens: int = 24) -> tuple[str, bool, list[int]]:
    output = lm.generate(
        inputs_embeds=prefix,
        attention_mask=torch.ones(prefix.shape[:2], dtype=torch.long, device=prefix.device),
        max_new_tokens=max_new_tokens,
        min_new_tokens=0,
        num_beams=1,
        do_sample=False,
        temperature=1.0,
        top_p=0.9,
        no_repeat_ngram_size=3,
        repetition_penalty=1.0,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    ids = output[0].detach().cpu().tolist()
    return tokenizer.decode(ids, skip_special_tokens=True).strip(), tokenizer.eos_token_id in ids, ids


@torch.inference_mode()
def evaluate_memorization(model: VideoCaptionModel, loader, targets: dict[str, str], step: int,
                          donors: torch.Tensor, device: str) -> dict:
    model.eval()
    prefixes = []
    video_ids = []
    for batch in loader:
        video = batch["video"].to(device)
        embedding = model.proj(model.encoder(video))
        prefixes.append(model.decoder.build_visual_prefix(embedding).detach().cpu())
        video_ids.extend(batch["video_id"])
    prefixes = torch.cat(prefixes, 0)
    if len(video_ids) != 16 or not torch.isfinite(prefixes).all():
        raise RuntimeError("Evaluation did not produce sixteen finite prefixes")

    tokenizer = model.decoder.tokenizer
    references = {
        video_id: [{"text": targets[video_id], "ids": tokenizer.encode(
            targets[video_id], add_special_tokens=False) + [tokenizer.eos_token_id]}]
        for video_id in video_ids
    }
    native_rows = []
    shuffled_rows = []
    for index, video_id in enumerate(video_ids):
        native_prefix = prefixes[index:index + 1].to(device)
        donor_index = int(donors[index])
        shuffled_prefix = prefixes[donor_index:donor_index + 1].to(device)
        native_nll = reference_nll(model.decoder.model, tokenizer, native_prefix,
                                   references[video_id], microbatch=1)[0]
        shuffled_nll = reference_nll(model.decoder.model, tokenizer, shuffled_prefix,
                                     references[video_id], microbatch=1)[0]
        native_text, native_eos, native_ids = generate_one(model.decoder.model, tokenizer, native_prefix)
        shuffled_text, shuffled_eos, shuffled_ids = generate_one(model.decoder.model, tokenizer, shuffled_prefix)
        native_rows.append({
            "video_id": video_id,
            "target": targets[video_id],
            "caption": native_text,
            "generated_ids": native_ids,
            "eos_terminated": native_eos,
            **native_nll,
        })
        shuffled_rows.append({
            "video_id": video_id,
            "donor_video_id": video_ids[donor_index],
            "caption": shuffled_text,
            "generated_ids": shuffled_ids,
            "eos_terminated": shuffled_eos,
            **shuffled_nll,
        })

    captions = [normalize_caption(row["caption"]) for row in native_rows]
    target_texts = [normalize_caption(row["target"]) for row in native_rows]
    counts = Counter(captions)
    deltas = [wrong["nll"] - right["nll"] for right, wrong in zip(native_rows, shuffled_rows)]
    shuffled_captions = [normalize_caption(row["caption"]) for row in shuffled_rows]
    mean_native = sum(row["nll"] for row in native_rows) / 16
    mean_delta = sum(deltas) / 16
    metrics = {
        "native_nll": mean_native,
        "shuffled_minus_native_nll": mean_delta,
        "positive_mismatch_videos": sum(delta > 0 for delta in deltas),
        "exact_match_count": sum(left == right for left, right in zip(captions, target_texts)),
        "exact_match_rate": sum(left == right for left, right in zip(captions, target_texts)) / 16,
        "nonempty_rate": sum(bool(text) for text in captions) / 16,
        "unique_count": len(counts),
        "unique_rate": len(counts) / 16,
        "largest_template_count": max(counts.values()),
        "largest_template_rate": max(counts.values()) / 16,
        "native_vs_shuffled_caption_change_rate": sum(
            left != right for left, right in zip(captions, shuffled_captions)) / 16,
        "eos_count": sum(row["eos_terminated"] for row in native_rows),
        "eos_rate": sum(row["eos_terminated"] for row in native_rows) / 16,
        "top_templates": [{"caption": text, "count": count} for text, count in counts.most_common()],
    }
    metrics["strict_memorization_gate"] = {
        "native_nll_le_0_50": mean_native <= 0.50,
        "shuffled_delta_ge_1_00": mean_delta >= 1.00,
        "exact_matches_ge_14": metrics["exact_match_count"] >= 14,
        "unique_ge_14": metrics["unique_count"] >= 14,
        "largest_template_le_2": metrics["largest_template_count"] <= 2,
        "eos_16_of_16": metrics["eos_count"] == 16,
    }
    metrics["strict_memorization_gate"]["pass"] = all(metrics["strict_memorization_gate"].values())
    return {"step": step, "metrics": metrics, "native": native_rows, "shuffled": shuffled_rows,
            "deltas": deltas, "donor_indices": donors.tolist(),
            "prefix_sha256": tensor_sha(prefixes)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--steps", type=int, choices=(2, 500), default=500,
                        help="2 is smoke only; 500 is the locked diagnostic")
    args = parser.parse_args()
    if args.output_root.exists():
        raise FileExistsError("Refusing to overwrite an existing diagnostic directory")
    if not torch.cuda.is_available():
        raise RuntimeError("The registered diagnostic requires the local CUDA GPU")

    args.output_root.mkdir(parents=True)
    subset_path, subset = build_subset(args.output_root)
    source_hashes = {name: sha256_file(ROOT / name) for name in SOURCE_FILES}
    full_contract = build_dataset_contract({"train": str(TRAIN)})
    subset_contract = build_dataset_contract({"overfit16": str(subset_path)})
    base_model_file = Path(cached_file("gpt2", "model.safetensors", local_files_only=True))
    if sha256_file(base_model_file) != BASE_GPT2_SHA256:
        raise RuntimeError("Original GPT-2 source hash changed")
    base_path = str(base_model_file.parent)

    set_seed(123)
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model = VideoCaptionModel(gpt2_name=base_path, freeze_vit=True,
                              vit_enable_torch_compile=False).to("cuda")
    named = configure_trainable(model)
    initial_hashes = component_hashes(model)
    optimizer = torch.optim.AdamW([parameter for _, parameter in named], lr=3e-4, weight_decay=0.01)
    if optimizer.state:
        raise RuntimeError("Optimizer unexpectedly inherited state")
    tokenizer = model.decoder.tokenizer
    token_audit = []
    for video_id, target in SELECTED:
        ids = tokenizer.encode(target, add_special_tokens=False)
        if not ids or len(ids) >= 64 or tokenizer.eos_token_id in ids:
            raise RuntimeError(f"Invalid target tokenization: {video_id}")
        token_audit.append({"video_id": video_id, "target": target, "tokens_without_eos": len(ids)})

    targets = dict(SELECTED)
    eval_loader = build_dataloader(str(subset_path), tokenizer, batch_size=8, max_len=64,
                                   num_frame=8, image_size=224, shuffle=False,
                                   num_wokers=0, random_caption=False)
    train_generator = torch.Generator().manual_seed(123)
    train_loader = build_dataloader(str(subset_path), tokenizer, batch_size=8, max_len=64,
                                    num_frame=8, image_size=224, shuffle=True,
                                    num_wokers=0, random_caption=False,
                                    generator=train_generator)
    if len(train_loader.dataset) != 16 or len(eval_loader.dataset) != 16 or len(train_loader) != 2:
        raise RuntimeError("Unexpected overfit subset loader shape")
    donors = donor_permutation(16, seed=321)
    trainable_schema = [{"name": name, "shape": list(parameter.shape), "numel": parameter.numel()}
                        for name, parameter in named]
    run_steps = args.steps
    milestones = (0, 2) if run_steps == 2 else MILESTONES
    preflight = {
        "protocol": "overfit16-v1",
        "paper_ready": False,
        "smoke_test": run_steps == 2,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": {"seed": 123, "donor_seed": 321, "batch_size": 8, "frames": 8,
                   "image_size": 224, "max_len": 64, "steps": run_steps,
                   "milestones": list(milestones), "optimizer": "AdamW", "lr": 3e-4,
                   "weight_decay": 0.01, "amp": False, "scheduler": False,
                   "gradient_clipping": False, "info_nce": False},
        "model_source": {"path": base_path, "model_file_sha256": BASE_GPT2_SHA256},
        "initial_component_hashes": initial_hashes,
        "trainable": trainable_schema,
        "token_audit": token_audit,
        "full_dataset_contract": full_contract,
        "subset_dataset_contract": subset_contract,
        "source_sha256": source_hashes,
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "runtime": {"python": sys.version, "torch": torch.__version__,
                    "gpu": torch.cuda.get_device_name(), "deterministic_algorithms": True,
                    "tf32": False},
    }
    with (args.output_root / "preflight.json").open("x", encoding="utf-8") as handle:
        json.dump(preflight, handle, ensure_ascii=False, indent=2, allow_nan=False)

    torch.cuda.reset_peak_memory_stats()
    evaluations = []
    model.eval()
    evaluations.append(evaluate_memorization(model, eval_loader, targets, 0, donors, "cuda"))
    print(json.dumps({"step": 0, **evaluations[-1]["metrics"]}, ensure_ascii=False), flush=True)
    set_seed(123)
    step = 0
    epoch = 0
    seen = Counter()
    losses = []
    started = time.monotonic()
    epoch_start_state = train_generator.get_state()
    trace_path = args.output_root / "trace.jsonl"
    metrics_path = args.output_root / "train_metrics.jsonl"
    with trace_path.open("x", encoding="utf-8") as trace, metrics_path.open("x", encoding="utf-8") as metrics:
        while step < run_steps:
            epoch_start_state = train_generator.get_state()
            for batch in train_loader:
                if step >= run_steps:
                    break
                step += 1
                for video_id in batch["video_id"]:
                    seen[video_id] += 1
                evidence = batch_evidence(batch, step, epoch)
                trace.write(json.dumps(evidence, sort_keys=True) + "\n")
                trace.flush()
                model.train()
                optimizer.zero_grad(set_to_none=True)
                video = batch["video"].to("cuda")
                captions = batch["caption_ids"].to("cuda")
                loss = compute_loss_local(model, video, captions, tokenizer.pad_token_id)
                if not torch.isfinite(loss):
                    raise RuntimeError(f"Nonfinite loss at step {step}")
                loss.backward()
                if any(parameter.grad is None or not torch.isfinite(parameter.grad).all()
                       for _, parameter in named):
                    raise RuntimeError(f"Missing/nonfinite trainable gradient at step {step}")
                optimizer.step()
                value = float(loss.detach())
                losses.append(value)
                metrics.write(json.dumps({"step": step, "epoch": epoch, "loss": value}) + "\n")
                metrics.flush()
                if step == 1 or step % 25 == 0:
                    print(f"[train] step={step}/{run_steps} loss={value:.6f} elapsed={time.monotonic()-started:.1f}s", flush=True)
                if step in milestones:
                    result = evaluate_memorization(model, eval_loader, targets, step, donors, "cuda")
                    evaluations.append(result)
                    print(json.dumps({"step": step, **result["metrics"]}, ensure_ascii=False), flush=True)
                del video, captions, loss
            epoch += 1

    final_hashes = component_hashes(model)
    if final_hashes["decoder"] != initial_hashes["decoder"] or final_hashes["backbone"] != initial_hashes["backbone"]:
        raise RuntimeError("Frozen decoder or ViT backbone changed")
    if any(final_hashes[key] == initial_hashes[key] for key in ("projection", "mapper")):
        raise RuntimeError("A trainable visual module did not change")
    if set(seen) != set(targets) or min(seen.values()) < 1:
        raise RuntimeError("Not every pinned sample was trained")

    contract = {
        "objective_version": 3,
        "caption_objective": "causal_lm_cross_entropy",
        "alignment_objective": "none",
        "alignment_weight": 0.0,
        "physical_batch_size": 8,
        "protocol": "overfit16-v1",
        "frozen_gpt2": True,
        "frozen_vit_backbone": True,
        "legacy_visual_weights_used": False,
    }
    saved_args = {"gpt2_name": base_path, "vit_name": "vit_base_patch16_224",
                  "cond_mode": "prefix", "prefix_len": 4, "unfreeze_gpt2_last": 0,
                  "freeze_vit": True, "seed": 123, "batch_size": 8, "max_len": 64,
                  "lr": 3e-4, "steps": run_steps}
    checkpoint_path = args.output_root / "last.pt"
    save_training_checkpoint(
        checkpoint_path,
        model=model,
        optimizer=optimizer,
        scheduler=None,
        progress={"step": step, "epoch": epoch, "epoch_complete": True,
                  "best_val": math.nan},
        args=saved_args,
        training_contract=contract,
        dataset_contract=subset_contract,
        loader_state={"current_generator_state": train_generator.get_state(),
                      "epoch_start_generator_state": epoch_start_state, "num_workers": 0},
    )
    final = evaluations[-1]
    classification = (
        "strict_memorization_pass" if final["metrics"]["strict_memorization_gate"]["pass"] else
        "teacher_forced_binding_but_generation_failure"
        if final["metrics"]["native_nll"] <= 0.50 and final["metrics"]["shuffled_minus_native_nll"] >= 1.00 else
        "target_fit_without_video_binding"
        if final["metrics"]["native_nll"] <= 0.50 else
        "tiny_set_not_fit_under_frozen_decoder_protocol"
    )
    if sha256_file(TRAIN) != TRAIN_SHA256:
        raise RuntimeError("Full training annotations changed during run")
    if source_hashes != {name: sha256_file(ROOT / name) for name in SOURCE_FILES}:
        raise RuntimeError("Experiment source changed during run")
    if sha256_file(base_model_file) != BASE_GPT2_SHA256:
        raise RuntimeError("Original GPT-2 source changed during run")
    report = {
        "protocol": "overfit16-v1",
        "paper_ready": False,
        "smoke_test": run_steps == 2,
        "classification": classification,
        "config": preflight["config"],
        "pairs": [{"video_id": video_id, "target": target} for video_id, target in SELECTED],
        "evaluations": evaluations,
        "initial_component_hashes": initial_hashes,
        "final_component_hashes": final_hashes,
        "frozen_weights_unchanged": True,
        "trainable_modules_changed": True,
        "seen_count_per_video": dict(seen),
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "trace_sha256": sha256_file(trace_path),
        "source_and_data_unchanged": True,
        "elapsed_seconds": time.monotonic() - started,
        "peak_allocated_mb": torch.cuda.max_memory_allocated() / 2**20,
        "peak_reserved_mb": torch.cuda.max_memory_reserved() / 2**20,
        "limits": [
            "Training-set memorization only; this is not validation or test performance.",
            "One seed, one fixed subset, a frozen decoder, and a fixed finite optimization budget.",
            "Failure does not alone separate supervision, optimization, and frozen-decoder capacity.",
            "Exact string match is deliberately strict and is interpreted with teacher-forced NLL.",
        ],
    }
    with (args.output_root / "result.json").open("x", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps({"output": str(args.output_root.resolve()), "classification": classification,
                      "final": final["metrics"], "checkpoint_sha256": report["checkpoint_sha256"]},
                     ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
