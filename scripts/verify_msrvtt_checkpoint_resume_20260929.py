"""Verify strict MSR-VTT epoch-20 resume against the uninterrupted trace.

This creates an independent verification run and never modifies the source run.
Only train/cache data are loaded; the test split is not opened.
"""
from __future__ import annotations

import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import argparse
from datetime import datetime, timezone
import itertools
import json
import math
from pathlib import Path
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from scripts import run_msrvtt_ab_formal_20260928 as formal
from src.training.checkpoint import sha256_file


def write_new(path: Path, payload: object) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, ensure_ascii=False, allow_nan=False)


def expected_training_contract(arm: str) -> dict:
    return {
        "objective_version": 3,
        "caption_objective": "causal_lm_cross_entropy",
        "caption_policy": "balanced_all_references",
        "alignment_objective": "infonce" if arm == "B" else "none",
        "alignment_weight": 0.1 if arm == "B" else 0.0,
        "info_nce_temperature": 0.07,
        "false_negative_policy": "exact_caption",
        "cpr_weight": 0.0,
        "cpr_margin": None,
        "cpr_negative_policy": "none",
        "physical_batch_size": 8,
        "last_batch_size": 9,
        "video_representation": "mean_visual_prefix_tokens_l2_normalized",
        "text_representation": "frozen_gpt2_wte_masked_mean_l2_normalized",
        "frozen_gpt2": True,
        "frozen_vit_backbone": True,
        "cached_frozen_backbone": True,
    }


def expected_dataset_contract() -> dict:
    return {
        "annotations": {
            split: {
                "path": str(formal.DATA / "annotations" / f"{split}.json"),
                "sha256": sha256_file(formal.DATA / "annotations" / f"{split}.json"),
            }
            for split in ("train", "val")
        },
        "manifest": {
            "path": str(formal.DATA / "manifest.json"),
            "sha256": sha256_file(formal.DATA / "manifest.json"),
        },
    }


def preflight(config_path: Path, source: Path, steps: int) -> tuple[dict, dict, dict, Path]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    formal.validate_config(config, config_path)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    pause = json.loads((source / "pause_20260928.json").read_text(encoding="utf-8"))
    checkpoint = source / "seed123/A/checkpoint_epoch020.pt"
    checkpoint_hash = sha256_file(checkpoint)
    if checkpoint_hash != pause["checkpoint_sha256"]:
        raise RuntimeError("Paused checkpoint hash changed")
    if sha256_file(config_path) != pause["formal_config_sha256"]:
        raise RuntimeError("Frozen config hash changed")
    if sha256_file(formal.DATA / "data_gate.json") != config["data_gate_sha256"]:
        raise RuntimeError("Data gate hash changed")
    sources = json.loads((source / "source_hashes.json").read_text(encoding="utf-8"))
    changed = [name for name, digest in sources.items() if sha256_file(ROOT / name) != digest]
    if changed:
        raise RuntimeError(f"Original training source changed: {changed}")
    schedule_path = source / "seed123/schedule.json"
    schedule_hash = sha256_file(schedule_path)
    checkpoint_payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if checkpoint_payload.get("checkpoint_version") != 3:
        raise RuntimeError("Full version-3 checkpoint required")
    if checkpoint_payload["progress"] != {"step": 16280, "epoch": 20, "epoch_complete": True}:
        raise RuntimeError("Checkpoint progress mismatch")
    if checkpoint_payload["loader_state"] != {
        "schedule_sha256": schedule_hash,
        "next_epoch": 21,
        "config_sha256": sha256_file(config_path),
    }:
        raise RuntimeError("Checkpoint loader state mismatch")
    if checkpoint_payload["args"]["seed"] != 123 or checkpoint_payload["args"]["arm"] != "A":
        raise RuntimeError("Checkpoint seed/arm mismatch")
    formal.validate_training_resume_contract(
        checkpoint_payload, expected_training_contract("A"), expected_dataset_contract()
    )
    if checkpoint_payload["training_contract"] != expected_training_contract("A"):
        raise RuntimeError("Full training contract differs")
    if checkpoint_payload["dataset_contract"] != expected_dataset_contract():
        raise RuntimeError("Full dataset contract differs")
    if not 1 <= steps <= pause["post_checkpoint_steps_not_recoverable"]:
        raise RuntimeError("Replay steps must fit the preserved uninterrupted suffix")
    return config, checkpoint_payload, json.loads(schedule_path.read_text(encoding="utf-8")), checkpoint


def main(config_path: Path, source: Path, out: Path, steps: int) -> None:
    if out.exists():
        raise FileExistsError(out)
    config_path = config_path.resolve()
    source = source.resolve()
    config, checkpoint_payload, schedule, checkpoint = preflight(config_path, source, steps)
    out.mkdir(parents=True)
    write_new(out / "preflight.json", {
        "at_utc": datetime.now(timezone.utc).isoformat(),
        "source_run": str(source),
        "source_checkpoint": str(checkpoint),
        "source_checkpoint_sha256": sha256_file(checkpoint),
        "config_sha256": sha256_file(config_path),
        "schedule_sha256": sha256_file(source / "seed123/schedule.json"),
        "adapter_sha256": sha256_file(Path(__file__)),
        "requested_replay_steps": steps,
        "test_split_read": False,
    })

    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = True
    formal.set_seed(123)
    model = formal.VideoCaptionModel(
        gpt2_name=config["gpt2_source_path"], freeze_vit=True,
        vit_enable_torch_compile=False,
    ).cuda()
    model.load_state_dict(checkpoint_payload["model_state"], strict=True)
    params = formal.configure_trainable(model)
    optimizer = torch.optim.AdamW(
        [parameter for _, parameter in params], lr=3e-4, weight_decay=0.01
    )
    restored = formal.restore_training_state(checkpoint_payload, optimizer)
    if restored != {"step": 16280, "epoch": 20, "epoch_complete": True}:
        raise RuntimeError("Restored progress mismatch")
    if len(optimizer.state) != len(params) or any(
        state["exp_avg"].device.type != "cuda" for state in optimizer.state.values()
    ):
        raise RuntimeError("Optimizer moments were not restored to the GPU")
    initial_hashes = formal.component_hashes(model)
    train = json.loads((formal.DATA / "annotations/train.json").read_text(encoding="utf-8"))
    cache = torch.load(formal.DATA / "feature_cache/train_raw_vit.pt", map_location="cpu", weights_only=True)
    ids = [row["video_id"] for row in train]
    if len(ids) != 6513 or cache["video_ids"] != ids:
        raise RuntimeError("Train cache order changed")
    features = cache["raw_features"]
    pad_id = model.decoder.tokenizer.pad_token_id
    torch.cuda.reset_peak_memory_stats()
    old_trace_path = source / "seed123/A/trace.jsonl"
    old_metrics_path = source / "seed123/A/train_metrics.jsonl"
    max_loss_abs_error = 0.0
    matched = 0
    with old_trace_path.open(encoding="utf-8") as old_trace, \
         old_metrics_path.open(encoding="utf-8") as old_metrics, \
         (out / "replay_trace.jsonl").open("x", encoding="utf-8") as new_trace, \
         (out / "replay_metrics.jsonl").open("x", encoding="utf-8") as new_metrics:
        trace_rows = itertools.islice(old_trace, 16280, 16280 + steps)
        metric_rows = itertools.islice(old_metrics, 16280, 16280 + steps)
        for step0, (trace_line, metric_line) in enumerate(zip(trace_rows, metric_rows), 16280):
            epoch_index, batch_index = divmod(step0, config["steps_per_epoch"])
            epoch = epoch_index + 1
            indices = schedule["batches"][epoch_index][batch_index]
            texts = []
            target_ids = []
            for index in indices:
                row = train[index]
                order = schedule["caption_orders"][row["video_id"]]
                caption_index = order[epoch_index % len(order)]
                text = row["captions"][caption_index]
                texts.append(text)
                target_ids.append(formal.tokenize_caption(model.decoder.tokenizer, text, max_len=96))
            captions = torch.stack(target_ids)
            batch_features = features[indices]
            evidence = formal.trace_evidence(step0 + 1, epoch_index, indices, ids,
                                             batch_features, captions, texts)
            expected_evidence = json.loads(trace_line)
            formal.assert_trace_equal(evidence, expected_evidence)
            new_trace.write(json.dumps(evidence, ensure_ascii=False, sort_keys=True) + "\n")
            mask = formal.exact_caption_positive_mask(captions, pad_id)
            if not torch.all(mask.any(dim=1)) or not torch.all((~mask).any(dim=1)):
                raise RuntimeError(f"Legal InfoNCE positives/negatives failed at {step0 + 1}")
            model.train()
            model.encoder.backbone.eval()
            optimizer.zero_grad(set_to_none=True)
            ce = formal.cached_caption_loss(model, batch_features.cuda(), captions.cuda())
            if not torch.isfinite(ce):
                raise RuntimeError(f"Nonfinite loss at {step0 + 1}")
            ce.backward()
            if any(parameter.grad is None or not torch.isfinite(parameter.grad).all()
                   for _, parameter in params):
                raise RuntimeError(f"Nonfinite gradient at {step0 + 1}")
            optimizer.step()
            actual = {"loss": float(ce.detach()), "caption_loss": float(ce.detach()),
                      "alignment_loss": 0.0, "step": step0 + 1, "epoch": epoch}
            expected = json.loads(metric_line)
            max_loss_abs_error = max(max_loss_abs_error, abs(actual["loss"] - expected["loss"]))
            if actual != expected:
                raise RuntimeError(f"Metric replay mismatch at step {step0 + 1}: {actual} != {expected}")
            new_metrics.write(json.dumps(actual) + "\n")
            matched += 1
    if matched != steps:
        raise RuntimeError(f"Only {matched}/{steps} uninterrupted steps were available")
    final_hashes = formal.component_hashes(model)
    if any(initial_hashes[key] != final_hashes[key] for key in ("decoder", "backbone")):
        raise RuntimeError("Frozen parameters changed during replay")
    if any(initial_hashes[key] == final_hashes[key] for key in ("projection", "mapper")):
        raise RuntimeError("Trainable parameters did not update")
    write_new(out / "result.json", {
        "at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS",
        "seed": 123,
        "arm": "A",
        "source_checkpoint_epoch": 20,
        "source_checkpoint_step": 16280,
        "first_replayed_step": 16281,
        "last_replayed_step": 16280 + matched,
        "exact_trace_steps": matched,
        "exact_metric_steps": matched,
        "max_loss_abs_error": max_loss_abs_error,
        "optimizer_state_restored": True,
        "cpu_cuda_rng_restored": True,
        "frozen_weights_unchanged": True,
        "trainable_weights_updated": True,
        "peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
        "replay_trace_sha256": sha256_file(out / "replay_trace.jsonl"),
        "replay_metrics_sha256": sha256_file(out / "replay_metrics.jsonl"),
        "test_split_read": False,
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=64)
    args = parser.parse_args()
    try:
        main(args.config, args.source, args.output, args.steps)
    except Exception:
        if args.output.is_dir() and not (args.output / "failure.json").exists():
            write_new(args.output / "failure.json", {
                "at_utc": datetime.now(timezone.utc).isoformat(),
                "traceback": traceback.format_exc(),
            })
        raise
