#!/usr/bin/env python3
"""Run the preregistered full-MSVD balanced-caption viability experiment."""
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

from scripts.diagnose_prefix_sensitivity import donor_permutation, paired_bootstrap, reference_nll
from scripts.diagnose_semantic_gate import corpus_bleu
from scripts.run_caption_policy_ab import (
    cached_caption_loss,
    raw_backbone_features,
    reference_seed,
    token_f1,
    tokenize_caption,
    tokenize_references,
    write_json,
)
from scripts.run_overfit16 import (
    BASE_GPT2_SHA256,
    component_hashes,
    configure_trainable,
    generate_one,
    normalize_caption,
    state_sha,
    tensor_sha,
)
from src.cli.train_caption_mapper import set_seed
from src.data.data_loader import build_dataloader
from src.models.caption_model import VideoCaptionModel
from src.training.checkpoint import (
    build_dataset_contract,
    capture_rng_state,
    save_training_checkpoint,
    sha256_file,
)

TRAIN = ROOT / "data/processed/msvd_full/train/annotations.json"
VAL = ROOT / "data/processed/msvd_full/val/annotations.json"
DATA_SHA256 = {
    "train": "ced443e28c5ad2ddd0780af55a820b5ec7da6462c092e1329951dc9c7598c7c5",
    "val": "bcda3ba0583704a54297e4da4cc9acbcc5ec1ebffd6ed436a6ba3ee1b8a03c8b",
}
FORMAL_EPOCHS = 66
FORMAL_STEPS = 9900
MILESTONE_EPOCHS = (5, 20, 40, 66)
SOURCE_FILES = (
    "experiments/full_msvd_balanced_protocol_20260905.md",
    "scripts/run_full_msvd_balanced.py",
    "scripts/run_caption_policy_ab.py",
    "scripts/run_overfit16.py",
    "scripts/diagnose_prefix_sensitivity.py",
    "scripts/diagnose_semantic_gate.py",
    "src/data/data_loader.py",
    "src/models/caption_model.py",
    "src/models/text_decoder.py",
    "src/models/video_encoder.py",
    "src/training/checkpoint.py",
)


def make_reference_orders(records: list[dict]) -> dict[str, list[int]]:
    orders = {}
    for record in records:
        video_id = str(record["video_id"])
        generator = torch.Generator().manual_seed(reference_seed(video_id))
        orders[video_id] = torch.randperm(len(record["captions"]), generator=generator).tolist()
    return orders


def make_batch_schedule(count: int, epochs: int, batch_size: int = 8) -> list[list[list[int]]]:
    if count % batch_size:
        raise ValueError("This locked protocol requires full fixed-size batches")
    generator = torch.Generator().manual_seed(456)
    return [torch.randperm(count, generator=generator).reshape(-1, batch_size).tolist()
            for _ in range(epochs)]


def coverage_is_valid(counts: Counter, expected_references: int, smoke: bool) -> bool:
    required_unique = 1 if smoke else expected_references
    return bool(counts) and len(counts) >= required_unique and min(counts.values()) >= 1


@torch.inference_mode()
def cache_split(model: VideoCaptionModel, loader, expected_ids: list[str], limit: int,
                label: str, device: str) -> tuple[torch.Tensor, dict]:
    model.eval()
    features = []
    video_ids = []
    video_hashes = []
    errors = []
    for batch_number, batch in enumerate(loader, 1):
        remaining = limit - len(video_ids)
        if remaining <= 0:
            break
        if len(batch["video_id"]) > remaining:
            raise RuntimeError("Cache limit must align with full batches")
        video = batch["video"].to(device)
        direct = model.encoder(video)
        raw = raw_backbone_features(model.encoder, video)
        cached = model.encoder.proj(raw)
        errors.append(float((direct - cached).abs().max()))
        features.append(raw.cpu())
        video_ids.extend(batch["video_id"])
        video_hashes.append(tensor_sha(batch["video"]))
        if batch_number == 1 or batch_number % 25 == 0 or len(video_ids) == limit:
            print(f"[cache:{label}] {len(video_ids)}/{limit}", flush=True)
    values = torch.cat(features, 0)
    maximum = max(errors)
    if video_ids != expected_ids[:limit] or maximum > 1e-6 or not torch.isfinite(values).all():
        raise RuntimeError(f"Invalid {label} feature cache")
    audit = {"count": len(video_ids), "video_ids": video_ids,
             "raw_feature_sha256": tensor_sha(values), "video_batch_sha256": video_hashes,
             "max_abs_error": maximum, "equivalence_pass": True}
    return values, audit


@torch.inference_mode()
def evaluate_validation(model: VideoCaptionModel, raw_features: torch.Tensor,
                        records: list[dict], references: dict[str, list[dict]],
                        donors: torch.Tensor, epoch: int, step: int, device: str) -> dict:
    model.eval()
    features = raw_features.to(device)
    prefixes = model.decoder.build_visual_prefix(model.proj(model.encoder.proj(features)))
    tokenizer = model.decoder.tokenizer
    rows = []
    generated = []
    shuffled_generated = []
    for index, record in enumerate(records):
        video_id = str(record["video_id"])
        donor_index = int(donors[index])
        native_prefix = prefixes[index:index + 1]
        wrong_prefix = prefixes[donor_index:donor_index + 1]
        native = reference_nll(model.decoder.model, tokenizer, native_prefix,
                               references[video_id], microbatch=8)
        wrong = reference_nll(model.decoder.model, tokenizer, wrong_prefix,
                              references[video_id], microbatch=8)
        native_mean = sum(item["nll"] for item in native) / len(native)
        wrong_mean = sum(item["nll"] for item in wrong) / len(wrong)
        native_text, native_eos, native_ids = generate_one(model.decoder.model, tokenizer, native_prefix)
        wrong_text, wrong_eos, wrong_ids = generate_one(model.decoder.model, tokenizer, wrong_prefix)
        ref_texts = [item["text"] for item in references[video_id]]
        normalized_refs = {normalize_caption(text) for text in ref_texts}
        generated.append(native_text)
        shuffled_generated.append(wrong_text)
        rows.append({"video_id": video_id, "donor_video_id": str(records[donor_index]["video_id"]),
                     "reference_count": len(native), "native_nll": native_mean,
                     "shuffled_nll": wrong_mean, "delta": wrong_mean - native_mean,
                     "native_reference_nll": [item["nll"] for item in native],
                     "shuffled_reference_nll": [item["nll"] for item in wrong],
                     "caption": native_text, "shuffled_caption": wrong_text,
                     "generated_ids": native_ids, "shuffled_generated_ids": wrong_ids,
                     "eos_terminated": native_eos, "shuffled_eos_terminated": wrong_eos,
                     "exact_any_reference": normalize_caption(native_text) in normalized_refs,
                     "max_reference_token_f1": max(token_f1(native_text, text) for text in ref_texts)})
        if (index + 1) % 20 == 0 or index + 1 == len(records):
            print(f"[val:e{epoch:02d}] {index + 1}/{len(records)}", flush=True)
    normalized = [normalize_caption(text) for text in generated]
    normalized_wrong = [normalize_caption(text) for text in shuffled_generated]
    counts = Counter(normalized)
    deltas = [row["delta"] for row in rows]
    native_nll = sum(row["native_nll"] for row in rows) / len(rows)
    mean_delta = sum(deltas) / len(deltas)
    bootstrap = paired_bootstrap(deltas, seed=123, resamples=10000)
    samples = [{"hypothesis": row["caption"], "references": record["captions"]}
               for row, record in zip(rows, records)]
    metrics = {"native_nll": native_nll, "shuffled_minus_native_nll": mean_delta,
               "grounding_bootstrap": bootstrap,
               "positive_mismatch_videos": sum(value > 0 for value in deltas),
               "nonempty_rate": sum(bool(text) for text in normalized) / len(rows),
               "unique_count": len(counts), "unique_rate": len(counts) / len(rows),
               "largest_template_count": max(counts.values()),
               "largest_template_rate": max(counts.values()) / len(rows),
               "caption_change_count": sum(left != right for left, right in zip(normalized, normalized_wrong)),
               "caption_change_rate": sum(left != right for left, right in zip(normalized, normalized_wrong)) / len(rows),
               "eos_count": sum(row["eos_terminated"] for row in rows),
               "eos_rate": sum(row["eos_terminated"] for row in rows) / len(rows),
               "exact_any_reference_count": sum(row["exact_any_reference"] for row in rows),
               "mean_max_reference_token_f1": sum(row["max_reference_token_f1"] for row in rows) / len(rows),
               "triage_bleu4_unsmoothed": corpus_bleu(samples),
               "top_templates": [{"caption": text, "count": count} for text, count in counts.most_common(10)]}
    diversity_gate = (metrics["nonempty_rate"] == 1.0 and metrics["unique_rate"] >= 0.30
                      and metrics["largest_template_rate"] <= 0.25)
    viability = {"nonempty_100_percent": metrics["nonempty_rate"] == 1.0,
                 "unique_rate_ge_30_percent": metrics["unique_rate"] >= 0.30,
                 "largest_template_le_25_percent": metrics["largest_template_rate"] <= 0.25,
                 "grounding_ci_lower_gt_zero": bootstrap["ci95_low"] > 0,
                 "positive_mismatch_ge_80": metrics["positive_mismatch_videos"] >= 80}
    viability["pass"] = all(viability.values())
    metrics["semantic_diversity_gate_pass"] = diversity_gate
    metrics["primary_viability_gate"] = viability
    return {"epoch": epoch, "step": step, "metrics": metrics,
            "video_ids": [str(record["video_id"]) for record in records],
            "donor_indices": donors.tolist(), "rows": rows,
            "prefix_sha256": tensor_sha(prefixes),
            "reference_count": sum(len(value) for value in references.values()),
            "weighting": "reference mean within video, then equal video mean"}


def trace_evidence(step: int, epoch: int, batch_indices: list[int], video_ids: list[str],
                   features: torch.Tensor, captions: torch.Tensor, texts: list[str]) -> dict:
    rng = capture_rng_state()
    return {"step": step, "epoch": epoch, "indices": batch_indices,
            "video_ids": [video_ids[index] for index in batch_indices],
            "target_texts": texts, "caption_ids_sha256": tensor_sha(captions),
            "visual_features_sha256": tensor_sha(features),
            "python_rng_sha256": hashlib.sha256(repr(rng["python"]).encode()).hexdigest(),
            "torch_cpu_rng_sha256": tensor_sha(rng["torch_cpu"]),
            "torch_cuda_rng_sha256": [tensor_sha(value) for value in rng["torch_cuda"] or []]}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.output_root.exists():
        raise FileExistsError("Refusing to overwrite an existing experiment directory")
    if not torch.cuda.is_available():
        raise RuntimeError("This registered run requires the local CUDA GPU")
    args.output_root.mkdir(parents=True)
    for split, path in (("train", TRAIN), ("val", VAL)):
        if sha256_file(path) != DATA_SHA256[split]:
            raise RuntimeError(f"Pinned {split} annotations changed")
    source_hashes = {name: sha256_file(ROOT / name) for name in SOURCE_FILES}
    train_records = json.loads(TRAIN.read_text(encoding="utf-8"))
    val_records = json.loads(VAL.read_text(encoding="utf-8"))
    if len(train_records) != 1200 or len(val_records) != 100:
        raise RuntimeError("Unexpected full-MSVD split sizes")
    train_limit, val_limit = (16, 8) if args.smoke else (1200, 100)
    epochs = 1 if args.smoke else FORMAL_EPOCHS
    steps = 2 if args.smoke else FORMAL_STEPS
    train_used = train_records[:train_limit]
    val_used = val_records[:val_limit]
    base_model_file = Path(cached_file("gpt2", "model.safetensors", local_files_only=True))
    if sha256_file(base_model_file) != BASE_GPT2_SHA256:
        raise RuntimeError("Original GPT-2 source changed")
    source_path = str(base_model_file.parent)
    set_seed(123)
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model = VideoCaptionModel(gpt2_name=source_path, freeze_vit=True,
                              vit_enable_torch_compile=False)
    initial_hashes = component_hashes(model)
    common_path = args.output_root / "initial_model.pt"
    torch.save({"model_state": {key: value.detach().cpu().clone()
                                for key, value in model.state_dict().items()}}, common_path)
    model = model.to("cuda")
    train_loader = build_dataloader(str(TRAIN), model.decoder.tokenizer, batch_size=8,
                                    max_len=64, num_frame=8, image_size=224, shuffle=False,
                                    num_wokers=0, random_caption=False)
    val_loader = build_dataloader(str(VAL), model.decoder.tokenizer, batch_size=8,
                                  max_len=64, num_frame=8, image_size=224, shuffle=False,
                                  num_wokers=0, random_caption=False)
    train_ids = [str(record["video_id"]) for record in train_records]
    val_ids = [str(record["video_id"]) for record in val_records]
    train_features, train_cache = cache_split(model, train_loader, train_ids,
                                               train_limit, "train", "cuda")
    val_features, val_cache = cache_split(model, val_loader, val_ids,
                                           val_limit, "val", "cuda")
    torch.save({"features": train_features, "video_ids": train_cache["video_ids"],
                "audit": train_cache}, args.output_root / "train_features.pt")
    torch.save({"features": val_features, "video_ids": val_cache["video_ids"],
                "audit": val_cache}, args.output_root / "val_features.pt")
    train_references = {str(record["video_id"]): tokenize_references(
        model.decoder.tokenizer, record["captions"]) for record in train_used}
    val_references = {str(record["video_id"]): tokenize_references(
        model.decoder.tokenizer, record["captions"]) for record in val_used}
    if not args.smoke and (sum(len(value) for value in train_references.values()) != 48774
                           or sum(len(value) for value in val_references.values()) != 4290):
        raise RuntimeError("Unexpected full reference counts")
    reference_orders = make_reference_orders(train_used)
    batch_schedule = make_batch_schedule(train_limit, epochs)
    donors = donor_permutation(val_limit, seed=321)
    schedule_path = args.output_root / "schedule.json"
    write_json(schedule_path, {"batch_seed": 456, "donor_seed": 321,
        "batch_schedule": batch_schedule, "donor_indices": donors.tolist(),
        "reference_seeds": {video_id: reference_seed(video_id) for video_id in train_cache["video_ids"]},
        "reference_orders": reference_orders})
    preflight = {"protocol": "full-msvd-balanced-v1", "paper_ready": False,
        "smoke_test": args.smoke, "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": {"seed": 123, "batch_seed": 456, "donor_seed": 321,
                   "epochs": epochs, "steps": steps, "batch_size": 8,
                   "frames": 8, "image_size": 224, "max_len": 64,
                   "lr": 3e-4, "weight_decay": 0.01,
                   "milestone_epochs": [1] if args.smoke else list(MILESTONE_EPOCHS)},
        "split_counts": {"train": train_limit, "val": val_limit, "test_read": False},
        "reference_counts": {"train": sum(len(value) for value in train_references.values()),
                             "val": sum(len(value) for value in val_references.values())},
        "train_cache": train_cache, "val_cache": val_cache,
        "train_cache_file_sha256": sha256_file(args.output_root / "train_features.pt"),
        "val_cache_file_sha256": sha256_file(args.output_root / "val_features.pt"),
        "initial_component_hashes": initial_hashes,
        "initial_model_sha256": sha256_file(common_path), "schedule_sha256": sha256_file(schedule_path),
        "dataset_contract": build_dataset_contract({"train": str(TRAIN), "val": str(VAL)}),
        "source_sha256": source_hashes, "model_source": {"path": source_path,
                                                          "sha256": BASE_GPT2_SHA256},
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "runtime": {"python": sys.version, "torch": torch.__version__,
                    "gpu": torch.cuda.get_device_name(), "deterministic_algorithms": True, "tf32": False}}
    write_json(args.output_root / "preflight.json", preflight)
    configure_trainable(model)
    trainable = [(name, parameter) for name, parameter in model.named_parameters()
                 if parameter.requires_grad]
    optimizer = torch.optim.AdamW([parameter for _, parameter in trainable],
                                  lr=3e-4, weight_decay=0.01)
    token_cache = {text: tokenize_caption(model.decoder.tokenizer, text)
                   for record in train_used for text in record["captions"]}
    video_ids = train_cache["video_ids"]
    by_id = {str(record["video_id"]): record for record in train_used}
    seen = Counter()
    caption_seen = {video_id: Counter() for video_id in video_ids}
    trace_path = args.output_root / "trace.jsonl"
    metric_path = args.output_root / "train_metrics.jsonl"
    evaluations = []
    milestone_set = {1} if args.smoke else set(MILESTONE_EPOCHS)
    set_seed(123)
    step = 0
    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    with trace_path.open("x", encoding="utf-8") as trace, metric_path.open("x", encoding="utf-8") as metrics:
        for epoch_index, epoch_batches in enumerate(batch_schedule):
            epoch_number = epoch_index + 1
            epoch_losses = []
            for batch_indices in epoch_batches:
                step += 1
                texts = []
                for index in batch_indices:
                    video_id = video_ids[index]
                    order = reference_orders[video_id]
                    caption_index = order[epoch_index % len(order)]
                    text = by_id[video_id]["captions"][caption_index]
                    texts.append(text)
                    seen[video_id] += 1
                    caption_seen[video_id][caption_index] += 1
                batch_features = train_features[batch_indices]
                captions = torch.stack([token_cache[text] for text in texts])
                trace.write(json.dumps(trace_evidence(step, epoch_index, batch_indices, video_ids,
                    batch_features, captions, texts), ensure_ascii=False, sort_keys=True) + "\n")
                trace.flush()
                model.train()
                model.encoder.backbone.eval()
                optimizer.zero_grad(set_to_none=True)
                loss = cached_caption_loss(model, batch_features.to("cuda"), captions.to("cuda"))
                if not torch.isfinite(loss):
                    raise RuntimeError(f"Nonfinite loss at step {step}")
                loss.backward()
                if any(parameter.grad is None or not torch.isfinite(parameter.grad).all()
                       for _, parameter in trainable):
                    raise RuntimeError(f"Missing/nonfinite gradient at step {step}")
                optimizer.step()
                value = float(loss.detach())
                epoch_losses.append(value)
                metrics.write(json.dumps({"step": step, "epoch": epoch_number, "loss": value}) + "\n")
                metrics.flush()
                if step == 1 or step % 250 == 0 or step == steps:
                    print(f"[train] epoch={epoch_number}/{epochs} step={step}/{steps} loss={value:.6f} elapsed={time.monotonic()-started:.1f}s", flush=True)
            print(f"[epoch] {epoch_number}/{epochs} mean_loss={sum(epoch_losses)/len(epoch_losses):.6f}", flush=True)
            if epoch_number in milestone_set:
                evaluation = evaluate_validation(model, val_features, val_used, val_references,
                                                 donors, epoch_number, step, "cuda")
                evaluations.append(evaluation)
                compact = dict(evaluation["metrics"])
                compact.pop("top_templates")
                print(json.dumps({"epoch": epoch_number, **compact}, ensure_ascii=False), flush=True)
                checkpoint_path = args.output_root / f"checkpoint_epoch{epoch_number:03d}.pt"
                contract = {"objective_version": 3, "caption_objective": "causal_lm_cross_entropy",
                            "caption_policy": "balanced_all_references", "alignment_objective": "none",
                            "alignment_weight": 0.0, "physical_batch_size": 8,
                            "protocol": "full-msvd-balanced-v1", "frozen_gpt2": True,
                            "frozen_vit_backbone": True, "cached_frozen_backbone": True,
                            "legacy_visual_weights_used": False}
                save_training_checkpoint(checkpoint_path, model=model, optimizer=optimizer, scheduler=None,
                    progress={"step": step, "epoch": epoch_number, "epoch_complete": True,
                              "best_val": evaluation["metrics"]["native_nll"]},
                    args={"gpt2_name": source_path, "vit_name": "vit_base_patch16_224",
                          "cond_mode": "prefix", "prefix_len": 4, "freeze_vit": True,
                          "seed": 123, "batch_size": 8, "max_len": 64, "lr": 3e-4,
                          "epochs": epochs, "caption_policy": "balanced_all_references"},
                    training_contract=contract, dataset_contract=preflight["dataset_contract"],
                    loader_state={"batch_schedule_sha256": state_sha({"schedule": torch.tensor(batch_schedule)}),
                                  "num_workers": 0})
                evaluation["checkpoint"] = str(checkpoint_path.resolve())
                evaluation["checkpoint_sha256"] = sha256_file(checkpoint_path)
    if step != steps or set(seen) != set(video_ids) or any(count != epochs for count in seen.values()):
        raise RuntimeError("Training coverage does not match the locked budget")
    coverage = {}
    for video_id in video_ids:
        expected = len(by_id[video_id]["captions"])
        counts = caption_seen[video_id]
        if not coverage_is_valid(counts, expected, args.smoke):
            raise RuntimeError(f"Not every reference was visited for {video_id}")
        coverage[video_id] = {"reference_count": expected, "unique_visited": len(counts),
                              "full_reference_coverage_required": not args.smoke,
                              "minimum_visits": min(counts.values()),
                              "maximum_visits": max(counts.values())}
    final_hashes = component_hashes(model)
    if final_hashes["decoder"] != initial_hashes["decoder"] or final_hashes["backbone"] != initial_hashes["backbone"]:
        raise RuntimeError("Frozen model component changed")
    if final_hashes["projection"] == initial_hashes["projection"] or final_hashes["mapper"] == initial_hashes["mapper"]:
        raise RuntimeError("A trainable component did not change")
    model.eval()
    first_batch = next(iter(train_loader))
    with torch.inference_mode():
        direct = model.encoder(first_batch["video"].to("cuda"))
        cached = model.encoder.proj(train_features[:8].to("cuda"))
        post_cache_error = float((direct - cached).abs().max())
    if post_cache_error > 1e-6:
        raise RuntimeError("Post-training cache equivalence failed")
    if any(sha256_file(path) != DATA_SHA256[split]
           for split, path in (("train", TRAIN), ("val", VAL))):
        raise RuntimeError("Dataset annotations changed during the run")
    if source_hashes != {name: sha256_file(ROOT / name) for name in SOURCE_FILES}:
        raise RuntimeError("Experiment source changed during the run")
    primary = evaluations[-1]["metrics"]["primary_viability_gate"]
    decision = "smoke_only" if args.smoke else (
        "proceed_to_infonce_ablation" if primary["pass"] else "stop_before_infonce_and_diagnose")
    result = {"protocol": "full-msvd-balanced-v1", "paper_ready": False,
              "smoke_test": args.smoke, "decision": decision,
              "config": preflight["config"], "evaluations": evaluations,
              "initial_component_hashes": initial_hashes, "final_component_hashes": final_hashes,
              "frozen_weights_unchanged": True, "trainable_modules_changed": True,
              "train_video_visits": dict(seen), "caption_coverage": coverage,
              "post_train_cache_equivalence_max_abs_error": post_cache_error,
              "trace_sha256": sha256_file(trace_path),
              "final_checkpoint": evaluations[-1]["checkpoint"],
              "final_checkpoint_sha256": evaluations[-1]["checkpoint_sha256"],
              "elapsed_seconds": time.monotonic() - started,
              "peak_allocated_mb": torch.cuda.max_memory_allocated() / 2**20,
              "peak_reserved_mb": torch.cuda.max_memory_reserved() / 2**20,
              "source_and_data_unchanged": True,
              "limits": ["One seed and frozen GPT-2/ViT backbone.",
                         "Validation-only viability result; test670 was not read.",
                         "Triage BLEU and token-F1 are not paper benchmark metrics.",
                         "Passing permits an InfoNCE ablation but does not establish paper quality."]}
    write_json(args.output_root / "result.json", result)
    print(json.dumps({"output": str(args.output_root.resolve()), "decision": decision,
                      "final": evaluations[-1]["metrics"],
                      "checkpoint_sha256": result["final_checkpoint_sha256"]},
                     ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
