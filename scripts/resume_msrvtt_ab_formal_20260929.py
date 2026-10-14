"""Continue the frozen six-arm MSR-VTT run from seed123/A epoch 20.

This adapter creates a new run identity. The paused source run is read-only.
It reuses the frozen runner for all fresh arms and replaces only the first
arm's setup/loop with a strict full-checkpoint continuation.
"""
from __future__ import annotations

import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import argparse
from collections import Counter
from datetime import datetime, timezone
import itertools
import json
from pathlib import Path
import shutil
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from scripts import run_msrvtt_ab_formal_20260928 as formal
from scripts.verify_msrvtt_checkpoint_resume_20260929 import (
    expected_dataset_contract, expected_training_contract, preflight,
)
from src.training.checkpoint import sha256_file

ORIGINAL_RUN_ARM = formal.run_arm
SOURCE: Path
VERIFICATION: Path


def copy_exact(source: Path, destination: Path) -> None:
    if destination.exists():
        raise FileExistsError(destination)
    shutil.copy2(source, destination)
    if sha256_file(source) != sha256_file(destination):
        raise RuntimeError(f"Copy hash mismatch: {source}")


def copy_prefix(source: Path, destination: Path, count: int) -> None:
    with source.open(encoding="utf-8") as old, destination.open("x", encoding="utf-8") as new:
        copied = 0
        for line in itertools.islice(old, count):
            copied += 1
            if json.loads(line)["step"] != copied:
                raise RuntimeError(f"Noncontiguous source trace/metrics at {copied}: {source}")
            new.write(line)
    if copied != count:
        raise RuntimeError(f"Only {copied}/{count} source rows available: {source}")


def resume_first_arm(model, arm: str, seed: int, folder: Path, initial: dict,
                     initial_hashes: dict, schedule: dict, config: dict, config_path: Path,
                     records: list[dict], train_features: torch.Tensor,
                     tokenized: list[list[torch.Tensor]], val_records: list[dict],
                     val_features: torch.Tensor, references: dict,
                     text_embeddings: torch.Tensor, gold: torch.Tensor,
                     source_hashes: dict) -> dict:
    if seed != 123 or arm != "A":
        return ORIGINAL_RUN_ARM(model, arm, seed, folder, initial, initial_hashes,
                                schedule, config, config_path, records, train_features,
                                tokenized, val_records, val_features, references,
                                text_embeddings, gold, source_hashes)
    if folder.exists():
        raise FileExistsError(folder)
    if sha256_file(folder.parent / "schedule.json") != sha256_file(SOURCE / "seed123/schedule.json"):
        raise RuntimeError("Regenerated paired schedule differs from source")
    if source_hashes != json.loads((SOURCE / "source_hashes.json").read_text(encoding="utf-8")):
        raise RuntimeError("Frozen training sources differ from source run")
    saved_initial = torch.load(SOURCE / "seed123/initial_model.pt", map_location="cpu", weights_only=True)["model_state"]
    if saved_initial.keys() != initial.keys() or any(
        not torch.equal(saved_initial[key], initial[key]) for key in initial
    ):
        raise RuntimeError("Regenerated paired initial model differs from source")
    folder.mkdir()
    checkpoint = SOURCE / "seed123/A/checkpoint_epoch020.pt"
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    training_contract = expected_training_contract("A")
    dataset_contract = expected_dataset_contract()
    formal.validate_training_resume_contract(payload, training_contract, dataset_contract)
    model.load_state_dict(payload["model_state"], strict=True)
    params = formal.configure_trainable(model)
    optimizer = torch.optim.AdamW([parameter for _, parameter in params],
                                  lr=3e-4, weight_decay=0.01)
    restored = formal.restore_training_state(payload, optimizer)
    if restored != {"step": 16280, "epoch": 20, "epoch_complete": True}:
        raise RuntimeError("Restored progress mismatch")
    if len(optimizer.state) != len(params) or any(
        state["exp_avg"].device.type != "cuda" for state in optimizer.state.values()
    ):
        raise RuntimeError("Optimizer state was not restored to GPU")
    formal.write_new(folder.parent.parent / "resume_manifest.json", {
        "at_utc": datetime.now(timezone.utc).isoformat(),
        "source_run": str(SOURCE),
        "source_checkpoint": str(checkpoint),
        "source_checkpoint_sha256": sha256_file(checkpoint),
        "source_schedule_sha256": sha256_file(SOURCE / "seed123/schedule.json"),
        "formal_config_sha256": sha256_file(config_path),
        "strict_replay_verification": str(VERIFICATION / "result.json"),
        "strict_replay_verification_sha256": sha256_file(VERIFICATION / "result.json"),
        "resume_adapter_sha256": sha256_file(Path(__file__)),
        "first_reused_step": 16280,
        "first_new_step": 16281,
        "source_run_preserved": True,
        "test_split_read": False,
    })

    evaluations = []
    for epoch in (5, 10, 20):
        name = f"checkpoint_epoch{epoch:03d}.pt"
        copy_exact(SOURCE / "seed123/A" / name, folder / name)
        evaluation = json.loads((SOURCE / "seed123/A" / f"evaluation_epoch{epoch:03d}.json").read_text(encoding="utf-8"))
        if evaluation["checkpoint_sha256"] != sha256_file(folder / name):
            raise RuntimeError(f"Copied checkpoint/evaluation mismatch at epoch {epoch}")
        evaluation["checkpoint"] = str(folder / name)
        formal.write_new(folder / f"evaluation_epoch{epoch:03d}.json", evaluation)
        evaluations.append(evaluation)
    copy_prefix(SOURCE / "seed123/A/trace.jsonl", folder / "trace.jsonl", 16280)
    copy_prefix(SOURCE / "seed123/A/train_metrics.jsonl", folder / "train_metrics.jsonl", 16280)

    ids = [row["video_id"] for row in records]
    coverage = {
        vid: {schedule["caption_orders"][vid][e % len(schedule["caption_orders"][vid])]
              for e in range(20)}
        for vid in ids
    }
    donor = formal.donor_permutation(len(val_records), seed=321)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    step = 16280
    step_times = []
    exact_source_steps = 0
    adapter_hash = sha256_file(Path(__file__))
    source_trace = (SOURCE / "seed123/A/trace.jsonl").open(encoding="utf-8")
    source_metrics = (SOURCE / "seed123/A/train_metrics.jsonl").open(encoding="utf-8")
    try:
        for _ in range(step):
            if not source_trace.readline() or not source_metrics.readline():
                raise RuntimeError("Source trace/metrics ended before checkpoint")
        with (folder / "trace.jsonl").open("a", encoding="utf-8") as trace_stream, \
             (folder / "train_metrics.jsonl").open("a", encoding="utf-8") as metric_stream:
            for epoch_index in range(20, config["epochs"]):
                epoch = epoch_index + 1
                ce_epoch = []
                visits = Counter()
                for indices in schedule["batches"][epoch_index]:
                    tick = time.perf_counter()
                    step += 1
                    texts = []
                    target_ids = []
                    for index in indices:
                        row = records[index]
                        vid = row["video_id"]
                        order = schedule["caption_orders"][vid]
                        caption_index = order[epoch_index % len(order)]
                        texts.append(row["captions"][caption_index])
                        target_ids.append(tokenized[index][caption_index])
                        visits[vid] += 1
                        coverage[vid].add(caption_index)
                    captions = torch.stack(target_ids)
                    features = train_features[indices]
                    evidence = formal.trace_evidence(step, epoch_index, indices, ids,
                                                     features, captions, texts)
                    old_trace_line = source_trace.readline()
                    if old_trace_line:
                        formal.assert_trace_equal(evidence, json.loads(old_trace_line))
                    trace_stream.write(json.dumps(evidence, ensure_ascii=False, sort_keys=True) + "\n")
                    trace_stream.flush()
                    mask = formal.exact_caption_positive_mask(captions, model.decoder.tokenizer.pad_token_id)
                    if not torch.all(mask.any(dim=1)) or not torch.all((~mask).any(dim=1)):
                        raise RuntimeError(f"InfoNCE legal-negative failure at step {step}")
                    model.train()
                    model.encoder.backbone.eval()
                    optimizer.zero_grad(set_to_none=True)
                    ce = formal.cached_caption_loss(model, features.cuda(), captions.cuda())
                    output = {"loss": ce, "caption_loss": ce,
                              "alignment_loss": ce.new_zeros(())}
                    if any(not torch.isfinite(output[key]) for key in output):
                        raise RuntimeError(f"Nonfinite objective at step {step}")
                    output["loss"].backward()
                    if any(parameter.grad is None or not torch.isfinite(parameter.grad).all()
                           for _, parameter in params):
                        raise RuntimeError(f"Missing/nonfinite gradient at step {step}")
                    optimizer.step()
                    numeric = {key: float(output[key].detach()) for key in
                               ("loss", "caption_loss", "alignment_loss")}
                    numeric.update({"step": step, "epoch": epoch})
                    old_metric_line = source_metrics.readline()
                    if old_metric_line:
                        if numeric != json.loads(old_metric_line):
                            raise RuntimeError(f"Metric replay mismatch at step {step}")
                        exact_source_steps += 1
                    metric_stream.write(json.dumps(numeric) + "\n")
                    metric_stream.flush()
                    ce_epoch.append(numeric["caption_loss"])
                    step_times.append(time.perf_counter() - tick)
                if len(visits) != 6513 or set(visits.values()) != {1}:
                    raise RuntimeError("Resumed epoch did not visit every train ID exactly once")
                print(json.dumps({"seed": seed, "arm": arm, "epoch": epoch,
                                  "step": step, "mean_train_ce": sum(ce_epoch) / len(ce_epoch),
                                  "peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
                                  "resumed": True}), flush=True)
                if epoch in config["milestone_epochs"]:
                    new_checkpoint = folder / f"checkpoint_epoch{epoch:03d}.pt"
                    formal.save_training_checkpoint(
                        new_checkpoint, model=model, optimizer=optimizer, scheduler=None,
                        progress={"step": step, "epoch": epoch, "epoch_complete": True},
                        args={"gpt2_name": config["gpt2_source_path"], "seed": seed,
                              "arm": arm, "max_len": 96, "epochs": config["epochs"],
                              "lr": 3e-4}, training_contract=training_contract,
                        dataset_contract=dataset_contract,
                        loader_state={"schedule_sha256": sha256_file(folder.parent / "schedule.json"),
                                      "next_epoch": epoch + 1,
                                      "config_sha256": sha256_file(config_path)},
                    )
                    caption = formal.evaluate_validation(model, val_features, val_records,
                                                         references, donor, epoch, step, "cuda")
                    retrieval = formal.retrieval_diagnostic(model, val_features,
                                                            text_embeddings, gold, "cuda")
                    gate = formal.caption_gate(caption["metrics"], len(val_records), config)
                    entry = {"epoch": epoch, "step": step, "caption": caption,
                             "retrieval": retrieval, "caption_gate": gate,
                             "checkpoint": str(new_checkpoint),
                             "checkpoint_sha256": sha256_file(new_checkpoint)}
                    formal.write_new(folder / f"evaluation_epoch{epoch:03d}.json", entry)
                    evaluations.append(entry)
                    print(json.dumps({"seed": seed, "arm": arm, "milestone": epoch,
                                      "val_native_nll": caption["metrics"]["native_nll"],
                                      "caption_gate_pass": gate["pass"],
                                      "val_t2v_r1": retrieval["true_labels"]["text_to_video"]["r_at_1"]}),
                          flush=True)
    finally:
        source_trace.close()
        source_metrics.close()
    if exact_source_steps != 4131:
        raise RuntimeError(f"Expected 4,131 exact post-checkpoint source steps, got {exact_source_steps}")
    if step != config["steps_per_arm"]:
        raise RuntimeError("Fixed endpoint step count mismatch")
    if any(len(coverage[row["video_id"]]) != len(row["captions"]) for row in records):
        raise RuntimeError("Some train references were never selected")
    final_hashes = formal.component_hashes(model)
    if any(final_hashes[key] != initial_hashes[key] for key in ("decoder", "backbone")):
        raise RuntimeError("Frozen model changed")
    if any(final_hashes[key] == initial_hashes[key] for key in ("projection", "mapper")):
        raise RuntimeError("Trainable module unchanged")
    if source_hashes != {name: sha256_file(ROOT / name) for name in source_hashes}:
        raise RuntimeError("Original formal source changed during resumed arm")
    if adapter_hash != sha256_file(Path(__file__)):
        raise RuntimeError("Resume adapter changed during arm")
    final_checkpoint = Path(evaluations[-1]["checkpoint"])
    final_payload = torch.load(final_checkpoint, map_location="cpu", weights_only=True)
    formal.validate_training_resume_contract(final_payload, training_contract, dataset_contract)
    model.load_state_dict(final_payload["model_state"], strict=True)
    progress = formal.restore_training_state(final_payload, optimizer)
    if (progress["step"] != step or progress["epoch"] != config["epochs"]
            or final_payload["loader_state"]["schedule_sha256"] != sha256_file(folder.parent / "schedule.json")
            or formal.component_hashes(model) != final_hashes):
        raise RuntimeError("Resumed endpoint checkpoint strict reload failed")
    result = {"seed": seed, "arm": arm, "steps": step, "epochs": config["epochs"],
              "initial_component_hashes": initial_hashes,
              "final_component_hashes": final_hashes,
              "trace_sha256": sha256_file(folder / "trace.jsonl"),
              "train_metrics_sha256": sha256_file(folder / "train_metrics.jsonl"),
              "evaluations": [{"epoch": entry["epoch"], "step": entry["step"],
                               "path": str(folder / f"evaluation_epoch{entry['epoch']:03d}.json"),
                               "checkpoint_sha256": entry["checkpoint_sha256"]}
                              for entry in evaluations],
              "all_train_references_covered": True,
              "frozen_weights_unchanged": True,
              "final_checkpoint_strict_reload": True,
              "mean_step_seconds": sum(step_times) / len(step_times),
              "elapsed_seconds": time.perf_counter() - started,
              "peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
              "resumed_from_epoch": 20,
              "resumed_from_step": 16280,
              "exact_source_suffix_steps": exact_source_steps,
              "test_split_read": False,
              "technical_run_complete": True,
              "paper_claim_supported": None}
    formal.write_new(folder / "result.json", result)
    del optimizer
    torch.cuda.empty_cache()
    return result


def main(config_path: Path, source: Path, verification: Path, out: Path) -> None:
    if out.exists():
        raise FileExistsError(out)
    source = source.resolve()
    verification = verification.resolve()
    config_path = config_path.resolve()
    verification_result = json.loads((verification / "result.json").read_text(encoding="utf-8"))
    if (verification_result.get("status") != "PASS"
            or verification_result.get("exact_trace_steps", 0) < 64
            or verification_result.get("exact_metric_steps", 0) < 64
            or verification_result.get("max_loss_abs_error") != 0.0
            or verification_result.get("test_split_read") is not False):
        raise RuntimeError("Strict checkpoint replay verification missing")
    preflight(config_path, source, 64)
    global SOURCE, VERIFICATION
    SOURCE = source
    VERIFICATION = verification
    formal.run_arm = resume_first_arm
    formal.main(config_path, out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--verification", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        main(args.config, args.source, args.verification, args.output)
    except Exception:
        if args.output.is_dir() and not (args.output / "failure.json").exists():
            formal.write_new(args.output / "failure.json", {
                "at_utc": datetime.now(timezone.utc).isoformat(),
                "traceback": traceback.format_exc(),
            })
        raise
