"""Resume the paused seed124-C registered run from its epoch-40 checkpoint.

The original registered runner and paused output are read-only.  A short replay
compares both RNG trace and objective values with the uninterrupted steps that
were recorded before shutdown.  A full run writes a new, complete arm directory.
"""
from __future__ import annotations

import os

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
import time
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from scripts import run_cpr_cd_matrix as registered
from scripts.diagnose_prefix_sensitivity import donor_permutation
from scripts.eval_retrieval import encode_texts
from scripts.run_caption_policy_ab import tokenize_caption, tokenize_references, write_json
from scripts.run_full_msvd_balanced import DATA_SHA256, TRAIN, VAL, evaluate_validation, trace_evidence
from scripts.run_full_msvd_infonce_ablation import (
    assert_trace_equal,
    load_strict,
    retrieval_diagnostic,
    verify_registered_baseline,
)
from scripts.run_overfit16 import component_hashes, configure_trainable
from src.cli.train_caption_mapper import set_seed
from src.models.caption_model import VideoCaptionModel
from src.training.checkpoint import (
    load_checkpoint,
    restore_training_state,
    save_training_checkpoint,
    sha256_file,
    validate_training_resume_contract,
)


SEED = 124
ARM = "C"
RESUME_EPOCH = 40
RESUME_STEP = 6000
PAUSED_ROOT = ROOT / "runs/q3_cpr_cd_formal_20260912"
PAUSED_ARM = PAUSED_ROOT / "seed124/C"
CHECKPOINT = PAUSED_ARM / "checkpoint_epoch040.pt"
PAUSED_SHA256 = {
    "checkpoint_epoch040.pt": "8217b670c4f791023e6b7c3680f4ba11b27f0b0efa27f7c887912eada9681ce4",
    "train_metrics.jsonl": "112cf971c93dc3baeaf99541dc1621cee28798c7e9764591159fb6a732dbe14c",
    "trace.jsonl": "caa5033b2f8e4f69946f5a9fb05ebff09aed22593e8ca8e298fea951290f7b6a",
}
SEED123_RESULTS = {
    "C": "340612740c2b29134eafd00917e46aa9e8ac3d2591e1a2eb13e3024daadec366",
    "D": "df7538d8528085f993bbb20863938f9f36822ec27f6e7b6940b205caa80af6d7",
}


def checked_lines(path: Path, count: int, *, start_step: int = 1) -> list[str]:
    """Read a validated prefix, without altering the source artifact."""
    lines: list[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for expected_step in range(start_step, start_step + count):
            line = handle.readline()
            if not line:
                raise RuntimeError(f"{path} ended before step {expected_step}")
            actual_step = json.loads(line)["step"]
            if actual_step != expected_step:
                raise RuntimeError(f"{path} step {actual_step} != {expected_step}")
            lines.append(line)
    return lines


def verify_frozen_inputs() -> tuple[dict[str, Any], dict[str, str]]:
    registered.verify_registered_inputs()
    _, baseline_preflight = verify_registered_baseline()
    for filename, expected in PAUSED_SHA256.items():
        path = PAUSED_ARM / filename
        if sha256_file(path) != expected:
            raise RuntimeError(f"paused artifact changed: {path}")
    for arm, expected in SEED123_RESULTS.items():
        path = PAUSED_ROOT / f"seed123/{arm}/result.json"
        if sha256_file(path) != expected:
            raise RuntimeError(f"completed seed123 result changed: {path}")
    preflight = json.loads((PAUSED_ROOT / "seed124/preflight.json").read_text(encoding="utf-8"))
    source_hashes = {name: sha256_file(ROOT / name) for name in registered.SOURCE_FILES}
    if source_hashes != preflight["source_sha256"]:
        raise RuntimeError("registered source files changed since the paused run")
    if preflight["test_split_read"] or preflight["smoke_test"]:
        raise RuntimeError("paused preflight is not the registered formal run")
    if preflight["schedule_sha256"] != registered.REGISTERED_INPUTS[SEED]["schedule"][1]:
        raise RuntimeError("paused schedule hash changed")
    if preflight["initial_checkpoint_sha256"] != registered.REGISTERED_INPUTS[SEED]["initial"][1]:
        raise RuntimeError("paused initial checkpoint hash changed")
    checked_lines(PAUSED_ARM / "train_metrics.jsonl", RESUME_STEP)
    checked_lines(PAUSED_ARM / "trace.jsonl", RESUME_STEP)
    for epoch in (5, 20, 40):
        item = json.loads((PAUSED_ARM / f"evaluation_epoch{epoch:03d}.json").read_text(encoding="utf-8"))
        if item["epoch"] != epoch or item["step"] != epoch * 150:
            raise RuntimeError(f"paused evaluation at epoch {epoch} has invalid progress")
        if sha256_file(Path(item["checkpoint"])) != item["checkpoint_sha256"]:
            raise RuntimeError(f"paused evaluation checkpoint changed at epoch {epoch}")
    return baseline_preflight, source_hashes


def validate_checkpoint(checkpoint: dict[str, Any], baseline_preflight: dict[str, Any]) -> None:
    progress = checkpoint.get("progress", {})
    if progress.get("step") != RESUME_STEP or progress.get("epoch") != RESUME_EPOCH or not progress.get("epoch_complete"):
        raise RuntimeError(f"unexpected checkpoint progress: {progress}")
    args = checkpoint.get("args", {})
    expected_args = {"seed": SEED, "arm": ARM, "batch_size": 8, "max_len": 64,
                     "lr": 3e-4, "epochs": registered.FORMAL_EPOCHS}
    if any(args.get(key) != value for key, value in expected_args.items()):
        raise RuntimeError("checkpoint training arguments differ from the registered run")
    loader = checkpoint.get("loader_state", {})
    if loader.get("registered_schedule_sha256") != registered.REGISTERED_INPUTS[SEED]["schedule"][1]:
        raise RuntimeError("checkpoint schedule hash differs from registered schedule")
    if Path(loader.get("registered_schedule_path", "")).resolve() != registered.REGISTERED_INPUTS[SEED]["schedule"][0].resolve():
        raise RuntimeError("checkpoint schedule path differs from registered schedule")
    if loader.get("num_workers") != 0:
        raise RuntimeError("checkpoint loader worker count changed")
    if checkpoint.get("training_contract") != registered.training_contract(ARM):
        raise RuntimeError("full training contract differs from the registered run")
    validate_training_resume_contract(
        checkpoint, registered.training_contract(ARM), baseline_preflight["dataset_contract"]
    )


def prepare_context() -> dict[str, Any]:
    baseline_preflight, source_hashes = verify_frozen_inputs()
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    records = json.loads(TRAIN.read_text(encoding="utf-8"))
    validation = json.loads(VAL.read_text(encoding="utf-8"))
    if len(records) != 1200 or len(validation) != 100:
        raise RuntimeError("unexpected MSVD train/validation split sizes")
    train_cache = torch.load(registered.CACHE_INPUTS["train"][0], map_location="cpu", weights_only=True)
    val_cache = torch.load(registered.CACHE_INPUTS["val"][0], map_location="cpu", weights_only=True)
    if train_cache["video_ids"] != [item["video_id"] for item in records]:
        raise RuntimeError("training cache order changed")
    if val_cache["video_ids"] != [item["video_id"] for item in validation]:
        raise RuntimeError("validation cache order changed")

    schedule = registered.load_registered_schedule(SEED)
    initial_path = registered.REGISTERED_INPUTS[SEED]["initial"][0]
    set_seed(SEED)
    model = VideoCaptionModel(
        gpt2_name=baseline_preflight["model_source"]["path"],
        freeze_vit=True,
        vit_enable_torch_compile=False,
    )
    load_strict(model, initial_path)
    initial_hashes = component_hashes(model)
    if initial_hashes != json.loads((PAUSED_ROOT / "seed124/preflight.json").read_text(encoding="utf-8"))["initial_component_hashes"]:
        raise RuntimeError("initial component hashes changed")
    model = model.cuda().eval()

    token_cache = {
        text: tokenize_caption(model.decoder.tokenizer, text)
        for record in records for text in record["captions"]
    }
    references = {
        item["video_id"]: tokenize_references(model.decoder.tokenizer, item["captions"])
        for item in validation
    }
    all_val_captions = [text for item in validation for text in item["captions"]]
    text_embeddings = encode_texts(model, all_val_captions, max_length=64, batch_size=128, device="cuda")
    caption_gold = torch.tensor([
        index for index, item in enumerate(validation) for _ in item["captions"]
    ])
    return {
        "baseline_preflight": baseline_preflight,
        "source_hashes": source_hashes,
        "records": records,
        "validation": validation,
        "schedule": schedule,
        "model": model,
        "initial_hashes": initial_hashes,
        "train_features": train_cache["features"],
        "val_features": val_cache["features"],
        "token_cache": token_cache,
        "references": references,
        "text_embeddings": text_embeddings,
        "caption_gold": caption_gold,
    }


def replay(context: dict[str, Any], output_root: Path, verify_steps: int | None) -> dict[str, Any]:
    model = context["model"]
    trainable = configure_trainable(model)
    optimizer = torch.optim.AdamW(
        [parameter for _, parameter in trainable], lr=3e-4, weight_decay=0.01
    )
    checkpoint = load_checkpoint(CHECKPOINT, map_location="cpu")
    validate_checkpoint(checkpoint, context["baseline_preflight"])
    model.load_state_dict(checkpoint["model_state"], strict=True)
    if any(component_hashes(model)[key] != context["initial_hashes"][key]
           for key in ("decoder", "backbone")):
        raise RuntimeError("frozen weights differ from initial state")
    donors = donor_permutation(len(context["validation"]), seed=321)
    progress = restore_training_state(checkpoint, optimizer)
    if progress["step"] != RESUME_STEP:
        raise RuntimeError("optimizer/RNG restore returned unexpected progress")
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()

    arm_dir = output_root / "seed124/C"
    arm_dir.mkdir(parents=True)
    trace_path = arm_dir / "trace.jsonl"
    metrics_path = arm_dir / "train_metrics.jsonl"
    ids = [str(item["video_id"]) for item in context["records"]]
    original_trace = PAUSED_ARM / "trace.jsonl"
    original_metrics = PAUSED_ARM / "train_metrics.jsonl"
    trace_prefix = checked_lines(original_trace, RESUME_STEP)
    metrics_prefix = checked_lines(original_metrics, RESUME_STEP)
    minimum_eligible_fraction = min(json.loads(line)["cpr_eligible_fraction"] for line in metrics_prefix)
    comparisons = 0
    step = RESUME_STEP
    target_steps = registered.FORMAL_STEPS if verify_steps is None else RESUME_STEP + verify_steps
    if target_steps > registered.FORMAL_STEPS:
        raise ValueError("verification exceeds registered training schedule")
    with trace_path.open("x", encoding="utf-8") as trace, metrics_path.open("x", encoding="utf-8") as log:
        if verify_steps is None:
            trace.writelines(trace_prefix)
            log.writelines(metrics_prefix)
            trace.flush()
            log.flush()
        with registered.REGISTERED_INPUTS[SEED]["trace"][0].open("r", encoding="utf-8") as historical:
            for _ in range(RESUME_STEP):
                historical.readline()
            with original_trace.open("r", encoding="utf-8") as old_trace, original_metrics.open("r", encoding="utf-8") as old_metrics:
                for _ in range(RESUME_STEP):
                    old_trace.readline()
                    old_metrics.readline()
                for epoch in range(RESUME_EPOCH + 1, registered.FORMAL_EPOCHS + 1):
                    epoch_losses = []
                    for indices in context["schedule"]["batches"][epoch - 1]:
                        step += 1
                        targets = []
                        selected_ids = []
                        for index in indices:
                            video_id = ids[index]
                            order = context["schedule"]["orders"][video_id]
                            caption_index = order[(epoch - 1) % len(order)]
                            targets.append(context["records"][index]["captions"][caption_index])
                            selected_ids.append(video_id)
                        batch_features = context["train_features"][indices]
                        captions = torch.stack([context["token_cache"][text] for text in targets])
                        evidence = trace_evidence(
                            step, epoch - 1, indices, ids, batch_features, captions, targets
                        )
                        historical_line = historical.readline()
                        if not historical_line:
                            raise RuntimeError("historical trace ended before registered schedule")
                        assert_trace_equal(evidence, json.loads(historical_line))
                        old_trace_line = old_trace.readline()
                        if old_trace_line:
                            assert_trace_equal(evidence, json.loads(old_trace_line))
                            comparisons += 1
                        trace.write(json.dumps(evidence, ensure_ascii=False, sort_keys=True) + "\n")
                        trace.flush()

                        model.train()
                        model.encoder.backbone.eval()
                        optimizer.zero_grad(set_to_none=True)
                        output = registered.cached_cd_objective(
                            model, batch_features.cuda(), captions.cuda(), selected_ids, ARM
                        )
                        required = (
                            "loss", "caption_loss", "alignment_loss", "cpr_loss",
                            "matched_nll", "mismatched_nll", "cpr_gap", "cpr_eligible_fraction",
                        )
                        if not all(torch.isfinite(output[key]).item() for key in required):
                            raise RuntimeError(f"nonfinite objective at resumed step {step}")
                        output["loss"].backward()
                        if any(parameter.grad is None or not torch.isfinite(parameter.grad).all()
                               for _, parameter in trainable):
                            raise RuntimeError(f"missing or nonfinite gradient at resumed step {step}")
                        optimizer.step()
                        row = {key: float(value.detach()) for key, value in output.items()}
                        row.update(step=step, epoch=epoch)
                        old_metric_line = old_metrics.readline()
                        if old_metric_line and row != json.loads(old_metric_line):
                            raise RuntimeError(f"objective diverged from uninterrupted log at step {step}")
                        log.write(json.dumps(row, sort_keys=True) + "\n")
                        log.flush()
                        epoch_losses.append(row["caption_loss"])
                        minimum_eligible_fraction = min(minimum_eligible_fraction, row["cpr_eligible_fraction"])
                        if step == target_steps:
                            break

                    if step == target_steps and verify_steps is not None:
                        break
                    print(
                        f"[resume seed={SEED} {ARM}] epoch={epoch}/{registered.FORMAL_EPOCHS} "
                        f"step={step} CE={sum(epoch_losses) / len(epoch_losses):.6f}",
                        flush=True,
                    )
                    if epoch == registered.FORMAL_EPOCHS:
                        caption = evaluate_validation(
                            model, context["val_features"], context["validation"],
                            context["references"], donors, epoch, step, "cuda"
                        )
                        retrieval = retrieval_diagnostic(
                            model, context["val_features"], context["text_embeddings"],
                            context["caption_gold"], "cuda"
                        )
                        final_checkpoint = arm_dir / f"checkpoint_epoch{epoch:03d}.pt"
                        save_training_checkpoint(
                            final_checkpoint,
                            model=model,
                            optimizer=optimizer,
                            scheduler=None,
                            progress={"step": step, "epoch": epoch, "epoch_complete": True,
                                      "best_val": caption["metrics"]["native_nll"]},
                            args=checkpoint["args"],
                            training_contract=registered.training_contract(ARM),
                            dataset_contract=context["baseline_preflight"]["dataset_contract"],
                            loader_state=checkpoint["loader_state"],
                        )
                        evaluation = {
                            "epoch": epoch, "step": step, "caption": caption,
                            "retrieval": retrieval,
                            "checkpoint": str(final_checkpoint.resolve()),
                            "checkpoint_sha256": sha256_file(final_checkpoint),
                        }
                        write_json(arm_dir / f"evaluation_epoch{epoch:03d}.json", evaluation)
                if verify_steps is None and historical.readline():
                    raise RuntimeError("historical trace has extra formal steps")

    if verify_steps is not None:
        if comparisons != verify_steps:
            raise RuntimeError("short replay did not compare every requested step")
        report = {
            "verified": True, "seed": SEED, "arm": ARM,
            "checkpoint_sha256": PAUSED_SHA256["checkpoint_epoch040.pt"],
            "start_step": RESUME_STEP + 1, "last_step": step,
            "trace_and_objective_exact_matches": comparisons,
            "test_split_read": False,
        }
        write_json(output_root / "verification.json", report)
        return report

    if step != registered.FORMAL_STEPS:
        raise RuntimeError(f"expected {registered.FORMAL_STEPS} steps, got {step}")
    if context["source_hashes"] != {
        name: sha256_file(ROOT / name) for name in registered.SOURCE_FILES
    }:
        raise RuntimeError("registered source changed during resumed training")
    for split, path in (("train", TRAIN), ("val", VAL)):
        if sha256_file(path) != DATA_SHA256[split]:
            raise RuntimeError(f"{split} annotations changed during resumed training")
    visits: Counter[str] = Counter()
    coverage = {video_id: Counter() for video_id in ids}
    for epoch, batches in enumerate(context["schedule"]["batches"], 1):
        for indices in batches:
            for index in indices:
                video_id = ids[index]
                visits[video_id] += 1
                order = context["schedule"]["orders"][video_id]
                coverage[video_id][order[(epoch - 1) % len(order)]] += 1
    if len(visits) != 1200 or set(visits.values()) != {66}:
        raise RuntimeError("formal video coverage is incomplete")
    if not all(len(coverage[video_id]) == len(context["records"][index]["captions"])
               for index, video_id in enumerate(ids)):
        raise RuntimeError("formal caption-reference coverage is incomplete")
    checked_lines(trace_path, registered.FORMAL_STEPS)
    if sha256_file(trace_path) != registered.REGISTERED_INPUTS[SEED]["trace"][1]:
        raise RuntimeError("combined trace differs from the complete historical trace")
    final_hashes = component_hashes(model)
    if any(final_hashes[key] != context["initial_hashes"][key]
           for key in ("decoder", "backbone")):
        raise RuntimeError("frozen component changed")
    if any(final_hashes[key] == context["initial_hashes"][key]
           for key in ("projection", "mapper")):
        raise RuntimeError("registered trainable component did not change")
    load_strict(model, final_checkpoint)
    if component_hashes(model) != final_hashes:
        raise RuntimeError("final checkpoint strict reload changed component hashes")
    if retrieval_diagnostic(model, context["val_features"], context["text_embeddings"],
                            context["caption_gold"], "cuda") != evaluation["retrieval"]:
        raise RuntimeError("final checkpoint strict reload changed retrieval diagnostic")
    evaluations = [
        json.loads((PAUSED_ARM / f"evaluation_epoch{epoch:03d}.json").read_text(encoding="utf-8"))
        for epoch in (5, 20, 40)
    ] + [evaluation]
    result = {
        "seed": SEED, "arm": ARM, "smoke_test": False, "paper_ready": False,
        "test_split_read": False, "steps": step,
        "initial_component_hashes": context["initial_hashes"],
        "final_component_hashes": final_hashes,
        "training_contract": registered.training_contract(ARM),
        "evaluations": evaluations,
        "trace_sha256": sha256_file(trace_path),
        "historical_trace_prefix_match": True,
        "minimum_cpr_eligible_fraction": minimum_eligible_fraction,
        "final_checkpoint_strict_reload": True,
        "frozen_weights_unchanged": True,
        "source_and_data_unchanged": True,
        "video_visits": dict(visits),
        "unique_reference_indices": sum(len(item) for item in coverage.values()),
        "elapsed_seconds_resumed_segment": time.monotonic() - started,
        "peak_allocated_mib_resumed_segment": torch.cuda.max_memory_allocated() / 2**20,
        "peak_reserved_mib_resumed_segment": torch.cuda.max_memory_reserved() / 2**20,
        "resume_provenance": {
            "checkpoint": str(CHECKPOINT.resolve()),
            "checkpoint_sha256": PAUSED_SHA256["checkpoint_epoch040.pt"],
            "resume_epoch": RESUME_EPOCH + 1,
            "resume_step": RESUME_STEP + 1,
            "compared_uninterrupted_steps": comparisons,
            "paused_trace_sha256": PAUSED_SHA256["trace.jsonl"],
            "paused_metrics_sha256": PAUSED_SHA256["train_metrics.jsonl"],
        },
    }
    write_json(arm_dir / "result.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True, type=Path)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--verify-steps", type=int)
    mode.add_argument("--complete", action="store_true")
    args = parser.parse_args()
    output_root = args.output_root.resolve()
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output_root}")
    if output_root == PAUSED_ROOT.resolve() or PAUSED_ROOT.resolve() in output_root.parents:
        raise ValueError("output must be outside the paused run")
    if args.verify_steps is not None and not 1 <= args.verify_steps <= 511:
        raise ValueError("verification must replay between 1 and 511 recorded steps")
    if not torch.cuda.is_available():
        raise RuntimeError("registered C/D resume requires the local CUDA GPU")
    context = prepare_context()
    output_root.mkdir(parents=True)
    report = replay(context, output_root, args.verify_steps)
    print(json.dumps(report if args.verify_steps is not None else {
        "completed": True, "seed": SEED, "arm": ARM,
        "result": str((output_root / "seed124/C/result.json").resolve()),
    }, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
