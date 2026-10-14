"""Execute the frozen MSR-VTT test2990 protocol in recoverable GPU windows.

Preparation builds the once-only ViT feature and text embedding caches after
validation freeze. Each later invocation evaluates exactly one fixed endpoint.
The frozen evaluation functions, decoding, labels, ranking and metrics are
imported from the registered runner; test quality never controls the schedule.
"""
from __future__ import annotations

import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import argparse
from datetime import datetime, timezone
import gc
import json
from pathlib import Path
import sys
import time
import traceback

CODE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(CODE_ROOT))

import torch

from scripts import run_msrvtt_test2990_20260929 as frozen
from scripts.caption_metrics import compute_caption_metrics
from scripts.eval_retrieval import encode_texts
from scripts.msrvtt_blockwise_retrieval import evaluate_blockwise
from scripts.run_full_msvd_infonce_ablation import load_strict
from scripts.run_overfit16 import component_hashes, state_sha, tensor_sha
from src.models.caption_model import VideoCaptionModel
from src.training.checkpoint import sha256_file

CONFIG_PATH = CODE_ROOT / "experiments/msrvtt_ab_formal_config_20260928.json"
DATA = CODE_ROOT / "runs/msrvtt_data_full_20260928"
ORDER = [(seed, arm) for seed in (123, 124, 125) for arm in ("A", "B")]
TRAINING_ROOT_NAME = "msrvtt_staged_125B"
OUTPUT_NAME = "msrvtt_test2990_staged_20261002"
SOURCES = (
    "scripts/run_msrvtt_test2990_20260929.py",
    "scripts/msrvtt_blockwise_retrieval.py",
    "scripts/caption_metrics.py",
    "scripts/eval_retrieval.py",
    "src/models/caption_model.py",
    "src/models/text_decoder.py",
)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_new(path: Path, value: object) -> None:
    frozen.write_new(path, value)


def setup_torch() -> None:
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = True


def frozen_inputs(training_root: Path):
    config = read_json(CONFIG_PATH)
    test_config_path = Path(config["test_evaluation_config_path"])
    if (config["status"] != "FROZEN" or config["epochs"] != 30
            or sha256_file(test_config_path) != config["test_evaluation_config_sha256"]):
        raise RuntimeError("Frozen training/test config changed")
    test_config = read_json(test_config_path)
    if test_config["status"] != "FROZEN_BEFORE_FORMAL_TRAINING":
        raise RuntimeError("Test protocol freeze missing")
    decode = frozen._decode_kwargs(test_config)
    freeze_path = training_root / "validation_freeze.json"
    completion_path = training_root / "completion.json"
    freeze = read_json(freeze_path)
    completion = read_json(completion_path)
    if (not freeze["six_arms_complete"] or not freeze["test_not_read"]
            or not completion["technical_training_complete"]
            or freeze["fixed_endpoint_epoch"] != config["epochs"]
            or freeze["test_evaluation_config_sha256"] != sha256_file(test_config_path)):
        raise RuntimeError("Six-arm validation freeze contract failed")
    if read_json(DATA / "data_gate.json")["status"] != "PASS":
        raise RuntimeError("Data gate changed")
    return config, test_config, test_config_path, decode, freeze_path


def records_and_ids():
    manifest = read_json(DATA / "manifest.json")
    selected = [row for row in manifest if row["split"] == "test"]
    ids = [row["video_id"] for row in selected]
    pinned = (DATA / "split_lists/test_list_full.txt").read_text(encoding="utf-8-sig").splitlines()
    if ids != pinned or len(ids) != 2990 or sum(row["caption_count"] for row in selected) != 59794:
        raise RuntimeError("Test split/annotation count changed")
    records = [{"video_id": row["video_id"], "video": row["media_path"],
                "frames_dir": row["frames_dir"],
                "captions": [ref["caption"] for ref in row["captions"]]}
               for row in selected]
    return records, ids


def endpoint_specs(training_root: Path, config: dict) -> list[dict]:
    specs = []
    for seed, arm in ORDER:
        folder = training_root / f"seed{seed}" / arm
        result = read_json(folder / "result.json")
        if (result["steps"] != config["steps_per_arm"]
                or result["epochs"] != config["epochs"]
                or not result["technical_run_complete"]
                or not result["final_checkpoint_strict_reload"]):
            raise RuntimeError(f"Training endpoint incomplete: {seed}{arm}")
        checkpoint = folder / f"checkpoint_epoch{config['epochs']:03d}.pt"
        expected_hash = result["evaluations"][-1]["checkpoint_sha256"]
        if sha256_file(checkpoint) != expected_hash:
            raise RuntimeError(f"Endpoint checkpoint changed: {seed}{arm}")
        specs.append({"seed": seed, "arm": arm, "path": str(checkpoint),
                      "sha256": expected_hash})
    return specs


def prepare(training_root: Path, out: Path) -> None:
    if out.exists():
        raise FileExistsError(out)
    config, test_config, test_config_path, _, freeze_path = frozen_inputs(training_root)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    specs = endpoint_specs(training_root, config)
    records, ids = records_and_ids()
    out.mkdir(parents=True)
    setup_torch()
    annotations = out / "test_annotations.json"
    write_new(annotations, records)
    model = VideoCaptionModel(gpt2_name=config["gpt2_source_path"], freeze_vit=True,
                              vit_enable_torch_compile=False).cuda().eval()
    cache_audit = read_json(DATA / "feature_cache/cache_audit.json")
    if state_sha(model.encoder.backbone.state_dict()) != cache_audit["backbone_state_sha256"]:
        raise RuntimeError("Test frozen ViT differs from train/val cache source")
    load_strict(model, Path(specs[0]["path"]))
    features, feature_audit = frozen.build_test_features(model, annotations, ids,
                                                         out / "test_raw_vit.pt")
    write_new(out / "test_feature_audit.json", feature_audit)
    captions = [text for row in records for text in row["captions"]]
    tokenizer = model.decoder.tokenizer
    text_limit = test_config["retrieval"]["text_tokenizer_max_length"]
    bad_lengths = []
    for index, text in enumerate(captions):
        length = len(tokenizer.encode(text, add_special_tokens=False))
        if not 0 < length < text_limit:
            bad_lengths.append((index, length))
    if bad_lengths:
        write_new(out / "test_text_length_failure.json", {"max_length": text_limit,
                                                            "invalid_caption_indices_and_lengths": bad_lengths})
        raise RuntimeError("Test retrieval text would be truncated/empty")
    text_embeddings = encode_texts(model, captions, text_limit, 128, "cuda")
    embedding_path = out / "test_text_embeddings.pt"
    with embedding_path.open("xb") as stream:
        torch.save({"embeddings": text_embeddings.cpu(), "caption_count": len(captions)}, stream)
    write_new(out / "preflight.json", {
        "at_utc": now(), "config_sha256": sha256_file(CONFIG_PATH),
        "test_config_sha256": sha256_file(test_config_path),
        "validation_freeze_sha256": sha256_file(freeze_path),
        "source_sha256": {name: sha256_file(CODE_ROOT / name) for name in SOURCES},
        "staged_runner_sha256": sha256_file(Path(__file__)),
        "test_annotations_sha256": sha256_file(annotations),
        "test_features": feature_audit,
        "text_embedding_count": len(text_embeddings),
        "text_embedding_tensor_sha256": tensor_sha(text_embeddings),
        "text_embedding_file_sha256": sha256_file(embedding_path),
        "checkpoints": specs, "test_selection_performed": False,
    })
    print(json.dumps({"phase": "prepare_complete", "video_count": len(ids),
                      "caption_count": len(captions),
                      "feature_seconds": feature_audit["elapsed_seconds"]}), flush=True)


def checked_preflight(training_root: Path, out: Path):
    config, test_config, test_config_path, decode, freeze_path = frozen_inputs(training_root)
    preflight = read_json(out / "preflight.json")
    if (preflight["config_sha256"] != sha256_file(CONFIG_PATH)
            or preflight["test_config_sha256"] != sha256_file(test_config_path)
            or preflight["validation_freeze_sha256"] != sha256_file(freeze_path)
            or preflight["staged_runner_sha256"] != sha256_file(Path(__file__))
            or preflight["source_sha256"] != {name: sha256_file(CODE_ROOT / name) for name in SOURCES}
            or preflight["test_selection_performed"] is not False):
        raise RuntimeError("Frozen test preflight changed")
    specs = endpoint_specs(training_root, config)
    if specs != preflight["checkpoints"]:
        raise RuntimeError("Frozen endpoints differ from preflight")
    records, ids = records_and_ids()
    if (sha256_file(out / "test_annotations.json") != preflight["test_annotations_sha256"]
            or sha256_file(out / "test_raw_vit.pt") != preflight["test_features"]["cache_sha256"]
            or sha256_file(out / "test_text_embeddings.pt") != preflight["text_embedding_file_sha256"]):
        raise RuntimeError("Test setup artifact changed")
    return config, test_config, decode, specs, records, ids, preflight


def run_arm(training_root: Path, out: Path, seed: int, arm: str) -> None:
    if (seed, arm) not in ORDER:
        raise ValueError("Unregistered seed/arm")
    config, _, decode, specs, records, ids, preflight = checked_preflight(training_root, out)
    index = ORDER.index((seed, arm))
    label = f"seed{seed}_{arm}"
    result_path = out / f"result_{label}.json"
    if result_path.exists():
        if read_json(result_path)["caption_metrics_status"] != "complete":
            raise RuntimeError(f"Arm result needs a documented metric repair: {label}")
        finish_or_pause(training_root, out, index)
        print(json.dumps({"phase": "already_scored", "seed": seed, "arm": arm}), flush=True)
        return
    for previous_seed, previous_arm in ORDER[:index]:
        prior = out / f"result_seed{previous_seed}_{previous_arm}.json"
        if not prior.is_file() or read_json(prior)["caption_metrics_status"] != "complete":
            raise RuntimeError(f"Prior frozen test arm incomplete: {prior}")
    for later_seed, later_arm in ORDER[index + 1:]:
        if (out / f"result_seed{later_seed}_{later_arm}.json").exists():
            raise RuntimeError("Test arm order violated")
    gpu_result_path = out / f"gpu_result_{label}.json"
    if gpu_result_path.exists():
        saved = read_json(gpu_result_path)
        if (saved["checkpoint_sha256"] != specs[index]["sha256"]
                or saved["prediction_count"] != 2990
                or saved["prediction_sha256"] != sha256_file(Path(saved["prediction_file"]))):
            raise RuntimeError("Saved GPU result changed")
        score_arm(out, records, saved)
        finish_or_pause(training_root, out, index)
        print(json.dumps({"phase": "cpu_score_recovered", "seed": seed, "arm": arm}), flush=True)
        return
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    setup_torch()
    spec = specs[index]
    model = VideoCaptionModel(gpt2_name=config["gpt2_source_path"], freeze_vit=True,
                              vit_enable_torch_compile=False).cuda().eval()
    feature_cache = torch.load(out / "test_raw_vit.pt", map_location="cpu", weights_only=True)
    if feature_cache["video_ids"] != ids or tuple(feature_cache["raw_features"].shape) != (2990, 768):
        raise RuntimeError("Test feature coverage/order changed")
    text_cache = torch.load(out / "test_text_embeddings.pt", map_location="cpu", weights_only=True)
    if text_cache["caption_count"] != 59794 or len(text_cache["embeddings"]) != 59794:
        raise RuntimeError("Test text embedding coverage changed")
    text_embeddings = text_cache["embeddings"]
    gold = torch.tensor([i for i, row in enumerate(records) for _ in row["captions"]], dtype=torch.long)
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    checkpoint = load_strict(model, Path(spec["path"]))
    if (checkpoint["progress"]["step"] != config["steps_per_arm"]
            or checkpoint["progress"]["epoch"] != config["epochs"]):
        raise RuntimeError("Checkpoint endpoint changed")
    hashes = component_hashes(model)
    frozen_hashes = {key: hashes[key] for key in ("decoder", "backbone")}
    if index:
        first = read_json(out / "gpu_result_seed123_A.json")
        if frozen_hashes != {key: first["component_hashes"][key] for key in ("decoder", "backbone")}:
            raise RuntimeError("Frozen model differs across arms")
    prediction_path = out / f"predictions_{label}.json"
    video_path = out / f"video_embeddings_{label}.pt"
    if prediction_path.exists() != video_path.exists():
        raise RuntimeError("Partial generation artifacts require targeted recovery")
    if prediction_path.exists():
        predictions = read_json(prediction_path)
        video_embeddings = torch.load(video_path, map_location="cpu", weights_only=True)["embeddings"]
        replay = True  # Confirmed on the original generation pass; see saved manifest.
        recovery = read_json(out / f"generation_{label}.json")
        if (recovery["prediction_sha256"] != sha256_file(prediction_path)
                or recovery["video_embedding_sha256"] != sha256_file(video_path)
                or not recovery["first_batch_replay_exact"]):
            raise RuntimeError("Saved generation artifacts changed")
    else:
        predictions, video_embeddings, replay = frozen.generate_and_embed(
            model, feature_cache["raw_features"], ids, decode)
        write_new(prediction_path, predictions)
        with video_path.open("xb") as stream:
            torch.save({"embeddings": video_embeddings.cpu(), "video_ids": ids}, stream)
        write_new(out / f"generation_{label}.json", {
            "at_utc": now(), "prediction_sha256": sha256_file(prediction_path),
            "video_embedding_sha256": sha256_file(video_path),
            "first_batch_replay_exact": replay,
        })
    if len(predictions) != 2990 or [row["video_id"] for row in predictions] != ids:
        raise RuntimeError("Prediction coverage/order changed")
    if tuple(video_embeddings.shape) != (2990, text_embeddings.shape[1]):
        raise RuntimeError("Video embedding coverage/shape changed")
    empty = [row["video_id"] for row in predictions if not row["caption"]]
    retrieval = evaluate_blockwise(video_embeddings, text_embeddings, gold,
                                   device="cuda", text_batch_size=256,
                                   video_batch_size=16, derangement_seed=123)
    gpu_result = {"seed": seed, "arm": arm,
                  "checkpoint": spec["path"], "checkpoint_sha256": spec["sha256"],
                  "checkpoint_progress": checkpoint["progress"],
                  "component_hashes": hashes, "decode": decode,
                  "prediction_file": str(prediction_path),
                  "prediction_sha256": sha256_file(prediction_path),
                  "prediction_count": len(predictions), "empty_prediction_ids": empty,
                  "first_batch_replay_exact": replay,
                  "caption_metrics_status": "pending_cpu_phase", "caption_metrics": None,
                  "retrieval": retrieval, "elapsed_seconds": time.perf_counter() - started,
                  "peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
                  "test_selection_performed": False}
    write_new(gpu_result_path, gpu_result)
    del model, text_embeddings, video_embeddings, feature_cache, gold
    gc.collect()
    torch.cuda.empty_cache()
    score_arm(out, records, gpu_result)
    finish_or_pause(training_root, out, index)
    print(json.dumps({"phase": "arm_complete", "seed": seed, "arm": arm,
                      "next_arm": ORDER[index + 1] if index < 5 else None,
                      "prediction_count": len(predictions)}), flush=True)


def score_arm(out: Path, records: list[dict], gpu_result: dict) -> None:
    label = f"seed{gpu_result['seed']}_{gpu_result['arm']}"
    if (out / f"result_{label}.json").exists():
        raise FileExistsError(f"Scored result already exists: {label}")
    result = dict(gpu_result)
    if result["empty_prediction_ids"]:
        result["caption_metrics_status"] = "unavailable_empty_prediction"
    else:
        predictions = read_json(Path(result["prediction_file"]))
        result["caption_metrics"] = compute_caption_metrics(records, predictions)
        result["caption_metrics_status"] = "complete"
    write_new(out / f"result_{label}.json", result)
    if result["caption_metrics_status"] != "complete":
        raise RuntimeError(f"Caption metric phase incomplete: {label}")


def finish_or_pause(training_root: Path, out: Path, index: int) -> None:
    if index == 5:
        if not (out / "completion.json").exists():
            finish(training_root, out)
        return
    seed, arm = ORDER[index]
    pause_path = out / f"planned_pause_after_seed{seed}_{arm}.json"
    if not pause_path.exists():
        write_new(pause_path, {
            "at_utc": now(), "status": "TEST_ARM_COMPLETE_PAUSED",
            "completed_arms": [list(item) for item in ORDER[:index + 1]],
            "next_arm": list(ORDER[index + 1]), "test_selection_performed": False,
        })


def finish(training_root: Path, out: Path) -> None:
    _, _, _, _, _, _, _ = checked_preflight(training_root, out)
    results = [read_json(out / f"result_seed{seed}_{arm}.json") for seed, arm in ORDER]
    if any(row["caption_metrics_status"] != "complete" or row["prediction_count"] != 2990
           for row in results):
        raise RuntimeError("Six-arm test completion contract failed")
    summary = frozen.summarize(results)
    write_new(out / "completion.json", {
        "at_utc": now(), "technical_test_run_complete": True,
        "caption_metrics_complete": summary["caption_metrics_complete"],
        "results": [{"seed": row["seed"], "arm": row["arm"]} for row in results],
        "failures": [], "summary": summary, "test_selection_performed": False,
        "staged_execution": True,
    })


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", required=True, choices=("prepare", "arm"))
    parser.add_argument("--arm", choices=tuple(f"{seed}{arm}" for seed, arm in ORDER))
    args = parser.parse_args()
    base = Path(__file__).parent
    training_root = base / "msrvtt_runs" / TRAINING_ROOT_NAME
    out = base / "msrvtt_runs" / OUTPUT_NAME
    try:
        if args.phase == "prepare":
            if args.arm is not None:
                raise ValueError("Prepare phase takes no arm")
            prepare(training_root, out)
        else:
            if args.arm is None:
                raise ValueError("Arm phase requires --arm")
            run_arm(training_root, out, int(args.arm[:3]), args.arm[3:])
    except Exception:
        if out.is_dir():
            failure = out / f"failure_staged_{args.phase}_{args.arm or 'setup'}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
            if not failure.exists():
                write_new(failure, {"at_utc": now(), "phase": args.phase,
                                    "arm": args.arm, "traceback": traceback.format_exc()})
        raise


if __name__ == "__main__":
    main()
