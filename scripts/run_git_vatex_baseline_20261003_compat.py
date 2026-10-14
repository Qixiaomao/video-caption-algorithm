"""Execute one pinned external GIT caption baseline on MSVD val or frozen test670.

The test mode requires a successful val engineering smoke and refuses to write
into an existing output. It never selects a decoding setting from test results.
"""
from __future__ import annotations

import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
BASE = ROOT / "runs/external_baseline_git_vatex_20260928"
MODEL = BASE / "model"
EXPECTED_MODEL_SHA = "04ea5c7d586ada3d9fb626eec35fef084c45d6f8b817469fb520ab9cbcca44ca"
EXPECTED_TEST_ANN_SHA = "9a1fb940f1b9513a882e7ead5620397b0bf39ad1237b9c8ccb57ce765de50b1f"
FRAME_POSITIONS = (0, 1, 3, 4, 6, 7)

import torch
from PIL import Image
from safetensors import safe_open
from transformers import AutoModelForCausalLM, AutoProcessor

from scripts.caption_metrics import compute_caption_metrics
from src.training.checkpoint import sha256_file


def write_new(path: Path, value: object) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)


def load_pinned_model():
    """Restore six checkpoint temporal tensors renamed in newer Transformers.

    The pinned file uses `temperal`; Transformers 4.57.1 expects `temporal`.
    Require this to be the entire loading difference before copying tensors.
    """
    model, info = AutoModelForCausalLM.from_pretrained(
        str(MODEL), local_files_only=True, use_safetensors=True,
        output_loading_info=True)
    old_keys = {f"git.img_temperal_embedding.{i}" for i in range(6)}
    new_keys = {f"git.img_temporal_embedding.{i}" for i in range(6)}
    if (set(info["unexpected_keys"]) != old_keys or
            set(info["missing_keys"]) != new_keys or
            info.get("mismatched_keys") or info.get("error_msgs")):
        raise RuntimeError(f"Unexpected checkpoint loading difference: {info}")
    with safe_open(str(MODEL / "model.safetensors"), framework="pt", device="cpu") as reader:
        if not old_keys.issubset(set(reader.keys())):
            raise RuntimeError("Pinned temporal tensors missing")
        with torch.no_grad():
            for i in range(6):
                tensor = reader.get_tensor(f"git.img_temperal_embedding.{i}")
                target = model.git.img_temporal_embedding[i]
                if tensor.shape != target.shape or tensor.dtype != target.dtype:
                    raise RuntimeError(f"Temporal tensor {i} shape/dtype changed")
                target.copy_(tensor)
                if not torch.equal(target.detach().cpu(), tensor):
                    raise RuntimeError(f"Temporal tensor {i} copy mismatch")
    return model.cuda().eval()


def validate_source() -> dict:
    record = json.loads((BASE / "model_download.json").read_text(encoding="utf-8"))
    if record["source"] != "microsoft/git-base-vatex" or record["revision"] != "623363cef8744c57438ce8d62b30bef1fbbec111" or record["license"] != "mit":
        raise RuntimeError("Candidate identity/license changed")
    listed = {item["name"]: item for item in record["files"]}
    if listed["model.safetensors"]["sha256"] != EXPECTED_MODEL_SHA:
        raise RuntimeError("Pinned weight hash registration changed")
    for name, item in listed.items():
        path = MODEL / name
        if path.stat().st_size != item["bytes"] or sha256_file(path) != item["sha256"]:
            raise RuntimeError(f"Candidate file changed: {name}")
    return record


def load_records(split: str) -> tuple[list[dict], str]:
    annotation = ROOT / "data/processed/msvd_full" / split / "annotations.json"
    source_hash = sha256_file(annotation)
    if split == "test" and source_hash != EXPECTED_TEST_ANN_SHA:
        raise RuntimeError("Frozen MSVD test annotation changed")
    records = json.loads(annotation.read_text(encoding="utf-8"))
    if len(records) != (100 if split == "val" else 670):
        raise RuntimeError("Unexpected MSVD split size")
    ids = [row["video_id"] for row in records]
    if len(set(ids)) != len(ids) or any(not row["captions"] for row in records):
        raise RuntimeError("Duplicate ID or missing references")
    for row in records:
        frames = sorted(Path(row["frames_dir"]).glob("frame_*.jpg"))
        if len(frames) != 8:
            raise RuntimeError(f"Expected eight prepared frames for {row['video_id']}: {len(frames)}")
        row["_frames"] = [str(frames[i]) for i in FRAME_POSITIONS]
    return records, source_hash


@torch.inference_mode()
def infer(model, processor, records: list[dict]) -> tuple[list[dict], list[dict]]:
    predictions = []
    timing = []
    for index, row in enumerate(records, 1):
        started = time.perf_counter()
        with_images = []
        for path in row["_frames"]:
            with Image.open(path) as image:
                with_images.append(image.convert("RGB"))
        inputs = processor(images=with_images, return_tensors="pt")
        if tuple(inputs["pixel_values"].shape) != (1, 6, 3, 224, 224):
            raise RuntimeError(f"Bad GIT pixel shape for {row['video_id']}")
        pixel_values = inputs["pixel_values"].cuda(non_blocking=True)
        generated = model.generate(pixel_values=pixel_values, max_new_tokens=24,
                                   min_new_tokens=0, num_beams=1, do_sample=False,
                                   no_repeat_ngram_size=3, repetition_penalty=1.0)
        torch.cuda.synchronize()
        caption = processor.batch_decode(generated, skip_special_tokens=True)[0].strip()
        predictions.append({"video_id": row["video_id"], "caption": caption})
        timing.append({"video_id": row["video_id"], "input_and_generate_seconds": time.perf_counter() - started})
        if index % 20 == 0 or index == len(records):
            print(json.dumps({"generated": index, "expected": len(records),
                              "peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20}), flush=True)
    if [item["video_id"] for item in predictions] != [row["video_id"] for row in records]:
        raise RuntimeError("Baseline prediction coverage/order mismatch")
    return predictions, timing


def main(mode: str) -> None:
    out = BASE / ("smoke_compat_20261003" if mode == "smoke" else "formal_test670_compat_20261003")
    if out.exists():
        raise FileExistsError(out)
    identity = validate_source()
    if mode == "formal_test670":
        smoke = json.loads((BASE / "smoke_compat_20261003/result.json").read_text(encoding="utf-8"))
        if smoke["status"] != "PASS" or not smoke["under_4gb"]:
            raise RuntimeError("External baseline smoke gate not passed")
        if smoke.get("legacy_temporal_key_correction") != "exact_six_tensor_copy":
            raise RuntimeError("External baseline smoke used an uncorrected checkpoint")
        if smoke["model_download_sha256"] != sha256_file(BASE / "model_download.json"):
            raise RuntimeError("External checkpoint changed since smoke")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    split = "val" if mode == "smoke" else "test"
    all_records, source_hash = load_records(split)
    records = all_records[:16] if mode == "smoke" else all_records
    out.mkdir(parents=True)
    started = time.perf_counter()
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(123)
    processor = AutoProcessor.from_pretrained(str(MODEL), local_files_only=True, use_fast=False)
    model = load_pinned_model()
    if model.config.num_image_with_embedding != 6 or model.config.eos_token_id != 102:
        raise RuntimeError("GIT video frame/EOS config changed")
    torch.cuda.reset_peak_memory_stats()
    predictions, timing = infer(model, processor, records)
    peak = torch.cuda.max_memory_reserved() / 2**20
    write_new(out / "predictions.json", predictions)
    write_new(out / "per_video_timing.json", timing)
    result = {"status": "PASS" if peak < 4096 else "FAIL", "at_utc": datetime.now(timezone.utc).isoformat(),
              "mode": mode, "split": split, "count": len(predictions),
              "full_split_count": len(all_records), "annotation_sha256": source_hash,
              "model_source": identity["source"], "revision": identity["revision"],
              "model_download_sha256": sha256_file(BASE / "model_download.json"),
              "model_weight_sha256": EXPECTED_MODEL_SHA,
              "legacy_temporal_key_correction": "exact_six_tensor_copy",
              "protocol_sha256": sha256_file(ROOT / "experiments/external_baseline_protocol_20260928.md"),
              "script_sha256": sha256_file(Path(__file__)),
              "transformers_version": __import__("transformers").__version__,
              "decode": {"max_new_tokens": 24, "min_new_tokens": 0, "num_beams": 1,
                         "do_sample": False, "no_repeat_ngram_size": 3,
                         "repetition_penalty": 1.0},
              "frame_positions_from_eight": list(FRAME_POSITIONS),
              "empty_prediction_ids": [item["video_id"] for item in predictions if not item["caption"]],
              "predictions_sha256": sha256_file(out / "predictions.json"),
              "timing_sha256": sha256_file(out / "per_video_timing.json"),
              "wall_seconds": time.perf_counter() - started,
              "peak_reserved_mib": peak, "under_4gb": peak < 4096,
              "test_metric_used_for_selection": False}
    if mode == "formal_test670":
        metrics = compute_caption_metrics(all_records, predictions)
        write_new(out / "caption_metrics.json", metrics)
        result["caption_metrics_status"] = "complete"
        result["caption_metrics_sha256"] = sha256_file(out / "caption_metrics.json")
        result["corpus_x100"] = metrics["corpus_x100"]
    write_new(out / "result.json", result)
    print(json.dumps({"status": result["status"], "mode": mode,
                      "count": result["count"], "peak_reserved_mib": peak,
                      "wall_seconds": result["wall_seconds"]}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True, choices=("smoke", "formal_test670"))
    args = parser.parse_args()
    try:
        main(args.mode)
    except Exception:
        out = BASE / ("smoke_compat_20261003" if args.mode == "smoke" else "formal_test670_compat_20261003")
        if out.is_dir() and not (out / "failure.json").exists():
            write_new(out / "failure.json", {"at_utc": datetime.now(timezone.utc).isoformat(),
                                             "traceback": traceback.format_exc()})
        raise
