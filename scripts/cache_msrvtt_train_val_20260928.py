"""Build pinned train/val frozen-ViT features after the full media audit passes.

All batches check direct encoder output against raw cached feature projection.
The test split is never loaded or passed to the model here.
"""
from __future__ import annotations

import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
RUN = ROOT / "runs/msrvtt_data_full_20260928"

import torch

from scripts.run_caption_policy_ab import raw_backbone_features
from scripts.run_overfit16 import state_sha, tensor_sha
from src.data.data_loader import build_dataloader
from src.models.caption_model import VideoCaptionModel


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    if json.loads((RUN / "data_audit.json").read_text(encoding="utf-8"))["status"] != "PASS":
        raise RuntimeError("Full media audit has not passed")
    if json.loads((RUN / "train_val_caption_token_audit.json").read_text(encoding="utf-8"))["status"] != "PASS":
        raise RuntimeError("Train/val caption token audit has not passed")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    target = RUN / "feature_cache"
    if target.exists():
        raise FileExistsError(f"Feature output already exists: {target}")
    target.mkdir()
    started = time.perf_counter()
    preflight = json.loads((ROOT / "runs/full_msvd_balanced_20260905/preflight.json").read_text(encoding="utf-8"))
    source_path = preflight["model_source"]["path"]
    model = VideoCaptionModel(gpt2_name=source_path, freeze_vit=True,
                              vit_enable_torch_compile=False)
    model.eval()
    model.encoder.to("cuda")
    if any(parameter.requires_grad for parameter in model.encoder.backbone.parameters()):
        raise RuntimeError("ViT backbone is not frozen")
    source_hash = state_sha(model.encoder.backbone.state_dict())
    tokenizer = model.decoder.tokenizer
    torch.cuda.reset_peak_memory_stats()
    results = {}
    for split, expected_count in (("train", 6513), ("val", 497)):
        ann = RUN / "annotations" / f"{split}.json"
        records = json.loads(ann.read_text(encoding="utf-8"))
        expected_ids = [row["video_id"] for row in records]
        loader = build_dataloader(str(ann), tokenizer, batch_size=8,
                                  max_len=96, num_frame=8, image_size=224,
                                  shuffle=False, random_caption=False)
        actual_ids = [row["video_id"] for row in loader.dataset.anns]
        if len(records) != expected_count or actual_ids != expected_ids:
            raise RuntimeError(f"Dataloader silently changed {split} ID set")
        values = []
        seen = []
        batch_errors = []
        batch_times = []
        split_start = time.perf_counter()
        with torch.inference_mode():
            for batch_number, batch in enumerate(loader, 1):
                tick = time.perf_counter()
                video = batch["video"].to("cuda", non_blocking=True)
                direct = model.encoder(video)
                raw = raw_backbone_features(model.encoder, video)
                cached = model.encoder.proj(raw)
                error = float((direct - cached).abs().max().item())
                if error > 1e-6 or not torch.isfinite(raw).all():
                    raise RuntimeError(f"{split} cache mismatch/nonfinite at batch {batch_number}: {error}")
                values.append(raw.cpu())
                seen.extend(batch["video_id"])
                batch_errors.append(error)
                batch_times.append(time.perf_counter() - tick)
                if batch_number == 1 or batch_number % 50 == 0 or len(seen) == expected_count:
                    print(json.dumps({"split": split, "cached": len(seen), "expected": expected_count,
                                      "elapsed_seconds": round(time.perf_counter() - split_start, 1),
                                      "peak_reserved_mib": round(torch.cuda.max_memory_reserved() / 2**20, 1)}), flush=True)
        if seen != expected_ids:
            raise RuntimeError(f"{split} cache order/coverage mismatch")
        raw_features = torch.cat(values, dim=0)
        if tuple(raw_features.shape) != (expected_count, 768):
            raise RuntimeError(f"Unexpected {split} feature shape: {tuple(raw_features.shape)}")
        cache_path = target / f"{split}_raw_vit.pt"
        with cache_path.open("xb") as stream:
            torch.save({"video_ids": seen, "raw_features": raw_features,
                        "source_annotation_sha256": sha_file(ann), "backbone_state_sha256": source_hash}, stream)
        results[split] = {"count": len(seen), "annotation_sha256": sha_file(ann),
                          "cache_path": str(cache_path), "cache_sha256": sha_file(cache_path),
                          "cache_bytes": cache_path.stat().st_size,
                          "raw_feature_tensor_sha256": tensor_sha(raw_features),
                          "shape": list(raw_features.shape), "max_abs_direct_cache_error": max(batch_errors),
                          "batch_count": len(batch_errors), "batch_seconds_sum": sum(batch_times),
                          "split_wall_seconds": time.perf_counter() - split_start}
    audit = {"created_at_utc": datetime.now(timezone.utc).isoformat(),
             "status": "PASS", "model_source": source_path,
             "backbone_state_sha256": source_hash,
             "test_used": False, "test_features_created": False,
             "peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
             "wall_seconds": time.perf_counter() - started, "splits": results,
             "script_sha256": sha_file(Path(__file__))}
    with (target / "cache_audit.json").open("x", encoding="utf-8") as stream:
        json.dump(audit, stream, ensure_ascii=False, indent=2, allow_nan=False)
    media_resource = json.loads((RUN / "resource_estimate.json").read_text(encoding="utf-8"))
    media_started = datetime.fromisoformat(json.loads((RUN / "run_config.json").read_text(encoding="utf-8"))["started_at_utc"]).timestamp()
    media_wall_seconds = (RUN / "media_audit.jsonl").stat().st_mtime - media_started
    disk = shutil.disk_usage(RUN)
    complete_resource = {**media_resource,
                         "media_processing_wall_seconds": media_wall_seconds,
                         "feature_cache_seconds": audit["wall_seconds"],
                         "feature_cache_bytes": sum(item["cache_bytes"] for item in results.values()),
                         "feature_cache_peak_reserved_mib": audit["peak_reserved_mib"],
                         "feature_cache_train_val_only": True,
                         "new_run_files_bytes_after_cache": sum(path.stat().st_size for path in RUN.rglob("*") if path.is_file()),
                         "disk_free_bytes_after_cache": disk.free,
                         "disk_total_bytes": disk.total}
    with (RUN / "resource_estimate_complete.json").open("x", encoding="utf-8") as stream:
        json.dump(complete_resource, stream, ensure_ascii=False, indent=2, allow_nan=False)
    gate = {"at_utc": datetime.now(timezone.utc).isoformat(), "status": "PASS",
            "media_audit_path": str(RUN / "data_audit.json"),
            "media_audit_sha256": sha_file(RUN / "data_audit.json"),
            "manifest_path": str(RUN / "manifest.json"),
            "manifest_sha256": sha_file(RUN / "manifest.json"),
            "cache_audit_path": str(target / "cache_audit.json"),
            "cache_audit_sha256": sha_file(target / "cache_audit.json"),
            "caption_token_audit_path": str(RUN / "train_val_caption_token_audit.json"),
            "caption_token_audit_sha256": sha_file(RUN / "train_val_caption_token_audit.json"),
            "resource_path": str(RUN / "resource_estimate_complete.json"),
            "test_model_features_predictions_metrics_created": False}
    with (RUN / "data_gate.json").open("x", encoding="utf-8") as stream:
        json.dump(gate, stream, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps({"status": "PASS", "audit": str(target / "cache_audit.json"),
                      "wall_seconds": audit["wall_seconds"],
                      "peak_reserved_mib": audit["peak_reserved_mib"]}), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        failure = RUN / "feature_cache" / "failure.json"
        if failure.parent.is_dir() and not failure.exists():
            failure.write_text(json.dumps({"at_utc": datetime.now(timezone.utc).isoformat(),
                                           "traceback": traceback.format_exc()}, indent=2), encoding="utf-8")
        raise
