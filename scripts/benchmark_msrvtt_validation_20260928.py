"""Time the registered full validation workload on a fixed 32-video val sample.

This is a resource benchmark on an untrained model. It does not select a model,
compute test predictions, or count as a formal validation result.
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
DATA = ROOT / "runs/msrvtt_data_full_20260928"

import torch

from scripts.diagnose_prefix_sensitivity import donor_permutation
from scripts.run_caption_policy_ab import tokenize_references
from scripts.run_full_msvd_balanced import evaluate_validation
from src.cli.train_caption_mapper import set_seed
from src.models.caption_model import VideoCaptionModel
from src.training.checkpoint import sha256_file


def main(out: Path) -> None:
    if out.exists():
        raise FileExistsError(out)
    if json.loads((DATA / "data_gate.json").read_text(encoding="utf-8"))["status"] != "PASS":
        raise RuntimeError("Data gate not passed")
    out.mkdir(parents=True)
    records = json.loads((DATA / "annotations/val.json").read_text(encoding="utf-8"))
    cache = torch.load(DATA / "feature_cache/val_raw_vit.pt", map_location="cpu", weights_only=True)
    if len(records) != 497 or cache["video_ids"] != [row["video_id"] for row in records]:
        raise RuntimeError("Validation cache coverage/order mismatch")
    indices = [round(i * (len(records) - 1) / 31) for i in range(32)]
    selected = [records[i] for i in indices]
    features = cache["raw_features"][indices]
    source = json.loads((ROOT / "runs/full_msvd_balanced_20260905/preflight.json").read_text(encoding="utf-8"))["model_source"]["path"]
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = True
    set_seed(123)
    model = VideoCaptionModel(gpt2_name=source, freeze_vit=True, vit_enable_torch_compile=False).cuda().eval()
    references = {row["video_id"]: tokenize_references(model.decoder.tokenizer, row["captions"])
                  for row in selected}
    donors = donor_permutation(len(selected), seed=321)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    result = evaluate_validation(model, features, selected, references, donors, 0, 0, "cuda")
    elapsed = time.perf_counter() - started
    payload = {"status": "PASS", "at_utc": datetime.now(timezone.utc).isoformat(),
               "scope": "timing only, untrained 32-video val sample; no model selection or test access",
               "indices": indices, "video_ids": [row["video_id"] for row in selected],
               "reference_count": result["reference_count"],
               "sample_count": len(selected), "full_val_count": len(records),
               "measured_seconds": elapsed, "seconds_per_video": elapsed / len(selected),
               "linear_497_seconds": elapsed * len(records) / len(selected),
               "peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
               "script_sha256": sha256_file(Path(__file__)),
               "val_annotation_sha256": sha256_file(DATA / "annotations/val.json"),
               "test_split_read": False}
    with (out / "result.json").open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps(payload, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    try:
        main(args.output)
    except Exception:
        if args.output.is_dir() and not (args.output / "failure.json").exists():
            (args.output / "failure.json").write_text(json.dumps({"at_utc": datetime.now(timezone.utc).isoformat(),
                "traceback": traceback.format_exc()}, ensure_ascii=False, indent=2), encoding="utf-8")
        raise
