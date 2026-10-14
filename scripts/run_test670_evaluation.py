"""Run the frozen three-seed A/B evaluation on the MSVD test670 split."""
from __future__ import annotations

import os

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import argparse
import gc
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys
import time
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import torch.nn.functional as F

from scripts.caption_metrics import METRIC_NAMES, compute_caption_metrics
from scripts.eval_retrieval import encode_texts, evaluate_scores
from scripts.run_caption_policy_ab import write_json
from scripts.run_full_msvd_balanced import cache_split
from scripts.run_full_msvd_infonce_ablation import load_strict
from scripts.run_overfit16 import component_hashes, tensor_sha
from src.data.data_loader import build_dataloader
from src.models.caption_model import VideoCaptionModel
from src.training.checkpoint import sha256_file


TEST = ROOT / "data/processed/msvd_full/test/annotations.json"
TEST_SHA256 = "9a1fb940f1b9513a882e7ead5620397b0bf39ad1237b9c8ccb57ce765de50b1f"
DECODE = {
    "prompt": "",
    "max_new_tokens": 24,
    "min_new_tokens": 0,
    "num_beams": 1,
    "temperature": 1.0,
    "top_p": 0.9,
    "no_repeat_ngram_size": 3,
    "repetition_penalty": 1.0,
}
CHECKPOINTS = (
    {
        "seed": 123,
        "arm": "A",
        "path": ROOT / "runs/full_msvd_balanced_20260905/checkpoint_epoch066.pt",
        "sha256": "8155d68c5b3ac9f3f7396027a32b2e0ebea052a17aef79ecce8cf10a1e0f9e6a",
    },
    {
        "seed": 123,
        "arm": "B",
        "path": ROOT / "runs/full_msvd_infonce_ablation_20260906/checkpoint_epoch066.pt",
        "sha256": "891cdcceef2b52d09092dde00ef35f9e1ee652e686b30178b5c1a9aee81c7e7c",
    },
    {
        "seed": 124,
        "arm": "A",
        "path": ROOT / "runs/repeated_seeds_20260908/seed124/A/checkpoint_epoch066.pt",
        "sha256": "8acc816040d14a8581095eb80262b2fcb8afb1934876c2b659936a8d3c062be5",
    },
    {
        "seed": 124,
        "arm": "B",
        "path": ROOT / "runs/repeated_seeds_20260908/seed124/B/checkpoint_epoch066.pt",
        "sha256": "147330abd9e909a50be78b91be426b7ade23104cfcdeac34ee1bb777327067a7",
    },
    {
        "seed": 125,
        "arm": "A",
        "path": ROOT / "runs/repeated_seeds_20260908/seed125/A/checkpoint_epoch066.pt",
        "sha256": "5e5d5cae5d663588cb5c531aaf76e45634bf9fbebb59e75b3fd5c8ab8f5cee37",
    },
    {
        "seed": 125,
        "arm": "B",
        "path": ROOT / "runs/repeated_seeds_20260908/seed125/B/checkpoint_epoch066.pt",
        "sha256": "9e59425dfd28b0b2df3539bd023f712506c533b8f65421ee07ca7679ddeaefe5",
    },
)
SOURCE_FILES = (
    "requirements.txt",
    "experiments/test670_protocol_20260908.md",
    "scripts/caption_metrics.py",
    "scripts/run_test670_evaluation.py",
    "scripts/eval_retrieval.py",
    "scripts/run_caption_policy_ab.py",
    "scripts/run_full_msvd_balanced.py",
    "scripts/run_full_msvd_infonce_ablation.py",
    "scripts/run_overfit16.py",
    "src/data/data_loader.py",
    "src/models/caption_model.py",
    "src/models/text_decoder.py",
    "src/models/video_encoder.py",
    "src/training/checkpoint.py",
)


def validate_checkpoint_payload(payload: dict[str, Any], spec: dict[str, Any]) -> dict[str, Any]:
    progress = payload.get("progress", {})
    contract = payload.get("training_contract", {})
    expected_objective = "infonce" if spec["arm"] == "B" else "none"
    expected_weight = 0.1 if spec["arm"] == "B" else 0.0
    gates = {
        "checkpoint_version_3": payload.get("checkpoint_version") == 3,
        "epoch_66": progress.get("epoch") == 66,
        "step_9900": progress.get("step") == 9900,
        "epoch_complete": progress.get("epoch_complete") is True,
        "balanced_references": contract.get("caption_policy") == "balanced_all_references",
        "alignment_objective": contract.get("alignment_objective") == expected_objective,
        "alignment_weight": contract.get("alignment_weight") == expected_weight,
        "frozen_gpt2": contract.get("frozen_gpt2") is True,
        "frozen_vit": contract.get("frozen_vit_backbone") is True,
    }
    if not all(gates.values()):
        raise RuntimeError(
            f"Checkpoint contract failed for seed {spec['seed']} arm {spec['arm']}: {gates}"
        )
    args = payload.get("args", {})
    for name in ("gpt2_name", "vit_name", "cond_mode", "prefix_len"):
        if name not in args:
            raise RuntimeError(f"Checkpoint is missing model argument {name}")
    return {
        "checkpoint_version": payload["checkpoint_version"],
        "progress": progress,
        "training_contract": contract,
        "args": {name: args[name] for name in ("gpt2_name", "vit_name", "cond_mode", "prefix_len")},
        "gates": gates,
    }


def mean_sd(values: list[float]) -> dict[str, float]:
    if len(values) < 2 or not all(math.isfinite(value) for value in values):
        raise ValueError("Mean/SD requires at least two finite values")
    return {"mean": statistics.mean(values), "sample_sd": statistics.stdev(values)}


def summarize_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    if {(item["seed"], item["arm"]) for item in results} != {
        (seed, arm) for seed in (123, 124, 125) for arm in ("A", "B")
    }:
        raise ValueError("Summary requires all six registered seed/arm results")
    fields: dict[str, tuple[str, ...]] = {
        **{name: ("caption_metrics", "corpus_x100", name) for name in METRIC_NAMES},
        "T2V_R1": ("retrieval", "true_labels", "text_to_video", "r_at_1"),
        "T2V_R5": ("retrieval", "true_labels", "text_to_video", "r_at_5"),
        "T2V_R10": ("retrieval", "true_labels", "text_to_video", "r_at_10"),
        "V2T_R1": ("retrieval", "true_labels", "video_to_text", "r_at_1"),
        "V2T_R5": ("retrieval", "true_labels", "video_to_text", "r_at_5"),
        "V2T_R10": ("retrieval", "true_labels", "video_to_text", "r_at_10"),
        "similarity_gap": ("retrieval", "similarity", "gap"),
    }

    def get(item: dict[str, Any], path: tuple[str, ...]) -> float:
        value: Any = item
        for key in path:
            value = value[key]
        return float(value)

    ordered = sorted(results, key=lambda item: (item["seed"], item["arm"]))
    metrics: dict[str, Any] = {}
    for name, path in fields.items():
        a = [get(item, path) for item in ordered if item["arm"] == "A"]
        b = [get(item, path) for item in ordered if item["arm"] == "B"]
        deltas = [right - left for left, right in zip(a, b)]
        metrics[name] = {
            "A": {"values": a, **mean_sd(a)},
            "B": {"values": b, **mean_sd(b)},
            "B_minus_A": {"values": deltas, **mean_sd(deltas)},
        }
    return {"seeds": [123, 124, 125], "metrics": metrics}


@torch.inference_mode()
def generate_and_encode_videos(
    model: VideoCaptionModel,
    raw_features: torch.Tensor,
    video_ids: list[str],
    batch_size: int,
    device: str,
) -> tuple[list[dict[str, str]], torch.Tensor, dict[str, Any]]:
    model.eval()
    predictions: list[dict[str, str]] = []
    video_embeddings: list[torch.Tensor] = []
    replay_captions: list[str] | None = None
    for start in range(0, len(video_ids), batch_size):
        stop = min(start + batch_size, len(video_ids))
        features = raw_features[start:stop].to(device)
        video_embedding = model.proj(model.encoder.proj(features))
        captions = model.decoder.generate(video_embedding, **DECODE)
        if start == 0:
            replay = model.decoder.generate(video_embedding, **DECODE)
            if replay != captions:
                raise RuntimeError("First test inference batch did not reproduce exactly")
            replay_captions = replay
        prefix = model.decoder.build_visual_prefix(video_embedding)
        video_embeddings.append(F.normalize(prefix.mean(dim=1), dim=-1).cpu())
        predictions.extend(
            {"video_id": video_id, "caption": caption.strip()}
            for video_id, caption in zip(video_ids[start:stop], captions)
        )
        if stop % 80 == 0 or stop == len(video_ids):
            print(f"[test:generate] {stop}/{len(video_ids)}", flush=True)
    if len(predictions) != len(video_ids) or any(not item["caption"] for item in predictions):
        raise RuntimeError("Test generation did not produce 670 nonempty ordered captions")
    if [item["video_id"] for item in predictions] != video_ids:
        raise RuntimeError("Test prediction order changed")
    return predictions, torch.cat(video_embeddings), {
        "first_batch_replay_exact": True,
        "first_batch_captions": replay_captions,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    if args.output_root.exists():
        raise FileExistsError("Refusing to overwrite an existing experiment directory")
    if args.batch_size < 1:
        raise ValueError("Batch size must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("The locked test run requires the local CUDA GPU")
    if sha256_file(TEST) != TEST_SHA256:
        raise RuntimeError("Pinned test670 annotations changed")
    for spec in CHECKPOINTS:
        if not spec["path"].is_file() or sha256_file(spec["path"]) != spec["sha256"]:
            raise RuntimeError(f"Registered checkpoint changed or is missing: {spec['path']}")
    source_hashes = {name: sha256_file(ROOT / name) for name in SOURCE_FILES}
    checkpoint_audits = []
    model_args = None
    for spec in CHECKPOINTS:
        payload = torch.load(spec["path"], map_location="cpu", weights_only=True)
        audit = validate_checkpoint_payload(payload, spec)
        if model_args is None:
            model_args = audit["args"]
        elif audit["args"] != model_args:
            raise RuntimeError("Registered checkpoints do not share model arguments")
        checkpoint_audits.append(
            {"seed": spec["seed"], "arm": spec["arm"], "path": str(spec["path"].resolve()),
             "sha256": spec["sha256"], **audit}
        )
        del payload
        gc.collect()
    assert model_args is not None
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model = VideoCaptionModel(
        gpt2_name=model_args["gpt2_name"],
        vit_name=model_args["vit_name"],
        cond_mode=model_args["cond_mode"],
        prefix_len=model_args["prefix_len"],
        freeze_vit=True,
        vit_enable_torch_compile=False,
    )
    frozen_hashes = None
    for spec in CHECKPOINTS:
        payload = load_strict(model, spec["path"])
        hashes = component_hashes(model)
        current = {name: hashes[name] for name in ("decoder", "backbone")}
        if frozen_hashes is None:
            frozen_hashes = current
        elif current != frozen_hashes:
            raise RuntimeError("Frozen GPT-2 or ViT differs across registered checkpoints")
        del payload
        gc.collect()
    args.output_root.mkdir(parents=True)
    records = json.loads(TEST.read_text(encoding="utf-8"))
    video_ids = [str(record["video_id"]) for record in records]
    reference_count = sum(len(record["captions"]) for record in records)
    if len(records) != 670 or len(set(video_ids)) != 670 or reference_count != 27763:
        raise RuntimeError("Unexpected test670 records or reference count")
    load_strict(model, CHECKPOINTS[0]["path"])
    model = model.to("cuda").eval()
    loader = build_dataloader(
        str(TEST), model.decoder.tokenizer, batch_size=args.batch_size, max_len=64,
        num_frame=8, image_size=224, shuffle=False, num_wokers=0, random_caption=False,
    )
    if [str(record["video_id"]) for record in loader.dataset.anns] != video_ids:
        raise RuntimeError("Test frame validation dropped or reordered a video")
    raw_features, cache_audit = cache_split(
        model, loader, video_ids, len(records), "test", "cuda"
    )
    cache_path = args.output_root / "test_features.pt"
    torch.save({"features": raw_features, "video_ids": video_ids, "audit": cache_audit}, cache_path)
    all_captions = [caption for record in records for caption in record["captions"]]
    caption_gold = torch.tensor(
        [index for index, record in enumerate(records) for _ in record["captions"]]
    )
    text_embeddings = encode_texts(model, all_captions, 64, 128, "cuda")
    text_hash = tensor_sha(text_embeddings)
    preflight = {
        "protocol": "frozen-three-seed-msvd-test670-v1",
        "git_head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "source_sha256": source_hashes,
        "test": {"path": str(TEST.resolve()), "sha256": TEST_SHA256,
                 "videos": len(records), "references": reference_count,
                 "test_split_read": True},
        "checkpoints": checkpoint_audits,
        "frozen_component_hashes": frozen_hashes,
        "decode": DECODE,
        "cache": {**cache_audit, "path": str(cache_path.resolve()),
                  "file_sha256": sha256_file(cache_path)},
        "text_embeddings": {"count": len(text_embeddings), "sha256": text_hash},
        "gpu": torch.cuda.get_device_name(),
        "torch_version": str(torch.__version__),
        "test_tuning_performed": False,
        "paper_ready": False,
    }
    write_json(args.output_root / "preflight.json", preflight)
    results: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for spec in CHECKPOINTS:
        label = f"seed{spec['seed']}_{spec['arm']}"
        print(f"[test:model] {label}", flush=True)
        started = time.monotonic()
        torch.cuda.reset_peak_memory_stats()
        try:
            payload = load_strict(model, spec["path"])
            hashes = component_hashes(model)
            predictions, video_embeddings, replay = generate_and_encode_videos(
                model, raw_features, video_ids, args.batch_size, "cuda"
            )
            prediction_path = args.output_root / f"predictions_{label}.json"
            write_json(prediction_path, predictions)
            caption_metrics = compute_caption_metrics(records, predictions)
            retrieval = evaluate_scores(video_embeddings, text_embeddings, caption_gold, seed=123)
            result = {
                "seed": spec["seed"],
                "arm": spec["arm"],
                "checkpoint": str(spec["path"].resolve()),
                "checkpoint_sha256": spec["sha256"],
                "checkpoint_progress": payload["progress"],
                "component_hashes": hashes,
                "decode": DECODE,
                "prediction_file": str(prediction_path.resolve()),
                "prediction_sha256": sha256_file(prediction_path),
                "prediction_count": len(predictions),
                "prediction_replay": replay,
                "caption_metrics": caption_metrics,
                "retrieval": retrieval,
                "elapsed_seconds": time.monotonic() - started,
                "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
                "peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
                "test_split_read": True,
                "test_tuning_performed": False,
                "paper_ready": False,
            }
            result_path = args.output_root / f"result_{label}.json"
            write_json(result_path, result)
            results.append(result)
            compact = caption_metrics["corpus_x100"]
            print(json.dumps({"model": label, "Bleu_4": compact["Bleu_4"],
                              "METEOR": compact["METEOR"], "ROUGE_L": compact["ROUGE_L"],
                              "CIDEr": compact["CIDEr"],
                              "T2V_R1": retrieval["true_labels"]["text_to_video"]["r_at_1"]}),
                  flush=True)
            del payload, predictions, video_embeddings
        except Exception as error:
            failure = {"seed": spec["seed"], "arm": spec["arm"],
                       "type": type(error).__name__, "message": str(error)}
            failures.append(failure)
            write_json(args.output_root / f"failure_{label}.json", failure)
            print(json.dumps({"failure": failure}), flush=True)
        gc.collect()
        torch.cuda.empty_cache()
    complete = len(results) == len(CHECKPOINTS) and not failures
    summary = summarize_results(results) if complete else None
    if source_hashes != {name: sha256_file(ROOT / name) for name in SOURCE_FILES}:
        raise RuntimeError("Pinned evaluation source changed during the formal run")
    if sha256_file(TEST) != TEST_SHA256:
        raise RuntimeError("Pinned test annotations changed during the formal run")
    completion = {
        "completed": complete,
        "protocol": "frozen-three-seed-msvd-test670-v1",
        "results": [{"seed": item["seed"], "arm": item["arm"]} for item in results],
        "failures": failures,
        "summary": summary,
        "test_split_read": True,
        "test_tuning_performed": False,
        "paper_ready": False,
    }
    write_json(args.output_root / "completion.json", completion)
    return 0 if complete else 1


if __name__ == "__main__":
    raise SystemExit(main())
