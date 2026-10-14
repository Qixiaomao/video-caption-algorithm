"""Run the once-only fixed MSR-VTT test2990 evaluation after validation freeze."""
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
import statistics
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
DATA = ROOT / "runs/msrvtt_data_full_20260928"

import torch
import torch.nn.functional as F

from scripts.caption_metrics import compute_caption_metrics
from scripts.eval_retrieval import encode_texts
from scripts.msrvtt_blockwise_retrieval import evaluate_blockwise
from scripts.run_caption_policy_ab import raw_backbone_features
from scripts.run_full_msvd_infonce_ablation import load_strict
from scripts.run_overfit16 import component_hashes, state_sha, tensor_sha
from src.data.data_loader import build_dataloader
from src.models.caption_model import VideoCaptionModel
from src.training.checkpoint import sha256_file


def write_new(path: Path, value: object) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)


def _decode_kwargs(test_config: dict) -> dict:
    registered = test_config["caption_generation"]
    if (registered["prompt"] != "" or registered["num_beams"] != 1
            or registered["do_sample"] is not False or registered["max_new_tokens"] != 24
            or registered["min_new_tokens"] != 0 or registered["temperature"] != 1.0
            or registered["no_repeat_ngram_size"] != 3
            or registered["pad_token_id"] != 50256 or registered["eos_token_id"] != 50256):
        raise RuntimeError("Frozen test decode changed")
    return {key: registered[key] for key in ("prompt", "max_new_tokens", "min_new_tokens",
                                             "num_beams", "temperature", "top_p",
                                             "no_repeat_ngram_size", "repetition_penalty")}


@torch.inference_mode()
def build_test_features(model: VideoCaptionModel, annotations: Path, ids: list[str], target: Path) -> tuple[torch.Tensor, dict]:
    loader = build_dataloader(str(annotations), model.decoder.tokenizer,
                              batch_size=8, max_len=96, num_frame=8, image_size=224,
                              shuffle=False, random_caption=False)
    if [row["video_id"] for row in loader.dataset.anns] != ids:
        raise RuntimeError("Test dataloader silently dropped/reordered IDs")
    model.eval()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    values = []
    seen = []
    errors = []
    for batch_number, batch in enumerate(loader, 1):
        video = batch["video"].cuda(non_blocking=True)
        direct = model.encoder(video)
        raw = raw_backbone_features(model.encoder, video)
        error = float((direct - model.encoder.proj(raw)).abs().max())
        if error > 1e-6 or not torch.isfinite(raw).all():
            raise RuntimeError(f"Test feature equivalence failed at batch {batch_number}: {error}")
        errors.append(error)
        seen.extend(batch["video_id"])
        values.append(raw.cpu())
        if batch_number == 1 or batch_number % 50 == 0 or len(seen) == len(ids):
            print(json.dumps({"phase": "test_feature_integrity", "cached": len(seen),
                              "expected": len(ids), "peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20}), flush=True)
    features = torch.cat(values)
    if seen != ids or tuple(features.shape) != (2990, 768):
        raise RuntimeError("Test feature cache coverage/shape mismatch")
    with target.open("xb") as stream:
        torch.save({"video_ids": seen, "raw_features": features,
                    "source_annotation_sha256": sha256_file(annotations)}, stream)
    audit = {"count": len(seen), "shape": list(features.shape),
             "cache_path": str(target), "cache_sha256": sha256_file(target),
             "raw_feature_tensor_sha256": tensor_sha(features),
             "max_abs_direct_cache_error": max(errors),
             "batch_count": len(errors), "elapsed_seconds": time.perf_counter() - started,
             "peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20}
    return features, audit


@torch.inference_mode()
def generate_and_embed(model: VideoCaptionModel, features: torch.Tensor,
                       ids: list[str], decode: dict) -> tuple[list[dict], torch.Tensor, bool]:
    model.eval()
    predictions = []
    embeddings = []
    first_replay = False
    for start in range(0, len(ids), 8):
        stop = min(start + 8, len(ids))
        video_embedding = model.proj(model.encoder.proj(features[start:stop].cuda()))
        captions = model.decoder.generate(video_embedding, **decode)
        if start == 0:
            first_replay = model.decoder.generate(video_embedding, **decode) == captions
            if not first_replay:
                raise RuntimeError("First test generation batch not deterministic")
        prefix = model.decoder.build_visual_prefix(video_embedding)
        embeddings.append(F.normalize(prefix.mean(dim=1), dim=-1).cpu())
        predictions.extend({"video_id": vid, "caption": caption.strip()}
                           for vid, caption in zip(ids[start:stop], captions))
        if stop % 80 == 0 or stop == len(ids):
            print(json.dumps({"phase": "test_generation", "generated": stop, "expected": len(ids)}), flush=True)
    if len(predictions) != len(ids) or [row["video_id"] for row in predictions] != ids:
        raise RuntimeError("Test prediction coverage/order mismatch")
    return predictions, torch.cat(embeddings), first_replay


def summarize(results: list[dict]) -> dict:
    if len(results) != 6 or {(row["seed"], row["arm"]) for row in results} != {(seed, arm) for seed in (123, 124, 125) for arm in ("A", "B")}:
        raise RuntimeError("Summary requires all six arms")
    paths = {"T2V_R1": ("retrieval", "true_labels", "text_to_video", "r_at_1"),
             "T2V_R5": ("retrieval", "true_labels", "text_to_video", "r_at_5"),
             "T2V_R10": ("retrieval", "true_labels", "text_to_video", "r_at_10"),
             "V2T_R1": ("retrieval", "true_labels", "video_to_text", "r_at_1"),
             "V2T_R5": ("retrieval", "true_labels", "video_to_text", "r_at_5"),
             "V2T_R10": ("retrieval", "true_labels", "video_to_text", "r_at_10")}
    if all(row["caption_metrics_status"] == "complete" for row in results):
        for name in ("Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4", "METEOR", "ROUGE_L", "CIDEr"):
            paths[name] = ("caption_metrics", "corpus_x100", name)
    metrics = {}
    ordered = sorted(results, key=lambda row: (row["seed"], row["arm"]))
    for name, path in paths.items():
        def value(row):
            result = row
            for key in path:
                result = result[key]
            return float(result)
        a = [value(row) for row in ordered if row["arm"] == "A"]
        b = [value(row) for row in ordered if row["arm"] == "B"]
        delta = [right - left for left, right in zip(a, b)]
        metrics[name] = {"A": {"values": a, "mean": statistics.mean(a), "sample_sd": statistics.stdev(a)},
                         "B": {"values": b, "mean": statistics.mean(b), "sample_sd": statistics.stdev(b)},
                         "B_minus_A": {"values": delta, "mean": statistics.mean(delta),
                                       "sample_sd": statistics.stdev(delta)}}
    return {"seeds": [123, 124, 125], "recall_raw_scale": "fraction",
            "recall_paper_display_scale": "multiply by 100", "caption_paper_scale": "raw times 100",
            "caption_metrics_complete": all(row["caption_metrics_status"] == "complete" for row in results),
            "metrics": metrics}


def main(config_path: Path, training_root: Path, out: Path) -> None:
    if out.exists():
        raise FileExistsError(out)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    test_config_path = Path(config["test_evaluation_config_path"])
    if config["status"] != "FROZEN" or sha256_file(test_config_path) != config["test_evaluation_config_sha256"]:
        raise RuntimeError("Frozen test configuration changed")
    test_config = json.loads(test_config_path.read_text(encoding="utf-8"))
    if test_config["status"] != "FROZEN_BEFORE_FORMAL_TRAINING":
        raise RuntimeError("Test protocol was not frozen before formal training")
    decode = _decode_kwargs(test_config)
    freeze_path = training_root / "validation_freeze.json"
    completion_path = training_root / "completion.json"
    if not freeze_path.is_file() or not completion_path.is_file():
        raise RuntimeError("All six training arms and validation decisions must finish before test")
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    if (not freeze["six_arms_complete"] or not freeze["test_not_read"]
            or not completion["technical_training_complete"]
            or freeze["test_evaluation_config_sha256"] != sha256_file(test_config_path)
            or freeze["fixed_endpoint_epoch"] != config["epochs"]):
        raise RuntimeError("Validation freeze/checkpoint endpoint contract failed")
    if json.loads((DATA / "data_gate.json").read_text(encoding="utf-8"))["status"] != "PASS":
        raise RuntimeError("Data gate changed")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    checkpoint_specs = []
    for seed in (123, 124, 125):
        for arm in ("A", "B"):
            arm_dir = training_root / f"seed{seed}" / arm
            result = json.loads((arm_dir / "result.json").read_text(encoding="utf-8"))
            if result["steps"] != config["steps_per_arm"] or result["epochs"] != config["epochs"]:
                raise RuntimeError("Formal arm endpoint mismatch")
            final = result["evaluations"][-1]
            checkpoint = arm_dir / f"checkpoint_epoch{config['epochs']:03d}.pt"
            if final["checkpoint_sha256"] != sha256_file(checkpoint):
                raise RuntimeError("Formal checkpoint hash changed")
            checkpoint_specs.append({"seed": seed, "arm": arm, "path": checkpoint,
                                     "sha256": final["checkpoint_sha256"]})
    out.mkdir(parents=True)
    manifest = json.loads((DATA / "manifest.json").read_text(encoding="utf-8"))
    selected = [row for row in manifest if row["split"] == "test"]
    ids = [row["video_id"] for row in selected]
    pinned_ids = (DATA / "split_lists/test_list_full.txt").read_text(encoding="utf-8-sig").splitlines()
    if ids != pinned_ids or len(ids) != 2990 or sum(row["caption_count"] for row in selected) != 59794:
        raise RuntimeError("Test split/annotation count changed")
    records = [{"video_id": row["video_id"], "video": row["media_path"],
                "frames_dir": row["frames_dir"],
                "captions": [ref["caption"] for ref in row["captions"]]}
               for row in selected]
    annotations = out / "test_annotations.json"
    write_new(annotations, records)
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = True  # Match the frozen train/val ViT cache.
    model = VideoCaptionModel(gpt2_name=config["gpt2_source_path"], freeze_vit=True,
                              vit_enable_torch_compile=False).cuda().eval()
    cache_audit = json.loads((DATA / "feature_cache/cache_audit.json").read_text(encoding="utf-8"))
    if state_sha(model.encoder.backbone.state_dict()) != cache_audit["backbone_state_sha256"]:
        raise RuntimeError("Test frozen ViT differs from train/val source")
    load_strict(model, checkpoint_specs[0]["path"])
    test_features, feature_audit = build_test_features(model, annotations, ids, out / "test_raw_vit.pt")
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
    gold = torch.tensor([index for index, row in enumerate(records) for _ in row["captions"]], dtype=torch.long)
    write_new(out / "preflight.json", {"at_utc": datetime.now(timezone.utc).isoformat(),
                                       "config_sha256": sha256_file(config_path),
                                       "test_config_sha256": sha256_file(test_config_path),
                                       "source_sha256": {name: sha256_file(ROOT / name) for name in (
                                           "scripts/run_msrvtt_test2990_20260929.py",
                                           "scripts/msrvtt_blockwise_retrieval.py",
                                           "scripts/caption_metrics.py",
                                           "scripts/eval_retrieval.py",
                                           "src/models/caption_model.py",
                                           "src/models/text_decoder.py")},
                                       "validation_freeze_sha256": sha256_file(freeze_path),
                                       "test_annotations_sha256": sha256_file(annotations),
                                       "test_features": feature_audit,
                                       "text_embedding_count": len(text_embeddings),
                                       "text_embedding_tensor_sha256": tensor_sha(text_embeddings),
                                       "checkpoints": [{**spec, "path": str(spec["path"])} for spec in checkpoint_specs],
                                       "test_selection_performed": False})
    results = []
    failures = []
    frozen_hashes = None
    for spec in checkpoint_specs:
        label = f"seed{spec['seed']}_{spec['arm']}"
        started = time.perf_counter()
        torch.cuda.reset_peak_memory_stats()
        try:
            checkpoint = load_strict(model, spec["path"])
            if checkpoint["progress"]["step"] != config["steps_per_arm"] or checkpoint["progress"]["epoch"] != config["epochs"]:
                raise RuntimeError("Checkpoint endpoint changed")
            hashes = component_hashes(model)
            frozen = {key: hashes[key] for key in ("decoder", "backbone")}
            if frozen_hashes is None:
                frozen_hashes = frozen
            elif frozen != frozen_hashes:
                raise RuntimeError("Frozen model differs across six arms")
            predictions, video_embeddings, replay = generate_and_embed(model, test_features, ids, decode)
            prediction_path = out / f"predictions_{label}.json"
            write_new(prediction_path, predictions)
            empty = [row["video_id"] for row in predictions if not row["caption"]]
            retrieval = evaluate_blockwise(video_embeddings, text_embeddings, gold,
                                           device="cuda", text_batch_size=256,
                                           video_batch_size=16, derangement_seed=123)
            result = {"seed": spec["seed"], "arm": spec["arm"],
                      "checkpoint": str(spec["path"]), "checkpoint_sha256": spec["sha256"],
                      "checkpoint_progress": checkpoint["progress"],
                      "component_hashes": hashes, "decode": decode,
                      "prediction_file": str(prediction_path),
                      "prediction_sha256": sha256_file(prediction_path),
                      "prediction_count": len(predictions), "empty_prediction_ids": empty,
                      "first_batch_replay_exact": replay,
                      "caption_metrics_status": "pending_cpu_phase",
                      "caption_metrics": None, "retrieval": retrieval,
                      "elapsed_seconds": time.perf_counter() - started,
                      "peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
                      "test_selection_performed": False}
            write_new(out / f"gpu_result_{label}.json", result)
            results.append(result)
            print(json.dumps({"model": label, "predictions": len(predictions),
                              "caption_metrics_status": "pending_cpu_phase",
                              "t2v_r1_fraction": retrieval["true_labels"]["text_to_video"]["r_at_1"]}), flush=True)
        except Exception:
            failure = {"seed": spec["seed"], "arm": spec["arm"],
                       "traceback": traceback.format_exc()}
            failures.append(failure)
            write_new(out / f"failure_{label}.json", failure)
            print(json.dumps({"failure": label}), flush=True)
        gc.collect()
        torch.cuda.empty_cache()
    del model, test_features, text_embeddings, gold
    gc.collect()
    torch.cuda.empty_cache()
    for result in results:
        label = f"seed{result['seed']}_{result['arm']}"
        try:
            if result["empty_prediction_ids"]:
                result["caption_metrics_status"] = "unavailable_empty_prediction"
            else:
                predictions = json.loads(Path(result["prediction_file"]).read_text(encoding="utf-8"))
                result["caption_metrics"] = compute_caption_metrics(records, predictions)
                result["caption_metrics_status"] = "complete"
            write_new(out / f"result_{label}.json", result)
        except Exception:
            result["caption_metrics_status"] = "failed_cpu_metric_phase"
            failure = {"seed": result["seed"], "arm": result["arm"],
                       "phase": "caption_metrics", "traceback": traceback.format_exc()}
            failures.append(failure)
            write_new(out / f"failure_caption_metrics_{label}.json", failure)
            write_new(out / f"result_{label}.json", result)
    complete = len(results) == 6 and not failures
    summary = summarize(results) if complete else None
    write_new(out / "completion.json", {"at_utc": datetime.now(timezone.utc).isoformat(),
                                        "technical_test_run_complete": complete,
                                        "caption_metrics_complete": complete and summary["caption_metrics_complete"],
                                        "results": [{"seed": row["seed"], "arm": row["arm"]} for row in results],
                                        "failures": failures, "summary": summary,
                                        "test_selection_performed": False})
    if not complete:
        raise RuntimeError("One or more frozen test evaluations failed; see preserved failure files")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--training-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    try:
        main(args.config, args.training_root, args.output)
    except Exception:
        if args.output.is_dir() and not (args.output / "failure_preflight_or_completion.json").exists():
            write_new(args.output / "failure_preflight_or_completion.json",
                      {"at_utc": datetime.now(timezone.utc).isoformat(),
                       "traceback": traceback.format_exc()})
        raise
