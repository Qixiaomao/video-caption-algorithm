"""Five-epoch paired train/val pilot for budget setting; no test or formal result."""
from __future__ import annotations

import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
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

from scripts.run_caption_policy_ab import cached_caption_loss, reference_seed, tokenize_caption
from scripts.run_full_msvd_infonce_ablation import cached_caption_infonce_objective
from scripts.run_overfit16 import component_hashes, configure_trainable, tensor_sha
from src.cli.train_caption_mapper import set_seed
from src.models.caption_model import VideoCaptionModel
from src.training.checkpoint import sha256_file
from src.training.contrastive import exact_caption_positive_mask


def write_new(path: Path, value: object) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)


def schedule_batches(count: int, epochs: int) -> list[list[list[int]]]:
    if count != 6513:
        raise RuntimeError("Pinned train count changed")
    generator = torch.Generator().manual_seed(456)
    answer = []
    for _ in range(epochs):
        permutation = torch.randperm(count, generator=generator).tolist()
        batches = [permutation[i:i + 8] for i in range(0, 6504, 8)]
        batches.append(permutation[6504:])
        if len(batches) != 814 or len(batches[-1]) != 9 or sorted(x for batch in batches for x in batch) != list(range(count)):
            raise RuntimeError("Epoch is not a bijection over train IDs")
        answer.append(batches)
    return answer


@torch.inference_mode()
def validation_ce(model, records, features, tokenizer) -> float:
    model.eval()
    values = []
    for start in range(0, len(records), 8):
        batch = records[start:start + 8]
        captions = torch.stack([tokenize_caption(tokenizer, row["captions"][0], max_len=96)
                                for row in batch]).cuda()
        loss = cached_caption_loss(model, features[start:start + len(batch)].cuda(), captions)
        values.append((float(loss), len(batch)))
    return sum(value * count for value, count in values) / sum(count for _, count in values)


def main(out: Path) -> None:
    if out.exists():
        raise FileExistsError(out)
    if json.loads((DATA / "data_gate.json").read_text(encoding="utf-8"))["status"] != "PASS":
        raise RuntimeError("Data gate has not passed")
    out.mkdir(parents=True)
    started = time.perf_counter()
    train = json.loads((DATA / "annotations/train.json").read_text(encoding="utf-8"))
    val = json.loads((DATA / "annotations/val.json").read_text(encoding="utf-8"))
    train_cache = torch.load(DATA / "feature_cache/train_raw_vit.pt", map_location="cpu", weights_only=True)
    val_cache = torch.load(DATA / "feature_cache/val_raw_vit.pt", map_location="cpu", weights_only=True)
    if train_cache["video_ids"] != [row["video_id"] for row in train] or val_cache["video_ids"] != [row["video_id"] for row in val]:
        raise RuntimeError("Cache and annotations disagree")
    train_features = train_cache["raw_features"]
    val_features = val_cache["raw_features"]
    epochs = 5
    batches = schedule_batches(len(train), epochs)
    orders = {row["video_id"]: torch.randperm(len(row["captions"]), generator=torch.Generator().manual_seed(reference_seed(row["video_id"]))).tolist()
              for row in train}
    write_new(out / "pilot_schedule.json", {"epochs": epochs, "batch_seed": 456,
                                             "batches": batches, "caption_orders": orders})
    source = json.loads((ROOT / "runs/full_msvd_balanced_20260905/preflight.json").read_text(encoding="utf-8"))["model_source"]["path"]
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = True  # Same frozen ViT precision as the data cache.
    set_seed(123)
    model = VideoCaptionModel(gpt2_name=source, freeze_vit=True, vit_enable_torch_compile=False).cuda()
    initial = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    initial_hashes = component_hashes(model)
    tokenizer = model.decoder.tokenizer
    results = {}
    expected_trace = None
    torch.cuda.reset_peak_memory_stats()
    for arm in ("A", "B"):
        model.load_state_dict(initial, strict=True)
        if component_hashes(model) != initial_hashes:
            raise RuntimeError("Paired initial state mismatch")
        params = configure_trainable(model)
        optimizer = torch.optim.AdamW([parameter for _, parameter in params], lr=3e-4, weight_decay=0.01)
        set_seed(123)
        trace_digest = hashlib.sha256()
        train_curve = []
        val_curve = [{"epoch": 0, "first_reference_ce": validation_ce(model, val, val_features, tokenizer)}]
        step_times = []
        step = 0
        for epoch, epoch_batches in enumerate(batches, 1):
            losses = []
            references_seen = Counter()
            for indices in epoch_batches:
                step += 1
                tick = time.perf_counter()
                texts = []
                for index in indices:
                    row = train[index]
                    order = orders[row["video_id"]]
                    ref_index = order[(epoch - 1) % len(order)]
                    texts.append(row["captions"][ref_index])
                    references_seen[row["video_id"]] += 1
                captions = torch.stack([tokenize_caption(tokenizer, text, max_len=96) for text in texts])
                features = train_features[indices]
                digest_line = json.dumps({"step": step, "indices": indices, "texts": texts,
                                          "captions_sha256": tensor_sha(captions),
                                          "features_sha256": tensor_sha(features)},
                                         ensure_ascii=False, sort_keys=True).encode()
                trace_digest.update(digest_line)
                mask = exact_caption_positive_mask(captions, tokenizer.pad_token_id)
                if not torch.all(mask.any(dim=1)) or not torch.all((~mask).any(dim=1)):
                    raise RuntimeError(f"No legal InfoNCE negative at pilot step {step}")
                model.train()
                model.encoder.backbone.eval()
                optimizer.zero_grad(set_to_none=True)
                if arm == "B":
                    result = cached_caption_infonce_objective(model, features.cuda(), captions.cuda())
                else:
                    ce = cached_caption_loss(model, features.cuda(), captions.cuda())
                    result = {"loss": ce, "caption_loss": ce, "alignment_loss": ce.new_zeros(())}
                if any(not torch.isfinite(result[key]) for key in ("loss", "caption_loss", "alignment_loss")):
                    raise RuntimeError(f"Nonfinite pilot objective {arm} step {step}")
                result["loss"].backward()
                if any(parameter.grad is None or not torch.isfinite(parameter.grad).all() for _, parameter in params):
                    raise RuntimeError(f"Missing/nonfinite pilot gradient {arm} step {step}")
                optimizer.step()
                losses.append(float(result["caption_loss"].detach()))
                step_times.append(time.perf_counter() - tick)
                if step % 100 == 0:
                    print(json.dumps({"arm": arm, "step": step, "epoch": epoch,
                                      "caption_ce": losses[-1],
                                      "peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20}), flush=True)
            if len(references_seen) != 6513 or set(references_seen.values()) != {1}:
                raise RuntimeError("Pilot epoch did not visit each train video once")
            train_curve.append({"epoch": epoch, "mean_caption_ce": sum(losses) / len(losses),
                                "last_100_mean_caption_ce": sum(losses[-100:]) / min(100, len(losses))})
            if epoch in (1, 3, 5):
                val_curve.append({"epoch": epoch, "first_reference_ce": validation_ce(model, val, val_features, tokenizer)})
            print(json.dumps({"arm": arm, "epoch": epoch, "train_ce": train_curve[-1]["mean_caption_ce"],
                              "val_ce": val_curve[-1]["first_reference_ce"]}), flush=True)
        final_hashes = component_hashes(model)
        if any(initial_hashes[key] != final_hashes[key] for key in ("decoder", "backbone")):
            raise RuntimeError("Frozen model changed in pilot")
        if any(initial_hashes[key] == final_hashes[key] for key in ("projection", "mapper")):
            raise RuntimeError("Trainable module unchanged in pilot")
        if expected_trace is not None and trace_digest.hexdigest() != expected_trace:
            raise RuntimeError("A/B pilot trace mismatch")
        expected_trace = trace_digest.hexdigest()
        results[arm] = {"steps": step, "trace_sha256": expected_trace,
                        "train_curve": train_curve, "val_curve": val_curve,
                        "mean_step_seconds": sum(step_times) / len(step_times),
                        "last_100_mean_step_seconds": sum(step_times[-100:]) / 100,
                        "frozen_hashes_unchanged": True}
        del optimizer
        torch.cuda.empty_cache()
    write_new(out / "result.json", {"status": "PASS", "at_utc": datetime.now(timezone.utc).isoformat(),
                                    "scope": "budget pilot only; not paper-ready; no test use",
                                    "source_model": source, "script_sha256": sha256_file(Path(__file__)),
                                    "data_gate_sha256": sha256_file(DATA / "data_gate.json"),
                                    "annotation_sha256": {split: sha256_file(DATA / "annotations" / f"{split}.json")
                                                          for split in ("train", "val")},
                                    "schedule_sha256": sha256_file(out / "pilot_schedule.json"),
                                    "max_len": 96, "peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
                                    "wall_seconds": time.perf_counter() - started,
                                    "paired_trace_equal": True, "arms": results,
                                    "test_split_read": False})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    try:
        main(args.output)
    except Exception:
        if args.output.is_dir() and not (args.output / "failure.json").exists():
            write_new(args.output / "failure.json", {"at_utc": datetime.now(timezone.utc).isoformat(),
                                                    "traceback": traceback.format_exc()})
        raise
