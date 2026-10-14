"""Independent MSR-VTT 16-video paired A/B engineering smoke; no test access."""
from __future__ import annotations

import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import argparse
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

from scripts.run_caption_policy_ab import cached_caption_loss, raw_backbone_features, tokenize_caption
from scripts.run_full_msvd_infonce_ablation import cached_caption_infonce_objective
from scripts.run_overfit16 import component_hashes, configure_trainable, normalize_caption, state_sha, tensor_sha
from src.cli.train_caption_mapper import set_seed
from src.data.data_loader import build_dataloader
from src.models.caption_model import VideoCaptionModel, build_caption_supervision
from src.training.checkpoint import (
    restore_model_state, restore_training_state, save_training_checkpoint, sha256_file,
)
from src.training.contrastive import exact_caption_positive_mask


def write_new(path: Path, value: object) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)


def trace(step: int, indices: list[int], ids: list[str], captions: torch.Tensor,
          texts: list[str], features: torch.Tensor) -> dict:
    return {"step": step, "indices": indices, "video_ids": [ids[i] for i in indices],
            "target_texts": texts, "caption_ids_sha256": tensor_sha(captions),
            "visual_features_sha256": tensor_sha(features),
            "torch_cpu_rng_sha256": hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest(),
            "torch_cuda_rng_sha256": hashlib.sha256(torch.cuda.get_rng_state().cpu().numpy().tobytes()).hexdigest()}


def objective(model, features, captions, arm):
    if arm == "B":
        return cached_caption_infonce_objective(model, features, captions)
    ce = cached_caption_loss(model, features, captions)
    return {"loss": ce, "caption_loss": ce, "alignment_loss": ce.new_zeros(())}


@torch.inference_mode()
def fixed_subset_ce(model, features: torch.Tensor, captions: torch.Tensor) -> float:
    model.eval()
    values = [float(cached_caption_loss(model, features[i:i + 8].cuda(),
                                        captions[i:i + 8].cuda())) for i in (0, 8)]
    return sum(values) / len(values)


def main(out: Path) -> None:
    if out.exists():
        raise FileExistsError(out)
    media_audit = json.loads((DATA / "data_audit.json").read_text(encoding="utf-8"))
    cache_audit = json.loads((DATA / "feature_cache/cache_audit.json").read_text(encoding="utf-8"))
    gate = json.loads((DATA / "data_gate.json").read_text(encoding="utf-8"))
    if media_audit["status"] != "PASS" or cache_audit["status"] != "PASS" or gate["status"] != "PASS":
        raise RuntimeError("Data gate not passed")
    out.mkdir(parents=True)
    started = time.perf_counter()
    records = json.loads((DATA / "annotations/train.json").read_text(encoding="utf-8"))
    cached = torch.load(DATA / "feature_cache/train_raw_vit.pt", map_location="cpu", weights_only=True)
    if cached["video_ids"] != [row["video_id"] for row in records]:
        raise RuntimeError("Cache/annotation ID mismatch")
    chosen = []
    distinct = set()
    for index, row in enumerate(records):
        text = row["captions"][0]
        normalized = normalize_caption(text)
        if normalized and normalized not in distinct:
            chosen.append(index)
            distinct.add(normalized)
        if len(chosen) == 16:
            break
    if len(chosen) != 16:
        raise RuntimeError("Could not select 16 unique nonempty captions")
    features = cached["raw_features"][chosen]
    chosen_records = [records[i] for i in chosen]
    ids = [row["video_id"] for row in chosen_records]
    texts = [row["captions"][0] for row in chosen_records]
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    # The frozen feature cache was built with PyTorch's default cuDNN TF32 setting.
    torch.backends.cudnn.allow_tf32 = True
    set_seed(123)
    source = json.loads((ROOT / "runs/full_msvd_balanced_20260905/preflight.json").read_text(encoding="utf-8"))["model_source"]["path"]
    model = VideoCaptionModel(gpt2_name=source, freeze_vit=True, vit_enable_torch_compile=False)
    model.cuda().eval()
    if state_sha(model.encoder.backbone.state_dict()) != cache_audit["backbone_state_sha256"]:
        raise RuntimeError("Smoke ViT backbone differs from feature-cache source")
    probe_loader = build_dataloader(str(DATA / "annotations/train.json"), model.decoder.tokenizer,
                                    batch_size=8, max_len=96, num_frame=8, image_size=224,
                                    shuffle=False, random_caption=False)
    if [row["video_id"] for row in probe_loader.dataset.anns] != [row["video_id"] for row in records]:
        raise RuntimeError("Smoke dataloader silently dropped train IDs")
    probe_batch = next(iter(probe_loader))
    with torch.inference_mode():
        live = raw_backbone_features(model.encoder, probe_batch["video"].cuda())
    live_cache_error = float((live.cpu() - cached["raw_features"][:8]).abs().max())
    if probe_batch["video_id"] != cached["video_ids"][:8] or live_cache_error > 1e-6:
        detail = {"expected_ids": cached["video_ids"][:8], "observed_ids": probe_batch["video_id"],
                  "max_abs_error": live_cache_error, "threshold": 1e-6}
        write_new(out / "live_cache_diagnostic.json", detail)
        raise RuntimeError(f"Smoke live/cached feature mismatch: {detail}")
    del probe_loader, probe_batch, live
    initial = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    initial_hashes = component_hashes(model)
    tokenizer = model.decoder.tokenizer
    tokenized = torch.stack([tokenize_caption(tokenizer, text, max_len=96) for text in texts])
    mask, labels = build_caption_supervision(tokenized, tokenizer.pad_token_id)
    lengths = [len(tokenizer.encode(text, add_special_tokens=False)) for text in texts]
    if min(lengths) < 1 or max(lengths) >= 96 or not torch.all(labels[:, 0] != -100):
        raise RuntimeError("First token or caption length supervision failure")
    for index, length in enumerate(lengths):
        if labels[index, length] != tokenizer.eos_token_id or torch.any(labels[index, length + 1:] != -100):
            raise RuntimeError("EOS supervision failure")
    positives = exact_caption_positive_mask(tokenized[:8], tokenizer.pad_token_id)
    if positives.shape != (8, 8) or not torch.all(positives.diag()) or not torch.all(positives.any(dim=1)) or not torch.all((~positives).any(dim=1)):
        raise RuntimeError("InfoNCE positive/negative mask invalid")
    # Two distinct 8-video batches per cycle, with identical order in A and B.
    generator = torch.Generator().manual_seed(456)
    schedule = [torch.randperm(16, generator=generator).reshape(2, 8).tolist() for _ in range(100)]
    write_new(out / "schedule.json", {"seed": 123, "batch_seed": 456, "batches": schedule,
                                       "video_ids": ids, "target_texts": texts})
    expected_trace = None
    results = {}
    torch.cuda.reset_peak_memory_stats()
    for arm in ("A", "B"):
        model.load_state_dict(initial, strict=True)
        if component_hashes(model) != initial_hashes:
            raise RuntimeError("A/B initial component mismatch")
        params = configure_trainable(model)
        optimizer = torch.optim.AdamW([parameter for _, parameter in params], lr=3e-4, weight_decay=0.01)
        set_seed(123)
        initial_fixed_ce = fixed_subset_ce(model, features, tokenized)
        if arm == "B" and initial_fixed_ce != results["A"]["initial_fixed_subset_ce"]:
            raise RuntimeError("A/B fixed-subset initial CE differs")
        arm_dir = out / arm
        arm_dir.mkdir()
        traces = []
        ce_values = []
        step_seconds = []
        for step in range(1, 201):
            tick = time.perf_counter()
            indices = schedule[(step - 1) // 2][(step - 1) % 2]
            batch_features = features[indices].cuda()
            batch_captions = tokenized[indices].cuda()
            item = trace(step, indices, ids, tokenized[indices], [texts[i] for i in indices], features[indices])
            if expected_trace is not None and item != expected_trace[step - 1]:
                raise RuntimeError(f"A/B paired trajectory mismatch at step {step}")
            traces.append(item)
            model.train()
            model.encoder.backbone.eval()
            optimizer.zero_grad(set_to_none=True)
            output = objective(model, batch_features, batch_captions, arm)
            if any(not torch.isfinite(output[key]) for key in ("loss", "caption_loss", "alignment_loss")):
                raise RuntimeError(f"Nonfinite objective at {arm} step {step}")
            if arm == "B" and step == 1:
                if float(output["caption_loss"].detach()) != results["A"]["first_step_ce"]:
                    raise RuntimeError("Initial A/B caption CE differs")
                output["alignment_loss"].backward(retain_graph=True)
                for module in (model.encoder.proj, model.decoder.mapper):
                    if not any(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0
                               for p in module.parameters()):
                        raise RuntimeError("InfoNCE-only gradient missing from trainable module")
                optimizer.zero_grad(set_to_none=True)
            output["loss"].backward()
            if any(parameter.grad is None or not torch.isfinite(parameter.grad).all() for _, parameter in params):
                raise RuntimeError(f"Missing/nonfinite gradient at {arm} step {step}")
            optimizer.step()
            ce_values.append(float(output["caption_loss"].detach()))
            step_seconds.append(time.perf_counter() - tick)
            if step == 100:
                checkpoint = arm_dir / "resume_step100.pt"
                training_contract = {"objective_version": 3, "caption_objective": "causal_lm_cross_entropy",
                                     "alignment_objective": "infonce" if arm == "B" else "none",
                                     "alignment_weight": 0.1 if arm == "B" else 0.0,
                                     "info_nce_temperature": 0.07, "false_negative_policy": "exact_caption",
                                     "cpr_weight": 0, "cpr_margin": None, "cpr_negative_policy": "none",
                                     "video_representation": "mean_visual_prefix_tokens_l2_normalized",
                                     "text_representation": "frozen_gpt2_wte_masked_mean_l2_normalized",
                                     "physical_batch_size": 8}
                dataset_contract = {"annotations": {"train": {"sha256": sha256_file(DATA / "annotations/train.json")}},
                                    "manifest": {"sha256": sha256_file(DATA / "manifest.json")}}
                save_training_checkpoint(checkpoint, model=model, optimizer=optimizer, scheduler=None,
                    progress={"step": step}, args={"seed": 123, "arm": arm, "max_len": 96},
                    training_contract=training_contract, dataset_contract=dataset_contract,
                    loader_state={"schedule_sha256": sha256_file(out / "schedule.json"), "next_step": 101})
            elif step == 101:
                uninterrupted_hashes = component_hashes(model)
                uninterrupted_loss = float(output["loss"].detach())
                checkpoint_payload = torch.load(arm_dir / "resume_step100.pt", map_location="cpu", weights_only=True)
                restore_model_state(model, checkpoint_payload)
                progress = restore_training_state(checkpoint_payload, optimizer)
                if progress["step"] != 100 or checkpoint_payload["loader_state"]["next_step"] != 101:
                    raise RuntimeError("Resume progress mismatch")
                model.train()
                model.encoder.backbone.eval()
                optimizer.zero_grad(set_to_none=True)
                replay = objective(model, batch_features, batch_captions, arm)
                replay["loss"].backward()
                optimizer.step()
                if float(replay["loss"].detach()) != uninterrupted_loss or component_hashes(model) != uninterrupted_hashes:
                    raise RuntimeError("Exact checkpoint replay mismatch")
            if step % 50 == 0:
                print(json.dumps({"arm": arm, "step": step, "caption_ce": ce_values[-1],
                                  "peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20}), flush=True)
        final_hashes = component_hashes(model)
        if any(final_hashes[key] != initial_hashes[key] for key in ("decoder", "backbone")):
            raise RuntimeError("Frozen component changed")
        if any(final_hashes[key] == initial_hashes[key] for key in ("projection", "mapper")):
            raise RuntimeError("Trainable component did not change")
        final_fixed_ce = fixed_subset_ce(model, features, tokenized)
        if final_fixed_ce >= initial_fixed_ce:
            raise RuntimeError(f"16-video fixed-target CE did not decrease: {arm}")
        with (arm_dir / "trace.jsonl").open("x", encoding="utf-8") as stream:
            for item in traces:
                stream.write(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n")
        results[arm] = {"steps": 200, "initial_fixed_subset_ce": initial_fixed_ce,
                        "final_fixed_subset_ce": final_fixed_ce,
                        "first_step_ce": ce_values[0], "final_step_ce": ce_values[-1],
                        "mean_step_seconds": sum(step_seconds) / len(step_seconds),
                        "final_hashes": final_hashes, "trace_sha256": sha256_file(arm_dir / "trace.jsonl"),
                        "resume_checkpoint_sha256": sha256_file(arm_dir / "resume_step100.pt"),
                        "exact_resume_pass": True, "frozen_weights_unchanged": True}
        expected_trace = traces
        del optimizer
        torch.cuda.empty_cache()
    if results["A"]["trace_sha256"] != results["B"]["trace_sha256"]:
        raise RuntimeError("A/B trace hashes differ")
    peak = torch.cuda.max_memory_reserved() / 2**20
    if peak >= 4096:
        raise RuntimeError(f"4 GB memory gate failed: {peak} MiB")
    write_new(out / "result.json", {"status": "PASS", "created_at_utc": datetime.now(timezone.utc).isoformat(),
                                    "data_audit_sha256": sha256_file(DATA / "data_audit.json"),
                                    "cache_audit_sha256": sha256_file(DATA / "feature_cache/cache_audit.json"),
                                    "script_sha256": sha256_file(Path(__file__)),
                                    "source_model_path": source, "subset_video_ids": ids,
                                    "max_caption_tokens": max(lengths), "max_len": 96,
                                    "infonce_mask_legal": True, "first_token_eos_supervision": True,
                                    "live_cached_feature_max_abs_error": live_cache_error,
                                    "paired_trace_equal": True, "peak_reserved_mib": peak,
                                    "wall_seconds": time.perf_counter() - started, "arms": results,
                                    "test_split_read": False})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    try:
        main(args.output)
    except Exception:
        if args.output.is_dir():
            write_new(args.output / "failure.json", {"at_utc": datetime.now(timezone.utc).isoformat(),
                                                    "traceback": traceback.format_exc()})
        raise
