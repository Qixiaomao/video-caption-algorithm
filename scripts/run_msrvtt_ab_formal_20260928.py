"""Execute the frozen, independent MSR-VTT A/B three-seed training protocol.

This runner never opens the test split. It requires a separately frozen JSON
configuration created after the media/cache and engineering gates pass.
"""
from __future__ import annotations

import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
DATA = ROOT / "runs/msrvtt_data_full_20260928"

import torch

from scripts.diagnose_prefix_sensitivity import donor_permutation
from scripts.eval_retrieval import encode_texts
from scripts.run_caption_policy_ab import cached_caption_loss, raw_backbone_features, reference_seed, tokenize_caption, tokenize_references
from scripts.run_full_msvd_balanced import evaluate_validation, trace_evidence
from scripts.run_full_msvd_infonce_ablation import (
    assert_trace_equal, cached_caption_infonce_objective, retrieval_diagnostic,
)
from scripts.run_overfit16 import component_hashes, configure_trainable, state_sha
from src.cli.train_caption_mapper import set_seed
from src.data.data_loader import build_dataloader
from src.models.caption_model import VideoCaptionModel
from src.training.checkpoint import (
    restore_training_state, save_training_checkpoint, sha256_file,
    validate_training_resume_contract,
)
from src.training.contrastive import exact_caption_positive_mask


def write_new(path: Path, value: object) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)


def validate_config(config: dict, config_path: Path) -> None:
    if config.get("status") != "FROZEN" or config.get("protocol") != "msrvtt-ab-v1":
        raise RuntimeError("Formal protocol is not frozen")
    if config.get("seeds") != [123, 124, 125] or config.get("arms") != ["A", "B"]:
        raise RuntimeError("Seed/arm registration changed")
    if config.get("data_gate_sha256") != sha256_file(DATA / "data_gate.json"):
        raise RuntimeError("Data gate hash changed")
    if config.get("engineering_smoke_sha256") != sha256_file(Path(config["engineering_smoke_result_path"])):
        raise RuntimeError("Engineering smoke hash changed")
    if config.get("pilot_sha256") != sha256_file(Path(config["pilot_result_path"])):
        raise RuntimeError("Train/val pilot hash changed")
    if config.get("validation_timing_sha256") != sha256_file(Path(config["validation_timing_result_path"])):
        raise RuntimeError("Validation timing benchmark hash changed")
    if config.get("test_evaluation_config_sha256") != sha256_file(Path(config["test_evaluation_config_path"])):
        raise RuntimeError("Frozen test evaluation protocol hash changed")
    if config.get("gpt2_model_sha256") != sha256_file(Path(config["gpt2_source_path"]) / "model.safetensors"):
        raise RuntimeError("GPT-2 source weights changed")
    epochs = int(config["epochs"])
    milestones = config["milestone_epochs"]
    if epochs < 20 or config.get("steps_per_epoch") != 814 or config.get("steps_per_arm") != epochs * 814:
        raise RuntimeError("Formal budget violates full reference coverage")
    if sorted(set(milestones)) != milestones or milestones[-1] != epochs:
        raise RuntimeError("Checkpoint milestones do not include the fixed endpoint")
    if config.get("batch_size") != 8 or config.get("last_batch_size") != 9 or config.get("max_len") != 96:
        raise RuntimeError("Batch/caption policy changed")
    if config.get("learning_rate") != 3e-4 or config.get("weight_decay") != 0.01:
        raise RuntimeError("Registered optimizer changed")
    if config.get("alignment_weight") != 0.1 or config.get("infonce_temperature") != 0.07:
        raise RuntimeError("Registered InfoNCE settings changed")
    if config.get("schedule") != "constant_lr" or config.get("cudnn_allow_tf32") is not True or config.get("cuda_matmul_allow_tf32") is not False:
        raise RuntimeError("Registered schedule/precision changed")
    if config.get("caption_gate") != {"nonempty_rate": 1.0, "unique_rate_min": 0.30,
                                       "largest_template_rate_max": 0.25, "grounding_ci95_low_gt": 0.0,
                                       "positive_mismatch_fraction_min": 0.80}:
        raise RuntimeError("Registered validation gate changed")
    if config_path.resolve().parent != (ROOT / "experiments").resolve():
        raise RuntimeError("Frozen config must be in experiments/")


def make_schedule(records: list[dict], seed: int, epochs: int) -> dict:
    generator = torch.Generator().manual_seed(seed + 333)
    batches = []
    for _ in range(epochs):
        permutation = torch.randperm(len(records), generator=generator).tolist()
        epoch_batches = [permutation[i:i + 8] for i in range(0, 6504, 8)]
        epoch_batches.append(permutation[6504:])
        if len(epoch_batches) != 814 or len(epoch_batches[-1]) != 9 or sorted(x for batch in epoch_batches for x in batch) != list(range(6513)):
            raise RuntimeError("Invalid full-coverage schedule")
        batches.append(epoch_batches)
    orders = {row["video_id"]: torch.randperm(len(row["captions"]),
        generator=torch.Generator().manual_seed(reference_seed(row["video_id"]) + seed - 123)).tolist()
        for row in records}
    return {"seed": seed, "batch_seed": seed + 333, "epochs": epochs,
            "batches": batches, "caption_orders": orders}


def caption_gate(metrics: dict, val_count: int, config: dict) -> dict:
    thresholds = config["caption_gate"]
    result = {
        "nonempty": metrics["nonempty_rate"] == thresholds["nonempty_rate"],
        "unique": metrics["unique_rate"] >= thresholds["unique_rate_min"],
        "largest_template": metrics["largest_template_rate"] <= thresholds["largest_template_rate_max"],
        "grounding": metrics["grounding_bootstrap"]["ci95_low"] > thresholds["grounding_ci95_low_gt"],
        "positive_mismatch": metrics["positive_mismatch_videos"] >= math.ceil(thresholds["positive_mismatch_fraction_min"] * val_count),
    }
    result["pass"] = all(result.values())
    return result


def run_arm(model, arm: str, seed: int, folder: Path, initial: dict,
            initial_hashes: dict, schedule: dict, config: dict, config_path: Path,
            records: list[dict], train_features: torch.Tensor, tokenized: list[list[torch.Tensor]],
            val_records: list[dict], val_features: torch.Tensor, references: dict,
            text_embeddings: torch.Tensor, gold: torch.Tensor, source_hashes: dict) -> dict:
    folder.mkdir()
    model.load_state_dict(initial, strict=True)
    if component_hashes(model) != initial_hashes:
        raise RuntimeError("Paired initial state changed")
    params = configure_trainable(model)
    optimizer = torch.optim.AdamW([parameter for _, parameter in params], lr=3e-4, weight_decay=0.01)
    set_seed(seed)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    ids = [row["video_id"] for row in records]
    donor = donor_permutation(len(val_records), seed=321)
    coverage = {vid: set() for vid in ids}
    milestone_epochs = set(config["milestone_epochs"])
    evaluations = []
    step = 0
    step_times = []
    prior_trace = (folder.parent / "A/trace.jsonl").open(encoding="utf-8") if arm == "B" else None
    try:
        with (folder / "trace.jsonl").open("x", encoding="utf-8") as trace_stream, \
             (folder / "train_metrics.jsonl").open("x", encoding="utf-8") as metric_stream:
            for epoch, epoch_batches in enumerate(schedule["batches"], 1):
                ce_epoch = []
                visits = Counter()
                for indices in epoch_batches:
                    tick = time.perf_counter()
                    step += 1
                    texts = []
                    target_ids = []
                    for index in indices:
                        row = records[index]
                        vid = row["video_id"]
                        order = schedule["caption_orders"][vid]
                        caption_index = order[(epoch - 1) % len(order)]
                        texts.append(row["captions"][caption_index])
                        target_ids.append(tokenized[index][caption_index])
                        visits[vid] += 1
                        coverage[vid].add(caption_index)
                    captions = torch.stack(target_ids)
                    features = train_features[indices]
                    evidence = trace_evidence(step, epoch - 1, indices, ids, features, captions, texts)
                    if prior_trace is not None:
                        old_line = prior_trace.readline()
                        if not old_line:
                            raise RuntimeError("A trace ended before B")
                        assert_trace_equal(evidence, json.loads(old_line))
                    trace_stream.write(json.dumps(evidence, ensure_ascii=False, sort_keys=True) + "\n")
                    trace_stream.flush()
                    mask = exact_caption_positive_mask(captions, model.decoder.tokenizer.pad_token_id)
                    if not torch.all(mask.any(dim=1)) or not torch.all((~mask).any(dim=1)):
                        raise RuntimeError(f"InfoNCE legal-negative failure at step {step}")
                    model.train()
                    model.encoder.backbone.eval()
                    optimizer.zero_grad(set_to_none=True)
                    if arm == "B":
                        output = cached_caption_infonce_objective(model, features.cuda(), captions.cuda())
                    else:
                        ce = cached_caption_loss(model, features.cuda(), captions.cuda())
                        output = {"loss": ce, "caption_loss": ce, "alignment_loss": ce.new_zeros(())}
                    if any(not torch.isfinite(output[key]) for key in ("loss", "caption_loss", "alignment_loss")):
                        raise RuntimeError(f"Nonfinite objective {seed}/{arm}/{step}")
                    output["loss"].backward()
                    if any(parameter.grad is None or not torch.isfinite(parameter.grad).all() for _, parameter in params):
                        raise RuntimeError(f"Missing/nonfinite gradient {seed}/{arm}/{step}")
                    optimizer.step()
                    numeric = {key: float(output[key].detach()) for key in ("loss", "caption_loss", "alignment_loss")}
                    numeric.update({"step": step, "epoch": epoch})
                    metric_stream.write(json.dumps(numeric) + "\n")
                    metric_stream.flush()
                    ce_epoch.append(numeric["caption_loss"])
                    step_times.append(time.perf_counter() - tick)
                if len(visits) != 6513 or set(visits.values()) != {1}:
                    raise RuntimeError("Epoch did not visit each train ID exactly once")
                print(json.dumps({"seed": seed, "arm": arm, "epoch": epoch,
                                  "step": step, "mean_train_ce": sum(ce_epoch) / len(ce_epoch),
                                  "peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20}), flush=True)
                if epoch in milestone_epochs:
                    checkpoint = folder / f"checkpoint_epoch{epoch:03d}.pt"
                    training_contract = {"objective_version": 3, "caption_objective": "causal_lm_cross_entropy",
                        "caption_policy": "balanced_all_references", "alignment_objective": "infonce" if arm == "B" else "none",
                        "alignment_weight": 0.1 if arm == "B" else 0.0, "info_nce_temperature": 0.07,
                        "false_negative_policy": "exact_caption", "cpr_weight": 0.0, "cpr_margin": None,
                        "cpr_negative_policy": "none", "physical_batch_size": 8,
                        "last_batch_size": 9,
                        "video_representation": "mean_visual_prefix_tokens_l2_normalized",
                        "text_representation": "frozen_gpt2_wte_masked_mean_l2_normalized",
                        "frozen_gpt2": True, "frozen_vit_backbone": True, "cached_frozen_backbone": True}
                    dataset_contract = {"annotations": {split: {"path": str(DATA / "annotations" / f"{split}.json"),
                                                            "sha256": sha256_file(DATA / "annotations" / f"{split}.json")}
                                                        for split in ("train", "val")},
                                        "manifest": {"path": str(DATA / "manifest.json"),
                                                     "sha256": sha256_file(DATA / "manifest.json")}}
                    save_training_checkpoint(checkpoint, model=model, optimizer=optimizer, scheduler=None,
                        progress={"step": step, "epoch": epoch, "epoch_complete": True},
                        args={"gpt2_name": config["gpt2_source_path"], "seed": seed,
                              "arm": arm, "max_len": 96, "epochs": config["epochs"],
                              "lr": 3e-4}, training_contract=training_contract,
                        dataset_contract=dataset_contract,
                        loader_state={"schedule_sha256": sha256_file(folder.parent / "schedule.json"),
                                      "next_epoch": epoch + 1, "config_sha256": sha256_file(config_path)})
                    caption = evaluate_validation(model, val_features, val_records, references,
                                                  donor, epoch, step, "cuda")
                    retrieval = retrieval_diagnostic(model, val_features, text_embeddings, gold, "cuda")
                    gate = caption_gate(caption["metrics"], len(val_records), config)
                    entry = {"epoch": epoch, "step": step, "caption": caption, "retrieval": retrieval,
                             "caption_gate": gate, "checkpoint": str(checkpoint),
                             "checkpoint_sha256": sha256_file(checkpoint)}
                    write_new(folder / f"evaluation_epoch{epoch:03d}.json", entry)
                    evaluations.append(entry)
                    print(json.dumps({"seed": seed, "arm": arm, "milestone": epoch,
                                      "val_native_nll": caption["metrics"]["native_nll"],
                                      "caption_gate_pass": gate["pass"],
                                      "val_t2v_r1": retrieval["true_labels"]["text_to_video"]["r_at_1"]}), flush=True)
            if prior_trace is not None and prior_trace.readline():
                raise RuntimeError("A trace has additional rows")
    finally:
        if prior_trace is not None:
            prior_trace.close()
    if step != config["steps_per_arm"]:
        raise RuntimeError("Fixed endpoint step count mismatch")
    if any(len(coverage[row["video_id"]]) != len(row["captions"]) for row in records):
        raise RuntimeError("Some train references were never selected")
    final_hashes = component_hashes(model)
    if any(final_hashes[key] != initial_hashes[key] for key in ("decoder", "backbone")):
        raise RuntimeError("Frozen model changed")
    if any(final_hashes[key] == initial_hashes[key] for key in ("projection", "mapper")):
        raise RuntimeError("Trainable module unchanged")
    if source_hashes != {name: sha256_file(ROOT / name) for name in source_hashes}:
        raise RuntimeError("Source changed during formal arm")
    final_checkpoint = Path(evaluations[-1]["checkpoint"])
    checkpoint_payload = torch.load(final_checkpoint, map_location="cpu", weights_only=True)
    validate_training_resume_contract(checkpoint_payload, training_contract, dataset_contract)
    model.load_state_dict(checkpoint_payload["model_state"], strict=True)
    resumed_progress = restore_training_state(checkpoint_payload, optimizer)
    if (resumed_progress["step"] != step or resumed_progress["epoch"] != config["epochs"]
            or checkpoint_payload["loader_state"]["schedule_sha256"] != sha256_file(folder.parent / "schedule.json")
            or component_hashes(model) != final_hashes):
        raise RuntimeError("Formal endpoint checkpoint strict reload failed")
    result = {"seed": seed, "arm": arm, "steps": step, "epochs": config["epochs"],
              "initial_component_hashes": initial_hashes, "final_component_hashes": final_hashes,
              "trace_sha256": sha256_file(folder / "trace.jsonl"),
              "train_metrics_sha256": sha256_file(folder / "train_metrics.jsonl"),
              "evaluations": [{"epoch": entry["epoch"], "step": entry["step"],
                               "path": str(folder / f"evaluation_epoch{entry['epoch']:03d}.json"),
                               "checkpoint_sha256": entry["checkpoint_sha256"]} for entry in evaluations],
              "all_train_references_covered": True, "frozen_weights_unchanged": True,
              "final_checkpoint_strict_reload": True,
              "mean_step_seconds": sum(step_times) / len(step_times),
              "elapsed_seconds": time.perf_counter() - started,
              "peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
              "test_split_read": False, "technical_run_complete": True,
              "paper_claim_supported": None}
    write_new(folder / "result.json", result)
    del optimizer
    torch.cuda.empty_cache()
    return result


def main(config_path: Path, out: Path) -> None:
    if out.exists():
        raise FileExistsError(out)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    validate_config(config, config_path)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    out.mkdir(parents=True)
    write_new(out / "launch.json", {"at_utc": datetime.now(timezone.utc).isoformat(),
                                    "config_path": str(config_path.resolve()),
                                    "config_sha256": sha256_file(config_path),
                                    "data_gate_sha256": sha256_file(DATA / "data_gate.json"),
                                    "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                                    "gpu": torch.cuda.get_device_name(),
                                    "torch_version": str(torch.__version__),
                                    "test_split_read": False})
    train = json.loads((DATA / "annotations/train.json").read_text(encoding="utf-8"))
    val = json.loads((DATA / "annotations/val.json").read_text(encoding="utf-8"))
    train_cache = torch.load(DATA / "feature_cache/train_raw_vit.pt", map_location="cpu", weights_only=True)
    val_cache = torch.load(DATA / "feature_cache/val_raw_vit.pt", map_location="cpu", weights_only=True)
    cache_audit = json.loads((DATA / "feature_cache/cache_audit.json").read_text(encoding="utf-8"))
    if train_cache["video_ids"] != [row["video_id"] for row in train] or val_cache["video_ids"] != [row["video_id"] for row in val]:
        raise RuntimeError("Cache order/coverage changed")
    if len(train) != 6513 or len(val) != 497:
        raise RuntimeError("Split cardinality changed")
    source_hashes = {name: sha256_file(ROOT / name) for name in (
        "scripts/run_msrvtt_ab_formal_20260928.py", "scripts/run_caption_policy_ab.py",
        "scripts/run_full_msvd_infonce_ablation.py", "scripts/run_full_msvd_balanced.py",
        "src/models/caption_model.py", "src/models/text_decoder.py", "src/models/video_encoder.py",
        "src/training/contrastive.py", "src/training/checkpoint.py")}
    write_new(out / "source_hashes.json", source_hashes)
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = True  # Same frozen ViT precision as the data cache.
    pairs = []
    seed_initial_hashes = []
    tokenized_cache = None
    text_embeddings_cache = None
    for seed in config["seeds"]:
        seed_folder = out / f"seed{seed}"
        seed_folder.mkdir()
        set_seed(seed)
        model = VideoCaptionModel(gpt2_name=config["gpt2_source_path"], freeze_vit=True,
                                  vit_enable_torch_compile=False).cuda()
        model.eval()
        if state_sha(model.encoder.backbone.state_dict()) != cache_audit["backbone_state_sha256"]:
            raise RuntimeError("Frozen ViT backbone differs from feature cache source")
        probe_loader = build_dataloader(str(DATA / "annotations/train.json"), model.decoder.tokenizer,
            batch_size=8, max_len=96, num_frame=8, image_size=224,
            shuffle=False, random_caption=False)
        if [row["video_id"] for row in probe_loader.dataset.anns] != [row["video_id"] for row in train]:
            raise RuntimeError("Dataloader silently dropped train videos")
        probe = next(iter(probe_loader))
        with torch.inference_mode():
            live = raw_backbone_features(model.encoder, probe["video"].cuda())
        if probe["video_id"] != train_cache["video_ids"][:8] or float((live.cpu() - train_cache["raw_features"][:8]).abs().max()) > 1e-6:
            raise RuntimeError("Live/cached frozen ViT feature mismatch")
        del probe_loader, probe, live
        initial = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        initial_hashes = component_hashes(model)
        for previous in seed_initial_hashes:
            if (initial_hashes["decoder"] != previous["decoder"]
                    or initial_hashes["backbone"] != previous["backbone"]
                    or initial_hashes["projection"] == previous["projection"]
                    or initial_hashes["mapper"] == previous["mapper"]):
                raise RuntimeError("Fresh seed initialization contract failed")
        seed_initial_hashes.append(initial_hashes)
        initial_path = seed_folder / "initial_model.pt"
        with initial_path.open("xb") as stream:
            torch.save({"model_state": initial}, stream)
        schedule = make_schedule(train, seed, config["epochs"])
        write_new(seed_folder / "schedule.json", schedule)
        tokenizer = model.decoder.tokenizer
        if tokenized_cache is None:
            tokenized_cache = [[tokenize_caption(tokenizer, text, max_len=96) for text in row["captions"]]
                               for row in train]
        tokenized = tokenized_cache
        references = {row["video_id"]: tokenize_references(tokenizer, row["captions"]) for row in val}
        if text_embeddings_cache is None:
            text_embeddings_cache = encode_texts(model, [text for row in val for text in row["captions"]], 64, 128, "cuda")
        text_embeddings = text_embeddings_cache
        gold = torch.tensor([i for i, row in enumerate(val) for _ in row["captions"]])
        results = {}
        for arm in config["arms"]:
            results[arm] = run_arm(model, arm, seed, seed_folder / arm, initial, initial_hashes,
                schedule, config, config_path, train, train_cache["raw_features"], tokenized,
                val, val_cache["raw_features"], references, text_embeddings, gold, source_hashes)
        if results["A"]["trace_sha256"] != results["B"]["trace_sha256"]:
            raise RuntimeError("Paired A/B trace hashes differ")
        a_eval = json.loads(Path(results["A"]["evaluations"][-1]["path"]).read_text(encoding="utf-8"))
        b_eval = json.loads(Path(results["B"]["evaluations"][-1]["path"]).read_text(encoding="utf-8"))
        a_r1 = a_eval["retrieval"]["true_labels"]["text_to_video"]["r_at_1"]
        b_r1 = b_eval["retrieval"]["true_labels"]["text_to_video"]["r_at_1"]
        a_caption_metrics = a_eval["caption"]["metrics"]
        b_caption_metrics = b_eval["caption"]["metrics"]
        nll_safeguard = b_caption_metrics["native_nll"] <= a_caption_metrics["native_nll"] + 0.15
        f1_safeguard = (b_caption_metrics["mean_max_reference_token_f1"]
                        >= a_caption_metrics["mean_max_reference_token_f1"] - 0.05)
        pair = {"seed": seed, "arms_complete": True, "paired_trace_equal": True,
                "A_caption_gate_pass": a_eval["caption_gate"]["pass"],
                "B_caption_gate_pass": b_eval["caption_gate"]["pass"],
                "B_native_nll_le_A_plus_0_15": nll_safeguard,
                "B_reference_token_f1_ge_A_minus_0_05": f1_safeguard,
                "A_val_t2v_r1": a_r1, "B_val_t2v_r1": b_r1,
                "B_minus_A_val_t2v_r1": b_r1 - a_r1,
                "technical_run_complete": True, "claim_supported": None,
                "test_split_read": False}
        write_new(seed_folder / "pair_result.json", pair)
        pairs.append(pair)
        del model, initial, tokenized, text_embeddings
        torch.cuda.empty_cache()
    mean_delta = sum(pair["B_minus_A_val_t2v_r1"] for pair in pairs) / len(pairs)
    positive_pairs = sum(pair["B_minus_A_val_t2v_r1"] > 0 for pair in pairs)
    registered_direction = all(pair["B_caption_gate_pass"] for pair in pairs) and mean_delta > 0 and positive_pairs >= 2
    write_new(out / "validation_freeze.json", {"at_utc": datetime.now(timezone.utc).isoformat(),
                                               "six_arms_complete": True, "test_not_read": True,
                                               "fixed_endpoint_epoch": config["epochs"],
                                               "pairs": pairs, "B_minus_A_val_t2v_r1_mean": mean_delta,
                                               "positive_pairs": positive_pairs,
                                               "registered_transfer_direction_supported_on_val": registered_direction,
                                               "test_evaluation_config_sha256": config["test_evaluation_config_sha256"]})
    write_new(out / "completion.json", {"at_utc": datetime.now(timezone.utc).isoformat(),
                                        "technical_training_complete": True,
                                        "seeds": config["seeds"], "arms": config["arms"],
                                        "validation_freeze_path": str(out / "validation_freeze.json"),
                                        "test_split_read": False})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    try:
        main(args.config, args.output)
    except Exception:
        if args.output.is_dir() and not (args.output / "failure.json").exists():
            write_new(args.output / "failure.json", {"at_utc": datetime.now(timezone.utc).isoformat(),
                                                    "traceback": traceback.format_exc()})
        raise
