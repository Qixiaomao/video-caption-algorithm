#!/usr/bin/env python3
"""Replay the locked full-MSVD caption run with one InfoNCE intervention."""
from __future__ import annotations

import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import torch.nn.functional as F

from scripts.diagnose_prefix_sensitivity import donor_permutation
from scripts.eval_retrieval import encode_texts, evaluate_scores
from scripts.run_caption_policy_ab import tokenize_caption, tokenize_references, write_json
from scripts.run_full_msvd_balanced import (
    DATA_SHA256,
    FORMAL_EPOCHS,
    FORMAL_STEPS,
    SOURCE_FILES as BASE_SOURCE_FILES,
    TRAIN,
    VAL,
    evaluate_validation,
    trace_evidence,
)
from scripts.run_overfit16 import component_hashes, configure_trainable, state_sha
from src.cli.train_caption_mapper import set_seed
from src.models.caption_model import VideoCaptionModel, build_caption_supervision
from src.training.checkpoint import capture_rng_state, save_training_checkpoint, sha256_file
from src.training.contrastive import (
    build_alignment_embeddings,
    exact_caption_positive_mask,
    similarity_diagnostics,
    symmetric_multi_positive_info_nce,
)


A_ROOT = ROOT / "runs/full_msvd_balanced_20260905"
A_HASHES = {
    "initial_model.pt": "b0deacbea2239da7dd8f6d815be13f75afc962cce68b0a84771784e0db08b1ff",
    "preflight.json": "6292037e2a58fa3a6386bf182eef6df56e85e4e8374854f736c8467bf9b27652",
    "schedule.json": "3520204739b0bac71a6cea9e5a2f26c4dbfee17e3b19790e2340e9efc054fd86",
    "train_features.pt": "8cbc39e504be44865db302f87856b97b48318b6c32b60db7f8ed1ecb224a7ad6",
    "val_features.pt": "5ac1dc53d8e03fcdf3c3c8abae98e1a5c518774ffe48a00eacbb3648503aeee4",
    "trace.jsonl": "8b236e9e3a7f3fbe7ab6dff679326c810f600ae519f275a11271789a4de27a9b",
    "result.json": "80063ad11a750741735124b3f4bc4a81555a6f8eb9c612d89bee821279c7051f",
    "checkpoint_epoch066.pt": "8155d68c5b3ac9f3f7396027a32b2e0ebea052a17aef79ecce8cf10a1e0f9e6a",
}
LAMBDA = 0.10
TEMPERATURE = 0.07
MILESTONES = (5, 20, 40, 66)
SOURCE_FILES = tuple(dict.fromkeys((
    "experiments/full_msvd_infonce_ablation_protocol_20260906.md",
    "scripts/run_full_msvd_infonce_ablation.py",
    "scripts/eval_retrieval.py",
    "src/training/contrastive.py",
    *BASE_SOURCE_FILES,
)))


def verify_registered_baseline() -> tuple[dict, dict]:
    for name, expected in A_HASHES.items():
        path = A_ROOT / name
        if not path.is_file() or sha256_file(path) != expected:
            raise RuntimeError(f"Registered caption-only artifact changed: {name}")
    result = json.loads((A_ROOT / "result.json").read_text(encoding="utf-8"))
    preflight = json.loads((A_ROOT / "preflight.json").read_text(encoding="utf-8"))
    expected_config = {"seed": 123, "batch_seed": 456, "donor_seed": 321,
                       "epochs": 66, "steps": 9900, "batch_size": 8,
                       "frames": 8, "image_size": 224, "max_len": 64,
                       "lr": 3e-4, "weight_decay": 0.01,
                       "milestone_epochs": [5, 20, 40, 66]}
    if result.get("decision") != "proceed_to_infonce_ablation":
        raise RuntimeError("Registered caption-only run did not pass its viability decision")
    if result.get("config") != expected_config or preflight.get("config") != expected_config:
        raise RuntimeError("Registered caption-only configuration changed")
    if preflight.get("split_counts") != {"train": 1200, "val": 100, "test_read": False}:
        raise RuntimeError("Registered caption-only split contract changed")
    visits = result.get("train_video_visits", {})
    coverage = result.get("caption_coverage", {})
    if len(visits) != 1200 or set(visits.values()) != {66}:
        raise RuntimeError("Registered caption-only video coverage changed")
    if len(coverage) != 1200 or sum(x["unique_visited"] for x in coverage.values()) != 48774:
        raise RuntimeError("Registered caption-only reference coverage changed")
    if not result.get("frozen_weights_unchanged") or not result.get("trainable_modules_changed"):
        raise RuntimeError("Registered caption-only component audit is invalid")
    if result.get("paper_ready") is not False or preflight.get("paper_ready") is not False:
        raise RuntimeError("Registered caption-only paper-ready boundary changed")
    return result, preflight


def assert_trace_equal(actual: dict, expected: dict) -> None:
    fields = (
        "step", "epoch", "indices", "video_ids", "target_texts",
        "caption_ids_sha256", "visual_features_sha256",
        "python_rng_sha256", "torch_cpu_rng_sha256", "torch_cuda_rng_sha256",
    )
    mismatched = [field for field in fields if actual.get(field) != expected.get(field)]
    if mismatched:
        raise RuntimeError(f"Caption-only trace mismatch at step {actual.get('step')}: {mismatched}")


def cached_caption_infonce_objective(
    model: VideoCaptionModel,
    raw_features: torch.Tensor,
    caption_ids: torch.Tensor,
) -> dict[str, torch.Tensor]:
    video_embedding = model.proj(model.encoder.proj(raw_features))
    attention_mask, labels = build_caption_supervision(
        caption_ids, model.decoder.tokenizer.pad_token_id,
    )
    visual_prefix = model.decoder.build_visual_prefix(video_embedding)
    decoder_result = model.decoder(
        video_embedding,
        caption_ids,
        attention_mask,
        labels,
        visual_prefix=visual_prefix,
    )
    caption_loss = decoder_result["loss"]
    with torch.no_grad():
        token_embeddings = model.decoder.model.transformer.wte(caption_ids)
    video_embeddings, text_embeddings = build_alignment_embeddings(
        visual_prefix=visual_prefix,
        token_embeddings=token_embeddings,
        caption_ids=caption_ids,
        pad_id=model.decoder.tokenizer.pad_token_id,
    )
    positive_mask = exact_caption_positive_mask(
        caption_ids, model.decoder.tokenizer.pad_token_id,
    )
    alignment_loss = symmetric_multi_positive_info_nce(
        video_embeddings,
        text_embeddings,
        temperature=TEMPERATURE,
        positive_mask=positive_mask,
    )
    diagnostics = similarity_diagnostics(video_embeddings, text_embeddings, positive_mask)
    return {
        "loss": caption_loss + LAMBDA * alignment_loss,
        "caption_loss": caption_loss,
        "alignment_loss": alignment_loss,
        **diagnostics,
    }


@torch.inference_mode()
def retrieval_diagnostic(
    model: VideoCaptionModel,
    raw_features: torch.Tensor,
    text_embeddings: torch.Tensor,
    caption_gold: torch.Tensor,
    device: str,
) -> dict:
    model.eval()
    features = raw_features.to(device)
    video_embedding = model.proj(model.encoder.proj(features))
    visual_prefix = model.decoder.build_visual_prefix(video_embedding)
    video_embeddings = F.normalize(visual_prefix.mean(dim=1), dim=-1).cpu()
    return evaluate_scores(video_embeddings, text_embeddings, caption_gold, seed=123)


def primary_decision(a_caption: dict, b_caption: dict,
                     a_retrieval: dict, b_retrieval: dict) -> dict:
    a_t2v = a_retrieval["true_labels"]["text_to_video"]["r_at_1"]
    b_t2v = b_retrieval["true_labels"]["text_to_video"]["r_at_1"]
    b_shuffled = b_retrieval["shuffled_label_control"]["text_to_video"]["r_at_1"]
    gates = {
        "caption_viability_pass": bool(b_caption["primary_viability_gate"]["pass"]),
        "native_nll_le_a_plus_0_15": b_caption["native_nll"] <= a_caption["native_nll"] + 0.15,
        "token_f1_ge_a_minus_0_05": (
            b_caption["mean_max_reference_token_f1"]
            >= a_caption["mean_max_reference_token_f1"] - 0.05
        ),
        "t2v_r1_gt_caption_only": b_t2v > a_t2v,
        "similarity_gap_gt_caption_only": (
            b_retrieval["similarity"]["gap"] > a_retrieval["similarity"]["gap"]
        ),
        "t2v_r1_gt_deranged_control": b_t2v > b_shuffled,
    }
    gates["pass"] = all(gates.values())
    gates["decision"] = (
        "infonce_candidate_for_repeated_seeds"
        if gates["pass"] else "retain_caption_only_or_redesign_infonce"
    )
    return gates


def load_strict(model: VideoCaptionModel, path: Path) -> dict:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    incompatible = model.load_state_dict(payload["model_state"], strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(f"Strict load failed for {path}")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.output_root.exists():
        raise FileExistsError("Refusing to overwrite an existing experiment directory")
    if not torch.cuda.is_available():
        raise RuntimeError("This registered run requires the local CUDA GPU")
    args.output_root.mkdir(parents=True)

    a_result, a_preflight = verify_registered_baseline()
    for split, path in (("train", TRAIN), ("val", VAL)):
        if sha256_file(path) != DATA_SHA256[split]:
            raise RuntimeError(f"Pinned {split} annotations changed")
    source_hashes = {name: sha256_file(ROOT / name) for name in SOURCE_FILES}
    train_records = json.loads(TRAIN.read_text(encoding="utf-8"))
    val_records = json.loads(VAL.read_text(encoding="utf-8"))
    schedule = json.loads((A_ROOT / "schedule.json").read_text(encoding="utf-8"))
    train_cache = torch.load(A_ROOT / "train_features.pt", map_location="cpu", weights_only=True)
    val_cache = torch.load(A_ROOT / "val_features.pt", map_location="cpu", weights_only=True)
    train_features = train_cache["features"]
    val_features = val_cache["features"]
    video_ids = [str(value) for value in train_cache["video_ids"]]
    if video_ids != [str(record["video_id"]) for record in train_records]:
        raise RuntimeError("Registered training cache order changed")
    if [str(value) for value in val_cache["video_ids"]] != [str(record["video_id"]) for record in val_records]:
        raise RuntimeError("Registered validation cache order changed")

    source_path = a_preflight["model_source"]["path"]
    model = VideoCaptionModel(gpt2_name=source_path, freeze_vit=True,
                              vit_enable_torch_compile=False)
    load_strict(model, A_ROOT / "initial_model.pt")
    initial_hashes = component_hashes(model)
    if initial_hashes != a_result["initial_component_hashes"]:
        raise RuntimeError("Common initial component hashes changed")
    model = model.to("cuda")
    configure_trainable(model)
    trainable = [(name, parameter) for name, parameter in model.named_parameters()
                 if parameter.requires_grad]
    optimizer = torch.optim.AdamW([parameter for _, parameter in trainable],
                                  lr=3e-4, weight_decay=0.01)

    by_id = {str(record["video_id"]): record for record in train_records}
    reference_orders = schedule["reference_orders"]
    token_cache = {text: tokenize_caption(model.decoder.tokenizer, text)
                   for record in train_records for text in record["captions"]}
    val_references = {str(record["video_id"]): tokenize_references(
        model.decoder.tokenizer, record["captions"]) for record in val_records}
    all_val_captions = []
    caption_gold = []
    for index, record in enumerate(val_records):
        all_val_captions.extend(record["captions"])
        caption_gold.extend([index] * len(record["captions"]))
    text_embeddings = encode_texts(
        model, all_val_captions, max_length=64, batch_size=128, device="cuda",
    )
    caption_gold_tensor = torch.tensor(caption_gold, dtype=torch.long)

    smoke_batches = schedule["batch_schedule"][0][:2]
    epoch_schedules = [smoke_batches] if args.smoke else schedule["batch_schedule"]
    expected_steps = 2 if args.smoke else FORMAL_STEPS
    milestone_set = {1} if args.smoke else set(MILESTONES)
    donors = donor_permutation(8 if args.smoke else 100, seed=321)
    eval_records = val_records[:8] if args.smoke else val_records
    eval_features = val_features[:8] if args.smoke else val_features
    eval_references = {str(record["video_id"]): val_references[str(record["video_id"])]
                       for record in eval_records}
    retrieval_text = text_embeddings if not args.smoke else encode_texts(
        model,
        [caption for record in eval_records for caption in record["captions"]],
        max_length=64,
        batch_size=128,
        device="cuda",
    )
    retrieval_gold = caption_gold_tensor if not args.smoke else torch.tensor(
        [index for index, record in enumerate(eval_records) for _ in record["captions"]],
        dtype=torch.long,
    )

    preflight = {
        "protocol": "full-msvd-caption-infonce-ablation-v1",
        "paper_ready": False,
        "smoke_test": args.smoke,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "intervention": {"alignment_objective": "infonce", "alignment_weight": LAMBDA,
                         "temperature": TEMPERATURE,
                         "false_negative_policy": "exact_caption"},
        "common_config": a_result["config"],
        "effective_steps": expected_steps,
        "a_artifact_sha256": A_HASHES,
        "source_sha256": source_hashes,
        "git_head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
        ).strip(),
        "split_counts": {"train": 1200, "val": 100, "test_read": False},
        "reference_counts": {"train": 48774, "val": 4290},
        "initial_component_hashes": initial_hashes,
        "runtime": {"python": sys.version, "torch": torch.__version__,
                    "gpu": torch.cuda.get_device_name(),
                    "deterministic_algorithms": True, "tf32": False},
    }
    write_json(args.output_root / "preflight.json", preflight)

    set_seed(123)
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    step = 0
    seen = Counter()
    caption_seen = {video_id: Counter() for video_id in video_ids}
    evaluations = []
    trace_path = args.output_root / "trace.jsonl"
    metrics_path = args.output_root / "train_metrics.jsonl"
    checkpoint_paths = []

    with (A_ROOT / "trace.jsonl").open("r", encoding="utf-8") as a_trace, \
         trace_path.open("x", encoding="utf-8") as b_trace, \
         metrics_path.open("x", encoding="utf-8") as metrics_file:
        for epoch_index, epoch_batches in enumerate(epoch_schedules):
            epoch_number = epoch_index + 1
            epoch_metrics = []
            for batch_indices in epoch_batches:
                step += 1
                texts = []
                for index in batch_indices:
                    video_id = video_ids[index]
                    order = reference_orders[video_id]
                    caption_index = order[epoch_index % len(order)]
                    text = by_id[video_id]["captions"][caption_index]
                    texts.append(text)
                    seen[video_id] += 1
                    caption_seen[video_id][caption_index] += 1
                batch_features = train_features[batch_indices]
                captions = torch.stack([token_cache[text] for text in texts])
                evidence = trace_evidence(
                    step, epoch_index, batch_indices, video_ids, batch_features, captions, texts,
                )
                expected_line = a_trace.readline()
                if not expected_line:
                    raise RuntimeError(f"Caption-only trace ended before step {step}")
                assert_trace_equal(evidence, json.loads(expected_line))
                b_trace.write(json.dumps(evidence, ensure_ascii=False, sort_keys=True) + "\n")
                b_trace.flush()

                model.train()
                model.encoder.backbone.eval()
                optimizer.zero_grad(set_to_none=True)
                objective = cached_caption_infonce_objective(
                    model, batch_features.to("cuda"), captions.to("cuda"),
                )
                if any(not torch.isfinite(objective[key]) for key in
                       ("loss", "caption_loss", "alignment_loss")):
                    raise RuntimeError(f"Nonfinite objective at step {step}")
                objective["loss"].backward()
                if any(parameter.grad is None or not torch.isfinite(parameter.grad).all()
                       for _, parameter in trainable):
                    raise RuntimeError(f"Missing/nonfinite gradient at step {step}")
                optimizer.step()
                row = {key: float(value.detach()) for key, value in objective.items()}
                row.update({"step": step, "epoch": epoch_number})
                epoch_metrics.append(row)
                metrics_file.write(json.dumps(row, sort_keys=True) + "\n")
                metrics_file.flush()
                if step == 1 or step % 250 == 0 or step == expected_steps:
                    print(
                        f"[train:B] epoch={epoch_number}/{len(epoch_schedules)} "
                        f"step={step}/{expected_steps} total={row['loss']:.6f} "
                        f"caption={row['caption_loss']:.6f} align={row['alignment_loss']:.6f} "
                        f"gap={row['similarity_gap']:.6f} elapsed={time.monotonic()-started:.1f}s",
                        flush=True,
                    )
            means = {key: sum(row[key] for row in epoch_metrics) / len(epoch_metrics)
                     for key in ("loss", "caption_loss", "alignment_loss", "similarity_gap")}
            print(f"[epoch:B] {epoch_number}/{len(epoch_schedules)} "
                  f"caption={means['caption_loss']:.6f} align={means['alignment_loss']:.6f} "
                  f"gap={means['similarity_gap']:.6f}", flush=True)
            if epoch_number in milestone_set:
                caption_eval = evaluate_validation(
                    model, eval_features, eval_records, eval_references,
                    donors, epoch_number, step, "cuda",
                )
                retrieval_eval = retrieval_diagnostic(
                    model, eval_features, retrieval_text, retrieval_gold, "cuda",
                )
                checkpoint_path = args.output_root / f"checkpoint_epoch{epoch_number:03d}.pt"
                contract = {
                    "objective_version": 3,
                    "caption_objective": "causal_lm_cross_entropy",
                    "caption_policy": "balanced_all_references",
                    "alignment_objective": "infonce",
                    "alignment_weight": LAMBDA,
                    "info_nce_temperature": TEMPERATURE,
                    "false_negative_policy": "exact_caption",
                    "video_representation": "mean_visual_prefix_tokens_l2_normalized",
                    "text_representation": "frozen_gpt2_wte_masked_mean_l2_normalized",
                    "physical_batch_size": 8,
                    "protocol": "full-msvd-caption-infonce-ablation-v1",
                    "frozen_gpt2": True,
                    "frozen_vit_backbone": True,
                    "cached_frozen_backbone": True,
                    "legacy_visual_weights_used": False,
                }
                save_training_checkpoint(
                    checkpoint_path,
                    model=model,
                    optimizer=optimizer,
                    scheduler=None,
                    progress={"step": step, "epoch": epoch_number,
                              "epoch_complete": True,
                              "best_val": caption_eval["metrics"]["native_nll"]},
                    args={"gpt2_name": source_path, "vit_name": "vit_base_patch16_224",
                          "cond_mode": "prefix", "prefix_len": 4, "freeze_vit": True,
                          "seed": 123, "batch_size": 8, "max_len": 64, "lr": 3e-4,
                          "epochs": FORMAL_EPOCHS,
                          "caption_policy": "balanced_all_references"},
                    training_contract=contract,
                    dataset_contract=a_preflight["dataset_contract"],
                    loader_state={"batch_schedule_sha256": state_sha(
                        {"schedule": torch.tensor(schedule["batch_schedule"])},
                    ), "num_workers": 0},
                )
                checkpoint_paths.append(checkpoint_path)
                caption_eval["checkpoint"] = str(checkpoint_path.resolve())
                caption_eval["checkpoint_sha256"] = sha256_file(checkpoint_path)
                evaluations.append({"epoch": epoch_number, "step": step,
                                    "caption": caption_eval,
                                    "retrieval": retrieval_eval,
                                    "train_epoch_means": means})
                compact = caption_eval["metrics"]
                print(json.dumps({"epoch": epoch_number,
                    "caption_native_nll": compact["native_nll"],
                    "caption_mismatch_delta": compact["shuffled_minus_native_nll"],
                    "caption_gate": compact["primary_viability_gate"]["pass"],
                    "t2v_r1": retrieval_eval["true_labels"]["text_to_video"]["r_at_1"],
                    "retrieval_gap": retrieval_eval["similarity"]["gap"]},
                    ensure_ascii=False), flush=True)

        if not args.smoke and a_trace.readline():
            raise RuntimeError("Caption-only trace contains unexpected records after step 9900")

    if step != expected_steps:
        raise RuntimeError("InfoNCE run did not complete the locked step budget")
    if not args.smoke:
        if len(seen) != 1200 or set(seen.values()) != {66}:
            raise RuntimeError("InfoNCE video coverage differs from caption-only")
        if any(len(caption_seen[video_id]) != len(by_id[video_id]["captions"])
               for video_id in video_ids):
            raise RuntimeError("InfoNCE reference coverage differs from caption-only")

    final_hashes = component_hashes(model)
    if final_hashes["decoder"] != initial_hashes["decoder"] or final_hashes["backbone"] != initial_hashes["backbone"]:
        raise RuntimeError("Frozen model component changed")
    if final_hashes["projection"] == initial_hashes["projection"] or final_hashes["mapper"] == initial_hashes["mapper"]:
        raise RuntimeError("A trainable component did not change")
    if source_hashes != {name: sha256_file(ROOT / name) for name in SOURCE_FILES}:
        raise RuntimeError("Experiment source changed during the run")
    if any(sha256_file(path) != DATA_SHA256[split]
           for split, path in (("train", TRAIN), ("val", VAL))):
        raise RuntimeError("Dataset annotations changed during the run")

    if args.smoke:
        decision = {"decision": "smoke_only", "pass": False}
        a_retrieval = None
        smoke_hashes = final_hashes
        smoke_checkpoint = checkpoint_paths[-1]
        smoke_checkpoint_sha256 = sha256_file(smoke_checkpoint)
        load_strict(model, smoke_checkpoint)
        b_reload_ok = component_hashes(model) == smoke_hashes
        if not b_reload_ok or sha256_file(smoke_checkpoint) != smoke_checkpoint_sha256:
            raise RuntimeError("Smoke checkpoint reload/hash audit failed")
    else:
        b_checkpoint = checkpoint_paths[-1]
        b_checkpoint_hash = sha256_file(b_checkpoint)
        b_hashes_before_reload = final_hashes
        a_payload = load_strict(model, A_ROOT / "checkpoint_epoch066.pt")
        if a_payload["progress"] != {"step": 9900, "epoch": 66,
                                     "epoch_complete": True,
                                     "best_val": a_result["evaluations"][-1]["metrics"]["native_nll"]}:
            raise RuntimeError("Caption-only final checkpoint progress changed")
        a_retrieval = retrieval_diagnostic(
            model, val_features, text_embeddings, caption_gold_tensor, "cuda",
        )
        b_payload = load_strict(model, b_checkpoint)
        b_reload_ok = component_hashes(model) == b_hashes_before_reload
        if not b_reload_ok or sha256_file(b_checkpoint) != b_checkpoint_hash:
            raise RuntimeError("InfoNCE final checkpoint reload/hash audit failed")
        b_retrieval_reload = retrieval_diagnostic(
            model, val_features, text_embeddings, caption_gold_tensor, "cuda",
        )
        b_retrieval = evaluations[-1]["retrieval"]
        if b_retrieval_reload != b_retrieval:
            raise RuntimeError("InfoNCE retrieval changed after strict checkpoint reload")
        decision = primary_decision(
            a_result["evaluations"][-1]["metrics"],
            evaluations[-1]["caption"]["metrics"],
            a_retrieval,
            b_retrieval,
        )

    result = {
        "protocol": "full-msvd-caption-infonce-ablation-v1",
        "paper_ready": False,
        "smoke_test": args.smoke,
        "decision": decision,
        "intervention": preflight["intervention"],
        "common_config": a_result["config"],
        "a_caption_only": {
            "source_result": str((A_ROOT / "result.json").resolve()),
            "source_result_sha256": A_HASHES["result.json"],
            "final_caption": a_result["evaluations"][-1]["metrics"],
            "final_retrieval": a_retrieval,
        },
        "b_caption_infonce_evaluations": evaluations,
        "initial_component_hashes": initial_hashes,
        "final_component_hashes": final_hashes,
        "frozen_weights_unchanged": True,
        "trainable_modules_changed": True,
        "trace_exact_match": True,
        "trace_sha256": sha256_file(trace_path),
        "final_checkpoint_strict_reload": b_reload_ok,
        "elapsed_seconds": time.monotonic() - started,
        "peak_allocated_mb": torch.cuda.max_memory_allocated() / 2**20,
        "peak_reserved_mb": torch.cuda.max_memory_reserved() / 2**20,
        "source_and_data_unchanged": True,
        "test_split_read": False,
        "limits": [
            "One seed and one fixed InfoNCE hyperparameter setting.",
            "Validation-only causal ablation; test670 was not read.",
            "Retrieval uses the registered prefix/WTE diagnostic representation and is not yet a paper benchmark.",
            "Triage BLEU and token-F1 are not paper benchmark metrics.",
        ],
    }
    write_json(args.output_root / "result.json", result)
    print(json.dumps({
        "output": str(args.output_root.resolve()),
        "decision": decision,
        "final_caption": evaluations[-1]["caption"]["metrics"],
        "final_retrieval": evaluations[-1]["retrieval"],
        "final_checkpoint_sha256": evaluations[-1]["caption"]["checkpoint_sha256"],
    }, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
