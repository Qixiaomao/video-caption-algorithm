"""Run the registered MSVD caption+CPR C/D matrix."""
from __future__ import annotations

import os

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import argparse
from collections import Counter
import gc
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from scripts.diagnose_prefix_sensitivity import donor_permutation
from scripts.eval_retrieval import encode_texts
from scripts.run_caption_policy_ab import (
    raw_backbone_features,
    tokenize_caption,
    tokenize_references,
    write_json,
)
from scripts.run_full_msvd_balanced import (
    DATA_SHA256,
    TRAIN,
    VAL,
    evaluate_validation,
    trace_evidence,
)
from scripts.run_full_msvd_infonce_ablation import (
    A_ROOT,
    assert_trace_equal,
    load_strict,
    retrieval_diagnostic,
    verify_registered_baseline,
)
from scripts.run_overfit16 import component_hashes, configure_trainable
from src.cli.train_caption_mapper import set_seed
from src.data.data_loader import build_dataloader
from src.models.caption_model import VideoCaptionModel, build_caption_supervision
from src.training.checkpoint import (
    capture_rng_state,
    restore_rng_state,
    save_training_checkpoint,
    sha256_file,
)
from src.training.contrastive import (
    build_alignment_embeddings,
    exact_caption_positive_mask,
    similarity_diagnostics,
    symmetric_multi_positive_info_nce,
)
from src.training.cpr import compute_decoder_cpr_objective


SEEDS = (123, 124, 125)
ARMS = ("C", "D")
CPR_WEIGHT = 0.10
CPR_MARGIN = 0.10
ALIGNMENT_WEIGHT = 0.10
TEMPERATURE = 0.07
FORMAL_EPOCHS = 66
FORMAL_STEPS = 9900
MILESTONES = (5, 20, 40, 66)

REPEATED_ROOT = ROOT / "runs/repeated_seeds_20260908"
INFONCE_ROOT = ROOT / "runs/full_msvd_infonce_ablation_20260906"
PROTOCOL = ROOT / "experiments/q3_cpr_cd_matrix_protocol_20260912.md"

REGISTERED_INPUTS = {
    123: {
        "initial": (
            A_ROOT / "initial_model.pt",
            "b0deacbea2239da7dd8f6d815be13f75afc962cce68b0a84771784e0db08b1ff",
        ),
        "schedule": (
            A_ROOT / "schedule.json",
            "3520204739b0bac71a6cea9e5a2f26c4dbfee17e3b19790e2340e9efc054fd86",
        ),
        "trace": (
            A_ROOT / "trace.jsonl",
            "8b236e9e3a7f3fbe7ab6dff679326c810f600ae519f275a11271789a4de27a9b",
        ),
    },
    124: {
        "initial": (
            REPEATED_ROOT / "seed124/initial_model.pt",
            "a5bac1323a66b2cbd1dd866739794323892f9174ce6ab119b4d01c564b27525f",
        ),
        "schedule": (
            REPEATED_ROOT / "seed124/schedule.json",
            "085efe6fe2e3cb0d5ed58ac01cd73804eaa89109572fa3de8d39d2894e93f670",
        ),
        "trace": (
            REPEATED_ROOT / "seed124/A/trace.jsonl",
            "805aa6c967c00bd547c59a9db226a5d3cda74ce935ad3e2c29d536d27719210e",
        ),
    },
    125: {
        "initial": (
            REPEATED_ROOT / "seed125/initial_model.pt",
            "1aedef75b37ed243f0e43782e15bdca85a27f0e8e3eac6d6eaab27947673d04f",
        ),
        "schedule": (
            REPEATED_ROOT / "seed125/schedule.json",
            "94c26d811594eaae11a32c68ec31c7eb8180edcaf7a7ca03ce0d7db64cbf35e7",
        ),
        "trace": (
            REPEATED_ROOT / "seed125/A/trace.jsonl",
            "899ffea9f0020352dfacbe104fc336d436169fa7d0c9b09620b9f37fcd55ea0f",
        ),
    },
}

CACHE_INPUTS = {
    "train": (
        A_ROOT / "train_features.pt",
        "8cbc39e504be44865db302f87856b97b48318b6c32b60db7f8ed1ecb224a7ad6",
    ),
    "val": (
        A_ROOT / "val_features.pt",
        "5ac1dc53d8e03fcdf3c3c8abae98e1a5c518774ffe48a00eacbb3648503aeee4",
    ),
}

SOURCE_FILES = (
    "experiments/q3_cpr_cd_matrix_protocol_20260912.md",
    "scripts/run_cpr_cd_matrix.py",
    "scripts/run_full_msvd_balanced.py",
    "scripts/run_full_msvd_infonce_ablation.py",
    "scripts/run_caption_policy_ab.py",
    "scripts/run_overfit16.py",
    "scripts/eval_retrieval.py",
    "src/models/caption_model.py",
    "src/models/text_decoder.py",
    "src/training/checkpoint.py",
    "src/training/contrastive.py",
    "src/training/cpr.py",
)


def verify_registered_inputs() -> None:
    verify_registered_baseline()
    for seed in SEEDS:
        for label, (path, expected_hash) in REGISTERED_INPUTS[seed].items():
            if not path.is_file() or sha256_file(path) != expected_hash:
                raise RuntimeError(f"registered seed-{seed} {label} changed: {path}")
    for label, (path, expected_hash) in CACHE_INPUTS.items():
        if not path.is_file() or sha256_file(path) != expected_hash:
            raise RuntimeError(f"registered {label} cache changed: {path}")
    for split, path in (("train", TRAIN), ("val", VAL)):
        if sha256_file(path) != DATA_SHA256[split]:
            raise RuntimeError(f"pinned {split} annotations changed")


def load_registered_schedule(seed: int) -> dict[str, Any]:
    raw = json.loads(REGISTERED_INPUTS[seed]["schedule"][0].read_text(encoding="utf-8"))
    if seed == 123:
        schedule = {"batches": raw["batch_schedule"], "orders": raw["reference_orders"]}
    else:
        schedule = {"batches": raw["batches"], "orders": raw["orders"]}
    if len(schedule["batches"]) != FORMAL_EPOCHS:
        raise RuntimeError(f"seed {seed} schedule has the wrong epoch count")
    if any(len(epoch) != 150 for epoch in schedule["batches"]):
        raise RuntimeError(f"seed {seed} schedule has the wrong batch count")
    return schedule


def same_rng_state(left: dict[str, Any], right: dict[str, Any]) -> bool:
    if left["python"] != right["python"]:
        return False
    if not torch.equal(left["torch_cpu"], right["torch_cpu"]):
        return False
    left_cuda = left.get("torch_cuda") or []
    right_cuda = right.get("torch_cuda") or []
    return len(left_cuda) == len(right_cuda) and all(
        torch.equal(a, b) for a, b in zip(left_cuda, right_cuda)
    )


def cached_cd_objective(
    model: VideoCaptionModel,
    raw_features: torch.Tensor,
    caption_ids: torch.Tensor,
    video_ids: list[str],
    arm: str,
) -> dict[str, torch.Tensor]:
    if arm not in ARMS:
        raise ValueError(f"unknown C/D arm: {arm}")

    pad_id = model.decoder.tokenizer.pad_token_id
    video_embedding = model.proj(model.encoder.proj(raw_features))
    attention_mask, labels = build_caption_supervision(caption_ids, pad_id)
    visual_prefix = model.decoder.build_visual_prefix(video_embedding)
    decoder_result = model.decoder(
        video_embedding,
        caption_ids,
        attention_mask,
        labels,
        visual_prefix=visual_prefix,
    )
    caption_loss = decoder_result["loss"]
    zero = caption_loss.new_zeros(())
    alignment_loss = zero
    diagnostics = {
        "positive_similarity": zero,
        "negative_similarity": zero,
        "similarity_gap": zero,
    }

    if arm == "D":
        with torch.no_grad():
            token_embeddings = model.decoder.model.transformer.wte(caption_ids)
        video_embeddings, text_embeddings = build_alignment_embeddings(
            visual_prefix,
            token_embeddings,
            caption_ids,
            pad_id,
        )
        positive_mask = exact_caption_positive_mask(caption_ids, pad_id)
        alignment_loss = symmetric_multi_positive_info_nce(
            video_embeddings,
            text_embeddings,
            temperature=TEMPERATURE,
            positive_mask=positive_mask,
        )
        diagnostics = similarity_diagnostics(
            video_embeddings, text_embeddings, positive_mask
        )

    cpr = compute_decoder_cpr_objective(
        decoder=model.decoder,
        video_embedding=video_embedding,
        visual_prefix=visual_prefix,
        matched_logits=decoder_result["logits"],
        caption_ids=caption_ids,
        attention_mask=attention_mask,
        labels=labels,
        pad_id=pad_id,
        video_ids=video_ids,
        margin=CPR_MARGIN,
        negative_policy="cyclic_derangement",
    )
    return {
        "loss": caption_loss
        + (ALIGNMENT_WEIGHT * alignment_loss if arm == "D" else zero)
        + CPR_WEIGHT * cpr["cpr_loss"],
        "caption_loss": caption_loss,
        "alignment_loss": alignment_loss,
        **cpr,
        **diagnostics,
    }


def smoke_objective_check(
    model: VideoCaptionModel,
    features: torch.Tensor,
    captions: torch.Tensor,
    video_ids: list[str],
    seed: int,
) -> dict[str, Any]:
    model.train()
    model.encoder.backbone.eval()
    configure_trainable(model)

    outputs = {}
    rng_states = {}
    for arm in ARMS:
        model.zero_grad(set_to_none=True)
        set_seed(seed)
        out = cached_cd_objective(model, features, captions, video_ids, arm)
        out["loss"].backward()
        rng_states[arm] = capture_rng_state()
        outputs[arm] = {key: value.detach().clone() for key, value in out.items()}

    paired_fields = (
        "caption_loss",
        "cpr_loss",
        "matched_nll",
        "mismatched_nll",
        "cpr_gap",
        "cpr_eligible_fraction",
    )
    unequal = [
        key for key in paired_fields if not torch.equal(outputs["C"][key], outputs["D"][key])
    ]
    if unequal:
        raise RuntimeError(f"C/D initial CPR path differs: {unequal}")
    if not same_rng_state(rng_states["C"], rng_states["D"]):
        raise RuntimeError("C/D objective changes main RNG consumption")
    if outputs["C"]["cpr_eligible_fraction"].item() <= 0:
        raise RuntimeError("smoke batch has no eligible CPR negatives")
    if outputs["C"]["cpr_loss"].item() <= 0:
        raise RuntimeError("smoke batch does not activate the CPR hinge")

    model.zero_grad(set_to_none=True)
    set_seed(seed)
    cpr_only = cached_cd_objective(model, features, captions, video_ids, "C")["cpr_loss"]
    cpr_only.backward()
    for name, module in (("encoder.proj", model.encoder.proj), ("decoder.mapper", model.decoder.mapper)):
        if not any(
            parameter.grad is not None
            and torch.isfinite(parameter.grad).all()
            and parameter.grad.abs().sum() > 0
            for parameter in module.parameters()
        ):
            raise RuntimeError(f"CPR gradient does not reach {name}")
    if any(parameter.grad is not None for parameter in model.decoder.model.parameters()):
        raise RuntimeError("CPR gradient reached frozen GPT-2 parameters")
    model.zero_grad(set_to_none=True)

    return {
        "c_d_caption_ce_equal": True,
        "c_d_cpr_values_equal": True,
        "c_d_rng_consumption_equal": True,
        "cpr_gradient_reaches_projection_mapper": True,
        "frozen_gpt2_has_no_gradient": True,
        "eligible_fraction": float(outputs["C"]["cpr_eligible_fraction"]),
        "initial_cpr_loss": float(outputs["C"]["cpr_loss"]),
        "initial_cpr_gap": float(outputs["C"]["cpr_gap"]),
    }


def training_contract(arm: str) -> dict[str, Any]:
    return {
        "objective_version": 4,
        "caption_objective": "causal_lm_cross_entropy",
        "caption_policy": "balanced_all_references",
        "alignment_objective": "infonce" if arm == "D" else "none",
        "alignment_weight": ALIGNMENT_WEIGHT if arm == "D" else 0.0,
        "info_nce_temperature": TEMPERATURE,
        "false_negative_policy": "exact_caption",
        "cpr_weight": CPR_WEIGHT,
        "cpr_margin": CPR_MARGIN,
        "cpr_negative_policy": "cyclic_derangement",
        "physical_batch_size": 8,
        "video_representation": "mean_visual_prefix_tokens_l2_normalized",
        "text_representation": "frozen_gpt2_wte_masked_mean_l2_normalized",
        "frozen_gpt2": True,
        "frozen_vit_backbone": True,
        "cached_frozen_backbone": True,
        "cpr_wrong_decoder_activation_checkpoint": True,
        "cpr_wrong_decoder_isolated_rng": True,
        "inference_extra_parameters": 0,
    }


def run_arm(
    *,
    model: VideoCaptionModel,
    arm: str,
    seed: int,
    folder: Path,
    initial_path: Path,
    initial_hashes: dict[str, str],
    schedule: dict[str, Any],
    records: list[dict[str, Any]],
    features: torch.Tensor,
    val_records: list[dict[str, Any]],
    val_features: torch.Tensor,
    references: dict[str, list[dict[str, Any]]],
    text_embeddings: torch.Tensor,
    caption_gold: torch.Tensor,
    token_cache: dict[str, torch.Tensor],
    source_hashes: dict[str, str],
    baseline_preflight: dict[str, Any],
    smoke: bool,
) -> dict[str, Any]:
    folder.mkdir()
    load_strict(model, initial_path)
    if component_hashes(model) != initial_hashes:
        raise RuntimeError(f"seed {seed} arm {arm} initial state mismatch")
    trainable = configure_trainable(model)
    optimizer = torch.optim.AdamW(
        [parameter for _, parameter in trainable], lr=3e-4, weight_decay=0.01
    )

    epoch_batches = [schedule["batches"][0][:2]] if smoke else schedule["batches"]
    milestones = {1} if smoke else set(MILESTONES)
    expected_steps = 2 if smoke else FORMAL_STEPS
    ids = [str(record["video_id"]) for record in records]
    visits: Counter[str] = Counter()
    coverage = {video_id: Counter() for video_id in ids}
    evaluations = []
    minimum_eligible_fraction = 1.0
    set_seed(seed)
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    step = 0
    donors = donor_permutation(len(val_records), seed=321)

    historical_trace_path = REGISTERED_INPUTS[seed]["trace"][0]
    paired_trace_path = folder.parent / "C/trace.jsonl" if arm == "D" else None
    historical_trace = historical_trace_path.open("r", encoding="utf-8")
    paired_trace = paired_trace_path.open("r", encoding="utf-8") if paired_trace_path else None
    try:
        with (folder / "trace.jsonl").open("x", encoding="utf-8") as trace, (
            folder / "train_metrics.jsonl"
        ).open("x", encoding="utf-8") as log:
            for epoch, batches in enumerate(epoch_batches, 1):
                epoch_caption_losses = []
                for indices in batches:
                    step += 1
                    targets = []
                    selected_ids = []
                    for index in indices:
                        video_id = ids[index]
                        order = schedule["orders"][video_id]
                        caption_index = order[(epoch - 1) % len(order)]
                        targets.append(records[index]["captions"][caption_index])
                        selected_ids.append(video_id)
                        visits[video_id] += 1
                        coverage[video_id][caption_index] += 1

                    batch_features = features[indices]
                    captions = torch.stack([token_cache[text] for text in targets])
                    evidence = trace_evidence(
                        step,
                        epoch - 1,
                        indices,
                        ids,
                        batch_features,
                        captions,
                        targets,
                    )
                    historical_line = historical_trace.readline()
                    if not historical_line:
                        raise RuntimeError("historical trace ended before the registered schedule")
                    assert_trace_equal(evidence, json.loads(historical_line))
                    if paired_trace is not None:
                        paired_line = paired_trace.readline()
                        if not paired_line:
                            raise RuntimeError("paired C trace ended before D")
                        assert_trace_equal(evidence, json.loads(paired_line))
                    trace.write(json.dumps(evidence, ensure_ascii=False, sort_keys=True) + "\n")
                    trace.flush()

                    model.train()
                    model.encoder.backbone.eval()
                    optimizer.zero_grad(set_to_none=True)
                    output = cached_cd_objective(
                        model,
                        batch_features.cuda(),
                        captions.cuda(),
                        selected_ids,
                        arm,
                    )
                    required = (
                        "loss",
                        "caption_loss",
                        "alignment_loss",
                        "cpr_loss",
                        "matched_nll",
                        "mismatched_nll",
                        "cpr_gap",
                        "cpr_eligible_fraction",
                    )
                    if not all(torch.isfinite(output[key]).item() for key in required):
                        raise RuntimeError(
                            f"nonfinite objective seed={seed} arm={arm} step={step}"
                        )
                    output["loss"].backward()
                    if any(
                        parameter.grad is None or not torch.isfinite(parameter.grad).all()
                        for _, parameter in trainable
                    ):
                        raise RuntimeError("missing or nonfinite trainable gradient")
                    optimizer.step()

                    row = {key: float(value.detach()) for key, value in output.items()}
                    row.update(step=step, epoch=epoch)
                    log.write(json.dumps(row, sort_keys=True) + "\n")
                    log.flush()
                    epoch_caption_losses.append(row["caption_loss"])
                    minimum_eligible_fraction = min(
                        minimum_eligible_fraction, row["cpr_eligible_fraction"]
                    )

                print(
                    f"[seed={seed} {arm}] epoch={epoch}/{len(epoch_batches)} "
                    f"step={step} CE={sum(epoch_caption_losses) / len(epoch_caption_losses):.6f}",
                    flush=True,
                )
                if epoch in milestones:
                    caption = evaluate_validation(
                        model,
                        val_features,
                        val_records,
                        references,
                        donors,
                        epoch,
                        step,
                        "cuda",
                    )
                    retrieval = retrieval_diagnostic(
                        model, val_features, text_embeddings, caption_gold, "cuda"
                    )
                    checkpoint_path = folder / f"checkpoint_epoch{epoch:03d}.pt"
                    save_training_checkpoint(
                        checkpoint_path,
                        model=model,
                        optimizer=optimizer,
                        scheduler=None,
                        progress={
                            "step": step,
                            "epoch": epoch,
                            "epoch_complete": True,
                            "best_val": caption["metrics"]["native_nll"],
                        },
                        args={
                            "gpt2_name": baseline_preflight["model_source"]["path"],
                            "vit_name": "vit_base_patch16_224",
                            "cond_mode": "prefix",
                            "prefix_len": 4,
                            "seed": seed,
                            "arm": arm,
                            "batch_size": 8,
                            "max_len": 64,
                            "lr": 3e-4,
                            "epochs": len(epoch_batches),
                        },
                        training_contract=training_contract(arm),
                        dataset_contract=baseline_preflight["dataset_contract"],
                        loader_state={
                            "registered_schedule_path": str(
                                REGISTERED_INPUTS[seed]["schedule"][0].resolve()
                            ),
                            "registered_schedule_sha256": REGISTERED_INPUTS[seed]["schedule"][1],
                            "num_workers": 0,
                        },
                    )
                    evaluation = {
                        "epoch": epoch,
                        "step": step,
                        "caption": caption,
                        "retrieval": retrieval,
                        "checkpoint": str(checkpoint_path.resolve()),
                        "checkpoint_sha256": sha256_file(checkpoint_path),
                    }
                    evaluations.append(evaluation)
                    write_json(folder / f"evaluation_epoch{epoch:03d}.json", evaluation)
                    print(
                        json.dumps(
                            {
                                "seed": seed,
                                "arm": arm,
                                "epoch": epoch,
                                "nll": caption["metrics"]["native_nll"],
                                "caption_gate": caption["metrics"]["primary_viability_gate"]["pass"],
                                "grounding_gap": caption["metrics"]["shuffled_minus_native_nll"],
                                "t2v_r1": retrieval["true_labels"]["text_to_video"]["r_at_1"],
                            },
                            sort_keys=True,
                        ),
                        flush=True,
                    )

        if not smoke and historical_trace.readline():
            raise RuntimeError("historical trace contains extra formal steps")
        if paired_trace is not None and not smoke and paired_trace.readline():
            raise RuntimeError("paired C trace contains extra formal steps")
    finally:
        historical_trace.close()
        if paired_trace is not None:
            paired_trace.close()

    if step != expected_steps:
        raise RuntimeError(f"expected {expected_steps} steps, got {step}")
    if not smoke:
        if len(visits) != 1200 or set(visits.values()) != {66}:
            raise RuntimeError("formal video coverage is incomplete")
        if not all(
            len(coverage[video_id]) == len(records[index]["captions"])
            for index, video_id in enumerate(ids)
        ):
            raise RuntimeError("formal caption-reference coverage is incomplete")

    final_hashes = component_hashes(model)
    if not all(final_hashes[key] == initial_hashes[key] for key in ("decoder", "backbone")):
        raise RuntimeError("a frozen component changed")
    if not all(final_hashes[key] != initial_hashes[key] for key in ("projection", "mapper")):
        raise RuntimeError("a registered trainable component did not change")
    load_strict(model, Path(evaluations[-1]["checkpoint"]))
    if component_hashes(model) != final_hashes:
        raise RuntimeError("strict checkpoint reload changed component hashes")
    if retrieval_diagnostic(
        model, val_features, text_embeddings, caption_gold, "cuda"
    ) != evaluations[-1]["retrieval"]:
        raise RuntimeError("strict checkpoint reload changed retrieval diagnostics")
    if source_hashes != {name: sha256_file(ROOT / name) for name in SOURCE_FILES}:
        raise RuntimeError("registered source changed during execution")
    for split, path in (("train", TRAIN), ("val", VAL)):
        if sha256_file(path) != DATA_SHA256[split]:
            raise RuntimeError(f"{split} annotations changed during execution")

    result = {
        "seed": seed,
        "arm": arm,
        "smoke_test": smoke,
        "paper_ready": False,
        "test_split_read": False,
        "steps": step,
        "initial_component_hashes": initial_hashes,
        "final_component_hashes": final_hashes,
        "training_contract": training_contract(arm),
        "evaluations": evaluations,
        "trace_sha256": sha256_file(folder / "trace.jsonl"),
        "historical_trace_prefix_match": True,
        "minimum_cpr_eligible_fraction": minimum_eligible_fraction,
        "final_checkpoint_strict_reload": True,
        "frozen_weights_unchanged": True,
        "source_and_data_unchanged": True,
        "video_visits": dict(visits),
        "unique_reference_indices": sum(len(item) for item in coverage.values()),
        "elapsed_seconds": time.monotonic() - started,
        "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
        "peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
    }
    write_json(folder / "result.json", result)
    del optimizer
    gc.collect()
    torch.cuda.empty_cache()
    return result


def historical_endpoint(seed: int, arm: str) -> dict[str, Any]:
    if seed == 123:
        result = json.loads((INFONCE_ROOT / "result.json").read_text(encoding="utf-8"))
        if arm == "A":
            return {
                "caption": result["a_caption_only"]["final_caption"],
                "retrieval": result["a_caption_only"]["final_retrieval"],
            }
        return result["b_caption_infonce_evaluations"][-1]
    path = REPEATED_ROOT / f"seed{seed}/{arm}/result.json"
    return json.loads(path.read_text(encoding="utf-8"))["evaluations"][-1]


def formal_decision(all_results: dict[int, dict[str, dict[str, Any]]]) -> dict[str, Any]:
    rows = []
    for seed in SEEDS:
        for new_arm, old_arm in (("C", "A"), ("D", "B")):
            new = all_results[seed][new_arm]["evaluations"][-1]
            old = historical_endpoint(seed, old_arm)
            new_caption = new["caption"]["metrics"]
            old_caption = old["caption"]["metrics"]
            rows.append(
                {
                    "seed": seed,
                    "stratum": f"{new_arm}-{old_arm}",
                    "caption_gate": bool(new_caption["primary_viability_gate"]["pass"]),
                    "delta_nll": new_caption["native_nll"] - old_caption["native_nll"],
                    "delta_token_f1": new_caption["mean_max_reference_token_f1"]
                    - old_caption["mean_max_reference_token_f1"],
                    "delta_grounding_gap": new_caption["shuffled_minus_native_nll"]
                    - old_caption["shuffled_minus_native_nll"],
                    "delta_t2v_r1": new["retrieval"]["true_labels"]["text_to_video"]["r_at_1"]
                    - old["retrieval"]["true_labels"]["text_to_video"]["r_at_1"],
                }
            )

    strata = {}
    for stratum in ("C-A", "D-B"):
        selected = [row for row in rows if row["stratum"] == stratum]
        strata[stratum] = {
            "mean_delta_grounding_gap": sum(row["delta_grounding_gap"] for row in selected)
            / len(selected),
            "positive_grounding_seeds": sum(row["delta_grounding_gap"] > 0 for row in selected),
        }
    gates = {
        "all_caption_gates_pass": all(row["caption_gate"] for row in rows),
        "all_nll_within_plus_0_15": all(row["delta_nll"] <= 0.15 for row in rows),
        "all_token_f1_within_minus_0_05": all(
            row["delta_token_f1"] >= -0.05 for row in rows
        ),
        "both_strata_mean_grounding_positive": all(
            item["mean_delta_grounding_gap"] > 0 for item in strata.values()
        ),
        "both_strata_at_least_two_positive_seeds": all(
            item["positive_grounding_seeds"] >= 2 for item in strata.values()
        ),
    }
    gates["pass"] = all(gates.values())
    gates["decision"] = "retain_cpr_in_main" if gates["pass"] else "appendix_or_remove_cpr"
    return {"rows": rows, "strata": strata, "gates": gates}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.output_root.exists():
        raise FileExistsError("refusing to overwrite an existing experiment directory")
    if not torch.cuda.is_available():
        raise RuntimeError("the registered C/D matrix requires the local CUDA GPU")

    verify_registered_inputs()
    baseline_result, baseline_preflight = verify_registered_baseline()
    del baseline_result
    source_hashes = {name: sha256_file(ROOT / name) for name in SOURCE_FILES}
    records = json.loads(TRAIN.read_text(encoding="utf-8"))
    validation = json.loads(VAL.read_text(encoding="utf-8"))
    if len(records) != 1200 or len(validation) != 100:
        raise RuntimeError("unexpected full-MSVD split sizes")

    train_cache = torch.load(CACHE_INPUTS["train"][0], weights_only=True, map_location="cpu")
    val_cache = torch.load(CACHE_INPUTS["val"][0], weights_only=True, map_location="cpu")
    if train_cache["video_ids"] != [record["video_id"] for record in records]:
        raise RuntimeError("training cache order changed")
    if val_cache["video_ids"] != [record["video_id"] for record in validation]:
        raise RuntimeError("validation cache order changed")

    args.output_root.mkdir(parents=True)
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    all_results: dict[int, dict[str, dict[str, Any]]] = {}
    pair_summaries = []
    for seed in SEEDS:
        seed_folder = args.output_root / f"seed{seed}"
        seed_folder.mkdir()
        schedule = load_registered_schedule(seed)
        initial_path = REGISTERED_INPUTS[seed]["initial"][0]

        set_seed(seed)
        model = VideoCaptionModel(
            gpt2_name=baseline_preflight["model_source"]["path"],
            freeze_vit=True,
            vit_enable_torch_compile=False,
        )
        load_strict(model, initial_path)
        initial_hashes = component_hashes(model)
        model = model.cuda().eval()

        loader = build_dataloader(
            str(TRAIN),
            model.decoder.tokenizer,
            batch_size=8,
            max_len=64,
            num_frame=8,
            image_size=224,
            shuffle=False,
            num_wokers=0,
            random_caption=False,
        )
        first_batch = next(iter(loader))
        with torch.inference_mode():
            direct_features = raw_backbone_features(model.encoder, first_batch["video"].cuda())
            cache_error = float(
                (direct_features.cpu() - train_cache["features"][:8]).abs().max()
            )
        if cache_error > 1e-6:
            raise RuntimeError(f"seed {seed} frozen feature cache mismatch: {cache_error}")

        token_cache = {
            text: tokenize_caption(model.decoder.tokenizer, text)
            for record in records
            for text in record["captions"]
        }
        selected_validation = validation[:8] if args.smoke else validation
        selected_val_features = val_cache["features"][: len(selected_validation)]
        references = {
            record["video_id"]: tokenize_references(
                model.decoder.tokenizer, record["captions"]
            )
            for record in selected_validation
        }
        all_val_captions = [
            text for record in selected_validation for text in record["captions"]
        ]
        text_embeddings = encode_texts(
            model, all_val_captions, max_length=64, batch_size=128, device="cuda"
        )
        caption_gold = torch.tensor(
            [
                index
                for index, record in enumerate(selected_validation)
                for _ in record["captions"]
            ]
        )

        smoke_checks = None
        if args.smoke:
            indices = schedule["batches"][0][0]
            targets = []
            selected_ids = []
            for index in indices:
                video_id = str(records[index]["video_id"])
                caption_index = schedule["orders"][video_id][0]
                targets.append(records[index]["captions"][caption_index])
                selected_ids.append(video_id)
            smoke_captions = torch.stack([token_cache[text] for text in targets]).cuda()
            smoke_checks = smoke_objective_check(
                model,
                train_cache["features"][indices].cuda(),
                smoke_captions,
                selected_ids,
                seed,
            )

        preflight = {
            "seed": seed,
            "smoke_test": args.smoke,
            "paper_ready": False,
            "test_split_read": False,
            "source_sha256": source_hashes,
            "git_head": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
            ).strip(),
            "initial_checkpoint": str(initial_path.resolve()),
            "initial_checkpoint_sha256": sha256_file(initial_path),
            "initial_component_hashes": initial_hashes,
            "schedule": str(REGISTERED_INPUTS[seed]["schedule"][0].resolve()),
            "schedule_sha256": REGISTERED_INPUTS[seed]["schedule"][1],
            "historical_trace": str(REGISTERED_INPUTS[seed]["trace"][0].resolve()),
            "historical_trace_sha256": REGISTERED_INPUTS[seed]["trace"][1],
            "cache_error": cache_error,
            "smoke_checks": smoke_checks,
            "runtime": {
                "python": sys.version,
                "torch": str(torch.__version__),
                "gpu": torch.cuda.get_device_name(),
                "physical_gpu_mib": torch.cuda.get_device_properties(0).total_memory / 2**20,
                "deterministic_algorithms": True,
                "tf32": False,
            },
        }
        write_json(seed_folder / "preflight.json", preflight)

        seed_results = {}
        for arm in ARMS:
            seed_results[arm] = run_arm(
                model=model,
                arm=arm,
                seed=seed,
                folder=seed_folder / arm,
                initial_path=initial_path,
                initial_hashes=initial_hashes,
                schedule=schedule,
                records=records,
                features=train_cache["features"],
                val_records=selected_validation,
                val_features=selected_val_features,
                references=references,
                text_embeddings=text_embeddings,
                caption_gold=caption_gold,
                token_cache=token_cache,
                source_hashes=source_hashes,
                baseline_preflight=baseline_preflight,
                smoke=args.smoke,
            )
        if seed_results["C"]["trace_sha256"] != seed_results["D"]["trace_sha256"]:
            raise RuntimeError(f"seed {seed} C/D trace hashes differ")
        pair = {
            "seed": seed,
            "smoke_test": args.smoke,
            "trace_match": True,
            "historical_trace_prefix_match": True,
            "initial_component_hashes": initial_hashes,
            "smoke_checks": smoke_checks,
            "results": {
                arm: str((seed_folder / arm / "result.json").resolve()) for arm in ARMS
            },
        }
        write_json(seed_folder / "pair_result.json", pair)
        pair_summaries.append(pair)
        all_results[seed] = seed_results
        print(json.dumps(pair, sort_keys=True), flush=True)

        del model, token_cache, text_embeddings
        gc.collect()
        torch.cuda.empty_cache()

    decision = None if args.smoke else formal_decision(all_results)
    completion = {
        "completed": True,
        "seeds": list(SEEDS),
        "arms": list(ARMS),
        "pairs": pair_summaries,
        "formal_decision": decision,
        "smoke_test": args.smoke,
        "paper_ready": False,
        "test_split_read": False,
    }
    write_json(args.output_root / "completion.json", completion)
    print(json.dumps(completion, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
