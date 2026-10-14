"""Hierarchical bootstrap summary for the frozen formal efficiency matrix."""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
FORMAL_ROOT = ROOT / "runs/formal_efficiency_matrix_20260909"
FORMAL_RESULT = FORMAL_ROOT / "result.json"
FORMAL_RESULT_SHA256 = "745ac386ec8c5b769df6fca6baa64de2a1a427d7c05aad787c2e6239d20edaa5"
PROTOCOL = ROOT / "experiments/formal_efficiency_uncertainty_protocol_20260912.md"
OUTPUT = ROOT / "runs/formal_efficiency_uncertainty_20260912"
CONFIG_IDS = ["E0", "E1", "E2", "E3", "E4"]
BATCH_SIZES = [1, 2, 4, 8, 16]
BLOCKS = [1, 2, 3]
ITERATIONS = 50
RESAMPLES = 10_000
BASE_SEED = 123
ABSOLUTE_METRICS = [
    "end_to_end_ms",
    "samples_per_second",
    "vit_encoder_ms",
    "peak_allocated_mib",
    "peak_reserved_mib",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def derived_seed(label: str) -> int:
    digest = hashlib.sha256(f"{BASE_SEED}:{label}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big", signed=False)


def percentile_interval(values: np.ndarray) -> tuple[float, float]:
    low, high = np.quantile(values, [0.025, 0.975])
    return float(low), float(high)


def hierarchical_median_interval(
    by_block: dict[int, np.ndarray],
    *,
    label: str,
    resamples: int = RESAMPLES,
) -> dict:
    block_ids = sorted(by_block)
    if block_ids != BLOCKS:
        raise ValueError(f"Expected blocks {BLOCKS}, found {block_ids}")
    pooled = np.concatenate([by_block[block] for block in block_ids])
    rng = np.random.default_rng(derived_seed(label))
    block_matrix = np.stack([by_block[block] for block in block_ids])
    if len({len(values) for values in by_block.values()}) != 1:
        raise ValueError("Vectorized bootstrap requires equal iteration counts per block")
    iterations = block_matrix.shape[1]
    sampled_block_indices = rng.integers(0, len(block_ids), size=(resamples, len(block_ids)))
    samples = np.empty((resamples, len(block_ids), iterations), dtype=np.float64)
    for slot in range(len(block_ids)):
        sampled_iterations = rng.integers(0, iterations, size=(resamples, iterations))
        samples[:, slot, :] = block_matrix[
            sampled_block_indices[:, slot, None], sampled_iterations
        ]
    estimates = np.median(samples.reshape(resamples, -1), axis=1)
    low, high = percentile_interval(estimates)
    return {
        "point": float(np.median(pooled)),
        "ci95_low": low,
        "ci95_high": high,
        "resamples": resamples,
        "method": "hierarchical_block_then_iteration_percentile_bootstrap",
        "seed": derived_seed(label),
    }


def contrast_value(reference: float, candidate: float, effect: str) -> float:
    if effect == "latency_reduction_percent":
        return (reference - candidate) / reference * 100.0
    if effect == "throughput_improvement_percent":
        return (candidate - reference) / reference * 100.0
    if effect == "memory_reduction_mib":
        return reference - candidate
    raise ValueError(f"Unsupported effect: {effect}")


def hierarchical_contrast_interval(
    reference: dict[int, np.ndarray],
    candidate: dict[int, np.ndarray],
    *,
    effect: str,
    label: str,
    resamples: int = RESAMPLES,
) -> dict:
    block_ids = sorted(set(reference) & set(candidate))
    if block_ids != BLOCKS:
        raise ValueError(f"Expected paired blocks {BLOCKS}, found {block_ids}")
    reference_pooled = np.concatenate([reference[block] for block in block_ids])
    candidate_pooled = np.concatenate([candidate[block] for block in block_ids])
    point = contrast_value(float(np.median(reference_pooled)), float(np.median(candidate_pooled)), effect)
    rng = np.random.default_rng(derived_seed(label))
    reference_matrix = np.stack([reference[block] for block in block_ids])
    candidate_matrix = np.stack([candidate[block] for block in block_ids])
    if reference_matrix.shape != candidate_matrix.shape:
        raise ValueError("Paired bootstrap requires matching block/iteration shapes")
    iterations = reference_matrix.shape[1]
    sampled_block_indices = rng.integers(0, len(block_ids), size=(resamples, len(block_ids)))
    reference_samples = np.empty((resamples, len(block_ids), iterations), dtype=np.float64)
    candidate_samples = np.empty_like(reference_samples)
    for slot in range(len(block_ids)):
        reference_iterations = rng.integers(0, iterations, size=(resamples, iterations))
        candidate_iterations = rng.integers(0, iterations, size=(resamples, iterations))
        reference_samples[:, slot, :] = reference_matrix[
            sampled_block_indices[:, slot, None], reference_iterations
        ]
        candidate_samples[:, slot, :] = candidate_matrix[
            sampled_block_indices[:, slot, None], candidate_iterations
        ]
    reference_medians = np.median(reference_samples.reshape(resamples, -1), axis=1)
    candidate_medians = np.median(candidate_samples.reshape(resamples, -1), axis=1)
    if effect == "latency_reduction_percent":
        estimates = (reference_medians - candidate_medians) / reference_medians * 100.0
    elif effect == "throughput_improvement_percent":
        estimates = (candidate_medians - reference_medians) / reference_medians * 100.0
    elif effect == "memory_reduction_mib":
        estimates = reference_medians - candidate_medians
    else:
        raise ValueError(f"Unsupported effect: {effect}")
    low, high = percentile_interval(estimates)
    return {
        "point": point,
        "ci95_low": low,
        "ci95_high": high,
        "directionally_supported": low > 0.0,
        "resamples": resamples,
        "method": "paired_block_hierarchical_percentile_bootstrap",
        "seed": derived_seed(label),
    }


def load_cells() -> tuple[dict, dict[tuple[str, int], dict[int, list[dict]]]]:
    if sha256_file(FORMAL_RESULT) != FORMAL_RESULT_SHA256:
        raise RuntimeError("Frozen formal efficiency result hash changed")
    formal_result = json.loads(FORMAL_RESULT.read_text(encoding="utf-8"))
    if formal_result.get("decision") != "formal_matrix_complete" or not formal_result.get("paper_ready"):
        raise RuntimeError("Formal efficiency matrix is not complete")
    paths = sorted((FORMAL_ROOT / "cells").glob("*.json"))
    if len(paths) != 75:
        raise RuntimeError(f"Expected 75 formal cells, found {len(paths)}")
    groups: dict[tuple[str, int], dict[int, list[dict]]] = defaultdict(dict)
    for path in paths:
        cell = json.loads(path.read_text(encoding="utf-8"))
        config_id = cell["config"]["id"]
        batch_size = int(cell["batch_size"])
        block = int(cell["block"])
        if cell.get("status") != "complete" or len(cell.get("iterations", [])) != ITERATIONS:
            raise RuntimeError(f"Incomplete or malformed formal cell: {path}")
        if config_id not in CONFIG_IDS or batch_size not in BATCH_SIZES or block not in BLOCKS:
            raise RuntimeError(f"Unexpected formal cell identity: {path}")
        if block in groups[(config_id, batch_size)]:
            raise RuntimeError(f"Duplicate formal cell: {config_id}, {batch_size}, {block}")
        groups[(config_id, batch_size)][block] = cell["iterations"]
    if len(groups) != 25 or any(sorted(by_block) != BLOCKS for by_block in groups.values()):
        raise RuntimeError("Formal matrix does not contain the complete 5 x 5 x 3 design")
    return formal_result, groups


def metric_blocks(groups, config_id: str, batch_size: int, metric: str) -> dict[int, np.ndarray]:
    return {
        block: np.asarray([float(row[metric]) for row in rows], dtype=np.float64)
        for block, rows in groups[(config_id, batch_size)].items()
    }


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty CSV: {path}")
    with path.open("x", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def flatten_interval(prefix: str, interval: dict) -> dict:
    return {
        f"{prefix}_point": interval["point"],
        f"{prefix}_ci95_low": interval["ci95_low"],
        f"{prefix}_ci95_high": interval["ci95_high"],
        f"{prefix}_supported": interval.get("directionally_supported"),
    }


def main() -> int:
    if OUTPUT.exists():
        raise FileExistsError(f"Refusing to overwrite uncertainty output: {OUTPUT}")
    status = subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).splitlines()
    if status:
        raise RuntimeError(f"Uncertainty summary requires a clean worktree: {status}")
    OUTPUT.mkdir(parents=True)
    formal_result, groups = load_cells()
    total_memory_mib = float(formal_result["metadata"]["runtime"]["total_gpu_memory_mib"])

    absolute = []
    absolute_csv = []
    for batch_size in BATCH_SIZES:
        for config_id in CONFIG_IDS:
            metrics = {}
            flat = {"config_id": config_id, "batch_size": batch_size}
            for metric in ABSOLUTE_METRICS:
                interval = hierarchical_median_interval(
                    metric_blocks(groups, config_id, batch_size, metric),
                    label=f"absolute:{config_id}:bs{batch_size}:{metric}",
                )
                metrics[metric] = interval
                flat.update(flatten_interval(metric, interval))
            reserved_values = np.concatenate(list(metric_blocks(
                groups, config_id, batch_size, "peak_reserved_mib",
            ).values()))
            memory = {
                "physical_memory_mib": total_memory_mib,
                "maximum_observed_reserved_mib": float(np.max(reserved_values)),
                "median_reserved_utilization_percent": metrics["peak_reserved_mib"]["point"] / total_memory_mib * 100.0,
                "median_physical_headroom_mib": total_memory_mib - metrics["peak_reserved_mib"]["point"],
                "ci95_high_below_physical_memory": metrics["peak_reserved_mib"]["ci95_high"] < total_memory_mib,
            }
            absolute.append({"config_id": config_id, "batch_size": batch_size,
                             "metrics": metrics, "memory": memory})
            flat.update(memory)
            absolute_csv.append(flat)

    reference_contrasts = []
    reference_csv = []
    contrast_specs = [
        ("end_to_end_ms", "latency_reduction_percent", "end_to_end_latency_reduction_percent"),
        ("samples_per_second", "throughput_improvement_percent", "throughput_improvement_percent"),
        ("peak_allocated_mib", "memory_reduction_mib", "allocated_memory_reduction_mib"),
        ("peak_reserved_mib", "memory_reduction_mib", "reserved_memory_reduction_mib"),
    ]
    for batch_size in BATCH_SIZES:
        for candidate_id in CONFIG_IDS[1:]:
            effects = {}
            flat = {"reference_id": "E0", "candidate_id": candidate_id, "batch_size": batch_size}
            for metric, effect, name in contrast_specs:
                interval = hierarchical_contrast_interval(
                    metric_blocks(groups, "E0", batch_size, metric),
                    metric_blocks(groups, candidate_id, batch_size, metric),
                    effect=effect,
                    label=f"reference:E0:{candidate_id}:bs{batch_size}:{name}",
                )
                effects[name] = interval
                flat.update(flatten_interval(name, interval))
            reference_contrasts.append({"reference_id": "E0", "candidate_id": candidate_id,
                                        "batch_size": batch_size, "effects": effects})
            reference_csv.append(flat)

    adjacent_pairs = [("E0", "E1"), ("E1", "E2"), ("E2", "E3"), ("E3", "E4")]
    adjacent_specs = [
        ("end_to_end_ms", "latency_reduction_percent", "end_to_end_latency_reduction_percent"),
        ("vit_encoder_ms", "latency_reduction_percent", "vit_latency_reduction_percent"),
        ("peak_allocated_mib", "memory_reduction_mib", "allocated_memory_reduction_mib"),
        ("peak_reserved_mib", "memory_reduction_mib", "reserved_memory_reduction_mib"),
    ]
    adjacent = []
    adjacent_csv = []
    for batch_size in BATCH_SIZES:
        for reference_id, candidate_id in adjacent_pairs:
            effects = {}
            flat = {"reference_id": reference_id, "candidate_id": candidate_id,
                    "batch_size": batch_size}
            for metric, effect, name in adjacent_specs:
                interval = hierarchical_contrast_interval(
                    metric_blocks(groups, reference_id, batch_size, metric),
                    metric_blocks(groups, candidate_id, batch_size, metric),
                    effect=effect,
                    label=f"adjacent:{reference_id}:{candidate_id}:bs{batch_size}:{name}",
                )
                effects[name] = interval
                flat.update(flatten_interval(name, interval))
            adjacent.append({"reference_id": reference_id, "candidate_id": candidate_id,
                             "batch_size": batch_size, "effects": effects})
            adjacent_csv.append(flat)

    maximum_reserved = max(
        row["memory"]["maximum_observed_reserved_mib"] for row in absolute
    )
    result = {
        "protocol": "formal-efficiency-uncertainty-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": {"path": str(FORMAL_RESULT.resolve()), "sha256": FORMAL_RESULT_SHA256,
                   "cells": 75, "retained_iterations": 3750},
        "bootstrap": {"base_seed": BASE_SEED, "resamples": RESAMPLES,
                      "interval": "percentile_95", "hierarchy": "block_then_iteration"},
        "absolute_intervals": absolute,
        "reference_contrasts": reference_contrasts,
        "adjacent_ablation_contrasts": adjacent,
        "memory_audit": {
            "physical_memory_mib": total_memory_mib,
            "maximum_observed_reserved_mib": maximum_reserved,
            "minimum_observed_physical_headroom_mib": total_memory_mib - maximum_reserved,
            "all_ci95_high_reserved_below_physical_memory": all(
                row["memory"]["ci95_high_below_physical_memory"] for row in absolute
            ),
            "all_batch16_cells_completed": True,
        },
        "limits": [
            "Post-measurement descriptive intervals; the original measurements were not filtered or rerun.",
            "Only three blocks on one local GPU; intervals are not broad hardware-population estimates.",
            "Repeated iterations can remain autocorrelated despite hierarchical resampling.",
            "Batch size 16 is a capability boundary rather than the primary paper operating point.",
        ],
    }
    write_json(OUTPUT / "result.json", result)
    write_csv(OUTPUT / "absolute_intervals.csv", absolute_csv)
    write_csv(OUTPUT / "reference_contrasts.csv", reference_csv)
    write_csv(OUTPUT / "adjacent_ablation_contrasts.csv", adjacent_csv)
    print(json.dumps({
        "output": str(OUTPUT.resolve()),
        "source_cells": 75,
        "retained_iterations": 3750,
        "bootstrap_resamples": RESAMPLES,
        "maximum_observed_reserved_mib": maximum_reserved,
        "minimum_physical_headroom_mib": total_memory_mib - maximum_reserved,
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
