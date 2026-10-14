"""Recompute the public evidence table using only the standard library.

Numeric extracts support arithmetic checks without distributing datasets/weights.
They cannot authenticate omitted original source files against their hashes.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]
KEY = ("dataset", "split", "experiment", "arm", "comparison", "seed", "batch_size", "metric", "statistic")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def close(actual, expected, label):
    require(math.isfinite(actual) and math.isfinite(expected), f"Nonfinite value: {label}")
    require(math.isclose(actual, expected, rel_tol=1e-11, abs_tol=1e-11),
            f"Numeric mismatch: {label}: {actual} != {expected}")


def evaluate(operation, operands, source_values):
    require(bool(operands), "Missing operands")
    values = []
    for operand in operands:
        raw = source_values[operand["source_value_id"]]["raw_value"]
        scale = float(operand["scale"])
        if operation == "mean_array":
            require(len(operands) == 1 and isinstance(raw, list) and bool(raw), "Invalid mean_array")
            return statistics.mean(float(v) * scale for v in raw)
        require(isinstance(raw, (int, float)) and not isinstance(raw, bool), "Expected numeric scalar")
        values.append(float(raw) * scale)
    if operation == "identity":
        require(len(values) == 1, "Invalid identity")
        return values[0]
    if operation == "difference":
        require(len(values) == 2, "Invalid difference")
        return values[1] - values[0]
    if operation in ("mean_difference", "sample_sd_difference"):
        require(len(values) % 2 == 0, "Unpaired operands")
        values = [values[i + 1] - values[i] for i in range(0, len(values), 2)]
    require(operation in ("mean", "sample_sd", "mean_difference", "sample_sd_difference"),
            f"Unknown operation: {operation}")
    return statistics.stdev(values) if operation.startswith("sample_sd") else statistics.mean(values)


def verify(evidence):
    manifest = json.loads((evidence / "release_manifest.json").read_text(encoding="utf-8"))
    expected_files = {"paper_master_results.csv", "source_values.json", "row_lineage.json", "source_manifest.json"}
    require(set(manifest["files"]) == expected_files, "Manifest file set changed")
    for name, expected in manifest["files"].items():
        require(hashlib.sha256((evidence / name).read_bytes()).hexdigest() == expected,
                f"Public artifact hash mismatch: {name}")
    with (evidence / "paper_master_results.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    require(len(rows) == manifest["rows"] == 1036, "Table row count changed")
    keys = [tuple(row[key] for key in KEY) for row in rows]
    require(len(keys) == len(set(keys)), "Duplicate record key")
    raw_values = json.loads((evidence / "source_values.json").read_text(encoding="utf-8"))["values"]
    values = {item["id"]: item for item in raw_values}
    require(len(values) == len(raw_values), "Duplicate source-value ID")
    lineage = json.loads((evidence / "row_lineage.json").read_text(encoding="utf-8"))["rows"]
    require([item["csv_data_row"] for item in lineage] == list(range(1, 1037)), "Incomplete row lineage")
    for item, row in zip(lineage, rows):
        require(bool(row["unit"]) and bool(row["source_field"]), "Missing unit/source field")
        close(evaluate(item["operation"], item["operands"], values), float(row["value"]),
              f"data row {item['csv_data_row']}")

    groups = defaultdict(list)
    group_keys = ("dataset", "split", "experiment", "arm", "comparison", "batch_size", "metric", "unit")
    for row in rows:
        if row["seed"] in ("123", "124", "125") and row["statistic"] in (
                "fixed_endpoint_seed_value", "seed_value_or_paired_difference",
                "paired_seed_difference", "epoch66_seed_value"):
            groups[tuple(row[k] for k in group_keys)].append(row)
    summary_checks = 0
    for row in rows:
        if row["statistic"] not in ("mean", "sample_sd") or row["seed"] != "123/124/125":
            continue
        group = tuple(row[k] for k in group_keys)
        members = groups.get(group, [])
        if row["comparison"] == "B-A" and not members:
            arms = {}
            for arm in ("A", "B"):
                matching = [r for r in rows if all(r[k] == row[k] for k in
                            ("dataset", "split", "experiment", "batch_size", "metric"))
                            and r["arm"] == arm and r["seed"] in ("123", "124", "125")]
                arms[arm] = {r["seed"]: float(r["value"]) for r in matching}
            require(set(arms["A"]) == set(arms["B"]) == {"123", "124", "125"}, "Missing paired seeds")
            numbers = [arms["B"][s] - arms["A"][s] for s in ("123", "124", "125")]
        else:
            require(len(members) == 3 and {m["seed"] for m in members} == {"123", "124", "125"},
                    f"Invalid three-seed group: {group}; {len(members)} members")
            numbers = [float(m["value"]) for m in members]
        expected = statistics.mean(numbers) if row["statistic"] == "mean" else statistics.stdev(numbers)
        close(expected, float(row["value"]), str(group))
        summary_checks += 1
    return {"status": "PASS", "rows_recomputed": len(rows), "numeric_source_fields": len(values),
            "seed_summaries_recomputed": summary_checks,
            "scope": "public artifact hashes, numeric extracts and arithmetic; no model execution or omitted-source authentication"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, default=ROOT / "reproducibility/evidence")
    args = parser.parse_args()
    print(json.dumps(verify(args.evidence), indent=2))
