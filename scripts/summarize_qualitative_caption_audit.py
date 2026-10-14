from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def load_ratings(path: Path) -> dict[tuple[str, str], dict]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 72:
        raise RuntimeError(f"expected 72 ratings, found {len(rows)}")
    output = {}
    for row in rows:
        key = (row["audit_id"], row["label"])
        if key in output:
            raise RuntimeError(f"duplicate rating: {key}")
        for field in ("subject", "action", "context", "hallucination", "fluency"):
            row[field] = int(row[field])
            if row[field] not in (0, 1):
                raise RuntimeError(f"{key} has invalid {field}")
        row["semantic_score"] = row["subject"] + row["action"] + row["context"]
        row["grounded"] = row["semantic_score"] >= 2 and row["hallucination"] == 0
        output[key] = row
    return output


def preference(a: dict, b: dict) -> str:
    if b["semantic_score"] > a["semantic_score"]:
        return "B"
    if a["semantic_score"] > b["semantic_score"]:
        return "A"
    if b["hallucination"] < a["hallucination"]:
        return "B"
    if a["hallucination"] < b["hallucination"]:
        return "A"
    return "tie"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ratings", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--key", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    ratings = load_ratings(args.ratings)
    manifest = {row["audit_id"]: row for row in load_json(args.manifest)}
    keys = load_json(args.key)
    identity = {}
    for item in keys:
        for row in item["mapping"]:
            identity[(item["audit_id"], row["label"])] = (row["seed"], row["arm"])
    if set(identity) != set(ratings):
        raise RuntimeError("ratings and unblinding key do not cover identical candidates")

    caption_by_label = {
        (audit_id, row["label"]): row["caption"]
        for audit_id, item in manifest.items()
        for row in item["candidates"]
    }
    unblinded = {}
    arm_rows = defaultdict(list)
    for candidate, rating in ratings.items():
        seed, arm = identity[candidate]
        row = dict(rating)
        row.update({"seed": seed, "arm": arm, "caption": caption_by_label[candidate]})
        unblinded[(candidate[0], seed, arm)] = row
        arm_rows[arm].append(row)

    arm_summary = {}
    for arm, rows in sorted(arm_rows.items()):
        arm_summary[arm] = {
            "captions": len(rows),
            "mean_semantic_score": sum(row["semantic_score"] for row in rows) / len(rows),
            "grounded": sum(row["grounded"] for row in rows),
            "hallucination": sum(row["hallucination"] for row in rows),
            "fluent": sum(row["fluency"] for row in rows),
            "subject_correct": sum(row["subject"] for row in rows),
            "action_correct": sum(row["action"] for row in rows),
            "context_compatible": sum(row["context"] for row in rows),
        }

    pair_rows = []
    pair_summary = defaultdict(lambda: {"B": 0, "A": 0, "tie": 0, "changed": 0})
    for audit_id in sorted(manifest):
        for seed in (123, 124, 125):
            a = unblinded[(audit_id, seed, "A")]
            b = unblinded[(audit_id, seed, "B")]
            winner = preference(a, b)
            changed = a["caption"] != b["caption"]
            pair_summary[str(seed)][winner] += 1
            pair_summary[str(seed)]["changed"] += int(changed)
            pair_summary["all"][winner] += 1
            pair_summary["all"]["changed"] += int(changed)
            pair_rows.append(
                {
                    "audit_id": audit_id,
                    "video_id": manifest[audit_id]["video_id"],
                    "seed": seed,
                    "A_caption": a["caption"],
                    "A_semantic_score": a["semantic_score"],
                    "A_hallucination": a["hallucination"],
                    "B_caption": b["caption"],
                    "B_semantic_score": b["semantic_score"],
                    "B_hallucination": b["hallucination"],
                    "changed": changed,
                    "preference": winner,
                }
            )

    output = {
        "protocol": "bounded-qualitative-caption-audit-v1",
        "ratings": len(ratings),
        "videos": len(manifest),
        "arm_summary": arm_summary,
        "pair_summary": dict(pair_summary),
        "pairs": pair_rows,
        "model_inference_performed": False,
        "test_tuning_performed": False,
        "human_evaluation": False,
    }
    write_json(args.output, output)


if __name__ == "__main__":
    main()
