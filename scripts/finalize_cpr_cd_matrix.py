"""Validate six completed C/D arms and write the registered validation decision.

The historical seed123 A result predates the repeated-seed result schema: its
final_caption is already a metrics object.  Normalize that one endpoint here,
leaving the source-hashed training runner and all model artifacts untouched.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import run_cpr_cd_matrix as registered
from scripts import resume_cpr_cd_matrix as resume
from scripts.run_caption_policy_ab import write_json
from src.training.checkpoint import sha256_file


OUTPUT_ROOT = ROOT / "runs/q3_cpr_cd_resumed_formal_20260925"
ORIGINAL_ENDPOINT = registered.historical_endpoint


def normalized_historical_endpoint(seed: int, arm: str) -> dict:
    old = ORIGINAL_ENDPOINT(seed, arm)
    if seed == 123 and arm == "A":
        if "metrics" in old["caption"]:
            raise RuntimeError("historical seed123 A schema unexpectedly changed")
        return {"caption": {"metrics": old["caption"]}, "retrieval": old["retrieval"]}
    if "metrics" not in old["caption"]:
        raise RuntimeError(f"historical seed{seed}-{arm} lacks caption metrics")
    return old


def load_and_verify() -> tuple[dict[int, dict[str, dict]], list[dict]]:
    registered.verify_registered_inputs()
    original = resume.PAUSED_ROOT
    source_hashes = {name: sha256_file(ROOT / name) for name in registered.SOURCE_FILES}
    results: dict[int, dict[str, dict]] = {}
    pairs = []
    for seed in registered.SEEDS:
        root = original if seed == 123 else OUTPUT_ROOT
        seed_dir = root / f"seed{seed}"
        preflight = json.loads((seed_dir / "preflight.json").read_text(encoding="utf-8"))
        if preflight["source_sha256"] != source_hashes:
            raise RuntimeError(f"seed{seed} registered source hash changed")
        if preflight["test_split_read"] or preflight["smoke_test"]:
            raise RuntimeError(f"seed{seed} preflight is not a sealed formal run")
        arms = {}
        for arm in registered.ARMS:
            arm_dir = seed_dir / arm
            result = json.loads((arm_dir / "result.json").read_text(encoding="utf-8"))
            if result["seed"] != seed or result["arm"] != arm or result["steps"] != registered.FORMAL_STEPS:
                raise RuntimeError(f"seed{seed}-{arm} has invalid progress")
            if result["smoke_test"] or result["test_split_read"] or result["paper_ready"]:
                raise RuntimeError(f"seed{seed}-{arm} has invalid evaluation boundary")
            for field in (
                "historical_trace_prefix_match", "final_checkpoint_strict_reload",
                "frozen_weights_unchanged", "source_and_data_unchanged",
            ):
                if result[field] is not True:
                    raise RuntimeError(f"seed{seed}-{arm} failed {field}")
            if result["training_contract"] != registered.training_contract(arm):
                raise RuntimeError(f"seed{seed}-{arm} training contract changed")
            if [item["epoch"] for item in result["evaluations"]] != list(registered.MILESTONES):
                raise RuntimeError(f"seed{seed}-{arm} evaluation milestones are incomplete")
            for evaluation in result["evaluations"]:
                if evaluation["step"] != evaluation["epoch"] * 150:
                    raise RuntimeError(f"seed{seed}-{arm} evaluation step changed")
                checkpoint = Path(evaluation["checkpoint"])
                if sha256_file(checkpoint) != evaluation["checkpoint_sha256"]:
                    raise RuntimeError(f"seed{seed}-{arm} checkpoint changed: {checkpoint}")
            trace = arm_dir / "trace.jsonl"
            if sha256_file(trace) != result["trace_sha256"]:
                raise RuntimeError(f"seed{seed}-{arm} trace file changed")
            if result["trace_sha256"] != registered.REGISTERED_INPUTS[seed]["trace"][1]:
                raise RuntimeError(f"seed{seed}-{arm} trace differs from registered A trace")
            arms[arm] = result
        if arms["C"]["trace_sha256"] != arms["D"]["trace_sha256"]:
            raise RuntimeError(f"seed{seed} C/D trace mismatch")
        pair = json.loads((seed_dir / "pair_result.json").read_text(encoding="utf-8"))
        if pair["seed"] != seed or not pair["trace_match"] or not pair["historical_trace_prefix_match"]:
            raise RuntimeError(f"seed{seed} pair verification failed")
        for arm in registered.ARMS:
            expected = (seed_dir / arm / "result.json").resolve()
            if Path(pair["results"][arm]).resolve() != expected:
                raise RuntimeError(f"seed{seed}-{arm} pair result path changed")
        results[seed] = arms
        pairs.append(pair)
    resumed = results[124]["C"].get("resume_provenance")
    if resumed is None or resumed["checkpoint_sha256"] != resume.PAUSED_SHA256["checkpoint_epoch040.pt"]:
        raise RuntimeError("seed124-C resume provenance missing or changed")
    return results, pairs


def main() -> int:
    completion_path = OUTPUT_ROOT / "completion.json"
    if completion_path.exists():
        raise FileExistsError(f"refusing to overwrite completion: {completion_path}")
    results, pairs = load_and_verify()
    with patch.object(registered, "historical_endpoint", normalized_historical_endpoint):
        decision = registered.formal_decision(results)
    completion = {
        "completed": True,
        "seeds": list(registered.SEEDS),
        "arms": list(registered.ARMS),
        "pairs": pairs,
        "formal_decision": decision,
        "resume_provenance": results[124]["C"]["resume_provenance"],
        "aggregation_repair": {
            "reason": "historical seed123 A final_caption is already a metrics object",
            "scope": "validation-only postprocessing; no training or evaluation rerun",
            "registered_runner_sha256": sha256_file(ROOT / "scripts/run_cpr_cd_matrix.py"),
            "finalizer_sha256": sha256_file(Path(__file__)),
        },
        "smoke_test": False,
        "paper_ready": False,
        "test_split_read": False,
    }
    write_json(completion_path, completion)
    print(json.dumps({
        "completed": True,
        "decision": decision["gates"]["decision"],
        "gates": decision["gates"],
        "completion": str(completion_path.resolve()),
        "test_split_read": False,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
