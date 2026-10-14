"""Freeze a new local MSR-VTT config after reproducing the preparation gates.

Never edits the original config, model, gate files or existing runs. Relocated
preparation artifacts acquire new hashes; this is a new reproduction record.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "experiments/msrvtt_ab_formal_config_20260928.json"


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def materialize(gpt2_source, output):
    output = output.resolve()
    if output.parent != ROOT / "experiments" or not output.name.endswith(".local.json"):
        raise ValueError("Output must be experiments/<name>.local.json (ignored by Git)")
    if output.exists():
        raise FileExistsError(output)
    config = json.loads(TEMPLATE.read_text(encoding="utf-8"))
    gate = ROOT / "runs/msrvtt_data_full_20260928/data_gate.json"
    gates = {
        "engineering_smoke": ROOT / "runs/msrvtt_engineering_smoke_20260928_v3/result.json",
        "pilot": ROOT / "runs/msrvtt_train_val_pilot_20260928/result.json",
        "validation_timing": ROOT / "runs/msrvtt_validation_timing_20260928/result.json",
    }
    if json.loads(gate.read_text(encoding="utf-8")).get("status") != "PASS":
        raise ValueError("Data gate is not PASS")
    config["data_gate_sha256"] = sha256(gate)
    for name, path in gates.items():
        value = json.loads(path.read_text(encoding="utf-8"))
        if value.get("status") != "PASS" or value.get("test_split_read") is not False:
            raise ValueError(f"Preparation gate invalid or test used: {name}")
        config[name + "_result_path"] = str(path)
        config[name + "_sha256"] = sha256(path)
    weights = gpt2_source.resolve() / "model.safetensors"
    if sha256(weights) != config["gpt2_model_sha256"]:
        raise ValueError("GPT-2 weights differ from the registered source")
    config["gpt2_source_path"] = str(gpt2_source.resolve())
    test = ROOT / "experiments/msrvtt_test_eval_config_20260928.json"
    config["test_evaluation_config_path"] = str(test)
    config["test_evaluation_config_sha256"] = sha256(test)
    config["frozen_at_utc"] = datetime.now(timezone.utc).isoformat()
    config["reproduction_record"] = {
        "original_config_sha256": "75fbdd3131fc9177d0fcb828f4a0f6be65e6e884c3249434813b39d4e7f9ec38",
        "public_template_sha256": sha256(TEMPLATE),
        "scope": "new local registration; original training hyperparameters and pinned GPT-2 weights retained",
    }
    with output.open("x", encoding="utf-8") as stream:
        json.dump(config, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpt2-source", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=ROOT / "experiments/msrvtt_reproduction.local.json")
    args = parser.parse_args()
    print(materialize(args.gpt2_source, args.output))
