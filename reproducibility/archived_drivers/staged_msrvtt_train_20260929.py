"""Run exactly one frozen MSR-VTT training arm per invocation.

Each invocation has a fresh output root. Completed arms are verified and
hard-linked into the next root on the same volume; training files are never
modified in place. The first invocation uses the already verified epoch-20
recovery adapter. No test data is accessed here.
"""
from __future__ import annotations

import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sys
import traceback

CODE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(CODE_ROOT))

from scripts import run_msrvtt_ab_formal_20260928 as formal
from scripts import resume_msrvtt_ab_formal_20260929 as resume
from src.training.checkpoint import sha256_file

ORDER = [(seed, arm) for seed in (123, 124, 125) for arm in ("A", "B")]
CONFIG = CODE_ROOT / "experiments/msrvtt_ab_formal_config_20260928.json"
ORIGINAL_ARM = formal.run_arm
ORIGINAL_RESUME_ARM = resume.resume_first_arm


class PlannedPause(Exception):
    pass


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def verified_prior(previous: Path, seed: int, arm: str, current_seed: Path) -> dict:
    source_seed = previous / f"seed{seed}"
    source = source_seed / arm
    result = read_json(source / "result.json")
    if (result.get("seed"), result.get("arm"), result.get("steps"),
            result.get("epochs"), result.get("technical_run_complete")) != (
            seed, arm, 24420, 30, True):
        raise RuntimeError(f"Incomplete prior arm: {source}")
    if sha256_file(source / "trace.jsonl") != result["trace_sha256"]:
        raise RuntimeError(f"Prior trace changed: {source}")
    if sha256_file(source / "train_metrics.jsonl") != result["train_metrics_sha256"]:
        raise RuntimeError(f"Prior metrics changed: {source}")
    endpoint = source / "checkpoint_epoch030.pt"
    final_eval = read_json(source / "evaluation_epoch030.json")
    if (not endpoint.is_file() or final_eval["step"] != 24420
            or final_eval["checkpoint_sha256"] != sha256_file(endpoint)):
        raise RuntimeError(f"Prior endpoint changed: {source}")
    for name in ("schedule.json", "initial_model.pt"):
        if sha256_file(source_seed / name) != sha256_file(current_seed / name):
            raise RuntimeError(f"Regenerated {name} differs for seed {seed}")
    return result


def reuse_arm(previous: Path, folder: Path, result: dict) -> dict:
    if folder.exists():
        raise FileExistsError(folder)
    source = previous / folder.parent.name / folder.name
    shutil.copytree(source, folder, copy_function=os.link)
    # Replace only the linked result file with a run-local path manifest.
    (folder / "result.json").unlink()
    result = dict(result)
    result["evaluations"] = [dict(row, path=str(folder / Path(row["path"]).name))
                             for row in result["evaluations"]]
    result["reused_from"] = str(source)
    formal.write_new(folder / "result.json", result)
    return result


def compare_extra_replay(out: Path, earlier: Path) -> int:
    """Check the 286 steps present only in the interrupted second run."""
    count = 0
    for name in ("trace.jsonl", "train_metrics.jsonl"):
        source = earlier / "seed123/A" / name
        target = out / "seed123/A" / name
        with source.open(encoding="utf-8") as old, target.open(encoding="utf-8") as new:
            for index, (left, right) in enumerate(zip(old, new), start=1):
                if 20412 <= index <= 20697:
                    if json.loads(left) != json.loads(right):
                        raise RuntimeError(f"Second-run replay mismatch in {name} at {index}")
                    count += 1
            if name == "train_metrics.jsonl" and index < 20697:
                raise RuntimeError("Second-run metrics shorter than expected")
    return count


def main(args: argparse.Namespace) -> None:
    out = args.output.resolve()
    if out.exists():
        raise FileExistsError(out)
    if sha256_file(CONFIG) != "75fbdd3131fc9177d0fcb828f4a0f6be65e6e884c3249434813b39d4e7f9ec38":
        raise RuntimeError("Frozen config changed")
    if args.previous:
        previous = args.previous.resolve()
        pause = read_json(previous / "planned_pause.json")
        if pause["status"] != "ARM_COMPLETE_PAUSED" or pause["next_arm"] is None:
            raise RuntimeError("Previous run has no valid next arm")
        target = tuple(pause["next_arm"])
        if target not in ORDER:
            raise RuntimeError("Unexpected target arm")
        completed = ORDER[:ORDER.index(target)]
        if pause["completed_arms"] != [list(item) for item in completed]:
            raise RuntimeError("Prior arm order is incomplete")
        if sha256_file(previous / "source_hashes.json") != pause["source_hashes_sha256"]:
            raise RuntimeError("Previous source manifest changed")
        prior_hashes = read_json(previous / "source_hashes.json")
        if prior_hashes != {key: sha256_file(CODE_ROOT / key) for key in prior_hashes}:
            raise RuntimeError("Frozen training code changed")

        def staged_arm(model, arm, seed, folder, *rest):
            key = (seed, arm)
            if key in completed:
                prior = verified_prior(previous, seed, arm, folder.parent)
                return reuse_arm(previous, folder, prior)
            if key == target:
                return ORIGINAL_ARM(model, arm, seed, folder, *rest)
            raise PlannedPause()

        formal.run_arm = staged_arm
        launch = lambda: formal.main(CONFIG, out)
    else:
        if not args.source or not args.verification or not args.earlier_run:
            raise ValueError("First stage requires --source, --verification, and --earlier-run")
        target = ORDER[0]
        completed = []

        def staged_resume_arm(model, arm, seed, folder, *rest):
            if (seed, arm) == target:
                return ORIGINAL_RESUME_ARM(model, arm, seed, folder, *rest)
            raise PlannedPause()

        resume.resume_first_arm = staged_resume_arm
        launch = lambda: resume.main(CONFIG, args.source, args.verification, out)

    paused = False
    try:
        launch()
    except PlannedPause:
        paused = True
    except Exception:
        if out.is_dir() and not (out / "failure.json").exists():
            formal.write_new(out / "failure.json", {
                "at_utc": datetime.now(timezone.utc).isoformat(),
                "traceback": traceback.format_exc(),
            })
        raise

    endpoint = out / f"seed{target[0]}" / target[1]
    verified_prior(out, target[0], target[1], out / f"seed{target[0]}")
    extra = compare_extra_replay(out, args.earlier_run) if not args.previous else None
    done = ORDER[:ORDER.index(target) + 1]
    next_arm = ORDER[len(done)] if len(done) < len(ORDER) else None
    if (next_arm is None) != (not paused):
        raise RuntimeError("Runner stop and expected arm boundary disagree")
    if paused:
        formal.write_new(out / "planned_pause.json", {
            "at_utc": datetime.now(timezone.utc).isoformat(),
            "status": "ARM_COMPLETE_PAUSED",
            "reason": "author_selected_one_arm_per_gpu_window",
            "completed_arms": [list(item) for item in done],
            "next_arm": list(next_arm),
            "last_endpoint": str(endpoint / "checkpoint_epoch030.pt"),
            "last_endpoint_sha256": sha256_file(endpoint / "checkpoint_epoch030.pt"),
            "source_hashes_sha256": sha256_file(out / "source_hashes.json"),
            "config_sha256": sha256_file(CONFIG),
            "staged_driver_sha256": sha256_file(Path(__file__)),
            "second_run_extra_replay_rows_checked": extra,
            "test_split_read": False,
        })
    print(json.dumps({"status": "ARM_COMPLETE_PAUSED" if paused else "SIX_ARMS_COMPLETE",
                      "completed": done, "next_arm": next_arm,
                      "output": str(out), "second_run_extra_replay_rows_checked": extra}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--previous", type=Path)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--verification", type=Path)
    parser.add_argument("--earlier-run", type=Path)
    main(parser.parse_args())
