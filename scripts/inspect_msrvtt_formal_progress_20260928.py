"""Read-only progress snapshot for the registered six-arm MSR-VTT run."""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs/msrvtt_ab_formal_20260928"


def last_json_line(path: Path) -> dict | None:
    if not path.is_file() or path.stat().st_size == 0:
        return None
    with path.open("rb") as stream:
        stream.seek(0, 2)
        position = stream.tell() - 1
        stream.seek(position)
        if stream.read(1) == b"\n":
            position -= 1
        while position >= 0:
            stream.seek(position)
            if stream.read(1) == b"\n":
                break
            position -= 1
        stream.seek(max(0, position + 1))
        raw = stream.readline().strip()
        if not raw:
            stream.seek(0)
            raw = stream.readline().strip()
        return json.loads(raw)


def main() -> None:
    config = json.loads((ROOT / "experiments/msrvtt_ab_formal_config_20260928.json").read_text(encoding="utf-8"))
    arms = []
    for seed in config["seeds"]:
        for arm in config["arms"]:
            directory = RUN / f"seed{seed}" / arm
            metric = directory / "train_metrics.jsonl"
            latest = last_json_line(metric)
            checkpoint_epochs = sorted(int(path.stem[-3:]) for path in directory.glob("checkpoint_epoch*.pt")) if directory.is_dir() else []
            eval_epochs = sorted(int(path.stem[-3:]) for path in directory.glob("evaluation_epoch*.json")) if directory.is_dir() else []
            arms.append({"seed": seed, "arm": arm, "directory_exists": directory.is_dir(),
                         "latest": latest, "metric_file_mtime_utc": datetime.fromtimestamp(metric.stat().st_mtime,timezone.utc).isoformat() if metric.exists() else None,
                         "checkpoints": checkpoint_epochs, "evaluations": eval_epochs,
                         "result_exists": (directory / "result.json").is_file()})
    print(json.dumps({"at_utc": datetime.now(timezone.utc).isoformat(),
                      "launch_exists": (RUN / "launch.json").is_file(),
                      "failure_exists": (RUN / "failure.json").is_file(),
                      "validation_freeze_exists": (RUN / "validation_freeze.json").is_file(),
                      "completion_exists": (RUN / "completion.json").is_file(),
                      "arms": arms}, ensure_ascii=False))


if __name__ == "__main__":
    main()
