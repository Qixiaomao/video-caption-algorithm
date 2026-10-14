"""Freeze the deterministic validation inputs for the formal efficiency matrix."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ANNOTATIONS = ROOT / "data/processed/msvd_full/val/annotations.json"
ANNOTATIONS_SHA256 = "bcda3ba0583704a54297e4da4cc9acbcc5ec1ebffd6ed436a6ba3ee1b8a03c8b"
SELECTION_SALT = "efficiency-matrix-v1"
SELECTION_SIZE = 16


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def selection_digest(video_id: str) -> str:
    return hashlib.sha256(f"{SELECTION_SALT}:{video_id}".encode("utf-8")).hexdigest()


def select_records(records: list[dict], size: int = SELECTION_SIZE) -> list[dict]:
    if len(records) < size:
        raise ValueError(f"Need at least {size} records, found {len(records)}")
    ranked = sorted(records, key=lambda row: (selection_digest(str(row["video_id"])), str(row["video_id"])))
    return ranked[:size]


def build_manifest() -> dict:
    annotation_hash = sha256_file(ANNOTATIONS)
    if annotation_hash != ANNOTATIONS_SHA256:
        raise RuntimeError("Frozen validation annotations changed")
    records = json.loads(ANNOTATIONS.read_text(encoding="utf-8"))
    if len(records) != 100:
        raise RuntimeError(f"Expected 100 validation records, found {len(records)}")

    selected = []
    for rank, record in enumerate(select_records(records), start=1):
        frames_dir = Path(record["frames_dir"]).resolve()
        frame_paths = sorted(frames_dir.glob("frame_*.jpg"))
        if len(frame_paths) != 8:
            raise RuntimeError(f"Expected exactly 8 frames for {record['video_id']}, found {len(frame_paths)}")
        selected.append(
            {
                "rank": rank,
                "video_id": str(record["video_id"]),
                "selection_sha256": selection_digest(str(record["video_id"])),
                "frames_dir": str(frames_dir),
                "frames": [
                    {"path": str(path.resolve()), "sha256": sha256_file(path)}
                    for path in frame_paths
                ],
            }
        )

    return {
        "protocol": "formal-efficiency-matrix-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "test_split_read": False,
        "source": {
            "split": "val",
            "annotations": str(ANNOTATIONS.resolve()),
            "annotations_sha256": annotation_hash,
            "record_count": len(records),
        },
        "selection": {
            "salt": SELECTION_SALT,
            "rule": "ascending_sha256(salt + ':' + video_id)",
            "size": SELECTION_SIZE,
            "nested_primary_batch_sizes": [1, 2, 4, 8],
            "optional_boundary_batch_size": 16,
        },
        "records": selected,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "experiments/formal_efficiency_inputs_20260909.json",
    )
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite frozen manifest: {args.output}")
    manifest = build_manifest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(args.output.resolve()),
        "records": len(manifest["records"]),
        "annotations_sha256": manifest["source"]["annotations_sha256"],
        "first_video_id": manifest["records"][0]["video_id"],
        "last_video_id": manifest["records"][-1]["video_id"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
