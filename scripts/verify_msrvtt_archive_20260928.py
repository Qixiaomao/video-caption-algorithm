"""Verify the downloaded archive without extracting data or reading model metrics."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
import zipfile

ROOT = Path(__file__).resolve().parents[1]
FOLDER = ROOT / "runs/msrvtt_preflight_20260927/archive_download"
ARCHIVE = FOLDER / "MSRVTT.zip"
OUTPUT = FOLDER / "download_verification.json"
EXPECTED_BYTES = 6552768292


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    started = time.perf_counter()
    result = {"archive_path": str(ARCHIVE), "verified_at_utc": datetime.now(timezone.utc).isoformat(),
              "expected_bytes": EXPECTED_BYTES, "status": "in_progress"}
    try:
        length = ARCHIVE.stat().st_size
        result["actual_bytes"] = length
        if length != EXPECTED_BYTES:
            raise RuntimeError(f"Archive length mismatch: {length}")
        digest = hashlib.sha256()
        with ARCHIVE.open("rb") as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(block)
        result["sha256"] = digest.hexdigest()
        print(json.dumps({"stage": "sha256_complete", "sha256": result["sha256"]}), flush=True)
        with zipfile.ZipFile(ARCHIVE) as z:
            members = z.namelist()
            result["member_count"] = len(members)
            result["video_member_count"] = sum(name.startswith("MSRVTT/videos/all/video") and name.endswith(".mp4") for name in members)
            result["required_members_present"] = all(name in members for name in (
                "MSRVTT/annotation/MSR_VTT.json",
                "MSRVTT/structured-symlinks/train_list_full.txt",
                "MSRVTT/structured-symlinks/val_list_full.txt",
                "MSRVTT/structured-symlinks/test_list_full.txt"))
            bad = z.testzip()
            result["first_crc_failed_member"] = bad
        result["status"] = "pass" if bad is None and result["video_member_count"] == 10000 and result["required_members_present"] else "fail"
    except Exception as exc:
        result["status"] = "fail"
        result["error_type"] = type(exc).__name__
        result["error"] = str(exc)
    result["elapsed_seconds"] = time.perf_counter() - started
    OUTPUT.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result), flush=True)
    if result["status"] != "pass":
        raise RuntimeError(f"Verification failed; see {OUTPUT}")


if __name__ == "__main__":
    main()
