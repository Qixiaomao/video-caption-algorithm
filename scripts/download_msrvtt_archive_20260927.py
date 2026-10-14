"""Resumable, isolated download of the pinned MSR-VTT archive; never extracts or overwrites old runs."""
from __future__ import annotations

import json
from pathlib import Path
import time

import requests

URL = "https://www.robots.ox.ac.uk/~maxbain/frozen-in-time/data/MSRVTT.zip"
EXPECTED_BYTES = 6552768292
OUT = Path(__file__).resolve().parents[1] / "runs/msrvtt_preflight_20260927/archive_download"
PART = OUT / "MSRVTT.zip.part"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / "MSRVTT.zip").exists():
        raise FileExistsError("Completed archive already exists")
    metadata = OUT / "download_job.json"
    if not metadata.exists():
        metadata.write_text(json.dumps({"source": URL, "expected_bytes": EXPECTED_BYTES,
                                        "started_unix": time.time(), "status": "downloading"}, indent=2))
    with requests.Session() as session:
        attempts = 0
        while True:
            current = PART.stat().st_size if PART.exists() else 0
            if current == EXPECTED_BYTES:
                break
            if current > EXPECTED_BYTES:
                raise RuntimeError("Partial file exceeds pinned archive size")
            headers = {"Range": f"bytes={current}-"} if current else {}
            try:
                with session.get(URL, headers=headers, stream=True, timeout=(30, 120)) as response:
                    response.raise_for_status()
                    if current and response.status_code != 206:
                        raise RuntimeError("Server did not honor resume range")
                    with PART.open("ab" if current else "xb") as stream:
                        last_report = time.monotonic()
                        for chunk in response.iter_content(chunk_size=1024 * 1024):
                            if chunk:
                                stream.write(chunk)
                            if time.monotonic() - last_report > 60:
                                print(json.dumps({"bytes": stream.tell(), "total": EXPECTED_BYTES,
                                                  "time_unix": time.time()}), flush=True)
                                stream.flush()
                                last_report = time.monotonic()
                attempts = 0
            except (requests.RequestException, TimeoutError) as exc:
                attempts += 1
                print(json.dumps({"retry": attempts, "error": str(exc),
                                  "bytes": PART.stat().st_size if PART.exists() else 0}), flush=True)
                if attempts >= 10:
                    raise
                time.sleep(min(60, attempts * 5))
    if PART.stat().st_size != EXPECTED_BYTES:
        raise RuntimeError("Download length mismatch")
    final = OUT / "MSRVTT.zip"
    PART.rename(final)
    metadata.write_text(json.dumps({"source": URL, "expected_bytes": EXPECTED_BYTES,
                                    "completed_unix": time.time(), "status": "downloaded_unverified",
                                    "path": str(final)}, indent=2))
    print(json.dumps({"status": "downloaded_unverified", "path": str(final)}), flush=True)


if __name__ == "__main__":
    main()
