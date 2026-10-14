"""Read MSR-VTT ZIP metadata and selected members with HTTP byte ranges.

This is a preflight probe, not a substitute for downloading and hashing every
video before a formal experiment. It never reads model predictions or scores.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import struct
import time
import zlib
from collections import Counter
from pathlib import Path

import requests


URL = "https://www.robots.ox.ac.uk/~maxbain/frozen-in-time/data/MSRVTT.zip"


def range_get(session: requests.Session, start: int, end: int) -> tuple[bytes, float]:
    started = time.perf_counter()
    response = session.get(URL, headers={"Range": f"bytes={start}-{end}"}, timeout=90)
    response.raise_for_status()
    if response.status_code != 206 or len(response.content) != end - start + 1:
        raise RuntimeError(f"Invalid range response: {response.status_code}, {len(response.content)}")
    return response.content, time.perf_counter() - started


def index_zip(session: requests.Session) -> tuple[list[dict], dict]:
    head = session.head(URL, timeout=30)
    head.raise_for_status()
    size = int(head.headers["Content-Length"])
    tail_start = size - 2 * 1024 * 1024
    tail, elapsed = range_get(session, tail_start, size - 1)
    eocd = tail.rfind(b"PK" + bytes([5, 6]))
    if eocd < 20:
        raise RuntimeError("ZIP end record unavailable")
    locator = struct.unpack_from("<4sLQL", tail, eocd - 20)
    if locator[0] != b"PK" + bytes([6, 7]):
        raise RuntimeError("ZIP64 locator unavailable")
    zip64_offset = locator[2] - tail_start
    record = struct.unpack_from("<4sQ2H2L4Q", tail, zip64_offset)
    if record[0] != b"PK" + bytes([6, 6]):
        raise RuntimeError("ZIP64 record unavailable")
    count, directory_size, directory_offset = record[7:10]
    if not (0 <= directory_offset - tail_start < len(tail)):
        raise RuntimeError("Central directory outside range sample")
    cursor = directory_offset - tail_start
    entries = []
    for _ in range(count):
        item = struct.unpack_from("<4s6H3L5H2L", tail, cursor)
        if item[0] != b"PK" + bytes([1, 2]):
            raise RuntimeError("Bad central directory entry")
        name_len, extra_len, comment_len = item[10:13]
        name = tail[cursor + 46:cursor + 46 + name_len].decode("utf-8")
        extra = tail[cursor + 46 + name_len:cursor + 46 + name_len + extra_len]
        local_offset = item[16]
        if local_offset == 0xFFFFFFFF:
            at = 0
            while at + 4 <= len(extra):
                typ, length = struct.unpack_from("<HH", extra, at)
                value = extra[at + 4:at + 4 + length]
                if typ == 1:
                    pos = 0
                    if item[9] == 0xFFFFFFFF:
                        pos += 8
                    if item[8] == 0xFFFFFFFF:
                        pos += 8
                    local_offset = struct.unpack_from("<Q", value, pos)[0]
                    break
                at += 4 + length
        entries.append({"name": name, "method": item[4], "crc32": f"{item[7]:08x}",
                        "compressed_bytes": item[8], "uncompressed_bytes": item[9],
                        "local_offset": local_offset})
        cursor += 46 + name_len + extra_len + comment_len
    if cursor != directory_offset - tail_start + directory_size:
        raise RuntimeError("Central directory length mismatch")
    return entries, {"url": URL, "content_length_bytes": size,
                     "last_modified": head.headers.get("Last-Modified"),
                     "accept_ranges": head.headers.get("Accept-Ranges"),
                     "tail_range_bytes": len(tail), "tail_download_seconds": elapsed}


def extract_member(session: requests.Session, entry: dict) -> tuple[bytes, dict]:
    header, header_seconds = range_get(session, entry["local_offset"], entry["local_offset"] + 29)
    if header[:4] != b"PK" + bytes([3, 4]):
        raise RuntimeError(f"Bad member header: {entry['name']}")
    name_len, extra_len = struct.unpack_from("<HH", header, 26)
    start = entry["local_offset"] + 30 + name_len + extra_len
    raw, data_seconds = range_get(session, start, start + entry["compressed_bytes"] - 1)
    if entry["method"] == 0:
        payload = raw
    elif entry["method"] == 8:
        payload = zlib.decompress(raw, -15)
    else:
        raise RuntimeError(f"Unsupported compression {entry['method']}")
    if len(payload) != entry["uncompressed_bytes"] or f"{zlib.crc32(payload):08x}" != entry["crc32"]:
        raise RuntimeError(f"Integrity check failed: {entry['name']}")
    return payload, {"member": entry["name"], "compressed_bytes": len(raw),
                     "uncompressed_bytes": len(payload), "header_seconds": header_seconds,
                     "download_seconds": data_seconds, "sha256": hashlib.sha256(payload).hexdigest()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    with requests.Session() as session:
        entries, archive = index_zip(session)
        annotation = next(e for e in entries if e["name"] == "MSRVTT/annotation/MSR_VTT.json")
        blob, annotation_probe = extract_member(session, annotation)
        (args.output / "MSR_VTT.json").write_bytes(blob)
        videos = [e for e in entries if e["name"].startswith("MSRVTT/videos/all/video") and e["name"].endswith(".mp4")]
        samples = []
        for low, high in [(0, 6512), (6513, 7009), (7010, 9999)]:
            candidates = [e for e in videos if low <= int(Path(e["name"]).stem[5:]) <= high]
            entry = min(candidates, key=lambda e: e["compressed_bytes"])
            payload, audit = extract_member(session, entry)
            name = Path(entry["name"]).name
            (args.output / name).write_bytes(payload)
            samples.append(audit)
    (args.output / "archive_index.json").write_text(json.dumps({"archive": archive, "entries": entries}, indent=2), encoding="utf-8")
    (args.output / "range_probe.json").write_text(json.dumps({"annotation": annotation_probe, "video_samples": samples,
        "archive_entry_count": len(entries), "video_entry_count": len(videos),
        "video_id_count": len(set(Path(e["name"]).stem for e in videos)),
        "video_duplicates": {k: v for k, v in Counter(Path(e["name"]).stem for e in videos).items() if v > 1}}, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "videos": len(videos), "samples": samples}, indent=2))


if __name__ == "__main__":
    main()
