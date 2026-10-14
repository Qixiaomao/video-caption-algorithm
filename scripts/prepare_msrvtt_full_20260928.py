"""Verify and extract all MSR-VTT v1.0 media into a new, auditable run.

No model is loaded and no test predictions or metrics are computed. This script
never changes the source ZIP, MSVD runs, or pre-existing source files.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
import zipfile

import cv2
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
PREFLIGHT = ROOT / "runs/msrvtt_preflight_20260927"
ARCHIVE = PREFLIGHT / "archive_download/MSRVTT.zip"
OUT = ROOT / "runs/msrvtt_data_full_20260928"
EXPECTED_ARCHIVE_SHA = "d8b3a79aee9891edcf9cf36f24b88dfcba596bac25b353e12460de41840c1894"
SPLIT_MEMBERS = {
    "train": ("MSRVTT/structured-symlinks/train_list_full.txt", "7f35fa96d45e94eed97dda0c2df98342a58fca769e7ffab352bd59c69c933251", 6513),
    "val": ("MSRVTT/structured-symlinks/val_list_full.txt", "be5b1994f643740b80796c392d7b472deed45ddce69ce13fee87609ce303ab3f", 497),
    "test": ("MSRVTT/structured-symlinks/test_list_full.txt", "e033df45731e025a1b7b4731ee3eddc400b6040c93fffd4ab3eb45a551f0526e", 2990),
}


def sha_bytes(blob: bytes) -> str:
    return hashlib.sha256(blob).hexdigest()


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_new(path: Path, obj: object) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def frame_indices(count: int) -> list[int]:
    if count < 1:
        raise ValueError("No decoded frames")
    return [round(i * (count - 1) / 7) for i in range(8)]


def decode_video(path: Path) -> tuple[list[object], int, int, bool]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError("OpenCV could not open media")
    declared = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    requested = frame_indices(declared) if declared > 0 else []
    wanted = set(requested)
    selected = {}
    actual = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if frame is None or frame.size == 0 or frame.shape[0] < 1 or frame.shape[1] < 1:
                raise RuntimeError(f"Invalid decoded frame {actual}")
            if actual in wanted:
                selected[actual] = frame.copy()
            actual += 1
    finally:
        capture.release()
    if actual < 1:
        raise RuntimeError("No decodable frame")
    mismatch = actual != declared
    indices = frame_indices(actual)
    if mismatch or any(index not in selected for index in indices):
        selected = {}
        wanted = set(indices)
        capture = cv2.VideoCapture(str(path))
        try:
            index = 0
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                if index in wanted:
                    selected[index] = frame.copy()
                index += 1
            if index != actual:
                raise RuntimeError(f"Decode count unstable: {actual} vs {index}")
        finally:
            capture.release()
    if any(index not in selected for index in indices):
        raise RuntimeError("Uniformly requested frame unavailable")
    return [selected[index] for index in indices], declared, actual, mismatch


def letterbox(frame, size: int = 224):
    height, width = frame.shape[:2]
    scale = min(size / width, size / height)
    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))
    resized = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)
    left = (size - new_width) // 2
    right = size - new_width - left
    top = (size - new_height) // 2
    bottom = size - new_height - top
    return cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(0, 0, 0))


def process_one(z: zipfile.ZipFile, video_id: str, split: str) -> dict:
    member = f"MSRVTT/videos/all/{video_id}.mp4"
    info = z.getinfo(member)
    media_dir = OUT / "media" / split
    frames_root = OUT / "frames" / split
    media_dir.mkdir(parents=True, exist_ok=True)
    frames_root.mkdir(parents=True, exist_ok=True)
    media = media_dir / f"{video_id}.mp4"
    temp_media = media_dir / f"{video_id}.mp4.part"
    frame_dir = frames_root / video_id
    temp_frames = frames_root / f"{video_id}.part"
    if any(path.exists() for path in (media, temp_media, frame_dir, temp_frames)):
        raise FileExistsError(f"Existing output requires manual recovery: {video_id}")
    started = time.perf_counter()
    digest = hashlib.sha256()
    with z.open(info) as source, temp_media.open("xb") as sink:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
            sink.write(block)
    if temp_media.stat().st_size != info.file_size:
        raise RuntimeError(f"Extracted length mismatch: {video_id}")
    extraction_s = time.perf_counter() - started
    temp_media.replace(media)
    started = time.perf_counter()
    selected, declared, decoded, mismatch = decode_video(media)
    decode_s = time.perf_counter() - started
    started = time.perf_counter()
    temp_frames.mkdir()
    frame_hashes = []
    frame_bytes = 0
    for i, frame in enumerate(selected):
        target = temp_frames / f"frame_{i + 1:05d}.jpg"
        if not cv2.imwrite(str(target), letterbox(frame), [cv2.IMWRITE_JPEG_QUALITY, 95]):
            raise RuntimeError(f"Could not write {target}")
        with Image.open(target) as image:
            image.verify()
        frame_hashes.append(sha_file(target))
        frame_bytes += target.stat().st_size
    if len(frame_hashes) != 8:
        raise RuntimeError("Eight-frame extraction incomplete")
    temp_frames.replace(frame_dir)
    frame_s = time.perf_counter() - started
    return {"video_id": video_id, "split": split, "archive_member": member,
            "archive_crc32": f"{info.CRC:08x}", "archive_compressed_bytes": info.compress_size,
            "media_path": str(media), "media_bytes": info.file_size, "media_sha256": digest.hexdigest(),
            "declared_frames": declared, "decoded_frames": decoded, "frame_count_metadata_mismatch": mismatch,
            "sample_indices": frame_indices(decoded), "frames_dir": str(frame_dir),
            "frame_sha256": frame_hashes, "frame_bytes": frame_bytes,
            "archive_extraction_seconds": extraction_s, "full_decode_seconds": decode_s,
            "frame_write_verify_seconds": frame_s, "status": "complete"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUT)
    args = parser.parse_args()
    if args.output.resolve() != OUT.resolve():
        raise ValueError("This protocol pins an isolated output path")
    archive_audit = json.loads((PREFLIGHT / "archive_download/download_verification.json").read_text())
    if archive_audit.get("status") != "pass" or archive_audit.get("sha256") != EXPECTED_ARCHIVE_SHA:
        raise RuntimeError("Pinned archive verification does not match")
    if OUT.exists():
        raise FileExistsError(f"Refusing to overwrite existing run: {OUT}")
    OUT.mkdir(parents=True)
    started = time.perf_counter()
    (OUT / "run_config.json").write_text(json.dumps({"started_at_utc": datetime.now(timezone.utc).isoformat(),
        "archive_path": str(ARCHIVE), "archive_sha256": EXPECTED_ARCHIVE_SHA,
        "script_path": str(Path(__file__).resolve()), "script_sha256": sha_file(Path(__file__)),
        "preprocess": "full sequential OpenCV decode; 8 round(linspace(0,N-1)) frames; 224x224 aspect-preserving black letterbox; JPEG quality 95",
        "short_video_rule": "if 1-7 frames decode, deterministic repeated uniform indices; zero frames fail",
        "test_use": "media and frame integrity only; no test model features, predictions, or metrics"}, indent=2), encoding="utf-8")
    with zipfile.ZipFile(ARCHIVE) as z:
        split_ids = {}
        for split, (member, expected_sha, count) in SPLIT_MEMBERS.items():
            blob = z.read(member)
            if sha_bytes(blob) != expected_sha:
                raise RuntimeError(f"Split hash mismatch: {split}")
            ids = blob.decode("utf-8-sig").splitlines()
            if len(ids) != count or len(set(ids)) != count:
                raise RuntimeError(f"Split count/duplicate error: {split}")
            split_ids[split] = ids
            target = OUT / "split_lists" / f"{split}_list_full.txt"
            target.parent.mkdir(exist_ok=True)
            target.write_bytes(blob)
        all_ids = [vid for split in ("train", "val", "test") for vid in split_ids[split]]
        if len(all_ids) != 10000 or len(set(all_ids)) != 10000:
            raise RuntimeError("Cross-split ID overlap or missing ID")
        annotations = z.read("MSRVTT/annotation/MSR_VTT.json")
        if sha_bytes(annotations) != "c6cb51bb725b76e78255f87c5fbb96886957ac53503a39bb03bff40b419cf996":
            raise RuntimeError("Annotation hash mismatch")
        (OUT / "MSR_VTT.json").write_bytes(annotations)
        parsed = json.loads(annotations)
        image_ids = [item["id"] for item in parsed["images"]]
        references = defaultdict(list)
        for item in parsed["annotations"]:
            references[item["image_id"]].append(item["caption"])
        if set(image_ids) != set(all_ids) or len(image_ids) != 10000 or any(not references[vid] for vid in all_ids):
            raise RuntimeError("Annotation/video ID coverage mismatch")
        media_members = [name for name in z.namelist() if name.startswith("MSRVTT/videos/all/video") and name.endswith(".mp4")]
        if len(media_members) != 10000 or set(media_members) != {f"MSRVTT/videos/all/{vid}.mp4" for vid in all_ids}:
            raise RuntimeError("Archive media ID coverage mismatch")
        journal = OUT / "media_audit.jsonl"
        failures = 0
        completed = 0
        with journal.open("x", encoding="utf-8") as stream:
            for split in ("train", "val", "test"):
                for vid in split_ids[split]:
                    try:
                        row = process_one(z, vid, split)
                        completed += 1
                    except Exception as exc:
                        failures += 1
                        row = {"video_id": vid, "split": split, "status": "failed",
                               "error_type": type(exc).__name__, "error": str(exc)}
                    stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                    stream.flush()
                    if (completed + failures) % 100 == 0:
                        print(json.dumps({"processed": completed + failures, "complete": completed,
                                          "failed": failures, "elapsed_seconds": round(time.perf_counter() - started, 1)}), flush=True)
    print(json.dumps({"processed": completed + failures, "complete": completed, "failed": failures,
                      "elapsed_seconds": round(time.perf_counter() - started, 1), "journal": str(journal)}), flush=True)


if __name__ == "__main__":
    main()
