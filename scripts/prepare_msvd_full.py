"""Build and validate the complete MSVD dataset from the pinned HF snapshot.

This script deliberately treats metadata, raw videos, and extracted frames as
three separate gates.  It exits non-zero before writing training annotations if
the official 1200/100/670 split, the 1970 video mapping, or ffprobe validation
does not pass exactly.

Example:
    python scripts/prepare_msvd_full.py \
      --metadata-root data/raw/msvd_full_hf \
      --video-root data/raw/msvd_full_hf/videos \
      --output-root data/processed/msvd_full \
      --archive data/raw/msvd_full_hf/MSVD_Videos.zip \
      --source-revision 9aa95a3039c106ee9e0969d82d7ce5a9efab1196 \
      --expected-archive-sha256 14563c61b50f600262bb50cb19a9170b34315ade723365f833a8d87c0675ee2a
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import shutil
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image

try:
    import cv2
except ImportError:  # ffmpeg remains a supported fallback
    cv2 = None


EXPECTED_SPLITS = {"train": 1200, "val": 100, "test": 670}
VIDEO_SUFFIXES = {".avi", ".mp4", ".mov", ".mkv", ".webm"}


class GateError(RuntimeError):
    """Raised when a dataset integrity gate fails."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def validate_metadata(metadata_root: Path) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    splits: dict[str, list[dict[str, Any]]] = {}
    all_ids: list[str] = []
    split_report: dict[str, Any] = {}

    for split, expected_count in EXPECTED_SPLITS.items():
        json_path = metadata_root / f"msvd_{split}.json"
        raw_list_path = metadata_root / "raw_data" / f"{split}_list.txt"
        if not json_path.is_file() or not raw_list_path.is_file():
            raise GateError(f"Missing metadata for split={split}: {json_path} or {raw_list_path}")

        records = read_json(json_path)
        if not isinstance(records, list):
            raise GateError(f"{json_path} must contain a JSON list")

        ids = [str(record.get("video_id", "")).strip() for record in records]
        raw_ids = [
            line.strip()
            for line in raw_list_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        empty_captions = [
            video_id
            for video_id, record in zip(ids, records)
            if not isinstance(record.get("caption"), list)
            or not [str(caption).strip() for caption in record["caption"] if str(caption).strip()]
        ]
        name_mismatches = [
            video_id
            for video_id, record in zip(ids, records)
            if Path(str(record.get("video", ""))).stem != video_id
        ]

        problems: list[str] = []
        if len(records) != expected_count:
            problems.append(f"records={len(records)} expected={expected_count}")
        if len(set(ids)) != expected_count or "" in ids:
            problems.append(f"unique_nonempty_ids={len(set(ids) - {''})} expected={expected_count}")
        if set(ids) != set(raw_ids) or len(raw_ids) != expected_count:
            problems.append("JSON IDs do not exactly match raw split list")
        if empty_captions:
            problems.append(f"empty_caption_records={len(empty_captions)}")
        if name_mismatches:
            problems.append(f"video_name_mismatches={len(name_mismatches)}")
        if problems:
            raise GateError(f"Metadata gate failed for {split}: " + "; ".join(problems))

        normalized: list[dict[str, Any]] = []
        for record in records:
            normalized.append(
                {
                    **record,
                    "video_id": str(record["video_id"]).strip(),
                    "caption": [
                        str(caption).strip()
                        for caption in record["caption"]
                        if str(caption).strip()
                    ],
                }
            )
        splits[split] = normalized
        all_ids.extend(ids)
        split_report[split] = {
            "videos": len(records),
            "captions": sum(len(record["caption"]) for record in normalized),
            "json": str(json_path.resolve()),
            "raw_split_list": str(raw_list_path.resolve()),
        }

    counts = Counter(all_ids)
    duplicate_ids = sorted(video_id for video_id, count in counts.items() if count != 1)
    if len(all_ids) != 1970 or len(counts) != 1970 or duplicate_ids:
        raise GateError(
            f"Global metadata gate failed: total={len(all_ids)}, unique={len(counts)}, "
            f"duplicate_or_overlapping={len(duplicate_ids)}"
        )

    return splits, {
        "valid": True,
        "total_videos": len(all_ids),
        "total_captions": sum(item["captions"] for item in split_report.values()),
        "splits": split_report,
    }


def index_videos(video_root: Path, expected_names: set[str]) -> tuple[dict[str, Path], list[str]]:
    candidates = sorted(
        path.resolve()
        for path in video_root.rglob("*")
        if path.is_file() and path.suffix.lower() in VIDEO_SUFFIXES
    )
    by_name: dict[str, list[Path]] = {}
    for path in candidates:
        by_name.setdefault(path.name, []).append(path)

    duplicate_names = sorted(name for name, paths in by_name.items() if len(paths) > 1)
    if duplicate_names:
        raise GateError(f"Duplicate raw video filenames: {duplicate_names[:10]}")

    missing = sorted(expected_names - set(by_name))
    unexpected = sorted(set(by_name) - expected_names)
    if missing or unexpected:
        raise GateError(
            f"Raw-video coverage gate failed: found={len(by_name)}, expected={len(expected_names)}, "
            f"missing={len(missing)} {missing[:10]}, unexpected={len(unexpected)} {unexpected[:10]}"
        )
    return {name: paths[0] for name, paths in by_name.items()}, unexpected


def probe_video(video_path: Path, ffprobe: str) -> dict[str, Any]:
    command = [
        ffprobe,
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=codec_name,width,height,avg_frame_rate,nb_frames,duration:format=duration",
        "-of",
        "json",
        str(video_path),
    ]
    result = subprocess.run(command, capture_output=True, text=True, timeout=60, check=False)
    if result.returncode != 0:
        raise GateError(result.stderr.strip() or f"ffprobe exited {result.returncode}")
    try:
        payload = json.loads(result.stdout)
        stream = payload["streams"][0]
    except (json.JSONDecodeError, KeyError, IndexError, TypeError) as exc:
        raise GateError(f"Invalid ffprobe output: {exc}") from exc
    duration_raw = stream.get("duration") or payload.get("format", {}).get("duration")
    try:
        duration = float(duration_raw)
        width = int(stream["width"])
        height = int(stream["height"])
    except (TypeError, ValueError, KeyError) as exc:
        raise GateError(f"Invalid video geometry/duration: {exc}") from exc
    if duration <= 0 or width <= 0 or height <= 0:
        raise GateError(f"Non-positive duration/geometry: duration={duration}, size={width}x{height}")
    return {
        "duration_seconds": duration,
        "width": width,
        "height": height,
        "codec": stream.get("codec_name"),
        "avg_frame_rate": stream.get("avg_frame_rate"),
        "nb_frames": stream.get("nb_frames"),
    }


def validate_frame_set(frame_dir: Path, num_frames: int, image_size: int) -> bool:
    frames = sorted(frame_dir.glob("frame_*.jpg"))
    if len(frames) != num_frames:
        return False
    try:
        for frame_path in frames:
            with Image.open(frame_path) as image:
                image.verify()
            with Image.open(frame_path) as image:
                if image.size != (image_size, image_size):
                    return False
    except (OSError, ValueError):
        return False
    return True


def extract_frames(
    video_path: Path,
    frame_dir: Path,
    duration: float,
    ffmpeg: str,
    num_frames: int,
    image_size: int,
) -> dict[str, Any]:
    if validate_frame_set(frame_dir, num_frames, image_size):
        return {"status": "reused", "decoded_frames": num_frames, "repeated_frames": 0}

    frame_dir.mkdir(parents=True, exist_ok=True)
    for stale_frame in frame_dir.glob("frame_*.jpg"):
        stale_frame.unlink()

    sample_fps = num_frames / duration
    video_filter = (
        f"fps=fps={sample_fps:.12f}:start_time=0,"
        f"scale={image_size}:{image_size}:force_original_aspect_ratio=decrease,"
        f"pad={image_size}:{image_size}:(ow-iw)/2:(oh-ih)/2"
    )
    command = [
        ffmpeg,
        "-v",
        "error",
        "-nostdin",
        "-y",
        "-i",
        str(video_path),
        "-vf",
        video_filter,
        "-frames:v",
        str(num_frames),
        "-q:v",
        "2",
        str(frame_dir / "frame_%05d.jpg"),
    ]
    result = subprocess.run(command, capture_output=True, text=True, timeout=180, check=False)
    frames = sorted(frame_dir.glob("frame_*.jpg"))
    if result.returncode != 0 or not frames:
        for stale_frame in frames:
            stale_frame.unlink()
        raise GateError(result.stderr.strip() or f"ffmpeg exited {result.returncode}")

    decoded_count = len(frames)
    repeated_count = 0
    while len(frames) < num_frames:
        destination = frame_dir / f"frame_{len(frames) + 1:05d}.jpg"
        shutil.copy2(frames[-1], destination)
        frames.append(destination)
        repeated_count += 1

    if not validate_frame_set(frame_dir, num_frames, image_size):
        raise GateError("Extracted frame-set validation failed")
    return {
        "status": "generated",
        "backend": "ffmpeg",
        "decoded_frames": decoded_count,
        "repeated_frames": repeated_count,
    }


def extract_frames_opencv(
    video_path: Path,
    frame_dir: Path,
    duration: float,
    ffmpeg: str,
    num_frames: int,
    image_size: int,
) -> dict[str, Any]:
    del duration, ffmpeg
    if cv2 is None:
        raise GateError("OpenCV backend requested but cv2 is not installed")
    if validate_frame_set(frame_dir, num_frames, image_size):
        return {
            "status": "reused",
            "backend": "opencv",
            "decoded_frames": num_frames,
            "repeated_frames": 0,
        }

    frame_dir.mkdir(parents=True, exist_ok=True)
    for stale_frame in frame_dir.glob("frame_*.jpg"):
        stale_frame.unlink()

    capture = cv2.VideoCapture(str(video_path))
    try:
        if not capture.isOpened():
            raise GateError("OpenCV could not open the video")
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            raise GateError(f"OpenCV reported invalid frame count: {total_frames}")
        if num_frames == 1:
            requested_indices = [max(0, (total_frames - 1) // 2)]
        else:
            requested_indices = [
                int(round(index * (total_frames - 1) / (num_frames - 1)))
                for index in range(num_frames)
            ]

        actual_indices: list[int] = []
        for output_index, requested_index in enumerate(requested_indices, start=1):
            frame = None
            actual_index = requested_index
            retry_indices = list(
                dict.fromkeys(
                    [
                        requested_index,
                        min(requested_index, total_frames - 2),
                        max(0, requested_index - 1),
                        max(0, requested_index - 5),
                    ]
                )
            )
            for candidate_index in retry_indices:
                capture.set(cv2.CAP_PROP_POS_FRAMES, candidate_index)
                ok, candidate_frame = capture.read()
                if ok and candidate_frame is not None and candidate_frame.size > 0:
                    frame = candidate_frame
                    actual_index = candidate_index
                    break
            if frame is None:
                raise GateError(
                    f"OpenCV could not decode requested frame {requested_index}/{total_frames}"
                )

            height, width = frame.shape[:2]
            scale = min(image_size / width, image_size / height)
            resized_width = max(1, int(round(width * scale)))
            resized_height = max(1, int(round(height * scale)))
            resized = cv2.resize(frame, (resized_width, resized_height), interpolation=cv2.INTER_AREA)
            left = (image_size - resized_width) // 2
            right = image_size - resized_width - left
            top = (image_size - resized_height) // 2
            bottom = image_size - resized_height - top
            square = cv2.copyMakeBorder(
                resized,
                top,
                bottom,
                left,
                right,
                cv2.BORDER_CONSTANT,
                value=(0, 0, 0),
            )
            destination = frame_dir / f"frame_{output_index:05d}.jpg"
            if not cv2.imwrite(str(destination), square, [cv2.IMWRITE_JPEG_QUALITY, 95]):
                raise GateError(f"OpenCV could not write {destination}")
            actual_indices.append(actual_index)
    finally:
        capture.release()

    if not validate_frame_set(frame_dir, num_frames, image_size):
        raise GateError("Extracted OpenCV frame-set validation failed")
    return {
        "status": "generated",
        "backend": "opencv",
        "decoded_frames": num_frames,
        "repeated_frames": num_frames - len(set(actual_indices)),
        "sampled_frame_indices": actual_indices,
    }


def metadata_hashes(metadata_root: Path) -> dict[str, dict[str, Any]]:
    relative_paths = [
        "README.md",
        "msvd_train.json",
        "msvd_val.json",
        "msvd_test.json",
        "raw_data/AllVideoDescriptions.txt",
        "raw_data/train_list.txt",
        "raw_data/val_list.txt",
        "raw_data/test_list.txt",
    ]
    result: dict[str, dict[str, Any]] = {}
    for relative_path in relative_paths:
        path = metadata_root / relative_path
        if path.is_file():
            result[relative_path] = {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata-root", type=Path, required=True)
    parser.add_argument("--video-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--source-repo", default="friedrichor/MSVD")
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--expected-archive-sha256", required=True)
    parser.add_argument("--num-frames", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1))
    parser.add_argument("--backend", choices=("auto", "opencv", "ffmpeg"), default="auto")
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--ffprobe", default="ffprobe")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    metadata_root = args.metadata_root.resolve()
    video_root = args.video_root.resolve()
    output_root = args.output_root.resolve()
    archive = args.archive.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    failures_path = output_root / "failures.json"
    manifest_path = output_root / "manifest.json"

    if args.num_frames <= 0 or args.image_size <= 0 or args.workers <= 0:
        raise GateError("num-frames, image-size, and workers must all be positive")
    backend = args.backend
    if backend == "auto":
        backend = "opencv" if cv2 is not None else "ffmpeg"
    if backend == "opencv" and cv2 is None:
        raise GateError("OpenCV backend requested but cv2 is not installed")
    if not archive.is_file():
        raise GateError(f"Archive not found: {archive}")
    archive_sha256 = sha256_file(archive)
    if archive_sha256.lower() != args.expected_archive_sha256.lower():
        raise GateError(
            f"Archive SHA-256 mismatch: actual={archive_sha256}, "
            f"expected={args.expected_archive_sha256.lower()}"
        )

    splits, metadata_report = validate_metadata(metadata_root)
    expected_names = {str(record["video"]) for records in splits.values() for record in records}
    video_index, _ = index_videos(video_root, expected_names)

    probes: dict[str, dict[str, Any]] = {}
    probe_failures: list[dict[str, str]] = []
    print(f"[1/3] Probing {len(video_index)} videos with {args.workers} workers...", flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        future_to_name = {
            executor.submit(probe_video, path, args.ffprobe): name
            for name, path in video_index.items()
        }
        for completed, future in enumerate(concurrent.futures.as_completed(future_to_name), start=1):
            name = future_to_name[future]
            try:
                probes[name] = future.result()
            except Exception as exc:  # keep every failure for the audit report
                probe_failures.append({"video": name, "stage": "ffprobe", "error": str(exc)})
            if completed % 100 == 0 or completed == len(future_to_name):
                print(f"  probed {completed}/{len(future_to_name)}", flush=True)
    if probe_failures:
        write_json(failures_path, probe_failures)
        raise GateError(f"ffprobe gate failed for {len(probe_failures)} videos; see {failures_path}")

    print(
        f"[2/3] Extracting {args.num_frames} frames for 1970 videos with {backend}...",
        flush=True,
    )
    extraction_results: dict[str, dict[str, Any]] = {}
    extraction_failures: list[dict[str, str]] = []
    jobs: list[tuple[str, str, Path, Path, float]] = []
    for split, records in splits.items():
        for record in records:
            video_name = str(record["video"])
            video_id = str(record["video_id"])
            frame_dir = output_root / split / "frames" / video_id
            jobs.append((split, video_id, video_index[video_name], frame_dir, probes[video_name]["duration_seconds"]))

    extractor = extract_frames_opencv if backend == "opencv" else extract_frames
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        future_to_job = {
            executor.submit(
                extractor,
                video_path,
                frame_dir,
                duration,
                args.ffmpeg,
                args.num_frames,
                args.image_size,
            ): (split, video_id, video_path, frame_dir)
            for split, video_id, video_path, frame_dir, duration in jobs
        }
        for completed, future in enumerate(concurrent.futures.as_completed(future_to_job), start=1):
            split, video_id, video_path, frame_dir = future_to_job[future]
            try:
                extraction_results[video_id] = future.result()
            except Exception as exc:  # keep every failure for the audit report
                extraction_failures.append(
                    {
                        "video_id": video_id,
                        "split": split,
                        "video": str(video_path),
                        "frames_dir": str(frame_dir),
                        "stage": "ffmpeg_extract",
                        "error": str(exc),
                    }
                )
            if completed % 100 == 0 or completed == len(future_to_job):
                print(f"  extracted/validated {completed}/{len(future_to_job)}", flush=True)

    if extraction_failures:
        write_json(failures_path, extraction_failures)
        raise GateError(
            f"Frame extraction gate failed for {len(extraction_failures)} videos; see {failures_path}"
        )

    print("[3/3] Writing annotations and final manifest...", flush=True)
    annotation_counts: dict[str, int] = {}
    for split, records in splits.items():
        annotations: list[dict[str, Any]] = []
        for record in records:
            video_id = str(record["video_id"])
            video_name = str(record["video"])
            frame_dir = (output_root / split / "frames" / video_id).resolve()
            annotations.append(
                {
                    "video_id": video_id,
                    "split": split,
                    "video": str(video_index[video_name]),
                    "frames_dir": str(frame_dir),
                    "captions": record["caption"],
                    "num_frames": args.num_frames,
                    "source": record.get("source"),
                }
            )
        annotation_path = output_root / split / "annotations.json"
        write_json(annotation_path, annotations)
        annotation_counts[split] = len(annotations)

    statuses = Counter(result["status"] for result in extraction_results.values())
    repeated_frames = sum(int(result["repeated_frames"]) for result in extraction_results.values())
    failures: list[dict[str, str]] = []
    write_json(failures_path, failures)
    manifest = {
        "schema_version": 1,
        "valid": True,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": {
            "repository": args.source_repo,
            "revision": args.source_revision,
            "metadata_root": str(metadata_root),
            "archive": {
                "path": str(archive),
                "bytes": archive.stat().st_size,
                "sha256": archive_sha256,
            },
            "metadata_files": metadata_hashes(metadata_root),
        },
        "metadata": metadata_report,
        "raw_videos": {
            "root": str(video_root),
            "expected": 1970,
            "found": len(video_index),
            "ffprobe_valid": len(probes),
            "missing": 0,
            "unexpected": 0,
        },
        "processed": {
            "root": str(output_root),
            "num_frames_per_video": args.num_frames,
            "image_size": [args.image_size, args.image_size],
            "extraction_backend": backend,
            "videos": len(extraction_results),
            "frames": len(extraction_results) * args.num_frames,
            "generated_videos": statuses.get("generated", 0),
            "reused_videos": statuses.get("reused", 0),
            "repeated_frames": repeated_frames,
            "annotation_counts": annotation_counts,
        },
        "failures": 0,
    }
    write_json(manifest_path, manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except GateError as exc:
        print(f"[DATASET GATE FAILED] {exc}", file=sys.stderr, flush=True)
        raise SystemExit(1)
