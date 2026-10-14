"""Finalize the full MSR-VTT media audit without touching the source archive.

This script fails closed. It never creates model features or test predictions.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import statistics
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs/msrvtt_data_full_20260928"
EXPECTED = {"train": 6513, "val": 497, "test": 2990}
SPLIT_SHA256 = {"train": "7f35fa96d45e94eed97dda0c2df98342a58fca769e7ffab352bd59c69c933251",
                "val": "be5b1994f643740b80796c392d7b472deed45ddce69ce13fee87609ce303ab3f",
                "test": "e033df45731e025a1b7b4731ee3eddc400b6040c93fffd4ab3eb45a551f0526e"}
ANNOTATION_SHA256 = "c6cb51bb725b76e78255f87c5fbb96886957ac53503a39bb03bff40b419cf996"


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_new(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)


def main() -> None:
    archive = ROOT / "runs/msrvtt_preflight_20260927/archive_download/MSRVTT.zip"
    archive_sha256 = sha_file(archive)
    if archive.stat().st_size != 6552768292 or archive_sha256 != "d8b3a79aee9891edcf9cf36f24b88dfcba596bac25b353e12460de41840c1894":
        raise RuntimeError("Complete local archive length/SHA-256 no longer matches pinned verification")
    lists = {split: (RUN / "split_lists" / f"{split}_list_full.txt").read_text(encoding="utf-8-sig").splitlines()
             for split in EXPECTED}
    for split, count in EXPECTED.items():
        if sha_file(RUN / "split_lists" / f"{split}_list_full.txt") != SPLIT_SHA256[split]:
            raise RuntimeError(f"Pinned split hash changed: {split}")
        if len(lists[split]) != count or len(set(lists[split])) != count:
            raise RuntimeError(f"Split cardinality/duplication error: {split}")
    ordered = [(split, vid) for split in EXPECTED for vid in lists[split]]
    all_id_set = {vid for _, vid in ordered}
    if len(all_id_set) != 10000:
        raise RuntimeError("Cross-split ID overlap")

    if sha_file(RUN / "MSR_VTT.json") != ANNOTATION_SHA256:
        raise RuntimeError("Pinned annotation hash changed")
    annotation = json.loads((RUN / "MSR_VTT.json").read_text(encoding="utf-8"))
    image_ids = [row["id"] for row in annotation["images"]]
    caption_ids = [row["id"] for row in annotation["annotations"]]
    if len(caption_ids) != 199994 or len(set(caption_ids)) != len(caption_ids):
        raise RuntimeError("Annotation caption count or caption-ID uniqueness error")
    if any(row["image_id"] not in all_id_set for row in annotation["annotations"]):
        raise RuntimeError("Annotation references ID outside the fixed split")
    by_id: dict[str, list[dict]] = defaultdict(list)
    for row in annotation["annotations"]:
        by_id[row["image_id"]].append({"caption_id": row["id"], "caption": row["caption"]})
    if len(image_ids) != 10000 or set(image_ids) != all_id_set:
        raise RuntimeError("Annotation coverage error")
    if any(not by_id[vid] for _, vid in ordered):
        raise RuntimeError("Missing caption references")
    caption_text_splits: dict[str, set[str]] = defaultdict(set)
    for split, vid in ordered:
        for ref in by_id[vid]:
            caption_text_splits[" ".join(ref["caption"].lower().split())].add(split)
    cross_split_caption_strings = sum(len(splits) > 1 for splits in caption_text_splits.values())

    journal = RUN / "media_audit.jsonl"
    rows = [json.loads(line) for line in journal.read_text(encoding="utf-8").splitlines()]
    if len(rows) != 10000 or [(row["split"], row["video_id"]) for row in rows] != ordered:
        raise RuntimeError("Media journal is incomplete or out of split order")
    media_hash_to_ids: dict[str, list[str]] = defaultdict(list)
    frame_hash_to_ids: dict[str, list[str]] = defaultdict(list)
    failures = []
    manifest = []
    frame_count_mismatch = []
    total_media_bytes = total_frame_bytes = 0
    timing = Counter()
    for row in rows:
        split, vid = row["split"], row["video_id"]
        references = by_id[vid]
        caption_payload = json.dumps(references, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
        item = {"video_id": vid, "split": split, "caption_count": len(references),
                "captions": references, "caption_sha256": hashlib.sha256(caption_payload).hexdigest(),
                "media_status": row["status"]}
        if row["status"] != "complete":
            failures.append({"video_id": vid, "split": split, "reason": row.get("error"),
                             "error_type": row.get("error_type")})
            manifest.append(item)
            continue
        media = Path(row["media_path"])
        frames = Path(row["frames_dir"])
        if not media.is_file() or media.stat().st_size != row["media_bytes"] or sha_file(media) != row["media_sha256"]:
            failures.append({"video_id": vid, "split": split, "reason": "media path/size/hash mismatch"})
            item["media_status"] = "failed_final_audit"
            manifest.append(item)
            continue
        frame_files = sorted(frames.glob("frame_*.jpg"))
        if ([path.name for path in frame_files] != [f"frame_{i:05d}.jpg" for i in range(1, 9)]
                or [sha_file(path) for path in frame_files] != row["frame_sha256"]):
            failures.append({"video_id": vid, "split": split, "reason": "frame count/hash mismatch"})
            item["media_status"] = "failed_final_audit"
            manifest.append(item)
            continue
        bad_image = False
        for frame_path in frame_files:
            try:
                with Image.open(frame_path) as image:
                    if image.size != (224, 224) or image.mode != "RGB":
                        bad_image = True
                        break
                    image.verify()
            except Exception:
                bad_image = True
                break
        if bad_image:
            failures.append({"video_id": vid, "split": split, "reason": "invalid 224x224 RGB JPEG frame"})
            item["media_status"] = "failed_final_audit"
            manifest.append(item)
            continue
        if row["decoded_frames"] < 1 or len(row["sample_indices"]) != 8:
            failures.append({"video_id": vid, "split": split, "reason": "invalid decode/frame selection"})
            item["media_status"] = "failed_final_audit"
            manifest.append(item)
            continue
        frame_digest = hashlib.sha256("\n".join(row["frame_sha256"]).encode()).hexdigest()
        media_hash_to_ids[row["media_sha256"]].append(vid)
        frame_hash_to_ids[frame_digest].append(vid)
        if row["frame_count_metadata_mismatch"]:
            frame_count_mismatch.append({"video_id": vid, "declared": row["declared_frames"],
                                         "decoded": row["decoded_frames"]})
        for key in ("archive_extraction_seconds", "full_decode_seconds", "frame_write_verify_seconds"):
            timing[key] += row[key]
        total_media_bytes += row["media_bytes"]
        total_frame_bytes += row["frame_bytes"]
        item.update({key: row[key] for key in ("archive_member", "archive_crc32", "archive_compressed_bytes",
                    "media_path", "media_bytes", "media_sha256", "declared_frames", "decoded_frames",
                    "frame_count_metadata_mismatch", "sample_indices", "frames_dir", "frame_sha256", "frame_bytes")})
        item["frame_set_sha256"] = frame_digest
        manifest.append(item)

    id_to_split = {vid: split for split, vid in ordered}
    exact_duplicates = [ids for ids in media_hash_to_ids.values() if len(ids) > 1]
    frame_duplicates = [ids for ids in frame_hash_to_ids.values() if len(ids) > 1]
    cross_split_media = [ids for ids in exact_duplicates if len({id_to_split[vid] for vid in ids}) > 1]
    cross_split_frames = [ids for ids in frame_duplicates if len({id_to_split[vid] for vid in ids}) > 1]
    if cross_split_media or cross_split_frames:
        failures.append({"reason": "cross-split exact media/frame duplicate",
                         "media_groups": cross_split_media, "frame_groups": cross_split_frames})
    status = "PASS" if not failures else "FAIL"
    audit = {"created_at_utc": datetime.now(timezone.utc).isoformat(), "status": status,
             "archive_path": str(archive), "archive_sha256_rechecked": archive_sha256,
             "scope": "media, split, annotation and extracted-frame audit only",
             "data_gate_status": "PENDING_FEATURE_CACHE" if status == "PASS" else "FAIL",
             "split_counts": {split: len(lists[split]) for split in EXPECTED},
             "split_sha256": SPLIT_SHA256, "annotation_sha256": ANNOTATION_SHA256,
             "unique_ids": len(set(vid for _, vid in ordered)),
             "caption_total": sum(len(by_id[vid]) for _, vid in ordered),
             "caption_counts": dict(Counter(len(by_id[vid]) for _, vid in ordered)),
             "cross_split_exact_caption_strings": cross_split_caption_strings,
             "caption_overlap_interpretation": "Shared text is not evidence of shared video media; media/frame hashes are checked separately.",
             "media_complete": sum(item["media_status"] == "complete" for item in manifest),
             "media_complete_by_split": {split: sum(item["media_status"] == "complete" and item["split"] == split
                                              for item in manifest) for split in EXPECTED},
             "decoded_frames_by_split": {split: {"minimum": min(row["decoded_frames"] for row in rows
                                                                 if row["split"] == split and row["status"] == "complete"),
                                                 "median": statistics.median(row["decoded_frames"] for row in rows
                                                                             if row["split"] == split and row["status"] == "complete")}
                                         for split in EXPECTED if any(row["split"] == split and row["status"] == "complete" for row in rows)},
             "failures": failures, "metadata_frame_count_mismatches": frame_count_mismatch,
             "exact_media_duplicate_groups": exact_duplicates,
             "exact_frame_set_duplicate_groups": frame_duplicates,
             "cross_split_exact_media_duplicate_groups": cross_split_media,
             "cross_split_exact_frame_set_duplicate_groups": cross_split_frames,
             "test_use": "file and extracted-frame integrity only; no model computation or metrics"}
    resource = {"created_at_utc": datetime.now(timezone.utc).isoformat(),
                "archive_bytes": (ROOT / "runs/msrvtt_preflight_20260927/archive_download/MSRVTT.zip").stat().st_size,
                "extracted_media_bytes": total_media_bytes, "extracted_frame_bytes": total_frame_bytes,
                "measured_task_seconds": dict(timing),
                "measured_task_seconds_sum": sum(timing.values()),
                "feature_cache_seconds": None, "feature_cache_bytes": None,
                "notes": "Full per-video timing sums; wall time and cache measurements added after cache completion."}
    write_new(RUN / "manifest.json", manifest)
    write_new(RUN / "data_audit.json", audit)
    write_new(RUN / "resource_estimate.json", resource)
    if status != "PASS":
        print(json.dumps({"status": status, "failures": len(failures), "audit": str(RUN / "data_audit.json")}))
        raise SystemExit(2)
    for split in ("train", "val"):
        records = [{"video_id": item["video_id"], "video": item["media_path"],
                    "frames_dir": item["frames_dir"],
                    "captions": [ref["caption"] for ref in item["captions"]]}
                   for item in manifest if item["split"] == split]
        write_new(RUN / "annotations" / f"{split}.json", records)
    print(json.dumps({"status": status, "manifest": str(RUN / "manifest.json"),
                      "audit": str(RUN / "data_audit.json"), "resource": str(RUN / "resource_estimate.json")}))


if __name__ == "__main__":
    main()
