from __future__ import annotations

import argparse
import hashlib
import json
import random
import textwrap
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
RUN_ROOT = ROOT / "runs" / "test670_evaluation_20260908"
ANNOTATIONS = ROOT / "data" / "processed" / "msvd_full" / "test" / "annotations.json"
SALT = "bounded-qualitative-caption-audit-v1"
SAMPLE_SIZE = 12
MODELS = [(seed, arm) for seed in (123, 124, 125) for arm in ("A", "B")]


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def unique_prefix(values: list[str], limit: int) -> list[str]:
    output = []
    seen = set()
    for value in values:
        normalized = " ".join(value.split())
        if normalized and normalized not in seen:
            seen.add(normalized)
            output.append(normalized)
        if len(output) == limit:
            break
    return output


def select_records(records: list[dict]) -> list[dict]:
    ranked = sorted(
        records,
        key=lambda row: sha256_bytes(f"{SALT}\0{row['video_id']}".encode("utf-8")),
    )
    return ranked[:SAMPLE_SIZE]


def load_predictions() -> tuple[dict[tuple[int, str], dict[str, str]], dict[str, str]]:
    predictions = {}
    hashes = {}
    for seed, arm in MODELS:
        path = RUN_ROOT / f"predictions_seed{seed}_{arm}.json"
        rows = load_json(path)
        mapping = {row["video_id"]: row["caption"] for row in rows}
        if len(mapping) != 670:
            raise RuntimeError(f"{path.name} contains {len(mapping)} unique videos, expected 670")
        predictions[(seed, arm)] = mapping
        hashes[path.name] = sha256_file(path)
    return predictions, hashes


def fit_image(path: Path, width: int, height: int) -> Image.Image:
    with Image.open(path) as image:
        rgb = image.convert("RGB")
    ratio = max(width / rgb.width, height / rgb.height)
    resized = rgb.resize((round(rgb.width * ratio), round(rgb.height * ratio)), Image.Resampling.LANCZOS)
    left = max(0, (resized.width - width) // 2)
    top = max(0, (resized.height - height) // 2)
    return resized.crop((left, top, left + width, top + height))


def wrap_lines(text: str, width: int) -> list[str]:
    return textwrap.wrap(text, width=width, break_long_words=False, break_on_hyphens=False) or [""]


def make_sheet(entry: dict, output_path: Path) -> None:
    canvas = Image.new("RGB", (1600, 1050), "white")
    draw = ImageDraw.Draw(canvas)
    font_path = Path("C:/Windows/Fonts/arial.ttf")
    bold_path = Path("C:/Windows/Fonts/arialbd.ttf")
    font = ImageFont.truetype(str(font_path), 25)
    small = ImageFont.truetype(str(font_path), 21)
    bold = ImageFont.truetype(str(bold_path), 29)

    draw.text((30, 20), f"Audit {entry['audit_id']} | video {entry['video_id']}", fill="black", font=bold)
    frame_paths = [Path(value) for value in entry["frame_paths"]]
    for index, frame_path in enumerate(frame_paths):
        row, col = divmod(index, 4)
        x = 30 + col * 390
        y = 70 + row * 230
        canvas.paste(fit_image(frame_path, 370, 205), (x, y))
        draw.rectangle((x, y, x + 370, y + 205), outline="#333333", width=2)

    y = 545
    draw.text((30, y), "Dataset references (first six unique):", fill="#17365d", font=bold)
    y += 38
    for ref in entry["references"]:
        for line_index, line in enumerate(wrap_lines(ref, 118)):
            prefix = "- " if line_index == 0 else "  "
            draw.text((45, y), prefix + line, fill="#333333", font=small)
            y += 25

    y += 8
    draw.text((30, y), "Anonymous frozen predictions:", fill="#17365d", font=bold)
    y += 40
    for candidate in entry["candidates"]:
        label = candidate["label"]
        lines = wrap_lines(candidate["caption"], 108)
        for line_index, line in enumerate(lines):
            prefix = f"{label}: " if line_index == 0 else "    "
            draw.text((45, y), prefix + line, fill="black", font=font)
            y += 30
    canvas.save(output_path, quality=95)


def prepare(output_dir: Path) -> None:
    records = load_json(ANNOTATIONS)
    if len(records) != 670 or len({row["video_id"] for row in records}) != 670:
        raise RuntimeError("test annotations do not contain exactly 670 unique videos")
    predictions, prediction_hashes = load_predictions()
    selected = select_records(records)

    output_dir.mkdir(parents=True, exist_ok=False)
    sheets_dir = output_dir / "contact_sheets"
    sheets_dir.mkdir()

    blinded_entries = []
    key_entries = []
    for audit_index, record in enumerate(selected, start=1):
        video_id = record["video_id"]
        frame_paths = sorted(Path(record["frames_dir"]).glob("*.jpg"))
        if len(frame_paths) != 8:
            raise RuntimeError(f"{video_id} has {len(frame_paths)} JPG frames, expected 8")

        model_rows = [
            {"seed": seed, "arm": arm, "caption": predictions[(seed, arm)][video_id]}
            for seed, arm in MODELS
        ]
        shuffle_seed = int(sha256_bytes(f"{SALT}\0blind\0{video_id}".encode("utf-8"))[:16], 16)
        random.Random(shuffle_seed).shuffle(model_rows)
        candidates = [
            {"label": f"C{index}", "caption": row["caption"]}
            for index, row in enumerate(model_rows, start=1)
        ]
        audit_id = f"Q{audit_index:02d}"
        blinded = {
            "audit_id": audit_id,
            "video_id": video_id,
            "selection_sha256": sha256_bytes(f"{SALT}\0{video_id}".encode("utf-8")),
            "frame_paths": [str(path) for path in frame_paths],
            "references": unique_prefix(record["captions"], 6),
            "candidates": candidates,
        }
        key = {
            "audit_id": audit_id,
            "video_id": video_id,
            "mapping": [
                {"label": f"C{index}", "seed": row["seed"], "arm": row["arm"]}
                for index, row in enumerate(model_rows, start=1)
            ],
        }
        blinded_entries.append(blinded)
        key_entries.append(key)
        make_sheet(blinded, sheets_dir / f"{audit_id}_{video_id}.jpg")

    selection = {
        "protocol": "bounded-qualitative-caption-audit-v1",
        "salt": SALT,
        "sample_size": SAMPLE_SIZE,
        "annotation_path": str(ANNOTATIONS),
        "annotation_sha256": sha256_file(ANNOTATIONS),
        "prediction_sha256": prediction_hashes,
        "selected": [
            {
                "audit_id": entry["audit_id"],
                "video_id": entry["video_id"],
                "selection_sha256": entry["selection_sha256"],
            }
            for entry in blinded_entries
        ],
        "model_inference_performed": False,
        "test_tuning_performed": False,
    }
    write_json(output_dir / "selection.json", selection)
    write_json(output_dir / "blinded_manifest.json", blinded_entries)
    write_json(output_dir / "unblinding_key.json", key_entries)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.output.resolve())


if __name__ == "__main__":
    main()
