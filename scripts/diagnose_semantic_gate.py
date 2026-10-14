#!/usr/bin/env python3
"""Run a small, auditable semantic sanity check on an MSVD checkpoint."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import sys

# Allow both `python -m scripts.diagnose_semantic_gate` and direct execution.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch

from core.preprocessing.frame_loader import load_video_tensor
from src.models.caption_model import VideoCaptionModel


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_model(checkpoint_path: Path, device: str) -> tuple[VideoCaptionModel, dict]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict):
        checkpoint = {"model_state": checkpoint}
    checkpoint_args = checkpoint.get("args") or {}
    model = VideoCaptionModel(
        vit_name=checkpoint_args.get("vit_name", "vit_base_patch16_224"),
        gpt2_name=checkpoint_args.get("gpt2_name", "gpt2"),
        cond_mode=checkpoint_args.get("cond_mode", "prefix"),
        prefix_len=int(checkpoint_args.get("prefix_len", 4)),
        freeze_vit=True,
        vit_enable_torch_compile=False,
    ).to(device).eval()
    state_dict = checkpoint.get("model_state", checkpoint)
    incompatible = model.load_state_dict(state_dict, strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            f"incompatible checkpoint: missing={incompatible.missing_keys}, "
            f"unexpected={incompatible.unexpected_keys}"
        )
    return model, checkpoint


def resolve_frames_dir(record: dict, annotation_path: Path) -> Path:
    configured = record.get("frames_dir")
    if configured:
        return Path(configured)
    return annotation_path.parent / "frames" / record["video_id"]


def normalize(text: str) -> str:
    return " ".join(text.lower().strip().split())


def corpus_bleu(samples: list[dict]) -> float:
    """Dependency-free, unsmoothed corpus BLEU-4 for checkpoint triage."""

    if not samples:
        return 0.0
    clipped_totals = [0, 0, 0, 0]
    candidate_totals = [0, 0, 0, 0]
    hypothesis_length = 0
    reference_length = 0
    for sample in samples:
        hypothesis = normalize(sample["hypothesis"]).split()
        references = [normalize(text).split() for text in sample["references"] if text.strip()] or [[]]
        hypothesis_length += len(hypothesis)
        reference_length += min((len(ref) for ref in references), key=lambda length: (abs(length - len(hypothesis)), length))
        for order in range(1, 5):
            hypothesis_ngrams = Counter(
                tuple(hypothesis[index:index + order])
                for index in range(max(0, len(hypothesis) - order + 1))
            )
            max_reference_counts: Counter = Counter()
            for reference in references:
                reference_counts = Counter(
                    tuple(reference[index:index + order])
                    for index in range(max(0, len(reference) - order + 1))
                )
                for ngram, count in reference_counts.items():
                    max_reference_counts[ngram] = max(max_reference_counts[ngram], count)
            clipped_totals[order - 1] += sum(
                min(count, max_reference_counts[ngram]) for ngram, count in hypothesis_ngrams.items()
            )
            candidate_totals[order - 1] += sum(hypothesis_ngrams.values())
    precisions = [
        clipped / total if total else 0.0
        for clipped, total in zip(clipped_totals, candidate_totals)
    ]
    if hypothesis_length == 0 or any(precision == 0 for precision in precisions):
        return 0.0
    brevity_penalty = 1.0 if hypothesis_length > reference_length else math.exp(1 - reference_length / hypothesis_length)
    return 100.0 * brevity_penalty * math.exp(sum(math.log(precision) for precision in precisions) / 4)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--annotations", default="data/processed/msvd/val/annotations.json")
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max_new_tokens", type=int, default=24)
    parser.add_argument("--summary_only", action="store_true")
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    annotation_path = Path(args.annotations)
    records = json.loads(annotation_path.read_text(encoding="utf-8"))

    model, checkpoint = load_model(checkpoint_path, args.device)

    samples = []
    for record in records:
        frames_dir = resolve_frames_dir(record, annotation_path)
        if not frames_dir.exists() or not any(frames_dir.glob("frame_*.jpg")):
            continue
        video = load_video_tensor(frames_dir, num_frames=8, image_size=224, device=args.device)
        captions = model.generate(
            video,
            prompt="",
            max_new_tokens=args.max_new_tokens,
            min_new_tokens=0,
            num_beams=1,
            temperature=1.0,
            no_repeat_ngram_size=3,
            repetition_penalty=1.0,
        )
        samples.append(
            {
                "video_id": record.get("video_id", frames_dir.name),
                "references": record.get("captions", []),
                "hypothesis": captions[0].strip(),
            }
        )
        if len(samples) >= args.limit:
            break

    hypotheses = [sample["hypothesis"] for sample in samples]
    normalized = [normalize(text) for text in hypotheses]
    nonempty = sum(bool(text) for text in normalized)
    unique = len(set(normalized))
    caption_counts = Counter(normalized)
    largest_template_count = max(caption_counts.values(), default=0)
    result = {
        "protocol": "semantic-gate-v2",
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "training_contract": checkpoint.get("training_contract"),
        "decode": {
            "prompt": "",
            "synthetic_bos": False,
            "min_new_tokens": 0,
            "max_new_tokens": args.max_new_tokens,
            "num_beams": 1,
        },
        "sample_count": len(samples),
        "nonempty_rate": nonempty / len(samples) if samples else 0.0,
        "unique_rate": unique / len(samples) if samples else 0.0,
        "largest_template_rate": largest_template_count / len(samples) if samples else 0.0,
        "top_templates": [
            {"caption": caption, "count": count}
            for caption, count in caption_counts.most_common(10)
        ],
        "corpus_bleu4_unsmoothed": corpus_bleu(samples),
        "samples": samples,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    displayed = {key: value for key, value in result.items() if key != "samples"} if args.summary_only else result
    print(json.dumps(displayed, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
