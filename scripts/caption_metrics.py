"""COCO-caption metrics for one prediction and many references per video."""
from __future__ import annotations

from importlib.metadata import version
import math
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import threading
from typing import Any


METRIC_NAMES = (
    "Bleu_1",
    "Bleu_2",
    "Bleu_3",
    "Bleu_4",
    "METEOR",
    "ROUGE_L",
    "CIDEr",
)


def _plain_number(value: Any) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"Nonfinite caption metric: {number}")
    return number


def validate_caption_inputs(
    records: list[dict[str, Any]], predictions: list[dict[str, str]]
) -> tuple[list[str], dict[int, list[dict[str, str]]], dict[int, list[dict[str, str]]]]:
    """Validate one-to-one coverage and build pycocoevalcap dictionaries."""
    if not records or len(records) != len(predictions):
        raise ValueError("Records and predictions must have equal nonzero length")
    video_ids = [str(record.get("video_id", "")) for record in records]
    prediction_ids = [str(item.get("video_id", "")) for item in predictions]
    if any(not video_id for video_id in video_ids) or len(set(video_ids)) != len(video_ids):
        raise ValueError("Reference video IDs must be nonempty and unique")
    if prediction_ids != video_ids:
        raise ValueError("Prediction IDs or order differ from references")
    gts: dict[int, list[dict[str, str]]] = {}
    res: dict[int, list[dict[str, str]]] = {}
    for index, (record, prediction) in enumerate(zip(records, predictions)):
        references = record.get("captions")
        caption = prediction.get("caption")
        if not isinstance(references, list) or not references:
            raise ValueError(f"Video {video_ids[index]} has no reference captions")
        cleaned = [text.strip() for text in references if isinstance(text, str) and text.strip()]
        if len(cleaned) != len(references):
            raise ValueError(f"Video {video_ids[index]} contains an invalid reference")
        if not isinstance(caption, str) or not caption.strip():
            raise ValueError(f"Video {video_ids[index]} has an empty prediction")
        gts[index] = [{"caption": text} for text in cleaned]
        res[index] = [{"caption": caption.strip()}]
    return video_ids, gts, res


def compute_caption_metrics(
    records: list[dict[str, Any]], predictions: list[dict[str, str]]
) -> dict[str, Any]:
    """Compute PTB-tokenized BLEU, METEOR, ROUGE-L, and CIDEr."""
    try:
        from pycocoevalcap.bleu.bleu import Bleu
        from pycocoevalcap.cider.cider import Cider
        from pycocoevalcap.meteor import meteor as meteor_module
        from pycocoevalcap.rouge.rouge import Rouge
        from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
    except ImportError as error:
        raise RuntimeError(
            "pycocoevalcap==1.2 is required for the locked test evaluation"
        ) from error

    class ManagedMeteor(meteor_module.Meteor):
        """Close every METEOR subprocess pipe; upstream 1.2 leaves two open."""

        def __init__(self) -> None:
            source_dir = Path(meteor_module.__file__).resolve().parent
            self._meteor_runtime = None
            if os.name != "nt" or " " not in str(source_dir):
                super().__init__()
                return

            # METEOR 1.5 double-encodes spaces in the JAR resource URL on
            # Windows. Launch an unchanged copy from the no-space temp path.
            self._meteor_runtime = tempfile.TemporaryDirectory(prefix="coco_meteor_")
            runtime_dir = Path(self._meteor_runtime.name)
            shutil.copytree(source_dir, runtime_dir, dirs_exist_ok=True)
            self.meteor_cmd = [
                "java",
                "-jar",
                "-Xmx2G",
                meteor_module.METEOR_JAR,
                "-",
                "-",
                "-stdio",
                "-l",
                "en",
                "-norm",
            ]
            self.meteor_p = subprocess.Popen(
                self.meteor_cmd,
                cwd=runtime_dir,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            self.lock = threading.Lock()

        def close(self) -> None:
            process = getattr(self, "meteor_p", None)
            if process is None:
                runtime = getattr(self, "_meteor_runtime", None)
                if runtime is not None:
                    runtime.cleanup()
                    self._meteor_runtime = None
                return
            try:
                if process.stdin is not None and not process.stdin.closed:
                    try:
                        process.stdin.close()
                    except OSError:
                        pass
                if process.poll() is None:
                    process.kill()
                process.wait()
            finally:
                for stream in (process.stdout, process.stderr):
                    if stream is not None and not stream.closed:
                        stream.close()
                self.meteor_p = None
                runtime = getattr(self, "_meteor_runtime", None)
                if runtime is not None:
                    runtime.cleanup()
                    self._meteor_runtime = None

        def __del__(self) -> None:
            self.close()

    video_ids, gts, res = validate_caption_inputs(records, predictions)
    tokenizer = PTBTokenizer()
    tokenized_gts = tokenizer.tokenize(gts)
    tokenized_res = tokenizer.tokenize(res)
    scorers = (
        (Bleu(4), ("Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4")),
        (ManagedMeteor(), ("METEOR",)),
        (Rouge(), ("ROUGE_L",)),
        (Cider(), ("CIDEr",)),
    )
    corpus: dict[str, float] = {}
    per_video = [{"video_id": video_id} for video_id in video_ids]
    try:
        for scorer, names in scorers:
            score, scores = scorer.compute_score(tokenized_gts, tokenized_res)
            if len(names) == 1:
                score_values = [score]
                per_values = [scores]
            else:
                score_values = list(score)
                per_values = list(scores)
            for name, aggregate, individual in zip(names, score_values, per_values):
                corpus[name] = _plain_number(aggregate)
                values = [_plain_number(value) for value in individual]
                if len(values) != len(video_ids):
                    raise RuntimeError(f"{name} returned the wrong per-video length")
                for row, value in zip(per_video, values):
                    row[name] = value
    finally:
        for scorer, _ in scorers:
            close = getattr(scorer, "close", None)
            if close is not None:
                close()
    if tuple(corpus) != METRIC_NAMES:
        raise RuntimeError(f"Unexpected metric set or order: {tuple(corpus)}")
    return {
        "implementation": {
            "package": "pycocoevalcap",
            "version": version("pycocoevalcap"),
            "tokenizer": "PTBTokenizer",
            "raw_scale": "native evaluator scale",
            "paper_scale": "raw score multiplied by 100",
        },
        "corpus_raw": corpus,
        "corpus_x100": {name: value * 100.0 for name, value in corpus.items()},
        "per_video_raw": per_video,
    }
