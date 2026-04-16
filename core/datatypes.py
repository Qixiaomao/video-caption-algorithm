from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


@dataclass(frozen=True)
class CaptionCandidates:
    """Three candidate captions generated from one video tensor."""

    s1: str
    s2: str
    s3: str


@dataclass(frozen=True)
class InferenceResult:
    """Patent-friendly result structure for the full tensor flow."""

    candidates: CaptionCandidates
    best_key: str
    best_text: str

    def to_api_dict(self) -> Dict[str, object]:
        return {
            "S1": self.candidates.s1,
            "S2": self.candidates.s2,
            "S3": self.candidates.s3,
            "BEST": {"key": self.best_key, "text": self.best_text},
        }

