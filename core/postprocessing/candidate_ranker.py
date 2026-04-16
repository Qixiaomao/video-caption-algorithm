from __future__ import annotations

import re
from typing import Iterable, Tuple


def score_sentence(text: str) -> float:
    """Heuristic candidate score used to choose the final subtitle."""

    if not text:
        return -1e9
    tokens = text.split()
    n_tokens = len(tokens)
    score = 0.0
    mu, sigma = 12.0, 4.0
    score += -((n_tokens - mu) ** 2) / (2 * sigma * sigma)
    if re.search(r"\b\w+ing\b", text):
        score += 1.0
    if re.search(r"\b(?:is|are|was|were)\b", text):
        score += 0.5
    if text.endswith((".", "!", "?")):
        score += 0.3
    if re.search(r"\b(?:[A-Z]\.){2,}\b", text):
        score -= 1.5
    if re.search(r"(?i)\b(click here|subscribe|report abuse|sign up|pastebin)\b", text):
        score -= 1.5
    if n_tokens < 4:
        score -= 2.0
    if text.strip().lower() in {"someone is sitting.", "someone is in the scene."}:
        score -= 0.8
    return score


def select_best(candidates: Iterable[Tuple[str, str]]):
    scored = [(key, value, score_sentence(value)) for key, value in candidates]
    return sorted(scored, key=lambda item: item[2], reverse=True)[0]

