from __future__ import annotations

import re

from core.postprocessing.candidate_ranker import score_sentence


def _strip_acronyms_and_countries(text: str) -> str:
    text = re.sub(r"\bU\.S\.A?\.?\b", "", text, flags=re.I)
    text = re.sub(r"\bUSA\b", "", text, flags=re.I)
    text = re.sub(r"\bUnited States of America\b", "", text, flags=re.I)
    text = re.sub(r"\bUnited States\b", "", text, flags=re.I)
    text = re.sub(r"\bAmerica\b", "", text, flags=re.I)
    return re.sub(r"\s{2,}", " ", text).strip()


def _collapse_prep_chain(text: str) -> str:
    text = re.sub(r"(?i)\bin\s+the\s+front\s+of\b", "in front of", text)
    text = re.sub(r"(?i)\bin\s+the\s+middle\s+of\b", "in the middle of", text)
    text = re.sub(r"(?i)\bat\s+the\s+side\s+of\b", "at the side of", text)
    return re.sub(r"\s{2,}", " ", text)


def _ensure_sit_complement(text: str) -> str:
    lowered = text.strip().lower()
    if re.match(r"^someone\s+is\b", lowered):
        return text
    if re.match(r"^someone\s+is\s+sitting\s*\.?$", lowered):
        return "Someone is sitting on a chair."
    if re.match(r"^someone\s+is\s+sitting\b", lowered) and not re.search(r"\b(in|on|at|by|with|near)\b", lowered):
        return text.rstrip(". ") + " on a chair."
    return text


def _truncate_on_noise(text: str) -> str:
    if not text:
        return text
    tokens = text.split()
    cut = len(tokens)
    for index, token in enumerate(tokens):
        raw = token.strip(",.;:!?()[]{}\"'`")
        if not raw:
            continue
        if re.search(r"[0-9/\\]", raw):
            cut = index
            break
        if re.match(r"^(?:[A-Za-z]\.){2,}$", raw):
            cut = index
            break
        if re.match(r"^[A-Z]{1,3}-[A-Za-z0-9]{1,6}$", raw):
            cut = index
            break
        if len(raw) <= 3 and raw.isupper():
            cut = index
            break
    trimmed = " ".join(tokens[:cut] if cut < len(tokens) else tokens).strip()
    if trimmed and trimmed[-1] not in ".!?":
        trimmed += "."
    return trimmed


def _prune_weird_tails(text: str) -> str:
    text = re.sub(r"(?i)\b(?:how|why|what|that|which)\b.*$", "", text).strip()
    text = re.sub(r"(?i)\bA\s+wonders\b.*$", "", text).strip()
    return text or "Someone is in the scene."


def _ensure_period_and_caps(text: str) -> str:
    text = text.strip()
    if text and text[0].isalpha():
        text = text[0].upper() + text[1:]
    if text and text[-1] not in ".!?":
        text += "."
    return text


def clean_text(raw: str) -> str:
    """Clean raw decoder output into a subtitle-like sentence."""

    text = (raw or "").strip()
    if re.fullmatch(r"[-_= \t]{6,}\.?", text):
        return ""
    text = re.sub(r"^\s*[-_= \t]{2,}\s*", "", text)
    if (
        re.match(r"^\s*(https?://|www\.|<a\b|&lt;a\b)", text, re.I)
        or re.match(r"^\s*(copyright\b)", text, re.I)
        or re.fullmatch(r'"\s*[^"]+\s*"\.?', text)
    ):
        return ""

    bad_leads = (
        r"you are about to\b",
        r"click here\b",
        r"subscribe\b",
        r"available on youtube\b",
        r"watch live\b",
        r"find out\b",
        r"the video will\b",
        r"on the road\b",
    )
    if re.match(r"^\s*(?:" + "|".join(bad_leads) + r")", text, re.I):
        return ""
    if re.search(r"(</?\w+>|reddit\.com|pastebin|mailto:)", text, re.I):
        return ""

    flagged = bool(re.search(r"(?i)\b(click here|subscribe|report abuse|pastebin|official facebook|video will be)\b", text))
    text = re.sub(r"(?i)\b(click here|subscribe|report abuse|pastebin|official facebook|video will be.*)$", "", text).strip()
    text = _strip_acronyms_and_countries(text)
    text = _collapse_prep_chain(text)
    if len(text.split()) >= 10:
        text = _truncate_on_noise(text)
    text = _prune_weird_tails(text)
    if flagged and len(text.split()) <= 2:
        text = "Someone is in the scene."
    text = _ensure_sit_complement(text)
    text = re.sub(r"(?i)\b(\w+)\b(?:\s+\1\b)+", r"\1", text)
    text = _ensure_period_and_caps(re.sub(r"\s{2,}", " ", text).strip())

    parts = [chunk.strip() for chunk in re.split(r"\s*(?<=\.|\!|\?)\s+", text) if chunk.strip()]
    if len(parts) > 1:
        text = max(parts, key=score_sentence)
    return parts[0] if parts and parts[0] else text

