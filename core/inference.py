from __future__ import annotations


def preset_to_kwargs(name: str):
    """Decode policy registry for repeatable inference and benchmarking."""

    name = (name or "precise").lower()
    if name == "precise":
        return dict(num_beams=3, max_new_tokens=24, temperature=1.0, top_p=1.0, no_repeat_ngram_size=3, repetition_penalty=1.1)
    if name == "detailed":
        return dict(num_beams=4, max_new_tokens=40, temperature=1.0, top_p=1.0, no_repeat_ngram_size=3, repetition_penalty=1.1)
    if name == "natural":
        return dict(num_beams=1, max_new_tokens=24, temperature=0.9, top_p=0.9, no_repeat_ngram_size=3, repetition_penalty=1.05)
    if name == "safe_sample":
        return dict(num_beams=1, max_new_tokens=22, temperature=0.8, top_p=0.85, no_repeat_ngram_size=3, repetition_penalty=1.1)
    return preset_to_kwargs("precise")

