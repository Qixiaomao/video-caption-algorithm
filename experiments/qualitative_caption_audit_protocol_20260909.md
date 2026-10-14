# Frozen bounded qualitative caption audit protocol

Frozen before inspecting the selected videos or comparing the six model
outputs on 2026-09-09.

## Purpose and boundary

This audit checks whether the frozen test670 captions describe the visible
subject and action, whether they introduce unsupported content, and whether
the InfoNCE arm shows an obvious qualitative trade-off. It is a small
AI-assisted error audit, not a human evaluation, statistical test, checkpoint
selection procedure, or decoding-tuning procedure.

No model inference, training, checkpoint selection, decoding change, or
efficiency benchmark is allowed during this audit. The six existing prediction
files from `runs/test670_evaluation_20260908` are the only model outputs used.

## Frozen sample selection

- Population: the 670 video IDs in the pinned test annotation order.
- Salt: `bounded-qualitative-caption-audit-v1`.
- Score: SHA-256 of `salt + "\0" + video_id`, interpreted lexicographically.
- Sample: the 12 video IDs with the smallest scores.
- This rule depends only on video ID. It does not inspect references,
  predictions, automatic metrics, or A/B differences.

## Blinding

For each selected video, the six predictions (seeds 123/124/125, arms A/B)
are assigned anonymous labels C1-C6 using a deterministic per-video shuffle.
The evaluator sees the eight extracted frames, up to six unique dataset
references, and the six anonymous captions. The arm/seed key remains unopened
until all ratings are written.

## Rating rubric

Each anonymous caption receives the following independent ratings:

| Field | Values | Meaning |
|---|---|---|
| subject | 0/1 | Main visible subject is wrong / acceptably described |
| action | 0/1 | Main action is wrong / acceptably described |
| context | 0/1 | Object or scene detail is contradictory / compatible |
| hallucination | 0/1 | Contains a material unsupported claim / does not |
| fluency | 0/1 | Broken or nonsensical English / readable English |

`semantic_score = subject + action + context`, from 0 to 3. A caption counts
as broadly grounded only when `semantic_score >= 2` and `hallucination = 0`.
Within each paired seed, B is qualitatively better than A only if it has a
higher semantic score, or the same semantic score with less hallucination.
The reverse rule marks A better; otherwise the pair is a tie. Fluency is
reported separately and does not override semantic correctness.

## Reporting

Report:

1. arm-level grounded-caption, hallucination, and fluent-caption counts;
2. paired B-better / A-better / tie counts across 36 seed-video pairs;
3. recurring error categories with representative examples;
4. explicit limitations and whether the result changes the automatic-metric
   interpretation.

The report must not claim human preference, semantic significance, or general
test-set prevalence from this 12-video sample. The test split remains closed
to any subsequent tuning.
