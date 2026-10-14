# InfoNCE stage-2 full-data ablation (2026-09-02)

> Follow-up (2026-09-03): the val32 intervention experiment in
> `prefix_sensitivity_val32_20260903.md` finds positive correct-versus-shuffled
> caption NLL advantages for both E0 and E6. First-token raw-logit similarity
> and a common initial article do not establish that GPT-2 ignores visual
> conditioning. This refines the causal interpretation below; the measured
> stage-2 values and the val100 generation-gate failure remain unchanged.

## Decision

Stage 2 is complete, but **no E3-E6 checkpoint passes the frozen semantic
gate**. InfoNCE produces a real retrieval improvement over the positive-only
control, and the strongest alignment weight partly restores visual-prefix
diversity, but generated captions remain dominated by one template. None of
these checkpoints is therefore eligible for the held-out test split or for a
paper-quality result table.

E6 (`InfoNCE`, `lambda=0.20`) is the most useful diagnostic checkpoint because
it has the largest positive-minus-negative similarity gap and is the only run
to recover the v5 prefix effective-rank baseline. E5 (`InfoNCE`, `lambda=0.10`)
has the best text-to-video R@1. This is evidence that negative samples help the
alignment objective; it is not evidence that the current text representation
repairs conditional caption generation.

## Frozen protocol

- Dataset: pinned full MSVD, Hugging Face revision
  `9aa95a3039c106ee9e0969d82d7ce5a9efab1196`.
- Train / validation: 1,200 / 100 videos. The 670-video test split was not
  loaded by any stage-2 command.
- Source checkpoint: `checkpoints/msvd_mapper_recovered_v5.pt`, 845,063,319
  bytes, SHA-256
  `17dda154899f48c89d12aa02e0014c01ac80af7a8965f41c330fe8d91cb1eec3`.
- Train annotations SHA-256:
  `ced443e28c5ad2ddd0780af55a820b5ec7da6462c092e1329951dc9c7598c7c5`.
- Validation annotations SHA-256:
  `bcda3ba0583704a54297e4da4cc9acbcc5ec1ebffd6ed436a6ba3ee1b8a03c8b`.
- GPU: NVIDIA GeForce RTX 3050 Laptop GPU, 4,096 MiB.
- Physical batch size 8, seed 123, one epoch / 150 optimizer steps, validation
  every 50 steps, and fully frozen GPT-2.
- Text representation: masked mean of frozen GPT-2 token embeddings.
- InfoNCE: symmetric in-batch objective, temperature `0.07`, with
  exact-caption multi-positive masking.
- Every comparison below uses the step-150 `.last.pt` checkpoint. Selecting a
  per-run best checkpoint would give different training exposure and is not
  allowed in this ablation.

## Experiment matrix

| ID | Alignment objective | Weight | Role |
|---|---|---:|---|
| E0 | none | 0.00 | recovered-v5 pre-training reference |
| E3 | positive cosine | 0.10 | no-negative control |
| E4 | InfoNCE | 0.05 | low alignment weight |
| E5 | InfoNCE | 0.10 | matched-weight comparison with E3 |
| E6 | InfoNCE | 0.20 | high alignment weight |

## End-of-run validation and memory

| ID | Total loss | Caption loss | Alignment loss | Positive cosine | Negative cosine | Gap | Peak allocated / reserved MB |
|---|---:|---:|---:|---:|---:|---:|---:|
| E3 | 3.2231 | 3.1678 | 0.5530 | 0.4470 | 0.4380 | 0.0090 | 2,330.9 / 2,542.0 |
| E4 | 3.2755 | 3.1785 | 1.9409 | 0.1372 | 0.1241 | 0.0131 | 2,330.9 / 2,542.0 |
| E5 | 3.3495 | 3.1589 | 1.9056 | 0.1422 | 0.1260 | 0.0162 | 2,330.9 / 2,542.0 |
| E6 | 3.5405 | 3.1697 | 1.8538 | 0.1315 | 0.1103 | 0.0213 | 2,330.9 / 2,542.0 |

All four runs completed without OOM, NaN, or checkpoint failure. The alignment
weight did not change peak memory at the recorded precision.

## Semantic-gate results on validation 100

| ID | Non-empty rate | Unique-caption rate | Largest-template rate | Unsmoothed triage BLEU-4 | Most frequent prediction |
|---|---:|---:|---:|---:|---|
| E0 | 1.00 | 0.14 | 0.24 | 22.21 | `a woman is slicing a carrot` (24) |
| E3 | 1.00 | 0.11 | 0.56 | 35.99 | `a man is playing a guitar` (56) |
| E4 | 1.00 | 0.13 | 0.56 | 36.46 | `a man is playing a guitar` (56) |
| E5 | 1.00 | 0.14 | 0.52 | 36.20 | `a man is playing a guitar` (52) |
| E6 | 1.00 | 0.17 | 0.52 | 36.71 | `a man is playing a guitar` (52) |

The BLEU column is an unsmoothed triage metric used only to keep this recovery
workflow comparable with its earlier diagnostics. It is not an official MSVD
caption score. Its increase cannot override the diversity failure: every
trained checkpoint predicts one sentence for at least 52% of validation
videos, compared with the frozen maximum of 25%.

## Visual-path diagnostics on validation 32

| ID | Encoder pair cosine | Encoder effective rank | Prefix pair cosine | Prefix effective rank | First-token-logit pair cosine | First-token-logit effective rank | Top-1 unique rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| E0 | 0.5785 | 10.34 | 0.7507 | 5.57 | 0.999668 | 1.016 | 0.03125 |
| E3 | 0.6831 | 9.24 | 0.8782 | 4.02 | 0.999893 | 1.020 | 0.03125 |
| E4 | 0.6530 | 9.70 | 0.8494 | 4.87 | 0.999835 | 1.023 | 0.03125 |
| E5 | 0.6467 | 9.73 | 0.8424 | 4.92 | 0.999835 | 1.026 | 0.03125 |
| E6 | 0.6247 | 10.07 | 0.8109 | 5.68 | 0.999778 | 1.016 | 0.03125 |

The positive-only control makes prefix collapse worse. Increasing the InfoNCE
weight reverses part of that damage: E6 recovers effective rank to 5.68, just
above E0's 5.57. However, E6 still misses the predeclared prefix-cosine ceiling
of 0.80, and the first-token logits remain almost rank one for every model.
This is consistent with a remaining break at or after the
decoder-conditioning/logit interface, but the diagnostic does not by itself
isolate the exact causal layer.

## Retrieval diagnostics on validation 100

This diagnostic embeds each video as the normalized mean visual prefix and
each caption as the normalized masked mean of frozen GPT-2 WTE embeddings. It
uses all 4,290 validation captions with real video labels and a deterministic
deranged-label control. These values are **not paper-ready retrieval metrics**.

| ID | Text-to-video R@1 / R@5 / R@10 | Video-to-text R@1 / R@5 / R@10 | Positive-negative similarity gap | Shuffled text-to-video R@1 | Shuffled video-to-text R@1 |
|---|---:|---:|---:|---:|---:|
| E0 | 1.82 / 9.91 / 17.02 | 2.00 / 4.00 / 10.00 | 0.0054 | 1.38 | 1.00 |
| E3 | 2.59 / 11.61 / 20.75 | 5.00 / 14.00 / 18.00 | 0.0085 | 0.33 | 2.00 |
| E4 | 6.22 / 22.54 / 38.41 | 6.00 / 15.00 / 25.00 | 0.0119 | 0.21 | 2.00 |
| E5 | **8.34** / 28.95 / 44.76 | 5.00 / 19.00 / 26.00 | 0.0147 | 0.09 | 2.00 |
| E6 | 7.79 / **29.09** / **46.22** | 5.00 / **21.00** / **27.00** | **0.0193** | 0.49 | 2.00 |

InfoNCE beats the positive-only E3 control in text-to-video retrieval at every
reported K. E5 has the best R@1, while E6 has the best R@5, R@10, and similarity
gap. This confirms that explicit negatives are doing useful work even though
the absolute gap remains below the frozen target of 0.05.

## Frozen gate audit

| Gate | E3 | E4 | E5 | E6 |
|---|---|---|---|---|
| Non-empty rate = 1.00 | Pass | Pass | Pass | Pass |
| Unique-caption rate >= 0.30 | **Fail** | **Fail** | **Fail** | **Fail** |
| Largest-template rate <= 0.25 | **Fail** | **Fail** | **Fail** | **Fail** |
| Triage BLEU-4 >= E0 (22.21) | Pass | Pass | Pass | Pass |
| Prefix effective rank >= E0 (5.57) | Fail | Fail | Fail | Pass |
| Prefix pair cosine <= 0.80 | **Fail** | **Fail** | **Fail** | **Fail** |
| Evaluation similarity gap >= 0.05 | **Fail** | **Fail** | **Fail** | **Fail** |
| Text-to-video R@1 > own shuffled control | Pass | Pass | Pass | Pass |
| Text-to-video R@1 > E3 | control | Pass | Pass | Pass |
| Overall semantic gate | **Fail** | **Fail** | **Fail** | **Fail** |

## Checkpoint audit

| ID | Fair-comparison checkpoint | Bytes | SHA-256 |
|---|---|---:|---|
| E3 | `checkpoints/stage2_e3_positive_cosine_l010.last.pt` | 852,983,470 | `fa522bcb763bb555b6ce854221e0eebcf752561ede4a7795729fd69906f9fd8c` |
| E4 | `checkpoints/stage2_e4_infonce_l005.last.pt` | 852,980,894 | `262fa77cf3ce23d77a9d0d43478011a251ea65639c5d7f0ba620262b5abb3d23` |
| E5 | `checkpoints/stage2_e5_infonce_l010.last.pt` | 852,980,894 | `46a95b8e13bd59e53d5e76629e9d8a89626284e3086e630295ca3389f6ddbca1` |
| E6 | `checkpoints/stage2_e6_infonce_l020.last.pt` | 852,980,894 | `f1e2159e2139fd49c1c41f2bcda41958a7a389ced11219e9e7878a2502a2565f` |

Both best-validation and last-step checkpoints were retained for each run,
using approximately 6.36 GiB in total. Only the last-step hashes above define
the fair stage-2 comparison.

## Artifact index

- Retrieval: `reports/stage2_retrieval_e{0,3,4,5,6}_*_val100_20260902.json`.
- Semantic gate: `reports/stage2_semantic_e{3,4,5,6}_*_val100_20260902.json`,
  plus `reports/semantic_gate_v5_msvd_full_val100_20260901.json` for E0.
- Visual path: `reports/stage2_visual_e{3,4,5,6}_*_val32_20260902.json`,
  plus `reports/visual_path_v5_val32_20260901.json` for E0.
- Training logs and checkpoint metadata: `runs/stage2_e3_positive_cosine_l010`,
  `runs/stage2_e4_infonce_l005`, `runs/stage2_e5_infonce_l010`, and
  `runs/stage2_e6_infonce_l020`.

Raw reports and checkpoints are intentionally git-ignored; this versioned file
is the durable experiment record.

## Verification

- The project test boundary passed: `python -m pytest tests -q` -> 33 passed.
- Unscoped repository-root collection also discovers two legacy manual scripts
  (`experiments/test_path.py` and `src/test_loader.py`) that reference the
  retired partial-MSVD path. Their collection errors are unrelated to the
  stage-2 code and are not counted as experiment failures.

## Next decision boundary

Do not spend the next run budget on more epochs, a larger negative queue, or
the held-out test set. The predeclared stop rule was triggered because every
lambda fails `unique-caption rate >= 0.30`.

The next stage should first review the representation path: the masked mean of
frozen WTE tokens may be adequate for a coarse retrieval signal but too weakly
connected to decoder-conditioned generation. A stage-2b plan should compare a
decoder-conditioned caption representation against the current WTE-mean
baseline, while retaining exact-caption multi-positive handling and E3 as the
no-negative control. That design change requires approval before execution.
