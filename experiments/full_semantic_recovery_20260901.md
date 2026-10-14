# Full-MSVD semantic gate and checkpoint recovery (2026-09-01)

## Scope

- Dataset: pinned `friedrichor/MSVD` revision
  `9aa95a3039c106ee9e0969d82d7ce5a9efab1196`.
- Standard splits: 1200 train / 100 val / 670 test.
- Recovery source: `checkpoints/msvd_mapper_recovered_v5.pt`.
- Hardware: RTX 3050 Laptop GPU, 4 GB.
- Caption decode protocol: greedy, empty prompt, 24 new tokens.
- The reported BLEU-4 is the repository's dependency-free, unsmoothed triage
  metric. It is not a paper-ready COCO-caption score.

## Pre-recovery gate

On all 100 standard validation videos, v5 loaded strictly and produced:

| Metric | v5 before full recovery |
|---|---:|
| Validation samples | 100 |
| Non-empty rate | 1.00 |
| Unique caption rate | 0.14 |
| Unsmoothed corpus BLEU-4 | 22.2098 |

The most common hypotheses were `a woman is slicing a carrot` (24/100)
and `a man is playing a guitar` (19/100). The checkpoint therefore passed
basic load/non-empty checks but failed the semantic-diversity gate.

## Recovery A: mapper/projection plus GPT-2 tail

Configuration:

- physical batch size 2, 8 frames, 224x224;
- encoder projection and visual-prefix mapper trainable;
- last four GPT-2 blocks trainable;
- learning rates: `3e-4` visual path, `5e-6` GPT-2 tail;
- 600 optimizer steps (one full train epoch), full val every 100 steps.

Results:

| Metric | Before | Recovery A |
|---|---:|---:|
| Best validation loss | 3.3221 smoke baseline | 3.0464 |
| Non-empty rate | 1.00 | 1.00 |
| Unique caption rate | 0.14 | 0.11 |
| Unsmoothed corpus BLEU-4 | 22.2098 | 32.8538 |
| Peak allocated CUDA memory | - | 2162.6 MB |
| Peak reserved CUDA memory | - | 2420.0 MB |

Loss and BLEU improved, but caption collapse worsened: `a man is playing a
guitar` appeared for 72/100 validation videos. This checkpoint does not pass
the semantic gate.

## Recovery B: mapper/projection only

Recovery B used the same source checkpoint, data, seed, batch size, learning
rate, and 600-step budget, but kept all GPT-2 weights frozen.

| Metric | Before | Recovery B |
|---|---:|---:|
| Best validation loss | 3.3221 smoke baseline | 3.1312 (step 500) |
| Non-empty rate | 1.00 | 1.00 |
| Unique caption rate | 0.14 | 0.14 |
| Unsmoothed corpus BLEU-4 | 22.2098 | 31.1809 |
| Peak allocated CUDA memory | - | 1841.7 MB |
| Peak reserved CUDA memory | - | 2012.0 MB |

Freezing GPT-2 prevented the additional diversity loss seen in Recovery A,
but did not restore cross-video caption diversity. Recovery B also does not
pass the semantic gate.

## Visual-path localization

The same first 32 standard validation videos were measured at four points.
Lower pairwise cosine and higher effective rank indicate more cross-video
variation.

| Checkpoint / stage | Pairwise cosine | Effective rank |
|---|---:|---:|
| v5 / ViT pooled | 0.3286 | 20.06 |
| v5 / encoder projection | 0.5785 | 10.34 |
| v5 / visual prefix | 0.7507 | 5.57 |
| v5 / first-token logits | 0.99967 | 1.02 |
| Recovery A / encoder projection | 0.6744 | 8.52 |
| Recovery A / visual prefix | 0.8940 | 3.63 |
| Recovery A / first-token logits | 0.99987 | 1.02 |
| Recovery B / encoder projection | 0.6861 | 8.52 |
| Recovery B / visual prefix | 0.8971 | 3.62 |
| Recovery B / first-token logits | 0.99988 | 1.03 |

The ViT backbone retains substantial video-to-video variation. Collapse starts
in the 768-to-256 encoder projection, becomes stronger in the visual-prefix
mapper, and is amplified by the language decoder. Full-data caption-only
training further reduces prefix rank even when GPT-2 is frozen.

## Conclusion and next decision

The full dataset, checkpoint load path, backward pass, and 4 GB training path
are operational. The remaining blocker is objective-level rather than
hardware-level: caption cross-entropy alone rewards high-frequency captions
and does not preserve cross-video separation.

The repository's existing alignment model is not InfoNCE. It uses only
positive-pair `CosineEmbeddingLoss`, with no in-batch negatives, so it is not a
valid anti-collapse remedy. The next experiment should be explicitly approved
as a method decision: implement a true contrastive alignment term with
in-batch negatives (and an auditable loss-weight ablation), or retain the
current caption-only model and narrow the paper claim.

## Local artifacts

- `reports/semantic_gate_v5_msvd_full_val100_20260901.json`
- `reports/semantic_gate_recovered_full_v1_val100_20260901.json`
- `reports/semantic_gate_recovered_full_mapperonly_v1_val100_20260901.json`
- `reports/visual_path_v5_val32_20260901.json`
- `reports/visual_path_recovered_full_v1_val32_20260901.json`
- `reports/visual_path_recovered_full_mapperonly_v1_val32_20260901.json`
- `checkpoints/msvd_mapper_recovered_full_v1.pt`
- `checkpoints/msvd_mapper_recovered_full_mapperonly_v1.pt`

Reports, checkpoints, and run logs are intentionally ignored by Git. This
summary and the diagnostic scripts are versioned.
