# Formal efficiency uncertainty and memory results (2026-09-12)

## Decision

The complete E4 stack retains a directionally supported advantage over the E0
FP32 unfused reference at every measured batch size. The descriptive 95%
hierarchical-bootstrap intervals are above zero for end-to-end latency
reduction, throughput improvement, allocated-memory reduction, and
reserved-memory reduction in all five E4-versus-E0 comparisons.

These are post-measurement descriptive intervals from three local measurement
blocks, not population-level confidence intervals across GPUs.

## E4 versus E0 with 95% intervals

Point estimates are the same pooled medians reported by the frozen formal
matrix. Intervals use 10,000 block-then-iteration bootstrap resamples.

| Batch | Latency reduction, % | Throughput gain, % | Allocated reduction, MiB | Reserved reduction, MiB |
|---:|---:|---:|---:|---:|
| 1 | 21.82 [17.06, 26.12] | 27.91 [20.44, 35.34] | 26.3 [26.3, 26.3] | 66 [66, 66] |
| 2 | 47.09 [45.25, 48.24] | 89.01 [82.63, 93.24] | 52.6 [52.6, 52.6] | 176 [176, 176] |
| 4 | 56.66 [52.21, 66.86] | 130.72 [108.78, 201.90] | 105.2 [105.2, 105.2] | 392 [392, 392] |
| 8 | **66.02 [62.80, 85.44]** | **194.30 [169.17, 587.05]** | **209.6 [209.6, 209.6]** | **750 [750, 750]** |
| 16, boundary | 71.79 [65.83, 81.26] | 254.49 [192.69, 433.97] | 418.9 [418.9, 418.9] | 1462 [1462, 1462] |

The wide upper bounds at batch sizes 8 and 16 reflect the already documented
between-block timing drift. The lower bounds remain large and positive, so the
direction is much more credible than any single exact percentage.

The memory intervals are degenerate because PyTorch reported the same
allocator peak for all 50 iterations and all three blocks of each cell. This
supports deterministic local allocator accounting; it does not imply zero
uncertainty on another machine or software stack.

## Memory feasibility on the 4 GB device

| Batch | E0 allocated / reserved | E4 allocated / reserved | E0 reserved use | E4 reserved use | E4 physical headroom |
|---:|---:|---:|---:|---:|---:|
| 1 | 905.0 / 1032 MiB | 878.7 / 966 MiB | 25.2% | 23.6% | 3129.6 MiB |
| 2 | 970.5 / 1198 MiB | 917.9 / 1022 MiB | 29.3% | 25.0% | 3073.6 MiB |
| 4 | 1101.5 / 1522 MiB | 996.3 / 1130 MiB | 37.2% | 27.6% | 2965.6 MiB |
| 8 | 1363.0 / 2160 MiB | 1153.4 / 1410 MiB | 52.7% | 34.4% | 2685.6 MiB |
| 16, boundary | 1887.4 / 3440 MiB | 1468.5 / 1978 MiB | 84.0% | 48.3% | 2117.6 MiB |

- Physical CUDA memory in the frozen runtime: 4095.5625 MiB.
- Maximum observed reserved memory anywhere in the 75-cell matrix: 3440 MiB
  (E0, batch size 16).
- Minimum observed physical headroom: 655.5625 MiB.
- Every batch-size-16 cell completed, and every reserved-memory upper interval
  remained below physical capacity.
- E4 batch size 16 used less reserved memory than E0 batch size 8 (1978 versus
  2160 MiB), although batch size 16 remains a boundary rather than the primary
  paper operating point.

## What the adjacent ablations support

The table counts batch sizes whose descriptive 95% interval is entirely above
zero. Positive means the newly added intervention helps.

| Addition | End-to-end latency | ViT latency | Allocated memory | Reserved memory | Verdict |
|---|---:|---:|---:|---:|---|
| E0 -> E1: FP16 | 5/5 | 5/5 | 5/5 | 5/5 | dominant supported intervention |
| E1 -> E2: SDPA | 1/5 | 5/5 | 5/5 | 5/5 | supported encoder/memory gain; total latency is masked at small batches |
| E2 -> E3: channels-last | 1/5 | 0/5 | 0/5 | 0/5 | no defensible standalone contribution |
| E3 -> E4: tanh GELU + in-place residual | 0/5 | 0/5 | 5/5 | 5/5 | supported memory gain, not isolated speed gain |

At batch size 8 specifically:

- FP16 versus E0: end-to-end -62.69% [59.01%, 78.78%], ViT -77.08%
  [71.39%, 85.51%].
- SDPA versus FP16: ViT -23.16% [22.61%, 37.82%], but end-to-end -4.23%
  [-24.12%, 8.73%]; the total-latency interval crosses zero.
- Channels-last versus E2: end-to-end point estimate worsens by 1.10%, ViT
  interval crosses zero, and allocated/reserved memory increase by 36.8/58 MiB.
- E4 additions versus E3: end-to-end -5.95% [-1.78%, 47.73%] and ViT point
  estimate worsens by 0.82%; neither speed interval is supported. Allocated and
  reserved memory fall by 55.5 and 80 MiB respectively, with supported local
  intervals.

## Paper wording

The paper may use the E4-versus-E0 interval and memory result, while naming E0
accurately as an FP32 unfused reference. Component claims should be narrower:

- FP16 is the primary end-to-end intervention.
- SDPA has independent ViT-stage and memory evidence.
- Channels-last should be reported as neutral/negative under this setup, not as
  a successful optimization.
- The combined E4 additions support a memory claim, but do not identify which
  addition caused it and do not establish an extra speed gain.

Recommended batch-size-8 statement:

> Across three local measurement blocks, the complete E4 stack reduced median
> end-to-end latency by 66.0% (descriptive hierarchical-bootstrap 95% interval:
> 62.8--85.4%) and peak allocated/reserved memory by 209.6/750 MiB relative to
> the registered FP32 unfused reference, while preserving the frozen decoded
> captions exactly.

## Artifacts

- Protocol: `experiments/formal_efficiency_uncertainty_protocol_20260912.md`
- Raw summary: `runs/formal_efficiency_uncertainty_20260912/result.json`
- Absolute intervals: `runs/formal_efficiency_uncertainty_20260912/absolute_intervals.csv`
- E0 contrasts: `runs/formal_efficiency_uncertainty_20260912/reference_contrasts.csv`
- Adjacent ablations: `runs/formal_efficiency_uncertainty_20260912/adjacent_ablation_contrasts.csv`
- Result SHA-256:
  `f9436c0d27a06125bf4562f1764dd67ff01ffe8b3ac3e18cfc46c82e7ed2711d`
- Analysis source commit:
  `2e22f4f6aa1d61c2534181fe815894b2789d34f3`
