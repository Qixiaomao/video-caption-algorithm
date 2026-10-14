# Formal efficiency matrix results (2026-09-09)

## Decision

**The preregistered formal matrix completed successfully and is usable as paper evidence.**

- 75/75 cells completed: five configurations, five batch sizes, three independent blocks.
- 60/60 required primary cells passed; all 15 batch-size-16 boundary cells also passed.
- Each cell contains 10 warm-ups and 50 measured iterations: 750 warm-ups and 3,750 retained measurements in total.
- No OOM, inactive requested optimization, non-finite tensor, empty caption, or within-cell output change occurred.
- Captions and generated-token vectors were exactly identical across all configurations and blocks at every batch size.
- The test670 split was not read. This experiment measures efficiency on the frozen 16-video validation input manifest and does not re-estimate caption quality.

## Primary full-stack result: E4 versus E0

Values are pooled medians over 150 retained measurements per cell. Latency IQR and all raw block results remain in `runs/formal_efficiency_matrix_20260909/result.json`.

| Batch | E0 latency (ms) | E4 latency (ms) | latency reduction | E0 throughput | E4 throughput | throughput gain | E0/E4 reserved VRAM |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 447.81 | 350.09 | 21.82% | 2.23 | 2.86 samples/s | 27.91% | 1032 / 966 MiB |
| 2 | 750.77 | 397.22 | 47.09% | 2.66 | 5.04 samples/s | 89.01% | 1198 / 1022 MiB |
| 4 | 1369.69 | 593.65 | 56.66% | 2.92 | 6.74 samples/s | 130.72% | 1522 / 1130 MiB |
| 8 | 2541.15 | 863.46 | 66.02% | 3.15 | 9.27 samples/s | 194.30% | 2160 / 1410 MiB |
| 16 (boundary) | 5072.38 | 1430.88 | 71.79% | 3.15 | 11.18 samples/s | 254.49% | 3440 / 1978 MiB |

At the paper's existing batch-size-8 operating point, E4 also reduced median peak allocated memory from 1363.0 to 1153.4 MiB (209.6 MiB), and median peak reserved memory by 750 MiB.

## Batch-size-8 ablation

| Config | End-to-end median ± IQR (ms) | Throughput median | ViT median (ms) | Allocated / reserved VRAM |
|---|---:|---:|---:|---:|
| E0 FP32, unfused reference | 2541.15 ± 1129.53 | 3.15 samples/s | 1918.51 | 1363.0 / 2160 MiB |
| E1 FP16 | 948.14 ± 239.57 | 8.44 samples/s | 439.66 | 1269.3 / 1530 MiB |
| E2 FP16 + SDPA | 908.03 ± 50.04 | 8.81 samples/s | 337.83 | 1172.1 / 1432 MiB |
| E3 + channels-last | 918.04 ± 102.72 | 8.71 samples/s | 335.35 | 1208.9 / 1490 MiB |
| E4 + tanh GELU + in-place residual | 863.46 ± 383.32 | 9.27 samples/s | 338.11 | 1153.4 / 1410 MiB |

The ablation supports the following narrower interpretation:

1. **FP16 is the dominant speed and memory intervention.** At batch size 8, E0 to E1 reduced end-to-end median latency by 62.69%, while ViT latency fell from 1918.51 to 439.66 ms.
2. **SDPA contributes an additional encoder gain.** E1 to E2 reduced batch-size-8 ViT median latency by 23.2%. Across batch sizes 2--16, the additional ViT reduction was 22.8--23.8% (11.2% at batch size 1).
3. **Channels-last has no defensible standalone benefit here.** E2 to E3 changed ViT latency by less than 1% at batch sizes 2--16, slightly worsened batch-size-8 end-to-end latency, and increased measured memory. It should not be presented as a proven contributor.
4. **The combined E4 additions mainly improve memory, not isolated ViT speed.** Against E3 at batch size 8, E4 reduced allocated memory by 55.5 MiB and reserved memory by 80 MiB, while ViT median latency was effectively unchanged. Because tanh GELU and in-place residual addition enter together, this matrix cannot attribute the memory change to either one alone.

## Block stability

E4 beat the matching E0 block at every batch size in all three blocks (15/15 directional comparisons):

| Batch | block-level E4 latency reduction range |
|---:|---:|
| 1 | 16.8--26.3% |
| 2 | 45.5--48.0% |
| 4 | 51.9--66.9% |
| 8 | 62.8--79.4% |
| 16 | 65.8--81.3% |

The direction is robust, but the exact effect size is noisy. E0 has especially wide IQR at batch sizes 4, 8, and 16, and some cells drifted during measurement. Therefore the paper should report median plus IQR (and preferably the block range), rather than presenting a single decimal percentage as universally precise.

## Semantic-equivalence evidence

Every one of the 15 cells per batch size produced the same caption-list hash and generated-token vector. For example, the batch-size-8 token counts were `[7, 5, 7, 9, 7, 7, 5, 5]` for E0--E4 in all blocks. This supports a fixed-input claim of bit-for-bit decoded-output preservation; it is not a substitute for the separate test670 quality evaluation.

## Paper-use recommendation

The paper can now replace the legacy one-shot efficiency numbers with this formal matrix, provided the baseline is named accurately as an **FP32 unfused reference**. The strongest defensible claim is:

> On the local RTX 3050 4 GB under a fixed repaired-checkpoint protocol, the FP16 + SDPA-centered inference stack preserves decoded captions and reduces median end-to-end latency across batch sizes 1--8; at batch size 8, the complete E4 stack reduces median latency by 66.0%, increases median throughput by 194.3%, and reduces median peak allocated memory by 209.6 MiB relative to the registered FP32 unfused reference.

Do not describe SDPA as a novel kernel, tanh GELU as fusion, or channels-last as a successful optimization. The research contribution should be framed as a reproducible consumer-GPU profiling and validation workflow plus a validated optimization stack, not as a new captioning architecture.

## Artifacts and verification

- Frozen protocol: `experiments/formal_efficiency_matrix_protocol_20260909.md`
- Frozen schedule: `experiments/formal_efficiency_schedule_20260909.json`
- Frozen inputs: `experiments/formal_efficiency_inputs_20260909.json`
- Run metadata SHA-256: `ac9e7ee74e8031277c0bfc7ebb4410dac83c94a6c5f0e29cac5730f3fa554dcc`
- Result SHA-256: `745ac386ec8c5b769df6fca6baa64de2a1a427d7c05aad787c2e6239d20edaa5`
- Formal-run git commit: `6fdfb40b20cdfa87ee47822bf280f2cd4674f47c`
- Maintained test suite after the run: 90 passed.
