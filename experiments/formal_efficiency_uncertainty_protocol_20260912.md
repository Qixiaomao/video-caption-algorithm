# Formal efficiency uncertainty and memory summary protocol

Frozen on 2026-09-12 before running the uncertainty summarizer. This is a
post-measurement, descriptive analysis of the already frozen formal efficiency
matrix; it does not alter, filter, or rerun any timing observation.

## Source

- Formal result:
  `runs/formal_efficiency_matrix_20260909/result.json`
- Result SHA-256:
  `745ac386ec8c5b769df6fca6baa64de2a1a427d7c05aad787c2e6239d20edaa5`
- Raw cells: 75 complete JSON files, each containing 50 retained iterations.
- Design: E0--E4, batch sizes 1/2/4/8/16, three measurement blocks.

Any source hash, cell count, status, block, configuration, batch-size, or
iteration-count mismatch invalidates the summary.

## Interval method

- Resamples: 10,000.
- Base seed: 123. Each contrast/metric receives a deterministic seed derived
  from the base seed and its stable analysis label.
- Interval: percentile 95% interval, bootstrap quantiles 2.5% and 97.5%.
- Point estimate: pooled median of all 150 observations for a configuration and
  batch size.
- Absolute intervals use a hierarchical bootstrap: resample the three blocks
  with replacement, then resample 50 iterations with replacement within each
  selected block, pool the resulting 150 values, and take the median.
- Contrasts use paired block resampling: the same resampled block IDs are used
  for the reference and candidate. Iterations are independently resampled
  within each selected arm because iterations were not temporally paired.

This hierarchy preserves the experiment's block structure. With only three
blocks, intervals are descriptive measures of this local run, not broad
population confidence intervals and not a substitute for replication on other
GPUs. No p-values or multiple-comparison claims are produced.

## Frozen outputs

### Absolute intervals

For every configuration and batch size:

- end-to-end latency median and 95% interval;
- samples/s median and 95% interval;
- ViT latency median and 95% interval;
- peak allocated MiB median and 95% interval;
- peak reserved MiB median and 95% interval.

### Reference contrasts

For E1--E4 versus E0 at the same batch size:

- end-to-end latency reduction percent;
- throughput improvement percent;
- allocated-memory reduction in MiB;
- reserved-memory reduction in MiB.

Positive values favor the candidate. A descriptive interval entirely above
zero is recorded as directionally supported.

### Adjacent ablations

For E1 versus E0, E2 versus E1, E3 versus E2, and E4 versus E3:

- end-to-end latency reduction percent;
- ViT latency reduction percent;
- allocated-memory reduction in MiB;
- reserved-memory reduction in MiB.

These contrasts determine which additions have independent evidence. E4
combines tanh GELU and in-place residual addition, so it cannot identify their
individual effects.

## Memory interpretation

- Physical device memory is taken from the frozen run metadata
  (4095.5625 MiB).
- Report median/interval, maximum observed reserved memory, utilization, and
  physical-memory headroom.
- Batch size 16 remains a boundary result, not a primary paper operating point.
- Allocated and reserved memory are reported separately; they must not be
  interchanged or converted into an unsupported percentage claim.
