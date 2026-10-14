# Benchmark Protocol v2

This protocol replaces the June 2026 benchmark outputs. Old reports remain on
disk but are not valid inputs to paper tables until a v2 correctness smoke test
and measured run pass.

## Validity gates

The benchmark now aborts when any of the following is true:

- a caption is empty;
- a sample generates fewer than `--min-new-tokens` tokens;
- a real batch does not contain enough distinct video directories;
- a requested optimization is proven inactive.

`--allow-repeated-input` is reserved for a labelled synthetic microbenchmark.
`--allow-inactive-optimizations` is diagnostic only; a run with recorded
optimization issues must not be used as an ablation result.

## Input layout

`--frames-dir` accepts either:

1. one video directory containing `frame_*.jpg` (valid for BS=1); or
2. a split-level directory containing one frame directory per video (required
   for real BS>1 measurements).

Video directories are sorted and the first `batch_size` distinct items are
recorded in both CSV and JSON outputs.

## Decoding and timing

The v2 benchmark uses deterministic greedy decoding with KV cache. EOS is
masked until `min_new_tokens` have been generated. This fixed protocol is not
the product engine's multi-preset beam/sampling policy, and the distinction is
written to every JSON output.

Per-token CUDA synchronization is disabled for end-to-end timing. Enable
`--profile-token-steps` only for token-level diagnostics because it perturbs
end-to-end latency.

## Explicit optimization switches

Every optimization has paired CLI flags, for example
`--vit-fp16` / `--no-vit-fp16`. JSON records both the requested configuration
and the encoder's effective state (`active`, `validated`, `reason`).

On the frozen RTX 3050 environment:

- `torch.compile` is unavailable because Triton is not installed;
- the installed timm block has no residual/LayerNorm fusion flag.

Strict runs must therefore disable those switches explicitly unless the
environment or backbone changes.

## Reproducibility record

JSON exports include:

- Git commit, branch, dirty state and changed files;
- checkpoint absolute path, size and SHA-256;
- complete inference configuration and benchmark CLI arguments;
- exact video IDs and frame directories per batch size;
- Python, PyTorch, CUDA, timm, transformers, CuPy, Triton and GPU details;
- requested/effective optimization state and any validation issues;
- p50, p95, p99, mean and standard deviation for latency stages.

The experiment matrix and the interpretation of results remain author-owned
research decisions. This document defines only the measurement mechanics.
