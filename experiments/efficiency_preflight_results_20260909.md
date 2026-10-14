# Efficiency matrix step-2 preflight results (2026-09-09)

## Decision

**PASS — E0--E4 are ready for user review before the formal matrix.** The formal repeated-timing matrix (step 3) has not started.

This conclusion only means that the registered model, fixed inputs, semantic-equivalence gates, optimization-state checks, and 4 GB memory feasibility passed. The one-shot elapsed values below are smoke diagnostics and are not paper results.

## Frozen artifacts

- Protocol: `experiments/formal_efficiency_matrix_protocol_20260909.md`
  - SHA-256: `906118c28d8f11b5873093981db807bc935e38470ab1021ac4a8e5e702ef1200`
- Input manifest: `experiments/formal_efficiency_inputs_20260909.json`
  - SHA-256: `ca870d4a3d09c9988212fce2cbb25c846f8083dd1b2bedd667c0f633ae90a946`
  - 16 deterministic MSVD val videos, 128 frame hashes, nested batches; test670 was not read.
- Checkpoint: `runs/full_msvd_infonce_ablation_20260906/checkpoint_epoch066.pt`
  - SHA-256: `891cdcceef2b52d09092dde00ef35f9e1ee652e686b30178b5c1a9aee81c7e7c`
  - Strict load passed; epoch 66 / step 9900 / objective version 3 / InfoNCE contract passed.
- Frozen protocol/input commit: `af559fb29b2b190758e3532fc5d6c84f2128c410`
- Preflight runner commit: `100719a0d6b484f963a3c8bbbb454eba23111c47`
- Result: `runs/efficiency_preflight_20260909/result.json`
  - SHA-256: `f41f778c52d4e75babec01d24b3d4e070b4bc282448924db457c7fc3c2f6a0c9`

## Gate results

| Config | Requested state exact | Caption match vs E0 | Encoder min cosine | Prefix min cosine | Peak reserved, bs=8 | bs=16 | Gate |
|---|---:|---:|---:|---:|---:|---|---|
| E0 FP32 reference | yes | reference | 1.000000 | 1.000000 | 1798 MiB | pass, 2696 MiB | PASS |
| E1 FP16 | yes | 8/8 | 0.999998 | 1.000000 | 1410 MiB | pass, 1882 MiB | PASS |
| E2 FP16 + SDPA | yes | 8/8 | 0.999999 | 1.000000 | 1366 MiB | pass, 1798 MiB | PASS |
| E3 + channels-last | yes | 8/8 | 0.999999 | 1.000000 | 1404 MiB | pass, 1872 MiB | PASS |
| E4 + tanh GELU + in-place residual | yes | 8/8 | 0.999969 | 0.999997 | 1330 MiB | pass, 1724 MiB | PASS |
| E5 + compile (exploratory) | unavailable | not run | — | — | — | not run | EXCLUDED |

Additional required gates also passed for E0--E4:

- all batch-8 captions were non-empty;
- the first caption was invariant between batch sizes 1 and 8;
- all encoder and prefix values were finite;
- batch-8 peak reserved memory remained below the preregistered 3,800 MiB limit;
- all requested flags were active and all excluded flags were inactive.

E5 was excluded exactly as preregistered because the Windows environment has no Triton. The runner detected this before inference and did not accept an eager-mode fallback as a compile result.

## Smoke-only observations

The single-pass batch-8 elapsed values were E0 2.655 s, E1 0.536 s, E2 0.451 s, E3 0.436 s, and E4 0.423 s. They are encouraging enough to justify step 3, but they are strongly affected by first-run effects and contain no repetitions or uncertainty estimate. They must not be quoted as speedups.

## Invalid first attempt retained for audit

The first runner attempt was moved intact to `runs/efficiency_preflight_20260909_invalid_gc_leak`. It showed monotonically accumulating CUDA allocations across E0--E4 because the prior `ViTFrameEncoder` instance contains a bound-method reference cycle and was not collected between configurations. That made the E3/E4 memory failures invalid.

The fix added explicit cyclic garbage collection before `torch.cuda.empty_cache()` between cells. It did not change the frozen checkpoint, inputs, configurations, thresholds, or decoder. The clean rerun starts each cell near the same model-memory baseline and passes the preregistered gates.

## Recommendation and boundary

Proceed to step 3 only after user approval. The formal primary matrix should contain E0--E4 at batch sizes 1, 2, 4, and 8, with batch 16 retained as the registered boundary cell. Keep E5 out of the primary table on this machine. No paper claim should be changed until the repeated formal matrix finishes.
