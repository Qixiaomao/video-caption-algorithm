# InfoNCE stage-1 GPU smoke results (2026-09-01)

## Decision

Use physical batch size **8** for the first full InfoNCE ablation sweep on the
RTX 3050 4GB. It completed the two-step forward/backward/optimizer/checkpoint
path with peak reserved memory of 2,542 MB, below the predeclared 3,600 MB
safety ceiling. No batch in the planned 2/4/8 range produced an OOM.

These are systems smoke results only. They are not caption-quality or retrieval
results and must not be reported as empirical paper findings.

## Frozen inputs

- GPU: NVIDIA GeForce RTX 3050 4GB Laptop GPU (4,096 MiB)
- PyTorch: 2.5.1+cu121
- CUDA available: yes
- Source checkpoint: `checkpoints/msvd_mapper_recovered_v5.pt`
- Source checkpoint bytes: 845,063,319
- Source checkpoint SHA-256:
  `17dda154899f48c89d12aa02e0014c01ac80af7a8965f41c330fe8d91cb1eec3`
- GPT-2 initialization contract: `checkpoints/gpt2_lm_stage3/best`
- Dataset: pinned full MSVD, Hugging Face revision
  `9aa95a3039c106ee9e0969d82d7ce5a9efab1196`
- Train annotations: 1,200 records, SHA-256
  `ced443e28c5ad2ddd0780af55a820b5ec7da6462c092e1329951dc9c7598c7c5`
- Validation annotations: 100 records, SHA-256
  `bcda3ba0583704a54297e4da4cc9acbcc5ec1ebffd6ed436a6ba3ee1b8a03c8b`

All runs used symmetric InfoNCE, `lambda=0.10`, `tau=0.07`, exact-caption
multi-positive masking, frozen GPT-2, seed 123, and two optimizer steps.

## Memory and numerical results

| Physical batch | Peak allocated MB | Peak reserved MB | Reserved / 4GB | Step 1 total / caption / NCE | Step 2 total / caption / NCE | Step 1 / 2 similarity gap | Result |
|---:|---:|---:|---:|---:|---:|---:|---|
| 2 | 1,843.5 | 1,986.0 | 48.5% | 1.7297 / 1.6585 / 0.7123 | 3.3004 / 3.2327 / 0.6762 | -0.0008 / 0.0071 | Pass |
| 4 | 1,973.0 | 2,242.0 | 54.7% | 2.6821 / 2.5544 / 1.2772 | 3.2455 / 3.1125 / 1.3304 | 0.0130 / 0.0111 | Pass |
| 8 | 2,330.9 | 2,542.0 | 62.1% | 3.0088 / 2.8134 / 1.9537 | 3.6433 / 3.4369 / 2.0643 | 0.0148 / 0.0047 | Pass / selected |

The larger raw NCE value at larger batch size is expected for an untrained
in-batch classification objective because the number of candidate texts grows.
Only later within-run trends and downstream semantic/retrieval gates can be used
to judge whether alignment improves.

## Checkpoint audit

| Run | Checkpoint | SHA-256 | Version / progress |
|---|---|---|---|
| batch 2 | `infonce_smoke_b2.last.pt` | `3a01eefab4f372c46364b167048062984e90d4d1e9b4e2d9df2babe4e09d4dbc` | v3, step 2 |
| batch 4 | `infonce_smoke_b4.last.pt` | `63861078a83bb05524d09a66938df3cb14a774c5f190e6395e51cf3eae6eaee2` | v3, step 2 |
| batch 8 | `infonce_smoke_b8.last.pt` | `ffda2211206640ae9996e7f7542ce180e65bed4a0c7e8a8a4fb37f15a4475079` | v3, step 2 |
| batch 8 resumed | `infonce_smoke_b8_resume.last.pt` | `131ff54efdbdd97d5b9161e5817b0021d57b28de3de98f98a2c2861be5306262` | v3, step 3 |

Every two-step checkpoint contains four populated optimizer-state entries plus
model, Python/Torch/CUDA RNG, DataLoader generator, objective, and dataset
contracts.

## Resume issue found and fixed

The first batch-8 mid-epoch resume attempt failed before training because
loading the checkpoint with `map_location=cuda` also moved the CPU RNG state to
CUDA. PyTorch requires CPU ByteTensors for both CPU and CUDA generator-state
APIs. Recovery now normalizes CPU RNG, CUDA RNG, and DataLoader generator states
to CPU before restoring them. A CUDA regression test was added, and the same
checkpoint then resumed from step 2 to step 3 successfully.

Resumed step 3 produced finite values: total loss 3.5443, caption loss 3.3465,
InfoNCE loss 1.9781, and similarity gap 0.0115.

## Non-blocking warnings

- CuPy reported that `CUDA_PATH` was not detected. The stage-1 path used the
  standard PyTorch mapper (`use_cupy_prefix_projector=False`), so this warning
  did not affect the runs.
- Transformers reported an unrecognized saved `loss_type=None` and selected its
  default causal-LM loss. This is consistent across all smoke configurations;
  the explicit caption supervision contract remains active.

## Stage-2 entry

Lock physical batch size to 8 for E3-E6. Every run must restart from the same v5
checkpoint above, not from a smoke checkpoint. Smoke checkpoints are diagnostic
artifacts only.
