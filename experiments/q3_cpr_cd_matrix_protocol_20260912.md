# Registered MSVD C/D CPR matrix protocol

Frozen on 2026-09-12 before the registered C/D runner smoke and before any
formal C/D training.

## Scientific role

Conditional Prefix Ranking (CPR) is a prior-inspired comparison arm, not a
claimed novel method. This experiment asks whether a decoder-level conditional
ranking constraint adds visual dependence beyond caption CE alone and beyond
embedding-level InfoNCE, while retaining caption viability.

The frozen 2x2 matrix is:

| arm | caption CE | InfoNCE | CPR |
|---|:---:|:---:|:---:|
| A | yes | no | no |
| B | yes | yes | no |
| C | yes | no | yes |
| D | yes | yes | yes |

A/B are completed historical arms and must not be rerun. This runner executes
only C/D.

## Reused historical inputs

Seeds 123, 124, and 125 reuse their exact historical initial checkpoint,
batch schedule, balanced caption-reference order, and caption-only trace.

| seed | initial checkpoint SHA-256 | schedule SHA-256 | historical A trace SHA-256 |
|---:|---|---|---|
| 123 | `b0deacbea2239da7dd8f6d815be13f75afc962cce68b0a84771784e0db08b1ff` | `3520204739b0bac71a6cea9e5a2f26c4dbfee17e3b19790e2340e9efc054fd86` | `8b236e9e3a7f3fbe7ab6dff679326c810f600ae519f275a11271789a4de27a9b` |
| 124 | `a5bac1323a66b2cbd1dd866739794323892f9174ce6ab119b4d01c564b27525f` | `085efe6fe2e3cb0d5ed58ac01cd73804eaa89109572fa3de8d39d2894e93f670` | `805aa6c967c00bd547c59a9db226a5d3cda74ce935ad3e2c29d536d27719210e` |
| 125 | `1aedef75b37ed243f0e43782e15bdca85a27f0e8e3eac6d6eaab27947673d04f` | `94c26d811594eaae11a32c68ec31c7eb8180edcaf7a7ca03ce0d7db64cbf35e7` | `899ffea9f0020352dfacbe104fc336d436169fa7d0c9b09620b9f37fcd55ea0f` |

The frozen train/validation feature caches remain:

- train: `8cbc39e504be44865db302f87856b97b48318b6c32b60db7f8ed1ecb224a7ad6`;
- validation: `5ac1dc53d8e03fcdf3c3c8abae98e1a5c518774ffe48a00eacbb3648503aeee4`.

The runner must fail closed if any registered artifact, train/validation
annotation, source file, component state, or trace field changes.

## Frozen training configuration

- MSVD train1200 and val100; test670 is not read;
- 66 epochs, 9,900 optimizer steps, physical batch size 8;
- all 48,774 training references covered under each seed schedule;
- AdamW, learning rate `3e-4`, weight decay `0.01`;
- no AMP, scheduler, early stopping, or hyperparameter search;
- ViT backbone and GPT-2 frozen;
- only `encoder.proj` and `decoder.mapper` trainable;
- cached frozen ViT features, max caption length 64;
- validation at epochs 5, 20, 40, and 66;
- epoch 66 is the sole formal endpoint.

InfoNCE remains weight `0.10`, temperature `0.07`, exact-caption
multi-positive, mean normalized prefix versus frozen GPT-2 WTE mean.

CPR remains weight `0.10`, margin `0.10` nats/content-token, one deterministic
cyclic in-batch derangement, same-video and exact-caption exclusion, and
row-skipping when no legal negative exists. Its wrong-prefix decoder pass uses
activation checkpointing and an isolated RNG fork. Thus C and D must preserve
the historical main dropout RNG trace.

## Registered smoke gate

Smoke executes two batches for each C/D arm and each seed, using val8 only.
It is engineering-only and always records `paper_ready=false`.

The smoke passes only if:

1. all six runs finish without OOM, NaN, or Inf;
2. every generated C/D trace row equals the corresponding historical A row;
3. within each seed, C and D traces are byte-identical;
4. C and D have identical initial caption CE, CPR loss, NLL gap, eligibility,
   and main RNG consumption;
5. CPR gradients reach projection and mapper, while frozen weights remain
   unchanged;
6. objective version 4 and all CPR fields are present in saved checkpoints;
7. strict checkpoint reload reproduces final component hashes and validation
   retrieval diagnostics;
8. source and data hashes remain unchanged.

Failure blocks formal training. Passing smoke authorizes, but does not itself
start, the formal C/D runs.

## Formal decision rule

Formal results compare C to historical A and D to historical B within each
seed. No test-set value is used for selection.

CPR is retained in the main paper only if all conditions hold at epoch 66:

1. all six C/D runs pass the existing caption viability gate;
2. each C NLL is at most A+0.15 and each D NLL is at most B+0.15;
3. each C token-F1 is at least A-0.05 and each D token-F1 is at least B-0.05;
4. the mean CPR change in `shuffled_minus_native_nll` is positive in both
   strata C-A and D-B;
5. at least two of three seeds have a positive grounding-gap change in each
   stratum.

Retrieval changes, generated-caption changes, the CPR training loss, and the
InfoNCE-by-CPR interaction are secondary descriptive outcomes. There is no
best-seed selection. If the registered rule fails, CPR moves to the appendix
or is removed; it is not rescued by a new margin/weight sweep.

## Execution boundary

The registered smoke command is:

```powershell
.\.venv\Scripts\python.exe scripts\run_cpr_cd_matrix.py `
  --output-root runs\q3_cpr_cd_registered_smoke_20260912 `
  --smoke
```

The formal command is intentionally not executed in this stage. The output
root must not already exist, and the runner never overwrites an experiment.
