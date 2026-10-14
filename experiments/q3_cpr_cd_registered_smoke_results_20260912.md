# Registered MSVD C/D CPR matrix smoke results

Date: 2026-09-12

## Decision

**The registered C/D smoke gate passed.** The runner is authorized for the
formal three-seed C/D experiment, but no formal run was started in this stage.

This result is engineering evidence only. It does not show that CPR improves
captioning, retrieval, or visual dependence, and it does not create a CPR
novelty claim.

## Executed scope

- seeds: 123, 124, and 125;
- arms: C = caption CE + CPR; D = caption CE + InfoNCE + CPR;
- two optimizer steps per arm, six runs and twelve steps total;
- cached frozen ViT features;
- validation limited to the first eight registered validation videos;
- `test_split_read=false`, `paper_ready=false`, `formal_decision=null`.

The runner reused each seed's exact historical initial checkpoint, 66-epoch
schedule, balanced caption-reference order, and caption-only trace. A/B were
not rerun.

## Paired-control audit

All registered controls passed for every seed:

| check | seed 123 | seed 124 | seed 125 |
|---|:---:|:---:|:---:|
| direct vs cached ViT max error = 0 | pass | pass | pass |
| C/D initial caption CE equal | pass | pass | pass |
| C/D initial CPR values equal | pass | pass | pass |
| C/D main RNG consumption equal | pass | pass | pass |
| historical A trace prefix match | pass | pass | pass |
| C/D trace byte-identical | pass | pass | pass |
| CPR gradient reaches projection/mapper | pass | pass | pass |
| frozen GPT-2 receives no gradient | pass | pass | pass |
| valid negative coverage | 100% | 100% | 100% |

The isolated CPR RNG fork is necessary: it prevents the extra wrong-prefix
GPT-2 dropout pass and its checkpoint recomputation from advancing the main
historical training RNG. Thus the input, caption, and dropout trace remain
comparable to A/B while CPR still participates in gradient computation.

Initial CPR diagnostics show that the hinge is active in every seed:

| seed | CPR loss | mismatched-minus-matched NLL gap |
|---:|---:|---:|
| 123 | 0.489778 | -0.193655 |
| 124 | 0.621244 | -0.265831 |
| 125 | 0.272414 | -0.129598 |

The negative initial gaps are not a quality conclusion. They only show that
the randomly initialized trainable visual path does not yet rank the matched
prefix above the deterministic mismatch.

## GPU and checkpoint audit

All runs completed on the NVIDIA GeForce RTX 3050 4GB Laptop GPU.

| seed | arm | steps | peak allocated MiB | peak reserved MiB | objective version | alignment | CPR weight |
|---:|:---:|---:|---:|---:|---:|---|---:|
| 123 | C | 2 | 2377.9 | 2508.0 | 4 | none | 0.10 |
| 123 | D | 2 | 2394.2 | 2420.0 | 4 | InfoNCE | 0.10 |
| 124 | C | 2 | 2378.1 | 2516.0 | 4 | none | 0.10 |
| 124 | D | 2 | 2399.7 | 2420.0 | 4 | InfoNCE | 0.10 |
| 125 | C | 2 | 2378.1 | 2516.0 | 4 | none | 0.10 |
| 125 | D | 2 | 2399.7 | 2420.0 | 4 | InfoNCE | 0.10 |

The worst reserved value was 2516.0 MiB, leaving 1579.6 MiB relative to the
4095.5625 MiB physical device. Every checkpoint records objective version 4,
the correct InfoNCE setting for its arm, CPR weight/margin/policy, cached
backbone status, isolated CPR RNG, and no inference-time parameter increase.
Every final checkpoint strict-reloaded to the recorded component hashes and
reproduced the retrieval diagnostic.

## Two-step validation values

| seed | arm | native NLL | shuffled-minus-native NLL | T2V R@1 | caption gate |
|---:|:---:|---:|---:|---:|:---:|
| 123 | C | 7.068826 | -0.014053 | 8.7079% | fail |
| 123 | D | 7.077315 | -0.015293 | 8.1461% | fail |
| 124 | C | 7.108423 | 0.060508 | 6.7416% | fail |
| 124 | D | 7.110357 | 0.060307 | 7.0225% | fail |
| 125 | C | 7.235039 | -0.038990 | 14.6067% | fail |
| 125 | D | 7.242354 | -0.038660 | 14.8876% | fail |

The caption gate failures after two steps are expected under the registered
smoke contract. Historical A/B also required long training before the semantic
gate became meaningful. These values cannot be compared with the epoch-66
formal endpoints.

## Regression and integrity status

- repository-wide `unittest` discovery: **100/100 passed**;
- all six runs: finite losses and gradients, no OOM;
- all source/data checks: unchanged throughout execution;
- C/D trainable projection and mapper changed;
- frozen ViT backbone and GPT-2 hashes did not change;
- output root was newly created and no experiment was overwritten.

## Key artifacts

| artifact | SHA-256 |
|---|---|
| `runs/q3_cpr_cd_registered_smoke_20260912/completion.json` | `149e5a5504752124a68d907b1b530be4753e536f2bb06609f9ea0ab7db728036` |
| `seed123/preflight.json` | `5534a31b5f1220ac699f5f1fff0602f07ab9fe4e8e2876e49d22c320f502804d` |
| `seed124/preflight.json` | `5fbb9887101639e9ebd50239d7302cd9d51598ab9872a40ec96e2db5083894ac` |
| `seed125/preflight.json` | `66c229db522e8bcd5584b32051d4bab3948df65e20531e7866834b739d3c0200` |
| `scripts/run_cpr_cd_matrix.py` | `af52b5e2f1adf32f94244bd59f85a210703b93acdf6d16205b07a9da2c2b1f5a` |
| `experiments/q3_cpr_cd_matrix_protocol_20260912.md` | `805b74fb0f2dd135596a4f0d902f21092d0cdfe4f34d78029507d3cb822fc12b` |

Per-seed C/D trace hashes are respectively:

- seed 123: `b3c974fc9698df1cb06fa0f05181c24691af849f447fa80936f16a18724aaf8f`;
- seed 124: `9cf257be8ff1c771bf97b097e7b4e0efb9617dcb2cd208bcd304b98d3d78f342`;
- seed 125: `d3c8c835edbf29dfd3bb4ce3bc6ec480e568ac11db40f71180773d3232a6dfb5`.

## Next boundary

The formal command remains unexecuted. If authorized, it must use this same
runner with a new output directory and without `--smoke`. It will execute C/D
for all three seeds to 66 epochs and 9,900 steps, evaluate val100 at the four
frozen milestones, and apply the preregistered CPR-retention rule. Test670
remains sealed until that decision is frozen.
