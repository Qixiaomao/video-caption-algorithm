# Q3 stage-2 CPR implementation and smoke results

Date: 2026-09-12

## Decision

The engineering gate for the prior-inspired Conditional Prefix Ranking (CPR)
comparison arm passed. This result establishes executability only. It is not a
model-quality result and does not support a novelty claim.

The next authorized experiment is the registered MSVD C/D three-seed matrix:

- C: caption CE + CPR;
- D: caption CE + InfoNCE + CPR.

The completed A/B runs remain frozen and must not be rerun.

## Frozen implementation

The implemented objective is:

```text
gap_i = nll(caption_i | wrong_prefix_i) - nll(caption_i | matched_prefix_i)
cpr_i = max(0, margin - gap_i)
total = caption_ce + alignment_weight * alignment_loss + cpr_weight * mean(cpr_i)
```

Registered configuration:

- `cpr_weight=0.10`;
- `cpr_margin=0.10` nats per content token;
- `cpr_negative_policy=cyclic_derangement`;
- reject same-video and exact-caption candidates;
- skip rows without a valid in-batch negative;
- update only the encoder projection and decoder mapper;
- use activation checkpointing for the wrong-prefix GPT-2 pass;
- add no inference parameter or inference operation.

The training objective contract is version 4 and records all three CPR fields.
An exact resume with a changed CPR contract is rejected.

## Regression evidence

The CPR core suite now contains 11 passing tests covering:

- satisfied and violated margins;
- length normalization and EOS/padding exclusion;
- same-video and exact-caption negative exclusion;
- no-valid-negative skipping;
- CPR batch-size validation;
- checkpoint-contract recording;
- projection/mapper gradient reach with frozen GPT-2 WTE;
- CPR branch RNG isolation across activation-checkpoint recomputation;
- unchanged caption-only and InfoNCE forward traces when `cpr_weight=0`.

The current virtual environment was missing the already pinned
`pycocoevalcap==1.2` dependency, so it was restored. METEOR 1.5 then exposed a
Windows-only path bug: its JAR double-encoded spaces in the repository path and
could not locate `data/paraphrase-en.gz`. The wrapper now launches an unchanged
temporary copy of the JAR and data from a no-space path on affected Windows
systems; the metric implementation and scale are unchanged.

After adding the registered C/D runner tests, final full `unittest` discovery:
**100/100 tests passed**, including the locked
COCO caption metric wrapper and all objective, checkpoint, semantic-contract,
prefix-sensitivity, configuration-chain, and benchmark suites.

## Real 4 GB GPU smoke

Hardware and runtime:

- NVIDIA GeForce RTX 3050 4GB Laptop GPU;
- physical memory reported by PyTorch: 4095.5625 MiB;
- PyTorch 2.5.1+cu121;
- batch size 8, 8 frames, 224x224, max caption length 64;
- two optimizer steps, no validation, no test split access;
- common initial checkpoint SHA-256:
  `b0deacbea2239da7dd8f6d815be13f75afc962cce68b0a84771784e0db08b1ff`.

The worst planned arm, caption CE + InfoNCE + CPR, completed without OOM:

| item | before wrong-branch checkpointing | final checkpointed run |
|---|---:|---:|
| retained steps | 2 | 2 |
| peak allocated | 3883.3 MiB | 3669.3 MiB |
| peak reserved | 4018.0 MiB | 3846.0 MiB |
| physical reserved headroom | 77.6 MiB | 249.6 MiB |
| events CSV | finite | finite |

The two event CSV files are byte-identical. Activation checkpointing therefore
reduced peak memory without changing any logged training value in this smoke.

Final joint-arm observations:

| step | total loss | caption loss | InfoNCE | CPR loss | matched NLL | mismatched NLL | CPR gap | eligible |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 7.21471691 | 6.94233513 | 2.12452650 | 0.59928751 | 7.23829556 | 7.10593033 | -0.13236523 | 1.0 |
| 2 | 7.98179913 | 7.73518181 | 2.22124267 | 0.24493170 | 7.83572292 | 7.79380560 | -0.04191732 | 1.0 |

Negative gaps at initialization are not a failure. They show that the frozen
initial model does not yet prefer the matched visual prefix and that the CPR
hinge is active. Only the registered long runs can determine whether this
signal improves visual dependence without damaging caption viability.

## Gate checklist

| condition | result |
|---|---|
| caption-only trace unchanged at CPR weight 0 | pass |
| InfoNCE trace unchanged at CPR weight 0 | pass |
| CPR loss and negative-selection unit tests | pass |
| gradients limited to intended trainable path | pass |
| CPR objective recorded in exact-resume contract | pass |
| batch-8 joint-arm smoke finite on 4 GB GPU | pass |
| relevant existing regression suites | pass |

## Standalone-smoke artifacts

The artifacts below preserve the initial standalone CLI smoke. The registered
C/D smoke is the current execution gate.

| artifact | SHA-256 |
|---|---|
| `runs/q3_infonce_cpr_stage2_smoke_b8_checkpointed_20260912/events.csv` | `2f87ffdd4e0ea4d27955da6695adf4490483419beabbd3306c65bb0b222e6305` |
| `checkpoints/q3_infonce_cpr_stage2_smoke_b8_checkpointed.last.pt` | `0f6ac28ff8acb26978f8be60aa57765652bc32aff3b053a13f0c81110ebaa4d4` |
| `src/training/cpr.py` | `4543295bcb84d51156181424fc2b3b1e0a75cefca9e1e2776b45e9131b7311fb` |
| `src/training/objectives.py` | `d45d30a525bcf65470a9279c38ebb35a04e768060d9773b403efa599927572b7` |
| `src/cli/train_caption_mapper.py` | `66f64da140d68f554d982b312e82daec2fb55feedf4f607669e3e63deaaa0406` |
| `tests/test_cpr_stage2.py` | `1112ab0a647b689115ca55b338e20f9017b39703ac471596bb4f25b7a93a4ede` |
| `scripts/caption_metrics.py` | `2e0f453781cd597f86ea7ea685708d9db0154fd0ebefe7bf76b5f41aa84e74a4` |

The working-tree Git HEAD before these uncommitted stage-2 changes was
`b8666e0c09a0ed530e897d0e75e9c7d0c516050f`.

## Next boundary

The dedicated registered runner and its six-run smoke are now complete. See
`experiments/q3_cpr_cd_registered_smoke_results_20260912.md`. Formal C/D
training remains unexecuted and requires a new output directory without the
`--smoke` flag.
