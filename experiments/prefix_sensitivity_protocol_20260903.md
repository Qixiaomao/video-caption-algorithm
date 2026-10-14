# Prefix sensitivity: implementation contract (2026-09-03)

> Execution follow-up: the subsequently authorized full val32 matrix is now
> complete. See `prefix_sensitivity_val32_20260903.md` for results. The scope
> and smoke-verification sections below retain the original pre-run contract.

## Scope and current step

Implement and test `scripts/diagnose_prefix_sensitivity.py`, then perform a
bounded E0/E6 smoke check. Do not train, modify model parameters, run test670,
or select a new architecture in this step. Full val32 and val100 intervention
results are separate subsequent experiments.

## Audited inputs

- E0: `checkpoints/msvd_mapper_recovered_v5.pt`, SHA-256
  `17dda154899f48c89d12aa02e0014c01ac80af7a8965f41c330fe8d91cb1eec3`.
- E6: `checkpoints/stage2_e6_infonce_l020.last.pt`, SHA-256
  `f1e2159e2139fd49c1c41f2bcda41958a7a389ced11219e9e7878a2502a2565f`.
- Only `data/processed/msvd_full/val/annotations.json` is accepted, SHA-256
  `bcda3ba0583704a54297e4da4cc9acbcc5ec1ebffd6ed436a6ba3ee1b8a03c8b`.
- Preflight found identical 149 GPT-2 weight tensors between E0 and E6, full
  state-dict compatibility, and 800 readable frames for 100 validation videos.
- Prefix shape: `[B,4,768]`; video input: 8 frames at 224x224. All runs use
  evaluation mode, inference mode for forward passes, and frozen parameters.

## Important interpretation correction before interventions

The previous visual-path reports already show large real-versus-zero prefix
KL divergence (E0 8.5893 nats; E6 9.9553 nats), with first-token argmax changing
from `a` to `and`. Therefore, the prior claim that GPT-2 completely ignores its
prefix is not supported. We must distinguish generic caption-mode conditioning
from use of video-specific semantic information.

The planned 20% first-token flip and 30% caption-change markers are retained as
descriptive heuristic flags, not acceptance gates. A shared initial article
does not establish semantic collapse. Raw-logit cosine/rank is sensitive to
per-row offsets irrelevant to softmax. JSD, vocabulary-centered logits,
full-caption changes, and matched-versus-mismatched caption NLL are needed.

## Frozen interventions

| Condition | Input |
|---|---|
| native | Correct video prefix |
| repeat | Independent rerun of exactly the same prefix |
| shuffled | Seeded one-to-one donor cycle with no self-pairs |
| zero | Zero vectors, retaining four positions and attention mask |
| mean | Mean prefix of the entire fixed selected validation subset |
| scale_0 / scale_025 / scale_05 / scale_1 / scale_2 | Multiply by 0 / 0.25 / 0.5 / 1 / 2 |

`full` includes all ten conditions; `core` includes native, repeat, shuffled,
and zero. Scale zero and scale one intentionally duplicate two controls.
Donor assignment and the subset mean do not depend on reference microbatch
size. Zero/scale interventions can be out of distribution; shuffled is the
main semantic control. Zero vectors are not a no-prefix LM baseline because
position embeddings and sequence positions are retained.

## Measurement contract

- Select the first N records in pinned annotation order, never silently skip
  missing frames or invalid captions. Full val32 uses the same IDs as stage 2.
- Seed 123. Process one video prefix at a time; default reference microbatch 4.
- Generate directly from prefix with empty prompt, no synthetic BOS, greedy
  decoding, max 24 new tokens, min 0, no-repeat-ngram size 3, repetition
  penalty 1.0. The native first video must match the existing decoder wrapper.
- Record first-token JSD in nats, Top-1 change rate, max absolute logit change,
  and paired vocabulary-centered logit distances.
- Capture embedding-plus-position, all 12 block outputs before final layer
  normalization, and final layer normalization separately (14 boundaries).
  Record paired relative L2/cosine at all prefix positions and at the last
  prefix position. These are prefix-only probes, not later generated states;
  normalization changes alone do not isolate a causal failure layer.
- For NLL, tokenize every annotated reference (except explicitly bounded
  smoke runs), append EOS, cap total text length including EOS at 40, and
  record all truncations. Preserve the first EOS and mask subsequent padding.
  The last prefix position predicts the first caption token.
- Compute token-mean NLL per reference, then reference-mean per video, then
  video-mean for the reported delta: NLL(intervention) minus NLL(native).
  Also report delta after excluding the first caption token; EOS remains.
- Paired percentile bootstrap: 10,000 resamples of video-level deltas, seed
  123. Captions are not treated as independent experimental units. The CI is
  descriptive for the fixed donor mapping and does not include rematching
  uncertainty. Smoke reports disable inferential interpretation.
- Save per-video captions, per-reference/token NLL, reference tokens, donor
  IDs, checkpoint/data/source hashes, source git state, and environment info.
- Assert repeat/scale-one consistency within 1e-5 and exact caption equality.
  Shuffled logits/captions must equal their native donor outputs. This is an
  instrumentation check, not proof of grounding.
- Existing reports are never overwritten. Checkpoint hash is checked both
  before and after the run. All outputs remain `paper_ready=false`.

## Bounded smoke commands

Run from the repository root, sequentially on the RTX 3050:

```powershell
.\.venv\Scripts\python.exe scripts/diagnose_prefix_sensitivity.py `
  --checkpoint checkpoints/msvd_mapper_recovered_v5.pt `
  --expected-checkpoint-sha256 17dda154899f48c89d12aa02e0014c01ac80af7a8965f41c330fe8d91cb1eec3 `
  --smoke-test --limit 2 --references-per-video 2 --preset full `
  --output reports/prefix_sensitivity_e0_smoke2_20260903_verified.json

.\.venv\Scripts\python.exe scripts/diagnose_prefix_sensitivity.py `
  --checkpoint checkpoints/stage2_e6_infonce_l020.last.pt `
  --expected-checkpoint-sha256 f1e2159e2139fd49c1c41f2bcda41958a7a389ced11219e9e7878a2502a2565f `
  --smoke-test --limit 2 --references-per-video 2 --preset full `
  --output reports/prefix_sensitivity_e6_smoke2_20260903_verified.json
```

Only after smoke verification, the next step can remove the two smoke flags,
use `--limit 32 --preset full`, and write new val32 report paths. Full runs
reject partial-reference settings. Do not interpret two-video smoke metrics
as MSVD experimental findings.

## Implementation and smoke verification completed

- Regression command: `.\.venv\Scripts\python.exe -m unittest discover -s tests -q`.
  All 48 tests passed (33 existing plus 15 new). Tests cover shifted NLL/HF
  loss equivalence, padding/EOS, microbatch neutrality, JSD, donor cycles,
  layer-hook cleanup, bootstrap reproducibility, validation-only boundaries,
  bounded smoke settings, and refusal to overwrite an existing report.
- Both final smoke runs used 2 videos, the first 2 annotated references per
  video, all 10 conditions, and all 14 layer boundaries. No reference was
  truncated in these bounded runs.
- Native generation matched the existing wrapper on the first video for
  both models. All shuffled outputs tracked the correct native donor.
- Repeat-control max absolute logit difference, caption-change rate, and
  paired NLL delta were all exactly zero for both models. Scale-one identity
  and hidden-state reproducibility checks passed as well.
- Each run peaked at 886.63 MiB allocated / 1,002 MiB reserved CUDA memory.
  This is an inference smoke measurement, not the full-run memory benchmark.
- Checkpoint, annotations, and implementation-source hashes remained
  unchanged during both final smoke runs.
- An initial E0 smoke report without the `_verified` suffix is retained as a
  development artifact. It predates the added prefix-distribution and
  in-flight source-hash audit fields; use only the `_verified` reports below.
- No production model/training code was modified; no training or test670
  evaluation occurred. Full val32/val100 experiments are not yet run.

| Artifact | SHA-256 |
|---|---|
| `scripts/diagnose_prefix_sensitivity.py` | `abd4b415a9f2d19c72991f49a6d8a9e0a01710b582438a0b660cbfce6eab402d` |
| `reports/prefix_sensitivity_e0_smoke2_20260903_verified.json` | `0cb0c08ae099fb302564f46d387a901c75a3a906ee2ded1f8c8fe3f7e7ebe351` |
| `reports/prefix_sensitivity_e6_smoke2_20260903_verified.json` | `9883cba8589051a3b6d2a014f280f01486b51c8553cdc951bf69a0efcd6e994f` |

Reports include hashes for the newly implemented source even though the
smoke runs occurred before its first commit. Raw JSONs remain git-ignored;
the protocol and smoke audit are versioned here.

Non-blocking messages: CuPy's CUDA_PATH warning did not affect the standard
PyTorch prefix path; Transformers ignores top_p with greedy decoding, as in
the existing wrapper. No dependencies were installed or updated.
