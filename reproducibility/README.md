# Reproduction guide and publication audit

Publication snapshot: 2026-10-14. Numeric evidence was frozen/audited on
2026-10-03 and its 67 source/dependency hashes were reverified locally for this
snapshot. Original experiment inputs and runs were not modified.

## What can be reproduced from a clone

The public evidence check is self-contained and uses Python's standard library:

```bash
python scripts/verify_public_evidence.py
python scripts/audit_public_release.py
python -m unittest discover -s tests -p test_public_evidence.py -v
```

`evidence/paper_master_results.csv` preserves the original 16-column structure
and 1,036 numeric rows. Personal absolute paths were replaced with symbolic
`${CODE_ROOT}`, `${PAPER_ROOT}` and `${HOME}` prefixes. These symbols describe
provenance, not filesystem paths to pass directly to a runner.

`source_values.json` contains 778 numeric field extracts, with their original
source field and source-file hash. `row_lineage.json` describes every row's
operands, scaling and arithmetic. `release_manifest.json` hashes the actual
public copies. `source_manifest.json` retains original hashes, including assets
that are not redistributed. The original master CSV SHA-256 is:

`bf7e56a3b5c52221358fa2abf14cbcad42eb96568f9aaea7be0150fcc646d642`

The public CSV has a different byte hash because its paths/encoding were adapted.
Numeric extraction and arithmetic verification cannot establish that an omitted
source file really has the recorded hash. The local publication audit performed
that check against all 67 original files; readers need those originals to repeat
that authentication. Neither check is a training or inference rerun.

## Experimental contracts

### MSVD A/B and test670

Read these protocols in order:

1. `experiments/full_msvd_balanced_protocol_20260905.md`
2. `experiments/full_msvd_infonce_ablation_protocol_20260906.md`
3. `experiments/repeated_seeds_protocol_20260908.md`
4. `experiments/test670_protocol_20260908.md`

Preparation: `scripts/prepare_msvd_full.py --help` describes the archive and
metadata inputs. Eight 224x224 frames per video and the 1,200/100/670 split are
used. The formal epoch66 endpoint has 9,900 steps. Seeds 124 and 125 add matched
A/B runs to seed123. A uses caption CE; B adds symmetric multi-positive InfoNCE
with weight 0.1 and temperature 0.07. Decoder/backbone stay frozen.

The registered entrypoints, from repository root, are:

```bash
python scripts/run_full_msvd_balanced.py --output-root runs/full_msvd_balanced_20260905
python scripts/run_full_msvd_infonce_ablation.py --output-root runs/full_msvd_infonce_ablation_20260906
python scripts/run_repeated_seeds.py --output-root runs/repeated_seeds_20260908
python scripts/run_test670_evaluation.py --output-root runs/test670_reproduction
```

These are strict archival replay runners, not a turnkey fresh-clone pipeline.
They require pinned annotations and historical initial-state/cache/checkpoint
hashes. Annotation JSON embeds frame paths, so relocating data changes its hash;
newly generated PyTorch files can also differ bytewise. Do not disable the checks
or change constants merely to obtain a PASS. A fresh-run pipeline for MSVD that
registers all new artifact hashes without depending on historical artifacts
remains unfinished. Dataset files and original model artifacts are not bundled.

### MSR-VTT A/B and test2990

Use `msrvtt_data_protocol_20260928.md`, `msrvtt_ab_protocol_20260928.md`,
`msrvtt_test_eval_config_20260928.json` and `msrvtt_resume_protocol_20260929.md`
under `experiments/`. The 20260927 protocols are superseded historical records.
The split is 6,513 train / 497 val / 2,990 test. All six A/B arms use 30 epochs,
24,420 steps, batch size 8 with the last 9 records kept together, and a fixed
endpoint selected before test access. Validation and test use different text
length limits and candidate sets; their absolute recall values are not directly
comparable. The test is not the standard 1K retrieval benchmark.

Preparation code is published in execution order:

1. `download_msrvtt_archive_20260927.py`: pinned archive location and size.
2. `prepare_msrvtt_full_20260928.py`: verify archive and extract/audit media.
3. `verify_msrvtt_archive_20260928.py` and `finalize_msrvtt_data_20260928.py`:
   split/media completion checks; inspect their existing-artifact assumptions.
4. `audit_msrvtt_captions_20260928.py`: tokenizer audit. Set `VC_GPT2_SOURCE`
   to the local pinned GPT-2 directory.
5. `cache_msrvtt_train_val_20260928.py`: build and verify frozen ViT features.
6. `run_msrvtt_engineering_smoke_20260928.py`,
   `run_msrvtt_train_val_pilot_20260928.py`,
   `benchmark_msrvtt_validation_20260928.py`: engineering/validation gates.

Preparation currently reads the local MSVD `preflight.json` for the pinned GPT-2
model source. This dependency must be supplied; publishing the scripts does not
remove it. Inspect each script before execution: some have fixed fresh output
names and no CLI. Do not reuse an existing output directory.

After all preparation gates pass without test access, materialize a **new**
ignored local configuration instead of editing the historical public template:

```bash
python scripts/materialize_msrvtt_config.py --gpt2-source models/gpt2
python scripts/run_msrvtt_ab_formal_20260928.py --config experiments/msrvtt_reproduction.local.json --output runs/msrvtt_reproduction_train
python scripts/run_msrvtt_test2990_20260929.py --config experiments/msrvtt_reproduction.local.json --training-root runs/msrvtt_reproduction_train --output runs/msrvtt_reproduction_test2990
```

The materializer checks preparation PASS/test-not-read states, retains all
registered hyperparameters and the original GPT-2 weight hash, records new local
preparation/config hashes, and refuses to overwrite a configuration. It does
not claim byte-identical reproduction and was not followed by a GPU run during
this publication update. The frozen public template contains symbolic paths
and is not directly executable.

`archived_drivers/staged_msrvtt_train_20260929.py` and
`staged_msrvtt_test2990.py` record the actual staged execution that produced the
paper results. Only the code-root location is adapted. Their fixed original
config hash, linked run layout, interrupted-run replay evidence and default
relative output layout remain historical dependencies. Do not use them as the
fresh-run recipe above or point them at original completed assets casually.

### E0-E4 efficiency and CPR

Efficiency protocols: `formal_efficiency_matrix_protocol_20260909.md` and
`formal_efficiency_uncertainty_protocol_20260912.md`. Code:
`run_efficiency_preflight.py`, `run_formal_efficiency_matrix.py`, and
`summarize_efficiency_uncertainty.py` under `scripts/`.

The matrix contains 75 cells and 3,750 retained timings (E0-E4, five batch sizes,
three blocks). Keep the frozen input order, warmup, CUDA synchronization and
memory accounting. Report E1 FP16 and E2 FP16+SDPA alongside E0 and E4. The
hierarchical bootstrap interval is descriptive for this machine. Its runners
require the original B checkpoint, frame hashes and preparation results.
`formal_efficiency_inputs_20260909.json` is a sanitized input manifest, not frame
data or a directly relocated input configuration.

CPR code: `src/training/cpr.py`, `scripts/run_cpr_cd_matrix.py`,
`resume_cpr_cd_matrix.py`, `finish_cpr_cd_matrix.py`, `finalize_cpr_cd_matrix.py`.
Read the registered 20260912 matrix protocol and 20260925 recovery notes.
The retained outcome is negative: each grounding stratum had only one positive
seed out of three. No test-based selection is permitted.

### GIT-base-VATEX external reference

Read `experiments/external_baseline_protocol_20260928.md`. The model is
`microsoft/git-base-vatex`, pinned at revision
`623363cef8744c57438ce8d62b30bef1fbbec111`. Obtain its files via
`scripts/prepare_git_vatex_baseline_20260928.py` under upstream terms.
Use `scripts/run_git_vatex_baseline_20261003_compat.py --mode smoke` followed by
`--mode formal_test670` only after smoke passes and original test inputs match.

The compatibility adapter strictly checks the six old `img_temperal_embedding`
keys against the six expected `img_temporal_embedding` keys and copies those
pinned tensors. The original 20260929 runner is retained to explain the failed
smoke; it is not the recommended baseline entrypoint. Failure evidence must not
be interpreted as a valid baseline result.

The 670-video reference uses six selected frames and different training/tokenizer
conditions from this project's eight-frame MSVD model. It is a caption reference,
not a controlled InfoNCE intervention or same-protocol speed comparison.

## Publication boundaries and outstanding work

Published: source code, tests, dated protocols, sanitized config/input manifests,
public result table, numeric source extracts, provenance and verification tools.

Kept local: original videos, annotations/reference/generated captions, weights,
features/index caches, large archives, full run logs, personal research/advisor
notes, drafts, account/environment files. A previously tracked sample video and
unrelated screenshots/notes are removed from this branch's current tree; this
does not erase the existing public Git history.

Remaining limitations:

- Fresh-clone full GPU reproduction and a clean dependency installation have not
  been performed for this update. MSVD/efficiency/CPR depend on historical assets;
  MSR-VTT preparation still depends on a model-source preflight record.
- Original checkpoints/caches are hashed but not hosted in a downloadable release.
  A future artifact release needs asset packaging and third-party rights checks.
- The publication scan covers the current tree and likely credential/path patterns,
  not all possible secrets or historical commits. Ignore rules do not untrack files.
- This update introduces no project-wide license. Third-party assets retain their
  own terms; a repository-wide license decision remains with the author.

The CI workflow validates the public arithmetic, release scan, Python syntax and
standard-library evidence tests. It does not run the neural-model test suite,
download datasets or claim full experimental reproducibility.
