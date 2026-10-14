# MSR-VTT A/B adaptation contract — draft at data gate (2026-09-27)

Status: **engineering gate NOT CHECKED**. The data gate is FAIL, so this file is an implementation contract and inspection record, not a retroactively frozen formal training protocol. Formal step count, endpoint and caption thresholds must be frozen after train/val smoke and before any six-arm run. No MSR-VTT model training or test inference has started.

## Existing runner assumptions found

- `scripts/run_full_msvd_balanced.py` pins `TRAIN`, `VAL`, their SHA-256, 1200/100 counts, `FORMAL_EPOCHS=66`, `FORMAL_STEPS=9900`, and milestones 5/20/40/66. `make_batch_schedule` rejects counts not divisible by 8. MSR-VTT train has 6,513 IDs, so path replacement is invalid.
- `scripts/run_repeated_seeds.py` imports those MSVD constants and historical cached features/checkpoint contract, assumes 1,200 visits and old reference counts, and uses MSVD-specific semantic gates. Its paired trace and strict reload logic can be reused as functions only after parameterization and testing.
- `scripts/run_test670_evaluation.py` pins the MSVD test annotation hash, six MSVD checkpoint paths, 670 videos/27,763 references, and fixed MSVD cached features. A new independent test runner is required, and may only be used after validation decisions are frozen.
- `src/data/data_loader.py` silently drops records with missing frames at dataset construction. The MSR-VTT adapter must audit exact expected IDs first and fail closed on any missing frame directory; it cannot inherit silent filtering.

## Separate adapter to implement after full data audit

Write new MSR-VTT annotation files from the pinned source manifest and the archive's full train/val/test lists. Keep existing MSVD scripts untouched. `frames_dir` must point to verified eight-frame directories; all IDs and counts must survive dataloader initialization. Use the existing 224×224 ImageNet normalization and frozen ViT-B/16 raw 768-dimensional pooled feature cache. Hash every annotation, frame source, cache and command config. The sample cache probe is evidence of feasibility on four train/val clips only; a full cache equality check remains required.

Use the original local GPT-2 and frozen ViT backbone. Train only projection and prefix mapper from fresh, shared per-seed initial states on MSR-VTT; never initialize from MSVD trained mapper checkpoints. A is caption CE; B is caption CE plus the already verified symmetric InfoNCE with weight 0.10, temperature 0.07, normalized mean visual prefix and frozen token embedding mean, and exact-caption multi-positive masking. No InfoNCE/CPR hyperparameter search.

For each seed 123/124/125, A and B must receive identical initial state, ordered video batches, reference choices, principal RNG state and step count. A full epoch can use 813 batches of 8 plus one batch of 9 distinct IDs, covering 6,513 videos without a singleton or duplicate within the final batch. Every InfoNCE row must have at least one legal positive and negative under the current mask. The 20 train references per video imply **at least 20 balanced visits**, or 16,280 optimizer steps per arm, to cover each train caption once. This is a coverage floor, not an approved convergence budget. Run a train/val pilot and measure actual step time before freezing a shared six-arm budget; do not import the MSVD 66 epochs/9,900 steps.

## Smoke checks and formal stop condition

After the data gate passes: (1) overfit 16 train videos and run short A/B steps; (2) assert first-token/EOS supervision and no truncation or empty captions; (3) assert legal InfoNCE positives/negatives and finite gradients to both trainable modules; (4) verify GPT-2 and ViT hash stability; (5) compare direct and cached features at max absolute error ≤1e−6; (6) save and strictly reload model, optimizer, RNG and loader schedule, then compare the resumed next step; (7) compare A/B trace fields and monitor peak memory below 4,096 MiB. Preserve any OOM/NaN/failure log as a result of this gate, not as a formal training result.

Only after those checks, freeze a versioned formal config: seeds, source and split hashes, exact batch/reference schedule, common step count and endpoint, checkpoint milestones, validation caption viability gate with data-specific justification, fixed greedy decode (`max_new_tokens=24`, no sampling, no-repeat trigram, EOS/padding), caption metrics (`pycocoevalcap==1.2`, raw and ×100), test2990 retrieval candidate/query rule, and technical failure/restart policy. Validation may inform the pre-run budget; test metrics may not. The already registered transfer interpretation remains: all B caption gates pass; B−A T2V R@1 mean is positive and at least two of three paired seeds are positive. Valid negative results stay in the report.

Current gate: NOT CHECKED. Smallest next action is complete media download and full data audit, then implement the independent adapter and 16-video smoke. No formal six-arm long training is authorized by this draft.
