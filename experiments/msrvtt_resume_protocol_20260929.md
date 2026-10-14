# MSR-VTT formal run continuation — 2026-09-29

The 2026-09-28 formal run was stopped at the user's request. Its `seed123/A` training log reached step 20,411, but the last full recoverable checkpoint is epoch 20 / step 16,280. The paused run, including its post-checkpoint trace and failed/incomplete state, remains read-only.

## Pinned recovery source

- Original run: `runs/msrvtt_ab_formal_20260928`.
- Checkpoint: `seed123/A/checkpoint_epoch020.pt`, SHA-256 `b1a22a97686d52a4b363048c47e4d2f4cd33ec0519379e05e4213cffddb79f2a`.
- Frozen config: `experiments/msrvtt_ab_formal_config_20260928.json`, SHA-256 `75fbdd3131fc9177d0fcb828f4a0f6be65e6e884c3249434813b39d4e7f9ec38`.
- Schedule: `seed123/schedule.json`, SHA-256 `7ecf8d9e30d79901e49c89621a4fe465ae786e4a461a0ef0b877a7167a356b42`.
- Data gate: SHA-256 `d870bf7c66a339323a8cad31ba2ccd9d1081e31928f20715cea54d7393f38138`.
- The checkpoint is version 3 and contains model, AdamW optimizer, CPU/CUDA/Python RNG, epoch-complete progress and loader state. Its training/dataset contracts and the original source-file hashes were checked against current files before any resumed GPU work.

## Recovery rule

`scripts/verify_msrvtt_checkpoint_resume_20260929.py` restored the full state in a separate run. `runs/msrvtt_resume_verify_20260929/result.json` records PASS: steps 16,281–16,344 exactly matched the uninterrupted trace and all floating-point loss values (maximum absolute error 0), with 2,314 MiB peak reserved GPU memory. It did not read test.

`scripts/resume_msrvtt_ab_formal_20260929.py` creates a distinct output root and refuses to overwrite one. It regenerates the initial model and schedule, requires equality with the original, copies only the completed 16,280-step prefix and completed checkpoint/validation milestones, restores epoch-20 full training state, and resumes seed123/A from epoch 21. For every available old suffix step 16,281–20,411 it requires exact trace and metric equality; a mismatch fails and leaves the new run as evidence. Then it finishes seed123/A at the already frozen epoch-30 endpoint and runs the remaining five arms through the unmodified formal arm implementation. All A/B paired traces must match before validation can be frozen. The frozen config, seeds, endpoints, validation rule, decode and test protocol are unchanged.

No test feature, prediction or metric is computed before all six arms finish and `validation_freeze.json` is written. A valid negative validation finding is retained. Source and new run paths, hashes, failures and checkpoints remain available for audit.
