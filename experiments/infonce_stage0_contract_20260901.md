# InfoNCE stage-0 implementation contract (2026-09-01)

## Scope

Stage 0 implements and tests the frozen experiment design. It does not run a
GPU training experiment and therefore produces no model-quality result.

## Frozen objective

- Caption objective: causal-LM cross-entropy with the first caption token
  predicted from the visual prefix.
- Video representation: mean over the visual prefix tokens, then L2 normalize.
- Text representation: frozen GPT-2 WTE embeddings, mean over tokens before the
  first EOS-as-padding token, then L2 normalize.
- Alignment choices: `none`, `positive_cosine`, or symmetric in-batch `infonce`.
- InfoNCE temperature: fixed at `0.07` for the first sweep.
- False-negative policy: captions with exactly identical token content share a
  multi-positive set (`exact_caption`).
- GPT-2 tail: frozen for every alignment ablation.
- No projection head, queue, hard-negative miner, or inference-time parameter is
  introduced in stage 0.

## Full-data contract

The training CLI now defaults to:

- `data/processed/msvd_full/train/annotations.json` (1,200 videos)
- `data/processed/msvd_full/val/annotations.json` (100 videos)

Each stage-0 checkpoint records annotation hashes plus the pinned dataset
manifest and Hugging Face source revision. Test data remains outside the
training CLI and must not be used before the configuration is locked.

## Checkpoint contract

Checkpoint version 3 contains:

- model, optimizer, and optional scheduler state;
- global step, epoch, and batch position;
- Python, CPU Torch, and CUDA RNG state;
- DataLoader generator state for mid-epoch continuation with `num_workers=0`;
- objective settings and physical batch size;
- annotation and dataset-manifest hashes.

`--resume_ckpt` preserves the old weight-restore behavior.
`--resume_training_state` enables exact training-state recovery and rejects any
objective, batch-size, or dataset-hash mismatch.

## Stage-1 entry commands (not executed in stage 0)

Use the same recovered v5 checkpoint and distinct output directories for each
smoke run. Start with `--smoke_test --max_steps 2 --val_every 0` and physical
batches 2, 4, then 8:

```powershell
.\.venv\Scripts\python.exe -m src.cli.train_caption_mapper `
  --resume_ckpt checkpoints/<V5_CHECKPOINT>.pt `
  --alignment_objective infonce `
  --alignment_weight 0.10 `
  --info_nce_temperature 0.07 `
  --false_negative_policy exact_caption `
  --batch_size 2 `
  --smoke_test `
  --max_steps 2 `
  --val_every 0 `
  --run_dir runs/infonce_smoke_b2 `
  --ckpt_name infonce_smoke_b2.pt
```

Replace `<V5_CHECKPOINT>` only after its exact filename and SHA-256 are checked.
Do not reuse the same `run_dir` or checkpoint name across batch sizes.

## Verification completed

- Aligned pairs have lower loss than shuffled pairs.
- Equal embeddings produce the expected `log(batch_size)` diagonal-InfoNCE
  baseline.
- Physical batch size 1 is rejected.
- Exact duplicate captions are not treated as negatives.
- Tokens at and after EOS-as-padding are excluded from the text representation.
- InfoNCE gradients reach `encoder.proj` and `decoder.mapper`, while frozen WTE
  receives no gradient.
- Full checkpoint state restores model, optimizer, and RNG.
- Strict model loading rejects architecture mismatches.
- Existing benchmark, configuration-chain, and caption-supervision regression
  tests remain green.
