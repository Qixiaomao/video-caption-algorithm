# Frozen three-seed MSVD test670 evaluation protocol

Registered 2026-09-08 before any test caption or retrieval score was computed.

## Question

After caption viability and the InfoNCE alignment gain replicated across three
validation seeds, what caption and retrieval performance do the frozen models
obtain on the untouched 670-video MSVD test split?

## Test data lock

- Annotation: `data/processed/msvd_full/test/annotations.json`
- SHA-256: `9a1fb940f1b9513a882e7ead5620397b0bf39ad1237b9c8ccb57ce765de50b1f`
- Videos: 670 unique video IDs
- References: 27,763 total, 21 to 81 per video
- Frames: 670 nonempty frame directories, 5,360 JPEG frames
- Sampling: 8 uniformly sampled frames per video at 224 x 224, using the
  repository data loader and ImageNet normalization

This stage is authorized to read test670. Test predictions or scores must not
be used to select a seed, checkpoint, epoch, decode configuration, loss weight,
or temperature.

## Frozen checkpoints

All six epoch-66 checkpoints will be evaluated and all outcomes retained.

| Seed | Arm | Checkpoint | SHA-256 |
|---:|:---:|---|---|
| 123 | A | `runs/full_msvd_balanced_20260905/checkpoint_epoch066.pt` | `8155d68c5b3ac9f3f7396027a32b2e0ebea052a17aef79ecce8cf10a1e0f9e6a` |
| 123 | B | `runs/full_msvd_infonce_ablation_20260906/checkpoint_epoch066.pt` | `891cdcceef2b52d09092dde00ef35f9e1ee652e686b30178b5c1a9aee81c7e7c` |
| 124 | A | `runs/repeated_seeds_20260908/seed124/A/checkpoint_epoch066.pt` | `8acc816040d14a8581095eb80262b2fcb8afb1934876c2b659936a8d3c062be5` |
| 124 | B | `runs/repeated_seeds_20260908/seed124/B/checkpoint_epoch066.pt` | `147330abd9e909a50be78b91be426b7ade23104cfcdeac34ee1bb777327067a7` |
| 125 | A | `runs/repeated_seeds_20260908/seed125/A/checkpoint_epoch066.pt` | `5e5d5cae5d663588cb5c531aaf76e45634bf9fbebb59e75b3fd5c8ab8f5cee37` |
| 125 | B | `runs/repeated_seeds_20260908/seed125/B/checkpoint_epoch066.pt` | `9e59425dfd28b0b2df3539bd023f712506c533b8f65421ee07ca7679ddeaefe5` |

Checkpoint loading must use `strict=True`. Each payload must record checkpoint
version 3, epoch 66, step 9,900, a completed epoch, balanced-all-references
caption supervision, and the registered A or B alignment contract. Frozen
GPT-2 and ViT component hashes must agree across all six checkpoints.

## Frozen inference

The decode configuration is inherited from the validation semantic-gate runs:

- empty text prompt;
- greedy decoding, one beam, no sampling;
- maximum 24 new tokens and minimum 0;
- temperature 1.0 and top-p 0.9, inactive because sampling is disabled;
- no-repeat n-gram size 3;
- repetition penalty 1.0;
- GPT-2 EOS used as EOS and padding.

The same raw frozen-ViT feature cache will be used for all models. The cache is
allowed only after direct-versus-cached equivalence has maximum absolute error
at most 1e-6. Model-specific projection and prefix-mapper layers are always
applied after the shared raw cache.

## Caption metrics

Use `pycocoevalcap==1.2` with its PTB tokenizer and all references per video.
Report:

- BLEU-1, BLEU-2, BLEU-3, and BLEU-4;
- METEOR;
- ROUGE-L;
- CIDEr.

Raw evaluator scores and the conventional score multiplied by 100 must both
be preserved. SPICE is excluded because it is not part of the locked paper
metric set and adds an external model/resource dependency. The earlier
dependency-free BLEU and token-F1 remain diagnostics and are not substituted
for these metrics.

## Retrieval metrics

Also evaluate the already registered representation on test670:

- video: normalized mean of the learned visual-prefix tokens;
- text: normalized masked mean of frozen GPT-2 token embeddings;
- all 27,763 test captions are text queries;
- report text-to-video and video-to-text R@1, R@5, R@10, median rank, mean
  rank, positive-minus-negative similarity gap, and a fixed seed-123
  deranged-label control.

These are held-out retrieval ranks but remain representation-specific. They
must not be described as an external benchmark implementation unless the
paper defines this representation and protocol explicitly.

## Aggregation and interpretation

- Report every seed and arm.
- Report arm mean and sample standard deviation across seeds.
- Report paired B-minus-A changes for every seed and their mean and sample
  standard deviation.
- Do not choose a best seed for the paper headline.
- Caption metrics answer whether InfoNCE transfers to caption quality.
- Retrieval metrics answer whether the registered alignment gain transfers to
  the held-out split.
- A null or negative caption change does not invalidate a positive retrieval
  result, but the claims must keep those outcomes separate.

## Execution and integrity gates

1. Protocol, evaluator source, dependency pins, and tests are committed before
   formal scoring.
2. Unit tests pass before test670 generation.
3. The output directory must not exist before the run and is never overwritten.
4. Each model emits exactly one nonempty prediction for each of 670 video IDs.
5. The first inference batch reproduces exactly on immediate replay.
6. Checkpoint, source, annotation, and feature-cache hashes are recorded.
7. Pinned source and data are unchanged at process exit.
8. All failures and all six outcomes are retained.
9. `test_split_read=true` after this stage and `test_tuning_performed=false`.
10. `paper_ready` remains false until the user verifies the generated
    predictions, report, and numerical claims.
