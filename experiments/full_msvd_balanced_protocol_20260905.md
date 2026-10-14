# Full-MSVD balanced-caption viability run: preregistered protocol

Locked before observing training results, 2026-09-05.

## Question

Can the repaired caption-only model learn a diverse, video-grounded caption
mapping on the full MSVD train1200 split when every training reference is
covered by a deterministic balanced schedule? This is the final viability gate
before adding InfoNCE or evaluating test670.

## Fixed training design

- Original local GPT-2 and pretrained ViT-B/16; prefix length four.
- Completely fresh `encoder.proj` and `decoder.mapper`. Do not load Stage3,
  historical mapper, recovered v5, or any prior overfit checkpoint.
- Freeze all GPT-2 parameters and the ViT backbone. Train only
  `encoder.proj` and `decoder.mapper`; no InfoNCE.
- Pinned full MSVD train1200 and val100. Do not read or score test670.
- Train annotations contain 48,774 captions, 18--66 per video. Each video gets
  one deterministic, per-video shuffled reference per epoch. Run 66 epochs so
  every stored training reference is visited at least once.
- Batch size 8, 150 steps per epoch, exactly 9,900 optimizer steps. Seed 123,
  dedicated batch-order seed 456, AdamW, constant learning rate 3e-4, weight
  decay 0.01. No AMP, scheduler, clipping, accumulation, or early stopping.
- Full-sequence causal supervision predicts the first caption token from the
  final visual-prefix position, supervises the first EOS, and ignores later EOS
  padding. Maximum length 64; all train/validation references must fit without
  truncation.
- Fixed validation milestones: epochs 5, 20, 40, and 66. Epoch 66 is the only
  primary endpoint; milestone results do not select a checkpoint or change the
  budget.

## Frozen-backbone caching

Load all train and validation videos with the production loader (8 uniformly
sampled 224x224 frames). Cache pooled 768-dimensional ViT features, while
keeping the trainable 768-to-256 projection inside the graph. For every cache
batch, `encoder.proj(cached_feature)` must match direct `encoder(video)` to
maximum absolute error <=1e-6. Save cache tensors, ordered video IDs, source
video-tensor hashes, and cache hashes. Recheck the first batch after training.

The smoke run uses the first 16 training and first 8 validation videos for two
steps; smoke weights never initialize the formal run.

## Validation endpoint

At each milestone, evaluate all 100 validation videos and all 4,290 references
with equal reference weighting inside each video and equal video weighting
overall.

- Native and fixed single-cycle shuffled-prefix teacher-forced NLL.
- Per-video shuffled-minus-native NLL and 10,000-sample paired video bootstrap.
- Greedy generation from an empty prompt: max 24 new tokens, min 0, no
  sampling, no-repeat-ngram 3, repetition penalty 1.0.
- Nonempty, exact-string uniqueness, largest exact template, native-versus-
  shuffled caption change, EOS termination, exact match to any reference, and
  maximum token-F1 against references. Token-F1 and unsmoothed BLEU-4 are
  triage statistics, not paper metrics.

## Primary viability gate at epoch 66

All conditions must hold:

1. nonempty generation rate =100%;
2. unique exact caption rate >=30%;
3. largest exact template rate <=25%;
4. 95% paired bootstrap lower bound for shuffled-minus-native NLL >0;
5. at least 80/100 videos have positive shuffled-minus-native NLL.

If the gate passes, proceed to the separately preregistered InfoNCE ablation
and only then fix a checkpoint before touching test670. If the gate fails, do
not add InfoNCE or switch language models opportunistically; diagnose the
failed component using the saved epoch trajectory.

## Audit controls and boundaries

- Save complete batch/RNG/caption/feature traces for all 9,900 steps.
- Verify every train video is seen 66 times and every stored reference at least
  once; verify source/data/model hashes and frozen weights.
- Save epoch 5/20/40/66 checkpoints for recovery and strict reload.
- One seed, one frozen-decoder configuration. Passing does not establish paper
  quality, standard benchmark performance, or human semantic correctness.
- No threshold, subset, decoding option, training budget, or primary endpoint
  may change after the formal run begins.
