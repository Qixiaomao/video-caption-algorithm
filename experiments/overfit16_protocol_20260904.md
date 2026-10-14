# Sixteen-video overfit diagnostic: preregistered protocol

Locked before observing training results, 2026-09-04.

## Question

Can the repaired caption-training path memorize sixteen fixed, semantically
distinct MSVD video-caption pairs when only the fresh visual projection and
prefix mapper are trainable? This is a pipeline-capability diagnostic, not a
generalization experiment and not paper-ready evidence.

## Fixed intervention and controls

- Original local GPT-2 weights (not Stage3), pretrained ViT-B/16, prefix length
  four, and one freshly initialized 768-to-256 encoder projection plus mapper.
- Freeze the ViT backbone and all GPT-2 parameters. Train only
  `encoder.proj` and `decoder.mapper`; do not load any historical visual or
  mapper checkpoint and do not use InfoNCE.
- Use the sixteen video IDs and exact canonical captions embedded in
  `scripts/run_overfit16.py`. Each caption is an existing MSVD training
  reference. No validation or test examples are used.
- Seed 123, batch size 8, eight uniformly sampled 224x224 frames, maximum 64
  caption tokens, AdamW, constant learning rate 3e-4, weight decay 0.01. No
  AMP, scheduler, gradient clipping, accumulation, or early stopping.
- Run exactly 500 optimizer steps. The sixteen samples form two batches per
  epoch; shuffled orders are deterministic. Evaluate at steps 0, 50, 100, 250,
  and 500 without changing the budget.
- Use full-sequence causal supervision: the last visual-prefix position
  predicts the first caption token, the first EOS is supervised, and later EOS
  padding is ignored.
- Use the real frame-loading, ViT, projection, mapper, and GPT-2 paths. Do not
  cache visual features for this check.
- Save input/RNG trace evidence for all steps. Verify source/data/model hashes,
  all sixteen samples seen, frozen weights unchanged, and both trainable
  modules changed. Refuse to overwrite an existing output directory.

## Measurements

At every evaluation milestone, use `eval()` mode and the same fixed donor
cycle for shuffled-prefix controls.

1. Video-balanced teacher-forced target NLL under the native prefix.
2. Shuffled-minus-native target NLL, including a per-video value.
3. Greedy free generation with empty prompt, maximum 24 new tokens, minimum
   zero, no sampling, no-repeat-ngram 3, and repetition penalty 1.0.
4. Normalized exact target matches, nonempty rate, unique-caption rate,
   largest-template rate, native-versus-shuffled caption-change rate, and EOS
   termination rate.

The final **strict memorization gate** passes only if all are true at step 500:

- native target NLL <= 0.50;
- mean shuffled-minus-native NLL >= 1.00;
- normalized exact target matches >= 14/16;
- unique native captions >= 14/16;
- largest native template <= 2/16;
- EOS termination = 16/16.

## Interpretation boundaries

- Full pass: this exact repaired frozen-decoder chain has enough capacity and
  optimization budget to memorize the tiny mapping. The full-data failure then
  points toward generalization, data ambiguity, objective balance, or training
  budget rather than a completely broken conditioning path.
- Low native NLL and positive mismatch penalty but failed generation: the
  teacher-forced path learned the pairs, while autoregressive generation or
  exposure bias remains the leading category.
- Low native NLL without mismatch penalty: the text targets were learned
  without reliable video identity binding.
- High native NLL: the current trainable subnetwork/optimizer/supervision path
  did not fit even the tiny set. This narrows the next check to training
  mechanics, optimization, or frozen-decoder capacity; it does not by itself
  identify which one.

No threshold or budget may be changed after seeing the run. A later
last-four-GPT-2-block unfreeze test, if needed, is a separate preregistered
experiment.
