# GPT-2 prefix sensitivity: full val32 matrix (2026-09-03)

## Decision

Both E0 and E6 use video-specific prefix information. The hypothesis that the
decoder completely ignores the prefix is contradicted by the paired-caption
NLL and generation interventions on this subset. E6 has a larger
correct-versus-mismatched NLL advantage, including after the first caption
token, but its free-running captions are more concentrated in a generic
template. Sensitivity, grounding under teacher forcing, and free-generation
quality must therefore be treated as different outcomes.

The strongest supported diagnosis is **remaining template concentration during
generation despite measurable visual conditioning**, not a completely broken
visual-input connection. These probes do not isolate the exact causal layer
or prove that replacing/fine-tuning GPT-2 will fix the problem.

## Scope and reproducibility

- Executed the previously frozen `prefix-sensitivity-v1` protocol, with no
  training or model-weight changes. Full val100 and test670 were not evaluated
  in this turn.
- E0 recovered-v5 versus E6 InfoNCE lambda 0.20, step-150 last checkpoint.
- Same first 32 validation videos, all 1,351 annotated references, same donor
  cycle, seed 123, 10 conditions each, 14 layer boundaries each.
- No reference truncations. Reference microbatch 4; video/prefix batch 1.
- GPT-2 float32; 8 frames at 224x224; greedy generation with empty prompt,
  no synthetic BOS, max 24 new tokens, min 0, no-repeat-ngram size 3,
  repetition penalty 1.0.
- Both runs used clean source commit
  `968177e00d1a987f6b3d1271bbb94e749e689ba5` and diagnostic script SHA-256
  `abd4b415a9f2d19c72991f49a6d8a9e0a01710b582438a0b660cbfce6eab402d`.
- Full input/configuration details remain in
  `experiments/prefix_sensitivity_protocol_20260903.md` and both raw reports.
- Each run completed all 320 video-condition evaluations and 13,510
  reference-condition likelihood evaluations (640 and 27,020 across both).
- Peak CUDA allocated / reserved memory: **904.74 / 1,104.00 MiB** for each
  run on RTX 3050 4GB. No OOM or nonfinite result occurred.

## Main semantic-control results

Define delta NLL as NLL(shuffled prefix) minus NLL(correct prefix). Each
reference is first token-averaged, then references are averaged within a video,
then videos receive equal weight. Positive delta favors the correct video.

| Metric | E0 v5 | E6 InfoNCE 0.20 |
|---|---:|---:|
| Native caption NLL | 3.3790 | 3.1237 |
| Shuffled delta NLL | 0.0811 | 0.2000 |
| Fixed-donor 95% bootstrap interval | [0.0318, 0.1316] | [0.1381, 0.2672] |
| Delta NLL excluding first caption token | 0.0622 | 0.2150 |
| Excluding-first 95% interval | [0.0172, 0.1067] | [0.1481, 0.2861] |
| Videos with positive delta NLL | 22/32 | 29/32 |
| First-token JSD, shuffled vs correct (nats) | 0.0541 | 0.0495 |
| First-token argmax changes | 0/32 | 0/32 |
| Full-caption changes | 30/32 (93.75%) | 22/32 (68.75%) |
| Native distinct exact captions | 14 | 9 |
| Native largest exact template | 8/32 (25.00%) | 20/32 (62.50%) |

The initial article stays `a`, but later words and reference likelihoods depend
on the video prefix. Thus zero first-token flips cannot diagnose a disconnected
decoder. E6's smaller caption-change rate also cannot by itself establish
weaker conditioning: with 20 identical native predictions, many donor swaps
necessarily collide on the same output sentence.

An exploratory paired comparison of the two per-video grounding deltas gives
E6 minus E0 = 0.1189, interval [0.0674, 0.1702]. Excluding the first token gives
0.1527, interval [0.1069, 0.1999]. These are descriptive post-run contrasts, not
additional predeclared acceptance tests or a causal attribution to InfoNCE
alone. E0 did not undergo the same additional training as E6; the E3 control
was not part of this prefix matrix.

All intervals use 10,000 video-level paired resamples under the fixed donor
mapping. They do not include donor-rematching or training-seed uncertainty,
and are not multiple-comparison corrected.

## Full intervention matrix

Counts use 32 videos. `Changed` compares full captions with native; `Flip`
compares the first-token argmax. `Unique` counts normalized exact strings,
including an empty string if present. `Max` is the largest exact-string count.
Reported JSD values near zero (about 6e-18) are rounded to zero.

### E0 recovered-v5

| Condition | Delta NLL [95% interval] | JSD | Changed | Flip | Nonempty | Unique | Max |
|---|---|---:|---:|---:|---:|---:|---:|
| native | 0 [0, 0] | 0 | 0 | 0 | 32 | 14 | 8 |
| repeat | 0 [0, 0] | 0 | 0 | 0 | 32 | 14 | 8 |
| shuffled | 0.0811 [0.0318, 0.1316] | 0.0541 | 30 | 0 | 32 | 14 | 8 |
| zero | 2.8235 [2.6364, 3.0207] | 0.6860 | 32 | 32 | 32 | 1 | 32 |
| mean | 0.0389 [0.0104, 0.0699] | 0.0269 | 31 | 0 | 32 | 1 | 32 |
| scale_0 | 2.8235 [2.6364, 3.0207] | 0.6860 | 32 | 32 | 32 | 1 | 32 |
| scale_025 | 0.1187 [0.0658, 0.1704] | 0.2371 | 30 | 5 | 27 | 11 | 7 |
| scale_05 | 0.0151 [-0.0042, 0.0343] | 0.0737 | 21 | 1 | 31 | 11 | 11 |
| scale_1 | 0 [0, 0] | 0 | 0 | 0 | 32 | 14 | 8 |
| scale_2 | 0.0168 [0.0003, 0.0353] | 0.0111 | 16 | 0 | 32 | 10 | 9 |

### E6 InfoNCE lambda 0.20

| Condition | Delta NLL [95% interval] | JSD | Changed | Flip | Nonempty | Unique | Max |
|---|---|---:|---:|---:|---:|---:|---:|
| native | 0 [0, 0] | 0 | 0 | 0 | 32 | 9 | 20 |
| repeat | 0 [0, 0] | 0 | 0 | 0 | 32 | 9 | 20 |
| shuffled | 0.2000 [0.1381, 0.2672] | 0.0495 | 22 | 0 | 32 | 9 | 20 |
| zero | 3.0788 [2.8818, 3.2902] | 0.6875 | 32 | 32 | 32 | 1 | 32 |
| mean | 0.1177 [0.0833, 0.1527] | 0.0242 | 12 | 0 | 32 | 1 | 32 |
| scale_0 | 3.0788 [2.8818, 3.2902] | 0.6875 | 32 | 32 | 32 | 1 | 32 |
| scale_025 | 0.1199 [0.0665, 0.1782] | 0.1761 | 27 | 1 | 31 | 17 | 6 |
| scale_05 | 0.0154 [-0.0034, 0.0340] | 0.0318 | 6 | 0 | 32 | 8 | 18 |
| scale_1 | 0 [0, 0] | 0 | 0 | 0 | 32 | 9 | 20 |
| scale_2 | 0.0007 [-0.0108, 0.0137] | 0.0060 | 5 | 0 | 32 | 10 | 18 |

## What the zero, mean, and scale controls show

- Zero prefix produces the same non-caption-like string for every video in
  both models, and incurs a large NLL penalty. This demonstrates strong
  dependence on having a prefix, but zero vectors are out-of-distribution and
  retain prefix positions/attention. This is not a pure no-prefix LM test.
- E6's fixed mean prefix produces `a man is playing a guitar` for all 32
  videos, the same template that already covers 20 native videos. Correct
  per-video prefixes have better reference likelihood than the mean prefix.
- E6 scale 0.25 increases exact-string diversity to 17/32 and reduces the
  largest exact template to 6/32, but produces one empty caption and worsens
  NLL by 0.1199. Furthermore, two nearly synonymous piano templates each
  occur six times: `a man is playing the piano` and `a man is playing piano`.
  Exact-string diversity is not semantic correctness or genuine task variety.
- E0 scale 0.25 produces five empty captions; E0 scale 0.5 produces one.
  Prefix scaling is therefore not selected as a repair or production setting.
- Scale 2 does not provide a reliable E6 NLL improvement; its interval crosses
  zero. Sensitivity to amplitude motivates investigation, not a winning scale.

## Layerwise shuffled-versus-native response

The following measurements use the last prefix position, which predicts the
first caption token. Relative L2 is normalized by the corresponding native
hidden-state norm. It is not an information-retention percentage.

| Boundary | E0 cosine | E0 relative L2 | E6 cosine | E6 relative L2 |
|---|---:|---:|---:|---:|
| Embedding + position | 0.8110 | 0.8039 | 0.8586 | 0.6944 |
| Block 1 | 0.8774 | 0.5377 | 0.9046 | 0.4702 |
| Block 2 | 0.8611 | 0.5645 | 0.9132 | 0.4511 |
| Block 3 | 0.8967 | 0.8650 | 0.9758 | 0.3457 |
| Block 4 | 0.8857 | 0.9319 | 0.9770 | 0.3552 |
| Block 5 | 0.8698 | 1.0183 | 0.9790 | 0.3627 |
| Block 6 | 0.8612 | 1.0622 | 0.9802 | 0.3663 |
| Block 7 | 0.8624 | 1.0758 | 0.9811 | 0.3672 |
| Block 8 | 0.8680 | 1.0574 | 0.9820 | 0.3633 |
| Block 9 | 0.8795 | 1.0183 | 0.9827 | 0.3577 |
| Block 10 | 0.8890 | 0.9564 | 0.9827 | 0.3516 |
| Block 11 | 0.8955 | 0.8419 | 0.9816 | 0.3408 |
| Block 12, before final LN | 0.8065 | 0.5890 | 0.9440 | 0.3060 |
| Final LN | 0.9901 | 0.2701 | 0.9938 | 0.2014 |

E6 becomes more directionally similar by block 3, and final LN raises cosine
for both models. However, the differences do not vanish: E6 still has relative
L2 0.2014 after final LN, nonzero output-distribution JSD, and a positive
after-first-token NLL advantage for the correct prefix. Responses are not a
simple monotonic decay across layers. These findings do not establish block 3
or final LN as a broken component. Later teacher-forced/generated positions
were not captured by these prefix-only layer probes.

## Verification and artifacts

- Native predictions match the corresponding prior val100 semantic-report
  predictions for all 32 videos in each model, with zero mismatches. This
  confirms the intervention runner preserves the existing native decode path.
- Repeated native and scale-one logits/captions/NLL and hidden states passed
  consistency checks. Zero and scale-zero per-video outputs are identical.
- Shuffled captions/logits follow the predeclared donor permutation.
- Independently re-aggregated all saved token losses into reference/video
  NLLs and per-video deltas; recomputed all 20 bootstrap summaries from saved
  deltas. All checks passed.
- Both runs have identical sample IDs, donor assignments, reference texts and
  token IDs, source hashes, and decoding settings. No truncated references.
- Checkpoint and source hashes were verified again after execution.
- Regression suite: 48 tests passed via
  `.\.venv\Scripts\python.exe -m unittest discover -s tests -q`.

| Raw artifact | Bytes | SHA-256 |
|---|---:|---|
| `reports/prefix_sensitivity_e0_val32_20260903.json` | 8,642,197 | `e1bffe8150fa0c9e575d2f3254d60cf955b63c65d5f58742adf4f085359b8361` |
| `reports/prefix_sensitivity_e6_val32_20260903.json` | 8,643,031 | `e4759cf1d6e3ac64eebf9e963b8673e470a76924706b278ca0cfc3c8618dff72` |

Raw reports are local/git-ignored. This versioned record preserves the
complete matrix and its limits. No original report was overwritten.

Commands executed from the repository root:

```powershell
.\.venv\Scripts\python.exe -u scripts/diagnose_prefix_sensitivity.py `
  --checkpoint checkpoints/msvd_mapper_recovered_v5.pt `
  --expected-checkpoint-sha256 17dda154899f48c89d12aa02e0014c01ac80af7a8965f41c330fe8d91cb1eec3 `
  --limit 32 --preset full `
  --output reports/prefix_sensitivity_e0_val32_20260903.json

.\.venv\Scripts\python.exe -u scripts/diagnose_prefix_sensitivity.py `
  --checkpoint checkpoints/stage2_e6_infonce_l020.last.pt `
  --expected-checkpoint-sha256 f1e2159e2139fd49c1c41f2bcda41958a7a389ced11219e9e7878a2502a2565f `
  --limit 32 --preset full `
  --output reports/prefix_sensitivity_e6_val32_20260903.json
```

## Next boundary

Proceed next, after authorization, with the planned val100 core replication
(native, repeat, shuffled, zero), using all 4,290 references. Do not change
weights, decoding, or prefix scale during that replication.

Do not upgrade the result to a full semantic-gate pass: val32 exact uniqueness
rates are sample-size dependent and not interchangeable with val100 rates.
Both checkpoints still have the previously established val100 generation
limitations. All outputs are explicitly `paper_ready=false`.

If the val100 findings replicate, the next design discussion should address
how useful conditional likelihood improvements translate into free-running
caption generation: prefix calibration, later-token conditional responses,
training objectives, and decoding behavior remain candidate mechanisms.
Replacing GPT-2, adding cross-attention, or unfreezing layers is not justified
as a necessary fix by this experiment alone.
