# Frozen bounded qualitative caption audit: results

Completed 2026-09-09 under the committed
`experiments/qualitative_caption_audit_protocol_20260909.md` protocol.

## Decision

The audit supports the same narrow interpretation as the automatic test670
metrics: **InfoNCE improves alignment and sometimes produces a more compatible
caption, but this 12-video audit does not show a reliable qualitative caption
improvement.** Fine-grained actions and objects remain the main failure mode.

The captions are consistently readable, and many preserve the broad scene or
subject. They are frequently factually wrong at the object/action level. This
means the paper may claim preserved aggregate caption quality and improved
retrieval, but it should not claim improved human semantic quality or robust
fine-grained video understanding.

## Frozen audit setup

- Population: 670 test videos from the unchanged official evaluation input.
- Sample: 12 videos selected only by salted SHA-256 of video ID.
- Outputs: the existing seed 123/124/125 A/B prediction files; no new model
  inference was performed.
- Evidence per video: eight extracted frames, six unique dataset references,
  and six anonymized frozen predictions.
- Blinding: the 72 ratings were committed at `2f9cf85` before the A/B key was
  opened.
- Rubric: subject, action, context, hallucination, and fluency, each binary.
  The preregistered broad-grounding rule was semantic score at least 2/3 with
  no material unsupported claim.
- This is an AI-assisted error audit, not a human evaluation.

## Arm-level results

Each arm contains 36 rated captions: 12 videos times three seeds.

| Audit measure | Caption-only A | Caption+InfoNCE B | B - A |
|---|---:|---:|---:|
| Mean semantic score, 0-3 | 1.528 | **1.722** | +0.194 |
| Broadly grounded | 9/36 | **10/36** | +1 |
| Material hallucination | 27/36 | **25/36** | -2 |
| Subject acceptable | 28/36 | **29/36** | +1 |
| Action acceptable | 13/36 | **14/36** | +1 |
| Context compatible | 14/36 | **19/36** | +5 |
| Fluent/readable | 36/36 | 36/36 | 0 |

The largest arm-level difference is context compatibility, not action
recognition. The broad-grounding difference is only one caption. Because this
is a small single-evaluator audit, these counts are descriptive and must not be
reported as test-set rates.

## Paired A/B result

The paired rule compares A and B within the same video and seed using semantic
score first and hallucination second.

| Seed | B better | A better | Tie | Caption changed |
|---:|---:|---:|---:|---:|
| 123 | 1 | 2 | 9 | 6/12 |
| 124 | 6 | 2 | 4 | 9/12 |
| 125 | 2 | 1 | 9 | 7/12 |
| **All 36 pairs** | **9** | **5** | **22** | **22/36** |

Among the 22 pairs where the caption text changed, B was better in 9, A was
better in 5, and 8 remained tied. The direction is mildly favorable to B but
is dominated by ties and varies across seeds. It is not evidence of a human
preference advantage.

## Representative changes

| Video | Seed | Caption-only A | Caption+InfoNCE B | Audit |
|---|---:|---|---|---|
| Motorcycle crash | 123 | `a man is riding a horse` | `a man is riding a motorcycle` | B corrects the object, but still misses the crash |
| Boxing target | 124 | `a man is throwing a stone` | `a man is throwing a punch` | Clear B improvement |
| Sandwich preparation | 125 | `a woman is cutting a cucumber` | `a woman is cooking` | B removes a false object but becomes generic |
| Baby and mirror | 123 | `a baby is playing` | `a baby is sitting on a stool` | A is less wrong; B adds an unsupported object/context |
| Motorcycle crash | 125 | `a man is riding a motorcycle` | `a man is riding a horse` | B regresses, showing seed-level instability |
| Group dancing | 123 | `two boys are performing` | `two boys are playing` | A better preserves the event category |

These examples show that InfoNCE sometimes replaces a false detail with the
right category or a safer generic phrase, but it can also introduce a new
wrong object or action.

## Recurring error patterns

1. **Broad scene before precise event.** Cooking, shopping, dancing, exercise,
   and outdoor activity are often recognized while the defining action is
   missed. Examples include moonwalking with a cart becoming `shopping`, a
   motorcycle crash becoming `riding`, and pouring sauce becoming `cooking`.
2. **Object substitution.** Onion becomes carrot; motorcycle becomes horse;
   cup manipulation becomes eating potato/chicken; sandwich preparation
   becomes cutting cucumber or carrot.
3. **Action priors override temporal evidence.** The parrot picking up a cup is
   described as eating, a baby kissing a mirror as laughing/playing/crying, and
   a couple riding a bicycle as singing or performing.
4. **Fluency hides factual errors.** All 72 captions are readable, but many are
   incompatible with the visible event. Language fluency therefore cannot be
   treated as evidence of visual grounding.
5. **InfoNCE does not remove seed variance.** The same motorcycle video improves
   under seed 123 but regresses under seed 125.

## Paper interpretation

This audit does not overturn the formal test670 result. It refines its scope:

- Supported: fixed InfoNCE improves held-out retrieval and preserves aggregate
  caption metrics; the small audit is mildly favorable on context compatibility.
- Not supported: InfoNCE reliably improves fine-grained action/object grounding,
  human preference, or semantic accuracy.
- Recommended wording: report automatic caption gains as small and
  metric-dependent, then state that a bounded qualitative audit found mixed
  caption-level changes and persistent fine-grained hallucinations.
- Do not present these 12 videos as a quantitative human-study table. Two or
  three cases may be used as transparent examples if both improvements and
  regressions are shown.

No checkpoint, hyperparameter, decoding setting, or test prediction will be
changed in response to this audit.

## Integrity and artifacts

- Protocol commit: `762c479`.
- Blinded-ratings commit: `2f9cf85`.
- Ratings SHA-256:
  `fe264b61acae074af8c576a469c0c8fc204cfd4c3f7ac3603dd41c09f5d42f42`.
- Selection SHA-256:
  `485433e2a0e1340c5875439d55132e378517332fc064ce82fc4d3f40afa40b13`.
- Blinded manifest SHA-256:
  `966e69bc8dfda481e17cdfa7d701ae69392bf2cc8dc2977ba34da4b6658296b1`.
- Unblinding key SHA-256:
  `72716f00369b96f510cce823da83b76fc7aba6f8c14a650c646089be7548465f`.
- Summary SHA-256:
  `3a67f02ab751b870f65077cc1f595e0eb1e9c4c6e471f5a58a8d84259275e875`.
- `model_inference_performed=false`, `test_tuning_performed=false`, and
  `human_evaluation=false` are explicit in the summary artifact.
