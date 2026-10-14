# Full-MSVD caption-only versus caption+InfoNCE ablation: results

Completed 2026-09-06 under the locked
`experiments/full_msvd_infonce_ablation_protocol_20260906.md` protocol.

## Decision

**InfoNCE passes the preregistered ablation decision and is a candidate for
repeated-seed validation.**

At the sole primary endpoint, epoch 66, caption+InfoNCE passed all six locked
criteria. Relative to the hash-locked caption-only arm, it substantially
improved the registered prefix/WTE retrieval diagnostic while preserving
caption likelihood, visual-prefix mismatch sensitivity, diversity, and mean
best-reference token-F1.

The formal decision is `infonce_candidate_for_repeated_seeds`. This is not yet
permission to claim a paper result or open test670. The effect must first be
checked across additional seeds, and the retrieval representation remains a
diagnostic aligned with the training objective rather than a standard MSVD
test benchmark.

## Causal comparison

The caption-only arm was the completed full-MSVD run from 2026-09-05. It was
not rerun. The InfoNCE arm strict-loaded its exact common `initial_model.pt`,
reused its frozen ViT feature caches and schedule, and matched its trace before
every one of 9,900 forward passes.

The following fields matched at every step: epoch, batch indices, video IDs,
target captions, token hash, cached visual-feature hash, Python RNG state,
Torch CPU RNG state, and CUDA RNG state. The two trace files are byte-for-byte
identical and have the same SHA-256.

The only intended intervention was:

- symmetric in-batch InfoNCE;
- alignment weight 0.10;
- temperature 0.07;
- exact-caption multi-positive masking.

Both arms used original GPT-2 Small, pretrained ViT-B/16, full balanced
caption coverage, batch 8, AdamW at 3e-4, weight decay 0.01, 66 epochs, and
9,900 steps. GPT-2 and the ViT backbone remained frozen; only
`encoder.proj` and `decoder.mapper` were trained. Test670 was not read.

## Primary decision audit

| Registered condition | Caption-only A | Caption+InfoNCE B | Pass |
|---|---:|---:|:---:|
| B passes five-part caption viability gate | pass | pass | yes |
| Native NLL <= A +0.15 | 2.878229 | 2.873869 | yes |
| Mean token-F1 >= A -0.05 | 0.697626 | 0.699268 | yes |
| Text-to-video R@1 strictly improves | 12.7739% | 18.5082% | yes |
| Retrieval similarity gap strictly improves | 0.011232 | 0.035044 | yes |
| B text-to-video R@1 beats deranged control | n/a | 18.5082% vs 0.2098% | yes |

All thresholds and the single InfoNCE setting were committed before smoke or
formal results were observed. No early stopping, checkpoint selection,
lambda sweep, or post-result threshold change was performed.

## Caption results at epoch 66

| Metric | Caption-only A | Caption+InfoNCE B | B - A |
|---|---:|---:|---:|
| Native all-reference NLL | 2.878229 | **2.873869** | -0.004360 |
| Shuffled - native NLL | +1.343572 | **+1.355657** | +0.012085 |
| Mismatch bootstrap 95% CI | [+1.197024, +1.493970] | [+1.207542, +1.506155] | — |
| Positive mismatch videos | 99/100 | **100/100** | +1 |
| Non-empty generations | 100/100 | 100/100 | 0 |
| Unique generations | **89/100** | 85/100 | -4 |
| Largest template | **4/100** | 5/100 | +1 |
| Caption changes under shuffled prefix | **100/100** | 99/100 | -1 |
| EOS termination | 100/100 | 100/100 | 0 |
| Exact match to any reference | 14/100 | **16/100** | +2 |
| Mean best-reference token-F1 | 0.697626 | **0.699268** | +0.001642 |
| Unsmoothed triage BLEU-4 | 47.2442 | 47.7789 | +0.5346 |

Both arms pass the same semantic and grounding gate. InfoNCE does not produce
a material aggregate caption-quality gain, but it also does not incur the
predeclared degradation. Its reliable benefit in this experiment is on the
alignment/retrieval side.

## Retrieval diagnostic at epoch 66

The diagnostic uses all 100 validation videos and 4,290 captions. Video
representations are normalized means of visual-prefix tokens; text
representations are normalized masked means of frozen GPT-2 token embeddings.

| Metric | Caption-only A | Caption+InfoNCE B | Absolute change |
|---|---:|---:|---:|
| Text-to-video R@1 | 12.7739% | **18.5082%** | +5.7343 pp |
| Text-to-video R@5 | 36.4569% | **51.9114%** | +15.4545 pp |
| Text-to-video R@10 | 53.5897% | **68.4615%** | +14.8718 pp |
| Video-to-text R@1 | 7.0000% | **16.0000%** | +9.0000 pp |
| Video-to-text R@5 | 21.0000% | **35.0000%** | +14.0000 pp |
| Video-to-text R@10 | 27.0000% | **50.0000%** | +23.0000 pp |
| Positive-minus-negative similarity gap | 0.011232 | **0.035044** | 3.12x A |

B's deranged-label text-to-video recall was 0.2098%, 2.0280%, and 4.7086% at
R@1, R@5, and R@10 respectively. The true-label scores are therefore not
explained by a label-independent ranking artifact.

## InfoNCE-arm trajectory

| Epoch | Caption NLL | Shuffled - native | CI lower | Positive videos | Unique | Largest template | Token-F1 | T2V R@1 | Retrieval gap | Caption gate |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| 5 | 3.0207 | +0.6602 | +0.5737 | 94/100 | 35% | 12% | 0.6733 | 9.1142% | 0.0109 | pass |
| 20 | 2.8549 | +1.2054 | +1.0769 | 98/100 | 75% | 4% | 0.6906 | 15.8275% | 0.0215 | pass |
| 40 | 2.8191 | +1.3262 | +1.1903 | 100/100 | 85% | 4% | 0.6804 | 16.6667% | 0.0292 | pass |
| 66 | 2.8739 | +1.3557 | +1.2075 | 100/100 | 85% | 5% | 0.6993 | 18.5082% | 0.0350 | **pass** |

Unlike the earlier 150-step historical InfoNCE run, the repaired full-caption
chain already passes the caption gate by epoch 5 and remains valid through
epoch 66. Retrieval R@1 and the validation similarity gap improve throughout
the fixed milestones.

## Qualitative boundary

InfoNCE changed 37 of 100 native greedy generations relative to caption-only.
Using best-reference token-F1 as a lexical triage statistic, 12 videos
improved, 14 worsened, and 74 were unchanged. The aggregate mean is nearly
identical.

Some changes look plausible, for example `a man is walking on a snow slope`
to `a man is walking`; others trade one questionable caption for another,
such as `a man is cooking a pan` to `a man is stirring a pan`. B still emits
awkward or incorrect outputs including `a man is lifting a man` and
`a cat is playing with a cat`. No human semantic score was invented to resolve
these cases.

## Integrity and engineering checks

- Formal process exited with code 0.
- The InfoNCE trace matched all 9,900 caption-only trace records and is
  byte-identical to the A trace.
- Every train video was visited exactly 66 times and every one of the 48,774
  training references was covered by the reused balanced schedule.
- The final checkpoint independently loaded with `strict=True` with zero
  missing or unexpected keys. Recomputed component hashes matched the result.
- GPT-2 and the ViT backbone matched the common initial hashes. The visual
  projection and prefix mapper both changed.
- Checkpoint version 3 recorded a completed epoch 66 and step 9,900.
- Pinned train/validation annotations and experiment source were unchanged
  during the formal run.
- Formal source Git commit:
  `c06e2fb28f5e52fe9abb7c11fbe9fa5245f12a77`.
- Training plus milestone evaluation took 2,867.90 seconds (47.80 minutes).
  Peak allocated CUDA memory was 1,694.8 MiB and peak reserved memory was
  1,790.0 MiB on the RTX 3050 4GB Laptop GPU.
- After the experiment, repository-wide test discovery initially found one
  pre-existing/current-Transformers hook-cleanup failure in
  `probe_layers`. Replacing `output_hidden_states=True` with an explicit
  embedding-boundary hook fixed the leak without changing experiment output.
  The isolated test and the full suite then passed: 73/73. The fix is separate
  commit `cc8ae26` and postdates the formal source commit.

Artifact hashes:

| Artifact | SHA-256 |
|---|---|
| `runs/full_msvd_infonce_ablation_20260906/preflight.json` | `5052cc1a7e18f7de9700e24f5d6f2784a468b19d730d262a67ff47a562702c9d` |
| `runs/full_msvd_infonce_ablation_20260906/result.json` | `4a28f87ee21436dd6fb12fc068c14d57ac8cee9adc5d75dadcfd0e4b2784a587` |
| `runs/full_msvd_infonce_ablation_20260906/trace.jsonl` | `8b236e9e3a7f3fbe7ab6dff679326c810f600ae519f275a11271789a4de27a9b` |
| `runs/full_msvd_infonce_ablation_20260906/train_metrics.jsonl` | `9d7b749376d8f720d64434c7b948fb7fbe6b693689d3f76cc2b5cddbbf02c2b9` |
| Epoch-5 checkpoint | `fe5423eaab375b85816341477e502df19862829ea619a75664940d03a51366ad` |
| Epoch-20 checkpoint | `26ab3daa82f246e90d444e5214fbac85fea97a78576570c9dd237c8db3dd980b` |
| Epoch-40 checkpoint | `81080e54c5e4a8b9b72fa5e81e797923032b66e55dd8404d596118f20f981499` |
| Epoch-66 checkpoint | `891cdcceef2b52d09092dde00ef35f9e1ee652e686b30178b5c1a9aee81c7e7c` |

## Interpretation and next decision

The experiment supports a narrow causal conclusion: on this repaired
caption-training chain and this seed, adding `lambda=0.10` InfoNCE improves the
registered visual-text alignment/retrieval diagnostic without materially
hurting caption validation behaviour. It also confirms that negatives become
useful once the caption supervision and reference coverage are healthy.

It does not show that InfoNCE improves human semantic caption quality, nor
does it validate a broad hyperparameter claim. The next justified investment
is a small repeated-seed confirmation of A versus B, not a GPT-2 replacement
or an immediate test-set run. If the retrieval gain and caption non-inferiority
replicate, freeze the selected method and then run the untouched standard
MSVD test evaluation with paper metrics.

## Boundaries

- One seed and one fixed InfoNCE setting.
- Validation-only causal result; test670 was not read.
- A was reused from a hash-locked prior run rather than recomputed concurrently.
  Byte-identical trace replay and a common initial state make the comparison
  strict, but an independent repeated run is still valuable.
- The retrieval representation is the representation optimized by InfoNCE;
  its recall values are diagnostic and not yet the paper's standard benchmark.
- Triage BLEU, token-F1, and exact-reference matches do not replace official
  caption metrics or human inspection.
- `paper_ready` remains false. The user must verify the code, report, and
  numerical claims, and later follow the target venue's AI-disclosure policy.
