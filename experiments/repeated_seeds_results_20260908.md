# Repeated-seed caption-only versus caption+InfoNCE: results

Completed 2026-09-08 under the locked
`experiments/repeated_seeds_protocol_20260908.md` protocol.

## Decision

**The fixed InfoNCE intervention replicated on both additional seeds and is
supported across all three completed seeds.**

For seeds 124 and 125, both caption-only A and caption+InfoNCE B completed all
66 epochs and 9,900 steps. Both B arms passed every preregistered decision
condition. Together with the earlier seed-123 causal comparison, text-to-video
R@1 improved in 3/3 seeds while caption viability was preserved.

This result justifies freezing the method for the next evaluation stage. It
does not yet make the experiment paper-ready: the registered retrieval measure
is an internal validation diagnostic aligned with the InfoNCE representation,
and the untouched MSVD test split has not been read.

## Locked comparison

Within each seed, A and B shared the same newly initialized model, frozen
ViT/GPT-2 weights, balanced reference schedule, video batches, target captions,
cached visual features, and pre-forward Python/CPU/CUDA RNG states. The only
intended intervention was:

- A: caption cross-entropy only;
- B: caption cross-entropy plus symmetric in-batch InfoNCE;
- alignment weight 0.10, temperature 0.07;
- exact-caption multi-positive masking.

Both arms used batch 8, AdamW at 3e-4, weight decay 0.01, 1,200 train videos,
all 48,774 train references, 100 validation videos, and all 4,290 validation
references. GPT-2 Small and the ViT-B/16 backbone remained frozen; only the
visual projection and prefix mapper were trained.

## Additional-seed primary results

All values below are from the preregistered primary endpoint, epoch 66.

| Seed | Arm | Caption gate | Native NLL | Token-F1 | T2V R@1 | T2V R@5 | T2V R@10 | V2T R@1 | Similarity gap |
|---:|:---:|:---:|---:|---:|---:|---:|---:|---:|---:|
| 124 | A | pass | 2.870012 | 0.687715 | 9.0443% | 29.8835% | 44.1492% | 11.0000% | 0.014169 |
| 124 | B | pass | 2.884483 | 0.702391 | **21.4685%** | **56.0373%** | **71.0256%** | **24.0000%** | **0.037728** |
| 125 | A | pass | 2.883416 | 0.701432 | 11.5851% | 34.8485% | 49.9301% | 12.0000% | 0.012548 |
| 125 | B | pass | 2.916360 | 0.702725 | **19.6503%** | **54.5921%** | **71.6317%** | **22.0000%** | **0.035704** |

Paired primary changes:

| Seed | Delta NLL | Delta token-F1 | Delta T2V R@1 | Delta T2V R@5 | Delta T2V R@10 | Delta V2T R@1 | Delta gap |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 124 | +0.014471 | +0.014676 | **+12.4242 pp** | **+26.1538 pp** | **+26.8765 pp** | **+13.0000 pp** | +0.023559 |
| 125 | +0.032944 | +0.001293 | **+8.0653 pp** | **+19.7436 pp** | **+21.7016 pp** | **+10.0000 pp** | +0.023156 |

For both seeds, B remained below the registered NLL ceiling of A + 0.15 and
above the registered token-F1 floor of A - 0.05. Both B arms also exceeded
their deranged-label controls: seed 124 scored 21.4685% versus 0.3497% R@1;
seed 125 scored 19.6503% versus 0.1865% R@1.

## Three-seed synthesis

The table includes the earlier seed-123 locked comparison and the two new
replications. Values are mean +/- sample standard deviation across seeds.

| Metric | Caption-only A | Caption+InfoNCE B | Paired B - A |
|---|---:|---:|---:|
| Native NLL | 2.877219 +/- 0.006759 | 2.891571 +/- 0.022114 | +0.014351 +/- 0.018652 |
| Mean best-reference token-F1 | 0.695591 +/- 0.007081 | **0.701461 +/- 0.001907** | +0.005871 +/- 0.007628 |
| Text-to-video R@1 | 11.1344% +/- 1.9052 | **19.8757% +/- 1.4930** | **+8.7413 +/- 3.3958 pp** |
| Text-to-video R@5 | 33.7296% +/- 3.4266 | **54.1803% +/- 2.0935** | **+20.4507 +/- 5.3846 pp** |
| Text-to-video R@10 | 49.2230% +/- 4.7598 | **70.3730% +/- 1.6828** | **+21.1500 +/- 6.0213 pp** |
| Video-to-text R@1 | 10.0000% +/- 2.6458 | **20.6667% +/- 4.1633** | **+10.6667 +/- 2.0817 pp** |
| Positive-minus-negative gap | 0.012650 +/- 0.001471 | **0.036159 +/- 0.001399** | +0.023509 +/- 0.000331 |

The paired T2V R@1 gains were +5.7343, +12.4242, and +8.0653 percentage
points for seeds 123, 124, and 125. The similarity-gap gains were +0.023812,
+0.023559, and +0.023156. Thus the direction replicated in every seed, and
the gap change was especially stable. With only three seeds, these descriptive
statistics should not be presented as a broad population estimate or as a
substitute for test-set benchmarking.

## Additional-seed trajectory

| Seed | Arm | Epoch 5 gate / R@1 | Epoch 20 gate / R@1 | Epoch 40 gate / R@1 | Epoch 66 gate / R@1 |
|---:|:---:|---:|---:|---:|---:|
| 124 | A | fail / 5.1748% | pass / 12.5874% | pass / 11.7016% | pass / 9.0443% |
| 124 | B | fail / 6.8531% | pass / 17.2261% | pass / 18.3450% | pass / 21.4685% |
| 125 | A | fail / 5.9441% | pass / 8.8345% | pass / 9.6037% | pass / 11.5851% |
| 125 | B | pass / 5.9207% | pass / 15.1748% | pass / 19.2774% | pass / 19.6503% |

The early epoch-5 gate failures in three arms resolved by epoch 20 without a
pipeline change. This is evidence that the early semantic-gate weakness was
primarily an undertraining state, not a 4GB-memory failure or proof that GPT-2
could not use the prefix. Caption-only eventually learned viable captions but
retained substantially weaker alignment than caption+InfoNCE.

## Preregistered decision audit

| Registered condition | Seed 124 | Seed 125 |
|---|:---:|:---:|
| Caption-only A passes the final caption gate | pass | pass |
| Caption+InfoNCE B passes the final caption gate | pass | pass |
| B native NLL <= A + 0.15 | pass | pass |
| B token-F1 >= A - 0.05 | pass | pass |
| B T2V R@1 > A | pass | pass |
| B similarity gap > A | pass | pass |
| B T2V R@1 > deranged-label control | pass | pass |
| Overall paired decision | **pass** | **pass** |

## Integrity and resource audit

- The formal process exited with code 0; `completion.json` records both seeds
  complete, both pair decisions passing, `test_split_read=false`, and
  `paper_ready=false`.
- Every arm completed 9,900 steps. Every one of 1,200 train videos was visited
  exactly 66 times, and each arm covered all 48,774 reference indices.
- Within each seed, A and B traces are byte-identical. Seed 124 uses trace
  `805aa6c...9210e`; seed 125 uses `899ffea...5ea0f`. The traces differ across
  seeds as registered.
- Frozen GPT-2 and ViT hashes are identical across seeds and unchanged after
  training. Projection and mapper initialization hashes differ between seeds.
- All four final checkpoints independently load as checkpoint version 3 with
  epoch 66, step 9,900, and `epoch_complete=true`. Their training contracts
  record A as `alignment_objective=none, weight=0.0` and B as
  `alignment_objective=infonce, weight=0.1`.
- Actual final-checkpoint SHA-256 values match their result records:
  seed124-A `8acc8160...62be5`, seed124-B `147330ab...067a7`, seed125-A
  `5e5d5cae...cee37`, and seed125-B `9e59425d...efe5`.
- Pinned source and data hashes remained unchanged. Direct versus cached frozen
  visual features had maximum absolute error 0.0 in both seed preflights.
- Formal source commit: `a24a048076fafefe06ff1e73628a7f84dd9d2af8`.
- The four arms took 2,579.61, 2,973.85, 2,654.29, and 2,954.13 seconds,
  respectively: 11,161.88 seconds total, or about 3 h 06 min.
- Maximum peak allocated CUDA memory was 1,695.4 MiB; maximum peak reserved
  memory was 1,800.0 MiB on the NVIDIA GeForce RTX 3050 4GB Laptop GPU.
- Repository-wide unit-test discovery passed 75/75 after the run.

Key artifact hashes:

| Artifact | SHA-256 |
|---|---|
| `completion.json` | `8a8a7d2426a547be9ead3fbdb44a6579a6316528dc4dfeab1e07832d8b940d9c` |
| `seed124/preflight.json` | `26353eb52997294e14b21053f9d454e268f41b220e7ea881f7b99225b05235ef` |
| `seed124/schedule.json` | `085efe6fe2e3cb0d5ed58ac01cd73804eaa89109572fa3de8d39d2894e93f670` |
| `seed124/A/result.json` | `a94a1e22e4884552daf1bb1404b4eaf9eef4860c8e60c7b6f5372cef4e0befdc` |
| `seed124/B/result.json` | `b57e76ea10bd28d5dd2d059a55003a329f37dbc526914089288a8cb089cb582c` |
| `seed125/preflight.json` | `b6ff173451d8acc6fcdf12d5cf8cffd3988af89fb6393bcd1ff5062eae7f5610` |
| `seed125/schedule.json` | `94c26d811594eaae11a32c68ec31c7eb8180edcaf7a7ca03ce0d7db64cbf35e7` |
| `seed125/A/result.json` | `41ead69eba750d233210ff6df0e2df2a474a3bffe09e73b448ef5fd348278f46` |
| `seed125/B/result.json` | `ebc566e5bc94d63a3f3f283281c6be3da1300e253630f2eb20c8d6c50bcbd70e` |

## Interpretation and next decision

The narrow causal conclusion is now replicated: on the repaired full-caption
training chain, adding the fixed InfoNCE objective substantially improves the
registered visual-text alignment and retrieval diagnostic without materially
damaging caption validation behaviour. The earlier failure was not a reason to
replace GPT-2; complete caption supervision and sufficient optimization were
necessary for the semantic gate, while InfoNCE addressed the remaining
alignment weakness.

The next justified stage is to freeze this method and prepare the three-seed
synthesis for the untouched standard MSVD test evaluation. That stage should
report official caption metrics and standard retrieval metrics if the pipeline
supports them, without tuning on test670. No new lambda/temperature sweep is
justified by the present result.

## Boundaries

- Three seeds, one fixed InfoNCE setting, and one model architecture.
- Validation-only causal evidence; test670 remains untouched.
- The retrieval representation is the one optimized by InfoNCE, so these
  recalls are diagnostic rather than a final paper benchmark.
- Token-F1 and unsmoothed triage BLEU do not replace official caption metrics
  or human semantic evaluation.
- `paper_ready` remains false until the user reviews the code and report, the
  method is frozen, and the held-out paper evaluation is completed.
