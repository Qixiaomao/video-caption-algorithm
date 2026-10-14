# Full-MSVD balanced-caption viability run: results

Completed 2026-09-05 under the locked
`experiments/full_msvd_balanced_protocol_20260905.md` protocol.

## Decision

**Proceed to the preregistered InfoNCE ablation.**

The epoch-66 primary endpoint passed all five locked viability criteria. This
is the first full-MSVD result in the repaired chain showing that the fresh
visual projection and prefix mapper can learn video-dependent conditioning
without historical weights or InfoNCE.

This result rejects the immediate hypothesis that original GPT-2, the basic
prefix interface, or the 4 GB GPU prevents the caption chain from learning.
It instead supports insufficient and poorly auditable historical caption
training/target coverage as a major cause of the earlier template collapse.
It does not prove that every historical failure has the same cause.

Passing is permission to run a matched InfoNCE ablation. It is not a paper
benchmark result, and the test split remains untouched.

## Locked setup

- Original local GPT-2 Small and pretrained ViT-B/16.
- Fresh `encoder.proj` and `decoder.mapper`; no historical weights, Stage3
  weights, or InfoNCE.
- GPT-2 and the ViT backbone were frozen. Only the 768-to-256 visual
  projection and prefix mapper were trained.
- Full MSVD train1200 and val100. The test670 split was not read or scored.
- All 48,774 train references and all 4,290 validation references were used.
- Deterministic balanced per-video caption permutations, 66 epochs, 9,900
  optimizer steps, batch 8, AdamW at 3e-4, weight decay 0.01.
- Each training video was visited exactly 66 times. Every stored training
  reference was visited at least once; individual references were visited one
  to four times depending on the video's reference count.
- Validation was fixed in advance at epochs 5, 20, 40, and 66. Epoch 66 was
  the only primary endpoint; no early stopping or result-driven parameter
  change was allowed.
- Grounding was measured on val100 using all references: NLL was averaged over
  references within a video and then equally over videos. A fixed derangement
  supplied the shuffled visual prefix. The 95% interval used 10,000
  video-level bootstrap resamples.

## Primary epoch-66 result

| Locked criterion | Threshold | Result | Pass |
|---|---:|---:|:---:|
| Non-empty generation | 100% | 100/100 | yes |
| Unique generation rate | >=30% | 89/100 (89%) | yes |
| Largest template share | <=25% | 4/100 (4%) | yes |
| Shuffled - native NLL bootstrap lower bound | >0 | +1.1970 | yes |
| Videos with positive mismatch delta | >=80/100 | 99/100 | yes |

Additional epoch-66 diagnostics:

- Native all-reference NLL: 2.878230.
- Shuffled minus native NLL: +1.343572; 95% bootstrap CI
  [+1.197024, +1.493970].
- Generations changed under the shuffled prefix: 100/100.
- EOS termination: 100/100.
- Exact match to any stored reference: 14/100.
- Mean best-reference token-F1: 0.6976 (triage only).
- Unsmoothed BLEU-4: 47.2442 (triage only, not a paper metric).
- Most frequent native generation: `a man is playing a guitar`, four cases.

## Validation trajectory

| Epoch | Native NLL | Shuffled - native | 95% CI | Positive videos | Unique | Largest template | Token-F1 | Primary gate |
|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| 5 | 3.0751 | +0.5568 | [+0.4722, +0.6491] | 93/100 | 23% | 23% | 0.6430 | fail |
| 20 | 2.8490 | +1.1625 | [+1.0358, +1.2920] | 98/100 | 75% | 5% | 0.6930 | pass |
| 40 | 2.8147 | +1.3067 | [+1.1682, +1.4474] | 98/100 | 87% | 5% | 0.6746 | pass |
| 66 | 2.8782 | +1.3436 | [+1.1970, +1.4940] | 99/100 | 89% | 4% | 0.6976 | **pass** |

At epoch 5, likelihood grounding was already positive but generation
diversity had not yet crossed the locked 30% threshold. By epoch 20, all five
criteria passed, and the pass remained stable at epochs 40 and 66. This
trajectory is consistent with the earlier five-epoch historical training
budget being too short for the balanced multi-reference objective.

The final native NLL is slightly worse than epoch 40 while grounding and
diversity are stronger. Because epoch 66 was preregistered as the sole primary
endpoint, no best-checkpoint selection was performed after seeing results.

## Integrity and engineering checks

- Formal process exited with code 0 and wrote decision
  `proceed_to_infonce_ablation`.
- All 1,200 train videos had exactly 66 visits. All 48,774 reference indices
  were covered; per-reference visit count ranged from one to four.
- Frozen ViT feature-cache equivalence maximum absolute error was exactly 0
  before training for train and validation, and exactly 0 after training.
- Source and pinned train/validation annotations were unchanged during the
  run. The MSVD manifest was valid and pinned to source revision
  `9aa95a3039c106ee9e0969d82d7ce5a9efab1196`.
- The epoch-66 checkpoint independently loaded with `strict=True` with zero
  missing and unexpected keys. Recomputed component hashes matched the result
  record.
- GPT-2 and the ViT backbone matched the initial hashes. Both trainable
  components changed.
- Checkpoint version was 3 and recorded completed progress at epoch 66,
  step 9,900.
- Twenty-two related regression tests passed after the run.
- Training plus milestone evaluation took 2,904.37 seconds (48.41 minutes),
  excluding initial video decoding/cache construction. Peak allocated CUDA
  memory was 1,677.9 MiB and peak reserved memory was 1,940.0 MiB on the
  NVIDIA GeForce RTX 3050 4GB Laptop GPU.
- Formal source Git commit:
  `98707885e12b434229128d2e8f5de2033a674967`.

Artifact hashes:

| Artifact | SHA-256 |
|---|---|
| `runs/full_msvd_balanced_20260905/preflight.json` | `6292037e2a58fa3a6386bf182eef6df56e85e4e8374854f736c8467bf9b27652` |
| `runs/full_msvd_balanced_20260905/result.json` | `80063ad11a750741735124b3f4bc4a81555a6f8eb9c612d89bee821279c7051f` |
| `runs/full_msvd_balanced_20260905/schedule.json` | `3520204739b0bac71a6cea9e5a2f26c4dbfee17e3b19790e2340e9efc054fd86` |
| `runs/full_msvd_balanced_20260905/trace.jsonl` | `8b236e9e3a7f3fbe7ab6dff679326c810f600ae519f275a11271789a4de27a9b` |
| `runs/full_msvd_balanced_20260905/initial_model.pt` | `b0deacbea2239da7dd8f6d815be13f75afc962cce68b0a84771784e0db08b1ff` |
| Epoch-5 checkpoint | `f879e6c0c51b4c42d9ead5d8e8c49a3c627952ad633dace8c5a2e9a21fd6d300` |
| Epoch-20 checkpoint | `6762e192431de506a36fa54d3b917fe1eb21136e85aabf5f11816997c7716313` |
| Epoch-40 checkpoint | `7248550183f15c9a6b95df14599ce0659654ebd9666ad0748c262e28965241c9` |
| Epoch-66 checkpoint | `8155d68c5b3ac9f3f7396027a32b2e0ebea052a17aef79ecce8cf10a1e0f9e6a` |

## Interpretation for the project

The core localization question is now materially narrower. Visual information
does reach the trainable projection/mapper and can control original GPT-2 on
unseen validation videos when caption targets are balanced and the training
budget covers the full reference set. Replacing GPT-2 is therefore not the
next justified move.

The evidence points to the historical training chain—especially five-epoch
training and untracked/random multi-reference target selection—rather than a
fundamental architecture or hardware limit. The next experiment should test
whether InfoNCE improves, leaves unchanged, or damages this now-working
caption-only baseline under a matched training contract.

## Boundaries

- One seed; GPT-2 and the ViT backbone remained frozen.
- Validation-only viability result. The test670 split was not read.
- The fixed donor assignment is bootstrapped over videos; its confidence
  interval does not include uncertainty from rematching donors.
- Token-F1, exact-reference matches, and unsmoothed BLEU are diagnostic only.
  Standard caption metrics and human semantic inspection are still pending.
- Some outputs remain visibly wrong or awkward, including
  `a man is holding a man`, `a man is lifting a man`, and
  `a man is cooking a pan`. The aggregate pass does not hide these failures.
- `paper_ready` remains false. Passing this gate does not establish an InfoNCE
  benefit, a publishable benchmark, or an efficiency/quality trade-off.

## Next experiment (not executed)

Freeze a matched caption-only versus caption-plus-InfoNCE ablation before
running either arm. Reuse the same data split, balanced reference schedule,
initial state, batch order, training budget, validation donors, and reporting
contract. Change only the alignment objective/weight, and keep test670 sealed
until the validation-side ablation decision is fixed.
