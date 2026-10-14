# Fixed-caption versus multi-reference trial: results

Completed 2026-09-05 under the locked
`experiments/caption_policy_ab_protocol_20260905.md` protocol.

## Decision

**Proceed with one full-MSVD viability run using a logged, balanced
multi-reference schedule and substantially more than five visits per video.**

Do not repeat the historical policy of choosing an untracked random caption
once per video per epoch for only five epochs. The small causal trial shows
that target variation does not destroy visual binding when references are
covered evenly and the model receives enough visits.

This is permission to run the full experiment, not evidence that the full run
will pass validation and not a paper result.

## Matched setup

- Same sixteen MSVD training videos as the passed overfit16 check.
- A_fixed always saw one pinned canonical caption.
- B_multi cycled deterministically through every stored caption using one
  fixed per-video permutation. The selected videos contained 668 references.
- Both arms used original GPT-2, pretrained ViT-B/16, an identical fresh model
  state, and the same video-batch schedule.
- GPT-2 and the ViT backbone were frozen. Only `encoder.proj` and
  `decoder.mapper` were trained; no InfoNCE or historical mapper was used.
- Batch 8, seed 123, AdamW at 3e-4, exactly 500 steps / 250 visits per video.
- Frozen pooled ViT features were cached only after direct-video equivalence
  passed. The trainable 768-to-256 projection remained in the graph.
- At all 500 steps, video IDs, cached visual tensors, and pre-forward
  Python/CPU/CUDA RNG states matched. Caption targets differed at all 500
  steps, as intended.

## Final results

| Metric at step 500 | A_fixed | B_multi |
|---|---:|---:|
| Canonical-caption native NLL | **0.012578** | 1.474255 |
| Canonical shuffled - native NLL | **+7.490499** | +2.382524 |
| Canonical positive mismatch videos | 16/16 | 16/16 |
| Exact canonical generations | **16/16** | 1/16 |
| Exact match to any video reference | 16/16 | **11/16** |
| Mean best-reference token-F1 (triage only) | 1.0000 | 0.9365 |
| Unique native generations | 16/16 | 16/16 |
| EOS termination | 16/16 | 16/16 |
| Generation changed under shuffled prefix | 16/16 | 16/16 |
| **All-reference native NLL** | 5.490610 | **1.878495** |
| All-reference shuffled - native NLL | **+3.399178** | +2.512647 |
| All-reference positive mismatch videos | 16/16 | 16/16 |
| Preregistered long-run readiness gate | fail | **pass** |

The multi-reference arm lowered all-reference NLL by 3.612116 relative to the
fixed arm. It was lower on every one of the sixteen videos. Its grounding
penalty was smaller than the fixed arm's, but remained well above the locked
+1.00 threshold and was positive for all videos.

The fixed arm therefore demonstrates a real trade-off: it memorizes one
wording perfectly and binds that wording very strongly, but assigns poor
likelihood to the other valid descriptions of the same video. The balanced
multi-reference arm learns the broader caption distribution while retaining
video-specific conditioning.

One B_multi generation (`a dog is eating a dog`) is visibly unreliable even
though the aggregate diagnostics pass. No human semantic score was invented
to repair this case. The reported token-F1 is only a lexical triage statistic.

## B_multi trajectory on canonical probes

| Step | Native NLL | Shuffled - native NLL | Positive mismatch | Exact any reference | Unique | EOS |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 7.009716 | -0.040759 | 8/16 | 0/16 | 16/16 | 0/16 |
| 50 | 3.859935 | +0.266722 | 13/16 | 0/16 | 13/16 | 0/16 |
| 100 | 2.747804 | +0.668574 | 16/16 | 2/16 | 13/16 | 10/16 |
| 250 | 1.726684 | +1.980560 | 16/16 | 8/16 | 16/16 | 16/16 |
| 500 | 1.474255 | +2.382524 | 16/16 | 11/16 | 16/16 | 16/16 |

All 668 selected references were visited. Depending on the video's stored
reference count, each reference appeared four to eleven times.

## Integrity and engineering checks

- An initial smoke attempt stopped before training because the manually entered
  expected reference total was 717 instead of the actual 668. The failed
  preflight directory was retained. The number was corrected in the protocol,
  implementation, and a data-backed unit test before smoke v2.
- Smoke v2 and the formal run had identical step-zero endpoints for both arms.
- Frozen-backbone cache maximum absolute error was exactly 0 before training
  and after both arms.
- Both step-500 checkpoints loaded strictly and matched their recorded
  component hashes. GPT-2 and the ViT backbone matched the common initial
  state; both trainable components changed.
- Source and MSVD annotations remained unchanged during the run.
- Seventeen related unit tests passed after the experiment.
- Runtime excluding shared preflight/cache extraction: 208.44 seconds for
  A_fixed and 190.60 seconds for B_multi. These sequential timings are run
  records, not an efficiency comparison.
- Peak allocated CUDA memory was about 1.70 GiB per arm; the GPU was released
  after completion.

Artifact hashes:

| Artifact | SHA-256 |
|---|---|
| `runs/caption_policy_ab_20260905/result.json` | `1927c6a8bebd2fc167db79087689b8709d107cf45da8fe089c2a56868560abdd` |
| `runs/caption_policy_ab_20260905/preflight.json` | `d10dff2549170f339ae5220d23cf4d423447bcc55ac98dabc2f3ada7114ba941` |
| `runs/caption_policy_ab_20260905/common_initial.pt` | `bd4c7650b2eab2093d0b19d0ee099aef8a86a21ac1e44911ee51a609b3458602` |
| `runs/caption_policy_ab_20260905/schedule.json` | `a323b58e4e5e9ef76b395ef63afa4cd0f7937fa9104a8e6b5d08bdf8505a5b68` |
| A_fixed checkpoint | `7f1427a58f05d0bcbf37f0382656b6f56b311ccfb30dd15025808609ce342c69` |
| B_multi checkpoint | `dfc649cfc45991486472b997eb71d08ae9fb5edb21bf77d9479f5adf490530f6` |
| A_fixed trace | `e0670a0c2aa41f1fa40e35891d8a09fd6c4ccd2b436cdf57b5d36253037cd0c9` |
| B_multi trace | `425aff6f43124eed3f872050a75d55344fe649eb38a973b63dc237ab77829054` |

The formal run started from Git commit
`6931873676703965ae2165d1f6fa9247f1ecab27`.

## Proposed next experiment (not executed)

The full MSVD training set has 1,200 videos and 48,774 captions: 18 to 66
captions per video, mean 40.65. A clean viability run should therefore:

1. cache and hash frozen ViT features for train1200 and val100 only after
   direct-path equivalence checks;
2. use original GPT-2 and a completely fresh visual projection/mapper;
3. use a deterministic balanced caption permutation for each video;
4. run 66 epochs / 9,900 optimizer steps at batch 8, so every training
   reference is visited at least once;
5. use fixed validation checkpoints (for example epochs 5, 20, 40, and 66),
   with the final epoch as the preregistered primary endpoint rather than
   result-dependent early stopping;
6. evaluate val100 semantic diversity, all-reference likelihood, and the
   correct-versus-shuffled prefix matrix; keep test670 untouched until the
   validation decision is fixed;
7. add InfoNCE only after the caption-only baseline passes its semantic gate.

This full-run protocol still needs to be frozen and approved separately before
execution.

## Boundaries

- One seed and one deliberately small training subset.
- The arms test target-policy capacity, not validation generalization.
- Exact reference matches and token-F1 do not replace human semantic review or
  standard caption metrics.
- Passing does not resolve the earlier mixed original-GPT-2 versus Stage3
  initialization comparison; the proposed clean run deliberately avoids the
  unauditable historical initialization chain.
