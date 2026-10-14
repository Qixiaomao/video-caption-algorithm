# Sixteen-video overfit diagnostic: results

Completed 2026-09-04 under the locked
`experiments/overfit16_protocol_20260904.md` protocol.

## Outcome

The strict memorization gate **passed** at the fixed final step 500. Under this
small-set protocol, the repaired training path can learn a video-specific
mapping, carry it through the visual prefix, and use it during autoregressive
generation. This result contradicts the hypothesis that the current basic
visual-conditioning path is completely disconnected or intrinsically
untrainable.

This is a training-set capability check, not evidence of MSVD generalization
and not a paper result.

## Locked setup

- Sixteen fixed, semantically distinct MSVD training videos; one existing
  canonical caption per video.
- Original local GPT-2 and pretrained ViT-B/16.
- Frozen GPT-2 and ViT backbone; only the fresh `encoder.proj` and
  `decoder.mapper` were trained.
- Prefix length 4, batch size 8, 8 frames at 224x224, seed 123.
- AdamW, learning rate 3e-4, weight decay 0.01, no InfoNCE, AMP, scheduler,
  clipping, accumulation, early stopping, or cached visual features.
- Exactly 500 optimizer steps; evaluations were fixed in advance at steps
  0, 50, 100, 250, and 500.

## Trajectory

| Step | Native target NLL | Shuffled - native NLL | Positive mismatch videos | Exact targets | Unique outputs | EOS | Strict gate |
|---:|---:|---:|---:|---:|---:|---:|:---:|
| 0 | 7.009716 | -0.040759 | 8/16 | 0/16 | 16/16 | 0/16 | fail |
| 50 | 3.881448 | +0.444706 | 16/16 | 0/16 | 16/16 | 0/16 | fail |
| 100 | 2.213636 | +1.384108 | 16/16 | 0/16 | 16/16 | 0/16 | fail |
| 250 | 0.108462 | +6.333885 | 16/16 | 13/16 | 16/16 | 16/16 | fail |
| 500 | **0.019069** | **+7.837950** | **16/16** | **15/16** | **16/16** | **16/16** | **pass** |

At step 500, every shuffled visual prefix increased its target caption NLL,
and every generated caption changed when its prefix was replaced by the fixed
donor prefix. The largest native template occurred once (1/16), so the final
outputs were not a repeated generic caption.

The sole non-exact output was:

- video `UbmZAe5u5FI_36_40`
- target: `a woman is squeezing the liquid out of cucumber slices with her hands over a glass bowl`
- generated: `a woman is squeezing a cucumber slices with her hands over a glass bowl`

It is visibly related to the target, but no semantic score was substituted for
the preregistered exact-match rule.

## Integrity and resource checks

- The independent two-step smoke and formal run had identical step-zero
  metrics.
- All sixteen samples were seen exactly 250 times in the formal run.
- A fresh reconstruction reproduced the recorded initial component hashes.
- The final checkpoint loaded strictly. GPT-2 and the ViT backbone matched
  their initial/source states; both trainable visual modules changed.
- Full MSVD annotations, original GPT-2 source, and experiment source files
  remained unchanged during the run.
- Twelve related unit tests passed after the run.
- Runtime: 1,441.14 seconds (about 24.0 minutes).
- Peak CUDA memory: 1,714.63 MiB allocated and 1,940 MiB reserved on the local
  RTX 3050. The experiment completed on 4GB VRAM without OOM.

Artifact hashes:

| Artifact | SHA-256 |
|---|---|
| `runs/overfit16_20260904/result.json` | `ef20cc319d682bf6c46ea1890ce3d396812a2e3402f5abe39a6c07a5c4d87ef7` |
| `runs/overfit16_20260904/preflight.json` | `3bf7f97c369981b06db286c0e450da331ba518c4eccc9e819ad08d9196f3faf3` |
| `runs/overfit16_20260904/last.pt` | `dd9a1357dba8e3605d20e9ded7bebbff15981bb72506724c34b3fb6cbd840e4b` |
| `runs/overfit16_20260904/trace.jsonl` | `a30f8f2ca44a458ed9a29e20ea3324bab8c719318541b8d9edae0e3d38ed1916` |

The formal run started from Git commit
`7f00eff009453e78525a1df95683c07d2d7d7377`.

## What this changes in the diagnosis

The remaining full-data failure should no longer be described as "the visual
information is definitely cut off". A more accurate diagnosis is:

1. The repaired chain is mechanically able to learn and generate from
   video-specific prefixes.
2. The earlier full-MSVD five-epoch runs did not learn a sufficiently strong,
   generalizable mapping. In those runs, each of 1,200 videos was visited only
   once per epoch and one caption was sampled from a much larger reference set;
   the sixteen-video diagnostic instead repeated one stable target 250 times.
3. The unresolved causal categories are therefore full-data supervision
   variability, insufficient per-video optimization, frozen-decoder capacity,
   and objective/generalization design. Hardware is not the cause of the basic
   semantic-gate failure demonstrated here.

The next highest-information experiment is a strictly matched sixteen-video
fixed-caption versus multi-reference-caption comparison. It can isolate whether
caption-target variability itself destroys the clean binding before spending
hours on another full-MSVD run. That experiment is not included here and must
have its own locked protocol.

## Boundaries

- One seed, one selected subset, and one frozen-decoder configuration.
- Memorization does not establish validation performance or semantic quality.
- This result does not resolve the earlier mixed original-GPT-2 versus Stage3
  initialization comparison.
- No captioning benchmark metric from this run should appear as a model-quality
  result in the paper.
