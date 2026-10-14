# Q3 stage-1 novelty and method gate

Frozen on 2026-09-12 before any CPR implementation or training.

## Scope

This stage audits the proposed decoder-level visual-dependence loss and freezes
the next implementation boundary. It changes no model code, starts no GPU run,
and produces no model-quality result.

## Decision

The original `VPDR` new-method claim is rejected. Mismatched-pair caption
probability, discriminative captioning, retrieval-guided video captioning, and
token-level counterfactual regularization all have direct prior art.

If the comparison proceeds, the neutral name is **Conditional Prefix Ranking
(CPR)**. CPR is a prior-inspired experimental arm, not a claimed novel loss.

Primary prior art:

- Dai and Lin, *Contrastive Learning for Image Captioning*, NeurIPS 2017:
  https://papers.neurips.cc/paper/6691-contrastive-learning-for-image-captioning.pdf
- Vedantam et al., *Context-aware Captions from Context-agnostic Supervision*,
  CVPR 2017:
  https://openaccess.thecvf.com/content_cvpr_2017/papers/Vedantam_Context-Aware_Captions_From_CVPR_2017_paper.pdf
- Velda et al., *Improving Distinctiveness in Video Captioning with Text-video
  Similarity*, Image and Vision Computing 2023:
  https://www.sciencedirect.com/science/article/pii/S0262885623001026
- Cao et al., *See or Guess: Counterfactually Regularized Image Captioning*,
  ACM MM 2024: https://arxiv.org/abs/2408.16809

## Frozen candidate objective

For the content-token mask `M_i`, compute the length-normalized NLL of the same
caption under its matched prefix and under one deterministic in-batch deranged
prefix. The candidate loss is:

```text
gap_i = nll(caption_i | wrong_prefix_i) - nll(caption_i | matched_prefix_i)
cpr_i = max(0, margin - gap_i)
total = caption_ce + alignment_weight * alignment_loss + cpr_weight * mean(cpr_i)
```

Registered defaults:

- `margin=0.10` nats/token;
- `cpr_weight=0.10`;
- one deterministic in-batch deranged negative per eligible sample;
- reject same-video and exact-caption positives as negatives;
- skip a sample when no valid negative exists;
- run matched and mismatched decoder forwards sequentially;
- keep ViT and GPT-2 frozen;
- add no inference-time parameter or operation.

## Stage-2 implementation gate

No formal training is allowed until all conditions pass:

1. Existing `none` and `infonce` paths are bitwise/trace compatible with the
   current implementation when CPR weight is zero.
2. Unit tests cover satisfied and violated margins, invalid negatives, length
   normalization, and exact-caption exclusion.
3. CPR gradients reach only `encoder.proj` and `decoder.mapper` among the
   intended trainable/frozen components.
4. Checkpoint objective version records CPR margin, weight, and negative policy
   and rejects incompatible exact resumes.
5. A bounded batch-8 smoke is finite and does not OOM on the RTX 3050 4 GB.
6. All existing objective, semantic, checkpoint, and benchmark tests remain
   green.

## Formal experiment boundary after the gate

- Reuse completed A (caption CE) and B (caption CE + InfoNCE) runs.
- Add C (caption CE + CPR) and D (caption CE + InfoNCE + CPR).
- Use seeds 123, 124, and 125 with the existing paired schedule and initial
  states.
- Use val100 for the registered decision and keep test670 sealed until the
  complete repeated-seed decision is frozen.
- A positive result supports a controlled comparison of alignment levels; it
  does not convert CPR into a new-method claim.
