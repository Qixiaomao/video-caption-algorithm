# Full-MSVD caption-only versus caption+InfoNCE ablation protocol

Frozen before the InfoNCE arm was executed on 2026-09-06.

## Question

On the repaired, full-reference caption training chain, does adding symmetric
in-batch InfoNCE improve the registered visual-prefix/WTE retrieval diagnostic
without destroying caption likelihood, semantic diversity, or visual-prefix
sensitivity?

This is a validation-side causal ablation. It is not a test-set benchmark and
does not by itself establish a paper contribution.

## Arms and the only intended intervention

| Arm | Caption loss | Alignment loss | Weight | Temperature |
|---|---|---|---:|---:|
| A_caption_only | causal LM cross-entropy | none | 0.00 | n/a |
| B_caption_infonce | causal LM cross-entropy | symmetric in-batch InfoNCE | 0.10 | 0.07 |

InfoNCE uses the normalized mean visual-prefix representation and the
normalized masked mean of frozen GPT-2 token embeddings. Exactly identical
caption token sequences inside a batch share a multi-positive mask; all other
in-batch pairs are negatives.

Lambda 0.10 is fixed because the earlier 150-step Stage-2 diagnostic gave it
the best text-to-video R@1 among 0.05, 0.10, and 0.20, and because 0.10 was the
original Stage-0 contract value. There is no weight or temperature sweep in
this experiment.

## Locked common conditions

- Original local GPT-2 Small and pretrained ViT-B/16.
- GPT-2 and the ViT backbone frozen. Only `encoder.proj` and
  `decoder.mapper` are trainable.
- No historical mapper, recovered-v5, Stage3, AMP, scheduler, gradient
  clipping, or early stopping.
- Full MSVD train1200 and val100. Test670 must not be read or scored.
- Deterministic balanced reference schedule covering all 48,774 train
  captions, 66 epochs, 9,900 steps, batch 8.
- AdamW, learning rate 3e-4, weight decay 0.01.
- Seed 123, batch seed 456, validation donor seed 321.
- Frozen raw ViT features and the exact batch/reference schedule are reused
  from the passed caption-only run.
- Caption validation uses all 4,290 references, equal weight per reference
  within a video and then equal weight per video.
- Fixed validation milestones: epochs 5, 20, 40, and 66. Epoch 66 is the sole
  primary comparison point.

## Registered caption-only source artifacts

The A arm is the completed run in `runs/full_msvd_balanced_20260905`. It is not
rerun. The B arm must strict-load the A arm's common initial state and must
match the A trace before every corresponding forward pass.

| Artifact | SHA-256 |
|---|---|
| `initial_model.pt` | `b0deacbea2239da7dd8f6d815be13f75afc962cce68b0a84771784e0db08b1ff` |
| `preflight.json` | `6292037e2a58fa3a6386bf182eef6df56e85e4e8374854f736c8467bf9b27652` |
| `schedule.json` | `3520204739b0bac71a6cea9e5a2f26c4dbfee17e3b19790e2340e9efc054fd86` |
| `train_features.pt` | `8cbc39e504be44865db302f87856b97b48318b6c32b60db7f8ed1ecb224a7ad6` |
| `val_features.pt` | `5ac1dc53d8e03fcdf3c3c8abae98e1a5c518774ffe48a00eacbb3648503aeee4` |
| `trace.jsonl` | `8b236e9e3a7f3fbe7ab6dff679326c810f600ae519f275a11271789a4de27a9b` |
| `result.json` | `80063ad11a750741735124b3f4bc4a81555a6f8eb9c612d89bee821279c7051f` |
| epoch-66 A checkpoint | `8155d68c5b3ac9f3f7396027a32b2e0ebea052a17aef79ecce8cf10a1e0f9e6a` |

Before formal execution, the implementation must independently recompute all
hashes. It must verify the registered A decision, configuration, split counts,
caption coverage, frozen-component hashes, and `paper_ready=false`.

## Trace equality

For each of the 9,900 B steps, the following pre-forward fields must equal the
corresponding A trace record exactly:

- step and epoch index;
- batch indices and video IDs;
- selected target texts and caption-token hash;
- cached visual-feature hash;
- Python, CPU Torch, and CUDA RNG-state hashes.

The alignment computation is deterministic and consumes no extra random
numbers. Therefore a trace mismatch invalidates the causal comparison and
must stop the run.

## Evaluation

At every milestone, run the same caption validation used by A:

- all-reference native NLL;
- fixed shuffled-prefix minus native NLL and 10,000-video bootstrap CI;
- number of positive mismatch videos;
- non-empty, unique, largest-template, caption-change, and EOS rates;
- exact-any-reference and best-reference token-F1 for triage only;
- unsmoothed BLEU-4 for continuity only, not as a paper metric.

Also evaluate the registered alignment representation on all 100 validation
videos and 4,290 captions:

- text-to-video R@1/R@5/R@10;
- video-to-text R@1/R@5/R@10;
- positive-minus-negative cosine similarity gap;
- a deterministic deranged-label retrieval control.

The retrieval derangement seed is fixed at 123.

After B training, load A's epoch-66 checkpoint strictly and compute the same
retrieval diagnostic in the same process. Reload B strictly afterward and
verify its recorded component hashes.

## Primary decision at epoch 66

B is a viable InfoNCE candidate only if every condition below holds:

1. B passes the existing five-part caption viability gate: non-empty 100%,
   unique >=30%, largest template <=25%, mismatch bootstrap lower bound >0,
   and at least 80/100 positive mismatch videos.
2. B native all-reference NLL is no more than A +0.15.
3. B mean best-reference token-F1 is no less than A -0.05.
4. B text-to-video R@1 is strictly greater than A text-to-video R@1.
5. B retrieval similarity gap is strictly greater than A's gap.
6. B true-label text-to-video R@1 is strictly greater than B's deranged-label
   control.

If all pass, the decision is `infonce_candidate_for_repeated_seeds`. If any
fail, the decision is `retain_caption_only_or_redesign_infonce`. No threshold
may be changed after results are visible.

## Smoke, integrity, and resource contract

- Run tests before smoke and after formal execution.
- Smoke uses the first two registered A trace steps, the corresponding cached
  train examples, and val8. Smoke results are engineering-only.
- Save milestone checkpoints, full B trace, per-step loss components, source
  and artifact hashes, component hashes, runtime, and peak CUDA memory.
- Refuse to overwrite an existing output directory.
- The final checkpoint must strict-load with no missing or unexpected keys.
- GPT-2 and ViT backbone hashes must remain equal to the common initial state;
  projection and mapper must change.
- `paper_ready` remains false. Test670 stays sealed until this ablation and any
  required repeated-seed validation are complete.
