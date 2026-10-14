# Matched GPT-2 initialization trial: preregistered protocol

Locked before observing training results, 2026-09-03.

## Question and intervention

Does the retained Stage3 decoder initialization harm caption learning compared
with original GPT-2, when the decoder is frozen and every visual/training input
is matched? A uses the local original GPT-2 snapshot; B uses Stage3 `best`.
This is a conditional finite-budget, one-seed initialization test, not a test
of GPT-2 architecture adequacy or a proof about full decoder fine-tuning.

## Fixed controls

- Same pretrained ViT-B/16; one shared newly initialized 768-to-256 projection
  and four-token mapper, saved to a common bundle and matched by tensor hash.
- No weights from legacy mapper or recovered v5; no InfoNCE.
- Entire GPT-2 and ViT backbone frozen. Only encoder projection and mapper
  trainable. Same dropout mode and probability in both arms.
- Full pinned MSVD train1200 and val100; no test670 scoring.
- Seed123, batch8, 8 frames, 224x224, 64 caption tokens. All 48,774 training
  captions fit with EOS (longest 52 tokens); no caption truncation. Both
  tokenizers must produce identical IDs for every train/validation reference.
- Five epochs / 750 AdamW steps per arm, constant LR3e-4, weight decay0.01,
  no gradient accumulation, AMP, clipping, scheduler, or early stopping.
- Full caption sequence, GPT-2 internal shift, first EOS supervised and later
  padding ignored. Each video contributes one randomly selected caption/epoch.
- Fresh optimizer; reset Python/CPU/CUDA RNG after model initialization. Local
  DataLoader generators match. Deterministic algorithms enabled, TF32 disabled.
- Every step records video IDs, caption IDs hash, input-frame tensor hash, and
  pre-forward Python/CPU/CUDA RNG hashes. All 750 steps must match exactly.
- Frozen decoder/backbone hashes must be unchanged after training; both visual
  trainable modules must update. Data and source files must remain unchanged.
- Smoke: a separate two-step run per arm with the same physical shapes. Smoke
  weights never initialize the formal trial. Any failure is fixed before the
  formal run; no result-dependent changes to the formal budget.

## Endpoints and decision rules

Use each arm's **last step750** checkpoint, not separately selected best steps.

1. Free-running native captions on all val100: nonempty rate, unique exact
   captions, largest exact template, top templates, triage BLEU. The existing
   gate requires nonempty100%, unique>=30%, largest template<=25%; passing it
   is necessary but does not establish semantic accuracy or paper readiness.
2. Core prefix matrix on val100/all4,290 references: native, repeat, shuffled,
   zero; same donors/seed and greedy decode (empty prompt, max24, min0,
   no-repeat-ngram3, repetition penalty1). Video-balanced NLL and shuffled
   minus native NLL, also excluding first token. Native EOS-token NLL is
   recoverable from the saved per-reference final token losses.
3. Compare B-minus-A per-video native NLL and grounding deltas with 10,000
   paired video bootstrap samples. These are descriptive fixed-seed/fixed-donor
   intervals, not training-seed uncertainty or definitive significance tests.
4. If B is worse on both usable generation and grounding/likelihood, this
   supports a harmful Stage3 initialization under this protocol. If B is better,
   it contradicts a blanket claim that Stage3 must be the blocker. If mixed,
   retain the ambiguity. If both fail the gate, neither initialization is fixed;
   budget, objectives, visual learning, or decoding may still limit results.

Do not infer which historical hyperparameter/EOS/model-selection choice caused
an effect: the treatment is the entire Stage3 weight state. Do not equate exact
string variety with semantic quality. All outputs remain `paper_ready=false`;
formal caption metrics and human grounding checks belong to a later evaluation.
