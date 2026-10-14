# Historical training-chain audit (2026-09-03)

## Scope

This audit traces the data, Stage-3 language-model tuning, early alignment
checkpoints, mapper training, repaired recovery checkpoints, and the E0/E3-E6
models currently used by the semantic diagnostics. It is a provenance and
implementation audit only. No weights were changed and no new training run was
started.

Evidence classes used below:

- **Confirmed**: directly established by code, checkpoint metadata, tensor
  equality, hashes, or retained logs.
- **High-confidence inference**: the available artifacts strongly identify a
  path, but the original command or training log is missing.
- **Unknown**: cannot be recovered from the retained artifacts.

## Reconstructed lineage

```text
old MSVD annotations (203 train / 25 val / 26 test videos)
  |
  +-- 2025-10-10 ViT-text alignment runs
  |     +-- msvd_joint_stage1.pt
  |     +-- msvd_decoder_stage2.pt
  |     `-- not caption-model checkpoints; old caption loading was invalid
  |
  +-- 2025-10-11 caption-only GPT-2 tuning
  |     `-- gpt2_lm_stage3/{latest,best}
  |           `-- likely scripts/train_stage3_lm.py; command/log unavailable
  |
  +-- 2025-10-15 legacy caption mapper runs from base GPT-2
        `-- msvd_mapper_finetune.pt (legacy supervision)
              |
              +-- 2026-08-31 recovered v3
              |     visual path <- msvd_mapper_finetune.pt
              |     decoder     <- gpt2_lm_stage3/best
              |     repaired supervision, last four GPT-2 blocks trainable
              |
              +-- recovered v4 <- all v3 weights
              `-- recovered v5 <- all v4 weights
                    |
                    +-- 2026-09-01 full-data recovery A/B (separate branches)
                    `-- 2026-09-02 E3-E6 ablations from v5
```

## 1. Dataset provenance

### Confirmed

The retained old annotation files have not changed since 2025-10-01:

| Split | Videos | Caption entries | SHA-256 |
|---|---:|---:|---|
| old train | 203 | 8,189 | `0d8508a2f695ac71f70b6df09c80145f0f607ebb15ce0491f0b1eea3c94a906c` |
| old val | 25 | 940 | `4efa61ab44d6d02dd26baaa37008fb45c96377dba9aeb7cad9d5ed2beb8e8fb8` |
| old test | 26 | 988 | `623c0d76a86fbb91be1e3c888b4af810d736365cfee3c84b317354376073400d` |

The 2025 training logs show that video training retained 196/203 train videos
and 24/25 validation videos after missing-frame filtering. Caption-only Stage-3
training does not require frames and its default path reads all caption entries
from the 203 training records.

All 254 old video IDs belong to the current standard 1,200-video training
split. There is zero video-ID overlap between the old train/val/test files and
the current 100-video validation or 670-video test splits. The current val100
semantic diagnosis therefore has no video-ID leakage from the old training
subset.

The validated current data remains 1,200/100/670 videos, with pinned train and
validation annotation hashes already recorded by the full-data preparation
pipeline.

### Implication

The historical model was trained on a small subset of the standard training
videos, not on a conflicting test split. Small coverage is a serious capacity
and representativeness limitation, but data leakage is not the explanation for
the present validation behavior.

## 2. The October 2025 Stage-1/Stage-2 checkpoints

### Confirmed defect

`msvd_joint_stage1.pt`, `msvd_decoder_stage2.pt`, and
`msvd_caption_model.pt` each contain a ViT-text alignment state with prefixes
such as `vit`, `vid_proj`, `txt_emb`, `txt_enc`, and `txt_proj`. They do not
contain `VideoCaptionModel` keys such as `encoder`, `decoder.mapper`, or
`decoder.model`.

The retained 2025-10-11 generation log loaded `msvd_joint_stage1.pt` into a
caption model with `strict=False`. It recorded essentially the full caption
model as missing and the alignment state as unexpected, but continued to
generate. The 2025-10-12 comparison log shows the same checkpoint-type
mismatch for another early evaluation.

### Implication

Those early caption generations and comparisons did not evaluate the claimed
trained caption checkpoint. They must be quarantined from paper evidence.
This defect does not directly contaminate recovered v5: v5 follows the mapper
line below rather than loading these alignment checkpoints.

## 3. Stage-3 caption-only GPT-2

### High-confidence training route

`checkpoints/gpt2_lm_stage3` contains only `latest/` and `best/`. The two
`model.safetensors` files are byte-identical:

`a37dc482789e8ee4068deef308aa2793719849664054ce82db00f92efd489dcd`

This directory layout and identical final weights match
`scripts/train_stage3_lm.py`, which overwrites `latest` after every epoch and
saves the final model again as `best`. The alternative
`src/cli/train_decoder_only.py` saves the latest model in the root output
directory and selects `best` by validation loss; that root model is absent.

The exact executed command, epochs, learning rate, per-epoch losses, optimizer
state, and validation result were not retained. Therefore the script identity
is a high-confidence inference, not a fully auditable fact.

### Confirmed objective risks

Both surviving Stage-3 implementations tokenize raw captions without
explicitly appending EOS. Because GPT-2 EOS is also assigned as the pad token,
padding positions are masked from the loss. The language-model fine-tune thus
does not explicitly train caption termination.

For the likely script, `best` is simply the final epoch rather than a
validation-selected checkpoint. It also stores no training arguments,
dataset hash, optimizer state, or loss history.

### Implication

The Stage-3 checkpoint is runnable but not reproducible or model-selection
auditable. Its small-data fine-tune and missing termination supervision are
credible contributors to language-template bias. The audit alone does not
prove they caused the current collapse.

## 4. Mapper and recovered-v5 lineage

### Confirmed

The first retained mapper checkpoint, `msvd_mapper_finetune.pt`, was trained
from base `gpt2` on the old 196-video usable training subset. Its legacy
training implementation had the defects repaired on 2026-08-31:

- manual caption shifting followed by GPT-2's internal shift (double shift);
- EOS padding present in labels without the repaired first-EOS policy;
- a possible dynamically created adapter path instead of one fixed
  train/inference path;
- freezing the complete encoder also froze its randomly initialized 768-to-256
  projection.

Recovered v3 was then constructed with `resume_visual_only=true`: the visual
path came from `msvd_mapper_finetune.pt`, while GPT-2 was initialized from
`gpt2_lm_stage3/best`. Repaired supervision trained the visual path and the
last four GPT-2 blocks for 200 steps on the old data. v4 resumed all v3 weights
for 300 steps, and v5 resumed all v4 weights for another 300 steps, again on
the old data.

Tensor comparison confirms the decoder lineage. Relative to Stage-3 GPT-2,
recovered v3, v4, and v5 each retain 100/148 saved GPT-2 tensors exactly. The
48 changed tensors are precisely the 12 parameter tensors in each of blocks
8-11. Embeddings, blocks 0-7, and final layer norm remain byte-identical to
Stage-3.

### Implication

The repaired runs made the checkpoint technically interpretable, but v5 is
not a clean retrain. It retains most of the unvalidated Stage-3 decoder and a
visual initialization learned under the legacy supervision bugs. Continuing
from v5 cannot by itself distinguish a persistent initialization problem from
an objective or decoding problem.

## 5. What the later experiments do and do not clear

- The 2026-09-01 full-data recovery A/B proves that the full dataset and 4 GB
  training path work. Both branches still start from v5, so neither is a clean
  initialization control.
- E3-E6 also start from v5. GPT-2 remains frozen, so InfoNCE improvements do
  not test whether the inherited language model is the blocker.
- The val32 prefix matrix contradicts a fully disconnected visual input, but
  it does not clear the Stage-3 or legacy mapper initialization.

## Audit verdict

| Gate | Status | Verdict |
|---|---|---|
| Full current MSVD data | Pass | Valid, pinned, and readable |
| Old/new video split leakage | Pass | No old video enters current val/test |
| October Stage-1/Stage-2 caption evaluation | Fail | Wrong checkpoint architecture was loaded |
| Stage-3 data coverage | Fail for current claims | Only 203 training videos / 8,189 entries |
| Stage-3 reproducibility | Fail | Command, args, loss log, and optimizer state absent |
| Stage-3 model selection | Fail/high risk | Final epoch saved as `best` on the likely route |
| Stage-3 EOS supervision | Fail/high risk | No explicit caption EOS target |
| Legacy mapper supervision | Fail | Confirmed double shift and frozen projection |
| Recovered-v5 loading | Pass | Current strict diagnostics restore the expected architecture |
| Recovered-v5 clean provenance | Fail | Inherits Stage-3 and legacy mapper initialization |

The historical fine-tuning chain is now a priority causal candidate, not a
residual guess. It is still a hypothesis until tested against a clean control.

## Next controlled decision

Do not add more epochs to v5 and do not replace GPT-2 ad hoc. The smallest
causal experiment is a matched clean-initialization comparison:

1. **Clean control**: base GPT-2 plus newly initialized encoder projection and
   prefix mapper.
2. **Legacy control**: Stage-3 GPT-2 plus the same newly initialized projection
   and mapper.
3. Train both with the repaired supervision, identical full train1200 data,
   seed, batch size, optimizer, step budget, and decoding protocol.
4. Evaluate validation NLL, EOS behavior, prefix grounding delta, exact/semantic
   diversity, template concentration, and standard caption metrics.

This comparison isolates the Stage-3 initialization. A later comparison can
hold GPT-2 fixed and test fresh versus legacy visual initialization if needed.
Neither comparison has been run as part of this audit.
