# Formal efficiency matrix protocol (frozen 2026-09-09)

Status: **preregistered; formal timing has not started**.

## 1. Question and scope

This experiment asks whether the repaired captioning checkpoint can reduce inference cost on the local RTX 3050 4 GB without changing its decoded captions. It does not re-estimate caption quality and it does not read the test670 split. Preflight timings are smoke diagnostics only and must not be used in the paper.

## 2. Frozen model and decoding

- Checkpoint: `runs/full_msvd_infonce_ablation_20260906/checkpoint_epoch066.pt`
- SHA-256: `891cdcceef2b52d09092dde00ef35f9e1ee652e686b30178b5c1a9aee81c7e7c`
- Required checkpoint contract: version 3, epoch 66, step 9900, prefix conditioning with four prefix tokens, frozen GPT-2 and frozen ViT backbone.
- Loader: `strict=True`; missing or unexpected keys invalidate the run.
- Decoder: empty prompt, greedy decoding, `max_new_tokens=24`, `min_new_tokens=0`, `num_beams=1`, `temperature=1.0`, `top_p=0.9`, `no_repeat_ngram_size=3`, `repetition_penalty=1.0`.
- Input preprocessing: eight uniformly selected RGB frames, resize to 224x224, `ToTensor`, ImageNet normalization.
- Determinism: seed 123, deterministic algorithms enabled, cuDNN benchmark disabled, TF32 disabled.

## 3. Frozen inputs

- Split: MSVD full validation (100 videos); test670 remains unread.
- Annotation file: `data/processed/msvd_full/val/annotations.json`
- Annotation SHA-256: `bcda3ba0583704a54297e4da4cc9acbcc5ec1ebffd6ed436a6ba3ee1b8a03c8b`
- Selection: choose the 16 records with the smallest SHA-256 of `efficiency-matrix-v1:<video_id>`, then order by that digest.
- The generated manifest records every selected video ID, absolute frame directory, eight frame paths, and every frame SHA-256. The preflight and formal runner must reject any mismatch.
- Batches are nested prefixes of the same ordered set. Primary batch sizes are 1, 2, 4, and 8. Batch size 16 is a capability boundary: OOM is recorded but does not invalidate batch sizes 1--8.

## 4. Frozen configurations

All settings not listed as enabled are disabled. CuPy pooling, CUDA Graph, residual-LayerNorm fusion, and the CuPy prefix projector are excluded from the primary matrix because they are either unsupported by the installed backbone or not part of the validated safe path.

| ID | FP16 | SDPA attention | channels-last | tanh GELU | in-place residual | compile | role |
|---|---:|---:|---:|---:|---:|---:|---|
| E0 | off | off | off | off | off | off | FP32 reference |
| E1 | on | off | off | off | off | off | precision ablation |
| E2 | on | on | off | off | off | off | attention ablation |
| E3 | on | on | on | off | off | off | memory-layout ablation |
| E4 | on | on | on | on | on | off | safe full stack |
| E5 | on | on | on | on | on | on | exploratory compile only |

E5 is not a required cell. If Triton/Inductor is unavailable, it is marked unavailable and excluded rather than silently falling back.

## 5. Step 2 preflight gates

Only batch sizes 1 and 8 are required; batch size 16 is attempted after them.

Required for E0--E4:

1. CUDA is available and the checkpoint/annotations/frame files match their frozen hashes.
2. Strict checkpoint load succeeds and the checkpoint contract matches.
3. Every requested optimization is reported active after a CUDA forward; no requested option silently falls back.
4. All batch-8 captions are non-empty and the first sample caption is invariant between batch sizes 1 and 8.
5. E1--E4 produce exactly the same 8 decoded captions as E0.
6. Encoder and visual-prefix values are finite. Against E0, minimum per-sample cosine similarity is at least 0.999 for both tensors.
7. Required batch sizes do not OOM and peak reserved CUDA memory stays below 3,800 MiB.

If caption equality fails, the full end-to-end formal matrix is blocked. Encoder-only profiling may later be proposed separately, but is not authorized by this protocol. E5 unavailability does not fail the primary preflight; an active E5 must satisfy the same semantic gates.

## 6. Step 3 formal matrix (not authorized yet)

After user approval only:

- Run E0--E4 at batch sizes 1, 2, 4, and 8; attempt batch size 16 as a boundary.
- Use three independent measurement blocks per cell, each with 10 untimed warm-ups and 50 timed iterations.
- Freeze the per-block configuration order using seed 123 and record it before the first timed iteration.
- Synchronize CUDA at every timing boundary. Report median and interquartile range across all measured iterations, with block-level results retained.
- Record preprocessing, ViT encoder, prefix/alignment, GPT-2 decode, end-to-end latency, samples/s, generated-token count, latency/token, peak allocated memory, and peak reserved memory.
- Reject a cell on hash/config mismatch, silent fallback, non-empty/equality failure, non-finite output, or required-batch OOM.
- Do not compare configurations using different inputs or decoding parameters. Do not replace unavailable cells after seeing results.

## 7. Interpretation limits

- The primary claim is local-device efficiency for this repaired checkpoint and fixed MSVD input protocol, not universal model acceleration.
- Tanh GELU is an approximation and must not be described as a fused kernel.
- In-place residual addition is an inference-only implementation change.
- Results from older checkpoints, synthetic inputs, or the legacy prompted/minimum-length decoder are not pooled with this matrix.
