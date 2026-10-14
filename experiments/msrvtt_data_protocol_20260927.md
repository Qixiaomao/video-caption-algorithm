# MSR-VTT v1.0 data protocol and preflight (2026-09-27)

Status: **data gate FAIL**. This document records the current protocol and measured sample; it does not authorize formal training. Actual work began at 23:36 Beijing time, after the planned 18:00 checkpoint. No retrospective preregistration is claimed.

## Sources and version

- Original dataset description: [Xu et al., CVPR 2016](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/06/cvpr16.msr-vtt.tmei_.pdf). It defines 10,000 clips and the 6,513 train / 497 validation / 2,990 test partition.
- Downloaded archive index and annotation: [Frozen in Time official research repository](https://github.com/m-bain/frozen-in-time), which links `https://www.robots.ox.ac.uk/~maxbain/frozen-in-time/data/MSRVTT.zip`. Archive HTTP length 6,552,768,292 bytes; Last-Modified `Thu, 10 Jun 2021 12:21:32 GMT`. Archive is not fully downloaded.
- `MSR_VTT.json` self-identifies as version 1.0, 2016-04-14. SHA-256 `c6cb51bb725b76e78255f87c5fbb96886957ac53503a39bb03bff40b419cf996`.
- Annotation lists eight source-license categories, including noncommercial and no-derivatives Creative Commons variants; there is no verified per-video license mapping in the extracted annotation. The Frozen in Time repository's MIT license applies to its code, not automatically to the video clips. Use the archive only for the intended research protocol; do not redistribute media or claim a blanket dataset license without resolving original clip rights.

## Fixed split and tasks

Use the archive's `MSRVTT/structured-symlinks/{train,val,test}_list_full.txt` lists. SHA-256: train `7f35fa96d45e94eed97dda0c2df98342a58fca769e7ffab352bd59c69c933251`, val `be5b1994f643740b80796c392d7b472deed45ddce69ce13fee87609ce303ab3f`, test `e033df45731e025a1b7b4731ee3eddc400b6040c93fffd4ab3eb45a551f0526e`. They exactly match `video0`–`video6512`, `video6513`–`video7009`, and `video7010`–`video9999`, respectively; none have duplicate IDs. The archive also includes `videos/train_list_new.txt` with 7,010 train+val IDs. It is not the chosen train split.

Caption training uses all references per train video under deterministic per-video permutations. Validation uses 497 videos and all their references. The eventual frozen caption test uses 2,990 videos and all their references, `pycocoevalcap==1.2`, BLEU-1/2/3/4, METEOR, ROUGE-L and CIDEr, preserving raw scores and display scores times 100. Test remains a data-integrity set until the training and validation decision is frozen.

Retrieval uses **the captioning test2990 split**, not MSR-VTT 1K retrieval. Candidate videos: all 2,990. Text queries: all 59,794 captions; correct target is the originating video ID. Video-to-text ranks each video against the same 59,794 caption candidates, with any of its references accepted as positive as in the MSVD implementation. Report T2V and V2T R@1/5/10 as percentages, ranks, and the exact representation/ranking rule. Repeated exact caption strings across videos make some queries ambiguous; disclose this limitation and do not silently redefine positives after inspecting test results.

## Manifest and audit

`runs/msrvtt_preflight_20260927/manifest.json` has one row per ID: split, caption IDs and text plus caption SHA-256, archive member/CRC32/size, and nullable local media/frame/cache paths and hashes. A ZIP central-directory CRC32 is a source claim, not a verified downloaded-file hash. `data_audit.json` records 10,000 unique IDs; 199,994 captions; 9,994 videos with 20 captions and six with 19; 10,000 indexed media IDs; no ID duplicates or cross-split ID overlap. It also records 14,637 case-normalized exact caption duplicates within videos and 3,121 exact strings appearing in multiple splits. Shared text alone is not proof of shared video content. Full media-content duplicate and decode checks remain open.

Nine local video samples have SHA-256; six size-stratified videos were fully decoded and uniformly sampled to eight 224×224 JPEG frames. Four train/val samples passed frozen ViT feature extraction, with direct-vs-cached projection max absolute error 0 and peak reserved GPU memory 1,030 MiB. Test samples underwent media/frame integrity checks only; no test model output or metric was computed. See `resource_samples.json`, `frame_feature_probe.json`, and `resource_estimate.json` for raw measurements.

## Missing-data rule before formal runs

Do not silently drop any video or caption. For a missing/corrupt archive member, retry the same pinned source and verify its CRC32 plus local SHA-256, then decode and sample again. If any of the 10,000 clips cannot be repaired, the full standard-split experiment is blocked. A smaller subset would require a separately named supplementary protocol fixed before training and cannot count as completion of this full MSR-VTT replication. Do not decide exclusions from test predictions or scores. The final manifest must include SHA-256 for every local media file and every frame/cache artifact used in training or evaluation.

## Gate and next action

FAIL because 9/10,000 media are locally present, the full ZIP has not been downloaded or verified, only six videos have full-decode measurements, and full train/val/test frame coverage is absent. Fixed split and annotation/archive index consistency pass. Next: download the pinned archive to a new path with resume support, verify integrity, extract without overwriting existing files, run full per-video hash/decode/frame audit, then repeat resource estimates on the full pipeline. Formal A/B training starts only after this data gate and the separate engineering smoke gate pass.
