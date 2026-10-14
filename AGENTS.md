# Repository instructions

This repository studies ViT-B/16 + GPT-2 Small video captioning and retrieval
on a single RTX 3050 with 4 GiB VRAM. Read `reproducibility/README.md` before
changing experiments; public numeric evidence is in `reproducibility/evidence/`.

- Use Python 3.11 and one environment for the whole workflow.
- Keep frozen protocols, seeds, split membership, decoding and endpoint selection
  separate from new experiments. Give new runs fresh output directories.
- Preserve failed runs and negative results. Never overwrite completed evidence
  or change a hash check merely to make an old result pass.
- Training/validation must not read test quality metrics. Test never selects a
  seed, checkpoint, threshold, objective weight or decoding policy.
- A/B use seeds 123/124/125. Compute B-A within seed before aggregating;
  report sample SD with n-1. Three seeds do not establish significance.
- MSVD test670 and MSR-VTT test2990 are same-set retrieval protocols.
  MSR-VTT test2990 is not standard 1K retrieval.
- Caption scores are raw COCO scores times 100, not percentage accuracy.
  Keep allocated and reserved GPU memory distinct and report MiB.
- E0-E4 timing claims apply to the recorded machine and measurement interval.
  Include E1 FP16 and E2 FP16+SDPA references. Channels-last has no demonstrated
  independent benefit; the full-stack gain is not a custom-kernel claim.
- CPR is an appendix negative result. GIT-base-VATEX is a single-checkpoint
  external caption reference with different training/frame preprocessing;
  do not infer a component effect or compare its timing against E0-E4.
- The bounded qualitative review was AI-assisted, not human evaluation.
- Never commit datasets, reference captions, generated captions, weights,
  caches, credentials, personal notes or machine-specific configuration.
  `.gitignore` does not protect files already tracked by Git.
- Before publishing run `python scripts/verify_public_evidence.py` and
  `python scripts/audit_public_release.py`. For code changes run
  `python -m unittest discover -s tests -v` in the documented environment.
- Do not launch full GPU training or benchmark reruns for documentation edits
  or publication checks. Record what was actually tested.
