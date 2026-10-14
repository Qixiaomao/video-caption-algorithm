# Video captioning and retrieval on a 4 GiB GPU

ViT-B/16 + GPT-2 Small captioning with a controlled InfoNCE intervention,
paired-seed retrieval evaluation, and a recorded E0-E4 inference study on an
RTX 3050. This repository also retains the original FastAPI/Chainlit demo and
profiling utilities.

Start with [the reproduction guide](reproducibility/README.md). The October 2026
release brings together the formal MSVD/MSR-VTT runners, protocols, CPR negative
result, GIT-base-VATEX reference adapter, and numeric evidence used for the paper.

## Check reported numbers without a GPU

Python 3.11; no third-party packages or downloads needed:

```bash
python scripts/verify_public_evidence.py
python scripts/audit_public_release.py
python -m unittest discover -s tests -p test_public_evidence.py -v
```

The verifier checks 1,036 table rows against 778 extracted numeric source fields
and independently recomputes 270 three-seed summaries. Original source hashes
are recorded separately from the hashes of these public extracts. Checking an
extract is not a replacement for authenticating the omitted original files or
rerunning a model.

## Evidence and interpretation

- MSVD test670: T2V R@1 A/B means 2.5706/5.7138%; paired B-A +3.1433 percentage
  points across seeds 123/124/125, with sample SD 1.0631 points.
- MSR-VTT test2990: T2V R@1 A/B means 0.8340/2.5025%; paired B-A +1.6685 points.
  This uses 2,990 video candidates and 59,794 original caption queries, not the
  standard MSR-VTT 1K retrieval benchmark.
- E4 versus E0 at batch 8: recorded median end-to-end latency 2,541.15 to 863.46 ms
  and peak reserved memory 2,160 to 1,410 MiB. FP16 is the major end-to-end
  contributor. E1 and E2 are required references; the result does not establish
  a universal hardware advantage or an independent channels-last benefit.
- Caption automatic metric changes are mixed. The bounded qualitative review
  was AI-assisted. No completed human evaluation is claimed.
- CPR did not meet the registered retention gate and remains a negative result.
  GIT-base-VATEX is a different-training, six-frame external caption reference;
  its resource observations do not support a speed ranking against E0-E4.

## Install the experiment environment

The recorded runtime is Python 3.11, PyTorch 2.5.1+cu121 and Torchvision
0.20.1+cu121. Use a single environment throughout:

```bash
python -m venv .venv
# Activate .venv in your shell, then:
python -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements-experiments.txt
python -m unittest discover -s tests -v
```

Full tests also require Java on PATH for the locked METEOR/PTB scorer.
Tests requiring local frozen datasets, checkpoints or training traces explicitly
skip when those assets are absent. They must pass when the assets are supplied.
The dependency list reflects the inspected local environment; a clean install
and full GPU reproduction have not been performed for this publication update.

## Where things live

- `src/`: caption model, dataloader, CE/InfoNCE/CPR objectives and checkpoint state.
- `core/`: inference, optimization switches, profiling and benchmark code.
- `scripts/`: preparation, formal experiment runners, scoring and release checks.
- `experiments/`: dated protocols and historical result notes. These are a history,
  not competing current specifications; follow the reproduction guide's entrypoints.
- `tests/`: model-free contract checks and asset-dependent integration checks.
- `reproducibility/evidence/`: public table, numeric extracts, lineage and hashes.
- `reproducibility/archived_drivers/`: original staged recovery/test adapters with
  repository-root relocation; these still require the original frozen run assets.
- `server/`, `frontend/`: the original interactive demo.

Raw videos, reference/generated captions, checkpoints, feature caches, original
full logs and private research notes are not distributed in this snapshot.
Acquire third-party assets from their sources under their respective terms.
See the reproduction guide for the remaining asset and relocation dependencies.

## Original local demo and profiling

These optional workflows use the broader `requirements.txt` environment:

```powershell
python -m pip install -r requirements.txt
powershell -ExecutionPolicy Bypass -File scripts/check_project_env.ps1
powershell -ExecutionPolicy Bypass -File scripts/start_app_stack.ps1
powershell -ExecutionPolicy Bypass -File scripts/run_profiling.ps1 -Mode benchmark
powershell -ExecutionPolicy Bypass -File scripts/run_nsys.ps1
powershell -ExecutionPolicy Bypass -File scripts/run_ncu.ps1 -Target GPT2_Decoder_Step
```

Nsight tools are installed separately. Generated profiler reports stay local.
Read the [reproduction guide](reproducibility/README.md) before changing the experimental contracts.
