# Video Captioning Transformer

This repository contains the implementation for a master's graduation project on video captioning with a ViT-based visual encoder and a GPT-2 text decoder. The project also includes local benchmarking and profiling entrypoints, plus a Chainlit + FastAPI demo stack for interactive inference.

## Overview

The system covers the full workflow for video captioning:
- video frame preprocessing
- multimodal caption generation
- training and evaluation scripts
- interactive local inference UI
- Nsight Systems / Nsight Compute profiling workflows

## Repository Layout

- `core/`: runtime, profiling, benchmarking, and shared inference logic
- `src/`: model and training code
- `server/`: FastAPI backend used by the demo stack
- `frontend/`: Chainlit interface for interactive caption generation
- `scripts/`: PowerShell and Python utility scripts for setup, preprocessing, and experiments
- `docs/`: development notes, public writeups, and profiling records
- `reports/`: generated benchmark and profiler outputs
- `data/`: local datasets and processed frame directories
- `checkpoints/`, `outputs/`, `runs/`: model artifacts and experiment outputs

## Models and Data

Large assets are intentionally not committed to GitHub.
- Model checkpoints are expected to be stored locally.
- Datasets such as MSVD or MSR-VTT should be prepared under the local `data/` directory.
- Some workflows default to offline model loading so local benchmarking stays reproducible.

## Features

- Vision Transformer based visual encoding
- GPT-2 based caption decoding
- frame-based video preprocessing pipeline
- CLI and UI inference paths
- local benchmarking for latency and throughput
- Nsight Systems and Nsight Compute profiling entrypoints
- FastAPI + Chainlit demo integration

## Environment Setup

This project uses a dedicated Python 3.11 virtual environment under `.venv`.

Install dependencies:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Check the local project environment:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\check_project_env.ps1
```

## Run the App Stack

Start backend and frontend together:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\start_app_stack.ps1
```

Manual fallback commands:

```powershell
.\.venv\Scripts\python.exe -m uvicorn server.app:app --host 127.0.0.1 --port 8001 --reload
.\.venv\Scripts\python.exe -m chainlit run frontend/chainlit_app.py -w --host 127.0.0.1 --port 8000
```

## CLI Inference

Example:

```powershell
python src/cli/inference.py --video_path example.mp4 --checkpoint outputs/checkpoints/best.ckpt --num_frames 16
```

## Benchmarking and Profiling

Run the optimization baseline benchmark:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_profiling.ps1 -Mode benchmark
```

Default outputs:

```text
reports\baseline_iterations.csv
reports\baseline_summary.json
```

Run the single profiling entrypoint:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_profiling.ps1 -Mode profile
```

Default output:

```text
reports\profile_once.json
```

Run Nsight Systems:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_nsys.ps1
```

Important:
- `nsys-ui.exe` is the GUI launcher and does not accept CLI profiling flags like `profile --trace ...`.
- Use the CLI binary `nsys.exe` (typically under `target-windows-x64`).

Default outputs:

```text
reports\profile_once.nsys-rep
reports\profile_once.json
```

If needed, override the `nsys.exe` path:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_nsys.ps1 -NsightSystemsExe "D:\programs\NsightSystems\target-windows-x64\nsys.exe"
```

Run Nsight Compute for the GPT-2 decoder hotspot:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_ncu.ps1 -Target GPT2_Decoder_Step
```

Or target the encoder:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_ncu.ps1 -Target ViT_Encoder
```

Example outputs:

```text
reports\ncu_gpt2_decoder.ncu-rep
reports\ncu_gpt2_decoder_meta.json
reports\ncu_vit_encoder.ncu-rep
reports\ncu_vit_encoder_meta.json
```

If needed, override the `ncu.bat` path:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_ncu.ps1 -Target GPT2_Decoder_Step -NsightComputeBat "D:\programs\Nsight Compute\ncu.bat"
```

## Notes

- Benchmarking and profiling scripts default to offline model loading unless you explicitly enable online checks.
- Generated reports under `reports/` are local artifacts and may differ by machine, driver, CUDA version, and installed profiler tooling.
