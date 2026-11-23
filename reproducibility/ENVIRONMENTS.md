# Environments and replay boundaries

The paper's experiments were not all run in one software environment. Installing
the public dependency file does not recreate every historical runtime.

## Recorded experiment environments

- **MSVD training:** frozen training preflights record PyTorch 2.11.0+cu128.
- **E0-E4 inference efficiency:** Windows 11, Python 3.14.3,
  PyTorch 2.11.0+cu128 and CUDA 12.8, on an RTX 3050 Laptop GPU with
  4,095.5625 MiB physical memory.
- **MSR-VTT training:** the formal launch and staged-run records specify
  PyTorch 2.5.1+cu121.

These are observations from the experiment records, not interchangeable
dependency recommendations. The paper does not compare training costs across
the two environments. Consult each dated protocol and its preflight/config
record before replaying an experiment. Preserve the registered precision,
determinism, decoding and checkpoint contracts as well as package versions.

## Public Python 3.11 dependency profile

`requirements-experiments.txt` is the public Python 3.11 profile, with
PyTorch 2.5.1 and Torchvision 0.20.1. Install their cu121 wheels first using the
command in the root README, then install the remaining requirements in the
same environment. This profile is not an exact environment lock for the
MSVD/efficiency runs. A clean dependency installation followed by a full GPU
reproduction has not been completed for this publication update.

The broader `requirements.txt` serves the optional application/profiling
workflow. Do not combine the two dependency files as a way to reconstruct a
historical run. Java on PATH is also required by the caption scoring workflow.

## Checks available without model execution

The public evidence verifier, release scan and standard-library evidence tests
run on Python 3.11 without third-party packages, datasets or model weights.
They validate the public files and their arithmetic. They do not establish
GPU compatibility, authenticate omitted original files or reproduce model
outputs. See the [reproduction guide](README.md) for the remaining asset and
relocation dependencies.
