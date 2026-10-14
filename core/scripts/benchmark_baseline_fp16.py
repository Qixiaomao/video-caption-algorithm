"""Compatibility entrypoint for the repaired benchmark with ViT FP16 enabled.

The old implementation duplicated the decoder loop and could terminate after a
single EOS token. Keep one benchmark implementation so FP32/FP16 use the same
input, decoding, validation, and metadata protocol.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.scripts import benchmark_baseline


def main():
    if "--vit-fp16" not in sys.argv and "--no-vit-fp16" not in sys.argv:
        sys.argv.append("--vit-fp16")
    benchmark_baseline.main()


if __name__ == "__main__":
    main()
