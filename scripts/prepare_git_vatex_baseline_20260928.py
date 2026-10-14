"""Pin and download official GIT-base-VATEX checkpoint for later CPU-side baseline prep."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys
from datetime import datetime, timezone

from huggingface_hub import HfApi, hf_hub_download

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/external_baseline_git_vatex_20260928"
REPO = "microsoft/git-base-vatex"
REVISION = "623363cef8744c57438ce8d62b30bef1fbbec111"
FILES = ("README.md", "config.json", "generation_config.json", "model.safetensors",
         "preprocessor_config.json", "special_tokens_map.json", "tokenizer.json",
         "tokenizer_config.json", "vocab.txt")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    if (OUT / "model_download.json").exists():
        raise FileExistsError("Completed model download already recorded")
    info = HfApi().model_info(REPO, revision=REVISION, files_metadata=True)
    if info.sha != REVISION or not info.cardData or info.cardData.get("license") != "mit":
        raise RuntimeError("Official revision/license changed")
    listed = {item.rfilename: item.size for item in info.siblings}
    if any(name not in listed for name in FILES):
        raise RuntimeError("Pinned model file list changed")
    OUT.mkdir(parents=True, exist_ok=True)
    model_dir = OUT / "model"
    model_dir.mkdir(exist_ok=True)
    files = []
    for name in FILES:
        path = Path(hf_hub_download(repo_id=REPO, filename=name, revision=REVISION,
                                    local_dir=str(model_dir)))
        if path.stat().st_size != listed[name]:
            raise RuntimeError(f"Size mismatch: {name}")
        files.append({"path": str(path), "name": name, "bytes": path.stat().st_size,
                      "sha256": sha256(path)})
        print(json.dumps({"downloaded": name, "bytes": path.stat().st_size}), flush=True)
    with (OUT / "model_download.json").open("x", encoding="utf-8") as stream:
        json.dump({"at_utc": datetime.now(timezone.utc).isoformat(), "source": REPO,
                   "revision": REVISION, "license": "mit", "files": files,
                   "scope": "candidate weights and metadata only; no GPU smoke or benchmark"},
                  stream, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    main()
