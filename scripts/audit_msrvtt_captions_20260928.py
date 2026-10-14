"""Audit all train/val caption tokens before fixing the MSR-VTT length policy."""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import os

from transformers import GPT2TokenizerFast


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs/msrvtt_data_full_20260928"
SOURCE = Path(os.environ.get("VC_GPT2_SOURCE", str(ROOT / "models/gpt2")))


def main() -> None:
    tokenizer = GPT2TokenizerFast.from_pretrained(str(SOURCE), local_files_only=True)
    dataset = json.loads((RUN / "MSR_VTT.json").read_text(encoding="utf-8"))
    sets = {split: set((RUN / "split_lists" / f"{split}_list_full.txt").read_text(encoding="utf-8-sig").splitlines())
            for split in ("train", "val")}
    result = {"at_utc": datetime.now(timezone.utc).isoformat(),
              "annotation_sha256": hashlib.sha256((RUN / "MSR_VTT.json").read_bytes()).hexdigest(),
              "tokenizer_source": str(SOURCE), "proposed_max_len_including_eos": 96,
              "splits": {}}
    for split, ids in sets.items():
        lengths = Counter()
        long_64 = []
        invalid = []
        count = 0
        for row in dataset["annotations"]:
            if row["image_id"] not in ids:
                continue
            count += 1
            tokens = tokenizer.encode(row["caption"], add_special_tokens=False)
            length = len(tokens)
            lengths[length] += 1
            if length >= 64:
                long_64.append({"video_id": row["image_id"], "caption_id": row["id"], "tokens": length})
            if length == 0 or length >= 96 or tokenizer.eos_token_id in tokens:
                invalid.append({"video_id": row["image_id"], "caption_id": row["id"], "tokens": length})
        result["splits"][split] = {"videos": len(ids), "captions": count,
                                      "max_tokens": max(lengths), "length_histogram": dict(sorted(lengths.items())),
                                      "captions_that_max_len_64_would_truncate": long_64,
                                      "invalid_under_max_len_96": invalid}
    result["status"] = "PASS" if all(not x["invalid_under_max_len_96"] for x in result["splits"].values()) else "FAIL"
    target = RUN / "train_val_caption_token_audit.json"
    with target.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps({"status": result["status"], "audit": str(target),
                      "train_max": result["splits"]["train"]["max_tokens"],
                      "val_max": result["splits"]["val"]["max_tokens"]}))
    if result["status"] != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
