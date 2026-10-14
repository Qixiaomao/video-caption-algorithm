#!/usr/bin/env python3
"""Validate and aggregate the registered A/B initialization endpoints."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.diagnose_prefix_sensitivity import paired_bootstrap
from scripts.diagnose_semantic_gate import corpus_bleu
from src.training.checkpoint import sha256_file


def mean(values):
    return sum(values) / len(values)


def aggregate(report):
    native = report["conditions"]["native"]
    shuffled = report["conditions"]["shuffled"]
    nll, eos, text_nll, after, delta, after_delta = [], [], [], [], [], []
    for a, b in zip(native["videos"], shuffled["videos"]):
        if a["video_id"] != b["video_id"]:
            raise ValueError("Native/shuffled video order differs")
        for item in (a, b):
            for ref in item["references"]:
                losses = ref["token_nll"]
                if not losses or not all(math.isfinite(x) for x in losses):
                    raise ValueError("Invalid raw token losses")
                if abs(mean(losses) - ref["nll"]) > 1e-6:
                    raise ValueError("Stored reference mean differs from raw token losses")
            if abs(mean([r["nll"] for r in item["references"]]) - item["nll"]) > 1e-6:
                raise ValueError("Stored video mean differs from raw reference losses")
        nll.append(a["nll"])
        eos.append(mean([r["token_nll"][-1] for r in a["references"]]))
        text_nll.append(mean([mean(r["token_nll"][:-1]) for r in a["references"]]))
        after.append(mean([mean(r["token_nll"][1:]) for r in a["references"]]))
        delta.append(b["nll"] - a["nll"])
        after_delta.append(mean([mean(r["token_nll"][1:]) for r in b["references"]]) - after[-1])
    summary = native["caption"]
    samples = [{"hypothesis": v["caption"], "references": [r["text"] for r in refs]}
               for v, refs in zip(native["videos"], report["references"])]
    gate = (summary["nonempty_rate"] == 1 and summary["unique_rate"] >= 0.30
            and summary["largest_template_rate"] <= 0.25)
    return dict(caption=summary, semantic_diversity_gate_pass=gate,
                triage_bleu4=corpus_bleu(samples), native_nll=mean(nll),
                native_eos_nll=mean(eos), native_after_first_nll=mean(after),
                native_non_eos_nll=mean(text_nll),
                grounding_delta=paired_bootstrap(delta),
                grounding_after_first_delta=paired_bootstrap(after_delta),
                per_video=dict(native_nll=nll, native_eos_nll=eos, native_non_eos_nll=text_nll,
                    native_after_first_nll=after, grounding_delta=delta,
                    grounding_after_first_delta=after_delta))


def validate_pair(trial, a, b):
    if trial["config"]["smoke_steps"] != 0 or not trial["matching"]["all_inputs_and_rng_identical"]:
        raise ValueError("Not a valid full matched trial")
    if trial["matching"]["matched_steps"] != 750:
        raise ValueError("Training budget mismatch")
    for label, report in (("A_base", a), ("B_stage3", b)):
        if report["smoke_test"] or report["sample_count"] != 100:
            raise ValueError("Endpoint is not full val100")
        if report["truncated_reference_count"] != 0:
            raise ValueError("Truncated endpoint references")
        if set(report["conditions"]) != {"native", "repeat", "shuffled", "zero"}:
            raise ValueError("Incomplete registered core intervention matrix")
        if not all(report["sanity"][k] for k in ("native_matches_wrapper_first_video",
                                                "identity_controls_passed", "shuffle_tracks_donor")):
            raise ValueError("Endpoint sanity controls failed")
        if trial["arms"][label]["steps"] != 750 or report["checkpoint_metadata"]["progress"]["step"] != 750:
            raise ValueError("Endpoint is not step750")
        if sum(map(len, report["references"])) != 4290:
            raise ValueError("Endpoint does not contain all validation references")
        if report["checkpoint_sha256"] != trial["arms"][label]["checkpoint_sha256"]:
            raise ValueError("Endpoint checkpoint is not registered last-step checkpoint")
        if sha256_file(Path(report["checkpoint"])) != report["checkpoint_sha256"]:
            raise ValueError("Checkpoint changed after endpoint evaluation")
    for key in ("video_ids", "donor_indices", "donor_video_ids", "references", "decode", "annotation_sha256"):
        if a[key] != b[key]:
            raise ValueError(f"Unmatched endpoint field: {key}")
    for key in ("seed", "reference_batch_size", "max_text_tokens", "source_sha256"):
        if a["runtime"][key] != b["runtime"][key]:
            raise ValueError(f"Unmatched endpoint runtime: {key}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trial-root", type=Path, required=True)
    parser.add_argument("--a-report", type=Path, required=True)
    parser.add_argument("--b-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Refusing to overwrite summary")
    paths = {"trial": args.trial_root / "result.json", "A_base": args.a_report, "B_stage3": args.b_report}
    data = {k: json.loads(p.read_text(encoding="utf-8")) for k, p in paths.items()}
    validate_pair(data["trial"], data["A_base"], data["B_stage3"])
    arms = {label: aggregate(data[label]) for label in ("A_base", "B_stage3")}
    contrasts = {}
    for metric in arms["A_base"]["per_video"]:
        values = [b-a for a, b in zip(arms["A_base"]["per_video"][metric],
                                     arms["B_stage3"]["per_video"][metric])]
        contrasts[metric] = dict(paired_bootstrap(values), delta_per_video=values)
    result = dict(protocol="matched-gpt2-init-v1", paper_ready=False,
        verified_full_training_and_endpoint_match=True, arms=arms,
        B_minus_A=contrasts, sign_convention="Lower native NLL is better; higher shuffled-minus-native grounding delta is better.",
        sources={k: {"path": str(p.resolve()), "sha256": sha256_file(p)} for k, p in paths.items()},
        limitations=["Single training seed, frozen GPT-2, fixed 750-step budget.",
                     "Non-EOS NLL is a descriptive decomposition, not an additional primary acceptance test.",
                     "Bootstrap intervals condition on trained weights and fixed donor assignments.",
                     "Exact diversity is not semantic accuracy; BLEU here is unsmoothed triage only."])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps({"output": str(args.output), "arms": {
        k: {m: v for m, v in value.items() if m != "per_video"} for k, value in arms.items()}}, indent=2))


if __name__ == "__main__":
    main()
