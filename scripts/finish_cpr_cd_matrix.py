"""Complete the paused registered C/D matrix without changing the original run."""
from __future__ import annotations

import os

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import argparse
import gc
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from scripts import resume_cpr_cd_matrix as resume
from scripts import run_cpr_cd_matrix as registered
from scripts.eval_retrieval import encode_texts
from scripts.run_caption_policy_ab import tokenize_caption, tokenize_references, write_json
from scripts.run_full_msvd_balanced import DATA_SHA256, TRAIN, VAL
from scripts.run_full_msvd_infonce_ablation import load_strict
from scripts.run_overfit16 import component_hashes
from src.cli.train_caption_mapper import set_seed
from src.data.data_loader import build_dataloader
from src.models.caption_model import VideoCaptionModel
from src.training.checkpoint import sha256_file


def make_pair(
    output_root: Path,
    seed: int,
    initial_hashes: dict[str, str],
    c_result: dict,
    d_result: dict,
) -> dict:
    if c_result["trace_sha256"] != d_result["trace_sha256"]:
        raise RuntimeError(f"seed {seed} C/D trace hashes differ")
    if c_result["trace_sha256"] != registered.REGISTERED_INPUTS[seed]["trace"][1]:
        raise RuntimeError(f"seed {seed} C/D trace differs from registered A trace")
    pair = {
        "seed": seed,
        "smoke_test": False,
        "trace_match": True,
        "historical_trace_prefix_match": True,
        "initial_component_hashes": initial_hashes,
        "results": {
            arm: str((output_root / f"seed{seed}/{arm}/result.json").resolve())
            for arm in registered.ARMS
        },
    }
    write_json(output_root / f"seed{seed}/pair_result.json", pair)
    print(json.dumps({"stage": "pair_complete", **pair}, sort_keys=True), flush=True)
    return pair


def run_seed125(
    output_root: Path,
    *,
    records: list[dict],
    validation: list[dict],
    train_features: torch.Tensor,
    val_features: torch.Tensor,
    baseline_preflight: dict,
    source_hashes: dict[str, str],
) -> tuple[dict[str, dict], dict]:
    seed = 125
    schedule = registered.load_registered_schedule(seed)
    initial_path = registered.REGISTERED_INPUTS[seed]["initial"][0]
    seed_folder = output_root / f"seed{seed}"
    seed_folder.mkdir()

    set_seed(seed)
    model = VideoCaptionModel(
        gpt2_name=baseline_preflight["model_source"]["path"],
        freeze_vit=True,
        vit_enable_torch_compile=False,
    )
    load_strict(model, initial_path)
    initial_hashes = component_hashes(model)
    model = model.cuda().eval()
    loader = build_dataloader(
        str(TRAIN), model.decoder.tokenizer,
        batch_size=8, max_len=64, num_frame=8, image_size=224,
        shuffle=False, num_wokers=0, random_caption=False,
    )
    first_batch = next(iter(loader))
    with torch.inference_mode():
        direct_features = registered.raw_backbone_features(
            model.encoder, first_batch["video"].cuda()
        )
        cache_error = float((direct_features.cpu() - train_features[:8]).abs().max())
    if cache_error > 1e-6:
        raise RuntimeError(f"seed {seed} frozen feature cache mismatch: {cache_error}")

    token_cache = {
        text: tokenize_caption(model.decoder.tokenizer, text)
        for record in records for text in record["captions"]
    }
    references = {
        item["video_id"]: tokenize_references(model.decoder.tokenizer, item["captions"])
        for item in validation
    }
    all_val_captions = [text for item in validation for text in item["captions"]]
    text_embeddings = encode_texts(
        model, all_val_captions, max_length=64, batch_size=128, device="cuda"
    )
    caption_gold = torch.tensor([
        index for index, item in enumerate(validation) for _ in item["captions"]
    ])

    preflight = {
        "seed": seed,
        "smoke_test": False,
        "paper_ready": False,
        "test_split_read": False,
        "source_sha256": source_hashes,
        "git_head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "initial_checkpoint": str(initial_path.resolve()),
        "initial_checkpoint_sha256": sha256_file(initial_path),
        "initial_component_hashes": initial_hashes,
        "schedule": str(registered.REGISTERED_INPUTS[seed]["schedule"][0].resolve()),
        "schedule_sha256": registered.REGISTERED_INPUTS[seed]["schedule"][1],
        "historical_trace": str(registered.REGISTERED_INPUTS[seed]["trace"][0].resolve()),
        "historical_trace_sha256": registered.REGISTERED_INPUTS[seed]["trace"][1],
        "cache_error": cache_error,
        "smoke_checks": None,
        "runtime": {
            "python": sys.version,
            "torch": str(torch.__version__),
            "gpu": torch.cuda.get_device_name(),
            "physical_gpu_mib": torch.cuda.get_device_properties(0).total_memory / 2**20,
            "deterministic_algorithms": True,
            "tf32": False,
        },
    }
    write_json(seed_folder / "preflight.json", preflight)

    results: dict[str, dict] = {}
    for arm in registered.ARMS:
        print(json.dumps({"stage": "arm_start", "seed": seed, "arm": arm}), flush=True)
        results[arm] = registered.run_arm(
            model=model,
            arm=arm,
            seed=seed,
            folder=seed_folder / arm,
            initial_path=initial_path,
            initial_hashes=initial_hashes,
            schedule=schedule,
            records=records,
            features=train_features,
            val_records=validation,
            val_features=val_features,
            references=references,
            text_embeddings=text_embeddings,
            caption_gold=caption_gold,
            token_cache=token_cache,
            source_hashes=source_hashes,
            baseline_preflight=baseline_preflight,
            smoke=False,
        )
    pair = make_pair(output_root, seed, initial_hashes, results["C"], results["D"])
    del model, token_cache, text_embeddings
    gc.collect()
    torch.cuda.empty_cache()
    return results, pair


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    output_root = args.output_root.resolve()
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output_root}")
    if output_root == resume.PAUSED_ROOT.resolve() or resume.PAUSED_ROOT.resolve() in output_root.parents:
        raise ValueError("output must be outside the paused formal run")
    if not torch.cuda.is_available():
        raise RuntimeError("registered C/D matrix requires a CUDA GPU")

    context = resume.prepare_context()
    registered.load_registered_schedule(125)
    if args.check_only:
        print(json.dumps({"preflight_pass": True, "test_split_read": False}), flush=True)
        return 0

    output_root.mkdir(parents=True)
    print(json.dumps({"stage": "arm_start", "seed": 124, "arm": "C", "resume_step": 6001}), flush=True)
    c124 = resume.replay(context, output_root, verify_steps=None)
    preflight124 = resume.PAUSED_ROOT / "seed124/preflight.json"
    write_json(
        output_root / "seed124/preflight.json",
        json.loads(preflight124.read_text(encoding="utf-8")),
    )
    print(json.dumps({"stage": "arm_complete", "seed": 124, "arm": "C", "steps": c124["steps"]}), flush=True)

    train_features = context["train_features"]
    val_features = context["val_features"]
    records = context["records"]
    validation = context["validation"]
    baseline_preflight = context["baseline_preflight"]
    source_hashes = context["source_hashes"]
    print(json.dumps({"stage": "arm_start", "seed": 124, "arm": "D"}), flush=True)
    d124 = registered.run_arm(
        model=context["model"],
        arm="D",
        seed=124,
        folder=output_root / "seed124/D",
        initial_path=registered.REGISTERED_INPUTS[124]["initial"][0],
        initial_hashes=context["initial_hashes"],
        schedule=context["schedule"],
        records=records,
        features=train_features,
        val_records=validation,
        val_features=val_features,
        references=context["references"],
        text_embeddings=context["text_embeddings"],
        caption_gold=context["caption_gold"],
        token_cache=context["token_cache"],
        source_hashes=source_hashes,
        baseline_preflight=baseline_preflight,
        smoke=False,
    )
    pair124 = make_pair(output_root, 124, context["initial_hashes"], c124, d124)
    del context
    gc.collect()
    torch.cuda.empty_cache()

    seed125, pair125 = run_seed125(
        output_root,
        records=records,
        validation=validation,
        train_features=train_features,
        val_features=val_features,
        baseline_preflight=baseline_preflight,
        source_hashes=source_hashes,
    )
    old_root = resume.PAUSED_ROOT
    seed123 = {
        arm: json.loads((old_root / f"seed123/{arm}/result.json").read_text(encoding="utf-8"))
        for arm in registered.ARMS
    }
    pair123 = json.loads((old_root / "seed123/pair_result.json").read_text(encoding="utf-8"))
    all_results = {123: seed123, 124: {"C": c124, "D": d124}, 125: seed125}
    decision = registered.formal_decision(all_results)
    completion = {
        "completed": True,
        "seeds": list(registered.SEEDS),
        "arms": list(registered.ARMS),
        "pairs": [pair123, pair124, pair125],
        "formal_decision": decision,
        "resume_provenance": c124["resume_provenance"],
        "smoke_test": False,
        "paper_ready": False,
        "test_split_read": False,
    }
    write_json(output_root / "completion.json", completion)
    print(json.dumps({
        "stage": "matrix_complete",
        "completion": str((output_root / "completion.json").resolve()),
        "decision": decision["gates"]["decision"],
        "test_split_read": False,
    }, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
