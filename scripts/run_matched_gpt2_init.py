#!/usr/bin/env python3
"""Matched frozen-GPT-2 initialization trial; never resumes a legacy mapper."""
from __future__ import annotations

import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import argparse
import gc
import hashlib
import json
import random
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import transformers
from transformers import AutoConfig, GPT2TokenizerFast
from transformers.utils.hub import cached_file

from src.cli.train_caption_mapper import compute_loss_local, evaluate, set_seed
from src.data.data_loader import build_dataloader
from src.models.caption_model import VideoCaptionModel
from src.training.checkpoint import (
    build_dataset_contract, capture_rng_state, restore_model_state,
    save_training_checkpoint, sha256_file,
)

TRAIN = ROOT / "data/processed/msvd_full/train/annotations.json"
VAL = ROOT / "data/processed/msvd_full/val/annotations.json"
DATA_HASHES = {
    "train": "ced443e28c5ad2ddd0780af55a820b5ec7da6462c092e1329951dc9c7598c7c5",
    "val": "bcda3ba0583704a54297e4da4cc9acbcc5ec1ebffd6ed436a6ba3ee1b8a03c8b",
}
STAGE3_SHA = "a37dc482789e8ee4068deef308aa2793719849664054ce82db00f92efd489dcd"
SOURCES = [
    "scripts/run_matched_gpt2_init.py", "src/cli/train_caption_mapper.py",
    "src/data/data_loader.py", "src/models/caption_model.py",
    "src/models/text_decoder.py", "src/models/video_encoder.py",
    "src/training/checkpoint.py", "src/training/objectives.py",
]


def tensor_sha(tensor):
    array = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str((str(array.dtype), tuple(array.shape))).encode())
    digest.update(array.numpy().tobytes())
    return digest.hexdigest()


def state_sha(state):
    digest = hashlib.sha256()
    for key in sorted(state):
        digest.update(key.encode())
        digest.update(tensor_sha(state[key]).encode())
    return digest.hexdigest()


def component_hashes(model):
    state = model.state_dict()
    return {
        "visual": state_sha({k: v for k, v in state.items() if not k.startswith("decoder.model.")}),
        "decoder": state_sha({k: v for k, v in state.items() if k.startswith("decoder.model.")}),
        "backbone": state_sha({k: v for k, v in state.items()
                               if k.startswith("encoder.") and not k.startswith("encoder.proj.")}),
        "projection": state_sha({k: v for k, v in state.items() if k.startswith("encoder.proj.")}),
        "mapper": state_sha({k: v for k, v in state.items() if k.startswith("decoder.mapper.")}),
    }


def batch_evidence(batch, step, epoch):
    rng = capture_rng_state()
    return {
        "step": step, "epoch": epoch, "video_ids": list(batch["video_id"]),
        "caption_ids_sha256": tensor_sha(batch["caption_ids"]),
        "video_tensor_sha256": tensor_sha(batch["video"]),
        "python_rng_sha256": hashlib.sha256(repr(rng["python"]).encode()).hexdigest(),
        "torch_cpu_rng_sha256": tensor_sha(rng["torch_cpu"]),
        "torch_cuda_rng_sha256": [tensor_sha(x) for x in rng["torch_cuda"] or []],
    }


def compare_traces(left, right, expected_steps):
    a, b = [json.loads(x) for x in left.read_text().splitlines()], [json.loads(x) for x in right.read_text().splitlines()]
    if len(a) != expected_steps or len(b) != expected_steps:
        raise RuntimeError(f"Trace length mismatch: {len(a)}, {len(b)}, expected {expected_steps}")
    for step, (x, y) in enumerate(zip(a, b), 1):
        if x != y:
            keys = [k for k in set(x) | set(y) if x.get(k) != y.get(k)]
            raise RuntimeError(f"Unmatched inputs/RNG at step {step}: {keys}")
    return {"matched_steps": len(a), "all_inputs_and_rng_identical": True,
            "left_sha256": sha256_file(left), "right_sha256": sha256_file(right)}


def write_json(path, payload):
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)


def make_model(source):
    return VideoCaptionModel(gpt2_name=source, freeze_vit=True,
                             vit_enable_torch_compile=False)


def configure_trainable(model):
    model.requires_grad_(False)
    model.encoder.proj.requires_grad_(True)
    model.decoder.mapper.requires_grad_(True)
    named = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
    if not named or any(not name.startswith(("encoder.proj.", "decoder.mapper.")) for name, _ in named):
        raise RuntimeError("Unexpected trainable parameter group")
    return named


def run_arm(label, source, common_path, directory, config, dataset_contract, expected_visual):
    directory.mkdir()
    set_seed(config["seed"])
    model = make_model(source)
    common = torch.load(common_path, map_location="cpu", weights_only=True)
    restore_model_state(model, common, visual_only=True)
    del common
    initial = component_hashes(model)
    if initial["visual"] != expected_visual:
        raise RuntimeError("Visual initialization did not match common bundle")
    model = model.to("cuda").train()
    named = configure_trainable(model)
    optimizer = torch.optim.AdamW([p for _, p in named], lr=config["lr"], weight_decay=0.01)
    if optimizer.state:
        raise RuntimeError("Optimizer unexpectedly has inherited state")
    tok = model.decoder.tokenizer
    train_gen = torch.Generator().manual_seed(config["seed"])
    val_gen = torch.Generator().manual_seed(config["seed"] + 1)
    loader_args = dict(tokenizer=tok, batch_size=config["batch_size"], num_frame=8,
                       image_size=224, max_len=config["max_len"], num_wokers=0)
    train = build_dataloader(str(TRAIN), shuffle=True, random_caption=True,
                             generator=train_gen, **loader_args)
    val = build_dataloader(str(VAL), shuffle=False, random_caption=False,
                           generator=val_gen, **loader_args)
    if len(train.dataset) != 1200 or len(val.dataset) != 100:
        raise RuntimeError("Loader silently dropped data")
    schema = [{"name": n, "shape": list(p.shape), "numel": p.numel()} for n, p in named]
    write_json(directory / "initial.json", {"label": label, "source": source, "hashes": initial,
        "trainable": schema, "optimizer": "AdamW", "lr": config["lr"], "weight_decay": 0.01,
        "fresh_optimizer": True})
    objective = dict(alignment_objective="none", alignment_weight=0.0,
                     temperature=0.07, false_negative_policy="exact_caption")
    contract = dict(objective_version=3, caption_objective="causal_lm_cross_entropy",
        alignment_objective="none", alignment_weight=0.0, info_nce_temperature=0.07,
        false_negative_policy="exact_caption", video_representation="mean_visual_prefix_tokens_l2_normalized",
        text_representation="frozen_gpt2_wte_masked_mean_l2_normalized",
        physical_batch_size=config["batch_size"], matched_initialization_protocol="matched-gpt2-init-v1",
        common_visual_sha256=expected_visual, decoder_initial_sha256=initial["decoder"],
        frozen_gpt2=True, seed_reset_after_model_initialization=True, legacy_visual_weights_used=False)
    saved_args = dict(config, gpt2_name=source, vit_name="vit_base_patch16_224",
                      cond_mode="prefix", prefix_len=4, unfreeze_gpt2_last=0, freeze_vit=True)
    set_seed(config["seed"])
    torch.cuda.reset_peak_memory_stats()
    step, best_val, history = 0, float("inf"), []
    started = time.monotonic()
    with (directory / "trace.jsonl").open("x", encoding="utf-8") as trace, \
         (directory / "metrics.jsonl").open("x", encoding="utf-8") as metrics_file:
        for epoch in range(config["epochs"]):
            epoch_start = train_gen.get_state()
            losses = []
            for batch_in_epoch, batch in enumerate(train, 1):
                step += 1
                trace.write(json.dumps(batch_evidence(batch, step, epoch), sort_keys=True) + "\n")
                trace.flush()
                video, caps = batch["video"].cuda(), batch["caption_ids"].cuda()
                optimizer.zero_grad(set_to_none=True)
                loss = compute_loss_local(model, video, caps, tok.pad_token_id)
                if not torch.isfinite(loss):
                    raise RuntimeError(f"Nonfinite loss at {label} step {step}")
                loss.backward()
                if any(p.grad is None or not torch.isfinite(p.grad).all() for _, p in named):
                    raise RuntimeError(f"Missing/nonfinite trainable gradient at {label} step {step}")
                optimizer.step()
                value = float(loss.detach())
                losses.append(value)
                metrics_file.write(json.dumps({"step": step, "epoch": epoch, "loss": value}) + "\n")
                metrics_file.flush()
                if step == 1 or step % 25 == 0:
                    print(f"[{label}] epoch={epoch + 1} step={step} loss={value:.4f} elapsed={time.monotonic()-started:.1f}s", flush=True)
                del video, caps, loss
                if config["smoke_steps"] and step >= config["smoke_steps"]:
                    break
            val_loss = evaluate(model, val, tok.pad_token_id, "cuda", objective)["loss"]
            if not torch.isfinite(torch.tensor(val_loss)):
                raise RuntimeError("Nonfinite validation loss")
            history.append(dict(epoch=epoch + 1, step=step, train_loss=sum(losses)/len(losses), val_loss=val_loss))
            best_val = min(best_val, val_loss)
            save_training_checkpoint(directory / "last.pt", model=model, optimizer=optimizer,
                scheduler=None, progress=dict(step=step, epoch=epoch, batch_in_epoch=batch_in_epoch,
                epoch_complete=batch_in_epoch == len(train), best_val=best_val), args=saved_args,
                training_contract=contract, dataset_contract=dataset_contract,
                loader_state=dict(current_generator_state=train_gen.get_state(),
                                  epoch_start_generator_state=epoch_start, num_workers=0))
            print(f"[{label}] epoch_end={epoch + 1} step={step} val_loss={val_loss:.4f}", flush=True)
            if config["smoke_steps"]:
                break
    optimizer.zero_grad(set_to_none=True)
    final = component_hashes(model)
    if final["decoder"] != initial["decoder"] or final["backbone"] != initial["backbone"]:
        raise RuntimeError("Frozen model weights changed")
    if any(final[k] == initial[k] for k in ("projection", "mapper")):
        raise RuntimeError("A trainable visual module did not update")
    result = dict(label=label, steps=step, history=history, initial=initial, final=final,
                  trainable=schema, frozen_weights_unchanged=True,
                  checkpoint=str((directory / "last.pt").resolve()),
                  checkpoint_sha256=sha256_file(directory / "last.pt"),
                  elapsed_seconds=time.monotonic()-started,
                  peak_allocated_mb=torch.cuda.max_memory_allocated()/2**20,
                  peak_reserved_mb=torch.cuda.max_memory_reserved()/2**20)
    write_json(directory / "result.json", result)
    del model, optimizer, train, val, named
    gc.collect()
    torch.cuda.empty_cache()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--smoke-steps", type=int, default=0)
    args = parser.parse_args()
    if args.smoke_steps not in (0, 2):
        parser.error("Only the locked 750-step trial or two-step smoke is supported")
    os.chdir(ROOT)
    if args.output_root.exists():
        raise FileExistsError("Refusing to overwrite an existing trial directory")
    if not torch.cuda.is_available():
        raise RuntimeError("This registered trial requires the local CUDA GPU")
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    config = dict(seed=123, batch_size=8, max_len=64, lr=3e-4,
                  epochs=5, smoke_steps=args.smoke_steps)
    source_hashes = {f: sha256_file(ROOT / f) for f in SOURCES}
    dataset_contract = build_dataset_contract({"train": str(TRAIN), "val": str(VAL)})
    for split, expected in DATA_HASHES.items():
        if dataset_contract["annotations"][split]["sha256"] != expected:
            raise RuntimeError(f"Pinned {split} annotation hash changed")
    base = str(Path(cached_file("gpt2", "model.safetensors", local_files_only=True)).parent)
    legacy = str(ROOT / "checkpoints/gpt2_lm_stage3/best")
    if sha256_file(Path(legacy) / "model.safetensors") != STAGE3_SHA:
        raise RuntimeError("Stage3 source changed")
    sources = {"A_base": base, "B_stage3": legacy}
    config_keys = ("n_embd", "n_layer", "n_head", "n_positions", "vocab_size",
        "activation_function", "resid_pdrop", "embd_pdrop", "attn_pdrop",
        "layer_norm_epsilon", "scale_attn_weights", "scale_attn_by_inverse_layer_idx",
        "reorder_and_upcast_attn", "tie_word_embeddings", "bos_token_id", "eos_token_id")
    lm_configs = {label: AutoConfig.from_pretrained(path, local_files_only=True)
                  for label, path in sources.items()}
    matched_config = {label: {k: getattr(cfg, k) for k in config_keys}
                      for label, cfg in lm_configs.items()}
    if matched_config["A_base"] != matched_config["B_stage3"]:
        raise RuntimeError("Decoder architecture/dropout configuration differs")
    decoder_files = {label: {str(p): sha256_file(p) for p in Path(path).iterdir() if p.is_file()}
                     for label, path in sources.items()}
    toks = [GPT2TokenizerFast.from_pretrained(path, local_files_only=True) for path in sources.values()]
    text_audit = {}
    for split, path in (("train", TRAIN), ("val", VAL)):
        records = json.loads(path.read_text(encoding="utf-8"))
        texts = [c for r in records for c in r["captions"]]
        encoded = [tok(texts, add_special_tokens=False)["input_ids"] for tok in toks]
        if encoded[0] != encoded[1]:
            raise RuntimeError("Tokenizer outputs differ between arms")
        if any(not x or len(x) >= config["max_len"] or toks[0].eos_token_id in x for x in encoded[0]):
            raise RuntimeError("Text would be empty/truncated or lose EOS supervision")
        text_audit[split] = dict(caption_count=len(texts), max_tokens=max(map(len, encoded[0])),
                                identical_tokenization=True, truncated=0)
    del toks, encoded
    set_seed(config["seed"])
    common_model = make_model(base)
    common_state = {k: v.detach().cpu() for k, v in common_model.state_dict().items()
                    if not k.startswith("decoder.model.")}
    visual_hash = state_sha(common_state)
    args.output_root.mkdir(parents=True)
    common_path = args.output_root / "common_visual.pt"
    torch.save({"model_state": common_state, "fresh_initialization": True}, common_path)
    del common_state, common_model
    gc.collect()
    write_json(args.output_root / "preflight.json", dict(protocol="matched-gpt2-init-v1",
        config=config, common_visual_sha256=visual_hash, common_file_sha256=sha256_file(common_path),
        sources=sources, decoder_files=decoder_files, matched_decoder_config=matched_config,
        dataset_contract=dataset_contract,
        tokenization=text_audit, source_sha256=source_hashes,
        git_head=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        runtime=dict(python=sys.version, torch=torch.__version__, transformers=transformers.__version__,
                     gpu=torch.cuda.get_device_name(), deterministic_algorithms=True, tf32=False)))
    arms = {label: run_arm(label, path, common_path, args.output_root / label, config,
                          dataset_contract, visual_hash) for label, path in sources.items()}
    matching = compare_traces(args.output_root / "A_base/trace.jsonl",
                              args.output_root / "B_stage3/trace.jsonl", args.smoke_steps or 750)
    if arms["A_base"]["trainable"] != arms["B_stage3"]["trainable"]:
        raise RuntimeError("Trainable schemas differ")
    if arms["A_base"]["initial"]["decoder"] == arms["B_stage3"]["initial"]["decoder"]:
        raise RuntimeError("Treatment is absent: decoder states are identical")
    for files in decoder_files.values():
        if any(sha256_file(Path(p)) != h for p, h in files.items()):
            raise RuntimeError("Decoder source changed during the trial")
    if source_hashes != {f: sha256_file(ROOT / f) for f in SOURCES}:
        raise RuntimeError("Training source changed during trial")
    if dataset_contract != build_dataset_contract({"train": str(TRAIN), "val": str(VAL)}):
        raise RuntimeError("Dataset contract changed during trial")
    result = dict(protocol="matched-gpt2-init-v1", paper_ready=False, config=config,
                  matching=matching, arms=arms, source_and_data_unchanged=True,
                  limits=["One seed and finite budget; this only estimates the frozen-decoder initialization effect.",
                          "A null or worse result does not isolate which Stage3 training choice caused it.",
                          "No legacy visual initialization, InfoNCE, test670 evaluation, or model replacement."])
    write_json(args.output_root / "result.json", result)
    print(json.dumps({"output": str(args.output_root.resolve()), "matching": matching}, indent=2), flush=True)


if __name__ == "__main__":
    main()
