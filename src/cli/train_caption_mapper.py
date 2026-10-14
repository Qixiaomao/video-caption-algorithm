# src/cli/train_caption_mapper.py
from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
from typing import Dict, List, Mapping

import torch
import torch.nn as nn
import torch.optim as optim
from transformers import GPT2TokenizerFast

from src.data.data_loader import build_dataloader
from src.models.caption_model import VideoCaptionModel, build_caption_supervision
from src.training.checkpoint import (
    build_dataset_contract,
    load_checkpoint,
    restore_model_state,
    restore_rng_state,
    restore_training_state,
    save_training_checkpoint,
    validate_training_resume_contract,
)
from src.training.objectives import (
    ALIGNMENT_OBJECTIVES,
    FALSE_NEGATIVE_POLICIES,
    compute_caption_alignment_objective,
)
from src.training.cpr import CPR_NEGATIVE_POLICIES


METRIC_NAMES = (
    "loss",
    "caption_loss",
    "alignment_loss",
    "cpr_loss",
    "matched_nll",
    "mismatched_nll",
    "cpr_gap",
    "cpr_eligible_fraction",
    "positive_similarity",
    "negative_similarity",
    "similarity_gap",
)


def set_seed(seed: int) -> None:
    import random

    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def compute_loss_local(
    model: VideoCaptionModel,
    video: torch.Tensor,
    caption_ids: torch.Tensor,
    pad_id: int,
) -> torch.Tensor:
    """Backward-compatible caption-only objective used by existing diagnostics."""

    attention_mask, labels = build_caption_supervision(caption_ids, pad_id)
    result = model(
        video=video,
        input_ids=caption_ids,
        attention_mask=attention_mask,
        labels=labels,
    )
    return result["loss"]


def objective_kwargs(args: argparse.Namespace) -> Dict[str, object]:
    return {
        "alignment_objective": args.alignment_objective,
        "alignment_weight": args.alignment_weight,
        "temperature": args.info_nce_temperature,
        "false_negative_policy": args.false_negative_policy,
        "cpr_weight": args.cpr_weight,
        "cpr_margin": args.cpr_margin,
        "cpr_negative_policy": args.cpr_negative_policy,
    }


@torch.no_grad()
def evaluate(
    model: VideoCaptionModel,
    loader,
    pad_id: int,
    device: str,
    objective_config: Mapping[str, object],
) -> Dict[str, float]:
    model.eval()
    totals = {name: 0.0 for name in METRIC_NAMES}
    batches = 0
    for batch in loader:
        video = batch["video"].to(device, non_blocking=True)
        captions = batch["caption_ids"].to(device, non_blocking=True)
        metrics = compute_caption_alignment_objective(
            model,
            video,
            captions,
            pad_id,
            video_ids=batch["video_id"],
            **objective_config,
        )
        for name in METRIC_NAMES:
            totals[name] += float(metrics[name].item())
        batches += 1
        if batches >= 50:
            break
    model.train()
    return {name: value / max(batches, 1) for name, value in totals.items()}


def append_metrics(path: Path, step: int, metrics: Mapping[str, float]) -> None:
    write_header = not path.exists() or path.stat().st_size == 0
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        if write_header:
            writer.writerow(("step", *METRIC_NAMES))
        writer.writerow((step, *(f"{metrics[name]:.8f}" for name in METRIC_NAMES)))


def format_metrics(metrics: Mapping[str, float]) -> str:
    return (
        f"loss={metrics['loss']:.4f} caption={metrics['caption_loss']:.4f} "
        f"align={metrics['alignment_loss']:.4f} sim_gap={metrics['similarity_gap']:.4f} "
        f"cpr={metrics['cpr_loss']:.4f} cpr_gap={metrics['cpr_gap']:.4f}"
    )


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("Fine-tune the caption mapper with an optional alignment objective")

    parser.add_argument("--ann_train", default="./data/processed/msvd_full/train/annotations.json")
    parser.add_argument("--ann_val", default="./data/processed/msvd_full/val/annotations.json")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_frame", type=int, default=8)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--max_len", type=int, default=48)
    parser.add_argument("--shuffle", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--vit_name", default="vit_base_patch16_224")
    parser.add_argument("--gpt2_name", default="gpt2")
    parser.add_argument("--cond_mode", choices=["prefix", "bos"], default="prefix")
    parser.add_argument("--prefix_len", type=int, default=4)

    parser.add_argument("--freeze_vit", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--unfreeze_gpt2_last",
        type=int,
        default=0,
        help="unfreeze the last N GPT-2 blocks; must remain 0 for stage-0 alignment runs",
    )

    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=3e-4, help="learning rate for projection/mapper")
    parser.add_argument("--lr_gpt2", type=float, default=1e-5)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--val_every", type=int, default=20)
    parser.add_argument("--max_steps", type=int, default=0, help="global optimizer-step cap; 0 disables it")
    parser.add_argument(
        "--smoke_test",
        action="store_true",
        help="skip validation for a bounded (at most 10 steps) systems smoke test",
    )

    parser.add_argument("--alignment_objective", choices=ALIGNMENT_OBJECTIVES, default="none")
    parser.add_argument("--alignment_weight", type=float, default=0.0)
    parser.add_argument("--info_nce_temperature", type=float, default=0.07)
    parser.add_argument(
        "--false_negative_policy",
        choices=FALSE_NEGATIVE_POLICIES,
        default="exact_caption",
    )
    parser.add_argument("--cpr_weight", type=float, default=0.0)
    parser.add_argument("--cpr_margin", type=float, default=0.10)
    parser.add_argument(
        "--cpr_negative_policy",
        choices=CPR_NEGATIVE_POLICIES,
        default="cyclic_derangement",
    )

    parser.add_argument("--run_dir", default="runs/mapper_ft")
    parser.add_argument("--ckpt_dir", default="checkpoints")
    parser.add_argument("--ckpt_name", default="msvd_mapper_finetune.pt")
    parser.add_argument("--resume_ckpt", default="", help="restore model weights from this checkpoint")
    parser.add_argument(
        "--resume_visual_only",
        action="store_true",
        help="restore encoder/projection/mapper while retaining the configured GPT-2 weights",
    )
    parser.add_argument(
        "--resume_training_state",
        action="store_true",
        help="also restore optimizer, RNG, epoch, batch position, and DataLoader generator",
    )
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.batch_size < 1:
        raise ValueError("batch_size must be positive")
    if args.alignment_weight < 0:
        raise ValueError("alignment_weight must be non-negative")
    if args.cpr_weight < 0:
        raise ValueError("cpr_weight must be non-negative")
    if args.cpr_margin <= 0:
        raise ValueError("cpr_margin must be positive")
    if args.alignment_objective == "none" and args.alignment_weight != 0:
        raise ValueError("alignment_weight must be 0 when alignment_objective=none")
    if args.alignment_objective != "none" and args.alignment_weight <= 0:
        raise ValueError("alignment_weight must be positive for an alignment objective")
    if args.alignment_objective == "infonce" and args.batch_size < 2:
        raise ValueError("InfoNCE requires batch_size >= 2")
    if args.cpr_weight > 0 and args.batch_size < 2:
        raise ValueError("CPR requires batch_size >= 2")
    if (args.alignment_objective != "none" or args.cpr_weight > 0) and args.unfreeze_gpt2_last != 0:
        raise ValueError("alignment and CPR experiments require GPT-2 to remain frozen")
    if args.info_nce_temperature <= 0:
        raise ValueError("info_nce_temperature must be positive")
    if args.resume_training_state and not args.resume_ckpt:
        raise ValueError("--resume_training_state requires --resume_ckpt")
    if args.resume_training_state and args.resume_visual_only:
        raise ValueError("exact training resume is incompatible with --resume_visual_only")
    if args.smoke_test and (args.max_steps < 1 or args.max_steps > 10):
        raise ValueError("--smoke_test requires 1 <= max_steps <= 10")
    if args.smoke_test and args.val_every != 0:
        raise ValueError("--smoke_test requires val_every=0")


def training_contract(args: argparse.Namespace) -> Dict[str, object]:
    return {
        "objective_version": 4,
        "caption_objective": "causal_lm_cross_entropy",
        "alignment_objective": args.alignment_objective,
        "alignment_weight": args.alignment_weight,
        "info_nce_temperature": args.info_nce_temperature,
        "false_negative_policy": args.false_negative_policy,
        "cpr_weight": args.cpr_weight,
        "cpr_margin": args.cpr_margin,
        "cpr_negative_policy": args.cpr_negative_policy,
        "video_representation": "mean_visual_prefix_tokens_l2_normalized",
        "text_representation": "frozen_gpt2_wte_masked_mean_l2_normalized",
        "symmetric_video_text_loss": args.alignment_objective == "infonce",
        "predict_first_token_from_visual_prefix": True,
        "ignore_eos_padding_in_caption_loss": True,
        "physical_batch_size": args.batch_size,
        "gradient_accumulation_creates_negatives": False,
        "inference_extra_parameters": 0,
        "smoke_test": args.smoke_test,
        "recovered_from": args.resume_ckpt or None,
        "decoder_initialized_from": args.gpt2_name,
        "resume_visual_only": args.resume_visual_only,
    }


def checkpoint_progress(
    *,
    step: int,
    epoch: int,
    batch_in_epoch: int,
    epoch_complete: bool,
    best_val: float,
) -> Dict[str, object]:
    return {
        "step": step,
        "epoch": epoch,
        "batch_in_epoch": batch_in_epoch,
        "epoch_complete": epoch_complete,
        "best_val": best_val,
    }


def main() -> None:
    args = build_argparser().parse_args()
    validate_args(args)
    set_seed(args.seed)

    run_dir = Path(args.run_dir)
    ckpt_dir = Path(args.ckpt_dir)
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)
    events_csv = run_dir / "events.csv"
    val_csv = run_dir / "val.csv"

    tokenizer = GPT2TokenizerFast.from_pretrained(args.gpt2_name)
    tokenizer.pad_token = tokenizer.eos_token

    use_pairwise_objective = args.alignment_objective != "none" or args.cpr_weight > 0
    train_generator = torch.Generator().manual_seed(args.seed)
    val_generator = torch.Generator().manual_seed(args.seed + 1)
    train_loader = build_dataloader(
        ann_path=args.ann_train,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        max_len=args.max_len,
        num_frame=args.num_frame,
        image_size=args.image_size,
        shuffle=args.shuffle,
        num_wokers=0,
        random_caption=True,
        drop_last=use_pairwise_objective,
        generator=train_generator,
    )
    val_loader = build_dataloader(
        ann_path=args.ann_val,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        max_len=args.max_len,
        num_frame=args.num_frame,
        image_size=args.image_size,
        shuffle=False,
        num_wokers=0,
        random_caption=False,
        drop_last=use_pairwise_objective,
        generator=val_generator,
    )
    if len(train_loader) == 0 or len(val_loader) == 0:
        raise RuntimeError("the selected batch size/drop_last policy produced an empty loader")

    dataset_contract = build_dataset_contract({"train": args.ann_train, "val": args.ann_val})
    contract = training_contract(args)
    print(f"[objective] {contract}")
    print(f"[dataset] {dataset_contract}")

    device = args.device
    model = VideoCaptionModel(
        vit_name=args.vit_name,
        gpt2_name=args.gpt2_name,
        cond_mode=args.cond_mode,
        prefix_len=args.prefix_len,
        freeze_vit=True,
        unfreeze_last=0,
        vit_enable_torch_compile=False,
    ).to(device).train()

    resume_checkpoint = None
    if args.resume_ckpt:
        resume_path = Path(args.resume_ckpt)
        resume_checkpoint = load_checkpoint(resume_path, map_location=device)
        restore_model_state(model, resume_checkpoint, visual_only=args.resume_visual_only)
        scope = "visual path only" if args.resume_visual_only else "all model weights"
        print(f"[resume] restored {scope} from {resume_path}")

    if args.freeze_vit:
        for parameter in model.encoder.parameters():
            parameter.requires_grad = False
        encoder_projection = getattr(model.encoder, "proj", None)
        if isinstance(encoder_projection, nn.Module):
            for parameter in encoder_projection.parameters():
                parameter.requires_grad = True

    gpt2 = getattr(model.decoder, "model", model.decoder)
    for parameter in gpt2.parameters():
        parameter.requires_grad = False

    train_blocks: List[tuple] = []
    encoder_projection = getattr(model.encoder, "proj", None)
    if isinstance(encoder_projection, nn.Module):
        train_blocks.append(("encoder.proj", encoder_projection, args.lr))

    if isinstance(model.proj, nn.Module):
        for parameter in model.proj.parameters():
            parameter.requires_grad = True
        train_blocks.append(("proj", model.proj, args.lr))

    mapper = getattr(model.decoder, "mapper", None)
    if isinstance(mapper, nn.Module):
        for parameter in mapper.parameters():
            parameter.requires_grad = True
        train_blocks.append(("decoder.mapper", mapper, args.lr))

    tail_parameters = []
    if args.unfreeze_gpt2_last > 0:
        for block in gpt2.transformer.h[-args.unfreeze_gpt2_last :]:
            for parameter in block.parameters():
                parameter.requires_grad = True
                tail_parameters.append(parameter)

    parameter_groups = []
    for _, module, learning_rate in train_blocks:
        parameters = [parameter for parameter in module.parameters() if parameter.requires_grad]
        if parameters:
            parameter_groups.append({"params": parameters, "lr": learning_rate})
    if tail_parameters:
        parameter_groups.append({"params": tail_parameters, "lr": args.lr_gpt2})

    print("[trainable groups]")
    for name, module, learning_rate in train_blocks:
        count = sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad)
        if count:
            print(f"  - {name:18s} | params={count:,} | lr={learning_rate}")
    if tail_parameters:
        print(
            f"  - gpt2_tail           | params={sum(p.numel() for p in tail_parameters):,} "
            f"| lr={args.lr_gpt2}"
        )
    total_trainable = sum(parameter.numel() for group in parameter_groups for parameter in group["params"])
    print(f"[trainable total] {total_trainable:,} params")
    if not parameter_groups:
        raise RuntimeError("no trainable projection/mapper parameters were found")

    optimizer = optim.AdamW(parameter_groups, weight_decay=0.01)
    scheduler = None
    start_epoch = 0
    resume_batch_in_epoch = 0
    best_val = float("inf")
    step = 0
    saved_resume_rng = None
    saved_epoch_generator_state = None
    resume_mid_epoch = False

    if args.resume_training_state:
        assert resume_checkpoint is not None
        validate_training_resume_contract(resume_checkpoint, contract, dataset_contract)
        progress = restore_training_state(resume_checkpoint, optimizer, scheduler)
        step = int(progress["step"])
        best_val = float(progress["best_val"])
        epoch_complete = bool(progress["epoch_complete"])
        start_epoch = int(progress["epoch"]) + (1 if epoch_complete else 0)
        resume_batch_in_epoch = 0 if epoch_complete else int(progress["batch_in_epoch"])
        loader_state = resume_checkpoint["loader_state"]
        if epoch_complete:
            train_generator.set_state(loader_state["current_generator_state"].cpu())
        else:
            saved_epoch_generator_state = loader_state["epoch_start_generator_state"].cpu()
            saved_resume_rng = resume_checkpoint["rng_state"]
            resume_mid_epoch = True
        print(
            f"[resume] training state step={step} start_epoch={start_epoch} "
            f"batch_in_epoch={resume_batch_in_epoch}"
        )

    objective_config = objective_kwargs(args)
    best_checkpoint_path = ckpt_dir / args.ckpt_name
    last_checkpoint_path = best_checkpoint_path.with_name(
        f"{best_checkpoint_path.stem}.last{best_checkpoint_path.suffix}"
    )
    if str(device).startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(device)

    stop_training = False
    for epoch in range(start_epoch, args.epochs):
        if resume_mid_epoch and epoch == start_epoch:
            assert saved_epoch_generator_state is not None and saved_resume_rng is not None
            train_generator.set_state(saved_epoch_generator_state)
            epoch_start_generator_state = saved_epoch_generator_state
            train_iterator = iter(train_loader)
            for _ in range(resume_batch_in_epoch):
                next(train_iterator)
            restore_rng_state(saved_resume_rng)
            batch_in_epoch = resume_batch_in_epoch
            resume_mid_epoch = False
        else:
            epoch_start_generator_state = train_generator.get_state()
            train_iterator = iter(train_loader)
            batch_in_epoch = 0

        for batch in train_iterator:
            video = batch["video"].to(device, non_blocking=True)
            captions = batch["caption_ids"].to(device, non_blocking=True)
            metrics = compute_caption_alignment_objective(
                model,
                video,
                captions,
                tokenizer.pad_token_id,
                video_ids=batch["video_id"],
                **objective_config,
            )

            optimizer.zero_grad()
            metrics["loss"].backward()
            optimizer.step()
            step += 1
            batch_in_epoch += 1

            numeric_metrics = {name: float(metrics[name].detach().item()) for name in METRIC_NAMES}
            if step == 1 or step % 10 == 0:
                print(f"[train] step {step:05d} | {format_metrics(numeric_metrics)}")
            append_metrics(events_csv, step, numeric_metrics)

            if args.val_every > 0 and step % args.val_every == 0:
                val_metrics = evaluate(
                    model,
                    val_loader,
                    tokenizer.pad_token_id,
                    device,
                    objective_config,
                )
                print(f"[val] step {step:05d} | {format_metrics(val_metrics)}")
                append_metrics(val_csv, step, val_metrics)
                if val_metrics["loss"] < best_val:
                    best_val = val_metrics["loss"]
                    save_training_checkpoint(
                        best_checkpoint_path,
                        model=model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        progress=checkpoint_progress(
                            step=step,
                            epoch=epoch,
                            batch_in_epoch=batch_in_epoch,
                            epoch_complete=False,
                            best_val=best_val,
                        ),
                        args=vars(args),
                        training_contract=contract,
                        dataset_contract=dataset_contract,
                        loader_state={
                            "current_generator_state": train_generator.get_state(),
                            "epoch_start_generator_state": epoch_start_generator_state,
                            "num_workers": 0,
                        },
                    )
                    print(f"[ckpt] saved best -> {best_checkpoint_path} (val={best_val:.4f})")

            if args.max_steps > 0 and step >= args.max_steps:
                stop_training = True
                break

        epoch_complete = not stop_training and batch_in_epoch == len(train_loader)
        progress = checkpoint_progress(
            step=step,
            epoch=epoch,
            batch_in_epoch=batch_in_epoch,
            epoch_complete=epoch_complete,
            best_val=best_val,
        )
        loader_state = {
            "current_generator_state": train_generator.get_state(),
            "epoch_start_generator_state": epoch_start_generator_state,
            "num_workers": 0,
        }
        if args.smoke_test:
            print("[val][smoke] skipped by explicit --smoke_test contract")
        else:
            val_metrics = evaluate(
                model,
                val_loader,
                tokenizer.pad_token_id,
                device,
                objective_config,
            )
            print(f"[val][epoch end] epoch {epoch} | {format_metrics(val_metrics)}")
            append_metrics(val_csv, step, val_metrics)
            progress["best_val"] = min(best_val, val_metrics["loss"])
            if val_metrics["loss"] < best_val:
                best_val = val_metrics["loss"]
                progress["best_val"] = best_val
                save_training_checkpoint(
                    best_checkpoint_path,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    progress=progress,
                    args=vars(args),
                    training_contract=contract,
                    dataset_contract=dataset_contract,
                    loader_state=loader_state,
                )
                print(f"[ckpt][epoch end] saved best -> {best_checkpoint_path} (val={best_val:.4f})")

        save_training_checkpoint(
            last_checkpoint_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            progress=progress,
            args=vars(args),
            training_contract=contract,
            dataset_contract=dataset_contract,
            loader_state=loader_state,
        )
        print(f"[ckpt][epoch end] saved resumable last -> {last_checkpoint_path}")
        if stop_training:
            break

    if str(device).startswith("cuda"):
        peak_allocated = torch.cuda.max_memory_allocated(device) / (1024**2)
        peak_reserved = torch.cuda.max_memory_reserved(device) / (1024**2)
        print(
            f"[cuda memory] peak_allocated_mb={peak_allocated:.1f} "
            f"peak_reserved_mb={peak_reserved:.1f}"
        )
    print("[OK] mapper fine-tuning finished.")


if __name__ == "__main__":
    main()
