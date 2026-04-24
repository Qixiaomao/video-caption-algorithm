from __future__ import annotations

import argparse
import json
import sys
from contextlib import nullcontext
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

import backend_config
from core.config import InferenceConfig, ViTOptimizeConfig
from core.env import assert_core_runtime_ready
from core.models.model_loader import load_caption_model
from core.preprocessing.frame_loader import load_video_tensor

MB = 1024.0 * 1024.0


class NvtxRange:
    def __init__(self, name: str):
        self.name = name

    def __enter__(self):
        torch.cuda.nvtx.range_push(self.name)
        return self

    def __exit__(self, exc_type, exc, tb):
        torch.cuda.nvtx.range_pop()
        return False


def autocast_context(device: torch.device, enabled: bool):
    if not enabled or device.type != "cuda":
        return nullcontext()
    return torch.autocast(device_type="cuda", dtype=torch.float16)


@torch.inference_mode()
def run_one_profile(
    model,
    config: InferenceConfig,
    frames_dir: str,
    prompt: str,
    max_new_tokens: int,
    use_autocast_fp16: bool,
):
    profile_device = torch.device(config.device)
    with NvtxRange("Inference_Once"):
        with NvtxRange("Preprocessing"):
            video = load_video_tensor(
                frames_dir,
                num_frames=config.num_frames,
                image_size=config.image_size,
                device=config.device,
            )

        with NvtxRange("ViT_Encoder"):
            with autocast_context(profile_device, use_autocast_fp16):
                encoder_out = model.encoder(video)

        with NvtxRange("Cross_Modal_Alignment"):
            with autocast_context(profile_device, use_autocast_fp16):
                emb = model.proj(encoder_out)
                if emb.dim() == 2:
                    emb = emb.unsqueeze(1)
                if config.ln_scale is not None and config.ln_scale > 0:
                    emb = torch.nn.functional.layer_norm(emb, emb.shape[-1:]) * config.ln_scale
                if config.in_weight is not None and config.in_weight > 0:
                    emb = emb * config.in_weight

                if model.decoder.cond_mode == "prefix":
                    hidden = model.decoder.model.config.n_embd
                    prefix_embeds = model.decoder.mapper(emb).view(emb.size(0), model.decoder.prefix_len, hidden)
                else:
                    prefix_embeds = model.decoder.mapper(emb)
                    if prefix_embeds.dim() == 2:
                        prefix_embeds = prefix_embeds.unsqueeze(1)

        decoder = model.decoder
        tokenizer = decoder.tokenizer
        gpt2 = decoder.model
        device = prefix_embeds.device

        if prompt:
            prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
        else:
            bos_token_id = tokenizer.bos_token_id
            if bos_token_id is None:
                bos_token_id = tokenizer.eos_token_id
            prompt_ids = torch.tensor([[bos_token_id]], dtype=torch.long, device=device)

        with autocast_context(device, use_autocast_fp16):
            prompt_embeds = gpt2.transformer.wte(prompt_ids)
            next_input_embeds = torch.cat([prefix_embeds, prompt_embeds], dim=1)
        running_attention_mask = torch.ones(next_input_embeds.shape[:2], dtype=torch.long, device=device)
        past_key_values = None
        generated_tokens: list[int] = []

        with NvtxRange("GPT2_Decoder_Step"):
            for step_idx in range(max_new_tokens):
                with NvtxRange(f"GPT2_Decoder_Step/token_{step_idx:02d}"):
                    with autocast_context(device, use_autocast_fp16):
                        outputs = gpt2(
                            inputs_embeds=next_input_embeds,
                            attention_mask=running_attention_mask,
                            past_key_values=past_key_values,
                            use_cache=True,
                            return_dict=True,
                        )

                logits = outputs.logits[:, -1, :]
                next_token = torch.argmax(logits, dim=-1)
                token_id = int(next_token.item())
                generated_tokens.append(token_id)
                past_key_values = outputs.past_key_values

                if token_id == tokenizer.eos_token_id:
                    break

                with autocast_context(device, use_autocast_fp16):
                    next_input_embeds = gpt2.transformer.wte(next_token).unsqueeze(1)
                running_attention_mask = torch.cat(
                    [
                        running_attention_mask,
                        torch.ones((next_input_embeds.shape[0], 1), dtype=torch.long, device=device),
                    ],
                    dim=1,
                )

    text = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
    return {"caption": text, "generated_tokens": len(generated_tokens)}


def get_env(device: torch.device) -> dict[str, object]:
    props = torch.cuda.get_device_properties(device)
    return {
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "device": torch.cuda.get_device_name(device),
        "compute_capability": f"{props.major}.{props.minor}",
        "total_vram_mb": props.total_memory / MB,
    }


def build_config(args: argparse.Namespace) -> InferenceConfig:
    return InferenceConfig(
        ckpt=args.ckpt,
        vit_name=args.vit_name,
        gpt2_name=args.gpt2_name,
        prefix_len=args.prefix_len,
        num_frames=args.num_frames,
        image_size=args.image_size,
        ln_scale=args.ln_scale,
        in_weight=args.in_weight,
        device=args.device,
        backend="torch",
        vit_opt=ViTOptimizeConfig(
            enable_fp16=backend_config.VIT_ENABLE_FP16,
            enable_attention_fastpath=backend_config.VIT_ENABLE_ATTENTION_FASTPATH,
            prefer_channels_last=backend_config.VIT_PREFER_CHANNELS_LAST,
            enable_torch_compile=backend_config.VIT_ENABLE_TORCH_COMPILE,
            torch_compile_mode=backend_config.VIT_TORCH_COMPILE_MODE,
            enable_mlp_bias_gelu_fusion=backend_config.VIT_ENABLE_MLP_BIAS_GELU_FUSION,
            enable_residual_layernorm_fusion=backend_config.VIT_ENABLE_RESIDUAL_LAYERNORM_FUSION,
            enable_inplace_residual_add_fusion=backend_config.VIT_ENABLE_INPLACE_RESIDUAL_ADD_FUSION,
            enable_cupy_fused_pool=backend_config.VIT_ENABLE_CUPY_FUSED_POOL,
            cupy_pool_force_fp16=backend_config.VIT_CUPY_POOL_FORCE_FP16,
        ),
        use_cupy_prefix_projector=backend_config.USE_CUPY_PREFIX_PROJECTOR,
        cupy_prefix_force_fp16=backend_config.CUPY_PREFIX_FORCE_FP16,
    )


def parse_args():
    parser = argparse.ArgumentParser("Single-run deterministic Nsight Systems profile entrypoint")
    parser.add_argument("--frames-dir", required=True, help="Directory that contains frame_*.jpg")
    parser.add_argument("--ckpt", default=str(backend_config.CKPT_PATH), help="Path to caption model checkpoint")
    parser.add_argument("--device", default="cuda", help="CUDA device, e.g. cuda or cuda:0")
    parser.add_argument("--vit-name", default=backend_config.VIT_NAME)
    parser.add_argument("--gpt2-name", default=backend_config.GPT2_NAME)
    parser.add_argument("--prefix-len", type=int, default=backend_config.PREFIX_LEN)
    parser.add_argument("--num-frames", type=int, default=backend_config.NUM_FRAMES)
    parser.add_argument("--image-size", type=int, default=backend_config.IMAGE_SIZE)
    parser.add_argument("--ln-scale", type=float, default=backend_config.LN_SCALE)
    parser.add_argument("--in-weight", type=float, default=backend_config.IN_WEIGHT)
    parser.add_argument("--prompt", default=backend_config.PROMPT3)
    parser.add_argument("--max-new-tokens", type=int, default=24)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--use-autocast-fp16", action="store_true", help="Enable CUDA autocast(fp16) during the profiled run")
    parser.add_argument("--export-json", default="", help="Optional JSON output path for single-run metadata")
    return parser.parse_args()


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for Nsight profiling.")

    assert_core_runtime_ready(device=args.device, require_cupy=False)
    config = build_config(args)
    model = load_caption_model(config)
    device = torch.device(args.device)
    env = get_env(device)

    print("=== Nsight Profile Environment ===")
    print(f"torch: {env['torch']}")
    print(f"torch.version.cuda: {env['torch_cuda']}")
    print(f"device: {env['device']}")
    print(f"frames_dir: {args.frames_dir}")
    print(f"ckpt: {args.ckpt}")
    print(f"precision: {'fp16_autocast' if args.use_autocast_fp16 else 'fp32'}")
    if args.export_json:
        print(f"export_json: {args.export_json}")
    print()

    for _ in range(args.warmup):
        _ = run_one_profile(model, config, args.frames_dir, args.prompt, args.max_new_tokens, args.use_autocast_fp16)
        torch.cuda.synchronize(device)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    result = run_one_profile(model, config, args.frames_dir, args.prompt, args.max_new_tokens, args.use_autocast_fp16)
    torch.cuda.synchronize(device)
    peak_memory_mb = torch.cuda.max_memory_allocated(device) / MB

    print(f"caption: {result['caption']!r}")
    print(f"generated_tokens: {result['generated_tokens']}")
    print(f"peak_memory_mb: {peak_memory_mb:.1f}")

    if args.export_json:
        output_path = Path(args.export_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as fh:
            json.dump(
                {
                    "env": env,
                    "config": {
                        "frames_dir": args.frames_dir,
                        "ckpt": args.ckpt,
                        "device": args.device,
                        "prompt": args.prompt,
                        "warmup": args.warmup,
                        "max_new_tokens": args.max_new_tokens,
                        "precision": "fp16_autocast" if args.use_autocast_fp16 else "fp32",
                        "num_frames": args.num_frames,
                        "image_size": args.image_size,
                        "prefix_len": args.prefix_len,
                        "ln_scale": args.ln_scale,
                        "in_weight": args.in_weight,
                    },
                    "result": {
                        "caption": result["caption"],
                        "generated_tokens": result["generated_tokens"],
                        "peak_memory_mb": peak_memory_mb,
                    },
                },
                fh,
                ensure_ascii=False,
                indent=2,
            )
        print(f"profile JSON exported to: {args.export_json}")


if __name__ == "__main__":
    main()





