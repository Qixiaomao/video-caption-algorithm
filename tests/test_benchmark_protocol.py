from __future__ import annotations

import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

import torch

from core.scripts.benchmark_baseline import (
    build_decoder_context,
    build_config,
    build_reproducibility_metadata,
    resolve_batch_frame_dirs,
    select_greedy_token,
    validate_decoder_outputs,
)


def benchmark_args(**overrides):
    values = {
        "ckpt": "checkpoint.pt",
        "vit_name": "vit_base_patch16_224",
        "gpt2_name": "gpt2",
        "prefix_len": 4,
        "num_frames": 8,
        "image_size": 224,
        "ln_scale": 0.6,
        "in_weight": 0.4,
        "device": "cuda",
        "vit_fp16": False,
        "vit_attention_fastpath": False,
        "vit_channels_last": False,
        "vit_compile": False,
        "vit_compile_mode": "default",
        "vit_mlp_gelu_approximation": False,
        "vit_residual_layernorm_fusion": False,
        "vit_inplace_residual_add": False,
        "vit_cupy_fused_pool": False,
        "vit_cupy_pool_force_fp16": False,
        "vit_cuda_graph": False,
        "cupy_prefix_projector": False,
        "cupy_prefix_force_fp16": False,
        "prompt": "Write a short caption:",
        "min_new_tokens": 8,
        "max_new_tokens": 24,
        "profile_token_steps": False,
        "allow_repeated_input": False,
    }
    values.update(overrides)
    return Namespace(**values)


class BatchInputTests(unittest.TestCase):
    def test_distinct_batch_uses_different_video_directories(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            for video_id in ("video_b", "video_a", "video_c"):
                video_dir = root / video_id
                video_dir.mkdir()
                (video_dir / "frame_0001.jpg").touch()

            resolved = resolve_batch_frame_dirs(root, batch_size=2)

            self.assertEqual([path.name for path in resolved], ["video_a", "video_b"])
            self.assertEqual(len(set(resolved)), 2)

    def test_single_video_cannot_silently_fill_larger_batch(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "frame_0001.jpg").touch()

            with self.assertRaisesRegex(ValueError, "distinct videos"):
                resolve_batch_frame_dirs(root, batch_size=2)

            repeated = resolve_batch_frame_dirs(root, batch_size=2, allow_repeated_input=True)
            self.assertEqual(repeated, [root, root])


class DecodeProtocolTests(unittest.TestCase):
    def test_empty_prompt_uses_visual_prefix_without_synthetic_bos(self):
        prefix = torch.randn(2, 4, 8)

        result = build_decoder_context(object(), prefix, prompt="")

        self.assertIs(result, prefix)

    def test_eos_is_masked_until_minimum_new_tokens(self):
        logits = torch.tensor([[1.0, 2.0, 9.0]])

        before_minimum = select_greedy_token(logits, eos_token_id=2, step_index=0, min_new_tokens=2)
        after_minimum = select_greedy_token(logits, eos_token_id=2, step_index=2, min_new_tokens=2)

        self.assertEqual(before_minimum.item(), 1)
        self.assertEqual(after_minimum.item(), 2)

    def test_invalid_caption_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "latency sample is rejected"):
            validate_decoder_outputs([""], [1], min_new_tokens=8)

    def test_valid_caption_passes(self):
        validate_decoder_outputs(["a person is cooking food"], [8], min_new_tokens=8)


class ExplicitConfigTests(unittest.TestCase):
    def test_build_config_uses_cli_switches_including_cuda_graph(self):
        args = benchmark_args(
            vit_fp16=True,
            vit_attention_fastpath=False,
            vit_compile=False,
            vit_cuda_graph=True,
            vit_inplace_residual_add=False,
        )

        config = build_config(args)

        self.assertTrue(config.vit_opt.enable_fp16)
        self.assertFalse(config.vit_opt.enable_attention_fastpath)
        self.assertFalse(config.vit_opt.enable_torch_compile)
        self.assertTrue(config.vit_opt.enable_cuda_graph)
        self.assertFalse(config.vit_opt.enable_inplace_residual_add_fusion)

    def test_reproducibility_metadata_records_config_hash_and_real_inputs(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            checkpoint = root / "checkpoint.pt"
            checkpoint.write_bytes(b"frozen checkpoint fixture")
            video_dir = root / "video_001"
            video_dir.mkdir()
            (video_dir / "frame_0001.jpg").touch()
            args = benchmark_args(ckpt=str(checkpoint), vit_cuda_graph=True)
            config = build_config(args)

            result = build_reproducibility_metadata(config, args, {1: [video_dir]})

            self.assertEqual(result["protocol_version"], 2)
            self.assertEqual(result["input_policy"], "distinct_videos")
            self.assertEqual(result["input_batches"]["1"][0]["video_id"], "video_001")
            self.assertEqual(len(result["checkpoint"]["sha256"]), 64)
            self.assertTrue(result["inference_config"]["vit_opt"]["enable_cuda_graph"])


if __name__ == "__main__":
    unittest.main()
