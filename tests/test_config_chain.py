from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

import torch.nn as nn

from core.config import InferenceConfig, ViTOptimizeConfig
from core.models.model_loader import load_torch_caption_model
from server.schemas import InferRequest
from server.services.inference_service import request_to_config
from src.models.caption_model import VideoCaptionModel
from src.models.video_encoder import ViTConfig, ViTFrameEncoder


class FakeAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.fused_attn = True

    def forward(self, x, attn_mask=None):
        return x


class FakeMlp(nn.Module):
    def __init__(self):
        super().__init__()
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(x)


class FakeBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = FakeAttention()
        self.mlp = FakeMlp()
        self.norm1 = nn.Identity()
        self.norm2 = nn.Identity()
        self.ls1 = nn.Identity()
        self.ls2 = nn.Identity()

    def forward(self, x, attn_mask=None):
        return x


def bare_encoder(cfg: ViTConfig) -> ViTFrameEncoder:
    encoder = object.__new__(ViTFrameEncoder)
    nn.Module.__init__(encoder)
    encoder.cfg = cfg
    encoder.blocks = nn.ModuleList([FakeBlock()])
    encoder._optimization_status = {}
    return encoder


class EncoderSwitchTests(unittest.TestCase):
    def test_attention_false_explicitly_disables_timm_default(self):
        encoder = bare_encoder(ViTConfig(enable_attention_fastpath=False, enable_torch_compile=False))

        encoder._configure_attention_fastpath()

        self.assertFalse(encoder.blocks[0].attn.fused_attn)
        self.assertTrue(encoder.get_optimization_status()["attention_fastpath"]["request_honored"])

    def test_inplace_residual_is_independent_from_residual_layernorm_flag(self):
        encoder = bare_encoder(
            ViTConfig(
                enable_residual_layernorm_fusion=False,
                enable_inplace_residual_add_fusion=True,
                enable_torch_compile=False,
            )
        )

        encoder._configure_residual_layernorm_fusion()
        encoder._configure_inplace_residual_add()

        status = encoder.get_optimization_status()
        self.assertFalse(status["residual_layernorm_fusion"]["active"])
        self.assertTrue(status["inplace_residual_add"]["active"])

    def test_unsupported_requested_residual_layernorm_is_reported(self):
        encoder = bare_encoder(
            ViTConfig(
                enable_residual_layernorm_fusion=True,
                enable_inplace_residual_add_fusion=False,
                enable_torch_compile=False,
            )
        )
        encoder._configure_residual_layernorm_fusion()

        self.assertIn(
            "residual_layernorm_fusion: installed_backbone_has_no_add_norm_fusion_flag",
            encoder.get_optimization_issues(),
        )


class EndToEndConfigChainTests(unittest.TestCase):
    @patch("core.models.caption_model.VideoCaptionModel")
    @patch("core.models.model_loader._load_checkpoint", return_value={})
    def test_model_loader_passes_cuda_graph_to_caption_model(self, _load_checkpoint, model_cls):
        model = MagicMock()
        model.to.return_value = model
        model.eval.return_value = model
        model.load_state_dict.return_value = ([], [])
        model_cls.return_value = model
        config = InferenceConfig(
            ckpt="checkpoint.pt",
            vit_opt=ViTOptimizeConfig(enable_cuda_graph=True),
        )

        load_torch_caption_model(config)

        self.assertTrue(model_cls.call_args.kwargs["vit_enable_cuda_graph"])

    @patch("src.models.caption_model.GPT2TextDecoder", return_value=nn.Identity())
    @patch("src.models.caption_model.build_vit_encoder", return_value=nn.Identity())
    def test_caption_model_passes_cuda_graph_to_encoder(self, build_encoder, _decoder):
        VideoCaptionModel(vit_enable_cuda_graph=True)

        self.assertTrue(build_encoder.call_args.kwargs["enable_cuda_graph"])

    def test_api_request_preserves_inplace_and_cuda_graph_flags(self):
        request = InferRequest(
            frames_dir="frames",
            vit_enable_inplace_residual_add_fusion=False,
            vit_enable_cuda_graph=True,
        )

        config = request_to_config(request)

        self.assertFalse(config.vit_opt.enable_inplace_residual_add_fusion)
        self.assertTrue(config.vit_opt.enable_cuda_graph)


if __name__ == "__main__":
    unittest.main()
