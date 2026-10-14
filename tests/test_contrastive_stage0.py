from __future__ import annotations

import math
import unittest

import torch
import torch.nn as nn

from src.cli.train_caption_mapper import build_argparser, validate_args
from src.training.contrastive import (
    build_alignment_embeddings,
    exact_caption_positive_mask,
    symmetric_multi_positive_info_nce,
)
from src.training.objectives import compute_caption_alignment_objective


class TinyEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(3, 4, bias=False)

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        return self.proj(video)


class TinyTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.wte = nn.Embedding(32, 4)
        for parameter in self.wte.parameters():
            parameter.requires_grad = False


class TinyLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.transformer = TinyTransformer()


class TinyDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.mapper = nn.Linear(4, 8, bias=False)
        self.model = TinyLanguageModel()


class TinyCaptionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = TinyEncoder()
        self.decoder = TinyDecoder()

    def forward(
        self,
        video: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
        return_visual_prefix: bool = False,
    ):
        del input_ids, attention_mask, labels
        video_embedding = self.encoder(video)
        visual_prefix = self.decoder.mapper(video_embedding).view(video.size(0), 2, 4)
        result = {"loss": visual_prefix.sum() * 0.0}
        if return_visual_prefix:
            result["visual_prefix"] = visual_prefix
        return result


class MultiPositiveInfoNCETests(unittest.TestCase):
    def test_aligned_pairs_have_lower_loss_than_shuffled_pairs(self):
        video = torch.eye(4)
        text = torch.eye(4)

        aligned = symmetric_multi_positive_info_nce(video, text, temperature=0.1)
        shuffled = symmetric_multi_positive_info_nce(video, text.roll(1, dims=0), temperature=0.1)

        self.assertLess(aligned.item(), shuffled.item())

    def test_equal_embeddings_match_log_batch_baseline(self):
        batch_size = 4
        embeddings = torch.ones(batch_size, 3)

        loss = symmetric_multi_positive_info_nce(embeddings, embeddings, temperature=0.07)

        self.assertAlmostEqual(loss.item(), math.log(batch_size), places=5)

    def test_batch_size_one_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "physical batch size"):
            symmetric_multi_positive_info_nce(torch.ones(1, 3), torch.ones(1, 3))

    def test_exact_duplicate_captions_are_multi_positives(self):
        pad_id = 99
        caption_ids = torch.tensor(
            [
                [1, 2, pad_id, pad_id],
                [1, 2, pad_id, 7],
                [3, 4, pad_id, pad_id],
            ]
        )

        positive_mask = exact_caption_positive_mask(caption_ids, pad_id)

        self.assertTrue(positive_mask[0, 1].item())
        self.assertTrue(positive_mask[1, 0].item())
        self.assertFalse(positive_mask[0, 2].item())

    def test_duplicate_mask_removes_the_false_negative_penalty(self):
        video = torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        text = video.clone()
        caption_ids = torch.tensor([[1, 9], [1, 9], [2, 9]])
        multi_positive = exact_caption_positive_mask(caption_ids, pad_id=9)

        diagonal_loss = symmetric_multi_positive_info_nce(video, text, temperature=0.1)
        multi_positive_loss = symmetric_multi_positive_info_nce(
            video,
            text,
            temperature=0.1,
            positive_mask=multi_positive,
        )

        self.assertLess(multi_positive_loss.item(), diagonal_loss.item())

    def test_text_pooling_excludes_eos_and_every_token_after_it(self):
        visual_prefix = torch.tensor([[[1.0, 0.0], [1.0, 0.0]]])
        token_embeddings = torch.tensor([[[1.0, 0.0], [0.0, 1.0], [9.0, 9.0]]])
        caption_ids = torch.tensor([[5, 99, 7]])

        _, text = build_alignment_embeddings(
            visual_prefix,
            token_embeddings,
            caption_ids,
            pad_id=99,
        )

        self.assertTrue(torch.allclose(text, torch.tensor([[1.0, 0.0]])))


class ObjectiveGradientTests(unittest.TestCase):
    def test_infonce_gradients_reach_encoder_projection_and_mapper(self):
        torch.manual_seed(7)
        model = TinyCaptionModel()
        video = torch.randn(4, 3)
        caption_ids = torch.tensor(
            [
                [1, 2, 31],
                [3, 4, 31],
                [5, 6, 31],
                [7, 8, 31],
            ]
        )

        result = compute_caption_alignment_objective(
            model,
            video,
            caption_ids,
            pad_id=31,
            alignment_objective="infonce",
            alignment_weight=0.1,
            temperature=0.07,
            false_negative_policy="exact_caption",
        )
        result["loss"].backward()

        encoder_gradient = model.encoder.proj.weight.grad
        mapper_gradient = model.decoder.mapper.weight.grad
        self.assertIsNotNone(encoder_gradient)
        self.assertIsNotNone(mapper_gradient)
        self.assertGreater(encoder_gradient.abs().sum().item(), 0.0)
        self.assertGreater(mapper_gradient.abs().sum().item(), 0.0)
        self.assertIsNone(model.decoder.model.transformer.wte.weight.grad)


class SmokeContractTests(unittest.TestCase):
    def test_smoke_test_requires_bounded_steps_and_no_validation(self):
        parser = build_argparser()

        valid = parser.parse_args(["--smoke_test", "--max_steps", "2", "--val_every", "0"])
        validate_args(valid)

        invalid_steps = parser.parse_args(["--smoke_test", "--max_steps", "0", "--val_every", "0"])
        with self.assertRaisesRegex(ValueError, "max_steps"):
            validate_args(invalid_steps)

        invalid_validation = parser.parse_args(["--smoke_test", "--max_steps", "2"])
        with self.assertRaisesRegex(ValueError, "val_every"):
            validate_args(invalid_validation)


if __name__ == "__main__":
    unittest.main()
