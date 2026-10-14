from __future__ import annotations

import unittest

import torch
import torch.nn as nn

from src.cli.train_caption_mapper import build_argparser, training_contract, validate_args
from src.training.objectives import compute_caption_alignment_objective
from src.training.cpr import (
    build_cpr_negative_indices,
    conditional_prefix_ranking_loss,
    length_normalized_caption_nll,
)


class TinyFrozenTransformer(nn.Module):
    def __init__(self, vocab_size: int, hidden_size: int):
        super().__init__()
        self.wte = nn.Embedding(vocab_size, hidden_size)
        for parameter in self.parameters():
            parameter.requires_grad = False


class TinyFrozenLanguageModel(nn.Module):
    def __init__(self, vocab_size: int, hidden_size: int):
        super().__init__()
        self.transformer = TinyFrozenTransformer(vocab_size, hidden_size)


class TinyCPRDecoder(nn.Module):
    def __init__(self, hidden_size: int = 4, vocab_size: int = 10, prefix_len: int = 2):
        super().__init__()
        self.mapper = nn.Linear(hidden_size, hidden_size * prefix_len, bias=False)
        self.model = TinyFrozenLanguageModel(vocab_size, hidden_size)
        self.prefix_len = prefix_len
        self.hidden_size = hidden_size
        self.register_buffer(
            "frozen_readout",
            torch.linspace(-1.0, 1.0, steps=vocab_size * hidden_size).view(
                vocab_size, hidden_size
            ),
        )
        self.extra_forward_calls = 0

    def build_visual_prefix(self, video_embedding: torch.Tensor) -> torch.Tensor:
        return self.mapper(video_embedding).view(
            video_embedding.size(0), self.prefix_len, self.hidden_size
        )

    def forward(
        self,
        video_embedding: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
        visual_prefix: torch.Tensor,
    ):
        del video_embedding, attention_mask, labels
        self.extra_forward_calls += 1
        prefix_logits = visual_prefix.mean(dim=1) @ self.frozen_readout.transpose(0, 1)
        if self.training:
            prefix_logits = prefix_logits + torch.rand_like(prefix_logits) * 0.0
        sequence_length = self.prefix_len + input_ids.size(1)
        return {"loss": None, "logits": prefix_logits[:, None, :].expand(-1, sequence_length, -1)}


class TinyCPRModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Module()
        self.encoder.proj = nn.Linear(3, 4, bias=False)
        self.decoder = TinyCPRDecoder()
        self.return_visual_prefix_values = []

    def forward(
        self,
        video: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
        return_visual_prefix: bool = False,
    ):
        del attention_mask, labels
        self.return_visual_prefix_values.append(return_visual_prefix)
        video_embedding = self.encoder.proj(video)
        visual_prefix = self.decoder.build_visual_prefix(video_embedding)
        prefix_logits = visual_prefix.mean(dim=1) @ self.decoder.frozen_readout.transpose(0, 1)
        sequence_length = self.decoder.prefix_len + input_ids.size(1)
        logits = prefix_logits[:, None, :].expand(-1, sequence_length, -1)
        result = {"loss": logits.sum() * 0.0, "logits": logits}
        if return_visual_prefix:
            result["visual_prefix"] = visual_prefix
            result["video_embedding"] = video_embedding
        return result


class CPRNegativeSelectionTests(unittest.TestCase):
    def test_derangement_excludes_same_video_and_exact_caption(self):
        pad_id = 9
        captions = torch.tensor(
            [
                [1, 2, pad_id],
                [3, 4, pad_id],
                [1, 2, pad_id],
                [5, 6, pad_id],
            ]
        )
        video_ids = ["v0", "v0", "v2", "v3"]

        indices, eligible = build_cpr_negative_indices(captions, pad_id, video_ids)

        self.assertTrue(eligible.all().item())
        for row, candidate in enumerate(indices.tolist()):
            self.assertNotEqual(video_ids[row], video_ids[candidate])
            self.assertFalse(torch.equal(captions[row], captions[candidate]))

    def test_rows_without_a_valid_negative_are_skipped(self):
        captions = torch.tensor([[1, 9], [1, 9]])

        indices, eligible = build_cpr_negative_indices(
            captions, pad_id=9, video_ids=["v0", "v1"]
        )

        self.assertEqual(indices.tolist(), [-1, -1])
        self.assertEqual(eligible.tolist(), [False, False])


class CPRLossTests(unittest.TestCase):
    def test_satisfied_margin_has_zero_loss(self):
        loss = conditional_prefix_ranking_loss(
            torch.tensor([1.0, 2.0]),
            torch.tensor([1.2, 2.3]),
            margin=0.1,
        )

        self.assertEqual(loss.item(), 0.0)

    def test_violated_margin_has_expected_hinge_loss(self):
        loss = conditional_prefix_ranking_loss(
            torch.tensor([1.0, 2.0]),
            torch.tensor([1.05, 1.9]),
            margin=0.1,
        )

        self.assertAlmostEqual(loss.item(), 0.125, places=6)

    def test_length_normalization_ignores_eos_and_padding(self):
        pad_id = 4
        captions = torch.tensor([[1, 2, pad_id], [3, pad_id, pad_id]])
        vocab = 5
        logits = torch.zeros(2, 5, vocab)
        logits[0, 1, 1] = 2.0
        logits[0, 2, 2] = 2.0
        logits[1, 1, 3] = 2.0
        logits[1, 2:, :] = -100.0

        nll = length_normalized_caption_nll(logits, captions, pad_id)
        expected = -torch.log_softmax(torch.tensor([2.0, 0.0, 0.0, 0.0, 0.0]), 0)[0]

        self.assertTrue(torch.allclose(nll, expected.expand_as(nll)))


class CPRContractTests(unittest.TestCase):
    def test_cpr_requires_physical_negative_batch(self):
        parser = build_argparser()
        args = parser.parse_args(["--batch_size", "1", "--cpr_weight", "0.1"])

        with self.assertRaisesRegex(ValueError, "CPR requires"):
            validate_args(args)

    def test_contract_records_cpr_configuration(self):
        parser = build_argparser()
        args = parser.parse_args(
            ["--cpr_weight", "0.1", "--cpr_margin", "0.2", "--batch_size", "8"]
        )
        validate_args(args)

        contract = training_contract(args)

        self.assertEqual(contract["objective_version"], 4)
        self.assertEqual(contract["cpr_weight"], 0.1)
        self.assertEqual(contract["cpr_margin"], 0.2)
        self.assertEqual(contract["cpr_negative_policy"], "cyclic_derangement")


class CPRObjectiveIntegrationTests(unittest.TestCase):
    def test_zero_weight_keeps_the_existing_single_forward_trace(self):
        model = TinyCPRModel()
        result = compute_caption_alignment_objective(
            model,
            torch.randn(2, 3),
            torch.tensor([[1, 2, 9], [3, 4, 9]]),
            pad_id=9,
        )

        self.assertEqual(model.return_visual_prefix_values, [False])
        self.assertEqual(model.decoder.extra_forward_calls, 0)
        self.assertEqual(result["cpr_loss"].item(), 0.0)
        self.assertEqual(result["loss"].item(), result["caption_loss"].item())

    def test_zero_cpr_weight_keeps_the_infonce_forward_trace(self):
        model = TinyCPRModel()
        result = compute_caption_alignment_objective(
            model,
            torch.randn(4, 3),
            torch.tensor(
                [
                    [1, 2, 9],
                    [3, 4, 9],
                    [5, 6, 9],
                    [7, 8, 9],
                ]
            ),
            pad_id=9,
            alignment_objective="infonce",
            alignment_weight=0.1,
            video_ids=["v0", "v1", "v2", "v3"],
        )

        self.assertEqual(model.return_visual_prefix_values, [True])
        self.assertEqual(model.decoder.extra_forward_calls, 0)
        expected = result["caption_loss"] + 0.1 * result["alignment_loss"]
        self.assertTrue(torch.equal(result["loss"], expected))
        self.assertEqual(result["cpr_loss"].item(), 0.0)

    def test_cpr_gradients_reach_projection_and_mapper_only(self):
        torch.manual_seed(23)
        model = TinyCPRModel()
        result = compute_caption_alignment_objective(
            model,
            torch.randn(4, 3),
            torch.tensor(
                [
                    [1, 2, 9],
                    [3, 4, 9],
                    [5, 6, 9],
                    [7, 8, 9],
                ]
            ),
            pad_id=9,
            cpr_weight=0.1,
            cpr_margin=10.0,
            video_ids=["v0", "v1", "v2", "v3"],
        )
        result["loss"].backward()

        self.assertGreater(model.encoder.proj.weight.grad.abs().sum().item(), 0.0)
        self.assertGreater(model.decoder.mapper.weight.grad.abs().sum().item(), 0.0)
        self.assertIsNone(model.decoder.model.transformer.wte.weight.grad)
        self.assertEqual(model.decoder.extra_forward_calls, 2)

    def test_cpr_branch_does_not_advance_the_main_rng(self):
        model = TinyCPRModel()
        video = torch.randn(4, 3)
        captions = torch.tensor(
            [[1, 2, 9], [3, 4, 9], [5, 6, 9], [7, 8, 9]]
        )

        torch.manual_seed(101)
        baseline = compute_caption_alignment_objective(
            model, video, captions, pad_id=9
        )
        baseline["loss"].backward()
        baseline_rng = torch.get_rng_state().clone()

        model.zero_grad(set_to_none=True)
        torch.manual_seed(101)
        cpr = compute_caption_alignment_objective(
            model,
            video,
            captions,
            pad_id=9,
            cpr_weight=0.1,
            cpr_margin=10.0,
            video_ids=["v0", "v1", "v2", "v3"],
        )
        cpr["loss"].backward()
        cpr_rng = torch.get_rng_state().clone()

        self.assertTrue(torch.equal(baseline_rng, cpr_rng))


if __name__ == "__main__":
    unittest.main()
