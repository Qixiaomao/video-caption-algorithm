from __future__ import annotations

import unittest

import torch

from scripts.eval_retrieval import (
    deterministic_derangement,
    evaluate_scores,
    multi_target_retrieval_metrics,
    single_target_retrieval_metrics,
)


class RetrievalMetricTests(unittest.TestCase):
    def test_single_target_perfect_ranking(self):
        scores = torch.eye(4)
        metrics = single_target_retrieval_metrics(scores, torch.arange(4))

        self.assertEqual(metrics["r_at_1"], 1.0)
        self.assertEqual(metrics["median_rank"], 1.0)

    def test_multi_target_uses_first_correct_target_rank(self):
        scores = torch.tensor([[0.9, 0.8, 0.1], [0.1, 0.2, 0.9]])
        positives = torch.tensor([[True, True, False], [False, False, True]])

        metrics = multi_target_retrieval_metrics(scores, positives)

        self.assertEqual(metrics["r_at_1"], 1.0)

    def test_derangement_has_no_fixed_labels(self):
        permutation = deterministic_derangement(20, seed=123)

        self.assertFalse(permutation.eq(torch.arange(20)).any().item())

    def test_perfect_embeddings_beat_shuffled_control(self):
        embeddings = torch.eye(8)

        result = evaluate_scores(
            embeddings,
            embeddings,
            torch.arange(8),
            seed=123,
        )

        true_r1 = result["true_labels"]["text_to_video"]["r_at_1"]
        shuffled_r1 = result["shuffled_label_control"]["text_to_video"]["r_at_1"]
        self.assertEqual(true_r1, 1.0)
        self.assertLess(shuffled_r1, true_r1)
        self.assertGreater(result["similarity"]["gap"], 0.0)


if __name__ == "__main__":
    unittest.main()
