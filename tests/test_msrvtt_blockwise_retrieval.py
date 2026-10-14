"""Check exact blockwise ranks against a small independent sorted reference."""
import unittest

import torch

from scripts.msrvtt_blockwise_retrieval import evaluate_blockwise


def reference_rank(values: list[float], positives: set[int]) -> int:
    ordering = sorted(range(len(values)), key=lambda index: (-values[index], index))
    return next(rank for rank, index in enumerate(ordering, 1) if index in positives)


class BlockwiseRetrievalTest(unittest.TestCase):
    def test_ties_and_irregular_caption_groups(self) -> None:
        # Identical vectors create ties, and the final video has one caption.
        videos = torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        texts = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0], [1.0, 0.0]])
        gold = torch.tensor([0, 0, 1, 1, 2])
        result = evaluate_blockwise(videos, texts, gold, device="cpu", text_batch_size=2,
                                    video_batch_size=2)
        scores = videos @ texts.T
        t2v = [reference_rank(scores[:, query].tolist(), {int(gold[query])})
               for query in range(len(gold))]
        v2t = [reference_rank(scores[video].tolist(),
                              {query for query in range(len(gold)) if gold[query] == video})
               for video in range(len(videos))]
        self.assertAlmostEqual(result["true_labels"]["text_to_video"]["r_at_1"], sum(x <= 1 for x in t2v) / len(t2v), places=6)
        self.assertAlmostEqual(result["true_labels"]["video_to_text"]["r_at_1"], sum(x <= 1 for x in v2t) / len(v2t), places=6)
        self.assertAlmostEqual(result["true_labels"]["text_to_video"]["mean_rank"], sum(t2v) / len(t2v), places=6)
        self.assertAlmostEqual(result["true_labels"]["video_to_text"]["mean_rank"], sum(v2t) / len(v2t), places=6)
        self.assertEqual(result["candidate_videos"], 3)
        self.assertEqual(result["caption_queries"], 5)


if __name__ == "__main__":
    unittest.main()
