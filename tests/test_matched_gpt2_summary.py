import copy
import unittest

from scripts.summarize_matched_gpt2_init import aggregate


class MatchedSummaryTests(unittest.TestCase):
    def fixture(self):
        videos = [{"video_id": str(i), "caption": "a person walks", "nll": 2.,
                   "references": [{"nll": 2., "token_nll": [1., 2., 3.]}]} for i in range(2)]
        shuffled = copy.deepcopy(videos)
        for v in shuffled:
            v["nll"] = 3.
            v["references"] = [{"nll": 3., "token_nll": [2., 3., 4.]}]
        return {"conditions": {
            "native": {"videos": videos, "caption": {"nonempty_rate": 1., "unique_rate": .5,
                                                         "largest_template_rate": 1.}},
            "shuffled": {"videos": shuffled}},
            "references": [[{"text": "a person walks"}] for _ in videos]}

    def test_reaggregate_eos_and_grounding(self):
        result = aggregate(self.fixture())
        self.assertEqual(result["native_nll"], 2.)
        self.assertEqual(result["native_eos_nll"], 3.)
        self.assertEqual(result["native_non_eos_nll"], 1.5)
        self.assertEqual(result["native_after_first_nll"], 2.5)
        self.assertEqual(result["grounding_delta"]["mean"], 1.)
        self.assertEqual(result["grounding_after_first_delta"]["mean"], 1.)
        self.assertFalse(result["semantic_diversity_gate_pass"])

    def test_reject_inconsistent_saved_loss(self):
        report = self.fixture()
        report["conditions"]["native"]["videos"][0]["references"][0]["nll"] = 5.
        with self.assertRaisesRegex(ValueError, "reference mean"):
            aggregate(report)


if __name__ == "__main__":
    unittest.main()
