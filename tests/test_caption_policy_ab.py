import json
import tempfile
import unittest
from pathlib import Path

from scripts.run_caption_policy_ab import (
    compare_traces,
    make_batch_schedule,
    make_reference_orders,
    reference_seed,
    token_f1,
)
from scripts.run_overfit16 import SELECTED, TRAIN


class CaptionPolicyABTests(unittest.TestCase):
    @unittest.skipUnless(TRAIN.is_file(), "requires local MSVD annotations")
    def test_pinned_subset_has_668_references(self):
        records = {record["video_id"]: record
                   for record in json.loads(TRAIN.read_text(encoding="utf-8"))}
        self.assertEqual(sum(len(records[video_id]["captions"])
                             for video_id, _ in SELECTED), 668)

    def test_reference_seed_and_orders_are_stable_and_complete(self):
        records = {"a": {"captions": ["x", "y", "z"]},
                   "b": {"captions": ["p", "q"]}}
        import scripts.run_caption_policy_ab as module
        original = module.SELECTED
        try:
            module.SELECTED = (("a", "x"), ("b", "p"))
            first = make_reference_orders(records)
            second = make_reference_orders(records)
            self.assertEqual(first, second)
            self.assertEqual(sorted(first["a"]), [0, 1, 2])
            self.assertEqual(sorted(first["b"]), [0, 1])
            self.assertEqual(reference_seed("a"), reference_seed("a"))
            self.assertNotEqual(reference_seed("a"), reference_seed("b"))
        finally:
            module.SELECTED = original

    def test_batch_schedule_visits_every_video_once_per_epoch(self):
        schedule = make_batch_schedule(3)
        self.assertEqual(schedule, make_batch_schedule(3))
        for epoch in schedule:
            self.assertEqual(sorted(epoch[0] + epoch[1]), list(range(16)))

    def test_token_f1(self):
        self.assertEqual(token_f1("A dog runs", "a dog runs"), 1.0)
        self.assertEqual(token_f1("", "a dog"), 0.0)
        self.assertAlmostEqual(token_f1("a dog", "a cat"), 0.5)

    def test_trace_comparison_ignores_only_caption_treatment(self):
        common = {"step": 1, "epoch": 0, "indices": [0], "video_ids": ["v"],
                  "visual_features_sha256": "same", "python_rng_sha256": "same",
                  "torch_cpu_rng_sha256": "same", "torch_cuda_rng_sha256": ["same"]}
        left = dict(common, policy="A_fixed", target_texts=["a"], caption_ids_sha256="a")
        right = dict(common, policy="B_multi", target_texts=["b"], caption_ids_sha256="b")
        with tempfile.TemporaryDirectory() as directory:
            a = Path(directory) / "a.jsonl"
            b = Path(directory) / "b.jsonl"
            a.write_text(json.dumps(left) + "\n", encoding="utf-8")
            b.write_text(json.dumps(right) + "\n", encoding="utf-8")
            self.assertTrue(compare_traces(a, b, 1)["all_non_treatment_inputs_and_rng_identical"])
            right["visual_features_sha256"] = "changed"
            b.write_text(json.dumps(right) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "visual_features"):
                compare_traces(a, b, 1)


if __name__ == "__main__":
    unittest.main()
