import math
import unittest

from scripts.caption_metrics import METRIC_NAMES, compute_caption_metrics, validate_caption_inputs
from scripts.run_test670_evaluation import (
    CHECKPOINTS,
    DECODE,
    mean_sd,
    summarize_results,
    validate_checkpoint_payload,
)


class CaptionMetricTests(unittest.TestCase):
    def setUp(self):
        self.records = [
            {"video_id": "v0", "captions": ["a man plays guitar", "a person plays guitar"]},
            {"video_id": "v1", "captions": ["a dog runs outside", "a dog is running"]},
        ]
        self.predictions = [
            {"video_id": "v0", "caption": "a man plays guitar"},
            {"video_id": "v1", "caption": "a dog runs outside"},
        ]

    def test_input_validation_preserves_order(self):
        video_ids, gts, res = validate_caption_inputs(self.records, self.predictions)
        self.assertEqual(video_ids, ["v0", "v1"])
        self.assertEqual(len(gts[0]), 2)
        self.assertEqual(res[1], [{"caption": "a dog runs outside"}])

    def test_input_validation_rejects_reordered_predictions(self):
        with self.assertRaisesRegex(ValueError, "order"):
            validate_caption_inputs(self.records, list(reversed(self.predictions)))

    def test_input_validation_rejects_empty_prediction(self):
        bad = [dict(item) for item in self.predictions]
        bad[0]["caption"] = " "
        with self.assertRaisesRegex(ValueError, "empty prediction"):
            validate_caption_inputs(self.records, bad)

    def test_metric_wrapper_returns_locked_metric_set(self):
        result = compute_caption_metrics(self.records, self.predictions)
        self.assertEqual(tuple(result["corpus_raw"]), METRIC_NAMES)
        self.assertEqual(result["implementation"]["version"], "1.2")
        self.assertEqual(len(result["per_video_raw"]), 2)
        for name in METRIC_NAMES:
            self.assertTrue(math.isfinite(result["corpus_raw"][name]))
            self.assertAlmostEqual(
                result["corpus_x100"][name], result["corpus_raw"][name] * 100.0
            )


class Test670ProtocolTests(unittest.TestCase):
    def test_registry_has_all_six_unique_models(self):
        self.assertEqual(len(CHECKPOINTS), 6)
        self.assertEqual(
            {(item["seed"], item["arm"]) for item in CHECKPOINTS},
            {(seed, arm) for seed in (123, 124, 125) for arm in ("A", "B")},
        )
        self.assertEqual(len({item["sha256"] for item in CHECKPOINTS}), 6)

    def test_decode_contract_is_greedy_and_fixed(self):
        self.assertEqual(DECODE["max_new_tokens"], 24)
        self.assertEqual(DECODE["min_new_tokens"], 0)
        self.assertEqual(DECODE["num_beams"], 1)
        self.assertEqual(DECODE["temperature"], 1.0)

    def test_checkpoint_contract_accepts_registered_arm(self):
        spec = {"seed": 124, "arm": "B"}
        payload = {
            "checkpoint_version": 3,
            "progress": {"epoch": 66, "step": 9900, "epoch_complete": True},
            "training_contract": {
                "caption_policy": "balanced_all_references",
                "alignment_objective": "infonce",
                "alignment_weight": 0.1,
                "frozen_gpt2": True,
                "frozen_vit_backbone": True,
            },
            "args": {"gpt2_name": "gpt2", "vit_name": "vit", "cond_mode": "prefix",
                     "prefix_len": 4},
        }
        audit = validate_checkpoint_payload(payload, spec)
        self.assertTrue(all(audit["gates"].values()))

    def test_checkpoint_contract_rejects_wrong_intervention(self):
        spec = {"seed": 125, "arm": "A"}
        payload = {
            "checkpoint_version": 3,
            "progress": {"epoch": 66, "step": 9900, "epoch_complete": True},
            "training_contract": {
                "caption_policy": "balanced_all_references",
                "alignment_objective": "infonce",
                "alignment_weight": 0.1,
                "frozen_gpt2": True,
                "frozen_vit_backbone": True,
            },
            "args": {"gpt2_name": "gpt2", "vit_name": "vit", "cond_mode": "prefix",
                     "prefix_len": 4},
        }
        with self.assertRaisesRegex(RuntimeError, "contract failed"):
            validate_checkpoint_payload(payload, spec)

    def test_mean_sd_uses_sample_standard_deviation(self):
        summary = mean_sd([1.0, 2.0, 3.0])
        self.assertEqual(summary["mean"], 2.0)
        self.assertEqual(summary["sample_sd"], 1.0)

    def test_summarize_requires_and_pairs_all_models(self):
        results = []
        for seed in (123, 124, 125):
            for arm in ("A", "B"):
                offset = 1.0 if arm == "B" else 0.0
                caption = {name: float(seed) + offset for name in METRIC_NAMES}
                retrieval = {
                    "true_labels": {
                        "text_to_video": {"r_at_1": 0.1 + offset, "r_at_5": 0.2 + offset,
                                          "r_at_10": 0.3 + offset},
                        "video_to_text": {"r_at_1": 0.4 + offset, "r_at_5": 0.5 + offset,
                                          "r_at_10": 0.6 + offset},
                    },
                    "similarity": {"gap": 0.01 + offset},
                }
                results.append({"seed": seed, "arm": arm,
                                "caption_metrics": {"corpus_x100": caption},
                                "retrieval": retrieval})
        summary = summarize_results(results)
        self.assertEqual(summary["metrics"]["Bleu_4"]["B_minus_A"]["values"], [1.0] * 3)
        self.assertAlmostEqual(summary["metrics"]["T2V_R1"]["B_minus_A"]["mean"], 1.0)
        with self.assertRaisesRegex(ValueError, "six"):
            summarize_results(results[:-1])


if __name__ == "__main__":
    unittest.main()
