import json
import unittest

from scripts.run_full_msvd_infonce_ablation import (
    A_ROOT,
    LAMBDA,
    TEMPERATURE,
    assert_trace_equal,
    primary_decision,
    verify_registered_baseline,
)


def caption_metrics(nll=2.8, token_f1=0.70, viable=True):
    return {
        "native_nll": nll,
        "mean_max_reference_token_f1": token_f1,
        "primary_viability_gate": {"pass": viable},
    }


def retrieval(r1=0.10, gap=0.02, shuffled_r1=0.01):
    return {
        "true_labels": {"text_to_video": {"r_at_1": r1}},
        "shuffled_label_control": {"text_to_video": {"r_at_1": shuffled_r1}},
        "similarity": {"gap": gap},
    }


class FullMSVDInfoNCEAblationTests(unittest.TestCase):
    def test_intervention_is_the_single_registered_setting(self):
        self.assertEqual(LAMBDA, 0.10)
        self.assertEqual(TEMPERATURE, 0.07)

    @unittest.skipUnless((A_ROOT / "initial_model.pt").is_file(), "requires original frozen baseline assets")
    def test_registered_caption_only_artifacts_and_contract_are_intact(self):
        result, preflight = verify_registered_baseline()
        self.assertFalse(result["paper_ready"])
        self.assertFalse(preflight["split_counts"]["test_read"])
        self.assertEqual(result["config"]["steps"], 9900)

    @unittest.skipUnless((A_ROOT / "trace.jsonl").is_file(), "requires original frozen training trace")
    def test_registered_trace_has_exact_locked_endpoints(self):
        first = None
        last = None
        count = 0
        with (A_ROOT / "trace.jsonl").open("r", encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                first = row if first is None else first
                last = row
                count += 1
        self.assertEqual(count, 9900)
        self.assertEqual((first["step"], first["epoch"]), (1, 0))
        self.assertEqual((last["step"], last["epoch"]), (9900, 65))

    def test_trace_comparison_accepts_equality_and_rejects_target_change(self):
        row = {
            "step": 1, "epoch": 0, "indices": [0, 1],
            "video_ids": ["a", "b"], "target_texts": ["x", "y"],
            "caption_ids_sha256": "c", "visual_features_sha256": "v",
            "python_rng_sha256": "p", "torch_cpu_rng_sha256": "t",
            "torch_cuda_rng_sha256": ["g"],
        }
        assert_trace_equal(dict(row), row)
        changed = dict(row, target_texts=["x", "z"])
        with self.assertRaisesRegex(RuntimeError, "target_texts"):
            assert_trace_equal(changed, row)

    def test_primary_decision_requires_every_registered_gate(self):
        passed = primary_decision(
            caption_metrics(), caption_metrics(nll=2.9, token_f1=0.66),
            retrieval(r1=0.10, gap=0.02), retrieval(r1=0.11, gap=0.03),
        )
        self.assertTrue(passed["pass"])
        self.assertEqual(passed["decision"], "infonce_candidate_for_repeated_seeds")
        failed = primary_decision(
            caption_metrics(), caption_metrics(nll=2.96, token_f1=0.66),
            retrieval(r1=0.10, gap=0.02), retrieval(r1=0.11, gap=0.03),
        )
        self.assertFalse(failed["native_nll_le_a_plus_0_15"])
        self.assertEqual(failed["decision"], "retain_caption_only_or_redesign_infonce")


if __name__ == "__main__":
    unittest.main()
