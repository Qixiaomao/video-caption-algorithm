from __future__ import annotations

import unittest
from unittest.mock import patch

from scripts import run_cpr_cd_matrix as matrix


def endpoint(grounding_gap: float, caption_gate: bool = True) -> dict:
    return {
        "caption": {
            "metrics": {
                "primary_viability_gate": {"pass": caption_gate},
                "native_nll": 2.0,
                "mean_max_reference_token_f1": 0.7,
                "shuffled_minus_native_nll": grounding_gap,
            }
        },
        "retrieval": {
            "true_labels": {"text_to_video": {"r_at_1": 0.1}}
        },
    }


class RegisteredMatrixTests(unittest.TestCase):
    @unittest.skipUnless(all(matrix.REGISTERED_INPUTS[s]["schedule"][0].is_file() for s in matrix.SEEDS),
                         "requires original frozen per-seed schedules")
    def test_registered_schedules_cover_all_formal_steps(self):
        for seed in matrix.SEEDS:
            schedule = matrix.load_registered_schedule(seed)
            self.assertEqual(len(schedule["batches"]), 66)
            self.assertTrue(all(len(epoch) == 150 for epoch in schedule["batches"]))
            self.assertEqual(len(schedule["orders"]), 1200)

    def test_c_and_d_contracts_differ_only_in_registered_alignment_fields(self):
        c = matrix.training_contract("C")
        d = matrix.training_contract("D")
        differences = {key for key in c if c[key] != d[key]}

        self.assertEqual(differences, {"alignment_objective", "alignment_weight"})
        self.assertEqual(c["objective_version"], 4)
        self.assertEqual(c["cpr_weight"], 0.1)
        self.assertTrue(c["cpr_wrong_decoder_isolated_rng"])

    def test_formal_decision_requires_both_grounding_strata(self):
        passing = {
            seed: {
                "C": {"evaluations": [endpoint(0.3)]},
                "D": {"evaluations": [endpoint(0.3)]},
            }
            for seed in matrix.SEEDS
        }
        with patch.object(matrix, "historical_endpoint", return_value=endpoint(0.2)):
            decision = matrix.formal_decision(passing)

        self.assertTrue(decision["gates"]["pass"])
        self.assertEqual(decision["gates"]["decision"], "retain_cpr_in_main")

        failing = {
            seed: {
                "C": {"evaluations": [endpoint(0.3)]},
                "D": {"evaluations": [endpoint(0.1)]},
            }
            for seed in matrix.SEEDS
        }
        with patch.object(matrix, "historical_endpoint", return_value=endpoint(0.2)):
            decision = matrix.formal_decision(failing)

        self.assertFalse(decision["gates"]["pass"])
        self.assertEqual(decision["gates"]["decision"], "appendix_or_remove_cpr")


if __name__ == "__main__":
    unittest.main()
