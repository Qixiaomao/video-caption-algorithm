"""Failure-oriented checks for the portable evidence verifier, no model assets."""
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from scripts.verify_public_evidence import ROOT, evaluate, verify


class PublicEvidenceTests(unittest.TestCase):
    def test_all_frozen_numeric_rows_recompute(self):
        result = verify(ROOT / "reproducibility/evidence")
        self.assertEqual(result["rows_recomputed"], 1036)

    def test_changed_csv_fails_before_numeric_use(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary) / "evidence"
            shutil.copytree(ROOT / "reproducibility/evidence", directory)
            with (directory / "paper_master_results.csv").open("a", encoding="utf-8") as stream:
                stream.write("corrupted\n")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                verify(directory)

    def test_paired_sd_uses_differences_before_aggregation(self):
        raw = [1, 2, 20, 22, 300, 303]
        values = {str(i): {"raw_value": value} for i, value in enumerate(raw)}
        operands = [{"source_value_id": str(i), "scale": 1} for i in range(len(raw))]
        self.assertEqual(evaluate("sample_sd_difference", operands, values), 1.0)

    def test_unknown_operation_and_incomplete_pair_fail(self):
        values = {"x": {"raw_value": 1}}
        operands = [{"source_value_id": "x", "scale": 1}]
        with self.assertRaisesRegex(ValueError, "Unknown operation"):
            evaluate("typo", operands, values)
        with self.assertRaisesRegex(ValueError, "Unpaired"):
            evaluate("mean_difference", operands, values)


if __name__ == "__main__":
    unittest.main()
