from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from scripts import resume_cpr_cd_matrix as adapter
from scripts import run_cpr_cd_matrix as registered


class ResumeAdapterTests(unittest.TestCase):
    def test_prefix_reader_rejects_missing_or_out_of_order_steps(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "trace.jsonl"
            path.write_text(
                json.dumps({"step": 1}) + "\n" + json.dumps({"step": 3}) + "\n",
                encoding="utf-8",
            )
            self.assertEqual(len(adapter.checked_lines(path, 1)), 1)
            with self.assertRaisesRegex(RuntimeError, "step 3 != 2"):
                adapter.checked_lines(path, 2)
            path.write_text(json.dumps({"step": 1}) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "ended before step 2"):
                adapter.checked_lines(path, 2)

    def test_full_checkpoint_contract_rejects_changed_cpr_weight(self):
        dataset_contract = {"annotations": {}, "manifest": None}
        checkpoint = {
            "progress": {"step": 6000, "epoch": 40, "epoch_complete": True},
            "args": {"seed": 124, "arm": "C", "batch_size": 8,
                     "max_len": 64, "lr": 3e-4, "epochs": 66},
            "loader_state": {
                "registered_schedule_sha256": registered.REGISTERED_INPUTS[124]["schedule"][1],
                "registered_schedule_path": str(registered.REGISTERED_INPUTS[124]["schedule"][0].resolve()),
                "num_workers": 0,
            },
            "training_contract": registered.training_contract("C"),
            "dataset_contract": dataset_contract,
        }
        adapter.validate_checkpoint(checkpoint, {"dataset_contract": dataset_contract})
        checkpoint["training_contract"] = dict(checkpoint["training_contract"])
        checkpoint["training_contract"]["cpr_weight"] = 0.2
        with self.assertRaisesRegex(RuntimeError, "full training contract"):
            adapter.validate_checkpoint(checkpoint, {"dataset_contract": dataset_contract})


if __name__ == "__main__":
    unittest.main()
