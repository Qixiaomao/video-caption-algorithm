"""Validate local config registration without importing models or using GPU assets."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from scripts import materialize_msrvtt_config as materializer


class MaterializeConfigTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        (self.root / "experiments").mkdir()
        self.model = self.root / "models/gpt2"
        self.model.mkdir(parents=True)
        (self.model / "model.safetensors").write_bytes(b"synthetic test weights")
        self.template = self.root / "experiments/template.json"
        self.template.write_text(json.dumps({"epochs": 30, "seeds": [123, 124, 125],
            "gpt2_model_sha256": materializer.sha256(self.model / "model.safetensors")}), encoding="utf-8")
        self.paths = [self.root / f"runs/{name}/result.json" for name in (
            "msrvtt_engineering_smoke_20260928_v3", "msrvtt_train_val_pilot_20260928",
            "msrvtt_validation_timing_20260928")]
        self.paths.append(self.root / "runs/msrvtt_data_full_20260928/data_gate.json")
        for path in self.paths:
            path.parent.mkdir(parents=True)
            path.write_text(json.dumps({"status": "PASS", "test_split_read": False}), encoding="utf-8")
        (self.root / "experiments/msrvtt_test_eval_config_20260928.json").write_text("{}", encoding="utf-8")
        self.output = self.root / "experiments/reproduction.local.json"
        self.patch_root = patch.object(materializer, "ROOT", self.root)
        self.patch_template = patch.object(materializer, "TEMPLATE", self.template)
        self.patch_root.start()
        self.patch_template.start()
        self.addCleanup(self.patch_root.stop)
        self.addCleanup(self.patch_template.stop)

    def test_registers_new_paths_hashes_without_changing_template(self):
        before = self.template.read_bytes()
        materializer.materialize(self.model, self.output)
        result = json.loads(self.output.read_text(encoding="utf-8"))
        self.assertEqual(result["epochs"], 30)
        self.assertEqual(result["seeds"], [123, 124, 125])
        self.assertEqual(result["engineering_smoke_sha256"], materializer.sha256(self.paths[0]))
        self.assertEqual(self.template.read_bytes(), before)
        with self.assertRaises(FileExistsError):
            materializer.materialize(self.model, self.output)

    def test_test_access_and_failed_gates_are_rejected(self):
        for value in ({"status": "FAIL", "test_split_read": False},
                      {"status": "PASS", "test_split_read": True}):
            self.paths[0].write_text(json.dumps(value), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "gate invalid"):
                materializer.materialize(self.model, self.output)
            self.assertFalse(self.output.exists())

    def test_changed_weights_and_unignored_output_rejected(self):
        (self.model / "model.safetensors").write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "weights differ"):
            materializer.materialize(self.model, self.output)
        with self.assertRaisesRegex(ValueError, "local.json"):
            materializer.materialize(self.model, self.root / "experiments/unignored.json")


if __name__ == "__main__":
    unittest.main()
