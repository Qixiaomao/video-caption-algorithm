from __future__ import annotations

import json
import random
import tempfile
import unittest
from pathlib import Path

import torch
import torch.nn as nn

from src.training.checkpoint import (
    build_dataset_contract,
    load_checkpoint,
    capture_rng_state,
    restore_model_state,
    restore_rng_state,
    restore_training_state,
    save_training_checkpoint,
    validate_training_resume_contract,
)


class TrainingCheckpointTests(unittest.TestCase):
    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA map-location behavior")
    def test_rng_restore_normalizes_map_location_cuda_tensors(self):
        torch.manual_seed(17)
        state = capture_rng_state()
        state["torch_cpu"] = state["torch_cpu"].cuda()
        state["torch_cuda"] = [generator_state.cuda() for generator_state in state["torch_cuda"]]

        restore_rng_state(state)

        self.assertEqual(torch.get_rng_state().device.type, "cpu")

    def test_full_checkpoint_restores_model_optimizer_and_rng(self):
        torch.manual_seed(11)
        random.seed(11)
        model = nn.Linear(3, 2)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
        loss = model(torch.ones(2, 3)).sum()
        loss.backward()
        optimizer.step()
        saved_weight = model.weight.detach().clone()

        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_path = Path(temp_dir) / "stage0.pt"
            save_training_checkpoint(
                checkpoint_path,
                model=model,
                optimizer=optimizer,
                scheduler=None,
                progress={
                    "step": 3,
                    "epoch": 0,
                    "batch_in_epoch": 3,
                    "epoch_complete": False,
                    "best_val": 1.25,
                },
                args={"alignment_objective": "infonce"},
                training_contract={"objective_version": 3},
                dataset_contract={"manifest": None},
                loader_state={"current_generator_state": torch.Generator().get_state()},
            )
            expected_python = random.random()
            expected_torch = torch.rand(3)

            with torch.no_grad():
                model.weight.add_(10)
            random.seed(999)
            torch.manual_seed(999)

            checkpoint = load_checkpoint(checkpoint_path, map_location="cpu")
            restore_model_state(model, checkpoint)
            progress = restore_training_state(checkpoint, optimizer)

        self.assertTrue(torch.equal(model.weight, saved_weight))
        self.assertEqual(progress["step"], 3)
        self.assertEqual(random.random(), expected_python)
        self.assertTrue(torch.equal(torch.rand(3), expected_torch))
        self.assertTrue(optimizer.state_dict()["state"])

    def test_strict_restore_rejects_unexpected_model_keys(self):
        model = nn.Linear(2, 2)
        state = dict(model.state_dict())
        state["unexpected"] = torch.tensor(1)

        with self.assertRaisesRegex(RuntimeError, "architecture-compatible"):
            restore_model_state(model, {"model_state": state})

    def test_dataset_contract_records_hash_counts_and_revision(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            for split, count in (("train", 2), ("val", 1)):
                split_dir = root / split
                split_dir.mkdir()
                (split_dir / "annotations.json").write_text(
                    json.dumps([{"video_id": str(index)} for index in range(count)]),
                    encoding="utf-8",
                )
            (root / "manifest.json").write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "valid": True,
                        "source": {"repository": "fixture/MSVD", "revision": "abc123"},
                    }
                ),
                encoding="utf-8",
            )

            contract = build_dataset_contract(
                {
                    "train": str(root / "train" / "annotations.json"),
                    "val": str(root / "val" / "annotations.json"),
                }
            )

        self.assertEqual(contract["annotations"]["train"]["records"], 2)
        self.assertEqual(len(contract["annotations"]["val"]["sha256"]), 64)
        self.assertEqual(contract["manifest"]["source_revision"], "abc123")

    def test_exact_resume_rejects_changed_alignment_weight(self):
        saved_training = {
            "objective_version": 3,
            "caption_objective": "causal_lm_cross_entropy",
            "alignment_objective": "infonce",
            "alignment_weight": 0.1,
            "info_nce_temperature": 0.07,
            "false_negative_policy": "exact_caption",
            "cpr_weight": 0.0,
            "cpr_margin": 0.1,
            "cpr_negative_policy": "cyclic_derangement",
            "video_representation": "prefix",
            "text_representation": "wte",
            "physical_batch_size": 4,
        }
        current_training = dict(saved_training, alignment_weight=0.2)
        dataset = {
            "annotations": {"train": {"sha256": "abc"}, "val": {"sha256": "def"}},
            "manifest": {"sha256": "manifest"},
        }

        with self.assertRaisesRegex(RuntimeError, "alignment_weight"):
            validate_training_resume_contract(
                {
                    "training_contract": saved_training,
                    "dataset_contract": dataset,
                },
                current_training,
                dataset,
            )

    def test_exact_resume_rejects_changed_cpr_margin(self):
        saved_training = {
            "objective_version": 4,
            "caption_objective": "causal_lm_cross_entropy",
            "alignment_objective": "none",
            "alignment_weight": 0.0,
            "info_nce_temperature": 0.07,
            "false_negative_policy": "exact_caption",
            "cpr_weight": 0.1,
            "cpr_margin": 0.1,
            "cpr_negative_policy": "cyclic_derangement",
            "video_representation": "prefix",
            "text_representation": "wte",
            "physical_batch_size": 8,
        }
        current_training = dict(saved_training, cpr_margin=0.2)
        dataset = {
            "annotations": {"train": {"sha256": "abc"}, "val": {"sha256": "def"}},
            "manifest": {"sha256": "manifest"},
        }

        with self.assertRaisesRegex(RuntimeError, "cpr_margin"):
            validate_training_resume_contract(
                {
                    "training_contract": saved_training,
                    "dataset_contract": dataset,
                },
                current_training,
                dataset,
            )


if __name__ == "__main__":
    unittest.main()
