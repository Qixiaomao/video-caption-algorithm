import json
import tempfile
import unittest
from pathlib import Path

import torch
from scripts.run_matched_gpt2_init import tensor_sha, state_sha, compare_traces, configure_trainable


class MatchedInitializationTests(unittest.TestCase):
    def test_hash_covers_dtype_shape_and_value(self):
        x = torch.arange(4.)
        self.assertEqual(tensor_sha(x), tensor_sha(x.clone()))
        self.assertNotEqual(tensor_sha(x), tensor_sha(x.reshape(2, 2)))
        self.assertNotEqual(tensor_sha(x), tensor_sha(x.double()))
        self.assertNotEqual(tensor_sha(x), tensor_sha(x + 1))

    def test_state_hash_order_invariant_and_name_sensitive(self):
        a, b = torch.ones(2), torch.zeros(2)
        self.assertEqual(state_sha({"a": a, "b": b}), state_sha({"b": b, "a": a}))
        self.assertNotEqual(state_sha({"a": a}), state_sha({"b": a}))

    def test_trace_comparison_rejects_any_unmatched_control(self):
        with tempfile.TemporaryDirectory() as tmp:
            left, right = Path(tmp) / "left", Path(tmp) / "right"
            x = {"step": 1, "video_ids": ["v1"], "rng": "same"}
            left.write_text(json.dumps(x) + "\n")
            right.write_text(json.dumps(x) + "\n")
            self.assertTrue(compare_traces(left, right, 1)["all_inputs_and_rng_identical"])
            with self.assertRaisesRegex(RuntimeError, "length"):
                compare_traces(left, right, 2)
            right.write_text(json.dumps(dict(x, rng="changed")) + "\n")
            with self.assertRaisesRegex(RuntimeError, "rng"):
                compare_traces(left, right, 1)

    def test_only_fresh_visual_modules_are_trainable(self):
        model = torch.nn.Module()
        model.encoder = torch.nn.Module()
        model.encoder.backbone = torch.nn.Linear(2, 2)
        model.encoder.proj = torch.nn.Linear(2, 2)
        model.decoder = torch.nn.Module()
        model.decoder.model = torch.nn.Linear(2, 2)
        model.decoder.mapper = torch.nn.Linear(2, 2)
        names = {n for n, _ in configure_trainable(model)}
        self.assertEqual(names, {"encoder.proj.weight", "encoder.proj.bias",
                                 "decoder.mapper.weight", "decoder.mapper.bias"})
        self.assertTrue(all(not p.requires_grad for p in model.decoder.model.parameters()))


if __name__ == "__main__":
    unittest.main()
