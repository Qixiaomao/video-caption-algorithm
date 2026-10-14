import unittest

import torch

from scripts.run_overfit16 import (
    SELECTED,
    configure_trainable,
    normalize_caption,
    state_sha,
    tensor_sha,
)


class Overfit16Tests(unittest.TestCase):
    def test_subset_has_sixteen_unique_ids_and_targets(self):
        self.assertEqual(len(SELECTED), 16)
        self.assertEqual(len({video_id for video_id, _ in SELECTED}), 16)
        self.assertEqual(len({normalize_caption(text) for _, text in SELECTED}), 16)

    def test_normalize_caption_is_case_space_and_punctuation_stable(self):
        self.assertEqual(normalize_caption(" A  Dog, RUNS! "), "a dog runs")

    def test_hash_covers_tensor_shape_dtype_and_value(self):
        value = torch.arange(4.0)
        self.assertEqual(tensor_sha(value), tensor_sha(value.clone()))
        self.assertNotEqual(tensor_sha(value), tensor_sha(value.reshape(2, 2)))
        self.assertNotEqual(tensor_sha(value), tensor_sha(value.double()))
        self.assertNotEqual(tensor_sha(value), tensor_sha(value + 1))
        self.assertEqual(state_sha({"a": value, "b": value + 1}),
                         state_sha({"b": value + 1, "a": value}))

    def test_only_projection_and_mapper_are_trainable(self):
        model = torch.nn.Module()
        model.encoder = torch.nn.Module()
        model.encoder.backbone = torch.nn.Linear(2, 2)
        model.encoder.proj = torch.nn.Linear(2, 2)
        model.decoder = torch.nn.Module()
        model.decoder.model = torch.nn.Linear(2, 2)
        model.decoder.mapper = torch.nn.Linear(2, 2)
        names = {name for name, _ in configure_trainable(model)}
        self.assertEqual(names, {"encoder.proj.weight", "encoder.proj.bias",
                                 "decoder.mapper.weight", "decoder.mapper.bias"})
        self.assertTrue(all(not parameter.requires_grad for parameter in model.decoder.model.parameters()))


if __name__ == "__main__":
    unittest.main()
