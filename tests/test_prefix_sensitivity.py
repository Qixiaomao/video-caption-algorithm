from __future__ import annotations

import contextlib
import io
import json
import math
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import torch
from transformers import GPT2Config, GPT2LMHeadModel

from scripts.diagnose_prefix_sensitivity import (
    FULL, caption_nll, caption_summary, check_identity, donor_permutation,
    intervene, js_divergence, main, paired_bootstrap, parse_args, probe_layers,
    reference_nll, sha256_file, tokenize_references, validate_validation_records,
)
from src.models.caption_model import build_caption_supervision


class PrefixInterventionTests(unittest.TestCase):
    def test_derangement_is_seeded_unique_and_without_self_pairs(self):
        for n in (2, 3, 32, 100):
            first = donor_permutation(n, 123)
            self.assertTrue(torch.equal(first, donor_permutation(n, 123)))
            self.assertEqual(sorted(first.tolist()), list(range(n)))
            self.assertFalse(first.eq(torch.arange(n)).any())
        with self.assertRaises(ValueError):
            donor_permutation(1, 123)

    def test_all_interventions_preserve_inputs_and_dimensions(self):
        prefixes = torch.arange(24.0).reshape(3, 2, 4)
        saved = prefixes.clone()
        donors = donor_permutation(3, 123)
        for name in FULL:
            self.assertEqual(intervene(prefixes, name, donors).shape, prefixes.shape)
        self.assertTrue(torch.equal(intervene(prefixes, "shuffled", donors), prefixes[donors]))
        self.assertTrue(torch.equal(intervene(prefixes, "scale_1", donors), prefixes))
        self.assertTrue(torch.equal(intervene(prefixes, "scale_0", donors), torch.zeros_like(prefixes)))
        self.assertTrue(torch.equal(intervene(prefixes, "mean", donors)[0], prefixes.mean(0)))
        copy = intervene(prefixes, "repeat", donors)
        copy.zero_()
        self.assertTrue(torch.equal(prefixes, saved))
        with self.assertRaises(ValueError):
            intervene(prefixes, "unknown", donors)

    def test_jsd_identity_symmetry_offset_invariance_and_bound(self):
        a = torch.tensor([[1., 2., 3.], [-1000., 1000., 0.]], dtype=torch.float64)
        b = a.flip(-1)
        self.assertTrue(torch.allclose(js_divergence(a, a), torch.zeros(2, dtype=torch.float64)))
        self.assertTrue(torch.allclose(js_divergence(a, b), js_divergence(b, a)))
        self.assertTrue(torch.allclose(js_divergence(a, b), js_divergence(a + 100, b - 50)))
        self.assertTrue((js_divergence(a, b) <= math.log(2) + 1e-12).all())
        self.assertTrue(torch.isfinite(js_divergence(a, b)).all())

    def test_bootstrap_sign_and_seed(self):
        result = paired_bootstrap([0.2, 0.2, 0.2], 123, 100)
        self.assertAlmostEqual(result["ci95_low"], 0.2)
        self.assertAlmostEqual(result["ci95_high"], 0.2)
        self.assertEqual(paired_bootstrap([-1., 0., 1.], 123, 200),
                         paired_bootstrap([-1., 0., 1.], 123, 200))
        with self.assertRaises(ValueError):
            paired_bootstrap([1.], 123, 100)

    def test_caption_comparison_normalizes_whitespace_and_case(self):
        result = caption_summary([" A  man ", "a dog"], ["a man", "a cat"])
        self.assertEqual(result["changed_rate"], 0.5)
        self.assertEqual(result["unique_rate"], 1.0)

    def test_identity_control_rejects_caption_or_nll_drift(self):
        base = {"logits": torch.zeros(2, 4), "videos": [
            {"caption": "a man", "nll": 1.}, {"caption": "a dog", "nll": 2.}]}
        check_identity(base, base)
        changed = {**base, "videos": [{"caption": "a cat", "nll": 1.}, base["videos"][1]]}
        with self.assertRaises(RuntimeError):
            check_identity(base, changed)


class PrefixNLLTests(unittest.TestCase):
    def test_first_caption_token_and_eos_are_included_padding_is_ignored(self):
        # Prefix positions 0,1. Position 1 predicts first caption token 3.
        logits = torch.zeros(1, 6, 7)
        logits[0, 1, 3] = 20
        logits[0, 2, 4] = 20
        logits[0, 3, 2] = 20
        ids = torch.tensor([[3, 4, 2, 2]])
        _, labels = build_caption_supervision(ids, 2)
        losses, valid = caption_nll(logits, labels, 2)
        self.assertEqual(valid.tolist(), [[True, True, True, False]])
        self.assertLess(float(losses.sum()), 1e-5)
        logits[0, 1] = 0
        changed, _ = caption_nll(logits, labels, 2)
        self.assertGreater(float(changed.sum()), 1.0)
        logits[0, 4] = 100
        padded, _ = caption_nll(logits, labels, 2)
        self.assertTrue(torch.equal(changed, padded))

    def test_nll_matches_hf_causal_loss_and_microbatch_is_neutral(self):
        torch.manual_seed(12)
        lm = GPT2LMHeadModel(GPT2Config(vocab_size=17, n_embd=8, n_head=2,
            n_layer=2, n_positions=32, eos_token_id=2, pad_token_id=2)).eval()
        prefix = torch.randn(1, 2, 8)
        ids = torch.tensor([[3, 4, 2, 2], [6, 7, 8, 2]])
        mask, labels = build_caption_supervision(ids, 2)
        embeds = torch.cat([prefix.expand(2, -1, -1), lm.transformer.wte(ids)], 1)
        extended_labels = torch.cat([torch.full((2, 2), -100), labels], 1)
        with torch.no_grad():
            out = lm(inputs_embeds=embeds,
                     attention_mask=torch.cat([torch.ones(2, 2, dtype=torch.long), mask], 1),
                     labels=extended_labels, use_cache=False)
        losses, valid = caption_nll(out.logits, labels, 2)
        self.assertAlmostEqual(float(losses.sum() / valid.sum()), float(out.loss), places=6)
        refs = [{"ids": [3, 4, 2]}, {"ids": [6, 7, 8, 2]}]
        one = reference_nll(lm, SimpleNamespace(eos_token_id=2), prefix, refs, 1)
        two = reference_nll(lm, SimpleNamespace(eos_token_id=2), prefix, refs, 2)
        for a, b in zip(one, two):
            self.assertAlmostEqual(a["nll"], b["nll"], places=5)
        weighted = sum(r["nll"] * r["token_count_including_eos"] for r in two) / 7
        self.assertAlmostEqual(weighted, float(out.loss), places=6)

    def test_tokenization_logs_truncation_and_appends_eos(self):
        tokenizer = SimpleNamespace(eos_token_id=2, encode=lambda text, **kw: [3, 4, 5, 6])
        refs = tokenize_references(tokenizer, ["sample"], 3)
        self.assertEqual(refs[0]["ids"], [3, 4, 2])
        self.assertTrue(refs[0]["truncated"])
        self.assertEqual(refs[0]["original_token_count"], 4)

    def test_layer_boundaries_include_last_block_before_and_after_ln(self):
        lm = GPT2LMHeadModel(GPT2Config(vocab_size=17, n_embd=8, n_head=2,
                                      n_layer=2, n_positions=32)).eval()
        prefix = torch.randn(1, 2, 8)
        logits, layers = probe_layers(lm, prefix)
        self.assertEqual(list(layers), ["embedding_plus_position", "block_01_pre_final_ln",
                                       "block_02_pre_final_ln", "final_ln"])
        self.assertEqual(tuple(logits.shape), (1, 17))
        with torch.no_grad():
            self.assertTrue(torch.allclose(lm.lm_head(layers["final_ln"][:, -1]), logits))
        self.assertEqual(len(lm.transformer.ln_f._forward_hooks), 0)
        self.assertTrue(all(len(b._forward_hooks) == 0 for b in lm.transformer.h))


class PrefixSafetyTests(unittest.TestCase):
    def test_existing_report_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            report = Path(tmp) / "existing.json"
            report.write_text("sentinel", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                main(["--checkpoint", "not-loaded.pt", "--expected-checkpoint-sha256", "x",
                      "--output", str(report)])
            self.assertEqual(report.read_text(encoding="utf-8"), "sentinel")

    def test_frame_directory_cannot_escape_validation_tree(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "annotations.json"
            records = [{"video_id": "a", "captions": ["a man"],
                        "frames_dir": str(Path(tmp).parent / "not-validation")},
                       {"video_id": "b", "captions": ["a dog"]}]
            path.write_text(json.dumps(records), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "outside validation tree"):
                validate_validation_records(path, 2, expected_path=path,
                                            expected_sha=sha256_file(path), expected_count=2)

    def test_non_validation_path_is_rejected_before_reading(self):
        with self.assertRaisesRegex(ValueError, "validation annotation path"):
            validate_validation_records(Path("not-real/test/annotations.json"), 2)

    def test_hash_mismatch_and_missing_frames_fail_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "annotations.json"
            path.write_text(json.dumps([{"video_id": "a", "captions": ["a man"]},
                                       {"video_id": "b", "captions": ["a dog"]}]), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                validate_validation_records(path, 2, expected_path=path,
                                            expected_sha="wrong", expected_count=2)
            with self.assertRaisesRegex(ValueError, "cannot silently skip"):
                validate_validation_records(path, 2, expected_path=path,
                                            expected_sha=sha256_file(path), expected_count=2)

    def test_cli_prevents_unbounded_smoke_and_partial_reference_full_run(self):
        base = ["--checkpoint", "local.pt", "--expected-checkpoint-sha256", "x", "--output", "out.json"]
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                parse_args(base + ["--smoke-test"])
            with self.assertRaises(SystemExit):
                parse_args(base + ["--references-per-video", "2"])
        args = parse_args(base + ["--smoke-test", "--limit", "2", "--references-per-video", "2"])
        self.assertTrue(args.smoke_test)


if __name__ == "__main__":
    unittest.main()
