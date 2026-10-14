"""Prevent private assistant metadata from entering the publication snapshot."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from scripts import audit_public_release as release_audit


class PublicReleaseAuditTests(unittest.TestCase):
    def test_nested_and_mixed_case_metadata_rejected(self):
        for name in ("AGENTS.md", "src/Agents.md", ".codex/specs/task.md",
                     "nested/.agents/skills/helper.md", ".cursor/rules/project.mdc",
                     ".specify/memory/constitution.md", "specs/001-feature/plan.md",
                     "codex_specs/run.json", "implementation.spec.md"):
            with self.subTest(name=name):
                self.assertTrue(release_audit.is_assistant_metadata(name))

    def test_research_protocols_and_application_code_allowed(self):
        for name in ("experiments/msrvtt_ab_protocol_20260928.md",
                     "reproducibility/README.md", "scripts/inspect_msrvtt_formal_progress_20260928.py",
                     ".github/workflows/public-evidence.yml", "src/models/caption_model.py"):
            with self.subTest(name=name):
                self.assertFalse(release_audit.is_assistant_metadata(name))

    def test_tracked_metadata_fails_even_if_ignore_rules_would_hide_it(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "AGENTS.md").write_text("local instructions", encoding="utf-8")
            with patch.object(release_audit, "ROOT", root), patch.object(
                    release_audit.subprocess, "check_output", return_value=b"AGENTS.md\0"):
                result = release_audit.audit()
            self.assertEqual(result["status"], "FAIL")
            self.assertEqual(result["findings"][0]["reason"],
                             "assistant instructions or generated planning metadata")


if __name__ == "__main__":
    unittest.main()
