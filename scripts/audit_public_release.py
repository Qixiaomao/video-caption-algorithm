"""Scan tracked/non-ignored release files; print locations, never secret values.

Heuristic only; this is not a Git-history credential audit.
"""
from __future__ import annotations
import json
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN_SUFFIXES = {".pt", ".pth", ".bin", ".ckpt", ".safetensors", ".onnx", ".engine",
                      ".pkl", ".npy", ".npz", ".zip", ".mp4", ".avi", ".mov", ".mkv", ".docx"}
FORBIDDEN_ROOTS = {"data", "runs", "checkpoints", "outputs", "logs", ".venv", ".secrets"}
ASSISTANT_FILES = {"agents.md", "agent.md", "claude.md", "gemini.md", "spec.md"}
ASSISTANT_DIRECTORIES = {".codex", ".agents", ".cursor", ".claude", ".gemini",
                         ".specify", ".speckit", ".kiro", "spec-kit", "specs", "codex_specs"}
SECRETS = re.compile(r"(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{40,}|"
                     r"AKIA[A-Z0-9]{16}|sk-(?:proj-)?[A-Za-z0-9_-]{30,}|"
                     r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----)")
PERSONAL_PATH = re.compile(r"(?:[A-Z]:[\\/](?:Users|Mylearn)[\\/]|/" + r"Users/[^/\s]+/)", re.I)


def is_assistant_metadata(name):
    parts = Path(name).parts
    return (parts[-1].casefold() in ASSISTANT_FILES
            or parts[-1].casefold().endswith(".spec.md")
            or any(part.casefold() in ASSISTANT_DIRECTORIES for part in parts[:-1]))


def audit():
    output = subprocess.check_output(["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"], cwd=ROOT)
    names = sorted(set(output.decode("utf-8").split("\0")) - {""})
    findings, checked = [], 0
    for name in names:
        path = ROOT / name
        if not path.is_file():
            continue
        checked += 1
        if is_assistant_metadata(name):
            findings.append({"path": name, "reason": "assistant instructions or generated planning metadata"})
        local_env = path.name == ".env" or path.name.startswith(".env.") and path.name != ".env.example"
        if path.suffix.lower() in FORBIDDEN_SUFFIXES or Path(name).parts[0] in FORBIDDEN_ROOTS or local_env:
            findings.append({"path": name, "reason": "local asset or secret configuration"})
        if path.stat().st_size > 10 * 1024 * 1024:
            findings.append({"path": name, "reason": "file larger than 10 MiB"})
        try:
            text = path.read_text(encoding="utf-8-sig")
        except UnicodeError:
            continue
        for number, line in enumerate(text.splitlines(), 1):
            if SECRETS.search(line):
                findings.append({"path": name, "line": number, "reason": "credential pattern"})
            if PERSONAL_PATH.search(line):
                findings.append({"path": name, "line": number, "reason": "personal absolute path"})
    return {"status": "PASS" if not findings else "FAIL", "files_checked": checked,
            "findings": findings, "scope": "tracked and non-ignored files; Git history not scanned"}


if __name__ == "__main__":
    result = audit()
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["status"] == "PASS" else 1)
