from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import torch
import torch.nn as nn


CHECKPOINT_VERSION = 3
RESUME_CONTRACT_KEYS = (
    "objective_version",
    "caption_objective",
    "alignment_objective",
    "alignment_weight",
    "info_nce_temperature",
    "false_negative_policy",
    "cpr_weight",
    "cpr_margin",
    "cpr_negative_policy",
    "video_representation",
    "text_representation",
    "physical_batch_size",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_item_count(path: Path) -> Optional[int]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return len(payload) if isinstance(payload, list) else None


def build_dataset_contract(annotation_paths: Mapping[str, str]) -> Dict[str, Any]:
    """Fingerprint annotations and retain the pinned dataset source revision."""

    annotations: Dict[str, Any] = {}
    resolved_paths = []
    for split, raw_path in annotation_paths.items():
        path = Path(raw_path).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"annotation file not found: {path}")
        resolved_paths.append(path)
        annotations[split] = {
            "path": str(path),
            "sha256": sha256_file(path),
            "records": _json_item_count(path),
        }

    roots = {path.parent.parent for path in resolved_paths}
    manifest: Optional[Dict[str, Any]] = None
    if len(roots) == 1:
        manifest_path = next(iter(roots)) / "manifest.json"
        if manifest_path.is_file():
            manifest_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            source = manifest_payload.get("source", {}) if isinstance(manifest_payload, dict) else {}
            manifest = {
                "path": str(manifest_path.resolve()),
                "sha256": sha256_file(manifest_path),
                "schema_version": manifest_payload.get("schema_version"),
                "valid": manifest_payload.get("valid"),
                "source_repository": source.get("repository"),
                "source_revision": source.get("revision"),
            }

    return {"annotations": annotations, "manifest": manifest}


def capture_rng_state() -> Dict[str, Any]:
    state: Dict[str, Any] = {
        "python": random.getstate(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": None,
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: Mapping[str, Any]) -> None:
    random.setstate(state["python"])
    # A full checkpoint is often loaded with map_location="cuda" so model and
    # optimizer tensors are ready immediately. RNG state tensors are the
    # exception: both APIs require CPU ByteTensors.
    torch.set_rng_state(state["torch_cpu"].cpu())
    cuda_state = state.get("torch_cuda")
    if cuda_state is not None:
        if not torch.cuda.is_available():
            raise RuntimeError("checkpoint contains CUDA RNG state but CUDA is unavailable")
        torch.cuda.set_rng_state_all([generator_state.cpu() for generator_state in cuda_state])


def load_checkpoint(path: Path, map_location: str | torch.device) -> Dict[str, Any]:
    try:
        state = torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        state = torch.load(path, map_location=map_location)
    if not isinstance(state, dict):
        return {"model_state": state, "checkpoint_version": 0}
    return state


def restore_model_state(
    model: nn.Module,
    checkpoint: Mapping[str, Any],
    visual_only: bool = False,
) -> None:
    state_dict = checkpoint.get("model_state", checkpoint)
    if not isinstance(state_dict, Mapping):
        raise TypeError("checkpoint model_state must be a mapping")
    if visual_only:
        state_dict = {
            key: value for key, value in state_dict.items() if not key.startswith("decoder.model.")
        }
    incompatible = model.load_state_dict(state_dict, strict=False)
    allowed_missing = visual_only and all(
        key.startswith("decoder.model.") for key in incompatible.missing_keys
    )
    if (incompatible.missing_keys and not allowed_missing) or incompatible.unexpected_keys:
        raise RuntimeError(
            "checkpoint is not architecture-compatible: "
            f"missing={incompatible.missing_keys}, unexpected={incompatible.unexpected_keys}"
        )


def restore_training_state(
    checkpoint: Mapping[str, Any],
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[Any] = None,
) -> Dict[str, Any]:
    if int(checkpoint.get("checkpoint_version", 0)) < CHECKPOINT_VERSION:
        raise RuntimeError(
            "exact training resume requires a stage-0 checkpoint with checkpoint_version >= 3"
        )
    optimizer_state = checkpoint.get("optimizer_state")
    if optimizer_state is None:
        raise RuntimeError("checkpoint does not contain optimizer_state")
    optimizer.load_state_dict(optimizer_state)

    scheduler_state = checkpoint.get("scheduler_state")
    if scheduler is not None:
        if scheduler_state is None:
            raise RuntimeError("checkpoint does not contain scheduler_state")
        scheduler.load_state_dict(scheduler_state)
    restore_rng_state(checkpoint["rng_state"])
    return dict(checkpoint["progress"])


def validate_training_resume_contract(
    checkpoint: Mapping[str, Any],
    training_contract: Mapping[str, Any],
    dataset_contract: Mapping[str, Any],
) -> None:
    """Reject a resume that silently changes the objective or dataset."""

    saved_training_contract = checkpoint.get("training_contract")
    saved_dataset_contract = checkpoint.get("dataset_contract")
    if not isinstance(saved_training_contract, Mapping) or not isinstance(
        saved_dataset_contract, Mapping
    ):
        raise RuntimeError("exact training resume requires saved training and dataset contracts")

    mismatches = []
    for key in RESUME_CONTRACT_KEYS:
        if saved_training_contract.get(key) != training_contract.get(key):
            mismatches.append(
                f"training_contract.{key}: saved={saved_training_contract.get(key)!r}, "
                f"current={training_contract.get(key)!r}"
            )

    saved_annotations = saved_dataset_contract.get("annotations", {})
    current_annotations = dataset_contract.get("annotations", {})
    if set(saved_annotations) != set(current_annotations):
        mismatches.append("dataset annotation split names changed")
    else:
        for split in saved_annotations:
            saved_hash = saved_annotations[split].get("sha256")
            current_hash = current_annotations[split].get("sha256")
            if saved_hash != current_hash:
                mismatches.append(
                    f"dataset_contract.annotations.{split}.sha256: "
                    f"saved={saved_hash!r}, current={current_hash!r}"
                )

    saved_manifest = saved_dataset_contract.get("manifest")
    current_manifest = dataset_contract.get("manifest")
    saved_manifest_hash = saved_manifest.get("sha256") if isinstance(saved_manifest, Mapping) else None
    current_manifest_hash = (
        current_manifest.get("sha256") if isinstance(current_manifest, Mapping) else None
    )
    if saved_manifest_hash != current_manifest_hash:
        mismatches.append(
            "dataset_contract.manifest.sha256: "
            f"saved={saved_manifest_hash!r}, current={current_manifest_hash!r}"
        )

    if mismatches:
        raise RuntimeError("training resume contract mismatch:\n- " + "\n- ".join(mismatches))


def save_training_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[Any],
    progress: Mapping[str, Any],
    args: Mapping[str, Any],
    training_contract: Mapping[str, Any],
    dataset_contract: Mapping[str, Any],
    loader_state: Mapping[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "checkpoint_version": CHECKPOINT_VERSION,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
        "progress": dict(progress),
        "rng_state": capture_rng_state(),
        "loader_state": dict(loader_state),
        "args": dict(args),
        "training_contract": dict(training_contract),
        "dataset_contract": dict(dataset_contract),
    }
    torch.save(payload, path)
