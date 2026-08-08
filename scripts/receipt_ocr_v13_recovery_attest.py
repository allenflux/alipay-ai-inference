"""Read-only checkpoint attestation for v13 guarded-evidence recovery."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, Mapping):
        raise ValueError(f"Expected one JSON object: {path}")
    return value


def _load_checkpoint(path: Path, *, torch: Any) -> Mapping[str, Any]:
    try:
        value = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:  # pragma: no cover - compatibility with older torch
        value = torch.load(path, map_location="cpu")
    if not isinstance(value, Mapping):
        raise ValueError(f"Expected one checkpoint mapping: {path}")
    return value


def _mapping(value: object, description: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{description} must be a mapping")
    return value


def attest(
    *,
    seed_checkpoint: Path,
    candidate_checkpoint: Path,
    training_summary_path: Path,
) -> dict[str, object]:
    import torch

    training = _load_json(training_summary_path)
    seed = _load_checkpoint(seed_checkpoint, torch=torch)
    candidate = _load_checkpoint(candidate_checkpoint, torch=torch)
    seed_state = _mapping(seed.get("state_dict"), "seed state_dict")
    candidate_state = _mapping(candidate.get("state_dict"), "candidate state_dict")

    if seed.get("kind") != "receipt_unified_field_reader_v12":
        raise ValueError("Seed checkpoint is not v12")
    if candidate.get("kind") != "receipt_unified_field_reader_v13":
        raise ValueError("Candidate checkpoint is not v13")
    best_epoch = training.get("best_checkpoint_epoch")
    if isinstance(best_epoch, bool) or not isinstance(best_epoch, int) or best_epoch <= 0:
        raise ValueError("Training summary best_checkpoint_epoch is invalid")
    if candidate.get("epoch") != best_epoch:
        raise ValueError("Candidate checkpoint epoch does not equal the selected best epoch")

    for key, checkpoint_key in (
        ("config", "config"),
        ("initialization", "initialization"),
        ("checkpoint_selection_policy", "checkpoint_selection_policy"),
        ("status_text_oov_by_split", "status_text_oov_by_split"),
        ("status_text_charset_sha256", "status_text_charset_sha256"),
        ("status_text_charset_source", "status_text_charset_source"),
        ("status_text_target", "status_text_target"),
        ("status_text_runtime_policy", "status_text_runtime_policy"),
        ("field_counts", "field_counts"),
        ("status_class_counts", "status_class_counts"),
    ):
        if candidate.get(checkpoint_key) != training.get(key):
            raise ValueError(f"Candidate checkpoint {checkpoint_key} differs from training summary")

    seed_keys = set(seed_state)
    candidate_keys = set(candidate_state)
    status_keys = {key for key in candidate_keys if str(key).startswith("status_text_")}
    if not status_keys or candidate_keys != seed_keys | status_keys:
        raise ValueError("Candidate must add only status_text_ tensors to the v12 seed")
    mismatches: list[str] = []
    for key in sorted(seed_keys):
        source = seed_state[key]
        observed = candidate_state[key]
        if not hasattr(source, "shape") or not hasattr(observed, "shape"):
            mismatches.append(str(key))
            continue
        if tuple(source.shape) != tuple(observed.shape) or source.dtype != observed.dtype:
            mismatches.append(str(key))
            continue
        if not torch.equal(source.detach().cpu(), observed.detach().cpu()):
            mismatches.append(str(key))
    if mismatches:
        preview = ", ".join(mismatches[:8])
        suffix = "..." if len(mismatches) > 8 else ""
        raise ValueError(f"Candidate changed frozen v12 tensors: {preview}{suffix}")

    initialization = _mapping(training.get("initialization"), "training initialization")
    if initialization.get("copied_legacy_tensor_count") != len(seed_keys):
        raise ValueError("Training copied_legacy_tensor_count differs from direct checkpoint audit")
    if initialization.get("new_status_text_tensor_count") != len(status_keys):
        raise ValueError("Training new_status_text_tensor_count differs from direct checkpoint audit")

    return {
        "schema_version": 1,
        "kind": "receipt_unified_v13_recovery_checkpoint_attestation_v1",
        "passed": True,
        "seed_checkpoint_sha256": _sha256(seed_checkpoint),
        "candidate_checkpoint_sha256": _sha256(candidate_checkpoint),
        "training_summary_sha256": _sha256(training_summary_path),
        "candidate_epoch": best_epoch,
        "legacy_tensor_count": len(seed_keys),
        "new_status_text_tensor_count": len(status_keys),
        "comparison": "torch.equal_cpu_for_every_non_status_text_tensor",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed-checkpoint", type=Path, required=True)
    parser.add_argument("--candidate-checkpoint", type=Path, required=True)
    parser.add_argument("--training-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite attestation output: {args.output}")
    result = attest(
        seed_checkpoint=args.seed_checkpoint.resolve(),
        candidate_checkpoint=args.candidate_checkpoint.resolve(),
        training_summary_path=args.training_summary.resolve(),
    )
    temporary = args.output.with_name(args.output.name + ".tmp")
    if temporary.exists():
        raise FileExistsError(f"Refusing to reuse temporary output: {temporary}")
    try:
        temporary.write_text(
            json.dumps(result, ensure_ascii=True, indent=2) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        temporary.replace(args.output)
    finally:
        temporary.unlink(missing_ok=True)
    print(json.dumps(result, ensure_ascii=True, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
