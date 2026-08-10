"""Seal the single failed recipient-v14 fresh-60 experiment.

This module is deliberately separate from the candidate runner.  It cannot
resume the failed model, export it, evaluate test, or authorize production.
It reopens the immutable source/A8 authorities and every artifact left by the
one permitted fresh-60 run, recomputes the complete validation schedule and
the fixed safety guards, and records the observed failure as analysis-only
evidence.  The only positive authority in the result is one *different data
view*, exactly-eight-epoch pilot that must start again from the attested legacy
source rather than either failed checkpoint.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

from . import recipient_full_crop_candidate_source as _candidate_source_verifier
from . import recipient_full_crop_pilot as _full_crop_pilot_verifier
from . import recipient_full_crop_seed_sanitizer as _seed_sanitizer_verifier
from .ocr_unified import (
    CHECKPOINT_SELECTION_RECIPIENT_PRIORITY,
    INIT_CHECKPOINT_MODE_RECIPIENT_VISUAL_CONTEXT_REINIT,
    KIND_V13,
    STATUS_TEXT_RUNTIME_POLICY,
    UnifiedReaderConfig,
    _checkpoint_config,
    _load_checkpoint,
    _recipient_train_augmentation_policy,
    _require_torch,
    _validate_recipient_visual_context_reinit_config,
)
from .recipient_full_crop_candidate_source import (
    CANDIDATE_PILOT_DECISION,
    CANDIDATE_PILOT_KIND,
    EXPECTED_RECIPIENT_VAL_RECORDS,
    RECIPIENT_DELIVERY_FLOOR,
    REQUIRED_BACKBONE,
    REQUIRED_SOURCE_BACKBONE,
    SOURCE_KIND,
    validate_full_crop_training_recipe,
    verify_full_crop_candidate_source,
    verify_residual_candidate_pilot,
)
from .recipient_full_crop_pilot import (
    AMOUNT_FLOOR,
    PAYMENT_FLOOR,
    STATUS_TEXT_FLOOR,
    TIME_FLOOR,
)


SCHEMA_VERSION = 1
KIND = "receipt_recipient_v14_fresh60_failure_attestation_v1"
DECISION = "analysis_only_authorize_different_data_view_exact8_pilot"
AUTHORIZATION = "different_training_data_view_exact_8_epoch_pilot_only"
SUBJECT_DOMAIN = "receipt-recipient-v14-fresh60-failure-subject-v1"
ATTEMPT_KIND = "receipt_recipient_v14_full_crop_training_attempt_v1"
ATTEMPT_DOMAIN = "receipt-v14-full-crop-candidate-60e-v1"
ATTEMPT_REGISTRY_NAME = "recipient-v14-full-crop-training-v1"
ATTEMPT_REGISTRY_PARENT = "ReceiptAI"
ATTEMPT_THREAT_MODEL = (
    "persistent local no-rerun guard; crash and failed training consume the fixed attempt"
)
EXPECTED_EPOCHS = 60
EXPECTED_BEST_EPOCH = 44
EXPECTED_BEST_MATCHES = 5919
EXPECTED_LAST_EPOCH = 60
EXPECTED_LAST_MATCHES = 5899
EXPECTED_VALIDATION_EPOCHS = [1, *range(2, EXPECTED_EPOCHS + 1, 2)]
EXPECTED_STRICT_PASS_MATCHES = 6111
EXPECTED_ATTEMPT_ID = (
    "155156f0678fd697904fcca953c611836896dcdc62cb09839eb66f8d62c5c66d"
)
EXPECTED_SOURCE_SUBJECT_ID = (
    "98f0617404d7d58e99a0794d2340da9154f81667f0aa6a546027dd19209b886a"
)
EXPECTED_CANDIDATE_PILOT_SUBJECT_ID = (
    "5d5c0cbe5041252dc9de8d69076400deb7c8d3909d81c424287863d59b49433e"
)
EXPECTED_AUTHORITY_DOCUMENT_PINS: dict[str, dict[str, int | str]] = {
    "source_contract": {
        "size_bytes": 9067,
        "sha256": "61fc5eebd72e215ad4e4f7be265b0d37be98b3de7a86e2d2e909437756153246",
    },
    "candidate_pilot_evidence": {
        "size_bytes": 7627,
        "sha256": "324ab62634a9fc054d3aab70b3ce9e2800da994534d9736119b8738bc8bff4b3",
    },
}
# These are read-only pins from the one consumed Windows fresh60 attempt.
# They deliberately cover both model serializations and the independently
# meaningful JSON/manifest surfaces, so a coherent synthetic splice with the
# same headline metrics is not a valid historical failure.
EXPECTED_RUN_ARTIFACT_PINS: dict[str, dict[str, int | str]] = {
    "training_summary": {
        "size_bytes": 236694,
        "sha256": "2f582138f6751fda4392e12b6398745c89b176ce3afe6ec25875b337376cb9b4",
    },
    "best_checkpoint": {
        "size_bytes": 39442731,
        "sha256": "2c800d418088fa11dcfd11eaacd7e14bbc6a4b4be820ddfefed590853f06ec81",
    },
    "last_checkpoint": {
        "size_bytes": 39442795,
        "sha256": "7186ce22a4f5981021a8f220f5f772b7777c324eb89cb2cfc357707b5297a742",
    },
    "training_labels": {
        "size_bytes": 68944,
        "sha256": "f5f0c26b20dba7e848a63d98b204e6125fd7e2c9f7f1dec5bff94a93ffa5123f",
    },
    "training_recipe": {
        "size_bytes": 1213,
        "sha256": "76e98292f7309c2e8e6e21f575deb39dccd22c14370b29b57cfb88aefc32b4a6",
    },
    "blind_manifest": {
        "size_bytes": 202226294,
        "sha256": "c303c8a34348532263d3ad84ed2cd6ddcd77c1bdd9dfc8a7c713ccc35a1ff5f1",
    },
    "blind_contract": {
        "size_bytes": 1011,
        "sha256": "bc103913e77e35a4a54ac302ea7ce3bc7bca688f50ab8e6e3bc090b488f0d4d0",
    },
    "training_attempt": {
        "size_bytes": 844,
        "sha256": "9e6916c91073cf2cad837f2cde593e5151d8315c630074991e4e682701ef5e24",
    },
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _strict_json_bytes(raw_bytes: bytes, *, description: str) -> dict[str, Any]:
    def reject_constant(value: str) -> None:
        raise ValueError(f"{description}: non-finite JSON constant {value!r}")

    try:
        text = raw_bytes.decode("utf-8-sig")
        raw = json.loads(
            text,
            parse_constant=reject_constant,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(
            f"Unable to read strict JSON object {description}: {error}"
        ) from error
    if not isinstance(raw, dict):
        raise ValueError(f"{description}: expected a JSON object")

    def reject_nonfinite(value: object, location: str) -> None:
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError(f"{description}: non-finite JSON number at {location}")
        if isinstance(value, Mapping):
            for key, child in value.items():
                reject_nonfinite(child, f"{location}.{key}")
        elif isinstance(value, list):
            for index, child in enumerate(value):
                reject_nonfinite(child, f"{location}[{index}]")

    reject_nonfinite(raw, "$")
    return raw


def _strict_json(path: Path) -> dict[str, Any]:
    try:
        raw_bytes = path.read_bytes()
    except OSError as error:
        raise ValueError(f"Unable to read strict JSON object {path}: {error}") from error
    return _strict_json_bytes(raw_bytes, description=str(path))


def _mapping(value: object, description: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{description} must be an object")
    return value


def _require_equal(actual: object, expected: object, description: str) -> None:
    if type(actual) is not type(expected) or actual != expected:
        raise ValueError(
            f"{description} mismatch: expected {expected!r}, found {actual!r}"
        )


def _json_equivalent(actual: object, expected: object, description: str) -> None:
    try:
        actual_hash = _canonical_sha256({"value": actual})
        expected_hash = _canonical_sha256({"value": expected})
    except (TypeError, ValueError) as error:
        raise ValueError(f"{description} is not strict JSON-compatible") from error
    if actual_hash != expected_hash:
        raise ValueError(f"{description} does not match its authoritative source")


def _finite_rate(value: object, description: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{description} must be a finite rate")
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{description} must be a finite rate") from error
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError(f"{description} must be between zero and one")
    return result


def _require_hex(value: object, description: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{description} must be a lowercase SHA-256 digest")
    return value


def _is_reparse_path(path: Path) -> bool:
    if path.is_symlink():
        return True
    try:
        attributes = int(getattr(path.lstat(), "st_file_attributes", 0))
    except OSError:
        return False
    return bool(attributes & 0x400)


def _existing_non_reparse(
    path: Path, *, directory: bool, description: str
) -> Path:
    raw = Path(os.path.abspath(os.fspath(Path(path).expanduser())))
    if not os.path.lexists(os.fspath(raw)):
        raise FileNotFoundError(f"Missing {description}: {raw}")
    current = raw
    while True:
        if os.path.lexists(os.fspath(current)) and _is_reparse_path(current):
            raise ValueError(
                f"{description} must not traverse a symlink/junction/reparse path"
            )
        if current == current.parent:
            break
        current = current.parent
    resolved = raw.resolve(strict=True)
    if directory and not resolved.is_dir():
        raise ValueError(f"{description} is not a directory: {resolved}")
    if not directory and not resolved.is_file():
        raise ValueError(f"{description} is not a file: {resolved}")
    return resolved


def _fresh_output(path: Path) -> Path:
    raw = Path(os.path.abspath(os.fspath(Path(path).expanduser())))
    if os.path.lexists(os.fspath(raw)):
        raise ValueError(f"Refusing to overwrite failure attestation: {raw}")
    parent = _existing_non_reparse(
        raw.parent, directory=True, description="failure attestation parent"
    )
    return parent / raw.name


def _samefile(left: Path, right: Path, description: str) -> None:
    try:
        same = os.path.samefile(left, right)
    except OSError as error:
        raise ValueError(f"Unable to verify {description} identity") from error
    if not same:
        raise ValueError(f"{description} is not the bound file")


def _windows_programdata_attempt_registry() -> Path:
    program_data = os.environ.get("ProgramData")
    if os.name != "nt" or not isinstance(program_data, str) or not program_data:
        raise ValueError(
            "training attempt registry can only be verified against Windows %ProgramData%"
        )
    return _existing_non_reparse(
        Path(program_data) / ATTEMPT_REGISTRY_PARENT / ATTEMPT_REGISTRY_NAME,
        directory=True,
        description="Windows ProgramData training attempt registry",
    )


def _binding(path: Path, *, description: str = "bound artifact") -> dict[str, object]:
    resolved = _existing_non_reparse(path, directory=False, description=description)
    return {
        "path": str(resolved),
        "sha256": _sha256(resolved),
        "size_bytes": resolved.stat().st_size,
    }


def _binding_path(
    artifacts: Mapping[str, Any], name: str, *, description: str | None = None
) -> Path:
    binding = _mapping(artifacts.get(name), f"{name} binding")
    raw_path = binding.get("path")
    claimed_sha = _require_hex(binding.get("sha256"), f"{name} binding SHA-256")
    claimed_size = binding.get("size_bytes")
    if not isinstance(raw_path, str) or not raw_path:
        raise ValueError(f"{name} binding has no path")
    if isinstance(claimed_size, bool) or not isinstance(claimed_size, int) or claimed_size < 0:
        raise ValueError(f"{name} binding has an invalid size")
    path = _existing_non_reparse(
        Path(raw_path), directory=False, description=description or name
    )
    if path.stat().st_size != claimed_size or _sha256(path) != claimed_sha:
        raise ValueError(f"{name} changed after contract creation")
    return path


def _artifact_paths(candidate_root: Path) -> dict[str, Path]:
    training = candidate_root / "training-v14-candidate"
    blind = candidate_root / "blind-train-val"
    return {
        "training_summary": training / "training_summary.json",
        "best_checkpoint": training / "best.pt",
        "last_checkpoint": training / "last.pt",
        "training_labels": training / "labels.json",
        "training_recipe": candidate_root / "recipient_v14_training_recipe.json",
        "blind_manifest": blind / "unified_fields.train-val.jsonl",
        "blind_contract": blind / "blind.contract.json",
    }


def _code_paths() -> dict[str, Path]:
    package = Path(__file__).resolve().parent
    repository = package.parents[1]
    return {
        "code_failure_attestor": Path(__file__).resolve(),
        "code_candidate_source_attestor": package
        / "recipient_full_crop_candidate_source.py",
        "code_full_crop_pilot": package / "recipient_full_crop_pilot.py",
        "code_blind_manifest": package / "recipient_blind_manifest.py",
        "code_ocr_unified": package / "ocr_unified.py",
        "script_failure_attestor": repository
        / "scripts"
        / "receipt-ocr-recipient-v14-failure-attest.py",
        "script_candidate_source_attestor": repository
        / "scripts"
        / "receipt-ocr-recipient-full-crop-candidate-source.py",
        "script_v14_candidate": repository
        / "scripts"
        / "receipt-ocr-recipient-v14-candidate-4090.ps1",
    }


def _validated_expected_pin(
    expected: Mapping[str, object], *, description: str
) -> tuple[str, int]:
    expected_sha = _require_hex(expected.get("sha256"), f"{description} SHA-256")
    expected_size = expected.get("size_bytes")
    if (
        isinstance(expected_size, bool)
        or not isinstance(expected_size, int)
        or expected_size < 0
    ):
        raise AssertionError(f"{description} size is invalid")
    return expected_sha, expected_size


def _require_bytes_pin(
    raw_bytes: bytes,
    expected: Mapping[str, object],
    *,
    description: str,
) -> tuple[str, int]:
    expected_sha, expected_size = _validated_expected_pin(
        expected, description=description
    )
    observed = (hashlib.sha256(raw_bytes).hexdigest(), len(raw_bytes))
    if observed != (expected_sha, expected_size):
        raise ValueError(
            f"{description} pin mismatch: expected sha256={expected_sha}, "
            f"size={expected_size}; found sha256={observed[0]}, size={observed[1]}"
        )
    return observed


def _iter_bound_file_strings(value: object, location: str = "$"):
    if isinstance(value, Mapping):
        for key, child in value.items():
            yield from _iter_bound_file_strings(child, f"{location}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from _iter_bound_file_strings(child, f"{location}[{index}]")
    elif isinstance(value, str):
        suffix = Path(value).suffix.lower()
        if suffix in {".json", ".jsonl", ".pt"}:
            yield location, value, suffix


def _freeze_bound_artifact(
    artifacts: Mapping[str, Any], name: str, *, description: str
) -> dict[str, object]:
    binding = _mapping(artifacts.get(name), f"{description} binding")
    raw_path = binding.get("path")
    if not isinstance(raw_path, str) or not raw_path:
        raise ValueError(f"{description} has no path")
    expected_sha, expected_size = _validated_expected_pin(
        binding, description=description
    )
    path = _existing_non_reparse(
        Path(raw_path), directory=False, description=description
    )
    try:
        with path.open("rb") as stream:
            opened_stat = os.fstat(stream.fileno())
            raw_bytes = stream.read()
            closed_stat = os.fstat(stream.fileno())
    except OSError as error:
        raise ValueError(f"Unable to freeze {description}: {error}") from error
    opened_identity = (int(opened_stat.st_dev), int(opened_stat.st_ino))
    if opened_identity != (int(closed_stat.st_dev), int(closed_stat.st_ino)):
        raise ValueError(f"{description} identity changed while it was frozen")
    _require_bytes_pin(raw_bytes, binding, description=description)
    if (
        int(opened_stat.st_size) != expected_size
        or int(closed_stat.st_size) != expected_size
    ):
        raise ValueError(f"{description} changed while it was frozen")
    try:
        path_stat = path.stat()
    except OSError as error:
        raise ValueError(f"Unable to close {description} identity: {error}") from error
    if (
        (int(path_stat.st_dev), int(path_stat.st_ino)) != opened_identity
        or path_stat.st_size != expected_size
        or hashlib.sha256(raw_bytes).hexdigest() != expected_sha
    ):
        raise ValueError(f"{description} changed while it was frozen")
    return {
        "path": path,
        "identity": opened_identity,
        "sha256": expected_sha,
        "size_bytes": expected_size,
        "bytes": raw_bytes,
        "roles": (),
    }


def _freeze_observed_artifact(path: Path, *, description: str) -> dict[str, object]:
    resolved = _existing_non_reparse(
        path, directory=False, description=description
    )
    try:
        with resolved.open("rb") as stream:
            opened_stat = os.fstat(stream.fileno())
            raw_bytes = stream.read()
            closed_stat = os.fstat(stream.fileno())
    except OSError as error:
        raise ValueError(f"Unable to freeze {description}: {error}") from error
    identity = (int(opened_stat.st_dev), int(opened_stat.st_ino))
    if (
        identity != (int(closed_stat.st_dev), int(closed_stat.st_ino))
        or int(opened_stat.st_size) != len(raw_bytes)
        or int(closed_stat.st_size) != len(raw_bytes)
    ):
        raise ValueError(f"{description} changed while it was frozen")
    try:
        current_stat = resolved.stat()
    except OSError as error:
        raise ValueError(f"Unable to close {description}: {error}") from error
    if (
        (int(current_stat.st_dev), int(current_stat.st_ino)) != identity
        or int(current_stat.st_size) != len(raw_bytes)
    ):
        raise ValueError(f"{description} identity changed while it was frozen")
    return {
        "path": resolved,
        "identity": identity,
        "sha256": hashlib.sha256(raw_bytes).hexdigest(),
        "size_bytes": len(raw_bytes),
        "bytes": raw_bytes,
        "roles": (),
    }


def _same_frozen_identity(path: Path, frozen: Mapping[str, object]) -> bool:
    try:
        stat = path.stat()
        identity = (int(stat.st_dev), int(stat.st_ino))
        frozen_path = Path(str(frozen["path"]))
        return identity == frozen.get("identity") and os.path.samefile(path, frozen_path)
    except (KeyError, OSError, TypeError, ValueError):
        return False


def _merge_frozen_artifact(
    frozen_by_identity: dict[tuple[int, int], dict[str, object]],
    frozen: Mapping[str, object],
    *,
    role: str,
) -> dict[str, object]:
    identity = frozen.get("identity")
    if not (
        isinstance(identity, tuple)
        and len(identity) == 2
        and all(isinstance(part, int) for part in identity)
    ):
        raise AssertionError("frozen authority artifact identity is invalid")
    previous = frozen_by_identity.get(identity)
    if previous is not None:
        if (
            previous.get("sha256") != frozen.get("sha256")
            or previous.get("size_bytes") != frozen.get("size_bytes")
            or previous.get("bytes") != frozen.get("bytes")
        ):
            raise ValueError("authority artifact has conflicting frozen bindings")
        roles = set(previous.get("roles", ()))
        roles.add(role)
        previous["roles"] = tuple(sorted(str(item) for item in roles))
        return previous
    merged = dict(frozen)
    merged["roles"] = (role,)
    frozen_by_identity[identity] = merged
    return merged


def _unique_frozen_artifacts(
    artifacts: Sequence[Mapping[str, object]],
) -> tuple[Mapping[str, object], ...]:
    by_identity: dict[tuple[int, int], Mapping[str, object]] = {}
    for frozen in artifacts:
        identity = frozen.get("identity")
        if not (
            isinstance(identity, tuple)
            and len(identity) == 2
            and all(isinstance(part, int) for part in identity)
        ):
            raise AssertionError("frozen artifact identity is invalid")
        previous = by_identity.get(identity)
        if previous is not None:
            if any(
                previous.get(key) != frozen.get(key)
                for key in ("sha256", "size_bytes", "bytes")
            ):
                raise ValueError("duplicate frozen artifact bindings conflict")
            continue
        by_identity[identity] = frozen
    return tuple(by_identity.values())


def _freeze_sanitizer_lineage(
    frozen_by_identity: dict[tuple[int, int], dict[str, object]],
    *,
    torch: Any,
) -> None:
    seeds = [
        frozen
        for frozen in frozen_by_identity.values()
        if "seed_checkpoint" in set(frozen.get("roles", ()))
    ]
    if not seeds:
        raise ValueError("source/A8 authority closure has no sanitized seed checkpoint")
    for seed_index, seed in enumerate(seeds):
        raw_seed = seed.get("bytes")
        if not isinstance(raw_seed, bytes):
            raise AssertionError("frozen sanitizer seed bytes are invalid")
        seed_payload = _load_checkpoint(io.BytesIO(raw_seed), torch=torch)
        attestation = _seed_sanitizer_verifier.validate_recipient_full_crop_seed_attestation(
            seed_payload
        )
        attestation = _mapping(
            attestation, f"frozen sanitizer seed {seed_index} attestation"
        )
        raw_lineage = _mapping(
            attestation.get("train_only_recipient_lineage"),
            f"frozen sanitizer seed {seed_index} lineage",
        )
        raw_entries = raw_lineage.get("entries")
        if not isinstance(raw_entries, Sequence) or isinstance(
            raw_entries, (str, bytes)
        ):
            raise ValueError("frozen sanitizer lineage entries are invalid")
        descriptors: list[tuple[str, Mapping[str, Any]]] = [
            (
                "sanitizer_status_checkpoint",
                _mapping(
                    attestation.get("status_checkpoint"),
                    "frozen sanitizer status checkpoint",
                ),
            ),
            (
                "sanitizer_train_checkpoint",
                _mapping(
                    attestation.get("train_only_recipient_checkpoint"),
                    "frozen sanitizer train checkpoint",
                ),
            ),
        ]
        for index, raw_entry in enumerate(raw_entries):
            entry = _mapping(raw_entry, f"frozen sanitizer lineage entry {index}")
            descriptors.append(
                (
                    f"sanitizer_lineage_checkpoint_{index}",
                    _mapping(
                        entry.get("checkpoint"),
                        f"frozen sanitizer lineage checkpoint {index}",
                    ),
                )
            )
        for role, descriptor in descriptors:
            raw_path = descriptor.get("path")
            if not isinstance(raw_path, str) or not raw_path:
                raise ValueError(f"frozen {role} has no path")
            resolved = _existing_non_reparse(
                Path(raw_path), directory=False, description=f"frozen {role}"
            )
            stat = resolved.stat()
            identity = (int(stat.st_dev), int(stat.st_ino))
            frozen = frozen_by_identity.get(identity)
            if frozen is not None:
                expected_sha, expected_size = _validated_expected_pin(
                    descriptor, description=f"frozen {role}"
                )
                if (
                    frozen.get("sha256") != expected_sha
                    or frozen.get("size_bytes") != expected_size
                    or not _same_frozen_identity(resolved, frozen)
                ):
                    raise ValueError("sanitizer lineage has conflicting bindings")
            else:
                frozen = _freeze_bound_artifact(
                    {role: descriptor}, role, description=f"frozen {role}"
                )
            _merge_frozen_artifact(frozen_by_identity, frozen, role=role)

        lineage_records = [
            frozen
            for frozen in frozen_by_identity.values()
            if any(
                str(role).startswith("sanitizer_train_checkpoint")
                or str(role).startswith("sanitizer_lineage_checkpoint_")
                for role in frozen.get("roles", ())
            )
        ]
        all_checkpoints = [
            frozen
            for frozen in frozen_by_identity.values()
            if Path(str(frozen["path"])).suffix.lower() == ".pt"
        ]
        for frozen in lineage_records:
            raw_checkpoint = frozen.get("bytes")
            if not isinstance(raw_checkpoint, bytes):
                raise AssertionError("frozen sanitizer lineage bytes are invalid")
            payload = _load_checkpoint(io.BytesIO(raw_checkpoint), torch=torch)
            initialization = _mapping(
                payload.get("initialization"),
                "frozen sanitizer lineage initialization",
            )
            if initialization.get("mode") == "random":
                continue
            raw_parent = initialization.get("checkpoint_path")
            claimed_parent_sha = _require_hex(
                initialization.get("checkpoint_sha256"),
                "frozen sanitizer lineage parent SHA-256",
            )
            if not isinstance(raw_parent, str) or not raw_parent:
                raise ValueError("frozen sanitizer lineage has no parent path")
            parent = _existing_non_reparse(
                Path(raw_parent),
                directory=False,
                description="frozen sanitizer lineage parent",
            )
            matches = [
                candidate
                for candidate in all_checkpoints
                if _same_frozen_identity(parent, candidate)
            ]
            if len(matches) != 1 or matches[0].get("sha256") != claimed_parent_sha:
                raise ValueError(
                    "frozen sanitizer lineage parent has no frozen hash/size/identity binding"
                )


def _freeze_authority_closure(
    documents: Sequence[tuple[str, Mapping[str, Any]]],
    *,
    torch: Any,
) -> tuple[dict[str, object], ...]:
    frozen_by_identity: dict[tuple[int, int], dict[str, object]] = {}
    parsed_documents: list[tuple[str, Mapping[str, Any]]] = list(documents)
    for document_name, document in documents:
        artifacts = _mapping(
            document.get("artifacts"), f"frozen {document_name} artifacts"
        )
        for raw_name, raw_binding in artifacts.items():
            name = str(raw_name)
            binding = _mapping(
                raw_binding, f"frozen {document_name} {name} binding"
            )
            raw_path = binding.get("path")
            if not isinstance(raw_path, str) or not raw_path:
                raise ValueError(f"frozen {document_name} {name} has no path")
            suffix = Path(raw_path).suffix.lower()
            if suffix not in {".json", ".jsonl", ".pt"}:
                continue
            resolved = _existing_non_reparse(
                Path(raw_path),
                directory=False,
                description=f"frozen {document_name} {name}",
            )
            stat = resolved.stat()
            identity = (int(stat.st_dev), int(stat.st_ino))
            frozen = frozen_by_identity.get(identity)
            if frozen is not None:
                expected_sha, expected_size = _validated_expected_pin(
                    binding, description=f"frozen {document_name} {name}"
                )
                if (
                    frozen.get("sha256") != expected_sha
                    or frozen.get("size_bytes") != expected_size
                    or not _same_frozen_identity(resolved, frozen)
                ):
                    raise ValueError(
                        "authority artifact has conflicting duplicate bindings"
                    )
            else:
                frozen = _freeze_bound_artifact(
                    artifacts,
                    name,
                    description=f"frozen {document_name} {name}",
                )
            frozen = _merge_frozen_artifact(
                frozen_by_identity, frozen, role=name
            )
            if suffix == ".json":
                parsed_documents.append(
                    (
                        f"{document_name}.{name}",
                        _strict_json_bytes(
                            frozen["bytes"],
                            description=f"frozen {document_name} {name}",
                        ),
                    )
                )

        for root_key in ("pilot_root", "candidate_root"):
            raw_root = document.get(root_key)
            if not isinstance(raw_root, str) or not raw_root:
                continue
            root = _existing_non_reparse(
                Path(raw_root),
                directory=True,
                description=f"frozen {document_name} {root_key}",
            )
            for nested_path in root.rglob("*"):
                if nested_path.suffix.lower() not in {".json", ".jsonl"}:
                    continue
                resolved_nested = _existing_non_reparse(
                    nested_path,
                    directory=False,
                    description=f"frozen {document_name} nested data",
                )
                nested_stat = resolved_nested.stat()
                nested_identity = (
                    int(nested_stat.st_dev),
                    int(nested_stat.st_ino),
                )
                frozen = frozen_by_identity.get(nested_identity)
                if frozen is None:
                    frozen = _freeze_observed_artifact(
                        resolved_nested,
                        description=f"frozen {document_name} nested data",
                    )
                merged = _merge_frozen_artifact(
                    frozen_by_identity,
                    frozen,
                    role=f"nested_{root_key}_{nested_path.name}",
                )
                if nested_path.suffix.lower() == ".json":
                    raw_nested = merged.get("bytes")
                    if not isinstance(raw_nested, bytes):
                        raise AssertionError("frozen nested JSON bytes are invalid")
                    parsed_documents.append(
                        (
                            f"{document_name}.{root_key}.{nested_path.name}",
                            _strict_json_bytes(
                                raw_nested,
                                description=f"frozen nested JSON {nested_path}",
                            ),
                        )
                    )

    _freeze_sanitizer_lineage(frozen_by_identity, torch=torch)

    frozen_checkpoints = tuple(
        frozen
        for frozen in frozen_by_identity.values()
        if Path(str(frozen["path"])).suffix.lower() == ".pt"
    )
    frozen_json = tuple(
        frozen
        for frozen in frozen_by_identity.values()
        if Path(str(frozen["path"])).suffix.lower() == ".json"
    )
    frozen_jsonl = tuple(
        frozen
        for frozen in frozen_by_identity.values()
        if Path(str(frozen["path"])).suffix.lower() == ".jsonl"
    )
    if not frozen_checkpoints:
        raise ValueError("source/A8 authority closure has no bound checkpoint")

    for document_name, document in parsed_documents:
        for location, raw_path, suffix in _iter_bound_file_strings(document):
            path = _existing_non_reparse(
                Path(raw_path),
                directory=False,
                description=f"frozen {document_name} file at {location}",
            )
            candidates = {
                ".pt": frozen_checkpoints,
                ".json": frozen_json,
                ".jsonl": frozen_jsonl,
            }[suffix]
            if not any(_same_frozen_identity(path, frozen) for frozen in candidates):
                raise ValueError(
                    f"frozen {document_name} file at {location} has no frozen "
                    "artifact hash/size/identity binding"
                )
    return tuple(frozen_by_identity.values())


def _reverify_frozen_authority_closure(
    frozen_artifacts: Sequence[Mapping[str, object]],
) -> None:
    for frozen in frozen_artifacts:
        raw_path = frozen.get("path")
        raw_bytes = frozen.get("bytes")
        expected_sha = frozen.get("sha256")
        expected_size = frozen.get("size_bytes")
        if not isinstance(raw_path, Path) or not isinstance(raw_bytes, bytes):
            raise AssertionError("frozen authority closure record is invalid")
        try:
            path = _existing_non_reparse(
                raw_path,
                directory=False,
                description="closing frozen authority artifact",
            )
            with path.open("rb") as stream:
                opened_stat = os.fstat(stream.fileno())
                closing_bytes = stream.read()
                closed_stat = os.fstat(stream.fileno())
        except OSError as error:
            raise ValueError(
                f"Unable to close frozen authority artifact {raw_path}: {error}"
            ) from error
        opened_identity = (int(opened_stat.st_dev), int(opened_stat.st_ino))
        if (
            opened_identity != frozen.get("identity")
            or opened_identity
            != (int(closed_stat.st_dev), int(closed_stat.st_ino))
            or not _same_frozen_identity(path, frozen)
        ):
            raise ValueError(
                "frozen authority artifact identity changed during verification"
            )
        if (
            int(opened_stat.st_size) != expected_size
            or int(closed_stat.st_size) != expected_size
            or len(closing_bytes) != expected_size
            or hashlib.sha256(closing_bytes).hexdigest() != expected_sha
            or closing_bytes != raw_bytes
        ):
            raise ValueError(
                "frozen authority artifact changed during verification; "
                "frozen artifact changed during attestation"
            )


def _validate_frozen_authority_document(
    path: Path, *, name: str
) -> tuple[dict[str, Any], dict[str, object]]:
    if set(EXPECTED_AUTHORITY_DOCUMENT_PINS) != {
        "source_contract",
        "candidate_pilot_evidence",
    }:
        raise AssertionError("frozen source/A8 authority pin set changed")
    expected = _mapping(
        EXPECTED_AUTHORITY_DOCUMENT_PINS.get(name), f"frozen {name} pin"
    )
    try:
        with path.open("rb") as stream:
            stat = os.fstat(stream.fileno())
            raw_bytes = stream.read()
    except OSError as error:
        raise ValueError(f"Unable to freeze {name}: {error}") from error
    observed_sha, observed_size = _require_bytes_pin(
        raw_bytes, expected, description=f"frozen {name}"
    )
    identity = (int(stat.st_dev), int(stat.st_ino))
    try:
        current_stat = path.stat()
    except OSError as error:
        raise ValueError(f"Unable to close frozen {name}: {error}") from error
    if (int(current_stat.st_dev), int(current_stat.st_ino)) != identity:
        raise ValueError(f"frozen {name} identity changed while it was read")
    document = _strict_json_bytes(raw_bytes, description=f"frozen {name}")
    claimed = document.get("integrity_sha256")
    unsigned = {
        key: value for key, value in document.items() if key != "integrity_sha256"
    }
    if not isinstance(claimed, str) or claimed != _canonical_sha256(unsigned):
        raise ValueError(f"frozen {name} integrity hash does not match")
    return document, {
        "path": path,
        "identity": identity,
        "sha256": observed_sha,
        "size_bytes": observed_size,
        "bytes": raw_bytes,
    }


def _validate_candidate_pilot_source_cross_binding(
    pilot_document: Mapping[str, Any],
    *,
    frozen_source: Mapping[str, object],
) -> None:
    artifacts = _mapping(
        pilot_document.get("artifacts"),
        "frozen candidate-pilot artifacts",
    )
    binding = _mapping(
        artifacts.get("source_contract"),
        "frozen candidate-pilot source contract binding",
    )
    bound_sha, bound_size = _validated_expected_pin(
        binding,
        description="frozen candidate-pilot source contract binding",
    )
    if (
        bound_sha != frozen_source.get("sha256")
        or bound_size != frozen_source.get("size_bytes")
    ):
        raise ValueError(
            "frozen candidate-pilot source contract binding does not match "
            "the frozen source authority"
        )
    raw_path = binding.get("path")
    if not isinstance(raw_path, str) or not raw_path:
        raise ValueError(
            "frozen candidate-pilot source contract binding has no path"
        )
    bound_path = _existing_non_reparse(
        Path(raw_path),
        directory=False,
        description="frozen candidate-pilot source contract binding",
    )
    if not _same_frozen_identity(bound_path, frozen_source):
        raise ValueError(
            "frozen candidate-pilot source contract binding is not the "
            "frozen source authority file"
        )


def _freeze_fixed_run_artifact(path: Path, *, name: str) -> dict[str, object]:
    expected = _mapping(
        EXPECTED_RUN_ARTIFACT_PINS.get(name), f"fresh60 fixed {name} pin"
    )
    return _freeze_bound_artifact(
        {
            name: {
                "path": str(path),
                "sha256": expected.get("sha256"),
                "size_bytes": expected.get("size_bytes"),
            }
        },
        name,
        description=f"fresh60 fixed {name}",
    )


def _freeze_fixed_run_artifacts(
    paths: Mapping[str, Path],
    *,
    existing_frozen: Sequence[Mapping[str, object]] = (),
) -> dict[str, dict[str, object]]:
    if set(EXPECTED_RUN_ARTIFACT_PINS) != {
        "training_summary",
        "best_checkpoint",
        "last_checkpoint",
        "training_labels",
        "training_recipe",
        "blind_manifest",
        "blind_contract",
        "training_attempt",
    }:
        raise AssertionError("fresh60 fixed artifact pin set changed")
    frozen: dict[str, dict[str, object]] = {}
    try:
        for name in EXPECTED_RUN_ARTIFACT_PINS:
            path = paths.get(name)
            if not isinstance(path, Path):
                raise ValueError(f"fresh60 fixed artifact path is missing {name}")
            expected = _mapping(
                EXPECTED_RUN_ARTIFACT_PINS.get(name), f"fresh60 fixed {name} pin"
            )
            resolved = _existing_non_reparse(
                path, directory=False, description=f"fresh60 fixed {name}"
            )
            matches = [
                candidate
                for candidate in existing_frozen
                if _same_frozen_identity(resolved, candidate)
            ]
            if len(matches) > 1:
                raise AssertionError(
                    "fixed run artifact has duplicate frozen identities"
                )
            if matches:
                _require_bytes_pin(
                    _frozen_bytes(
                        matches[0], description=f"fresh60 fixed {name}"
                    ),
                    expected,
                    description=f"fresh60 fixed {name}",
                )
                frozen[name] = dict(matches[0])
            else:
                frozen[name] = _freeze_fixed_run_artifact(resolved, name=name)
    except Exception:
        _reverify_frozen_authority_closure(tuple(frozen.values()))
        raise
    return frozen


def _frozen_bytes(
    frozen: Mapping[str, object], *, description: str
) -> bytes:
    raw_bytes = frozen.get("bytes")
    if not isinstance(raw_bytes, bytes):
        raise AssertionError(f"{description} frozen bytes are invalid")
    return raw_bytes


def _binding_from_frozen(
    path: Path,
    frozen: Mapping[str, object],
    *,
    description: str,
) -> dict[str, object]:
    resolved = _existing_non_reparse(
        path, directory=False, description=description
    )
    if not _same_frozen_identity(resolved, frozen):
        raise ValueError(f"{description} is not the frozen file")
    sha256 = _require_hex(frozen.get("sha256"), f"{description} SHA-256")
    size_bytes = frozen.get("size_bytes")
    if (
        isinstance(size_bytes, bool)
        or not isinstance(size_bytes, int)
        or size_bytes < 0
    ):
        raise AssertionError(f"{description} frozen size is invalid")
    return {
        "path": str(resolved),
        "sha256": sha256,
        "size_bytes": size_bytes,
    }


def _validate_fixed_run_pins(
    snapshot: Mapping[str, tuple[str, int]],
) -> None:
    if set(EXPECTED_RUN_ARTIFACT_PINS) != {
        "training_summary",
        "best_checkpoint",
        "last_checkpoint",
        "training_labels",
        "training_recipe",
        "blind_manifest",
        "blind_contract",
        "training_attempt",
    }:
        raise AssertionError("fresh60 fixed artifact pin set changed")
    for name, expected in EXPECTED_RUN_ARTIFACT_PINS.items():
        observed = snapshot.get(name)
        if observed is None:
            raise ValueError(f"fresh60 fixed artifact pin is missing {name}")
        expected_sha, expected_size = _validated_expected_pin(
            expected, description=f"fresh60 fixed {name}"
        )
        if observed != (expected_sha, expected_size):
            raise ValueError(
                f"fresh60 fixed {name} pin mismatch: expected "
                f"sha256={expected_sha}, size={expected_size}; found "
                f"sha256={observed[0]}, size={observed[1]}"
            )


_SUBJECT_TRAINING_ARG_KEYS = (
    "device",
    "epochs",
    "batch_size",
    "learning_rate",
    "validation_every",
    "seed",
    "num_workers",
    "prefetch_factor",
    "persistent_workers",
    "cuda_tf32",
    "cudnn_benchmark",
)

_VALIDATION_TIMING_KEYS = frozenset(
    {
        "validation_seconds",
        "epoch_seconds",
        "val_seconds",
        "val_duration_seconds",
        "val_elapsed_seconds",
        "val_wall_seconds",
    }
)


def _is_validation_curve_semantic_key(key: str) -> bool:
    if key in _VALIDATION_TIMING_KEYS or key.endswith(("_seconds", "_milliseconds")):
        return False
    return (
        key.startswith("val_")
        or key.startswith("checkpoint_selection_")
        or key == "checkpoint_protection"
        or key.startswith("checkpoint_protection_")
    )


def _path_free_curve_value(value: object, *, description: str) -> object:
    if isinstance(value, Mapping):
        result: dict[str, object] = {}
        for raw_key, raw_value in sorted(value.items(), key=lambda item: str(item[0])):
            if not isinstance(raw_key, str) or not raw_key:
                raise ValueError(f"{description} has a non-string semantic key")
            normalized = raw_key.casefold().replace("-", "_")
            if normalized in {
                "path",
                "root",
                "file",
                "directory",
                "sha256",
                "size_bytes",
            } or normalized.endswith(
                (
                    "_path",
                    "_root",
                    "_file",
                    "_directory",
                    "_sha256",
                    "_size_bytes",
                )
            ):
                raise ValueError(f"{description} contains path/artifact identity data")
            result[raw_key] = _path_free_curve_value(
                raw_value, description=f"{description}.{raw_key}"
            )
        return result
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [
            _path_free_curve_value(item, description=f"{description}[{index}]")
            for index, item in enumerate(value)
        ]
    if isinstance(value, str):
        windows_absolute = (
            len(value) >= 3
            and value[0].isalpha()
            and value[1] == ":"
            and value[2] in {"/", "\\"}
        ) or value.startswith("\\\\")
        if value.startswith("/") or windows_absolute:
            raise ValueError(f"{description} contains an absolute path")
    return value


def _validated_curve_semantics(summary: Mapping[str, Any]) -> list[dict[str, object]]:
    raw_records = summary.get("records")
    if not isinstance(raw_records, Sequence) or isinstance(
        raw_records, (str, bytes)
    ):
        raise ValueError("subject validation curve has invalid records")
    result: list[dict[str, object]] = []
    for raw_record in raw_records:
        record = _mapping(raw_record, "subject validation-curve record")
        epoch = record.get("epoch")
        if epoch not in EXPECTED_VALIDATION_EPOCHS:
            continue
        _require_equal(
            record.get("validation_performed"),
            True,
            f"subject validation-curve epoch {epoch}",
        )
        semantic_keys = sorted(
            key
            for key in record
            if isinstance(key, str) and _is_validation_curve_semantic_key(key)
        )
        result.append(
            {
                "epoch": epoch,
                **{
                    key: _path_free_curve_value(
                        record.get(key),
                        description=f"subject validation-curve epoch {epoch} {key}",
                    )
                    for key in semantic_keys
                },
            }
        )
    if [record["epoch"] for record in result] != EXPECTED_VALIDATION_EPOCHS:
        raise ValueError("subject validation curve is incomplete")
    return result


def _failure_subject_material(
    *,
    source_subject_id: str,
    candidate_pilot_subject_id: str,
    attempt_id: str,
    blind_semantic: Mapping[str, object],
    observed_failure: Mapping[str, object],
    recipe: Mapping[str, Any],
    summary: Mapping[str, Any],
) -> dict[str, object]:
    training_args = _mapping(recipe.get("training_args"), "subject training args")
    initialization = _mapping(
        summary.get("initialization"), "subject initialization"
    )
    return {
        "domain": SUBJECT_DOMAIN,
        "kind": KIND,
        "authorization": AUTHORIZATION,
        "source_subject_id": source_subject_id,
        "candidate_pilot_subject_id": candidate_pilot_subject_id,
        "attempt_id": attempt_id,
        "blind_manifest": dict(blind_semantic),
        "observed_failure": dict(observed_failure),
        "per_validation_epoch_curve": _validated_curve_semantics(summary),
        "recipe_semantics": {
            "stage": recipe.get("stage"),
            "training_args": {
                key: training_args.get(key) for key in _SUBJECT_TRAINING_ARG_KEYS
            },
            "recipient_train_augmentation_policy": summary.get(
                "recipient_train_augmentation_policy"
            ),
            "recipient_train_split_policy": summary.get(
                "recipient_train_split_policy"
            ),
            "validation_schedule": {
                "epochs": EXPECTED_EPOCHS,
                "validation_every": training_args.get("validation_every"),
                "validated_epochs": list(EXPECTED_VALIDATION_EPOCHS),
            },
        },
        "config_transition": {
            "mode": initialization.get("mode"),
            "init_checkpoint_mode": initialization.get("init_checkpoint_mode"),
            "source_kind": initialization.get("source_kind"),
            "source_config": initialization.get("source_config"),
            "target_config": summary.get("config"),
            "financial_label_policy": initialization.get("financial_label_policy"),
            "fine_tune_policy": summary.get("fine_tune_policy"),
        },
        "selector_semantics": {
            "checkpoint_selection_policy": summary.get(
                "checkpoint_selection_policy"
            ),
            "best_checkpoint_epoch": summary.get("best_checkpoint_epoch"),
            "best_checkpoint_score": summary.get("best_checkpoint_score"),
        },
        "fixed_floors": {
            "recipient_strictly_above": RECIPIENT_DELIVERY_FLOOR,
            "amount_candidate_exact": AMOUNT_FLOOR,
            "time_candidate_exact": TIME_FLOOR,
            "payment_candidate_exact": PAYMENT_FLOOR,
            "visible_status_raw_exact": STATUS_TEXT_FLOOR,
            "status_non_success_to_success_max": 0,
        },
        "status_text_runtime_policy": summary.get("status_text_runtime_policy"),
    }


def _exact_count_metric(
    metric: Mapping[str, Any], *, description: str
) -> tuple[int, float]:
    records = metric.get("records")
    matches = metric.get("exact_matches")
    if (
        isinstance(records, bool)
        or not isinstance(records, int)
        or records != EXPECTED_RECIPIENT_VAL_RECORDS
    ):
        raise ValueError(
            f"{description} records must equal {EXPECTED_RECIPIENT_VAL_RECORDS}"
        )
    if (
        isinstance(matches, bool)
        or not isinstance(matches, int)
        or not 0 <= matches <= records
    ):
        raise ValueError(f"{description} exact_matches is invalid")
    exact = _finite_rate(metric.get("exact_match"), f"{description} exact_match")
    if not math.isclose(exact, matches / records, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(f"{description} count/rate is inconsistent")
    return matches, exact


def _validate_summary_recipe(
    summary: Mapping[str, Any],
    *,
    recipe: Mapping[str, Any],
    source_subject_id: str,
    candidate_pilot_subject_id: str,
    source_checkpoint: Path,
    source_checkpoint_sha256: str,
    full_manifest_sha256: str,
) -> tuple[dict[str, object], list[Mapping[str, Any]]]:
    validate_full_crop_training_recipe(
        recipe,
        stage="candidate-60e",
        source_subject_id=source_subject_id,
        candidate_pilot_subject_id=candidate_pilot_subject_id,
        source_checkpoint_sha256=source_checkpoint_sha256,
        full_manifest_sha256=full_manifest_sha256,
    )

    _require_equal(summary.get("schema_version"), SCHEMA_VERSION, "summary schema")
    _require_equal(summary.get("kind"), KIND_V13, "summary kind")
    config = _mapping(summary.get("config"), "fresh60 config")
    initialization = _mapping(summary.get("initialization"), "fresh60 initialization")
    source_config = _mapping(
        initialization.get("source_config"), "fresh60 source config"
    )
    fine_tune = _mapping(summary.get("fine_tune_policy"), "fresh60 fine tune")
    runtime = _mapping(summary.get("training_runtime"), "fresh60 runtime")
    split_policy = _mapping(
        summary.get("recipient_train_split_policy"), "fresh60 split policy"
    )
    augmentation = _mapping(
        summary.get("recipient_train_augmentation_policy"),
        "fresh60 augmentation policy",
    )
    checkpoint_policy = _mapping(
        summary.get("checkpoint_selection_policy"),
        "fresh60 checkpoint policy",
    )
    protected = _mapping(
        checkpoint_policy.get("protected_minimum_candidate_exact"),
        "fresh60 protected floors",
    )
    try:
        source_reader = UnifiedReaderConfig(**dict(source_config))
        target_reader = UnifiedReaderConfig(**dict(config))
        source_reader.validate()
        target_reader.validate()
        _validate_recipient_visual_context_reinit_config(source_reader, target_reader)
    except (TypeError, ValueError) as error:
        raise ValueError("fresh60 summary has an invalid residual config transition") from error

    expected_augmentation = _recipient_train_augmentation_policy(
        mode="robust_v2", seed=42
    )
    if (
        config.get("recipient_backbone") != REQUIRED_BACKBONE
        or source_config.get("recipient_backbone") != REQUIRED_SOURCE_BACKBONE
        or int(config.get("architecture_version", -1)) != 13
        or int(config.get("recipient_input_height", -1)) != 128
        or int(config.get("recipient_input_width", -1)) != 1536
        or int(config.get("recipient_branch_channels", -1)) != 16
        or int(config.get("recipient_hidden_size", -1)) != 192
        or int(config.get("recipient_open_text_layers", -1)) != 4
        or int(config.get("recipient_open_text_heads", -1)) != 8
        or int(config.get("recipient_open_text_feedforward", -1)) != 1536
        or not math.isclose(
            _finite_rate(config.get("recipient_open_text_dropout"), "fresh60 dropout"),
            0.10,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        or not math.isclose(
            _finite_rate(config.get("recipient_value_left_trim"), "fresh60 trim"),
            0.0,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        or not math.isclose(
            _finite_rate(source_config.get("recipient_value_left_trim"), "source trim"),
            0.0,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        or initialization.get("mode")
        != "parameter_only_recipient_visual_context_reinit"
        or initialization.get("init_checkpoint_mode")
        != INIT_CHECKPOINT_MODE_RECIPIENT_VISUAL_CONTEXT_REINIT
        or initialization.get("source_kind") != KIND_V13
        or initialization.get("checkpoint_sha256") != source_checkpoint_sha256
        or _mapping(
            _mapping(
                initialization.get("financial_label_policy"),
                "fresh60 financial label policy",
            ).get("recipient_character_map"),
            "fresh60 recipient character map",
        ).get("mode")
        != "fresh_train_only_reinitialized_recipient_v1"
        or fine_tune.get("mode") != "recipient_only_v13"
        or fine_tune.get("trainable_parameter_prefix") != "recipient_"
        or fine_tune.get("training_forward")
        != "private_recipient_branch_only_v13"
        or runtime.get("device") != "cuda:0"
        or runtime.get("uses_cuda") is not True
        or "4090" not in str(runtime.get("cuda_device_name", ""))
        or runtime.get("num_workers") != 4
        or runtime.get("prefetch_factor") != 2
        or runtime.get("persistent_workers") is not True
        or runtime.get("validation_every") != 2
        or runtime.get("cuda_tf32_requested") is not True
        or runtime.get("cudnn_benchmark_requested") is not True
        or split_policy.get("mode") != "standard_train_only"
        or list(split_policy.get("splits", [])) != ["train"]
        or dict(augmentation) != expected_augmentation
        or checkpoint_policy.get("mode")
        != CHECKPOINT_SELECTION_RECIPIENT_PRIORITY
        or summary.get("status_text_runtime_policy")
        != STATUS_TEXT_RUNTIME_POLICY
    ):
        raise ValueError("fresh60 summary does not prove the fixed guarded recipe")

    raw_source = initialization.get("checkpoint_path")
    if not isinstance(raw_source, str) or not raw_source:
        raise ValueError("fresh60 summary does not bind its source checkpoint path")
    summary_source = _existing_non_reparse(
        Path(raw_source), directory=False, description="fresh60 source checkpoint"
    )
    _samefile(source_checkpoint, summary_source, "fresh60 source checkpoint")
    if _sha256(summary_source) != source_checkpoint_sha256:
        raise ValueError("fresh60 source checkpoint bytes changed")

    for name, floor in {
        "amount": AMOUNT_FLOOR,
        "time": TIME_FLOOR,
        "payment_method_field": PAYMENT_FLOOR,
    }.items():
        observed = _finite_rate(protected.get(name), f"fresh60 {name} floor")
        if not math.isclose(observed, floor, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(f"fresh60 {name} protection floor changed")

    field_counts = _mapping(summary.get("field_counts"), "fresh60 field counts")
    for field, raw_counts in field_counts.items():
        counts = _mapping(raw_counts, f"fresh60 {field} counts")
        _require_equal(counts.get("test"), 0, f"fresh60 {field} test count")
    recipient_counts = _mapping(
        field_counts.get("recipient_field"), "fresh60 recipient field counts"
    )
    _require_equal(
        recipient_counts.get("val"),
        EXPECTED_RECIPIENT_VAL_RECORDS,
        "fresh60 recipient val count",
    )
    recipient_oov = _mapping(
        summary.get("recipient_oov_by_split"), "fresh60 recipient OOV"
    )
    _require_equal(
        _mapping(recipient_oov.get("val"), "fresh60 val OOV").get("records"),
        EXPECTED_RECIPIENT_VAL_RECORDS,
        "fresh60 val OOV records",
    )
    _require_equal(
        _mapping(recipient_oov.get("test"), "fresh60 test OOV").get("records"),
        0,
        "fresh60 test OOV records",
    )

    raw_records = summary.get("records")
    if not isinstance(raw_records, Sequence) or isinstance(raw_records, (str, bytes)):
        raise ValueError("fresh60 summary has invalid epoch records")
    records = [_mapping(record, "fresh60 epoch record") for record in raw_records]
    if [record.get("epoch") for record in records] != list(
        range(1, EXPECTED_EPOCHS + 1)
    ):
        raise ValueError("fresh60 summary requires ordered epochs 1 through 60")

    recipient_metrics: dict[int, tuple[int, float]] = {}
    minimum_guards = {
        "amount": 1.0,
        "time": 1.0,
        "payment_method_field": 1.0,
        "visible_transfer_status_cjk_text": 1.0,
    }
    for record in records:
        epoch = int(record["epoch"])
        train_loss = record.get("train_loss")
        if (
            isinstance(train_loss, bool)
            or not isinstance(train_loss, (int, float))
            or not math.isfinite(float(train_loss))
        ):
            raise ValueError(
                f"fresh60 epoch {epoch} train_loss must be a finite int/float"
            )
        validated = epoch in EXPECTED_VALIDATION_EPOCHS
        _require_equal(
            record.get("validation_performed"),
            validated,
            f"fresh60 epoch {epoch} validation schedule",
        )
        if not validated:
            for key in (
                "val_candidate_text_by_field",
                "val_ctc_by_field",
                "val_status_non_success_to_success",
                "checkpoint_selection_score",
                "checkpoint_protection",
            ):
                _require_equal(
                    record.get(key), None, f"fresh60 epoch {epoch} skipped {key}"
                )
            _require_equal(
                record.get("checkpoint_selection_eligible"),
                False,
                f"fresh60 epoch {epoch} checkpoint eligibility",
            )
            _require_equal(
                record.get("checkpoint_selection_protection_failures"),
                ["full_validation_not_scheduled"],
                f"fresh60 epoch {epoch} checkpoint failures",
            )
            continue

        fields = _mapping(
            record.get("val_candidate_text_by_field"),
            f"fresh60 epoch {epoch} fields",
        )
        recipient = _mapping(
            fields.get("recipient_field"),
            f"fresh60 epoch {epoch} recipient metric",
        )
        recipient_metrics[epoch] = _exact_count_metric(
            recipient, description=f"fresh60 epoch {epoch} recipient metric"
        )
        for name, floor in {
            "amount": AMOUNT_FLOOR,
            "time": TIME_FLOOR,
            "payment_method_field": PAYMENT_FLOOR,
        }.items():
            metric = _mapping(fields.get(name), f"fresh60 epoch {epoch} {name}")
            exact = _finite_rate(
                metric.get("exact_match"), f"fresh60 epoch {epoch} {name} exact"
            )
            if exact < floor:
                raise ValueError(f"fresh60 epoch {epoch} violated the {name} floor")
            minimum_guards[name] = min(minimum_guards[name], exact)
        status_fields = _mapping(
            record.get("val_ctc_by_field"), f"fresh60 epoch {epoch} CTC fields"
        )
        status = _mapping(
            status_fields.get("transfer_status"),
            f"fresh60 epoch {epoch} status metric",
        )
        status_exact = _finite_rate(
            status.get("exact_match"), f"fresh60 epoch {epoch} status exact"
        )
        if status_exact < STATUS_TEXT_FLOOR:
            raise ValueError(f"fresh60 epoch {epoch} violated the status floor")
        minimum_guards["visible_transfer_status_cjk_text"] = min(
            minimum_guards["visible_transfer_status_cjk_text"], status_exact
        )
        _require_equal(
            record.get("val_status_non_success_to_success"),
            0,
            f"fresh60 epoch {epoch} unsafe status errors",
        )
        _require_equal(
            record.get("checkpoint_selection_eligible"),
            True,
            f"fresh60 epoch {epoch} checkpoint eligibility",
        )
        _require_equal(
            record.get("checkpoint_selection_protection_failures"),
            [],
            f"fresh60 epoch {epoch} checkpoint failures",
        )
        if record.get("checkpoint_selection_score") is None:
            raise ValueError(f"fresh60 epoch {epoch} has no checkpoint score")
        protection = _mapping(
            record.get("checkpoint_protection"),
            f"fresh60 epoch {epoch} checkpoint protection",
        )
        _require_equal(
            protection.get("failures"),
            [],
            f"fresh60 epoch {epoch} protection failures",
        )

    if sorted(recipient_metrics) != EXPECTED_VALIDATION_EPOCHS:
        raise ValueError("fresh60 validated epoch schedule is incomplete")
    _require_equal(
        summary.get("best_checkpoint_epoch"),
        EXPECTED_BEST_EPOCH,
        "fresh60 best checkpoint epoch",
    )
    best_matches, best_exact = recipient_metrics[EXPECTED_BEST_EPOCH]
    last_matches, last_exact = recipient_metrics[EXPECTED_LAST_EPOCH]
    _require_equal(best_matches, EXPECTED_BEST_MATCHES, "fresh60 best matches")
    _require_equal(last_matches, EXPECTED_LAST_MATCHES, "fresh60 last matches")
    if max(matches for matches, _ in recipient_metrics.values()) != EXPECTED_BEST_MATCHES:
        raise ValueError("fresh60 epoch 44 is not tied for maximum recipient matches")
    if best_exact > RECIPIENT_DELIVERY_FLOOR:
        raise ValueError("fresh60 unexpectedly passed the strict recipient gate")
    strict_pass_matches = (
        math.floor(RECIPIENT_DELIVERY_FLOOR * EXPECTED_RECIPIENT_VAL_RECORDS) + 1
    )
    if strict_pass_matches != EXPECTED_STRICT_PASS_MATCHES:
        raise AssertionError("frozen strictly-above-90-percent count changed")
    if strict_pass_matches - best_matches != 192:
        raise AssertionError("frozen fresh60 failure gap changed")
    best_record = records[EXPECTED_BEST_EPOCH - 1]
    last_record = records[EXPECTED_LAST_EPOCH - 1]
    if summary.get("best_checkpoint_score") is not None:
        _json_equivalent(
            summary.get("best_checkpoint_score"),
            best_record.get("checkpoint_selection_score"),
            "fresh60 best checkpoint score",
        )
    return (
        {
            "validated_epochs": EXPECTED_VALIDATION_EPOCHS,
            "recipient_val_records": EXPECTED_RECIPIENT_VAL_RECORDS,
            "recipient_candidate_coverage": 1.0,
            "best_epoch": EXPECTED_BEST_EPOCH,
            "best_recipient_exact_matches": best_matches,
            "best_recipient_exact": best_exact,
            "last_epoch": EXPECTED_LAST_EPOCH,
            "last_recipient_exact_matches": last_matches,
            "last_recipient_exact": last_exact,
            "strict_pass_exact_matches": strict_pass_matches,
            "strict_pass_gap_matches": strict_pass_matches - best_matches,
            "minimum_guards": minimum_guards,
            "status_non_success_to_success_max": 0,
            "candidate_result": "failed_recipient_strictly_above_90_percent_gate",
        },
        records,
    )


_CHECKPOINT_SHARED_KEYS = (
    "config",
    "initialization",
    "fine_tune_policy",
    "checkpoint_selection_policy",
    "recipient_train_split_policy",
    "field_counts",
    "status_text_runtime_policy",
    "training_runtime",
)


def _validate_checkpoint(
    frozen_bytes: bytes,
    *,
    torch: Any,
    summary: Mapping[str, Any],
    record: Mapping[str, Any],
    expected_epoch: int,
    description: str,
) -> Mapping[str, Any]:
    payload = _load_checkpoint(io.BytesIO(frozen_bytes), torch=torch)
    if not isinstance(payload, Mapping):
        raise ValueError(f"{description} payload must be an object")
    _assert_no_unsafe_true_claims(payload, location=description)
    _require_equal(payload.get("schema_version"), SCHEMA_VERSION, f"{description} schema")
    _require_equal(payload.get("kind"), KIND_V13, f"{description} kind")
    _require_equal(payload.get("epoch"), expected_epoch, f"{description} epoch")
    state = payload.get("state_dict")
    if not isinstance(state, Mapping) or not state:
        raise ValueError(f"{description} has no model state")
    _json_equivalent(payload.get("metrics"), record, f"{description} metrics")
    try:
        config = _checkpoint_config(payload)
        config.validate()
    except (TypeError, ValueError) as error:
        raise ValueError(f"{description} has an invalid config") from error
    _json_equivalent(asdict(config), summary.get("config"), f"{description} config")
    for key in _CHECKPOINT_SHARED_KEYS:
        _json_equivalent(payload.get(key), summary.get(key), f"{description} {key}")
    return payload


def _validate_labels(
    labels: Mapping[str, Any],
    *,
    summary: Mapping[str, Any],
    best: Mapping[str, Any],
    last: Mapping[str, Any],
) -> None:
    _require_equal(labels.get("schema_version"), SCHEMA_VERSION, "training labels schema")
    common = (
        "checkpoint_selection_policy",
        "initialization",
        "training_runtime",
        "fine_tune_policy",
        "recipient_oov_by_split",
        "recipient_sampling_policy",
        "recipient_confidence_policy",
        "recipient_tail_loss_policy",
        "recipient_train_augmentation_policy",
        "recipient_train_split_policy",
        "recipient_target",
        "status_text_runtime_policy",
    )
    for key in common:
        if key in summary:
            _json_equivalent(labels.get(key), summary.get(key), f"labels {key}")
        _json_equivalent(best.get(key), labels.get(key), f"best checkpoint labels {key}")
        _json_equivalent(last.get(key), labels.get(key), f"last checkpoint labels {key}")
    characters = labels.get("recipient_characters")
    if not isinstance(characters, list) or not characters or any(
        not isinstance(character, str) or not character for character in characters
    ):
        raise ValueError("training labels has an invalid recipient character map")
    charset_sha = hashlib.sha256("".join(characters).encode("utf-8")).hexdigest()
    _require_equal(
        labels.get("recipient_charset_sha256"), charset_sha, "training labels charset"
    )
    for payload, description in ((best, "best checkpoint"), (last, "last checkpoint")):
        _json_equivalent(
            payload.get("recipient_characters"), characters, f"{description} characters"
        )
        _require_equal(
            payload.get("recipient_charset_sha256"),
            charset_sha,
            f"{description} charset",
        )


def _validate_attempt(
    path: Path,
    *,
    payload: Mapping[str, Any],
    registry: Path,
    candidate_root: Path,
    source_subject_id: str,
    candidate_pilot_subject_id: str,
    full_manifest_sha256: str,
) -> tuple[dict[str, Any], str]:
    if registry.name != ATTEMPT_REGISTRY_NAME or registry.parent.name != ATTEMPT_REGISTRY_PARENT:
        raise ValueError("training attempt registry is not the ProgramData ReceiptAI registry")
    attempt_subject = (
        f"{ATTEMPT_DOMAIN}|{source_subject_id}|{candidate_pilot_subject_id}"
    )
    attempt_id = hashlib.sha256(attempt_subject.encode("utf-8")).hexdigest()
    _require_equal(attempt_id, EXPECTED_ATTEMPT_ID, "fixed fresh60 attempt id")
    expected_path = registry / f"{attempt_id}.attempt.json"
    _samefile(path, expected_path, "fresh60 one-shot attempt")
    expected_keys = {
        "schema_version",
        "kind",
        "created_at_utc",
        "attempt_id",
        "stage",
        "source_subject_id",
        "candidate_pilot_subject_id",
        "output_root",
        "full_manifest_sha256",
        "threat_model",
    }
    if set(payload) != expected_keys:
        raise ValueError("fresh60 one-shot attempt keys changed")
    _require_equal(payload.get("schema_version"), SCHEMA_VERSION, "attempt schema")
    _require_equal(payload.get("kind"), ATTEMPT_KIND, "attempt kind")
    _require_equal(payload.get("attempt_id"), attempt_id, "attempt id")
    _require_equal(payload.get("stage"), "candidate-60e", "attempt stage")
    _require_equal(
        payload.get("source_subject_id"), source_subject_id, "attempt source subject"
    )
    _require_equal(
        payload.get("candidate_pilot_subject_id"),
        candidate_pilot_subject_id,
        "attempt candidate-pilot subject",
    )
    _require_equal(
        payload.get("full_manifest_sha256"),
        full_manifest_sha256,
        "attempt full manifest",
    )
    _require_equal(payload.get("threat_model"), ATTEMPT_THREAT_MODEL, "attempt threat model")
    created = payload.get("created_at_utc")
    if not isinstance(created, str) or not created:
        raise ValueError("attempt creation timestamp is missing")
    try:
        datetime.fromisoformat(created.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError("attempt creation timestamp is invalid") from error
    output_root = payload.get("output_root")
    if not isinstance(output_root, str) or not output_root:
        raise ValueError("attempt output_root is missing")
    bound_output = _existing_non_reparse(
        Path(output_root), directory=True, description="attempt output root"
    )
    _samefile(candidate_root, bound_output, "attempt output root")
    return payload, attempt_id


_UNSAFE_TRUE_CLAIMS = {
    "test_evaluated",
    "test_labels_used",
    "test_metrics_computed",
    "test_examples_emitted",
    "test_opened",
    "test_opened_by_training",
    "external_test_artifacts_opened",
    "test_evaluation_authorized",
    "test_authorized",
    "onnx_exported",
    "onnx_export_authorized",
    "warmstart_authorized",
    "warm_start_authorized",
    "same_route_authorized",
    "same_route_retry_authorized",
    "same_route_continuation_authorized",
    "retry_authorized",
    "continuation_authorized",
    "production_authorized",
    "production_route_authorized",
    "prod_authorized",
    "prod_route_authorized",
}


def _unsafe_true_claim(key: object) -> bool:
    if not isinstance(key, str):
        return False
    normalized = key.lower().replace("-", "_")
    if normalized in _UNSAFE_TRUE_CLAIMS:
        return True
    tokens = {token for token in normalized.split("_") if token}
    if "test" in tokens:
        return True
    return bool(
        tokens.intersection(
            {
                "production",
                "prod",
                "onnx",
                "warmstart",
                "continuation",
                "retry",
            }
        )
        or {"warm", "start"}.issubset(tokens)
        or {"same", "route"}.issubset(tokens)
    )


def _assert_no_unsafe_true_claims(value: object, *, location: str) -> None:
    if isinstance(value, Mapping):
        if value.get("evaluation_split") == "test":
            raise ValueError(
                f"failed fresh60 evidence contains test evidence at {location}"
            )
        for key, child in value.items():
            if _unsafe_true_claim(key) and child is True:
                raise ValueError(f"failed fresh60 evidence contains unsafe {key} at {location}")
            _assert_no_unsafe_true_claims(child, location=f"{location}.{key}")
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for index, child in enumerate(value):
            _assert_no_unsafe_true_claims(child, location=f"{location}[{index}]")


def _assert_failed_candidate_surface(
    root: Path,
    *,
    frozen_artifacts: Sequence[Mapping[str, object]] = (),
    inspect_json: bool = True,
) -> None:
    for forbidden in (
        root / "recipient_v14_candidate.json",
        root / "artifacts",
        root / "onnx-val-gpu",
    ):
        if os.path.lexists(os.fspath(forbidden)):
            raise ValueError(f"failed fresh60 root contains forbidden candidate output: {forbidden}")

    for path in root.rglob("*"):
        if _is_reparse_path(path):
            raise ValueError("failed fresh60 root contains a reparse entry")
        if path.is_file() and path.suffix.lower() == ".onnx":
            raise ValueError("failed fresh60 root contains an ONNX artifact")
        if inspect_json and path.is_file() and path.suffix.lower() == ".json":
            lexical_path = os.path.normcase(os.path.abspath(os.fspath(path)))
            bound_matches = [
                frozen
                for frozen in frozen_artifacts
                if Path(str(frozen.get("path", ""))).suffix.lower() == ".json"
                and os.path.normcase(
                    os.path.abspath(os.fspath(Path(str(frozen.get("path")))))
                )
                == lexical_path
            ]
            if len(bound_matches) > 1:
                raise AssertionError("candidate JSON has duplicate frozen paths")
            if bound_matches and not _same_frozen_identity(path, bound_matches[0]):
                raise ValueError("candidate JSON identity changed after it was frozen")
            payload = (
                _strict_json_bytes(
                    _frozen_bytes(bound_matches[0], description="candidate JSON"),
                    description=f"frozen candidate JSON {path}",
                )
                if bound_matches
                else _strict_json(path)
            )
            try:
                _assert_no_unsafe_true_claims(payload, location=str(path))
            except ValueError as error:
                message = str(error).replace(
                    "failed fresh60 evidence", "failed fresh60 root"
                )
                raise ValueError(message) from error


def _frozen_artifact_for_path(
    path: Path,
    frozen_artifacts: Sequence[Mapping[str, object]],
    *,
    suffixes: set[str],
    description: str,
) -> tuple[Path, Mapping[str, object]]:
    resolved = _existing_non_reparse(
        Path(path), directory=False, description=description
    )
    if resolved.suffix.lower() not in suffixes:
        raise ValueError(f"{description} has an unsupported frozen file type")
    matches = [
        frozen
        for frozen in frozen_artifacts
        if Path(str(frozen.get("path", ""))).suffix.lower() in suffixes
        and _same_frozen_identity(resolved, frozen)
    ]
    if len(matches) != 1:
        raise ValueError(f"{description} is not in the frozen authority closure")
    return resolved, matches[0]


def _frozen_jsonl_rows(raw_bytes: bytes, *, description: str):
    def reject_constant(value: str) -> None:
        raise ValueError(f"{description} contains non-finite JSON constant {value!r}")

    try:
        with io.TextIOWrapper(io.BytesIO(raw_bytes), encoding="utf-8-sig") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line, parse_constant=reject_constant)
                except json.JSONDecodeError as error:
                    raise ValueError(
                        f"{description} line {line_number} is invalid JSON"
                    ) from error
                if not isinstance(row, Mapping):
                    raise ValueError(
                        f"{description} line {line_number} is not an object"
                    )
                yield line_number, row
    except UnicodeError as error:
        raise ValueError(f"{description} is not valid UTF-8") from error


def _verify_frozen_blind_manifest_contract(
    *,
    records_path: Path,
    blind_contract_path: Path,
    frozen_artifacts: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    records, frozen_records = _frozen_artifact_for_path(
        records_path,
        frozen_artifacts,
        suffixes={".jsonl"},
        description="deep frozen blind manifest",
    )
    contract_path, frozen_contract = _frozen_artifact_for_path(
        blind_contract_path,
        frozen_artifacts,
        suffixes={".json"},
        description="deep frozen blind contract",
    )
    contract_bytes = frozen_contract.get("bytes")
    records_bytes = frozen_records.get("bytes")
    if not isinstance(contract_bytes, bytes) or not isinstance(records_bytes, bytes):
        raise AssertionError("frozen blind artifact bytes are invalid")
    contract = _strict_json_bytes(
        contract_bytes, description="deep frozen blind contract"
    )
    raw_manifest = contract.get("blind_manifest")
    raw_source = contract.get("source_manifest")
    if not isinstance(raw_manifest, str) or not raw_manifest:
        raise ValueError("frozen blind contract has no bound manifest")
    if not isinstance(raw_source, str) or not raw_source:
        raise ValueError("frozen blind contract has no source manifest")
    bound_manifest, _ = _frozen_artifact_for_path(
        Path(raw_manifest),
        frozen_artifacts,
        suffixes={".jsonl"},
        description="deep frozen blind-bound manifest",
    )
    bound_source, frozen_source = _frozen_artifact_for_path(
        Path(raw_source),
        frozen_artifacts,
        suffixes={".jsonl"},
        description="deep frozen blind source manifest",
    )
    _samefile(records, bound_manifest, "deep frozen blind manifest")
    if os.path.samefile(bound_source, records):
        raise ValueError("frozen blind source and blind manifest are the same file")
    expected_sha = _require_hex(
        contract.get("blind_manifest_sha256"), "frozen blind manifest SHA-256"
    )
    source_sha = _require_hex(
        contract.get("source_manifest_sha256"), "frozen blind source SHA-256"
    )
    if (
        contract.get("schema_version")
        != _full_crop_pilot_verifier.BLIND_CONTRACT_SCHEMA_VERSION
        or contract.get("kind") != _full_crop_pilot_verifier.BLIND_MANIFEST_KIND
        or frozen_records.get("sha256") != expected_sha
        or frozen_source.get("sha256") != source_sha
        or contract.get("test_labels_used") is not False
        or contract.get("test_metrics_computed") is not False
        or contract.get("test_examples_emitted") is not False
        or contract.get("optimizer_supervision_splits") != ["train"]
        or contract.get("checkpoint_selection_splits") != ["val"]
        or contract.get("final_gate_only_splits") != ["test"]
    ):
        raise ValueError("frozen blind manifest contract is incomplete or unsafe")

    split_counts: Counter[str] = Counter()
    seen_ids: set[str] = set()
    for _, row in _frozen_jsonl_rows(
        records_bytes, description="deep frozen blind manifest"
    ):
        split = row.get("split")
        if split not in {"train", "val"}:
            raise ValueError("frozen blind manifest physically contains a test row")
        record_id = row.get("id")
        if not isinstance(record_id, str) or not record_id or record_id in seen_ids:
            raise ValueError("frozen blind manifest has a missing or duplicate record id")
        seen_ids.add(record_id)
        split_counts[str(split)] += 1
    contract_counts = _mapping(contract.get("split_counts"), "frozen blind split counts")
    try:
        expected_train = int(contract_counts.get("train", -1))
        expected_val = int(contract_counts.get("val", -1))
        excluded_test = int(contract_counts.get("test_excluded", -1))
    except (TypeError, ValueError) as error:
        raise ValueError("frozen blind manifest split counts are invalid") from error
    if (
        split_counts != Counter({"train": expected_train, "val": expected_val})
        or expected_train <= 0
        or expected_val <= 0
        or excluded_test <= 0
    ):
        raise ValueError("frozen blind manifest split counts do not match")
    return {
        "schema_version": _full_crop_pilot_verifier.BLIND_CONTRACT_SCHEMA_VERSION,
        "kind": _full_crop_pilot_verifier.BLIND_MANIFEST_KIND,
        "contract_path": str(contract_path),
        "source_manifest": str(bound_source),
        "source_manifest_sha256": source_sha,
        "blind_manifest": str(records),
        "blind_manifest_sha256": expected_sha,
        "split_counts": {
            "train": expected_train,
            "val": expected_val,
            "test_excluded": excluded_test,
        },
        "optimizer_supervision_splits": ["train"],
        "checkpoint_selection_splits": ["val"],
        "test_opened_by_training": False,
    }


def _frozen_blind_recipient_val_records(
    binding: Mapping[str, Any],
    *,
    frozen_artifacts: Sequence[Mapping[str, object]],
) -> int:
    raw_path = binding.get("blind_manifest")
    if not isinstance(raw_path, str) or not raw_path:
        raise ValueError("frozen blind binding has no manifest path")
    _, frozen = _frozen_artifact_for_path(
        Path(raw_path),
        frozen_artifacts,
        suffixes={".jsonl"},
        description="deep frozen recipient manifest",
    )
    raw_bytes = frozen.get("bytes")
    if not isinstance(raw_bytes, bytes):
        raise AssertionError("frozen recipient manifest bytes are invalid")
    records = 0
    for line_number, row in _frozen_jsonl_rows(
        raw_bytes, description="deep frozen recipient manifest"
    ):
        slots = row.get("slots")
        if not isinstance(slots, Mapping):
            raise ValueError(
                f"deep frozen recipient manifest line {line_number} has invalid slots"
            )
        recipient = slots.get("recipient_field")
        if recipient is not None and not isinstance(recipient, Mapping):
            raise ValueError(
                f"deep frozen recipient manifest line {line_number} has invalid recipient"
            )
        if isinstance(recipient, Mapping):
            text = recipient.get("text")
            if (
                not isinstance(text, str)
                or not text
                or any(not character.isprintable() for character in text)
            ):
                raise ValueError(
                    f"deep frozen recipient manifest line {line_number} has invalid text"
                )
            if row.get("split") == "val":
                records += 1
    return records


def _validated_authorities(
    *,
    source_contract_path: Path,
    source_document: Mapping[str, Any],
    candidate_pilot_evidence_path: Path,
    full_records: Path,
    frozen_artifacts: Sequence[Mapping[str, object]],
    torch: Any,
) -> tuple[dict[str, Any], dict[str, Any], Path, str, str, str]:
    source_file = _existing_non_reparse(
        source_contract_path, directory=False, description="full-crop source contract"
    )
    raw_pilot_root = source_document.get("pilot_root")
    if not isinstance(raw_pilot_root, str) or not raw_pilot_root:
        raise ValueError("full-crop source contract has no pilot root")
    original_candidate_loader = _candidate_source_verifier._load_checkpoint
    original_sanitizer_loader = _seed_sanitizer_verifier._load_checkpoint
    original_candidate_strict_json = _candidate_source_verifier._strict_json
    original_candidate_sha256 = _candidate_source_verifier._sha256
    original_sanitizer_sha256 = _seed_sanitizer_verifier._sha256
    original_blind_verifier = _candidate_source_verifier.verify_blind_manifest_contract
    original_blind_counter = _candidate_source_verifier._blind_recipient_val_records

    def frozen_loader(original_loader, *, description: str):
        def load(path: object, *, torch: Any) -> Mapping[str, object]:
            try:
                raw_path = os.fspath(path)
            except TypeError as error:
                raise ValueError(
                    f"{description} requested a non-path checkpoint"
                ) from error
            _, frozen = _frozen_artifact_for_path(
                Path(raw_path),
                frozen_artifacts,
                suffixes={".pt"},
                description=description,
            )
            raw_bytes = frozen.get("bytes")
            if not isinstance(raw_bytes, bytes):
                raise AssertionError("frozen checkpoint bytes are invalid")
            return original_loader(io.BytesIO(raw_bytes), torch=torch)

        return load

    def frozen_strict_json(path: Path) -> dict[str, Any]:
        _, frozen = _frozen_artifact_for_path(
            path,
            frozen_artifacts,
            suffixes={".json"},
            description="deep authority JSON",
        )
        raw_bytes = frozen.get("bytes")
        if not isinstance(raw_bytes, bytes):
            raise AssertionError("frozen authority JSON bytes are invalid")
        return _strict_json_bytes(raw_bytes, description="deep frozen authority JSON")

    def frozen_sha256(original_sha256, *, description: str):
        def digest(path: Path) -> str:
            suffix = Path(path).suffix.lower()
            if suffix not in {".json", ".jsonl", ".pt"}:
                return original_sha256(path)
            _, frozen = _frozen_artifact_for_path(
                path,
                frozen_artifacts,
                suffixes={suffix},
                description=description,
            )
            return _require_hex(frozen.get("sha256"), f"{description} SHA-256")

        return digest

    _candidate_source_verifier._load_checkpoint = frozen_loader(
        original_candidate_loader,
        description="deep candidate-source checkpoint",
    )
    _seed_sanitizer_verifier._load_checkpoint = frozen_loader(
        original_sanitizer_loader,
        description="deep sanitizer checkpoint",
    )
    _candidate_source_verifier._strict_json = frozen_strict_json
    _candidate_source_verifier._sha256 = frozen_sha256(
        original_candidate_sha256,
        description="deep candidate-source content",
    )
    _seed_sanitizer_verifier._sha256 = frozen_sha256(
        original_sanitizer_sha256,
        description="deep sanitizer content",
    )
    _candidate_source_verifier.verify_blind_manifest_contract = lambda **kwargs: (
        _verify_frozen_blind_manifest_contract(
            **kwargs, frozen_artifacts=frozen_artifacts
        )
    )
    _candidate_source_verifier._blind_recipient_val_records = lambda binding: (
        _frozen_blind_recipient_val_records(
            binding, frozen_artifacts=frozen_artifacts
        )
    )
    try:
        source = verify_full_crop_candidate_source(
            pilot_root=Path(raw_pilot_root),
            contract_path=source_file,
            full_records=full_records,
            torch=torch,
        )
        pilot = verify_residual_candidate_pilot(
            evidence_path=candidate_pilot_evidence_path,
            source_contract_path=source_file,
            full_records=full_records,
            torch=torch,
        )
    finally:
        _candidate_source_verifier._load_checkpoint = original_candidate_loader
        _seed_sanitizer_verifier._load_checkpoint = original_sanitizer_loader
        _candidate_source_verifier._strict_json = original_candidate_strict_json
        _candidate_source_verifier._sha256 = original_candidate_sha256
        _seed_sanitizer_verifier._sha256 = original_sanitizer_sha256
        _candidate_source_verifier.verify_blind_manifest_contract = (
            original_blind_verifier
        )
        _candidate_source_verifier._blind_recipient_val_records = (
            original_blind_counter
        )
    for payload, kind, description in (
        (source, SOURCE_KIND, "source contract"),
        (pilot, CANDIDATE_PILOT_KIND, "candidate-pilot evidence"),
    ):
        _require_equal(payload.get("schema_version"), SCHEMA_VERSION, f"{description} schema")
        _require_equal(payload.get("kind"), kind, f"{description} kind")
        _require_equal(payload.get("analysis_only"), True, f"{description} analysis_only")
        _require_equal(
            payload.get("production_route_authorized"),
            False,
            f"{description} production authorization",
        )
        _require_equal(payload.get("test_opened"), False, f"{description} test_opened")
        _require_equal(payload.get("onnx_exported"), False, f"{description} onnx_exported")
    _require_equal(pilot.get("passed"), True, "candidate-pilot passed")
    _require_equal(
        pilot.get("decision"), CANDIDATE_PILOT_DECISION, "candidate-pilot decision"
    )
    source_subject = _require_hex(source.get("source_subject_id"), "source subject id")
    _require_equal(
        source_subject, EXPECTED_SOURCE_SUBJECT_ID, "fixed source subject id"
    )
    _require_equal(
        pilot.get("source_subject_id"), source_subject, "candidate-pilot source subject"
    )
    candidate_subject = _require_hex(
        pilot.get("candidate_pilot_subject_id"), "candidate-pilot subject id"
    )
    _require_equal(
        candidate_subject,
        EXPECTED_CANDIDATE_PILOT_SUBJECT_ID,
        "fixed candidate-pilot subject id",
    )
    source_artifacts = _mapping(source.get("artifacts"), "source artifacts")
    source_checkpoint = _binding_path(
        source_artifacts,
        "best_checkpoint",
        description="attested source best checkpoint",
    )
    source_sha = _require_hex(
        _mapping(source_artifacts.get("best_checkpoint"), "source best binding").get(
            "sha256"
        ),
        "source checkpoint SHA-256",
    )
    return (
        dict(source),
        dict(pilot),
        source_checkpoint,
        source_sha,
        source_subject,
        candidate_subject,
    )


def _finish_payload_from_frozen(
    *,
    root: Path,
    registry: Path,
    paths: Mapping[str, Path],
    code_paths: Mapping[str, Path],
    frozen_by_name: Mapping[str, Mapping[str, object]],
    frozen_artifacts: Sequence[Mapping[str, object]],
    source: Mapping[str, Any],
    pilot: Mapping[str, Any],
    source_checkpoint: Path,
    source_checkpoint_sha: str,
    source_subject: str,
    candidate_subject: str,
    torch: Any,
) -> dict[str, object]:
    fixed_snapshot = {
        name: (
            _require_hex(frozen.get("sha256"), f"fresh60 fixed {name} SHA-256"),
            int(frozen.get("size_bytes", -1)),
        )
        for name, frozen in frozen_by_name.items()
        if name in EXPECTED_RUN_ARTIFACT_PINS
    }
    _validate_fixed_run_pins(fixed_snapshot)

    full_sha = _require_hex(
        frozen_by_name["full_manifest"].get("sha256"),
        "frozen fresh60 full manifest SHA-256",
    )
    blind_binding = _verify_frozen_blind_manifest_contract(
        records_path=paths["blind_manifest"],
        blind_contract_path=paths["blind_contract"],
        frozen_artifacts=frozen_artifacts,
    )
    blind_binding = {
        **blind_binding,
        "recipient_val_records": _frozen_blind_recipient_val_records(
            blind_binding, frozen_artifacts=frozen_artifacts
        ),
    }
    _require_equal(
        blind_binding.get("source_manifest_sha256"),
        full_sha,
        "fresh60 blind source hash",
    )

    recipe = _strict_json_bytes(
        _frozen_bytes(
            frozen_by_name["training_recipe"], description="fresh60 training recipe"
        ),
        description="frozen fresh60 training recipe",
    )
    summary = _strict_json_bytes(
        _frozen_bytes(
            frozen_by_name["training_summary"],
            description="fresh60 training summary",
        ),
        description="frozen fresh60 training summary",
    )
    labels = _strict_json_bytes(
        _frozen_bytes(
            frozen_by_name["training_labels"], description="fresh60 training labels"
        ),
        description="frozen fresh60 training labels",
    )
    attempt_payload = _strict_json_bytes(
        _frozen_bytes(
            frozen_by_name["training_attempt"],
            description="fresh60 training attempt",
        ),
        description="frozen fresh60 training attempt",
    )
    blind_contract_payload = _strict_json_bytes(
        _frozen_bytes(
            frozen_by_name["blind_contract"],
            description="fresh60 blind contract",
        ),
        description="frozen fresh60 blind contract",
    )
    for payload, description in (
        (recipe, "fresh60 training recipe"),
        (summary, "fresh60 training summary"),
        (labels, "fresh60 training labels"),
        (attempt_payload, "fresh60 training attempt"),
        (blind_contract_payload, "fresh60 blind contract"),
    ):
        _assert_no_unsafe_true_claims(payload, location=description)

    observed, records = _validate_summary_recipe(
        summary,
        recipe=recipe,
        source_subject_id=source_subject,
        candidate_pilot_subject_id=candidate_subject,
        source_checkpoint=source_checkpoint,
        source_checkpoint_sha256=source_checkpoint_sha,
        full_manifest_sha256=full_sha,
    )
    best = _validate_checkpoint(
        _frozen_bytes(
            frozen_by_name["best_checkpoint"],
            description="fresh60 best checkpoint",
        ),
        torch=torch,
        summary=summary,
        record=records[EXPECTED_BEST_EPOCH - 1],
        expected_epoch=EXPECTED_BEST_EPOCH,
        description="fresh60 best checkpoint",
    )
    last = _validate_checkpoint(
        _frozen_bytes(
            frozen_by_name["last_checkpoint"],
            description="fresh60 last checkpoint",
        ),
        torch=torch,
        summary=summary,
        record=records[EXPECTED_LAST_EPOCH - 1],
        expected_epoch=EXPECTED_LAST_EPOCH,
        description="fresh60 last checkpoint",
    )
    _validate_labels(labels, summary=summary, best=best, last=last)
    _, attempt_id = _validate_attempt(
        paths["training_attempt"],
        payload=attempt_payload,
        registry=registry,
        candidate_root=root,
        source_subject_id=source_subject,
        candidate_pilot_subject_id=candidate_subject,
        full_manifest_sha256=full_sha,
    )
    _assert_failed_candidate_surface(
        root, frozen_artifacts=frozen_artifacts
    )

    code_before = {
        name: (_sha256(path), path.stat().st_size)
        for name, path in code_paths.items()
    }
    artifacts = {
        name: _binding_from_frozen(
            path,
            frozen_by_name[name],
            description=name,
        )
        for name, path in paths.items()
    }
    code = {
        name: _binding(path, description=name) for name, path in code_paths.items()
    }
    for name, binding in code.items():
        if (
            binding.get("sha256"),
            binding.get("size_bytes"),
        ) != code_before[name]:
            raise ValueError("fresh60 code changed while bindings were sealed")
    code_after = {
        name: (_sha256(path), path.stat().st_size)
        for name, path in code_paths.items()
    }
    if code_before != code_after:
        raise ValueError("fresh60 code changed during attestation")

    blind_semantic = {
        key: blind_binding.get(key)
        for key in (
            "schema_version",
            "kind",
            "source_manifest_sha256",
            "blind_manifest_sha256",
            "split_counts",
            "optimizer_supervision_splits",
            "checkpoint_selection_splits",
            "test_opened_by_training",
            "recipient_val_records",
        )
    }
    failure_subject_id = _canonical_sha256(
        _failure_subject_material(
            source_subject_id=source_subject,
            candidate_pilot_subject_id=candidate_subject,
            attempt_id=attempt_id,
            blind_semantic=blind_semantic,
            observed_failure=observed,
            recipe=recipe,
            summary=summary,
        )
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": KIND,
        "analysis_only": True,
        "new_view_pilot_authority": True,
        "decision": DECISION,
        "authorization": AUTHORIZATION,
        "production_route_authorized": False,
        "same_route_retry_authorized": False,
        "same_route_continuation_authorized": False,
        "warmstart_authorized": False,
        "failed_checkpoint_initialization_authorized": False,
        "onnx_export_authorized": False,
        "test_evaluation_authorized": False,
        "test_opened": False,
        "onnx_exported": False,
        "authorization_scope": {
            "epochs": 8,
            "training_data_view": "must_differ_from_failed_standard_full_crop_view",
            "source_initialization": "same_attested_legacy_source_fresh_visual_context_reinit",
            "failed_best_checkpoint_use": "forbidden",
            "failed_last_checkpoint_use": "forbidden",
            "optimizer_supervision_splits": ["train"],
            "checkpoint_selection_splits": ["val"],
            "final_gate_only_splits": ["test"],
            "production": "forbidden",
            "onnx": "forbidden",
            "test": "forbidden",
        },
        "failure_subject_id": failure_subject_id,
        "source_subject_id": source_subject,
        "candidate_pilot_subject_id": candidate_subject,
        "attempt_id": attempt_id,
        "candidate_root": str(root),
        "attempt_registry": str(registry),
        "attempt_consumed": "yes",
        "candidate_evidence": "absent",
        "onnx_artifacts": "absent",
        "test_evidence": "absent",
        "observed_failure": observed,
        "blind_manifest_contract": blind_semantic,
        "artifacts": artifacts,
        "code": code,
        "authority_chain": {
            "source_contract_kind": source.get("kind"),
            "source_contract_sha256": frozen_by_name["source_contract"].get(
                "sha256"
            ),
            "candidate_pilot_kind": pilot.get("kind"),
            "candidate_pilot_evidence_sha256": frozen_by_name[
                "candidate_pilot_evidence"
            ].get("sha256"),
        },
    }


def _build_payload(
    *,
    candidate_root: Path,
    source_contract_path: Path,
    candidate_pilot_evidence_path: Path,
    full_records: Path,
    attempt_registry: Path,
    torch: Any,
) -> dict[str, object]:
    root = _existing_non_reparse(
        candidate_root, directory=True, description="failed fresh60 candidate root"
    )
    registry = _existing_non_reparse(
        attempt_registry, directory=True, description="ProgramData training attempt registry"
    )
    _samefile(
        registry,
        _windows_programdata_attempt_registry(),
        "Windows ProgramData training attempt registry",
    )
    _assert_failed_candidate_surface(root, inspect_json=False)
    full = _existing_non_reparse(
        full_records, directory=False, description="fresh60 full manifest"
    )
    source_file = _existing_non_reparse(
        source_contract_path, directory=False, description="full-crop source contract"
    )
    pilot_file = _existing_non_reparse(
        candidate_pilot_evidence_path,
        directory=False,
        description="residual A8 evidence",
    )
    source_document, frozen_source_document = _validate_frozen_authority_document(
        source_file, name="source_contract"
    )
    pilot_document, frozen_pilot_document = _validate_frozen_authority_document(
        pilot_file, name="candidate_pilot_evidence"
    )
    _validate_candidate_pilot_source_cross_binding(
        pilot_document,
        frozen_source=frozen_source_document,
    )
    _require_equal(
        source_document.get("source_subject_id"),
        EXPECTED_SOURCE_SUBJECT_ID,
        "frozen source subject id",
    )
    _require_equal(
        pilot_document.get("source_subject_id"),
        EXPECTED_SOURCE_SUBJECT_ID,
        "frozen candidate-pilot source subject id",
    )
    _require_equal(
        pilot_document.get("candidate_pilot_subject_id"),
        EXPECTED_CANDIDATE_PILOT_SUBJECT_ID,
        "frozen candidate-pilot subject id",
    )
    frozen_authority_closure = _freeze_authority_closure(
        (
            ("source_contract", source_document),
            ("candidate_pilot_evidence", pilot_document),
        ),
        torch=torch,
    )
    deep_frozen_authority_closure = _unique_frozen_artifacts(
        (
            frozen_source_document,
            frozen_pilot_document,
            *frozen_authority_closure,
        )
    )
    try:
        (
            source,
            pilot,
            source_checkpoint,
            source_checkpoint_sha,
            source_subject,
            candidate_subject,
        ) = _validated_authorities(
            source_contract_path=source_file,
            source_document=source_document,
            candidate_pilot_evidence_path=pilot_file,
            full_records=full,
            frozen_artifacts=deep_frozen_authority_closure,
            torch=torch,
        )
    finally:
        _reverify_frozen_authority_closure(
            deep_frozen_authority_closure
        )

    paths = {
        name: _existing_non_reparse(path, directory=False, description=name)
        for name, path in _artifact_paths(root).items()
    }
    attempt_subject = f"{ATTEMPT_DOMAIN}|{source_subject}|{candidate_subject}"
    expected_attempt_id = hashlib.sha256(attempt_subject.encode("utf-8")).hexdigest()
    attempt_path = _existing_non_reparse(
        registry / f"{expected_attempt_id}.attempt.json",
        directory=False,
        description="fresh60 one-shot attempt",
    )
    paths.update(
        {
            "source_contract": source_file,
            "candidate_pilot_evidence": pilot_file,
            "full_manifest": full,
            "source_best_checkpoint": source_checkpoint,
            "training_attempt": attempt_path,
        }
    )
    code_paths = {
        name: _existing_non_reparse(path, directory=False, description=name)
        for name, path in _code_paths().items()
    }

    frozen_by_name: dict[str, Mapping[str, object]] = {}
    for name, suffixes in (
        ("source_contract", {".json"}),
        ("candidate_pilot_evidence", {".json"}),
        ("full_manifest", {".jsonl"}),
        ("source_best_checkpoint", {".pt"}),
    ):
        _, frozen = _frozen_artifact_for_path(
            paths[name],
            deep_frozen_authority_closure,
            suffixes=suffixes,
            description=f"fresh60 frozen {name.replace('_', ' ')}",
        )
        frozen_by_name[name] = frozen
    fixed_run = _freeze_fixed_run_artifacts(
        paths, existing_frozen=deep_frozen_authority_closure
    )
    frozen_by_name.update(fixed_run)
    closing_candidates = (
        *deep_frozen_authority_closure,
        *fixed_run.values(),
    )
    closing_closure: tuple[Mapping[str, object], ...] | None = None
    try:
        closing_closure = _unique_frozen_artifacts(closing_candidates)
        return _finish_payload_from_frozen(
            root=root,
            registry=registry,
            paths=paths,
            code_paths=code_paths,
            frozen_by_name=frozen_by_name,
            frozen_artifacts=closing_closure,
            source=source,
            pilot=pilot,
            source_checkpoint=source_checkpoint,
            source_checkpoint_sha=source_checkpoint_sha,
            source_subject=source_subject,
            candidate_subject=candidate_subject,
            torch=torch,
        )
    finally:
        _reverify_frozen_authority_closure(
            closing_candidates if closing_closure is None else closing_closure
        )


def attest_fresh60_failure(
    *,
    candidate_root: Path,
    source_contract_path: Path,
    candidate_pilot_evidence_path: Path,
    full_records: Path,
    attempt_registry: Path,
    output_evidence: Path,
    torch: Any | None = None,
) -> dict[str, object]:
    """Write one fresh, content-bound failure attestation."""

    if torch is None:
        torch, _ = _require_torch()
    output = _fresh_output(output_evidence)
    payload = _build_payload(
        candidate_root=candidate_root,
        source_contract_path=source_contract_path,
        candidate_pilot_evidence_path=candidate_pilot_evidence_path,
        full_records=full_records,
        attempt_registry=attempt_registry,
        torch=torch,
    )
    sealed = {**payload, "integrity_sha256": _canonical_sha256(payload)}
    encoded = json.dumps(
        sealed,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        indent=2,
    ) + "\n"
    try:
        with output.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
    except FileExistsError as error:
        raise ValueError(f"Refusing to overwrite failure attestation: {output}") from error
    return sealed


def verify_fresh60_failure(
    *,
    evidence_path: Path,
    source_contract_path: Path,
    candidate_pilot_evidence_path: Path,
    full_records: Path,
    attempt_registry: Path,
    torch: Any | None = None,
) -> dict[str, object]:
    """Rebuild the attestation from its bound root and require exact equality."""

    if torch is None:
        torch, _ = _require_torch()
    evidence_file = _existing_non_reparse(
        evidence_path, directory=False, description="fresh60 failure attestation"
    )
    evidence = _strict_json(evidence_file)
    claimed = evidence.get("integrity_sha256")
    unsigned = {key: value for key, value in evidence.items() if key != "integrity_sha256"}
    if not isinstance(claimed, str) or claimed != _canonical_sha256(unsigned):
        raise ValueError("fresh60 failure attestation integrity hash does not match")
    _require_equal(evidence.get("schema_version"), SCHEMA_VERSION, "attestation schema")
    _require_equal(evidence.get("kind"), KIND, "attestation kind")
    raw_root = evidence.get("candidate_root")
    if not isinstance(raw_root, str) or not raw_root:
        raise ValueError("fresh60 failure attestation has no candidate root")
    rebuilt = _build_payload(
        candidate_root=Path(raw_root),
        source_contract_path=source_contract_path,
        candidate_pilot_evidence_path=candidate_pilot_evidence_path,
        full_records=full_records,
        attempt_registry=attempt_registry,
        torch=torch,
    )
    _json_equivalent(unsigned, rebuilt, "fresh60 failure attestation")
    return evidence


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Seal or verify the failed recipient-v14 fresh60 run"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    attest = subparsers.add_parser("attest-failure")
    attest.add_argument("--candidate-root", type=Path, required=True)
    attest.add_argument("--source-contract", type=Path, required=True)
    attest.add_argument("--candidate-pilot-evidence", type=Path, required=True)
    attest.add_argument("--full-records", type=Path, required=True)
    attest.add_argument("--attempt-registry", type=Path, required=True)
    attest.add_argument("--output-evidence", type=Path, required=True)
    verify = subparsers.add_parser("verify-failure")
    verify.add_argument("--evidence", type=Path, required=True)
    verify.add_argument("--source-contract", type=Path, required=True)
    verify.add_argument("--candidate-pilot-evidence", type=Path, required=True)
    verify.add_argument("--full-records", type=Path, required=True)
    verify.add_argument("--attempt-registry", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "attest-failure":
        payload = attest_fresh60_failure(
            candidate_root=args.candidate_root,
            source_contract_path=args.source_contract,
            candidate_pilot_evidence_path=args.candidate_pilot_evidence,
            full_records=args.full_records,
            attempt_registry=args.attempt_registry,
            output_evidence=args.output_evidence,
        )
    else:
        payload = verify_fresh60_failure(
            evidence_path=args.evidence,
            source_contract_path=args.source_contract,
            candidate_pilot_evidence_path=args.candidate_pilot_evidence,
            full_records=args.full_records,
            attempt_registry=args.attempt_registry,
        )
    print(json.dumps(payload, ensure_ascii=False, allow_nan=False, sort_keys=True))


if __name__ == "__main__":
    main()
