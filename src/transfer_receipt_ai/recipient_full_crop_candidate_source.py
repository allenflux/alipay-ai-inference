"""Seal and reopen the analysis-only full-crop source for recipient v14.

The full-crop pilot is deliberately not a production checkpoint.  A passed
pilot with validation recipient accuracy still below the delivery floor may,
however, authorize one separate residual-recipient experiment.  This module
turns that narrow authorization into a content-bound contract.  Verification
always reruns the original pilot evaluator, reopens the blind-manifest
binding, checks the selected checkpoint against the training summary and
revalidates the sanitizer seed provenance.

The source contract authorizes neither ONNX export nor test evaluation.  The
PowerShell candidate runner keeps the bound files under read leases while a
fresh eight-epoch residual pilot (and, only after that pilot passes, a fresh
60-epoch candidate) runs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .ocr_unified import (
    CHECKPOINT_SELECTION_RECIPIENT_PRIORITY,
    INIT_CHECKPOINT_MODE_RECIPIENT_VISUAL_CONTEXT_REINIT,
    KIND_V13,
    STATUS_TEXT_RUNTIME_POLICY,
    UnifiedReaderConfig,
    _checkpoint_config,
    _load_checkpoint,
    _require_torch,
    _validate_recipient_full_crop_seed_policy,
    _validate_recipient_visual_context_reinit_config,
)
from .recipient_full_crop_pilot import (
    AMOUNT_FLOOR,
    PAYMENT_FLOOR,
    PILOT_EPOCHS,
    PILOT_MINIMUM_BEST_RECIPIENT,
    PILOT_MINIMUM_EPOCH4_TO_8_GAIN,
    STATUS_TEXT_FLOOR,
    TIME_FLOOR,
    evaluate_pilot_summary,
    verify_blind_manifest_contract,
)
from .recipient_full_crop_seed_sanitizer import _partition_descriptor


SCHEMA_VERSION = 1
SOURCE_KIND = "receipt_recipient_full_crop_candidate_source_v1"
CANDIDATE_PILOT_KIND = "receipt_recipient_v14_full_crop_residual_pilot_v1"
SOURCE_DECISION = "analysis_only_continue_to_separate_guarded_candidate"
CANDIDATE_PILOT_DECISION = "analysis_only_continue_to_fresh_60_epoch_candidate"
SOURCE_SUBJECT_DOMAIN = "receipt-recipient-full-crop-source-subject-v1"
CANDIDATE_PILOT_SUBJECT_DOMAIN = "receipt-recipient-v14-residual-pilot-subject-v1"
TRAINING_RECIPE_KIND = "receipt_recipient_v14_full_crop_training_recipe_v1"
RECIPIENT_DELIVERY_FLOOR = 0.90
EXPECTED_RECIPIENT_VAL_RECORDS = 6789
REQUIRED_BACKBONE = "residual_positional_transformer_v2"
REQUIRED_SOURCE_BACKBONE = "legacy_depthwise_gru_v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(payload: Mapping[str, object]) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _strict_json(path: Path) -> dict[str, Any]:
    def reject_constant(value: str) -> None:
        raise ValueError(f"{path}: non-finite JSON constant {value!r}")

    try:
        payload = json.loads(
            path.read_text(encoding="utf-8-sig"),
            parse_constant=reject_constant,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"Unable to read strict JSON object {path}: {error}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected a JSON object")

    def reject_nonfinite(value: object, location: str) -> None:
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError(f"{path}: non-finite JSON number at {location}")
        if isinstance(value, Mapping):
            for key, child in value.items():
                reject_nonfinite(child, f"{location}.{key}")
        elif isinstance(value, list):
            for index, child in enumerate(value):
                reject_nonfinite(child, f"{location}[{index}]")

    reject_nonfinite(payload, "$")
    return payload


def _mapping(value: object, description: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{description} must be an object")
    return value


def _finite_rate(value: object, description: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{description} must be a finite rate")
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{description} must be a finite rate") from error
    if not math.isfinite(number) or not 0.0 <= number <= 1.0:
        raise ValueError(f"{description} must be between zero and one")
    return number


def _require_equal(actual: object, expected: object, description: str) -> None:
    if type(actual) is not type(expected) or actual != expected:
        raise ValueError(
            f"{description} mismatch: expected {expected!r}, found {actual!r}"
        )


def _is_reparse_path(path: Path) -> bool:
    if path.is_symlink():
        return True
    try:
        attributes = int(getattr(path.lstat(), "st_file_attributes", 0))
    except OSError:
        return False
    return bool(attributes & 0x400)


def _existing_non_reparse(path: Path, *, directory: bool, description: str) -> Path:
    """Resolve an existing path only after rejecting every reparse boundary."""

    raw = Path(os.path.abspath(os.fspath(Path(path).expanduser())))
    if not os.path.lexists(os.fspath(raw)):
        raise FileNotFoundError(f"Missing {description}: {raw}")
    current = raw
    while True:
        if os.path.lexists(os.fspath(current)) and _is_reparse_path(current):
            raise ValueError(f"{description} must not traverse a symlink/junction/reparse path")
        if current == current.parent:
            break
        current = current.parent
    resolved = raw.resolve(strict=True)
    if directory and not resolved.is_dir():
        raise ValueError(f"{description} is not a directory: {resolved}")
    if not directory and not resolved.is_file():
        raise ValueError(f"{description} is not a file: {resolved}")
    return resolved


def _fresh_contract_path(path: Path) -> Path:
    raw = Path(os.path.abspath(os.fspath(Path(path).expanduser())))
    if os.path.lexists(os.fspath(raw)):
        raise ValueError(f"Refusing to overwrite source contract: {raw}")
    parent = _existing_non_reparse(
        raw.parent,
        directory=True,
        description="source contract parent",
    )
    return parent / raw.name


def _samefile(left: Path, right: Path, description: str) -> None:
    try:
        same = os.path.samefile(left, right)
    except OSError as error:
        raise ValueError(f"Unable to verify {description} identity") from error
    if not same:
        raise ValueError(f"{description} is not the bound file")


def _binding(path: Path) -> dict[str, object]:
    resolved = _existing_non_reparse(path, directory=False, description="bound artifact")
    return {
        "path": str(resolved),
        "sha256": _sha256(resolved),
        "size_bytes": resolved.stat().st_size,
    }


def _checkpoint_state_identity(state: Mapping[str, object]) -> dict[str, object]:
    """Describe tensor values without depending on torch serialization bytes."""

    return {
        "non_recipient": _partition_descriptor(state, recipient=False),
        "recipient": _partition_descriptor(state, recipient=True),
    }


def _code_content_identity(
    artifacts: Mapping[str, Mapping[str, object]],
) -> dict[str, dict[str, object]]:
    identity: dict[str, dict[str, object]] = {}
    for name in sorted(artifacts):
        if not (name.startswith("code_") or name.startswith("script_")):
            continue
        binding = _mapping(artifacts[name], f"{name} binding")
        sha256 = binding.get("sha256")
        size = binding.get("size_bytes")
        if not isinstance(sha256, str) or len(sha256) != 64:
            raise ValueError(f"{name} binding has no valid SHA-256")
        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            raise ValueError(f"{name} binding has no valid size")
        identity[name] = {"sha256": sha256, "size_bytes": size}
    if not identity:
        raise ValueError("subject identity has no bound verifier/runner code")
    return identity


def _blind_semantic_identity(binding: Mapping[str, Any]) -> dict[str, object]:
    return {
        "schema_version": binding.get("schema_version"),
        "kind": binding.get("kind"),
        "source_manifest_sha256": binding.get("source_manifest_sha256"),
        "blind_manifest_sha256": binding.get("blind_manifest_sha256"),
        "split_counts": binding.get("split_counts"),
        "recipient_val_records": binding.get("recipient_val_records"),
        "optimizer_supervision_splits": binding.get("optimizer_supervision_splits"),
        "checkpoint_selection_splits": binding.get("checkpoint_selection_splits"),
        "test_opened_by_training": binding.get("test_opened_by_training"),
    }


def _blind_recipient_val_records(binding: Mapping[str, Any]) -> int:
    """Recount the recipient validation denominator from frozen manifest bytes."""

    raw_path = binding.get("blind_manifest")
    if not isinstance(raw_path, str) or not raw_path:
        raise ValueError("blind-manifest binding has no manifest path")
    path = _existing_non_reparse(
        Path(raw_path), directory=False, description="blind recipient manifest"
    )
    records = 0
    with path.open("r", encoding="utf-8-sig") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"blind recipient manifest line {line_number} is invalid JSON"
                ) from error
            if not isinstance(row, Mapping):
                raise ValueError(
                    f"blind recipient manifest line {line_number} is not an object"
                )
            slots = row.get("slots")
            if not isinstance(slots, Mapping):
                raise ValueError(
                    f"blind recipient manifest line {line_number} has invalid slots"
                )
            recipient = slots.get("recipient_field")
            if recipient is not None and not isinstance(recipient, Mapping):
                raise ValueError(
                    f"blind recipient manifest line {line_number} has an invalid recipient slot"
                )
            if isinstance(recipient, Mapping):
                text = recipient.get("text")
                if (
                    not isinstance(text, str)
                    or not text
                    or any(not character.isprintable() for character in text)
                ):
                    raise ValueError(
                        f"blind recipient manifest line {line_number} has no valid recipient target"
                    )
            if row.get("split") == "val" and recipient is not None:
                records += 1
    if records != EXPECTED_RECIPIENT_VAL_RECORDS:
        raise ValueError(
            "blind manifest recipient val denominator mismatch: "
            f"expected {EXPECTED_RECIPIENT_VAL_RECORDS}, found {records}"
        )
    return records


def _exact_count_metric(
    metric: Mapping[str, Any], *, expected_records: int, description: str
) -> tuple[int, float]:
    records = metric.get("records")
    matches = metric.get("exact_matches")
    if (
        isinstance(records, bool)
        or not isinstance(records, int)
        or records != expected_records
    ):
        raise ValueError(
            f"{description} records must equal the frozen recipient val denominator "
            f"{expected_records}"
        )
    if (
        isinstance(matches, bool)
        or not isinstance(matches, int)
        or not 0 <= matches <= records
    ):
        raise ValueError(f"{description} exact_matches must be an integer in [0, records]")
    exact = _finite_rate(metric.get("exact_match"), f"{description} exact_match")
    expected_exact = matches / records
    if not math.isclose(exact, expected_exact, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(
            f"{description} exact_match is inconsistent with exact_matches/records"
        )
    return matches, exact


def validate_full_crop_candidate_training_metrics(
    summary: Mapping[str, Any],
) -> dict[str, object]:
    """Replay recipient coverage/count semantics for the fixed fresh 60e run."""

    field_counts = _mapping(summary.get("field_counts"), "candidate field counts")
    recipient_counts = _mapping(
        field_counts.get("recipient_field"), "candidate recipient field counts"
    )
    _require_equal(
        recipient_counts.get("val"),
        EXPECTED_RECIPIENT_VAL_RECORDS,
        "candidate recipient val count",
    )
    _require_equal(recipient_counts.get("test"), 0, "candidate recipient test count")
    recipient_oov = _mapping(
        summary.get("recipient_oov_by_split"), "candidate recipient OOV"
    )
    recipient_val_oov = _mapping(recipient_oov.get("val"), "candidate val OOV")
    recipient_test_oov = _mapping(recipient_oov.get("test"), "candidate test OOV")
    _require_equal(
        recipient_val_oov.get("records"),
        EXPECTED_RECIPIENT_VAL_RECORDS,
        "candidate val OOV records",
    )
    _require_equal(
        recipient_test_oov.get("records"), 0, "candidate test OOV records"
    )

    raw_records = summary.get("records")
    if not isinstance(raw_records, Sequence) or isinstance(raw_records, (str, bytes)):
        raise ValueError("candidate 60e summary has invalid epoch records")
    records = [_mapping(record, "candidate 60e epoch record") for record in raw_records]
    if [record.get("epoch") for record in records] != list(range(1, 61)):
        raise ValueError("candidate 60e summary requires ordered epochs 1 through 60")

    validated_epochs: list[int] = []
    recipient_by_epoch: dict[int, tuple[int, float]] = {}
    for record in records:
        epoch = int(record["epoch"])
        expected_validation = epoch == 1 or epoch == 60 or epoch % 2 == 0
        _require_equal(
            record.get("validation_performed"),
            expected_validation,
            f"candidate epoch {epoch} validation schedule",
        )
        fields = record.get("val_candidate_text_by_field")
        if not expected_validation:
            _require_equal(fields, None, f"candidate epoch {epoch} skipped validation metrics")
            continue
        validated_epochs.append(epoch)
        field_metrics = _mapping(fields, f"candidate epoch {epoch} fields")
        recipient_metric = _mapping(
            field_metrics.get("recipient_field"),
            f"candidate epoch {epoch} recipient metric",
        )
        recipient_by_epoch[epoch] = _exact_count_metric(
            recipient_metric,
            expected_records=EXPECTED_RECIPIENT_VAL_RECORDS,
            description=f"candidate epoch {epoch} recipient metric",
        )

    best_epoch = summary.get("best_checkpoint_epoch")
    if (
        isinstance(best_epoch, bool)
        or not isinstance(best_epoch, int)
        or best_epoch not in recipient_by_epoch
    ):
        raise ValueError("candidate best checkpoint epoch has no complete validation metric")
    maximum = max(exact for _, exact in recipient_by_epoch.values())
    best_matches, best_exact = recipient_by_epoch[best_epoch]
    if not math.isclose(best_exact, maximum, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("candidate best checkpoint is not recipient-optimal")
    if best_exact <= RECIPIENT_DELIVERY_FLOOR:
        raise ValueError("candidate best recipient exact must be strictly above 90%")
    return {
        "recipient_val_records": EXPECTED_RECIPIENT_VAL_RECORDS,
        "recipient_candidate_coverage": 1.0,
        "validated_epochs": validated_epochs,
        "best_epoch": best_epoch,
        "best_recipient_exact_matches": best_matches,
        "best_recipient_exact": best_exact,
    }


def validate_full_crop_candidate_val_metrics(
    summary: Mapping[str, Any],
) -> dict[str, object]:
    """Validate the independently exported ONNX val recipient count evidence."""

    _require_equal(summary.get("evaluation_split"), "val", "candidate evaluation split")
    by_field = _mapping(summary.get("by_field"), "candidate evaluation fields")
    recipient = _mapping(
        by_field.get("recipient_field"), "candidate evaluation recipient metric"
    )
    metric = {
        "records": recipient.get("records"),
        "exact_matches": recipient.get("raw_exact_matches"),
        "exact_match": recipient.get("raw_exact_match"),
    }
    matches, exact = _exact_count_metric(
        metric,
        expected_records=EXPECTED_RECIPIENT_VAL_RECORDS,
        description="candidate ONNX val recipient metric",
    )
    if exact <= RECIPIENT_DELIVERY_FLOOR:
        raise ValueError("candidate ONNX val recipient exact must be strictly above 90%")
    return {
        "recipient_records": EXPECTED_RECIPIENT_VAL_RECORDS,
        "recipient_exact_matches": matches,
        "recipient_exact_match": exact,
        "recipient_candidate_coverage": 1.0,
    }


def _validate_source_pilot_recipient_metrics(
    summary: Mapping[str, Any], *, expected_records: int
) -> dict[str, object]:
    raw_records = summary.get("records")
    if not isinstance(raw_records, Sequence) or isinstance(raw_records, (str, bytes)):
        raise ValueError("source pilot has invalid epoch records")
    validated_epochs: list[int] = []
    for raw_record in raw_records:
        record = _mapping(raw_record, "source pilot epoch record")
        epoch = record.get("epoch")
        if isinstance(epoch, bool) or not isinstance(epoch, int):
            raise ValueError("source pilot epoch must be an integer")
        fields = _mapping(
            record.get("val_candidate_text_by_field"),
            f"source pilot epoch {epoch} fields",
        )
        recipient = _mapping(
            fields.get("recipient_field"),
            f"source pilot epoch {epoch} recipient metric",
        )
        _exact_count_metric(
            recipient,
            expected_records=expected_records,
            description=f"source pilot epoch {epoch} recipient metric",
        )
        validated_epochs.append(epoch)
    return {
        "recipient_val_records": expected_records,
        "recipient_candidate_coverage": 1.0,
        "validated_epochs": validated_epochs,
    }


def _binding_path(
    artifacts: Mapping[str, Any], name: str, *, description: str | None = None
) -> Path:
    binding = _mapping(artifacts.get(name), f"{name} binding")
    raw_path = binding.get("path")
    claimed_sha256 = binding.get("sha256")
    claimed_size = binding.get("size_bytes")
    if not isinstance(raw_path, str) or not raw_path:
        raise ValueError(f"{name} binding has no path")
    if (
        not isinstance(claimed_sha256, str)
        or len(claimed_sha256) != 64
        or any(character not in "0123456789abcdef" for character in claimed_sha256)
    ):
        raise ValueError(f"{name} binding has an invalid SHA-256")
    if isinstance(claimed_size, bool) or not isinstance(claimed_size, int) or claimed_size < 0:
        raise ValueError(f"{name} binding has an invalid size")
    path = _existing_non_reparse(
        Path(raw_path),
        directory=False,
        description=description or name,
    )
    if path.stat().st_size != claimed_size or _sha256(path) != claimed_sha256:
        raise ValueError(f"{name} changed after contract creation")
    return path


def _code_paths() -> dict[str, Path]:
    package = Path(__file__).resolve().parent
    repository = package.parents[1]
    return {
        "code_candidate_source_attestor": Path(__file__).resolve(),
        "code_full_crop_pilot": package / "recipient_full_crop_pilot.py",
        "code_ocr_unified": package / "ocr_unified.py",
        "code_blind_manifest": package / "recipient_blind_manifest.py",
        "code_seed_sanitizer": package / "recipient_full_crop_seed_sanitizer.py",
        "script_candidate_source_attestor": (
            repository / "scripts" / "receipt-ocr-recipient-full-crop-candidate-source.py"
        ),
        "script_full_crop_pilot": (
            repository / "scripts" / "receipt-ocr-recipient-full-crop-pilot-4090.ps1"
        ),
        "script_v14_candidate": (
            repository / "scripts" / "receipt-ocr-recipient-v14-candidate-4090.ps1"
        ),
    }


def _json_equivalent(actual: object, expected: object, description: str) -> None:
    try:
        actual_hash = _canonical_sha256({"value": actual})
        expected_hash = _canonical_sha256({"value": expected})
    except (TypeError, ValueError) as error:
        raise ValueError(f"{description} is not strict JSON-compatible") from error
    if actual_hash != expected_hash:
        raise ValueError(f"{description} does not match its authoritative source")


def _validate_blind_binding_equivalent(
    stored: Mapping[str, Any], current: Mapping[str, Any]
) -> None:
    """Compare a blind binding while allowing only same-file path aliases."""

    path_fields = ("contract_path", "source_manifest", "blind_manifest")
    stored_semantic = dict(stored)
    current_semantic = dict(current)
    for field in path_fields:
        stored_raw = stored_semantic.pop(field, None)
        current_raw = current_semantic.pop(field, None)
        if not isinstance(stored_raw, str) or not stored_raw:
            raise ValueError(f"stored blind-manifest binding has no {field}")
        if not isinstance(current_raw, str) or not current_raw:
            raise ValueError(f"recomputed blind-manifest binding has no {field}")
        stored_path = _existing_non_reparse(
            Path(stored_raw), directory=False, description=f"stored blind binding {field}"
        )
        current_path = _existing_non_reparse(
            Path(current_raw), directory=False, description=f"current blind binding {field}"
        )
        _samefile(stored_path, current_path, f"blind binding {field}")
    _json_equivalent(
        stored_semantic,
        current_semantic,
        "stored full-crop blind-manifest semantic binding",
    )


def _recipient_exact(record: Mapping[str, Any], description: str) -> float:
    by_field = _mapping(record.get("val_candidate_text_by_field"), f"{description} fields")
    recipient = _mapping(by_field.get("recipient_field"), f"{description} recipient")
    return _finite_rate(recipient.get("exact_match"), f"{description} recipient exact")


def _assert_no_onnx(root: Path) -> None:
    unsafe_true_claims = {
        "test_evaluated",
        "test_labels_used",
        "test_metrics_computed",
        "test_examples_emitted",
        "test_opened",
        "test_opened_by_training",
        "external_test_artifacts_opened",
        "production_route_authorized",
    }

    def inspect_claims(value: object, location: str) -> None:
        if isinstance(value, Mapping):
            if value.get("evaluation_split") == "test":
                raise ValueError(f"analysis root contains test evaluation evidence at {location}")
            for key, child in value.items():
                if key in unsafe_true_claims and child is True:
                    raise ValueError(f"analysis root contains an unsafe {key} claim at {location}")
                inspect_claims(child, f"{location}.{key}")
        elif isinstance(value, list):
            for index, child in enumerate(value):
                inspect_claims(child, f"{location}[{index}]")

    for path in root.rglob("*"):
        if _is_reparse_path(path):
            raise ValueError("full-crop pilot root contains a symlink/junction/reparse entry")
        if path.is_file() and path.suffix.lower() == ".onnx":
            raise ValueError("full-crop pilot root contains an ONNX artifact")
        if path.is_file() and path.suffix.lower() == ".json":
            inspect_claims(_strict_json(path), str(path))


def _pilot_paths(pilot_root: Path) -> dict[str, Path]:
    training = pilot_root / "training-full-crop-pilot"
    blind = pilot_root / "blind-train-val"
    return {
        "best_checkpoint": training / "best.pt",
        "training_summary": training / "training_summary.json",
        "pilot_decision": training / "pilot_decision.json",
        "blind_manifest": blind / "unified_fields.train-val.jsonl",
        "blind_contract": blind / "blind.contract.json",
    }


def _load_and_validate_pilot(
    pilot_root: Path,
    *,
    torch: Any,
) -> tuple[dict[str, object], dict[str, Path], dict[str, object]]:
    root = _existing_non_reparse(
        pilot_root,
        directory=True,
        description="full-crop pilot root",
    )
    _assert_no_onnx(root)
    paths = {
        name: _existing_non_reparse(path, directory=False, description=name)
        for name, path in _pilot_paths(root).items()
    }
    before = {name: _sha256(path) for name, path in paths.items()}

    summary = _strict_json(paths["training_summary"])
    decision = _strict_json(paths["pilot_decision"])
    blind_binding = verify_blind_manifest_contract(
        records_path=paths["blind_manifest"],
        blind_contract_path=paths["blind_contract"],
    )
    full_manifest = _existing_non_reparse(
        Path(str(blind_binding["source_manifest"])),
        directory=False,
        description="pilot full manifest",
    )
    full_manifest_sha256 = _sha256(full_manifest)
    if full_manifest_sha256 != blind_binding.get("source_manifest_sha256"):
        raise ValueError("pilot full manifest changed after blind-contract creation")
    paths["full_manifest"] = full_manifest
    before["full_manifest"] = full_manifest_sha256

    recomputed = evaluate_pilot_summary(summary)
    recomputed = {**recomputed, "blind_manifest_contract": blind_binding}
    _require_equal(
        summary.get("status_text_runtime_policy"),
        STATUS_TEXT_RUNTIME_POLICY,
        "full-crop status-text runtime policy",
    )
    stored_decision = dict(decision)
    recomputed_decision = dict(recomputed)
    stored_blind = _mapping(
        stored_decision.pop("blind_manifest_contract", None),
        "stored pilot blind-manifest binding",
    )
    recomputed_blind = _mapping(
        recomputed_decision.pop("blind_manifest_contract", None),
        "recomputed pilot blind-manifest binding",
    )
    _json_equivalent(
        stored_decision,
        recomputed_decision,
        "stored full-crop pilot decision",
    )
    _validate_blind_binding_equivalent(stored_blind, recomputed_blind)
    blind_binding = {
        **blind_binding,
        "recipient_val_records": _blind_recipient_val_records(blind_binding),
    }
    recipient_metric_audit = _validate_source_pilot_recipient_metrics(
        summary,
        expected_records=int(blind_binding["recipient_val_records"]),
    )
    recomputed = {
        **recomputed,
        "blind_manifest_contract": blind_binding,
        "recipient_candidate_metric_audit": recipient_metric_audit,
    }
    _require_equal(decision.get("passed"), True, "pilot passed")
    _require_equal(decision.get("analysis_only"), True, "pilot analysis_only")
    _require_equal(
        decision.get("production_route_authorized"),
        False,
        "pilot production_route_authorized",
    )
    _require_equal(decision.get("decision"), SOURCE_DECISION, "pilot decision")
    decision_blind = _mapping(
        decision.get("blind_manifest_contract"), "pilot blind-manifest binding"
    )
    _require_equal(
        decision_blind.get("test_opened_by_training"),
        False,
        "pilot test_opened_by_training",
    )

    observed = _mapping(decision.get("observed"), "pilot observations")
    best_recipient = _finite_rate(
        observed.get("best_recipient_exact"), "pilot best recipient exact"
    )
    if best_recipient < PILOT_MINIMUM_BEST_RECIPIENT:
        raise ValueError("full-crop source did not pass its fixed recipient pilot floor")
    if best_recipient > RECIPIENT_DELIVERY_FLOOR:
        raise ValueError(
            "full-crop source already exceeded the 90% recipient delivery floor; "
            "the separate residual candidate route is not authorized"
        )

    best_payload = _load_checkpoint(paths["best_checkpoint"], torch=torch)
    if not isinstance(best_payload, Mapping):
        raise ValueError("full-crop best checkpoint payload must be an object")
    if best_payload.get("kind") != KIND_V13:
        raise ValueError("full-crop best checkpoint is not a v13 reader")
    best_state = best_payload.get("state_dict")
    if not isinstance(best_state, Mapping) or not best_state:
        raise ValueError("full-crop best checkpoint has no model state")
    best_state_identity = _checkpoint_state_identity(best_state)
    try:
        best_config = _checkpoint_config(best_payload)
        best_config.validate()
    except (TypeError, ValueError) as error:
        raise ValueError("full-crop best checkpoint has an invalid model config") from error
    _json_equivalent(
        asdict(best_config), summary.get("config"), "best checkpoint validated config"
    )
    best_epoch = summary.get("best_checkpoint_epoch")
    if isinstance(best_epoch, bool) or not isinstance(best_epoch, int):
        raise ValueError("full-crop training summary has an invalid best epoch")
    _require_equal(best_payload.get("epoch"), best_epoch, "best checkpoint epoch")
    records = summary.get("records")
    if not isinstance(records, Sequence) or isinstance(records, (str, bytes)):
        raise ValueError("full-crop training summary has invalid records")
    best_rows = [
        _mapping(record, "full-crop epoch record")
        for record in records
        if isinstance(record, Mapping) and record.get("epoch") == best_epoch
    ]
    if len(best_rows) != 1:
        raise ValueError("full-crop summary does not contain exactly one best epoch record")
    _json_equivalent(best_payload.get("metrics"), best_rows[0], "best checkpoint metrics")
    for key in (
        "config",
        "initialization",
        "fine_tune_policy",
        "checkpoint_selection_policy",
        "recipient_train_split_policy",
        "field_counts",
        "status_text_runtime_policy",
        "training_runtime",
    ):
        _json_equivalent(best_payload.get(key), summary.get(key), f"best checkpoint {key}")

    initialization = _mapping(summary.get("initialization"), "pilot initialization")
    raw_seed_path = initialization.get("checkpoint_path")
    claimed_seed_sha = initialization.get("checkpoint_sha256")
    if not isinstance(raw_seed_path, str) or not raw_seed_path:
        raise ValueError("full-crop summary does not bind its sanitizer seed path")
    if not isinstance(claimed_seed_sha, str) or len(claimed_seed_sha) != 64:
        raise ValueError("full-crop summary does not bind its sanitizer seed SHA-256")
    seed = _existing_non_reparse(
        Path(raw_seed_path), directory=False, description="full-crop sanitizer seed"
    )
    if _sha256(seed) != claimed_seed_sha.lower():
        raise ValueError("full-crop sanitizer seed changed after pilot training")
    seed_payload = _load_checkpoint(seed, torch=torch)
    _validate_recipient_full_crop_seed_policy(seed_payload, torch=torch)
    source_config = _checkpoint_config(seed_payload)
    _json_equivalent(
        asdict(source_config),
        initialization.get("source_config"),
        "pilot seed source config",
    )
    paths["seed_checkpoint"] = seed
    before["seed_checkpoint"] = _sha256(seed)

    after = {name: _sha256(path) for name, path in paths.items()}
    if before != after:
        raise ValueError("full-crop source artifacts changed during attestation")
    subject_material = {
        "pilot_decision": {
            "schema_version": recomputed.get("schema_version"),
            "kind": recomputed.get("kind"),
            "analysis_only": recomputed.get("analysis_only"),
            "production_route_authorized": recomputed.get(
                "production_route_authorized"
            ),
            "epochs": recomputed.get("epochs"),
            "source_config": recomputed.get("source_config"),
            "target_config": recomputed.get("target_config"),
            "fixed_gates": recomputed.get("fixed_gates"),
            "observed": recomputed.get("observed"),
            "passed": recomputed.get("passed"),
            "failures": recomputed.get("failures"),
            "decision": recomputed.get("decision"),
            "recipient_candidate_metric_audit": recomputed.get(
                "recipient_candidate_metric_audit"
            ),
            "blind_manifest_contract": _blind_semantic_identity(blind_binding),
        },
        "best_checkpoint": {
            "kind": best_payload.get("kind"),
            "epoch": best_epoch,
            "config": asdict(best_config),
            "state": best_state_identity,
        },
        "blind_manifest": _blind_semantic_identity(blind_binding),
    }
    return recomputed, paths, subject_material


def _source_contract_payload(pilot_root: Path, *, torch: Any) -> dict[str, object]:
    root = _existing_non_reparse(
        pilot_root, directory=True, description="full-crop pilot root"
    )
    decision, paths, subject_material = _load_and_validate_pilot(root, torch=torch)
    artifacts = {name: _binding(path) for name, path in paths.items()}
    for name, path in _code_paths().items():
        artifacts[name] = _binding(path)
    observed = _mapping(decision.get("observed"), "pilot observations")
    fixed_source_gate = {
        "minimum_best_recipient_exact": PILOT_MINIMUM_BEST_RECIPIENT,
        "maximum_best_recipient_exact_inclusive": RECIPIENT_DELIVERY_FLOOR,
        "minimum_epoch4_to_8_gain": PILOT_MINIMUM_EPOCH4_TO_8_GAIN,
    }
    source_subject_id = _canonical_sha256(
        {
            "domain": SOURCE_SUBJECT_DOMAIN,
            "kind": SOURCE_KIND,
            "authorization": "separate_guarded_recipient_v14_candidate_only",
            "fixed_source_gate": fixed_source_gate,
            "decision": subject_material["pilot_decision"],
            "best_checkpoint": subject_material["best_checkpoint"],
            "blind_manifest": subject_material["blind_manifest"],
            "code": _code_content_identity(artifacts),
        }
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": SOURCE_KIND,
        "analysis_only": True,
        "production_route_authorized": False,
        "test_opened": False,
        "onnx_exported": False,
        "authorization": "separate_guarded_recipient_v14_candidate_only",
        "source_subject_id": source_subject_id,
        "pilot_root": str(root),
        "fixed_source_gate": fixed_source_gate,
        "observed": dict(observed),
        "artifacts": artifacts,
        "recomputed_pilot_decision": decision,
    }


def attest_full_crop_candidate_source(
    *, pilot_root: Path, output_contract: Path, torch: Any | None = None
) -> dict[str, object]:
    """Write one fresh, content-bound candidate-source contract."""

    if torch is None:
        torch, _ = _require_torch()
    output = _fresh_contract_path(output_contract)
    payload = _source_contract_payload(pilot_root, torch=torch)
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
        raise ValueError(f"Refusing to overwrite source contract: {output}") from error
    return sealed


def verify_full_crop_candidate_source(
    *,
    pilot_root: Path,
    contract_path: Path,
    full_records: Path | None = None,
    torch: Any | None = None,
) -> dict[str, object]:
    """Rebuild a source contract and require exact content equality."""

    if torch is None:
        torch, _ = _require_torch()
    root = _existing_non_reparse(
        pilot_root, directory=True, description="full-crop pilot root"
    )
    contract_file = _existing_non_reparse(
        contract_path, directory=False, description="full-crop source contract"
    )
    contract = _strict_json(contract_file)
    claimed_integrity = contract.get("integrity_sha256")
    unsigned = {key: value for key, value in contract.items() if key != "integrity_sha256"}
    if (
        not isinstance(claimed_integrity, str)
        or claimed_integrity != _canonical_sha256(unsigned)
    ):
        raise ValueError("full-crop source contract integrity hash does not match")
    if contract.get("schema_version") != SCHEMA_VERSION or contract.get("kind") != SOURCE_KIND:
        raise ValueError("full-crop source contract kind/schema is unsupported")
    raw_bound_root = contract.get("pilot_root")
    if not isinstance(raw_bound_root, str) or not raw_bound_root:
        raise ValueError("full-crop source contract has no pilot root")
    bound_root = _existing_non_reparse(
        Path(raw_bound_root), directory=True, description="bound full-crop pilot root"
    )
    _samefile(root, bound_root, "full-crop pilot root")

    rebuilt = _source_contract_payload(root, torch=torch)
    _json_equivalent(unsigned, rebuilt, "full-crop source contract")
    artifacts = _mapping(contract.get("artifacts"), "source contract artifacts")
    for name in artifacts:
        _binding_path(artifacts, str(name))
    if full_records is not None:
        supplied = _existing_non_reparse(
            full_records, directory=False, description="candidate full manifest"
        )
        bound = _binding_path(
            artifacts, "full_manifest", description="source-contract full manifest"
        )
        _samefile(supplied, bound, "candidate full manifest")
    return contract


def validate_full_crop_training_recipe(
    recipe: Mapping[str, Any],
    *,
    stage: str,
    source_subject_id: str,
    candidate_pilot_subject_id: str | None,
    source_checkpoint_sha256: str,
    full_manifest_sha256: str,
) -> dict[str, object]:
    """Validate the path-free runner recipe written before a training attempt."""

    if stage not in {"residual-8e", "candidate-60e"}:
        raise ValueError("full-crop training recipe stage is unsupported")
    for value, description in (
        (source_subject_id, "source subject id"),
        (source_checkpoint_sha256, "source checkpoint SHA-256"),
        (full_manifest_sha256, "full manifest SHA-256"),
    ):
        if (
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
        ):
            raise ValueError(f"full-crop training recipe has an invalid {description}")
    if candidate_pilot_subject_id is not None and (
        not isinstance(candidate_pilot_subject_id, str)
        or len(candidate_pilot_subject_id) != 64
        or any(
            character not in "0123456789abcdef"
            for character in candidate_pilot_subject_id
        )
    ):
        raise ValueError("full-crop training recipe has an invalid candidate-pilot subject id")

    expected_candidate_subject = (
        None if stage == "residual-8e" else candidate_pilot_subject_id
    )
    expected_runtime = {
        "device": "cuda:0",
        "epochs": 8 if stage == "residual-8e" else 60,
        "batch_size": 10,
        "learning_rate": 0.0003,
        "validation_every": 1 if stage == "residual-8e" else 2,
        "seed": 42,
        "num_workers": 4,
        "prefetch_factor": 2,
        "persistent_workers": True,
        "cuda_tf32": True,
        "cudnn_benchmark": True,
    }
    expected_keys = {
        "schema_version",
        "kind",
        "analysis_only",
        "production_route_authorized",
        "test_opened",
        "stage",
        "source_subject_id",
        "candidate_pilot_subject_id",
        "source_checkpoint_sha256",
        "full_manifest_sha256",
        "training_args",
    }
    if set(recipe) != expected_keys:
        raise ValueError("full-crop training recipe keys changed")
    _require_equal(recipe.get("schema_version"), SCHEMA_VERSION, "training recipe schema")
    _require_equal(recipe.get("kind"), TRAINING_RECIPE_KIND, "training recipe kind")
    _require_equal(recipe.get("analysis_only"), True, "training recipe analysis_only")
    _require_equal(
        recipe.get("production_route_authorized"),
        False,
        "training recipe production authorization",
    )
    _require_equal(recipe.get("test_opened"), False, "training recipe test_opened")
    _require_equal(recipe.get("stage"), stage, "training recipe stage")
    _require_equal(
        recipe.get("source_subject_id"), source_subject_id, "training recipe source subject"
    )
    _require_equal(
        recipe.get("candidate_pilot_subject_id"),
        expected_candidate_subject,
        "training recipe candidate-pilot subject",
    )
    _require_equal(
        recipe.get("source_checkpoint_sha256"),
        source_checkpoint_sha256,
        "training recipe source checkpoint",
    )
    _require_equal(
        recipe.get("full_manifest_sha256"),
        full_manifest_sha256,
        "training recipe full manifest",
    )
    _json_equivalent(
        _mapping(recipe.get("training_args"), "training recipe arguments"),
        expected_runtime,
        "fixed full-crop training recipe",
    )
    return {
        "stage": stage,
        "source_subject_id": source_subject_id,
        "candidate_pilot_subject_id": expected_candidate_subject,
        "source_checkpoint_sha256": source_checkpoint_sha256,
        "full_manifest_sha256": full_manifest_sha256,
        "training_args": expected_runtime,
    }


def evaluate_residual_candidate_pilot(
    summary: Mapping[str, Any],
    *,
    recipe: Mapping[str, Any],
    source_subject_id: str,
    source_checkpoint: Path,
    source_checkpoint_sha256: str,
    full_manifest_sha256: str,
    expected_recipient_records: int = EXPECTED_RECIPIENT_VAL_RECORDS,
) -> dict[str, object]:
    """Validate the fixed eight-epoch trim-zero residual candidate pilot."""

    if (
        isinstance(expected_recipient_records, bool)
        or not isinstance(expected_recipient_records, int)
        or expected_recipient_records != EXPECTED_RECIPIENT_VAL_RECORDS
    ):
        raise ValueError(
            "candidate pilot recipient val denominator must be exactly "
            f"{EXPECTED_RECIPIENT_VAL_RECORDS}"
        )

    config = _mapping(summary.get("config"), "candidate-pilot config")
    initialization = _mapping(summary.get("initialization"), "candidate-pilot initialization")
    source_config = _mapping(initialization.get("source_config"), "candidate-pilot source config")
    fine_tune = _mapping(summary.get("fine_tune_policy"), "candidate-pilot fine-tune policy")
    runtime = _mapping(summary.get("training_runtime"), "candidate-pilot runtime")
    validate_full_crop_training_recipe(
        recipe,
        stage="residual-8e",
        source_subject_id=source_subject_id,
        candidate_pilot_subject_id=None,
        source_checkpoint_sha256=source_checkpoint_sha256,
        full_manifest_sha256=full_manifest_sha256,
    )
    split_policy = _mapping(
        summary.get("recipient_train_split_policy"), "candidate-pilot split policy"
    )
    checkpoint_policy = _mapping(
        summary.get("checkpoint_selection_policy"), "candidate-pilot checkpoint policy"
    )
    protected = _mapping(
        checkpoint_policy.get("protected_minimum_candidate_exact"),
        "candidate-pilot protected floors",
    )
    try:
        source_reader = UnifiedReaderConfig(**dict(source_config))
        target_reader = UnifiedReaderConfig(**dict(config))
        source_reader.validate()
        target_reader.validate()
        _validate_recipient_visual_context_reinit_config(source_reader, target_reader)
    except (TypeError, ValueError) as error:
        raise ValueError("candidate pilot does not prove the sealed residual config transition") from error
    if (
        summary.get("kind") != KIND_V13
        or int(config.get("architecture_version", -1)) != 13
        or config.get("recipient_backbone") != REQUIRED_BACKBONE
        or int(config.get("recipient_open_text_layers", -1)) != 4
        or not math.isclose(
            _finite_rate(config.get("recipient_open_text_dropout"), "candidate dropout"),
            0.10,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        or not math.isclose(
            _finite_rate(config.get("recipient_value_left_trim"), "candidate trim"),
            0.0,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        or source_config.get("recipient_backbone") != REQUIRED_SOURCE_BACKBONE
        or not math.isclose(
            _finite_rate(source_config.get("recipient_value_left_trim"), "source trim"),
            0.0,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        or initialization.get("mode") != "parameter_only_recipient_visual_context_reinit"
        or initialization.get("init_checkpoint_mode")
        != INIT_CHECKPOINT_MODE_RECIPIENT_VISUAL_CONTEXT_REINIT
        or initialization.get("source_kind") != KIND_V13
        or initialization.get("checkpoint_sha256") != source_checkpoint_sha256
        or fine_tune.get("mode") != "recipient_only_v13"
        or fine_tune.get("trainable_parameter_prefix") != "recipient_"
        or fine_tune.get("training_forward") != "private_recipient_branch_only_v13"
        or runtime.get("device") != "cuda:0"
        or runtime.get("uses_cuda") is not True
        or "4090" not in str(runtime.get("cuda_device_name", ""))
        or runtime.get("num_workers") != 4
        or runtime.get("prefetch_factor") != 2
        or runtime.get("persistent_workers") is not True
        or runtime.get("validation_every") != 1
        or runtime.get("cuda_tf32_requested") is not True
        or runtime.get("cudnn_benchmark_requested") is not True
        or split_policy.get("mode") != "standard_train_only"
        or list(split_policy.get("splits", [])) != ["train"]
        or checkpoint_policy.get("mode") != CHECKPOINT_SELECTION_RECIPIENT_PRIORITY
        or summary.get("status_text_runtime_policy") != STATUS_TEXT_RUNTIME_POLICY
    ):
        raise ValueError("candidate pilot does not prove the guarded trim-zero residual recipe")
    raw_source_path = initialization.get("checkpoint_path")
    if not isinstance(raw_source_path, str) or not raw_source_path:
        raise ValueError("candidate pilot does not bind its source checkpoint path")
    bound_source = _existing_non_reparse(
        Path(raw_source_path), directory=False, description="candidate-pilot source checkpoint"
    )
    _samefile(source_checkpoint, bound_source, "candidate-pilot source checkpoint")
    if _sha256(bound_source) != source_checkpoint_sha256:
        raise ValueError("candidate-pilot source checkpoint bytes changed")
    for name, floor in {
        "amount": AMOUNT_FLOOR,
        "time": TIME_FLOOR,
        "payment_method_field": PAYMENT_FLOOR,
    }.items():
        actual = _finite_rate(protected.get(name), f"candidate-pilot {name} floor")
        if not math.isclose(actual, floor, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(f"candidate-pilot {name} protection floor changed")

    field_counts = _mapping(summary.get("field_counts"), "candidate-pilot field counts")
    for field, raw_counts in field_counts.items():
        counts = _mapping(raw_counts, f"candidate-pilot {field} counts")
        _require_equal(counts.get("test"), 0, f"candidate-pilot {field} test count")
    recipient_counts = _mapping(
        field_counts.get("recipient_field"), "candidate-pilot recipient field counts"
    )
    _require_equal(
        recipient_counts.get("val"),
        expected_recipient_records,
        "candidate-pilot recipient val count",
    )
    recipient_oov = _mapping(
        summary.get("recipient_oov_by_split"), "candidate-pilot recipient OOV"
    )
    recipient_val = _mapping(recipient_oov.get("val"), "candidate-pilot val OOV")
    _require_equal(
        recipient_val.get("records"),
        expected_recipient_records,
        "candidate-pilot val OOV records",
    )
    recipient_test = _mapping(recipient_oov.get("test"), "candidate-pilot test OOV")
    _require_equal(recipient_test.get("records"), 0, "candidate-pilot test OOV records")

    raw_records = summary.get("records")
    if not isinstance(raw_records, Sequence) or isinstance(raw_records, (str, bytes)):
        raise ValueError("candidate pilot has invalid epoch records")
    records = [_mapping(record, "candidate-pilot epoch record") for record in raw_records]
    if [record.get("epoch") for record in records] != list(range(1, PILOT_EPOCHS + 1)):
        raise ValueError("candidate pilot requires ordered epochs 1 through 8")
    if any(record.get("validation_performed") is not True for record in records):
        raise ValueError("candidate pilot requires validation at every epoch")
    for record in records:
        epoch = int(record["epoch"])
        fields = _mapping(
            record.get("val_candidate_text_by_field"), f"candidate-pilot epoch {epoch} fields"
        )
        recipient_metric = _mapping(
            fields.get("recipient_field"),
            f"candidate-pilot epoch {epoch} recipient metric",
        )
        _exact_count_metric(
            recipient_metric,
            expected_records=expected_recipient_records,
            description=f"candidate-pilot epoch {epoch} recipient metric",
        )
        for name, floor in {
            "amount": AMOUNT_FLOOR,
            "time": TIME_FLOOR,
            "payment_method_field": PAYMENT_FLOOR,
        }.items():
            metric = _mapping(fields.get(name), f"candidate-pilot epoch {epoch} {name}")
            if _finite_rate(metric.get("exact_match"), f"epoch {epoch} {name} exact") < floor:
                raise ValueError(f"candidate-pilot epoch {epoch} violated the {name} floor")
        if (
            record.get("checkpoint_selection_eligible") is not True
            or record.get("checkpoint_selection_protection_failures") != []
        ):
            raise ValueError(f"candidate-pilot epoch {epoch} was not checkpoint eligible")
        _require_equal(
            record.get("val_status_non_success_to_success"),
            0,
            f"candidate-pilot epoch {epoch} unsafe status errors",
        )
        raw_ctc = _mapping(record.get("val_ctc_by_field"), f"epoch {epoch} CTC fields")
        status = _mapping(raw_ctc.get("transfer_status"), f"epoch {epoch} status metric")
        if _finite_rate(status.get("exact_match"), f"epoch {epoch} status exact") < STATUS_TEXT_FLOOR:
            raise ValueError(f"candidate-pilot epoch {epoch} violated the status floor")

    best_epoch = summary.get("best_checkpoint_epoch")
    if isinstance(best_epoch, bool) or not isinstance(best_epoch, int):
        raise ValueError("candidate pilot has an invalid best checkpoint epoch")
    best_rows = [record for record in records if record.get("epoch") == best_epoch]
    if len(best_rows) != 1:
        raise ValueError("candidate pilot does not have exactly one best epoch record")
    best_recipient = _recipient_exact(best_rows[0], "candidate-pilot best checkpoint")
    maximum = max(_recipient_exact(record, f"epoch {record['epoch']}") for record in records)
    if not math.isclose(best_recipient, maximum, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("candidate-pilot best checkpoint is not recipient-optimal")
    epoch4 = next(record for record in records if record.get("epoch") == 4)
    epoch8 = next(record for record in records if record.get("epoch") == 8)
    epoch4_recipient = _recipient_exact(epoch4, "candidate-pilot epoch 4")
    epoch8_recipient = _recipient_exact(epoch8, "candidate-pilot epoch 8")
    gain = epoch8_recipient - epoch4_recipient
    failures: list[str] = []
    if best_recipient < PILOT_MINIMUM_BEST_RECIPIENT:
        failures.append("best_recipient_below_75_percent")
    if gain < PILOT_MINIMUM_EPOCH4_TO_8_GAIN:
        failures.append("epoch4_to_8_gain_below_2pp")
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": CANDIDATE_PILOT_KIND,
        "analysis_only": True,
        "production_route_authorized": False,
        "test_opened": False,
        "onnx_exported": False,
        "epochs": PILOT_EPOCHS,
        "fixed_gates": {
            "minimum_best_recipient_exact": PILOT_MINIMUM_BEST_RECIPIENT,
            "minimum_epoch4_to_8_gain": PILOT_MINIMUM_EPOCH4_TO_8_GAIN,
            "amount_candidate_exact_floor": AMOUNT_FLOOR,
            "time_candidate_exact_floor": TIME_FLOOR,
            "payment_candidate_exact_floor": PAYMENT_FLOOR,
            "visible_status_raw_exact_floor": STATUS_TEXT_FLOOR,
            "status_non_success_to_success_max": 0,
            "required_recipient_candidate_records": expected_recipient_records,
        },
        "observed": {
            "best_epoch": best_epoch,
            "best_recipient_exact": best_recipient,
            "epoch4_recipient_exact": epoch4_recipient,
            "epoch8_recipient_exact": epoch8_recipient,
            "epoch4_to_8_gain": gain,
            "recipient_candidate_records": expected_recipient_records,
            "recipient_candidate_coverage": 1.0,
        },
        "passed": not failures,
        "failures": failures,
        "decision": CANDIDATE_PILOT_DECISION if not failures else "analysis_only_stop",
    }


def _candidate_pilot_paths(candidate_root: Path) -> dict[str, Path]:
    training = candidate_root / "training-v14-candidate"
    blind = candidate_root / "blind-train-val"
    return {
        "candidate_best_checkpoint": training / "best.pt",
        "candidate_training_summary": training / "training_summary.json",
        "candidate_blind_manifest": blind / "unified_fields.train-val.jsonl",
        "candidate_blind_contract": blind / "blind.contract.json",
        "training_recipe": candidate_root / "recipient_v14_training_recipe.json",
    }


def _candidate_pilot_payload(
    *,
    candidate_root: Path,
    source_contract_path: Path,
    full_records: Path,
    torch: Any,
) -> dict[str, object]:
    root = _existing_non_reparse(
        candidate_root, directory=True, description="residual candidate pilot root"
    )
    _assert_no_onnx(root)
    source_contract_file = _existing_non_reparse(
        source_contract_path,
        directory=False,
        description="full-crop source contract",
    )
    source_contract_document = _strict_json(source_contract_file)
    raw_pilot_root = source_contract_document.get("pilot_root")
    if not isinstance(raw_pilot_root, str) or not raw_pilot_root:
        raise ValueError("full-crop source contract has no pilot root")
    source_contract = verify_full_crop_candidate_source(
        pilot_root=Path(raw_pilot_root),
        contract_path=source_contract_file,
        full_records=full_records,
        torch=torch,
    )
    source_artifacts = _mapping(source_contract.get("artifacts"), "source artifacts")
    source_checkpoint = _binding_path(source_artifacts, "best_checkpoint")
    source_sha = str(_mapping(source_artifacts["best_checkpoint"], "best binding")["sha256"])
    paths = {
        name: _existing_non_reparse(path, directory=False, description=name)
        for name, path in _candidate_pilot_paths(root).items()
    }
    before = {name: _sha256(path) for name, path in paths.items()}
    blind_binding = verify_blind_manifest_contract(
        records_path=paths["candidate_blind_manifest"],
        blind_contract_path=paths["candidate_blind_contract"],
    )
    blind_binding = {
        **blind_binding,
        "recipient_val_records": _blind_recipient_val_records(blind_binding),
    }
    supplied_full = _existing_non_reparse(
        full_records, directory=False, description="candidate full manifest"
    )
    bound_full = _existing_non_reparse(
        Path(str(blind_binding["source_manifest"])),
        directory=False,
        description="candidate-pilot bound full manifest",
    )
    _samefile(supplied_full, bound_full, "candidate-pilot full manifest")
    supplied_full_sha256 = _sha256(supplied_full)
    if supplied_full_sha256 != blind_binding.get("source_manifest_sha256"):
        raise ValueError("candidate-pilot full manifest changed after blind-contract creation")
    source_subject_id = source_contract.get("source_subject_id")
    if not isinstance(source_subject_id, str) or len(source_subject_id) != 64:
        raise ValueError("source contract has no valid path-independent subject id")
    summary = _strict_json(paths["candidate_training_summary"])
    recipe = _strict_json(paths["training_recipe"])
    decision = evaluate_residual_candidate_pilot(
        summary,
        recipe=recipe,
        source_subject_id=source_subject_id,
        source_checkpoint=source_checkpoint,
        source_checkpoint_sha256=source_sha,
        full_manifest_sha256=supplied_full_sha256,
        expected_recipient_records=int(blind_binding["recipient_val_records"]),
    )
    _require_equal(decision.get("passed"), True, "candidate pilot passed")
    paths["full_manifest"] = supplied_full
    paths["source_contract"] = _existing_non_reparse(
        source_contract_file, directory=False, description="full-crop source contract"
    )
    before.update(
        {
            "full_manifest": supplied_full_sha256,
            "source_contract": _sha256(paths["source_contract"]),
        }
    )

    best_payload = _load_checkpoint(paths["candidate_best_checkpoint"], torch=torch)
    if not isinstance(best_payload, Mapping):
        raise ValueError("candidate-pilot best checkpoint payload must be an object")
    if best_payload.get("kind") != KIND_V13:
        raise ValueError("candidate-pilot best checkpoint is not a v13 reader")
    best_state = best_payload.get("state_dict")
    if not isinstance(best_state, Mapping) or not best_state:
        raise ValueError("candidate-pilot best checkpoint has no model state")
    best_state_identity = _checkpoint_state_identity(best_state)
    try:
        best_config = _checkpoint_config(best_payload)
        best_config.validate()
    except (TypeError, ValueError) as error:
        raise ValueError("candidate-pilot best checkpoint has an invalid model config") from error
    _json_equivalent(
        asdict(best_config), summary.get("config"), "candidate-pilot validated config"
    )
    best_epoch = summary.get("best_checkpoint_epoch")
    _require_equal(best_payload.get("epoch"), best_epoch, "candidate-pilot best epoch")
    best_rows = [
        record
        for record in summary.get("records", [])
        if isinstance(record, Mapping) and record.get("epoch") == best_epoch
    ]
    if len(best_rows) != 1:
        raise ValueError("candidate pilot has no unique best epoch record")
    for key, expected in (
        ("metrics", best_rows[0]),
        ("config", summary.get("config")),
        ("initialization", summary.get("initialization")),
        ("fine_tune_policy", summary.get("fine_tune_policy")),
        ("status_text_runtime_policy", summary.get("status_text_runtime_policy")),
        ("training_runtime", summary.get("training_runtime")),
    ):
        _json_equivalent(best_payload.get(key), expected, f"candidate-pilot best {key}")
    after = {name: _sha256(path) for name, path in paths.items()}
    if before != after:
        raise ValueError("candidate-pilot evidence changed while it was being sealed")

    artifacts = {name: _binding(path) for name, path in paths.items()}
    artifacts["source_best_checkpoint"] = _binding(source_checkpoint)
    for name, path in _code_paths().items():
        artifacts[name] = _binding(path)
    recipe_identity = validate_full_crop_training_recipe(
        recipe,
        stage="residual-8e",
        source_subject_id=source_subject_id,
        candidate_pilot_subject_id=None,
        source_checkpoint_sha256=source_sha,
        full_manifest_sha256=supplied_full_sha256,
    )
    candidate_subject_id = _canonical_sha256(
        {
            "domain": CANDIDATE_PILOT_SUBJECT_DOMAIN,
            "kind": CANDIDATE_PILOT_KIND,
            "source_subject_id": source_subject_id,
            "decision": decision,
            "best_checkpoint": {
                "kind": best_payload.get("kind"),
                "epoch": best_epoch,
                "config": asdict(best_config),
                "state": best_state_identity,
            },
            "blind_manifest": _blind_semantic_identity(blind_binding),
            "training_recipe": {
                "stage": recipe_identity["stage"],
                "source_subject_id": recipe_identity["source_subject_id"],
                "full_manifest_sha256": recipe_identity["full_manifest_sha256"],
                "training_args": recipe_identity["training_args"],
            },
            "code": _code_content_identity(artifacts),
        }
    )
    return {
        **decision,
        "source_subject_id": source_subject_id,
        "candidate_pilot_subject_id": candidate_subject_id,
        "candidate_root": str(root),
        "source_contract_sha256": _sha256(paths["source_contract"]),
        "source_best_checkpoint_sha256": source_sha,
        "blind_manifest_contract": blind_binding,
        "artifacts": artifacts,
    }


def seal_residual_candidate_pilot(
    *,
    candidate_root: Path,
    source_contract_path: Path,
    full_records: Path,
    output_evidence: Path,
    torch: Any | None = None,
) -> dict[str, object]:
    if torch is None:
        torch, _ = _require_torch()
    output = _fresh_contract_path(output_evidence)
    payload = _candidate_pilot_payload(
        candidate_root=candidate_root,
        source_contract_path=source_contract_path,
        full_records=full_records,
        torch=torch,
    )
    sealed = {**payload, "integrity_sha256": _canonical_sha256(payload)}
    encoded = json.dumps(
        sealed, ensure_ascii=False, allow_nan=False, sort_keys=True, indent=2
    ) + "\n"
    try:
        with output.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
    except FileExistsError as error:
        raise ValueError(f"Refusing to overwrite candidate-pilot evidence: {output}") from error
    return sealed


def verify_residual_candidate_pilot(
    *,
    evidence_path: Path,
    source_contract_path: Path,
    full_records: Path,
    torch: Any | None = None,
) -> dict[str, object]:
    if torch is None:
        torch, _ = _require_torch()
    evidence_file = _existing_non_reparse(
        evidence_path, directory=False, description="candidate-pilot evidence"
    )
    evidence = _strict_json(evidence_file)
    claimed = evidence.get("integrity_sha256")
    unsigned = {key: value for key, value in evidence.items() if key != "integrity_sha256"}
    if not isinstance(claimed, str) or claimed != _canonical_sha256(unsigned):
        raise ValueError("candidate-pilot evidence integrity hash does not match")
    if evidence.get("schema_version") != SCHEMA_VERSION or evidence.get("kind") != CANDIDATE_PILOT_KIND:
        raise ValueError("candidate-pilot evidence kind/schema is unsupported")
    raw_root = evidence.get("candidate_root")
    if not isinstance(raw_root, str) or not raw_root:
        raise ValueError("candidate-pilot evidence has no candidate root")
    rebuilt = _candidate_pilot_payload(
        candidate_root=Path(raw_root),
        source_contract_path=source_contract_path,
        full_records=full_records,
        torch=torch,
    )
    _json_equivalent(unsigned, rebuilt, "candidate-pilot evidence")
    return evidence


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Attest the full-crop pilot source and guarded residual pilot"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    attest = subparsers.add_parser("attest-source")
    attest.add_argument("--pilot-root", type=Path, required=True)
    attest.add_argument("--output-contract", type=Path, required=True)
    verify = subparsers.add_parser("verify-source")
    verify.add_argument("--pilot-root", type=Path, required=True)
    verify.add_argument("--contract", type=Path, required=True)
    verify.add_argument("--full-records", type=Path)
    seal_pilot = subparsers.add_parser("seal-candidate-pilot")
    seal_pilot.add_argument("--candidate-root", type=Path, required=True)
    seal_pilot.add_argument("--source-contract", type=Path, required=True)
    seal_pilot.add_argument("--full-records", type=Path, required=True)
    seal_pilot.add_argument("--output-evidence", type=Path, required=True)
    verify_pilot = subparsers.add_parser("verify-candidate-pilot")
    verify_pilot.add_argument("--evidence", type=Path, required=True)
    verify_pilot.add_argument("--source-contract", type=Path, required=True)
    verify_pilot.add_argument("--full-records", type=Path, required=True)
    verify_training = subparsers.add_parser("verify-candidate-training")
    verify_training.add_argument("--summary", type=Path, required=True)
    verify_val = subparsers.add_parser("verify-candidate-val")
    verify_val.add_argument("--summary", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "attest-source":
        payload = attest_full_crop_candidate_source(
            pilot_root=args.pilot_root,
            output_contract=args.output_contract,
        )
    elif args.command == "verify-source":
        payload = verify_full_crop_candidate_source(
            pilot_root=args.pilot_root,
            contract_path=args.contract,
            full_records=args.full_records,
        )
    elif args.command == "seal-candidate-pilot":
        payload = seal_residual_candidate_pilot(
            candidate_root=args.candidate_root,
            source_contract_path=args.source_contract,
            full_records=args.full_records,
            output_evidence=args.output_evidence,
        )
    elif args.command == "verify-candidate-pilot":
        payload = verify_residual_candidate_pilot(
            evidence_path=args.evidence,
            source_contract_path=args.source_contract,
            full_records=args.full_records,
        )
    elif args.command == "verify-candidate-training":
        payload = validate_full_crop_candidate_training_metrics(
            _strict_json(
                _existing_non_reparse(
                    args.summary,
                    directory=False,
                    description="candidate training summary",
                )
            )
        )
    else:
        payload = validate_full_crop_candidate_val_metrics(
            _strict_json(
                _existing_non_reparse(
                    args.summary,
                    directory=False,
                    description="candidate validation summary",
                )
            )
        )
    print(json.dumps(payload, ensure_ascii=False, allow_nan=False, sort_keys=True))


if __name__ == "__main__":
    main()
