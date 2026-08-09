"""Cryptographic preflight for the recipient-v14 one-shot test gate.

The final held-out test is intentionally opened only after a candidate has
been selected on validation.  Paths are not identities: copying an unchanged
model bundle must not create another test attempt.  This module therefore
derives a path-independent subject identifier from the bytes that determine
test inference and independently cross-checks every train/validation evidence
artifact before the PowerShell launcher creates its persistent attempt lock.

SHA-256 establishes local byte identity, not authorship.  An administrator can
still delete a local audit registry; deployments that must resist a malicious
administrator need a signed manifest and an external append-only registry.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any


SCHEMA_VERSION = 1
CANDIDATE_KIND = "receipt_recipient_v14_blind_candidate_v1"
INSPECTION_KIND = "receipt_recipient_v14_final_gate_subject_v1"
BLIND_KIND = "receipt_recipient_blind_train_val_manifest_v1"
MODEL_KIND = "receipt_unified_field_reader_v13"
REQUIRED_BACKBONE = "residual_positional_transformer_v2"
REQUIRED_STATUS_POLICY = "decode_and_normalize_review_only"
REQUIRED_INIT_MODE = "parameter_only_recipient_visual_context_reinit"
REQUIRED_FINE_TUNE_MODE = "recipient_only_v13"
SUBJECT_DOMAIN = "receipt-recipient-v14-final-gate-subject-v1"
EVIDENCE_DOMAIN = "receipt-recipient-v14-final-gate-evidence-v1"

FIXED_FLOORS: dict[str, float] = {
    "amount": 0.7885,
    "time": 0.9840,
    "payment_method_field": 0.9325,
    "recipient_field": 0.90,
    "visible_transfer_status_cjk_text": 0.90,
}
EVALUATION_KINDS = {
    "receipt_unified_field_reader_truth_evaluation_v1",
    "receipt_unified_field_reader_teacher_parity_v1",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
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


def _load_json(path: Path) -> dict[str, Any]:
    def reject_constant(value: str) -> None:
        raise ValueError(f"{path}: non-finite JSON constant {value!r}")

    try:
        value = json.loads(path.read_text(encoding="utf-8-sig"), parse_constant=reject_constant)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"Unable to read strict JSON object {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")

    # ``parse_constant`` rejects the non-standard NaN/Infinity tokens, but a
    # standards-compliant numeric token such as ``1e999`` still overflows to
    # ``float('inf')`` in CPython.  Walk the complete document so an ignored or
    # future numeric field cannot smuggle a non-finite value through the gate.
    def reject_nonfinite(item: object, location: str) -> None:
        if isinstance(item, float) and not math.isfinite(item):
            raise ValueError(f"{path}: non-finite JSON number at {location}")
        if isinstance(item, Mapping):
            for key, child in item.items():
                reject_nonfinite(child, f"{location}.{key}")
        elif isinstance(item, list):
            for index, child in enumerate(item):
                reject_nonfinite(child, f"{location}[{index}]")

    reject_nonfinite(value, "$")
    return value


def _mapping(value: object, description: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{description} must be an object")
    return value


def _path_from(
    owner: Mapping[str, Any],
    key: str,
    *,
    base: Path,
    description: str,
) -> Path:
    raw = owner.get(key)
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError(f"Missing {description} path")
    path = Path(raw)
    if not path.is_absolute():
        path = base / path
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Missing {description}: {path}")
    return path


def _reject_reparse_chain(raw_path: Path, *, base: Path, description: str) -> None:
    """Reject symlink/junction/reparse aliases before ``Path.resolve`` hides them."""

    path = raw_path if raw_path.is_absolute() else base / raw_path
    path = Path(os.path.abspath(os.fspath(path.expanduser())))
    current = path
    while True:
        if os.path.lexists(os.fspath(current)):
            try:
                attributes = int(getattr(current.lstat(), "st_file_attributes", 0))
            except OSError as error:
                raise ValueError(f"Unable to inspect {description} path") from error
            if current.is_symlink() or bool(attributes & 0x400):
                raise ValueError(f"{description} must not traverse a reparse path")
        if current == current.parent:
            break
        current = current.parent


def _require_equal(actual: object, expected: object, description: str) -> None:
    # JSON booleans compare equal to integers in Python (``True == 1`` and
    # ``False == 0``).  Gate contract fields are typed, so equality alone is
    # not sufficient for schema/security checks.
    if type(actual) is not type(expected) or actual != expected:
        raise ValueError(f"{description} mismatch: expected {expected!r}, found {actual!r}")


def _finite_rate(value: object, description: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{description} must be a finite rate")
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{description} must be a finite rate") from error
    if not math.isfinite(number) or number < 0.0 or number > 1.0:
        raise ValueError(f"{description} must be between zero and one")
    return number


def _fixed_floor_payload(value: object, description: str) -> dict[str, float]:
    floors = _mapping(value, description)
    output: dict[str, float] = {}
    if set(floors) != set(FIXED_FLOORS):
        raise ValueError(f"{description} must contain exactly the fixed delivery floors")
    for name, expected in FIXED_FLOORS.items():
        actual = _finite_rate(floors.get(name), f"{description}.{name}")
        if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(
                f"{description}.{name} changed: expected {expected:.12g}, found {actual:.12g}"
            )
        output[name] = actual
    return output


def derive_gate_identity(
    *,
    model: Path,
    full_manifest: Path,
    contract: Path,
    labels: Path,
    evidence_binding: Mapping[str, object],
) -> dict[str, str]:
    """Return path-independent subject/evidence identities from actual bytes.

    ``gate_subject_id`` alone names the persistent one-shot lock.  Evidence
    hashes are deliberately excluded from that identifier: editing or copying
    metadata must not yield another attempt for the same model/test semantics.
    ``evidence_identity`` remains in the lock payload for provenance auditing.
    """

    actual = {
        "model_sha256": _sha256(model.resolve()),
        "full_manifest_sha256": _sha256(full_manifest.resolve()),
        "contract_sha256": _sha256(contract.resolve()),
        "labels_sha256": _sha256(labels.resolve()),
    }
    subject_payload: dict[str, object] = {
        "domain": SUBJECT_DOMAIN,
        **actual,
    }
    evidence_payload: dict[str, object] = {
        "domain": EVIDENCE_DOMAIN,
        "gate_subject_id": _canonical_sha256(subject_payload),
        **dict(evidence_binding),
    }
    return {
        **actual,
        "gate_subject_id": _canonical_sha256(subject_payload),
        "evidence_identity": _canonical_sha256(evidence_payload),
    }


def _require_claimed_hash(
    owner: Mapping[str, Any],
    key: str,
    actual: str,
    description: str,
) -> None:
    claimed = owner.get(key)
    if not isinstance(claimed, str) or claimed.lower() != actual:
        raise ValueError(f"{description} claimed SHA-256 does not match actual bytes")


def _raw_exact(summary: Mapping[str, Any], field: str, metric: str = "raw_exact_match") -> float:
    by_field = _mapping(summary.get("by_field"), "evaluation by_field")
    field_metrics = _mapping(by_field.get(field), f"evaluation by_field.{field}")
    records = field_metrics.get("records")
    if isinstance(records, bool) or not isinstance(records, int) or records <= 0:
        raise ValueError(f"evaluation by_field.{field}.records must be positive")
    if metric == "ctc_raw_exact_match" and field_metrics.get("ctc_records") != records:
        raise ValueError(f"evaluation by_field.{field}.ctc_records must equal records")
    return _finite_rate(field_metrics.get(metric), f"evaluation by_field.{field}.{metric}")


def _validate_evaluation_kind(summary: Mapping[str, Any], description: str) -> None:
    if summary.get("schema_version") != SCHEMA_VERSION or summary.get("kind") not in EVALUATION_KINDS:
        raise ValueError(f"{description} kind/schema is unsupported")


def _validate_acceptance(
    summary: Mapping[str, Any],
    *,
    require_passed: bool,
) -> list[str]:
    acceptance = _mapping(summary.get("acceptance"), "evaluation acceptance")
    _require_equal(acceptance.get("requested"), True, "evaluation acceptance.requested")
    requested = {
        "min_amount_exact_match": FIXED_FLOORS["amount"],
        "min_time_exact_match": FIXED_FLOORS["time"],
        "min_payment_exact_match": FIXED_FLOORS["payment_method_field"],
        "min_recipient_exact_match": FIXED_FLOORS["recipient_field"],
        "min_status_exact_match": FIXED_FLOORS["visible_transfer_status_cjk_text"],
    }
    for name, expected in requested.items():
        actual = _finite_rate(acceptance.get(name), f"evaluation acceptance.{name}")
        if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(f"evaluation acceptance.{name} does not equal the fixed floor")
    _require_equal(
        acceptance.get("max_non_success_to_success"),
        0,
        "evaluation acceptance.max_non_success_to_success",
    )
    raw_failures = acceptance.get("failures")
    if not isinstance(raw_failures, list) or not all(isinstance(item, str) for item in raw_failures):
        raise ValueError("evaluation acceptance.failures must be a string list")
    if require_passed:
        _require_equal(acceptance.get("passed"), True, "evaluation acceptance.passed")
        if raw_failures:
            raise ValueError("accepted validation evidence contains failures")
    return list(raw_failures)


def _validate_status_policy(summary: Mapping[str, Any]) -> None:
    policy = _mapping(summary.get("status_text_policy"), "status text policy")
    _require_equal(policy.get("runtime_policy"), REQUIRED_STATUS_POLICY, "status runtime policy")
    _require_equal(policy.get("review_value"), "review", "status review value")


def inspect_candidate(
    candidate_evidence: Path,
    *,
    trusted_full_manifest_sha256: str,
) -> dict[str, Any]:
    candidate_evidence = candidate_evidence.resolve()
    if not candidate_evidence.is_file():
        raise FileNotFoundError(candidate_evidence)
    trusted_full_manifest_sha256 = trusted_full_manifest_sha256.strip().lower()
    if len(trusted_full_manifest_sha256) != 64 or any(
        character not in "0123456789abcdef" for character in trusted_full_manifest_sha256
    ):
        raise ValueError("trusted full-manifest SHA-256 must be exactly 64 lowercase hex characters")

    candidate = _load_json(candidate_evidence)
    _require_equal(candidate.get("schema_version"), SCHEMA_VERSION, "candidate schema_version")
    _require_equal(candidate.get("kind"), CANDIDATE_KIND, "candidate kind")
    split_policy = _mapping(candidate.get("split_policy"), "candidate split_policy")
    _require_equal(split_policy.get("optimizer_supervision"), ["train"], "optimizer supervision")
    _require_equal(split_policy.get("checkpoint_selection"), ["val"], "checkpoint selection")
    _require_equal(split_policy.get("final_gate_only"), ["test"], "final-gate split")
    _require_equal(split_policy.get("test_evaluated"), False, "candidate test_evaluated")
    floors = _fixed_floor_payload(candidate.get("fixed_floors"), "candidate fixed_floors")

    base = candidate_evidence.parent
    candidate_model = _mapping(candidate.get("candidate"), "candidate model binding")
    training_binding = _mapping(candidate.get("training"), "candidate training binding")
    val_binding = _mapping(candidate.get("val_evaluation"), "candidate val binding")
    full_manifest = _path_from(candidate, "full_manifest", base=base, description="full manifest")
    blind_manifest = _path_from(candidate, "blind_manifest", base=base, description="blind manifest")
    blind_contract = _path_from(
        split_policy,
        "blind_contract",
        base=base,
        description="blind manifest contract",
    )
    model = _path_from(candidate_model, "model", base=base, description="candidate ONNX")
    contract = _path_from(candidate_model, "contract", base=base, description="candidate contract")
    labels = _path_from(candidate_model, "labels", base=base, description="candidate labels")
    checkpoint = _path_from(candidate_model, "checkpoint", base=base, description="candidate checkpoint")
    training_summary = _path_from(
        training_binding,
        "summary",
        base=base,
        description="training summary",
    )
    val_summary = _path_from(val_binding, "summary", base=base, description="validation summary")

    if contract != model.with_suffix(".contract.json") or labels != model.with_suffix(".labels.json"):
        raise ValueError("candidate contract and labels must be adjacent to the ONNX model")

    actual_hashes = {
        "candidate_evidence": _sha256(candidate_evidence),
        "full_manifest": _sha256(full_manifest),
        "blind_manifest": _sha256(blind_manifest),
        "blind_contract": _sha256(blind_contract),
        "model": _sha256(model),
        "contract": _sha256(contract),
        "labels": _sha256(labels),
        "checkpoint": _sha256(checkpoint),
        "training_summary": _sha256(training_summary),
        "val_summary": _sha256(val_summary),
    }
    if actual_hashes["full_manifest"] != trusted_full_manifest_sha256:
        raise ValueError("actual full manifest does not match the independently trusted SHA-256")
    _require_claimed_hash(candidate, "full_manifest_sha256", actual_hashes["full_manifest"], "full manifest")
    _require_claimed_hash(candidate, "blind_manifest_sha256", actual_hashes["blind_manifest"], "blind manifest")
    _require_claimed_hash(split_policy, "blind_contract_sha256", actual_hashes["blind_contract"], "blind contract")
    for key in ("model", "contract", "labels", "checkpoint"):
        _require_claimed_hash(candidate_model, f"{key}_sha256", actual_hashes[key], key)
    _require_claimed_hash(training_binding, "summary_sha256", actual_hashes["training_summary"], "training summary")
    _require_claimed_hash(val_binding, "summary_sha256", actual_hashes["val_summary"], "validation summary")

    # The production loader is the authoritative model/sidecar contract
    # validator.  It checks all static shapes, charset hashes, output ABI and
    # runtime policies without opening the held-out test split.
    from .ocr_unified import _load_onnx_artifact_details

    config, _payment, _recipient, _contract = _load_onnx_artifact_details(model)
    _require_equal(config.architecture_version, 13, "candidate architecture_version")
    _require_equal(config.recipient_backbone, REQUIRED_BACKBONE, "candidate recipient backbone")
    _require_equal(config.recipient_open_text_layers, 4, "candidate recipient transformer layers")
    if not math.isclose(config.recipient_open_text_dropout, 0.10, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("candidate recipient dropout is not the sealed v14 value")
    _require_equal(candidate_model.get("architecture_version"), 13, "candidate evidence architecture")
    _require_equal(candidate_model.get("backbone"), REQUIRED_BACKBONE, "candidate evidence backbone")

    blind = _load_json(blind_contract)
    _require_equal(blind.get("schema_version"), SCHEMA_VERSION, "blind contract schema_version")
    _require_equal(blind.get("kind"), BLIND_KIND, "blind contract kind")
    _require_equal(blind.get("source_manifest_sha256"), actual_hashes["full_manifest"], "blind source hash")
    _require_equal(blind.get("blind_manifest_sha256"), actual_hashes["blind_manifest"], "blind manifest hash")
    _require_equal(blind.get("optimizer_supervision_splits"), ["train"], "blind optimizer splits")
    _require_equal(blind.get("checkpoint_selection_splits"), ["val"], "blind checkpoint splits")
    _require_equal(blind.get("final_gate_only_splits"), ["test"], "blind final-gate splits")
    for key in ("test_labels_used", "test_metrics_computed", "test_examples_emitted"):
        _require_equal(blind.get(key), False, f"blind contract {key}")
    split_counts = _mapping(blind.get("split_counts"), "blind split_counts")
    for key in ("train", "val", "test_excluded"):
        value = split_counts.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"blind split_counts.{key} must be positive")

    training = _load_json(training_summary)
    _require_equal(training.get("kind"), MODEL_KIND, "training kind")
    training_config = _mapping(training.get("config"), "training config")
    _require_equal(training_config.get("architecture_version"), 13, "training architecture")
    _require_equal(training_config.get("recipient_backbone"), REQUIRED_BACKBONE, "training backbone")
    _require_equal(training_config.get("recipient_open_text_layers"), 4, "training recipient layers")
    initialization = _mapping(training.get("initialization"), "training initialization")
    _require_equal(initialization.get("mode"), REQUIRED_INIT_MODE, "training initialization mode")
    training_runtime = _mapping(training.get("training_runtime"), "training runtime")
    fine_tune = _mapping(training.get("fine_tune_policy"), "training fine-tune policy")
    _require_equal(fine_tune.get("mode"), REQUIRED_FINE_TUNE_MODE, "training fine-tune mode")
    _require_equal(fine_tune.get("trainable_parameter_prefix"), "recipient_", "training parameter prefix")
    train_split_policy = _mapping(training.get("recipient_train_split_policy"), "recipient train split policy")
    _require_equal(train_split_policy.get("mode"), "standard_train_only", "recipient train split mode")
    _require_equal(train_split_policy.get("splits"), ["train"], "recipient train splits")
    field_counts = _mapping(training.get("field_counts"), "training field_counts")
    recipient_counts = _mapping(field_counts.get("recipient_field"), "training recipient counts")
    _require_equal(recipient_counts.get("test"), 0, "training recipient test count")
    recipient_oov = _mapping(training.get("recipient_oov_by_split"), "training recipient OOV audit")
    recipient_test_oov = _mapping(recipient_oov.get("test"), "training test recipient OOV audit")
    _require_equal(recipient_test_oov.get("records"), 0, "training test recipient OOV records")
    _require_equal(training.get("best_checkpoint_epoch"), training_binding.get("best_epoch"), "best epoch")

    # Historical v14 evidence predates an explicit source-route object.  It is
    # accepted only for the original 0.30-trim recipe.  Every newly emitted
    # candidate has an explicit route; trim zero is authorized exclusively by
    # a content-bound passed full-crop source plus its passed residual pilot.
    source_guard_paths: dict[str, Path] = {}
    source_route_binding: dict[str, object] = {"mode": "historical_legacy_v14"}
    raw_source_route = candidate.get("source_route")
    initialization_source_config = _mapping(
        initialization.get("source_config"), "training initialization source config"
    )
    target_trim = _finite_rate(
        training_config.get("recipient_value_left_trim"), "training recipient left trim"
    )
    source_trim = _finite_rate(
        initialization_source_config.get("recipient_value_left_trim"),
        "training source recipient left trim",
    )
    if not math.isclose(
        config.recipient_value_left_trim, target_trim, rel_tol=0.0, abs_tol=1e-12
    ):
        raise ValueError("ONNX and training recipient left trim differ")
    if raw_source_route is None:
        if not math.isclose(target_trim, 0.30, rel_tol=0.0, abs_tol=1e-12) or not math.isclose(
            source_trim, 0.30, rel_tol=0.0, abs_tol=1e-12
        ):
            raise ValueError(
                "historical v14 evidence may not authorize an unbound trim-zero source"
            )
        # Optional claims on historical evidence remain typed when present.
        if "analysis_only" in candidate:
            _require_equal(candidate.get("analysis_only"), True, "candidate analysis_only")
        if "production_route_authorized" in candidate:
            _require_equal(
                candidate.get("production_route_authorized"),
                False,
                "candidate production_route_authorized",
            )
    else:
        _require_equal(candidate.get("analysis_only"), True, "candidate analysis_only")
        _require_equal(
            candidate.get("production_route_authorized"),
            False,
            "candidate production_route_authorized",
        )
        source_route = _mapping(raw_source_route, "candidate source_route")
        mode = source_route.get("mode")
        if mode not in {
            "legacy_v13_visual_context_reinit",
            "attested_full_crop_pilot_visual_context_reinit",
        }:
            raise ValueError("candidate source_route mode is unsupported")
        route_trim = _finite_rate(
            source_route.get("recipient_value_left_trim"), "candidate source-route trim"
        )
        raw_source_checkpoint = source_route.get("source_checkpoint")
        if not isinstance(raw_source_checkpoint, str) or not raw_source_checkpoint:
            raise ValueError("candidate source route has no source checkpoint")
        _reject_reparse_chain(
            Path(raw_source_checkpoint),
            base=base,
            description="candidate source checkpoint",
        )
        source_checkpoint = _path_from(
            source_route,
            "source_checkpoint",
            base=base,
            description="candidate source checkpoint",
        )
        source_checkpoint_sha256 = _sha256(source_checkpoint)
        _require_claimed_hash(
            source_route,
            "source_checkpoint_sha256",
            source_checkpoint_sha256,
            "candidate source checkpoint",
        )
        raw_init_checkpoint = initialization.get("checkpoint_path")
        if not isinstance(raw_init_checkpoint, str) or not raw_init_checkpoint:
            raise ValueError("training initialization has no source checkpoint path")
        init_checkpoint = Path(raw_init_checkpoint)
        if not init_checkpoint.is_absolute():
            init_checkpoint = training_summary.parent / init_checkpoint
        init_checkpoint = init_checkpoint.resolve()
        if not init_checkpoint.is_file() or not source_checkpoint.samefile(init_checkpoint):
            raise ValueError("training initialization did not use the bound source checkpoint")
        _require_equal(
            initialization.get("checkpoint_sha256"),
            source_checkpoint_sha256,
            "training source checkpoint SHA-256",
        )
        source_guard_paths["source_checkpoint"] = source_checkpoint
        if mode == "legacy_v13_visual_context_reinit":
            if not math.isclose(route_trim, 0.30, rel_tol=0.0, abs_tol=1e-12):
                raise ValueError("legacy candidate source route must keep trim 0.30")
            if not math.isclose(target_trim, 0.30, rel_tol=0.0, abs_tol=1e-12) or not math.isclose(
                source_trim, 0.30, rel_tol=0.0, abs_tol=1e-12
            ):
                raise ValueError("legacy candidate training must keep trim 0.30")
            source_route_binding = {
                "mode": mode,
                "source_checkpoint_sha256": source_checkpoint_sha256,
            }
        else:
            if not math.isclose(route_trim, 0.0, rel_tol=0.0, abs_tol=1e-12):
                raise ValueError("full-crop candidate source route must lock trim zero")
            if not math.isclose(target_trim, 0.0, rel_tol=0.0, abs_tol=1e-12) or not math.isclose(
                source_trim, 0.0, rel_tol=0.0, abs_tol=1e-12
            ):
                raise ValueError("full-crop candidate training must keep source and target trim zero")
            fixed_runtime = {
                "num_workers": 4,
                "prefetch_factor": 2,
                "validation_every": 2,
            }
            for runtime_name, expected_runtime_value in fixed_runtime.items():
                _require_equal(
                    training_runtime.get(runtime_name),
                    expected_runtime_value,
                    f"full-crop training runtime {runtime_name}",
                )
            for runtime_name in (
                "persistent_workers",
                "cuda_tf32_requested",
                "cudnn_benchmark_requested",
            ):
                _require_equal(
                    training_runtime.get(runtime_name),
                    True,
                    f"full-crop training runtime {runtime_name}",
                )
            _require_equal(
                training_runtime.get("device"),
                "cuda:0",
                "full-crop training runtime device",
            )
            _require_equal(
                training_runtime.get("uses_cuda"),
                True,
                "full-crop training runtime uses_cuda",
            )
            if "4090" not in str(training_runtime.get("cuda_device_name", "")):
                raise ValueError("full-crop training runtime is not bound to an RTX 4090")
            raw_source_contract = source_route.get("source_contract")
            if not isinstance(raw_source_contract, str) or not raw_source_contract:
                raise ValueError("full-crop source route has no source contract")
            _reject_reparse_chain(
                Path(raw_source_contract),
                base=base,
                description="full-crop source contract",
            )
            source_contract = _path_from(
                source_route,
                "source_contract",
                base=base,
                description="full-crop source contract",
            )
            source_contract_sha256 = _sha256(source_contract)
            _require_claimed_hash(
                source_route,
                "source_contract_sha256",
                source_contract_sha256,
                "full-crop source contract",
            )
            raw_pilot_root = source_route.get("full_crop_pilot_root")
            if not isinstance(raw_pilot_root, str) or not raw_pilot_root:
                raise ValueError("full-crop source route has no pilot root")
            pilot_root = Path(raw_pilot_root)
            if not pilot_root.is_absolute():
                pilot_root = base / pilot_root
            _reject_reparse_chain(
                pilot_root,
                base=base,
                description="full-crop pilot root",
            )
            pilot_root = pilot_root.resolve()
            if not pilot_root.is_dir():
                raise FileNotFoundError(f"Missing full-crop pilot root: {pilot_root}")
            raw_candidate_pilot = source_route.get("candidate_pilot_evidence")
            if not isinstance(raw_candidate_pilot, str) or not raw_candidate_pilot:
                raise ValueError("full-crop source route has no residual pilot evidence")
            _reject_reparse_chain(
                Path(raw_candidate_pilot),
                base=base,
                description="residual candidate-pilot evidence",
            )
            candidate_pilot_evidence = _path_from(
                source_route,
                "candidate_pilot_evidence",
                base=base,
                description="residual candidate-pilot evidence",
            )
            candidate_pilot_sha256 = _sha256(candidate_pilot_evidence)
            _require_claimed_hash(
                source_route,
                "candidate_pilot_evidence_sha256",
                candidate_pilot_sha256,
                "residual candidate-pilot evidence",
            )
            from .recipient_full_crop_candidate_source import (
                validate_full_crop_training_recipe,
                verify_full_crop_candidate_source,
                verify_residual_candidate_pilot,
            )

            source_contract_payload = verify_full_crop_candidate_source(
                pilot_root=pilot_root,
                contract_path=source_contract,
                full_records=full_manifest,
            )
            candidate_pilot_payload = verify_residual_candidate_pilot(
                evidence_path=candidate_pilot_evidence,
                source_contract_path=source_contract,
                full_records=full_manifest,
            )
            source_subject_id = source_contract_payload.get("source_subject_id")
            candidate_pilot_subject_id = candidate_pilot_payload.get(
                "candidate_pilot_subject_id"
            )
            if not isinstance(source_subject_id, str) or len(source_subject_id) != 64:
                raise ValueError("full-crop source contract has no valid subject identity")
            if (
                not isinstance(candidate_pilot_subject_id, str)
                or len(candidate_pilot_subject_id) != 64
            ):
                raise ValueError("candidate-pilot evidence has no valid subject identity")
            _require_equal(
                candidate_pilot_payload.get("source_subject_id"),
                source_subject_id,
                "candidate-pilot source subject identity",
            )
            _require_equal(
                source_route.get("source_subject_id"),
                source_subject_id,
                "candidate source-route source subject identity",
            )
            _require_equal(
                source_route.get("candidate_pilot_subject_id"),
                candidate_pilot_subject_id,
                "candidate source-route pilot subject identity",
            )
            raw_training_recipe = training_binding.get("recipe")
            if not isinstance(raw_training_recipe, str) or not raw_training_recipe:
                raise ValueError("full-crop candidate has no bound training recipe")
            _reject_reparse_chain(
                Path(raw_training_recipe),
                base=base,
                description="full-crop candidate training recipe",
            )
            training_recipe = _path_from(
                training_binding,
                "recipe",
                base=base,
                description="full-crop candidate training recipe",
            )
            training_recipe_sha256 = _sha256(training_recipe)
            _require_claimed_hash(
                training_binding,
                "recipe_sha256",
                training_recipe_sha256,
                "full-crop candidate training recipe",
            )
            validate_full_crop_training_recipe(
                _load_json(training_recipe),
                stage="candidate-60e",
                source_subject_id=source_subject_id,
                candidate_pilot_subject_id=candidate_pilot_subject_id,
                source_checkpoint_sha256=source_checkpoint_sha256,
                full_manifest_sha256=actual_hashes["full_manifest"],
            )
            source_artifacts = _mapping(
                source_contract_payload.get("artifacts"), "full-crop source artifacts"
            )
            source_best = _path_from(
                _mapping(source_artifacts.get("best_checkpoint"), "source best binding"),
                "path",
                base=source_contract.parent,
                description="source-contract best checkpoint",
            )
            source_best_sha256 = _sha256(source_best)
            _require_claimed_hash(
                _mapping(source_artifacts.get("best_checkpoint"), "source best binding"),
                "sha256",
                source_best_sha256,
                "source-contract best checkpoint",
            )
            if not source_best.samefile(source_checkpoint):
                raise ValueError("candidate source checkpoint is not the source-contract pilot best.pt")

            source_guard_paths["source_contract"] = source_contract
            source_guard_paths["candidate_pilot_evidence"] = candidate_pilot_evidence
            source_guard_paths["training_recipe"] = training_recipe
            for prefix, payload in (
                ("full_crop_source", source_contract_payload),
                ("candidate_pilot", candidate_pilot_payload),
            ):
                bound_artifacts = _mapping(payload.get("artifacts"), f"{prefix} artifacts")
                for artifact_name, raw_binding in bound_artifacts.items():
                    binding = _mapping(raw_binding, f"{prefix} {artifact_name} binding")
                    artifact_path = _path_from(
                        binding,
                        "path",
                        base=source_contract.parent,
                        description=f"{prefix} {artifact_name}",
                    )
                    _require_claimed_hash(
                        binding,
                        "sha256",
                        _sha256(artifact_path),
                        f"{prefix} {artifact_name}",
                    )
                    source_guard_paths[f"{prefix}_{artifact_name}"] = artifact_path
            source_route_binding = {
                "mode": mode,
                "source_contract_sha256": source_contract_sha256,
                "source_subject_id": source_subject_id,
                "source_checkpoint_sha256": source_checkpoint_sha256,
                "candidate_pilot_evidence_sha256": candidate_pilot_sha256,
                "candidate_pilot_subject_id": candidate_pilot_subject_id,
                "training_recipe_sha256": training_recipe_sha256,
            }

    val = _load_json(val_summary)
    _validate_evaluation_kind(val, "validation summary")
    _require_equal(val.get("model_sha256"), actual_hashes["model"], "validation model hash")
    _require_equal(val.get("records_sha256"), actual_hashes["blind_manifest"], "validation records hash")
    _require_equal(val.get("evaluation_split"), "val", "validation split")
    providers = val.get("providers")
    if not isinstance(providers, list) or "CUDAExecutionProvider" not in providers:
        raise ValueError("validation did not use CUDAExecutionProvider")
    _validate_status_policy(val)
    _validate_acceptance(val, require_passed=True)
    val_metrics = {
        "amount": _raw_exact(val, "amount"),
        "time": _raw_exact(val, "time"),
        "payment_method_field": _raw_exact(val, "payment_method_field"),
        "recipient_field": _raw_exact(val, "recipient_field"),
        "visible_transfer_status_cjk_text": _raw_exact(
            val,
            "transfer_status",
            "ctc_raw_exact_match",
        ),
    }
    status_metrics = _mapping(
        _mapping(val.get("by_field"), "validation by_field").get("transfer_status"),
        "validation transfer_status metrics",
    )
    _require_equal(status_metrics.get("non_success_to_success"), 0, "validation unsafe status errors")
    _require_equal(
        val_binding.get("status_non_success_to_success"),
        0,
        "candidate validation unsafe status errors",
    )
    for name, floor in FIXED_FLOORS.items():
        if val_metrics[name] < floor:
            raise ValueError(f"validation {name} is below the fixed floor")
        claimed = _finite_rate(val_binding.get(name), f"candidate val_evaluation.{name}")
        if not math.isclose(claimed, val_metrics[name], rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(f"candidate val_evaluation.{name} differs from its bound summary")

    evidence_binding: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "candidate_kind": CANDIDATE_KIND,
        "checkpoint_sha256": actual_hashes["checkpoint"],
        "blind_manifest_sha256": actual_hashes["blind_manifest"],
        "blind_contract_sha256": actual_hashes["blind_contract"],
        "training_summary_sha256": actual_hashes["training_summary"],
        "val_summary_sha256": actual_hashes["val_summary"],
        "fixed_floors": floors,
        "source_route": source_route_binding,
    }
    identities = derive_gate_identity(
        model=model,
        full_manifest=full_manifest,
        contract=contract,
        labels=labels,
        evidence_binding=evidence_binding,
    )
    unique_source_guards: dict[str, dict[str, str]] = {}
    for name, path in source_guard_paths.items():
        resolved = path.resolve()
        sha256 = _sha256(resolved)
        unique_source_guards.setdefault(
            str(resolved),
            {"name": name, "path": str(resolved), "sha256": sha256},
        )
    source_guard_artifacts = sorted(
        unique_source_guards.values(), key=lambda item: (item["path"], item["name"])
    )
    source_guard_digest = _canonical_sha256({"artifacts": source_guard_artifacts})
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": INSPECTION_KIND,
        **identities,
        "candidate_evidence": str(candidate_evidence),
        "candidate_evidence_sha256": actual_hashes["candidate_evidence"],
        "paths": {
            "full_manifest": str(full_manifest),
            "blind_manifest": str(blind_manifest),
            "blind_contract": str(blind_contract),
            "model": str(model),
            "contract": str(contract),
            "labels": str(labels),
            "checkpoint": str(checkpoint),
            "training_summary": str(training_summary),
            "val_summary": str(val_summary),
        },
        "artifact_sha256": actual_hashes,
        "source_guard_artifacts": source_guard_artifacts,
        "source_guard_digest": source_guard_digest,
        "evidence_binding": evidence_binding,
        "fixed_floors": floors,
        "val_metrics": val_metrics,
        "status_non_success_to_success": 0,
    }


def verify_test_summary(*, inspection: Path, summary: Path) -> dict[str, Any]:
    expected = _load_json(inspection.resolve())
    _require_equal(expected.get("kind"), INSPECTION_KIND, "inspection kind")
    paths = _mapping(expected.get("paths"), "inspection paths")
    hashes = _mapping(expected.get("artifact_sha256"), "inspection artifact hashes")
    for name in (
        "full_manifest",
        "blind_manifest",
        "blind_contract",
        "model",
        "contract",
        "labels",
        "checkpoint",
        "training_summary",
        "val_summary",
    ):
        path = Path(str(paths.get(name))).resolve()
        if not path.is_file() or _sha256(path) != hashes.get(name):
            raise ValueError(f"{name} changed after the one-shot lock was created")
    candidate_path = Path(str(expected.get("candidate_evidence"))).resolve()
    if not candidate_path.is_file() or _sha256(candidate_path) != expected.get("candidate_evidence_sha256"):
        raise ValueError("candidate evidence changed after the one-shot lock was created")
    raw_source_guards = expected.get("source_guard_artifacts", [])
    if not isinstance(raw_source_guards, list):
        raise ValueError("inspection source_guard_artifacts must be a list")
    normalized_source_guards: list[dict[str, str]] = []
    for index, raw_guard in enumerate(raw_source_guards):
        guard = _mapping(raw_guard, f"inspection source guard {index}")
        name = guard.get("name")
        raw_path = guard.get("path")
        sha256 = guard.get("sha256")
        if not isinstance(name, str) or not name:
            raise ValueError(f"inspection source guard {index} has no name")
        if not isinstance(raw_path, str) or not raw_path:
            raise ValueError(f"inspection source guard {index} has no path")
        if not isinstance(sha256, str) or len(sha256) != 64:
            raise ValueError(f"inspection source guard {index} has no valid SHA-256")
        path = Path(raw_path).resolve()
        if not path.is_file() or _sha256(path) != sha256:
            raise ValueError(f"source guard {name} changed after the one-shot lock was created")
        normalized_source_guards.append(
            {"name": name, "path": str(path), "sha256": sha256}
        )
    if raw_source_guards or "source_guard_digest" in expected:
        _require_equal(
            expected.get("source_guard_digest"),
            _canonical_sha256({"artifacts": normalized_source_guards}),
            "inspection source_guard_digest",
        )

    # The inspection document is written after the persistent lock.  Rebuild
    # both identities from current artifact bytes so editing that intermediate
    # JSON cannot redirect post-test verification to another subject/evidence
    # tuple even if a caller bypasses the PowerShell read lease.
    rebuilt_identity = derive_gate_identity(
        model=Path(str(paths.get("model"))),
        full_manifest=Path(str(paths.get("full_manifest"))),
        contract=Path(str(paths.get("contract"))),
        labels=Path(str(paths.get("labels"))),
        evidence_binding=_mapping(expected.get("evidence_binding"), "inspection evidence binding"),
    )
    for name in (
        "model_sha256",
        "full_manifest_sha256",
        "contract_sha256",
        "labels_sha256",
        "gate_subject_id",
        "evidence_identity",
    ):
        _require_equal(expected.get(name), rebuilt_identity[name], f"inspection {name}")

    result = _load_json(summary.resolve())
    _validate_evaluation_kind(result, "test summary")
    _require_equal(result.get("model_sha256"), hashes.get("model"), "test model hash")
    _require_equal(result.get("records_sha256"), hashes.get("full_manifest"), "test records hash")
    _require_equal(result.get("evaluation_split"), "test", "test evaluation split")
    providers = result.get("providers")
    if not isinstance(providers, list) or "CUDAExecutionProvider" not in providers:
        raise ValueError("test evaluation did not use CUDAExecutionProvider")
    _validate_status_policy(result)
    reported_failures = _validate_acceptance(result, require_passed=False)

    metrics = {
        "amount": _raw_exact(result, "amount"),
        "time": _raw_exact(result, "time"),
        "payment_method_field": _raw_exact(result, "payment_method_field"),
        "recipient_field": _raw_exact(result, "recipient_field"),
        "visible_transfer_status_cjk_text": _raw_exact(
            result,
            "transfer_status",
            "ctc_raw_exact_match",
        ),
    }
    status_metrics = _mapping(
        _mapping(result.get("by_field"), "test by_field").get("transfer_status"),
        "test transfer_status metrics",
    )
    unsafe = status_metrics.get("non_success_to_success")
    if isinstance(unsafe, bool) or not isinstance(unsafe, int) or unsafe < 0:
        raise ValueError("test non_success_to_success must be a non-negative integer")
    computed_failures = [
        f"{name}_below_floor"
        for name, floor in FIXED_FLOORS.items()
        if metrics[name] < floor
    ]
    if unsafe != 0:
        computed_failures.append("status_non_success_to_success")
    passed = not computed_failures
    _require_equal(result.get("acceptance", {}).get("passed"), passed, "test acceptance.passed")
    if passed and reported_failures:
        raise ValueError("passing test summary contains acceptance failures")
    if not passed and not reported_failures:
        raise ValueError("failing test summary omits acceptance failures")
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": "receipt_recipient_v14_verified_test_summary_v1",
        "passed": passed,
        "failures": computed_failures,
        "metrics": metrics,
        "status_non_success_to_success": unsafe,
        "summary": str(summary.resolve()),
        "summary_sha256": _sha256(summary.resolve()),
        "gate_subject_id": expected.get("gate_subject_id"),
        "evidence_identity": expected.get("evidence_identity"),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Verify recipient-v14 one-shot gate evidence")
    subparsers = parser.add_subparsers(dest="command", required=True)
    inspect_parser = subparsers.add_parser("inspect")
    inspect_parser.add_argument("--candidate-evidence", type=Path, required=True)
    inspect_parser.add_argument("--trusted-full-manifest-sha256", required=True)
    verify_parser = subparsers.add_parser("verify-test")
    verify_parser.add_argument("--inspection", type=Path, required=True)
    verify_parser.add_argument("--summary", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "inspect":
        payload = inspect_candidate(
            args.candidate_evidence,
            trusted_full_manifest_sha256=args.trusted_full_manifest_sha256,
        )
    else:
        payload = verify_test_summary(inspection=args.inspection, summary=args.summary)
    print(json.dumps(payload, ensure_ascii=False, allow_nan=False, sort_keys=True))


if __name__ == "__main__":
    main()
