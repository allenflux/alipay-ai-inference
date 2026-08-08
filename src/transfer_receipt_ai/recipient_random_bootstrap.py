"""Audit a blind, random-root v12 recipient branch bootstrap.

This module is deliberately not a delivery trainer.  It binds a physical
train/validation-only manifest and every referenced crop, then verifies two
fresh training stages:

* one epoch from a completely random v12 width-1536/layers-2 root; and
* eight recipient-only epochs from that root via strict warm-start.

The resulting checkpoint may be used only as a source of ``recipient_*``
tensors for a later sanitizer.  Its randomly trained financial branches are
explicitly non-authoritative.  Delivery floors are recorded unchanged and no
ONNX or production authorization can be emitted here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import stat
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


SCHEMA_VERSION = 1
INPUT_KIND = "receipt_recipient_random_bootstrap_input_contract_v1"
DECISION_KIND = "receipt_recipient_random_bootstrap_decision_v1"
BLIND_KIND = "receipt_recipient_blind_train_val_manifest_v1"
CHECKPOINT_KIND = "receipt_unified_field_reader_v12"

DELIVERY_FLOORS = {
    "amount": 0.7885,
    "time": 0.9840,
    "payment_method_field": 0.9325,
    "recipient_field": 0.90,
}
CONTINUATION_RECIPIENT_FLOOR = 0.75
CONTINUATION_EPOCH4_TO_8_GAIN_FLOOR = 0.02
ROOT_EPOCHS = 1
PILOT_EPOCHS = 8
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

FIXED_TOPOLOGY = {
    "architecture_version": 12,
    "image_height": 80,
    "image_width": 512,
    "base_channels": 32,
    "numeric_hidden_size": 96,
    "payment_hidden_size": 128,
    "recipient_hidden_size": 256,
    "recipient_value_left_trim": 0.30,
    "recipient_input_height": 128,
    "recipient_input_width": 1536,
    "recipient_branch_channels": 24,
    "recipient_open_text_layers": 2,
    "recipient_open_text_heads": 8,
    "recipient_open_text_feedforward": 2048,
    "recipient_open_text_dropout": 0.0,
    "recipient_backbone": "legacy_depthwise_gru_v1",
    "pooled_width": 8,
    "amount_format_min_confidence": 0.80,
}
FIXED_RECIPE = {
    "device": "cuda:0",
    "random_root_seed": 424242,
    "root_epochs": ROOT_EPOCHS,
    "pilot_epochs": PILOT_EPOCHS,
    "recipient_train_splits": ["train"],
    "validation_every": 1,
    "root_initialization": "random",
    "pilot_initialization": "strict_parameter_only",
    "checkpoint_selection": "balanced_analysis_only",
    "financial_delivery_checkpoint_eligibility": False,
    "onnx_export": False,
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_load(path: Path) -> dict[str, Any]:
    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON number {value!r} is forbidden")

    try:
        value = json.loads(path.read_text(encoding="utf-8"), parse_constant=reject_constant)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"unable to read JSON evidence {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"JSON evidence must be an object: {path}")
    return value


def _training_json_load(path: Path) -> dict[str, Any]:
    """Read trainer JSON while normalizing its legacy NaN tokens to null."""

    try:
        value = json.loads(path.read_text(encoding="utf-8"), parse_constant=lambda _value: None)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"unable to read training JSON evidence {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"training JSON evidence must be an object: {path}")
    return value


def _atomic_write_json(path: Path, payload: Mapping[str, object]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite evidence: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    if temporary.exists():
        raise FileExistsError(f"refusing to reuse temporary evidence: {temporary}")
    try:
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
    except BaseException:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise


def _is_reparse(path: Path) -> bool:
    try:
        status = path.lstat()
    except FileNotFoundError:
        return False
    attributes = int(getattr(status, "st_file_attributes", 0))
    return stat.S_ISLNK(status.st_mode) or bool(attributes & 0x400)


def _require_no_reparse(path: Path, *, include_leaf: bool = True) -> None:
    """Reject symlink/junction/reparse traversal without resolving it away."""

    candidate = Path(os.path.abspath(os.fspath(path)))
    current = candidate if include_leaf else candidate.parent
    while True:
        if _is_reparse(current):
            raise ValueError(f"path traverses a symlink/junction/reparse point: {current}")
        parent = current.parent
        if parent == current:
            break
        current = parent


def _require_file(path: Path, description: str) -> Path:
    _require_no_reparse(path)
    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"missing {description}: {resolved}")
    return resolved


def _require_read_only_file(path: Path, description: str) -> Path:
    resolved = _require_file(path, description)
    if bool(resolved.stat().st_mode & stat.S_IWUSR):
        raise ValueError(f"{description} is no longer read-only: {resolved}")
    return resolved


def _require_directory(path: Path, description: str) -> Path:
    _require_no_reparse(path)
    resolved = path.resolve()
    if not resolved.is_dir():
        raise NotADirectoryError(f"missing {description}: {resolved}")
    return resolved


def _same_path(left: object, right: Path) -> bool:
    if not isinstance(left, str) or not left:
        return False
    return os.path.normcase(str(Path(left).resolve())) == os.path.normcase(str(right.resolve()))


def _require_sha(value: object, description: str) -> str:
    if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{description} must be a lowercase SHA-256")
    return value


def _copy_bound_crop(*, source: Path, target: Path, expected_sha256: str) -> None:
    if target.exists():
        if (
            not target.is_file()
            or bool(target.stat().st_mode & stat.S_IWUSR)
            or _sha256(target) != expected_sha256
        ):
            raise ValueError(f"snapshot crop collision: {target}")
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    if temporary.exists():
        raise FileExistsError(f"refusing to reuse snapshot temporary file: {temporary}")
    try:
        with source.open("rb") as reader, temporary.open("xb") as writer:
            shutil.copyfileobj(reader, writer, length=1024 * 1024)
        if _sha256(temporary) != expected_sha256:
            raise ValueError(f"snapshot crop SHA-256 mismatch: {source}")
        temporary.replace(target)
        target.chmod(0o444)
    except BaseException:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise


def _blind_crop_fingerprint(
    blind_manifest: Path,
    dataset_root: Path,
    *,
    snapshot_root: Path | None = None,
    require_read_only: bool = False,
) -> dict[str, object]:
    entries: list[tuple[str, str, str, str, int, str]] = []
    seen_ids: set[str] = set()
    split_counts = {"train": 0, "val": 0}
    field_counts: defaultdict[str, dict[str, int]] = defaultdict(lambda: {"train": 0, "val": 0})
    if snapshot_root is not None:
        snapshot_root = Path(os.path.abspath(os.fspath(snapshot_root)))
        _require_no_reparse(snapshot_root, include_leaf=False)
        if snapshot_root.exists():
            raise FileExistsError(f"refusing to reuse crop snapshot: {snapshot_root}")
        snapshot_root.mkdir(parents=True)
    with blind_manifest.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{blind_manifest}:{line_number}: invalid JSON") from error
            if not isinstance(row, Mapping):
                raise ValueError(f"{blind_manifest}:{line_number}: record must be an object")
            record_id = row.get("id")
            split = row.get("split")
            slots = row.get("slots")
            if not isinstance(record_id, str) or not record_id or record_id in seen_ids:
                raise ValueError(f"{blind_manifest}:{line_number}: invalid or duplicate record id")
            if split not in split_counts:
                raise ValueError(f"{blind_manifest}:{line_number}: test/unknown split is physically forbidden")
            if not isinstance(slots, Mapping) or not slots:
                raise ValueError(f"{blind_manifest}:{line_number}: slots must be a non-empty object")
            seen_ids.add(record_id)
            split_counts[str(split)] += 1
            for field, raw_slot in slots.items():
                if raw_slot is None:
                    continue
                if not isinstance(field, str) or not isinstance(raw_slot, Mapping):
                    raise ValueError(f"{blind_manifest}:{line_number}: invalid slot")
                image = raw_slot.get("image")
                if not isinstance(image, str) or not image:
                    raise ValueError(f"{blind_manifest}:{line_number}: {field} slot has no image")
                relative = Path(image)
                if relative.is_absolute() or any(part in {".", ".."} for part in relative.parts):
                    raise ValueError(
                        f"{blind_manifest}:{line_number}: crop path must be normalized and relative"
                    )
                crop = dataset_root / relative
                _require_no_reparse(crop)
                crop = crop.resolve()
                try:
                    crop.relative_to(dataset_root)
                except ValueError:
                    raise ValueError(f"{blind_manifest}:{line_number}: crop escapes dataset root") from None
                if not crop.is_file():
                    raise FileNotFoundError(f"missing bound crop: {crop}")
                if require_read_only and bool(crop.stat().st_mode & stat.S_IWUSR):
                    raise ValueError(f"bound crop snapshot is no longer read-only: {crop}")
                declared = _require_sha(raw_slot.get("crop_sha256"), "crop_sha256")
                observed = _sha256(crop)
                if observed != declared:
                    raise ValueError(f"crop SHA-256 mismatch: {crop}")
                if snapshot_root is not None:
                    target = (snapshot_root / relative).resolve()
                    try:
                        target.relative_to(snapshot_root)
                    except ValueError:
                        raise ValueError(
                            f"{blind_manifest}:{line_number}: snapshot crop escapes snapshot root"
                        ) from None
                    _copy_bound_crop(source=crop, target=target, expected_sha256=declared)
                entries.append(
                    (str(split), record_id, field, relative.as_posix(), crop.stat().st_size, declared)
                )
                field_counts[field][str(split)] += 1
    if split_counts["train"] <= 0 or split_counts["val"] <= 0:
        raise ValueError("blind manifest must contain non-empty train and val splits")
    for field in ("amount", "time", "payment_method_field", "recipient_field"):
        if field_counts[field]["train"] <= 0 or field_counts[field]["val"] <= 0:
            raise ValueError(f"blind manifest has incomplete train/val coverage for {field}")
    canonical = json.dumps(sorted(entries), ensure_ascii=False, separators=(",", ":"))
    return {
        "record_count": len(seen_ids),
        "split_counts": split_counts,
        "field_counts": {
            field: dict(field_counts[field]) for field in sorted(field_counts)
        },
        "crop_reference_count": len(entries),
        "crop_reference_fingerprint_sha256": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
        "all_declared_crop_hashes_verified": True,
        "bound_crop_files_read_only": bool(snapshot_root is not None or require_read_only),
    }


def _verify_blind_contract(
    *, source: Path, blind_manifest: Path, blind_contract: Path
) -> dict[str, Any]:
    contract = _json_load(blind_contract)
    expected_keys = {
        "schema_version",
        "kind",
        "source_manifest",
        "source_manifest_sha256",
        "blind_manifest",
        "blind_manifest_sha256",
        "split_counts",
        "optimizer_supervision_splits",
        "checkpoint_selection_splits",
        "final_gate_only_splits",
        "test_labels_used",
        "test_metrics_computed",
        "test_examples_emitted",
    }
    if set(contract) != expected_keys:
        raise ValueError("blind manifest contract has an unexpected schema")
    counts = contract.get("split_counts")
    if (
        contract.get("schema_version") != SCHEMA_VERSION
        or contract.get("kind") != BLIND_KIND
        or not _same_path(contract.get("source_manifest"), source)
        or not _same_path(contract.get("blind_manifest"), blind_manifest)
        or contract.get("source_manifest_sha256") != _sha256(source)
        or contract.get("blind_manifest_sha256") != _sha256(blind_manifest)
        or not isinstance(counts, Mapping)
        or set(counts) != {"train", "val", "test_excluded"}
        or any(not isinstance(counts[name], int) or counts[name] <= 0 for name in counts)
        or contract.get("optimizer_supervision_splits") != ["train"]
        or contract.get("checkpoint_selection_splits") != ["val"]
        or contract.get("final_gate_only_splits") != ["test"]
        or contract.get("test_labels_used") is not False
        or contract.get("test_metrics_computed") is not False
        or contract.get("test_examples_emitted") is not False
    ):
        raise ValueError("blind manifest contract does not prove physical train/val isolation")
    return contract


def build_input_contract(
    *,
    source_manifest: Path,
    blind_manifest: Path,
    blind_contract: Path,
    dataset_root: Path,
    snapshot_root: Path,
    output: Path,
    runner: Path,
    trainer: Path,
    blind_builder: Path,
    verifier: Path,
) -> dict[str, object]:
    """Bind immutable train/val inputs before either trainer is launched."""

    source_manifest = _require_file(source_manifest, "full v12 r3 manifest")
    blind_manifest = _require_file(blind_manifest, "blind train/val manifest")
    blind_contract = _require_file(blind_contract, "blind manifest contract")
    dataset_root = _require_directory(dataset_root, "crop dataset root")
    snapshot_root = Path(os.path.abspath(os.fspath(snapshot_root)))
    _require_no_reparse(snapshot_root, include_leaf=False)
    if snapshot_root.exists():
        raise FileExistsError(f"refusing to reuse crop snapshot: {snapshot_root}")
    output = Path(os.path.abspath(os.fspath(output)))
    _require_no_reparse(output, include_leaf=False)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite input contract: {output}")
    code_paths = {
        "runner": _require_file(runner, "PowerShell runner"),
        "trainer": _require_file(trainer, "unified trainer"),
        "blind_builder": _require_file(blind_builder, "blind manifest builder"),
        "verifier": _require_file(verifier, "bootstrap verifier"),
    }
    blind = _verify_blind_contract(
        source=source_manifest,
        blind_manifest=blind_manifest,
        blind_contract=blind_contract,
    )
    crops = _blind_crop_fingerprint(
        blind_manifest,
        dataset_root,
        snapshot_root=snapshot_root,
    )
    if crops["split_counts"] != {
        "train": int(blind["split_counts"]["train"]),
        "val": int(blind["split_counts"]["val"]),
    }:
        raise ValueError("blind contract counts do not match the materialized manifest")
    payload: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "kind": INPUT_KIND,
        "analysis_only": True,
        "branch_source_only": True,
        "production_route_authorized": False,
        "source_manifest": str(source_manifest),
        "source_manifest_sha256": _sha256(source_manifest),
        "blind_manifest": str(blind_manifest),
        "blind_manifest_sha256": _sha256(blind_manifest),
        "blind_contract": str(blind_contract),
        "blind_contract_sha256": _sha256(blind_contract),
        "source_dataset_root": str(dataset_root),
        "snapshot_dataset_root": str(snapshot_root),
        "dataset_binding": crops,
        "code_inputs": {
            name: {"path": str(path), "sha256": _sha256(path)}
            for name, path in code_paths.items()
        },
        "fixed_topology": FIXED_TOPOLOGY,
        "fixed_recipe": FIXED_RECIPE,
        "delivery_floors_unchanged": DELIVERY_FLOORS,
        "analysis_continuation_gates": {
            "minimum_best_recipient_exact": CONTINUATION_RECIPIENT_FLOOR,
            "minimum_epoch4_to_8_gain": CONTINUATION_EPOCH4_TO_8_GAIN_FLOOR,
        },
        "optimizer_supervision_splits": ["train"],
        "checkpoint_selection_splits": ["val"],
        "test_rows_physically_present_in_training_manifest": False,
        "test_labels_used_by_training": False,
        "test_metrics_computed": False,
        "blind_training_manifest_read_only": True,
        "crop_snapshot_read_only": True,
    }
    blind_manifest.chmod(0o444)
    blind_contract.chmod(0o444)
    _atomic_write_json(output, payload)
    output.chmod(0o444)
    return payload


def _finite_rate(value: object, description: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{description} must be numeric") from None
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError(f"{description} must be a finite rate")
    return result


def _candidate_metric(
    record: Mapping[str, object], field: str, *, expected_records: int
) -> float:
    by_field = record.get("val_candidate_text_by_field")
    if not isinstance(by_field, Mapping) or not isinstance(by_field.get(field), Mapping):
        raise ValueError(f"training record has no candidate metric for {field}")
    metric = by_field[field]
    assert isinstance(metric, Mapping)
    rate = _finite_rate(metric.get("exact_match"), f"{field} exact_match")
    matches = metric.get("exact_matches")
    records = metric.get("records")
    if (
        isinstance(matches, bool)
        or not isinstance(matches, int)
        or isinstance(records, bool)
        or not isinstance(records, int)
        or records != expected_records
        or not 0 <= matches <= records
        or not math.isclose(rate, matches / records, rel_tol=0.0, abs_tol=1e-12)
    ):
        raise ValueError(f"training record has inconsistent candidate metric for {field}")
    return rate


def _validated_records(summary: Mapping[str, object], *, epochs: int) -> list[Mapping[str, object]]:
    raw = summary.get("records")
    if not isinstance(raw, list) or len(raw) != epochs:
        raise ValueError(f"training summary must contain exactly {epochs} epoch records")
    records: list[Mapping[str, object]] = []
    for expected_epoch, record in enumerate(raw, start=1):
        if (
            not isinstance(record, Mapping)
            or record.get("epoch") != expected_epoch
            or record.get("validation_performed") is not True
        ):
            raise ValueError("every requested epoch must have a complete validation record")
        records.append(record)
    return records


def _assert_mapping_subset(actual: object, expected: Mapping[str, object], description: str) -> None:
    if not isinstance(actual, Mapping):
        raise ValueError(f"{description} is missing")
    mismatches = [key for key, value in expected.items() if actual.get(key) != value]
    if mismatches:
        raise ValueError(f"{description} differs in: {', '.join(sorted(mismatches))}")


def _validate_common_summary(
    summary: Mapping[str, object],
    *,
    epochs: int,
    fine_tune_mode: str,
    expected_field_counts: Mapping[str, Mapping[str, int]],
) -> list[Mapping[str, object]]:
    if summary.get("schema_version") != SCHEMA_VERSION or summary.get("kind") != CHECKPOINT_KIND:
        raise ValueError("training summary is not a v12 unified-reader artifact")
    _assert_mapping_subset(summary.get("config"), FIXED_TOPOLOGY, "training topology")
    policy = summary.get("recipient_train_split_policy")
    if not isinstance(policy, Mapping) or policy.get("mode") != "standard_train_only" or policy.get("splits") != ["train"]:
        raise ValueError("recipient supervision must be train split only")
    checkpoint_policy = summary.get("checkpoint_selection_policy")
    if (
        not isinstance(checkpoint_policy, Mapping)
        or checkpoint_policy.get("mode") != "balanced"
        or checkpoint_policy.get("protected_minimum_candidate_exact") != {}
    ):
        raise ValueError("branch-source training must use the unrelaxed balanced analysis selector")
    fine_tune = summary.get("fine_tune_policy")
    if not isinstance(fine_tune, Mapping) or fine_tune.get("mode") != fine_tune_mode:
        raise ValueError(f"unexpected fine-tune policy; expected {fine_tune_mode}")
    runtime = summary.get("training_runtime")
    if (
        not isinstance(runtime, Mapping)
        or "4090" not in str(runtime.get("cuda_device_name", ""))
        or runtime.get("cuda_tf32_requested") is not True
        or runtime.get("cudnn_benchmark_requested") is not True
        or runtime.get("validation_every") != 1
        or runtime.get("recipient_only_private_branch_training") is not (fine_tune_mode == "recipient_only_v12")
    ):
        raise ValueError("training summary does not prove the fixed CUDA:0 4090 recipe")
    fields = summary.get("field_counts")
    if not isinstance(fields, Mapping) or set(fields) != set(expected_field_counts):
        raise ValueError("training summary has no field split counts")
    for field, expected in expected_field_counts.items():
        counts = fields.get(field)
        if not isinstance(counts, Mapping) or counts != {
            "train": expected["train"],
            "val": expected["val"],
            "test": 0,
        }:
            raise ValueError(f"training summary field counts are not bound to blind data: {field}")
    return _validated_records(summary, epochs=epochs)


def build_analysis_decision(
    *,
    root_summary: Mapping[str, object],
    pilot_summary: Mapping[str, object],
    expected_field_counts: Mapping[str, Mapping[str, int]],
) -> dict[str, object]:
    """Validate summaries and make the narrow 8->16 epoch analysis decision."""

    root_records = _validate_common_summary(
        root_summary,
        epochs=ROOT_EPOCHS,
        fine_tune_mode="all_parameters",
        expected_field_counts=expected_field_counts,
    )
    root_init = root_summary.get("initialization")
    if not isinstance(root_init, Mapping) or root_init != {
        "mode": "random",
        "optimizer_restored": False,
        "epoch_reset": True,
    }:
        raise ValueError("one-epoch topology root was not initialized completely at random")
    if root_summary.get("best_checkpoint_epoch") != 1:
        raise ValueError("one-epoch random root must select epoch one")

    pilot_records = _validate_common_summary(
        pilot_summary,
        epochs=PILOT_EPOCHS,
        fine_tune_mode="recipient_only_v12",
        expected_field_counts=expected_field_counts,
    )
    fine_tune = pilot_summary["fine_tune_policy"]
    assert isinstance(fine_tune, Mapping)
    if (
        fine_tune.get("trainable_parameter_prefix") != "recipient_"
        or fine_tune.get("training_forward") != "private_recipient_branch_only_v12"
        or fine_tune.get("open_text_legacy_recipient_unfrozen") is not False
    ):
        raise ValueError("strict warm-start must train only the complete private recipient branch")

    protected_fields = ("amount", "time", "payment_method_field")
    protected_observed: dict[str, float] = {}
    for field in protected_fields:
        expected_records = expected_field_counts[field]["val"]
        root_value = _candidate_metric(root_records[0], field, expected_records=expected_records)
        values = [
            _candidate_metric(record, field, expected_records=expected_records)
            for record in pilot_records
        ]
        if any(not math.isclose(value, root_value, rel_tol=0.0, abs_tol=0.0) for value in values):
            raise ValueError(f"frozen random-root {field} metric changed during recipient-only training")
        protected_observed[field] = root_value

    recipient_by_epoch = {
        int(record["epoch"]): _candidate_metric(
            record,
            "recipient_field",
            expected_records=expected_field_counts["recipient_field"]["val"],
        )
        for record in pilot_records
    }
    best_recipient = max(recipient_by_epoch.values())
    best_epochs = [epoch for epoch, value in recipient_by_epoch.items() if value == best_recipient]
    best_checkpoint_epoch = pilot_summary.get("best_checkpoint_epoch")
    if best_checkpoint_epoch not in best_epochs:
        raise ValueError("best.pt is not an epoch with maximum strict recipient exact accuracy")
    epoch4 = recipient_by_epoch[4]
    epoch8 = recipient_by_epoch[8]
    gain = epoch8 - epoch4
    continuation_authorized = (
        best_recipient >= CONTINUATION_RECIPIENT_FLOOR
        and gain >= CONTINUATION_EPOCH4_TO_8_GAIN_FLOOR
    )
    observed = {
        **protected_observed,
        "recipient_field": best_recipient,
    }
    return {
        "analysis_only": True,
        "branch_source_only": True,
        "production_route_authorized": False,
        "onnx_delivery_authorized": False,
        "delivery_gate_evaluated": False,
        "financial_delivery_checkpoint_eligible": False,
        "delivery_floor_parameters": DELIVERY_FLOORS,
        "nonrecipient_metrics_authoritative_for_delivery": False,
        "nonrecipient_ineligibility_reason": (
            "amount/time/payment tensors originate from a one-epoch random root and must be discarded; "
            "only recipient_* tensors may enter the later sanitizer"
        ),
        "observed_analysis_metrics": observed,
        "would_meet_delivery_floor": {
            field: observed[field] >= floor for field, floor in DELIVERY_FLOORS.items()
        },
        "recipient_delivery_target_reached": best_recipient >= DELIVERY_FLOORS["recipient_field"],
        "continuation_16_epoch_authorized": continuation_authorized,
        "continuation_gates": {
            "minimum_best_recipient_exact": CONTINUATION_RECIPIENT_FLOOR,
            "minimum_epoch4_to_8_gain": CONTINUATION_EPOCH4_TO_8_GAIN_FLOOR,
        },
        "recipient_observed": {
            "best_exact": best_recipient,
            "best_epochs": best_epochs,
            "selected_best_epoch": best_checkpoint_epoch,
            "epoch4_exact": epoch4,
            "epoch8_exact": epoch8,
            "epoch4_to_8_gain": gain,
            "by_epoch": {str(epoch): value for epoch, value in recipient_by_epoch.items()},
        },
        "epoch4_evidence_authority": "analysis_continuation_only_no_checkpoint_or_delivery_authority",
    }


def _torch_load(path: Path) -> tuple[dict[str, Any], Any]:
    try:
        import torch
    except ImportError as error:
        raise RuntimeError("PyTorch is required to inspect bootstrap checkpoints") from error
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError(f"checkpoint must be an object: {path}")
    return payload, torch


def _tensor_bytes(value: Any, *, torch: Any) -> bytes:
    if not isinstance(value, torch.Tensor):
        raise ValueError("checkpoint state contains a non-tensor value")
    tensor = value.detach().cpu().contiguous().reshape(-1)
    return tensor.view(torch.uint8).numpy().tobytes()


def _partition_manifest(payload: Mapping[str, object], *, prefix: str, torch: Any) -> dict[str, object]:
    state = payload.get("state_dict")
    if not isinstance(state, Mapping):
        raise ValueError("checkpoint has no state_dict")
    entries: list[dict[str, object]] = []
    for name in sorted(state):
        if not str(name).startswith(prefix):
            continue
        value = state[name]
        raw = _tensor_bytes(value, torch=torch)
        entries.append(
            {
                "name": str(name),
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "sha256": hashlib.sha256(raw).hexdigest(),
            }
        )
    if not entries:
        raise ValueError(f"checkpoint tensor partition {prefix!r} is empty")
    canonical = json.dumps(entries, ensure_ascii=False, separators=(",", ":"))
    return {
        "tensor_count": len(entries),
        "manifest_sha256": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
    }


def _nonrecipient_manifest(payload: Mapping[str, object], *, torch: Any) -> dict[str, object]:
    state = payload.get("state_dict")
    if not isinstance(state, Mapping):
        raise ValueError("checkpoint has no state_dict")
    entries: list[dict[str, object]] = []
    for name in sorted(state):
        if str(name).startswith("recipient_"):
            continue
        value = state[name]
        raw = _tensor_bytes(value, torch=torch)
        entries.append(
            {
                "name": str(name),
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "sha256": hashlib.sha256(raw).hexdigest(),
            }
        )
    if not entries:
        raise ValueError("checkpoint nonrecipient tensor partition is empty")
    canonical = json.dumps(entries, ensure_ascii=False, separators=(",", ":"))
    return {
        "tensor_count": len(entries),
        "manifest_sha256": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
    }


def _validate_checkpoint_common(payload: Mapping[str, object]) -> None:
    if payload.get("schema_version") != SCHEMA_VERSION or payload.get("kind") != CHECKPOINT_KIND:
        raise ValueError("checkpoint is not a v12 unified-reader artifact")
    forbidden_state = {"optimizer", "optimizer_state_dict", "scheduler", "scaler"} & set(payload)
    if forbidden_state:
        raise ValueError("branch-source checkpoint unexpectedly carries resumable optimizer state")
    _assert_mapping_subset(payload.get("config"), FIXED_TOPOLOGY, "checkpoint topology")
    policy = payload.get("recipient_train_split_policy")
    if not isinstance(policy, Mapping) or policy.get("mode") != "standard_train_only" or policy.get("splits") != ["train"]:
        raise ValueError("checkpoint does not prove train-only recipient supervision")


def _assert_checkpoint_metrics_match_summary(
    payload: Mapping[str, object],
    summary_record: Mapping[str, object],
    *,
    expected_field_counts: Mapping[str, Mapping[str, int]],
    description: str,
) -> None:
    metrics = payload.get("metrics")
    if not isinstance(metrics, Mapping) or metrics.get("epoch") != summary_record.get("epoch"):
        raise ValueError(f"{description} has no matching embedded epoch metrics")
    if metrics.get("validation_performed") is not True or summary_record.get("validation_performed") is not True:
        raise ValueError(f"{description} is not backed by complete validation")
    for field in ("amount", "time", "payment_method_field", "recipient_field"):
        expected_records = expected_field_counts[field]["val"]
        embedded = _candidate_metric(metrics, field, expected_records=expected_records)
        summarized = _candidate_metric(summary_record, field, expected_records=expected_records)
        if not math.isclose(embedded, summarized, rel_tol=0.0, abs_tol=0.0):
            raise ValueError(f"{description} embedded {field} metric differs from training summary")


def _validate_output_tree(output_root: Path) -> None:
    _require_no_reparse(output_root)
    for root, directories, files in os.walk(output_root, followlinks=False):
        root_path = Path(root)
        for name in [*directories, *files]:
            child = root_path / name
            if _is_reparse(child):
                raise ValueError(f"output tree contains a symlink/junction/reparse point: {child}")
            if child.is_file() and child.suffix.lower() == ".onnx":
                raise ValueError(f"analysis-only bootstrap must not export ONNX: {child}")


def _verify_bound_inputs(contract: Mapping[str, object], contract_path: Path) -> None:
    if contract.get("schema_version") != SCHEMA_VERSION or contract.get("kind") != INPUT_KIND:
        raise ValueError("unsupported bootstrap input contract")
    for key in ("source_manifest", "blind_manifest", "blind_contract"):
        raw_path = Path(str(contract.get(key, "")))
        path = (
            _require_read_only_file(raw_path, key)
            if key in {"blind_manifest", "blind_contract"}
            else _require_file(raw_path, key)
        )
        expected = _require_sha(contract.get(key + "_sha256"), key + "_sha256")
        if _sha256(path) != expected:
            raise ValueError(f"bound input changed after preflight: {path}")
    dataset_root = _require_directory(
        Path(str(contract.get("snapshot_dataset_root", ""))), "snapshot dataset root"
    )
    observed_crops = _blind_crop_fingerprint(
        Path(str(contract["blind_manifest"])),
        dataset_root,
        require_read_only=True,
    )
    if observed_crops != contract.get("dataset_binding"):
        raise ValueError("bound train/val crop set changed during training")
    if contract.get("fixed_topology") != FIXED_TOPOLOGY:
        raise ValueError("input contract topology changed")
    if contract.get("fixed_recipe") != FIXED_RECIPE:
        raise ValueError("input contract recipe changed")
    if contract.get("delivery_floors_unchanged") != DELIVERY_FLOORS:
        raise ValueError("input contract changed a delivery floor")
    if contract.get("analysis_continuation_gates") != {
        "minimum_best_recipient_exact": CONTINUATION_RECIPIENT_FLOOR,
        "minimum_epoch4_to_8_gain": CONTINUATION_EPOCH4_TO_8_GAIN_FLOOR,
    }:
        raise ValueError("input contract changed an analysis continuation gate")
    code_inputs = contract.get("code_inputs")
    if not isinstance(code_inputs, Mapping) or set(code_inputs) != {
        "runner",
        "trainer",
        "blind_builder",
        "verifier",
    }:
        raise ValueError("input contract code bindings are missing")
    for name, raw in code_inputs.items():
        if not isinstance(raw, Mapping):
            raise ValueError(f"input contract code binding is invalid: {name}")
        path = _require_file(Path(str(raw.get("path", ""))), f"bound {name} code")
        expected = _require_sha(raw.get("sha256"), f"bound {name} code SHA-256")
        if _sha256(path) != expected:
            raise ValueError(f"bound code changed during training: {path}")
    if contract.get("production_route_authorized") is not False:
        raise ValueError("input contract cannot authorize production")
    _require_read_only_file(contract_path, "bootstrap input contract")


def finalize(
    *,
    input_contract: Path,
    root_output: Path,
    pilot_output: Path,
    output: Path,
) -> dict[str, object]:
    """Verify checkpoints/summaries and atomically publish analysis evidence."""

    input_contract = _require_file(input_contract, "bootstrap input contract")
    root_output = _require_directory(root_output, "random-root output")
    pilot_output = _require_directory(pilot_output, "strict warm-start output")
    output_root = input_contract.parent
    for stage in (root_output, pilot_output):
        try:
            stage.relative_to(output_root)
        except ValueError:
            raise ValueError(f"training output escapes the fresh bootstrap root: {stage}") from None
    _validate_output_tree(output_root)
    output = Path(os.path.abspath(os.fspath(output)))
    _require_no_reparse(output, include_leaf=False)
    try:
        output.relative_to(output_root)
    except ValueError:
        raise ValueError(f"decision output escapes the fresh bootstrap root: {output}") from None
    if output.exists():
        raise FileExistsError(f"refusing to overwrite decision: {output}")

    contract = _json_load(input_contract)
    _verify_bound_inputs(contract, input_contract)
    root_summary_path = _require_read_only_file(
        root_output / "training_summary.json", "random-root summary"
    )
    pilot_summary_path = _require_read_only_file(
        pilot_output / "training_summary.json", "strict warm-start summary"
    )
    root_summary = _training_json_load(root_summary_path)
    pilot_summary = _training_json_load(pilot_summary_path)
    if root_summary.get("config") != pilot_summary.get("config"):
        raise ValueError("strict warm-start changed the random-root model configuration")
    dataset_binding = contract.get("dataset_binding")
    if not isinstance(dataset_binding, Mapping):
        raise ValueError("input contract has no bound dataset evidence")
    expected_field_counts = dataset_binding.get("field_counts")
    if not isinstance(expected_field_counts, Mapping):
        raise ValueError("input contract has no bound per-field split counts")
    decision = build_analysis_decision(
        root_summary=root_summary,
        pilot_summary=pilot_summary,
        expected_field_counts=expected_field_counts,
    )

    root_best_path = _require_read_only_file(root_output / "best.pt", "random-root best checkpoint")
    root_last_path = _require_read_only_file(root_output / "last.pt", "random-root last checkpoint")
    pilot_best_path = _require_read_only_file(
        pilot_output / "best.pt", "strict warm-start best checkpoint"
    )
    pilot_last_path = _require_read_only_file(
        pilot_output / "last.pt", "strict warm-start last checkpoint"
    )
    root_best, torch = _torch_load(root_best_path)
    root_last, _ = _torch_load(root_last_path)
    pilot_best, _ = _torch_load(pilot_best_path)
    pilot_last, _ = _torch_load(pilot_last_path)
    for payload in (root_best, root_last, pilot_best, pilot_last):
        _validate_checkpoint_common(payload)
    expected_config = root_summary.get("config")
    if any(payload.get("config") != expected_config for payload in (root_best, root_last, pilot_best, pilot_last)):
        raise ValueError("checkpoint and summary configurations do not match exactly")
    if root_best.get("epoch") != 1 or root_last.get("epoch") != 1:
        raise ValueError("random-root checkpoints must come from epoch one")
    if root_best.get("initialization") != {
        "mode": "random",
        "optimizer_restored": False,
        "epoch_reset": True,
    }:
        raise ValueError("random-root best checkpoint has non-random ancestry")
    best_epoch = pilot_summary.get("best_checkpoint_epoch")
    if pilot_best.get("epoch") != best_epoch or pilot_last.get("epoch") != PILOT_EPOCHS:
        raise ValueError("strict warm-start checkpoint epochs do not match the summary")
    root_records = root_summary.get("records")
    pilot_records = pilot_summary.get("records")
    if not isinstance(root_records, list) or not isinstance(pilot_records, list):
        raise ValueError("training summaries have no epoch records")
    _assert_checkpoint_metrics_match_summary(
        root_best,
        root_records[0],
        expected_field_counts=expected_field_counts,
        description="random-root best checkpoint",
    )
    _assert_checkpoint_metrics_match_summary(
        root_last,
        root_records[0],
        expected_field_counts=expected_field_counts,
        description="random-root last checkpoint",
    )
    _assert_checkpoint_metrics_match_summary(
        pilot_best,
        pilot_records[int(best_epoch) - 1],
        expected_field_counts=expected_field_counts,
        description="strict warm-start best checkpoint",
    )
    _assert_checkpoint_metrics_match_summary(
        pilot_last,
        pilot_records[PILOT_EPOCHS - 1],
        expected_field_counts=expected_field_counts,
        description="strict warm-start last checkpoint",
    )
    expected_root_sha = _sha256(root_best_path)
    for name, payload in (("best", pilot_best), ("last", pilot_last)):
        initialization = payload.get("initialization")
        if (
            not isinstance(initialization, Mapping)
            or initialization.get("mode") != "parameter_only"
            or initialization.get("checkpoint_sha256") != expected_root_sha
            or initialization.get("optimizer_restored") is not False
            or initialization.get("epoch_reset") is not True
        ):
            raise ValueError(f"pilot {name} checkpoint is not a strict fresh warm-start from the random root")
    pilot_summary_init = pilot_summary.get("initialization")
    if not isinstance(pilot_summary_init, Mapping) or pilot_summary_init.get("checkpoint_sha256") != expected_root_sha:
        raise ValueError("pilot summary is not hash-bound to the random root checkpoint")

    root_nonrecipient = _nonrecipient_manifest(root_best, torch=torch)
    if _nonrecipient_manifest(root_last, torch=torch) != root_nonrecipient:
        raise ValueError("random-root best/last nonrecipient partitions disagree")
    if _nonrecipient_manifest(pilot_best, torch=torch) != root_nonrecipient:
        raise ValueError("strict warm-start best checkpoint changed nonrecipient tensors")
    if _nonrecipient_manifest(pilot_last, torch=torch) != root_nonrecipient:
        raise ValueError("strict warm-start last checkpoint changed nonrecipient tensors")
    root_recipient = _partition_manifest(root_best, prefix="recipient_", torch=torch)
    pilot_best_recipient = _partition_manifest(pilot_best, prefix="recipient_", torch=torch)
    pilot_last_recipient = _partition_manifest(pilot_last, prefix="recipient_", torch=torch)
    if pilot_last_recipient == root_recipient:
        raise ValueError("eight recipient-only epochs did not change any recipient tensor")

    label_keys = (
        "amount_characters",
        "time_characters",
        "payment_characters",
        "recipient_characters",
        "status_classes",
        "payment_bank_prefix_classes",
    )
    for key in label_keys:
        if pilot_best.get(key) != root_best.get(key) or pilot_last.get(key) != root_best.get(key):
            raise ValueError(f"strict warm-start changed semantic label map {key}")

    payload: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "kind": DECISION_KIND,
        **decision,
        "input_contract": str(input_contract),
        "input_contract_sha256": _sha256(input_contract),
        "blind_manifest_sha256": contract["blind_manifest_sha256"],
        "test_rows_physically_present_in_training_manifest": False,
        "test_labels_used_by_training": False,
        "test_metrics_computed": False,
        "random_root": {
            "output": str(root_output),
            "best_checkpoint": str(root_best_path),
            "best_checkpoint_sha256": expected_root_sha,
            "summary_sha256": _sha256(root_summary_path),
            "initialization": "random",
            "epochs": ROOT_EPOCHS,
        },
        "strict_recipient_warmstart": {
            "output": str(pilot_output),
            "best_checkpoint": str(pilot_best_path),
            "best_checkpoint_sha256": _sha256(pilot_best_path),
            "last_checkpoint": str(pilot_last_path),
            "last_checkpoint_sha256": _sha256(pilot_last_path),
            "summary_sha256": _sha256(pilot_summary_path),
            "epochs": PILOT_EPOCHS,
            "optimizer_restored": False,
            "epoch_reset": True,
            "recipient_best_tensor_manifest": pilot_best_recipient,
            "recipient_last_tensor_manifest": pilot_last_recipient,
            "nonrecipient_tensor_manifest": root_nonrecipient,
            "nonrecipient_byte_identical_to_random_root": True,
        },
        "authorized_16_epoch_warmstart_checkpoint": (
            str(pilot_best_path) if decision["continuation_16_epoch_authorized"] else None
        ),
        "notice": (
            "ANALYSIS ONLY. Even a 90% recipient result is not delivery acceptance. "
            "Discard every non-recipient tensor and run the later full v13 protected validation."
        ),
        "epoch4_evidence_limit": (
            "Epoch 4 exists only in the fresh atomic trainer summary and is used solely for the 8-to-16 "
            "analysis continuation decision. It has no checkpoint, model-selection, sanitizer, or delivery authority."
        ),
    }
    _atomic_write_json(output, payload)
    return payload


def probe_cuda() -> dict[str, object]:
    try:
        import torch
    except ImportError as error:
        raise RuntimeError("CUDA PyTorch is not installed in the fixed training environment") from error
    if not torch.cuda.is_available() or torch.cuda.device_count() <= 0:
        raise RuntimeError("CUDA device 0 is unavailable")
    name = str(torch.cuda.get_device_name(0))
    if "4090" not in name:
        raise RuntimeError(f"CUDA device 0 must be an RTX 4090; observed {name!r}")
    return {"device": "cuda:0", "name": name, "cuda_available": True}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Bind and verify the random-root recipient bootstrap")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("probe-cuda")
    bind = commands.add_parser("bind")
    bind.add_argument("--source-manifest", type=Path, required=True)
    bind.add_argument("--blind-manifest", type=Path, required=True)
    bind.add_argument("--blind-contract", type=Path, required=True)
    bind.add_argument("--dataset-root", type=Path, required=True)
    bind.add_argument("--snapshot-root", type=Path, required=True)
    bind.add_argument("--output", type=Path, required=True)
    bind.add_argument("--runner", type=Path, required=True)
    bind.add_argument("--trainer", type=Path, required=True)
    bind.add_argument("--blind-builder", type=Path, required=True)
    bind.add_argument("--verifier", type=Path, required=True)
    finish = commands.add_parser("finalize")
    finish.add_argument("--input-contract", type=Path, required=True)
    finish.add_argument("--root-output", type=Path, required=True)
    finish.add_argument("--pilot-output", type=Path, required=True)
    finish.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "probe-cuda":
        print(json.dumps(probe_cuda(), ensure_ascii=False, allow_nan=False))
        return
    if args.command == "bind":
        result = build_input_contract(
            source_manifest=args.source_manifest,
            blind_manifest=args.blind_manifest,
            blind_contract=args.blind_contract,
            dataset_root=args.dataset_root,
            snapshot_root=args.snapshot_root,
            output=args.output,
            runner=args.runner,
            trainer=args.trainer,
            blind_builder=args.blind_builder,
            verifier=args.verifier,
        )
        print(
            "recipient_random_bootstrap_inputs "
            f"records={result['dataset_binding']['record_count']} "
            f"crops={result['dataset_binding']['crop_reference_count']}"
        )
        return
    result = finalize(
        input_contract=args.input_contract,
        root_output=args.root_output,
        pilot_output=args.pilot_output,
        output=args.output,
    )
    observed = result["recipient_observed"]
    assert isinstance(observed, Mapping)
    print(
        "recipient_random_bootstrap_decision "
        f"best={float(observed['best_exact']):.2%} "
        f"epoch4_to_8_gain={float(observed['epoch4_to_8_gain']):+.2%} "
        f"continuation16={result['continuation_16_epoch_authorized']} "
        "production=False"
    )


if __name__ == "__main__":
    main()
