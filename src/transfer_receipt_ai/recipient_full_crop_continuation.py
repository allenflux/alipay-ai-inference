"""Seal and run the fixed eight-epoch legacy full-crop continuation pilot.

This route is deliberately narrower than either an ordinary strict warm start
or the separate v14 candidate route.  It reopens the passed trim-zero v13
legacy pilot, embeds a content-bound authority in an otherwise byte-equivalent
copy of that pilot's best checkpoint, resets all training state, and trains
only ``recipient_*`` parameters for exactly eight epochs.  It never opens test
labels, exports ONNX, or authorizes production.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import stat
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any
from uuid import uuid4

from .ocr_unified import (
    CHECKPOINT_SELECTION_RECIPIENT_PRIORITY,
    INIT_CHECKPOINT_MODE_RECIPIENT_FULL_CROP_CONTINUATION,
    KIND_V13,
    STATUS_CLASSES,
    UnifiedReaderConfig,
    _checkpoint_config,
    _checkpoint_labels,
    _checkpoint_status_text_characters,
    _load_checkpoint,
    _require_torch,
    _validate_recipient_full_crop_continuation_config,
    _validate_recipient_full_crop_seed_policy,
    train_unified_reader,
)
from .recipient_full_crop_pilot import (
    AMOUNT_FLOOR,
    BLIND_CONTRACT_SCHEMA_VERSION,
    BLIND_MANIFEST_KIND,
    PAYMENT_FLOOR,
    STATUS_TEXT_FLOOR,
    TIME_FLOOR,
    evaluate_pilot_summary,
    verify_blind_manifest_contract,
)
from .recipient_full_crop_seed_sanitizer import (
    ATTESTATION_KEY as SEED_SANITIZER_ATTESTATION_KEY,
    _canonical_sha256,
    _partition_descriptor,
    _require_checkpoint_without_optimizer_state,
    _state_dict,
    _tensor_signature,
)


SCHEMA_VERSION = 1
AUTHORITY_KEY = "full_crop_continuation_authority"
SOURCE_KIND = "receipt_recipient_full_crop_legacy_continuation_source_v1"
DECISION_KIND = "receipt_recipient_full_crop_legacy_continuation_decision_v1"
AUTHORIZATION = "fixed_8_epoch_legacy_trim0_continuation_pilot_only"
SOURCE_ROOT_NAME = "full-crop-pilot-8e-r2"
SOURCE_PARENT_NAME = "recipient-full-crop-analysis-20260809-r031004-06"
SOURCE_BEST_EPOCH = 6
RECIPIENT_DENOMINATOR = 6789
SOURCE_RECIPIENT_MATCHES = 5468
CONTINUATION_EPOCHS = 8
MINIMUM_BEST_MATCHES = 5790
MINIMUM_EPOCH4_TO_8_GAIN_MATCHES = 136
MAXIMUM_BEST_TO_EPOCH8_GAP_MATCHES = 67
FINAL_TARGET_MATCHES = 6111
FINAL_TARGET_GAP_FROM_SOURCE = 643
FIXED_BATCH_SIZE = 10
FIXED_LEARNING_RATE = 0.0001
FIXED_SEED = 42
FIXED_AUGMENTATION = "robust_v2"
PASS_AUTHORIZATION = "fresh_exactly_16_from_original_pilot_best_only"
FIXED_SOURCE_ARTIFACTS = {
    "source_best_checkpoint": {
        "sha256": "1f908aa2b47ab83d6ff7ac01e7653f3763383d3e4ad3431b4ddbcaf34cf653a6",
        "size_bytes": 39163899,
    },
    "source_training_summary": {
        "sha256": "c7fd3695ad228337d516738395e7a552c0f282fe0186ee54c24579ee42141e97",
        "size_bytes": 89967,
    },
    "source_pilot_decision": {
        "sha256": "b86cb9ecd06d81defff28d4178edc21d3d2ae30dfa334c1bd34ffe57cd93757d",
        "size_bytes": 3314,
    },
    "blind_manifest": {
        "sha256": "c303c8a34348532263d3ad84ed2cd6ddcd77c1bdd9dfc8a7c713ccc35a1ff5f1",
        "size_bytes": 202226294,
    },
    "blind_contract": {
        "sha256": "1167ea06f667169c72bbe572c1fe0b516389b6f7b92120d05338f0af8cf3465c",
        "size_bytes": 996,
    },
    "full_manifest": {
        "sha256": "7b7f51605c3471a4b38a85bea1bbe32212d8b3c004f248e075b17b8363f850a7",
        "size_bytes": 224911119,
    },
    "sanitized_seed": {
        "sha256": "b4e30ac514a89cb83e54cbde6d42ba007c370635785c12c1240e232e75e7c17c",
        "size_bytes": 39155451,
    },
}
FIXED_SOURCE_SUBJECT_ID = (
    "504271a800a63deb9c0e9e4c37fc4d7001932ed27393cc957bd8a955de80dbd3"
)

_HEX = frozenset("0123456789abcdef")
_FORBIDDEN_STATE_KEYS = frozenset(
    {
        "optimizer",
        "optimizer_state",
        "optimizer_state_dict",
        "scheduler",
        "scheduler_state",
        "scheduler_state_dict",
        "scaler",
        "scaler_state",
        "scaler_state_dict",
    }
)
_CHECKPOINT_SUMMARY_METADATA_KEYS = (
    "config",
    "initialization",
    "fine_tune_policy",
    "checkpoint_selection_policy",
    "recipient_train_split_policy",
    "field_counts",
    "status_text_runtime_policy",
    "training_runtime",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _mapping(value: object, description: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"continuation evidence has invalid {description}")
    return value


def _finite_rate(value: object, description: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"continuation evidence has invalid {description}")
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"continuation evidence has invalid {description}") from error
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError(f"continuation evidence has invalid {description}")
    return result


def _strict_json(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid JSON evidence: {path}") from error
    return dict(_mapping(value, str(path)))


def _strict_json_bytes(data: bytes, description: str) -> dict[str, object]:
    try:
        value = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid JSON evidence: {description}") from error
    return dict(_mapping(value, description))


def _is_reparse(path: Path) -> bool:
    info = path.stat(follow_symlinks=False)
    return stat.S_ISLNK(info.st_mode) or bool(
        int(getattr(info, "st_file_attributes", 0))
        & int(getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400))
    )


def _absolute_without_reparse(path: Path, description: str) -> Path:
    absolute = Path(os.path.abspath(os.path.expanduser(os.fspath(path))))
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current = current / part
        if os.path.lexists(current) and _is_reparse(current):
            raise ValueError(f"{description} traverses a symlink/junction/reparse point: {current}")
    return absolute


def _existing(path: Path, *, directory: bool, description: str) -> Path:
    absolute = _absolute_without_reparse(path, description)
    try:
        info = absolute.stat(follow_symlinks=False)
    except FileNotFoundError:
        raise FileNotFoundError(absolute) from None
    expected = stat.S_ISDIR(info.st_mode) if directory else stat.S_ISREG(info.st_mode)
    if not expected:
        raise ValueError(f"{description} has the wrong file type: {absolute}")
    return absolute


def _read_frozen_regular_file(
    path: Path, *, description: str
) -> tuple[bytes, tuple[int, int, int, str]]:
    """Read one inode/handle once and bind exactly the bytes that were read.

    Semantic validators consume the returned bytes, rather than reopening the
    pathname.  A later identity check therefore proves that the bytes named in
    the decision are the same bytes that were validated.
    """

    regular = _existing(path, directory=False, description=description)
    try:
        with regular.open("rb") as stream:
            before = os.fstat(stream.fileno())
            if not stat.S_ISREG(before.st_mode):
                raise ValueError(f"{description} is not a regular file")
            data = stream.read()
            after = os.fstat(stream.fileno())
        current = regular.stat(follow_symlinks=False)
    except OSError as error:
        raise ValueError(f"cannot freeze {description}: {regular}") from error
    before_key = (before.st_dev, before.st_ino, before.st_size)
    after_key = (after.st_dev, after.st_ino, after.st_size)
    current_key = (current.st_dev, current.st_ino, current.st_size)
    if (
        before_key != after_key
        or before_key != current_key
        or len(data) != before.st_size
        or not stat.S_ISREG(current.st_mode)
    ):
        raise ValueError(f"{description} changed while its bytes were frozen")
    digest = hashlib.sha256(data).hexdigest()
    return data, (before.st_dev, before.st_ino, len(data), digest)


def _fresh_file(path: Path, *, suffix: str, description: str) -> Path:
    absolute = _absolute_without_reparse(path, description)
    if absolute.suffix.lower() != suffix:
        raise ValueError(f"{description} must use the {suffix} extension")
    if os.path.lexists(absolute):
        raise ValueError(f"Refusing to overwrite {description}: {absolute}")
    _existing(absolute.parent, directory=True, description=f"{description} parent")
    return absolute


def _fresh_directory(path: Path) -> Path:
    absolute = _absolute_without_reparse(path, "continuation output")
    if os.path.lexists(absolute):
        raise ValueError(f"Refusing to reuse continuation output: {absolute}")
    _existing(absolute.parent, directory=True, description="continuation output parent")
    return absolute


def _samefile(left: Path, right: Path, description: str) -> None:
    try:
        equal = os.path.samefile(left, right)
    except OSError as error:
        raise ValueError(f"cannot verify {description} identity") from error
    if not equal:
        raise ValueError(f"{description} is not the bound source")


def _binding(path: Path) -> dict[str, object]:
    _, identity = _read_frozen_regular_file(path, description="binding source")
    return _binding_from_identity(path, identity)


def _binding_from_identity(
    path: Path, identity: tuple[int, int, int, str]
) -> dict[str, object]:
    return {
        "path": str(path),
        "sha256": identity[3],
        "size_bytes": identity[2],
    }


def _file_identity(path: Path) -> tuple[int, int, int, str]:
    _, identity = _read_frozen_regular_file(path, description="file identity")
    return identity


def _require_fixed_source_artifacts(
    paths: Mapping[str, Path], names: Sequence[str]
) -> None:
    """Pin the one real r031004-06/r2 closure before semantic parsing."""

    _capture_fixed_source_artifacts(paths, names)


def _capture_fixed_source_artifacts(
    paths: Mapping[str, Path], names: Sequence[str]
) -> dict[str, tuple[bytes, tuple[int, int, int, str]]]:
    """Freeze and hard-pin bytes that subsequent semantic checks must consume."""

    captures: dict[str, tuple[bytes, tuple[int, int, int, str]]] = {}
    for name in names:
        expected = FIXED_SOURCE_ARTIFACTS.get(name)
        path = paths.get(name)
        if expected is None or path is None:
            raise ValueError(f"fixed continuation source has no pinned {name}")
        capture = _read_frozen_regular_file(
            path, description=f"fixed continuation source artifact {name}"
        )
        observed = _binding_from_identity(path, capture[1])
        if (
            observed["sha256"] != expected["sha256"]
            or observed["size_bytes"] != expected["size_bytes"]
        ):
            raise ValueError(
                f"fixed continuation source artifact {name} does not match the real r2 digest"
            )
        captures[name] = capture
    return captures


def _verify_binding(binding: object, description: str) -> Path:
    raw = _mapping(binding, f"{description} binding")
    path_value = raw.get("path")
    digest = raw.get("sha256")
    size = raw.get("size_bytes")
    if not isinstance(path_value, str) or not path_value:
        raise ValueError(f"{description} binding has no path")
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or any(character not in _HEX for character in digest)
    ):
        raise ValueError(f"{description} binding has an invalid SHA-256")
    if isinstance(size, bool) or not isinstance(size, int) or size < 0:
        raise ValueError(f"{description} binding has an invalid size")
    path = _existing(Path(path_value), directory=False, description=description)
    _, identity = _read_frozen_regular_file(path, description=description)
    if identity[2] != size or identity[3] != digest:
        raise ValueError(f"{description} changed after sealing")
    return path


def _json_equal(actual: object, expected: object, description: str) -> None:
    if _canonical_sha256(actual, description=description) != _canonical_sha256(
        expected, description=description
    ):
        raise ValueError(f"{description} does not match its authoritative source")


def _require_checkpoint_summary_metadata(
    payload: Mapping[str, object],
    summary: Mapping[str, object],
    *,
    description: str,
) -> None:
    """Keep analysis-only lineage and all audited policy metadata on every output."""

    for key in _CHECKPOINT_SUMMARY_METADATA_KEYS:
        _json_equal(payload.get(key), summary.get(key), f"{description} {key}")


def _load_frozen_v13_checkpoint(
    data: bytes, *, torch: Any, description: str
) -> Mapping[str, object]:
    """Deserialize the exact checkpoint bytes captured for a decision binding."""

    stream = io.BytesIO(data)
    try:
        payload: Any = torch.load(stream, map_location="cpu", weights_only=True)
    except TypeError:  # PyTorch before the weights_only argument.
        stream.seek(0)
        payload = torch.load(stream, map_location="cpu")
    if not isinstance(payload, Mapping):
        raise ValueError(f"{description} must be a checkpoint mapping")
    if payload.get("schema_version") != SCHEMA_VERSION or payload.get("kind") != KIND_V13:
        raise ValueError(f"{description} is not a supported v13 checkpoint")
    return payload


def _exact_metric(raw_value: object, description: str) -> dict[str, object]:
    raw = _mapping(raw_value, description)
    records = raw.get("records")
    matches = raw.get("exact_matches")
    rate = _finite_rate(raw.get("exact_match"), f"{description} exact")
    if (
        isinstance(records, bool)
        or not isinstance(records, int)
        or records <= 0
        or isinstance(matches, bool)
        or not isinstance(matches, int)
        or not 0 <= matches <= records
        or not math.isclose(rate, matches / records, rel_tol=0.0, abs_tol=1e-12)
    ):
        raise ValueError(f"{description} count/rate evidence is inconsistent")
    return {"records": records, "exact_matches": matches, "exact_match": rate}


def _metric(record: Mapping[str, object], field: str, description: str) -> dict[str, object]:
    fields = _mapping(record.get("val_candidate_text_by_field"), f"{description} fields")
    return _exact_metric(fields.get(field), f"{description} {field}")


def _blind_recipient_validation_denominator(
    records_path: Path, *, expected_sha256: str | None = None
) -> int:
    """Rescan the bound blind manifest for labelled validation recipients."""

    records = _existing(
        records_path, directory=False, description="bound blind training manifest"
    )
    count = 0
    seen_ids: set[str] = set()
    digest = hashlib.sha256()
    try:
        with records.open("rb") as stream:
            for line_number, raw_line in enumerate(stream, start=1):
                digest.update(raw_line)
                if not raw_line.strip():
                    continue
                try:
                    line = raw_line.decode("utf-8")
                    raw = json.loads(line)
                except (UnicodeDecodeError, json.JSONDecodeError) as error:
                    raise ValueError(
                        f"blind manifest line {line_number} is invalid JSON"
                    ) from error
                row = _mapping(raw, f"blind manifest line {line_number}")
                split = row.get("split")
                if split not in {"train", "val"}:
                    raise ValueError("bound blind manifest physically contains a test row")
                record_id = row.get("id")
                if (
                    not isinstance(record_id, str)
                    or not record_id
                    or record_id in seen_ids
                ):
                    raise ValueError("bound blind manifest has a missing or duplicate record id")
                seen_ids.add(record_id)
                slots = _mapping(row.get("slots"), f"blind manifest line {line_number} slots")
                recipient = slots.get("recipient_field")
                if split == "val" and isinstance(recipient, Mapping):
                    text = recipient.get("text")
                    if not isinstance(text, str) or not text:
                        raise ValueError(
                            "bound blind manifest has an invalid labelled validation recipient"
                        )
                    count += 1
    except OSError as error:
        raise ValueError("unable to rescan the bound blind manifest") from error
    observed_sha256 = digest.hexdigest()
    if expected_sha256 is not None and observed_sha256 != expected_sha256:
        raise ValueError("bound blind manifest changed during recipient denominator scan")
    if count <= 0:
        raise ValueError("bound blind manifest has no labelled validation recipient")
    return count


def _verify_frozen_blind_manifest_contract(
    *,
    records_path: Path,
    records_data: bytes,
    contract_path: Path,
    contract_data: bytes,
) -> dict[str, object]:
    """Validate the pinned blind contract and manifest without pathname reopen."""

    contract = _strict_json_bytes(contract_data, "fixed blind manifest contract")
    blind_value = contract.get("blind_manifest")
    source_value = contract.get("source_manifest")
    expected_blind_sha256 = contract.get("blind_manifest_sha256")
    source_sha256 = contract.get("source_manifest_sha256")
    if not isinstance(blind_value, str) or not blind_value:
        raise ValueError("fixed blind manifest contract has no blind manifest")
    if not isinstance(source_value, str) or not source_value:
        raise ValueError("fixed blind manifest contract has no full manifest")
    bound_blind = _existing(
        Path(blind_value), directory=False, description="contract-bound blind manifest"
    )
    bound_source = _existing(
        Path(source_value), directory=False, description="contract-bound full manifest"
    )
    _samefile(records_path, bound_blind, "blind manifest")
    if (
        contract.get("schema_version") != BLIND_CONTRACT_SCHEMA_VERSION
        or contract.get("kind") != BLIND_MANIFEST_KIND
        or not isinstance(expected_blind_sha256, str)
        or len(expected_blind_sha256) != 64
        or hashlib.sha256(records_data).hexdigest() != expected_blind_sha256.lower()
        or not isinstance(source_sha256, str)
        or len(source_sha256) != 64
        or contract.get("test_labels_used") is not False
        or contract.get("test_metrics_computed") is not False
        or contract.get("test_examples_emitted") is not False
        or contract.get("optimizer_supervision_splits") != ["train"]
        or contract.get("checkpoint_selection_splits") != ["val"]
        or contract.get("final_gate_only_splits") != ["test"]
    ):
        raise ValueError("fixed blind manifest contract is incomplete or unsafe")
    try:
        if os.path.samefile(bound_source, records_path):
            raise ValueError("full manifest and blind manifest must be different files")
    except OSError as error:
        raise ValueError("cannot verify full/blind manifest identity") from error

    split_counts: Counter[str] = Counter()
    seen_ids: set[str] = set()
    for line_number, raw_line in enumerate(io.BytesIO(records_data), start=1):
        if not raw_line.strip():
            continue
        try:
            row = json.loads(raw_line.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError(
                f"fixed blind manifest line {line_number} is invalid JSON"
            ) from error
        record = _mapping(row, f"fixed blind manifest line {line_number}")
        split = record.get("split")
        if split not in {"train", "val"}:
            raise ValueError("fixed blind manifest physically contains a test row")
        record_id = record.get("id")
        if (
            not isinstance(record_id, str)
            or not record_id
            or record_id in seen_ids
        ):
            raise ValueError("fixed blind manifest has a missing or duplicate record id")
        seen_ids.add(record_id)
        split_counts[str(split)] += 1
    counts = _mapping(contract.get("split_counts"), "fixed blind split counts")
    try:
        expected_train = int(counts.get("train", -1))
        expected_val = int(counts.get("val", -1))
        excluded_test = int(counts.get("test_excluded", -1))
    except (TypeError, ValueError) as error:
        raise ValueError("fixed blind manifest contract has invalid split counts") from error
    if (
        split_counts != Counter({"train": expected_train, "val": expected_val})
        or expected_train <= 0
        or expected_val <= 0
        or excluded_test <= 0
    ):
        raise ValueError("fixed blind manifest split counts do not match its contract")
    return {
        "schema_version": BLIND_CONTRACT_SCHEMA_VERSION,
        "kind": BLIND_MANIFEST_KIND,
        "contract_path": str(contract_path),
        "source_manifest": str(bound_source),
        "source_manifest_sha256": source_sha256.lower(),
        "blind_manifest": str(records_path),
        "blind_manifest_sha256": expected_blind_sha256.lower(),
        "split_counts": {
            "train": expected_train,
            "val": expected_val,
            "test_excluded": excluded_test,
        },
        "optimizer_supervision_splits": ["train"],
        "checkpoint_selection_splits": ["val"],
        "test_opened_by_training": False,
    }


def _code_paths() -> dict[str, Path]:
    package = Path(__file__).resolve().parent
    repository = package.parents[1]
    return {
        "code_package_init": package / "__init__.py",
        "code_labels": package / "labels.py",
        "code_model": package / "model.py",
        "code_ocr": package / "ocr.py",
        "code_onnx_runtime": package / "onnx_runtime.py",
        "code_recipient_beam": package / "recipient_beam.py",
        "code_recipient_audit": package / "recipient_audit.py",
        "code_ocr_unified_dataset": package / "ocr_unified_dataset.py",
        "code_ocr_unified_targets": package / "ocr_unified_targets.py",
        "code_continuation": Path(__file__).resolve(),
        "code_ocr_unified": package / "ocr_unified.py",
        "code_full_crop_pilot": package / "recipient_full_crop_pilot.py",
        "code_blind_manifest": package / "recipient_blind_manifest.py",
        "code_seed_sanitizer": package / "recipient_full_crop_seed_sanitizer.py",
        "script_full_crop_pilot": repository
        / "scripts"
        / "receipt-ocr-recipient-full-crop-pilot-4090.ps1",
        "script_continuation": repository
        / "scripts"
        / "receipt-ocr-recipient-full-crop-continuation-4090.ps1",
        "script_json_normalizer": repository / "scripts" / "normalize_json_summary.py",
    }


def _pilot_paths(root: Path) -> dict[str, Path]:
    training = root / "training-full-crop-pilot"
    blind = root / "blind-train-val"
    return {
        "source_best_checkpoint": training / "best.pt",
        "source_training_summary": training / "training_summary.json",
        "source_pilot_decision": training / "pilot_decision.json",
        "blind_manifest": blind / "unified_fields.train-val.jsonl",
        "blind_contract": blind / "blind.contract.json",
    }


def _assert_analysis_tree(
    root: Path,
) -> dict[str, tuple[Path, tuple[int, int, int, str]]]:
    unsafe_true = {
        "test_evaluated",
        "test_labels_used",
        "test_metrics_computed",
        "test_examples_emitted",
        "test_opened",
        "test_opened_by_training",
        "external_test_artifacts_opened",
        "production_route_authorized",
    }

    def inspect(value: object, location: str) -> None:
        if isinstance(value, Mapping):
            if value.get("evaluation_split") == "test":
                raise ValueError(f"pilot closure contains test evaluation at {location}")
            for key, child in value.items():
                if key in unsafe_true and child is True:
                    raise ValueError(f"pilot closure contains unsafe claim {key} at {location}")
                inspect(child, f"{location}.{key}")
        elif isinstance(value, list):
            for index, child in enumerate(value):
                inspect(child, f"{location}[{index}]")

    initial_entries = sorted(root.rglob("*"), key=lambda item: item.relative_to(root).as_posix())
    captures: dict[str, tuple[Path, tuple[int, int, int, str]]] = {}
    for index, path in enumerate(initial_entries):
        if _is_reparse(path):
            raise ValueError("pilot closure contains a symlink/junction/reparse entry")
        if path.is_file() and path.suffix.lower() == ".onnx":
            raise ValueError("pilot closure contains an ONNX artifact")
        if path.is_file() and path.suffix.lower() == ".json":
            data, identity = _read_frozen_regular_file(
                path, description="pilot analysis-tree JSON"
            )
            inspect(_strict_json_bytes(data, str(path)), str(path))
            relative = path.relative_to(root).as_posix()
            name = (
                f"pilot_tree_json_{index:03d}_"
                + hashlib.sha256(relative.encode("utf-8")).hexdigest()[:12]
            )
            captures[name] = (path, identity)
    closing_entries = sorted(
        root.rglob("*"), key=lambda item: item.relative_to(root).as_posix()
    )
    if [path.relative_to(root).as_posix() for path in closing_entries] != [
        path.relative_to(root).as_posix() for path in initial_entries
    ]:
        raise ValueError("pilot analysis tree changed while it was inspected")
    for path in closing_entries:
        if _is_reparse(path) or (path.is_file() and path.suffix.lower() == ".onnx"):
            raise ValueError("pilot analysis tree became unsafe while it was inspected")
    return captures


def _checkpoint_metadata(payload: Mapping[str, object]) -> dict[str, object]:
    return {
        key: value
        for key, value in payload.items()
        if key not in {"state_dict", AUTHORITY_KEY}
    }


def _label_map_proof(payload: Mapping[str, object]) -> dict[str, object]:
    config = _checkpoint_config(payload)
    amount, time, payment, recipient, status, banks = _checkpoint_labels(
        payload, config=config
    )
    status_text = _checkpoint_status_text_characters(payload, config=config)
    values: dict[str, Sequence[str] | None] = {
        "amount_characters": amount,
        "time_characters": time,
        "payment_characters": payment,
        "payment_bank_prefix_classes": banks,
        "recipient_characters": recipient,
        "status_classes": status,
        "status_text_characters": status_text,
    }
    proof: dict[str, object] = {}
    for name, ordered in values.items():
        if ordered is None:
            raise ValueError(f"continuation source has no {name}")
        ordered_list = list(ordered)
        proof[name] = {
            "count": len(ordered_list),
            "ordered_sha256": _canonical_sha256(
                ordered_list, description=f"{name} ordered values"
            ),
        }
    return proof


def _sanitizer_transitive_source_paths(
    seed_payload: Mapping[str, object],
) -> dict[str, Path]:
    """Expose every source reopened by the sanitizer so Windows can lease it."""

    attestation = _mapping(
        seed_payload.get(SEED_SANITIZER_ATTESTATION_KEY),
        "sanitized seed attestation",
    )
    raw_descriptors: list[tuple[str, Mapping[str, object]]] = [
        (
            "sanitizer_status_checkpoint",
            _mapping(attestation.get("status_checkpoint"), "sanitizer status checkpoint"),
        ),
        (
            "sanitizer_train_checkpoint",
            _mapping(
                attestation.get("train_only_recipient_checkpoint"),
                "sanitizer train checkpoint",
            ),
        ),
    ]
    lineage = _mapping(
        attestation.get("train_only_recipient_lineage"),
        "sanitizer recipient lineage",
    )
    entries = lineage.get("entries")
    if not isinstance(entries, Sequence) or isinstance(entries, (str, bytes)) or not entries:
        raise ValueError("sanitizer recipient lineage has no checkpoint entries")
    for index, entry in enumerate(entries):
        raw_descriptors.append(
            (
                f"sanitizer_lineage_checkpoint_{index:03d}",
                _mapping(
                    _mapping(entry, f"sanitizer lineage entry {index}").get("checkpoint"),
                    f"sanitizer lineage checkpoint {index}",
                ),
            )
        )
    result: dict[str, Path] = {}
    seen: dict[str, dict[str, object]] = {}
    for name, descriptor in raw_descriptors:
        path = _verify_binding(descriptor, name)
        canonical = os.path.normcase(os.path.abspath(os.fspath(path)))
        observed = _binding(path)
        previous = seen.get(canonical)
        if previous is not None:
            if previous != observed:
                raise ValueError("sanitizer lineage repeats a path with a different identity")
            continue
        seen[canonical] = observed
        result[name] = path
    return result


def _recompute_pilot_closure(
    pilot_root: Path, *, torch: Any
) -> tuple[dict[str, object], Mapping[str, object], dict[str, Path]]:
    root = _existing(pilot_root, directory=True, description="fixed full-crop pilot root")
    if root.name != SOURCE_ROOT_NAME or root.parent.name != SOURCE_PARENT_NAME:
        raise ValueError("continuation authority is fixed to the r031004-06/full-crop-pilot-8e-r2 source")
    paths = {
        name: _existing(path, directory=False, description=name)
        for name, path in _pilot_paths(root).items()
    }
    fixed_captures = _capture_fixed_source_artifacts(paths, tuple(paths))
    identities = {name: capture[1] for name, capture in fixed_captures.items()}
    summary = _strict_json_bytes(
        fixed_captures["source_training_summary"][0], "fixed pilot training summary"
    )
    decision = _strict_json_bytes(
        fixed_captures["source_pilot_decision"][0], "fixed pilot decision"
    )
    blind = _verify_frozen_blind_manifest_contract(
        records_path=paths["blind_manifest"],
        records_data=fixed_captures["blind_manifest"][0],
        contract_path=paths["blind_contract"],
        contract_data=fixed_captures["blind_contract"][0],
    )
    # The 202 MB blind bytes have now been semantically consumed; retain only
    # their frozen identity for the final closure recheck.
    del fixed_captures["blind_manifest"]
    full_manifest = _existing(
        Path(str(blind["source_manifest"])),
        directory=False,
        description="full source manifest",
    )
    paths["full_manifest"] = full_manifest
    full_capture = _capture_fixed_source_artifacts(paths, ("full_manifest",))[
        "full_manifest"
    ]
    identities["full_manifest"] = full_capture[1]
    if full_capture[1][3] != blind.get("source_manifest_sha256"):
        raise ValueError("full source manifest changed after blind sealing")
    del full_capture

    recomputed = {**evaluate_pilot_summary(summary), "blind_manifest_contract": blind}
    _json_equal(decision, recomputed, "recomputed pilot decision")
    if (
        decision.get("passed") is not True
        or decision.get("analysis_only") is not True
        or decision.get("production_route_authorized") is not False
        or summary.get("best_checkpoint_epoch") != SOURCE_BEST_EPOCH
    ):
        raise ValueError("pilot closure is not the fixed passed epoch-6 analysis source")
    raw_records = summary.get("records")
    if not isinstance(raw_records, Sequence) or isinstance(raw_records, (str, bytes)):
        raise ValueError("pilot summary has invalid records")
    best_rows = [
        _mapping(row, "source best epoch record")
        for row in raw_records
        if isinstance(row, Mapping) and row.get("epoch") == SOURCE_BEST_EPOCH
    ]
    if len(best_rows) != 1:
        raise ValueError("pilot summary does not have exactly one source best epoch")
    source_metric = _metric(best_rows[0], "recipient_field", "source best epoch")
    if (
        source_metric["records"] != RECIPIENT_DENOMINATOR
        or source_metric["exact_matches"] != SOURCE_RECIPIENT_MATCHES
    ):
        raise ValueError("pilot source is not the fixed 5468/6789 checkpoint")

    best_payload = _load_frozen_v13_checkpoint(
        fixed_captures["source_best_checkpoint"][0],
        torch=torch,
        description="fixed pilot best checkpoint",
    )
    _require_checkpoint_without_optimizer_state(best_payload, description="pilot best checkpoint")
    config = _checkpoint_config(best_payload)
    _validate_recipient_full_crop_continuation_config(config, config)
    if best_payload.get("kind") != KIND_V13 or best_payload.get("epoch") != SOURCE_BEST_EPOCH:
        raise ValueError("pilot best checkpoint kind/epoch is not authoritative")
    _json_equal(asdict(config), summary.get("config"), "pilot best config")
    _json_equal(best_payload.get("metrics"), best_rows[0], "pilot best metrics")
    for key in (
        "initialization",
        "fine_tune_policy",
        "checkpoint_selection_policy",
        "recipient_train_split_policy",
        "field_counts",
        "status_text_runtime_policy",
        "training_runtime",
    ):
        _json_equal(best_payload.get(key), summary.get(key), f"pilot best {key}")

    initialization = _mapping(summary.get("initialization"), "pilot initialization")
    seed_path = initialization.get("checkpoint_path")
    seed_sha = initialization.get("checkpoint_sha256")
    if not isinstance(seed_path, str) or not seed_path or not isinstance(seed_sha, str):
        raise ValueError("pilot summary does not bind its sanitized seed")
    seed = _existing(Path(seed_path), directory=False, description="sanitized seed")
    paths["sanitized_seed"] = seed
    seed_capture = _capture_fixed_source_artifacts(paths, ("sanitized_seed",))[
        "sanitized_seed"
    ]
    identities["sanitized_seed"] = seed_capture[1]
    if seed_capture[1][3] != seed_sha:
        raise ValueError("sanitized seed changed after pilot training")
    # The five directly addressable anchors were pinned before any JSON/Torch
    # semantic read; full/seed were then discovered only through those pinned
    # anchors and pinned themselves.  Scan the whole analysis tree only after
    # all seven real-r2 artifacts are fixed, so an unpinned lookalike tree can
    # never influence the authority.
    tree_captures = _assert_analysis_tree(root)
    existing_by_path = {
        os.path.normcase(os.path.abspath(os.fspath(path))): name
        for name, path in paths.items()
    }
    for name, (path, identity) in tree_captures.items():
        canonical = os.path.normcase(os.path.abspath(os.fspath(path)))
        existing_name = existing_by_path.get(canonical)
        if existing_name is not None:
            if identities[existing_name] != identity:
                raise ValueError(
                    "pinned pilot JSON changed during analysis-tree inspection"
                )
            continue
        paths[name] = path
        identities[name] = identity
        existing_by_path[canonical] = name
    seed_payload = _load_frozen_v13_checkpoint(
        seed_capture[0], torch=torch, description="fixed sanitized seed"
    )
    del seed_capture
    _validate_recipient_full_crop_seed_policy(seed_payload, torch=torch)
    for name, path in _sanitizer_transitive_source_paths(seed_payload).items():
        paths[name] = path
        identities[name] = _file_identity(path)

    for name, path in _code_paths().items():
        paths[name] = _existing(path, directory=False, description=name)
        identities[name] = _file_identity(paths[name])
    closing_identities = {name: _file_identity(path) for name, path in paths.items()}
    if identities != closing_identities:
        raise ValueError("pilot closure changed while it was being recomputed")
    closure = {
        "pilot_root": str(root),
        "source_best_epoch": SOURCE_BEST_EPOCH,
        "source_recipient": source_metric,
        "blind_manifest_contract": blind,
        "recomputed_pilot_decision": recomputed,
        "artifacts": {
            name: _binding_from_identity(path, identities[name])
            for name, path in paths.items()
        },
    }
    return closure, best_payload, paths


def _authority_payload(
    closure: Mapping[str, object], best_payload: Mapping[str, object]
) -> dict[str, object]:
    state = _state_dict(best_payload, description="pilot best checkpoint")
    config = _checkpoint_config(best_payload)
    artifacts = _mapping(closure.get("artifacts"), "pilot closure artifacts")
    derived_subject_id = _canonical_sha256(
        {
            "domain": "receipt-recipient-full-crop-legacy-continuation-source-v1",
            "authorization": AUTHORIZATION,
            "source_best_epoch": SOURCE_BEST_EPOCH,
            "source_recipient_matches": SOURCE_RECIPIENT_MATCHES,
            "source_recipient_denominator": RECIPIENT_DENOMINATOR,
            "fixed_source_artifacts": FIXED_SOURCE_ARTIFACTS,
        },
        description="continuation source subject",
    )
    if derived_subject_id != FIXED_SOURCE_SUBJECT_ID:
        raise AssertionError("fixed continuation source subject constant is inconsistent")
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": SOURCE_KIND,
        "authorization": AUTHORIZATION,
        "analysis_only": True,
        "production_route_authorized": False,
        "test_opened": False,
        "onnx_exported": False,
        "source_subject_id": FIXED_SOURCE_SUBJECT_ID,
        "pilot_root": closure["pilot_root"],
        "source_best_epoch": SOURCE_BEST_EPOCH,
        "source_recipient": dict(_mapping(closure["source_recipient"], "source recipient")),
        "config_sha256": _canonical_sha256(asdict(config), description="source config"),
        "label_maps": _label_map_proof(best_payload),
        "state_proof": {
            "all_state": {
                "recipient": _partition_descriptor(state, recipient=True),
                "non_recipient": _partition_descriptor(state, recipient=False),
            },
            "copy_policy": "all_state_tensors_exact_no_optimizer_or_history_v1",
        },
        "checkpoint_metadata_sha256": _canonical_sha256(
            _checkpoint_metadata(best_payload), description="source checkpoint metadata"
        ),
        "artifacts": dict(artifacts),
        "fresh_state_policy": {
            "optimizer_restored": False,
            "scheduler_restored": False,
            "epoch_reset": True,
            "sampler_state_restored": False,
            "best_history_restored": False,
        },
        "fixed_recipe": fixed_recipe(),
    }


def fixed_recipe() -> dict[str, object]:
    return {
        "epochs": CONTINUATION_EPOCHS,
        "validation_every": 1,
        "device": "cuda:0",
        "required_gpu": "RTX 4090",
        "batch_size": FIXED_BATCH_SIZE,
        "learning_rate": FIXED_LEARNING_RATE,
        "weight_decay": 0.0001,
        "seed": FIXED_SEED,
        "recipient_train_augmentation": FIXED_AUGMENTATION,
        "recipient_low_confidence_threshold": 0.95,
        "recipient_low_confidence_loss_weight": 0.50,
        "recipient_confidence_curriculum_epochs": 10,
        "recipient_tail_rare_character_max_support": 3,
        "recipient_tail_rare_character_loss_weight": 1.5,
        "recipient_tail_long_text_min_length": 9,
        "recipient_tail_long_text_loss_weight": 1.5,
        "recipient_train_splits": ["train"],
        "recipient_only_fine_tune": True,
        "checkpoint_selection": CHECKPOINT_SELECTION_RECIPIENT_PRIORITY,
        "amount_floor": AMOUNT_FLOOR,
        "time_floor": TIME_FLOOR,
        "payment_floor": PAYMENT_FLOOR,
        "status_text_floor": STATUS_TEXT_FLOOR,
        "unsafe_status_max": 0,
        "ctc_loss_weight": 1.0,
        "structured_loss_weight": 1.0,
        "payment_bank_prefix_min_support": 3,
        "cuda_tf32": True,
        "cudnn_benchmark": True,
        "init_checkpoint_mode": INIT_CHECKPOINT_MODE_RECIPIENT_FULL_CROP_CONTINUATION,
    }


def validate_embedded_continuation_authority(
    payload: Mapping[str, object], *, torch: Any
) -> Mapping[str, object]:
    raw = _mapping(payload.get(AUTHORITY_KEY), "embedded continuation authority")
    claimed = raw.get("integrity_sha256")
    unsigned = {key: value for key, value in raw.items() if key != "integrity_sha256"}
    if not isinstance(claimed, str) or claimed != _canonical_sha256(
        unsigned, description="embedded continuation authority"
    ):
        raise ValueError("embedded continuation authority integrity hash does not match")
    if (
        raw.get("schema_version") != SCHEMA_VERSION
        or raw.get("kind") != SOURCE_KIND
        or raw.get("authorization") != AUTHORIZATION
        or raw.get("analysis_only") is not True
        or raw.get("production_route_authorized") is not False
        or raw.get("test_opened") is not False
        or raw.get("onnx_exported") is not False
        or raw.get("source_best_epoch") != SOURCE_BEST_EPOCH
    ):
        raise ValueError("embedded continuation authority policy is invalid")
    forbidden = sorted(_FORBIDDEN_STATE_KEYS.intersection(payload))
    if forbidden:
        raise ValueError("authorized continuation checkpoint contains training state")
    root_value = raw.get("pilot_root")
    if not isinstance(root_value, str) or not root_value:
        raise ValueError("embedded continuation authority has no pilot root")
    closure, best_payload, _ = _recompute_pilot_closure(Path(root_value), torch=torch)
    expected = _authority_payload(closure, best_payload)
    _json_equal(unsigned, expected, "embedded continuation authority")
    state = _state_dict(payload, description="authorized continuation checkpoint")
    proof = _mapping(raw.get("state_proof"), "continuation state proof")
    all_state = _mapping(proof.get("all_state"), "continuation all-state proof")
    if (
        _partition_descriptor(state, recipient=True) != all_state.get("recipient")
        or _partition_descriptor(state, recipient=False) != all_state.get("non_recipient")
        or _canonical_sha256(asdict(_checkpoint_config(payload)), description="checkpoint config")
        != raw.get("config_sha256")
        or _label_map_proof(payload) != raw.get("label_maps")
        or _canonical_sha256(
            _checkpoint_metadata(payload), description="authorized checkpoint metadata"
        )
        != raw.get("checkpoint_metadata_sha256")
    ):
        raise ValueError("authorized continuation checkpoint content does not match its authority")
    return raw


def _publish_checkpoint_no_clobber(
    output: Path, payload: Mapping[str, object], *, torch: Any
) -> None:
    temporary = output.with_name(f".{output.name}.{uuid4().hex}.tmp")
    try:
        torch.save(dict(payload), temporary)
        with temporary.open("rb") as stream:
            os.fsync(stream.fileno())
        os.link(temporary, output)
    except FileExistsError as error:
        raise ValueError(f"Refusing to overwrite authorized continuation checkpoint: {output}") from error
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _publish_json_no_clobber(
    output: Path, payload: Mapping[str, object], *, description: str
) -> None:
    encoded = json.dumps(
        dict(payload), ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False
    ) + "\n"
    try:
        with output.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
    except FileExistsError as error:
        raise ValueError(f"Refusing to overwrite {description}: {output}") from error


def seal_continuation_source(
    *,
    pilot_root: Path,
    output_checkpoint: Path,
    output_contract: Path,
    torch: Any | None = None,
) -> dict[str, object]:
    """Publish one no-clobber authorized copy and its independent source seal."""

    if torch is None:
        torch, _ = _require_torch()
    checkpoint = _fresh_file(output_checkpoint, suffix=".pt", description="authorized checkpoint")
    contract = _fresh_file(output_contract, suffix=".json", description="source contract")
    closure, best_payload, _ = _recompute_pilot_closure(pilot_root, torch=torch)
    unsigned_authority = _authority_payload(closure, best_payload)
    authority = {
        **unsigned_authority,
        "integrity_sha256": _canonical_sha256(
            unsigned_authority, description="continuation authority"
        ),
    }
    authorized_payload = {**dict(best_payload), AUTHORITY_KEY: authority}
    _publish_checkpoint_no_clobber(checkpoint, authorized_payload, torch=torch)
    reloaded = _load_checkpoint(checkpoint, torch=torch)
    validate_embedded_continuation_authority(reloaded, torch=torch)
    unsigned_contract = {
        "schema_version": SCHEMA_VERSION,
        "kind": SOURCE_KIND,
        "authorization": AUTHORIZATION,
        "analysis_only": True,
        "production_route_authorized": False,
        "test_opened": False,
        "onnx_exported": False,
        "source_subject_id": authority["source_subject_id"],
        "pilot_root": authority["pilot_root"],
        "authorized_checkpoint": _binding(checkpoint),
        "embedded_authority_integrity_sha256": authority["integrity_sha256"],
        "source_artifacts": authority["artifacts"],
        "fixed_recipe": fixed_recipe(),
    }
    sealed_contract = {
        **unsigned_contract,
        "integrity_sha256": _canonical_sha256(
            unsigned_contract, description="continuation source contract"
        ),
    }
    _publish_json_no_clobber(
        contract,
        sealed_contract,
        description="continuation source contract",
    )
    verify_continuation_source(
        pilot_root=pilot_root,
        contract_path=contract,
        authorized_checkpoint=checkpoint,
        torch=torch,
    )
    return sealed_contract


def verify_continuation_source(
    *,
    pilot_root: Path,
    contract_path: Path,
    authorized_checkpoint: Path,
    full_records: Path | None = None,
    torch: Any | None = None,
) -> dict[str, object]:
    if torch is None:
        torch, _ = _require_torch()
    contract_file = _existing(contract_path, directory=False, description="continuation source contract")
    contract = _strict_json(contract_file)
    claimed = contract.get("integrity_sha256")
    unsigned = {key: value for key, value in contract.items() if key != "integrity_sha256"}
    if not isinstance(claimed, str) or claimed != _canonical_sha256(
        unsigned, description="continuation source contract"
    ):
        raise ValueError("continuation source contract integrity hash does not match")
    checkpoint = _existing(
        authorized_checkpoint, directory=False, description="authorized continuation checkpoint"
    )
    bound_checkpoint = _verify_binding(
        contract.get("authorized_checkpoint"), "authorized continuation checkpoint"
    )
    _samefile(checkpoint, bound_checkpoint, "authorized continuation checkpoint")
    payload = _load_checkpoint(checkpoint, torch=torch)
    authority = validate_embedded_continuation_authority(payload, torch=torch)
    root = _existing(pilot_root, directory=True, description="fixed full-crop pilot root")
    bound_root = _existing(
        Path(str(authority["pilot_root"])), directory=True, description="bound pilot root"
    )
    _samefile(root, bound_root, "pilot root")
    if (
        contract.get("schema_version") != SCHEMA_VERSION
        or contract.get("kind") != SOURCE_KIND
        or contract.get("authorization") != AUTHORIZATION
        or contract.get("analysis_only") is not True
        or contract.get("production_route_authorized") is not False
        or contract.get("test_opened") is not False
        or contract.get("onnx_exported") is not False
        or contract.get("source_subject_id") != authority.get("source_subject_id")
        or contract.get("embedded_authority_integrity_sha256")
        != authority.get("integrity_sha256")
        or contract.get("fixed_recipe") != fixed_recipe()
        or contract.get("source_artifacts") != authority.get("artifacts")
    ):
        raise ValueError("continuation source contract policy does not match its embedded authority")
    for name, binding in _mapping(
        contract.get("source_artifacts"), "source artifacts"
    ).items():
        _verify_binding(binding, str(name))
    if full_records is not None:
        supplied = _existing(full_records, directory=False, description="supplied full manifest")
        full_binding = _mapping(contract["source_artifacts"], "source artifacts").get(
            "full_manifest"
        )
        bound = _verify_binding(full_binding, "bound full manifest")
        _samefile(supplied, bound, "full manifest")
    return contract


def evaluate_continuation_summary(
    summary: Mapping[str, object],
    *,
    expected_authority: Mapping[str, object] | None = None,
    bound_recipient_val_denominator: int | None = None,
) -> dict[str, object]:
    """Apply the fixed count-exact B8 gate to epochs zero through eight."""

    denominator = (
        RECIPIENT_DENOMINATOR
        if bound_recipient_val_denominator is None
        else bound_recipient_val_denominator
    )
    if (
        isinstance(denominator, bool)
        or not isinstance(denominator, int)
        or denominator != RECIPIENT_DENOMINATOR
    ):
        raise ValueError("bound blind manifest recipient denominator is not exactly 6789")

    config = UnifiedReaderConfig(**dict(_mapping(summary.get("config"), "config")))
    config.validate()
    _validate_recipient_full_crop_continuation_config(config, config)
    initialization = _mapping(summary.get("initialization"), "initialization")
    authority = _mapping(
        initialization.get("source_full_crop_continuation_authority"),
        "source continuation authority",
    )
    if expected_authority is not None:
        _json_equal(authority, expected_authority, "training embedded source authority")
    fine_tune = _mapping(summary.get("fine_tune_policy"), "fine-tune policy")
    runtime = _mapping(summary.get("training_runtime"), "training runtime")
    split_policy = _mapping(summary.get("recipient_train_split_policy"), "split policy")
    checkpoint_policy = _mapping(summary.get("checkpoint_selection_policy"), "checkpoint policy")
    minima = _mapping(checkpoint_policy.get("protected_minimum_candidate_exact"), "protected minima")
    augmentation = _mapping(
        summary.get("recipient_train_augmentation_policy"), "augmentation policy"
    )
    confidence = _mapping(summary.get("recipient_confidence_policy"), "confidence policy")
    tail = _mapping(summary.get("recipient_tail_loss_policy"), "tail policy")
    if (
        summary.get("kind") != KIND_V13
        or initialization.get("mode")
        != "parameter_only_recipient_full_crop_continuation_all_state_copy"
        or initialization.get("init_checkpoint_mode")
        != INIT_CHECKPOINT_MODE_RECIPIENT_FULL_CROP_CONTINUATION
        or initialization.get("optimizer_restored") is not False
        or initialization.get("scheduler_restored") is not False
        or initialization.get("sampler_state_restored") is not False
        or initialization.get("best_history_restored") is not False
        or initialization.get("source_epoch_restored") is not False
        or initialization.get("epoch_reset") is not True
        or initialization.get("all_state_key_set_exact") is not True
        or initialization.get("all_state_dtype_shape_exact") is not True
        or int(initialization.get("all_state_tensor_count_copied", 0)) <= 0
        or authority.get("kind") != SOURCE_KIND
        or authority.get("authorization") != AUTHORIZATION
        or authority.get("analysis_only") is not True
        or authority.get("production_route_authorized") is not False
        or authority.get("source_best_epoch") != SOURCE_BEST_EPOCH
        or fine_tune.get("mode") != "recipient_only_v13"
        or fine_tune.get("trainable_parameter_prefix") != "recipient_"
        or fine_tune.get("frozen_non_recipient_byte_guard")
        != "before_every_full_validation"
        or fine_tune.get("initialization_non_recipient_byte_guard")
        != "before_epoch_zero_validation"
        or int(fine_tune.get("frozen_non_recipient_state_entry_count", 0)) <= 0
        or fine_tune.get("validation_every") != 1
        or runtime.get("device") != "cuda:0"
        or runtime.get("uses_cuda") is not True
        or "4090" not in str(runtime.get("cuda_device_name", ""))
        or runtime.get("validation_every") != 1
        or split_policy.get("mode") != "standard_train_only"
        or split_policy.get("splits") != ["train"]
        or checkpoint_policy.get("mode") != CHECKPOINT_SELECTION_RECIPIENT_PRIORITY
        or not math.isclose(
            _finite_rate(minima.get("amount"), "amount floor"),
            AMOUNT_FLOOR,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        or not math.isclose(
            _finite_rate(minima.get("time"), "time floor"),
            TIME_FLOOR,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        or not math.isclose(
            _finite_rate(minima.get("payment_method_field"), "payment floor"),
            PAYMENT_FLOOR,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        or augmentation.get("mode") != FIXED_AUGMENTATION
        or augmentation.get("seed") != FIXED_SEED
        or confidence.get("low_confidence_threshold") != 0.95
        or confidence.get("low_confidence_loss_weight") != 0.50
        or confidence.get("curriculum_epochs") != 10
        or tail.get("rare_character_max_support") != 3
        or tail.get("rare_character_loss_weight") != 1.5
        or tail.get("long_text_min_length") != 9
        or tail.get("long_text_loss_weight") != 1.5
    ):
        raise ValueError("training summary does not prove the fixed continuation recipe")
    financial_policy = _mapping(
        initialization.get("financial_label_policy"), "financial label policy"
    )
    if financial_policy.get("mode") != "checkpoint_all_label_maps_recipient_full_crop_continuation_v1":
        raise ValueError("continuation summary does not prove exact source label maps")
    field_counts = _mapping(summary.get("field_counts"), "field counts")
    for field, value in field_counts.items():
        counts = _mapping(value, f"{field} counts")
        if counts.get("test") != 0:
            raise ValueError("continuation training physically included a test row")
    recipient_field_counts = _mapping(
        field_counts.get("recipient_field"), "recipient field counts"
    )
    if recipient_field_counts.get("val") != denominator:
        raise ValueError("continuation validation recipient denominator is not exactly 6789")

    raw_records = summary.get("records")
    if not isinstance(raw_records, Sequence) or isinstance(raw_records, (str, bytes)):
        raise ValueError("continuation summary has invalid epoch records")
    records = [_mapping(record, "epoch record") for record in raw_records]
    if [record.get("epoch") for record in records] != list(range(CONTINUATION_EPOCHS + 1)):
        raise ValueError("continuation requires epoch zero plus exactly epochs one through eight")
    if any(record.get("validation_performed") is not True for record in records):
        raise ValueError("continuation requires validation at every epoch")

    recipient_metrics: dict[int, dict[str, object]] = {}
    recipient_coverages: dict[int, float] = {}
    for record in records:
        epoch = int(record["epoch"])
        recipient = _metric(record, "recipient_field", f"epoch {epoch}")
        if recipient["records"] != denominator:
            raise ValueError(f"continuation epoch {epoch} recipient denominator is not 6789")
        recipient_metrics[epoch] = recipient
        recipient_coverage = int(recipient["records"]) / denominator
        if not math.isclose(
            recipient_coverage, 1.0, rel_tol=0.0, abs_tol=1e-12
        ):
            raise ValueError(
                f"continuation epoch {epoch} recipient candidate coverage is not 100%"
            )
        recipient_coverages[epoch] = recipient_coverage
        for field, floor in (
            ("amount", AMOUNT_FLOOR),
            ("time", TIME_FLOOR),
            ("payment_method_field", PAYMENT_FLOOR),
        ):
            metric = _metric(record, field, f"epoch {epoch}")
            if float(metric["exact_match"]) < floor:
                raise ValueError(f"continuation epoch {epoch} violated the {field} floor")
        raw_ctc = _mapping(record.get("val_ctc_by_field"), f"epoch {epoch} raw CTC")
        status = _exact_metric(
            raw_ctc.get("transfer_status"), f"epoch {epoch} transfer_status"
        )
        if float(status["exact_match"]) < STATUS_TEXT_FLOOR:
            raise ValueError(f"continuation epoch {epoch} violated the visible-status floor")
        unsafe = record.get("val_status_non_success_to_success")
        if isinstance(unsafe, bool) or not isinstance(unsafe, int) or unsafe != 0:
            raise ValueError(f"continuation epoch {epoch} violated status safety")
        if (
            record.get("checkpoint_selection_eligible") is not True
            or record.get("checkpoint_selection_protection_failures") != []
        ):
            raise ValueError(f"continuation epoch {epoch} was not checkpoint eligible")
    if recipient_metrics[0]["exact_matches"] != SOURCE_RECIPIENT_MATCHES:
        raise ValueError("continuation epoch-zero identity is not exactly 5468/6789")

    best_epoch = summary.get("best_checkpoint_epoch")
    if isinstance(best_epoch, bool) or not isinstance(best_epoch, int) or best_epoch not in recipient_metrics:
        raise ValueError("continuation best checkpoint epoch is invalid")
    maximum = max(int(metric["exact_matches"]) for metric in recipient_metrics.values())
    best_matches = int(recipient_metrics[best_epoch]["exact_matches"])
    if best_matches != maximum:
        raise ValueError("continuation best checkpoint is not recipient-optimal")
    epoch4_matches = int(recipient_metrics[4]["exact_matches"])
    epoch8_matches = int(recipient_metrics[8]["exact_matches"])
    gain = epoch8_matches - epoch4_matches
    tail_gap = best_matches - epoch8_matches
    failures: list[str] = []
    if best_matches < MINIMUM_BEST_MATCHES:
        failures.append("best_recipient_below_5790_of_6789")
    if gain < MINIMUM_EPOCH4_TO_8_GAIN_MATCHES:
        failures.append("epoch4_to_8_gain_below_136_matches")
    if tail_gap > MAXIMUM_BEST_TO_EPOCH8_GAP_MATCHES:
        failures.append("best_to_epoch8_decay_above_67_matches")
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": DECISION_KIND,
        "analysis_only": True,
        "production_route_authorized": False,
        "test_opened": False,
        "onnx_exported": False,
        "epochs": CONTINUATION_EPOCHS,
        "source_kind": SOURCE_KIND,
        "source_authorization": AUTHORIZATION,
        "fixed_recipe": fixed_recipe(),
        "fixed_gates": {
            "recipient_denominator": RECIPIENT_DENOMINATOR,
            "source_recipient_matches": SOURCE_RECIPIENT_MATCHES,
            "minimum_best_matches": MINIMUM_BEST_MATCHES,
            "minimum_epoch4_to_8_gain_matches": MINIMUM_EPOCH4_TO_8_GAIN_MATCHES,
            "maximum_best_to_epoch8_gap_matches": MAXIMUM_BEST_TO_EPOCH8_GAP_MATCHES,
            "final_target_matches": FINAL_TARGET_MATCHES,
            "final_target_gap_from_source": FINAL_TARGET_GAP_FROM_SOURCE,
            "amount_candidate_exact_floor": AMOUNT_FLOOR,
            "time_candidate_exact_floor": TIME_FLOOR,
            "payment_candidate_exact_floor": PAYMENT_FLOOR,
            "visible_status_raw_exact_floor": STATUS_TEXT_FLOOR,
            "status_non_success_to_success_max": 0,
            "minimum_recipient_candidate_coverage": 1.0,
        },
        "observed": {
            "best_epoch": best_epoch,
            "best_matches": best_matches,
            "epoch4_matches": epoch4_matches,
            "epoch8_matches": epoch8_matches,
            "epoch4_to_8_gain_matches": gain,
            "best_to_epoch8_gap_matches": tail_gap,
            "bound_manifest_recipient_val_denominator": denominator,
            "recipient_candidate_coverage": min(recipient_coverages.values()),
        },
        "passed": not failures,
        "failures": failures,
        "decision": (
            "analysis_only_authorize_fresh_exactly_16_from_original_pilot_best"
            if not failures
            else "analysis_only_stop_return_to_route_a"
        ),
        "pass_authorization": (
            {
                "authorization": PASS_AUTHORIZATION,
                "source": "original_pilot_best_not_b8_best",
                "source_best_epoch": SOURCE_BEST_EPOCH,
                "epochs": 16,
                "fresh_optimizer": True,
                "validation_every": 1,
                "same_recipe": True,
                "required_final_best_matches": FINAL_TARGET_MATCHES,
                "required_recipient_denominator": RECIPIENT_DENOMINATOR,
                "requires_strictly_greater_than_90_percent": True,
                "no_24_epoch_route": True,
                "no_80_epoch_route": True,
                "test_opened": False,
                "onnx_exported": False,
                "production_route_authorized": False,
            }
            if not failures
            else None
        ),
    }


def _validate_continuation_training_artifacts(
    *,
    output: Path,
    source_payload: Mapping[str, object],
    torch: Any,
) -> tuple[
    dict[str, object],
    dict[str, dict[str, object]],
    dict[str, tuple[int, int, int, str]],
]:
    """Reopen target config/maps/state before the B8 decision is written."""

    artifact_paths = {
        "training_summary": _existing(
            output / "training_summary.json",
            directory=False,
            description="continuation training summary",
        ),
        "labels": _existing(
            output / "labels.json", directory=False, description="continuation labels"
        ),
        "best_checkpoint": _existing(
            output / "best.pt", directory=False, description="continuation best checkpoint"
        ),
        "last_checkpoint": _existing(
            output / "last.pt", directory=False, description="continuation last checkpoint"
        ),
    }
    frozen = {
        name: _read_frozen_regular_file(
            path, description=f"continuation {name.replace('_', ' ')}"
        )
        for name, path in artifact_paths.items()
    }
    identities = {name: captured[1] for name, captured in frozen.items()}
    bindings = {
        name: {
            "path": str(artifact_paths[name]),
            "sha256": identity[3],
            "size_bytes": identity[2],
        }
        for name, identity in identities.items()
    }
    summary = _strict_json_bytes(
        frozen["training_summary"][0], "continuation training summary"
    )
    for path in output.rglob("*"):
        if _is_reparse(path):
            raise ValueError("continuation output contains a symlink/junction/reparse entry")
        if path.is_file() and path.suffix.lower() == ".onnx":
            raise ValueError("continuation output contains a forbidden ONNX artifact")
    source_config = _checkpoint_config(source_payload)
    source_labels = _label_map_proof(source_payload)
    source_state = _state_dict(source_payload, description="authorized continuation source")
    best_payload = _load_frozen_v13_checkpoint(
        frozen["best_checkpoint"][0],
        torch=torch,
        description="continuation best checkpoint",
    )
    _require_checkpoint_without_optimizer_state(
        best_payload, description="continuation best checkpoint"
    )
    best_config = _checkpoint_config(best_payload)
    _validate_recipient_full_crop_continuation_config(source_config, best_config)
    if _label_map_proof(best_payload) != source_labels:
        raise ValueError("continuation best checkpoint changed an ordered label map")
    best_state = _state_dict(best_payload, description="continuation best checkpoint")
    if _partition_descriptor(best_state, recipient=False) != _partition_descriptor(
        source_state, recipient=False
    ):
        raise ValueError("continuation best checkpoint changed frozen non-recipient state")
    source_recipient_keys = {
        name for name in source_state if name.startswith("recipient_")
    }
    best_recipient_keys = {
        name for name in best_state if name.startswith("recipient_")
    }
    if source_recipient_keys != best_recipient_keys or any(
        _tensor_signature(source_state[name], name=name)
        != _tensor_signature(best_state[name], name=name)
        for name in source_recipient_keys
    ):
        raise ValueError("continuation best checkpoint changed recipient state topology")
    best_epoch = summary.get("best_checkpoint_epoch")
    if best_payload.get("epoch") != best_epoch:
        raise ValueError("continuation best checkpoint epoch does not match its summary")
    raw_records = summary.get("records")
    if not isinstance(raw_records, Sequence) or isinstance(raw_records, (str, bytes)):
        raise ValueError("continuation summary has invalid records")
    best_rows = [
        row
        for row in raw_records
        if isinstance(row, Mapping) and row.get("epoch") == best_epoch
    ]
    if len(best_rows) != 1:
        raise ValueError("continuation summary has no unique best epoch record")
    _json_equal(best_payload.get("metrics"), best_rows[0], "continuation best metrics")
    _require_checkpoint_summary_metadata(
        best_payload,
        summary,
        description="continuation best",
    )

    last_payload = _load_frozen_v13_checkpoint(
        frozen["last_checkpoint"][0],
        torch=torch,
        description="continuation last checkpoint",
    )
    _require_checkpoint_without_optimizer_state(
        last_payload, description="continuation last checkpoint"
    )
    last_config = _checkpoint_config(last_payload)
    _validate_recipient_full_crop_continuation_config(source_config, last_config)
    if last_payload.get("epoch") != CONTINUATION_EPOCHS:
        raise ValueError("continuation last checkpoint is not epoch eight")
    if _label_map_proof(last_payload) != source_labels:
        raise ValueError("continuation last checkpoint changed an ordered label map")
    last_state = _state_dict(last_payload, description="continuation last checkpoint")
    if _partition_descriptor(last_state, recipient=False) != _partition_descriptor(
        source_state, recipient=False
    ):
        raise ValueError("continuation last checkpoint changed frozen non-recipient state")
    last_recipient_keys = {
        name for name in last_state if name.startswith("recipient_")
    }
    if source_recipient_keys != last_recipient_keys or any(
        _tensor_signature(source_state[name], name=name)
        != _tensor_signature(last_state[name], name=name)
        for name in source_recipient_keys
    ):
        raise ValueError("continuation last checkpoint changed recipient state topology")
    last_rows = [
        row
        for row in raw_records
        if isinstance(row, Mapping) and row.get("epoch") == CONTINUATION_EPOCHS
    ]
    if len(last_rows) != 1:
        raise ValueError("continuation summary has no unique epoch-eight record")
    _json_equal(last_payload.get("metrics"), last_rows[0], "continuation last metrics")
    _require_checkpoint_summary_metadata(
        last_payload,
        summary,
        description="continuation last",
    )

    labels_payload = _strict_json_bytes(frozen["labels"][0], "continuation labels")
    source_config = _checkpoint_config(source_payload)
    amount, time, payment, recipient, status, banks = _checkpoint_labels(
        source_payload, config=source_config
    )
    status_text = _checkpoint_status_text_characters(source_payload, config=source_config)
    expected_labels = {
        "amount_characters": list(amount),
        "time_characters": list(time),
        "payment_characters": list(payment),
        "recipient_characters": list(recipient or []),
        "status_classes": list(status),
        "status_text_characters": list(status_text or []),
        "payment_bank_prefix_classes": list(banks or []),
    }
    for name, expected in expected_labels.items():
        if labels_payload.get(name) != expected:
            raise ValueError(f"continuation labels artifact changed {name}")
    for path in output.rglob("*"):
        if _is_reparse(path):
            raise ValueError("continuation output contains a symlink/junction/reparse entry")
        if path.is_file() and path.suffix.lower() == ".onnx":
            raise ValueError("continuation output contains a forbidden ONNX artifact")
    closing_identities = {
        name: _file_identity(path) for name, path in artifact_paths.items()
    }
    if closing_identities != identities:
        raise ValueError("continuation training artifacts changed during validation")
    return summary, bindings, identities


def run_continuation(
    *,
    pilot_root: Path,
    source_contract: Path,
    authorized_checkpoint: Path,
    records_path: Path,
    blind_contract_path: Path,
    dataset_root: Path,
    output_dir: Path,
    device: str = "cuda:0",
    num_workers: int = 4,
    prefetch_factor: int = 2,
    train_progress_every: int = 250,
) -> dict[str, object]:
    if device != "cuda:0":
        raise ValueError("legacy continuation is hard-locked to cuda:0")
    if num_workers < 0 or prefetch_factor <= 0 or train_progress_every < 0:
        raise ValueError("invalid operational loader settings")
    torch, _ = _require_torch()
    try:
        cuda_available = bool(torch.cuda.is_available())
        cuda_name = str(torch.cuda.get_device_name(0)) if cuda_available else ""
    except (AttributeError, RuntimeError) as error:
        raise ValueError("legacy continuation requires CUDA device 0 on an RTX 4090") from error
    if not cuda_available or "4090" not in cuda_name:
        raise ValueError(
            f"legacy continuation requires CUDA device 0 on an RTX 4090; observed {cuda_name!r}"
        )
    contract = verify_continuation_source(
        pilot_root=pilot_root,
        contract_path=source_contract,
        authorized_checkpoint=authorized_checkpoint,
        torch=torch,
    )
    blind = verify_blind_manifest_contract(
        records_path=records_path, blind_contract_path=blind_contract_path
    )
    dataset = _existing(
        dataset_root, directory=True, description="recipient crop dataset root"
    )
    source_artifacts = _mapping(contract.get("source_artifacts"), "source artifacts")
    bound_blind = _verify_binding(source_artifacts.get("blind_manifest"), "bound blind manifest")
    bound_blind_contract = _verify_binding(
        source_artifacts.get("blind_contract"), "bound blind contract"
    )
    _samefile(_existing(records_path, directory=False, description="blind manifest"), bound_blind, "blind manifest")
    _samefile(
        _existing(blind_contract_path, directory=False, description="blind contract"),
        bound_blind_contract,
        "blind contract",
    )
    bound_recipient_val_denominator = _blind_recipient_validation_denominator(
        bound_blind,
        expected_sha256=str(blind["blind_manifest_sha256"]),
    )
    if bound_recipient_val_denominator != RECIPIENT_DENOMINATOR:
        raise ValueError("bound blind manifest recipient denominator is not exactly 6789")
    output = _fresh_directory(output_dir)
    output.mkdir(parents=False, exist_ok=False)
    if _is_reparse(output):
        raise ValueError("fresh continuation output unexpectedly became a reparse point")
    payload = _load_checkpoint(Path(authorized_checkpoint), torch=torch)
    verified_authority = validate_embedded_continuation_authority(payload, torch=torch)
    config = _checkpoint_config(payload)
    _validate_recipient_full_crop_continuation_config(config, config)
    train_unified_reader(
        records_path=Path(records_path),
        dataset_root=dataset,
        output_dir=output,
        config=config,
        device=device,
        epochs=CONTINUATION_EPOCHS,
        batch_size=FIXED_BATCH_SIZE,
        learning_rate=FIXED_LEARNING_RATE,
        weight_decay=0.0001,
        recipient_low_confidence_threshold=0.95,
        recipient_low_confidence_loss_weight=0.50,
        recipient_confidence_curriculum_epochs=10,
        recipient_tail_rare_character_max_support=3,
        recipient_tail_rare_character_loss_weight=1.5,
        recipient_tail_long_text_min_length=9,
        recipient_tail_long_text_loss_weight=1.5,
        recipient_train_augmentation=FIXED_AUGMENTATION,
        recipient_train_splits=("train",),
        recipient_only_fine_tune=True,
        validation_every=1,
        checkpoint_selection=CHECKPOINT_SELECTION_RECIPIENT_PRIORITY,
        checkpoint_min_amount_candidate_exact=AMOUNT_FLOOR,
        checkpoint_min_time_candidate_exact=TIME_FLOOR,
        checkpoint_min_payment_candidate_exact=PAYMENT_FLOOR,
        init_checkpoint=Path(authorized_checkpoint),
        init_checkpoint_mode=INIT_CHECKPOINT_MODE_RECIPIENT_FULL_CROP_CONTINUATION,
        ctc_loss_weight=1.0,
        structured_loss_weight=1.0,
        payment_bank_prefix_min_support=3,
        seed=FIXED_SEED,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        persistent_workers=num_workers > 0,
        train_progress_every=train_progress_every,
        cuda_tf32=True,
        cudnn_benchmark=True,
    )
    final_contract = verify_continuation_source(
        pilot_root=pilot_root,
        contract_path=source_contract,
        authorized_checkpoint=authorized_checkpoint,
        torch=torch,
    )
    final_blind = verify_blind_manifest_contract(
        records_path=records_path, blind_contract_path=blind_contract_path
    )
    if final_contract != contract or final_blind != blind:
        raise ValueError("continuation source or blind binding changed during training")
    summary, training_artifact_bindings, training_artifact_identities = (
        _validate_continuation_training_artifacts(
            output=output,
            source_payload=payload,
            torch=torch,
        )
    )
    decision = evaluate_continuation_summary(
        summary,
        expected_authority=verified_authority,
        bound_recipient_val_denominator=bound_recipient_val_denominator,
    )
    decision = {
        **decision,
        "source_subject_id": contract["source_subject_id"],
        "source_contract": _binding(_existing(source_contract, directory=False, description="source contract")),
        "authorized_checkpoint": _binding(
            _existing(authorized_checkpoint, directory=False, description="authorized checkpoint")
        ),
        "blind_manifest_contract": blind,
        "training_artifacts": training_artifact_bindings,
    }
    decision_path = _fresh_file(
        output / "continuation_decision.json",
        suffix=".json",
        description="continuation decision",
    )
    _publish_json_no_clobber(
        decision_path,
        decision,
        description="continuation decision",
    )
    for name, binding in training_artifact_bindings.items():
        closing_path = _verify_binding(binding, f"validated training artifact {name}")
        if _file_identity(closing_path) != training_artifact_identities[name]:
            raise ValueError(
                f"continuation training artifact {name} changed before decision closure"
            )
    if not decision["passed"]:
        raise ValueError("CONTINUATION STOP -> route A: " + "; ".join(decision["failures"]))
    return decision


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Seal or run the fixed legacy full-crop continuation")
    subparsers = parser.add_subparsers(dest="command", required=True)
    seal = subparsers.add_parser("seal")
    seal.add_argument("--pilot-root", type=Path, required=True)
    seal.add_argument("--output-checkpoint", type=Path, required=True)
    seal.add_argument("--output-contract", type=Path, required=True)
    inspect = subparsers.add_parser("inspect")
    inspect.add_argument("--pilot-root", type=Path, required=True)
    inspect.add_argument("--full-records", type=Path, required=True)
    verify = subparsers.add_parser("verify")
    verify.add_argument("--pilot-root", type=Path, required=True)
    verify.add_argument("--contract", type=Path, required=True)
    verify.add_argument("--authorized-checkpoint", type=Path, required=True)
    verify.add_argument("--full-records", type=Path)
    run = subparsers.add_parser("run")
    run.add_argument("--pilot-root", type=Path, required=True)
    run.add_argument("--source-contract", type=Path, required=True)
    run.add_argument("--authorized-checkpoint", type=Path, required=True)
    run.add_argument("--records", type=Path, required=True)
    run.add_argument("--blind-contract", type=Path, required=True)
    run.add_argument("--dataset-root", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--device", choices=("cuda:0",), default="cuda:0")
    run.add_argument("--num-workers", type=int, default=4)
    run.add_argument("--prefetch-factor", type=int, default=2)
    run.add_argument("--train-progress-every", type=int, default=250)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "seal":
        contract = seal_continuation_source(
            pilot_root=args.pilot_root,
            output_checkpoint=args.output_checkpoint,
            output_contract=args.output_contract,
        )
        print(f"CONTINUATION SOURCE SEALED: {contract['source_subject_id']}")
    elif args.command == "inspect":
        torch, _ = _require_torch()
        closure, _, _ = _recompute_pilot_closure(args.pilot_root, torch=torch)
        artifacts = _mapping(closure["artifacts"], "source artifacts")
        supplied = _existing(args.full_records, directory=False, description="supplied full manifest")
        bound = _verify_binding(artifacts.get("full_manifest"), "bound full manifest")
        _samefile(supplied, bound, "full manifest")
        print("CONTINUATION SOURCE INSPECTION PASSED: fixed 5468/6789 epoch-6 source")
    elif args.command == "verify":
        contract = verify_continuation_source(
            pilot_root=args.pilot_root,
            contract_path=args.contract,
            authorized_checkpoint=args.authorized_checkpoint,
            full_records=args.full_records,
        )
        print(f"CONTINUATION SOURCE VERIFIED: {contract['source_subject_id']}")
    else:
        decision = run_continuation(
            pilot_root=args.pilot_root,
            source_contract=args.source_contract,
            authorized_checkpoint=args.authorized_checkpoint,
            records_path=args.records,
            blind_contract_path=args.blind_contract,
            dataset_root=args.dataset_root,
            output_dir=args.output,
            device=args.device,
            num_workers=args.num_workers,
            prefetch_factor=args.prefetch_factor,
            train_progress_every=args.train_progress_every,
        )
        observed = _mapping(decision["observed"], "decision observations")
        print(
            "CONTINUATION PASS (analysis only): "
            f"best={observed['best_matches']}/{RECIPIENT_DENOMINATOR}, "
            f"e4->e8={int(observed['epoch4_to_8_gain_matches']):+d}, "
            f"best-e8={observed['best_to_epoch8_gap_matches']}"
        )


if __name__ == "__main__":
    main()
