"""Build an analysis-only v13 seed with a train-only recipient branch.

The accepted v13 status checkpoint was produced by a status-text-only run.  Its
visible-status tensors and every legacy non-status tensor are therefore useful,
but its frozen recipient branch may descend from a transductive Paddle-fit v12
checkpoint.  This module performs one deliberately narrow state transplant:

* every key not beginning with ``recipient_`` comes from the v13 status seed;
* every key beginning with ``recipient_`` comes from a compatible, train-only
  wide1536 v12 checkpoint;
* every top-level ``recipient_`` metadata item comes from that v12 checkpoint;
* all remaining model/visible-status metadata comes from the v13 status seed;
  its old initialization/runtime/metrics are retained only under an explicit
  non-operative history record, while fresh dual-source sanitizer provenance
  becomes the output's operative metadata.

The output remains a v13 checkpoint and is analysis-only.  It does not restore
an optimizer, inspect a manifest, open held-out data, export ONNX, or authorize
production.  Publication uses a same-directory hard link so an existing output
can never be overwritten, even if it appears after preflight.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import stat
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any
from uuid import uuid4

from .ocr_unified import (
    KIND_V12,
    KIND_V13,
    RECIPIENT_BLANK_INDEX,
    STATUS_TEXT_CHARSET_SOURCE,
    V12_ONNX_OUTPUT_NAMES,
    V13_ONNX_OUTPUT_NAMES,
    _checkpoint_config,
    _checkpoint_labels,
    _checkpoint_status_text_characters,
    _load_checkpoint,
    _recipient_artifact_metadata,
    _recipient_train_split_policy,
    _validate_recipient_capacity_reinit_config,
    _validate_recipient_input_width_expansion_config,
    _validate_recipient_open_text_adapter_config,
    _validated_recipient_oov_audit,
    build_unified_reader,
)


SCHEMA_VERSION = 1
ATTESTATION_KEY = "full_crop_seed_sanitizer_attestation"
ATTESTATION_KIND = "receipt_recipient_full_crop_seed_sanitizer_attestation_v1"
RECIPIENT_PREFIX = "recipient_"
STATUS_TEXT_PREFIX = "status_text_"
V12_PASSIVE_STATUS_TEXT_LOSS_WEIGHT_KEY = "status_text_loss_weight"
V12_PASSIVE_STATUS_TEXT_LOSS_WEIGHT = 1.0
V12_STATUS_SOURCE_LEGACY_NULL_ALIASES = {
    "recipient_backbone": "legacy_depthwise_gru_v1",
    "recipient_open_text_dropout": 0.0,
}
REQUIRED_RECIPIENT_INPUT_WIDTH = 1536
PUBLICATION_POLICY = "same_directory_hard_link_no_clobber_v1"
TOPOLOGY_POLICY = "v12_v13_private_recipient_prefix_partition_v1"
LINEAGE_POLICY = "hash_bound_recursive_train_only_v12_to_random_v1"
MAX_LINEAGE_DEPTH = 32
SANITIZED_INITIALIZATION_MODE = "analysis_only_full_crop_seed_sanitizer_dual_source_v1"
RECIPIENT_CLASSIFIER_KEYS = frozenset(
    {"recipient_classifier.weight", "recipient_classifier.bias"}
)

_HEX = frozenset("0123456789abcdef")
_FORBIDDEN_STATEFUL_KEYS = frozenset(
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
_FIXED_RECIPIENT_METADATA_KEYS = frozenset(
    {
        "recipient_characters",
        "recipient_blank_index",
        "recipient_charset_sha256",
        "recipient_charset_source",
        "recipient_target",
        "recipient_oov_by_split",
        "recipient_train_split_policy",
        "recipient_loss_weight",
    }
)


FileIdentity = tuple[int, int, int, str]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_sha256(value: object, *, description: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in _HEX for character in value)
    ):
        raise ValueError(f"{description} must be a lowercase SHA-256 digest")
    return value


def _canonical_value(value: object, *, description: str) -> object:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{description} contains a non-finite float")
        return value
    if isinstance(value, Mapping):
        result: dict[str, object] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError(f"{description} contains a non-string mapping key")
            result[key] = _canonical_value(item, description=f"{description}.{key}")
        return result
    if isinstance(value, (list, tuple)):
        return [
            _canonical_value(item, description=f"{description}[{index}]")
            for index, item in enumerate(value)
        ]
    raise ValueError(f"{description} contains unsupported value type {type(value).__name__}")


def _canonical_sha256(value: object, *, description: str) -> str:
    canonical = _canonical_value(value, description=description)
    encoded = json.dumps(
        canonical,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _is_reparse(path: Path) -> bool:
    info = path.stat(follow_symlinks=False)
    if stat.S_ISLNK(info.st_mode):
        return True
    attributes = int(getattr(info, "st_file_attributes", 0))
    reparse_flag = int(getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400))
    return bool(attributes & reparse_flag)


def _absolute_without_reparse(path: Path, *, description: str) -> Path:
    absolute = Path(os.path.abspath(os.path.expanduser(os.fspath(path))))
    anchor = Path(absolute.anchor)
    current = anchor
    for part in absolute.parts[1:]:
        current = current / part
        if not os.path.lexists(current):
            continue
        if _is_reparse(current):
            raise ValueError(f"{description} contains a symlink, junction, or reparse point: {current}")
    return absolute


def _existing_regular_file(path: Path, *, description: str) -> Path:
    absolute = _absolute_without_reparse(path, description=description)
    try:
        info = absolute.stat(follow_symlinks=False)
    except FileNotFoundError:
        raise FileNotFoundError(absolute) from None
    if not stat.S_ISREG(info.st_mode):
        raise ValueError(f"{description} must be a regular file: {absolute}")
    return absolute


def _fresh_output_file(path: Path) -> Path:
    absolute = _absolute_without_reparse(path, description="sanitized output path")
    if absolute.suffix.lower() != ".pt":
        raise ValueError("sanitized output checkpoint must use a .pt extension")
    if os.path.lexists(absolute):
        raise ValueError(f"Refusing to overwrite an existing sanitized output: {absolute}")
    parent = _absolute_without_reparse(absolute.parent, description="sanitized output parent")
    try:
        info = parent.stat(follow_symlinks=False)
    except FileNotFoundError:
        raise ValueError(f"sanitized output parent must already exist: {parent}") from None
    if not stat.S_ISDIR(info.st_mode):
        raise ValueError(f"sanitized output parent is not a directory: {parent}")
    return absolute


def _file_identity(path: Path) -> FileIdentity:
    info = path.stat(follow_symlinks=False)
    if not stat.S_ISREG(info.st_mode):
        raise ValueError(f"expected a regular checkpoint file: {path}")
    return info.st_dev, info.st_ino, info.st_size, _sha256(path)


def _same_file_identity(path: Path, expected: FileIdentity) -> bool:
    try:
        return _file_identity(path) == expected
    except (OSError, ValueError):
        return False


def _require_checkpoint_without_optimizer_state(payload: Mapping[str, object], *, description: str) -> None:
    forbidden = sorted(_FORBIDDEN_STATEFUL_KEYS.intersection(payload))
    if forbidden:
        raise ValueError(f"{description} contains optimizer/scheduler state: {', '.join(forbidden)}")


def _state_dict(payload: Mapping[str, object], *, description: str) -> Mapping[str, object]:
    raw = payload.get("state_dict")
    if not isinstance(raw, Mapping) or not raw:
        raise ValueError(f"{description} has no model state_dict")
    if not all(isinstance(key, str) and key for key in raw):
        raise ValueError(f"{description} state_dict keys must be non-empty strings")
    return raw


def _tensor_bytes(value: object, *, name: str) -> tuple[str, list[int], bytes]:
    if not all(hasattr(value, member) for member in ("detach", "cpu", "contiguous", "shape", "dtype")):
        raise ValueError(f"state_dict entry {name!r} is not a tensor")
    tensor = value.detach().cpu().contiguous()
    try:
        # Viewing a flat contiguous tensor as bytes works for floating and
        # integer dtypes, including bfloat16 whose direct NumPy conversion is
        # not supported by every NumPy release.
        import torch

        raw = tensor.reshape(-1).view(torch.uint8).numpy().tobytes()
    except (ImportError, RuntimeError, TypeError, ValueError) as error:
        raise ValueError(f"unable to obtain exact bytes for state_dict entry {name!r}") from error
    return str(tensor.dtype), [int(dimension) for dimension in tensor.shape], raw


def _partition_manifest(
    state: Mapping[str, object], *, recipient: bool
) -> tuple[list[dict[str, object]], int]:
    manifest: list[dict[str, object]] = []
    total_bytes = 0
    for name in sorted(state):
        if name.startswith(RECIPIENT_PREFIX) != recipient:
            continue
        dtype, shape, raw = _tensor_bytes(state[name], name=name)
        total_bytes += len(raw)
        manifest.append(
            {
                "name": name,
                "dtype": dtype,
                "shape": shape,
                "byte_length": len(raw),
                "value_sha256": hashlib.sha256(raw).hexdigest(),
            }
        )
    if not manifest:
        label = "recipient" if recipient else "non-recipient"
        raise ValueError(f"checkpoint has no {label} state tensors")
    return manifest, total_bytes


def _partition_descriptor(state: Mapping[str, object], *, recipient: bool) -> dict[str, object]:
    manifest, total_bytes = _partition_manifest(state, recipient=recipient)
    keys = [str(entry["name"]) for entry in manifest]
    return {
        "tensor_count": len(manifest),
        "total_bytes": total_bytes,
        "key_set_sha256": _canonical_sha256(keys, description="state partition key set"),
        "tensor_manifest_sha256": _canonical_sha256(
            manifest, description="state partition tensor manifest"
        ),
    }


def _tensor_signature(value: object, *, name: str) -> tuple[str, tuple[int, ...]]:
    dtype, shape, _ = _tensor_bytes(value, name=name)
    return dtype, tuple(shape)


def _validate_state_matches_declared_model(
    payload: Mapping[str, object],
    *,
    config: Any,
    state: Mapping[str, object],
    description: str,
) -> None:
    """Bind checkpoint tensor topology to its declared v12 config and labels."""

    (
        _,
        _,
        payment_characters,
        recipient_characters,
        _,
        payment_bank_prefix_classes,
    ) = _checkpoint_labels(payload, config=config)
    if recipient_characters is None or payment_bank_prefix_classes is None:
        raise ValueError(f"{description} has incomplete model label maps")
    status_text_characters = _checkpoint_status_text_characters(payload, config=config)
    model = build_unified_reader(
        payment_vocab_size=len(payment_characters) + 1,
        config=config,
        payment_bank_prefix_vocab_size=len(payment_bank_prefix_classes),
        recipient_vocab_size=len(recipient_characters) + 1,
        status_text_vocab_size=(
            len(status_text_characters) + 1
            if status_text_characters is not None
            else None
        ),
    )
    expected_state = model.state_dict()
    observed_keys = set(state)
    expected_keys = set(expected_state)
    if observed_keys != expected_keys:
        raise ValueError(
            f"{description} tensor keys do not match its declared model: "
            f"missing={sorted(expected_keys - observed_keys)}, "
            f"unexpected={sorted(observed_keys - expected_keys)}"
        )
    for name in sorted(expected_keys):
        if _tensor_signature(state[name], name=name) != _tensor_signature(
            expected_state[name], name=name
        ):
            raise ValueError(
                f"{description} tensor {name} shape/dtype does not match its declared model"
            )


def _require_recipient_charset_expansion(
    source_characters: Sequence[str],
    target_characters: Sequence[str],
    *,
    description: str,
) -> None:
    missing = sorted(set(source_characters) - set(target_characters))
    if missing:
        raise ValueError(
            f"{description} cannot discard source recipient characters: {''.join(missing)!r}"
        )


def _validate_classifier_row_transition(
    *,
    source_state: Mapping[str, object],
    target_state: Mapping[str, object],
    source_characters: Sequence[str],
    target_characters: Sequence[str],
    allow_hidden_change: bool,
    description: str,
) -> None:
    for name in sorted(RECIPIENT_CLASSIFIER_KEYS):
        source_dtype, source_shape = _tensor_signature(source_state[name], name=name)
        target_dtype, target_shape = _tensor_signature(target_state[name], name=name)
        if (
            source_dtype != target_dtype
            or not source_shape
            or not target_shape
            or source_shape[0] != len(source_characters) + 1
            or target_shape[0] != len(target_characters) + 1
            or (not allow_hidden_change and source_shape[1:] != target_shape[1:])
        ):
            raise ValueError(f"{description} has an invalid recipient classifier row transition")


def _validate_recipient_lineage_transition(
    *,
    child_payload: Mapping[str, object],
    child_config: Any,
    child_state: Mapping[str, object],
    initialization: Mapping[str, object],
    parent_payload: Mapping[str, object],
    parent_config: Any,
    parent_state: Mapping[str, object],
    description: str,
) -> None:
    """Prove that child→parent is a transition the real trainer can create."""

    child_labels = _checkpoint_labels(child_payload, config=child_config)
    parent_labels = _checkpoint_labels(parent_payload, config=parent_config)
    for index, label in (
        (0, "amount"),
        (1, "time"),
        (2, "payment"),
        (4, "status class"),
        (5, "bank"),
    ):
        if child_labels[index] != parent_labels[index]:
            raise ValueError(f"{description} changed the frozen {label} label map")
    child_characters = child_labels[3]
    parent_characters = parent_labels[3]
    if child_characters is None or parent_characters is None:
        raise ValueError(f"{description} has incomplete recipient label maps")
    child_keys = set(child_state)
    parent_keys = set(parent_state)
    mode = initialization.get("mode")
    mode_to_cli = {
        "parameter_only_recipient_unicode_expansion": "recipient_only_expansion",
        "parameter_only_recipient_input_width_expansion": "recipient_input_width_expansion",
        "parameter_only_recipient_capacity_reinit": "recipient_capacity_reinit",
        "parameter_only_recipient_open_text_adapter": "recipient_open_text_adapter",
    }
    if mode == "parameter_only":
        if initialization.get("init_checkpoint_mode") is not None:
            raise ValueError(f"{description} strict parameter-only mode has an expansion mode")
        if child_config != parent_config or child_labels != parent_labels:
            raise ValueError(f"{description} strict parameter-only config/labels are incompatible")
        if child_keys != parent_keys:
            raise ValueError(f"{description} strict parameter-only tensor keys are incompatible")
        for name in sorted(child_keys):
            if _tensor_signature(child_state[name], name=name) != _tensor_signature(
                parent_state[name], name=name
            ):
                raise ValueError(
                    f"{description} strict parameter-only tensor topology is incompatible for {name}"
                )
        return
    expected_cli_mode = mode_to_cli.get(str(mode))
    if initialization.get("init_checkpoint_mode") != expected_cli_mode:
        raise ValueError(f"{description} initialization mode does not match init_checkpoint_mode")
    _require_recipient_charset_expansion(
        parent_characters, child_characters, description=description
    )
    if mode == "parameter_only_recipient_input_width_expansion":
        _validate_recipient_input_width_expansion_config(parent_config, child_config)
    elif mode == "parameter_only_recipient_capacity_reinit":
        _validate_recipient_capacity_reinit_config(parent_config, child_config)
    elif mode == "parameter_only_recipient_open_text_adapter":
        _validate_recipient_open_text_adapter_config(parent_config, child_config)
    elif mode == "parameter_only_recipient_unicode_expansion":
        if child_config != parent_config:
            raise ValueError(f"{description} Unicode expansion changed model config")
    else:
        raise ValueError(f"{description} has unsupported lineage transition mode {mode!r}")

    if mode == "parameter_only_recipient_open_text_adapter":
        if not parent_keys.issubset(child_keys):
            raise ValueError(f"{description} open-text adapter discarded legacy tensors")
        extra = child_keys - parent_keys
        if not extra or "recipient_open_text_gate" not in extra or any(
            not name.startswith("recipient_open_text_") for name in extra
        ):
            raise ValueError(f"{description} open-text adapter tensor set is incompatible")
        comparable_keys = parent_keys
    else:
        if child_keys != parent_keys:
            raise ValueError(f"{description} transition tensor keys are incompatible")
        comparable_keys = child_keys

    for name in sorted(comparable_keys - RECIPIENT_CLASSIFIER_KEYS):
        if mode == "parameter_only_recipient_capacity_reinit" and name.startswith(
            RECIPIENT_PREFIX
        ):
            continue
        if _tensor_signature(child_state[name], name=name) != _tensor_signature(
            parent_state[name], name=name
        ):
            raise ValueError(f"{description} changed an incompatible tensor topology for {name}")
    _validate_classifier_row_transition(
        source_state=parent_state,
        target_state=child_state,
        source_characters=parent_characters,
        target_characters=child_characters,
        allow_hidden_change=mode == "parameter_only_recipient_capacity_reinit",
        description=description,
    )


def _metadata_partitions(payload: Mapping[str, object]) -> tuple[dict[str, object], dict[str, object]]:
    recipient: dict[str, object] = {}
    non_recipient: dict[str, object] = {}
    for key, value in payload.items():
        if key in {"state_dict", ATTESTATION_KEY}:
            continue
        target = recipient if key.startswith(RECIPIENT_PREFIX) else non_recipient
        target[key] = value
    return recipient, non_recipient


_STATUS_HISTORY_FIELDS = ("initialization", "training_runtime", "metrics")


def _sanitized_metadata_overrides(
    *,
    status_payload: Mapping[str, object],
    train_payload: Mapping[str, object],
    status_source: Mapping[str, object],
    train_source: Mapping[str, object],
    train_lineage: Mapping[str, object],
) -> dict[str, object]:
    preserved = {
        key: status_payload[key] for key in _STATUS_HISTORY_FIELDS if key in status_payload
    }
    return {
        "status_source_history": {
            "kind": "status_source_nonoperative_history_v1",
            "operative_recipient_claim": False,
            "preserved_keys": sorted(preserved),
            "values": preserved,
        },
        "initialization": {
            "mode": SANITIZED_INITIALIZATION_MODE,
            "analysis_only": True,
            "production_route_authorized": False,
            "status_checkpoint": dict(status_source),
            "train_only_recipient_checkpoint": dict(train_source),
            "train_only_recipient_lineage_policy": LINEAGE_POLICY,
            "train_only_recipient_lineage_sha256": _canonical_sha256(
                train_lineage, description="sanitized initialization recipient lineage"
            ),
            "optimizer_restored": False,
            "epoch_reset": True,
        },
        "training_runtime": {
            "mode": "seed_sanitization_only_v1",
            "training_performed": False,
            "optimizer_state_loaded": False,
            "external_test_artifacts_opened": False,
            "recipient_train_split_policy": dict(
                train_payload["recipient_train_split_policy"]
            ),
            "status_source_history_is_operative_recipient_claim": False,
        },
        "metrics": {
            "mode": "seed_sanitization_only_v1",
            "training_metrics_carried_forward": False,
        },
    }


def _validate_sanitized_operative_metadata(
    payload: Mapping[str, object], *, attestation: Mapping[str, object]
) -> None:
    history = _require_exact_keys(
        payload.get("status_source_history"),
        {"kind", "operative_recipient_claim", "preserved_keys", "values"},
        description="status source history",
    )
    values = history.get("values")
    if (
        history.get("kind") != "status_source_nonoperative_history_v1"
        or history.get("operative_recipient_claim") is not False
        or not isinstance(values, Mapping)
        or history.get("preserved_keys") != sorted(values)
        or any(key not in _STATUS_HISTORY_FIELDS for key in values)
    ):
        raise ValueError("status source history is not explicitly non-operative")
    initialization = _require_exact_keys(
        payload.get("initialization"),
        {
            "mode",
            "analysis_only",
            "production_route_authorized",
            "status_checkpoint",
            "train_only_recipient_checkpoint",
            "train_only_recipient_lineage_policy",
            "train_only_recipient_lineage_sha256",
            "optimizer_restored",
            "epoch_reset",
        },
        description="sanitized checkpoint initialization",
    )
    expected_lineage_sha = _canonical_sha256(
        attestation["train_only_recipient_lineage"],
        description="attested recipient lineage for initialization",
    )
    if (
        initialization.get("mode") != SANITIZED_INITIALIZATION_MODE
        or initialization.get("analysis_only") is not True
        or initialization.get("production_route_authorized") is not False
        or initialization.get("status_checkpoint") != attestation.get("status_checkpoint")
        or initialization.get("train_only_recipient_checkpoint")
        != attestation.get("train_only_recipient_checkpoint")
        or initialization.get("train_only_recipient_lineage_policy") != LINEAGE_POLICY
        or initialization.get("train_only_recipient_lineage_sha256") != expected_lineage_sha
        or initialization.get("optimizer_restored") is not False
        or initialization.get("epoch_reset") is not True
    ):
        raise ValueError("sanitized checkpoint dual-source initialization is invalid")
    runtime = _require_exact_keys(
        payload.get("training_runtime"),
        {
            "mode",
            "training_performed",
            "optimizer_state_loaded",
            "external_test_artifacts_opened",
            "recipient_train_split_policy",
            "status_source_history_is_operative_recipient_claim",
        },
        description="sanitized checkpoint runtime",
    )
    if (
        runtime.get("mode") != "seed_sanitization_only_v1"
        or runtime.get("training_performed") is not False
        or runtime.get("optimizer_state_loaded") is not False
        or runtime.get("external_test_artifacts_opened") is not False
        or runtime.get("recipient_train_split_policy")
        != _recipient_train_split_policy(["train"])
        or runtime.get("status_source_history_is_operative_recipient_claim") is not False
    ):
        raise ValueError("sanitized checkpoint runtime policy is invalid")
    metrics = _require_exact_keys(
        payload.get("metrics"),
        {"mode", "training_metrics_carried_forward"},
        description="sanitized checkpoint metrics",
    )
    if (
        metrics.get("mode") != "seed_sanitization_only_v1"
        or metrics.get("training_metrics_carried_forward") is not False
    ):
        raise ValueError("sanitized checkpoint metrics must be seed-sanitization-only")
    operative_payload = {
        key: value
        for key, value in payload.items()
        if key not in {ATTESTATION_KEY, "status_source_history"}
    }
    _validate_no_laundered_recipient_policy(operative_payload)


def _validate_train_split_policy(value: object, *, description: str, require_train_only: bool) -> None:
    if not isinstance(value, Mapping):
        raise ValueError(f"{description} recipient train split policy is missing")
    splits = value.get("splits")
    if not isinstance(splits, Sequence) or isinstance(splits, (str, bytes)):
        raise ValueError(f"{description} recipient train split policy is invalid")
    expected = _recipient_train_split_policy([str(split) for split in splits])
    if dict(value) != expected:
        raise ValueError(f"{description} recipient train split policy is invalid")
    if require_train_only and expected != _recipient_train_split_policy(["train"]):
        raise ValueError(f"{description} does not prove train-only recipient supervision")


def _iter_nested(value: object, *, path: str = "root") -> Sequence[tuple[str, object]]:
    rows: list[tuple[str, object]] = [(path, value)]
    if isinstance(value, Mapping):
        for key, item in value.items():
            rows.extend(_iter_nested(item, path=f"{path}.{key}"))
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            rows.extend(_iter_nested(item, path=f"{path}[{index}]"))
    return rows


def _validate_no_laundered_recipient_policy(payload: Mapping[str, object]) -> None:
    found = 0
    for path, value in _iter_nested(payload):
        # Warm-start provenance uses ``source_recipient_train_split_policy``.
        # Treat it as a first-class policy claim too; otherwise a child could
        # advertise a train-only top-level policy while retaining an explicit
        # transductive source-policy record under ``initialization``.
        if not path.endswith("recipient_train_split_policy"):
            continue
        found += 1
        _validate_train_split_policy(
            value,
            description=f"train-only v12 checkpoint at {path}",
            require_train_only=True,
        )
    if found == 0:
        raise ValueError("train-only v12 checkpoint has no recipient train split policy")


def _validate_recipient_metadata(
    payload: Mapping[str, object], *, config: Any, description: str, require_train_only: bool
) -> tuple[list[str], dict[str, object]]:
    labels = _checkpoint_labels(payload, config=config)
    recipient_characters = labels[3]
    if recipient_characters is None:
        raise ValueError(f"{description} has no recipient charset")
    _validated_recipient_oov_audit(
        payload.get("recipient_oov_by_split"), source=description
    )
    _validate_train_split_policy(
        payload.get("recipient_train_split_policy"),
        description=description,
        require_train_only=require_train_only,
    )
    expected_artifact = _recipient_artifact_metadata(
        config,
        recipient_sampling_policy=payload.get("recipient_sampling_policy"),
        recipient_confidence_policy=payload.get("recipient_confidence_policy"),
        recipient_tail_loss_policy=payload.get("recipient_tail_loss_policy"),
        recipient_train_augmentation_policy=payload.get("recipient_train_augmentation_policy"),
    )
    for key, expected in expected_artifact.items():
        if payload.get(key) != expected:
            raise ValueError(f"{description} recipient artifact metadata {key!r} is incompatible")
    observed_keys = {key for key in payload if key.startswith(RECIPIENT_PREFIX)}
    expected_keys = set(expected_artifact) | set(_FIXED_RECIPIENT_METADATA_KEYS)
    if observed_keys != expected_keys:
        raise ValueError(
            f"{description} has an unsupported recipient metadata key set: "
            f"missing={sorted(expected_keys - observed_keys)}, unexpected={sorted(observed_keys - expected_keys)}"
        )
    loss_weight = payload.get("recipient_loss_weight")
    try:
        normalized_loss_weight = float(loss_weight)
    except (TypeError, ValueError):
        raise ValueError(f"{description} recipient_loss_weight is invalid") from None
    if not math.isfinite(normalized_loss_weight) or normalized_loss_weight <= 0.0:
        raise ValueError(f"{description} recipient_loss_weight is invalid")
    recipient_metadata, _ = _metadata_partitions(payload)
    return recipient_characters, recipient_metadata


def _valid_epoch(payload: Mapping[str, object], *, description: str) -> int:
    epoch = payload.get("epoch")
    if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch <= 0:
        raise ValueError(f"{description} epoch must be a positive integer")
    return epoch


def _validate_v12_passive_status_text_metadata(
    payload: Mapping[str, object], *, description: str
) -> None:
    """Allow only the inert status-loss scalar emitted by every v12 checkpoint.

    The unified trainer persists ``status_text_loss_weight`` for every
    architecture even though v12 has no visible-status CTC head.  The fixed
    bootstrap recipe leaves that inert scalar at its CLI default of ``1.0``.
    Treating its name as proof of a v13 branch rejects genuine v12 artifacts;
    accepting any other ``status_text_`` metadata would weaken the intended
    v12/v13 boundary.
    """

    observed = {
        key for key in payload if isinstance(key, str) and key.startswith(STATUS_TEXT_PREFIX)
    }
    unexpected = sorted(observed - {V12_PASSIVE_STATUS_TEXT_LOSS_WEIGHT_KEY})
    if unexpected:
        raise ValueError(
            f"{description} unexpectedly contains active status-text metadata: "
            f"{', '.join(unexpected)}"
        )
    if V12_PASSIVE_STATUS_TEXT_LOSS_WEIGHT_KEY not in observed:
        raise ValueError(
            f"{description} is missing the fixed v12 passive status-text loss weight"
        )
    raw_weight = payload.get(V12_PASSIVE_STATUS_TEXT_LOSS_WEIGHT_KEY)
    if isinstance(raw_weight, bool) or not isinstance(raw_weight, (int, float)):
        raise ValueError(
            f"{description} has an invalid fixed v12 passive status-text loss weight"
        )
    weight = float(raw_weight)
    if not math.isfinite(weight) or not math.isclose(
        weight,
        V12_PASSIVE_STATUS_TEXT_LOSS_WEIGHT,
        rel_tol=0.0,
        abs_tol=0.0,
    ):
        raise ValueError(
            f"{description} has an invalid fixed v12 passive status-text loss weight"
        )


def _validated_status_v12_source_config(
    value: object, *, expected: Mapping[str, object]
) -> dict[str, object]:
    """Validate the exact v12 config recorded by a legacy v13 warm start.

    Two recipient options were historically absent or serialized as ``null``
    and are loaded by :func:`ocr_unified._config_from_mapping` as their current
    byte-compatible defaults.  Accept only those two established aliases.
    Every other missing key, every extra key, and every other value change
    remain fatal.
    """

    if not isinstance(value, Mapping):
        raise ValueError("status v13 checkpoint has no v12 source config provenance")
    source = dict(value)
    expected_values = dict(expected)
    source_keys = set(source)
    expected_keys = set(expected_values)
    allowed_alias_keys = set(V12_STATUS_SOURCE_LEGACY_NULL_ALIASES)
    missing_keys = expected_keys - source_keys
    extra_keys = source_keys - expected_keys
    if extra_keys or not missing_keys.issubset(allowed_alias_keys):
        missing = sorted(str(key) for key in missing_keys)
        extra = sorted(str(key) for key in extra_keys)
        raise ValueError(
            "status v13 source config has keys outside the two legacy aliases: "
            f"missing={missing}, extra={extra}"
        )
    normalized = dict(source)
    for key, default in V12_STATUS_SOURCE_LEGACY_NULL_ALIASES.items():
        if key not in normalized or normalized[key] is None:
            normalized[key] = default
    if normalized != expected_values:
        changed = sorted(
            key for key in expected_keys if normalized[key] != expected_values[key]
        )
        raise ValueError(
            "status v13 source config differs outside the two legacy null aliases: "
            f"{', '.join(changed)}"
        )
    return normalized


def _validate_status_only_v13(payload: Mapping[str, object]) -> tuple[Any, Mapping[str, object]]:
    if payload.get("kind") != KIND_V13:
        raise ValueError("status checkpoint must be a v13 checkpoint")
    if ATTESTATION_KEY in payload:
        raise ValueError("status checkpoint must be the original status-only artifact, not a prior sanitizer output")
    _require_checkpoint_without_optimizer_state(payload, description="status v13 checkpoint")
    config = _checkpoint_config(payload)
    if config.recipient_input_width != REQUIRED_RECIPIENT_INPUT_WIDTH:
        raise ValueError("status v13 checkpoint is not the required wide1536 configuration")
    _checkpoint_labels(payload, config=config)
    status_characters = _checkpoint_status_text_characters(payload, config=config)
    if not status_characters:
        raise ValueError("status v13 checkpoint has no visible-status charset")
    state = _state_dict(payload, description="status v13 checkpoint")
    status_keys = {key for key in state if key.startswith(STATUS_TEXT_PREFIX)}
    if not status_keys:
        raise ValueError("status v13 checkpoint has no status_text_ tensors")
    _validate_state_matches_declared_model(
        payload,
        config=config,
        state=state,
        description="status v13 checkpoint",
    )
    initialization = payload.get("initialization")
    fine_tune = payload.get("fine_tune_policy")
    runtime = payload.get("training_runtime")
    if not isinstance(initialization, Mapping):
        raise ValueError("status v13 checkpoint has no initialization provenance")
    if not isinstance(fine_tune, Mapping) or not isinstance(runtime, Mapping):
        raise ValueError("status v13 checkpoint has no status-only training provenance")
    source_config = initialization.get("source_config")
    expected_source_config = asdict(config)
    expected_source_config["architecture_version"] = 12
    _validated_status_v12_source_config(
        source_config,
        expected=expected_source_config,
    )
    legacy_count = len(state) - len(status_keys)
    financial_policy = initialization.get("financial_label_policy")
    if (
        initialization.get("mode") != "parameter_only_v12_to_v13_status_text_expansion"
        or initialization.get("source_kind") != KIND_V12
        or initialization.get("optimizer_restored") is not False
        or initialization.get("epoch_reset") is not True
        or initialization.get("new_parameter_prefix") != STATUS_TEXT_PREFIX
        or initialization.get("copied_legacy_tensor_count") != legacy_count
        or initialization.get("new_status_text_tensor_count") != len(status_keys)
        or initialization.get("frozen_legacy_output_count") != len(V12_ONNX_OUTPUT_NAMES)
        or not isinstance(financial_policy, Mapping)
        or financial_policy.get("mode") != "checkpoint_legacy_label_maps_status_text_only_v1"
    ):
        raise ValueError("status v13 checkpoint does not prove additive status-text-only initialization")
    _require_sha256(
        initialization.get("checkpoint_sha256"),
        description="status v13 source checkpoint hash",
    )
    if (
        fine_tune.get("mode") != "status_text_only_v13"
        or fine_tune.get("trainable_parameter_prefix") != STATUS_TEXT_PREFIX
        or fine_tune.get("frozen_legacy_output_count") != len(V12_ONNX_OUTPUT_NAMES)
        or fine_tune.get("full_validation_schedule") != "epoch_1_every_n_and_final_epoch"
        or isinstance(fine_tune.get("validation_every"), bool)
        or not isinstance(fine_tune.get("validation_every"), int)
        or int(fine_tune["validation_every"]) <= 0
        or runtime.get("status_text_only_training") is not True
        or runtime.get("recipient_only_private_branch_training") is not False
        or runtime.get("full_validation_schedule") != "epoch_1_every_n_and_final_epoch"
        or runtime.get("validation_every") != fine_tune.get("validation_every")
    ):
        raise ValueError("status v13 checkpoint does not prove status-text-only frozen-legacy training")
    _valid_epoch(payload, description="status v13 checkpoint")
    return config, state


def _validate_train_only_v12(payload: Mapping[str, object]) -> tuple[Any, Mapping[str, object]]:
    if payload.get("kind") != KIND_V12:
        raise ValueError("train-only recipient checkpoint must be a v12 checkpoint")
    if ATTESTATION_KEY in payload:
        raise ValueError("train-only recipient checkpoint must not be a sanitizer output")
    _require_checkpoint_without_optimizer_state(payload, description="train-only v12 checkpoint")
    config = _checkpoint_config(payload)
    if config.recipient_input_width != REQUIRED_RECIPIENT_INPUT_WIDTH:
        raise ValueError("train-only v12 checkpoint is not the required wide1536 configuration")
    _checkpoint_labels(payload, config=config)
    _validate_v12_passive_status_text_metadata(
        payload, description="train-only v12 checkpoint"
    )
    state = _state_dict(payload, description="train-only v12 checkpoint")
    if any(key.startswith(STATUS_TEXT_PREFIX) for key in state):
        raise ValueError("train-only v12 checkpoint unexpectedly contains status_text_ tensors")
    _validate_state_matches_declared_model(
        payload,
        config=config,
        state=state,
        description="train-only v12 checkpoint",
    )
    _validate_no_laundered_recipient_policy(payload)
    _valid_epoch(payload, description="train-only v12 checkpoint")
    return config, state


def _lineage_entry(
    *, path: Path, identity: FileIdentity, payload: Mapping[str, object]
) -> dict[str, object]:
    config = _checkpoint_config(payload)
    state = _state_dict(payload, description=f"recipient lineage checkpoint {path}")
    policy = payload.get("recipient_train_split_policy")
    initialization = payload.get("initialization")
    if not isinstance(initialization, Mapping) or not isinstance(initialization.get("mode"), str):
        raise ValueError(f"recipient lineage checkpoint has no initialization mode: {path}")
    is_root = initialization.get("mode") == "random"
    recorded_parent_config = initialization.get("source_config")
    return {
        "checkpoint": _source_descriptor(path, identity, payload),
        "config_sha256": _canonical_sha256(
            asdict(config), description=f"recipient lineage config {path}"
        ),
        "recipient_charset_sha256": payload.get("recipient_charset_sha256"),
        "recipient_train_split_policy_sha256": _canonical_sha256(
            policy, description=f"recipient lineage train policy {path}"
        ),
        "initialization_mode": initialization.get("mode"),
        "parent_checkpoint_path": None if is_root else initialization.get("checkpoint_path"),
        "parent_checkpoint_sha256": None if is_root else initialization.get("checkpoint_sha256"),
        "parent_config_sha256": (
            None
            if is_root
            else _canonical_sha256(
                recorded_parent_config,
                description=f"recipient lineage recorded parent config {path}",
            )
        ),
        "parent_epoch": None if is_root else initialization.get("source_epoch"),
        "recipient_state": _partition_descriptor(state, recipient=True),
    }


def _validate_train_only_lineage_checkpoint(
    payload: Mapping[str, object], *, description: str
) -> tuple[Any, Mapping[str, object], Mapping[str, object]]:
    if payload.get("kind") != KIND_V12:
        raise ValueError(f"{description} must remain entirely within v12")
    _require_checkpoint_without_optimizer_state(payload, description=description)
    config = _checkpoint_config(payload)
    _checkpoint_labels(payload, config=config)
    _validate_v12_passive_status_text_metadata(payload, description=description)
    state = _state_dict(payload, description=description)
    if any(key.startswith(STATUS_TEXT_PREFIX) for key in state):
        raise ValueError(f"{description} unexpectedly contains status_text_ tensors")
    _validate_state_matches_declared_model(
        payload,
        config=config,
        state=state,
        description=description,
    )
    _validate_no_laundered_recipient_policy(payload)
    _validate_recipient_metadata(
        payload,
        config=config,
        description=description,
        require_train_only=True,
    )
    _valid_epoch(payload, description=description)
    initialization = payload.get("initialization")
    if not isinstance(initialization, Mapping):
        raise ValueError(f"{description} has no initialization provenance")
    if (
        initialization.get("optimizer_restored") is not False
        or initialization.get("epoch_reset") is not True
    ):
        raise ValueError(f"{description} does not prove fresh optimizer/epoch state")
    mode = initialization.get("mode")
    allowed_modes = {
        "random",
        "parameter_only",
        "parameter_only_recipient_unicode_expansion",
        "parameter_only_recipient_input_width_expansion",
        "parameter_only_recipient_capacity_reinit",
        "parameter_only_recipient_open_text_adapter",
    }
    if mode not in allowed_modes:
        raise ValueError(f"{description} has unsupported recipient lineage mode {mode!r}")
    if mode == "random" and any(
        initialization.get(key) is not None
        for key in (
            "checkpoint_path",
            "checkpoint_sha256",
            "source_kind",
            "source_epoch",
            "source_config",
        )
    ):
        raise ValueError(f"{description} random initialization must not claim a parent")
    return config, state, initialization


def _build_train_only_recipient_lineage(
    *, leaf_path: Path, leaf_payload: Mapping[str, object], torch: Any
) -> dict[str, object]:
    """Follow hash-bound v12 checkpoint ancestry to a random-init root.

    Checking only the leaf's top-level split policy is insufficient: a leaf
    can train on ``train`` while retaining recipient tensors from a
    transductive warmstart.  Every parameter-only ancestor is therefore
    reopened by its recorded absolute path and SHA-256, and every layer must
    independently prove train-only supervision.  No optimizer, manifest, crop
    or held-out artifact is opened.
    """

    entries: list[dict[str, object]] = []
    seen_files: set[tuple[int, int]] = set()
    current_path = leaf_path
    current_payload = leaf_payload
    for depth in range(MAX_LINEAGE_DEPTH):
        current_path = _existing_regular_file(
            current_path, description=f"recipient lineage checkpoint {depth}"
        )
        identity = _file_identity(current_path)
        inode = identity[:2]
        if inode in seen_files:
            raise ValueError("recipient checkpoint lineage contains a cycle")
        seen_files.add(inode)
        current_config, current_state, initialization = _validate_train_only_lineage_checkpoint(
            current_payload, description=f"recipient lineage checkpoint {depth}"
        )
        entries.append(_lineage_entry(path=current_path, identity=identity, payload=current_payload))
        mode = initialization.get("mode")
        if mode == "random":
            return {
                "policy": LINEAGE_POLICY,
                "checkpoint_count": len(entries),
                "root_initialization_mode": "random",
                "entries": entries,
            }
        raw_source_path = initialization.get("checkpoint_path")
        if not isinstance(raw_source_path, str) or not raw_source_path:
            raise ValueError(
                f"recipient lineage checkpoint {depth} has no source checkpoint path"
            )
        source_path = _existing_regular_file(
            Path(raw_source_path), description=f"recipient lineage source {depth + 1}"
        )
        source_identity = _file_identity(source_path)
        expected_hash = _require_sha256(
            initialization.get("checkpoint_sha256"),
            description=f"recipient lineage source hash {depth + 1}",
        )
        if source_identity[3] != expected_hash:
            raise ValueError(
                f"recipient lineage source {depth + 1} no longer matches its recorded SHA-256"
            )
        if initialization.get("source_kind") != KIND_V12:
            raise ValueError(f"recipient lineage source {depth + 1} is not recorded as v12")
        source_payload = _load_checkpoint(source_path, torch=torch)
        source_config, source_state, _ = _validate_train_only_lineage_checkpoint(
            source_payload,
            description=f"recipient lineage source {depth + 1}",
        )
        recorded_source_config = initialization.get("source_config")
        if not isinstance(recorded_source_config, Mapping) or dict(recorded_source_config) != asdict(
            source_config
        ):
            raise ValueError(
                f"recipient lineage source {depth + 1} config does not match child provenance"
            )
        if initialization.get("source_epoch") != source_payload.get("epoch"):
            raise ValueError(
                f"recipient lineage source {depth + 1} epoch does not match child provenance"
            )
        _validate_recipient_lineage_transition(
            child_payload=current_payload,
            child_config=current_config,
            child_state=current_state,
            initialization=initialization,
            parent_payload=source_payload,
            parent_config=source_config,
            parent_state=source_state,
            description=f"recipient lineage transition {depth}->{depth + 1}",
        )
        if not _same_file_identity(source_path, source_identity):
            raise RuntimeError(
                f"recipient lineage source {depth + 1} changed while it was being inspected"
            )
        current_path = source_path
        current_payload = source_payload
    raise ValueError(f"recipient checkpoint lineage exceeds {MAX_LINEAGE_DEPTH} checkpoints")


def _validate_compatible_sources(
    status_payload: Mapping[str, object], train_payload: Mapping[str, object]
) -> dict[str, object]:
    status_config, status_state = _validate_status_only_v13(status_payload)
    train_config, train_state = _validate_train_only_v12(train_payload)
    expected_train_config = asdict(status_config)
    expected_train_config["architecture_version"] = 12
    if asdict(train_config) != expected_train_config:
        changed = [
            key
            for key in sorted(expected_train_config)
            if asdict(train_config).get(key) != expected_train_config[key]
        ]
        raise ValueError(
            "v13 status and train-only v12 configs may differ only by architecture_version; "
            f"incompatible fields: {', '.join(changed)}"
        )
    status_labels = _checkpoint_labels(status_payload, config=status_config)
    train_labels = _checkpoint_labels(train_payload, config=train_config)
    for index, description in ((0, "amount"), (1, "time"), (2, "payment"), (4, "status class"), (5, "bank")):
        if status_labels[index] != train_labels[index]:
            raise ValueError(f"v13 status and train-only v12 {description} label maps do not match")
    status_characters, status_recipient_metadata = _validate_recipient_metadata(
        status_payload,
        config=status_config,
        description="status v13 checkpoint",
        require_train_only=False,
    )
    train_characters, train_recipient_metadata = _validate_recipient_metadata(
        train_payload,
        config=train_config,
        description="train-only v12 checkpoint",
        require_train_only=True,
    )
    if not set(train_characters).issubset(set(status_characters)):
        raise ValueError(
            "train-only v12 recipient charset must be a subset of the status source charset"
        )
    status_recipient_keys = {key for key in status_state if key.startswith(RECIPIENT_PREFIX)}
    train_recipient_keys = {key for key in train_state if key.startswith(RECIPIENT_PREFIX)}
    status_status_keys = {key for key in status_state if key.startswith(STATUS_TEXT_PREFIX)}
    status_shared_keys = set(status_state) - status_recipient_keys - status_status_keys
    train_shared_keys = set(train_state) - train_recipient_keys
    if status_recipient_keys != train_recipient_keys:
        raise ValueError(
            "recipient state key mismatch: "
            f"missing={sorted(status_recipient_keys - train_recipient_keys)}, "
            f"unexpected={sorted(train_recipient_keys - status_recipient_keys)}"
        )
    if status_shared_keys != train_shared_keys:
        raise ValueError(
            "shared state key mismatch: "
            f"missing={sorted(status_shared_keys - train_shared_keys)}, "
            f"unexpected={sorted(train_shared_keys - status_shared_keys)}"
        )
    for name in sorted(status_shared_keys):
        if _tensor_signature(status_state[name], name=name) != _tensor_signature(
            train_state[name], name=name
        ):
            raise ValueError(f"checkpoint tensor shape/dtype mismatch for {name}")
    for name in sorted(status_recipient_keys - RECIPIENT_CLASSIFIER_KEYS):
        if _tensor_signature(status_state[name], name=name) != _tensor_signature(
            train_state[name], name=name
        ):
            raise ValueError(f"checkpoint tensor shape/dtype mismatch for {name}")
    _validate_classifier_row_transition(
        source_state=status_state,
        target_state=train_state,
        source_characters=status_characters,
        target_characters=train_characters,
        allow_hidden_change=False,
        description="v13 status to train-only v12 transplant",
    )
    if set(status_recipient_metadata) != set(train_recipient_metadata):
        raise ValueError("v13 status and train-only v12 recipient metadata key sets do not match")
    return {
        "status_config": status_config,
        "train_config": train_config,
        "status_state": status_state,
        "train_state": train_state,
        "status_recipient_metadata": status_recipient_metadata,
        "train_recipient_metadata": train_recipient_metadata,
        "recipient_characters": train_characters,
        "status_recipient_characters": status_characters,
        "status_text_key_count": len(status_status_keys),
        "shared_key_count": len(status_shared_keys),
    }


def _source_descriptor(
    path: Path, identity: FileIdentity, payload: Mapping[str, object]
) -> dict[str, object]:
    return {
        "path": str(path),
        "sha256": identity[3],
        "size_bytes": identity[2],
        "kind": payload.get("kind"),
        "epoch": payload.get("epoch"),
    }


def _build_sanitized_payload(
    *,
    status_payload: Mapping[str, object],
    train_payload: Mapping[str, object],
    status_source: Mapping[str, object],
    train_source: Mapping[str, object],
    train_lineage: Mapping[str, object],
) -> dict[str, object]:
    compatible = _validate_compatible_sources(status_payload, train_payload)
    status_source = dict(status_source)
    train_source = dict(train_source)
    status_state = compatible["status_state"]
    train_state = compatible["train_state"]
    assert isinstance(status_state, Mapping) and isinstance(train_state, Mapping)
    output: dict[str, object] = dict(status_payload)
    for key in [key for key in output if key.startswith(RECIPIENT_PREFIX)]:
        del output[key]
    train_recipient_metadata = compatible["train_recipient_metadata"]
    assert isinstance(train_recipient_metadata, Mapping)
    output.update(train_recipient_metadata)
    output.update(
        _sanitized_metadata_overrides(
            status_payload=status_payload,
            train_payload=train_payload,
            status_source=status_source,
            train_source=train_source,
            train_lineage=train_lineage,
        )
    )
    output_state = {
        name: (train_state[name] if name.startswith(RECIPIENT_PREFIX) else value)
        for name, value in status_state.items()
    }
    output["state_dict"] = output_state
    recipient_metadata, non_recipient_metadata = _metadata_partitions(output)
    status_recipient_metadata = compatible["status_recipient_metadata"]
    assert isinstance(status_recipient_metadata, Mapping)
    attestation = {
        "schema_version": SCHEMA_VERSION,
        "kind": ATTESTATION_KIND,
        "analysis_only": True,
        "production_route_authorized": False,
        "optimizer_state_loaded": False,
        "external_test_artifacts_opened": False,
        "publication_policy": PUBLICATION_POLICY,
        "topology_policy": TOPOLOGY_POLICY,
        "recipient_parameter_prefix": RECIPIENT_PREFIX,
        "status_checkpoint": status_source,
        "train_only_recipient_checkpoint": train_source,
        "train_only_recipient_lineage": dict(train_lineage),
        "compatibility": {
            "status_architecture_version": 13,
            "recipient_architecture_version": 12,
            "only_config_difference": "architecture_version",
            "recipient_input_width": REQUIRED_RECIPIENT_INPUT_WIDTH,
            "recipient_charset_relation": "train_only_subset_of_status_source_v1",
            "status_recipient_charset_sha256": status_payload[
                "recipient_charset_sha256"
            ],
            "recipient_charset_sha256": train_payload["recipient_charset_sha256"],
            "output_abi": list(V13_ONNX_OUTPUT_NAMES),
            "output_config_sha256": _canonical_sha256(
                output["config"], description="sanitized output config"
            ),
        },
        "state_proof": {
            "non_recipient_source": "status_checkpoint",
            "recipient_source": "train_only_recipient_checkpoint",
            "output_non_recipient": _partition_descriptor(output_state, recipient=False),
            "output_recipient": _partition_descriptor(output_state, recipient=True),
            "discarded_status_recipient": _partition_descriptor(status_state, recipient=True),
            "ignored_train_checkpoint_non_recipient": _partition_descriptor(
                train_state, recipient=False
            ),
            "status_text_tensor_count": compatible["status_text_key_count"],
            "shared_legacy_tensor_count": compatible["shared_key_count"],
        },
        "metadata_proof": {
            "non_recipient_source": "status_checkpoint_with_operative_provenance_rebound_v1",
            "recipient_source": "train_only_recipient_checkpoint",
            "rebound_non_recipient_keys": [
                "initialization",
                "metrics",
                "status_source_history",
                "training_runtime",
            ],
            "recipient_metadata_keys": sorted(recipient_metadata),
            "output_non_recipient_sha256": _canonical_sha256(
                non_recipient_metadata, description="sanitized non-recipient metadata"
            ),
            "output_recipient_sha256": _canonical_sha256(
                recipient_metadata, description="sanitized recipient metadata"
            ),
            "discarded_status_recipient_sha256": _canonical_sha256(
                status_recipient_metadata, description="discarded status recipient metadata"
            ),
            "status_source_history_sha256": _canonical_sha256(
                output["status_source_history"], description="status source history"
            ),
            "operative_metadata_sha256": _canonical_sha256(
                {
                    key: output[key]
                    for key in ("initialization", "training_runtime", "metrics")
                },
                description="sanitized operative metadata",
            ),
        },
    }
    attestation["integrity_sha256"] = _canonical_sha256(
        attestation, description="full-crop sanitizer attestation payload"
    )
    output[ATTESTATION_KEY] = attestation
    validate_recipient_full_crop_seed_attestation(output)
    return output


def _require_exact_keys(value: object, expected: set[str], *, description: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or set(value) != expected:
        observed = set(value) if isinstance(value, Mapping) else set()
        raise ValueError(
            f"{description} key set is invalid: missing={sorted(expected - observed)}, "
            f"unexpected={sorted(observed - expected)}"
        )
    return value


def _validate_partition_descriptor(value: object, *, description: str) -> Mapping[str, object]:
    descriptor = _require_exact_keys(
        value,
        {"tensor_count", "total_bytes", "key_set_sha256", "tensor_manifest_sha256"},
        description=description,
    )
    for key in ("tensor_count", "total_bytes"):
        raw = descriptor.get(key)
        if isinstance(raw, bool) or not isinstance(raw, int) or raw <= 0:
            raise ValueError(f"{description} {key} is invalid")
    _require_sha256(descriptor.get("key_set_sha256"), description=f"{description} key-set hash")
    _require_sha256(
        descriptor.get("tensor_manifest_sha256"), description=f"{description} tensor-manifest hash"
    )
    return descriptor


def _validate_source_descriptor(value: object, *, kind: str, description: str) -> Mapping[str, object]:
    source = _require_exact_keys(
        value, {"path", "sha256", "size_bytes", "kind", "epoch"}, description=description
    )
    raw_path = source.get("path")
    if (
        not isinstance(raw_path, str)
        or not raw_path
        or not (PurePosixPath(raw_path).is_absolute() or PureWindowsPath(raw_path).is_absolute())
    ):
        raise ValueError(f"{description} path must be absolute")
    _require_sha256(source.get("sha256"), description=f"{description} hash")
    if source.get("kind") != kind:
        raise ValueError(f"{description} kind is invalid")
    for key in ("size_bytes", "epoch"):
        raw = source.get(key)
        if isinstance(raw, bool) or not isinstance(raw, int) or raw <= 0:
            raise ValueError(f"{description} {key} is invalid")
    return source


def _validate_lineage_attestation(
    value: object,
    *,
    leaf_source: Mapping[str, object],
    leaf_recipient_charset_sha256: object,
    leaf_recipient_state: Mapping[str, object],
) -> Mapping[str, object]:
    lineage = _require_exact_keys(
        value,
        {"policy", "checkpoint_count", "root_initialization_mode", "entries"},
        description="train-only recipient lineage",
    )
    entries = lineage.get("entries")
    count = lineage.get("checkpoint_count")
    if (
        lineage.get("policy") != LINEAGE_POLICY
        or lineage.get("root_initialization_mode") != "random"
        or isinstance(count, bool)
        or not isinstance(count, int)
        or count <= 0
        or count > MAX_LINEAGE_DEPTH
        or not isinstance(entries, Sequence)
        or isinstance(entries, (str, bytes))
        or len(entries) != count
    ):
        raise ValueError("train-only recipient lineage policy/count is invalid")
    allowed_modes = {
        "random",
        "parameter_only",
        "parameter_only_recipient_unicode_expansion",
        "parameter_only_recipient_input_width_expansion",
        "parameter_only_recipient_capacity_reinit",
        "parameter_only_recipient_open_text_adapter",
    }
    train_only_policy_sha256 = _canonical_sha256(
        _recipient_train_split_policy(["train"]),
        description="canonical train-only recipient policy",
    )
    observed_sources: list[Mapping[str, object]] = []
    seen_hashes: set[str] = set()
    for index, raw_entry in enumerate(entries):
        entry = _require_exact_keys(
            raw_entry,
            {
                "checkpoint",
                "config_sha256",
                "recipient_charset_sha256",
                "recipient_train_split_policy_sha256",
                "initialization_mode",
                "parent_checkpoint_path",
                "parent_checkpoint_sha256",
                "parent_config_sha256",
                "parent_epoch",
                "recipient_state",
            },
            description=f"recipient lineage entry {index}",
        )
        source = _validate_source_descriptor(
            entry.get("checkpoint"),
            kind=KIND_V12,
            description=f"recipient lineage source {index}",
        )
        observed_sources.append(source)
        source_hash = str(source["sha256"])
        if source_hash in seen_hashes:
            raise ValueError("train-only recipient lineage repeats a checkpoint hash")
        seen_hashes.add(source_hash)
        for key, description in (
            ("config_sha256", "config hash"),
            ("recipient_charset_sha256", "recipient charset hash"),
            ("recipient_train_split_policy_sha256", "recipient policy hash"),
        ):
            _require_sha256(
                entry.get(key), description=f"recipient lineage entry {index} {description}"
            )
        if entry.get("recipient_train_split_policy_sha256") != train_only_policy_sha256:
            raise ValueError(
                f"recipient lineage entry {index} is not bound to the canonical train-only policy"
            )
        mode = entry.get("initialization_mode")
        if mode not in allowed_modes:
            raise ValueError(f"recipient lineage entry {index} initialization mode is invalid")
        if (index == len(entries) - 1) != (mode == "random"):
            raise ValueError("only the root recipient lineage entry may use random initialization")
        if mode == "random":
            if any(
                entry.get(key) is not None
                for key in (
                    "parent_checkpoint_path",
                    "parent_checkpoint_sha256",
                    "parent_config_sha256",
                    "parent_epoch",
                )
            ):
                raise ValueError("random recipient lineage root must not claim a parent")
        else:
            raw_parent_path = entry.get("parent_checkpoint_path")
            raw_parent_epoch = entry.get("parent_epoch")
            if (
                not isinstance(raw_parent_path, str)
                or not raw_parent_path
                or not (
                    PurePosixPath(raw_parent_path).is_absolute()
                    or PureWindowsPath(raw_parent_path).is_absolute()
                )
                or isinstance(raw_parent_epoch, bool)
                or not isinstance(raw_parent_epoch, int)
                or raw_parent_epoch <= 0
            ):
                raise ValueError(f"recipient lineage entry {index} parent link is invalid")
            _require_sha256(
                entry.get("parent_checkpoint_sha256"),
                description=f"recipient lineage entry {index} parent checkpoint hash",
            )
            _require_sha256(
                entry.get("parent_config_sha256"),
                description=f"recipient lineage entry {index} parent config hash",
            )
        _validate_partition_descriptor(
            entry.get("recipient_state"),
            description=f"recipient lineage entry {index} state proof",
        )
    if dict(observed_sources[0]) != dict(leaf_source):
        raise ValueError("train-only recipient lineage leaf does not match the transplanted checkpoint")
    leaf = entries[0]
    if leaf.get("recipient_charset_sha256") != leaf_recipient_charset_sha256:
        raise ValueError("recipient lineage leaf charset does not match the transplanted checkpoint")
    if dict(leaf.get("recipient_state", {})) != dict(leaf_recipient_state):
        raise ValueError("recipient lineage leaf state does not match the transplanted bytes")
    for index in range(len(entries) - 1):
        child = entries[index]
        parent = entries[index + 1]
        parent_source = parent["checkpoint"]
        if (
            child.get("parent_checkpoint_path") != parent_source.get("path")
            or child.get("parent_checkpoint_sha256") != parent_source.get("sha256")
            or child.get("parent_config_sha256") != parent.get("config_sha256")
            or child.get("parent_epoch") != parent_source.get("epoch")
        ):
            raise ValueError(f"recipient lineage child {index} does not bind parent {index + 1}")
    return lineage


def _load_attested_source(
    value: object,
    *,
    kind: str,
    description: str,
    torch: Any,
) -> tuple[Path, FileIdentity, Mapping[str, object]]:
    """Reopen one attested checkpoint and bind the descriptor to its bytes."""

    source = _validate_source_descriptor(value, kind=kind, description=description)
    path = _existing_regular_file(Path(str(source["path"])), description=description)
    identity = _file_identity(path)
    if identity[2] != source.get("size_bytes") or identity[3] != source.get("sha256"):
        raise ValueError(f"{description} bytes no longer match the sanitizer attestation")
    payload = _load_checkpoint(path, torch=torch)
    if payload.get("kind") != kind or payload.get("epoch") != source.get("epoch"):
        raise ValueError(f"{description} kind/epoch no longer match the sanitizer attestation")
    if not _same_file_identity(path, identity):
        raise RuntimeError(f"{description} changed while it was being revalidated")
    return path, identity, payload


def validate_recipient_full_crop_seed_attestation(
    payload: Mapping[str, object],
) -> dict[str, object]:
    """Validate the self-contained sanitizer proof used by warmstart.

    Source files are intentionally not reopened here.  Their hashes are
    provenance, while the current checkpoint's state and metadata partition
    hashes are recomputed.  This prevents copying only a top-level train-only
    policy or a stale attestation onto a different checkpoint.
    """

    if payload.get("kind") != KIND_V13:
        raise ValueError("full-crop sanitizer attestation requires a v13 checkpoint")
    _require_checkpoint_without_optimizer_state(payload, description="sanitized v13 checkpoint")
    config = _checkpoint_config(payload)
    if config.architecture_version != 13 or config.recipient_input_width != REQUIRED_RECIPIENT_INPUT_WIDTH:
        raise ValueError("sanitized v13 checkpoint must use the wide1536 v13 config")
    _checkpoint_labels(payload, config=config)
    _checkpoint_status_text_characters(payload, config=config)
    _validate_recipient_metadata(
        payload,
        config=config,
        description="sanitized v13 checkpoint",
        require_train_only=True,
    )
    state = _state_dict(payload, description="sanitized v13 checkpoint")
    if not any(key.startswith(STATUS_TEXT_PREFIX) for key in state):
        raise ValueError("sanitized v13 checkpoint has no visible-status tensors")
    _validate_state_matches_declared_model(
        payload,
        config=config,
        state=state,
        description="sanitized v13 checkpoint",
    )
    attestation = _require_exact_keys(
        payload.get(ATTESTATION_KEY),
        {
            "schema_version",
            "kind",
            "analysis_only",
            "production_route_authorized",
            "optimizer_state_loaded",
            "external_test_artifacts_opened",
            "publication_policy",
            "topology_policy",
            "recipient_parameter_prefix",
            "status_checkpoint",
            "train_only_recipient_checkpoint",
            "train_only_recipient_lineage",
            "compatibility",
            "state_proof",
            "metadata_proof",
            "integrity_sha256",
        },
        description="full-crop sanitizer attestation",
    )
    observed_integrity = _require_sha256(
        attestation.get("integrity_sha256"), description="sanitizer attestation integrity hash"
    )
    unsigned_attestation = {
        key: value for key, value in attestation.items() if key != "integrity_sha256"
    }
    if observed_integrity != _canonical_sha256(
        unsigned_attestation, description="full-crop sanitizer attestation payload"
    ):
        raise ValueError("full-crop sanitizer attestation integrity hash does not match")
    if (
        attestation.get("schema_version") != SCHEMA_VERSION
        or attestation.get("kind") != ATTESTATION_KIND
        or attestation.get("analysis_only") is not True
        or attestation.get("production_route_authorized") is not False
        or attestation.get("optimizer_state_loaded") is not False
        or attestation.get("external_test_artifacts_opened") is not False
        or attestation.get("publication_policy") != PUBLICATION_POLICY
        or attestation.get("topology_policy") != TOPOLOGY_POLICY
        or attestation.get("recipient_parameter_prefix") != RECIPIENT_PREFIX
    ):
        raise ValueError("full-crop sanitizer analysis/safety policy is invalid")
    _validate_sanitized_operative_metadata(payload, attestation=attestation)
    _validate_source_descriptor(
        attestation.get("status_checkpoint"), kind=KIND_V13, description="status source"
    )
    train_source = _validate_source_descriptor(
        attestation.get("train_only_recipient_checkpoint"),
        kind=KIND_V12,
        description="train-only recipient source",
    )
    _validate_lineage_attestation(
        attestation.get("train_only_recipient_lineage"),
        leaf_source=train_source,
        leaf_recipient_charset_sha256=payload.get("recipient_charset_sha256"),
        leaf_recipient_state=_partition_descriptor(state, recipient=True),
    )
    compatibility = _require_exact_keys(
        attestation.get("compatibility"),
        {
            "status_architecture_version",
            "recipient_architecture_version",
            "only_config_difference",
            "recipient_input_width",
            "recipient_charset_relation",
            "status_recipient_charset_sha256",
            "recipient_charset_sha256",
            "output_abi",
            "output_config_sha256",
        },
        description="sanitizer compatibility proof",
    )
    if (
        compatibility.get("status_architecture_version") != 13
        or compatibility.get("recipient_architecture_version") != 12
        or compatibility.get("only_config_difference") != "architecture_version"
        or compatibility.get("recipient_input_width") != REQUIRED_RECIPIENT_INPUT_WIDTH
        or compatibility.get("recipient_charset_relation")
        != "train_only_subset_of_status_source_v1"
        or compatibility.get("recipient_charset_sha256") != payload.get("recipient_charset_sha256")
        or compatibility.get("output_abi") != list(V13_ONNX_OUTPUT_NAMES)
        or compatibility.get("output_config_sha256")
        != _canonical_sha256(payload["config"], description="sanitized output config")
    ):
        raise ValueError("sanitizer compatibility proof does not match the checkpoint")
    _require_sha256(
        compatibility.get("status_recipient_charset_sha256"),
        description="status-source recipient charset hash",
    )
    _require_sha256(
        compatibility.get("recipient_charset_sha256"), description="sanitized recipient charset hash"
    )
    _require_sha256(
        compatibility.get("output_config_sha256"), description="sanitized config hash"
    )
    state_proof = _require_exact_keys(
        attestation.get("state_proof"),
        {
            "non_recipient_source",
            "recipient_source",
            "output_non_recipient",
            "output_recipient",
            "discarded_status_recipient",
            "ignored_train_checkpoint_non_recipient",
            "status_text_tensor_count",
            "shared_legacy_tensor_count",
        },
        description="sanitizer state proof",
    )
    if (
        state_proof.get("non_recipient_source") != "status_checkpoint"
        or state_proof.get("recipient_source") != "train_only_recipient_checkpoint"
    ):
        raise ValueError("sanitizer state source policy is invalid")
    expected_non_recipient = _partition_descriptor(state, recipient=False)
    expected_recipient = _partition_descriptor(state, recipient=True)
    observed_non_recipient = _validate_partition_descriptor(
        state_proof.get("output_non_recipient"), description="output non-recipient state proof"
    )
    observed_recipient = _validate_partition_descriptor(
        state_proof.get("output_recipient"), description="output recipient state proof"
    )
    _validate_partition_descriptor(
        state_proof.get("discarded_status_recipient"),
        description="discarded status recipient state proof",
    )
    _validate_partition_descriptor(
        state_proof.get("ignored_train_checkpoint_non_recipient"),
        description="ignored train non-recipient state proof",
    )
    if dict(observed_non_recipient) != expected_non_recipient:
        raise ValueError("sanitized non-recipient state no longer matches its attested bytes")
    if dict(observed_recipient) != expected_recipient:
        raise ValueError("sanitized recipient state no longer matches its attested bytes")
    status_count = sum(key.startswith(STATUS_TEXT_PREFIX) for key in state)
    shared_count = sum(
        not key.startswith(STATUS_TEXT_PREFIX) and not key.startswith(RECIPIENT_PREFIX)
        for key in state
    )
    if (
        state_proof.get("status_text_tensor_count") != status_count
        or state_proof.get("shared_legacy_tensor_count") != shared_count
    ):
        raise ValueError("sanitizer state partition counts do not match the checkpoint")
    recipient_metadata, non_recipient_metadata = _metadata_partitions(payload)
    metadata_proof = _require_exact_keys(
        attestation.get("metadata_proof"),
        {
            "non_recipient_source",
            "recipient_source",
            "rebound_non_recipient_keys",
            "recipient_metadata_keys",
            "output_non_recipient_sha256",
            "output_recipient_sha256",
            "discarded_status_recipient_sha256",
            "status_source_history_sha256",
            "operative_metadata_sha256",
        },
        description="sanitizer metadata proof",
    )
    if (
        metadata_proof.get("non_recipient_source")
        != "status_checkpoint_with_operative_provenance_rebound_v1"
        or metadata_proof.get("recipient_source") != "train_only_recipient_checkpoint"
        or metadata_proof.get("rebound_non_recipient_keys")
        != ["initialization", "metrics", "status_source_history", "training_runtime"]
        or metadata_proof.get("recipient_metadata_keys") != sorted(recipient_metadata)
        or metadata_proof.get("output_non_recipient_sha256")
        != _canonical_sha256(
            non_recipient_metadata, description="sanitized non-recipient metadata"
        )
        or metadata_proof.get("output_recipient_sha256")
        != _canonical_sha256(recipient_metadata, description="sanitized recipient metadata")
        or metadata_proof.get("status_source_history_sha256")
        != _canonical_sha256(
            payload["status_source_history"], description="sanitized status source history"
        )
        or metadata_proof.get("operative_metadata_sha256")
        != _canonical_sha256(
            {
                key: payload[key]
                for key in ("initialization", "training_runtime", "metrics")
            },
            description="sanitized operative metadata",
        )
    ):
        raise ValueError("sanitizer metadata proof does not match the checkpoint")
    _require_sha256(
        metadata_proof.get("discarded_status_recipient_sha256"),
        description="discarded status recipient metadata hash",
    )
    return dict(attestation)


def verify_recipient_full_crop_seed_source_provenance(
    payload: Mapping[str, object], *, torch: Any
) -> dict[str, object]:
    """Reopen every attested source before a full-crop warm start.

    The self-contained hashes detect accidental corruption, but they are not a
    signature: somebody could splice entries and recompute a public digest.
    The guarded warm-start boundary therefore reopens A, B, and every recorded
    B ancestor, checks their file hashes and checkpoint provenance, and
    reconstructs the lineage from the source checkpoints themselves.  Only
    checkpoint files are opened; no optimizer, manifest, crop, validation, or
    test artifact is consulted.
    """

    attestation = validate_recipient_full_crop_seed_attestation(payload)
    status_path, status_identity, status_payload = _load_attested_source(
        attestation["status_checkpoint"],
        kind=KIND_V13,
        description="attested status source",
        torch=torch,
    )
    train_path, train_identity, train_payload = _load_attested_source(
        attestation["train_only_recipient_checkpoint"],
        kind=KIND_V12,
        description="attested train-only recipient source",
        torch=torch,
    )
    compatible = _validate_compatible_sources(status_payload, train_payload)
    compatibility_proof = attestation["compatibility"]
    if not isinstance(compatibility_proof, Mapping) or (
        compatibility_proof.get("status_recipient_charset_sha256")
        != status_payload.get("recipient_charset_sha256")
        or compatibility_proof.get("recipient_charset_sha256")
        != train_payload.get("recipient_charset_sha256")
    ):
        raise ValueError("sanitizer charset compatibility proof does not match reopened sources")
    rebuilt_lineage = _build_train_only_recipient_lineage(
        leaf_path=train_path,
        leaf_payload=train_payload,
        torch=torch,
    )
    if _canonical_sha256(
        rebuilt_lineage, description="rebuilt train-only recipient lineage"
    ) != _canonical_sha256(
        attestation["train_only_recipient_lineage"],
        description="attested train-only recipient lineage",
    ):
        raise ValueError("attested train-only recipient lineage does not match its source files")

    output_state = _state_dict(payload, description="sanitized v13 checkpoint")
    status_state = compatible["status_state"]
    train_state = compatible["train_state"]
    if not isinstance(status_state, Mapping) or not isinstance(train_state, Mapping):
        raise AssertionError("compatible source validation returned invalid state mappings")
    state_proof = attestation["state_proof"]
    if not isinstance(state_proof, Mapping):
        raise AssertionError("validated sanitizer attestation returned invalid state proof")
    if (
        _partition_descriptor(output_state, recipient=False)
        != _partition_descriptor(status_state, recipient=False)
        or _partition_descriptor(output_state, recipient=True)
        != _partition_descriptor(train_state, recipient=True)
        or state_proof.get("discarded_status_recipient")
        != _partition_descriptor(status_state, recipient=True)
        or state_proof.get("ignored_train_checkpoint_non_recipient")
        != _partition_descriptor(train_state, recipient=False)
    ):
        raise ValueError("sanitized state partitions do not match the reopened source checkpoints")

    output_recipient, output_non_recipient = _metadata_partitions(payload)
    train_recipient, _ = _metadata_partitions(train_payload)
    status_recipient, status_non_recipient = _metadata_partitions(status_payload)
    expected_non_recipient = dict(status_non_recipient)
    for key in _STATUS_HISTORY_FIELDS:
        expected_non_recipient.pop(key, None)
    expected_non_recipient.update(
        _sanitized_metadata_overrides(
            status_payload=status_payload,
            train_payload=train_payload,
            status_source=attestation["status_checkpoint"],
            train_source=attestation["train_only_recipient_checkpoint"],
            train_lineage=attestation["train_only_recipient_lineage"],
        )
    )
    metadata_proof = attestation["metadata_proof"]
    if not isinstance(metadata_proof, Mapping):
        raise AssertionError("validated sanitizer attestation returned invalid metadata proof")
    if (
        _canonical_sha256(output_recipient, description="output recipient metadata")
        != _canonical_sha256(train_recipient, description="train source recipient metadata")
        or _canonical_sha256(output_non_recipient, description="output non-recipient metadata")
        != _canonical_sha256(
            expected_non_recipient,
            description="status source metadata with sanitized operative provenance",
        )
        or metadata_proof.get("discarded_status_recipient_sha256")
        != _canonical_sha256(
            status_recipient, description="reopened discarded status recipient metadata"
        )
    ):
        raise ValueError("sanitized metadata partitions do not match the reopened source checkpoints")
    if not _same_file_identity(status_path, status_identity) or not _same_file_identity(
        train_path, train_identity
    ):
        raise RuntimeError("a sanitizer source changed during provenance revalidation")
    return attestation


def sanitize_recipient_full_crop_seed(
    *,
    status_checkpoint: Path,
    train_only_recipient_checkpoint: Path,
    output_checkpoint: Path,
    torch: Any,
) -> dict[str, object]:
    status_path = _existing_regular_file(status_checkpoint, description="status v13 checkpoint")
    train_path = _existing_regular_file(
        train_only_recipient_checkpoint, description="train-only v12 checkpoint"
    )
    output_path = _fresh_output_file(output_checkpoint)
    if status_path == train_path:
        raise ValueError("status and train-only recipient checkpoints must be different files")
    if output_path in {status_path, train_path}:
        raise ValueError("sanitized output must differ from both input checkpoints")
    status_identity = _file_identity(status_path)
    train_identity = _file_identity(train_path)
    if status_identity[:2] == train_identity[:2]:
        raise ValueError(
            "status and train-only recipient checkpoints must not be aliases of the same file"
        )
    status_payload = _load_checkpoint(status_path, torch=torch)
    train_payload = _load_checkpoint(train_path, torch=torch)
    train_lineage = _build_train_only_recipient_lineage(
        leaf_path=train_path,
        leaf_payload=train_payload,
        torch=torch,
    )
    output_payload = _build_sanitized_payload(
        status_payload=status_payload,
        train_payload=train_payload,
        status_source=_source_descriptor(status_path, status_identity, status_payload),
        train_source=_source_descriptor(train_path, train_identity, train_payload),
        train_lineage=train_lineage,
    )
    temporary = output_path.parent / f".{output_path.name}.sanitizer-{uuid4().hex}.tmp"
    descriptor = os.open(temporary, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    temporary_identity: FileIdentity | None = None
    publication_complete = False
    try:
        with os.fdopen(descriptor, "wb") as stream:
            descriptor = -1
            torch.save(output_payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        temporary_identity = _file_identity(temporary)
        reloaded = _load_checkpoint(temporary, torch=torch)
        validate_recipient_full_crop_seed_attestation(reloaded)
        verify_recipient_full_crop_seed_source_provenance(reloaded, torch=torch)
        reloaded_state = _state_dict(reloaded, description="reloaded sanitized checkpoint")
        status_state = _state_dict(status_payload, description="status v13 checkpoint")
        train_state = _state_dict(train_payload, description="train-only v12 checkpoint")
        for name in sorted(reloaded_state):
            expected = train_state[name] if name.startswith(RECIPIENT_PREFIX) else status_state[name]
            observed_signature = _tensor_signature(reloaded_state[name], name=name)
            expected_signature = _tensor_signature(expected, name=name)
            if observed_signature != expected_signature:
                raise AssertionError(f"published state signature changed for {name}")
            _, _, observed_bytes = _tensor_bytes(reloaded_state[name], name=name)
            _, _, expected_bytes = _tensor_bytes(expected, name=name)
            if observed_bytes != expected_bytes:
                source = "train-only v12" if name.startswith(RECIPIENT_PREFIX) else "status v13"
                raise AssertionError(f"sanitized state entry {name} is not byte-identical to {source}")
        expected_recipient, expected_non_recipient = _metadata_partitions(output_payload)
        observed_recipient, observed_non_recipient = _metadata_partitions(reloaded)
        if _canonical_sha256(expected_recipient, description="expected recipient metadata") != _canonical_sha256(
            observed_recipient, description="observed recipient metadata"
        ):
            raise AssertionError("sanitized recipient metadata changed during serialization")
        if _canonical_sha256(
            expected_non_recipient, description="expected non-recipient metadata"
        ) != _canonical_sha256(observed_non_recipient, description="observed non-recipient metadata"):
            raise AssertionError("sanitized non-recipient metadata changed during serialization")
        if not _same_file_identity(status_path, status_identity):
            raise RuntimeError("status v13 checkpoint changed during sanitizer execution")
        if not _same_file_identity(train_path, train_identity):
            raise RuntimeError("train-only v12 checkpoint changed during sanitizer execution")
        _fresh_output_file(output_path)
        os.link(temporary, output_path)
        output_identity = _file_identity(output_path)
        if temporary_identity is None or output_identity != temporary_identity:
            raise AssertionError("atomic sanitizer publication did not preserve the verified file identity")
        temporary.unlink()
        publication_complete = True
        return {
            "kind": ATTESTATION_KIND,
            "analysis_only": True,
            "production_route_authorized": False,
            "output_checkpoint": str(output_path),
            "output_checkpoint_sha256": output_identity[3],
            "output_size_bytes": output_identity[2],
            "status_checkpoint_sha256": status_identity[3],
            "train_only_recipient_checkpoint_sha256": train_identity[3],
            "recipient_tensor_count": output_payload[ATTESTATION_KEY]["state_proof"][
                "output_recipient"
            ]["tensor_count"],
            "non_recipient_tensor_count": output_payload[ATTESTATION_KEY]["state_proof"][
                "output_non_recipient"
            ]["tensor_count"],
        }
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        if (
            not publication_complete
            and temporary_identity is not None
            and _same_file_identity(output_path, temporary_identity)
        ):
            output_path.unlink()
        if temporary_identity is not None and _same_file_identity(temporary, temporary_identity):
            temporary.unlink()
        elif temporary_identity is None and os.path.lexists(temporary):
            # The path was created by this process but failed before its first
            # post-save identity snapshot.  Never recurse or remove anything
            # other than that exact reserved regular-file path.
            try:
                info = temporary.stat(follow_symlinks=False)
                if stat.S_ISREG(info.st_mode):
                    temporary.unlink()
            except OSError:
                pass


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--status-checkpoint", type=Path, required=True)
    parser.add_argument("--train-only-recipient-checkpoint", type=Path, required=True)
    parser.add_argument("--output-checkpoint", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    try:
        import torch
    except ModuleNotFoundError as error:
        raise RuntimeError("recipient full-crop seed sanitization requires PyTorch") from error
    summary = sanitize_recipient_full_crop_seed(
        status_checkpoint=args.status_checkpoint,
        train_only_recipient_checkpoint=args.train_only_recipient_checkpoint,
        output_checkpoint=args.output_checkpoint,
        torch=torch,
    )
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
