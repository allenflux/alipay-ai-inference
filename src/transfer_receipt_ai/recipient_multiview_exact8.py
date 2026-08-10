"""Run one guarded fixed-two-view recipient exact-eight pilot.

The route is intentionally narrow.  It consumes an independently sealed
``standard``/``fixed_value`` composite manifest, reopens the original
full-crop source and the frozen A8 baseline, and starts a fresh
``recipient_visual_context_reinit`` experiment.  No checkpoint produced here
may initialize this pilot or a later candidate.  The only positive outcome is
authority for one separate fresh-60 run that starts from the original pilot
best again and uses the exact same fixed-two-view overlay.

This module never exports ONNX, opens test, or authorizes production.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import io
import json
import math
import os
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .ocr_unified import (
    CHECKPOINT_SELECTION_RECIPIENT_PRIORITY,
    INIT_CHECKPOINT_MODE_RECIPIENT_VISUAL_CONTEXT_REINIT,
    KIND_V13,
    NUMERIC_BLANK_INDEX,
    PAYMENT_BLANK_INDEX,
    STATUS_TEXT_RUNTIME_POLICY,
    UnifiedReaderConfig,
    _checkpoint_config,
    _checkpoint_protection_report,
    _checkpoint_selection_policy,
    _checkpoint_selection_score,
    _load_checkpoint,
    _recipient_artifact_metadata,
    _recipient_confidence_policy,
    _recipient_train_augmentation_policy,
    _require_torch,
    _validate_recipient_tail_loss_policy,
    _validate_recipient_visual_context_reinit_config,
    train_unified_reader,
)
from .recipient_full_crop_candidate_source import (
    CANDIDATE_PILOT_KIND,
    EXPECTED_RECIPIENT_VAL_RECORDS,
    REQUIRED_BACKBONE,
    SOURCE_KIND,
    verify_full_crop_candidate_source,
    verify_residual_candidate_pilot,
)
from .recipient_full_crop_continuation import (
    FIXED_SOURCE_SUBJECT_ID,
    _recompute_pilot_closure,
)
from .recipient_full_crop_pilot import (
    AMOUNT_FLOOR,
    PAYMENT_FLOOR,
    STATUS_TEXT_FLOOR,
    TIME_FLOOR,
)
from .recipient_full_crop_seed_sanitizer import (
    _partition_descriptor,
    _require_checkpoint_without_optimizer_state,
    _state_dict,
    _validate_state_matches_declared_model,
)
from .recipient_v14_failure_attestor import (
    AUTHORIZATION as FAILURE_AUTHORIZATION,
    DECISION as FAILURE_DECISION,
    KIND as FAILURE_KIND,
    verify_fresh60_failure,
)


SCHEMA_VERSION = 1
INSPECTION_KIND = "receipt_recipient_multiview_fixed2_exact8_subject_v1"
RECIPE_KIND = "receipt_recipient_multiview_fixed2_exact8_recipe_v1"
DECISION_KIND = "receipt_recipient_multiview_fixed2_exact8_decision_v1"
ATTEMPT_KIND = "receipt_recipient_multiview_fixed2_exact8_attempt_v1"
ATTEMPT_REGISTRY_NAME = "recipient-v14-multiview-fixed2-training-v1"
ATTEMPT_REGISTRY_PARENT = "ReceiptAI"
ATTEMPT_THREAT_MODEL = (
    "persistent current-SID no-rerun guard; crash and failed training consume "
    "fixed2 exact8; owner WRITE_DAC and local-administrator bypass are out of scope"
)
OVERLAY_KIND = "receipt_recipient_fixed2_overlay_contract_v1"
ROUTE_DOMAIN = "receipt-recipient-multiview-fixed2-exact8-v1"
SELECTOR_MODE = "sha256_rank_parity_v1"
SELECTED_VIEWS = ["standard", "fixed_value"]
PASS_AUTHORIZATION = "fresh_fixed2_60_from_original_pilot_best_only"

FIXED_EPOCHS = 8
FIXED_BATCH_SIZE = 10
FIXED_LEARNING_RATE = 0.0003
FIXED_WEIGHT_DECAY = 0.0001
FIXED_SEED = 42
FIXED_NUM_WORKERS = 4
FIXED_PREFETCH_FACTOR = 2
FIXED_PROGRESS_EVERY = 250
FIXED_AUGMENTATION = "robust_v2"
BASELINE_GAIN_MATCHES = 68
MINIMUM_BEST_MATCHES = 5790
MINIMUM_EPOCH4_TO_8_GAIN_MATCHES = 136
MAXIMUM_BEST_TO_EPOCH8_GAP_MATCHES = 67
STRICT_RECIPIENT_PASS_MATCHES = 6111
EXPECTED_CANDIDATE_VAL_RECORDS = {
    "amount": 1428,
    "time": 3738,
    "payment_method_field": 5242,
    "recipient_field": EXPECTED_RECIPIENT_VAL_RECORDS,
}

_WINDOWS_DELETE = 0x00010000
_WINDOWS_FILE_DELETE_CHILD = 0x00000040
_WINDOWS_READ_CONTROL = 0x00020000
_WINDOWS_TOKEN_DUPLICATE = 0x0002
_WINDOWS_TOKEN_QUERY = 0x0008
_WINDOWS_TOKEN_USER = 1
_WINDOWS_SECURITY_IMPERSONATION = 2
_WINDOWS_SE_FILE_OBJECT = 1
_WINDOWS_OWNER_SECURITY_INFORMATION = 0x00000001
_WINDOWS_GROUP_SECURITY_INFORMATION = 0x00000002
_WINDOWS_DACL_SECURITY_INFORMATION = 0x00000004
_WINDOWS_ACCESS_DENIED_ACE_TYPE = 0x01
_WINDOWS_OBJECT_INHERIT_ACE = 0x01
_WINDOWS_CONTAINER_INHERIT_ACE = 0x02
_WINDOWS_INHERIT_ONLY_ACE = 0x08
_WINDOWS_INHERITED_ACE = 0x10
_WINDOWS_ACL_SIZE_INFORMATION = 2
_WINDOWS_FILE_GENERIC_READ = 0x00120089
_WINDOWS_FILE_GENERIC_WRITE = 0x00120116
_WINDOWS_FILE_GENERIC_EXECUTE = 0x001200A0
_WINDOWS_FILE_ALL_ACCESS = 0x001F01FF
_WINDOWS_RPC_E_CHANGED_MODE = -2147417850
_WINDOWS_PROGRAM_DATA_GUID = (
    0x62AB5D82,
    0xFDC1,
    0x4DC3,
    (0xA9, 0xDD, 0x07, 0x0D, 0x1D, 0x49, 0x5D, 0x97),
)


class _WindowsGuid(ctypes.Structure):
    _fields_ = (
        ("data1", ctypes.c_uint32),
        ("data2", ctypes.c_uint16),
        ("data3", ctypes.c_uint16),
        ("data4", ctypes.c_ubyte * 8),
    )


class _WindowsSidAndAttributes(ctypes.Structure):
    _fields_ = (("sid", ctypes.c_void_p), ("attributes", ctypes.c_uint32))


class _WindowsTokenUser(ctypes.Structure):
    _fields_ = (("user", _WindowsSidAndAttributes),)


class _WindowsAclSizeInformation(ctypes.Structure):
    _fields_ = (
        ("ace_count", ctypes.c_uint32),
        ("acl_bytes_in_use", ctypes.c_uint32),
        ("acl_bytes_free", ctypes.c_uint32),
    )


class _WindowsAceHeader(ctypes.Structure):
    _fields_ = (
        ("ace_type", ctypes.c_ubyte),
        ("ace_flags", ctypes.c_ubyte),
        ("ace_size", ctypes.c_uint16),
    )


class _WindowsAccessDeniedAce(ctypes.Structure):
    _fields_ = (
        ("header", _WindowsAceHeader),
        ("mask", ctypes.c_uint32),
        ("sid_start", ctypes.c_uint32),
    )


class _WindowsGenericMapping(ctypes.Structure):
    _fields_ = (
        ("generic_read", ctypes.c_uint32),
        ("generic_write", ctypes.c_uint32),
        ("generic_execute", ctypes.c_uint32),
        ("generic_all", ctypes.c_uint32),
    )


A8_SUMMARY_DATA_KEYS = (
    "field_counts",
    "status_class_counts",
    "status_head_policy",
    "structured_target_counts",
    "status_text_oov_by_split",
    "payment_oov_by_split",
    "payment_bank_prefix_classes",
    "payment_bank_prefix_min_support",
    "payment_bank_prefix_class_counts",
    "payment_bank_prefix_train_class_counts",
    "payment_bank_prefix_oov_by_split",
    "recipient_oov_by_split",
    "recipient_sampling_policy",
    "recipient_confidence_policy",
    "recipient_tail_loss_policy",
    "recipient_train_augmentation_policy",
    "recipient_train_split_policy",
    "recipient_target",
    "status_text_charset_sha256",
    "status_text_charset_source",
    "status_text_target",
)
A8_ORDERED_LABEL_KEYS = (
    "amount_characters",
    "time_characters",
    "payment_characters",
    "status_classes",
    "status_text_blank_index",
    "status_text_characters",
    "status_text_charset_sha256",
    "status_text_charset_source",
    "status_text_target",
    "recipient_blank_index",
    "recipient_characters",
    "recipient_charset_sha256",
    "recipient_charset_source",
    "recipient_target",
    "payment_bank_prefix_classes",
    "payment_bank_prefix_min_support",
    "payment_bank_prefix_class_counts",
    "payment_bank_prefix_train_class_counts",
    "payment_bank_prefix_oov_by_split",
)
A8_ORDERED_MAP_KEYS = (
    "amount_characters",
    "time_characters",
    "payment_characters",
    "status_classes",
    "status_text_characters",
    "recipient_characters",
)
A8_BLANK_INDEX_KEYS = (
    "amount_blank_index",
    "time_blank_index",
    "payment_blank_index",
    "status_text_blank_index",
    "recipient_blank_index",
)
A8_BLANK_INDEX_PROOF = {
    "amount_blank_index": ("fixed_protocol_constants", "NUMERIC_BLANK_INDEX"),
    "time_blank_index": ("fixed_protocol_constants", "NUMERIC_BLANK_INDEX"),
    "payment_blank_index": ("fixed_protocol_constants", "PAYMENT_BLANK_INDEX"),
    "status_text_blank_index": (
        "A8_checkpoint_explicit",
        "status_text_blank_index",
    ),
    "recipient_blank_index": (
        "A8_checkpoint_explicit",
        "recipient_blank_index",
    ),
}
EXACT8_LABEL_BASE_KEYS = frozenset(
    {
        "schema_version",
        "amount_blank_index",
        "amount_characters",
        "time_blank_index",
        "time_characters",
        "payment_blank_index",
        "payment_characters",
        "payment_charset_sha256",
        "status_classes",
        "status_text_blank_index",
        "status_text_characters",
        "status_text_charset_sha256",
        "status_text_charset_source",
        "status_text_target",
        "status_text_oov_by_split",
        "status_text_runtime_policy",
        "structured_target_counts",
        "checkpoint_selection_policy",
        "initialization",
        "training_runtime",
        "fine_tune_policy",
        "recipient_blank_index",
        "recipient_characters",
        "recipient_charset_sha256",
        "recipient_charset_source",
        "recipient_target",
        "recipient_oov_by_split",
        "recipient_sampling_policy",
        "recipient_confidence_policy",
        "recipient_tail_loss_policy",
        "recipient_train_augmentation_policy",
        "recipient_train_split_policy",
        "payment_bank_prefix_classes",
        "payment_bank_prefix_min_support",
        "payment_bank_prefix_class_counts",
        "payment_bank_prefix_train_class_counts",
        "payment_bank_prefix_oov_by_split",
    }
)
ATTESTED_SOURCE_SUBJECT_ID = (
    "98f0617404d7d58e99a0794d2340da9154f81667f0aa6a546027dd19209b886a"
)
ATTESTED_A8_SUBJECT_ID = (
    "5d5c0cbe5041252dc9de8d69076400deb7c8d3909d81c424287863d59b49433e"
)


@dataclass(frozen=True)
class _FrozenFile:
    path: Path
    data: bytes
    sha256: str
    size_bytes: int
    identity: tuple[int, int, int, int, int]


def _stat_identity(stat: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        int(stat.st_dev),
        int(stat.st_ino),
        int(stat.st_size),
        int(stat.st_mtime_ns),
        int(stat.st_ctime_ns),
    )


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


def _strict_json_bytes(data: bytes, *, description: str) -> dict[str, Any]:
    def reject_constant(value: str) -> None:
        raise ValueError(f"{description}: non-finite JSON constant {value!r}")

    try:
        raw = json.loads(
            data.decode("utf-8-sig"),
            parse_constant=reject_constant,
        )
    except (UnicodeError, json.JSONDecodeError) as error:
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
        data = path.read_bytes()
    except OSError as error:
        raise ValueError(f"Unable to read strict JSON object {path}: {error}") from error
    return _strict_json_bytes(data, description=str(path))


def _mapping(value: object, description: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{description} must be an object")
    return value


def _require_equal(actual: object, expected: object, description: str) -> None:
    if type(actual) is not type(expected) or actual != expected:
        raise ValueError(
            f"{description} mismatch: expected {expected!r}, found {actual!r}"
        )


def _json_equal(actual: object, expected: object, description: str) -> None:
    try:
        actual_digest = _canonical_sha256({"value": actual})
        expected_digest = _canonical_sha256({"value": expected})
    except (TypeError, ValueError) as error:
        raise ValueError(f"{description} is not strict JSON-compatible") from error
    if actual_digest != expected_digest:
        raise ValueError(f"{description} does not match its authoritative value")


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


def _is_reparse(path: Path) -> bool:
    if path.is_symlink():
        return True
    try:
        return bool(int(getattr(path.lstat(), "st_file_attributes", 0)) & 0x400)
    except OSError:
        return False


def _directory_descriptor(path: Path, *, description: str) -> dict[str, int]:
    directory = _existing(path, directory=True, description=description)
    try:
        info = directory.stat(follow_symlinks=False)
    except OSError as error:
        raise ValueError(f"Unable to identify {description}: {directory}: {error}") from error
    return {
        "device": int(info.st_dev),
        "inode": int(info.st_ino),
        "file_attributes": int(getattr(info, "st_file_attributes", 0)),
    }


def _existing(path: Path, *, directory: bool, description: str) -> Path:
    raw = Path(os.path.abspath(os.fspath(Path(path).expanduser())))
    if not os.path.lexists(os.fspath(raw)):
        raise FileNotFoundError(f"Missing {description}: {raw}")
    current = raw
    while True:
        if os.path.lexists(os.fspath(current)) and _is_reparse(current):
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


def _freeze_file(path: Path, *, description: str) -> _FrozenFile:
    resolved = _existing(path, directory=False, description=description)
    try:
        with resolved.open("rb") as stream:
            before = _stat_identity(os.fstat(stream.fileno()))
            data = stream.read()
            after = _stat_identity(os.fstat(stream.fileno()))
    except OSError as error:
        raise ValueError(f"Unable to freeze {description}: {resolved}: {error}") from error
    if before != after or len(data) != before[2]:
        raise ValueError(f"{description} changed while its bytes were frozen")
    try:
        path_identity = _stat_identity(resolved.stat())
    except OSError as error:
        raise ValueError(f"Unable to restat {description}: {resolved}: {error}") from error
    if path_identity != before:
        raise ValueError(f"{description} path identity changed while bytes were frozen")
    return _FrozenFile(
        path=resolved,
        data=data,
        sha256=hashlib.sha256(data).hexdigest(),
        size_bytes=len(data),
        identity=before,
    )


def _require_frozen_current(snapshot: _FrozenFile, *, description: str) -> None:
    current = _freeze_file(snapshot.path, description=description)
    if (
        current.identity != snapshot.identity
        or current.size_bytes != snapshot.size_bytes
        or current.sha256 != snapshot.sha256
    ):
        raise ValueError(f"{description} changed after semantic verification")


def _fresh_directory(path: Path) -> Path:
    raw = Path(os.path.abspath(os.fspath(Path(path).expanduser())))
    if os.path.lexists(os.fspath(raw)):
        raise ValueError(f"Refusing to reuse exact8 output: {raw}")
    _existing(raw.parent, directory=True, description="exact8 output parent")
    return raw


def _require_formal_windows_output_anchor() -> None:
    if os.name != "nt":
        raise ValueError(
            "exact8 run requires Windows deny-delete directory handles; "
            "this platform cannot safely anchor the formal output"
        )


def _exact8_output_anchor_hook(
    checkpoint: str, *, parent: Path, output_root: Path
) -> None:
    """No-op concurrency hook used only by deterministic race tests."""


@dataclass
class _GuardedOutputDirectory:
    path: Path
    parent: Path
    parent_identity: tuple[int, int, int]
    parent_lease: Any
    output_lease: Any | None = None
    owns_parent_lease: bool = True
    closed: bool = False

    def require(self, checkpoint: str) -> None:
        from .recipient_multiview_overlay import (
            _directory_identity,
            _require_directory_lease_identity,
            _same_anchored_directory_entry,
        )

        if self.closed:
            raise ValueError(f"exact8 guarded output closed before {checkpoint}")
        try:
            _require_directory_lease_identity(self.parent_lease)
            if _directory_identity(self.parent) != self.parent_identity:
                raise ValueError("output parent path identity changed")
            if self.output_lease is not None:
                _require_directory_lease_identity(self.output_lease)
                if not _same_anchored_directory_entry(
                    self.parent_lease,
                    name=self.path.name,
                    expected=self.output_lease.identity,
                ):
                    raise ValueError("output entry identity changed")
                if _directory_identity(self.path) != self.output_lease.identity:
                    raise ValueError("output path identity changed")
        except (OSError, ValueError) as error:
            raise ValueError(
                f"exact8 guarded output identity changed at {checkpoint}: {error}"
            ) from error

    def create(self) -> Path:
        from .recipient_multiview_overlay import (
            _create_stage_lease,
            create_anchored_stage_directory,
        )

        self.require("before_output_freshness_check")
        if os.path.lexists(os.fspath(self.path)):
            raise ValueError(f"Refusing to reuse exact8 output: {self.path}")
        _exact8_output_anchor_hook(
            "post_check_pre_atomic_create",
            parent=self.parent,
            output_root=self.path,
        )
        self.require("immediately_before_output_creation")
        if os.path.lexists(os.fspath(self.path)):
            raise ValueError(f"Refusing to reuse exact8 output: {self.path}")
        try:
            if os.name == "nt":
                self.output_lease = create_anchored_stage_directory(
                    self.parent_lease,
                    name=self.path.name,
                )
            else:
                # Formal execution fails closed before reaching this branch.
                # It remains only for descriptor-anchored POSIX unit tests.
                self.output_lease = _create_stage_lease(
                    self.parent_lease,
                    stage=self.path,
                )
        except FileExistsError as error:
            raise ValueError(f"Refusing to reuse exact8 output: {self.path}") from error
        _exact8_output_anchor_hook(
            "post_atomic_create_pre_validation",
            parent=self.parent,
            output_root=self.path,
        )
        self.require("immediately_after_output_creation")
        return self.path

    def close(self) -> None:
        if self.closed:
            return
        if self.output_lease is not None:
            self.output_lease.close()
            self.output_lease = None
        if self.owns_parent_lease:
            self.parent_lease.close()
        self.closed = True


def _open_guarded_output_parent(path: Path) -> _GuardedOutputDirectory:
    from .recipient_multiview_overlay import _directory_identity, _open_directory_lease

    _require_formal_windows_output_anchor()
    output = _fresh_directory(path)
    if output.name in {"", ".", ".."}:
        raise ValueError("exact8 output must be one simple child of its existing parent")
    parent = _existing(
        output.parent,
        directory=True,
        description="exact8 output parent",
    )
    parent_identity = _directory_identity(parent)
    parent_lease = _open_directory_lease(parent, expected=parent_identity)
    anchor = _GuardedOutputDirectory(
        path=output,
        parent=parent,
        parent_identity=parent_identity,
        parent_lease=parent_lease,
    )
    try:
        anchor.require("after_output_parent_lease")
        if os.path.lexists(os.fspath(output)):
            raise ValueError(f"Refusing to reuse exact8 output: {output}")
    except BaseException:
        anchor.close()
        raise
    return anchor


def _open_guarded_child(
    parent_anchor: _GuardedOutputDirectory, path: Path
) -> _GuardedOutputDirectory:
    parent_anchor.require("before_guarded_child_parent_reuse")
    if parent_anchor.output_lease is None:
        raise ValueError("exact8 guarded child requires a created parent lease")
    child = _fresh_directory(path)
    if child.parent != parent_anchor.path or child.name in {"", ".", ".."}:
        raise ValueError("exact8 guarded child is not under its leased parent")
    anchor = _GuardedOutputDirectory(
        path=child,
        parent=parent_anchor.path,
        parent_identity=parent_anchor.output_lease.identity,
        parent_lease=parent_anchor.output_lease,
        owns_parent_lease=False,
    )
    anchor.require("after_guarded_child_parent_reuse")
    if os.path.lexists(os.fspath(child)):
        raise ValueError(f"Refusing to reuse exact8 output: {child}")
    return anchor


def _paths_overlap(left: Path, right: Path) -> bool:
    left_raw = os.path.normcase(os.path.abspath(os.fspath(left)))
    right_raw = os.path.normcase(os.path.abspath(os.fspath(right)))
    try:
        common = os.path.commonpath((left_raw, right_raw))
    except ValueError:
        return False
    return common in {left_raw, right_raw}


def _assert_output_disjoint(
    output: Path, *, protected_directories: Sequence[Path]
) -> None:
    for protected in protected_directories:
        if _paths_overlap(output, protected):
            raise ValueError(
                f"exact8 output overlaps protected authority directory: {protected}"
            )


def _binding(path: Path, *, description: str = "bound artifact") -> dict[str, object]:
    resolved = _existing(path, directory=False, description=description)
    return {
        "path": str(resolved),
        "sha256": _sha256(resolved),
        "size_bytes": resolved.stat().st_size,
    }


def _binding_from_frozen(snapshot: _FrozenFile) -> dict[str, object]:
    return {
        "path": str(snapshot.path),
        "sha256": snapshot.sha256,
        "size_bytes": snapshot.size_bytes,
    }


def _verify_frozen_binding(
    raw: object,
    *,
    snapshot: _FrozenFile,
    expected_path: Path,
    description: str,
) -> None:
    binding = _mapping(raw, f"{description} binding")
    if set(binding) != {"path", "sha256", "size_bytes"}:
        raise ValueError(f"{description} binding keys changed")
    bound_path = _existing(
        Path(str(binding.get("path"))), directory=False, description=description
    )
    _samefile(bound_path, expected_path, description)
    _samefile(bound_path, snapshot.path, f"{description} frozen path")
    _require_equal(binding.get("sha256"), snapshot.sha256, f"{description} SHA-256")
    _require_equal(
        binding.get("size_bytes"), snapshot.size_bytes, f"{description} size"
    )


def _binding_path(
    artifacts: Mapping[str, Any], name: str, *, description: str | None = None
) -> Path:
    raw = _mapping(artifacts.get(name), f"{name} binding")
    path_value = raw.get("path")
    sha256 = _require_hex(raw.get("sha256"), f"{name} SHA-256")
    size = raw.get("size_bytes")
    if not isinstance(path_value, str) or not path_value:
        raise ValueError(f"{name} binding has no path")
    if isinstance(size, bool) or not isinstance(size, int) or size < 0:
        raise ValueError(f"{name} binding has invalid size")
    path = _existing(
        Path(path_value), directory=False, description=description or name
    )
    if path.stat().st_size != size or _sha256(path) != sha256:
        raise ValueError(f"{name} binding changed")
    return path


def _samefile(left: Path, right: Path, description: str) -> None:
    try:
        same = os.path.samefile(left, right)
    except OSError as error:
        raise ValueError(f"Unable to verify {description} identity") from error
    if not same:
        raise ValueError(f"{description} is not the bound file")


def _write_json_no_clobber(path: Path, payload: Mapping[str, object]) -> None:
    encoded = json.dumps(
        dict(payload),
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        indent=2,
    ) + "\n"
    try:
        with path.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
    except FileExistsError as error:
        raise ValueError(f"Refusing to overwrite exact8 evidence: {path}") from error


def _code_paths() -> dict[str, Path]:
    package = Path(__file__).resolve().parent
    repository = package.parents[1]
    return {
        "code_exact8": Path(__file__).resolve(),
        "code_overlay": package / "recipient_multiview_overlay.py",
        "code_multiview_exporter": package / "recipient_multiview_teacher_export.py",
        "code_failure_attestor": package / "recipient_v14_failure_attestor.py",
        "code_candidate_source": package / "recipient_full_crop_candidate_source.py",
        "code_b8_closure": package / "recipient_full_crop_continuation.py",
        "code_full_crop_pilot": package / "recipient_full_crop_pilot.py",
        "code_seed_sanitizer": package / "recipient_full_crop_seed_sanitizer.py",
        "code_ocr_unified": package / "ocr_unified.py",
        "script_exact8": repository
        / "scripts"
        / "receipt-ocr-recipient-multiview-exact8-4090.ps1",
    }


def _verify_overlay(
    *,
    contract_path: Path,
    full_records: Path,
    blind_records: Path,
    blind_contract: Path,
    original_dataset_root: Path,
) -> Mapping[str, Any]:
    """Late-bind the independently implemented overlay verifier."""

    try:
        from .recipient_multiview_overlay import (
            FIXED2_PUBLICATION_AUTHORITY,
            verify_fixed2_overlay_contract,
        )
    except ImportError as error:  # pragma: no cover - exercised before integration
        raise RuntimeError(
            "recipient_multiview_overlay.verify_fixed2_overlay_contract is unavailable"
        ) from error
    # The independently sealed contract names the immutable export root.  The
    # verifier reopens the contract, validates its integrity seal, and then
    # requires this exact root while rebuilding every selected composite row.
    raw_contract = _strict_json(contract_path)
    multiview_value = raw_contract.get("multiview_root")
    if not isinstance(multiview_value, str) or not multiview_value:
        raise ValueError("fixed2 overlay contract has no multiview root")
    multiview_root = _existing(
        Path(multiview_value), directory=True, description="fixed2 multiview root"
    )
    payload = verify_fixed2_overlay_contract(
        contract_path=contract_path,
        blind_records=blind_records,
        blind_contract=blind_contract,
        multiview_root=multiview_root,
        expected_full_records=full_records,
        original_dataset_root=original_dataset_root,
    )
    overlay = _mapping(payload, "fixed2 overlay verification")
    _require_equal(overlay.get("kind"), OVERLAY_KIND, "fixed2 overlay kind")
    _require_equal(
        overlay.get("publication_authority"),
        FIXED2_PUBLICATION_AUTHORITY,
        "fixed2 overlay publication authority",
    )
    _require_equal(
        overlay.get("consumer_optimizer_input_ready"),
        True,
        "fixed2 overlay optimizer-input readiness",
    )
    _require_equal(overlay.get("analysis_only"), True, "fixed2 overlay analysis_only")
    _require_equal(
        overlay.get("production_route_authorized"),
        False,
        "fixed2 overlay production authorization",
    )
    _require_equal(overlay.get("test_opened"), False, "fixed2 overlay test_opened")
    _require_equal(
        overlay.get("selected_views"), SELECTED_VIEWS, "fixed2 selected views"
    )
    _require_equal(
        overlay.get("selector_mode"), SELECTOR_MODE, "fixed2 selector mode"
    )
    _require_equal(overlay.get("train_multiplier"), 1, "fixed2 train multiplier")
    _require_equal(overlay.get("val_unchanged"), True, "fixed2 val identity")
    _require_hex(overlay.get("overlay_subject_id"), "fixed2 overlay subject id")
    records_value = overlay.get("composite_records")
    root_value = overlay.get("composite_dataset_root")
    if not isinstance(records_value, str) or not records_value:
        raise ValueError("fixed2 overlay has no composite records path")
    if not isinstance(root_value, str) or not root_value:
        raise ValueError("fixed2 overlay has no composite dataset root")
    _existing(Path(records_value), directory=False, description="fixed2 composite records")
    _existing(Path(root_value), directory=True, description="fixed2 composite dataset root")
    artifacts = _mapping(overlay.get("artifacts"), "fixed2 overlay artifacts")
    for name in artifacts:
        _binding_path(artifacts, str(name), description=f"fixed2 overlay {name}")
    return overlay


def _count_rate_metric(
    value: Mapping[str, Any],
    *,
    expected_records: int,
    description: str,
) -> dict[str, object]:
    raw_records = value.get("records")
    raw_matches = value.get("exact_matches")
    if (
        isinstance(raw_records, bool)
        or not isinstance(raw_records, int)
        or raw_records != expected_records
    ):
        raise ValueError(
            f"{description} denominator must equal the frozen field val count "
            f"{expected_records}"
        )
    if (
        isinstance(raw_matches, bool)
        or not isinstance(raw_matches, int)
        or not 0 <= raw_matches <= raw_records
    ):
        raise ValueError(f"{description} exact_matches is invalid")
    exact = _finite_rate(value.get("exact_match"), f"{description} exact_match")
    if not math.isclose(exact, raw_matches / raw_records, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(f"{description} count/rate is inconsistent")
    return {"records": raw_records, "exact_matches": raw_matches, "exact_match": exact}


def _observed_count_rate_metric(
    value: Mapping[str, Any], *, description: str
) -> dict[str, object]:
    records = value.get("records")
    if isinstance(records, bool) or not isinstance(records, int) or records <= 0:
        raise ValueError(f"{description} denominator must be a positive integer")
    return _count_rate_metric(
        value, expected_records=records, description=description
    )


def _expected_checkpoint_policy(config: UnifiedReaderConfig) -> dict[str, object]:
    return _checkpoint_selection_policy(
        config=config,
        checkpoint_selection=CHECKPOINT_SELECTION_RECIPIENT_PRIORITY,
        checkpoint_min_amount_candidate_exact=AMOUNT_FLOOR,
        checkpoint_min_time_candidate_exact=TIME_FLOOR,
        checkpoint_min_payment_candidate_exact=PAYMENT_FLOOR,
        status_text_only_fine_tune=False,
    )


def _validation_from_epoch_record(
    record: Mapping[str, Any], *, description: str
) -> dict[str, object]:
    validation = {
        str(key)[4:]: value
        for key, value in record.items()
        if isinstance(key, str) and key.startswith("val_")
    }
    required = {
        "loss",
        "candidate_text_exact_match",
        "candidate_text_macro_exact_match",
        "candidate_text_by_field",
        "ctc_by_field",
        "status_non_success_to_success",
        "verifier_macro_exact_match",
    }
    missing = sorted(required - set(validation))
    if missing:
        raise ValueError(
            f"{description} lacks checkpoint-selection validation fields: {missing}"
        )
    return validation


def _assert_checkpoint_has_no_unsafe_claims(
    payload: Mapping[str, Any], *, description: str
) -> None:
    explicit = {
        "test_opened",
        "test_evaluated",
        "test_labels_used",
        "test_metrics_computed",
        "test_examples_emitted",
        "test_evaluation_authorized",
        "onnx_exported",
        "onnx_export_authorized",
        "production_route_authorized",
        "production_authorized",
        "production_ready",
        "warmstart_authorized",
        "same_route_authorized",
        "same_route_retry_authorized",
        "same_route_continuation_authorized",
        "failed_checkpoint_initialization_authorized",
    }

    def inspect(value: object, location: str) -> None:
        if isinstance(value, Mapping):
            for raw_key, child in value.items():
                key = str(raw_key).lower().replace("-", "_")
                unsafe_key = (
                    key in explicit
                    or key.startswith(("test_", "onnx_", "production_", "prod_"))
                    or "warmstart" in key
                    or key.startswith("same_route")
                    or "continuation" in key
                    or "retry" in key
                )
                if child is True and unsafe_key:
                    raise ValueError(
                        f"{description} contains unsafe true claim at {location}.{key}"
                    )
                inspect(child, f"{location}.{key}")
        elif isinstance(value, (list, tuple)):
            for index, child in enumerate(value):
                inspect(child, f"{location}[{index}]")

    inspect(payload, "$")


def _metric(record: Mapping[str, Any], field: str, description: str) -> dict[str, object]:
    fields = _mapping(record.get("val_candidate_text_by_field"), f"{description} fields")
    value = _mapping(fields.get(field), f"{description} {field}")
    return _count_rate_metric(
        value,
        expected_records=EXPECTED_RECIPIENT_VAL_RECORDS,
        description=f"{description} recipient",
    )


def _a8_data_label_proof(
    *, summary: Mapping[str, Any], best: Mapping[str, Any]
) -> dict[str, object]:
    initialization = _mapping(summary.get("initialization"), "A8 initialization")
    financial_policy = _mapping(
        initialization.get("financial_label_policy"), "A8 financial label policy"
    )
    summary_fields = {key: summary.get(key) for key in A8_SUMMARY_DATA_KEYS}
    for key in (
        "field_counts",
        "payment_oov_by_split",
        "recipient_oov_by_split",
        "recipient_sampling_policy",
        "status_text_oov_by_split",
    ):
        _mapping(summary_fields.get(key), f"A8 {key}")
    for key in A8_SUMMARY_DATA_KEYS:
        _json_equal(
            best.get(key),
            summary_fields.get(key),
            f"A8 best checkpoint data {key}",
        )
    best_initialization = _mapping(
        best.get("initialization"), "A8 best checkpoint initialization"
    )
    _json_equal(
        best_initialization.get("financial_label_policy"),
        financial_policy,
        "A8 best checkpoint financial label policy",
    )
    ordered_labels = {key: best.get(key) for key in A8_ORDERED_LABEL_KEYS}
    ordered_map_proof: dict[str, dict[str, object]] = {}
    for key in A8_ORDERED_MAP_KEYS:
        value = ordered_labels.get(key)
        if (
            not isinstance(value, Sequence)
            or isinstance(value, (str, bytes))
            or not value
            or any(not isinstance(item, str) or not item for item in value)
        ):
            raise ValueError(f"A8 ordered label map {key} is invalid")
        material = list(value)
        ordered_map_proof[key] = {
            "count": len(material),
            "sha256": _canonical_sha256(material),
        }
    for characters_key, hash_key in (
        ("status_text_characters", "status_text_charset_sha256"),
        ("recipient_characters", "recipient_charset_sha256"),
    ):
        expected = hashlib.sha256(
            "".join(ordered_labels[characters_key]).encode("utf-8")
        ).hexdigest()
        _require_equal(ordered_labels.get(hash_key), expected, f"A8 {hash_key}")
    payment_charset_sha256 = hashlib.sha256(
        "".join(ordered_labels["payment_characters"]).encode("utf-8")
    ).hexdigest()
    ordered_labels["payment_charset_sha256"] = payment_charset_sha256
    protocol_blank_indices = {
        "amount_blank_index": NUMERIC_BLANK_INDEX,
        "time_blank_index": NUMERIC_BLANK_INDEX,
        "payment_blank_index": PAYMENT_BLANK_INDEX,
    }
    blank_indices: dict[str, dict[str, object]] = {}
    for key in A8_BLANK_INDEX_KEYS:
        source, semantic = A8_BLANK_INDEX_PROOF[key]
        value = (
            protocol_blank_indices[key]
            if source == "fixed_protocol_constants"
            else best.get(key)
        )
        _require_equal(value, 0, f"A8 {semantic}")
        blank_indices[key] = {
            "source": source,
            "semantic": semantic,
            "value": value,
        }
    return {
        "summary_fields": summary_fields,
        "financial_label_policy": dict(financial_policy),
        "ordered_labels": ordered_labels,
        "ordered_label_maps": ordered_map_proof,
        "blank_indices": blank_indices,
    }


def _a8_baseline(pilot: Mapping[str, Any], *, torch: Any) -> dict[str, object]:
    artifacts = _mapping(pilot.get("artifacts"), "A8 artifacts")
    summary_path = _binding_path(
        artifacts,
        "candidate_training_summary",
        description="A8 training summary",
    )
    summary = _strict_json(summary_path)
    best_path = _binding_path(
        artifacts,
        "candidate_best_checkpoint",
        description="A8 best checkpoint",
    )
    best_payload = _load_checkpoint(best_path, torch=torch)
    raw_records = summary.get("records")
    if not isinstance(raw_records, Sequence) or isinstance(raw_records, (str, bytes)):
        raise ValueError("A8 summary has invalid epoch records")
    records = [_mapping(record, "A8 epoch record") for record in raw_records]
    if [record.get("epoch") for record in records] != list(range(1, 9)):
        raise ValueError("A8 summary must contain ordered epochs 1 through 8")
    by_epoch = {
        int(record["epoch"]): _metric(record, "recipient_field", f"A8 epoch {record['epoch']}")
        for record in records
    }
    denominators: dict[str, int] = {}
    for field in ("amount", "time", "payment_method_field"):
        observed: list[int] = []
        for record in records:
            epoch = int(record["epoch"])
            fields = _mapping(
                record.get("val_candidate_text_by_field"), f"A8 epoch {epoch} fields"
            )
            metric = _observed_count_rate_metric(
                _mapping(fields.get(field), f"A8 epoch {epoch} {field}"),
                description=f"A8 epoch {epoch} {field}",
            )
            observed.append(int(metric["records"]))
        expected = EXPECTED_CANDIDATE_VAL_RECORDS[field]
        if observed != [expected] * FIXED_EPOCHS:
            raise ValueError(
                f"A8 {field} candidate denominator changed; expected {expected} every epoch"
            )
        denominators[field] = expected
    status_denominators: list[int] = []
    for record in records:
        epoch = int(record["epoch"])
        status_fields = _mapping(
            record.get("val_ctc_by_field"), f"A8 epoch {epoch} CTC fields"
        )
        status = _observed_count_rate_metric(
            _mapping(
                status_fields.get("transfer_status"), f"A8 epoch {epoch} status"
            ),
            description=f"A8 epoch {epoch} visible status",
        )
        status_denominators.append(int(status["records"]))
    if len(set(status_denominators)) != 1:
        raise ValueError("A8 visible-status CTC denominator changed across epochs")
    denominators["transfer_status"] = status_denominators[0]
    denominators["recipient_field"] = EXPECTED_RECIPIENT_VAL_RECORDS
    best_epoch = summary.get("best_checkpoint_epoch")
    if (
        isinstance(best_epoch, bool)
        or not isinstance(best_epoch, int)
        or best_epoch not in by_epoch
    ):
        raise ValueError("A8 summary has invalid best checkpoint epoch")
    best = int(by_epoch[best_epoch]["exact_matches"])
    if best != max(int(value["exact_matches"]) for value in by_epoch.values()):
        raise ValueError("A8 best checkpoint is not recipient-optimal")
    return {
        "best_epoch": best_epoch,
        "best_matches": best,
        "epoch4_matches": int(by_epoch[4]["exact_matches"]),
        "epoch8_matches": int(by_epoch[8]["exact_matches"]),
        "candidate_denominators": denominators,
        "data_label_proof": _a8_data_label_proof(
            summary=summary, best=best_payload
        ),
    }


def _target_config(source_checkpoint: Path, *, torch: Any) -> UnifiedReaderConfig:
    payload = _load_checkpoint(source_checkpoint, torch=torch)
    source = _checkpoint_config(payload)
    target = UnifiedReaderConfig(
        **{
            **asdict(source),
            "recipient_branch_channels": 16,
            "recipient_hidden_size": 192,
            "recipient_open_text_layers": 4,
            "recipient_open_text_heads": 8,
            "recipient_open_text_feedforward": 1536,
            "recipient_open_text_dropout": 0.10,
            "recipient_backbone": REQUIRED_BACKBONE,
        }
    )
    source.validate()
    target.validate()
    _validate_recipient_visual_context_reinit_config(source, target)
    return target


def inspect_exact8_subject(
    *,
    full_records: Path,
    original_dataset_root: Path,
    full_crop_pilot_root: Path,
    source_contract_path: Path,
    candidate_pilot_evidence_path: Path,
    failure_evidence_path: Path,
    failure_attempt_registry: Path,
    overlay_contract_path: Path,
    torch: Any | None = None,
) -> dict[str, object]:
    """Reopen every authority and return the path-independent exact8 subject."""

    if torch is None:
        torch, _ = _require_torch()
    full = _existing(full_records, directory=False, description="full v13 manifest")
    dataset = _existing(
        original_dataset_root,
        directory=True,
        description="original recipient dataset root",
    )
    pilot_root = _existing(
        full_crop_pilot_root, directory=True, description="full-crop pilot root"
    )
    source_file = _existing(
        source_contract_path, directory=False, description="full-crop source contract"
    )
    a8_file = _existing(
        candidate_pilot_evidence_path,
        directory=False,
        description="A8 candidate-pilot evidence",
    )
    failure_file = _existing(
        failure_evidence_path,
        directory=False,
        description="fresh60 failure evidence",
    )
    failure_registry = _existing(
        failure_attempt_registry,
        directory=True,
        description="fresh60 attempt registry",
    )
    overlay_file = _existing(
        overlay_contract_path,
        directory=False,
        description="fixed2 overlay contract",
    )

    source = verify_full_crop_candidate_source(
        pilot_root=pilot_root,
        contract_path=source_file,
        full_records=full,
        torch=torch,
    )
    a8 = verify_residual_candidate_pilot(
        evidence_path=a8_file,
        source_contract_path=source_file,
        full_records=full,
        torch=torch,
    )
    failure = verify_fresh60_failure(
        evidence_path=failure_file,
        source_contract_path=source_file,
        candidate_pilot_evidence_path=a8_file,
        full_records=full,
        attempt_registry=failure_registry,
        torch=torch,
    )
    _require_equal(source.get("kind"), SOURCE_KIND, "source contract kind")
    _require_equal(a8.get("kind"), CANDIDATE_PILOT_KIND, "A8 evidence kind")
    _require_equal(failure.get("kind"), FAILURE_KIND, "failure evidence kind")
    _require_equal(failure.get("analysis_only"), True, "failure analysis_only")
    _require_equal(
        failure.get("new_view_pilot_authority"), True, "failure new-view authority"
    )
    _require_equal(failure.get("decision"), FAILURE_DECISION, "failure decision")
    _require_equal(
        failure.get("authorization"), FAILURE_AUTHORIZATION, "failure authorization"
    )
    for key in (
        "production_route_authorized",
        "same_route_retry_authorized",
        "same_route_continuation_authorized",
        "warmstart_authorized",
        "failed_checkpoint_initialization_authorized",
        "onnx_export_authorized",
        "test_evaluation_authorized",
        "test_opened",
        "onnx_exported",
    ):
        _require_equal(failure.get(key), False, f"failure {key}")
    scope = _mapping(failure.get("authorization_scope"), "failure authorization scope")
    _require_equal(scope.get("epochs"), FIXED_EPOCHS, "failure-authorized epochs")
    _require_equal(
        scope.get("source_initialization"),
        "same_attested_legacy_source_fresh_visual_context_reinit",
        "failure-authorized initialization",
    )
    _require_equal(
        scope.get("training_data_view"),
        "must_differ_from_failed_standard_full_crop_view",
        "failure-authorized data view",
    )
    _require_equal(
        scope.get("failed_best_checkpoint_use"), "forbidden", "failed best use"
    )
    _require_equal(
        scope.get("failed_last_checkpoint_use"), "forbidden", "failed last use"
    )

    source_subject = _require_hex(source.get("source_subject_id"), "source subject id")
    _require_equal(
        source_subject, ATTESTED_SOURCE_SUBJECT_ID, "fixed source subject identity"
    )
    a8_source = _require_hex(a8.get("source_subject_id"), "A8 source subject id")
    _require_equal(a8_source, source_subject, "A8 source subject")
    a8_subject = _require_hex(a8.get("candidate_pilot_subject_id"), "A8 subject id")
    _require_equal(a8_subject, ATTESTED_A8_SUBJECT_ID, "fixed A8 subject identity")
    _require_equal(
        failure.get("source_subject_id"), source_subject, "failure source subject"
    )
    _require_equal(
        failure.get("candidate_pilot_subject_id"), a8_subject, "failure A8 subject"
    )
    failure_subject = _require_hex(
        failure.get("failure_subject_id"), "failure subject id"
    )

    source_artifacts = _mapping(source.get("artifacts"), "source artifacts")
    source_checkpoint = _binding_path(
        source_artifacts, "best_checkpoint", description="original pilot best checkpoint"
    )
    source_full = _binding_path(
        source_artifacts, "full_manifest", description="source full manifest"
    )
    _samefile(full, source_full, "source full manifest")
    b8_closure, _b8_source_payload, _b8_paths = _recompute_pilot_closure(
        pilot_root, torch=torch
    )
    b8_artifacts = _mapping(b8_closure.get("artifacts"), "B8 source closure artifacts")
    b8_checkpoint = _binding_path(
        b8_artifacts,
        "source_best_checkpoint",
        description="B8-closed original pilot best",
    )
    b8_full = _binding_path(
        b8_artifacts, "full_manifest", description="B8-closed full manifest"
    )
    _samefile(source_checkpoint, b8_checkpoint, "candidate/B8 source checkpoint")
    _samefile(full, b8_full, "candidate/B8 full manifest")

    a8_artifacts = _mapping(a8.get("artifacts"), "A8 artifacts")
    blind_records = _binding_path(
        a8_artifacts,
        "candidate_blind_manifest",
        description="A8 blind manifest",
    )
    blind_contract = _binding_path(
        a8_artifacts,
        "candidate_blind_contract",
        description="A8 blind contract",
    )
    overlay = _verify_overlay(
        contract_path=overlay_file,
        full_records=full,
        blind_records=blind_records,
        blind_contract=blind_contract,
        original_dataset_root=dataset,
    )
    overlay_subject = _require_hex(
        overlay.get("overlay_subject_id"), "overlay subject id"
    )
    baseline = _a8_baseline(a8, torch=torch)
    target = _target_config(source_checkpoint, torch=torch)

    route_material = {
        "domain": ROUTE_DOMAIN,
        "source_subject_id": source_subject,
        "candidate_pilot_subject_id": a8_subject,
        "b8_fixed_source_subject_id": FIXED_SOURCE_SUBJECT_ID,
        "failure_subject_id": failure_subject,
        "overlay_subject_id": overlay_subject,
        "selector_mode": SELECTOR_MODE,
        "selected_views": SELECTED_VIEWS,
    }
    route_subject_id = _canonical_sha256(route_material)
    code = {
        name: _binding(path, description=name) for name, path in _code_paths().items()
    }
    guard_paths: set[str] = {
        str(full),
        str(source_file),
        str(a8_file),
        str(failure_file),
        str(overlay_file),
        str(source_checkpoint),
        str(blind_records),
        str(blind_contract),
        str(Path(str(overlay["composite_records"])).resolve()),
    }
    binding_collections: list[Mapping[str, Any]] = [
        source_artifacts,
        a8_artifacts,
        b8_artifacts,
        _mapping(failure.get("artifacts"), "failure artifacts"),
        _mapping(failure.get("code"), "failure code"),
    ]
    for owner, name in ((source, "source code"), (a8, "A8 code")):
        raw_collection = owner.get("code")
        if raw_collection is not None:
            binding_collections.append(_mapping(raw_collection, name))
    for collection in binding_collections:
        for name in collection:
            guard_paths.add(str(_binding_path(collection, str(name))))
    for raw in _mapping(overlay.get("artifacts"), "overlay artifacts").values():
        binding = _mapping(raw, "overlay artifact binding")
        path_value = binding.get("path")
        if isinstance(path_value, str) and path_value:
            guard_paths.add(
                str(
                    _existing(
                        Path(path_value),
                        directory=False,
                        description="overlay artifact",
                    )
                )
            )
    guard_paths.update(str(Path(str(binding["path"]))) for binding in code.values())
    guard_directories = sorted(
        {
            str(dataset),
            str(Path(str(overlay["multiview_root"])).resolve()),
        }
    )
    guard_directory_identities = [
        {
            "path": path,
            **_directory_descriptor(
                Path(path), description="exact8 guarded overlay data directory"
            ),
        }
        for path in guard_directories
    ]

    return {
        "schema_version": SCHEMA_VERSION,
        "kind": INSPECTION_KIND,
        "analysis_only": True,
        "production_route_authorized": False,
        "test_opened": False,
        "onnx_exported": False,
        "route_subject_id": route_subject_id,
        "attempt_id": route_subject_id,
        "route_material": route_material,
        "source_subject_id": source_subject,
        "candidate_pilot_subject_id": a8_subject,
        "failure_subject_id": failure_subject,
        "overlay_subject_id": overlay_subject,
        "source_checkpoint": str(source_checkpoint),
        "source_checkpoint_sha256": _sha256(source_checkpoint),
        "full_manifest_sha256": _sha256(full),
        "composite_records": str(Path(str(overlay["composite_records"])).resolve()),
        "composite_dataset_root": str(
            Path(str(overlay["composite_dataset_root"])).resolve()
        ),
        "overlay_contract_sha256": _sha256(overlay_file),
        "baseline": baseline,
        "target_config": asdict(target),
        "fixed_gates": {
            "recipient_denominator": EXPECTED_RECIPIENT_VAL_RECORDS,
            "minimum_absolute_best_matches": MINIMUM_BEST_MATCHES,
            "minimum_best_gain_over_A8_matches": BASELINE_GAIN_MATCHES,
            "minimum_epoch8_gain_over_A8_matches": BASELINE_GAIN_MATCHES,
            "minimum_epoch4_to_8_gain_matches": MINIMUM_EPOCH4_TO_8_GAIN_MATCHES,
            "maximum_best_to_epoch8_gap_matches": MAXIMUM_BEST_TO_EPOCH8_GAP_MATCHES,
            "strict_fresh60_recipient_target_matches": STRICT_RECIPIENT_PASS_MATCHES,
            "amount_floor": AMOUNT_FLOOR,
            "time_floor": TIME_FLOOR,
            "payment_floor": PAYMENT_FLOOR,
            "visible_status_floor": STATUS_TEXT_FLOOR,
            "unsafe_status_max": 0,
        },
        "code": code,
        "guard_paths": sorted(guard_paths),
        "guard_directories": guard_directories,
        "guard_directory_identities": guard_directory_identities,
    }


def _fixed_training_args() -> dict[str, object]:
    return {
        "device": "cuda:0",
        "epochs": FIXED_EPOCHS,
        "batch_size": FIXED_BATCH_SIZE,
        "learning_rate": FIXED_LEARNING_RATE,
        "weight_decay": FIXED_WEIGHT_DECAY,
        "validation_every": 1,
        "seed": FIXED_SEED,
        "num_workers": FIXED_NUM_WORKERS,
        "prefetch_factor": FIXED_PREFETCH_FACTOR,
        "persistent_workers": True,
        "train_progress_every": FIXED_PROGRESS_EVERY,
        "cuda_tf32": True,
        "cudnn_benchmark": True,
        "recipient_train_augmentation": FIXED_AUGMENTATION,
        "recipient_train_splits": ["train"],
        "recipient_only_fine_tune": True,
        "init_checkpoint_mode": INIT_CHECKPOINT_MODE_RECIPIENT_VISUAL_CONTEXT_REINIT,
        "checkpoint_selection": CHECKPOINT_SELECTION_RECIPIENT_PRIORITY,
        "recipient_low_confidence_threshold": 0.95,
        "recipient_low_confidence_loss_weight": 0.50,
        "recipient_confidence_curriculum_epochs": 10,
        "recipient_tail_rare_character_max_support": 3,
        "recipient_tail_rare_character_loss_weight": 1.5,
        "recipient_tail_long_text_min_length": 9,
        "recipient_tail_long_text_loss_weight": 1.5,
        "ctc_loss_weight": 1.0,
        "structured_loss_weight": 1.0,
        "payment_bank_prefix_min_support": 3,
        "checkpoint_min_amount_candidate_exact": AMOUNT_FLOOR,
        "checkpoint_min_time_candidate_exact": TIME_FLOOR,
        "checkpoint_min_payment_candidate_exact": PAYMENT_FLOOR,
    }


def _recipe(inspection: Mapping[str, Any]) -> dict[str, object]:
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": RECIPE_KIND,
        "analysis_only": True,
        "production_route_authorized": False,
        "test_opened": False,
        "onnx_exported": False,
        "route_subject_id": inspection["route_subject_id"],
        "source_subject_id": inspection["source_subject_id"],
        "candidate_pilot_subject_id": inspection["candidate_pilot_subject_id"],
        "failure_subject_id": inspection["failure_subject_id"],
        "overlay_subject_id": inspection["overlay_subject_id"],
        "source_checkpoint_sha256": inspection["source_checkpoint_sha256"],
        "full_manifest_sha256": inspection["full_manifest_sha256"],
        "overlay_contract_sha256": inspection["overlay_contract_sha256"],
        "selector_mode": SELECTOR_MODE,
        "selected_views": SELECTED_VIEWS,
        "train_multiplier": 1,
        "val_unchanged": True,
        "baseline": inspection["baseline"],
        "fixed_gates": inspection["fixed_gates"],
        "training_args": _fixed_training_args(),
        "code": inspection["code"],
    }


def _windows_close_native_handle(handle: int) -> None:
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    close_handle = kernel32.CloseHandle
    close_handle.argtypes = (ctypes.c_void_p,)
    close_handle.restype = ctypes.c_int32
    close_handle(ctypes.c_void_p(handle))


def _windows_known_program_data_path() -> Path:
    """Resolve ProgramData from the Windows Known Folder authority only."""

    shell32 = ctypes.WinDLL("shell32", use_last_error=True)
    ole32 = ctypes.WinDLL("ole32", use_last_error=True)
    co_initialize = ole32.CoInitializeEx
    co_initialize.argtypes = (ctypes.c_void_p, ctypes.c_uint32)
    co_initialize.restype = ctypes.c_int32
    initialization = int(co_initialize(None, 0x2))  # COINIT_APARTMENTTHREADED
    initialized_here = initialization in {0, 1}  # S_OK / S_FALSE
    if initialization < 0 and initialization != _WINDOWS_RPC_E_CHANGED_MODE:
        raise ValueError(
            "unable to initialize Windows Known Folder lookup: "
            f"HRESULT=0x{initialization & 0xFFFFFFFF:08x}"
        )

    parts = _WINDOWS_PROGRAM_DATA_GUID
    guid = _WindowsGuid(parts[0], parts[1], parts[2], (ctypes.c_ubyte * 8)(*parts[3]))
    raw_path = ctypes.c_void_p()
    try:
        get_known_folder = shell32.SHGetKnownFolderPath
        get_known_folder.argtypes = (
            ctypes.POINTER(_WindowsGuid),
            ctypes.c_uint32,
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_void_p),
        )
        get_known_folder.restype = ctypes.c_int32
        result = int(
            get_known_folder(
                ctypes.byref(guid),
                0,
                None,
                ctypes.byref(raw_path),
            )
        )
        if result < 0:
            raise ValueError(
                "Windows FOLDERID_ProgramData lookup failed: "
                f"HRESULT=0x{result & 0xFFFFFFFF:08x}"
            )
        if raw_path.value is None:
            raise ValueError("Windows FOLDERID_ProgramData returned no path")
        value = ctypes.wstring_at(raw_path.value)
        if not value or not value.strip():
            raise ValueError("Windows FOLDERID_ProgramData returned an empty path")
        # Do not resolve here: _existing must inspect the original chain for
        # junctions/reparse points before it anchors the registry directory.
        return Path(os.path.abspath(value))
    finally:
        if raw_path.value is not None:
            co_task_mem_free = ole32.CoTaskMemFree
            co_task_mem_free.argtypes = (ctypes.c_void_p,)
            co_task_mem_free.restype = None
            co_task_mem_free(ctypes.c_void_p(raw_path.value))
        if initialized_here:
            co_uninitialize = ole32.CoUninitialize
            co_uninitialize.argtypes = ()
            co_uninitialize.restype = None
            co_uninitialize()


def _common_application_data_path() -> Path | None:
    if os.name != "nt":
        return None
    return _windows_known_program_data_path()


@dataclass
class _WindowsAclSubject:
    sid: bytes
    impersonation_token: int
    closed: bool = False

    def close(self) -> None:
        if self.closed:
            return
        _windows_close_native_handle(self.impersonation_token)
        self.closed = True


def _windows_current_acl_subject() -> _WindowsAclSubject:
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    advapi32 = ctypes.WinDLL("advapi32", use_last_error=True)
    get_current_process = kernel32.GetCurrentProcess
    get_current_process.argtypes = ()
    get_current_process.restype = ctypes.c_void_p
    open_process_token = advapi32.OpenProcessToken
    open_process_token.argtypes = (
        ctypes.c_void_p,
        ctypes.c_uint32,
        ctypes.POINTER(ctypes.c_void_p),
    )
    open_process_token.restype = ctypes.c_int32
    process_token = ctypes.c_void_p()
    if not open_process_token(
        get_current_process(),
        _WINDOWS_TOKEN_QUERY | _WINDOWS_TOKEN_DUPLICATE,
        ctypes.byref(process_token),
    ):
        error_number = ctypes.get_last_error()
        raise OSError(error_number, os.strerror(error_number))

    duplicate_token = ctypes.c_void_p()
    try:
        get_token_information = advapi32.GetTokenInformation
        get_token_information.argtypes = (
            ctypes.c_void_p,
            ctypes.c_uint32,
            ctypes.c_void_p,
            ctypes.c_uint32,
            ctypes.POINTER(ctypes.c_uint32),
        )
        get_token_information.restype = ctypes.c_int32
        required = ctypes.c_uint32()
        get_token_information(
            process_token,
            _WINDOWS_TOKEN_USER,
            None,
            0,
            ctypes.byref(required),
        )
        if required.value < ctypes.sizeof(_WindowsTokenUser):
            error_number = ctypes.get_last_error()
            raise OSError(error_number, "unable to size current Windows token user")
        token_buffer = ctypes.create_string_buffer(required.value)
        if not get_token_information(
            process_token,
            _WINDOWS_TOKEN_USER,
            token_buffer,
            required.value,
            ctypes.byref(required),
        ):
            error_number = ctypes.get_last_error()
            raise OSError(error_number, os.strerror(error_number))
        token_user = ctypes.cast(
            token_buffer, ctypes.POINTER(_WindowsTokenUser)
        ).contents
        if token_user.user.sid is None:
            raise ValueError("current Windows token has no user SID")
        is_valid_sid = advapi32.IsValidSid
        is_valid_sid.argtypes = (ctypes.c_void_p,)
        is_valid_sid.restype = ctypes.c_int32
        if not is_valid_sid(token_user.user.sid):
            raise ValueError("current Windows token user SID is invalid")
        get_length_sid = advapi32.GetLengthSid
        get_length_sid.argtypes = (ctypes.c_void_p,)
        get_length_sid.restype = ctypes.c_uint32
        sid_length = int(get_length_sid(token_user.user.sid))
        if sid_length <= 0 or sid_length > required.value:
            raise ValueError("current Windows token user SID length is invalid")
        sid = ctypes.string_at(token_user.user.sid, sid_length)

        duplicate = advapi32.DuplicateToken
        duplicate.argtypes = (
            ctypes.c_void_p,
            ctypes.c_uint32,
            ctypes.POINTER(ctypes.c_void_p),
        )
        duplicate.restype = ctypes.c_int32
        if not duplicate(
            process_token,
            _WINDOWS_SECURITY_IMPERSONATION,
            ctypes.byref(duplicate_token),
        ):
            error_number = ctypes.get_last_error()
            raise OSError(error_number, os.strerror(error_number))
        if duplicate_token.value is None:
            raise ValueError("Windows DuplicateToken returned no impersonation token")
        return _WindowsAclSubject(
            sid=sid,
            impersonation_token=int(duplicate_token.value),
        )
    except BaseException:
        if duplicate_token.value is not None:
            _windows_close_native_handle(int(duplicate_token.value))
        raise
    finally:
        if process_token.value is not None:
            _windows_close_native_handle(int(process_token.value))


def _windows_security_descriptor(handle: int) -> tuple[int, int]:
    advapi32 = ctypes.WinDLL("advapi32", use_last_error=True)
    owner = ctypes.c_void_p()
    group = ctypes.c_void_p()
    dacl = ctypes.c_void_p()
    security_descriptor = ctypes.c_void_p()
    get_security_info = advapi32.GetSecurityInfo
    get_security_info.argtypes = (
        ctypes.c_void_p,
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_void_p),
    )
    get_security_info.restype = ctypes.c_uint32
    result = int(
        get_security_info(
            ctypes.c_void_p(handle),
            _WINDOWS_SE_FILE_OBJECT,
            _WINDOWS_OWNER_SECURITY_INFORMATION
            | _WINDOWS_GROUP_SECURITY_INFORMATION
            | _WINDOWS_DACL_SECURITY_INFORMATION,
            ctypes.byref(owner),
            ctypes.byref(group),
            ctypes.byref(dacl),
            None,
            ctypes.byref(security_descriptor),
        )
    )
    if result != 0:
        if security_descriptor.value is not None:
            _windows_free_security_descriptor(int(security_descriptor.value))
        raise OSError(result, os.strerror(result))
    if security_descriptor.value is None:
        raise ValueError("Windows GetSecurityInfo returned no security descriptor")
    if owner.value is None or group.value is None:
        _windows_free_security_descriptor(int(security_descriptor.value))
        raise ValueError("Windows one-shot object security descriptor lacks owner/group")
    if dacl.value is None:
        _windows_free_security_descriptor(int(security_descriptor.value))
        raise ValueError("Windows one-shot object has a null DACL")
    return int(security_descriptor.value), int(dacl.value)


def _windows_free_security_descriptor(security_descriptor: int) -> None:
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    local_free = kernel32.LocalFree
    local_free.argtypes = (ctypes.c_void_p,)
    local_free.restype = ctypes.c_void_p
    local_free(ctypes.c_void_p(security_descriptor))


def _windows_current_sid_deny_aces(
    dacl: int, *, current_sid: bytes
) -> list[tuple[int, int]]:
    advapi32 = ctypes.WinDLL("advapi32", use_last_error=True)
    is_valid_acl = advapi32.IsValidAcl
    is_valid_acl.argtypes = (ctypes.c_void_p,)
    is_valid_acl.restype = ctypes.c_int32
    if not is_valid_acl(ctypes.c_void_p(dacl)):
        raise ValueError("Windows one-shot DACL is invalid")
    information = _WindowsAclSizeInformation()
    get_acl_information = advapi32.GetAclInformation
    get_acl_information.argtypes = (
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_uint32,
        ctypes.c_uint32,
    )
    get_acl_information.restype = ctypes.c_int32
    if not get_acl_information(
        ctypes.c_void_p(dacl),
        ctypes.byref(information),
        ctypes.sizeof(information),
        _WINDOWS_ACL_SIZE_INFORMATION,
    ):
        error_number = ctypes.get_last_error()
        raise OSError(error_number, os.strerror(error_number))
    current_sid_buffer = ctypes.create_string_buffer(current_sid)
    equal_sid = advapi32.EqualSid
    equal_sid.argtypes = (ctypes.c_void_p, ctypes.c_void_p)
    equal_sid.restype = ctypes.c_int32
    is_valid_sid = advapi32.IsValidSid
    is_valid_sid.argtypes = (ctypes.c_void_p,)
    is_valid_sid.restype = ctypes.c_int32
    get_length_sid = advapi32.GetLengthSid
    get_length_sid.argtypes = (ctypes.c_void_p,)
    get_length_sid.restype = ctypes.c_uint32
    get_ace = advapi32.GetAce
    get_ace.argtypes = (
        ctypes.c_void_p,
        ctypes.c_uint32,
        ctypes.POINTER(ctypes.c_void_p),
    )
    get_ace.restype = ctypes.c_int32
    matches: list[tuple[int, int]] = []
    for index in range(int(information.ace_count)):
        ace_pointer = ctypes.c_void_p()
        if not get_ace(
            ctypes.c_void_p(dacl), index, ctypes.byref(ace_pointer)
        ):
            error_number = ctypes.get_last_error()
            raise OSError(error_number, os.strerror(error_number))
        if ace_pointer.value is None:
            raise ValueError("Windows GetAce returned a null ACE")
        header = ctypes.cast(
            ace_pointer, ctypes.POINTER(_WindowsAceHeader)
        ).contents
        if header.ace_type != _WINDOWS_ACCESS_DENIED_ACE_TYPE:
            continue
        if header.ace_size < ctypes.sizeof(_WindowsAccessDeniedAce):
            raise ValueError("Windows access-denied ACE is truncated")
        denied = ctypes.cast(
            ace_pointer, ctypes.POINTER(_WindowsAccessDeniedAce)
        ).contents
        sid_address = int(ace_pointer.value) + _WindowsAccessDeniedAce.sid_start.offset
        sid_pointer = ctypes.c_void_p(sid_address)
        if not is_valid_sid(sid_pointer):
            raise ValueError("Windows access-denied ACE has an invalid SID")
        sid_length = int(get_length_sid(sid_pointer))
        if (
            sid_length <= 0
            or _WindowsAccessDeniedAce.sid_start.offset + sid_length
            > int(header.ace_size)
        ):
            raise ValueError("Windows access-denied ACE SID exceeds the ACE")
        if equal_sid(
            sid_pointer,
            ctypes.cast(current_sid_buffer, ctypes.c_void_p),
        ):
            matches.append((int(denied.mask), int(header.ace_flags)))
    return matches


def _windows_access_allowed(
    security_descriptor: int,
    *,
    impersonation_token: int,
    desired_access: int,
) -> bool:
    advapi32 = ctypes.WinDLL("advapi32", use_last_error=True)
    access_check = advapi32.AccessCheck
    access_check.argtypes = (
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_uint32,
        ctypes.POINTER(_WindowsGenericMapping),
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.POINTER(ctypes.c_int32),
    )
    access_check.restype = ctypes.c_int32
    mapping = _WindowsGenericMapping(
        _WINDOWS_FILE_GENERIC_READ,
        _WINDOWS_FILE_GENERIC_WRITE,
        _WINDOWS_FILE_GENERIC_EXECUTE,
        _WINDOWS_FILE_ALL_ACCESS,
    )
    privilege_buffer = ctypes.create_string_buffer(65536)
    privilege_length = ctypes.c_uint32(len(privilege_buffer))
    granted_access = ctypes.c_uint32()
    access_status = ctypes.c_int32()
    # AccessCheck's return value reports whether the API completed.  The
    # independent AccessStatus out-parameter is the allow/deny decision.
    if not access_check(
        ctypes.c_void_p(security_descriptor),
        ctypes.c_void_p(impersonation_token),
        desired_access,
        ctypes.byref(mapping),
        privilege_buffer,
        ctypes.byref(privilege_length),
        ctypes.byref(granted_access),
        ctypes.byref(access_status),
    ):
        error_number = ctypes.get_last_error()
        raise OSError(error_number, os.strerror(error_number))
    return bool(access_status.value)


def _require_windows_acl_evidence(
    *,
    description: str,
    deny_aces: Sequence[tuple[int, int]],
    required_mask: int | None,
    required_flags: int,
    forbidden_flags: int,
    effective_access: Mapping[int, bool],
) -> None:
    if required_mask is not None and not any(
        mask & required_mask == required_mask
        and flags & required_flags == required_flags
        and not flags & forbidden_flags
        for mask, flags in deny_aces
    ):
        raise ValueError(
            f"{description} lacks the required current-SID deny ACE"
        )
    for access_mask, allowed in effective_access.items():
        if allowed:
            raise ValueError(
                f"{description} does not effectively deny access 0x{access_mask:08x}"
            )


def _require_windows_acl_policy(
    handle: int,
    *,
    description: str,
    required_mask: int | None,
    required_flags: int,
    forbidden_flags: int,
    effective_denied_accesses: Sequence[int],
) -> None:
    subject = _windows_current_acl_subject()
    security_descriptor: int | None = None
    try:
        security_descriptor, dacl = _windows_security_descriptor(handle)
        deny_aces = _windows_current_sid_deny_aces(
            dacl, current_sid=subject.sid
        )
        effective_access = {
            access: _windows_access_allowed(
                security_descriptor,
                impersonation_token=subject.impersonation_token,
                desired_access=access,
            )
            for access in effective_denied_accesses
        }
        _require_windows_acl_evidence(
            description=description,
            deny_aces=deny_aces,
            required_mask=required_mask,
            required_flags=required_flags,
            forbidden_flags=forbidden_flags,
            effective_access=effective_access,
        )
    finally:
        if security_descriptor is not None:
            _windows_free_security_descriptor(security_descriptor)
        subject.close()


def _require_attempt_registry_acl(registry_lease: Any) -> None:
    if os.name != "nt":
        return
    handle = getattr(registry_lease, "windows_handle", None)
    if handle is None:
        raise ValueError("exact8 attempt registry has no Windows ACL handle")
    _require_windows_acl_policy(
        int(handle),
        description="exact8 attempt registry DACL",
        required_mask=_WINDOWS_DELETE | _WINDOWS_FILE_DELETE_CHILD,
        required_flags=_WINDOWS_OBJECT_INHERIT_ACE | _WINDOWS_CONTAINER_INHERIT_ACE,
        forbidden_flags=_WINDOWS_INHERITED_ACE | _WINDOWS_INHERIT_ONLY_ACE,
        effective_denied_accesses=(_WINDOWS_DELETE, _WINDOWS_FILE_DELETE_CHILD),
    )


def _require_attempt_program_data_acl(program_data_lease: Any) -> None:
    if os.name != "nt":
        return
    handle = getattr(program_data_lease, "windows_handle", None)
    if handle is None:
        raise ValueError("exact8 ProgramData has no Windows ACL handle")
    _require_windows_acl_policy(
        int(handle),
        description="exact8 ProgramData DACL",
        required_mask=None,
        required_flags=0,
        forbidden_flags=0,
        effective_denied_accesses=(_WINDOWS_FILE_DELETE_CHILD,),
    )


def _require_attempt_receipt_root_acl(receipt_root_lease: Any) -> None:
    if os.name != "nt":
        return
    handle = getattr(receipt_root_lease, "windows_handle", None)
    if handle is None:
        raise ValueError("exact8 ReceiptAI root has no Windows ACL handle")
    _require_windows_acl_policy(
        int(handle),
        description="exact8 ReceiptAI root DACL",
        required_mask=_WINDOWS_DELETE | _WINDOWS_FILE_DELETE_CHILD,
        required_flags=_WINDOWS_OBJECT_INHERIT_ACE | _WINDOWS_CONTAINER_INHERIT_ACE,
        forbidden_flags=_WINDOWS_INHERITED_ACE | _WINDOWS_INHERIT_ONLY_ACE,
        effective_denied_accesses=(_WINDOWS_DELETE, _WINDOWS_FILE_DELETE_CHILD),
    )


def _require_attempt_marker_acl(handle: int) -> None:
    _require_windows_acl_policy(
        handle,
        description="exact8 attempt marker DACL",
        required_mask=_WINDOWS_DELETE,
        required_flags=_WINDOWS_INHERITED_ACE,
        forbidden_flags=_WINDOWS_INHERIT_ONLY_ACE,
        effective_denied_accesses=(_WINDOWS_DELETE,),
    )


def _expected_attempt_path(
    path: Path, *, inspection: Mapping[str, Any]
) -> Path:
    program_data = _common_application_data_path()
    if program_data is None:
        raise ValueError(
            "exact8 run requires the fixed Windows CommonApplicationData registry"
        )
    expected_registry = _existing(
        program_data / ATTEMPT_REGISTRY_PARENT / ATTEMPT_REGISTRY_NAME,
        directory=True,
        description="CommonApplicationData exact8 attempt registry",
    )
    supplied = Path(os.path.abspath(os.fspath(Path(path).expanduser())))
    _samefile(
        _existing(
            supplied.parent,
            directory=True,
            description="supplied exact8 attempt registry",
        ),
        expected_registry,
        "CommonApplicationData exact8 attempt registry",
    )
    expected_name = f"{inspection['attempt_id']}.attempt.json"
    _require_equal(supplied.name, expected_name, "attempt lock filename")
    return supplied


def _attempt_payload(
    *, inspection: Mapping[str, Any], output_root: Path
) -> dict[str, object]:
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": ATTEMPT_KIND,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "attempt_id": inspection["attempt_id"],
        "route_subject_id": inspection["route_subject_id"],
        "source_subject_id": inspection["source_subject_id"],
        "candidate_pilot_subject_id": inspection["candidate_pilot_subject_id"],
        "failure_subject_id": inspection["failure_subject_id"],
        "overlay_subject_id": inspection["overlay_subject_id"],
        "output_root": str(output_root),
        "epochs": FIXED_EPOCHS,
        "selector_mode": SELECTOR_MODE,
        "full_manifest_sha256": inspection["full_manifest_sha256"],
        "threat_model": ATTEMPT_THREAT_MODEL,
    }


@dataclass
class _AttemptFileLease:
    path: Path
    posix_fd: int | None = None
    windows_handle: int | None = None
    closed: bool = False

    def require(self, snapshot: _FrozenFile, *, checkpoint: str) -> None:
        if self.closed:
            raise ValueError(f"exact8 attempt file lease closed before {checkpoint}")
        if self.posix_fd is not None:
            if _stat_identity(os.fstat(self.posix_fd)) != snapshot.identity:
                raise ValueError(f"exact8 attempt handle identity changed at {checkpoint}")
        elif self.windows_handle is not None:
            _windows_attempt_handle_identity(self.windows_handle)
            _require_attempt_marker_acl(self.windows_handle)
        else:
            raise ValueError(f"exact8 attempt has no live file lease at {checkpoint}")
        _require_frozen_current(
            snapshot,
            description=f"exact8 attempt marker at {checkpoint}",
        )

    def close(self) -> None:
        if self.closed:
            return
        if self.posix_fd is not None:
            os.close(self.posix_fd)
            self.posix_fd = None
        if self.windows_handle is not None:
            _windows_close_native_handle(self.windows_handle)
            self.windows_handle = None
        self.closed = True


@dataclass
class _GuardedAttempt:
    path: Path
    snapshot: _FrozenFile
    program_data_identity: tuple[int, int, int]
    program_data_lease: Any
    receipt_root_identity: tuple[int, int, int]
    receipt_root_lease: Any
    registry_identity: tuple[int, int, int]
    registry_lease: Any
    file_lease: _AttemptFileLease
    closed: bool = False

    def require(self, checkpoint: str) -> None:
        from .recipient_multiview_overlay import (
            _directory_identity,
            _require_directory_lease_identity,
        )

        if self.closed:
            raise ValueError(f"exact8 attempt guard closed before {checkpoint}")
        try:
            _require_directory_lease_identity(self.registry_lease)
            _require_directory_lease_identity(self.receipt_root_lease)
            _require_directory_lease_identity(self.program_data_lease)
            if _directory_identity(self.path.parent.parent.parent) != (
                self.program_data_identity
            ):
                raise ValueError("ProgramData path identity changed")
            if _directory_identity(self.path.parent.parent) != self.receipt_root_identity:
                raise ValueError("ReceiptAI root path identity changed")
            if _directory_identity(self.path.parent) != self.registry_identity:
                raise ValueError("attempt registry path identity changed")
            _require_attempt_program_data_acl(self.program_data_lease)
            _require_attempt_receipt_root_acl(self.receipt_root_lease)
            _require_attempt_registry_acl(self.registry_lease)
            self.file_lease.require(self.snapshot, checkpoint=checkpoint)
        except (OSError, ValueError) as error:
            raise ValueError(
                f"exact8 one-shot guard changed at {checkpoint}: {error}"
            ) from error

    def close(self) -> None:
        if self.closed:
            return
        try:
            self.require("immediately_before_attempt_guard_close")
        finally:
            self.file_lease.close()
            self.registry_lease.close()
            self.receipt_root_lease.close()
            self.program_data_lease.close()
            self.closed = True


def _windows_attempt_handle_identity(handle: int) -> tuple[int, int, int, int]:
    class _ByHandleFileInformation(ctypes.Structure):
        _fields_ = (
            ("file_attributes", ctypes.c_uint32),
            ("creation_time_low", ctypes.c_uint32),
            ("creation_time_high", ctypes.c_uint32),
            ("access_time_low", ctypes.c_uint32),
            ("access_time_high", ctypes.c_uint32),
            ("write_time_low", ctypes.c_uint32),
            ("write_time_high", ctypes.c_uint32),
            ("volume_serial_number", ctypes.c_uint32),
            ("file_size_high", ctypes.c_uint32),
            ("file_size_low", ctypes.c_uint32),
            ("number_of_links", ctypes.c_uint32),
            ("file_index_high", ctypes.c_uint32),
            ("file_index_low", ctypes.c_uint32),
        )

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    information = _ByHandleFileInformation()
    get_information = kernel32.GetFileInformationByHandle
    get_information.argtypes = (ctypes.c_void_p, ctypes.c_void_p)
    get_information.restype = ctypes.c_int
    if not get_information(ctypes.c_void_p(handle), ctypes.byref(information)):
        error_number = ctypes.get_last_error()
        raise OSError(error_number, os.strerror(error_number))
    attributes = int(information.file_attributes)
    if attributes & 0x00000010 or attributes & 0x00000400:
        raise ValueError("exact8 attempt handle is a directory or reparse point")
    return (
        int(information.volume_serial_number),
        (int(information.file_index_high) << 32) | int(information.file_index_low),
        (int(information.file_size_high) << 32) | int(information.file_size_low),
        attributes,
    )


def _create_attempt_file_lease(path: Path, payload: bytes) -> _AttemptFileLease:
    if os.name == "nt":
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        create_file = kernel32.CreateFileW
        create_file.argtypes = (
            ctypes.c_wchar_p,
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.c_void_p,
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.c_void_p,
        )
        create_file.restype = ctypes.c_void_p
        handle = create_file(
            str(path),
            0x80000000
            | 0x40000000
            | _WINDOWS_READ_CONTROL,  # GENERIC_READ | GENERIC_WRITE | READ_CONTROL
            0x00000001,  # FILE_SHARE_READ; deny write/delete while held
            None,
            1,  # CREATE_NEW
            0x00200000,  # FILE_FLAG_OPEN_REPARSE_POINT
            None,
        )
        invalid = ctypes.c_void_p(-1).value
        if handle in (None, invalid):
            error_number = ctypes.get_last_error()
            if error_number in {80, 183}:  # ERROR_FILE_EXISTS / ALREADY_EXISTS
                raise ValueError(f"exact8 one-shot attempt is already consumed: {path}")
            raise ValueError(
                f"unable to atomically consume exact8 one-shot attempt: {path}: "
                f"{OSError(error_number, os.strerror(error_number))}"
            )
        lease = _AttemptFileLease(path=path, windows_handle=int(handle))
        try:
            _windows_attempt_handle_identity(int(handle))
            written = ctypes.c_uint32()
            write_file = kernel32.WriteFile
            write_file.argtypes = (
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_uint32,
                ctypes.POINTER(ctypes.c_uint32),
                ctypes.c_void_p,
            )
            write_file.restype = ctypes.c_int
            buffer = ctypes.create_string_buffer(payload)
            if not write_file(
                ctypes.c_void_p(handle),
                buffer,
                len(payload),
                ctypes.byref(written),
                None,
            ) or int(written.value) != len(payload):
                error_number = ctypes.get_last_error()
                raise OSError(error_number, os.strerror(error_number))
            flush = kernel32.FlushFileBuffers
            flush.argtypes = (ctypes.c_void_p,)
            flush.restype = ctypes.c_int
            if not flush(ctypes.c_void_p(handle)):
                error_number = ctypes.get_last_error()
                raise OSError(error_number, os.strerror(error_number))
            _require_attempt_marker_acl(int(handle))
            return lease
        except BaseException:
            lease.close()
            raise

    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    flags |= getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags, 0o600)
    except FileExistsError as error:
        raise ValueError(f"exact8 one-shot attempt is already consumed: {path}") from error
    except OSError as error:
        raise ValueError(
            f"unable to atomically consume exact8 one-shot attempt: {path}: {error}"
        ) from error
    lease = _AttemptFileLease(path=path, posix_fd=descriptor)
    try:
        remaining = memoryview(payload)
        while remaining:
            written = os.write(descriptor, remaining)
            if written <= 0:
                raise OSError("unable to complete exact8 attempt marker write")
            remaining = remaining[written:]
        os.fsync(descriptor)
        return lease
    except BaseException:
        lease.close()
        raise


def _consume_attempt(
    path: Path,
    *,
    inspection: Mapping[str, Any],
    output_root: Path,
) -> _GuardedAttempt:
    """Atomically and permanently consume this semantic exact8 subject."""

    attempt_path = _expected_attempt_path(path, inspection=inspection)
    payload = _attempt_payload(inspection=inspection, output_root=output_root)
    encoded = (
        json.dumps(
            payload,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            indent=2,
        )
        + "\n"
    ).encode("utf-8")
    from .recipient_multiview_overlay import _directory_identity, _open_directory_lease

    program_data_value = _common_application_data_path()
    if program_data_value is None:
        raise ValueError("exact8 attempt consumption requires Windows ProgramData")
    program_data = _existing(
        program_data_value,
        directory=True,
        description="Windows FOLDERID_ProgramData",
    )
    receipt_root = _existing(
        program_data / ATTEMPT_REGISTRY_PARENT,
        directory=True,
        description="exact8 ReceiptAI root",
    )
    _samefile(
        attempt_path.parent.parent,
        receipt_root,
        "exact8 ReceiptAI root",
    )
    program_data_identity = _directory_identity(program_data)
    receipt_root_identity = _directory_identity(receipt_root)
    registry_identity = _directory_identity(attempt_path.parent)
    program_data_lease = _open_directory_lease(
        program_data,
        expected=program_data_identity,
    )
    receipt_root_lease: Any | None = None
    registry_lease: Any | None = None
    file_lease: _AttemptFileLease | None = None
    try:
        receipt_root_lease = _open_directory_lease(
            receipt_root,
            expected=receipt_root_identity,
        )
        registry_lease = _open_directory_lease(
            attempt_path.parent,
            expected=registry_identity,
        )
        _require_attempt_program_data_acl(program_data_lease)
        _require_attempt_receipt_root_acl(receipt_root_lease)
        _require_attempt_registry_acl(registry_lease)
        file_lease = _create_attempt_file_lease(attempt_path, encoded)
        snapshot = _freeze_file(
            attempt_path,
            description="Python-created exact8 one-shot attempt lock",
        )
        _validate_attempt(
            attempt_path,
            inspection=inspection,
            output_root=output_root,
            snapshot=snapshot,
        )
        guard = _GuardedAttempt(
            path=attempt_path,
            snapshot=snapshot,
            program_data_identity=program_data_identity,
            program_data_lease=program_data_lease,
            receipt_root_identity=receipt_root_identity,
            receipt_root_lease=receipt_root_lease,
            registry_identity=registry_identity,
            registry_lease=registry_lease,
            file_lease=file_lease,
        )
        guard.require("immediately_after_CreateNew")
        return guard
    except BaseException:
        # Never remove a partial marker: reaching CREATE_NEW consumes one-shot
        # authority even if validation or durable writing fails afterwards.
        if file_lease is not None:
            file_lease.close()
        if registry_lease is not None:
            registry_lease.close()
        if receipt_root_lease is not None:
            receipt_root_lease.close()
        program_data_lease.close()
        raise


def _validate_attempt(
    path: Path,
    *,
    inspection: Mapping[str, Any],
    output_root: Path,
    snapshot: _FrozenFile | None = None,
) -> Mapping[str, Any]:
    attempt_path = _existing(path, directory=False, description="exact8 attempt lock")
    if snapshot is None:
        attempt = _strict_json(attempt_path)
    else:
        _samefile(attempt_path, snapshot.path, "frozen exact8 attempt lock")
        attempt = _strict_json_bytes(
            snapshot.data, description="frozen exact8 attempt lock"
        )
    expected_keys = {
        "schema_version",
        "kind",
        "created_at_utc",
        "attempt_id",
        "route_subject_id",
        "source_subject_id",
        "candidate_pilot_subject_id",
        "failure_subject_id",
        "overlay_subject_id",
        "output_root",
        "epochs",
        "selector_mode",
        "full_manifest_sha256",
        "threat_model",
    }
    if set(attempt) != expected_keys:
        raise ValueError("exact8 attempt lock keys changed")
    _require_equal(attempt.get("schema_version"), SCHEMA_VERSION, "attempt schema")
    _require_equal(attempt.get("kind"), ATTEMPT_KIND, "attempt kind")
    _require_equal(attempt.get("attempt_id"), inspection["attempt_id"], "attempt id")
    _require_equal(
        attempt.get("route_subject_id"),
        inspection["route_subject_id"],
        "attempt route subject",
    )
    _require_equal(
        attempt.get("source_subject_id"),
        inspection["source_subject_id"],
        "attempt source subject",
    )
    _require_equal(
        attempt.get("candidate_pilot_subject_id"),
        inspection["candidate_pilot_subject_id"],
        "attempt A8 subject",
    )
    _require_equal(
        attempt.get("failure_subject_id"),
        inspection["failure_subject_id"],
        "attempt failure subject",
    )
    _require_equal(
        attempt.get("overlay_subject_id"),
        inspection["overlay_subject_id"],
        "attempt overlay subject",
    )
    raw_output = attempt.get("output_root")
    if not isinstance(raw_output, str) or not raw_output:
        raise ValueError("attempt lock has no output root")
    if os.path.normcase(os.path.abspath(raw_output)) != os.path.normcase(
        os.path.abspath(os.fspath(output_root))
    ):
        raise ValueError("attempt lock output root does not match")
    _require_equal(attempt.get("epochs"), FIXED_EPOCHS, "attempt epochs")
    _require_equal(attempt.get("selector_mode"), SELECTOR_MODE, "attempt selector")
    _require_equal(
        attempt.get("full_manifest_sha256"),
        inspection["full_manifest_sha256"],
        "attempt full manifest",
    )
    _require_equal(
        attempt.get("threat_model"), ATTEMPT_THREAT_MODEL, "attempt threat model"
    )
    created = attempt.get("created_at_utc")
    if not isinstance(created, str) or not created:
        raise ValueError("exact8 attempt lock has no creation timestamp")
    expected_name = f"{inspection['attempt_id']}.attempt.json"
    _require_equal(attempt_path.name, expected_name, "attempt lock filename")
    _require_equal(
        attempt_path.parent.name, ATTEMPT_REGISTRY_NAME, "attempt registry name"
    )
    _require_equal(
        attempt_path.parent.parent.name,
        ATTEMPT_REGISTRY_PARENT,
        "attempt registry parent",
    )
    program_data = _common_application_data_path()
    if program_data is not None:
        expected_registry = _existing(
            program_data / ATTEMPT_REGISTRY_PARENT / ATTEMPT_REGISTRY_NAME,
            directory=True,
            description="CommonApplicationData exact8 attempt registry",
        )
        _samefile(
            attempt_path.parent,
            expected_registry,
            "CommonApplicationData exact8 attempt registry",
        )
    return attempt


def _checkpoint_artifact(
    snapshot: _FrozenFile,
    *,
    summary: Mapping[str, Any],
    labels: Mapping[str, Any],
    epoch_record: Mapping[str, Any],
    expected_epoch: int,
    source_state: Mapping[str, object],
    torch: Any,
    description: str,
) -> Mapping[str, object]:
    checkpoint = _load_checkpoint(io.BytesIO(snapshot.data), torch=torch)
    if not isinstance(checkpoint, Mapping):
        raise ValueError(f"{description} payload must be an object")
    _assert_checkpoint_has_no_unsafe_claims(checkpoint, description=description)
    _require_checkpoint_without_optimizer_state(checkpoint, description=description)
    _require_equal(checkpoint.get("schema_version"), SCHEMA_VERSION, f"{description} schema")
    _require_equal(checkpoint.get("kind"), KIND_V13, f"{description} kind")
    _require_equal(checkpoint.get("epoch"), expected_epoch, f"{description} epoch")
    for key in (
        "config",
        "initialization",
        "fine_tune_policy",
        "status_text_runtime_policy",
        "training_runtime",
        "field_counts",
        "checkpoint_selection_policy",
        "recipient_oov_by_split",
        "recipient_sampling_policy",
        "recipient_confidence_policy",
        "recipient_tail_loss_policy",
        "recipient_train_augmentation_policy",
        "recipient_train_split_policy",
        "recipient_target",
        *A8_SUMMARY_DATA_KEYS,
    ):
        _json_equal(checkpoint.get(key), summary.get(key), f"{description} {key}")
    for key in (
        "amount_characters",
        "time_characters",
        "payment_characters",
        "status_classes",
        "status_text_blank_index",
        "status_text_characters",
        "status_text_charset_sha256",
        "status_text_charset_source",
        "status_text_target",
        "recipient_characters",
        "recipient_blank_index",
        "recipient_charset_sha256",
        "recipient_charset_source",
        "payment_bank_prefix_classes",
        "payment_bank_prefix_min_support",
        "payment_bank_prefix_class_counts",
        "payment_bank_prefix_train_class_counts",
        "payment_bank_prefix_oov_by_split",
    ):
        _json_equal(checkpoint.get(key), labels.get(key), f"{description} labels {key}")
    for key, expected in (
        ("recipient_loss_weight", 1.0),
        ("ctc_loss_weight", 1.0),
        ("structured_loss_weight", 1.0),
    ):
        value = checkpoint.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{description} {key} is invalid")
        if not math.isfinite(float(value)) or not math.isclose(
            float(value), expected, rel_tol=0.0, abs_tol=1e-12
        ):
            raise ValueError(f"{description} {key} changed")
    _json_equal(checkpoint.get("metrics"), epoch_record, f"{description} metrics")
    state = _state_dict(checkpoint, description=description)
    declared_config = _checkpoint_config(checkpoint)
    _json_equal(
        asdict(declared_config), summary.get("config"), f"{description} declared config"
    )
    _validate_state_matches_declared_model(
        checkpoint,
        config=declared_config,
        state=state,
        description=description,
    )
    if _partition_descriptor(state, recipient=False) != _partition_descriptor(
        source_state, recipient=False
    ):
        raise ValueError(f"{description} changed frozen non-recipient state")
    return checkpoint


def _validate_labels(
    labels: Mapping[str, Any],
    *,
    summary: Mapping[str, Any],
    data_label_proof: Mapping[str, Any],
) -> None:
    _require_equal(labels.get("schema_version"), SCHEMA_VERSION, "labels schema")
    if set(data_label_proof) != {
        "summary_fields",
        "financial_label_policy",
        "ordered_labels",
        "ordered_label_maps",
        "blank_indices",
    }:
        raise ValueError("A8 data/label proof keys changed")
    config = UnifiedReaderConfig(**dict(_mapping(summary.get("config"), "labels config")))
    expected_metadata = _recipient_artifact_metadata(
        config,
        recipient_sampling_policy=summary.get("recipient_sampling_policy"),
        recipient_confidence_policy=summary.get("recipient_confidence_policy"),
        recipient_tail_loss_policy=summary.get("recipient_tail_loss_policy"),
        recipient_train_augmentation_policy=summary.get(
            "recipient_train_augmentation_policy"
        ),
    )
    expected_keys = EXACT8_LABEL_BASE_KEYS | set(expected_metadata)
    if set(labels) != expected_keys:
        raise ValueError(
            "exact8 labels key set changed; "
            f"missing={sorted(expected_keys - set(labels))}, "
            f"unexpected={sorted(set(labels) - expected_keys)}"
        )
    for key, expected in expected_metadata.items():
        _json_equal(labels.get(key), expected, f"labels metadata {key}")
    for key in (
        "initialization",
        "fine_tune_policy",
        "status_text_runtime_policy",
        "training_runtime",
        "checkpoint_selection_policy",
        "structured_target_counts",
        "status_text_oov_by_split",
        "recipient_oov_by_split",
        "recipient_sampling_policy",
        "recipient_confidence_policy",
        "recipient_tail_loss_policy",
        "recipient_train_augmentation_policy",
        "recipient_train_split_policy",
        "recipient_target",
    ):
        _json_equal(labels.get(key), summary.get(key), f"labels {key}")
    blank_indices = _mapping(
        data_label_proof.get("blank_indices"), "A8 frozen blank indices"
    )
    if set(blank_indices) != set(A8_BLANK_INDEX_KEYS):
        raise ValueError("A8 frozen blank-index proof keys changed")
    for key in A8_BLANK_INDEX_KEYS:
        proof = _mapping(blank_indices.get(key), f"A8 frozen {key} proof")
        if set(proof) != {"source", "semantic", "value"}:
            raise ValueError(f"A8 frozen {key} proof keys changed")
        expected_source, expected_semantic = A8_BLANK_INDEX_PROOF[key]
        _require_equal(proof.get("source"), expected_source, f"A8 frozen {key} source")
        _require_equal(
            proof.get("semantic"), expected_semantic, f"A8 frozen {key} semantic"
        )
        _require_equal(proof.get("value"), 0, f"A8 frozen {key} value")
        _require_equal(labels.get(key), proof.get("value"), f"labels {key}")
    characters = labels.get("recipient_characters")
    if (
        not isinstance(characters, Sequence)
        or isinstance(characters, (str, bytes))
        or not characters
        or any(not isinstance(character, str) or not character for character in characters)
    ):
        raise ValueError("labels recipient character map is invalid")
    expected_charset = hashlib.sha256("".join(characters).encode("utf-8")).hexdigest()
    _require_equal(
        labels.get("recipient_charset_sha256"),
        expected_charset,
        "labels recipient charset hash",
    )
    ordered = _mapping(
        data_label_proof.get("ordered_labels"), "A8 frozen ordered labels"
    )
    ordered_map_proof = _mapping(
        data_label_proof.get("ordered_label_maps"),
        "A8 frozen ordered-label map proof",
    )
    expected_ordered_keys = {*A8_ORDERED_LABEL_KEYS, "payment_charset_sha256"}
    if set(ordered) != expected_ordered_keys:
        raise ValueError("A8 frozen ordered-label proof keys changed")
    if set(ordered_map_proof) != set(A8_ORDERED_MAP_KEYS):
        raise ValueError("A8 frozen ordered-label map proof keys changed")
    for key in sorted(expected_ordered_keys):
        _json_equal(labels.get(key), ordered.get(key), f"A8-frozen label {key}")
    for key in A8_ORDERED_MAP_KEYS:
        proof = _mapping(
            ordered_map_proof.get(key), f"A8 frozen ordered-label map {key} proof"
        )
        if set(proof) != {"count", "sha256"}:
            raise ValueError(f"A8 frozen ordered-label map {key} proof changed")
        authoritative = ordered[key]
        _require_equal(
            proof.get("count"), len(authoritative), f"A8 frozen {key} count"
        )
        _require_equal(
            proof.get("sha256"),
            _canonical_sha256(authoritative),
            f"A8 frozen {key} SHA-256",
        )


def evaluate_exact8_summary(
    summary: Mapping[str, Any],
    *,
    inspection: Mapping[str, Any],
    recipe: Mapping[str, Any],
) -> dict[str, object]:
    """Recompute the fixed integer gates without opening any held-out test row."""

    _json_equal(recipe, _recipe(inspection), "exact8 training recipe")
    baseline = _mapping(inspection.get("baseline"), "A8 baseline")
    data_label_proof = _mapping(
        baseline.get("data_label_proof"), "A8 data/label proof"
    )
    _require_equal(summary.get("schema_version"), SCHEMA_VERSION, "summary schema")
    _require_equal(summary.get("kind"), KIND_V13, "summary kind")
    _json_equal(summary.get("config"), inspection["target_config"], "target config")
    initialization = _mapping(summary.get("initialization"), "exact8 initialization")
    source_config = _mapping(
        initialization.get("source_config"), "exact8 source config"
    )
    target_config = UnifiedReaderConfig(**dict(_mapping(summary["config"], "config")))
    source_reader = UnifiedReaderConfig(**dict(source_config))
    _validate_recipient_visual_context_reinit_config(source_reader, target_config)
    _require_equal(
        initialization.get("mode"),
        "parameter_only_recipient_visual_context_reinit",
        "exact8 initialization mode",
    )
    _require_equal(
        initialization.get("init_checkpoint_mode"),
        INIT_CHECKPOINT_MODE_RECIPIENT_VISUAL_CONTEXT_REINIT,
        "exact8 init checkpoint mode",
    )
    _require_equal(
        initialization.get("checkpoint_sha256"),
        inspection["source_checkpoint_sha256"],
        "exact8 source checkpoint hash",
    )
    _require_equal(initialization.get("source_kind"), KIND_V13, "source checkpoint kind")
    raw_checkpoint_path = initialization.get("checkpoint_path")
    if not isinstance(raw_checkpoint_path, str) or not raw_checkpoint_path:
        raise ValueError("exact8 initialization has no source checkpoint path")
    _samefile(
        _existing(
            Path(raw_checkpoint_path),
            directory=False,
            description="summary source checkpoint",
        ),
        Path(str(inspection["source_checkpoint"])),
        "summary source checkpoint",
    )
    financial_labels = _mapping(
        initialization.get("financial_label_policy"),
        "exact8 financial label policy",
    )
    recipient_map = _mapping(
        financial_labels.get("recipient_character_map"),
        "exact8 recipient character map",
    )
    _require_equal(
        recipient_map.get("mode"),
        "fresh_train_only_reinitialized_recipient_v1",
        "exact8 recipient character map mode",
    )
    _json_equal(
        financial_labels,
        data_label_proof.get("financial_label_policy"),
        "A8-frozen financial label policy",
    )
    fine_tune = _mapping(summary.get("fine_tune_policy"), "exact8 fine tune")
    _require_equal(fine_tune.get("mode"), "recipient_only_v13", "fine-tune mode")
    _require_equal(
        fine_tune.get("trainable_parameter_prefix"), "recipient_", "trainable prefix"
    )
    _require_equal(
        fine_tune.get("training_forward"),
        "private_recipient_branch_only_v13",
        "training forward",
    )
    runtime = _mapping(summary.get("training_runtime"), "exact8 runtime")
    for key, expected in {
        "device": "cuda:0",
        "uses_cuda": True,
        "num_workers": FIXED_NUM_WORKERS,
        "prefetch_factor": FIXED_PREFETCH_FACTOR,
        "persistent_workers": True,
        "train_progress_every": FIXED_PROGRESS_EVERY,
        "validation_every": 1,
        "cuda_tf32_requested": True,
        "cudnn_benchmark_requested": True,
    }.items():
        _require_equal(runtime.get(key), expected, f"runtime {key}")
    if "4090" not in str(runtime.get("cuda_device_name", "")):
        raise ValueError("exact8 runtime is not an RTX 4090")
    _require_equal(
        summary.get("status_text_runtime_policy"),
        STATUS_TEXT_RUNTIME_POLICY,
        "status-text runtime policy",
    )
    _json_equal(
        summary.get("recipient_train_augmentation_policy"),
        _recipient_train_augmentation_policy(mode=FIXED_AUGMENTATION, seed=FIXED_SEED),
        "recipient augmentation policy",
    )
    _json_equal(
        summary.get("recipient_confidence_policy"),
        _recipient_confidence_policy(
            low_confidence_threshold=0.95,
            low_confidence_loss_weight=0.50,
            curriculum_epochs=10,
        ),
        "recipient confidence policy",
    )
    sampling = _mapping(summary.get("recipient_sampling_policy"), "sampling policy")
    _require_equal(sampling.get("mode"), "uniform", "recipient sampling mode")
    tail = _validate_recipient_tail_loss_policy(
        summary.get("recipient_tail_loss_policy")
    )
    for key, expected in {
        "mode": "rare_long_tail_ctc_v1",
        "rare_character_max_support": 3,
        "rare_character_loss_weight": 1.5,
        "long_text_min_length": 9,
        "long_text_loss_weight": 1.5,
    }.items():
        _require_equal(tail.get(key), expected, f"recipient tail policy {key}")
    recipient_loss_weight = summary.get("recipient_loss_weight")
    if (
        isinstance(recipient_loss_weight, bool)
        or not isinstance(recipient_loss_weight, (int, float))
        or not math.isfinite(float(recipient_loss_weight))
        or not math.isclose(
            float(recipient_loss_weight), 1.0, rel_tol=0.0, abs_tol=1e-12
        )
    ):
        raise ValueError("exact8 recipient loss weight changed")
    split = _mapping(summary.get("recipient_train_split_policy"), "split policy")
    _require_equal(split.get("mode"), "standard_train_only", "split mode")
    _require_equal(split.get("splits"), ["train"], "training splits")
    checkpoint_policy = _mapping(
        summary.get("checkpoint_selection_policy"), "checkpoint policy"
    )
    expected_checkpoint_policy = _expected_checkpoint_policy(target_config)
    _json_equal(
        checkpoint_policy,
        expected_checkpoint_policy,
        "fixed exact8 checkpoint selection policy",
    )
    status_policy = _mapping(summary.get("status_head_policy"), "status head policy")

    field_counts = _mapping(summary.get("field_counts"), "field counts")
    frozen_summary_fields = _mapping(
        data_label_proof.get("summary_fields"), "A8 frozen summary fields"
    )
    if set(frozen_summary_fields) != set(A8_SUMMARY_DATA_KEYS):
        raise ValueError("A8 frozen summary-field proof keys changed")
    for key in A8_SUMMARY_DATA_KEYS:
        _json_equal(
            summary.get(key),
            frozen_summary_fields.get(key),
            f"A8-frozen summary field {key}",
        )
    for field, raw_counts in field_counts.items():
        counts = _mapping(raw_counts, f"{field} counts")
        _require_equal(counts.get("test"), 0, f"{field} test count")
    recipient_counts = _mapping(
        field_counts.get("recipient_field"), "recipient field counts"
    )
    _require_equal(
        recipient_counts.get("val"),
        EXPECTED_RECIPIENT_VAL_RECORDS,
        "recipient val denominator",
    )
    recipient_oov = _mapping(
        summary.get("recipient_oov_by_split"), "recipient OOV audit"
    )
    _require_equal(
        _mapping(recipient_oov.get("val"), "recipient val OOV").get("records"),
        EXPECTED_RECIPIENT_VAL_RECORDS,
        "recipient val OOV records",
    )
    _require_equal(
        _mapping(recipient_oov.get("test"), "recipient test OOV").get("records"),
        0,
        "recipient test OOV records",
    )
    frozen_denominators = _mapping(
        baseline.get("candidate_denominators"), "A8 candidate denominators"
    )
    expected_val_counts: dict[str, int] = {}
    for name in ("amount", "time", "payment_method_field", "transfer_status"):
        counts = _mapping(field_counts.get(name), f"{name} field counts")
        val_count = counts.get("val")
        if isinstance(val_count, bool) or not isinstance(val_count, int) or val_count <= 0:
            raise ValueError(f"{name} val count must be a positive integer")
        candidate_count = frozen_denominators.get(name)
        if (
            isinstance(candidate_count, bool)
            or not isinstance(candidate_count, int)
            or candidate_count <= 0
        ):
            raise ValueError(f"A8 {name} candidate denominator is invalid")
        if (
            name in EXPECTED_CANDIDATE_VAL_RECORDS
            and candidate_count != EXPECTED_CANDIDATE_VAL_RECORDS[name]
        ):
            raise ValueError(f"A8 {name} candidate denominator changed")
        expected_val_counts[name] = candidate_count
    raw_records = summary.get("records")
    if not isinstance(raw_records, Sequence) or isinstance(raw_records, (str, bytes)):
        raise ValueError("exact8 summary has invalid epoch records")
    records = [_mapping(record, "exact8 epoch record") for record in raw_records]
    if [record.get("epoch") for record in records] != list(range(1, 9)):
        raise ValueError("exact8 summary requires ordered epochs 1 through 8")

    recipient_by_epoch: dict[int, dict[str, object]] = {}
    selection_scores: dict[int, tuple[float, ...]] = {}
    guard_failures: list[str] = []
    for record in records:
        epoch = int(record["epoch"])
        _require_equal(
            record.get("validation_performed"), True, f"epoch {epoch} validation"
        )
        recipient_by_epoch[epoch] = _metric(
            record, "recipient_field", f"exact8 epoch {epoch}"
        )
        fields = _mapping(
            record.get("val_candidate_text_by_field"), f"epoch {epoch} fields"
        )
        for name, floor in {
            "amount": AMOUNT_FLOOR,
            "time": TIME_FLOOR,
            "payment_method_field": PAYMENT_FLOOR,
        }.items():
            metric = _mapping(fields.get(name), f"epoch {epoch} {name}")
            exact = float(
                _count_rate_metric(
                    metric,
                    expected_records=expected_val_counts[name],
                    description=f"epoch {epoch} {name}",
                )["exact_match"]
            )
            if exact < floor:
                guard_failures.append(f"epoch_{epoch}_{name}_below_floor")
        status_fields = _mapping(
            record.get("val_ctc_by_field"), f"epoch {epoch} CTC fields"
        )
        status = _mapping(
            status_fields.get("transfer_status"), f"epoch {epoch} status"
        )
        status_exact = float(
            _count_rate_metric(
                status,
                expected_records=expected_val_counts["transfer_status"],
                description=f"epoch {epoch} visible status",
            )["exact_match"]
        )
        if status_exact < STATUS_TEXT_FLOOR:
            guard_failures.append(f"epoch_{epoch}_visible_status_below_floor")
        unsafe = record.get("val_status_non_success_to_success")
        if isinstance(unsafe, bool) or not isinstance(unsafe, int):
            raise ValueError(f"epoch {epoch} unsafe status count is invalid")
        if unsafe != 0:
            guard_failures.append(f"epoch_{epoch}_unsafe_status_nonzero")
        validation = _validation_from_epoch_record(
            record, description=f"epoch {epoch} validation record"
        )
        score, protection_failures = _checkpoint_selection_score(
            validation,
            config=target_config,
            status_policy=status_policy,
            policy=expected_checkpoint_policy,
        )
        protection = _checkpoint_protection_report(
            validation,
            policy=expected_checkpoint_policy,
            failures=protection_failures,
        )
        expected_score = list(score) if score is not None else None
        _json_equal(
            record.get("checkpoint_selection_score"),
            expected_score,
            f"epoch {epoch} recomputed checkpoint score",
        )
        _json_equal(
            record.get("checkpoint_selection_protection_failures"),
            protection_failures,
            f"epoch {epoch} recomputed checkpoint failures",
        )
        _json_equal(
            record.get("checkpoint_protection"),
            protection,
            f"epoch {epoch} recomputed checkpoint protection",
        )
        _require_equal(
            record.get("checkpoint_selection_eligible"),
            score is not None,
            f"epoch {epoch} recomputed checkpoint eligibility",
        )
        if score is None:
            guard_failures.append(f"epoch_{epoch}_checkpoint_not_eligible")
            guard_failures.append(f"epoch_{epoch}_checkpoint_protection_failed")
        else:
            if len(score) != 6:
                raise ValueError("exact8 recomputed checkpoint score is not six-dimensional")
            selection_scores[epoch] = score

    expected_best_epoch: int | None = None
    expected_best_score: tuple[float, ...] | None = None
    for epoch in range(1, FIXED_EPOCHS + 1):
        score = selection_scores.get(epoch)
        if score is not None and (
            expected_best_score is None or score > expected_best_score
        ):
            expected_best_epoch = epoch
            expected_best_score = score
    if expected_best_epoch is None or expected_best_score is None:
        raise ValueError("exact8 has no checkpoint-selection-eligible epoch")
    best_epoch = summary.get("best_checkpoint_epoch")
    if (
        isinstance(best_epoch, bool)
        or not isinstance(best_epoch, int)
        or best_epoch not in recipient_by_epoch
    ):
        raise ValueError("exact8 best checkpoint epoch is invalid")
    _require_equal(
        best_epoch,
        expected_best_epoch,
        "exact8 strict-greater-than first-best checkpoint epoch",
    )
    best_matches = int(recipient_by_epoch[best_epoch]["exact_matches"])
    _json_equal(
        summary.get("best_checkpoint_score"),
        list(expected_best_score),
        "exact8 recomputed best checkpoint score",
    )
    epoch4_matches = int(recipient_by_epoch[4]["exact_matches"])
    epoch8_matches = int(recipient_by_epoch[8]["exact_matches"])
    baseline_best = int(baseline["best_matches"])
    baseline_epoch8 = int(baseline["epoch8_matches"])
    minimum_best = max(MINIMUM_BEST_MATCHES, baseline_best + BASELINE_GAIN_MATCHES)
    failures = list(dict.fromkeys(guard_failures))
    if best_matches < minimum_best:
        failures.append("best_below_absolute_or_A8_plus_68")
    if epoch8_matches < baseline_epoch8 + BASELINE_GAIN_MATCHES:
        failures.append("epoch8_below_A8_epoch8_plus_68")
    if epoch8_matches - epoch4_matches < MINIMUM_EPOCH4_TO_8_GAIN_MATCHES:
        failures.append("epoch4_to_8_gain_below_136")
    if best_matches - epoch8_matches > MAXIMUM_BEST_TO_EPOCH8_GAP_MATCHES:
        failures.append("best_to_epoch8_decay_above_67")
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": DECISION_KIND,
        "analysis_only": True,
        "production_route_authorized": False,
        "test_opened": False,
        "onnx_exported": False,
        "passed": not failures,
        "failures": failures,
        "decision": (
            "analysis_only_authorize_one_fresh_fixed2_60_from_original_pilot_best"
            if not failures
            else "analysis_only_stop_fixed2_route"
        ),
        "baseline": dict(baseline),
        "fixed_gates": dict(_mapping(inspection["fixed_gates"], "fixed gates")),
        "observed": {
            "best_epoch": best_epoch,
            "best_matches": best_matches,
            "epoch4_matches": epoch4_matches,
            "epoch8_matches": epoch8_matches,
            "best_gain_over_A8_matches": best_matches - baseline_best,
            "epoch8_gain_over_A8_matches": epoch8_matches - baseline_epoch8,
            "epoch4_to_8_gain_matches": epoch8_matches - epoch4_matches,
            "best_to_epoch8_gap_matches": best_matches - epoch8_matches,
            "recipient_denominator": EXPECTED_RECIPIENT_VAL_RECORDS,
            "recipient_candidate_coverage": 1.0,
        },
        "pass_authorization": (
            {
                "authorization": PASS_AUTHORIZATION,
                "source": "original_full_crop_pilot_best_not_exact8_best",
                "initialization": INIT_CHECKPOINT_MODE_RECIPIENT_VISUAL_CONTEXT_REINIT,
                "training_data_view": "same_fixed2_overlay_subject",
                "epochs": 60,
                "fresh_optimizer": True,
                "required_final_recipient_matches": STRICT_RECIPIENT_PASS_MATCHES,
                "requires_strictly_greater_than_90_percent": True,
                "exact8_checkpoint_initialization": "forbidden",
                "test_opened": False,
                "onnx_exported": False,
                "production_route_authorized": False,
            }
            if not failures
            else None
        ),
    }


def _assert_no_delivery_artifacts(root: Path) -> None:
    def reject_test_split(value: object, location: str) -> None:
        if isinstance(value, Mapping):
            if value.get("evaluation_split") == "test":
                raise ValueError(f"exact8 output contains test evaluation at {location}")
            for key, child in value.items():
                reject_test_split(child, f"{location}.{key}")
        elif isinstance(value, list):
            for index, child in enumerate(value):
                reject_test_split(child, f"{location}[{index}]")

    for path in root.rglob("*"):
        if _is_reparse(path):
            raise ValueError("exact8 output contains a symlink/junction/reparse entry")
        if path.is_file() and path.suffix.lower() == ".onnx":
            raise ValueError("exact8 output contains a forbidden ONNX artifact")
        if path.is_file() and path.suffix.lower() == ".json":
            payload = _strict_json(path)
            _assert_checkpoint_has_no_unsafe_claims(
                payload,
                description=f"exact8 output JSON {path}",
            )
            reject_test_split(payload, str(path))


def _assert_exact_output_tree(root: Path, *, expected_files: Sequence[Path]) -> None:
    expected = {path.resolve(strict=True) for path in expected_files}
    observed = {path.resolve(strict=True) for path in root.rglob("*") if path.is_file()}
    if observed != expected:
        unexpected = sorted(str(path) for path in observed - expected)
        missing = sorted(str(path) for path in expected - observed)
        raise ValueError(
            "exact8 output file set changed; "
            f"unexpected={unexpected!r}, missing={missing!r}"
        )


def _sealed_decision_payload(
    *,
    decision: Mapping[str, Any],
    inspection: Mapping[str, Any],
    artifacts: Mapping[str, Any],
) -> dict[str, object]:
    return {
        **dict(decision),
        "route_subject_id": inspection["route_subject_id"],
        "attempt_id": inspection["attempt_id"],
        "source_subject_id": inspection["source_subject_id"],
        "candidate_pilot_subject_id": inspection["candidate_pilot_subject_id"],
        "failure_subject_id": inspection["failure_subject_id"],
        "overlay_subject_id": inspection["overlay_subject_id"],
        "source_checkpoint_sha256": inspection["source_checkpoint_sha256"],
        "overlay_contract_sha256": inspection["overlay_contract_sha256"],
        "overlay_closure": {
            "verified_before_training": True,
            "verified_after_training": True,
            "continuous_train_and_validation_image_read_leases": False,
            "opening_directory_identities": list(
                inspection.get("guard_directory_identities", [])
            ),
            "residual_risk": (
                "selected train overlay and unchanged validation image "
                "swap-and-restore during training are not excluded by the "
                "pre/post contract verification"
            ),
        },
        "output_closure": {
            "formal_platform": "windows",
            "program_data_resolved_by_known_folder_api": True,
            "one_shot_marker_created_by_python_run": True,
            "program_data_delete_child_effectively_denied": True,
            "receipt_root_explicit_inheritable_delete_dacl": True,
            "attempt_registry_explicit_inheritable_delete_dacl": True,
            "attempt_marker_inherited_delete_dacl": True,
            "attempt_acl_reverified_before_guard_close": True,
            "owner_write_dac_and_local_admin_bypass_out_of_scope": True,
            "program_data_receipt_root_registry_deny_delete_leases": True,
            "attempt_registry_deny_delete_lease": True,
            "attempt_file_deny_write_delete_lease": True,
            "output_parent_deny_delete_lease": True,
            "output_and_training_parent_handle_relative_atomic_create": True,
            "output_root_deny_delete_lease": True,
            "training_directory_deny_delete_lease": True,
            "leases_held_through_decision_publication": True,
        },
        "training_artifacts": dict(artifacts),
        "code": inspection["code"],
    }


def run_exact8(
    *,
    full_records: Path,
    original_dataset_root: Path,
    full_crop_pilot_root: Path,
    source_contract_path: Path,
    candidate_pilot_evidence_path: Path,
    failure_evidence_path: Path,
    failure_attempt_registry: Path,
    overlay_contract_path: Path,
    output_root: Path,
    attempt_lock: Path,
    torch: Any | None = None,
) -> dict[str, object]:
    if torch is None:
        torch, _ = _require_torch()
    inspection = inspect_exact8_subject(
        full_records=full_records,
        original_dataset_root=original_dataset_root,
        full_crop_pilot_root=full_crop_pilot_root,
        source_contract_path=source_contract_path,
        candidate_pilot_evidence_path=candidate_pilot_evidence_path,
        failure_evidence_path=failure_evidence_path,
        failure_attempt_registry=failure_attempt_registry,
        overlay_contract_path=overlay_contract_path,
        torch=torch,
    )
    output = _fresh_directory(output_root)
    attempt_path = _expected_attempt_path(attempt_lock, inspection=inspection)
    _assert_output_disjoint(
        output,
        protected_directories=(
            Path(original_dataset_root).resolve(),
            Path(full_crop_pilot_root).resolve(),
            Path(failure_attempt_registry).resolve(),
            attempt_path.parent,
            Path(str(inspection["composite_records"])).resolve().parent,
            *(
                Path(path).resolve()
                for path in inspection.get("guard_directories", [])
                if isinstance(path, str)
            ),
        ),
    )
    output_anchor = _open_guarded_output_parent(output)
    attempt_guard: _GuardedAttempt | None = None
    try:
        # CreateNew is owned by this Python run, not by a fallible wrapper.
        # Any exception after this point permanently consumes the subject.
        attempt_guard = _consume_attempt(
            attempt_path,
            inspection=inspection,
            output_root=output,
        )
        output = output_anchor.create()
        attempt_guard.require("immediately_after_output_creation")
        return _run_exact8_anchored(
            full_records=full_records,
            original_dataset_root=original_dataset_root,
            full_crop_pilot_root=full_crop_pilot_root,
            source_contract_path=source_contract_path,
            candidate_pilot_evidence_path=candidate_pilot_evidence_path,
            failure_evidence_path=failure_evidence_path,
            failure_attempt_registry=failure_attempt_registry,
            overlay_contract_path=overlay_contract_path,
            output=output,
            inspection=inspection,
            attempt_snapshot=attempt_guard.snapshot,
            attempt_guard=attempt_guard,
            output_anchor=output_anchor,
            torch=torch,
        )
    finally:
        if attempt_guard is not None:
            attempt_guard.close()
        output_anchor.close()


def _run_exact8_anchored(
    *,
    full_records: Path,
    original_dataset_root: Path,
    full_crop_pilot_root: Path,
    source_contract_path: Path,
    candidate_pilot_evidence_path: Path,
    failure_evidence_path: Path,
    failure_attempt_registry: Path,
    overlay_contract_path: Path,
    output: Path,
    inspection: Mapping[str, Any],
    attempt_snapshot: _FrozenFile,
    attempt_guard: _GuardedAttempt,
    output_anchor: _GuardedOutputDirectory,
    torch: Any,
) -> dict[str, object]:
    attempt_guard.require("before_recipe_publication")
    output_anchor.require("before_recipe_publication")
    recipe = _recipe(inspection)
    recipe_path = output / "recipient_multiview_exact8_recipe.json"
    _write_json_no_clobber(recipe_path, recipe)
    output_anchor.require("after_recipe_publication")
    recipe_snapshot = _freeze_file(recipe_path, description="exact8 recipe")
    _json_equal(
        _strict_json_bytes(recipe_snapshot.data, description="frozen exact8 recipe"),
        recipe,
        "published exact8 recipe",
    )
    training = output / "training-multiview-fixed2-exact8"
    training_anchor = _open_guarded_child(output_anchor, training)
    try:
        training = training_anchor.create()
        attempt_guard.require("immediately_after_training_directory_creation")
        return _finish_exact8_run(
            full_records=full_records,
            original_dataset_root=original_dataset_root,
            full_crop_pilot_root=full_crop_pilot_root,
            source_contract_path=source_contract_path,
            candidate_pilot_evidence_path=candidate_pilot_evidence_path,
            failure_evidence_path=failure_evidence_path,
            failure_attempt_registry=failure_attempt_registry,
            overlay_contract_path=overlay_contract_path,
            output=output,
            training=training,
            inspection=inspection,
            attempt_snapshot=attempt_snapshot,
            attempt_guard=attempt_guard,
            output_anchor=output_anchor,
            training_anchor=training_anchor,
            recipe=recipe,
            recipe_path=recipe_path,
            recipe_snapshot=recipe_snapshot,
            torch=torch,
        )
    finally:
        training_anchor.close()


def _finish_exact8_run(
    *,
    full_records: Path,
    original_dataset_root: Path,
    full_crop_pilot_root: Path,
    source_contract_path: Path,
    candidate_pilot_evidence_path: Path,
    failure_evidence_path: Path,
    failure_attempt_registry: Path,
    overlay_contract_path: Path,
    output: Path,
    training: Path,
    inspection: Mapping[str, Any],
    attempt_snapshot: _FrozenFile,
    attempt_guard: _GuardedAttempt,
    output_anchor: _GuardedOutputDirectory,
    training_anchor: _GuardedOutputDirectory,
    recipe: Mapping[str, Any],
    recipe_path: Path,
    recipe_snapshot: _FrozenFile,
    torch: Any,
) -> dict[str, object]:
    target = UnifiedReaderConfig(**dict(_mapping(inspection["target_config"], "target config")))
    attempt_guard.require("before_training")
    output_anchor.require("before_training")
    training_anchor.require("before_training")
    train_unified_reader(
        records_path=Path(str(inspection["composite_records"])),
        dataset_root=Path(str(inspection["composite_dataset_root"])),
        output_dir=training,
        config=target,
        device="cuda:0",
        epochs=FIXED_EPOCHS,
        batch_size=FIXED_BATCH_SIZE,
        learning_rate=FIXED_LEARNING_RATE,
        weight_decay=FIXED_WEIGHT_DECAY,
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
        init_checkpoint=Path(str(inspection["source_checkpoint"])),
        init_checkpoint_mode=INIT_CHECKPOINT_MODE_RECIPIENT_VISUAL_CONTEXT_REINIT,
        ctc_loss_weight=1.0,
        structured_loss_weight=1.0,
        payment_bank_prefix_min_support=3,
        seed=FIXED_SEED,
        num_workers=FIXED_NUM_WORKERS,
        prefetch_factor=FIXED_PREFETCH_FACTOR,
        persistent_workers=True,
        train_progress_every=FIXED_PROGRESS_EVERY,
        cuda_tf32=True,
        cudnn_benchmark=True,
    )
    attempt_guard.require("after_training")
    output_anchor.require("after_training")
    training_anchor.require("after_training")
    closing = inspect_exact8_subject(
        full_records=full_records,
        original_dataset_root=original_dataset_root,
        full_crop_pilot_root=full_crop_pilot_root,
        source_contract_path=source_contract_path,
        candidate_pilot_evidence_path=candidate_pilot_evidence_path,
        failure_evidence_path=failure_evidence_path,
        failure_attempt_registry=failure_attempt_registry,
        overlay_contract_path=overlay_contract_path,
        torch=torch,
    )
    _json_equal(closing, inspection, "exact8 authority closure")
    summary_path = _existing(
        training / "training_summary.json",
        directory=False,
        description="exact8 training summary",
    )
    best_path = _existing(
        training / "best.pt", directory=False, description="exact8 best checkpoint"
    )
    last_path = _existing(
        training / "last.pt", directory=False, description="exact8 last checkpoint"
    )
    labels_path = _existing(
        training / "labels.json", directory=False, description="exact8 labels"
    )
    summary_snapshot = _freeze_file(summary_path, description="exact8 training summary")
    best_snapshot = _freeze_file(best_path, description="exact8 best checkpoint")
    last_snapshot = _freeze_file(last_path, description="exact8 last checkpoint")
    labels_snapshot = _freeze_file(labels_path, description="exact8 labels")
    summary = _strict_json_bytes(
        summary_snapshot.data, description="frozen exact8 training summary"
    )
    labels = _strict_json_bytes(labels_snapshot.data, description="frozen exact8 labels")
    decision = evaluate_exact8_summary(summary, inspection=inspection, recipe=recipe)
    data_label_proof = _mapping(
        _mapping(inspection["baseline"], "A8 baseline").get("data_label_proof"),
        "A8 data/label proof",
    )
    _validate_labels(
        labels, summary=summary, data_label_proof=data_label_proof
    )
    records = [_mapping(record, "exact8 record") for record in summary["records"]]
    best_epoch = int(_mapping(decision["observed"], "observed")["best_epoch"])
    source_payload = _load_checkpoint(Path(str(inspection["source_checkpoint"])), torch=torch)
    source_state = _state_dict(source_payload, description="original pilot best")
    _checkpoint_artifact(
        best_snapshot,
        summary=summary,
        labels=labels,
        epoch_record=records[best_epoch - 1],
        expected_epoch=best_epoch,
        source_state=source_state,
        torch=torch,
        description="exact8 best checkpoint",
    )
    _checkpoint_artifact(
        last_snapshot,
        summary=summary,
        labels=labels,
        epoch_record=records[-1],
        expected_epoch=FIXED_EPOCHS,
        source_state=source_state,
        torch=torch,
        description="exact8 last checkpoint",
    )
    frozen_artifacts = {
        "recipe": recipe_snapshot,
        "training_summary": summary_snapshot,
        "best_checkpoint": best_snapshot,
        "last_checkpoint": last_snapshot,
        "labels": labels_snapshot,
        "attempt_lock": attempt_snapshot,
    }
    for name, snapshot in frozen_artifacts.items():
        _require_frozen_current(snapshot, description=f"exact8 {name}")
    _assert_no_delivery_artifacts(output)
    for name, snapshot in frozen_artifacts.items():
        _require_frozen_current(snapshot, description=f"exact8 {name}")
    artifacts = {
        name: _binding_from_frozen(snapshot)
        for name, snapshot in frozen_artifacts.items()
    }
    sealed_payload = _sealed_decision_payload(
        decision=decision,
        inspection=inspection,
        artifacts=artifacts,
    )
    sealed = {
        **sealed_payload,
        "integrity_sha256": _canonical_sha256(sealed_payload),
    }
    decision_path = output / "recipient_multiview_exact8_decision.json"
    attempt_guard.require("before_decision_publication")
    output_anchor.require("before_decision_publication")
    training_anchor.require("before_decision_publication")
    _write_json_no_clobber(decision_path, sealed)
    attempt_guard.require("after_decision_publication")
    output_anchor.require("after_decision_publication")
    training_anchor.require("after_decision_publication")
    decision_snapshot = _freeze_file(decision_path, description="exact8 decision")
    closing_after_decision = inspect_exact8_subject(
        full_records=full_records,
        original_dataset_root=original_dataset_root,
        full_crop_pilot_root=full_crop_pilot_root,
        source_contract_path=source_contract_path,
        candidate_pilot_evidence_path=candidate_pilot_evidence_path,
        failure_evidence_path=failure_evidence_path,
        failure_attempt_registry=failure_attempt_registry,
        overlay_contract_path=overlay_contract_path,
        torch=torch,
    )
    _json_equal(
        closing_after_decision, inspection, "post-decision exact8 authority closure"
    )
    for name, snapshot in frozen_artifacts.items():
        _require_frozen_current(snapshot, description=f"sealed exact8 {name}")
    _require_frozen_current(decision_snapshot, description="sealed exact8 decision")
    _assert_no_delivery_artifacts(output)
    for name, snapshot in frozen_artifacts.items():
        _require_frozen_current(snapshot, description=f"sealed exact8 {name}")
    _require_frozen_current(decision_snapshot, description="sealed exact8 decision")
    _assert_exact_output_tree(
        output,
        expected_files=(
            recipe_path,
            summary_path,
            best_path,
            last_path,
            labels_path,
            decision_path,
        ),
    )
    attempt_guard.require("before_run_return")
    output_anchor.require("before_run_return")
    training_anchor.require("before_run_return")
    return sealed


def verify_exact8_decision(
    *,
    full_records: Path,
    original_dataset_root: Path,
    full_crop_pilot_root: Path,
    source_contract_path: Path,
    candidate_pilot_evidence_path: Path,
    failure_evidence_path: Path,
    failure_attempt_registry: Path,
    overlay_contract_path: Path,
    output_root: Path,
    attempt_lock: Path,
    decision_path: Path | None = None,
    torch: Any | None = None,
) -> dict[str, object]:
    """Independently reopen and reproduce one exact8 PASS or FAIL decision."""

    if torch is None:
        torch, _ = _require_torch()
    inspection = inspect_exact8_subject(
        full_records=full_records,
        original_dataset_root=original_dataset_root,
        full_crop_pilot_root=full_crop_pilot_root,
        source_contract_path=source_contract_path,
        candidate_pilot_evidence_path=candidate_pilot_evidence_path,
        failure_evidence_path=failure_evidence_path,
        failure_attempt_registry=failure_attempt_registry,
        overlay_contract_path=overlay_contract_path,
        torch=torch,
    )
    output = _existing(
        output_root, directory=True, description="completed fixed2 exact8 output"
    )
    _assert_output_disjoint(
        output,
        protected_directories=(
            Path(original_dataset_root).resolve(),
            Path(full_crop_pilot_root).resolve(),
            Path(failure_attempt_registry).resolve(),
            Path(attempt_lock).resolve().parent,
            Path(str(inspection["composite_records"])).resolve().parent,
            *(
                Path(path).resolve()
                for path in inspection.get("guard_directories", [])
                if isinstance(path, str)
            ),
        ),
    )
    attempt_path = _existing(
        attempt_lock, directory=False, description="exact8 attempt lock"
    )
    attempt_snapshot = _freeze_file(
        attempt_path, description="verified exact8 attempt lock"
    )
    _validate_attempt(
        attempt_path,
        inspection=inspection,
        output_root=output,
        snapshot=attempt_snapshot,
    )
    canonical_decision = output / "recipient_multiview_exact8_decision.json"
    supplied_decision = canonical_decision if decision_path is None else decision_path
    decision_file = _existing(
        supplied_decision, directory=False, description="exact8 decision evidence"
    )
    _samefile(decision_file, canonical_decision, "canonical exact8 decision")
    decision_snapshot = _freeze_file(
        decision_file, description="verified exact8 decision evidence"
    )
    sealed = _strict_json_bytes(
        decision_snapshot.data, description="frozen exact8 decision evidence"
    )
    claimed_integrity = _require_hex(
        sealed.get("integrity_sha256"), "exact8 decision integrity"
    )
    unsigned = {key: value for key, value in sealed.items() if key != "integrity_sha256"}
    _require_equal(
        claimed_integrity,
        _canonical_sha256(unsigned),
        "exact8 decision integrity",
    )

    training = output / "training-multiview-fixed2-exact8"
    expected_paths = {
        "recipe": output / "recipient_multiview_exact8_recipe.json",
        "training_summary": training / "training_summary.json",
        "best_checkpoint": training / "best.pt",
        "last_checkpoint": training / "last.pt",
        "labels": training / "labels.json",
        "attempt_lock": attempt_path,
    }
    snapshots = {
        "recipe": _freeze_file(
            expected_paths["recipe"], description="verified exact8 recipe"
        ),
        "training_summary": _freeze_file(
            expected_paths["training_summary"],
            description="verified exact8 training summary",
        ),
        "best_checkpoint": _freeze_file(
            expected_paths["best_checkpoint"],
            description="verified exact8 best checkpoint",
        ),
        "last_checkpoint": _freeze_file(
            expected_paths["last_checkpoint"],
            description="verified exact8 last checkpoint",
        ),
        "labels": _freeze_file(
            expected_paths["labels"], description="verified exact8 labels"
        ),
        "attempt_lock": attempt_snapshot,
    }
    artifacts = _mapping(sealed.get("training_artifacts"), "decision artifacts")
    if set(artifacts) != set(expected_paths):
        raise ValueError("exact8 decision artifact key set changed")
    for name, expected in expected_paths.items():
        _verify_frozen_binding(
            artifacts.get(name),
            snapshot=snapshots[name],
            expected_path=expected,
            description=f"exact8 decision artifact {name}",
        )

    recipe = _strict_json_bytes(
        snapshots["recipe"].data, description="frozen verified exact8 recipe"
    )
    expected_recipe = _recipe(inspection)
    _json_equal(recipe, expected_recipe, "sealed exact8 recipe")
    summary = _strict_json_bytes(
        snapshots["training_summary"].data,
        description="frozen verified exact8 training summary",
    )
    labels = _strict_json_bytes(
        snapshots["labels"].data, description="frozen verified exact8 labels"
    )
    recomputed_decision = evaluate_exact8_summary(
        summary, inspection=inspection, recipe=recipe
    )
    data_label_proof = _mapping(
        _mapping(inspection["baseline"], "A8 baseline").get("data_label_proof"),
        "A8 data/label proof",
    )
    _validate_labels(
        labels, summary=summary, data_label_proof=data_label_proof
    )
    records = [_mapping(record, "exact8 record") for record in summary["records"]]
    best_epoch = int(
        _mapping(recomputed_decision["observed"], "recomputed observed")["best_epoch"]
    )
    source_payload = _load_checkpoint(
        Path(str(inspection["source_checkpoint"])), torch=torch
    )
    source_state = _state_dict(source_payload, description="original pilot best")
    _checkpoint_artifact(
        snapshots["best_checkpoint"],
        summary=summary,
        labels=labels,
        epoch_record=records[best_epoch - 1],
        expected_epoch=best_epoch,
        source_state=source_state,
        torch=torch,
        description="verified exact8 best checkpoint",
    )
    _checkpoint_artifact(
        snapshots["last_checkpoint"],
        summary=summary,
        labels=labels,
        epoch_record=records[-1],
        expected_epoch=FIXED_EPOCHS,
        source_state=source_state,
        torch=torch,
        description="verified exact8 last checkpoint",
    )
    expected_payload = _sealed_decision_payload(
        decision=recomputed_decision,
        inspection=inspection,
        artifacts=artifacts,
    )
    _json_equal(unsigned, expected_payload, "recomputed exact8 decision")
    for name, snapshot in snapshots.items():
        _require_frozen_current(snapshot, description=f"verified exact8 {name}")
    _require_frozen_current(
        decision_snapshot, description="verified exact8 decision evidence"
    )
    closing = inspect_exact8_subject(
        full_records=full_records,
        original_dataset_root=original_dataset_root,
        full_crop_pilot_root=full_crop_pilot_root,
        source_contract_path=source_contract_path,
        candidate_pilot_evidence_path=candidate_pilot_evidence_path,
        failure_evidence_path=failure_evidence_path,
        failure_attempt_registry=failure_attempt_registry,
        overlay_contract_path=overlay_contract_path,
        torch=torch,
    )
    _json_equal(closing, inspection, "decision authority closure")
    for name, snapshot in snapshots.items():
        _require_frozen_current(snapshot, description=f"closed exact8 {name}")
    _require_frozen_current(
        decision_snapshot, description="closed exact8 decision evidence"
    )
    _assert_no_delivery_artifacts(output)
    for name, snapshot in snapshots.items():
        _require_frozen_current(snapshot, description=f"sealed exact8 {name}")
    _require_frozen_current(
        decision_snapshot, description="sealed exact8 decision evidence"
    )
    _assert_exact_output_tree(
        output,
        expected_files=(
            snapshots["recipe"].path,
            snapshots["training_summary"].path,
            snapshots["best_checkpoint"].path,
            snapshots["last_checkpoint"].path,
            snapshots["labels"].path,
            decision_file,
        ),
    )
    return dict(sealed)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Guarded recipient fixed2 exact8 route")
    subparsers = parser.add_subparsers(dest="command", required=True)

    def common(target: argparse.ArgumentParser) -> None:
        target.add_argument("--full-records", type=Path, required=True)
        target.add_argument("--dataset-root", type=Path, required=True)
        target.add_argument("--full-crop-pilot-root", type=Path, required=True)
        target.add_argument("--source-contract", type=Path, required=True)
        target.add_argument("--candidate-pilot-evidence", type=Path, required=True)
        target.add_argument("--failure-evidence", type=Path, required=True)
        target.add_argument("--failure-attempt-registry", type=Path, required=True)
        target.add_argument("--overlay-contract", type=Path, required=True)

    inspect_parser = subparsers.add_parser("inspect")
    common(inspect_parser)
    run_parser = subparsers.add_parser("run")
    common(run_parser)
    run_parser.add_argument("--output-root", type=Path, required=True)
    run_parser.add_argument("--attempt-lock", type=Path, required=True)
    verify_parser = subparsers.add_parser("verify-decision")
    common(verify_parser)
    verify_parser.add_argument("--output-root", type=Path, required=True)
    verify_parser.add_argument("--attempt-lock", type=Path, required=True)
    verify_parser.add_argument("--decision", type=Path)
    return parser


def _common_kwargs(args: argparse.Namespace) -> dict[str, Path]:
    return {
        "full_records": args.full_records,
        "original_dataset_root": args.dataset_root,
        "full_crop_pilot_root": args.full_crop_pilot_root,
        "source_contract_path": args.source_contract,
        "candidate_pilot_evidence_path": args.candidate_pilot_evidence,
        "failure_evidence_path": args.failure_evidence,
        "failure_attempt_registry": args.failure_attempt_registry,
        "overlay_contract_path": args.overlay_contract,
    }


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "inspect":
        payload = inspect_exact8_subject(**_common_kwargs(args))
    elif args.command == "run":
        payload = run_exact8(
            **_common_kwargs(args),
            output_root=args.output_root,
            attempt_lock=args.attempt_lock,
        )
    else:
        payload = verify_exact8_decision(
            **_common_kwargs(args),
            output_root=args.output_root,
            attempt_lock=args.attempt_lock,
            decision_path=args.decision,
        )
    print(json.dumps(payload, ensure_ascii=False, allow_nan=False, sort_keys=True))
    if args.command == "run" and payload.get("passed") is not True:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
