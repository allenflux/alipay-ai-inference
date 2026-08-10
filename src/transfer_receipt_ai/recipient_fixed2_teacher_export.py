"""Export the canonical two-view recipient teacher source.

This is intentionally a separate producer from the four-view diagnostic
export.  It emits only the two views consumed by the fixed2 route:
``standard`` and ``fixed_value``.  The public materialize/verify boundary is
Windows-only and uses parent-handle-relative directory creation plus a
handle-relative no-replace rename.  A private POSIX analysis publisher exists
only so the semantic verifier can be exercised in unit tests; it cannot mint a
formal kind, authority, or optimizer-ready artifact.

All generated selected-view pixels are closed globally against their blind
manifest owner.  A decoded pixel hash may not cross a split, target, or group,
including the same-target/different-group case.  The complete collision pass
runs before any staging directory is created.
"""

from __future__ import annotations

import argparse
import ctypes
import errno
import hashlib
import io
import json
import os
import stat
import sys
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageOps

from .ocr_pseudolabels import _bbox, _crop_digest
from .ocr_unified_dataset import RECIPIENT_QUALITY_POLICY_VERSION
from .pipeline import crop_field_with_margin
from .recipient_multiview_teacher_export import (
    ALLOWED_SPLITS,
    FIXED_VALUE_LEFT_TRIM,
    HELD_OUT_SPLITS,
    SCHEMA_VERSION,
    STANDARD_MARGIN_RATIO,
    SUPPORTED_UNIFIED_KINDS,
    TRAIN_SPLIT,
    _GeneratedViewOwner,
    _absolute_existing_file,
    _assert_no_reparse_components,
    _assert_output_separate,
    _canonical_sha256,
    _fixed_value_view,
    _path_identity,
    _production_standard_view,
    _register_generated_view_owner,
    _relative_existing_file,
    _require_sha256,
    _sha256,
    _target_from_train_record,
)
from .status_crops import _result_payload, _source_path, reconstruct_rectified


KIND = "receipt_recipient_fixed2_teacher_train_export_v1"
ANALYSIS_KIND = "receipt_recipient_fixed2_teacher_train_export_analysis_fixture_v1"
RECORD_KIND = "receipt_recipient_fixed2_teacher_train_record_v1"
ANALYSIS_RECORD_KIND = (
    "receipt_recipient_fixed2_teacher_train_record_analysis_fixture_v1"
)
VIEWS = ("standard", "fixed_value")
PUBLICATION_AUTHORITY = (
    "windows_parent_relative_ntcreatefile_two_stage_hard_pin_fixed2_teacher_v1"
)
ANALYSIS_PUBLICATION_AUTHORITY = "posix_no_replace_fixed2_teacher_analysis_fixture_v1"
HARD_ATTESTATION_SCHEME = "two_stage_code_pinned_contract_sha_size_subject_v1"
CONTRACT_NAME = "dataset.contract.json"
ANALYSIS_CONTRACT_NAME = "fixed2_teacher.analysis.json"
MANIFEST_NAME = "multiview_train.jsonl"
SUBJECT_DOMAIN = "receipt-recipient-fixed2-teacher-subject-v1"
ADAPTER_MARKER = "strict_recipient_multiview_overlay_loader_not_implemented"
RECORD_KEYS = frozenset(
    (
        "schema_version",
        "kind",
        "id",
        "group_id",
        "source_record_id",
        "split",
        "field",
        "view",
        "image",
        "text",
        "target_sha256",
        "target_source",
        "target_source_manifest_sha256",
        "optimizer_supervision_split_eligible",
        "optimizer_consumable",
        "group_closure_sha256",
        "group_view_count",
        "source",
        "source_sha256",
        "result_json",
        "result_json_sha256",
        "bbox_rectified",
        "paddle_crop",
        "paddle_crop_pixel_sha256",
        "paddle_crop_file_sha256",
        "view_width",
        "view_height",
        "view_pixel_sha256",
        "view_file_sha256",
    )
)
CONTRACT_KEYS = frozenset(
    (
        "schema_version",
        "kind",
        "record_kind",
        "publication_profile",
        "formal_windows_publication",
        "analysis_fixture",
        "publication_authority",
        "hard_attestation_scheme",
        "public_verification_requires_hard_attestation",
        "publication_identity",
        "nominal_output_root",
        "analysis_only",
        "production_route_authorized",
        "source_manifest",
        "source_manifest_sha256",
        "source_manifest_semantic_sha256",
        "source_dataset_contract",
        "source_dataset_contract_sha256",
        "source_dataset_kind",
        "source_dataset_root",
        "target_source",
        "target_label_authority",
        "target_recomputed",
        "optimizer_supervision_splits",
        "optimizer_input_ready",
        "records_role",
        "optimizer_adapter_required",
        "held_out_splits_excluded",
        "held_out_target_values_used",
        "held_out_target_values_validated",
        "held_out_target_values_emitted",
        "source_manifest_split_counts",
        "source_split_counts",
        "source_train_recipient_records",
        "source_train_records_without_recipient_target",
        "output_records",
        "output_split_counts",
        "view_order",
        "view_counts",
        "view_geometry",
        "selected_view_hash_closure",
        "producer_subject_id",
        "subject_domain",
        "subject_path_stable",
        "subject_output_stable",
        "subject_code_stable",
        "train_manifest",
        "train_manifest_sha256",
        "artifacts",
        "publication",
        "commit_marker",
        "publication_complete",
        "failure_policy",
        "integrity_sha256",
    )
)

_WINDOWS_FILE_SHARE_READ = 0x00000001
_WINDOWS_FILE_SHARE_WRITE = 0x00000002
_WINDOWS_FILE_SHARE_DELETE = 0x00000004
_WINDOWS_FILE_READ_ATTRIBUTES = 0x00000080
_WINDOWS_FILE_TRAVERSE = 0x00000020
_WINDOWS_DELETE = 0x00010000
_WINDOWS_GENERIC_EXECUTE = 0x20000000
_WINDOWS_SYNCHRONIZE = 0x00100000
_WINDOWS_OPEN_EXISTING = 3
_WINDOWS_FILE_CREATE = 2
_WINDOWS_FILE_OPEN = 1
_WINDOWS_FILE_DIRECTORY_FILE = 0x00000001
_WINDOWS_FILE_SYNCHRONOUS_IO_NONALERT = 0x00000020
_WINDOWS_FILE_OPEN_REPARSE_POINT = 0x00200000
_WINDOWS_FILE_FLAG_BACKUP_SEMANTICS = 0x02000000
_WINDOWS_FILE_FLAG_OPEN_REPARSE_POINT = 0x00200000
_WINDOWS_DIRECTORY_ACCESS = (
    _WINDOWS_FILE_READ_ATTRIBUTES | _WINDOWS_FILE_TRAVERSE | _WINDOWS_GENERIC_EXECUTE
)
_WINDOWS_DIRECTORY_SHARE = _WINDOWS_FILE_SHARE_READ | _WINDOWS_FILE_SHARE_WRITE

FileIdentity = tuple[int, int, int, int, int, str]
DirectoryIdentity = tuple[int, int, int]


def _running_on_windows() -> bool:
    """Policy boundary kept separate for deterministic profile attack tests."""

    return os.name == "nt"


@dataclass(frozen=True)
class _PreparedView:
    name: str
    pixel_sha256: str
    width: int
    height: int


@dataclass(frozen=True)
class _PreparedRow:
    line_number: int
    record_id: str
    group_id: str
    target: str
    target_sha256: str
    source: Path
    source_sha256: str
    result_json: Path
    result_json_sha256: str
    paddle_crop: Path
    paddle_crop_pixel_sha256: str
    paddle_crop_file_sha256: str
    bbox: tuple[float, float, float, float]
    views: tuple[_PreparedView, ...]
    group_closure_sha256: str


@dataclass(frozen=True)
class _FrozenFileSnapshot:
    path: Path
    identity: FileIdentity
    data: bytes

    @property
    def sha256(self) -> str:
        return self.identity[-1]

    @property
    def size_bytes(self) -> int:
        return self.identity[3]


@dataclass(frozen=True)
class _FrozenFileSeal:
    """Lightweight identity retained after transient snapshot bytes are freed."""

    path: Path
    identity: FileIdentity

    @property
    def sha256(self) -> str:
        return self.identity[-1]

    @property
    def size_bytes(self) -> int:
        return self.identity[3]


def _snapshot_seal(snapshot: _FrozenFileSnapshot) -> _FrozenFileSeal:
    return _FrozenFileSeal(path=snapshot.path, identity=snapshot.identity)


def _snapshot_use_hook(
    checkpoint: str, *, snapshot: _FrozenFileSnapshot, description: str
) -> None:
    """No-op hook for deterministic swap/read/restore regression tests."""


def _snapshot_file(path: Path, *, description: str) -> _FrozenFileSnapshot:
    """Read one regular file once and bind every consumer to those exact bytes."""

    raw = Path(os.path.abspath(os.fspath(path)))
    _assert_no_reparse_components(raw, description=description)
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    if os.name == "nt":
        flags |= getattr(os, "O_BINARY", 0)
    try:
        descriptor = os.open(raw, flags)
    except OSError as error:
        raise ValueError(f"unable to open snapshot for {description}: {raw}") from error
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or _is_reparse_stat(before):
            raise ValueError(f"{description} must be a regular non-reparse file: {raw}")
        chunks: list[bytes] = []
        while True:
            block = os.read(descriptor, 1024 * 1024)
            if not block:
                break
            chunks.append(block)
        data = b"".join(chunks)
        after = os.fstat(descriptor)
        before_identity = (
            before.st_dev,
            before.st_ino,
            _file_attributes(before),
            before.st_size,
            before.st_mtime_ns,
        )
        after_identity = (
            after.st_dev,
            after.st_ino,
            _file_attributes(after),
            after.st_size,
            after.st_mtime_ns,
        )
        if before_identity != after_identity or len(data) != after.st_size:
            raise ValueError(f"{description} changed while its bytes were snapshotted: {raw}")
        identity: FileIdentity = (
            *after_identity,
            hashlib.sha256(data).hexdigest(),
        )
        # Keep the lexical absolute name that was opened.  Resolving the name
        # after closing the descriptor would be a second pathname lookup and
        # could bind this frozen byte snapshot to a replacement object.
        return _FrozenFileSnapshot(path=raw, identity=identity, data=data)
    finally:
        os.close(descriptor)


def _strict_json_bytes(data: bytes, *, description: str) -> dict[str, Any]:
    def reject_constant(value: str) -> None:
        raise ValueError(f"{description}: non-finite JSON constant {value!r}")

    def reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"{description}: duplicate JSON key {key!r}")
            result[key] = value
        return result

    try:
        text = data.decode("utf-8-sig")
        raw = json.loads(
            text,
            parse_constant=reject_constant,
            object_pairs_hook=reject_duplicate_pairs,
        )
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"unable to read strict JSON object {description}: {error}") from error
    if not isinstance(raw, dict):
        raise ValueError(f"{description}: expected a JSON object")
    return raw


def _exact_json_value(actual: object, expected: object) -> bool:
    """Compare JSON-like values without Python's bool/int equivalence."""

    if type(actual) is not type(expected):
        return False
    if isinstance(expected, dict):
        return set(actual) == set(expected) and all(  # type: ignore[arg-type]
            _exact_json_value(actual[key], value)  # type: ignore[index]
            for key, value in expected.items()
        )
    if isinstance(expected, list):
        return len(actual) == len(expected) and all(  # type: ignore[arg-type]
            _exact_json_value(left, right)
            for left, right in zip(actual, expected)  # type: ignore[arg-type]
        )
    return actual == expected


def _snapshot_json(
    path: Path, *, description: str, hook_prefix: str
) -> tuple[_FrozenFileSnapshot, dict[str, Any]]:
    snapshot = _snapshot_file(path, description=description)
    _snapshot_use_hook(
        f"{hook_prefix}_before_parse",
        snapshot=snapshot,
        description=description,
    )
    value = _strict_json_bytes(snapshot.data, description=description)
    _snapshot_use_hook(
        f"{hook_prefix}_after_parse",
        snapshot=snapshot,
        description=description,
    )
    return snapshot, value


def _snapshot_rgb(
    snapshot: _FrozenFileSnapshot, *, description: str, hook_prefix: str
) -> np.ndarray:
    _snapshot_use_hook(
        f"{hook_prefix}_before_decode",
        snapshot=snapshot,
        description=description,
    )
    try:
        with Image.open(io.BytesIO(snapshot.data)) as opened:
            upright = ImageOps.exif_transpose(opened).convert("RGB")
            pixels = np.asarray(upright).copy()
    except (OSError, ValueError) as error:
        raise ValueError(f"unable to decode {description} snapshot: {snapshot.path}") from error
    _snapshot_use_hook(
        f"{hook_prefix}_after_decode",
        snapshot=snapshot,
        description=description,
    )
    if pixels.ndim != 3 or pixels.shape[2] != 3 or min(pixels.shape[:2]) <= 0:
        raise ValueError(f"{description} snapshot is not a non-empty RGB image")
    return pixels


def _strict_json(path: Path) -> dict[str, Any]:
    _snapshot, value = _snapshot_json(
        path,
        description=f"strict JSON {path}",
        hook_prefix="strict_json",
    )
    return value


def _strict_jsonl_records_bytes(
    data: bytes, *, description: str
) -> list[tuple[int, dict[str, object]]]:
    try:
        text = data.decode("utf-8-sig")
    except UnicodeError as error:
        raise ValueError(f"{description}: invalid UTF-8") from error
    records: list[tuple[int, dict[str, object]]] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue

        def reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
            value: dict[str, Any] = {}
            for key, child in pairs:
                if key in value:
                    raise ValueError(
                        f"{description}:{line_number}: duplicate JSON key {key!r}"
                    )
                value[key] = child
            return value

        try:
            raw = json.loads(line, object_pairs_hook=reject_duplicate_pairs)
        except json.JSONDecodeError as error:
            raise ValueError(f"{description}:{line_number}: invalid JSON") from error
        if not isinstance(raw, Mapping):
            raise ValueError(f"{description}:{line_number}: record must be an object")
        records.append((line_number, dict(raw)))
    if not records:
        raise ValueError(f"{description}: manifest is empty")
    return records


def _snapshot_records(
    path: Path, *, description: str, hook_prefix: str
) -> tuple[_FrozenFileSnapshot, list[tuple[int, dict[str, object]]]]:
    snapshot = _snapshot_file(path, description=description)
    _snapshot_use_hook(
        f"{hook_prefix}_before_parse",
        snapshot=snapshot,
        description=description,
    )
    records = _strict_jsonl_records_bytes(snapshot.data, description=description)
    _snapshot_use_hook(
        f"{hook_prefix}_after_parse",
        snapshot=snapshot,
        description=description,
    )
    return snapshot, records


def _validated_contract_payload(
    raw: Mapping[str, Any], *, description: str
) -> dict[str, object]:
    contract = dict(raw)
    if contract.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"{description}: unsupported unified dataset contract schema")
    if contract.get("kind") not in SUPPORTED_UNIFIED_KINDS:
        raise ValueError(f"{description}: fixed2 export requires a v11-v13 manifest")
    if contract.get("recipient_charset_source") != "train_only_anchored_recipient_value":
        raise ValueError(f"{description}: recipient charset is not train-only anchored text")
    quality = contract.get("recipient_quality_policy")
    if (
        not isinstance(quality, Mapping)
        or quality.get("version") != RECIPIENT_QUALITY_POLICY_VERSION
        or quality.get("requires_leading_recipient_label") is not True
        or quality.get("target") != "anchored_recipient_value"
    ):
        raise ValueError(f"{description}: recipient quality policy is not frozen")
    return contract


def _file_attributes(info: os.stat_result) -> int:
    return int(getattr(info, "st_file_attributes", 0))


def _is_reparse_stat(info: os.stat_result) -> bool:
    return stat.S_ISLNK(info.st_mode) or bool(
        _file_attributes(info) & int(getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400))
    )


def _file_identity(path: Path) -> FileIdentity:
    before = path.stat(follow_symlinks=False)
    if not stat.S_ISREG(before.st_mode) or _is_reparse_stat(before):
        raise ValueError(f"expected a regular non-reparse file: {path}")
    digest = _sha256(path)
    after = path.stat(follow_symlinks=False)
    left = (before.st_dev, before.st_ino, _file_attributes(before), before.st_size, before.st_mtime_ns)
    right = (after.st_dev, after.st_ino, _file_attributes(after), after.st_size, after.st_mtime_ns)
    if left != right:
        raise ValueError(f"file changed while recording identity: {path}")
    return (*right, digest)


def _directory_identity(path: Path) -> DirectoryIdentity:
    if os.name == "nt":
        handle = _windows_open_path_directory(path, desired_access=_WINDOWS_DIRECTORY_ACCESS, share_access=(
            _WINDOWS_FILE_SHARE_READ | _WINDOWS_FILE_SHARE_WRITE | _WINDOWS_FILE_SHARE_DELETE
        ))
        try:
            return _windows_directory_identity(handle)
        finally:
            _windows_close(handle)
    info = path.stat(follow_symlinks=False)
    if not stat.S_ISDIR(info.st_mode) or _is_reparse_stat(info):
        raise ValueError(f"expected a regular non-reparse directory: {path}")
    return info.st_dev, info.st_ino, _file_attributes(info)


def _same_directory_identity(path: Path, expected: DirectoryIdentity) -> bool:
    try:
        return _directory_identity(path) == expected
    except (OSError, ValueError):
        return False


def _identity_payload(identity: FileIdentity) -> dict[str, object]:
    device, inode, attributes, size, mtime_ns, digest = identity
    return {
        "volume_serial_number": int(device),
        "file_index": int(inode),
        "file_attributes": int(attributes),
        "size_bytes": int(size),
        "mtime_ns": int(mtime_ns),
        "sha256": digest,
    }


def _directory_identity_payload(identity: DirectoryIdentity) -> dict[str, object]:
    device, inode, attributes = identity
    return {
        "volume_serial_number": int(device),
        "file_index": int(inode),
        "file_attributes": int(attributes),
    }


@dataclass
class _DirectoryLease:
    path: Path
    identity: DirectoryIdentity
    windows_handle: int | None = None
    posix_fd: int | None = None
    rename_capable: bool = False

    def close(self) -> None:
        if self.windows_handle is not None:
            _windows_close(self.windows_handle)
            self.windows_handle = None
        if self.posix_fd is not None:
            try:
                os.close(self.posix_fd)
            except OSError:
                pass
            self.posix_fd = None


def _windows_close(handle: int) -> None:
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    function = kernel32.CloseHandle
    function.argtypes = (ctypes.c_void_p,)
    function.restype = ctypes.c_int
    function(ctypes.c_void_p(handle))


def _windows_open_path_directory(path: Path, *, desired_access: int, share_access: int) -> int:
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    function = kernel32.CreateFileW
    function.argtypes = (
        ctypes.c_wchar_p, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_void_p,
        ctypes.c_uint32, ctypes.c_uint32, ctypes.c_void_p,
    )
    function.restype = ctypes.c_void_p
    handle = function(
        str(path), desired_access, share_access, None, _WINDOWS_OPEN_EXISTING,
        _WINDOWS_FILE_FLAG_BACKUP_SEMANTICS | _WINDOWS_FILE_FLAG_OPEN_REPARSE_POINT, None,
    )
    invalid = ctypes.c_void_p(-1).value
    if handle in (None, invalid):
        number = ctypes.get_last_error()
        raise OSError(number, os.strerror(number), os.fspath(path))
    return int(handle)


def _windows_directory_identity(handle: int) -> DirectoryIdentity:
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

    class Info(ctypes.Structure):
        _fields_ = (
            ("attributes", ctypes.c_uint32), ("creation_low", ctypes.c_uint32),
            ("creation_high", ctypes.c_uint32), ("access_low", ctypes.c_uint32),
            ("access_high", ctypes.c_uint32), ("write_low", ctypes.c_uint32),
            ("write_high", ctypes.c_uint32), ("volume", ctypes.c_uint32),
            ("size_high", ctypes.c_uint32), ("size_low", ctypes.c_uint32),
            ("links", ctypes.c_uint32), ("index_high", ctypes.c_uint32),
            ("index_low", ctypes.c_uint32),
        )

    info = Info()
    function = kernel32.GetFileInformationByHandle
    function.argtypes = (ctypes.c_void_p, ctypes.c_void_p)
    function.restype = ctypes.c_int
    if not function(ctypes.c_void_p(handle), ctypes.byref(info)):
        number = ctypes.get_last_error()
        raise OSError(number, os.strerror(number))
    if not int(info.attributes) & 0x10 or int(info.attributes) & 0x400:
        raise ValueError("Windows directory handle is not a non-reparse directory")
    return int(info.volume), (int(info.index_high) << 32) | int(info.index_low), int(info.attributes)


def _simple_name(name: str) -> str:
    if not name or name in {".", ".."} or any(value in name for value in ("/", "\\", ":", "\0")):
        raise ValueError("anchored creation requires one simple child name")
    return name


def _failed_ntstatus(return_status: int, completion_status: int) -> int | None:
    """Require synchronous NT completion; pending/informational is not success."""

    if return_status != 0:
        return return_status
    if completion_status != 0:
        return completion_status
    return None


def _windows_nt_directory(
    parent_handle: int,
    *,
    name: str,
    disposition: int,
    desired_access: int,
    share_access: int,
) -> int:
    name = _simple_name(name)
    ntdll = ctypes.WinDLL("ntdll", use_last_error=True)

    class UnicodeString(ctypes.Structure):
        _fields_ = (("length", ctypes.c_uint16), ("maximum_length", ctypes.c_uint16), ("buffer", ctypes.c_void_p))

    class ObjectAttributes(ctypes.Structure):
        _fields_ = (
            ("length", ctypes.c_uint32), ("root", ctypes.c_void_p),
            ("name", ctypes.POINTER(UnicodeString)), ("attributes", ctypes.c_uint32),
            ("security", ctypes.c_void_p), ("quality", ctypes.c_void_p),
        )

    class StatusUnion(ctypes.Union):
        _fields_ = (("status", ctypes.c_int32), ("pointer", ctypes.c_void_p))

    class IoStatus(ctypes.Structure):
        _fields_ = (("status_or_pointer", StatusUnion), ("information", ctypes.c_size_t))

    encoded = name.encode("utf-16-le")
    buffer = ctypes.create_string_buffer(encoded + b"\0\0")
    unicode = UnicodeString(len(encoded), len(encoded) + 2, ctypes.addressof(buffer))
    attributes = ObjectAttributes(ctypes.sizeof(ObjectAttributes), parent_handle, ctypes.pointer(unicode), 0x40, None, None)
    io_status = IoStatus()
    handle = ctypes.c_void_p()
    function = ntdll.NtCreateFile
    function.argtypes = (
        ctypes.POINTER(ctypes.c_void_p), ctypes.c_uint32, ctypes.POINTER(ObjectAttributes),
        ctypes.POINTER(IoStatus), ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint32,
        ctypes.c_uint32, ctypes.c_uint32, ctypes.c_void_p, ctypes.c_uint32,
    )
    function.restype = ctypes.c_int32
    options = _WINDOWS_FILE_DIRECTORY_FILE | _WINDOWS_FILE_OPEN_REPARSE_POINT
    if desired_access & _WINDOWS_SYNCHRONIZE:
        options |= _WINDOWS_FILE_SYNCHRONOUS_IO_NONALERT
    status = int(function(
        ctypes.byref(handle), desired_access, ctypes.byref(attributes), ctypes.byref(io_status),
        None, 0, share_access, disposition, options, None, 0,
    ))
    completion_status = int(io_status.status_or_pointer.status)
    failed_status = _failed_ntstatus(status, completion_status)
    if failed_status is not None or handle.value is None:
        if handle.value is not None:
            _windows_close(int(handle.value))
        translate = ntdll.RtlNtStatusToDosError
        translate.argtypes = (ctypes.c_int32,)
        translate.restype = ctypes.c_uint32
        status_to_translate = failed_status if failed_status is not None else -1073741823
        number = int(translate(status_to_translate))
        if number in {80, 183}:
            raise FileExistsError(number, os.strerror(number), name)
        raise OSError(number, os.strerror(number), name)
    return int(handle.value)


def _open_parent(path: Path) -> _DirectoryLease:
    identity = _directory_identity(path)
    if os.name == "nt":
        handle = _windows_open_path_directory(
            path,
            desired_access=_WINDOWS_DIRECTORY_ACCESS,
            share_access=_WINDOWS_DIRECTORY_SHARE,
        )
        observed = _windows_directory_identity(handle)
        if observed != identity:
            _windows_close(handle)
            raise ValueError("fixed2 producer output parent changed before lease acquisition")
        return _DirectoryLease(path, identity, windows_handle=handle)
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    observed_stat = os.fstat(descriptor)
    observed = (observed_stat.st_dev, observed_stat.st_ino, _file_attributes(observed_stat))
    if observed != identity:
        os.close(descriptor)
        raise ValueError("fixed2 producer output parent changed before lease acquisition")
    return _DirectoryLease(path, identity, posix_fd=descriptor)


def _create_child(parent: _DirectoryLease, *, name: str, rename_capable: bool) -> _DirectoryLease:
    name = _simple_name(name)
    path = parent.path / name
    if parent.windows_handle is not None:
        handle = _windows_nt_directory(
            parent.windows_handle,
            name=name,
            disposition=_WINDOWS_FILE_CREATE,
            desired_access=_WINDOWS_DIRECTORY_ACCESS | _WINDOWS_DELETE | _WINDOWS_SYNCHRONIZE,
            share_access=_WINDOWS_DIRECTORY_SHARE,
        )
        identity = _windows_directory_identity(handle)
        observer = _windows_nt_directory(
            parent.windows_handle,
            name=name,
            disposition=_WINDOWS_FILE_OPEN,
            desired_access=_WINDOWS_FILE_READ_ATTRIBUTES,
            share_access=(
                _WINDOWS_FILE_SHARE_READ | _WINDOWS_FILE_SHARE_WRITE | _WINDOWS_FILE_SHARE_DELETE
            ),
        )
        try:
            if _windows_directory_identity(observer) != identity:
                raise ValueError("created directory is not the parent-relative entry")
        finally:
            _windows_close(observer)
        return _DirectoryLease(path, identity, windows_handle=handle, rename_capable=rename_capable)
    if parent.posix_fd is None:
        raise OSError(errno.ENOTSUP, "fixed2 producer has no anchored parent", os.fspath(path))
    os.mkdir(name, mode=0o700, dir_fd=parent.posix_fd)
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(name, flags, dir_fd=parent.posix_fd)
    info = os.fstat(descriptor)
    identity = (info.st_dev, info.st_ino, _file_attributes(info))
    return _DirectoryLease(path, identity, posix_fd=descriptor, rename_capable=rename_capable)


def _require_lease(lease: _DirectoryLease) -> None:
    if lease.windows_handle is not None:
        if _windows_directory_identity(lease.windows_handle) != lease.identity:
            raise ValueError("fixed2 producer Windows directory lease changed")
    elif lease.posix_fd is not None:
        info = os.fstat(lease.posix_fd)
        if (info.st_dev, info.st_ino, _file_attributes(info)) != lease.identity:
            raise ValueError("fixed2 producer POSIX directory lease changed")
    else:
        raise ValueError("fixed2 producer directory lease is closed")


def _anchored_names(lease: _DirectoryLease) -> set[str]:
    _require_lease(lease)
    names = set(os.listdir(lease.posix_fd)) if lease.posix_fd is not None else {entry.name for entry in os.scandir(lease.path)}
    _require_lease(lease)
    return names


def _write_file(lease: _DirectoryLease, *, name: str, payload: bytes) -> FileIdentity:
    name = _simple_name(name)
    _require_lease(lease)
    if lease.posix_fd is not None:
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(name, flags, 0o600, dir_fd=lease.posix_fd)
        try:
            view = memoryview(payload)
            while view:
                written = os.write(descriptor, view)
                if written <= 0:
                    raise OSError("unable to complete fixed2 producer write")
                view = view[written:]
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    else:
        path = lease.path / name
        with path.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    _require_lease(lease)
    return _file_identity(lease.path / name)


def _rename_no_replace(parent: _DirectoryLease, source: _DirectoryLease, destination: Path) -> None:
    _require_lease(parent)
    _require_lease(source)
    if parent.windows_handle is not None:
        if source.windows_handle is None or not source.rename_capable:
            raise OSError(errno.ENOTSUP, "fixed2 producer source is not rename capable")

        class Prefix(ctypes.Structure):
            _fields_ = (
                ("flags", ctypes.c_uint32), ("root", ctypes.c_void_p),
                ("length", ctypes.c_uint32), ("name", ctypes.c_uint16 * 1),
            )

        class StatusUnion(ctypes.Union):
            _fields_ = (("status", ctypes.c_int32), ("pointer", ctypes.c_void_p))

        class IoStatus(ctypes.Structure):
            _fields_ = (("status", StatusUnion), ("information", ctypes.c_size_t))

        encoded = destination.name.encode("utf-16-le")
        size = ctypes.sizeof(Prefix) + len(encoded)
        buffer = ctypes.create_string_buffer(size)
        info = Prefix.from_buffer(buffer)
        info.flags = 0
        info.root = parent.windows_handle
        info.length = len(encoded)
        ctypes.memmove(ctypes.addressof(buffer) + Prefix.name.offset, encoded, len(encoded))
        status_block = IoStatus()
        ntdll = ctypes.WinDLL("ntdll", use_last_error=True)
        function = ntdll.NtSetInformationFile
        function.argtypes = (ctypes.c_void_p, ctypes.POINTER(IoStatus), ctypes.c_void_p, ctypes.c_uint32, ctypes.c_int)
        function.restype = ctypes.c_int32
        status = int(function(ctypes.c_void_p(source.windows_handle), ctypes.byref(status_block), ctypes.byref(buffer), size, 10))
        failed = status if status != 0 else int(status_block.status.status)
        if failed != 0:
            translate = ntdll.RtlNtStatusToDosError
            translate.argtypes = (ctypes.c_int32,)
            translate.restype = ctypes.c_uint32
            number = int(translate(failed))
            if number in {80, 183}:
                raise FileExistsError(number, os.strerror(number), os.fspath(destination))
            raise OSError(number, os.strerror(number), os.fspath(destination))
        return
    if parent.posix_fd is None or source.posix_fd is None:
        raise OSError(errno.ENOTSUP, "fixed2 analysis publisher has no POSIX descriptors")
    library = ctypes.CDLL(None, use_errno=True)
    if sys.platform == "darwin":
        function = library.renameatx_np
        function.argtypes = (ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint)
        function.restype = ctypes.c_int
        result = function(parent.posix_fd, os.fsencode(source.path.name), parent.posix_fd, os.fsencode(destination.name), 0x4)
    elif sys.platform.startswith("linux") and hasattr(library, "renameat2"):
        function = library.renameat2
        function.argtypes = (ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint)
        function.restype = ctypes.c_int
        result = function(parent.posix_fd, os.fsencode(source.path.name), parent.posix_fd, os.fsencode(destination.name), 0x1)
    else:
        raise OSError(errno.ENOTSUP, "fixed2 analysis no-replace rename unavailable")
    if result != 0:
        number = ctypes.get_errno()
        if number in {errno.EEXIST, errno.ENOTEMPTY}:
            raise FileExistsError(number, os.strerror(number), os.fspath(destination))
        raise OSError(number, os.strerror(number), os.fspath(destination))


def _png_bytes(pixels: np.ndarray) -> bytes:
    stream = io.BytesIO()
    Image.fromarray(pixels, mode="RGB").save(stream, format="PNG")
    return stream.getvalue()


def _json_bytes(value: Mapping[str, object]) -> bytes:
    return (json.dumps(dict(value), ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")


def _jsonl_bytes(rows: Sequence[Mapping[str, object]]) -> bytes:
    return "".join(
        json.dumps(dict(row), ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n"
        for row in rows
    ).encode("utf-8")


def _selected_views(
    *,
    manifest: Path,
    line_number: int,
    record_id: str,
    source_snapshot: _FrozenFileSnapshot,
    result_snapshot: _FrozenFileSnapshot,
    crop_snapshot: _FrozenFileSnapshot,
    declared_crop_sha256: str,
    bbox: tuple[float, float, float, float],
    hook_prefix: str,
) -> dict[str, np.ndarray]:
    if source_snapshot.identity[4] > result_snapshot.identity[4]:
        raise ValueError(
            f"{manifest}:{line_number}: live source is newer than its Paddle result; "
            "refusing to apply the frozen target to changed context pixels"
        )
    _snapshot_use_hook(
        f"{hook_prefix}_result_before_parse",
        snapshot=result_snapshot,
        description=f"result JSON for {record_id}",
    )
    document = _strict_json_bytes(
        result_snapshot.data,
        description=f"result JSON for {record_id}",
    )
    _snapshot_use_hook(
        f"{hook_prefix}_result_after_parse",
        snapshot=result_snapshot,
        description=f"result JSON for {record_id}",
    )
    payload = _result_payload(document)
    if payload is None:
        raise ValueError(f"{result_snapshot.path}: not a receipt result bundle")
    if _source_path(payload, result_snapshot.path) != source_snapshot.path:
        raise ValueError(f"{manifest}:{line_number}: manifest and result bundle source paths disagree")
    source_rgb = _snapshot_rgb(
        source_snapshot,
        description=f"source image for {record_id}",
        hook_prefix=f"{hook_prefix}_source",
    )
    rectified = reconstruct_rectified(payload, source_rgb)
    paddle_standard = np.ascontiguousarray(crop_field_with_margin(rectified, bbox))
    if paddle_standard.ndim != 3 or paddle_standard.shape[2] != 3 or min(paddle_standard.shape[:2]) <= 0:
        raise ValueError(f"{manifest}:{line_number}: standard recipient crop is empty")
    if _crop_digest(paddle_standard) != declared_crop_sha256:
        raise ValueError(f"{manifest}:{line_number}: reconstructed standard crop hash changed")
    stored_crop = _snapshot_rgb(
        crop_snapshot,
        description=f"Paddle crop for {record_id}",
        hook_prefix=f"{hook_prefix}_crop",
    )
    if not np.array_equal(stored_crop, paddle_standard):
        raise ValueError(f"{manifest}:{line_number}: stored Paddle crop pixels changed")
    standard = _production_standard_view(rectified, bbox)
    return {
        "standard": standard,
        "fixed_value": _fixed_value_view(standard),
    }


def _artifact_binding_from_snapshot(
    snapshot: _FrozenFileSnapshot | _FrozenFileSeal,
) -> dict[str, object]:
    return {
        "path": str(snapshot.path),
        "sha256": snapshot.sha256,
        "size_bytes": snapshot.size_bytes,
    }


def _artifact_binding(path: Path) -> dict[str, object]:
    return _artifact_binding_from_snapshot(
        _snapshot_file(path, description=f"artifact binding {path}")
    )


def _resnapshot_matches(
    expected: _FrozenFileSnapshot | _FrozenFileSeal, *, description: str
) -> _FrozenFileSnapshot:
    observed = _snapshot_file(expected.path, description=description)
    if observed.identity != expected.identity:
        raise ValueError(f"{description} snapshot changed since preflight: {expected.path}")
    return observed


def _code_artifacts() -> dict[str, dict[str, object]]:
    package = Path(__file__).parent
    files = {
        "producer_code": Path(__file__),
        "four_view_geometry_authority_code": package / "recipient_multiview_teacher_export.py",
        "geometry_helper_code": package / "geometry.py",
        "ocr_helper_code": package / "ocr.py",
        "pseudolabel_helper_code": package / "ocr_pseudolabels.py",
        "unified_dataset_helper_code": package / "ocr_unified_dataset.py",
        "pipeline_crop_helper_code": package / "pipeline.py",
        "status_crop_helper_code": package / "status_crops.py",
    }
    return {name: _artifact_binding(path) for name, path in files.items()}


def _prepare(
    *, manifest: Path, dataset_contract: Path, dataset_root: Path
) -> tuple[list[_PreparedRow], dict[str, object]]:
    contract_snapshot, raw_contract = _snapshot_json(
        dataset_contract,
        description="fixed2 source dataset contract",
        hook_prefix="preflight_contract",
    )
    contract = _validated_contract_payload(
        raw_contract,
        description=str(dataset_contract),
    )
    declared_root = contract.get("dataset_root")
    if declared_root is not None:
        if not isinstance(declared_root, str) or not declared_root:
            raise ValueError("fixed2 source dataset contract root is invalid")
        _require_samefile(
            dataset_root,
            Path(declared_root).resolve(),
            description="fixed2 preflight dataset root",
        )
    manifest_snapshot, source_records = _snapshot_records(
        manifest,
        description="fixed2 source blind manifest",
        hook_prefix="preflight_manifest",
    )
    manifest_sha256 = manifest_snapshot.sha256
    contract_sha256 = contract_snapshot.sha256
    preflight_snapshots: dict[Path, _FrozenFileSeal] = {
        manifest_snapshot.path: _snapshot_seal(manifest_snapshot),
        contract_snapshot.path: _snapshot_seal(contract_snapshot),
    }
    split_counts: Counter[str] = Counter()
    recipient_counts: Counter[str] = Counter()
    train_missing = 0
    group_splits: dict[str, str] = {}
    source_splits: dict[str, str] = {}
    crop_splits: dict[str, str] = {}
    crop_targets: dict[str, str] = {}
    ids: set[str] = set()
    train_rows: list[tuple[int, dict[str, object], str, dict[str, object]]] = []
    semantic_source_rows: list[dict[str, object]] = []
    for line_number, record in source_records:
        record_id = record.get("id")
        group_id = record.get("group_id")
        split = record.get("split")
        if not isinstance(record_id, str) or not record_id or record_id in ids:
            raise ValueError(f"{manifest}:{line_number}: invalid or duplicate id")
        ids.add(record_id)
        if not isinstance(group_id, str) or not group_id:
            raise ValueError(f"{manifest}:{line_number}: invalid group_id")
        if split not in ALLOWED_SPLITS:
            raise ValueError(f"{manifest}:{line_number}: unsupported split {split!r}")
        split = str(split)
        split_counts[split] += 1
        if group_splits.setdefault(group_id, split) != split:
            raise ValueError(f"{manifest}:{line_number}: group crosses split boundary")
        raw_source = record.get("source")
        if not isinstance(raw_source, str) or not raw_source:
            raise ValueError(f"{manifest}:{line_number}: invalid source")
        source_key = _path_identity(raw_source)
        if source_splits.setdefault(source_key, split) != split:
            raise ValueError(f"{manifest}:{line_number}: source crosses split boundary")
        slots = record.get("slots")
        recipient = slots.get("recipient_field") if isinstance(slots, Mapping) else None
        crop_hash: str | None = None
        if recipient is not None:
            if not isinstance(recipient, Mapping):
                raise ValueError(f"{manifest}:{line_number}: recipient slot must be an object")
            recipient_counts[split] += 1
            crop_hash = _require_sha256(
                recipient.get("crop_sha256"),
                description=f"{manifest}:{line_number}: recipient crop_sha256",
            )
            if crop_splits.setdefault(crop_hash, split) != split:
                raise ValueError(f"{manifest}:{line_number}: recipient crop crosses split boundary")
        semantic_source_rows.append(
            {
                "id": record_id,
                "group_id": group_id,
                "split": split,
                "recipient_crop_sha256": crop_hash,
            }
        )
        if split != TRAIN_SPLIT:
            continue
        if recipient is None:
            train_missing += 1
            continue
        target, slot = _target_from_train_record(record, source=manifest, line_number=line_number)
        declared = _require_sha256(
            slot.get("crop_sha256"),
            description=f"{manifest}:{line_number}: train recipient crop_sha256",
        )
        if crop_targets.setdefault(declared, target) != target:
            raise ValueError(f"{manifest}:{line_number}: one recipient crop has conflicting train targets")
        train_rows.append((line_number, record, target, slot))
    if not train_rows:
        raise ValueError("source manifest has no train recipient targets")

    # The source semantic closure is global and immutable after the first
    # manifest pass.  Hash it once; recomputing the full ~88k-row payload for
    # every train recipient would make the formal producer quadratic.
    source_semantic_sha = _canonical_sha256(semantic_source_rows)
    owners: dict[str, _GeneratedViewOwner] = {}
    prepared: list[_PreparedRow] = []
    for line_number, record, target, slot in train_rows:
        record_id = str(record["id"])
        group_id = str(record["group_id"])
        source = _absolute_existing_file(record.get("source"), description=f"{manifest}:{line_number}: source")
        result = _absolute_existing_file(record.get("result_json"), description=f"{manifest}:{line_number}: result_json")
        crop = _relative_existing_file(dataset_root, slot.get("image"), description=f"{manifest}:{line_number}: recipient image")
        declared = _require_sha256(slot.get("crop_sha256"), description=f"{manifest}:{line_number}: crop")
        bbox = _bbox(slot.get("bbox_rectified"))
        source_snapshot = _snapshot_file(
            source,
            description=f"preflight source image {record_id}",
        )
        result_snapshot = _snapshot_file(
            result,
            description=f"preflight result JSON {record_id}",
        )
        crop_snapshot = _snapshot_file(
            crop,
            description=f"preflight Paddle crop {record_id}",
        )
        source_sha = source_snapshot.sha256
        result_sha = result_snapshot.sha256
        crop_file_sha = crop_snapshot.sha256
        for snapshot in (source_snapshot, result_snapshot, crop_snapshot):
            seal = _snapshot_seal(snapshot)
            prior = preflight_snapshots.setdefault(snapshot.path, seal)
            if prior.identity != seal.identity:
                raise ValueError(f"preflight artifact changed across records: {snapshot.path}")
        pixels = _selected_views(
            manifest=manifest,
            line_number=line_number,
            record_id=record_id,
            source_snapshot=source_snapshot,
            result_snapshot=result_snapshot,
            crop_snapshot=crop_snapshot,
            declared_crop_sha256=declared,
            bbox=bbox,
            hook_prefix=f"preflight_{record_id}",
        )
        target_sha = hashlib.sha256(target.encode("utf-8")).hexdigest()
        view_specs: list[_PreparedView] = []
        for name in VIEWS:
            view = pixels[name]
            digest = _crop_digest(view)
            declared_split = crop_splits.get(digest)
            if declared_split is not None and declared_split != TRAIN_SPLIT:
                raise ValueError(
                    f"generated fixed2 train view hash {digest} crosses declared {declared_split} crop boundary"
                )
            _register_generated_view_owner(
                owners,
                pixel_sha256=digest,
                owner=_GeneratedViewOwner(
                    line_number=line_number,
                    record_id=record_id,
                    view=name,
                    group_id=group_id,
                    target_sha256=target_sha,
                    shape=tuple(int(size) for size in view.shape),
                ),
            )
            view_specs.append(_PreparedView(name, digest, int(view.shape[1]), int(view.shape[0])))
        closure_payload = {
            "source_record_id": record_id,
            "source_group_id": group_id,
            "source_manifest_semantic_sha256": source_semantic_sha,
            "target_sha256": target_sha,
            "source_sha256": source_sha,
            "result_json_sha256": result_sha,
            "paddle_crop_pixel_sha256": declared,
            "views": [
                {"view": item.name, "pixel_sha256": item.pixel_sha256}
                for item in view_specs
            ],
        }
        prepared.append(
            _PreparedRow(
                line_number, record_id, group_id, target, target_sha,
                source, source_sha, result, result_sha, crop, declared, crop_file_sha,
                bbox, tuple(view_specs), _canonical_sha256(closure_payload),
            )
        )
    prepared.sort(key=lambda item: item.record_id)
    selected_semantic = [
        {
            "source_record_id": item.record_id,
            "group_id": item.group_id,
            "target_sha256": item.target_sha256,
            "paddle_crop_pixel_sha256": item.paddle_crop_pixel_sha256,
            "views": [
                {
                    "view": view.name,
                    "pixel_sha256": view.pixel_sha256,
                    "width": view.width,
                    "height": view.height,
                }
                for view in item.views
            ],
        }
        for item in prepared
    ]
    subject_material = {
        "domain": SUBJECT_DOMAIN,
        "schema_version": SCHEMA_VERSION,
        "source_dataset_kind": contract["kind"],
        "source_manifest_semantic_sha256": source_semantic_sha,
        "target_authority": "existing_paddle_train_manifest_only",
        "view_order": list(VIEWS),
        "view_geometry": {
            "standard_margin_ratio": STANDARD_MARGIN_RATIO,
            "fixed_value_left_trim": FIXED_VALUE_LEFT_TRIM,
        },
        "selected_semantic_bindings": selected_semantic,
        "same_target_different_group_duplicate_policy": "reject",
    }
    evidence: dict[str, object] = {
        "source_contract": contract,
        "manifest_sha256": manifest_sha256,
        "contract_sha256": contract_sha256,
        "source_manifest_semantic_sha256": source_semantic_sha,
        "producer_subject_id": _canonical_sha256(subject_material),
        "subject_material": subject_material,
        "split_counts": split_counts,
        "recipient_counts": recipient_counts,
        "train_missing": train_missing,
        "preflight_snapshots": preflight_snapshots,
    }
    return prepared, evidence


def _normalize_inputs(
    *, manifest: Path, output_root: Path, dataset_root: Path | None, dataset_contract: Path | None
) -> tuple[Path, Path, Path, Path]:
    manifest = manifest.resolve()
    if not manifest.is_file():
        raise FileNotFoundError(manifest)
    contract = (dataset_contract.resolve() if dataset_contract is not None else (manifest.parent / "dataset.contract.json").resolve())
    if not contract.is_file():
        raise FileNotFoundError(contract)
    if dataset_root is not None:
        raw_root: Path | str | object = dataset_root
    else:
        _contract_snapshot, raw_contract = _snapshot_json(
            contract,
            description="fixed2 source dataset contract root lookup",
            hook_prefix="normalize_contract",
        )
        raw_root = _validated_contract_payload(
            raw_contract,
            description=str(contract),
        ).get("dataset_root")
    if isinstance(raw_root, Path):
        root = raw_root.resolve()
    elif isinstance(raw_root, str) and raw_root:
        root = Path(raw_root).resolve()
    else:
        raise ValueError("dataset_root is required")
    if not root.is_dir():
        raise NotADirectoryError(root)
    raw_output = Path(os.path.abspath(os.fspath(output_root)))
    _assert_no_reparse_components(raw_output, description="recipient fixed2 teacher output")
    if os.path.lexists(raw_output):
        raise FileExistsError(f"refusing to overwrite recipient fixed2 teacher export: {raw_output}")
    if not raw_output.parent.is_dir():
        raise NotADirectoryError(raw_output.parent)
    _assert_no_reparse_components(raw_output.parent, description="recipient fixed2 teacher output parent")
    for protected, description in (
        (manifest.parent, "manifest directory"),
        (contract.parent, "contract directory"),
        (root, "Paddle dataset root"),
    ):
        _assert_output_separate(raw_output, protected, description=description)
    return manifest, contract, root, raw_output


def _publication_identity(
    *, root_identity: DirectoryIdentity, images_identity: DirectoryIdentity,
    manifest_identity: FileIdentity, image_identities: Mapping[str, FileIdentity]
) -> dict[str, object]:
    image_closure = [
        {"name": name, **_identity_payload(identity)}
        for name, identity in sorted(image_identities.items())
    ]
    return {
        "scheme": "native_directory_file_identity_and_image_closure_v1",
        "root_directory": _directory_identity_payload(root_identity),
        "images_directory": _directory_identity_payload(images_identity),
        "manifest": _identity_payload(manifest_identity),
        "image_count": len(image_closure),
        "image_file_identity_closure_sha256": _canonical_sha256(image_closure),
    }


def _publication_hook(
    checkpoint: str, *, parent: Path, stage: Path, output_root: Path
) -> None:
    """No-op deterministic race hook used by publication boundary tests."""


def _materialize_impl(
    *, formal: bool, manifest: Path, output_root: Path,
    dataset_root: Path | None = None, dataset_contract: Path | None = None
) -> dict[str, object]:
    if formal and not _running_on_windows():
        raise OSError(errno.ENOTSUP, "formal fixed2 teacher publication requires Windows", os.fspath(output_root))
    if not formal and _running_on_windows():
        raise OSError(
            errno.ENOTSUP,
            "analysis fixed2 teacher publication is disabled on Windows",
            os.fspath(output_root),
        )
    kind = KIND if formal else ANALYSIS_KIND
    record_kind = RECORD_KIND if formal else ANALYSIS_RECORD_KIND
    authority = PUBLICATION_AUTHORITY if formal else ANALYSIS_PUBLICATION_AUTHORITY
    contract_marker_name = CONTRACT_NAME if formal else ANALYSIS_CONTRACT_NAME
    manifest, contract, root, output = _normalize_inputs(
        manifest=manifest, output_root=output_root, dataset_root=dataset_root, dataset_contract=dataset_contract
    )
    prepared, evidence = _prepare(manifest=manifest, dataset_contract=contract, dataset_root=root)
    preflight_snapshots = evidence["preflight_snapshots"]
    assert isinstance(preflight_snapshots, Mapping)
    # The second manifest/contract snapshot is taken before staging and must
    # match the exact preflight bytes that supplied every semantic decision.
    _resnapshot_matches(
        preflight_snapshots[manifest.resolve()],  # type: ignore[index]
        description="publication source blind manifest",
    )
    _resnapshot_matches(
        preflight_snapshots[contract.resolve()],  # type: ignore[index]
        description="publication source dataset contract",
    )
    # The complete selected-view collision closure has passed before this line.
    parent = output.parent
    parent_lease = _open_parent(parent)
    stage = parent / f".{output.name}.{hashlib.sha256(os.urandom(32)).hexdigest()[:24]}.tmp"
    stage_lease: _DirectoryLease | None = None
    images_lease: _DirectoryLease | None = None
    renamed = False

    def checkpoint(name: str) -> None:
        _publication_hook(
            name,
            parent=parent,
            stage=stage,
            output_root=output,
        )
        _require_lease(parent_lease)
        if not _same_directory_identity(parent, parent_lease.identity):
            raise ValueError(
                f"fixed2 teacher output parent identity changed at {name}: {parent}"
            )

    def require_inputs_unchanged(checkpoint_name: str) -> None:
        for raw_expected in preflight_snapshots.values():
            if not isinstance(raw_expected, _FrozenFileSeal):
                raise TypeError("fixed2 preflight snapshot closure changed")
            _resnapshot_matches(
                raw_expected,
                description=f"{checkpoint_name} bound input",
            )

    def require_stage_unchanged(
        *,
        expected_manifest_identity: FileIdentity,
        expected_image_identities: Mapping[str, FileIdentity],
        expected_contract_identity: FileIdentity | None = None,
    ) -> None:
        """Close the staged byte set once, immediately before publication."""

        if stage_lease is None or images_lease is None:
            raise ValueError("fixed2 teacher stage leases are unavailable")
        if _anchored_names(images_lease) != set(expected_image_identities):
            raise ValueError("fixed2 producer image file set changed before publication")
        expected_stage_names = {"images", MANIFEST_NAME}
        if expected_contract_identity is not None:
            expected_stage_names.add(contract_marker_name)
        if _anchored_names(stage_lease) != expected_stage_names:
            raise ValueError("fixed2 producer stage entry set changed before publication")
        observed_manifest = _snapshot_file(
            stage_lease.path / MANIFEST_NAME,
            description="fixed2 staged manifest closing snapshot",
        )
        if observed_manifest.identity != expected_manifest_identity:
            raise ValueError("fixed2 staged manifest changed before publication")
        if expected_contract_identity is not None:
            observed_contract = _snapshot_file(
                stage_lease.path / contract_marker_name,
                description="fixed2 staged contract closing snapshot",
            )
            if observed_contract.identity != expected_contract_identity:
                raise ValueError("fixed2 staged contract changed before publication")
        for name, expected_identity in expected_image_identities.items():
            observed = _snapshot_file(
                images_lease.path / name,
                description=f"fixed2 staged image closing snapshot {name}",
            )
            if observed.identity != expected_identity:
                raise ValueError(f"fixed2 staged image changed before publication: {name}")

    def quarantine_published_output() -> Path:
        """Move a failed nominal publication back under a fresh hidden name."""

        nonlocal renamed
        if stage_lease is None or not renamed:
            return stage_lease.path if stage_lease is not None else stage
        last_collision: BaseException | None = None
        for _attempt in range(8):
            quarantine = parent / (
                f".{output.name}.{hashlib.sha256(os.urandom(32)).hexdigest()[:24]}.failed"
            )
            try:
                _rename_no_replace(parent_lease, stage_lease, quarantine)
            except FileExistsError as error:
                last_collision = error
                continue
            stage_lease.path = quarantine
            if images_lease is not None:
                images_lease.path = quarantine / "images"
            renamed = False
            if output.exists():
                raise ValueError("failed fixed2 nominal output remained after quarantine")
            return quarantine
        raise RuntimeError("unable to reserve fixed2 quarantine name") from last_collision

    try:
        retained_prefix = f".{output.name}."
        if any(name.startswith(retained_prefix) for name in _anchored_names(parent_lease)):
            raise FileExistsError("refusing publication while fixed2 teacher failure evidence exists")
        checkpoint("before_stage_creation")
        stage_lease = _create_child(parent_lease, name=stage.name, rename_capable=True)
        checkpoint("after_stage_creation")
        images_lease = _create_child(stage_lease, name="images", rename_capable=False)
        checkpoint("after_images_directory_creation")
        image_identities: dict[str, FileIdentity] = {}
        rows: list[dict[str, object]] = []
        for item in prepared:
            source_snapshot = _resnapshot_matches(
                preflight_snapshots[item.source.resolve()],  # type: ignore[index]
                description=f"publication source image {item.record_id}",
            )
            result_snapshot = _resnapshot_matches(
                preflight_snapshots[item.result_json.resolve()],  # type: ignore[index]
                description=f"publication result JSON {item.record_id}",
            )
            crop_snapshot = _resnapshot_matches(
                preflight_snapshots[item.paddle_crop.resolve()],  # type: ignore[index]
                description=f"publication Paddle crop {item.record_id}",
            )
            pixels = _selected_views(
                manifest=manifest,
                line_number=item.line_number,
                record_id=item.record_id,
                source_snapshot=source_snapshot,
                result_snapshot=result_snapshot,
                crop_snapshot=crop_snapshot,
                declared_crop_sha256=item.paddle_crop_pixel_sha256,
                bbox=item.bbox,
                hook_prefix=f"publication_{item.record_id}",
            )
            record_key = hashlib.sha256(item.record_id.encode("utf-8")).hexdigest()[:24]
            expected = {view.name: view for view in item.views}
            for view_name in VIEWS:
                checkpoint(f"before_image_write:{item.record_id}:{view_name}")
                view = pixels[view_name]
                spec = expected[view_name]
                if (
                    _crop_digest(view) != spec.pixel_sha256
                    or int(view.shape[1]) != spec.width
                    or int(view.shape[0]) != spec.height
                ):
                    raise ValueError(f"fixed2 selected view changed between closure and publication: {item.record_id}/{view_name}")
                name = f"{record_key}-{view_name.replace('_', '-')}.png"
                identity = _write_file(images_lease, name=name, payload=_png_bytes(view))
                image_identities[name] = identity
                png_snapshot = _snapshot_file(
                    images_lease.path / name,
                    description=f"staged fixed2 PNG {item.record_id}/{view_name}",
                )
                if png_snapshot.identity != identity:
                    raise ValueError(f"staged fixed2 PNG identity changed: {name}")
                decoded = _snapshot_rgb(
                    png_snapshot,
                    description=f"staged fixed2 PNG {item.record_id}/{view_name}",
                    hook_prefix=f"staged_png_{item.record_id}_{view_name}",
                )
                if _crop_digest(decoded) != spec.pixel_sha256:
                    raise ValueError(f"published fixed2 PNG decoded pixels changed: {name}")
                rows.append(
                    {
                        "schema_version": SCHEMA_VERSION,
                        "kind": record_kind,
                        "id": f"recipient-{record_key}-{view_name.replace('_', '-')}",
                        "group_id": item.group_id,
                        "source_record_id": item.record_id,
                        "split": TRAIN_SPLIT,
                        "field": "recipient_field",
                        "view": view_name,
                        "image": f"images/{name}",
                        "text": item.target,
                        "target_sha256": item.target_sha256,
                        "target_source": "slots.recipient_field.text",
                        "target_source_manifest_sha256": evidence["manifest_sha256"],
                        "optimizer_supervision_split_eligible": True,
                        "optimizer_consumable": False,
                        "group_closure_sha256": item.group_closure_sha256,
                        "group_view_count": len(VIEWS),
                        "source": str(item.source),
                        "source_sha256": item.source_sha256,
                        "result_json": str(item.result_json),
                        "result_json_sha256": item.result_json_sha256,
                        "bbox_rectified": [round(float(value), 4) for value in item.bbox],
                        "paddle_crop": str(item.paddle_crop),
                        "paddle_crop_pixel_sha256": item.paddle_crop_pixel_sha256,
                        "paddle_crop_file_sha256": item.paddle_crop_file_sha256,
                        "view_width": spec.width,
                        "view_height": spec.height,
                        "view_pixel_sha256": spec.pixel_sha256,
                        "view_file_sha256": identity[-1],
                    }
                )
        rows_by_source: dict[str, list[dict[str, object]]] = {}
        prepared_by_source = {item.record_id: item for item in prepared}
        for row in rows:
            rows_by_source.setdefault(str(row["source_record_id"]), []).append(row)
        for source_id, source_rows in rows_by_source.items():
            item = prepared_by_source[source_id]
            ordered_rows = sorted(
                source_rows,
                key=lambda row: VIEWS.index(str(row["view"])),
            )
            closure_payload = {
                "source_record_id": source_id,
                "source_group_id": item.group_id,
                "source_manifest_sha256": evidence["manifest_sha256"],
                "target_sha256": item.target_sha256,
                "source_sha256": item.source_sha256,
                "result_json_sha256": item.result_json_sha256,
                "paddle_crop_pixel_sha256": item.paddle_crop_pixel_sha256,
                "views": [
                    {
                        "view": row["view"],
                        "pixel_sha256": row["view_pixel_sha256"],
                        "file_sha256": row["view_file_sha256"],
                    }
                    for row in ordered_rows
                ],
            }
            closure = _canonical_sha256(closure_payload)
            for row in source_rows:
                row["group_closure_sha256"] = closure
        rows.sort(key=lambda row: (str(row["source_record_id"]), VIEWS.index(str(row["view"]))))
        checkpoint("before_manifest_write")
        manifest_identity = _write_file(stage_lease, name=MANIFEST_NAME, payload=_jsonl_bytes(rows))
        if _anchored_names(images_lease) != set(image_identities):
            raise ValueError("fixed2 producer image file set changed before publication")
        if _anchored_names(stage_lease) != {"images", MANIFEST_NAME}:
            raise ValueError("fixed2 producer stage entry set changed before publication")
        publication = _publication_identity(
            root_identity=stage_lease.identity,
            images_identity=images_lease.identity,
            manifest_identity=manifest_identity,
            image_identities=image_identities,
        )
        split_counts = evidence["split_counts"]
        recipient_counts = evidence["recipient_counts"]
        assert isinstance(split_counts, Counter) and isinstance(recipient_counts, Counter)
        view_counts = Counter(str(row["view"]) for row in rows)
        artifacts = {
            "source_manifest": _artifact_binding_from_snapshot(
                preflight_snapshots[manifest.resolve()]  # type: ignore[index]
            ),
            "source_dataset_contract": _artifact_binding_from_snapshot(
                preflight_snapshots[contract.resolve()]  # type: ignore[index]
            ),
            **_code_artifacts(),
        }
        payload: dict[str, object] = {
            "schema_version": SCHEMA_VERSION,
            "kind": kind,
            "record_kind": record_kind,
            "publication_profile": (
                "formal_windows_canonical_v1"
                if formal
                else "posix_analysis_fixture_v1"
            ),
            "formal_windows_publication": formal,
            "analysis_fixture": not formal,
            "publication_authority": authority,
            "hard_attestation_scheme": HARD_ATTESTATION_SCHEME if formal else None,
            "public_verification_requires_hard_attestation": formal,
            "publication_identity": publication if formal else None,
            "nominal_output_root": str(output),
            "analysis_only": True,
            "production_route_authorized": False,
            "source_manifest": str(manifest),
            "source_manifest_sha256": evidence["manifest_sha256"],
            "source_manifest_semantic_sha256": evidence["source_manifest_semantic_sha256"],
            "source_dataset_contract": str(contract),
            "source_dataset_contract_sha256": evidence["contract_sha256"],
            "source_dataset_kind": evidence["source_contract"]["kind"],  # type: ignore[index]
            "source_dataset_root": str(root),
            "target_source": "slots.recipient_field.text",
            "target_label_authority": "existing_paddle_train_manifest_only",
            "target_recomputed": False,
            "optimizer_supervision_splits": [TRAIN_SPLIT],
            "optimizer_input_ready": False,
            "records_role": "recipient_fixed2_overlay_source_only",
            "optimizer_adapter_required": ADAPTER_MARKER,
            "held_out_splits_excluded": sorted(HELD_OUT_SPLITS),
            "held_out_target_values_used": False,
            "held_out_target_values_validated": False,
            "held_out_target_values_emitted": False,
            "source_manifest_split_counts": {
                split: int(split_counts[split]) for split in ("train", "val", "test", "formal")
            },
            "source_split_counts": {
                split: int(recipient_counts[split]) for split in ("train", "val", "test", "formal")
            },
            "source_train_recipient_records": len(prepared),
            "source_train_records_without_recipient_target": int(evidence["train_missing"]),
            "output_records": len(rows),
            "output_split_counts": {TRAIN_SPLIT: len(rows)},
            "view_order": list(VIEWS),
            "view_counts": {view: int(view_counts[view]) for view in VIEWS},
            "view_geometry": {
                "standard": {"margin_ratio": STANDARD_MARGIN_RATIO, "arithmetic": "csharp_ieee754_float32"},
                "fixed_value": {
                    "base": "production_standard",
                    "left_trim_fraction": FIXED_VALUE_LEFT_TRIM,
                    "rounding": "bankers_round_midpoint_to_even",
                },
            },
            "selected_view_hash_closure": {
                "views_per_train_record": len(VIEWS),
                "decoded_pixels_reverified": True,
                "blind_owner_fields": ["split", "group_id", "target_sha256"],
                "cross_split_conflicts": 0,
                "cross_target_conflicts": 0,
                "cross_group_conflicts": 0,
                "same_target_different_group_duplicate_policy": "reject",
            },
            "producer_subject_id": evidence["producer_subject_id"],
            "subject_domain": SUBJECT_DOMAIN,
            "subject_path_stable": True,
            "subject_output_stable": True,
            "subject_code_stable": True,
            "train_manifest": MANIFEST_NAME,
            "train_manifest_sha256": manifest_identity[-1],
            "artifacts": artifacts,
            "publication": "private_stage_contract_verified_then_anchored_no_replace_directory_v1",
            "commit_marker": contract_marker_name,
            "publication_complete": True,
            "failure_policy": "preflight_conflicts_create_no_stage; publication_failures_are_quarantined",
        }
        sealed = {**payload, "integrity_sha256": _canonical_sha256(payload)}
        checkpoint("before_prepublication_verify")
        _verify_payload(
            sealed,
            export_root=stage_lease.path,
            expected_kind=kind,
            expected_record_kind=record_kind,
            expected_authority=authority,
            expected_contract_marker=contract_marker_name,
            require_publication_identity=formal,
            require_hard_attestation=False,
            declared_nominal_output_root=output,
            actual_root_identity=stage_lease.identity,
            actual_images_identity=images_lease.identity,
        )
        checkpoint("after_prepublication_verify")
        checkpoint("before_contract_commit")
        require_stage_unchanged(
            expected_manifest_identity=manifest_identity,
            expected_image_identities=image_identities,
        )
        require_inputs_unchanged("before-contract-commit")
        contract_identity = _write_file(
            stage_lease,
            name=contract_marker_name,
            payload=_json_bytes(sealed),
        )
        if _anchored_names(stage_lease) != {
            "images",
            MANIFEST_NAME,
            contract_marker_name,
        }:
            raise ValueError("fixed2 producer private commit entry set changed")
        checkpoint("after_contract_commit")
        private_contract_snapshot, private_contract = _snapshot_json(
            stage_lease.path / contract_marker_name,
            description="fixed2 private staged contract marker",
            hook_prefix="private_contract_marker",
        )
        if (
            private_contract != sealed
            or private_contract_snapshot.identity != contract_identity
        ):
            raise ValueError("fixed2 private staged contract marker bytes changed")
        _verify_payload(
            private_contract,
            export_root=stage_lease.path,
            expected_kind=kind,
            expected_record_kind=record_kind,
            expected_authority=authority,
            expected_contract_marker=contract_marker_name,
            require_publication_identity=formal,
            require_hard_attestation=False,
            contract_snapshot=private_contract_snapshot,
            declared_nominal_output_root=output,
            actual_root_identity=stage_lease.identity,
            actual_images_identity=images_lease.identity,
        )
        checkpoint("after_private_contract_verify")
        checkpoint("immediately_before_rename")
        require_stage_unchanged(
            expected_manifest_identity=manifest_identity,
            expected_image_identities=image_identities,
            expected_contract_identity=contract_identity,
        )
        require_inputs_unchanged("immediately-before-rename")
        _rename_no_replace(parent_lease, stage_lease, output)
        renamed = True
        stage_lease.path = output
        images_lease.path = output / "images"
        _require_lease(parent_lease)
        _require_lease(stage_lease)
        _require_lease(images_lease)
        checkpoint("immediately_after_rename")
        committed_snapshot, committed_contract = _snapshot_json(
            output / contract_marker_name,
            description="fixed2 committed contract marker",
            hook_prefix="publication_contract_marker",
        )
        if committed_contract != sealed:
            raise ValueError("fixed2 committed contract marker bytes changed")
        verified = _verify_payload(
            committed_contract,
            export_root=output,
            expected_kind=kind,
            expected_record_kind=record_kind,
            expected_authority=authority,
            expected_contract_marker=contract_marker_name,
            require_publication_identity=formal,
            require_hard_attestation=False,
            contract_snapshot=committed_snapshot,
            actual_root_identity=stage_lease.identity,
            actual_images_identity=images_lease.identity,
        )
        return verified
    except BaseException as error:
        if stage_lease is not None:
            quarantine_error: BaseException | None = None
            try:
                retained = quarantine_published_output()
            except BaseException as caught:
                quarantine_error = caught
                retained = output if renamed else stage_lease.path
            note = (
                f"fixed2 teacher publication retained failure evidence at {retained}; "
                f"{contract_marker_name} is absent, invalid, or requires independent verification; "
                "no files or directories were deleted"
            )
            if quarantine_error is not None:
                note += f"; nominal-output quarantine failed: {quarantine_error}"
            setattr(error, "fixed2_teacher_quarantine", note)
            if hasattr(error, "add_note"):
                error.add_note(note)
        raise
    finally:
        if images_lease is not None:
            images_lease.close()
        if stage_lease is not None:
            stage_lease.close()
        parent_lease.close()


def _artifact_snapshot_verify(
    binding: object, *, name: str
) -> _FrozenFileSnapshot:
    if not isinstance(binding, Mapping):
        raise ValueError(f"fixed2 teacher artifact {name} must be an object")
    raw_path = Path(str(binding.get("path")))
    if not raw_path.is_absolute():
        raise ValueError(f"fixed2 teacher artifact {name} path must be absolute")
    _assert_no_reparse_components(raw_path, description=f"fixed2 teacher artifact {name}")
    snapshot = _snapshot_file(
        raw_path,
        description=f"fixed2 teacher artifact {name}",
    )
    if (
        binding.get("sha256") != snapshot.sha256
        or binding.get("size_bytes") != snapshot.size_bytes
    ):
        raise ValueError(f"fixed2 teacher artifact {name} binding changed")
    return snapshot


def _require_samefile(left: Path, right: Path, *, description: str) -> None:
    try:
        same = os.path.samefile(left, right)
    except OSError as error:
        raise ValueError(f"unable to compare {description}") from error
    if not same:
        raise ValueError(f"{description} is not the bound file")


def _require_formal_hard_attestation(
    *,
    contract_snapshot: _FrozenFileSnapshot,
    contract: Mapping[str, Any],
) -> None:
    """Require the second-stage reviewed raw marker and semantic subject pins."""

    from . import recipient_fixed2_teacher_attestation as attestation

    pinned_sha = attestation.ATTESTED_FIXED2_CONTRACT_SHA256
    pinned_size = attestation.ATTESTED_FIXED2_CONTRACT_SIZE_BYTES
    pinned_subject = attestation.ATTESTED_FIXED2_PRODUCER_SUBJECT_ID
    if (
        not isinstance(pinned_sha, str)
        or len(pinned_sha) != 64
        or any(character not in "0123456789abcdef" for character in pinned_sha)
        or isinstance(pinned_size, bool)
        or not isinstance(pinned_size, int)
        or pinned_size <= 0
        or not isinstance(pinned_subject, str)
        or len(pinned_subject) != 64
        or any(character not in "0123456789abcdef" for character in pinned_subject)
    ):
        raise ValueError(
            "formal fixed2 teacher publication is not second-stage hard-attested"
        )
    if contract_snapshot.sha256 != pinned_sha:
        raise ValueError("formal fixed2 teacher contract SHA does not match hard attestation")
    if contract_snapshot.size_bytes != pinned_size:
        raise ValueError("formal fixed2 teacher contract size does not match hard attestation")
    if contract.get("producer_subject_id") != pinned_subject:
        raise ValueError("formal fixed2 teacher subject does not match hard attestation")


def _verify_payload(
    contract: Mapping[str, Any], *, export_root: Path, expected_kind: str,
    expected_record_kind: str, expected_authority: str,
    expected_contract_marker: str, require_publication_identity: bool,
    require_hard_attestation: bool = False,
    contract_snapshot: _FrozenFileSnapshot | None = None,
    declared_nominal_output_root: Path | None = None,
    actual_root_identity: DirectoryIdentity | None = None,
    actual_images_identity: DirectoryIdentity | None = None,
) -> dict[str, object]:
    formal_profile = (
        expected_kind,
        expected_record_kind,
        expected_authority,
        expected_contract_marker,
        require_publication_identity,
    ) == (
        KIND,
        RECORD_KIND,
        PUBLICATION_AUTHORITY,
        CONTRACT_NAME,
        True,
    )
    analysis_profile = (
        expected_kind,
        expected_record_kind,
        expected_authority,
        expected_contract_marker,
        require_publication_identity,
    ) == (
        ANALYSIS_KIND,
        ANALYSIS_RECORD_KIND,
        ANALYSIS_PUBLICATION_AUTHORITY,
        ANALYSIS_CONTRACT_NAME,
        False,
    )
    if not formal_profile and not analysis_profile:
        raise ValueError("unsupported fixed2 teacher publication profile")
    if require_hard_attestation and not formal_profile:
        raise ValueError("hard attestation is only valid for the formal fixed2 profile")
    if formal_profile and not _running_on_windows():
        raise OSError(
            errno.ENOTSUP,
            "formal fixed2 teacher payload verification requires Windows",
            os.fspath(export_root),
        )
    if analysis_profile and _running_on_windows():
        raise OSError(
            errno.ENOTSUP,
            "analysis fixed2 teacher payload verification is disabled on Windows",
            os.fspath(export_root),
        )
    if set(contract) != CONTRACT_KEYS:
        raise ValueError("fixed2 teacher contract key set changed")
    if contract.get("integrity_sha256") != _canonical_sha256(
        {key: value for key, value in contract.items() if key != "integrity_sha256"}
    ):
        raise ValueError("fixed2 teacher contract integrity changed")
    if require_hard_attestation:
        if contract_snapshot is None:
            raise ValueError("formal hard attestation requires the frozen contract marker")
        _require_formal_hard_attestation(
            contract_snapshot=contract_snapshot,
            contract=contract,
        )
    for key, expected in (
        ("schema_version", SCHEMA_VERSION), ("kind", expected_kind),
        ("record_kind", expected_record_kind), ("publication_authority", expected_authority),
        (
            "hard_attestation_scheme",
            HARD_ATTESTATION_SCHEME if formal_profile else None,
        ),
        ("public_verification_requires_hard_attestation", formal_profile),
        (
            "publication_profile",
            "formal_windows_canonical_v1"
            if formal_profile
            else "posix_analysis_fixture_v1",
        ),
        ("formal_windows_publication", formal_profile),
        ("analysis_fixture", analysis_profile),
        ("optimizer_input_ready", False), ("view_order", list(VIEWS)),
        ("train_manifest", MANIFEST_NAME), ("commit_marker", expected_contract_marker),
        ("publication_complete", True), ("production_route_authorized", False),
        ("analysis_only", True),
        ("subject_domain", SUBJECT_DOMAIN),
        ("subject_path_stable", True),
        ("subject_output_stable", True),
        ("subject_code_stable", True),
        (
            "publication",
            "private_stage_contract_verified_then_anchored_no_replace_directory_v1",
        ),
        (
            "failure_policy",
            "preflight_conflicts_create_no_stage; publication_failures_are_quarantined",
        ),
        ("producer_subject_id", contract.get("producer_subject_id")),
    ):
        if type(contract.get(key)) is not type(expected) or contract.get(key) != expected:
            raise ValueError(f"fixed2 teacher contract {key} mismatch")
    subject_id = contract.get("producer_subject_id")
    if not isinstance(subject_id, str) or len(subject_id) != 64 or any(ch not in "0123456789abcdef" for ch in subject_id):
        raise ValueError("fixed2 teacher producer subject id is invalid")
    expected_view_geometry = {
        "standard": {
            "margin_ratio": STANDARD_MARGIN_RATIO,
            "arithmetic": "csharp_ieee754_float32",
        },
        "fixed_value": {
            "base": "production_standard",
            "left_trim_fraction": FIXED_VALUE_LEFT_TRIM,
            "rounding": "bankers_round_midpoint_to_even",
        },
    }
    if not _exact_json_value(contract.get("view_geometry"), expected_view_geometry):
        raise ValueError("fixed2 teacher contract view_geometry changed")
    root = export_root.resolve()
    _assert_no_reparse_components(root, description="fixed2 teacher export root")
    raw_nominal_root = contract.get("nominal_output_root")
    if not isinstance(raw_nominal_root, str) or not raw_nominal_root:
        raise ValueError("fixed2 teacher nominal output root is invalid")
    nominal_root = Path(raw_nominal_root)
    normalized_nominal_root = Path(os.path.abspath(os.fspath(nominal_root)))
    if (
        not nominal_root.is_absolute()
        or raw_nominal_root != os.fspath(normalized_nominal_root)
    ):
        raise ValueError("fixed2 teacher nominal output root is not canonical absolute")
    if declared_nominal_output_root is not None:
        private_nominal_root = Path(
            os.path.abspath(os.fspath(declared_nominal_output_root))
        )
        if nominal_root != private_nominal_root:
            raise ValueError("fixed2 teacher private nominal output binding changed")
    else:
        _assert_no_reparse_components(
            nominal_root,
            description="fixed2 teacher declared nominal output root",
        )
        _require_samefile(
            root,
            nominal_root,
            description="fixed2 teacher nominal output root",
        )
        expected_root_identity = actual_root_identity or _directory_identity(root)
        if _directory_identity(nominal_root) != expected_root_identity:
            raise ValueError("fixed2 teacher nominal output identity changed")
    if contract_snapshot is not None:
        marker_path = root / expected_contract_marker
        _require_samefile(
            contract_snapshot.path,
            marker_path,
            description="fixed2 teacher commit marker snapshot",
        )
        snapshotted_contract = _strict_json_bytes(
            contract_snapshot.data,
            description=f"fixed2 teacher commit marker {marker_path}",
        )
        if snapshotted_contract != dict(contract):
            raise ValueError("fixed2 teacher commit marker snapshot changed")
    images = root / "images"
    manifest_path = root / MANIFEST_NAME
    if not images.is_dir() or not manifest_path.is_file():
        raise ValueError("fixed2 teacher export is incomplete")
    output_manifest_snapshot, output_manifest_records = _snapshot_records(
        manifest_path,
        description="fixed2 teacher output manifest",
        hook_prefix="verify_output_manifest",
    )
    if contract.get("train_manifest_sha256") != output_manifest_snapshot.sha256:
        raise ValueError("fixed2 teacher manifest binding changed")
    artifacts = contract.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise ValueError("fixed2 teacher artifacts must be an object")
    required_artifacts = {
        "source_manifest", "source_dataset_contract", "producer_code",
        "four_view_geometry_authority_code", "geometry_helper_code", "ocr_helper_code",
        "pseudolabel_helper_code", "unified_dataset_helper_code", "pipeline_crop_helper_code",
        "status_crop_helper_code",
    }
    if set(artifacts) != required_artifacts:
        raise ValueError("fixed2 teacher artifact closure changed")
    artifact_snapshots = {
        name: _artifact_snapshot_verify(binding, name=name)
        for name, binding in artifacts.items()
    }
    paths = {
        name: snapshot.path
        for name, snapshot in artifact_snapshots.items()
    }
    declared_manifest = _absolute_existing_file(
        contract.get("source_manifest"), description="fixed2 teacher declared source manifest"
    )
    declared_source_contract = _absolute_existing_file(
        contract.get("source_dataset_contract"),
        description="fixed2 teacher declared source dataset contract",
    )
    _require_samefile(
        paths["source_manifest"], declared_manifest,
        description="fixed2 teacher source manifest artifact",
    )
    _require_samefile(
        paths["source_dataset_contract"], declared_source_contract,
        description="fixed2 teacher source dataset contract artifact",
    )
    expected_code = _code_artifacts()
    for name in required_artifacts - {"source_manifest", "source_dataset_contract"}:
        expected_path = Path(str(expected_code[name]["path"]))
        _require_samefile(
            paths[name], expected_path,
            description=f"fixed2 teacher {name}",
        )
    if (
        contract.get("source_manifest_sha256")
        != artifact_snapshots["source_manifest"].sha256
    ):
        raise ValueError("fixed2 teacher source manifest changed")
    if (
        contract.get("source_dataset_contract_sha256")
        != artifact_snapshots["source_dataset_contract"].sha256
    ):
        raise ValueError("fixed2 teacher source dataset contract changed")
    _snapshot_use_hook(
        "verify_source_contract_before_parse",
        snapshot=artifact_snapshots["source_dataset_contract"],
        description="fixed2 source dataset contract",
    )
    source_contract = _validated_contract_payload(
        _strict_json_bytes(
            artifact_snapshots["source_dataset_contract"].data,
            description="fixed2 source dataset contract",
        ),
        description=str(paths["source_dataset_contract"]),
    )
    _snapshot_use_hook(
        "verify_source_contract_after_parse",
        snapshot=artifact_snapshots["source_dataset_contract"],
        description="fixed2 source dataset contract",
    )
    if contract.get("source_dataset_kind") != source_contract.get("kind"):
        raise ValueError("fixed2 teacher source dataset kind changed")
    raw_dataset_root = contract.get("source_dataset_root")
    if not isinstance(raw_dataset_root, str) or not raw_dataset_root:
        raise ValueError("fixed2 teacher source dataset root is invalid")
    source_dataset_root = Path(raw_dataset_root).resolve()
    _assert_no_reparse_components(
        source_dataset_root, description="fixed2 teacher source dataset root"
    )
    if not source_dataset_root.is_dir():
        raise NotADirectoryError(source_dataset_root)
    declared_contract_root = source_contract.get("dataset_root")
    if declared_contract_root is not None:
        if not isinstance(declared_contract_root, str) or not declared_contract_root:
            raise ValueError("fixed2 teacher dataset contract root is invalid")
        _require_samefile(
            source_dataset_root,
            Path(declared_contract_root).resolve(),
            description="fixed2 teacher source dataset root",
        )
    _snapshot_use_hook(
        "verify_source_manifest_before_parse",
        snapshot=artifact_snapshots["source_manifest"],
        description="fixed2 source blind manifest",
    )
    source_rows = _strict_jsonl_records_bytes(
        artifact_snapshots["source_manifest"].data,
        description=str(paths["source_manifest"]),
    )
    _snapshot_use_hook(
        "verify_source_manifest_after_parse",
        snapshot=artifact_snapshots["source_manifest"],
        description="fixed2 source blind manifest",
    )
    owners_by_id: dict[str, dict[str, object]] = {}
    crop_splits: dict[str, str] = {}
    semantic_source: list[dict[str, object]] = []
    source_split_counts: Counter[str] = Counter()
    source_recipient_counts: Counter[str] = Counter()
    train_missing_recipient = 0
    source_ids: set[str] = set()
    source_group_splits: dict[str, str] = {}
    source_path_splits: dict[str, str] = {}
    for line_number, row in source_rows:
        record_id, group_id, split = row.get("id"), row.get("group_id"), row.get("split")
        if not isinstance(record_id, str) or not isinstance(group_id, str) or split not in ALLOWED_SPLITS:
            raise ValueError(f"fixed2 teacher source row {line_number} owner changed")
        if not record_id or not group_id or record_id in source_ids:
            raise ValueError(f"fixed2 teacher source row {line_number} id/group changed")
        source_ids.add(record_id)
        if source_group_splits.setdefault(group_id, str(split)) != split:
            raise ValueError("fixed2 teacher source group crosses split boundary")
        raw_source = row.get("source")
        if not isinstance(raw_source, str) or not raw_source:
            raise ValueError(f"fixed2 teacher source row {line_number} source changed")
        source_key = _path_identity(raw_source)
        if source_path_splits.setdefault(source_key, str(split)) != split:
            raise ValueError("fixed2 teacher source path crosses split boundary")
        slots = row.get("slots")
        recipient = slots.get("recipient_field") if isinstance(slots, Mapping) else None
        if recipient is not None and not isinstance(recipient, Mapping):
            raise ValueError(f"fixed2 teacher source row {line_number} recipient changed")
        source_split_counts[str(split)] += 1
        if recipient is None and split == TRAIN_SPLIT:
            train_missing_recipient += 1
        crop_hash = None
        if isinstance(recipient, Mapping):
            source_recipient_counts[str(split)] += 1
            crop_hash = _require_sha256(recipient.get("crop_sha256"), description="fixed2 teacher blind crop")
            prior_split = crop_splits.setdefault(crop_hash, str(split))
            if prior_split != split:
                raise ValueError("fixed2 teacher source crop crosses split boundary")
        semantic_source.append(
            {
                "id": record_id, "group_id": group_id, "split": str(split),
                "recipient_crop_sha256": crop_hash,
            }
        )
        if split == TRAIN_SPLIT and isinstance(recipient, Mapping):
            target, slot = _target_from_train_record(
                row,
                source=paths["source_manifest"],
                line_number=line_number,
            )
            source_path = _absolute_existing_file(
                row.get("source"),
                description=f"fixed2 teacher source row {record_id}",
            )
            result_path = _absolute_existing_file(
                row.get("result_json"),
                description=f"fixed2 teacher result row {record_id}",
            )
            paddle_crop = _relative_existing_file(
                source_dataset_root,
                slot.get("image"),
                description=f"fixed2 teacher Paddle crop {record_id}",
            )
            bbox = _bbox(slot.get("bbox_rectified"))
            owners_by_id[record_id] = {
                "line_number": line_number,
                "group_id": group_id,
                "target": target,
                "target_sha256": hashlib.sha256(target.encode("utf-8")).hexdigest(),
                "source": source_path,
                "result_json": result_path,
                "paddle_crop": paddle_crop,
                "paddle_crop_pixel_sha256": crop_hash,
                "bbox_rectified": [round(float(value), 4) for value in bbox],
                "bbox": bbox,
            }
    if contract.get("source_manifest_semantic_sha256") != _canonical_sha256(semantic_source):
        raise ValueError("fixed2 teacher source manifest semantic closure changed")
    for key, expected in (
        ("analysis_only", True),
        ("target_source", "slots.recipient_field.text"),
        ("target_label_authority", "existing_paddle_train_manifest_only"),
        ("target_recomputed", False),
        ("optimizer_supervision_splits", [TRAIN_SPLIT]),
        ("records_role", "recipient_fixed2_overlay_source_only"),
        ("optimizer_adapter_required", ADAPTER_MARKER),
        ("held_out_splits_excluded", sorted(HELD_OUT_SPLITS)),
        ("held_out_target_values_used", False),
        ("held_out_target_values_validated", False),
        ("held_out_target_values_emitted", False),
        ("source_train_recipient_records", len(owners_by_id)),
        ("source_train_records_without_recipient_target", train_missing_recipient),
    ):
        if type(contract.get(key)) is not type(expected) or contract.get(key) != expected:
            raise ValueError(f"fixed2 teacher contract {key} changed")
    declared_source_counts = contract.get("source_manifest_split_counts")
    declared_recipient_counts = contract.get("source_split_counts")
    if not isinstance(declared_source_counts, Mapping) or not isinstance(
        declared_recipient_counts, Mapping
    ):
        raise ValueError("fixed2 teacher source count closures must be objects")
    split_keys = {"train", "val", "test", "formal"}
    if set(declared_source_counts) != split_keys or set(declared_recipient_counts) != split_keys:
        raise ValueError("fixed2 teacher source count closure key set changed")
    for split in sorted(split_keys):
        source_count = declared_source_counts.get(split)
        recipient_count = declared_recipient_counts.get(split)
        if type(source_count) is not int or source_count != int(source_split_counts[split]):
            raise ValueError(f"fixed2 teacher source {split} count changed")
        if type(recipient_count) is not int or recipient_count != int(source_recipient_counts[split]):
            raise ValueError(f"fixed2 teacher recipient {split} count changed")
    rows: list[dict[str, Any]] = [
        dict(raw)
        for _line_number, raw in output_manifest_records
    ]
    expected_row_order = [
        (source_id, view)
        for source_id in sorted(owners_by_id)
        for view in VIEWS
    ]
    observed_row_order = [
        (row.get("source_record_id"), row.get("view")) for row in rows
    ]
    if observed_row_order != expected_row_order:
        raise ValueError("fixed2 teacher manifest canonical row order changed")
    by_source: dict[str, set[str]] = {}
    generated: dict[str, _GeneratedViewOwner] = {}
    png_seals_by_name: dict[str, _FrozenFileSeal] = {}
    closing_input_seals: dict[Path, _FrozenFileSeal] = {}
    semantic_selected: list[dict[str, object]] = []
    semantic_by_source: dict[str, list[dict[str, object]]] = {}
    manifest_rows_by_source: dict[str, dict[str, dict[str, Any]]] = {}
    current_source_id: str | None = None
    current_expected_views: dict[str, np.ndarray] | None = None
    for number, row in enumerate(rows, start=1):
        if set(row) != RECORD_KEYS:
            raise ValueError(f"fixed2 teacher row {number} key set changed")
        if not _exact_json_value(row.get("schema_version"), SCHEMA_VERSION):
            raise ValueError(f"fixed2 teacher row {number} schema_version changed")
        if row.get("kind") != expected_record_kind:
            raise ValueError(f"fixed2 teacher row {number} kind changed")
        source_id, view = row.get("source_record_id"), row.get("view")
        if not isinstance(source_id, str) or source_id not in owners_by_id or view not in VIEWS:
            raise ValueError(f"fixed2 teacher row {number} owner/view changed")
        owner = owners_by_id[source_id]
        if source_id != current_source_id:
            source_snapshot = _snapshot_file(
                Path(str(owner["source"])),
                description=f"verified source image {source_id}",
            )
            result_snapshot = _snapshot_file(
                Path(str(owner["result_json"])),
                description=f"verified result JSON {source_id}",
            )
            crop_snapshot = _snapshot_file(
                Path(str(owner["paddle_crop"])),
                description=f"verified Paddle crop {source_id}",
            )
            current_expected_views = _selected_views(
                manifest=paths["source_manifest"],
                line_number=int(owner["line_number"]),
                record_id=source_id,
                source_snapshot=source_snapshot,
                result_snapshot=result_snapshot,
                crop_snapshot=crop_snapshot,
                declared_crop_sha256=str(owner["paddle_crop_pixel_sha256"]),
                bbox=tuple(owner["bbox"]),  # type: ignore[arg-type]
                hook_prefix=f"verify_{source_id}",
            )
            for snapshot in (source_snapshot, result_snapshot, crop_snapshot):
                closing_input_seals[snapshot.path] = _snapshot_seal(snapshot)
            owner["source_sha256"] = source_snapshot.sha256
            owner["result_json_sha256"] = result_snapshot.sha256
            owner["paddle_crop_file_sha256"] = crop_snapshot.sha256
            current_source_id = source_id
        if current_expected_views is None:
            raise AssertionError("fixed2 expected view cache is unavailable")
        group_id = str(owner["group_id"])
        target = str(owner["target"])
        target_sha = str(owner["target_sha256"])
        record_key = hashlib.sha256(source_id.encode("utf-8")).hexdigest()[:24]
        expected_row_id = f"recipient-{record_key}-{str(view).replace('_', '-')}"
        expected_image = f"images/{record_key}-{str(view).replace('_', '-')}.png"
        for key, expected in (
            ("id", expected_row_id),
            ("group_id", group_id),
            ("split", TRAIN_SPLIT),
            ("field", "recipient_field"),
            ("image", expected_image),
            ("text", target),
            ("target_sha256", target_sha), ("group_view_count", len(VIEWS)),
            ("target_source", "slots.recipient_field.text"),
            ("target_source_manifest_sha256", contract.get("source_manifest_sha256")),
            ("optimizer_supervision_split_eligible", True),
            ("optimizer_consumable", False),
            ("source", str(owner["source"])),
            ("source_sha256", owner["source_sha256"]),
            ("result_json", str(owner["result_json"])),
            ("result_json_sha256", owner["result_json_sha256"]),
            ("bbox_rectified", owner["bbox_rectified"]),
            ("paddle_crop", str(owner["paddle_crop"])),
            ("paddle_crop_pixel_sha256", owner["paddle_crop_pixel_sha256"]),
            ("paddle_crop_file_sha256", owner["paddle_crop_file_sha256"]),
        ):
            if not _exact_json_value(row.get(key), expected):
                raise ValueError(f"fixed2 teacher row {number} {key} changed")
        seen = by_source.setdefault(source_id, set())
        if str(view) in seen:
            raise ValueError(f"fixed2 teacher duplicate view for {source_id}")
        seen.add(str(view))
        relative = Path(str(row.get("image")))
        if relative.is_absolute() or ".." in relative.parts or relative.parent != Path("images"):
            raise ValueError("fixed2 teacher image path escapes images")
        image = root / relative
        png_snapshot = _snapshot_file(
            image,
            description=f"fixed2 teacher selected PNG {source_id}/{view}",
        )
        if row.get("view_file_sha256") != png_snapshot.sha256:
            raise ValueError("fixed2 teacher selected PNG file hash changed")
        decoded = _snapshot_rgb(
            png_snapshot,
            description=f"fixed2 teacher selected PNG {source_id}/{view}",
            hook_prefix=f"verify_png_{source_id}_{view}",
        )
        width = row.get("view_width")
        height = row.get("view_height")
        if (
            isinstance(width, bool)
            or not isinstance(width, int)
            or isinstance(height, bool)
            or not isinstance(height, int)
            or width <= 0
            or height <= 0
            or width != int(decoded.shape[1])
            or height != int(decoded.shape[0])
        ):
            raise ValueError("fixed2 teacher selected PNG dimensions changed")
        digest = _crop_digest(decoded)
        if row.get("view_pixel_sha256") != digest:
            raise ValueError("fixed2 teacher selected PNG decoded pixels changed")
        if not np.array_equal(decoded, current_expected_views[str(view)]):
            raise ValueError(
                f"fixed2 teacher selected PNG differs from recomputed {view} view"
            )
        declared_split = crop_splits.get(digest)
        if declared_split is not None and declared_split != TRAIN_SPLIT:
            raise ValueError("fixed2 teacher selected PNG crosses split boundary")
        _register_generated_view_owner(
            generated,
            pixel_sha256=digest,
            owner=_GeneratedViewOwner(number, str(row.get("id")), str(view), group_id, target_sha, tuple(int(x) for x in decoded.shape)),
        )
        png_seals_by_name[relative.name] = _snapshot_seal(png_snapshot)
        closing_input_seals[png_snapshot.path] = _snapshot_seal(png_snapshot)
        semantic_by_source.setdefault(source_id, []).append(
            {"view": str(view), "pixel_sha256": digest, "width": int(decoded.shape[1]), "height": int(decoded.shape[0])}
        )
        manifest_rows_by_source.setdefault(source_id, {})[str(view)] = row
    if set(by_source) != set(owners_by_id) or any(views != set(VIEWS) for views in by_source.values()):
        raise ValueError("fixed2 teacher manifest coverage changed")
    expected_view_counts = {view: len(owners_by_id) for view in VIEWS}
    actual_view_counts = Counter(str(row["view"]) for row in rows)
    for key, expected in (
        ("output_records", len(rows)),
        ("output_split_counts", {TRAIN_SPLIT: len(rows)}),
        ("view_counts", expected_view_counts),
    ):
        observed = contract.get(key)
        if key == "output_records":
            valid = type(observed) is int and observed == expected
        elif isinstance(observed, Mapping) and set(observed) == set(expected):
            valid = all(
                type(observed[name]) is int and observed[name] == expected[name]
                for name in expected
            )
        else:
            valid = False
        if not valid:
            raise ValueError(f"fixed2 teacher contract {key} changed")
    if dict(actual_view_counts) != expected_view_counts:
        raise ValueError("fixed2 teacher manifest view counts changed")
    closure = contract.get("selected_view_hash_closure")
    expected_closure = {
        "views_per_train_record": len(VIEWS),
        "decoded_pixels_reverified": True,
        "blind_owner_fields": ["split", "group_id", "target_sha256"],
        "cross_split_conflicts": 0,
        "cross_target_conflicts": 0,
        "cross_group_conflicts": 0,
        "same_target_different_group_duplicate_policy": "reject",
    }
    if (
        not isinstance(closure, Mapping)
        or set(closure) != set(expected_closure)
        or any(
            type(closure[key]) is not type(expected)
            or closure[key] != expected
            for key, expected in expected_closure.items()
        )
    ):
        raise ValueError("fixed2 teacher selected-view closure declaration changed")
    expected_image_names = {
        Path(str(row["image"])).name
        for row in rows
    }
    observed_image_names = {entry.name for entry in os.scandir(images)}
    if observed_image_names != expected_image_names:
        raise ValueError("fixed2 teacher image file set changed")
    observed_root_names = {entry.name for entry in os.scandir(root)}
    allowed_root_names = {"images", MANIFEST_NAME, expected_contract_marker}
    if observed_root_names not in (
        {"images", MANIFEST_NAME},
        allowed_root_names,
    ):
        raise ValueError("fixed2 teacher root entry set changed")
    for source_id in sorted(owners_by_id):
        owner = owners_by_id[source_id]
        group_id = str(owner["group_id"])
        target_sha = str(owner["target_sha256"])
        # The per-source view index was built while validating the manifest.
        # Use it directly: rescanning every output row here would make the
        # formal 78k-source verifier quadratic.
        row_for_source = manifest_rows_by_source[source_id][VIEWS[0]]
        closure_payload = {
            "source_record_id": source_id,
            "source_group_id": group_id,
            "source_manifest_sha256": contract["source_manifest_sha256"],
            "target_sha256": target_sha,
            "source_sha256": owner["source_sha256"],
            "result_json_sha256": owner["result_json_sha256"],
            "paddle_crop_pixel_sha256": owner["paddle_crop_pixel_sha256"],
            "views": [
                {
                    "view": view,
                    "pixel_sha256": manifest_rows_by_source[source_id][view]["view_pixel_sha256"],
                    "file_sha256": manifest_rows_by_source[source_id][view]["view_file_sha256"],
                }
                for view in VIEWS
            ],
        }
        expected_group_closure = _canonical_sha256(closure_payload)
        for view in VIEWS:
            if (
                manifest_rows_by_source[source_id][view].get("group_closure_sha256")
                != expected_group_closure
            ):
                raise ValueError(f"fixed2 teacher group closure changed: {source_id}")
        semantic_selected.append(
            {
                "source_record_id": source_id,
                "group_id": group_id,
                "target_sha256": target_sha,
                "paddle_crop_pixel_sha256": row_for_source["paddle_crop_pixel_sha256"],
                "views": sorted(semantic_by_source[source_id], key=lambda item: VIEWS.index(str(item["view"]))),
            }
        )
    subject_material = {
        "domain": SUBJECT_DOMAIN,
        "schema_version": SCHEMA_VERSION,
        "source_dataset_kind": source_contract.get("kind"),
        "source_manifest_semantic_sha256": _canonical_sha256(semantic_source),
        "target_authority": "existing_paddle_train_manifest_only",
        "view_order": list(VIEWS),
        "view_geometry": {
            "standard_margin_ratio": STANDARD_MARGIN_RATIO,
            "fixed_value_left_trim": FIXED_VALUE_LEFT_TRIM,
        },
        "selected_semantic_bindings": semantic_selected,
        "same_target_different_group_duplicate_policy": "reject",
    }
    if contract.get("producer_subject_id") != _canonical_sha256(subject_material):
        raise ValueError("fixed2 teacher path-stable semantic subject changed")
    identity = contract.get("publication_identity")
    if require_publication_identity:
        if not isinstance(identity, Mapping):
            raise ValueError("formal fixed2 teacher publication identity is missing")
        root_identity = actual_root_identity or _directory_identity(root)
        images_identity = actual_images_identity or _directory_identity(images)
        recomputed = _publication_identity(
            root_identity=root_identity,
            images_identity=images_identity,
            manifest_identity=output_manifest_snapshot.identity,
            image_identities={
                name: seal.identity
                for name, seal in png_seals_by_name.items()
            },
        )
        if not _exact_json_value(dict(identity), recomputed):
            raise ValueError("formal fixed2 teacher publication identity changed")
    elif identity is not None:
        raise ValueError("analysis fixed2 teacher fixture must not claim publication identity")
    # Closing proof: every source/result/crop/PNG and the manifest/artifact
    # closure must still be the same object and bytes consumed above.  Only
    # lightweight seals are retained, so peak memory remains bounded by one
    # source row plus its two selected views.
    for seal in closing_input_seals.values():
        _resnapshot_matches(seal, description="fixed2 verifier closing input")
    _resnapshot_matches(
        _snapshot_seal(output_manifest_snapshot),
        description="fixed2 verifier closing output manifest",
    )
    for name, snapshot in artifact_snapshots.items():
        closing = _resnapshot_matches(
            _snapshot_seal(snapshot),
            description=f"fixed2 teacher closing artifact {name}",
        )
        _require_samefile(
            closing.path,
            paths[str(name)],
            description=f"fixed2 teacher closing artifact {name}",
        )
    if contract_snapshot is not None:
        _resnapshot_matches(
            contract_snapshot,
            description="fixed2 verifier closing commit marker",
        )
    return dict(contract)


def materialize_recipient_fixed2_teacher(
    *, manifest: Path, output_root: Path,
    dataset_root: Path | None = None, dataset_contract: Path | None = None
) -> dict[str, object]:
    """Publish one canonical fixed2 teacher source on Windows only."""

    return _materialize_impl(
        formal=True, manifest=manifest, output_root=output_root,
        dataset_root=dataset_root, dataset_contract=dataset_contract,
    )


def _materialize_recipient_fixed2_teacher_analysis_test_only(
    *, manifest: Path, output_root: Path,
    dataset_root: Path | None = None, dataset_contract: Path | None = None
) -> dict[str, object]:
    return _materialize_impl(
        formal=False, manifest=manifest, output_root=output_root,
        dataset_root=dataset_root, dataset_contract=dataset_contract,
    )


def verify_recipient_fixed2_teacher(*, export_root: Path) -> dict[str, object]:
    """Reopen one canonical fixed2 teacher publication on Windows only."""

    if not _running_on_windows():
        raise OSError(errno.ENOTSUP, "formal fixed2 teacher verification requires Windows", os.fspath(export_root))
    root = Path(export_root).resolve()
    contract_path = root / CONTRACT_NAME
    _assert_no_reparse_components(
        contract_path, description="formal fixed2 teacher commit marker"
    )
    contract_snapshot, contract = _snapshot_json(
        contract_path,
        description="formal fixed2 teacher commit marker",
        hook_prefix="formal_contract_marker",
    )
    return _verify_payload(
        contract, export_root=root, expected_kind=KIND,
        expected_record_kind=RECORD_KIND,
        expected_authority=PUBLICATION_AUTHORITY,
        expected_contract_marker=CONTRACT_NAME,
        require_publication_identity=True,
        require_hard_attestation=True,
        contract_snapshot=contract_snapshot,
    )


def inspect_recipient_fixed2_teacher_attestation_candidate(
    *, export_root: Path
) -> dict[str, object]:
    """Content-verify and report pins without granting formal authority."""

    if not _running_on_windows():
        raise OSError(
            errno.ENOTSUP,
            "formal fixed2 teacher candidate inspection requires Windows",
            os.fspath(export_root),
        )
    root = Path(export_root).resolve()
    contract_path = root / CONTRACT_NAME
    _assert_no_reparse_components(
        contract_path, description="formal fixed2 teacher candidate marker"
    )
    contract_snapshot, contract = _snapshot_json(
        contract_path,
        description="formal fixed2 teacher candidate marker",
        hook_prefix="formal_candidate_contract_marker",
    )
    verified = _verify_payload(
        contract,
        export_root=root,
        expected_kind=KIND,
        expected_record_kind=RECORD_KIND,
        expected_authority=PUBLICATION_AUTHORITY,
        expected_contract_marker=CONTRACT_NAME,
        require_publication_identity=True,
        require_hard_attestation=False,
        contract_snapshot=contract_snapshot,
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": "receipt_recipient_fixed2_teacher_attestation_candidate_v1",
        "formal_authority_granted": False,
        "contract_path": str(contract_snapshot.path),
        "contract_sha256": contract_snapshot.sha256,
        "contract_size_bytes": contract_snapshot.size_bytes,
        "producer_subject_id": verified["producer_subject_id"],
        "publication_identity_sha256": _canonical_sha256(
            verified["publication_identity"]
        ),
        "required_pin_module": "recipient_fixed2_teacher_attestation.py",
    }


def _verify_recipient_fixed2_teacher_analysis_test_only(*, export_root: Path) -> dict[str, object]:
    if _running_on_windows():
        raise OSError(
            errno.ENOTSUP,
            "analysis fixed2 teacher verification is disabled on Windows",
            os.fspath(export_root),
        )
    root = Path(export_root).resolve()
    contract_path = root / ANALYSIS_CONTRACT_NAME
    _assert_no_reparse_components(
        contract_path, description="analysis fixed2 teacher commit marker"
    )
    contract_snapshot, contract = _snapshot_json(
        contract_path,
        description="analysis fixed2 teacher commit marker",
        hook_prefix="analysis_contract_marker",
    )
    return _verify_payload(
        contract, export_root=root, expected_kind=ANALYSIS_KIND,
        expected_record_kind=ANALYSIS_RECORD_KIND,
        expected_authority=ANALYSIS_PUBLICATION_AUTHORITY,
        expected_contract_marker=ANALYSIS_CONTRACT_NAME,
        require_publication_identity=False,
        require_hard_attestation=False,
        contract_snapshot=contract_snapshot,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Canonical Windows-only recipient fixed2 teacher producer")
    subparsers = parser.add_subparsers(dest="command", required=True)
    materialize = subparsers.add_parser("materialize")
    materialize.add_argument("--manifest", type=Path, required=True)
    materialize.add_argument("--dataset-contract", type=Path)
    materialize.add_argument("--dataset-root", type=Path)
    materialize.add_argument("--output", type=Path, required=True)
    verify = subparsers.add_parser("verify")
    verify.add_argument("--export-root", type=Path, required=True)
    inspect_candidate = subparsers.add_parser("inspect-candidate")
    inspect_candidate.add_argument("--export-root", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "materialize":
        result = materialize_recipient_fixed2_teacher(
            manifest=args.manifest,
            dataset_contract=args.dataset_contract,
            dataset_root=args.dataset_root,
            output_root=args.output,
        )
    elif args.command == "verify":
        result = verify_recipient_fixed2_teacher(export_root=args.export_root)
    else:
        result = inspect_recipient_fixed2_teacher_attestation_candidate(
            export_root=args.export_root
        )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
