"""Strictly consume a train-only recipient multiview export.

The producer deliberately marks its JSONL as *not* optimizer consumable.  This
module is the narrow trust boundary that reopens every bound artifact, proves
one complete four-view group for each blind-manifest train recipient, and then
attaches those views to the original in-memory receipt records.  It never
duplicates a receipt and never attaches anything to validation rows.
"""

from __future__ import annotations

import argparse
import ctypes
import errno
import hashlib
import json
import math
import os
import stat
import sys
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .ocr_pseudolabels import _crop_digest
from .ocr_unified_dataset import RECIPIENT_QUALITY_POLICY_VERSION
from .recipient_blind_manifest import KIND as BLIND_CONTRACT_KIND
from .recipient_fixed2_teacher_export import (
    ADAPTER_MARKER as FIXED2_SOURCE_ADAPTER_MARKER,
    ANALYSIS_CONTRACT_NAME as FIXED2_SOURCE_ANALYSIS_CONTRACT_NAME,
    ANALYSIS_KIND as FIXED2_SOURCE_ANALYSIS_KIND,
    ANALYSIS_PUBLICATION_AUTHORITY as FIXED2_SOURCE_ANALYSIS_PUBLICATION_AUTHORITY,
    ANALYSIS_RECORD_KIND as FIXED2_SOURCE_ANALYSIS_RECORD_KIND,
    CONTRACT_NAME as FIXED2_SOURCE_CONTRACT_NAME,
    HARD_ATTESTATION_SCHEME as FIXED2_SOURCE_HARD_ATTESTATION_SCHEME,
    KIND as FIXED2_SOURCE_KIND,
    MANIFEST_NAME as FIXED2_SOURCE_MANIFEST_NAME,
    PUBLICATION_AUTHORITY as FIXED2_SOURCE_PUBLICATION_AUTHORITY,
    RECORD_KIND as FIXED2_SOURCE_RECORD_KIND,
    VIEWS as FIXED2_SOURCE_VIEWS,
    _verify_recipient_fixed2_teacher_analysis_test_only,
    verify_recipient_fixed2_teacher,
)
from .recipient_multiview_teacher_export import (
    KIND as EXPORT_CONTRACT_KIND,
    RECORD_KIND as EXPORT_RECORD_KIND,
    SUPPORTED_UNIFIED_KINDS,
    VIEWS,
    _GeneratedViewOwner,
    _register_generated_view_owner,
)


SCHEMA_VERSION = 1
CONSUMER_KIND = "receipt_recipient_multiview_overlay_consumer_v1"
FIXED2_CONTRACT_KIND = "receipt_recipient_fixed2_overlay_contract_v2"
FIXED2_ANALYSIS_CONTRACT_KIND = (
    "receipt_recipient_fixed2_overlay_analysis_fixture_v2"
)
FIXED2_PUBLICATION_AUTHORITY = "windows_parent_relative_ntcreatefile_v1"
FIXED2_ANALYSIS_PUBLICATION_AUTHORITY = (
    "posix_descriptor_anchored_analysis_fixture_v1"
)
FIXED2_CANONICAL_CONTRACT_NAME = "fixed2_overlay.contract.json"
FIXED2_ANALYSIS_MARKER_NAME = "fixed2_overlay.analysis.json"
ATTACHMENT_KEY = "_recipient_multiview_overlay_v1"
SELECTOR_MODE = "sha256_seed_source_offset_plus_epoch_cycle_v1"
FIXED2_SELECTOR_MODE = "context_distinct_fixed_value_pair_anti_repeat_v2"
FIXED2_VIEWS = ("standard", "fixed_value")
FIXED2_SELECTOR_DOMAIN = "receipt-recipient-fixed2-context-distinct-pair-v2"
FIXED2_SUBJECT_DOMAIN = "receipt-recipient-fixed2-overlay-subject-v2"
FIXED2_REUSE_POLICY = "context_distinct_fixed_value_pair_reuse_v1"
FIXED2_REUSE_CLASS_DOMAIN = (
    "receipt-recipient-fixed2-context-distinct-fixed-value-pair-v1"
)
FIXED2_NONBLANK_PREDICATE = "decoded_rgb_global_range_gt_zero_v1"
EXPECTED_ADAPTER_MARKER = "strict_recipient_multiview_overlay_loader_not_implemented"
_HEX = frozenset("0123456789abcdef")
_CODE_ARTIFACT_FILES = {
    "consumer_code": "recipient_multiview_overlay.py",
    "fixed2_producer_code": "recipient_fixed2_teacher_export.py",
    "fixed2_attestation_code": "recipient_fixed2_teacher_attestation.py",
    "producer_code": "recipient_multiview_teacher_export.py",
    "geometry_helper_code": "geometry.py",
    "ocr_helper_code": "ocr.py",
    "pseudolabel_helper_code": "ocr_pseudolabels.py",
    "unified_dataset_helper_code": "ocr_unified_dataset.py",
    "pipeline_crop_helper_code": "pipeline.py",
    "status_crop_helper_code": "status_crops.py",
}
_DATA_ARTIFACT_NAMES = frozenset(
    (
        "full_records",
        "blind_records",
        "blind_contract",
        "multiview_export_contract",
        "multiview_export_manifest",
        "source_dataset_contract",
        "composite_records",
        "composite_dataset_contract",
    )
)
_SEMANTIC_ARTIFACT_NAMES = _DATA_ARTIFACT_NAMES - {
    "composite_records",
    "composite_dataset_contract",
}
_SUBJECT_SEMANTIC_ARTIFACT_NAMES = _SEMANTIC_ARTIFACT_NAMES - {
    # These two files are strict integrity artifacts, but their producer
    # publication identity and PNG encoding are intentionally not semantic
    # route identity.  The producer's path-free subject and manifest closure
    # are bound separately below.
    "multiview_export_contract",
    "multiview_export_manifest",
    "source_dataset_contract",
}
_SOURCE_DATASET_OPTIONAL_ABI_FIELDS = (
    "architecture",
    "recipient_target",
    "recipient_charset",
    "recipient_charset_sha256",
    "recipient_oov_by_split",
    "status_text_target",
    "status_text_charset",
    "status_text_charset_sha256",
    "status_text_charset_source",
    "status_text_source_counts",
    "status_text_missing_reasons",
    "status_text_oov_by_split",
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
_WINDOWS_DIRECTORY_LEASE_ACCESS = (
    _WINDOWS_FILE_READ_ATTRIBUTES | _WINDOWS_FILE_TRAVERSE | _WINDOWS_GENERIC_EXECUTE
)
_WINDOWS_DIRECTORY_LEASE_SHARE = _WINDOWS_FILE_SHARE_READ | _WINDOWS_FILE_SHARE_WRITE


@dataclass(frozen=True)
class RecipientMultiviewOverlayVerification:
    """JSON evidence plus compact per-record attachments for the DataLoader."""

    policy: dict[str, object]
    attachments: dict[str, dict[str, object]]


@dataclass(frozen=True)
class _Fixed2DecodedViewOwner:
    source_record_id: str
    group_id: str
    target_sha256: str
    view: str
    pixel_sha256: str
    width: int
    height: int
    rgb_min: int
    rgb_max: int
    standard_pixel_sha256: str
    source: Path
    source_sha256: str
    result_json: Path
    result_json_sha256: str


FileIdentity = tuple[int, int, int, int, int, str]
DirectoryIdentity = tuple[int, int, int]
WindowsDirectoryIdentity = tuple[int, int, int]


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


def _exact_json_value(actual: object, expected: object) -> bool:
    """Compare JSON-shaped values without Python's bool/int equivalence."""

    if type(actual) is not type(expected):
        return False
    if isinstance(expected, Mapping):
        if set(actual) != set(expected):  # type: ignore[arg-type]
            return False
        return all(
            _exact_json_value(actual[key], value)  # type: ignore[index]
            for key, value in expected.items()
        )
    if isinstance(expected, list):
        return len(actual) == len(expected) and all(  # type: ignore[arg-type]
            _exact_json_value(left, right)
            for left, right in zip(actual, expected)  # type: ignore[arg-type]
        )
    return actual == expected


def _strict_json(path: Path) -> dict[str, Any]:
    def reject_constant(value: str) -> None:
        raise ValueError(f"{path}: non-finite JSON constant {value!r}")

    try:
        value = json.loads(
            path.read_text(encoding="utf-8-sig"),
            parse_constant=reject_constant,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"Unable to read strict JSON object {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    _reject_nonfinite(value, location=str(path))
    return value


def _strict_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    def reject_constant(value: str) -> None:
        raise ValueError(f"{path}: non-finite JSON constant {value!r}")

    with path.open("r", encoding="utf-8-sig") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                raw = json.loads(line, parse_constant=reject_constant)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from error
            if not isinstance(raw, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            _reject_nonfinite(raw, location=f"{path}:{line_number}")
            rows.append(raw)
    if not rows:
        raise ValueError(f"{path}: manifest is empty")
    return rows


def _reject_nonfinite(value: object, *, location: str) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{location}: non-finite JSON number")
    if isinstance(value, Mapping):
        for key, child in value.items():
            _reject_nonfinite(child, location=f"{location}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _reject_nonfinite(child, location=f"{location}[{index}]")


def _file_attributes(info: os.stat_result) -> int:
    return int(getattr(info, "st_file_attributes", 0))


def _stat_is_reparse(info: os.stat_result) -> bool:
    return stat.S_ISLNK(info.st_mode) or bool(
        _file_attributes(info)
        & int(getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400))
    )


def _is_reparse(path: Path) -> bool:
    return _stat_is_reparse(path.stat(follow_symlinks=False))


def _existing(path: Path, *, directory: bool, description: str) -> Path:
    raw = Path(os.path.abspath(os.fspath(path)))
    if not os.path.lexists(raw):
        raise FileNotFoundError(f"Missing {description}: {raw}")
    current = raw
    while True:
        if os.path.lexists(current) and _is_reparse(current):
            raise ValueError(f"{description} traverses a symlink/junction/reparse path: {current}")
        if current == current.parent:
            break
        current = current.parent
    resolved = raw.resolve(strict=True)
    if directory and not resolved.is_dir():
        raise ValueError(f"{description} is not a directory: {resolved}")
    if not directory and not resolved.is_file():
        raise ValueError(f"{description} is not a file: {resolved}")
    return resolved


def _relative_file(root: Path, value: object, *, description: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{description} must be a non-empty relative path")
    relative = Path(value)
    if relative.is_absolute() or relative.drive or relative.anchor or ".." in relative.parts:
        raise ValueError(f"{description} must be relative and contained")
    candidate = Path(os.path.abspath(os.fspath(root / relative)))
    path = _existing(candidate, directory=False, description=description)
    try:
        path.relative_to(root)
    except ValueError:
        raise ValueError(f"{description} escapes its bound root") from None
    return path


def _absolute_file(value: object, *, description: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{description} must be a non-empty absolute path")
    path = Path(value)
    if not path.is_absolute():
        raise ValueError(f"{description} must be absolute")
    return _existing(path, directory=False, description=description)


def _declared_absolute_path(value: object, *, description: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{description} must be a non-empty absolute path")
    path = Path(value)
    if not path.is_absolute() or path != Path(os.path.abspath(os.fspath(path))):
        raise ValueError(f"{description} must be an absolute normalized path")
    return path


def _samefile(left: Path, right: Path, *, description: str) -> None:
    try:
        same = os.path.samefile(left, right)
    except OSError as error:
        raise ValueError(f"Unable to compare {description}") from error
    if not same:
        raise ValueError(f"{description} is not the bound file")


def _require_sha(value: object, *, description: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in _HEX for character in value)
    ):
        raise ValueError(f"{description} must be a lowercase SHA-256")
    return value


def _binding(path: Path) -> dict[str, object]:
    return {
        "path": str(path),
        "sha256": _sha256(path),
        "size_bytes": path.stat().st_size,
    }


def _file_identity(path: Path) -> FileIdentity:
    before = path.stat(follow_symlinks=False)
    if not stat.S_ISREG(before.st_mode) or _stat_is_reparse(before):
        raise ValueError(f"expected a regular non-reparse file: {path}")
    digest = _sha256(path)
    after = path.stat(follow_symlinks=False)
    if not stat.S_ISREG(after.st_mode) or _stat_is_reparse(after):
        raise ValueError(f"expected a regular non-reparse file: {path}")
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
    if before_identity != after_identity:
        raise ValueError(f"file changed while its identity was recorded: {path}")
    return (*after_identity, digest)


def _file_identity_from_fd(descriptor: int) -> FileIdentity:
    before = os.fstat(descriptor)
    if not stat.S_ISREG(before.st_mode) or _stat_is_reparse(before):
        raise ValueError("expected a regular non-reparse anchored file")
    os.lseek(descriptor, 0, os.SEEK_SET)
    digest = hashlib.sha256()
    while True:
        block = os.read(descriptor, 1024 * 1024)
        if not block:
            break
        digest.update(block)
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
    if before_identity != after_identity:
        raise ValueError("anchored file changed while its identity was recorded")
    return (*after_identity, digest.hexdigest())


def _anchored_file_identity(stage: _DirectoryLease, name: str) -> FileIdentity:
    if stage.posix_fd is None:
        return _file_identity(stage.path / name)
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(name, flags, dir_fd=stage.posix_fd)
    try:
        return _file_identity_from_fd(descriptor)
    finally:
        os.close(descriptor)


def _write_anchored_stage_file(
    stage: _DirectoryLease, *, name: str, payload: bytes
) -> FileIdentity:
    if stage.posix_fd is None:
        path = stage.path / name
        with path.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        return _file_identity(path)
    flags = (
        os.O_RDWR
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_NOFOLLOW", 0)
    )
    descriptor = os.open(name, flags, 0o600, dir_fd=stage.posix_fd)
    try:
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("unable to complete anchored stage write")
            view = view[written:]
        os.fsync(descriptor)
        return _file_identity_from_fd(descriptor)
    finally:
        os.close(descriptor)


def _directory_identity(path: Path) -> DirectoryIdentity:
    if os.name == "nt":
        handle = _windows_open_path_directory_handle(
            path,
            desired_access=_WINDOWS_DIRECTORY_LEASE_ACCESS,
            share_access=(
                _WINDOWS_FILE_SHARE_READ
                | _WINDOWS_FILE_SHARE_WRITE
                | _WINDOWS_FILE_SHARE_DELETE
            ),
        )
        try:
            return _windows_directory_handle_identity(handle)
        finally:
            _windows_close_handle(handle)
    info = path.stat(follow_symlinks=False)
    return _directory_identity_from_stat(info, path=path)


def _directory_identity_from_stat(
    info: os.stat_result, *, path: Path | None = None
) -> DirectoryIdentity:
    if not stat.S_ISDIR(info.st_mode) or _stat_is_reparse(info):
        suffix = f": {path}" if path is not None else ""
        raise ValueError(f"expected a regular non-reparse directory{suffix}")
    return info.st_dev, info.st_ino, _file_attributes(info)


def _same_file_identity(path: Path, expected: FileIdentity) -> bool:
    try:
        return _file_identity(path) == expected
    except (OSError, ValueError):
        return False


def _same_directory_identity(path: Path, expected: DirectoryIdentity) -> bool:
    try:
        return _directory_identity(path) == expected
    except (OSError, ValueError):
        return False


@dataclass
class _DirectoryLease:
    path: Path
    identity: DirectoryIdentity
    posix_fd: int | None = None
    windows_handle: int | None = None
    windows_rename_capable: bool = False
    windows_identity: WindowsDirectoryIdentity | None = None

    def close(self) -> None:
        if self.posix_fd is not None:
            try:
                os.close(self.posix_fd)
            except OSError:
                pass
            self.posix_fd = None
        if self.windows_handle is not None:
            _windows_close_handle(self.windows_handle)
            self.windows_handle = None


def _windows_close_handle(handle: int) -> None:
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    close_handle = kernel32.CloseHandle
    close_handle.argtypes = (ctypes.c_void_p,)
    close_handle.restype = ctypes.c_int
    close_handle(ctypes.c_void_p(handle))


def _windows_open_path_directory_handle(
    path: Path,
    *,
    desired_access: int,
    share_access: int,
) -> int:
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
        desired_access,
        share_access,
        None,
        _WINDOWS_OPEN_EXISTING,
        _WINDOWS_FILE_FLAG_BACKUP_SEMANTICS
        | _WINDOWS_FILE_FLAG_OPEN_REPARSE_POINT,
        None,
    )
    invalid = ctypes.c_void_p(-1).value
    if handle in (None, invalid):
        error_number = ctypes.get_last_error()
        raise OSError(error_number, os.strerror(error_number), os.fspath(path))
    return int(handle)


def _windows_directory_handle_identity(handle: int) -> WindowsDirectoryIdentity:
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

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

    information = _ByHandleFileInformation()
    get_information = kernel32.GetFileInformationByHandle
    get_information.argtypes = (ctypes.c_void_p, ctypes.c_void_p)
    get_information.restype = ctypes.c_int
    if not get_information(ctypes.c_void_p(handle), ctypes.byref(information)):
        error_number = ctypes.get_last_error()
        raise OSError(error_number, os.strerror(error_number))
    attributes = int(information.file_attributes)
    if not attributes & 0x00000010 or attributes & 0x00000400:
        raise ValueError("Windows directory handle is not a non-reparse directory")
    return (
        int(information.volume_serial_number),
        (int(information.file_index_high) << 32) | int(information.file_index_low),
        attributes,
    )


def _directory_lease_open_hook(
    checkpoint: str, *, path: Path, handle: int | None
) -> None:
    """No-op concurrency hook for deterministic Windows lease tests."""


def _stage_directory_creation_hook(
    checkpoint: str,
    *,
    parent: _DirectoryLease,
    name: str,
    handle: int | None,
) -> None:
    """No-op concurrency hook for deterministic stage-creation tests."""


def _require_simple_child_name(name: str) -> str:
    if (
        not isinstance(name, str)
        or not name
        or name in {".", ".."}
        or "/" in name
        or "\\" in name
        or ":" in name
        or "\x00" in name
    ):
        raise ValueError("anchored directory creation requires one simple child name")
    return name


def _windows_nt_directory_handle(
    parent_handle: int,
    *,
    name: str,
    create_disposition: int,
    desired_access: int,
    share_access: int,
) -> int:
    """Create/open one directory relative to an already anchored parent handle."""

    name = _require_simple_child_name(name)
    ntdll = ctypes.WinDLL("ntdll", use_last_error=True)

    class _UnicodeString(ctypes.Structure):
        _fields_ = (
            ("length", ctypes.c_uint16),
            ("maximum_length", ctypes.c_uint16),
            ("buffer", ctypes.c_void_p),
        )

    class _ObjectAttributes(ctypes.Structure):
        _fields_ = (
            ("length", ctypes.c_uint32),
            ("root_directory", ctypes.c_void_p),
            ("object_name", ctypes.POINTER(_UnicodeString)),
            ("attributes", ctypes.c_uint32),
            ("security_descriptor", ctypes.c_void_p),
            ("security_quality_of_service", ctypes.c_void_p),
        )

    class _IoStatusBlock(ctypes.Structure):
        _fields_ = (
            ("status_or_pointer", ctypes.c_void_p),
            ("information", ctypes.c_size_t),
        )

    encoded_name = name.encode("utf-16-le")
    name_buffer = ctypes.create_string_buffer(encoded_name + b"\x00\x00")
    unicode_name = _UnicodeString(
        length=len(encoded_name),
        maximum_length=len(encoded_name) + 2,
        buffer=ctypes.addressof(name_buffer),
    )
    attributes = _ObjectAttributes(
        length=ctypes.sizeof(_ObjectAttributes),
        root_directory=parent_handle,
        object_name=ctypes.pointer(unicode_name),
        attributes=0x00000040,  # OBJ_CASE_INSENSITIVE
        security_descriptor=None,
        security_quality_of_service=None,
    )
    io_status = _IoStatusBlock()
    handle = ctypes.c_void_p()
    nt_create_file = ntdll.NtCreateFile
    nt_create_file.argtypes = (
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.c_uint32,
        ctypes.POINTER(_ObjectAttributes),
        ctypes.POINTER(_IoStatusBlock),
        ctypes.c_void_p,
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.c_void_p,
        ctypes.c_uint32,
    )
    nt_create_file.restype = ctypes.c_int32
    create_options = _WINDOWS_FILE_DIRECTORY_FILE | _WINDOWS_FILE_OPEN_REPARSE_POINT
    if desired_access & _WINDOWS_SYNCHRONIZE:
        # NtSetInformationFile must complete before publication validation.
        # A synchronous stage handle prevents STATUS_PENDING; NtCreateFile
        # requires SYNCHRONIZE access when this option is present.
        create_options |= _WINDOWS_FILE_SYNCHRONOUS_IO_NONALERT
    status = int(
        nt_create_file(
            ctypes.byref(handle),
            desired_access,
            ctypes.byref(attributes),
            ctypes.byref(io_status),
            None,
            0,
            share_access,
            create_disposition,
            create_options,
            None,
            0,
        )
    )
    if status < 0 or handle.value is None:
        rtl_error = ntdll.RtlNtStatusToDosError
        rtl_error.argtypes = (ctypes.c_int32,)
        rtl_error.restype = ctypes.c_uint32
        error_number = int(rtl_error(status))
        if error_number in {80, 183}:  # ERROR_FILE_EXISTS / ERROR_ALREADY_EXISTS
            raise FileExistsError(error_number, os.strerror(error_number), name)
        raise OSError(error_number, os.strerror(error_number), name)
    return int(handle.value)


def _require_directory_lease_identity(directory: _DirectoryLease) -> None:
    if directory.posix_fd is not None:
        if _directory_identity_from_stat(os.fstat(directory.posix_fd)) != directory.identity:
            raise ValueError("POSIX directory lease identity changed")
        return
    if directory.windows_handle is None or directory.windows_identity is None:
        raise ValueError("directory lease has no anchored descriptor or handle")
    if _windows_directory_handle_identity(directory.windows_handle) != directory.windows_identity:
        raise ValueError("Windows directory lease identity changed")


def _open_directory_lease(
    path: Path,
    *,
    expected: DirectoryIdentity,
    windows_rename_capable: bool = False,
) -> _DirectoryLease:
    """Anchor one directory, denying Windows rename/delete while held."""

    if os.name == "nt":
        _directory_lease_open_hook(
            "before_windows_open",
            path=path,
            handle=None,
        )
        handle = _windows_open_path_directory_handle(
            path,
            desired_access=(
                _WINDOWS_DIRECTORY_LEASE_ACCESS
                | (_WINDOWS_DELETE if windows_rename_capable else 0)
            ),
            share_access=_WINDOWS_DIRECTORY_LEASE_SHARE,
        )
        try:
            _directory_lease_open_hook(
                "after_windows_open_before_identity",
                path=path,
                handle=handle,
            )
            windows_identity = _windows_directory_handle_identity(handle)
            if windows_identity != expected:
                raise ValueError(
                    f"directory handle identity does not match expected identity: {path}"
                )
        except BaseException:
            _windows_close_handle(handle)
            raise
        return _DirectoryLease(
            path=path,
            identity=expected,
            windows_handle=handle,
            windows_rename_capable=windows_rename_capable,
            windows_identity=windows_identity,
        )

    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        observed = _directory_identity_from_stat(os.fstat(descriptor))
        if observed != expected:
            raise ValueError(f"directory changed before its POSIX lease was acquired: {path}")
    except BaseException:
        os.close(descriptor)
        raise
    return _DirectoryLease(path=path, identity=expected, posix_fd=descriptor)


def create_anchored_stage_directory(
    parent: _DirectoryLease,
    *,
    name: str,
) -> _DirectoryLease:
    """Atomically create and lease one child directory on Windows.

    The child is created relative to the already-open parent handle.  The
    returned handle requests DELETE access while deliberately withholding
    FILE_SHARE_DELETE, so the child cannot be renamed or substituted while the
    lease is held.  A second parent-handle-relative open proves that the handle
    returned by FILE_CREATE is the directory entry now bound under ``name``.

    There is no portable POSIX mkdir-and-open primitive with the same atomic
    guarantee.  Formal callers must therefore fail closed off Windows; the
    private POSIX helper below exists only for analysis/test materialization.
    """

    name = _require_simple_child_name(name)
    if parent.windows_handle is None or parent.windows_identity is None:
        raise OSError(
            errno.ENOTSUP,
            "atomic anchored stage creation requires a Windows directory lease",
            name,
        )
    if parent.posix_fd is not None:
        raise ValueError("directory lease cannot contain both POSIX and Windows handles")
    _require_directory_lease_identity(parent)
    created_handle = _windows_nt_directory_handle(
        parent.windows_handle,
        name=name,
        create_disposition=_WINDOWS_FILE_CREATE,
        desired_access=(
            _WINDOWS_DIRECTORY_LEASE_ACCESS
            | _WINDOWS_DELETE
            | _WINDOWS_SYNCHRONIZE
        ),
        share_access=_WINDOWS_DIRECTORY_LEASE_SHARE,
    )
    try:
        created_identity = _windows_directory_handle_identity(created_handle)
        _stage_directory_creation_hook(
            "after_windows_atomic_create_before_parent_relative_reopen",
            parent=parent,
            name=name,
            handle=created_handle,
        )
        _require_directory_lease_identity(parent)
        entry_handle = _windows_nt_directory_handle(
            parent.windows_handle,
            name=name,
            create_disposition=_WINDOWS_FILE_OPEN,
            desired_access=_WINDOWS_FILE_READ_ATTRIBUTES,
            # The existing create handle owns DELETE access, so this observer
            # must share DELETE even though it does not request it.  The create
            # handle itself still withholds FILE_SHARE_DELETE and therefore
            # prevents any competing rename/delete handle from opening.
            share_access=(
                _WINDOWS_FILE_SHARE_READ
                | _WINDOWS_FILE_SHARE_WRITE
                | _WINDOWS_FILE_SHARE_DELETE
            ),
        )
        try:
            entry_identity = _windows_directory_handle_identity(entry_handle)
        finally:
            _windows_close_handle(entry_handle)
        if entry_identity != created_identity:
            raise ValueError(
                "atomically created stage handle is not the child entry bound "
                "under its anchored parent"
            )
        _require_directory_lease_identity(parent)
        return _DirectoryLease(
            path=parent.path / name,
            identity=created_identity,
            windows_handle=created_handle,
            windows_rename_capable=True,
            windows_identity=created_identity,
        )
    except BaseException:
        _windows_close_handle(created_handle)
        raise


def _create_stage_lease(
    parent: _DirectoryLease, *, stage: Path
) -> _DirectoryLease:
    """Create a stage for analysis/tests, delegating formal Windows to NT.

    The POSIX branch is descriptor-anchored after acquisition and detects a
    deterministic post-create substitution, but mkdirat+openat is not atomic
    against an active writer.  It must not be used as a formal publication
    authority.
    """

    if stage.parent != parent.path:
        raise ValueError("stage must be one simple child of its leased parent")
    _require_simple_child_name(stage.name)
    if parent.windows_handle is not None:
        return create_anchored_stage_directory(parent, name=stage.name)
    if parent.posix_fd is not None:
        os.mkdir(stage.name, mode=0o700, dir_fd=parent.posix_fd)
        expected = _anchored_directory_entry_identity(parent, name=stage.name)
        _stage_directory_creation_hook(
            "after_posix_mkdir_before_open",
            parent=parent,
            name=stage.name,
            handle=None,
        )
        flags = (
            os.O_RDONLY
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_NOFOLLOW", 0)
        )
        descriptor = os.open(stage.name, flags, dir_fd=parent.posix_fd)
        try:
            identity = _directory_identity_from_stat(os.fstat(descriptor))
            if identity != expected:
                raise ValueError(
                    "POSIX stage directory changed between creation and lease acquisition"
                )
        except BaseException:
            os.close(descriptor)
            raise
        return _DirectoryLease(path=stage, identity=identity, posix_fd=descriptor)
    raise OSError(
        errno.ENOTSUP,
        "stage creation requires an anchored parent descriptor or handle",
        os.fspath(stage),
    )


def _code_artifact_paths() -> dict[str, Path]:
    package_root = Path(__file__).parent
    return {
        name: _existing(
            package_root / filename,
            directory=False,
            description=f"fixed2 {name}",
        )
        for name, filename in _CODE_ARTIFACT_FILES.items()
    }


def _require_directory_snapshot(
    directory: Path,
    *,
    expected_directory: DirectoryIdentity,
    expected_files: Mapping[str, FileIdentity],
    description: str,
) -> None:
    if not _same_directory_identity(directory, expected_directory):
        raise ValueError(f"{description} directory identity changed")
    try:
        observed_names = {entry.name for entry in os.scandir(directory)}
    except OSError as error:
        raise ValueError(f"unable to inspect {description} directory") from error
    if observed_names != set(expected_files):
        raise ValueError(f"{description} file set changed")
    for name, identity in expected_files.items():
        if not _same_file_identity(directory / name, identity):
            raise ValueError(f"{description} file identity changed: {name}")
    if not _same_directory_identity(directory, expected_directory):
        raise ValueError(f"{description} directory identity changed")


def _require_anchored_directory_snapshot(
    directory: _DirectoryLease,
    *,
    expected_directory: DirectoryIdentity,
    expected_files: Mapping[str, FileIdentity],
    description: str,
) -> None:
    _require_directory_lease_identity(directory)
    if directory.posix_fd is None:
        _require_directory_snapshot(
            directory.path,
            expected_directory=expected_directory,
            expected_files=expected_files,
            description=description,
        )
        _require_directory_lease_identity(directory)
        return
    if _directory_identity_from_stat(os.fstat(directory.posix_fd)) != expected_directory:
        raise ValueError(f"{description} anchored directory identity changed")
    observed_names = set(os.listdir(directory.posix_fd))
    if observed_names != set(expected_files):
        raise ValueError(f"{description} anchored file set changed")
    for name, identity in expected_files.items():
        try:
            observed = _anchored_file_identity(directory, name)
        except (OSError, ValueError) as error:
            raise ValueError(f"{description} anchored file changed: {name}") from error
        if observed != identity:
            raise ValueError(f"{description} anchored file identity changed: {name}")
    if _directory_identity_from_stat(os.fstat(directory.posix_fd)) != expected_directory:
        raise ValueError(f"{description} anchored directory identity changed")


def _anchored_directory_entry_identity(
    parent: _DirectoryLease, *, name: str
) -> DirectoryIdentity:
    if parent.posix_fd is None:
        return _directory_identity(parent.path / name)
    information = os.stat(name, dir_fd=parent.posix_fd, follow_symlinks=False)
    return _directory_identity_from_stat(information)


def _anchored_directory_names(directory: _DirectoryLease) -> set[str]:
    _require_directory_lease_identity(directory)
    if directory.posix_fd is not None:
        names = set(os.listdir(directory.posix_fd))
    else:
        names = {entry.name for entry in os.scandir(directory.path)}
    _require_directory_lease_identity(directory)
    return names


def _same_anchored_directory_entry(
    parent: _DirectoryLease, *, name: str, expected: DirectoryIdentity
) -> bool:
    try:
        return _anchored_directory_entry_identity(parent, name=name) == expected
    except (OSError, ValueError):
        return False


def _rename_directory_no_replace(source: Path, destination: Path) -> None:
    """Atomically rename a directory while refusing an existing destination."""

    if os.name == "nt":
        # Windows os.rename is no-replace for an existing destination.
        source.rename(destination)
        return
    library = ctypes.CDLL(None, use_errno=True)
    source_bytes = os.fsencode(source)
    destination_bytes = os.fsencode(destination)
    if sys.platform == "darwin":
        rename = library.renamex_np
        rename.argtypes = (ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint)
        rename.restype = ctypes.c_int
        result = rename(source_bytes, destination_bytes, 0x00000004)  # RENAME_EXCL
    elif sys.platform.startswith("linux") and hasattr(library, "renameat2"):
        rename = library.renameat2
        rename.argtypes = (
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        )
        rename.restype = ctypes.c_int
        result = rename(-100, source_bytes, -100, destination_bytes, 0x00000001)
    else:
        raise OSError(
            errno.ENOTSUP,
            "atomic no-replace directory publication is unavailable",
            os.fspath(destination),
        )
    if result == 0:
        return
    error_number = ctypes.get_errno()
    if error_number in {errno.EEXIST, errno.ENOTEMPTY}:
        raise FileExistsError(
            error_number,
            os.strerror(error_number),
            os.fspath(destination),
        )
    raise OSError(error_number, os.strerror(error_number), os.fspath(destination))


def _rename_directory_no_replace_anchored(
    parent: _DirectoryLease,
    source_lease: _DirectoryLease,
    *,
    source: Path,
    destination: Path,
) -> None:
    if parent.posix_fd is None:
        if (
            parent.windows_handle is None
            or source_lease.windows_handle is None
            or not source_lease.windows_rename_capable
        ):
            raise OSError(
                errno.ENOTSUP,
                "Windows anchored publication requires parent and rename-capable source handles",
                os.fspath(destination),
            )
        if source.parent != parent.path or destination.parent != parent.path:
            raise ValueError("Windows anchored publication must stay within its leased parent")
        if source.name in {"", ".", ".."} or destination.name in {"", ".", ".."}:
            raise ValueError("Windows anchored publication requires simple entry names")
        _require_directory_lease_identity(parent)
        _require_directory_lease_identity(source_lease)

        class _FileRenameInformationPrefix(ctypes.Structure):
            _fields_ = (
                ("flags", ctypes.c_uint32),
                ("root_directory", ctypes.c_void_p),
                ("file_name_length", ctypes.c_uint32),
                ("file_name", ctypes.c_uint16 * 1),
            )

        class _IoStatusUnion(ctypes.Union):
            _fields_ = (
                ("status", ctypes.c_int32),
                ("pointer", ctypes.c_void_p),
            )

        class _IoStatusBlock(ctypes.Structure):
            _fields_ = (
                ("status_or_pointer", _IoStatusUnion),
                ("information", ctypes.c_size_t),
            )

        encoded_name = destination.name.encode("utf-16-le")
        name_offset = _FileRenameInformationPrefix.file_name.offset
        # Windows requires the reported FILE_RENAME_INFORMATION buffer length to
        # include the complete fixed structure *and* FileNameLength bytes.
        # On 64-bit Windows sizeof(FILE_RENAME_INFO) is 24 while FileName is at
        # offset 20, so using offsetof(FileName) here under-reports the buffer
        # by four bytes and SetFileInformationByHandle fails with
        # ERROR_INVALID_PARAMETER.
        buffer_size = ctypes.sizeof(_FileRenameInformationPrefix) + len(encoded_name)
        buffer = ctypes.create_string_buffer(buffer_size)
        information = _FileRenameInformationPrefix.from_buffer(buffer)
        information.flags = 0  # FileRenameInformation: ReplaceIfExists == FALSE.
        information.root_directory = parent.windows_handle
        information.file_name_length = len(encoded_name)
        ctypes.memmove(
            ctypes.addressof(buffer) + name_offset,
            encoded_name,
            len(encoded_name),
        )
        io_status = _IoStatusBlock()
        ntdll = ctypes.WinDLL("ntdll", use_last_error=True)
        set_information = ntdll.NtSetInformationFile
        set_information.argtypes = (
            ctypes.c_void_p,
            ctypes.POINTER(_IoStatusBlock),
            ctypes.c_void_p,
            ctypes.c_uint32,
            ctypes.c_int,
        )
        set_information.restype = ctypes.c_int32
        status = int(
            set_information(
                ctypes.c_void_p(source_lease.windows_handle),
                ctypes.byref(io_status),
                ctypes.byref(buffer),
                buffer_size,
                10,  # FileRenameInformation
            )
        )
        completion_status = int(io_status.status_or_pointer.status)
        failed_status = status if status != 0 else completion_status
        if failed_status != 0:
            rtl_error = ntdll.RtlNtStatusToDosError
            rtl_error.argtypes = (ctypes.c_int32,)
            rtl_error.restype = ctypes.c_uint32
            error_number = int(rtl_error(failed_status))
            if error_number in {80, 183}:  # ERROR_FILE_EXISTS / ERROR_ALREADY_EXISTS
                raise FileExistsError(
                    error_number,
                    os.strerror(error_number),
                    os.fspath(destination),
                )
            raise OSError(error_number, os.strerror(error_number), os.fspath(destination))
        return
    if source_lease.posix_fd is None:
        raise OSError(
            errno.ENOTSUP,
            "POSIX anchored publication requires a source directory descriptor",
            os.fspath(source),
        )
    if _directory_identity_from_stat(os.fstat(source_lease.posix_fd)) != source_lease.identity:
        raise ValueError("POSIX anchored publication source identity changed")
    library = ctypes.CDLL(None, use_errno=True)
    source_bytes = os.fsencode(source.name)
    destination_bytes = os.fsencode(destination.name)
    if sys.platform == "darwin":
        rename = library.renameatx_np
        rename.argtypes = (
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        )
        rename.restype = ctypes.c_int
        result = rename(
            parent.posix_fd,
            source_bytes,
            parent.posix_fd,
            destination_bytes,
            0x00000004,
        )
    elif sys.platform.startswith("linux") and hasattr(library, "renameat2"):
        rename = library.renameat2
        rename.argtypes = (
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        )
        rename.restype = ctypes.c_int
        result = rename(
            parent.posix_fd,
            source_bytes,
            parent.posix_fd,
            destination_bytes,
            0x00000001,
        )
    else:
        raise OSError(
            errno.ENOTSUP,
            "anchored no-replace directory publication is unavailable",
            os.fspath(destination),
        )
    if result == 0:
        return
    error_number = ctypes.get_errno()
    if error_number in {errno.EEXIST, errno.ENOTEMPTY}:
        raise FileExistsError(
            error_number,
            os.strerror(error_number),
            os.fspath(destination),
        )
    raise OSError(error_number, os.strerror(error_number), os.fspath(destination))


def _fixed2_publication_hook(
    checkpoint: str,
    *,
    parent: Path,
    stage: Path,
    output_root: Path,
) -> None:
    """No-op concurrency hook used only by deterministic race tests."""


def _fixed2_publication_use_hook(
    checkpoint: str,
    *,
    parent: Path,
    stage: Path,
    output_root: Path,
) -> None:
    """No-op hook placed after identity checks and before anchored mutation."""


def _require_output_parent_identity(
    parent: Path,
    *,
    expected: DirectoryIdentity,
    checkpoint: str,
    stage: Path,
    output_root: Path,
) -> None:
    _fixed2_publication_hook(
        checkpoint,
        parent=parent,
        stage=stage,
        output_root=output_root,
    )
    if not _same_directory_identity(parent, expected):
        raise ValueError(
            f"fixed2 output parent identity changed at {checkpoint}: {parent}"
        )


def _expect(actual: object, expected: object, *, description: str) -> None:
    if type(actual) is not type(expected) or actual != expected:
        raise ValueError(f"{description} mismatch: expected {expected!r}, found {actual!r}")


def _mapping(value: object, *, description: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{description} must be an object")
    return value


def _blind_recipient_rows(
    *,
    blind_manifest: Path,
    dataset_root: Path,
) -> tuple[dict[str, dict[str, object]], Counter[str], Counter[str], list[dict[str, object]]]:
    raw_rows = _strict_jsonl(blind_manifest)
    ids: set[str] = set()
    group_splits: dict[str, str] = {}
    split_counts: Counter[str] = Counter()
    recipient_counts: Counter[str] = Counter()
    recipients: dict[str, dict[str, object]] = {}
    base_bindings: list[dict[str, object]] = []
    for index, raw in enumerate(raw_rows, start=1):
        record_id = raw.get("id")
        group_id = raw.get("group_id")
        split = raw.get("split")
        if not isinstance(record_id, str) or not record_id:
            raise ValueError(f"{blind_manifest}:{index}: invalid id")
        if record_id in ids:
            raise ValueError(f"{blind_manifest}:{index}: duplicate id {record_id!r}")
        ids.add(record_id)
        if not isinstance(group_id, str) or not group_id:
            raise ValueError(f"{blind_manifest}:{index}: invalid group_id")
        if split not in {"train", "val"}:
            raise ValueError(f"{blind_manifest}:{index}: blind manifest contains non-train/val row")
        split = str(split)
        split_counts[split] += 1
        prior = group_splits.setdefault(group_id, split)
        if prior != split:
            raise ValueError(f"{blind_manifest}:{index}: group crosses train/val boundary")
        slots = raw.get("slots")
        if not isinstance(slots, Mapping):
            raise ValueError(f"{blind_manifest}:{index}: slots must be an object")
        recipient = slots.get("recipient_field")
        if recipient is None:
            continue
        if not isinstance(recipient, Mapping):
            raise ValueError(f"{blind_manifest}:{index}: recipient slot must be an object")
        recipient_counts[split] += 1
        if split != "train":
            continue
        target = recipient.get("text")
        if (
            not isinstance(target, str)
            or not target
            or target != target.strip()
            or any(not character.isprintable() for character in target)
        ):
            raise ValueError(f"{blind_manifest}:{index}: invalid train recipient target")
        image = _relative_file(
            dataset_root,
            recipient.get("image"),
            description=f"blind recipient image {record_id}",
        )
        declared_crop_sha = _require_sha(
            recipient.get("crop_sha256"),
            description=f"blind recipient crop {record_id}",
        )
        with Image.open(image) as opened:
            decoded = np.asarray(opened.convert("RGB"))
        if decoded.ndim != 3 or decoded.shape[2] != 3 or min(decoded.shape[:2]) <= 0:
            raise ValueError(f"blind recipient image {record_id} is empty")
        if _crop_digest(decoded) != declared_crop_sha:
            raise ValueError(f"blind recipient image {record_id} pixel hash changed")
        source = _absolute_file(raw.get("source"), description=f"blind source {record_id}")
        result = _absolute_file(raw.get("result_json"), description=f"blind result {record_id}")
        base_bindings.append(
            {
                "source_record_id": record_id,
                "image": str(image),
                "file_sha256": _sha256(image),
                "file_size_bytes": image.stat().st_size,
                "pixel_sha256": declared_crop_sha,
                "width": int(decoded.shape[1]),
                "height": int(decoded.shape[0]),
            }
        )
        recipients[record_id] = {
            "source_record_id": record_id,
            "group_id": group_id,
            "target": target,
            "target_sha256": hashlib.sha256(target.encode("utf-8")).hexdigest(),
            "base_image": image,
            "crop_pixel_sha256": declared_crop_sha,
            "bbox_rectified": recipient.get("bbox_rectified"),
            "source": source,
            "source_sha256": _sha256(source),
            "result_json": result,
            "result_json_sha256": _sha256(result),
        }
    if not recipients:
        raise ValueError("blind manifest has no train recipient rows")
    return recipients, split_counts, recipient_counts, base_bindings


def _verify_blind_contract(
    *, blind_manifest: Path, blind_contract: Path, split_counts: Counter[str]
) -> tuple[dict[str, Any], Path]:
    contract = _strict_json(blind_contract)
    _expect(contract.get("schema_version"), SCHEMA_VERSION, description="blind contract schema")
    _expect(contract.get("kind"), BLIND_CONTRACT_KIND, description="blind contract kind")
    bound_blind = _absolute_file(contract.get("blind_manifest"), description="bound blind manifest")
    _samefile(bound_blind, blind_manifest, description="blind manifest")
    _expect(
        contract.get("blind_manifest_sha256"),
        _sha256(blind_manifest),
        description="blind manifest SHA-256",
    )
    counts = _mapping(contract.get("split_counts"), description="blind split counts")
    _expect(counts.get("train"), int(split_counts["train"]), description="blind train count")
    _expect(counts.get("val"), int(split_counts["val"]), description="blind val count")
    test_excluded = counts.get("test_excluded")
    if isinstance(test_excluded, bool) or not isinstance(test_excluded, int) or test_excluded <= 0:
        raise ValueError("blind contract does not prove an excluded test split")
    for key, expected in (
        ("optimizer_supervision_splits", ["train"]),
        ("checkpoint_selection_splits", ["val"]),
        ("final_gate_only_splits", ["test"]),
        ("test_labels_used", False),
        ("test_metrics_computed", False),
        ("test_examples_emitted", False),
    ):
        _expect(contract.get(key), expected, description=f"blind contract {key}")
    full_manifest = _absolute_file(
        contract.get("source_manifest"), description="blind source full manifest"
    )
    _expect(
        contract.get("source_manifest_sha256"),
        _sha256(full_manifest),
        description="blind source full manifest SHA-256",
    )
    return contract, full_manifest


def _verify_export_contract(
    *,
    contract: Mapping[str, Any],
    export_manifest: Path,
    blind_manifest: Path,
    dataset_root: Path,
    source_dataset_contract: Path,
    split_counts: Counter[str],
    recipient_counts: Counter[str],
    recipient_total: int,
) -> None:
    for key, expected in (
        ("schema_version", SCHEMA_VERSION),
        ("kind", EXPORT_CONTRACT_KIND),
        ("source_manifest_sha256", _sha256(blind_manifest)),
        ("source_dataset_contract_sha256", _sha256(source_dataset_contract)),
        ("target_source", "slots.recipient_field.text"),
        ("target_label_authority", "existing_paddle_train_manifest_only"),
        ("target_recomputed", False),
        ("optimizer_supervision_splits", ["train"]),
        ("optimizer_input_ready", False),
        ("records_role", "recipient_multiview_overlay_source_only"),
        ("optimizer_adapter_required", EXPECTED_ADAPTER_MARKER),
        ("held_out_target_values_used", False),
        ("held_out_target_values_validated", False),
        ("held_out_target_values_emitted", False),
        ("held_out_splits_excluded", ["formal", "test", "val"]),
        ("view_order", list(VIEWS)),
        ("train_manifest", "multiview_train.jsonl"),
        ("train_manifest_sha256", _sha256(export_manifest)),
        ("commit_marker", "dataset.contract.json"),
        ("publication_complete", True),
        ("production_route_authorized", False),
        ("source_train_recipient_records", recipient_total),
        ("output_records", recipient_total * len(VIEWS)),
    ):
        _expect(contract.get(key), expected, description=f"export contract {key}")
    bound_manifest = _absolute_file(
        contract.get("source_manifest"), description="export source blind manifest"
    )
    _samefile(bound_manifest, blind_manifest, description="export source blind manifest")
    bound_source_contract = _absolute_file(
        contract.get("source_dataset_contract"), description="export source dataset contract"
    )
    _samefile(
        bound_source_contract,
        source_dataset_contract,
        description="export source dataset contract",
    )
    bound_root = _existing(
        Path(str(contract.get("source_dataset_root"))),
        directory=True,
        description="export source dataset root",
    )
    _samefile(bound_root, dataset_root, description="export source dataset root")
    source_counts = _mapping(
        contract.get("source_manifest_split_counts"), description="export source split counts"
    )
    for split in ("train", "val"):
        _expect(source_counts.get(split), int(split_counts[split]), description=f"export {split} count")
    for split in ("test", "formal"):
        _expect(source_counts.get(split), 0, description=f"export excluded {split} count")
    source_recipient_counts = _mapping(
        contract.get("source_split_counts"), description="export source recipient counts"
    )
    for split in ("train", "val"):
        _expect(
            source_recipient_counts.get(split),
            int(recipient_counts[split]),
            description=f"export {split} recipient count",
        )
    for split in ("test", "formal"):
        _expect(source_recipient_counts.get(split), 0, description=f"export excluded {split} recipient")
    view_counts = _mapping(contract.get("view_counts"), description="export view counts")
    for view in VIEWS:
        _expect(view_counts.get(view), recipient_total, description=f"export {view} count")
    _expect(
        contract.get("output_split_counts"),
        {"train": recipient_total * len(VIEWS)},
        description="export output split counts",
    )
    closure = _mapping(contract.get("group_hash_closure"), description="export group closure")
    for key, expected in (
        ("views_per_train_record", len(VIEWS)),
        ("cross_split_group_conflicts", 0),
        ("cross_split_source_conflicts", 0),
        ("cross_split_recipient_crop_conflicts", 0),
        ("generated_view_target_or_group_conflicts", 0),
    ):
        _expect(closure.get(key), expected, description=f"export group closure {key}")


def _verify_source_dataset_contract(
    path: Path, *, expected_kind: object, expected_dataset_root: Path
) -> dict[str, Any]:
    contract = _strict_json(path)
    if contract.get("schema_version") != SCHEMA_VERSION or contract.get("kind") not in SUPPORTED_UNIFIED_KINDS:
        raise ValueError("source dataset contract is not a supported v11-v13 dataset")
    _expect(contract.get("kind"), expected_kind, description="source dataset kind")
    _expect(
        contract.get("recipient_charset_source"),
        "train_only_anchored_recipient_value",
        description="source dataset recipient charset authority",
    )
    quality = _mapping(contract.get("recipient_quality_policy"), description="recipient quality policy")
    _expect(
        quality.get("version"),
        RECIPIENT_QUALITY_POLICY_VERSION,
        description="recipient quality policy version",
    )
    _expect(quality.get("requires_leading_recipient_label"), True, description="recipient anchor policy")
    _expect(quality.get("target"), "anchored_recipient_value", description="recipient target policy")
    raw_root = contract.get("dataset_root")
    if raw_root is not None:
        # Dataset roots are directories; keep this separate from the file-only
        # helper so Windows drive and reparse checks remain identical.
        if not isinstance(raw_root, str) or not raw_root:
            raise ValueError("source dataset contract has an invalid dataset_root")
        declared_root = _existing(
            Path(raw_root), directory=True, description="source dataset contract root"
        )
        _samefile(
            declared_root,
            expected_dataset_root,
            description="source dataset contract root",
        )
    return contract


def _source_dataset_contract_semantic_payload(
    contract: Mapping[str, Any],
) -> dict[str, object]:
    """Normalize the path-free source ABI copied into the fixed2 dataset."""

    required = {
        "schema_version",
        "kind",
        "slot_order",
        "status_classes",
        "recipient_charset_source",
        "recipient_quality_policy",
    }
    missing = sorted(required - set(contract))
    if missing:
        raise ValueError(
            "source dataset contract lacks semantic ABI fields: "
            + ",".join(missing)
        )

    def string_list(value: object, *, description: str) -> list[str]:
        if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
            raise ValueError(f"{description} must be a string array")
        normalized = list(value)
        if any(not isinstance(item, str) or not item for item in normalized):
            raise ValueError(f"{description} must contain non-empty strings")
        if len(set(normalized)) != len(normalized):
            raise ValueError(f"{description} contains duplicates")
        return normalized

    schema = contract.get("schema_version")
    kind = contract.get("kind")
    if isinstance(schema, bool) or not isinstance(schema, int):
        raise ValueError("source dataset schema ABI must be an integer")
    if not isinstance(kind, str) or kind not in SUPPORTED_UNIFIED_KINDS:
        raise ValueError("source dataset kind ABI is unsupported")
    recipient_source = contract.get("recipient_charset_source")
    if recipient_source != "train_only_anchored_recipient_value":
        raise ValueError("source dataset recipient charset authority changed")
    quality = _mapping(
        contract.get("recipient_quality_policy"),
        description="source dataset recipient quality ABI",
    )
    if set(quality) != {
        "version",
        "requires_leading_recipient_label",
        "target",
    }:
        raise ValueError("source dataset recipient quality ABI key set changed")
    _expect(
        quality.get("version"),
        RECIPIENT_QUALITY_POLICY_VERSION,
        description="source dataset recipient quality ABI version",
    )
    _expect(
        quality.get("requires_leading_recipient_label"),
        True,
        description="source dataset recipient quality ABI anchor",
    )
    _expect(
        quality.get("target"),
        "anchored_recipient_value",
        description="source dataset recipient quality ABI target",
    )
    optional: dict[str, object] = {}
    for key in _SOURCE_DATASET_OPTIONAL_ABI_FIELDS:
        if key not in contract:
            continue
        value = contract[key]
        if key in {"recipient_charset", "status_text_charset"}:
            optional[key] = string_list(value, description=f"source dataset {key}")
        elif key in {"recipient_charset_sha256", "status_text_charset_sha256"}:
            optional[key] = _require_sha(value, description=f"source dataset {key}")
        elif key in {
            "recipient_oov_by_split",
            "status_text_source_counts",
            "status_text_missing_reasons",
            "status_text_oov_by_split",
        }:
            optional[key] = dict(
                _mapping(value, description=f"source dataset {key}")
            )
        else:
            if not isinstance(value, str) or not value:
                raise ValueError(f"source dataset {key} must be a non-empty string")
            optional[key] = value
    return {
        "domain": "receipt-recipient-fixed2-source-dataset-abi-v1",
        "schema_version": schema,
        "kind": kind,
        "slot_order": string_list(
            contract.get("slot_order"), description="source dataset slot_order"
        ),
        "status_classes": string_list(
            contract.get("status_classes"),
            description="source dataset status_classes",
        ),
        "recipient_charset_source": recipient_source,
        "recipient_quality_policy": dict(quality),
        "present_optional_fields": sorted(optional),
        "optional_fields": optional,
    }


def _source_dataset_contract_semantic_sha256(
    contract: Mapping[str, Any],
) -> str:
    return _canonical_sha256(_source_dataset_contract_semantic_payload(contract))


def _rounded_bbox(value: object, *, description: str) -> list[float]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 4:
        raise ValueError(f"{description} must contain four coordinates")
    result: list[float] = []
    for item in value:
        if isinstance(item, bool):
            raise ValueError(f"{description} contains a non-number")
        try:
            number = float(item)
        except (TypeError, ValueError) as error:
            raise ValueError(f"{description} contains a non-number") from error
        if not math.isfinite(number):
            raise ValueError(f"{description} contains a non-finite number")
        result.append(round(number, 4))
    return result


def _fixed2_declared_heldout_crop_hashes(blind_manifest: Path) -> set[str]:
    """Read only declared held-out crop identities, never held-out targets/files."""

    declared: set[str] = set()
    for row_number, row in enumerate(_strict_jsonl(blind_manifest), start=1):
        if row.get("split") != "val":
            continue
        slots = row.get("slots")
        if not isinstance(slots, Mapping):
            raise ValueError(
                f"{blind_manifest}:{row_number}: held-out slots must be an object"
            )
        recipient = slots.get("recipient_field")
        if recipient is None:
            continue
        if not isinstance(recipient, Mapping):
            raise ValueError(
                f"{blind_manifest}:{row_number}: held-out recipient must be an object"
            )
        declared.add(
            _require_sha(
                recipient.get("crop_sha256"),
                description=(
                    f"{blind_manifest}:{row_number}: held-out recipient crop"
                ),
            )
        )
    return declared


def _same_physical_file(left: Path, right: Path, *, description: str) -> bool:
    try:
        return os.path.samefile(left, right)
    except OSError as error:
        raise ValueError(
            f"unable to compare fixed2 consumer {description} identity"
        ) from error


def _fixed2_consumer_selected_view_hash_closure(
    *,
    attachments: Mapping[str, Mapping[str, object]],
    recipients: Mapping[str, Mapping[str, object]],
    heldout_crop_hashes: set[str],
) -> dict[str, object]:
    """Independently rebuild the fixed2 producer's decoded-pixel reuse ABI."""

    if set(attachments) != set(recipients):
        raise ValueError("fixed2 consumer reuse closure source coverage changed")
    owners: list[_Fixed2DecodedViewOwner] = []
    for source_id in sorted(attachments):
        attachment = _mapping(
            attachments[source_id],
            description=f"fixed2 consumer attachment {source_id}",
        )
        blind = _mapping(
            recipients[source_id],
            description=f"fixed2 consumer blind owner {source_id}",
        )
        views = _mapping(
            attachment.get("views"),
            description=f"fixed2 consumer views {source_id}",
        )
        if set(views) != set(FIXED2_VIEWS):
            raise ValueError(
                f"fixed2 consumer generated view coverage changed: {source_id}"
            )
        standard = _mapping(
            views["standard"],
            description=f"fixed2 consumer standard view {source_id}",
        )
        standard_sha256 = _require_sha(
            standard.get("pixel_sha256"),
            description=f"fixed2 consumer standard pixels {source_id}",
        )
        for view_name in FIXED2_VIEWS:
            view = _mapping(
                views[view_name],
                description=f"fixed2 consumer {view_name} view {source_id}",
            )
            numeric: dict[str, int] = {}
            for field in ("width", "height", "rgb_min", "rgb_max"):
                value = view.get(field)
                if isinstance(value, bool) or not isinstance(value, int):
                    raise ValueError(
                        f"fixed2 consumer {source_id}/{view_name} {field} changed"
                    )
                if field in {"width", "height"} and value <= 0:
                    raise ValueError(
                        f"fixed2 consumer {source_id}/{view_name} {field} changed"
                    )
                if field in {"rgb_min", "rgb_max"} and not 0 <= value <= 255:
                    raise ValueError(
                        f"fixed2 consumer {source_id}/{view_name} {field} changed"
                    )
                numeric[field] = value
            owners.append(
                _Fixed2DecodedViewOwner(
                    source_record_id=source_id,
                    group_id=str(blind["group_id"]),
                    target_sha256=_require_sha(
                        blind.get("target_sha256"),
                        description=f"fixed2 consumer target {source_id}",
                    ),
                    view=view_name,
                    pixel_sha256=_require_sha(
                        view.get("pixel_sha256"),
                        description=(
                            f"fixed2 consumer {view_name} pixels {source_id}"
                        ),
                    ),
                    width=numeric["width"],
                    height=numeric["height"],
                    rgb_min=numeric["rgb_min"],
                    rgb_max=numeric["rgb_max"],
                    standard_pixel_sha256=standard_sha256,
                    source=Path(str(blind["source"])),
                    source_sha256=_require_sha(
                        blind.get("source_sha256"),
                        description=f"fixed2 consumer source {source_id}",
                    ),
                    result_json=Path(str(blind["result_json"])),
                    result_json_sha256=_require_sha(
                        blind.get("result_json_sha256"),
                        description=f"fixed2 consumer result {source_id}",
                    ),
                )
            )

    by_record: dict[str, dict[str, _Fixed2DecodedViewOwner]] = {}
    by_pixel: dict[str, list[_Fixed2DecodedViewOwner]] = {}
    for owner in owners:
        views = by_record.setdefault(owner.source_record_id, {})
        if owner.view in views:
            raise ValueError(
                f"fixed2 consumer duplicate generated view "
                f"{owner.source_record_id}/{owner.view}"
            )
        views[owner.view] = owner
        if owner.pixel_sha256 in heldout_crop_hashes:
            raise ValueError(
                f"fixed2 generated train view {owner.pixel_sha256} crosses "
                "declared held-out crop boundary"
            )
        by_pixel.setdefault(owner.pixel_sha256, []).append(owner)
    if any(set(views) != set(FIXED2_VIEWS) for views in by_record.values()):
        raise ValueError(
            "fixed2 consumer generated view coverage changed before reuse closure"
        )

    reuse_classes: list[dict[str, object]] = []
    for pixel_sha256, raw_bucket in sorted(by_pixel.items()):
        if len(raw_bucket) == 1:
            continue
        bucket = sorted(
            raw_bucket,
            key=lambda item: (item.source_record_id, item.view),
        )
        if len({item.target_sha256 for item in bucket}) != 1:
            raise ValueError(
                f"generated view hash {pixel_sha256} conflict: "
                "target_conflict=true; fixed2 cross-target pixel reuse is forbidden"
            )
        if any(item.view != "fixed_value" for item in bucket):
            views = sorted({item.view for item in bucket})
            raise ValueError(
                f"generated view hash {pixel_sha256} standard/cross-view collision: "
                f"views={views!r}"
            )
        if len(bucket) != 2:
            raise ValueError(
                f"generated fixed_value hash {pixel_sha256} reuse class has "
                f"class_size={len(bucket)}; exactly two owners are required"
            )
        first, second = bucket
        if first.source_record_id == second.source_record_id:
            raise ValueError(
                f"generated fixed_value hash {pixel_sha256} reuses one source record"
            )
        if first.group_id == second.group_id:
            raise ValueError(
                f"generated fixed_value hash {pixel_sha256} same-group reuse is forbidden"
            )
        if first.standard_pixel_sha256 == second.standard_pixel_sha256:
            raise ValueError(
                f"generated fixed_value hash {pixel_sha256} has a standard context collision"
            )
        if (
            first.width != second.width
            or first.height != second.height
            or first.rgb_min != second.rgb_min
            or first.rgb_max != second.rgb_max
        ):
            raise ValueError(
                f"generated fixed_value hash {pixel_sha256} decoded proof changed "
                "across owners"
            )
        if first.rgb_max <= first.rgb_min:
            raise ValueError(
                f"generated fixed_value hash {pixel_sha256} is uniform under "
                f"{FIXED2_NONBLANK_PREDICATE}"
            )
        for name, left_path, right_path, left_sha, right_sha in (
            (
                "source",
                first.source,
                second.source,
                first.source_sha256,
                second.source_sha256,
            ),
            (
                "result_json",
                first.result_json,
                second.result_json,
                first.result_json_sha256,
                second.result_json_sha256,
            ),
        ):
            if _same_physical_file(
                left_path,
                right_path,
                description=name,
            ) or left_sha == right_sha:
                raise ValueError(
                    f"generated fixed_value hash {pixel_sha256} reuses {name} lineage"
                )
        class_material: dict[str, object] = {
            "domain": FIXED2_REUSE_CLASS_DOMAIN,
            "split": "train",
            "view": "fixed_value",
            "fixed_value_pixel_sha256": pixel_sha256,
            "target_sha256": first.target_sha256,
            "width": first.width,
            "height": first.height,
            "rgb_min": first.rgb_min,
            "rgb_max": first.rgb_max,
            "nonblank_predicate": FIXED2_NONBLANK_PREDICATE,
            "owners": [
                {
                    "source_record_id": item.source_record_id,
                    "group_id": item.group_id,
                    "standard_pixel_sha256": item.standard_pixel_sha256,
                }
                for item in bucket
            ],
        }
        reuse_classes.append(
            {
                **class_material,
                "reuse_class_id": _canonical_sha256(class_material),
            }
        )
    reuse_classes.sort(key=lambda item: str(item["reuse_class_id"]))
    return {
        "views_per_train_record": len(FIXED2_VIEWS),
        "decoded_pixels_reverified": True,
        "blind_owner_fields": [
            "split",
            "source_record_id",
            "group_id",
            "target_sha256",
        ],
        "cross_split_conflicts": 0,
        "cross_target_conflicts": 0,
        "forbidden_cross_group_conflicts": 0,
        "policy": FIXED2_REUSE_POLICY,
        "allowed_reuse_view": "fixed_value",
        "allowed_reuse_class_size": 2,
        "nonblank_predicate": FIXED2_NONBLANK_PREDICATE,
        "standard_pixel_reuse_allowed": False,
        "cross_view_pixel_reuse_allowed": False,
        "same_group_pixel_reuse_allowed": False,
        "source_or_result_lineage_reuse_allowed": False,
        "fixed_value_reuse_class_count": len(reuse_classes),
        "allowed_cross_group_reuse_class_count": len(reuse_classes),
        "fixed_value_reused_record_count": len(reuse_classes) * 2,
        "fixed_value_reuse_classes": reuse_classes,
        "fixed_value_reuse_class_closure_sha256": _canonical_sha256(
            reuse_classes
        ),
    }


def _verify_overlay_rows_impl(
    *,
    rows: Sequence[Mapping[str, Any]],
    export_root: Path,
    blind_sha256: str,
    recipients: Mapping[str, Mapping[str, object]],
    expected_views: Sequence[str] = VIEWS,
    expected_record_kind: str = EXPORT_RECORD_KIND,
    fixed2_reuse_policy: bool,
    heldout_crop_hashes: set[str] | None,
) -> tuple[
    dict[str, dict[str, object]],
    list[dict[str, object]],
    list[str],
    dict[str, object] | None,
]:
    expected_views = tuple(expected_views)
    if not expected_views or len(set(expected_views)) != len(expected_views):
        raise ValueError("overlay expected view profile is invalid")
    by_source: dict[str, dict[str, Mapping[str, Any]]] = {}
    ids: set[str] = set()
    image_bindings: list[dict[str, object]] = []
    source_artifacts: dict[str, dict[str, object]] = {}
    generated_hash_owners: dict[str, _GeneratedViewOwner] = {}
    if fixed2_reuse_policy:
        if (
            expected_views != FIXED2_VIEWS
            or expected_record_kind
            not in {FIXED2_SOURCE_RECORD_KIND, FIXED2_SOURCE_ANALYSIS_RECORD_KIND}
            or heldout_crop_hashes is None
        ):
            raise ValueError("fixed2 consumer reuse profile is invalid")
    elif heldout_crop_hashes is not None:
        raise ValueError("legacy overlay cannot accept fixed2 held-out evidence")
    for row_number, row in enumerate(rows, start=1):
        if row.get("schema_version") != SCHEMA_VERSION or row.get("kind") != expected_record_kind:
            raise ValueError(f"overlay row {row_number} kind/schema changed")
        row_id = row.get("id")
        source_id = row.get("source_record_id")
        view = row.get("view")
        if not isinstance(row_id, str) or not row_id or row_id in ids:
            raise ValueError(f"overlay row {row_number} has an invalid or duplicate id")
        ids.add(row_id)
        if not isinstance(source_id, str) or source_id not in recipients:
            raise ValueError(f"overlay row {row_number} is extra or is not a blind train recipient")
        if view not in expected_views:
            raise ValueError(f"overlay row {row_number} has an unsupported view")
        views = by_source.setdefault(source_id, {})
        if str(view) in views:
            raise ValueError(f"overlay recipient {source_id!r} has duplicate view {view!r}")
        blind = recipients[source_id]
        for key, expected in (
            ("group_id", blind["group_id"]),
            ("split", "train"),
            ("field", "recipient_field"),
            ("text", blind["target"]),
            ("target_sha256", blind["target_sha256"]),
            ("target_source", "slots.recipient_field.text"),
            ("target_source_manifest_sha256", blind_sha256),
            ("optimizer_supervision_split_eligible", True),
            ("optimizer_consumable", False),
            ("group_view_count", len(expected_views)),
            ("paddle_crop_pixel_sha256", blind["crop_pixel_sha256"]),
            ("source_sha256", blind["source_sha256"]),
            ("result_json_sha256", blind["result_json_sha256"]),
            ("bbox_rectified", _rounded_bbox(blind["bbox_rectified"], description="blind bbox")),
        ):
            _expect(row.get(key), expected, description=f"overlay row {row_number} {key}")
        source = _absolute_file(row.get("source"), description=f"overlay source {source_id}")
        result = _absolute_file(row.get("result_json"), description=f"overlay result {source_id}")
        crop = _absolute_file(row.get("paddle_crop"), description=f"overlay base crop {source_id}")
        _samefile(source, Path(str(blind["source"])), description=f"overlay source {source_id}")
        _samefile(result, Path(str(blind["result_json"])), description=f"overlay result {source_id}")
        _samefile(crop, Path(str(blind["base_image"])), description=f"overlay base crop {source_id}")
        _expect(
            row.get("paddle_crop_file_sha256"),
            _sha256(crop),
            description=f"overlay base crop file SHA {source_id}",
        )
        image = _relative_file(
            export_root,
            row.get("image"),
            description=f"overlay image {source_id}/{view}",
        )
        try:
            image.relative_to(export_root / "images")
        except ValueError:
            raise ValueError("overlay image must be contained by the export images directory") from None
        file_sha = _sha256(image)
        _expect(row.get("view_file_sha256"), file_sha, description=f"overlay file SHA {source_id}/{view}")
        with Image.open(image) as opened:
            pixels = np.asarray(opened.convert("RGB"))
        if pixels.ndim != 3 or pixels.shape[2] != 3 or min(pixels.shape[:2]) <= 0:
            raise ValueError(f"overlay image {source_id}/{view} is empty")
        width = row.get("view_width")
        height = row.get("view_height")
        if isinstance(width, bool) or not isinstance(width, int) or width <= 0:
            raise ValueError(f"overlay image {source_id}/{view} has invalid width")
        if isinstance(height, bool) or not isinstance(height, int) or height <= 0:
            raise ValueError(f"overlay image {source_id}/{view} has invalid height")
        _expect(width, int(pixels.shape[1]), description=f"overlay width {source_id}/{view}")
        _expect(height, int(pixels.shape[0]), description=f"overlay height {source_id}/{view}")
        pixel_sha = _crop_digest(pixels)
        _expect(row.get("view_pixel_sha256"), pixel_sha, description=f"overlay pixel SHA {source_id}/{view}")
        if not fixed2_reuse_policy:
            # The legacy four-view path remains strictly fail-closed.  Its
            # shared registrar is intentionally never taught the fixed2 v2
            # exception.
            _register_generated_view_owner(
                generated_hash_owners,
                pixel_sha256=pixel_sha,
                owner=_GeneratedViewOwner(
                    line_number=row_number,
                    record_id=row_id,
                    view=str(view),
                    group_id=str(blind["group_id"]),
                    target_sha256=str(blind["target_sha256"]),
                    shape=tuple(int(size) for size in pixels.shape),
                ),
            )
        image_bindings.append(
            {
                "source_record_id": source_id,
                "view": view,
                "path": str(image),
                "file_sha256": file_sha,
                "file_size_bytes": image.stat().st_size,
                "pixel_sha256": pixel_sha,
                "width": width,
                "height": height,
                "rgb_min": int(np.min(pixels)),
                "rgb_max": int(np.max(pixels)),
            }
        )
        source_artifacts[source_id] = {
            "source_sha256": blind["source_sha256"],
            "source_size_bytes": Path(str(blind["source"])).stat().st_size,
            "result_json_sha256": blind["result_json_sha256"],
            "result_json_size_bytes": Path(str(blind["result_json"])).stat().st_size,
            "base_crop_file_sha256": _sha256(crop),
            "base_crop_size_bytes": crop.stat().st_size,
        }
        views[str(view)] = row
    if set(by_source) != set(recipients):
        missing = sorted(set(recipients) - set(by_source))
        extra = sorted(set(by_source) - set(recipients))
        raise ValueError(f"overlay source coverage mismatch: missing={missing[:3]} extra={extra[:3]}")

    attachments: dict[str, dict[str, object]] = {}
    closure_ids: list[str] = []
    image_by_key = {
        (str(item["source_record_id"]), str(item["view"])): item for item in image_bindings
    }
    for source_id in sorted(recipients):
        views = by_source[source_id]
        if set(views) != set(expected_views):
            raise ValueError(
                f"overlay recipient {source_id!r} does not have exactly "
                f"{len(expected_views)} expected views"
            )
        blind = recipients[source_id]
        ordered = [views[view] for view in expected_views]
        closure_payload = {
            "source_record_id": source_id,
            "source_group_id": blind["group_id"],
            "source_manifest_sha256": blind_sha256,
            "target_sha256": blind["target_sha256"],
            "source_sha256": blind["source_sha256"],
            "result_json_sha256": blind["result_json_sha256"],
            "paddle_crop_pixel_sha256": blind["crop_pixel_sha256"],
            "views": [
                {
                    "view": view,
                    "pixel_sha256": image_by_key[(source_id, view)]["pixel_sha256"],
                    "file_sha256": image_by_key[(source_id, view)]["file_sha256"],
                }
                for view in expected_views
            ],
        }
        closure = _canonical_sha256(closure_payload)
        for row in ordered:
            _expect(
                row.get("group_closure_sha256"),
                closure,
                description=f"overlay group closure {source_id}",
            )
        closure_ids.append(closure)
        attachments[source_id] = {
            "schema_version": SCHEMA_VERSION,
            "source_record_id": source_id,
            "group_id": blind["group_id"],
            "target": blind["target"],
            "target_sha256": blind["target_sha256"],
            "views": {
                view: {
                    "path": str(image_by_key[(source_id, view)]["path"]),
                    "file_sha256": image_by_key[(source_id, view)]["file_sha256"],
                    "file_size_bytes": image_by_key[(source_id, view)]["file_size_bytes"],
                    "pixel_sha256": image_by_key[(source_id, view)]["pixel_sha256"],
                    "width": image_by_key[(source_id, view)]["width"],
                    "height": image_by_key[(source_id, view)]["height"],
                    "rgb_min": image_by_key[(source_id, view)]["rgb_min"],
                    "rgb_max": image_by_key[(source_id, view)]["rgb_max"],
                }
                for view in expected_views
            },
        }
    source_binding_rows = [
        {"source_record_id": source_id, **source_artifacts[source_id]}
        for source_id in sorted(source_artifacts)
    ]
    image_bindings.sort(
        key=lambda item: (
            str(item["source_record_id"]),
            expected_views.index(str(item["view"])),
        )
    )
    reuse_closure = (
        _fixed2_consumer_selected_view_hash_closure(
            attachments=attachments,
            recipients=recipients,
            heldout_crop_hashes=(
                heldout_crop_hashes
                if heldout_crop_hashes is not None
                else set()
            ),
        )
        if fixed2_reuse_policy
        else None
    )
    return (
        attachments,
        image_bindings + source_binding_rows,
        closure_ids,
        reuse_closure,
    )


def _verify_overlay_rows(
    *,
    rows: Sequence[Mapping[str, Any]],
    export_root: Path,
    blind_sha256: str,
    recipients: Mapping[str, Mapping[str, object]],
    expected_views: Sequence[str] = VIEWS,
    expected_record_kind: str = EXPORT_RECORD_KIND,
) -> tuple[dict[str, dict[str, object]], list[dict[str, object]], list[str]]:
    """Verify the legacy profile with its original strict collision policy."""

    attachments, bindings, closures, reuse = _verify_overlay_rows_impl(
        rows=rows,
        export_root=export_root,
        blind_sha256=blind_sha256,
        recipients=recipients,
        expected_views=expected_views,
        expected_record_kind=expected_record_kind,
        fixed2_reuse_policy=False,
        heldout_crop_hashes=None,
    )
    if reuse is not None:
        raise AssertionError("legacy overlay unexpectedly produced fixed2 reuse evidence")
    return attachments, bindings, closures


def _verify_fixed2_overlay_rows(
    *,
    rows: Sequence[Mapping[str, Any]],
    export_root: Path,
    blind_sha256: str,
    recipients: Mapping[str, Mapping[str, object]],
    expected_record_kind: str,
    heldout_crop_hashes: set[str],
) -> tuple[
    dict[str, dict[str, object]],
    list[dict[str, object]],
    list[str],
    dict[str, object],
]:
    """Verify only a producer-verified fixed2 v2 source and rebuild reuse."""

    attachments, bindings, closures, reuse = _verify_overlay_rows_impl(
        rows=rows,
        export_root=export_root,
        blind_sha256=blind_sha256,
        recipients=recipients,
        expected_views=FIXED2_SOURCE_VIEWS,
        expected_record_kind=expected_record_kind,
        fixed2_reuse_policy=True,
        heldout_crop_hashes=heldout_crop_hashes,
    )
    if reuse is None:
        raise AssertionError("fixed2 overlay reuse evidence was not rebuilt")
    return attachments, bindings, closures, reuse


def verify_recipient_multiview_overlay(
    *,
    blind_manifest: Path,
    blind_contract: Path,
    export_root: Path,
    dataset_root: Path,
    seed: int,
) -> RecipientMultiviewOverlayVerification:
    """Reopen and bind one complete train-only multiview overlay export."""

    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("recipient multiview selector seed must be a non-negative integer")
    blind_manifest = _existing(blind_manifest, directory=False, description="blind manifest")
    blind_contract = _existing(blind_contract, directory=False, description="blind contract")
    export_root = _existing(export_root, directory=True, description="multiview export root")
    dataset_root = _existing(dataset_root, directory=True, description="base dataset root")
    export_contract_path = _existing(
        export_root / "dataset.contract.json",
        directory=False,
        description="multiview export contract",
    )
    export_manifest = _existing(
        export_root / "multiview_train.jsonl",
        directory=False,
        description="multiview export manifest",
    )
    export_contract = _strict_json(export_contract_path)
    raw_source_contract = export_contract.get("source_dataset_contract")
    source_dataset_contract = _absolute_file(
        raw_source_contract, description="multiview source dataset contract"
    )
    source_contract = _verify_source_dataset_contract(
        source_dataset_contract,
        expected_kind=export_contract.get("source_dataset_kind"),
        expected_dataset_root=dataset_root,
    )
    recipients, split_counts, recipient_counts, base_bindings = _blind_recipient_rows(
        blind_manifest=blind_manifest,
        dataset_root=dataset_root,
    )
    _blind, full_manifest = _verify_blind_contract(
        blind_manifest=blind_manifest,
        blind_contract=blind_contract,
        split_counts=split_counts,
    )
    _verify_export_contract(
        contract=export_contract,
        export_manifest=export_manifest,
        blind_manifest=blind_manifest,
        dataset_root=dataset_root,
        source_dataset_contract=source_dataset_contract,
        split_counts=split_counts,
        recipient_counts=recipient_counts,
        recipient_total=len(recipients),
    )
    rows = _strict_jsonl(export_manifest)
    attachments, overlay_bindings, closure_ids = _verify_overlay_rows(
        rows=rows,
        export_root=export_root,
        blind_sha256=_sha256(blind_manifest),
        recipients=recipients,
    )
    view_counts = Counter(
        view for attachment in attachments.values() for view in attachment["views"]
    )
    bindings = {
        "blind_manifest": _binding(blind_manifest),
        "blind_contract": _binding(blind_contract),
        "full_manifest": _binding(full_manifest),
        "export_contract": _binding(export_contract_path),
        "export_manifest": _binding(export_manifest),
        "source_dataset_contract": _binding(source_dataset_contract),
    }
    policy: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "kind": CONSUMER_KIND,
        "analysis_only": True,
        "production_route_authorized": False,
        "producer_optimizer_input_ready": False,
        "consumer_optimizer_input_ready": True,
        "producer_adapter_marker": EXPECTED_ADAPTER_MARKER,
        "optimizer_supervision_splits": ["train"],
        "validation_records_modified": False,
        "train_records_expanded": False,
        "recipient_value_left_trim": 0.0,
        "bindings": bindings,
        "dataset_root": str(dataset_root),
        "source_dataset_kind": source_contract["kind"],
        "counts": {
            "blind_train_records": int(split_counts["train"]),
            "blind_val_records": int(split_counts["val"]),
            "blind_train_recipient_records": len(recipients),
            "overlay_records": len(rows),
            "views_per_train_recipient": len(VIEWS),
            "attached_train_records": len(attachments),
            "attached_val_records": 0,
        },
        "view_order": list(VIEWS),
        "view_counts": {view: int(view_counts[view]) for view in VIEWS},
        "selector": {
            "mode": SELECTOR_MODE,
            "seed": seed,
            "source_identity": "source_record_id",
            "epoch_origin": 1,
            "cycle_length": len(VIEWS),
            "exact_8_epochs_each_view_count": 2,
        },
        "base_recipient_bindings_sha256": _canonical_sha256(base_bindings),
        "overlay_artifact_bindings_sha256": _canonical_sha256(overlay_bindings),
        "group_closure_set_sha256": _canonical_sha256(sorted(closure_ids)),
        "post_training_full_reverification_required": True,
    }
    return RecipientMultiviewOverlayVerification(policy=policy, attachments=attachments)


def _verify_fixed2_teacher_overlay_source(
    *,
    blind_manifest: Path,
    blind_contract: Path,
    export_root: Path,
    dataset_root: Path,
    formal_windows_source: bool,
) -> RecipientMultiviewOverlayVerification:
    """Independently reopen the dedicated two-view producer for fixed2 only.

    The legacy four-view verifier above remains available for diagnostics, but
    is deliberately not an accepted input to the canonical fixed2
    materializer.  Formal overlay publication accepts only the formal Windows
    producer profile; the private analysis fixture accepts only the producer's
    analysis profile.
    """

    blind_manifest = _existing(
        blind_manifest, directory=False, description="fixed2 blind manifest"
    )
    blind_contract = _existing(
        blind_contract, directory=False, description="fixed2 blind contract"
    )
    export_root = _existing(
        export_root, directory=True, description="fixed2 teacher export root"
    )
    dataset_root = _existing(
        dataset_root, directory=True, description="fixed2 base dataset root"
    )
    export_contract = (
        verify_recipient_fixed2_teacher(export_root=export_root)
        if formal_windows_source
        else _verify_recipient_fixed2_teacher_analysis_test_only(
            export_root=export_root
        )
    )
    expected_kind = FIXED2_SOURCE_KIND if formal_windows_source else FIXED2_SOURCE_ANALYSIS_KIND
    expected_record_kind = (
        FIXED2_SOURCE_RECORD_KIND
        if formal_windows_source
        else FIXED2_SOURCE_ANALYSIS_RECORD_KIND
    )
    expected_contract_marker = (
        FIXED2_SOURCE_CONTRACT_NAME
        if formal_windows_source
        else FIXED2_SOURCE_ANALYSIS_CONTRACT_NAME
    )
    expected_authority = (
        FIXED2_SOURCE_PUBLICATION_AUTHORITY
        if formal_windows_source
        else FIXED2_SOURCE_ANALYSIS_PUBLICATION_AUTHORITY
    )
    for key, expected in (
        ("schema_version", SCHEMA_VERSION),
        ("kind", expected_kind),
        ("record_kind", expected_record_kind),
        ("publication_authority", expected_authority),
        (
            "hard_attestation_scheme",
            FIXED2_SOURCE_HARD_ATTESTATION_SCHEME
            if formal_windows_source
            else None,
        ),
        (
            "public_verification_requires_hard_attestation",
            formal_windows_source,
        ),
        ("optimizer_input_ready", False),
        ("records_role", "recipient_fixed2_overlay_source_only"),
        ("optimizer_adapter_required", FIXED2_SOURCE_ADAPTER_MARKER),
        ("view_order", list(FIXED2_SOURCE_VIEWS)),
        ("train_manifest", FIXED2_SOURCE_MANIFEST_NAME),
        ("commit_marker", expected_contract_marker),
        ("publication_complete", True),
        ("production_route_authorized", False),
        ("held_out_target_values_used", False),
        ("held_out_target_values_validated", False),
        ("held_out_target_values_emitted", False),
    ):
        _expect(
            export_contract.get(key),
            expected,
            description=f"fixed2 teacher contract {key}",
        )
    bound_manifest = _absolute_file(
        export_contract.get("source_manifest"),
        description="fixed2 teacher source blind manifest",
    )
    _samefile(
        bound_manifest,
        blind_manifest,
        description="fixed2 teacher source blind manifest",
    )
    _expect(
        export_contract.get("source_manifest_sha256"),
        _sha256(blind_manifest),
        description="fixed2 teacher source blind manifest SHA-256",
    )
    bound_root = _existing(
        Path(str(export_contract.get("source_dataset_root"))),
        directory=True,
        description="fixed2 teacher source dataset root",
    )
    _samefile(bound_root, dataset_root, description="fixed2 teacher dataset root")
    source_dataset_contract = _absolute_file(
        export_contract.get("source_dataset_contract"),
        description="fixed2 teacher source dataset contract",
    )
    source_contract = _verify_source_dataset_contract(
        source_dataset_contract,
        expected_kind=export_contract.get("source_dataset_kind"),
        expected_dataset_root=dataset_root,
    )
    source_dataset_contract_semantic_sha256 = (
        _source_dataset_contract_semantic_sha256(source_contract)
    )
    recipients, split_counts, recipient_counts, base_bindings = _blind_recipient_rows(
        blind_manifest=blind_manifest,
        dataset_root=dataset_root,
    )
    _blind, full_manifest = _verify_blind_contract(
        blind_manifest=blind_manifest,
        blind_contract=blind_contract,
        split_counts=split_counts,
    )
    source_counts = _mapping(
        export_contract.get("source_manifest_split_counts"),
        description="fixed2 teacher source split counts",
    )
    recipient_source_counts = _mapping(
        export_contract.get("source_split_counts"),
        description="fixed2 teacher recipient split counts",
    )
    for split in ("train", "val"):
        _expect(
            source_counts.get(split),
            int(split_counts[split]),
            description=f"fixed2 teacher {split} source count",
        )
        _expect(
            recipient_source_counts.get(split),
            int(recipient_counts[split]),
            description=f"fixed2 teacher {split} recipient count",
        )
    for split in ("test", "formal"):
        _expect(
            source_counts.get(split),
            0,
            description=f"fixed2 teacher excluded {split} source count",
        )
        _expect(
            recipient_source_counts.get(split),
            0,
            description=f"fixed2 teacher excluded {split} recipient count",
        )
    _expect(
        export_contract.get("source_train_recipient_records"),
        len(recipients),
        description="fixed2 teacher train recipient count",
    )
    _expect(
        export_contract.get("output_records"),
        len(recipients) * len(FIXED2_SOURCE_VIEWS),
        description="fixed2 teacher output count",
    )
    view_counts = _mapping(
        export_contract.get("view_counts"),
        description="fixed2 teacher view counts",
    )
    for view in FIXED2_SOURCE_VIEWS:
        _expect(
            view_counts.get(view),
            len(recipients),
            description=f"fixed2 teacher {view} count",
        )
    export_manifest = _existing(
        export_root / FIXED2_SOURCE_MANIFEST_NAME,
        directory=False,
        description="fixed2 teacher manifest",
    )
    rows = _strict_jsonl(export_manifest)
    # This is a second, consumer-owned decoded-pixel -> blind-owner closure.
    # It independently rebuilds the complete v2 reuse class ABI and then
    # requires exact equality with the producer declaration.
    (
        attachments,
        overlay_bindings,
        closure_ids,
        selected_view_hash_closure,
    ) = _verify_fixed2_overlay_rows(
        rows=rows,
        export_root=export_root,
        blind_sha256=_sha256(blind_manifest),
        recipients=recipients,
        expected_record_kind=expected_record_kind,
        heldout_crop_hashes=_fixed2_declared_heldout_crop_hashes(
            blind_manifest
        ),
    )
    producer_reuse_closure = export_contract.get("selected_view_hash_closure")
    if not _exact_json_value(
        producer_reuse_closure,
        selected_view_hash_closure,
    ):
        raise ValueError(
            "fixed2 consumer selected-view reuse closure differs from producer"
        )
    bindings = {
        "blind_manifest": _binding(blind_manifest),
        "blind_contract": _binding(blind_contract),
        "full_manifest": _binding(full_manifest),
        "export_contract": _binding(export_root / expected_contract_marker),
        "export_manifest": _binding(export_manifest),
        "source_dataset_contract": _binding(source_dataset_contract),
    }
    policy: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "kind": CONSUMER_KIND,
        "analysis_only": True,
        "production_route_authorized": False,
        "producer_kind": expected_kind,
        "producer_publication_authority": expected_authority,
        "producer_optimizer_input_ready": False,
        "consumer_optimizer_input_ready": True,
        "producer_adapter_marker": FIXED2_SOURCE_ADAPTER_MARKER,
        "optimizer_supervision_splits": ["train"],
        "validation_records_modified": False,
        "train_records_expanded": False,
        "recipient_value_left_trim": 0.0,
        "bindings": bindings,
        "dataset_root": str(dataset_root),
        "source_dataset_kind": source_contract["kind"],
        "source_dataset_contract_semantic_sha256": (
            source_dataset_contract_semantic_sha256
        ),
        "producer_subject_id": export_contract["producer_subject_id"],
        "producer_manifest_semantic_sha256": export_contract[
            "source_manifest_semantic_sha256"
        ],
        "selected_view_hash_closure": selected_view_hash_closure,
        "selected_view_hash_closure_sha256": _canonical_sha256(
            selected_view_hash_closure
        ),
        "counts": {
            "blind_train_records": int(split_counts["train"]),
            "blind_val_records": int(split_counts["val"]),
            "blind_train_recipient_records": len(recipients),
            "overlay_records": len(rows),
            "views_per_train_recipient": len(FIXED2_SOURCE_VIEWS),
            "attached_train_records": len(attachments),
            "attached_val_records": 0,
        },
        "view_order": list(FIXED2_SOURCE_VIEWS),
        "view_counts": {view: len(recipients) for view in FIXED2_SOURCE_VIEWS},
        "base_recipient_bindings_sha256": _canonical_sha256(base_bindings),
        "overlay_artifact_bindings_sha256": _canonical_sha256(overlay_bindings),
        "group_closure_set_sha256": _canonical_sha256(sorted(closure_ids)),
        "post_training_full_reverification_required": True,
    }
    return RecipientMultiviewOverlayVerification(
        policy=policy,
        attachments=attachments,
    )


def attach_recipient_multiview_overlay(
    records: Sequence[Mapping[str, object]],
    *,
    verification: RecipientMultiviewOverlayVerification,
) -> list[dict[str, object]]:
    """Attach four-view metadata without adding rows or touching validation."""

    expected = set(verification.attachments)
    observed: set[str] = set()
    attached: list[dict[str, object]] = []
    for record in records:
        record_id = record.get("id")
        split = record.get("split")
        slots = record.get("slots")
        recipient = slots.get("recipient_field") if isinstance(slots, Mapping) else None
        if split != "train" or not isinstance(recipient, Mapping):
            # Preserve validation object identity as an explicit no-mutation guarantee.
            attached.append(record if isinstance(record, dict) else dict(record))
            continue
        if not isinstance(record_id, str) or record_id not in verification.attachments:
            raise ValueError("loaded train recipient records do not match the verified overlay")
        attachment = verification.attachments[record_id]
        _expect(record.get("group_id"), attachment["group_id"], description="loaded overlay group")
        observed.add(record_id)
        updated = dict(record)
        updated[ATTACHMENT_KEY] = {
            **attachment,
            "selector": dict(verification.policy["selector"]),
        }
        attached.append(updated)
    if observed != expected:
        raise ValueError("verified overlay does not cover exactly the loaded train recipient records")
    if len(attached) != len(records):
        raise AssertionError("recipient multiview overlay expanded the in-memory receipt list")
    return attached


def select_recipient_multiview_name(*, seed: int, source_record_id: str, epoch: int) -> str:
    """Return the fixed four-cycle view for ``(seed, source id, epoch)``."""

    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("recipient multiview selector seed is invalid")
    if not isinstance(source_record_id, str) or not source_record_id:
        raise ValueError("recipient multiview source_record_id is invalid")
    if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 1:
        raise ValueError("recipient multiview epoch must start at one")
    digest = hashlib.sha256(f"{seed}\0{source_record_id}".encode("utf-8")).digest()
    offset = int.from_bytes(digest[:8], byteorder="little", signed=False) % len(VIEWS)
    return VIEWS[(offset + epoch - 1) % len(VIEWS)]


def selected_recipient_multiview_path(
    record: Mapping[str, object], *, epoch: int
) -> Path | None:
    """Resolve the selected attached view, or ``None`` for an ordinary record."""

    raw = record.get(ATTACHMENT_KEY)
    if raw is None:
        return None
    attachment = _mapping(raw, description="recipient multiview attachment")
    selector = _mapping(attachment.get("selector"), description="recipient multiview selector")
    _expect(selector.get("mode"), SELECTOR_MODE, description="recipient multiview selector mode")
    seed = selector.get("seed")
    source_id = attachment.get("source_record_id")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("recipient multiview attachment seed is invalid")
    if not isinstance(source_id, str):
        raise ValueError("recipient multiview attachment source id is invalid")
    view = select_recipient_multiview_name(seed=seed, source_record_id=source_id, epoch=epoch)
    views = _mapping(attachment.get("views"), description="recipient multiview paths")
    raw_path = views.get(view)
    view_binding = _mapping(raw_path, description=f"recipient multiview {view} binding")
    raw_path = view_binding.get("path")
    if not isinstance(raw_path, str) or not raw_path:
        raise ValueError(f"recipient multiview attachment has no {view} path")
    return Path(raw_path)


def _paths_overlap(left: Path, right: Path) -> bool:
    left = left.resolve()
    right = right.resolve()
    return left == right or left in right.parents or right in left.parents


def _relative_to_common(path: Path, common_root: Path) -> str:
    try:
        relative = path.relative_to(common_root)
    except ValueError:
        raise ValueError(f"composite artifact {path} is outside the common dataset root") from None
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"composite artifact {path} is not safely relative")
    return relative.as_posix()


def _fixed2_bound_rank_sha256(
    source_id: str, *, blind_manifest_sha256: str
) -> str:
    return hashlib.sha256(
        b"\x00".join(
            (
                FIXED2_SELECTOR_DOMAIN.encode("ascii"),
                blind_manifest_sha256.encode("ascii"),
                source_id.encode("utf-8"),
            )
        )
    ).hexdigest()


def _fixed2_selection(
    attachments: Mapping[str, Mapping[str, object]],
    *,
    reuse_closure: Mapping[str, object],
    blind_manifest_sha256: str,
) -> tuple[
    dict[str, str],
    dict[str, dict[str, object]],
    dict[str, object],
]:
    blind_manifest_sha256 = _require_sha(
        blind_manifest_sha256,
        description="fixed2 selector blind manifest",
    )
    source_ids = set(attachments)
    if not source_ids or any(
        not isinstance(source_id, str) or not source_id
        for source_id in source_ids
    ):
        raise ValueError("fixed2 selector source identities are invalid")
    raw_classes = reuse_closure.get("fixed_value_reuse_classes")
    if not isinstance(raw_classes, Sequence) or isinstance(
        raw_classes, (str, bytes)
    ):
        raise ValueError("fixed2 selector reuse classes must be a list")
    declared_count = reuse_closure.get("fixed_value_reuse_class_count")
    if (
        isinstance(declared_count, bool)
        or not isinstance(declared_count, int)
        or declared_count != len(raw_classes)
    ):
        raise ValueError("fixed2 selector reuse class count changed")

    rank_sha256 = {
        source_id: _fixed2_bound_rank_sha256(
            source_id,
            blind_manifest_sha256=blind_manifest_sha256,
        )
        for source_id in source_ids
    }
    selected: dict[str, str] = {}
    assignment_rows: list[dict[str, object]] = []
    pair_assignment_rows: list[dict[str, object]] = []
    paired_source_ids: set[str] = set()
    for raw_class in raw_classes:
        reuse_class = _mapping(
            raw_class,
            description="fixed2 selector reuse class",
        )
        reuse_class_id = _require_sha(
            reuse_class.get("reuse_class_id"),
            description="fixed2 selector reuse class id",
        )
        class_material = {
            key: value
            for key, value in reuse_class.items()
            if key != "reuse_class_id"
        }
        if _canonical_sha256(class_material) != reuse_class_id:
            raise ValueError("fixed2 selector reuse class id changed")
        if (
            reuse_class.get("view") != "fixed_value"
            or reuse_class.get("split") != "train"
        ):
            raise ValueError("fixed2 selector reuse class profile changed")
        class_pixel_sha256 = _require_sha(
            reuse_class.get("fixed_value_pixel_sha256"),
            description="fixed2 selector reuse fixed pixels",
        )
        class_target_sha256 = _require_sha(
            reuse_class.get("target_sha256"),
            description="fixed2 selector reuse target",
        )
        raw_owners = reuse_class.get("owners")
        if (
            not isinstance(raw_owners, Sequence)
            or isinstance(raw_owners, (str, bytes))
            or len(raw_owners) != 2
        ):
            raise ValueError("fixed2 selector reuse class must have two owners")
        members: list[str] = []
        for raw_owner in raw_owners:
            owner = _mapping(
                raw_owner,
                description=f"fixed2 selector reuse owner {reuse_class_id}",
            )
            source_id = owner.get("source_record_id")
            if (
                not isinstance(source_id, str)
                or source_id not in source_ids
                or source_id in paired_source_ids
            ):
                raise ValueError(
                    "fixed2 selector reuse class membership overlaps or is unknown"
                )
            attachment = _mapping(
                attachments[source_id],
                description=f"fixed2 selector attachment {source_id}",
            )
            views = _mapping(
                attachment.get("views"),
                description=f"fixed2 selector views {source_id}",
            )
            standard = _mapping(
                views.get("standard"),
                description=f"fixed2 selector standard view {source_id}",
            )
            fixed_value = _mapping(
                views.get("fixed_value"),
                description=f"fixed2 selector fixed view {source_id}",
            )
            for actual, expected, description in (
                (
                    attachment.get("group_id"),
                    owner.get("group_id"),
                    "group",
                ),
                (
                    attachment.get("target_sha256"),
                    class_target_sha256,
                    "target",
                ),
                (
                    standard.get("pixel_sha256"),
                    owner.get("standard_pixel_sha256"),
                    "standard pixels",
                ),
                (
                    fixed_value.get("pixel_sha256"),
                    class_pixel_sha256,
                    "fixed pixels",
                ),
            ):
                if actual != expected:
                    raise ValueError(
                        f"fixed2 selector reuse owner {source_id} {description} changed"
                    )
            paired_source_ids.add(source_id)
            members.append(source_id)
        ordered_members = sorted(
            members,
            key=lambda source_id: (rank_sha256[source_id], source_id),
        )
        pair_views = ("standard", "fixed_value")
        pair_roles = ("reuse_pair_standard", "reuse_pair_fixed_value")
        for pair_rank, (source_id, view, role) in enumerate(
            zip(ordered_members, pair_views, pair_roles)
        ):
            selected[source_id] = view
            assignment_rows.append(
                {
                    "source_record_id": source_id,
                    "partition": "reuse_pair",
                    "partition_rank": pair_rank,
                    "bound_rank_sha256": rank_sha256[source_id],
                    "reuse_class_id": reuse_class_id,
                    "selection_role": role,
                    "selected_view": view,
                }
            )
        pair_assignment_rows.append(
            {
                "reuse_class_id": reuse_class_id,
                "standard_source_record_id": ordered_members[0],
                "fixed_value_source_record_id": ordered_members[1],
                "fixed_value_pixel_sha256": class_pixel_sha256,
                "target_sha256": class_target_sha256,
            }
        )

    singleton_ids = sorted(
        source_ids - paired_source_ids,
        key=lambda source_id: (rank_sha256[source_id], source_id),
    )
    for singleton_rank, source_id in enumerate(singleton_ids):
        view = FIXED2_VIEWS[singleton_rank % len(FIXED2_VIEWS)]
        selected[source_id] = view
        assignment_rows.append(
            {
                "source_record_id": source_id,
                "partition": "singleton",
                "partition_rank": singleton_rank,
                "bound_rank_sha256": rank_sha256[source_id],
                "reuse_class_id": None,
                "selection_role": "singleton_rank_parity",
                "selected_view": view,
            }
        )
    if set(selected) != source_ids:
        raise AssertionError("fixed2 selector assignment coverage changed")
    assignment_rows.sort(key=lambda item: str(item["source_record_id"]))
    pair_assignment_rows.sort(key=lambda item: str(item["reuse_class_id"]))
    selector_assignment_sha256 = _canonical_sha256(assignment_rows)
    assignments = {
        str(row["source_record_id"]): {
            "reuse_class_id": row["reuse_class_id"],
            "selection_role": row["selection_role"],
            "selector_assignment_sha256": selector_assignment_sha256,
        }
        for row in assignment_rows
    }
    return (
        selected,
        assignments,
        {
            "reuse_class_count": len(raw_classes),
            "reuse_class_record_count": len(paired_source_ids),
            "singleton_record_count": len(singleton_ids),
            "reuse_pair_assignment_rule": (
                "ascending_bound_rank_then_standard_fixed_value_v1"
            ),
            "singleton_assignment_rule": (
                "ascending_bound_rank_then_standard_fixed_value_parity_v1"
            ),
            "reuse_class_assignment_sha256": _canonical_sha256(
                pair_assignment_rows
            ),
            "selector_assignment_sha256": selector_assignment_sha256,
        },
    )


def _selected_pixel_target_closure(
    bindings: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    owners: dict[tuple[str, str], str] = {}
    for raw in bindings:
        binding = _mapping(raw, description="fixed2 selected pixel/target binding")
        record_id = binding.get("record_id")
        target = binding.get("target")
        if not isinstance(record_id, str) or not record_id:
            raise ValueError("fixed2 selected pixel/target record identity changed")
        if not isinstance(target, str) or not target:
            raise ValueError("fixed2 selected pixel/target value changed")
        pixel_sha256 = _require_sha(
            binding.get("pixel_sha256"),
            description=f"fixed2 selected pixels {record_id}",
        )
        target_sha256 = hashlib.sha256(target.encode("utf-8")).hexdigest()
        key = (pixel_sha256, target_sha256)
        prior = owners.setdefault(key, record_id)
        if prior != record_id:
            raise ValueError(
                "fixed2 selected (pixel,target) identity is duplicated: "
                f"prior={prior!r} current={record_id!r}"
            )
        rows.append(
            {
                "record_id": record_id,
                "pixel_sha256": pixel_sha256,
                "target_sha256": target_sha256,
            }
        )
    rows.sort(key=lambda item: str(item["record_id"]))
    return {
        "selected_pixel_target_unique": True,
        "selected_pixel_target_count": len(rows),
        "selected_pixel_target_closure_sha256": _canonical_sha256(rows),
    }


def _composite_rows(
    *,
    blind_records: Path,
    original_dataset_root: Path,
    composite_dataset_root: Path,
    verification: RecipientMultiviewOverlayVerification,
) -> tuple[
    list[dict[str, object]],
    dict[str, object],
    list[dict[str, object]],
    list[dict[str, object]],
]:
    raw_rows = _strict_jsonl(blind_records)
    verification_bindings = _mapping(
        verification.policy.get("bindings"),
        description="overlay selector bindings",
    )
    blind_binding = _mapping(
        verification_bindings.get("blind_manifest"),
        description="overlay selector blind manifest binding",
    )
    bound_blind_manifest_sha256 = _require_sha(
        blind_binding.get("sha256"),
        description="overlay selector blind manifest binding",
    )
    reuse_closure = _mapping(
        verification.policy.get("selected_view_hash_closure"),
        description="fixed2 selector reuse closure",
    )
    selected, assignment_by_source, assignment_evidence = _fixed2_selection(
        verification.attachments,
        reuse_closure=reuse_closure,
        blind_manifest_sha256=bound_blind_manifest_sha256,
    )
    selected_counts: Counter[str] = Counter()
    train_recipient_ids: set[str] = set()
    split_counts: Counter[str] = Counter()
    output: list[dict[str, object]] = []
    val_before: list[object] = []
    val_after: list[object] = []
    selected_composite_bindings: list[dict[str, object]] = []
    validation_pixel_bindings: list[dict[str, object]] = []
    for row_number, raw in enumerate(raw_rows, start=1):
        split = raw.get("split")
        if split not in {"train", "val"}:
            raise ValueError(f"{blind_records}:{row_number}: composite source is not blind")
        split = str(split)
        split_counts[split] += 1
        record_id = raw.get("id")
        if not isinstance(record_id, str) or not record_id:
            raise ValueError(f"{blind_records}:{row_number}: invalid record id")
        slots = raw.get("slots")
        if not isinstance(slots, Mapping):
            raise ValueError(f"{blind_records}:{row_number}: slots must be an object")
        rewritten_slots: dict[str, object] = {}
        semantic_before_slots: dict[str, object] = {}
        semantic_after_slots: dict[str, object] = {}
        for field, raw_slot in slots.items():
            field_name = str(field)
            if raw_slot is None:
                rewritten_slots[field_name] = None
                semantic_before_slots[field_name] = None
                semantic_after_slots[field_name] = None
                continue
            if not isinstance(raw_slot, Mapping):
                raise ValueError(f"{blind_records}:{row_number}: slot {field} is not an object")
            slot = dict(raw_slot)
            base_image = _relative_file(
                original_dataset_root,
                slot.get("image"),
                description=f"composite base image {record_id}/{field}",
            )
            if split == "val":
                declared_pixel_sha256 = _require_sha(
                    slot.get("crop_sha256"),
                    description=f"validation slot crop {record_id}/{field_name}",
                )
                base_identity = _file_identity(base_image)
                with Image.open(base_image) as opened:
                    decoded = np.asarray(opened.convert("RGB"))
                if (
                    decoded.ndim != 3
                    or decoded.shape[2] != 3
                    or min(decoded.shape[:2]) <= 0
                ):
                    raise ValueError(
                        f"validation slot image {record_id}/{field_name} is empty"
                    )
                actual_pixel_sha256 = _crop_digest(decoded)
                if actual_pixel_sha256 != declared_pixel_sha256:
                    raise ValueError(
                        f"validation slot image {record_id}/{field_name} pixel hash changed"
                    )
                if not _same_file_identity(base_image, base_identity):
                    raise ValueError(
                        f"validation slot image {record_id}/{field_name} changed while decoded"
                    )
                validation_pixel_bindings.append(
                    {
                        "record_id": record_id,
                        "field": field_name,
                        "pixel_sha256": actual_pixel_sha256,
                        "file_sha256": base_identity[-1],
                        "size_bytes": base_identity[3],
                        "width": int(decoded.shape[1]),
                        "height": int(decoded.shape[0]),
                    }
                )
            semantic_before = {key: value for key, value in slot.items() if key != "image"}
            if split == "train" and field == "recipient_field":
                attachment = verification.attachments.get(record_id)
                if attachment is None:
                    raise ValueError(f"composite train recipient {record_id!r} has no verified overlay")
                selected_view = selected[record_id]
                views = _mapping(attachment.get("views"), description="verified overlay views")
                view_binding = _mapping(
                    views.get(selected_view), description=f"verified {selected_view} view"
                )
                _expect(
                    raw.get("group_id"),
                    attachment.get("group_id"),
                    description=f"selected composite group {record_id}",
                )
                _expect(
                    slot.get("text"),
                    attachment.get("target"),
                    description=f"selected composite target {record_id}",
                )
                selected_path = _existing(
                    Path(str(view_binding.get("path"))),
                    directory=False,
                    description=f"selected overlay image {record_id}/{selected_view}",
                )
                slot["image"] = _relative_to_common(selected_path, composite_dataset_root)
                slot["crop_sha256"] = view_binding["pixel_sha256"]
                selected_counts[selected_view] += 1
                train_recipient_ids.add(record_id)
                assignment = _mapping(
                    assignment_by_source.get(record_id),
                    description=f"fixed2 selector assignment {record_id}",
                )
                selected_composite_bindings.append(
                    {
                        "record_id": record_id,
                        "group_id": raw["group_id"],
                        "split": "train",
                        "target": slot["text"],
                        "view": selected_view,
                        "pixel_sha256": view_binding["pixel_sha256"],
                        "file_sha256": view_binding["file_sha256"],
                        "reuse_class_id": assignment["reuse_class_id"],
                        "selection_role": assignment["selection_role"],
                        "selector_assignment_sha256": assignment[
                            "selector_assignment_sha256"
                        ],
                    }
                )
            else:
                slot["image"] = _relative_to_common(base_image, composite_dataset_root)
            rewritten_slots[field_name] = slot
            semantic_after = {key: value for key, value in slot.items() if key != "image"}
            semantic_before_slots[field_name] = semantic_before
            semantic_after_slots[field_name] = semantic_after
        rewritten = dict(raw)
        rewritten["slots"] = rewritten_slots
        output.append(rewritten)
        if split == "val":
            val_before.append({**raw, "slots": semantic_before_slots})
            val_after.append({**rewritten, "slots": semantic_after_slots})
    if train_recipient_ids != set(verification.attachments):
        raise ValueError("composite does not replace exactly every blind train recipient")
    if val_before != val_after:
        raise AssertionError("composite changed validation content beyond path rebasing")
    if sum(selected_counts.values()) != len(verification.attachments):
        raise AssertionError("fixed2 selector changed the train multiplier")
    if abs(selected_counts[FIXED2_VIEWS[0]] - selected_counts[FIXED2_VIEWS[1]]) > 1:
        raise AssertionError("fixed2 v2 selector did not balance the selected views")
    selected_composite_bindings.sort(key=lambda item: str(item["record_id"]))
    selected_pixel_target_evidence = _selected_pixel_target_closure(
        selected_composite_bindings
    )
    validation_pixel_bindings.sort(
        key=lambda item: (str(item["record_id"]), str(item["field"]))
    )
    evidence: dict[str, object] = {
        "selector_mode": FIXED2_SELECTOR_MODE,
        "selector_domain": FIXED2_SELECTOR_DOMAIN,
        "selected_views": list(FIXED2_VIEWS),
        "verified_reuse_policy": reuse_closure["policy"],
        "verified_reuse_class_closure_sha256": reuse_closure[
            "fixed_value_reuse_class_closure_sha256"
        ],
        "selector_input": (
            "sha256(domain_nul_bound_blind_manifest_sha256_nul_utf8_source_record_id);"
            "verified_reuse_pairs_sorted_independently_then_one_standard_one_fixed;"
            "singletons_sorted_independently_then_rank_parity"
        ),
        "bound_blind_manifest_sha256": bound_blind_manifest_sha256,
        "even_rank_view": FIXED2_VIEWS[0],
        "odd_rank_view": FIXED2_VIEWS[1],
        "train_multiplier": 1,
        "val_unchanged": True,
        "val_path_rebased_only": True,
        "split_counts": {"train": int(split_counts["train"]), "val": int(split_counts["val"])},
        "train_recipient_records": len(train_recipient_ids),
        "selected_view_counts": {
            view: int(selected_counts[view]) for view in FIXED2_VIEWS
        },
        **assignment_evidence,
        **selected_pixel_target_evidence,
    }
    return output, evidence, selected_composite_bindings, validation_pixel_bindings


def _jsonl_bytes(rows: Sequence[Mapping[str, object]]) -> bytes:
    return "".join(
        json.dumps(
            dict(row),
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        )
        + "\n"
        for row in rows
    ).encode("utf-8")


def _json_bytes(value: Mapping[str, object]) -> bytes:
    return (
        json.dumps(
            dict(value),
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            indent=2,
        )
        + "\n"
    ).encode("utf-8")


def _require_fixed2_publication_profile(
    *,
    contract_kind: str,
    publication_authority: str,
    consumer_optimizer_input_ready: bool,
) -> bool:
    profile = (
        contract_kind,
        publication_authority,
        consumer_optimizer_input_ready,
    )
    formal_profile = (
        FIXED2_CONTRACT_KIND,
        FIXED2_PUBLICATION_AUTHORITY,
        True,
    )
    if profile == formal_profile:
        if not _formal_windows_publication_available():
            raise OSError(
                errno.ENOTSUP,
                "formal fixed2 publication profiles require Windows atomic "
                "parent-handle-relative authority",
            )
        return True
    if profile == (
        FIXED2_ANALYSIS_CONTRACT_KIND,
        FIXED2_ANALYSIS_PUBLICATION_AUTHORITY,
        False,
    ):
        return False
    raise ValueError("unsupported fixed2 publication profile")


def _composite_dataset_contract(
    *,
    contract_kind: str,
    publication_authority: str,
    consumer_optimizer_input_ready: bool,
    source_contract: Mapping[str, Any],
    source_contract_path: Path,
    composite_records_path: Path,
    composite_dataset_root: Path,
    rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    _require_fixed2_publication_profile(
        contract_kind=contract_kind,
        publication_authority=publication_authority,
        consumer_optimizer_input_ready=consumer_optimizer_input_ready,
    )
    required = {
        "schema_version",
        "kind",
        "slot_order",
        "status_classes",
        "recipient_charset_source",
        "recipient_quality_policy",
    }
    missing = sorted(required - set(source_contract))
    if missing:
        raise ValueError("source dataset contract lacks composite ABI fields: " + ",".join(missing))
    split_counts = Counter(str(row.get("split")) for row in rows)
    payload: dict[str, object] = {
        "schema_version": source_contract["schema_version"],
        "kind": source_contract["kind"],
        "source_records": str(composite_records_path),
        "dataset_root": str(composite_dataset_root),
        "slot_order": source_contract["slot_order"],
        "status_classes": source_contract["status_classes"],
        "records": len(rows),
        "by_split": {
            "train": int(split_counts["train"]),
            "val": int(split_counts["val"]),
            "test": 0,
        },
        "recipient_charset_source": source_contract["recipient_charset_source"],
        "recipient_quality_policy": source_contract["recipient_quality_policy"],
        "fixed2_overlay": {
            "kind": contract_kind,
            "publication_authority": publication_authority,
            "analysis_only": True,
            "production_route_authorized": False,
            "consumer_optimizer_input_ready": consumer_optimizer_input_ready,
            "selected_views": list(FIXED2_VIEWS),
            "selector_mode": FIXED2_SELECTOR_MODE,
            "train_multiplier": 1,
            "val_unchanged": True,
            "source_dataset_contract": str(source_contract_path),
            "source_dataset_contract_sha256": _sha256(source_contract_path),
        },
    }
    for key in (
        "architecture",
        "recipient_target",
        "recipient_charset",
        "recipient_charset_sha256",
        "recipient_oov_by_split",
        "status_text_target",
        "status_text_charset",
        "status_text_charset_sha256",
        "status_text_charset_source",
        "status_text_source_counts",
        "status_text_missing_reasons",
        "status_text_oov_by_split",
    ):
        if key in source_contract:
            payload[key] = source_contract[key]
    return payload


def _artifact_for_stage(*, stage_path: Path, final_path: Path) -> dict[str, object]:
    return {
        "path": str(final_path),
        "sha256": _sha256(stage_path),
        "size_bytes": stage_path.stat().st_size,
    }


def _artifact_from_identity(
    *, identity: FileIdentity, final_path: Path
) -> dict[str, object]:
    return {
        "path": str(final_path),
        "sha256": identity[-1],
        "size_bytes": identity[3],
    }


def _binding_identity(binding: Mapping[str, object]) -> dict[str, object]:
    return {
        "sha256": binding.get("sha256"),
        "size_bytes": binding.get("size_bytes"),
    }


def _normalized_selected_composite_bindings(
    bindings: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    expected_keys = {
        "record_id",
        "group_id",
        "split",
        "target",
        "view",
        "pixel_sha256",
        "file_sha256",
        "reuse_class_id",
        "selection_role",
        "selector_assignment_sha256",
    }
    normalized: list[dict[str, object]] = []
    identities: set[str] = set()
    for index, raw in enumerate(bindings):
        binding = _mapping(raw, description=f"selected composite binding {index}")
        if set(binding) != expected_keys:
            raise ValueError("selected composite binding fields changed")
        record_id = binding.get("record_id")
        group_id = binding.get("group_id")
        target = binding.get("target")
        view = binding.get("view")
        if not isinstance(record_id, str) or not record_id or record_id in identities:
            raise ValueError("selected composite record identity is invalid or duplicated")
        if not isinstance(group_id, str) or not group_id:
            raise ValueError("selected composite group identity is invalid")
        if not isinstance(target, str) or not target:
            raise ValueError("selected composite target is invalid")
        if binding.get("split") != "train" or view not in FIXED2_VIEWS:
            raise ValueError("selected composite split/view is invalid")
        reuse_class_id = binding.get("reuse_class_id")
        selection_role = binding.get("selection_role")
        if selection_role == "reuse_pair_standard":
            if view != "standard" or reuse_class_id is None:
                raise ValueError("selected composite reuse-pair standard role changed")
        elif selection_role == "reuse_pair_fixed_value":
            if view != "fixed_value" or reuse_class_id is None:
                raise ValueError("selected composite reuse-pair fixed role changed")
        elif selection_role == "singleton_rank_parity":
            if reuse_class_id is not None:
                raise ValueError("selected composite singleton reuse class changed")
        else:
            raise ValueError("selected composite selection role changed")
        normalized_reuse_class_id = (
            _require_sha(
                reuse_class_id,
                description=f"selected composite reuse class {record_id}",
            )
            if reuse_class_id is not None
            else None
        )
        identities.add(record_id)
        normalized.append(
            {
                "record_id": record_id,
                "group_id": group_id,
                "split": "train",
                "target": target,
                "view": view,
                "pixel_sha256": _require_sha(
                    binding.get("pixel_sha256"),
                    description=f"selected composite pixel {record_id}",
                ),
                "file_sha256": _require_sha(
                    binding.get("file_sha256"),
                    description=f"selected composite file {record_id}",
                ),
                "reuse_class_id": normalized_reuse_class_id,
                "selection_role": selection_role,
                "selector_assignment_sha256": _require_sha(
                    binding.get("selector_assignment_sha256"),
                    description=f"selected composite assignment {record_id}",
                ),
            }
        )
    normalized.sort(key=lambda item: str(item["record_id"]))
    return normalized


def _normalized_validation_pixel_bindings(
    bindings: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    expected_keys = {
        "record_id",
        "field",
        "pixel_sha256",
        "file_sha256",
        "size_bytes",
        "width",
        "height",
    }
    normalized: list[dict[str, object]] = []
    identities: set[tuple[str, str]] = set()
    for index, raw in enumerate(bindings):
        binding = _mapping(raw, description=f"validation pixel binding {index}")
        if set(binding) != expected_keys:
            raise ValueError("validation pixel binding fields changed")
        record_id = binding.get("record_id")
        field = binding.get("field")
        if not isinstance(record_id, str) or not record_id:
            raise ValueError("validation pixel record identity is invalid")
        if not isinstance(field, str) or not field:
            raise ValueError("validation pixel field identity is invalid")
        identity = (record_id, field)
        if identity in identities:
            raise ValueError("validation pixel identity is duplicated")
        identities.add(identity)
        numeric: dict[str, int] = {}
        for name in ("size_bytes", "width", "height"):
            value = binding.get(name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"validation pixel {name} is invalid")
            numeric[name] = value
        normalized.append(
            {
                "record_id": record_id,
                "field": field,
                "pixel_sha256": _require_sha(
                    binding.get("pixel_sha256"),
                    description=f"validation pixel {record_id}/{field}",
                ),
                "file_sha256": _require_sha(
                    binding.get("file_sha256"),
                    description=f"validation file {record_id}/{field}",
                ),
                **numeric,
            }
        )
    normalized.sort(key=lambda item: (str(item["record_id"]), str(item["field"])))
    return normalized


def _identity_integer(value: object, *, description: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{description} must be a non-negative integer")
    return value


def _fixed2_publication_identity(
    *,
    directory_identity: DirectoryIdentity,
    pre_marker_file_identities: Mapping[str, FileIdentity],
) -> dict[str, object]:
    required_files = {
        "unified_fields.train-val.fixed2.jsonl",
        "dataset.contract.json",
    }
    if set(pre_marker_file_identities) != required_files:
        raise ValueError("fixed2 publication identity file closure changed")
    volume_serial, file_index, directory_attributes = directory_identity
    files: dict[str, dict[str, object]] = {}
    for name in sorted(required_files):
        (
            device,
            inode,
            file_attributes,
            size_bytes,
            mtime_ns,
            sha256,
        ) = pre_marker_file_identities[name]
        files[name] = {
            "volume_serial_number": _identity_integer(
                device,
                description=f"fixed2 publication file {name} volume serial",
            ),
            "file_index": _identity_integer(
                inode,
                description=f"fixed2 publication file {name} index",
            ),
            "file_attributes": _identity_integer(
                file_attributes,
                description=f"fixed2 publication file {name} attributes",
            ),
            "size_bytes": _identity_integer(
                size_bytes,
                description=f"fixed2 publication file {name} size",
            ),
            "mtime_ns": _identity_integer(
                mtime_ns,
                description=f"fixed2 publication file {name} mtime",
            ),
            "sha256": _require_sha(
                sha256,
                description=f"fixed2 publication file {name}",
            ),
        }
    return {
        "scheme": "windows_native_directory_and_file_identity_v1",
        "directory": {
            "volume_serial_number": _identity_integer(
                volume_serial,
                description="fixed2 publication directory volume serial",
            ),
            "file_index": _identity_integer(
                file_index,
                description="fixed2 publication directory index",
            ),
            "file_attributes": _identity_integer(
                directory_attributes,
                description="fixed2 publication directory attributes",
            ),
        },
        "pre_marker_files": files,
    }


def _normalized_fixed2_publication_identity(
    value: object,
) -> dict[str, object]:
    identity = _mapping(value, description="fixed2 publication identity")
    if set(identity) != {"scheme", "directory", "pre_marker_files"}:
        raise ValueError("fixed2 publication identity key set changed")
    _expect(
        identity.get("scheme"),
        "windows_native_directory_and_file_identity_v1",
        description="fixed2 publication identity scheme",
    )
    directory = _mapping(
        identity.get("directory"),
        description="fixed2 publication directory identity",
    )
    if set(directory) != {
        "volume_serial_number",
        "file_index",
        "file_attributes",
    }:
        raise ValueError("fixed2 publication directory identity key set changed")
    normalized_directory = {
        name: _identity_integer(
            directory.get(name),
            description=f"fixed2 publication directory {name}",
        )
        for name in ("volume_serial_number", "file_index", "file_attributes")
    }
    raw_files = _mapping(
        identity.get("pre_marker_files"),
        description="fixed2 publication pre-marker files",
    )
    required_files = {
        "unified_fields.train-val.fixed2.jsonl",
        "dataset.contract.json",
    }
    if set(raw_files) != required_files:
        raise ValueError("fixed2 publication pre-marker file set changed")
    normalized_files: dict[str, dict[str, object]] = {}
    numeric_fields = (
        "volume_serial_number",
        "file_index",
        "file_attributes",
        "size_bytes",
        "mtime_ns",
    )
    for name in sorted(required_files):
        raw = _mapping(
            raw_files.get(name),
            description=f"fixed2 publication pre-marker file {name}",
        )
        if set(raw) != {*numeric_fields, "sha256"}:
            raise ValueError(f"fixed2 publication file identity key set changed: {name}")
        normalized_files[name] = {
            field: _identity_integer(
                raw.get(field),
                description=f"fixed2 publication file {name} {field}",
            )
            for field in numeric_fields
        }
        normalized_files[name]["sha256"] = _require_sha(
            raw.get("sha256"),
            description=f"fixed2 publication file {name}",
        )
    return {
        "scheme": "windows_native_directory_and_file_identity_v1",
        "directory": normalized_directory,
        "pre_marker_files": normalized_files,
    }


def _fixed2_contract_payload(
    *,
    contract_kind: str,
    publication_authority: str,
    consumer_optimizer_input_ready: bool,
    publication_identity: Mapping[str, object] | None,
    multiview_root: Path,
    original_dataset_root: Path,
    composite_records_path: Path,
    composite_dataset_contract_path: Path,
    composite_dataset_root: Path,
    producer_subject_id: str,
    producer_manifest_semantic_sha256: str,
    source_dataset_contract_semantic_sha256: str,
    selector_evidence: Mapping[str, object],
    selected_composite_bindings: Sequence[Mapping[str, object]],
    validation_pixel_bindings: Sequence[Mapping[str, object]],
    artifacts: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    formal_publication = _require_fixed2_publication_profile(
        contract_kind=contract_kind,
        publication_authority=publication_authority,
        consumer_optimizer_input_ready=consumer_optimizer_input_ready,
    )
    if formal_publication:
        normalized_publication_identity: dict[str, object] | None = (
            _normalized_fixed2_publication_identity(publication_identity)
        )
    else:
        if publication_identity is not None:
            raise ValueError("analysis fixtures must not claim a publication identity")
        normalized_publication_identity = None
    expected_artifact_names = _DATA_ARTIFACT_NAMES | frozenset(_CODE_ARTIFACT_FILES)
    if set(artifacts) != expected_artifact_names:
        raise ValueError("fixed2 contract artifact closure is incomplete")
    semantic_artifacts = {
        name: _binding_identity(artifacts[name])
        for name in sorted(_SUBJECT_SEMANTIC_ARTIFACT_NAMES)
    }
    code_artifacts = {
        name: _binding_identity(artifacts[name])
        for name in sorted(_CODE_ARTIFACT_FILES)
    }
    selected_bindings = _normalized_selected_composite_bindings(
        selected_composite_bindings
    )
    selector_assignment_sha256 = _require_sha(
        selector_evidence.get("selector_assignment_sha256"),
        description="fixed2 selector assignment closure",
    )
    if {
        binding["selector_assignment_sha256"]
        for binding in selected_bindings
    } != {selector_assignment_sha256}:
        raise ValueError(
            "selected composite bindings do not bind one selector assignment closure"
        )
    validation_bindings = _normalized_validation_pixel_bindings(
        validation_pixel_bindings
    )
    selected_composite_closure_sha256 = _canonical_sha256(selected_bindings)
    selected_composite_semantic = [
        {
            "record_id": binding["record_id"],
            "group_id": binding["group_id"],
            "split": binding["split"],
            "target_sha256": hashlib.sha256(
                str(binding["target"]).encode("utf-8")
            ).hexdigest(),
            "view": binding["view"],
            "pixel_sha256": binding["pixel_sha256"],
            "reuse_class_id": binding["reuse_class_id"],
            "selection_role": binding["selection_role"],
            "selector_assignment_sha256": binding[
                "selector_assignment_sha256"
            ],
        }
        for binding in selected_bindings
    ]
    selected_composite_semantic_closure_sha256 = _canonical_sha256(
        selected_composite_semantic
    )
    validation_pixel_semantic = [
        {
            "record_id": binding["record_id"],
            "field": binding["field"],
            "pixel_sha256": binding["pixel_sha256"],
            "width": binding["width"],
            "height": binding["height"],
        }
        for binding in validation_bindings
    ]
    validation_pixel_semantic_sha256 = _canonical_sha256(
        validation_pixel_semantic
    )
    producer_subject_id = _require_sha(
        producer_subject_id,
        description="fixed2 producer semantic subject",
    )
    producer_manifest_semantic_sha256 = _require_sha(
        producer_manifest_semantic_sha256,
        description="fixed2 producer manifest semantic closure",
    )
    source_dataset_contract_semantic_sha256 = _require_sha(
        source_dataset_contract_semantic_sha256,
        description="fixed2 source dataset semantic ABI closure",
    )
    # The raw blind-manifest digest remains sealed in selector evidence and
    # artifacts for integrity.  Subject identity binds the path-free selector
    # result instead, so relocating/re-publishing equal producer semantics
    # cannot mint another route or one-shot attempt.
    selector_subject = {
        key: value
        for key, value in selector_evidence.items()
        if key != "bound_blind_manifest_sha256"
    }
    subject_material = {
        "domain": FIXED2_SUBJECT_DOMAIN,
        "selected_views": list(FIXED2_VIEWS),
        "selector": selector_subject,
        "producer_subject_id": producer_subject_id,
        "producer_manifest_semantic_sha256": producer_manifest_semantic_sha256,
        "source_dataset_contract_semantic_sha256": (
            source_dataset_contract_semantic_sha256
        ),
        "semantic_artifacts": semantic_artifacts,
        "selected_composite_semantic_closure_sha256": (
            selected_composite_semantic_closure_sha256
        ),
        "validation_pixel_semantic_sha256": validation_pixel_semantic_sha256,
    }
    overlay_subject_id = _canonical_sha256(subject_material)
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": contract_kind,
        "publication_authority": publication_authority,
        "publication_identity": normalized_publication_identity,
        "analysis_only": True,
        "production_route_authorized": False,
        "test_opened": False,
        "onnx_exported": False,
        "producer_optimizer_input_ready": False,
        "consumer_optimizer_input_ready": consumer_optimizer_input_ready,
        "selected_views": list(FIXED2_VIEWS),
        "selector_mode": FIXED2_SELECTOR_MODE,
        "selector": dict(selector_evidence),
        "train_multiplier": 1,
        "val_unchanged": True,
        "multiview_root": str(multiview_root),
        "original_dataset_root": str(original_dataset_root),
        "composite_records": str(composite_records_path),
        "composite_records_path": str(composite_records_path),
        "composite_dataset_contract": str(composite_dataset_contract_path),
        "composite_dataset_root": str(composite_dataset_root),
        "overlay_subject_id": overlay_subject_id,
        "producer_subject_id": producer_subject_id,
        "producer_manifest_semantic_sha256": producer_manifest_semantic_sha256,
        "source_dataset_contract_semantic_sha256": (
            source_dataset_contract_semantic_sha256
        ),
        "selected_composite_bindings": selected_bindings,
        "selected_composite_closure_sha256": selected_composite_closure_sha256,
        "selected_composite_semantic_closure_sha256": (
            selected_composite_semantic_closure_sha256
        ),
        "validation_pixel_bindings": validation_bindings,
        "validation_pixel_semantic_sha256": validation_pixel_semantic_sha256,
        "validation_file_integrity_sha256": _canonical_sha256(validation_bindings),
        "validation_slot_count": len(validation_bindings),
        "test_physical_files_opened": 0,
        "semantic_artifact_names": sorted(_SUBJECT_SEMANTIC_ARTIFACT_NAMES),
        "code_artifact_names": sorted(_CODE_ARTIFACT_FILES),
        "code_closure_sha256": _canonical_sha256(code_artifacts),
        "artifacts": {name: dict(binding) for name, binding in artifacts.items()},
    }


def _materialize_fixed2_overlay_impl(
    *,
    formal_windows_publication: bool,
    multiview_root: Path,
    full_records: Path,
    blind_records: Path,
    blind_contract: Path,
    original_dataset_root: Path,
    output_root: Path,
) -> dict[str, object]:
    """Materialize one explicitly formal or analysis-only publication profile."""

    if formal_windows_publication:
        if not _formal_windows_publication_available():
            raise OSError(
                errno.ENOTSUP,
                "formal fixed2 overlay materialization requires Windows atomic "
                "parent-handle-relative stage creation",
                os.fspath(output_root),
            )
        contract_kind = FIXED2_CONTRACT_KIND
        publication_authority = FIXED2_PUBLICATION_AUTHORITY
        consumer_optimizer_input_ready = True
        contract_marker_name = FIXED2_CANONICAL_CONTRACT_NAME
    else:
        contract_kind = FIXED2_ANALYSIS_CONTRACT_KIND
        publication_authority = FIXED2_ANALYSIS_PUBLICATION_AUTHORITY
        consumer_optimizer_input_ready = False
        contract_marker_name = FIXED2_ANALYSIS_MARKER_NAME

    multiview_root = _existing(multiview_root, directory=True, description="multiview root")
    full_records = _existing(full_records, directory=False, description="full records")
    blind_records = _existing(blind_records, directory=False, description="blind records")
    blind_contract = _existing(blind_contract, directory=False, description="blind contract")
    original_dataset_root = _existing(
        original_dataset_root, directory=True, description="original dataset root"
    )
    output_raw = Path(os.path.abspath(os.fspath(output_root)))
    if os.path.lexists(output_raw):
        raise FileExistsError(f"refusing to reuse fixed2 overlay output: {output_raw}")
    parent = _existing(output_raw.parent, directory=True, description="fixed2 output parent")
    parent_identity = _directory_identity(parent)
    output_root = parent / output_raw.name
    for protected, description in (
        (multiview_root, "multiview root"),
        (original_dataset_root, "original dataset root"),
        (blind_records.parent, "blind evidence directory"),
    ):
        if _paths_overlap(output_root, protected):
            raise ValueError(f"fixed2 output overlaps {description}")
    common_raw = Path(os.path.commonpath((str(original_dataset_root), str(multiview_root))))
    composite_dataset_root = _existing(
        common_raw, directory=True, description="common composite dataset root"
    )
    if composite_dataset_root == Path(composite_dataset_root.anchor):
        raise ValueError("common composite dataset root must not be a filesystem root")
    verification = _verify_fixed2_teacher_overlay_source(
        blind_manifest=blind_records,
        blind_contract=blind_contract,
        export_root=multiview_root,
        dataset_root=original_dataset_root,
        formal_windows_source=formal_windows_publication,
    )
    bound_full = Path(
        str(_mapping(verification.policy["bindings"], description="overlay bindings")["full_manifest"]["path"])
    )
    _samefile(bound_full, full_records, description="fixed2 full records")
    (
        rows,
        selector_evidence,
        selected_composite_bindings,
        validation_pixel_bindings,
    ) = _composite_rows(
        blind_records=blind_records,
        original_dataset_root=original_dataset_root,
        composite_dataset_root=composite_dataset_root,
        verification=verification,
    )
    source_contract_binding = _mapping(
        _mapping(verification.policy["bindings"], description="overlay bindings").get(
            "source_dataset_contract"
        ),
        description="source dataset contract binding",
    )
    source_contract_path = _existing(
        Path(str(source_contract_binding["path"])),
        directory=False,
        description="source dataset contract",
    )
    source_contract = _strict_json(source_contract_path)
    code_artifact_paths = _code_artifact_paths()
    stage = parent / f".{output_root.name}.{hashlib.sha256(os.urandom(32)).hexdigest()[:24]}.tmp"
    parent_lease = _open_directory_lease(parent, expected=parent_identity)

    def require_parent(checkpoint: str) -> None:
        _require_output_parent_identity(
            parent,
            expected=parent_identity,
            checkpoint=checkpoint,
            stage=stage,
            output_root=output_root,
        )

    def before_anchored_use(checkpoint: str) -> None:
        _fixed2_publication_use_hook(
            checkpoint,
            parent=parent,
            stage=stage,
            output_root=output_root,
        )

    stage_lease: _DirectoryLease | None = None
    stage_identity: DirectoryIdentity | None = None
    stage_file_identities: dict[str, FileIdentity] = {}
    renamed_to_output = False
    try:
        retained_prefix = f".{output_root.name}."
        retained_entries = sorted(
            name
            for name in _anchored_directory_names(parent_lease)
            if name.startswith(retained_prefix)
        )
        if retained_entries:
            raise FileExistsError(
                "refusing a fixed2 publication while retained staging/failure "
                f"evidence exists: {retained_entries}"
            )
        require_parent("before_stage_creation")
        before_anchored_use("before_stage_creation")
        stage_lease = (
            create_anchored_stage_directory(parent_lease, name=stage.name)
            if formal_windows_publication
            else _create_stage_lease(parent_lease, stage=stage)
        )
        stage_identity = stage_lease.identity
        require_parent("after_stage_creation")
        stage_records = stage / "unified_fields.train-val.fixed2.jsonl"
        final_records = output_root / stage_records.name
        require_parent("before_composite_records_write")
        before_anchored_use("before_composite_records_write")
        stage_file_identities[stage_records.name] = _write_anchored_stage_file(
            stage_lease,
            name=stage_records.name,
            payload=_jsonl_bytes(rows),
        )
        require_parent("before_composite_records_snapshot")
        if _anchored_file_identity(stage_lease, stage_records.name) != (
            stage_file_identities[stage_records.name]
        ):
            raise ValueError("fixed2 composite records changed after anchored write")
        require_parent("after_composite_records_snapshot")
        stage_dataset_contract = stage / "dataset.contract.json"
        final_dataset_contract = output_root / stage_dataset_contract.name
        dataset_contract_payload = _composite_dataset_contract(
            contract_kind=contract_kind,
            publication_authority=publication_authority,
            consumer_optimizer_input_ready=consumer_optimizer_input_ready,
            source_contract=source_contract,
            source_contract_path=source_contract_path,
            composite_records_path=final_records,
            composite_dataset_root=composite_dataset_root,
            rows=rows,
        )
        require_parent("before_composite_dataset_contract_write")
        before_anchored_use("before_composite_dataset_contract_write")
        stage_file_identities[stage_dataset_contract.name] = _write_anchored_stage_file(
            stage_lease,
            name=stage_dataset_contract.name,
            payload=_json_bytes(dataset_contract_payload),
        )
        require_parent("before_composite_dataset_contract_snapshot")
        if _anchored_file_identity(stage_lease, stage_dataset_contract.name) != (
            stage_file_identities[stage_dataset_contract.name]
        ):
            raise ValueError("fixed2 dataset contract changed after anchored write")
        require_parent("after_composite_dataset_contract_snapshot")
        export_contract_path = multiview_root / (
            FIXED2_SOURCE_CONTRACT_NAME
            if formal_windows_publication
            else FIXED2_SOURCE_ANALYSIS_CONTRACT_NAME
        )
        export_manifest_path = multiview_root / "multiview_train.jsonl"
        artifacts: dict[str, dict[str, object]] = {
            "full_records": _binding(full_records),
            "blind_records": _binding(blind_records),
            "blind_contract": _binding(blind_contract),
            "multiview_export_contract": _binding(export_contract_path),
            "multiview_export_manifest": _binding(export_manifest_path),
            "source_dataset_contract": _binding(source_contract_path),
            "composite_records": _artifact_from_identity(
                identity=stage_file_identities[stage_records.name],
                final_path=final_records,
            ),
            "composite_dataset_contract": _artifact_from_identity(
                identity=stage_file_identities[stage_dataset_contract.name],
                final_path=final_dataset_contract,
            ),
        }
        artifacts.update(
            {name: _binding(path) for name, path in code_artifact_paths.items()}
        )
        if stage_identity is None:
            raise ValueError("fixed2 stage identity was not acquired")
        publication_identity = (
            _fixed2_publication_identity(
                directory_identity=stage_identity,
                pre_marker_file_identities=stage_file_identities,
            )
            if formal_windows_publication
            else None
        )
        require_parent("before_fixed2_contract_seal")
        payload = _fixed2_contract_payload(
            contract_kind=contract_kind,
            publication_authority=publication_authority,
            consumer_optimizer_input_ready=consumer_optimizer_input_ready,
            publication_identity=publication_identity,
            multiview_root=multiview_root,
            original_dataset_root=original_dataset_root,
            composite_records_path=final_records,
            composite_dataset_contract_path=final_dataset_contract,
            composite_dataset_root=composite_dataset_root,
            producer_subject_id=str(
                verification.policy["producer_subject_id"]
            ),
            producer_manifest_semantic_sha256=str(
                verification.policy["producer_manifest_semantic_sha256"]
            ),
            source_dataset_contract_semantic_sha256=str(
                verification.policy[
                    "source_dataset_contract_semantic_sha256"
                ]
            ),
            selector_evidence=selector_evidence,
            selected_composite_bindings=selected_composite_bindings,
            validation_pixel_bindings=validation_pixel_bindings,
            artifacts=artifacts,
        )
        sealed = {**payload, "integrity_sha256": _canonical_sha256(payload)}
        require_parent("after_fixed2_contract_seal")
        require_parent("before_stage_snapshot")
        _require_anchored_directory_snapshot(
            stage_lease,
            expected_directory=stage_identity,
            expected_files=stage_file_identities,
            description="fixed2 publication stage",
        )
        require_parent("after_stage_snapshot")
        require_parent("before_final_verify")
        result = _verify_fixed2_overlay_payload(
            sealed,
            expected_kind=contract_kind,
            expected_publication_authority=publication_authority,
            actual_publication_identity=publication_identity,
            declared_contract_directory=output_root,
            actual_composite_records=stage_records,
            actual_composite_dataset_contract=stage_dataset_contract,
            blind_records=blind_records,
            blind_contract=blind_contract,
            multiview_root=multiview_root,
            expected_full_records=full_records,
            original_dataset_root=original_dataset_root,
        )
        require_parent("after_final_verify")
        _require_anchored_directory_snapshot(
            stage_lease,
            expected_directory=stage_identity,
            expected_files=stage_file_identities,
            description="fixed2 precommit publication stage",
        )
        require_parent("immediately_before_rename")
        before_anchored_use("immediately_before_rename")
        _rename_directory_no_replace_anchored(
            parent_lease,
            stage_lease,
            source=stage,
            destination=output_root,
        )
        renamed_to_output = True
        stage_lease.path = output_root
        _require_directory_lease_identity(parent_lease)
        _require_directory_lease_identity(stage_lease)
        require_parent("immediately_after_rename")
        if not _same_anchored_directory_entry(
            parent_lease,
            name=output_root.name,
            expected=stage_identity,
        ):
            raise ValueError("fixed2 published output entry identity changed")
        _require_anchored_directory_snapshot(
            stage_lease,
            expected_directory=stage_identity,
            expected_files=stage_file_identities,
            description="fixed2 published output",
        )
        require_parent("after_published_output_snapshot")
        final_contract = output_root / contract_marker_name
        require_parent("before_fixed2_contract_commit")
        before_anchored_use("before_fixed2_contract_commit")
        stage_file_identities[final_contract.name] = _write_anchored_stage_file(
            stage_lease,
            name=final_contract.name,
            payload=_json_bytes(sealed),
        )
        _require_anchored_directory_snapshot(
            stage_lease,
            expected_directory=stage_identity,
            expected_files=stage_file_identities,
            description="fixed2 committed output",
        )
        if not _same_anchored_directory_entry(
            parent_lease,
            name=output_root.name,
            expected=stage_identity,
        ):
            raise ValueError("fixed2 committed output entry identity changed")
        require_parent("after_fixed2_contract_commit")
        return result
    except BaseException as error:
        if stage_lease is not None:
            retained = output_root if renamed_to_output else stage
            quarantine_note = (
                f"fixed2 publication retained failure evidence at {retained}; "
                f"{contract_marker_name} commit marker is absent, invalid, or requires "
                "independent verification; no files or directories were deleted"
            )
            setattr(error, "fixed2_quarantine", quarantine_note)
            if hasattr(error, "add_note"):
                error.add_note(quarantine_note)
            else:
                error.args = (*error.args, quarantine_note)
        raise
    finally:
        if stage_lease is not None:
            stage_lease.close()
        parent_lease.close()


def _materialize_fixed2_overlay_analysis_test_only(
    *,
    multiview_root: Path,
    full_records: Path,
    blind_records: Path,
    blind_contract: Path,
    original_dataset_root: Path,
    output_root: Path,
) -> dict[str, object]:
    """Publish a non-canonical fixture that can never authorize formal exact8."""

    return _materialize_fixed2_overlay_impl(
        formal_windows_publication=False,
        multiview_root=multiview_root,
        full_records=full_records,
        blind_records=blind_records,
        blind_contract=blind_contract,
        original_dataset_root=original_dataset_root,
        output_root=output_root,
    )


def _formal_windows_publication_available() -> bool:
    return os.name == "nt"


def materialize_fixed2_overlay(
    *,
    multiview_root: Path,
    full_records: Path,
    blind_records: Path,
    blind_contract: Path,
    original_dataset_root: Path,
    output_root: Path,
) -> dict[str, object]:
    """Formally publish one fixed-two-view overlay on anchored Windows only."""

    if not _formal_windows_publication_available():
        raise OSError(
            errno.ENOTSUP,
            "formal fixed2 overlay materialization requires Windows atomic "
            "parent-handle-relative stage creation",
            os.fspath(output_root),
        )
    return _materialize_fixed2_overlay_impl(
        formal_windows_publication=True,
        multiview_root=multiview_root,
        full_records=full_records,
        blind_records=blind_records,
        blind_contract=blind_contract,
        original_dataset_root=original_dataset_root,
        output_root=output_root,
    )


def _verify_artifact_bindings(
    artifacts: Mapping[str, Any],
    *,
    expected_paths: Mapping[str, Path],
    actual_paths: Mapping[str, Path] | None = None,
) -> dict[str, dict[str, object]]:
    if set(artifacts) != set(expected_paths):
        raise ValueError("fixed2 artifact key set changed")
    actual_paths = {} if actual_paths is None else actual_paths
    if not set(actual_paths).issubset(expected_paths):
        raise ValueError("fixed2 actual artifact override key set changed")
    verified: dict[str, dict[str, object]] = {}
    for name, expected_path in expected_paths.items():
        binding = _mapping(artifacts.get(name), description=f"fixed2 artifact {name}")
        if name in actual_paths:
            declared_path = _declared_absolute_path(
                binding.get("path"), description=f"fixed2 artifact {name}"
            )
            if declared_path != expected_path:
                raise ValueError(f"fixed2 artifact {name} is not the declared file")
            path = _existing(
                actual_paths[name],
                directory=False,
                description=f"fixed2 staged artifact {name}",
            )
        else:
            path = _absolute_file(binding.get("path"), description=f"fixed2 artifact {name}")
            _samefile(path, expected_path, description=f"fixed2 artifact {name}")
        expected_sha = _require_sha(binding.get("sha256"), description=f"fixed2 artifact {name}")
        if _sha256(path) != expected_sha:
            raise ValueError(f"fixed2 artifact {name} SHA-256 changed")
        size = binding.get("size_bytes")
        if isinstance(size, bool) or not isinstance(size, int) or size != path.stat().st_size:
            raise ValueError(f"fixed2 artifact {name} size changed")
        verified[name] = dict(binding)
    return verified


def _verify_fixed2_overlay_payload(
    contract: Mapping[str, Any],
    *,
    expected_kind: str,
    expected_publication_authority: str,
    actual_publication_identity: Mapping[str, object] | None,
    declared_contract_directory: Path,
    actual_composite_records: Path | None,
    actual_composite_dataset_contract: Path | None,
    blind_records: Path,
    blind_contract: Path,
    multiview_root: Path | None = None,
    expected_full_records: Path | None = None,
    full_records: Path | None = None,
    original_dataset_root: Path | None = None,
) -> dict[str, object]:
    """Rebuild sealed evidence, optionally reading the two artifacts from stage."""

    formal_publication = _require_fixed2_publication_profile(
        contract_kind=expected_kind,
        publication_authority=expected_publication_authority,
        consumer_optimizer_input_ready=(expected_kind == FIXED2_CONTRACT_KIND),
    )
    if (actual_composite_records is None) != (
        actual_composite_dataset_contract is None
    ):
        raise ValueError("fixed2 staged artifact paths must be provided together")
    contract = _mapping(contract, description="fixed2 sealed contract")
    declared_contract_directory = Path(
        os.path.abspath(os.fspath(declared_contract_directory))
    )
    if expected_full_records is not None and full_records is not None:
        raise ValueError("provide expected_full_records or full_records, not both")
    expected_full_records = expected_full_records if expected_full_records is not None else full_records
    blind_records = _existing(blind_records, directory=False, description="fixed2 blind records")
    blind_contract = _existing(blind_contract, directory=False, description="fixed2 blind contract")
    supplied_multiview_root = (
        _existing(multiview_root, directory=True, description="fixed2 multiview root")
        if multiview_root is not None
        else None
    )
    claimed_integrity = contract.get("integrity_sha256")
    unsigned = {key: value for key, value in contract.items() if key != "integrity_sha256"}
    if not isinstance(claimed_integrity, str) or claimed_integrity != _canonical_sha256(unsigned):
        raise ValueError("fixed2 contract integrity SHA-256 changed")
    _expect(contract.get("schema_version"), SCHEMA_VERSION, description="fixed2 schema")
    if (expected_kind, expected_publication_authority) not in {
        (FIXED2_CONTRACT_KIND, FIXED2_PUBLICATION_AUTHORITY),
        (
            FIXED2_ANALYSIS_CONTRACT_KIND,
            FIXED2_ANALYSIS_PUBLICATION_AUTHORITY,
        ),
    }:
        raise ValueError("unsupported fixed2 verifier publication profile")
    _expect(contract.get("kind"), expected_kind, description="fixed2 kind")
    _expect(
        contract.get("publication_authority"),
        expected_publication_authority,
        description="fixed2 publication authority",
    )
    bound_multiview = _existing(
        Path(str(contract.get("multiview_root"))), directory=True, description="bound multiview root"
    )
    if supplied_multiview_root is not None:
        _samefile(
            bound_multiview,
            supplied_multiview_root,
            description="fixed2 multiview root",
        )
    multiview_root = bound_multiview
    bound_original_root = _existing(
        Path(str(contract.get("original_dataset_root"))),
        directory=True,
        description="bound original dataset root",
    )
    if original_dataset_root is not None:
        supplied_root = _existing(
            original_dataset_root, directory=True, description="supplied original dataset root"
        )
        _samefile(bound_original_root, supplied_root, description="fixed2 original dataset root")
    common_root = _existing(
        Path(str(contract.get("composite_dataset_root"))),
        directory=True,
        description="fixed2 common dataset root",
    )
    recomputed_common = _existing(
        Path(os.path.commonpath((str(bound_original_root), str(multiview_root)))),
        directory=True,
        description="recomputed common dataset root",
    )
    _samefile(common_root, recomputed_common, description="fixed2 common dataset root")
    verification = _verify_fixed2_teacher_overlay_source(
        blind_manifest=blind_records,
        blind_contract=blind_contract,
        export_root=multiview_root,
        dataset_root=bound_original_root,
        formal_windows_source=formal_publication,
    )
    bindings = _mapping(verification.policy["bindings"], description="verified overlay bindings")
    bound_full = _absolute_file(
        _mapping(bindings["full_manifest"], description="full manifest binding")["path"],
        description="verified full records",
    )
    if expected_full_records is not None:
        supplied_full = _existing(
            expected_full_records, directory=False, description="expected full records"
        )
        _samefile(bound_full, supplied_full, description="fixed2 full records")
    declared_composite_records = _declared_absolute_path(
        contract.get("composite_records_path"),
        description="fixed2 composite records",
    )
    composite_records = (
        _existing(
            actual_composite_records,
            directory=False,
            description="fixed2 staged composite records",
        )
        if actual_composite_records is not None
        else _absolute_file(
            contract.get("composite_records_path"),
            description="fixed2 composite records",
        )
    )
    _expect(
        contract.get("composite_records"),
        str(declared_composite_records),
        description="fixed2 composite records alias",
    )
    declared_dataset_contract = _declared_absolute_path(
        contract.get("composite_dataset_contract"),
        description="fixed2 composite dataset contract",
    )
    composite_dataset_contract = (
        _existing(
            actual_composite_dataset_contract,
            directory=False,
            description="fixed2 staged composite dataset contract",
        )
        if actual_composite_dataset_contract is not None
        else _absolute_file(
            contract.get("composite_dataset_contract"),
            description="fixed2 composite dataset contract",
        )
    )
    declared_parent = declared_composite_records.parent
    if (
        declared_dataset_contract.parent != declared_parent
        or declared_parent != declared_contract_directory
    ):
        raise ValueError("fixed2 composite artifacts are not contained by the contract directory")
    (
        expected_rows,
        selector_evidence,
        selected_composite_bindings,
        validation_pixel_bindings,
    ) = _composite_rows(
        blind_records=blind_records,
        original_dataset_root=bound_original_root,
        composite_dataset_root=common_root,
        verification=verification,
    )
    if composite_records.read_bytes() != _jsonl_bytes(expected_rows):
        raise ValueError("fixed2 composite records do not match the verified overlay selection")
    source_contract_path = _absolute_file(
        _mapping(bindings["source_dataset_contract"], description="source contract binding")["path"],
        description="fixed2 source dataset contract",
    )
    source_contract = _strict_json(source_contract_path)
    expected_dataset_contract = _composite_dataset_contract(
        contract_kind=expected_kind,
        publication_authority=expected_publication_authority,
        consumer_optimizer_input_ready=(expected_kind == FIXED2_CONTRACT_KIND),
        source_contract=source_contract,
        source_contract_path=source_contract_path,
        composite_records_path=declared_composite_records,
        composite_dataset_root=common_root,
        rows=expected_rows,
    )
    if composite_dataset_contract.read_bytes() != _json_bytes(expected_dataset_contract):
        raise ValueError("fixed2 composite dataset contract changed")
    producer_contract_name = (
        FIXED2_SOURCE_CONTRACT_NAME
        if formal_publication
        else FIXED2_SOURCE_ANALYSIS_CONTRACT_NAME
    )
    expected_paths = {
        "full_records": bound_full,
        "blind_records": blind_records,
        "blind_contract": blind_contract,
        "multiview_export_contract": multiview_root / producer_contract_name,
        "multiview_export_manifest": multiview_root / "multiview_train.jsonl",
        "source_dataset_contract": source_contract_path,
        "composite_records": declared_composite_records,
        "composite_dataset_contract": declared_dataset_contract,
        **_code_artifact_paths(),
    }
    artifacts = _verify_artifact_bindings(
        _mapping(contract.get("artifacts"), description="fixed2 artifacts"),
        expected_paths=expected_paths,
        actual_paths=(
            {
                "composite_records": composite_records,
                "composite_dataset_contract": composite_dataset_contract,
            }
            if actual_composite_records is not None
            else None
        ),
    )
    if formal_publication:
        observed_publication_identity = (
            _normalized_fixed2_publication_identity(actual_publication_identity)
            if actual_publication_identity is not None
            else _fixed2_publication_identity(
                directory_identity=_directory_identity(declared_contract_directory),
                pre_marker_file_identities={
                    declared_composite_records.name: _file_identity(composite_records),
                    declared_dataset_contract.name: _file_identity(
                        composite_dataset_contract
                    ),
                },
            )
        )
    else:
        if actual_publication_identity is not None:
            raise ValueError("analysis verification cannot accept formal identity evidence")
        observed_publication_identity = None
    rebuilt = _fixed2_contract_payload(
        contract_kind=expected_kind,
        publication_authority=expected_publication_authority,
        consumer_optimizer_input_ready=(expected_kind == FIXED2_CONTRACT_KIND),
        publication_identity=observed_publication_identity,
        multiview_root=multiview_root,
        original_dataset_root=bound_original_root,
        composite_records_path=declared_composite_records,
        composite_dataset_contract_path=declared_dataset_contract,
        composite_dataset_root=common_root,
        producer_subject_id=str(verification.policy["producer_subject_id"]),
        producer_manifest_semantic_sha256=str(
            verification.policy["producer_manifest_semantic_sha256"]
        ),
        source_dataset_contract_semantic_sha256=str(
            verification.policy["source_dataset_contract_semantic_sha256"]
        ),
        selector_evidence=selector_evidence,
        selected_composite_bindings=selected_composite_bindings,
        validation_pixel_bindings=validation_pixel_bindings,
        artifacts=artifacts,
    )
    if _canonical_sha256(unsigned) != _canonical_sha256(rebuilt):
        raise ValueError("fixed2 contract does not match recomputed source evidence")
    return dict(contract)


def _verify_fixed2_overlay_analysis_test_only(
    contract_path: Path,
    *,
    blind_records: Path,
    blind_contract: Path,
    multiview_root: Path | None = None,
    expected_full_records: Path | None = None,
    full_records: Path | None = None,
    original_dataset_root: Path | None = None,
) -> dict[str, object]:
    """Reopen only the explicitly non-canonical analysis fixture marker."""

    contract_path = _existing(
        contract_path,
        directory=False,
        description="fixed2 analysis fixture marker",
    )
    if contract_path.name != FIXED2_ANALYSIS_MARKER_NAME:
        raise ValueError("fixed2 analysis fixture must use its non-canonical filename")
    contract = _strict_json(contract_path)
    return _verify_fixed2_overlay_payload(
        contract,
        expected_kind=FIXED2_ANALYSIS_CONTRACT_KIND,
        expected_publication_authority=FIXED2_ANALYSIS_PUBLICATION_AUTHORITY,
        actual_publication_identity=None,
        declared_contract_directory=contract_path.parent,
        actual_composite_records=None,
        actual_composite_dataset_contract=None,
        blind_records=blind_records,
        blind_contract=blind_contract,
        multiview_root=multiview_root,
        expected_full_records=expected_full_records,
        full_records=full_records,
        original_dataset_root=original_dataset_root,
    )


def verify_fixed2_overlay_contract(
    contract_path: Path,
    *,
    blind_records: Path,
    blind_contract: Path,
    multiview_root: Path | None = None,
    expected_full_records: Path | None = None,
    full_records: Path | None = None,
    original_dataset_root: Path | None = None,
) -> dict[str, object]:
    """Reopen only a canonical, committed fixed2 publication."""

    if not _formal_windows_publication_available():
        raise OSError(
            errno.ENOTSUP,
            "formal fixed2 contract verification requires Windows publication authority",
            os.fspath(contract_path),
        )
    contract_path = _existing(
        contract_path,
        directory=False,
        description="fixed2 canonical contract commit marker",
    )
    if contract_path.name != FIXED2_CANONICAL_CONTRACT_NAME:
        raise ValueError("fixed2 contract must use the canonical commit-marker filename")
    contract = _strict_json(contract_path)
    return _verify_fixed2_overlay_payload(
        contract,
        expected_kind=FIXED2_CONTRACT_KIND,
        expected_publication_authority=FIXED2_PUBLICATION_AUTHORITY,
        actual_publication_identity=None,
        declared_contract_directory=contract_path.parent,
        actual_composite_records=None,
        actual_composite_dataset_contract=None,
        blind_records=blind_records,
        blind_contract=blind_contract,
        multiview_root=multiview_root,
        expected_full_records=expected_full_records,
        full_records=full_records,
        original_dataset_root=original_dataset_root,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Materialize or verify one sealed fixed-two-view recipient overlay."
    )
    commands = parser.add_subparsers(dest="command", required=True)

    materialize = commands.add_parser(
        "materialize-fixed2",
        help="Publish a new no-clobber fixed-two-view overlay contract.",
    )
    materialize.add_argument("--multiview-root", type=Path, required=True)
    materialize.add_argument("--full-records", type=Path, required=True)
    materialize.add_argument("--blind-records", type=Path, required=True)
    materialize.add_argument("--blind-contract", type=Path, required=True)
    materialize.add_argument("--original-dataset-root", type=Path, required=True)
    materialize.add_argument("--output-root", type=Path, required=True)

    verify = commands.add_parser(
        "verify-fixed2",
        help="Reopen and fully verify an existing fixed-two-view overlay contract.",
    )
    verify.add_argument("--contract-path", type=Path, required=True)
    verify.add_argument("--blind-records", type=Path, required=True)
    verify.add_argument("--blind-contract", type=Path, required=True)
    verify.add_argument("--multiview-root", type=Path)
    verify.add_argument("--full-records", type=Path)
    verify.add_argument("--original-dataset-root", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "materialize-fixed2":
        result = materialize_fixed2_overlay(
            multiview_root=args.multiview_root,
            full_records=args.full_records,
            blind_records=args.blind_records,
            blind_contract=args.blind_contract,
            original_dataset_root=args.original_dataset_root,
            output_root=args.output_root,
        )
    elif args.command == "verify-fixed2":
        result = verify_fixed2_overlay_contract(
            contract_path=args.contract_path,
            blind_records=args.blind_records,
            blind_contract=args.blind_contract,
            multiview_root=args.multiview_root,
            expected_full_records=args.full_records,
            original_dataset_root=args.original_dataset_root,
        )
    else:  # pragma: no cover - argparse owns the command choice
        raise AssertionError(f"unsupported fixed2 command: {args.command}")
    print(
        json.dumps(
            result,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    )


if __name__ == "__main__":
    main()
