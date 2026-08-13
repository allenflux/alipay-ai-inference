#!/usr/bin/env python3
"""Materialize the sealed first-1,000 prefix of a received 10,000-image publication.

This command is intentionally local and fail-closed.  It validates the receive
receipt, source contract, complete manifest, deterministic selection order, and
the exact source tree.  It then re-hashes all 10,000 source images while copying
only the first 1,000 rows into a fresh staged publication.  The final directory
is made visible with an exclusive atomic rename; an existing destination or
receipt is never replaced.
"""

from __future__ import annotations

import argparse
import ctypes
import datetime as dt
import errno
import hashlib
import json
import os
import re
import shutil
import stat
import sys
import time
import unicodedata
import urllib.parse
import uuid
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO, Iterable


DEFAULT_BASE = Path("/Volumes/CodexData/white-input")
SOURCE_IMAGE_COUNT = 10_000
PILOT_IMAGE_COUNT = 1_000
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
VERSION_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
DOTNET_ROUND_TRIP_FRACTION_RE = re.compile(
    r"^(?P<prefix>.*T\d{2}:\d{2}:\d{2}\.\d{6})\d"
    r"(?P<offset>Z|[+-]\d{2}:\d{2})$"
)

MANIFEST_KEYS = {
    "schema_version",
    "index",
    "selection_key",
    "source_relative_path",
    "archive_relative_path",
    "size_bytes",
    "sha256",
    "source_last_write_utc",
}
CONTRACT_KEYS = {
    "schema_version",
    "kind",
    "source_semantics",
    "source_root_at_capture",
    "source_files_modified",
    "selection",
    "payload",
    "labels",
    "captured_utc",
    "host",
    "powershell_version",
}
SELECTION_KEYS = {
    "algorithm",
    "salt",
    "normalized_path",
    "supported_extensions",
    "candidate_count",
    "requested_count",
    "selected_count",
    "prefix_stable_when_candidate_set_and_salt_are_unchanged",
}
PAYLOAD_KEYS = {
    "images_root",
    "manifest",
    "manifest_sha256",
    "image_count",
    "image_bytes",
}
LABEL_KEYS = {
    "human_labels_present",
    "paddle_teacher_labels_present",
    "intended_next_step",
}
RECEIPT_KEYS = {
    "schema_version",
    "kind",
    "status",
    "download",
    "package_subject_sha256",
    "contract",
    "manifest",
    "verified_payload",
    "publication",
    "completed_utc",
}
DOWNLOAD_KEYS = {"url", "archive_path", "size_bytes", "sha256"}
RECEIPT_CONTRACT_KEYS = {"path", "size_bytes", "sha256", "kind"}
RECEIPT_MANIFEST_KEYS = {"path", "size_bytes", "sha256"}
VERIFIED_PAYLOAD_KEYS = {
    "image_count",
    "image_bytes",
    "every_file_size_and_sha256_verified",
    "archive_file_closure_exact",
}
PUBLICATION_KEYS = {"version", "raw_version_root", "brand_new", "atomic_rename"}


class MaterializeError(RuntimeError):
    """A source-integrity or publication-safety gate failed."""


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise MaterializeError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise MaterializeError(f"non-finite JSON number is forbidden: {value}")


def _load_json(raw: bytes, source: str) -> Any:
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise MaterializeError(f"{source} is not valid UTF-8") from exc
    try:
        return json.loads(
            text,
            object_pairs_hook=_strict_object,
            parse_constant=_reject_constant,
        )
    except MaterializeError:
        raise
    except (json.JSONDecodeError, ValueError) as exc:
        raise MaterializeError(f"invalid JSON in {source}: {exc}") from exc


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode(
        "utf-8"
    )


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _require_exact_keys(value: Any, expected: set[str], source: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise MaterializeError(f"{source} must be a JSON object")
    actual = set(value)
    if actual != expected:
        raise MaterializeError(
            f"{source} keys differ: missing={sorted(expected - actual)!r} "
            f"extra={sorted(actual - expected)!r}"
        )
    return value


def _require_string(value: Any, source: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise MaterializeError(f"{source} must be a non-empty string")
    return value


def _require_int(value: Any, source: str, *, positive: bool = False) -> int:
    if not _is_int(value) or value < (1 if positive else 0):
        qualifier = "positive" if positive else "non-negative"
        raise MaterializeError(f"{source} must be a {qualifier} integer")
    return value


def _require_sha(value: Any, source: str) -> str:
    if not isinstance(value, str) or not SHA256_RE.fullmatch(value):
        raise MaterializeError(f"{source} must be a lowercase SHA256")
    return value


def _require_utc(value: Any, source: str) -> str:
    text = _require_string(value, source)
    parse_text = text
    # PowerShell/.NET's round-trip ("o") format emits seven fractional-second
    # digits, while the Python runtime used by this delivery accepts at most
    # six.  Drop only the seventh digit in the temporary parse spelling.  The
    # sealed source spelling is returned unchanged and is therefore still
    # bound byte-for-byte by the manifest/contract hashes.
    dotnet_match = DOTNET_ROUND_TRIP_FRACTION_RE.fullmatch(parse_text)
    if dotnet_match is not None:
        parse_text = dotnet_match.group("prefix") + dotnet_match.group("offset")
    try:
        parsed = dt.datetime.fromisoformat(parse_text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise MaterializeError(f"{source} is not an ISO-8601 timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() != dt.timedelta(0):
        raise MaterializeError(f"{source} must include a UTC offset")
    return text


def _safe_relative_path(value: Any, source: str) -> str:
    text = _require_string(value, source)
    if "\x00" in text or "\\" in text or text.startswith("/"):
        raise MaterializeError(f"unsafe path in {source}: {text!r}")
    if unicodedata.normalize("NFC", text) != text:
        raise MaterializeError(f"path in {source} must use Unicode NFC: {text!r}")
    path = PurePosixPath(text)
    if not path.parts or any(part in {"", ".", ".."} for part in path.parts):
        raise MaterializeError(f"unsafe path in {source}: {text!r}")
    if path.as_posix() != text or re.match(r"^[A-Za-z]:", path.parts[0]):
        raise MaterializeError(f"non-canonical path in {source}: {text!r}")
    return text


def _collision_key(path: str) -> str:
    return unicodedata.normalize("NFC", path).casefold()


def _signature(info: os.stat_result) -> tuple[int, int, int, int, int]:
    return (info.st_dev, info.st_ino, info.st_mode, info.st_size, info.st_mtime_ns)


def _read_regular_file(path: Path, source: str, *, limit: int) -> bytes:
    try:
        before = path.lstat()
    except OSError as exc:
        raise MaterializeError(f"cannot stat {source}: {exc}") from exc
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
        raise MaterializeError(f"{source} must be a regular non-symlink file")
    if before.st_size > limit:
        raise MaterializeError(f"{source} is too large: {before.st_size} > {limit}")
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
        with os.fdopen(descriptor, "rb") as stream:
            opened = os.fstat(stream.fileno())
            if _signature(opened) != _signature(before):
                raise MaterializeError(f"{source} changed while opening")
            raw = stream.read(limit + 1)
            after_open = os.fstat(stream.fileno())
    except MaterializeError:
        raise
    except OSError as exc:
        raise MaterializeError(f"cannot read {source}: {exc}") from exc
    try:
        after_path = path.lstat()
    except OSError as exc:
        raise MaterializeError(f"cannot restat {source}: {exc}") from exc
    if (
        len(raw) != before.st_size
        or len(raw) > limit
        or _signature(after_open) != _signature(before)
        or _signature(after_path) != _signature(before)
    ):
        raise MaterializeError(f"{source} changed while reading")
    return raw


def _open_output(path: Path) -> BinaryIO:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    return os.fdopen(os.open(path, flags, 0o600), "wb")


def _write_exclusive(path: Path, payload: bytes) -> None:
    with _open_output(path) as output:
        output.write(payload)
        output.flush()
        os.fsync(output.fileno())


def _parse_manifest(raw: bytes) -> tuple[list[dict[str, Any]], list[bytes]]:
    if not raw or not raw.endswith(b"\n"):
        raise MaterializeError("manifest.jsonl must be non-empty and newline terminated")
    encoded_lines = raw.splitlines(keepends=True)
    rows: list[dict[str, Any]] = []
    source_paths: dict[str, str] = {}
    archive_paths: dict[str, str] = {}
    previous_sort_key: tuple[str, str] | None = None
    for line_number, encoded_line in enumerate(encoded_lines, 1):
        if not encoded_line.endswith(b"\n"):
            raise MaterializeError(f"manifest line {line_number} is not newline terminated")
        document = encoded_line[:-1]
        if document.endswith(b"\r"):
            document = document[:-1]
        if not document:
            raise MaterializeError(f"blank manifest line at {line_number}")
        row = _require_exact_keys(
            _load_json(document, f"manifest line {line_number}"),
            MANIFEST_KEYS,
            f"manifest line {line_number}",
        )
        if (
            not _is_int(row["schema_version"])
            or row["schema_version"] != 1
            or not _is_int(row["index"])
            or row["index"] != line_number
        ):
            raise MaterializeError(f"manifest line {line_number} has invalid schema/index")
        selection_key = _require_sha(
            row["selection_key"], f"manifest line {line_number}.selection_key"
        )
        source_path = _safe_relative_path(
            row["source_relative_path"],
            f"manifest line {line_number}.source_relative_path",
        )
        archive_path = _safe_relative_path(
            row["archive_relative_path"],
            f"manifest line {line_number}.archive_relative_path",
        )
        if archive_path != f"images/{source_path}":
            raise MaterializeError(
                f"manifest line {line_number} archive path is not the source path below images/"
            )
        if _require_int(row["size_bytes"], f"manifest line {line_number}.size_bytes") == 0:
            raise MaterializeError(f"manifest line {line_number} describes an empty image")
        _require_sha(row["sha256"], f"manifest line {line_number}.sha256")
        _require_utc(
            row["source_last_write_utc"],
            f"manifest line {line_number}.source_last_write_utc",
        )
        normalized = unicodedata.normalize("NFC", source_path).lower()
        sort_key = (selection_key, normalized)
        if previous_sort_key is not None and sort_key <= previous_sort_key:
            raise MaterializeError("manifest is not in strict deterministic selection order")
        previous_sort_key = sort_key
        for seen, candidate, label in (
            (source_paths, source_path, "source"),
            (archive_paths, archive_path, "archive"),
        ):
            collision = _collision_key(candidate)
            if collision in seen:
                raise MaterializeError(
                    f"manifest {label} path collision: {candidate!r} and {seen[collision]!r}"
                )
            seen[collision] = candidate
        rows.append(row)
    return rows, encoded_lines


def _parse_contract(
    raw: bytes,
    *,
    manifest_sha256: str,
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    contract = _require_exact_keys(
        _load_json(raw, "dataset.contract.json"), CONTRACT_KEYS, "dataset.contract.json"
    )
    if not _is_int(contract["schema_version"]) or contract["schema_version"] != 1:
        raise MaterializeError("unsupported source contract schema_version")
    if contract["kind"] != "otherimages_white_sync_dataset_v1":
        raise MaterializeError("unexpected source contract kind")
    if contract["source_semantics"] != "white_image_unlabeled_source_only":
        raise MaterializeError("unexpected source semantics")
    _require_string(contract["source_root_at_capture"], "contract.source_root_at_capture")
    if contract["source_files_modified"] is not False:
        raise MaterializeError("source contract must assert source_files_modified=false")
    _require_utc(contract["captured_utc"], "contract.captured_utc")
    _require_string(contract["host"], "contract.host")
    _require_string(contract["powershell_version"], "contract.powershell_version")

    selection = _require_exact_keys(contract["selection"], SELECTION_KEYS, "contract.selection")
    if selection["algorithm"] != "sha256_utf8_salt_lf_normalized_relative_path_sort_v1":
        raise MaterializeError("unexpected source selection algorithm")
    salt = _require_string(selection["salt"], "contract.selection.salt")
    if selection["normalized_path"] != "unicode_nfc_lowercase_forward_slashes":
        raise MaterializeError("unexpected source path normalization")
    extensions = selection["supported_extensions"]
    if (
        not isinstance(extensions, list)
        or not extensions
        or any(
            not isinstance(item, str)
            or not re.fullmatch(r"\.[a-z0-9]+", item)
            for item in extensions
        )
        or len(extensions) != len(set(extensions))
    ):
        raise MaterializeError("contract.selection.supported_extensions is invalid")
    candidate_count = _require_int(selection["candidate_count"], "candidate_count")
    requested_count = _require_int(selection["requested_count"], "requested_count")
    selected_count = _require_int(selection["selected_count"], "selected_count")
    if selection["prefix_stable_when_candidate_set_and_salt_are_unchanged"] is not True:
        raise MaterializeError("source contract lacks the prefix-stability assertion")
    if (
        len(rows) != SOURCE_IMAGE_COUNT
        or requested_count != SOURCE_IMAGE_COUNT
        or selected_count != SOURCE_IMAGE_COUNT
        or candidate_count < SOURCE_IMAGE_COUNT
    ):
        raise MaterializeError(
            f"source contract/manifest must bind exactly {SOURCE_IMAGE_COUNT} selected images"
        )

    extension_set = set(extensions)
    for line_number, row in enumerate(rows, 1):
        normalized = unicodedata.normalize("NFC", row["source_relative_path"]).lower()
        expected_key = hashlib.sha256(f"{salt}\n{normalized}".encode("utf-8")).hexdigest()
        if row["selection_key"] != expected_key:
            raise MaterializeError(f"manifest line {line_number} selection key is not reproducible")
        if PurePosixPath(row["source_relative_path"]).suffix.lower() not in extension_set:
            raise MaterializeError(f"manifest line {line_number} has an unsupported extension")

    payload = _require_exact_keys(contract["payload"], PAYLOAD_KEYS, "contract.payload")
    if payload["images_root"] != "images" or payload["manifest"] != "manifest.jsonl":
        raise MaterializeError("unexpected source payload paths")
    if _require_sha(payload["manifest_sha256"], "payload.manifest_sha256") != manifest_sha256:
        raise MaterializeError("source contract manifest SHA256 mismatch")
    image_bytes = sum(row["size_bytes"] for row in rows)
    if (
        _require_int(payload["image_count"], "payload.image_count") != SOURCE_IMAGE_COUNT
        or _require_int(payload["image_bytes"], "payload.image_bytes") != image_bytes
    ):
        raise MaterializeError("source contract payload counts do not bind the manifest")

    labels = _require_exact_keys(contract["labels"], LABEL_KEYS, "contract.labels")
    if labels["human_labels_present"] is not False:
        raise MaterializeError("source contract unexpectedly claims human labels")
    if labels["paddle_teacher_labels_present"] is not False:
        raise MaterializeError("source contract unexpectedly claims teacher labels")
    if (
        labels["intended_next_step"]
        != "paddle_teacher_generation_then_frozen_split_training_and_validation"
    ):
        raise MaterializeError("unexpected source intended_next_step")
    return contract


def _parse_receive_receipt(
    raw: bytes,
    *,
    receipt_path: Path,
    source_root: Path,
    contract_raw: bytes,
    manifest_raw: bytes,
    contract: dict[str, Any],
    rows: list[dict[str, Any]],
) -> tuple[dict[str, Any], str]:
    receipt = _require_exact_keys(
        _load_json(raw, "receive receipt"), RECEIPT_KEYS, "receive receipt"
    )
    if not _is_int(receipt["schema_version"]) or receipt["schema_version"] != 1:
        raise MaterializeError("unsupported receive receipt schema_version")
    if receipt["kind"] != "otherimages_white_sync_receive_receipt_v1":
        raise MaterializeError("unexpected receive receipt kind")
    if receipt["status"] != "complete":
        raise MaterializeError("receive receipt is not complete")
    _require_utc(receipt["completed_utc"], "receive receipt.completed_utc")

    download = _require_exact_keys(receipt["download"], DOWNLOAD_KEYS, "receive receipt.download")
    parsed_url = urllib.parse.urlsplit(_require_string(download["url"], "download.url"))
    if parsed_url.scheme not in {"http", "https"} or not parsed_url.netloc:
        raise MaterializeError("receive receipt download URL must be absolute HTTP(S)")
    _require_string(download["archive_path"], "download.archive_path")
    _require_int(download["size_bytes"], "download.size_bytes", positive=True)
    _require_sha(download["sha256"], "download.sha256")

    contract_sha = _sha256_bytes(contract_raw)
    manifest_sha = _sha256_bytes(manifest_raw)
    contract_binding = _require_exact_keys(
        receipt["contract"], RECEIPT_CONTRACT_KEYS, "receive receipt.contract"
    )
    if (
        _require_string(contract_binding["path"], "receive receipt.contract.path")
        != "dataset.contract.json"
        or _require_int(
            contract_binding["size_bytes"], "receive receipt.contract.size_bytes"
        )
        != len(contract_raw)
        or _require_sha(
            contract_binding["sha256"], "receive receipt.contract.sha256"
        )
        != contract_sha
        or _require_string(contract_binding["kind"], "receive receipt.contract.kind")
        != contract["kind"]
    ):
        raise MaterializeError("receive receipt contract binding mismatch")
    manifest_binding = _require_exact_keys(
        receipt["manifest"], RECEIPT_MANIFEST_KEYS, "receive receipt.manifest"
    )
    if (
        _require_string(manifest_binding["path"], "receive receipt.manifest.path")
        != "manifest.jsonl"
        or _require_int(
            manifest_binding["size_bytes"], "receive receipt.manifest.size_bytes"
        )
        != len(manifest_raw)
        or _require_sha(
            manifest_binding["sha256"], "receive receipt.manifest.sha256"
        )
        != manifest_sha
    ):
        raise MaterializeError("receive receipt manifest binding mismatch")

    image_bytes = sum(row["size_bytes"] for row in rows)
    verified = _require_exact_keys(
        receipt["verified_payload"],
        VERIFIED_PAYLOAD_KEYS,
        "receive receipt.verified_payload",
    )
    if (
        _require_int(verified["image_count"], "verified_payload.image_count")
        != SOURCE_IMAGE_COUNT
        or _require_int(verified["image_bytes"], "verified_payload.image_bytes")
        != image_bytes
        or verified["every_file_size_and_sha256_verified"] is not True
        or verified["archive_file_closure_exact"] is not True
    ):
        raise MaterializeError("receive receipt verified payload is incomplete or mismatched")

    publication = _require_exact_keys(
        receipt["publication"], PUBLICATION_KEYS, "receive receipt.publication"
    )
    if _require_string(publication["version"], "publication.version") != source_root.name:
        raise MaterializeError("receive receipt version does not match the source root")
    bound_root = Path(_require_string(publication["raw_version_root"], "publication.raw_version_root"))
    try:
        if not bound_root.is_absolute() or not bound_root.samefile(source_root):
            raise MaterializeError("receive receipt raw root does not bind the supplied source root")
    except FileNotFoundError as exc:
        raise MaterializeError("receive receipt raw root does not exist") from exc
    if publication["brand_new"] is not True or publication["atomic_rename"] is not True:
        raise MaterializeError("receive receipt does not assert a fresh atomic publication")

    subject = hashlib.sha256(f"{manifest_sha}\n{contract_sha}\n".encode()).hexdigest()
    if _require_sha(receipt["package_subject_sha256"], "package_subject_sha256") != subject:
        raise MaterializeError("receive receipt package subject mismatch")
    if receipt_path.is_symlink():
        raise MaterializeError("receive receipt path must not be a symlink")
    return receipt, subject


def _expected_directories(files: Iterable[str]) -> set[str]:
    directories: set[str] = set()
    for name in files:
        parent = PurePosixPath(name).parent
        while parent.as_posix() != ".":
            directories.add(parent.as_posix())
            parent = parent.parent
    return directories


def _tree_closure(root: Path) -> tuple[set[str], set[str]]:
    files: set[str] = set()
    directories: set[str] = set()
    stack = [root]
    while stack:
        current = stack.pop()
        try:
            entries = list(os.scandir(current))
        except OSError as exc:
            raise MaterializeError(f"cannot scan publication tree {current}: {exc}") from exc
        for entry in entries:
            relative = Path(entry.path).relative_to(root).as_posix()
            try:
                info = entry.stat(follow_symlinks=False)
            except OSError as exc:
                raise MaterializeError(f"cannot stat publication entry {relative}: {exc}") from exc
            if entry.is_symlink():
                raise MaterializeError(f"publication contains a symlink: {relative}")
            if stat.S_ISDIR(info.st_mode):
                directories.add(relative)
                stack.append(Path(entry.path))
            elif stat.S_ISREG(info.st_mode):
                files.add(relative)
            else:
                raise MaterializeError(f"publication contains a non-regular entry: {relative}")
    return files, directories


def _assert_tree_closure(root: Path, expected_files: set[str], source: str) -> None:
    actual_files, actual_directories = _tree_closure(root)
    expected_directories = _expected_directories(expected_files)
    if actual_files != expected_files or actual_directories != expected_directories:
        raise MaterializeError(
            f"{source} tree closure mismatch: "
            f"missing_files={sorted(expected_files - actual_files)[:10]!r} "
            f"extra_files={sorted(actual_files - expected_files)[:10]!r} "
            f"missing_dirs={sorted(expected_directories - actual_directories)[:10]!r} "
            f"extra_dirs={sorted(actual_directories - expected_directories)[:10]!r}"
        )


def _copy_or_hash_source(
    source: Path,
    *,
    expected_size: int,
    expected_sha256: str,
    destination: Path | None,
) -> None:
    try:
        before = source.lstat()
    except OSError as exc:
        raise MaterializeError(f"cannot stat source image {source}: {exc}") from exc
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
        raise MaterializeError(f"source image is not a regular non-symlink file: {source}")
    if before.st_size != expected_size:
        raise MaterializeError(f"source image size mismatch: {source}")
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    digest = hashlib.sha256()
    observed = 0
    destination_stream: BinaryIO | None = None
    try:
        descriptor = os.open(source, flags)
        with os.fdopen(descriptor, "rb") as input_stream:
            opened = os.fstat(input_stream.fileno())
            if _signature(opened) != _signature(before):
                raise MaterializeError(f"source image changed while opening: {source}")
            if destination is not None:
                destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
                destination_stream = _open_output(destination)
            try:
                for chunk in iter(lambda: input_stream.read(1024 * 1024), b""):
                    observed += len(chunk)
                    if observed > expected_size:
                        raise MaterializeError(f"source image exceeded manifest size: {source}")
                    digest.update(chunk)
                    if destination_stream is not None:
                        destination_stream.write(chunk)
                if destination_stream is not None:
                    destination_stream.flush()
                    os.fsync(destination_stream.fileno())
                after_open = os.fstat(input_stream.fileno())
            finally:
                if destination_stream is not None:
                    destination_stream.close()
    except MaterializeError:
        raise
    except OSError as exc:
        raise MaterializeError(f"cannot read/copy source image {source}: {exc}") from exc
    try:
        after_path = source.lstat()
    except OSError as exc:
        raise MaterializeError(f"cannot restat source image {source}: {exc}") from exc
    actual_sha = digest.hexdigest()
    if (
        observed != expected_size
        or actual_sha != expected_sha256
        or _signature(after_open) != _signature(before)
        or _signature(after_path) != _signature(before)
    ):
        raise MaterializeError(f"source image size/SHA256 or stability mismatch: {source}")
    if destination is not None:
        info = destination.lstat()
        if (
            not stat.S_ISREG(info.st_mode)
            or info.st_size != expected_size
            or _sha256_file(destination) != expected_sha256
        ):
            raise MaterializeError(f"staged image verification failed: {destination}")


def _atomic_rename_exclusive(source: Path, destination: Path) -> None:
    """Atomically rename while asking the OS to fail if destination exists."""

    source_bytes = os.fsencode(source)
    destination_bytes = os.fsencode(destination)
    if sys.platform == "darwin":
        libc = ctypes.CDLL(None, use_errno=True)
        renamex_np = libc.renamex_np
        renamex_np.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint]
        renamex_np.restype = ctypes.c_int
        result = renamex_np(source_bytes, destination_bytes, 0x00000004)  # RENAME_EXCL
    elif sys.platform.startswith("linux"):
        libc = ctypes.CDLL(None, use_errno=True)
        renameat2 = getattr(libc, "renameat2", None)
        if renameat2 is None:
            raise MaterializeError("platform lacks atomic exclusive rename support")
        renameat2.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        renameat2.restype = ctypes.c_int
        result = renameat2(-100, source_bytes, -100, destination_bytes, 1)  # RENAME_NOREPLACE
    elif os.name == "nt":
        try:
            os.rename(source, destination)
            return
        except FileExistsError as exc:
            raise MaterializeError(f"refusing to overwrite existing publication: {destination}") from exc
    else:
        raise MaterializeError("platform lacks atomic exclusive rename support")
    if result != 0:
        error_number = ctypes.get_errno()
        if error_number in {errno.EEXIST, errno.ENOTEMPTY}:
            raise MaterializeError(f"refusing to overwrite existing publication: {destination}")
        raise MaterializeError(
            f"exclusive atomic rename failed for {destination}: {os.strerror(error_number)}"
        )


def _ensure_root(path: Path, source: str) -> Path:
    if path.is_symlink():
        raise MaterializeError(f"{source} must not be a symlink")
    path.mkdir(parents=True, exist_ok=True)
    if path.is_symlink() or not path.is_dir():
        raise MaterializeError(f"{source} must be a non-symlink directory")
    return path.resolve(strict=True)


def _paths_overlap(left: Path, right: Path) -> bool:
    return left == right or left in right.parents or right in left.parents


def _exists_or_symlink(path: Path) -> bool:
    return path.exists() or path.is_symlink()


def materialize_pilot(
    *,
    source_root: Path,
    source_receipt: Path,
    output_root: Path,
    evidence_root: Path,
    version: str | None = None,
) -> dict[str, Any]:
    if PILOT_IMAGE_COUNT <= 0 or SOURCE_IMAGE_COUNT < PILOT_IMAGE_COUNT:
        raise MaterializeError("invalid fixed source/pilot image counts")
    if source_root.is_symlink():
        raise MaterializeError("source root must not be a symlink")
    try:
        source_root = source_root.resolve(strict=True)
    except OSError as exc:
        raise MaterializeError(f"source root does not exist: {source_root}") from exc
    if not source_root.is_dir():
        raise MaterializeError("source root must be a directory")
    if source_receipt.is_symlink():
        raise MaterializeError("receive receipt path must not be a symlink")
    try:
        source_receipt = source_receipt.resolve(strict=True)
    except OSError as exc:
        raise MaterializeError(f"receive receipt does not exist: {source_receipt}") from exc
    output_root = _ensure_root(output_root, "output root")
    evidence_root = _ensure_root(evidence_root, "evidence root")
    if _paths_overlap(source_root, output_root) or _paths_overlap(source_root, evidence_root):
        raise MaterializeError("source, output, and evidence roots must not overlap")

    manifest_path = source_root / "manifest.jsonl"
    contract_path = source_root / "dataset.contract.json"
    manifest_raw = _read_regular_file(manifest_path, "source manifest", limit=128 * 1024**2)
    contract_raw = _read_regular_file(contract_path, "source contract", limit=2 * 1024**2)
    receipt_raw = _read_regular_file(source_receipt, "receive receipt", limit=2 * 1024**2)
    manifest_sha = _sha256_bytes(manifest_raw)
    contract_sha = _sha256_bytes(contract_raw)
    receipt_sha = _sha256_bytes(receipt_raw)
    rows, encoded_lines = _parse_manifest(manifest_raw)
    contract = _parse_contract(contract_raw, manifest_sha256=manifest_sha, rows=rows)
    _receipt, source_subject = _parse_receive_receipt(
        receipt_raw,
        receipt_path=source_receipt,
        source_root=source_root,
        contract_raw=contract_raw,
        manifest_raw=manifest_raw,
        contract=contract,
        rows=rows,
    )

    prefix_rows = rows[:PILOT_IMAGE_COUNT]
    prefix_manifest_raw = b"".join(encoded_lines[:PILOT_IMAGE_COUNT])
    prefix_manifest_sha = _sha256_bytes(prefix_manifest_raw)
    source_image_bytes = sum(row["size_bytes"] for row in rows)
    prefix_image_bytes = sum(row["size_bytes"] for row in prefix_rows)
    expected_source_files = {"manifest.jsonl", "dataset.contract.json"}
    expected_source_files.update(row["archive_relative_path"] for row in rows)
    _assert_tree_closure(source_root, expected_source_files, "source")

    pilot_contract = {
        "schema_version": 1,
        "kind": "otherimages_white_sync_prefix_pilot_dataset_v1",
        "source_semantics": "white_image_unlabeled_source_only",
        "analysis_only": True,
        "production_route_authorized": False,
        "source": {
            "package_subject_sha256": source_subject,
            "raw_version_root": str(source_root),
            "receive_receipt": {
                "path": str(source_receipt),
                "size_bytes": len(receipt_raw),
                "sha256": receipt_sha,
                "kind": "otherimages_white_sync_receive_receipt_v1",
            },
            "contract": {
                "path": "dataset.contract.json",
                "size_bytes": len(contract_raw),
                "sha256": contract_sha,
                "kind": contract["kind"],
            },
            "full_manifest": {
                "path": "manifest.jsonl",
                "size_bytes": len(manifest_raw),
                "sha256": manifest_sha,
                "image_count": SOURCE_IMAGE_COUNT,
                "image_bytes": source_image_bytes,
            },
        },
        "selection": {
            "policy": "exact_first_source_manifest_rows_v1",
            "prefix_count": PILOT_IMAGE_COUNT,
            "exact_byte_prefix_of_source_manifest": True,
            "source_selection_algorithm": contract["selection"]["algorithm"],
            "source_selection_salt": contract["selection"]["salt"],
        },
        "payload": {
            "images_root": "images",
            "manifest": "manifest.jsonl",
            "manifest_sha256": prefix_manifest_sha,
            "image_count": PILOT_IMAGE_COUNT,
            "image_bytes": prefix_image_bytes,
        },
        "labels": contract["labels"],
    }
    pilot_contract_raw = _json_bytes(pilot_contract)
    pilot_contract_sha = _sha256_bytes(pilot_contract_raw)
    subject_material = (
        "otherimages-white-prefix-pilot-subject-v1\n"
        f"{source_subject}\n{manifest_sha}\n{prefix_manifest_sha}\n{pilot_contract_sha}\n"
    ).encode("utf-8")
    pilot_subject = _sha256_bytes(subject_material)
    resolved_version = version or f"white-pilot-{PILOT_IMAGE_COUNT}-{pilot_subject[:12]}"
    if not VERSION_RE.fullmatch(resolved_version):
        raise MaterializeError("version must match [A-Za-z0-9][A-Za-z0-9._-]{0,127}")
    destination = output_root / resolved_version
    external_receipt = evidence_root / f"{resolved_version}.pilot.receipt.json"
    if _exists_or_symlink(destination):
        raise MaterializeError(f"pilot version already exists; refusing to overwrite: {destination}")
    if _exists_or_symlink(external_receipt):
        raise MaterializeError(f"pilot receipt already exists; refusing to overwrite: {external_receipt}")

    stage = output_root / f".{resolved_version}.stage-{uuid.uuid4().hex}"
    receipt_temp = evidence_root / f".{external_receipt.name}.stage-{uuid.uuid4().hex}"
    stage.mkdir(mode=0o700)
    (stage / "images").mkdir(mode=0o700)
    published = False
    try:
        _write_exclusive(stage / "manifest.jsonl", prefix_manifest_raw)
        _write_exclusive(stage / "dataset.contract.json", pilot_contract_raw)
        started = time.monotonic()
        verified_bytes = 0
        for position, row in enumerate(rows, 1):
            source_path = source_root.joinpath(*PurePosixPath(row["archive_relative_path"]).parts)
            destination_path = None
            if position <= PILOT_IMAGE_COUNT:
                destination_path = stage.joinpath(
                    *PurePosixPath(row["archive_relative_path"]).parts
                )
            _copy_or_hash_source(
                source_path,
                expected_size=row["size_bytes"],
                expected_sha256=row["sha256"],
                destination=destination_path,
            )
            verified_bytes += row["size_bytes"]
            if position % 250 == 0 or position == SOURCE_IMAGE_COUNT:
                elapsed = max(time.monotonic() - started, 0.001)
                print(
                    "WHITE_PREFIX_VERIFY_ALIVE "
                    f"verified={position}/{SOURCE_IMAGE_COUNT} "
                    f"copied={min(position, PILOT_IMAGE_COUNT)}/{PILOT_IMAGE_COUNT} "
                    f"bytes={verified_bytes} elapsed_s={elapsed:.1f} files_per_s={position / elapsed:.2f}",
                    flush=True,
                )

        # Reopen all authority metadata and tree membership after the long data pass.
        if (
            _sha256_bytes(_read_regular_file(manifest_path, "source manifest", limit=128 * 1024**2))
            != manifest_sha
            or _sha256_bytes(_read_regular_file(contract_path, "source contract", limit=2 * 1024**2))
            != contract_sha
            or _sha256_bytes(_read_regular_file(source_receipt, "receive receipt", limit=2 * 1024**2))
            != receipt_sha
        ):
            raise MaterializeError("source authority metadata changed during materialization")
        _assert_tree_closure(source_root, expected_source_files, "source")

        receipt = {
            "schema_version": 1,
            "kind": "otherimages_white_sync_prefix_pilot_receipt_v1",
            "status": "complete",
            "pilot_subject_sha256": pilot_subject,
            "source": {
                "package_subject_sha256": source_subject,
                "raw_version_root": str(source_root),
                "receive_receipt": {
                    "path": str(source_receipt),
                    "size_bytes": len(receipt_raw),
                    "sha256": receipt_sha,
                },
                "contract": {
                    "path": "dataset.contract.json",
                    "size_bytes": len(contract_raw),
                    "sha256": contract_sha,
                    "kind": contract["kind"],
                },
                "full_manifest": {
                    "path": "manifest.jsonl",
                    "size_bytes": len(manifest_raw),
                    "sha256": manifest_sha,
                    "image_count": SOURCE_IMAGE_COUNT,
                    "image_bytes": source_image_bytes,
                },
            },
            "prefix": {
                "policy": "exact_first_source_manifest_rows_v1",
                "manifest": {
                    "path": "manifest.jsonl",
                    "size_bytes": len(prefix_manifest_raw),
                    "sha256": prefix_manifest_sha,
                    "exact_byte_prefix_of_source_manifest": True,
                },
                "contract": {
                    "path": "dataset.contract.json",
                    "size_bytes": len(pilot_contract_raw),
                    "sha256": pilot_contract_sha,
                    "kind": pilot_contract["kind"],
                },
                "image_count": PILOT_IMAGE_COUNT,
                "image_bytes": prefix_image_bytes,
            },
            "validation": {
                "receive_receipt_contract_manifest_strict": True,
                "source_file_and_directory_closure_exact": True,
                "every_source_file_size_and_sha256_revalidated": True,
                "every_prefix_copy_size_and_sha256_verified": True,
                "source_files_written": False,
                "output_file_and_directory_closure_exact": True,
            },
            "publication": {
                "version": resolved_version,
                "pilot_root": str(destination),
                "internal_receipt": "pilot.receipt.json",
                "external_receipt": str(external_receipt),
                "brand_new": True,
                "atomic_exclusive_rename": True,
                "overwrite_performed": False,
            },
            "completed_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        }
        receipt_raw_out = _json_bytes(receipt)
        _write_exclusive(stage / "pilot.receipt.json", receipt_raw_out)
        _write_exclusive(receipt_temp, receipt_raw_out)
        expected_output_files = {"manifest.jsonl", "dataset.contract.json", "pilot.receipt.json"}
        expected_output_files.update(row["archive_relative_path"] for row in prefix_rows)
        _assert_tree_closure(stage, expected_output_files, "staged pilot")
        _atomic_rename_exclusive(stage, destination)
        published = True
        _atomic_rename_exclusive(receipt_temp, external_receipt)
        print(
            "WHITE_PREFIX_MATERIALIZE_OK "
            f"version={resolved_version} images={PILOT_IMAGE_COUNT} "
            f"pilot_subject_sha256={pilot_subject} root={destination}",
            flush=True,
        )
        return receipt
    finally:
        if not published and stage.exists():
            shutil.rmtree(stage)
        # If the directory was published but the external receipt rename failed,
        # retain the fully written hidden receipt as recovery evidence.  The
        # byte-identical internal receipt is already inside the atomic output.
        if not published and receipt_temp.exists():
            receipt_temp.unlink()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--source-receipt", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_BASE / "pilot")
    parser.add_argument("--evidence-root", type=Path, default=DEFAULT_BASE / "evidence")
    parser.add_argument("--version")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        receipt = materialize_pilot(
            source_root=args.source_root,
            source_receipt=args.source_receipt,
            output_root=args.output_root,
            evidence_root=args.evidence_root,
            version=args.version,
        )
    except (MaterializeError, OSError) as exc:
        print(f"WHITE_PREFIX_MATERIALIZE_FAILED {exc}", file=sys.stderr, flush=True)
        return 2
    print(json.dumps(receipt, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
