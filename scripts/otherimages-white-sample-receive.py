#!/usr/bin/env python3
"""Download, verify, and atomically publish a sealed white-image package.

The receiver is deliberately fail-closed.  It accepts only the v1 package
produced for the OtherImages white-image handoff, verifies the transport and
the complete payload closure, and publishes a brand-new immutable version
below ``raw/``.  It never merges into or replaces an existing version.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import re
import shutil
import stat
import tempfile
import unicodedata
import urllib.parse
import urllib.request
import uuid
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO


DEFAULT_BASE = Path("/Volumes/CodexData/white-input")
DEFAULT_MAX_BYTES = 30 * 1024**3
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
VERSION_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
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


class ReceiveError(RuntimeError):
    """The package failed a transport, schema, or content-integrity gate."""


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
            raise ReceiveError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ReceiveError(f"non-finite JSON number is forbidden: {value}")


def _load_json(raw: bytes, source: str) -> Any:
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ReceiveError(f"{source} is not valid UTF-8") from exc
    try:
        return json.loads(
            text,
            object_pairs_hook=_strict_object,
            parse_constant=_reject_constant,
        )
    except ReceiveError:
        raise
    except (json.JSONDecodeError, ValueError) as exc:
        raise ReceiveError(f"invalid JSON in {source}: {exc}") from exc


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _require_exact_keys(value: Any, expected: set[str], source: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ReceiveError(f"{source} must be a JSON object")
    actual = set(value)
    if actual != expected:
        raise ReceiveError(
            f"{source} keys differ: missing={sorted(expected - actual)!r} "
            f"extra={sorted(actual - expected)!r}"
        )
    return value


def _require_nonempty_string(value: Any, source: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ReceiveError(f"{source} must be a non-empty string")
    return value


def _require_nonnegative_int(value: Any, source: str) -> int:
    if not _is_int(value) or value < 0:
        raise ReceiveError(f"{source} must be a non-negative integer")
    return value


def _require_sha(value: Any, source: str) -> str:
    if not isinstance(value, str) or not SHA256_RE.fullmatch(value):
        raise ReceiveError(f"{source} must be a lowercase SHA256")
    return value


def _require_utc_timestamp(value: Any, source: str) -> str:
    text = _require_nonempty_string(value, source)
    parseable = text.replace("Z", "+00:00")
    # PowerShell/.NET round-trip (``o``) timestamps carry seven fractional
    # second digits, while Python's datetime parser accepts microseconds. Keep
    # the sealed spelling unchanged and trim only the temporary parse value.
    parseable = re.sub(
        r"(?<=\d)\.(\d{7,})(?=[+-]\d{2}:\d{2}$)",
        lambda match: "." + match.group(1)[:6],
        parseable,
    )
    try:
        parsed = dt.datetime.fromisoformat(parseable)
    except ValueError as exc:
        raise ReceiveError(f"{source} is not an ISO-8601 timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() != dt.timedelta(0):
        raise ReceiveError(f"{source} must include a UTC offset")
    return text


def _safe_relative_path(value: Any, source: str) -> str:
    text = _require_nonempty_string(value, source)
    if "\x00" in text or "\\" in text or text.startswith("/"):
        raise ReceiveError(f"unsafe path in {source}: {text!r}")
    if unicodedata.normalize("NFC", text) != text:
        raise ReceiveError(f"path in {source} must use Unicode NFC: {text!r}")
    path = PurePosixPath(text)
    if not path.parts or any(part in {"", ".", ".."} for part in path.parts):
        raise ReceiveError(f"unsafe path in {source}: {text!r}")
    if path.as_posix() != text:
        raise ReceiveError(f"non-canonical path in {source}: {text!r}")
    if re.match(r"^[A-Za-z]:", path.parts[0]):
        raise ReceiveError(f"drive-qualified path in {source}: {text!r}")
    return path.as_posix()


def _safe_zip_relative_path(value: Any, source: str) -> str:
    """Validate and canonicalize one ZIP name without weakening the manifest.

    ``ZipFile.CreateFromDirectory`` on Windows writes directory separators as
    backslashes.  Those names are safe to accept only after treating the
    separator convention as syntax: a name may use forward slashes or
    backslashes, never both, and every component must already be canonical.
    The returned name is always POSIX-style so the ZIP file closure can be
    compared byte-for-byte with the forward-slash manifest paths.
    """

    text = _require_nonempty_string(value, source)
    if "\x00" in text:
        raise ReceiveError(f"unsafe path in {source}: {text!r}")
    if unicodedata.normalize("NFC", text) != text:
        raise ReceiveError(f"path in {source} must use Unicode NFC: {text!r}")
    if text.startswith(("/", "\\")):
        raise ReceiveError(f"absolute/UNC path in {source}: {text!r}")
    if re.match(r"^[A-Za-z]:", text):
        raise ReceiveError(f"drive-qualified path in {source}: {text!r}")

    has_forward = "/" in text
    has_backward = "\\" in text
    if has_forward and has_backward:
        raise ReceiveError(f"mixed ZIP path separators in {source}: {text!r}")
    separator = "\\" if has_backward else "/"
    parts = text.split(separator)
    if not parts or any(part in {"", ".", ".."} for part in parts):
        raise ReceiveError(f"unsafe path in {source}: {text!r}")

    canonical = "/".join(parts)
    # Run the ordinary manifest-path policy over the normalized spelling too.
    # This deliberately keeps ZIP compatibility narrower than filesystem path
    # compatibility and makes the later closure comparison unambiguous.
    return _safe_relative_path(canonical, source)


def _collision_key(path: str) -> str:
    return unicodedata.normalize("NFC", path).casefold()


class _HttpOnlyRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req: Any, fp: Any, code: int, msg: str, headers: Any, newurl: str) -> Any:
        parsed = urllib.parse.urlsplit(newurl)
        if parsed.scheme not in {"http", "https"}:
            raise ReceiveError(f"redirect to forbidden URL scheme: {parsed.scheme!r}")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def _download(
    url: str,
    destination: Path,
    expected_sha256: str,
    expected_bytes: int,
    timeout_seconds: float,
) -> None:
    parsed = urllib.parse.urlsplit(url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ReceiveError("download URL must be an absolute HTTP(S) URL")
    opener = urllib.request.build_opener(_HttpOnlyRedirect())
    request = urllib.request.Request(url, headers={"User-Agent": "white-package-receiver/1"})
    digest = hashlib.sha256()
    observed = 0
    try:
        response = opener.open(request, timeout=timeout_seconds)
        try:
            content_length = response.headers.get("Content-Length")
            if content_length is not None:
                try:
                    declared = int(content_length)
                except ValueError as exc:
                    raise ReceiveError("invalid Content-Length from download server") from exc
                if declared != expected_bytes:
                    raise ReceiveError(
                        f"download Content-Length mismatch: expected {expected_bytes}, got {declared}"
                    )
            with destination.open("xb") as output:
                while True:
                    chunk = response.read(min(1024 * 1024, expected_bytes - observed + 1))
                    if not chunk:
                        break
                    observed += len(chunk)
                    if observed > expected_bytes:
                        raise ReceiveError("download exceeded expected byte count")
                    digest.update(chunk)
                    output.write(chunk)
                output.flush()
                os.fsync(output.fileno())
        finally:
            response.close()
    except ReceiveError:
        raise
    except Exception as exc:
        raise ReceiveError(f"download failed: {exc}") from exc
    if observed != expected_bytes:
        raise ReceiveError(f"download size mismatch: expected {expected_bytes}, got {observed}")
    actual_sha = digest.hexdigest()
    if actual_sha != expected_sha256:
        raise ReceiveError(
            f"download SHA256 mismatch: expected {expected_sha256}, got {actual_sha}"
        )


def _zip_entries(
    archive: zipfile.ZipFile,
    max_entries: int,
    max_uncompressed_bytes: int,
) -> dict[str, zipfile.ZipInfo]:
    entries: dict[str, zipfile.ZipInfo] = {}
    collision_names: dict[str, str] = {}
    infos = archive.infolist()
    if len(infos) > max_entries:
        raise ReceiveError(f"ZIP has too many entries: {len(infos)} > {max_entries}")
    total = 0
    for info in infos:
        raw_name = info.filename
        mode = (info.external_attr >> 16) & 0xFFFF
        declares_directory = (
            info.is_dir()
            or raw_name.endswith(("/", "\\"))
            or stat.S_ISDIR(mode)
            or bool(info.external_attr & 0x10)
        )
        if declares_directory:
            raise ReceiveError(f"ZIP directory entry is forbidden: {raw_name!r}")
        safe = _safe_zip_relative_path(raw_name, f"ZIP entry {raw_name!r}")
        if stat.S_ISLNK(mode):
            raise ReceiveError(f"ZIP symlink is forbidden: {raw_name!r}")
        if mode and stat.S_IFMT(mode) not in {0, stat.S_IFREG, stat.S_IFDIR}:
            raise ReceiveError(f"non-regular ZIP entry is forbidden: {raw_name!r}")
        key = _collision_key(safe)
        if key in collision_names:
            raise ReceiveError(
                f"duplicate/case-NFC ZIP path collision: {raw_name!r} and {collision_names[key]!r}"
            )
        collision_names[key] = raw_name
        entries[safe] = info
        total += info.file_size
        if total > max_uncompressed_bytes:
            raise ReceiveError(
                f"ZIP uncompressed payload exceeds limit: {total} > {max_uncompressed_bytes}"
            )
    return entries


def _read_bounded(archive: zipfile.ZipFile, info: zipfile.ZipInfo, limit: int, source: str) -> bytes:
    if info.file_size > limit:
        raise ReceiveError(f"{source} is too large: {info.file_size} > {limit}")
    with archive.open(info, "r") as stream:
        data = stream.read(limit + 1)
    if len(data) != info.file_size or len(data) > limit:
        raise ReceiveError(f"{source} size changed while reading")
    return data


def _parse_manifest(raw: bytes) -> list[dict[str, Any]]:
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ReceiveError("manifest.jsonl is not valid UTF-8") from exc
    if not text or not text.endswith("\n"):
        raise ReceiveError("manifest.jsonl must be non-empty and newline terminated")
    rows: list[dict[str, Any]] = []
    archive_keys: dict[str, str] = {}
    source_keys: dict[str, str] = {}
    for line_number, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            raise ReceiveError(f"blank manifest line at {line_number}")
        row = _require_exact_keys(
            _load_json(line.encode("utf-8"), f"manifest line {line_number}"),
            MANIFEST_KEYS,
            f"manifest line {line_number}",
        )
        if row["schema_version"] != 1 or row["index"] != line_number:
            raise ReceiveError(f"manifest line {line_number} has invalid schema_version/index")
        _require_sha(row["selection_key"], f"manifest line {line_number}.selection_key")
        source_path = _safe_relative_path(
            row["source_relative_path"], f"manifest line {line_number}.source_relative_path"
        )
        archive_path = _safe_relative_path(
            row["archive_relative_path"], f"manifest line {line_number}.archive_relative_path"
        )
        if not archive_path.startswith("images/") or archive_path == "images/":
            raise ReceiveError(f"manifest line {line_number} must point below images/")
        size = _require_nonnegative_int(row["size_bytes"], f"manifest line {line_number}.size_bytes")
        if size == 0:
            raise ReceiveError(f"manifest line {line_number} describes an empty image")
        _require_sha(row["sha256"], f"manifest line {line_number}.sha256")
        _require_utc_timestamp(
            row["source_last_write_utc"], f"manifest line {line_number}.source_last_write_utc"
        )
        for seen, candidate, label in (
            (archive_keys, archive_path, "archive"),
            (source_keys, source_path, "source"),
        ):
            key = _collision_key(candidate)
            if key in seen:
                raise ReceiveError(
                    f"manifest {label} path collision: {candidate!r} and {seen[key]!r}"
                )
            seen[key] = candidate
        rows.append(row)
    return rows


def _parse_contract(raw: bytes, manifest_sha256: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    contract = _require_exact_keys(
        _load_json(raw, "dataset.contract.json"),
        CONTRACT_KEYS,
        "dataset.contract.json",
    )
    if contract["schema_version"] != 1:
        raise ReceiveError("unsupported contract schema_version")
    if contract["kind"] != "otherimages_white_sync_dataset_v1":
        raise ReceiveError("unexpected contract kind")
    if contract["source_semantics"] != "white_image_unlabeled_source_only":
        raise ReceiveError("unexpected source_semantics")
    _require_nonempty_string(contract["source_root_at_capture"], "contract.source_root_at_capture")
    if contract["source_files_modified"] is not False:
        raise ReceiveError("contract must assert source_files_modified=false")
    _require_utc_timestamp(contract["captured_utc"], "contract.captured_utc")
    _require_nonempty_string(contract["host"], "contract.host")
    _require_nonempty_string(contract["powershell_version"], "contract.powershell_version")

    selection = _require_exact_keys(contract["selection"], SELECTION_KEYS, "contract.selection")
    if selection["algorithm"] != "sha256_utf8_salt_lf_normalized_relative_path_sort_v1":
        raise ReceiveError("unexpected selection algorithm")
    _require_nonempty_string(selection["salt"], "contract.selection.salt")
    if selection["normalized_path"] != "unicode_nfc_lowercase_forward_slashes":
        raise ReceiveError("unexpected normalized_path policy")
    extensions = selection["supported_extensions"]
    if not isinstance(extensions, list) or not extensions or any(
        not isinstance(item, str) or not item.startswith(".") for item in extensions
    ):
        raise ReceiveError("contract.selection.supported_extensions is invalid")
    candidate_count = _require_nonnegative_int(selection["candidate_count"], "candidate_count")
    requested_count = _require_nonnegative_int(selection["requested_count"], "requested_count")
    selected_count = _require_nonnegative_int(selection["selected_count"], "selected_count")
    if selection["prefix_stable_when_candidate_set_and_salt_are_unchanged"] is not True:
        raise ReceiveError("selection prefix-stability assertion is missing")
    if selected_count != len(rows) or requested_count != len(rows) or candidate_count < len(rows):
        raise ReceiveError("contract selection counts do not bind the manifest")

    payload = _require_exact_keys(contract["payload"], PAYLOAD_KEYS, "contract.payload")
    if payload["images_root"] != "images" or payload["manifest"] != "manifest.jsonl":
        raise ReceiveError("unexpected payload paths")
    if _require_sha(payload["manifest_sha256"], "contract.payload.manifest_sha256") != manifest_sha256:
        raise ReceiveError("contract manifest SHA256 mismatch")
    if _require_nonnegative_int(payload["image_count"], "image_count") != len(rows):
        raise ReceiveError("contract image_count mismatch")
    image_bytes = sum(row["size_bytes"] for row in rows)
    if _require_nonnegative_int(payload["image_bytes"], "image_bytes") != image_bytes:
        raise ReceiveError("contract image_bytes mismatch")

    labels = _require_exact_keys(contract["labels"], LABEL_KEYS, "contract.labels")
    if labels["human_labels_present"] is not False or labels["paddle_teacher_labels_present"] is not False:
        raise ReceiveError("source-only package unexpectedly claims labels")
    if labels["intended_next_step"] != "paddle_teacher_generation_then_frozen_split_training_and_validation":
        raise ReceiveError("unexpected intended_next_step")
    return contract


def _open_output(path: Path) -> BinaryIO:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    return os.fdopen(os.open(path, flags, 0o600), "wb")


def _extract_and_verify(
    archive: zipfile.ZipFile,
    entries: dict[str, zipfile.ZipInfo],
    rows: list[dict[str, Any]],
    manifest_raw: bytes,
    contract_raw: bytes,
    stage: Path,
) -> None:
    expected_files = {"manifest.jsonl", "dataset.contract.json"}
    expected_files.update(row["archive_relative_path"] for row in rows)
    if set(entries) != expected_files:
        raise ReceiveError(
            "ZIP file closure differs from manifest: "
            f"missing={sorted(expected_files - set(entries))[:10]!r} "
            f"extra={sorted(set(entries) - expected_files)[:10]!r}"
        )
    stage.mkdir(mode=0o700)
    (stage / "images").mkdir(mode=0o700)
    (stage / "manifest.jsonl").write_bytes(manifest_raw)
    (stage / "dataset.contract.json").write_bytes(contract_raw)
    for row in rows:
        archive_name = row["archive_relative_path"]
        info = entries[archive_name]
        if info.file_size != row["size_bytes"]:
            raise ReceiveError(f"ZIP/manifest size mismatch: {row['archive_relative_path']}")
        relative = PurePosixPath(row["archive_relative_path"])
        target = stage.joinpath(*relative.parts)
        target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        digest = hashlib.sha256()
        observed = 0
        with archive.open(info, "r") as source, _open_output(target) as output:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                observed += len(chunk)
                if observed > row["size_bytes"]:
                    raise ReceiveError(f"image exceeded manifest size: {row['archive_relative_path']}")
                digest.update(chunk)
                output.write(chunk)
            output.flush()
            os.fsync(output.fileno())
        if observed != row["size_bytes"] or digest.hexdigest() != row["sha256"]:
            raise ReceiveError(f"image content binding failed: {row['archive_relative_path']}")


def _write_json_exclusive(path: Path, value: dict[str, Any]) -> None:
    payload = (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")
    with _open_output(path) as output:
        output.write(payload)
        output.flush()
        os.fsync(output.fileno())


def receive_package(
    *,
    url: str,
    expected_archive_sha256: str,
    expected_archive_bytes: int,
    incoming_root: Path,
    raw_root: Path,
    evidence_root: Path,
    version: str | None = None,
    timeout_seconds: float = 60.0,
    max_entries: int = 100_100,
    max_archive_bytes: int = DEFAULT_MAX_BYTES,
    max_uncompressed_bytes: int = DEFAULT_MAX_BYTES,
) -> dict[str, Any]:
    expected_archive_sha256 = expected_archive_sha256.lower()
    _require_sha(expected_archive_sha256, "expected archive SHA256")
    if not _is_int(expected_archive_bytes) or expected_archive_bytes <= 0:
        raise ReceiveError("expected archive bytes must be a positive integer")
    if (
        timeout_seconds <= 0
        or max_entries < 3
        or max_archive_bytes <= 0
        or max_uncompressed_bytes <= 0
    ):
        raise ReceiveError("receiver limits must be positive")
    if expected_archive_bytes > max_archive_bytes:
        raise ReceiveError(
            f"expected archive exceeds limit: {expected_archive_bytes} > {max_archive_bytes}"
        )

    incoming_root = incoming_root.resolve()
    raw_root = raw_root.resolve()
    evidence_root = evidence_root.resolve()
    for root in (incoming_root, raw_root, evidence_root):
        root.mkdir(parents=True, exist_ok=True)
        if root.is_symlink():
            raise ReceiveError(f"root must not be a symlink: {root}")

    run_id = f"receive-{dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{uuid.uuid4().hex[:12]}"
    run_root = incoming_root / run_id
    run_root.mkdir(mode=0o700)
    archive_path = run_root / "package.zip"
    stage: Path | None = None
    try:
        _download(url, archive_path, expected_archive_sha256, expected_archive_bytes, timeout_seconds)
        with zipfile.ZipFile(archive_path, "r") as archive:
            entries = _zip_entries(archive, max_entries, max_uncompressed_bytes)
            manifest_info = entries.get("manifest.jsonl")
            contract_info = entries.get("dataset.contract.json")
            if manifest_info is None or contract_info is None:
                raise ReceiveError("ZIP must contain root manifest.jsonl and dataset.contract.json")
            manifest_raw = _read_bounded(
                archive, manifest_info, 64 * 1024**2, "manifest.jsonl"
            )
            contract_raw = _read_bounded(
                archive, contract_info, 1024 * 1024, "dataset.contract.json"
            )
            manifest_sha = hashlib.sha256(manifest_raw).hexdigest()
            contract_sha = hashlib.sha256(contract_raw).hexdigest()
            rows = _parse_manifest(manifest_raw)
            if not rows:
                raise ReceiveError("manifest contains no images")
            contract = _parse_contract(contract_raw, manifest_sha, rows)
            subject = hashlib.sha256(f"{manifest_sha}\n{contract_sha}\n".encode()).hexdigest()
            resolved_version = version or f"white-{len(rows)}-{subject[:12]}"
            if not VERSION_RE.fullmatch(resolved_version):
                raise ReceiveError("version must match [A-Za-z0-9][A-Za-z0-9._-]{0,127}")
            destination = raw_root / resolved_version
            receipt_path = evidence_root / f"{resolved_version}.receive.receipt.json"
            if destination.exists() or destination.is_symlink():
                raise ReceiveError(f"raw version already exists; refusing to overwrite: {destination}")
            if receipt_path.exists() or receipt_path.is_symlink():
                raise ReceiveError(f"receipt already exists; refusing to overwrite: {receipt_path}")
            stage = raw_root / f".{resolved_version}.stage-{uuid.uuid4().hex}"
            _extract_and_verify(archive, entries, rows, manifest_raw, contract_raw, stage)

        receipt = {
            "schema_version": 1,
            "kind": "otherimages_white_sync_receive_receipt_v1",
            "status": "complete",
            "download": {
                "url": url,
                "archive_path": str(archive_path),
                "size_bytes": expected_archive_bytes,
                "sha256": expected_archive_sha256,
            },
            "package_subject_sha256": subject,
            "contract": {
                "path": "dataset.contract.json",
                "size_bytes": len(contract_raw),
                "sha256": contract_sha,
                "kind": contract["kind"],
            },
            "manifest": {
                "path": "manifest.jsonl",
                "size_bytes": len(manifest_raw),
                "sha256": manifest_sha,
            },
            "verified_payload": {
                "image_count": len(rows),
                "image_bytes": sum(row["size_bytes"] for row in rows),
                "every_file_size_and_sha256_verified": True,
                "archive_file_closure_exact": True,
            },
            "publication": {
                "version": resolved_version,
                "raw_version_root": str(destination),
                "brand_new": True,
                "atomic_rename": True,
            },
            "completed_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        }
        receipt_temp = evidence_root / f".{receipt_path.name}.stage-{uuid.uuid4().hex}"
        _write_json_exclusive(receipt_temp, receipt)
        os.replace(stage, destination)
        stage = None
        try:
            os.replace(receipt_temp, receipt_path)
        except Exception:
            # Publication is immutable; preserve the fully written receipt with an
            # unmistakable recovery name rather than deleting either artifact.
            raise
        return receipt
    except (zipfile.BadZipFile, OSError) as exc:
        if isinstance(exc, ReceiveError):
            raise
        raise ReceiveError(f"package receive failed: {exc}") from exc
    finally:
        if stage is not None and stage.exists():
            shutil.rmtree(stage)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True)
    parser.add_argument("--expected-archive-sha256", required=True)
    parser.add_argument("--expected-archive-bytes", required=True, type=int)
    parser.add_argument("--incoming-root", type=Path, default=DEFAULT_BASE / "incoming")
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_BASE / "raw")
    parser.add_argument("--evidence-root", type=Path, default=DEFAULT_BASE / "evidence")
    parser.add_argument("--version")
    parser.add_argument("--timeout-seconds", type=float, default=60.0)
    parser.add_argument("--max-entries", type=int, default=100_100)
    parser.add_argument("--max-archive-bytes", type=int, default=DEFAULT_MAX_BYTES)
    parser.add_argument("--max-uncompressed-bytes", type=int, default=DEFAULT_MAX_BYTES)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        receipt = receive_package(
            url=args.url,
            expected_archive_sha256=args.expected_archive_sha256,
            expected_archive_bytes=args.expected_archive_bytes,
            incoming_root=args.incoming_root,
            raw_root=args.raw_root,
            evidence_root=args.evidence_root,
            version=args.version,
            timeout_seconds=args.timeout_seconds,
            max_entries=args.max_entries,
            max_archive_bytes=args.max_archive_bytes,
            max_uncompressed_bytes=args.max_uncompressed_bytes,
        )
    except ReceiveError as exc:
        print(f"WHITE_RECEIVE_FAILED error={exc}", file=os.sys.stderr)
        return 2
    print(json.dumps(receipt, ensure_ascii=False, sort_keys=True))
    print(
        "WHITE_RECEIVE_OK "
        f"version={receipt['publication']['version']} "
        f"images={receipt['verified_payload']['image_count']} "
        f"subject={receipt['package_subject_sha256']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
