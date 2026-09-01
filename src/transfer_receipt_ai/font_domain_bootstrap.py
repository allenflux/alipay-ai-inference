"""Bootstrap weak font-domain labels from existing OCR pseudo-label exports.

This adapter is deliberately separate from the deployed OCR pipeline.  It
turns a frozen ``pseudo_labels.jsonl`` plus its source/result provenance into
the strict document manifest consumed by :mod:`font_domain_dataset`.  Device
platform is only a weak pseudo label: uncertain, conflicting, and low
confidence device decisions are recorded as rejections rather than promoted
to a training class.
"""

from __future__ import annotations

import ctypes
import errno
import hashlib
import io
import json
import math
import os
import re
import secrets
import shutil
import stat
import sys
import tempfile
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import numpy as np
from PIL import Image, ImageOps

from .font_domain import SCHEMA_VERSION
from .font_domain_dataset import DOCUMENT_KIND, load_font_domain_dataset


BOOTSTRAP_KIND: Final[str] = "receipt_font_domain_bootstrap_v1"
REJECTION_KIND: Final[str] = "receipt_font_domain_bootstrap_rejection_v1"
LABEL_PROVENANCE: Final[str] = "device_platform_weak_pseudo_v1"
DEFAULT_SPLIT_SEED: Final[str] = "font-domain-bootstrap-v1"

_FIELD_TO_ROLE: Final[dict[str, str]] = {
    "amount": "amount",
    "recipient_field": "recipient",
    "transfer_status": "transfer_status",
    "payment_method_field": "payment_method",
}
_IGNORED_FIELDS: Final[frozenset[str]] = frozenset({"time", "status_bar"})
_PLATFORM_TO_DOMAIN: Final[dict[str, str]] = {
    "ios": "ios_alipay_device_pseudo_v1",
    "android": "android_alipay_device_pseudo_v1",
}
_SAFE_SUFFIX = re.compile(r"^\.[A-Za-z0-9]{1,10}$")

MAXIMUM_RECORDS_BYTES: Final[int] = 512 * 1024 * 1024
MAXIMUM_LINE_BYTES: Final[int] = 4 * 1024 * 1024
MAXIMUM_RECORDS: Final[int] = 1_000_000
MAXIMUM_RESULT_BYTES: Final[int] = 16 * 1024 * 1024
MAXIMUM_SOURCE_BYTES: Final[int] = 64 * 1024 * 1024
MAXIMUM_CROP_BYTES: Final[int] = 32 * 1024 * 1024
MAXIMUM_SOURCE_PIXELS: Final[int] = 50_000_000
MAXIMUM_CROP_PIXELS: Final[int] = 20_000_000


def _strict_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in pairs:
        if key in output:
            raise ValueError(f"duplicate JSON key {key!r}")
        output[key] = value
    return output


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON number {value!r}")


def _require_finite_tree(value: object, *, location: str) -> None:
    """Reject finite-looking JSON exponents that Python rounded to infinity."""

    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{location}: non-finite JSON number")
    if isinstance(value, Mapping):
        for key, child in value.items():
            _require_finite_tree(child, location=f"{location}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _require_finite_tree(child, location=f"{location}[{index}]")


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _stable_id(prefix: str, *parts: str) -> str:
    digest = hashlib.sha256("\0".join(parts).encode("utf-8")).hexdigest()
    return f"{prefix}-{digest}"


def _json_bytes(value: object, *, pretty: bool = False) -> bytes:
    if pretty:
        encoded = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
    else:
        encoded = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    return (encoded + "\n").encode("utf-8")


def _jsonl_bytes(rows: list[dict[str, object]]) -> bytes:
    return b"".join(_json_bytes(row) for row in rows)


def _read_snapshot(
    path: Path,
    *,
    maximum_bytes: int,
    description: str,
    expected_size: int | None = None,
    expected_mtime_ns: int | None = None,
    expected_sha256: str | None = None,
) -> bytes:
    if maximum_bytes < 1:
        raise ValueError("maximum_bytes must be positive")
    try:
        with path.open("rb") as stream:
            before = os.fstat(stream.fileno())
            if not stat.S_ISREG(before.st_mode):
                raise ValueError(f"{description} is not a regular file: {path}")
            if expected_size is not None and before.st_size != expected_size:
                raise ValueError(f"{description} size changed before final snapshot: {path}")
            if expected_mtime_ns is not None and before.st_mtime_ns != expected_mtime_ns:
                raise ValueError(f"{description} mtime changed before final snapshot: {path}")
            if before.st_size > maximum_bytes:
                raise ValueError(
                    f"{description} exceeds the {maximum_bytes}-byte limit: {path}"
                )
            data = stream.read(maximum_bytes + 1)
            after = os.fstat(stream.fileno())
    except OSError as error:
        raise ValueError(f"unable to read {description} {path}: {error}") from error
    if len(data) > maximum_bytes:
        raise ValueError(f"{description} exceeds the {maximum_bytes}-byte limit: {path}")
    identity_before = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
    identity_after = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    if identity_before != identity_after or len(data) != before.st_size:
        raise ValueError(f"{description} changed while it was read: {path}")
    if expected_sha256 is not None and _sha256(data) != expected_sha256:
        raise ValueError(f"{description} content changed before final snapshot: {path}")
    return data


def _decode_utf8_json(data: bytes, *, location: str) -> Mapping[str, Any]:
    try:
        text = data.decode("utf-8-sig")
    except UnicodeDecodeError as error:
        raise ValueError(f"{location}: JSON must be UTF-8: {error}") from None
    try:
        value: Any = json.loads(
            text,
            object_pairs_hook=_strict_json_object,
            parse_constant=_reject_json_constant,
        )
    except (json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"{location}: invalid JSON: {error}") from None
    if not isinstance(value, Mapping):
        raise ValueError(f"{location}: JSON root must be an object")
    _require_finite_tree(value, location=location)
    return value


@dataclass
class _RecordStreamState:
    records: int = 0
    sha256: str | None = None


def _iter_records(
    path: Path,
    state: _RecordStreamState,
) -> Iterator[tuple[int, Mapping[str, Any]]]:
    """Stream a stable JSONL snapshot without retaining its raw contents."""

    digest = hashlib.sha256()
    total_bytes = 0
    try:
        with path.open("rb") as stream:
            before = os.fstat(stream.fileno())
            if not stat.S_ISREG(before.st_mode):
                raise ValueError(f"pseudo-label manifest is not a regular file: {path}")
            if before.st_size > MAXIMUM_RECORDS_BYTES:
                raise ValueError(
                    f"pseudo-label manifest exceeds the {MAXIMUM_RECORDS_BYTES}-byte limit: {path}"
                )
            line_number = 0
            while True:
                raw_line = stream.readline(MAXIMUM_LINE_BYTES + 3)
                if not raw_line:
                    break
                line_number += 1
                total_bytes += len(raw_line)
                if total_bytes > MAXIMUM_RECORDS_BYTES:
                    raise ValueError(
                        f"pseudo-label manifest exceeds the {MAXIMUM_RECORDS_BYTES}-byte limit: {path}"
                    )
                digest.update(raw_line)
                content = raw_line.rstrip(b"\r\n")
                if len(content) > MAXIMUM_LINE_BYTES:
                    raise ValueError(
                        f"{path}:{line_number}: record exceeds the "
                        f"{MAXIMUM_LINE_BYTES}-byte limit"
                    )
                try:
                    text = content.decode("utf-8-sig" if line_number == 1 else "utf-8")
                except UnicodeDecodeError as error:
                    raise ValueError(
                        f"{path}:{line_number}: pseudo-label manifest must be UTF-8: {error}"
                    ) from None
                if not text.strip():
                    continue
                if state.records >= MAXIMUM_RECORDS:
                    raise ValueError(
                        f"{path}: manifest exceeds the {MAXIMUM_RECORDS}-record limit"
                    )
                try:
                    value: Any = json.loads(
                        text,
                        object_pairs_hook=_strict_json_object,
                        parse_constant=_reject_json_constant,
                    )
                except (json.JSONDecodeError, ValueError) as error:
                    raise ValueError(
                        f"{path}:{line_number}: invalid JSON: {error}"
                    ) from None
                if not isinstance(value, Mapping):
                    raise ValueError(f"{path}:{line_number}: record must be an object")
                _require_finite_tree(value, location=f"{path}:{line_number}")
                state.records += 1
                yield line_number, value
            after = os.fstat(stream.fileno())
    except OSError as error:
        raise ValueError(f"unable to read pseudo-label manifest {path}: {error}") from error
    identity_before = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
    identity_after = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    if identity_before != identity_after or total_bytes != before.st_size:
        raise ValueError(f"pseudo-label manifest changed while it was read: {path}")
    if state.records == 0:
        raise ValueError(f"pseudo-label manifest contains no records: {path}")
    state.sha256 = digest.hexdigest()


def _absolute_file(value: object, *, description: str) -> Path:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{description} must be a non-empty unpadded absolute path")
    lexical = Path(value).expanduser()
    if not lexical.is_absolute():
        raise ValueError(f"{description} must be an absolute path")
    try:
        resolved = lexical.resolve(strict=True)
    except OSError as error:
        raise ValueError(f"unable to resolve {description} {lexical}: {error}") from error
    if not resolved.is_file():
        raise ValueError(f"{description} is not a file: {resolved}")
    return resolved


def _cached_absolute_file(
    value: object,
    *,
    cache: dict[str, Path],
    description: str,
) -> Path:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{description} must be a non-empty unpadded absolute path")
    cached = cache.get(value)
    if cached is not None:
        return cached
    resolved = _absolute_file(value, description=description)
    cache[value] = resolved
    return resolved


@dataclass(frozen=True)
class _SafeCropPath:
    relative: str
    lexical: Path

    def recheck(self, root: Path) -> Path:
        try:
            current = self.lexical.resolve(strict=True)
        except OSError as error:
            raise ValueError(f"unable to re-resolve crop {self.lexical}: {error}") from error
        try:
            current.relative_to(root)
        except ValueError:
            raise ValueError(f"crop escapes the records root: {self.lexical}") from None
        if not current.is_file():
            raise ValueError(f"crop is not a regular file: {self.lexical}")
        return current


def _safe_crop_path(root: Path, value: object, *, location: str) -> _SafeCropPath:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{location}: image must be a non-empty unpadded relative path")
    if (
        "\\" in value
        or value.startswith("/")
        or ":" in value.split("/", 1)[0]
        or any(part in {"", ".", ".."} for part in value.split("/"))
    ):
        raise ValueError(f"{location}: image must be a safe POSIX relative path")
    lexical = root.joinpath(*value.split("/"))
    suffix = lexical.suffix
    if not _SAFE_SUFFIX.fullmatch(suffix):
        raise ValueError(f"{location}: image suffix is unsafe or unsupported: {suffix!r}")
    # Existence and containment are checked only for the deterministically
    # selected pilot rows.  This avoids hundreds of thousands of unnecessary
    # filesystem resolutions while preserving fail-closed checks before any
    # output directory is published.
    return _SafeCropPath(relative=value, lexical=lexical)


def _finite_probability(value: object, *, description: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{description} must be numeric")
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError(f"{description} must be finite and between 0 and 1")
    return result


def _decode_rgb(
    path: Path,
    data: bytes,
    *,
    maximum_pixels: int,
    description: str,
) -> np.ndarray:
    try:
        with Image.open(io.BytesIO(data)) as opened:
            width, height = (int(item) for item in opened.size)
            if width < 1 or height < 1 or width * height > maximum_pixels:
                raise ValueError(
                    f"decoded dimensions {width}x{height} exceed the {maximum_pixels}-pixel limit"
                )
            rgb = np.asarray(ImageOps.exif_transpose(opened).convert("RGB"), dtype=np.uint8)
    except (OSError, ValueError) as error:
        raise ValueError(f"unable to decode {description} {path}: {error}") from error
    if rgb.ndim != 3 or rgb.shape[2] != 3 or min(rgb.shape[:2]) < 1:
        raise ValueError(f"decoded {description} is empty or invalid: {path}")
    return np.ascontiguousarray(rgb)


def _pixel_sha256(rgb: np.ndarray) -> str:
    pixels = np.ascontiguousarray(rgb, dtype=np.uint8)
    digest = hashlib.sha256()
    digest.update(str(tuple(int(value) for value in pixels.shape)).encode("ascii"))
    digest.update(b"\0uint8\0RGB\0")
    digest.update(pixels.tobytes(order="C"))
    return digest.hexdigest()


def _legacy_crop_sha256(rgb: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(str(rgb.shape).encode("ascii"))
    digest.update(rgb.tobytes(order="C"))
    return digest.hexdigest()


def _normalise_visible_text(value: object, *, location: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{location}: text must be a string")
    text = unicodedata.normalize("NFC", value).strip()
    if not text or any(not character.isprintable() for character in text):
        raise ValueError(f"{location}: text must be non-empty and printable")
    if len(text) > 1024:
        raise ValueError(f"{location}: text exceeds the 1024-character limit")
    return text


def _normalise_content_text(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def _producer_group_id(value: object, *, location: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{location}: group_id must be a string")
    group_id = unicodedata.normalize("NFC", value)
    if (
        not group_id
        or group_id != group_id.strip()
        or len(group_id) > 1024
        or any(not character.isprintable() for character in group_id)
    ):
        raise ValueError(
            f"{location}: group_id must be a non-empty, printable, unpadded string"
        )
    return group_id


@dataclass(frozen=True)
class _InputRegion:
    line_number: int
    role: str
    text: str
    crop: _SafeCropPath
    detector_score: float
    paddle_confidence: float
    tie_breaker: str
    expected_legacy_crop_sha256: str | None

    @property
    def selection_key(self) -> tuple[float, float, str]:
        return (self.detector_score, self.paddle_confidence, self.tie_breaker)


@dataclass
class _GroupAccumulator:
    source: Path
    result_json: Path
    producer_group_id: str
    regions: dict[str, _InputRegion]


@dataclass(frozen=True)
class _ResultBinding:
    source: Path
    result_sha256: str
    platform: str | None
    confidence: float | None
    device_rejection: str | None


@dataclass
class _Candidate:
    document_id: str
    source_group_id: str
    content_group_id: str
    source: Path
    result_json: Path
    source_sha256: str
    result_sha256: str
    platform: str
    confidence: float
    domain: str
    producer_group_id: str
    regions: dict[str, _InputRegion]
    expected_source_size: int
    expected_source_mtime_ns: int
    split: str | None = None


def _result_source(payload: Mapping[str, Any], result_path: Path) -> Path:
    value = payload.get("source")
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{result_path}: result has no source path")
    lexical = Path(value)
    if not lexical.is_absolute():
        lexical = result_path.parent / lexical
    try:
        resolved = lexical.resolve(strict=True)
    except OSError as error:
        raise ValueError(f"{result_path}: unable to resolve result source: {error}") from error
    if not resolved.is_file():
        raise ValueError(f"{result_path}: result source is not a file: {resolved}")
    return resolved


def _device_rejection(
    payload: Mapping[str, Any],
    *,
    minimum_confidence: float,
) -> tuple[str | None, float | None, str | None]:
    device = payload.get("device")
    if not isinstance(device, Mapping):
        return None, None, "device_metadata_missing_or_invalid"
    platform = device.get("platform")
    confidence_value = device.get("confidence")
    conflict = device.get("device_prior_conflict")
    if not isinstance(platform, str) or isinstance(conflict, bool) is False:
        return None, None, "device_metadata_missing_or_invalid"
    try:
        confidence = _finite_probability(confidence_value, description="device confidence")
    except ValueError:
        return platform, None, "device_metadata_missing_or_invalid"
    platform = platform.strip().lower()
    if platform not in _PLATFORM_TO_DOMAIN:
        return platform, confidence, "device_platform_unknown"
    if conflict:
        return platform, confidence, "device_prior_conflict"
    if confidence < minimum_confidence:
        return platform, confidence, "device_confidence_below_threshold"
    return platform, confidence, None


def _rejection(
    *,
    source: Path,
    result_json: Path,
    reason: str,
    detail: object | None = None,
    platform: str | None = None,
    confidence: float | None = None,
    document_id: str | None = None,
) -> dict[str, object]:
    row: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "kind": REJECTION_KIND,
        "source": source.as_posix(),
        "result_json": result_json.as_posix(),
        "reason": reason,
    }
    if document_id is not None:
        row["document_id"] = document_id
    if platform is not None:
        row["device_platform"] = platform
    if confidence is not None:
        row["device_confidence"] = confidence
    if detail is not None:
        row["detail"] = detail
    return row


class _DisjointSet:
    def __init__(self, values: list[str]) -> None:
        self.parent = {value: value for value in values}

    def find(self, value: str) -> str:
        parent = self.parent[value]
        if parent != value:
            self.parent[value] = self.find(parent)
        return self.parent[value]

    def union(self, left: str, right: str) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        if left_root < right_root:
            self.parent[right_root] = left_root
        else:
            self.parent[left_root] = right_root


def _assign_source_components(candidates: list[_Candidate]) -> int:
    """Collapse upstream capture groups and exact source aliases together."""

    by_id = {candidate.document_id: candidate for candidate in candidates}
    disjoint = _DisjointSet(sorted(by_id))
    first_by_producer: dict[str, str] = {}
    first_by_source_hash: dict[str, str] = {}
    for candidate in sorted(candidates, key=lambda value: value.document_id):
        for mapping, group_id in (
            (first_by_producer, candidate.producer_group_id),
            (first_by_source_hash, candidate.source_sha256),
        ):
            previous = mapping.setdefault(group_id, candidate.document_id)
            disjoint.union(previous, candidate.document_id)
    components: dict[str, list[str]] = defaultdict(list)
    for document_id in sorted(by_id):
        components[disjoint.find(document_id)].append(document_id)
    for members in sorted(components.values(), key=lambda value: tuple(value)):
        source_group_id = _stable_id("source-component", *members)
        for document_id in members:
            by_id[document_id].source_group_id = source_group_id
    return len(components)


def _assign_component_splits(
    candidates: list[_Candidate],
    *,
    split_seed: str,
    train_ratio: float,
    calibration_ratio: float,
) -> dict[str, int]:
    by_id = {candidate.document_id: candidate for candidate in candidates}
    disjoint = _DisjointSet(sorted(by_id))
    first_by_source: dict[str, str] = {}
    first_by_content: dict[str, str] = {}
    for candidate in sorted(candidates, key=lambda value: value.document_id):
        for mapping, group_id in (
            (first_by_source, candidate.source_group_id),
            (first_by_content, candidate.content_group_id),
        ):
            previous = mapping.setdefault(group_id, candidate.document_id)
            disjoint.union(previous, candidate.document_id)
    components: dict[str, list[str]] = defaultdict(list)
    for document_id in sorted(by_id):
        components[disjoint.find(document_id)].append(document_id)
    component_counts: Counter[str] = Counter()
    for members in sorted(components.values(), key=lambda value: tuple(value)):
        identity = "\0".join(members)
        bucket = int.from_bytes(
            hashlib.sha256(f"{split_seed}\0split\0{identity}".encode("utf-8")).digest()[:8],
            "big",
        ) / float(2**64)
        if bucket < train_ratio:
            split = "train"
        elif bucket < train_ratio + calibration_ratio:
            split = "calibration"
        else:
            split = "test"
        component_counts[split] += 1
        for document_id in members:
            by_id[document_id].split = split
    return dict(sorted(component_counts.items()))


def _write_new_file(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb", closefd=True) as stream:
            descriptor = -1
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _fsync_directory(path: Path) -> None:
    try:
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    except OSError:
        pass


def _publish_directory_no_clobber(staging: Path, output: Path) -> None:
    """Atomically rename a sibling directory without replacing a raced target."""

    if os.path.lexists(output):
        raise FileExistsError(f"refusing to overwrite bootstrap output: {output}")
    if os.name == "nt":
        # Windows MoveFileEx without MOVEFILE_REPLACE_EXISTING is an atomic
        # same-volume rename and fails if another process won the target name.
        # Python's os.rename uses those no-replace semantics on Windows.
        try:
            os.rename(staging, output)
        except FileExistsError:
            raise FileExistsError(f"refusing to overwrite bootstrap output: {output}") from None
        return
    libc = ctypes.CDLL(None, use_errno=True)
    source_bytes = os.fsencode(staging)
    output_bytes = os.fsencode(output)
    if sys.platform == "darwin" and hasattr(libc, "renamex_np"):
        renamex = libc.renamex_np
        renamex.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint]
        renamex.restype = ctypes.c_int
        if renamex(source_bytes, output_bytes, 0x00000004) != 0:  # RENAME_EXCL
            error_number = ctypes.get_errno()
            if error_number == errno.EEXIST:
                raise FileExistsError(f"refusing to overwrite bootstrap output: {output}")
            raise OSError(error_number, os.strerror(error_number), os.fspath(output))
        return
    if sys.platform.startswith("linux") and hasattr(libc, "renameat2"):
        renameat2 = libc.renameat2
        renameat2.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        renameat2.restype = ctypes.c_int
        if renameat2(-100, source_bytes, -100, output_bytes, 1) != 0:  # RENAME_NOREPLACE
            error_number = ctypes.get_errno()
            if error_number == errno.EEXIST:
                raise FileExistsError(f"refusing to overwrite bootstrap output: {output}")
            raise OSError(error_number, os.strerror(error_number), os.fspath(output))
        return
    raise RuntimeError("this platform has no supported atomic no-clobber directory rename")


def _validate_parameters(
    *,
    minimum_device_confidence: float,
    minimum_regions: int,
    maximum_documents_per_domain: int,
    split_seed: str,
    train_ratio: float,
    calibration_ratio: float,
) -> None:
    _finite_probability(minimum_device_confidence, description="minimum_device_confidence")
    if isinstance(minimum_regions, bool) or not isinstance(minimum_regions, int) or not 1 <= minimum_regions <= 4:
        raise ValueError("minimum_regions must be an integer between 1 and 4")
    if (
        isinstance(maximum_documents_per_domain, bool)
        or not isinstance(maximum_documents_per_domain, int)
        or maximum_documents_per_domain < 1
    ):
        raise ValueError("maximum_documents_per_domain must be a positive integer")
    if not isinstance(split_seed, str) or not split_seed or split_seed != split_seed.strip():
        raise ValueError("split_seed must be a non-empty unpadded string")
    train = _finite_probability(train_ratio, description="train_ratio")
    calibration = _finite_probability(calibration_ratio, description="calibration_ratio")
    if train <= 0.0 or calibration <= 0.0 or train + calibration >= 1.0:
        raise ValueError("train_ratio and calibration_ratio must be positive and leave a positive test ratio")


def bootstrap_existing_pseudolabels(
    records_path: Path,
    output_dir: Path,
    *,
    minimum_device_confidence: float = 0.9,
    minimum_regions: int = 3,
    maximum_documents_per_domain: int = 500,
    split_seed: str = DEFAULT_SPLIT_SEED,
    train_ratio: float = 0.6,
    calibration_ratio: float = 0.2,
) -> dict[str, object]:
    """Create a deterministic, weakly-labelled font-domain training manifest.

    The returned dictionary is byte-for-byte the payload written to the final
    ``bootstrap.json`` completion marker.  No model is fitted here.
    """

    _validate_parameters(
        minimum_device_confidence=minimum_device_confidence,
        minimum_regions=minimum_regions,
        maximum_documents_per_domain=maximum_documents_per_domain,
        split_seed=split_seed,
        train_ratio=train_ratio,
        calibration_ratio=calibration_ratio,
    )
    try:
        records = records_path.expanduser().resolve(strict=True)
    except OSError as error:
        raise ValueError(f"unable to resolve pseudo-label manifest {records_path}: {error}") from error
    if not records.is_file():
        raise ValueError(f"pseudo-label manifest is not a file: {records}")
    records_root = records.parent.resolve(strict=True)

    output_lexical = Path(os.path.abspath(os.fspath(output_dir.expanduser())))
    try:
        output_parent = output_lexical.parent.resolve(strict=True)
    except OSError as error:
        raise ValueError(f"unable to resolve output parent {output_lexical.parent}: {error}") from error
    output = output_parent / output_lexical.name
    if not output.name or output.name in {".", ".."}:
        raise ValueError("output_dir must name a new directory")
    if not output_parent.is_dir():
        raise ValueError(f"output parent is not a directory: {output_parent}")
    if os.path.lexists(output):
        raise FileExistsError(f"refusing to overwrite bootstrap output: {output}")

    record_state = _RecordStreamState()
    grouped_rows: dict[tuple[Path, Path], _GroupAccumulator] = {}
    ignored_fields: Counter[str] = Counter()
    result_binding_cache: dict[Path, _ResultBinding] = {}
    source_path_cache: dict[str, Path] = {}
    result_path_cache: dict[str, Path] = {}
    source_stat_cache: dict[Path, os.stat_result] = {}
    duplicate_role_rows = 0

    for line_number, raw in _iter_records(records, record_state):
        location = f"{records}:{line_number}"
        source = _cached_absolute_file(
            raw.get("source"),
            cache=source_path_cache,
            description=f"{location}: source",
        )
        result_json = _cached_absolute_file(
            raw.get("result_json"),
            cache=result_path_cache,
            description=f"{location}: result_json",
        )
        field = raw.get("field")
        if not isinstance(field, str) or not field or field != field.strip():
            raise ValueError(f"{location}: field must be a non-empty unpadded string")
        producer_group_id = _producer_group_id(raw.get("group_id"), location=location)
        if result_json not in result_binding_cache:
            result_bytes = _read_snapshot(
                result_json,
                maximum_bytes=MAXIMUM_RESULT_BYTES,
                description="OCR result JSON",
            )
            result_payload = _decode_utf8_json(
                result_bytes, location=result_json.as_posix()
            )
            platform, confidence, device_reason = _device_rejection(
                result_payload,
                minimum_confidence=minimum_device_confidence,
            )
            result_binding_cache[result_json] = _ResultBinding(
                source=_result_source(result_payload, result_json),
                result_sha256=_sha256(result_bytes),
                platform=platform,
                confidence=confidence,
                device_rejection=device_reason,
            )
        result_binding = result_binding_cache[result_json]
        if result_binding.source != source:
            raise ValueError(
                f"{location}: pseudo-label source {source} differs from result source "
                f"{result_binding.source}"
            )
        source_stat = source_stat_cache.get(source)
        if source_stat is None:
            try:
                source_stat = source.stat()
            except OSError as error:
                raise ValueError(f"{location}: unable to stat source {source}: {error}") from error
            if not stat.S_ISREG(source_stat.st_mode):
                raise ValueError(f"{location}: source is not a regular file: {source}")
            source_stat_cache[source] = source_stat
        expected_size = raw.get("source_size_bytes")
        if expected_size is not None and (
            isinstance(expected_size, bool)
            or not isinstance(expected_size, int)
            or expected_size != source_stat.st_size
        ):
            raise ValueError(f"{location}: source_size_bytes does not bind the current source")
        expected_mtime = raw.get("source_mtime_ns")
        if expected_mtime is not None and (
            isinstance(expected_mtime, bool)
            or not isinstance(expected_mtime, int)
            or expected_mtime != source_stat.st_mtime_ns
        ):
            raise ValueError(f"{location}: source_mtime_ns does not bind the current source")
        # Preserve the source/result document even when this particular row is
        # intentionally ignored.  A time/status-bar-only document must still
        # receive an auditable ``insufficient_body_regions`` rejection.
        group_key = (source, result_json)
        group = grouped_rows.get(group_key)
        if group is None:
            group = _GroupAccumulator(
                source=source,
                result_json=result_json,
                producer_group_id=producer_group_id,
                regions={},
            )
            grouped_rows[group_key] = group
        elif group.producer_group_id != producer_group_id:
            raise ValueError(
                f"{location}: group_id changes within one source/result document"
            )
        role = _FIELD_TO_ROLE.get(field)
        if role is None:
            ignored_fields[field] += 1
            continue
        crop = _safe_crop_path(records_root, raw.get("image"), location=location)
        text = _normalise_visible_text(raw.get("text"), location=location)
        detector_score = _finite_probability(
            raw.get("detector_score"), description=f"{location}: detector_score"
        )
        paddle_confidence = _finite_probability(
            raw.get("paddle_confidence"), description=f"{location}: paddle_confidence"
        )
        tie_payload = {
            "line": line_number,
            "image": crop.relative,
            "text": text,
            "field": field,
        }
        expected_crop = raw.get("crop_sha256")
        if expected_crop is not None and (
            not isinstance(expected_crop, str)
            or not re.fullmatch(r"[0-9a-f]{64}", expected_crop)
        ):
            raise ValueError(f"{location}: crop_sha256 must be lowercase SHA-256")
        region = _InputRegion(
            line_number=line_number,
            role=role,
            text=text,
            crop=crop,
            detector_score=detector_score,
            paddle_confidence=paddle_confidence,
            tie_breaker=_sha256(_json_bytes(tie_payload)),
            expected_legacy_crop_sha256=expected_crop,
        )
        previous = group.regions.get(role)
        if previous is not None:
            duplicate_role_rows += 1
        if previous is None or region.selection_key > previous.selection_key:
            group.regions[role] = region

    records_sha256 = record_state.sha256
    if records_sha256 is None:  # The exhausted iterator always finalises this state.
        raise RuntimeError("pseudo-label manifest stream did not finalise")
    grouped_document_count = len(grouped_rows)

    candidates: list[_Candidate] = []
    rejections: list[dict[str, object]] = []

    for (source, result_json), group in sorted(
        grouped_rows.items(), key=lambda item: (item[0][0].as_posix(), item[0][1].as_posix())
    ):
        result_binding = result_binding_cache[result_json]
        platform = result_binding.platform
        confidence = result_binding.confidence
        device_reason = result_binding.device_rejection
        result_sha = result_binding.result_sha256
        document_id = _stable_id(
            "document",
            source.as_posix(),
            result_json.as_posix(),
            result_sha,
        )
        if device_reason is not None:
            rejections.append(
                _rejection(
                    source=source,
                    result_json=result_json,
                    reason=device_reason,
                    platform=platform,
                    confidence=confidence,
                    document_id=document_id,
                )
            )
            continue
        assert platform is not None and confidence is not None
        best_by_role = group.regions
        if len(best_by_role) < minimum_regions:
            rejections.append(
                _rejection(
                    source=source,
                    result_json=result_json,
                    reason="insufficient_body_regions",
                    detail={"found": len(best_by_role), "required": minimum_regions},
                    platform=platform,
                    confidence=confidence,
                    document_id=document_id,
                )
            )
            continue
        content_pairs = [
            [role, _normalise_content_text(region.text)]
            for role, region in sorted(best_by_role.items())
        ]
        content_group_id = _stable_id(
            "content",
            json.dumps(
                content_pairs,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ),
        )
        candidates.append(
            _Candidate(
                document_id=document_id,
                source_group_id=_stable_id("source-path", source.as_posix()),
                content_group_id=content_group_id,
                source=source,
                result_json=result_json,
                source_sha256="",
                result_sha256=result_sha,
                platform=platform,
                confidence=confidence,
                domain=_PLATFORM_TO_DOMAIN[platform],
                producer_group_id=group.producer_group_id,
                regions=best_by_role,
                expected_source_size=source_stat_cache[source].st_size,
                expected_source_mtime_ns=source_stat_cache[source].st_mtime_ns,
            )
        )

    # Large input-only structures are no longer needed beyond candidate
    # construction.  Releasing them here keeps the pilot's peak memory bounded.
    del (
        grouped_rows,
        result_binding_cache,
        source_path_cache,
        result_path_cache,
        source_stat_cache,
    )

    domains_by_producer_group: dict[str, set[str]] = defaultdict(set)
    for candidate in candidates:
        domains_by_producer_group[candidate.producer_group_id].add(candidate.domain)
    conflicted_producer_groups = {
        group_id
        for group_id, domains in domains_by_producer_group.items()
        if len(domains) > 1
    }
    if conflicted_producer_groups:
        retained_by_producer: list[_Candidate] = []
        for candidate in candidates:
            if candidate.producer_group_id not in conflicted_producer_groups:
                retained_by_producer.append(candidate)
                continue
            rejections.append(
                _rejection(
                    source=candidate.source,
                    result_json=candidate.result_json,
                    reason="producer_group_domain_conflict",
                    detail={
                        "producer_group_id": candidate.producer_group_id,
                        "domains": sorted(
                            domains_by_producer_group[candidate.producer_group_id]
                        ),
                    },
                    platform=candidate.platform,
                    confidence=candidate.confidence,
                    document_id=candidate.document_id,
                )
            )
        candidates = retained_by_producer

    domains_by_source: dict[str, set[str]] = defaultdict(set)
    for candidate in candidates:
        domains_by_source[candidate.source.as_posix()].add(candidate.domain)
    conflicted_sources = {
        source_sha for source_sha, domains in domains_by_source.items() if len(domains) > 1
    }
    if conflicted_sources:
        retained: list[_Candidate] = []
        for candidate in candidates:
            source_key = candidate.source.as_posix()
            if source_key not in conflicted_sources:
                retained.append(candidate)
                continue
            rejections.append(
                _rejection(
                    source=candidate.source,
                    result_json=candidate.result_json,
                    reason="source_domain_conflict",
                    detail={"domains": sorted(domains_by_source[source_key])},
                    platform=candidate.platform,
                    confidence=candidate.confidence,
                    document_id=candidate.document_id,
                )
            )
        candidates = retained

    by_domain: dict[str, list[_Candidate]] = defaultdict(list)
    for candidate in candidates:
        by_domain[candidate.domain].append(candidate)
    available_counts = {domain: len(by_domain.get(domain, [])) for domain in sorted(_PLATFORM_TO_DOMAIN.values())}
    nonempty_counts = [count for count in available_counts.values() if count > 0]
    balanced_target = min(maximum_documents_per_domain, min(nonempty_counts)) if nonempty_counts else 0
    selected: list[_Candidate] = []
    ordered: list[_Candidate] = []
    for domain in sorted(by_domain):
        ordered = sorted(
            by_domain[domain],
            key=lambda candidate: (
                _stable_id("sample", split_seed, candidate.document_id),
                candidate.document_id,
            ),
        )
        selected.extend(ordered[:balanced_target])
        for candidate in ordered[balanced_target:]:
            rejections.append(
                _rejection(
                    source=candidate.source,
                    result_json=candidate.result_json,
                    reason="domain_balancing_cap",
                    detail={"balanced_target": balanced_target, "domain": domain},
                    platform=candidate.platform,
                    confidence=candidate.confidence,
                    document_id=candidate.document_id,
                )
            )
    del (
        candidates,
        by_domain,
        domains_by_source,
        conflicted_sources,
        domains_by_producer_group,
        conflicted_producer_groups,
        ordered,
    )
    # Expensive source-byte hashing and image decoding happen only after the
    # deterministic per-domain cap.  On the existing 117k-result corpus this
    # keeps a 1,000-document pilot from decoding the entire source archive.
    selected_source_hashes: dict[Path, str] = {}
    verified_results: set[tuple[Path, str]] = set()
    for candidate in selected:
        result_binding_key = (candidate.result_json, candidate.result_sha256)
        if result_binding_key not in verified_results:
            _read_snapshot(
                candidate.result_json,
                maximum_bytes=MAXIMUM_RESULT_BYTES,
                description="selected OCR result JSON",
                expected_sha256=candidate.result_sha256,
            )
            verified_results.add(result_binding_key)
        source_sha = selected_source_hashes.get(candidate.source)
        if source_sha is None:
            source_bytes = _read_snapshot(
                candidate.source,
                maximum_bytes=MAXIMUM_SOURCE_BYTES,
                description="selected source image",
                expected_size=candidate.expected_source_size,
                expected_mtime_ns=candidate.expected_source_mtime_ns,
            )
            _decode_rgb(
                candidate.source,
                source_bytes,
                maximum_pixels=MAXIMUM_SOURCE_PIXELS,
                description="selected source image",
            )
            source_sha = _sha256(source_bytes)
            selected_source_hashes[candidate.source] = source_sha
        candidate.source_sha256 = source_sha
        candidate.source_group_id = _stable_id("source", source_sha)
        candidate.document_id = _stable_id(
            "document",
            source_sha,
            candidate.result_json.as_posix(),
            candidate.result_sha256,
        )

    domains_by_selected_hash: dict[str, set[str]] = defaultdict(set)
    for candidate in selected:
        domains_by_selected_hash[candidate.source_sha256].add(candidate.domain)
    selected_hash_conflicts = {
        source_sha
        for source_sha, domains in domains_by_selected_hash.items()
        if len(domains) > 1
    }
    if selected_hash_conflicts:
        retained_selected: list[_Candidate] = []
        for candidate in selected:
            if candidate.source_sha256 not in selected_hash_conflicts:
                retained_selected.append(candidate)
                continue
            rejections.append(
                _rejection(
                    source=candidate.source,
                    result_json=candidate.result_json,
                    reason="source_domain_conflict",
                    detail={
                        "domains": sorted(
                            domains_by_selected_hash[candidate.source_sha256]
                        )
                    },
                    platform=candidate.platform,
                    confidence=candidate.confidence,
                    document_id=candidate.document_id,
                )
            )
        selected = retained_selected

    selected.sort(key=lambda value: value.document_id)
    selected_source_components = _assign_source_components(selected)
    component_counts = _assign_component_splits(
        selected,
        split_seed=split_seed,
        train_ratio=train_ratio,
        calibration_ratio=calibration_ratio,
    )

    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{output.name}.",
            suffix=f".{secrets.token_hex(8)}.staging",
            dir=output_parent,
        )
    )
    published = False
    try:
        manifest_rows: list[dict[str, object]] = []
        provenance: list[dict[str, object]] = []
        for candidate in selected:
            assert candidate.split is not None
            region_rows: list[dict[str, object]] = []
            for role, region in sorted(candidate.regions.items()):
                current_crop = region.crop.recheck(records_root)
                crop_bytes = _read_snapshot(
                    current_crop,
                    maximum_bytes=MAXIMUM_CROP_BYTES,
                    description="pseudo-label crop",
                )
                crop_rgb = _decode_rgb(
                    current_crop,
                    crop_bytes,
                    maximum_pixels=MAXIMUM_CROP_PIXELS,
                    description="pseudo-label crop",
                )
                if (
                    region.expected_legacy_crop_sha256 is not None
                    and _legacy_crop_sha256(crop_rgb) != region.expected_legacy_crop_sha256
                ):
                    raise ValueError(
                        f"{records}:{region.line_number}: crop_sha256 does not bind decoded crop pixels"
                    )
                raw_sha = _sha256(crop_bytes)
                pixel_sha = _pixel_sha256(crop_rgb)
                suffix = current_crop.suffix.lower()
                relative_output = (
                    Path("regions") / candidate.document_id / f"{role}{suffix}"
                ).as_posix()
                output_crop = staging.joinpath(*relative_output.split("/"))
                _write_new_file(output_crop, crop_bytes)
                copied_bytes = _read_snapshot(
                    output_crop,
                    maximum_bytes=MAXIMUM_CROP_BYTES,
                    description="copied font-domain crop",
                )
                if _sha256(copied_bytes) != raw_sha:
                    raise ValueError(f"copied crop bytes differ from source crop: {output_crop}")
                region_rows.append(
                    {
                        "id": _stable_id("region", candidate.document_id, role, raw_sha),
                        "role": role,
                        "image": relative_output,
                        "include_in_consistency": True,
                        "text": region.text,
                        "raw_sha256": raw_sha,
                        "pixel_sha256": pixel_sha,
                    }
                )
            manifest_rows.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "kind": DOCUMENT_KIND,
                    "id": candidate.document_id,
                    "source_group_id": candidate.source_group_id,
                    "content_group_id": candidate.content_group_id,
                    "split": candidate.split,
                    "font_domain": candidate.domain,
                    "label_source": LABEL_PROVENANCE,
                    "device_prior_domain": candidate.domain,
                    "source_image_sha256": candidate.source_sha256,
                    "regions": region_rows,
                }
            )
            provenance.append(
                {
                    "document_id": candidate.document_id,
                    "source": candidate.source.as_posix(),
                    "source_sha256": candidate.source_sha256,
                    "result_json": candidate.result_json.as_posix(),
                    "result_json_sha256": candidate.result_sha256,
                    "producer_group_id": candidate.producer_group_id,
                    "source_group_id": candidate.source_group_id,
                    "device_platform": candidate.platform,
                    "device_confidence": candidate.confidence,
                }
            )

        manifest_bytes = _jsonl_bytes(manifest_rows)
        manifest_path = staging / "font_domain.auto.jsonl"
        _write_new_file(manifest_path, manifest_bytes)
        rejections.sort(
            key=lambda value: (
                str(value.get("document_id", "")),
                str(value["reason"]),
                str(value["result_json"]),
            )
        )
        _write_new_file(staging / "rejected.jsonl", _jsonl_bytes(rejections))

        dataset = None
        if selected:
            dataset = load_font_domain_dataset(
                manifest_path,
                require_labels=True,
                minimum_regions=minimum_regions,
                require_leakage_metadata=True,
            )
        selected_domain_counts = Counter(candidate.domain for candidate in selected)
        selected_split_counts = Counter(candidate.split for candidate in selected)
        calibration_by_domain = Counter(
            candidate.domain for candidate in selected if candidate.split == "calibration"
        )
        calibration_source_groups: dict[str, int] = {
            domain: len(
                {
                    candidate.source_group_id
                    for candidate in selected
                    if candidate.domain == domain and candidate.split == "calibration"
                }
            )
            for domain in sorted(_PLATFORM_TO_DOMAIN.values())
        }
        rejection_counts = Counter(str(row["reason"]) for row in rejections)
        missing_domains = [domain for domain, count in available_counts.items() if count == 0]
        insufficient_calibration_domains = [
            domain
            for domain in sorted(_PLATFORM_TO_DOMAIN.values())
            if calibration_source_groups[domain] < 20
        ]
        report: dict[str, object] = {
            "schema_version": SCHEMA_VERSION,
            "kind": BOOTSTRAP_KIND,
            "completed": True,
            "label_provenance": LABEL_PROVENANCE,
            "publication_prerequisites_recorded": False,
            "publication": False,
            "evaluation_status": "not_assessed",
            "evaluation": "not_assessed",
            "authenticity": "not_assessed",
            "counts": {
                "input_rows": record_state.records,
                "grouped_documents": grouped_document_count,
                "accepted_documents": len(selected),
                "rejected_documents": len(rejections),
            },
            "rejection_reasons": dict(sorted(rejection_counts.items())),
            "ignored_fields": dict(sorted(ignored_fields.items())),
            "input": {
                "records": records.as_posix(),
                "records_sha256": records_sha256,
                "rows": record_state.records,
                "grouped_source_result_pairs": grouped_document_count,
            },
            "parameters": {
                "minimum_device_confidence": minimum_device_confidence,
                "minimum_regions": minimum_regions,
                "maximum_documents_per_domain": maximum_documents_per_domain,
                "split_seed": split_seed,
                "train_ratio": train_ratio,
                "calibration_ratio": calibration_ratio,
                "test_ratio": 1.0 - train_ratio - calibration_ratio,
            },
            "outputs": {
                "manifest": "font_domain.auto.jsonl",
                "rejections": "rejected.jsonl",
                "completion_marker": "bootstrap.json",
            },
            "selection": {
                "available_by_domain": dict(sorted(available_counts.items())),
                "balanced_target_per_present_domain": balanced_target,
                "selected_documents": len(selected),
                "selected_by_domain": dict(sorted(selected_domain_counts.items())),
                "selected_by_split": dict(sorted((str(k), v) for k, v in selected_split_counts.items())),
                "selected_source_components": selected_source_components,
                "split_components": component_counts,
                "duplicate_role_rows_discarded": duplicate_role_rows,
                "ignored_fields": dict(sorted(ignored_fields.items())),
            },
            "unknown_or_rejected": {
                "documents": len(rejections),
                "by_reason": dict(sorted(rejection_counts.items())),
            },
            "readiness": {
                "expected_domains": sorted(_PLATFORM_TO_DOMAIN.values()),
                "missing_domains": missing_domains,
                "minimum_independent_calibration_groups_per_domain": 20,
                "calibration_documents_by_domain": dict(sorted(calibration_by_domain.items())),
                "calibration_source_groups_by_domain": calibration_source_groups,
                "insufficient_calibration_domains": insufficient_calibration_domains,
                "count_prerequisites_met": not missing_domains and not insufficient_calibration_domains,
                "note": "Weak device labels require held-out evaluation before any publication use.",
            },
            "validation": {
                "passed": dataset is not None,
                "status": "passed" if dataset is not None else "skipped_empty_manifest",
                "manifest_sha256": (
                    dataset.manifest_sha256 if dataset is not None else _sha256(manifest_bytes)
                ),
                "dataset_snapshot_sha256": (
                    dataset.snapshot_sha256 if dataset is not None else None
                ),
                "documents": len(dataset.documents) if dataset is not None else 0,
                "regions": (
                    sum(len(document.regions) for document in dataset.documents)
                    if dataset is not None
                    else 0
                ),
            },
            "provenance": provenance,
        }
        # This is the completion marker and must be the last file created in staging.
        _write_new_file(staging / "bootstrap.json", _json_bytes(report, pretty=True))
        for directory in sorted(
            (path for path in staging.rglob("*") if path.is_dir()),
            key=lambda path: len(path.parts),
            reverse=True,
        ):
            _fsync_directory(directory)
        _fsync_directory(staging)
        _publish_directory_no_clobber(staging, output)
        published = True
        _fsync_directory(output_parent)
        return report
    finally:
        if not published and staging.exists():
            shutil.rmtree(staging)


__all__ = [
    "BOOTSTRAP_KIND",
    "DEFAULT_SPLIT_SEED",
    "LABEL_PROVENANCE",
    "REJECTION_KIND",
    "bootstrap_existing_pseudolabels",
]
