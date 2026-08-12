"""Build a read-only inventory for a heterogeneous receipt-image directory.

This module is intentionally separate from every detector, OCR, training, and
attestation path in the repository.  It opens source files read-only and
publishes JSON/JSONL evidence into a caller-provided *new* directory.  It never
copies, edits, rotates, labels, or otherwise mutates a source image.

The inventory is also the unlabeled input contract for a later automated
PaddleOCR teacher.  No OCR text is present here.  Exact decoded duplicates are
quarantined, perceptual-near candidates are assigned to one suggested split,
and any later low-confidence or conflicting teacher result must be excluded or
quarantined rather than converted into a guessed label.
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
import tempfile
import sys
import warnings
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError


SCHEMA_VERSION = 1
CONTRACT_KIND = "otherimages_read_only_inventory_v1"
IMAGE_KIND = "otherimages_image_inventory_record_v1"
TEACHER_PENDING_KIND = "otherimages_paddle_teacher_pending_v1"
LAYOUT_SAMPLE_KIND = "otherimages_layout_sample_v1"
NEAR_DUPLICATE_KIND = "otherimages_phash_candidate_v1"
EXACT_DUPLICATE_KIND = "otherimages_exact_duplicate_group_v1"

TOP_STRIP_FRACTION = 0.08
PHASH_SIZE = 32
PHASH_LOW_FREQUENCY_SIZE = 8
DEFAULT_PHASH_DISTANCE = 6
DEFAULT_LAYOUT_SAMPLE_SIZE = 64
DEFAULT_SPLIT_SEED = "otherimages-split-v1"
DEFAULT_LAYOUT_SAMPLE_SEED = "otherimages-layout-sample-v1"
DEFAULT_MAX_PHASH_CANDIDATES = 100_000
_WINDOWS = os.name == "nt"

KNOWN_IMAGE_SUFFIXES = frozenset(
    {
        ".bmp",
        ".dib",
        ".gif",
        ".jfif",
        ".jpe",
        ".jpeg",
        ".jpg",
        ".png",
        ".tif",
        ".tiff",
        ".webp",
    }
)

OUTPUT_FILENAMES = (
    "images.jsonl",
    "exact_duplicates.jsonl",
    "near_duplicate_candidates.jsonl",
    "layout_sample.jsonl",
    "paddle_teacher_pending.jsonl",
    "ignored_non_images.jsonl",
    "errors.jsonl",
)

_ORIENTATION_NAMES = {
    1: "normal",
    2: "mirror_horizontal",
    3: "rotate_180",
    4: "mirror_vertical",
    5: "transpose",
    6: "rotate_90_clockwise",
    7: "transverse",
    8: "rotate_90_counterclockwise",
}


class SourceChangedError(RuntimeError):
    """Raised when source bytes or directory membership change during audit."""


def _json_line(payload: Mapping[str, object]) -> str:
    return json.dumps(
        dict(payload),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ) + "\n"


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, object]]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite inventory evidence: {path}")
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(_json_line(row))


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite inventory evidence: {path}")
    text = json.dumps(
        dict(payload),
        ensure_ascii=False,
        sort_keys=True,
        indent=2,
        allow_nan=False,
    ) + "\n"
    path.write_text(text, encoding="utf-8", newline="\n")


def _raw_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact_binding(path: Path) -> dict[str, object]:
    data = path.read_bytes()
    return {
        "path": path.name,
        "sha256": hashlib.sha256(data).hexdigest(),
        "size_bytes": len(data),
        "line_count": data.count(b"\n"),
    }


def _decoded_pixel_sha256(rgb: np.ndarray) -> str:
    """Match the decoded-RGB identity already used by unified OCR manifests."""
    digest = hashlib.sha256()
    digest.update(str(rgb.shape).encode("ascii"))
    digest.update(np.ascontiguousarray(rgb, dtype=np.uint8).tobytes(order="C"))
    return digest.hexdigest()


def _dct_matrix(size: int) -> np.ndarray:
    positions = np.arange(size, dtype=np.float64)
    frequencies = positions[:, None]
    matrix = np.cos((math.pi / size) * (positions + 0.5) * frequencies)
    matrix[0] *= math.sqrt(1.0 / size)
    matrix[1:] *= math.sqrt(2.0 / size)
    return matrix


_PHASH_DCT = _dct_matrix(PHASH_SIZE)


def _perceptual_hash64(rgb: np.ndarray) -> str:
    grayscale = Image.fromarray(rgb).convert("L").resize(
        (PHASH_SIZE, PHASH_SIZE),
        Image.Resampling.BICUBIC,
    )
    pixels = np.asarray(grayscale, dtype=np.float64)
    # ``np.dot`` avoids spurious Accelerate/NumPy 2.x matmul floating-point
    # warnings observed for these small, finite matrices on macOS.
    coefficients = np.dot(np.dot(_PHASH_DCT, pixels), _PHASH_DCT.T)
    low = coefficients[:PHASH_LOW_FREQUENCY_SIZE, :PHASH_LOW_FREQUENCY_SIZE].reshape(-1)
    threshold = float(np.median(low[1:]))
    bits = low > threshold
    value = 0
    for bit in bits:
        value = (value << 1) | int(bool(bit))
    return f"{value:016x}"


def _hamming_distance(left: int, right: int) -> int:
    return (left ^ right).bit_count()


class _BKTree:
    """A compact exact Hamming-radius index for distinct 64-bit pHashes."""

    def __init__(self) -> None:
        self._root: tuple[int, dict[int, Any]] | None = None

    def add(self, value: int) -> None:
        if self._root is None:
            self._root = (value, {})
            return
        node = self._root
        while True:
            distance = _hamming_distance(value, node[0])
            child = node[1].get(distance)
            if child is None:
                node[1][distance] = (value, {})
                return
            node = child

    def query(self, value: int, maximum_distance: int) -> list[tuple[int, int]]:
        if self._root is None:
            return []
        matches: list[tuple[int, int]] = []
        pending = [self._root]
        while pending:
            node = pending.pop()
            distance = _hamming_distance(value, node[0])
            if distance <= maximum_distance:
                matches.append((distance, node[0]))
            lower = distance - maximum_distance
            upper = distance + maximum_distance
            pending.extend(child for edge, child in node[1].items() if lower <= edge <= upper)
        matches.sort(key=lambda item: (item[0], item[1]))
        return matches


class _UnionFind:
    def __init__(self, size: int) -> None:
        self._parent = list(range(size))
        self._rank = [0] * size

    def find(self, value: int) -> int:
        while self._parent[value] != value:
            self._parent[value] = self._parent[self._parent[value]]
            value = self._parent[value]
        return value

    def union(self, left: int, right: int) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        if self._rank[left_root] < self._rank[right_root]:
            left_root, right_root = right_root, left_root
        self._parent[right_root] = left_root
        if self._rank[left_root] == self._rank[right_root]:
            self._rank[left_root] += 1


def _is_reparse(path: Path) -> bool:
    status = path.lstat()
    attributes = int(getattr(status, "st_file_attributes", 0))
    reparse_attribute = int(getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400))
    return stat.S_ISLNK(status.st_mode) or bool(attributes & reparse_attribute)


def _require_no_reparse_ancestors(path: Path, *, include_leaf: bool = True) -> None:
    candidate = Path(os.path.abspath(os.path.expanduser(os.fspath(path))))
    current = candidate if include_leaf else candidate.parent
    while True:
        if current.exists() and _is_reparse(current):
            raise ValueError(f"path traverses a symlink/junction/reparse point: {current}")
        parent = current.parent
        if parent == current:
            return
        current = parent


def _path_key(path: Path) -> tuple[str, str]:
    text = path.as_posix()
    return text.casefold(), text


def _iter_regular_files(root: Path) -> list[Path]:
    """Recursively enumerate files without following directory links."""
    files: list[Path] = []
    pending = [root]
    while pending:
        directory = pending.pop()
        try:
            entries = list(os.scandir(directory))
        except OSError as error:
            raise OSError(f"unable to enumerate source directory {directory}: {error}") from error
        for entry in entries:
            path = Path(entry.path)
            try:
                status = path.lstat()
            except OSError as error:
                raise OSError(f"unable to stat source entry {path}: {error}") from error
            attributes = int(getattr(status, "st_file_attributes", 0))
            reparse_attribute = int(getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400))
            if stat.S_ISLNK(status.st_mode) or bool(attributes & reparse_attribute):
                raise ValueError(f"source tree contains a symlink/junction/reparse point: {path}")
            if stat.S_ISDIR(status.st_mode):
                pending.append(path)
            elif stat.S_ISREG(status.st_mode):
                files.append(path)
    files.sort(key=lambda path: _path_key(path.relative_to(root)))
    return files


def _paths_overlap(left: Path, right: Path) -> bool:
    left_text = os.path.normcase(os.path.abspath(os.fspath(left)))
    right_text = os.path.normcase(os.path.abspath(os.fspath(right)))
    try:
        common = os.path.commonpath((left_text, right_text))
    except ValueError:
        return False
    return common == left_text or common == right_text


def _filesystem_identity(status: os.stat_result) -> tuple[int, int]:
    return int(getattr(status, "st_dev", 0)), int(getattr(status, "st_ino", 0))


def _windows_open_path_handle(path: Path, *, access: int, share: int) -> object:
    from ctypes import wintypes

    create_file = ctypes.WinDLL("kernel32", use_last_error=True).CreateFileW
    create_file.argtypes = (
        wintypes.LPCWSTR,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.LPVOID,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.HANDLE,
    )
    create_file.restype = wintypes.HANDLE
    handle = create_file(
        os.fspath(path),
        access,
        share,
        None,
        3,  # OPEN_EXISTING
        0x02000000 | 0x00200000,  # BACKUP_SEMANTICS | OPEN_REPARSE_POINT
        None,
    )
    if handle == ctypes.c_void_p(-1).value:
        error_number = ctypes.get_last_error()
        raise OSError(error_number, ctypes.FormatError(error_number), os.fspath(path))
    return handle


def _windows_close_handle(handle: object) -> None:
    from ctypes import wintypes

    close_handle = ctypes.WinDLL("kernel32", use_last_error=True).CloseHandle
    close_handle.argtypes = (wintypes.HANDLE,)
    close_handle.restype = wintypes.BOOL
    close_handle(handle)


def _windows_handle_information(handle: object) -> tuple[tuple[int, int], int]:
    from ctypes import wintypes

    class ByHandleFileInformation(ctypes.Structure):
        _fields_ = (
            ("file_attributes", wintypes.DWORD),
            ("creation_time", wintypes.FILETIME),
            ("last_access_time", wintypes.FILETIME),
            ("last_write_time", wintypes.FILETIME),
            ("volume_serial_number", wintypes.DWORD),
            ("file_size_high", wintypes.DWORD),
            ("file_size_low", wintypes.DWORD),
            ("number_of_links", wintypes.DWORD),
            ("file_index_high", wintypes.DWORD),
            ("file_index_low", wintypes.DWORD),
        )

    information = ByHandleFileInformation()
    get_information = ctypes.WinDLL("kernel32", use_last_error=True).GetFileInformationByHandle
    get_information.argtypes = (wintypes.HANDLE, ctypes.POINTER(ByHandleFileInformation))
    get_information.restype = wintypes.BOOL
    if not get_information(handle, ctypes.byref(information)):
        error_number = ctypes.get_last_error()
        raise OSError(error_number, ctypes.FormatError(error_number))
    identity = (
        int(information.volume_serial_number),
        (int(information.file_index_high) << 32) | int(information.file_index_low),
    )
    return identity, int(information.file_attributes)


def _bind_stage_identity(path: Path, *, directory: bool) -> tuple[int, int]:
    if _WINDOWS:
        handle = _windows_open_path_handle(
            path,
            access=0x0080,  # FILE_READ_ATTRIBUTES
            share=0x00000001 | 0x00000002 | 0x00000004,
        )
        try:
            identity, attributes = _windows_handle_information(handle)
        finally:
            _windows_close_handle(handle)
        if attributes & 0x00000400:
            raise SourceChangedError(f"stage became a symlink/junction/reparse point: {path}")
        if directory != bool(attributes & 0x00000010):
            expected = "directory" if directory else "regular file"
            raise SourceChangedError(f"stage is not a {expected}: {path}")
        return identity
    status = path.lstat()
    if _is_reparse(path):
        raise SourceChangedError(f"stage became a symlink/junction/reparse point: {path}")
    if directory and not stat.S_ISDIR(status.st_mode):
        raise SourceChangedError(f"stage is not a directory: {path}")
    if not directory and not stat.S_ISREG(status.st_mode):
        raise SourceChangedError(f"stage is not a regular file: {path}")
    return _filesystem_identity(status)


def _rename_directory_no_replace(
    source: Path,
    destination: Path,
    *,
    expected_parent_identity: tuple[int, int] | None = None,
    expected_stage_identity: tuple[int, int] | None = None,
) -> None:
    """Atomically publish a directory while refusing an existing destination."""
    if source.parent != destination.parent:
        raise ValueError("no-replace publication must stay within one parent directory")
    parent_before = _bind_stage_identity(source.parent, directory=True)
    parent_identity = expected_parent_identity or parent_before
    if parent_before != parent_identity:
        raise SourceChangedError("output parent identity changed before no-replace publication")
    source_is_directory = source.is_dir()
    source_before = _bind_stage_identity(source, directory=source_is_directory)
    stage_identity = expected_stage_identity or source_before
    if source_before != stage_identity:
        raise SourceChangedError("publication stage identity changed before no-replace publication")
    if destination.exists():
        raise FileExistsError(f"refusing to replace existing publication directory: {destination}")
    if _bind_stage_identity(source.parent, directory=True) != parent_identity:
        raise SourceChangedError("output parent identity changed before no-replace publication")
    if _WINDOWS:
        from ctypes import wintypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        parent_handle = _windows_open_path_handle(
            source.parent,
            access=0x00000001 | 0x00000020 | 0x00000080,
            share=0x00000001 | 0x00000002,
        )
        try:
            source_handle = _windows_open_path_handle(
                source,
                access=0x00010000 | 0x00000080,  # DELETE | FILE_READ_ATTRIBUTES
                share=0x00000001 | 0x00000002 | 0x00000004,
            )
        except BaseException:
            _windows_close_handle(parent_handle)
            raise
        try:
            observed_parent, parent_attributes = _windows_handle_information(parent_handle)
            observed_source, source_attributes = _windows_handle_information(source_handle)
            if observed_parent != parent_identity or not parent_attributes & 0x00000010:
                raise SourceChangedError("output parent handle identity changed before Windows rename")
            if parent_attributes & 0x00000400:
                raise SourceChangedError("output parent became a reparse point before Windows rename")
            if observed_source != stage_identity or source_attributes & 0x00000400:
                raise SourceChangedError("publication stage handle identity changed before Windows rename")

            destination_name = destination.name

            class FileRenameInfo(ctypes.Structure):
                _fields_ = (
                    ("flags", wintypes.DWORD),
                    ("root_directory", wintypes.HANDLE),
                    ("file_name_length", wintypes.DWORD),
                    ("file_name", wintypes.WCHAR * len(destination_name)),
                )

            information = FileRenameInfo()
            information.flags = 0
            information.root_directory = parent_handle
            information.file_name_length = len(destination_name.encode("utf-16-le"))
            information.file_name = destination_name
            set_information = kernel32.SetFileInformationByHandle
            set_information.argtypes = (
                wintypes.HANDLE,
                ctypes.c_int,
                wintypes.LPVOID,
                wintypes.DWORD,
            )
            set_information.restype = wintypes.BOOL
            if not set_information(
                source_handle,
                3,  # FileRenameInfo
                ctypes.byref(information),
                ctypes.sizeof(information),
            ):
                error_number = ctypes.get_last_error()
                if error_number in {80, 183}:
                    raise FileExistsError(error_number, ctypes.FormatError(error_number), os.fspath(destination))
                raise OSError(error_number, ctypes.FormatError(error_number), os.fspath(destination))
            renamed_identity, _renamed_attributes = _windows_handle_information(source_handle)
            if renamed_identity != stage_identity:
                raise SourceChangedError("published Windows handle identity differs from bound stage")
        finally:
            _windows_close_handle(source_handle)
            _windows_close_handle(parent_handle)
    else:
        library = ctypes.CDLL(None, use_errno=True)
        open_flags = os.O_RDONLY | int(getattr(os, "O_DIRECTORY", 0))
        parent_fd = os.open(source.parent, open_flags)
        try:
            if _filesystem_identity(os.fstat(parent_fd)) != parent_identity:
                raise SourceChangedError("output parent identity changed before anchored no-replace rename")
            source_status = os.stat(source.name, dir_fd=parent_fd, follow_symlinks=False)
            if _filesystem_identity(source_status) != stage_identity:
                raise SourceChangedError("publication stage identity changed before anchored rename")
            source_bytes = os.fsencode(source.name)
            destination_bytes = os.fsencode(destination.name)
            if sys.platform == "darwin":
                function = library.renameatx_np
                function.argtypes = (
                    ctypes.c_int,
                    ctypes.c_char_p,
                    ctypes.c_int,
                    ctypes.c_char_p,
                    ctypes.c_uint,
                )
                function.restype = ctypes.c_int
                result = function(
                    parent_fd,
                    source_bytes,
                    parent_fd,
                    destination_bytes,
                    0x00000004,
                )  # RENAME_EXCL
            elif sys.platform.startswith("linux") and hasattr(library, "renameat2"):
                function = library.renameat2
                function.argtypes = (
                    ctypes.c_int,
                    ctypes.c_char_p,
                    ctypes.c_int,
                    ctypes.c_char_p,
                    ctypes.c_uint,
                )
                function.restype = ctypes.c_int
                result = function(
                    parent_fd,
                    source_bytes,
                    parent_fd,
                    destination_bytes,
                    0x00000001,
                )  # RENAME_NOREPLACE
            else:
                raise OSError(
                    errno.ENOTSUP,
                    "atomic no-replace directory publication is unavailable",
                    os.fspath(destination),
                )
            if result != 0:
                error_number = ctypes.get_errno()
                if error_number in {errno.EEXIST, errno.ENOTEMPTY}:
                    raise FileExistsError(error_number, os.strerror(error_number), os.fspath(destination))
                raise OSError(error_number, os.strerror(error_number), os.fspath(destination))
            destination_status = os.stat(destination.name, dir_fd=parent_fd, follow_symlinks=False)
            if _filesystem_identity(destination_status) != stage_identity:
                raise SourceChangedError("published destination identity differs from bound stage")
        finally:
            os.close(parent_fd)


def _stat_signature(status: os.stat_result) -> tuple[int, int, int, int]:
    return (
        int(status.st_size),
        int(status.st_mtime_ns),
        int(getattr(status, "st_dev", 0)),
        int(getattr(status, "st_ino", 0)),
    )


def _source_signature(path: Path) -> tuple[int, int, int, int]:
    status = path.stat()
    if not stat.S_ISREG(status.st_mode):
        raise SourceChangedError(f"source is no longer a regular file: {path}")
    return _stat_signature(status)


def _iso_utc_from_ns(value: int) -> str:
    return datetime.fromtimestamp(value / 1_000_000_000, tz=timezone.utc).isoformat().replace("+00:00", "Z")


def _safe_exif_text(value: object) -> str | None:
    if value is None:
        return None
    if isinstance(value, bytes):
        text = value.decode("utf-8", errors="replace")
    elif isinstance(value, (str, int, float)):
        text = str(value)
    else:
        return None
    text = text.replace("\x00", "").strip()
    return text[:256] or None


def _exif_summary(image: Image.Image) -> dict[str, object]:
    try:
        exif = image.getexif()
    except (OSError, ValueError, SyntaxError) as error:
        return {
            "present": False,
            "parse_error": f"{type(error).__name__}: {str(error)[:256]}",
            "sensitive_values_omitted": ["gps_coordinates", "user_comment"],
        }
    orientation_value = exif.get(274)
    orientation = int(orientation_value) if isinstance(orientation_value, int) and 1 <= orientation_value <= 8 else None
    return {
        "present": bool(exif),
        "tag_count": len(exif),
        "tag_ids": sorted(int(tag) for tag in exif.keys() if isinstance(tag, int)),
        "orientation": orientation,
        "orientation_name": _ORIENTATION_NAMES.get(orientation),
        "orientation_applied": orientation not in (None, 1),
        "make": _safe_exif_text(exif.get(271)),
        "model": _safe_exif_text(exif.get(272)),
        "software": _safe_exif_text(exif.get(305)),
        "datetime": _safe_exif_text(exif.get(306)),
        "datetime_original": _safe_exif_text(exif.get(36867)),
        "gps_present": 34853 in exif,
        "sensitive_values_omitted": ["gps_coordinates", "user_comment"],
    }


def _grayscale_metrics(gray: np.ndarray) -> dict[str, float]:
    values = np.asarray(gray, dtype=np.float32)
    p05, p50, p95 = (float(value) for value in np.percentile(values, (5, 50, 95)))
    histogram = np.bincount(np.asarray(np.rint(values), dtype=np.uint8).reshape(-1), minlength=256).astype(np.float64)
    probabilities = histogram[histogram > 0] / values.size
    entropy = float(-(probabilities * np.log2(probabilities)).sum())
    horizontal = np.abs(np.diff(values, axis=1)) if values.shape[1] > 1 else np.empty((0,), dtype=np.float32)
    vertical = np.abs(np.diff(values, axis=0)) if values.shape[0] > 1 else np.empty((0,), dtype=np.float32)
    edge_values = np.concatenate((horizontal.reshape(-1), vertical.reshape(-1)))
    edge_density = float(np.mean(edge_values >= 20.0)) if edge_values.size else 0.0
    if values.shape[0] >= 3 and values.shape[1] >= 3:
        center = values[1:-1, 1:-1]
        laplacian = (
            -4.0 * center
            + values[:-2, 1:-1]
            + values[2:, 1:-1]
            + values[1:-1, :-2]
            + values[1:-1, 2:]
        )
        laplacian_variance = float(np.var(laplacian))
    else:
        laplacian_variance = 0.0
    return {
        "mean": round(float(values.mean()), 6),
        "stddev": round(float(values.std()), 6),
        "p05": round(p05, 6),
        "p50": round(p50, 6),
        "p95": round(p95, 6),
        "dynamic_range_p95_p05": round(p95 - p05, 6),
        "entropy_bits": round(entropy, 6),
        "edge_density_at_20": round(edge_density, 8),
        "foreground_fraction_from_median_at_24": round(float(np.mean(np.abs(values - p50) >= 24.0)), 8),
        "dark_clipped_fraction": round(float(np.mean(values <= 2.0)), 8),
        "light_clipped_fraction": round(float(np.mean(values >= 253.0)), 8),
        "laplacian_variance": round(laplacian_variance, 6),
    }


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _top_strip_statusbar_metrics(rgb: np.ndarray) -> dict[str, object]:
    height, width = rgb.shape[:2]
    strip_height = max(1, int(round(height * TOP_STRIP_FRACTION)))
    top = rgb[:strip_height]
    next_band = rgb[strip_height : min(height, strip_height * 2)]
    top_canvas = Image.fromarray(top).resize((512, 64), Image.Resampling.BICUBIC)
    gray_top = np.asarray(top_canvas.convert("L"), dtype=np.uint8)
    top_metrics = _grayscale_metrics(gray_top)
    if next_band.size:
        next_canvas = Image.fromarray(next_band).resize((512, 64), Image.Resampling.BICUBIC)
        gray_next = np.asarray(next_canvas.convert("L"), dtype=np.uint8)
        next_metrics: dict[str, float] | None = _grayscale_metrics(gray_next)
    else:
        next_metrics = None

    contrast_signal = _clamp01(float(top_metrics["dynamic_range_p95_p05"]) / 96.0)
    edge_signal = _clamp01(float(top_metrics["edge_density_at_20"]) / 0.035)
    foreground_signal = _clamp01(float(top_metrics["foreground_fraction_from_median_at_24"]) / 0.12)
    presence_score = 0.45 * contrast_signal + 0.35 * edge_signal + 0.20 * foreground_signal
    if presence_score >= 0.45:
        presence_state = "likely_present"
    elif presence_score >= 0.15:
        presence_state = "uncertain"
    else:
        presence_state = "unlikely_present"

    resolution_signal = _clamp01(strip_height / 16.0) * _clamp01(width / 320.0)
    sharpness_signal = _clamp01(float(top_metrics["laplacian_variance"]) / 500.0)
    quality_score = resolution_signal * (0.45 * sharpness_signal + 0.35 * contrast_signal + 0.20 * edge_signal)
    if quality_score >= 0.55:
        quality_state = "adequate"
    elif quality_score >= 0.20:
        quality_state = "limited"
    else:
        quality_state = "poor"

    comparison: dict[str, float] | None
    if next_metrics is None:
        comparison = None
    else:
        comparison = {
            "top_minus_next_mean": round(float(top_metrics["mean"]) - float(next_metrics["mean"]), 6),
            "top_minus_next_stddev": round(float(top_metrics["stddev"]) - float(next_metrics["stddev"]), 6),
            "top_minus_next_edge_density": round(
                float(top_metrics["edge_density_at_20"]) - float(next_metrics["edge_density_at_20"]),
                8,
            ),
        }
    return {
        "strip_contract": "exif_upright_rgb_top_round_height_times_0.08_resize_512x64_bicubic_v1",
        "fraction_requested": TOP_STRIP_FRACTION,
        "height_pixels": strip_height,
        "width_pixels": width,
        "measurement_canvas_width": 512,
        "measurement_canvas_height": 64,
        "measurement_resampling": "bicubic",
        "fraction_actual": round(strip_height / height, 8),
        "presence_is_heuristic_not_ground_truth": True,
        "presence_heuristic": "contrast_edge_foreground_v1",
        "presence_score": round(presence_score, 6),
        "presence_state": presence_state,
        "quality_score": round(quality_score, 6),
        "quality_state": quality_state,
        "top_band": top_metrics,
        "next_band": next_metrics,
        "top_vs_next": comparison,
    }


def _full_image_quality(rgb: np.ndarray) -> dict[str, object]:
    image = Image.fromarray(rgb)
    image.thumbnail((512, 512), Image.Resampling.BICUBIC)
    grayscale = np.asarray(image.convert("L"), dtype=np.uint8)
    metrics = _grayscale_metrics(grayscale)
    low_information = float(metrics["stddev"]) < 2.0 and float(metrics["entropy_bits"]) < 0.25
    return {
        "measurement_contract": "exif_upright_rgb_fit_within_512x512_bicubic_luma_v1",
        "measurement_width": int(grayscale.shape[1]),
        "measurement_height": int(grayscale.shape[0]),
        "metrics": metrics,
        "low_information": low_information,
        "phash_usable": not low_information,
    }


def _aspect_bucket(width: int, height: int) -> str:
    ratio = width / height
    if ratio < 0.80:
        return "portrait"
    if ratio > 1.25:
        return "landscape"
    return "squareish"


def _resolution_bucket(width: int, height: int) -> str:
    pixels = width * height
    if pixels < 1_000_000:
        return "lt_1mp"
    if pixels < 4_000_000:
        return "1_to_4mp"
    return "ge_4mp"


def _probe_unknown_suffix(path: Path) -> bool:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(path) as image:
                return bool(image.format)
    except (UnidentifiedImageError, OSError, ValueError, SyntaxError):
        return False


def _inspect_image(path: Path, root: Path) -> tuple[dict[str, object], tuple[int, int, int, int]]:
    with path.open("rb") as source_stream:
        before = _stat_signature(os.fstat(source_stream.fileno()))
        digest = hashlib.sha256()
        for chunk in iter(lambda: source_stream.read(1024 * 1024), b""):
            digest.update(chunk)
        raw_sha256 = digest.hexdigest()
        source_stream.seek(0)
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(source_stream) as opened:
                raw_width, raw_height = opened.size
                detected_format = opened.format or "UNKNOWN"
                raw_mode = opened.mode
                frame_count = int(getattr(opened, "n_frames", 1))
                exif = _exif_summary(opened)
                upright = ImageOps.exif_transpose(opened).convert("RGB")
                rgb = np.asarray(upright, dtype=np.uint8)
        descriptor_after = _stat_signature(os.fstat(source_stream.fileno()))
    if descriptor_after != before:
        raise SourceChangedError(f"source changed while its open descriptor was decoded: {path}")
    after = _source_signature(path)
    if after != before:
        raise SourceChangedError(f"source changed while it was decoded: {path}")
    height, width = rgb.shape[:2]
    relative_path = path.relative_to(root).as_posix()
    record_id = hashlib.sha256(
        b"otherimages-image-record-v1\0"
        + relative_path.encode("utf-8")
        + b"\0"
        + raw_sha256.encode("ascii")
    ).hexdigest()
    statusbar = _top_strip_statusbar_metrics(rgb)
    full_quality = _full_image_quality(rgb)
    top_band = dict(statusbar["top_band"])
    quality_flags: list[str] = []
    if width < 64 or height < 64:
        quality_flags.append("tiny")
    if bool(full_quality["low_information"]):
        quality_flags.append("low_information_full")
    if float(top_band["stddev"]) < 2.0 and float(top_band["entropy_bits"]) < 0.25:
        quality_flags.append("low_information_top8")
    if str(statusbar["quality_state"]) == "poor" and not bool(full_quality["low_information"]):
        quality_flags.append("likely_blur_or_low_detail_top8")
    if frame_count > 1:
        quality_flags.append("multiframe_first_frame_only")
    return (
        {
            "schema_version": SCHEMA_VERSION,
            "kind": IMAGE_KIND,
            "record_id": record_id,
            "source": {
                "root": str(root),
                "relative_path": relative_path,
                "absolute_path": str(path),
                "file_name": path.name,
                "suffix": path.suffix.lower(),
                "size_bytes": before[0],
                "mtime_ns": before[1],
                "mtime_utc": _iso_utc_from_ns(before[1]),
            },
            "container": {
                "format": detected_format,
                "mime": Image.MIME.get(detected_format),
                "mode": raw_mode,
                "frame_count": frame_count,
                "decoded_frame_policy": "first_frame",
            },
            "geometry": {
                "stored_width": raw_width,
                "stored_height": raw_height,
                "upright_width": width,
                "upright_height": height,
                "aspect_bucket": _aspect_bucket(width, height),
                "resolution_bucket": _resolution_bucket(width, height),
            },
            "exif": exif,
            "hashes": {
                "raw_sha256": raw_sha256,
                "decoded_pixel_sha256": _decoded_pixel_sha256(rgb),
                "decoded_pixel_contract": "ascii_numpy_shape_then_exif_upright_first_frame_rgb8_bytes_v1",
                "phash64": _perceptual_hash64(rgb),
                "phash_contract": "exif_upright_first_frame_luma_bicubic32_dct_low8_excluding_dc_median_v1",
            },
            "top_8_percent_statusbar": statusbar,
            "full_image_quality": full_quality,
            "quality_flags": quality_flags,
            "operations": {
                "source_open_mode": "read_only",
                "source_mutated": False,
                "image_copied": False,
                "ocr_performed": False,
                "training_performed": False,
            },
        },
        before,
    )


def _duplicate_groups(
    records: Sequence[Mapping[str, object]],
    key_name: str,
) -> tuple[list[dict[str, object]], dict[str, str]]:
    by_digest: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for record in records:
        hashes = record["hashes"]
        assert isinstance(hashes, Mapping)
        by_digest[str(hashes[key_name])].append(record)
    groups: list[dict[str, object]] = []
    duplicate_of: dict[str, str] = {}
    for digest, members in sorted(by_digest.items()):
        if len(members) < 2:
            continue
        members = sorted(members, key=lambda item: _path_key(Path(str(dict(item["source"])["relative_path"]))))
        canonical_id = str(members[0]["record_id"])
        member_rows = [
            {
                "record_id": str(member["record_id"]),
                "relative_path": str(dict(member["source"])["relative_path"]),
            }
            for member in members
        ]
        groups.append(
            {
                "schema_version": SCHEMA_VERSION,
                "kind": EXACT_DUPLICATE_KIND,
                "identity": key_name,
                "digest": digest,
                "member_count": len(members),
                "canonical_record_id": canonical_id,
                "members": member_rows,
            }
        )
        for member in members[1:]:
            duplicate_of[str(member["record_id"])] = canonical_id
    return groups, duplicate_of


def _phash_candidates_and_groups(
    records: Sequence[dict[str, object]],
    *,
    maximum_distance: int,
    maximum_candidates: int,
) -> tuple[list[dict[str, object]], dict[str, str], dict[str, object]]:
    decoded_members: dict[str, list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        decoded_members[str(dict(record["hashes"])["decoded_pixel_sha256"])].append(index)

    # Index one canonical record per exact decoded-pixel group.  Exact copies
    # are already conservatively unioned below, and expanding them into every
    # pHash candidate pair would produce quadratic duplicate evidence.
    representative_indices = [members[0] for _digest, members in sorted(decoded_members.items())]
    hash_members: dict[int, list[int]] = defaultdict(list)
    phash_unusable_images = 0
    phash_unusable_representatives = 0
    for index, record in enumerate(records):
        full_quality = record["full_image_quality"]
        assert isinstance(full_quality, Mapping)
        if not bool(full_quality["phash_usable"]):
            phash_unusable_images += 1
    for index in representative_indices:
        record = records[index]
        full_quality = record["full_image_quality"]
        assert isinstance(full_quality, Mapping)
        if not bool(full_quality["phash_usable"]):
            phash_unusable_representatives += 1
            continue
        hashes = record["hashes"]
        assert isinstance(hashes, Mapping)
        hash_members[int(str(hashes["phash64"]), 16)].append(index)

    union = _UnionFind(len(records))
    for members in decoded_members.values():
        for member in members[1:]:
            union.union(members[0], member)

    candidates: list[dict[str, object]] = []
    distinct_edges: set[tuple[int, int]] = set()

    def append_candidate(left_index: int, right_index: int, distance: int) -> None:
        left_geometry = dict(records[left_index]["geometry"])
        right_geometry = dict(records[right_index]["geometry"])
        left_ratio = float(left_geometry["upright_width"]) / float(left_geometry["upright_height"])
        right_ratio = float(right_geometry["upright_width"]) / float(right_geometry["upright_height"])
        aspect_relative_delta = abs(left_ratio - right_ratio) / max(left_ratio, right_ratio)
        left_hashes = dict(records[left_index]["hashes"])
        right_hashes = dict(records[right_index]["hashes"])
        left_source = dict(records[left_index]["source"])
        right_source = dict(records[right_index]["source"])
        union.union(left_index, right_index)
        candidates.append(
            {
                "schema_version": SCHEMA_VERSION,
                "kind": NEAR_DUPLICATE_KIND,
                "candidate_type": "identical_phash" if distance == 0 else "near_phash",
                "distance": distance,
                "aspect_evidence_reference_delta": 0.01,
                "aspect_relative_delta": round(aspect_relative_delta, 8),
                "aspect_risk": (
                    "within_1pct"
                    if aspect_relative_delta <= 0.01
                    else "outside_1pct_conservative_same_split_union"
                ),
                "left_record_id": records[left_index]["record_id"],
                "right_record_id": records[right_index]["record_id"],
                "left_relative_path": left_source["relative_path"],
                "right_relative_path": right_source["relative_path"],
                "left_phash64": left_hashes["phash64"],
                "right_phash64": right_hashes["phash64"],
                "left_exact_decoded_group_members": len(
                    decoded_members[str(left_hashes["decoded_pixel_sha256"])]
                ),
                "right_exact_decoded_group_members": len(
                    decoded_members[str(right_hashes["decoded_pixel_sha256"])]
                ),
                "candidate_edge_contract": "decoded_pixel_group_canonical_representatives_v1",
                "same_raw_sha256": left_hashes["raw_sha256"] == right_hashes["raw_sha256"],
                "same_decoded_pixel_sha256": (
                    left_hashes["decoded_pixel_sha256"] == right_hashes["decoded_pixel_sha256"]
                ),
                "automatic_drop_authorized": False,
            }
        )
        if len(candidates) > maximum_candidates:
            raise ValueError(
                "pHash candidate evidence exceeds --max-phash-candidates; "
                "raise the explicit cap rather than publishing truncated leakage evidence"
            )

    for value, members in sorted(hash_members.items()):
        for position, left_index in enumerate(members):
            for right_index in members[position + 1 :]:
                append_candidate(left_index, right_index, 0)

    tree = _BKTree()
    for value in sorted(hash_members):
        for distance, other in tree.query(value, maximum_distance):
            left_value, right_value = sorted((value, other))
            left_members = hash_members[left_value]
            right_members = hash_members[right_value]
            before_count = len(candidates)
            for left_index in left_members:
                for right_index in right_members:
                    append_candidate(left_index, right_index, distance)
            if len(candidates) > before_count:
                distinct_edges.add((left_value, right_value))
        tree.add(value)

    component_members: dict[int, list[int]] = defaultdict(list)
    for index in range(len(records)):
        component_members[union.find(index)].append(index)
    group_by_record_id: dict[str, str] = {}
    for members in component_members.values():
        component_hashes = sorted(
            {
                str(dict(records[index]["hashes"])["decoded_pixel_sha256"])
                for index in members
            }
        )
        group_digest = hashlib.sha256(
            ("otherimages-leakage-component-v1\0" + "\0".join(component_hashes)).encode("ascii")
        ).hexdigest()
        group_id = f"leakage-group:{group_digest}"
        for index in members:
            group_by_record_id[str(records[index]["record_id"])] = group_id
    candidates.sort(key=lambda row: _json_line(row))
    return (
        candidates,
        group_by_record_id,
        {
            "distinct_phashes": len(hash_members),
            "decoded_pixel_groups": len(decoded_members),
            "phash_index_representatives": len(representative_indices) - phash_unusable_representatives,
            "exact_decoded_copies_collapsed_before_phash_index": len(records) - len(decoded_members),
            "phash_unusable_images": phash_unusable_images,
            "candidate_evidence_rows": len(candidates),
            "distinct_phash_edges": len(distinct_edges),
            "represented_record_pairs": len(candidates),
            "connected_components": len(component_members),
            "maximum_distance": maximum_distance,
            "aspect_evidence_reference_delta": 0.01,
            "all_radius_candidates_conservatively_union_same_split": True,
            "aspect_delta_is_evidence_not_a_grouping_gate": True,
            "truncated": False,
        },
    )


def _suggested_split(group_id: str, *, validation_ratio: float, test_ratio: float, seed: str) -> str:
    digest = hashlib.sha256((seed + "\0" + group_id).encode("utf-8")).digest()
    unit = int.from_bytes(digest[:8], "big") / 2**64
    train_boundary = 1.0 - validation_ratio - test_ratio
    if unit < train_boundary:
        return "train"
    if unit < train_boundary + validation_ratio:
        return "val"
    return "test"


def _teacher_pending_records(
    records: Sequence[dict[str, object]],
    *,
    decoded_duplicate_of: Mapping[str, str],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for record in records:
        record_id = str(record["record_id"])
        source = dict(record["source"])
        hashes = dict(record["hashes"])
        statusbar = dict(record["top_8_percent_statusbar"])
        canonical_id = decoded_duplicate_of.get(record_id)
        frame_count = int(dict(record["container"])["frame_count"])
        if canonical_id is not None:
            quarantine_reason = "decoded_pixel_duplicate"
            next_action = "exclude_exact_duplicate"
        elif frame_count > 1:
            quarantine_reason = "multiframe_first_frame_only"
            next_action = "exclude_or_expand_frames_in_a_separate_versioned_pipeline"
        else:
            quarantine_reason = None
            next_action = "paddle_teacher_ocr_and_validation"
        quarantined = quarantine_reason is not None
        rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "kind": TEACHER_PENDING_KIND,
                "record_id": record_id,
                "group_id": record["group_id"],
                "suggested_split": record["suggested_split"],
                "split_is_recommendation_not_authorization": True,
                "source_root": source["root"],
                "source_relative_path": source["relative_path"],
                "source_absolute_path": source["absolute_path"],
                "raw_sha256": hashes["raw_sha256"],
                "decoded_pixel_sha256": hashes["decoded_pixel_sha256"],
                "phash64": hashes["phash64"],
                "phash_usable": dict(record["full_image_quality"])["phash_usable"],
                "upright_width": dict(record["geometry"])["upright_width"],
                "upright_height": dict(record["geometry"])["upright_height"],
                "statusbar_presence_state": statusbar["presence_state"],
                "statusbar_presence_score": statusbar["presence_score"],
                "statusbar_quality_state": statusbar["quality_state"],
                "quality_flags": record["quality_flags"],
                "teacher_state": "quarantine" if quarantined else "pending",
                "next_action": next_action,
                "quarantine_reason": quarantine_reason,
                "canonical_record_id": canonical_id,
                "labels_present": False,
                "ocr_performed": False,
                "training_eligible": False,
                "low_confidence_or_conflict_policy": "exclude_or_quarantine_never_guess",
                "manual_review_required": False,
            }
        )
    return rows


def _layout_sample(
    records: Sequence[dict[str, object]],
    *,
    sample_size: int,
    seed: str,
) -> list[dict[str, object]]:
    group_members: dict[str, list[dict[str, object]]] = defaultdict(list)
    for record in records:
        group_members[str(record["group_id"])].append(record)
    representatives: list[dict[str, object]] = []
    member_count_by_group: dict[str, int] = {}
    for group_id, members in group_members.items():
        members.sort(
            key=lambda record: hashlib.sha256(
                (seed + "\0representative\0" + str(record["record_id"])).encode("utf-8")
            ).hexdigest()
        )
        representatives.append(members[0])
        member_count_by_group[group_id] = len(members)

    strata: dict[str, list[dict[str, object]]] = defaultdict(list)
    for record in representatives:
        geometry = dict(record["geometry"])
        statusbar = dict(record["top_8_percent_statusbar"])
        container = dict(record["container"])
        stratum = "/".join(
            (
                str(geometry["aspect_bucket"]),
                str(geometry["resolution_bucket"]),
                str(statusbar["presence_state"]),
                str(container["format"]).lower(),
            )
        )
        strata[stratum].append(record)
    for stratum, members in strata.items():
        members.sort(
            key=lambda record: hashlib.sha256(
                (seed + "\0" + stratum + "\0" + str(record["record_id"])).encode("utf-8")
            ).hexdigest()
        )
    selected: list[tuple[str, dict[str, object]]] = []
    stratum_names = sorted(strata)
    cursor = {name: 0 for name in stratum_names}
    while len(selected) < min(sample_size, len(representatives)):
        made_progress = False
        for name in stratum_names:
            index = cursor[name]
            if index >= len(strata[name]):
                continue
            selected.append((name, strata[name][index]))
            cursor[name] += 1
            made_progress = True
            if len(selected) >= min(sample_size, len(representatives)):
                break
        if not made_progress:
            break
    rows: list[dict[str, object]] = []
    for rank, (stratum, record) in enumerate(selected, start=1):
        source = dict(record["source"])
        rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "kind": LAYOUT_SAMPLE_KIND,
                "rank": rank,
                "stratum": stratum,
                "record_id": record["record_id"],
                "group_id": record["group_id"],
                "group_member_count": member_count_by_group[str(record["group_id"])],
                "one_representative_per_leakage_group": True,
                "suggested_split": record["suggested_split"],
                "source_relative_path": source["relative_path"],
                "source_absolute_path": source["absolute_path"],
                "geometry": record["geometry"],
                "container_format": dict(record["container"])["format"],
                "phash64": dict(record["hashes"])["phash64"],
                "top_8_percent_statusbar": record["top_8_percent_statusbar"],
                "image_copied": False,
                "ocr_performed": False,
            }
        )
    return rows


def _assert_source_closure(
    root: Path,
    original_files: Sequence[Path],
    observations: Mapping[Path, tuple[tuple[int, int, int, int], str | None]],
) -> None:
    observed_relative = [path.relative_to(root).as_posix() for path in original_files]
    current_files = _iter_regular_files(root)
    current_relative = [path.relative_to(root).as_posix() for path in current_files]
    if current_relative != observed_relative:
        raise SourceChangedError("source directory membership changed while inventory was being built")
    for path, (expected_signature, expected_sha256) in observations.items():
        before = _source_signature(path)
        if before != expected_signature:
            raise SourceChangedError(f"source metadata changed before closure verification: {path}")
        observed_sha256 = _raw_sha256(path) if expected_sha256 is not None else None
        after = _source_signature(path)
        if after != before or (expected_sha256 is not None and observed_sha256 != expected_sha256):
            raise SourceChangedError(f"source bytes changed before closure verification: {path}")


def _validate_options(
    *,
    phash_distance: int,
    layout_sample_size: int,
    validation_ratio: float,
    test_ratio: float,
    split_seed: str,
    layout_sample_seed: str,
    maximum_phash_candidates: int,
) -> None:
    if not 0 <= phash_distance <= 16:
        raise ValueError("phash_distance must be between 0 and 16")
    if layout_sample_size <= 0:
        raise ValueError("layout_sample_size must be positive")
    if maximum_phash_candidates <= 0:
        raise ValueError("maximum_phash_candidates must be positive")
    if not split_seed:
        raise ValueError("split_seed must not be empty")
    if not layout_sample_seed:
        raise ValueError("layout_sample_seed must not be empty")
    if any(not math.isfinite(value) or not 0.0 <= value < 1.0 for value in (validation_ratio, test_ratio)):
        raise ValueError("validation_ratio and test_ratio must be finite values in [0, 1)")
    if validation_ratio + test_ratio >= 1.0:
        raise ValueError("validation_ratio + test_ratio must be less than 1")


def build_otherimages_inventory(
    *,
    input_dir: Path,
    output_dir: Path,
    phash_distance: int = DEFAULT_PHASH_DISTANCE,
    layout_sample_size: int = DEFAULT_LAYOUT_SAMPLE_SIZE,
    validation_ratio: float = 0.10,
    test_ratio: float = 0.10,
    split_seed: str = DEFAULT_SPLIT_SEED,
    layout_sample_seed: str = DEFAULT_LAYOUT_SAMPLE_SEED,
    maximum_phash_candidates: int = DEFAULT_MAX_PHASH_CANDIDATES,
) -> dict[str, object]:
    """Inventory source images and atomically publish read-only JSON evidence.

    ``output_dir`` must not exist.  All output is built in a unique sibling
    directory, source membership and bytes are checked again, and the sibling
    is renamed into place only after ``inventory.contract.json`` is complete.
    """
    _validate_options(
        phash_distance=phash_distance,
        layout_sample_size=layout_sample_size,
        validation_ratio=validation_ratio,
        test_ratio=test_ratio,
        split_seed=split_seed,
        layout_sample_seed=layout_sample_seed,
        maximum_phash_candidates=maximum_phash_candidates,
    )
    _require_no_reparse_ancestors(input_dir)
    source_root = Path(input_dir).expanduser().resolve(strict=True)
    if not source_root.is_dir():
        raise NotADirectoryError(source_root)
    output = Path(os.path.abspath(os.path.expanduser(os.fspath(output_dir))))
    _require_no_reparse_ancestors(output, include_leaf=False)
    if output.exists():
        raise FileExistsError(f"output directory must be brand-new: {output}")
    if not output.parent.is_dir():
        raise NotADirectoryError(f"output parent directory must already exist: {output.parent}")
    output_parent_identity = _bind_stage_identity(output.parent, directory=True)
    if _paths_overlap(source_root, output):
        raise ValueError("input and output directories must not overlap")

    original_files = _iter_regular_files(source_root)
    if not original_files:
        raise ValueError(f"input directory contains no regular files: {source_root}")
    casefold_paths: dict[str, str] = {}
    for path in original_files:
        relative_path = path.relative_to(source_root).as_posix()
        folded = relative_path.casefold()
        previous = casefold_paths.setdefault(folded, relative_path)
        if previous != relative_path:
            raise ValueError(
                "source contains relative paths that collide under Windows case-folding: "
                f"{previous!r} and {relative_path!r}"
            )

    if _bind_stage_identity(output.parent, directory=True) != output_parent_identity:
        raise SourceChangedError("output parent identity changed before inventory staging")
    stage = Path(tempfile.mkdtemp(prefix=f".{output.name}.inventory-building-", dir=output.parent))
    stage_identity = _bind_stage_identity(stage, directory=True)
    published = False
    try:
        records: list[dict[str, object]] = []
        errors: list[dict[str, object]] = []
        ignored: list[dict[str, object]] = []
        observations: dict[Path, tuple[tuple[int, int, int, int], str | None]] = {}
        for path in original_files:
            relative_path = path.relative_to(source_root).as_posix()
            try:
                known_suffix = path.suffix.lower() in KNOWN_IMAGE_SUFFIXES
                if not known_suffix and not _probe_unknown_suffix(path):
                    signature = _source_signature(path)
                    observations[path] = (signature, None)
                    ignored.append(
                        {
                            "schema_version": SCHEMA_VERSION,
                            "relative_path": relative_path,
                            "size_bytes": signature[0],
                            "reason": "not_a_decodable_raster_image",
                        }
                    )
                    continue
                record, signature = _inspect_image(path, source_root)
            except SourceChangedError:
                raise
            except (
                Image.DecompressionBombWarning,
                Image.DecompressionBombError,
                UnidentifiedImageError,
                OSError,
                ValueError,
                SyntaxError,
            ) as error:
                signature = _source_signature(path)
                raw_sha256 = _raw_sha256(path)
                observations[path] = (signature, raw_sha256)
                errors.append(
                    {
                        "schema_version": SCHEMA_VERSION,
                        "relative_path": relative_path,
                        "absolute_path": str(path),
                        "raw_sha256": raw_sha256,
                        "size_bytes": signature[0],
                        "error_type": type(error).__name__,
                        "error": str(error)[:1024],
                        "disposition": "quarantine",
                        "ocr_performed": False,
                        "training_eligible": False,
                    }
                )
                continue
            records.append(record)
            observations[path] = (signature, str(dict(record["hashes"])["raw_sha256"]))
        if not records:
            raise ValueError("input directory contains no decodable raster images")

        records.sort(key=lambda record: _path_key(Path(str(dict(record["source"])["relative_path"]))))
        exact_raw, _raw_duplicate_of = _duplicate_groups(records, "raw_sha256")
        exact_decoded, decoded_duplicate_of = _duplicate_groups(records, "decoded_pixel_sha256")
        exact_duplicates = exact_raw + exact_decoded
        exact_duplicates.sort(key=lambda row: (str(row["identity"]), str(row["digest"])))
        near_duplicates, group_by_record_id, phash_summary = _phash_candidates_and_groups(
            records,
            maximum_distance=phash_distance,
            maximum_candidates=maximum_phash_candidates,
        )
        for record in records:
            group_id = group_by_record_id[str(record["record_id"])]
            record["group_id"] = group_id
            record["group_contract"] = "exact_decoded_plus_all_phash_radius_candidates_component_v2"
            record["suggested_split"] = _suggested_split(
                group_id,
                validation_ratio=validation_ratio,
                test_ratio=test_ratio,
                seed=split_seed,
            )
            record["split_is_recommendation_not_authorization"] = True

        teacher_pending = _teacher_pending_records(records, decoded_duplicate_of=decoded_duplicate_of)
        layout_sample = _layout_sample(records, sample_size=layout_sample_size, seed=layout_sample_seed)

        _write_jsonl(stage / "images.jsonl", records)
        _write_jsonl(stage / "exact_duplicates.jsonl", exact_duplicates)
        _write_jsonl(stage / "near_duplicate_candidates.jsonl", near_duplicates)
        _write_jsonl(stage / "layout_sample.jsonl", layout_sample)
        _write_jsonl(stage / "paddle_teacher_pending.jsonl", teacher_pending)
        _write_jsonl(stage / "ignored_non_images.jsonl", ignored)
        _write_jsonl(stage / "errors.jsonl", errors)

        _assert_source_closure(source_root, original_files, observations)
        format_counts = Counter(str(dict(record["container"])["format"]) for record in records)
        status_counts = Counter(
            str(dict(record["top_8_percent_statusbar"])["presence_state"])
            for record in records
        )
        quality_counts = Counter(
            str(dict(record["top_8_percent_statusbar"])["quality_state"])
            for record in records
        )
        split_counts = Counter(str(record["suggested_split"]) for record in records)
        teacher_state_counts = Counter(str(row["teacher_state"]) for row in teacher_pending)
        generated_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        contract: dict[str, object] = {
            "schema_version": SCHEMA_VERSION,
            "kind": CONTRACT_KIND,
            "generated_at_utc": generated_at,
            "source": {
                "input_directory": str(source_root),
                "recursive": True,
                "regular_files_observed": len(original_files),
                "source_membership_rechecked": True,
                "image_source_raw_sha256_rechecked": True,
                "ignored_non_image_metadata_rechecked": True,
                "source_mutation_detected": False,
                "symlinks_junctions_reparse_points_allowed": False,
            },
            "output": {
                "output_directory": str(output),
                "must_be_brand_new": True,
                "publication": "sibling_staging_atomic_platform_no_replace_rename_v2",
                "publication_parent_identity_checked": True,
                "source_images_copied": False,
            },
            "configuration": {
                "top_strip_fraction": TOP_STRIP_FRACTION,
                "phash_distance": phash_distance,
                "phash_candidate_cap": maximum_phash_candidates,
                "layout_sample_size_requested": layout_sample_size,
                "layout_sample_seed": layout_sample_seed,
                "split_seed": split_seed,
                "validation_ratio": validation_ratio,
                "test_ratio": test_ratio,
                "train_ratio": 1.0 - validation_ratio - test_ratio,
                "split_grouping": "exact_decoded_plus_all_phash_radius_candidates_component_v2",
            },
            "counts": {
                "images": len(records),
                "image_errors_quarantined": len(errors),
                "ignored_non_images": len(ignored),
                "exact_raw_duplicate_groups": len(exact_raw),
                "exact_decoded_pixel_duplicate_groups": len(exact_decoded),
                "layout_sample_records": len(layout_sample),
                "formats": dict(sorted(format_counts.items())),
                "statusbar_presence_states": dict(sorted(status_counts.items())),
                "statusbar_quality_states": dict(sorted(quality_counts.items())),
                "suggested_splits": dict(sorted(split_counts.items())),
                "teacher_states": dict(sorted(teacher_state_counts.items())),
            },
            "phash_candidates": phash_summary,
            "paddle_teacher_contract": {
                "manifest": "paddle_teacher_pending.jsonl",
                "inventory_contains_labels": False,
                "inventory_performed_ocr": False,
                "inventory_performed_training": False,
                "manual_review_required": False,
                "pending_records_may_be_processed_automatically": True,
                "exact_decoded_duplicates": "quarantine_noncanonical",
                "near_phash_candidates": (
                    "all radius matches share one suggested split regardless of aspect delta; not automatic duplicate drop"
                ),
                "downstream_acceptance_rule": (
                    "accept only independently validated Paddle teacher output meeting configured confidence and "
                    "agreement gates; low-confidence or conflicting output must be excluded or quarantined"
                ),
                "guessed_or_synthetic_labels_forbidden": True,
                "training_eligibility_before_teacher_validation": False,
            },
            "artifacts": [_artifact_binding(stage / name) for name in OUTPUT_FILENAMES],
        }
        _write_json(stage / "inventory.contract.json", contract)
        contract_binding = _artifact_binding(stage / "inventory.contract.json")
        _assert_source_closure(source_root, original_files, observations)
        if output.exists():
            raise FileExistsError(f"output directory appeared during inventory build: {output}")
        _rename_directory_no_replace(
            stage,
            output,
            expected_parent_identity=output_parent_identity,
            expected_stage_identity=stage_identity,
        )
        if _bind_stage_identity(output.parent, directory=True) != output_parent_identity:
            raise SourceChangedError("inventory output parent changed after publication")
        if _bind_stage_identity(output, directory=True) != stage_identity:
            raise SourceChangedError("published inventory directory differs from bound stage")
        expected_members = {*OUTPUT_FILENAMES, "inventory.contract.json"}
        if {path.name for path in output.iterdir()} != expected_members:
            raise SourceChangedError("published inventory directory membership differs after publication")
        for expected in contract["artifacts"]:
            if not isinstance(expected, Mapping) or _artifact_binding(output / str(expected["path"])) != expected:
                raise SourceChangedError("published inventory artifact failed exact readback")
        if _artifact_binding(output / "inventory.contract.json") != contract_binding:
            raise SourceChangedError("published inventory contract failed exact readback")
        if _bind_stage_identity(output.parent, directory=True) != output_parent_identity:
            raise SourceChangedError("inventory output parent changed during publication readback")
        if _bind_stage_identity(output, directory=True) != stage_identity:
            raise SourceChangedError("published inventory identity changed during readback")
        published = True
        return contract
    finally:
        if not published:
            # Failure stages are evidence.  Never recursively delete through a
            # pathname that another process could have replaced.
            pass


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Read-only recursive image inventory for OtherImages; emits hashes, EXIF/geometry, top-8% status-bar "
            "statistics, pHash candidates, layout sampling, and an unlabeled Paddle teacher manifest"
        )
    )
    parser.add_argument("--input", type=Path, required=True, help=r"source image directory, e.g. D:\download2\OtherImages")
    parser.add_argument("--output", type=Path, required=True, help="brand-new external evidence directory")
    parser.add_argument("--phash-distance", type=int, default=DEFAULT_PHASH_DISTANCE)
    parser.add_argument("--layout-sample-size", type=int, default=DEFAULT_LAYOUT_SAMPLE_SIZE)
    parser.add_argument("--validation-ratio", type=float, default=0.10)
    parser.add_argument("--test-ratio", type=float, default=0.10)
    parser.add_argument("--split-seed", default=DEFAULT_SPLIT_SEED)
    parser.add_argument("--layout-sample-seed", default=DEFAULT_LAYOUT_SAMPLE_SEED)
    parser.add_argument("--max-phash-candidates", type=int, default=DEFAULT_MAX_PHASH_CANDIDATES)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    arguments = build_parser().parse_args(argv)
    try:
        contract = build_otherimages_inventory(
            input_dir=arguments.input,
            output_dir=arguments.output,
            phash_distance=arguments.phash_distance,
            layout_sample_size=arguments.layout_sample_size,
            validation_ratio=arguments.validation_ratio,
            test_ratio=arguments.test_ratio,
            split_seed=arguments.split_seed,
            layout_sample_seed=arguments.layout_sample_seed,
            maximum_phash_candidates=arguments.max_phash_candidates,
        )
    except (OSError, RuntimeError, ValueError) as error:
        raise SystemExit(f"OtherImages read-only inventory failed:\n{error}") from None
    counts = dict(contract["counts"])
    print(
        f"Inventoried {counts['images']} image(s); quarantined {counts['image_errors_quarantined']} decode error(s); "
        f"published read-only evidence to {Path(arguments.output).expanduser().absolute()}"
    )


if __name__ == "__main__":  # pragma: no cover
    main()
