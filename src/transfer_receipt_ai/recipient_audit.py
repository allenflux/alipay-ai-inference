"""Deterministic image-only geometry diagnostics for recipient crop trims.

The v11 recipient reader removes a frozen fraction from the left side of its
field crop before resizing.  This module makes that geometry observable without
loading an ONNX model or inspecting any OCR text.  It deliberately works on the
stored pre-resize crop so a caller can distinguish an unsafe cut through visible
ink from a recognition-model error.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image


FOREGROUND_DETECTION_METHOD = "dominant_grayscale_mode_abs_difference"
DEFAULT_FOREGROUND_CONTRAST_THRESHOLD = 24
DEFAULT_CUT_RADIUS = 2


@dataclass(frozen=True)
class BlankColumnGap:
    """One contiguous sequence of nearly blank source-image columns.

    ``end_px_exclusive`` follows Python slice semantics.  ``distance_to_trim_px``
    measures the shortest distance from the v11 trim boundary to the gap,
    treating a boundary exactly on either edge as distance zero.
    """

    start_px: int
    end_px_exclusive: int
    width_px: int
    distance_to_trim_px: int
    touches_trim_boundary: bool


@dataclass(frozen=True)
class RecipientCropGeometryAudit:
    """Serializable geometry evidence for one recipient crop.

    The trim boundary is the first retained pixel column: source columns
    ``[0, trim_px)`` are removed and ``[trim_px, width)`` are retained.  A pixel
    counts as ink when its grayscale value differs from the crop's dominant
    grayscale background by at least ``foreground_contrast_threshold``.  That
    detects both black text on white and white text on a blue UI background.
    """

    width: int
    height: int
    left_trim_ratio: float
    trim_px: int
    retained_width_px: int
    retained_width_ratio: float
    retained_aspect_ratio: float
    foreground_detection_method: str
    dominant_background_luma: int
    foreground_contrast_threshold: int
    blank_column_max_ink: int
    total_ink_pixels: int
    left_ink_pixels: int
    right_ink_pixels: int
    cut_window_start_px: int
    cut_window_end_px_exclusive: int
    cut_window_ink_pixels: int
    cut_column_indices: tuple[int, ...]
    cut_column_ink_counts: tuple[int, ...]
    nearest_blank_gap: BlankColumnGap | None
    image_path: str | None = None

    @property
    def cut_window_has_ink(self) -> bool:
        """Whether any visible ink falls within the configured cut window."""
        return self.cut_window_ink_pixels > 0

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-serialisable representation with stable primitive values."""
        payload = asdict(self)
        # ``cut_window_has_ink`` is intentionally a derived property rather
        # than stored mutable state, but the batch audit needs it in JSONL for
        # aggregate evidence and simple filtering.
        payload["cut_window_has_ink"] = self.cut_window_has_ink
        return payload


def _validate_trim_ratio(left_trim_ratio: float) -> float:
    try:
        ratio = float(left_trim_ratio)
    except (TypeError, ValueError):
        raise ValueError("left_trim_ratio must be numeric") from None
    if not math.isfinite(ratio) or not 0.0 <= ratio < 1.0:
        raise ValueError("left_trim_ratio must be in [0, 1)")
    return ratio


def _validate_nonnegative_int(value: int, *, name: str, maximum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    integer = int(value)
    if integer < 0 or (maximum is not None and integer > maximum):
        detail = f"between 0 and {maximum}" if maximum is not None else "non-negative"
        raise ValueError(f"{name} must be {detail}")
    return integer


def _validate_positive_int(value: int, *, name: str, maximum: int | None = None) -> int:
    integer = _validate_nonnegative_int(value, name=name, maximum=maximum)
    if integer == 0:
        raise ValueError(f"{name} must be positive")
    return integer


def _uint8_grayscale(pixels: np.ndarray) -> np.ndarray:
    """Validate a two-dimensional grayscale array and convert it deterministically."""
    array = np.asarray(pixels)
    if array.ndim != 2:
        raise ValueError("recipient crop pixels must be a two-dimensional grayscale array")
    if not array.size or array.shape[0] <= 0 or array.shape[1] <= 0:
        raise ValueError("recipient crop pixels must be non-empty")
    if not np.issubdtype(array.dtype, np.number):
        raise ValueError("recipient crop pixels must be numeric")
    if not np.isfinite(array).all():
        raise ValueError("recipient crop pixels must be finite")
    if array.dtype == np.uint8:
        return np.ascontiguousarray(array)
    return np.clip(np.rint(array), 0, 255).astype(np.uint8, copy=False)


def dominant_background_luma(pixels: np.ndarray) -> int:
    """Return the deterministic grayscale-mode estimate of a crop's background.

    Crops typically contain far more background than text.  A grayscale mode
    avoids assuming a white background, which is important for Alipay's blue
    panels where the foreground text is bright instead of dark.
    """
    gray = _uint8_grayscale(pixels)
    counts = np.bincount(gray.reshape(-1), minlength=256)
    return int(np.argmax(counts))


def foreground_ink_mask(
    pixels: np.ndarray,
    *,
    contrast_threshold: int = DEFAULT_FOREGROUND_CONTRAST_THRESHOLD,
) -> tuple[np.ndarray, int]:
    """Return an image-only foreground mask and its dominant background luma.

    The mask is based solely on an absolute grayscale difference from the mode.
    No OCR model, text label, or color-specific heuristic participates.
    """
    gray = _uint8_grayscale(pixels)
    threshold = _validate_positive_int(
        contrast_threshold,
        name="contrast_threshold",
        maximum=255,
    )
    background = dominant_background_luma(gray)
    difference = np.abs(gray.astype(np.int16) - background)
    return difference >= threshold, background


def nearest_blank_column_gap(
    column_ink_counts: Sequence[int] | np.ndarray,
    *,
    trim_px: int,
    blank_column_max_ink: int = 0,
) -> BlankColumnGap | None:
    """Find the blank column run nearest a crop's left-trim boundary.

    A column is blank if its ink count is at most ``blank_column_max_ink``.
    Ties prefer the wider gap and then the earlier source position so results
    remain reproducible.  ``trim_px`` may equal the image width to support a
    caller auditing an arbitrary boundary, even though v11 itself caps it at
    ``width - 1``.
    """
    counts = np.asarray(column_ink_counts)
    if counts.ndim != 1 or counts.size == 0:
        raise ValueError("column_ink_counts must be a non-empty one-dimensional sequence")
    if not np.issubdtype(counts.dtype, np.number) or not np.isfinite(counts).all():
        raise ValueError("column_ink_counts must contain finite numeric values")
    if np.any(counts < 0):
        raise ValueError("column_ink_counts must not contain negative values")
    trim = _validate_nonnegative_int(trim_px, name="trim_px", maximum=int(counts.size))
    max_ink = _validate_nonnegative_int(blank_column_max_ink, name="blank_column_max_ink")

    gaps: list[BlankColumnGap] = []
    start: int | None = None
    for index, count in enumerate(counts.tolist()):
        if count <= max_ink:
            if start is None:
                start = index
            continue
        if start is not None:
            gaps.append(_blank_gap(start, index, trim))
            start = None
    if start is not None:
        gaps.append(_blank_gap(start, int(counts.size), trim))
    if not gaps:
        return None
    return min(gaps, key=lambda gap: (gap.distance_to_trim_px, -gap.width_px, gap.start_px))


def _blank_gap(start_px: int, end_px_exclusive: int, trim_px: int) -> BlankColumnGap:
    # A trim boundary at either outside edge is safe: it sits adjacent to, not
    # through, the blank gap.  This also makes an all-blank image return zero.
    distance = 0 if start_px <= trim_px <= end_px_exclusive else min(
        abs(trim_px - start_px), abs(trim_px - end_px_exclusive)
    )
    return BlankColumnGap(
        start_px=start_px,
        end_px_exclusive=end_px_exclusive,
        width_px=end_px_exclusive - start_px,
        distance_to_trim_px=distance,
        touches_trim_boundary=distance == 0,
    )


def audit_recipient_pixels(
    pixels: np.ndarray,
    *,
    left_trim_ratio: float,
    foreground_contrast_threshold: int = DEFAULT_FOREGROUND_CONTRAST_THRESHOLD,
    cut_radius: int = DEFAULT_CUT_RADIUS,
    blank_column_max_ink: int = 0,
) -> RecipientCropGeometryAudit:
    """Audit v11-style left trimming for an in-memory grayscale recipient crop.

    This is the pure-ish core API used by tests and batch callers.  It does not
    resize, OCR, or mutate ``pixels``.  The trim calculation exactly matches
    ``ocr_unified.preprocess_image``: ``round(width * ratio)``, clamped to the
    last valid source column.  Foreground ink is detected relative to the
    dominant crop background, not by assuming a white UI surface.
    """
    gray = _uint8_grayscale(pixels)
    height, width = (int(value) for value in gray.shape)
    ratio = _validate_trim_ratio(left_trim_ratio)
    contrast_threshold = _validate_positive_int(
        foreground_contrast_threshold,
        name="foreground_contrast_threshold",
        maximum=255,
    )
    radius = _validate_nonnegative_int(cut_radius, name="cut_radius")
    max_blank_ink = _validate_nonnegative_int(
        blank_column_max_ink,
        name="blank_column_max_ink",
        maximum=height,
    )

    trim_px = min(width - 1, max(0, int(round(width * ratio))))
    ink, background_luma = foreground_ink_mask(gray, contrast_threshold=contrast_threshold)
    column_ink_counts = ink.sum(axis=0, dtype=np.int64)
    cut_start = max(0, trim_px - radius)
    cut_end = min(width, trim_px + radius + 1)
    cut_indices = tuple(range(cut_start, cut_end))
    cut_counts = tuple(int(value) for value in column_ink_counts[cut_start:cut_end])
    nearest_gap = nearest_blank_column_gap(
        column_ink_counts,
        trim_px=trim_px,
        blank_column_max_ink=max_blank_ink,
    )
    return RecipientCropGeometryAudit(
        width=width,
        height=height,
        left_trim_ratio=ratio,
        trim_px=trim_px,
        retained_width_px=width - trim_px,
        retained_width_ratio=(width - trim_px) / width,
        retained_aspect_ratio=(width - trim_px) / height,
        foreground_detection_method=FOREGROUND_DETECTION_METHOD,
        dominant_background_luma=background_luma,
        foreground_contrast_threshold=contrast_threshold,
        blank_column_max_ink=max_blank_ink,
        total_ink_pixels=int(ink.sum(dtype=np.int64)),
        left_ink_pixels=int(ink[:, :trim_px].sum(dtype=np.int64)),
        right_ink_pixels=int(ink[:, trim_px:].sum(dtype=np.int64)),
        cut_window_start_px=cut_start,
        cut_window_end_px_exclusive=cut_end,
        cut_window_ink_pixels=int(ink[:, cut_start:cut_end].sum(dtype=np.int64)),
        cut_column_indices=cut_indices,
        cut_column_ink_counts=cut_counts,
        nearest_blank_gap=nearest_gap,
    )


def audit_recipient_crop(
    image_path: str | Path,
    *,
    left_trim_ratio: float,
    foreground_contrast_threshold: int = DEFAULT_FOREGROUND_CONTRAST_THRESHOLD,
    cut_radius: int = DEFAULT_CUT_RADIUS,
    blank_column_max_ink: int = 0,
) -> RecipientCropGeometryAudit:
    """Load one crop and return deterministic image-only v11 trim evidence."""
    path = Path(image_path).expanduser().resolve()
    with Image.open(path) as image:
        gray = np.asarray(image.convert("L"), dtype=np.uint8).copy()
    audit = audit_recipient_pixels(
        gray,
        left_trim_ratio=left_trim_ratio,
        foreground_contrast_threshold=foreground_contrast_threshold,
        cut_radius=cut_radius,
        blank_column_max_ink=blank_column_max_ink,
    )
    return replace(audit, image_path=path.as_posix())
