"""Torch-free validation baseline for receipt font-domain consistency.

This module intentionally implements a conservative *rendering-domain*
classifier rather than exact font-file identification.  It extracts a fixed
64-dimensional, polarity-tolerant feature vector from a text-line crop and
compares it with robust per-domain prototypes.  Low-information, ambiguous or
out-of-distribution crops are rejected as ``unknown`` by :mod:`font_domain`.

The JSON model is self-describing and carries a canonical SHA-256 over every
decision-relevant field.  Loading is strict (duplicate keys, non-finite
numbers, malformed shapes and hash mismatches are rejected).  This makes the
baseline suitable for reproducible experiments; it does not make its output
an authenticity verdict.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import secrets
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Final

import cv2
import numpy as np
from PIL import Image, ImageOps, __version__ as PILLOW_VERSION

from .font_domain import (
    SCHEMA_VERSION,
    UNKNOWN_DOMAIN,
    FontDomainConsistencyResult,
    FontDomainLinePrediction,
    aggregate_font_domain_predictions,
    prediction_from_probabilities,
)
from .font_domain_dataset import (
    PERCEPTUAL_HASH_ABI,
    FontDomainDataset,
    FontDomainDocument,
    FontDomainRegion,
)


MODEL_KIND: Final[str] = "receipt_font_domain_prototype_model_v1"
FEATURE_ABI: Final[str] = "font-domain-classical-64-v1"
FEATURE_DIMENSION: Final[int] = 64
MAXIMUM_MODEL_BYTES: Final[int] = 16 * 1024 * 1024
MAXIMUM_CONFORMAL_CALIBRATION_COUNT: Final[int] = 100_000
_DOMAIN_PATTERN = re.compile(r"^[a-z0-9][a-z0-9._-]{0,63}$")

# Blocks are individually averaged before weighting.  Consequently a large
# histogram cannot dominate a small but useful stroke-width block merely by
# containing more bins.
FEATURE_BLOCKS: Final[tuple[tuple[str, int, int, float], ...]] = (
    ("lbp", 0, 20, 0.20),
    ("hog", 20, 29, 0.15),
    ("stroke_width", 29, 34, 0.25),
    ("erosion", 34, 38, 0.15),
    ("components", 38, 56, 0.20),
    ("edge_tone", 56, 64, 0.05),
)
_QUANTILES_3: Final[tuple[float, ...]] = (0.25, 0.50, 0.75)
_QUANTILES_5: Final[tuple[float, ...]] = (0.10, 0.25, 0.50, 0.75, 0.90)


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _finite(value: object, *, description: str, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{description} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{description} must be finite")
    if minimum is not None and result < minimum:
        raise ValueError(f"{description} must be at least {minimum}")
    return result


def _unit(value: object, *, description: str) -> float:
    result = _finite(value, description=description)
    if not 0.0 <= result <= 1.0:
        raise ValueError(f"{description} must be between 0 and 1")
    return result


def _nonempty(value: object, *, description: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{description} must be a non-empty string")
    return value.strip()


def _vector(
    value: object,
    *,
    description: str,
    length: int = FEATURE_DIMENSION,
    minimum: float | None = None,
) -> tuple[float, ...]:
    if not isinstance(value, (list, tuple)) or len(value) != length:
        raise ValueError(f"{description} must contain exactly {length} numbers")
    return tuple(
        _finite(item, description=f"{description}[{index}]", minimum=minimum)
        for index, item in enumerate(value)
    )


def _normalized_histogram(values: np.ndarray, bins: int, value_range: tuple[float, float]) -> np.ndarray:
    if values.size == 0:
        return np.zeros(bins, dtype=np.float64)
    histogram, _ = np.histogram(values, bins=bins, range=value_range)
    result = histogram.astype(np.float64)
    total = float(result.sum())
    if total > 0.0:
        result /= total
    return result


def _load_upright_rgb(path: Path) -> np.ndarray:
    try:
        with Image.open(path) as opened:
            rgb = np.asarray(ImageOps.exif_transpose(opened).convert("RGB"), dtype=np.uint8)
    except (OSError, ValueError) as error:
        raise ValueError(f"unable to decode region image {path}: {error}") from error
    if rgb.ndim != 3 or rgb.shape[2] != 3 or min(rgb.shape[:2]) < 1:
        raise ValueError(f"decoded region image is empty or invalid: {path}")
    return np.ascontiguousarray(rgb)


def _pixel_sha256(rgb: np.ndarray) -> str:
    pixels = np.ascontiguousarray(rgb, dtype=np.uint8)
    digest = hashlib.sha256()
    digest.update(str(tuple(int(item) for item in pixels.shape)).encode("ascii"))
    digest.update(b"\0uint8\0RGB\0")
    digest.update(pixels.tobytes(order="C"))
    return digest.hexdigest()


def _raw_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_bound_region_rgb(region: FontDomainRegion) -> np.ndarray:
    """Reload a region through the dataset's binding API when available.

    ``FontDomainRegion.load_bound_rgb`` verifies both file bytes and decoded
    pixels and is the canonical TOCTOU guard.  The explicit fallback keeps this
    module source-compatible with the first manifest-contract revision while
    applying the same two bindings itself.
    """

    loader = getattr(region, "load_bound_rgb", None)
    if callable(loader):
        rgb = np.asarray(loader(), dtype=np.uint8)
        if rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError(f"bound region loader returned invalid RGB: {region.region_id}")
        return np.ascontiguousarray(rgb)
    if _raw_sha256(region.image_path) != region.raw_sha256:
        raise ValueError(f"region image bytes changed after manifest validation: {region.region_id}")
    rgb = _load_upright_rgb(region.image_path)
    if _pixel_sha256(rgb) != region.pixel_sha256:
        raise ValueError(f"region decoded pixels changed after manifest validation: {region.region_id}")
    return rgb


@dataclass(frozen=True)
class RegionFeatureResult:
    """One crop's deterministic feature vector and information gate."""

    values: tuple[float, ...]
    quality: float
    usable: bool
    reasons: tuple[str, ...]
    metrics: Mapping[str, float]

    def __post_init__(self) -> None:
        _vector(self.values, description="feature values")
        _unit(self.quality, description="feature quality")
        if not isinstance(self.usable, bool):
            raise ValueError("feature usable must be boolean")
        if not isinstance(self.reasons, tuple) or not self.reasons:
            raise ValueError("feature reasons must be a non-empty tuple")
        for reason in self.reasons:
            _nonempty(reason, description="feature reason")
        for name, value in self.metrics.items():
            _nonempty(name, description="feature metric name")
            _finite(value, description=f"feature metric {name}", minimum=0.0)

    def as_dict(self) -> dict[str, object]:
        return {
            "feature_abi": FEATURE_ABI,
            "dimension": FEATURE_DIMENSION,
            "quality": round(float(self.quality), 6),
            "usable": self.usable,
            "reasons": list(self.reasons),
            "metrics": {
                key: round(float(self.metrics[key]), 6) for key in sorted(self.metrics)
            },
        }


def _connected_components(mask: np.ndarray) -> tuple[np.ndarray, list[tuple[int, int, int, int, int]]]:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    kept: list[tuple[int, int, int, int, int]] = []
    cleaned = np.zeros_like(mask, dtype=np.uint8)
    for label in range(1, count):
        x, y, width, height, area = (int(item) for item in stats[label])
        if area < 3:
            continue
        cleaned[labels == label] = 1
        kept.append((x, y, width, height, area))
    return cleaned, kept


def _uniform_lbp_histogram(image: np.ndarray, mask: np.ndarray, radius: int) -> np.ndarray:
    if image.shape[0] <= radius * 2 or image.shape[1] <= radius * 2:
        return np.zeros(10, dtype=np.float64)
    # Clockwise integer samples.  Radius two intentionally remains an integer
    # neighbourhood so OpenCV/Pillow interpolation differences cannot alter
    # the feature ABI.
    offsets = (
        (-radius, 0),
        (-radius, radius),
        (0, radius),
        (radius, radius),
        (radius, 0),
        (radius, -radius),
        (0, -radius),
        (-radius, -radius),
    )
    center = image[radius:-radius, radius:-radius]
    bits: list[np.ndarray] = []
    height, width = image.shape
    for dy, dx in offsets:
        neighbour = image[radius + dy : height - radius + dy, radius + dx : width - radius + dx]
        bits.append(neighbour >= center)
    stack = np.stack(bits, axis=0).astype(np.uint8)
    transitions = np.sum(stack != np.roll(stack, shift=1, axis=0), axis=0)
    ones = np.sum(stack, axis=0)
    codes = np.where(transitions <= 2, ones, 9).astype(np.int32)
    support = cv2.dilate(mask.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=1)
    selected = codes[support[radius:-radius, radius:-radius].astype(bool)]
    if selected.size == 0:
        return np.zeros(10, dtype=np.float64)
    histogram = np.bincount(selected, minlength=10).astype(np.float64)
    return histogram / max(float(histogram.sum()), 1.0)


def _feature_canvas(
    contrast: np.ndarray,
    mask: np.ndarray,
    components: Sequence[tuple[int, int, int, int, int]],
) -> tuple[np.ndarray, np.ndarray, float]:
    if components:
        left = min(item[0] for item in components)
        top = min(item[1] for item in components)
        right = max(item[0] + item[2] for item in components)
        bottom = max(item[1] + item[3] for item in components)
        pad = 2
        left, top = max(0, left - pad), max(0, top - pad)
        right, bottom = min(mask.shape[1], right + pad), min(mask.shape[0], bottom + pad)
        contrast = contrast[top:bottom, left:right]
        mask = mask[top:bottom, left:right]
        median_height = float(np.median([item[3] for item in components]))
    else:
        median_height = 1.0
    scale = float(np.clip(24.0 / max(median_height, 1.0), 0.5, 4.0))
    target_width = max(8, min(1024, int(round(contrast.shape[1] * scale))))
    target_height = max(8, min(256, int(round(contrast.shape[0] * scale))))
    resized_contrast = cv2.resize(
        contrast.astype(np.float32), (target_width, target_height), interpolation=cv2.INTER_LINEAR
    )
    resized_mask = cv2.resize(
        mask.astype(np.uint8), (target_width, target_height), interpolation=cv2.INTER_NEAREST
    )
    return resized_contrast, resized_mask, scale


def extract_font_domain_features(rgb: np.ndarray) -> RegionFeatureResult:
    """Extract the fixed 64-D validation feature from an RGB text-line crop.

    The foreground is defined by absolute deviation from the border median, so
    dark-on-light and light-on-dark renderings share the same representation.
    Returned vectors remain finite even for rejected/blank inputs, allowing a
    caller to emit useful ``UNKNOWN`` evidence instead of crashing.
    """

    array = np.asarray(rgb)
    if array.ndim != 3 or array.shape[2] != 3 or min(array.shape[:2]) < 1:
        raise ValueError("rgb must be a non-empty HxWx3 array")
    if not np.issubdtype(array.dtype, np.number) or not np.all(np.isfinite(array)):
        raise ValueError("rgb must contain finite numeric samples")
    array = np.clip(array, 0, 255).astype(np.uint8)
    height, width = (int(item) for item in array.shape[:2])
    gray = cv2.cvtColor(array, cv2.COLOR_RGB2GRAY).astype(np.float32)
    border = np.concatenate((gray[0, :], gray[-1, :], gray[:, 0], gray[:, -1]))
    background = float(np.median(border))
    contrast = np.abs(gray - background)
    contrast_u8 = np.clip(np.rint(contrast), 0, 255).astype(np.uint8)
    otsu_threshold, raw_mask = cv2.threshold(
        contrast_u8, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    # Otsu returns a full foreground for a constant non-zero image.  Absolute
    # minimum contrast prevents this from turning a flat crop into text.
    threshold = max(float(otsu_threshold), 7.0)
    raw_mask = (contrast > threshold).astype(np.uint8)
    mask, components = _connected_components(raw_mask)
    ink_pixels = int(mask.sum())
    foreground_ratio = ink_pixels / max(float(width * height), 1.0)
    component_count = len(components)
    component_heights = [item[3] for item in components]
    median_component_height = float(np.median(component_heights)) if component_heights else 0.0
    contrast_p95 = float(np.percentile(contrast, 95.0))
    ring = np.zeros_like(mask, dtype=np.uint8)
    ring[:2, :] = 1
    ring[-2:, :] = 1
    ring[:, :2] = 1
    ring[:, -2:] = 1
    border_ink_ratio = float((mask * ring).sum()) / max(float(ink_pixels), 1.0)

    reasons: list[str] = []
    if width < 48:
        reasons.append("WIDTH_BELOW_48")
    if height < 24:
        reasons.append("HEIGHT_BELOW_24")
    if ink_pixels < 128:
        reasons.append("INK_BELOW_128")
    if component_count < 3:
        reasons.append("COMPONENTS_BELOW_3")
    if median_component_height < 8.0:
        reasons.append("COMPONENT_HEIGHT_BELOW_8")
    if not 0.01 <= foreground_ratio <= 0.55:
        reasons.append("FOREGROUND_RATIO_OUT_OF_RANGE")
    if contrast_p95 < 20.0:
        reasons.append("CONTRAST_BELOW_20")
    if border_ink_ratio > 0.25:
        reasons.append("BORDER_INK_ABOVE_0_25")

    scores = (
        min(1.0, width / 48.0),
        min(1.0, height / 24.0),
        min(1.0, ink_pixels / 128.0),
        min(1.0, component_count / 3.0),
        min(1.0, median_component_height / 8.0),
        min(1.0, foreground_ratio / 0.01) if foreground_ratio <= 0.55 else 0.55 / foreground_ratio,
        min(1.0, contrast_p95 / 20.0),
        min(1.0, 0.25 / max(border_ink_ratio, 1e-9)),
    )
    quality = float(math.exp(sum(math.log(max(value, 1e-9)) for value in scores) / len(scores)))
    if reasons:
        quality = min(quality, 0.249999)

    feature_contrast, feature_mask, scale = _feature_canvas(contrast, mask, components)
    lbp = np.concatenate(
        (
            _uniform_lbp_histogram(feature_contrast, feature_mask, 1),
            _uniform_lbp_histogram(feature_contrast, feature_mask, 2),
        )
    )

    dx = cv2.Sobel(feature_contrast, cv2.CV_32F, 1, 0, ksize=3)
    dy = cv2.Sobel(feature_contrast, cv2.CV_32F, 0, 1, ksize=3)
    magnitude, angle = cv2.cartToPolar(dx, dy, angleInDegrees=True)
    angle = np.mod(angle, 180.0)
    hog_support = cv2.dilate(feature_mask, np.ones((3, 3), np.uint8), iterations=1).astype(bool)
    hog = np.zeros(9, dtype=np.float64)
    if np.any(hog_support):
        bins = np.minimum((angle[hog_support] / 20.0).astype(np.int32), 8)
        hog = np.bincount(bins, weights=magnitude[hog_support], minlength=9).astype(np.float64)
        if float(hog.sum()) > 0.0:
            hog /= float(hog.sum())

    distance = cv2.distanceTransform(feature_mask.astype(np.uint8), cv2.DIST_L2, 5)
    local_maximum = distance >= cv2.dilate(distance, np.ones((3, 3), np.uint8)) - 1e-6
    stroke_samples = (2.0 * distance[local_maximum & (distance > 0.0)]) / 24.0
    if stroke_samples.size:
        stroke = np.quantile(stroke_samples, _QUANTILES_5).astype(np.float64)
    else:
        stroke = np.zeros(5, dtype=np.float64)
    stroke = np.clip(stroke, 0.0, 4.0)

    base_ink = max(float(feature_mask.sum()), 1.0)
    erosion = np.asarray(
        [
            float(
                cv2.erode(
                    feature_mask.astype(np.uint8),
                    cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size)),
                    iterations=1,
                ).sum()
            )
            / base_ink
            for size in (3, 5, 7, 9)
        ],
        dtype=np.float64,
    )

    component_properties: list[list[float]] = [[] for _ in range(6)]
    median_height = max(median_component_height, 1.0)
    for x, y, component_width, component_height, area in components:
        local = mask[y : y + component_height, x : x + component_width].astype(np.uint8)
        contours, _ = cv2.findContours(local, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        perimeter = sum(float(cv2.arcLength(contour, True)) for contour in contours)
        values = (
            component_width / median_height,
            component_height / median_height,
            area / max(float(component_width * component_height), 1.0),
            area / (median_height * median_height),
            component_width / max(float(component_height), 1.0),
            perimeter / max(float(2 * (component_width + component_height)), 1.0),
        )
        for collection, value in zip(component_properties, values, strict=True):
            collection.append(float(np.clip(value, 0.0, 8.0)))
    component_features: list[float] = []
    for values in component_properties:
        if values:
            component_features.extend(float(item) for item in np.quantile(values, _QUANTILES_3))
        else:
            component_features.extend((0.0, 0.0, 0.0))

    edge = feature_mask.astype(bool) & ~cv2.erode(
        feature_mask.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=1
    ).astype(bool)
    edge_tone = _normalized_histogram(feature_contrast[edge], 8, (0.0, 256.0))

    vector = np.concatenate(
        (
            lbp,
            hog,
            stroke,
            erosion,
            np.asarray(component_features, dtype=np.float64),
            edge_tone,
        )
    )
    if vector.shape != (FEATURE_DIMENSION,) or not np.all(np.isfinite(vector)):
        raise RuntimeError("font-domain feature ABI produced an invalid vector")
    metrics = {
        "width": float(width),
        "height": float(height),
        "ink_pixels": float(ink_pixels),
        "component_count": float(component_count),
        "median_component_height": median_component_height,
        "foreground_ratio": foreground_ratio,
        "contrast_p95": contrast_p95,
        "border_ink_ratio": border_ink_ratio,
        "otsu_threshold": float(otsu_threshold),
        "normalization_scale": scale,
    }
    return RegionFeatureResult(
        values=tuple(float(item) for item in vector),
        quality=float(np.clip(quality, 0.0, 1.0)),
        usable=not reasons and quality >= 0.25,
        reasons=tuple(reasons or ("INFORMATION_GATE_PASSED",)),
        metrics=metrics,
    )


@dataclass(frozen=True)
class FontDomainGates:
    confidence: float = 0.60
    margin: float = 0.08
    quality: float = 0.25
    fit_p_value: float = 0.05

    def __post_init__(self) -> None:
        _unit(self.confidence, description="confidence gate")
        _unit(self.margin, description="margin gate")
        _unit(self.quality, description="quality gate")
        _unit(self.fit_p_value, description="fit p-value gate")

    def as_dict(self) -> dict[str, float]:
        return {
            "confidence": float(self.confidence),
            "margin": float(self.margin),
            "quality": float(self.quality),
            "fit_p_value": float(self.fit_p_value),
        }


def minimum_conformal_calibration_count(alpha: float) -> int | None:
    """Return the smallest reference count that can produce p < ``alpha``."""

    alpha = _unit(alpha, description="conformal alpha")
    if alpha == 0.0:
        return None
    reciprocal = 1.0 / alpha
    if not math.isfinite(reciprocal) or reciprocal > MAXIMUM_CONFORMAL_CALIBRATION_COUNT:
        raise ValueError(
            "conformal alpha is too small for the bounded font-domain dataset contract"
        )
    return max(1, int(math.floor(reciprocal + 1e-12)))


@dataclass(frozen=True)
class FontDomainPublicationSafety:
    """Self-hashed record of input leakage gates used before fitting."""

    leakage_metadata: str = "not_asserted"
    near_duplicate_audit: str = "not_run"
    perceptual_hash_abi: str = PERCEPTUAL_HASH_ABI
    maximum_hamming_distance: int | None = None
    checked_regions: int | None = None
    cross_split_comparisons: int | None = None

    def __post_init__(self) -> None:
        if self.leakage_metadata not in {
            "required_and_present",
            "incomplete_allowed",
            "not_asserted",
        }:
            raise ValueError("invalid leakage_metadata publication status")
        if self.near_duplicate_audit not in {"passed", "skipped", "not_run"}:
            raise ValueError("invalid near_duplicate_audit publication status")
        if self.perceptual_hash_abi != PERCEPTUAL_HASH_ABI:
            raise ValueError("unsupported perceptual_hash_abi")
        values = (
            ("maximum_hamming_distance", self.maximum_hamming_distance),
            ("checked_regions", self.checked_regions),
            ("cross_split_comparisons", self.cross_split_comparisons),
        )
        if self.near_duplicate_audit == "passed":
            for name, value in values:
                if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                    raise ValueError(f"{name} must be a non-negative integer after an audit")
            if self.maximum_hamming_distance is not None and self.maximum_hamming_distance > 64:
                raise ValueError("maximum_hamming_distance cannot exceed 64")
        elif any(value is not None for _, value in values):
            raise ValueError("near-duplicate metrics require a passed audit")

    @property
    def required_checks_recorded(self) -> bool:
        return (
            self.leakage_metadata == "required_and_present"
            and self.near_duplicate_audit == "passed"
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "leakage_metadata": self.leakage_metadata,
            "near_duplicate_audit": self.near_duplicate_audit,
            "perceptual_hash_abi": self.perceptual_hash_abi,
            "maximum_hamming_distance": self.maximum_hamming_distance,
            "checked_regions": self.checked_regions,
            "cross_split_comparisons": self.cross_split_comparisons,
            "required_checks_recorded": self.required_checks_recorded,
        }


@dataclass(frozen=True)
class FontDomainPrototypeModel:
    domains: tuple[str, ...]
    scaler_median: tuple[float, ...]
    scaler_scale: tuple[float, ...]
    pooled_variance: tuple[float, ...]
    prototypes: Mapping[str, tuple[float, ...]]
    role_prototypes: Mapping[str, Mapping[str, tuple[float, ...]]]
    known_roles: tuple[str, ...]
    calibration_distances: Mapping[str, tuple[float, ...]]
    calibration_source: Mapping[str, str]
    temperature: float
    gates: FontDomainGates
    manifest_sha256: str
    dataset_snapshot_sha256: str
    training_counts: Mapping[str, int]
    calibration_counts: Mapping[str, int]
    training_group_counts: Mapping[str, int]
    calibration_group_counts: Mapping[str, int]
    rejected_counts: Mapping[str, int]
    minimum_role_regions_per_domain: int
    minimum_calibration_groups_per_domain: int
    publication_safety: FontDomainPublicationSafety
    dependency_versions: Mapping[str, str]
    model_sha256: str | None = None

    def __post_init__(self) -> None:
        if len(self.domains) < 2 or tuple(sorted(set(self.domains))) != self.domains:
            raise ValueError("model domains must contain at least two unique sorted domains")
        for domain in self.domains:
            _nonempty(domain, description="model domain")
            if domain == UNKNOWN_DOMAIN or not _DOMAIN_PATTERN.fullmatch(domain):
                raise ValueError(f"invalid trained model domain {domain!r}")
        _vector(self.scaler_median, description="scaler median")
        _vector(self.scaler_scale, description="scaler scale", minimum=1e-12)
        _vector(self.pooled_variance, description="pooled variance", minimum=1e-12)
        if set(self.prototypes) != set(self.domains):
            raise ValueError("generic prototypes must exactly match model domains")
        for domain, vector in self.prototypes.items():
            _vector(vector, description=f"prototype {domain}")
        for role, prototypes in self.role_prototypes.items():
            _nonempty(role, description="role prototype name")
            if set(prototypes) != set(self.domains):
                raise ValueError(f"role {role!r} prototypes must exactly match model domains")
            for domain, vector in prototypes.items():
                _vector(vector, description=f"role prototype {role}/{domain}")
        if tuple(sorted(set(self.known_roles))) != self.known_roles:
            raise ValueError("known_roles must be unique and sorted")
        if set(self.calibration_distances) != set(self.domains):
            raise ValueError("calibration distances must exactly match model domains")
        if set(self.calibration_source) != set(self.domains):
            raise ValueError("calibration sources must exactly match model domains")
        for domain, distances in self.calibration_distances.items():
            if not distances:
                raise ValueError(f"calibration distances for {domain!r} must not be empty")
            for index, distance in enumerate(distances):
                _finite(distance, description=f"calibration distance {domain}[{index}]", minimum=0.0)
            if tuple(sorted(distances)) != tuple(distances):
                raise ValueError(f"calibration distances for {domain!r} must be sorted")
            if self.calibration_source[domain] not in {"calibration", "train_fallback"}:
                raise ValueError(f"invalid calibration source for {domain!r}")
        _finite(self.temperature, description="temperature", minimum=1e-6)
        for name, value in (
            ("minimum_role_regions_per_domain", self.minimum_role_regions_per_domain),
            ("minimum_calibration_groups_per_domain", self.minimum_calibration_groups_per_domain),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        conformal_minimum = minimum_conformal_calibration_count(self.gates.fit_p_value)
        if (
            conformal_minimum is not None
            and self.minimum_calibration_groups_per_domain < conformal_minimum
        ):
            raise ValueError(
                "minimum_calibration_groups_per_domain cannot make the configured conformal "
                "fit-p gate effective"
            )
        if len(self.manifest_sha256) != 64 or any(ch not in "0123456789abcdef" for ch in self.manifest_sha256):
            raise ValueError("manifest_sha256 must be a lowercase SHA-256")
        if len(self.dataset_snapshot_sha256) != 64 or any(
            ch not in "0123456789abcdef" for ch in self.dataset_snapshot_sha256
        ):
            raise ValueError("dataset_snapshot_sha256 must be a lowercase SHA-256")
        if not isinstance(self.publication_safety, FontDomainPublicationSafety):
            raise ValueError("publication_safety must use FontDomainPublicationSafety")
        for name, counts in (
            ("training_counts", self.training_counts),
            ("calibration_counts", self.calibration_counts),
            ("training_group_counts", self.training_group_counts),
            ("calibration_group_counts", self.calibration_group_counts),
            ("rejected_counts", self.rejected_counts),
        ):
            for key, value in counts.items():
                _nonempty(key, description=f"{name} key")
                if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                    raise ValueError(f"{name}[{key!r}] must be a non-negative integer")
        if set(self.training_counts) != set(self.domains):
            raise ValueError("training_counts must exactly match model domains")
        if set(self.calibration_counts) != set(self.domains):
            raise ValueError("calibration_counts must exactly match model domains")
        if set(self.training_group_counts) != set(self.domains):
            raise ValueError("training_group_counts must exactly match model domains")
        if set(self.calibration_group_counts) != set(self.domains):
            raise ValueError("calibration_group_counts must exactly match model domains")
        if any(self.training_counts[domain] < 1 for domain in self.domains):
            raise ValueError("every model domain must have at least one training region")
        if any(self.training_group_counts[domain] < 1 for domain in self.domains):
            raise ValueError("every model domain must have at least one training source group")
        for domain in self.domains:
            if self.training_group_counts[domain] > self.training_counts[domain]:
                raise ValueError("training group counts cannot exceed training region counts")
            if self.calibration_group_counts[domain] > self.calibration_counts[domain]:
                raise ValueError("calibration group counts cannot exceed calibration region counts")
            if self.calibration_source[domain] == "calibration":
                if self.calibration_group_counts[domain] < 1:
                    raise ValueError("calibrated domains require at least one calibration source group")
                if len(self.calibration_distances[domain]) != self.calibration_counts[domain]:
                    raise ValueError("calibration distance count must equal calibration region count")
            else:
                if self.calibration_counts[domain] != 0 or self.calibration_group_counts[domain] != 0:
                    raise ValueError("train fallback domains cannot claim calibration samples")
                if len(self.calibration_distances[domain]) != self.training_counts[domain]:
                    raise ValueError("train fallback distance count must equal training region count")
        if not self.dependency_versions:
            raise ValueError("dependency_versions must not be empty")
        for name, version in self.dependency_versions.items():
            _nonempty(name, description="dependency name")
            _nonempty(version, description=f"dependency version {name}")
        if self.model_sha256 is not None and (
            len(self.model_sha256) != 64
            or any(ch not in "0123456789abcdef" for ch in self.model_sha256)
        ):
            raise ValueError("model_sha256 must be a lowercase SHA-256")

    def core_dict(self) -> dict[str, object]:
        return {
            "schema_version": SCHEMA_VERSION,
            "kind": MODEL_KIND,
            "feature_abi": FEATURE_ABI,
            "feature_dimension": FEATURE_DIMENSION,
            "feature_blocks": [
                {"name": name, "start": start, "end": end, "weight": weight}
                for name, start, end, weight in FEATURE_BLOCKS
            ],
            "domains": list(self.domains),
            "scaler": {
                "median": list(self.scaler_median),
                "scale": list(self.scaler_scale),
            },
            "pooled_variance": list(self.pooled_variance),
            "prototypes": {
                domain: list(self.prototypes[domain]) for domain in self.domains
            },
            "role_prototypes": {
                role: {
                    domain: list(self.role_prototypes[role][domain]) for domain in self.domains
                }
                for role in sorted(self.role_prototypes)
            },
            "known_roles": list(self.known_roles),
            "calibration_distances": {
                domain: list(self.calibration_distances[domain]) for domain in self.domains
            },
            "calibration_source": {
                domain: self.calibration_source[domain] for domain in self.domains
            },
            "temperature": float(self.temperature),
            "gates": self.gates.as_dict(),
            "source": {
                "manifest_sha256": self.manifest_sha256,
                "dataset_snapshot_sha256": self.dataset_snapshot_sha256,
                "training_counts": dict(sorted(self.training_counts.items())),
                "calibration_counts": dict(sorted(self.calibration_counts.items())),
                "training_group_counts": dict(sorted(self.training_group_counts.items())),
                "calibration_group_counts": dict(sorted(self.calibration_group_counts.items())),
                "rejected_counts": dict(sorted(self.rejected_counts.items())),
            },
            "minimum_role_regions_per_domain": self.minimum_role_regions_per_domain,
            "minimum_calibration_groups_per_domain": self.minimum_calibration_groups_per_domain,
            "publication_safety": self.publication_safety.as_dict(),
            "dependency_versions": dict(sorted(self.dependency_versions.items())),
            "authenticity": "not_assessed",
        }

    def as_dict(self) -> dict[str, object]:
        core = self.core_dict()
        digest = _canonical_sha256(core)
        if self.model_sha256 is not None and self.model_sha256 != digest:
            raise ValueError("in-memory model_sha256 does not match decision fields")
        return {**core, "model_sha256": digest}


def _robust_scale(samples: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    median = np.median(samples, axis=0)
    mad = 1.4826 * np.median(np.abs(samples - median), axis=0)
    q25, q75 = np.quantile(samples, (0.25, 0.75), axis=0)
    iqr_scale = (q75 - q25) / 1.349
    standard = np.std(samples, axis=0)
    scale = np.where(mad >= 1e-3, mad, np.where(iqr_scale >= 1e-3, iqr_scale, standard))
    scale = np.maximum(scale, 1e-3)
    return median.astype(np.float64), scale.astype(np.float64)


def _standardize(values: np.ndarray, median: np.ndarray, scale: np.ndarray) -> np.ndarray:
    return np.clip((values - median) / scale, -8.0, 8.0)


def _distance(
    vector: np.ndarray,
    prototype: Sequence[float],
    variance: np.ndarray,
) -> float:
    delta2 = np.square(vector - np.asarray(prototype, dtype=np.float64)) / variance
    total = 0.0
    for _, start, end, weight in FEATURE_BLOCKS:
        total += weight * float(np.mean(delta2[start:end]))
    return float(math.sqrt(max(total, 0.0)))


def _prototype_set(
    *,
    role: str,
    generic: Mapping[str, tuple[float, ...]],
    role_specific: Mapping[str, Mapping[str, tuple[float, ...]]],
) -> tuple[Mapping[str, tuple[float, ...]], bool]:
    selected = role_specific.get(role)
    if selected is not None:
        return selected, False
    # Generic prototypes are the primary model until at least one fully
    # balanced role model has been fitted.  Once specialization exists, an
    # unseen/under-supported role is explicitly marked as a weaker fallback.
    return generic, bool(role_specific)


def _distances_for(
    vector: np.ndarray,
    *,
    role: str,
    domains: Sequence[str],
    prototypes: Mapping[str, tuple[float, ...]],
    role_prototypes: Mapping[str, Mapping[str, tuple[float, ...]]],
    variance: np.ndarray,
) -> tuple[dict[str, float], bool]:
    selected, fallback = _prototype_set(
        role=role, generic=prototypes, role_specific=role_prototypes
    )
    return (
        {
            domain: _distance(vector, selected[domain], variance)
            for domain in domains
        },
        fallback,
    )


def _softmax_supports(distances: Mapping[str, float], temperature: float) -> dict[str, float]:
    domains = sorted(distances)
    logits = np.asarray([-distances[domain] / temperature for domain in domains], dtype=np.float64)
    logits -= float(np.max(logits))
    values = np.exp(logits)
    values /= max(float(values.sum()), 1e-300)
    return {domain: float(value) for domain, value in zip(domains, values, strict=True)}


def _fit_temperature(rows: Sequence[tuple[Mapping[str, float], str]]) -> float:
    if not rows:
        return 1.0
    candidates = np.exp(np.linspace(math.log(0.10), math.log(5.0), 81))
    best_temperature = 1.0
    best_loss = float("inf")
    for candidate in candidates:
        loss = 0.0
        for distances, label in rows:
            support = _softmax_supports(distances, float(candidate))[label]
            loss -= math.log(max(support, 1e-12))
        loss /= len(rows)
        # Prefer the less confident (larger) temperature for numerical ties.
        if loss < best_loss - 1e-12 or (
            math.isclose(loss, best_loss, abs_tol=1e-12) and candidate > best_temperature
        ):
            best_loss = loss
            best_temperature = float(candidate)
    return best_temperature


def fit_font_domain_model(
    dataset: FontDomainDataset,
    *,
    gates: FontDomainGates | None = None,
    minimum_train_regions_per_domain: int = 3,
    minimum_role_regions_per_domain: int = 3,
    minimum_calibration_groups_per_domain: int = 20,
    publication_safety: FontDomainPublicationSafety | None = None,
) -> FontDomainPrototypeModel:
    """Fit a robust prototype model from train and optional calibration rows.

    Only information-gate-passing crops participate.  Calibration rows never
    affect the scaler or prototypes.  Where a domain has no usable calibration
    row, its sorted train distances are retained as an explicit
    ``train_fallback`` rather than silently pretending they are held-out.
    """

    if (
        isinstance(minimum_train_regions_per_domain, bool)
        or isinstance(minimum_role_regions_per_domain, bool)
        or isinstance(minimum_calibration_groups_per_domain, bool)
        or minimum_train_regions_per_domain < 1
        or minimum_role_regions_per_domain < 1
        or minimum_calibration_groups_per_domain < 1
    ):
        raise ValueError("minimum training counts must be positive")
    gates = gates or FontDomainGates()
    conformal_minimum = minimum_conformal_calibration_count(gates.fit_p_value)
    if conformal_minimum is not None:
        minimum_calibration_groups_per_domain = max(
            minimum_calibration_groups_per_domain,
            conformal_minimum,
        )
    publication_safety = publication_safety or FontDomainPublicationSafety()
    if not isinstance(publication_safety, FontDomainPublicationSafety):
        raise ValueError("publication_safety must use FontDomainPublicationSafety")
    rows: list[tuple[str, str, str, np.ndarray, float]] = []
    rejected: Counter[str] = Counter()
    observed_labeled_splits: set[str] = set()
    declared_train_domains: set[str] = set()
    declared_calibration_domains: set[str] = set()
    known_roles: set[str] = set()
    usable_groups: dict[tuple[str, str], set[str]] = defaultdict(set)
    for document in dataset.documents:
        if document.split in {"test", "inference"}:
            continue
        if document.font_domain is None:
            raise ValueError("train/calibration documents must have a font_domain")
        observed_labeled_splits.add(document.split)
        if document.split == "train":
            declared_train_domains.add(document.font_domain)
        else:
            declared_calibration_domains.add(document.font_domain)
        for region in document.regions:
            if not region.include_in_consistency:
                continue
            known_roles.add(region.role)
            rgb = _load_bound_region_rgb(region)
            feature = extract_font_domain_features(rgb)
            if not feature.usable or feature.quality < gates.quality:
                rejected[f"{document.split}:{document.font_domain}"] += 1
                continue
            rows.append(
                (
                    document.split,
                    document.font_domain,
                    region.role,
                    np.asarray(feature.values, dtype=np.float64),
                    feature.quality,
                )
            )
            usable_groups[(document.split, document.font_domain)].add(document.source_group_id)
    if "train" not in observed_labeled_splits:
        raise ValueError("font-domain fitting requires train documents")
    train_rows = [row for row in rows if row[0] == "train"]
    if not train_rows:
        raise ValueError("no train regions passed the information gate")
    domains = tuple(sorted({row[1] for row in train_rows}))
    if len(domains) < 2:
        raise ValueError("font-domain fitting requires at least two train domains")
    rejected_domains = sorted(declared_train_domains - set(domains))
    if rejected_domains:
        raise ValueError(
            "all train regions failed the information gate for domains: "
            f"{rejected_domains}"
        )
    train_counts = Counter(row[1] for row in train_rows)
    too_small = {
        domain: train_counts[domain]
        for domain in domains
        if train_counts[domain] < minimum_train_regions_per_domain
    }
    if too_small:
        raise ValueError(
            "insufficient information-gate-passing train regions by domain: "
            + ", ".join(f"{key}={value}" for key, value in sorted(too_small.items()))
        )
    foreign_calibration = sorted(declared_calibration_domains - set(domains))
    if foreign_calibration:
        raise ValueError(f"calibration contains unseen domains: {foreign_calibration}")

    train_matrix = np.stack([row[3] for row in train_rows], axis=0)
    median, scale = _robust_scale(train_matrix)
    train_z = _standardize(train_matrix, median, scale)
    prototypes = {
        domain: tuple(float(item) for item in np.median(
            train_z[[row[1] == domain for row in train_rows]], axis=0
        ))
        for domain in domains
    }
    residuals = np.stack(
        [
            train_z[index] - np.asarray(prototypes[row[1]], dtype=np.float64)
            for index, row in enumerate(train_rows)
        ],
        axis=0,
    )
    residual_mad = 1.4826 * np.median(np.abs(residuals), axis=0)
    residual_std = np.std(residuals, axis=0)
    robust_sigma = np.where(residual_mad >= 0.05, residual_mad, residual_std)
    # Shrink toward standardized unit variance.  The floor avoids a constant
    # training coordinate becoming an infinite-distance tripwire.
    pooled_variance = np.clip(0.5 * np.square(robust_sigma) + 0.5, 0.10, 16.0)

    role_prototypes: dict[str, dict[str, tuple[float, ...]]] = {}
    for role in sorted(known_roles):
        indexes_by_domain = {
            domain: [
                index
                for index, row in enumerate(train_rows)
                if row[1] == domain and row[2] == role
            ]
            for domain in domains
        }
        if all(len(indexes_by_domain[domain]) >= minimum_role_regions_per_domain for domain in domains):
            role_prototypes[role] = {
                domain: tuple(
                    float(item) for item in np.median(train_z[indexes_by_domain[domain]], axis=0)
                )
                for domain in domains
            }

    calibrated_rows: list[tuple[Mapping[str, float], str]] = []
    true_distances: dict[str, list[float]] = defaultdict(list)
    calibration_counts: Counter[str] = Counter()
    for row in rows:
        if row[0] != "calibration":
            continue
        z = _standardize(row[3], median, scale)
        distances, _ = _distances_for(
            z,
            role=row[2],
            domains=domains,
            prototypes=prototypes,
            role_prototypes=role_prototypes,
            variance=pooled_variance,
        )
        calibrated_rows.append((distances, row[1]))
        true_distances[row[1]].append(distances[row[1]])
        calibration_counts[row[1]] += 1

    calibration_source: dict[str, str] = {}
    for domain in domains:
        if true_distances[domain]:
            calibration_source[domain] = "calibration"
            continue
        calibration_source[domain] = "train_fallback"
        for index, row in enumerate(train_rows):
            if row[1] != domain:
                continue
            distances, _ = _distances_for(
                train_z[index],
                role=row[2],
                domains=domains,
                prototypes=prototypes,
                role_prototypes=role_prototypes,
                variance=pooled_variance,
            )
            true_distances[domain].append(distances[domain])

    if not calibrated_rows:
        # This fallback is declared in the artifact and keeps the first MVP
        # runnable.  Publication thresholds should be selected on a separate
        # calibration split before operational use.
        for index, row in enumerate(train_rows):
            distances, _ = _distances_for(
                train_z[index],
                role=row[2],
                domains=domains,
                prototypes=prototypes,
                role_prototypes=role_prototypes,
                variance=pooled_variance,
            )
            calibrated_rows.append((distances, row[1]))
    temperature = _fit_temperature(calibrated_rows)

    model = FontDomainPrototypeModel(
        domains=domains,
        scaler_median=tuple(float(item) for item in median),
        scaler_scale=tuple(float(item) for item in scale),
        pooled_variance=tuple(float(item) for item in pooled_variance),
        prototypes=prototypes,
        role_prototypes=role_prototypes,
        known_roles=tuple(sorted(known_roles)),
        calibration_distances={
            domain: tuple(sorted(float(item) for item in true_distances[domain]))
            for domain in domains
        },
        calibration_source=calibration_source,
        temperature=temperature,
        gates=gates,
        manifest_sha256=dataset.manifest_sha256,
        dataset_snapshot_sha256=dataset.snapshot_sha256,
        training_counts=dict(sorted(train_counts.items())),
        calibration_counts={domain: calibration_counts[domain] for domain in domains},
        training_group_counts={
            domain: len(usable_groups[("train", domain)]) for domain in domains
        },
        calibration_group_counts={
            domain: len(usable_groups[("calibration", domain)]) for domain in domains
        },
        rejected_counts=dict(sorted(rejected.items())),
        minimum_role_regions_per_domain=minimum_role_regions_per_domain,
        minimum_calibration_groups_per_domain=minimum_calibration_groups_per_domain,
        publication_safety=publication_safety,
        dependency_versions={
            "numpy": np.__version__,
            "opencv": cv2.__version__,
            "pillow": PILLOW_VERSION,
        },
    )
    digest = _canonical_sha256(model.core_dict())
    return replace(model, model_sha256=digest)


def _atomic_write_bytes_no_clobber(path: Path, data: bytes) -> None:
    if os.path.lexists(path):
        raise FileExistsError(f"refusing to overwrite model artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    if os.path.lexists(path):
        raise FileExistsError(f"refusing to overwrite model artifact: {path}")
    temporary = path.with_name(f".{path.name}.{secrets.token_hex(16)}.tmp")
    descriptor: int | None = None
    try:
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "wb", closefd=True) as stream:
            descriptor = None
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
        try:
            directory_descriptor = os.open(
                path.parent,
                os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
            )
            try:
                os.fsync(directory_descriptor)
            finally:
                os.close(directory_descriptor)
        except OSError:
            pass
    finally:
        if descriptor is not None:
            os.close(descriptor)
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def save_font_domain_model(model: FontDomainPrototypeModel, path: Path) -> dict[str, object]:
    """Publish a validated model JSON without overwriting existing evidence."""

    payload = model.as_dict()
    encoded = (
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n"
    ).encode("utf-8")
    if len(encoded) > MAXIMUM_MODEL_BYTES:
        raise ValueError(
            f"font-domain model exceeds the {MAXIMUM_MODEL_BYTES}-byte publication limit"
        )
    destination = Path(os.path.abspath(os.fspath(path.expanduser())))
    _atomic_write_bytes_no_clobber(destination, encoded)
    return {
        "path": destination.as_posix(),
        "sha256": hashlib.sha256(encoded).hexdigest(),
        "model_sha256": payload["model_sha256"],
        "size_bytes": len(encoded),
    }


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON number {value!r}")


def _require_mapping(value: object, *, description: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{description} must be an object")
    return value


def _require_exact_keys(value: Mapping[str, Any], keys: set[str], *, description: str) -> None:
    if set(value) != keys:
        missing = sorted(keys - set(value))
        extra = sorted(set(value) - keys)
        raise ValueError(f"{description} keys differ; missing={missing}, extra={extra}")


def load_font_domain_model(path: Path) -> FontDomainPrototypeModel:
    """Load and fully validate a self-hashed font-domain model artifact."""

    source = path.expanduser().resolve(strict=True)
    if not source.is_file():
        raise FileNotFoundError(source)
    if source.stat().st_size > MAXIMUM_MODEL_BYTES:
        raise ValueError(
            f"font-domain model exceeds the {MAXIMUM_MODEL_BYTES}-byte limit: {source}"
        )
    with source.open("rb") as stream:
        encoded = stream.read(MAXIMUM_MODEL_BYTES + 1)
    if len(encoded) > MAXIMUM_MODEL_BYTES:
        raise ValueError(
            f"font-domain model exceeds the {MAXIMUM_MODEL_BYTES}-byte limit: {source}"
        )
    try:
        text = encoded.decode("utf-8-sig")
        raw = json.loads(
            text,
            object_pairs_hook=_strict_object,
            parse_constant=_reject_constant,
        )
    except (json.JSONDecodeError, UnicodeError, ValueError) as error:
        raise ValueError(f"invalid font-domain model JSON {source}: {error}") from None
    root = _require_mapping(raw, description="model")
    root_keys = {
        "schema_version", "kind", "feature_abi", "feature_dimension", "feature_blocks",
        "domains", "scaler", "pooled_variance", "prototypes", "role_prototypes",
        "known_roles", "calibration_distances", "calibration_source", "temperature",
        "gates", "source", "minimum_role_regions_per_domain", "dependency_versions",
        "minimum_calibration_groups_per_domain", "publication_safety", "authenticity",
        "model_sha256",
    }
    _require_exact_keys(root, root_keys, description="model")
    if (
        root.get("schema_version") != SCHEMA_VERSION
        or root.get("kind") != MODEL_KIND
        or root.get("feature_abi") != FEATURE_ABI
        or root.get("feature_dimension") != FEATURE_DIMENSION
        or root.get("authenticity") != "not_assessed"
    ):
        raise ValueError("unsupported font-domain model contract")
    expected_blocks = [
        {"name": name, "start": start, "end": end, "weight": weight}
        for name, start, end, weight in FEATURE_BLOCKS
    ]
    if root.get("feature_blocks") != expected_blocks:
        raise ValueError("model feature_blocks differ from the feature ABI")
    supplied_digest = root.get("model_sha256")
    supplied_digest = _nonempty(supplied_digest, description="model_sha256")
    core = dict(root)
    del core["model_sha256"]
    observed_digest = _canonical_sha256(core)
    if supplied_digest != observed_digest:
        raise ValueError("model_sha256 does not match decision fields")

    domains_raw = root["domains"]
    if not isinstance(domains_raw, list):
        raise ValueError("domains must be an array")
    domains = tuple(_nonempty(item, description="domain") for item in domains_raw)
    scaler = _require_mapping(root["scaler"], description="scaler")
    _require_exact_keys(scaler, {"median", "scale"}, description="scaler")
    prototypes_raw = _require_mapping(root["prototypes"], description="prototypes")
    role_raw = _require_mapping(root["role_prototypes"], description="role_prototypes")
    role_prototypes: dict[str, dict[str, tuple[float, ...]]] = {}
    for role, raw_prototypes in role_raw.items():
        role_map = _require_mapping(raw_prototypes, description=f"role prototype {role}")
        role_prototypes[str(role)] = {
            str(domain): _vector(vector, description=f"role prototype {role}/{domain}")
            for domain, vector in role_map.items()
        }
    calibration_raw = _require_mapping(
        root["calibration_distances"], description="calibration_distances"
    )
    calibration_source_raw = _require_mapping(
        root["calibration_source"], description="calibration_source"
    )
    gates_raw = _require_mapping(root["gates"], description="gates")
    _require_exact_keys(gates_raw, {"confidence", "margin", "quality", "fit_p_value"}, description="gates")
    source_raw = _require_mapping(root["source"], description="source")
    _require_exact_keys(
        source_raw,
        {
            "manifest_sha256",
            "dataset_snapshot_sha256",
            "training_counts",
            "calibration_counts",
            "training_group_counts",
            "calibration_group_counts",
            "rejected_counts",
        },
        description="source",
    )
    dependency_raw = _require_mapping(root["dependency_versions"], description="dependency_versions")
    publication_raw = _require_mapping(root["publication_safety"], description="publication_safety")
    _require_exact_keys(
        publication_raw,
        {
            "leakage_metadata",
            "near_duplicate_audit",
            "perceptual_hash_abi",
            "maximum_hamming_distance",
            "checked_regions",
            "cross_split_comparisons",
            "required_checks_recorded",
        },
        description="publication_safety",
    )
    known_roles_raw = root["known_roles"]
    if not isinstance(known_roles_raw, list):
        raise ValueError("known_roles must be an array")
    training_counts_raw = _require_mapping(
        source_raw["training_counts"], description="training_counts"
    )
    calibration_counts_raw = _require_mapping(
        source_raw["calibration_counts"], description="calibration_counts"
    )
    training_group_counts_raw = _require_mapping(
        source_raw["training_group_counts"], description="training_group_counts"
    )
    calibration_group_counts_raw = _require_mapping(
        source_raw["calibration_group_counts"], description="calibration_group_counts"
    )
    rejected_counts_raw = _require_mapping(
        source_raw["rejected_counts"], description="rejected_counts"
    )
    minimum_role_count = root["minimum_role_regions_per_domain"]
    if isinstance(minimum_role_count, bool) or not isinstance(minimum_role_count, int):
        raise ValueError("minimum_role_regions_per_domain must be an integer")
    minimum_calibration_group_count = root["minimum_calibration_groups_per_domain"]
    if (
        isinstance(minimum_calibration_group_count, bool)
        or not isinstance(minimum_calibration_group_count, int)
    ):
        raise ValueError("minimum_calibration_groups_per_domain must be an integer")
    parsed_calibration: dict[str, tuple[float, ...]] = {}
    for domain, distances in calibration_raw.items():
        if not isinstance(distances, list):
            raise ValueError(f"calibration distances for {domain!r} must be an array")
        parsed_calibration[str(domain)] = tuple(
            _finite(item, description=f"calibration distance {domain}", minimum=0.0)
            for item in distances
        )
    model = FontDomainPrototypeModel(
        domains=domains,
        scaler_median=_vector(scaler["median"], description="scaler median"),
        scaler_scale=_vector(scaler["scale"], description="scaler scale", minimum=1e-12),
        pooled_variance=_vector(root["pooled_variance"], description="pooled variance", minimum=1e-12),
        prototypes={
            str(domain): _vector(vector, description=f"prototype {domain}")
            for domain, vector in prototypes_raw.items()
        },
        role_prototypes=role_prototypes,
        known_roles=tuple(_nonempty(item, description="known role") for item in known_roles_raw),
        calibration_distances=parsed_calibration,
        calibration_source={
            str(domain): _nonempty(value, description=f"calibration source {domain}")
            for domain, value in calibration_source_raw.items()
        },
        temperature=_finite(root["temperature"], description="temperature", minimum=1e-6),
        gates=FontDomainGates(
            confidence=_unit(gates_raw["confidence"], description="confidence gate"),
            margin=_unit(gates_raw["margin"], description="margin gate"),
            quality=_unit(gates_raw["quality"], description="quality gate"),
            fit_p_value=_unit(gates_raw["fit_p_value"], description="fit p-value gate"),
        ),
        manifest_sha256=_nonempty(source_raw["manifest_sha256"], description="manifest_sha256"),
        dataset_snapshot_sha256=_nonempty(
            source_raw["dataset_snapshot_sha256"], description="dataset_snapshot_sha256"
        ),
        training_counts={str(key): value for key, value in training_counts_raw.items()},
        calibration_counts={str(key): value for key, value in calibration_counts_raw.items()},
        training_group_counts={
            str(key): value for key, value in training_group_counts_raw.items()
        },
        calibration_group_counts={
            str(key): value for key, value in calibration_group_counts_raw.items()
        },
        rejected_counts={str(key): value for key, value in rejected_counts_raw.items()},
        minimum_role_regions_per_domain=minimum_role_count,
        minimum_calibration_groups_per_domain=minimum_calibration_group_count,
        publication_safety=FontDomainPublicationSafety(
            leakage_metadata=_nonempty(
                publication_raw["leakage_metadata"], description="leakage_metadata"
            ),
            near_duplicate_audit=_nonempty(
                publication_raw["near_duplicate_audit"], description="near_duplicate_audit"
            ),
            perceptual_hash_abi=_nonempty(
                publication_raw["perceptual_hash_abi"], description="perceptual_hash_abi"
            ),
            maximum_hamming_distance=publication_raw["maximum_hamming_distance"],
            checked_regions=publication_raw["checked_regions"],
            cross_split_comparisons=publication_raw["cross_split_comparisons"],
        ),
        dependency_versions={
            str(key): _nonempty(value, description=f"dependency version {key}")
            for key, value in dependency_raw.items()
        },
        model_sha256=supplied_digest,
    )
    # Re-serialization catches semantic normalization surprises (for example a
    # numeric field encoded as 1 where the canonical model writes 1.0).
    if model.as_dict() != root:
        raise ValueError("model JSON is not in canonical semantic form")
    return model


def _predict_feature(
    model: FontDomainPrototypeModel,
    *,
    region_id: str,
    role: str,
    include_in_consistency: bool,
    feature: RegionFeatureResult,
    allow_uncalibrated: bool = False,
) -> FontDomainLinePrediction:
    if not isinstance(allow_uncalibrated, bool):
        raise ValueError("allow_uncalibrated must be boolean")
    vector = _standardize(
        np.asarray(feature.values, dtype=np.float64),
        np.asarray(model.scaler_median, dtype=np.float64),
        np.asarray(model.scaler_scale, dtype=np.float64),
    )
    distances, generic_fallback = _distances_for(
        vector,
        role=role,
        domains=model.domains,
        prototypes=model.prototypes,
        role_prototypes=model.role_prototypes,
        variance=np.asarray(model.pooled_variance, dtype=np.float64),
    )
    supports = _softmax_supports(distances, model.temperature)
    candidate = sorted(supports, key=lambda domain: (-supports[domain], domain))[0]
    reference = model.calibration_distances[candidate]
    fit_p_value = (1.0 + sum(distance >= distances[candidate] for distance in reference)) / (
        len(reference) + 1.0
    )
    gated = prediction_from_probabilities(
        region_id=region_id,
        role=role,
        probabilities=supports,
        quality=feature.quality,
        include_in_consistency=include_in_consistency,
        confidence_threshold=model.gates.confidence,
        margin_threshold=model.gates.margin,
        quality_threshold=model.gates.quality,
        fit_p_value=fit_p_value,
        fit_p_threshold=model.gates.fit_p_value,
        generic_fallback=generic_fallback,
    )
    reasons = list(gated.reasons)
    if feature.reasons != ("INFORMATION_GATE_PASSED",):
        reasons.extend(reason for reason in feature.reasons if reason not in reasons)
    if not feature.usable:
        reasons = [reason for reason in reasons if reason != "KNOWN_DOMAIN_EVIDENCE"]
        if "LOW_INFORMATION" not in reasons:
            reasons.insert(0, "LOW_INFORMATION")
    train_fallback = model.calibration_source[candidate] == "train_fallback"
    insufficient_calibration = (
        not train_fallback
        and model.calibration_group_counts[candidate]
        < model.minimum_calibration_groups_per_domain
    )
    uncalibrated = train_fallback or insufficient_calibration
    if uncalibrated:
        if not allow_uncalibrated:
            reasons = [reason for reason in reasons if reason != "KNOWN_DOMAIN_EVIDENCE"]
        if train_fallback:
            reasons.append(
                "UNCALIBRATED_TRAIN_FALLBACK" if allow_uncalibrated else "UNCALIBRATED_MODEL"
            )
        else:
            reasons.append(
                "INSUFFICIENT_CALIBRATION_SUPPORT_POC"
                if allow_uncalibrated
                else "INSUFFICIENT_CALIBRATION_SUPPORT"
            )
    return replace(
        gated,
        label=(
            UNKNOWN_DOMAIN
            if not feature.usable or (uncalibrated and not allow_uncalibrated)
            else gated.label
        ),
        distances=distances,
        reasons=tuple(dict.fromkeys(reasons)),
    )


def predict_region(
    model: FontDomainPrototypeModel,
    region: FontDomainRegion,
    *,
    allow_uncalibrated: bool = False,
) -> FontDomainLinePrediction:
    """Classify one validated region and re-check its decoded-pixel binding.

    By default a candidate calibrated only on its own training distances is
    returned as ``unknown``.  ``allow_uncalibrated=True`` is an explicit PoC
    escape hatch and remains visible in the prediction reasons.
    """

    if not isinstance(allow_uncalibrated, bool):
        raise ValueError("allow_uncalibrated must be boolean")
    rgb = _load_bound_region_rgb(region)
    return _predict_feature(
        model,
        region_id=region.region_id,
        role=region.role,
        include_in_consistency=region.include_in_consistency,
        feature=extract_font_domain_features(rgb),
        allow_uncalibrated=allow_uncalibrated,
    )


def predict_document(
    model: FontDomainPrototypeModel,
    document: FontDomainDocument,
    *,
    allow_uncalibrated: bool = False,
    **aggregation_options: object,
) -> FontDomainConsistencyResult:
    """Predict every crop and apply the model-agnostic document gate."""

    if not isinstance(allow_uncalibrated, bool):
        raise ValueError("allow_uncalibrated must be boolean")
    predictions = tuple(
        predict_region(model, region, allow_uncalibrated=allow_uncalibrated)
        for region in document.regions
    )
    return aggregate_font_domain_predictions(
        document_id=document.document_id,
        predictions=predictions,
        device_prior_domain=document.device_prior_domain,
        **aggregation_options,
    )


__all__ = [
    "FEATURE_ABI",
    "FEATURE_BLOCKS",
    "FEATURE_DIMENSION",
    "MODEL_KIND",
    "FontDomainGates",
    "FontDomainPublicationSafety",
    "FontDomainPrototypeModel",
    "RegionFeatureResult",
    "extract_font_domain_features",
    "fit_font_domain_model",
    "load_font_domain_model",
    "minimum_conformal_calibration_count",
    "predict_document",
    "predict_region",
    "save_font_domain_model",
]
