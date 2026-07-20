"""Image orientation, screen detection and perspective rectification.

The detector is trained on the rectified image.  This module stores the full
homography so predicted boxes can be drawn back on the input photo without
losing the effect of a perspective correction.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence

import cv2
import numpy as np
from PIL import Image, ImageOps

PointArray = np.ndarray
OrientationScorer = Callable[[np.ndarray], float]


@dataclass
class RectificationOptions:
    """Controls used by :func:`rectify_receipt`.

    ``screen_quad`` uses the coordinates of the EXIF-normalised source image,
    ordered arbitrarily as four ``[x, y]`` points.  It is the reliable manual
    fallback for photos where the screen boundary cannot be detected.
    """

    orientation_degrees: int | None = None
    prefer_portrait: bool = True
    auto_screen: bool = True
    screen_quad: Sequence[Sequence[float]] | None = None
    max_side: int = 1600
    orientation_scorer: OrientationScorer | None = None


@dataclass
class RectificationResult:
    """Rectified pixels plus transformations relative to the source pixels."""

    source_rgb: np.ndarray
    rectified_rgb: np.ndarray
    original_to_rectified: np.ndarray
    rectified_to_original: np.ndarray
    screen_quad_original: np.ndarray
    rotation_degrees: int
    screen_detected: bool

    def manifest(self) -> dict[str, object]:
        """Return JSON-serialisable geometry needed to reproduce this result."""
        source_height, source_width = self.source_rgb.shape[:2]
        rectified_height, rectified_width = self.rectified_rgb.shape[:2]
        return {
            "source_size": {"width": source_width, "height": source_height},
            "rectified_size": {"width": rectified_width, "height": rectified_height},
            "rotation_degrees": self.rotation_degrees,
            "screen_detected": self.screen_detected,
            "screen_quad_original": np.round(self.screen_quad_original, 3).tolist(),
            "H_original_to_rectified": np.round(self.original_to_rectified, 8).tolist(),
            "H_rectified_to_original": np.round(self.rectified_to_original, 8).tolist(),
        }


def load_upright_rgb(path: str | Path) -> np.ndarray:
    """Load an image in RGB and apply its EXIF orientation exactly once."""
    with Image.open(path) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        return np.asarray(image).copy()


def save_rgb(path: str | Path, image_rgb: np.ndarray, quality: int = 95) -> None:
    """Save a RGB array with a deterministic JPEG/PNG-safe conversion."""
    Image.fromarray(image_rgb).save(path, quality=quality)


def _validate_rotation(degrees: int) -> int:
    degrees = int(degrees) % 360
    if degrees not in {0, 90, 180, 270}:
        raise ValueError("orientation_degrees must be one of 0, 90, 180, 270")
    return degrees


def rotation_homography(width: int, height: int, degrees: int) -> np.ndarray:
    """Map source pixel coordinates to OpenCV's right-angle rotated image."""
    degrees = _validate_rotation(degrees)
    if degrees == 0:
        return np.eye(3, dtype=np.float64)
    if degrees == 90:  # clockwise: (x, y) -> (h - 1 - y, x)
        return np.array([[0, -1, height - 1], [1, 0, 0], [0, 0, 1]], dtype=np.float64)
    if degrees == 180:
        return np.array([[-1, 0, width - 1], [0, -1, height - 1], [0, 0, 1]], dtype=np.float64)
    # 270 clockwise / 90 counter-clockwise: (x, y) -> (y, w - 1 - x)
    return np.array([[0, 1, 0], [-1, 0, width - 1], [0, 0, 1]], dtype=np.float64)


def rotate_right_angle(image_rgb: np.ndarray, degrees: int) -> np.ndarray:
    """Rotate an array by a multiple of 90 degrees, matching the homography."""
    degrees = _validate_rotation(degrees)
    if degrees == 0:
        return image_rgb.copy()
    if degrees == 90:
        return cv2.rotate(image_rgb, cv2.ROTATE_90_CLOCKWISE)
    if degrees == 180:
        return cv2.rotate(image_rgb, cv2.ROTATE_180)
    return cv2.rotate(image_rgb, cv2.ROTATE_90_COUNTERCLOCKWISE)


def transform_points(points: np.ndarray | Iterable[Iterable[float]], homography: np.ndarray) -> np.ndarray:
    """Apply a 3×3 homography to ``N×2`` points."""
    points_array = np.asarray(points, dtype=np.float32).reshape(-1, 1, 2)
    transformed = cv2.perspectiveTransform(points_array, homography.astype(np.float64))
    return transformed.reshape(-1, 2)


def order_quad(points: np.ndarray | Sequence[Sequence[float]]) -> np.ndarray:
    """Return a quadrilateral in top-left, top-right, bottom-right, bottom-left order."""
    points_array = np.asarray(points, dtype=np.float32).reshape(4, 2)
    if len({tuple(point) for point in points_array}) != 4:
        raise ValueError("screen_quad must contain four distinct corners")
    center = points_array.mean(axis=0)
    # Angular sorting is robust for highly skewed quadrilaterals, where the
    # common min/max x+y method can select the same corner twice.
    angles = np.arctan2(points_array[:, 1] - center[1], points_array[:, 0] - center[0])
    cyclic = points_array[np.argsort(angles)]
    top_left_index = int(np.argmin(cyclic.sum(axis=1)))
    return np.roll(cyclic, -top_left_index, axis=0)


def full_image_quad(width: int, height: int) -> np.ndarray:
    return np.array(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
        dtype=np.float32,
    )


SCREEN_SCORE_FLOOR = 0.10


def _quad_side_lengths(quad: np.ndarray) -> tuple[float, float]:
    """Return the mean horizontal and vertical side lengths of an ordered quad."""
    top = float(np.linalg.norm(quad[1] - quad[0]))
    bottom = float(np.linalg.norm(quad[2] - quad[3]))
    left = float(np.linalg.norm(quad[3] - quad[0]))
    right = float(np.linalg.norm(quad[2] - quad[1]))
    return (top + bottom) / 2.0, (left + right) / 2.0


def _quad_score(
    quad: np.ndarray,
    image_width: int,
    image_height: int,
    *,
    prefer_portrait: bool = True,
) -> float:
    image_area = float(image_width * image_height)
    area = abs(float(cv2.contourArea(quad)))
    if area < image_area * 0.15 or area > image_area * 0.995:
        return -1.0

    x, y, width, height = cv2.boundingRect(quad.astype(np.int32))
    if width < 40 or height < 40:
        return -1.0
    quad_width, quad_height = _quad_side_lengths(quad)
    # Use the perspective-aware side lengths rather than the enclosing bounding
    # box. A phone photographed at an angle can have a misleading bbox aspect.
    if prefer_portrait and quad_width >= quad_height:
        return -1.0
    # A transfer receipt is rendered on a phone-sized/roughly rectangular screen.
    # Extremely thin panels are normally UI cards, not a device screen.
    short_to_long_side = min(quad_width, quad_height) / max(quad_width, quad_height)
    if short_to_long_side < 0.35:
        return -1.0
    # A contour around the entire image is normally an edge artefact, not a screen.
    touches_all_edges = x <= 2 and y <= 2 and x + width >= image_width - 2 and y + height >= image_height - 2
    if touches_all_edges:
        return -1.0

    rectangularity = area / max(float(width * height), 1.0)
    if rectangularity < 0.48:
        return -1.0
    touches_left = x <= 2
    touches_top = y <= 2
    touches_right = x + width >= image_width - 2
    touches_bottom = y + height >= image_height - 2
    touches = sum((touches_left, touches_top, touches_right, touches_bottom))
    coverage = area / image_area
    width_span = width / image_width
    height_span = height / image_height
    adjacent_edge_contact = (
        (touches_left and touches_top)
        or (touches_top and touches_right)
        or (touches_right and touches_bottom)
        or (touches_bottom and touches_left)
    )
    # Interior UI cards can be rectangular as well. A completely interior screen
    # must be large in *both* image dimensions. Touching opposite edges (for
    # example, a horizontal card spanning left/right) is intentionally not treated
    # as a cropped phone. Adjacent edges are the only safe relaxed case, because a
    # photographed phone can genuinely be cut off at a corner of the source photo.
    if adjacent_edge_contact:
        if coverage < 0.18:
            return -1.0
    elif coverage < 0.30 or width_span < 0.45 or height_span < 0.45:
        return -1.0
    # Large, rectangular candidates are desirable; a tiny penalty allows a screen
    # to touch one or two image borders, as often happens with a cropped photo.
    return coverage * (0.7 + 0.3 * rectangularity) - 0.025 * touches


def find_screen_quad(image_rgb: np.ndarray, *, prefer_portrait: bool = True) -> np.ndarray | None:
    """Find the most likely phone/screen quadrilateral using image geometry.

    This is deliberately conservative.  Returning ``None`` falls back to the
    whole image rather than applying a wrong perspective warp to a screenshot.
    Supply a manual quad in the correction JSON for difficult reflective photos.
    """
    height, width = image_rgb.shape[:2]
    if min(height, width) < 80:
        return None

    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 45, 135)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    best_quad: np.ndarray | None = None
    best_score = -1.0
    for contour in contours:
        perimeter = cv2.arcLength(contour, True)
        if perimeter <= 0:
            continue
        quad = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
        if len(quad) != 4 or not cv2.isContourConvex(quad):
            continue
        candidate = order_quad(quad.reshape(4, 2))
        score = _quad_score(candidate, width, height, prefer_portrait=prefer_portrait)
        if score > best_score:
            best_quad, best_score = candidate, score
    return best_quad if best_score >= SCREEN_SCORE_FLOOR else None


def _quad_dimensions(quad: np.ndarray) -> tuple[int, int]:
    top = np.linalg.norm(quad[1] - quad[0])
    bottom = np.linalg.norm(quad[2] - quad[3])
    left = np.linalg.norm(quad[3] - quad[0])
    right = np.linalg.norm(quad[2] - quad[1])
    # Corner coordinates span ``width - 1`` / ``height - 1`` pixels, so retain
    # the endpoint pixel when a direct screenshot is rectified.
    return max(2, int(round(max(top, bottom))) + 1), max(2, int(round(max(left, right))) + 1)


def warp_quad(image_rgb: np.ndarray, quad: np.ndarray, max_side: int = 1600) -> tuple[np.ndarray, np.ndarray]:
    """Perspective-warp a quadrilateral and return ``(image, H)``.

    ``H`` maps the coordinate system of ``image_rgb`` to the returned image.
    """
    if max_side < 0 or max_side == 1:
        raise ValueError("max_side must be 0 (keep original resolution) or at least 2")
    quad = order_quad(quad)
    output_width, output_height = _quad_dimensions(quad)
    longest_side = max(output_width, output_height)
    if max_side > 0 and longest_side > max_side:
        scale = max_side / longest_side
        output_width = max(2, int(round(output_width * scale)))
        output_height = max(2, int(round(output_height * scale)))
    destination = full_image_quad(output_width, output_height)
    homography = cv2.getPerspectiveTransform(quad.astype(np.float32), destination.astype(np.float32))
    warped = cv2.warpPerspective(
        image_rgb,
        homography,
        (output_width, output_height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return warped, homography.astype(np.float64)


def _choose_orientation(image_rgb: np.ndarray, options: RectificationOptions) -> int:
    if options.orientation_degrees is not None:
        return _validate_rotation(options.orientation_degrees)

    if options.orientation_scorer is not None:
        scored: list[tuple[float, int]] = []
        for degrees in (0, 90, 180, 270):
            candidate = rotate_right_angle(image_rgb, degrees)
            try:
                score = float(options.orientation_scorer(candidate))
            except Exception:
                score = float("-inf")
            if options.prefer_portrait and candidate.shape[0] >= candidate.shape[1]:
                score += 0.02
            scored.append((score, degrees))
        best_score, best_degrees = max(scored, key=lambda item: item[0])
        if np.isfinite(best_score):
            return best_degrees

    # Geometry can reliably correct 90°/270° photos of a portrait receipt.  It
    # cannot distinguish upright from upside-down text, hence OCR scoring above.
    height, width = image_rgb.shape[:2]
    return 90 if options.prefer_portrait and width > height else 0


def rectify_receipt(image_rgb: np.ndarray, options: RectificationOptions | None = None) -> RectificationResult:
    """Rotate and perspective-correct a receipt photo or direct screenshot."""
    if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
        raise ValueError("image_rgb must be an H×W×3 RGB array")
    options = options or RectificationOptions()
    source_rgb = image_rgb.copy()
    source_height, source_width = source_rgb.shape[:2]
    degrees = _choose_orientation(source_rgb, options)
    rotate_h = rotation_homography(source_width, source_height, degrees)
    rotated_rgb = rotate_right_angle(source_rgb, degrees)

    if options.screen_quad is not None:
        quad_original = order_quad(options.screen_quad)
        quad_rotated = order_quad(transform_points(quad_original, rotate_h))
        screen_detected = True
    elif options.auto_screen:
        detected_quad = find_screen_quad(rotated_rgb, prefer_portrait=options.prefer_portrait)
        quad_rotated = detected_quad if detected_quad is not None else full_image_quad(rotated_rgb.shape[1], rotated_rgb.shape[0])
        screen_detected = detected_quad is not None
        quad_original = transform_points(quad_rotated, np.linalg.inv(rotate_h))
    else:
        quad_rotated = full_image_quad(rotated_rgb.shape[1], rotated_rgb.shape[0])
        quad_original = transform_points(quad_rotated, np.linalg.inv(rotate_h))
        screen_detected = False

    rectified_rgb, rotated_to_rectified = warp_quad(rotated_rgb, quad_rotated, max_side=options.max_side)
    original_to_rectified = rotated_to_rectified @ rotate_h
    rectified_to_original = np.linalg.inv(original_to_rectified)
    return RectificationResult(
        source_rgb=source_rgb,
        rectified_rgb=rectified_rgb,
        original_to_rectified=original_to_rectified,
        rectified_to_original=rectified_to_original,
        screen_quad_original=order_quad(quad_original),
        rotation_degrees=degrees,
        screen_detected=screen_detected,
    )


def bbox_to_polygon(bbox_xyxy: Sequence[float]) -> np.ndarray:
    """Convert ``[x1, y1, x2, y2]`` into a clockwise four-point polygon."""
    x1, y1, x2, y2 = (float(value) for value in bbox_xyxy)
    return np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)
