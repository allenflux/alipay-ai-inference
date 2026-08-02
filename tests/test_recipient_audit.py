from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from transfer_receipt_ai.recipient_audit import (
    audit_recipient_crop,
    audit_recipient_pixels,
    nearest_blank_column_gap,
)


def test_audit_crop_matches_v11_trim_math_and_reports_blank_gap(tmp_path) -> None:
    pixels = np.full((6, 20), 255, dtype=np.uint8)
    pixels[:, 3:6] = 0
    pixels[:, 13:17] = 0
    image_path = tmp_path / "recipient.png"
    Image.fromarray(pixels, mode="L").save(image_path)

    audit = audit_recipient_crop(image_path, left_trim_ratio=0.50)

    assert audit.image_path == image_path.resolve().as_posix()
    assert (audit.width, audit.height, audit.trim_px) == (20, 6, 10)
    assert audit.retained_width_px == 10
    assert audit.retained_width_ratio == 0.5
    assert audit.retained_aspect_ratio == pytest.approx(10 / 6)
    assert audit.foreground_detection_method == "dominant_grayscale_mode_abs_difference"
    assert audit.dominant_background_luma == 255
    assert audit.total_ink_pixels == 42
    assert audit.left_ink_pixels == 18
    assert audit.right_ink_pixels == 24
    assert audit.cut_column_indices == (8, 9, 10, 11, 12)
    assert audit.cut_column_ink_counts == (0, 0, 0, 0, 0)
    assert audit.cut_window_ink_pixels == 0
    assert not audit.cut_window_has_ink
    assert audit.nearest_blank_gap is not None
    assert (audit.nearest_blank_gap.start_px, audit.nearest_blank_gap.end_px_exclusive) == (6, 13)
    assert audit.nearest_blank_gap.touches_trim_boundary
    assert audit.as_dict()["image_path"] == image_path.resolve().as_posix()
    assert audit.as_dict()["cut_window_has_ink"] is False


def test_audit_marks_ink_that_crosses_trim_boundary() -> None:
    pixels = np.full((4, 20), 255, dtype=np.uint8)
    pixels[:, 7:13] = 0

    audit = audit_recipient_pixels(pixels, left_trim_ratio=0.50, cut_radius=2)

    assert audit.trim_px == 10
    assert audit.cut_column_indices == (8, 9, 10, 11, 12)
    assert audit.cut_column_ink_counts == (4, 4, 4, 4, 4)
    assert audit.cut_window_ink_pixels == 20
    assert audit.cut_window_has_ink
    assert audit.as_dict()["cut_window_has_ink"] is True
    assert audit.nearest_blank_gap is not None
    assert audit.nearest_blank_gap.distance_to_trim_px == 3
    assert not audit.nearest_blank_gap.touches_trim_boundary


def test_audit_detects_white_text_on_a_blue_background(tmp_path) -> None:
    pixels = np.zeros((6, 20, 3), dtype=np.uint8)
    pixels[:, :] = (20, 110, 240)
    pixels[:, 11:14] = (255, 255, 255)
    image_path = tmp_path / "blue-recipient.png"
    Image.fromarray(pixels, mode="RGB").save(image_path)

    audit = audit_recipient_crop(image_path, left_trim_ratio=0.50)

    assert audit.dominant_background_luma < 150
    assert audit.cut_column_ink_counts == (0, 0, 0, 6, 6)
    assert audit.total_ink_pixels == 18
    assert audit.cut_window_has_ink


def test_nearest_blank_gap_prefers_gap_touching_trim_boundary() -> None:
    gap = nearest_blank_column_gap([3, 3, 0, 0, 0, 3], trim_px=3)

    assert gap is not None
    assert (gap.start_px, gap.end_px_exclusive, gap.width_px) == (2, 5, 3)
    assert gap.distance_to_trim_px == 0
    assert gap.touches_trim_boundary


@pytest.mark.parametrize("ratio", (-0.1, 1.0, float("nan")))
def test_audit_rejects_invalid_trim_ratio(ratio: float) -> None:
    with pytest.raises(ValueError, match="left_trim_ratio"):
        audit_recipient_pixels(np.full((2, 2), 255, dtype=np.uint8), left_trim_ratio=ratio)


def test_audit_requires_two_dimensional_grayscale_pixels() -> None:
    with pytest.raises(ValueError, match="two-dimensional grayscale"):
        audit_recipient_pixels(np.zeros((2, 2, 3), dtype=np.uint8), left_trim_ratio=0.3)
