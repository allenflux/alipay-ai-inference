"""Draw perspective-safe circles around detected receipt fields."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .geometry import transform_points

DISPLAY_NAMES = {
    "time": "时间",
    "amount": "金额",
    "transfer_status": "转账状态",
    "recipient_field": "收款方",
    "payment_method_field": "付款方式",
}
COLORS = {
    "time": (222, 82, 255),
    "amount": (255, 80, 80),
    "transfer_status": (255, 210, 0),
    "recipient_field": (72, 202, 128),
    "payment_method_field": (80, 160, 255),
}

STATUS_STYLE_DISPLAY_VALUES = {
    "check_offset": "Android",
    "check_aligned": "iOS",
    "check_absent": "疑似假图",
    "unknown": "待复核",
}
STATUS_STYLE_COLORS = {
    "check_offset": (66, 153, 89),
    "check_aligned": (70, 118, 210),
    "check_absent": (224, 74, 74),
    "unknown": (225, 145, 35),
}


@dataclass(frozen=True)
class RenderItem:
    label: str
    score: float
    bbox_xyxy: tuple[float, float, float, float]
    text: str | None = None


@dataclass(frozen=True)
class StatusStyleRenderItem:
    """One coordinate-free classifier result appended to the side legend.

    This deliberately is not a ``RenderItem``: status style is inferred from
    the existing ``transfer_status`` crop and does not introduce a sixth
    detector box on the receipt.
    """

    label: str
    confidence: float


def ellipse_polygon(bbox_xyxy: Sequence[float], samples: int = 40) -> np.ndarray:
    """Approximate the circle/ellipse that encloses one bounding box."""
    x1, y1, x2, y2 = (float(value) for value in bbox_xyxy)
    center_x, center_y = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    radius_x, radius_y = max(2.0, (x2 - x1) / 2.0 + 3.0), max(2.0, (y2 - y1) / 2.0 + 3.0)
    angles = np.linspace(0, 2 * np.pi, samples, endpoint=True)
    return np.column_stack((center_x + radius_x * np.cos(angles), center_y + radius_y * np.sin(angles))).astype(np.float32)


def _find_font(size: int) -> ImageFont.ImageFont:
    candidates = [
        "C:/Windows/Fonts/msyh.ttc",
        "C:/Windows/Fonts/msyhbd.ttc",
        "C:/Windows/Fonts/simhei.ttf",
        "C:/Windows/Fonts/simsun.ttc",
        "/System/Library/Fonts/PingFang.ttc",
        "/System/Library/Fonts/STHeiti Light.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
    ]
    for candidate in candidates:
        if Path(candidate).is_file():
            return ImageFont.truetype(candidate, size=size)
    return ImageFont.load_default()


def _item_caption(item: RenderItem) -> str:
    name = DISPLAY_NAMES.get(item.label, item.label)
    suffix = f" {item.text}" if item.text else ""
    return f"{name}{suffix} ({item.score:.0%})"


def _status_style_caption(item: StatusStyleRenderItem) -> str:
    value = STATUS_STYLE_DISPLAY_VALUES.get(item.label, STATUS_STYLE_DISPLAY_VALUES["unknown"])
    return f"设备/风险标签 {value} ({item.confidence:.0%})"


def _wrap_caption(draw: ImageDraw.ImageDraw, caption: str, font: ImageFont.ImageFont, max_width: int) -> str:
    """Wrap mixed Chinese/ASCII captions without relying on whitespace."""
    lines: list[str] = []
    current = ""
    for character in caption:
        candidate = current + character
        if current and draw.textlength(candidate, font=font) > max_width:
            lines.append(current)
            current = character
        else:
            current = candidate
    if current:
        lines.append(current)
    return "\n".join(lines)


def _draw_items(
    image_rgb: np.ndarray,
    item_polygons: Iterable[tuple[RenderItem, np.ndarray]],
    *,
    status_style: StatusStyleRenderItem | None = None,
) -> np.ndarray:
    pairs = sorted(
        list(item_polygons),
        key=lambda pair: tuple(DISPLAY_NAMES).index(pair[0].label) if pair[0].label in DISPLAY_NAMES else 999,
    )
    image = Image.fromarray(image_rgb.copy())
    image_draw = ImageDraw.Draw(image)
    height, width = image_rgb.shape[:2]
    line_width = max(3, min(7, int(round(max(height, width) * 0.002))))
    for item, polygon in pairs:
        color = COLORS.get(item.label, (255, 0, 255))
        points = [tuple(float(value) for value in point) for point in polygon]
        image_draw.line(points, fill=color, width=line_width, joint="curve")

    if not pairs and status_style is None:
        return np.asarray(image)

    # Keep OCR text entirely outside the source pixels. The annotated image is
    # widened with a compact legend so circles remain visible without covering
    # the receipt text that the user needs to inspect.
    font_size = max(16, min(30, height // 40, int(round(max(height, width) * 0.012))))
    font = _find_font(font_size)
    title_font = _find_font(min(34, font_size + 3))
    padding = max(10, font_size // 2)
    panel_width = max(300, min(680, int(round(width * 0.52))))
    canvas = Image.new("RGB", (width + panel_width, height), (247, 248, 250))
    canvas.paste(image, (0, 0))
    draw = ImageDraw.Draw(canvas)
    draw.line([(width, 0), (width, height)], fill=(190, 195, 205), width=max(2, line_width // 2))

    title = "识别结果"
    try:
        draw.text((width + padding, padding), title, fill=(25, 28, 35), font=title_font)
    except UnicodeEncodeError:
        draw.text((width + padding, padding), "Detection results", fill=(25, 28, 35), font=title_font)
    cursor_y = padding + font_size + padding
    text_width = panel_width - padding * 3
    for index, (item, _) in enumerate(pairs, start=1):
        color = COLORS.get(item.label, (255, 0, 255))
        caption = f"{index}. {_item_caption(item)}"
        try:
            caption = _wrap_caption(draw, caption, font, max(100, text_width))
            text_box = draw.multiline_textbbox((0, 0), caption, font=font, spacing=max(3, font_size // 5))
        except UnicodeEncodeError:
            caption = f"{index}. {item.label} ({item.score:.0%})"
            text_box = draw.multiline_textbbox((0, 0), caption, font=font, spacing=3)
        entry_height = max(font_size + padding, text_box[3] - text_box[1] + padding * 2)
        if cursor_y + entry_height > height - padding:
            # A very short landscape image may not have room for all text. Keep
            # the source unobstructed and stop the legend instead of overlaying it.
            break
        left = width + padding
        right = width + panel_width - padding
        draw.rounded_rectangle(
            (left, cursor_y, right, cursor_y + entry_height),
            radius=max(4, padding // 2),
            fill=(255, 255, 255),
            outline=color,
            width=max(2, line_width // 2),
        )
        stripe_width = max(5, padding // 2)
        draw.rounded_rectangle(
            (left, cursor_y, left + stripe_width, cursor_y + entry_height),
            radius=max(3, stripe_width // 2),
            fill=color,
        )
        draw.multiline_text(
            (left + stripe_width + padding, cursor_y + padding),
            caption,
            fill=(25, 28, 35),
            font=font,
            spacing=max(3, font_size // 5),
        )
        cursor_y += entry_height + max(7, padding // 2)

    if status_style is not None:
        color = STATUS_STYLE_COLORS.get(status_style.label, STATUS_STYLE_COLORS["unknown"])
        caption = f"{len(pairs) + 1}. {_status_style_caption(status_style)}"
        try:
            caption = _wrap_caption(draw, caption, font, max(100, text_width))
            text_box = draw.multiline_textbbox((0, 0), caption, font=font, spacing=max(3, font_size // 5))
        except UnicodeEncodeError:
            display_value = STATUS_STYLE_DISPLAY_VALUES.get(
                status_style.label,
                STATUS_STYLE_DISPLAY_VALUES["unknown"],
            )
            caption = f"{len(pairs) + 1}. Device/risk {display_value} ({status_style.confidence:.0%})"
            text_box = draw.multiline_textbbox((0, 0), caption, font=font, spacing=3)
        entry_height = max(font_size + padding, text_box[3] - text_box[1] + padding * 2)
        if cursor_y + entry_height <= height - padding:
            left = width + padding
            right = width + panel_width - padding
            draw.rounded_rectangle(
                (left, cursor_y, right, cursor_y + entry_height),
                radius=max(4, padding // 2),
                fill=(255, 255, 255),
                outline=color,
                width=max(2, line_width // 2),
            )
            stripe_width = max(5, padding // 2)
            draw.rounded_rectangle(
                (left, cursor_y, left + stripe_width, cursor_y + entry_height),
                radius=max(3, stripe_width // 2),
                fill=color,
            )
            draw.multiline_text(
                (left + stripe_width + padding, cursor_y + padding),
                caption,
                fill=(25, 28, 35),
                font=font,
                spacing=max(3, font_size // 5),
            )
    return np.asarray(canvas)


def draw_rectified_circles(
    image_rgb: np.ndarray,
    items: Sequence[RenderItem],
    *,
    status_style: StatusStyleRenderItem | None = None,
) -> np.ndarray:
    """Draw ellipse outlines in the detector's rectified coordinate system."""
    return _draw_items(
        image_rgb,
        ((item, ellipse_polygon(item.bbox_xyxy)) for item in items),
        status_style=status_style,
    )


def draw_original_circles(
    image_rgb: np.ndarray,
    items: Sequence[RenderItem],
    rectified_to_original: np.ndarray,
    *,
    status_style: StatusStyleRenderItem | None = None,
) -> np.ndarray:
    """Project rectified ellipses back into the photo before drawing them.

    A circle therefore becomes the correct perspective curve in the original
    photo instead of an inaccurate axis-aligned rectangle.
    """
    return _draw_items(
        image_rgb,
        (
            (item, transform_points(ellipse_polygon(item.bbox_xyxy), rectified_to_original))
            for item in items
        ),
        status_style=status_style,
    )
