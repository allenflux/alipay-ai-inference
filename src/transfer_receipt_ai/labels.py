"""Shared label contract for annotations, training and inference."""

from __future__ import annotations

from typing import Final

# Keep this order fixed: model labels are one-based and background is zero.
# Each class corresponds to one complete region marked by the user's red lines.
DETECTION_CLASSES: Final[tuple[str, ...]] = (
    "time",
    "amount",
    "transfer_status",
    "recipient_field",
    "payment_method_field",
)

# require-complete 判定用的"核心交易字段":这几个必须都检测到才算完整。
# 刻意排除 time(顶部状态栏时钟)——它不是交易字段,而且设备识别已从状态栏读取,
# 不应因为少了时钟(常见于安卓右侧时钟漏检、通知横幅遮挡)就把整张回单丢弃。
REQUIRED_DETECTION_CLASSES: Final[tuple[str, ...]] = tuple(
    name for name in DETECTION_CLASSES if name != "time"
)

BACKGROUND_LABEL: Final[int] = 0
LABEL_TO_ID: Final[dict[str, int]] = {
    name: index for index, name in enumerate(DETECTION_CLASSES, start=1)
}
ID_TO_LABEL: Final[dict[int, str]] = {value: key for key, value in LABEL_TO_ID.items()}
NUM_MODEL_CLASSES: Final[int] = len(DETECTION_CLASSES) + 1


def validate_label(name: str) -> str:
    """Return a validated label name or raise a helpful error."""
    if name not in LABEL_TO_ID:
        accepted = ", ".join(DETECTION_CLASSES)
        raise ValueError(f"Unknown label {name!r}. Expected one of: {accepted}")
    return name
