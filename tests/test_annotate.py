"""--annotate flagged/none 的 _should_annotate 决策逻辑单测(不依赖模型/图片)。"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from transfer_receipt_ai.labels import REQUIRED_DETECTION_CLASSES  # noqa: E402
from transfer_receipt_ai.pipeline import _should_annotate  # noqa: E402


def _result(labels):
    """构造一个只有 detections.detection.label 的最小假 result。"""
    dets = [SimpleNamespace(detection=SimpleNamespace(label=name)) for name in labels]
    return SimpleNamespace(detections=dets)


def test_all_mode_always_annotates():
    assert _should_annotate(_result(REQUIRED_DETECTION_CLASSES), "all") is True
    assert _should_annotate(_result([]), "all") is True


def test_none_mode_never_annotates():
    assert _should_annotate(_result(REQUIRED_DETECTION_CLASSES), "none") is False
    assert _should_annotate(_result([]), "none") is False


def test_flagged_skips_complete_receipts():
    # 核心字段齐全(含/不含时钟均可)→ flagged 跳过、不画标注
    assert _should_annotate(_result(("time", *REQUIRED_DETECTION_CLASSES)), "flagged") is False
    assert _should_annotate(_result(REQUIRED_DETECTION_CLASSES), "flagged") is False


def test_flagged_annotates_incomplete_receipts():
    # 缺任一核心字段 → flagged 画标注(需人工复核)
    missing_one = list(REQUIRED_DETECTION_CLASSES)[1:]
    assert _should_annotate(_result(missing_one), "flagged") is True
    # 空检测(核心全缺)也画
    assert _should_annotate(_result([]), "flagged") is True


def test_unknown_mode_falls_back_to_annotate():
    assert _should_annotate(_result(REQUIRED_DETECTION_CLASSES), "whatever") is True
