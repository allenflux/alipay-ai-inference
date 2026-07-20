"""OCR abstraction and conservative normalisation for detected receipt fields."""

from __future__ import annotations

import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from typing import Any, Protocol

import numpy as np


@dataclass(frozen=True)
class OCRResult:
    text: str
    confidence: float | None
    lines: tuple[tuple[str, float], ...] = ()


class TextRecognizer(Protocol):
    def recognize(self, image_rgb: np.ndarray) -> OCRResult:
        """Read text from an RGB image crop."""


def clean_text(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


_FIELD_LABELS = {
    "recipient": ("收款方", "收款人", "收款账户", "收款账号"),
    "payment_method": ("付款方式", "交易方式", "付款渠道", "支付方式"),
}


def extract_field_value(raw_text: str, field: str) -> str:
    """Extract the right-side value from OCR of a complete receipt row.

    The detector boxes the entire row selected by the user (for example,
    ``收款方 富森(**森)``). The raw OCR remains available, while this function
    returns just the business value for structured output.
    """
    text = clean_text(raw_text)
    labels = _FIELD_LABELS.get(field)
    if labels is None:
        raise ValueError(f"Unknown field: {field}")
    for label in labels:
        position = text.find(label)
        if position >= 0:
            value = text[position + len(label) :].lstrip(" :：-—")
            if value:
                return value
            # OCR occasionally emits the right-hand value before the left-hand
            # label. Keep that value instead of returning the whole row.
            value_before_label = text[:position].rstrip(" :：-—")
            if value_before_label:
                return value_before_label
    return text


def _extract_paddle_lines(payload: Any) -> list[tuple[str, float]]:
    """Support PaddleOCR 2.x lists and PaddleOCR 3.x ``Result`` objects."""
    lines: list[tuple[str, float]] = []

    def visit(node: Any) -> None:
        if node is None:
            return
        # PaddleOCR 3.x ``predict`` yields Result objects.  Their public
        # ``json`` attribute is the stable, documented representation.
        if not isinstance(node, (dict, list, tuple)) and hasattr(node, "json"):
            json_payload = node.json
            if callable(json_payload):
                json_payload = json_payload()
            visit(json_payload)
            return
        if isinstance(node, dict):
            texts = node.get("rec_texts")
            if texts is None:
                texts = node.get("texts")
            scores = node.get("rec_scores")
            if scores is None:
                scores = node.get("scores")
            if isinstance(texts, np.ndarray):
                texts = texts.tolist()
            if isinstance(texts, (list, tuple)):
                if isinstance(scores, np.ndarray):
                    scores = scores.tolist()
                scores = scores if isinstance(scores, (list, tuple)) else [None] * len(texts)
                for text, score in zip(texts, scores):
                    if isinstance(text, str):
                        lines.append((text, float(score) if score is not None else 0.0))
                return
            for value in node.values():
                visit(value)
            return
        if isinstance(node, (list, tuple)):
            # PaddleOCR v2 line: [[x1,y1], ...], ("text", confidence)
            if (
                len(node) >= 2
                and isinstance(node[1], (list, tuple))
                and len(node[1]) >= 2
                and isinstance(node[1][0], str)
            ):
                try:
                    lines.append((node[1][0], float(node[1][1])))
                except (TypeError, ValueError):
                    lines.append((node[1][0], 0.0))
                return
            for value in node:
                visit(value)

    visit(payload)
    return lines


class PaddleOCRReader:
    """Chinese OCR backed by PaddleOCR, imported only when the feature is used."""

    def __init__(self, language: str = "ch", use_angle_cls: bool = True) -> None:
        try:
            import paddleocr
        except ModuleNotFoundError as error:
            raise ImportError(
                "PaddleOCR is not installed. Install a PaddlePaddle wheel for this platform, "
                "then run `pip install -r requirements-ocr.txt`."
            ) from error
        PaddleOCR = paddleocr.PaddleOCR
        self._use_angle_cls = use_angle_cls

        # PaddleOCR 3.x replaced ``use_angle_cls``/``show_log`` and the
        # ``ocr`` call with pipeline options plus ``predict``.  Prefer its
        # documented API, then fall back to the 2.x constructor.
        v3_options = {
            "lang": language,
            "use_doc_orientation_classify": False,
            "use_doc_unwarping": False,
            "use_textline_orientation": use_angle_cls,
        }

        def initialise_v2() -> Any:
            v2_options = {"lang": language, "use_angle_cls": use_angle_cls, "show_log": False}
            try:
                return PaddleOCR(**v2_options)
            except (TypeError, ValueError):  # Some late 2.x builds removed show_log.
                v2_options.pop("show_log")
                return PaddleOCR(**v2_options)

        version_match = re.match(r"(\d+)", str(getattr(paddleocr, "__version__", "")))
        major_version = int(version_match.group(1)) if version_match else None
        if major_version is not None and major_version >= 3:
            self._engine = PaddleOCR(**v3_options)
            self._api_version = 3
        elif major_version == 2:
            self._engine = initialise_v2()
            self._api_version = 2
        else:
            # Unknown/private builds do not always expose __version__. Probe
            # the v3 constructor, then retain compatibility with v2.
            try:
                self._engine = PaddleOCR(**v3_options)
                self._api_version = 3
            except (TypeError, ValueError):
                self._engine = initialise_v2()
                self._api_version = 2

    def recognize(self, image_rgb: np.ndarray) -> OCRResult:
        if self._api_version == 3:
            raw = self._engine.predict(image_rgb)
        else:
            raw = self._engine.ocr(image_rgb, cls=self._use_angle_cls)
        lines = _extract_paddle_lines(raw)
        if not lines:
            return OCRResult(text="", confidence=None)
        text = " ".join(clean_text(line[0]) for line in lines if clean_text(line[0]))
        confidence = sum(line[1] for line in lines) / len(lines)
        return OCRResult(text=text, confidence=confidence, lines=tuple(lines))

    def orientation_score(self, image_rgb: np.ndarray) -> float:
        """Return a text-quality score for one image orientation.

        The image preparation command evaluates this for 0/90/180/270 degrees.
        It is slower than EXIF/geometry correction but can distinguish upside-down
        Chinese text, which a rectangle-only method cannot do.
        """
        result = self.recognize(image_rgb)
        if not result.text or result.confidence is None:
            return float("-inf")
        return result.confidence + min(len(result.lines), 10) * 0.01


_AMOUNT_PATTERN = re.compile(r"(?:[¥￥]\s*)?([0-9OoIl]{1,3}(?:,[0-9OoIl]{3})*(?:\.\d{1,2})?)")
_TIME_PATTERN = re.compile(r"(?<!\d)(\d{1,2}:\d{2}(?::\d{2})?)(?!\d)")


def normalize_time(raw_text: str) -> str | None:
    """Extract the visible status-bar time without inventing a transaction time."""
    # OCR commonly returns a full-width Chinese colon. Do not use ``\b`` here:
    # Chinese characters next to the time count as Unicode word characters.
    candidates = _TIME_PATTERN.findall(clean_text(raw_text).replace("：", ":"))

    def valid_status_time(candidate: str) -> bool:
        values = [int(value) for value in candidate.split(":")]
        if len(values) not in {2, 3}:
            return False
        hour, minute = values[:2]
        second = values[2] if len(values) == 3 else 0
        return 0 <= hour <= 23 and 0 <= minute <= 59 and 0 <= second <= 59

    for candidate in candidates:
        if valid_status_time(candidate):
            return candidate
    # PaddleOCR occasionally reads a short status-bar crop from right to left,
    # e.g. visible 00:08 becomes 80:00. Only reverse when the OCR candidate is
    # impossible as a clock time and the full reversed string is valid.
    for candidate in candidates:
        reversed_candidate = candidate[::-1]
        if valid_status_time(reversed_candidate):
            return reversed_candidate
    return None


def normalize_amount(raw_text: str) -> dict[str, object] | None:
    """Normalise a CNY amount while retaining the raw OCR text separately."""
    raw_text = clean_text(raw_text)
    matches = list(_AMOUNT_PATTERN.finditer(raw_text))
    if not matches:
        return None
    # Currency-marked and decimal candidates are much more likely to be the value
    # than unrelated UI digits such as a time in the status bar.
    match = max(
        matches,
        key=lambda item: (
            1 if item.group(0).lstrip().startswith(("¥", "￥")) else 0,
            1 if "." in item.group(1) else 0,
            len(item.group(1)),
        ),
    )
    numeric = match.group(1).translate(str.maketrans({"O": "0", "o": "0", "I": "1", "l": "1"})).replace(",", "")
    try:
        value = Decimal(numeric).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
    except InvalidOperation:
        return None
    if value < 0:
        return None
    fen = int(value * 100)
    return {
        "raw": raw_text,
        "normalized": f"¥{value:.2f}",
        "amount_fen": fen,
        "currency": "CNY",
    }


def normalize_status(raw_text: str) -> str:
    compact = re.sub(r"\s+", "", raw_text)
    if any(token in compact for token in ("失败", "未成功", "已撤销")):
        return "failed"
    if any(token in compact for token in ("处理中", "待处理", "进行中")):
        return "pending"
    if any(token in compact for token in ("转账成功", "交易成功", "付款成功", "支付成功", "转帐成功")):
        return "success"
    return "unknown"


def normalize_payment_method(raw_text: str) -> dict[str, str]:
    raw = clean_text(raw_text)
    compact = re.sub(r"\s+", "", raw)
    if "余额宝" in compact:
        kind = "yuebao"
    elif "余额" in compact:
        kind = "balance"
    elif "花呗" in compact:
        kind = "huabei"
    elif any(token in compact for token in ("银行卡", "储蓄卡", "信用卡")):
        kind = "bank_card"
    else:
        kind = "other"
    return {"raw": raw, "normalized": kind}
