from __future__ import annotations

import pytest

from transfer_receipt_ai.ocr import normalize_amount, normalize_status, normalize_time


@pytest.mark.parametrize(
    ("raw", "normalized", "fen"),
    (
        ("¥1,234.56", "¥1234.56", 123456),
        ("-¥1,234.56", "-¥1234.56", -123456),
        ("¥-1,234.56", "-¥1234.56", -123456),
    ),
)
def test_normalize_amount_keeps_strict_signed_cny_semantics(raw: str, normalized: str, fen: int) -> None:
    result = normalize_amount(raw)

    assert result is not None
    assert result["normalized"] == normalized
    assert result["amount_fen"] == fen
    assert result["strict_display"] is True


@pytest.mark.parametrize("raw", ("¥12,34.56", "¥1O0.00", "-0.00"))
def test_normalize_amount_does_not_silently_repair_invalid_strict_display(raw: str) -> None:
    # The legacy fallback must not turn a malformed signed/CNY form into a
    # different business value; it is a review candidate instead.
    assert normalize_amount(raw) is None


def test_normalize_time_prefers_a_valid_complete_dashed_datetime_to_its_clock_tail() -> None:
    assert normalize_time("交易时间 2026-07-27 12:34") == "2026-07-27 12:34"
    assert normalize_time("2026-02-30 12:34") == "12:34"


@pytest.mark.parametrize(
    "raw",
    (
        "未转账成功",
        "转账成功与否",
        "转账成功不了",
        "转账成功吗",
        "无法确认该笔款项已经转账成功",
        "这不是一笔真正的转账成功",
    ),
)
def test_normalize_status_rejects_global_negation_or_uncertainty(raw: str) -> None:
    assert normalize_status(raw) == "unknown"


@pytest.mark.parametrize(
    "raw",
    ("转账成功", "交易成功", "付款成功", "支付成功", "转帐成功"),
)
def test_normalize_status_keeps_unqualified_visible_success(raw: str) -> None:
    assert normalize_status(raw) == "success"
