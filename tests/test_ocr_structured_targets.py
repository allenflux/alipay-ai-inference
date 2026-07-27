from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from transfer_receipt_ai.ocr_unified_targets import (
    AMOUNT_AUX_FORMAT,
    AMOUNT_DISPLAY_AUX_FORMAT,
    AMOUNT_SIGN_CLASSES,
    PAYMENT_CARD_TAIL_FORMAT,
    PAYMENT_BANK_PREFIX_FORMAT,
    PARENTHESIS_STYLE_ASCII,
    PARENTHESIS_STYLE_FULLWIDTH,
    TIME_AUX_FORMAT,
    TIME_DISPLAY_AUX_FORMAT,
    TIME_DISPLAY_FORMAT_CLASSES,
    is_structured_target,
    parse_amount_aux_target,
    parse_amount_display_target,
    parse_payment_bank_prefix_target,
    parse_payment_card_tail_target,
    parse_time_aux_target,
    parse_time_display_target,
    recompose_payment_card_tail_target,
    structured_target_config,
)


@pytest.mark.parametrize(
    ("value", "canonical", "sign", "currency", "grouped"),
    (
        ("¥1,234.56", "1234.56", "positive", "¥", True),
        ("￥1,234.56", "1234.56", "positive", "￥", True),
        ("-¥1,234.56", "-1234.56", "negative", "¥", True),
        ("¥-1,234.56", "-1234.56", "negative", "¥", True),
        ("99.99", "99.99", "positive", None, False),
    ),
)
def test_v6_amount_display_target_keeps_visible_symbols_but_has_a_strict_canonical_value(
    value: str, canonical: str, sign: str, currency: str | None, grouped: bool
) -> None:
    target = parse_amount_display_target(value)

    assert target is not None
    assert target["format"] == AMOUNT_DISPLAY_AUX_FORMAT
    assert target["visible_text"] == value
    assert target["canonical_decimal"] == canonical
    assert target["sign"] == sign
    assert target["currency"] == currency
    assert target["grouped_thousands"] is grouped
    assert target["sign"] in AMOUNT_SIGN_CLASSES


@pytest.mark.parametrize(
    "value",
    (
        "12,34.56",
        "1,,234.56",
        "¥1.2",
        "¥1O0.00",
        "--¥1.00",
        "-0.00",
        "¥-0.00",
        " 1.00",
        "1.00 ",
    ),
)
def test_v6_amount_display_target_refuses_ambiguous_or_noncanonical_visible_formats(value: str) -> None:
    assert parse_amount_display_target(value) is None


@pytest.mark.parametrize(
    ("value", "format_name", "digits"),
    (
        ("1:44", "clock_h_mm", "0144"),
        ("01:44", "clock_hh_mm", "0144"),
        ("01:44:05", "clock_hh_mm_ss", "014405"),
        ("2026-07-27 12:34", "date_ymd_hh_mm", "202607271234"),
        ("2026-07-27 12:34:56", "date_ymd_hh_mm_ss", "20260727123456"),
    ),
)
def test_v6_time_display_target_accepts_only_explicit_clock_or_datetime_grammar(
    value: str, format_name: str, digits: str
) -> None:
    target = parse_time_display_target(value)

    assert target is not None
    assert target["format"] == TIME_DISPLAY_AUX_FORMAT
    assert target["format_name"] == format_name
    assert "".join(target["canonical_digits"]) == digits
    assert target["format_name"] in TIME_DISPLAY_FORMAT_CLASSES


@pytest.mark.parametrize("value", ("12-34", "2026-19-27 12:34", "2026-02-30 12:34", "28:99", "2026-07-27T12:34"))
def test_v6_time_display_target_refuses_invalid_or_unlisted_templates(value: str) -> None:
    assert parse_time_display_target(value) is None
from transfer_receipt_ai.ocr_unified_dataset import build_unified_dataset


@pytest.mark.parametrize(
    ("visible_text", "prefix", "tail", "parentheses"),
    (
        ("建设银行储蓄卡(3667)", "建设银行储蓄卡", "3667", PARENTHESIS_STYLE_ASCII),
        ("中国银行储蓄卡（7320）", "中国银行储蓄卡", "7320", PARENTHESIS_STYLE_FULLWIDTH),
    ),
)
def test_payment_card_tail_parser_preserves_visible_prefix_tail_and_parentheses(
    visible_text: str,
    prefix: str,
    tail: str,
    parentheses: str,
) -> None:
    target = parse_payment_card_tail_target(visible_text)

    assert target == {
        "schema_version": 1,
        "format": PAYMENT_CARD_TAIL_FORMAT,
        "visible_text": visible_text,
        "prefix_text": prefix,
        "card_tail": tail,
        "parentheses": parentheses,
    }
    assert is_structured_target(target, expected_format=PAYMENT_CARD_TAIL_FORMAT)
    assert recompose_payment_card_tail_target(
        prefix_text=prefix,
        card_tail=tail,
        parentheses=parentheses,
    ) == visible_text


@pytest.mark.parametrize(
    "visible_text",
    (
        "建设银行储蓄卡(366)",
        "建设银行储蓄卡(36677)",
        "建设银行储蓄卡（３６６７）",
        "建设银行储蓄卡（3667)",
        "建设银行储蓄卡（3667）尾",
        "建设银行储蓄卡 (3667)",
        " 建设银行储蓄卡(3667)",
        "(3667)",
        "",
    ),
)
def test_payment_card_tail_parser_rejects_ambiguous_or_non_four_digit_values(visible_text: str) -> None:
    assert parse_payment_card_tail_target(visible_text) is None


def test_v6_payment_bank_prefix_target_uses_only_an_exact_visible_card_prefix() -> None:
    assert parse_payment_bank_prefix_target("建设银行储蓄卡（3667）") == {
        "schema_version": 1,
        "format": PAYMENT_BANK_PREFIX_FORMAT,
        "visible_prefix": "建设银行储蓄卡",
    }
    assert parse_payment_bank_prefix_target("建设银行储蓄卡（366）") is None


@pytest.mark.parametrize(
    ("prefix", "tail", "parentheses"),
    (
        ("建设银行储蓄卡", "366", PARENTHESIS_STYLE_ASCII),
        ("建设银行储蓄卡", "３６６７", PARENTHESIS_STYLE_FULLWIDTH),
        ("建设银行储蓄卡", "3667", "mixed"),
        ("", "3667", PARENTHESIS_STYLE_ASCII),
    ),
)
def test_payment_card_tail_recomposition_refuses_unverifiable_parts(
    prefix: str,
    tail: str,
    parentheses: str,
) -> None:
    assert recompose_payment_card_tail_target(
        prefix_text=prefix,
        card_tail=tail,
        parentheses=parentheses,
    ) is None


def test_amount_aux_parser_keeps_all_digits_and_masks_only_left_padding() -> None:
    target = parse_amount_aux_target("99.99")

    assert target == {
        "schema_version": 1,
        "format": AMOUNT_AUX_FORMAT,
        "canonical_decimal": "99.99",
        "integer_digits": "99",
        "cents_digits": "99",
        "integer_digit_count": 2,
        "right_aligned_width": 9,
        "right_aligned_digits": [None, None, None, None, None, "9", "9", "9", "9"],
        "right_aligned_mask": [False, False, False, False, False, True, True, True, True],
        "right_aligned_digit_count": 4,
    }
    assert is_structured_target(target, expected_format=AMOUNT_AUX_FORMAT)


@pytest.mark.parametrize(
    "amount",
    (
        "00.00",
        "99.9",
        "99",
        "-1.00",
        "99999999.00",
        "¥99.99",
        "99.99 ",
    ),
)
def test_amount_aux_parser_refuses_values_that_would_need_reformatting(amount: str) -> None:
    assert parse_amount_aux_target(amount) is None


@pytest.mark.parametrize(
    ("value", "text", "hour", "minute", "hour_width"),
    (
        ("1:44", "1:44", "1", "44", 1),
        ("01:44", "01:44", "01", "44", 2),
        ("23：59", "23:59", "23", "59", 2),
    ),
)
def test_time_aux_parser_preserves_hour_width_without_inventing_a_leading_zero(
    value: str,
    text: str,
    hour: str,
    minute: str,
    hour_width: int,
) -> None:
    target = parse_time_aux_target(value)

    assert target == {
        "schema_version": 1,
        "format": TIME_AUX_FORMAT,
        "text": text,
        "hour_text": hour,
        "minute_text": minute,
        "hour_width": hour_width,
    }
    assert is_structured_target(target, expected_format=TIME_AUX_FORMAT)


@pytest.mark.parametrize(
    "value",
    (
        "24:00",
        "12:60",
        "1:4",
        "12:34:56",
        " 12:34",
        "１２:３４",
    ),
)
def test_time_aux_parser_rejects_invalid_or_ambiguous_clock_strings(value: str) -> None:
    assert parse_time_aux_target(value) is None


def test_structured_target_contract_describes_strict_auxiliary_formats() -> None:
    contract = structured_target_config()

    assert contract["schema_version"] == 1
    assert contract["amount_aux"]["format"] == AMOUNT_AUX_FORMAT
    assert contract["amount_aux"]["right_aligned_width"] == 9
    assert contract["time_aux"]["format"] == TIME_AUX_FORMAT
    assert contract["time_aux"]["preserve_hour_width"] is True
    assert contract["payment_card_tail"]["format"] == PAYMENT_CARD_TAIL_FORMAT
    assert contract["payment_card_tail"]["tail_digits"] == 4


def _write_crop(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.full((12, 32, 3), 255, dtype=np.uint8)).save(path)


def _flat_record(
    *,
    index: int,
    field: str,
    text: str,
    semantic_value: str,
    result_json: str,
    group_id: str,
    split: str,
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "id": f"target-{index}",
        "image": f"images/{index}.png",
        "field": field,
        "text": text,
        "paddle_text": text,
        "semantic_value": semantic_value,
        "paddle_confidence": 0.99,
        "detector_score": 0.99,
        "result_json": result_json,
        "source": result_json.removesuffix(".json") + ".png",
        "group_id": group_id,
        "split": split,
        "label_source": "paddle_pseudo",
    }


def test_unified_dataset_attaches_only_safe_structured_targets_without_replacing_visible_ctc_text(
    tmp_path: Path,
) -> None:
    source = tmp_path / "pseudo"
    records = [
        _flat_record(
            index=1,
            field="amount",
            text="¥99.99",
            semantic_value="¥99.99",
            result_json="D:/results/one.json",
            group_id="receipt-one",
            split="train",
        ),
        _flat_record(
            index=2,
            field="time",
            text="1:44",
            semantic_value="1:44",
            result_json="D:/results/one.json",
            group_id="receipt-one",
            split="train",
        ),
        _flat_record(
            index=3,
            field="payment_method_field",
            text="付款方式 建设银行储蓄卡（3667）",
            semantic_value="bank_card",
            result_json="D:/results/one.json",
            group_id="receipt-one",
            split="train",
        ),
        _flat_record(
            index=4,
            field="payment_method_field",
            text="付款方式 余额",
            semantic_value="balance",
            result_json="D:/results/two.json",
            group_id="receipt-two",
            split="val",
        ),
    ]
    for record in records:
        _write_crop(source / str(record["image"]))
    records_path = source / "pseudo_labels.jsonl"
    records_path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )

    output = tmp_path / "unified"
    summary = build_unified_dataset(records_path=records_path, output_dir=output)
    rows = [
        json.loads(line)
        for line in (output / "unified_fields.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    first = next(row for row in rows if row["group_id"] == "receipt-one")
    amount = first["slots"]["amount"]
    time = first["slots"]["time"]
    payment = first["slots"]["payment_method_field"]

    assert amount["text"] == "99.99"
    assert amount["amount_aux"]["canonical_decimal"] == "99.99"
    assert amount["visible_text"] == "¥99.99"
    assert amount["amount_display"]["canonical_decimal"] == "99.99"
    assert time["text"] == "1:44"
    assert time["time_aux"]["hour_width"] == 1
    assert time["visible_text"] == "1:44"
    assert time["time_display"]["format_name"] == "clock_h_mm"
    assert payment["text"] == "建设银行储蓄卡（3667）"
    assert payment["payment_card_tail"]["prefix_text"] == "建设银行储蓄卡"
    assert payment["payment_card_tail"]["card_tail"] == "3667"
    assert payment["payment_bank_prefix"]["visible_prefix"] == "建设银行储蓄卡"

    second = next(row for row in rows if row["group_id"] == "receipt-two")
    assert second["slots"]["payment_method_field"]["text"] == "余额"
    assert "payment_card_tail" not in second["slots"]["payment_method_field"]
    assert summary["structured_target_counts"] == {
        "amount_aux": 1,
        "amount_aux_unparsed": 0,
        "time_aux": 1,
        "time_aux_unparsed": 0,
        "payment_card_tail": 1,
        "payment_card_tail_unparsed": 1,
        "amount_display": 1,
        "amount_display_unparsed": 0,
        "time_display": 1,
        "time_display_unparsed": 0,
        "payment_bank_prefix": 1,
        "payment_bank_prefix_unparsed": 1,
    }
