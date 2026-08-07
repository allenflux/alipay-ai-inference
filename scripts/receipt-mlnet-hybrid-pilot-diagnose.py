#!/usr/bin/env python3
"""Summarize fail-closed hybrid-recipient pilot records without rerunning OCR."""

from __future__ import annotations

import argparse
from decimal import Decimal, InvalidOperation
import json
import math
from pathlib import Path
import re
from typing import Any, Mapping


RECIPIENT_LABELS = ("收款方", "收款人", "收款账户", "收款账号")
NON_RECIPIENT_LABELS = (
    "付款方式",
    "支付方式",
    "交易方式",
    "付款渠道",
    "转账成功",
    "支付成功",
    "交易成功",
    "付款成功",
    "金额",
    "时间",
    "订单号",
    "商品",
    "优惠",
    "活动",
    "充值",
    "奖励",
    "红包",
    "积分",
    "广告",
    "推荐",
)
PAIR_PATTERN = re.compile(
    r"^(?P<merchant>.+?)\s*[¥￥]\s*(?P<amount>(?:[0-9]+|[0-9]{1,3}(?:,[0-9]{3})+)(?:\.[0-9]{1,2})?)$"
)
EXPECTED_AMOUNT_PATTERN = re.compile(
    r"^(?:[¥￥]\s*)?(?P<amount>(?:[0-9]+|[0-9]{1,3}(?:,[0-9]{3})+)(?:\.[0-9]{1,2})?)$"
)


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _source_key(value: object) -> str:
    return str(value or "").replace("\\", "/").casefold()


def _box(record: Mapping[str, Any] | None) -> list[float] | None:
    if not isinstance(record, Mapping):
        return None
    raw = record.get("bbox_image")
    if not isinstance(raw, list) or len(raw) < 4:
        return None
    try:
        values = [float(value) for value in raw[:4]]
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(value) for value in values):
        return None
    if values[2] <= values[0] or values[3] <= values[1]:
        return None
    return values


def _score(record: Mapping[str, Any] | None) -> float | None:
    if not isinstance(record, Mapping):
        return None
    try:
        value = float(record.get("score"))
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _geometry_reasons(
    result: Mapping[str, Any], detections: Mapping[str, Mapping[str, Any]]
) -> list[str]:
    geometry = result.get("geometry")
    size = geometry.get("rectified_size") if isinstance(geometry, Mapping) else None
    try:
        width = int(size.get("width"))
        height = int(size.get("height"))
    except (AttributeError, TypeError, ValueError):
        return ["rectified_size_missing"]
    recipient = detections.get("recipient_field")
    amount = detections.get("amount")
    payment = detections.get("payment_method_field")
    reasons: list[str] = []
    for name, record, floor in (
        ("recipient", recipient, 0.90),
        ("amount", amount, 0.80),
        ("payment", payment, 0.80),
    ):
        score = _score(record)
        if score is None:
            reasons.append(f"{name}_score_missing")
        elif score < floor:
            reasons.append(f"{name}_score_below_{floor:.2f}")
        if _box(record) is None:
            reasons.append(f"{name}_box_invalid")
    recipient_box = _box(recipient)
    amount_box = _box(amount)
    payment_box = _box(payment)
    if reasons or recipient_box is None or amount_box is None or payment_box is None:
        return reasons
    recipient_width = recipient_box[2] - recipient_box[0]
    recipient_height = recipient_box[3] - recipient_box[1]
    recipient_center = (recipient_box[1] + recipient_box[3]) * 0.5
    amount_center = (amount_box[1] + amount_box[3]) * 0.5
    payment_center = (payment_box[1] + payment_box[3]) * 0.5
    tolerance = max(4.0, recipient_height * 0.25)
    checks = (
        (recipient_box[0] <= width * 0.20, "recipient_left_edge"),
        (recipient_box[2] >= width * 0.80, "recipient_right_edge"),
        (recipient_width >= width * 0.60, "recipient_width"),
        (recipient_height <= height * 0.15, "recipient_height"),
        (amount_center < recipient_center, "amount_before_recipient"),
        (recipient_center < payment_center, "recipient_before_payment"),
        (recipient_box[1] >= amount_box[3] - tolerance, "amount_edge_overlap"),
        (recipient_box[3] <= payment_box[1] + tolerance, "payment_edge_overlap"),
    )
    reasons.extend(reason for passed, reason in checks if not passed)
    return reasons


def _amount_fen(value: object) -> int | None:
    match = EXPECTED_AMOUNT_PATTERN.fullmatch(str(value or "").strip())
    if match is None:
        return None
    try:
        amount = Decimal(match.group("amount").replace(",", ""))
    except InvalidOperation:
        return None
    return int((amount.quantize(Decimal("0.01")) * 100).to_integral_exact())


def _aggregate_pair_reasons(raw: object, line_count: object, expected: object) -> list[str]:
    text = " ".join(str(raw or "").split())
    reasons: list[str] = []
    if line_count != 2:
        reasons.append("line_count_not_2")
    match = PAIR_PATTERN.fullmatch(text)
    if match is None:
        reasons.append("aggregate_not_merchant_currency_amount")
        return reasons
    merchant = match.group("merchant").strip()
    if len(merchant) < 2 or len(merchant) > 64:
        reasons.append("merchant_length")
    if not any("\u3400" <= character <= "\u9fff" for character in merchant):
        reasons.append("merchant_has_no_cjk")
    if any(label in merchant for label in RECIPIENT_LABELS + NON_RECIPIENT_LABELS):
        reasons.append("merchant_blocklisted")
    observed = _amount_fen(match.group("amount"))
    expected_fen = _amount_fen(expected)
    if observed is None or expected_fen is None:
        reasons.append("amount_parse_failed")
    elif observed != expected_fen:
        reasons.append("amount_mismatch")
    return reasons


def diagnose(
    comparison: Path, hybrid: Path, records: Path | None = None
) -> list[dict[str, Any]]:
    comparison_rows = [
        json.loads(line)
        for line in (comparison / "comparisons.jsonl").read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]
    failed = [row for row in comparison_rows if row.get("invariant") is False]
    manifest = _load_json(hybrid / "inference_manifest.json")
    results: dict[str, Path] = {}
    for record in manifest:
        result_path = Path(str(record["result"]))
        if not result_path.is_absolute():
            result_path = hybrid / result_path
        results[_source_key(record.get("source"))] = result_path
    references: dict[str, dict[str, Any]] = {}
    if records is not None:
        for line in records.read_text(encoding="utf-8-sig").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("split") != "val":
                continue
            slots = record.get("slots")
            if not isinstance(slots, Mapping):
                continue
            recipient_slot = slots.get("recipient_field")
            amount_slot = slots.get("amount")
            references[_source_key(record.get("source"))] = {
                "recipient": recipient_slot.get("text")
                if isinstance(recipient_slot, Mapping)
                else None,
                "amount": (
                    amount_slot.get("visible_text") or amount_slot.get("text")
                )
                if isinstance(amount_slot, Mapping)
                else None,
            }

    diagnostics: list[dict[str, Any]] = []
    for comparison_row in failed:
        source = comparison_row.get("source")
        result = _load_json(results[_source_key(source)])
        fields = result.get("fields") or {}
        recipient = fields.get("recipient") or {}
        amount = fields.get("amount") or {}
        expected_amount = amount.get("candidate")
        detections = {
            record.get("label"): record
            for record in result.get("detections") or []
            if isinstance(record, Mapping)
        }
        geometry_reasons = _geometry_reasons(result, detections)
        first_reasons = _aggregate_pair_reasons(
            recipient.get("hybrid_ocr_first_raw"),
            recipient.get("hybrid_ocr_first_line_count"),
            expected_amount,
        )
        retry_reasons = _aggregate_pair_reasons(
            recipient.get("hybrid_ocr_retry_raw"),
            recipient.get("hybrid_ocr_retry_line_count"),
            expected_amount,
        )
        if geometry_reasons:
            likely = geometry_reasons
        elif not first_reasons or not retry_reasons:
            likely = ["per_line_confidence_or_exact_line_split"]
        else:
            likely = sorted(set(first_reasons + retry_reasons))
        diagnostics.append(
            {
                "source": source,
                "reference": references.get(_source_key(source)),
                "failures": comparison_row.get("failures"),
                "amount_candidate": expected_amount,
                "recipient_score": _score(detections.get("recipient_field")),
                "amount_score": _score(detections.get("amount")),
                "payment_score": _score(detections.get("payment_method_field")),
                "geometry_reasons": geometry_reasons,
                "first_raw": recipient.get("hybrid_ocr_first_raw"),
                "first_line_count": recipient.get("hybrid_ocr_first_line_count"),
                "first_pair_reasons": first_reasons,
                "retry_raw": recipient.get("hybrid_ocr_retry_raw"),
                "retry_line_count": recipient.get("hybrid_ocr_retry_line_count"),
                "retry_pair_reasons": retry_reasons,
                "likely_blocker": likely,
            }
        )
    return diagnostics


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison", type=Path, required=True)
    parser.add_argument("--hybrid", type=Path, required=True)
    parser.add_argument("--records", type=Path)
    args = parser.parse_args()
    rows = diagnose(
        args.comparison.resolve(),
        args.hybrid.resolve(),
        args.records.resolve() if args.records is not None else None,
    )
    for row in rows:
        print(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
    print(json.dumps({"failed_records": len(rows)}, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
