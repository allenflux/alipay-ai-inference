from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from transfer_receipt_ai.ocr_unified_dataset import (
    KIND_V10,
    SLOT_ORDER,
    V10_SLOT_ORDER,
    build_unified_dataset,
)


def _write_image(path: Path, shade: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.full((24, 96, 3), shade, dtype=np.uint8)).save(path)


def _record(
    *,
    index: int,
    field: str,
    text: str,
    semantic_value: str,
    result_json: str,
    group_id: str,
    split: str,
    confidence: float = 0.99,
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "id": f"sample-{index}",
        "image": f"images/{field}/{index}.png",
        "field": field,
        "text": text,
        "paddle_text": text,
        "semantic_value": semantic_value,
        "paddle_confidence": confidence,
        "detector_score": 0.95,
        "result_json": result_json,
        "source": result_json.removesuffix(".json") + ".png",
        "group_id": group_id,
        "split": split,
        "label_source": "paddle_pseudo",
    }


def test_build_unified_dataset_groups_slots_and_keeps_payment_text(tmp_path: Path) -> None:
    source = tmp_path / "pseudo"
    records = [
        _record(
            index=1,
            field="amount",
            text="¥100.00",
            semantic_value="¥100.00",
            result_json="D:/results/one.json",
            group_id="receipt:GWCZ-one",
            split="train",
        ),
        _record(
            index=2,
            field="time",
            text="12:06",
            semantic_value="12:06",
            result_json="D:/results/one.json",
            group_id="receipt:GWCZ-one",
            split="train",
        ),
        _record(
            index=3,
            field="transfer_status",
            text="转账成功",
            semantic_value="success",
            result_json="D:/results/one.json",
            group_id="receipt:GWCZ-one",
            split="train",
        ),
        _record(
            index=4,
            field="payment_method_field",
            text="付款方式 建设银行储蓄卡(3667)",
            semantic_value="bank_card",
            result_json="D:/results/one.json",
            group_id="receipt:GWCZ-one",
            split="train",
        ),
        _record(
            index=5,
            field="amount",
            text="¥200.00",
            semantic_value="¥200.00",
            result_json="D:/results/two.json",
            group_id="receipt:GWCZ-two",
            split="test",
        ),
    ]
    for index, record in enumerate(records):
        _write_image(source / str(record["image"]), 20 + index)
    records_path = source / "pseudo_labels.jsonl"
    records_path.write_text("".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records), encoding="utf-8")

    output = tmp_path / "unified"
    summary = build_unified_dataset(records_path=records_path, output_dir=output)

    rows = [json.loads(line) for line in (output / "unified_fields.jsonl").read_text(encoding="utf-8").splitlines()]
    assert summary["records"] == 2
    assert summary["complete_records"] == 1
    assert len(rows) == 2
    first = next(row for row in rows if row["group_id"] == "receipt:GWCZ-one")
    assert first["slot_order"] == list(SLOT_ORDER)
    assert first["complete"] is True
    assert first["slots"]["amount"]["text"] == "100.00"
    assert first["slots"]["amount"]["amount_aux"]["canonical_decimal"] == "100.00"
    assert first["slots"]["amount"]["amount_aux"]["right_aligned_digit_count"] == 5
    assert first["slots"]["amount"]["visible_text"] == "¥100.00"
    assert first["slots"]["amount"]["amount_display"]["canonical_decimal"] == "100.00"
    assert first["slots"]["time"]["text"] == "12:06"
    assert first["slots"]["time"]["time_aux"] == {
        "schema_version": 1,
        "format": "clock_h_mm_or_hh_mm_no_seconds_v1",
        "text": "12:06",
        "hour_text": "12",
        "minute_text": "06",
        "hour_width": 2,
    }
    assert first["slots"]["transfer_status"]["class_name"] == "success"
    assert first["slots"]["payment_method_field"]["text"] == "建设银行储蓄卡(3667)"
    assert first["slots"]["payment_method_field"]["payment_card_tail"] == {
        "schema_version": 1,
        "format": "visible_prefix_exact_ascii_4_digit_card_tail_v1",
        "visible_text": "建设银行储蓄卡(3667)",
        "prefix_text": "建设银行储蓄卡",
        "card_tail": "3667",
        "parentheses": "ascii",
    }
    assert first["slots"]["payment_method_field"]["payment_bank_prefix"] == {
        "schema_version": 1,
        "format": "visible_payment_bank_prefix_v6",
        "visible_prefix": "建设银行储蓄卡",
    }
    assert summary["structured_target_counts"] == {
        "amount_aux": 2,
        "amount_aux_unparsed": 0,
        "time_aux": 1,
        "time_aux_unparsed": 0,
        "payment_card_tail": 1,
        "payment_card_tail_unparsed": 0,
        "amount_display": 2,
        "amount_display_unparsed": 0,
        "time_display": 1,
        "time_display_unparsed": 0,
        "payment_bank_prefix": 1,
        "payment_bank_prefix_unparsed": 0,
    }
    assert summary["structured_target_counts_by_split"]["payment_card_tail"] == {
        "train": 1,
        "val": 0,
        "test": 0,
    }
    assert summary["structured_target_config"]["payment_card_tail"]["tail_character_set"] == "ASCII 0-9"


def test_build_unified_dataset_quarantines_conflicting_duplicate_slot(tmp_path: Path) -> None:
    source = tmp_path / "pseudo"
    records = [
        _record(
            index=1,
            field="amount",
            text="¥100.00",
            semantic_value="¥100.00",
            result_json="D:/results/one.json",
            group_id="receipt:GWCZ-one",
            split="train",
            confidence=0.90,
        ),
        _record(
            index=2,
            field="amount",
            text="¥101.00",
            semantic_value="¥101.00",
            result_json="D:/results/one.json",
            group_id="receipt:GWCZ-one",
            split="train",
            confidence=0.99,
        ),
        _record(
            index=3,
            field="time",
            text="12:06",
            semantic_value="12:06",
            result_json="D:/results/one.json",
            group_id="receipt:GWCZ-one",
            split="train",
        ),
    ]
    for index, record in enumerate(records):
        _write_image(source / str(record["image"]), 20 + index)
    records_path = source / "pseudo_labels.jsonl"
    records_path.write_text("".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records), encoding="utf-8")

    output = tmp_path / "unified"
    summary = build_unified_dataset(records_path=records_path, output_dir=output)
    row = json.loads((output / "unified_fields.jsonl").read_text(encoding="utf-8").strip())
    assert "amount" not in row["slots"]
    assert row["slots"]["time"]["text"] == "12:06"
    assert row["ambiguous_slots"] == ["amount"]
    assert summary["ambiguous_slot_records"]["amount"] == 1
    rejected = [json.loads(line) for line in (output / "rejected.jsonl").read_text(encoding="utf-8").splitlines()]
    assert {row["reason"] for row in rejected} == {"ambiguous_duplicate_slot"}


def test_build_unified_dataset_does_not_use_truth_payment_category_as_visible_text(tmp_path: Path) -> None:
    source = tmp_path / "truth"
    amount = _record(
        index=1,
        field="amount",
        text="¥100.00",
        semantic_value="¥100.00",
        result_json="D:/results/one.json",
        group_id="receipt:GWCZ-one",
        split="train",
    )
    payment = _record(
        index=2,
        field="payment_method_field",
        text="bank_card",
        semantic_value="bank_card",
        result_json="D:/results/one.json",
        group_id="receipt:GWCZ-one",
        split="train",
    )
    payment["label_source"] = "transaction_truth"
    for index, record in enumerate((amount, payment)):
        _write_image(source / str(record["image"]), 30 + index)
    records_path = source / "pseudo_labels.jsonl"
    records_path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in (amount, payment)),
        encoding="utf-8",
    )

    output = tmp_path / "unified"
    build_unified_dataset(records_path=records_path, output_dir=output)
    row = json.loads((output / "unified_fields.jsonl").read_text(encoding="utf-8").strip())
    assert set(row["slots"]) == {"amount"}
    rejected = [json.loads(line) for line in (output / "rejected.jsonl").read_text(encoding="utf-8").splitlines()]
    assert rejected[0]["id"] == "sample-2"
    assert rejected[0]["reason"] == "invalid_unified_target"


def test_v10_recipient_ctc_target_retains_visible_line_and_keeps_value_separately(tmp_path: Path) -> None:
    """v10 aligns recipient CTC text with the full detector crop line.

    The data source deliberately carries an unrelated semantic value so this
    test verifies the v10 manifest derives the right-side recipient value from
    the visible teacher line rather than trusting a stale flat-record field.
    It also builds v9 from the same source to prove its value-only payload is
    unchanged.
    """
    source = tmp_path / "recipient-pseudo"
    source_rows = [
        _record(
            index=1,
            field="recipient_field",
            text="收款方   商户甲",
            semantic_value="legacy-semantic-value",
            result_json="D:/results/train.json",
            group_id="receipt:train",
            split="train",
        ),
        _record(
            index=2,
            field="recipient_field",
            text="收款方 商户丙",
            semantic_value="legacy-semantic-value",
            result_json="D:/results/val.json",
            group_id="receipt:val",
            split="val",
        ),
        _record(
            index=3,
            field="recipient_field",
            text="收款方 商户丁",
            semantic_value="legacy-semantic-value",
            result_json="D:/results/test.json",
            group_id="receipt:test",
            split="test",
        ),
    ]
    for index, record in enumerate(source_rows):
        _write_image(source / str(record["image"]), 60 + index)
    records_path = source / "pseudo_labels.jsonl"
    records_path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in source_rows),
        encoding="utf-8",
    )

    v10_output = tmp_path / "unified-v10"
    summary = build_unified_dataset(
        records_path=records_path,
        output_dir=v10_output,
        architecture="v10",
    )
    v10_rows = [
        json.loads(line)
        for line in (v10_output / "unified_fields.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    train_v10 = next(row for row in v10_rows if row["split"] == "train")
    recipient_v10 = train_v10["slots"]["recipient_field"]

    assert summary["kind"] == KIND_V10
    assert summary["architecture"] == "v10"
    assert summary["slot_order"] == list(V10_SLOT_ORDER)
    assert summary["recipient_target"] == "visible_recipient_line_then_extract_value"
    assert summary["recipient_charset_source"] == "train_only_visible_recipient_line"
    assert train_v10["slot_order"] == list(V10_SLOT_ORDER)
    assert recipient_v10["text"] == "收款方 商户甲"
    assert recipient_v10["recipient_visible_text"] == "收款方 商户甲"
    assert recipient_v10["recipient_value"] == "商户甲"
    assert recipient_v10["semantic_value"] == "商户甲"
    assert set(summary["recipient_charset"]) == set("收款方 商户甲")
    assert summary["recipient_oov_by_split"]["train"]["oov_records"] == 0
    assert summary["recipient_oov_by_split"]["val"]["oov_records"] == 1
    assert summary["recipient_oov_by_split"]["test"]["oov_records"] == 1
    assert (v10_output / "recipient_charset.txt").read_text(encoding="utf-8") == (
        "".join(summary["recipient_charset"]) + "\n"
    )

    v9_output = tmp_path / "unified-v9"
    v9_summary = build_unified_dataset(
        records_path=records_path,
        output_dir=v9_output,
        architecture="v9",
    )
    v9_rows = [
        json.loads(line)
        for line in (v9_output / "unified_fields.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    recipient_v9 = next(row for row in v9_rows if row["split"] == "train")["slots"]["recipient_field"]
    assert v9_summary["recipient_target"] == "visible_recipient_value"
    assert v9_summary["recipient_charset_source"] == "train_only_visible_recipient_text"
    assert recipient_v9["text"] == "商户甲"
    assert recipient_v9["semantic_value"] == "legacy-semantic-value"
    assert "recipient_visible_text" not in recipient_v9
    assert "recipient_value" not in recipient_v9
