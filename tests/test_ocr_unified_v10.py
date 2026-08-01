"""Focused manifest contract tests for the v10 recipient-reader protocol."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import pytest

from transfer_receipt_ai.ocr_unified import (
    KIND_V10 as READER_KIND_V10,
    V10_ONNX_OUTPUT_NAMES,
    UnifiedReaderConfig,
    _checkpoint_config,
    _delivery_text,
    _recipient_candidate_value,
    _recipient_expected_value,
    build_unified_reader,
    evaluate_unified_onnx,
    export_unified_onnx,
    load_records,
    train_unified_reader,
)
from transfer_receipt_ai.ocr_unified_dataset import (
    KIND_V10,
    V9_SLOT_ORDER,
    V10_SLOT_ORDER,
    build_unified_dataset,
    slot_order_for_architecture,
)


# A valid 1x1 opaque PNG.  The dataset builder checks source-image existence
# only; keeping this fixture dependency-free makes the v10 manifest contract
# test useful even in a minimal non-training Python environment.
_TINY_PNG = bytes.fromhex(
    "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c489"
    "0000000d49444154789c6360f8cfc00000030101874f5dc30000000049454e44ae426082"
)


def _source_row(*, index: int, split: str, text: str) -> dict[str, object]:
    return {
        "schema_version": 1,
        "id": f"recipient-{split}-{index}",
        "image": f"images/recipient_field/{split}-{index}.png",
        "field": "recipient_field",
        "text": text,
        "paddle_text": text,
        # Deliberately stale: v10 must derive its semantic value from the
        # visible teacher line rather than copying this flat-manifest value.
        "semantic_value": "stale-flat-semantic-value",
        "paddle_confidence": 0.99,
        "detector_score": 0.95,
        "result_json": f"D:/teacher/{split}.json",
        "source": f"D:/source/{split}.png",
        "group_id": f"receipt:{split}",
        "split": split,
        "label_source": "paddle_pseudo",
    }


def _write_source_manifest(tmp_path: Path) -> Path:
    source = tmp_path / "teacher-labels"
    rows = (
        _source_row(index=1, split="train", text="收款方   商户甲"),
        _source_row(index=2, split="val", text="收款方 商户丙"),
        _source_row(index=3, split="test", text="收款方 商户丁"),
    )
    for row in rows:
        image = source / str(row["image"])
        image.parent.mkdir(parents=True, exist_ok=True)
        image.write_bytes(_TINY_PNG)
    manifest = source / "pseudo_labels.jsonl"
    manifest.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    return manifest


def _rows(output: Path) -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in (output / "unified_fields.jsonl").read_text(encoding="utf-8").splitlines()
    ]


def _write_complete_source_manifest(tmp_path: Path) -> Path:
    """Make one small five-slot receipt per split for lifecycle coverage."""
    source = tmp_path / "complete-teacher-labels"
    records: list[dict[str, object]] = []
    recipients = {
        "train": "收款方 商户甲",
        "val": "收款方 商户丙",
        "test": "收款方 商户丁",
    }
    index = 0
    for split, recipient_line in recipients.items():
        values = (
            ("amount", "¥100.00", "¥100.00"),
            ("time", "12:06", "12:06"),
            ("transfer_status", "转账成功", "success"),
            ("payment_method_field", "付款方式 建设银行储蓄卡(3667)", "bank_card"),
            ("recipient_field", recipient_line, recipient_line.removeprefix("收款方 ")),
        )
        for field, text, semantic_value in values:
            index += 1
            image_name = f"images/{field}/{split}-{index}.png"
            image = source / image_name
            image.parent.mkdir(parents=True, exist_ok=True)
            image.write_bytes(_TINY_PNG)
            records.append(
                {
                    "schema_version": 1,
                    "id": f"sample-{split}-{index}",
                    "image": image_name,
                    "field": field,
                    "text": text,
                    "paddle_text": text,
                    "semantic_value": semantic_value,
                    "paddle_confidence": 0.99,
                    "detector_score": 0.95,
                    "result_json": f"D:/results/{split}.json",
                    "source": f"D:/images/{split}.png",
                    "group_id": f"receipt:{split}",
                    "split": split,
                    "label_source": "paddle_pseudo",
                }
            )
    manifest = source / "pseudo_labels.jsonl"
    manifest.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )
    return manifest


def test_v10_recipient_manifest_uses_visible_line_ctc_target_and_extracted_value(tmp_path: Path) -> None:
    """v10 learns the pixels it sees while keeping business comparison value-only."""
    manifest = _write_source_manifest(tmp_path)
    output = tmp_path / "v10"

    summary = build_unified_dataset(
        records_path=manifest,
        output_dir=output,
        architecture="v10",
    )

    train = next(row for row in _rows(output) if row["split"] == "train")
    slot = dict(train["slots"])["recipient_field"]
    assert isinstance(slot, dict)
    assert slot_order_for_architecture("v10") == V10_SLOT_ORDER == V9_SLOT_ORDER
    assert summary["kind"] == KIND_V10
    assert summary["architecture"] == "v10"
    assert summary["slot_order"] == list(V10_SLOT_ORDER)
    assert summary["recipient_target"] == "visible_recipient_line_then_extract_value"
    assert summary["recipient_charset_source"] == "train_only_visible_recipient_line"

    # Whitespace normalisation is part of the frozen target.  It preserves the
    # entire line including the label, while the semantic/runtime value remains
    # just the recipient name.
    assert slot["text"] == "收款方 商户甲"
    assert slot["recipient_visible_text"] == "收款方 商户甲"
    assert slot["recipient_value"] == "商户甲"
    assert slot["semantic_value"] == "商户甲"
    assert set(summary["recipient_charset"]) == set("收款方 商户甲")
    assert summary["recipient_oov_by_split"]["train"]["oov_records"] == 0
    assert summary["recipient_oov_by_split"]["val"]["oov_records"] == 1
    assert summary["recipient_oov_by_split"]["test"]["oov_records"] == 1
    assert (output / "recipient_charset.txt").read_text(encoding="utf-8") == (
        "".join(summary["recipient_charset"]) + "\n"
    )


def test_v10_does_not_change_v9_value_only_recipient_manifest(tmp_path: Path) -> None:
    """The v9 payload remains an incompatible, value-only protocol."""
    manifest = _write_source_manifest(tmp_path)
    output = tmp_path / "v9"

    summary = build_unified_dataset(
        records_path=manifest,
        output_dir=output,
        architecture="v9",
    )

    train = next(row for row in _rows(output) if row["split"] == "train")
    slot = dict(train["slots"])["recipient_field"]
    assert isinstance(slot, dict)
    assert summary["architecture"] == "v9"
    assert summary["recipient_target"] == "visible_recipient_value"
    assert summary["recipient_charset_source"] == "train_only_visible_recipient_text"
    assert slot["text"] == "商户甲"
    assert slot["semantic_value"] == "stale-flat-semantic-value"
    assert "recipient_visible_text" not in slot
    assert "recipient_value" not in slot


@pytest.mark.parametrize(
    ("key", "value", "message"),
    (
        ("recipient_visible_text", "商户甲", "recipient_visible_text must equal the CTC target"),
        ("recipient_value", "错误商户", "recipient_value must match"),
        ("semantic_value", "错误商户", "semantic_value must match"),
    ),
)
def test_v10_loader_rejects_recipient_payload_that_breaks_visible_row_contract(
    tmp_path: Path,
    key: str,
    value: str,
    message: str,
) -> None:
    """A legacy/value-only payload cannot silently reintroduce v9 supervision."""
    source_manifest = _write_source_manifest(tmp_path)
    output = tmp_path / "v10"
    build_unified_dataset(records_path=source_manifest, output_dir=output, architecture="v10")
    records_path = output / "unified_fields.jsonl"
    rows = _rows(output)
    recipient_slot = dict(next(row for row in rows if row["split"] == "train")["slots"])["recipient_field"]
    assert isinstance(recipient_slot, dict)
    recipient_slot[key] = value
    records_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=message):
        load_records(
            records_path,
            dataset_root=source_manifest.parent,
            config=_tiny_v10_config(),
        )


def _tiny_v10_config() -> UnifiedReaderConfig:
    return UnifiedReaderConfig(
        architecture_version=10,
        image_height=32,
        image_width=64,
        base_channels=8,
        numeric_hidden_size=16,
        payment_hidden_size=16,
        pooled_width=2,
    )


def test_v10_recipient_value_is_extracted_after_full_visible_line_decode() -> None:
    """v10's CTC target is the line, but business comparison uses its value."""
    config = _tiny_v10_config()
    record = {
        "slots": {
            "recipient_field": {
                "text": "收款方 商户甲",
                "recipient_visible_text": "收款方 商户甲",
                "recipient_value": "商户甲",
                "semantic_value": "商户甲",
            }
        }
    }
    assert _recipient_expected_value(record, config=config) == "商户甲"
    assert _recipient_candidate_value("收款方 商户甲", config=config) == "商户甲"
    # A model occasionally omitting the visual title must still leave an
    # auditable merchant candidate, rather than returning an empty value.
    assert _recipient_candidate_value("商户甲", config=config) == "商户甲"


def test_v10_model_uses_the_five_slot_onnx_protocol() -> None:
    """The new supervision is a new kind, not a second runtime model."""
    torch = pytest.importorskip("torch")
    config = _tiny_v10_config()
    model = build_unified_reader(
        payment_vocab_size=6,
        payment_bank_prefix_vocab_size=2,
        recipient_vocab_size=7,
        config=config,
    )
    outputs = model(torch.zeros((2, len(V10_SLOT_ORDER), 1, 32, 64), dtype=torch.float32))

    assert len(outputs) == len(V10_ONNX_OUTPUT_NAMES) == 15
    assert list(outputs[-1].shape) == [16, 2, 7]
    assert _checkpoint_config(
        {"kind": READER_KIND_V10, "config": asdict(config)}
    ).architecture_version == 10
    assert _delivery_text(
        architecture_version=10,
        field="recipient_field",
        candidate_text="商户甲",
        ctc_text="收款方 商户甲",
        structured_text=None,
    ) == "review"


def test_v10_train_export_load_and_evaluate_when_onnx_is_available(tmp_path: Path) -> None:
    """Exercise v10's full-line CTC contract across its deployment boundary."""
    torch = pytest.importorskip("torch")
    onnx = pytest.importorskip("onnx")
    ort = pytest.importorskip("onnxruntime")
    flat_manifest = _write_complete_source_manifest(tmp_path)
    unified_dir = tmp_path / "unified-v10"
    build_unified_dataset(
        records_path=flat_manifest,
        output_dir=unified_dir,
        architecture="v10",
    )
    records_path = unified_dir / "unified_fields.jsonl"
    checkpoint = train_unified_reader(
        records_path=records_path,
        dataset_root=flat_manifest.parent,
        output_dir=tmp_path / "run-v10",
        config=_tiny_v10_config(),
        device="cpu",
        epochs=1,
        batch_size=1,
        payment_bank_prefix_min_support=1,
        recipient_loss_weight=3.0,
    )

    model_path, labels_path, contract_path = export_unified_onnx(
        checkpoint_path=checkpoint,
        output_path=tmp_path / "reader-v10.onnx",
    )
    onnx.checker.check_model(onnx.load_model(model_path))
    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    assert [item.name for item in session.get_outputs()] == list(V10_ONNX_OUTPUT_NAMES)
    labels = json.loads(labels_path.read_text(encoding="utf-8"))
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    assert labels["recipient_charset_source"] == "train_only_visible_recipient_line"
    assert labels["recipient_target"] == "visible_recipient_line_then_extract_value"
    assert contract["outputs"]["recipient_logits"]["target"] == labels["recipient_target"]

    summary, failures = evaluate_unified_onnx(
        model_path=model_path,
        records_path=records_path,
        dataset_root=flat_manifest.parent,
        output_dir=tmp_path / "eval-v10",
        split="test",
        device="cpu",
    )
    assert failures == []
    assert summary["by_field"]["recipient_field"]["records"] == 1
    comparisons = [
        json.loads(line)
        for line in (tmp_path / "eval-v10" / "comparisons.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    recipient = next(row for row in comparisons if row["field"] == "recipient_field")
    assert recipient["reference_text"] == "商户丁"
    assert recipient["ctc_reference_text"] == "收款方 商户丁"
    assert recipient["ctc_candidate_text"] is not None
    assert recipient["runtime_policy"].startswith("review_only")
