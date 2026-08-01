"""Focused tests for the v11 anchored-recipient single-ONNX protocol."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import pytest

from transfer_receipt_ai.ocr import parse_anchored_recipient_row
from transfer_receipt_ai.ocr_unified import (
    KIND_V11 as READER_KIND_V11,
    V11_ONNX_OUTPUT_NAMES,
    UnifiedReaderConfig,
    _checkpoint_config,
    _input_tensor,
    build_unified_reader,
    evaluate_unified_onnx,
    export_unified_onnx,
    load_records,
    train_unified_reader,
)
from transfer_receipt_ai.ocr_unified_dataset import (
    KIND_V11,
    V11_SLOT_ORDER,
    build_unified_dataset,
    slot_order_for_architecture,
)


# A valid 1x1 opaque PNG.  The data-contract tests intentionally do not need
# a heavyweight image stack; the train/export test uses the same real file.
_TINY_PNG = bytes.fromhex(
    "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c489"
    "0000000d49444154789c6360f8cfc00000030101874f5dc30000000049454e44ae426082"
)


def _source_row(
    *,
    index: int,
    split: str,
    field: str,
    text: str,
    semantic_value: str,
    group: str | None = None,
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "id": f"{field}-{split}-{index}",
        "image": f"images/{field}/{split}-{index}.png",
        "field": field,
        "text": text,
        "paddle_text": text,
        "semantic_value": semantic_value,
        "paddle_confidence": 0.99,
        "detector_score": 0.95,
        "result_json": f"D:/teacher/{split}.json",
        "source": f"D:/source/{split}.png",
        "group_id": group or f"receipt:{split}",
        "split": split,
        "label_source": "paddle_pseudo",
    }


def _write_complete_source_manifest(tmp_path: Path, *, include_bad_recipient: bool = False) -> Path:
    source = tmp_path / "teacher-labels"
    records: list[dict[str, object]] = []
    recipients = {"train": "商户甲", "val": "商户丙", "test": "商户丁"}
    index = 0
    for split, recipient_value in recipients.items():
        values = (
            ("amount", "¥100.00", "¥100.00"),
            ("time", "12:06", "12:06"),
            ("transfer_status", "转账成功", "success"),
            ("payment_method_field", "付款方式 建设银行储蓄卡(3667)", "bank_card"),
            ("recipient_field", f"收款方 {recipient_value}", recipient_value),
        )
        for field, text, semantic_value in values:
            index += 1
            row = _source_row(
                index=index,
                split=split,
                field=field,
                text=text,
                semantic_value=semantic_value,
            )
            image = source / str(row["image"])
            image.parent.mkdir(parents=True, exist_ok=True)
            image.write_bytes(_TINY_PNG)
            records.append(row)
    if include_bad_recipient:
        index += 1
        row = _source_row(
            index=index,
            split="train",
            field="recipient_field",
            text="收款方 商户污染 付款方式 建设银行储蓄卡(3667)",
            semantic_value="商户污染",
            group="receipt:polluted",
        )
        image = source / str(row["image"])
        image.parent.mkdir(parents=True, exist_ok=True)
        image.write_bytes(_TINY_PNG)
        records.append(row)
    manifest = source / "pseudo_labels.jsonl"
    manifest.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )
    return manifest


def _rows(output: Path) -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in (output / "unified_fields.jsonl").read_text(encoding="utf-8").splitlines()
    ]


def _tiny_v11_config(*, recipient_hidden_size: int | None = None) -> UnifiedReaderConfig:
    return UnifiedReaderConfig(
        architecture_version=11,
        image_height=32,
        image_width=64,
        base_channels=8,
        numeric_hidden_size=16,
        payment_hidden_size=16,
        recipient_hidden_size=recipient_hidden_size,
        recipient_value_left_trim=0.30,
        pooled_width=2,
    )


def test_v11_recipient_manifest_filters_pollution_and_trains_only_on_value(tmp_path: Path) -> None:
    manifest = _write_complete_source_manifest(tmp_path, include_bad_recipient=True)
    output = tmp_path / "v11"

    summary = build_unified_dataset(records_path=manifest, output_dir=output, architecture="v11")

    train = next(
        row
        for row in _rows(output)
        if row["split"] == "train" and "recipient_field" in row["slots"]
    )
    slot = dict(train["slots"])["recipient_field"]
    assert isinstance(slot, dict)
    assert summary["kind"] == KIND_V11
    assert summary["architecture"] == "v11"
    assert summary["slot_order"] == list(V11_SLOT_ORDER)
    assert slot_order_for_architecture("v11") == V11_SLOT_ORDER
    assert summary["recipient_target"] == "anchored_recipient_value_with_value_view_crop"
    assert summary["recipient_charset_source"] == "train_only_anchored_recipient_value"
    assert slot["text"] == "商户甲"
    assert slot["recipient_visible_text"] == "收款方 商户甲"
    assert slot["recipient_value"] == "商户甲"
    assert slot["recipient_label"] == "收款方"
    assert slot["recipient_quality_policy"] == "anchored_value_right_crop_v1"
    assert "收款方" not in summary["recipient_charset"]
    assert summary["recipient_quality_audit"]["quality_rejected"] == 1
    audit = [
        json.loads(line)
        for line in (output / "recipient_quality_audit.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    polluted = next(item for item in audit if "商户污染" in str(item["source_text"]))
    assert polluted["quality_decision"] == "rejected"
    assert str(polluted["quality_reason"]).startswith("context_or_currency_pollution:")


@pytest.mark.parametrize(
    ("key", "value", "message"),
    (
        ("recipient_visible_text", "商户甲", "must begin with an anchored label"),
        ("recipient_label", "收款人", "recipient label does not match"),
        ("recipient_quality_policy", "legacy", "quality policy is unsupported"),
    ),
)
def test_v11_loader_rejects_broken_value_view_contract(
    tmp_path: Path,
    key: str,
    value: str,
    message: str,
) -> None:
    source_manifest = _write_complete_source_manifest(tmp_path)
    output = tmp_path / "v11"
    build_unified_dataset(records_path=source_manifest, output_dir=output, architecture="v11")
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
        load_records(records_path, dataset_root=source_manifest.parent, config=_tiny_v11_config())


def test_v11_config_freezes_recipient_branch_and_value_crop() -> None:
    torch = pytest.importorskip("torch")
    config = _tiny_v11_config(recipient_hidden_size=24)
    model = build_unified_reader(
        payment_vocab_size=6,
        payment_bank_prefix_vocab_size=2,
        recipient_vocab_size=7,
        config=config,
    )
    outputs = model(torch.zeros((2, len(V11_SLOT_ORDER), 1, 32, 64), dtype=torch.float32))

    assert len(outputs) == len(V11_ONNX_OUTPUT_NAMES) == 15
    assert model.recipient_ctc_sequence.hidden_size == 24
    assert list(outputs[-1].shape) == [16, 2, 7]
    assert _checkpoint_config({"kind": READER_KIND_V11, "config": asdict(config)}).architecture_version == 11
    with pytest.raises(ValueError, match="recipient_hidden_size is supported only by architecture v11"):
        UnifiedReaderConfig(architecture_version=10, recipient_hidden_size=24).validate()
    with pytest.raises(ValueError, match="recipient_value_left_trim is supported only by architecture v11"):
        UnifiedReaderConfig(architecture_version=10, recipient_value_left_trim=0.25).validate()


def test_v11_input_tensor_trims_the_static_recipient_label_before_centered_resize(tmp_path: Path) -> None:
    Image = pytest.importorskip("PIL.Image")
    config = _tiny_v11_config()
    image_path = tmp_path / "recipient.png"
    image = Image.new("RGB", (100, 20), "white")
    for x in range(30):
        for y in range(20):
            image.putpixel((x, y), (0, 0, 0))
    image.save(image_path)
    record = {
        "slots": {
            "recipient_field": {
                "image_path": image_path,
                "text": "商户甲",
            }
        }
    }

    field_images = _input_tensor(record, config=config)
    recipient_index = V11_SLOT_ORDER.index("recipient_field")
    # The black left 30% represents the field title.  With v11's frozen 30%
    # trim it does not leak into the value-reader input channel.
    assert float(field_images[recipient_index].min()) > 0.99


def test_v11_train_export_load_and_evaluate_when_onnx_is_available(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    onnx = pytest.importorskip("onnx")
    ort = pytest.importorskip("onnxruntime")
    flat_manifest = _write_complete_source_manifest(tmp_path)
    unified_dir = tmp_path / "unified-v11"
    build_unified_dataset(records_path=flat_manifest, output_dir=unified_dir, architecture="v11")
    records_path = unified_dir / "unified_fields.jsonl"
    config = _tiny_v11_config(recipient_hidden_size=24)
    checkpoint = train_unified_reader(
        records_path=records_path,
        dataset_root=flat_manifest.parent,
        output_dir=tmp_path / "run-v11",
        config=config,
        device="cpu",
        epochs=1,
        batch_size=1,
        payment_bank_prefix_min_support=1,
        recipient_loss_weight=3.0,
        recipient_sampling_weight=2.0,
    )

    model_path, labels_path, contract_path = export_unified_onnx(
        checkpoint_path=checkpoint,
        output_path=tmp_path / "reader-v11.onnx",
    )
    onnx.checker.check_model(onnx.load_model(model_path))
    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    assert [item.name for item in session.get_outputs()] == list(V11_ONNX_OUTPUT_NAMES)
    labels = json.loads(labels_path.read_text(encoding="utf-8"))
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    assert labels["recipient_input_preprocess"] == "left_trim_then_centered_aspect_resize"
    assert labels["recipient_value_left_trim"] == pytest.approx(0.30)
    assert labels["recipient_hidden_size"] == 24
    assert labels["recipient_sampling_policy"]["mode"] == "weighted_receipt_sampler_v1"
    assert contract["recipient_sampling_policy"] == labels["recipient_sampling_policy"]
    recipient_output = contract["outputs"]["recipient_logits"]
    assert recipient_output["left_trim_fraction"] == pytest.approx(0.30)
    assert recipient_output["horizontal_alignment"] == "center"

    summary, failures = evaluate_unified_onnx(
        model_path=model_path,
        records_path=records_path,
        dataset_root=flat_manifest.parent,
        output_dir=tmp_path / "eval-v11",
        split="test",
        device="cpu",
    )
    assert failures == []
    assert summary["by_field"]["recipient_field"]["records"] == 1


def test_v11_rejects_nonfinite_optional_geometry_gate(tmp_path: Path) -> None:
    manifest = _write_complete_source_manifest(tmp_path)
    with pytest.raises(ValueError, match="recipient_min_crop_aspect"):
        build_unified_dataset(
            records_path=manifest,
            output_dir=tmp_path / "bad-v11",
            architecture="v11",
            recipient_min_crop_aspect=float("nan"),
        )


def test_anchored_recipient_parser_is_not_the_legacy_loose_parser() -> None:
    assert parse_anchored_recipient_row("收款方  商户甲") == ("收款方", "商户甲")
    assert parse_anchored_recipient_row("商户甲 收款方") is None
