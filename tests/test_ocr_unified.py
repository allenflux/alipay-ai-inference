from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

torch = pytest.importorskip("torch")

from transfer_receipt_ai.ocr_unified import (
    KIND_V3,
    UnifiedReaderConfig,
    _checkpoint_config,
    _format_exact_match,
    build_unified_reader,
    decode_ctc_logits,
    export_unified_onnx,
    train_unified_reader,
)


def _write_image(path: Path, shade: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pixels = np.full((20, 72, 3), 255, dtype=np.uint8)
    pixels[:, 8:24] = shade
    Image.fromarray(pixels).save(path)


def _receipt(index: int, split: str, status: str) -> dict[str, object]:
    return {
        "schema_version": 1,
        "id": f"receipt-{index}",
        "group_id": f"group-{index}",
        "split": split,
        "slot_order": ["amount", "time", "transfer_status", "payment_method_field"],
        "slots": {
            "amount": {"image": f"images/{index}-amount.png", "text": "1.00", "semantic_value": "¥1.00"},
            "time": {"image": f"images/{index}-time.png", "text": "12:06", "semantic_value": "12:06"},
            "transfer_status": {"image": f"images/{index}-status.png", "class_name": status},
            "payment_method_field": {
                "image": f"images/{index}-payment.png",
                "text": "建行卡1",
                "semantic_value": "bank_card",
            },
        },
    }


def _write_dataset(tmp_path: Path) -> Path:
    dataset = tmp_path / "dataset"
    records = [
        _receipt(1, "train", "success"),
        _receipt(2, "train", "pending"),
        _receipt(3, "val", "success"),
        _receipt(4, "val", "pending"),
        _receipt(5, "test", "failed"),
    ]
    for receipt_index, record in enumerate(records):
        for slot_index, slot in enumerate(dict(record["slots"]).values()):
            _write_image(dataset / str(dict(slot)["image"]), 20 + receipt_index * 25 + slot_index)
    records_path = dataset / "unified_fields.jsonl"
    records_path.write_text("".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records), encoding="utf-8")
    return records_path


def _tiny_config(*, architecture_version: int) -> UnifiedReaderConfig:
    return UnifiedReaderConfig(
        architecture_version=architecture_version,
        image_height=32,
        image_width=64,
        base_channels=8,
        numeric_hidden_size=16,
        payment_hidden_size=16,
        pooled_width=2,
    )


def test_unified_v4_model_emits_all_head_shapes_with_independent_numeric_heads() -> None:
    config = _tiny_config(architecture_version=4)
    model = build_unified_reader(payment_vocab_size=6, config=config)
    numeric, payment, status = model(torch.zeros((2, 4, 1, 32, 64), dtype=torch.float32))
    assert list(numeric.shape) == [16, 2, 2, 13]
    assert list(payment.shape) == [16, 2, 6]
    assert list(status.shape) == [2, 3]
    assert model.amount_sequence is not model.time_sequence
    assert model.amount_classifier is not model.time_classifier


def test_unified_v3_config_keeps_the_existing_output_protocol() -> None:
    config = _tiny_config(architecture_version=3)
    model = build_unified_reader(payment_vocab_size=6, config=config)
    numeric, payment, status = model(torch.zeros((2, 4, 1, 32, 64), dtype=torch.float32))
    assert list(numeric.shape) == [16, 2, 2, 13]
    assert list(payment.shape) == [16, 2, 6]
    assert list(status.shape) == [2, 3]
    assert hasattr(model, "numeric_sequence")
    assert not hasattr(model, "amount_sequence")


def test_legacy_v3_checkpoint_config_infers_architecture_from_kind() -> None:
    config = _checkpoint_config(
        {
            "kind": KIND_V3,
            "config": {
                "image_height": 48,
                "image_width": 384,
                "base_channels": 24,
                "numeric_hidden_size": 64,
                "payment_hidden_size": 96,
                "pooled_width": 8,
            },
        }
    )
    assert config.architecture_version == 3
    assert config.image_height == 48


def test_optional_exact_metric_formats_as_na_without_status_labels() -> None:
    assert _format_exact_match(None) == "n/a"
    assert _format_exact_match(0.5) == "50.00%"


def test_unified_ctc_decoder_collapses_repeats_and_blanks() -> None:
    logits = np.zeros((6, 1, 3), dtype=np.float32)
    for time, index in enumerate((1, 1, 0, 2, 2, 1)):
        logits[time, 0, index] = 1.0
    assert decode_ctc_logits(logits, characters=["A", "B"]) == ["ABA"]


def test_tiny_unified_training_writes_checkpoint(tmp_path: Path) -> None:
    records_path = _write_dataset(tmp_path)
    checkpoint = train_unified_reader(
        records_path=records_path,
        output_dir=tmp_path / "run",
        config=_tiny_config(architecture_version=4),
        device="cpu",
        epochs=1,
        batch_size=2,
    )
    assert checkpoint.is_file()
    assert (checkpoint.parent / "last.pt").is_file()
    assert (checkpoint.parent / "labels.json").is_file()
    summary = json.loads((checkpoint.parent / "training_summary.json").read_text(encoding="utf-8"))
    assert summary["kind"] == "receipt_unified_field_reader_v4"
    assert summary["status_head_policy"]["runtime_policy"] == "review_only"
    assert summary["status_head_policy"]["training_enabled"] is False


def test_unified_export_has_one_fixed_receipt_input_when_onnx_is_available(tmp_path: Path) -> None:
    onnx = pytest.importorskip("onnx")
    ort = pytest.importorskip("onnxruntime")
    records_path = _write_dataset(tmp_path)
    config = _tiny_config(architecture_version=4)
    checkpoint = train_unified_reader(
        records_path=records_path,
        output_dir=tmp_path / "run",
        config=config,
        device="cpu",
        epochs=1,
        batch_size=2,
    )
    model_path, labels_path, contract_path = export_unified_onnx(
        checkpoint_path=checkpoint,
        output_path=tmp_path / "reader.onnx",
    )
    onnx.checker.check_model(onnx.load_model(model_path))
    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    outputs = session.run(None, {"field_images": np.zeros((4, 1, 32, 64), dtype=np.float32)})
    assert [list(value.shape) for value in outputs] == [[16, 13], [16, 13], [16, 5], [3]]
    assert labels_path.is_file()
    assert contract_path.is_file()
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    assert contract["kind"] == "receipt_unified_field_reader_v4"
    assert contract["input"]["shape"] == [4, 1, 32, 64]
    assert contract["model"]["architecture_version"] == 4
    assert contract["status_head_policy"]["runtime_policy"] == "review_only"
