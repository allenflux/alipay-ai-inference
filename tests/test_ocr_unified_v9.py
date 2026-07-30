from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

torch = pytest.importorskip("torch")

from transfer_receipt_ai.ocr_unified import (
    KIND_V8,
    KIND_V9,
    V8_ONNX_OUTPUT_NAMES,
    V9_ONNX_OUTPUT_NAMES,
    UnifiedReaderConfig,
    _checkpoint_config,
    _delivery_text,
    _load_onnx_artifact_details,
    build_unified_reader,
    evaluate_unified_onnx,
    export_unified_onnx,
    train_unified_reader,
)
from transfer_receipt_ai.ocr_unified_dataset import (
    KIND_V9 as DATASET_KIND_V9,
    V9_SLOT_ORDER,
    build_unified_dataset,
)


def _write_image(path: Path, shade: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.full((24, 96, 3), shade, dtype=np.uint8)).save(path)


def _flat_record(
    *,
    index: int,
    field: str,
    text: str,
    semantic_value: str,
    split: str,
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "id": f"sample-{split}-{index}",
        "image": f"images/{field}/{split}-{index}.png",
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


def _write_v9_source_manifest(tmp_path: Path) -> Path:
    source = tmp_path / "pseudo"
    records: list[dict[str, object]] = []
    recipient_texts = {
        "train": "收款方 商户甲",
        "val": "收款方 商户丙",
        "test": "收款方 商户丁",
    }
    index = 0
    for split, recipient_text in recipient_texts.items():
        values = (
            ("amount", "¥100.00", "¥100.00"),
            ("time", "12:06", "12:06"),
            ("transfer_status", "转账成功", "success"),
            ("payment_method_field", "付款方式 建设银行储蓄卡(3667)", "bank_card"),
            ("recipient_field", recipient_text, recipient_text.removeprefix("收款方 ")),
        )
        for field, text, semantic_value in values:
            index += 1
            record = _flat_record(
                index=index,
                field=field,
                text=text,
                semantic_value=semantic_value,
                split=split,
            )
            _write_image(source / str(record["image"]), 10 + index)
            records.append(record)
    manifest = source / "pseudo_labels.jsonl"
    manifest.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )
    return manifest


def _tiny_v9_config() -> UnifiedReaderConfig:
    return UnifiedReaderConfig(
        architecture_version=9,
        image_height=32,
        image_width=64,
        base_channels=8,
        numeric_hidden_size=16,
        payment_hidden_size=16,
        pooled_width=2,
    )


def test_v9_dataset_appends_recipient_slot_and_freezes_charset_from_train_only(tmp_path: Path) -> None:
    """The five-slot manifest must not leak held-out merchant characters."""
    output = tmp_path / "unified-v9"
    summary = build_unified_dataset(
        records_path=_write_v9_source_manifest(tmp_path),
        output_dir=output,
        architecture="v9",
    )

    rows = [
        json.loads(line)
        for line in (output / "unified_fields.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    train = next(row for row in rows if row["split"] == "train")
    assert summary["kind"] == DATASET_KIND_V9
    assert summary["architecture"] == "v9"
    assert summary["slot_order"] == list(V9_SLOT_ORDER)
    assert train["slot_order"] == list(V9_SLOT_ORDER)
    assert train["complete"] is True
    assert train["slots"]["recipient_field"]["text"] == "商户甲"

    charset = summary["recipient_charset"]
    assert set(charset) == set("商户甲")
    assert summary["recipient_charset_source"] == "train_only_visible_recipient_text"
    assert summary["recipient_charset_sha256"] == hashlib.sha256(
        "".join(charset).encode("utf-8")
    ).hexdigest()
    assert summary["recipient_oov_by_split"]["train"] == {
        "records": 1,
        "oov_records": 0,
        "oov_characters": 0,
        "examples": [],
    }
    assert summary["recipient_oov_by_split"]["val"]["oov_records"] == 1
    assert summary["recipient_oov_by_split"]["test"]["oov_records"] == 1
    assert (output / "recipient_charset.txt").read_text(encoding="utf-8").strip() == "".join(charset)


def test_v9_model_keeps_v8_output_prefix_and_appends_recipient_ctc_head() -> None:
    """One five-slot input/session adds only the free-recipient CTC output."""
    config = _tiny_v9_config()
    model = build_unified_reader(
        payment_vocab_size=6,
        payment_bank_prefix_vocab_size=2,
        recipient_vocab_size=5,
        config=config,
    )
    outputs = model(torch.zeros((2, len(V9_SLOT_ORDER), 1, 32, 64), dtype=torch.float32))

    assert V9_ONNX_OUTPUT_NAMES[:-1] == V8_ONNX_OUTPUT_NAMES
    assert V9_ONNX_OUTPUT_NAMES[-1] == "recipient_logits"
    assert len(outputs) == len(V9_ONNX_OUTPUT_NAMES)
    assert list(outputs[0].shape) == [16, 2, 13]  # v8 compact amount CTC
    assert list(outputs[2].shape) == [16, 2, 6]  # v8 payment CTC
    assert list(outputs[-1].shape) == [16, 2, 5]  # recipient CTC, including blank

    outputs[-1].sum().backward()
    assert model.recipient_ctc_vertical_reducer.depthwise.weight.grad is not None
    assert model.recipient_ctc_sequence.weight_ih_l0.grad is not None
    assert model.recipient_classifier.weight.grad is not None

    # A five-slot artifact must refuse the legacy four-slot tensor rather than
    # silently reading the payment crop as a recipient crop.
    with pytest.raises(ValueError):
        model(torch.zeros((1, 4, 1, 32, 64), dtype=torch.float32))


def test_v9_checkpoint_identity_and_delivery_policy_are_distinct_from_v8() -> None:
    config = _checkpoint_config(
        {
            "kind": KIND_V9,
            "config": asdict(_tiny_v9_config()),
        }
    )
    assert config.architecture_version == 9
    assert KIND_V9 != KIND_V8
    assert _delivery_text(
        architecture_version=9,
        field="recipient_field",
        candidate_text="商户甲",
        ctc_text="商户甲",
        structured_text=None,
    ) == "review"


def test_v9_tiny_training_writes_a_train_only_recipient_charset(tmp_path: Path) -> None:
    """Exercise the fifth head's loss/checkpoint path without ONNX packages.

    The ONNX boundary test below is intentionally skipped on lean developer
    environments.  Keep a non-ONNX training regression here so an export
    dependency cannot hide a broken five-slot CTC loss, a missing recipient
    head, or accidental held-out charset leakage.
    """
    flat_manifest = _write_v9_source_manifest(tmp_path)
    unified_dir = tmp_path / "unified-v9"
    build_unified_dataset(
        records_path=flat_manifest,
        output_dir=unified_dir,
        architecture="v9",
    )

    checkpoint = train_unified_reader(
        records_path=unified_dir / "unified_fields.jsonl",
        dataset_root=flat_manifest.parent,
        output_dir=tmp_path / "run-v9",
        config=_tiny_v9_config(),
        device="cpu",
        epochs=1,
        batch_size=1,
        payment_bank_prefix_min_support=1,
    )

    assert checkpoint.is_file()
    labels = json.loads((checkpoint.parent / "labels.json").read_text(encoding="utf-8"))
    summary = json.loads((checkpoint.parent / "training_summary.json").read_text(encoding="utf-8"))
    assert summary["kind"] == KIND_V9
    assert labels["recipient_characters"] == sorted("商户甲")
    assert labels["recipient_oov_by_split"]["train"] == {
        "records": 1,
        "oov_records": 0,
    }
    assert labels["recipient_oov_by_split"]["val"]["oov_records"] == 1
    assert set(summary["records"][0]["val_candidate_text_by_field"]) == {
        "amount",
        "time",
        "payment_method_field",
        "recipient_field",
    }


def test_v9_exports_loads_and_evaluates_a_five_slot_onnx_when_available(tmp_path: Path) -> None:
    """Exercise the actual v9 delivery boundary when ONNX dependencies exist.

    The compatibility promise is not merely a fifth PyTorch output: a v9
    artifact must expose a five-slot ONNX input, the frozen v8 output prefix,
    and an auditable recipient CTC sidecar.  Keep this test separate from the
    model-forward test so environments that intentionally omit ONNX can still
    validate the v9 data and training contracts above.
    """
    onnx = pytest.importorskip("onnx")
    ort = pytest.importorskip("onnxruntime")
    flat_manifest = _write_v9_source_manifest(tmp_path)
    unified_dir = tmp_path / "unified-v9"
    build_unified_dataset(
        records_path=flat_manifest,
        output_dir=unified_dir,
        architecture="v9",
    )
    records_path = unified_dir / "unified_fields.jsonl"
    checkpoint = train_unified_reader(
        records_path=records_path,
        dataset_root=flat_manifest.parent,
        output_dir=tmp_path / "run-v9",
        config=_tiny_v9_config(),
        device="cpu",
        epochs=1,
        batch_size=1,
        payment_bank_prefix_min_support=1,
    )

    model_path, labels_path, contract_path = export_unified_onnx(
        checkpoint_path=checkpoint,
        output_path=tmp_path / "reader-v9.onnx",
    )
    onnx.checker.check_model(onnx.load_model(model_path))
    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    assert [item.name for item in session.get_outputs()] == list(V9_ONNX_OUTPUT_NAMES)
    outputs = session.run(
        None,
        {"field_images": np.zeros((len(V9_SLOT_ORDER), 1, 32, 64), dtype=np.float32)},
    )
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    assert contract["kind"] == KIND_V9
    assert contract["slot_order"] == list(V9_SLOT_ORDER)
    assert contract["input"]["shape"] == [5, 1, 32, 64]
    assert [list(value.shape) for value in outputs] == [
        contract["outputs"][name]["shape"] for name in V9_ONNX_OUTPUT_NAMES
    ]
    labels = json.loads(labels_path.read_text(encoding="utf-8"))
    assert labels["recipient_characters"] == sorted(labels["recipient_characters"])
    assert labels["recipient_blank_index"] == 0

    loaded_config, _, recipient_characters, loaded_contract = _load_onnx_artifact_details(model_path)
    assert loaded_config.architecture_version == 9
    assert recipient_characters == labels["recipient_characters"]
    assert loaded_contract["outputs"]["recipient_logits"]["runtime_policy"] == "review_only"

    summary, failures = evaluate_unified_onnx(
        model_path=model_path,
        records_path=records_path,
        dataset_root=flat_manifest.parent,
        output_dir=tmp_path / "eval-v9",
        split="test",
        device="cpu",
    )
    assert failures == []
    assert summary["by_field"]["recipient_field"]["records"] == 1
    # ``商户丁`` occurs only in the held-out test split.  It must remain an
    # explicit OOV diagnostic rather than being leaked into the train-only
    # deployed recipient alphabet just to improve teacher-parity scores.
    assert summary["by_field"]["recipient_field"]["oov_reference_rate"] == 1.0
    comparisons = [
        json.loads(line)
        for line in (tmp_path / "eval-v9" / "comparisons.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert {row["field"] for row in comparisons} == set(V9_SLOT_ORDER)
    recipient = next(row for row in comparisons if row["field"] == "recipient_field")
    assert recipient["ctc_candidate_text"] is not None
    assert recipient["runtime_policy"].startswith("review_only")
    assert recipient["delivery_text"] == "review"
