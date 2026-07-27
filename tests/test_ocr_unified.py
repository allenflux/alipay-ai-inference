from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

torch = pytest.importorskip("torch")

from transfer_receipt_ai.ocr_unified import (
    KIND_V3,
    KIND_V5,
    KIND_V6,
    PAYMENT_BANK_OTHER_CLASS,
    V6_AMOUNT_CHARACTERS,
    V6_ONNX_OUTPUT_NAMES,
    V6_TIME_CHARACTERS,
    V5_ONNX_OUTPUT_NAMES,
    UnifiedReaderConfig,
    _checkpoint_config,
    _delivery_text,
    _format_exact_match,
    _load_onnx_artifacts,
    _amount_v6_structured_target,
    _structured_amount_predictions,
    _structured_amount_v6_predictions,
    _structured_payment_predictions,
    _structured_payment_v6_predictions,
    _structured_time_predictions,
    _structured_time_v6_predictions,
    _time_v6_structured_target,
    build_unified_reader,
    decode_ctc_logits,
    evaluate_unified_onnx,
    export_unified_onnx,
    preprocess_image,
    train_unified_reader,
)
from transfer_receipt_ai.ocr_unified_targets import (
    parse_amount_aux_target,
    parse_amount_display_target,
    parse_payment_bank_prefix_target,
    parse_payment_card_tail_target,
    parse_time_aux_target,
    parse_time_display_target,
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


def _structured_receipt(index: int, split: str, status: str, *, card_tail: bool = True) -> dict[str, object]:
    """A minimal v5 row whose strict auxiliary targets are all present."""
    record = _receipt(index, split, status)
    slots = dict(record["slots"])
    amount = dict(slots["amount"])
    amount["text"] = "199.00"
    amount["semantic_value"] = "¥199.00"
    amount["amount_aux"] = parse_amount_aux_target("199.00")
    time = dict(slots["time"])
    time["text"] = "1:44"
    time["semantic_value"] = "1:44"
    time["time_aux"] = parse_time_aux_target("1:44")
    payment = dict(slots["payment_method_field"])
    if card_tail:
        payment["text"] = "建设银行储蓄卡（3667）"
        payment["semantic_value"] = "bank_card"
        payment["payment_card_tail"] = parse_payment_card_tail_target(payment["text"])
    else:
        payment["text"] = "余额"
        payment["semantic_value"] = "balance"
    slots.update(
        {
            "amount": amount,
            "time": time,
            "payment_method_field": payment,
        }
    )
    record["slots"] = slots
    return record


def _write_v5_dataset(tmp_path: Path) -> Path:
    dataset = tmp_path / "dataset-v5"
    records = [
        _structured_receipt(11, "train", "success"),
        _structured_receipt(12, "train", "pending", card_tail=False),
        _structured_receipt(13, "val", "success"),
        _structured_receipt(14, "val", "pending", card_tail=False),
        _structured_receipt(15, "test", "failed"),
    ]
    for receipt_index, record in enumerate(records):
        for slot_index, slot in enumerate(dict(record["slots"]).values()):
            _write_image(dataset / str(dict(slot)["image"]), 20 + receipt_index * 25 + slot_index)
    records_path = dataset / "unified_fields.jsonl"
    records_path.write_text("".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records), encoding="utf-8")
    return records_path


def _v6_structured_receipt(index: int, split: str, status: str) -> dict[str, object]:
    """A minimal v6 row with exact visible-format and bank verifier targets."""
    record = _structured_receipt(index, split, status)
    slots = dict(record["slots"])

    amount = dict(slots["amount"])
    amount["visible_text"] = "¥199.00"
    amount["amount_display"] = parse_amount_display_target("¥199.00")

    time = dict(slots["time"])
    time["visible_text"] = "1:44"
    time["time_display"] = parse_time_display_target("1:44")

    payment = dict(slots["payment_method_field"])
    payment["payment_bank_prefix"] = parse_payment_bank_prefix_target(str(payment["text"]))

    slots.update(
        {
            "amount": amount,
            "time": time,
            "payment_method_field": payment,
        }
    )
    record["slots"] = slots
    return record


def _write_v6_dataset(tmp_path: Path) -> Path:
    dataset = tmp_path / "dataset-v6"
    records = [
        _v6_structured_receipt(21, "train", "success"),
        _v6_structured_receipt(22, "train", "pending"),
        _v6_structured_receipt(23, "val", "success"),
        _v6_structured_receipt(24, "val", "pending"),
        _v6_structured_receipt(25, "test", "failed"),
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


def test_unified_v5_model_keeps_one_input_and_emits_structured_heads() -> None:
    config = _tiny_config(architecture_version=5)
    model = build_unified_reader(payment_vocab_size=6, config=config)
    outputs = model(torch.zeros((2, 4, 1, 32, 64), dtype=torch.float32))

    assert len(outputs) == 11
    (
        numeric,
        payment,
        status,
        amount_length,
        amount_digits,
        time_digits,
        time_hour_width,
        payment_prefix,
        payment_tail,
        payment_structure,
        payment_parentheses,
    ) = outputs
    assert list(numeric.shape) == [16, 2, 2, 13]
    assert list(payment.shape) == [16, 2, 6]
    assert list(status.shape) == [2, 3]
    assert list(amount_length.shape) == [2, 7]
    assert list(amount_digits.shape) == [2, 9, 10]
    assert list(time_digits.shape) == [2, 4, 10]
    assert list(time_hour_width.shape) == [2, 2]
    assert list(payment_prefix.shape) == [16, 2, 6]
    assert list(payment_tail.shape) == [2, 4, 10]
    assert list(payment_structure.shape) == [2, 2]
    assert list(payment_parentheses.shape) == [2, 2]
    assert hasattr(model, "amount_vertical_reducer")
    assert hasattr(model, "payment_prefix_sequence")


def test_unified_v6_model_keeps_one_input_but_separates_visible_ctc_and_verifier_paths() -> None:
    config = _tiny_config(architecture_version=6)
    model = build_unified_reader(
        payment_vocab_size=6,
        payment_bank_prefix_vocab_size=2,
        config=config,
    )
    outputs = model(torch.zeros((2, 4, 1, 32, 64), dtype=torch.float32))

    assert len(outputs) == len(V6_ONNX_OUTPUT_NAMES)
    (
        amount,
        time,
        payment,
        status,
        amount_sign,
        amount_length,
        amount_digits,
        time_format,
        time_digits,
        payment_prefix,
        payment_bank,
        payment_tail,
        payment_structure,
        payment_parentheses,
    ) = outputs
    assert list(amount.shape) == [16, 2, len(V6_AMOUNT_CHARACTERS) + 1]
    assert list(time.shape) == [16, 2, len(V6_TIME_CHARACTERS) + 1]
    assert list(payment.shape) == [16, 2, 6]
    assert list(status.shape) == [2, 3]
    assert list(amount_sign.shape) == [2, 2]
    assert list(amount_length.shape) == [2, 7]
    assert list(amount_digits.shape) == [2, 9, 10]
    assert list(time_format.shape) == [2, 6]
    assert list(time_digits.shape) == [2, 14, 10]
    assert list(payment_prefix.shape) == [16, 2, 6]
    assert list(payment_bank.shape) == [2, 2]
    assert list(payment_tail.shape) == [2, 4, 10]
    assert list(payment_structure.shape) == [2, 2]
    assert list(payment_parentheses.shape) == [2, 2]
    assert model.amount_ctc_sequence is not model.amount_verifier_sequence
    assert model.time_ctc_sequence is not model.time_verifier_sequence
    assert model.payment_ctc_sequence is not model.payment_prefix_sequence


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


def test_v5_structured_decoders_keep_valid_forms_and_reject_invalid_financial_forms() -> None:
    amount_length = np.full((2, 7), -6.0, dtype=np.float32)
    amount_length[:, 1] = 6.0  # two integer digits
    amount_digits = np.full((2, 9, 10), -6.0, dtype=np.float32)
    for row, values in enumerate(((1, 9, 0, 0), (0, 1, 0, 0))):
        for column, value in zip(range(5, 9), values):
            amount_digits[row, column, value] = 6.0
    amount_predictions = _structured_amount_predictions(amount_length, amount_digits)
    assert [text for text, _ in amount_predictions] == ["19.00", None]
    assert amount_predictions[0][1] > 0.99

    time_digits = np.full((2, 4, 10), -6.0, dtype=np.float32)
    for row in range(2):
        for column, value in enumerate((0, 1, 4, 4)):
            time_digits[row, column, value] = 6.0
    hour_width = np.full((2, 2), -6.0, dtype=np.float32)
    hour_width[0, 0] = 6.0
    hour_width[1, 1] = 6.0
    time_predictions = _structured_time_predictions(time_digits, hour_width)
    assert [text for text, _ in time_predictions] == ["1:44", "01:44"]
    assert all(confidence > 0.99 for _, confidence in time_predictions)

    characters = list("建设银行储蓄卡")
    prefix_logits = np.full((15, 1, len(characters) + 1), -6.0, dtype=np.float32)
    prefix_logits[:, 0, 0] = 6.0
    for offset, character in enumerate(characters):
        prefix_logits[offset * 2, 0, characters.index(character) + 1] = 12.0
    tail_logits = np.full((1, 4, 10), -6.0, dtype=np.float32)
    for column, value in enumerate((3, 6, 6, 7)):
        tail_logits[0, column, value] = 6.0
    card_structure = np.asarray([[-6.0, 6.0]], dtype=np.float32)
    fullwidth_parentheses = np.asarray([[-6.0, 6.0]], dtype=np.float32)
    payment_predictions = _structured_payment_predictions(
        prefix_logits,
        tail_logits,
        card_structure,
        fullwidth_parentheses,
        payment_characters=characters,
    )
    assert [text for text, _ in payment_predictions] == ["建设银行储蓄卡（3667）"]
    assert payment_predictions[0][1] > 0.99
    assert _structured_payment_predictions(
        prefix_logits,
        tail_logits,
        np.asarray([[6.0, -6.0]], dtype=np.float32),
        fullwidth_parentheses,
        payment_characters=characters,
    ) == [(None, 0.0)]


def test_v6_verifier_decoders_require_valid_signed_time_and_known_bank_forms() -> None:
    amount_length = np.full((1, 7), -6.0, dtype=np.float32)
    amount_length[0, 1] = 6.0  # two integer digits
    amount_digits = np.full((1, 9, 10), -6.0, dtype=np.float32)
    for column, value in zip(range(5, 9), (1, 9, 0, 0)):
        amount_digits[0, column, value] = 6.0
    amount_sign = np.asarray([[-6.0, 6.0]], dtype=np.float32)
    assert _structured_amount_v6_predictions(amount_sign, amount_length, amount_digits)[0][0] == "-19.00"

    time_format = np.full((1, 6), -6.0, dtype=np.float32)
    time_format[0, 4] = 6.0  # date_ymd_hh_mm
    time_digits = np.full((1, 14, 10), -6.0, dtype=np.float32)
    for column, value in enumerate((2, 0, 2, 6, 0, 7, 2, 7, 1, 2, 3, 4, 0, 0)):
        time_digits[0, column, value] = 6.0
    assert _structured_time_v6_predictions(time_format, time_digits)[0][0] == "2026-07-27 12:34"

    bank_classes = [PAYMENT_BANK_OTHER_CLASS, "建设银行储蓄卡"]
    bank_logits = np.asarray([[-6.0, 6.0]], dtype=np.float32)
    tail_logits = np.full((1, 4, 10), -6.0, dtype=np.float32)
    for column, value in enumerate((3, 6, 6, 7)):
        tail_logits[0, column, value] = 6.0
    structure = np.asarray([[-6.0, 6.0]], dtype=np.float32)
    parentheses = np.asarray([[-6.0, 6.0]], dtype=np.float32)
    prediction = _structured_payment_v6_predictions(
        bank_logits,
        tail_logits,
        structure,
        parentheses,
        payment_bank_prefix_classes=bank_classes,
    )
    assert prediction[0][0] == "建设银行储蓄卡（3667）"
    assert _structured_payment_v6_predictions(
        np.asarray([[6.0, -6.0]], dtype=np.float32),
        tail_logits,
        structure,
        parentheses,
        payment_bank_prefix_classes=bank_classes,
    ) == [(None, 0.0)]


def test_v5_right_aligned_preprocess_has_a_distinct_fixed_position_from_legacy_center(tmp_path: Path) -> None:
    image_path = tmp_path / "narrow.png"
    Image.fromarray(np.zeros((20, 20), dtype=np.uint8)).save(image_path)
    config = _tiny_config(architecture_version=5)
    centered = preprocess_image(image_path, config=config, horizontal_alignment="center")[0]
    right = preprocess_image(image_path, config=config, horizontal_alignment="right")[0]
    assert np.where(centered < 0.5)[1].min() == 16
    assert np.where(right < 0.5)[1].min() == 32


def test_v5_refuses_a_legacy_manifest_without_strict_auxiliary_targets(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="structured train/val labels"):
        train_unified_reader(
            records_path=_write_dataset(tmp_path),
            output_dir=tmp_path / "run-v5-without-aux",
            config=_tiny_config(architecture_version=5),
            device="cpu",
            epochs=1,
            batch_size=2,
        )


def test_v6_refuses_a_legacy_manifest_without_visible_format_and_bank_targets(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="strict visible amount/time and exact payment bank-prefix"):
        train_unified_reader(
            records_path=_write_v5_dataset(tmp_path),
            output_dir=tmp_path / "run-v6-without-visible-aux",
            config=_tiny_config(architecture_version=6),
            device="cpu",
            epochs=1,
            batch_size=2,
            payment_bank_prefix_min_support=1,
        )


def test_v6_verifier_targets_reparse_visible_text_before_trusting_auxiliary_json() -> None:
    """A stale auxiliary object must not contradict the v6 CTC label."""
    amount_record = _v6_structured_receipt(31, "train", "success")
    amount_slot = dict(dict(amount_record["slots"])["amount"])
    amount_aux = dict(amount_slot["amount_display"])
    amount_aux["integer_digits"] = "999"
    amount_slot["amount_display"] = amount_aux
    amount_slots = dict(amount_record["slots"])
    amount_slots["amount"] = amount_slot
    amount_record["slots"] = amount_slots
    assert _amount_v6_structured_target(amount_record) is None

    time_record = _v6_structured_receipt(32, "train", "success")
    time_slot = dict(dict(time_record["slots"])["time"])
    time_aux = dict(time_slot["time_display"])
    time_aux["format_name"] = "clock_hh_mm"
    time_slot["time_display"] = time_aux
    time_slots = dict(time_record["slots"])
    time_slots["time"] = time_slot
    time_record["slots"] = time_slots
    assert _time_v6_structured_target(time_record) is None


def test_v5_delivery_is_review_only_until_independent_calibration_exists() -> None:
    assert _delivery_text(
        architecture_version=5,
        field="amount",
        candidate_text="99.99",
        ctc_text="99.98",
        structured_text="99.99",
    ) == "review"
    assert _delivery_text(
        architecture_version=5,
        field="payment_method_field",
        candidate_text="余额",
        ctc_text="余额",
        structured_text=None,
    ) == "review"
    assert _delivery_text(
        architecture_version=5,
        field="time",
        candidate_text="1:44",
        ctc_text="1:44",
        structured_text="1:44",
    ) == "review"
    assert _delivery_text(
        architecture_version=6,
        field="amount",
        candidate_text="¥99.99",
        ctc_text="¥99.99",
        structured_text="99.99",
    ) == "review"
    assert _delivery_text(
        architecture_version=4,
        field="amount",
        candidate_text="99.99",
        ctc_text="99.99",
        structured_text=None,
    ) == "99.99"


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


def test_tiny_v6_training_writes_train_only_bank_classes_and_visible_charsets(tmp_path: Path) -> None:
    checkpoint = train_unified_reader(
        records_path=_write_v6_dataset(tmp_path),
        output_dir=tmp_path / "run-v6",
        config=_tiny_config(architecture_version=6),
        device="cpu",
        epochs=1,
        batch_size=2,
        payment_bank_prefix_min_support=1,
    )
    assert checkpoint.is_file()
    labels = json.loads((checkpoint.parent / "labels.json").read_text(encoding="utf-8"))
    summary = json.loads((checkpoint.parent / "training_summary.json").read_text(encoding="utf-8"))
    assert summary["kind"] == KIND_V6
    assert labels["amount_characters"] == list(V6_AMOUNT_CHARACTERS)
    assert labels["time_characters"] == list(V6_TIME_CHARACTERS)
    assert labels["payment_bank_prefix_classes"] == [PAYMENT_BANK_OTHER_CLASS, "建设银行储蓄卡"]
    assert labels["payment_bank_prefix_oov_by_split"]["test"] == {"records": 1, "other": 0}
    assert summary["records"][0]["val_verifier_macro_exact_match"] is not None
    assert set(summary["records"][0]["val_verifier_by_field"]) == {
        "amount",
        "time",
        "payment_method_field",
    }


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


def test_unified_v5_export_loads_and_evaluates_one_receipt_when_onnx_is_available(tmp_path: Path) -> None:
    onnx = pytest.importorskip("onnx")
    ort = pytest.importorskip("onnxruntime")
    records_path = _write_v5_dataset(tmp_path)
    config = _tiny_config(architecture_version=5)
    checkpoint = train_unified_reader(
        records_path=records_path,
        output_dir=tmp_path / "run-v5",
        config=config,
        device="cpu",
        epochs=1,
        batch_size=2,
    )
    model_path, labels_path, contract_path = export_unified_onnx(
        checkpoint_path=checkpoint,
        output_path=tmp_path / "reader-v5.onnx",
    )
    onnx.checker.check_model(onnx.load_model(model_path))
    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    assert [item.name for item in session.get_outputs()] == list(V5_ONNX_OUTPUT_NAMES)
    outputs = session.run(None, {"field_images": np.zeros((4, 1, 32, 64), dtype=np.float32)})
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    assert [list(value.shape) for value in outputs] == [
        contract["outputs"][name]["shape"] for name in V5_ONNX_OUTPUT_NAMES
    ]
    config_from_contract, _, loaded_contract = _load_onnx_artifacts(model_path)
    assert config_from_contract.architecture_version == 5
    assert loaded_contract["kind"] == KIND_V5
    assert labels_path.is_file()

    summary, failures = evaluate_unified_onnx(
        model_path=model_path,
        records_path=records_path,
        output_dir=tmp_path / "eval-v5",
        split="test",
        device="cpu",
    )
    assert failures == []
    assert summary["by_field"]["amount"]["records"] == 1
    comparisons = [
        json.loads(line)
        for line in (tmp_path / "eval-v5" / "comparisons.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert {row["field"] for row in comparisons} == {
        "amount",
        "time",
        "payment_method_field",
        "transfer_status",
    }
    assert all("delivery_text" in row and "decoder_agrees" in row for row in comparisons)


def test_unified_v6_export_loads_and_evaluates_one_receipt_when_onnx_is_available(tmp_path: Path) -> None:
    onnx = pytest.importorskip("onnx")
    ort = pytest.importorskip("onnxruntime")
    records_path = _write_v6_dataset(tmp_path)
    config = _tiny_config(architecture_version=6)
    checkpoint = train_unified_reader(
        records_path=records_path,
        output_dir=tmp_path / "run-v6",
        config=config,
        device="cpu",
        epochs=1,
        batch_size=2,
        payment_bank_prefix_min_support=1,
    )
    model_path, labels_path, contract_path = export_unified_onnx(
        checkpoint_path=checkpoint,
        output_path=tmp_path / "reader-v6.onnx",
    )
    onnx.checker.check_model(onnx.load_model(model_path))
    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    assert [item.name for item in session.get_outputs()] == list(V6_ONNX_OUTPUT_NAMES)
    outputs = session.run(None, {"field_images": np.zeros((4, 1, 32, 64), dtype=np.float32)})
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    assert [list(value.shape) for value in outputs] == [
        contract["outputs"][name]["shape"] for name in V6_ONNX_OUTPUT_NAMES
    ]
    config_from_contract, _, loaded_contract = _load_onnx_artifacts(model_path)
    assert config_from_contract.architecture_version == 6
    assert loaded_contract["kind"] == KIND_V6
    assert loaded_contract["text_delivery_policy"]["runtime_policy"].startswith("review_only")
    assert labels_path.is_file()

    summary, failures = evaluate_unified_onnx(
        model_path=model_path,
        records_path=records_path,
        output_dir=tmp_path / "eval-v6",
        split="test",
        device="cpu",
    )
    assert failures == []
    assert summary["by_field"]["amount"]["records"] == 1
    comparisons = [
        json.loads(line)
        for line in (tmp_path / "eval-v6" / "comparisons.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    amount = next(row for row in comparisons if row["field"] == "amount")
    assert amount["reference_text"] == "¥199.00"
    assert amount["runtime_policy"].startswith("review_only")
    assert amount["delivery_text"] == "review"
