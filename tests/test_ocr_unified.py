from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

torch = pytest.importorskip("torch")

import transfer_receipt_ai.ocr_unified as ocr_unified

from transfer_receipt_ai.ocr_unified import (
    KIND_V3,
    KIND_V5,
    KIND_V6,
    KIND_V7,
    KIND_V8,
    CHECKPOINT_SELECTION_BALANCED,
    CHECKPOINT_SELECTION_RECIPIENT_PRIORITY,
    ONNX_EXPORT_ATOL,
    ONNX_EXPORT_PAYMENT_LOGITS_ATOL,
    ONNX_EXPORT_RTOL,
    ONNX_EXPORT_TIME_LOGITS_ATOL,
    ONNX_EXPORT_V11_CTC_LOGITS_ATOL,
    ONNX_EXPORT_V11_CTC_LOGITS_MEAN_ABS_CAP,
    ONNX_EXPORT_V12_RECIPIENT_LOGITS_MEAN_ABS_CAP,
    PAYMENT_BANK_OTHER_CLASS,
    V6_AMOUNT_CHARACTERS,
    V6_ONNX_OUTPUT_NAMES,
    V6_TIME_CHARACTERS,
    V8_AMOUNT_CHARACTERS,
    V8_ONNX_OUTPUT_NAMES,
    V5_ONNX_OUTPUT_NAMES,
    UnifiedReaderConfig,
    _checkpoint_config,
    _checkpoint_selection_policy,
    _checkpoint_selection_score,
    _delivery_text,
    _format_exact_match,
    _load_onnx_artifacts,
    _onnx_export_atol,
    _amount_v6_structured_target,
    _select_report_candidates,
    _structured_amount_predictions,
    _structured_amount_v6_predictions,
    _structured_amount_v8_predictions,
    _structured_payment_predictions,
    _structured_payment_v6_predictions,
    _structured_time_predictions,
    _structured_time_v6_predictions,
    _time_v6_structured_target,
    _validate_exported_onnx,
    build_unified_reader,
    decode_ctc_logits,
    evaluate_unified_onnx,
    export_unified_onnx,
    preprocess_image,
    train_unified_reader,
)
from transfer_receipt_ai.ocr_unified_targets import (
    AMOUNT_VISIBLE_FORMAT_V8,
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


def _checkpoint_selection_validation(
    *,
    amount: float = 0.84,
    time: float = 0.99,
    payment: float = 0.95,
    recipient: float = 0.54,
    macro: float = 0.83,
    exact: float = 0.82,
    verifier: float = 0.90,
    loss: float = 0.20,
    unsafe_status_to_success: int = 0,
) -> dict[str, object]:
    """Minimal validation metrics for pure best-checkpoint selection tests."""
    return {
        "loss": loss,
        "exact_match": exact,
        "delivery_exact_overall": exact,
        "candidate_text_macro_exact_match": macro,
        "candidate_text_exact_match": exact,
        "verifier_macro_exact_match": verifier,
        "status_non_success_to_success": unsafe_status_to_success,
        "candidate_text_by_field": {
            field: {"records": 10, "exact_match": score}
            for field, score in {
                "amount": amount,
                "time": time,
                "payment_method_field": payment,
                "recipient_field": recipient,
            }.items()
        },
    }


def test_balanced_checkpoint_selection_preserves_the_legacy_v12_score() -> None:
    config = _tiny_config(architecture_version=12)
    policy = _checkpoint_selection_policy(
        config=config,
        checkpoint_selection=CHECKPOINT_SELECTION_BALANCED,
        checkpoint_min_amount_candidate_exact=None,
        checkpoint_min_time_candidate_exact=None,
        checkpoint_min_payment_candidate_exact=None,
    )
    validation = _checkpoint_selection_validation(
        macro=0.83,
        exact=0.82,
        verifier=0.90,
        loss=0.20,
        unsafe_status_to_success=2,
    )
    score, failures = _checkpoint_selection_score(
        validation,
        config=config,
        status_policy={"training_enabled": True},
        policy=policy,
    )

    assert policy["mode"] == CHECKPOINT_SELECTION_BALANCED
    assert failures == []
    assert score == (-2.0, 0.83, 0.82, 0.90, -0.20)


def test_recipient_priority_checkpoint_selection_prefers_recipient_after_protection() -> None:
    config = _tiny_config(architecture_version=12)
    policy = _checkpoint_selection_policy(
        config=config,
        checkpoint_selection=CHECKPOINT_SELECTION_RECIPIENT_PRIORITY,
        checkpoint_min_amount_candidate_exact=0.80,
        checkpoint_min_time_candidate_exact=0.98,
        checkpoint_min_payment_candidate_exact=0.94,
    )
    lower_recipient, failures = _checkpoint_selection_score(
        _checkpoint_selection_validation(recipient=0.54),
        config=config,
        status_policy={"training_enabled": False},
        policy=policy,
    )
    higher_recipient, higher_failures = _checkpoint_selection_score(
        _checkpoint_selection_validation(recipient=0.55),
        config=config,
        status_policy={"training_enabled": False},
        policy=policy,
    )

    assert failures == higher_failures == []
    assert lower_recipient == (0.0, 0.54, 0.83, 0.82, 0.90, -0.20)
    assert higher_recipient is not None and lower_recipient is not None
    assert higher_recipient > lower_recipient


@pytest.mark.parametrize(
    ("field", "validation_kwargs"),
    (
        ("amount", {"amount": 0.799}),
        ("time", {"time": 0.979}),
        ("payment_method_field", {"payment": 0.939}),
    ),
)
def test_recipient_priority_rejects_epochs_below_any_protected_floor(
    field: str,
    validation_kwargs: dict[str, float],
) -> None:
    config = _tiny_config(architecture_version=12)
    policy = _checkpoint_selection_policy(
        config=config,
        checkpoint_selection=CHECKPOINT_SELECTION_RECIPIENT_PRIORITY,
        checkpoint_min_amount_candidate_exact=0.80,
        checkpoint_min_time_candidate_exact=0.98,
        checkpoint_min_payment_candidate_exact=0.94,
    )
    score, failures = _checkpoint_selection_score(
        _checkpoint_selection_validation(**validation_kwargs),
        config=config,
        status_policy={"training_enabled": False},
        policy=policy,
    )

    assert score is None
    assert failures and failures[0].startswith(f"{field}=")


@pytest.mark.parametrize("invalid_floor", (float("nan"), float("inf"), -0.01, 1.01))
def test_recipient_priority_requires_a_recipient_protocol_and_complete_valid_floors(
    invalid_floor: float,
) -> None:
    with pytest.raises(ValueError, match="requires architecture v9, v10, v11, or v12"):
        _checkpoint_selection_policy(
            config=_tiny_config(architecture_version=8),
            checkpoint_selection=CHECKPOINT_SELECTION_RECIPIENT_PRIORITY,
            checkpoint_min_amount_candidate_exact=0.80,
            checkpoint_min_time_candidate_exact=0.98,
            checkpoint_min_payment_candidate_exact=0.94,
        )
    with pytest.raises(ValueError, match="requires candidate-exact floors"):
        _checkpoint_selection_policy(
            config=_tiny_config(architecture_version=12),
            checkpoint_selection=CHECKPOINT_SELECTION_RECIPIENT_PRIORITY,
            checkpoint_min_amount_candidate_exact=None,
            checkpoint_min_time_candidate_exact=0.98,
            checkpoint_min_payment_candidate_exact=0.94,
        )
    with pytest.raises(ValueError, match="must be between 0 and 1"):
        _checkpoint_selection_policy(
            config=_tiny_config(architecture_version=12),
            checkpoint_selection=CHECKPOINT_SELECTION_RECIPIENT_PRIORITY,
            checkpoint_min_amount_candidate_exact=invalid_floor,
            checkpoint_min_time_candidate_exact=0.98,
            checkpoint_min_payment_candidate_exact=0.94,
        )


def _ctc_logits_for_test_text(text: str, characters: tuple[str, ...], *, time_steps: int) -> np.ndarray:
    """Return a deterministic greedy CTC tensor for evaluator-only tests."""
    logits = np.full((time_steps, len(characters) + 1), -9.0, dtype=np.float32)
    previous = 0
    position = 0
    for character in text:
        current = characters.index(character) + 1
        if current == previous:
            logits[position, 0] = 9.0
            position += 1
        logits[position, current] = 9.0
        previous = current
        position += 1
    logits[position:, 0] = 9.0
    return logits


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
    assert model.payment_ctc_sequence is not model.payment_prefix_sequence


def test_unified_v6_checkpoint_topology_keeps_the_original_separate_time_verifier() -> None:
    """v6's existing state dict and forward route must remain frozen."""
    config = _tiny_config(architecture_version=6)
    model = build_unified_reader(
        payment_vocab_size=6,
        payment_bank_prefix_vocab_size=2,
        config=config,
    )
    outputs = model(torch.zeros((2, 4, 1, 32, 64), dtype=torch.float32))
    (outputs[7].sum() + outputs[8].sum()).backward()
    assert model.time_ctc_sequence.weight_ih_l0.grad is None
    assert model.time_verifier_sequence.weight_ih_l0.grad is not None

    reloaded = build_unified_reader(
        payment_vocab_size=6,
        payment_bank_prefix_vocab_size=2,
        config=config,
    )
    reloaded.load_state_dict(model.state_dict(), strict=True)
    assert hasattr(reloaded, "time_verifier_vertical_reducer")
    assert hasattr(reloaded, "time_verifier_sequence")


def test_unified_v7_time_format_heads_backpropagate_through_the_time_ctc_branch() -> None:
    """v7 lets format supervision reinforce the CTC time reader."""
    config = _tiny_config(architecture_version=7)
    model = build_unified_reader(
        payment_vocab_size=6,
        payment_bank_prefix_vocab_size=2,
        config=config,
    )
    outputs = model(torch.zeros((2, 4, 1, 32, 64), dtype=torch.float32))
    assert len(outputs) == len(V6_ONNX_OUTPUT_NAMES)
    (outputs[7].sum() + outputs[8].sum()).backward()
    assert model.time_ctc_sequence.weight_ih_l0.grad is not None
    assert model.time_ctc_vertical_reducer.depthwise.weight.grad is not None
    assert not hasattr(model, "time_verifier_sequence")


def test_unified_v8_model_shares_amount_ctc_state_with_tiny_format_heads() -> None:
    """v8 keeps one graph while moving visible amount punctuation out of CTC."""
    config = _tiny_config(architecture_version=8)
    model = build_unified_reader(
        payment_vocab_size=6,
        payment_bank_prefix_vocab_size=2,
        config=config,
    )
    outputs = model(torch.zeros((2, 4, 1, 32, 64), dtype=torch.float32))

    assert len(outputs) == len(V8_ONNX_OUTPUT_NAMES)
    (
        amount,
        time,
        payment,
        status,
        amount_currency_style,
        amount_grouped_thousands,
        amount_sign_position,
        time_format,
        time_digits,
        payment_prefix,
        payment_bank,
        payment_tail,
        payment_structure,
        payment_parentheses,
    ) = outputs
    assert list(amount.shape) == [16, 2, len(V8_AMOUNT_CHARACTERS) + 1]
    assert list(time.shape) == [16, 2, len(V6_TIME_CHARACTERS) + 1]
    assert list(payment.shape) == [16, 2, 6]
    assert list(status.shape) == [2, 3]
    assert list(amount_currency_style.shape) == [2, 5]
    assert list(amount_grouped_thousands.shape) == [2, 2]
    assert list(amount_sign_position.shape) == [2, 3]
    assert list(time_format.shape) == [2, 6]
    assert list(time_digits.shape) == [2, 14, 10]
    assert list(payment_prefix.shape) == [16, 2, 6]
    assert list(payment_bank.shape) == [2, 2]
    assert list(payment_tail.shape) == [2, 4, 10]
    assert list(payment_structure.shape) == [2, 2]
    assert list(payment_parentheses.shape) == [2, 2]
    assert not hasattr(model, "amount_verifier_sequence")

    # The three finite format heads must reinforce the canonical amount CTC
    # sequence rather than adding a second amount reader/session.
    (amount_currency_style.sum() + amount_grouped_thousands.sum() + amount_sign_position.sum()).backward()
    assert model.amount_ctc_sequence.weight_ih_l0.grad is not None
    assert model.amount_ctc_vertical_reducer.depthwise.weight.grad is not None


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


def test_v7_checkpoint_config_has_its_own_kind_and_does_not_relabel_v6() -> None:
    config = _checkpoint_config(
        {
            "kind": KIND_V7,
            "config": {
                "architecture_version": 7,
                "image_height": 32,
                "image_width": 64,
                "base_channels": 8,
                "numeric_hidden_size": 16,
                "payment_hidden_size": 16,
                "pooled_width": 2,
            },
        }
    )
    assert config.architecture_version == 7
    assert KIND_V6 != KIND_V7


def test_v8_checkpoint_config_has_a_distinct_kind_and_keeps_the_compact_amount_charset() -> None:
    config = _checkpoint_config(
        {
            "kind": KIND_V8,
            "config": {
                "architecture_version": 8,
                "image_height": 32,
                "image_width": 64,
                "base_channels": 8,
                "numeric_hidden_size": 16,
                "payment_hidden_size": 16,
                "pooled_width": 2,
            },
        }
    )
    assert config.architecture_version == 8
    assert KIND_V7 != KIND_V8
    assert set(V8_AMOUNT_CHARACTERS) == set("0123456789.-")


def test_optional_exact_metric_formats_as_na_without_status_labels() -> None:
    assert _format_exact_match(None) == "n/a"
    assert _format_exact_match(0.5) == "50.00%"


def test_ctc_onnx_export_tolerances_allow_bounded_gru_drift_without_argmax_change(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Permit scoped CTC GRU round-off, never a changed greedy decision."""
    expected = np.asarray([[0.0, 0.5, -0.4]], dtype=np.float32)
    payment_actual = expected.copy()
    payment_actual[0, 0] += np.float32(0.001765964)
    time_actual = expected.copy()
    # This is the v10 server-export drift that previously stopped a completed
    # training run.  Its per-position CTC decision did not change.
    time_actual[0, 0] += np.float32(0.0014093737)
    v11_amount_expected = np.repeat(expected, repeats=128, axis=0)
    v11_amount_actual = v11_amount_expected.copy()
    # This is the v11 server-export amount drift.  It was measured on the
    # fixed [5, 1, H, W] export probe and preserves every CTC decision.
    v11_amount_actual[:14, 0] += np.float32(0.0095449686)
    v11_time_expected = np.repeat(expected, repeats=128, axis=0)
    v11_time_actual = v11_time_expected.copy()
    # v11's time CTC branch can use a different CPU GRU accumulation order.
    # The measured max drift remains below the dedicated hard 0.03 cap and
    # keeps its per-position greedy CTC decisions unchanged.
    v11_time_actual[:8, 0] += np.float32(0.026508331)
    assert np.argmax(payment_actual, axis=-1).tolist() == np.argmax(expected, axis=-1).tolist()
    assert np.argmax(time_actual, axis=-1).tolist() == np.argmax(expected, axis=-1).tolist()
    assert not np.allclose(payment_actual, expected, rtol=ONNX_EXPORT_RTOL, atol=ONNX_EXPORT_ATOL)
    assert not np.allclose(time_actual, expected, rtol=ONNX_EXPORT_RTOL, atol=ONNX_EXPORT_ATOL)
    assert not np.allclose(v11_amount_actual, v11_amount_expected, rtol=ONNX_EXPORT_RTOL, atol=ONNX_EXPORT_ATOL)
    assert not np.allclose(v11_time_actual, v11_time_expected, rtol=ONNX_EXPORT_RTOL, atol=ONNX_EXPORT_ATOL)
    assert np.allclose(payment_actual, expected, rtol=ONNX_EXPORT_RTOL, atol=ONNX_EXPORT_PAYMENT_LOGITS_ATOL)
    assert np.allclose(time_actual, expected, rtol=ONNX_EXPORT_RTOL, atol=ONNX_EXPORT_TIME_LOGITS_ATOL)
    assert np.allclose(
        v11_amount_actual,
        v11_amount_expected,
        rtol=ONNX_EXPORT_RTOL,
        atol=ONNX_EXPORT_V11_CTC_LOGITS_ATOL,
    )
    assert np.allclose(
        v11_time_actual,
        v11_time_expected,
        rtol=ONNX_EXPORT_RTOL,
        atol=ONNX_EXPORT_V11_CTC_LOGITS_ATOL,
    )
    assert decode_ctc_logits(v11_time_actual[:, np.newaxis, :], characters=["A", "B"]) == decode_ctc_logits(
        v11_time_expected[:, np.newaxis, :],
        characters=["A", "B"],
    )
    assert _onnx_export_atol("payment_logits") == ONNX_EXPORT_PAYMENT_LOGITS_ATOL
    assert _onnx_export_atol("time_logits") == ONNX_EXPORT_TIME_LOGITS_ATOL
    assert _onnx_export_atol("amount_logits") == ONNX_EXPORT_ATOL
    v11_config = UnifiedReaderConfig(architecture_version=11)
    assert _onnx_export_atol("amount_logits", config=v11_config) == ONNX_EXPORT_V11_CTC_LOGITS_ATOL
    assert _onnx_export_atol("time_logits", config=v11_config) == ONNX_EXPORT_V11_CTC_LOGITS_ATOL
    assert _onnx_export_atol("payment_logits", config=v11_config) == ONNX_EXPORT_V11_CTC_LOGITS_ATOL
    assert _onnx_export_atol("payment_prefix_logits", config=v11_config) == ONNX_EXPORT_V11_CTC_LOGITS_ATOL
    assert _onnx_export_atol("recipient_logits", config=v11_config) == ONNX_EXPORT_V11_CTC_LOGITS_ATOL
    assert _onnx_export_atol("status_logits", config=v11_config) == ONNX_EXPORT_ATOL
    assert ONNX_EXPORT_V11_CTC_LOGITS_MEAN_ABS_CAP == 1e-3
    v12_config = UnifiedReaderConfig(architecture_version=12)
    assert (
        ocr_unified._onnx_export_mean_abs_cap("recipient_logits", config=v12_config)
        == ONNX_EXPORT_V12_RECIPIENT_LOGITS_MEAN_ABS_CAP
    )
    assert (
        ocr_unified._onnx_export_mean_abs_cap("amount_logits", config=v12_config)
        == ONNX_EXPORT_V11_CTC_LOGITS_MEAN_ABS_CAP
    )
    assert _onnx_export_atol(
        "amount_logits",
        config=UnifiedReaderConfig(architecture_version=10),
    ) == ONNX_EXPORT_ATOL

    def validate(
        output_name: str,
        runtime_output: np.ndarray,
        torch_output: np.ndarray,
        *,
        config: UnifiedReaderConfig | None = None,
    ) -> None:
        class NamedItem:
            def __init__(self, name: str) -> None:
                self.name = name

        class FakeSession:
            def get_inputs(self) -> list[NamedItem]:
                return [NamedItem("field_images")]

            def get_outputs(self) -> list[NamedItem]:
                return [NamedItem(output_name)]

            def run(self, names: list[str], feed: dict[str, np.ndarray]) -> list[np.ndarray]:
                assert names == [output_name]
                assert list(feed) == ["field_images"]
                return [runtime_output]

        class FakeOnnxRuntime:
            def InferenceSession(self, path: str, providers: list[str]) -> FakeSession:
                assert providers == ["CPUExecutionProvider"]
                return FakeSession()

        monkeypatch.setattr(ocr_unified, "_require_onnxruntime", lambda: FakeOnnxRuntime())
        _validate_exported_onnx(
            Path("ignored.onnx"),
            dummy=torch.zeros((1, 4), dtype=torch.float32),
            output_names=[output_name],
            expected_outputs=[torch.as_tensor(torch_output)],
            config=config,
        )

    validate("payment_logits", payment_actual, expected)
    validate("time_logits", time_actual, expected)
    validate("amount_logits", v11_amount_actual, v11_amount_expected, config=v11_config)
    validate("time_logits", v11_time_actual, v11_time_expected, config=v11_config)

    changed_decision = np.asarray([[0.0018, 0.0005]], dtype=np.float32)
    expected_decision = np.asarray([[0.0, 0.0005]], dtype=np.float32)
    with pytest.raises(ValueError, match=r"argmax_mismatches=1/1"):
        validate("time_logits", changed_decision, expected_decision)

    with pytest.raises(ValueError, match=r"amount_logits.*atol=0.001"):
        validate("amount_logits", time_actual, expected)

    with pytest.raises(ValueError, match=r"amount_logits.*atol=0.001"):
        validate(
            "amount_logits",
            v11_amount_actual,
            v11_amount_expected,
            config=UnifiedReaderConfig(architecture_version=10),
        )

    with pytest.raises(ValueError, match=r"time_logits.*atol=0.002"):
        validate(
            "time_logits",
            v11_time_actual,
            v11_time_expected,
            config=UnifiedReaderConfig(architecture_version=10),
        )

    v11_ctc_cap_expected = np.repeat(
        np.asarray([[40.0, 41.0, -4.0]], dtype=np.float32),
        repeats=128,
        axis=0,
    )
    v11_ctc_over_cap = v11_ctc_cap_expected.copy()
    v11_ctc_over_cap[0, 0] += np.float32(0.0301)
    with pytest.raises(ValueError, match=r"amount_logits.*atol=0.03.*max_abs_cap=0.03"):
        validate("amount_logits", v11_ctc_over_cap, v11_ctc_cap_expected, config=v11_config)

    v11_ctc_mean_expected = np.repeat(expected, repeats=128, axis=0)
    v11_ctc_mean_over_cap = v11_ctc_mean_expected + np.float32(0.0011)
    with pytest.raises(ValueError, match=r"amount_logits.*mean_abs_cap=0.001"):
        validate("amount_logits", v11_ctc_mean_over_cap, v11_ctc_mean_expected, config=v11_config)

    v12_recipient_mean_expected = np.repeat(expected, repeats=128, axis=0)
    # Measured server export drift: it is still inside the 0.03 hard cap and
    # preserves every greedy CTC decision, but lies just above the generic
    # 0.001 mean cap.  This exception is intentionally recipient/v12-only.
    v12_recipient_measured_drift = v12_recipient_mean_expected + np.float32(0.001011668)
    validate(
        "recipient_logits",
        v12_recipient_measured_drift,
        v12_recipient_mean_expected,
        config=v12_config,
    )
    with pytest.raises(ValueError, match=r"recipient_logits.*mean_abs_cap=0.00105"):
        validate(
            "recipient_logits",
            v12_recipient_mean_expected + np.float32(0.00106),
            v12_recipient_mean_expected,
            config=v12_config,
        )
    with pytest.raises(ValueError, match=r"amount_logits.*mean_abs_cap=0.001"):
        validate(
            "amount_logits",
            v12_recipient_measured_drift,
            v12_recipient_mean_expected,
            config=v12_config,
        )

    changed_v11_amount_decision = np.asarray([[0.029, 0.0005]], dtype=np.float32)
    with pytest.raises(ValueError, match=r"argmax_mismatches=1/1"):
        validate(
            "amount_logits",
            changed_v11_amount_decision,
            expected_decision,
            config=v11_config,
        )

    time_over_cap = expected.copy()
    time_over_cap[0, 0] += np.float32(0.0021)
    with pytest.raises(ValueError, match=r"time_logits.*atol=0.002"):
        validate("time_logits", time_over_cap, expected)


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


@pytest.mark.parametrize("architecture_version", (6, 7))
def test_v6_protocol_report_candidate_uses_time_template_but_retains_raw_amount_display(
    architecture_version: int,
) -> None:
    ctc = {
        "amount": ("￥99.96", 0.91),
        "time": ("0240", 0.92),
        "payment_method_field": ("建设银行储蓄卡（3667）", 0.93),
    }
    structured = {
        "amount": ("99.96", 0.99),
        "time": ("02:40", 0.99),
        "payment_method_field": ("建设银行储蓄卡（3667）", 0.99),
    }
    selected = _select_report_candidates(ctc, structured, config=_tiny_config(architecture_version=architecture_version))
    assert selected["time"] == ("02:40", 0.99)
    assert selected["amount"] == ("￥99.96", 0.91)
    assert selected["payment_method_field"] == ("建设银行储蓄卡（3667）", 0.93)


def test_v8_report_candidate_applies_only_a_safe_amount_format_and_time_template() -> None:
    ctc = {
        "amount": ("1234.56", 0.91),
        "time": ("0240", 0.92),
        "payment_method_field": ("建设银行储蓄卡（3667）", 0.93),
    }
    structured = {
        "amount": ("¥1,234.56", 0.99),
        "time": ("02:40", 0.99),
        # Payment's known-bank heads remain audit-only until there is a
        # calibrated acceptance policy, so CTC must remain the candidate.
        "payment_method_field": ("工商银行储蓄卡（3667）", 0.99),
    }
    selected = _select_report_candidates(ctc, structured, config=_tiny_config(architecture_version=8))

    assert selected["amount"] == ("¥1,234.56", 0.99)
    assert selected["time"] == ("02:40", 0.99)
    assert selected["payment_method_field"] == ("建设银行储蓄卡（3667）", 0.93)


def test_v8_positive_amount_does_not_gate_on_irrelevant_sign_confidence() -> None:
    """A positive canonical amount has no sign placement to decide.

    Low confidence in the unused negative-sign classes must not discard an
    otherwise safe visible-CNY rendering.  The currency decision remains
    explicit and high-confidence here.
    """
    rendered = _structured_amount_v8_predictions(
        [("99.99", 0.99)],
        np.asarray([[-9.0, 9.0, -9.0, -9.0, -9.0]], dtype=np.float32),  # ¥
        np.asarray([[9.0, -9.0]], dtype=np.float32),  # ungrouped
        np.asarray([[0.0, 0.0, 0.0]], dtype=np.float32),  # irrelevant for a positive value
        min_confidence=0.90,
    )
    assert rendered == [("¥99.99", 0.99)]


def test_v8_sub_thousand_amount_does_not_gate_on_irrelevant_grouping_confidence() -> None:
    """Grouping cannot change a <1000 integer, so it is not a safety gate."""
    rendered = _structured_amount_v8_predictions(
        [("99.99", 0.99)],
        np.asarray([[-9.0, 9.0, -9.0, -9.0, -9.0]], dtype=np.float32),  # ¥
        np.asarray([[0.0, 0.0]], dtype=np.float32),  # irrelevant for a two-digit integer
        np.asarray([[9.0, -9.0, -9.0]], dtype=np.float32),  # no sign
        min_confidence=0.90,
    )
    assert rendered == [("¥99.99", 0.99)]


def test_v8_bare_negative_amount_does_not_gate_on_irrelevant_sign_position_confidence() -> None:
    """A bare negative can only render with its leading minus already in CTC."""
    rendered = _structured_amount_v8_predictions(
        [("-99.99", 0.99)],
        np.asarray([[9.0, -9.0, -9.0, -9.0, -9.0]], dtype=np.float32),  # no currency
        np.asarray([[9.0, -9.0]], dtype=np.float32),
        np.asarray([[0.0, 0.0, 0.0]], dtype=np.float32),  # no visible choice without currency
        min_confidence=0.90,
    )
    assert rendered == [("-99.99", 0.99)]


def test_v8_amount_keeps_currency_style_as_a_required_confidence_gate() -> None:
    """The visible currency symbol can change, so a low-confidence vote stays review-only."""
    rendered = _structured_amount_v8_predictions(
        [("99.99", 0.99)],
        np.asarray([[0.0, 0.0, 0.0, 0.0, 0.0]], dtype=np.float32),  # no style reaches 0.90
        np.asarray([[9.0, -9.0]], dtype=np.float32),
        np.asarray([[9.0, -9.0, -9.0]], dtype=np.float32),
        min_confidence=0.90,
    )
    assert rendered == [(None, 0.0)]


@pytest.mark.parametrize(
    ("canonical", "currency_logits", "grouping_logits", "sign_logits"),
    (
        # A negative amount needs its sign position.  Its argmax is correct,
        # but the confidence is intentionally below the delivery threshold.
        (
            "-99.99",
            [[-9.0, 9.0, -9.0, -9.0, -9.0]],
            [[9.0, -9.0]],
            [[0.0, 0.1, 0.0]],
        ),
        # A four-digit amount needs an explicit grouping decision.  Its
        # ungrouped argmax is correct but deliberately not confident enough.
        (
            "1234.56",
            [[-9.0, 9.0, -9.0, -9.0, -9.0]],
            [[0.1, 0.0]],
            [[9.0, -9.0, -9.0]],
        ),
    ),
)
def test_v8_amount_keeps_relevant_sign_and_grouping_as_confidence_gates(
    canonical: str,
    currency_logits: list[list[float]],
    grouping_logits: list[list[float]],
    sign_logits: list[list[float]],
) -> None:
    rendered = _structured_amount_v8_predictions(
        [(canonical, 0.99)],
        np.asarray(currency_logits, dtype=np.float32),
        np.asarray(grouping_logits, dtype=np.float32),
        np.asarray(sign_logits, dtype=np.float32),
        min_confidence=0.90,
    )
    assert rendered == [(None, 0.0)]


def test_v8_evaluation_amount_format_override_changes_candidate_summary_without_mutating_artifact_policy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A calibration override is evaluation-only, never an artifact rewrite.

    This uses a deterministic fake ONNX session so the test exercises the
    public evaluator path without requiring ONNX Runtime.  The amount CTC has
    the correct canonical digits, while its selected ¥ style has deliberately
    low confidence: the persisted 0.90 gate must retain ``199.00``; a 0.0
    evaluator override may render ``¥199.00`` and improve only the report.
    """
    records_path = _write_v6_dataset(tmp_path)
    model_path = tmp_path / "reader-v8.onnx"
    model_path.write_bytes(b"test-v8-artifact-must-not-change")
    artifact_bytes = model_path.read_bytes()
    artifact_config = _tiny_config(architecture_version=8)
    assert artifact_config.amount_format_min_confidence == 0.90

    payment_characters = ("x",)
    time_steps = artifact_config.image_width // 4
    output_names = list(V8_ONNX_OUTPUT_NAMES)

    def classification_logits(classes: int, selected: int, *, high: bool = True) -> np.ndarray:
        logits = np.full((classes,), -9.0 if high else 0.0, dtype=np.float32)
        logits[selected] = 9.0 if high else 0.2
        return logits

    time_digits = np.full((ocr_unified.TIME_DISPLAY_DIGIT_SLOTS, 10), -9.0, dtype=np.float32)
    # clock_h_mm consumes the first 4 positions as HHMM: 01:44 -> 1:44.
    for index, digit in enumerate((0, 1, 4, 4)):
        time_digits[index, digit] = 9.0
    time_digits[4:, 0] = 9.0
    runtime_outputs = {
        "amount_logits": _ctc_logits_for_test_text("199.00", V8_AMOUNT_CHARACTERS, time_steps=time_steps),
        "time_logits": _ctc_logits_for_test_text("1:44", V6_TIME_CHARACTERS, time_steps=time_steps),
        "payment_logits": np.full((time_steps, len(payment_characters) + 1), -9.0, dtype=np.float32),
        "status_logits": classification_logits(3, 0),
        # Index 1 is ¥ but a 0.2-vs-0.0 logit is deliberately below 0.90.
        "amount_currency_style_logits": classification_logits(5, 1, high=False),
        "amount_grouped_thousands_logits": classification_logits(2, 0),
        "amount_sign_position_logits": classification_logits(3, 0),
        "time_format_logits": classification_logits(len(ocr_unified.TIME_DISPLAY_FORMAT_CLASSES), 0),
        "time_digit_logits": time_digits,
        "payment_prefix_logits": np.full((time_steps, len(payment_characters) + 1), -9.0, dtype=np.float32),
        "payment_bank_prefix_logits": classification_logits(2, 0),
        "payment_tail_digit_logits": np.eye(10, dtype=np.float32)[[0, 0, 0, 0]] * 18.0 - 9.0,
        "payment_structure_logits": classification_logits(2, 0),
        "payment_parentheses_logits": classification_logits(2, 0),
    }
    # The CTC blank has to win for the unused payment CTC stream.
    runtime_outputs["payment_logits"][:, 0] = 9.0
    runtime_outputs["payment_prefix_logits"][:, 0] = 9.0

    class NamedValue:
        def __init__(self, name: str, shape: list[int]) -> None:
            self.name = name
            self.shape = shape

    class FakeSession:
        def get_inputs(self) -> list[NamedValue]:
            return [NamedValue("field_images", [4, 1, 32, 64])]

        def get_outputs(self) -> list[NamedValue]:
            return [NamedValue(name, list(np.asarray(runtime_outputs[name]).shape)) for name in output_names]

        def get_providers(self) -> list[str]:
            return ["CPUExecutionProvider"]

        def run(self, names: list[str], feed: dict[str, np.ndarray]) -> list[np.ndarray]:
            assert names == output_names
            assert list(feed) == ["field_images"]
            assert list(feed["field_images"].shape) == [4, 1, 32, 64]
            return [runtime_outputs[name] for name in names]

    status_counts = {
        split: {"success": 1, "pending": 0, "failed": 0}
        for split in ("train", "val", "test")
    }
    contract = {
        "payment_bank_prefix_classes": [PAYMENT_BANK_OTHER_CLASS, "建设银行储蓄卡"],
        "training_status_class_counts": status_counts,
        "status_head_policy": {"runtime_policy": "review_only"},
    }
    monkeypatch.setattr(
        ocr_unified,
        "_load_onnx_artifacts",
        lambda _path: (artifact_config, list(payment_characters), contract),
    )
    monkeypatch.setattr(ocr_unified, "_require_onnxruntime", lambda: object())
    monkeypatch.setattr(
        ocr_unified,
        "_create_onnx_session",
        lambda _ort, _path, *, device: (FakeSession(), ["CPUExecutionProvider"]),
    )

    baseline, baseline_failures = evaluate_unified_onnx(
        model_path=model_path,
        records_path=records_path,
        output_dir=tmp_path / "eval-artifact-threshold",
        split="test",
        device="cpu",
    )
    overridden, overridden_failures = evaluate_unified_onnx(
        model_path=model_path,
        records_path=records_path,
        output_dir=tmp_path / "eval-override-threshold",
        split="test",
        device="cpu",
        amount_format_min_confidence_override=0.0,
    )

    assert baseline_failures == overridden_failures == []
    assert baseline["amount_format_policy"] == {
        "artifact_min_confidence": 0.90,
        "effective_min_confidence": 0.90,
        "evaluation_override": None,
    }
    assert overridden["amount_format_policy"] == {
        "artifact_min_confidence": 0.90,
        "effective_min_confidence": 0.0,
        "evaluation_override": 0.0,
    }
    assert baseline["by_field"]["amount"]["raw_exact_match"] == 0.0
    assert overridden["by_field"]["amount"]["raw_exact_match"] == 1.0

    def amount_comparison(output_dir: Path) -> dict[str, object]:
        rows = [
            json.loads(line)
            for line in (output_dir / "comparisons.jsonl").read_text(encoding="utf-8").splitlines()
        ]
        return next(row for row in rows if row["field"] == "amount")

    assert amount_comparison(tmp_path / "eval-artifact-threshold")["candidate_text"] == "199.00"
    assert amount_comparison(tmp_path / "eval-override-threshold")["candidate_text"] == "¥199.00"
    # ``replace`` in the evaluator must not mutate the loaded frozen config,
    # and evaluation must not rewrite the ONNX binary.
    assert artifact_config.amount_format_min_confidence == 0.90
    assert model_path.read_bytes() == artifact_bytes


def test_v8_amount_format_override_is_validated_and_rejected_for_non_v8_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(ValueError, match="amount_format_min_confidence_override must be between 0 and 1"):
        evaluate_unified_onnx(
            model_path=tmp_path / "ignored.onnx",
            records_path=tmp_path / "ignored.jsonl",
            output_dir=tmp_path / "ignored-output",
            amount_format_min_confidence_override=1.01,
        )

    monkeypatch.setattr(
        ocr_unified,
        "_load_onnx_artifacts",
        lambda _path: (_tiny_config(architecture_version=7), ["x"], {}),
    )
    with pytest.raises(ValueError, match="supported only by v8-v12 ONNX artifacts"):
        evaluate_unified_onnx(
            model_path=tmp_path / "ignored-v7.onnx",
            records_path=tmp_path / "ignored-v7.jsonl",
            output_dir=tmp_path / "ignored-v7-output",
            amount_format_min_confidence_override=0.0,
        )


def test_evaluate_parser_exposes_v8_amount_format_confidence_override() -> None:
    args = ocr_unified.build_parser().parse_args(
        [
            "evaluate",
            "--model",
            "reader.onnx",
            "--records",
            "records.jsonl",
            "--output",
            "eval",
            "--amount-format-min-confidence-override",
            "0.25",
        ]
    )
    assert args.amount_format_min_confidence_override == 0.25


def test_train_parser_and_main_forward_recipient_priority_checkpoint_protection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = ocr_unified.build_parser().parse_args(
        [
            "train",
            "--records",
            "records.jsonl",
            "--output",
            "run",
            "--architecture",
            "v12",
            "--checkpoint-selection",
            "recipient_priority",
            "--checkpoint-min-amount-candidate-exact",
            "0.80",
            "--checkpoint-min-time-candidate-exact",
            "0.98",
            "--checkpoint-min-payment-candidate-exact",
            "0.94",
        ]
    )
    assert args.checkpoint_selection == CHECKPOINT_SELECTION_RECIPIENT_PRIORITY
    assert args.checkpoint_min_amount_candidate_exact == 0.80
    assert args.checkpoint_min_time_candidate_exact == 0.98
    assert args.checkpoint_min_payment_candidate_exact == 0.94

    observed: dict[str, object] = {}

    def fake_train(**kwargs: object) -> Path:
        observed.update(kwargs)
        return tmp_path / "best.pt"

    monkeypatch.setattr(ocr_unified, "train_unified_reader", fake_train)
    ocr_unified.main(
        [
            "train",
            "--records",
            "records.jsonl",
            "--output",
            "run",
            "--architecture",
            "v12",
            "--checkpoint-selection",
            "recipient_priority",
            "--checkpoint-min-amount-candidate-exact",
            "0.80",
            "--checkpoint-min-time-candidate-exact",
            "0.98",
            "--checkpoint-min-payment-candidate-exact",
            "0.94",
        ]
    )
    assert observed["checkpoint_selection"] == CHECKPOINT_SELECTION_RECIPIENT_PRIORITY
    assert observed["checkpoint_min_amount_candidate_exact"] == 0.80
    assert observed["checkpoint_min_time_candidate_exact"] == 0.98
    assert observed["checkpoint_min_payment_candidate_exact"] == 0.94


def test_export_parser_and_main_forward_v8_amount_format_confidence_override(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The export CLI must forward the policy into a new bundle, not evaluation."""
    args = ocr_unified.build_parser().parse_args(
        [
            "export",
            "--checkpoint",
            "reader.pt",
            "--output",
            "reader.onnx",
            "--amount-format-min-confidence",
            "0.80",
        ]
    )
    assert args.amount_format_min_confidence == 0.80

    observed: dict[str, object] = {}

    def fake_export(**kwargs: object) -> tuple[Path, Path, Path]:
        observed.update(kwargs)
        return tmp_path / "reader.onnx", tmp_path / "reader.labels.json", tmp_path / "reader.contract.json"

    monkeypatch.setattr(ocr_unified, "export_unified_onnx", fake_export)
    ocr_unified.main(
        [
            "export",
            "--checkpoint",
            "checkpoint.pt",
            "--output",
            "bundle.onnx",
            "--amount-format-min-confidence",
            "0.80",
        ]
    )
    assert observed == {
        "checkpoint_path": Path("checkpoint.pt"),
        "output_path": Path("bundle.onnx"),
        "amount_format_min_confidence": 0.80,
    }


@pytest.mark.parametrize("threshold", (-0.01, 1.01, float("nan"), float("inf")))
def test_export_amount_format_threshold_rejects_invalid_values_before_writing(
    threshold: float,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A malformed override must never create a partial ONNX bundle."""
    checkpoint_path = tmp_path / "reader.pt"
    checkpoint_path.write_bytes(b"not-loaded-because-validation-precedes-export")
    output_path = tmp_path / "reader.onnx"
    checkpoint_payload = {
        "schema_version": ocr_unified.SCHEMA_VERSION,
        "kind": KIND_V8,
        "config": asdict(_tiny_config(architecture_version=8)),
    }
    monkeypatch.setattr(ocr_unified, "_require_torch", lambda: (object(), object()))
    monkeypatch.setattr(ocr_unified, "_load_checkpoint", lambda _path, *, torch: checkpoint_payload)

    with pytest.raises(ValueError, match="amount_format_min_confidence must be between 0 and 1"):
        export_unified_onnx(
            checkpoint_path=checkpoint_path,
            output_path=output_path,
            amount_format_min_confidence=threshold,
        )

    assert not output_path.exists()
    assert not output_path.with_suffix(".labels.json").exists()
    assert not output_path.with_suffix(".contract.json").exists()
    assert not output_path.with_name(".reader.exporting.onnx").exists()


def test_export_amount_format_threshold_rejects_non_v8_before_writing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Historical decoding protocols must stay unchanged by the v8 option."""
    checkpoint_path = tmp_path / "reader-v7.pt"
    checkpoint_path.write_bytes(b"not-loaded-because-v7-rejects-before-export")
    output_path = tmp_path / "reader-v7.onnx"
    checkpoint_payload = {
        "schema_version": ocr_unified.SCHEMA_VERSION,
        "kind": KIND_V7,
        "config": asdict(_tiny_config(architecture_version=7)),
    }
    monkeypatch.setattr(ocr_unified, "_require_torch", lambda: (object(), object()))
    monkeypatch.setattr(ocr_unified, "_load_checkpoint", lambda _path, *, torch: checkpoint_payload)

    with pytest.raises(ValueError, match="export override is supported only by v8-v12 checkpoints"):
        export_unified_onnx(
            checkpoint_path=checkpoint_path,
            output_path=output_path,
            amount_format_min_confidence=0.80,
        )

    assert not output_path.exists()
    assert not output_path.with_suffix(".labels.json").exists()
    assert not output_path.with_suffix(".contract.json").exists()


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


def test_tiny_v8_training_writes_compact_amount_charset_and_candidate_metrics(tmp_path: Path) -> None:
    """v8's canonical amount CTC and display grammar must train without ONNX.

    Keep this separate from the ONNX integration test so a broken v8 training
    path is caught even on environments that intentionally do not install ONNX
    Runtime.
    """
    checkpoint = train_unified_reader(
        records_path=_write_v6_dataset(tmp_path),
        output_dir=tmp_path / "run-v8",
        config=_tiny_config(architecture_version=8),
        device="cpu",
        epochs=1,
        batch_size=2,
        payment_bank_prefix_min_support=1,
    )

    assert checkpoint.is_file()
    assert (checkpoint.parent / "last.pt").is_file()
    labels = json.loads((checkpoint.parent / "labels.json").read_text(encoding="utf-8"))
    summary = json.loads((checkpoint.parent / "training_summary.json").read_text(encoding="utf-8"))
    assert summary["kind"] == KIND_V8
    assert labels["amount_characters"] == list(V8_AMOUNT_CHARACTERS)
    assert labels["time_characters"] == list(V6_TIME_CHARACTERS)
    assert labels["payment_bank_prefix_classes"] == [PAYMENT_BANK_OTHER_CLASS, "建设银行储蓄卡"]

    structured_counts = labels["structured_target_counts"]
    for target in ("amount_visible_format_v8", "time_display", "payment_bank_prefix"):
        assert structured_counts[target]["train"] > 0
        assert structured_counts[target]["val"] > 0

    epoch = summary["records"][0]
    assert epoch["val_candidate_text_exact_match"] is not None
    assert epoch["val_candidate_text_macro_exact_match"] is not None
    assert set(epoch["val_candidate_text_by_field"]) == {
        "amount",
        "time",
        "payment_method_field",
    }
    assert epoch["val_candidate_text_by_field"]["amount"]["records"] > 0


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


def test_unified_v7_export_loads_the_same_visible_format_protocol_when_onnx_is_available(tmp_path: Path) -> None:
    onnx = pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    records_path = _write_v6_dataset(tmp_path)
    checkpoint = train_unified_reader(
        records_path=records_path,
        output_dir=tmp_path / "run-v7",
        config=_tiny_config(architecture_version=7),
        device="cpu",
        epochs=1,
        batch_size=2,
        payment_bank_prefix_min_support=1,
    )
    model_path, _, contract_path = export_unified_onnx(
        checkpoint_path=checkpoint,
        output_path=tmp_path / "reader-v7.onnx",
    )
    onnx.checker.check_model(onnx.load_model(model_path))
    config_from_contract, _, loaded_contract = _load_onnx_artifacts(model_path)
    assert config_from_contract.architecture_version == 7
    assert loaded_contract["kind"] == KIND_V7
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    assert set(contract["outputs"]) == set(V6_ONNX_OUTPUT_NAMES)


def test_unified_v8_export_loads_and_evaluates_guarded_visible_amounts_when_onnx_is_available(
    tmp_path: Path,
) -> None:
    """The v8 artifact must carry its grammar through the deployed ONNX path."""
    onnx = pytest.importorskip("onnx")
    ort = pytest.importorskip("onnxruntime")
    records_path = _write_v6_dataset(tmp_path)
    checkpoint = train_unified_reader(
        records_path=records_path,
        output_dir=tmp_path / "run-v8",
        config=_tiny_config(architecture_version=8),
        device="cpu",
        epochs=1,
        batch_size=2,
        payment_bank_prefix_min_support=1,
    )
    model_path, labels_path, contract_path = export_unified_onnx(
        checkpoint_path=checkpoint,
        output_path=tmp_path / "reader-v8.onnx",
        amount_format_min_confidence=0.80,
    )
    onnx.checker.check_model(onnx.load_model(model_path))
    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    assert [item.name for item in session.get_outputs()] == list(V8_ONNX_OUTPUT_NAMES)
    outputs = session.run(None, {"field_images": np.zeros((4, 1, 32, 64), dtype=np.float32)})
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    assert [list(value.shape) for value in outputs] == [
        contract["outputs"][name]["shape"] for name in V8_ONNX_OUTPUT_NAMES
    ]
    labels = json.loads(labels_path.read_text(encoding="utf-8"))
    assert labels["amount_characters"] == list(V8_AMOUNT_CHARACTERS)
    assert labels["structured_decoder"]["amount_visible_format"] == AMOUNT_VISIBLE_FORMAT_V8
    assert labels["structured_decoder"]["amount_format_min_confidence"] == 0.80
    assert contract["model"]["amount_format_min_confidence"] == 0.80

    config_from_contract, _, loaded_contract = _load_onnx_artifacts(model_path)
    assert config_from_contract.architecture_version == 8
    assert config_from_contract.amount_format_min_confidence == 0.80
    assert loaded_contract["kind"] == KIND_V8
    assert loaded_contract["text_delivery_policy"]["runtime_policy"].startswith("review_only")
    # The export policy belongs only to the new bundle: the training
    # checkpoint remains an auditable record of the original 0.90 policy.
    checkpoint_config = _checkpoint_config(ocr_unified._load_checkpoint(checkpoint, torch=torch))
    assert checkpoint_config.amount_format_min_confidence == 0.90

    summary, failures = evaluate_unified_onnx(
        model_path=model_path,
        records_path=records_path,
        output_dir=tmp_path / "eval-v8",
        split="test",
        device="cpu",
    )
    assert failures == []
    assert summary["by_field"]["amount"]["records"] == 1
    comparisons = [
        json.loads(line)
        for line in (tmp_path / "eval-v8" / "comparisons.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    amount = next(row for row in comparisons if row["field"] == "amount")
    # v8's CTC target is canonical, but its report/evaluation target is the
    # audited visible teacher text so rendered CNY punctuation is measured.
    assert amount["reference_text"] == "¥199.00"
    assert amount["ctc_candidate_text"] is not None
    assert "structured_candidate_text" in amount
    assert amount["runtime_policy"].startswith("review_only")
