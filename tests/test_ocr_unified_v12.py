"""Focused tests for v12's single-reader, dual-static-input OCR protocol."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest

import transfer_receipt_ai.ocr_unified as ocr_unified

from transfer_receipt_ai.ocr_unified import (
    INIT_CHECKPOINT_MODE_RECIPIENT_ONLY_EXPANSION,
    KIND_V12,
    PAYMENT_BANK_OTHER_CLASS,
    STATUS_CLASSES,
    V6_TIME_CHARACTERS,
    V8_AMOUNT_CHARACTERS,
    V12_ONNX_OUTPUT_NAMES,
    UnifiedReaderConfig,
    _checkpoint_config,
    _assert_non_recipient_parameter_bytes,
    _full_validation_epoch_reason,
    _is_full_validation_epoch,
    _load_onnx_artifacts,
    _non_recipient_parameter_bytes,
    _parameter_only_initialization,
    _recipient_artifact_metadata,
    _recipient_charset_source,
    _recipient_only_expansion_label_override,
    _recipient_only_logits,
    _recipient_tail_loss_policy,
    _recipient_target_mode,
    _recipient_time_steps,
    _slot_order,
    build_unified_reader,
    evaluate_unified_onnx,
    export_unified_onnx,
    train_unified_reader,
)
from transfer_receipt_ai.ocr_unified_dataset import KIND_V12 as DATASET_KIND_V12
from transfer_receipt_ai.ocr_unified_dataset import V12_SLOT_ORDER, build_unified_dataset


# A valid 1x1 opaque PNG.  The test needs real image files for the complete
# preprocessing/training/export path but not meaningful visual content.
_TINY_PNG = bytes.fromhex(
    "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c489"
    "0000000b49444154789c63f80f040009fb03fdfb5e6b2b0000000049454e44ae426082"
)


def _tiny_v12_config() -> UnifiedReaderConfig:
    """Keep the dual-input graph small enough for a fast CPU-only unit test."""
    return UnifiedReaderConfig(
        architecture_version=12,
        image_height=32,
        image_width=64,
        base_channels=8,
        numeric_hidden_size=16,
        payment_hidden_size=16,
        recipient_hidden_size=16,
        recipient_input_height=32,
        recipient_input_width=128,
        recipient_branch_channels=8,
        pooled_width=2,
    )


def _tiny_v12_model(config: UnifiedReaderConfig):
    torch = pytest.importorskip("torch")
    torch.manual_seed(7)
    model = build_unified_reader(
        payment_vocab_size=6,
        payment_bank_prefix_vocab_size=2,
        recipient_vocab_size=9,
        config=config,
    )
    model.eval()
    return torch, model


def _write_v12_source_manifest(tmp_path: Path) -> Path:
    """Write one complete, anchored five-field receipt per data split.

    Keep every recipient target in the train charset.  This is an ABI/lifecycle
    test rather than an OOV-quality test, so it should fail only when the
    v12 dual-input protocol regresses.
    """
    source = tmp_path / "teacher-labels"
    records: list[dict[str, object]] = []
    index = 0
    for split in ("train", "val", "test"):
        values = (
            ("amount", "¥100.00", "¥100.00"),
            ("time", "12:06", "12:06"),
            ("transfer_status", "转账成功", "success"),
            ("payment_method_field", "付款方式 建设银行储蓄卡(3667)", "bank_card"),
            ("recipient_field", "收款方 商户甲", "商户甲"),
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
                    "id": f"v12-{split}-{index}",
                    "image": image_name,
                    "field": field,
                    "text": text,
                    "paddle_text": text,
                    "semantic_value": semantic_value,
                    "paddle_confidence": 0.99,
                    "detector_score": 0.95,
                    # One stable result JSON per split makes these five rows
                    # one receipt in the unified manifest.
                    "result_json": f"D:/teacher/{split}.json",
                    "source": f"D:/source/{split}.png",
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


def test_v12_forward_requires_private_high_resolution_recipient_input() -> None:
    config = _tiny_v12_config()
    torch, model = _tiny_v12_model(config)
    field_images = torch.zeros((2, len(_slot_order(config)), 1, 32, 64), dtype=torch.float32)

    with pytest.raises(ValueError, match="requires recipient_value_image"):
        model(field_images)
    with pytest.raises(ValueError, match=r"\[batch,1,32,128\]"):
        model(field_images, torch.zeros((2, 1, 32, 64), dtype=torch.float32))

    with torch.no_grad():
        outputs = model(field_images, torch.ones((2, 1, 32, 128), dtype=torch.float32))

    assert len(outputs) == len(V12_ONNX_OUTPUT_NAMES) == 15
    # The private high-resolution branch downsamples the 128px recipient
    # input horizontally by four, independently of the 64px field tensor.
    assert list(outputs[-1].shape) == [32, 2, 9]


def test_v12_recipient_logits_ignore_reserved_fifth_slot_but_use_private_input() -> None:
    config = _tiny_v12_config()
    torch, model = _tiny_v12_model(config)
    field_images = torch.randn((1, len(_slot_order(config)), 1, 32, 64), dtype=torch.float32)
    recipient_value = torch.ones((1, 1, 32, 128), dtype=torch.float32)
    fifth_slot_changed = field_images.clone()
    fifth_slot_changed[:, 4] = torch.zeros_like(fifth_slot_changed[:, 4])
    recipient_value_changed = recipient_value.clone()
    recipient_value_changed[:, :, 6:26, 20:108] = 0.0

    with torch.no_grad():
        original = model(field_images, recipient_value)[-1]
        fifth_slot_only = model(fifth_slot_changed, recipient_value)[-1]
        private_input_changed = model(field_images, recipient_value_changed)[-1]

    # v12 reserves the legacy fifth low-resolution input for ABI stability;
    # it must not leak into the merchant CTC head.
    torch.testing.assert_close(original, fifth_slot_only, rtol=0.0, atol=0.0)
    # Conversely, the dedicated high-resolution value view is a real model
    # input, rather than sidecar-only metadata.
    assert not torch.equal(original, private_input_changed)


def test_v12_recipient_only_logits_match_the_full_private_branch() -> None:
    """The accelerated training path must preserve recipient CTC logits exactly."""
    config = _tiny_v12_config()
    torch, model = _tiny_v12_model(config)
    field_images = torch.randn((2, len(_slot_order(config)), 1, 32, 64), dtype=torch.float32)
    recipient_value = torch.randn((2, 1, 32, 128), dtype=torch.float32)

    with torch.no_grad():
        full_logits = model(field_images, recipient_value)[-1]
        private_logits = _recipient_only_logits(model, recipient_value, config=config)

    torch.testing.assert_close(full_logits, private_logits, rtol=0.0, atol=0.0)


def test_v12_light_augmentation_sees_each_epoch_with_persistent_spawn_workers(tmp_path: Path) -> None:
    """A persistent Windows-style worker must not keep the first epoch's crop.

    This deliberately uses ``spawn`` even on platforms that default to fork,
    because Windows DataLoader workers receive a pickled dataset.  The worker
    output must exactly match the existing deterministic augmentation formula
    for both epochs; only the epoch transport changes.
    """
    torch = pytest.importorskip("torch")
    config = _tiny_v12_config()
    image_path = tmp_path / "recipient.png"
    image_path.write_bytes(_TINY_PNG)
    record: dict[str, object] = {
        "id": "persistent-epoch-recipient",
        "slots": {"recipient_field": {"image_path": str(image_path), "text": "商户甲"}},
    }
    policy = ocr_unified._recipient_train_augmentation_policy(mode="light_v1", seed=42)
    dataset = ocr_unified._UnifiedReceiptDataset(
        [record],
        config=config,
        recipient_train_augmentation_policy=policy,
        recipient_only=True,
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=1,
        collate_fn=ocr_unified._collate_recipient_only,
        persistent_workers=True,
        multiprocessing_context="spawn",
    )
    try:
        dataset.set_epoch(1)
        first_images, first_records = list(loader)[0]
        dataset.set_epoch(2)
        second_images, second_records = list(loader)[0]
    finally:
        # Persistent workers otherwise outlive the short test iterator until
        # garbage collection, which is especially slow/flaky on Windows.
        iterator = getattr(loader, "_iterator", None)
        if iterator is not None:
            iterator._shutdown_workers()

    base = ocr_unified._recipient_value_input_tensor(record, config=config)
    expected_first = torch.from_numpy(
        ocr_unified._augment_recipient_value_input(base, record=record, policy=policy, epoch=1)
    ).unsqueeze(0)
    expected_second = torch.from_numpy(
        ocr_unified._augment_recipient_value_input(base, record=record, policy=policy, epoch=2)
    ).unsqueeze(0)

    assert [item["id"] for item in first_records] == [record["id"]]
    assert [item["id"] for item in second_records] == [record["id"]]
    torch.testing.assert_close(first_images, expected_first, rtol=0.0, atol=0.0)
    torch.testing.assert_close(second_images, expected_second, rtol=0.0, atol=0.0)
    assert not torch.equal(first_images, second_images)


def test_v12_sparse_full_validation_schedule_keeps_epoch_one_and_final() -> None:
    """A long recipient-only run may skip only interior non-N epochs."""
    planned = [
        epoch
        for epoch in range(1, 9)
        if _is_full_validation_epoch(epoch=epoch, epochs=8, validation_every=3)
    ]

    assert planned == [1, 3, 6, 8]
    assert _full_validation_epoch_reason(epoch=1, epochs=8, validation_every=3) == "epoch_1"
    assert _full_validation_epoch_reason(epoch=3, epochs=8, validation_every=3) == "every_n"
    assert _full_validation_epoch_reason(epoch=8, epochs=8, validation_every=3) == "final_epoch"
    assert _full_validation_epoch_reason(epoch=2, epochs=8, validation_every=3) == "scheduled_skip"


def test_v12_frozen_non_recipient_parameter_snapshot_is_byte_exact() -> None:
    """Guarded sparse validation must fail before evaluating mutated frozen state."""
    torch, model = _tiny_v12_model(_tiny_v12_config())
    model.register_buffer("financial_guard_probe", torch.tensor([1.0]))
    expected = _non_recipient_parameter_bytes(model)

    _assert_non_recipient_parameter_bytes(model, expected)
    with torch.no_grad():
        model.financial_guard_probe.add_(1.0)

    with pytest.raises(AssertionError, match="mutated frozen non-recipient parameters"):
        _assert_non_recipient_parameter_bytes(model, expected)


def _scheduled_validation_metrics(score: float) -> dict[str, object]:
    """Small complete metric payload for checkpoint-selection scheduling tests."""
    exact = {"records": 1, "exact_matches": 1, "exact_match": 1.0}
    return {
        "loss": 1.0 - score,
        "exact_match": score,
        "delivery_coverage": 1.0,
        "delivery_exact_match": score,
        "delivery_exact_overall": score,
        "delivery_false_accepts": 0,
        "verifier_exact_match": score,
        "verifier_macro_exact_match": score,
        "verifier_by_field": {},
        "candidate_text_exact_match": score,
        "candidate_text_macro_exact_match": score,
        "candidate_text_by_field": {
            "amount": dict(exact),
            "time": dict(exact),
            "payment_method_field": dict(exact),
            "recipient_field": dict(exact),
        },
        "ctc_by_field": {},
        "by_field": {},
        "status_non_success_to_success": 0,
    }


def test_v12_sparse_validation_skips_unvalidated_epochs_from_best_selection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Only full-validation epochs may score/select best.pt in guarded v12 mode."""
    torch = pytest.importorskip("torch")
    flat_manifest = _write_v12_source_manifest(tmp_path)
    unified_dir = tmp_path / "unified-v12"
    build_unified_dataset(
        records_path=flat_manifest,
        output_dir=unified_dir,
        architecture="v12",
    )
    config = _tiny_v12_config()
    seed, _, _, _ = _write_v12_expansion_seed(tmp_path, config=config, torch=torch)
    calls: list[int] = []
    checkpoint_writes: list[tuple[str, int]] = []
    original_write_checkpoint = ocr_unified._write_checkpoint

    def fake_evaluate(*_args: object, **_kwargs: object) -> dict[str, object]:
        calls.append(len(calls) + 1)
        return _scheduled_validation_metrics(0.50 + 0.10 * len(calls))

    def record_checkpoint_write(path: Path, payload: dict[str, object], *, torch: object) -> None:
        checkpoint_writes.append((path.name, int(payload["epoch"])))
        original_write_checkpoint(path, payload, torch=torch)

    monkeypatch.setattr(ocr_unified, "_evaluate_model", fake_evaluate)
    monkeypatch.setattr(ocr_unified, "_write_checkpoint", record_checkpoint_write)
    checkpoint = train_unified_reader(
        records_path=unified_dir / "unified_fields.jsonl",
        dataset_root=flat_manifest.parent,
        output_dir=tmp_path / "scheduled-run",
        config=config,
        device="cpu",
        epochs=5,
        batch_size=1,
        payment_bank_prefix_min_support=1,
        recipient_only_fine_tune=True,
        init_checkpoint=seed,
        init_checkpoint_mode=INIT_CHECKPOINT_MODE_RECIPIENT_ONLY_EXPANSION,
        validation_every=3,
    )

    summary = json.loads((checkpoint.parent / "training_summary.json").read_text(encoding="utf-8"))
    assert calls == [1, 2, 3]
    assert [record["validation_performed"] for record in summary["records"]] == [True, False, True, False, True]
    assert summary["records"][1]["checkpoint_selection_eligible"] is False
    assert summary["records"][1]["checkpoint_selection_score"] is None
    assert summary["records"][1]["checkpoint_protection"] is None
    assert summary["records"][3]["checkpoint_selection_eligible"] is False
    assert summary["best_checkpoint_epoch"] == 5
    assert summary["training_runtime"]["validation_every"] == 3
    assert [epoch for name, epoch in checkpoint_writes if name == "last.pt"] == [1, 3, 5]
    assert [epoch for name, epoch in checkpoint_writes if name == "best.pt"] == [1, 3, 5]


def _write_v12_expansion_seed(
    tmp_path: Path,
    *,
    config: UnifiedReaderConfig,
    torch: object,
) -> tuple[Path, dict[str, object], list[str], list[str]]:
    """Persist a minimal valid seed whose new recipient rows must be remapped."""
    source_payment = ["余"]
    source_recipient = ["乙", "甲"]
    source_bank = [PAYMENT_BANK_OTHER_CLASS, "建设银行"]
    model = build_unified_reader(
        payment_vocab_size=len(source_payment) + 1,
        payment_bank_prefix_vocab_size=len(source_bank),
        recipient_vocab_size=len(source_recipient) + 1,
        config=config,
    )
    source_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
    for index, value in enumerate(source_state.values(), start=1):
        value.fill_(float(index))
    payload = {
        "schema_version": 1,
        "kind": KIND_V12,
        "state_dict": source_state,
        "config": asdict(config),
        "amount_characters": list(V8_AMOUNT_CHARACTERS),
        "time_characters": list(V6_TIME_CHARACTERS),
        "payment_characters": source_payment,
        "recipient_characters": source_recipient,
        "recipient_blank_index": 0,
        "recipient_charset_sha256": hashlib.sha256("".join(source_recipient).encode("utf-8")).hexdigest(),
        "recipient_charset_source": _recipient_charset_source(config),
        "recipient_target": _recipient_target_mode(config),
        "recipient_sampling_policy": {
            "mode": "uniform",
            "recipient_sampling_weight": 1.0,
            "recipient_train_records": 1,
            "train_records": 1,
        },
        "recipient_confidence_policy": {
            "mode": "none",
            "low_confidence_threshold": None,
            "low_confidence_loss_weight": 1.0,
            "curriculum_epochs": 0,
        },
        "recipient_train_augmentation_policy": {"mode": "none"},
        "status_classes": list(STATUS_CLASSES),
        "payment_bank_prefix_classes": source_bank,
    }
    path = tmp_path / "v12-expansion-seed.pt"
    torch.save(payload, path)
    return path, source_state, source_payment, source_bank


def test_v12_recipient_only_expansion_maps_unicode_rows_and_locks_financial_maps(tmp_path: Path) -> None:
    """Inserted recipient characters must not shift old classifier row semantics."""
    torch = pytest.importorskip("torch")
    config = _tiny_v12_config()
    seed, source_state, source_payment, source_bank = _write_v12_expansion_seed(
        tmp_path,
        config=config,
        torch=torch,
    )
    target_recipient = ["丙", "乙", "甲"]
    effective_payment, effective_bank, effective_recipient, label_policy = _recipient_only_expansion_label_override(
        init_checkpoint=seed,
        config=config,
        amount_characters=list(V8_AMOUNT_CHARACTERS),
        time_characters=list(V6_TIME_CHARACTERS),
        payment_characters=["余", "额"],
        recipient_characters=target_recipient,
        payment_bank_prefix_classes=[PAYMENT_BANK_OTHER_CLASS, "交通银行", "建设银行"],
        torch=torch,
    )
    assert effective_payment == source_payment
    assert effective_bank == source_bank
    # The persisted map remains sorted even when old seed-only characters are
    # retained; the checkpoint loader remaps old rows by Unicode semantics.
    assert effective_recipient == ["丙", "乙", "甲"]
    assert label_policy["payment_character_map"]["identical"] is False
    assert label_policy["payment_bank_prefix_class_map"]["identical"] is False

    target_model = build_unified_reader(
        payment_vocab_size=len(effective_payment) + 1,
        payment_bank_prefix_vocab_size=len(effective_bank),
        recipient_vocab_size=len(effective_recipient) + 1,
        config=config,
    )
    target_state = {key: value.detach().clone() for key, value in target_model.state_dict().items()}
    initial_new_row = target_state["recipient_classifier.weight"][1].clone()
    with pytest.raises(ValueError, match="recipient character map"):
        _parameter_only_initialization(
            init_checkpoint=seed,
            config=config,
            amount_characters=list(V8_AMOUNT_CHARACTERS),
            time_characters=list(V6_TIME_CHARACTERS),
            payment_characters=effective_payment,
            recipient_characters=target_recipient,
            payment_bank_prefix_classes=effective_bank,
            torch=torch,
        )
    mapped_state, initialization = _parameter_only_initialization(
        init_checkpoint=seed,
        init_checkpoint_mode=INIT_CHECKPOINT_MODE_RECIPIENT_ONLY_EXPANSION,
        config=config,
        amount_characters=list(V8_AMOUNT_CHARACTERS),
        time_characters=list(V6_TIME_CHARACTERS),
        payment_characters=effective_payment,
        recipient_characters=effective_recipient,
        payment_bank_prefix_classes=effective_bank,
        torch=torch,
        target_state_dict=target_state,
    )

    assert initialization["mode"] == "parameter_only_recipient_unicode_expansion"
    assert initialization["recipient_classifier_row_mapping"]["shared_character_rows_copied"] == 2
    assert initialization["recipient_classifier_row_mapping"]["new_target_character_rows_kept_at_seed"] == 1
    assert mapped_state is not None
    # Blank, 乙, and 甲 come from their semantic source rows despite their
    # reordering in the sorted union. 丙 is new and remains deterministic init.
    torch.testing.assert_close(mapped_state["recipient_classifier.weight"][0], source_state["recipient_classifier.weight"][0])
    torch.testing.assert_close(mapped_state["recipient_classifier.weight"][2], source_state["recipient_classifier.weight"][1])
    torch.testing.assert_close(mapped_state["recipient_classifier.weight"][3], source_state["recipient_classifier.weight"][2])
    torch.testing.assert_close(mapped_state["recipient_classifier.weight"][1], initial_new_row)
    torch.testing.assert_close(mapped_state["recipient_classifier.bias"][2], source_state["recipient_classifier.bias"][1])
    for key, source_value in source_state.items():
        if key not in {"recipient_classifier.weight", "recipient_classifier.bias"}:
            torch.testing.assert_close(mapped_state[key], source_value, rtol=0.0, atol=0.0)
    target_model.load_state_dict(mapped_state, strict=True)


def test_v12_recipient_only_expansion_retains_a_missing_seed_character(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    config = _tiny_v12_config()
    seed, _, _, _ = _write_v12_expansion_seed(tmp_path, config=config, torch=torch)

    _, _, effective_recipient, label_policy = _recipient_only_expansion_label_override(
        init_checkpoint=seed,
        config=config,
        amount_characters=list(V8_AMOUNT_CHARACTERS),
        time_characters=list(V6_TIME_CHARACTERS),
        payment_characters=["余"],
        recipient_characters=["丙", "甲"],
        payment_bank_prefix_classes=[PAYMENT_BANK_OTHER_CLASS, "建设银行"],
        torch=torch,
    )

    assert effective_recipient == ["丙", "乙", "甲"]
    assert label_policy["recipient_character_map"]["checkpoint_characters_retained_not_in_current_train_count"] == 1


def test_v12_metadata_freezes_the_two_static_input_shapes() -> None:
    config = _tiny_v12_config()
    tail_policy = _recipient_tail_loss_policy(
        rare_character_max_support=3,
        rare_character_loss_weight=2.5,
        long_text_min_length=10,
        long_text_loss_weight=3.5,
        records=[],
    )
    restored = _checkpoint_config({"kind": KIND_V12, "config": asdict(config)})
    metadata = _recipient_artifact_metadata(
        config,
        recipient_sampling_policy={
            "mode": "weighted_receipt_sampler_v1",
            "recipient_sampling_weight": 2.0,
            "recipient_train_records": 3,
            "train_records": 8,
            "replacement": True,
            "seed": 42,
        },
        recipient_tail_loss_policy=tail_policy,
    )
    legacy_metadata = _recipient_artifact_metadata(
        config,
        recipient_sampling_policy={
            "mode": "uniform",
            "recipient_sampling_weight": 1.0,
            "recipient_train_records": 1,
            "train_records": 1,
        },
    )

    assert restored == config
    assert _slot_order(config) == (
        "amount",
        "time",
        "transfer_status",
        "payment_method_field",
        "recipient_field",
    )
    assert _recipient_time_steps(config) == 32
    assert metadata["recipient_input_name"] == "recipient_value_image"
    assert metadata["recipient_input_shape"] == [1, 1, 32, 128]
    assert metadata["recipient_time_steps"] == 32
    assert metadata["recipient_branch_channels"] == 8
    assert metadata["recipient_input_preprocess"] == "left_trim_then_centered_aspect_resize_high_resolution"
    assert metadata["recipient_tail_loss_policy"] == tail_policy
    # Published checkpoints/sidecars before this train-only policy do not
    # contain it.  Missing legacy provenance must remain loadable.
    assert "recipient_tail_loss_policy" not in legacy_metadata


def test_v12_train_export_ort_load_and_evaluate_two_static_inputs_when_onnx_is_available(
    tmp_path: Path,
) -> None:
    """Exercise v12 from teacher rows through one dual-input ONNX session.

    The fifth legacy field slot is preserved for deployment ABI stability, but
    recipient pixels must be supplied through ``recipient_value_image`` in the
    *same* ONNX ``session.run``.  Checking the real exported session catches a
    sidecar-only or accidentally single-input implementation.
    """
    pytest.importorskip("torch")
    onnx = pytest.importorskip("onnx")
    ort = pytest.importorskip("onnxruntime")
    pytest.importorskip("PIL")

    flat_manifest = _write_v12_source_manifest(tmp_path)
    unified_dir = tmp_path / "unified-v12"
    dataset_summary = build_unified_dataset(
        records_path=flat_manifest,
        output_dir=unified_dir,
        architecture="v12",
    )
    records_path = unified_dir / "unified_fields.jsonl"
    assert dataset_summary["kind"] == DATASET_KIND_V12
    assert dataset_summary["slot_order"] == list(V12_SLOT_ORDER)

    config = _tiny_v12_config()
    checkpoint = train_unified_reader(
        records_path=records_path,
        dataset_root=flat_manifest.parent,
        output_dir=tmp_path / "run-v12",
        config=config,
        device="cpu",
        epochs=1,
        batch_size=1,
        payment_bank_prefix_min_support=1,
        recipient_loss_weight=3.0,
        recipient_sampling_weight=2.0,
        recipient_tail_rare_character_max_support=1,
        recipient_tail_rare_character_loss_weight=2.5,
        recipient_tail_long_text_min_length=3,
        recipient_tail_long_text_loss_weight=3.5,
    )
    model_path, labels_path, contract_path = export_unified_onnx(
        checkpoint_path=checkpoint,
        output_path=tmp_path / "reader-v12.onnx",
    )

    onnx.checker.check_model(onnx.load_model(model_path))
    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    assert [item.name for item in session.get_inputs()] == ["field_images", "recipient_value_image"]
    assert [list(item.shape) for item in session.get_inputs()] == [[5, 1, 32, 64], [1, 1, 32, 128]]
    assert [item.name for item in session.get_outputs()] == list(V12_ONNX_OUTPUT_NAMES)

    # Both tensors are required by the exported graph and are provided in one
    # inference invocation.  This is intentionally not two model sessions.
    outputs = session.run(
        None,
        {
            "field_images": np.zeros((5, 1, 32, 64), dtype=np.float32),
            "recipient_value_image": np.zeros((1, 1, 32, 128), dtype=np.float32),
        },
    )
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    labels = json.loads(labels_path.read_text(encoding="utf-8"))
    assert contract["kind"] == KIND_V12
    assert [item["name"] for item in contract["inputs"]] == ["field_images", "recipient_value_image"]
    assert [item["shape"] for item in contract["inputs"]] == [[5, 1, 32, 64], [1, 1, 32, 128]]
    assert [list(value.shape) for value in outputs] == [
        contract["outputs"][name]["shape"] for name in V12_ONNX_OUTPUT_NAMES
    ]
    assert labels_path.is_file()

    loaded_config, _, loaded_contract = _load_onnx_artifacts(model_path)
    assert loaded_config == config
    assert loaded_contract["kind"] == KIND_V12
    assert loaded_contract["recipient_input_name"] == "recipient_value_image"
    expected_tail_policy = _recipient_tail_loss_policy(
        rare_character_max_support=1,
        rare_character_loss_weight=2.5,
        long_text_min_length=3,
        long_text_loss_weight=3.5,
        records=[{"slots": {"recipient_field": {"text": "商户甲"}}}],
    )
    assert labels["recipient_tail_loss_policy"] == expected_tail_policy
    assert contract["recipient_tail_loss_policy"] == expected_tail_policy

    summary, failures = evaluate_unified_onnx(
        model_path=model_path,
        records_path=records_path,
        dataset_root=flat_manifest.parent,
        output_dir=tmp_path / "eval-v12",
        split="test",
        device="cpu",
    )
    assert failures == []
    assert summary["providers"] == ["CPUExecutionProvider"]
    assert summary["by_field"]["recipient_field"]["records"] == 1
