"""Focused tests for v12's single-reader, dual-static-input OCR protocol."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest

from transfer_receipt_ai.ocr_unified import (
    KIND_V12,
    V12_ONNX_OUTPUT_NAMES,
    UnifiedReaderConfig,
    _checkpoint_config,
    _load_onnx_artifacts,
    _recipient_artifact_metadata,
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
    "0000000d49444154789c6360f8cfc00000030101874f5dc30000000049454e44ae426082"
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


def test_v12_metadata_freezes_the_two_static_input_shapes() -> None:
    config = _tiny_v12_config()
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
