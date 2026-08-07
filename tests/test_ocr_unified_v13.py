"""Focused compatibility and safety tests for visible status-text OCR v13."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from transfer_receipt_ai import ocr_unified
from transfer_receipt_ai.ocr_unified import (
    KIND_V12,
    KIND_V13,
    STATUS_CLASSES,
    STATUS_TEXT_BLANK_INDEX,
    STATUS_TEXT_CHARSET_SOURCE,
    STATUS_TEXT_RUNTIME_POLICY,
    STATUS_TEXT_TARGET,
    V12_ONNX_OUTPUT_NAMES,
    V13_ONNX_OUTPUT_NAMES,
    V6_TIME_CHARACTERS,
    V8_AMOUNT_CHARACTERS,
    UnifiedReaderConfig,
    _batch_loss,
    _comparison_metrics,
    _load_onnx_artifact_details,
    _parameter_only_initialization,
    _unified_acceptance_failures,
    _validate_status_text_oov_audit,
    build_unified_reader,
    export_unified_onnx,
)
from transfer_receipt_ai.ocr_unified_dataset import (
    KIND_V13 as DATASET_KIND_V13,
    V13_SLOT_ORDER,
    build_unified_dataset,
)


def _tiny_config(version: int) -> UnifiedReaderConfig:
    return UnifiedReaderConfig(
        architecture_version=version,
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


def _model(config: UnifiedReaderConfig, *, status_text_vocab_size: int | None = None):
    torch = pytest.importorskip("torch")
    return torch, build_unified_reader(
        payment_vocab_size=5,
        payment_bank_prefix_vocab_size=2,
        recipient_vocab_size=3,
        status_text_vocab_size=status_text_vocab_size,
        config=config,
    )


def _write_crop(path: Path, shade: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.full((20, 80, 3), shade, dtype=np.uint8)).save(path)


def _flat_status_record(
    *, index: int, split: str, text: str, paddle_text: str, semantic_value: str
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "id": f"status-{split}-{index}",
        "image": f"images/status-{split}-{index}.png",
        "field": "transfer_status",
        "text": text,
        "paddle_text": paddle_text,
        "semantic_value": semantic_value,
        "paddle_confidence": 0.99,
        "detector_score": 0.98,
        "result_json": f"D:/results/{split}-{index}.json",
        "source": f"D:/source/{split}-{index}.png",
        "group_id": f"receipt:{split}:{index}",
        "split": split,
        "label_source": "transaction_truth" if index == 3 else "paddle_pseudo",
    }


def test_v13_dataset_uses_checked_visible_text_and_audits_fallback(tmp_path: Path) -> None:
    source = tmp_path / "flat"
    rows = [
        _flat_status_record(
            index=1,
            split="train",
            text="  转账成功  ",
            paddle_text="交易成功",
            semantic_value="success",
        ),
        _flat_status_record(
            index=2,
            split="val",
            text="success",
            paddle_text="转账成功",
            semantic_value="success",
        ),
        _flat_status_record(
            index=3,
            split="test",
            text="success",
            paddle_text="success",
            semantic_value="success",
        ),
    ]
    for index, row in enumerate(rows):
        _write_crop(source / str(row["image"]), 40 + index)
    manifest = source / "pseudo_labels.jsonl"
    manifest.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )

    output = tmp_path / "v13"
    summary = build_unified_dataset(
        records_path=manifest,
        output_dir=output,
        architecture="v13",
    )
    unified = [
        json.loads(line)
        for line in (output / "unified_fields.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    by_split = {row["split"]: row["slots"]["transfer_status"] for row in unified}

    assert summary["kind"] == DATASET_KIND_V13
    assert summary["slot_order"] == list(V13_SLOT_ORDER)
    assert by_split["train"]["text"] == "转账成功"
    assert by_split["train"]["status_text_source"] == "record_text"
    assert by_split["val"]["text"] == "转账成功"
    assert by_split["val"]["status_text_source"] == "paddle_text_fallback"
    assert "text" not in by_split["test"]
    assert by_split["test"]["class_name"] == "success"
    assert "normalizes_to_unknown" in by_split["test"]["status_text_audit"]["reason"]
    assert summary["status_text_charset_source"] == STATUS_TEXT_CHARSET_SOURCE
    assert summary["status_text_target"] == STATUS_TEXT_TARGET
    assert summary["status_text_source_counts"] == {
        "paddle_text_fallback": 1,
        "record_text": 1,
    }
    expected_characters = sorted(set("转账成功"))
    assert summary["status_text_charset"] == expected_characters
    assert summary["status_text_charset_sha256"] == hashlib.sha256(
        "".join(expected_characters).encode("utf-8")
    ).hexdigest()
    assert (output / "status_text_charset.txt").read_text(encoding="utf-8") == (
        "".join(expected_characters) + "\n"
    )


def test_v13_appends_status_ctc_without_changing_v12_outputs() -> None:
    torch = pytest.importorskip("torch")
    torch.manual_seed(7)
    _, v12 = _model(_tiny_config(12))
    torch.manual_seed(11)
    _, v13 = _model(_tiny_config(13), status_text_vocab_size=6)
    target_state = v13.state_dict()
    for name, value in v12.state_dict().items():
        target_state[name].copy_(value)
    v13.load_state_dict(target_state, strict=True)
    v12.eval()
    v13.eval()
    fields = torch.randn((2, 5, 1, 32, 64), dtype=torch.float32)
    recipient = torch.randn((2, 1, 32, 128), dtype=torch.float32)

    with torch.no_grad():
        old_outputs = v12(fields, recipient)
        new_outputs = v13(fields, recipient)

    assert len(old_outputs) == len(V12_ONNX_OUTPUT_NAMES) == 15
    assert len(new_outputs) == len(V13_ONNX_OUTPUT_NAMES) == 16
    for old, new in zip(old_outputs, new_outputs[:15]):
        assert torch.equal(old, new)
    assert list(new_outputs[-1].shape) == [16, 2, 6]


def test_status_text_only_loss_masks_missing_visible_labels() -> None:
    torch = pytest.importorskip("torch")
    logits = torch.randn((12, 2, 5), dtype=torch.float32, requires_grad=True)
    records = [
        {"slots": {"transfer_status": {"text": "成功", "class_name": "success"}}},
        {"slots": {"transfer_status": {"class_name": "success"}}},
    ]
    loss, metrics = _batch_loss(
        None,
        None,
        None,
        None,
        records,
        amount_to_id={},
        time_to_id={},
        payment_to_id={},
        status_text_logits=logits,
        status_text_to_id={"成": 1, "功": 2},
        payment_bank_prefix_classes=None,
        payment_bank_class_weights=None,
        status_to_id={name: index for index, name in enumerate(STATUS_CLASSES)},
        status_criterion=None,
        status_enabled=False,
        payment_loss_weight=1.0,
        recipient_loss_weight=1.0,
        status_text_loss_weight=1.0,
        config=_tiny_config(13),
        structured_outputs=None,
        ctc_loss_weight=1.0,
        structured_loss_weight=1.0,
        torch=torch,
        status_text_only=True,
    )
    assert loss is not None
    assert metrics == {
        "transfer_status_text": {
            "loss": pytest.approx(float(loss.detach())),
            "used": 1,
            "oov": 0,
        }
    }
    loss.backward()
    assert logits.grad is not None
    assert torch.count_nonzero(logits.grad[:, 0, :]) > 0
    assert torch.count_nonzero(logits.grad[:, 1, :]) == 0


def _v12_seed_payload(config: UnifiedReaderConfig, state_dict: object) -> dict[str, object]:
    recipient_characters = ["商", "户"]
    return {
        "schema_version": 1,
        "kind": KIND_V12,
        "config": asdict(config),
        "state_dict": state_dict,
        "amount_characters": list(V8_AMOUNT_CHARACTERS),
        "time_characters": list(V6_TIME_CHARACTERS),
        "payment_characters": ["卡", "行", "银", "储"],
        "recipient_characters": recipient_characters,
        "recipient_blank_index": 0,
        "recipient_charset_sha256": hashlib.sha256(
            "".join(recipient_characters).encode("utf-8")
        ).hexdigest(),
        "recipient_charset_source": "train_only_anchored_recipient_value",
        "recipient_target": "anchored_recipient_value_with_dedicated_high_resolution_value_view",
        "recipient_sampling_policy": {
            "mode": "uniform",
            "recipient_sampling_weight": 1.0,
            "recipient_train_records": 1,
            "train_records": 1,
        },
        "status_classes": list(STATUS_CLASSES),
        "payment_bank_prefix_classes": ["__other__", "银行"],
        "epoch": 4,
    }


def _status_oov_audit() -> dict[str, dict[str, object]]:
    return {
        split: {
            "records": 1,
            "oov_records": 0,
            "oov_characters": 0,
            "examples": [],
        }
        for split in ("train", "val", "test")
    }


def test_v12_to_v13_warmstart_adds_only_status_text_parameters(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    source_config = _tiny_config(12)
    target_config = _tiny_config(13)
    _, source = _model(source_config)
    _, target = _model(target_config, status_text_vocab_size=6)
    checkpoint = tmp_path / "v12.pt"
    torch.save(_v12_seed_payload(source_config, source.state_dict()), checkpoint)
    fresh_status = {
        name: value.detach().clone()
        for name, value in target.state_dict().items()
        if name.startswith("status_text_")
    }

    state, provenance = _parameter_only_initialization(
        init_checkpoint=checkpoint,
        config=target_config,
        amount_characters=list(V8_AMOUNT_CHARACTERS),
        time_characters=list(V6_TIME_CHARACTERS),
        payment_characters=["卡", "行", "银", "储"],
        recipient_characters=["商", "户"],
        status_text_characters=sorted(set("转账成功")),
        payment_bank_prefix_classes=["__other__", "银行"],
        torch=torch,
        target_state_dict=target.state_dict(),
        allow_v12_status_text_expansion=True,
    )

    assert state is not None
    assert provenance["mode"] == "parameter_only_v12_to_v13_status_text_expansion"
    assert provenance["copied_legacy_tensor_count"] == len(source.state_dict())
    assert provenance["new_status_text_tensor_count"] == len(fresh_status)
    for name, value in source.state_dict().items():
        assert torch.equal(state[name], value)
    for name, value in fresh_status.items():
        assert torch.equal(state[name], value)
    assert set(state) == set(target.state_dict())


def test_v13_contract_constants_are_additive_and_review_only() -> None:
    assert V13_ONNX_OUTPUT_NAMES[:15] == V12_ONNX_OUTPUT_NAMES
    assert V13_ONNX_OUTPUT_NAMES[-1] == "status_text_logits"
    assert STATUS_TEXT_BLANK_INDEX == 0
    assert STATUS_TEXT_TARGET == "visible_transfer_status_text"
    assert STATUS_TEXT_RUNTIME_POLICY == "decode_and_normalize_review_only"
    assert KIND_V13 == "receipt_unified_field_reader_v13"


def test_v13_status_acceptance_uses_visible_ctc_exact_not_only_semantic_normalization() -> None:
    def field(raw: float, *, ctc: float | None = None) -> dict[str, object]:
        return {
            "raw_exact_match": raw,
            "ctc_raw_exact_match": ctc,
            "oov_reference_rate": 0.0,
            "non_success_to_success": 0,
            "delivery_coverage": 0.0,
            "delivery_exact_match": 0.0,
            "delivery_false_accepts": 0,
        }

    metrics = {
        "amount": field(1.0),
        "time": field(1.0),
        "payment_method_field": field(1.0),
        "recipient_field": field(1.0),
        # Semantic normalization can still be perfect when the visible OCR
        # drops the leading characters (for example 成功 vs 转账成功).
        "transfer_status": field(1.0, ctc=0.5),
    }
    failures = _unified_acceptance_failures(
        metrics,
        min_amount_exact_match=None,
        min_time_exact_match=None,
        min_payment_exact_match=None,
        min_recipient_exact_match=None,
        min_status_exact_match=0.9,
        max_payment_oov_rate=None,
        max_recipient_oov_rate=None,
        max_non_success_to_success=None,
        min_delivery_coverage=None,
        min_delivery_exact_match=None,
        max_delivery_false_accepts=None,
        status_visible_text_ctc=True,
    )
    assert failures == ["transfer_status: ctc_raw_exact_match=0.5000 < 0.9000"]


def test_v13_status_oov_audit_fails_closed() -> None:
    audit = _status_oov_audit()
    _validate_status_text_oov_audit(audit, source="test")

    malformed = json.loads(json.dumps(audit))
    malformed["train"]["oov_records"] = 1
    malformed["train"]["oov_characters"] = 1
    malformed["train"]["examples"] = [
        {"id": "train-1", "characters": "异", "text": "异常"}
    ]
    with pytest.raises(ValueError, match="train split"):
        _validate_status_text_oov_audit(malformed, source="test")
    with pytest.raises(ValueError, match="OOV audit"):
        _validate_status_text_oov_audit(None, source="test")


def test_v13_metrics_keep_visible_ctc_exact_separate_from_normalized_status() -> None:
    rows = []
    for visible_reference, visible_candidate in (
        ("转账成功", "转账成功"),
        ("交易成功", "转账成功"),
    ):
        rows.append(
            {
                "raw_exact": True,
                "ctc_reference_text": visible_reference,
                "ctc_candidate_text": visible_candidate,
                "ctc_raw_exact": visible_reference == visible_candidate,
                "reference_semantic": "success",
                "semantic_exact": True,
                "cer_edits": 0,
                "reference_characters": len(visible_reference),
                "reference_has_oov_character": False,
                "non_success_to_success": False,
                "delivery_text": "review",
                "candidate_text": "success",
                "delivery_raw_exact": False,
            }
        )

    metrics = _comparison_metrics(rows)
    assert metrics["raw_exact_match"] == 1.0
    assert metrics["ctc_records"] == 2
    assert metrics["ctc_raw_exact_match"] == 0.5


def test_v13_export_sidecars_bind_status_text_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    torch = pytest.importorskip("torch")
    config = _tiny_config(13)
    status_characters = sorted(set("转账成功"))
    _, model = _model(config, status_text_vocab_size=len(status_characters) + 1)
    checkpoint = tmp_path / "v13.pt"
    payload = _v12_seed_payload(config, model.state_dict())
    payload.update(
        {
            "kind": KIND_V13,
            "status_text_blank_index": STATUS_TEXT_BLANK_INDEX,
            "status_text_characters": status_characters,
            "status_text_charset_sha256": hashlib.sha256(
                "".join(status_characters).encode("utf-8")
            ).hexdigest(),
            "status_text_charset_source": STATUS_TEXT_CHARSET_SOURCE,
            "status_text_target": STATUS_TEXT_TARGET,
            "status_text_runtime_policy": STATUS_TEXT_RUNTIME_POLICY,
            "status_text_oov_by_split": _status_oov_audit(),
            "recipient_oov_by_split": {
                split: {"records": 1, "oov_records": 0}
                for split in ("train", "val", "test")
            },
            "field_counts": {
                field: {split: 1 for split in ("train", "val", "test")}
                for field in V13_SLOT_ORDER
            },
            # Complete legacy-class coverage deliberately proves that v13
            # still forces the superseded status_logits output review-only.
            "status_class_counts": {
                split: {name: 1 for name in STATUS_CLASSES}
                for split in ("train", "val", "test")
            },
            "structured_target_counts": {},
        }
    )
    torch.save(payload, checkpoint)

    def fake_export(_wrapper: object, _args: object, output: Path, **_kwargs: object) -> None:
        Path(output).write_bytes(b"fake-v13-onnx")

    monkeypatch.setattr(torch.onnx, "export", fake_export)
    monkeypatch.setattr(ocr_unified, "_validate_exported_onnx", lambda *_args, **_kwargs: None)

    model_path, labels_path, contract_path = export_unified_onnx(
        checkpoint_path=checkpoint,
        output_path=tmp_path / "v13.onnx",
    )
    labels = json.loads(labels_path.read_text(encoding="utf-8"))
    contract = json.loads(contract_path.read_text(encoding="utf-8"))

    assert model_path.read_bytes() == b"fake-v13-onnx"
    expected_hash = hashlib.sha256("".join(status_characters).encode("utf-8")).hexdigest()
    assert labels["status_text_blank_index"] == 0
    assert labels["status_text_characters"] == status_characters
    assert labels["status_text_charset_sha256"] == expected_hash
    assert labels["status_text_charset_source"] == STATUS_TEXT_CHARSET_SOURCE
    assert labels["status_text_target"] == STATUS_TEXT_TARGET
    assert labels["status_text_runtime_policy"] == STATUS_TEXT_RUNTIME_POLICY
    assert contract["kind"] == KIND_V13
    assert contract["status_text_charset_sha256"] == expected_hash
    assert contract["status_text_charset_source"] == STATUS_TEXT_CHARSET_SOURCE
    assert contract["status_text_target"] == STATUS_TEXT_TARGET
    assert contract["status_text_runtime_policy"] == STATUS_TEXT_RUNTIME_POLICY
    assert contract["status_head_policy"]["runtime_policy"] == "review_only"
    assert set(contract["outputs"]) == set(V13_ONNX_OUTPUT_NAMES)
    assert contract["outputs"]["status_text_logits"] == {
        "shape": [16, len(status_characters) + 1],
        "layout": "[time,class]",
        "decoder": "ctc_greedy",
        "blank_index": 0,
        "characters": "status_text_characters",
        "target": STATUS_TEXT_TARGET,
        "runtime_policy": STATUS_TEXT_RUNTIME_POLICY,
        "review_value": "review",
        "normalizer": "normalize_status",
    }
    loaded_config, _, _, loaded_contract = _load_onnx_artifact_details(model_path)
    assert loaded_config == config
    assert loaded_contract["kind"] == KIND_V13

    contract["outputs"]["status_text_logits"]["normalizer"] = "unsafe_passthrough"
    contract_path.write_text(
        json.dumps(contract, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="status-text output contract"):
        _load_onnx_artifact_details(model_path)
