"""Contracts for the honest post-v13 recipient open-text candidate."""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import asdict
from pathlib import Path

import pytest

from transfer_receipt_ai.ocr_unified import (
    INIT_CHECKPOINT_MODE_RECIPIENT_VISUAL_CONTEXT_REINIT,
    KIND_V13,
    STATUS_CLASSES,
    STATUS_TEXT_BLANK_INDEX,
    STATUS_TEXT_CHARSET_SOURCE,
    STATUS_TEXT_RUNTIME_POLICY,
    STATUS_TEXT_TARGET,
    V13_ONNX_OUTPUT_NAMES,
    V6_TIME_CHARACTERS,
    V8_AMOUNT_CHARACTERS,
    UnifiedReaderConfig,
    _parameter_only_initialization,
    _recipient_only_expansion_label_override,
    _recipient_only_logits,
    _v12_recipient_export_probe,
    _validate_recipient_visual_context_reinit_config,
    build_unified_reader,
    export_unified_onnx,
)
from transfer_receipt_ai.recipient_blind_manifest import KIND as BLIND_KIND
from transfer_receipt_ai.recipient_blind_manifest import build_blind_manifest
from transfer_receipt_ai.recipient_final_gate import derive_gate_identity
from transfer_receipt_ai.recipient_final_gate import verify_test_summary


def _legacy_config() -> UnifiedReaderConfig:
    return UnifiedReaderConfig(
        architecture_version=13,
        image_height=32,
        image_width=64,
        base_channels=8,
        numeric_hidden_size=16,
        payment_hidden_size=16,
        recipient_hidden_size=16,
        recipient_input_height=32,
        recipient_input_width=128,
        recipient_branch_channels=8,
        recipient_open_text_layers=2,
        recipient_open_text_heads=4,
        recipient_open_text_feedforward=64,
        pooled_width=2,
    )


def _candidate_config() -> UnifiedReaderConfig:
    return UnifiedReaderConfig(
        **{
            **asdict(_legacy_config()),
            "recipient_backbone": "residual_positional_transformer_v2",
            "recipient_open_text_layers": 3,
            "recipient_open_text_dropout": 0.10,
        }
    )


def _model(config: UnifiedReaderConfig):
    return build_unified_reader(
        payment_vocab_size=5,
        payment_bank_prefix_vocab_size=2,
        recipient_vocab_size=3,
        status_text_vocab_size=5,
        config=config,
    )


def _onnx_parity_fixture_model(config: UnifiedReaderConfig, *, torch: object):
    """Build a well-conditioned export fixture independent of process RNG state."""

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        model = _model(config).eval()
        # A random untrained three-class head can put two logits on an
        # artificial near-tie.  Keep the production exact-argmax policy and
        # give only this synthetic export fixture a deterministic margin.
        with torch.no_grad():
            model.recipient_classifier.weight.mul_(0.01)
            model.recipient_classifier.bias.copy_(
                torch.linspace(
                    -4.0,
                    4.0,
                    steps=model.recipient_classifier.bias.numel(),
                    dtype=model.recipient_classifier.bias.dtype,
                    device=model.recipient_classifier.bias.device,
                )
            )
    return model


def _v13_seed_payload(config: UnifiedReaderConfig, state_dict: object) -> dict[str, object]:
    recipient_characters = ["商", "户"]
    status_text_characters = sorted(set("转账成功"))
    return {
        "schema_version": 1,
        "kind": KIND_V13,
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
        "recipient_oov_by_split": {
            split: {"records": 1, "oov_records": 0}
            for split in ("train", "val", "test")
        },
        "status_classes": list(STATUS_CLASSES),
        "status_text_characters": status_text_characters,
        "status_text_blank_index": STATUS_TEXT_BLANK_INDEX,
        "status_text_charset_sha256": hashlib.sha256(
            "".join(status_text_characters).encode("utf-8")
        ).hexdigest(),
        "status_text_charset_source": STATUS_TEXT_CHARSET_SOURCE,
        "status_text_target": STATUS_TEXT_TARGET,
        "status_text_runtime_policy": STATUS_TEXT_RUNTIME_POLICY,
        "status_text_oov_by_split": {
            split: {
                "records": 1,
                "oov_records": 0,
                "oov_characters": 0,
                "examples": [],
            }
            for split in ("train", "val", "test")
        },
        "payment_bank_prefix_classes": ["__other__", "银行"],
        "epoch": 30,
    }


def test_blind_manifest_physically_excludes_test_rows(tmp_path: Path) -> None:
    source = tmp_path / "full.jsonl"
    rows = [
        {"id": "train-one", "split": "train", "slots": {"recipient_field": {"text": "甲"}}},
        {"id": "val-one", "split": "val", "slots": {"recipient_field": {"text": "乙"}}},
        {"id": "test-secret", "split": "test", "slots": {"recipient_field": {"text": "绝密"}}},
    ]
    source.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    output = tmp_path / "blind.jsonl"
    contract_path = tmp_path / "blind.contract.json"

    contract = build_blind_manifest(source=source, output=output, contract=contract_path)

    blind_rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert [row["split"] for row in blind_rows] == ["train", "val"]
    assert "test-secret" not in output.read_text(encoding="utf-8")
    assert contract["kind"] == BLIND_KIND
    assert contract["optimizer_supervision_splits"] == ["train"]
    assert contract["checkpoint_selection_splits"] == ["val"]
    assert contract["final_gate_only_splits"] == ["test"]
    assert contract["test_labels_used"] is False
    assert contract["test_metrics_computed"] is False
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        build_blind_manifest(source=source, output=output, contract=contract_path)


def test_residual_recipient_backbone_is_v13_only_and_distinct() -> None:
    source = _legacy_config()
    target = _candidate_config()
    _validate_recipient_visual_context_reinit_config(source, target)
    with pytest.raises(ValueError, match="v13 source and target"):
        _validate_recipient_visual_context_reinit_config(
            UnifiedReaderConfig(**{**asdict(source), "architecture_version": 12}),
            target,
        )
    with pytest.raises(ValueError, match="residual recipient backbone is supported only"):
        UnifiedReaderConfig(
            **{
                **asdict(target),
                "architecture_version": 12,
            }
        ).validate()


def test_visual_context_reinit_copies_every_non_recipient_v13_tensor(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    source_config = _legacy_config()
    target_config = _candidate_config()
    source_model = _model(source_config)
    target_model = _model(target_config)
    source_state = {name: value.detach().clone() for name, value in source_model.state_dict().items()}
    for index, value in enumerate(source_state.values(), start=1):
        value.fill_(float(index) / 100.0)
    checkpoint = tmp_path / "v13.pt"
    torch.save(_v13_seed_payload(source_config, source_state), checkpoint)
    _, _, blind_characters, label_policy = _recipient_only_expansion_label_override(
        init_checkpoint=checkpoint,
        init_checkpoint_mode=INIT_CHECKPOINT_MODE_RECIPIENT_VISUAL_CONTEXT_REINIT,
        config=target_config,
        amount_characters=list(V8_AMOUNT_CHARACTERS),
        time_characters=list(V6_TIME_CHARACTERS),
        payment_characters=["卡", "行", "银", "储"],
        recipient_characters=["商"],
        payment_bank_prefix_classes=["__other__", "银行"],
        torch=torch,
    )
    assert blind_characters == ["商"]
    assert label_policy["recipient_character_map"]["mode"] == (
        "fresh_train_only_reinitialized_recipient_v1"
    )
    assert label_policy["recipient_character_map"][
        "checkpoint_characters_discarded_for_blind_reinit_count"
    ] == 1
    fresh_target = {name: value.detach().clone() for name, value in target_model.state_dict().items()}
    status_characters = sorted(set("转账成功"))

    state, provenance = _parameter_only_initialization(
        init_checkpoint=checkpoint,
        init_checkpoint_mode=INIT_CHECKPOINT_MODE_RECIPIENT_VISUAL_CONTEXT_REINIT,
        config=target_config,
        amount_characters=list(V8_AMOUNT_CHARACTERS),
        time_characters=list(V6_TIME_CHARACTERS),
        payment_characters=["卡", "行", "银", "储"],
        recipient_characters=["商", "户"],
        status_text_characters=status_characters,
        payment_bank_prefix_classes=["__other__", "银行"],
        torch=torch,
        target_state_dict=target_model.state_dict(),
    )

    assert state is not None
    assert provenance["mode"] == "parameter_only_recipient_visual_context_reinit"
    mapping = provenance["recipient_visual_context_mapping"]
    assert mapping["source_backbone"] == "legacy_depthwise_gru_v1"
    assert mapping["target_backbone"] == "residual_positional_transformer_v2"
    for name, value in state.items():
        if name.startswith("recipient_"):
            torch.testing.assert_close(value, fresh_target[name], rtol=0.0, atol=0.0)
        else:
            torch.testing.assert_close(value, source_state[name], rtol=0.0, atol=0.0)
    target_model.load_state_dict(state, strict=True)


def test_residual_private_forward_matches_full_v13_recipient_output() -> None:
    torch = pytest.importorskip("torch")
    config = _candidate_config()
    model = _model(config).eval()
    field_images = torch.randn((2, 5, 1, 32, 64), dtype=torch.float32)
    recipient_images = torch.randn((2, 1, 32, 128), dtype=torch.float32)
    with torch.no_grad():
        full_recipient = model(field_images, recipient_images)[14]
        private_recipient = _recipient_only_logits(model, recipient_images, config=config)
    assert list(full_recipient.shape) == [16, 2, 3]
    torch.testing.assert_close(full_recipient, private_recipient, rtol=0.0, atol=0.0)


def test_candidate_and_final_wrappers_keep_test_out_of_selection() -> None:
    repo = Path(__file__).parents[1]
    candidate = (repo / "scripts" / "receipt-ocr-recipient-v14-candidate-4090.ps1").read_text(
        encoding="utf-8"
    )
    final_gate = (repo / "scripts" / "receipt-ocr-recipient-v14-final-gate-4090.ps1").read_text(
        encoding="utf-8"
    )
    assert '"--recipient-train-splits", "train"' in candidate
    assert '"--split", "val"' in candidate
    assert '"--split", "test"' not in candidate
    assert 'test_evaluated = $false' in candidate
    assert 'recipient_visual_context_reinit' in candidate
    assert 'residual_positional_transformer_v2' in candidate
    assert '[switch]$Pilot' in candidate
    assert '$pilotMinimumBestRecipient = 0.75' in candidate
    assert '$pilotMinimumEpoch4To8Gain = 0.02' in candidate
    assert 'PILOT STOP:' in candidate
    assert candidate.index('PILOT PASS:') < candidate.index('"-m", "transfer_receipt_ai.ocr_unified", "export"')
    assert '"--max-non-success-to-success", "0"' in candidate
    assert "[IO.FileMode]::CreateNew" in final_gate
    assert final_gate.count("[IO.FileMode]::CreateNew") >= 1
    assert "--split test" in final_gate
    gate_helper = (
        repo / "src" / "transfer_receipt_ai" / "recipient_final_gate.py"
    ).read_text(encoding="utf-8")
    assert "def _fails_fixed_floor" in gate_helper
    assert "recipient_field_not_strictly_above_floor" in gate_helper
    assert "if _fails_fixed_floor(name, metrics[name], floor)" in gate_helper
    assert "checkpoint_selection_used_test = $false" in final_gate
    assert "CommonApplicationData" in final_gate
    assert "gate_subject_id" in final_gate
    assert "evidence_identity" in final_gate
    assert ".attempt.json" in final_gate
    assert "TrustedFullManifestSha256" in final_gate
    assert "recipient_final_gate" in final_gate
    assert "FileShare]::Read" in final_gate
    assert "leased candidate reinspection" in final_gate
    assert "IsNaN" in final_gate and "IsInfinity" in final_gate
    assert "--min-amount-exact-match" in final_gate
    assert "--min-time-exact-match" in final_gate
    assert "--min-payment-exact-match" in final_gate
    assert "--min-recipient-exact-match" in final_gate
    assert "--min-status-exact-match" in final_gate
    assert "--max-non-success-to-success 0" in final_gate
    assert "malicious administrator" in final_gate
    assert "ChangeExtension" not in final_gate


def test_gate_subject_identity_is_path_independent_and_evidence_is_not_a_rerun_key(
    tmp_path: Path,
) -> None:
    first = tmp_path / "first"
    second = tmp_path / "copied"
    first.mkdir()
    model = first / "candidate.onnx"
    contract = first / "candidate.contract.json"
    labels = first / "candidate.labels.json"
    manifest = first / "full.jsonl"
    model.write_bytes(b"same-model-bytes")
    contract.write_bytes(b"same-contract-bytes")
    labels.write_bytes(b"same-label-bytes")
    manifest.write_bytes(b'{"id":"held-out","split":"test"}\n')
    evidence = {
        "checkpoint_sha256": "a" * 64,
        "val_summary_sha256": "b" * 64,
        "fixed_floors": {"recipient_field": 0.90},
    }
    original = derive_gate_identity(
        model=model,
        full_manifest=manifest,
        contract=contract,
        labels=labels,
        evidence_binding=evidence,
    )
    shutil.copytree(first, second)
    copied = derive_gate_identity(
        model=second / model.name,
        full_manifest=second / manifest.name,
        contract=second / contract.name,
        labels=second / labels.name,
        evidence_binding=evidence,
    )
    edited_evidence = derive_gate_identity(
        model=second / model.name,
        full_manifest=second / manifest.name,
        contract=second / contract.name,
        labels=second / labels.name,
        evidence_binding={**evidence, "val_summary_sha256": "c" * 64},
    )

    assert copied["gate_subject_id"] == original["gate_subject_id"]
    assert copied["evidence_identity"] == original["evidence_identity"]
    assert edited_evidence["gate_subject_id"] == original["gate_subject_id"]
    assert edited_evidence["evidence_identity"] != original["evidence_identity"]


def test_verified_test_summary_cross_checks_hashes_and_rejects_nonfinite_json(
    tmp_path: Path,
) -> None:
    paths: dict[str, Path] = {}
    for name in (
        "full_manifest",
        "blind_manifest",
        "blind_contract",
        "model",
        "contract",
        "labels",
        "checkpoint",
        "training_summary",
        "val_summary",
    ):
        path = tmp_path / f"{name}.bin"
        if name == "full_manifest":
            path.write_text(
                "".join(
                    json.dumps(
                        {
                            "id": f"test-{index}",
                            "split": "test",
                            "slots": {"recipient_field": {"text": "乙"}},
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                    for index in range(10)
                ),
                encoding="utf-8",
            )
        else:
            path.write_bytes((name + "-sealed").encode())
        paths[name] = path
    candidate_evidence = tmp_path / "candidate.json"
    candidate_evidence.write_text("{}\n", encoding="utf-8")
    hashes = {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in paths.items()}
    evidence_binding = {
        "schema_version": 1,
        "checkpoint_sha256": hashes["checkpoint"],
        "blind_manifest_sha256": hashes["blind_manifest"],
        "blind_contract_sha256": hashes["blind_contract"],
        "training_summary_sha256": hashes["training_summary"],
        "val_summary_sha256": hashes["val_summary"],
    }
    identity = derive_gate_identity(
        model=paths["model"],
        full_manifest=paths["full_manifest"],
        contract=paths["contract"],
        labels=paths["labels"],
        evidence_binding=evidence_binding,
    )
    inspection = tmp_path / "inspection.json"
    inspection.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "kind": "receipt_recipient_v14_final_gate_subject_v1",
                **identity,
                "candidate_evidence": str(candidate_evidence),
                "candidate_evidence_sha256": hashlib.sha256(candidate_evidence.read_bytes()).hexdigest(),
                "paths": {name: str(path) for name, path in paths.items()},
                "artifact_sha256": hashes,
                "evidence_binding": evidence_binding,
            }
        ),
        encoding="utf-8",
    )
    summary_payload: dict[str, object] = {
        "schema_version": 1,
        "kind": "receipt_unified_field_reader_teacher_parity_v1",
        "model_sha256": hashes["model"],
        "records_sha256": hashes["full_manifest"],
        "evaluation_split": "test",
        "providers": ["CUDAExecutionProvider"],
        "status_text_policy": {
            "runtime_policy": "decode_and_normalize_review_only",
            "review_value": "review",
        },
        "acceptance": {
            "min_amount_exact_match": 0.7885,
            "min_time_exact_match": 0.9840,
            "min_payment_exact_match": 0.9325,
            "min_recipient_exact_match": 0.90,
            "min_status_exact_match": 0.90,
            "max_non_success_to_success": 0,
            "requested": True,
            "passed": True,
            "failures": [],
        },
        "by_field": {
            "amount": {"records": 1, "raw_exact_match": 0.80},
            "time": {"records": 1, "raw_exact_match": 0.99},
            "payment_method_field": {"records": 1, "raw_exact_match": 0.94},
            "recipient_field": {
                "records": 10,
                "raw_exact_matches": 10,
                "raw_exact_match": 1.0,
            },
            "transfer_status": {
                "records": 1,
                "ctc_records": 1,
                "ctc_raw_exact_match": 0.91,
                "non_success_to_success": 0,
            },
        },
    }
    summary = tmp_path / "summary.json"
    summary.write_text(json.dumps(summary_payload), encoding="utf-8")
    verified = verify_test_summary(inspection=inspection, summary=summary)
    assert verified["passed"] is True
    assert verified["metrics"]["recipient_field"] == 1.0
    assert verified["recipient_records"] == 10
    assert verified["recipient_exact_matches"] == 10
    assert verified["recipient_candidate_coverage"] == 1.0

    for recipient_matches, expected_pass in (
        (9, False),
        (10, True),
    ):
        recipient_rate = recipient_matches / 10
        summary_payload["by_field"]["recipient_field"][
            "raw_exact_matches"
        ] = recipient_matches  # type: ignore[index]
        summary_payload["by_field"]["recipient_field"]["raw_exact_match"] = recipient_rate  # type: ignore[index]
        summary_payload["acceptance"]["passed"] = expected_pass  # type: ignore[index]
        summary_payload["acceptance"]["failures"] = (  # type: ignore[index]
            [] if expected_pass else ["recipient delivery requires strictly above 90%"]
        )
        summary.write_text(json.dumps(summary_payload), encoding="utf-8")
        boundary = verify_test_summary(inspection=inspection, summary=summary)
        assert boundary["passed"] is expected_pass
        if expected_pass:
            assert boundary["failures"] == []
        else:
            assert "recipient_field_not_strictly_above_floor" in boundary["failures"]

    inspection_payload = json.loads(inspection.read_text(encoding="utf-8"))
    inspection_payload["gate_subject_id"] = "f" * 64
    inspection.write_text(json.dumps(inspection_payload), encoding="utf-8")
    with pytest.raises(ValueError, match="inspection gate_subject_id"):
        verify_test_summary(inspection=inspection, summary=summary)
    inspection_payload["gate_subject_id"] = identity["gate_subject_id"]
    inspection.write_text(json.dumps(inspection_payload), encoding="utf-8")

    summary_payload["acceptance"]["max_non_success_to_success"] = False  # type: ignore[index]
    summary.write_text(json.dumps(summary_payload), encoding="utf-8")
    with pytest.raises(ValueError, match="max_non_success_to_success"):
        verify_test_summary(inspection=inspection, summary=summary)
    summary_payload["acceptance"]["max_non_success_to_success"] = 0  # type: ignore[index]

    original_contract = paths["contract"].read_bytes()
    paths["contract"].write_bytes(b"tampered")
    with pytest.raises(ValueError, match="contract changed"):
        verify_test_summary(inspection=inspection, summary=summary)
    paths["contract"].write_bytes(original_contract)

    summary_payload["by_field"]["recipient_field"]["raw_exact_match"] = float("nan")  # type: ignore[index]
    summary.write_text(json.dumps(summary_payload), encoding="utf-8")
    with pytest.raises(ValueError, match="non-finite JSON constant"):
        verify_test_summary(inspection=inspection, summary=summary)

    summary_payload["by_field"]["recipient_field"]["raw_exact_matches"] = 10  # type: ignore[index]
    summary_payload["by_field"]["recipient_field"]["raw_exact_match"] = 1.0  # type: ignore[index]
    overflow_json = json.dumps(summary_payload)[:-1] + ', "ignored_future_metric": 1e999}'
    summary.write_text(overflow_json, encoding="utf-8")
    with pytest.raises(ValueError, match="non-finite JSON number"):
        verify_test_summary(inspection=inspection, summary=summary)


def test_v14_onnx_parity_fixture_ignores_prior_rng_and_has_decision_margin() -> None:
    torch = pytest.importorskip("torch")
    config = _candidate_config()
    repeated_logits: list[list[object]] = []

    with torch.random.fork_rng(devices=[]):
        for perturbation_seed in (17, 29):
            torch.manual_seed(perturbation_seed)
            torch.rand(257)
            model = _onnx_parity_fixture_model(config, torch=torch)
            recipient_zero = torch.zeros((1, 1, 32, 128), dtype=torch.float32)
            recipient_probe = _v12_recipient_export_probe(recipient_zero, torch=torch)
            current_logits: list[object] = []
            with torch.no_grad():
                for recipient_input in (recipient_zero, recipient_probe):
                    logits = _recipient_only_logits(model, recipient_input, config=config)
                    top_two = torch.topk(logits, k=2, dim=-1).values
                    assert float((top_two[..., 0] - top_two[..., 1]).min()) > 2.0
                    current_logits.append(logits.detach().clone())
            repeated_logits.append(current_logits)

    for first, second in zip(repeated_logits[0], repeated_logits[1]):
        torch.testing.assert_close(first, second, rtol=0.0, atol=0.0)


def test_v14_real_onnx_export_shapes_and_internal_ort_parity_when_available(
    tmp_path: Path,
) -> None:
    torch = pytest.importorskip("torch")
    onnx = pytest.importorskip("onnx")
    ort = pytest.importorskip("onnxruntime")
    config = _candidate_config()
    model = _onnx_parity_fixture_model(config, torch=torch)
    checkpoint = tmp_path / "recipient-v14.pt"
    payload = _v13_seed_payload(config, model.state_dict())
    slot_order = ("amount", "time", "transfer_status", "payment_method_field", "recipient_field")
    payload.update(
        {
            "field_counts": {
                field: {split: 1 for split in ("train", "val", "test")}
                for field in slot_order
            },
            "status_class_counts": {
                split: {name: 1 for name in STATUS_CLASSES}
                for split in ("train", "val", "test")
            },
            "structured_target_counts": {},
        }
    )
    torch.save(payload, checkpoint)

    model_path, _labels_path, contract_path = export_unified_onnx(
        checkpoint_path=checkpoint,
        output_path=tmp_path / "recipient-v14.onnx",
    )
    onnx.checker.check_model(onnx.load_model(model_path))
    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    assert [item.name for item in session.get_inputs()] == ["field_images", "recipient_value_image"]
    assert [list(item.shape) for item in session.get_inputs()] == [[5, 1, 32, 64], [1, 1, 32, 128]]
    assert [item.name for item in session.get_outputs()] == list(V13_ONNX_OUTPUT_NAMES)
    contract_payload = json.loads(contract_path.read_text(encoding="utf-8"))
    outputs = session.run(
        None,
        {
            "field_images": __import__("numpy").zeros((5, 1, 32, 64), dtype="float32"),
            "recipient_value_image": __import__("numpy").zeros((1, 1, 32, 128), dtype="float32"),
        },
    )
    assert [list(value.shape) for value in outputs] == [
        contract_payload["outputs"][name]["shape"] for name in V13_ONNX_OUTPUT_NAMES
    ]
