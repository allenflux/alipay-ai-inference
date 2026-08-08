"""Contracts for the v13 production-full-crop recipient pilot."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import pytest

from transfer_receipt_ai.ocr_unified import (
    INIT_CHECKPOINT_MODE_RECIPIENT_FULL_CROP_WARMSTART,
    KIND_V13,
    STATUS_CLASSES,
    STATUS_TEXT_BLANK_INDEX,
    STATUS_TEXT_CHARSET_SOURCE,
    STATUS_TEXT_RUNTIME_POLICY,
    STATUS_TEXT_TARGET,
    V6_TIME_CHARACTERS,
    V8_AMOUNT_CHARACTERS,
    V12_ONNX_OUTPUT_NAMES,
    UnifiedReaderConfig,
    _recipient_artifact_metadata,
    _recipient_train_split_policy,
    _parameter_only_initialization,
    _recipient_only_expansion_label_override,
    _validate_recipient_full_crop_seed_policy,
    _validate_recipient_full_crop_warmstart_config,
    build_parser,
    build_unified_reader,
    train_unified_reader,
)
from transfer_receipt_ai.recipient_full_crop_pilot import (
    PILOT_EPOCHS,
    _fresh_output_path,
    evaluate_pilot_summary,
    target_config_from_seed,
    verify_blind_manifest_contract,
)
from transfer_receipt_ai.recipient_blind_manifest import build_blind_manifest
from transfer_receipt_ai.recipient_full_crop_seed_sanitizer import (
    ATTESTATION_KIND as SEED_SANITIZER_ATTESTATION_KIND,
    sanitize_recipient_full_crop_seed,
)


def _source_config() -> UnifiedReaderConfig:
    return UnifiedReaderConfig(
        architecture_version=13,
        image_height=32,
        image_width=64,
        base_channels=8,
        numeric_hidden_size=16,
        payment_hidden_size=16,
        recipient_hidden_size=16,
        recipient_value_left_trim=0.30,
        recipient_input_height=32,
        recipient_input_width=1536,
        recipient_branch_channels=8,
        recipient_open_text_layers=2,
        recipient_open_text_heads=4,
        recipient_open_text_feedforward=64,
        pooled_width=2,
    )


def _target_config() -> UnifiedReaderConfig:
    return UnifiedReaderConfig(
        **{**asdict(_source_config()), "recipient_value_left_trim": 0.0}
    )


def _model(config: UnifiedReaderConfig, recipient_vocab_size: int = 3):
    return build_unified_reader(
        payment_vocab_size=5,
        payment_bank_prefix_vocab_size=2,
        recipient_vocab_size=recipient_vocab_size,
        status_text_vocab_size=len(set("转账成功")) + 1,
        config=config,
    )


def _seed_source_payloads(
    config: UnifiedReaderConfig, state_dict: object
) -> tuple[dict[str, object], dict[str, object]]:
    recipient_characters = ["商", "户"]
    status_characters = sorted(set("转账成功"))
    policies = {
        "recipient_sampling_policy": {
            "mode": "uniform",
            "recipient_sampling_weight": 1.0,
            "recipient_train_records": 2,
            "train_records": 2,
        },
        "recipient_confidence_policy": {
            "mode": "none",
            "low_confidence_threshold": None,
            "low_confidence_loss_weight": 1.0,
            "curriculum_epochs": 0,
        },
        "recipient_tail_loss_policy": {
            "mode": "none",
            "rare_character_max_support": 0,
            "rare_character_loss_weight": 1.0,
            "long_text_min_length": 0,
            "long_text_loss_weight": 1.0,
            "recipient_train_records": 2,
            "rare_character_train_records": 0,
            "long_text_train_records": 0,
            "combined_boost_train_records": 0,
        },
        "recipient_train_augmentation_policy": {"mode": "none"},
    }
    state = dict(state_dict)
    status_keys = [key for key in state if key.startswith("status_text_")]
    source_config = asdict(config)
    source_config["architecture_version"] = 12
    status_payload: dict[str, object] = {
        "schema_version": 1,
        "kind": KIND_V13,
        "config": asdict(config),
        "state_dict": state,
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
        "recipient_train_split_policy": _recipient_train_split_policy(["train", "val"]),
        "recipient_loss_weight": 1.0,
        "recipient_oov_by_split": {
            split: {"records": 1, "oov_records": 0}
            for split in ("train", "val", "test")
        },
        "status_classes": list(STATUS_CLASSES),
        "status_text_characters": status_characters,
        "status_text_blank_index": STATUS_TEXT_BLANK_INDEX,
        "status_text_charset_sha256": hashlib.sha256(
            "".join(status_characters).encode("utf-8")
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
        "epoch": 9,
        "initialization": {
            "mode": "parameter_only_v12_to_v13_status_text_expansion",
            "source_kind": "receipt_unified_field_reader_v12",
            "source_config": source_config,
            "checkpoint_sha256": "a" * 64,
            "optimizer_restored": False,
            "epoch_reset": True,
            "new_parameter_prefix": "status_text_",
            "copied_legacy_tensor_count": len(state) - len(status_keys),
            "new_status_text_tensor_count": len(status_keys),
            "frozen_legacy_output_count": len(V12_ONNX_OUTPUT_NAMES),
            "financial_label_policy": {
                "mode": "checkpoint_legacy_label_maps_status_text_only_v1"
            },
        },
        "fine_tune_policy": {
            "mode": "status_text_only_v13",
            "trainable_parameter_prefix": "status_text_",
            "frozen_legacy_output_count": len(V12_ONNX_OUTPUT_NAMES),
            "full_validation_schedule": "epoch_1_every_n_and_final_epoch",
            "validation_every": 1,
        },
        "training_runtime": {
            "status_text_only_training": True,
            "recipient_only_private_branch_training": False,
            "full_validation_schedule": "epoch_1_every_n_and_final_epoch",
            "validation_every": 1,
            "recipient_train_split_policy": _recipient_train_split_policy(["train", "val"]),
        },
    }
    status_payload.update(policies)
    status_payload.update(
        _recipient_artifact_metadata(
            config,
            recipient_sampling_policy=policies["recipient_sampling_policy"],
            recipient_confidence_policy=policies["recipient_confidence_policy"],
            recipient_tail_loss_policy=policies["recipient_tail_loss_policy"],
            recipient_train_augmentation_policy=policies[
                "recipient_train_augmentation_policy"
            ],
        )
    )
    train_payload = {
        **status_payload,
        "kind": "receipt_unified_field_reader_v12",
        "config": source_config,
        "state_dict": {
            key: value for key, value in state.items() if not key.startswith("status_text_")
        },
        "recipient_train_split_policy": _recipient_train_split_policy(["train"]),
        "training_runtime": {
            "recipient_train_split_policy": _recipient_train_split_policy(["train"])
        },
        "initialization": {
            "mode": "random",
            "optimizer_restored": False,
            "epoch_reset": True,
        },
    }
    for key in [key for key in train_payload if key.startswith("status_text_")]:
        del train_payload[key]
    for key in ("fine_tune_policy",):
        del train_payload[key]
    return status_payload, train_payload


def _write_seed(tmp_path: Path):
    torch = pytest.importorskip("torch")
    source = _source_config()
    model = _model(source)
    state = {name: value.detach().clone() for name, value in model.state_dict().items()}
    for index, value in enumerate(state.values(), start=1):
        value.fill_(float(index) / 100.0)
    status_payload, train_payload = _seed_source_payloads(source, state)
    status_checkpoint = tmp_path / "v13-status-source.pt"
    train_checkpoint = tmp_path / "v12-train-only-source.pt"
    checkpoint = tmp_path / "v13-trim30.pt"
    torch.save(status_payload, status_checkpoint)
    torch.save(train_payload, train_checkpoint)
    sanitize_recipient_full_crop_seed(
        status_checkpoint=status_checkpoint,
        train_only_recipient_checkpoint=train_checkpoint,
        output_checkpoint=checkpoint,
        torch=torch,
    )
    return torch, checkpoint, state


def test_full_crop_config_allows_only_v13_trim_30_to_zero() -> None:
    source = _source_config()
    target = _target_config()
    _validate_recipient_full_crop_warmstart_config(source, target)

    with pytest.raises(ValueError, match="v13 source and target"):
        _validate_recipient_full_crop_warmstart_config(
            UnifiedReaderConfig(**{**asdict(source), "architecture_version": 12}),
            target,
        )
    with pytest.raises(ValueError, match="0.30-trim"):
        _validate_recipient_full_crop_warmstart_config(
            UnifiedReaderConfig(**{**asdict(source), "recipient_value_left_trim": 0.20}),
            target,
        )
    with pytest.raises(ValueError, match="target recipient_value_left_trim=0"):
        _validate_recipient_full_crop_warmstart_config(
            source,
            UnifiedReaderConfig(**{**asdict(target), "recipient_value_left_trim": 0.10}),
        )
    with pytest.raises(ValueError, match="incompatible config fields: recipient_hidden_size"):
        _validate_recipient_full_crop_warmstart_config(
            source,
            UnifiedReaderConfig(**{**asdict(target), "recipient_hidden_size": 24}),
        )


def test_full_crop_seed_requires_content_bound_sanitizer_not_top_level_policy() -> None:
    payload = {
        "kind": KIND_V13,
        "recipient_train_split_policy": _recipient_train_split_policy(["train"]),
    }
    with pytest.raises(ValueError, match="content-bound seed sanitizer attestation"):
        _validate_recipient_full_crop_seed_policy(payload)


def test_blind_contract_is_hash_bound_and_physically_excludes_test(tmp_path: Path) -> None:
    full = tmp_path / "full.jsonl"
    rows = [
        {"id": "train-one", "split": "train", "slots": {}},
        {"id": "val-one", "split": "val", "slots": {}},
        {"id": "test-secret", "split": "test", "slots": {"recipient_field": {"text": "绝密"}}},
    ]
    full.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    blind = tmp_path / "blind.jsonl"
    contract = tmp_path / "blind.contract.json"
    build_blind_manifest(source=full, output=blind, contract=contract)
    binding = verify_blind_manifest_contract(
        records_path=blind,
        blind_contract_path=contract,
    )
    assert binding["split_counts"] == {"train": 1, "val": 1, "test_excluded": 1}
    assert "test-secret" not in blind.read_text(encoding="utf-8")

    blind.write_text(
        blind.read_text(encoding="utf-8")
        + json.dumps(rows[-1], ensure_ascii=False)
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="changed after contract"):
        verify_blind_manifest_contract(records_path=blind, blind_contract_path=contract)


def test_output_no_clobber_rejects_existing_and_reparse_paths(tmp_path: Path) -> None:
    existing = tmp_path / "existing"
    existing.mkdir()
    with pytest.raises(ValueError, match="Refusing to reuse"):
        _fresh_output_path(existing)

    target = tmp_path / "not-created"
    link = tmp_path / "dangling-output"
    try:
        link.symlink_to(target, target_is_directory=True)
    except OSError as error:
        pytest.skip(f"test host cannot create a directory symlink: {error}")
    with pytest.raises(ValueError, match="existing, symlink, or reparse"):
        _fresh_output_path(link)
    assert not target.exists()

    real_parent = tmp_path / "real-parent"
    real_parent.mkdir()
    linked_parent = tmp_path / "linked-parent"
    linked_parent.symlink_to(real_parent, target_is_directory=True)
    with pytest.raises(ValueError, match="must not traverse"):
        _fresh_output_path(linked_parent / "out")
    assert not (real_parent / "out").exists()


def test_full_crop_warmstart_copies_seed_and_maps_only_new_unicode_rows(tmp_path: Path) -> None:
    torch, checkpoint, source_state = _write_seed(tmp_path)
    target_config = _target_config()
    payment, banks, recipient, policy = _recipient_only_expansion_label_override(
        init_checkpoint=checkpoint,
        init_checkpoint_mode=INIT_CHECKPOINT_MODE_RECIPIENT_FULL_CROP_WARMSTART,
        config=target_config,
        amount_characters=list(V8_AMOUNT_CHARACTERS),
        time_characters=list(V6_TIME_CHARACTERS),
        payment_characters=["新", "值"],
        recipient_characters=["商", "新"],
        payment_bank_prefix_classes=["__other__", "新银行"],
        torch=torch,
    )
    assert payment == ["卡", "行", "银", "储"]
    assert banks == ["__other__", "银行"]
    assert recipient == ["商", "户", "新"]
    assert policy["mode"] == "checkpoint_financial_label_maps_recipient_full_crop_warmstart_v1"
    assert policy["source_recipient_value_left_trim"] == pytest.approx(0.30)
    assert policy["target_recipient_value_left_trim"] == pytest.approx(0.0)
    assert policy["recipient_character_map"]["mode"] == "checkpoint_base_plus_train_only_additions_v1"

    target_model = _model(target_config, recipient_vocab_size=len(recipient) + 1)
    fresh_state = {
        name: value.detach().clone() for name, value in target_model.state_dict().items()
    }
    state, provenance = _parameter_only_initialization(
        init_checkpoint=checkpoint,
        init_checkpoint_mode=INIT_CHECKPOINT_MODE_RECIPIENT_FULL_CROP_WARMSTART,
        config=target_config,
        amount_characters=list(V8_AMOUNT_CHARACTERS),
        time_characters=list(V6_TIME_CHARACTERS),
        payment_characters=payment,
        recipient_characters=recipient,
        status_text_characters=sorted(set("转账成功")),
        payment_bank_prefix_classes=banks,
        torch=torch,
        target_state_dict=target_model.state_dict(),
    )
    assert state is not None
    assert provenance["mode"] == "parameter_only_recipient_full_crop_warmstart"
    assert provenance["source_recipient_value_left_trim"] == pytest.approx(0.30)
    assert provenance["target_recipient_value_left_trim"] == pytest.approx(0.0)
    assert provenance["source_recipient_train_split_policy"]["splits"] == ["train"]
    assert provenance["recipient_classifier_row_mapping"]["new_target_character_rows_kept_at_seed"] == 1
    for name, source_value in source_state.items():
        if name in {"recipient_classifier.weight", "recipient_classifier.bias"}:
            continue
        torch.testing.assert_close(state[name], source_value, rtol=0.0, atol=0.0)
    for key in ("recipient_classifier.weight", "recipient_classifier.bias"):
        torch.testing.assert_close(state[key][0], source_state[key][0], rtol=0.0, atol=0.0)
        for source_index, character in enumerate(("商", "户"), start=1):
            target_index = recipient.index(character) + 1
            torch.testing.assert_close(
                state[key][target_index],
                source_state[key][source_index],
                rtol=0.0,
                atol=0.0,
            )
    new_row = recipient.index("新") + 1
    torch.testing.assert_close(
        state["recipient_classifier.weight"][new_row],
        fresh_state["recipient_classifier.weight"][new_row],
        rtol=0.0,
        atol=0.0,
    )
    target_model.load_state_dict(state, strict=True)


def test_target_config_is_derived_from_seed_without_other_changes(tmp_path: Path) -> None:
    torch, checkpoint, _ = _write_seed(tmp_path)
    target = target_config_from_seed(checkpoint, torch=torch)
    source_values = asdict(_source_config())
    target_values = asdict(target)
    assert target_values.pop("recipient_value_left_trim") == pytest.approx(0.0)
    assert source_values.pop("recipient_value_left_trim") == pytest.approx(0.30)
    assert target_values == source_values


def _summary(*, best: float = 0.80, epoch4: float = 0.76, epoch8: float = 0.79):
    records = []
    for epoch in range(PILOT_EPOCHS + 1):
        exact = epoch4 if epoch == 4 else epoch8 if epoch == 8 else 0.70
        if epoch == 5:
            exact = best
        records.append(
            {
                "epoch": epoch,
                "validation_performed": True,
                "val_candidate_text_by_field": {
                    "amount": {"exact_match": 0.80},
                    "time": {"exact_match": 0.99},
                    "payment_method_field": {"exact_match": 0.94},
                    "recipient_field": {"exact_match": exact},
                },
                "val_ctc_by_field": {"transfer_status": {"exact_match": 0.91}},
                "val_status_non_success_to_success": 0,
                "checkpoint_selection_eligible": True,
                "checkpoint_selection_protection_failures": [],
            }
        )
    return {
        "kind": KIND_V13,
        "config": asdict(_target_config()),
        "initialization": {
            "mode": "parameter_only_recipient_full_crop_warmstart",
            "init_checkpoint_mode": INIT_CHECKPOINT_MODE_RECIPIENT_FULL_CROP_WARMSTART,
            "source_kind": KIND_V13,
            "source_config": asdict(_source_config()),
            "source_recipient_train_split_policy": {
                "mode": "standard_train_only",
                "splits": ["train"],
            },
            "source_full_crop_seed_sanitizer_attestation": {
                "kind": SEED_SANITIZER_ATTESTATION_KIND,
                "analysis_only": True,
                "production_route_authorized": False,
                "optimizer_state_loaded": False,
                "external_test_artifacts_opened": False,
                "publication_policy": "same_directory_hard_link_no_clobber_v1",
                "topology_policy": "v12_v13_private_recipient_prefix_partition_v1",
                "compatibility": {
                    "status_architecture_version": 13,
                    "recipient_architecture_version": 12,
                    "only_config_difference": "architecture_version",
                    "recipient_input_width": 1536,
                },
                "state_proof": {
                    "non_recipient_source": "status_checkpoint",
                    "recipient_source": "train_only_recipient_checkpoint",
                },
            },
            "recipient_classifier_row_mapping": {
                "blank_row_copied": True,
                "shared_character_rows_copied": 2,
                "new_target_character_rows_kept_at_seed": 1,
                "checkpoint_character_count": 2,
                "target_character_count": 3,
            },
            "financial_label_policy": {
                "mode": "checkpoint_financial_label_maps_recipient_full_crop_warmstart_v1",
                "source_recipient_value_left_trim": 0.30,
                "target_recipient_value_left_trim": 0.0,
            },
        },
        "fine_tune_policy": {
            "mode": "recipient_only_v13",
            "trainable_parameter_prefix": "recipient_",
            "frozen_non_recipient_byte_guard": "before_every_full_validation",
            "frozen_non_recipient_state_entry_count": 10,
        },
        "training_runtime": {
            "device": "cuda:0",
            "uses_cuda": True,
            "cuda_device_name": "NVIDIA GeForce RTX 4090",
        },
        "recipient_train_split_policy": {"mode": "standard_train_only", "splits": ["train"]},
        "checkpoint_selection_policy": {
            "mode": "recipient_priority",
            "protected_minimum_candidate_exact": {
                "amount": 0.7885,
                "time": 0.9840,
                "payment_method_field": 0.9325,
            },
        },
        "field_counts": {
            field: {"train": 5, "val": 3, "test": 0}
            for field in (
                "amount",
                "time",
                "payment_method_field",
                "recipient_field",
                "transfer_status",
            )
        },
        "best_checkpoint_epoch": 5,
        "records": records,
    }


def test_pilot_decision_gate_is_fixed_and_analysis_only() -> None:
    decision = evaluate_pilot_summary(_summary())
    assert decision["passed"] is True
    assert decision["production_route_authorized"] is False
    assert decision["fixed_gates"]["minimum_best_recipient_exact"] == pytest.approx(0.75)
    assert decision["fixed_gates"]["minimum_epoch4_to_8_gain"] == pytest.approx(0.02)

    low = evaluate_pilot_summary(_summary(best=0.74, epoch4=0.70, epoch8=0.73))
    assert low["passed"] is False
    assert "best_recipient_below_75_percent" in low["failures"]
    flat = evaluate_pilot_summary(_summary(epoch4=0.78, epoch8=0.79))
    assert flat["passed"] is False
    assert "epoch4_to_8_gain_below_2pp" in flat["failures"]


def test_pilot_rejects_test_rows_and_any_non_trim_config_change() -> None:
    with_test = _summary()
    with_test["field_counts"]["recipient_field"]["test"] = 1
    with pytest.raises(ValueError, match="physically included test row"):
        evaluate_pilot_summary(with_test)

    changed = _summary()
    changed["config"]["recipient_hidden_size"] = 24
    # Summary-level checks bind the exact source/target relation as well as
    # the lower-level initializer validator.
    with pytest.raises(ValueError, match="does not prove"):
        evaluate_pilot_summary(changed)


def test_pilot_rejects_lowered_or_violated_protection_gates() -> None:
    lowered = _summary()
    lowered["checkpoint_selection_policy"]["protected_minimum_candidate_exact"][
        "amount"
    ] = 0.70
    with pytest.raises(ValueError, match="does not prove"):
        evaluate_pilot_summary(lowered)

    financial = _summary()
    financial["records"][3]["val_candidate_text_by_field"]["time"][
        "exact_match"
    ] = 0.90
    with pytest.raises(ValueError, match="violated the time floor"):
        evaluate_pilot_summary(financial)

    status = _summary()
    status["records"][8]["val_ctc_by_field"]["transfer_status"][
        "exact_match"
    ] = 0.89
    with pytest.raises(ValueError, match="visible-status floor"):
        evaluate_pilot_summary(status)


def test_full_crop_training_api_rejects_transductive_recipient_splits(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="train-split supervision only"):
        train_unified_reader(
            records_path=tmp_path / "not-opened.jsonl",
            dataset_root=tmp_path,
            output_dir=tmp_path / "not-created",
            config=_target_config(),
            recipient_train_splits=("train", "val"),
            recipient_only_fine_tune=True,
            init_checkpoint=tmp_path / "not-opened.pt",
            init_checkpoint_mode=INIT_CHECKPOINT_MODE_RECIPIENT_FULL_CROP_WARMSTART,
        )


def test_full_crop_training_api_rejects_a_physical_test_row_before_loading(tmp_path: Path) -> None:
    records = tmp_path / "unsafe.jsonl"
    records.write_text(
        json.dumps({"id": "secret", "split": "test", "slots": {}}, ensure_ascii=False)
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="physically excludes test rows"):
        train_unified_reader(
            records_path=records,
            dataset_root=tmp_path,
            output_dir=tmp_path / "not-created",
            config=_target_config(),
            recipient_train_splits=("train",),
            recipient_only_fine_tune=True,
            init_checkpoint=tmp_path / "not-opened.pt",
            init_checkpoint_mode=INIT_CHECKPOINT_MODE_RECIPIENT_FULL_CROP_WARMSTART,
        )
    assert not (tmp_path / "not-created").exists()


def test_cli_and_powershell_runner_are_hard_locked_to_blind_eight_epoch_pilot() -> None:
    args = build_parser().parse_args(
        [
            "train",
            "--records",
            "records.jsonl",
            "--output",
            "run",
            "--init-checkpoint-mode",
            INIT_CHECKPOINT_MODE_RECIPIENT_FULL_CROP_WARMSTART,
        ]
    )
    assert args.init_checkpoint_mode == INIT_CHECKPOINT_MODE_RECIPIENT_FULL_CROP_WARMSTART

    repo = Path(__file__).parents[1]
    runner = (repo / "scripts" / "receipt-ocr-recipient-full-crop-pilot-4090.ps1").read_text(
        encoding="utf-8"
    )
    assert "$pilotEpochs = 8" in runner
    assert "$recipientStopFloor = 0.75" in runner
    assert "$epoch4To8GainFloor = 0.02" in runner
    assert '$gpuRows[0] -notmatch "4090"' in runner
    assert "recipient_full_crop_warmstart" in runner
    assert "test_recipient_full_crop_seed_sanitizer.py" in runner
    assert "content-bound sanitizer attestation" in runner
    assert "complete hash-bound train-only lineage" in runner
    assert "transfer_receipt_ai.recipient_blind_manifest" in runner
    assert "transfer_receipt_ai.recipient_full_crop_pilot" in runner
    assert '"--blind-contract", $blindContractPath' in runner
    assert "Require-FreshNonReparseOutput $OutputRoot" in runner
    assert '"--split", "test"' not in runner
    assert "transfer_receipt_ai.ocr_unified\", \"export" not in runner
    assert "production_route_authorized -ne $false" in runner
