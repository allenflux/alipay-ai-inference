"""Contracts for the fixed legacy full-crop B8 continuation."""

from __future__ import annotations

import hashlib
import inspect
import json
import math
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace

import pytest

import transfer_receipt_ai.ocr_unified as unified
import transfer_receipt_ai.recipient_full_crop_continuation as continuation
from transfer_receipt_ai.ocr_unified import (
    INIT_CHECKPOINT_MODE_RECIPIENT_FULL_CROP_CONTINUATION,
    KIND_V13,
    STATUS_CLASSES,
    UnifiedReaderConfig,
    _parameter_only_initialization,
    _recipient_only_expansion_label_override,
    _validate_recipient_full_crop_continuation_config,
)
from transfer_receipt_ai.recipient_full_crop_continuation import (
    AUTHORIZATION,
    AUTHORITY_KEY,
    CONTINUATION_EPOCHS,
    FIXED_SOURCE_ARTIFACTS,
    FIXED_SOURCE_SUBJECT_ID,
    MAXIMUM_BEST_TO_EPOCH8_GAP_MATCHES,
    MINIMUM_BEST_MATCHES,
    MINIMUM_EPOCH4_TO_8_GAIN_MATCHES,
    RECIPIENT_DENOMINATOR,
    SOURCE_BEST_EPOCH,
    SOURCE_KIND,
    SOURCE_RECIPIENT_MATCHES,
    _binding,
    _blind_recipient_validation_denominator,
    _file_identity,
    _fresh_file,
    _read_frozen_regular_file,
    _require_checkpoint_summary_metadata,
    _require_fixed_source_artifacts,
    _sanitizer_transitive_source_paths,
    _verify_binding,
    _verify_frozen_blind_manifest_contract,
    evaluate_continuation_summary,
    fixed_recipe,
)


def _config() -> UnifiedReaderConfig:
    return UnifiedReaderConfig(
        architecture_version=13,
        image_height=32,
        image_width=64,
        base_channels=8,
        numeric_hidden_size=16,
        payment_hidden_size=16,
        recipient_hidden_size=16,
        recipient_value_left_trim=0.0,
        recipient_input_height=32,
        recipient_input_width=1536,
        recipient_branch_channels=8,
        recipient_open_text_layers=2,
        recipient_open_text_heads=4,
        recipient_open_text_feedforward=64,
        recipient_backbone="legacy_depthwise_gru_v1",
        pooled_width=2,
    )


def _epoch_record(epoch: int, matches: int) -> dict[str, object]:
    return {
        "epoch": epoch,
        "validation_performed": True,
        "val_candidate_text_by_field": {
            "amount": {"records": 100, "exact_matches": 80, "exact_match": 0.80},
            "time": {"records": 100, "exact_matches": 99, "exact_match": 0.99},
            "payment_method_field": {
                "records": 100,
                "exact_matches": 94,
                "exact_match": 0.94,
            },
            "recipient_field": {
                "records": RECIPIENT_DENOMINATOR,
                "exact_matches": matches,
                "exact_match": matches / RECIPIENT_DENOMINATOR,
            },
        },
        "val_ctc_by_field": {
            "transfer_status": {"records": 100, "exact_matches": 91, "exact_match": 0.91}
        },
        "val_status_non_success_to_success": 0,
        "checkpoint_selection_eligible": True,
        "checkpoint_selection_protection_failures": [],
    }


def _summary(
    *,
    best_matches: int = MINIMUM_BEST_MATCHES,
    epoch4_matches: int = 5600,
    epoch8_matches: int = MINIMUM_BEST_MATCHES,
) -> dict[str, object]:
    records: list[dict[str, object]] = []
    for epoch in range(CONTINUATION_EPOCHS + 1):
        matches = SOURCE_RECIPIENT_MATCHES + epoch * 20
        if epoch == 0:
            matches = SOURCE_RECIPIENT_MATCHES
        elif epoch == 4:
            matches = epoch4_matches
        elif epoch == 5:
            matches = best_matches
        elif epoch == 8:
            matches = epoch8_matches
        records.append(_epoch_record(epoch, matches))
    authority = {
        "kind": SOURCE_KIND,
        "authorization": AUTHORIZATION,
        "analysis_only": True,
        "production_route_authorized": False,
        "source_best_epoch": SOURCE_BEST_EPOCH,
    }
    return {
        "kind": KIND_V13,
        "config": asdict(_config()),
        "initialization": {
            "mode": "parameter_only_recipient_full_crop_continuation_all_state_copy",
            "init_checkpoint_mode": INIT_CHECKPOINT_MODE_RECIPIENT_FULL_CROP_CONTINUATION,
            "optimizer_restored": False,
            "scheduler_restored": False,
            "sampler_state_restored": False,
            "best_history_restored": False,
            "source_epoch_restored": False,
            "epoch_reset": True,
            "all_state_tensor_count_copied": 20,
            "all_state_key_set_exact": True,
            "all_state_dtype_shape_exact": True,
            "source_full_crop_continuation_authority": authority,
            "financial_label_policy": {
                "mode": "checkpoint_all_label_maps_recipient_full_crop_continuation_v1"
            },
        },
        "fine_tune_policy": {
            "mode": "recipient_only_v13",
            "trainable_parameter_prefix": "recipient_",
            "frozen_non_recipient_byte_guard": "before_every_full_validation",
            "initialization_non_recipient_byte_guard": "before_epoch_zero_validation",
            "frozen_non_recipient_state_entry_count": 10,
            "validation_every": 1,
        },
        "training_runtime": {
            "device": "cuda:0",
            "uses_cuda": True,
            "cuda_device_name": "NVIDIA GeForce RTX 4090",
            "validation_every": 1,
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
        "recipient_train_augmentation_policy": {"mode": "robust_v2", "seed": 42},
        "recipient_confidence_policy": {
            "low_confidence_threshold": 0.95,
            "low_confidence_loss_weight": 0.50,
            "curriculum_epochs": 10,
        },
        "recipient_tail_loss_policy": {
            "rare_character_max_support": 3,
            "rare_character_loss_weight": 1.5,
            "long_text_min_length": 9,
            "long_text_loss_weight": 1.5,
        },
        "field_counts": {
            field: {
                "train": 10,
                "val": RECIPIENT_DENOMINATOR if field == "recipient_field" else 10,
                "test": 0,
            }
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


def test_continuation_mode_is_private_and_config_is_exact() -> None:
    assert INIT_CHECKPOINT_MODE_RECIPIENT_FULL_CROP_CONTINUATION in unified.INIT_CHECKPOINT_MODES
    assert INIT_CHECKPOINT_MODE_RECIPIENT_FULL_CROP_CONTINUATION in unified.RECIPIENT_ONLY_INIT_CHECKPOINT_MODES
    assert INIT_CHECKPOINT_MODE_RECIPIENT_FULL_CROP_CONTINUATION in unified.V13_PRIVATE_RECIPIENT_INIT_CHECKPOINT_MODES
    config = _config()
    _validate_recipient_full_crop_continuation_config(config, config)
    with pytest.raises(ValueError, match="exact source/target config"):
        _validate_recipient_full_crop_continuation_config(
            config, replace(config, recipient_hidden_size=24)
        )
    with pytest.raises(ValueError, match="trim-zero"):
        _validate_recipient_full_crop_continuation_config(
            replace(config, recipient_value_left_trim=0.30), config
        )
    with pytest.raises(ValueError, match="legacy recipient backbone"):
        _validate_recipient_full_crop_continuation_config(
            replace(config, recipient_backbone="residual_positional_transformer_v2"),
            replace(config, recipient_backbone="residual_positional_transformer_v2"),
        )


def test_continuation_refuses_missing_authority_before_manifest_io(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    checkpoint = tmp_path / "plain.pt"
    checkpoint.write_bytes(b"plain")
    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(
            is_available=lambda: True,
            get_device_name=lambda index: "NVIDIA GeForce RTX 4090",
        )
    )
    monkeypatch.setattr(unified, "_require_torch", lambda: (fake_torch, None))
    monkeypatch.setattr(unified, "_load_checkpoint", lambda *args, **kwargs: {"config": asdict(_config())})
    monkeypatch.setattr(
        unified,
        "_validate_recipient_full_crop_continuation_policy",
        lambda *args, **kwargs: (_ for _ in ()).throw(ValueError("missing embedded authority")),
    )
    monkeypatch.setattr(
        unified,
        "load_records",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("manifest was opened")),
    )
    with pytest.raises(ValueError, match="missing embedded authority"):
        unified.train_unified_reader(
            records_path=tmp_path / "must-not-open.jsonl",
            dataset_root=tmp_path,
            output_dir=tmp_path / "must-not-create",
            config=_config(),
            device="cuda:0",
            epochs=8,
            batch_size=10,
            learning_rate=0.0001,
            weight_decay=0.0001,
            recipient_low_confidence_threshold=0.95,
            recipient_low_confidence_loss_weight=0.50,
            recipient_confidence_curriculum_epochs=10,
            recipient_tail_rare_character_max_support=3,
            recipient_tail_rare_character_loss_weight=1.5,
            recipient_tail_long_text_min_length=9,
            recipient_tail_long_text_loss_weight=1.5,
            recipient_train_augmentation="robust_v2",
            recipient_only_fine_tune=True,
            validation_every=1,
            checkpoint_selection="recipient_priority",
            checkpoint_min_amount_candidate_exact=0.7885,
            checkpoint_min_time_candidate_exact=0.9840,
            checkpoint_min_payment_candidate_exact=0.9325,
            init_checkpoint=checkpoint,
            init_checkpoint_mode=INIT_CHECKPOINT_MODE_RECIPIENT_FULL_CROP_CONTINUATION,
            ctc_loss_weight=1.0,
            structured_loss_weight=1.0,
            seed=42,
            cuda_tf32=True,
            cudnn_benchmark=True,
        )
    assert not (tmp_path / "must-not-create").exists()


def test_continuation_mode_rejects_24_or_80_before_torch_or_manifest(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    checkpoint = tmp_path / "authorized.pt"
    checkpoint.write_bytes(b"placeholder")
    monkeypatch.setattr(
        unified,
        "_require_torch",
        lambda: (_ for _ in ()).throw(AssertionError("torch was opened")),
    )
    for epochs in (24, 80):
        with pytest.raises(ValueError, match="hard-locked to the audited fixed B8 recipe"):
            unified.train_unified_reader(
                records_path=tmp_path / "must-not-open.jsonl",
                dataset_root=tmp_path,
                output_dir=tmp_path / f"must-not-create-{epochs}",
                config=_config(),
                epochs=epochs,
                recipient_only_fine_tune=True,
                init_checkpoint=checkpoint,
                init_checkpoint_mode=INIT_CHECKPOINT_MODE_RECIPIENT_FULL_CROP_CONTINUATION,
            )
        assert not (tmp_path / f"must-not-create-{epochs}").exists()


def test_generic_train_parser_hides_private_continuation_mode() -> None:
    parser = unified.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "train",
                "--records",
                "records.jsonl",
                "--output",
                "out",
                "--init-checkpoint-mode",
                INIT_CHECKPOINT_MODE_RECIPIENT_FULL_CROP_CONTINUATION,
            ]
        )


@pytest.mark.parametrize(
    "lineage",
    [
        {AUTHORITY_KEY: {"analysis_only": True}},
        {
            "initialization": {
                "init_checkpoint_mode": INIT_CHECKPOINT_MODE_RECIPIENT_FULL_CROP_CONTINUATION,
                "source_full_crop_continuation_authority": {"analysis_only": True},
            }
        },
    ],
)
def test_export_rejects_analysis_only_continuation_lineage(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, lineage: dict[str, object]
) -> None:
    checkpoint = tmp_path / "analysis.pt"
    checkpoint.write_bytes(b"placeholder")
    monkeypatch.setattr(unified, "_require_torch", lambda: (object(), object()))
    monkeypatch.setattr(
        unified,
        "_load_checkpoint",
        lambda *args, **kwargs: lineage,
    )
    with pytest.raises(ValueError, match="cannot be exported to ONNX"):
        unified.export_unified_onnx(
            checkpoint_path=checkpoint, output_path=tmp_path / "forbidden.onnx"
        )
    assert not (tmp_path / "forbidden.onnx").exists()


@pytest.mark.parametrize(
    "lineage",
    [
        {AUTHORITY_KEY: {"analysis_only": True}},
        {
            "initialization": {
                "init_checkpoint_mode": INIT_CHECKPOINT_MODE_RECIPIENT_FULL_CROP_CONTINUATION
            }
        },
    ],
)
def test_strict_and_status_init_reject_continuation_source_or_child_lineage(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    lineage: dict[str, object],
) -> None:
    checkpoint = tmp_path / "analysis.pt"
    checkpoint.write_bytes(b"placeholder")
    monkeypatch.setattr(unified, "_load_checkpoint", lambda *args, **kwargs: lineage)
    with pytest.raises(ValueError, match="cannot be used by another init mode"):
        _parameter_only_initialization(
            init_checkpoint=checkpoint,
            init_checkpoint_mode=unified.INIT_CHECKPOINT_MODE_STRICT,
            config=_config(),
            amount_characters=["0"],
            time_characters=["0"],
            payment_characters=["卡"],
            recipient_characters=["商"],
            status_text_characters=["成"],
            payment_bank_prefix_classes=["__other__"],
            torch=object(),
            target_state_dict={},
        )
    with pytest.raises(ValueError, match="cannot seed status-text training"):
        unified._status_text_only_legacy_label_override(
            init_checkpoint=checkpoint,
            config=_config(),
            amount_characters=["0"],
            time_characters=["0"],
            payment_characters=["卡"],
            recipient_characters=["商"],
            payment_bank_prefix_classes=["__other__"],
            torch=object(),
        )


def test_b8_gate_exact_boundaries_and_pass_authorization() -> None:
    decision = evaluate_continuation_summary(
        _summary(
            best_matches=MINIMUM_BEST_MATCHES,
            epoch4_matches=MINIMUM_BEST_MATCHES - MINIMUM_EPOCH4_TO_8_GAIN_MATCHES,
            epoch8_matches=MINIMUM_BEST_MATCHES,
        )
    )
    assert decision["passed"] is True
    assert decision["analysis_only"] is True
    assert decision["production_route_authorized"] is False
    assert decision["test_opened"] is False
    assert decision["onnx_exported"] is False
    assert decision["pass_authorization"] == {
        "authorization": "fresh_exactly_16_from_original_pilot_best_only",
        "source": "original_pilot_best_not_b8_best",
        "source_best_epoch": 6,
        "epochs": 16,
        "fresh_optimizer": True,
        "validation_every": 1,
        "same_recipe": True,
        "required_final_best_matches": 6111,
        "required_recipient_denominator": 6789,
        "requires_strictly_greater_than_90_percent": True,
        "no_24_epoch_route": True,
        "no_80_epoch_route": True,
        "test_opened": False,
        "onnx_exported": False,
        "production_route_authorized": False,
    }

    below_best = evaluate_continuation_summary(
        _summary(best_matches=5789, epoch4_matches=5653, epoch8_matches=5789)
    )
    assert "best_recipient_below_5790_of_6789" in below_best["failures"]
    below_gain = evaluate_continuation_summary(
        _summary(best_matches=5790, epoch4_matches=5655, epoch8_matches=5790)
    )
    assert "epoch4_to_8_gain_below_136_matches" in below_gain["failures"]

    tail_boundary = evaluate_continuation_summary(
        _summary(best_matches=5857, epoch4_matches=5654, epoch8_matches=5790)
    )
    assert tail_boundary["passed"] is True
    tail_failure = evaluate_continuation_summary(
        _summary(best_matches=5858, epoch4_matches=5654, epoch8_matches=5790)
    )
    assert "best_to_epoch8_decay_above_67_matches" in tail_failure["failures"]


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda summary: summary["records"][0]["val_candidate_text_by_field"]["recipient_field"].update(records=6788), "count/rate"),
        (lambda summary: summary["records"][0]["val_candidate_text_by_field"]["recipient_field"].update(exact_matches=5467, exact_match=5467 / 6789), "epoch-zero identity"),
        (lambda summary: summary["records"][3]["val_candidate_text_by_field"]["amount"].update(exact_matches=78, exact_match=0.78), "amount floor"),
        (lambda summary: summary["records"][3]["val_candidate_text_by_field"]["time"].update(exact_matches=98, exact_match=0.98), "time floor"),
        (lambda summary: summary["records"][3]["val_candidate_text_by_field"]["payment_method_field"].update(exact_matches=93, exact_match=0.93), "payment_method_field floor"),
        (lambda summary: summary["records"][3]["val_ctc_by_field"]["transfer_status"].update(exact_matches=89, exact_match=0.89), "visible-status floor"),
        (lambda summary: summary["records"][3].update(val_status_non_success_to_success=1), "status safety"),
        (lambda summary: summary["records"][3].update(checkpoint_selection_eligible=False), "checkpoint eligible"),
        (lambda summary: summary["field_counts"]["recipient_field"].update(val=6788), "validation recipient denominator"),
        (lambda summary: summary["field_counts"]["recipient_field"].update(test=1), "physically included"),
        (lambda summary: summary["records"][2]["val_candidate_text_by_field"]["amount"].update(exact_matches=79), "count/rate evidence"),
        (lambda summary: summary["records"][2]["val_ctc_by_field"]["transfer_status"].update(records=99), "count/rate evidence"),
        (lambda summary: summary["records"][3]["val_candidate_text_by_field"]["amount"].update(exact_match=math.nan), "invalid epoch 3 amount"),
    ],
)
def test_b8_rejects_safety_denominator_nan_and_test_rows(mutation, message: str) -> None:
    summary = _summary()
    mutation(summary)
    with pytest.raises(ValueError, match=message):
        evaluate_continuation_summary(summary)


def test_fixed_recipe_cannot_be_24_or_80_epochs() -> None:
    recipe = fixed_recipe()
    assert recipe["epochs"] == 8
    assert recipe["batch_size"] == 10
    assert recipe["learning_rate"] == pytest.approx(1e-4)
    assert recipe["seed"] == 42
    assert recipe["recipient_train_augmentation"] == "robust_v2"
    assert recipe["validation_every"] == 1


def test_real_r2_source_is_hard_pinned_against_coherent_reseal(
    tmp_path: Path,
) -> None:
    assert FIXED_SOURCE_SUBJECT_ID == (
        "acad17751217302330e3413698943d66f18683bea0fac229fec120941e5857bf"
    )
    assert FIXED_SOURCE_ARTIFACTS == {
        "source_best_checkpoint": {
            "sha256": "1f908aa2b47ab83d6ff7ac01e7653f3763383d3e4ad3431b4ddbcaf34cf653a6",
            "size_bytes": 39163899,
        },
        "source_training_summary": {
            "sha256": "c7fd3695ad228337d516738395e7a552c0f282fe0186ee54c24579ee42141e97",
            "size_bytes": 89967,
        },
        "source_pilot_decision": {
            "sha256": "b86cb9ecd06d81defff28d4178edc21d3d2ae30dfa334c1bd34ffe57cd93757d",
            "size_bytes": 3314,
        },
        "blind_manifest": {
            "sha256": "c303c8a34348532263d3ad84ed2c6dddcd77c1bdd9dfc8a7c713ccc35a1ff5f1",
            "size_bytes": 202226294,
        },
        "blind_contract": {
            "sha256": "1167ea06f667169c72bbe572c1fe0b516389b6f7b92120d05338f0af8cf3465c",
            "size_bytes": 996,
        },
        "full_manifest": {
            "sha256": "7b7f51605c3471a4b38a85bea1bbe32212d8b3c004f248e075b17b8363f850a7",
            "size_bytes": 224911119,
        },
        "sanitized_seed": {
            "sha256": "b4e30ac514a89cb83e54cbde6d42ba007c370635785c12c1240e232e75e7c17c",
            "size_bytes": 39154511,
        },
    }
    assert continuation._canonical_sha256(
        {
            "domain": "receipt-recipient-full-crop-legacy-continuation-source-v1",
            "authorization": AUTHORIZATION,
            "source_best_epoch": SOURCE_BEST_EPOCH,
            "source_recipient_matches": SOURCE_RECIPIENT_MATCHES,
            "source_recipient_denominator": RECIPIENT_DENOMINATOR,
            "fixed_source_artifacts": FIXED_SOURCE_ARTIFACTS,
        },
        description="fixed test subject",
    ) == FIXED_SOURCE_SUBJECT_ID
    forged_same_topology = tmp_path / "best.pt"
    forged_same_topology.write_bytes(b"same topology and metadata, different tensor bytes")
    for name in FIXED_SOURCE_ARTIFACTS:
        with pytest.raises(ValueError, match="does not match the real r2 digest"):
            _require_fixed_source_artifacts(
                {name: forged_same_topology},
                (name,),
            )


def test_real_r2_pins_precede_corresponding_semantic_reads() -> None:
    source = inspect.getsource(continuation._recompute_pilot_closure)
    initial_pin = source.index("_capture_fixed_source_artifacts(paths, tuple(paths))")
    first_json_read = source.index("summary = _strict_json_bytes(")
    full_pin = source.index("full_capture = _capture_fixed_source_artifacts(")
    best_load = source.index("best_payload = _load_frozen_v13_checkpoint(")
    seed_pin = source.index("seed_capture = _capture_fixed_source_artifacts(")
    tree_scan = source.index("_assert_analysis_tree(root)")
    seed_load = source.index("seed_payload = _load_frozen_v13_checkpoint(")
    assert (
        initial_pin
        < first_json_read
        < full_pin
        < best_load
        < seed_pin
        < tree_scan
        < seed_load
    )


def test_sanitizer_transitive_sources_are_exposed_for_deny_write_leases(
    tmp_path: Path,
) -> None:
    status = tmp_path / "status.pt"
    train = tmp_path / "train.pt"
    ancestor = tmp_path / "ancestor.pt"
    status.write_bytes(b"status")
    train.write_bytes(b"train")
    ancestor.write_bytes(b"ancestor")

    def descriptor(path: Path, kind: str, epoch: int) -> dict[str, object]:
        return {**_binding(path), "kind": kind, "epoch": epoch}

    train_descriptor = descriptor(train, "receipt_unified_field_reader_v12", 8)
    seed = {
        "full_crop_seed_sanitizer_attestation": {
            "status_checkpoint": descriptor(
                status, "receipt_unified_field_reader_v13", 4
            ),
            "train_only_recipient_checkpoint": train_descriptor,
            "train_only_recipient_lineage": {
                "entries": [
                    {"checkpoint": train_descriptor},
                    {
                        "checkpoint": descriptor(
                            ancestor, "receipt_unified_field_reader_v12", 1
                        )
                    },
                ]
            },
        }
    }
    observed = _sanitizer_transitive_source_paths(seed)
    assert set(observed.values()) == {status, train, ancestor}
    ancestor.write_bytes(b"changed")
    with pytest.raises(ValueError, match="changed after sealing"):
        _sanitizer_transitive_source_paths(seed)


def test_binding_detects_toctou_and_fresh_file_rejects_reuse_and_reparse(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.json"
    source.write_text("one", encoding="utf-8")
    sealed = _binding(source)
    assert _verify_binding(sealed, "source") == source
    source.write_text("two", encoding="utf-8")
    with pytest.raises(ValueError, match="changed after sealing"):
        _verify_binding(sealed, "source")
    with pytest.raises(ValueError, match="Refusing to overwrite"):
        _fresh_file(source, suffix=".json", description="source")
    link = tmp_path / "link.json"
    try:
        link.symlink_to(source)
    except OSError:
        pytest.skip("symlinks unavailable")
    with pytest.raises(ValueError, match="symlink/junction/reparse"):
        _fresh_file(link, suffix=".json", description="source")


def test_frozen_validation_bytes_and_binding_cannot_diverge(tmp_path: Path) -> None:
    artifact = tmp_path / "training_summary.json"
    original = b'{"passed":true}\n'
    replacement = b'{"passed":null}\n'
    assert len(original) == len(replacement)
    artifact.write_bytes(original)
    frozen, identity = _read_frozen_regular_file(
        artifact, description="training summary"
    )
    assert frozen == original
    assert identity[2:] == (len(original), hashlib.sha256(original).hexdigest())

    # A same-size pathname replacement cannot make a decision bind different
    # bytes from those consumed by the semantic validator.
    artifact.write_bytes(replacement)
    assert _file_identity(artifact) != identity


def test_last_checkpoint_cannot_strip_analysis_only_policy_metadata() -> None:
    summary = {
        key: {"bound": key}
        for key in continuation._CHECKPOINT_SUMMARY_METADATA_KEYS
    }
    last = dict(summary)
    _require_checkpoint_summary_metadata(
        last, summary, description="continuation last"
    )
    last["initialization"] = {}
    with pytest.raises(ValueError, match="continuation last initialization"):
        _require_checkpoint_summary_metadata(
            last, summary, description="continuation last"
        )
    validator_source = inspect.getsource(
        continuation._validate_continuation_training_artifacts
    )
    assert validator_source.count("_require_checkpoint_summary_metadata(") == 2


def test_recipient_coverage_denominator_is_rescanned_from_bound_blind_manifest(
    tmp_path: Path,
) -> None:
    manifest = tmp_path / "blind.jsonl"
    rows = [
        {"id": "train", "split": "train", "slots": {"recipient_field": {"text": "甲"}}},
        {"id": "val-labelled", "split": "val", "slots": {"recipient_field": {"text": "乙"}}},
        {"id": "val-unlabelled", "split": "val", "slots": {}},
    ]
    manifest.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
    assert (
        _blind_recipient_validation_denominator(
            manifest, expected_sha256=digest
        )
        == 1
    )
    with pytest.raises(ValueError, match="changed during recipient denominator scan"):
        _blind_recipient_validation_denominator(
            manifest, expected_sha256="0" * 64
        )
    with pytest.raises(ValueError, match="bound blind manifest recipient denominator"):
        evaluate_continuation_summary(
            _summary(), bound_recipient_val_denominator=1
        )


def test_pilot_blind_semantics_are_read_from_hard_pinned_bytes(tmp_path: Path) -> None:
    full = tmp_path / "full.jsonl"
    full.write_text('{"id":"test","split":"test"}\n', encoding="utf-8")
    blind = tmp_path / "blind.jsonl"
    blind.write_text(
        '{"id":"train","split":"train","slots":{}}\n'
        '{"id":"val","split":"val","slots":{}}\n',
        encoding="utf-8",
    )
    captured = blind.read_bytes()
    contract = tmp_path / "blind.contract.json"
    contract_payload = {
        "schema_version": 1,
        "kind": "receipt_recipient_blind_train_val_manifest_v1",
        "source_manifest": str(full),
        "source_manifest_sha256": hashlib.sha256(full.read_bytes()).hexdigest(),
        "blind_manifest": str(blind),
        "blind_manifest_sha256": hashlib.sha256(captured).hexdigest(),
        "split_counts": {"train": 1, "val": 1, "test_excluded": 1},
        "optimizer_supervision_splits": ["train"],
        "checkpoint_selection_splits": ["val"],
        "final_gate_only_splits": ["test"],
        "test_labels_used": False,
        "test_metrics_computed": False,
        "test_examples_emitted": False,
    }
    contract.write_text(json.dumps(contract_payload), encoding="utf-8")
    blind.write_text('{"id":"forged","split":"test","slots":{}}\n', encoding="utf-8")
    observed = _verify_frozen_blind_manifest_contract(
        records_path=blind,
        records_data=captured,
        contract_path=contract,
        contract_data=contract.read_bytes(),
    )
    assert observed["split_counts"] == {"train": 1, "val": 1, "test_excluded": 1}
    assert _file_identity(blind)[3] != hashlib.sha256(captured).hexdigest()


def test_windows_wrapper_locks_gpu_recipe_leases_and_analysis_boundaries() -> None:
    repo = Path(__file__).resolve().parents[1]
    runner = (
        repo / "scripts" / "receipt-ocr-recipient-full-crop-continuation-4090.ps1"
    ).read_text(encoding="utf-8")
    for required in (
        "$epochs = 8",
        "$batchSize = 10",
        "$learningRate = 0.0001",
        "$seed = 42",
        '$augmentation = "robust_v2"',
        '"--device", "cuda:0"',
        "4090",
        "Open-ReadLease",
        "[IO.FileShare]::Read",
        "continuation-source.contract.json",
        "authorized-pilot-best.pt",
        "no 24/80 epoch route",
        "no_24_epoch_route",
        "no_80_epoch_route",
        "foreach ($property in $sealed.source_artifacts.PSObject.Properties)",
        "continuation code-closure artifact",
        "ocr_unified_dataset.py",
        "ocr_unified_targets.py",
        "normalize_json_summary.py",
    ):
        assert required in runner
    assert "SeedCheckpoint" not in runner
    assert "recipient-v14" not in runner


def test_code_closure_contains_training_and_metric_dependencies() -> None:
    code_paths = continuation._code_paths()
    assert {
        "code_package_init",
        "code_labels",
        "code_model",
        "code_ocr",
        "code_onnx_runtime",
        "code_recipient_beam",
        "code_recipient_audit",
        "code_ocr_unified_dataset",
        "code_ocr_unified_targets",
        "code_continuation",
        "code_ocr_unified",
        "code_full_crop_pilot",
        "code_blind_manifest",
        "code_seed_sanitizer",
        "script_full_crop_pilot",
        "script_continuation",
        "script_json_normalizer",
    } <= set(code_paths)
    assert all(path.is_file() for path in code_paths.values())


def _checkpoint_payload(torch) -> tuple[dict[str, object], dict[str, object]]:
    config = _config()
    payload: dict[str, object] = {
        "schema_version": 1,
        "kind": KIND_V13,
        "epoch": SOURCE_BEST_EPOCH,
        "config": asdict(config),
        "state_dict": {
            "shared.weight": torch.tensor([1.0, -0.0]),
            "recipient_weight": torch.tensor([2.0, 3.0]),
        },
        "amount_characters": ["0", "1"],
        "time_characters": ["0", ":"],
        "payment_characters": ["卡", "行"],
        "recipient_characters": ["商", "户"],
        "status_classes": list(STATUS_CLASSES),
        "status_text_characters": ["成", "功", "账", "转"],
        "payment_bank_prefix_classes": ["__other__", "邮储银行"],
        "recipient_train_split_policy": {"mode": "standard_train_only", "splits": ["train"]},
        "metrics": {"epoch": SOURCE_BEST_EPOCH},
    }
    closure = {
        "pilot_root": "/bound/r031004-06/full-crop-pilot-8e-r2",
        "source_recipient": {
            "records": RECIPIENT_DENOMINATOR,
            "exact_matches": SOURCE_RECIPIENT_MATCHES,
            "exact_match": SOURCE_RECIPIENT_MATCHES / RECIPIENT_DENOMINATOR,
        },
        "artifacts": {},
    }
    unsigned = continuation._authority_payload(closure, payload)
    authority = {
        **unsigned,
        "integrity_sha256": continuation._canonical_sha256(
            unsigned, description="test authority"
        ),
    }
    return payload, {**payload, AUTHORITY_KEY: authority}


def test_embedded_authority_binds_state_config_maps_and_source(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch = pytest.importorskip("torch")
    best, authorized = _checkpoint_payload(torch)
    authority = authorized[AUTHORITY_KEY]
    closure = {
        "pilot_root": authority["pilot_root"],
        "source_recipient": authority["source_recipient"],
        "artifacts": authority["artifacts"],
    }
    monkeypatch.setattr(
        continuation,
        "_recompute_pilot_closure",
        lambda *args, **kwargs: (closure, best, {}),
    )
    validated = continuation.validate_embedded_continuation_authority(
        authorized, torch=torch
    )
    assert validated["authorization"] == AUTHORIZATION

    tampered_state = dict(authorized)
    tampered_state["state_dict"] = dict(authorized["state_dict"])
    tampered_state["state_dict"]["recipient_weight"] = torch.tensor([2.0, 4.0])
    with pytest.raises(ValueError, match="content does not match"):
        continuation.validate_embedded_continuation_authority(tampered_state, torch=torch)

    tampered_map = dict(authorized)
    tampered_map["recipient_characters"] = ["户", "商"]
    with pytest.raises(ValueError, match="content does not match"):
        continuation.validate_embedded_continuation_authority(tampered_map, torch=torch)

    tampered_authority = dict(authorized)
    tampered_authority[AUTHORITY_KEY] = dict(authorized[AUTHORITY_KEY])
    tampered_authority[AUTHORITY_KEY]["authorization"] = "ordinary_strict"
    with pytest.raises(ValueError, match="integrity hash"):
        continuation.validate_embedded_continuation_authority(
            tampered_authority, torch=torch
        )


def test_initializer_copies_every_state_tensor_and_rejects_map_or_tensor_changes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    torch = pytest.importorskip("torch")
    best, authorized = _checkpoint_payload(torch)
    checkpoint = tmp_path / "authorized.pt"
    checkpoint.write_bytes(b"placeholder")
    monkeypatch.setattr(unified, "_load_checkpoint", lambda *args, **kwargs: authorized)
    monkeypatch.setattr(
        unified,
        "_validate_recipient_full_crop_continuation_policy",
        lambda *args, **kwargs: authorized[AUTHORITY_KEY],
    )
    target_state = {name: torch.zeros_like(value) for name, value in authorized["state_dict"].items()}
    state, initialization = _parameter_only_initialization(
        init_checkpoint=checkpoint,
        init_checkpoint_mode=INIT_CHECKPOINT_MODE_RECIPIENT_FULL_CROP_CONTINUATION,
        config=_config(),
        amount_characters=["0", "1"],
        time_characters=["0", ":"],
        payment_characters=["卡", "行"],
        recipient_characters=["商", "户"],
        status_text_characters=["成", "功", "账", "转"],
        payment_bank_prefix_classes=["__other__", "邮储银行"],
        torch=torch,
        target_state_dict=target_state,
    )
    assert state is authorized["state_dict"]
    assert initialization["optimizer_restored"] is False
    assert initialization["epoch_reset"] is True
    assert initialization["all_state_tensor_count_copied"] == len(target_state)
    for name in target_state:
        torch.testing.assert_close(state[name], authorized["state_dict"][name], rtol=0, atol=0)

    with pytest.raises(ValueError, match="payment character map"):
        _recipient_only_expansion_label_override(
            init_checkpoint=checkpoint,
            config=_config(),
            amount_characters=["0", "1"],
            time_characters=["0", ":"],
            payment_characters=["行", "卡"],
            recipient_characters=["商", "户"],
            payment_bank_prefix_classes=["__other__", "邮储银行"],
            torch=torch,
            init_checkpoint_mode=INIT_CHECKPOINT_MODE_RECIPIENT_FULL_CROP_CONTINUATION,
        )
    missing = dict(target_state)
    missing.pop("shared.weight")
    with pytest.raises(ValueError, match="all-state key match"):
        _parameter_only_initialization(
            init_checkpoint=checkpoint,
            init_checkpoint_mode=INIT_CHECKPOINT_MODE_RECIPIENT_FULL_CROP_CONTINUATION,
            config=_config(),
            amount_characters=["0", "1"],
            time_characters=["0", ":"],
            payment_characters=["卡", "行"],
            recipient_characters=["商", "户"],
            status_text_characters=["成", "功", "账", "转"],
            payment_bank_prefix_classes=["__other__", "邮储银行"],
            torch=torch,
            target_state_dict=missing,
        )
