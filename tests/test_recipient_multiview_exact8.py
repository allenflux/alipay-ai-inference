from __future__ import annotations

import ctypes
import hashlib
import inspect
import json
import os
import runpy
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import transfer_receipt_ai.recipient_multiview_exact8 as exact8
import transfer_receipt_ai.recipient_multiview_overlay as overlay_module
from transfer_receipt_ai.ocr_unified import (
    INIT_CHECKPOINT_MODE_RECIPIENT_VISUAL_CONTEXT_REINIT,
    STATUS_TEXT_RUNTIME_POLICY,
    UnifiedReaderConfig,
    _recipient_confidence_policy,
    _recipient_train_augmentation_policy,
)
import transfer_receipt_ai.recipient_full_crop_seed_sanitizer as seed_sanitizer
from transfer_receipt_ai.recipient_full_crop_candidate_source import (
    CANDIDATE_PILOT_KIND,
    SOURCE_KIND,
)
from transfer_receipt_ai.recipient_v14_failure_attestor import (
    AUTHORIZATION as FAILURE_AUTHORIZATION,
    DECISION as FAILURE_DECISION,
    KIND as FAILURE_KIND,
)


class _FakeWinFunction:
    argtypes: object = None
    restype: object = None

    def __init__(self, callback: Any) -> None:
        self.callback = callback

    def __call__(self, *args: object) -> object:
        return self.callback(*args)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True, indent=2)
        + "\n",
        encoding="utf-8",
    )


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _binding(path: Path) -> dict[str, object]:
    return {
        "path": str(path.resolve()),
        "sha256": _sha(path),
        "size_bytes": path.stat().st_size,
    }


def _configs() -> tuple[dict[str, object], dict[str, object]]:
    source = UnifiedReaderConfig(
        architecture_version=13,
        recipient_value_left_trim=0.0,
        recipient_input_height=128,
        recipient_input_width=1536,
        recipient_branch_channels=16,
        recipient_hidden_size=192,
    )
    target = replace(
        source,
        recipient_backbone="residual_positional_transformer_v2",
        recipient_open_text_layers=4,
        recipient_open_text_heads=8,
        recipient_open_text_feedforward=1536,
        recipient_open_text_dropout=0.10,
    )
    source.validate()
    target.validate()
    return asdict(source), asdict(target)


def _financial_label_policy() -> dict[str, object]:
    return {
        "recipient_character_map": {
            "mode": "fresh_train_only_reinitialized_recipient_v1"
        },
        "payment_character_map": {"mode": "checkpoint_financial_label_map_v1"},
    }


def _field_counts() -> dict[str, dict[str, int]]:
    return {
        field: {
            "train": 20,
            "val": {
                "amount": 1606,
                "time": 4000,
                "payment_method_field": 5400,
                "recipient_field": 6789,
                "transfer_status": 120,
            }[field],
            "test": 0,
        }
        for field in (
            "amount",
            "time",
            "payment_method_field",
            "recipient_field",
            "transfer_status",
        )
    }


def _tail_policy() -> dict[str, object]:
    return {
        "mode": "rare_long_tail_ctc_v1",
        "rare_character_max_support": 3,
        "rare_character_loss_weight": 1.5,
        "long_text_min_length": 9,
        "long_text_loss_weight": 1.5,
        "recipient_train_records": 20,
        "rare_character_train_records": 4,
        "long_text_train_records": 3,
        "combined_boost_train_records": 6,
    }


def _summary_data_fields() -> dict[str, object]:
    return {
        "field_counts": _field_counts(),
        "status_class_counts": {"train": {"success": 20}, "val": {"success": 120}},
        "status_head_policy": {
            "training_enabled": False,
            "runtime_policy": "review_only",
        },
        "structured_target_counts": {"train": 20, "val": 120},
        "status_text_oov_by_split": {
            "train": {"records": 20},
            "val": {"records": 100},
            "test": {"records": 0},
        },
        "payment_oov_by_split": {
            "train": {"records": 20},
            "val": {"records": 5400},
            "test": {"records": 0},
        },
        "payment_bank_prefix_classes": ["__other__", "bank_a"],
        "payment_bank_prefix_min_support": 3,
        "payment_bank_prefix_class_counts": {"__other__": 0, "bank_a": 20},
        "payment_bank_prefix_train_class_counts": {"__other__": 0, "bank_a": 20},
        "payment_bank_prefix_oov_by_split": {
            "train": {"records": 20},
            "val": {"records": 5400},
            "test": {"records": 0},
        },
        "recipient_oov_by_split": {
            "train": {"records": 20},
            "val": {"records": 6789},
            "test": {"records": 0},
        },
        "recipient_sampling_policy": {
            "mode": "uniform",
            "recipient_sampling_weight": 1.0,
            "recipient_train_records": 20,
            "train_records": 20,
        },
        "recipient_confidence_policy": _recipient_confidence_policy(
            low_confidence_threshold=0.95,
            low_confidence_loss_weight=0.50,
            curriculum_epochs=10,
        ),
        "recipient_tail_loss_policy": _tail_policy(),
        "recipient_train_augmentation_policy": _recipient_train_augmentation_policy(
            mode="robust_v2", seed=42
        ),
        "recipient_train_split_policy": {
            "mode": "standard_train_only",
            "splits": ["train"],
        },
        "recipient_target": "anchored_recipient_value_with_dedicated_high_resolution_value_view",
        "status_text_charset_sha256": hashlib.sha256("成功".encode("utf-8")).hexdigest(),
        "status_text_charset_source": "train_only_visible_status_text",
        "status_text_target": "visible_transfer_status_cjk_text",
    }


def _ordered_labels() -> dict[str, object]:
    payment = ["A", "B"]
    recipient = ["甲", "乙"]
    status_text = ["成", "功"]
    return {
        "amount_characters": ["0", "1"],
        "time_characters": ["0", "1", ":"],
        "payment_characters": payment,
        "payment_charset_sha256": hashlib.sha256("".join(payment).encode()).hexdigest(),
        "status_classes": ["success", "pending", "failed"],
        "status_text_blank_index": 0,
        "status_text_characters": status_text,
        "status_text_charset_sha256": hashlib.sha256(
            "".join(status_text).encode("utf-8")
        ).hexdigest(),
        "status_text_charset_source": "train_only_visible_status_text",
        "status_text_target": "visible_transfer_status_cjk_text",
        "recipient_blank_index": 0,
        "recipient_characters": recipient,
        "recipient_charset_sha256": hashlib.sha256(
            "".join(recipient).encode("utf-8")
        ).hexdigest(),
        "recipient_charset_source": "train_only_anchored_recipient_value",
        "recipient_target": "anchored_recipient_value_with_dedicated_high_resolution_value_view",
        "payment_bank_prefix_classes": ["__other__", "bank_a"],
        "payment_bank_prefix_min_support": 3,
        "payment_bank_prefix_class_counts": {"__other__": 0, "bank_a": 20},
        "payment_bank_prefix_train_class_counts": {"__other__": 0, "bank_a": 20},
        "payment_bank_prefix_oov_by_split": {
            "train": {"records": 20},
            "val": {"records": 5400},
            "test": {"records": 0},
        },
    }


def _data_label_proof() -> dict[str, object]:
    labels = _ordered_labels()
    return {
        "summary_fields": _summary_data_fields(),
        "financial_label_policy": _financial_label_policy(),
        "ordered_labels": labels,
        "ordered_label_maps": {
            key: {
                "count": len(labels[key]),
                "sha256": exact8._canonical_sha256(labels[key]),
            }
            for key in exact8.A8_ORDERED_MAP_KEYS
        },
        "blank_indices": {
            key: {
                "source": exact8.A8_BLANK_INDEX_PROOF[key][0],
                "semantic": exact8.A8_BLANK_INDEX_PROOF[key][1],
                "value": 0,
            }
            for key in exact8.A8_BLANK_INDEX_KEYS
        },
    }


def _inspection(tmp_path: Path) -> dict[str, object]:
    source_checkpoint = tmp_path / "source-best.pt"
    source_checkpoint.write_bytes(b"original-full-crop-pilot-best")
    composite_records = tmp_path / "fixed2" / "unified_fields.train-val.fixed2.jsonl"
    composite_records.parent.mkdir()
    composite_records.write_text("", encoding="utf-8")
    composite_root = tmp_path / "composite-root"
    composite_root.mkdir()
    _source_config, target_config = _configs()
    return {
        "schema_version": 1,
        "kind": exact8.INSPECTION_KIND,
        "analysis_only": True,
        "production_route_authorized": False,
        "test_opened": False,
        "onnx_exported": False,
        "route_subject_id": "a" * 64,
        "attempt_id": "a" * 64,
        "source_subject_id": exact8.ATTESTED_SOURCE_SUBJECT_ID,
        "candidate_pilot_subject_id": exact8.ATTESTED_A8_SUBJECT_ID,
        "failure_subject_id": "b" * 64,
        "overlay_subject_id": "c" * 64,
        "source_checkpoint": str(source_checkpoint.resolve()),
        "source_checkpoint_sha256": _sha(source_checkpoint),
        "full_manifest_sha256": "d" * 64,
        "composite_records": str(composite_records.resolve()),
        "composite_dataset_root": str(composite_root.resolve()),
        "overlay_contract_sha256": "e" * 64,
        "baseline": {
            "best_epoch": 8,
            "best_matches": 5722,
            "epoch4_matches": 5500,
            "epoch8_matches": 5722,
            "candidate_denominators": {
                "amount": 1428,
                "time": 3738,
                "payment_method_field": 5242,
                "recipient_field": 6789,
                "transfer_status": 100,
            },
            "data_label_proof": _data_label_proof(),
        },
        "target_config": target_config,
        "fixed_gates": {
            "recipient_denominator": 6789,
            "minimum_absolute_best_matches": 5790,
            "minimum_best_gain_over_A8_matches": 68,
            "minimum_epoch8_gain_over_A8_matches": 68,
            "minimum_epoch4_to_8_gain_matches": 136,
            "maximum_best_to_epoch8_gap_matches": 67,
            "strict_fresh60_recipient_target_matches": 6111,
            "amount_floor": exact8.AMOUNT_FLOOR,
            "time_floor": exact8.TIME_FLOOR,
            "payment_floor": exact8.PAYMENT_FLOOR,
            "visible_status_floor": exact8.STATUS_TEXT_FLOOR,
            "unsafe_status_max": 0,
        },
        "code": {},
        "guard_paths": [],
        "guard_directories": [],
        "guard_directory_identities": [],
    }


def _summary(
    inspection: dict[str, object],
    *,
    matches: list[int] | None = None,
) -> dict[str, object]:
    source_config, target_config = _configs()
    source_checkpoint = Path(str(inspection["source_checkpoint"]))
    initialization = {
        "mode": "parameter_only_recipient_visual_context_reinit",
        "init_checkpoint_mode": INIT_CHECKPOINT_MODE_RECIPIENT_VISUAL_CONTEXT_REINIT,
        "source_kind": "receipt_unified_field_reader_v13",
        "source_config": source_config,
        "checkpoint_path": str(source_checkpoint.resolve()),
        "checkpoint_sha256": _sha(source_checkpoint),
        "financial_label_policy": _financial_label_policy(),
    }
    fine_tune = {
        "mode": "recipient_only_v13",
        "trainable_parameter_prefix": "recipient_",
        "training_forward": "private_recipient_branch_only_v13",
    }
    runtime = {
        "device": "cuda:0",
        "uses_cuda": True,
        "cuda_device_name": "NVIDIA GeForce RTX 4090",
        "num_workers": 4,
        "prefetch_factor": 2,
        "persistent_workers": True,
        "train_progress_every": 250,
        "validation_every": 1,
        "cuda_tf32_requested": True,
        "cudnn_benchmark_requested": True,
    }
    target_reader = UnifiedReaderConfig(**target_config)
    checkpoint_policy = exact8._expected_checkpoint_policy(target_reader)
    data_fields = _summary_data_fields()
    status_policy = data_fields["status_head_policy"]
    recipient_matches = matches or [5500, 5550, 5600, 5654, 5700, 5740, 5770, 5790]
    records: list[dict[str, object]] = []
    for epoch, recipient in enumerate(recipient_matches, start=1):
        recipient_rate = recipient / 6789
        records.append(
            {
                "epoch": epoch,
                "validation_performed": True,
                "val_candidate_text_by_field": {
                    "amount": {
                        "records": 1428,
                        "exact_matches": 1428,
                        "exact_match": 1.0,
                    },
                    "time": {
                        "records": 3738,
                        "exact_matches": 3738,
                        "exact_match": 1.0,
                    },
                    "payment_method_field": {
                        "records": 5242,
                        "exact_matches": 5242,
                        "exact_match": 1.0,
                    },
                    "recipient_field": {
                        "records": 6789,
                        "exact_matches": recipient,
                        "exact_match": recipient_rate,
                    },
                },
                "val_ctc_by_field": {
                    "transfer_status": {
                        "records": 100,
                        "exact_matches": 100,
                        "exact_match": 1.0,
                    }
                },
                "val_loss": 1.0,
                "val_candidate_text_macro_exact_match": 1.0,
                "val_candidate_text_exact_match": 1.0,
                "val_verifier_macro_exact_match": 1.0,
                "val_status_non_success_to_success": 0,
            }
        )
        record = records[-1]
        validation = exact8._validation_from_epoch_record(
            record, description=f"synthetic epoch {epoch}"
        )
        score, failures = exact8._checkpoint_selection_score(
            validation,
            config=target_reader,
            status_policy=status_policy,
            policy=checkpoint_policy,
        )
        record["checkpoint_selection_eligible"] = score is not None
        record["checkpoint_selection_protection_failures"] = failures
        record["checkpoint_selection_score"] = (
            list(score) if score is not None else None
        )
        record["checkpoint_protection"] = exact8._checkpoint_protection_report(
            validation,
            policy=checkpoint_policy,
            failures=failures,
        )
    best_epoch = 0
    best_score: tuple[float, ...] | None = None
    for record in records:
        raw_score = record["checkpoint_selection_score"]
        if raw_score is None:
            continue
        score = tuple(float(item) for item in raw_score)
        if best_score is None or score > best_score:
            best_score = score
            best_epoch = int(record["epoch"])
    assert best_score is not None and best_epoch > 0
    return {
        "schema_version": 1,
        "kind": "receipt_unified_field_reader_v13",
        "config": target_config,
        "initialization": initialization,
        "fine_tune_policy": fine_tune,
        "training_runtime": runtime,
        "checkpoint_selection_policy": checkpoint_policy,
        "status_text_runtime_policy": STATUS_TEXT_RUNTIME_POLICY,
        **data_fields,
        "recipient_loss_weight": 1.0,
        "best_checkpoint_epoch": best_epoch,
        "best_checkpoint_score": records[best_epoch - 1]["checkpoint_selection_score"],
        "records": records,
    }


def _refresh_checkpoint_evidence(summary: dict[str, object]) -> None:
    config = UnifiedReaderConfig(**summary["config"])
    policy = summary["checkpoint_selection_policy"]
    status_policy = summary["status_head_policy"]
    best_epoch = 0
    best_score: tuple[float, ...] | None = None
    for record in summary["records"]:
        validation = exact8._validation_from_epoch_record(
            record, description=f"synthetic epoch {record['epoch']}"
        )
        score, failures = exact8._checkpoint_selection_score(
            validation,
            config=config,
            status_policy=status_policy,
            policy=policy,
        )
        record["checkpoint_selection_eligible"] = score is not None
        record["checkpoint_selection_protection_failures"] = failures
        record["checkpoint_selection_score"] = (
            list(score) if score is not None else None
        )
        record["checkpoint_protection"] = exact8._checkpoint_protection_report(
            validation,
            policy=policy,
            failures=failures,
        )
        if score is not None and (best_score is None or score > best_score):
            best_score = score
            best_epoch = int(record["epoch"])
    assert best_score is not None and best_epoch > 0
    summary["best_checkpoint_epoch"] = best_epoch
    summary["best_checkpoint_score"] = list(best_score)


def _labels(summary: dict[str, object]) -> dict[str, object]:
    keys = (
        "initialization",
        "fine_tune_policy",
        "status_text_runtime_policy",
        "training_runtime",
        "checkpoint_selection_policy",
        "structured_target_counts",
        "status_text_oov_by_split",
        "recipient_oov_by_split",
        "recipient_sampling_policy",
        "recipient_confidence_policy",
        "recipient_tail_loss_policy",
        "recipient_train_augmentation_policy",
        "recipient_train_split_policy",
        "recipient_target",
    )
    labels = {
        "schema_version": 1,
        "amount_blank_index": 0,
        "time_blank_index": 0,
        "payment_blank_index": 0,
        **{key: summary[key] for key in keys},
        **_ordered_labels(),
    }
    labels.update(
        exact8._recipient_artifact_metadata(
            UnifiedReaderConfig(**summary["config"]),
            recipient_sampling_policy=summary["recipient_sampling_policy"],
            recipient_confidence_policy=summary["recipient_confidence_policy"],
            recipient_tail_loss_policy=summary["recipient_tail_loss_policy"],
            recipient_train_augmentation_policy=summary[
                "recipient_train_augmentation_policy"
            ],
        )
    )
    return labels


def _checkpoint_payload(
    *,
    summary: dict[str, object],
    labels: dict[str, object],
    record: dict[str, object],
    state: dict[str, object],
) -> dict[str, object]:
    checkpoint: dict[str, object] = {
        "schema_version": 1,
        "kind": "receipt_unified_field_reader_v13",
        "epoch": record["epoch"],
        "state_dict": state,
        "metrics": record,
        "recipient_loss_weight": 1.0,
        "ctc_loss_weight": 1.0,
        "structured_loss_weight": 1.0,
    }
    for key in (
        "config",
        "initialization",
        "fine_tune_policy",
        "status_text_runtime_policy",
        "training_runtime",
        "checkpoint_selection_policy",
        *exact8.A8_SUMMARY_DATA_KEYS,
    ):
        checkpoint[key] = summary[key]
    for key in exact8.A8_ORDERED_LABEL_KEYS:
        checkpoint[key] = labels[key]
    return checkpoint


class _FakeTensor:
    def __init__(self, shape: tuple[int, ...], dtype: str = "float32") -> None:
        self.shape = shape
        self.dtype = dtype


def _valid_declared_model_state() -> dict[str, object]:
    return {
        "trunk.weight": _FakeTensor((4, 4)),
        "recipient_conv.weight": _FakeTensor((16, 1, 3, 3)),
        "recipient_encoder.weight": _FakeTensor((192, 16)),
        "recipient_classifier.weight": _FakeTensor((3, 192)),
        "recipient_classifier.bias": _FakeTensor((3,)),
    }


def _attempt(path: Path, inspection: dict[str, object], output: Path) -> None:
    _write_json(
        path,
        {
            "schema_version": 1,
            "kind": exact8.ATTEMPT_KIND,
            "attempt_id": inspection["attempt_id"],
            "route_subject_id": inspection["route_subject_id"],
            "source_subject_id": inspection["source_subject_id"],
            "candidate_pilot_subject_id": inspection["candidate_pilot_subject_id"],
            "failure_subject_id": inspection["failure_subject_id"],
            "overlay_subject_id": inspection["overlay_subject_id"],
            "output_root": str(output.resolve()),
            "epochs": 8,
            "selector_mode": exact8.SELECTOR_MODE,
            "created_at_utc": "2026-08-10T00:00:00Z",
            "full_manifest_sha256": inspection["full_manifest_sha256"],
            "threat_model": exact8.ATTEMPT_THREAT_MODEL,
        },
    )


def _formal_attempt_path(
    tmp_path: Path,
    inspection: dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
) -> Path:
    program_data = tmp_path / "ProgramData"
    registry = (
        program_data / exact8.ATTEMPT_REGISTRY_PARENT / exact8.ATTEMPT_REGISTRY_NAME
    )
    registry.mkdir(parents=True)
    monkeypatch.setattr(exact8, "_common_application_data_path", lambda: program_data)
    # Unit tests exercise the same descriptor-anchored lifecycle with POSIX
    # directory descriptors. Production run itself remains Windows-only.
    monkeypatch.setattr(exact8, "_require_formal_windows_output_anchor", lambda: None)

    def require_fixture_acl_policy(
        _handle: int,
        *,
        description: str,
        required_mask: int | None,
        required_flags: int,
        forbidden_flags: int,
        effective_denied_accesses: tuple[int, ...],
    ) -> None:
        child_mask = exact8._WINDOWS_DELETE | exact8._WINDOWS_FILE_DELETE_CHILD
        child_flags = (
            exact8._WINDOWS_OBJECT_INHERIT_ACE
            | exact8._WINDOWS_CONTAINER_INHERIT_ACE
        )
        child_forbidden = (
            exact8._WINDOWS_INHERITED_ACE
            | exact8._WINDOWS_INHERIT_ONLY_ACE
        )
        if description == "exact8 ProgramData DACL":
            assert required_mask is None
            assert required_flags == 0
            assert forbidden_flags == 0
            assert effective_denied_accesses == ()
            deny_aces: list[tuple[int, int]] = []
        elif description in {
            "exact8 ReceiptAI root DACL",
            "exact8 attempt registry DACL",
        }:
            assert required_mask == child_mask
            assert required_flags == child_flags
            assert forbidden_flags == child_forbidden
            assert effective_denied_accesses == (
                exact8._WINDOWS_DELETE,
                exact8._WINDOWS_FILE_DELETE_CHILD,
            )
            deny_aces = [(child_mask, child_flags)]
        elif description == "exact8 attempt marker DACL":
            assert required_mask == exact8._WINDOWS_DELETE
            assert required_flags == exact8._WINDOWS_INHERITED_ACE
            assert forbidden_flags == exact8._WINDOWS_INHERIT_ONLY_ACE
            assert effective_denied_accesses == (exact8._WINDOWS_DELETE,)
            deny_aces = [
                (exact8._WINDOWS_DELETE, exact8._WINDOWS_INHERITED_ACE)
            ]
        else:
            raise AssertionError(f"unexpected exact8 ACL fixture policy: {description}")
        exact8._require_windows_acl_evidence(
            description=description,
            deny_aces=deny_aces,
            required_mask=required_mask,
            required_flags=required_flags,
            forbidden_flags=forbidden_flags,
            effective_access={access: False for access in effective_denied_accesses},
        )

    monkeypatch.setattr(
        exact8,
        "_require_windows_acl_policy",
        require_fixture_acl_policy,
    )
    return registry / f"{inspection['attempt_id']}.attempt.json"


def test_integer_gate_boundaries_and_fixed2_pass_authority(tmp_path: Path) -> None:
    inspection = _inspection(tmp_path)
    recipe = exact8._recipe(inspection)
    boundary = _summary(
        inspection,
        matches=[5500, 5550, 5600, 5654, 5700, 5740, 5857, 5790],
    )
    passed = exact8.evaluate_exact8_summary(
        boundary, inspection=inspection, recipe=recipe
    )

    assert passed["passed"] is True
    assert passed["observed"]["epoch4_to_8_gain_matches"] == 136
    assert passed["observed"]["best_to_epoch8_gap_matches"] == 67
    assert passed["pass_authorization"] == {
        "authorization": exact8.PASS_AUTHORIZATION,
        "source": "original_full_crop_pilot_best_not_exact8_best",
        "initialization": INIT_CHECKPOINT_MODE_RECIPIENT_VISUAL_CONTEXT_REINIT,
        "training_data_view": "same_fixed2_overlay_subject",
        "epochs": 60,
        "fresh_optimizer": True,
        "required_final_recipient_matches": 6111,
        "requires_strictly_greater_than_90_percent": True,
        "exact8_checkpoint_initialization": "forbidden",
        "test_opened": False,
        "onnx_exported": False,
        "production_route_authorized": False,
    }
    decayed = _summary(
        inspection,
        matches=[5500, 5550, 5600, 5654, 5700, 5740, 5858, 5790],
    )
    stopped = exact8.evaluate_exact8_summary(
        decayed, inspection=inspection, recipe=recipe
    )
    assert stopped["passed"] is False
    assert "best_to_epoch8_decay_above_67" in stopped["failures"]
    assert stopped["pass_authorization"] is None


def test_best_epoch_follows_full_checkpoint_score_for_recipient_tie(
    tmp_path: Path,
) -> None:
    inspection = _inspection(tmp_path)
    summary = _summary(
        inspection,
        matches=[5500, 5550, 5600, 5654, 5700, 5740, 5857, 5857],
    )
    summary["records"][6]["val_candidate_text_macro_exact_match"] = 0.99
    _refresh_checkpoint_evidence(summary)

    decision = exact8.evaluate_exact8_summary(
        summary, inspection=inspection, recipe=exact8._recipe(inspection)
    )
    assert decision["passed"] is True
    assert decision["observed"]["best_epoch"] == 8


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("single_score", "recomputed checkpoint score"),
        ("selection_metric", "fixed exact8 checkpoint selection policy"),
        ("status_priority", "fixed exact8 checkpoint selection policy"),
        ("protection", "recomputed checkpoint protection"),
        ("failures", "recomputed checkpoint failures"),
        ("eligible", "recomputed checkpoint eligibility"),
        ("strict_tie", "strict-greater-than first-best"),
    ],
)
def test_checkpoint_selection_evidence_is_independently_recomputed(
    tmp_path: Path, mutation: str, message: str
) -> None:
    inspection = _inspection(tmp_path)
    matches = (
        [5500, 5550, 5600, 5654, 5700, 5740, 5857, 5857]
        if mutation == "strict_tie"
        else None
    )
    summary = _summary(inspection, matches=matches)
    first = summary["records"][0]
    if mutation == "single_score":
        first["checkpoint_selection_score"] = [1.0]
    elif mutation == "selection_metric":
        summary["checkpoint_selection_policy"]["selection_metric"] = "forged"
    elif mutation == "status_priority":
        summary["checkpoint_selection_policy"]["status_text_ctc_priority"] = True
    elif mutation == "protection":
        first["checkpoint_protection"]["margin"]["amount"] = 999.0
    elif mutation == "failures":
        first["checkpoint_selection_protection_failures"] = ["forged"]
    elif mutation == "eligible":
        first["checkpoint_selection_eligible"] = False
    else:
        assert summary["best_checkpoint_epoch"] == 7
        summary["best_checkpoint_epoch"] = 8
        summary["best_checkpoint_score"] = summary["records"][7][
            "checkpoint_selection_score"
        ]
    with pytest.raises(ValueError, match=message):
        exact8.evaluate_exact8_summary(
            summary, inspection=inspection, recipe=exact8._recipe(inspection)
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("financial_rate", "count/rate"),
        ("financial_denominator", "denominator"),
        ("status_rate", "count/rate"),
        ("recipient_rate", "count/rate"),
    ],
)
def test_all_guard_metrics_require_integer_count_rate_consistency(
    tmp_path: Path, mutation: str, message: str
) -> None:
    inspection = _inspection(tmp_path)
    summary = _summary(inspection)
    first = summary["records"][0]
    if mutation == "financial_rate":
        first["val_candidate_text_by_field"]["amount"]["exact_matches"] = 99
    elif mutation == "financial_denominator":
        first["val_candidate_text_by_field"]["time"]["records"] = 99
    elif mutation == "status_rate":
        first["val_ctc_by_field"]["transfer_status"]["exact_matches"] = 99
    else:
        first["val_candidate_text_by_field"]["recipient_field"]["exact_match"] = 0.99
    with pytest.raises(ValueError, match=message):
        exact8.evaluate_exact8_summary(
            summary, inspection=inspection, recipe=exact8._recipe(inspection)
        )


def test_a8_freezes_candidate_denominators_not_raw_field_counts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    records: list[dict[str, object]] = []
    for epoch in range(1, 9):
        recipient = 5700 + epoch
        records.append(
            {
                "epoch": epoch,
                "val_candidate_text_by_field": {
                    "amount": {
                        "records": 1428,
                        "exact_matches": 1400,
                        "exact_match": 1400 / 1428,
                    },
                    "time": {
                        "records": 3738,
                        "exact_matches": 3700,
                        "exact_match": 3700 / 3738,
                    },
                    "payment_method_field": {
                        "records": 5242,
                        "exact_matches": 5200,
                        "exact_match": 5200 / 5242,
                    },
                    "recipient_field": {
                        "records": 6789,
                        "exact_matches": recipient,
                        "exact_match": recipient / 6789,
                    },
                },
                "val_ctc_by_field": {
                    "transfer_status": {
                        "records": 111,
                        "exact_matches": 110,
                        "exact_match": 110 / 111,
                    }
                },
            }
        )
    summary = {
        **_summary_data_fields(),
        "initialization": {"financial_label_policy": _financial_label_policy()},
        "best_checkpoint_epoch": 8,
        "records": records,
    }
    path = tmp_path / "a8-summary.json"
    _write_json(path, summary)
    best = tmp_path / "a8-best.pt"
    best.write_bytes(b"a8-best")
    best_payload = {
        **_summary_data_fields(),
        **_ordered_labels(),
        "initialization": {"financial_label_policy": _financial_label_policy()},
    }
    monkeypatch.setattr(
        exact8, "_load_checkpoint", lambda *_a, **_k: best_payload
    )
    pilot = {
        "artifacts": {
            "candidate_training_summary": _binding(path),
            "candidate_best_checkpoint": _binding(best),
        }
    }

    baseline = exact8._a8_baseline(pilot, torch=object())
    assert baseline["candidate_denominators"] == {
        "amount": 1428,
        "time": 3738,
        "payment_method_field": 5242,
        "recipient_field": 6789,
        "transfer_status": 111,
    }
    # The historical trainer checkpoint explicitly persists only the two
    # optional text-head blanks.  The three fixed protocol blanks must remain
    # distinguishable from those checkpoint-sourced facts in the proof.
    assert not {
        "amount_blank_index",
        "time_blank_index",
        "payment_blank_index",
    } & set(best_payload)
    assert baseline["data_label_proof"]["blank_indices"] == {
        "amount_blank_index": {
            "source": "fixed_protocol_constants",
            "semantic": "NUMERIC_BLANK_INDEX",
            "value": 0,
        },
        "time_blank_index": {
            "source": "fixed_protocol_constants",
            "semantic": "NUMERIC_BLANK_INDEX",
            "value": 0,
        },
        "payment_blank_index": {
            "source": "fixed_protocol_constants",
            "semantic": "PAYMENT_BLANK_INDEX",
            "value": 0,
        },
        "status_text_blank_index": {
            "source": "A8_checkpoint_explicit",
            "semantic": "status_text_blank_index",
            "value": 0,
        },
        "recipient_blank_index": {
            "source": "A8_checkpoint_explicit",
            "semantic": "recipient_blank_index",
            "value": 0,
        },
    }
    for explicit_key in ("status_text_blank_index", "recipient_blank_index"):
        original = best_payload.pop(explicit_key)
        with pytest.raises(ValueError, match=explicit_key):
            exact8._a8_baseline(pilot, torch=object())
        best_payload[explicit_key] = original
    for constant_name in ("NUMERIC_BLANK_INDEX", "PAYMENT_BLANK_INDEX"):
        with monkeypatch.context() as context:
            context.setattr(exact8, constant_name, 1)
            with pytest.raises(ValueError, match=constant_name):
                exact8._a8_baseline(pilot, torch=object())

    summary["records"][1]["val_candidate_text_by_field"]["amount"] = {
        "records": 1427,
        "exact_matches": 1400,
        "exact_match": 1400 / 1427,
    }
    _write_json(path, summary)
    with pytest.raises(ValueError, match="amount candidate denominator changed"):
        exact8._a8_baseline(
            {
                "artifacts": {
                    "candidate_training_summary": _binding(path),
                    "candidate_best_checkpoint": _binding(best),
                }
            },
            torch=object(),
        )


@pytest.mark.parametrize(
    "mutation",
    ["raw_field_count", "payment_oov", "financial_policy"],
)
def test_a8_data_proof_rejects_coherent_summary_drift(
    tmp_path: Path, mutation: str
) -> None:
    inspection = _inspection(tmp_path)
    summary = _summary(inspection)
    if mutation == "raw_field_count":
        summary["field_counts"]["amount"]["train"] += 1
    elif mutation == "payment_oov":
        summary["payment_oov_by_split"]["val"]["records"] -= 1
    else:
        summary["initialization"]["financial_label_policy"]["payment_character_map"][
            "mode"
        ] = "forged"
    with pytest.raises(ValueError, match="A8-frozen"):
        exact8.evaluate_exact8_summary(
            summary, inspection=inspection, recipe=exact8._recipe(inspection)
        )


@pytest.mark.parametrize(
    "key",
    ["payment_characters", "status_text_characters", "payment_bank_prefix_classes"],
)
def test_a8_ordered_label_proof_rejects_payment_status_and_bank_map_drift(
    tmp_path: Path, key: str
) -> None:
    inspection = _inspection(tmp_path)
    summary = _summary(inspection)
    labels = _labels(summary)
    labels[key] = [*labels[key], "forged"]
    with pytest.raises(ValueError, match="A8-frozen label"):
        exact8._validate_labels(
            labels,
            summary=summary,
            data_label_proof=inspection["baseline"]["data_label_proof"],
        )


@pytest.mark.parametrize(
    "mutation",
    [
        "amount_blank_index",
        "time_blank_index",
        "payment_blank_index",
        "status_text_blank_index",
        "recipient_blank_index",
        "unexpected_extra",
        "missing_known_key",
    ],
)
def test_labels_require_all_zero_blank_indices_and_exact_schema(
    tmp_path: Path, mutation: str
) -> None:
    inspection = _inspection(tmp_path)
    summary = _summary(inspection)
    labels = _labels(summary)
    if mutation == "unexpected_extra":
        labels["unreviewed_claim"] = False
    elif mutation == "missing_known_key":
        labels.pop("recipient_input_preprocess")
    else:
        labels[mutation] = 1
    with pytest.raises(ValueError, match="labels|blank"):
        exact8._validate_labels(
            labels,
            summary=summary,
            data_label_proof=inspection["baseline"]["data_label_proof"],
        )


@pytest.mark.parametrize(
    ("key", "field", "forged"),
    [
        ("amount_blank_index", "source", "A8_checkpoint_explicit"),
        ("payment_blank_index", "semantic", "NUMERIC_BLANK_INDEX"),
        ("recipient_blank_index", "source", "fixed_protocol_constants"),
        ("status_text_blank_index", "semantic", "STATUS_TEXT_BLANK_INDEX"),
        ("time_blank_index", "unexpected", False),
    ],
)
def test_labels_reject_misrepresented_a8_blank_index_proof(
    tmp_path: Path, key: str, field: str, forged: object
) -> None:
    inspection = _inspection(tmp_path)
    summary = _summary(inspection)
    labels = _labels(summary)
    proof = inspection["baseline"]["data_label_proof"]
    proof["blank_indices"][key][field] = forged
    with pytest.raises(ValueError, match="blank|proof"):
        exact8._validate_labels(
            labels,
            summary=summary,
            data_label_proof=proof,
        )


@pytest.mark.parametrize(
    "claim",
    [
        "test_opened",
        "onnx_exported",
        "production_route_authorized",
        "warmstart_authorized",
        "same_route_retry_authorized",
        "failed_checkpoint_initialization_authorized",
    ],
)
@pytest.mark.parametrize("nested", [False, True])
def test_checkpoint_recursive_unsafe_claims_are_rejected(
    claim: str, nested: bool
) -> None:
    payload: dict[str, object] = {"state_dict": {"recipient.weight": object()}}
    if nested:
        payload["nested"] = [{"deeper": {claim: True}}]
    else:
        payload[claim] = True
    with pytest.raises(ValueError, match="unsafe true claim"):
        exact8._assert_checkpoint_has_no_unsafe_claims(
            payload, description="synthetic checkpoint"
        )


@pytest.mark.parametrize("mutation", ["missing", "extra", "shape", "dtype"])
def test_checkpoint_declared_model_abi_rejects_recipient_state_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    inspection = _inspection(tmp_path)
    summary = _summary(inspection)
    labels = _labels(summary)
    record = summary["records"][-1]
    original_state = _valid_declared_model_state()
    state = dict(original_state)
    expected_state = dict(original_state)

    class FakeModel:
        def state_dict(self) -> dict[str, object]:
            return expected_state

    monkeypatch.setattr(
        seed_sanitizer, "build_unified_reader", lambda **_kwargs: FakeModel()
    )
    ordered = _ordered_labels()
    monkeypatch.setattr(
        seed_sanitizer,
        "_checkpoint_labels",
        lambda _payload, *, config: (
            ordered["amount_characters"],
            ordered["time_characters"],
            ordered["payment_characters"],
            ordered["recipient_characters"],
            ordered["status_classes"],
            ordered["payment_bank_prefix_classes"],
        ),
    )
    monkeypatch.setattr(
        seed_sanitizer,
        "_checkpoint_status_text_characters",
        lambda _payload, *, config: ordered["status_text_characters"],
    )
    monkeypatch.setattr(
        seed_sanitizer,
        "_tensor_signature",
        lambda value, *, name: (value.dtype, tuple(value.shape)),
    )
    recipient_names = sorted(
        name for name in state if name.startswith("recipient_")
    )
    assert recipient_names
    if mutation == "missing":
        state.pop(recipient_names[0])
    elif mutation == "extra":
        state["recipient_forged.weight"] = _FakeTensor((1,))
    elif mutation == "shape":
        name = next(name for name in recipient_names if state[name].shape[0] > 1)
        tensor = state[name]
        state[name] = _FakeTensor((tensor.shape[0] - 1, *tensor.shape[1:]))
    else:
        name = recipient_names[0]
        state[name] = _FakeTensor(state[name].shape, "float64")
    checkpoint = _checkpoint_payload(
        summary=summary,
        labels=labels,
        record=record,
        state=state,
    )
    artifact = tmp_path / f"abi-{mutation}.pt"
    artifact.write_bytes(b"synthetic checkpoint")
    snapshot = exact8._freeze_file(artifact, description="synthetic ABI checkpoint")
    monkeypatch.setattr(exact8, "_load_checkpoint", lambda *_a, **_k: checkpoint)
    with pytest.raises(ValueError, match="tensor keys|shape/dtype"):
        exact8._checkpoint_artifact(
            snapshot,
            summary=summary,
            labels=labels,
            epoch_record=record,
            expected_epoch=8,
            source_state=original_state,
            torch=object(),
            description="synthetic ABI checkpoint",
        )


@pytest.mark.parametrize(
    "mutation",
    ["payment_characters", "status_classes", "payment_bank_prefix_classes", "raw_count"],
)
def test_checkpoint_rejects_data_and_ordered_label_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mutation: str
) -> None:
    inspection = _inspection(tmp_path)
    summary = _summary(inspection)
    labels = _labels(summary)
    record = summary["records"][-1]
    checkpoint = {
        "schema_version": 1,
        "kind": "receipt_unified_field_reader_v13",
        "epoch": 8,
        "state_dict": {"frozen.weight": "same"},
        "metrics": record,
        "recipient_loss_weight": 1.0,
        "ctc_loss_weight": 1.0,
        "structured_loss_weight": 1.0,
    }
    for key in (
        "config",
        "initialization",
        "fine_tune_policy",
        "status_text_runtime_policy",
        "training_runtime",
        "checkpoint_selection_policy",
        *exact8.A8_SUMMARY_DATA_KEYS,
    ):
        checkpoint[key] = summary[key]
    for key in (*exact8.A8_ORDERED_LABEL_KEYS,):
        checkpoint[key] = labels[key]
    checkpoint = json.loads(json.dumps(checkpoint))
    if mutation == "raw_count":
        checkpoint["field_counts"]["amount"]["train"] += 1
    else:
        checkpoint[mutation] = [*checkpoint[mutation], "forged"]
    artifact = tmp_path / "checkpoint.pt"
    artifact.write_bytes(b"frozen checkpoint bytes")
    snapshot = exact8._freeze_file(artifact, description="synthetic checkpoint")
    monkeypatch.setattr(exact8, "_load_checkpoint", lambda *_a, **_k: checkpoint)
    monkeypatch.setattr(
        exact8, "_require_checkpoint_without_optimizer_state", lambda *_a, **_k: None
    )
    monkeypatch.setattr(
        exact8, "_state_dict", lambda payload, **_k: payload["state_dict"]
    )
    monkeypatch.setattr(
        exact8,
        "_partition_descriptor",
        lambda _state, *, recipient: {"recipient": recipient, "identity": "same"},
    )
    with pytest.raises(ValueError, match="does not match"):
        exact8._checkpoint_artifact(
            snapshot,
            summary=summary,
            labels=labels,
            epoch_record=record,
            expected_epoch=8,
            source_state={"frozen.weight": "same"},
            torch=object(),
            description="synthetic exact8 checkpoint",
        )


def test_fixed_recipe_is_exactly_two_views_one_train_copy_and_eight_epochs(
    tmp_path: Path,
) -> None:
    inspection = _inspection(tmp_path)
    recipe = exact8._recipe(inspection)
    args = recipe["training_args"]

    assert recipe["selected_views"] == ["standard", "fixed_value"]
    assert recipe["selector_mode"] == "sha256_rank_parity_v1"
    assert recipe["train_multiplier"] == 1
    assert recipe["val_unchanged"] is True
    assert args["epochs"] == 8
    assert args["validation_every"] == 1
    assert args["init_checkpoint_mode"] == "recipient_visual_context_reinit"
    assert args["recipient_train_splits"] == ["train"]
    assert args["recipient_only_fine_tune"] is True
    assert recipe["code"] == inspection["code"]


def test_analysis_overlay_fixture_cannot_be_promoted_to_exact8_authority(
    tmp_path: Path,
) -> None:
    helpers = runpy.run_path(
        str(Path(__file__).with_name("test_recipient_multiview_overlay.py"))
    )
    fixture = helpers["_fixture"](tmp_path)
    output = tmp_path / "exact8-fixed2"
    contract = overlay_module._materialize_fixed2_overlay_analysis_test_only(
        multiview_root=fixture["multiview_root"],
        full_records=fixture["full_records"],
        blind_records=fixture["blind_records"],
        blind_contract=fixture["blind_contract"],
        original_dataset_root=fixture["dataset_root"],
        output_root=output,
    )
    analysis_marker = output / overlay_module.FIXED2_ANALYSIS_MARKER_NAME
    canonical_marker = output / overlay_module.FIXED2_CANONICAL_CONTRACT_NAME

    assert contract["kind"] == overlay_module.FIXED2_ANALYSIS_CONTRACT_KIND
    assert (
        contract["publication_authority"]
        == overlay_module.FIXED2_ANALYSIS_PUBLICATION_AUTHORITY
    )
    assert contract["consumer_optimizer_input_ready"] is False
    assert analysis_marker.is_file()
    assert not canonical_marker.exists()
    analysis_rejection = (
        "canonical commit-marker filename"
        if os.name == "nt"
        else "requires Windows publication authority"
    )
    with pytest.raises((OSError, ValueError), match=analysis_rejection):
        exact8._verify_overlay(
            contract_path=analysis_marker,
            full_records=fixture["full_records"],
            blind_records=fixture["blind_records"],
            blind_contract=fixture["blind_contract"],
            original_dataset_root=fixture["dataset_root"],
        )

    # A byte-for-byte analysis marker cannot gain formal authority merely by
    # being copied to the canonical filename.
    canonical_marker.write_bytes(analysis_marker.read_bytes())
    renamed_rejection = (
        "fixed2 kind" if os.name == "nt" else "requires Windows publication authority"
    )
    with pytest.raises((OSError, ValueError), match=renamed_rejection):
        exact8._verify_overlay(
            contract_path=canonical_marker,
            full_records=fixture["full_records"],
            blind_records=fixture["blind_records"],
            blind_contract=fixture["blind_contract"],
            original_dataset_root=fixture["dataset_root"],
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("publication_authority", "publication authority"),
        ("consumer_optimizer_input_ready", "optimizer-input readiness"),
    ],
)
def test_exact8_adapter_independently_requires_formal_overlay_authority(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
    message: str,
) -> None:
    multiview_root = tmp_path / "multiview"
    multiview_root.mkdir()
    contract_path = tmp_path / overlay_module.FIXED2_CANONICAL_CONTRACT_NAME
    _write_json(contract_path, {"multiview_root": str(multiview_root)})
    verified: dict[str, object] = {
        "kind": exact8.OVERLAY_KIND,
        "publication_authority": overlay_module.FIXED2_PUBLICATION_AUTHORITY,
        "consumer_optimizer_input_ready": True,
    }
    verified[mutation] = (
        overlay_module.FIXED2_ANALYSIS_PUBLICATION_AUTHORITY
        if mutation == "publication_authority"
        else False
    )
    monkeypatch.setattr(
        overlay_module,
        "verify_fixed2_overlay_contract",
        lambda **_kwargs: verified,
    )

    with pytest.raises(ValueError, match=message):
        exact8._verify_overlay(
            contract_path=contract_path,
            full_records=tmp_path / "unused-full",
            blind_records=tmp_path / "unused-blind",
            blind_contract=tmp_path / "unused-blind-contract",
            original_dataset_root=tmp_path / "unused-dataset",
        )


def test_output_anchor_dependency_is_bound_in_exact8_code_closure() -> None:
    paths = exact8._code_paths()
    assert paths["code_overlay"].name == "recipient_multiview_overlay.py"
    assert paths["code_overlay"].is_file()


def test_inspection_attempt_subject_excludes_code_and_guards_failure_closure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    full = tmp_path / "full.jsonl"
    full.write_text("{}\n", encoding="utf-8")
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    pilot = tmp_path / "pilot"
    pilot.mkdir()
    source_contract = tmp_path / "source.json"
    a8_evidence = tmp_path / "a8.json"
    failure_evidence = tmp_path / "failure.json"
    overlay_contract = tmp_path / "overlay.json"
    for path in (source_contract, a8_evidence, failure_evidence, overlay_contract):
        path.write_text("{}", encoding="utf-8")
    failure_registry = tmp_path / "failure-registry"
    failure_registry.mkdir()
    checkpoint = tmp_path / "source-best.pt"
    checkpoint.write_bytes(b"source")
    blind = tmp_path / "blind.jsonl"
    blind.write_text("{}\n", encoding="utf-8")
    blind_contract = tmp_path / "blind.contract.json"
    blind_contract.write_text("{}", encoding="utf-8")
    a8_summary = tmp_path / "a8-summary.json"
    a8_summary.write_text("{}", encoding="utf-8")
    failure_summary = tmp_path / "fresh60-summary.json"
    failure_best = tmp_path / "fresh60-best.pt"
    failure_last = tmp_path / "fresh60-last.pt"
    failure_attempt = tmp_path / "fresh60-attempt.json"
    failure_code = tmp_path / "failure-code.py"
    for index, path in enumerate(
        (failure_summary, failure_best, failure_last, failure_attempt, failure_code)
    ):
        path.write_bytes(f"failure-{index}".encode())
    composite = tmp_path / "fixed2.jsonl"
    composite.write_text("{}\n", encoding="utf-8")
    multiview_root = tmp_path / "multiview"
    multiview_root.mkdir()
    overlay_artifact = tmp_path / "overlay-artifact.json"
    overlay_artifact.write_text("{}", encoding="utf-8")
    drift_code = tmp_path / "exact-code.py"
    drift_code.write_text("version = 1\n", encoding="utf-8")

    source = {
        "kind": SOURCE_KIND,
        "source_subject_id": exact8.ATTESTED_SOURCE_SUBJECT_ID,
        "artifacts": {
            "best_checkpoint": _binding(checkpoint),
            "full_manifest": _binding(full),
        },
    }
    a8 = {
        "kind": CANDIDATE_PILOT_KIND,
        "source_subject_id": exact8.ATTESTED_SOURCE_SUBJECT_ID,
        "candidate_pilot_subject_id": exact8.ATTESTED_A8_SUBJECT_ID,
        "artifacts": {
            "candidate_blind_manifest": _binding(blind),
            "candidate_blind_contract": _binding(blind_contract),
            "candidate_training_summary": _binding(a8_summary),
        },
    }
    failure = {
        "kind": FAILURE_KIND,
        "analysis_only": True,
        "new_view_pilot_authority": True,
        "decision": FAILURE_DECISION,
        "authorization": FAILURE_AUTHORIZATION,
        "production_route_authorized": False,
        "same_route_retry_authorized": False,
        "same_route_continuation_authorized": False,
        "warmstart_authorized": False,
        "failed_checkpoint_initialization_authorized": False,
        "onnx_export_authorized": False,
        "test_evaluation_authorized": False,
        "test_opened": False,
        "onnx_exported": False,
        "authorization_scope": {
            "epochs": 8,
            "training_data_view": "must_differ_from_failed_standard_full_crop_view",
            "source_initialization": "same_attested_legacy_source_fresh_visual_context_reinit",
            "failed_best_checkpoint_use": "forbidden",
            "failed_last_checkpoint_use": "forbidden",
        },
        "source_subject_id": exact8.ATTESTED_SOURCE_SUBJECT_ID,
        "candidate_pilot_subject_id": exact8.ATTESTED_A8_SUBJECT_ID,
        "failure_subject_id": "b" * 64,
        "artifacts": {
            "training_summary": _binding(failure_summary),
            "best_checkpoint": _binding(failure_best),
            "last_checkpoint": _binding(failure_last),
            "attempt_lock": _binding(failure_attempt),
        },
        "code": {"failure_attestor": _binding(failure_code)},
    }
    b8 = {
        "artifacts": {
            "source_best_checkpoint": _binding(checkpoint),
            "full_manifest": _binding(full),
        }
    }
    overlay = {
        "overlay_subject_id": "c" * 64,
        "composite_records": str(composite.resolve()),
        "composite_dataset_root": str(tmp_path.resolve()),
        "multiview_root": str(multiview_root.resolve()),
        "artifacts": {"overlay_artifact": _binding(overlay_artifact)},
    }
    _source, target = _configs()
    monkeypatch.setattr(exact8, "verify_full_crop_candidate_source", lambda **_: source)
    monkeypatch.setattr(exact8, "verify_residual_candidate_pilot", lambda **_: a8)
    monkeypatch.setattr(exact8, "verify_fresh60_failure", lambda **_: failure)
    monkeypatch.setattr(exact8, "_recompute_pilot_closure", lambda *_a, **_k: (b8, {}, {}))
    monkeypatch.setattr(exact8, "_verify_overlay", lambda **_: overlay)
    monkeypatch.setattr(
        exact8,
        "_a8_baseline",
        lambda _value, **_kwargs: {
            "best_epoch": 8,
            "best_matches": 5722,
            "epoch4_matches": 5500,
            "epoch8_matches": 5722,
            "candidate_denominators": {
                "amount": 1428,
                "time": 3738,
                "payment_method_field": 5242,
                "recipient_field": 6789,
                "transfer_status": 100,
            },
            "data_label_proof": _data_label_proof(),
        },
    )
    monkeypatch.setattr(
        exact8, "_target_config", lambda *_a, **_k: UnifiedReaderConfig(**target)
    )
    monkeypatch.setattr(exact8, "_code_paths", lambda: {"code_exact8": drift_code})
    kwargs = {
        "full_records": full,
        "original_dataset_root": dataset,
        "full_crop_pilot_root": pilot,
        "source_contract_path": source_contract,
        "candidate_pilot_evidence_path": a8_evidence,
        "failure_evidence_path": failure_evidence,
        "failure_attempt_registry": failure_registry,
        "overlay_contract_path": overlay_contract,
        "torch": object(),
    }

    first = exact8.inspect_exact8_subject(**kwargs)
    drift_code.write_text("version = 2\n", encoding="utf-8")
    second = exact8.inspect_exact8_subject(**kwargs)

    assert first["route_subject_id"] == second["route_subject_id"]
    assert first["attempt_id"] == second["attempt_id"]
    assert first["code"] != second["code"]
    guarded = set(first["guard_paths"])
    for path in (
        failure_summary,
        failure_best,
        failure_last,
        failure_attempt,
        failure_code,
        overlay_artifact,
    ):
        assert str(path.resolve()) in guarded
    expected_directories = sorted(
        (str(dataset.resolve()), str(multiview_root.resolve()))
    )
    assert first["guard_directories"] == expected_directories
    assert [item["path"] for item in first["guard_directory_identities"]] == (
        expected_directories
    )

    prior_dataset = tmp_path / "dataset-before-replacement"
    dataset.rename(prior_dataset)
    dataset.mkdir()
    replaced = exact8.inspect_exact8_subject(**kwargs)
    assert replaced["route_subject_id"] == second["route_subject_id"]
    assert replaced["guard_directory_identities"] != second[
        "guard_directory_identities"
    ]
    with pytest.raises(ValueError, match="authority closure"):
        exact8._json_equal(replaced, second, "exact8 authority closure")


def test_attempt_must_samefile_real_common_application_data_registry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inspection = _inspection(tmp_path)
    output = tmp_path / "output"
    program_data = tmp_path / "ProgramData"
    registry = (
        program_data / exact8.ATTEMPT_REGISTRY_PARENT / exact8.ATTEMPT_REGISTRY_NAME
    )
    canonical = registry / f"{inspection['attempt_id']}.attempt.json"
    _attempt(canonical, inspection, output)
    monkeypatch.setattr(exact8, "_common_application_data_path", lambda: program_data)
    snapshot = exact8._freeze_file(canonical, description="canonical attempt")
    exact8._validate_attempt(
        canonical,
        inspection=inspection,
        output_root=output,
        snapshot=snapshot,
    )

    fake = (
        tmp_path
        / "fake"
        / exact8.ATTEMPT_REGISTRY_PARENT
        / exact8.ATTEMPT_REGISTRY_NAME
        / canonical.name
    )
    _attempt(fake, inspection, output)
    with pytest.raises(ValueError, match="CommonApplicationData"):
        exact8._validate_attempt(
            fake,
            inspection=inspection,
            output_root=output,
            snapshot=exact8._freeze_file(fake, description="fake attempt"),
        )


def test_program_data_known_folder_abi_ignores_programdata_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    trusted = tmp_path / "trusted-ProgramData"
    attacker = tmp_path / "attacker-ProgramData"
    path_buffer = ctypes.create_unicode_buffer(str(trusted))
    captured: dict[str, object] = {}

    def co_initialize(reserved: object, flags: int) -> int:
        captured["co_initialize"] = (reserved, flags)
        return 0

    def get_known_folder(
        guid_pointer: object,
        flags: int,
        token: object,
        output_pointer: object,
    ) -> int:
        guid = guid_pointer._obj  # type: ignore[attr-defined]
        captured["guid"] = (
            guid.data1,
            guid.data2,
            guid.data3,
            tuple(guid.data4),
        )
        captured["known_folder_args"] = (flags, token)
        output_pointer._obj.value = ctypes.addressof(path_buffer)  # type: ignore[attr-defined]
        return 0

    fake_co_initialize = _FakeWinFunction(co_initialize)
    fake_get_known_folder = _FakeWinFunction(get_known_folder)
    fake_free = _FakeWinFunction(
        lambda pointer: captured.update({"freed": pointer.value})
    )
    fake_uninitialize = _FakeWinFunction(
        lambda: captured.update({"co_uninitialize": True})
    )

    class _Shell32:
        SHGetKnownFolderPath = fake_get_known_folder

    class _Ole32:
        CoInitializeEx = fake_co_initialize
        CoTaskMemFree = fake_free
        CoUninitialize = fake_uninitialize

    monkeypatch.setenv("PROGRAMDATA", str(attacker))
    monkeypatch.setattr(
        exact8.ctypes,
        "WinDLL",
        lambda name, **_kwargs: _Shell32() if name == "shell32" else _Ole32(),
        raising=False,
    )

    assert exact8._windows_known_program_data_path() == trusted
    assert captured == {
        "co_initialize": (None, 0x2),
        "guid": exact8._WINDOWS_PROGRAM_DATA_GUID,
        "known_folder_args": (0, None),
        "freed": ctypes.addressof(path_buffer),
        "co_uninitialize": True,
    }
    assert fake_get_known_folder.argtypes == (
        ctypes.POINTER(exact8._WindowsGuid),
        ctypes.c_uint32,
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_void_p),
    )
    assert fake_get_known_folder.restype == ctypes.c_int32
    assert ctypes.sizeof(exact8._WindowsGuid) == 16
    assert ctypes.sizeof(exact8._WindowsAceHeader) == 4
    assert exact8._WindowsAccessDeniedAce.sid_start.offset == 8
    assert ctypes.sizeof(exact8._WindowsGenericMapping) == 16


@pytest.mark.parametrize(
    ("mode", "freed"),
    [("hresult", True), ("null", False), ("empty", True)],
)
def test_program_data_known_folder_failures_are_closed_and_freed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
    freed: bool,
) -> None:
    path_buffer = ctypes.create_unicode_buffer("" if mode == "empty" else str(tmp_path))
    free_calls: list[int] = []
    uninitializations: list[bool] = []

    def get_known_folder(
        _guid: object,
        _flags: int,
        _token: object,
        output_pointer: object,
    ) -> int:
        if mode != "null":
            output_pointer._obj.value = ctypes.addressof(path_buffer)  # type: ignore[attr-defined]
        return ctypes.c_int32(0x80004005).value if mode == "hresult" else 0

    class _Shell32:
        SHGetKnownFolderPath = _FakeWinFunction(get_known_folder)

    class _Ole32:
        CoInitializeEx = _FakeWinFunction(lambda _reserved, _flags: 0)
        CoTaskMemFree = _FakeWinFunction(
            lambda pointer: free_calls.append(pointer.value)
        )
        CoUninitialize = _FakeWinFunction(
            lambda: uninitializations.append(True)
        )

    monkeypatch.setattr(
        exact8.ctypes,
        "WinDLL",
        lambda name, **_kwargs: _Shell32() if name == "shell32" else _Ole32(),
        raising=False,
    )
    with pytest.raises(ValueError, match="FOLDERID_ProgramData"):
        exact8._windows_known_program_data_path()
    assert len(free_calls) == int(freed)
    assert uninitializations == [True]


def test_cli_run_rejects_attempt_under_forged_programdata_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inspection = _inspection(tmp_path)
    trusted_program_data = tmp_path / "trusted-ProgramData"
    trusted_registry = (
        trusted_program_data
        / exact8.ATTEMPT_REGISTRY_PARENT
        / exact8.ATTEMPT_REGISTRY_NAME
    )
    trusted_registry.mkdir(parents=True)
    forged_program_data = tmp_path / "forged-ProgramData"
    forged_registry = (
        forged_program_data
        / exact8.ATTEMPT_REGISTRY_PARENT
        / exact8.ATTEMPT_REGISTRY_NAME
    )
    forged_registry.mkdir(parents=True)
    forged_attempt = forged_registry / f"{inspection['attempt_id']}.attempt.json"
    monkeypatch.setenv("PROGRAMDATA", str(forged_program_data))
    monkeypatch.setattr(
        exact8,
        "_common_application_data_path",
        lambda: trusted_program_data,
    )

    def validate_cli_attempt(**kwargs: object) -> dict[str, object]:
        exact8._expected_attempt_path(
            Path(str(kwargs["attempt_lock"])),
            inspection=inspection,
        )
        return {"passed": True}

    monkeypatch.setattr(exact8, "run_exact8", validate_cli_attempt)
    with pytest.raises(ValueError, match="CommonApplicationData"):
        exact8.main(
            [
                "run",
                "--full-records",
                "unused-full",
                "--dataset-root",
                "unused-data",
                "--full-crop-pilot-root",
                "unused-pilot",
                "--source-contract",
                "unused-source",
                "--candidate-pilot-evidence",
                "unused-a8",
                "--failure-evidence",
                "unused-failure",
                "--failure-attempt-registry",
                "unused-failure-registry",
                "--overlay-contract",
                "unused-overlay",
                "--output-root",
                "unused-output",
                "--attempt-lock",
                str(forged_attempt),
            ]
        )


def test_valid_registry_marker_and_programdata_acl_evidence_is_accepted() -> None:
    exact8._require_windows_acl_evidence(
        description="registry DACL",
        deny_aces=[(
            exact8._WINDOWS_DELETE | exact8._WINDOWS_FILE_DELETE_CHILD,
            exact8._WINDOWS_OBJECT_INHERIT_ACE
            | exact8._WINDOWS_CONTAINER_INHERIT_ACE,
        )],
        required_mask=exact8._WINDOWS_DELETE | exact8._WINDOWS_FILE_DELETE_CHILD,
        required_flags=exact8._WINDOWS_OBJECT_INHERIT_ACE
        | exact8._WINDOWS_CONTAINER_INHERIT_ACE,
        forbidden_flags=exact8._WINDOWS_INHERITED_ACE
        | exact8._WINDOWS_INHERIT_ONLY_ACE,
        effective_access={
            exact8._WINDOWS_DELETE: False,
            exact8._WINDOWS_FILE_DELETE_CHILD: False,
        },
    )
    exact8._require_windows_acl_evidence(
        description="marker DACL",
        deny_aces=[(exact8._WINDOWS_DELETE, exact8._WINDOWS_INHERITED_ACE)],
        required_mask=exact8._WINDOWS_DELETE,
        required_flags=exact8._WINDOWS_INHERITED_ACE,
        forbidden_flags=exact8._WINDOWS_INHERIT_ONLY_ACE,
        effective_access={exact8._WINDOWS_DELETE: False},
    )
    exact8._require_windows_acl_evidence(
        description="ProgramData DACL",
        deny_aces=[],
        required_mask=None,
        required_flags=0,
        forbidden_flags=0,
        effective_access={},
    )


def test_programdata_admin_delete_child_is_out_of_scope_but_child_acl_gates_remain(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[dict[str, object]] = []

    monkeypatch.setattr(exact8.os, "name", "nt")
    monkeypatch.setattr(
        exact8,
        "_require_windows_acl_policy",
        lambda handle, **kwargs: captured.append({"handle": handle, **kwargs}),
    )
    exact8._require_attempt_program_data_acl(
        SimpleNamespace(windows_handle=1701)
    )
    exact8._require_attempt_receipt_root_acl(
        SimpleNamespace(windows_handle=1702)
    )
    exact8._require_attempt_registry_acl(SimpleNamespace(windows_handle=1703))
    exact8._require_attempt_marker_acl(1704)
    child_mask = exact8._WINDOWS_DELETE | exact8._WINDOWS_FILE_DELETE_CHILD
    child_flags = (
        exact8._WINDOWS_OBJECT_INHERIT_ACE
        | exact8._WINDOWS_CONTAINER_INHERIT_ACE
    )
    assert captured == [
        {
            "handle": 1701,
            "description": "exact8 ProgramData DACL",
            "required_mask": None,
            "required_flags": 0,
            "forbidden_flags": 0,
            "effective_denied_accesses": (),
        },
        {
            "handle": 1702,
            "description": "exact8 ReceiptAI root DACL",
            "required_mask": child_mask,
            "required_flags": child_flags,
            "forbidden_flags": (
                exact8._WINDOWS_INHERITED_ACE
                | exact8._WINDOWS_INHERIT_ONLY_ACE
            ),
            "effective_denied_accesses": (
                exact8._WINDOWS_DELETE,
                exact8._WINDOWS_FILE_DELETE_CHILD,
            ),
        },
        {
            "handle": 1703,
            "description": "exact8 attempt registry DACL",
            "required_mask": child_mask,
            "required_flags": child_flags,
            "forbidden_flags": (
                exact8._WINDOWS_INHERITED_ACE
                | exact8._WINDOWS_INHERIT_ONLY_ACE
            ),
            "effective_denied_accesses": (
                exact8._WINDOWS_DELETE,
                exact8._WINDOWS_FILE_DELETE_CHILD,
            ),
        },
        {
            "handle": 1704,
            "description": "exact8 attempt marker DACL",
            "required_mask": exact8._WINDOWS_DELETE,
            "required_flags": exact8._WINDOWS_INHERITED_ACE,
            "forbidden_flags": exact8._WINDOWS_INHERIT_ONLY_ACE,
            "effective_denied_accesses": (exact8._WINDOWS_DELETE,),
        },
    ]
    assert "local-administrator bypass are out of scope" in exact8.ATTEMPT_THREAT_MODEL


@pytest.mark.parametrize(
    ("deny_aces", "effective", "message"),
    [
        ([], {0x10000: False, 0x40: False}, "deny ACE"),
        ([(0x10000, 0x03)], {0x10000: False, 0x40: False}, "deny ACE"),
        ([(0x40, 0x03)], {0x10000: False, 0x40: False}, "deny ACE"),
        ([(0x10040, 0x01)], {0x10000: False, 0x40: False}, "deny ACE"),
        ([(0x10040, 0x02)], {0x10000: False, 0x40: False}, "deny ACE"),
        ([(0x10040, 0x13)], {0x10000: False, 0x40: False}, "deny ACE"),
        ([(0x10040, 0x0B)], {0x10000: False, 0x40: False}, "deny ACE"),
        ([(0x10040, 0x03)], {0x10000: True, 0x40: False}, "effectively deny"),
        ([(0x10040, 0x03)], {0x10000: False, 0x40: True}, "effectively deny"),
    ],
)
def test_registry_acl_requires_explicit_inheritable_and_effective_delete_denies(
    deny_aces: list[tuple[int, int]],
    effective: dict[int, bool],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        exact8._require_windows_acl_evidence(
            description="registry DACL",
            deny_aces=deny_aces,
            required_mask=exact8._WINDOWS_DELETE
            | exact8._WINDOWS_FILE_DELETE_CHILD,
            required_flags=exact8._WINDOWS_OBJECT_INHERIT_ACE
            | exact8._WINDOWS_CONTAINER_INHERIT_ACE,
            forbidden_flags=exact8._WINDOWS_INHERITED_ACE
            | exact8._WINDOWS_INHERIT_ONLY_ACE,
            effective_access=effective,
        )


@pytest.mark.parametrize(
    ("deny_aces", "effective", "message"),
    [
        ([], {0x10000: False}, "deny ACE"),
        ([(0, 0x10)], {0x10000: False}, "deny ACE"),
        ([(0x10000, 0)], {0x10000: False}, "deny ACE"),
        ([(0x10000, 0x18)], {0x10000: False}, "deny ACE"),
        ([(0x10000, 0x10)], {0x10000: True}, "effectively deny"),
    ],
)
def test_marker_acl_requires_inherited_and_effective_delete_deny(
    deny_aces: list[tuple[int, int]],
    effective: dict[int, bool],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        exact8._require_windows_acl_evidence(
            description="marker DACL",
            deny_aces=deny_aces,
            required_mask=exact8._WINDOWS_DELETE,
            required_flags=exact8._WINDOWS_INHERITED_ACE,
            forbidden_flags=exact8._WINDOWS_INHERIT_ONLY_ACE,
            effective_access=effective,
        )


@pytest.mark.parametrize("allowed", [False, True])
def test_access_check_uses_access_status_not_api_success(
    monkeypatch: pytest.MonkeyPatch, allowed: bool
) -> None:
    captured: dict[str, object] = {}

    def access_check(
        security_descriptor: object,
        token: object,
        desired_access: int,
        mapping_pointer: object,
        _privileges: object,
        _privilege_length: object,
        granted_pointer: object,
        status_pointer: object,
    ) -> int:
        mapping = mapping_pointer._obj  # type: ignore[attr-defined]
        captured.update(
            {
                "security_descriptor": security_descriptor.value,
                "token": token.value,
                "desired_access": desired_access,
                "mapping": (
                    mapping.generic_read,
                    mapping.generic_write,
                    mapping.generic_execute,
                    mapping.generic_all,
                ),
            }
        )
        granted_pointer._obj.value = desired_access if allowed else 0  # type: ignore[attr-defined]
        status_pointer._obj.value = int(allowed)  # type: ignore[attr-defined]
        return 1

    fake_access_check = _FakeWinFunction(access_check)

    class _Advapi32:
        AccessCheck = fake_access_check

    monkeypatch.setattr(
        exact8.ctypes,
        "WinDLL",
        lambda _name, **_kwargs: _Advapi32(),
        raising=False,
    )
    assert exact8._windows_access_allowed(
        401,
        impersonation_token=402,
        desired_access=exact8._WINDOWS_DELETE,
    ) is allowed
    assert captured == {
        "security_descriptor": 401,
        "token": 402,
        "desired_access": exact8._WINDOWS_DELETE,
        "mapping": (
            exact8._WINDOWS_FILE_GENERIC_READ,
            exact8._WINDOWS_FILE_GENERIC_WRITE,
            exact8._WINDOWS_FILE_GENERIC_EXECUTE,
            exact8._WINDOWS_FILE_ALL_ACCESS,
        ),
    }
    assert fake_access_check.restype == ctypes.c_int32


def test_security_descriptor_is_read_from_anchored_handle_with_read_control(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}
    freed: list[int] = []

    def get_security_info(
        handle: object,
        object_type: int,
        security_information: int,
        owner_pointer: object,
        group_pointer: object,
        dacl_pointer: object,
        _sacl: object,
        descriptor_pointer: object,
    ) -> int:
        captured.update(
            {
                "handle": handle.value,
                "object_type": object_type,
                "security_information": security_information,
            }
        )
        owner_pointer._obj.value = 503  # type: ignore[attr-defined]
        group_pointer._obj.value = 504  # type: ignore[attr-defined]
        dacl_pointer._obj.value = 502  # type: ignore[attr-defined]
        descriptor_pointer._obj.value = 501  # type: ignore[attr-defined]
        return 0

    fake_get_security_info = _FakeWinFunction(get_security_info)
    fake_local_free = _FakeWinFunction(
        lambda pointer: freed.append(pointer.value) or None
    )

    class _Advapi32:
        GetSecurityInfo = fake_get_security_info

    class _Kernel32:
        LocalFree = fake_local_free

    monkeypatch.setattr(
        exact8.ctypes,
        "WinDLL",
        lambda name, **_kwargs: _Advapi32() if name == "advapi32" else _Kernel32(),
        raising=False,
    )
    descriptor, dacl = exact8._windows_security_descriptor(500)
    assert (descriptor, dacl) == (501, 502)
    assert captured == {
        "handle": 500,
        "object_type": exact8._WINDOWS_SE_FILE_OBJECT,
        "security_information": (
            exact8._WINDOWS_OWNER_SECURITY_INFORMATION
            | exact8._WINDOWS_GROUP_SECURITY_INFORMATION
            | exact8._WINDOWS_DACL_SECURITY_INFORMATION
        ),
    }
    exact8._windows_free_security_descriptor(descriptor)
    assert freed == [501]
    assert fake_get_security_info.restype == ctypes.c_uint32
    assert "GetNamedSecurityInfo" not in inspect.getsource(exact8)
    create_source = inspect.getsource(exact8._create_attempt_file_lease)
    assert "_WINDOWS_READ_CONTROL" in create_source


@pytest.mark.parametrize("missing", ["owner", "group"])
def test_security_descriptor_requires_owner_and_group_for_access_check(
    monkeypatch: pytest.MonkeyPatch,
    missing: str,
) -> None:
    freed: list[int] = []

    def get_security_info(
        _handle: object,
        _object_type: int,
        _security_information: int,
        owner_pointer: object,
        group_pointer: object,
        dacl_pointer: object,
        _sacl: object,
        descriptor_pointer: object,
    ) -> int:
        if missing != "owner":
            owner_pointer._obj.value = 503  # type: ignore[attr-defined]
        if missing != "group":
            group_pointer._obj.value = 504  # type: ignore[attr-defined]
        dacl_pointer._obj.value = 502  # type: ignore[attr-defined]
        descriptor_pointer._obj.value = 501  # type: ignore[attr-defined]
        return 0

    class _Advapi32:
        GetSecurityInfo = _FakeWinFunction(get_security_info)

    class _Kernel32:
        LocalFree = _FakeWinFunction(
            lambda pointer: freed.append(pointer.value) or None
        )

    monkeypatch.setattr(
        exact8.ctypes,
        "WinDLL",
        lambda name, **_kwargs: _Advapi32() if name == "advapi32" else _Kernel32(),
        raising=False,
    )
    with pytest.raises(ValueError, match="lacks owner/group"):
        exact8._windows_security_descriptor(500)
    assert freed == [501]


def test_current_sid_deny_ace_parser_uses_getace_borrowed_pointer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sid = b"SID!"
    ace_buffer = ctypes.create_string_buffer(8 + len(sid))
    ace = ctypes.cast(
        ace_buffer, ctypes.POINTER(exact8._WindowsAccessDeniedAce)
    ).contents
    ace.header.ace_type = exact8._WINDOWS_ACCESS_DENIED_ACE_TYPE
    ace.header.ace_flags = exact8._WINDOWS_OBJECT_INHERIT_ACE
    ace.header.ace_size = len(ace_buffer)
    ace.mask = exact8._WINDOWS_DELETE
    ctypes.memmove(
        ctypes.addressof(ace_buffer)
        + exact8._WindowsAccessDeniedAce.sid_start.offset,
        sid,
        len(sid),
    )

    def get_acl_information(
        _dacl: object,
        information_pointer: object,
        _length: int,
        information_class: int,
    ) -> int:
        assert information_class == exact8._WINDOWS_ACL_SIZE_INFORMATION
        information_pointer._obj.ace_count = 1  # type: ignore[attr-defined]
        return 1

    def get_ace(
        _dacl: object, index: int, ace_pointer: object
    ) -> int:
        assert index == 0
        ace_pointer._obj.value = ctypes.addressof(ace_buffer)  # type: ignore[attr-defined]
        return 1

    fake_get_ace = _FakeWinFunction(get_ace)

    class _Advapi32:
        IsValidAcl = _FakeWinFunction(lambda _dacl: 1)
        GetAclInformation = _FakeWinFunction(get_acl_information)
        EqualSid = _FakeWinFunction(
            lambda left, right: int(
                ctypes.string_at(left, len(sid)) == ctypes.string_at(right, len(sid))
            )
        )
        IsValidSid = _FakeWinFunction(lambda _sid: 1)
        GetLengthSid = _FakeWinFunction(lambda _sid: len(sid))
        GetAce = fake_get_ace

    monkeypatch.setattr(
        exact8.ctypes,
        "WinDLL",
        lambda _name, **_kwargs: _Advapi32(),
        raising=False,
    )
    assert exact8._windows_current_sid_deny_aces(
        701, current_sid=sid
    ) == [(exact8._WINDOWS_DELETE, exact8._WINDOWS_OBJECT_INHERIT_ACE)]
    assert fake_get_ace.argtypes == (
        ctypes.c_void_p,
        ctypes.c_uint32,
        ctypes.POINTER(ctypes.c_void_p),
    )
    assert fake_get_ace.restype == ctypes.c_int32


def test_registry_acl_failure_precedes_attempt_create_new(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inspection = _inspection(tmp_path)
    attempt = _formal_attempt_path(tmp_path, inspection, monkeypatch)
    created: list[Path] = []
    monkeypatch.setattr(
        exact8,
        "_require_attempt_registry_acl",
        lambda _lease: (_ for _ in ()).throw(
            ValueError("exact8 attempt registry DACL rejected")
        ),
    )
    monkeypatch.setattr(
        exact8,
        "_create_attempt_file_lease",
        lambda path, _payload: created.append(path),
    )

    with pytest.raises(ValueError, match="exact8 attempt registry DACL rejected"):
        exact8._consume_attempt(
            attempt,
            inspection=inspection,
            output_root=tmp_path / "unused-output",
        )
    assert created == []
    assert not attempt.exists()


def test_marker_acl_failure_after_create_new_retains_consumed_marker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    marker = tmp_path / "attempt.json"
    closed: list[int] = []
    payload_written: list[bytes] = []

    def create_file(
        _path: str,
        _desired: int,
        _share: int,
        _security: object,
        _disposition: int,
        _flags: int,
        _template: object,
    ) -> int:
        marker.write_bytes(b"")
        return 601

    def write_file(
        _handle: object,
        buffer: object,
        length: int,
        written_pointer: object,
        _overlapped: object,
    ) -> int:
        payload = ctypes.string_at(buffer, length)
        marker.write_bytes(payload)
        payload_written.append(payload)
        written_pointer._obj.value = length  # type: ignore[attr-defined]
        return 1

    class _Kernel32:
        CreateFileW = _FakeWinFunction(create_file)
        WriteFile = _FakeWinFunction(write_file)
        FlushFileBuffers = _FakeWinFunction(lambda _handle: 1)

    monkeypatch.setattr(exact8.os, "name", "nt")
    monkeypatch.setattr(
        exact8.ctypes,
        "WinDLL",
        lambda _name, **_kwargs: _Kernel32(),
        raising=False,
    )
    monkeypatch.setattr(
        exact8,
        "_windows_attempt_handle_identity",
        lambda _handle: (1, 2, marker.stat().st_size, 0),
    )
    monkeypatch.setattr(
        exact8,
        "_require_attempt_marker_acl",
        lambda _handle: (_ for _ in ()).throw(
            ValueError("exact8 attempt marker DACL inherited deny missing")
        ),
    )
    monkeypatch.setattr(
        exact8,
        "_windows_close_native_handle",
        lambda handle: closed.append(handle),
    )

    with pytest.raises(
        ValueError,
        match="exact8 attempt marker DACL inherited deny missing",
    ):
        exact8._create_attempt_file_lease(marker, b"consumed\n")
    assert payload_written == [b"consumed\n"]
    assert marker.read_bytes() == b"consumed\n"
    assert closed == [601]


@pytest.mark.parametrize("mode", ["in_place_mutation", "same_bytes_inode_swap"])
def test_frozen_artifact_rejects_permanent_and_swap_races(
    tmp_path: Path, mode: str
) -> None:
    artifact = tmp_path / "summary.json"
    artifact.write_bytes(b'{"version":1}\n')
    snapshot = exact8._freeze_file(artifact, description="race artifact")
    frozen_binding = exact8._binding_from_frozen(snapshot)
    if mode == "in_place_mutation":
        artifact.write_bytes(b'{"version":2}\n')
    else:
        replacement = tmp_path / "replacement.json"
        replacement.write_bytes(snapshot.data)
        replacement.replace(artifact)

    with pytest.raises(ValueError, match="changed after semantic verification"):
        exact8._require_frozen_current(snapshot, description="race artifact")
    assert frozen_binding["sha256"] == hashlib.sha256(snapshot.data).hexdigest()
    assert frozen_binding["size_bytes"] == len(snapshot.data)


@pytest.mark.parametrize("passed", [True, False])
def test_mocked_run_uses_original_source_and_independent_verifier_replays_decision(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, passed: bool
) -> None:
    inspection = _inspection(tmp_path)
    output = tmp_path / ("run-pass" if passed else "run-fail")
    attempt = _formal_attempt_path(tmp_path, inspection, monkeypatch)
    captured: dict[str, Any] = {}
    matches = (
        [5500, 5550, 5600, 5654, 5700, 5740, 5770, 5790]
        if passed
        else [5500, 5550, 5600, 5654, 5700, 5740, 5770, 5789]
    )
    summary = _summary(inspection, matches=matches)
    labels = _labels(summary)
    anchor_checkpoints: list[str] = []
    observed_output_anchors: list[exact8._GuardedOutputDirectory] = []
    original_require = exact8._GuardedOutputDirectory.require

    def record_anchor_checkpoint(
        self: exact8._GuardedOutputDirectory, checkpoint: str
    ) -> None:
        if self not in observed_output_anchors:
            observed_output_anchors.append(self)
        anchor_checkpoints.append(checkpoint)
        original_require(self, checkpoint)

    monkeypatch.setattr(
        exact8._GuardedOutputDirectory,
        "require",
        record_anchor_checkpoint,
    )
    attempt_checkpoints: list[str] = []
    observed_attempt_guards: list[exact8._GuardedAttempt] = []
    original_attempt_require = exact8._GuardedAttempt.require

    def record_attempt_checkpoint(
        self: exact8._GuardedAttempt, checkpoint: str
    ) -> None:
        assert self.closed is False
        assert self.file_lease.closed is False
        if self not in observed_attempt_guards:
            observed_attempt_guards.append(self)
        attempt_checkpoints.append(checkpoint)
        original_attempt_require(self, checkpoint)

    monkeypatch.setattr(
        exact8._GuardedAttempt,
        "require",
        record_attempt_checkpoint,
    )

    def fake_train(**kwargs: Any) -> None:
        captured.update(kwargs)
        training = Path(kwargs["output_dir"])
        training.mkdir(parents=True, exist_ok=True)
        _write_json(training / "training_summary.json", summary)
        _write_json(training / "labels.json", labels)
        (training / "best.pt").write_bytes(b"best")
        (training / "last.pt").write_bytes(b"last")

    monkeypatch.setattr(exact8, "inspect_exact8_subject", lambda **_: inspection)
    monkeypatch.setattr(exact8, "train_unified_reader", fake_train)
    monkeypatch.setattr(exact8, "_checkpoint_artifact", lambda *_a, **_k: {})
    monkeypatch.setattr(
        exact8,
        "_load_checkpoint",
        lambda *_a, **_k: {"state_dict": {"frozen.weight": b"same"}},
    )

    sealed = exact8.run_exact8(
        full_records=tmp_path / "unused-full",
        original_dataset_root=tmp_path / "unused-dataset",
        full_crop_pilot_root=tmp_path / "unused-pilot",
        source_contract_path=tmp_path / "unused-source",
        candidate_pilot_evidence_path=tmp_path / "unused-a8",
        failure_evidence_path=tmp_path / "unused-failure",
        failure_attempt_registry=tmp_path / "unused-registry",
        overlay_contract_path=tmp_path / "unused-overlay",
        output_root=output,
        attempt_lock=attempt,
        torch=object(),
    )

    assert sealed["passed"] is passed
    assert captured["epochs"] == 8
    assert captured["validation_every"] == 1
    assert captured["records_path"] == Path(str(inspection["composite_records"]))
    assert captured["dataset_root"] == Path(str(inspection["composite_dataset_root"]))
    assert captured["init_checkpoint"] == Path(str(inspection["source_checkpoint"]))
    assert captured["init_checkpoint_mode"] == "recipient_visual_context_reinit"
    assert sealed["pass_authorization"] is (sealed["pass_authorization"] if passed else None)
    closure = sealed["overlay_closure"]
    assert closure["continuous_train_and_validation_image_read_leases"] is False
    assert closure["opening_directory_identities"] == []
    assert "selected train overlay" in closure["residual_risk"]
    assert "validation image swap-and-restore" in closure["residual_risk"]
    assert sealed["output_closure"] == {
        "formal_platform": "windows",
        "program_data_resolved_by_known_folder_api": True,
        "one_shot_marker_created_by_python_run": True,
        "program_data_directory_identity_lease_held": True,
        "program_data_acl_descriptor_reverified": True,
        "program_data_delete_child_accesscheck_required": False,
        "program_data_delete_child_threat_scope": (
            "excluded_elevated_local_admin_and_owner_WRITE_DAC"
        ),
        "receipt_root_explicit_inheritable_delete_dacl": True,
        "attempt_registry_explicit_inheritable_delete_dacl": True,
        "attempt_marker_inherited_delete_dacl": True,
        "attempt_acl_reverified_before_guard_close": True,
        "owner_write_dac_and_local_admin_bypass_out_of_scope": True,
        "program_data_receipt_root_registry_deny_delete_leases": True,
        "attempt_registry_deny_delete_lease": True,
        "attempt_file_deny_write_delete_lease": True,
        "output_parent_deny_delete_lease": True,
        "output_and_training_parent_handle_relative_atomic_create": True,
        "output_root_deny_delete_lease": True,
        "training_directory_deny_delete_lease": True,
        "leases_held_through_decision_publication": True,
    }
    for checkpoint in (
        "immediately_after_output_creation",
        "before_training",
        "after_training",
        "before_decision_publication",
        "after_decision_publication",
        "before_run_return",
    ):
        assert checkpoint in anchor_checkpoints
        assert checkpoint in attempt_checkpoints
    assert "immediately_before_attempt_guard_close" in attempt_checkpoints
    assert len(observed_attempt_guards) == 1
    assert observed_attempt_guards[0].closed is True
    assert observed_attempt_guards[0].file_lease.closed is True
    assert {anchor.path.name for anchor in observed_output_anchors} == {
        output.name,
        "training-multiview-fixed2-exact8",
    }
    assert all(anchor.closed for anchor in observed_output_anchors)

    verified = exact8.verify_exact8_decision(
        full_records=tmp_path / "unused-full",
        original_dataset_root=tmp_path / "unused-dataset",
        full_crop_pilot_root=tmp_path / "unused-pilot",
        source_contract_path=tmp_path / "unused-source",
        candidate_pilot_evidence_path=tmp_path / "unused-a8",
        failure_evidence_path=tmp_path / "unused-failure",
        failure_attempt_registry=tmp_path / "unused-registry",
        overlay_contract_path=tmp_path / "unused-overlay",
        output_root=output,
        attempt_lock=attempt,
        torch=object(),
    )
    assert verified == sealed


def test_verify_decision_rejects_tampered_integrity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inspection = _inspection(tmp_path)
    output = tmp_path / "tampered"
    attempt = _formal_attempt_path(tmp_path, inspection, monkeypatch)
    summary = _summary(inspection)

    def fake_train(**kwargs: Any) -> None:
        training = Path(kwargs["output_dir"])
        training.mkdir(parents=True, exist_ok=True)
        _write_json(training / "training_summary.json", summary)
        _write_json(training / "labels.json", _labels(summary))
        (training / "best.pt").write_bytes(b"best")
        (training / "last.pt").write_bytes(b"last")

    monkeypatch.setattr(exact8, "inspect_exact8_subject", lambda **_: inspection)
    monkeypatch.setattr(exact8, "train_unified_reader", fake_train)
    monkeypatch.setattr(exact8, "_checkpoint_artifact", lambda *_a, **_k: {})
    monkeypatch.setattr(
        exact8,
        "_load_checkpoint",
        lambda *_a, **_k: {"state_dict": {"frozen.weight": b"same"}},
    )
    exact8.run_exact8(
        full_records=tmp_path / "unused-full",
        original_dataset_root=tmp_path / "unused-dataset",
        full_crop_pilot_root=tmp_path / "unused-pilot",
        source_contract_path=tmp_path / "unused-source",
        candidate_pilot_evidence_path=tmp_path / "unused-a8",
        failure_evidence_path=tmp_path / "unused-failure",
        failure_attempt_registry=tmp_path / "unused-registry",
        overlay_contract_path=tmp_path / "unused-overlay",
        output_root=output,
        attempt_lock=attempt,
        torch=object(),
    )
    decision_path = output / "recipient_multiview_exact8_decision.json"
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    decision["decision"] = "forged_pass"
    _write_json(decision_path, decision)

    with pytest.raises(ValueError, match="integrity"):
        exact8.verify_exact8_decision(
            full_records=tmp_path / "unused-full",
            original_dataset_root=tmp_path / "unused-dataset",
            full_crop_pilot_root=tmp_path / "unused-pilot",
            source_contract_path=tmp_path / "unused-source",
            candidate_pilot_evidence_path=tmp_path / "unused-a8",
            failure_evidence_path=tmp_path / "unused-failure",
            failure_attempt_registry=tmp_path / "unused-registry",
            overlay_contract_path=tmp_path / "unused-overlay",
            output_root=output,
            attempt_lock=attempt,
            torch=object(),
        )


@pytest.mark.parametrize(
    "child_name",
    ["exact8-output", "training-multiview-fixed2-exact8"],
)
def test_output_and_training_parent_replacement_after_check_never_writes_replacement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, child_name: str
) -> None:
    parent = tmp_path / "output-parent"
    parent.mkdir()
    owned_parent = tmp_path / "output-parent-owned"
    replacement = tmp_path / "replacement-parent"
    output = parent / child_name
    monkeypatch.setattr(exact8, "_require_formal_windows_output_anchor", lambda: None)
    anchor = exact8._open_guarded_output_parent(output)
    triggered = False
    replacement_blocked = False

    def replace_after_check(
        checkpoint: str, *, parent: Path, output_root: Path
    ) -> None:
        nonlocal triggered, replacement_blocked
        del output_root
        if triggered or checkpoint != "post_check_pre_atomic_create":
            return
        triggered = True
        try:
            parent.rename(owned_parent)
        except OSError as error:
            if os.name != "nt":
                raise
            replacement_blocked = True
            raise RuntimeError("Windows parent lease blocked replacement") from error
        replacement.mkdir()
        (replacement / "foreign.txt").write_text("must-survive", encoding="utf-8")
        replacement.rename(parent)

    monkeypatch.setattr(exact8, "_exact8_output_anchor_hook", replace_after_check)
    try:
        with pytest.raises(
            (ValueError, RuntimeError),
            match="output identity changed|parent lease blocked replacement",
        ):
            anchor.create()
    finally:
        anchor.close()
    assert triggered is True
    if replacement_blocked:
        assert os.name == "nt"
        assert parent.is_dir()
        assert list(parent.iterdir()) == []
        assert not owned_parent.exists()
        assert not replacement.exists()
        return
    assert list(owned_parent.iterdir()) == []
    assert [path.name for path in parent.iterdir()] == ["foreign.txt"]
    assert (parent / "foreign.txt").read_text(encoding="utf-8") == "must-survive"
    assert not os.path.lexists(output)


def test_leased_output_root_replacement_before_training_mkdir_never_writes_foreign(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    parent = tmp_path / "run-parent"
    parent.mkdir()
    output = parent / "exact8-output"
    owned_output = parent / "exact8-output-owned"
    monkeypatch.setattr(exact8, "_require_formal_windows_output_anchor", lambda: None)
    output_anchor = exact8._open_guarded_output_parent(output)
    output_anchor.create()
    training = output / "training-multiview-fixed2-exact8"
    training_anchor = exact8._open_guarded_child(output_anchor, training)
    triggered = False
    replacement_blocked = False

    def replace_output_after_check(
        checkpoint: str, *, parent: Path, output_root: Path
    ) -> None:
        nonlocal triggered, replacement_blocked
        del output_root
        if triggered or checkpoint != "post_check_pre_atomic_create":
            return
        triggered = True
        try:
            parent.rename(owned_output)
        except OSError as error:
            if os.name != "nt":
                raise
            replacement_blocked = True
            raise RuntimeError("Windows output lease blocked replacement") from error
        parent.mkdir()
        (parent / "foreign.txt").write_text("must-survive", encoding="utf-8")

    monkeypatch.setattr(exact8, "_exact8_output_anchor_hook", replace_output_after_check)
    try:
        with pytest.raises(
            (ValueError, RuntimeError),
            match="output identity changed|output lease blocked replacement",
        ):
            training_anchor.create()
    finally:
        training_anchor.close()
        output_anchor.close()
    assert triggered is True
    if replacement_blocked:
        assert os.name == "nt"
        assert output.is_dir()
        assert list(output.iterdir()) == []
        assert not owned_output.exists()
        return
    assert list(owned_output.iterdir()) == []
    assert [path.name for path in output.iterdir()] == ["foreign.txt"]
    assert (output / "foreign.txt").read_text(encoding="utf-8") == "must-survive"
    assert not os.path.lexists(training)


@pytest.mark.parametrize("nested_training", [False, True], ids=["output", "training"])
def test_output_and_training_reject_replacement_after_atomic_create(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    nested_training: bool,
) -> None:
    parent = tmp_path / "run-parent"
    parent.mkdir()
    output = parent / "exact8-output"
    monkeypatch.setattr(exact8, "_require_formal_windows_output_anchor", lambda: None)
    output_anchor = exact8._open_guarded_output_parent(output)
    child_anchor = None
    if nested_training:
        output_anchor.create()
        child_anchor = exact8._open_guarded_child(
            output_anchor,
            output / "training-multiview-fixed2-exact8",
        )
        anchor = child_anchor
    else:
        anchor = output_anchor
    target = anchor.path
    owned = target.with_name(f"{target.name}-owned")
    replacement_blocked = False
    triggered = False

    def replace_after_atomic_create(
        checkpoint: str, *, parent: Path, output_root: Path
    ) -> None:
        nonlocal replacement_blocked, triggered
        del parent
        if (
            triggered
            or checkpoint != "post_atomic_create_pre_validation"
            or output_root != target
        ):
            return
        triggered = True
        try:
            target.rename(owned)
        except OSError as error:
            if os.name != "nt":
                raise
            replacement_blocked = True
            raise RuntimeError("atomic child lease blocked replacement") from error
        target.mkdir()
        (target / "foreign.txt").write_text("must-survive", encoding="utf-8")

    monkeypatch.setattr(
        exact8,
        "_exact8_output_anchor_hook",
        replace_after_atomic_create,
    )
    try:
        with pytest.raises(
            (ValueError, RuntimeError),
            match="output entry identity changed|atomic child lease blocked replacement",
        ):
            anchor.create()
    finally:
        if child_anchor is not None:
            child_anchor.close()
        output_anchor.close()
    assert triggered is True
    if replacement_blocked:
        assert os.name == "nt"
        assert target.is_dir()
        assert list(target.iterdir()) == []
        assert not owned.exists()
        return
    assert owned.is_dir()
    assert list(owned.iterdir()) == []
    assert (target / "foreign.txt").read_text(encoding="utf-8") == "must-survive"


@pytest.mark.parametrize(
    "name",
    ["exact8-output", "training-multiview-fixed2-exact8"],
)
def test_windows_atomic_child_create_rejects_post_create_entry_replacement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    name: str,
) -> None:
    parent_identity = (11, 12, 0x10)
    created_identity = (11, 13, 0x10)
    replacement_identity = (11, 14, 0x10)
    parent = overlay_module._DirectoryLease(
        path=tmp_path,
        identity=parent_identity,
        windows_handle=100,
        windows_identity=parent_identity,
    )
    dispositions: list[int] = []
    closed: list[int] = []
    replacement_installed = False
    foreign = tmp_path / "foreign"
    foreign.mkdir()
    (foreign / "foreign.txt").write_text("must-survive", encoding="utf-8")

    def fake_nt_open(
        parent_handle: int,
        *,
        name: str,
        create_disposition: int,
        desired_access: int,
        share_access: int,
    ) -> int:
        del name, desired_access, share_access
        assert parent_handle == 100
        dispositions.append(create_disposition)
        return 101 if len(dispositions) == 1 else 102

    def fake_handle_identity(handle: int) -> tuple[int, int, int]:
        if handle == 100:
            return parent_identity
        if handle == 101:
            return created_identity
        assert handle == 102
        assert replacement_installed is True
        return replacement_identity

    def replace_before_parent_relative_reopen(
        checkpoint: str,
        *,
        parent: object,
        name: str,
        handle: int | None,
    ) -> None:
        nonlocal replacement_installed
        del parent, name
        assert checkpoint == "after_windows_atomic_create_before_parent_relative_reopen"
        assert handle == 101
        replacement_installed = True

    monkeypatch.setattr(
        overlay_module,
        "_windows_nt_directory_handle",
        fake_nt_open,
    )
    monkeypatch.setattr(
        overlay_module,
        "_windows_directory_handle_identity",
        fake_handle_identity,
    )
    monkeypatch.setattr(
        overlay_module,
        "_windows_close_handle",
        lambda handle: closed.append(handle),
    )
    monkeypatch.setattr(
        overlay_module,
        "_stage_directory_creation_hook",
        replace_before_parent_relative_reopen,
    )

    with pytest.raises(ValueError, match="not the child entry bound"):
        overlay_module.create_anchored_stage_directory(parent, name=name)
    assert replacement_installed is True
    assert dispositions == [
        overlay_module._WINDOWS_FILE_CREATE,
        overlay_module._WINDOWS_FILE_OPEN,
    ]
    assert closed == [102, 101]
    assert (foreign / "foreign.txt").read_text(encoding="utf-8") == "must-survive"


@pytest.mark.parametrize(
    "name",
    ["exact8-output", "training-multiview-fixed2-exact8"],
)
def test_formal_windows_output_and_training_use_atomic_child_primitive(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    name: str,
) -> None:
    class _FakeParentLease:
        windows_handle = 301

    class _FakeOutputLease:
        identity = (31, 32, 0x10)

    parent_lease = _FakeParentLease()
    output_lease = _FakeOutputLease()
    calls: list[tuple[object, str]] = []
    checkpoints: list[str] = []
    anchor = exact8._GuardedOutputDirectory(
        path=tmp_path / name,
        parent=tmp_path,
        parent_identity=(31, 31, 0x10),
        parent_lease=parent_lease,
    )

    monkeypatch.setattr(exact8.os, "name", "nt")
    monkeypatch.setattr(exact8.os.path, "lexists", lambda _path: False)
    monkeypatch.setattr(
        exact8._GuardedOutputDirectory,
        "require",
        lambda _self, checkpoint: checkpoints.append(checkpoint),
    )
    monkeypatch.setattr(
        overlay_module,
        "create_anchored_stage_directory",
        lambda parent, *, name: calls.append((parent, name)) or output_lease,
    )
    monkeypatch.setattr(
        overlay_module,
        "_create_stage_lease",
        lambda *_args, **_kwargs: pytest.fail(
            "formal Windows output used the non-atomic analysis helper"
        ),
    )

    assert anchor.create() == tmp_path / name
    assert anchor.output_lease is output_lease
    assert calls == [(parent_lease, name)]
    assert checkpoints == [
        "before_output_freshness_check",
        "immediately_before_output_creation",
        "immediately_after_output_creation",
    ]


def test_windows_directory_open_lease_rejects_swap_open_restore_handle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    expected_identity = (21, 22, 0x10)
    replacement_identity = (21, 23, 0x10)
    checkpoints: list[str] = []
    path_identity_reads: list[Path] = []
    closed: list[int] = []

    monkeypatch.setattr(overlay_module.os, "name", "nt")
    monkeypatch.setattr(
        overlay_module,
        "_windows_open_path_directory_handle",
        lambda *_args, **_kwargs: 201,
    )
    monkeypatch.setattr(
        overlay_module,
        "_windows_directory_handle_identity",
        lambda handle: replacement_identity if handle == 201 else expected_identity,
    )
    monkeypatch.setattr(
        overlay_module,
        "_windows_close_handle",
        lambda handle: closed.append(handle),
    )
    monkeypatch.setattr(
        overlay_module,
        "_directory_identity",
        lambda path: path_identity_reads.append(path) or expected_identity,
    )
    monkeypatch.setattr(
        overlay_module,
        "_directory_lease_open_hook",
        lambda checkpoint, **_kwargs: checkpoints.append(checkpoint),
    )

    with pytest.raises(ValueError, match="handle identity does not match expected"):
        overlay_module._open_directory_lease(
            tmp_path,
            expected=expected_identity,
        )
    assert checkpoints == [
        "before_windows_open",
        "after_windows_open_before_identity",
    ]
    assert path_identity_reads == []
    assert closed == [201]


@pytest.mark.skipif(os.name == "nt", reason="non-Windows fail-closed boundary")
def test_formal_output_anchor_fails_closed_without_windows_deny_delete_handles(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="requires Windows deny-delete"):
        exact8._open_guarded_output_parent(tmp_path / "unsupported-output")


def test_python_run_consumes_marker_before_output_and_second_run_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inspection = _inspection(tmp_path)
    attempt = _formal_attempt_path(tmp_path, inspection, monkeypatch)
    first_output = tmp_path / "crash-after-marker"
    monkeypatch.setattr(exact8, "inspect_exact8_subject", lambda **_: inspection)

    def crash_after_marker(
        checkpoint: str, *, parent: Path, output_root: Path
    ) -> None:
        del parent, output_root
        if checkpoint == "post_check_pre_atomic_create":
            raise RuntimeError("injected crash after marker consumption")

    monkeypatch.setattr(exact8, "_exact8_output_anchor_hook", crash_after_marker)
    kwargs = {
        "full_records": tmp_path / "unused-full",
        "original_dataset_root": tmp_path / "unused-dataset",
        "full_crop_pilot_root": tmp_path / "unused-pilot",
        "source_contract_path": tmp_path / "unused-source",
        "candidate_pilot_evidence_path": tmp_path / "unused-a8",
        "failure_evidence_path": tmp_path / "unused-failure",
        "failure_attempt_registry": tmp_path / "unused-registry",
        "overlay_contract_path": tmp_path / "unused-overlay",
        "attempt_lock": attempt,
        "torch": object(),
    }
    with pytest.raises(RuntimeError, match="after marker consumption"):
        exact8.run_exact8(output_root=first_output, **kwargs)
    assert attempt.is_file()
    assert not os.path.lexists(first_output)
    marker = json.loads(attempt.read_text(encoding="utf-8"))
    assert marker["attempt_id"] == inspection["attempt_id"]
    assert marker["output_root"] == str(first_output.resolve())

    monkeypatch.setattr(
        exact8,
        "_exact8_output_anchor_hook",
        lambda _checkpoint, *, parent, output_root: None,
    )
    second_output = tmp_path / "second-run"
    with pytest.raises(ValueError, match="already consumed"):
        exact8.run_exact8(output_root=second_output, **kwargs)
    assert not os.path.lexists(second_output)


def test_python_run_failure_before_consumption_leaves_no_marker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inspection = _inspection(tmp_path)
    attempt = _formal_attempt_path(tmp_path, inspection, monkeypatch)
    monkeypatch.setattr(
        exact8,
        "inspect_exact8_subject",
        lambda **_: (_ for _ in ()).throw(ValueError("injected inspection failure")),
    )
    with pytest.raises(ValueError, match="inspection failure"):
        exact8.run_exact8(
            full_records=tmp_path / "unused-full",
            original_dataset_root=tmp_path / "unused-dataset",
            full_crop_pilot_root=tmp_path / "unused-pilot",
            source_contract_path=tmp_path / "unused-source",
            candidate_pilot_evidence_path=tmp_path / "unused-a8",
            failure_evidence_path=tmp_path / "unused-failure",
            failure_attempt_registry=tmp_path / "unused-registry",
            overlay_contract_path=tmp_path / "unused-overlay",
            output_root=tmp_path / "unused-output",
            attempt_lock=attempt,
            torch=object(),
        )
    assert not os.path.lexists(attempt)


def test_python_run_marker_acl_failure_consumes_marker_before_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inspection = _inspection(tmp_path)
    attempt = _formal_attempt_path(tmp_path, inspection, monkeypatch)
    output = tmp_path / "must-not-be-created"
    monkeypatch.setattr(exact8, "inspect_exact8_subject", lambda **_: inspection)

    def create_then_reject(path: Path, payload: bytes) -> None:
        path.write_bytes(payload)
        raise ValueError("exact8 attempt marker DACL inherited deny missing")

    monkeypatch.setattr(
        exact8,
        "_create_attempt_file_lease",
        create_then_reject,
    )
    with pytest.raises(
        ValueError,
        match="exact8 attempt marker DACL inherited deny missing",
    ):
        exact8.run_exact8(
            full_records=tmp_path / "unused-full",
            original_dataset_root=tmp_path / "unused-dataset",
            full_crop_pilot_root=tmp_path / "unused-pilot",
            source_contract_path=tmp_path / "unused-source",
            candidate_pilot_evidence_path=tmp_path / "unused-a8",
            failure_evidence_path=tmp_path / "unused-failure",
            failure_attempt_registry=tmp_path / "unused-registry",
            overlay_contract_path=tmp_path / "unused-overlay",
            output_root=output,
            attempt_lock=attempt,
            torch=object(),
        )
    assert attempt.is_file()
    assert not os.path.lexists(output)


@pytest.mark.parametrize(
    "claim",
    [
        "test_evaluation_authorized",
        "onnx_export_authorized",
        "production_ready",
        "warmstart_authorized",
        "same_route_retry_authorized",
        "continuation_authorized",
        "retry_authorized",
    ],
)
@pytest.mark.parametrize("nested", [False, True])
def test_output_json_rejects_complete_unsafe_true_claim_family(
    tmp_path: Path, claim: str, nested: bool
) -> None:
    output = tmp_path / "unsafe-output"
    output.mkdir()
    payload: dict[str, object] = {"safe": False}
    if nested:
        payload["nested"] = [{"claim": {claim: True}}]
    else:
        payload[claim] = True
    _write_json(output / "evidence.json", payload)
    with pytest.raises(ValueError, match="unsafe true claim"):
        exact8._assert_no_delivery_artifacts(output)


def test_windows_runner_is_one_shot_streaming_and_never_opens_delivery_route() -> None:
    script = (
        Path(__file__).parents[1]
        / "scripts"
        / "receipt-ocr-recipient-multiview-exact8-4090.ps1"
    ).read_text(encoding="utf-8")
    lower = script.lower()

    assert "verify-decision" in script
    assert "$runLines" not in script
    assert "& $pythonExe @runArguments" in script
    assert "[switch]$CheckOnly" in script
    assert "Write-CreateNewUtf8" not in script
    assert "[IO.FileMode]::CreateNew" not in script
    assert "Python run owns the atomic CreateNew" in script
    assert "Protect-AuditRoot $receiptRoot" in script
    assert "Protect-AuditRoot $auditRoot" in script
    assert script.index("Protect-AuditRoot $receiptRoot") < script.index(
        "Protect-AuditRoot $auditRoot"
    )
    assert script.index("Protect-AuditRoot $auditRoot") < script.index(
        "& $pythonExe @runArguments"
    )
    assert script.index("if ($CheckOnly)") < script.index(
        "New-Item -ItemType Directory -Path $auditRoot"
    )
    assert script.index("Test-Path -LiteralPath $attemptPath") < script.index(
        "if ($CheckOnly)"
    )
    module = Path(exact8.__file__).read_text(encoding="utf-8")
    assert "os.O_WRONLY | os.O_CREAT | os.O_EXCL" in module
    assert 'os.environ.get("PROGRAMDATA")' not in module
    assert "SHGetKnownFolderPath" in module
    assert "GetSecurityInfo" in module
    assert "AccessCheck" in module
    run_source = inspect.getsource(exact8.run_exact8)
    assert run_source.index("attempt_guard = _consume_attempt(") < run_source.index(
        "output = output_anchor.create()"
    )
    assert "_open_directory_lease" in module
    assert "create_anchored_stage_directory" in module
    assert "_open_guarded_child(output_anchor, training)" in module
    assert "before_training" in module and "after_decision_publication" in module
    assert "ocr_unified\", \"export" not in lower
    assert "ocr_unified\", \"evaluate" not in lower
    assert "materialize_fixed2_overlay" not in script
    assert "--output-root\", $OutputRoot" in script


def test_cli_exposes_independent_verify_without_treating_verified_fail_as_error() -> None:
    parser = exact8.build_parser()
    args = parser.parse_args(
        [
            "verify-decision",
            "--full-records",
            "full",
            "--dataset-root",
            "data",
            "--full-crop-pilot-root",
            "pilot",
            "--source-contract",
            "source",
            "--candidate-pilot-evidence",
            "a8",
            "--failure-evidence",
            "failure",
            "--failure-attempt-registry",
            "registry",
            "--overlay-contract",
            "overlay",
            "--output-root",
            "output",
            "--attempt-lock",
            "attempt",
        ]
    )
    assert args.command == "verify-decision"
    assert args.decision is None
