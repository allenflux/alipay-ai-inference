"""Contracts for the content-bound recipient-v14 fresh60 failure."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace

import pytest

import transfer_receipt_ai.recipient_v14_failure_attestor as failure_attestor
from transfer_receipt_ai.ocr_unified import (
    INIT_CHECKPOINT_MODE_RECIPIENT_VISUAL_CONTEXT_REINIT,
    STATUS_TEXT_RUNTIME_POLICY,
    UnifiedReaderConfig,
    _recipient_train_augmentation_policy,
)
from transfer_receipt_ai.recipient_blind_manifest import build_blind_manifest
from transfer_receipt_ai.recipient_full_crop_candidate_source import (
    CANDIDATE_PILOT_DECISION,
    CANDIDATE_PILOT_KIND,
    EXPECTED_RECIPIENT_VAL_RECORDS,
    SOURCE_KIND,
)
from transfer_receipt_ai.recipient_full_crop_pilot import (
    AMOUNT_FLOOR,
    PAYMENT_FLOOR,
    STATUS_TEXT_FLOOR,
    TIME_FLOOR,
)
from transfer_receipt_ai.recipient_v14_failure_attestor import (
    ATTEMPT_DOMAIN,
    ATTEMPT_KIND,
    ATTEMPT_REGISTRY_NAME,
    ATTEMPT_THREAT_MODEL,
    AUTHORIZATION,
    DECISION,
    EXPECTED_BEST_EPOCH,
    EXPECTED_BEST_MATCHES,
    EXPECTED_LAST_MATCHES,
    EXPECTED_STRICT_PASS_MATCHES,
    KIND,
    attest_fresh60_failure,
    verify_fresh60_failure,
)


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def _canonical_sha(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _sealed(payload: dict[str, object]) -> dict[str, object]:
    return {**payload, "integrity_sha256": _canonical_sha(payload)}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _binding(path: Path) -> dict[str, object]:
    return {
        "path": str(path.resolve()),
        "sha256": _sha(path),
        "size_bytes": path.stat().st_size,
    }


def _build_full_and_blind(candidate_root: Path) -> tuple[Path, Path, Path]:
    full = candidate_root.parent / "full.jsonl"
    rows = [
        {
            "id": "train-one",
            "split": "train",
            "slots": {"recipient_field": {"text": "甲"}},
        },
        *(
            {
                "id": f"val-{index}",
                "split": "val",
                "slots": {"recipient_field": {"text": "乙"}},
            }
            for index in range(EXPECTED_RECIPIENT_VAL_RECORDS)
        ),
        {
            "id": "test-secret",
            "split": "test",
            "slots": {"recipient_field": {"text": "绝密"}},
        },
    ]
    full.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    blind_root = candidate_root / "blind-train-val"
    blind_root.mkdir(parents=True)
    blind = blind_root / "unified_fields.train-val.jsonl"
    contract = blind_root / "blind.contract.json"
    build_blind_manifest(source=full, output=blind, contract=contract)
    return full, blind, contract


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


def _recipe(
    *,
    source_subject: str,
    candidate_subject: str,
    source_checkpoint_sha: str,
    full_sha: str,
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "kind": "receipt_recipient_v14_full_crop_training_recipe_v1",
        "analysis_only": True,
        "production_route_authorized": False,
        "test_opened": False,
        "stage": "candidate-60e",
        "source_subject_id": source_subject,
        "candidate_pilot_subject_id": candidate_subject,
        "source_checkpoint_sha256": source_checkpoint_sha,
        "full_manifest_sha256": full_sha,
        "training_args": {
            "device": "cuda:0",
            "epochs": 60,
            "batch_size": 10,
            "learning_rate": 0.0003,
            "validation_every": 2,
            "seed": 42,
            "num_workers": 4,
            "prefetch_factor": 2,
            "persistent_workers": True,
            "cuda_tf32": True,
            "cudnn_benchmark": True,
        },
    }


def _summary(*, source_checkpoint: Path) -> dict[str, object]:
    source_config, target_config = _configs()
    initialization = {
        "mode": "parameter_only_recipient_visual_context_reinit",
        "init_checkpoint_mode": INIT_CHECKPOINT_MODE_RECIPIENT_VISUAL_CONTEXT_REINIT,
        "source_kind": "receipt_unified_field_reader_v13",
        "source_config": source_config,
        "checkpoint_path": str(source_checkpoint.resolve()),
        "checkpoint_sha256": _sha(source_checkpoint),
        "financial_label_policy": {
            "recipient_character_map": {
                "mode": "fresh_train_only_reinitialized_recipient_v1"
            }
        },
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
        "validation_every": 2,
        "cuda_tf32_requested": True,
        "cudnn_benchmark_requested": True,
    }
    split_policy = {"mode": "standard_train_only", "splits": ["train"]}
    checkpoint_policy = {
        "mode": "recipient_priority",
        "protected_minimum_candidate_exact": {
            "amount": AMOUNT_FLOOR,
            "time": TIME_FLOOR,
            "payment_method_field": PAYMENT_FLOOR,
        },
    }
    augmentation = _recipient_train_augmentation_policy(mode="robust_v2", seed=42)
    records: list[dict[str, object]] = []
    for epoch in range(1, 61):
        validated = epoch == 1 or epoch == 60 or epoch % 2 == 0
        if not validated:
            records.append(
                {
                    "epoch": epoch,
                    "train_loss": 1.0 / epoch,
                    "validation_performed": False,
                    "val_loss": None,
                    "val_candidate_text_by_field": None,
                    "val_ctc_by_field": None,
                    "val_status_non_success_to_success": None,
                    "checkpoint_selection_eligible": False,
                    "checkpoint_selection_protection_failures": [
                        "full_validation_not_scheduled"
                    ],
                    "checkpoint_selection_score": None,
                    "checkpoint_protection": None,
                }
            )
            continue
        matches = 5800
        if epoch == EXPECTED_BEST_EPOCH:
            matches = EXPECTED_BEST_MATCHES
        elif epoch == 60:
            matches = EXPECTED_LAST_MATCHES
        recipient_exact = matches / EXPECTED_RECIPIENT_VAL_RECORDS
        score = [recipient_exact, 0.80, 0.99, 0.94]
        records.append(
            {
                "epoch": epoch,
                "train_loss": 1.0 / epoch,
                "validation_performed": True,
                "val_loss": 0.5,
                "val_candidate_text_by_field": {
                    "amount": {"exact_match": 0.80},
                    "time": {"exact_match": 0.99},
                    "payment_method_field": {"exact_match": 0.94},
                    "recipient_field": {
                        "records": EXPECTED_RECIPIENT_VAL_RECORDS,
                        "exact_matches": matches,
                        "exact_match": recipient_exact,
                    },
                },
                "val_ctc_by_field": {
                    "transfer_status": {"exact_match": 0.91}
                },
                "val_status_non_success_to_success": 0,
                "checkpoint_selection_eligible": True,
                "checkpoint_selection_protection_failures": [],
                "checkpoint_selection_score": score,
                "checkpoint_protection": {"failures": []},
            }
        )
    common_recipient = {
        "recipient_oov_by_split": {
            "train": {"records": 1},
            "val": {"records": EXPECTED_RECIPIENT_VAL_RECORDS},
            "test": {"records": 0},
        },
        "recipient_sampling_policy": {"mode": "uniform"},
        "recipient_confidence_policy": {"mode": "bounded_train_only"},
        "recipient_tail_loss_policy": {"mode": "rare_long_v1"},
        "recipient_train_augmentation_policy": augmentation,
        "recipient_train_split_policy": split_policy,
        "recipient_target": "anchored_recipient_value_with_dedicated_high_resolution_value_view",
    }
    return {
        "schema_version": 1,
        "kind": "receipt_unified_field_reader_v13",
        "config": target_config,
        "initialization": initialization,
        "fine_tune_policy": fine_tune,
        "training_runtime": runtime,
        "checkpoint_selection_policy": checkpoint_policy,
        "status_text_runtime_policy": STATUS_TEXT_RUNTIME_POLICY,
        "field_counts": {
            field: {
                "train": 1,
                "val": (
                    EXPECTED_RECIPIENT_VAL_RECORDS
                    if field == "recipient_field"
                    else 3
                ),
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
        **common_recipient,
        "best_checkpoint_epoch": EXPECTED_BEST_EPOCH,
        "best_checkpoint_score": records[EXPECTED_BEST_EPOCH - 1][
            "checkpoint_selection_score"
        ],
        "records": records,
    }


def _labels(summary: dict[str, object]) -> dict[str, object]:
    characters = ["甲", "乙"]
    charset = hashlib.sha256("".join(characters).encode("utf-8")).hexdigest()
    keys = (
        "checkpoint_selection_policy",
        "initialization",
        "training_runtime",
        "fine_tune_policy",
        "recipient_oov_by_split",
        "recipient_sampling_policy",
        "recipient_confidence_policy",
        "recipient_tail_loss_policy",
        "recipient_train_augmentation_policy",
        "recipient_train_split_policy",
        "recipient_target",
        "status_text_runtime_policy",
    )
    return {
        "schema_version": 1,
        **{key: summary[key] for key in keys},
        "recipient_characters": characters,
        "recipient_blank_index": 0,
        "recipient_charset_sha256": charset,
    }


def _checkpoint(
    summary: dict[str, object], labels: dict[str, object], *, epoch: int
) -> dict[str, object]:
    record = summary["records"][epoch - 1]
    shared = (
        "config",
        "initialization",
        "fine_tune_policy",
        "checkpoint_selection_policy",
        "recipient_train_split_policy",
        "field_counts",
        "status_text_runtime_policy",
        "training_runtime",
    )
    label_keys = (
        "recipient_oov_by_split",
        "recipient_sampling_policy",
        "recipient_confidence_policy",
        "recipient_tail_loss_policy",
        "recipient_train_augmentation_policy",
        "recipient_target",
        "recipient_characters",
        "recipient_blank_index",
        "recipient_charset_sha256",
    )
    return {
        "schema_version": 1,
        "kind": "receipt_unified_field_reader_v13",
        "state_dict": {"recipient_weight": f"epoch-{epoch}"},
        "epoch": epoch,
        "metrics": record,
        **{key: summary[key] for key in shared},
        **{key: labels[key] for key in label_keys},
    }


def _make_failure_fixture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> SimpleNamespace:
    tmp_path.mkdir(parents=True, exist_ok=True)
    candidate_root = tmp_path / "fresh60-failed"
    candidate_root.mkdir()
    full, blind, blind_contract = _build_full_and_blind(candidate_root)

    source_checkpoint = tmp_path / "source-best.pt"
    source_checkpoint.write_bytes(b"attested-legacy-source-checkpoint")
    candidate_pilot_checkpoint = tmp_path / "candidate-pilot-best.pt"
    candidate_pilot_checkpoint.write_bytes(b"attested-residual-a8-checkpoint")
    sanitizer_status_checkpoint = tmp_path / "sanitizer-status.pt"
    sanitizer_status_checkpoint.write_bytes(b"sanitizer-status-checkpoint")
    sanitizer_train_checkpoint = tmp_path / "sanitizer-train.pt"
    sanitizer_train_checkpoint.write_bytes(b"sanitizer-train-checkpoint")
    sanitizer_root_checkpoint = tmp_path / "sanitizer-root.pt"
    sanitizer_root_checkpoint.write_bytes(b"sanitizer-root-checkpoint")
    sanitizer_seed_checkpoint = tmp_path / "sanitized-seed.pt"
    sanitizer_seed_checkpoint.write_bytes(b"sanitized-seed-checkpoint")
    status_descriptor = {
        **_binding(sanitizer_status_checkpoint),
        "kind": "receipt_unified_field_reader_v13",
        "epoch": 9,
    }
    train_descriptor = {
        **_binding(sanitizer_train_checkpoint),
        "kind": "receipt_unified_field_reader_v12",
        "epoch": 8,
    }
    root_descriptor = {
        **_binding(sanitizer_root_checkpoint),
        "kind": "receipt_unified_field_reader_v12",
        "epoch": 4,
    }
    sanitizer_attestation = {
        "status_checkpoint": status_descriptor,
        "train_only_recipient_checkpoint": train_descriptor,
        "train_only_recipient_lineage": {
            "entries": [
                {"checkpoint": train_descriptor},
                {"checkpoint": root_descriptor},
            ]
        },
    }
    sanitizer_seed_payload = {
        "schema_version": 1,
        "kind": "receipt_unified_field_reader_v13",
        "full_crop_seed_sanitizer_attestation": sanitizer_attestation,
    }
    sanitizer_status_payload = {
        "schema_version": 1,
        "kind": "receipt_unified_field_reader_v13",
        "initialization": {"mode": "random"},
    }
    sanitizer_train_payload = {
        "schema_version": 1,
        "kind": "receipt_unified_field_reader_v12",
        "initialization": {
            "mode": "parameter_only",
            "checkpoint_path": str(sanitizer_root_checkpoint.resolve()),
            "checkpoint_sha256": _sha(sanitizer_root_checkpoint),
        },
    }
    sanitizer_root_payload = {
        "schema_version": 1,
        "kind": "receipt_unified_field_reader_v12",
        "initialization": {"mode": "random"},
    }
    monkeypatch.setattr(
        failure_attestor._seed_sanitizer_verifier,
        "validate_recipient_full_crop_seed_attestation",
        lambda payload: sanitizer_attestation,
    )
    candidate_pilot_summary = tmp_path / "candidate-pilot-summary.json"
    _write_json(
        candidate_pilot_summary,
        {"initialization": {"checkpoint_path": str(source_checkpoint.resolve())}},
    )
    source_training_summary = tmp_path / "source-training-summary.json"
    _write_json(
        source_training_summary,
        {
            "initialization": {
                "checkpoint_path": str(sanitizer_seed_checkpoint.resolve())
            }
        },
    )
    pilot_root = tmp_path / "source-pilot"
    pilot_root.mkdir()
    source_subject = failure_attestor.EXPECTED_SOURCE_SUBJECT_ID
    candidate_subject = failure_attestor.EXPECTED_CANDIDATE_PILOT_SUBJECT_ID
    source_contract = tmp_path / "source.contract.json"
    _write_json(
        source_contract,
        _sealed(
            {
                "schema_version": 1,
                "kind": SOURCE_KIND,
                "source_subject_id": source_subject,
                "pilot_root": str(pilot_root.resolve()),
                "artifacts": {
                    "best_checkpoint": _binding(source_checkpoint),
                    "seed_checkpoint": _binding(sanitizer_seed_checkpoint),
                    "training_summary": _binding(source_training_summary),
                    "full_manifest": _binding(full),
                    "blind_manifest": _binding(blind),
                    "blind_contract": _binding(blind_contract),
                },
            }
        ),
    )
    candidate_pilot_evidence = tmp_path / "candidate-pilot.json"
    _write_json(
        candidate_pilot_evidence,
        _sealed(
            {
                "schema_version": 1,
                "kind": CANDIDATE_PILOT_KIND,
                "source_subject_id": source_subject,
                "candidate_pilot_subject_id": candidate_subject,
                "artifacts": {
                    "candidate_best_checkpoint": _binding(
                        candidate_pilot_checkpoint
                    ),
                    "source_best_checkpoint": _binding(source_checkpoint),
                    "candidate_training_summary": _binding(
                        candidate_pilot_summary
                    ),
                    "source_contract": _binding(source_contract),
                    "full_manifest": _binding(full),
                    "candidate_blind_manifest": _binding(blind),
                    "candidate_blind_contract": _binding(blind_contract),
                },
            }
        ),
    )
    source_payload = {
        "schema_version": 1,
        "kind": SOURCE_KIND,
        "analysis_only": True,
        "production_route_authorized": False,
        "test_opened": False,
        "onnx_exported": False,
        "source_subject_id": source_subject,
        "artifacts": {
            "best_checkpoint": _binding(source_checkpoint),
            "seed_checkpoint": _binding(sanitizer_seed_checkpoint),
            "training_summary": _binding(source_training_summary),
            "full_manifest": _binding(full),
            "blind_manifest": _binding(blind),
            "blind_contract": _binding(blind_contract),
        },
    }
    pilot_payload = {
        "schema_version": 1,
        "kind": CANDIDATE_PILOT_KIND,
        "analysis_only": True,
        "production_route_authorized": False,
        "test_opened": False,
        "onnx_exported": False,
        "passed": True,
        "decision": CANDIDATE_PILOT_DECISION,
        "source_subject_id": source_subject,
        "candidate_pilot_subject_id": candidate_subject,
        "artifacts": {
            "candidate_best_checkpoint": _binding(candidate_pilot_checkpoint),
            "source_best_checkpoint": _binding(source_checkpoint),
            "candidate_training_summary": _binding(candidate_pilot_summary),
            "source_contract": _binding(source_contract),
            "full_manifest": _binding(full),
            "candidate_blind_manifest": _binding(blind),
            "candidate_blind_contract": _binding(blind_contract),
        },
    }
    monkeypatch.setattr(
        failure_attestor,
        "verify_full_crop_candidate_source",
        lambda **kwargs: source_payload,
    )
    monkeypatch.setattr(
        failure_attestor,
        "verify_residual_candidate_pilot",
        lambda **kwargs: pilot_payload,
    )

    training = candidate_root / "training-v14-candidate"
    training.mkdir()
    summary = _summary(source_checkpoint=source_checkpoint)
    summary_path = training / "training_summary.json"
    _write_json(summary_path, summary)
    labels = _labels(summary)
    labels_path = training / "labels.json"
    _write_json(labels_path, labels)
    recipe_path = candidate_root / "recipient_v14_training_recipe.json"
    _write_json(
        recipe_path,
        _recipe(
            source_subject=source_subject,
            candidate_subject=candidate_subject,
            source_checkpoint_sha=_sha(source_checkpoint),
            full_sha=_sha(full),
        ),
    )

    best_path = training / "best.pt"
    last_path = training / "last.pt"
    best_path.write_bytes(b"failed-fresh60-best-checkpoint")
    last_path.write_bytes(b"failed-fresh60-last-checkpoint")
    checkpoint_payloads = {
        best_path.resolve(): _checkpoint(summary, labels, epoch=EXPECTED_BEST_EPOCH),
        last_path.resolve(): _checkpoint(summary, labels, epoch=60),
        sanitizer_seed_checkpoint.resolve(): sanitizer_seed_payload,
        sanitizer_status_checkpoint.resolve(): sanitizer_status_payload,
        sanitizer_train_checkpoint.resolve(): sanitizer_train_payload,
        sanitizer_root_checkpoint.resolve(): sanitizer_root_payload,
    }
    checkpoint_paths_by_sha = {
        _sha(path): path.resolve() for path in checkpoint_payloads
    }

    def load_checkpoint(path: object, *, torch: object):
        del torch
        if hasattr(path, "read"):
            raw_bytes = path.read()
            return checkpoint_payloads[
                checkpoint_paths_by_sha[hashlib.sha256(raw_bytes).hexdigest()]
            ]
        return checkpoint_payloads[Path(path).resolve()]

    monkeypatch.setattr(failure_attestor, "_load_checkpoint", load_checkpoint)

    registry = (
        tmp_path
        / "ProgramData"
        / "ReceiptAI"
        / ATTEMPT_REGISTRY_NAME
    )
    registry.mkdir(parents=True)
    attempt_subject = f"{ATTEMPT_DOMAIN}|{source_subject}|{candidate_subject}"
    attempt_id = hashlib.sha256(attempt_subject.encode("utf-8")).hexdigest()
    attempt_path = registry / f"{attempt_id}.attempt.json"
    _write_json(
        attempt_path,
        {
            "schema_version": 1,
            "kind": ATTEMPT_KIND,
            "created_at_utc": "2026-08-10T00:00:00+00:00",
            "attempt_id": attempt_id,
            "stage": "candidate-60e",
            "source_subject_id": source_subject,
            "candidate_pilot_subject_id": candidate_subject,
            "output_root": str(candidate_root.resolve()),
            "full_manifest_sha256": _sha(full),
            "threat_model": ATTEMPT_THREAT_MODEL,
        },
    )
    assert attempt_id == failure_attestor.EXPECTED_ATTEMPT_ID
    fixture_pin_paths = {
        "training_summary": summary_path,
        "best_checkpoint": best_path,
        "last_checkpoint": last_path,
        "training_labels": labels_path,
        "training_recipe": recipe_path,
        "blind_manifest": candidate_root
        / "blind-train-val"
        / "unified_fields.train-val.jsonl",
        "blind_contract": candidate_root / "blind-train-val" / "blind.contract.json",
        "training_attempt": attempt_path,
    }
    run_pins = {
        name: {"sha256": _sha(path), "size_bytes": path.stat().st_size}
        for name, path in fixture_pin_paths.items()
    }
    authority_pins = {
        "source_contract": {
            "sha256": _sha(source_contract),
            "size_bytes": source_contract.stat().st_size,
        },
        "candidate_pilot_evidence": {
            "sha256": _sha(candidate_pilot_evidence),
            "size_bytes": candidate_pilot_evidence.stat().st_size,
        },
    }
    monkeypatch.setattr(failure_attestor, "EXPECTED_RUN_ARTIFACT_PINS", run_pins)
    monkeypatch.setattr(
        failure_attestor, "EXPECTED_AUTHORITY_DOCUMENT_PINS", authority_pins
    )
    monkeypatch.setattr(
        failure_attestor,
        "_windows_programdata_attempt_registry",
        lambda: registry.resolve(),
    )
    return SimpleNamespace(
        candidate_root=candidate_root,
        full=full,
        blind=blind,
        blind_contract=blind_contract,
        source_contract=source_contract,
        candidate_pilot_evidence=candidate_pilot_evidence,
        registry=registry,
        attempt_path=attempt_path,
        source_checkpoint=source_checkpoint,
        candidate_pilot_checkpoint=candidate_pilot_checkpoint,
        candidate_pilot_summary=candidate_pilot_summary,
        source_training_summary=source_training_summary,
        sanitizer_seed_checkpoint=sanitizer_seed_checkpoint,
        sanitizer_status_checkpoint=sanitizer_status_checkpoint,
        sanitizer_train_checkpoint=sanitizer_train_checkpoint,
        sanitizer_root_checkpoint=sanitizer_root_checkpoint,
        sanitizer_attestation=sanitizer_attestation,
        summary_path=summary_path,
        labels_path=labels_path,
        recipe_path=recipe_path,
        best_path=best_path,
        last_path=last_path,
        summary=summary,
        labels=labels,
        source_payload=source_payload,
        pilot_payload=pilot_payload,
        checkpoint_payloads=checkpoint_payloads,
        load_checkpoint=load_checkpoint,
        fixture_pin_paths=fixture_pin_paths,
        run_pins=run_pins,
        authority_pins=authority_pins,
    )


def _activate_failure_fixture(
    fixture: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        failure_attestor,
        "verify_full_crop_candidate_source",
        lambda **kwargs: fixture.source_payload,
    )
    monkeypatch.setattr(
        failure_attestor,
        "verify_residual_candidate_pilot",
        lambda **kwargs: fixture.pilot_payload,
    )
    monkeypatch.setattr(failure_attestor, "_load_checkpoint", fixture.load_checkpoint)
    monkeypatch.setattr(
        failure_attestor._seed_sanitizer_verifier,
        "validate_recipient_full_crop_seed_attestation",
        lambda payload: fixture.sanitizer_attestation,
    )
    monkeypatch.setattr(
        failure_attestor, "EXPECTED_RUN_ARTIFACT_PINS", fixture.run_pins
    )
    monkeypatch.setattr(
        failure_attestor,
        "EXPECTED_AUTHORITY_DOCUMENT_PINS",
        fixture.authority_pins,
    )
    monkeypatch.setattr(
        failure_attestor,
        "_windows_programdata_attempt_registry",
        lambda: fixture.registry.resolve(),
    )


@pytest.fixture()
def failure_fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    return _make_failure_fixture(tmp_path, monkeypatch)


def _attest(fixture: SimpleNamespace, output: Path) -> dict[str, object]:
    return attest_fresh60_failure(
        candidate_root=fixture.candidate_root,
        source_contract_path=fixture.source_contract,
        candidate_pilot_evidence_path=fixture.candidate_pilot_evidence,
        full_records=fixture.full,
        attempt_registry=fixture.registry,
        output_evidence=output,
        torch=object(),
    )


def _verify(fixture: SimpleNamespace, evidence: Path) -> dict[str, object]:
    return verify_fresh60_failure(
        evidence_path=evidence,
        source_contract_path=fixture.source_contract,
        candidate_pilot_evidence_path=fixture.candidate_pilot_evidence,
        full_records=fixture.full,
        attempt_registry=fixture.registry,
        torch=object(),
    )


def _refresh_fixture_pin(name: str, path: Path) -> None:
    failure_attestor.EXPECTED_RUN_ARTIFACT_PINS[name] = {
        "sha256": _sha(path),
        "size_bytes": path.stat().st_size,
    }


def _refresh_authority_pin(name: str, path: Path) -> None:
    failure_attestor.EXPECTED_AUTHORITY_DOCUMENT_PINS[name] = {
        "sha256": _sha(path),
        "size_bytes": path.stat().st_size,
    }


def _true_boolean_paths(value: object, prefix: str = "") -> list[str]:
    result: list[str] = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            if child is True:
                result.append(path)
            result.extend(_true_boolean_paths(child, path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            result.extend(_true_boolean_paths(child, f"{prefix}[{index}]"))
    return result


def test_seals_and_reopens_only_the_different_view_exact8_authority(
    failure_fixture: SimpleNamespace, tmp_path: Path
) -> None:
    evidence = tmp_path / "fresh60.failure.json"
    sealed = _attest(failure_fixture, evidence)
    verified = _verify(failure_fixture, evidence)
    assert verified == sealed
    assert sealed["kind"] == KIND
    assert sealed["decision"] == DECISION
    assert sealed["authorization"] == AUTHORIZATION
    assert sealed["analysis_only"] is True
    assert sealed["new_view_pilot_authority"] is True
    assert _true_boolean_paths(sealed) == ["analysis_only", "new_view_pilot_authority"]
    assert sealed["production_route_authorized"] is False
    assert sealed["same_route_retry_authorized"] is False
    assert sealed["same_route_continuation_authorized"] is False
    assert sealed["warmstart_authorized"] is False
    assert sealed["failed_checkpoint_initialization_authorized"] is False
    assert sealed["onnx_export_authorized"] is False
    assert sealed["test_evaluation_authorized"] is False
    assert sealed["authorization_scope"]["epochs"] == 8
    assert sealed["authorization_scope"]["failed_best_checkpoint_use"] == "forbidden"
    assert sealed["authorization_scope"]["failed_last_checkpoint_use"] == "forbidden"
    observed = sealed["observed_failure"]
    assert observed["best_epoch"] == 44
    assert observed["best_recipient_exact_matches"] == EXPECTED_BEST_MATCHES
    assert observed["last_recipient_exact_matches"] == EXPECTED_LAST_MATCHES
    assert observed["strict_pass_exact_matches"] == EXPECTED_STRICT_PASS_MATCHES
    assert observed["strict_pass_gap_matches"] == 192
    assert observed["recipient_val_records"] == EXPECTED_RECIPIENT_VAL_RECORDS
    assert observed["recipient_candidate_coverage"] == 1.0
    assert observed["validated_epochs"] == [1, *range(2, 61, 2)]
    assert sealed["candidate_evidence"] == "absent"
    assert sealed["onnx_artifacts"] == "absent"
    assert sealed["test_evidence"] == "absent"
    assert len(sealed["failure_subject_id"]) == 64
    assert set(sealed["artifacts"]) >= {
        "source_contract",
        "candidate_pilot_evidence",
        "training_recipe",
        "training_summary",
        "best_checkpoint",
        "last_checkpoint",
        "training_labels",
        "training_attempt",
        "full_manifest",
        "blind_manifest",
        "blind_contract",
    }
    with pytest.raises(ValueError, match="Refusing to overwrite"):
        _attest(failure_fixture, evidence)


def test_failure_subject_is_path_stable_across_two_strictly_verified_roots(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first_fixture = _make_failure_fixture(tmp_path / "root-one", monkeypatch)
    first_evidence = tmp_path / "root-one.failure.json"
    first = _attest(first_fixture, first_evidence)
    assert _verify(first_fixture, first_evidence) == first

    second_fixture = _make_failure_fixture(tmp_path / "root-two", monkeypatch)
    second_evidence = tmp_path / "root-two.failure.json"
    second = _attest(second_fixture, second_evidence)
    assert _verify(second_fixture, second_evidence) == second

    assert first["candidate_root"] != second["candidate_root"]
    assert first["artifacts"] != second["artifacts"]
    assert first["failure_subject_id"] == second["failure_subject_id"]

    _activate_failure_fixture(first_fixture, monkeypatch)
    assert _verify(first_fixture, first_evidence) == first
    _activate_failure_fixture(second_fixture, monkeypatch)
    assert _verify(second_fixture, second_evidence) == second


@pytest.mark.parametrize("metric", ["financial_guard", "val_loss"])
def test_nonbest_validated_epoch_metric_changes_the_subject_curve(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, metric: str
) -> None:
    first_fixture = _make_failure_fixture(tmp_path / "curve-one", monkeypatch)
    first_evidence = tmp_path / "curve-one.failure.json"
    first = _attest(first_fixture, first_evidence)

    second_fixture = _make_failure_fixture(tmp_path / "curve-two", monkeypatch)
    summary = json.loads(second_fixture.summary_path.read_text(encoding="utf-8"))
    if metric == "financial_guard":
        summary["records"][1]["val_candidate_text_by_field"]["amount"][
            "exact_match"
        ] = 0.81
    else:
        summary["records"][1]["val_loss"] = 0.49
    _write_json(second_fixture.summary_path, summary)
    second_fixture.run_pins["training_summary"] = {
        "sha256": _sha(second_fixture.summary_path),
        "size_bytes": second_fixture.summary_path.stat().st_size,
    }
    second_evidence = tmp_path / "curve-two.failure.json"
    second = _attest(second_fixture, second_evidence)
    assert first["observed_failure"] == second["observed_failure"]
    assert first["failure_subject_id"] != second["failure_subject_id"]

    _activate_failure_fixture(first_fixture, monkeypatch)
    assert _verify(first_fixture, first_evidence) == first
    _activate_failure_fixture(second_fixture, monkeypatch)
    assert _verify(second_fixture, second_evidence) == second


def test_subject_curve_excludes_runtime_timing_fields(
    failure_fixture: SimpleNamespace, tmp_path: Path
) -> None:
    sealed = _attest(failure_fixture, tmp_path / "subject-timing.json")
    summary = json.loads(failure_fixture.summary_path.read_text(encoding="utf-8"))
    recipe = json.loads(failure_fixture.recipe_path.read_text(encoding="utf-8"))
    baseline = failure_attestor._failure_subject_material(
        source_subject_id=sealed["source_subject_id"],
        candidate_pilot_subject_id=sealed["candidate_pilot_subject_id"],
        attempt_id=sealed["attempt_id"],
        blind_semantic=sealed["blind_manifest_contract"],
        observed_failure=sealed["observed_failure"],
        recipe=recipe,
        summary=summary,
    )
    assert len(baseline["per_validation_epoch_curve"]) == 31
    assert all(
        "train_loss" not in record
        for record in baseline["per_validation_epoch_curve"]
    )
    for record in summary["records"]:
        if record["validation_performed"]:
            record["train_loss"] += 10.0
            record["validation_seconds"] = 123.0 + record["epoch"]
            record["epoch_seconds"] = 456.0 + record["epoch"]
            record["val_elapsed_seconds"] = 789.0 + record["epoch"]
    changed_timing = failure_attestor._failure_subject_material(
        source_subject_id=sealed["source_subject_id"],
        candidate_pilot_subject_id=sealed["candidate_pilot_subject_id"],
        attempt_id=sealed["attempt_id"],
        blind_semantic=sealed["blind_manifest_contract"],
        observed_failure=sealed["observed_failure"],
        recipe=recipe,
        summary=summary,
    )
    assert changed_timing == baseline


@pytest.mark.parametrize("mutation", ["target", "metrics", "recipe", "selector"])
def test_failure_subject_changes_for_each_semantic_axis(
    failure_fixture: SimpleNamespace, tmp_path: Path, mutation: str
) -> None:
    sealed = _attest(failure_fixture, tmp_path / f"subject-{mutation}.json")
    summary = json.loads(failure_fixture.summary_path.read_text(encoding="utf-8"))
    recipe = json.loads(failure_fixture.recipe_path.read_text(encoding="utf-8"))
    observed = json.loads(json.dumps(sealed["observed_failure"]))
    blind = json.loads(json.dumps(sealed["blind_manifest_contract"]))

    baseline = failure_attestor._failure_subject_material(
        source_subject_id=sealed["source_subject_id"],
        candidate_pilot_subject_id=sealed["candidate_pilot_subject_id"],
        attempt_id=sealed["attempt_id"],
        blind_semantic=blind,
        observed_failure=observed,
        recipe=recipe,
        summary=summary,
    )
    encoded = json.dumps(baseline, sort_keys=True)
    assert str(failure_fixture.candidate_root) not in encoded
    assert str(failure_fixture.source_checkpoint) not in encoded
    assert "artifact_content" not in baseline
    assert "checkpoint_path" not in encoded
    assert "source_checkpoint_sha256" not in encoded
    assert failure_attestor._canonical_sha256(baseline) == sealed["failure_subject_id"]

    if mutation == "target":
        summary["config"]["recipient_open_text_layers"] += 1
    elif mutation == "metrics":
        observed["best_recipient_exact_matches"] -= 1
    elif mutation == "recipe":
        recipe["training_args"]["batch_size"] += 1
    else:
        summary["checkpoint_selection_policy"]["mode"] = "different_selector"
    changed = failure_attestor._failure_subject_material(
        source_subject_id=sealed["source_subject_id"],
        candidate_pilot_subject_id=sealed["candidate_pilot_subject_id"],
        attempt_id=sealed["attempt_id"],
        blind_semantic=blind,
        observed_failure=observed,
        recipe=recipe,
        summary=summary,
    )
    assert failure_attestor._canonical_sha256(changed) != sealed["failure_subject_id"]


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("schedule", "validation schedule"),
        ("count_rate", "count/rate"),
        ("best", "best matches"),
        ("last", "last matches"),
        ("amount_guard", "amount floor"),
        ("status_guard", "status floor"),
        ("unsafe_status", "unsafe status errors"),
        ("eligibility", "checkpoint eligibility"),
    ],
)
def test_recomputes_schedule_counts_best_last_and_guards(
    failure_fixture: SimpleNamespace,
    tmp_path: Path,
    mutation: str,
    message: str,
) -> None:
    summary = json.loads(failure_fixture.summary_path.read_text(encoding="utf-8"))
    if mutation == "schedule":
        summary["records"][2]["validation_performed"] = True
    elif mutation == "count_rate":
        summary["records"][1]["val_candidate_text_by_field"]["recipient_field"][
            "exact_match"
        ] += 0.001
    elif mutation == "best":
        metric = summary["records"][43]["val_candidate_text_by_field"][
            "recipient_field"
        ]
        metric["exact_matches"] -= 1
        metric["exact_match"] = metric["exact_matches"] / EXPECTED_RECIPIENT_VAL_RECORDS
    elif mutation == "last":
        metric = summary["records"][59]["val_candidate_text_by_field"][
            "recipient_field"
        ]
        metric["exact_matches"] -= 1
        metric["exact_match"] = metric["exact_matches"] / EXPECTED_RECIPIENT_VAL_RECORDS
    elif mutation == "amount_guard":
        summary["records"][1]["val_candidate_text_by_field"]["amount"][
            "exact_match"
        ] = AMOUNT_FLOOR - 0.001
    elif mutation == "status_guard":
        summary["records"][1]["val_ctc_by_field"]["transfer_status"][
            "exact_match"
        ] = STATUS_TEXT_FLOOR - 0.001
    elif mutation == "unsafe_status":
        summary["records"][1]["val_status_non_success_to_success"] = 1
    else:
        summary["records"][1]["checkpoint_selection_eligible"] = False
    _write_json(failure_fixture.summary_path, summary)
    _refresh_fixture_pin("training_summary", failure_fixture.summary_path)
    with pytest.raises(ValueError, match=message):
        _attest(failure_fixture, tmp_path / f"{mutation}.json")


@pytest.mark.parametrize(
    ("epoch_index", "invalid"),
    [
        (0, "missing"),
        (2, True),
        (30, None),
        (43, "0.25"),
        (59, float("inf")),
    ],
)
def test_requires_finite_numeric_nonbool_train_loss_on_all_60_records(
    failure_fixture: SimpleNamespace,
    tmp_path: Path,
    epoch_index: int,
    invalid: object,
) -> None:
    summary = json.loads(failure_fixture.summary_path.read_text(encoding="utf-8"))
    if invalid == "missing":
        del summary["records"][epoch_index]["train_loss"]
    else:
        summary["records"][epoch_index]["train_loss"] = invalid
    if invalid == float("inf"):
        failure_fixture.summary_path.write_text(
            json.dumps(summary, allow_nan=True, sort_keys=True), encoding="utf-8"
        )
    else:
        _write_json(failure_fixture.summary_path, summary)
    _refresh_fixture_pin("training_summary", failure_fixture.summary_path)
    with pytest.raises(ValueError, match="train_loss|non-finite"):
        _attest(
            failure_fixture,
            tmp_path / f"bad-train-loss-{epoch_index}.json",
        )


@pytest.mark.parametrize(
    ("surface", "message"),
    [
        ("onnx", "ONNX artifact"),
        ("candidate_evidence", "forbidden candidate output"),
        ("artifact_root", "forbidden candidate output"),
        ("test_json", "test evidence"),
    ],
)
def test_requires_candidate_onnx_and_test_absence(
    failure_fixture: SimpleNamespace,
    tmp_path: Path,
    surface: str,
    message: str,
) -> None:
    if surface == "onnx":
        (failure_fixture.candidate_root / "unexpected.onnx").write_bytes(b"forbidden")
    elif surface == "candidate_evidence":
        _write_json(
            failure_fixture.candidate_root / "recipient_v14_candidate.json", {}
        )
    elif surface == "artifact_root":
        (failure_fixture.candidate_root / "artifacts").mkdir()
    else:
        _write_json(
            failure_fixture.candidate_root / "unexpected-test.json",
            {"evaluation_split": "test"},
        )
    with pytest.raises(ValueError, match=message):
        _attest(failure_fixture, tmp_path / f"{surface}.json")


@pytest.mark.parametrize(
    "claim",
    [
        "onnx_exported",
        "onnx_export_authorized",
        "test_evaluation_authorized",
        "warmstart_authorized",
        "warm_start_authorized",
        "same_route_retry_authorized",
        "same_route_continuation_authorized",
        "continuation_authorized",
        "production_authorized",
        "prod_authorized",
        "prod_route_authorized",
        "test_route_enabled",
    ],
)
@pytest.mark.parametrize("nesting", ["top", "nested"])
def test_rejects_every_contradictory_true_authority_claim(
    failure_fixture: SimpleNamespace, tmp_path: Path, claim: str, nesting: str
) -> None:
    payload = {claim: True}
    if nesting == "nested":
        payload = {"nested_authority": [{"claims": payload}]}
    _write_json(
        failure_fixture.candidate_root / f"contradictory-{claim}-{nesting}.json",
        payload,
    )
    with pytest.raises(ValueError, match=f"unsafe {claim}"):
        _attest(
            failure_fixture,
            tmp_path / f"contradictory-{claim}-{nesting}.evidence.json",
        )


@pytest.mark.parametrize(
    "claim", ["prod_authorized", "prod_route_authorized", "warm_start_authorized"]
)
@pytest.mark.parametrize("checkpoint", ["best", "last"])
@pytest.mark.parametrize("nesting", ["top", "nested"])
def test_rejects_unsafe_true_claims_inside_frozen_checkpoint_payloads(
    failure_fixture: SimpleNamespace,
    tmp_path: Path,
    claim: str,
    checkpoint: str,
    nesting: str,
) -> None:
    path = (
        failure_fixture.best_path
        if checkpoint == "best"
        else failure_fixture.last_path
    )
    payload = failure_fixture.checkpoint_payloads[path.resolve()]
    if nesting == "top":
        payload[claim] = True
    else:
        payload["nested_authority"] = [{"claims": {claim: True}}]
    with pytest.raises(ValueError, match=f"unsafe {claim}"):
        _attest(
            failure_fixture,
            tmp_path / f"checkpoint-{checkpoint}-{claim}-{nesting}.json",
        )


def test_binds_checkpoint_labels_attempt_and_authority_bytes(
    failure_fixture: SimpleNamespace, tmp_path: Path
) -> None:
    evidence = tmp_path / "fresh60.failure.json"
    _attest(failure_fixture, evidence)

    failure_fixture.last_path.write_bytes(
        failure_fixture.last_path.read_bytes() + b"tampered"
    )
    with pytest.raises(ValueError, match="pin mismatch"):
        _verify(failure_fixture, evidence)


def test_real_run_pins_reject_a_semantically_coherent_reserialization(
    failure_fixture: SimpleNamespace, tmp_path: Path
) -> None:
    failure_fixture.summary_path.write_text(
        failure_fixture.summary_path.read_text(encoding="utf-8") + " ",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="fixed training_summary pin mismatch"):
        _attest(failure_fixture, tmp_path / "coherent-splice.json")


def test_unpinned_checkpoint_is_rejected_before_torch_load(
    failure_fixture: SimpleNamespace,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    failure_fixture.best_path.write_bytes(
        failure_fixture.best_path.read_bytes() + b"untrusted-pickle"
    )
    untrusted_bytes = failure_fixture.best_path.read_bytes()
    calls = 0
    original_load = failure_attestor._load_checkpoint

    def forbidden_load(source, *, torch):
        nonlocal calls
        if hasattr(source, "getvalue") and source.getvalue() == untrusted_bytes:
            calls += 1
            raise AssertionError("an unpinned checkpoint reached torch.load")
        return original_load(source, torch=torch)

    monkeypatch.setattr(failure_attestor, "_load_checkpoint", forbidden_load)
    with pytest.raises(ValueError, match="fixed best_checkpoint pin mismatch"):
        _attest(failure_fixture, tmp_path / "untrusted-checkpoint.json")
    assert calls == 0


@pytest.mark.parametrize("authority", ["source", "candidate_pilot"])
def test_unpinned_authority_checkpoint_is_rejected_before_any_torch_load(
    failure_fixture: SimpleNamespace,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    authority: str,
) -> None:
    checkpoint = (
        failure_fixture.source_checkpoint
        if authority == "source"
        else failure_fixture.candidate_pilot_checkpoint
    )
    checkpoint.write_bytes(checkpoint.read_bytes() + b"untrusted-pickle")
    calls = 0

    def forbidden_load(*args, **kwargs):
        nonlocal calls
        calls += 1
        raise AssertionError("an unpinned authority checkpoint reached torch.load")

    monkeypatch.setattr(
        failure_attestor._candidate_source_verifier,
        "_load_checkpoint",
        forbidden_load,
    )

    def transitive_verifier_would_load(**kwargs):
        del kwargs
        return failure_attestor._candidate_source_verifier._load_checkpoint(
            checkpoint, torch=object()
        )

    monkeypatch.setattr(
        failure_attestor,
        "verify_full_crop_candidate_source",
        transitive_verifier_would_load,
    )
    monkeypatch.setattr(
        failure_attestor,
        "verify_residual_candidate_pilot",
        transitive_verifier_would_load,
    )
    with pytest.raises(ValueError, match="pin mismatch"):
        _attest(failure_fixture, tmp_path / f"bad-{authority}-authority.json")
    assert calls == 0


def test_deep_verifier_rejects_checkpoint_replacement_after_prevalidation(
    failure_fixture: SimpleNamespace,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0

    def forbidden_deserialize(*args, **kwargs):
        nonlocal calls
        calls += 1
        raise AssertionError("replacement bytes reached checkpoint deserialization")

    monkeypatch.setattr(
        failure_attestor._candidate_source_verifier,
        "_load_checkpoint",
        forbidden_deserialize,
    )

    def replace_then_request_load(**kwargs):
        del kwargs
        checkpoint = failure_fixture.source_checkpoint
        held = checkpoint.with_suffix(".held.pt")
        checkpoint.rename(held)
        checkpoint.write_bytes(held.read_bytes())
        return failure_attestor._candidate_source_verifier._load_checkpoint(
            checkpoint, torch=object()
        )

    monkeypatch.setattr(
        failure_attestor,
        "verify_full_crop_candidate_source",
        replace_then_request_load,
    )
    with pytest.raises(ValueError, match="identity changed during verification"):
        _attest(failure_fixture, tmp_path / "replaced-after-prevalidation.json")
    assert calls == 0


def test_deep_verifier_loads_bound_checkpoints_from_the_frozen_bytes_only(
    failure_fixture: SimpleNamespace,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loaded: list[bytes] = []

    def deserialize_frozen(stream: object, *, torch: object):
        del torch
        assert hasattr(stream, "read")
        loaded.append(stream.read())
        return {}

    monkeypatch.setattr(
        failure_attestor._candidate_source_verifier,
        "_load_checkpoint",
        deserialize_frozen,
    )

    def source_verifier(**kwargs):
        del kwargs
        failure_attestor._candidate_source_verifier._load_checkpoint(
            failure_fixture.source_checkpoint, torch=object()
        )
        return failure_fixture.source_payload

    def pilot_verifier(**kwargs):
        del kwargs
        failure_attestor._candidate_source_verifier._load_checkpoint(
            failure_fixture.candidate_pilot_checkpoint, torch=object()
        )
        return failure_fixture.pilot_payload

    monkeypatch.setattr(
        failure_attestor, "verify_full_crop_candidate_source", source_verifier
    )
    monkeypatch.setattr(
        failure_attestor, "verify_residual_candidate_pilot", pilot_verifier
    )
    _attest(failure_fixture, tmp_path / "frozen-deep-loads.json")
    assert loaded == [
        failure_fixture.source_checkpoint.read_bytes(),
        failure_fixture.candidate_pilot_checkpoint.read_bytes(),
    ]


def test_deep_verifier_consumes_nested_json_and_jsonl_from_frozen_bytes(
    failure_fixture: SimpleNamespace,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path_reopens = 0

    def forbidden_reopen(*args, **kwargs):
        nonlocal path_reopens
        path_reopens += 1
        raise AssertionError("deep verifier reopened a frozen JSON/JSONL path")

    monkeypatch.setattr(
        failure_attestor._candidate_source_verifier,
        "_strict_json",
        forbidden_reopen,
    )
    monkeypatch.setattr(
        failure_attestor._candidate_source_verifier,
        "verify_blind_manifest_contract",
        forbidden_reopen,
    )
    monkeypatch.setattr(
        failure_attestor._candidate_source_verifier,
        "_blind_recipient_val_records",
        forbidden_reopen,
    )

    def source_verifier(**kwargs):
        del kwargs
        summary = failure_attestor._candidate_source_verifier._strict_json(
            failure_fixture.candidate_pilot_summary
        )
        assert summary["initialization"]["checkpoint_path"] == str(
            failure_fixture.source_checkpoint.resolve()
        )
        binding = (
            failure_attestor._candidate_source_verifier.verify_blind_manifest_contract(
                records_path=failure_fixture.blind,
                blind_contract_path=failure_fixture.blind_contract,
            )
        )
        assert (
            failure_attestor._candidate_source_verifier._blind_recipient_val_records(
                binding
            )
            == EXPECTED_RECIPIENT_VAL_RECORDS
        )
        return failure_fixture.source_payload

    monkeypatch.setattr(
        failure_attestor, "verify_full_crop_candidate_source", source_verifier
    )
    _attest(failure_fixture, tmp_path / "frozen-json-jsonl.json")
    assert path_reopens == 0
    assert (
        failure_attestor._candidate_source_verifier._strict_json
        is forbidden_reopen
    )
    assert (
        failure_attestor._candidate_source_verifier.verify_blind_manifest_contract
        is forbidden_reopen
    )
    assert (
        failure_attestor._candidate_source_verifier._blind_recipient_val_records
        is forbidden_reopen
    )


@pytest.mark.parametrize("same_bytes", [False, True])
def test_post_hash_checkpoint_swap_restore_never_reaches_path_loader(
    failure_fixture: SimpleNamespace,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    same_bytes: bool,
) -> None:
    path_hashes = 0
    path_loads = 0

    def forbidden_hash(*args, **kwargs):
        nonlocal path_hashes
        path_hashes += 1
        raise AssertionError("swapped checkpoint reached path hashing")

    def forbidden_load(*args, **kwargs):
        nonlocal path_loads
        path_loads += 1
        raise AssertionError("swapped checkpoint reached deserialization")

    monkeypatch.setattr(
        failure_attestor._candidate_source_verifier, "_sha256", forbidden_hash
    )
    monkeypatch.setattr(
        failure_attestor._candidate_source_verifier,
        "_load_checkpoint",
        forbidden_load,
    )

    def swap_restore_then_request(**kwargs):
        del kwargs
        checkpoint = failure_fixture.source_checkpoint
        original = checkpoint.read_bytes()
        held = checkpoint.with_suffix(".swap-held.pt")
        checkpoint.rename(held)
        checkpoint.write_bytes(original if same_bytes else b"X" * len(original))
        try:
            failure_attestor._candidate_source_verifier._sha256(checkpoint)
            return failure_attestor._candidate_source_verifier._load_checkpoint(
                checkpoint, torch=object()
            )
        finally:
            checkpoint.unlink()
            held.rename(checkpoint)

    monkeypatch.setattr(
        failure_attestor,
        "verify_full_crop_candidate_source",
        swap_restore_then_request,
    )
    with pytest.raises(ValueError, match="not in the frozen authority closure"):
        _attest(failure_fixture, tmp_path / f"swap-restore-{same_bytes}.json")
    assert path_hashes == 0
    assert path_loads == 0
    assert failure_attestor._candidate_source_verifier._sha256 is forbidden_hash
    assert failure_attestor._candidate_source_verifier._load_checkpoint is forbidden_load


@pytest.mark.parametrize("unknown", ["descendant", "ancestor"])
def test_unknown_sanitizer_lineage_checkpoint_never_reaches_module_loader(
    failure_fixture: SimpleNamespace,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    unknown: str,
) -> None:
    rogue = tmp_path / f"unknown-{unknown}.pt"
    rogue.write_bytes(b"unknown-sanitizer-checkpoint")
    path_hashes = 0
    path_loads = 0

    def forbidden_hash(*args, **kwargs):
        nonlocal path_hashes
        path_hashes += 1
        raise AssertionError("unknown sanitizer checkpoint reached path hashing")

    def forbidden_load(*args, **kwargs):
        nonlocal path_loads
        path_loads += 1
        raise AssertionError("unknown sanitizer checkpoint was deserialized")

    monkeypatch.setattr(
        failure_attestor._seed_sanitizer_verifier,
        "_sha256",
        forbidden_hash,
    )
    monkeypatch.setattr(
        failure_attestor._seed_sanitizer_verifier,
        "_load_checkpoint",
        forbidden_load,
    )
    if unknown == "ancestor":
        train_payload = failure_fixture.checkpoint_payloads[
            failure_fixture.sanitizer_train_checkpoint.resolve()
        ]
        train_payload["initialization"]["checkpoint_path"] = str(rogue.resolve())
        train_payload["initialization"]["checkpoint_sha256"] = _sha(rogue)
    else:

        def request_unknown_descendant(**kwargs):
            del kwargs
            try:
                failure_attestor._seed_sanitizer_verifier._sha256(rogue)
            except ValueError:
                pass
            return failure_attestor._seed_sanitizer_verifier._load_checkpoint(
                rogue, torch=object()
            )

        monkeypatch.setattr(
            failure_attestor,
            "verify_full_crop_candidate_source",
            request_unknown_descendant,
        )
    with pytest.raises(ValueError, match="frozen|authority closure"):
        _attest(failure_fixture, tmp_path / f"unknown-{unknown}.json")
    assert path_hashes == 0
    assert path_loads == 0
    assert failure_attestor._seed_sanitizer_verifier._sha256 is forbidden_hash
    assert failure_attestor._seed_sanitizer_verifier._load_checkpoint is forbidden_load


def test_nested_json_checkpoint_redirect_is_rejected_before_deserialization(
    failure_fixture: SimpleNamespace,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rogue = tmp_path / "unbound-rogue.pt"
    rogue.write_bytes(b"unbound-pickle")
    _write_json(
        failure_fixture.candidate_pilot_summary,
        {"initialization": {"checkpoint_path": str(rogue.resolve())}},
    )
    evidence = json.loads(
        failure_fixture.candidate_pilot_evidence.read_text(encoding="utf-8")
    )
    unsigned = {
        key: value for key, value in evidence.items() if key != "integrity_sha256"
    }
    unsigned["artifacts"]["candidate_training_summary"] = _binding(
        failure_fixture.candidate_pilot_summary
    )
    _write_json(failure_fixture.candidate_pilot_evidence, _sealed(unsigned))
    _refresh_authority_pin(
        "candidate_pilot_evidence", failure_fixture.candidate_pilot_evidence
    )
    calls = 0

    def forbidden_deserialize(*args, **kwargs):
        nonlocal calls
        calls += 1
        raise AssertionError("nested redirect reached checkpoint deserialization")

    monkeypatch.setattr(
        failure_attestor._candidate_source_verifier,
        "_load_checkpoint",
        forbidden_deserialize,
    )
    with pytest.raises(ValueError, match="no frozen artifact hash/size/identity binding"):
        _attest(failure_fixture, tmp_path / "nested-redirect.json")
    assert calls == 0


def test_binds_attestor_and_runner_code_hashes(
    failure_fixture: SimpleNamespace,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frozen_code = tmp_path / "frozen-verifier.py"
    frozen_code.write_text("VERSION = 1\n", encoding="utf-8")
    monkeypatch.setattr(
        failure_attestor,
        "_code_paths",
        lambda: {"code_test_frozen_verifier": frozen_code},
    )
    evidence = tmp_path / "code-bound.failure.json"
    first = _attest(failure_fixture, evidence)
    frozen_code.write_text("VERSION = 2\n", encoding="utf-8")
    second = _attest(failure_fixture, tmp_path / "code-drift.failure.json")
    assert second["failure_subject_id"] == first["failure_subject_id"]
    assert second["code"] != first["code"]
    with pytest.raises(ValueError, match="authoritative source"):
        _verify(failure_fixture, evidence)


def test_requires_the_real_programdata_registry_identity(
    failure_fixture: SimpleNamespace,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    impostor = tmp_path / "OtherProgramData" / "ReceiptAI" / ATTEMPT_REGISTRY_NAME
    impostor.mkdir(parents=True)
    monkeypatch.setattr(
        failure_attestor,
        "_windows_programdata_attempt_registry",
        lambda: failure_fixture.registry.resolve(),
    )
    with pytest.raises(ValueError, match="Windows ProgramData training attempt registry"):
        attest_fresh60_failure(
            candidate_root=failure_fixture.candidate_root,
            source_contract_path=failure_fixture.source_contract,
            candidate_pilot_evidence_path=failure_fixture.candidate_pilot_evidence,
            full_records=failure_fixture.full,
            attempt_registry=impostor,
            output_evidence=tmp_path / "impostor-registry.json",
            torch=object(),
        )


def test_checkpoint_validation_uses_the_already_pinned_frozen_bytes(
    failure_fixture: SimpleNamespace,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = failure_attestor._validate_summary_recipe

    def replace_after_freeze(*args, **kwargs):
        result = original(*args, **kwargs)
        failure_fixture.best_path.write_bytes(b"replacement-after-freeze")
        return result

    monkeypatch.setattr(
        failure_attestor, "_validate_summary_recipe", replace_after_freeze
    )
    with pytest.raises(ValueError, match="changed during attestation"):
        _attest(failure_fixture, tmp_path / "frozen-checkpoint.json")


def test_fixed_artifacts_have_one_freeze_open_and_only_required_closing_opens(
    failure_fixture: SimpleNamespace,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixed_paths = {
        path.absolute(): name
        for name, path in failure_fixture.fixture_pin_paths.items()
    }
    open_counts = {name: 0 for name in fixed_paths.values()}
    original_open = Path.open

    def counted_open(self: Path, *args, **kwargs):
        name = fixed_paths.get(self.absolute())
        if name is not None:
            open_counts[name] += 1
        return original_open(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", counted_open)
    _attest(failure_fixture, tmp_path / "fixed-single-open.json")
    # The blind artifacts are already members of the source/A8 closure, so
    # they have one freeze open plus both the deep-verifier and outer closes.
    # Every other fixed-run artifact has one freeze open and the outer close.
    assert open_counts == {
        name: (3 if name in {"blind_manifest", "blind_contract"} else 2)
        for name in open_counts
    }


@pytest.mark.parametrize("same_bytes", [False, True])
def test_fixed_json_swap_read_restore_never_reopens_the_path(
    failure_fixture: SimpleNamespace,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    same_bytes: bool,
) -> None:
    names = (
        "training_summary",
        "training_recipe",
        "training_labels",
        "training_attempt",
    )
    held_paths: dict[Path, Path] = {}
    original_freeze = failure_attestor._freeze_fixed_run_artifacts
    original_validate = failure_attestor._validate_summary_recipe
    original_strict_json = failure_attestor._strict_json
    path_reopens = 0

    def freeze_then_swap(paths, **kwargs):
        frozen = original_freeze(paths, **kwargs)
        for index, name in enumerate(names):
            path = paths[name]
            # Keep the injected sibling name compact.  In particular, the
            # 64-hex attempt marker already sits below a deep pytest/ProgramData
            # fixture path and appending its original leaf can cross legacy
            # Win32 MAX_PATH even though the production artifact is valid.
            held = path.with_name(f".swap-held-{index}")
            path.rename(held)
            path.write_bytes(
                held.read_bytes() if same_bytes else b"{malicious-path-json"
            )
            held_paths[path] = held
        return frozen

    def restore() -> None:
        for path, held in held_paths.items():
            if held.exists():
                path.unlink()
                held.rename(path)

    def restore_then_validate(*args, **kwargs):
        restore()
        return original_validate(*args, **kwargs)

    fixed_lexical = {
        failure_fixture.fixture_pin_paths[name].absolute() for name in names
    }

    def forbid_fixed_path_reopen(path: Path):
        nonlocal path_reopens
        if Path(path).absolute() in fixed_lexical:
            path_reopens += 1
            raise AssertionError("fixed JSON was reparsed from its pathname")
        return original_strict_json(path)

    monkeypatch.setattr(
        failure_attestor, "_freeze_fixed_run_artifacts", freeze_then_swap
    )
    monkeypatch.setattr(
        failure_attestor, "_validate_summary_recipe", restore_then_validate
    )
    monkeypatch.setattr(failure_attestor, "_strict_json", forbid_fixed_path_reopen)
    try:
        _attest(
            failure_fixture,
            tmp_path / f"fixed-json-swap-restore-{same_bytes}.json",
        )
    finally:
        restore()
    assert path_reopens == 0


def test_fixed_blind_jsonl_swap_read_restore_uses_frozen_bytes(
    failure_fixture: SimpleNamespace,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    names = ("blind_contract", "blind_manifest", "full_manifest")
    held_paths: dict[Path, Path] = {}
    original_freeze = failure_attestor._freeze_fixed_run_artifacts
    original_frozen_verify = failure_attestor._verify_frozen_blind_manifest_contract
    path_verifier_calls = 0

    def freeze_then_swap(paths, **kwargs):
        frozen = original_freeze(paths, **kwargs)
        for name in names:
            path = (
                failure_fixture.full
                if name == "full_manifest"
                else paths[name]
            )
            held = path.with_name(f"{path.name}.{name}.held")
            path.rename(held)
            path.write_bytes(b"malicious-path-data")
            held_paths[path] = held
        return frozen

    def restore() -> None:
        for path, held in held_paths.items():
            if held.exists():
                path.unlink()
                held.rename(path)

    def restore_then_verify(**kwargs):
        restore()
        return original_frozen_verify(**kwargs)

    def forbidden_path_verifier(*args, **kwargs):
        nonlocal path_verifier_calls
        path_verifier_calls += 1
        raise AssertionError("blind data reached the pathname verifier")

    monkeypatch.setattr(
        failure_attestor, "_freeze_fixed_run_artifacts", freeze_then_swap
    )
    monkeypatch.setattr(
        failure_attestor,
        "_verify_frozen_blind_manifest_contract",
        restore_then_verify,
    )
    monkeypatch.setattr(
        failure_attestor._full_crop_pilot_verifier,
        "verify_blind_manifest_contract",
        forbidden_path_verifier,
    )
    try:
        _attest(failure_fixture, tmp_path / "fixed-blind-swap-restore.json")
    finally:
        restore()
    assert path_verifier_calls == 0


def test_rejects_checkpoint_labels_and_attempt_semantic_tampering(
    failure_fixture: SimpleNamespace, tmp_path: Path
) -> None:
    failure_fixture.checkpoint_payloads[failure_fixture.best_path.resolve()]["epoch"] = 45
    with pytest.raises(ValueError, match="best checkpoint epoch"):
        _attest(failure_fixture, tmp_path / "bad-checkpoint.json")

    failure_fixture.checkpoint_payloads[failure_fixture.best_path.resolve()]["epoch"] = 44
    labels = json.loads(failure_fixture.labels_path.read_text(encoding="utf-8"))
    labels["recipient_charset_sha256"] = "0" * 64
    _write_json(failure_fixture.labels_path, labels)
    _refresh_fixture_pin("training_labels", failure_fixture.labels_path)
    with pytest.raises(ValueError, match="training labels charset"):
        _attest(failure_fixture, tmp_path / "bad-labels.json")

    _write_json(failure_fixture.labels_path, failure_fixture.labels)
    _refresh_fixture_pin("training_labels", failure_fixture.labels_path)
    attempt = json.loads(failure_fixture.attempt_path.read_text(encoding="utf-8"))
    attempt["stage"] = "candidate-61e"
    _write_json(failure_fixture.attempt_path, attempt)
    _refresh_fixture_pin("training_attempt", failure_fixture.attempt_path)
    with pytest.raises(ValueError, match="attempt stage"):
        _attest(failure_fixture, tmp_path / "bad-attempt.json")


def test_rejects_same_bytes_at_a_different_full_manifest_path(
    failure_fixture: SimpleNamespace, tmp_path: Path
) -> None:
    evidence = tmp_path / "fresh60.failure.json"
    _attest(failure_fixture, evidence)
    copied_full = tmp_path / "copied-full.jsonl"
    copied_full.write_bytes(failure_fixture.full.read_bytes())
    with pytest.raises(ValueError, match="full manifest"):
        verify_fresh60_failure(
            evidence_path=evidence,
            source_contract_path=failure_fixture.source_contract,
            candidate_pilot_evidence_path=failure_fixture.candidate_pilot_evidence,
            full_records=copied_full,
            attempt_registry=failure_fixture.registry,
            torch=object(),
        )


def test_detects_toctou_mutation_after_validation(
    failure_fixture: SimpleNamespace,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = failure_attestor._validate_labels

    def mutate_after_validation(*args, **kwargs):
        original(*args, **kwargs)
        failure_fixture.recipe_path.write_text(
            failure_fixture.recipe_path.read_text(encoding="utf-8") + " ",
            encoding="utf-8",
        )

    monkeypatch.setattr(failure_attestor, "_validate_labels", mutate_after_validation)
    with pytest.raises(ValueError, match="changed during attestation"):
        _attest(failure_fixture, tmp_path / "toctou.json")


def test_detects_source_authority_mutation_during_reverification(
    failure_fixture: SimpleNamespace,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = failure_attestor.verify_full_crop_candidate_source

    def mutate_authority(**kwargs):
        result = original(**kwargs)
        failure_fixture.source_contract.write_text(
            failure_fixture.source_contract.read_text(encoding="utf-8") + " ",
            encoding="utf-8",
        )
        return result

    monkeypatch.setattr(
        failure_attestor,
        "verify_full_crop_candidate_source",
        mutate_authority,
    )
    with pytest.raises(ValueError, match="changed during verification"):
        _attest(failure_fixture, tmp_path / "source-toctou.json")


def test_detects_same_bytes_authority_identity_replacement_after_deep_verify(
    failure_fixture: SimpleNamespace,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = failure_attestor.verify_full_crop_candidate_source

    def replace_authority_identity(**kwargs):
        result = original(**kwargs)
        source = failure_fixture.source_contract
        held = source.with_suffix(".held.json")
        source.rename(held)
        source.write_bytes(held.read_bytes())
        return result

    monkeypatch.setattr(
        failure_attestor,
        "verify_full_crop_candidate_source",
        replace_authority_identity,
    )
    with pytest.raises(ValueError, match="identity changed during verification"):
        _attest(failure_fixture, tmp_path / "source-identity-toctou.json")


def test_rejects_reparse_artifact_and_reparse_output_parent(
    failure_fixture: SimpleNamespace, tmp_path: Path
) -> None:
    real_labels = tmp_path / "real-labels.json"
    real_labels.write_bytes(failure_fixture.labels_path.read_bytes())
    failure_fixture.labels_path.unlink()
    try:
        failure_fixture.labels_path.symlink_to(real_labels)
    except OSError:
        pytest.skip("symlink creation unavailable")
    with pytest.raises(ValueError, match="reparse"):
        _attest(failure_fixture, tmp_path / "reparse-artifact.json")

    failure_fixture.labels_path.unlink()
    failure_fixture.labels_path.write_bytes(real_labels.read_bytes())
    real_parent = tmp_path / "real-parent"
    real_parent.mkdir()
    linked_parent = tmp_path / "linked-parent"
    linked_parent.symlink_to(real_parent, target_is_directory=True)
    with pytest.raises(ValueError, match="reparse"):
        _attest(failure_fixture, linked_parent / "failure.json")


def test_wrapper_and_cli_are_read_only_attestor_entrypoints() -> None:
    repository = Path(__file__).parents[1]
    wrapper = (
        repository / "scripts" / "receipt-ocr-recipient-v14-failure-attest.py"
    ).read_text(encoding="utf-8")
    module = (
        repository
        / "src"
        / "transfer_receipt_ai"
        / "recipient_v14_failure_attestor.py"
    ).read_text(encoding="utf-8")
    assert "recipient_v14_failure_attestor import main" in wrapper
    assert 'subparsers.add_parser("attest-failure")' in module
    assert 'subparsers.add_parser("verify-failure")' in module
    assert "--candidate-pilot-evidence" in module
    assert "--attempt-registry" in module
    assert "new_view_pilot_authority" in module
    assert "same_route_retry_authorized" in module
    assert "failed_checkpoint_initialization_authorized" in module
    assert "onnx_export_authorized" in module
    assert "test_evaluation_authorized" in module


def test_real_failed_run_artifact_pins_are_exactly_frozen() -> None:
    assert failure_attestor.EXPECTED_SOURCE_SUBJECT_ID == (
        "98f0617404d7d58e99a0794d2340da9154f81667f0aa6a546027dd19209b886a"
    )
    assert failure_attestor.EXPECTED_CANDIDATE_PILOT_SUBJECT_ID == (
        "5d5c0cbe5041252dc9de8d69076400deb7c8d3909d81c424287863d59b49433e"
    )
    assert failure_attestor.EXPECTED_ATTEMPT_ID == (
        "155156f0678fd697904fcca953c611836896dcdc62cb09839eb66f8d62c5c66d"
    )
    assert failure_attestor.EXPECTED_AUTHORITY_DOCUMENT_PINS == {
        "source_contract": {
            "size_bytes": 9007,
            "sha256": "5f1c5eebd72e215ad4e4f7be265b0d37be98b3de7a86e2d2e909437756153246",
        },
        "candidate_pilot_evidence": {
            "size_bytes": 7627,
            "sha256": "324ab62634a9fc054d3aab70b3ce9e2800da994534d9736119b8738bc8bff4b3",
        },
    }
    assert failure_attestor.EXPECTED_RUN_ARTIFACT_PINS == {
        "training_summary": {
            "size_bytes": 230694,
            "sha256": "2f582138f6751fda4392e12b6398745c89b176ce3afe6ec25875b337376cb9b4",
        },
        "best_checkpoint": {
            "size_bytes": 39442731,
            "sha256": "2c800d418088fa11dcfd11eaacd7e14bbc6a4b4be820ddfefed590853f06ec81",
        },
        "last_checkpoint": {
            "size_bytes": 39442795,
            "sha256": "7186ce22a4f5981021a8f220f5f772b7777c324eb89cb2cfc357707b5297a742",
        },
        "training_labels": {
            "size_bytes": 68944,
            "sha256": "f5f0c26b20dba7e848a63d98b204e6125fd7e2c9f7f1dec5bff94a93ffa5123f",
        },
        "training_recipe": {
            "size_bytes": 1213,
            "sha256": "76e98292f7309c2e8e6e21f575deb39dccd22c14370b29b57cfb88aefc32b4a6",
        },
        "blind_manifest": {
            "size_bytes": 202226294,
            "sha256": "c303c8a34348532263d3ad84ed2cd6ddcd77c1bdd9dfc8a7c713ccc35a1ff5f1",
        },
        "blind_contract": {
            "size_bytes": 1011,
            "sha256": "bc103913e77e35a4a54ac302ea7ce3bc7bca688f50ab8e6e3bc090b488f0d440",
        },
        "training_attempt": {
            "size_bytes": 844,
            "sha256": "9e6916c91073cf2ca8037f2cde593e5151d8315c630074991e4e682701ef5e24",
        },
    }
