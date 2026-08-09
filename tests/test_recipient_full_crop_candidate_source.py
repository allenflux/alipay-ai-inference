"""Content-bound contracts for the post-full-crop recipient v14 route."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from dataclasses import asdict
from pathlib import Path

import pytest

import transfer_receipt_ai.recipient_full_crop_candidate_source as source_contract
from tests.test_recipient_full_crop_pilot import _summary, _write_seed
from transfer_receipt_ai.ocr_unified import (
    INIT_CHECKPOINT_MODE_RECIPIENT_VISUAL_CONTEXT_REINIT,
    UnifiedReaderConfig,
)
from transfer_receipt_ai.recipient_blind_manifest import build_blind_manifest
from transfer_receipt_ai.recipient_full_crop_candidate_source import (
    CANDIDATE_PILOT_DECISION,
    CANDIDATE_PILOT_KIND,
    SOURCE_DECISION,
    SOURCE_KIND,
    attest_full_crop_candidate_source,
    evaluate_residual_candidate_pilot,
    seal_residual_candidate_pilot,
    verify_full_crop_candidate_source,
    verify_residual_candidate_pilot,
)
from transfer_receipt_ai.recipient_full_crop_pilot import (
    evaluate_pilot_summary,
    verify_blind_manifest_contract,
)
from transfer_receipt_ai.recipient_final_gate import inspect_candidate


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, allow_nan=False, sort_keys=True, indent=2)
        + "\n",
        encoding="utf-8",
    )


@pytest.fixture()
def fake_checkpoints(monkeypatch: pytest.MonkeyPatch):
    payloads: dict[Path, dict[str, object]] = {}

    def register(path: Path, payload: dict[str, object]) -> None:
        path.write_bytes((path.name + "-sealed-checkpoint").encode("utf-8"))
        payloads[path.resolve()] = payload

    def load(path: Path, *, torch: object):
        del torch
        resolved = Path(path).resolve()
        try:
            return payloads[resolved]
        except KeyError:
            for registered_path, payload in payloads.items():
                try:
                    if os.path.samefile(resolved, registered_path):
                        return payload
                except OSError:
                    continue
            raise AssertionError(f"unexpected checkpoint load: {path}")

    def rewrite_equivalent(source: Path, target: Path) -> None:
        source_payload = payloads[source.resolve()]
        target.unlink()
        target.write_bytes(b"different-serialization-of-the-same-tested-state")
        payloads[target.resolve()] = {
            **source_payload,
            "ignored_checkpoint_nonce": "must-not-create-a-new-subject",
        }

    register.rewrite_equivalent = rewrite_equivalent

    monkeypatch.setattr(source_contract, "_load_checkpoint", load)
    monkeypatch.setattr(
        source_contract,
        "_validate_recipient_full_crop_seed_policy",
        lambda payload, *, torch: None,
    )
    monkeypatch.setattr(
        source_contract,
        "_checkpoint_state_identity",
        lambda state: {"test_state_keys": sorted(str(key) for key in state)},
    )
    return register


def _full_and_blind(root: Path) -> tuple[Path, Path, Path]:
    full = root / "full.jsonl"
    rows = [
        {"id": "train-one", "split": "train", "slots": {}},
        {"id": "val-one", "split": "val", "slots": {}},
        {"id": "test-secret", "split": "test", "slots": {}},
    ]
    full.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    blind_root = root / "blind-train-val"
    blind_root.mkdir()
    blind = blind_root / "unified_fields.train-val.jsonl"
    contract = blind_root / "blind.contract.json"
    build_blind_manifest(source=full, output=blind, contract=contract)
    return full, blind, contract


def _build_passed_full_crop_pilot(
    tmp_path: Path,
    register,
    *,
    best: float = 0.80,
) -> tuple[Path, Path, dict[str, object]]:
    pilot_root = tmp_path / "full-crop-pilot"
    pilot_root.mkdir()
    full, blind, blind_contract = _full_and_blind(pilot_root)
    training = pilot_root / "training-full-crop-pilot"
    training.mkdir()
    seed = tmp_path / "sanitized-seed.pt"
    summary = _summary(best=best, epoch4=0.76, epoch8=0.79)
    summary["status_text_runtime_policy"] = "decode_and_normalize_review_only"
    initialization = dict(summary["initialization"])
    initialization.update(
        {
            "checkpoint_path": str(seed.resolve()),
            "checkpoint_sha256": "pending",
        }
    )
    summary["initialization"] = initialization
    seed_payload = {
        "kind": "receipt_unified_field_reader_v13",
        "config": summary["initialization"]["source_config"],
    }
    register(seed, seed_payload)
    initialization["checkpoint_sha256"] = hashlib.sha256(seed.read_bytes()).hexdigest()

    best_epoch = summary["best_checkpoint_epoch"]
    best_row = next(record for record in summary["records"] if record["epoch"] == best_epoch)
    best_checkpoint = training / "best.pt"
    register(
        best_checkpoint,
        {
            "kind": "receipt_unified_field_reader_v13",
            "state_dict": {"recipient_fake": "sealed"},
            "epoch": best_epoch,
            "metrics": best_row,
            "config": summary["config"],
            "initialization": summary["initialization"],
            "fine_tune_policy": summary["fine_tune_policy"],
            "checkpoint_selection_policy": summary["checkpoint_selection_policy"],
            "recipient_train_split_policy": summary["recipient_train_split_policy"],
            "field_counts": summary["field_counts"],
            "status_text_runtime_policy": summary["status_text_runtime_policy"],
            "training_runtime": summary["training_runtime"],
        },
    )
    summary_path = training / "training_summary.json"
    _write_json(summary_path, summary)
    binding = verify_blind_manifest_contract(
        records_path=blind,
        blind_contract_path=blind_contract,
    )
    decision = {
        **evaluate_pilot_summary(summary),
        "blind_manifest_contract": binding,
    }
    _write_json(training / "pilot_decision.json", decision)
    return pilot_root, full, summary


def test_source_attestor_replays_every_binding_and_is_no_clobber(
    tmp_path: Path,
    fake_checkpoints,
) -> None:
    pilot_root, full, _ = _build_passed_full_crop_pilot(tmp_path, fake_checkpoints)
    contract_path = tmp_path / "source.contract.json"

    contract = attest_full_crop_candidate_source(
        pilot_root=pilot_root,
        output_contract=contract_path,
        torch=object(),
    )
    verified = verify_full_crop_candidate_source(
        pilot_root=pilot_root,
        contract_path=contract_path,
        full_records=full,
        torch=object(),
    )

    assert verified == contract
    assert contract["kind"] == SOURCE_KIND
    assert contract["analysis_only"] is True
    assert contract["production_route_authorized"] is False
    assert contract["test_opened"] is False
    assert contract["onnx_exported"] is False
    assert len(contract["source_subject_id"]) == 64
    assert contract["recomputed_pilot_decision"]["decision"] == SOURCE_DECISION
    assert contract["observed"]["best_recipient_exact"] == pytest.approx(0.80)
    assert {
        "best_checkpoint",
        "training_summary",
        "pilot_decision",
        "full_manifest",
        "blind_manifest",
        "blind_contract",
        "seed_checkpoint",
        "code_candidate_source_attestor",
        "code_full_crop_pilot",
        "code_ocr_unified",
        "code_blind_manifest",
        "code_seed_sanitizer",
    } <= set(contract["artifacts"])

    copied_root = tmp_path / "copied-full-crop-pilot"
    shutil.copytree(pilot_root, copied_root, copy_function=os.link)
    copied_blind = copied_root / "blind-train-val" / "unified_fields.train-val.jsonl"
    copied_blind_contract = copied_root / "blind-train-val" / "blind.contract.json"
    copied_full = copied_root / "full.jsonl"
    copied_contract_payload = json.loads(copied_blind_contract.read_text(encoding="utf-8"))
    copied_contract_payload["source_manifest"] = str(copied_full.resolve())
    copied_contract_payload["blind_manifest"] = str(copied_blind.resolve())
    copied_blind_contract.unlink()
    _write_json(copied_blind_contract, copied_contract_payload)
    copied_binding = verify_blind_manifest_contract(
        records_path=copied_blind,
        blind_contract_path=copied_blind_contract,
    )
    copied_decision_path = (
        copied_root / "training-full-crop-pilot" / "pilot_decision.json"
    )
    copied_decision = json.loads(copied_decision_path.read_text(encoding="utf-8"))
    copied_decision["blind_manifest_contract"] = copied_binding
    copied_decision_path.unlink()
    _write_json(copied_decision_path, copied_decision)
    copied_summary_path = (
        copied_root / "training-full-crop-pilot" / "training_summary.json"
    )
    copied_summary = json.loads(copied_summary_path.read_text(encoding="utf-8"))
    copied_summary["ignored_extension_nonce"] = "must-not-create-a-new-subject"
    copied_summary_path.unlink()
    _write_json(copied_summary_path, copied_summary)
    fake_checkpoints.rewrite_equivalent(
        pilot_root / "training-full-crop-pilot" / "best.pt",
        copied_root / "training-full-crop-pilot" / "best.pt",
    )
    copied_contract = attest_full_crop_candidate_source(
        pilot_root=copied_root,
        output_contract=tmp_path / "copied-source.contract.json",
        torch=object(),
    )
    assert copied_contract["source_subject_id"] == contract["source_subject_id"]
    assert copied_contract["integrity_sha256"] != contract["integrity_sha256"]

    with pytest.raises(ValueError, match="Refusing to overwrite"):
        attest_full_crop_candidate_source(
            pilot_root=pilot_root,
            output_contract=contract_path,
            torch=object(),
        )


def test_source_attestor_rejects_mutation_copy_and_onnx(
    tmp_path: Path,
    fake_checkpoints,
) -> None:
    pilot_root, full, _ = _build_passed_full_crop_pilot(tmp_path, fake_checkpoints)
    contract_path = tmp_path / "source.contract.json"
    attest_full_crop_candidate_source(
        pilot_root=pilot_root,
        output_contract=contract_path,
        torch=object(),
    )

    copied_full = tmp_path / "copied-full.jsonl"
    copied_full.write_bytes(full.read_bytes())
    with pytest.raises(ValueError, match="not the bound file"):
        verify_full_crop_candidate_source(
            pilot_root=pilot_root,
            contract_path=contract_path,
            full_records=copied_full,
            torch=object(),
        )

    decision_path = pilot_root / "training-full-crop-pilot" / "pilot_decision.json"
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    decision["analysis_only"] = False
    _write_json(decision_path, decision)
    with pytest.raises(ValueError, match="decision|changed"):
        verify_full_crop_candidate_source(
            pilot_root=pilot_root,
            contract_path=contract_path,
            full_records=full,
            torch=object(),
        )

    # A source is rejected for ONNX presence even when all JSON claims remain
    # analysis-only.
    decision["analysis_only"] = True
    _write_json(decision_path, decision)
    (pilot_root / "forbidden.onnx").write_bytes(b"not-an-analysis-artifact")
    with pytest.raises(ValueError, match="ONNX"):
        attest_full_crop_candidate_source(
            pilot_root=pilot_root,
            output_contract=tmp_path / "second.contract.json",
            torch=object(),
        )


def test_source_attestor_rejects_best_at_or_above_delivery_floor(
    tmp_path: Path,
    fake_checkpoints,
) -> None:
    pilot_root, _, _ = _build_passed_full_crop_pilot(
        tmp_path,
        fake_checkpoints,
        best=0.90,
    )
    with pytest.raises(ValueError, match="already reached the 90%"):
        attest_full_crop_candidate_source(
            pilot_root=pilot_root,
            output_contract=tmp_path / "source.contract.json",
            torch=object(),
        )


def test_source_attestor_rejects_same_path_full_manifest_byte_change(
    tmp_path: Path,
    fake_checkpoints,
) -> None:
    pilot_root, full, _ = _build_passed_full_crop_pilot(tmp_path, fake_checkpoints)
    full.write_bytes(full.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="full manifest changed after blind-contract"):
        attest_full_crop_candidate_source(
            pilot_root=pilot_root,
            output_contract=tmp_path / "source.contract.json",
            torch=object(),
        )


def test_source_attestor_reopens_real_torch_checkpoints_when_available(
    tmp_path: Path,
) -> None:
    torch, seed, state = _write_seed(tmp_path)
    pilot_root = tmp_path / "real-full-crop-pilot"
    pilot_root.mkdir()
    full, blind, blind_contract = _full_and_blind(pilot_root)
    training = pilot_root / "training-full-crop-pilot"
    training.mkdir()
    summary = _summary()
    summary["status_text_runtime_policy"] = "decode_and_normalize_review_only"
    seed_payload = torch.load(seed, map_location="cpu", weights_only=False)
    initialization = dict(summary["initialization"])
    initialization.update(
        {
            "checkpoint_path": str(seed.resolve()),
            "checkpoint_sha256": hashlib.sha256(seed.read_bytes()).hexdigest(),
            "source_config": seed_payload["config"],
            "source_full_crop_seed_sanitizer_attestation": seed_payload[
                "full_crop_seed_sanitizer_attestation"
            ],
        }
    )
    summary["initialization"] = initialization
    best_epoch = summary["best_checkpoint_epoch"]
    best_row = next(row for row in summary["records"] if row["epoch"] == best_epoch)
    torch.save(
        {
            "kind": "receipt_unified_field_reader_v13",
            "state_dict": state,
            "epoch": best_epoch,
            "metrics": best_row,
            "config": summary["config"],
            "initialization": summary["initialization"],
            "fine_tune_policy": summary["fine_tune_policy"],
            "checkpoint_selection_policy": summary["checkpoint_selection_policy"],
            "recipient_train_split_policy": summary["recipient_train_split_policy"],
            "field_counts": summary["field_counts"],
            "status_text_runtime_policy": summary["status_text_runtime_policy"],
            "training_runtime": summary["training_runtime"],
        },
        training / "best.pt",
    )
    _write_json(training / "training_summary.json", summary)
    binding = verify_blind_manifest_contract(
        records_path=blind,
        blind_contract_path=blind_contract,
    )
    _write_json(
        training / "pilot_decision.json",
        {**evaluate_pilot_summary(summary), "blind_manifest_contract": binding},
    )
    contract_path = tmp_path / "real-source.contract.json"
    sealed = attest_full_crop_candidate_source(
        pilot_root=pilot_root,
        output_contract=contract_path,
        torch=torch,
    )
    verified = verify_full_crop_candidate_source(
        pilot_root=pilot_root,
        contract_path=contract_path,
        full_records=full,
        torch=torch,
    )
    assert verified == sealed


def _residual_summary(
    *,
    source_summary: dict[str, object],
    source_checkpoint: Path,
    source_sha256: str,
) -> dict[str, object]:
    source_config = dict(source_summary["config"])
    target_config = {
        **source_config,
        "recipient_backbone": "residual_positional_transformer_v2",
        "recipient_branch_channels": 16,
        "recipient_hidden_size": 192,
        "recipient_open_text_layers": 4,
        "recipient_open_text_heads": 8,
        "recipient_open_text_feedforward": 1536,
        "recipient_open_text_dropout": 0.10,
    }
    records = []
    for epoch in range(1, 9):
        recipient = 0.76 if epoch == 4 else 0.79 if epoch == 8 else 0.70
        if epoch == 5:
            recipient = 0.80
        records.append(
            {
                "epoch": epoch,
                "validation_performed": True,
                "val_candidate_text_by_field": {
                    "amount": {"exact_match": 0.80},
                    "time": {"exact_match": 0.99},
                    "payment_method_field": {"exact_match": 0.94},
                    "recipient_field": {"exact_match": recipient},
                },
                "val_ctc_by_field": {"transfer_status": {"exact_match": 0.91}},
                "val_status_non_success_to_success": 0,
                "checkpoint_selection_eligible": True,
                "checkpoint_selection_protection_failures": [],
            }
        )
    return {
        "kind": "receipt_unified_field_reader_v13",
        "status_text_runtime_policy": "decode_and_normalize_review_only",
        "config": target_config,
        "initialization": {
            "mode": "parameter_only_recipient_visual_context_reinit",
            "init_checkpoint_mode": INIT_CHECKPOINT_MODE_RECIPIENT_VISUAL_CONTEXT_REINIT,
            "source_kind": "receipt_unified_field_reader_v13",
            "source_config": source_config,
            "checkpoint_path": str(source_checkpoint.resolve()),
            "checkpoint_sha256": source_sha256,
            "financial_label_policy": {
                "recipient_character_map": {
                    "mode": "fresh_train_only_reinitialized_recipient_v1"
                }
            },
        },
        "fine_tune_policy": {
            "mode": "recipient_only_v13",
            "trainable_parameter_prefix": "recipient_",
            "training_forward": "private_recipient_branch_only_v13",
        },
        "training_runtime": {
            "device": "cuda:0",
            "uses_cuda": True,
            "cuda_device_name": "NVIDIA GeForce RTX 4090",
            "num_workers": 4,
            "prefetch_factor": 2,
            "persistent_workers": True,
            "validation_every": 1,
            "cuda_tf32_requested": True,
            "cudnn_benchmark_requested": True,
        },
        "recipient_train_split_policy": {
            "mode": "standard_train_only",
            "splits": ["train"],
        },
        "checkpoint_selection_policy": {
            "mode": "recipient_priority",
            "protected_minimum_candidate_exact": {
                "amount": 0.7885,
                "time": 0.9840,
                "payment_method_field": 0.9325,
            },
        },
        "recipient_oov_by_split": {
            "train": {"records": 5},
            "val": {"records": 3},
            "test": {"records": 0},
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


def _residual_recipe(
    *,
    source_subject_id: str,
    source_checkpoint_sha256: str,
    full_manifest_sha256: str,
    stage: str = "residual-8e",
    candidate_pilot_subject_id: str | None = None,
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "kind": "receipt_recipient_v14_full_crop_training_recipe_v1",
        "analysis_only": True,
        "production_route_authorized": False,
        "test_opened": False,
        "stage": stage,
        "source_subject_id": source_subject_id,
        "candidate_pilot_subject_id": candidate_pilot_subject_id,
        "source_checkpoint_sha256": source_checkpoint_sha256,
        "full_manifest_sha256": full_manifest_sha256,
        "training_args": {
            "device": "cuda:0",
            "epochs": 8 if stage == "residual-8e" else 60,
            "batch_size": 10,
            "learning_rate": 0.0003,
            "validation_every": 1 if stage == "residual-8e" else 2,
            "seed": 42,
            "num_workers": 4,
            "prefetch_factor": 2,
            "persistent_workers": True,
            "cuda_tf32": True,
            "cudnn_benchmark": True,
        },
    }


def test_residual_pilot_gate_requires_trim_zero_full_fields_and_safe_status(
    tmp_path: Path,
) -> None:
    source_checkpoint = tmp_path / "best.pt"
    source_checkpoint.write_bytes(b"full-crop-best")
    source_sha = hashlib.sha256(source_checkpoint.read_bytes()).hexdigest()
    summary = _residual_summary(
        source_summary=_summary(),
        source_checkpoint=source_checkpoint,
        source_sha256=source_sha,
    )
    source_subject_id = "a" * 64
    full_manifest_sha256 = "b" * 64
    recipe = _residual_recipe(
        source_subject_id=source_subject_id,
        source_checkpoint_sha256=source_sha,
        full_manifest_sha256=full_manifest_sha256,
    )

    decision = evaluate_residual_candidate_pilot(
        summary,
        recipe=recipe,
        source_subject_id=source_subject_id,
        source_checkpoint=source_checkpoint,
        source_checkpoint_sha256=source_sha,
        full_manifest_sha256=full_manifest_sha256,
    )
    assert decision["kind"] == CANDIDATE_PILOT_KIND
    assert decision["passed"] is True
    assert decision["decision"] == CANDIDATE_PILOT_DECISION
    assert decision["production_route_authorized"] is False

    wrong_trim = json.loads(json.dumps(summary))
    wrong_trim["config"]["recipient_value_left_trim"] = 0.30
    with pytest.raises(ValueError, match="config transition|trim-zero"):
        evaluate_residual_candidate_pilot(
            wrong_trim,
            recipe=recipe,
            source_subject_id=source_subject_id,
            source_checkpoint=source_checkpoint,
            source_checkpoint_sha256=source_sha,
            full_manifest_sha256=full_manifest_sha256,
        )
    leaked_test = json.loads(json.dumps(summary))
    leaked_test["field_counts"]["recipient_field"]["test"] = 1
    with pytest.raises(ValueError, match="test count"):
        evaluate_residual_candidate_pilot(
            leaked_test,
            recipe=recipe,
            source_subject_id=source_subject_id,
            source_checkpoint=source_checkpoint,
            source_checkpoint_sha256=source_sha,
            full_manifest_sha256=full_manifest_sha256,
        )
    unsafe_status = json.loads(json.dumps(summary))
    unsafe_status["records"][2]["val_status_non_success_to_success"] = 1
    with pytest.raises(ValueError, match="unsafe status"):
        evaluate_residual_candidate_pilot(
            unsafe_status,
            recipe=recipe,
            source_subject_id=source_subject_id,
            source_checkpoint=source_checkpoint,
            source_checkpoint_sha256=source_sha,
            full_manifest_sha256=full_manifest_sha256,
        )
    changed_recipe = json.loads(json.dumps(recipe))
    changed_recipe["training_args"]["learning_rate"] = 0.0004
    with pytest.raises(ValueError, match="fixed full-crop training recipe"):
        evaluate_residual_candidate_pilot(
            summary,
            recipe=changed_recipe,
            source_subject_id=source_subject_id,
            source_checkpoint=source_checkpoint,
            source_checkpoint_sha256=source_sha,
            full_manifest_sha256=full_manifest_sha256,
        )
    nonce_recipe = json.loads(json.dumps(recipe))
    nonce_recipe["ignored_nonce"] = "must-be-rejected"
    with pytest.raises(ValueError, match="recipe keys changed"):
        evaluate_residual_candidate_pilot(
            summary,
            recipe=nonce_recipe,
            source_subject_id=source_subject_id,
            source_checkpoint=source_checkpoint,
            source_checkpoint_sha256=source_sha,
            full_manifest_sha256=full_manifest_sha256,
        )


def test_residual_pilot_evidence_is_required_content_bound_and_no_onnx(
    tmp_path: Path,
    fake_checkpoints,
) -> None:
    pilot_root, full, source_summary = _build_passed_full_crop_pilot(
        tmp_path, fake_checkpoints
    )
    source_contract_path = tmp_path / "source.contract.json"
    source = attest_full_crop_candidate_source(
        pilot_root=pilot_root,
        output_contract=source_contract_path,
        torch=object(),
    )
    source_best = Path(source["artifacts"]["best_checkpoint"]["path"])
    source_sha = source["artifacts"]["best_checkpoint"]["sha256"]

    candidate_root = tmp_path / "residual-pilot"
    candidate_root.mkdir()
    _, candidate_blind, candidate_blind_contract = _full_and_blind(candidate_root)
    # _full_and_blind created a separate full manifest. Replace the contract
    # with one bound to the authoritative source full manifest.
    candidate_blind.unlink()
    candidate_blind_contract.unlink()
    build_blind_manifest(
        source=full,
        output=candidate_blind,
        contract=candidate_blind_contract,
    )
    training = candidate_root / "training-v14-candidate"
    training.mkdir()
    candidate_summary = _residual_summary(
        source_summary=source_summary,
        source_checkpoint=source_best,
        source_sha256=source_sha,
    )
    _write_json(training / "training_summary.json", candidate_summary)
    _write_json(
        candidate_root / "recipient_v14_training_recipe.json",
        _residual_recipe(
            source_subject_id=source["source_subject_id"],
            source_checkpoint_sha256=source_sha,
            full_manifest_sha256=hashlib.sha256(full.read_bytes()).hexdigest(),
        ),
    )
    best_epoch = candidate_summary["best_checkpoint_epoch"]
    best_row = next(
        row for row in candidate_summary["records"] if row["epoch"] == best_epoch
    )
    fake_checkpoints(
        training / "best.pt",
        {
            "kind": "receipt_unified_field_reader_v13",
            "state_dict": {"recipient_fake": "sealed"},
            "epoch": best_epoch,
            "metrics": best_row,
            "config": candidate_summary["config"],
            "initialization": candidate_summary["initialization"],
            "fine_tune_policy": candidate_summary["fine_tune_policy"],
            "status_text_runtime_policy": candidate_summary[
                "status_text_runtime_policy"
            ],
            "training_runtime": candidate_summary["training_runtime"],
        },
    )
    evidence_path = candidate_root / "recipient_v14_candidate_pilot.json"
    sealed = seal_residual_candidate_pilot(
        candidate_root=candidate_root,
        source_contract_path=source_contract_path,
        full_records=full,
        output_evidence=evidence_path,
        torch=object(),
    )
    verified = verify_residual_candidate_pilot(
        evidence_path=evidence_path,
        source_contract_path=source_contract_path,
        full_records=full,
        torch=object(),
    )
    assert verified == sealed
    assert sealed["passed"] is True
    assert sealed["test_opened"] is False
    assert sealed["onnx_exported"] is False
    assert len(sealed["candidate_pilot_subject_id"]) == 64
    assert sealed["source_subject_id"] == source["source_subject_id"]

    copied_candidate_root = tmp_path / "copied-residual-pilot"
    shutil.copytree(candidate_root, copied_candidate_root, copy_function=os.link)
    copied_evidence_path = copied_candidate_root / "recipient_v14_candidate_pilot.json"
    copied_evidence_path.unlink()
    copied_candidate_blind = (
        copied_candidate_root / "blind-train-val" / "unified_fields.train-val.jsonl"
    )
    copied_candidate_blind_contract = (
        copied_candidate_root / "blind-train-val" / "blind.contract.json"
    )
    copied_candidate_contract_payload = json.loads(
        copied_candidate_blind_contract.read_text(encoding="utf-8")
    )
    copied_candidate_contract_payload["blind_manifest"] = str(
        copied_candidate_blind.resolve()
    )
    copied_candidate_blind_contract.unlink()
    _write_json(copied_candidate_blind_contract, copied_candidate_contract_payload)
    copied_candidate_summary_path = (
        copied_candidate_root / "training-v14-candidate" / "training_summary.json"
    )
    copied_candidate_summary = json.loads(
        copied_candidate_summary_path.read_text(encoding="utf-8")
    )
    copied_candidate_summary["ignored_extension_nonce"] = (
        "must-not-create-a-new-candidate-subject"
    )
    copied_candidate_summary_path.unlink()
    _write_json(copied_candidate_summary_path, copied_candidate_summary)
    fake_checkpoints.rewrite_equivalent(
        candidate_root / "training-v14-candidate" / "best.pt",
        copied_candidate_root / "training-v14-candidate" / "best.pt",
    )
    copied_sealed = seal_residual_candidate_pilot(
        candidate_root=copied_candidate_root,
        source_contract_path=source_contract_path,
        full_records=full,
        output_evidence=copied_evidence_path,
        torch=object(),
    )
    assert (
        copied_sealed["candidate_pilot_subject_id"]
        == sealed["candidate_pilot_subject_id"]
    )
    assert copied_sealed["integrity_sha256"] != sealed["integrity_sha256"]

    (candidate_root / "premature.onnx").write_bytes(b"forbidden")
    with pytest.raises(ValueError, match="ONNX"):
        verify_residual_candidate_pilot(
            evidence_path=evidence_path,
            source_contract_path=source_contract_path,
            full_records=full,
            torch=object(),
        )


def test_candidate_runner_has_mutually_exclusive_source_and_8_to_60_gates() -> None:
    repo = Path(__file__).parents[1]
    runner = (repo / "scripts" / "receipt-ocr-recipient-v14-candidate-4090.ps1").read_text(
        encoding="utf-8"
    )
    wrapper = (
        repo / "scripts" / "receipt-ocr-recipient-full-crop-candidate-source.py"
    ).read_text(encoding="utf-8")
    final_gate = (
        repo / "scripts" / "receipt-ocr-recipient-v14-final-gate-4090.ps1"
    ).read_text(encoding="utf-8")
    assert "FullCropPilotRoot" in runner
    assert "FullCropSourceContract" in runner
    assert "CandidatePilotEvidence" in runner
    assert "SeedCheckpoint cannot be mixed" in runner
    assert "requires passed CandidatePilotEvidence" in runner
    assert "fixed to exactly 60 fresh epochs" in runner
    assert '$recipientValueLeftTrim = 0.0' in runner
    assert '"--recipient-value-left-trim", "$recipientValueLeftTrim"' in runner
    assert '"verify-source"' in runner
    assert '"seal-candidate-pilot"' in runner
    assert '"verify-candidate-pilot"' in runner
    assert "Open-ReadLease" in runner
    assert "[IO.FileMode]::CreateNew" in runner
    assert "CommonApplicationData" in runner
    assert "receipt-v14-full-crop-residual-8e-v1" in runner
    assert "receipt-v14-full-crop-candidate-60e-v1" in runner
    assert "source_subject_id" in runner
    assert "candidate_pilot_subject_id" in runner
    assert "crash and failed training consume" in runner
    assert "Require-FreshNonReparseOutput $OutputRoot" in runner
    assert "production_route_authorized = $false" in runner
    assert "test_evaluated = $false" in runner
    assert "recipient_full_crop_candidate_source import main" in wrapper
    assert "source_guard_artifacts" in final_gate
    assert "source_guard_digest" in final_gate


def test_final_gate_accepts_only_the_bound_full_crop_subject_route(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    full = tmp_path / "full.jsonl"
    full.write_text(
        '{"id":"train","split":"train","slots":{}}\n'
        '{"id":"val","split":"val","slots":{}}\n'
        '{"id":"test","split":"test","slots":{}}\n',
        encoding="utf-8",
    )
    blind = tmp_path / "blind.jsonl"
    blind_contract = tmp_path / "blind.contract.json"
    build_blind_manifest(source=full, output=blind, contract=blind_contract)

    source_checkpoint = tmp_path / "full-crop-best.pt"
    source_checkpoint.write_bytes(b"full-crop-source")
    source_contract_path = tmp_path / "source.contract.json"
    source_contract_path.write_text("{}\n", encoding="utf-8")
    candidate_pilot_path = tmp_path / "candidate-pilot.json"
    candidate_pilot_path.write_text("{}\n", encoding="utf-8")
    pilot_root = tmp_path / "pilot-root"
    pilot_root.mkdir()
    source_subject_id = "a" * 64
    candidate_pilot_subject_id = "b" * 64
    sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()

    source_config = UnifiedReaderConfig(
        architecture_version=13,
        image_height=32,
        image_width=64,
        base_channels=8,
        numeric_hidden_size=16,
        payment_hidden_size=16,
        recipient_hidden_size=16,
        recipient_value_left_trim=0.0,
        recipient_input_height=32,
        recipient_input_width=128,
        recipient_branch_channels=8,
        recipient_open_text_layers=2,
        recipient_open_text_heads=4,
        recipient_open_text_feedforward=64,
        pooled_width=2,
    )
    target_config = UnifiedReaderConfig(
        **{
            **asdict(source_config),
            "recipient_backbone": "residual_positional_transformer_v2",
            "recipient_branch_channels": 16,
            "recipient_hidden_size": 192,
            "recipient_open_text_layers": 4,
            "recipient_open_text_heads": 8,
            "recipient_open_text_feedforward": 1536,
            "recipient_open_text_dropout": 0.10,
        }
    )
    training_summary = tmp_path / "training_summary.json"
    _write_json(
        training_summary,
        {
            "kind": "receipt_unified_field_reader_v13",
            "config": asdict(target_config),
            "initialization": {
                "mode": "parameter_only_recipient_visual_context_reinit",
                "source_config": asdict(source_config),
                "checkpoint_path": str(source_checkpoint),
                "checkpoint_sha256": sha(source_checkpoint),
            },
            "fine_tune_policy": {
                "mode": "recipient_only_v13",
                "trainable_parameter_prefix": "recipient_",
            },
            "training_runtime": {
                "device": "cuda:0",
                "uses_cuda": True,
                "cuda_device_name": "NVIDIA GeForce RTX 4090",
                "num_workers": 4,
                "prefetch_factor": 2,
                "persistent_workers": True,
                "validation_every": 2,
                "cuda_tf32_requested": True,
                "cudnn_benchmark_requested": True,
            },
            "recipient_train_split_policy": {
                "mode": "standard_train_only",
                "splits": ["train"],
            },
            "field_counts": {"recipient_field": {"train": 1, "val": 1, "test": 0}},
            "recipient_oov_by_split": {"test": {"records": 0}},
            "best_checkpoint_epoch": 60,
        },
    )
    recipe_path = tmp_path / "recipient_v14_training_recipe.json"
    _write_json(
        recipe_path,
        _residual_recipe(
            source_subject_id=source_subject_id,
            candidate_pilot_subject_id=candidate_pilot_subject_id,
            source_checkpoint_sha256=sha(source_checkpoint),
            full_manifest_sha256=sha(full),
            stage="candidate-60e",
        ),
    )

    model = tmp_path / "candidate.onnx"
    model.write_bytes(b"sealed-onnx")
    model_contract = model.with_suffix(".contract.json")
    model_labels = model.with_suffix(".labels.json")
    model_contract.write_bytes(b"sealed-contract")
    model_labels.write_bytes(b"sealed-labels")
    checkpoint = tmp_path / "candidate-best.pt"
    checkpoint.write_bytes(b"sealed-candidate-checkpoint")
    val_summary = tmp_path / "val.json"
    by_field = {
        "amount": {"records": 1, "raw_exact_match": 0.80},
        "time": {"records": 1, "raw_exact_match": 0.99},
        "payment_method_field": {"records": 1, "raw_exact_match": 0.94},
        "recipient_field": {"records": 1, "raw_exact_match": 0.91},
        "transfer_status": {
            "records": 1,
            "ctc_records": 1,
            "ctc_raw_exact_match": 0.92,
            "non_success_to_success": 0,
        },
    }
    acceptance = {
        "requested": True,
        "min_amount_exact_match": 0.7885,
        "min_time_exact_match": 0.9840,
        "min_payment_exact_match": 0.9325,
        "min_recipient_exact_match": 0.90,
        "min_status_exact_match": 0.90,
        "max_non_success_to_success": 0,
        "passed": True,
        "failures": [],
    }
    _write_json(
        val_summary,
        {
            "schema_version": 1,
            "kind": "receipt_unified_field_reader_truth_evaluation_v1",
            "model_sha256": sha(model),
            "records_sha256": sha(blind),
            "evaluation_split": "val",
            "providers": ["CUDAExecutionProvider"],
            "status_text_policy": {
                "runtime_policy": "decode_and_normalize_review_only",
                "review_value": "review",
            },
            "acceptance": acceptance,
            "by_field": by_field,
        },
    )
    source_payload = {
        "source_subject_id": source_subject_id,
        "artifacts": {
            "best_checkpoint": {
                "path": str(source_checkpoint),
                "sha256": sha(source_checkpoint),
            }
        },
    }
    candidate_pilot_payload = {
        "source_subject_id": source_subject_id,
        "candidate_pilot_subject_id": candidate_pilot_subject_id,
        "artifacts": {},
    }
    monkeypatch.setattr(
        source_contract,
        "verify_full_crop_candidate_source",
        lambda **kwargs: source_payload,
    )
    monkeypatch.setattr(
        source_contract,
        "verify_residual_candidate_pilot",
        lambda **kwargs: candidate_pilot_payload,
    )
    import transfer_receipt_ai.ocr_unified as ocr_unified

    monkeypatch.setattr(
        ocr_unified,
        "_load_onnx_artifact_details",
        lambda path: (target_config, [], [], {}),
    )
    evidence = tmp_path / "candidate.json"
    candidate_document = {
        "schema_version": 1,
        "kind": "receipt_recipient_v14_blind_candidate_v1",
        "analysis_only": True,
        "production_route_authorized": False,
        "split_policy": {
            "optimizer_supervision": ["train"],
            "checkpoint_selection": ["val"],
            "final_gate_only": ["test"],
            "test_evaluated": False,
            "blind_contract": str(blind_contract),
            "blind_contract_sha256": sha(blind_contract),
        },
        "fixed_floors": {
            "amount": 0.7885,
            "time": 0.9840,
            "payment_method_field": 0.9325,
            "recipient_field": 0.90,
            "visible_transfer_status_cjk_text": 0.90,
        },
        "full_manifest": str(full),
        "full_manifest_sha256": sha(full),
        "blind_manifest": str(blind),
        "blind_manifest_sha256": sha(blind),
        "source_route": {
            "mode": "attested_full_crop_pilot_visual_context_reinit",
            "recipient_value_left_trim": 0.0,
            "source_contract": str(source_contract_path),
            "source_contract_sha256": sha(source_contract_path),
            "source_subject_id": source_subject_id,
            "full_crop_pilot_root": str(pilot_root),
            "source_checkpoint": str(source_checkpoint),
            "source_checkpoint_sha256": sha(source_checkpoint),
            "candidate_pilot_evidence": str(candidate_pilot_path),
            "candidate_pilot_evidence_sha256": sha(candidate_pilot_path),
            "candidate_pilot_subject_id": candidate_pilot_subject_id,
        },
        "candidate": {
            "model": str(model),
            "model_sha256": sha(model),
            "contract": str(model_contract),
            "contract_sha256": sha(model_contract),
            "labels": str(model_labels),
            "labels_sha256": sha(model_labels),
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": sha(checkpoint),
            "architecture_version": 13,
            "backbone": "residual_positional_transformer_v2",
        },
        "training": {
            "summary": str(training_summary),
            "summary_sha256": sha(training_summary),
            "recipe": str(recipe_path),
            "recipe_sha256": sha(recipe_path),
            "best_epoch": 60,
        },
        "val_evaluation": {
            "summary": str(val_summary),
            "summary_sha256": sha(val_summary),
            "amount": 0.80,
            "time": 0.99,
            "payment_method_field": 0.94,
            "recipient_field": 0.91,
            "visible_transfer_status_cjk_text": 0.92,
            "status_non_success_to_success": 0,
        },
    }
    _write_json(evidence, candidate_document)
    inspection = inspect_candidate(evidence, trusted_full_manifest_sha256=sha(full))
    assert inspection["evidence_binding"]["source_route"]["source_subject_id"] == source_subject_id
    assert inspection["source_guard_artifacts"]

    candidate_document["source_route"]["source_subject_id"] = "c" * 64
    _write_json(evidence, candidate_document)
    with pytest.raises(ValueError, match="source subject identity"):
        inspect_candidate(evidence, trusted_full_manifest_sha256=sha(full))


def test_final_gate_mechanically_rejects_unbound_trim_zero_candidate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    full = tmp_path / "full.jsonl"
    full.write_text(
        '{"id":"train","split":"train","slots":{}}\n'
        '{"id":"val","split":"val","slots":{}}\n'
        '{"id":"test","split":"test","slots":{}}\n',
        encoding="utf-8",
    )
    blind = tmp_path / "blind.jsonl"
    blind_contract = tmp_path / "blind.contract.json"
    build_blind_manifest(source=full, output=blind, contract=blind_contract)
    model = tmp_path / "candidate.onnx"
    model.write_bytes(b"sealed-onnx")
    model_contract = model.with_suffix(".contract.json")
    model_labels = model.with_suffix(".labels.json")
    model_contract.write_bytes(b"sealed-contract")
    model_labels.write_bytes(b"sealed-labels")
    checkpoint = tmp_path / "best.pt"
    checkpoint.write_bytes(b"sealed-checkpoint")

    source_config = UnifiedReaderConfig(
        architecture_version=13,
        image_height=32,
        image_width=64,
        base_channels=8,
        numeric_hidden_size=16,
        payment_hidden_size=16,
        recipient_hidden_size=16,
        recipient_value_left_trim=0.0,
        recipient_input_height=32,
        recipient_input_width=128,
        recipient_branch_channels=8,
        recipient_open_text_layers=2,
        recipient_open_text_heads=4,
        recipient_open_text_feedforward=64,
        pooled_width=2,
    )
    target_config = UnifiedReaderConfig(
        **{
            **asdict(source_config),
            "recipient_backbone": "residual_positional_transformer_v2",
            "recipient_branch_channels": 16,
            "recipient_hidden_size": 192,
            "recipient_open_text_layers": 4,
            "recipient_open_text_heads": 8,
            "recipient_open_text_feedforward": 1536,
            "recipient_open_text_dropout": 0.10,
        }
    )
    training_summary = tmp_path / "training_summary.json"
    _write_json(
        training_summary,
        {
            "kind": "receipt_unified_field_reader_v13",
            "config": asdict(target_config),
            "initialization": {
                "mode": "parameter_only_recipient_visual_context_reinit",
                "source_config": asdict(source_config),
            },
            "fine_tune_policy": {
                "mode": "recipient_only_v13",
                "trainable_parameter_prefix": "recipient_",
            },
            "training_runtime": {},
            "recipient_train_split_policy": {
                "mode": "standard_train_only",
                "splits": ["train"],
            },
            "field_counts": {"recipient_field": {"train": 1, "val": 1, "test": 0}},
            "recipient_oov_by_split": {"test": {"records": 0}},
            "best_checkpoint_epoch": 8,
        },
    )
    val_summary = tmp_path / "val.json"
    _write_json(val_summary, {})
    sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    evidence = tmp_path / "candidate.json"
    _write_json(
        evidence,
        {
            "schema_version": 1,
            "kind": "receipt_recipient_v14_blind_candidate_v1",
            "split_policy": {
                "optimizer_supervision": ["train"],
                "checkpoint_selection": ["val"],
                "final_gate_only": ["test"],
                "test_evaluated": False,
                "blind_contract": str(blind_contract),
                "blind_contract_sha256": sha(blind_contract),
            },
            "fixed_floors": {
                "amount": 0.7885,
                "time": 0.9840,
                "payment_method_field": 0.9325,
                "recipient_field": 0.90,
                "visible_transfer_status_cjk_text": 0.90,
            },
            "full_manifest": str(full),
            "full_manifest_sha256": sha(full),
            "blind_manifest": str(blind),
            "blind_manifest_sha256": sha(blind),
            "candidate": {
                "model": str(model),
                "model_sha256": sha(model),
                "contract": str(model_contract),
                "contract_sha256": sha(model_contract),
                "labels": str(model_labels),
                "labels_sha256": sha(model_labels),
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": sha(checkpoint),
                "architecture_version": 13,
                "backbone": "residual_positional_transformer_v2",
            },
            "training": {
                "summary": str(training_summary),
                "summary_sha256": sha(training_summary),
                "best_epoch": 8,
            },
            "val_evaluation": {
                "summary": str(val_summary),
                "summary_sha256": sha(val_summary),
            },
        },
    )
    import transfer_receipt_ai.ocr_unified as ocr_unified

    monkeypatch.setattr(
        ocr_unified,
        "_load_onnx_artifact_details",
        lambda path: (target_config, [], [], {}),
    )
    with pytest.raises(ValueError, match="unbound trim-zero"):
        inspect_candidate(
            evidence,
            trusted_full_manifest_sha256=sha(full),
        )
