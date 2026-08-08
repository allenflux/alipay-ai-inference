from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from transfer_receipt_ai.recipient_blind_manifest import build_blind_manifest
from transfer_receipt_ai.recipient_random_bootstrap import (
    CONTINUATION_EPOCH4_TO_8_GAIN_FLOOR,
    CONTINUATION_RECIPIENT_FLOOR,
    DELIVERY_FLOORS,
    FIXED_TOPOLOGY,
    INPUT_KIND,
    _assert_checkpoint_metrics_match_summary,
    build_analysis_decision,
    build_input_contract,
)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _row(*, record_id: str, split: str, image: str, crop: bytes) -> dict[str, object]:
    common_slot = {
        "image": image,
        "crop_sha256": _sha256_bytes(crop),
    }
    return {
        "schema_version": 1,
        "id": record_id,
        "group_id": "receipt:" + record_id,
        "split": split,
        "slot_order": [
            "amount",
            "time",
            "transfer_status",
            "payment_method_field",
            "recipient_field",
        ],
        "slots": {
            "amount": {**common_slot, "text": "100.00"},
            "time": {**common_slot, "text": "12:06"},
            "transfer_status": {**common_slot, "class_name": "success"},
            "payment_method_field": {**common_slot, "text": "银行卡(1234)"},
            "recipient_field": {**common_slot, "text": "收款方甲", "visible_text": "收款方 收款方甲"},
        },
    }


def _write_contract_fixture(
    tmp_path: Path, *, train_image: str = "train.bin"
) -> dict[str, Path]:
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    train_crop = b"train-crop"
    val_crop = b"val-crop"
    (dataset / "train.bin").write_bytes(train_crop)
    (dataset / "val.bin").write_bytes(val_crop)
    # Deliberately do not materialize the test crop.  Input binding must open
    # only the physically separated train/val manifest.
    rows = [
        _row(record_id="train-1", split="train", image=train_image, crop=train_crop),
        _row(record_id="val-1", split="val", image="val.bin", crop=val_crop),
        _row(record_id="test-1", split="test", image="does-not-exist.bin", crop=b"test"),
    ]
    source = tmp_path / "unified_fields.jsonl"
    source.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    blind = tmp_path / "blind" / "unified_fields.train-val.jsonl"
    blind_contract = tmp_path / "blind" / "blind.contract.json"
    build_blind_manifest(source=source, output=blind, contract=blind_contract)
    code_paths: dict[str, Path] = {}
    for name in ("runner", "trainer", "blind_builder", "verifier"):
        path = tmp_path / (name + ".txt")
        path.write_text(name, encoding="utf-8")
        code_paths[name] = path
    return {
        "dataset": dataset,
        "source": source,
        "blind": blind,
        "blind_contract": blind_contract,
        **code_paths,
    }


def test_input_contract_binds_only_blind_train_val_crops(tmp_path: Path) -> None:
    paths = _write_contract_fixture(tmp_path)
    output = tmp_path / "bootstrap-input.contract.json"
    snapshot = tmp_path / "snapshot"
    payload = build_input_contract(
        source_manifest=paths["source"],
        blind_manifest=paths["blind"],
        blind_contract=paths["blind_contract"],
        dataset_root=paths["dataset"],
        snapshot_root=snapshot,
        output=output,
        runner=paths["runner"],
        trainer=paths["trainer"],
        blind_builder=paths["blind_builder"],
        verifier=paths["verifier"],
    )

    assert output.is_file()
    assert payload["kind"] == INPUT_KIND
    assert payload["dataset_binding"]["split_counts"] == {"train": 1, "val": 1}
    assert payload["dataset_binding"]["crop_reference_count"] == 10
    assert payload["dataset_binding"]["field_counts"]["recipient_field"] == {"train": 1, "val": 1}
    assert (snapshot / "train.bin").read_bytes() == b"train-crop"
    assert (snapshot / "val.bin").read_bytes() == b"val-crop"
    assert not (snapshot / "does-not-exist.bin").exists()
    assert payload["test_rows_physically_present_in_training_manifest"] is False
    assert payload["test_labels_used_by_training"] is False
    assert payload["fixed_topology"] == FIXED_TOPOLOGY
    assert payload["delivery_floors_unchanged"] == DELIVERY_FLOORS
    assert payload["production_route_authorized"] is False


def test_input_contract_rejects_crop_hash_mismatch(tmp_path: Path) -> None:
    paths = _write_contract_fixture(tmp_path)
    (paths["dataset"] / "train.bin").write_bytes(b"changed")
    with pytest.raises(ValueError, match="crop SHA-256 mismatch"):
        build_input_contract(
            source_manifest=paths["source"],
            blind_manifest=paths["blind"],
            blind_contract=paths["blind_contract"],
            dataset_root=paths["dataset"],
            snapshot_root=tmp_path / "snapshot",
            output=tmp_path / "input.json",
            runner=paths["runner"],
            trainer=paths["trainer"],
            blind_builder=paths["blind_builder"],
            verifier=paths["verifier"],
        )


def test_input_contract_rejects_reparse_crop(tmp_path: Path) -> None:
    paths = _write_contract_fixture(tmp_path)
    real = paths["dataset"] / "real.bin"
    real.write_bytes(b"train-crop")
    crop = paths["dataset"] / "train.bin"
    crop.unlink()
    try:
        crop.symlink_to(real.name)
    except (NotImplementedError, OSError):
        pytest.skip("symlinks are unavailable")
    with pytest.raises(ValueError, match="reparse point"):
        build_input_contract(
            source_manifest=paths["source"],
            blind_manifest=paths["blind"],
            blind_contract=paths["blind_contract"],
            dataset_root=paths["dataset"],
            snapshot_root=tmp_path / "snapshot",
            output=tmp_path / "input.json",
            runner=paths["runner"],
            trainer=paths["trainer"],
            blind_builder=paths["blind_builder"],
            verifier=paths["verifier"],
        )


def test_input_contract_rejects_parent_alias_that_would_escape_snapshot(tmp_path: Path) -> None:
    paths = _write_contract_fixture(tmp_path, train_image="../dataset/train.bin")
    with pytest.raises(ValueError, match="normalized and relative"):
        build_input_contract(
            source_manifest=paths["source"],
            blind_manifest=paths["blind"],
            blind_contract=paths["blind_contract"],
            dataset_root=paths["dataset"],
            snapshot_root=tmp_path / "snapshot",
            output=tmp_path / "input.json",
            runner=paths["runner"],
            trainer=paths["trainer"],
            blind_builder=paths["blind_builder"],
            verifier=paths["verifier"],
        )


def _metric(rate: float, *, records: int = 100) -> dict[str, object]:
    return {
        "exact_matches": round(rate * records),
        "records": records,
        "exact_match": rate,
    }


def _record(epoch: int, recipient: float) -> dict[str, object]:
    return {
        "epoch": epoch,
        "validation_performed": True,
        "val_candidate_text_by_field": {
            "amount": _metric(0.10),
            "time": _metric(0.20),
            "payment_method_field": _metric(0.30),
            "recipient_field": _metric(recipient),
        },
    }


def _runtime(*, recipient_only: bool) -> dict[str, object]:
    return {
        "cuda_device_name": "NVIDIA GeForce RTX 4090",
        "cuda_tf32_requested": True,
        "cudnn_benchmark_requested": True,
        "validation_every": 1,
        "recipient_only_private_branch_training": recipient_only,
    }


def _summary(*, recipient_values: list[float], recipient_only: bool) -> dict[str, object]:
    best_epoch = max(range(1, len(recipient_values) + 1), key=lambda epoch: recipient_values[epoch - 1])
    return {
        "schema_version": 1,
        "kind": "receipt_unified_field_reader_v12",
        "config": dict(FIXED_TOPOLOGY),
        "field_counts": {
            field: {"train": 10, "val": 100, "test": 0}
            for field in ("amount", "time", "payment_method_field", "recipient_field")
        },
        "recipient_train_split_policy": {"mode": "standard_train_only", "splits": ["train"]},
        "checkpoint_selection_policy": {
            "mode": "balanced",
            "protected_minimum_candidate_exact": {},
        },
        "fine_tune_policy": (
            {
                "mode": "recipient_only_v12",
                "trainable_parameter_prefix": "recipient_",
                "training_forward": "private_recipient_branch_only_v12",
                "open_text_legacy_recipient_unfrozen": False,
            }
            if recipient_only
            else {"mode": "all_parameters"}
        ),
        "training_runtime": _runtime(recipient_only=recipient_only),
        "initialization": (
            {"mode": "parameter_only", "optimizer_restored": False, "epoch_reset": True}
            if recipient_only
            else {"mode": "random", "optimizer_restored": False, "epoch_reset": True}
        ),
        "best_checkpoint_epoch": best_epoch,
        "records": [_record(epoch, value) for epoch, value in enumerate(recipient_values, start=1)],
    }


def _expected_field_counts() -> dict[str, dict[str, int]]:
    return {
        field: {"train": 10, "val": 100}
        for field in ("amount", "time", "payment_method_field", "recipient_field")
    }


def test_continuation_gate_is_not_a_delivery_floor_or_authorization() -> None:
    root = _summary(recipient_values=[0.01], recipient_only=False)
    pilot = _summary(
        recipient_values=[0.60, 0.65, 0.70, 0.75, 0.76, 0.77, 0.78, 0.79],
        recipient_only=True,
    )
    decision = build_analysis_decision(
        root_summary=root,
        pilot_summary=pilot,
        expected_field_counts=_expected_field_counts(),
    )

    assert CONTINUATION_RECIPIENT_FLOOR == 0.75
    assert CONTINUATION_EPOCH4_TO_8_GAIN_FLOOR == 0.02
    assert DELIVERY_FLOORS["recipient_field"] == 0.90
    assert decision["continuation_16_epoch_authorized"] is True
    assert decision["recipient_delivery_target_reached"] is False
    assert decision["production_route_authorized"] is False
    assert decision["onnx_delivery_authorized"] is False
    assert decision["delivery_gate_evaluated"] is False
    assert decision["nonrecipient_metrics_authoritative_for_delivery"] is False


def test_continuation_requires_epoch4_to_8_gain_even_when_best_exceeds_75() -> None:
    root = _summary(recipient_values=[0.01], recipient_only=False)
    pilot = _summary(
        recipient_values=[0.70, 0.76, 0.80, 0.81, 0.81, 0.81, 0.81, 0.82],
        recipient_only=True,
    )
    decision = build_analysis_decision(
        root_summary=root,
        pilot_summary=pilot,
        expected_field_counts=_expected_field_counts(),
    )
    assert decision["recipient_observed"]["best_exact"] == 0.82
    assert decision["recipient_observed"]["epoch4_to_8_gain"] == pytest.approx(0.01)
    assert decision["continuation_16_epoch_authorized"] is False


def test_even_90_percent_recipient_never_authorizes_delivery() -> None:
    root = _summary(recipient_values=[0.01], recipient_only=False)
    pilot = _summary(
        recipient_values=[0.70, 0.75, 0.80, 0.86, 0.88, 0.90, 0.92, 0.94],
        recipient_only=True,
    )
    decision = build_analysis_decision(
        root_summary=root,
        pilot_summary=pilot,
        expected_field_counts=_expected_field_counts(),
    )
    assert decision["recipient_delivery_target_reached"] is True
    assert decision["continuation_16_epoch_authorized"] is True
    assert decision["delivery_gate_evaluated"] is False
    assert decision["financial_delivery_checkpoint_eligible"] is False
    assert decision["production_route_authorized"] is False
    assert decision["onnx_delivery_authorized"] is False


def test_decision_rejects_changed_frozen_financial_metric() -> None:
    root = _summary(recipient_values=[0.01], recipient_only=False)
    pilot = _summary(recipient_values=[0.60] * 8, recipient_only=True)
    pilot["records"][5]["val_candidate_text_by_field"]["amount"] = _metric(0.11)
    with pytest.raises(ValueError, match="frozen random-root amount metric changed"):
        build_analysis_decision(
            root_summary=root,
            pilot_summary=pilot,
            expected_field_counts=_expected_field_counts(),
        )


def test_decision_rejects_high_score_on_partial_validation_denominator() -> None:
    root = _summary(recipient_values=[0.01], recipient_only=False)
    pilot = _summary(
        recipient_values=[0.25, 0.50, 0.50, 0.75, 0.75, 0.75, 0.75, 1.00],
        recipient_only=True,
    )
    counts = _expected_field_counts()
    counts["recipient_field"]["val"] = 1000
    with pytest.raises(ValueError, match="field counts are not bound|inconsistent candidate metric"):
        build_analysis_decision(
            root_summary=root,
            pilot_summary=pilot,
            expected_field_counts=counts,
        )


def test_checkpoint_embedded_metrics_must_match_summary() -> None:
    embedded = _record(8, 0.80)
    summary = _record(8, 0.90)
    with pytest.raises(ValueError, match="embedded recipient_field metric differs"):
        _assert_checkpoint_metrics_match_summary(
            {"metrics": embedded},
            summary,
            expected_field_counts=_expected_field_counts(),
            description="pilot last",
        )


def test_powershell_launcher_is_fixed_random_root_then_strict_warmstart() -> None:
    script = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "receipt-ocr-recipient-random-bootstrap-4090.ps1"
    ).read_text(encoding="utf-8")
    assert '"--device", "cuda:0"' in script
    assert '"--recipient-input-width", "1536"' in script
    assert '"--recipient-open-text-layers", "2"' in script
    assert '"--validation-every", "1"' in script
    assert '"--recipient-train-splits", "train"' in script
    assert '"--init-checkpoint-mode", "strict"' in script
    assert '"--recipient-only-fine-tune"' in script
    assert '"--snapshot-root", $snapshotRoot' in script
    assert '"--dataset-root", $snapshotRoot' in script
    assert script.count("Seal-ReadOnlyEvidence") >= 7
    assert "$rootArgs = $commonTrainArgs" in script
    assert "--init-checkpoint" not in script.split("$rootArgs =", 1)[1].split("Invoke-Python $rootArgs", 1)[0]
    assert "--onnx-output" not in script
    assert "$recipientDeliveryFloor = 0.90" in script
    assert "$recipientContinuationFloor = 0.75" in script
    assert "DELIVERY=NOT AUTHORIZED" in script
