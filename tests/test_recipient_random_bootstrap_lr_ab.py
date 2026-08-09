from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from PIL import Image

import transfer_receipt_ai.recipient_random_bootstrap_lr_ab as lr_ab_module
from transfer_receipt_ai.recipient_random_bootstrap import FIXED_TOPOLOGY
from transfer_receipt_ai.recipient_random_bootstrap_lr_ab import (
    BASELINE_LEARNING_RATE,
    CANDIDATE_LEARNING_RATE,
    EXPECTED_031004_CANDIDATE_DENOMINATORS,
    EXPECTED_031004_INPUT_CONTRACT_SHA256,
    EXPECTED_031004_RAW_VAL_COUNTS,
    EXPECTED_031004_RECIPIENT_OBSERVED,
    FIXED_BASELINE_RECIPE,
    FIXED_CANDIDATE_RECIPE,
    PUBLICATION_POLICY,
    SOURCE_RECOVERY_DECISION_KIND,
    _actual_train_argv_evidence,
    _assert_expected_031004_source_identity,
    _atomic_write_json_no_clobber,
    _baseline_learning_rate_proof,
    _checkpoint_metrics_match,
    _recipe_difference,
    _source_decision_descriptor,
    build_lr_ab_decision,
    check_source_preflight,
    prepare_input_contract,
    training_validation_candidate_denominators_v12,
)


def _metric(matches: int, records: int) -> dict[str, object]:
    return {
        "exact_matches": matches,
        "records": records,
        "exact_match": matches / records,
    }


def _record(epoch: int, *, amount_records: int = 1428) -> dict[str, object]:
    return {
        "epoch": epoch,
        "validation_performed": True,
        "val_candidate_text_by_field": {
            "amount": _metric(88, amount_records),
            "time": _metric(3700, 3738),
            "payment_method_field": _metric(5000, 5242),
            "recipient_field": _metric(4467, 6789),
        },
    }


def test_recipe_changes_only_learning_rate() -> None:
    assert BASELINE_LEARNING_RATE == 0.0001
    assert CANDIDATE_LEARNING_RATE == 0.0003
    assert _recipe_difference() == {
        "learning_rate": {
            "baseline": 0.0001,
            "candidate": 0.0003,
        }
    }
    assert set(FIXED_BASELINE_RECIPE) == set(FIXED_CANDIDATE_RECIPE)


def test_baseline_learning_rate_is_proved_by_bound_source_runner(tmp_path: Path) -> None:
    repo = Path(__file__).resolve().parents[1]
    runner = tmp_path / "source-runner.ps1"
    runner.write_text(
        (repo / "scripts" / "receipt-ocr-recipient-random-bootstrap-4090.ps1").read_text(
            encoding="utf-8"
        ),
        encoding="utf-8",
    )
    digest = hashlib.sha256(runner.read_bytes()).hexdigest()
    proof = _baseline_learning_rate_proof(
        {"code_inputs": {"runner": {"path": str(runner), "sha256": digest}}}
    )
    assert proof["learning_rate"] == 0.0001
    assert proof["normalized_train_namespace"]["weight_decay"] == 0.0001
    runner.write_text(runner.read_text(encoding="utf-8").replace("0.0001", "0.0003"), encoding="utf-8")
    with pytest.raises(ValueError, match="changed after input binding"):
        _baseline_learning_rate_proof(
            {"code_inputs": {"runner": {"path": str(runner), "sha256": digest}}}
        )


def test_prepare_contract_is_fresh_and_binds_new_code(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_root = tmp_path / "source"
    source_root.mkdir()
    root_best = source_root / "best.pt"
    root_best.write_bytes(b"root")
    root_descriptor = {
        "path": str(root_best),
        "size_bytes": 4,
        "sha256": hashlib.sha256(b"root").hexdigest(),
        "read_only_required": True,
    }
    closure = {
        "random_root": {"best_checkpoint": root_descriptor},
        "baseline_learning_rate_proof": _baseline_learning_rate_proof(
            {
                "code_inputs": {
                    "runner": {
                        "path": str(
                            Path(__file__).resolve().parents[1]
                            / "scripts"
                            / "receipt-ocr-recipient-random-bootstrap-4090.ps1"
                        ),
                        "sha256": hashlib.sha256(
                            (
                                Path(__file__).resolve().parents[1]
                                / "scripts"
                                / "receipt-ocr-recipient-random-bootstrap-4090.ps1"
                            ).read_bytes()
                        ).hexdigest(),
                    }
                }
            }
        ),
        "failed_lr1e4_pilot": {
            "observed": _observed(best=0.65, epoch4=0.38, epoch8=0.65)
        },
    }
    source_input = {
        "dataset_binding": {"fingerprint": "bound"},
        "blind_manifest": str(source_root / "blind.jsonl"),
        "blind_manifest_sha256": "a" * 64,
        "snapshot_dataset_root": str(source_root / "snapshot"),
        "fixed_topology": FIXED_TOPOLOGY,
        "delivery_floors_unchanged": lr_ab_module.base.DELIVERY_FLOORS,
        "analysis_continuation_gates": {
            "minimum_best_recipient_exact": 0.75,
            "minimum_epoch4_to_8_gain": 0.02,
        },
    }
    source_summary = {key: None for key in lr_ab_module._PERSISTED_RECIPE_KEYS}
    monkeypatch.setattr(
        lr_ab_module,
        "_validate_source_closure",
        lambda *_args, **_kwargs: (closure, source_input, source_summary),
    )
    repo = Path(__file__).resolve().parents[1]
    runner = repo / "scripts" / "receipt-ocr-recipient-random-bootstrap-lr-ab-4090.ps1"
    verifier = repo / "src" / "transfer_receipt_ai" / "recipient_random_bootstrap_lr_ab.py"
    output = tmp_path / "fresh-ab"

    payload = prepare_input_contract(
        source_bootstrap_root=source_root,
        output_root=output,
        runner=runner,
        verifier=verifier,
    )

    assert payload["recipe_difference"] == _recipe_difference()
    assert payload["fresh_start"]["checkpoint"] == root_descriptor
    assert payload["fresh_start"]["failed_pilot_checkpoint_reused"] is False
    assert payload["actual_train_argv_evidence"]["namespace_difference"] == {
        "learning_rate": {"baseline": 0.0001, "candidate": 0.0003}
    }
    assert payload["actual_train_argv_evidence"]["implicit_weight_decay"] == 0.0001
    assert payload["publication_policy"] == PUBLICATION_POLICY
    assert set(payload["code_inputs"]) == {
        "runner",
        "verifier",
        "training_target_parser",
    }
    assert (output / "lr-ab-input.contract.json").is_file()
    with pytest.raises(FileExistsError, match="refusing to reuse"):
        prepare_input_contract(
            source_bootstrap_root=source_root,
            output_root=output,
            runner=runner,
            verifier=verifier,
        )


def test_check_source_preflight_is_read_only_and_binds_canonical_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_root = tmp_path / "source"
    source_decision = source_root / "analysis-decision.recovered.json"
    observed = _observed(best=4467 / 6789, epoch4=2651 / 6789, epoch8=4467 / 6789)
    closure = {
        "source_bootstrap_root": str(source_root),
        "source_decision": {"kind": SOURCE_RECOVERY_DECISION_KIND, "used": True},
        "baseline_learning_rate_proof": _real_baseline_proof(),
        "failed_lr1e4_pilot": {"observed": observed},
        "training_validation_candidate_denominators": dict(
            EXPECTED_031004_CANDIDATE_DENOMINATORS
        ),
    }
    calls: list[tuple[Path, Path | None]] = []

    def validate_source(
        root: Path, *, source_decision: Path | None = None
    ) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
        calls.append((root, source_decision))
        return closure, {}, {}

    monkeypatch.setattr(lr_ab_module, "_validate_source_closure", validate_source)
    repo = Path(__file__).resolve().parents[1]
    runner = repo / "scripts" / "receipt-ocr-recipient-random-bootstrap-lr-ab-4090.ps1"

    result = check_source_preflight(
        source_bootstrap_root=source_root,
        source_decision=source_decision,
        runner=runner,
    )

    assert calls == [(source_root, source_decision)]
    assert result["candidate_denominators"] == EXPECTED_031004_CANDIDATE_DENOMINATORS
    assert result["source_decision"] == closure["source_decision"]
    assert result["actual_train_argv_evidence"]["namespace_difference"] == {
        "learning_rate": {"baseline": 0.0001, "candidate": 0.0003}
    }
    assert not source_root.exists()


def _real_baseline_proof() -> dict[str, object]:
    runner = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "receipt-ocr-recipient-random-bootstrap-4090.ps1"
    )
    return _baseline_learning_rate_proof(
        {
            "code_inputs": {
                "runner": {
                    "path": str(runner),
                    "sha256": hashlib.sha256(runner.read_bytes()).hexdigest(),
                }
            }
        }
    )


def test_actual_train_argv_machine_comparison_allows_only_learning_rate() -> None:
    runner = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "receipt-ocr-recipient-random-bootstrap-lr-ab-4090.ps1"
    )
    evidence = _actual_train_argv_evidence(
        baseline_proof=_real_baseline_proof(), candidate_runner=runner
    )

    assert evidence["namespace_difference"] == {
        "learning_rate": {"baseline": 0.0001, "candidate": 0.0003}
    }
    assert evidence["implicit_weight_decay"] == 0.0001
    assert evidence["baseline_option_names"] == evidence["candidate_option_names"]


def test_actual_train_argv_rejects_injected_weight_decay(tmp_path: Path) -> None:
    real_runner = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "receipt-ocr-recipient-random-bootstrap-lr-ab-4090.ps1"
    )
    source = real_runner.read_text(encoding="utf-8")
    injected = source.replace(
        '    "--learning-rate", "$candidateLearningRate",',
        '    "--learning-rate", "$candidateLearningRate",\n    "--weight-decay", "0.001",',
    )
    assert injected != source
    runner = tmp_path / "candidate.ps1"
    runner.write_text(injected, encoding="utf-8")

    with pytest.raises(ValueError, match="option set differs"):
        _actual_train_argv_evidence(
            baseline_proof=_real_baseline_proof(), candidate_runner=runner
        )


def test_031004_identity_is_fixed_by_hash_denominators_and_metrics() -> None:
    assert EXPECTED_031004_INPUT_CONTRACT_SHA256 == (
        "7f6f2b07b33a5707ea376739e6853629c806675b22b320a9898c45f5bede91fc"
    )
    _assert_expected_031004_source_identity(
        input_contract_sha256=EXPECTED_031004_INPUT_CONTRACT_SHA256,
        candidate_denominators=EXPECTED_031004_CANDIDATE_DENOMINATORS,
        observed=EXPECTED_031004_RECIPIENT_OBSERVED,
    )
    with pytest.raises(ValueError, match="hash-bound 031004"):
        _assert_expected_031004_source_identity(
            input_contract_sha256=(
                "7f6f2b07b3335707ea376739e6853629c806675b22b320a9898c45f5bede91fc"
            ),
            candidate_denominators=EXPECTED_031004_CANDIDATE_DENOMINATORS,
            observed=EXPECTED_031004_RECIPIENT_OBSERVED,
        )
    with pytest.raises(ValueError, match="hash-bound 031004"):
        _assert_expected_031004_source_identity(
            input_contract_sha256="0" * 64,
            candidate_denominators=EXPECTED_031004_CANDIDATE_DENOMINATORS,
            observed=EXPECTED_031004_RECIPIENT_OBSERVED,
        )
    raw_amount = dict(EXPECTED_031004_CANDIDATE_DENOMINATORS)
    raw_amount["amount"] = 1606
    with pytest.raises(ValueError, match="candidate denominators"):
        _assert_expected_031004_source_identity(
            input_contract_sha256=EXPECTED_031004_INPUT_CONTRACT_SHA256,
            candidate_denominators=raw_amount,
            observed=EXPECTED_031004_RECIPIENT_OBSERVED,
        )


def test_recovery_decision_is_mandatory_and_strictly_bound(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_root = tmp_path / "source"
    source_root.mkdir()
    input_contract = source_root / "bootstrap-input.contract.json"
    input_contract.write_text(
        json.dumps(
            {
                "dataset_binding": {
                    "field_counts": {
                        field: {"val": count}
                        for field, count in EXPECTED_031004_RAW_VAL_COUNTS.items()
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="requires analysis-decision.recovered.json"):
        _source_decision_descriptor(
            decision_path=None,
            source_root=source_root,
            input_contract=input_contract,
            candidate_denominators=EXPECTED_031004_CANDIDATE_DENOMINATORS,
            observed=EXPECTED_031004_RECIPIENT_OBSERVED,
        )

    recovery = source_root / "analysis-decision.recovered.json"
    payload = {
        "kind": SOURCE_RECOVERY_DECISION_KIND,
        "analysis_only": True,
        "production_route_authorized": False,
        "onnx_delivery_authorized": False,
        "continuation_16_epoch_authorized": False,
        "authorized_16_epoch_warmstart_checkpoint": None,
        "input_contract_sha256": EXPECTED_031004_INPUT_CONTRACT_SHA256,
        "recipient_observed": EXPECTED_031004_RECIPIENT_OBSERVED,
        "candidate_denominator_evidence": {
            "policy": "v12_candidate_reference_eligibility_v1",
            "candidate_val_denominators": EXPECTED_031004_CANDIDATE_DENOMINATORS,
            "raw_val_field_counts": EXPECTED_031004_RAW_VAL_COUNTS,
        },
    }
    recovery.write_text(json.dumps(payload), encoding="utf-8")
    recovery.chmod(0o444)
    real_sha256 = lr_ab_module.base._sha256

    def bound_sha256(path: Path) -> str:
        if Path(path) == input_contract:
            return EXPECTED_031004_INPUT_CONTRACT_SHA256
        return real_sha256(Path(path))

    monkeypatch.setattr(lr_ab_module.base, "_sha256", bound_sha256)
    descriptor = _source_decision_descriptor(
        decision_path=recovery,
        source_root=source_root,
        input_contract=input_contract,
        candidate_denominators=EXPECTED_031004_CANDIDATE_DENOMINATORS,
        observed=EXPECTED_031004_RECIPIENT_OBSERVED,
    )
    assert descriptor["used"] is True
    assert descriptor["kind"] == SOURCE_RECOVERY_DECISION_KIND
    assert descriptor["continuation_16_epoch_authorized"] is False

    recovery.chmod(0o644)
    payload["continuation_16_epoch_authorized"] = True
    recovery.write_text(json.dumps(payload), encoding="utf-8")
    recovery.chmod(0o444)
    with pytest.raises(ValueError, match="not compatible 031004"):
        _source_decision_descriptor(
            decision_path=recovery,
            source_root=source_root,
            input_contract=input_contract,
            candidate_denominators=EXPECTED_031004_CANDIDATE_DENOMINATORS,
            observed=EXPECTED_031004_RECIPIENT_OBSERVED,
        )


def test_no_clobber_writer_proves_closing_identity_and_preserves_existing(
    tmp_path: Path,
) -> None:
    output = tmp_path / "evidence.json"
    identity = _atomic_write_json_no_clobber(output, {"analysis_only": True})
    original = output.read_bytes()
    assert identity["policy"] == PUBLICATION_POLICY
    assert identity["sha256"] == hashlib.sha256(original).hexdigest()
    assert identity["read_only"] is True
    assert not list(tmp_path.glob(".*.tmp"))
    with pytest.raises(FileExistsError):
        _atomic_write_json_no_clobber(output, {"analysis_only": False})
    assert output.read_bytes() == original


def test_windows_publication_handle_shares_delete_but_not_write() -> None:
    mode = lr_ab_module._WINDOWS_PUBLICATION_SHARE_MODE
    assert mode & lr_ab_module._WINDOWS_FILE_SHARE_READ
    assert mode & lr_ab_module._WINDOWS_FILE_SHARE_DELETE
    assert not mode & lr_ab_module._WINDOWS_FILE_SHARE_WRITE


def test_no_clobber_writer_loses_race_without_touching_competitor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "evidence.json"

    def losing_link(_source: Path, target: Path, **_kwargs: object) -> None:
        Path(target).write_bytes(b"competitor")
        raise FileExistsError("simulated competing publisher")

    monkeypatch.setattr(lr_ab_module.os, "link", losing_link)
    with pytest.raises(FileExistsError, match="simulated competing publisher"):
        _atomic_write_json_no_clobber(output, {"analysis_only": True})
    assert output.read_bytes() == b"competitor"
    assert not list(tmp_path.glob(".*.tmp"))


def test_checkpoint_metric_uses_training_candidate_denominator_not_raw_slot_count() -> None:
    record = _record(8)
    _checkpoint_metrics_match(
        {"metrics": record},
        record,
        expected_candidate_records={
            "amount": 1428,
            "time": 3738,
            "payment_method_field": 5242,
            "recipient_field": 6789,
        },
        description="real denominator fixture",
    )
    with pytest.raises(ValueError, match="inconsistent candidate metric for amount"):
        _checkpoint_metrics_match(
            {"metrics": record},
            record,
            expected_candidate_records={
                "amount": 1606,
                "time": 3738,
                "payment_method_field": 5242,
                "recipient_field": 6789,
            },
            description="raw-slot denominator fixture",
        )


def _slot(image: str, text: str, **extra: object) -> dict[str, object]:
    return {"image": image, "text": text, **extra}


def test_candidate_denominators_reconstruct_v12_amount_eligibility(tmp_path: Path) -> None:
    dataset = tmp_path / "snapshot"
    dataset.mkdir()
    rows: list[dict[str, object]] = []
    slot_rows = [
        {
            "amount": _slot("valid-amount.png", "100.00", visible_text="¥100.00"),
        },
        {
            # A non-null raw amount slot is deliberately not a candidate
            # metric record when its visible display target cannot be parsed.
            "amount": _slot("invalid-amount.png", "200.00", visible_text="金额未知"),
        },
        {
            "time": _slot("time.png", "12:06", visible_text="12:06"),
        },
        {
            "payment_method_field": _slot("payment.png", "银行卡(1234)"),
        },
        {
            "recipient_field": _slot(
                "recipient.png",
                "商户甲",
                recipient_visible_text="收款方 商户甲",
                recipient_value="商户甲",
                recipient_label="收款方",
                recipient_quality_policy="anchored_value_right_crop_v1",
            ),
        },
    ]
    for index, slots in enumerate(slot_rows, start=1):
        image_name = str(next(iter(slots.values()))["image"])
        Image.new("RGB", (8, 4), (index * 20, 30, 40)).save(dataset / image_name)
        rows.append(
            {
                "schema_version": 1,
                "id": f"val-{index}",
                "group_id": f"receipt:val-{index}",
                "split": "val",
                "slots": slots,
            }
        )
    manifest = tmp_path / "blind.jsonl"
    manifest.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )

    counts = training_validation_candidate_denominators_v12(
        blind_manifest=manifest,
        snapshot_dataset_root=dataset,
        config_value=FIXED_TOPOLOGY,
    )

    assert counts == {
        "amount": 1,
        "time": 1,
        "payment_method_field": 1,
        "recipient_field": 1,
    }


def _observed(*, best: float, epoch4: float, epoch8: float) -> dict[str, object]:
    return {
        "best_exact": best,
        "best_epochs": [8],
        "selected_best_epoch": 8,
        "epoch4_exact": epoch4,
        "epoch8_exact": epoch8,
        "epoch4_to_8_gain": epoch8 - epoch4,
        "by_epoch": {},
    }


def test_lr_ab_pass_authorizes_only_separate_analysis() -> None:
    decision = build_lr_ab_decision(
        source_observed=_observed(best=0.657976, epoch4=0.390485, epoch8=0.657976),
        candidate_observed=_observed(best=0.82, epoch4=0.70, epoch8=0.82),
    )

    assert decision["continuation_16_epoch_authorized"] is True
    assert decision["candidate_best_delta"] == pytest.approx(0.162024)
    assert decision["production_route_authorized"] is False
    assert decision["onnx_delivery_authorized"] is False
    assert decision["delivery_gate_evaluated"] is False
    assert decision["financial_delivery_checkpoint_eligible"] is False


def test_lr_ab_failure_stops_without_lowering_gates() -> None:
    decision = build_lr_ab_decision(
        source_observed=_observed(best=0.657976, epoch4=0.390485, epoch8=0.657976),
        candidate_observed=_observed(best=0.74, epoch4=0.60, epoch8=0.74),
    )
    assert decision["continuation_16_epoch_authorized"] is False
    assert decision["analysis_continuation_gates"] == {
        "minimum_best_recipient_exact": 0.75,
        "minimum_epoch4_to_8_gain": 0.02,
    }
    assert decision["decision"] == "analysis_only_stop_lr_rescue_failed_fixed_gates"


@pytest.mark.parametrize(
    "source",
    [
        _observed(best=0.75, epoch4=0.50, epoch8=0.75),
        _observed(best=0.65, epoch4=0.64, epoch8=0.65),
    ],
)
def test_lr_ab_rejects_source_without_exact_rescue_precondition(
    source: dict[str, object],
) -> None:
    with pytest.raises(ValueError, match="source"):
        build_lr_ab_decision(
            source_observed=source,
            candidate_observed=_observed(best=0.82, epoch4=0.70, epoch8=0.82),
        )


def test_powershell_runner_is_fixed_fresh_lr_only_ab() -> None:
    source = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "receipt-ocr-recipient-random-bootstrap-lr-ab-4090.ps1"
    ).read_text(encoding="utf-8")
    assert "$baselineLearningRate = 0.0001" in source
    assert "$candidateLearningRate = 0.0003" in source
    assert '"--learning-rate", "$candidateLearningRate"' in source
    assert '"--init-checkpoint", $rootCheckpoint' in source
    assert '"--init-checkpoint-mode", "strict"' in source
    assert '"--recipient-only-fine-tune"' in source
    assert '"--recipient-train-splits", "train"' in source
    assert '"--validation-every", "1"' in source
    assert '"--epochs", "$pilotEpochs"' in source
    assert '"--seed", "$seed"' in source
    assert "strict-recipient-lr3e4-8e" in source
    assert "strict-recipient-warmstart-8e\\best.pt" not in source
    assert "random-root-1e\\best.pt" in source
    assert "analysis-decision.recovered.json" in source
    assert '"--source-decision", $SourceDecision' in source
    assert "--onnx-output" not in source
    assert source.count("Seal-ReadOnlyEvidence") >= 4
    assert "DELIVERY=NOT AUTHORIZED" in source
    source_decision_check = source.index('Require-File $SourceDecision "031004 recovery decision"')
    check_only = source.index("if ($CheckOnly)")
    source_validation = source.index('"check-source"', check_only)
    preflight_passed = source.index("preflight=passed", source_validation)
    assert source_decision_check < check_only < source_validation < preflight_passed
