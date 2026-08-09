from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import pytest

from transfer_receipt_ai import recipient_random_bootstrap as bootstrap
from transfer_receipt_ai import recipient_random_bootstrap_recovery as recovery
from transfer_receipt_ai import ocr_unified as trainer
from transfer_receipt_ai.recipient_random_bootstrap_recovery import (
    _FrozenEvidence,
    _assert_checkpoint_metrics_match_summary,
    _atomic_write_json_no_clobber,
    _candidate_metric,
    build_analysis_decision,
    rebuild_candidate_denominators,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _slot(text: str, *, visible: str | None = None, **extra: object) -> dict[str, object]:
    value: dict[str, object] = {"image": "unused.png", "text": text, **extra}
    if visible is not None:
        value["visible_text"] = visible
    return value


def _row(record_id: str, split: str, *, amount_visible: str) -> dict[str, object]:
    return {
        "id": record_id,
        "split": split,
        "slots": {
            "amount": _slot("12.34", visible=amount_visible),
            "time": _slot("12:34", visible="12:34"),
            "payment_method_field": _slot("银行卡(1234)"),
            "recipient_field": _slot("收款方 张三", semantic_value="张三"),
        },
    }


def _denominator_contract(tmp_path: Path) -> tuple[dict[str, object], Path]:
    manifest = tmp_path / "blind.jsonl"
    rows = [
        _row("train-1", "train", amount_visible="1.00"),
        _row("val-valid", "val", amount_visible="¥1,234.56"),
        _row("val-invalid", "val", amount_visible="金额不明"),
    ]
    manifest.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    manifest.chmod(0o444)
    return (
        {
            "blind_manifest": str(manifest),
            "blind_manifest_sha256": _sha256(manifest),
            "dataset_binding": {
                "split_counts": {"train": 1, "val": 2},
                "field_counts": {
                    field: {"train": 1, "val": 2}
                    for field in ("amount", "time", "payment_method_field", "recipient_field")
                },
            },
        },
        manifest,
    )


def test_rebuilds_v12_candidate_denominators_and_record_id_bindings(tmp_path: Path) -> None:
    contract, _ = _denominator_contract(tmp_path)
    evidence = rebuild_candidate_denominators(
        contract=contract, summary_config=dict(bootstrap.FIXED_TOPOLOGY)
    )

    assert evidence["raw_val_field_counts"] == {
        "amount": 2,
        "time": 2,
        "payment_method_field": 2,
        "recipient_field": 2,
    }
    assert evidence["candidate_val_denominators"] == {
        "amount": 1,
        "time": 2,
        "payment_method_field": 2,
        "recipient_field": 2,
    }
    assert evidence["excluded_from_candidate_metric"]["amount"] == 1
    assert set(evidence["eligible_record_ids_sha256"]) == {
        "amount", "time", "payment_method_field", "recipient_field"
    }


def test_rebuild_matches_v12_trainer_string_presence_rules(tmp_path: Path) -> None:
    contract, manifest = _denominator_contract(tmp_path)
    rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines()]
    rows[1]["slots"]["time"]["visible_text"] = ""
    rows[1]["slots"]["time"]["text"] = ""
    rows[1]["slots"]["payment_method_field"]["text"] = ""
    rows[1]["slots"]["recipient_field"].pop("text")
    rows[1]["slots"]["recipient_field"]["semantic_value"] = "张三"
    manifest.chmod(0o644)
    manifest.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    manifest.chmod(0o444)
    contract["blind_manifest_sha256"] = _sha256(manifest)

    evidence = rebuild_candidate_denominators(
        contract=contract, summary_config=dict(bootstrap.FIXED_TOPOLOGY)
    )
    # Trainer _slot_text counts any str, including an empty one.  V12 recipient
    # validation does not use v10's semantic-value fallback when text is absent.
    assert evidence["candidate_val_denominators"] == {
        "amount": 1,
        "time": 2,
        "payment_method_field": 2,
        "recipient_field": 1,
    }


def test_candidate_presence_mirrors_v12_training_validation_branches() -> None:
    config = trainer.UnifiedReaderConfig(**bootstrap.FIXED_TOPOLOGY)
    rows = [
        _row("valid", "val", amount_visible="¥1,234.56"),
        _row("invalid-amount", "val", amount_visible="金额不明"),
        _row("empty-time-visible", "val", amount_visible="1.00"),
        _row("semantic-only-recipient", "val", amount_visible="1.00"),
    ]
    rows[2]["slots"]["time"]["visible_text"] = ""
    rows[3]["slots"]["recipient_field"].pop("text")
    rows[3]["slots"]["recipient_field"]["semantic_value"] = "张三"

    def trainer_presence(row: dict[str, object], field: str) -> bool:
        if field in {"amount", "time"}:
            expected = trainer._ctc_slot_text(row, field, config=config)
        elif field == "payment_method_field":
            expected = trainer._slot_text(row, field)
        elif field == "recipient_field":
            expected = trainer._recipient_expected_value(row, config=config)
        else:  # pragma: no cover - the fixed candidate field set is exhaustive.
            raise AssertionError(field)
        return expected is not None

    for row in rows:
        for field in recovery.CANDIDATE_FIELDS:
            assert recovery._candidate_reference_present(row, field) is trainer_presence(
                row, field
            )


def _metric(matches: int, records: int, *, rate: float | None = None) -> dict[str, object]:
    return {
        "exact_matches": matches,
        "records": records,
        "exact_match": matches / records if rate is None else rate,
    }


def _record(epoch: int, recipient_matches: int) -> dict[str, object]:
    return {
        "epoch": epoch,
        "validation_performed": True,
        "val_candidate_text_by_field": {
            "amount": _metric(88, 1428),
            "time": _metric(100, 3738),
            "payment_method_field": _metric(200, 5242),
            "recipient_field": _metric(recipient_matches, 6789),
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


def _summary(*, recipient_matches: list[int], recipient_only: bool) -> dict[str, object]:
    best_epoch = max(
        range(1, len(recipient_matches) + 1),
        key=lambda epoch: recipient_matches[epoch - 1],
    )
    return {
        "schema_version": 1,
        "kind": bootstrap.CHECKPOINT_KIND,
        "config": dict(bootstrap.FIXED_TOPOLOGY),
        "field_counts": {
            "amount": {"train": 10, "val": 1606, "test": 0},
            "time": {"train": 10, "val": 3738, "test": 0},
            "payment_method_field": {"train": 10, "val": 5242, "test": 0},
            "recipient_field": {"train": 10, "val": 6789, "test": 0},
        },
        "recipient_train_split_policy": {"mode": "standard_train_only", "splits": ["train"]},
        "checkpoint_selection_policy": {
            "mode": "balanced", "protected_minimum_candidate_exact": {}
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
        "records": [
            _record(epoch, matches)
            for epoch, matches in enumerate(recipient_matches, start=1)
        ],
    }


def _raw_counts() -> dict[str, dict[str, int]]:
    return {
        "amount": {"train": 10, "val": 1606},
        "time": {"train": 10, "val": 3738},
        "payment_method_field": {"train": 10, "val": 5242},
        "recipient_field": {"train": 10, "val": 6789},
    }


def _denominators() -> dict[str, int]:
    return {
        "amount": 1428,
        "time": 3738,
        "payment_method_field": 5242,
        "recipient_field": 6789,
    }


def test_real_031004_shape_fails_closed_below_absolute_continuation_floor() -> None:
    root = _summary(recipient_matches=[10], recipient_only=False)
    pilot = _summary(
        recipient_matches=[700, 1114, 1997, 2651, 3378, 3880, 4108, 4467],
        recipient_only=True,
    )
    decision = build_analysis_decision(
        root_summary=root,
        pilot_summary=pilot,
        expected_field_counts=_raw_counts(),
        candidate_denominators=_denominators(),
    )

    assert decision["recipient_observed"]["epoch8_exact"] == 4467 / 6789
    assert decision["recipient_observed"]["epoch4_to_8_gain"] == pytest.approx(
        (4467 - 2651) / 6789
    )
    assert decision["recipient_observed"]["epoch4_to_8_gain"] > 0.02
    assert decision["recipient_observed"]["best_exact"] < 0.75
    assert decision["continuation_16_epoch_authorized"] is False
    assert decision["production_route_authorized"] is False
    assert decision["onnx_delivery_authorized"] is False


@pytest.mark.parametrize(
    "metric,match",
    [
        (_metric(1, 2), "inconsistent candidate metric"),
        (_metric(1, 1, rate=0.5), "inconsistent candidate metric"),
        ({"records": 1, "exact_matches": True, "exact_match": 1.0}, "inconsistent candidate metric"),
    ],
)
def test_metric_rejects_raw_denominator_rate_or_count_substitution(
    metric: dict[str, object], match: str
) -> None:
    record = {"val_candidate_text_by_field": {"amount": metric}}
    with pytest.raises(ValueError, match=match):
        _candidate_metric(record, "amount", expected_records=1)


def test_all_epoch_and_checkpoint_metrics_use_rebuilt_denominators() -> None:
    root = _summary(recipient_matches=[10], recipient_only=False)
    pilot = _summary(recipient_matches=[1000] * 8, recipient_only=True)
    pilot["records"][5]["val_candidate_text_by_field"]["amount"] = _metric(88, 1606)
    with pytest.raises(ValueError, match="inconsistent candidate metric for amount"):
        build_analysis_decision(
            root_summary=root,
            pilot_summary=pilot,
            expected_field_counts=_raw_counts(),
            candidate_denominators=_denominators(),
        )

    summary_record = _record(8, 4467)
    embedded = _record(8, 4467)
    embedded["val_candidate_text_by_field"]["amount"] = _metric(88, 1606)
    with pytest.raises(ValueError, match="inconsistent candidate metric for amount"):
        _assert_checkpoint_metrics_match_summary(
            {"metrics": embedded},
            summary_record,
            candidate_denominators=_denominators(),
            description="pilot last",
        )


def test_no_clobber_atomic_writer_rejects_existing_and_broken_symlink(tmp_path: Path) -> None:
    output = tmp_path / "decision.json"
    _atomic_write_json_no_clobber(output, {"production_route_authorized": False})
    original = output.read_bytes()
    with pytest.raises(FileExistsError):
        _atomic_write_json_no_clobber(output, {"production_route_authorized": True})
    assert output.read_bytes() == original

    broken = tmp_path / "broken.json"
    try:
        broken.symlink_to("missing-target")
    except (NotImplementedError, OSError):
        pytest.skip("symlinks are unavailable")
    with pytest.raises(FileExistsError):
        _atomic_write_json_no_clobber(broken, {"production_route_authorized": True})


def test_frozen_evidence_rejects_reparse_replacement(tmp_path: Path) -> None:
    evidence = tmp_path / "evidence.json"
    evidence.write_text("bound", encoding="utf-8")
    frozen = _FrozenEvidence({"evidence": evidence})
    moved = tmp_path / "moved.json"
    try:
        try:
            evidence.rename(moved)
        except PermissionError:
            # Windows opens the frozen handle without delete sharing, so the
            # kernel itself prevents the replacement before verify() runs.
            assert evidence.read_text(encoding="utf-8") == "bound"
            frozen.verify()
            return
        try:
            evidence.symlink_to(moved.name)
        except (NotImplementedError, OSError):
            pytest.skip("symlinks are unavailable")
        with pytest.raises(ValueError, match="reparse point"):
            frozen.verify()
    finally:
        frozen.close()


def test_frozen_json_is_read_from_open_inode_and_replacement_is_rejected(tmp_path: Path) -> None:
    evidence = tmp_path / "summary.json"
    evidence.write_text('{"epoch": 1}\n', encoding="utf-8")
    frozen = _FrozenEvidence({"summary": evidence})
    replacement = tmp_path / "replacement.json"
    replacement.write_text('{"epoch": 8}\n', encoding="utf-8")
    try:
        try:
            evidence.unlink()
        except PermissionError:
            # On Windows the live frozen handle is itself a no-replace gate.
            assert frozen.json_object("summary", training=True) == {"epoch": 1}
            frozen.verify()
            return
        replacement.rename(evidence)
        assert frozen.json_object("summary", training=True) == {"epoch": 1}
        with pytest.raises(RuntimeError, match="path was replaced"):
            frozen.verify()
    finally:
        frozen.close()


def test_finalize_wrapper_closes_frozen_evidence_on_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    closed: list[bool] = []

    class FakeFrozen:
        def close(self) -> None:
            closed.append(True)

    def fail(*, frozen_evidence: list[object], **_kwargs: object) -> dict[str, object]:
        frozen_evidence.append(FakeFrozen())
        raise ValueError("forced failure after handles opened")

    monkeypatch.setattr(recovery, "_finalize_recovery_impl", fail)
    with pytest.raises(ValueError, match="forced failure"):
        recovery.finalize_recovery(
            input_contract=Path("input"),
            root_output=Path("root"),
            pilot_output=Path("pilot"),
            output=Path("output"),
            recovery_verifier=Path("verifier"),
            recovery_launcher=Path("launcher"),
        )
    assert closed == [True]


def test_recovery_launcher_is_analysis_only_and_uses_fresh_decision() -> None:
    script = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "receipt-ocr-recipient-random-bootstrap-recover.ps1"
    ).read_text(encoding="utf-8")
    assert "recipient_random_bootstrap_recovery" in script
    assert '"--recovery-verifier", $recoveryVerifier' in script
    assert '"--recovery-launcher", $MyInvocation.MyCommand.Path' in script
    assert "analysis-decision.recovered.json" in script
    assert "DELIVERY=NOT AUTHORIZED" in script
    assert "ONNX=NOT EXPORTED" in script
    assert "--onnx-output" not in script
    assert "recipientDeliveryFloor" not in script


def test_recovery_launcher_parses_in_available_powershell() -> None:
    executable = shutil.which("pwsh") or shutil.which("powershell")
    if executable is None:
        pytest.skip("PowerShell is unavailable on this host; Windows preflight runs this test")
    script_path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "receipt-ocr-recipient-random-bootstrap-recover.ps1"
    )
    escaped = str(script_path).replace("'", "''")
    parser_command = (
        "$tokens = $null; $errors = $null; "
        f"[System.Management.Automation.Language.Parser]::ParseFile('{escaped}', "
        "[ref]$tokens, [ref]$errors) | Out-Null; "
        "if ($errors.Count -ne 0) { $errors | ForEach-Object { Write-Error $_.Message }; exit 1 }"
    )
    completed = subprocess.run(
        [executable, "-NoProfile", "-Command", parser_command],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
