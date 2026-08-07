from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
WRAPPER = ROOT / "scripts" / "receipt-ocr-status-text-v13-4090.ps1"


def _source() -> str:
    return WRAPPER.read_text(encoding="utf-8")


def test_wrapper_builds_v13_from_existing_flat_records_without_teacher_inference() -> None:
    source = _source()

    assert '[string]$PseudoLabels' in source
    assert '"-m", "transfer_receipt_ai.ocr_unified_dataset"' in source
    assert '"--records", $PseudoLabels' in source
    assert '"--architecture", "v13"' in source
    assert "receipt_unified_field_dataset_v6" in source
    assert "ocr_pseudolabels" not in source
    assert "paddle" not in source.casefold()


def test_wrapper_is_cuda_only_status_head_training_from_wide_v12() -> None:
    source = _source()

    assert '.venv-cu126\\Scripts\\python.exe' in source
    assert "nvidia-smi" in source
    assert '"--device", "cuda:0"' in source
    assert '"--recipient-input-width", "1536"' in source
    assert '"--recipient-open-text-layers", "2"' in source
    assert '"--recipient-open-text-heads", "8"' in source
    assert '"--recipient-open-text-feedforward", "2048"' in source
    assert '"--architecture", "v13"' in source
    assert '"--status-text-only-fine-tune"' in source
    assert '"--init-checkpoint", $SeedCheckpoint' in source
    assert '"--init-checkpoint-mode", "strict"' in source
    assert 'initialization.mode -ne "parameter_only_v12_to_v13_status_text_expansion"' in source
    assert 'financial_label_policy.mode -ne "checkpoint_legacy_label_maps_status_text_only_v1"' in source
    assert "initialization.source_config.recipient_input_width -ne 1536" in source
    assert "$runtime.uses_cuda -ne $true" in source
    assert "$runtime.status_text_only_training -ne $true" in source
    assert "[int]$ValidationEvery = 4" in source
    assert '"--validation-every", "$ValidationEvery"' in source
    assert 'fineTune.full_validation_schedule -ne "epoch_1_every_n_and_final_epoch"' in source
    assert "fineTune.validation_every -ne $ValidationEvery" in source
    assert "runtime.validation_every -ne $ValidationEvery" in source


def test_wrapper_freezes_old_15_outputs_and_preserves_four_field_floors() -> None:
    source = _source()

    assert source.count('"recipient_logits"') >= 1
    assert "$seedOutputProperties.Count -ne 15" in source
    assert "$candidateOutputProperties.Count -ne 16" in source
    assert "frozen_legacy_output_count -ne 15" in source
    assert "Legacy output ABI parity failed" in source
    assert "source exact-tensor test + frozen legacy parameter audit" in source
    assert "$amountFloor = 0.7885" in source
    assert "$timeFloor = 0.9840" in source
    assert "$paymentFloor = 0.9325" in source
    assert "$recipientFloor = 0.90" in source
    assert "[double]$AmountFloor" not in source
    assert "[double]$TimeFloor" not in source
    assert "[double]$PaymentFloor" not in source
    assert "[double]$RecipientFloor" not in source
    assert '"--min-recipient-exact-match", "$recipientFloor"' in source


def test_wrapper_fails_closed_on_visible_status_text_and_review_only_delivery() -> None:
    source = _source()

    assert "[double]$StatusTextFloor = 0.90" in source
    assert '"--min-status-exact-match", "$StatusTextFloor"' in source
    assert '"ctc_raw_exact_match"' in source
    assert '"decode_and_normalize_review_only"' in source
    assert '$requiredReviewValue = "review"' in source
    assert 'Properties["status_text_logits"]' in source
    assert "candidateContract.status_head_policy.runtime_policy -ne \"review_only\"" in source
    assert 'Get-StatusOovAudit $datasetContract "val"' in source
    assert 'Get-StatusOovAudit $datasetContract "test"' in source
    assert "train status text must have zero OOV" in source
    assert "held-out train-charset OOV is retained as an error" in source
    assert "max_possible_exact_match -lt $StatusTextFloor" in source
    assert "train-charset OOV makes the requested status exact floor impossible" in source
    assert '"--max-non-success-to-success", "0"' in source
    assert "no pending/failed truth in this split; no non-success safety claim is made" in source
    assert "non_success_safety_calibrated" in source
    assert "status_safety_then_transfer_status_raw_ctc_exact_then_recipient_exact_after_protected_candidate_exact_floors" in source
    assert "checkpoint_selection_policy.status_text_ctc_priority -ne $true" in source
    assert "checkpoint_selection_score[0] -ne $expectedStatusSafetyScore" in source
    assert "checkpoint_selection_score[1]" in source
    assert "val_ctc_by_field.transfer_status.exact_match" in source
    assert "val_ctc_by_field.transfer_status.exact_match -lt $StatusTextFloor" in source


def test_wrapper_exports_and_evaluates_but_delegates_cpu_packaging() -> None:
    source = _source()

    assert source.count('"transfer_receipt_ai.ocr_unified", "export"') == 2
    assert '"transfer_receipt_ai.ocr_unified", "evaluate"' in source
    assert "CUDAExecutionProvider" in source
    assert "v13_status_ocr_validation.json" in source
    assert "cpu_packaging" in source
    assert "performed = $false" in source
    assert "receipt-mlnet-unified-package-validate-4090.ps1" in source
    assert "& $pythonExe @CommandArguments" in source
    assert "$LASTEXITCODE -ne 0" in source


def test_wrapper_powershell_parses_when_powershell_is_available() -> None:
    executable = shutil.which("pwsh") or shutil.which("powershell")
    if executable is None:
        pytest.skip("PowerShell is not installed")
    escaped_path = str(WRAPPER).replace("'", "''")
    parser_command = (
        "$errors=$null; "
        "[void][System.Management.Automation.Language.Parser]::ParseFile("
        f"'{escaped_path}',[ref]$null,[ref]$errors); "
        "if($errors.Count -gt 0){$errors | ForEach-Object { Write-Error $_ }; exit 1}"
    )
    completed = subprocess.run(
        [executable, "-NoProfile", "-NonInteractive", "-Command", parser_command],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
