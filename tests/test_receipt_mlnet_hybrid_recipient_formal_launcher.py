from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "scripts" / "receipt-mlnet-hybrid-recipient-formal-ab.ps1"
V13_GENERATOR = ROOT / "scripts" / "receipt-ocr-status-text-v13-4090.ps1"


def _source() -> str:
    payload = LAUNCHER.read_bytes()
    assert all(byte < 128 for byte in payload), "formal launcher must remain ASCII-only"
    return payload.decode("ascii").replace("\r\n", "\n")


def test_launcher_requires_only_explicit_evidence_bundle_output_and_optional_dotnet() -> None:
    source = _source()

    for parameter in ("V13EvidencePath", "PaddleDeliveryBundle", "OutputDirectory"):
        assert "[Parameter(Mandatory = $true)]\n    [string]$" + parameter in source
    assert "[string]$DotnetExe" in source
    assert "TeacherRoot" not in source
    assert 'GetFileName($V13EvidencePath).Equals(' in source
    assert '"v13_status_ocr_validation.json"' in source


def test_launcher_fails_closed_on_v13_schema_and_complete_cpu_binding() -> None:
    source = _source()
    generator = V13_GENERATOR.read_text(encoding="utf-8")

    assert "schema_version = 1" in generator
    assert 'kind = "receipt_unified_status_text_v13_guarded_validation_v1"' in generator
    assert "return ,$property.Value" in source
    for integer_type in (
        "sbyte",
        "byte",
        "int16",
        "uint16",
        "int32",
        "uint32",
        "int64",
        "uint64",
    ):
        assert f"$Value -is [{integer_type}]" in source
    assert "$Value -is [bool]" in source
    assert "$Value -is [string]" in source
    for token in (
        "Test-JsonIntegerEqual $schemaVersion 1",
        "Test-JsonIntegerEqual $candidateArchitecture 13",
        'schema-version-1 guarded v13 visible-status OCR evidence',
        'receipt_unified_status_text_v13_guarded_validation_v1',
        'receipt_unified_field_reader_v13',
        'architecture_version',
        'decode_and_normalize_review_only',
        'review_value',
        'required_runtime_flavor',
        'required_rectification',
        'include_device_model',
        'complete production CPU/device pipeline',
    ):
        assert token in source
    assert '$runtimeFlavor -isnot [string]' in source
    assert '[string]$runtimeFlavor -ne "cpu"' in source
    assert '$rectification -isnot [string]' in source
    assert '[string]$rectification -ne "max-side-1600"' in source
    assert '$includeDeviceModel -isnot [bool]' in source


def test_launcher_resolves_contained_records_and_identical_candidate_model() -> None:
    source = _source()

    assert source.count("Resolve-BoundFile `") == 8
    assert 'Get-RequiredProperty $manifest "records"' in source
    assert 'Get-RequiredProperty $cpuPackaging "unified_model_path"' in source
    assert 'Get-RequiredProperty $candidate "model"' in source
    assert 'Get-RequiredProperty $candidate "contract"' in source
    assert 'Get-RequiredProperty $candidate "labels"' in source
    assert 'Get-RequiredProperty $cpuPackaging "onnx_validation_summary_path"' in source
    assert 'Get-RequiredProperty $valEvidence[0] "summary_path"' in source
    assert 'Get-RequiredProperty $testEvidence[0] "summary_path"' in source
    assert "Test-PathWithin $candidate $EvidenceDirectory" in source
    assert "path escapes its evidence directory" in source
    assert "$unifiedModel.Equals($candidateModel, [StringComparison]::OrdinalIgnoreCase)" in source
    assert "CPU packaging unified model does not equal candidate.model" in source
    assert '[IO.Path]::ChangeExtension($unifiedModel, ".contract.json")' in source
    assert '[IO.Path]::ChangeExtension($unifiedModel, ".labels.json")' in source
    assert "$candidateContract.Equals($adjacentContract" in source
    assert "$candidateLabels.Equals($adjacentLabels" in source
    assert "sidecars do not equal the contract/labels files loaded beside" in source


def test_launcher_verifies_all_records_and_model_hash_bindings_before_and_after_run() -> None:
    source = _source()

    for token in (
        'Get-RequiredProperty $manifest "records_sha256"',
        'Get-RequiredProperty $candidate "model_sha256"',
        'Get-RequiredProperty $cpuPackaging "unified_model_sha256"',
        "^[0-9a-f]{64}$",
        "$recordsExpectedSha256 -cne $recordsSha256",
        "$candidateExpectedSha256 -cne $unifiedModelSha256",
        "$packagingExpectedSha256 -cne $unifiedModelSha256",
        "Get-Sha256 $V13EvidencePath",
        "bound records/model/sidecar/evaluation file changed during formal A/B",
        'Get-RequiredProperty $unifiedContract "onnx_file"',
        'Get-RequiredProperty $unifiedContract "labels_file"',
        'Get-RequiredProperty $unifiedContract "onnx_sha256"',
        'Get-RequiredProperty $unifiedContract "labels_sha256"',
        "not hash/name-bound to its adjacent ONNX and labels",
    ):
        assert token in source
    call = source.index("& $launcher @arguments")
    post_run_check = source.rindex("(Get-Sha256 $V13EvidencePath) -cne $evidenceSha256")
    assert call < post_run_check


def test_launcher_requires_hash_bound_passed_val_and_test_gpu_evidence() -> None:
    source = _source()

    for token in (
        'Get-RequiredProperty $candidate "contract_sha256"',
        'Get-RequiredProperty $candidate "labels_sha256"',
        'Get-RequiredProperty $cpuPackaging "onnx_validation_summary_sha256"',
        'Get-RequiredProperty $valEvidence[0] "summary_sha256"',
        'Get-RequiredProperty $testEvidence[0] "summary_sha256"',
        "$valEvidence.Count -ne 1",
        "$testEvidence.Count -ne 1",
        '$evaluated -isnot [bool]',
        'Assert-PassedGpuSummary $valSummary "val"',
        'Assert-PassedGpuSummary $testSummary "test"',
        'CUDAExecutionProvider',
        'StartsWith("recipient_field:"',
        'Field = "amount"; Metric = "raw_exact_match"',
        'Field = "time"; Metric = "raw_exact_match"',
        'Field = "payment_method_field"; Metric = "raw_exact_match"',
        'Field = "transfer_status"; Metric = "ctc_raw_exact_match"',
        'max_non_success_to_success',
        'zero non-success safety line',
        'guarded status metrics do not match its GPU summary',
    ):
        assert token in source
    assert "$validationSummary.Equals($valSummaryPath" in source
    assert "$validationExpectedSha256 -cne $validationSummarySha256" in source
    assert "$valExpectedSha256 -cne $validationSummarySha256" in source
    assert "$testExpectedSha256 -cne $testSummarySha256" in source


def test_launcher_delegates_exact_full_formal_without_tunable_guards() -> None:
    source = _source()

    assert 'Join-Path $PSScriptRoot "receipt-mlnet-hybrid-recipient-cpu-ab.ps1"' in source
    assert 'Mode = "formal"' in source
    assert "Limit = 0" in source
    assert "& $launcher @arguments" in source
    assert '$arguments["DotnetExe"] = $DotnetExe' in source
    for forbidden in (
        "AmountFloor",
        "TimeFloor",
        "PaymentFloor",
        "RecipientFloor",
        "StatusFloor",
        "DetectorModel =",
        "$arguments[\"DeviceModel\"]",
        "MaxP95OverheadMs",
        "DetectorIntraOpThreads",
    ):
        assert forbidden not in source
    assert "Formal A/B output already exists; refusing result reuse" in source


def test_launcher_prints_every_resolved_binding_path() -> None:
    source = _source()

    for output in (
        "evidence=$V13EvidencePath",
        "evidence-root=$evidenceDirectory",
        "records=$records",
        "unified-model=$unifiedModel",
        "unified-contract=$candidateContract",
        "unified-labels=$candidateLabels",
        "val-summary=$validationSummary",
        "test-summary=$testSummaryPath",
        "recipient-ppocr=$PaddleDeliveryBundle",
        "output=$OutputDirectory",
        "mode=formal; limit=0; delegated-device=cpu; detector/device enabled",
    ):
        assert output in source


def test_launcher_powershell_parses_when_available() -> None:
    executable = shutil.which("powershell") or shutil.which("pwsh")
    if executable is None:
        pytest.skip("PowerShell is not installed")
    escaped_path = str(LAUNCHER).replace("'", "''")
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


def test_launcher_rejects_top_level_json_arrays_before_powershell_can_enumerate_them() -> None:
    source = _source()

    object_guard = source.index('$trimmedJson.StartsWith("{"')
    conversion = source.index("$document = ConvertFrom-Json -InputObject $trimmedJson")
    assert '$trimmedJson.EndsWith("}", [StringComparison]::Ordinal)' in source
    assert object_guard < conversion
