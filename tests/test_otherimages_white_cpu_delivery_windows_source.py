from __future__ import annotations

import shutil
import subprocess
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "otherimages-white-cpu-delivery-windows.ps1"


def _source() -> str:
    return SCRIPT.read_text(encoding="utf-8")


def test_wrapper_freezes_independent_clone_teacher_training_and_never_guesses_assets() -> None:
    source = _source()

    assert "C:\\f3-white-code-3080a69" in source
    assert "D:\\alipay-ai-data\\alipay-ai-inference\\.venv-cu126\\Scripts\\python.exe" in source
    assert "3080a692a37d7efb0f926cce46de831d17f0e4db" in source
    assert "fb7a21f99139edd15eb1bb10e311039ebe28ebf5" in source
    assert "C:\\f3-white-teacher-3080a69-pilot1000-a" in source
    assert "C:\\f3-white-train-3080a69-pilot1000-a" in source
    assert "C:\\f3-white-cpu-delivery-3080a69-pilot1000-a" in source
    assert "status --porcelain=v1 --untracked-files=all" in source
    assert "diff --no-ext-diff --quiet --exit-code HEAD" in source
    assert "ExpectedDeviceModelSha256" in source
    assert "ExpectedDeviceContractSha256" in source
    assert "ExpectedOcrContractSha256" in source
    assert "ExpectedOcrBundleClosureSha256" in source
    assert "PP-OCR bundle closure differs from its predeclared SHA-256" in source
    assert "contains a reparse member" in source
    assert "Join-Path $RepoRoot '.venv-cu126" not in source


def test_build_and_tool_caches_are_confined_to_fresh_c_run_root() -> None:
    source = _source()

    assert "$BuildSourceRoot = Join-Path $RunRoot 'build-source'" in source
    assert "Copy-Item -LiteralPath (Join-Path $RepoRoot 'dotnet\\ReceiptMlNet.Cli')" in source
    assert "$info.EnvironmentVariables['DOTNET_CLI_HOME'] = (Join-Path $RunRoot 'dotnet-cli-home')" in source
    assert "$info.EnvironmentVariables['NUGET_PACKAGES'] = (Join-Path $RunRoot 'nuget-packages')" in source
    assert "$info.EnvironmentVariables['NUGET_HTTP_CACHE_PATH'] = (Join-Path $RunRoot 'nuget-http-cache')" in source
    assert "$info.EnvironmentVariables['TEMP'] = (Join-Path $RunRoot 'temp')" in source
    assert "$info.EnvironmentVariables['TMP'] = (Join-Path $RunRoot 'temp')" in source
    assert "$info.EnvironmentVariables['PYTHONNOUSERSITE'] = '1'" in source
    assert "$info.EnvironmentVariables['PYTHONPATH'] = (Join-Path $RepoRoot 'src')" in source
    assert "$info.EnvironmentVariables.Remove('PYTHONHOME')" in source


def test_incomplete_assets_or_budgets_are_diagnostic_rc3_without_reserving_formal_root() -> None:
    source = _source()
    required = (
        "DeviceModel",
        "ExpectedDeviceModelSha256",
        "ExpectedDeviceContractSha256",
        "OcrBundle",
        "ExpectedOcrContractSha256",
        "ExpectedOcrBundleClosureSha256",
        "MaxP50LatencyMilliseconds",
        "MaxP95LatencyMilliseconds",
        "MinThroughputImagesPerSecond",
        "MaxPeakWorkingSetMiB",
        "MaxPeakPrivateBytesMiB",
        "MaxPackageSizeMiB",
    )
    for name in required:
        assert f"'{name}'" in source
    assert "otherimages_white_cpu_delivery_preflight_diagnostic_v1" in source
    assert "formal_output_reserved = $false" in source
    assert "Write-DiagnosticAndExit" in source
    assert "exit 3" in source
    assert "-DeviceModel(file_not_found)" in source
    assert "-DeviceModel(adjacent_contract_not_found)" in source
    assert "-OcrBundle(directory_not_found)" in source
    assert "-OcrBundle(delivery_contract_not_found)" in source
    assert "$MaxP50LatencyMilliseconds -gt $MaxP95LatencyMilliseconds" in source
    assert "all six predeclared absolute efficiency budgets" in source


def test_student_and_teacher_publications_are_closed_before_any_delivery_run() -> None:
    source = _source()

    assert "otherimages_white_teacher_windows_pipeline_receipt_v1" in source
    assert "otherimages_paddle_teacher_contract_v1" in source
    assert "otherimages_paddle_teacher_receipt_v1" in source
    assert "otherimages_white_student_training_windows_pipeline_receipt_v1" in source
    assert "55227d5782e55f359d2e3a5f6deee9f24bb6ed19947f847e2d719f6b8d3e5518" in source
    assert "[int64]51503" in source
    assert "RequiredTrainWrapperBytes" in source
    assert "$TrainPipeline.training_performed -ne $true" in source
    assert "test_split_oov_zero" in source
    assert "test_split_used_for_training" in source
    assert "generic_test_oov_fail_closed_by_source" in source
    assert "independent_business_accuracy_proven" in source
    assert "cpu_publication_performed" in source
    assert "test_inference_performed" in source
    assert "receipt_ocr_ctc_v1" in source
    assert "generic_text_line" in source
    assert "opencv_exact_rgb_gray_letterbox_v1" in source
    assert "White student bundle must contain exactly one *.contract.json" in source
    assert "roots.student_bundle differs from student_bundle.root" in source
    assert "Assert-ExactFileTree $StudentBundle" in source
    assert "Training receipt student $bindingName binding differs" in source
    assert "teacher_contract_closure_sha256" in source


def test_same_exact_test_split_drives_smoke_full_benchmark_and_formal_scorer() -> None:
    source = _source()

    assert "[string]$record.split -cne 'test'" in source
    assert "heldout_test" in source
    assert "training_eligible -ne $false" in source
    assert "evaluation_only -ne $true" in source
    assert "held_out -ne $true" in source
    assert "accepted_by_split.test" in source
    assert "frozen-test-inputs.txt" in source
    assert "-WarmupRuns','1'" in source
    assert "-WarmupImages','8'" in source
    assert "-Repetitions','3'" in source
    assert "runs\\measured-01\\output" in source
    assert "'--split','test'" in source
    assert "'--max-cer','0.05'" in source
    assert "'--min-document-exact','0.90'" in source
    assert "'--min-line-precision','0.90'" in source
    assert "'--min-line-recall','0.90'" in source
    assert "'--max-three-of-three-cer','0.03'" in source
    assert "--allow-extra-results" not in source
    assert "same_frozen_test_split_for_benchmark_and_scorer=$true" in source
    assert "result_coverage_100_percent=$true" in source
    assert "Assert-ExactWhiteResultsTree" in source
    assert "$directories.Count -ne 1" in source
    assert "[string]$directories[0].Name -cne 'input-list'" in source
    assert "$resultDirectories.Count -ne 0" in source
    assert "^[0-9a-f]{64}$" in source
    assert "inference_summary.json" in source
    assert "inference_manifest.json" in source
    assert "inference_errors.jsonl" in source
    assert "every_inference_output_exact_flat_tree=$true" in source


def test_cpu_provider_performance_memory_and_package_evidence_are_formal_gates() -> None:
    source = _source()

    assert "-p:OnnxRuntimeFlavor=cpu" in source
    assert "--runtime','win-x64'" in source
    assert "--self-contained','true'" in source
    assert "otherimages-dotnet-cpu-benchmark.ps1" in source
    assert "deps_contains_cpu_onnxruntime" in source
    assert "deps_contains_gpu_onnxruntime" in source
    assert "forbidden_gpu_runtime_file_count" in source
    assert "p50_latency_ms" in source
    assert "p95_latency_ms" in source
    assert "throughput_images_per_second" in source
    assert "peak_working_set_bytes" in source
    assert "peak_private_bytes" in source
    assert "package_payload_bytes" in source
    assert "absolute_efficiency_gate_passed=$true" in source
    assert "strict_cpu_onnxruntime=$true" in source
    assert "gpu_runtime_absent=$true" in source


def test_final_portable_package_is_materialized_before_and_is_the_only_inference_source() -> None:
    source = _source()

    assert "$DeliveryPackageRoot = Join-Path $RunRoot 'publication\\white-document-cpu-win-x64'" in source
    assert "$PublishRoot = Join-Path $DeliveryPackageRoot 'app'" in source
    assert "$PackagedDeviceRoot = Join-Path $DeliveryPackageRoot 'statusbar'" in source
    assert "$PackagedOcrBundle = Join-Path $DeliveryPackageRoot 'ppocr'" in source
    assert "$PackagedStudentBundle = Join-Path $DeliveryPackageRoot 'white-student'" in source
    assert "Copy-Item -LiteralPath $OcrBundle -Destination $PackagedOcrBundle -Recurse" in source
    assert "Copy-Item -LiteralPath $StudentBundle -Destination $PackagedStudentBundle -Recurse" in source
    assert "'-DeviceModel',$PackagedDeviceModel" in source
    assert "'-OcrBundle',$PackagedOcrBundle" in source
    assert "'-WhiteStudentBundle',$PackagedStudentBundle" in source
    assert "DeliveryPackageClosure.size_bytes -ne [int64]$Benchmark.measured.delivery_package_payload_bytes" in source
    assert "final_package_materialized_before_inference=$true" in source
    assert "benchmark_executed_only_from_final_package=$true" in source


def test_fresh_no_clobber_process_finally_and_evidence_closure_are_explicit() -> None:
    source = _source()

    assert "CreateExclusive($RunRoot)" in source
    assert "RunRoot must be brand-new" in source
    assert "Refusing to overwrite evidence" in source
    assert "RedirectStandardOutput = $true" in source
    assert "RedirectStandardError = $true" in source
    assert "ReadToEndAsync" in source
    assert "finally" in source
    assert "Stop-ProcessTree" in source
    assert "Add-ObservedDescendantPids" in source
    assert "Wait-ProcessIdsAbsent" in source
    assert "Stop-ObservedProcessIds" in source
    assert "Get-CimInstance Win32_Process" in source
    assert "forced_process_tree_stop" in source
    assert "observed_descendant_pids" in source
    assert "descendant_absence_proven" in source
    assert "WaitForExit(30000)" in source
    assert "Redirected child stream did not close within bounded cleanup wait" in source
    assert "Write-RcNew" in source
    assert "exact ASCII integer CRLF" in source
    assert "otherimages_white_cpu_delivery_windows_stage_failure_v1" in source
    assert "$stderrNonEmpty" in source
    assert "finally { $process.Dispose() }" in source
    assert "$result.stderr.size_bytes -ne 0" in source
    assert "every_stage_rc_zero=$true" in source
    assert "every_stage_stderr_zero_bytes=$true" in source
    assert "every_stage_descendant_absence_proven=$true" in source
    assert "process_tree_cleanup_on_failure=$true" in source
    assert "Stage RC/stderr/descendant closure failed before pipeline receipt" in source
    assert "input_and_code_closure_stable=$true" in source
    assert "otherimages_white_cpu_delivery_windows_pipeline_receipt_v1" in source
    assert "otherimages_white_cpu_delivery_windows_failure_v1" in source


def test_windows_powershell_parser_accepts_source_when_available() -> None:
    powershell = shutil.which("powershell") or shutil.which("pwsh")
    if powershell is None:
        return
    command = f"[scriptblock]::Create([IO.File]::ReadAllText('{SCRIPT}')) | Out-Null"
    completed = subprocess.run(
        [powershell, "-NoLogo", "-NoProfile", "-NonInteractive", "-Command", command],
        check=False,
        text=True,
        capture_output=True,
    )
    assert completed.returncode == 0, completed.stderr
