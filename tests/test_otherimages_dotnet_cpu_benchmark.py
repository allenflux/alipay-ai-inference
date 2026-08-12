from __future__ import annotations

from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "otherimages-dotnet-cpu-benchmark.ps1"


def test_white_cpu_benchmark_is_independent_strict_cpu_and_samples_both_memory_metrics() -> None:
    source = SCRIPT.read_text(encoding="utf-8")

    assert '"--document-type", "white"' in source
    assert '[string]$WhiteStudentBundle' in source
    assert '"--white-student-bundle", $WhiteStudentBundle' in source
    assert '"--device", "cpu"' in source
    assert '"--ocr", "onnx"' in source
    assert '"--rectification", "none"' in source
    assert '"--annotate", "none"' in source
    assert "Start-Process" in source
    assert "$process.WorkingSet64" in source
    assert "$process.PrivateMemorySize64" in source
    assert "$process.PeakWorkingSet64" in source
    assert "memory_poll_interval_ms" in source
    assert "total_processor_seconds" in source
    assert "peak_private_bytes" in source
    assert "Get-CimInstance Win32_Processor" in source
    assert "throughput_images_per_second" in source
    assert "inference_latency_ms" in source
    assert '"p50", "p95"' in source
    assert "Microsoft\\.ML\\.OnnxRuntime\\.Gpu/" in source
    assert "onnxruntime_providers_cuda" in source
    assert "app_payload = Get-DirectoryPayloadEvidence" in source
    assert "ocr_bundle = Get-DirectoryPayloadEvidence" in source
    assert "white_student_bundle = Get-DirectoryPayloadEvidence" in source
    assert '[string]$summary.white_student_provider -ne "cpu"' in source
    assert '[string]$result.ocr.student_provider -ne "cpu"' in source
    assert 'same_paddle_db_cls_oriented_crop' in source
    assert "white_student_model_sha256" in source
    assert "white_student_charset_sha256" in source
    assert "white_student_contract_sha256" in source
    assert "white_student_runtime_source" in source
    assert "closure_sha256" in source
    assert "Published app, PP-OCR, or white student bundle closure changed" in source


def test_white_cpu_benchmark_defaults_to_one_warmup_and_three_measured_repetitions() -> None:
    source = SCRIPT.read_text(encoding="utf-8")

    assert "[int]$WarmupRuns = 1" in source
    assert "[int]$WarmupImages = 8" in source
    assert "[int]$Repetitions = 3" in source
    assert "[int]$PollIntervalMilliseconds = 200" in source
    assert "Refusing to reuse benchmark output root" in source
    assert "A benchmark artifact changed while the CPU runs were executing" in source


def test_optional_baseline_gate_covers_throughput_latency_and_memory() -> None:
    source = SCRIPT.read_text(encoding="utf-8")

    assert "$MaxThroughputRegressionPercent = 5.0" in source
    assert "$MaxLatencyRegressionPercent = 5.0" in source
    assert "$MaxMemoryRegressionPercent = 10.0" in source
    assert "$MaxMemoryAbsoluteIncreaseMiB = 128" in source
    assert "throughput regressed" in source
    assert "inference $latencyName regressed" in source
    assert "$memoryName regressed" in source
    assert "$memoryName increased" in source


def test_no_baseline_requires_the_complete_absolute_budget_set_and_is_diagnostic_only() -> None:
    source = SCRIPT.read_text(encoding="utf-8")

    parameters = (
        "MaxP50LatencyMilliseconds",
        "MaxP95LatencyMilliseconds",
        "MinThroughputImagesPerSecond",
        "MaxPeakWorkingSetMiB",
        "MaxPeakPrivateBytesMiB",
        "MaxPackageSizeMiB",
    )
    for parameter in parameters:
        assert f"[double]${parameter}" in source
        assert f'"{parameter}"' in source

    assert "if (-not $absoluteBudgetComplete)" in source
    assert "Formal acceptance requires all absolute CPU budget parameters" in source
    assert "$formalGateConfigured = $absoluteBudgetComplete" in source
    assert '"baseline_regression"' not in source
    assert "$diagnosticOnly = $true" in source
    assert (
        "$reportAccepted = $formalGateConfigured -and -not $diagnosticOnly "
        "-and $gateFailures.Count -eq 0"
    ) in source
    assert "diagnostic_only = $diagnosticOnly" in source
    assert '"diagnostic_only_incomplete_budget_configuration"' in source


def test_absolute_gate_covers_latency_throughput_memory_and_unique_package_payload() -> None:
    source = SCRIPT.read_text(encoding="utf-8")

    assert "Get-UniqueDeliveryPayloadEvidence" in source
    assert 'accounting = "unique_resolved_file_paths_case_insensitive_v1"' in source
    assert "delivery_package_payload_bytes" in source
    assert "measured_package_payload_bytes" in source
    assert "inference p50 exceeds absolute budget" in source
    assert "inference p95 exceeds absolute budget" in source
    assert "throughput is below absolute budget" in source
    assert "peak working set exceeds absolute budget" in source
    assert "peak private bytes exceeds absolute budget" in source
    assert "delivery package payload exceeds absolute budget" in source
    assert "if ($absoluteBudgetComplete)" in source
    assert "required_without_baseline = $true" in source


def test_baseline_regression_and_absolute_gate_can_layer_with_explicit_exit_codes() -> None:
    source = SCRIPT.read_text(encoding="utf-8")

    assert '"baseline_regression_and_absolute_budget"' in source
    assert "$null -ne $baselineBinding -and $absoluteBudgetComplete" in source
    assert "baseline_regression = [ordered]@{" in source
    assert "absolute_budget = [ordered]@{" in source
    assert "accepted_exit_code = 0" in source
    assert "rejected_exit_code = 2" in source
    assert "diagnostic_only_exit_code = 3" in source
    assert "selected_exit_code = $selectedExitCode" in source
    assert "if ($diagnosticOnly)" in source
    assert "exit 3" in source
    assert "if (-not $reportAccepted)" in source
    assert "exit 2" in source
