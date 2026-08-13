from __future__ import annotations

from pathlib import Path
import re


SCRIPT = Path(__file__).parents[1] / "scripts" / "otherimages-white-pilot-windows.ps1"


def _source() -> str:
    return SCRIPT.read_text(encoding="utf-8")


def test_wrapper_is_winps51_fail_closed_and_uses_fixed_fresh_c_roots() -> None:
    source = _source()

    assert "Set-StrictMode -Version Latest" in source
    assert "$ErrorActionPreference = 'Stop'" in source
    assert "C:\\f3-white-sync\\sample-10000-a\\white-sample-10000-21054be8b1eb.zip" in source
    assert "C:\\f3-white-pilot-21054be8b1eb-a" in source
    assert "RunRoot must be brand-new" in source
    assert "New-ExclusiveDirectory $RunRoot" in source
    assert "[WhitePilotNativeDirectoryV1]::CreateExclusive" in source
    assert "CreateDirectoryW" in source
    assert "GetFileInformationByHandle" in source
    assert source.count("CharSet = CharSet.Unicode, SetLastError = true, ExactSpelling = true") == 2
    assert "Add-Type -TypeDefinition $source -Language CSharp | Out-Null" in source
    assert "$RunRootOwned = $false" in source
    assert "$RunRootOwned = $true" in source
    assert "if ($RunRootOwned -and (Test-Path -LiteralPath $RunRoot -PathType Container))" in source
    assert "Refusing to write failure receipt after RunRoot identity changed" in source
    assert source.index("New-ExclusiveDirectory $RunRoot") < source.index("$RunRootOwned = $true")
    assert source.index("$RunRootOwned = $true") < source.index("$LogsRoot = Join-Path $RunRoot 'logs'")
    assert "if ([string]::IsNullOrWhiteSpace($ArchivePath)) { $ArchivePath = $receiptArchivePath }" in source
    assert "function Require-CPath" in source
    assert "function Assert-NoReparseChain" in source
    assert "GetPathRoot($full) -cne 'C:\\'" in source
    assert "??" not in source
    assert "-AsByteStream" not in source
    assert "ForEach-Object -Parallel" not in source


def test_archive_is_bound_to_prepare_receipt_size_and_sha_before_local_server() -> None:
    source = _source()
    hash_gate = source.index("$observedArchiveSha = Get-ArchiveSha256")
    server_start = source.index("$server = Start-LoopbackServer")

    assert "$receiptArchivePath.Equals($ArchivePath,[StringComparison]::OrdinalIgnoreCase)" in source
    assert "6b0b93d8651ee6cebbcdf62e1200c0f8041f508cb9904e8e955c510be682481e" in source
    assert "2809303412" in source
    assert "21054be8b1eb04f478c5b7e817cbe3367fef82e0b8d259b8025253df3f6af71e" in source
    assert "45fcca6b8b6b4fe97691a794e8aaf287026ee7a53ac9a4fb1b8be04ac6dc5938" in source
    assert "Archive size differs from prepare receipt before hashing" in source
    assert "Archive SHA256 differs from prepare receipt" in source
    assert hash_gate < server_start
    assert "if ((Get-ArchiveSha256 $ArchivePath $receiptBytes) -cne $receiptSha)" in source
    assert "source_archive_modified = $false" in source
    assert "Prepare receipt changed during pilot pipeline" in source
    assert "Prepare receipt semantic binding changed during pilot pipeline" in source


def test_receiver_is_formal_loopback_only_and_server_stops_before_prefix() -> None:
    source = _source()
    server_start = source.index("$server = Start-LoopbackServer")
    receive = source.index("$receiveStage = Invoke-PythonStage 'receive'")
    server_stop = source.index("$serverEvidence = Stop-LoopbackServer")
    prefix = source.index("$prefixStage = Invoke-PythonStage 'prefix1000'")

    assert "'-m','http.server'" in source
    assert "'--bind','127.0.0.1'" in source
    assert "http://127.0.0.1:" in source
    assert "scripts\\otherimages-white-sample-receive.py" in source
    assert server_start < receive < server_stop < prefix
    assert "startup_marker_valid = $true" in source
    assert "successful_get_200_observed = $true" in source
    assert "Loopback server exited unexpectedly" in source
    assert "--max-archive-bytes" in source
    assert "--max-uncompressed-bytes" in source
    start_body = source[source.index("function Start-LoopbackServer"):source.index("function Stop-LoopbackServer")]
    assert "catch {" in start_body
    assert "$process.Kill()" in start_body
    assert "$process.WaitForExit(30000)" in start_body
    assert "$process.Dispose()" in start_body
    stop_body = source[source.index("function Stop-LoopbackServer"):source.index("try {", source.index("function Stop-LoopbackServer"))]
    # The complete function uses two bounded kill/wait attempts and proves exit
    # before consuming evidence; its finally block always disposes the handle.
    stop_body = source[source.index("function Stop-LoopbackServer"):source.index("\ntry {\n    $PrepareReceipt")]
    assert stop_body.count("$process.WaitForExit(30000)") == 2
    assert "Loopback server remained alive after two kill/wait attempts" in stop_body
    assert "Loopback server cleanup did not prove process exit" in stop_body
    assert "finally { $process.Dispose() }" in stop_body


def test_each_python_stage_persists_utf8_stdout_empty_stderr_and_exact_ascii_rc() -> None:
    source = _source()

    assert "function Invoke-PythonStage" in source
    assert "ReadToEndAsync()" in source
    assert "PYTHONIOENCODING'] = 'utf-8:strict'" in source
    assert "PYTHONUTF8'] = '1'" in source
    assert "[IO.File]::WriteAllText($stdoutPath, $stdout, $Utf8NoBom)" in source
    assert "[IO.File]::WriteAllText($stderrPath, $stderr, $Utf8NoBom)" in source
    assert "Write-RcNew $rcPath $process.ExitCode" in source
    assert "$Ascii.GetBytes(([string]$Rc) + \"`r`n\")" in source
    assert "if ($result.rc -ne 0)" in source
    assert "if ($result.stderr.size_bytes -ne 0)" in source
    assert "otherimages_white_pilot_windows_stage_receipt_v1" in source


def test_long_stages_emit_pid_cpu_memory_heartbeats() -> None:
    source = _source()

    assert "WHITE_PILOT_STAGE_START" in source
    assert "WHITE_PILOT_STAGE_ALIVE" in source
    assert "elapsed_s=" in source
    assert "pid=" in source
    assert "cpu_s=" in source
    assert "ws_bytes=" in source
    assert "WHITE_PILOT_STAGE_EXIT" in source
    assert "WHITE_PILOT_ARCHIVE_HASH_ALIVE" in source


def test_prefix_and_inventory_use_formal_modules_and_strict_publication_gates() -> None:
    source = _source()

    assert "scripts\\otherimages-white-prefix-materialize.py" in source
    assert "scripts\\otherimages-inventory.py" in source
    assert "source.full_manifest.image_count -ne 10000" in source
    assert "value.prefix.image_count -ne 1000" in source
    assert "every_prefix_copy_size_and_sha256_verified" in source
    assert "source_files_written -ne $false" in source
    assert "Internal/external prefix receipts are not byte-identical" in source
    assert "inventory.counts.images -ne 1000" in source
    assert "inventory.counts.image_errors_quarantined -ne 0" in source
    assert "inventory_performed_ocr -ne $false" in source
    assert "inventory_performed_training -ne $false" in source
    assert "training_eligibility_before_teacher_validation -ne $false" in source
    assert "Inventory output tree is not the exact eight-file closure" in source
    assert "Inventory contract input root differs from the exact prefix images root" in source
    assert "'--max-phash-candidates','500000'" in source
    assert "suggested_splits.train -ne 912" in source
    assert "suggested_splits.val -ne 52" in source
    assert "suggested_splits.test -ne 36" in source
    assert "teacher_states.pending -ne 999" in source
    assert "teacher_states.quarantine -ne 1" in source
    assert "candidate_evidence_rows -ne 6055" in source
    assert "represented_record_pairs -ne 6055" in source
    assert "phash_candidates.truncated -ne $false" in source


def test_inventory_portable_projection_matches_python_canonical_json_contract() -> None:
    source = _source()

    assert "function New-PortableProjection" in source
    assert "function ConvertTo-PythonJsonString" in source
    projection_encoder = source[source.index("function ConvertTo-PythonJsonString"):source.index("function New-PortableProjection")]
    assert "switch ($code)" not in projection_encoder
    assert "if ($code -eq 34) { [void]$builder.Append('\\\"'); continue }" in projection_encoder
    assert "if ($code -eq 92) { [void]$builder.Append('\\\\'); continue }" in projection_encoder
    assert "'decoded_pixel_sha256','group_id','phash64','quarantine_reason','raw_sha256'" in source
    assert "'record_id','source_relative_path','suggested_split','teacher_state'" in source
    assert "Sort-Object { [string]$_.record_id }" in source
    assert "$writer.Write(\"}`n\")" in source
    assert "bd1b964117595a2e71b898d45f66393e5c15b92f863fbabd2f790499dbee009c" in source
    assert "527892" in source
    assert "Portable projection row count differs from 1000" in source
    assert "Portable projection record_id order is not strictly unique" in source
    assert "Inventory portable projection SHA/size/count failed" in source


def test_pipeline_receipt_is_analysis_only_and_no_training_or_ocr_is_run() -> None:
    source = _source()

    assert "otherimages_white_pilot_windows_pipeline_receipt_v1" in source
    assert "analysis_only = $true" in source
    assert "production_route_authorized = $false" in source
    assert "training_performed = $false" in source
    assert "ocr_performed = $false" in source
    assert "every_stage_rc_zero=$true" in source
    assert "every_stage_stderr_zero_bytes=$true" in source
    assert "WHITE_PILOT_PIPELINE_OK" in source
    assert "WHITE_PILOT_PIPELINE_FAILED" in source
    assert not re.search(r"(?i)\b(train|fit|onnx|cuda)(\.py|\.ps1|\s+--)", source)


def test_wrapper_only_invokes_the_three_expected_formal_stage_scripts() -> None:
    source = _source()
    calls = re.findall(r"Invoke-PythonStage '([^']+)'", source)

    assert calls == ["receive", "prefix1000", "inventory"]
    assert "Remove-Item" not in source
    assert "Copy-Item" not in source
    assert "Move-Item" not in source
