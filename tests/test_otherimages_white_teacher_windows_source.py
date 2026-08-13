from __future__ import annotations

from pathlib import Path
import hashlib
import json
import subprocess
import sys


SCRIPT = Path(__file__).parents[1] / "scripts" / "otherimages-white-teacher-windows.ps1"


def _source() -> str:
    return SCRIPT.read_text(encoding="utf-8")


def test_wrapper_is_winps51_fixed_commit_clean_tree_and_frozen_code() -> None:
    source = _source()

    assert "Set-StrictMode -Version Latest" in source
    assert "$ErrorActionPreference = 'Stop'" in source
    assert "C:\\f3-white-code-3080a69" in source
    assert "C:\\f3-white-pilot-21054be8b1eb-a" in source
    assert "C:\\f3-white-teacher-3080a69-pilot1000-a" in source
    assert "D:\\alipay-ai-data\\alipay-ai-inference\\.venv-cu126\\Scripts\\python.exe" in source
    assert "Join-Path $RepoRoot '.venv-cu126\\Scripts\\python.exe'" not in source
    assert "PythonExe must be the fixed read-only CUDA environment on D:" in source
    assert "3080a692a37d7efb0f926cce46de831d17f0e4db" in source
    assert "fb7a21f99139edd15eb1bb10e311039ebe28ebf5" in source
    assert "status --porcelain=v1 --untracked-files=all" in source
    assert "diff --no-ext-diff --quiet --exit-code HEAD" in source
    assert "rev-parse 'HEAD^{tree}'" in source
    assert "C:\\Program Files\\Git\\cmd\\git.exe" in source
    assert "[string]$GitExe" not in source
    assert "Assert-BindingUnchanged $ExpectedGitBinding 'fixed Git executable before query'" in source
    assert "Assert-BindingUnchanged $ExpectedGitBinding 'fixed Git executable after query'" in source
    assert "Source checkout is not completely clean" in source
    assert "395ad109e260ba58f282023a75d439f93958b22e1159b476d98be0c4c3777308" in source
    assert "470c2753c7fba63e1bd0e2e24e0a04ef7a3f523638933838995567395eae5494" in source
    assert "2155e7b1f49401ee49770241db2183a5ff7d02f34212afa9eb158f64132847c1" in source
    assert "b193f20d4560643c89648019e151c39ecd0b53b42f29e55a7b4813df03e8202b" in source
    assert "6c5ac75aec7d42eaac283b52dda86f3c440c74cde3f6f53799cf5206fe1373cb" in source
    assert "4118d4913d4d4256dbb8c47f853f14d37b87ace456ae5098acfff9e51208b4b4" in source
    assert "FROZEN" not in source
    assert "??" not in source
    assert "-AsByteStream" not in source
    assert "ForEach-Object -Parallel" not in source


def test_cuda_python_is_read_only_bound_each_stage_and_executes_c_source() -> None:
    source = _source()
    stage = source[source.index("function Invoke-PythonStage"):source.index("function Complete-Stage")]

    assert "Assert-BindingUnchanged $PythonBinding 'fixed read-only CUDA Python executable before stage'" in stage
    assert "Assert-BindingUnchanged $PythonBinding 'fixed read-only CUDA Python executable after stage'" in stage
    assert "$info.FileName = $PythonExe" in stage
    assert "$info.WorkingDirectory = $RepoRoot" in stage
    assert "$info.EnvironmentVariables['PYTHONDONTWRITEBYTECODE'] = '1'" in stage
    assert "$info.EnvironmentVariables['PYTHONNOUSERSITE'] = '1'" in stage
    assert "$info.EnvironmentVariables['PYTHONPATH'] = Join-Path $RepoRoot 'src'" in stage
    for name in ("PYTHONHOME", "PYTHONSTARTUP", "PYTHONINSPECT", "PYTEST_ADDOPTS", "PYTEST_PLUGINS"):
        assert name in stage
    assert "$info.EnvironmentVariables.Remove($inheritedPythonVariable)" in stage
    assert "python_environment_read_only=$true" in source
    assert "python_executable_sha_size_stable_each_stage=$true" in source
    assert "python_working_directory_and_path_on_c_source=$true" in source


def test_import_attestation_precedes_long_capture_and_binds_exact_c_modules() -> None:
    source = _source()

    attestation = source.index("Invoke-PythonStage 'import-attestation'")
    capture = source.index("Invoke-PythonStage 'capture-three-view'")
    assert attestation < capture
    assert "otherimages_white_teacher_import_attestation_v1" in source
    assert 'os.environ.get("PYTHONPATH", "")' in source
    assert 'import attestation PYTHONPATH is not frozen C source/src' in source
    assert 'import attestation sys.path omits frozen C source/src' in source
    assert 'Path(module.__file__).resolve(strict=True)' in source
    assert 'import authority escaped frozen C source' in source
    assert 'import authority SHA differs' in source
    assert "Import attestation stdout must be exactly one complete JSON value" in source
    assert "exact_three_module_file_and_sha_authority=$true" in source
    assert "stages=[ordered]@{ import_attestation=$importStageReceipt; capture=$captureStageReceipt" in source


def test_pilot_receipt_inventory_and_bd1_projection_are_exact_gates() -> None:
    source = _source()

    assert "otherimages_white_pilot_windows_pipeline_receipt_v1" in source
    assert "inventory-prefix1000" in source
    assert "paddle_teacher_pending.jsonl" in source
    assert "suggested_splits.train -ne 912" in source
    assert "suggested_splits.val -ne 52" in source
    assert "suggested_splits.test -ne 36" in source
    assert "teacher_states.pending -ne 999" in source
    assert "teacher_states.quarantine -ne 1" in source
    assert "bd1b964117595a2e71b898d45f66393e5c15b92f863fbabd2f790499dbee009c" in source
    assert "527892" in source
    assert "Inventory publication" in source
    assert "Pilot receipt is not bound to the frozen wrapper/inventory source" in source
    assert "Inventory source root differs from the frozen pilot image publication" in source
    assert "guessed_or_synthetic_labels_forbidden -ne $true" in source


def test_capture_is_cuda_single_process_all_views_and_rejects_concurrency() -> None:
    source = _source()

    assert "Get-CimInstance Win32_Process" in source
    assert "receipt-ocr-recipient-multiview-exact8|f3e8|exact8|otherimages-paddle-capture\\.py" in source
    assert "Assert-NoConflictingWork 'preflight'" in source
    assert "Assert-NoConflictingWork 'immediately-before-capture'" in source
    assert "FreePhysicalMemory" in source
    assert "16 GiB free RAM" in source
    assert "OTHERIMAGES_PADDLE_DEVICE='cuda'" in source
    assert "'--view-id','all','--json'" in source
    assert "execution_device -cne 'gpu:0'" in source
    assert "effective_paddle_args.use_gpu -ne $true" in source
    assert "effective_paddle_args.gpu_id -ne 0" in source
    assert "records_per_view -ne 999" in source
    assert "capture_errors -ne 0" in source
    assert "Capture receipt does not bind exactly the three canonical views" in source
    assert source.count("Invoke-PythonStage 'capture-three-view'") == 1
    assert "original_rgb.jsonl" in source
    assert "grayscale_clahe.jsonl" in source
    assert "upscale_sharpen.jsonl" in source


def test_fresh_root_strict_process_evidence_and_heartbeats() -> None:
    source = _source()

    assert "CreateDirectoryW" in source
    assert "GetFileInformationByHandle" in source
    assert "RunRoot must be brand-new" in source
    assert "Capture output must be brand-new" in source
    assert "Teacher output must be brand-new" in source
    assert "ReadToEndAsync()" in source
    assert "Write-RcNew $rcPath" in source
    assert "$Rc.ToString([Globalization.CultureInfo]::InvariantCulture)" in source
    assert "$bytes.Length -ne $expected.Length" in source
    assert "$bytes[$index] -ne $expected[$index]" in source
    stage = source[source.index("function Invoke-PythonStage"):source.index("function Complete-Stage")]
    assert "try {" in stage and "catch { $stageError = $_ }" in stage and "finally {" in stage
    assert stage.count("$process.Kill()") == 2
    assert stage.count("$process.WaitForExit(30000)") == 2
    assert "$streamTask.Wait(30000)" in stage
    assert "$process.Dispose()" in stage
    assert "otherimages_white_teacher_windows_stage_failure_v1" in stage
    assert "process_started=$startedProcess" in stage
    assert "forced_stop=$forcedStop" in stage
    assert "if ($result.rc -ne 0)" in source
    assert "if ($result.stderr.size_bytes -ne 0)" in source
    assert "WHITE_TEACHER_STAGE_START" in source
    assert "WHITE_TEACHER_STAGE_ALIVE" in source
    assert "elapsed_s=" in source
    assert "cpu_s=" in source
    assert "ws_bytes=" in source
    assert "WHITE_TEACHER_STAGE_EXIT" in source


def test_teacher_contract_counts_no_guess_and_receipt_closure_are_strict() -> None:
    source = _source()

    assert "otherimages_paddle_teacher_contract_v1" in source
    assert "otherimages_paddle_teacher_receipt_v1" in source
    assert "inventory_records -ne 1000" in source
    assert "pending_records -ne 999" in source
    assert "accepted_teacher_records + [int]$teacher.counts.quarantined_records" in source
    assert "quarantined_records -lt 1" in source
    assert "Teacher pilot produced no usable train or held-out evaluation evidence" in source
    assert "training_authorization -ne $false" in source
    assert "quarantine_never_guess" in source
    assert "manual_review_required -ne $false" in source
    assert "groups_may_cross_splits -ne $false" in source
    assert "Teacher receipt contract SHA/size binding failed" in source
    assert "Teacher artifact SHA/size readback failed" in source
    assert "Publication root contains a retained sibling or unexpected member" in source
    assert "otherimages_white_teacher_independent_closure_v1" in source
    assert "canonical_closure_recomputed=$true" in source
    assert "exact_view_ids_paths_sha_size_line_count=$true" in source
    assert "manifest_reject_contract_receipt_artifacts_bound=$true" in source
    assert "otherimages_white_teacher_windows_pipeline_receipt_v1" in source
    assert "training_performed=$false" in source
    assert "WHITE_TEACHER_PIPELINE_OK" in source
    assert "WHITE_TEACHER_PIPELINE_FAILED" in source


def test_wrapper_only_runs_capture_then_offline_teacher_and_never_trains() -> None:
    source = _source()

    capture = source.index("Invoke-PythonStage 'capture-three-view'")
    teacher = source.index("Invoke-PythonStage 'teacher-consensus'")
    independent = source.index("Invoke-PythonStage 'teacher-independent-verify'")
    assert capture < teacher < independent
    assert "scripts\\otherimages-paddle-capture.py" in source
    assert "scripts\\otherimages-paddle-teacher.py" in source
    assert "--view-result" in source
    assert "training_performed=$false" in source
    for forbidden in ("Remove-Item", "Copy-Item", "Move-Item", "Start-Job"):
        assert forbidden not in source


def _independent_verifier_source() -> str:
    source = _source()
    start = source.index("$TeacherVerifierSource = @'") + len("$TeacherVerifierSource = @'\n")
    end = source.index("\n'@\n", start)
    return source[start:end]


def _binding(path: Path, *, public: bool) -> dict[str, object]:
    data = path.read_bytes()
    return {
        "path": str(path.resolve()) if public else path.name,
        "sha256": hashlib.sha256(data).hexdigest(),
        "size_bytes": len(data),
        "line_count": data.count(b"\n"),
    }


def _write_independent_fixture(root: Path) -> tuple[Path, Path, Path, Path]:
    capture = root / "capture"
    teacher = root / "teacher"
    capture.mkdir(parents=True)
    teacher.mkdir()
    views = ("original_rgb", "grayscale_clahe", "upscale_sharpen")
    capture_views: list[dict[str, object]] = []
    contract_views: list[dict[str, object]] = []
    for view_id in views:
        path = capture / f"{view_id}.jsonl"
        path.write_text("{}\n" * 999, encoding="utf-8")
        binding = _binding(path, public=True)
        capture_views.append({"view_id": view_id, **binding})
        public = dict(binding)
        public.pop("line_count")
        contract_views.append({"view_id": view_id, "result": public})
    capture_receipt = root / "capture.receipt.json"
    capture_receipt.write_text(
        json.dumps(
            {
                "kind": "otherimages_paddle_three_view_capture_receipt_v2",
                "output_directory": str(capture.resolve()),
                "views": capture_views,
            }
        ),
        encoding="utf-8",
    )
    manifest = teacher / "teacher_manifest.jsonl"
    reject = teacher / "reject_manifest.jsonl"
    manifest.write_text("{}\n", encoding="utf-8")
    reject.write_text("{}\n" * 999, encoding="utf-8")
    artifacts = [_binding(manifest, public=False), _binding(reject, public=False)]
    closure = {
        "schema_version": 1,
        "inputs": {"views": contract_views},
        "configuration": {},
        "counts": {"accepted_teacher_records": 1, "quarantined_records": 999},
        "split_use": {},
        "artifacts": artifacts,
    }
    closure_sha = hashlib.sha256(
        json.dumps(closure, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    contract = {
        **closure,
        "kind": "otherimages_paddle_teacher_contract_v1",
        "output_directory": str(teacher.resolve()),
        "closure_sha256": closure_sha,
    }
    contract_path = teacher / "teacher.contract.json"
    contract_path.write_text(json.dumps(contract), encoding="utf-8")
    receipt_path = teacher / "teacher.receipt.json"
    receipt_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "kind": "otherimages_paddle_teacher_receipt_v1",
                "sealed": True,
                "contract": _binding(contract_path, public=False),
                "contract_closure_sha256": closure_sha,
            }
        ),
        encoding="utf-8",
    )
    verifier = root / "verify.py"
    verifier.write_text(_independent_verifier_source(), encoding="utf-8")
    return verifier, teacher, capture, capture_receipt


def _run_verifier(paths: tuple[Path, Path, Path, Path]) -> subprocess.CompletedProcess[str]:
    verifier, teacher, capture, receipt = paths
    return subprocess.run(
        [
            sys.executable,
            str(verifier),
            "--teacher-root",
            str(teacher),
            "--capture-root",
            str(capture),
            "--capture-receipt",
            str(receipt),
        ],
        text=True,
        capture_output=True,
        check=False,
    )


def test_independent_teacher_verifier_accepts_exact_closure(tmp_path: Path) -> None:
    result = _run_verifier(_write_independent_fixture(tmp_path))
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["kind"] == "otherimages_white_teacher_independent_closure_v1"
    assert payload["accepted"] + payload["quarantined"] == 1000


def test_independent_teacher_verifier_rejects_capture_line_count_attack(tmp_path: Path) -> None:
    paths = _write_independent_fixture(tmp_path)
    receipt = json.loads(paths[3].read_text(encoding="utf-8"))
    receipt["views"][0]["line_count"] = 998
    paths[3].write_text(json.dumps(receipt), encoding="utf-8")
    result = _run_verifier(paths)
    assert result.returncode != 0
    assert "line_count differs" in result.stderr


def test_independent_teacher_verifier_rejects_contract_view_path_attack(tmp_path: Path) -> None:
    paths = _write_independent_fixture(tmp_path)
    contract_path = paths[1] / "teacher.contract.json"
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    contract["inputs"]["views"][0]["result"]["path"] = str(tmp_path / "attacker.jsonl")
    closure = {key: contract[key] for key in ("schema_version", "inputs", "configuration", "counts", "split_use", "artifacts")}
    contract["closure_sha256"] = hashlib.sha256(
        json.dumps(closure, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    contract_path.write_text(json.dumps(contract), encoding="utf-8")
    receipt_path = paths[1] / "teacher.receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["contract"] = _binding(contract_path, public=False)
    receipt["contract_closure_sha256"] = contract["closure_sha256"]
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    result = _run_verifier(paths)
    assert result.returncode != 0
    assert "view result path/SHA/size differs" in result.stderr
