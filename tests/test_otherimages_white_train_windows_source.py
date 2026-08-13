from __future__ import annotations

from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "otherimages-white-train-windows.ps1"


def _source() -> str:
    return SCRIPT.read_text(encoding="utf-8")


def test_fixed_commit_tree_clone_teacher_and_code_authority() -> None:
    source = _source()
    assert "3080a692a37d7efb0f926cce46de831d17f0e4db" in source
    assert "fb7a21f99139edd15eb1bb10e311039ebe28ebf5" in source
    assert "C:\\f3-white-code-3080a69" in source
    assert "D:\\alipay-ai-data\\alipay-ai-inference\\.venv-cu126\\Scripts\\python.exe" in source
    assert "$info.WorkingDirectory=$RepoRoot" in source
    assert "$info.EnvironmentVariables['PYTHONPATH']=(Join-Path $RepoRoot 'src')" in source
    assert "$info.EnvironmentVariables['PYTHONDONTWRITEBYTECODE']='1'" in source
    for name in ("PYTHONHOME", "PYTHONSTARTUP", "PYTHONINSPECT", "PYTEST_ADDOPTS", "PYTEST_PLUGINS"):
        assert name in source
    assert "otherimages_white_train_import_attestation_v1" in source
    assert "Path(module.__file__).resolve(strict=True)" in source
    assert "source_only_from_fixed_clone=$true" in source
    assert "C:\\f3-white-teacher-3080a69-pilot1000-a\\publications\\paddle-teacher-consensus" in source
    assert "C:\\Program Files\\Git\\cmd\\git.exe" in source
    assert "C:\\Program Files\\NVIDIA Corporation\\NVSMI\\nvidia-smi.exe" in source
    assert "status --porcelain=v1 --untracked-files=all" in source
    assert "HEAD^{tree}" in source
    assert "84192d84b5b57c434bc93b97f1a752b37b5585793e537f101ac106e19d72aa28" in source
    assert "c84e066aceb8c79779118bb404638092ef9b5e576fa5107e3c4cd9c860ee749b" in source
    assert "RequiredCode" in source and "Get-CodeBindings" in source
    assert "$gitBefore=Get-Binding $GitExe" in source
    assert "Assert-GitAuthority $gitBefore" in source
    assert source.index("$gitBefore=Get-Binding $GitExe") < source.index("Assert-GitAuthority $gitBefore")
    assert "fixed Git executable before HEAD query" in source
    assert "fixed Git executable after diff query" in source
    assert source.count("fixed Git executable before ") == 4
    assert source.count("fixed Git executable after ") == 4


def test_teacher_seal_artifacts_and_every_accepted_split_are_preflight_gates() -> None:
    source = _source()
    assert "Assert-TeacherAuthority" in source
    assert "Teacher publication" in source
    assert "otherimages_paddle_teacher_contract_v1" in source
    assert "otherimages_paddle_teacher_receipt_v1" in source
    assert "contract.output_directory" in source
    assert "contract.artifacts).Count -ne 2" in source
    assert "reject_manifest.jsonl|teacher_manifest.jsonl" in source
    assert "accepted_by_split.train" in source
    assert "accepted_by_split.val" in source
    assert "accepted_by_split.test" in source
    assert "Teacher must contain accepted train, val, and test records" in source
    assert "accepted+$quarantined" in source
    assert "TeacherVerifierSource" in source
    assert "teacher canonical closure differs" in source
    assert "otherimages_white_train_teacher_independent_closure_v1" in source
    assert "canonical_closure_recomputed=$true" in source


def test_line_dataset_uses_actual_authorization_cli_and_checks_seal() -> None:
    source = _source()
    assert "scripts\\otherimages-line-dataset.py" in source
    assert "'--teacher',$TeacherRoot,'--output',$LineDataset,'--authorize-training'" in source
    assert "otherimages_generic_text_line_dataset_contract_v1" in source
    assert "otherimages_generic_text_line_dataset_receipt_v1" in source
    assert "explicit_materializer_flag" in source
    assert "teacher_parity_only_not_independent_business_truth" in source
    assert "generic_test_oov_fail_closed_by_source=$true" in source


def test_student_cli_is_exact_cuda_15_epoch_accelerated_configuration() -> None:
    source = _source()
    assert "'-m','transfer_receipt_ai.ocr_train'" in source
    assert "'--fields','generic_text_line'" in source
    assert "'--device','cuda:0'" in source
    assert "'--epochs','15','--batch-size','128'" in source
    assert "'--num-workers','4','--persistent-workers','--prefetch-factor','4'" in source
    assert "'--cuda-tf32','--cudnn-benchmark','--validation-every','3'" in source
    assert "'--onnx-output',$OnnxPath" in source
    assert "3,6,9,12,15" in source
    assert "training_history.json" in source


def test_gpu_and_exact8_concurrency_are_fail_closed_before_training() -> None:
    source = _source()
    assert "Get-CimInstance Win32_Process" in source
    assert "receipt-ocr-recipient-multiview-exact8|f3e8|exact8" in source
    assert "--query-compute-apps=pid,process_name" in source
    assert "& $NvidiaSmiExe --query-gpu" in source
    assert "& $NvidiaSmiExe --query-compute-apps" in source
    assert "$NvidiaSmiBinding=Get-Binding $NvidiaSmiExe" in source
    assert source.index("$NvidiaSmiBinding=Get-Binding $NvidiaSmiExe") < source.index(
        "Assert-NoConflictingGpuWork 'preflight'"
    )
    assert "& nvidia-smi.exe" not in source
    assert "fixed nvidia-smi immediately before Python stage" in source
    assert "fixed nvidia-smi immediately after Python stage" in source
    assert "fixed nvidia-smi in finalizer after Python stage" in source
    assert "CUDA GPU already has active compute work" in source
    assert "Assert-NoConflictingGpuWork 'preflight'" in source
    assert "Assert-NoConflictingGpuWork 'immediately-before-student-train'" in source


def test_fresh_no_resume_kernel_job_cleanup_and_failure_evidence() -> None:
    source = _source()
    assert "CreateDirectoryW" in source
    assert "RunRoot must be brand-new and is never resumed" in source
    for symbol in (
        "CreateJobObjectW",
        "SetInformationJobObject",
        "JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE",
        "CreateProcessW",
        "CREATE_SUSPENDED",
        "STARTUPINFOEX",
        "EXTENDED_STARTUPINFO_PRESENT",
        "PROC_THREAD_ATTRIBUTE_HANDLE_LIST",
        "InitializeProcThreadAttributeList",
        "UpdateProcThreadAttribute",
        "DeleteProcThreadAttributeList",
        "AssignProcessToJobObject",
        "ResumeThread",
        "QueryInformationJobObject",
        "TerminateJobObject",
        "WaitForJobEmpty",
    ):
        assert symbol in source
    assert "StartSuspendedAssigned" in source
    assert "nested job constraints may forbid safe containment" in source
    assert "forced_job_termination" in source
    assert "finally { if ($null -ne $process) { $process.Dispose() } }" in source
    stage_helper = source[source.index("function Invoke-PythonStage") : source.index("function Complete-Stage")]
    assert stage_helper.index("$process.CloseJob()") < stage_helper.index("[IO.File]::ReadAllText")
    assert stage_helper.index("finally { if ($null -ne $process) { $process.Dispose() } }") > stage_helper.index(
        "Write-StageFailureEvidenceBestEffort"
    )
    assert "$process.WaitForJobEmpty(30000)" in stage_helper
    assert "$process.CloseJob(); $jobHandleClosed=$true" in stage_helper
    assert "stage.failure.json" in source
    assert "stage.failure.fallback.txt" in source
    assert "pipeline.failure.json" in source
    assert "oom_detected" in source
    assert "resume_or_reuse_allowed=$false" in source
    for forbidden in ("Remove-Item", "Move-Item", "Copy-Item", "Start-Job"):
        assert forbidden not in source


def test_suspended_root_is_assigned_before_resume_without_pid_based_kill_races() -> None:
    source = _source()
    native = source[source.index("public sealed class WhiteTrainNativeJobProcessV1") : source.index("function Get-DirectoryIdentity")]
    launch = native[native.index("StartSuspendedAssigned") : native.index("private static long ToLong")]
    assert launch.index("CreateProcessW(applicationName") < launch.index("AssignProcessToJobObject(job,processInfo.hProcess)")
    assert launch.index("AssignProcessToJobObject(job,processInfo.hProcess)") < launch.index("accounting.ActiveProcesses!=1")
    assert launch.index("accounting.ActiveProcesses!=1") < launch.index("ResumeThread(processInfo.hThread)")
    assert "TerminateProcess(processInfo.hProcess,254)" in launch
    assert "CREATE_SUSPENDED|CREATE_UNICODE_ENVIRONMENT|CREATE_NO_WINDOW" in launch
    assert "|EXTENDED_STARTUPINFO_PRESENT" in launch
    assert "Marshal.WriteIntPtr(handleList,0*IntPtr.Size,stdinHandle)" in launch
    assert "Marshal.WriteIntPtr(handleList,1*IntPtr.Size,stdoutHandle)" in launch
    assert "Marshal.WriteIntPtr(handleList,2*IntPtr.Size,stderrHandle)" in launch
    assert "UpdateProcThreadAttribute(attributeList,0,PROC_THREAD_ATTRIBUTE_HANDLE_LIST" in launch
    assert launch.index("UpdateProcThreadAttribute(attributeList") < launch.index("CreateProcessW(applicationName")
    assert "if (attributeListInitialized) DeleteProcThreadAttributeList(attributeList)" in launch
    assert native.count("ExactSpelling=true") >= 3
    assert "attributeBytes.ToUInt64()==0" in launch
    stage = source[source.index("function Invoke-PythonStage") : source.index("function Complete-Stage")]
    assert stage.index("StartSuspendedAssigned") < stage.index("while (-not $process.WaitForExit(1000))")
    assert "foreach ($environmentName in" in stage
    assert "foreach ($name in" not in stage
    assert "$process.Start()" not in source
    for forbidden in (
        "Get-CimProcessIdentity",
        "ParentProcessId",
        "Stop-Process -Id",
        "Stop-ProcessTree",
        "Wait-ProcessIdsAbsent",
        "observed_descendant_pids",
    ):
        assert forbidden not in source
    assert "New-Object Collections.Generic.List" not in source
    assert "New-Object 'Collections.Generic.List" not in source
    assert "New-Object 'Collections.Generic.HashSet" not in source
    assert "[Collections.Generic.List" not in source
    assert "[Collections.Generic.HashSet" not in source
    assert "Collections.ArrayList" not in source


def test_stream_or_evidence_failure_still_has_create_new_fallback_failure_evidence() -> None:
    source = _source()
    helper = source[
        source.index("function Write-StageFailureEvidenceBestEffort") : source.index("function Invoke-PythonStage")
    ]
    assert "Write-JsonNew $failurePath" in helper
    assert "stage.failure.fallback.txt" in helper
    assert "otherimages_white_train_windows_stage_failure_fallback_v1" in helper
    assert "[IO.FileMode]::CreateNew" in helper
    stage = source[source.index("function Invoke-PythonStage") : source.index("function Complete-Stage")]
    assert "$evidenceError=$null" in stage
    assert "catch { $evidenceError=$_" in stage
    assert "-or $null -ne $evidenceError" in stage
    assert "Write-StageFailureEvidenceBestEffort" in stage
    assert stage.index("Write-StageFailureEvidenceBestEffort") > stage.index("$process.CloseJob()")
    assert stage.index("Write-StageFailureEvidenceBestEffort") > stage.index("[IO.File]::ReadAllText")


def test_rc_stderr_output_and_onnx_closure_are_strict() -> None:
    source = _source()
    assert "Write-RcNew" in source and "Assert-ZeroRc" in source
    assert "$bytes[0] -ne 0x30" in source
    assert "$bytes[1] -ne 0x0d" in source
    assert "$bytes[2] -ne 0x0a" in source
    assert "stderr.size_bytes -ne 0" in source
    assert "stderr_nonempty=$StderrNonEmpty" in source
    stage_helper = source[source.index("function Invoke-PythonStage") : source.index("function Complete-Stage")]
    assert "-or $stderrNonEmpty" in stage_helper
    assert stage_helper.index("-or $stderrNonEmpty") < stage_helper.index(
        "Write-StageFailureEvidenceBestEffort"
    )
    assert "WHITE_TRAIN_STAGE_ALIVE" in source
    assert "Student checkpoint output" in source
    assert "Student analysis candidate" in source
    assert "ONNX/charset SHA binding failed" in source
    assert "ONNX training field count differs" in source
    assert "source_teacher_and_executables_stable=$true" in source


def test_receipt_explicitly_forbids_misrepresenting_candidate_as_cpu_delivery() -> None:
    source = _source()
    assert "otherimages_white_student_training_windows_pipeline_receipt_v1" in source
    assert "training_performed=$true" in source
    assert "generic_text_line_only=$true" in source
    assert "inputs=[ordered]@{teacher_root=$TeacherRoot" in source
    assert "teacher_contract=$teacherBefore.contract" in source
    assert "teacher_contract_closure_sha256=[string]$teacher.contract.closure_sha256" in source
    assert "student_bundle=[ordered]@{root=$StudentBundle;bindings=[ordered]@{model=$onnxBinding;charset=$charsetBinding;contract=$studentContractBinding}" in source
    assert "student_bundle=$StudentBundle" in source
    assert "model=$onnxBinding" in source
    assert "charset=$charsetBinding" in source
    assert "contract=$studentContractBinding" in source
    assert "student_model=$onnxBinding" in source
    assert "student_charset=$charsetBinding" in source
    assert "test_split_oov_zero=$true" in source
    assert "test_split_used_for_training=$false" in source
    assert "train_val_test_closed=$true" in source
    assert "onnx_export_complete=$true" in source
    assert "analysis_candidate_only=$true" in source
    assert "teacher_parity_only=$true" in source
    assert "independent_business_accuracy_proven=$false" in source
    assert "cpu_publication_performed=$false" in source
    assert "cpu_delivery_gate_passed=$false" in source
    assert "test_inference_performed=$false" in source
    assert "WHITE_TRAIN_PIPELINE_OK" in source
