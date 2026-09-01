param(
    [Parameter(Mandatory = $true)][string]$SourceArchive,
    [Parameter(Mandatory = $true)][string]$RunId,
    [string]$Records = "D:\alipay-ai-data\receipt-lite-teacher-120k-v1\paddle-teacher-labels-5field-recipient95-v12-r3-4090-r1\pseudo_labels.jsonl",
    [string]$SharedRoot = "\\tsclient\alipay-ai-inference-temp",
    [string]$PreparedInput = "",
    [int]$Epochs = 5
)

$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"
$Python = "D:\alipay-ai-data\alipay-ai-inference\.venv-cu126\Scripts\python.exe"
$RunRoot = Join-Path "D:\alipay-ai-data\experiments\font-routed-ocr-validation-v1" $RunId
$SourceRoot = Join-Path $RunRoot "source"
$GeneratedPrepared = Join-Path $RunRoot "prepared-resolution-primary"
$Prepared = if ([string]::IsNullOrWhiteSpace($PreparedInput)) { $GeneratedPrepared } else { $PreparedInput }
$Logs = Join-Path $RunRoot "logs"
$ModelsRoot = Join-Path $RunRoot "models"
$EvaluationsRoot = Join-Path $RunRoot "evaluations"
$MergedRoot = Join-Path $RunRoot "merged"
$Summary = Join-Path $RunRoot "routed-ab-summary.json"
$Status = Join-Path $SharedRoot ($RunId + ".status.json")
$SharedSummary = Join-Path $SharedRoot ($RunId + ".routed-ab-summary.json")
$SharedPrepare = Join-Path $SharedRoot ($RunId + ".prepare.json")
$Script:Stage = "starting"
$Script:Detail = ""
$Script:CompletedSteps = New-Object System.Collections.Generic.List[string]
$Script:Underpowered = New-Object System.Collections.Generic.List[string]
$Script:OrtSmokeCompleted = $false
$Script:CanPublishStatus = $true

function Write-Utf8NoBom {
    param([string]$Path, [string]$Text)
    $Encoding = New-Object System.Text.UTF8Encoding($false)
    $Temporary = $Path + ".tmp-" + $PID + "-" + [Guid]::NewGuid().ToString("N")
    try {
        [System.IO.File]::WriteAllText($Temporary, $Text, $Encoding)
        [System.IO.File]::Copy($Temporary, $Path, $true)
    } finally {
        if (Test-Path -LiteralPath $Temporary) {
            [System.IO.File]::Delete($Temporary)
        }
    }
}

function Publish-Status {
    param([bool]$Succeeded)
    $Payload = [ordered]@{
        schema_version = 1
        kind = "receipt_font_routed_ocr_windows_pilot_status_v1"
        run_id = $RunId
        stage = $Script:Stage
        succeeded = $Succeeded
        detail = $Script:Detail
        updated_at_utc = (Get-Date).ToUniversalTime().ToString("o")
        host = $env:COMPUTERNAME
        python = $Python
        epochs = $Epochs
        completed_steps = @($Script:CompletedSteps)
        underpowered_test_slices = @($Script:Underpowered)
        paths = [ordered]@{
            source_archive = $SourceArchive
            records = $Records
            run_root = $RunRoot
            prepared = $Prepared
            prepared_input = $PreparedInput
            summary = $Summary
            shared_summary = $SharedSummary
        }
    }
    Write-Utf8NoBom -Path $Status -Text ($Payload | ConvertTo-Json -Depth 12)
}

function Invoke-PythonStep {
    param(
        [Parameter(Mandatory = $true)][string]$Name,
        [Parameter(Mandatory = $true)][string[]]$Arguments
    )
    $Script:Stage = $Name
    $Script:Detail = "running"
    Publish-Status -Succeeded $false
    $Stdout = Join-Path $Logs ($Name + ".stdout.txt")
    $Stderr = Join-Path $Logs ($Name + ".stderr.txt")
    $PreviousErrorActionPreference = $ErrorActionPreference
    $ExitCode = 1
    try {
        # Windows PowerShell 5.1 wraps native stderr as an ErrorRecord.  Training
        # libraries legitimately emit warnings on stderr, so only the native
        # exit code is authoritative for this step.
        $ErrorActionPreference = "Continue"
        & $Python @Arguments 1> $Stdout 2> $Stderr
        $ExitCode = $LASTEXITCODE
    } finally {
        $ErrorActionPreference = $PreviousErrorActionPreference
    }
    if ($ExitCode -ne 0) {
        $Tail = ""
        if (Test-Path -LiteralPath $Stderr) {
            $Tail = ((Get-Content -LiteralPath $Stderr -Tail 30) -join " | ")
        }
        throw "$Name failed with exit code $ExitCode. $Tail"
    }
    $Script:CompletedSteps.Add($Name)
    $Script:Detail = "completed"
    Publish-Status -Succeeded $false
}

function Model-Configuration {
    param([string]$Field)
    if ($Field -eq "amount") {
        return [ordered]@{ Height = 32; Width = 256; Base = 16; Hidden = 64; Layers = 1 }
    }
    if ($Field -eq "time") {
        return [ordered]@{ Height = 32; Width = 192; Base = 16; Hidden = 64; Layers = 1 }
    }
    return [ordered]@{ Height = 48; Width = 384; Base = 24; Hidden = 64; Layers = 1 }
}

try {
    if ($Epochs -lt 1) {
        throw "Epochs must be positive."
    }
    if (Test-Path -LiteralPath $Status) {
        $Script:CanPublishStatus = $false
        throw "Refusing to overwrite an existing shared output: $Status"
    }
    foreach ($SharedOutput in @($SharedSummary, $SharedPrepare)) {
        if (Test-Path -LiteralPath $SharedOutput) {
            throw "Refusing to overwrite an existing shared output: $SharedOutput"
        }
    }
    Publish-Status -Succeeded $false
    foreach ($Required in @($SourceArchive, $Records, $Python)) {
        if (-not (Test-Path -LiteralPath $Required -PathType Leaf)) {
            throw "Required file is missing: $Required"
        }
    }
    if (Test-Path -LiteralPath $RunRoot) {
        throw "Refusing to overwrite an existing experiment directory: $RunRoot"
    }
    New-Item -ItemType Directory -Path $SourceRoot -Force | Out-Null
    New-Item -ItemType Directory -Path $Logs -Force | Out-Null
    New-Item -ItemType Directory -Path $ModelsRoot -Force | Out-Null
    New-Item -ItemType Directory -Path $EvaluationsRoot -Force | Out-Null
    New-Item -ItemType Directory -Path $MergedRoot -Force | Out-Null
    Expand-Archive -LiteralPath $SourceArchive -DestinationPath $SourceRoot
    $env:PYTHONPATH = Join-Path $SourceRoot "src"
    Invoke-PythonStep -Name "preflight" -Arguments @(
        "-c",
        "import json,sys,torch,onnx,onnxruntime,numpy,cv2,PIL; assert (3,10) <= sys.version_info[:2] <= (3,12); assert torch.cuda.is_available(); providers=onnxruntime.get_available_providers(); assert 'CUDAExecutionProvider' in providers, providers; print(json.dumps({'python':sys.version,'torch':torch.__version__,'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(0),'onnxruntime':onnxruntime.__version__,'providers':providers},sort_keys=True))"
    )
    if ([string]::IsNullOrWhiteSpace($PreparedInput)) {
        Invoke-PythonStep -Name "prepare" -Arguments @(
            "-m", "transfer_receipt_ai.font_routed_ocr_pilot", "prepare",
            "--records", $Records,
            "--output", $Prepared,
            "--fields", "amount,time,payment_method_field",
            "--minimum-device-confidence", "0.90",
            "--allowed-device-sources", "resolution",
            "--split-seed", "font-routed-ocr-pilot-v1",
            "--maximum-train-per-platform-field", "6000",
            "--maximum-validation-per-platform-field", "1000",
            "--maximum-test-per-platform-field", "1500"
        )
    } else {
        $Script:Stage = "prepare-reused"
        $Script:Detail = "validating existing resolution-only prepared manifests"
        Publish-Status -Succeeded $false
        if (-not (Test-Path -LiteralPath $Prepared -PathType Container)) {
            throw "PreparedInput directory is missing: $Prepared"
        }
        $Script:CompletedSteps.Add("prepare-reused")
        $Script:Detail = "completed"
        Publish-Status -Succeeded $false
    }
    $PrepareReportPath = Join-Path $Prepared "prepare.json"
    $PrepareReport = Get-Content -LiteralPath $PrepareReportPath -Raw | ConvertFrom-Json
    if (-not [string]::IsNullOrWhiteSpace($PreparedInput)) {
        $PreparedRecords = [System.IO.Path]::GetFullPath([string]$PrepareReport.input.records)
        $CurrentRecords = [System.IO.Path]::GetFullPath($Records)
        if ($PreparedRecords -ine $CurrentRecords) {
            throw "PreparedInput belongs to a different records file: $PreparedRecords"
        }
    }
    if ($PrepareReport.completed -ne $true `
        -or $PrepareReport.route_independence.time_circularity_controlled -ne $true `
        -or ($PrepareReport.parameters.allowed_device_sources -join ",") -cne "resolution") {
        throw "Prepared dataset did not preserve the resolution-only primary route contract."
    }
    foreach ($Platform in @("ios", "android")) {
        foreach ($Field in @("amount", "time", "payment_method_field")) {
            $Count = [int]$PrepareReport.counts_by_platform_field_split.$Platform.$Field.test
            if ($Count -lt 200) {
                throw "Underpowered fatal slice: $Platform/$Field test=$Count, required>=200."
            }
            if ($Count -lt 1000) {
                $Script:Underpowered.Add("$Platform/$Field=$Count (<1000)")
            }
        }
    }

    $Fields = @("amount", "time", "payment_method_field")
    $ModelNames = @("global", "ios", "android", "random_a", "random_b")
    foreach ($Field in $Fields) {
        $Config = Model-Configuration -Field $Field
        foreach ($ModelName in $ModelNames) {
            $ModelDirectory = Join-Path (Join-Path $ModelsRoot $Field) $ModelName
            $Onnx = Join-Path $ModelDirectory ($ModelName + ".onnx")
            $Manifest = Join-Path $Prepared ($ModelName + ".jsonl")
            Invoke-PythonStep -Name ("train-" + $Field + "-" + $ModelName) -Arguments @(
                "-m", "transfer_receipt_ai.ocr_train",
                "--records", $Manifest,
                "--dataset-root", (Split-Path -Parent $Records),
                "--output", $ModelDirectory,
                "--fields", $Field,
                "--device", "cuda:0",
                "--epochs", [string]$Epochs,
                "--batch-size", "128",
                "--learning-rate", "0.001",
                "--weight-decay", "0.0001",
                "--image-height", [string]$Config.Height,
                "--image-width", [string]$Config.Width,
                "--base-channels", [string]$Config.Base,
                "--hidden-size", [string]$Config.Hidden,
                "--lstm-layers", [string]$Config.Layers,
                "--seed", "42",
                "--num-workers", "0",
                "--validation-every", "1",
                "--onnx-output", $Onnx
            )
            if (-not $Script:OrtSmokeCompleted) {
                Invoke-PythonStep -Name "onnxruntime-cuda-smoke" -Arguments @(
                    "-c",
                    "import json,sys,onnxruntime as ort; s=ort.InferenceSession(sys.argv[1],providers=['CUDAExecutionProvider','CPUExecutionProvider']); assert s.get_providers()[0]=='CUDAExecutionProvider',s.get_providers(); print(json.dumps({'active_providers':s.get_providers()},sort_keys=True))",
                    $Onnx
                )
                $Script:OrtSmokeCompleted = $true
            }
        }
    }

    $EvaluationPlan = @(
        [ordered]@{ Name = "generic_ios"; Model = "global"; Manifest = "ios" },
        [ordered]@{ Name = "routed_ios"; Model = "ios"; Manifest = "ios" },
        [ordered]@{ Name = "wrong_ios"; Model = "android"; Manifest = "ios" },
        [ordered]@{ Name = "generic_android"; Model = "global"; Manifest = "android" },
        [ordered]@{ Name = "routed_android"; Model = "android"; Manifest = "android" },
        [ordered]@{ Name = "wrong_android"; Model = "ios"; Manifest = "android" },
        [ordered]@{ Name = "generic_random_a"; Model = "global"; Manifest = "random_a" },
        [ordered]@{ Name = "routed_random_a"; Model = "random_a"; Manifest = "random_a" },
        [ordered]@{ Name = "generic_random_b"; Model = "global"; Manifest = "random_b" },
        [ordered]@{ Name = "routed_random_b"; Model = "random_b"; Manifest = "random_b" }
    )
    foreach ($Field in $Fields) {
        foreach ($Item in $EvaluationPlan) {
            $Model = Join-Path (Join-Path (Join-Path $ModelsRoot $Field) $Item.Model) ($Item.Model + ".onnx")
            $Manifest = Join-Path $Prepared ($Item.Manifest + ".jsonl")
            $Output = Join-Path (Join-Path $EvaluationsRoot $Field) $Item.Name
            Invoke-PythonStep -Name ("eval-" + $Field + "-" + $Item.Name) -Arguments @(
                "-m", "transfer_receipt_ai.ocr_evaluate",
                "--model", $Model,
                "--records", $Manifest,
                "--dataset-root", (Split-Path -Parent $Records),
                "--output", $Output,
                "--split", "test",
                "--fields", $Field,
                "--training-splits", "train,val",
                "--device", "cuda:0"
            )
        }
    }

    foreach ($Item in $EvaluationPlan) {
        $Inputs = @()
        foreach ($Field in $Fields) {
            $Inputs += Join-Path (Join-Path (Join-Path $EvaluationsRoot $Field) $Item.Name) "comparisons.jsonl"
        }
        $Arguments = @(
            "-m", "transfer_receipt_ai.font_routed_ocr_pilot", "merge-comparisons", "--inputs"
        ) + $Inputs + @("--output", (Join-Path $MergedRoot ($Item.Name + ".jsonl")))
        Invoke-PythonStep -Name ("merge-" + $Item.Name) -Arguments $Arguments
    }

    Invoke-PythonStep -Name "summarize" -Arguments @(
        "-m", "transfer_receipt_ai.font_routed_ocr_pilot", "summarize",
        "--prepare-report", $PrepareReportPath,
        "--generic-ios", (Join-Path $MergedRoot "generic_ios.jsonl"),
        "--routed-ios", (Join-Path $MergedRoot "routed_ios.jsonl"),
        "--wrong-ios", (Join-Path $MergedRoot "wrong_ios.jsonl"),
        "--generic-android", (Join-Path $MergedRoot "generic_android.jsonl"),
        "--routed-android", (Join-Path $MergedRoot "routed_android.jsonl"),
        "--wrong-android", (Join-Path $MergedRoot "wrong_android.jsonl"),
        "--generic-random-a", (Join-Path $MergedRoot "generic_random_a.jsonl"),
        "--routed-random-a", (Join-Path $MergedRoot "routed_random_a.jsonl"),
        "--generic-random-b", (Join-Path $MergedRoot "generic_random_b.jsonl"),
        "--routed-random-b", (Join-Path $MergedRoot "routed_random_b.jsonl"),
        "--output", $Summary
    )
    Copy-Item -LiteralPath $Summary -Destination $SharedSummary
    Copy-Item -LiteralPath $PrepareReportPath -Destination $SharedPrepare
    $Script:Stage = "completed"
    $Script:Detail = "Resolution-only platform experts, wrong-route controls, and random two-expert controls completed."
    Publish-Status -Succeeded $true
    exit 0
} catch {
    $Script:Stage = "failed"
    $Script:Detail = [string]$_.Exception.Message
    if ($Script:CanPublishStatus) {
        try { Publish-Status -Succeeded $false } catch {}
    }
    exit 1
}
