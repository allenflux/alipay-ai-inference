[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$AuditBundle,
    [Parameter(Mandatory = $true)]
    [string]$ValEvidenceDirectory,
    [Parameter(Mandatory = $true)]
    [string]$DeliveryDirectory,
    [Parameter(Mandatory = $true)]
    [string]$TrustedManifestSha256,
    [ValidateRange(200, 100000)]
    [int]$ParitySamples = 200,
    [string]$WorkDirectory,
    [string]$DotnetExe
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$expectedFullValRecords = 6789

function Get-Sha256([string]$Path) {
    return (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()
}

$repositoryRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")).Path
$pythonExe = Join-Path $repositoryRoot ".venv-cu126\Scripts\python.exe"
$project = Join-Path $repositoryRoot "dotnet\ReceiptMlNet.Cli\ReceiptMlNet.Cli.csproj"
$parserContractProject = Join-Path $repositoryRoot (
    "dotnet\ReceiptMlNet.Cli.PaddleRecipientContractTests\ReceiptMlNet.Cli.PaddleRecipientContractTests.csproj"
)
$dotnetParityProject = Join-Path $repositoryRoot (
    "dotnet\ReceiptMlNet.Cli.PaddleParity\ReceiptMlNet.Cli.PaddleParity.csproj"
)
$sampleBuilder = Join-Path $PSScriptRoot "receipt-ppocr-val-parity-sample.py"
$dotnetParityComparator = Join-Path $PSScriptRoot "receipt-ppocr-dotnet-parity.py"

if ([string]::IsNullOrWhiteSpace($DotnetExe)) {
    $dotnetCommand = Get-Command dotnet -ErrorAction SilentlyContinue
    if ($null -ne $dotnetCommand) {
        $DotnetExe = $dotnetCommand.Source
    } else {
        $portableDotnet = Join-Path $repositoryRoot "artifacts\dotnet8\dotnet.exe"
        if (Test-Path -LiteralPath $portableDotnet -PathType Leaf) {
            $DotnetExe = $portableDotnet
        }
    }
}
if ([string]::IsNullOrWhiteSpace($DotnetExe)) {
    throw "Missing .NET 8 host. Install dotnet, place the portable host at artifacts\dotnet8\dotnet.exe, or pass -DotnetExe."
}
$DotnetExe = [IO.Path]::GetFullPath($DotnetExe)

$AuditBundle = [IO.Path]::GetFullPath($AuditBundle)
$ValEvidenceDirectory = [IO.Path]::GetFullPath($ValEvidenceDirectory)
$DeliveryDirectory = [IO.Path]::GetFullPath($DeliveryDirectory)
$TrustedManifestSha256 = $TrustedManifestSha256.Trim().ToLowerInvariant()
if ($TrustedManifestSha256 -notmatch '^[0-9a-f]{64}$') {
    throw "TrustedManifestSha256 must be a lowercase 64-character SHA-256."
}
if ([string]::IsNullOrWhiteSpace($WorkDirectory)) {
    $WorkDirectory = Join-Path (Split-Path -Parent $DeliveryDirectory) (
        "ppocrv4-recipient-onnx-work-" + (Get-Date -Format "yyyyMMdd-HHmmss")
    )
}
$WorkDirectory = [IO.Path]::GetFullPath($WorkDirectory)

foreach ($required in @(
    @{ Name = "project Python"; Path = $pythonExe; Kind = "Leaf" },
    @{ Name = ".NET host"; Path = $DotnetExe; Kind = "Leaf" },
    @{ Name = "PP-OCR audit bundle"; Path = $AuditBundle; Kind = "Container" },
    @{ Name = "full val evidence"; Path = $ValEvidenceDirectory; Kind = "Container" },
    @{ Name = "val parity sample builder"; Path = $sampleBuilder; Kind = "Leaf" },
    @{ Name = ".NET parity comparator"; Path = $dotnetParityComparator; Kind = "Leaf" },
    @{ Name = ".NET project"; Path = $project; Kind = "Leaf" },
    @{ Name = "recipient parser contract project"; Path = $parserContractProject; Kind = "Leaf" },
    @{ Name = "production C# Paddle parity project"; Path = $dotnetParityProject; Kind = "Leaf" }
)) {
    if (-not (Test-Path -LiteralPath $required.Path -PathType $required.Kind)) {
        throw "Missing $($required.Name): $($required.Path)"
    }
}
foreach ($fresh in @($DeliveryDirectory, $WorkDirectory)) {
    if (Test-Path -LiteralPath $fresh) {
        throw "Refusing to reuse PP-OCR conversion output: $fresh"
    }
}

$auditContractPath = Join-Path $AuditBundle "paddle_ocr_bundle.contract.json"
$valSummaryPath = Join-Path $ValEvidenceDirectory "summary.json"
$valComparisonsPath = Join-Path $ValEvidenceDirectory "comparisons.jsonl"
foreach ($requiredFile in @($auditContractPath, $valSummaryPath, $valComparisonsPath)) {
    if (-not (Test-Path -LiteralPath $requiredFile -PathType Leaf)) {
        throw "Missing required PP-OCR evidence file: $requiredFile"
    }
}
$auditContract = Get-Content -LiteralPath $auditContractPath -Raw -Encoding UTF8 | ConvertFrom-Json
$valSummary = Get-Content -LiteralPath $valSummaryPath -Raw -Encoding UTF8 | ConvertFrom-Json
$valPropertyNames = @($valSummary.PSObject.Properties.Name)
if (
    [int]$valSummary.schema_version -ne 1 `
    -or [string]$valSummary.kind -ne "receipt_paddle_recipient_teacher_parity_v1" `
    -or [string]$valSummary.evaluation_split -ne "val" `
    -or $valPropertyNames -notcontains "limit" `
    -or $null -ne $valSummary.limit `
    -or [int]$valSummary.records -ne $expectedFullValRecords `
    -or [string]$valSummary.inference_mode.name -ne "full_det_cls_rec" `
    -or $valSummary.inference_mode.experimental -ne $false `
    -or $valSummary.inference_mode.detection_enabled -ne $true `
    -or $valSummary.inference_mode.angle_classifier_enabled -ne $true `
    -or $valSummary.inference_mode.recognizer_enabled -ne $true `
    -or -not ([string]$valSummary.requested_device).StartsWith("cuda") `
    -or -not ([string]$valSummary.runtime.active_paddle_device).StartsWith("gpu") `
    -or $valSummary.acceptance.passed -ne $true `
    -or [double]$valSummary.acceptance.target_anchored_value_exact_match -lt 0.90 `
    -or [double]$valSummary.anchored_value_exact_match -lt 0.90
) {
    throw "PP-OCR evidence is not the complete, unbounded 6789-record CUDA full det/cls/rec val run. Pilot evidence is rejected."
}

$auditContractSha256 = Get-Sha256 $auditContractPath
$auditBundleFromEvidence = [IO.Path]::GetFullPath([string]$valSummary.frozen_bundle.path)
if (
    [int]$auditContract.schema_version -ne 1 `
    -or [string]$auditContract.kind -ne "paddle_ocr_v2_bundle" `
    -or @($auditContract.onnx.PSObject.Properties.Name).Count -ne 3 `
    -or @($auditContract.onnx.PSObject.Properties.Name) -notcontains "det" `
    -or @($auditContract.onnx.PSObject.Properties.Name) -notcontains "rec" `
    -or @($auditContract.onnx.PSObject.Properties.Name) -notcontains "cls" `
    -or -not $auditBundleFromEvidence.Equals($AuditBundle, [StringComparison]::OrdinalIgnoreCase) `
    -or [string]$valSummary.frozen_bundle.contract_kind -ne "paddle_ocr_v2_bundle" `
    -or [string]$valSummary.frozen_bundle.contract_sha256 -ne $auditContractSha256 `
    -or [string]$valSummary.frozen_bundle.native_asset_identity_sha256 -ne [string]$auditContract.native_asset_identity.sha256 `
    -or $valSummary.frozen_bundle.native_component_sha256.det -ne $auditContract.native_asset_identity.components.det `
    -or $valSummary.frozen_bundle.native_component_sha256.rec -ne $auditContract.native_asset_identity.components.rec `
    -or $valSummary.frozen_bundle.native_component_sha256.cls -ne $auditContract.native_asset_identity.components.cls `
    -or $valSummary.frozen_bundle.native_component_sha256.dictionary -ne $auditContract.native_asset_identity.components.dictionary `
    -or $valSummary.frozen_bundle.live_source_bytes_verified -ne $true `
    -or $valSummary.frozen_bundle.verified_before_and_after -ne $true
) {
    throw "Full val evidence is not cryptographically bound to this exact exported audit bundle and its native source bytes."
}
if ([string]$valSummary.comparisons_sha256 -ne (Get-Sha256 $valComparisonsPath)) {
    throw "Full val comparisons SHA-256 differs from summary."
}
$manifestPath = [IO.Path]::GetFullPath([string]$valSummary.manifest)
if (
    -not (Test-Path -LiteralPath $manifestPath -PathType Leaf) `
    -or [string]$valSummary.manifest_sha256 -ne $TrustedManifestSha256 `
    -or (Get-Sha256 $manifestPath) -ne $TrustedManifestSha256
) {
    throw "Full val evidence does not match TrustedManifestSha256."
}

$deliveryParent = Split-Path -Parent $DeliveryDirectory
if (-not (Test-Path -LiteralPath $deliveryParent -PathType Container)) {
    New-Item -ItemType Directory -Path $deliveryParent | Out-Null
}
$deliveryLeaf = Split-Path -Leaf $DeliveryDirectory
$deliveryStage = Join-Path $deliveryParent ("." + $deliveryLeaf + ".stage-" + [Guid]::NewGuid().ToString("N"))
$paritySample = Join-Path $WorkDirectory "val-parity-sample"
$wrapperParityReport = Join-Path $WorkDirectory "native-onnx-parity"
$dotnetParityReport = Join-Path $WorkDirectory "dotnet-cpu-parity"
$dotnetComparisonReport = Join-Path $WorkDirectory "dotnet-wrapper-parity"

Write-Host "receipt_ocr_ppocrv4_recipient_onnx_package"
Write-Host "  val=$($valSummary.anchored_value_exact_matches)/$($valSummary.records)=$([double]$valSummary.anchored_value_exact_match)"
Write-Host "  audit-contract-sha256=$auditContractSha256"
Write-Host "  native-identity-sha256=$($auditContract.native_asset_identity.sha256)"
Write-Host "  manifest-sha256=$TrustedManifestSha256"
Write-Host "  stages=det + angle-cls + SVTR_LCNet rec"
Write-Host "  output=pure ONNX + dictionary + hash/preprocess contract"
Write-Host "  publication=hidden same-volume stage, then one atomic rename after every gate"

New-Item -ItemType Directory -Path $WorkDirectory | Out-Null
$previousPythonPath = $env:PYTHONPATH
$published = $false
try {
    $env:PYTHONPATH = Join-Path $repositoryRoot "src"
    & $pythonExe -m transfer_receipt_ai.paddle_ocr_bundle verify `
        --bundle $AuditBundle `
        --require-onnx
    if ($LASTEXITCODE -ne 0) {
        throw "PP-OCR exported audit verification failed with exit code $LASTEXITCODE"
    }

    & $pythonExe $sampleBuilder `
        --evidence $ValEvidenceDirectory `
        --output $paritySample `
        --limit $ParitySamples `
        --audit-bundle $AuditBundle `
        --trusted-manifest-sha256 $TrustedManifestSha256 `
        --expected-records $expectedFullValRecords
    if ($LASTEXITCODE -ne 0) {
        throw "Could not validate full evidence/create val-only parity sample; exit code $LASTEXITCODE"
    }

    & $pythonExe -m transfer_receipt_ai.paddle_ocr_bundle validate-onnx `
        --bundle $AuditBundle `
        --input (Join-Path $paritySample "images") `
        --output $wrapperParityReport `
        --min-text-exact-match 1.0 `
        --max-confidence-delta 0.01
    if ($LASTEXITCODE -ne 0) {
        throw "Native-frozen/ONNX PP-OCR parity failed with exit code $LASTEXITCODE"
    }

    & $DotnetExe run --project $parserContractProject -c Release
    if ($LASTEXITCODE -ne 0) {
        throw "PP-OCR recipient parser contract tests failed with exit code $LASTEXITCODE"
    }
    & $DotnetExe build $project -c Release "-p:OnnxRuntimeFlavor=cpu"
    if ($LASTEXITCODE -ne 0) {
        throw "Hybrid ML.NET CPU build failed with exit code $LASTEXITCODE"
    }

    & $pythonExe -m transfer_receipt_ai.paddle_ocr_bundle package-delivery `
        --bundle $AuditBundle `
        --output $deliveryStage
    if ($LASTEXITCODE -ne 0) {
        throw "PP-OCR ONNX staging package failed with exit code $LASTEXITCODE"
    }
    & $pythonExe -m transfer_receipt_ai.paddle_ocr_bundle verify-delivery --delivery $deliveryStage
    if ($LASTEXITCODE -ne 0) {
        throw "PP-OCR ONNX staging verification failed with exit code $LASTEXITCODE"
    }

    & $DotnetExe run --project $dotnetParityProject -c Release "-p:OnnxRuntimeFlavor=cpu" -- `
        --bundle $deliveryStage `
        --input (Join-Path $paritySample "images") `
        --output $dotnetParityReport
    if ($LASTEXITCODE -ne 0) {
        throw "Production C# PP-OCR CPU parity execution failed with exit code $LASTEXITCODE"
    }
    & $pythonExe $dotnetParityComparator `
        --wrapper $wrapperParityReport `
        --dotnet $dotnetParityReport `
        --delivery $deliveryStage `
        --output $dotnetComparisonReport
    if ($LASTEXITCODE -ne 0) {
        throw "Production C# PP-OCR exact parity gate failed with exit code $LASTEXITCODE"
    }

    & $pythonExe -m transfer_receipt_ai.paddle_ocr_bundle verify-delivery --delivery $deliveryStage
    if ($LASTEXITCODE -ne 0) {
        throw "Final staging verification failed with exit code $LASTEXITCODE"
    }
    Move-Item -LiteralPath $deliveryStage -Destination $DeliveryDirectory
    $published = $true
}
finally {
    $env:PYTHONPATH = $previousPythonPath
    if (-not $published -and (Test-Path -LiteralPath $deliveryStage)) {
        Remove-Item -LiteralPath $deliveryStage -Recurse -Force
    }
}

$wrapperParity = Get-Content -LiteralPath (Join-Path $wrapperParityReport "summary.json") -Raw -Encoding UTF8 | ConvertFrom-Json
$dotnetParity = Get-Content -LiteralPath (Join-Path $dotnetComparisonReport "summary.json") -Raw -Encoding UTF8 | ConvertFrom-Json
Write-Host ""
Write-Host "PP-OCRv4 RECIPIENT ONNX PACKAGE: PASS" -ForegroundColor Green
Write-Host ("  native/ONNX exact: {0:P2} ({1} val crops)" -f [double]$wrapperParity.text_exact_match, [int]$wrapperParity.records)
Write-Host ("  C#/ONNX exact: {0:P2} ({1} val crops)" -f [double]$dotnetParity.exact_match, [int]$dotnetParity.records)
Write-Host ("  C# CPU parity p50/p95: {0:N2}/{1:N2} ms" -f [double]$dotnetParity.dotnet_latency_ms.p50, [double]$dotnetParity.dotnet_latency_ms.p95)
Write-Host "  delivery=$DeliveryDirectory"
Write-Host "  evidence=$WorkDirectory"
Write-Host "Next: run receipt-mlnet-hybrid-recipient-cpu-ab.ps1 in pilot mode, then a separate full formal mode."
