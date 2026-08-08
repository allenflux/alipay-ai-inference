[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$PilotDirectory,
    [Parameter(Mandatory = $true)]
    [string]$UnifiedModel,
    [Parameter(Mandatory = $true)]
    [string]$PaddleDeliveryBundle,
    [Parameter(Mandatory = $true)]
    [string]$OutputDirectory,
    [string]$DotnetExe
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$repositoryRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")).Path
if ([string]::IsNullOrWhiteSpace($DotnetExe)) {
    $DotnetExe = Join-Path $repositoryRoot "artifacts\dotnet8\dotnet.exe"
}

$project = Join-Path $repositoryRoot "dotnet\ReceiptMlNet.Cli\ReceiptMlNet.Cli.csproj"
$detector = Join-Path $repositoryRoot "artifacts\receipt_lrcnn_v1.onnx"
$deviceModel = Join-Path $repositoryRoot "artifacts\statusbar_device_v1.onnx"
$comparison = Join-Path ([IO.Path]::GetFullPath($PilotDirectory)) "comparison\comparisons.jsonl"
$OutputDirectory = [IO.Path]::GetFullPath($OutputDirectory)
$inputList = "$OutputDirectory.inputs.txt"
$cliApp = "$OutputDirectory.cli-app"
$cliAssembly = Join-Path $cliApp "ReceiptMlNet.Cli.dll"

function Get-OptionalProperty($Object, [string]$Name) {
    $property = $Object.PSObject.Properties[$Name]
    if ($null -eq $property) {
        return $null
    }
    return $property.Value
}

foreach ($required in @(
    @{ Name = ".NET host"; Path = $DotnetExe; Kind = "Leaf" },
    @{ Name = "ML.NET project"; Path = $project; Kind = "Leaf" },
    @{ Name = "pilot comparison"; Path = $comparison; Kind = "Leaf" },
    @{ Name = "receipt detector"; Path = $detector; Kind = "Leaf" },
    @{ Name = "device classifier"; Path = $deviceModel; Kind = "Leaf" },
    @{ Name = "unified OCR model"; Path = $UnifiedModel; Kind = "Leaf" },
    @{ Name = "Paddle OCR delivery"; Path = $PaddleDeliveryBundle; Kind = "Container" }
)) {
    if (-not (Test-Path -LiteralPath $required.Path -PathType $required.Kind)) {
        throw "Missing $($required.Name): $($required.Path)"
    }
}
foreach ($fresh in @($OutputDirectory, $inputList, $cliApp)) {
    if (Test-Path -LiteralPath $fresh) {
        throw "Refusing to overwrite replay evidence: $fresh"
    }
}

$failedSources = @(
    Get-Content -LiteralPath $comparison -Encoding UTF8 |
        Where-Object { -not [string]::IsNullOrWhiteSpace($_) } |
        ForEach-Object { $_ | ConvertFrom-Json } |
        Where-Object { $_.invariant -eq $false } |
        ForEach-Object { [string]$_.source }
)
if ($failedSources.Count -le 0) {
    throw "Pilot comparison contains no failed sources to replay."
}
if (@($failedSources | Sort-Object -Unique).Count -ne $failedSources.Count) {
    throw "Pilot comparison contains duplicate failed sources."
}
foreach ($source in $failedSources) {
    if (-not (Test-Path -LiteralPath $source -PathType Leaf)) {
        throw "Missing failed source image: $source"
    }
}

$outputParent = Split-Path -Parent $OutputDirectory
New-Item -ItemType Directory -Path $outputParent -Force | Out-Null
[IO.File]::WriteAllLines(
    $inputList,
    [string[]]$failedSources,
    (New-Object Text.UTF8Encoding($false)))

& $DotnetExe publish $project `
    -c Release `
    -r win-x64 `
    --self-contained false `
    "-p:OnnxRuntimeFlavor=cpu" `
    -o $cliApp
if ($LASTEXITCODE -ne 0) {
    throw "Could not publish replay CLI; exit code $LASTEXITCODE"
}

& $DotnetExe $cliAssembly `
    --detector $detector `
    --device-model $deviceModel `
    --ocr hybrid-recipient `
    --ocr-model $UnifiedModel `
    --ocr-bundle $PaddleDeliveryBundle `
    --input-list $inputList `
    --output $OutputDirectory `
    --device cpu `
    --rectification max-side-1600 `
    --annotate none
if ($LASTEXITCODE -ne 0) {
    throw "Hybrid failed-source replay exited with code $LASTEXITCODE"
}

$manifest = Get-Content -LiteralPath (Join-Path $OutputDirectory "inference_manifest.json") -Raw -Encoding UTF8 | ConvertFrom-Json
$evidence = foreach ($row in @($manifest)) {
    $result = Get-Content -LiteralPath ([string]$row.result) -Raw -Encoding UTF8 | ConvertFrom-Json
    $recipient = $result.fields.recipient
    [pscustomobject][ordered]@{
        source = [string]$row.source
        inference_ms = [double]$row.inference_ms
        state = [string]$recipient.state
        candidate = Get-OptionalProperty $recipient "candidate"
        route = Get-OptionalProperty $recipient "hybrid_ocr_route"
        failure_reason = Get-OptionalProperty $recipient "hybrid_ocr_failure_reason"
        first_raw = Get-OptionalProperty $recipient "hybrid_ocr_first_raw"
        retry_raw = Get-OptionalProperty $recipient "hybrid_ocr_retry_raw"
        third_route = Get-OptionalProperty $recipient "hybrid_ocr_third_route"
        right_value_raw = Get-OptionalProperty $recipient "hybrid_ocr_right_value_raw"
        right_value_line_count = Get-OptionalProperty $recipient "hybrid_ocr_right_value_line_count"
        right_value_crop_width = Get-OptionalProperty $recipient "hybrid_ocr_right_value_crop_width"
        right_value_crop_height = Get-OptionalProperty $recipient "hybrid_ocr_right_value_crop_height"
        right_value_line_confidences = Get-OptionalProperty $recipient "hybrid_ocr_right_value_line_confidences"
    }
}

$evidenceCount = @($evidence).Count
Write-Host "HYBRID FAILED-SOURCE CPU REPLAY"
Write-Host ("  records={0}; output={1}" -f $evidenceCount, $OutputDirectory)
$evidence | ConvertTo-Json -Depth 6 -Compress
