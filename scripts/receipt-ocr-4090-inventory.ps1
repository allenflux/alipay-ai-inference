[CmdletBinding()]
param(
    [string]$TeacherRoot = "D:\alipay-ai-data\receipt-lite-teacher-120k-v1"
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path -LiteralPath $TeacherRoot)) {
    throw "Teacher root does not exist: $TeacherRoot"
}

function Get-ArtifactState([System.IO.DirectoryInfo]$Directory) {
    $hasManifest = Test-Path -LiteralPath (Join-Path $Directory.FullName "unified_fields.jsonl")
    $hasLabels = Test-Path -LiteralPath (Join-Path $Directory.FullName "pseudo_labels.jsonl")
    $hasCheckpoint = Test-Path -LiteralPath (Join-Path $Directory.FullName "best.pt")
    return [pscustomobject]@{
        Name = $Directory.Name
        Manifest = $hasManifest
        Labels = $hasLabels
        BestCheckpoint = $hasCheckpoint
        Modified = $Directory.LastWriteTime.ToString("yyyy-MM-dd HH:mm:ss")
    }
}

$directories = @(Get-ChildItem -LiteralPath $TeacherRoot -Directory | Sort-Object Name)
$relevant = @(
    $directories | Where-Object {
        $_.Name -like "paddle-teacher-labels*" -or
        $_.Name -like "unified-manifest-v12*" -or
        $_.Name -like "unified-run-v12*" -or
        $_.Name -like "unified-eval-v12*"
    }
)

Write-Host "receipt_ocr_4090_inventory"
Write-Host "  teacher_root=$TeacherRoot"
Write-Host ""
Write-Host "directories (Manifest/Labels/BestCheckpoint are direct child files):"
$relevant | ForEach-Object { Get-ArtifactState $_ } | Format-Table -AutoSize

$models = Join-Path $TeacherRoot "models"
if (Test-Path -LiteralPath $models) {
    Write-Host ""
    Write-Host "v12 model files:"
    Get-ChildItem -LiteralPath $models -File -Filter "*v12*.onnx" |
        Sort-Object Name |
        Select-Object Name, @{Name = "MiB"; Expression = { [math]::Round($_.Length / 1MB, 1) }}, LastWriteTime |
        Format-Table -AutoSize
}

Write-Host ""
Write-Host "v12 warm-start checkpoints:"
$warmStarts = @($directories | Where-Object { $_.Name -like "unified-run-v12*" } | ForEach-Object {
    $checkpoint = Join-Path $_.FullName "best.pt"
    if (Test-Path -LiteralPath $checkpoint) {
        [pscustomobject]@{
            Run = $_.Name
            Checkpoint = $checkpoint
            MiB = [math]::Round((Get-Item -LiteralPath $checkpoint).Length / 1MB, 1)
            Modified = (Get-Item -LiteralPath $checkpoint).LastWriteTime.ToString("yyyy-MM-dd HH:mm:ss")
        }
    }
})
if ($warmStarts.Count -eq 0) {
    Write-Host "  none"
}
else {
    $warmStarts | Format-Table -AutoSize
}
