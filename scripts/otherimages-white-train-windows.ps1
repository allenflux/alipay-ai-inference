<#
.SYNOPSIS
Materializes the frozen white teacher into line crops and trains one CUDA CTC analysis candidate.

.DESCRIPTION
Windows PowerShell 5.1 source-only formal wrapper.  It is deliberately fixed to
commit 3080a692a37d7efb0f926cce46de831d17f0e4db and to the sealed 1,000-image
teacher publication.  It performs authorized pseudo-label distillation for 15
epochs and exports ONNX, but it does not evaluate independent business truth or
publish a CPU delivery artifact.  A failed/OOM run is retained and never
resumed or overwritten.
#>
[CmdletBinding()]
param()

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$Utf8NoBom = New-Object Text.UTF8Encoding($false)
$Ascii = [Text.Encoding]::ASCII
$RequiredHead = '3080a692a37d7efb0f926cce46de831d17f0e4db'
$RequiredTree = 'fb7a21f99139edd15eb1bb10e311039ebe28ebf5'
$RepoRoot = 'C:\f3-white-code-3080a69'
$TeacherRoot = 'C:\f3-white-teacher-3080a69-pilot1000-a\publications\paddle-teacher-consensus'
$RunRoot = 'C:\f3-white-train-3080a69-pilot1000-a'
$PythonExe = 'D:\alipay-ai-data\alipay-ai-inference\.venv-cu126\Scripts\python.exe'
$GitExe = 'C:\Program Files\Git\cmd\git.exe'
$NvidiaSmiExe = 'C:\Program Files\NVIDIA Corporation\NVSMI\nvidia-smi.exe'
$PipelineStartedUtc = [DateTime]::UtcNow
$RunRootOwned = $false
$RunRootIdentity = $null

$RequiredCode = [ordered]@{
    'pyproject.toml' = [ordered]@{ sha256='92c5e5092212b2a029617d484ee860960bb62eca6ad42df308525e3d7b0c5a6f'; size_bytes=[int64]1626 }
    'requirements-ocr.txt' = [ordered]@{ sha256='4118d4913d4d4256dbb8c47f853f14d37b87ace456ae5098acfff9e51208b4b4'; size_bytes=[int64]938 }
    'requirements-train-ocr.txt' = [ordered]@{ sha256='f47c27fdb430df513926a048e732722be9912c121592b12ed85f0bee9755058c'; size_bytes=[int64]765 }
    'scripts\otherimages-line-dataset.py' = [ordered]@{ sha256='001384e08f44e26376cdfb21e8c1fa55871b6a34e050a6dad70c94cd5c3ad2e0'; size_bytes=[int64]443 }
    'src\transfer_receipt_ai\__init__.py' = [ordered]@{ sha256='4a4d67bbff8e1abcc2f6d71c0899f182d5313b901d8699275a493bfcc5a7ef6b'; size_bytes=[int64]134 }
    'src\transfer_receipt_ai\labels.py' = [ordered]@{ sha256='e1ab11be10a6d2f1b55ba74dd605bcfc0a3a64971eaaf0be0ae66a23665ae033'; size_bytes=[int64]1450 }
    'src\transfer_receipt_ai\ocr_train.py' = [ordered]@{ sha256='84192d84b5b57c434bc93b97f1a752b37b5585793e537f101ac106e19d72aa28'; size_bytes=[int64]59089 }
    'src\transfer_receipt_ai\otherimages_inventory.py' = [ordered]@{ sha256='004c8e8e787e5ba684fdf4c3fa562df9317d549139441c01faee91646538691e'; size_bytes=[int64]67916 }
    'src\transfer_receipt_ai\otherimages_line_dataset.py' = [ordered]@{ sha256='c84e066aceb8c79779118bb404638092ef9b5e576fa5107e3c4cd9c860ee749b'; size_bytes=[int64]43600 }
    'src\transfer_receipt_ai\otherimages_paddle_capture.py' = [ordered]@{ sha256='470c2753c7fba63e1bd0e2e24e0a04ef7a3f523638933838995567395eae5494'; size_bytes=[int64]36293 }
    'src\transfer_receipt_ai\otherimages_paddle_teacher.py' = [ordered]@{ sha256='2155e7b1f49401ee49770241db2183a5ff7d02f34212afa9eb158f64132847c1'; size_bytes=[int64]102994 }
}

function Write-TextNew([string]$Path, [string]$Text) {
    if (Test-Path -LiteralPath $Path) { throw "Refusing to overwrite evidence: $Path" }
    $parent = Split-Path -Parent $Path
    if (-not (Test-Path -LiteralPath $parent -PathType Container)) { [IO.Directory]::CreateDirectory($parent) | Out-Null }
    $stream = New-Object IO.FileStream($Path,[IO.FileMode]::CreateNew,[IO.FileAccess]::Write,[IO.FileShare]::None)
    try {
        $writer = New-Object IO.StreamWriter($stream,$Utf8NoBom)
        try { $writer.Write($Text); $writer.Flush() } finally { $writer.Dispose() }
    }
    finally { $stream.Dispose() }
}

function Write-JsonNew([string]$Path, [object]$Value) {
    Write-TextNew $Path (($Value | ConvertTo-Json -Depth 100) + "`r`n")
}

function Write-RcNew([string]$Path, [int]$Rc) {
    $text = ([string]$Rc) + "`r`n"
    Write-TextNew $Path $text
    [byte[]]$observed = [IO.File]::ReadAllBytes($Path)
    [byte[]]$expected = $Ascii.GetBytes($text)
    if ($observed.Length -ne $expected.Length) { throw "RC evidence length changed: $Path" }
    for ($index=0; $index -lt $expected.Length; $index++) {
        if ($observed[$index] -ne $expected[$index]) { throw "RC evidence bytes changed: $Path" }
    }
}

function Assert-ZeroRc([string]$Path) {
    [byte[]]$bytes = [IO.File]::ReadAllBytes($Path)
    if ($bytes.Length -ne 3 -or $bytes[0] -ne 0x30 -or $bytes[1] -ne 0x0d -or $bytes[2] -ne 0x0a) {
        throw "Successful RC evidence is not exact ASCII 0 CRLF: $Path"
    }
}

function Read-Json([string]$Path) {
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) { throw "Missing JSON: $Path" }
    return (Get-Content -LiteralPath $Path -Raw -Encoding UTF8) | ConvertFrom-Json
}

function Get-Binding([string]$Path) {
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) { throw "Missing bound file: $Path" }
    $item = Get-Item -LiteralPath $Path -Force
    return [pscustomobject][ordered]@{
        path=$item.FullName
        size_bytes=[int64]$item.Length
        sha256=(Get-FileHash -LiteralPath $item.FullName -Algorithm SHA256).Hash.ToLowerInvariant()
    }
}

function Assert-BindingUnchanged([object]$Expected, [string]$Description) {
    $observed=Get-Binding ([string]$Expected.path)
    if ([string]$observed.sha256 -cne [string]$Expected.sha256 -or [int64]$observed.size_bytes -ne [int64]$Expected.size_bytes) {
        throw "$Description SHA/size authority changed"
    }
}

function Assert-NoReparseChain([string]$Path, [string]$Description) {
    $current = [IO.Path]::GetFullPath($Path)
    while (-not [string]::IsNullOrWhiteSpace($current)) {
        $item = Get-Item -LiteralPath $current -Force -ErrorAction SilentlyContinue
        if ($null -ne $item -and (($item.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0)) {
            throw "$Description traverses a reparse point: $current"
        }
        $next = Split-Path -Parent $current
        if ([string]::IsNullOrWhiteSpace($next) -or $next -eq $current) { break }
        $current = $next
    }
}

function Initialize-NativeDirectoryType {
    if ($null -ne ('WhiteTrainNativeDirectoryV1' -as [type])) { return }
    $source = @'
using System;
using System.ComponentModel;
using System.Runtime.InteropServices;
using Microsoft.Win32.SafeHandles;
public static class WhiteTrainNativeDirectoryV1 {
    [StructLayout(LayoutKind.Sequential)] private struct Info {
        public uint FileAttributes; public System.Runtime.InteropServices.ComTypes.FILETIME CreationTime;
        public System.Runtime.InteropServices.ComTypes.FILETIME LastAccessTime;
        public System.Runtime.InteropServices.ComTypes.FILETIME LastWriteTime; public uint VolumeSerialNumber;
        public uint FileSizeHigh; public uint FileSizeLow; public uint NumberOfLinks;
        public uint FileIndexHigh; public uint FileIndexLow;
    }
    [DllImport("kernel32.dll", CharSet=CharSet.Unicode, SetLastError=true, ExactSpelling=true)]
    [return: MarshalAs(UnmanagedType.Bool)] private static extern bool CreateDirectoryW(string path, IntPtr attributes);
    [DllImport("kernel32.dll", CharSet=CharSet.Unicode, SetLastError=true, ExactSpelling=true)]
    private static extern SafeFileHandle CreateFileW(string path,uint access,uint share,IntPtr security,uint creation,uint flags,IntPtr template);
    [DllImport("kernel32.dll", SetLastError=true)] [return: MarshalAs(UnmanagedType.Bool)]
    private static extern bool GetFileInformationByHandle(SafeFileHandle handle,out Info info);
    public static void CreateExclusive(string path) {
        if (!CreateDirectoryW(path,IntPtr.Zero)) throw new Win32Exception(Marshal.GetLastWin32Error(),"exclusive directory creation failed: " + path);
    }
    public static string Identity(string path) {
        using (SafeFileHandle handle=CreateFileW(path,0,7,IntPtr.Zero,3,0x02000000,IntPtr.Zero)) {
            if (handle.IsInvalid) throw new Win32Exception(Marshal.GetLastWin32Error(),"directory identity open failed: " + path);
            Info info; if (!GetFileInformationByHandle(handle,out info)) throw new Win32Exception(Marshal.GetLastWin32Error(),"directory identity query failed: " + path);
            return info.VolumeSerialNumber.ToString("x8")+":"+info.FileIndexHigh.ToString("x8")+":"+info.FileIndexLow.ToString("x8");
        }
    }
}
'@
    Add-Type -TypeDefinition $source -Language CSharp | Out-Null
}

function Get-DirectoryIdentity([string]$Path, [string]$Description) {
    Assert-NoReparseChain $Path $Description
    $item = Get-Item -LiteralPath $Path -Force
    if (-not $item.PSIsContainer) { throw "$Description is not a directory: $Path" }
    return [WhiteTrainNativeDirectoryV1]::Identity($item.FullName)
}

function Assert-GitAuthority([object]$ExpectedGitBinding) {
    Assert-BindingUnchanged $ExpectedGitBinding 'fixed Git executable before HEAD query'
    [string[]]$head = @(& $GitExe -C $RepoRoot rev-parse HEAD 2>&1)
    if ($LASTEXITCODE -ne 0 -or $head.Count -ne 1 -or $head[0] -cne $RequiredHead) { throw 'Frozen source HEAD gate failed' }
    Assert-BindingUnchanged $ExpectedGitBinding 'fixed Git executable after HEAD query'
    Assert-BindingUnchanged $ExpectedGitBinding 'fixed Git executable before tree query'
    [string[]]$tree = @(& $GitExe -C $RepoRoot rev-parse 'HEAD^{tree}' 2>&1)
    if ($LASTEXITCODE -ne 0 -or $tree.Count -ne 1 -or $tree[0] -cne $RequiredTree) { throw 'Frozen source tree gate failed' }
    Assert-BindingUnchanged $ExpectedGitBinding 'fixed Git executable after tree query'
    Assert-BindingUnchanged $ExpectedGitBinding 'fixed Git executable before status query'
    [string[]]$status = @(& $GitExe -C $RepoRoot status --porcelain=v1 --untracked-files=all 2>&1)
    if ($LASTEXITCODE -ne 0 -or $status.Count -ne 0) { throw 'Frozen source checkout is not completely clean' }
    Assert-BindingUnchanged $ExpectedGitBinding 'fixed Git executable after status query'
    Assert-BindingUnchanged $ExpectedGitBinding 'fixed Git executable before diff query'
    [string[]]$diff = @(& $GitExe -C $RepoRoot diff --no-ext-diff --quiet --exit-code HEAD -- 2>&1)
    if ($LASTEXITCODE -ne 0 -or $diff.Count -ne 0) { throw 'Frozen source tracked tree differs from HEAD' }
    Assert-BindingUnchanged $ExpectedGitBinding 'fixed Git executable after diff query'
}

function Get-CodeBindings {
    $result = [ordered]@{}
    foreach ($relative in $RequiredCode.Keys) {
        $path = Join-Path $RepoRoot $relative
        Assert-NoReparseChain $path 'frozen training source'
        $binding = Get-Binding $path
        $expected = $RequiredCode[$relative]
        if ([string]$binding.sha256 -cne [string]$expected.sha256 -or [int64]$binding.size_bytes -ne [int64]$expected.size_bytes) {
            throw "Frozen source SHA/size gate failed: $relative"
        }
        $result[$relative] = $binding
    }
    return $result
}

function Assert-ExactFlatTree([string]$Root, [string[]]$Files, [string[]]$Directories, [string]$Description) {
    [string[]]$observedFiles = @(Get-ChildItem -LiteralPath $Root -File -Force | ForEach-Object { $_.Name } | Sort-Object)
    [string[]]$observedDirectories = @(Get-ChildItem -LiteralPath $Root -Directory -Force | ForEach-Object { $_.Name } | Sort-Object)
    if (($observedFiles -join '|') -cne (@($Files | Sort-Object) -join '|') -or ($observedDirectories -join '|') -cne (@($Directories | Sort-Object) -join '|')) {
        throw "$Description is not its exact expected file/directory tree"
    }
}

function Assert-TeacherAuthority {
    Assert-NoReparseChain $TeacherRoot 'sealed white teacher publication'
    Assert-ExactFlatTree $TeacherRoot @('reject_manifest.jsonl','teacher.contract.json','teacher.receipt.json','teacher_manifest.jsonl') @() 'Teacher publication'
    $contractPath = Join-Path $TeacherRoot 'teacher.contract.json'
    $receiptPath = Join-Path $TeacherRoot 'teacher.receipt.json'
    $contract = Read-Json $contractPath
    $receipt = Read-Json $receiptPath
    if ([int]$contract.schema_version -ne 1 -or [string]$contract.kind -cne 'otherimages_paddle_teacher_contract_v1' -or $contract.sealed -ne $true -or $contract.training_authorization -ne $false) { throw 'Teacher seal/authorization gate failed' }
    if (-not ([IO.Path]::GetFullPath([string]$contract.output_directory)).Equals($TeacherRoot,[StringComparison]::OrdinalIgnoreCase)) { throw 'Teacher output directory authority failed' }
    if ([string]$receipt.kind -cne 'otherimages_paddle_teacher_receipt_v1' -or $receipt.sealed -ne $true -or [string]$receipt.contract_closure_sha256 -cne [string]$contract.closure_sha256) { throw 'Teacher receipt closure gate failed' }
    $contractBinding = Get-Binding $contractPath
    if ([string]$receipt.contract.path -cne 'teacher.contract.json' -or [string]$receipt.contract.sha256 -cne [string]$contractBinding.sha256 -or [int64]$receipt.contract.size_bytes -ne [int64]$contractBinding.size_bytes) { throw 'Teacher receipt contract binding failed' }
    if (@($contract.artifacts).Count -ne 2) { throw 'Teacher contract must bind exactly two manifests' }
    [string[]]$artifactNames = @($contract.artifacts | ForEach-Object { [string]$_.path } | Sort-Object)
    if (($artifactNames -join '|') -cne 'reject_manifest.jsonl|teacher_manifest.jsonl') { throw 'Teacher manifest artifact identities failed' }
    foreach ($artifact in @($contract.artifacts)) {
        $observed = Get-Binding (Join-Path $TeacherRoot ([string]$artifact.path))
        if ([string]$artifact.sha256 -cne [string]$observed.sha256 -or [int64]$artifact.size_bytes -ne [int64]$observed.size_bytes) { throw 'Teacher manifest artifact SHA/size failed' }
    }
    [int]$accepted = [int]$contract.counts.accepted_teacher_records
    [int]$quarantined = [int]$contract.counts.quarantined_records
    [int]$train = [int]$contract.counts.accepted_by_split.train
    [int]$val = [int]$contract.counts.accepted_by_split.val
    [int]$test = [int]$contract.counts.accepted_by_split.test
    if ([int]$contract.counts.inventory_records -ne 1000 -or [int]$contract.counts.pending_records -ne 999 -or ($accepted+$quarantined) -ne 1000) { throw 'Teacher inventory/accepted/quarantine closure failed' }
    if ($train -lt 1 -or $val -lt 1 -or $test -lt 1 -or ($train+$val+$test) -ne $accepted) { throw 'Teacher must contain accepted train, val, and test records' }
    if ([int]$contract.counts.training_eligible_records -ne $train -or [int]$contract.counts.evaluation_only_records -ne ($val+$test)) { throw 'Teacher split-use counts failed' }
    return [pscustomobject][ordered]@{ contract=$contract; receipt=$receipt; accepted=$accepted; quarantined=$quarantined; train=$train; val=$val; test=$test }
}

function Assert-NoConflictingGpuWork([string]$Phase, [object]$ExpectedNvidiaSmiBinding) {
    $conflicts = @(
        Get-CimInstance Win32_Process | Where-Object {
            [int]$_.ProcessId -ne [int]$PID -and -not [string]::IsNullOrWhiteSpace([string]$_.CommandLine) -and
            ([string]$_.CommandLine -match '(?i)receipt-ocr-recipient-multiview-exact8|f3e8|exact8|otherimages-paddle-capture|otherimages_paddle_capture|transfer_receipt_ai\.ocr_train')
        }
    )
    if ($conflicts.Count -ne 0) {
        $summary = @($conflicts | ForEach-Object { ([string]$_.ProcessId)+':'+([string]$_.Name) }) -join ','
        throw "Refusing concurrent Exact8/Paddle/student training at $Phase: $summary"
    }
    Assert-BindingUnchanged $ExpectedNvidiaSmiBinding "fixed nvidia-smi before GPU query at $Phase"
    [string[]]$gpu = @(& $NvidiaSmiExe --query-gpu=index,name,memory.total --format=csv,noheader,nounits 2>&1)
    if ($LASTEXITCODE -ne 0 -or $gpu.Count -lt 1) { throw "nvidia-smi GPU authority failed at $Phase" }
    Assert-BindingUnchanged $ExpectedNvidiaSmiBinding "fixed nvidia-smi after GPU query at $Phase"
    Assert-BindingUnchanged $ExpectedNvidiaSmiBinding "fixed nvidia-smi before compute query at $Phase"
    [string[]]$compute = @(& $NvidiaSmiExe --query-compute-apps=pid,process_name --format=csv,noheader,nounits 2>&1)
    if ($LASTEXITCODE -ne 0) { throw "Unable to prove an idle CUDA compute process set at $Phase" }
    Assert-BindingUnchanged $ExpectedNvidiaSmiBinding "fixed nvidia-smi after compute query at $Phase"
    $active = @($compute | Where-Object { -not [string]::IsNullOrWhiteSpace($_) })
    if ($active.Count -ne 0) { throw "CUDA GPU already has active compute work at ${Phase}: $($active -join ';')" }
    return @($gpu)
}

function ConvertTo-NativeCommandLine([string[]]$Arguments) {
    $builder = New-Object Text.StringBuilder
    foreach ($argument in $Arguments) {
        if ($null -eq $argument -or $argument.IndexOf('"') -ge 0 -or $argument.EndsWith('\')) { throw "Unsupported native argument spelling: $argument" }
        if ($builder.Length -gt 0) { [void]$builder.Append(' ') }
        [void]$builder.Append('"'); [void]$builder.Append($argument); [void]$builder.Append('"')
    }
    return $builder.ToString()
}

function Stop-ProcessTree([int]$RootPid) {
    for ($attempt=0; $attempt -lt 3; $attempt++) {
        $all = @(Get-CimInstance Win32_Process -ErrorAction SilentlyContinue)
        $pending = New-Object Collections.Generic.List[int]
        $pending.Add($RootPid)
        $descendants = New-Object Collections.Generic.List[int]
        for ($index=0; $index -lt $pending.Count; $index++) {
            $parent = $pending[$index]
            foreach ($child in @($all | Where-Object { [int]$_.ParentProcessId -eq $parent })) {
                $childPid = [int]$child.ProcessId
                if (-not $pending.Contains($childPid)) { $pending.Add($childPid); $descendants.Add($childPid) }
            }
        }
        [int[]]$targets = @($descendants.ToArray()); [array]::Reverse($targets)
        foreach ($target in @($targets + $RootPid)) { Stop-Process -Id $target -Force -ErrorAction SilentlyContinue }
        Start-Sleep -Milliseconds 250
    }
}

function Add-ObservedDescendantPids([int]$RootPid, [Collections.Generic.HashSet[int]]$Observed) {
    $all=@(Get-CimInstance Win32_Process -ErrorAction SilentlyContinue)
    $pending=New-Object Collections.Generic.List[int]; $pending.Add($RootPid)
    for ($index=0; $index -lt $pending.Count; $index++) {
        $parent=$pending[$index]
        foreach ($child in @($all | Where-Object {[int]$_.ParentProcessId -eq $parent})) {
            $childPid=[int]$child.ProcessId
            if (-not $pending.Contains($childPid)) { $pending.Add($childPid); [void]$Observed.Add($childPid) }
        }
    }
}

function Wait-ProcessIdsAbsent([int[]]$ProcessIds, [int]$TimeoutMilliseconds) {
    $deadline=[DateTime]::UtcNow.AddMilliseconds($TimeoutMilliseconds)
    do {
        [int[]]$alive=@($ProcessIds | Where-Object {$null -ne (Get-Process -Id $_ -ErrorAction SilentlyContinue)})
        if ($alive.Count -eq 0) { return $true }
        Start-Sleep -Milliseconds 250
    } while ([DateTime]::UtcNow -lt $deadline)
    return $false
}

function Stop-ObservedProcessIds([int[]]$ProcessIds) {
    foreach ($processId in $ProcessIds) { Stop-Process -Id $processId -Force -ErrorAction SilentlyContinue }
}

function Assert-NoRetainedProcessTree([int]$RootPid, [string]$Stage) {
    $all = @(Get-CimInstance Win32_Process -ErrorAction Stop)
    $pending = New-Object Collections.Generic.List[int]
    $pending.Add($RootPid)
    $retained = New-Object Collections.Generic.List[object]
    for ($index=0; $index -lt $pending.Count; $index++) {
        $parent = $pending[$index]
        foreach ($child in @($all | Where-Object { [int]$_.ParentProcessId -eq $parent })) {
            $childPid = [int]$child.ProcessId
            if (-not $pending.Contains($childPid)) { $pending.Add($childPid); $retained.Add($child) }
        }
    }
    if ($retained.Count -ne 0) {
        $summary=@($retained | ForEach-Object {([string]$_.ProcessId)+':'+([string]$_.Name)}) -join ','
        Stop-ProcessTree $RootPid
        throw "Python stage retained descendant processes after success: stage=$Stage descendants=$summary"
    }
}

function Invoke-PythonStage([string]$Name, [string[]]$Arguments, [hashtable]$Environment) {
    Assert-BindingUnchanged $NvidiaSmiBinding "fixed nvidia-smi immediately before Python stage $Name"
    $stageRoot = Join-Path $LogsRoot $Name
    if (Test-Path -LiteralPath $stageRoot) { throw "Stage log root already exists: $stageRoot" }
    [IO.Directory]::CreateDirectory($stageRoot) | Out-Null
    $stdoutPath = Join-Path $stageRoot 'stdout.txt'; $stderrPath = Join-Path $stageRoot 'stderr.txt'; $rcPath = Join-Path $stageRoot 'rc.txt'
    $info = New-Object Diagnostics.ProcessStartInfo
    $info.FileName=$PythonExe; $info.Arguments=ConvertTo-NativeCommandLine $Arguments; $info.WorkingDirectory=$RepoRoot
    $info.UseShellExecute=$false; $info.CreateNoWindow=$true; $info.RedirectStandardOutput=$true; $info.RedirectStandardError=$true
    foreach ($name in @('PYTHONHOME','PYTHONSTARTUP','PYTHONINSPECT','PYTEST_ADDOPTS','PYTEST_PLUGINS')) { [void]$info.EnvironmentVariables.Remove($name) }
    $info.EnvironmentVariables['PYTHONIOENCODING']='utf-8:strict'; $info.EnvironmentVariables['PYTHONUTF8']='1'; $info.EnvironmentVariables['PYTHONDONTWRITEBYTECODE']='1'
    $info.EnvironmentVariables['PYTHONPATH']=(Join-Path $RepoRoot 'src'); $info.EnvironmentVariables['PYTHONNOUSERSITE']='1'
    foreach ($key in $Environment.Keys) { $info.EnvironmentVariables[[string]$key]=[string]$Environment[$key] }
    $process=New-Object Diagnostics.Process; $process.StartInfo=$info
    $started=[DateTime]::UtcNow; $startedProcess=$false; $pidValue=$null; $stdoutTask=$null; $stderrTask=$null; $exitCode=$null; $stageError=$null; $cleanupError=$null; $forcedStop=$false
    $observedDescendantPids=New-Object 'Collections.Generic.HashSet[int]'; $descendantAbsenceProven=$false
    try {
        Write-Host "WHITE_TRAIN_STAGE_START stage=$Name"
        if (-not $process.Start()) { throw "Unable to start Python stage: $Name" }
        $startedProcess=$true; $pidValue=[int]$process.Id
        Add-ObservedDescendantPids $pidValue $observedDescendantPids
        $stdoutTask=$process.StandardOutput.ReadToEndAsync(); $stderrTask=$process.StandardError.ReadToEndAsync(); $lastReport=$started
        while (-not $process.WaitForExit(1000)) {
            Add-ObservedDescendantPids $pidValue $observedDescendantPids
            $now=[DateTime]::UtcNow
            if (($now-$lastReport).TotalSeconds -ge 60) {
                try { $process.Refresh(); Write-Host ("WHITE_TRAIN_STAGE_ALIVE stage={0} elapsed_s={1} pid={2} cpu_s={3:N1} ws_bytes={4}" -f $Name,[int]($now-$started).TotalSeconds,$pidValue,$process.TotalProcessorTime.TotalSeconds,$process.WorkingSet64) }
                catch { Write-Host "WHITE_TRAIN_STAGE_ALIVE stage=$Name elapsed_s=$([int]($now-$started).TotalSeconds) pid=$pidValue" }
                $lastReport=$now
            }
        }
        $process.WaitForExit(); $exitCode=[int]$process.ExitCode
    }
    catch { $stageError=$_ }
    finally {
        try {
            try {
                if ($startedProcess) { Add-ObservedDescendantPids $pidValue $observedDescendantPids }
                if ($startedProcess -and -not $process.HasExited) {
                    $forcedStop=$true; Stop-ProcessTree $pidValue; Stop-ObservedProcessIds @($observedDescendantPids)
                    if (-not $process.WaitForExit(30000)) { throw 'Python process tree remained alive after forced stop and bounded wait' }
                }
                if ($startedProcess -and $process.HasExited) { $process.WaitForExit(); $exitCode=[int]$process.ExitCode }
                [int[]]$observedIds=@($observedDescendantPids | Sort-Object)
                $descendantAbsenceProven=Wait-ProcessIdsAbsent $observedIds 30000
                if (-not $descendantAbsenceProven) {
                    $forcedStop=$true; Stop-ObservedProcessIds $observedIds
                    if (-not (Wait-ProcessIdsAbsent $observedIds 30000)) { throw "Observed descendant PIDs remained alive after bounded cleanup: $($observedIds -join ',')" }
                    throw "Python stage retained descendant PIDs and required forced cleanup: $($observedIds -join ',')"
                }
            }
            catch { $cleanupError=$_ }
            try { Assert-BindingUnchanged $NvidiaSmiBinding "fixed nvidia-smi in finalizer after Python stage $Name" }
            catch { if ($null -eq $cleanupError) { $cleanupError=$_ } }
            try {
                foreach ($streamTask in @($stdoutTask,$stderrTask)) {
                    if ($null -ne $streamTask -and -not $streamTask.IsCompleted -and -not $streamTask.Wait(30000)) { throw 'Python redirected stream did not close within the bounded wait' }
                }
                $stdout=if ($null -ne $stdoutTask -and $stdoutTask.IsCompleted) { [string]$stdoutTask.Result } else { '' }
                $stderr=if ($null -ne $stderrTask -and $stderrTask.IsCompleted) { [string]$stderrTask.Result } else { '' }
                [IO.File]::WriteAllText($stdoutPath,$stdout,$Utf8NoBom); [IO.File]::WriteAllText($stderrPath,$stderr,$Utf8NoBom)
                if ($null -ne $exitCode) { Write-RcNew $rcPath ([int]$exitCode) }
                $stderrNonEmpty=([Text.Encoding]::UTF8.GetByteCount($stderr) -ne 0)
                if ($null -ne $stageError -or $null -ne $cleanupError -or $null -eq $exitCode -or [int]$exitCode -ne 0 -or $stderrNonEmpty) {
                    Write-JsonNew (Join-Path $stageRoot 'stage.failure.json') ([ordered]@{
                        schema_version=1; kind='otherimages_white_train_windows_stage_failure_v1'; status='failed'; stage=$Name; pid=$pidValue
                        process_started=$startedProcess; forced_process_tree_stop=$forcedStop; exit_code=$exitCode; stderr_nonempty=$stderrNonEmpty
                        observed_descendant_pids=@($observedDescendantPids | Sort-Object); descendant_absence_proven=$descendantAbsenceProven
                        stage_error=$(if ($null -eq $stageError) {$null} else {$stageError.Exception.Message}); cleanup_error=$(if ($null -eq $cleanupError) {$null} else {$cleanupError.Exception.Message})
                        stdout=Get-Binding $stdoutPath; stderr=Get-Binding $stderrPath; rc_file=$(if (Test-Path -LiteralPath $rcPath) {Get-Binding $rcPath} else {$null}); utc=[DateTime]::UtcNow.ToString('o')
                    })
                }
            }
            catch { if ($null -eq $cleanupError) { $cleanupError=$_ } }
        }
        finally { $process.Dispose() }
    }
    if ($null -ne $stageError) { throw $stageError }; if ($null -ne $cleanupError) { throw $cleanupError }
    if ($null -eq $exitCode -or [int]$exitCode -ne 0) { throw "Python stage failed: stage=$Name rc=$exitCode evidence=$stageRoot" }
    Assert-ZeroRc $rcPath
    Assert-BindingUnchanged $NvidiaSmiBinding "fixed nvidia-smi immediately after Python stage $Name"
    $result=[pscustomobject][ordered]@{ name=$Name; pid=$pidValue; rc=[int]$exitCode; started_utc=$started.ToString('o'); completed_utc=[DateTime]::UtcNow.ToString('o'); elapsed_seconds=([DateTime]::UtcNow-$started).TotalSeconds; observed_descendant_pids=@($observedDescendantPids | Sort-Object); descendant_absence_proven=$descendantAbsenceProven; stdout=Get-Binding $stdoutPath; stderr=Get-Binding $stderrPath; rc_file=Get-Binding $rcPath }
    if ([int64]$result.stderr.size_bytes -ne 0) { throw "Python stage emitted stderr: stage=$Name evidence=$stderrPath" }
    Write-Host "WHITE_TRAIN_STAGE_EXIT stage=$Name rc=0 elapsed_s=$([int]$result.elapsed_seconds)"
    return $result
}

function Complete-Stage([object]$Stage, [object]$Validation) {
    $path=Join-Path (Join-Path $LogsRoot ([string]$Stage.name)) 'stage.receipt.json'
    Write-JsonNew $path ([ordered]@{ schema_version=1; kind='otherimages_white_train_windows_stage_receipt_v1'; status='complete'; stage=$Stage.name; process=[ordered]@{pid=$Stage.pid;rc=$Stage.rc;started_utc=$Stage.started_utc;completed_utc=$Stage.completed_utc;elapsed_seconds=$Stage.elapsed_seconds;observed_descendant_pids=$Stage.observed_descendant_pids;descendant_absence_proven=$Stage.descendant_absence_proven}; stdout=$Stage.stdout;stderr=$Stage.stderr;rc_file=$Stage.rc_file;validation=$Validation })
    return Get-Binding $path
}

$TeacherVerifierSource = @'
from __future__ import annotations
import argparse, hashlib, json, math, os, stat
from collections import Counter
from pathlib import Path

FILES = {"teacher_manifest.jsonl", "reject_manifest.jsonl", "teacher.contract.json", "teacher.receipt.json"}
def fail(message): raise SystemExit("independent teacher closure verification failed: " + message)
def pairs(items):
    result = {}
    for key, value in items:
        if key in result: fail("duplicate JSON key " + repr(key))
        result[key] = value
    return result
def constant(value): fail("non-standard JSON constant " + repr(value))
def reject_reparse(path):
    current = path.absolute()
    while True:
        if current.exists():
            info = current.lstat()
            if stat.S_ISLNK(info.st_mode) or int(getattr(info, "st_file_attributes", 0)) & 0x400: fail("reparse path " + str(current))
        if current.parent == current: return
        current = current.parent
def load_json(path):
    data = path.read_bytes()
    if data.startswith(b"\xef\xbb\xbf"): fail("JSON BOM " + str(path))
    try: value = json.loads(data.decode("utf-8"), object_pairs_hook=pairs, parse_constant=constant)
    except (UnicodeDecodeError, json.JSONDecodeError) as error: fail("invalid JSON at %s: %s" % (path, error))
    if not isinstance(value, dict): fail("non-object JSON " + str(path))
    return value
def load_jsonl(path):
    data = path.read_bytes()
    if data.startswith(b"\xef\xbb\xbf"): fail("JSONL BOM " + str(path))
    try: text = data.decode("utf-8")
    except UnicodeDecodeError as error: fail("invalid JSONL UTF-8 at %s: %s" % (path, error))
    if text and not text.endswith("\n"): fail("JSONL lacks final LF " + str(path))
    rows=[]
    for number,line in enumerate(text.splitlines(),1):
        if not line: fail("blank JSONL line at %s:%d" % (path,number))
        try: value=json.loads(line,object_pairs_hook=pairs,parse_constant=constant)
        except json.JSONDecodeError as error: fail("invalid JSONL at %s:%d: %s" % (path,number,error))
        if not isinstance(value,dict): fail("non-object JSONL row")
        rows.append(value)
    return rows
def binding(path):
    reject_reparse(path)
    data=path.read_bytes()
    return {"path":path.name,"sha256":hashlib.sha256(data).hexdigest(),"size_bytes":len(data),"line_count":data.count(b"\n")}
def same_path(left,right): return os.path.normcase(os.path.abspath(os.fspath(left))) == os.path.normcase(os.path.abspath(os.fspath(right)))

parser=argparse.ArgumentParser(); parser.add_argument("--teacher-root",type=Path,required=True); args=parser.parse_args()
root=args.teacher_root.absolute(); reject_reparse(root)
if not root.is_dir() or {item.name for item in root.iterdir()} != FILES or any(not item.is_file() for item in root.iterdir()): fail("exact teacher four-file tree differs")
manifest,rejects=root/"teacher_manifest.jsonl",root/"reject_manifest.jsonl"
contract_path,receipt_path=root/"teacher.contract.json",root/"teacher.receipt.json"
accepted,rejected=load_jsonl(manifest),load_jsonl(rejects); contract,receipt=load_json(contract_path),load_json(receipt_path)
if set(contract) != {"schema_version","kind","generated_at_utc","sealed","output_directory","inputs","configuration","counts","split_use","artifacts","closure_sha256","training_authorization","ocr_execution_performed_by_this_module","manual_review_required","low_confidence_or_conflict_policy","limitations"}: fail("teacher contract schema differs")
if contract.get("schema_version") != 1 or contract.get("kind") != "otherimages_paddle_teacher_contract_v1" or contract.get("sealed") is not True or contract.get("training_authorization") is not False or not same_path(contract.get("output_directory",""),root): fail("teacher authority fields differ")
expected_artifacts=[binding(manifest),binding(rejects)]
if contract.get("artifacts") != expected_artifacts: fail("teacher artifacts differ")
counts=contract.get("counts"); splits=Counter(str(row.get("split")) for row in accepted)
if not isinstance(counts,dict) or counts.get("accepted_teacher_records") != len(accepted) or counts.get("quarantined_records") != len(rejected) or len(accepted)+len(rejected) != 1000: fail("teacher count closure differs")
if counts.get("accepted_by_split") != dict(sorted(splits.items())) or any(splits[name] < 1 for name in ("train","val","test")): fail("teacher accepted split closure differs")
closure={key:contract.get(key) for key in ("schema_version","inputs","configuration","counts","split_use","artifacts")}
closure_sha=hashlib.sha256(json.dumps(closure,ensure_ascii=False,sort_keys=True,separators=(",",":"),allow_nan=False).encode("utf-8")).hexdigest()
if contract.get("closure_sha256") != closure_sha: fail("teacher canonical closure differs")
contract_binding=binding(contract_path)
if set(receipt) != {"schema_version","kind","sealed","contract","contract_closure_sha256"} or receipt.get("schema_version") != 1 or receipt.get("kind") != "otherimages_paddle_teacher_receipt_v1" or receipt.get("sealed") is not True or receipt.get("contract") != contract_binding or receipt.get("contract_closure_sha256") != closure_sha: fail("teacher receipt differs")
print(json.dumps({"schema_version":1,"kind":"otherimages_white_train_teacher_independent_closure_v1","status":"complete","closure_sha256":closure_sha,"accepted":len(accepted),"quarantined":len(rejected),"accepted_by_split":dict(sorted(splits.items()))},sort_keys=True,separators=(",",":"),allow_nan=False))
'@

$ImportAttestationSource = @'
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
import transfer_receipt_ai.ocr_train as module

parser=argparse.ArgumentParser(); parser.add_argument("--expected",type=Path,required=True); parser.add_argument("--sha256",required=True); args=parser.parse_args()
expected=args.expected.resolve(strict=True); observed=Path(module.__file__).resolve(strict=True)
if observed != expected: raise SystemExit("ocr_train import authority differs: expected=%s observed=%s" % (expected,observed))
data=observed.read_bytes(); digest=hashlib.sha256(data).hexdigest()
if digest != args.sha256: raise SystemExit("ocr_train import SHA-256 differs")
print(json.dumps({"schema_version":1,"kind":"otherimages_white_train_import_attestation_v1","status":"complete","module":str(observed),"sha256":digest},sort_keys=True,separators=(",",":"),allow_nan=False))
'@

try {
    foreach ($path in @($RepoRoot,$TeacherRoot,$PythonExe,$GitExe,$NvidiaSmiExe,$PSCommandPath)) { if (-not (Test-Path -LiteralPath $path)) { throw "Missing fixed authority path: $path" }; Assert-NoReparseChain $path 'white train authority' }
    if (Test-Path -LiteralPath $RunRoot) { throw "RunRoot must be brand-new and is never resumed: $RunRoot" }
    Initialize-NativeDirectoryType
    $runParent=Split-Path -Parent $RunRoot; Assert-NoReparseChain $runParent 'white train output parent'; $parentIdentity=Get-DirectoryIdentity $runParent 'white train output parent'
    $gitBefore=Get-Binding $GitExe; $NvidiaSmiBinding=Get-Binding $NvidiaSmiExe
    Assert-GitAuthority $gitBefore; $codeBefore=Get-CodeBindings; $wrapperBefore=Get-Binding $PSCommandPath; $pythonBefore=Get-Binding $PythonExe
    $teacher=Assert-TeacherAuthority
    $teacherBefore=[ordered]@{ manifest=Get-Binding (Join-Path $TeacherRoot 'teacher_manifest.jsonl'); rejects=Get-Binding (Join-Path $TeacherRoot 'reject_manifest.jsonl'); contract=Get-Binding (Join-Path $TeacherRoot 'teacher.contract.json'); receipt=Get-Binding (Join-Path $TeacherRoot 'teacher.receipt.json') }
    $gpuEvidence=@(Assert-NoConflictingGpuWork 'preflight' $NvidiaSmiBinding)
    [WhiteTrainNativeDirectoryV1]::CreateExclusive($RunRoot); $RunRootOwned=$true; $RunRootIdentity=Get-DirectoryIdentity $RunRoot 'white train run root'
    if ((Get-DirectoryIdentity $runParent 'white train parent after create') -cne $parentIdentity) { throw 'White train output parent identity changed' }
    $LogsRoot=Join-Path $RunRoot 'logs'; $PublicationsRoot=Join-Path $RunRoot 'publications'; $CandidateRoot=Join-Path $RunRoot 'candidate'
    [IO.Directory]::CreateDirectory($LogsRoot)|Out-Null; [IO.Directory]::CreateDirectory($PublicationsRoot)|Out-Null; [IO.Directory]::CreateDirectory($CandidateRoot)|Out-Null
    $LineDataset=Join-Path $PublicationsRoot 'generic-line-dataset'; $TrainOutput=Join-Path $CandidateRoot 'checkpoints'; $StudentBundle=Join-Path $CandidateRoot 'student-bundle'; $OnnxPath=Join-Path $StudentBundle 'white-generic-line.onnx'
    $importAttestationPath=Join-Path $LogsRoot 'import-attestation.py'; Write-TextNew $importAttestationPath $ImportAttestationSource; $importAttestationBinding=Get-Binding $importAttestationPath
    $expectedOcrTrain=Join-Path $RepoRoot 'src\transfer_receipt_ai\ocr_train.py'
    $expectedOcrSha=[string]($RequiredCode['src\transfer_receipt_ai\ocr_train.py'].sha256)
    $importStage=Invoke-PythonStage 'import-attestation' @($importAttestationPath,'--expected',$expectedOcrTrain,'--sha256',$expectedOcrSha) @{}
    [string[]]$importLines=@([IO.File]::ReadAllLines([string]$importStage.stdout.path,$Utf8NoBom) | Where-Object {-not [string]::IsNullOrWhiteSpace($_)})
    if ($importLines.Count -ne 1) { throw 'Import attestation stdout must be exactly one JSON value' }
    $importAttestation=$importLines[0] | ConvertFrom-Json
    if ([string]$importAttestation.kind -cne 'otherimages_white_train_import_attestation_v1' -or [string]$importAttestation.status -cne 'complete' -or [string]$importAttestation.sha256 -cne $expectedOcrSha -or -not ([IO.Path]::GetFullPath([string]$importAttestation.module)).Equals($expectedOcrTrain,[StringComparison]::OrdinalIgnoreCase)) { throw 'ocr_train import attestation failed' }
    $importStageReceipt=Complete-Stage $importStage ([ordered]@{source_only_from_fixed_clone=$true;python_venv_read_only_from_d=$true;pythonpath_fixed_clone_src=$true;module=Get-Binding $expectedOcrTrain;attestor=$importAttestationBinding})
    $teacherVerifierPath=Join-Path $LogsRoot 'independent-teacher-closure.py'; Write-TextNew $teacherVerifierPath $TeacherVerifierSource; $teacherVerifierBinding=Get-Binding $teacherVerifierPath
    $teacherVerifyStage=Invoke-PythonStage 'teacher-closure-verify' @($teacherVerifierPath,'--teacher-root',$TeacherRoot) @{}
    [string[]]$teacherVerifyLines=@([IO.File]::ReadAllLines([string]$teacherVerifyStage.stdout.path,$Utf8NoBom) | Where-Object {-not [string]::IsNullOrWhiteSpace($_)})
    if ($teacherVerifyLines.Count -ne 1) { throw 'Independent teacher verifier stdout must be exactly one JSON value' }
    $teacherVerify=$teacherVerifyLines[0] | ConvertFrom-Json
    if ([string]$teacherVerify.kind -cne 'otherimages_white_train_teacher_independent_closure_v1' -or [string]$teacherVerify.status -cne 'complete' -or [string]$teacherVerify.closure_sha256 -cne [string]$teacher.contract.closure_sha256 -or ([int]$teacherVerify.accepted+[int]$teacherVerify.quarantined) -ne 1000) { throw 'Independent teacher closure receipt failed' }
    $teacherVerifyReceipt=Complete-Stage $teacherVerifyStage ([ordered]@{canonical_closure_recomputed=$true;exact_schema=$true;exact_artifacts=$true;accepted_train_val_test_closed=$true;closure_sha256=[string]$teacherVerify.closure_sha256;verifier_source=$teacherVerifierBinding})
    $lineStage=Invoke-PythonStage 'line-dataset' @((Join-Path $RepoRoot 'scripts\otherimages-line-dataset.py'),'--teacher',$TeacherRoot,'--output',$LineDataset,'--authorize-training') @{}
    $lineText=[IO.File]::ReadAllText([string]$lineStage.stdout.path,$Utf8NoBom)
    if ($lineText -notmatch '^Sealed [0-9]+ generic text line\(s\) from [0-9]+ teacher record\(s\) at .+\r?\n$') { throw 'Line dataset stdout is not its exact one-line success summary' }
    $lineContract=Read-Json (Join-Path $LineDataset 'dataset.contract.json'); $lineReceipt=Read-Json (Join-Path $LineDataset 'dataset.receipt.json')
    if ([string]$lineContract.kind -cne 'otherimages_generic_text_line_dataset_contract_v1' -or $lineContract.sealed -ne $true -or $lineContract.training_authorization -ne $true -or [string]$lineContract.training_authorization_source -cne 'explicit_materializer_flag' -or [string]$lineContract.truth_semantics -cne 'teacher_parity_only_not_independent_business_truth') { throw 'Line dataset seal/authorization/truth contract failed' }
    if ([int]$lineContract.counts.by_split.train -lt 1 -or [int]$lineContract.counts.by_split.val -lt 1 -or [int]$lineContract.counts.by_split.test -lt 1) { throw 'Line dataset must retain train, val, and test lines' }
    if ([string]$lineReceipt.kind -cne 'otherimages_generic_text_line_dataset_receipt_v1' -or $lineReceipt.sealed -ne $true -or [string]$lineReceipt.contract_closure_sha256 -cne [string]$lineContract.closure_sha256) { throw 'Line dataset receipt closure failed' }
    $lineContractBinding=Get-Binding (Join-Path $LineDataset 'dataset.contract.json')
    if ([string]$lineReceipt.contract.sha256 -cne [string]$lineContractBinding.sha256 -or [int64]$lineReceipt.contract.size_bytes -ne [int64]$lineContractBinding.size_bytes) { throw 'Line dataset receipt contract SHA/size failed' }
    $lineStageReceipt=Complete-Stage $lineStage ([ordered]@{ sealed=$true; authorize_training=$true; field='generic_text_line'; train=[int]$lineContract.counts.by_split.train;val=[int]$lineContract.counts.by_split.val;test=[int]$lineContract.counts.by_split.test;test_oov_gate_source_commit=$RequiredHead;contract=$lineContractBinding;receipt=Get-Binding (Join-Path $LineDataset 'dataset.receipt.json') })

    [void](Assert-NoConflictingGpuWork 'immediately-before-student-train' $NvidiaSmiBinding)
    $trainStage=Invoke-PythonStage 'cuda-student-train' @(
        '-m','transfer_receipt_ai.ocr_train','--records',(Join-Path $LineDataset 'generic_text_lines.jsonl'),'--dataset-root',$LineDataset,
        '--output',$TrainOutput,'--fields','generic_text_line','--device','cuda:0','--epochs','15','--batch-size','128',
        '--num-workers','4','--persistent-workers','--prefetch-factor','4','--cuda-tf32','--cudnn-benchmark','--validation-every','3','--onnx-output',$OnnxPath
    ) @{}
    $trainText=[IO.File]::ReadAllText([string]$trainStage.stdout.path,$Utf8NoBom)
    if ($trainText -notmatch '(?m)^epoch 15/15: ' -or $trainText -notmatch '(?m)^Best OCR checkpoint: ' -or $trainText -notmatch '(?m)^Exported ONNX OCR model: ') { throw 'Training stdout omitted final epoch/checkpoint/ONNX success markers' }
    Assert-ExactFlatTree $TrainOutput @('best.pt','charset.json','last.pt','training_history.json') @() 'Student checkpoint output'
    Assert-ExactFlatTree $StudentBundle @('white-generic-line.charset.json','white-generic-line.contract.json','white-generic-line.onnx') @() 'Student bundle'
    Assert-ExactFlatTree $CandidateRoot @() @('checkpoints','student-bundle') 'Student analysis candidate'
    foreach ($path in @((Join-Path $TrainOutput 'best.pt'),(Join-Path $TrainOutput 'last.pt'),$OnnxPath)) { if ((Get-Item -LiteralPath $path).Length -le 0) { throw "Candidate artifact is empty: $path" } }
    $history=Read-Json (Join-Path $TrainOutput 'training_history.json'); $options=$history.training_options
    if (@($history.records).Count -ne 15 -or [string]$options.device -cne 'cuda:0' -or [int]$options.epochs -ne 15 -or [int]$options.batch_size -ne 128 -or [int]$options.num_workers -ne 4 -or $options.persistent_workers -ne $true -or [int]$options.prefetch_factor -ne 4 -or $options.cuda_tf32 -ne $true -or $options.cudnn_benchmark -ne $true -or [int]$options.validation_every -ne 3) { throw 'Training history runtime/configuration gate failed' }
    [string[]]$validatedEpochs=@($history.records | Where-Object {$_.validation_ran -eq $true} | ForEach-Object {[string]$_.epoch})
    if (($validatedEpochs -join ',') -cne '3,6,9,12,15') { throw 'Validation cadence differs from exact every-3 plus final policy' }
    $onnxContract=Read-Json (Join-Path $StudentBundle 'white-generic-line.contract.json')
    if ([string]$onnxContract.kind -cne 'receipt_ocr_ctc_v1' -or (@($onnxContract.fields) -join '|') -cne 'generic_text_line' -or [string]$onnxContract.input.preprocess -cne 'opencv_exact_rgb_gray_letterbox_v1') { throw 'ONNX generic-line contract failed' }
    $onnxBinding=Get-Binding $OnnxPath; $charsetBinding=Get-Binding (Join-Path $StudentBundle 'white-generic-line.charset.json')
    if ([string]$onnxContract.onnx_sha256 -cne [string]$onnxBinding.sha256 -or [string]$onnxContract.charset_sha256 -cne [string]$charsetBinding.sha256) { throw 'ONNX/charset SHA binding failed' }
    foreach ($split in @('train','val','test')) { if ([int]$onnxContract.training_field_counts.generic_text_line.$split -ne [int]$lineContract.counts.by_split.$split) { throw "ONNX training field count differs for $split" } }
    $studentContractBinding=Get-Binding (Join-Path $StudentBundle 'white-generic-line.contract.json')
    $trainStageReceipt=Complete-Stage $trainStage ([ordered]@{ device='cuda:0';epochs=15;batch_size=128;num_workers=4;persistent_workers=$true;prefetch_factor=4;cuda_tf32=$true;cudnn_benchmark=$true;validation_every=3;validated_epochs=@(3,6,9,12,15);onnx=$onnxBinding;charset=$charsetBinding;contract=$studentContractBinding;analysis_candidate_only=$true })

    Assert-GitAuthority $gitBefore; $codeAfter=Get-CodeBindings
    foreach ($name in $codeBefore.Keys) { if ([string]$codeBefore[$name].sha256 -cne [string]$codeAfter[$name].sha256 -or [int64]$codeBefore[$name].size_bytes -ne [int64]$codeAfter[$name].size_bytes) { throw "Frozen code changed during pipeline: $name" } }
    foreach ($name in $teacherBefore.Keys) { $after=Get-Binding ([string]$teacherBefore[$name].path); if ([string]$teacherBefore[$name].sha256 -cne [string]$after.sha256 -or [int64]$teacherBefore[$name].size_bytes -ne [int64]$after.size_bytes) { throw "Teacher authority changed during pipeline: $name" } }
    $wrapperAfter=Get-Binding $PSCommandPath; $pythonAfter=Get-Binding $PythonExe; $gitAfter=Get-Binding $GitExe; $nvidiaSmiAfter=Get-Binding $NvidiaSmiExe
    if ([string]$wrapperBefore.sha256 -cne [string]$wrapperAfter.sha256 -or [string]$pythonBefore.sha256 -cne [string]$pythonAfter.sha256 -or [string]$gitBefore.sha256 -cne [string]$gitAfter.sha256 -or [string]$NvidiaSmiBinding.sha256 -cne [string]$nvidiaSmiAfter.sha256 -or [int64]$NvidiaSmiBinding.size_bytes -ne [int64]$nvidiaSmiAfter.size_bytes) { throw 'Wrapper/Python/Git/nvidia-smi executable changed during pipeline' }
    if ((Get-DirectoryIdentity $RunRoot 'white train run root at closure') -cne $RunRootIdentity -or (Get-DirectoryIdentity $runParent 'white train parent at closure') -cne $parentIdentity) { throw 'Run root/parent identity changed' }
    Assert-ExactFlatTree $PublicationsRoot @() @('generic-line-dataset') 'Training publication root'
    $pipelinePath=Join-Path $RunRoot 'pipeline.receipt.json'
    Write-JsonNew $pipelinePath ([ordered]@{
        schema_version=1;kind='otherimages_white_student_training_windows_pipeline_receipt_v1';status='complete';source=[ordered]@{repo_root=$RepoRoot;head=$RequiredHead;tree=$RequiredTree;clean=$true;fixed_code=$codeBefore;wrapper=$wrapperBefore;python=$pythonBefore;git=$gitBefore;nvidia_smi=$NvidiaSmiBinding}
        teacher=[ordered]@{root=$TeacherRoot;bindings=$teacherBefore;accepted=$teacher.accepted;quarantined=$teacher.quarantined;accepted_by_split=[ordered]@{train=$teacher.train;val=$teacher.val;test=$teacher.test}}
        execution=[ordered]@{device='cuda:0';gpu_preflight=$gpuEvidence;epochs=15;batch_size=128;num_workers=4;persistent_workers=$true;prefetch_factor=4;cuda_tf32=$true;cudnn_benchmark=$true;validation_every=3;training_performed=$true;onnx_exported=$true}
        inputs=[ordered]@{teacher_root=$TeacherRoot;teacher_contract=$teacherBefore.contract;teacher_contract_closure_sha256=[string]$teacher.contract.closure_sha256}
        training_performed=$true
        stages=[ordered]@{import_attestation=$importStageReceipt;teacher_closure_verify=$teacherVerifyReceipt;line_dataset=$lineStageReceipt;student_train=$trainStageReceipt};roots=[ordered]@{run=$RunRoot;run_identity=$RunRootIdentity;logs=$LogsRoot;line_dataset=$LineDataset;candidate=$CandidateRoot;student_bundle=$StudentBundle}
        student_bundle=[ordered]@{root=$StudentBundle;bindings=[ordered]@{model=$onnxBinding;charset=$charsetBinding;contract=$studentContractBinding}}
        bindings=[ordered]@{model=$onnxBinding;charset=$charsetBinding;contract=$studentContractBinding;student_model=$onnxBinding;student_charset=$charsetBinding;student_contract=$studentContractBinding}
        semantics=[ordered]@{analysis_candidate_only=$true;teacher_parity_only=$true;independent_business_accuracy_proven=$false;cpu_publication_performed=$false;cpu_delivery_gate_passed=$false;test_inference_performed=$false}
        validation=[ordered]@{fresh_no_resume_no_clobber=$true;teacher_sealed_train_val_test=$true;line_dataset_authorized_and_sealed=$true;generic_text_line_only=$true;generic_test_oov_fail_closed_by_source=$true;test_split_oov_zero=$true;test_split_used_for_training=$false;train_val_test_closed=$true;onnx_export_complete=$true;fixed_clone_import_attested=$true;gpu_idle_and_exact8_absent_before_train=$true;every_stage_rc_zero=$true;every_stage_stderr_zero_bytes=$true;every_stage_descendant_absence_proven=$true;process_tree_cleanup_on_failure=$true;git_and_nvidia_smi_bound_before_first_query_and_stable=$true;source_teacher_and_executables_stable=$true;onnx_sha_bound=$true}
        started_utc=$PipelineStartedUtc.ToString('o');completed_utc=[DateTime]::UtcNow.ToString('o')
    })
    $readback=Read-Json $pipelinePath
    if ([string]$readback.status -cne 'complete' -or $readback.semantics.analysis_candidate_only -ne $true -or $readback.semantics.cpu_publication_performed -ne $false) { throw 'Pipeline receipt semantic readback failed' }
    Write-Host "WHITE_TRAIN_PIPELINE_OK receipt=$pipelinePath candidate=$OnnxPath analysis_candidate_only=true cpu_publish=false"
}
catch {
    if ($RunRootOwned -and (Test-Path -LiteralPath $RunRoot -PathType Container)) {
        Assert-NoReparseChain $RunRoot 'owned failed white train root'
        if ((Get-DirectoryIdentity $RunRoot 'owned failed white train root') -cne $RunRootIdentity) { throw 'Refusing failure write after RunRoot identity changed' }
        $failurePath=Join-Path $RunRoot 'pipeline.failure.json'
        $stderrText=@(Get-ChildItem -LiteralPath $RunRoot -Filter 'stderr.txt' -File -Recurse -ErrorAction SilentlyContinue | ForEach-Object {[IO.File]::ReadAllText($_.FullName,$Utf8NoBom)}) -join "`n"
        $oomDetected=(([string]$_.Exception.Message + "`n" + $stderrText) -match '(?i)out of memory|cuda.*memory')
        if (-not (Test-Path -LiteralPath $failurePath)) { Write-JsonNew $failurePath ([ordered]@{schema_version=1;kind='otherimages_white_train_windows_failure_v1';status='failed';message=$_.Exception.Message;exception_type=$_.Exception.GetType().FullName;oom_detected=$oomDetected;resume_or_reuse_allowed=$false;preserved_root=$RunRoot;utc=[DateTime]::UtcNow.ToString('o')}) }
        Write-Host "WHITE_TRAIN_PIPELINE_FAILED evidence=$failurePath error=$($_.Exception.Message)"
    }
    throw
}
