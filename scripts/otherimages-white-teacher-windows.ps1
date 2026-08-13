<#
.SYNOPSIS
Captures and seals the frozen 1,000-image white-document Paddle teacher pilot.

.DESCRIPTION
Windows PowerShell 5.1 compatible.  This is a formal, no-resume wrapper for the
already-published white pilot at C:\f3-white-pilot-21054be8b1eb-a and the
independent source checkout pinned to commit 3080a69.  It performs one Paddle
process with --view-id all on CUDA, followed by the offline consensus process.
It never trains or exports a student model.
#>
[CmdletBinding()]
param(
    [string]$RepoRoot = 'C:\f3-white-code-3080a69',
    [string]$PilotRunRoot = 'C:\f3-white-pilot-21054be8b1eb-a',
    [string]$RunRoot = 'C:\f3-white-teacher-3080a69-pilot1000-a',
    [string]$PythonExe
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$Utf8NoBom = New-Object Text.UTF8Encoding($false)
$Ascii = [Text.Encoding]::ASCII
$RequiredHead = '3080a692a37d7efb0f926cce46de831d17f0e4db'
$RequiredTree = 'fb7a21f99139edd15eb1bb10e311039ebe28ebf5'
$RequiredProjectionSha256 = 'bd1b964117595a2e71b898d45f66393e5c15b92f863fbabd2f790499dbee009c'
$RequiredProjectionBytes = [int64]527892
$PipelineStartedUtc = [DateTime]::UtcNow
$RunRootOwned = $false
$PipelineFailed = $null
$GitExe = 'C:\Program Files\Git\cmd\git.exe'

$RequiredCode = [ordered]@{
    'scripts\otherimages-paddle-capture.py' = [ordered]@{ sha256='395ad109e260ba58f282023a75d439f93958b22e1159b476d98be0c4c3777308'; size_bytes=[int64]434 }
    'scripts\otherimages-paddle-teacher.py' = [ordered]@{ sha256='f191739b44134160ebdc63a85cabbf2ebd4037fde8ce3f7b10b0ea739b696d31'; size_bytes=[int64]445 }
    'scripts\otherimages-white-pilot-windows.ps1' = [ordered]@{ sha256='b8fda875d6169b83f82a259622ad2d7481f842144cd2878cdbd542b82e6994d4'; size_bytes=[int64]45534 }
    'src\transfer_receipt_ai\otherimages_inventory.py' = [ordered]@{ sha256='004c8e8e787e5ba684fdf4c3fa562df9317d549139441c01faee91646538691e'; size_bytes=[int64]67916 }
    'src\transfer_receipt_ai\ocr.py' = [ordered]@{ sha256='6c5ac75aec7d42eaac283b52dda86f3c440c74cde3f6f53799cf5206fe1373cb'; size_bytes=[int64]21386 }
    'src\transfer_receipt_ai\otherimages_paddle_capture.py' = [ordered]@{ sha256='470c2753c7fba63e1bd0e2e24e0a04ef7a3f523638933838995567395eae5494'; size_bytes=[int64]36293 }
    'src\transfer_receipt_ai\otherimages_paddle_teacher.py' = [ordered]@{ sha256='2155e7b1f49401ee49770241db2183a5ff7d02f34212afa9eb158f64132847c1'; size_bytes=[int64]102994 }
    'src\transfer_receipt_ai\otherimages_paddle_v2_adapter.py' = [ordered]@{ sha256='b193f20d4560643c89648019e151c39ecd0b53b42f29e55a7b4813df03e8202b'; size_bytes=[int64]16511 }
    'requirements-ocr.txt' = [ordered]@{ sha256='4118d4913d4d4256dbb8c47f853f14d37b87ace456ae5098acfff9e51208b4b4'; size_bytes=[int64]938 }
}

function Require-FixedPath([string]$Actual, [string]$Required, [string]$Description) {
    $full = [IO.Path]::GetFullPath($Actual)
    $expected = [IO.Path]::GetFullPath($Required)
    if (-not $full.Equals($expected,[StringComparison]::OrdinalIgnoreCase)) {
        throw "$Description must equal the frozen path: expected=$expected observed=$full"
    }
    if ([IO.Path]::GetPathRoot($full) -cne 'C:\') { throw "$Description must stay on C:\: $full" }
    return $full
}

function Write-TextNew([string]$Path, [string]$Text) {
    if (Test-Path -LiteralPath $Path) { throw "Refusing to overwrite evidence: $Path" }
    $parent = Split-Path -Parent $Path
    if (-not (Test-Path -LiteralPath $parent -PathType Container)) {
        [IO.Directory]::CreateDirectory($parent) | Out-Null
    }
    $stream = New-Object IO.FileStream($Path,[IO.FileMode]::CreateNew,[IO.FileAccess]::Write,[IO.FileShare]::None)
    try {
        $writer = New-Object IO.StreamWriter($stream,$Utf8NoBom)
        try { $writer.Write($Text); $writer.Flush() } finally { $writer.Dispose() }
    }
    finally { $stream.Dispose() }
}

function Write-JsonNew([string]$Path, [object]$Value) {
    Write-TextNew $Path (($Value | ConvertTo-Json -Depth 80) + "`r`n")
}

function Write-RcNew([string]$Path, [int]$Rc) {
    if (Test-Path -LiteralPath $Path) { throw "Refusing to overwrite RC evidence: $Path" }
    $text = $Rc.ToString([Globalization.CultureInfo]::InvariantCulture) + "`r`n"
    [byte[]]$expected = $Ascii.GetBytes($text)
    [IO.File]::WriteAllBytes($Path,$expected)
    [byte[]]$bytes = [IO.File]::ReadAllBytes($Path)
    if ($bytes.Length -ne $expected.Length) { throw "RC evidence length differs after readback: $Path" }
    for ($index=0; $index -lt $expected.Length; $index++) {
        if ($bytes[$index] -ne $expected[$index]) { throw "RC evidence differs from exact ASCII integer CRLF: $Path" }
    }
}

function Read-Json([string]$Path) {
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) { throw "Missing JSON evidence: $Path" }
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
    if ($null -ne ('WhiteTeacherNativeDirectoryV1' -as [type])) { return }
    $source = @'
using System;
using System.ComponentModel;
using System.Runtime.InteropServices;
using Microsoft.Win32.SafeHandles;
public static class WhiteTeacherNativeDirectoryV1 {
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
    return [WhiteTeacherNativeDirectoryV1]::Identity($item.FullName)
}

function Assert-BindingUnchanged([object]$Expected, [string]$Description) {
    $observed = Get-Binding ([string]$Expected.path)
    if ([string]$observed.sha256 -cne [string]$Expected.sha256 -or [int64]$observed.size_bytes -ne [int64]$Expected.size_bytes) {
        throw "$Description SHA/size authority changed"
    }
}

function Assert-GitAuthority([object]$ExpectedGitBinding) {
    Assert-BindingUnchanged $ExpectedGitBinding 'fixed Git executable before query'
    [string[]]$headLines = @(& $GitExe -C $RepoRoot rev-parse HEAD 2>&1)
    if ($LASTEXITCODE -ne 0 -or $headLines.Count -ne 1 -or $headLines[0] -cne $RequiredHead) {
        throw "Source checkout HEAD is not the frozen commit $RequiredHead"
    }
    [string[]]$treeLines = @(& $GitExe -C $RepoRoot rev-parse 'HEAD^{tree}' 2>&1)
    if ($LASTEXITCODE -ne 0 -or $treeLines.Count -ne 1 -or $treeLines[0] -cne $RequiredTree) { throw "Source checkout tree is not the frozen tree $RequiredTree" }
    [string[]]$statusLines = @(& $GitExe -C $RepoRoot status --porcelain=v1 --untracked-files=all 2>&1)
    if ($LASTEXITCODE -ne 0) { throw 'Unable to prove source checkout tracked/untracked status' }
    if ($statusLines.Count -ne 0) { throw 'Source checkout is not completely clean, including untracked files' }
    [string[]]$diffLines = @(& $GitExe -C $RepoRoot diff --no-ext-diff --quiet --exit-code HEAD -- 2>&1)
    if ($LASTEXITCODE -ne 0 -or $diffLines.Count -ne 0) { throw 'Source checkout tracked tree differs from HEAD' }
    Assert-BindingUnchanged $ExpectedGitBinding 'fixed Git executable after query'
}

function Get-CodeBindings {
    $result = [ordered]@{}
    foreach ($relative in $RequiredCode.Keys) {
        $path = Join-Path $RepoRoot $relative
        Assert-NoReparseChain $path 'frozen source file'
        $binding = Get-Binding $path
        $expected = $RequiredCode[$relative]
        if ([string]$binding.sha256 -cne [string]$expected.sha256 -or [int64]$binding.size_bytes -ne [int64]$expected.size_bytes) {
            throw "Frozen source SHA/size gate failed: $relative"
        }
        $result[$relative] = $binding
    }
    return $result
}

function Assert-NoConflictingWork([string]$Phase) {
    $conflicts = @(
        Get-CimInstance Win32_Process | Where-Object {
            [int]$_.ProcessId -ne [int]$PID -and
            -not [string]::IsNullOrWhiteSpace([string]$_.CommandLine) -and
            ([string]$_.CommandLine -match '(?i)receipt-ocr-recipient-multiview-exact8|f3e8|exact8|otherimages-paddle-capture\.py|transfer_receipt_ai\.otherimages_paddle_capture')
        }
    )
    if ($conflicts.Count -ne 0) {
        $summary = @($conflicts | ForEach-Object { ([string]$_.ProcessId) + ':' + ([string]$_.Name) }) -join ','
        throw "Refusing concurrent Exact8/OtherImages capture at $Phase: $summary"
    }
}

function Assert-FreeMemory {
    $operatingSystem = Get-CimInstance Win32_OperatingSystem
    [int64]$freeBytes = [int64]$operatingSystem.FreePhysicalMemory * 1024
    [int64]$minimumBytes = [int64]16 * 1024 * 1024 * 1024
    if ($freeBytes -lt $minimumBytes) { throw "CUDA capture requires at least 16 GiB free RAM; observed_bytes=$freeBytes" }
    return $freeBytes
}

function ConvertTo-NativeCommandLine([string[]]$Arguments) {
    $builder = New-Object Text.StringBuilder
    foreach ($argument in $Arguments) {
        if ($argument.IndexOf('"') -ge 0 -or $argument.EndsWith('\')) { throw "Unsupported native argument spelling: $argument" }
        if ($builder.Length -gt 0) { [void]$builder.Append(' ') }
        [void]$builder.Append('"'); [void]$builder.Append($argument); [void]$builder.Append('"')
    }
    return $builder.ToString()
}

function Invoke-PythonStage([string]$Name, [string[]]$Arguments, [hashtable]$Environment) {
    $stageRoot = Join-Path $LogsRoot $Name
    if (Test-Path -LiteralPath $stageRoot) { throw "Stage log root already exists: $stageRoot" }
    [IO.Directory]::CreateDirectory($stageRoot) | Out-Null
    $stdoutPath = Join-Path $stageRoot 'stdout.txt'
    $stderrPath = Join-Path $stageRoot 'stderr.txt'
    $rcPath = Join-Path $stageRoot 'rc.txt'
    $receiptPath = Join-Path $stageRoot 'stage.receipt.json'
    $info = New-Object Diagnostics.ProcessStartInfo
    $info.FileName = $PythonExe
    $info.Arguments = ConvertTo-NativeCommandLine $Arguments
    $info.WorkingDirectory = $RepoRoot
    $info.UseShellExecute = $false
    $info.CreateNoWindow = $true
    $info.RedirectStandardOutput = $true
    $info.RedirectStandardError = $true
    $info.EnvironmentVariables['PYTHONIOENCODING'] = 'utf-8:strict'
    $info.EnvironmentVariables['PYTHONUTF8'] = '1'
    $info.EnvironmentVariables['PYTHONDONTWRITEBYTECODE'] = '1'
    foreach ($key in $Environment.Keys) { $info.EnvironmentVariables[[string]$key] = [string]$Environment[$key] }
    $process = New-Object Diagnostics.Process
    $process.StartInfo = $info
    $started = [DateTime]::UtcNow
    $startedProcess = $false
    $pidValue = $null
    $stdoutTask = $null
    $stderrTask = $null
    $exitCode = $null
    $stageError = $null
    $cleanupError = $null
    $forcedStop = $false
    try {
        Write-Host "WHITE_TEACHER_STAGE_START stage=$Name requested_device=$($Environment['OTHERIMAGES_PADDLE_DEVICE'])"
        if (-not $process.Start()) { throw "Unable to start Python stage: $Name" }
        $startedProcess = $true
        $pidValue = [int]$process.Id
        $stdoutTask = $process.StandardOutput.ReadToEndAsync()
        $stderrTask = $process.StandardError.ReadToEndAsync()
        $lastReport = $started
        while (-not $process.WaitForExit(1000)) {
            $now = [DateTime]::UtcNow
            if (($now-$lastReport).TotalSeconds -ge 60) {
                try {
                    $process.Refresh()
                    Write-Host ("WHITE_TEACHER_STAGE_ALIVE stage={0} elapsed_s={1} pid={2} cpu_s={3:N1} ws_bytes={4}" -f $Name,[int]($now-$started).TotalSeconds,$pidValue,$process.TotalProcessorTime.TotalSeconds,$process.WorkingSet64)
                }
                catch { Write-Host "WHITE_TEACHER_STAGE_ALIVE stage=$Name elapsed_s=$([int]($now-$started).TotalSeconds) pid=$pidValue" }
                $lastReport = $now
            }
        }
        $process.WaitForExit()
        $exitCode = [int]$process.ExitCode
    }
    catch { $stageError = $_ }
    finally {
        try {
            if ($startedProcess) {
                if (-not $process.HasExited) {
                    $forcedStop = $true
                    try { $process.Kill() } catch { $cleanupError = $_ }
                    if (-not $process.WaitForExit(30000)) {
                        try { $process.Kill() } catch { if ($null -eq $cleanupError) { $cleanupError = $_ } }
                        if (-not $process.WaitForExit(30000)) {
                            if ($null -eq $cleanupError) {
                                try { throw 'Python child remained alive after two kill and bounded-wait attempts' }
                                catch { $cleanupError = $_ }
                            }
                        }
                    }
                }
                if ($process.HasExited) {
                    $process.WaitForExit()
                    $exitCode = [int]$process.ExitCode
                }
            }
        }
        catch { if ($null -eq $cleanupError) { $cleanupError = $_ } }
        try {
            foreach ($streamTask in @($stdoutTask,$stderrTask)) {
                if ($null -ne $streamTask -and -not $streamTask.IsCompleted -and -not $streamTask.Wait(30000)) {
                    if ($null -eq $cleanupError) {
                        try { throw 'Python redirected stream did not close within the bounded wait' }
                        catch { $cleanupError = $_ }
                    }
                }
            }
            $stdout = if ($null -ne $stdoutTask -and $stdoutTask.IsCompleted) { [string]$stdoutTask.Result } else { '' }
            $stderr = if ($null -ne $stderrTask -and $stderrTask.IsCompleted) { [string]$stderrTask.Result } else { '' }
            [IO.File]::WriteAllText($stdoutPath,$stdout,$Utf8NoBom)
            [IO.File]::WriteAllText($stderrPath,$stderr,$Utf8NoBom)
            if ($null -ne $exitCode) { Write-RcNew $rcPath ([int]$exitCode) }
            $failureEvidencePath = Join-Path $stageRoot 'stage.failure.json'
            if ($null -ne $stageError -or $null -ne $cleanupError -or $null -eq $exitCode -or [int]$exitCode -ne 0) {
                Write-JsonNew $failureEvidencePath ([ordered]@{
                    schema_version=1; kind='otherimages_white_teacher_windows_stage_failure_v1'; status='failed'; stage=$Name
                    pid=$pidValue; process_started=$startedProcess; forced_stop=$forcedStop; exit_code=$exitCode
                    stage_error=$(if ($null -eq $stageError) { $null } else { $stageError.Exception.Message })
                    cleanup_error=$(if ($null -eq $cleanupError) { $null } else { $cleanupError.Exception.Message })
                    stdout=$(Get-Binding $stdoutPath); stderr=$(Get-Binding $stderrPath)
                    rc_file=$(if (Test-Path -LiteralPath $rcPath -PathType Leaf) { Get-Binding $rcPath } else { $null })
                    utc=[DateTime]::UtcNow.ToString('o')
                })
            }
        }
        catch { if ($null -eq $cleanupError) { $cleanupError = $_ } }
        $process.Dispose()
    }
    if ($null -ne $stageError) { throw $stageError }
    if ($null -ne $cleanupError) { throw $cleanupError }
    if ($null -eq $exitCode) { throw "Python stage exit code is unavailable: stage=$Name evidence=$stageRoot" }
    $result = [pscustomobject][ordered]@{
        name=$Name; pid=$pidValue; rc=[int]$exitCode
        started_utc=$started.ToString('o'); completed_utc=[DateTime]::UtcNow.ToString('o')
        elapsed_seconds=([DateTime]::UtcNow-$started).TotalSeconds
        stdout=Get-Binding $stdoutPath; stderr=Get-Binding $stderrPath; rc_file=Get-Binding $rcPath
        receipt_path=$receiptPath
    }
    if ($result.rc -ne 0) { throw "Python stage failed: stage=$Name rc=$($result.rc) evidence=$stageRoot" }
    if ($result.stderr.size_bytes -ne 0) { throw "Python stage emitted stderr: stage=$Name evidence=$stderrPath" }
    Write-Host "WHITE_TEACHER_STAGE_EXIT stage=$Name rc=0 elapsed_s=$([int]$result.elapsed_seconds)"
    return $result
}

function Complete-Stage([object]$Stage, [object]$Validation) {
    $receipt = [ordered]@{
        schema_version=1; kind='otherimages_white_teacher_windows_stage_receipt_v1'; status='complete'; stage=[string]$Stage.name
        process=[ordered]@{ pid=[int]$Stage.pid; rc=[int]$Stage.rc; started_utc=$Stage.started_utc; completed_utc=$Stage.completed_utc; elapsed_seconds=$Stage.elapsed_seconds }
        stdout=$Stage.stdout; stderr=$Stage.stderr; rc_file=$Stage.rc_file; validation=$Validation
    }
    Write-JsonNew $Stage.receipt_path $receipt
    return Get-Binding $Stage.receipt_path
}

function Assert-ExactFileTree([string]$Root, [string[]]$Expected, [string]$Description) {
    [string[]]$files = @(Get-ChildItem -LiteralPath $Root -File -Force | ForEach-Object { $_.Name } | Sort-Object)
    [string[]]$directories = @(Get-ChildItem -LiteralPath $Root -Directory -Force | ForEach-Object { $_.Name })
    [string[]]$sortedExpected = @($Expected | Sort-Object)
    if (($files -join '|') -cne ($sortedExpected -join '|') -or $directories.Count -ne 0) {
        throw "$Description is not its exact closed file tree"
    }
}

$TeacherVerifierSource = @'
from __future__ import annotations
import argparse, hashlib, json, math, os, stat
from pathlib import Path

VIEWS = {"original_rgb", "grayscale_clahe", "upscale_sharpen"}
FILES = {"teacher_manifest.jsonl", "reject_manifest.jsonl", "teacher.contract.json", "teacher.receipt.json"}

def fail(message):
    raise SystemExit("independent teacher closure verification failed: " + message)

def pairs(items):
    result = {}
    for key, value in items:
        if key in result: fail("duplicate JSON key " + repr(key))
        result[key] = value
    return result

def constant(value):
    fail("non-standard JSON constant " + repr(value))

def load_json(path):
    data = path.read_bytes()
    if data.startswith(b"\xef\xbb\xbf"): fail("UTF-8 BOM: " + str(path))
    try:
        value = json.loads(data.decode("utf-8"), object_pairs_hook=pairs, parse_constant=constant)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        fail("invalid strict JSON at %s: %s" % (path, error))
    if not isinstance(value, dict): fail("JSON root is not an object: " + str(path))
    return value

def reject_reparse(path):
    current = path.absolute()
    while True:
        if current.exists():
            status = current.lstat()
            if stat.S_ISLNK(status.st_mode) or int(getattr(status, "st_file_attributes", 0)) & 0x400:
                fail("path traverses reparse point: " + str(current))
        if current.parent == current: return
        current = current.parent

def binding(path, *, public):
    reject_reparse(path)
    if not path.is_file(): fail("missing regular file: " + str(path))
    data = path.read_bytes()
    return {"path": str(path.resolve()) if public else path.name,
            "sha256": hashlib.sha256(data).hexdigest(), "size_bytes": len(data),
            "line_count": data.count(b"\n")}

def public_binding(path):
    value = binding(path, public=True)
    value.pop("line_count")
    return value

def same_path(left, right):
    return os.path.normcase(os.path.abspath(os.fspath(left))) == os.path.normcase(os.path.abspath(os.fspath(right)))

def load_jsonl(path):
    data = path.read_bytes()
    if data.startswith(b"\xef\xbb\xbf"): fail("UTF-8 BOM: " + str(path))
    try: text = data.decode("utf-8")
    except UnicodeDecodeError as error: fail("invalid UTF-8 JSONL at %s: %s" % (path, error))
    rows = []
    for number, line in enumerate(text.splitlines(), 1):
        if not line: fail("blank JSONL line at %s:%d" % (path, number))
        try: value = json.loads(line, object_pairs_hook=pairs, parse_constant=constant)
        except json.JSONDecodeError as error: fail("invalid JSONL at %s:%d: %s" % (path, number, error))
        if not isinstance(value, dict): fail("non-object JSONL row at %s:%d" % (path, number))
        rows.append(value)
    if text and not text.endswith("\n"): fail("JSONL lacks final LF: " + str(path))
    return rows

parser = argparse.ArgumentParser()
parser.add_argument("--teacher-root", type=Path, required=True)
parser.add_argument("--capture-root", type=Path, required=True)
parser.add_argument("--capture-receipt", type=Path, required=True)
args = parser.parse_args()
teacher_root, capture_root = args.teacher_root.absolute(), args.capture_root.absolute()
for root in (teacher_root, capture_root):
    reject_reparse(root)
    if not root.is_dir(): fail("missing publication directory: " + str(root))
members = {item.name for item in teacher_root.iterdir()}
if members != FILES or any(not item.is_file() for item in teacher_root.iterdir()): fail("teacher exact four-file tree differs")
capture_members = {item.name for item in capture_root.iterdir()}
if capture_members != {view + ".jsonl" for view in VIEWS} or any(not item.is_file() for item in capture_root.iterdir()): fail("capture exact three-file tree differs")

capture_receipt = load_json(args.capture_receipt)
if capture_receipt.get("kind") != "otherimages_paddle_three_view_capture_receipt_v2" or not same_path(capture_receipt.get("output_directory", ""), capture_root): fail("capture receipt root/kind differs")
capture_views = capture_receipt.get("views")
if not isinstance(capture_views, list) or len(capture_views) != 3: fail("capture receipt view count differs")
capture_by_view = {}
for item in capture_views:
    if not isinstance(item, dict) or set(item) != {"view_id", "path", "sha256", "size_bytes", "line_count"}: fail("capture receipt view binding shape differs")
    view_id = item.get("view_id")
    if view_id not in VIEWS or view_id in capture_by_view: fail("capture receipt view identities differ")
    path = capture_root / (view_id + ".jsonl")
    actual = binding(path, public=True)
    if not same_path(item["path"], path) or {key: item[key] for key in ("sha256", "size_bytes", "line_count")} != {key: actual[key] for key in ("sha256", "size_bytes", "line_count")} or actual["line_count"] != 999: fail("capture view path/SHA/size/line_count differs: " + view_id)
    capture_by_view[view_id] = actual

contract_path, receipt_path = teacher_root / "teacher.contract.json", teacher_root / "teacher.receipt.json"
manifest_path, reject_path = teacher_root / "teacher_manifest.jsonl", teacher_root / "reject_manifest.jsonl"
contract, receipt = load_json(contract_path), load_json(receipt_path)
if contract.get("kind") != "otherimages_paddle_teacher_contract_v1" or not same_path(contract.get("output_directory", ""), teacher_root): fail("teacher contract output_directory/kind differs")
inputs = contract.get("inputs")
if not isinstance(inputs, dict) or not isinstance(inputs.get("views"), list) or len(inputs["views"]) != 3: fail("teacher input views differ")
seen = set()
for item in inputs["views"]:
    if not isinstance(item, dict) or item.get("view_id") not in VIEWS or item["view_id"] in seen: fail("teacher input view IDs differ")
    view_id, result = item["view_id"], item.get("result")
    expected = public_binding(capture_root / (view_id + ".jsonl"))
    if not isinstance(result, dict) or set(result) != {"path", "sha256", "size_bytes"} or not same_path(result["path"], expected["path"]) or {key: result[key] for key in ("sha256", "size_bytes")} != {key: expected[key] for key in ("sha256", "size_bytes")}: fail("teacher view result path/SHA/size differs: " + view_id)
    if capture_by_view[view_id]["line_count"] != 999: fail("teacher view source line count differs: " + view_id)
    seen.add(view_id)
if seen != VIEWS: fail("teacher input canonical view closure differs")

teacher_rows, reject_rows = load_jsonl(manifest_path), load_jsonl(reject_path)
artifacts = contract.get("artifacts")
expected_artifacts = [binding(manifest_path, public=False), binding(reject_path, public=False)]
if artifacts != expected_artifacts: fail("teacher manifest/reject artifact bindings differ")
counts = contract.get("counts")
if not isinstance(counts, dict) or counts.get("accepted_teacher_records") != len(teacher_rows) or counts.get("quarantined_records") != len(reject_rows) or len(teacher_rows) + len(reject_rows) != 1000: fail("teacher manifest/reject counts differ")
closure = {key: contract.get(key) for key in ("schema_version", "inputs", "configuration", "counts", "split_use", "artifacts")}
closure_sha = hashlib.sha256(json.dumps(closure, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()
if contract.get("closure_sha256") != closure_sha: fail("teacher canonical closure SHA-256 differs")
contract_binding = binding(contract_path, public=False)
if set(receipt) != {"schema_version", "kind", "sealed", "contract", "contract_closure_sha256"} or receipt.get("kind") != "otherimages_paddle_teacher_receipt_v1" or receipt.get("sealed") is not True or receipt.get("contract") != contract_binding or receipt.get("contract_closure_sha256") != closure_sha: fail("teacher receipt contract/closure binding differs")
print(json.dumps({"schema_version": 1, "kind": "otherimages_white_teacher_independent_closure_v1", "status": "complete", "closure_sha256": closure_sha, "accepted": len(teacher_rows), "quarantined": len(reject_rows), "views": sorted(seen)}, sort_keys=True, separators=(",", ":"), allow_nan=False))
'@

try {
    $RepoRoot = Require-FixedPath $RepoRoot 'C:\f3-white-code-3080a69' 'source checkout'
    $PilotRunRoot = Require-FixedPath $PilotRunRoot 'C:\f3-white-pilot-21054be8b1eb-a' 'pilot run root'
    $RunRoot = Require-FixedPath $RunRoot 'C:\f3-white-teacher-3080a69-pilot1000-a' 'teacher run root'
    if ([string]::IsNullOrWhiteSpace($PythonExe)) { $PythonExe = Join-Path $RepoRoot '.venv-cu126\Scripts\python.exe' }
    $PythonExe = [IO.Path]::GetFullPath($PythonExe)
    if (-not $PythonExe.Equals((Join-Path $RepoRoot '.venv-cu126\Scripts\python.exe'),[StringComparison]::OrdinalIgnoreCase)) { throw 'PythonExe must be the fixed CUDA environment in the independent checkout' }
    foreach ($path in @($RepoRoot,$PilotRunRoot,$PythonExe,$GitExe,$PSCommandPath)) {
        if (-not (Test-Path -LiteralPath $path)) { throw "Missing required authority path: $path" }
        Assert-NoReparseChain $path 'white teacher authority path'
    }
    if (Test-Path -LiteralPath $RunRoot) { throw "RunRoot must be brand-new: $RunRoot" }
    $RunParent = Split-Path -Parent $RunRoot
    Assert-NoReparseChain $RunParent 'white teacher output parent'
    Initialize-NativeDirectoryType
    $RunParentIdentity = Get-DirectoryIdentity $RunParent 'white teacher output parent'
    $GitBinding = Get-Binding $GitExe
    Assert-GitAuthority $GitBinding
    $CodeBindings = Get-CodeBindings
    $WrapperBinding = Get-Binding $PSCommandPath
    $PythonBinding = Get-Binding $PythonExe

    $PilotReceiptPath = Join-Path $PilotRunRoot 'pipeline.receipt.json'
    $InventoryRoot = Join-Path $PilotRunRoot 'inventory-prefix1000'
    $InventoryManifest = Join-Path $InventoryRoot 'paddle_teacher_pending.jsonl'
    $InventoryContractPath = Join-Path $InventoryRoot 'inventory.contract.json'
    $ProjectionPath = Join-Path $PilotRunRoot 'logs\inventory-portable-projection.jsonl'
    foreach ($path in @($PilotReceiptPath,$InventoryManifest,$InventoryContractPath,$ProjectionPath)) {
        if (-not (Test-Path -LiteralPath $path -PathType Leaf)) { throw "Missing pilot authority file: $path" }
        Assert-NoReparseChain $path 'pilot authority file'
    }
    $pilot = Read-Json $PilotReceiptPath
    if ([string]$pilot.kind -cne 'otherimages_white_pilot_windows_pipeline_receipt_v1' -or [string]$pilot.status -cne 'complete') { throw 'Pilot pipeline receipt kind/status failed' }
    if (-not ([IO.Path]::GetFullPath([string]$pilot.roots.run)).Equals($PilotRunRoot,[StringComparison]::OrdinalIgnoreCase)) { throw 'Pilot receipt run root differs from the frozen pilot path' }
    if ([string]$pilot.code.wrapper.sha256 -cne [string]$RequiredCode['scripts\otherimages-white-pilot-windows.ps1'].sha256 -or [string]$pilot.code.inventory_module.sha256 -cne [string]$RequiredCode['src\transfer_receipt_ai\otherimages_inventory.py'].sha256) { throw 'Pilot receipt is not bound to the frozen wrapper/inventory source' }
    if ($pilot.analysis_only -ne $true -or $pilot.training_performed -ne $false -or $pilot.ocr_performed -ne $false) { throw 'Pilot receipt analysis-only policy failed' }
    if ([int]$pilot.validation.inventory_image_count -ne 1000 -or [int]$pilot.validation.inventory_decode_errors -ne 0) { throw 'Pilot receipt inventory count/error gate failed' }
    if ([int]$pilot.validation.suggested_splits.train -ne 912 -or [int]$pilot.validation.suggested_splits.val -ne 52 -or [int]$pilot.validation.suggested_splits.test -ne 36) { throw 'Pilot receipt frozen split gate failed' }
    if ([int]$pilot.validation.teacher_states.pending -ne 999 -or [int]$pilot.validation.teacher_states.quarantine -ne 1) { throw 'Pilot receipt frozen teacher-state gate failed' }
    if ([string]$pilot.validation.portable_projection_sha256 -cne $RequiredProjectionSha256 -or [int64]$pilot.validation.portable_projection_size_bytes -ne $RequiredProjectionBytes) { throw 'Pilot receipt projection binding failed' }
    foreach ($flag in @('every_stage_rc_zero','every_stage_stderr_zero_bytes','core_code_archive_prepare_stable','run_root_exclusive_and_identity_stable','no_clobber_publications')) {
        if ($pilot.validation.$flag -ne $true) { throw "Pilot receipt validation flag failed: $flag" }
    }
    $inventory = Read-Json $InventoryContractPath
    if ([string]$inventory.kind -cne 'otherimages_read_only_inventory_v1' -or [int]$inventory.counts.images -ne 1000 -or [int]$inventory.counts.image_errors_quarantined -ne 0) { throw 'Inventory exact count/error gate failed' }
    if ([int]$inventory.counts.suggested_splits.train -ne 912 -or [int]$inventory.counts.suggested_splits.val -ne 52 -or [int]$inventory.counts.suggested_splits.test -ne 36) { throw 'Inventory exact split gate failed' }
    if ([int]$inventory.counts.teacher_states.pending -ne 999 -or [int]$inventory.counts.teacher_states.quarantine -ne 1) { throw 'Inventory exact teacher-state gate failed' }
    $expectedInventoryInput = Join-Path $PilotRunRoot 'pilot\white-pilot-1000-21054be8b1eb-local-a\images'
    if (-not ([IO.Path]::GetFullPath([string]$inventory.source.input_directory)).Equals([IO.Path]::GetFullPath($expectedInventoryInput),[StringComparison]::OrdinalIgnoreCase)) { throw 'Inventory source root differs from the frozen pilot image publication' }
    if ($inventory.paddle_teacher_contract.inventory_contains_labels -ne $false -or $inventory.paddle_teacher_contract.inventory_performed_ocr -ne $false -or $inventory.paddle_teacher_contract.guessed_or_synthetic_labels_forbidden -ne $true -or $inventory.paddle_teacher_contract.training_eligibility_before_teacher_validation -ne $false) { throw 'Inventory unlabeled teacher-pending contract failed' }
    Assert-ExactFileTree $InventoryRoot @('errors.jsonl','exact_duplicates.jsonl','ignored_non_images.jsonl','images.jsonl','inventory.contract.json','layout_sample.jsonl','near_duplicate_candidates.jsonl','paddle_teacher_pending.jsonl') 'Inventory publication'
    $projectionBinding = Get-Binding $ProjectionPath
    if ([string]$projectionBinding.sha256 -cne $RequiredProjectionSha256 -or [int64]$projectionBinding.size_bytes -ne $RequiredProjectionBytes) { throw 'Inventory bd1 portable projection SHA/size gate failed' }
    $PilotBindings = [ordered]@{ pipeline_receipt=Get-Binding $PilotReceiptPath; inventory_manifest=Get-Binding $InventoryManifest; inventory_contract=Get-Binding $InventoryContractPath; portable_projection=$projectionBinding }

    $FreeMemoryBytes = Assert-FreeMemory
    Assert-NoConflictingWork 'preflight'
    [WhiteTeacherNativeDirectoryV1]::CreateExclusive($RunRoot)
    $RunRootOwned = $true
    $RunRootIdentity = Get-DirectoryIdentity $RunRoot 'white teacher run root'
    if ((Get-DirectoryIdentity $RunParent 'white teacher output parent after creation') -cne $RunParentIdentity) { throw 'Teacher output parent identity changed during exclusive creation' }
    $LogsRoot = Join-Path $RunRoot 'logs'
    $PublicationsRoot = Join-Path $RunRoot 'publications'
    [IO.Directory]::CreateDirectory($LogsRoot) | Out-Null
    [IO.Directory]::CreateDirectory($PublicationsRoot) | Out-Null
    $TeacherVerifierPath = Join-Path $LogsRoot 'independent-teacher-closure.py'
    Write-TextNew $TeacherVerifierPath ($TeacherVerifierSource + "`n")
    $TeacherVerifierBinding = Get-Binding $TeacherVerifierPath
    $CaptureOutput = Join-Path $PublicationsRoot 'paddle-three-view-capture'
    $TeacherOutput = Join-Path $PublicationsRoot 'paddle-teacher-consensus'
    if (Test-Path -LiteralPath $CaptureOutput) { throw 'Capture output must be brand-new' }
    if (Test-Path -LiteralPath $TeacherOutput) { throw 'Teacher output must be brand-new' }

    Assert-NoConflictingWork 'immediately-before-capture'
    $captureScript = Join-Path $RepoRoot 'scripts\otherimages-paddle-capture.py'
    $captureStage = Invoke-PythonStage 'capture-three-view' @(
        $captureScript,'--inventory',$InventoryManifest,'--inventory-contract',$InventoryContractPath,
        '--output',$CaptureOutput,'--view-id','all','--json'
    ) @{ OTHERIMAGES_PADDLE_DEVICE='cuda' }
    $captureText = [IO.File]::ReadAllText([string]$captureStage.stdout.path,$Utf8NoBom)
    if ([string]::IsNullOrWhiteSpace($captureText)) { throw 'Capture stdout must contain exactly one complete JSON value' }
    $capture = $captureText | ConvertFrom-Json
    if ([string]$capture.kind -cne 'otherimages_paddle_three_view_capture_receipt_v2' -or [int]$capture.records_per_view -ne 999 -or [int]$capture.capture_errors -ne 0) { throw 'Capture receipt kind/count/error gate failed' }
    if ([string]$capture.output_directory -cne $CaptureOutput) { throw 'Capture receipt output directory differs' }
    if ([string]$capture.adapter.execution_device -cne 'gpu:0' -or $capture.adapter.effective_paddle_args.use_gpu -ne $true -or [int]$capture.adapter.effective_paddle_args.gpu_id -ne 0) { throw 'Capture did not honor explicit cuda/gpu:0 execution' }
    if ($capture.adapter.stages.db -ne $true -or $capture.adapter.stages.cls -ne $true -or $capture.adapter.stages.rec -ne $true) { throw 'Capture adapter DB/CLS/REC contract failed' }
    [string[]]$captureFiles = @('original_rgb.jsonl','grayscale_clahe.jsonl','upscale_sharpen.jsonl')
    Assert-ExactFileTree $CaptureOutput $captureFiles 'Three-view capture publication'
    if (@($capture.views).Count -ne 3 -or (@($capture.views | ForEach-Object { [string]$_.view_id } | Sort-Object) -join '|') -cne 'grayscale_clahe|original_rgb|upscale_sharpen') { throw 'Capture receipt does not bind exactly the three canonical views' }
    foreach ($view in @($capture.views)) {
        if (@('original_rgb','grayscale_clahe','upscale_sharpen') -cnotcontains [string]$view.view_id -or [int64]$view.line_count -ne 999) { throw 'Capture view identity/row count gate failed' }
        $observed = Get-Binding ([string]$view.path)
        if ([string]$observed.sha256 -cne [string]$view.sha256 -or [int64]$observed.size_bytes -ne [int64]$view.size_bytes) { throw 'Capture view SHA/size readback failed' }
    }
    $captureStdoutReceiptPath = Join-Path (Split-Path -Parent $captureStage.receipt_path) 'capture.stdout.receipt.json'
    Write-JsonNew $captureStdoutReceiptPath $capture
    $captureStageReceipt = Complete-Stage $captureStage ([ordered]@{ requested_device='cuda'; execution_device='gpu:0'; one_process_view_id_all=$true; records_per_view=999; capture_errors=0; exact_three_file_tree=$true; db_cls_rec=$true; formal_capture_receipt=Get-Binding $captureStdoutReceiptPath })

    $teacherScript = Join-Path $RepoRoot 'scripts\otherimages-paddle-teacher.py'
    $teacherStage = Invoke-PythonStage 'teacher-consensus' @(
        $teacherScript,'--inventory',$InventoryManifest,'--inventory-contract',$InventoryContractPath,
        '--view-result',(Join-Path $CaptureOutput 'original_rgb.jsonl'),
        '--view-result',(Join-Path $CaptureOutput 'grayscale_clahe.jsonl'),
        '--view-result',(Join-Path $CaptureOutput 'upscale_sharpen.jsonl'),
        '--output',$TeacherOutput
    ) @{}
    [string[]]$teacherLines = @([IO.File]::ReadAllLines([string]$teacherStage.stdout.path,$Utf8NoBom) | Where-Object { -not [string]::IsNullOrWhiteSpace($_) })
    if ($teacherLines.Count -ne 1 -or $teacherLines[0] -notmatch '^Sealed [0-9]+ Paddle teacher record\(s\); quarantined [0-9]+ record\(s\) at ') { throw 'Teacher stdout is not its one exact success summary' }
    Assert-ExactFileTree $TeacherOutput @('teacher_manifest.jsonl','reject_manifest.jsonl','teacher.contract.json','teacher.receipt.json') 'Teacher publication'
    $teacherContractPath = Join-Path $TeacherOutput 'teacher.contract.json'
    $teacherReceiptPath = Join-Path $TeacherOutput 'teacher.receipt.json'
    $teacher = Read-Json $teacherContractPath
    $teacherReceipt = Read-Json $teacherReceiptPath
    if ([string]$teacher.kind -cne 'otherimages_paddle_teacher_contract_v1' -or $teacher.sealed -ne $true -or $teacher.training_authorization -ne $false -or $teacher.ocr_execution_performed_by_this_module -ne $false) { throw 'Teacher sealed/non-training contract gate failed' }
    if ([int]$teacher.counts.inventory_records -ne 1000 -or [int]$teacher.counts.pending_records -ne 999) { throw 'Teacher input count gate failed' }
    if (([int]$teacher.counts.accepted_teacher_records + [int]$teacher.counts.quarantined_records) -ne 1000) { throw 'Teacher accepted/quarantined closure does not equal 1000' }
    if ([int]$teacher.counts.accepted_teacher_records -lt 1 -or [int]$teacher.counts.training_eligible_records -lt 1 -or [int]$teacher.counts.evaluation_only_records -lt 1) { throw 'Teacher pilot produced no usable train or held-out evaluation evidence' }
    if ([int]$teacher.counts.quarantined_records -lt 1) { throw 'Teacher output omitted the frozen inventory quarantine record' }
    if (([int]$teacher.counts.training_eligible_records + [int]$teacher.counts.evaluation_only_records) -ne [int]$teacher.counts.accepted_teacher_records) { throw 'Teacher train/evaluation count contract failed' }
    if ([string]$teacher.split_use.train -cne 'training_eligible' -or [string]$teacher.split_use.val -cne 'heldout_evaluation_only' -or [string]$teacher.split_use.test -cne 'heldout_evaluation_only' -or $teacher.split_use.groups_may_cross_splits -ne $false) { throw 'Teacher split-use contract failed' }
    if ([string]$teacher.low_confidence_or_conflict_policy -cne 'quarantine_never_guess' -or $teacher.manual_review_required -ne $false) { throw 'Teacher no-guess/no-manual-label policy failed' }
    if (@($teacher.inputs.views).Count -ne 3) { throw 'Teacher contract must bind exactly three captured views' }
    foreach ($view in @($teacher.inputs.views)) {
        if ([string]$view.adapter.execution_device -cne 'gpu:0' -or $view.adapter.effective_paddle_args.use_gpu -ne $true -or [int]$view.adapter.effective_paddle_args.gpu_id -ne 0) { throw 'Teacher input view is not bound to cuda/gpu:0 capture' }
    }
    if ([string]$teacherReceipt.kind -cne 'otherimages_paddle_teacher_receipt_v1' -or $teacherReceipt.sealed -ne $true -or [string]$teacherReceipt.contract_closure_sha256 -cne [string]$teacher.closure_sha256) { throw 'Teacher receipt closure gate failed' }
    $contractBinding = Get-Binding $teacherContractPath
    if ([string]$teacherReceipt.contract.sha256 -cne [string]$contractBinding.sha256 -or [int64]$teacherReceipt.contract.size_bytes -ne [int64]$contractBinding.size_bytes) { throw 'Teacher receipt contract SHA/size binding failed' }
    foreach ($artifact in @($teacher.artifacts)) {
        $observed = Get-Binding (Join-Path $TeacherOutput ([string]$artifact.path))
        if ([string]$observed.sha256 -cne [string]$artifact.sha256 -or [int64]$observed.size_bytes -ne [int64]$artifact.size_bytes) { throw 'Teacher artifact SHA/size readback failed' }
    }
    $independentStage = Invoke-PythonStage 'teacher-independent-verify' @(
        $TeacherVerifierPath,'--teacher-root',$TeacherOutput,'--capture-root',$CaptureOutput,
        '--capture-receipt',([string]$captureStage.stdout.path)
    ) @{}
    [string[]]$independentLines = @([IO.File]::ReadAllLines([string]$independentStage.stdout.path,$Utf8NoBom) | Where-Object { -not [string]::IsNullOrWhiteSpace($_) })
    if ($independentLines.Count -ne 1) { throw 'Independent teacher verifier stdout must be one complete JSON value' }
    $independent = $independentLines[0] | ConvertFrom-Json
    if ([string]$independent.kind -cne 'otherimages_white_teacher_independent_closure_v1' -or [string]$independent.status -cne 'complete' -or [string]$independent.closure_sha256 -cne [string]$teacher.closure_sha256 -or ([int]$independent.accepted + [int]$independent.quarantined) -ne 1000) { throw 'Independent teacher closure receipt failed' }
    $independentReceiptPath = Join-Path (Split-Path -Parent $independentStage.receipt_path) 'independent.stdout.receipt.json'
    Write-JsonNew $independentReceiptPath $independent
    $independentStageReceipt = Complete-Stage $independentStage ([ordered]@{ canonical_closure_recomputed=$true; output_directory_bound=$true; exact_view_ids_paths_sha_size_line_count=$true; manifest_reject_contract_receipt_artifacts_bound=$true; exact_teacher_tree=$true; closure_sha256=[string]$independent.closure_sha256; formal_receipt=Get-Binding $independentReceiptPath; verifier_source=$TeacherVerifierBinding })
    [string[]]$publicationFiles = @(Get-ChildItem -LiteralPath $PublicationsRoot -File -Force | ForEach-Object { $_.Name })
    [string[]]$publicationDirectories = @(Get-ChildItem -LiteralPath $PublicationsRoot -Directory -Force | ForEach-Object { $_.Name } | Sort-Object)
    if ($publicationFiles.Count -ne 0 -or ($publicationDirectories -join '|') -cne 'paddle-teacher-consensus|paddle-three-view-capture') { throw 'Publication root contains a retained sibling or unexpected member' }
    $teacherStageReceipt = Complete-Stage $teacherStage ([ordered]@{ sealed=$true; inventory_records=1000; pending_records=999; accepted_teacher_records=[int]$teacher.counts.accepted_teacher_records; quarantined_records=[int]$teacher.counts.quarantined_records; accepted_plus_quarantined=1000; training_authorization=$false; quarantine_never_guess=$true; exact_four_file_tree=$true; contract=Get-Binding $teacherContractPath; formal_receipt=Get-Binding $teacherReceiptPath })

    Assert-BindingUnchanged $GitBinding 'fixed Git executable before final source query'
    Assert-GitAuthority $GitBinding
    $CodeBindingsAfter = Get-CodeBindings
    foreach ($name in $CodeBindings.Keys) {
        if ([string]$CodeBindings[$name].sha256 -cne [string]$CodeBindingsAfter[$name].sha256 -or [int64]$CodeBindings[$name].size_bytes -ne [int64]$CodeBindingsAfter[$name].size_bytes) { throw "Source code changed during teacher pipeline: $name" }
    }
    $WrapperAfter = Get-Binding $PSCommandPath
    if ([string]$WrapperAfter.sha256 -cne [string]$WrapperBinding.sha256 -or [int64]$WrapperAfter.size_bytes -ne [int64]$WrapperBinding.size_bytes) { throw 'Wrapper changed during teacher pipeline' }
    $PythonAfter = Get-Binding $PythonExe
    if ([string]$PythonAfter.sha256 -cne [string]$PythonBinding.sha256 -or [int64]$PythonAfter.size_bytes -ne [int64]$PythonBinding.size_bytes) { throw 'Python executable changed during teacher pipeline' }
    Assert-BindingUnchanged $TeacherVerifierBinding 'independent teacher verifier'
    foreach ($name in $PilotBindings.Keys) {
        $before = $PilotBindings[$name]
        $after = Get-Binding ([string]$before.path)
        if ([string]$before.sha256 -cne [string]$after.sha256 -or [int64]$before.size_bytes -ne [int64]$after.size_bytes) { throw "Pilot authority changed during teacher pipeline: $name" }
    }
    if ((Get-DirectoryIdentity $RunRoot 'white teacher run root at closure') -cne $RunRootIdentity -or (Get-DirectoryIdentity $RunParent 'white teacher output parent at closure') -cne $RunParentIdentity) { throw 'Teacher run/parent directory identity changed' }

    $pipelineReceiptPath = Join-Path $RunRoot 'pipeline.receipt.json'
    $pipelineReceipt = [ordered]@{
        schema_version=1; kind='otherimages_white_teacher_windows_pipeline_receipt_v1'; status='complete'
        source=[ordered]@{ repo_root=$RepoRoot; head=$RequiredHead; tree=$RequiredTree; tracked_and_untracked_clean=$true; fixed_code=$CodeBindings; wrapper=$WrapperBinding; python=$PythonBinding; git=$GitBinding; independent_verifier=$TeacherVerifierBinding }
        pilot=[ordered]@{ run_root=$PilotRunRoot; bindings=$PilotBindings; image_count=1000; suggested_splits=[ordered]@{train=912;val=52;test=36}; initial_teacher_states=[ordered]@{pending=999;quarantine=1}; portable_projection_sha256=$RequiredProjectionSha256 }
        execution=[ordered]@{ requested_device='cuda'; observed_device='gpu:0'; minimum_free_ram_bytes=[int64]17179869184; observed_free_ram_bytes=$FreeMemoryBytes; capture_processes=1; capture_view_id='all'; training_performed=$false }
        roots=[ordered]@{ run=$RunRoot; run_identity=$RunRootIdentity; parent=$RunParent; parent_identity=$RunParentIdentity; logs=$LogsRoot; capture=$CaptureOutput; teacher=$TeacherOutput }
        stages=[ordered]@{ capture=$captureStageReceipt; teacher=$teacherStageReceipt; independent_teacher_closure=$independentStageReceipt }
        counts=[ordered]@{ inventory=1000; pending=999; accepted=[int]$teacher.counts.accepted_teacher_records; quarantined=[int]$teacher.counts.quarantined_records; closure=1000 }
        validation=[ordered]@{ source_head_tree_clean_sha_size_stable=$true; git_executable_sha_size_stable=$true; pilot_receipt_inventory_projection_stable=$true; no_concurrent_exact8_or_other_capture_at_start=$true; free_ram_at_least_16_gib=$true; explicit_cuda_gpu0=$true; single_process_three_view_capture=$true; every_stage_rc_zero=$true; every_stage_stderr_zero_bytes=$true; capture_errors_zero=$true; teacher_sealed=$true; teacher_closure_independently_recomputed=$true; no_training=$true; quarantine_never_guess=$true; fresh_no_clobber_publications=$true; run_root_identity_stable=$true }
        started_utc=$PipelineStartedUtc.ToString('o'); completed_utc=[DateTime]::UtcNow.ToString('o')
    }
    Write-JsonNew $pipelineReceiptPath $pipelineReceipt
    $readback = Read-Json $pipelineReceiptPath
    if ([string]$readback.status -cne 'complete' -or $readback.validation.every_stage_rc_zero -ne $true -or [int]$readback.counts.closure -ne 1000) { throw 'Teacher pipeline receipt readback failed' }
    Write-Host "WHITE_TEACHER_PIPELINE_OK receipt=$pipelineReceiptPath accepted=$($teacher.counts.accepted_teacher_records) quarantined=$($teacher.counts.quarantined_records) device=gpu:0"
}
catch {
    $PipelineFailed = $_
    if ($RunRootOwned -and (Test-Path -LiteralPath $RunRoot -PathType Container)) {
        Assert-NoReparseChain $RunRoot 'white teacher owned failure root'
        if ((Get-DirectoryIdentity $RunRoot 'white teacher owned failure root') -cne $RunRootIdentity) { throw 'Refusing to write failure receipt after RunRoot identity changed' }
        $failurePath = Join-Path $RunRoot 'pipeline.failure.json'
        if (-not (Test-Path -LiteralPath $failurePath)) {
            Write-JsonNew $failurePath ([ordered]@{ schema_version=1; kind='otherimages_white_teacher_windows_failure_v1'; status='failed'; message=$_.Exception.Message; exception_type=$_.Exception.GetType().FullName; script_stack_trace=$_.ScriptStackTrace; utc=[DateTime]::UtcNow.ToString('o') })
        }
        Write-Host "WHITE_TEACHER_PIPELINE_FAILED evidence=$failurePath error=$($_.Exception.Message)"
    }
    throw
}
