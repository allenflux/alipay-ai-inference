<#
.SYNOPSIS
Runs the sealed 10,000-image white-input package through the formal local
receiver, the exact first-1,000 prefix materializer, and the read-only inventory.

.DESCRIPTION
Windows PowerShell 5.1 compatible.  The prepared ZIP is never modified.  A
loopback-only HTTP server exists only for the receiver stage.  Every publication
and evidence path is below one brand-new, caller-visible run root; no stage is
resumed or overwritten.  Native child stdout, stderr, and exact ASCII RC bytes
are persisted separately and a failure stops the pipeline immediately.
#>
[CmdletBinding()]
param(
    [string]$ArchivePath = 'C:\f3-white-sync\sample-10000-a\white-sample-10000-21054be8b1eb.zip',
    [string]$PrepareReceipt = 'C:\f3-white-sync\sample-10000-a\prepare.receipt.json',
    [string]$ExpectedArchiveSha256 = '6b0b93d8651ee6cebbcdf62e1200c0f8041f508cb9904e8e955c510be682481e',
    [ValidateRange(1, 32212254720)][int64]$ExpectedArchiveBytes = 2809303412,
    [string]$RunRoot = 'C:\f3-white-pilot-21054be8b1eb-a',
    [string]$PythonExe,
    [ValidateRange(1024, 65535)][int]$LoopbackPort = 18731
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$Utf8NoBom = New-Object Text.UTF8Encoding($false)
$Ascii = [Text.Encoding]::ASCII
$MaximumBytes = [int64]32212254720
$RequiredArchiveSha256 = '6b0b93d8651ee6cebbcdf62e1200c0f8041f508cb9904e8e955c510be682481e'
$RequiredArchiveBytes = [int64]2809303412
$RequiredPackageSubjectSha256 = '21054be8b1eb04f478c5b7e817cbe3367fef82e0b8d259b8025253df3f6af71e'
$RequiredPrefixManifestSha256 = '45fcca6b8b6b4fe97691a794e8aaf287026ee7a53ac9a4fb1b8be04ac6dc5938'
$RequiredProjectionSha256 = 'bd1b964117595a2e71b898d45f66393e5c15b92f863fbabd2f790499dbee009c'
$RequiredProjectionBytes = [int64]527892
$ReceiveVersion = 'white-10000-21054be8b1eb-local-a'
$PilotVersion = 'white-pilot-1000-21054be8b1eb-local-a'
$PipelineStartedUtc = [DateTime]::UtcNow
$PipelineFailed = $null
$ArchiveBinding = $null
$RunRootOwned = $false

$RepoRoot = [IO.Path]::GetFullPath((Split-Path -Parent $PSScriptRoot))
if ([string]::IsNullOrWhiteSpace($PythonExe)) {
    $PythonExe = Join-Path $RepoRoot '.venv-cu126\Scripts\python.exe'
}

function Write-TextNew([string]$Path, [string]$Text) {
    if (Test-Path -LiteralPath $Path) { throw "Refusing to overwrite evidence: $Path" }
    $parent = Split-Path -Parent $Path
    if (-not (Test-Path -LiteralPath $parent -PathType Container)) {
        [IO.Directory]::CreateDirectory($parent) | Out-Null
    }
    $stream = New-Object IO.FileStream($Path, [IO.FileMode]::CreateNew, [IO.FileAccess]::Write, [IO.FileShare]::None)
    try {
        $writer = New-Object IO.StreamWriter($stream, $Utf8NoBom)
        try { $writer.Write($Text); $writer.Flush() } finally { $writer.Dispose() }
    }
    finally { $stream.Dispose() }
}

function Write-JsonNew([string]$Path, [object]$Value) {
    Write-TextNew $Path (($Value | ConvertTo-Json -Depth 60) + "`r`n")
}

function Write-RcNew([string]$Path, [int]$Rc) {
    if (Test-Path -LiteralPath $Path) { throw "Refusing to overwrite RC evidence: $Path" }
    [IO.File]::WriteAllBytes($Path, $Ascii.GetBytes(([string]$Rc) + "`r`n"))
}

function Read-Json([string]$Path) {
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) { throw "Missing JSON evidence: $Path" }
    return (Get-Content -LiteralPath $Path -Raw -Encoding UTF8) | ConvertFrom-Json
}

function Get-Binding([string]$Path) {
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) { throw "Missing bound file: $Path" }
    $item = Get-Item -LiteralPath $Path -Force
    return [pscustomobject][ordered]@{
        path = $item.FullName
        size_bytes = [int64]$item.Length
        sha256 = (Get-FileHash -LiteralPath $item.FullName -Algorithm SHA256).Hash.ToLowerInvariant()
    }
}

function Assert-NoReparseChain([string]$Path, [string]$Description) {
    $current = [IO.Path]::GetFullPath($Path)
    while (-not [string]::IsNullOrWhiteSpace($current)) {
        $entry = Get-Item -LiteralPath $current -Force -ErrorAction SilentlyContinue
        if ($null -ne $entry -and (($entry.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0)) {
            throw "$Description traverses a reparse point: $current"
        }
        $next = Split-Path -Parent $current
        if ([string]::IsNullOrWhiteSpace($next) -or $next -eq $current) { break }
        $current = $next
    }
}

function Initialize-NativeDirectoryType {
    if ($null -ne ('WhitePilotNativeDirectoryV1' -as [type])) { return }
    $source = @'
using System;
using System.ComponentModel;
using System.Runtime.InteropServices;
using Microsoft.Win32.SafeHandles;

public static class WhitePilotNativeDirectoryV1 {
    [StructLayout(LayoutKind.Sequential)]
    private struct ByHandleFileInformation {
        public uint FileAttributes;
        public System.Runtime.InteropServices.ComTypes.FILETIME CreationTime;
        public System.Runtime.InteropServices.ComTypes.FILETIME LastAccessTime;
        public System.Runtime.InteropServices.ComTypes.FILETIME LastWriteTime;
        public uint VolumeSerialNumber;
        public uint FileSizeHigh;
        public uint FileSizeLow;
        public uint NumberOfLinks;
        public uint FileIndexHigh;
        public uint FileIndexLow;
    }

    [DllImport("kernel32.dll", CharSet = CharSet.Unicode, SetLastError = true, ExactSpelling = true)]
    [return: MarshalAs(UnmanagedType.Bool)]
    private static extern bool CreateDirectoryW(string path, IntPtr securityAttributes);

    [DllImport("kernel32.dll", CharSet = CharSet.Unicode, SetLastError = true, ExactSpelling = true)]
    private static extern SafeFileHandle CreateFileW(
        string path, uint desiredAccess, uint shareMode, IntPtr securityAttributes,
        uint creationDisposition, uint flagsAndAttributes, IntPtr templateFile);

    [DllImport("kernel32.dll", SetLastError = true)]
    [return: MarshalAs(UnmanagedType.Bool)]
    private static extern bool GetFileInformationByHandle(
        SafeFileHandle handle, out ByHandleFileInformation information);

    public static void CreateExclusive(string path) {
        if (!CreateDirectoryW(path, IntPtr.Zero)) {
            throw new Win32Exception(Marshal.GetLastWin32Error(),
                "CreateDirectoryW exclusive creation failed for " + path);
        }
    }

    public static string Identity(string path) {
        const uint ShareReadWriteDelete = 0x00000007;
        const uint OpenExisting = 3;
        const uint BackupSemantics = 0x02000000;
        using (SafeFileHandle handle = CreateFileW(
            path, 0, ShareReadWriteDelete, IntPtr.Zero,
            OpenExisting, BackupSemantics, IntPtr.Zero)) {
            if (handle.IsInvalid) {
                throw new Win32Exception(Marshal.GetLastWin32Error(),
                    "CreateFileW identity open failed for " + path);
            }
            ByHandleFileInformation information;
            if (!GetFileInformationByHandle(handle, out information)) {
                throw new Win32Exception(Marshal.GetLastWin32Error(),
                    "GetFileInformationByHandle failed for " + path);
            }
            return information.VolumeSerialNumber.ToString("x8") + ":" +
                information.FileIndexHigh.ToString("x8") + ":" +
                information.FileIndexLow.ToString("x8");
        }
    }
}
'@
    Add-Type -TypeDefinition $source -Language CSharp | Out-Null
}

function Get-DirectoryIdentity([string]$Path, [string]$Description) {
    $item = Get-Item -LiteralPath $Path -Force
    if (-not $item.PSIsContainer) { throw "$Description is not a directory: $Path" }
    if (($item.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0) {
        throw "$Description is a reparse point: $Path"
    }
    $expected = [IO.Path]::GetFullPath($Path)
    $observed = [IO.Path]::GetFullPath($item.FullName)
    if ($expected.Length -gt ([IO.Path]::GetPathRoot($expected)).Length) { $expected = $expected.TrimEnd('\') }
    if ($observed.Length -gt ([IO.Path]::GetPathRoot($observed)).Length) { $observed = $observed.TrimEnd('\') }
    if (-not $observed.Equals($expected, [StringComparison]::OrdinalIgnoreCase)) {
        throw "$Description resolved path differs: expected=$expected observed=$observed"
    }
    return [WhitePilotNativeDirectoryV1]::Identity($observed)
}

function New-ExclusiveDirectory([string]$Path) {
    [WhitePilotNativeDirectoryV1]::CreateExclusive([IO.Path]::GetFullPath($Path))
}

function Require-CPath([string]$Path, [string]$Description) {
    $full = [IO.Path]::GetFullPath($Path)
    if ([IO.Path]::GetPathRoot($full) -cne 'C:\') { throw "$Description must stay on C:\: $full" }
    return $full
}

function Get-ArchiveSha256([string]$Path, [int64]$ExpectedBytes) {
    $digest = [Security.Cryptography.SHA256]::Create()
    $stream = New-Object IO.FileStream($Path, [IO.FileMode]::Open, [IO.FileAccess]::Read, [IO.FileShare]::Read)
    $buffer = New-Object byte[] 8388608
    $observed = [int64]0
    $started = [DateTime]::UtcNow
    $lastReport = $started
    try {
        while (($read = $stream.Read($buffer, 0, $buffer.Length)) -gt 0) {
            [void]$digest.TransformBlock($buffer, 0, $read, $null, 0)
            $observed += $read
            $now = [DateTime]::UtcNow
            if (($now - $lastReport).TotalSeconds -ge 60) {
                Write-Host "WHITE_PILOT_ARCHIVE_HASH_ALIVE bytes=$observed/$ExpectedBytes elapsed_s=$([int]($now-$started).TotalSeconds)"
                $lastReport = $now
            }
        }
        [void]$digest.TransformFinalBlock((New-Object byte[] 0), 0, 0)
        if ($observed -ne $ExpectedBytes) { throw "Archive size changed while hashing: $observed/$ExpectedBytes" }
        return ([BitConverter]::ToString($digest.Hash)).Replace('-', '').ToLowerInvariant()
    }
    finally { $stream.Dispose(); $digest.Dispose() }
}

function ConvertTo-NativeCommandLine([string[]]$Arguments) {
    $builder = New-Object Text.StringBuilder
    foreach ($argument in $Arguments) {
        if ($argument.IndexOf('"') -ge 0 -or $argument.EndsWith('\')) {
            throw "Unsupported native argument spelling: $argument"
        }
        if ($builder.Length -gt 0) { [void]$builder.Append(' ') }
        [void]$builder.Append('"'); [void]$builder.Append($argument); [void]$builder.Append('"')
    }
    return $builder.ToString()
}

function New-PythonProcess([string[]]$Arguments, [bool]$Redirect) {
    $info = New-Object Diagnostics.ProcessStartInfo
    $info.FileName = $PythonExe
    $info.Arguments = ConvertTo-NativeCommandLine $Arguments
    $info.WorkingDirectory = $RepoRoot
    $info.UseShellExecute = $false
    $info.CreateNoWindow = $true
    $info.RedirectStandardOutput = $Redirect
    $info.RedirectStandardError = $Redirect
    $info.EnvironmentVariables['PYTHONIOENCODING'] = 'utf-8:strict'
    $info.EnvironmentVariables['PYTHONUTF8'] = '1'
    $process = New-Object Diagnostics.Process
    $process.StartInfo = $info
    return $process
}

function Invoke-PythonStage([string]$Name, [string[]]$Arguments) {
    $stageRoot = Join-Path $LogsRoot $Name
    if (Test-Path -LiteralPath $stageRoot) { throw "Stage log root already exists: $stageRoot" }
    [IO.Directory]::CreateDirectory($stageRoot) | Out-Null
    $stdoutPath = Join-Path $stageRoot 'stdout.txt'
    $stderrPath = Join-Path $stageRoot 'stderr.txt'
    $rcPath = Join-Path $stageRoot 'rc.txt'
    $receiptPath = Join-Path $stageRoot 'stage.receipt.json'
    $started = [DateTime]::UtcNow
    $process = New-PythonProcess $Arguments $true
    Write-Host "WHITE_PILOT_STAGE_START stage=$Name"
    if (-not $process.Start()) { throw "Unable to start Python stage: $Name" }
    $pidValue = $process.Id
    $stdoutTask = $process.StandardOutput.ReadToEndAsync()
    $stderrTask = $process.StandardError.ReadToEndAsync()
    $lastReport = $started
    while (-not $process.WaitForExit(1000)) {
        $now = [DateTime]::UtcNow
        if (($now - $lastReport).TotalSeconds -ge 60) {
            try {
                $process.Refresh()
                Write-Host ("WHITE_PILOT_STAGE_ALIVE stage={0} elapsed_s={1} pid={2} cpu_s={3:N1} ws_bytes={4}" -f $Name,[int]($now-$started).TotalSeconds,$pidValue,$process.TotalProcessorTime.TotalSeconds,$process.WorkingSet64)
            }
            catch { Write-Host "WHITE_PILOT_STAGE_ALIVE stage=$Name elapsed_s=$([int]($now-$started).TotalSeconds) pid=$pidValue" }
            $lastReport = $now
        }
    }
    $process.WaitForExit()
    $stdout = [string]$stdoutTask.Result
    $stderr = [string]$stderrTask.Result
    [IO.File]::WriteAllText($stdoutPath, $stdout, $Utf8NoBom)
    [IO.File]::WriteAllText($stderrPath, $stderr, $Utf8NoBom)
    Write-RcNew $rcPath $process.ExitCode
    $result = [pscustomobject][ordered]@{
        name = $Name
        pid = $pidValue
        rc = [int]$process.ExitCode
        started_utc = $started.ToString('o')
        completed_utc = [DateTime]::UtcNow.ToString('o')
        elapsed_seconds = ([DateTime]::UtcNow - $started).TotalSeconds
        stdout = Get-Binding $stdoutPath
        stderr = Get-Binding $stderrPath
        rc_file = Get-Binding $rcPath
        receipt_path = $receiptPath
    }
    if ($result.rc -ne 0) { throw "Python stage failed: stage=$Name rc=$($result.rc) evidence=$stageRoot" }
    if ($result.stderr.size_bytes -ne 0) { throw "Python stage emitted stderr: stage=$Name evidence=$stderrPath" }
    Write-Host "WHITE_PILOT_STAGE_EXIT stage=$Name rc=0 elapsed_s=$([int]$result.elapsed_seconds)"
    return $result
}

function Complete-Stage([object]$Stage, [object]$Validation) {
    $receipt = [ordered]@{
        schema_version = 1
        kind = 'otherimages_white_pilot_windows_stage_receipt_v1'
        status = 'complete'
        stage = [string]$Stage.name
        process = [ordered]@{ pid=[int]$Stage.pid; rc=[int]$Stage.rc; started_utc=$Stage.started_utc; completed_utc=$Stage.completed_utc; elapsed_seconds=$Stage.elapsed_seconds }
        stdout = $Stage.stdout
        stderr = $Stage.stderr
        rc_file = $Stage.rc_file
        validation = $Validation
    }
    Write-JsonNew $Stage.receipt_path $receipt
    return Get-Binding $Stage.receipt_path
}

function Get-NonemptyLines([string]$Text) {
    return @($Text -split "`r?`n" | Where-Object { -not [string]::IsNullOrWhiteSpace($_) })
}

function ConvertTo-PythonJsonString([AllowNull()][object]$Value) {
    if ($null -eq $Value) { return 'null' }
    $text = [string]$Value
    $builder = New-Object Text.StringBuilder
    [void]$builder.Append('"')
    for ($index = 0; $index -lt $text.Length; $index++) {
        $code = [int][char]$text[$index]
        if ($code -eq 8) { [void]$builder.Append('\b'); continue }
        if ($code -eq 9) { [void]$builder.Append('\t'); continue }
        if ($code -eq 10) { [void]$builder.Append('\n'); continue }
        if ($code -eq 12) { [void]$builder.Append('\f'); continue }
        if ($code -eq 13) { [void]$builder.Append('\r'); continue }
        if ($code -eq 34) { [void]$builder.Append('\"'); continue }
        if ($code -eq 92) { [void]$builder.Append('\\'); continue }
        if ($code -lt 32) {
            [void]$builder.Append('\u')
            [void]$builder.Append($code.ToString('x4',[Globalization.CultureInfo]::InvariantCulture))
            continue
        }
        if ($code -ge 0xD800 -and $code -le 0xDBFF) {
            if (($index + 1) -ge $text.Length) { throw 'Portable projection contains an unpaired high surrogate' }
            $nextCode = [int][char]$text[$index + 1]
            if ($nextCode -lt 0xDC00 -or $nextCode -gt 0xDFFF) { throw 'Portable projection contains an unpaired high surrogate' }
            [void]$builder.Append($text[$index])
            $index++
            [void]$builder.Append($text[$index])
            continue
        }
        if ($code -ge 0xDC00 -and $code -le 0xDFFF) { throw 'Portable projection contains an unpaired low surrogate' }
        [void]$builder.Append($text[$index])
    }
    [void]$builder.Append('"')
    return $builder.ToString()
}

function New-PortableProjection([string]$PendingPath, [string]$OutputPath) {
    if (-not (Test-Path -LiteralPath $PendingPath -PathType Leaf)) { throw "Missing teacher-pending manifest: $PendingPath" }
    if (Test-Path -LiteralPath $OutputPath) { throw "Refusing to overwrite portable projection: $OutputPath" }
    $fieldNames = @(
        'decoded_pixel_sha256','group_id','phash64','quarantine_reason','raw_sha256',
        'record_id','source_relative_path','suggested_split','teacher_state'
    )
    $rows = New-Object Collections.Generic.List[object]
    foreach ($line in [IO.File]::ReadLines($PendingPath, $Utf8NoBom)) {
        if ([string]::IsNullOrWhiteSpace($line)) { throw 'Teacher-pending manifest contains a blank line' }
        $row = $line | ConvertFrom-Json
        foreach ($fieldName in $fieldNames) {
            if ($null -eq $row.PSObject.Properties[$fieldName]) { throw "Teacher-pending projection field is absent: $fieldName" }
        }
        if ([string]$row.record_id -notmatch '^[0-9a-f]{64}$' -or
            [string]$row.raw_sha256 -notmatch '^[0-9a-f]{64}$' -or
            [string]$row.decoded_pixel_sha256 -notmatch '^[0-9a-f]{64}$' -or
            [string]$row.phash64 -notmatch '^[0-9a-f]{16}$' -or
            [string]$row.group_id -notmatch '^leakage-group:[0-9a-f]{64}$') {
            throw 'Teacher-pending projection identity/hash spelling is invalid'
        }
        if (@('train','val','test') -cnotcontains [string]$row.suggested_split) { throw 'Teacher-pending projection split is invalid' }
        if (@('pending','quarantine') -cnotcontains [string]$row.teacher_state) { throw 'Teacher-pending projection state is invalid' }
        $rows.Add($row)
    }
    [object[]]$sortedRows = @($rows | Sort-Object { [string]$_.record_id })
    if ($sortedRows.Count -ne 1000) { throw "Portable projection row count differs from 1000: $($sortedRows.Count)" }
    for ($index=1; $index -lt $sortedRows.Count; $index++) {
        if ([string]::CompareOrdinal([string]$sortedRows[$index-1].record_id,[string]$sortedRows[$index].record_id) -ge 0) {
            throw 'Portable projection record_id order is not strictly unique'
        }
    }
    $stream = New-Object IO.FileStream($OutputPath, [IO.FileMode]::CreateNew, [IO.FileAccess]::Write, [IO.FileShare]::None)
    try {
        $writer = New-Object IO.StreamWriter($stream, $Utf8NoBom)
        try {
            foreach ($row in $sortedRows) {
                $writer.Write('{')
                for ($fieldIndex=0; $fieldIndex -lt $fieldNames.Count; $fieldIndex++) {
                    if ($fieldIndex -gt 0) { $writer.Write(',') }
                    $fieldName = $fieldNames[$fieldIndex]
                    $writer.Write('"' + $fieldName + '":')
                    $writer.Write((ConvertTo-PythonJsonString $row.$fieldName))
                }
                $writer.Write("}`n")
            }
            $writer.Flush()
        }
        finally { $writer.Dispose() }
    }
    finally { $stream.Dispose() }
    return [pscustomobject][ordered]@{ rows=$sortedRows.Count; file=Get-Binding $OutputPath }
}

function Test-PortOpen([int]$Port) {
    $client = New-Object Net.Sockets.TcpClient
    try {
        $async = $client.BeginConnect('127.0.0.1', $Port, $null, $null)
        if (-not $async.AsyncWaitHandle.WaitOne(250)) { return $false }
        $client.EndConnect($async)
        return $true
    }
    catch { return $false }
    finally { $client.Dispose() }
}

function Start-LoopbackServer([string]$Archive) {
    if (Test-PortOpen $LoopbackPort) { throw "Loopback port is already occupied: $LoopbackPort" }
    $serverRoot = Join-Path $LogsRoot 'loopback-server'
    [IO.Directory]::CreateDirectory($serverRoot) | Out-Null
    $arguments = @('-u','-m','http.server',([string]$LoopbackPort),'--bind','127.0.0.1','--directory',(Split-Path -Parent $Archive))
    $process = New-PythonProcess $arguments $true
    $started = $false
    $stdoutTask = $null
    $stderrTask = $null
    try {
        if (-not $process.Start()) { throw 'Unable to start loopback archive server' }
        $started = $true
        $stdoutTask = $process.StandardOutput.ReadToEndAsync()
        $stderrTask = $process.StandardError.ReadToEndAsync()
        $deadline = [DateTime]::UtcNow.AddSeconds(30)
        while (-not (Test-PortOpen $LoopbackPort)) {
            if ($process.HasExited) { throw "Loopback server exited during startup: rc=$($process.ExitCode)" }
            if ([DateTime]::UtcNow -ge $deadline) { throw 'Loopback server did not become ready in 30 seconds' }
            Start-Sleep -Milliseconds 100
        }
        Write-Host "WHITE_PILOT_LOOPBACK_READY pid=$($process.Id) address=127.0.0.1:$LoopbackPort"
        return [pscustomobject]@{ process=$process; stdout_task=$stdoutTask; stderr_task=$stderrTask; root=$serverRoot; archive=(Split-Path -Leaf $Archive) }
    }
    catch {
        $startupError = $_
        $cleanupError = $null
        if ($started) {
            try {
                if (-not $process.HasExited) { $process.Kill() }
            }
            catch { $cleanupError = $_ }
            try {
                if (-not $process.WaitForExit(30000)) {
                    try { $process.Kill() } catch { if ($null -eq $cleanupError) { $cleanupError = $_ } }
                    if (-not $process.WaitForExit(30000)) { throw 'Loopback server remained alive after two kill/wait attempts' }
                }
            }
            catch { if ($null -eq $cleanupError) { $cleanupError = $_ } }
            if ($process.HasExited) {
                try {
                    if ($null -ne $stdoutTask) { [IO.File]::WriteAllText((Join-Path $serverRoot 'startup.stdout.txt'),[string]$stdoutTask.Result,$Utf8NoBom) }
                    if ($null -ne $stderrTask) { [IO.File]::WriteAllText((Join-Path $serverRoot 'startup.stderr.txt'),[string]$stderrTask.Result,$Utf8NoBom) }
                    Write-RcNew (Join-Path $serverRoot 'startup.rc.txt') ([int]$process.ExitCode)
                }
                catch { if ($null -eq $cleanupError) { $cleanupError = $_ } }
            }
        }
        $process.Dispose()
        if ($null -ne $cleanupError) {
            throw "Loopback startup failed and cleanup was not clean: startup=$($startupError.Exception.Message) cleanup=$($cleanupError.Exception.Message)"
        }
        throw $startupError
    }
}

function Stop-LoopbackServer([object]$Server) {
    $process = $Server.process
    $forced = $false
    try {
        if (-not $process.HasExited) {
            try { $process.Kill(); $forced = $true }
            catch {
                $firstKillError = $_
                $process.Refresh()
                if (-not $process.HasExited) {
                    Write-Host "WHITE_PILOT_LOOPBACK_FIRST_KILL_ERROR error=$($firstKillError.Exception.Message)"
                }
            }
            if (-not $process.WaitForExit(30000)) {
                try { $process.Kill(); $forced = $true }
                catch {
                    $secondKillError = $_
                    $process.Refresh()
                    if (-not $process.HasExited) { throw "Loopback second kill failed: $($secondKillError.Exception.Message)" }
                }
                if (-not $process.WaitForExit(30000)) { throw 'Loopback server remained alive after two kill/wait attempts' }
            }
        }
        else { $process.WaitForExit() }
        if (-not $process.HasExited) { throw 'Loopback server cleanup did not prove process exit' }
        $pidValue = [int]$process.Id
        $exitCode = [int]$process.ExitCode
        $stdoutPath = Join-Path $Server.root 'stdout.txt'
        $stderrPath = Join-Path $Server.root 'stderr.txt'
        [IO.File]::WriteAllText($stdoutPath, [string]$Server.stdout_task.Result, $Utf8NoBom)
        [IO.File]::WriteAllText($stderrPath, [string]$Server.stderr_task.Result, $Utf8NoBom)
        $stdoutBinding = Get-Binding $stdoutPath
        $stderrBinding = Get-Binding $stderrPath
        $serverStdout = Get-Content -LiteralPath $stdoutPath -Raw -Encoding UTF8
        [string[]]$serverStdoutLines = @($serverStdout -split "`r?`n" | Where-Object { -not [string]::IsNullOrWhiteSpace($_) })
        if ($serverStdoutLines.Count -ne 1 -or $serverStdoutLines[0] -notmatch ('^Serving HTTP on 127\.0\.0\.1 port ' + $LoopbackPort + ' ')) {
            throw "Loopback server stdout is not its one expected startup marker: $stdoutPath"
        }
        $serverStderr = Get-Content -LiteralPath $stderrPath -Raw -Encoding UTF8
        if ($serverStderr -notmatch '127\.0\.0\.1' -or $serverStderr -notmatch '"GET /' -or $serverStderr -notmatch ' 200 ') {
            throw "Loopback server did not retain one successful local GET access log: $stderrPath"
        }
        if (-not $forced) { throw 'Loopback server exited unexpectedly instead of remaining alive until receiver completion' }
        return [pscustomobject][ordered]@{
            pid = $pidValue
            forced_exit_code = $exitCode
            bound_address = "127.0.0.1:$LoopbackPort"
            intentionally_stopped_after_receive = $forced
            stdout = $stdoutBinding
            stderr = $stderrBinding
            startup_marker_valid = $true
            successful_get_200_observed = $true
        }
    }
    finally { $process.Dispose() }
}

try {
    $PrepareReceipt = Require-CPath $PrepareReceipt 'prepare receipt'
    if (-not (Test-Path -LiteralPath $PrepareReceipt -PathType Leaf)) { throw "Missing prepare receipt: $PrepareReceipt" }
    $prepare = Read-Json $PrepareReceipt
    if ([string]$prepare.status -cne 'complete') { throw 'Prepare receipt is not complete' }
    $receiptArchivePath = [IO.Path]::GetFullPath([string]$prepare.archive.path)
    if ([string]::IsNullOrWhiteSpace($ArchivePath)) { $ArchivePath = $receiptArchivePath }
    $ArchivePath = Require-CPath $ArchivePath 'sealed archive'
    $RunRoot = Require-CPath $RunRoot 'pilot run root'
    $PythonExe = [IO.Path]::GetFullPath($PythonExe)
    $receiverScript = Join-Path $RepoRoot 'scripts\otherimages-white-sample-receive.py'
    $prefixScript = Join-Path $RepoRoot 'scripts\otherimages-white-prefix-materialize.py'
    $inventoryScript = Join-Path $RepoRoot 'scripts\otherimages-inventory.py'
    $inventoryModule = Join-Path $RepoRoot 'src\transfer_receipt_ai\otherimages_inventory.py'
    foreach ($required in @($ArchivePath,$PrepareReceipt,$PythonExe,$receiverScript,$prefixScript,$inventoryScript,$inventoryModule)) {
        if (-not (Test-Path -LiteralPath $required -PathType Leaf)) { throw "Missing required file: $required" }
        Assert-NoReparseChain $required 'white pilot authority file'
    }
    if (Test-Path -LiteralPath $RunRoot) { throw "RunRoot must be brand-new: $RunRoot" }
    $RunParent = Split-Path -Parent $RunRoot
    Assert-NoReparseChain $RunParent 'white pilot output parent'
    Initialize-NativeDirectoryType
    $RunParentIdentity = Get-DirectoryIdentity $RunParent 'white pilot output parent'
    if ($ArchivePath.StartsWith($RunRoot + '\',[StringComparison]::OrdinalIgnoreCase)) { throw 'Archive must stay outside RunRoot' }

    if (-not $receiptArchivePath.Equals($ArchivePath,[StringComparison]::OrdinalIgnoreCase)) { throw 'Explicit archive path differs from prepare receipt' }
    $receiptSha = ([string]$prepare.archive.sha256).ToLowerInvariant()
    $receiptBytes = [int64]$prepare.archive.size_bytes
    if ($receiptSha -notmatch '^[0-9a-f]{64}$' -or $receiptBytes -le 0 -or $receiptBytes -gt $MaximumBytes) { throw 'Prepare receipt archive binding is invalid' }
    if ([string]::IsNullOrWhiteSpace($ExpectedArchiveSha256) -or $ExpectedArchiveSha256.ToLowerInvariant() -cne $RequiredArchiveSha256) { throw 'Explicit expected archive SHA must equal the frozen archive SHA' }
    if ($ExpectedArchiveBytes -ne $RequiredArchiveBytes) { throw 'Explicit expected archive bytes must equal the frozen archive size' }
    if ($receiptSha -cne $RequiredArchiveSha256 -or $receiptBytes -ne $RequiredArchiveBytes) { throw 'Prepare receipt differs from the frozen archive SHA/size' }
    $archiveInfo = Get-Item -LiteralPath $ArchivePath -Force
    if ($archiveInfo.Length -ne $receiptBytes) { throw 'Archive size differs from prepare receipt before hashing' }
    New-ExclusiveDirectory $RunRoot
    Assert-NoReparseChain $RunRoot 'white pilot owned run root'
    if ((Get-DirectoryIdentity $RunParent 'white pilot output parent after creation') -cne $RunParentIdentity) { throw 'White pilot output parent identity changed during exclusive creation' }
    $RunRootIdentity = Get-DirectoryIdentity $RunRoot 'white pilot owned run root'
    $RunRootOwned = $true
    $LogsRoot = Join-Path $RunRoot 'logs'
    [IO.Directory]::CreateDirectory($LogsRoot) | Out-Null
    Write-Host "WHITE_PILOT_ARCHIVE_HASH_START bytes=$receiptBytes path=$ArchivePath"
    $observedArchiveSha = Get-ArchiveSha256 $ArchivePath $receiptBytes
    if ($observedArchiveSha -cne $receiptSha) { throw 'Archive SHA256 differs from prepare receipt' }
    $ArchiveBinding = [pscustomobject][ordered]@{ path=$ArchivePath; size_bytes=$receiptBytes; sha256=$receiptSha; prepare_receipt=(Get-Binding $PrepareReceipt) }
    Write-Host "WHITE_PILOT_ARCHIVE_HASH_OK bytes=$receiptBytes sha256=$receiptSha"

    $IncomingRoot = Join-Path $RunRoot 'incoming'
    $RawRoot = Join-Path $RunRoot 'raw'
    $EvidenceRoot = Join-Path $RunRoot 'evidence'
    $PilotRoot = Join-Path $RunRoot 'pilot'
    $InventoryRoot = Join-Path $RunRoot 'inventory-prefix1000'
    $RawVersionRoot = Join-Path $RawRoot $ReceiveVersion
    $ReceiveReceipt = Join-Path $EvidenceRoot ($ReceiveVersion + '.receive.receipt.json')
    $PilotVersionRoot = Join-Path $PilotRoot $PilotVersion
    $PilotReceipt = Join-Path $EvidenceRoot ($PilotVersion + '.pilot.receipt.json')

    $codeBindings = [ordered]@{
        wrapper = Get-Binding $PSCommandPath
        receiver = Get-Binding $receiverScript
        prefix_materializer = Get-Binding $prefixScript
        inventory_wrapper = Get-Binding $inventoryScript
        inventory_module = Get-Binding $inventoryModule
        python = Get-Binding $PythonExe
    }

    $server = $null
    $serverEvidence = $null
    try {
        $server = Start-LoopbackServer $ArchivePath
        $archiveUrl = 'http://127.0.0.1:' + $LoopbackPort + '/' + [Uri]::EscapeDataString((Split-Path -Leaf $ArchivePath))
        $receiveArguments = @(
            $receiverScript,
            '--url',$archiveUrl,
            '--expected-archive-sha256',$receiptSha,
            '--expected-archive-bytes',([string]$receiptBytes),
            '--incoming-root',$IncomingRoot,
            '--raw-root',$RawRoot,
            '--evidence-root',$EvidenceRoot,
            '--version',$ReceiveVersion,
            '--timeout-seconds','3600',
            '--max-archive-bytes',([string]$MaximumBytes),
            '--max-uncompressed-bytes',([string]$MaximumBytes)
        )
        $receiveStage = Invoke-PythonStage 'receive' $receiveArguments
    }
    finally {
        if ($null -ne $server) { $serverEvidence = Stop-LoopbackServer $server }
    }

    [string[]]$receiveLines = @(Get-NonemptyLines ([IO.File]::ReadAllText($receiveStage.stdout.path,$Utf8NoBom)))
    if ($receiveLines.Count -ne 2 -or $receiveLines[1] -notmatch '^WHITE_RECEIVE_OK ') { throw 'Receiver stdout is not one JSON object plus one success marker' }
    $receiveStdout = $receiveLines[0] | ConvertFrom-Json
    $receivePersisted = Read-Json $ReceiveReceipt
    foreach ($value in @($receiveStdout,$receivePersisted)) {
        if ([string]$value.kind -cne 'otherimages_white_sync_receive_receipt_v1' -or [string]$value.status -cne 'complete') { throw 'Receiver receipt kind/status failed' }
        if ([int]$value.verified_payload.image_count -ne 10000 -or $value.verified_payload.every_file_size_and_sha256_verified -ne $true -or $value.verified_payload.archive_file_closure_exact -ne $true) { throw 'Receiver payload validation failed' }
        if ([string]$value.download.sha256 -cne $receiptSha -or [int64]$value.download.size_bytes -ne $receiptBytes) { throw 'Receiver transport binding failed' }
        if ([string]$value.package_subject_sha256 -cne $RequiredPackageSubjectSha256) { throw 'Receiver package subject differs from the frozen package subject' }
        if ([string]$value.publication.version -cne $ReceiveVersion -or $value.publication.brand_new -ne $true -or $value.publication.atomic_rename -ne $true) { throw 'Receiver publication claims failed' }
    }
    if ([string]$receiveStdout.package_subject_sha256 -cne [string]$receivePersisted.package_subject_sha256) { throw 'Receiver stdout/persisted subject differs' }
    $receiveStageReceipt = Complete-Stage $receiveStage ([ordered]@{ formal_receipt=Get-Binding $ReceiveReceipt; stdout_and_persisted_subject_equal=$true; image_count=10000; transport_sha_bytes_bound=$true; raw_publication_brand_new_atomic=$true; loopback_server=$serverEvidence })

    $prefixArguments = @(
        $prefixScript,
        '--source-root',$RawVersionRoot,
        '--source-receipt',$ReceiveReceipt,
        '--output-root',$PilotRoot,
        '--evidence-root',$EvidenceRoot,
        '--version',$PilotVersion
    )
    $prefixStage = Invoke-PythonStage 'prefix1000' $prefixArguments
    [string[]]$prefixLines = @(Get-NonemptyLines ([IO.File]::ReadAllText($prefixStage.stdout.path,$Utf8NoBom)))
    if ($prefixLines.Count -lt 3) { throw 'Prefix stdout lacks progress, success marker, or JSON receipt' }
    for ($lineIndex=0; $lineIndex -lt ($prefixLines.Count-1); $lineIndex++) {
        if ($prefixLines[$lineIndex] -notmatch '^WHITE_PREFIX_VERIFY_ALIVE ' -and $prefixLines[$lineIndex] -notmatch '^WHITE_PREFIX_MATERIALIZE_OK ') { throw "Unexpected prefix stdout line: $($prefixLines[$lineIndex])" }
    }
    if (@($prefixLines | Where-Object { $_ -match '^WHITE_PREFIX_MATERIALIZE_OK ' }).Count -ne 1) { throw 'Prefix stdout must contain exactly one success marker' }
    $prefixStdout = $prefixLines[$prefixLines.Count-1] | ConvertFrom-Json
    $prefixPersisted = Read-Json $PilotReceipt
    foreach ($value in @($prefixStdout,$prefixPersisted)) {
        if ([string]$value.kind -cne 'otherimages_white_sync_prefix_pilot_receipt_v1' -or [string]$value.status -cne 'complete') { throw 'Prefix receipt kind/status failed' }
        if ([int]$value.source.full_manifest.image_count -ne 10000 -or [int]$value.prefix.image_count -ne 1000 -or $value.prefix.manifest.exact_byte_prefix_of_source_manifest -ne $true) { throw 'Prefix count/policy validation failed' }
        if ([string]$value.source.package_subject_sha256 -cne $RequiredPackageSubjectSha256 -or [string]$value.prefix.manifest.sha256 -cne $RequiredPrefixManifestSha256) { throw 'Prefix source subject or frozen prefix manifest SHA failed' }
        if ($value.validation.receive_receipt_contract_manifest_strict -ne $true -or $value.validation.every_source_file_size_and_sha256_revalidated -ne $true -or $value.validation.every_prefix_copy_size_and_sha256_verified -ne $true -or $value.validation.source_files_written -ne $false -or $value.validation.output_file_and_directory_closure_exact -ne $true) { throw 'Prefix integrity validation failed' }
        if ($value.publication.brand_new -ne $true -or $value.publication.atomic_exclusive_rename -ne $true -or $value.publication.overwrite_performed -ne $false) { throw 'Prefix publication claims failed' }
    }
    $internalPilotReceipt = Join-Path $PilotVersionRoot 'pilot.receipt.json'
    if ((Get-FileHash -LiteralPath $internalPilotReceipt -Algorithm SHA256).Hash -cne (Get-FileHash -LiteralPath $PilotReceipt -Algorithm SHA256).Hash) { throw 'Internal/external prefix receipts are not byte-identical' }
    if ([string]$prefixStdout.pilot_subject_sha256 -cne [string]$prefixPersisted.pilot_subject_sha256) { throw 'Prefix stdout/persisted subject differs' }
    $prefixStageReceipt = Complete-Stage $prefixStage ([ordered]@{ formal_receipt=Get-Binding $PilotReceipt; internal_receipt=Get-Binding $internalPilotReceipt; internal_external_receipts_byte_identical=$true; source_count=10000; prefix_count=1000; source_revalidated=$true; prefix_publication_brand_new_atomic=$true })

    $inventoryArguments = @(
        $inventoryScript,
        '--input',(Join-Path $PilotVersionRoot 'images'),
        '--output',$InventoryRoot,
        '--layout-sample-size','64',
        '--validation-ratio','0.10',
        '--test-ratio','0.10',
        '--split-seed','otherimages-split-v1',
        '--layout-sample-seed','otherimages-layout-sample-v1',
        '--max-phash-candidates','500000'
    )
    $inventoryStage = Invoke-PythonStage 'inventory' $inventoryArguments
    [string[]]$inventoryLines = @(Get-NonemptyLines ([IO.File]::ReadAllText($inventoryStage.stdout.path,$Utf8NoBom)))
    if ($inventoryLines.Count -ne 1 -or $inventoryLines[0] -notmatch '^Inventoried 1000 image\(s\); quarantined 0 decode error\(s\); ') { throw 'Inventory stdout summary failed strict count/error gate' }
    $inventoryContractPath = Join-Path $InventoryRoot 'inventory.contract.json'
    $inventory = Read-Json $inventoryContractPath
    if ([string]$inventory.kind -cne 'otherimages_read_only_inventory_v1' -or [int]$inventory.counts.images -ne 1000 -or [int]$inventory.counts.image_errors_quarantined -ne 0 -or [int]$inventory.counts.ignored_non_images -ne 0) { throw 'Inventory contract count gate failed' }
    if ([int]$inventory.counts.suggested_splits.train -ne 912 -or [int]$inventory.counts.suggested_splits.val -ne 52 -or [int]$inventory.counts.suggested_splits.test -ne 36) { throw 'Inventory frozen split counts failed' }
    if ([int]$inventory.counts.teacher_states.pending -ne 999 -or [int]$inventory.counts.teacher_states.quarantine -ne 1) { throw 'Inventory frozen teacher-state counts failed' }
    if ([int]$inventory.configuration.phash_candidate_cap -ne 500000 -or [int]$inventory.phash_candidates.candidate_evidence_rows -ne 6055 -or [int]$inventory.phash_candidates.represented_record_pairs -ne 6055 -or $inventory.phash_candidates.truncated -ne $false) { throw 'Inventory frozen pHash candidate evidence failed' }
    if ($inventory.source.source_membership_rechecked -ne $true -or $inventory.source.image_source_raw_sha256_rechecked -ne $true -or $inventory.source.source_mutation_detected -ne $false -or $inventory.output.source_images_copied -ne $false) { throw 'Inventory read-only/source closure gate failed' }
    if ($inventory.paddle_teacher_contract.inventory_contains_labels -ne $false -or $inventory.paddle_teacher_contract.inventory_performed_ocr -ne $false -or $inventory.paddle_teacher_contract.inventory_performed_training -ne $false -or $inventory.paddle_teacher_contract.guessed_or_synthetic_labels_forbidden -ne $true -or $inventory.paddle_teacher_contract.training_eligibility_before_teacher_validation -ne $false) { throw 'Inventory teacher-pending policy gate failed' }
    $expectedInventoryInput = [IO.Path]::GetFullPath((Join-Path $PilotVersionRoot 'images'))
    if (-not ([IO.Path]::GetFullPath([string]$inventory.source.input_directory)).Equals($expectedInventoryInput,[StringComparison]::OrdinalIgnoreCase)) { throw 'Inventory contract input root differs from the exact prefix images root' }
    [string[]]$expectedInventoryFiles = @('errors.jsonl','exact_duplicates.jsonl','ignored_non_images.jsonl','images.jsonl','inventory.contract.json','layout_sample.jsonl','near_duplicate_candidates.jsonl','paddle_teacher_pending.jsonl')
    [string[]]$actualInventoryFiles = @(Get-ChildItem -LiteralPath $InventoryRoot -File -Force | ForEach-Object { $_.Name } | Sort-Object)
    if (($actualInventoryFiles -join '|') -cne ($expectedInventoryFiles -join '|') -or @(Get-ChildItem -LiteralPath $InventoryRoot -Directory -Force).Count -ne 0) { throw 'Inventory output tree is not the exact eight-file closure' }
    $projectionPath = Join-Path $LogsRoot 'inventory-portable-projection.jsonl'
    $projection = New-PortableProjection (Join-Path $InventoryRoot 'paddle_teacher_pending.jsonl') $projectionPath
    if ([int]$projection.rows -ne 1000 -or [int64]$projection.file.size_bytes -ne $RequiredProjectionBytes -or [string]$projection.file.sha256 -cne $RequiredProjectionSha256) { throw 'Inventory portable projection SHA/size/count failed' }
    $inventoryStageReceipt = Complete-Stage $inventoryStage ([ordered]@{ contract=Get-Binding $inventoryContractPath; images=1000; decode_errors=0; ignored_non_images=0; suggested_splits=[ordered]@{train=912;val=52;test=36}; teacher_states=[ordered]@{pending=999;quarantine=1}; near_duplicate_candidates=6055; portable_projection=$projection; source_read_only_rechecked=$true; teacher_pending_unlabeled=$true; exact_eight_file_tree=$true })

    foreach ($name in $codeBindings.Keys) {
        $before = $codeBindings[$name]
        $after = Get-Binding ([string]$before.path)
        if ([string]$before.sha256 -cne [string]$after.sha256 -or [int64]$before.size_bytes -ne [int64]$after.size_bytes) { throw "Code authority changed during pilot pipeline: $name" }
    }
    Assert-NoReparseChain $RunRoot 'white pilot owned run root at closure'
    if ((Get-DirectoryIdentity $RunRoot 'white pilot owned run root at closure') -cne $RunRootIdentity) { throw 'White pilot RunRoot identity changed during pipeline' }
    if ((Get-DirectoryIdentity $RunParent 'white pilot output parent at closure') -cne $RunParentIdentity) { throw 'White pilot output parent identity changed during pipeline' }
    $prepareAfter = Get-Binding $PrepareReceipt
    if ([string]$prepareAfter.sha256 -cne [string]$ArchiveBinding.prepare_receipt.sha256 -or [int64]$prepareAfter.size_bytes -ne [int64]$ArchiveBinding.prepare_receipt.size_bytes) { throw 'Prepare receipt changed during pilot pipeline' }
    $prepareReadback = Read-Json $PrepareReceipt
    if ([string]$prepareReadback.status -cne 'complete' -or
        -not ([IO.Path]::GetFullPath([string]$prepareReadback.archive.path)).Equals($ArchivePath,[StringComparison]::OrdinalIgnoreCase) -or
        ([string]$prepareReadback.archive.sha256).ToLowerInvariant() -cne $RequiredArchiveSha256 -or
        [int64]$prepareReadback.archive.size_bytes -ne $RequiredArchiveBytes) { throw 'Prepare receipt semantic binding changed during pilot pipeline' }
    if ((Get-ArchiveSha256 $ArchivePath $receiptBytes) -cne $receiptSha) { throw 'Archive changed after pilot pipeline' }

    $pipelineReceiptPath = Join-Path $RunRoot 'pipeline.receipt.json'
    $pipelineReceipt = [ordered]@{
        schema_version = 1
        kind = 'otherimages_white_pilot_windows_pipeline_receipt_v1'
        status = 'complete'
        analysis_only = $true
        production_route_authorized = $false
        training_performed = $false
        ocr_performed = $false
        source_archive_modified = $false
        archive = $ArchiveBinding
        code = $codeBindings
        roots = [ordered]@{ run=$RunRoot; run_identity=$RunRootIdentity; parent=$RunParent; parent_identity=$RunParentIdentity; raw=$RawVersionRoot; pilot=$PilotVersionRoot; inventory=$InventoryRoot; evidence=$EvidenceRoot; logs=$LogsRoot }
        stages = [ordered]@{ receive=$receiveStageReceipt; prefix1000=$prefixStageReceipt; inventory=$inventoryStageReceipt }
        validation = [ordered]@{ receiver_image_count=10000; prefix_image_count=1000; inventory_image_count=1000; inventory_decode_errors=0; suggested_splits=[ordered]@{train=912;val=52;test=36}; teacher_states=[ordered]@{pending=999;quarantine=1}; near_duplicate_candidates=6055; portable_projection_sha256=$RequiredProjectionSha256; portable_projection_size_bytes=$RequiredProjectionBytes; localhost_formal_receiver=$true; every_stage_rc_zero=$true; every_stage_stderr_zero_bytes=$true; core_code_archive_prepare_stable=$true; run_root_exclusive_and_identity_stable=$true; no_clobber_publications=$true }
        started_utc = $PipelineStartedUtc.ToString('o')
        completed_utc = [DateTime]::UtcNow.ToString('o')
    }
    Write-JsonNew $pipelineReceiptPath $pipelineReceipt
    $readback = Read-Json $pipelineReceiptPath
    if ([string]$readback.status -cne 'complete' -or $readback.validation.every_stage_rc_zero -ne $true) { throw 'Pipeline receipt readback failed' }
    Write-Host "WHITE_PILOT_PIPELINE_OK receipt=$pipelineReceiptPath raw_images=10000 prefix_images=1000 inventory_images=1000 archive_sha256=$receiptSha"
}
catch {
    $PipelineFailed = $_
    if ($RunRootOwned -and (Test-Path -LiteralPath $RunRoot -PathType Container)) {
        Assert-NoReparseChain $RunRoot 'white pilot owned failure root'
        if ((Get-DirectoryIdentity $RunRoot 'white pilot owned failure root') -cne $RunRootIdentity) { throw 'Refusing to write failure receipt after RunRoot identity changed' }
        $failurePath = Join-Path $RunRoot 'pipeline.failure.json'
        if (-not (Test-Path -LiteralPath $failurePath)) {
            Write-JsonNew $failurePath ([ordered]@{
                schema_version=1
                kind='otherimages_white_pilot_windows_failure_v1'
                status='failed'
                message=$_.Exception.Message
                exception_type=$_.Exception.GetType().FullName
                script_stack_trace=$_.ScriptStackTrace
                archive=$ArchiveBinding
                utc=[DateTime]::UtcNow.ToString('o')
            })
        }
        Write-Host "WHITE_PILOT_PIPELINE_FAILED evidence=$failurePath error=$($_.Exception.Message)"
    }
    throw
}
