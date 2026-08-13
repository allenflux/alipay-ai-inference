<#
.SYNOPSIS
Builds and formally validates the frozen white-document .NET CPU delivery.

.DESCRIPTION
Windows PowerShell 5.1 compatible.  The source checkout, teacher publication,
and student-training publication are fixed.  Status-bar/PP-OCR asset identities
and all six absolute efficiency budgets must be declared before the run; absent
declarations produce one diagnostic JSON value on stdout and exit code 3
without reserving the formal output root.

The formal path publishes a CPU-only .NET executable, uses the repository's
strict white CPU benchmark for one smoke/warmup plus three complete frozen-test
runs, and scores the first complete run against the same sealed Paddle teacher
test split.  Teacher agreement is pseudo-label parity, not human ground truth.
#>
[CmdletBinding()]
param(
    [string]$RepoRoot = 'C:\f3-white-code-3080a69',
    [string]$TeacherRunRoot = 'C:\f3-white-teacher-3080a69-pilot1000-a',
    [string]$TrainRunRoot = 'C:\f3-white-train-3080a69-pilot1000-a',
    [string]$RunRoot = 'C:\f3-white-cpu-delivery-3080a69-pilot1000-a',
    [string]$PythonExe = 'D:\alipay-ai-data\alipay-ai-inference\.venv-cu126\Scripts\python.exe',
    [string]$DeviceModel,
    [string]$ExpectedDeviceModelSha256,
    [string]$ExpectedDeviceContractSha256,
    [string]$OcrBundle,
    [string]$ExpectedOcrContractSha256,
    [string]$ExpectedOcrBundleClosureSha256,
    [double]$MaxP50LatencyMilliseconds,
    [double]$MaxP95LatencyMilliseconds,
    [double]$MinThroughputImagesPerSecond,
    [double]$MaxPeakWorkingSetMiB,
    [double]$MaxPeakPrivateBytesMiB,
    [double]$MaxPackageSizeMiB
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$Utf8NoBom = New-Object Text.UTF8Encoding($false)
$Ascii = [Text.Encoding]::ASCII
$RequiredHead = '3080a692a37d7efb0f926cce46de831d17f0e4db'
$RequiredTree = 'fb7a21f99139edd15eb1bb10e311039ebe28ebf5'
$RequiredTrainWrapperSha256 = '55227d5782e55f359d2e3a5f6deee9f24bb6ed19947f847e2d719f6b8d3e5518'
$RequiredTrainWrapperBytes = [int64]51503
$GitExe = 'C:\Program Files\Git\cmd\git.exe'
$DotnetExe = 'C:\Program Files\dotnet\dotnet.exe'
$PipelineStartedUtc = [DateTime]::UtcNow
$RunRootOwned = $false
$RunRootIdentity = $null
$RunParentIdentity = $null
$Failure = $null

$RequiredCode = [ordered]@{
    'scripts\otherimages-dotnet-cpu-benchmark.ps1' = [ordered]@{ sha256='7ee5dc8554c3fb47f178de5ee2995be21863846b0ce7d76fa3b432a10b993767'; size_bytes=[int64]43667 }
    'scripts\otherimages-dotnet-evaluate.py' = [ordered]@{ sha256='4941e438a45f79dced77e884d2f501b03cb0ce17bf3080a766cc815adc5f548f'; size_bytes=[int64]481 }
    'src\transfer_receipt_ai\otherimages_dotnet_evaluate.py' = [ordered]@{ sha256='eceefe64495e828bcaeb4af83fc309009f4fe427872b2bf81b4ca57e3dd2f2ad'; size_bytes=[int64]54667 }
    'dotnet\ReceiptMlNet.Cli\ReceiptMlNet.Cli.csproj' = [ordered]@{ sha256='55baaec1d4d46d4d6518e8e8fafd207b47afa98ac87cb2124f623231f83eba45'; size_bytes=[int64]1625 }
    'dotnet\ReceiptMlNet.Cli\Program.cs' = [ordered]@{ sha256='8543deda90eeb14dbb1ec84b58c0edade01b9a030a34e04900438ade83cfd64c'; size_bytes=[int64]90262 }
    'dotnet\ReceiptMlNet.Cli\PaddleOcrDeliveryBundle.cs' = [ordered]@{ sha256='0dceae19ce602ef3ab6ef00a8cbbacba93b355ea4ad8972b63bbe94b966944f4'; size_bytes=[int64]42608 }
    'dotnet\ReceiptMlNet.Cli\PaddleOcrEngine.cs' = [ordered]@{ sha256='29d24e280fdc32f5445aedb56ec5b0af7fa77d0d405353c6c4a8e1cf16a836cf'; size_bytes=[int64]33783 }
    'dotnet\ReceiptMlNet.Cli\WhiteDocumentRoute.cs' = [ordered]@{ sha256='338cea78e5be30dfb5ff56cbfab530d446fedd0daf250d7de323c90447187770'; size_bytes=[int64]35825 }
    'dotnet\ReceiptMlNet.Cli\WhiteLineStudentBundle.cs' = [ordered]@{ sha256='bb3062943f713bc6364e063fa0666fa92a7e6e2ee887c70442717ab7587dfd84'; size_bytes=[int64]13078 }
    'dotnet\ReceiptMlNet.Cli\WhiteLineStudentEngine.cs' = [ordered]@{ sha256='88352d7344b19d8ad6d53e495dbcc3e2ac55ff7547619a5dc362c027ebd495aa'; size_bytes=[int64]7092 }
}

function Write-DiagnosticAndExit([string[]]$Missing) {
    $payload = [ordered]@{
        schema_version = 1
        kind = 'otherimages_white_cpu_delivery_preflight_diagnostic_v1'
        status = 'diagnostic_only'
        accepted = $false
        exit_code = 3
        formal_output_reserved = $false
        missing_declarations = @($Missing)
        message = 'Formal CPU delivery requires frozen status-bar/PP-OCR asset identities and all six predeclared absolute efficiency budgets.'
        utc = [DateTime]::UtcNow.ToString('o')
    }
    [Console]::Out.WriteLine(($payload | ConvertTo-Json -Depth 12 -Compress))
    exit 3
}

$declarations = @(
    'DeviceModel','ExpectedDeviceModelSha256','ExpectedDeviceContractSha256',
    'OcrBundle','ExpectedOcrContractSha256','ExpectedOcrBundleClosureSha256',
    'MaxP50LatencyMilliseconds','MaxP95LatencyMilliseconds','MinThroughputImagesPerSecond',
    'MaxPeakWorkingSetMiB','MaxPeakPrivateBytesMiB','MaxPackageSizeMiB'
)
[string[]]$missingDeclarations = @(
    foreach ($name in $declarations) {
        if (-not $PSBoundParameters.ContainsKey($name)) { "-$name"; continue }
        $value = $PSBoundParameters[$name]
        if ($value -is [string] -and [string]::IsNullOrWhiteSpace([string]$value)) { "-$name" }
    }
)
if ($missingDeclarations.Count -ne 0) { Write-DiagnosticAndExit $missingDeclarations }

[string[]]$unavailableAssets = @()
if ([string]::IsNullOrWhiteSpace($DeviceModel) -or -not (Test-Path -LiteralPath $DeviceModel -PathType Leaf)) {
    $unavailableAssets += '-DeviceModel(file_not_found)'
}
else {
    $declaredDeviceContract = $DeviceModel -replace '\.onnx$','.contract.json'
    if ($declaredDeviceContract -ceq $DeviceModel -or -not (Test-Path -LiteralPath $declaredDeviceContract -PathType Leaf)) {
        $unavailableAssets += '-DeviceModel(adjacent_contract_not_found)'
    }
}
if ([string]::IsNullOrWhiteSpace($OcrBundle) -or -not (Test-Path -LiteralPath $OcrBundle -PathType Container)) {
    $unavailableAssets += '-OcrBundle(directory_not_found)'
}
elseif (-not (Test-Path -LiteralPath (Join-Path $OcrBundle 'paddle_ocr_delivery.contract.json') -PathType Leaf)) {
    $unavailableAssets += '-OcrBundle(delivery_contract_not_found)'
}
if ($unavailableAssets.Count -ne 0) { Write-DiagnosticAndExit $unavailableAssets }

function Require-Sha256([string]$Value, [string]$Description) {
    if ($Value -cnotmatch '^[0-9a-f]{64}$') { throw "$Description must be one predeclared lowercase SHA-256." }
    return $Value
}

function Require-PositiveFinite([double]$Value, [string]$Description) {
    if ([double]::IsNaN($Value) -or [double]::IsInfinity($Value) -or $Value -le 0) {
        throw "$Description must be finite and positive."
    }
    return $Value
}

foreach ($budgetName in @(
    'MaxP50LatencyMilliseconds','MaxP95LatencyMilliseconds','MinThroughputImagesPerSecond',
    'MaxPeakWorkingSetMiB','MaxPeakPrivateBytesMiB','MaxPackageSizeMiB')) {
    [void](Require-PositiveFinite ([double]$PSBoundParameters[$budgetName]) "-$budgetName")
}
if ($MaxP50LatencyMilliseconds -gt $MaxP95LatencyMilliseconds) {
    throw '-MaxP50LatencyMilliseconds may not exceed -MaxP95LatencyMilliseconds.'
}
foreach ($hashDeclaration in @(
    @($ExpectedDeviceModelSha256,'-ExpectedDeviceModelSha256'),
    @($ExpectedDeviceContractSha256,'-ExpectedDeviceContractSha256'),
    @($ExpectedOcrContractSha256,'-ExpectedOcrContractSha256'),
    @($ExpectedOcrBundleClosureSha256,'-ExpectedOcrBundleClosureSha256'))) {
    [void](Require-Sha256 ([string]$hashDeclaration[0]) ([string]$hashDeclaration[1]))
}

function Require-FixedPath([string]$Actual, [string]$Expected, [string]$Description) {
    $full = [IO.Path]::GetFullPath($Actual)
    $fixed = [IO.Path]::GetFullPath($Expected)
    if (-not $full.Equals($fixed,[StringComparison]::OrdinalIgnoreCase)) {
        throw "$Description must equal frozen path: expected=$fixed observed=$full"
    }
    if ([IO.Path]::GetPathRoot($full) -cne 'C:\') { throw "$Description must remain on C:\: $full" }
    return $full
}

function Resolve-RequiredFile([string]$Path, [string]$Description) {
    if ([string]::IsNullOrWhiteSpace($Path) -or -not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        throw "Missing $Description at predeclared path: $Path"
    }
    return [IO.Path]::GetFullPath((Resolve-Path -LiteralPath $Path).ProviderPath)
}

function Resolve-RequiredDirectory([string]$Path, [string]$Description) {
    if ([string]::IsNullOrWhiteSpace($Path) -or -not (Test-Path -LiteralPath $Path -PathType Container)) {
        throw "Missing $Description at predeclared path: $Path"
    }
    return [IO.Path]::GetFullPath((Resolve-Path -LiteralPath $Path).ProviderPath)
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

function Get-Sha256([string]$Path) {
    return (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()
}

function Get-Binding([string]$Path) {
    $resolved = Resolve-RequiredFile $Path 'bound file'
    $item = Get-Item -LiteralPath $resolved -Force
    return [pscustomobject][ordered]@{ path=$resolved; sha256=Get-Sha256 $resolved; size_bytes=[int64]$item.Length }
}

function Assert-Binding([object]$Expected, [string]$Description) {
    $observed = Get-Binding ([string]$Expected.path)
    if ([string]$observed.sha256 -cne [string]$Expected.sha256 -or [int64]$observed.size_bytes -ne [int64]$Expected.size_bytes) {
        throw "$Description SHA-256/size binding changed."
    }
}

function Get-TextSha256([string]$Text) {
    $hasher = [Security.Cryptography.SHA256]::Create()
    try {
        $bytes = $Utf8NoBom.GetBytes($Text)
        return ([BitConverter]::ToString($hasher.ComputeHash($bytes))).Replace('-','').ToLowerInvariant()
    }
    finally { $hasher.Dispose() }
}

function Get-DirectoryClosure([string]$Path, [string]$Description) {
    $resolved = Resolve-RequiredDirectory $Path $Description
    $reparseMembers = @(
        Get-ChildItem -LiteralPath $resolved -Recurse -Force |
            Where-Object { ($_.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0 }
    )
    if ($reparseMembers.Count -ne 0) { throw "$Description contains a reparse member: $($reparseMembers[0].FullName)" }
    [string[]]$lines = @(
        foreach ($file in @(Get-ChildItem -LiteralPath $resolved -Recurse -File -Force | Sort-Object FullName)) {
            $relative = $file.FullName.Substring($resolved.TrimEnd('\').Length).TrimStart('\').Replace('\','/')
            "{0}`t{1}`t{2}" -f $relative,(Get-Sha256 $file.FullName),[int64]$file.Length
        }
    )
    $closureText = if ($lines.Count -eq 0) { '' } else { ($lines -join "`n") + "`n" }
    return [pscustomobject][ordered]@{
        path=$resolved
        file_count=$lines.Count
        size_bytes=[int64](($lines | ForEach-Object {
            $parts = $_ -split "`t"; [int64]$parts[2]
        } | Measure-Object -Sum).Sum)
        closure_format='relative_path_tab_sha256_tab_bytes_lf_v1'
        closure_sha256=Get-TextSha256 $closureText
    }
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
    Write-TextNew $Path (($Value | ConvertTo-Json -Depth 80) + "`r`n")
}

function Write-RcNew([string]$Path, [int]$Rc) {
    if (Test-Path -LiteralPath $Path) { throw "Refusing to overwrite RC evidence: $Path" }
    [byte[]]$expected = $Ascii.GetBytes($Rc.ToString([Globalization.CultureInfo]::InvariantCulture) + "`r`n")
    [IO.File]::WriteAllBytes($Path,$expected)
    [byte[]]$observed = [IO.File]::ReadAllBytes($Path)
    if ($observed.Length -ne $expected.Length) { throw "RC evidence length differs: $Path" }
    for ($index=0; $index -lt $expected.Length; $index++) {
        if ($observed[$index] -ne $expected[$index]) { throw "RC evidence is not exact ASCII integer CRLF: $Path" }
    }
}

function Read-Json([string]$Path) {
    return (Get-Content -LiteralPath (Resolve-RequiredFile $Path 'JSON evidence') -Raw -Encoding UTF8) | ConvertFrom-Json
}

function Initialize-NativeDirectoryType {
    if ($null -ne ('WhiteCpuDeliveryNativeDirectoryV1' -as [type])) { return }
    $source = @'
using System;
using System.ComponentModel;
using System.Runtime.InteropServices;
using Microsoft.Win32.SafeHandles;
public static class WhiteCpuDeliveryNativeDirectoryV1 {
    [StructLayout(LayoutKind.Sequential)] private struct Info {
        public uint FileAttributes; public System.Runtime.InteropServices.ComTypes.FILETIME CreationTime;
        public System.Runtime.InteropServices.ComTypes.FILETIME LastAccessTime;
        public System.Runtime.InteropServices.ComTypes.FILETIME LastWriteTime; public uint VolumeSerialNumber;
        public uint FileSizeHigh; public uint FileSizeLow; public uint NumberOfLinks;
        public uint FileIndexHigh; public uint FileIndexLow;
    }
    [DllImport("kernel32.dll",CharSet=CharSet.Unicode,SetLastError=true,ExactSpelling=true)]
    [return: MarshalAs(UnmanagedType.Bool)] private static extern bool CreateDirectoryW(string path,IntPtr attributes);
    [DllImport("kernel32.dll",CharSet=CharSet.Unicode,SetLastError=true,ExactSpelling=true)]
    private static extern SafeFileHandle CreateFileW(string path,uint access,uint share,IntPtr security,uint creation,uint flags,IntPtr template);
    [DllImport("kernel32.dll",SetLastError=true)] [return: MarshalAs(UnmanagedType.Bool)]
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
    return [WhiteCpuDeliveryNativeDirectoryV1]::Identity((Resolve-RequiredDirectory $Path $Description))
}

function Assert-GitAuthority([object]$GitBinding) {
    Assert-Binding $GitBinding 'fixed Git executable'
    [string[]]$head = @(& $GitExe -C $RepoRoot rev-parse HEAD 2>&1)
    if ($LASTEXITCODE -ne 0 -or $head.Count -ne 1 -or $head[0] -cne $RequiredHead) { throw "Source checkout HEAD must be $RequiredHead" }
    [string[]]$tree = @(& $GitExe -C $RepoRoot rev-parse 'HEAD^{tree}' 2>&1)
    if ($LASTEXITCODE -ne 0 -or $tree.Count -ne 1 -or $tree[0] -cne $RequiredTree) { throw "Source checkout tree must be $RequiredTree" }
    [string[]]$status = @(& $GitExe -C $RepoRoot status --porcelain=v1 --untracked-files=all 2>&1)
    if ($LASTEXITCODE -ne 0 -or $status.Count -ne 0) { throw 'Independent source checkout is not completely clean, including untracked files.' }
    [string[]]$diff = @(& $GitExe -C $RepoRoot diff --no-ext-diff --quiet --exit-code HEAD -- 2>&1)
    if ($LASTEXITCODE -ne 0 -or $diff.Count -ne 0) { throw 'Independent source checkout tracked tree differs from HEAD.' }
    Assert-Binding $GitBinding 'fixed Git executable after authority query'
}

function Get-CodeBindings {
    $bindings = [ordered]@{}
    foreach ($relative in $RequiredCode.Keys) {
        $binding = Get-Binding (Join-Path $RepoRoot $relative)
        $expected = $RequiredCode[$relative]
        if ([string]$binding.sha256 -cne [string]$expected.sha256 -or [int64]$binding.size_bytes -ne [int64]$expected.size_bytes) {
            throw "Frozen source SHA/size gate failed: $relative"
        }
        $bindings[$relative] = $binding
    }
    return $bindings
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

function Format-InvariantDouble([double]$Value) {
    return $Value.ToString('R',[Globalization.CultureInfo]::InvariantCulture)
}

function Stop-ProcessTree([int]$RootPid) {
    for ($attempt=0; $attempt -lt 3; $attempt++) {
        $all = @(Get-CimInstance Win32_Process -ErrorAction SilentlyContinue)
        $pending = New-Object Collections.Generic.List[int]
        $descendants = New-Object Collections.Generic.List[int]
        $pending.Add($RootPid)
        for ($index=0; $index -lt $pending.Count; $index++) {
            $parentPid = $pending[$index]
            foreach ($child in @($all | Where-Object { [int]$_.ParentProcessId -eq $parentPid })) {
                $childPid = [int]$child.ProcessId
                if (-not $pending.Contains($childPid)) { $pending.Add($childPid); $descendants.Add($childPid) }
            }
        }
        [int[]]$targets = @($descendants.ToArray())
        [array]::Reverse($targets)
        foreach ($targetPid in @($targets + $RootPid)) {
            Stop-Process -Id $targetPid -Force -ErrorAction SilentlyContinue
        }
        Start-Sleep -Milliseconds 250
    }
}

function Add-ObservedDescendantPids([int]$RootPid, [Collections.Generic.HashSet[int]]$Observed) {
    $all = @(Get-CimInstance Win32_Process -ErrorAction SilentlyContinue)
    $pending = New-Object Collections.Generic.List[int]
    $pending.Add($RootPid)
    for ($index=0; $index -lt $pending.Count; $index++) {
        $parentPid = $pending[$index]
        foreach ($child in @($all | Where-Object { [int]$_.ParentProcessId -eq $parentPid })) {
            $childPid = [int]$child.ProcessId
            if (-not $pending.Contains($childPid)) { $pending.Add($childPid); [void]$Observed.Add($childPid) }
        }
    }
}

function Wait-ProcessIdsAbsent([int[]]$ProcessIds, [int]$TimeoutMilliseconds) {
    $deadline = [DateTime]::UtcNow.AddMilliseconds($TimeoutMilliseconds)
    do {
        [int[]]$alive = @($ProcessIds | Where-Object { $null -ne (Get-Process -Id $_ -ErrorAction SilentlyContinue) })
        if ($alive.Count -eq 0) { return $true }
        Start-Sleep -Milliseconds 250
    } while ([DateTime]::UtcNow -lt $deadline)
    return $false
}

function Stop-ObservedProcessIds([int[]]$ProcessIds) {
    foreach ($processId in $ProcessIds) { Stop-Process -Id $processId -Force -ErrorAction SilentlyContinue }
}

function Invoke-ClosedStage([string]$Name, [string]$Executable, [string[]]$Arguments, [string]$WorkingDirectory) {
    $stageRoot = Join-Path $LogsRoot $Name
    if (Test-Path -LiteralPath $stageRoot) { throw "Stage root already exists: $stageRoot" }
    [IO.Directory]::CreateDirectory($stageRoot) | Out-Null
    $stdoutPath = Join-Path $stageRoot 'stdout.txt'
    $stderrPath = Join-Path $stageRoot 'stderr.txt'
    $rcPath = Join-Path $stageRoot 'rc.txt'
    $info = New-Object Diagnostics.ProcessStartInfo
    $info.FileName = $Executable
    $info.Arguments = ConvertTo-NativeCommandLine $Arguments
    $info.WorkingDirectory = $WorkingDirectory
    $info.UseShellExecute = $false
    $info.CreateNoWindow = $true
    $info.RedirectStandardOutput = $true
    $info.RedirectStandardError = $true
    $info.EnvironmentVariables['PYTHONUTF8'] = '1'
    $info.EnvironmentVariables['PYTHONIOENCODING'] = 'utf-8:strict'
    $info.EnvironmentVariables['PYTHONDONTWRITEBYTECODE'] = '1'
    $info.EnvironmentVariables['PYTHONNOUSERSITE'] = '1'
    $info.EnvironmentVariables['PYTHONPATH'] = (Join-Path $RepoRoot 'src')
    [void]$info.EnvironmentVariables.Remove('PYTHONHOME')
    $info.EnvironmentVariables['DOTNET_CLI_HOME'] = (Join-Path $RunRoot 'dotnet-cli-home')
    $info.EnvironmentVariables['NUGET_PACKAGES'] = (Join-Path $RunRoot 'nuget-packages')
    $info.EnvironmentVariables['NUGET_HTTP_CACHE_PATH'] = (Join-Path $RunRoot 'nuget-http-cache')
    $info.EnvironmentVariables['DOTNET_CLI_TELEMETRY_OPTOUT'] = '1'
    $info.EnvironmentVariables['DOTNET_NOLOGO'] = '1'
    $info.EnvironmentVariables['TEMP'] = (Join-Path $RunRoot 'temp')
    $info.EnvironmentVariables['TMP'] = (Join-Path $RunRoot 'temp')
    $process = New-Object Diagnostics.Process
    $process.StartInfo = $info
    $started = [DateTime]::UtcNow
    $startedProcess = $false
    $pidValue = $null
    $forcedStop = $false
    $stdoutTask = $null
    $stderrTask = $null
    $exitCode = $null
    $stageError = $null
    $cleanupError = $null
    $observedDescendantPids = New-Object 'Collections.Generic.HashSet[int]'
    $descendantAbsenceProven = $false
    try {
        Write-Host "WHITE_CPU_STAGE_START stage=$Name"
        if (-not $process.Start()) { throw "Unable to start stage: $Name" }
        $startedProcess = $true
        $pidValue = [int]$process.Id
        Add-ObservedDescendantPids $pidValue $observedDescendantPids
        $stdoutTask = $process.StandardOutput.ReadToEndAsync()
        $stderrTask = $process.StandardError.ReadToEndAsync()
        $lastReport = $started
        while (-not $process.WaitForExit(1000)) {
            Add-ObservedDescendantPids $pidValue $observedDescendantPids
            $now = [DateTime]::UtcNow
            if (($now-$lastReport).TotalSeconds -ge 60) {
                try {
                    $process.Refresh()
                    Write-Host ("WHITE_CPU_STAGE_ALIVE stage={0} elapsed_s={1} pid={2} cpu_s={3:N1} ws_bytes={4}" -f $Name,[int]($now-$started).TotalSeconds,$process.Id,$process.TotalProcessorTime.TotalSeconds,$process.WorkingSet64)
                }
                catch { Write-Host "WHITE_CPU_STAGE_ALIVE stage=$Name elapsed_s=$([int]($now-$started).TotalSeconds)" }
                $lastReport = $now
            }
        }
        $process.WaitForExit()
        $exitCode = [int]$process.ExitCode
    }
    catch { $stageError = $_ }
    finally {
        try {
            try {
                if ($startedProcess) { Add-ObservedDescendantPids $pidValue $observedDescendantPids }
                if ($startedProcess -and -not $process.HasExited) {
                    $forcedStop = $true
                    Stop-ProcessTree $pidValue
                    Stop-ObservedProcessIds @($observedDescendantPids)
                    if (-not $process.WaitForExit(30000)) { throw 'Process tree remained alive after forced stop and bounded wait.' }
                }
                if ($startedProcess -and $process.HasExited) { $process.WaitForExit(); $exitCode = [int]$process.ExitCode }
                [int[]]$observedIds = @($observedDescendantPids | Sort-Object)
                $descendantAbsenceProven = Wait-ProcessIdsAbsent $observedIds 30000
                if (-not $descendantAbsenceProven) {
                    $forcedStop = $true
                    Stop-ObservedProcessIds $observedIds
                    if (-not (Wait-ProcessIdsAbsent $observedIds 30000)) { throw "Observed descendant PIDs remained alive after bounded cleanup: $($observedIds -join ',')" }
                    throw "Stage retained descendant PIDs and required forced cleanup: $($observedIds -join ',')"
                }
            }
            catch { $cleanupError = $_ }
            try {
                foreach ($streamTask in @($stdoutTask,$stderrTask)) {
                    if ($null -ne $streamTask -and -not $streamTask.IsCompleted -and -not $streamTask.Wait(30000)) {
                        throw 'Redirected child stream did not close within bounded cleanup wait.'
                    }
                }
                $stdout = if ($null -ne $stdoutTask -and $stdoutTask.IsCompleted) { [string]$stdoutTask.Result } else { '' }
                $stderr = if ($null -ne $stderrTask -and $stderrTask.IsCompleted) { [string]$stderrTask.Result } else { '' }
                [IO.File]::WriteAllText($stdoutPath,$stdout,$Utf8NoBom)
                [IO.File]::WriteAllText($stderrPath,$stderr,$Utf8NoBom)
                if ($null -ne $exitCode) { Write-RcNew $rcPath ([int]$exitCode) }
                $stderrNonEmpty = ([Text.Encoding]::UTF8.GetByteCount($stderr) -ne 0)
                if ($null -ne $stageError -or $null -ne $cleanupError -or $null -eq $exitCode -or [int]$exitCode -ne 0 -or $stderrNonEmpty) {
                    Write-JsonNew (Join-Path $stageRoot 'stage.failure.json') ([ordered]@{
                        schema_version=1; kind='otherimages_white_cpu_delivery_windows_stage_failure_v1'; status='failed'; stage=$Name
                        pid=$pidValue; process_started=$startedProcess; forced_process_tree_stop=$forcedStop; exit_code=$exitCode; stderr_nonempty=$stderrNonEmpty
                        observed_descendant_pids=@($observedDescendantPids | Sort-Object); descendant_absence_proven=$descendantAbsenceProven
                        stage_error=$(if ($null -eq $stageError) { $null } else { $stageError.Exception.Message })
                        cleanup_error=$(if ($null -eq $cleanupError) { $null } else { $cleanupError.Exception.Message })
                        stdout=Get-Binding $stdoutPath; stderr=Get-Binding $stderrPath
                        rc_file=$(if (Test-Path -LiteralPath $rcPath) { Get-Binding $rcPath } else { $null })
                        utc=[DateTime]::UtcNow.ToString('o')
                    })
                }
            }
            catch { if ($null -eq $cleanupError) { $cleanupError = $_ } }
        }
        finally { $process.Dispose() }
    }
    if ($null -ne $stageError) { throw $stageError }
    if ($null -ne $cleanupError) { throw $cleanupError }
    if ($null -eq $exitCode) { throw "Stage exit code unavailable: $Name" }
    $result = [pscustomobject][ordered]@{
        name=$Name; pid=$pidValue; rc=[int]$exitCode; forced_process_tree_stop=$forcedStop
        started_utc=$started.ToString('o'); completed_utc=[DateTime]::UtcNow.ToString('o')
        elapsed_seconds=([DateTime]::UtcNow-$started).TotalSeconds
        observed_descendant_pids=@($observedDescendantPids | Sort-Object); descendant_absence_proven=$descendantAbsenceProven
        stdout=Get-Binding $stdoutPath; stderr=Get-Binding $stderrPath; rc_file=Get-Binding $rcPath
    }
    if ($result.rc -ne 0) { throw "Stage failed: stage=$Name rc=$($result.rc) logs=$stageRoot" }
    if ([int64]$result.stderr.size_bytes -ne 0) { throw "Stage emitted stderr: stage=$Name path=$stderrPath" }
    Write-Host "WHITE_CPU_STAGE_EXIT stage=$Name rc=0 elapsed_s=$([int]$result.elapsed_seconds)"
    return $result
}

function Assert-ExactFileTree([string]$Root, [string[]]$Expected, [string]$Description) {
    [string[]]$files = @(Get-ChildItem -LiteralPath $Root -File -Force | ForEach-Object { $_.Name } | Sort-Object)
    [string[]]$directories = @(Get-ChildItem -LiteralPath $Root -Directory -Force | ForEach-Object { $_.Name })
    [string[]]$expectedSorted = @($Expected | Sort-Object)
    if (($files -join '|') -cne ($expectedSorted -join '|') -or $directories.Count -ne 0) { throw "$Description is not its exact closed file tree." }
}

function Assert-ExactWhiteResultsTree([string]$ResultsRoot, [int]$ExpectedResults) {
    $resolved = Resolve-RequiredDirectory $ResultsRoot 'white inference results root'
    [object[]]$files = @(Get-ChildItem -LiteralPath $resolved -File -Force)
    [object[]]$directories = @(Get-ChildItem -LiteralPath $resolved -Directory -Force)
    if ($files.Count -ne 3 -or $directories.Count -ne 1 -or [string]$directories[0].Name -cne 'input-list') {
        throw "White inference output root tree failed: root=$resolved files=$($files.Count) directories=$($directories.Count) expected_results=$ExpectedResults"
    }
    foreach ($required in @('inference_summary.json','inference_manifest.json','inference_errors.jsonl')) {
        if (-not (Test-Path -LiteralPath (Join-Path $resolved $required) -PathType Leaf)) { throw "White inference output omitted required file: $required" }
    }
    $resultRoot = [string]$directories[0].FullName
    [object[]]$resultDirectories = @(Get-ChildItem -LiteralPath $resultRoot -Directory -Force)
    [object[]]$resultFiles = @(Get-ChildItem -LiteralPath $resultRoot -File -Force)
    if ($resultDirectories.Count -ne 0 -or $resultFiles.Count -ne $ExpectedResults -or @($resultFiles | Where-Object { $_.Extension -cne '.json' -or $_.BaseName -cnotmatch '^[0-9a-f]{64}$' }).Count -ne 0) {
        throw 'White inference output contains a non-JSON or wrong-count result member.'
    }
}

try {
    $RepoRoot = Require-FixedPath $RepoRoot 'C:\f3-white-code-3080a69' 'source clone'
    $TeacherRunRoot = Require-FixedPath $TeacherRunRoot 'C:\f3-white-teacher-3080a69-pilot1000-a' 'teacher run root'
    $TrainRunRoot = Require-FixedPath $TrainRunRoot 'C:\f3-white-train-3080a69-pilot1000-a' 'training run root'
    $RunRoot = Require-FixedPath $RunRoot 'C:\f3-white-cpu-delivery-3080a69-pilot1000-a' 'CPU delivery run root'
    $PythonExe = [IO.Path]::GetFullPath($PythonExe)
    $fixedPython = 'D:\alipay-ai-data\alipay-ai-inference\.venv-cu126\Scripts\python.exe'
    if (-not $PythonExe.Equals($fixedPython,[StringComparison]::OrdinalIgnoreCase)) { throw "PythonExe must equal fixed scorer interpreter: $fixedPython" }
    $DeviceModel = Resolve-RequiredFile $DeviceModel 'frozen status-bar device model'
    $DeviceContract = $DeviceModel -replace '\.onnx$','.contract.json'
    if ($DeviceContract -ceq $DeviceModel) { throw 'Frozen status-bar device model path must end in .onnx.' }
    $DeviceContract = Resolve-RequiredFile $DeviceContract 'adjacent status-bar device contract'
    $OcrBundle = Resolve-RequiredDirectory $OcrBundle 'frozen PP-OCR delivery bundle'
    foreach ($path in @($RepoRoot,$TeacherRunRoot,$TrainRunRoot,$PythonExe,$GitExe,$DotnetExe,$DeviceModel,$DeviceContract,$OcrBundle,$PSCommandPath)) {
        if (-not (Test-Path -LiteralPath $path)) { throw "Missing required authority path: $path" }
        Assert-NoReparseChain $path 'CPU delivery authority path'
    }
    if ((Get-Sha256 $DeviceModel) -cne $ExpectedDeviceModelSha256) { throw 'Status-bar device model differs from its predeclared SHA-256.' }
    if ((Get-Sha256 $DeviceContract) -cne $ExpectedDeviceContractSha256) { throw 'Status-bar device contract differs from its predeclared SHA-256.' }
    $OcrContractPath = Resolve-RequiredFile (Join-Path $OcrBundle 'paddle_ocr_delivery.contract.json') 'PP-OCR delivery contract'
    if ((Get-Sha256 $OcrContractPath) -cne $ExpectedOcrContractSha256) { throw 'PP-OCR delivery contract differs from its predeclared SHA-256.' }
    $OcrClosure = Get-DirectoryClosure $OcrBundle 'PP-OCR delivery bundle'
    if ([string]$OcrClosure.closure_sha256 -cne $ExpectedOcrBundleClosureSha256) { throw 'PP-OCR bundle closure differs from its predeclared SHA-256.' }

    $GitBinding = Get-Binding $GitExe
    $DotnetBinding = Get-Binding $DotnetExe
    $PythonBinding = Get-Binding $PythonExe
    Assert-GitAuthority $GitBinding
    $CodeBindings = Get-CodeBindings
    $WrapperBinding = Get-Binding $PSCommandPath

    $TeacherPipelinePath = Join-Path $TeacherRunRoot 'pipeline.receipt.json'
    $TeacherPipeline = Read-Json $TeacherPipelinePath
    if ([string]$TeacherPipeline.kind -cne 'otherimages_white_teacher_windows_pipeline_receipt_v1' -or [string]$TeacherPipeline.status -cne 'complete') { throw 'Teacher pipeline receipt kind/status failed.' }
    if ([string]$TeacherPipeline.source.head -cne $RequiredHead -or [string]$TeacherPipeline.source.tree -cne $RequiredTree -or $TeacherPipeline.source.tracked_and_untracked_clean -ne $true -or $TeacherPipeline.validation.teacher_sealed -ne $true -or $TeacherPipeline.validation.no_training -ne $true -or $TeacherPipeline.validation.every_stage_rc_zero -ne $true -or $TeacherPipeline.validation.every_stage_stderr_zero_bytes -ne $true -or $TeacherPipeline.validation.teacher_closure_independently_recomputed -ne $true) { throw 'Teacher pipeline formal validation gates failed.' }
    $TeacherRoot = Resolve-RequiredDirectory ([string]$TeacherPipeline.roots.teacher) 'sealed teacher publication'
    if (-not $TeacherRoot.Equals((Join-Path $TeacherRunRoot 'publications\paddle-teacher-consensus'),[StringComparison]::OrdinalIgnoreCase)) { throw 'Teacher publication is outside the frozen teacher run.' }
    Assert-ExactFileTree $TeacherRoot @('teacher_manifest.jsonl','reject_manifest.jsonl','teacher.contract.json','teacher.receipt.json') 'Teacher publication'
    $TeacherManifestPath = Join-Path $TeacherRoot 'teacher_manifest.jsonl'
    $TeacherContractPath = Join-Path $TeacherRoot 'teacher.contract.json'
    $TeacherReceiptPath = Join-Path $TeacherRoot 'teacher.receipt.json'
    $TeacherContract = Read-Json $TeacherContractPath
    $TeacherReceipt = Read-Json $TeacherReceiptPath
    if ([string]$TeacherContract.kind -cne 'otherimages_paddle_teacher_contract_v1' -or $TeacherContract.sealed -ne $true -or $TeacherContract.training_authorization -ne $false -or [int]$TeacherContract.counts.accepted_by_split.test -lt 1) { throw 'Teacher contract lacks a sealed nonempty frozen test split.' }
    $TeacherContractBinding = Get-Binding $TeacherContractPath
    if ([string]$TeacherReceipt.kind -cne 'otherimages_paddle_teacher_receipt_v1' -or $TeacherReceipt.sealed -ne $true -or [string]$TeacherReceipt.contract.sha256 -cne [string]$TeacherContractBinding.sha256 -or [int64]$TeacherReceipt.contract.size_bytes -ne [int64]$TeacherContractBinding.size_bytes -or [string]$TeacherReceipt.contract_closure_sha256 -cne [string]$TeacherContract.closure_sha256) { throw 'Teacher receipt does not close the teacher contract.' }

    $TrainPipelinePath = Join-Path $TrainRunRoot 'pipeline.receipt.json'
    $TrainPipeline = Read-Json $TrainPipelinePath
    if ([string]$TrainPipeline.kind -cne 'otherimages_white_student_training_windows_pipeline_receipt_v1' -or [string]$TrainPipeline.status -cne 'complete') { throw 'Student training pipeline receipt kind/status failed.' }
    if ([string]$TrainPipeline.source.head -cne $RequiredHead -or [string]$TrainPipeline.source.tree -cne $RequiredTree -or $TrainPipeline.source.clean -ne $true -or [string]$TrainPipeline.source.wrapper.sha256 -cne $RequiredTrainWrapperSha256 -or [int64]$TrainPipeline.source.wrapper.size_bytes -ne $RequiredTrainWrapperBytes -or $TrainPipeline.training_performed -ne $true -or $TrainPipeline.execution.training_performed -ne $true -or $TrainPipeline.execution.onnx_exported -ne $true -or $TrainPipeline.semantics.analysis_candidate_only -ne $true -or $TrainPipeline.semantics.teacher_parity_only -ne $true -or $TrainPipeline.semantics.independent_business_accuracy_proven -ne $false -or $TrainPipeline.semantics.cpu_publication_performed -ne $false -or $TrainPipeline.semantics.cpu_delivery_gate_passed -ne $false -or $TrainPipeline.semantics.test_inference_performed -ne $false -or $TrainPipeline.validation.every_stage_rc_zero -ne $true -or $TrainPipeline.validation.every_stage_stderr_zero_bytes -ne $true -or $TrainPipeline.validation.generic_text_line_only -ne $true -or $TrainPipeline.validation.generic_test_oov_fail_closed_by_source -ne $true -or $TrainPipeline.validation.test_split_oov_zero -ne $true -or $TrainPipeline.validation.train_val_test_closed -ne $true -or $TrainPipeline.validation.onnx_export_complete -ne $true -or $TrainPipeline.validation.test_split_used_for_training -ne $false) { throw 'Student training formal validation gates failed.' }
    if (-not ([IO.Path]::GetFullPath([string]$TrainPipeline.inputs.teacher_root)).Equals($TeacherRoot,[StringComparison]::OrdinalIgnoreCase) -or [string]$TrainPipeline.inputs.teacher_contract.sha256 -cne [string]$TeacherContractBinding.sha256 -or [string]$TrainPipeline.inputs.teacher_contract_closure_sha256 -cne [string]$TeacherContract.closure_sha256) { throw 'Student training receipt is not bound to this frozen teacher publication.' }
    $StudentBundle = Resolve-RequiredDirectory ([string]$TrainPipeline.student_bundle.root) 'white student bundle'
    if (-not ([IO.Path]::GetFullPath([string]$TrainPipeline.roots.student_bundle)).Equals($StudentBundle,[StringComparison]::OrdinalIgnoreCase)) { throw 'Training receipt roots.student_bundle differs from student_bundle.root.' }
    if (-not $StudentBundle.StartsWith($TrainRunRoot.TrimEnd('\')+'\',[StringComparison]::OrdinalIgnoreCase)) { throw 'Student bundle is outside the frozen training run.' }
    [object[]]$StudentContracts = @(Get-ChildItem -LiteralPath $StudentBundle -File -Filter '*.contract.json' -Force)
    if ($StudentContracts.Count -ne 1) { throw 'White student bundle must contain exactly one *.contract.json.' }
    $StudentContractPath = [string]$StudentContracts[0].FullName
    $StudentContract = Read-Json $StudentContractPath
    if ([string]$StudentContract.kind -cne 'receipt_ocr_ctc_v1' -or @($StudentContract.fields).Count -ne 1 -or [string]$StudentContract.fields[0] -cne 'generic_text_line' -or [string]$StudentContract.input.name -cne 'image' -or [string]$StudentContract.input.preprocess -cne 'opencv_exact_rgb_gray_letterbox_v1' -or [string]$StudentContract.output.name -cne 'logits') { throw 'White student contract kind/field/ABI/preprocess is invalid.' }
    $StudentModelPath = Resolve-RequiredFile (Join-Path $StudentBundle ([string]$StudentContract.onnx_file)) 'white student ONNX'
    $StudentCharsetPath = Resolve-RequiredFile (Join-Path $StudentBundle ([string]$StudentContract.charset_file)) 'white student charset'
    Assert-ExactFileTree $StudentBundle @([IO.Path]::GetFileName($StudentModelPath),[IO.Path]::GetFileName($StudentCharsetPath),[IO.Path]::GetFileName($StudentContractPath)) 'White student bundle'
    if ((Get-Sha256 $StudentModelPath) -cne [string]$StudentContract.onnx_sha256 -or (Get-Sha256 $StudentCharsetPath) -cne [string]$StudentContract.charset_sha256) { throw 'White student ONNX/charset differs from its contract.' }
    foreach ($bindingName in @('model','charset','contract')) {
        $declared = $TrainPipeline.student_bundle.bindings.$bindingName
        $actualPath = if ($bindingName -eq 'model') { $StudentModelPath } elseif ($bindingName -eq 'charset') { $StudentCharsetPath } else { $StudentContractPath }
        $actual = Get-Binding $actualPath
        if (-not ([IO.Path]::GetFullPath([string]$declared.path)).Equals($actualPath,[StringComparison]::OrdinalIgnoreCase) -or [string]$declared.sha256 -cne [string]$actual.sha256 -or [int64]$declared.size_bytes -ne [int64]$actual.size_bytes) { throw "Training receipt student $bindingName binding differs." }
    }
    $StudentClosure = Get-DirectoryClosure $StudentBundle 'white student bundle'

    if (Test-Path -LiteralPath $RunRoot) { throw "RunRoot must be brand-new: $RunRoot" }
    $RunParent = Split-Path -Parent $RunRoot
    Assert-NoReparseChain $RunParent 'CPU delivery output parent'
    Initialize-NativeDirectoryType
    $RunParentIdentity = Get-DirectoryIdentity $RunParent 'CPU delivery output parent'
    [WhiteCpuDeliveryNativeDirectoryV1]::CreateExclusive($RunRoot)
    $RunRootOwned = $true
    $RunRootIdentity = Get-DirectoryIdentity $RunRoot 'CPU delivery run root'
    $LogsRoot = Join-Path $RunRoot 'logs'
    $BuildSourceRoot = Join-Path $RunRoot 'build-source'
    $DeliveryPackageRoot = Join-Path $RunRoot 'publication\white-document-cpu-win-x64'
    $PublishRoot = Join-Path $DeliveryPackageRoot 'app'
    $PackagedDeviceRoot = Join-Path $DeliveryPackageRoot 'statusbar'
    $PackagedOcrBundle = Join-Path $DeliveryPackageRoot 'ppocr'
    $PackagedStudentBundle = Join-Path $DeliveryPackageRoot 'white-student'
    $BenchmarkRoot = Join-Path $RunRoot 'benchmark'
    $EvaluationRoot = Join-Path $RunRoot 'teacher-agreement-test'
    [IO.Directory]::CreateDirectory($LogsRoot) | Out-Null
    [IO.Directory]::CreateDirectory($BuildSourceRoot) | Out-Null
    foreach ($cacheRoot in @('dotnet-cli-home','nuget-packages','nuget-http-cache','temp')) {
        [IO.Directory]::CreateDirectory((Join-Path $RunRoot $cacheRoot)) | Out-Null
    }

    Copy-Item -LiteralPath (Join-Path $RepoRoot 'dotnet\ReceiptMlNet.Cli') -Destination $BuildSourceRoot -Recurse
    $CopiedProject = Join-Path $BuildSourceRoot 'ReceiptMlNet.Cli\ReceiptMlNet.Cli.csproj'
    $publishStage = Invoke-ClosedStage 'dotnet-publish-cpu' $DotnetExe @(
        'publish',$CopiedProject,'--configuration','Release','--runtime','win-x64','--self-contained','true',
        '--output',$PublishRoot,'-p:OnnxRuntimeFlavor=cpu','-p:ContinuousIntegrationBuild=true'
    ) $RepoRoot
    $Executable = Resolve-RequiredFile (Join-Path $PublishRoot 'ReceiptMlNet.Cli.exe') 'published .NET CPU executable'

    [IO.Directory]::CreateDirectory($PackagedDeviceRoot) | Out-Null
    $PackagedDeviceModel = Join-Path $PackagedDeviceRoot ([IO.Path]::GetFileName($DeviceModel))
    $PackagedDeviceContract = Join-Path $PackagedDeviceRoot ([IO.Path]::GetFileName($DeviceContract))
    Copy-Item -LiteralPath $DeviceModel -Destination $PackagedDeviceModel
    Copy-Item -LiteralPath $DeviceContract -Destination $PackagedDeviceContract
    Copy-Item -LiteralPath $OcrBundle -Destination $PackagedOcrBundle -Recurse
    Copy-Item -LiteralPath $StudentBundle -Destination $PackagedStudentBundle -Recurse
    $PackagedDeviceModel = Resolve-RequiredFile $PackagedDeviceModel 'packaged status-bar model'
    $PackagedDeviceContract = Resolve-RequiredFile $PackagedDeviceContract 'packaged status-bar contract'
    if ((Get-Sha256 $PackagedDeviceModel) -cne $ExpectedDeviceModelSha256 -or (Get-Sha256 $PackagedDeviceContract) -cne $ExpectedDeviceContractSha256) { throw 'Packaged status-bar model/contract differs from frozen input.' }
    $PackagedOcrClosure = Get-DirectoryClosure $PackagedOcrBundle 'packaged PP-OCR bundle'
    if ([string]$PackagedOcrClosure.closure_sha256 -cne [string]$OcrClosure.closure_sha256 -or [int64]$PackagedOcrClosure.size_bytes -ne [int64]$OcrClosure.size_bytes -or [int]$PackagedOcrClosure.file_count -ne [int]$OcrClosure.file_count) { throw 'Packaged PP-OCR bundle differs from frozen input closure.' }
    $PackagedStudentClosure = Get-DirectoryClosure $PackagedStudentBundle 'packaged white student bundle'
    if ([string]$PackagedStudentClosure.closure_sha256 -cne [string]$StudentClosure.closure_sha256 -or [int64]$PackagedStudentClosure.size_bytes -ne [int64]$StudentClosure.size_bytes -or [int]$PackagedStudentClosure.file_count -ne [int]$StudentClosure.file_count) { throw 'Packaged student bundle differs from frozen training output.' }

    $TestInputList = Join-Path $RunRoot 'frozen-test-inputs.txt'
    $testSources = New-Object Collections.Generic.List[string]
    $seenTestSources = New-Object 'Collections.Generic.HashSet[string]' ([StringComparer]::OrdinalIgnoreCase)
    foreach ($line in [IO.File]::ReadAllLines($TeacherManifestPath,$Utf8NoBom)) {
        if ([string]::IsNullOrWhiteSpace($line)) { throw 'Teacher manifest contains a blank line.' }
        $record = $line | ConvertFrom-Json
        if ([string]$record.split -cne 'test') { continue }
        if ([string]$record.split_use -cne 'heldout_test' -or $record.training_eligible -ne $false -or $record.evaluation_only -ne $true -or $record.held_out -ne $true) { throw 'Selected teacher test record has invalid held-out flags.' }
        $source = Resolve-RequiredFile ([string]$record.source_absolute_path) 'frozen teacher test image'
        if ((Get-Sha256 $source) -cne [string]$record.raw_sha256) { throw "Frozen teacher test image SHA differs: $source" }
        if (-not $seenTestSources.Add($source)) { throw "Duplicate frozen teacher test source: $source" }
        $testSources.Add($source)
    }
    if ($testSources.Count -ne [int]$TeacherContract.counts.accepted_by_split.test -or $testSources.Count -lt 1) { throw 'Frozen test input count differs from the sealed teacher contract.' }
    [IO.File]::WriteAllLines($TestInputList,$testSources.ToArray(),$Utf8NoBom)

    $benchmarkScript = Join-Path $RepoRoot 'scripts\otherimages-dotnet-cpu-benchmark.ps1'
    $benchmarkStage = Invoke-ClosedStage 'cpu-smoke-and-full-test' 'powershell.exe' @(
        '-NoLogo','-NoProfile','-NonInteractive','-ExecutionPolicy','Bypass','-File',$benchmarkScript,
        '-Executable',$Executable,'-DeviceModel',$PackagedDeviceModel,'-OcrBundle',$PackagedOcrBundle,
        '-WhiteStudentBundle',$PackagedStudentBundle,'-InputList',$TestInputList,'-OutputRoot',$BenchmarkRoot,
        '-WarmupRuns','1','-WarmupImages','8','-Repetitions','3','-PollIntervalMilliseconds','200',
        '-MaxP50LatencyMilliseconds',(Format-InvariantDouble $MaxP50LatencyMilliseconds),
        '-MaxP95LatencyMilliseconds',(Format-InvariantDouble $MaxP95LatencyMilliseconds),
        '-MinThroughputImagesPerSecond',(Format-InvariantDouble $MinThroughputImagesPerSecond),
        '-MaxPeakWorkingSetMiB',(Format-InvariantDouble $MaxPeakWorkingSetMiB),
        '-MaxPeakPrivateBytesMiB',(Format-InvariantDouble $MaxPeakPrivateBytesMiB),
        '-MaxPackageSizeMiB',(Format-InvariantDouble $MaxPackageSizeMiB)
    ) $RepoRoot
    $BenchmarkReportPath = Join-Path $BenchmarkRoot 'white-cpu-benchmark.json'
    $Benchmark = Read-Json $BenchmarkReportPath
    if ([string]$Benchmark.kind -cne 'otherimages_dotnet_white_cpu_benchmark_v1' -or $Benchmark.accepted -ne $true -or $Benchmark.diagnostic_only -ne $false -or $Benchmark.efficiency_gate.formal_gate_configured -ne $true -or $Benchmark.efficiency_gate.accepted -ne $true -or @($Benchmark.warmup).Count -ne 1 -or @($Benchmark.measured.runs).Count -ne 3 -or [int]$Benchmark.workload.input_count -ne $testSources.Count -or [string]$Benchmark.workload.requested_device -cne 'cpu' -or [string]$Benchmark.workload.paddle_ocr_provider -cne 'cpu' -or [string]$Benchmark.workload.white_student_provider -cne 'cpu') { throw 'CPU smoke/full-test benchmark did not pass its complete formal gate.' }
    if ([string]$Benchmark.artifacts.device_model.sha256 -cne $ExpectedDeviceModelSha256 -or [string]$Benchmark.artifacts.device_contract.sha256 -cne $ExpectedDeviceContractSha256 -or [string]$Benchmark.artifacts.ocr_contract.sha256 -cne $ExpectedOcrContractSha256 -or $Benchmark.artifacts.cpu_provider.deps_contains_cpu_onnxruntime -ne $true -or $Benchmark.artifacts.cpu_provider.deps_contains_gpu_onnxruntime -ne $false -or [int]$Benchmark.artifacts.cpu_provider.forbidden_gpu_runtime_file_count -ne 0) { throw 'CPU benchmark asset/provider closure differs from predeclared inputs.' }
    [int]$WarmupExpected = [Math]::Min(8,$testSources.Count)
    Assert-ExactWhiteResultsTree (Join-Path $BenchmarkRoot 'runs\warmup-01\output') $WarmupExpected
    foreach ($runName in @('measured-01','measured-02','measured-03')) {
        Assert-ExactWhiteResultsTree (Join-Path $BenchmarkRoot ("runs\"+$runName+'\output')) $testSources.Count
    }

    $MeasuredResultsRoot = Join-Path $BenchmarkRoot 'runs\measured-01\output'
    $evaluateScript = Join-Path $RepoRoot 'scripts\otherimages-dotnet-evaluate.py'
    $evaluationStage = Invoke-ClosedStage 'full-test-teacher-agreement' $PythonExe @(
        $evaluateScript,'--teacher-manifest',$TeacherManifestPath,'--teacher-contract',$TeacherContractPath,
        '--results',$MeasuredResultsRoot,'--output',$EvaluationRoot,'--split','test',
        '--max-cer','0.05','--min-document-exact','0.90','--min-line-precision','0.90',
        '--min-line-recall','0.90','--max-three-of-three-cer','0.03'
    ) $RepoRoot
    $EvaluationSummaryPath = Join-Path $EvaluationRoot 'summary.json'
    $Evaluation = Read-Json $EvaluationSummaryPath
    if ([string]$Evaluation.kind -cne 'otherimages_dotnet_white_teacher_agreement_v1' -or $Evaluation.accepted -ne $true -or [string]$Evaluation.evaluation_split -cne 'test' -or [double]$Evaluation.coverage.result_coverage -ne 1.0 -or @($Evaluation.failures).Count -ne 0 -or [string]$Evaluation.teacher_agreement.metric_subject -cne 'white_line_student') { throw 'Full frozen-test student teacher-agreement gate failed.' }
    foreach ($stage in @($publishStage,$benchmarkStage,$evaluationStage)) {
        if ([int]$stage.rc -ne 0 -or [int64]$stage.stderr.size_bytes -ne 0 -or $stage.descendant_absence_proven -ne $true) {
            throw "Stage RC/stderr/descendant closure failed before pipeline receipt: $($stage.name)"
        }
    }

    Assert-GitAuthority $GitBinding
    $CodeBindingsAfter = Get-CodeBindings
    foreach ($name in $CodeBindings.Keys) {
        if ([string]$CodeBindings[$name].sha256 -cne [string]$CodeBindingsAfter[$name].sha256 -or [int64]$CodeBindings[$name].size_bytes -ne [int64]$CodeBindingsAfter[$name].size_bytes) { throw "Source code changed during CPU delivery: $name" }
    }
    Assert-Binding $WrapperBinding 'CPU delivery wrapper'
    Assert-Binding $PythonBinding 'fixed Python scorer'
    Assert-Binding $DotnetBinding 'fixed dotnet executable'
    if ((Get-Sha256 $DeviceModel) -cne $ExpectedDeviceModelSha256) { throw 'Status-bar device model changed during CPU delivery.' }
    if ((Get-Sha256 $DeviceContract) -cne $ExpectedDeviceContractSha256) { throw 'Status-bar device contract changed during CPU delivery.' }
    $OcrClosureAfter = Get-DirectoryClosure $OcrBundle 'PP-OCR bundle after validation'
    if ([string]$OcrClosureAfter.closure_sha256 -cne [string]$OcrClosure.closure_sha256 -or [int64]$OcrClosureAfter.size_bytes -ne [int64]$OcrClosure.size_bytes) { throw 'PP-OCR bundle changed during CPU delivery.' }
    Assert-Binding $TeacherContractBinding 'teacher contract'
    foreach ($path in @($StudentModelPath,$StudentCharsetPath,$StudentContractPath)) {
        $before = if ($path -eq $StudentModelPath) { $TrainPipeline.student_bundle.bindings.model } elseif ($path -eq $StudentCharsetPath) { $TrainPipeline.student_bundle.bindings.charset } else { $TrainPipeline.student_bundle.bindings.contract }
        $after = Get-Binding $path
        if ([string]$after.sha256 -cne [string]$before.sha256 -or [int64]$after.size_bytes -ne [int64]$before.size_bytes) { throw "Student bundle artifact changed during CPU delivery: $path" }
    }
    $DeliveryPackageClosure = Get-DirectoryClosure $DeliveryPackageRoot 'validated white CPU delivery package'
    $PackagedOcrClosureAfter = Get-DirectoryClosure $PackagedOcrBundle 'packaged PP-OCR bundle after validation'
    $PackagedStudentClosureAfter = Get-DirectoryClosure $PackagedStudentBundle 'packaged white student bundle after validation'
    if ([string]$PackagedOcrClosureAfter.closure_sha256 -cne [string]$PackagedOcrClosure.closure_sha256 -or [string]$PackagedStudentClosureAfter.closure_sha256 -cne [string]$PackagedStudentClosure.closure_sha256) { throw 'Validated delivery package model closure changed during execution.' }
    if ([int64]$DeliveryPackageClosure.size_bytes -ne [int64]$Benchmark.measured.delivery_package_payload_bytes) { throw 'Validated package byte count differs from benchmark package accounting.' }
    if ((Get-DirectoryIdentity $RunRoot 'CPU delivery run root at closure') -cne $RunRootIdentity -or (Get-DirectoryIdentity (Split-Path -Parent $RunRoot) 'CPU delivery parent at closure') -cne $RunParentIdentity) { throw 'CPU delivery run/parent identity changed.' }

    $ReceiptPath = Join-Path $RunRoot 'pipeline.receipt.json'
    $Receipt = [ordered]@{
        schema_version=1; kind='otherimages_white_cpu_delivery_windows_pipeline_receipt_v1'; status='complete'; accepted=$true
        source=[ordered]@{ repo_root=$RepoRoot; head=$RequiredHead; tree=$RequiredTree; clean=$true; fixed_code=$CodeBindings; wrapper=$WrapperBinding; python=$PythonBinding; dotnet=$DotnetBinding }
        inputs=[ordered]@{
            teacher=[ordered]@{ run_root=$TeacherRunRoot; pipeline_receipt=Get-Binding $TeacherPipelinePath; manifest=Get-Binding $TeacherManifestPath; contract=$TeacherContractBinding; receipt=Get-Binding $TeacherReceiptPath; split='test'; test_records=$testSources.Count }
            training=[ordered]@{ run_root=$TrainRunRoot; pipeline_receipt=Get-Binding $TrainPipelinePath; student_bundle=$StudentBundle; model=Get-Binding $StudentModelPath; charset=Get-Binding $StudentCharsetPath; contract=Get-Binding $StudentContractPath }
            statusbar=[ordered]@{ model=Get-Binding $DeviceModel; contract=Get-Binding $DeviceContract; predeclared=$true }
            ppocr=[ordered]@{ root=$OcrBundle; contract=Get-Binding $OcrContractPath; closure=$OcrClosure; predeclared=$true }
        }
        budgets=[ordered]@{ max_p50_latency_ms=$MaxP50LatencyMilliseconds; max_p95_latency_ms=$MaxP95LatencyMilliseconds; min_throughput_images_per_second=$MinThroughputImagesPerSecond; max_peak_working_set_mib=$MaxPeakWorkingSetMiB; max_peak_private_bytes_mib=$MaxPeakPrivateBytesMiB; max_package_size_mib=$MaxPackageSizeMiB; all_predeclared=$true }
        outputs=[ordered]@{ run_root=$RunRoot; run_identity=$RunRootIdentity; delivery_package=$DeliveryPackageClosure; published_app=$PublishRoot; packaged_statusbar=[ordered]@{model=Get-Binding $PackagedDeviceModel;contract=Get-Binding $PackagedDeviceContract}; packaged_ppocr=$PackagedOcrClosureAfter; packaged_student=$PackagedStudentClosureAfter; frozen_test_input_list=Get-Binding $TestInputList; benchmark=Get-Binding $BenchmarkReportPath; teacher_agreement=Get-Binding $EvaluationSummaryPath }
        stages=[ordered]@{ publish=$publishStage; benchmark=$benchmarkStage; evaluation=$evaluationStage }
        measured=[ordered]@{ p50_latency_ms=$Benchmark.measured.inference_latency_ms.p50; p95_latency_ms=$Benchmark.measured.inference_latency_ms.p95; throughput_images_per_second=$Benchmark.measured.throughput_images_per_second; peak_working_set_bytes=$Benchmark.measured.peak_working_set_bytes.maximum; peak_private_bytes=$Benchmark.measured.peak_private_bytes.maximum; package_payload_bytes=$Benchmark.measured.delivery_package_payload_bytes; teacher_agreement=$Evaluation.teacher_agreement.overall }
        validation=[ordered]@{ fixed_head_tree_clean=$true; strict_cpu_onnxruntime=$true; gpu_runtime_absent=$true; statusbar_and_ppocr_predeclared_and_stable=$true; student_bundle_exact_three_file_closed=$true; final_package_materialized_before_inference=$true; benchmark_executed_only_from_final_package=$true; package_byte_count_matches_benchmark=$true; same_frozen_test_split_for_benchmark_and_scorer=$true; smoke_complete=$true; three_full_test_runs_complete=$true; every_inference_output_exact_flat_tree=$true; result_coverage_100_percent=$true; teacher_agreement_gate_passed=$true; absolute_efficiency_gate_passed=$true; every_stage_rc_zero=$true; every_stage_stderr_zero_bytes=$true; every_stage_descendant_absence_proven=$true; process_tree_cleanup_on_failure=$true; fresh_no_clobber=$true; input_and_code_closure_stable=$true }
        warning='Teacher agreement is Paddle pseudo-label parity and is not independent human ground truth.'
        started_utc=$PipelineStartedUtc.ToString('o'); completed_utc=[DateTime]::UtcNow.ToString('o')
    }
    Write-JsonNew $ReceiptPath $Receipt
    $Readback = Read-Json $ReceiptPath
    if ([string]$Readback.status -cne 'complete' -or $Readback.accepted -ne $true -or $Readback.validation.strict_cpu_onnxruntime -ne $true -or $Readback.validation.teacher_agreement_gate_passed -ne $true -or $Readback.validation.absolute_efficiency_gate_passed -ne $true) { throw 'CPU delivery receipt readback failed.' }
    Write-Host "WHITE_CPU_DELIVERY_OK receipt=$ReceiptPath test_records=$($testSources.Count) p50_ms=$($Benchmark.measured.inference_latency_ms.p50) p95_ms=$($Benchmark.measured.inference_latency_ms.p95) throughput=$($Benchmark.measured.throughput_images_per_second)"
}
catch {
    $Failure = $_
    if ($RunRootOwned -and (Test-Path -LiteralPath $RunRoot -PathType Container)) {
        Assert-NoReparseChain $RunRoot 'owned CPU delivery failure root'
        if ($null -ne $RunRootIdentity -and (Get-DirectoryIdentity $RunRoot 'owned CPU delivery failure root') -cne $RunRootIdentity) { throw 'Refusing failure evidence after RunRoot identity changed.' }
        $failurePath = Join-Path $RunRoot 'pipeline.failure.json'
        if (-not (Test-Path -LiteralPath $failurePath)) {
            Write-JsonNew $failurePath ([ordered]@{ schema_version=1; kind='otherimages_white_cpu_delivery_windows_failure_v1'; status='failed'; message=$_.Exception.Message; exception_type=$_.Exception.GetType().FullName; script_stack_trace=$_.ScriptStackTrace; utc=[DateTime]::UtcNow.ToString('o') })
        }
        Write-Host "WHITE_CPU_DELIVERY_FAILED evidence=$failurePath error=$($_.Exception.Message)"
    }
    throw
}
