param(
    [switch]$Doctor,
    [switch]$CaptureMessages,
    [switch]$RestartProxy,
    [string]$Resume,
    [Parameter(ValueFromRemainingArguments = $true)][string[]]$ClaudeArgs
)

$ErrorActionPreference = 'Stop'
$runtimeRoot = Join-Path $PSScriptRoot 'runtime'
$proxyExecutable = Join-Path $runtimeRoot 'Scripts/headroom.exe'
$modExecutable = Join-Path $runtimeRoot 'Scripts/headroom-mod.exe'
$proxyUrl = 'http://127.0.0.1:18787'
$pidFile = Join-Path $PSScriptRoot 'proxy.pid'
$logRoot = Join-Path $PSScriptRoot 'logs'
if (-not (Test-Path -LiteralPath $proxyExecutable) -or -not (Test-Path -LiteralPath $modExecutable)) {
    throw 'The dedicated Headroom Sidebar runtime is missing. Reinstall the runtime before launching.'
}

function Get-SidebarHealth {
    try {
        $response = Invoke-WebRequest -Uri "$proxyUrl/headroom-mod/v1/health" -NoProxy -MaximumRedirection 0 -TimeoutSec 2
        if ($response.Content.Length -gt 16384) { return $null }
        $health = $response.Content | ConvertFrom-Json
        if ($health.service -eq 'headroom-claude-mod' -and $health.schema_version -eq 1 -and ($health.read_only -eq $true -or $health.session_controls -eq $true)) {
            return $health
        }
    } catch { }
    return $null
}

if ($RestartProxy) {
    $listener = Get-NetTCPConnection -State Listen -LocalPort 18787 -ErrorAction SilentlyContinue
    if ($listener) {
        if (-not (Test-Path -LiteralPath $pidFile)) { throw 'Port 18787 is occupied by a process this launcher does not own.' }
        $recordedPid = [int](Get-Content -LiteralPath $pidFile -Raw)
        $proxyProcess = Get-CimInstance Win32_Process -Filter "ProcessId = $($listener.OwningProcess)"
        if ($proxyProcess.ProcessId -ne $recordedPid -or
            $proxyProcess.CommandLine.IndexOf($proxyExecutable, [StringComparison]::OrdinalIgnoreCase) -lt 0 -or
            $proxyProcess.CommandLine -notmatch '18787' -or $proxyProcess.CommandLine -notmatch 'claude_mod') {
            throw 'The listener no longer matches the dedicated proxy recorded by this launcher.'
        }
        Stop-Process -Id $recordedPid
    }
}

$health = Get-SidebarHealth
if (-not $health) {
    if (Get-NetTCPConnection -State Listen -LocalPort 18787 -ErrorAction SilentlyContinue) {
        throw 'Port 18787 is occupied without a compatible sidebar companion.'
    }
    New-Item -ItemType Directory -Path $logRoot -Force | Out-Null
    $proxyArguments = @('proxy', '--host', '127.0.0.1', '--port', '18787', '--proxy-extension', 'claude_mod', '--mode', 'token')
    if ($CaptureMessages) { $proxyArguments += '--log-messages' }
    # Detach the proxy from the invoking console and close inherited handles.
    $spawnCode = 'import subprocess,sys; out=open(sys.argv[2],"ab"); err=open(sys.argv[3],"ab"); p=subprocess.Popen([sys.argv[1]]+sys.argv[4:],stdin=subprocess.DEVNULL,stdout=out,stderr=err,close_fds=True,creationflags=subprocess.DETACHED_PROCESS|subprocess.CREATE_NEW_PROCESS_GROUP); print(p.pid)'
    $spawnedPid = [int](& (Join-Path $runtimeRoot 'Scripts/python.exe') -c $spawnCode $proxyExecutable (Join-Path $logRoot 'proxy-out.log') (Join-Path $logRoot 'proxy-err.log') @proxyArguments)
    if ($LASTEXITCODE -ne 0) { throw 'Could not launch the dedicated proxy.' }
    $launchedProxy = Get-Process -Id $spawnedPid -ErrorAction SilentlyContinue
    try {
    $deadline = [DateTime]::UtcNow.AddSeconds(90)
    do {
        Start-Sleep -Milliseconds 500
        $health = Get-SidebarHealth
        if ($launchedProxy.HasExited -and -not $health) {
            throw "The dedicated proxy exited. See $(Join-Path $logRoot 'proxy-err.log')."
        }
    } until ($health -or [DateTime]::UtcNow -gt $deadline)
    if (-not $health) { throw "The proxy did not become ready. See $logRoot." }
    $listener = Get-NetTCPConnection -State Listen -LocalPort 18787
    $owner = Get-CimInstance Win32_Process -Filter "ProcessId = $($listener.OwningProcess)"
    if ($owner.CommandLine.IndexOf($proxyExecutable, [StringComparison]::OrdinalIgnoreCase) -lt 0 -or
        $owner.CommandLine -notmatch '18787' -or $owner.CommandLine -notmatch 'claude_mod') {
        throw 'The ready listener does not match the dedicated proxy.'
    }
    $listener.OwningProcess | Set-Content -LiteralPath $pidFile
    } catch {
        # Only stop the verified process we just created, and its verified children.
        $spawned = Get-CimInstance Win32_Process -Filter "ProcessId = $spawnedPid"
        if ($spawned -and $spawned.CommandLine.IndexOf($proxyExecutable, [StringComparison]::OrdinalIgnoreCase) -ge 0 -and
            $spawned.CommandLine -match '18787' -and $spawned.CommandLine -match 'claude_mod') {
            $children = Get-CimInstance Win32_Process -Filter "ParentProcessId = $spawnedPid"
            foreach ($child in $children) {
                if ($child.CommandLine.IndexOf($proxyExecutable, [StringComparison]::OrdinalIgnoreCase) -ge 0 -and $child.CommandLine -match 'claude_mod') {
                    Stop-Process -Id $child.ProcessId -ErrorAction SilentlyContinue
                }
            }
            Stop-Process -Id $spawnedPid -ErrorAction SilentlyContinue
        }
        throw
    }
}

if ($CaptureMessages -and -not $health.log_full_messages) {
    throw 'The proxy is running with capture off. Use -CaptureMessages -RestartProxy to explicitly restart the dedicated proxy with inspection enabled.'
}

if ($Doctor) {
    & $modExecutable doctor --proxy-url $proxyUrl
    exit $LASTEXITCODE
}
$launchArguments = @('run', '--proxy-url', $proxyUrl)
if ($Resume) { $launchArguments += @('--resume', $Resume) }
if ($ClaudeArgs) { $launchArguments += '--'; $launchArguments += $ClaudeArgs }
& $modExecutable @launchArguments
exit $LASTEXITCODE
