$ErrorActionPreference = 'Stop'

$ImageDefault = 'ghcr.io/chopratejas/headroom:latest'
$InstallImage = if ($env:HEADROOM_DOCKER_IMAGE) { $env:HEADROOM_DOCKER_IMAGE } else { $ImageDefault }
$InstallDir = Join-Path $HOME '.local\bin'
if (-not (Test-Path (Join-Path $HOME '.local'))) {
    $InstallDir = Join-Path $HOME 'bin'
}

function Write-Info {
    param([string]$Message)
    Write-Host "==> $Message"
}

function Require-Command {
    param([string]$Name)
    if (-not (Get-Command $Name -ErrorAction SilentlyContinue)) {
        throw "Missing required command: $Name"
    }
}

function Ensure-PathEntry {
    param([string]$PathEntry)

    $currentPath = [Environment]::GetEnvironmentVariable('Path', 'User')
    $parts = @()
    if ($currentPath) {
        $parts = $currentPath -split ';' | Where-Object { $_ }
    }
    if ($parts -notcontains $PathEntry) {
        $newPath = @($PathEntry) + $parts
        [Environment]::SetEnvironmentVariable('Path', ($newPath -join ';'), 'User')
    }
}

function Ensure-ProfileBlock {
    param([string]$PathEntry)

    $markerStart = '# >>> headroom docker-native >>>'
    $markerEnd = '# <<< headroom docker-native <<<'
    $escapedPathEntry = $PathEntry.Replace("'", "''")
    $block = @"
$markerStart
if (-not ((`$env:Path -split ';') -contains '$escapedPathEntry')) {
    `$env:Path = '$escapedPathEntry;' + `$env:Path
}
$markerEnd
"@

    $profileDir = Split-Path -Parent $PROFILE
    if (-not (Test-Path $profileDir)) {
        New-Item -ItemType Directory -Force -Path $profileDir | Out-Null
    }
    if (-not (Test-Path $PROFILE)) {
        New-Item -ItemType File -Force -Path $PROFILE | Out-Null
    }

    $existing = Get-Content -Raw -Path $PROFILE
    if ($existing -notmatch [regex]::Escape($markerStart)) {
        Add-Content -Path $PROFILE -Value "`n$block"
    }
}

function Write-Wrapper {
    param([string]$TargetDir)

    $wrapperPath = Join-Path $TargetDir 'headroom.ps1'
    $cmdPath = Join-Path $TargetDir 'headroom.cmd'
    $resolvedInstallImage = $InstallImage.Replace("'", "''")

    $wrapper = @'
$ErrorActionPreference = 'Stop'

$HeadroomImage = if ($env:HEADROOM_DOCKER_IMAGE) { $env:HEADROOM_DOCKER_IMAGE } else { '__HEADROOM_INSTALL_IMAGE__' }
$ContainerHome = if ($env:HEADROOM_CONTAINER_HOME) { $env:HEADROOM_CONTAINER_HOME } else { '/tmp/headroom-home' }
$HostHome = $HOME

function Fail {
    param([string]$Message)
    throw $Message
}

function Require-Command {
    param([string]$Name)
    if (-not (Get-Command $Name -ErrorAction SilentlyContinue)) {
        Fail "Missing required command: $Name"
    }
}

function Get-RtkTarget {
    $arch = if ($env:PROCESSOR_ARCHITECTURE -match 'ARM64') { 'aarch64' } else { 'x86_64' }
    return "${arch}-pc-windows-msvc"
}

function Ensure-HostDirs {
    foreach ($dir in @(
        (Join-Path $HostHome '.headroom'),
        (Join-Path $HostHome '.claude'),
        (Join-Path $HostHome '.codex'),
        (Join-Path $HostHome '.gemini')
    )) {
        if (-not (Test-Path $dir)) {
            New-Item -ItemType Directory -Force -Path $dir | Out-Null
        }
    }
}

function Get-PassthroughEnvArgs {
    $args = New-Object System.Collections.Generic.List[string]
    $prefixes = @(
        'HEADROOM_','ANTHROPIC_','OPENAI_','GEMINI_','AWS_','AZURE_','VERTEX_',
        'GOOGLE_','GOOGLE_CLOUD_','MISTRAL_','GROQ_','OPENROUTER_','XAI_',
        'TOGETHER_','COHERE_','OLLAMA_','LITELLM_','OTEL_','SUPABASE_',
        'QDRANT_','NEO4J_','LANGSMITH_'
    )

    foreach ($item in Get-ChildItem Env:) {
        foreach ($prefix in $prefixes) {
            if ($item.Name.StartsWith($prefix, [System.StringComparison]::OrdinalIgnoreCase)) {
                $args.Add('--env')
                $args.Add($item.Name)
                break
            }
        }
    }

    return ,$args.ToArray()
}

function Get-SharedDockerArgs {
    Ensure-HostDirs
    $args = New-Object System.Collections.Generic.List[string]
    $args.Add('--workdir')
    $args.Add('/workspace')
    $args.Add('--env')
    $args.Add("HOME=$ContainerHome")
    $args.Add('--env')
    $args.Add('PYTHONUNBUFFERED=1')
    # Canonical Headroom filesystem contract (issue #175).
    $args.Add('--env')
    $args.Add("HEADROOM_WORKSPACE_DIR=$ContainerHome/.headroom")
    $args.Add('--env')
    $args.Add("HEADROOM_CONFIG_DIR=$ContainerHome/.headroom/config")
    $args.Add('--volume')
    $args.Add("${PWD}:/workspace")
    $args.Add('--volume')
    $args.Add((Join-Path $HostHome '.headroom') + ":$ContainerHome/.headroom")
    $args.Add('--volume')
    $args.Add((Join-Path $HostHome '.claude') + ":$ContainerHome/.claude")
    $args.Add('--volume')
    $args.Add((Join-Path $HostHome '.codex') + ":$ContainerHome/.codex")
    $args.Add('--volume')
    $args.Add((Join-Path $HostHome '.gemini') + ":$ContainerHome/.gemini")

    foreach ($entry in (Get-PassthroughEnvArgs)) {
        $args.Add($entry)
    }

    return ,$args.ToArray()
}

function Add-TtyArgs {
    param($ArgsList)

    if (-not [Console]::IsInputRedirected -and -not [Console]::IsOutputRedirected) {
        $ArgsList.Add('-it')
        return
    }
    if (-not [Console]::IsInputRedirected) {
        $ArgsList.Add('-i')
    }
    if (-not [Console]::IsOutputRedirected) {
        $ArgsList.Add('-t')
    }
}

function Invoke-HeadroomDocker {
    param([string[]]$Arguments)

    $dockerArgs = New-Object System.Collections.Generic.List[string]
    $dockerArgs.AddRange([string[]]@('run','--rm'))
    Add-TtyArgs -ArgsList $dockerArgs
    $dockerArgs.AddRange((Get-SharedDockerArgs))
    $dockerArgs.Add('--entrypoint')
    $dockerArgs.Add('headroom')
    $dockerArgs.Add($HeadroomImage)
    foreach ($arg in $Arguments) {
        $dockerArgs.Add($arg)
    }

    & docker @dockerArgs
    if ($LASTEXITCODE -ne 0) {
        exit $LASTEXITCODE
    }
}

function Wait-Proxy {
    param(
        [string]$ContainerName,
        [int]$Port
    )

    for ($attempt = 0; $attempt -lt 45; $attempt++) {
        try {
            Invoke-WebRequest -UseBasicParsing -Uri "http://127.0.0.1:$Port/readyz" | Out-Null
            return
        } catch {
            $running = docker ps --format '{{.Names}}'
            if ($running -notcontains $ContainerName) {
                break
            }
            Start-Sleep -Seconds 1
        }
    }

    docker logs $ContainerName | Write-Error
    throw "Headroom proxy failed to start on port $Port"
}

function Start-ProxyContainer {
    param(
        [int]$Port,
        [string[]]$ProxyArgs
    )

    $containerName = "headroom-proxy-$Port-$PID"
    $dockerArgs = New-Object System.Collections.Generic.List[string]
    $dockerArgs.AddRange([string[]]@('run','-d','--rm','--name',$containerName,'-p',"$Port`:$Port"))
    $dockerArgs.AddRange((Get-SharedDockerArgs))
    $dockerArgs.Add($HeadroomImage)
    $dockerArgs.Add('--host')
    $dockerArgs.Add('0.0.0.0')
    $dockerArgs.Add('--port')
    $dockerArgs.Add("$Port")
    foreach ($arg in $ProxyArgs) {
        $dockerArgs.Add($arg)
    }

    & docker @dockerArgs | Out-Null
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to start Headroom proxy container"
    }

    Wait-Proxy -ContainerName $containerName -Port $Port
    return $containerName
}

function Stop-ProxyContainer {
    param([string]$ContainerName)
    if ($ContainerName) {
        docker stop $ContainerName | Out-Null
    }
}

function Get-PersistentProfileRoot {
    param([string]$Profile)
    Assert-ValidProfileName -Profile $Profile
    return Join-Path (Join-Path $HostHome '.headroom\deploy') $Profile
}

function Get-PersistentStatePath {
    param([string]$Profile)
    return Join-Path (Get-PersistentProfileRoot -Profile $Profile) 'docker-native.json'
}

function Get-PersistentManifestPath {
    param([string]$Profile)
    return Join-Path (Get-PersistentProfileRoot -Profile $Profile) 'manifest.json'
}

function Get-PersistentContainerName {
    param([string]$Profile)
    return "headroom-$Profile"
}

function Assert-ValidProfileName {
    param([string]$Profile)
    if ($Profile -notmatch '^[A-Za-z0-9._-]+$' -or $Profile -in @('.', '..')) {
        Fail "Invalid profile name '$Profile'"
    }
}

function Parse-PortValue {
    param([string]$Value)

    $parsed = 0
    if (-not [int]::TryParse($Value, [ref]$parsed) -or $parsed -lt 1 -or $parsed -gt 65535) {
        Fail "Invalid port '$Value'"
    }
    return $parsed
}

function Parse-PositiveIntegerValue {
    param([string]$Value)

    $parsed = 0
    if (-not [int]::TryParse($Value, [ref]$parsed) -or $parsed -lt 1) {
        Fail "Invalid value '$Value'"
    }
    return $parsed
}

function Require-OptionValue {
    param(
        [string[]]$Arguments,
        [int]$Index,
        [string]$Option
    )

    if ($Index + 1 -ge $Arguments.Count) {
        Fail "Option $Option requires a value"
    }
}

function Write-Utf8NoBomFile {
    param(
        [string]$Path,
        [string]$Content
    )

    [System.IO.File]::WriteAllText($Path, $Content, [System.Text.UTF8Encoding]::new($false))
}

function Get-PersistentDockerArgs {
    Ensure-HostDirs
    $args = New-Object System.Collections.Generic.List[string]
    $args.Add('--workdir')
    $args.Add($ContainerHome)
    $args.Add('--env')
    $args.Add("HOME=$ContainerHome")
    $args.Add('--env')
    $args.Add('PYTHONUNBUFFERED=1')
    # Canonical Headroom filesystem contract (issue #175).
    $args.Add('--env')
    $args.Add("HEADROOM_WORKSPACE_DIR=$ContainerHome/.headroom")
    $args.Add('--env')
    $args.Add("HEADROOM_CONFIG_DIR=$ContainerHome/.headroom/config")
    $args.Add('--volume')
    $args.Add((Join-Path $HostHome '.headroom') + ":$ContainerHome/.headroom")
    $args.Add('--volume')
    $args.Add((Join-Path $HostHome '.claude') + ":$ContainerHome/.claude")
    $args.Add('--volume')
    $args.Add((Join-Path $HostHome '.codex') + ":$ContainerHome/.codex")
    $args.Add('--volume')
    $args.Add((Join-Path $HostHome '.gemini') + ":$ContainerHome/.gemini")

    foreach ($entry in (Get-PassthroughEnvArgs)) {
        $args.Add($entry)
    }

    return ,$args.ToArray()
}

function Get-ManifestProxyArgs {
    param(
        [int]$Port,
        [string]$Backend,
        [string]$AnyllmProvider,
        [string]$Region,
        [string]$Mode,
        [bool]$Memory,
        [bool]$TelemetryEnabled
    )

    $args = New-Object System.Collections.Generic.List[string]
    $args.AddRange([string[]]@('--host','127.0.0.1','--port',"$Port",'--mode',$Mode,'--backend',$Backend))
    if (-not $TelemetryEnabled) {
        $args.Add('--no-telemetry')
    }
    if ($Memory) {
        $args.AddRange([string[]]@('--memory','--memory-db-path',"$ContainerHome/.headroom/memory.db"))
    }
    if ($AnyllmProvider) {
        $args.AddRange([string[]]@('--anyllm-provider', $AnyllmProvider))
    }
    if ($Region) {
        $args.AddRange([string[]]@('--region', $Region))
    }

    return ,$args.ToArray()
}

function Write-PersistentState {
    param(
        [string]$Profile,
        [string]$Image,
        [int]$Port,
        [string]$Backend,
        [string]$AnyllmProvider,
        [string]$Region,
        [string]$Mode,
        [bool]$Memory,
        [bool]$TelemetryEnabled
    )

    $root = Get-PersistentProfileRoot -Profile $Profile
    New-Item -ItemType Directory -Force -Path $root | Out-Null
    $state = [ordered]@{
        profile = $Profile
        image = $Image
        port = $Port
        backend = $Backend
        anyllm_provider = $AnyllmProvider
        region = $Region
        proxy_mode = $Mode
        memory_enabled = $Memory
        telemetry_enabled = $TelemetryEnabled
        container_name = Get-PersistentContainerName -Profile $Profile
        health_url = "http://127.0.0.1:$Port/readyz"
    }
    Write-Utf8NoBomFile -Path (Get-PersistentStatePath -Profile $Profile) -Content ($state | ConvertTo-Json -Depth 4)
}

function Write-PersistentManifest {
    param(
        [string]$Profile,
        [string]$Image,
        [int]$Port,
        [string]$Backend,
        [string]$AnyllmProvider,
        [string]$Region,
        [string]$Mode,
        [bool]$Memory,
        [bool]$TelemetryEnabled,
        [string[]]$ProxyArgs
    )

    $root = Get-PersistentProfileRoot -Profile $Profile
    New-Item -ItemType Directory -Force -Path $root | Out-Null

    $baseEnv = [ordered]@{
        HEADROOM_PORT = "$Port"
        HEADROOM_HOST = '127.0.0.1'
        HEADROOM_MODE = $Mode
        HEADROOM_BACKEND = $Backend
    }

    $manifest = [ordered]@{
        profile = $Profile
        preset = 'persistent-docker'
        runtime_kind = 'docker'
        supervisor_kind = 'none'
        scope = 'user'
        provider_mode = 'manual'
        targets = @()
        port = $Port
        host = '127.0.0.1'
        backend = $Backend
        anyllm_provider = if ($AnyllmProvider) { $AnyllmProvider } else { $null }
        region = if ($Region) { $Region } else { $null }
        proxy_mode = $Mode
        memory_enabled = $Memory
        memory_db_path = "$ContainerHome/.headroom/memory.db"
        telemetry_enabled = $TelemetryEnabled
        image = $Image
        service_name = "headroom-$Profile"
        container_name = Get-PersistentContainerName -Profile $Profile
        health_url = "http://127.0.0.1:$Port/readyz"
        base_env = $baseEnv
        tool_envs = @{}
        proxy_args = $ProxyArgs
        mutations = @()
        artifacts = @()
    }

    Write-Utf8NoBomFile -Path (Get-PersistentManifestPath -Profile $Profile) -Content ($manifest | ConvertTo-Json -Depth 8)
}

function Read-PersistentState {
    param([string]$Profile)

    Assert-ValidProfileName -Profile $Profile
    $statePath = Get-PersistentStatePath -Profile $Profile
    if (-not (Test-Path $statePath)) {
        Fail "No docker-native persistent deployment profile named '$Profile'"
    }
    return Get-Content -Raw -Path $statePath | ConvertFrom-Json
}

function Start-PersistentDockerInstall {
    param(
        [string]$Profile,
        [string]$Image,
        [int]$Port,
        [string]$Backend,
        [string]$AnyllmProvider,
        [string]$Region,
        [string]$Mode,
        [bool]$Memory,
        [bool]$TelemetryEnabled
    )

    Assert-ValidProfileName -Profile $Profile
    $containerName = Get-PersistentContainerName -Profile $Profile
    $proxyArgs = Get-ManifestProxyArgs -Port $Port -Backend $Backend -AnyllmProvider $AnyllmProvider -Region $Region -Mode $Mode -Memory $Memory -TelemetryEnabled $TelemetryEnabled

    docker rm -f $containerName | Out-Null 2>$null

    $dockerArgs = New-Object System.Collections.Generic.List[string]
    $dockerArgs.AddRange([string[]]@('run','-d','--restart','unless-stopped','--name',$containerName,'-p',"$Port`:$Port"))
    $dockerArgs.AddRange((Get-PersistentDockerArgs))
    $dockerArgs.AddRange([string[]]@(
        '--env',"HEADROOM_DEPLOYMENT_PROFILE=$Profile",
        '--env','HEADROOM_DEPLOYMENT_PRESET=persistent-docker',
        '--env','HEADROOM_DEPLOYMENT_RUNTIME=docker',
        '--env','HEADROOM_DEPLOYMENT_SUPERVISOR=none',
        '--env','HEADROOM_DEPLOYMENT_SCOPE=user'
    ))
    $dockerArgs.Add($Image)
    $dockerArgs.Add('--host')
    $dockerArgs.Add('0.0.0.0')
    for ($i = 2; $i -lt $proxyArgs.Count; $i++) {
        $dockerArgs.Add($proxyArgs[$i])
    }

    & docker @dockerArgs | Out-Null
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to start docker-native persistent deployment"
    }

    try {
        Wait-Proxy -ContainerName $containerName -Port $Port
    } catch {
        docker rm -f $containerName | Out-Null 2>$null
        throw
    }
    Write-PersistentState -Profile $Profile -Image $Image -Port $Port -Backend $Backend -AnyllmProvider $AnyllmProvider -Region $Region -Mode $Mode -Memory $Memory -TelemetryEnabled $TelemetryEnabled
    Write-PersistentManifest -Profile $Profile -Image $Image -Port $Port -Backend $Backend -AnyllmProvider $AnyllmProvider -Region $Region -Mode $Mode -Memory $Memory -TelemetryEnabled $TelemetryEnabled -ProxyArgs $proxyArgs
}

function Stop-PersistentDockerInstall {
    param([string]$Profile)

    $state = Read-PersistentState -Profile $Profile
    docker stop $state.container_name | Out-Null 2>$null
    docker rm -f $state.container_name | Out-Null 2>$null
}

function Remove-PersistentDockerInstall {
    param([string]$Profile)

    $state = Read-PersistentState -Profile $Profile
    docker stop $state.container_name | Out-Null 2>$null
    docker rm -f $state.container_name | Out-Null 2>$null
    $root = Get-PersistentProfileRoot -Profile $Profile
    if (Test-Path $root) {
        Remove-Item -Recurse -Force -Path $root
    }
}

function Show-PersistentDockerInstallStatus {
    param([string]$Profile)

    $state = Read-PersistentState -Profile $Profile
    $status = 'stopped'
    $ready = 'no'
    $running = docker ps --format '{{.Names}}'
    if ($running -contains $state.container_name) {
        $status = 'running'
        try {
            Invoke-WebRequest -UseBasicParsing -Uri $state.health_url | Out-Null
            $ready = 'yes'
        } catch {
            $ready = 'no'
        }
    }

    Write-Host "Profile:    $($state.profile)"
    Write-Host 'Preset:     persistent-docker'
    Write-Host 'Runtime:    docker'
    Write-Host 'Supervisor: none'
    Write-Host "Port:       $($state.port)"
    Write-Host "Status:     $status"
    Write-Host "Ready:      $ready"
    Write-Host "Health URL: $($state.health_url)"
}

function Show-InstallHelp {
    $lines = @(
        'Usage: headroom install [OPTIONS] COMMAND [ARGS]...',
        '',
        '  Manage persistent Docker-native Headroom deployments.',
        '',
        '  The Docker-native wrapper currently supports the persistent-docker preset only.',
        '  Use the Python-native `headroom install` command for persistent-service and',
        '  persistent-task installs, or when you need provider/user/system config mutation.',
        '',
        'Options:',
        '  -?, --help  Show this message and exit.',
        '',
        'Commands:',
        '  apply    Install a persistent Docker deployment.',
        '  remove   Remove a persistent Docker deployment.',
        '  restart  Restart a persistent Docker deployment.',
        '  start    Start a persistent Docker deployment.',
        '  status   Show persistent Docker deployment status.',
        '  stop     Stop a persistent Docker deployment.'
    )
    Write-Host ($lines -join [Environment]::NewLine)
}

function Show-InstallApplyHelp {
    $lines = @(
        'Usage: headroom install apply [OPTIONS]',
        '',
        '  Install a persistent Docker deployment.',
        '',
        'Options:',
        '  --preset [persistent-docker]  Docker-native wrapper supports persistent-docker only.',
        '  --runtime [docker]            Docker-native wrapper supports runtime=docker only.',
        '  --profile TEXT                Deployment profile name.  [default: default]',
        '  -p, --port INTEGER            Persistent proxy port.  [default: 8787]',
        '  --backend TEXT                Proxy backend.  [default: anthropic]',
        '  --anyllm-provider TEXT        Provider for any-llm backends.',
        '  --region TEXT                 Cloud region for Bedrock / Vertex style backends.',
        '  --mode TEXT                   Proxy optimization mode.  [default: token]',
        '  --memory                      Enable persistent memory in the runtime.',
        '  --no-telemetry                Disable anonymous telemetry in the runtime.',
        '  --image TEXT                  Docker image to use.  [default: HEADROOM_DOCKER_IMAGE or ghcr.io/chopratejas/headroom:latest]',
        '  -?, --help                    Show this message and exit.'
    )
    Write-Host ($lines -join [Environment]::NewLine)
}

function Show-WrapHelp {
    $lines = @(
        'Usage: headroom wrap <COMMAND> [OPTIONS] [-- ARGS...]',
        '',
        '  Launch supported host tools through a Docker-native Headroom proxy.',
        '',
        'Supported commands:',
        '  claude',
        '  codex',
        '  aider',
        '  cursor',
        '  openclaw',
        '',
        'Notes:',
        '  - GitHub Copilot CLI wrapping is not supported by the Docker-native wrapper.',
        '  - Use the Python-native CLI for unsupported wrap targets.'
    )
    Write-Host ($lines -join [Environment]::NewLine)
}

function Parse-InstallApplyArgs {
    param([string[]]$Arguments)

    $profile = 'default'
    $port = 8787
    $backend = 'anthropic'
    $anyllmProvider = $null
    $region = $null
    $mode = 'token'
    $memory = $false
    $telemetryEnabled = $true
    $image = $HeadroomImage

    $i = 0
    while ($i -lt $Arguments.Count) {
        $arg = $Arguments[$i]
        switch -Regex ($arg) {
            '^--preset$' {
                Require-OptionValue -Arguments $Arguments -Index $i -Option '--preset'
                if ($Arguments[$i + 1] -ne 'persistent-docker') { Fail 'Docker-native wrapper supports only --preset persistent-docker' }
                $i += 2
                continue
            }
            '^--preset=' {
                if (($arg -replace '^--preset=', '') -ne 'persistent-docker') { Fail 'Docker-native wrapper supports only --preset persistent-docker' }
                $i += 1
                continue
            }
            '^--runtime$' {
                Require-OptionValue -Arguments $Arguments -Index $i -Option '--runtime'
                if ($Arguments[$i + 1] -ne 'docker') { Fail 'Docker-native wrapper supports only --runtime docker' }
                $i += 2
                continue
            }
            '^--runtime=' {
                if (($arg -replace '^--runtime=', '') -ne 'docker') { Fail 'Docker-native wrapper supports only --runtime docker' }
                $i += 1
                continue
            }
            '^(--scope|--providers|--target)$' { Fail 'Docker-native wrapper install does not support provider/user/system mutation flags; use the Python-native CLI for those flows' }
            '^(--scope=|--providers=|--target=)' { Fail 'Docker-native wrapper install does not support provider/user/system mutation flags; use the Python-native CLI for those flows' }
            '^--profile$' {
                Require-OptionValue -Arguments $Arguments -Index $i -Option '--profile'
                $profile = $Arguments[$i + 1]
                $i += 2
                continue
            }
            '^--profile=' {
                $profile = $arg -replace '^--profile=', ''
                $i += 1
                continue
            }
            '^(--port|-p)$' {
                Require-OptionValue -Arguments $Arguments -Index $i -Option $arg
                $port = Parse-PortValue -Value $Arguments[$i + 1]
                $i += 2
                continue
            }
            '^(--port=|-p=)' {
                $port = Parse-PortValue -Value ($arg -replace '^(--port=|-p=)', '')
                $i += 1
                continue
            }
            '^--backend$' {
                Require-OptionValue -Arguments $Arguments -Index $i -Option '--backend'
                $backend = $Arguments[$i + 1]
                $i += 2
                continue
            }
            '^--backend=' {
                $backend = $arg -replace '^--backend=', ''
                $i += 1
                continue
            }
            '^--anyllm-provider$' {
                Require-OptionValue -Arguments $Arguments -Index $i -Option '--anyllm-provider'
                $anyllmProvider = $Arguments[$i + 1]
                $i += 2
                continue
            }
            '^--anyllm-provider=' {
                $anyllmProvider = $arg -replace '^--anyllm-provider=', ''
                $i += 1
                continue
            }
            '^--region$' {
                Require-OptionValue -Arguments $Arguments -Index $i -Option '--region'
                $region = $Arguments[$i + 1]
                $i += 2
                continue
            }
            '^--region=' {
                $region = $arg -replace '^--region=', ''
                $i += 1
                continue
            }
            '^--mode$' {
                Require-OptionValue -Arguments $Arguments -Index $i -Option '--mode'
                $mode = $Arguments[$i + 1]
                $i += 2
                continue
            }
            '^--mode=' {
                $mode = $arg -replace '^--mode=', ''
                $i += 1
                continue
            }
            '^--memory$' {
                $memory = $true
                $i += 1
                continue
            }
            '^--no-telemetry$' {
                $telemetryEnabled = $false
                $i += 1
                continue
            }
            '^--image$' {
                Require-OptionValue -Arguments $Arguments -Index $i -Option '--image'
                $image = $Arguments[$i + 1]
                $i += 2
                continue
            }
            '^--image=' {
                $image = $arg -replace '^--image=', ''
                $i += 1
                continue
            }
            '^(--help|-\?)$' {
                Show-InstallApplyHelp
                exit 0
            }
            default {
                Fail "Unsupported option for 'headroom install apply': $arg"
            }
        }
    }

    return [pscustomobject]@{
        Profile = $profile
        Port = $port
        Backend = $backend
        AnyllmProvider = $anyllmProvider
        Region = $region
        Mode = $mode
        Memory = $memory
        TelemetryEnabled = $telemetryEnabled
        Image = $image
    }
}

function Parse-InstallProfileArgs {
    param([string[]]$Arguments)

    $profile = 'default'
    $i = 0
    while ($i -lt $Arguments.Count) {
        $arg = $Arguments[$i]
        switch -Regex ($arg) {
            '^--profile$' {
                Require-OptionValue -Arguments $Arguments -Index $i -Option '--profile'
                $profile = $Arguments[$i + 1]
                $i += 2
                continue
            }
            '^--profile=' {
                $profile = $arg -replace '^--profile=', ''
                $i += 1
                continue
            }
            '^(--help|-\?)$' {
                Show-InstallHelp
                exit 0
            }
            default {
                Fail "Unsupported option for 'headroom install': $arg"
            }
        }
    }

    return $profile
}

function Invoke-ClaudeRtkInit {
    $rtkPath = Join-Path $HostHome '.headroom\bin\rtk.exe'
    if (-not (Test-Path $rtkPath)) {
        Write-Warning "rtk was not installed at $rtkPath; Claude hooks were not registered"
        return
    }

    try {
        & $rtkPath init --global --auto-patch | Out-Null
    } catch {
        Write-Warning "Failed to register Claude hooks with rtk; continuing without hook registration"
    }
}

function Get-ContextTool {
    $value = $env:HEADROOM_CONTEXT_TOOL
    if ([string]::IsNullOrWhiteSpace($value)) {
        return 'rtk'
    }

    $value = $value.Trim().ToLowerInvariant().Replace('_', '-')
    if ($value -eq 'leanctx') {
        return 'lean-ctx'
    }
    if ($value -ne 'rtk' -and $value -ne 'lean-ctx') {
        Fail 'HEADROOM_CONTEXT_TOOL must be one of: lean-ctx, rtk'
    }
    return $value
}

function Invoke-LeanCtxInit {
    param([string]$Agent)

    $cmd = Get-Command lean-ctx -ErrorAction SilentlyContinue
    if (-not $cmd) {
        Write-Warning "lean-ctx is not installed on PATH; $Agent lean-ctx setup was skipped"
        return
    }

    try {
        & $cmd.Source init --agent $Agent | Out-Null
    } catch {
        Write-Warning "Failed to initialize lean-ctx for $Agent; continuing without lean-ctx setup"
    }
}

function Invoke-WithTemporaryEnv {
    param(
        [hashtable]$Environment,
        [string]$Command,
        [string[]]$Arguments
    )

    $previous = @{}
    foreach ($pair in $Environment.GetEnumerator()) {
        $previous[$pair.Key] = [Environment]::GetEnvironmentVariable($pair.Key, 'Process')
        [Environment]::SetEnvironmentVariable($pair.Key, $pair.Value, 'Process')
    }

    try {
        & $Command @Arguments
        return $LASTEXITCODE
    } finally {
        foreach ($pair in $Environment.GetEnumerator()) {
            [Environment]::SetEnvironmentVariable($pair.Key, $previous[$pair.Key], 'Process')
        }
    }
}

function Test-HelpFlag {
    param([string[]]$Arguments)

    foreach ($arg in $Arguments) {
        if ($arg -eq '--') {
            break
        }
        if ($arg -eq '--help' -or $arg -eq '-?') {
            return $true
        }
    }

    return $false
}

function Parse-OpenClawWrapArgs {
    param([string[]]$Arguments)

    $gatewayProviderIds = New-Object System.Collections.Generic.List[string]
    $pluginPath = $null
    $pluginSpec = 'headroom-ai/openclaw'
    $skipBuild = $false
    $copy = $false
    $proxyPort = 8787
    $startupTimeoutMs = 20000
    $pythonPath = $null
    $noAutoStart = $false
    $noRestart = $false
    $verbose = $false

    $i = 0
    while ($i -lt $Arguments.Count) {
        $arg = $Arguments[$i]
        switch -Regex ($arg) {
            '^--plugin-path$' {
                Require-OptionValue -Arguments $Arguments -Index $i -Option $arg
                $pluginPath = $Arguments[$i + 1]
                $i += 2
                continue
            }
            '^--plugin-path=' {
                $pluginPath = $arg -replace '^--plugin-path=', ''
                $i += 1
                continue
            }
            '^--plugin-spec$' {
                Require-OptionValue -Arguments $Arguments -Index $i -Option $arg
                $pluginSpec = $Arguments[$i + 1]
                $i += 2
                continue
            }
            '^--plugin-spec=' {
                $pluginSpec = $arg -replace '^--plugin-spec=', ''
                $i += 1
                continue
            }
            '^--skip-build$' {
                $skipBuild = $true
                $i += 1
                continue
            }
            '^--copy$' {
                $copy = $true
                $i += 1
                continue
            }
            '^--proxy-port$' {
                Require-OptionValue -Arguments $Arguments -Index $i -Option $arg
                $proxyPort = Parse-PortValue -Value $Arguments[$i + 1]
                $i += 2
                continue
            }
            '^--proxy-port=' {
                $proxyPort = Parse-PortValue -Value ($arg -replace '^--proxy-port=', '')
                $i += 1
                continue
            }
            '^--startup-timeout-ms$' {
                Require-OptionValue -Arguments $Arguments -Index $i -Option $arg
                $startupTimeoutMs = Parse-PositiveIntegerValue -Value $Arguments[$i + 1]
                $i += 2
                continue
            }
            '^--startup-timeout-ms=' {
                $startupTimeoutMs = Parse-PositiveIntegerValue -Value ($arg -replace '^--startup-timeout-ms=', '')
                $i += 1
                continue
            }
            '^--gateway-provider-id$' {
                Require-OptionValue -Arguments $Arguments -Index $i -Option $arg
                $gatewayProviderIds.Add($Arguments[$i + 1])
                $i += 2
                continue
            }
            '^--gateway-provider-id=' {
                $gatewayProviderIds.Add($arg -replace '^--gateway-provider-id=', '')
                $i += 1
                continue
            }
            '^--python-path$' {
                Require-OptionValue -Arguments $Arguments -Index $i -Option $arg
                $pythonPath = $Arguments[$i + 1]
                $i += 2
                continue
            }
            '^--python-path=' {
                $pythonPath = $arg -replace '^--python-path=', ''
                $i += 1
                continue
            }
            '^--no-auto-start$' {
                $noAutoStart = $true
                $i += 1
                continue
            }
            '^--no-restart$' {
                $noRestart = $true
                $i += 1
                continue
            }
            '^--verbose$|^-v$' {
                $verbose = $true
                $i += 1
                continue
            }
            default {
                Fail "Unsupported option for 'headroom wrap openclaw': $arg"
            }
        }
    }

    [pscustomobject]@{
        PluginPath = $pluginPath
        PluginSpec = $pluginSpec
        SkipBuild = $skipBuild
        Copy = $copy
        ProxyPort = $proxyPort
        StartupTimeoutMs = $startupTimeoutMs
        GatewayProviderIds = $gatewayProviderIds.ToArray()
        PythonPath = $pythonPath
        NoAutoStart = $noAutoStart
        NoRestart = $noRestart
        Verbose = $verbose
    }
}

function Parse-OpenClawUnwrapArgs {
    param([string[]]$Arguments)

    $noRestart = $false
    $verbose = $false
    $i = 0
    while ($i -lt $Arguments.Count) {
        $arg = $Arguments[$i]
        switch -Regex ($arg) {
            '^--no-restart$' {
                $noRestart = $true
                $i += 1
                continue
            }
            '^--verbose$|^-v$' {
                $verbose = $true
                $i += 1
                continue
            }
            default {
                Fail "Unsupported option for 'headroom unwrap openclaw': $arg"
            }
        }
    }

    [pscustomobject]@{
        NoRestart = $noRestart
        Verbose = $verbose
    }
}

function Invoke-CapturedCommand {
    param(
        [string]$Action,
        [string]$Command,
        [string[]]$Arguments,
        [string]$WorkingDirectory
    )

    $previous = $null
    try {
        if ($WorkingDirectory) {
            $previous = Get-Location
            Set-Location $WorkingDirectory
        }

        $output = (& $Command @Arguments 2>&1 | Out-String).Trim()
        $exitCode = $LASTEXITCODE
    } finally {
        if ($previous) {
            Set-Location $previous
        }
    }

    if ($exitCode -ne 0) {
        if (-not $output) {
            $output = "exit code $exitCode"
        }
        Fail "$Action failed: $output"
    }

    return $output
}

function Get-OpenClawExistingEntryJson {
    $output = (& openclaw config get plugins.entries.headroom 2>$null | Out-String).Trim()
    if ($LASTEXITCODE -ne 0) {
        return $null
    }
    return $output
}

function Invoke-OpenClawPrepareEntryJson {
    param(
        [string]$ExistingEntryJson,
        [pscustomobject]$Parsed
    )

    $dockerArgs = New-Object System.Collections.Generic.List[string]
    $dockerArgs.AddRange([string[]]@('run','--rm'))
    $dockerArgs.AddRange((Get-SharedDockerArgs))
    $dockerArgs.Add('--entrypoint')
    $dockerArgs.Add('headroom')
    $dockerArgs.Add($HeadroomImage)
    $dockerArgs.AddRange([string[]]@('wrap','openclaw','--prepare-only','--proxy-port',"$($Parsed.ProxyPort)",'--startup-timeout-ms',"$($Parsed.StartupTimeoutMs)"))
    if ($ExistingEntryJson) {
        $dockerArgs.Add('--existing-entry-json')
        $dockerArgs.Add($ExistingEntryJson)
    }
    if ($Parsed.PythonPath) {
        $dockerArgs.Add('--python-path')
        $dockerArgs.Add($Parsed.PythonPath)
    }
    if ($Parsed.NoAutoStart) {
        $dockerArgs.Add('--no-auto-start')
    }
    foreach ($providerId in $Parsed.GatewayProviderIds) {
        $dockerArgs.Add('--gateway-provider-id')
        $dockerArgs.Add($providerId)
    }

    $output = (& docker @dockerArgs 2>&1 | Out-String).Trim()
    if ($LASTEXITCODE -ne 0) {
        Fail "Failed to prepare docker-native OpenClaw config: $output"
    }

    return $output
}

function Invoke-OpenClawPrepareUnwrapEntryJson {
    param([string]$ExistingEntryJson)

    $dockerArgs = New-Object System.Collections.Generic.List[string]
    $dockerArgs.AddRange([string[]]@('run','--rm'))
    $dockerArgs.AddRange((Get-SharedDockerArgs))
    $dockerArgs.Add('--entrypoint')
    $dockerArgs.Add('headroom')
    $dockerArgs.Add($HeadroomImage)
    $dockerArgs.AddRange([string[]]@('unwrap','openclaw','--prepare-only'))
    if ($ExistingEntryJson) {
        $dockerArgs.Add('--existing-entry-json')
        $dockerArgs.Add($ExistingEntryJson)
    }

    $output = (& docker @dockerArgs 2>&1 | Out-String).Trim()
    if ($LASTEXITCODE -ne 0) {
        Fail "Failed to prepare docker-native OpenClaw unwrap config: $output"
    }

    return $output
}

function Resolve-OpenClawExtensionsDir {
    $configOutput = Invoke-CapturedCommand -Action 'openclaw config file' -Command 'openclaw' -Arguments @('config','file')
    $configPath = ($configOutput -split "`r?`n")[-1].Trim()
    if (-not $configPath) {
        Fail 'Unable to resolve OpenClaw config path.'
    }
    return (Join-Path (Split-Path -Parent $configPath) 'extensions')
}

function Copy-OpenClawPluginIntoExtensions {
    param([string]$PluginPath)

    $distDir = Join-Path $PluginPath 'dist'
    $hookShimDir = Join-Path $PluginPath 'hook-shim'
    if (-not (Test-Path $distDir)) {
        Fail "Plugin dist folder missing at $distDir. Build the plugin first."
    }
    if (-not (Test-Path $hookShimDir)) {
        Fail "Plugin hook-shim folder missing at $hookShimDir. Build the plugin first."
    }

    $extensionsDir = Resolve-OpenClawExtensionsDir
    $targetDir = Join-Path $extensionsDir 'headroom'
    $targetDist = Join-Path $targetDir 'dist'
    $targetHookShim = Join-Path $targetDir 'hook-shim'
    New-Item -ItemType Directory -Force -Path $targetDir | Out-Null
    if (Test-Path $targetDist) { Remove-Item -Recurse -Force $targetDist }
    if (Test-Path $targetHookShim) { Remove-Item -Recurse -Force $targetHookShim }
    Copy-Item -Recurse -Force $distDir $targetDist
    Copy-Item -Recurse -Force $hookShimDir $targetHookShim

    foreach ($fileName in @('openclaw.plugin.json','package.json','README.md')) {
        $source = Join-Path $PluginPath $fileName
        if (Test-Path $source) {
            Copy-Item -Force $source (Join-Path $targetDir $fileName)
        }
    }

    return $targetDir
}

function Install-OpenClawPlugin {
    param([pscustomobject]$Parsed)

    if ($Parsed.PluginPath) {
        if (-not (Test-Path $Parsed.PluginPath)) {
            Fail "Plugin path not found: $($Parsed.PluginPath)."
        }
        if (-not (Test-Path (Join-Path $Parsed.PluginPath 'package.json'))) {
            Fail "Invalid plugin path (missing package.json): $($Parsed.PluginPath)"
        }
        if (-not (Test-Path (Join-Path $Parsed.PluginPath 'openclaw.plugin.json'))) {
            Fail "Invalid plugin path (missing openclaw.plugin.json): $($Parsed.PluginPath)"
        }
    }

    if ($Parsed.PluginPath -and -not $Parsed.SkipBuild) {
        Require-Command npm
        Write-Host '  Building OpenClaw plugin (npm install + npm run build)...'
        [void](Invoke-CapturedCommand -Action 'npm install' -Command 'npm' -Arguments @('install') -WorkingDirectory $Parsed.PluginPath)
        [void](Invoke-CapturedCommand -Action 'npm run build' -Command 'npm' -Arguments @('run','build') -WorkingDirectory $Parsed.PluginPath)
    }

    if ($Parsed.PluginPath) {
        if ($Parsed.Copy) {
            $arguments = @('plugins','install','--dangerously-force-unsafe-install',$Parsed.PluginPath)
            $workingDirectory = $null
        } else {
            $arguments = @('plugins','install','--dangerously-force-unsafe-install','--link','.')
            $workingDirectory = $Parsed.PluginPath
        }
    } else {
        $arguments = @('plugins','install','--dangerously-force-unsafe-install',$Parsed.PluginSpec)
        $workingDirectory = $null
    }

    $previous = $null
    try {
        if ($workingDirectory) {
            $previous = Get-Location
            Set-Location $workingDirectory
        }
        $installOutput = (& openclaw @arguments 2>&1 | Out-String).Trim()
        $installExitCode = $LASTEXITCODE
    } finally {
        if ($previous) {
            Set-Location $previous
        }
    }

    if ($installExitCode -eq 0) {
        if ($Parsed.Verbose -and $installOutput) {
            Write-Host $installOutput
        }
        return
    }

    $lowerOutput = $installOutput.ToLowerInvariant()
    if ($lowerOutput.Contains('plugin already exists')) {
        Write-Host '  Plugin already installed; continuing with configuration/update steps.'
        return
    }

    if ($Parsed.PluginPath -and -not $Parsed.Copy -and $lowerOutput.Contains('also not a valid hook pack')) {
        Write-Host '  OpenClaw linked-path install bug detected; applying extension-path fallback...'
        $targetDir = Copy-OpenClawPluginIntoExtensions -PluginPath $Parsed.PluginPath
        Write-Host "  Fallback plugin copy completed: $targetDir"
        return
    }

    if (-not $installOutput) {
        $installOutput = "exit code $installExitCode"
    }
    Fail "openclaw plugins install failed: $installOutput"
}

function Restart-OrStartOpenClawGateway {
    $restartOutput = (& openclaw gateway restart 2>&1 | Out-String).Trim()
    if ($LASTEXITCODE -eq 0) {
        return [pscustomobject]@{ Action = 'restarted'; Output = $restartOutput }
    }

    $startOutput = Invoke-CapturedCommand -Action 'openclaw gateway start' -Command 'openclaw' -Arguments @('gateway','start')
    return [pscustomobject]@{ Action = 'started'; Output = $startOutput }
}

function Invoke-OpenClawWrap {
    param([string[]]$Arguments)

    Require-Command openclaw
    $parsed = Parse-OpenClawWrapArgs -Arguments $Arguments
    $existingEntryJson = Get-OpenClawExistingEntryJson
    $entryJson = Invoke-OpenClawPrepareEntryJson -ExistingEntryJson $existingEntryJson -Parsed $parsed

    Write-Host ""
    Write-Host "  ╔═══════════════════════════════════════════════╗"
    Write-Host "  ║           HEADROOM WRAP: OPENCLAW             ║"
    Write-Host "  ╚═══════════════════════════════════════════════╝"
    Write-Host ""
    if ($parsed.PluginPath) {
        Write-Host "  Plugin source: local ($($parsed.PluginPath))"
    } else {
        Write-Host "  Plugin source: npm ($($parsed.PluginSpec))"
    }

    Write-Host '  Writing plugin configuration...'
    [void](Invoke-CapturedCommand -Action 'openclaw config set plugins.entries.headroom' -Command 'openclaw' -Arguments @('config','set','plugins.entries.headroom',$entryJson,'--strict-json'))
    Write-Host '  Installing OpenClaw plugin with required unsafe-install flag...'
    Install-OpenClawPlugin -Parsed $parsed
    [void](Invoke-CapturedCommand -Action 'openclaw config set plugins.slots.contextEngine' -Command 'openclaw' -Arguments @('config','set','plugins.slots.contextEngine','"headroom"','--strict-json'))
    [void](Invoke-CapturedCommand -Action 'openclaw config validate' -Command 'openclaw' -Arguments @('config','validate'))

    if ($parsed.NoRestart) {
        Write-Host '  Skipping gateway restart (--no-restart).'
        Write-Host '  Run `openclaw gateway restart` (or `openclaw gateway start`) to apply plugin changes.'
    } else {
        Write-Host '  Applying plugin changes to OpenClaw gateway...'
        $gatewayResult = Restart-OrStartOpenClawGateway
        Write-Host "  Gateway $($gatewayResult.Action)."
        if ($parsed.Verbose -and $gatewayResult.Output) {
            Write-Host $gatewayResult.Output
        }
    }

    $inspectOutput = Invoke-CapturedCommand -Action 'openclaw plugins inspect headroom' -Command 'openclaw' -Arguments @('plugins','inspect','headroom')
    if ($parsed.Verbose -and $inspectOutput) {
        Write-Host $inspectOutput
    }

    Write-Host ""
    Write-Host "✓ OpenClaw is configured to use Headroom context compression."
    Write-Host "  Plugin: headroom"
    Write-Host "  Slot:   plugins.slots.contextEngine = headroom"
    Write-Host ""
}

function Invoke-OpenClawUnwrap {
    param([string[]]$Arguments)

    Require-Command openclaw
    $parsed = Parse-OpenClawUnwrapArgs -Arguments $Arguments
    $existingEntryJson = Get-OpenClawExistingEntryJson
    $entryJson = Invoke-OpenClawPrepareUnwrapEntryJson -ExistingEntryJson $existingEntryJson

    Write-Host ""
    Write-Host "  ╔═══════════════════════════════════════════════╗"
    Write-Host "  ║          HEADROOM UNWRAP: OPENCLAW            ║"
    Write-Host "  ╚═══════════════════════════════════════════════╝"
    Write-Host ""
    Write-Host '  Disabling Headroom plugin and removing engine mapping...'

    [void](Invoke-CapturedCommand -Action 'openclaw config set plugins.entries.headroom' -Command 'openclaw' -Arguments @('config','set','plugins.entries.headroom',$entryJson,'--strict-json'))
    [void](Invoke-CapturedCommand -Action 'openclaw config set plugins.slots.contextEngine' -Command 'openclaw' -Arguments @('config','set','plugins.slots.contextEngine','"legacy"','--strict-json'))
    [void](Invoke-CapturedCommand -Action 'openclaw config validate' -Command 'openclaw' -Arguments @('config','validate'))

    if ($parsed.NoRestart) {
        Write-Host '  Skipping gateway restart (--no-restart).'
        Write-Host '  Run `openclaw gateway restart` (or `openclaw gateway start`) to apply unwrap changes.'
    } else {
        Write-Host '  Applying unwrap changes to OpenClaw gateway...'
        $gatewayResult = Restart-OrStartOpenClawGateway
        Write-Host "  Gateway $($gatewayResult.Action)."
        if ($parsed.Verbose -and $gatewayResult.Output) {
            Write-Host $gatewayResult.Output
        }
    }

    if ($parsed.Verbose) {
        $inspectOutput = Invoke-CapturedCommand -Action 'openclaw plugins inspect headroom' -Command 'openclaw' -Arguments @('plugins','inspect','headroom')
        if ($inspectOutput) {
            Write-Host $inspectOutput
        }
    }

    Write-Host ""
    Write-Host "✓ OpenClaw Headroom wrap removed."
    Write-Host "  Plugin: headroom (installed, disabled)"
    Write-Host "  Slot:   plugins.slots.contextEngine = legacy"
    Write-Host ""
}

function Parse-WrapArgs {
    param([string[]]$Arguments)

    $known = New-Object System.Collections.Generic.List[string]
    $hostArgs = New-Object System.Collections.Generic.List[string]
    $port = 8787
    $noRtk = $false
    $noProxy = $false
    $learn = $false
    $backend = $null
    $anyllm = $null
    $region = $null

    $i = 0
    while ($i -lt $Arguments.Count) {
        $arg = $Arguments[$i]
        switch -Regex ($arg) {
            '^--$' {
                for ($j = $i + 1; $j -lt $Arguments.Count; $j++) {
                    $hostArgs.Add($Arguments[$j])
                }
                $i = $Arguments.Count
                continue
            }
            '^--port$|^-p$' {
                Require-OptionValue -Arguments $Arguments -Index $i -Option $arg
                $port = Parse-PortValue -Value $Arguments[$i + 1]
                $known.Add($arg)
                $known.Add($Arguments[$i + 1])
                $i += 2
                continue
            }
            '^--port=' {
                $port = Parse-PortValue -Value ($arg -replace '^--port=', '')
                $known.Add($arg)
                $i += 1
                continue
            }
            '^--no-rtk$' {
                $noRtk = $true
                $known.Add($arg)
                $i += 1
                continue
            }
            '^--no-proxy$' {
                $noProxy = $true
                $known.Add($arg)
                $i += 1
                continue
            }
            '^--learn$' {
                $learn = $true
                $known.Add($arg)
                $i += 1
                continue
            }
            '^--verbose$|^-v$' {
                $known.Add($arg)
                $i += 1
                continue
            }
            '^--backend$' {
                Require-OptionValue -Arguments $Arguments -Index $i -Option $arg
                $backend = $Arguments[$i + 1]
                $known.Add($arg)
                $known.Add($Arguments[$i + 1])
                $i += 2
                continue
            }
            '^--backend=' {
                $backend = $arg -replace '^--backend=', ''
                $known.Add($arg)
                $i += 1
                continue
            }
            '^--anyllm-provider$' {
                Require-OptionValue -Arguments $Arguments -Index $i -Option $arg
                $anyllm = $Arguments[$i + 1]
                $known.Add($arg)
                $known.Add($Arguments[$i + 1])
                $i += 2
                continue
            }
            '^--anyllm-provider=' {
                $anyllm = $arg -replace '^--anyllm-provider=', ''
                $known.Add($arg)
                $i += 1
                continue
            }
            '^--region$' {
                Require-OptionValue -Arguments $Arguments -Index $i -Option $arg
                $region = $Arguments[$i + 1]
                $known.Add($arg)
                $known.Add($Arguments[$i + 1])
                $i += 2
                continue
            }
            '^--region=' {
                $region = $arg -replace '^--region=', ''
                $known.Add($arg)
                $i += 1
                continue
            }
            default {
                for ($j = $i; $j -lt $Arguments.Count; $j++) {
                    $hostArgs.Add($Arguments[$j])
                }
                $i = $Arguments.Count
            }
        }
    }

    [pscustomobject]@{
        KnownArgs = $known.ToArray()
        HostArgs = $hostArgs.ToArray()
        Port = $port
        NoRtk = $noRtk
        NoProxy = $noProxy
        Learn = $learn
        Backend = $backend
        Anyllm = $anyllm
        Region = $region
    }
}

function Invoke-PrepareOnly {
    param(
        [string]$Tool,
        [string[]]$KnownArgs
    )

    $dockerArgs = New-Object System.Collections.Generic.List[string]
    $dockerArgs.AddRange([string[]]@('run','--rm'))
    Add-TtyArgs -ArgsList $dockerArgs
    $dockerArgs.AddRange((Get-SharedDockerArgs))
    $dockerArgs.Add('--env')
    $dockerArgs.Add("HEADROOM_RTK_TARGET=$(Get-RtkTarget)")
    $dockerArgs.Add('--entrypoint')
    $dockerArgs.Add('headroom')
    $dockerArgs.Add($HeadroomImage)
    $dockerArgs.AddRange([string[]]@('wrap',$Tool,'--prepare-only'))
    foreach ($arg in $KnownArgs) {
        $dockerArgs.Add($arg)
    }

    & docker @dockerArgs
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to prepare docker-native wrap for $Tool"
    }
}

Require-Command docker

if ($args.Count -eq 0) {
    Invoke-HeadroomDocker -Arguments @('--help')
    exit 0
}

switch ($args[0]) {
    'install' {
        if ($args.Count -eq 1 -or $args[1] -eq '--help' -or $args[1] -eq '-?') {
            Show-InstallHelp
            exit 0
        }

        $installCommand = $args[1]
        $installArgs = if ($args.Count -gt 2) { $args[2..($args.Count - 1)] } else { @() }

        switch ($installCommand) {
            'apply' {
                $parsed = Parse-InstallApplyArgs -Arguments $installArgs
                Start-PersistentDockerInstall -Profile $parsed.Profile -Image $parsed.Image -Port $parsed.Port -Backend $parsed.Backend -AnyllmProvider $parsed.AnyllmProvider -Region $parsed.Region -Mode $parsed.Mode -Memory $parsed.Memory -TelemetryEnabled $parsed.TelemetryEnabled
                Write-Host "Installed docker-native persistent deployment '$($parsed.Profile)' on port $($parsed.Port)."
                exit 0
            }
            'status' {
                $profile = Parse-InstallProfileArgs -Arguments $installArgs
                Show-PersistentDockerInstallStatus -Profile $profile
                exit 0
            }
            'start' {
                $profile = Parse-InstallProfileArgs -Arguments $installArgs
                $state = Read-PersistentState -Profile $profile
                Start-PersistentDockerInstall -Profile $state.profile -Image $state.image -Port $state.port -Backend $state.backend -AnyllmProvider $state.anyllm_provider -Region $state.region -Mode $state.proxy_mode -Memory ([bool]$state.memory_enabled) -TelemetryEnabled ([bool]$state.telemetry_enabled)
                Write-Host "Started docker-native persistent deployment '$profile'."
                exit 0
            }
            'stop' {
                $profile = Parse-InstallProfileArgs -Arguments $installArgs
                Stop-PersistentDockerInstall -Profile $profile
                Write-Host "Stopped docker-native persistent deployment '$profile'."
                exit 0
            }
            'restart' {
                $profile = Parse-InstallProfileArgs -Arguments $installArgs
                $state = Read-PersistentState -Profile $profile
                Start-PersistentDockerInstall -Profile $state.profile -Image $state.image -Port $state.port -Backend $state.backend -AnyllmProvider $state.anyllm_provider -Region $state.region -Mode $state.proxy_mode -Memory ([bool]$state.memory_enabled) -TelemetryEnabled ([bool]$state.telemetry_enabled)
                Write-Host "Restarted docker-native persistent deployment '$profile'."
                exit 0
            }
            'remove' {
                $profile = Parse-InstallProfileArgs -Arguments $installArgs
                Remove-PersistentDockerInstall -Profile $profile
                Write-Host "Removed docker-native persistent deployment '$profile'."
                exit 0
            }
            default {
                Fail "Unsupported install target: $installCommand"
            }
        }
    }
    'wrap' {
        if ($args.Count -eq 1 -or $args[1] -eq '--help' -or $args[1] -eq '-?') {
            Show-WrapHelp
            exit 0
        }

        if ($args.Count -lt 2) {
            Fail 'Usage: headroom wrap <claude|codex|aider|cursor|openclaw|opencode> [...]'
        }

        $tool = $args[1]
        $wrapArgs = if ($args.Count -gt 2) { $args[2..($args.Count - 1)] } else { @() }

        switch ($tool) {
            'claude' { }
            'codex' { }
            'aider' { }
            'cursor' { }
            'openclaw' { }
            'opencode' { }
            default {
                Fail "Docker-native wrapper does not support 'wrap $tool'. Supported targets: claude, codex, aider, cursor, openclaw, opencode"
            }
        }

        if ($tool -eq 'openclaw') {
            if (Test-HelpFlag -Arguments $wrapArgs) {
                $helpArgs = @('wrap','openclaw') + $wrapArgs
                Invoke-HeadroomDocker -Arguments $helpArgs
                exit 0
            }

            Invoke-OpenClawWrap -Arguments $wrapArgs
            exit 0
        }

        if (Test-HelpFlag -Arguments $wrapArgs) {
            $helpArgs = @('wrap', $tool) + $wrapArgs
            Invoke-HeadroomDocker -Arguments $helpArgs
            exit 0
        }

        $parsed = Parse-WrapArgs -Arguments $wrapArgs
        $contextTool = Get-ContextTool
        $proxyArgs = New-Object System.Collections.Generic.List[string]
        if ($parsed.Learn) { $proxyArgs.Add('--learn') }
        if ($parsed.Backend) { $proxyArgs.AddRange([string[]]@('--backend', $parsed.Backend)) }
        if ($parsed.Anyllm) { $proxyArgs.AddRange([string[]]@('--anyllm-provider', $parsed.Anyllm)) }
        if ($parsed.Region) { $proxyArgs.AddRange([string[]]@('--region', $parsed.Region)) }

        $containerName = $null
        try {
            if (-not $parsed.NoProxy) {
                $containerName = Start-ProxyContainer -Port $parsed.Port -ProxyArgs $proxyArgs.ToArray()
            }

            $prepareArgs = New-Object System.Collections.Generic.List[string]
            foreach ($arg in $parsed.KnownArgs) {
                $prepareArgs.Add($arg)
            }
            if (-not $parsed.NoProxy) {
                $prepareArgs.Add('--no-proxy')
            }
            if ((-not $parsed.NoRtk) -and $contextTool -eq 'lean-ctx') {
                $prepareArgs.Add('--no-rtk')
            }
            Invoke-PrepareOnly -Tool $tool -KnownArgs $prepareArgs.ToArray()

            if ((-not $parsed.NoRtk) -and $contextTool -eq 'lean-ctx') {
                Invoke-LeanCtxInit -Agent $tool
            }

            switch ($tool) {
                'claude' {
                    if ((-not $parsed.NoRtk) -and $contextTool -eq 'rtk') {
                        Invoke-ClaudeRtkInit
                    }
                    $exitCode = Invoke-WithTemporaryEnv -Environment @{ ANTHROPIC_BASE_URL = "http://127.0.0.1:$($parsed.Port)" } -Command 'claude' -Arguments $parsed.HostArgs
                    exit $exitCode
                }
                'codex' {
                    $exitCode = Invoke-WithTemporaryEnv -Environment @{ OPENAI_BASE_URL = "http://127.0.0.1:$($parsed.Port)/v1" } -Command 'codex' -Arguments $parsed.HostArgs
                    exit $exitCode
                }
                'aider' {
                    $exitCode = Invoke-WithTemporaryEnv -Environment @{
                        OPENAI_API_BASE = "http://127.0.0.1:$($parsed.Port)/v1"
                        ANTHROPIC_BASE_URL = "http://127.0.0.1:$($parsed.Port)"
                    } -Command 'aider' -Arguments $parsed.HostArgs
                    exit $exitCode
                }
                'cursor' {
                    Write-Host "Headroom proxy is running for Cursor."
                    Write-Host ""
                    Write-Host "OpenAI base URL:     http://127.0.0.1:$($parsed.Port)/v1"
                    Write-Host "Anthropic base URL:  http://127.0.0.1:$($parsed.Port)"
                    Write-Host ""
                    Write-Host "Press Ctrl+C to stop the proxy."
                    while ($true) { Start-Sleep -Seconds 1 }
                }
            }
        } finally {
            Stop-ProxyContainer -ContainerName $containerName
        }
    }
    'unwrap' {
        if ($args.Count -eq 1 -or $args[1] -eq '--help' -or $args[1] -eq '-?') {
            Invoke-HeadroomDocker -Arguments @('unwrap','--help')
            exit 0
        }

        if ($args.Count -ge 2 -and $args[1] -eq 'openclaw') {
            $unwrapArgs = if ($args.Count -gt 2) { $args[2..($args.Count - 1)] } else { @() }
            if (Test-HelpFlag -Arguments $unwrapArgs) {
                $helpArgs = @('unwrap','openclaw') + $unwrapArgs
                Invoke-HeadroomDocker -Arguments $helpArgs
                exit 0
            }

            Invoke-OpenClawUnwrap -Arguments $unwrapArgs
            exit 0
        }
        Invoke-HeadroomDocker -Arguments $args
    }
    'proxy' {
        $port = 8787
        $forwardArgs = New-Object System.Collections.Generic.List[string]
        foreach ($arg in $args) { $forwardArgs.Add($arg) }
        for ($i = 1; $i -lt $args.Count; $i++) {
            if ($args[$i] -eq '--port' -or $args[$i] -eq '-p') {
                Require-OptionValue -Arguments $args -Index $i -Option $args[$i]
                $port = Parse-PortValue -Value $args[$i + 1]
                break
            }
            if ($args[$i] -match '^--port=') {
                $port = Parse-PortValue -Value ($args[$i] -replace '^--port=', '')
                break
            }
        }

        $dockerArgs = New-Object System.Collections.Generic.List[string]
        $dockerArgs.AddRange([string[]]@('run','--rm'))
        Add-TtyArgs -ArgsList $dockerArgs
        $dockerArgs.AddRange([string[]]@('-p',"$port`:$port"))
        $dockerArgs.AddRange((Get-SharedDockerArgs))
        $dockerArgs.Add('--entrypoint')
        $dockerArgs.Add('headroom')
        $dockerArgs.Add($HeadroomImage)
        foreach ($arg in $forwardArgs) {
            $dockerArgs.Add($arg)
        }

        & docker @dockerArgs
        exit $LASTEXITCODE
    }
    default {
        Invoke-HeadroomDocker -Arguments $args
    }
}
'@

    $wrapper = $wrapper.Replace('__HEADROOM_INSTALL_IMAGE__', $resolvedInstallImage)

    $cmdWrapper = ([string][char]64) + "echo off`r`npowershell -NoLogo -NoProfile -ExecutionPolicy Bypass -File ""%~dp0headroom.ps1"" %*`r`n"

    Set-Content -Path $wrapperPath -Value $wrapper -Encoding utf8
    Set-Content -Path $cmdPath -Value $cmdWrapper -Encoding ascii
}

Require-Command docker
docker version | Out-Null
if ($LASTEXITCODE -ne 0) {
    throw 'Docker is installed but not available to the current user'
}

New-Item -ItemType Directory -Force -Path $InstallDir | Out-Null
Write-Wrapper -TargetDir $InstallDir
Ensure-PathEntry -PathEntry $InstallDir
Ensure-ProfileBlock -PathEntry $InstallDir

if ($env:HEADROOM_DOCKER_IMAGE) {
    $null = docker image inspect $InstallImage 2>$null
    if ($LASTEXITCODE -eq 0) {
        Write-Info "Using existing HEADROOM_DOCKER_IMAGE=$InstallImage"
    } else {
        Write-Info "Pulling $InstallImage"
        docker pull $InstallImage | Out-Null
    }
} else {
    Write-Info "Pulling $ImageDefault"
    docker pull $ImageDefault | Out-Null
}

Write-Host ""
Write-Host "Headroom Docker-native install complete."
Write-Host ""
Write-Host "Installed wrappers:"
Write-Host "  $InstallDir\headroom.ps1"
Write-Host "  $InstallDir\headroom.cmd"
Write-Host ""
Write-Host "Next steps:"
Write-Host "  1. Restart PowerShell"
Write-Host "  2. Try: headroom proxy"
Write-Host "  3. Docs: https://github.com/chopratejas/headroom/blob/main/docs/docker-install.md"
