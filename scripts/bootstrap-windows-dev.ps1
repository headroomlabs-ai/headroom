param(
    [string]$Python = "",
    [switch]$CheckOnly,
    [switch]$SkipRust,
    [switch]$SkipNode,
    [switch]$SkipDocs,
    [switch]$SkipSmoke
)

$ErrorActionPreference = "Stop"

function Write-Step {
    param([string]$Message)
    Write-Host ""
    Write-Host "==> $Message"
}

function Get-CommandPath {
    param([string]$Name)
    $cmd = Get-Command $Name -ErrorAction SilentlyContinue
    if ($cmd) {
        return $cmd.Source
    }
    return $null
}

function Resolve-Python {
    param([string]$Requested)

    if ($Requested) {
        if (-not (Test-Path -LiteralPath $Requested)) {
            throw "Requested Python path does not exist: $Requested"
        }
        return (Resolve-Path -LiteralPath $Requested).Path
    }

    if ($env:HEADROOM_DEV_PYTHON -and (Test-Path -LiteralPath $env:HEADROOM_DEV_PYTHON)) {
        return (Resolve-Path -LiteralPath $env:HEADROOM_DEV_PYTHON).Path
    }

    $repoVenv = Join-Path $script:RepoRoot ".venv\Scripts\python.exe"
    if (Test-Path -LiteralPath $repoVenv) {
        return (Resolve-Path -LiteralPath $repoVenv).Path
    }

    $auditVenv = Join-Path $script:RepoRoot "..\..\.venv\Scripts\python.exe"
    if (Test-Path -LiteralPath $auditVenv) {
        return (Resolve-Path -LiteralPath $auditVenv).Path
    }

    $systemPython = Get-CommandPath "python.exe"
    if (-not $systemPython) {
        throw "python.exe not found. Install Python 3.10+ first."
    }

    Write-Step "Creating local .venv"
    & $systemPython -m venv (Join-Path $script:RepoRoot ".venv")
    return (Resolve-Path -LiteralPath $repoVenv).Path
}

function Ensure-Rust {
    if ($SkipRust) {
        return
    }

    $cargoBin = Join-Path $env:USERPROFILE ".cargo\bin"
    if (Test-Path -LiteralPath $cargoBin) {
        $env:PATH = "$cargoBin;$env:PATH"
    }

    if (Get-CommandPath "cargo.exe") {
        return
    }

    $winget = Get-CommandPath "winget.exe"
    if (-not $winget) {
        throw "cargo.exe not found and winget.exe is unavailable. Install Rustup from https://rustup.rs/."
    }

    Write-Step "Installing Rustup"
    & $winget install --id Rustlang.Rustup -e --silent --accept-package-agreements --accept-source-agreements
    if (Test-Path -LiteralPath $cargoBin) {
        $env:PATH = "$cargoBin;$env:PATH"
    }
}

function Invoke-NpmCi {
    param([string]$RelativePath)

    if ($SkipNode) {
        return
    }

    $npm = Get-CommandPath "npm.cmd"
    if (-not $npm) {
        throw "npm.cmd not found. Install Node.js 18+."
    }

    $dir = Join-Path $script:RepoRoot $RelativePath
    if (-not (Test-Path -LiteralPath (Join-Path $dir "package-lock.json"))) {
        return
    }

    Write-Step "npm ci in $RelativePath"
    Push-Location $dir
    try {
        & $npm ci
    }
    finally {
        Pop-Location
    }
}

$script:RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $script:RepoRoot

$pythonExe = Resolve-Python $Python
$cargoBinPath = Join-Path $env:USERPROFILE ".cargo\bin"
if (Test-Path -LiteralPath $cargoBinPath) {
    $env:PATH = "$cargoBinPath;$env:PATH"
}

Write-Step "Tool versions"
& $pythonExe --version
& $pythonExe -m pip --version
node.exe --version
npm.cmd --version
if (Get-CommandPath "cargo.exe") {
    cargo.exe --version
}

if ($CheckOnly) {
    exit 0
}

Ensure-Rust

Write-Step "Installing Python build/dev tools"
& $pythonExe -m pip install --upgrade pip
& $pythonExe -m pip install --upgrade `
    "maturin>=1.5,<2" uv ruff mypy pre-commit pytest-cov `
    opentelemetry-sdk opentelemetry-exporter-otlp-proto-http `
    "tree-sitter-language-pack>=0.10.0,<1.0" "tree-sitter>=0.25.2,<0.26" `
    openpyxl

Write-Step "Building native extension with maturin ci profile"
& $pythonExe -m maturin develop -m "crates\headroom-py\Cargo.toml" --profile ci

Write-Step "Installing runtime extras without rebuilding headroom-ai"
& $pythonExe -m pip install --upgrade `
    "tree-sitter-language-pack>=0.10.0,<1.0" `
    anthropic ollama langchain-ollama hnswlib `
    "sentence-transformers>=2.2.0,<6.0" fastembed jinja2 xlrd

Invoke-NpmCi "sdk\typescript"
Invoke-NpmCi "plugins\openclaw"
if (-not $SkipDocs) {
    Invoke-NpmCi "docs"
}

if (-not $SkipSmoke) {
    Write-Step "Smoke checks"
    & $pythonExe -c "import importlib.util, headroom; assert importlib.util.find_spec('headroom._core'); print(headroom.__version__)"
    & $pythonExe -m headroom.cli --version
    & $pythonExe -m pip check
    Push-Location (Join-Path $script:RepoRoot "sdk\typescript")
    try { npm.cmd run build } finally { Pop-Location }
    Push-Location (Join-Path $script:RepoRoot "plugins\openclaw")
    try { npm.cmd run build } finally { Pop-Location }
}

Write-Host ""
Write-Host "Headroom Windows dev environment is ready."
