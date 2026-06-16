@echo off
setlocal enabledelayedexpansion
REM Headroom Windows One-Click Setup
REM Fork: https://github.com/dd2673/headroom (fix/windows-adaptation branch)
REM Requires: Python 3.11+, Rust toolchain, VS Build Tools with MSVC

echo ============================================
echo  Headroom Windows Setup
echo ============================================
echo.

set VENV_DIR=D:\tools\headroom-venv
set REPO_DIR=D:\AI\headroom
set FORK_URL=https://github.com/dd2673/headroom.git
set BRANCH=fix/windows-adaptation

REM Step 1: Check prerequisites
echo [1/7] Checking prerequisites...
python --version >nul 2>&1 || (echo ERROR: Python not found && exit /b 1)
rustc --version >nul 2>&1 || (echo ERROR: Rust not found. Install: https://rustup.rs && exit /b 1)
where git >nul 2>&1 || (echo ERROR: Git not found && exit /b 1)
echo    Python: OK
echo    Rust: OK
echo    Git: OK

REM Step 2: Find MSVC
echo.
echo [2/7] Finding MSVC Build Tools...
set VCVARSALL=
for /f "delims=" %%i in ('dir /s /b "C:\Program Files*\Microsoft Visual Studio\*\VC\Auxiliary\Build\vcvarsall.bat" 2^>nul') do set VCVARSALL=%%i
for /f "delims=" %%i in ('dir /s /b "D:\vs*\VC\Auxiliary\Build\vcvarsall.bat" 2^>nul') do set VCVARSALL=%%i
if "%VCVARSALL%"=="" (
    echo ERROR: MSVC Build Tools not found.
    echo Install Visual Studio Build Tools with "C++ desktop development" workload.
    exit /b 1
)
echo    Found: %VCVARSALL%

REM Step 3: Clone repo
echo.
echo [3/7] Cloning fork...
if exist "%REPO_DIR%\.git" (
    echo    Repo exists at %REPO_DIR%
    cd /d "%REPO_DIR%"
    git fetch fork 2>nul
    git checkout %BRANCH% 2>nul || (
        git remote add fork %FORK_URL% 2>nul
        git fetch fork
        git checkout -b %BRANCH% fork/%BRANCH%
    )
) else (
    git clone -b %BRANCH% %FORK_URL% "%REPO_DIR%"
    cd /d "%REPO_DIR%"
    git remote add origin https://github.com/chopratejas/headroom.git 2>nul
)

REM Step 4: Create venv
echo.
echo [4/7] Creating virtual environment at %VENV_DIR%...
if not exist "%VENV_DIR%\Scripts\python.exe" (
    python -m venv "%VENV_DIR%"
)
echo    Venv: OK

REM Step 5: Install headroom from local source
echo.
echo [5/7] Installing headroom from local source (includes Rust extension)...
"%VENV_DIR%\Scripts\pip.exe" install --upgrade pip >nul 2>&1
"%VENV_DIR%\Scripts\pip.exe" install -e ".[proxy]" --no-build-isolation 2>&1
if errorlevel 1 (
    echo    Trying maturin build...
    "%VENV_DIR%\Scripts\pip.exe" install maturin
    call "%VCVARSALL%" x64 >nul 2>&1
    "%VENV_DIR%\Scripts\maturin.exe" develop --release
)
echo    Headroom: OK

REM Step 6: Verify Rust extension
echo.
echo [6/7] Verifying Rust extension...
"%VENV_DIR%\Scripts\python.exe" -c "from headroom._core import hello; print('Rust:', hello())" 2>&1
if errorlevel 1 (
    echo WARNING: Rust extension not loaded, building manually...
    call "%VCVARSALL%" x64 >nul 2>&1
    "%VENV_DIR%\Scripts\maturin.exe" develop --release
)

REM Step 7: Install autostart script
echo.
echo [7/7] Installing autostart script...
copy /y "%~dp0headroom_autostart.py" "%USERPROFILE%\.claude\headroom_autostart.py" >nul 2>&1
if not exist "%USERPROFILE%\.claude\headroom_autostart.py" (
    echo WARNING: headroom_autostart.py not found in script directory.
    echo Copy it manually to %USERPROFILE%\.claude\headroom_autostart.py
)

echo.
echo ============================================
echo  Setup complete!
echo ============================================
echo.
echo Next steps:
echo   1. Add SessionStart hook to %USERPROFILE%\.claude\settings.json
echo   2. Set ANTHROPIC_BASE_URL to http://127.0.0.1:8787
echo   3. Start a new Claude Code session (proxy auto-starts)
echo   4. Run: headroom perf
echo.
