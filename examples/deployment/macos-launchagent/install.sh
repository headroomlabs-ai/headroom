#!/usr/bin/env bash
#
# Headroom Proxy LaunchAgent Installer for macOS
#
# This script installs the headroom proxy as a macOS LaunchAgent for automatic
# startup and management. The service will start on login and restart on crash.
#
# Usage: ./install.sh [--port PORT] [--unattended]
#
# Options:
#   --port PORT       Port for proxy server (default: 8787)
#   --unattended      Skip interactive prompts (use defaults)
#

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
PLIST_LABEL="com.headroom.proxy"
PLIST_FILENAME="${PLIST_LABEL}.plist"
PLIST_DEST="${HOME}/Library/LaunchAgents/${PLIST_FILENAME}"
LOG_DIR="${HOME}/Library/Logs/headroom"
DEFAULT_PORT=8787
USER_UID=$(id -u)

# Parse command line arguments
CUSTOM_PORT=""
UNATTENDED=false
while [[ $# -gt 0 ]]; do
  case $1 in
    --port)
      CUSTOM_PORT="$2"
      shift 2
      ;;
    --unattended)
      UNATTENDED=true
      shift
      ;;
    *)
      echo "Unknown option: $1"
      echo "Usage: $0 [--port PORT] [--unattended]"
      exit 1
      ;;
  esac
done

# Helper functions
info() {
  echo -e "${BLUE}==>${NC} $*"
}

success() {
  echo -e "${GREEN}✓${NC} $*"
}

warning() {
  echo -e "${YELLOW}⚠${NC} $*"
}

error() {
  echo -e "${RED}✗${NC} $*" >&2
}

fatal() {
  error "$*"
  exit 1
}

# Check if we're on macOS
OS_NAME=$(uname -s)
if [[ "${OS_NAME}" != "Darwin" ]]; then
  fatal "This script is only for macOS. Use systemd on Linux."
fi

# Check if headroom is installed
info "Checking for headroom installation..."
HEADROOM_PATH=$(command -v headroom || true)
if [[ -z "${HEADROOM_PATH}" ]]; then
  fatal "headroom not found in PATH. Please install it first: pip install headroom-ai[proxy]"
fi
success "Found headroom at: ${HEADROOM_PATH}"

# Verify proxy support
info "Verifying proxy support..."
if ! "${HEADROOM_PATH}" proxy --help >/dev/null 2>&1; then
  fatal "headroom proxy command not available. Install with: pip install headroom-ai[proxy]"
fi
success "Proxy support verified"

# Check if service is already installed
if [[ -f "${PLIST_DEST}" ]]; then
  warning "LaunchAgent already installed at: ${PLIST_DEST}"

  # Check if service is running
  if launchctl print "gui/${USER_UID}/${PLIST_LABEL}" >/dev/null 2>&1; then
    info "Stopping existing service..."
    launchctl bootout "gui/${USER_UID}/${PLIST_LABEL}" 2>/dev/null || true
  fi

  if [[ "${UNATTENDED}" == false ]]; then
    read -rp "Reinstall? [y/N] " response
    if [[ ! "${response}" =~ ^[Yy]$ ]]; then
      echo "Installation cancelled."
      exit 0
    fi
  fi
fi

# Get port configuration
if [[ -n "${CUSTOM_PORT}" ]]; then
  PORT="${CUSTOM_PORT}"
elif [[ "${UNATTENDED}" == true ]]; then
  PORT="${DEFAULT_PORT}"
else
  read -rp "Port for proxy server (default: ${DEFAULT_PORT}): " PORT
  PORT="${PORT:-${DEFAULT_PORT}}"
fi

# Validate port number
if ! [[ "${PORT}" =~ ^[0-9]+$ ]] || [[ "${PORT}" -lt 1024 ]] || [[ "${PORT}" -gt 65535 ]]; then
  fatal "Invalid port number: ${PORT} (must be 1024-65535)"
fi

# Check if port is in use
if lsof -iTCP:"${PORT}" -sTCP:LISTEN -t >/dev/null 2>&1; then
  warning "Port ${PORT} is already in use"
  if [[ "${UNATTENDED}" == false ]]; then
    read -rp "Continue anyway? [y/N] " response
    if [[ ! "${response}" =~ ^[Yy]$ ]]; then
      echo "Installation cancelled."
      exit 0
    fi
  fi
fi

# Create log directory
info "Creating log directory..."
mkdir -p "${LOG_DIR}"
success "Log directory: ${LOG_DIR}"

# Find template file
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEMPLATE_FILE="${SCRIPT_DIR}/${PLIST_FILENAME}.template"

if [[ ! -f "${TEMPLATE_FILE}" ]]; then
  fatal "Template file not found: ${TEMPLATE_FILE}"
fi

# Generate plist from template
info "Generating LaunchAgent plist..."
sed -e "s|__HEADROOM_PATH__|${HEADROOM_PATH}|g" \
  -e "s|__PORT__|${PORT}|g" \
  -e "s|__HOME__|${HOME}|g" \
  "${TEMPLATE_FILE}" >"${PLIST_DEST}"

success "Created: ${PLIST_DEST}"

# Set correct permissions
chmod 644 "${PLIST_DEST}"

# Load the LaunchAgent
info "Loading LaunchAgent..."
if launchctl bootstrap "gui/${USER_UID}" "${PLIST_DEST}" 2>/dev/null; then
  success "LaunchAgent loaded successfully"
else
  # If bootstrap fails, try to bootout first in case it was already loaded
  launchctl bootout "gui/${USER_UID}/${PLIST_LABEL}" 2>/dev/null || true
  if launchctl bootstrap "gui/${USER_UID}" "${PLIST_DEST}"; then
    success "LaunchAgent loaded successfully"
  else
    fatal "Failed to load LaunchAgent. Check logs at: ${LOG_DIR}"
  fi
fi

# Wait a moment for service to start
sleep 2

# Verify the service is running
info "Verifying service status..."
if launchctl print "gui/${USER_UID}/${PLIST_LABEL}" >/dev/null 2>&1; then
  success "Service is running"

  # Check if port is listening
  if lsof -iTCP:"${PORT}" -sTCP:LISTEN -t >/dev/null 2>&1; then
    success "Port ${PORT} is listening"
  else
    warning "Service is running but port ${PORT} is not listening yet"
    warning "Check logs: tail -f ${LOG_DIR}/proxy-error.log"
  fi
else
  fatal "Service failed to start. Check logs at: ${LOG_DIR}"
fi

# Display success message
echo ""
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}✓ Headroom proxy installed successfully!${NC}"
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""
echo "Service details:"
echo "  • Port: ${PORT}"
echo "  • Logs: ${LOG_DIR}"
echo "  • Label: ${PLIST_LABEL}"
echo ""
echo "Shell integration:"
echo "  Add to ~/.bashrc or ~/.zshrc:"
echo ""
echo "    export HEADROOM_PROXY_PORT=${PORT}"
echo "    source <path-to>/shell-integration.sh"
echo ""
echo "  Or manually set:"
echo ""
echo "    export ANTHROPIC_BASE_URL=http://localhost:${PORT}"
echo ""
echo "Useful commands:"
echo "  • Check status:  launchctl print gui/\${USER_UID}/${PLIST_LABEL}"
echo "  • View logs:     tail -f ${LOG_DIR}/proxy.log"
echo "  • View errors:   tail -f ${LOG_DIR}/proxy-error.log"
echo "  • Restart:       launchctl kickstart -k gui/\${USER_UID}/${PLIST_LABEL}"
echo "  • Uninstall:     ./uninstall.sh"
echo ""
