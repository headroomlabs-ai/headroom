#!/usr/bin/env bash
#
# Headroom Proxy LaunchAgent Uninstaller for macOS
#
# This script removes the headroom proxy LaunchAgent and optionally cleans up logs.
#
# Usage: ./uninstall.sh [--remove-logs]
#
# Options:
#   --remove-logs     Remove log directory (prompts if not specified)
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
PLIST_PATH="${HOME}/Library/LaunchAgents/${PLIST_FILENAME}"
LOG_DIR="${HOME}/Library/Logs/headroom"
USER_UID=$(id -u)

# Parse command line arguments
REMOVE_LOGS=false
UNATTENDED=false
while [[ $# -gt 0 ]]; do
  case $1 in
    --remove-logs)
      REMOVE_LOGS=true
      shift
      ;;
    --unattended)
      UNATTENDED=true
      shift
      ;;
    *)
      echo "Unknown option: $1"
      echo "Usage: $0 [--remove-logs] [--unattended]"
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

# Check if we're on macOS
OS_NAME=$(uname -s)
if [[ "${OS_NAME}" != "Darwin" ]]; then
  error "This script is only for macOS."
  exit 1
fi

# Check if LaunchAgent is installed
if [[ ! -f "${PLIST_PATH}" ]]; then
  warning "LaunchAgent not found at: ${PLIST_PATH}"
  warning "Nothing to uninstall"
  exit 0
fi

# Check if service is running and stop it
info "Checking service status..."
if launchctl print "gui/${USER_UID}/${PLIST_LABEL}" >/dev/null 2>&1; then
  info "Stopping service..."
  if launchctl bootout "gui/${USER_UID}/${PLIST_LABEL}" 2>/dev/null; then
    success "Service stopped"
  else
    warning "Could not stop service (it may not be running)"
  fi
else
  info "Service is not running"
fi

# Remove plist file
info "Removing LaunchAgent plist..."
if rm -f "${PLIST_PATH}"; then
  success "Removed: ${PLIST_PATH}"
else
  error "Failed to remove: ${PLIST_PATH}"
  exit 1
fi

# Handle log directory
if [[ -d "${LOG_DIR}" ]]; then
  if [[ "${REMOVE_LOGS}" == true ]]; then
    info "Removing log directory..."
    if rm -rf "${LOG_DIR}"; then
      success "Removed: ${LOG_DIR}"
    else
      warning "Failed to remove: ${LOG_DIR}"
    fi
  elif [[ "${UNATTENDED}" == false ]]; then
    read -rp "Remove log directory at ${LOG_DIR}? [y/N] " response
    if [[ "${response}" =~ ^[Yy]$ ]]; then
      if rm -rf "${LOG_DIR}"; then
        success "Removed: ${LOG_DIR}"
      else
        warning "Failed to remove: ${LOG_DIR}"
      fi
    else
      info "Log directory preserved at: ${LOG_DIR}"
    fi
  else
    info "Log directory preserved at: ${LOG_DIR}"
  fi
else
  info "No log directory found"
fi

# Display success message
echo ""
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}✓ Headroom proxy uninstalled successfully!${NC}"
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""
echo "Next steps:"
echo "  • Remove shell integration from ~/.bashrc or ~/.zshrc"
echo "  • Remove or comment out: export ANTHROPIC_BASE_URL=..."
echo "  • Remove or comment out: source <path-to>/shell-integration.sh"
echo ""
if [[ "${REMOVE_LOGS}" == false ]] && [[ -d "${LOG_DIR}" ]]; then
  echo "Log directory still exists at: ${LOG_DIR}"
  echo "  • Remove manually with: rm -rf ${LOG_DIR}"
  echo ""
fi
