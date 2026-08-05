# macOS LaunchAgent Deployment

This directory contains templates and scripts for running the headroom proxy server as a persistent background service on macOS using LaunchAgent.

## Quick Start

```bash
# Install the proxy service
./install.sh

# Add shell integration to ~/.bashrc or ~/.zshrc
export HEADROOM_PORT=8787
source /path/to/shell-integration.sh
```

## Files

- **com.headroom.proxy.plist.template**: LaunchAgent plist template
- **install.sh**: Automated installation script
- **uninstall.sh**: Automated removal script
- **shell-integration.sh**: Shell integration for automatic ANTHROPIC_BASE_URL configuration

## Features

- **Automatic Startup**: Service starts on user login
- **Crash Recovery**: Automatically restarts if the proxy crashes
- **Configurable Port**: Default 8787, customizable during installation
- **Standard Logging**: Logs to `~/Library/Logs/headroom/`
- **Shell Integration**: Automatically sets `ANTHROPIC_BASE_URL` for Claude clients

## Requirements

- macOS 10.13+ (High Sierra or later)
- headroom-ai installed with proxy support: `pip install headroom-ai[proxy]`
- Anthropic API key configured in environment

## Installation Options

### Quick Install (Recommended)

```bash
./install.sh
```

### Custom Port

```bash
./install.sh --port 9000
```

### Unattended Install

```bash
./install.sh --port 8787 --unattended
```

## Verification

Check if the service is running:

```bash
# Check LaunchAgent status
launchctl print gui/$(id -u)/com.headroom.proxy

# Check if port is listening
lsof -iTCP:8787 -sTCP:LISTEN

# Test health endpoint
curl http://localhost:8787/health
```

## Logs

View logs:

```bash
# Standard output
tail -f ~/Library/Logs/headroom/proxy.log

# Error output
tail -f ~/Library/Logs/headroom/proxy-error.log
```

## Uninstallation

```bash
# Remove service only
./uninstall.sh

# Remove service and logs
./uninstall.sh --remove-logs
```

## Troubleshooting

### Service won't start

Check logs for errors:

```bash
tail -n 50 ~/Library/Logs/headroom/proxy-error.log
```

Common causes:

- Missing ANTHROPIC_API_KEY environment variable
- Port already in use
- headroom not installed with proxy support

### Port already in use

Find what's using the port:

```bash
lsof -iTCP:8787 -sTCP:LISTEN
```

Change to a different port:

```bash
./uninstall.sh
./install.sh --port 9000
```

### Service not auto-starting

Verify LaunchAgent is loaded:

```bash
launchctl list | grep headroom
```

If not loaded:

```bash
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.headroom.proxy.plist
```

## Manual Installation

If you prefer manual installation:

1. Copy template and customize:

   ```bash
   cp com.headroom.proxy.plist.template ~/Library/LaunchAgents/com.headroom.proxy.plist
   ```

2. Edit the plist file:
   - Replace `__HEADROOM_PATH__` with output of `command -v headroom`
   - Replace `__PORT__` with your desired port
   - Replace `__HOME__` with your home directory path

3. Create log directory:

   ```bash
   mkdir -p ~/Library/Logs/headroom
   ```

4. Load the LaunchAgent:

   ```bash
   launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.headroom.proxy.plist
   ```

## Documentation

For complete documentation, see [wiki/macos-deployment.md](../../../wiki/macos-deployment.md)
