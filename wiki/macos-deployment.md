# macOS Deployment Guide

This guide covers deploying the headroom proxy server as a background service on macOS using LaunchAgent. The service will start automatically on login and restart on crash.

## Overview

macOS LaunchAgent provides a native way to run background services with:

- **Automatic startup** on user login
- **Crash recovery** with automatic restart
- **Standard logging** to `~/Library/Logs/`
- **Native lifecycle management** via `launchctl`

This is ideal for local development environments where you want "set and forget" proxy configuration.

## Prerequisites

- macOS 10.13+ (High Sierra or later)
- headroom-ai installed with proxy support
- Anthropic API key configured

### Installing Headroom with Proxy Support

```bash
# Install with proxy support
pip install headroom-ai[proxy]

# Verify installation
headroom proxy --help
```

### API Key Configuration

Your Anthropic API key can be configured in several ways:

**Option 1: Shell environment (recommended)**

```bash
# Add to ~/.bashrc or ~/.zshrc
export ANTHROPIC_API_KEY="sk-ant-..."
```

**Option 2: LaunchAgent plist**

```xml
<key>EnvironmentVariables</key>
<dict>
    <key>ANTHROPIC_API_KEY</key>
    <string>sk-ant-...</string>
</dict>
```

**Option 3: System environment**

```bash
# Add to /etc/launchd.conf (requires admin)
setenv ANTHROPIC_API_KEY sk-ant-...
```

## Quick Install

The automated installer handles all setup:

```bash
# Clone or navigate to headroom repository
cd examples/deployment/macos-launchagent

# Run installer
./install.sh
```

The installer will:

1. Detect your headroom installation
2. Prompt for port configuration (default: 8787)
3. Create log directory
4. Generate LaunchAgent plist
5. Load and start the service
6. Verify service is running

### Installation Options

**Custom port:**

```bash
./install.sh --port 9000
```

**Unattended install (no prompts):**

```bash
./install.sh --port 8787 --unattended
```

**Reinstall over existing:**

```bash
# Installer will prompt to reinstall if service exists
./install.sh
```

## Manual Installation

If you prefer full control over the installation:

### Step 1: Create Log Directory

```bash
mkdir -p ~/Library/Logs/headroom
```

### Step 2: Generate LaunchAgent Plist

Copy and customize the template:

```bash
cd examples/deployment/macos-launchagent
cp com.headroom.proxy.plist.template ~/Library/LaunchAgents/com.headroom.proxy.plist
```

Edit `~/Library/LaunchAgents/com.headroom.proxy.plist`:

1. Replace `__HEADROOM_PATH__` with your headroom path:

   ```bash
   command -v headroom
   # Example output: /usr/local/bin/headroom
   ```

2. Replace `__PORT__` with your desired port (e.g., `8787`)

3. Replace `__HOME__` with your home directory:

   ```bash
   echo $HOME
   # Example output: /Users/yourusername
   ```

### Step 3: Load the LaunchAgent

```bash
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.headroom.proxy.plist
```

### Step 4: Verify Service

```bash
# Check if service is running
launchctl print gui/$(id -u)/com.headroom.proxy

# Check if port is listening
lsof -iTCP:8787 -sTCP:LISTEN

# Test health endpoint
curl http://localhost:8787/health
```

## Configuration

### Port Customization

The default port is 8787. To use a custom port:

**During installation:**

```bash
./install.sh --port 9000
```

**After installation:**

1. Uninstall: `./uninstall.sh`
2. Reinstall with new port: `./install.sh --port 9000`
3. Update shell integration: `export HEADROOM_PORT=9000`

### Log Location

Logs are written to standard macOS locations:

- **Standard output**: `~/Library/Logs/headroom/proxy.log`
- **Error output**: `~/Library/Logs/headroom/proxy-error.log`

To change log locations, edit the plist:

```xml
<key>StandardOutPath</key>
<string>/custom/path/proxy.log</string>
```

### Environment Variables

Configure additional options in the plist `EnvironmentVariables` section:

```xml
<key>EnvironmentVariables</key>
<dict>
    <!-- Required: Proxy port -->
    <key>HEADROOM_PORT</key>
    <string>8787</string>

    <!-- Optional: API key (or set in shell) -->
    <key>ANTHROPIC_API_KEY</key>
    <string>sk-ant-...</string>

</dict>
```

**Note:** The earlier LLMLingua-2 launch-agent variables
(`HEADROOM_COMPRESSION_PROVIDER=llmlingua`, `HEADROOM_LLMLINGUA_DEVICE`,
the `headroom-ai[llmlingua]` extra) were retired with the
`--llmlingua` flag. For ML compression today, install the `[ml]`
extra and follow `wiki/transforms.md`.

### Crash Recovery

The LaunchAgent is configured with:

- **KeepAlive**: Automatically restarts on crash
- **ThrottleInterval**: 10 seconds between restart attempts

To disable automatic restart, edit the plist:

```xml
<key>KeepAlive</key>
<false/>
```

## Shell Integration

Automatically configure your shell to use the proxy when available.

### Setup

Add to `~/.bashrc` (bash) or `~/.zshrc` (zsh):

```bash
# Configure port (optional, defaults to 8787)
export HEADROOM_PORT=8787

# Source shell integration
source /path/to/headroom/examples/deployment/macos-launchagent/shell-integration.sh
```

### What It Does

The shell integration script:

1. Checks if proxy is running on configured port
2. If running, sets `ANTHROPIC_BASE_URL=http://localhost:8787`
3. If not running, attempts to start the LaunchAgent
4. Provides status messages on first load

This makes Claude clients automatically use the proxy without manual configuration.

### Manual Configuration

If you prefer not to use shell integration:

```bash
# Add to ~/.bashrc or ~/.zshrc
export ANTHROPIC_BASE_URL=http://localhost:8787
```

## Service Management

### Check Status

```bash
# View service status
launchctl print gui/$(id -u)/com.headroom.proxy

# Check if port is listening
lsof -iTCP:8787 -sTCP:LISTEN

# Test health endpoint
curl http://localhost:8787/health
```

### View Logs

```bash
# Tail standard output
tail -f ~/Library/Logs/headroom/proxy.log

# Tail error output
tail -f ~/Library/Logs/headroom/proxy-error.log

# View last 50 lines
tail -n 50 ~/Library/Logs/headroom/proxy-error.log
```

### Restart Service

```bash
# Graceful restart (stop and let KeepAlive restart it)
launchctl kickstart -k gui/$(id -u)/com.headroom.proxy

# Manual stop/start
launchctl bootout gui/$(id -u)/com.headroom.proxy
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.headroom.proxy.plist
```

### Stop Service Temporarily

```bash
# Disable without uninstalling
launchctl disable gui/$(id -u)/com.headroom.proxy

# Re-enable
launchctl enable gui/$(id -u)/com.headroom.proxy
```

## Verification

After installation, verify everything is working:

### 1. Check Service Status

```bash
launchctl print gui/$(id -u)/com.headroom.proxy
```

Expected output includes:

```
state = running
```

### 2. Check Port

```bash
lsof -iTCP:8787 -sTCP:LISTEN
```

Should show headroom listening on port 8787.

### 3. Test Health Endpoint

```bash
curl http://localhost:8787/health
```

Expected response:

```json
{"status": "healthy"}
```

### 4. Test Proxy Functionality

```bash
# Set base URL
export ANTHROPIC_BASE_URL=http://localhost:8787

# Test with Python
python -c "
import anthropic
client = anthropic.Anthropic()
response = client.messages.create(
    model='claude-3-5-sonnet-20241022',
    max_tokens=50,
    messages=[{'role': 'user', 'content': 'Hi'}]
)
print(response.content[0].text)
"
```

### 5. Check Logs for Errors

```bash
tail -n 20 ~/Library/Logs/headroom/proxy-error.log
```

Should show no errors. Common startup errors are listed in [Troubleshooting](#troubleshooting).

## Troubleshooting

### Service Won't Start

**Symptom:** `launchctl print` shows service not loaded or failed state

**Check logs:**

```bash
tail -n 50 ~/Library/Logs/headroom/proxy-error.log
```

**Common causes:**

| Error | Solution |
|-------|----------|
| `ANTHROPIC_API_KEY not set` | Set API key in environment or plist |
| `ModuleNotFoundError: No module named 'headroom'` | Install: `pip install headroom-ai[proxy]` |
| `command not found: headroom` | Update plist with correct path: `command -v headroom` |
| `Address already in use` | Change port or stop conflicting service |

### Port Already in Use

**Symptom:** Service starts but port not listening, logs show "Address already in use"

**Find what's using the port:**

```bash
lsof -iTCP:8787 -sTCP:LISTEN
```

**Solutions:**

1. Stop conflicting service
2. Use different port: `./uninstall.sh && ./install.sh --port 9000`

### Service Crashes Immediately

**Symptom:** Service starts but immediately exits

**Check for Python errors:**

```bash
tail -f ~/Library/Logs/headroom/proxy-error.log
```

**Common causes:**

- Missing dependencies: `pip install headroom-ai[proxy]`
- Invalid API key: Verify `ANTHROPIC_API_KEY`
- Python version incompatible: Requires Python 3.9+

### ANTHROPIC_BASE_URL Not Set

**Symptom:** Shell integration not setting environment variable

**Verify proxy is running:**

```bash
curl http://localhost:8787/health
```

**Reload shell configuration:**

```bash
source ~/.bashrc  # or ~/.zshrc
```

**Check shell integration is sourced:**

```bash
# Should be set to 1
echo $HEADROOM_SHELL_INTEGRATION_LOADED
```

### Service Not Auto-Starting on Login

**Symptom:** Service doesn't start after reboot

**Verify LaunchAgent is loaded:**

```bash
launchctl list | grep headroom
```

**If not listed:**

```bash
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.headroom.proxy.plist
```

**Check RunAtLoad is enabled:**

```bash
grep -A1 RunAtLoad ~/Library/LaunchAgents/com.headroom.proxy.plist
```

Should show:

```xml
<key>RunAtLoad</key>
<true/>
```

### Permission Issues

**Symptom:** "Operation not permitted" errors

**Ensure plist has correct permissions:**

```bash
chmod 644 ~/Library/LaunchAgents/com.headroom.proxy.plist
```

**Verify ownership:**

```bash
ls -l ~/Library/LaunchAgents/com.headroom.proxy.plist
```

Should be owned by your user, not root.

## Uninstallation

### Quick Uninstall

```bash
cd examples/deployment/macos-launchagent
./uninstall.sh
```

This will:

1. Stop the service
2. Remove LaunchAgent plist
3. Optionally remove log directory (prompts)

### Remove Everything

```bash
# Uninstall service and remove logs
./uninstall.sh --remove-logs

# Remove shell integration from ~/.bashrc or ~/.zshrc
# Delete or comment out:
#   export HEADROOM_PORT=8787
#   source .../shell-integration.sh
```

### Manual Uninstall

```bash
# Stop service
launchctl bootout gui/$(id -u)/com.headroom.proxy

# Remove plist
rm ~/Library/LaunchAgents/com.headroom.proxy.plist

# Remove logs (optional)
rm -rf ~/Library/Logs/headroom
```

## Production Deployment

For production environments, consider:

- **System-wide LaunchDaemon** instead of per-user LaunchAgent
- **Resource limits** in plist (CPU, memory)
- **Log rotation** for long-running deployments
- **Monitoring** via external tools
- **Multiple instances** on different ports for redundancy

LaunchAgent is designed for single-user development. For production, evaluate:

- [Docker deployment](proxy.md#docker-deployment) for containerized environments
- systemd on Linux servers
- Cloud-native solutions (ECS, Cloud Run, etc.)

## Related Documentation

- [Proxy Server Documentation](proxy.md) - Core proxy configuration and features
- [Configuration Guide](configuration.md) - Detailed configuration options
- [Architecture](ARCHITECTURE.md) - How Headroom works internally
- [Troubleshooting](troubleshooting.md) - General troubleshooting guide

## Platform Alternatives

- **Linux**: Use systemd instead of LaunchAgent
- **Windows**: Use Task Scheduler or NSSM (Non-Sucking Service Manager)
- **Docker**: See [proxy.md](proxy.md) for containerized deployment

## Security Considerations

### LaunchAgent vs LaunchDaemon

**LaunchAgent** (used here):

- Runs in user context
- No root privileges required
- Starts on user login
- Per-user isolation

**LaunchDaemon** (not covered):

- Runs as root or specific user
- System-wide service
- Starts on boot
- Requires admin privileges

For single-user development, LaunchAgent is recommended for security.

### API Key Security

Store API keys securely:

- ✅ Use environment variables in shell config
- ✅ Use macOS Keychain (advanced)
- ✅ Restrict plist file permissions: `chmod 600`
- ❌ Don't commit API keys to version control
- ❌ Don't store in world-readable files

### Network Security

The proxy binds to `127.0.0.1` (localhost only) by default:

- ✅ Only accessible from local machine
- ✅ No external network exposure
- ❌ Don't bind to `0.0.0.0` without firewall rules

## Advanced Configuration

### Multiple Proxy Instances

Run multiple proxies on different ports:

```bash
# Install first instance
./install.sh --port 8787

# For second instance, manually create plist with different label
cp com.headroom.proxy.plist.template ~/Library/LaunchAgents/com.headroom.proxy-2.plist
# Edit: Change Label to com.headroom.proxy-2, port to 8788
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.headroom.proxy-2.plist
```

### Custom LaunchAgent Schedule

Run proxy only during business hours:

```xml
<!-- Add to plist -->
<key>StartCalendarInterval</key>
<dict>
    <key>Hour</key>
    <integer>9</integer>
    <key>Minute</key>
    <integer>0</integer>
</dict>
```

### Resource Limits

Limit CPU and memory usage:

```xml
<!-- Add to plist -->
<key>HardResourceLimits</key>
<dict>
    <key>NumberOfProcesses</key>
    <integer>1</integer>
    <key>MemoryMax</key>
    <integer>536870912</integer> <!-- 512 MB -->
</dict>
```

## Apple GPU (MPS) Embedding Offload

On Apple Silicon, the proxy's memory embedder can run on the Apple GPU (MPS)
instead of the default ONNX CPU backend. Offloading embedding to the GPU frees
the CPU under load, keeping the proxy responsive — useful on fanless Macs (e.g.
the M5 Air) that are prone to CPU-saturation timeouts.

Enable it by installing the extra and setting the env var:

```bash
pip install 'headroom-ai[pytorch-mps]'   # also works as [pytorch_mps]
export HEADROOM_EMBEDDER_RUNTIME=pytorch_mps
```

Under a LaunchAgent, set the env var in the plist `EnvironmentVariables`
section:

```xml
<key>HEADROOM_EMBEDDER_RUNTIME</key>
<string>pytorch_mps</string>
```

It only engages when Apple MPS is actually available (Apple Silicon + torch).
If MPS is unavailable or the dependencies are missing, the proxy logs a warning
and uses the existing default embedder selection path. This is strictly opt-in;
default behavior is unchanged. See [Memory](memory.md#embedding-runtime--gpu-offload-apple-silicon)
for details.

## FAQ

**Q: Why LaunchAgent instead of running `headroom proxy` manually?**

A: LaunchAgent provides automatic startup, crash recovery, and proper lifecycle management. You don't have to remember to start the proxy or keep a terminal window open.

**Q: Can I use this in production?**

A: LaunchAgent is designed for development. For production, use Docker, systemd, or cloud-native deployment.

**Q: How much does the proxy impact performance?**

A: Minimal. The proxy adds ~10-50ms latency while reducing token costs by 50-90%. The cost savings far outweigh the latency.

**Q: Do I need to restart the proxy when configuration changes?**

A: Yes. After changing the plist, reload the service:

```bash
launchctl kickstart -k gui/$(id -u)/com.headroom.proxy
```

**Q: Can I use this with multiple API providers?**

A: The LaunchAgent setup is Anthropic-specific. For other providers, see [proxy.md](proxy.md) for configuration options.

**Q: Does this work with Apple Silicon (M1/M2/M3)?**

A: Yes, fully compatible. ML compression (Kompress, opt-in via `headroom-ai[ml]`) auto-detects MPS on Apple Silicon.
