# Docker Containers for Claude Code in VSCode on Windows 10

This guide covers running Claude Code inside Docker containers with VSCode Dev Containers on Windows 10, optimized for the QuantConnect Trading Bot project.

## Overview

Anthropic officially supports running Claude Code inside Docker containers with:
- **Reference devcontainer implementation** at `github.com/anthropics/claude-code/.devcontainer/`
- **Official devcontainer feature** at `ghcr.io/anthropics/devcontainer-features/claude-code:1` (196,000+ downloads)
- **Built-in sandboxing** via `/sandbox` command

## Prerequisites (Windows 10)

1. **Docker Desktop** with WSL2 backend enabled
2. **VSCode** with Dev Containers extension (`ms-vscode-remote.remote-containers`)
3. **WSL2** distribution installed (Ubuntu recommended)

### Performance Recommendation

Store projects in the **WSL filesystem** for significantly better performance:
```
\\wsl$\Ubuntu\home\<user>\projects\    (RECOMMENDED - fast)
/mnt/c/...                              (SLOW - cross-filesystem)
```

WSL2 advantages over Hyper-V:
- Cold start: ~10 seconds (vs ~60 seconds)
- Better file watching and hot module replacement

## Quick Start

1. Open this project in VSCode
2. Press `F1` and select **"Dev Containers: Reopen in Container"**
3. Wait for the container to build and start
4. Claude Code is available in the integrated terminal

## Configuration Files

| File | Purpose |
|------|---------|
| `devcontainer.json` | Main container configuration |
| `Dockerfile` | Container image definition |
| `init-firewall.sh` | Network security rules |

## Architecture Options

### Option 1: Docker-in-Docker (Recommended for AI Tools)

Provides complete isolation - inner Docker environment is independent from host:

```json
{
  "features": {
    "ghcr.io/devcontainers/features/docker-in-docker:2": {
      "version": "latest",
      "moby": true,
      "enableNonRootDocker": true
    }
  },
  "privileged": true
}
```

**Benefits:**
- Containers created by Claude Code exist only within the inner Docker daemon
- Better security isolation for autonomous agents
- No access to host containers

### Option 2: Docker Socket Mounting (Higher Risk)

Lower overhead but grants access to ALL host containers:

```json
{
  "features": {
    "ghcr.io/devcontainers/features/docker-outside-of-docker:1": {}
  },
  "mounts": [
    "source=/var/run/docker.sock,target=/var/run/docker.sock,type=bind"
  ]
}
```

**Security Warning:** Any process with socket access can:
- Execute any Docker command (effectively root access)
- Leak environment variables via `docker inspect`
- Mount host filesystem into new containers (container escape)

## Security Configuration

### Network Isolation with iptables

The `init-firewall.sh` script implements default-deny with whitelisted domains:

```bash
#!/bin/bash
# Allow only specific HTTPS destinations
iptables -A OUTPUT -p tcp --dport 443 -d api.anthropic.com -j ACCEPT
iptables -A OUTPUT -p tcp --dport 443 -d api.quantconnect.com -j ACCEPT
iptables -A OUTPUT -p tcp --dport 443 -d github.com -j ACCEPT
iptables -A OUTPUT -p tcp --dport 443 -j DROP
```

Requires `--cap-add=NET_ADMIN` in `runArgs`.

### Filesystem Isolation Options

**Read-only source mounts:**
```json
{
  "mounts": [
    "source=${localWorkspaceFolder}/src,target=/workspace/src,type=bind,readonly",
    "source=build-output,target=/workspace/dist,type=volume"
  ]
}
```

**Complete workspace isolation:**
```json
{
  "workspaceMount": "",
  "workspaceFolder": "/home/node/isolated-workspace"
}
```

### Resource Limits

Prevent denial-of-service from runaway processes:

```json
{
  "hostRequirements": {
    "cpus": 4,
    "memory": "16gb"
  },
  "runArgs": ["--memory=4g", "--cpus=2", "--pids-limit=100"]
}
```

## Credential Persistence

To prevent re-authentication on every container restart, mount the Claude config directory:

```json
{
  "mounts": [
    "source=${localEnv:HOME}/.claude,target=/home/node/.claude,type=bind"
  ],
  "containerEnv": {
    "CLAUDE_CONFIG_DIR": "/home/node/.claude"
  }
}
```

Or use a named volume for isolation:
```json
{
  "mounts": [
    "source=claude-code-config-${devcontainerId},target=/home/node/.claude,type=volume"
  ]
}
```

## Autonomous/Unattended Operation

For autonomous agent development inside containers, Anthropic supports:

```bash
claude --dangerously-skip-permissions
```

The enhanced security of containerized environment (isolation + firewall rules) justifies bypassing permission prompts.

**Warning from Anthropic:** "Devcontainers do not prevent a malicious project from exfiltrating anything accessible in the devcontainer including Claude Code credentials."

### Alternative: Native Sandbox

Claude Code includes built-in sandboxing without Docker overhead:

```bash
# In Claude Code terminal
/sandbox

# Or via CLI
claude --sandbox
```

Uses Linux bubblewrap / macOS seatbelt OS primitives.

### Docker Desktop Sandbox

```bash
docker sandbox run claude
docker sandbox run claude --continue  # Resume session
```

Pre-configured with Docker CLI, GitHub CLI, Node.js, Go, Python 3, and Git.

## VSCode Command Palette Options

| Command | Description |
|---------|-------------|
| `Dev Containers: Reopen in Container` | Open current folder in container |
| `Dev Containers: Rebuild Container` | Rebuild after config changes |
| `Dev Containers: Clone Repository in Container Volume` | Clone directly into isolated volume |

## Known Limitations

### IDE Integration in Containers

When running Claude Code in containers:
- No access to IDE diagnostics
- No access to currently open files
- No access to selected line ranges
- Operates through **terminal interface only**

To block IDE communication (if needed):
```bash
iptables -A OUTPUT -p udp --dport 59778 -j DROP
```

### Windows 10 Specific Issues

1. **WebDAV Security Warning:** Running Claude Code on Windows with WebDAV enabled may allow network requests to remote hosts that bypass the permission system. Avoid paths like `\\*` containing WebDAV subdirectories.

2. **Cross-Filesystem Performance:** Accessing `/mnt/c/` from WSL breaks hot module replacement and degrades file watching.

3. **WSL2 Networking with JetBrains:** Configure Windows Firewall or use mirrored networking:
   ```ini
   # ~/.wslconfig
   [wsl2]
   networkingMode=mirrored
   ```

4. **Node.js Path Conflicts:** If encountering "spawn node ENOENT" errors, verify Node.js paths aren't conflicting between Windows and WSL.

5. **ARM64 Build Failures:** Avoid hardcoded ARM64 package downloads on x86_64 systems (GitHub issue #62).

## Project-Specific Configuration

This devcontainer is configured for the QuantConnect Trading Bot with:

- **Python 3.10** for algorithm development
- **Node.js 20** for Claude Code
- **LEAN CLI** for backtesting
- **QuantConnect stubs** for IDE support
- **Network access** to QuantConnect API and GitHub
- **Docker-in-Docker** for running isolated backtest containers

### Allowed Network Destinations

| Domain | Purpose |
|--------|---------|
| api.anthropic.com | Claude API |
| api.quantconnect.com | QuantConnect API |
| github.com | Git operations |
| pypi.org | Python packages |
| registry.npmjs.org | NPM packages |

## Troubleshooting

### Container Won't Start

1. Ensure Docker Desktop is running with WSL2 backend
2. Check Docker Desktop settings: Resources > WSL Integration
3. Rebuild: `Dev Containers: Rebuild Container`

### Permission Denied Errors

```bash
# Inside container
sudo chown -R node:node /workspace
```

### Slow File Operations

Move project to WSL filesystem:
```bash
# From Windows
cp -r /mnt/c/Projects/trading-bot ~/projects/
```

### Claude Code Not Found

```bash
# Verify installation
which claude
npm list -g @anthropic-ai/claude-code

# Reinstall if needed
npm install -g @anthropic-ai/claude-code
```

### Network Issues

```bash
# Check firewall rules
sudo iptables -L -v

# Test connectivity
curl -I https://api.anthropic.com
```

## References

- [Anthropic Claude Code Documentation](https://docs.anthropic.com/claude-code)
- [Official Devcontainer Feature](https://github.com/anthropics/devcontainer-features)
- [VSCode Dev Containers](https://code.visualstudio.com/docs/devcontainers/containers)
- [Docker-in-Docker Feature](https://github.com/devcontainers/features/tree/main/src/docker-in-docker)
