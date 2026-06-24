# 006. Actors

**Status:** done

## Actor Types

### End User

The developer using an AI coding agent with Headroom.

**Interactions:**
- Configures Headroom via environment variables or config file
- Uses wrapped CLI commands or SDK
- Views savings in dashboard
- Optionally enables learn mode

**Needs:**
- Transparent compression (doesn't break workflows)
- Clear savings metrics
- Easy opt-out of specific features

**Configuration:**
```bash
# Minimal setup
export ANTHROPIC_API_KEY=sk-...

# Optional overrides
export HEADROOM_MODE=token
```

---

### Operator

The person deploying and managing Headroom in production.

**Interactions:**
- Deploys Headroom (Docker, native, or embedded)
- Configures profiles and presets
- Monitors health endpoints
- Reviews metrics and logs
- Manages upgrades

**Needs:**
- Clear deployment documentation
- Health and readiness checks
- Metrics for capacity planning
- Upgrade and rollback procedures

**Configuration:**
```yaml
# ~/.headroom/config.yaml
proxy:
  host: 0.0.0.0
  port: 8787

compression:
  enabled: true
  cache:
    enabled: true
    ttl: 3600

telemetry:
  metrics:
    enabled: true
```

**Health Endpoints:**
```bash
curl http://localhost:8787/health
curl http://localhost:8787/livez
curl http://localhost:8787/readyz
curl http://localhost:8787/metrics
```

---

### Plugin Author

The developer creating a custom learn plugin for a specific agent.

**Interactions:**
- Implements `LearnPlugin` interface
- Plugins auto-discovered from `headroom/learn/plugins/` directory
- Contributes to Headroom

**Needs:**
- Clear plugin interface documentation
- Example plugins to reference
- Test utilities

**Plugin Template:**
```python
from headroom.learn.base import LearnPlugin, ConversationScanner

class MyAgentPlugin(LearnPlugin):
    @property
    def name(self) -> str:
        return "my_agent"

    @property
    def display_name(self) -> str:
        return "My Agent"

    def detect(self) -> bool:
        # Check if this agent has data on the current machine
        pass

    def discover_projects(self) -> list[ProjectInfo]:
        # Discover all projects with conversation data
        pass

    def scan_project(self, project: ProjectInfo, max_workers: int = 1) -> list[SessionData]:
        # Scan all sessions for a project
        pass

    def create_writer(self) -> ContextWriter:
        # Return the appropriate ContextWriter for this agent
        pass
```

---

### Enterprise Evaluator

The person assessing Headroom for organizational adoption.

**Interactions:**
- Reviews security documentation
- Assesses compliance guarantees
- Evaluates operational characteristics

**Needs:**
- Clear data handling guarantees
- Security and privacy documentation
- Compliance certifications (if any)
- SOC2/GDPR considerations

**Security Configuration:**
```bash
# Maximum privacy settings
HEADROOM_TELEMETRY=off
HEADROOM_STATELESS=true
headroom proxy --no-cache --no-optimize
```

---

## Interaction Patterns

### User → Headroom

```
┌──────────────┐       ┌────────────────┐       ┌─────────────┐
│ User's AI    │──────▶│  Headroom      │──────▶│  Provider   │
│ Agent        │       │  Proxy         │       │  API        │
└──────────────┘       └───────┬────────┘       └─────────────┘
                               │
                               ▼
                        ┌──────────────┐
                        │  Dashboard   │
                        │  (optional)  │
                        └──────────────┘
```

**Flow:**
1. User's AI agent sends request to Headroom proxy
2. Headroom compresses context
3. Compressed request sent to provider API
4. Response returned to agent
5. Optional: savings logged to dashboard

---

### Operator → Headroom

```
┌──────────┐       ┌────────────────┐       ┌─────────────┐
│ Operator │──────▶│  Health        │──────▶│  Metrics    │
│          │       │  Endpoints     │       │  Server     │
└──────────┘       └────────────────┘       └─────────────┘
       │
       ▼
┌────────────────┐
│  Logs          │
│  (stdout/file) │
└────────────────┘
```

**Flow:**
1. Operator checks health endpoints
2. Reviews Prometheus metrics
3. Monitors logs for errors
4. Manages configuration

---

### Plugin Author → Headroom

```
┌──────────────┐       ┌────────────────┐       ┌─────────────┐
│ Plugin       │──────▶│  Plugin         │──────▶│  Registry   │
│ Author       │       │  Interface      │       │  + Tests    │
└──────────────┘       └────────────────┘       └─────────────┘
```

**Flow:**
1. Plugin author implements LearnPlugin
2. Plugins are auto-discovered from `headroom/learn/plugins/` directory
3. Writes unit tests
4. Submits contribution

---

## Permissions Model

| Actor | Config | Read Metrics | Admin | Plugin |
|-------|--------|--------------|-------|--------|
| End User | ✓ (own) | ✓ (own) | - | - |
| Operator | ✓ (full) | ✓ (all) | ✓ | - |
| Plugin Author | - | - | - | ✓ (write) |
| Enterprise Evaluator | - | ✓ (security) | - | - |

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0-draft | 2026-04-16 | Initial actors document |
