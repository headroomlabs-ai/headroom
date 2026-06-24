/**
 * Canonical filesystem contract for Headroom — parity shell for the npm SDK.
 *
 * The TypeScript SDK is an HTTP client today and does not touch the
 * filesystem directly. This module mirrors `headroom/paths.py` so that
 * future local features (e.g. cache/log co-location with the Python
 * proxy) land on the same contract.
 *
 * Two canonical roots:
 *   - HEADROOM_CONFIG_DIR     — read-mostly configuration
 *                               (default: ~/.headroom/config)
 *   - HEADROOM_WORKSPACE_DIR  — read-write state
 *                               (default: ~/.headroom)
 *
 * Precedence for every per-resource helper is:
 *   explicit argument > per-resource env var > derived from canonical
 *   root > default.
 *
 * Browser behavior: when `process` is not available (typeof process ===
 * "undefined"), all helpers return the empty string. Consumers running
 * in a browser should not call these helpers; they exist here so the
 * shape of the API matches Python's `headroom.paths` module.
 */

// ---------------------------------------------------------------------------
// Env var names
// ---------------------------------------------------------------------------

export const HEADROOM_CONFIG_DIR_ENV = "HEADROOM_CONFIG_DIR";
export const HEADROOM_WORKSPACE_DIR_ENV = "HEADROOM_WORKSPACE_DIR";

export const HEADROOM_SAVINGS_PATH_ENV = "HEADROOM_SAVINGS_PATH";
export const HEADROOM_TOIN_PATH_ENV = "HEADROOM_TOIN_PATH";
export const HEADROOM_SUBSCRIPTION_STATE_PATH_ENV =
  "HEADROOM_SUBSCRIPTION_STATE_PATH";

// ---------------------------------------------------------------------------
// Node / browser guard (mirrors the pattern in client.ts::getEnv)
// ---------------------------------------------------------------------------

function isNode(): boolean {
  return (
    typeof process !== "undefined" &&
    typeof process.versions !== "undefined" &&
    typeof process.versions.node === "string"
  );
}

function getEnv(name: string): string {
  if (typeof process === "undefined" || !process.env) {
    return "";
  }
  const value = process.env[name];
  return typeof value === "string" ? value.trim() : "";
}

function homeDir(): string {
  if (!isNode()) {
    return "";
  }
  // Prefer the userInfo API, fall back to HOME / USERPROFILE env vars.
  try {
    // eslint-disable-next-line @typescript-eslint/no-var-requires
    const os = require("os") as { homedir?: () => string };
    if (os.homedir) {
      const home = os.homedir();
      if (home) return home;
    }
  } catch {
    // ignore — fall back to env
  }
  return getEnv("HOME") || getEnv("USERPROFILE") || "";
}

function joinPath(...parts: string[]): string {
  if (!isNode()) {
    // Browser fallback — keep it simple, use forward slashes.
    return parts.filter((p) => p !== "").join("/");
  }
  try {
    // eslint-disable-next-line @typescript-eslint/no-var-requires
    const path = require("path") as {
      join?: (...p: string[]) => string;
    };
    if (path.join) return path.join(...parts.filter((p) => p !== ""));
  } catch {
    // ignore — fall back
  }
  const sep = process.platform === "win32" ? "\\" : "/";
  return parts.filter((p) => p !== "").join(sep);
}

function expandTilde(p: string): string {
  if (!p.startsWith("~")) return p;
  const home = homeDir();
  if (!home) return p;
  if (p === "~") return home;
  if (p.startsWith("~/") || p.startsWith("~\\")) {
    return joinPath(home, p.slice(2));
  }
  return p;
}

function resolve(
  explicit: string | undefined,
  envVar: string,
  derived: string,
): string {
  if (!isNode()) {
    // Browser fallback: no filesystem, return empty string.
    return "";
  }
  if (explicit !== undefined && explicit !== "") {
    return expandTilde(explicit);
  }
  const envValue = getEnv(envVar);
  if (envValue) {
    return expandTilde(envValue);
  }
  return derived;
}

// ---------------------------------------------------------------------------
// Canonical roots
// ---------------------------------------------------------------------------

export function workspaceDir(): string {
  if (!isNode()) return "";
  const envValue = getEnv(HEADROOM_WORKSPACE_DIR_ENV);
  if (envValue) return expandTilde(envValue);
  const home = homeDir();
  if (!home) return "";
  return joinPath(home, ".headroom");
}

export function configDir(): string {
  if (!isNode()) return "";
  const envValue = getEnv(HEADROOM_CONFIG_DIR_ENV);
  if (envValue) return expandTilde(envValue);
  const workspaceEnv = getEnv(HEADROOM_WORKSPACE_DIR_ENV);
  if (workspaceEnv) {
    return joinPath(expandTilde(workspaceEnv), "config");
  }
  const home = homeDir();
  if (!home) return "";
  return joinPath(home, ".headroom", "config");
}

// ---------------------------------------------------------------------------
// Per-resource helpers -- workspace bucket
// ---------------------------------------------------------------------------

export function savingsPath(explicit?: string): string {
  return resolve(
    explicit,
    HEADROOM_SAVINGS_PATH_ENV,
    joinPath(workspaceDir(), "proxy_savings.json"),
  );
}

export function toinPath(explicit?: string): string {
  return resolve(
    explicit,
    HEADROOM_TOIN_PATH_ENV,
    joinPath(workspaceDir(), "toin.json"),
  );
}

export function subscriptionStatePath(explicit?: string): string {
  return resolve(
    explicit,
    HEADROOM_SUBSCRIPTION_STATE_PATH_ENV,
    joinPath(workspaceDir(), "subscription_state.json"),
  );
}

export function memoryDbPath(): string {
  if (!isNode()) return "";
  return joinPath(workspaceDir(), "memory.db");
}

export function nativeMemoryDir(): string {
  if (!isNode()) return "";
  return joinPath(workspaceDir(), "memories");
}

export function licenseCachePath(): string {
  if (!isNode()) return "";
  return joinPath(workspaceDir(), "license_cache.json");
}

export function sessionStatsPath(): string {
  if (!isNode()) return "";
  return joinPath(workspaceDir(), "session_stats.jsonl");
}

export function syncStatePath(): string {
  if (!isNode()) return "";
  return joinPath(workspaceDir(), "sync_state.json");
}

export function bridgeStatePath(): string {
  if (!isNode()) return "";
  return joinPath(workspaceDir(), "bridge_state.json");
}

export function logDir(): string {
  if (!isNode()) return "";
  return joinPath(workspaceDir(), "logs");
}

export function proxyLogPath(): string {
  if (!isNode()) return "";
  return joinPath(logDir(), "proxy.log");
}

export function debug400Dir(): string {
  if (!isNode()) return "";
  return joinPath(logDir(), "debug_400");
}

export function binDir(): string {
  if (!isNode()) return "";
  return joinPath(workspaceDir(), "bin");
}

export function rtkPath(): string {
  if (!isNode()) return "";
  const name = process.platform === "win32" ? "rtk.exe" : "rtk";
  return joinPath(binDir(), name);
}

export function deployRoot(): string {
  if (!isNode()) return "";
  return joinPath(workspaceDir(), "deploy");
}

export function beaconLockPath(port: number): string {
  if (!isNode()) return "";
  return joinPath(workspaceDir(), `.beacon_lock_${Math.trunc(port)}`);
}

// ---------------------------------------------------------------------------
// Per-resource helpers -- config bucket
// ---------------------------------------------------------------------------

export function modelsConfigPath(): string {
  if (!isNode()) return "";
  return joinPath(configDir(), "models.json");
}

// ---------------------------------------------------------------------------
// Plugin-author entry points
// ---------------------------------------------------------------------------

function assertPluginName(name: string): void {
  if (!name || name.includes("/") || name.includes("\\")) {
    throw new Error(`invalid plugin name: ${JSON.stringify(name)}`);
  }
}

export function pluginConfigDir(pluginName: string): string {
  assertPluginName(pluginName);
  if (!isNode()) return "";
  return joinPath(configDir(), "plugins", pluginName);
}

export function pluginWorkspaceDir(pluginName: string): string {
  assertPluginName(pluginName);
  if (!isNode()) return "";
  return joinPath(workspaceDir(), "plugins", pluginName);
}
