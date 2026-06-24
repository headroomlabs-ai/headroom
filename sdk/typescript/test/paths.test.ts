/**
 * Tests for the filesystem contract module — mirrors the Python
 * `tests/test_paths.py` precedence matrix to keep the SDK honest as a
 * parity shell.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import * as os from "os";
import * as path from "path";

import {
  HEADROOM_CONFIG_DIR_ENV,
  HEADROOM_SAVINGS_PATH_ENV,
  HEADROOM_SUBSCRIPTION_STATE_PATH_ENV,
  HEADROOM_TOIN_PATH_ENV,
  HEADROOM_WORKSPACE_DIR_ENV,
  beaconLockPath,
  binDir,
  bridgeStatePath,
  configDir,
  debug400Dir,
  deployRoot,
  licenseCachePath,
  logDir,
  memoryDbPath,
  modelsConfigPath,
  nativeMemoryDir,
  pluginConfigDir,
  pluginWorkspaceDir,
  proxyLogPath,
  rtkPath,
  savingsPath,
  sessionStatsPath,
  subscriptionStatePath,
  syncStatePath,
  toinPath,
  workspaceDir,
} from "../src/paths.js";

// ---------------------------------------------------------------------------
// Env var housekeeping
// ---------------------------------------------------------------------------

const ENV_VARS = [
  HEADROOM_CONFIG_DIR_ENV,
  HEADROOM_WORKSPACE_DIR_ENV,
  HEADROOM_SAVINGS_PATH_ENV,
  HEADROOM_TOIN_PATH_ENV,
  HEADROOM_SUBSCRIPTION_STATE_PATH_ENV,
];

function saveEnv(): Record<string, string | undefined> {
  const snap: Record<string, string | undefined> = {};
  for (const k of ENV_VARS) snap[k] = process.env[k];
  return snap;
}

function restoreEnv(snap: Record<string, string | undefined>): void {
  for (const k of ENV_VARS) {
    if (snap[k] === undefined) {
      delete process.env[k];
    } else {
      process.env[k] = snap[k];
    }
  }
}

function clearEnv(): void {
  for (const k of ENV_VARS) delete process.env[k];
}

// ---------------------------------------------------------------------------
// Canonical roots
// ---------------------------------------------------------------------------

describe("canonical roots", () => {
  let snap: Record<string, string | undefined>;
  beforeEach(() => {
    snap = saveEnv();
    clearEnv();
  });
  afterEach(() => restoreEnv(snap));

  it("workspaceDir defaults to ~/.headroom", () => {
    expect(workspaceDir()).toBe(path.join(os.homedir(), ".headroom"));
  });

  it("workspaceDir honors HEADROOM_WORKSPACE_DIR env override", () => {
    process.env[HEADROOM_WORKSPACE_DIR_ENV] = "/tmp/alt_ws";
    expect(workspaceDir()).toBe("/tmp/alt_ws");
  });

  it("workspaceDir ignores blank env value", () => {
    process.env[HEADROOM_WORKSPACE_DIR_ENV] = "   ";
    expect(workspaceDir()).toBe(path.join(os.homedir(), ".headroom"));
  });

  it("workspaceDir expands tilde", () => {
    process.env[HEADROOM_WORKSPACE_DIR_ENV] = "~/custom-ws";
    expect(workspaceDir()).toBe(path.join(os.homedir(), "custom-ws"));
  });

  it("configDir defaults to ~/.headroom/config", () => {
    expect(configDir()).toBe(path.join(os.homedir(), ".headroom", "config"));
  });

  it("configDir follows HEADROOM_WORKSPACE_DIR when only workspace set", () => {
    process.env[HEADROOM_WORKSPACE_DIR_ENV] = "/tmp/alt_ws";
    expect(configDir()).toBe(path.join("/tmp/alt_ws", "config"));
  });

  it("explicit HEADROOM_CONFIG_DIR env beats workspace env", () => {
    process.env[HEADROOM_WORKSPACE_DIR_ENV] = "/tmp/alt_ws";
    process.env[HEADROOM_CONFIG_DIR_ENV] = "/tmp/alt_cfg";
    expect(configDir()).toBe("/tmp/alt_cfg");
  });
});

// ---------------------------------------------------------------------------
// Per-resource precedence matrix
// ---------------------------------------------------------------------------

type ResourceCase = {
  name: string;
  fn: (explicit?: string) => string;
  envVar: string;
  filename: string;
};

const RESOURCES: ResourceCase[] = [
  {
    name: "savingsPath",
    fn: savingsPath,
    envVar: HEADROOM_SAVINGS_PATH_ENV,
    filename: "proxy_savings.json",
  },
  {
    name: "toinPath",
    fn: toinPath,
    envVar: HEADROOM_TOIN_PATH_ENV,
    filename: "toin.json",
  },
  {
    name: "subscriptionStatePath",
    fn: subscriptionStatePath,
    envVar: HEADROOM_SUBSCRIPTION_STATE_PATH_ENV,
    filename: "subscription_state.json",
  },
];

describe.each(RESOURCES)(
  "resource precedence: $name",
  ({ fn, envVar, filename }) => {
    let snap: Record<string, string | undefined>;
    beforeEach(() => {
      snap = saveEnv();
      clearEnv();
    });
    afterEach(() => restoreEnv(snap));

    it("default under ~/.headroom", () => {
      expect(fn()).toBe(path.join(os.homedir(), ".headroom", filename));
    });

    it("derived from HEADROOM_WORKSPACE_DIR when set", () => {
      process.env[HEADROOM_WORKSPACE_DIR_ENV] = "/tmp/state";
      expect(fn()).toBe(path.join("/tmp/state", filename));
    });

    it("legacy env var wins over workspace-derived", () => {
      process.env[HEADROOM_WORKSPACE_DIR_ENV] = "/tmp/state";
      process.env[envVar] = "/tmp/legacy.json";
      expect(fn()).toBe("/tmp/legacy.json");
    });

    it("explicit arg wins over everything", () => {
      process.env[HEADROOM_WORKSPACE_DIR_ENV] = "/tmp/state";
      process.env[envVar] = "/tmp/legacy.json";
      expect(fn("/tmp/explicit.json")).toBe("/tmp/explicit.json");
    });

    it("explicit empty string falls through to default", () => {
      expect(fn("")).toBe(path.join(os.homedir(), ".headroom", filename));
    });

    it("legacy env expands tilde", () => {
      process.env[envVar] = "~/foo.json";
      expect(fn()).toBe(path.join(os.homedir(), "foo.json"));
    });
  },
);

// ---------------------------------------------------------------------------
// Resources without a legacy env var
// ---------------------------------------------------------------------------

describe("derived-only resources", () => {
  let snap: Record<string, string | undefined>;
  beforeEach(() => {
    snap = saveEnv();
    clearEnv();
  });
  afterEach(() => restoreEnv(snap));

  it("memoryDbPath", () => {
    expect(memoryDbPath()).toBe(
      path.join(os.homedir(), ".headroom", "memory.db"),
    );
  });

  it("nativeMemoryDir", () => {
    expect(nativeMemoryDir()).toBe(
      path.join(os.homedir(), ".headroom", "memories"),
    );
  });

  it("licenseCachePath", () => {
    expect(licenseCachePath()).toBe(
      path.join(os.homedir(), ".headroom", "license_cache.json"),
    );
  });

  it("sessionStatsPath", () => {
    expect(sessionStatsPath()).toBe(
      path.join(os.homedir(), ".headroom", "session_stats.jsonl"),
    );
  });

  it("syncStatePath", () => {
    expect(syncStatePath()).toBe(
      path.join(os.homedir(), ".headroom", "sync_state.json"),
    );
  });

  it("bridgeStatePath", () => {
    expect(bridgeStatePath()).toBe(
      path.join(os.homedir(), ".headroom", "bridge_state.json"),
    );
  });

  it("logDir", () => {
    expect(logDir()).toBe(path.join(os.homedir(), ".headroom", "logs"));
  });

  it("proxyLogPath", () => {
    expect(proxyLogPath()).toBe(
      path.join(os.homedir(), ".headroom", "logs", "proxy.log"),
    );
  });

  it("debug400Dir", () => {
    expect(debug400Dir()).toBe(
      path.join(os.homedir(), ".headroom", "logs", "debug_400"),
    );
  });

  it("binDir", () => {
    expect(binDir()).toBe(path.join(os.homedir(), ".headroom", "bin"));
  });

  it("rtkPath ends with rtk or rtk.exe", () => {
    const p = rtkPath();
    const expected = process.platform === "win32" ? "rtk.exe" : "rtk";
    expect(path.basename(p)).toBe(expected);
  });

  it("deployRoot", () => {
    expect(deployRoot()).toBe(path.join(os.homedir(), ".headroom", "deploy"));
  });

  it("beaconLockPath includes port", () => {
    expect(beaconLockPath(8787)).toBe(
      path.join(os.homedir(), ".headroom", ".beacon_lock_8787"),
    );
  });

  it("modelsConfigPath under configDir", () => {
    expect(modelsConfigPath()).toBe(
      path.join(os.homedir(), ".headroom", "config", "models.json"),
    );
  });

  it("modelsConfigPath follows config env override", () => {
    process.env[HEADROOM_CONFIG_DIR_ENV] = "/tmp/cfg";
    expect(modelsConfigPath()).toBe(path.join("/tmp/cfg", "models.json"));
  });

  it("modelsConfigPath follows workspace env", () => {
    process.env[HEADROOM_WORKSPACE_DIR_ENV] = "/tmp/ws";
    expect(modelsConfigPath()).toBe(
      path.join("/tmp/ws", "config", "models.json"),
    );
  });
});

// ---------------------------------------------------------------------------
// Derived-only helpers must follow HEADROOM_WORKSPACE_DIR end-to-end
// ---------------------------------------------------------------------------

describe("derived-only helpers follow workspace env", () => {
  let snap: Record<string, string | undefined>;
  beforeEach(() => {
    snap = saveEnv();
    clearEnv();
    process.env[HEADROOM_WORKSPACE_DIR_ENV] = "/tmp/alt_ws";
  });
  afterEach(() => restoreEnv(snap));

  it("memoryDbPath", () => {
    expect(memoryDbPath()).toBe(path.join("/tmp/alt_ws", "memory.db"));
  });
  it("nativeMemoryDir", () => {
    expect(nativeMemoryDir()).toBe(path.join("/tmp/alt_ws", "memories"));
  });
  it("licenseCachePath", () => {
    expect(licenseCachePath()).toBe(
      path.join("/tmp/alt_ws", "license_cache.json"),
    );
  });
  it("sessionStatsPath", () => {
    expect(sessionStatsPath()).toBe(
      path.join("/tmp/alt_ws", "session_stats.jsonl"),
    );
  });
  it("syncStatePath", () => {
    expect(syncStatePath()).toBe(
      path.join("/tmp/alt_ws", "sync_state.json"),
    );
  });
  it("bridgeStatePath", () => {
    expect(bridgeStatePath()).toBe(
      path.join("/tmp/alt_ws", "bridge_state.json"),
    );
  });
  it("logDir", () => {
    expect(logDir()).toBe(path.join("/tmp/alt_ws", "logs"));
  });
  it("proxyLogPath", () => {
    expect(proxyLogPath()).toBe(
      path.join("/tmp/alt_ws", "logs", "proxy.log"),
    );
  });
  it("debug400Dir", () => {
    expect(debug400Dir()).toBe(
      path.join("/tmp/alt_ws", "logs", "debug_400"),
    );
  });
  it("binDir", () => {
    expect(binDir()).toBe(path.join("/tmp/alt_ws", "bin"));
  });
  it("rtkPath", () => {
    const expected = process.platform === "win32" ? "rtk.exe" : "rtk";
    expect(rtkPath()).toBe(path.join("/tmp/alt_ws", "bin", expected));
  });
  it("deployRoot", () => {
    expect(deployRoot()).toBe(path.join("/tmp/alt_ws", "deploy"));
  });
  it("beaconLockPath", () => {
    expect(beaconLockPath(9999)).toBe(
      path.join("/tmp/alt_ws", ".beacon_lock_9999"),
    );
  });
  it("pluginConfigDir follows derived config (workspace/config)", () => {
    expect(pluginConfigDir("alpha")).toBe(
      path.join("/tmp/alt_ws", "config", "plugins", "alpha"),
    );
  });
  it("pluginWorkspaceDir", () => {
    expect(pluginWorkspaceDir("alpha")).toBe(
      path.join("/tmp/alt_ws", "plugins", "alpha"),
    );
  });
});

// ---------------------------------------------------------------------------
// Plugin namespace isolation
// ---------------------------------------------------------------------------

describe("plugin dirs", () => {
  let snap: Record<string, string | undefined>;
  beforeEach(() => {
    snap = saveEnv();
    clearEnv();
  });
  afterEach(() => restoreEnv(snap));

  it("pluginConfigDir namespaced under configDir/plugins", () => {
    const a = pluginConfigDir("alpha");
    const b = pluginConfigDir("beta");
    expect(a).not.toBe(b);
    expect(a).toBe(
      path.join(os.homedir(), ".headroom", "config", "plugins", "alpha"),
    );
  });

  it("pluginWorkspaceDir namespaced under workspaceDir/plugins", () => {
    const a = pluginWorkspaceDir("alpha");
    const b = pluginWorkspaceDir("beta");
    expect(a).not.toBe(b);
    expect(a).toBe(
      path.join(os.homedir(), ".headroom", "plugins", "alpha"),
    );
  });

  for (const bad of ["", "foo/bar", "foo\\bar"]) {
    it(`rejects invalid plugin name ${JSON.stringify(bad)}`, () => {
      expect(() => pluginConfigDir(bad)).toThrow();
      expect(() => pluginWorkspaceDir(bad)).toThrow();
    });
  }
});

// ---------------------------------------------------------------------------
// Browser fallback (simulated absence of `process`)
// ---------------------------------------------------------------------------

describe("browser fallback", () => {
  it("all helpers return empty string when Node guard fails", async () => {
    const snap = saveEnv();
    clearEnv();
    const originalDescriptor = Object.getOwnPropertyDescriptor(
      process,
      "versions",
    );
    // Simulate a non-Node runtime by making `process.versions.node` go away.
    // `process.versions` is read-only in Node, so swap via defineProperty.
    Object.defineProperty(process, "versions", {
      value: { v8: "x" } as NodeJS.ProcessVersions,
      configurable: true,
      writable: true,
    });
    try {
      vi.resetModules();
      const mod = await import("../src/paths.js");
      expect(mod.workspaceDir()).toBe("");
      expect(mod.configDir()).toBe("");
      expect(mod.savingsPath()).toBe("");
      expect(mod.toinPath()).toBe("");
      expect(mod.memoryDbPath()).toBe("");
      expect(mod.modelsConfigPath()).toBe("");
    } finally {
      if (originalDescriptor) {
        Object.defineProperty(process, "versions", originalDescriptor);
      }
      restoreEnv(snap);
    }
  });
});
