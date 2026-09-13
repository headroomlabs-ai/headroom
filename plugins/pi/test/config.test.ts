import { mkdtemp, mkdir, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, describe, expect, it } from "vitest";

import {
  DEFAULT_CONFIG,
  loadConfig,
  validateConfig,
} from "../src/config.js";

const originalEnv = { ...process.env };

afterEach(() => {
  process.env = { ...originalEnv };
});

describe("validateConfig", () => {
  it("returns the documented defaults", () => {
    expect(validateConfig({}, "test").config).toEqual(DEFAULT_CONFIG);
  });

  it("falls back per invalid field", () => {
    const result = validateConfig(
      { enabled: false, minResultChars: -1 },
      "test",
    );

    expect(result.config.enabled).toBe(false);
    expect(result.config.minResultChars).toBe(4_000);
    expect(result.warnings).toContain(
      "test.minResultChars must be a positive integer; using 4000",
    );
  });

  it("requires explicit enablement and an exact host allowlist for remote endpoints", () => {
    expect(
      validateConfig({ baseUrl: "http://example.com" }, "test").config.baseUrl,
    ).toBe(DEFAULT_CONFIG.baseUrl);
    expect(
      validateConfig(
        { baseUrl: "https://example.com", allowRemote: true },
        "test",
      ).config.baseUrl,
    ).toBe(DEFAULT_CONFIG.baseUrl);

    const allowed = validateConfig(
      {
        baseUrl: "https://Proxy.Example.com:9443",
        allowRemote: true,
        remoteHosts: [" proxy.example.com ", "proxy.example.com"],
      },
      "test",
    );

    expect(allowed.config.baseUrl).toBe("https://proxy.example.com:9443");
    expect(allowed.config.remoteHosts).toEqual(["proxy.example.com"]);
  });

  it("rejects a remote endpoint absent from the exact host allowlist", () => {
    const result = validateConfig(
      {
        baseUrl: "https://other.example.com",
        allowRemote: true,
        remoteHosts: ["proxy.example.com"],
      },
      "test",
    );

    expect(result.config.baseUrl).toBe(DEFAULT_CONFIG.baseUrl);
    expect(result.warnings).toContain(
      "test.baseUrl host other.example.com is not in remoteHosts; using http://127.0.0.1:8787",
    );
  });

  it("normalizes and deduplicates protected tool names", () => {
    const result = validateConfig(
      { protectedTools: [" Bash ", "bash", "READ"] },
      "test",
    );

    expect(result.config.protectedTools).toEqual([
      ...DEFAULT_CONFIG.protectedTools,
      "bash",
    ]);
  });
});

describe("loadConfig", () => {
  it("applies environment values over JSON values", async () => {
    const homeDir = await mkdtemp(join(tmpdir(), "headroom-pi-config-"));
    const configDir = join(homeDir, ".headroom", "integrations");
    await mkdir(configDir, { recursive: true });
    await writeFile(
      join(configDir, "pi-extension.json"),
      JSON.stringify({ minResultChars: 5_000, enabled: false }),
    );

    const result = await loadConfig(
      {
        HEADROOM_PI_ENABLED: "true",
        HEADROOM_PI_MIN_RESULT_CHARS: "6000",
      },
      homeDir,
    );

    expect(result.config.enabled).toBe(true);
    expect(result.config.minResultChars).toBe(6_000);
    expect(result.warnings).toEqual([]);
  });

  it("parses the remote host allowlist from the environment", async () => {
    const homeDir = await mkdtemp(join(tmpdir(), "headroom-pi-config-"));

    const result = await loadConfig(
      {
        HEADROOM_PI_ALLOW_REMOTE: "true",
        HEADROOM_PI_BASE_URL: "https://proxy.example.com:9443",
        HEADROOM_PI_REMOTE_HOSTS: " proxy.example.com,backup.example.com ",
      },
      homeDir,
    );

    expect(result.config.allowRemote).toBe(true);
    expect(result.config.baseUrl).toBe("https://proxy.example.com:9443");
    expect(result.config.remoteHosts).toEqual([
      "proxy.example.com",
      "backup.example.com",
    ]);
    expect(result.warnings).toEqual([]);
  });

  it("revokes a file-level remote endpoint when the environment disables it", async () => {
    const homeDir = await mkdtemp(join(tmpdir(), "headroom-pi-config-"));
    const configDir = join(homeDir, ".headroom", "integrations");
    await mkdir(configDir, { recursive: true });
    await writeFile(
      join(configDir, "pi-extension.json"),
      JSON.stringify({
        allowRemote: true,
        baseUrl: "https://headroom.example.com",
        remoteHosts: ["headroom.example.com"],
      }),
    );

    const result = await loadConfig(
      { HEADROOM_PI_ALLOW_REMOTE: "false" },
      homeDir,
    );

    expect(result.config.allowRemote).toBe(false);
    expect(result.config.baseUrl).toBe(DEFAULT_CONFIG.baseUrl);
  });

  it("fails open on malformed JSON", async () => {
    const homeDir = await mkdtemp(join(tmpdir(), "headroom-pi-config-"));
    const configDir = join(homeDir, ".headroom", "integrations");
    await mkdir(configDir, { recursive: true });
    await writeFile(join(configDir, "pi-extension.json"), "{");

    const result = await loadConfig({}, homeDir);

    expect(result.config).toEqual(DEFAULT_CONFIG);
    expect(result.warnings).toHaveLength(1);
    expect(result.warnings[0]).toContain("could not be parsed");
  });
});
