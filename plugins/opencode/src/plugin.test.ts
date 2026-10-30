import childProcess from "node:child_process";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { afterEach, describe, expect, it, vi } from "vitest";

import HeadroomOpenCodePlugin, { HeadroomPlugin, headroomSetup } from "./plugin.js";
import { ToolPolicyEnforcementError } from "./transport.js";

const TRANSPORT_STATE = Symbol.for("headroom.opencode.transport");

function pluginInput() {
  return {
    client: {},
    project: { id: "project-1" },
    directory: "/repo",
    worktree: "/repo",
    experimental_workspace: {
      register: vi.fn(),
    },
    $: {},
  } as never;
}

interface FakeModel {
  id: string;
  providerID: string;
  package?: string;
  settings?: Record<string, unknown>;
  headers?: Record<string, string>;
}

interface FakeProvider {
  id: string;
  package: string;
  settings?: Record<string, unknown>;
}

// Minimal OpenCode 2.x plugin context: records what setup registers.
function pluginContextV2(
  options: Record<string, unknown>,
  catalog: { providers?: FakeProvider[]; models?: FakeModel[] } = {},
) {
  const tools: Array<Record<string, any>> = [];
  const shellHooks: Array<(input: { env: Record<string, string | undefined> }) => unknown> = [];
  const toolHooks: Record<string, Array<(input: any) => unknown>> = {};
  const sessionHooks: Record<string, Array<(input: any) => unknown>> = {};
  const disposals: Record<string, ReturnType<typeof vi.fn>[]> = {};
  const providers = catalog.providers ?? [];
  const models = catalog.models ?? [];
  const catalogProviders = [
    ...providers,
    ...models
      .filter(
        (model) => !providers.some((provider) => provider.id === model.providerID),
      )
      .map((model) => ({
        id: model.providerID,
        package: model.package ?? "",
      })),
  ];
  const registration = (name: string) => {
    const dispose = vi.fn(async () => {});
    (disposals[name] ??= []).push(dispose);
    return { dispose };
  };
  const routingEditor = {
    provider: {
      list: () =>
        catalogProviders.map((provider) => ({
          provider,
          models: new Map(
            models
              .filter((model) => model.providerID === provider.id)
              .map((model) => [model.id, model]),
          ),
        })),
      get: (providerID: string) => {
        const provider = catalogProviders.find(
          (candidate) => candidate.id === providerID,
        );
        return provider ? { provider, models: new Map() } : undefined;
      },
    },
    model: {
      get: (providerID: string, modelID: string) =>
        models.find(
          (model) => model.providerID === providerID && model.id === modelID,
        ),
      update: (
        providerID: string,
        modelID: string,
        update: (model: FakeModel) => void,
      ) => {
        const model = models.find(
          (candidate) =>
            candidate.providerID === providerID && candidate.id === modelID,
        );
        if (model) update(model);
      },
    },
  };
  const context = {
    options,
    location: {
      directory: "/repo",
      project: { id: "project-1", directory: "/repo", canonical: "/repo" },
    },
    catalog: {
      transform: vi.fn(async (callback: (editor: Record<string, any>) => void) => {
        callback(routingEditor);
        return registration("catalog.transform");
      }),
    },
    session: {
      get: vi.fn(async () => ({ location: { directory: "/repo" } })),
      hook: vi.fn(async (name: string, callback: (input: any) => unknown) => {
        (sessionHooks[name] ??= []).push(callback);
        return registration(`session.${name}`);
      }),
    },
    tool: {
      transform: vi.fn(async (callback: (editor: { add: (tool: Record<string, any>) => void }) => void) => {
        callback({ add: (tool) => tools.push(tool) });
        return registration("tool.transform");
      }),
      hook: vi.fn(async (name: string, callback: (input: any) => unknown) => {
        (toolHooks[name] ??= []).push(callback);
        return registration(`tool.${name}`);
      }),
    },
    shell: {
      hook: vi.fn(async (name: string, callback: (input: { env: Record<string, string | undefined> }) => unknown) => {
        expect(name).toBe("create.before");
        shellHooks.push(callback);
        return registration(`shell.${name}`);
      }),
    },
  };
  return {
    context: context as never,
    tools,
    shellHooks,
    toolHooks,
    sessionHooks,
    disposals,
  };
}

function transportState(): { refs: number } | undefined {
  return (globalThis as Record<symbol, { refs: number } | undefined>)[TRANSPORT_STATE];
}

afterEach(() => {
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
});

describe("HeadroomPlugin", () => {
  it("adds only Headroom metadata to shell env", async () => {
    const plugin = await HeadroomPlugin(pluginInput(), {
      proxyUrl: "http://127.0.0.1:8787/",
      backend: "litellm",
      toolPolicy: {
        rules: [{ id: "deny-curl", scope: "shell", action: "deny", command: "curl" }],
      },
    });
    const output = {
      env: {
        OPENAI_BASE_URL: "https://deepseek.example/v1",
        ANTHROPIC_BASE_URL: "https://anthropic.example",
      },
    };

    await plugin["shell.env"]?.({ cwd: "/repo" }, output);
    await plugin.dispose?.();

    expect(output.env).toMatchObject({
      HEADROOM_ACTIVE: "1",
      HEADROOM_PROXY_URL: "http://127.0.0.1:8787",
      HEADROOM_PROJECT: "project-1",
      HEADROOM_BACKEND: "litellm",
      HEADROOM_TOOL_POLICY_JSON: JSON.stringify({
        version: 1,
        mode: "enforce",
        defaultAction: "allow",
        rules: [{ id: "deny-curl", scope: "shell", action: "deny", command: "curl" }],
      }),
      OPENAI_BASE_URL: "https://deepseek.example/v1",
      ANTHROPIC_BASE_URL: "https://anthropic.example",
    });
    expect(output.env).not.toHaveProperty("HEADROOM_OPENCODE_TOOL_POLICY_JSON");
    await plugin.dispose?.();
  });

  it("locks allowed arguments before later hooks can mutate them", async () => {
    const plugin = await HeadroomPlugin(pluginInput(), {
      proxyUrl: "http://127.0.0.1:8787",
      toolPolicy: { rules: [] },
    });
    const output = {
      args: { command: "echo ok", nested: { value: "bound" } },
    };

    await plugin["tool.execute.before"]?.(
      { tool: "bash", sessionID: "session-lock", callID: "call-lock" },
      output,
    );

    expect(() => {
      output.args.command = "curl example.test";
    }).toThrow();
    expect(() => {
      output.args.nested.value = "mutated";
    }).toThrow();
    expect(() => {
      output.args = { command: "curl example.test", nested: { value: "mutated" } };
    }).toThrow();
    await plugin.dispose?.();
  });

  it("never exposes remote policy credentials or URLs to shell commands", async () => {
    const previousToken = process.env.HEADROOM_TOOL_POLICY_TOKEN;
    const previousUrl = process.env.HEADROOM_TOOL_POLICY_URL;
    process.env.HEADROOM_TOOL_POLICY_TOKEN = "private-policy-token";
    process.env.HEADROOM_TOOL_POLICY_URL =
      "https://policy.example.test/v1?signature=private-signature";
    try {
      const plugin = await HeadroomPlugin(pluginInput(), {
        proxyUrl: "http://127.0.0.1:8787",
        toolPolicy: { rules: [] },
      });
      const output = {
        env: {
          HEADROOM_TOOL_POLICY_TOKEN: "inherited-token",
          HEADROOM_TOOL_POLICY_URL: "https://inherited.example.test/?token=secret",
        } as Record<string, string>,
      };

      await plugin["shell.env"]?.({ cwd: "/repo" }, output);

      expect(output.env).not.toHaveProperty("HEADROOM_TOOL_POLICY_TOKEN");
      expect(output.env).not.toHaveProperty("HEADROOM_TOOL_POLICY_URL");
      await plugin.dispose?.();
    } finally {
      if (previousToken === undefined) delete process.env.HEADROOM_TOOL_POLICY_TOKEN;
      else process.env.HEADROOM_TOOL_POLICY_TOKEN = previousToken;
      if (previousUrl === undefined) delete process.env.HEADROOM_TOOL_POLICY_URL;
      else process.env.HEADROOM_TOOL_POLICY_URL = previousUrl;
    }
  });

  it("acknowledges allowed execution only after matching final arguments", async () => {
    const stderr = vi.spyOn(process.stderr, "write").mockImplementation(() => true);
    const plugin = await HeadroomPlugin(pluginInput(), {
      proxyUrl: "http://127.0.0.1:8787",
      toolPolicy: { rules: [] },
    });
    const input = { tool: "bash", sessionID: "session-allow", callID: "call-allow" };
    const args = { command: "echo ok" };

    await plugin["tool.execute.before"]?.(input, { args });
    expect(
      stderr.mock.calls.some(([value]) =>
        String(value).includes("headroom_tool_policy_enforcement_acknowledgement"),
      ),
    ).toBe(false);

    await plugin["tool.execute.after"]?.(
      { ...input, args: { command: "echo ok" } },
      {
        title: "shell",
        output: "ok",
        metadata: {},
      },
    );
    expect(
      stderr.mock.calls
        .map(([value]) => String(value))
        .join("")
        .match(/headroom_tool_policy_enforcement_acknowledgement/g),
    ).toHaveLength(1);
    await plugin.dispose?.();
  });

  it("bounds missing postflights and records terminal unknown outcomes", async () => {
    const stderr = vi.spyOn(process.stderr, "write").mockImplementation(() => true);
    const plugin = await HeadroomPlugin(pluginInput(), {
      proxyUrl: "http://127.0.0.1:8787",
      toolPolicy: { rules: [] },
      maxPendingPreflights: 2,
    });

    for (let index = 0; index < 5; index += 1) {
      await plugin["tool.execute.before"]?.(
        { tool: "bash", sessionID: "session-bounded", callID: `call-${index}` },
        { args: { command: `echo ${index}` } },
      );
    }

    const recordsBeforeDispose = stderr.mock.calls
      .map(([value]) => String(value).trim())
      .filter((value) => value.startsWith("{"))
      .map((value) => JSON.parse(value) as Record<string, unknown>);
    expect(
      recordsBeforeDispose.filter(
        (record) =>
          record.event === "headroom_tool_policy_enforcement_acknowledgement" &&
          record.effect === "unknown" &&
          record.reason === "capacity_evicted",
      ),
    ).toHaveLength(3);

    await plugin.dispose?.();
    const recordsAfterDispose = stderr.mock.calls
      .map(([value]) => String(value).trim())
      .filter((value) => value.startsWith("{"))
      .map((value) => JSON.parse(value) as Record<string, unknown>);
    expect(
      recordsAfterDispose.filter(
        (record) =>
          record.event === "headroom_tool_policy_enforcement_acknowledgement" &&
          record.effect === "unknown",
      ),
    ).toHaveLength(5);
    expect(
      recordsAfterDispose.filter(
        (record) => record.effect === "unknown" && record.reason === "plugin_disposed",
      ),
    ).toHaveLength(2);
  });

  it("expires a missing postflight with an unknown outcome", async () => {
    vi.useFakeTimers();
    const stderr = vi.spyOn(process.stderr, "write").mockImplementation(() => true);
    try {
      const plugin = await HeadroomPlugin(pluginInput(), {
        proxyUrl: "http://127.0.0.1:8787",
        toolPolicy: { rules: [] },
        pendingPreflightTtlMs: 25,
      });
      await plugin["tool.execute.before"]?.(
        { tool: "bash", sessionID: "session-timeout", callID: "call-timeout" },
        { args: { command: "echo waiting" } },
      );

      await vi.advanceTimersByTimeAsync(25);

      expect(
        stderr.mock.calls.some(([value]) => {
          const output = String(value);
          return output.includes('"effect":"unknown"') &&
            output.includes('"reason":"postflight_timeout"');
        }),
      ).toBe(true);
      await plugin.dispose?.();
    } finally {
      vi.useRealTimers();
    }
  });

  it("rejects an acknowledgement when final arguments differ from preflight", async () => {
    const stderr = vi.spyOn(process.stderr, "write").mockImplementation(() => true);
    const plugin = await HeadroomPlugin(pluginInput(), {
      proxyUrl: "http://127.0.0.1:8787",
      toolPolicy: { rules: [] },
    });
    const input = { tool: "bash", sessionID: "session-mutate", callID: "call-mutate" };

    await plugin["tool.execute.before"]?.(input, { args: { command: "echo ok" } });
    await expect(
      plugin["tool.execute.after"]?.(
        { ...input, args: { command: "curl example.test" } },
        {
          title: "shell",
          output: "",
          metadata: {},
        },
      ),
    ).rejects.toThrow(/did not match the bound preflight/);
    expect(
      stderr.mock.calls.some(([value]) => {
        const output = String(value);
        return output.includes('"effect":"unknown"') &&
          output.includes('"reason":"postflight_mismatch"');
      }),
    ).toBe(true);
    await plugin.dispose?.();
  });

  it("uses a unique decision ID when a call identity is reused", async () => {
    const stderr = vi.spyOn(process.stderr, "write").mockImplementation(() => true);
    const plugin = await HeadroomPlugin(pluginInput(), {
      proxyUrl: "http://127.0.0.1:8787",
      toolPolicy: { rules: [] },
    });
    const input = { tool: "bash", sessionID: "session-reused", callID: "call-reused" };

    await plugin["tool.execute.before"]?.(input, { args: { command: "echo same" } });
    await plugin["tool.execute.before"]?.(input, { args: { command: "echo same" } });

    const decisions = stderr.mock.calls
      .map(([value]) => String(value).trim())
      .filter((value) => value.startsWith("{"))
      .map((value) => JSON.parse(value) as Record<string, unknown>)
      .filter((record) => record.event === "headroom_tool_policy_decision");
    expect(decisions).toHaveLength(2);
    expect(decisions[0].decision_id).not.toBe(decisions[1].decision_id);
    expect(
      stderr.mock.calls.some(([value]) => {
        const output = String(value);
        return output.includes('"effect":"unknown"') &&
          output.includes('"reason":"call_replaced"');
      }),
    ).toBe(true);
    await plugin.dispose?.();
  });

  it("never acknowledges a late postflight as a newer reused call", async () => {
    const stderr = vi.spyOn(process.stderr, "write").mockImplementation(() => true);
    const plugin = await HeadroomPlugin(pluginInput(), {
      proxyUrl: "http://127.0.0.1:8787",
      toolPolicy: { rules: [] },
    });
    const input = { tool: "bash", sessionID: "session-late", callID: "call-late" };
    const firstArgs = { command: "echo same" };
    const secondArgs = { command: "echo same" };

    await plugin["tool.execute.before"]?.(input, { args: firstArgs });
    await plugin["tool.execute.before"]?.(input, { args: secondArgs });
    await plugin["tool.execute.after"]?.(
      { ...input, args: firstArgs },
      { title: "shell", output: "late", metadata: {} },
    );

    const acknowledgements = stderr.mock.calls
      .map(([value]) => String(value).trim())
      .filter((value) => value.startsWith("{"))
      .map((value) => JSON.parse(value) as Record<string, unknown>)
      .filter(
        (record) => record.event === "headroom_tool_policy_enforcement_acknowledgement",
      );
    expect(acknowledgements.map((record) => record.effect)).toEqual(["unknown", "unknown"]);
    expect(acknowledgements.some((record) => record.effect === "allowed")).toBe(false);
    await plugin.dispose?.();
  });

  it("retires a successfully acknowledged argument graph", async () => {
    const stderr = vi.spyOn(process.stderr, "write").mockImplementation(() => true);
    const plugin = await HeadroomPlugin(pluginInput(), {
      proxyUrl: "http://127.0.0.1:8787",
      toolPolicy: { rules: [] },
    });
    const input = { tool: "bash", sessionID: "session-retired", callID: "call-retired" };
    const args = { command: "echo same" };

    await plugin["tool.execute.before"]?.(input, { args });
    await plugin["tool.execute.after"]?.(
      { ...input, args },
      { title: "shell", output: "first", metadata: {} },
    );
    await plugin["tool.execute.before"]?.(input, { args });
    await plugin["tool.execute.after"]?.(
      { ...input, args },
      { title: "shell", output: "ambiguous", metadata: {} },
    );

    const acknowledgements = stderr.mock.calls
      .map(([value]) => String(value).trim())
      .filter((value) => value.startsWith("{"))
      .map((value) => JSON.parse(value) as Record<string, unknown>)
      .filter(
        (record) => record.event === "headroom_tool_policy_enforcement_acknowledgement",
      );
    expect(acknowledgements.map((record) => record.effect)).toEqual(["allowed", "unknown"]);
    expect(acknowledgements[1].reason).toBe("ambiguous_reused_call");
    await plugin.dispose?.();
  });

  it("never forgets retired call identities when the TTL expires", async () => {
    vi.useFakeTimers();
    const stderr = vi.spyOn(process.stderr, "write").mockImplementation(() => true);
    try {
      const plugin = await HeadroomPlugin(pluginInput(), {
        proxyUrl: "http://127.0.0.1:8787",
        toolPolicy: { rules: [] },
        pendingPreflightTtlMs: 25,
      });
      const input = { tool: "bash", sessionID: "session-expired", callID: "call-expired" };

      await plugin["tool.execute.before"]?.(input, { args: { command: "echo first" } });
      await plugin["tool.execute.after"]?.(
        { ...input, args: { command: "echo first" } },
        { title: "shell", output: "first", metadata: {} },
      );
      await vi.advanceTimersByTimeAsync(25);
      await plugin["tool.execute.before"]?.(input, { args: { command: "echo second" } });
      await plugin["tool.execute.after"]?.(
        { ...input, args: { command: "echo second" } },
        { title: "shell", output: "second", metadata: {} },
      );

      const acknowledgements = stderr.mock.calls
        .map(([value]) => String(value).trim())
        .filter((value) => value.startsWith("{"))
        .map((value) => JSON.parse(value) as Record<string, unknown>)
        .filter(
          (record) => record.event === "headroom_tool_policy_enforcement_acknowledgement",
        );
      expect(acknowledgements.map((record) => record.effect)).toEqual(["allowed", "unknown"]);
      expect(acknowledgements[1].reason).toBe("ambiguous_reused_call");
      await plugin.dispose?.();
    } finally {
      vi.useRealTimers();
    }
  });

  it("preserves retired call identities beyond the pending capacity", async () => {
    const stderr = vi.spyOn(process.stderr, "write").mockImplementation(() => true);
    const plugin = await HeadroomPlugin(pluginInput(), {
      proxyUrl: "http://127.0.0.1:8787",
      toolPolicy: { rules: [] },
      maxPendingPreflights: 2,
    });

    for (let index = 0; index < 3; index += 1) {
      const input = { tool: "bash", sessionID: "session-retired-bound", callID: `call-${index}` };
      const args = { command: `echo ${index}` };
      await plugin["tool.execute.before"]?.(input, { args });
      await plugin["tool.execute.after"]?.(
        { ...input, args: { ...args } },
        { title: "shell", output: String(index), metadata: {} },
      );
    }

    const evictedInput = {
      tool: "bash",
      sessionID: "session-retired-bound",
      callID: "call-0",
    };
    await plugin["tool.execute.before"]?.(evictedInput, { args: { command: "echo again" } });
    await plugin["tool.execute.after"]?.(
      { ...evictedInput, args: { command: "echo again" } },
      { title: "shell", output: "again", metadata: {} },
    );

    const acknowledgements = stderr.mock.calls
      .map(([value]) => String(value).trim())
      .filter((value) => value.startsWith("{"))
      .map((value) => JSON.parse(value) as Record<string, unknown>)
      .filter(
        (record) => record.event === "headroom_tool_policy_enforcement_acknowledgement",
      );
    expect(acknowledgements.map((record) => record.effect)).toEqual([
      "allowed",
      "allowed",
      "allowed",
      "unknown",
    ]);
    expect(acknowledgements[3].reason).toBe("ambiguous_reused_call");
    await plugin.dispose?.();
  });

  it("does not let a late completion acknowledge a reused identity after eviction", async () => {
    const stderr = vi.spyOn(process.stderr, "write").mockImplementation(() => true);
    const plugin = await HeadroomPlugin(pluginInput(), {
      proxyUrl: "http://127.0.0.1:8787",
      toolPolicy: { rules: [] },
      maxPendingPreflights: 1,
    });
    const first = { tool: "bash", sessionID: "session-generation", callID: "call-a" };
    const other = { tool: "bash", sessionID: "session-generation", callID: "call-b" };
    const args = { command: "echo same" };

    await plugin["tool.execute.before"]?.(first, { args: { ...args } });
    await plugin["tool.execute.before"]?.(other, { args: { command: "echo other" } });
    await plugin["tool.execute.after"]?.(
      { ...other, args: { command: "echo other" } },
      { title: "shell", output: "other", metadata: {} },
    );
    await plugin["tool.execute.before"]?.(first, { args: { ...args } });
    await plugin["tool.execute.after"]?.(
      { ...first, args: { ...args } },
      { title: "shell", output: "late", metadata: {} },
    );

    const acknowledgements = stderr.mock.calls
      .map(([value]) => String(value).trim())
      .filter((value) => value.startsWith("{"))
      .map((value) => JSON.parse(value) as Record<string, unknown>)
      .filter(
        (record) => record.event === "headroom_tool_policy_enforcement_acknowledgement",
      );
    expect(acknowledgements.map((record) => record.effect)).toEqual([
      "unknown",
      "allowed",
      "unknown",
    ]);
    expect(acknowledgements[0].reason).toBe("capacity_evicted");
    expect(acknowledgements[2].reason).toBe("ambiguous_reused_call");
    await plugin.dispose?.();
  });

  it("fails closed deterministically when retired identity capacity is exhausted", async () => {
    const stderr = vi.spyOn(process.stderr, "write").mockImplementation(() => true);
    const plugin = await HeadroomPlugin(pluginInput(), {
      proxyUrl: "http://127.0.0.1:8787",
      toolPolicy: { rules: [] },
      maxRetiredCallIdentities: 2,
    });

    for (let index = 0; index < 4; index += 1) {
      const input = {
        tool: "bash",
        sessionID: "session-history-capacity",
        callID: `call-${index}`,
      };
      await plugin["tool.execute.before"]?.(input, {
        args: { command: `echo ${index}` },
      });
      await plugin["tool.execute.after"]?.(
        { ...input, args: { command: `echo ${index}` } },
        { title: "shell", output: String(index), metadata: {} },
      );
    }

    const acknowledgements = stderr.mock.calls
      .map(([value]) => String(value).trim())
      .filter((value) => value.startsWith("{"))
      .map((value) => JSON.parse(value) as Record<string, unknown>)
      .filter(
        (record) => record.event === "headroom_tool_policy_enforcement_acknowledgement",
      );
    expect(acknowledgements.map((record) => record.effect)).toEqual([
      "allowed",
      "allowed",
      "allowed",
      "unknown",
    ]);
    expect(acknowledgements[3].reason).toBe("ambiguous_reused_call");
    await plugin.dispose?.();
  });

  it("blocks native shell tools using the filesystem project directory", async () => {
    const repo = fs.mkdtempSync(path.join(os.tmpdir(), "headroom-native-policy-"));
    const previousConfigDir = process.env.HEADROOM_CONFIG_DIR;
    process.env.HEADROOM_CONFIG_DIR = path.join(repo, "empty-global-config");
    fs.mkdirSync(path.join(repo, ".headroom"));
    fs.writeFileSync(
      path.join(repo, ".headroom", "tool_policy.json"),
      JSON.stringify({
        rules: [{ id: "deny-wrapped", scope: "shell", action: "deny", command: "curl" }],
      }),
    );
    const input = pluginInput() as unknown as { directory: string; worktree: string };
    input.directory = repo;
    input.worktree = "";
    try {
      const plugin = await HeadroomPlugin(input as never, {
        proxyUrl: "http://127.0.0.1:8787",
      });
      await expect(
        plugin["tool.execute.before"]?.(
          { tool: "bash", sessionID: "s", callID: "c" },
          { args: { command: "echo ok && sudo env X=1 curl example.test" } },
        ),
      ).rejects.toThrow(/deny-wrapped/);
      await plugin.dispose?.();
    } finally {
      if (previousConfigDir === undefined) delete process.env.HEADROOM_CONFIG_DIR;
      else process.env.HEADROOM_CONFIG_DIR = previousConfigDir;
      fs.rmSync(repo, { recursive: true, force: true });
    }
  });

  it("blocks a non-shell native tool by tool name", async () => {
    const plugin = await HeadroomPlugin(pluginInput(), {
      proxyUrl: "http://127.0.0.1:8787",
      toolPolicy: {
        rules: [
          {
            id: "deny-write",
            scope: "tool_call",
            action: "deny",
            tool: "write",
            argsPattern: '"path":"secret\\.txt"',
          },
        ],
      },
    });

    await expect(
      plugin["tool.execute.before"]?.(
        { tool: "write", sessionID: "s", callID: "c" },
        { args: { value: "data", path: "secret.txt" } },
      ),
    ).rejects.toThrow(/deny-write/);
    await plugin.dispose?.();
  });

  it("does not let one allowed binary authorize a compound shell command", async () => {
    const plugin = await HeadroomPlugin(pluginInput(), {
      proxyUrl: "http://127.0.0.1:8787",
      toolPolicy: {
        defaultAction: "deny",
        rules: [{ id: "allow-echo", scope: "shell", action: "allow", command: "echo" }],
      },
    });

    await expect(
      plugin["tool.execute.before"]?.(
        { tool: "bash", sessionID: "s-compound", callID: "c-compound" },
        { args: { command: "echo ok; curl attacker.example" } },
      ),
    ).rejects.toThrow(/Tool policy denied shell/);
    await plugin.dispose?.();
  });

  it("fails closed on dynamic shell grammar in an allowlist policy", async () => {
    const plugin = await HeadroomPlugin(pluginInput(), {
      proxyUrl: "http://127.0.0.1:8787",
      toolPolicy: {
        defaultAction: "deny",
        rules: [{ id: "allow-echo", scope: "shell", action: "allow", command: "echo" }],
      },
    });

    await expect(
      plugin["tool.execute.before"]?.(
        { tool: "bash", sessionID: "s-allowlist", callID: "c-allowlist" },
        { args: { command: "x=echo; $x dynamic" } },
      ),
    ).rejects.toThrow(/dynamic or escaped shell execution cannot be safely authorized/);
    await plugin.dispose?.();
  });

  it.each([
    "echo 'if'",
    'echo "eval"',
    "echo if then else",
  ])("allows reserved words used only as literal arguments: %s", async (command) => {
    const plugin = await HeadroomPlugin(pluginInput(), {
      proxyUrl: "http://127.0.0.1:8787",
      toolPolicy: {
        defaultAction: "deny",
        rules: [{ id: "allow-echo", scope: "shell", action: "allow", command: "echo" }],
      },
    });

    await expect(
      plugin["tool.execute.before"]?.(
        { tool: "bash", sessionID: `s-literal-${command}`, callID: "c-literal" },
        { args: { command } },
      ),
    ).resolves.toBeUndefined();
    await plugin.dispose?.();
  });

  it("matches and binds the shell tool's effective workdir", async () => {
    const plugin = await HeadroomPlugin(pluginInput(), {
      proxyUrl: "http://127.0.0.1:8787",
      toolPolicy: {
        rules: [
          {
            id: "deny-app-curl",
            scope: "shell",
            action: "deny",
            command: "curl",
            cwdPattern: "packages[\\\\/]app$",
          },
        ],
      },
    });
    let error: unknown;
    try {
      await plugin["tool.execute.before"]?.(
        { tool: "bash", sessionID: "s-workdir", callID: "c-workdir" },
        { args: { command: "curl example.test", workdir: "packages/app" } },
      );
    } catch (caught) {
      error = caught;
    }
    expect(error).toBeInstanceOf(ToolPolicyEnforcementError);
    expect((error as ToolPolicyEnforcementError).decision.binding?.cwd).toBe(
      path.resolve("/repo", "packages/app"),
    );
    await plugin.dispose?.();
  });

  it.each([
    "cu\\rl example.test",
    "x=curl; $x example.test",
    'x=curl; "$x" example.test',
    "x=rl; cu${x} example.test",
    "x=curl; /usr/bin/${x} example.test",
    "$'curl' example.test",
    "set CMD=curl && %CMD% example.test",
    "cmd=curl; & ($cmd) example.test",
    "echo harmless > >(curl attacker.example)",
    "if true; then curl attacker.example; fi",
    "eval curl example.test",
  ])(
    "fails closed for dynamic shell command %s",
    async (command) => {
      const plugin = await HeadroomPlugin(pluginInput(), {
        proxyUrl: "http://127.0.0.1:8787",
        toolPolicy: {
          rules: [{ id: "deny-curl", scope: "shell", action: "deny", command: "curl" }],
        },
      });
      await expect(
        plugin["tool.execute.before"]?.(
          { tool: "bash", sessionID: `s-${command}`, callID: `c-${command}` },
          { args: { command } },
        ),
      ).rejects.toThrow(/Tool policy denied shell/);
      await plugin.dispose?.();
    },
  );

  it("refuses to replace another workspace's active policy", async () => {
    const restrictive = await HeadroomPlugin(pluginInput(), {
      proxyUrl: "http://127.0.0.1:8787",
      toolPolicy: {
        rules: [{ id: "deny-curl", scope: "shell", action: "deny", command: "curl" }],
      },
    });
    await expect(
      HeadroomPlugin(pluginInput(), {
        proxyUrl: "http://127.0.0.1:8787",
        toolPolicy: { rules: [] },
      }),
    ).rejects.toThrow(/refusing to replace the active policy/);
    await expect(
      restrictive["tool.execute.before"]?.(
        { tool: "bash", sessionID: "s-isolation", callID: "c-isolation" },
        { args: { command: "curl example.test" } },
      ),
    ).rejects.toThrow(/deny-curl/);
    await restrictive.dispose?.();
  });

  it("denies the same action through native and alternate execution paths", async () => {
    const stderr = vi.spyOn(process.stderr, "write").mockImplementation(() => true);
    const plugin = await HeadroomPlugin(pluginInput(), {
      proxyUrl: "http://127.0.0.1:8787",
      toolPolicy: {
        rules: [{ id: "deny-node", scope: "shell", action: "deny", command: "node" }],
      },
    });

    let nativeError: unknown;
    try {
      await plugin["tool.execute.before"]?.(
        { tool: "bash", sessionID: "session-7", callID: "call-9" },
        { args: { command: "node -e \"console.log('not executed')\"" } },
      );
    } catch (error) {
      nativeError = error;
    }
    expect(nativeError).toBeInstanceOf(ToolPolicyEnforcementError);
    const enforcementError = nativeError as ToolPolicyEnforcementError;
    expect(enforcementError.decision).toMatchObject({
      version: 1,
      authority: "authoritative",
      binding: {
        caller: "opencode",
        adapter: "tool.execute.before",
        sessionID: "session-7",
        taskID: "call-9",
        callID: "call-9",
        toolName: "bash",
        cwd: path.resolve("/repo"),
      },
    });
    expect(enforcementError.decision.decisionId).toMatch(/^[a-f0-9]{64}$/);
    expect(enforcementError.decision.binding?.canonicalArgsHash).toMatch(/^[a-f0-9]{16}$/);
    expect(enforcementError.acknowledgement).toMatchObject({
      decisionId: enforcementError.decision.decisionId,
      authority: "authoritative",
      effect: "blocked",
      binding: { sessionID: "session-7", callID: "call-9" },
    });

    expect(() =>
      childProcess.exec("node -e \"console.log('alternate path not executed')\""),
    ).toThrow(/deny-node/);

    const records = stderr.mock.calls
      .map(([value]) => String(value).trim())
      .filter((value) => value.startsWith("{"))
      .map((value) => JSON.parse(value) as Record<string, unknown>);
    expect(
      records.filter(
        (record) => record.event === "headroom_tool_policy_enforcement_acknowledgement",
      ),
    ).toHaveLength(1);
    expect(
      records
        .filter((record) => record.event === "headroom_tool_policy_decision")
        .map((record) => record.authority),
    ).toEqual(["authoritative", "advisory"]);
    await plugin.dispose?.();
  });

  it("exposes a headroom_retrieve tool backed by the proxy", async () => {
    const fetchMock = vi.fn(async () => ({
      ok: true,
      json: async () => "original content",
    }));
    vi.stubGlobal("fetch", fetchMock);

    const plugin = await HeadroomPlugin(pluginInput(), {
      proxyUrl: "http://127.0.0.1:8787",
    });
    const result = await plugin.tool?.headroom_retrieve.execute(
      { hash: "0123456789abcdef01234567" },
      {} as never,
    );
    await plugin.dispose?.();

    expect(result).toBe("original content");
    expect(fetchMock).toHaveBeenCalledWith(
      "http://127.0.0.1:8787/v1/retrieve/0123456789abcdef01234567",
      expect.any(Object),
    );
  });
});

describe("headroomSetup (OpenCode 2.x)", () => {
  it("uses the session directory when the live runtime omits setup location", async () => {
    const { context, sessionHooks, shellHooks } = pluginContextV2({
      proxyUrl: "http://127.0.0.1:8787",
      toolPolicy: { rules: [] },
    });
    const runtimeContext = context as unknown as {
      location?: unknown;
      model?: unknown;
    };
    runtimeContext.location = undefined;
    runtimeContext.model = undefined;

    const cleanup = await headroomSetup(context);
    const modelRequest = { sessionID: "session-live", headers: {} as Record<string, string> };
    await sessionHooks["model.request"][0](modelRequest);
    const shell = { env: {} as Record<string, string | undefined> };
    await shellHooks[0](shell);

    expect(modelRequest.headers["x-headroom-project"]).toBe(process.cwd());
    expect(shell.env.HEADROOM_PROJECT).toBe(process.cwd());
    await cleanup?.();
  });

  it("preserves the configured project label on model requests", async () => {
    const { context, sessionHooks } = pluginContextV2({
      proxyUrl: "http://127.0.0.1:8787",
      toolPolicy: { rules: [] },
    });
    const cleanup = await headroomSetup(context);
    const modelRequest = {
      sessionID: "session-project-label",
      headers: {} as Record<string, string>,
    };

    await sessionHooks["model.request"][0](modelRequest);

    expect(modelRequest.headers["x-headroom-project"]).toBe("project-1");
    expect(
      (context as unknown as { session: { get: ReturnType<typeof vi.fn> } })
        .session.get,
    ).not.toHaveBeenCalled();
    await cleanup?.();
  });

  it("discovers repository policy from the authoritative session directory", async () => {
    const repo = fs.mkdtempSync(path.join(os.tmpdir(), "headroom-v2-policy-"));
    const previousConfigDir = process.env.HEADROOM_CONFIG_DIR;
    process.env.HEADROOM_CONFIG_DIR = path.join(repo, "empty-global-config");
    fs.mkdirSync(path.join(repo, ".headroom"));
    fs.writeFileSync(
      path.join(repo, ".headroom", "tool_policy.json"),
      JSON.stringify({
        rules: [
          { id: "deny-session-curl", scope: "shell", action: "deny", command: "curl" },
        ],
      }),
    );
    const { context, toolHooks } = pluginContextV2({
      proxyUrl: "http://127.0.0.1:8787",
    });
    const runtimeContext = context as unknown as {
      location?: unknown;
      session: { get: ReturnType<typeof vi.fn> };
    };
    runtimeContext.location = undefined;
    runtimeContext.session.get = vi.fn(async () => ({
      location: { directory: repo },
    }));

    try {
      const cleanup = await headroomSetup(context);
      await expect(
        toolHooks["execute.before"][0]({
          tool: "shell",
          sessionID: "session-repository",
          messageID: "message-repository",
          id: "call-repository",
          input: { command: "curl example.test" },
        }),
      ).rejects.toThrow(/deny-session-curl/);
      expect(runtimeContext.session.get).toHaveBeenCalledWith({
        sessionID: "session-repository",
      });
      await cleanup?.();
    } finally {
      if (previousConfigDir === undefined) delete process.env.HEADROOM_CONFIG_DIR;
      else process.env.HEADROOM_CONFIG_DIR = previousConfigDir;
      fs.rmSync(repo, { recursive: true, force: true });
    }
  });

  it("denies a native tool in execute.before before execution", async () => {
    const { context, toolHooks } = pluginContextV2({
      proxyUrl: "http://127.0.0.1:8787",
      toolPolicy: {
        rules: [{ id: "deny-curl", scope: "shell", action: "deny", command: "curl" }],
      },
    });
    const cleanup = await headroomSetup(context);

    expect(toolHooks["execute.before"]).toHaveLength(1);
    await expect(
      toolHooks["execute.before"][0]({
        tool: "bash",
        sessionID: "session-denied",
        messageID: "message-denied",
        id: "call-denied",
        input: { command: "curl example.test" },
      }),
    ).rejects.toThrow(/deny-curl/);
    await cleanup?.();
  });

  it("binds and acknowledges a matching allowed native execution", async () => {
    const stderr = vi.spyOn(process.stderr, "write").mockImplementation(() => true);
    const { context, toolHooks } = pluginContextV2({
      proxyUrl: "http://127.0.0.1:8787",
      toolPolicy: { rules: [] },
    });
    const cleanup = await headroomSetup(context);
    const before = {
      tool: "bash",
      sessionID: "session-allowed",
      messageID: "message-allowed",
      id: "call-allowed",
      input: { command: "echo ok", nested: { value: "bound" } },
    };

    await toolHooks["execute.before"][0](before);
    expect(() => {
      before.input.command = "curl example.test";
    }).toThrow();
    expect(() => {
      before.input.nested.value = "mutated";
    }).toThrow();
    expect(() => {
      before.input = { command: "curl example.test", nested: { value: "mutated" } };
    }).toThrow();

    await toolHooks["execute.after"][0]({
      ...before,
      input: { command: "echo ok", nested: { value: "bound" } },
      status: "completed",
      result: { content: "ok" },
    });
    expect(
      stderr.mock.calls.some(([value]) => {
        const output = String(value);
        return output.includes('"effect":"allowed"') &&
          output.includes('"taskID":"message-allowed"');
      }),
    ).toBe(true);
    await cleanup?.();
  });

  it("records a mismatched native completion as unknown", async () => {
    const stderr = vi.spyOn(process.stderr, "write").mockImplementation(() => true);
    const { context, toolHooks } = pluginContextV2({
      proxyUrl: "http://127.0.0.1:8787",
      toolPolicy: { rules: [] },
    });
    const cleanup = await headroomSetup(context);
    const identity = {
      tool: "bash",
      sessionID: "session-mismatch",
      messageID: "message-mismatch",
      id: "call-mismatch",
    };

    await toolHooks["execute.before"][0]({
      ...identity,
      input: { command: "echo ok" },
    });
    expect(() =>
      toolHooks["execute.after"][0]({
        ...identity,
        input: { command: "curl example.test" },
        status: "completed",
        result: { content: "" },
      }),
    ).toThrow(/did not match the bound preflight/);
    expect(
      stderr.mock.calls.some(([value]) =>
        String(value).includes('"reason":"postflight_mismatch"'),
      ),
    ).toBe(true);
    await cleanup?.();
  });

  it("retires pending native calls during cleanup", async () => {
    const stderr = vi.spyOn(process.stderr, "write").mockImplementation(() => true);
    const { context, toolHooks } = pluginContextV2({
      proxyUrl: "http://127.0.0.1:8787",
      toolPolicy: { rules: [] },
    });
    const cleanup = await headroomSetup(context);

    await toolHooks["execute.before"][0]({
      tool: "bash",
      sessionID: "session-pending",
      messageID: "message-pending",
      id: "call-pending",
      input: { command: "echo waiting" },
    });
    await cleanup?.();

    expect(
      stderr.mock.calls.some(([value]) =>
        String(value).includes('"reason":"plugin_disposed"'),
      ),
    ).toBe(true);
  });

  it("records errored native executions as unknown", async () => {
    const stderr = vi.spyOn(process.stderr, "write").mockImplementation(() => true);
    const { context, toolHooks } = pluginContextV2({
      proxyUrl: "http://127.0.0.1:8787",
      toolPolicy: { rules: [] },
    });
    const cleanup = await headroomSetup(context);
    const identity = {
      tool: "bash",
      sessionID: "session-error",
      messageID: "message-error",
      id: "call-error",
      input: { command: "exit 1" },
    };

    await toolHooks["execute.before"][0](identity);
    await toolHooks["execute.after"][0]({
      ...identity,
      status: "error",
      error: { name: "ToolError", message: "failed" },
    });

    expect(
      stderr.mock.calls.some(([value]) =>
        String(value).includes('"reason":"postflight_error"'),
      ),
    ).toBe(true);
    await cleanup?.();
  });

  it("enforces an environment policy and removes its credentials from shells", async () => {
    vi.stubEnv(
      "HEADROOM_TOOL_POLICY_JSON",
      JSON.stringify({
        rules: [{ id: "deny-write", scope: "tool_call", action: "deny", tool: "write" }],
      }),
    );
    vi.stubEnv("HEADROOM_TOOL_POLICY_TOKEN", "private-token");
    vi.stubEnv("HEADROOM_TOOL_POLICY_URL", "https://policy.example.test/?token=private");
    try {
      const { context, shellHooks, toolHooks } = pluginContextV2({
        proxyUrl: "http://127.0.0.1:8787",
      });
      const cleanup = await headroomSetup(context);
      const shell = {
        env: {
          HEADROOM_TOOL_POLICY_TOKEN: "inherited-token",
          HEADROOM_TOOL_POLICY_URL: "https://inherited.example.test/",
        } as Record<string, string | undefined>,
      };

      await shellHooks[0](shell);
      expect(shell.env).not.toHaveProperty("HEADROOM_TOOL_POLICY_TOKEN");
      expect(shell.env).not.toHaveProperty("HEADROOM_TOOL_POLICY_URL");
      expect(shell.env.HEADROOM_TOOL_POLICY_JSON).toContain("deny-write");
      await expect(
        toolHooks["execute.before"][0]({
          tool: "write",
          sessionID: "session-environment",
          messageID: "message-environment",
          id: "call-environment",
          input: { path: "secret.txt", value: "data" },
        }),
      ).rejects.toThrow(/deny-write/);
      await cleanup?.();
    } finally {
      vi.unstubAllEnvs();
    }
  });

  it("registers headroom_retrieve as a direct tool backed by the proxy", async () => {
    const { context, tools } = pluginContextV2({ proxyUrl: "http://127.0.0.1:8787/" });
    const cleanup = await headroomSetup(context);

    const fetchMock = vi.fn(async () => ({
      ok: true,
      json: async () => "original content",
    }));
    vi.stubGlobal("fetch", fetchMock);

    expect(tools).toHaveLength(1);
    const [retrieve] = tools;
    expect(retrieve).toMatchObject({
      name: "headroom_retrieve",
      options: { codemode: false },
      input: { type: "object", required: ["hash"] },
    });
    const result = await retrieve.execute(
      { hash: "0123456789abcdef01234567" },
      { signal: new AbortController().signal, progress: async () => {} },
    );
    await cleanup?.();

    expect(result).toEqual({ content: "original content" });
    expect(fetchMock).toHaveBeenCalledWith(
      "http://127.0.0.1:8787/v1/retrieve/0123456789abcdef01234567",
      expect.any(Object),
    );
  });

  it("adds only Headroom metadata to shell env", async () => {
    const { context, shellHooks } = pluginContextV2({
      proxyUrl: "http://127.0.0.1:8787/",
      backend: "litellm",
    });
    const cleanup = await headroomSetup(context);
    const shell = {
      env: {
        OPENAI_BASE_URL: "https://deepseek.example/v1",
        ANTHROPIC_BASE_URL: "https://anthropic.example",
      } as Record<string, string | undefined>,
    };

    expect(shellHooks).toHaveLength(1);
    await shellHooks[0](shell);
    await cleanup?.();

    expect(shell.env).toMatchObject({
      HEADROOM_ACTIVE: "1",
      HEADROOM_PROXY_URL: "http://127.0.0.1:8787",
      HEADROOM_PROJECT: "project-1",
      HEADROOM_BACKEND: "litellm",
      OPENAI_BASE_URL: "https://deepseek.example/v1",
      ANTHROPIC_BASE_URL: "https://anthropic.example",
    });
  });

  it("points remote OpenAI-wire models at the proxy and names the real upstream", async () => {
    const models: FakeModel[] = [
      { id: "deepseek-v4.1-flash", providerID: "opencode-go" },
      { id: "gpt-6-luna", providerID: "gateway", package: "@opencode/ai/providers/openai/responses" },
      { id: "space-bunny-free", providerID: "opencode" },
    ];
    const { context } = pluginContextV2(
      { proxyUrl: "http://127.0.0.1:8787/" },
      {
        providers: [
          {
            id: "opencode-go",
            package: "@opencode/ai/providers/openai-compatible",
            settings: { baseURL: "https://opencode.ai/zen/go/v1/" },
          },
          {
            id: "gateway",
            package: "@opencode/ai/providers/openai-compatible",
            settings: { baseURL: "https://gateway.example/api/v1" },
          },
          {
            id: "opencode",
            package: "aisdk:@ai-sdk/openai-compatible",
            settings: { baseURL: "https://opencode.ai/inference/v1" },
          },
        ],
        models,
      },
    );

    const cleanup = await headroomSetup(context);
    await cleanup?.();

    expect(models[0]).toMatchObject({
      settings: { baseURL: "http://127.0.0.1:8787/v1" },
      headers: {
        "x-headroom-base-url": "https://opencode.ai",
        "x-headroom-original-path": "/zen/go/v1/chat/completions",
        "x-headroom-project": "project-1",
      },
    });
    expect(models[1]).toMatchObject({
      settings: { baseURL: "http://127.0.0.1:8787/v1" },
      headers: {
        "x-headroom-base-url": "https://gateway.example",
        "x-headroom-original-path": "/api/v1/responses",
      },
    });
    expect(models[2]).toMatchObject({
      settings: { baseURL: "http://127.0.0.1:8787/v1" },
      headers: {
        "x-headroom-base-url": "https://opencode.ai",
        "x-headroom-original-path": "/inference/v1/chat/completions",
      },
    });
  });

  it("prefers a model's own baseURL over its provider's", async () => {
    const models: FakeModel[] = [
      {
        id: "custom",
        providerID: "opencode-go",
        settings: { baseURL: "https://model.example/v1", temperature: 0 },
        headers: { "x-user": "kept" },
      },
    ];
    const { context } = pluginContextV2(
      { proxyUrl: "http://127.0.0.1:8787" },
      {
        providers: [
          {
            id: "opencode-go",
            package: "@opencode/ai/providers/openai-compatible",
            settings: { baseURL: "https://opencode.ai/zen/go/v1" },
          },
        ],
        models,
      },
    );

    const cleanup = await headroomSetup(context);
    await cleanup?.();

    expect(models[0]).toEqual({
      id: "custom",
      providerID: "opencode-go",
      settings: { baseURL: "http://127.0.0.1:8787/v1", temperature: 0 },
      headers: {
        "x-user": "kept",
        "x-headroom-base-url": "https://model.example",
        "x-headroom-original-path": "/v1/chat/completions",
        "x-headroom-project": "project-1",
      },
    });
  });

  it("keeps hosts excluded by option off the model rewrite", async () => {
    const originalExclude = process.env.HEADROOM_OPENCODE_EXCLUDE_HOSTS;
    const models: FakeModel[] = [
      { id: "zen", providerID: "opencode-go" },
      { id: "kept", providerID: "gateway" },
    ];
    const { context } = pluginContextV2(
      { proxyUrl: "http://127.0.0.1:8787", excludeHosts: [".opencode.ai"] },
      {
        providers: [
          {
            id: "opencode-go",
            package: "@opencode/ai/providers/openai-compatible",
            settings: { baseURL: "https://opencode.ai/zen/go/v1" },
          },
          {
            id: "gateway",
            package: "@opencode/ai/providers/openai-compatible",
            settings: { baseURL: "https://gateway.example/api/v1" },
          },
        ],
        models,
      },
    );

    try {
      const cleanup = await headroomSetup(context);
      await cleanup?.();
    } finally {
      if (originalExclude === undefined) {
        delete process.env.HEADROOM_OPENCODE_EXCLUDE_HOSTS;
      } else {
        process.env.HEADROOM_OPENCODE_EXCLUDE_HOSTS = originalExclude;
      }
    }

    expect(models[0].settings).toBeUndefined();
    expect(models[0].headers).toBeUndefined();
    expect(models[1]).toMatchObject({
      settings: { baseURL: "http://127.0.0.1:8787/v1" },
      headers: { "x-headroom-base-url": "https://gateway.example" },
    });
  });

  it("keeps hosts excluded by environment off the model rewrite", async () => {
    const originalExclude = process.env.HEADROOM_OPENCODE_EXCLUDE_HOSTS;
    process.env.HEADROOM_OPENCODE_EXCLUDE_HOSTS = "opencode.ai";
    try {
      const models: FakeModel[] = [{ id: "zen", providerID: "opencode-go" }];
      const { context } = pluginContextV2(
        { proxyUrl: "http://127.0.0.1:8787" },
        {
          providers: [
            {
              id: "opencode-go",
              package: "@opencode/ai/providers/openai-compatible",
              settings: { baseURL: "https://zen.opencode.ai/zen/go/v1" },
            },
          ],
          models,
        },
      );

      const cleanup = await headroomSetup(context);
      await cleanup?.();

      expect(models[0].settings).toBeUndefined();
      expect(models[0].headers).toBeUndefined();
    } finally {
      if (originalExclude === undefined) {
        delete process.env.HEADROOM_OPENCODE_EXCLUDE_HOSTS;
      } else {
        process.env.HEADROOM_OPENCODE_EXCLUDE_HOSTS = originalExclude;
      }
    }
  });

  it("handles adversarial slash runs in model base URLs in bounded time", async () => {
    const trailing = "/".repeat(20000);
    const models: FakeModel[] = [
      {
        id: "non-slash-tail",
        providerID: "gateway",
        settings: { baseURL: `https://gateway.example/api/v1${trailing}x` },
      },
      {
        id: "slash-tail",
        providerID: "gateway",
        settings: { baseURL: `https://gateway.example/api/v1${trailing}` },
      },
    ];
    const { context } = pluginContextV2(
      { proxyUrl: "http://127.0.0.1:8787" },
      {
        providers: [
          {
            id: "gateway",
            package: "@opencode/ai/providers/openai-compatible",
          },
        ],
        models,
      },
    );

    const cleanup = await headroomSetup(context);
    await cleanup?.();

    expect(models[0]).toMatchObject({
      settings: { baseURL: "http://127.0.0.1:8787/v1" },
      headers: {
        "x-headroom-base-url": "https://gateway.example",
        "x-headroom-original-path": `/api/v1${trailing}x/chat/completions`,
      },
    });
    expect(models[1]).toMatchObject({
      settings: { baseURL: "http://127.0.0.1:8787/v1" },
      headers: {
        "x-headroom-base-url": "https://gateway.example",
        "x-headroom-original-path": "/api/v1/chat/completions",
      },
    });
  });

  it("leaves local, already-proxied, non-OpenAI-wire and URL-less models alone", async () => {
    const models: FakeModel[] = [
      { id: "llama", providerID: "ollama" },
      { id: "gpt-4o", providerID: "openai" },
      { id: "claude", providerID: "anthropic-gateway" },
      { id: "glm", providerID: "zai-coding-plan", package: "@opencode/ai/providers/zai-coding-plan/messages" },
      { id: "sonnet", providerID: "anthropic" },
      { id: "gemini", providerID: "google" },
    ];
    const snapshot = structuredClone(models);
    const { context } = pluginContextV2(
      { proxyUrl: "http://127.0.0.1:8787" },
      {
        providers: [
          {
            id: "ollama",
            package: "@opencode/ai/providers/openai-compatible",
            settings: { baseURL: "http://127.0.0.1:11434/v1" },
          },
          {
            id: "openai",
            package: "@opencode/ai/providers/openai/responses",
            settings: { baseURL: "http://127.0.0.1:8787/v1" },
          },
          {
            id: "anthropic-gateway",
            package: "@opencode/ai/providers/anthropic",
            settings: { baseURL: "https://gateway.example/v1" },
          },
          {
            id: "zai-coding-plan",
            package: "@opencode/ai/providers/zai-coding-plan/chat",
            settings: { baseURL: "https://api.z.ai/api/coding/paas/v4" },
          },
          { id: "anthropic", package: "@opencode/ai/providers/anthropic" },
          {
            id: "google",
            package: "@ai-sdk/google",
            settings: { baseURL: "https://generativelanguage.googleapis.com/v1beta" },
          },
        ],
        models,
      },
    );

    const cleanup = await headroomSetup(context);
    await cleanup?.();

    expect(models).toEqual(snapshot);
  });

  it("releases the transport on cleanup so reloads do not leak it", async () => {
    const { context, disposals } = pluginContextV2({
      proxyUrl: "http://127.0.0.1:8787",
    });

    const first = await headroomSetup(context);
    expect(transportState()?.refs).toBe(1);
    await first?.();
    expect(transportState()).toBeUndefined();
    expect(Object.values(disposals).flat()).toHaveLength(6);
    for (const dispose of Object.values(disposals).flat()) {
      expect(dispose).toHaveBeenCalledOnce();
    }

    const second = await headroomSetup(context);
    expect(transportState()?.refs).toBe(1);
    await second?.();
    expect(transportState()).toBeUndefined();
  });

  it("releases the transport when hook registration fails", async () => {
    const { context, disposals } = pluginContextV2({
      proxyUrl: "http://127.0.0.1:8787",
    });
    (context as { shell: { hook: () => Promise<never> } }).shell.hook = async () => {
      throw new Error("registration failed");
    };

    await expect(headroomSetup(context)).rejects.toThrow("registration failed");
    expect(transportState()).toBeUndefined();
    expect(disposals["catalog.transform"][0]).toHaveBeenCalledOnce();
    expect(disposals["tool.transform"][0]).toHaveBeenCalledOnce();
  });
});

describe("default export", () => {
  it("carries both the OpenCode 1.x and 2.x entry points", () => {
    expect(HeadroomOpenCodePlugin).toEqual({
      id: "headroom",
      server: HeadroomPlugin,
      setup: headroomSetup,
    });
  });
});


describe("OpenCode 2.x workspace session identity", () => {
  it.each([undefined, "explicit-session"])("carries the resolved session token through both routes (%s)", async (sessionToken) => {
    vi.stubEnv("HEADROOM_OPENCODE_SESSION_TOKEN", "environment-session");
    const models: FakeModel[] = [{
      id: "custom", providerID: "gateway",
      package: "@opencode/ai/providers/openai-compatible",
      settings: { baseURL: "https://gateway.example/v1" },
    }];
    const { context } = pluginContextV2({ sessionToken }, { models });
    const cleanup = await headroomSetup(context);
    try {
      const expected = sessionToken ?? "environment-session";
      expect(models[0].headers?.["x-headroom-session-token"]).toBe(expected);
      expect((globalThis as Record<symbol, { sessionToken?: string }>)[TRANSPORT_STATE]?.sessionToken).toBe(expected);
    } finally {
      await cleanup?.();
      vi.unstubAllEnvs();
    }
  });
});
