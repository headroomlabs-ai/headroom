import childProcess from "node:child_process";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { afterEach, describe, expect, it, vi } from "vitest";

import { HeadroomPlugin } from "./plugin.js";
import { ToolPolicyEnforcementError } from "./transport.js";

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

afterEach(() => {
  vi.restoreAllMocks();
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

    expect(output.env).toMatchObject({
      HEADROOM_ACTIVE: "1",
      HEADROOM_PROXY_URL: "http://127.0.0.1:8787",
      HEADROOM_PROJECT: "/repo",
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
      { ...input, args },
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
    ).rejects.toThrow(/did not match the bound preflight decision/);
    expect(
      stderr.mock.calls.some(([value]) =>
        String(value).includes("headroom_tool_policy_enforcement_acknowledgement"),
      ),
    ).toBe(false);
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

    expect(result).toBe("original content");
    expect(fetchMock).toHaveBeenCalledWith(
      "http://127.0.0.1:8787/v1/retrieve/0123456789abcdef01234567",
      expect.any(Object),
    );
  });
});
