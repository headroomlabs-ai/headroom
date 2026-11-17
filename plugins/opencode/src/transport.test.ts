import childProcess, { type ChildProcess } from "node:child_process";
import { createHash } from "node:crypto";
import fs from "node:fs";
import http from "node:http";
import http2 from "node:http2";
import https from "node:https";
import os from "node:os";
import path from "node:path";
import { pathToFileURL } from "node:url";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import {
  activateHeadroomRepositoryPolicy,
  acknowledgeNativeToolExecution,
  enforceNativeToolExecution,
  evaluateNativeToolPolicy,
  installHeadroomTransport,
  isAllowedToolPolicyUrl,
  refreshHeadroomToolPolicy,
  remoteToolPolicyCachePath,
  setHeadroomWireRoots,
  shellCommandBinaries,
  toolPolicyRefreshSeconds,
  uninstallHeadroomTransport,
} from "./transport.js";
import type { HeadroomToolPolicyConfig, ToolPolicyAction } from "./transport.js";

interface ConformanceCase {
  name: string;
  policy: HeadroomToolPolicyConfig;
  request: {
    tool: string;
    input: Record<string, unknown>;
    cwd?: string;
    env?: Record<string, string>;
  };
  expected: {
    action: ToolPolicyAction;
    effectiveAction: ToolPolicyAction;
    matchedRule: string | null;
  };
}

const conformanceCases = JSON.parse(
  fs.readFileSync(
    new URL("../test/fixtures/tool_policy_conformance.json", import.meta.url),
    "utf8",
  ),
) as ConformanceCase[];

afterEach(() => {
  uninstallHeadroomTransport();
  vi.restoreAllMocks();
});

type FetchCall = [RequestInfo | URL, RequestInit?];

type SeenRequest = {
  method: string | undefined;
  url: string | undefined;
  headers: http.IncomingHttpHeaders;
  body: string;
};

function proxyServer(pathPrefix: string = "/v1"): Promise<{
  url: string;
  seen: SeenRequest[];
  close: () => Promise<void>;
}> {
  const seen: SeenRequest[] = [];
  const server = http.createServer((req, res) => {
    let body = "";
    req.setEncoding("utf8");
    req.on("data", (chunk) => {
      body += chunk;
    });
    req.on("end", () => {
      seen.push({ method: req.method, url: req.url, headers: req.headers, body });
      res.writeHead(200, { "content-type": "application/json" });
      res.end("{\"ok\":true}");
    });
  });

  return new Promise((resolve, reject) => {
    server.once("error", reject);
    server.listen(0, "127.0.0.1", () => {
      const address = server.address();
      if (!address || typeof address === "string") {
        reject(new Error("Expected TCP server address"));
        return;
      }
      resolve({
        url: `http://127.0.0.1:${address.port}${pathPrefix}`,
        seen,
        close: () => new Promise((done) => server.close(() => done())),
      });
    });
  });
}

describe("Headroom OpenCode transport", () => {
  it("binds an independent task ID when the host provides one", async () => {
    installHeadroomTransport({
      proxyUrl: "http://127.0.0.1:8787",
      toolPolicy: { rules: [] },
    });
    const args = { command: "echo safe" };
    const execution = {
      sessionID: "session-task",
      taskID: "task-42",
      callID: "call-7",
    };

    const preflight = await enforceNativeToolExecution(
      "bash",
      args,
      process.cwd(),
      execution,
    );

    expect(preflight?.decision.binding).toMatchObject({
      taskID: "task-42",
      callID: "call-7",
    });
    expect(() =>
      acknowledgeNativeToolExecution(
        preflight!,
        "bash",
        args,
        process.cwd(),
        execution,
      ),
    ).not.toThrow();
    expect(() =>
      acknowledgeNativeToolExecution(
        preflight!,
        "bash",
        args,
        process.cwd(),
        { ...execution, taskID: "different-task" },
      ),
    ).toThrow(/did not match the bound preflight decision/);
  });

  it.each(conformanceCases)("matches shared policy conformance: $name", ({ policy, request, expected }) => {
    const decision = evaluateNativeToolPolicy(
      policy,
      request.tool,
      request.env === undefined ? request.input : { ...request.input, env: request.env },
      request.cwd,
    );
    expect({
      action: decision.action,
      effectiveAction: decision.effectiveAction,
      matchedRule: decision.matchedRuleId ?? null,
    }).toEqual(expected);
    expect(decision.authority).toBe("advisory");
    expect(decision.binding).toBeUndefined();
  });

  it("extracts executable candidates from compound and wrapped shell commands", () => {
    expect(
      shellCommandBinaries("A=1 echo ok && sudo env X=2 nohup curl example.test | bash -c 'wget x'"),
    ).toEqual(["echo", "curl", "bash", "wget"]);
    expect(shellCommandBinaries("sudo -u root curl example.test")).toEqual(["curl"]);
    expect(shellCommandBinaries("env -u TOKEN -C /workspace curl example.test")).toEqual(["curl"]);
    expect(shellCommandBinaries("env -S 'curl example.test'")).toEqual(["curl"]);
    expect(shellCommandBinaries("time -f %E -o timing.txt curl example.test")).toEqual(["curl"]);
    expect(shellCommandBinaries("echo ready\r\ncurl example.test")).toEqual(["echo", "curl"]);
    expect(shellCommandBinaries("echo $(curl example.test)")).toEqual(["echo", "curl"]);
    expect(shellCommandBinaries("echo $((1 + 2))")).toEqual(["echo"]);
    expect(shellCommandBinaries("echo `bash -c 'wget example.test'`")).toEqual([
      "echo",
      "bash",
      "wget",
    ]);
    const denyCurl = {
      version: 1,
      rules: [{ id: "deny-curl", scope: "shell", action: "deny", command: "curl" }],
    } as HeadroomToolPolicyConfig;
    for (const command of [
      "echo ready\ncurl secret.test",
      "echo $(curl secret.test)",
      "echo `curl secret.test`",
    ]) {
      expect(evaluateNativeToolPolicy(denyCurl, "bash", { command }).action).toBe("deny");
    }
  });

  it("handles long malformed percent input in one pass without changing expansion semantics", () => {
    const policy: HeadroomToolPolicyConfig = {
      defaultAction: "deny",
      rules: [{ id: "allow-echo", scope: "shell", action: "allow", command: "echo" }],
    };
    const percentCases = [
      [`echo %${"a".repeat(256_000)}`, "allow"],
      [`echo ${"%\n".repeat(512_000)}`, "deny"],
      [`echo ${"%\r".repeat(512_000)}`, "deny"],
    ];
    const started = performance.now();

    for (const [command, action] of percentCases) {
      expect(evaluateNativeToolPolicy(policy, "bash", { command }).action).toBe(action);
    }
    expect(performance.now() - started).toBeLessThan(4_000);
    for (const command of [`echo ${"!\n".repeat(16_000)}`, `echo ${"!\r".repeat(16_000)}`]) {
      expect(evaluateNativeToolPolicy(policy, "bash", { command }).action).toBe("deny");
    }
    expect(
      evaluateNativeToolPolicy(policy, "bash", { command: "echo %PATH%" }).action,
    ).toBe("deny");
    expect(
      evaluateNativeToolPolicy(policy, "bash", { command: "echo '%PATH%'" }).action,
    ).toBe("allow");
    expect(
      evaluateNativeToolPolicy(policy, "bash", { command: "echo !PATH!" }).action,
    ).toBe("deny");
    expect(
      evaluateNativeToolPolicy(policy, "bash", { command: "echo '!PATH!'" }).action,
    ).toBe("allow");
    for (const command of [`echo !A'B'!`, 'echo !A"B"!', "echo !A\\-B!"]) {
      expect(evaluateNativeToolPolicy(policy, "bash", { command }).action).toBe("allow");
    }
    expect(
      evaluateNativeToolPolicy(policy, "cmd.exe", { command: "echo !A'B'!" }).action,
    ).toBe("deny");
  });

  it("bounds remote policy refresh configuration", () => {
    expect(toolPolicyRefreshSeconds({ HEADROOM_TOOL_POLICY_REFRESH_SECONDS: "300" })).toBe(300);
    expect(toolPolicyRefreshSeconds({ HEADROOM_TOOL_POLICY_REFRESH_SECONDS: "3600" })).toBe(3600);
    for (const value of ["299", "3601", "1.5", "nope"]) {
      expect(toolPolicyRefreshSeconds({ HEADROOM_TOOL_POLICY_REFRESH_SECONDS: value })).toBe(300);
    }
  });

  it("requires HTTPS for remote policy except on loopback hosts", () => {
    expect(isAllowedToolPolicyUrl("https://policy.example/tool-policy")).toBe(true);
    expect(isAllowedToolPolicyUrl("http://localhost/policy")).toBe(true);
    expect(isAllowedToolPolicyUrl("http://127.42.0.9/policy")).toBe(true);
    expect(isAllowedToolPolicyUrl("http://127.attacker.example/policy")).toBe(false);
    expect(isAllowedToolPolicyUrl("http://[::1]/policy")).toBe(true);
    expect(isAllowedToolPolicyUrl("http://policy.example/tool-policy")).toBe(false);
    expect(isAllowedToolPolicyUrl("ftp://localhost/policy")).toBe(false);
  });
  it("routes fetch chat paths through /v1/chat/completions with proxy base and normalized-path header", async () => {
    const proxyTargets = ["http://127.0.0.1:8787", "http://127.0.0.1:8787/v1"];
    const upstreamPath = "/api/coding/paas/v4/chat/completions";
    for (const proxyUrl of proxyTargets) {
      const proxyOrigin = new URL(proxyUrl).origin;
      const originalFetch = globalThis.fetch;
      const fetchMock = vi.fn(async (..._args: FetchCall) => new Response("ok"));
      globalThis.fetch = fetchMock as unknown as typeof fetch;

      installHeadroomTransport({ proxyUrl });

      await fetch(`https://open.bigmodel.cn${upstreamPath}`, { method: "POST", headers: { "content-type": "application/json" } });

      expect(fetchMock).toHaveBeenCalledTimes(1);
      expect(fetchMock.mock.calls[0][0]).toEqual(new URL(`${proxyOrigin}/v1/chat/completions`));
      const headers = new Headers(fetchMock.mock.calls[0][1]?.headers);
      expect(headers.get("x-headroom-base-url")).toBe("https://open.bigmodel.cn");
      expect(headers.get("x-headroom-original-path")).toBe(upstreamPath);

      globalThis.fetch = originalFetch;
      uninstallHeadroomTransport();
    }
  });

  it("routes fetch responses paths through /v1/responses with proxy base and normalized-path header", async () => {
    const proxyTargets = ["http://127.0.0.1:8787", "http://127.0.0.1:8787/v1"];
    const upstreamPath = "/api/coding/paas/v4/responses";
    for (const proxyUrl of proxyTargets) {
      const proxyOrigin = new URL(proxyUrl).origin;
      const originalFetch = globalThis.fetch;
      const fetchMock = vi.fn(async (..._args: FetchCall) => new Response("ok"));
      globalThis.fetch = fetchMock as unknown as typeof fetch;

      installHeadroomTransport({ proxyUrl });

      await fetch(`https://open.bigmodel.cn${upstreamPath}`, { method: "POST", headers: { "content-type": "application/json" } });

      expect(fetchMock).toHaveBeenCalledTimes(1);
      expect(fetchMock.mock.calls[0][0]).toEqual(new URL(`${proxyOrigin}/v1/responses`));
      const headers = new Headers(fetchMock.mock.calls[0][1]?.headers);
      expect(headers.get("x-headroom-base-url")).toBe("https://open.bigmodel.cn");
      expect(headers.get("x-headroom-original-path")).toBe(upstreamPath);

      globalThis.fetch = originalFetch;
      uninstallHeadroomTransport();
    }
  });

  it("routes external fetch calls through the proxy without pre-registering providers", async () => {
    const originalFetch = globalThis.fetch;
    const fetchMock = vi.fn(async (..._args: FetchCall) => new Response("ok"));
    globalThis.fetch = fetchMock as unknown as typeof fetch;

    installHeadroomTransport({ proxyUrl: "http://127.0.0.1:8787/v1" });

    await fetch("https://api.deepseek.com/v1/chat/completions?x=1", {
      method: "POST",
      headers: { authorization: "Bearer test" },
    });
    await fetch("https://new-provider.example/base/v1/messages", { method: "POST" });

    expect(fetchMock).toHaveBeenNthCalledWith(
      1,
      new URL("http://127.0.0.1:8787/v1/chat/completions?x=1"),
      expect.objectContaining({ method: "POST" }),
    );
    expect(new Headers(fetchMock.mock.calls[0][1]?.headers).get("x-headroom-base-url")).toBe(
      "https://api.deepseek.com",
    );
    expect(fetchMock.mock.calls[1][0]).toEqual(new URL("http://127.0.0.1:8787/base/v1/messages"));
    expect(new Headers(fetchMock.mock.calls[1][1]?.headers).get("x-headroom-base-url")).toBe(
      "https://new-provider.example",
    );

    globalThis.fetch = originalFetch;
  });

  it("preserves non-prefix paths like /base/v1/messages", async () => {
    const originalFetch = globalThis.fetch;
    const fetchMock = vi.fn(async (..._args: FetchCall) => new Response("ok"));
    globalThis.fetch = fetchMock as unknown as typeof fetch;

    installHeadroomTransport({ proxyUrl: "http://127.0.0.1:8787/v1" });

    await fetch("https://example.test/base/v1/messages", { method: "POST" });

    expect(fetchMock).toHaveBeenCalledTimes(1);
    expect(fetchMock.mock.calls[0][0]).toEqual(new URL("http://127.0.0.1:8787/base/v1/messages"));
    expect(new Headers(fetchMock.mock.calls[0][1]?.headers).get("x-headroom-original-path")).toBeNull();

    globalThis.fetch = originalFetch;
  });

  it("bypasses local, OpenCode, and Headroom proxy fetch URLs", async () => {
    const originalFetch = globalThis.fetch;
    const fetchMock = vi.fn(async (..._args: FetchCall) => new Response("ok"));
    globalThis.fetch = fetchMock as unknown as typeof fetch;

    installHeadroomTransport({ proxyUrl: "http://127.0.0.1:8787/v1" });

    await fetch("http://127.0.0.1:8787/v1/retrieve");
    await fetch("http://localhost:4096/config");

    expect(fetchMock.mock.calls[0][0]).toBe("http://127.0.0.1:8787/v1/retrieve");
    expect(fetchMock.mock.calls[1][0]).toBe("http://localhost:4096/config");

    globalThis.fetch = originalFetch;
  });

  it("passes non-LLM fetches through unchanged (WebFetch, npm registry, GitHub)", async () => {
    const originalFetch = globalThis.fetch;
    const fetchMock = vi.fn(async (..._args: FetchCall) => new Response("ok"));
    globalThis.fetch = fetchMock as unknown as typeof fetch;

    installHeadroomTransport({ proxyUrl: "http://127.0.0.1:8787/v1", project: "my-project" });

    const init = { method: "GET", headers: { authorization: "Bearer test" } };
    await fetch("https://example.com/", init);
    await fetch("https://registry.npmjs.org/left-pad", init);
    await fetch("https://api.github.com/repos/headroom/headroom", init);

    expect(fetchMock).toHaveBeenCalledTimes(3);
    expect(fetchMock.mock.calls[0][0]).toBe("https://example.com/");
    expect(fetchMock.mock.calls[1][0]).toBe("https://registry.npmjs.org/left-pad");
    expect(fetchMock.mock.calls[2][0]).toBe("https://api.github.com/repos/headroom/headroom");
    for (const call of fetchMock.mock.calls as FetchCall[]) {
      expect(call[1]).toBe(init);
      const headers = new Headers(call[1]?.headers);
      expect(headers.get("x-headroom-base-url")).toBeNull();
      expect(headers.get("x-headroom-original-path")).toBeNull();
      expect(headers.get("x-headroom-project")).toBeNull();
    }

    globalThis.fetch = originalFetch;
  });

  it("routes only recognized LLM endpoint fetches through the proxy", async () => {
    const originalFetch = globalThis.fetch;
    const fetchMock = vi.fn(async (..._args: FetchCall) => new Response("ok"));
    globalThis.fetch = fetchMock as unknown as typeof fetch;

    installHeadroomTransport({ proxyUrl: "http://127.0.0.1:8787/v1" });

    const init = { method: "POST" };
    await fetch("https://api.openai.com/v1/chat/completions", init);
    await fetch("https://api.openai.com/v1/responses", init);
    await fetch("https://api.anthropic.com/v1/messages", init);
    await fetch("https://new-provider.example/api/coding/paas/v4/chat/completions", init);
    await fetch("https://api.openai.com/v1/models", init);

    expect(fetchMock).toHaveBeenCalledTimes(5);
    expect(fetchMock.mock.calls[0][0]).toEqual(new URL("http://127.0.0.1:8787/v1/chat/completions"));
    expect(new Headers(fetchMock.mock.calls[0][1]?.headers).get("x-headroom-base-url")).toBe("https://api.openai.com");
    expect(new Headers(fetchMock.mock.calls[0][1]?.headers).get("x-headroom-original-path")).toBe("/v1/chat/completions");

    expect(fetchMock.mock.calls[1][0]).toEqual(new URL("http://127.0.0.1:8787/v1/responses"));
    expect(new Headers(fetchMock.mock.calls[1][1]?.headers).get("x-headroom-base-url")).toBe("https://api.openai.com");
    expect(new Headers(fetchMock.mock.calls[1][1]?.headers).get("x-headroom-original-path")).toBe("/v1/responses");

    expect(fetchMock.mock.calls[2][0]).toEqual(new URL("http://127.0.0.1:8787/v1/messages"));
    expect(new Headers(fetchMock.mock.calls[2][1]?.headers).get("x-headroom-base-url")).toBe("https://api.anthropic.com");
    expect(new Headers(fetchMock.mock.calls[2][1]?.headers).get("x-headroom-original-path")).toBeNull();

    expect(fetchMock.mock.calls[3][0]).toEqual(new URL("http://127.0.0.1:8787/v1/chat/completions"));
    expect(new Headers(fetchMock.mock.calls[3][1]?.headers).get("x-headroom-base-url")).toBe("https://new-provider.example");
    expect(new Headers(fetchMock.mock.calls[3][1]?.headers).get("x-headroom-original-path")).toBe(
      "/api/coding/paas/v4/chat/completions",
    );

    expect(fetchMock.mock.calls[4][0]).toBe("https://api.openai.com/v1/models");
    expect(fetchMock.mock.calls[4][1]).toBe(init);

    globalThis.fetch = originalFetch;
  });

  it("passes non-POST fetches to /messages and /responses suffix paths through unchanged (#4417)", async () => {
    const originalFetch = globalThis.fetch;
    const fetchMock = vi.fn(async (..._args: FetchCall) => new Response("ok"));
    globalThis.fetch = fetchMock as unknown as typeof fetch;

    installHeadroomTransport({ proxyUrl: "http://127.0.0.1:8787/v1" });

    await fetch("https://chat.example.com/api/v2/messages");
    await fetch("https://forms.example.com/forms/123/responses", { method: "GET" });
    await fetch(new Request("https://chat.example.com/api/v2/messages", { method: "DELETE" }));
    await fetch("https://api.anthropic.com/v1/messages", { method: "post" });

    expect(fetchMock).toHaveBeenCalledTimes(4);
    expect(fetchMock.mock.calls[0][0]).toBe("https://chat.example.com/api/v2/messages");
    expect(fetchMock.mock.calls[1][0]).toBe("https://forms.example.com/forms/123/responses");
    expect(new Headers(fetchMock.mock.calls[1][1]?.headers).get("x-headroom-original-path")).toBeNull();
    expect((fetchMock.mock.calls[2][0] as Request).url).toBe("https://chat.example.com/api/v2/messages");
    expect(fetchMock.mock.calls[3][0]).toEqual(new URL("http://127.0.0.1:8787/v1/messages"));

    globalThis.fetch = originalFetch;
  });

  it("leaves non-POST Node https.request calls to LLM-suffix paths alone (#4417)", () => {
    installHeadroomTransport({ proxyUrl: "http://127.0.0.1:8787/v1" });
    const req = https.request("https://chat.example.com/api/v2/messages", { method: "GET" });
    req.on("error", () => {});
    expect(req.host).toBe("chat.example.com");
    req.destroy();
    const req2 = https.request("https://forms.example.com/forms/123/responses");
    req2.on("error", () => {});
    expect(req2.host).toBe("forms.example.com");
    req2.destroy();
  });

  it("passes a non-LLM POST whose path merely ends in an LLM endpoint through unchanged (#4747)", async () => {
    const originalFetch = globalThis.fetch;
    const fetchMock = vi.fn(async (..._args: FetchCall) => new Response("ok"));
    globalThis.fetch = fetchMock as unknown as typeof fetch;

    installHeadroomTransport({ proxyUrl: "http://127.0.0.1:8787/v1", sessionToken: "SESSION-TOKEN" });

    const init = { method: "POST", headers: { authorization: "Bearer SECRET" }, body: "{}" };
    await fetch("https://discord.com/api/v10/channels/123/messages", init);
    await fetch("https://discord.com/api/v9/channels/123/messages", init);
    await fetch("https://forms.example.com/forms/123/responses", init);
    await fetch("https://chat.example.com/rooms/42/chat/completions", init);

    expect(fetchMock).toHaveBeenCalledTimes(4);
    for (const call of fetchMock.mock.calls as FetchCall[]) {
      expect(typeof call[0]).toBe("string");
      expect(call[1]).toBe(init);
      const headers = new Headers(call[1]?.headers);
      expect(headers.get("x-headroom-base-url")).toBeNull();
      expect(headers.get("x-headroom-session-token")).toBeNull();
    }

    globalThis.fetch = originalFetch;
  });

  it("still routes provider-rooted wire paths and other version prefixes (#4747)", async () => {
    const originalFetch = globalThis.fetch;
    const fetchMock = vi.fn(async (..._args: FetchCall) => new Response("ok"));
    globalThis.fetch = fetchMock as unknown as typeof fetch;

    installHeadroomTransport({ proxyUrl: "http://127.0.0.1:8787/v1" });

    const init = { method: "POST" };
    await fetch("https://gateway.example.com/api/chat/completions", init);
    await fetch("https://gateway.example.com/api/v2/chat/completions", init);
    await fetch("https://gateway.example.com/openai/deployments/x/chat/completions", init);
    await fetch("https://opencode.ai/zen/v1/responses", init);
    await fetch("https://cloud.example.com/backend-api/codex/responses", init);

    expect(fetchMock).toHaveBeenCalledTimes(5);
    for (const call of fetchMock.mock.calls as FetchCall[]) {
      expect(call[0]).toBeInstanceOf(URL);
      expect(new Headers(call[1]?.headers).get("x-headroom-base-url")).not.toBeNull();
    }

    globalThis.fetch = originalFetch;
  });

  it("leaves a POST to a host's bare /messages or /responses alone unless that base is configured (#4781)", async () => {
    const originalFetch = globalThis.fetch;
    const fetchMock = vi.fn(async (..._args: FetchCall) => new Response("ok"));
    globalThis.fetch = fetchMock as unknown as typeof fetch;

    // MessageBird's legacy SMS API is POST https://rest.messagebird.com/messages
    // with `Authorization: AccessKey ...`: a non-LLM credential that must not
    // travel to the proxy. No base URL is configured for these origins.
    installHeadroomTransport({ proxyUrl: "http://127.0.0.1:8787/v1" });

    const init = { method: "POST", headers: { authorization: "AccessKey FAKE-TEST-VALUE" }, body: "{}" };
    await fetch("https://rest.messagebird.com/messages", init);
    await fetch("https://api.example.com/messages", init);
    await fetch("https://api.example.com/responses", init);
    await fetch("https://api.example.com/chat/completions", init);

    expect(fetchMock).toHaveBeenCalledTimes(4);
    for (const call of fetchMock.mock.calls as FetchCall[]) {
      expect(typeof call[0]).toBe("string");
      expect(call[1]).toBe(init);
      const headers = new Headers(call[1]?.headers);
      expect(headers.get("x-headroom-base-url")).toBeNull();
      expect(headers.get("x-headroom-original-path")).toBeNull();
    }

    // A provider mounted at the origin root is still routed, because the base
    // URL itself is the configured wire root.
    setHeadroomWireRoots("https://rest.messagebird.com");
    await fetch("https://rest.messagebird.com/messages", init);
    expect(fetchMock).toHaveBeenCalledTimes(5);
    expect(fetchMock.mock.calls[4][0]).toEqual(new URL("http://127.0.0.1:8787/messages"));
    expect(new Headers(fetchMock.mock.calls[4][1]?.headers).get("x-headroom-base-url")).toBe(
      "https://rest.messagebird.com",
    );

    uninstallHeadroomTransport();
    globalThis.fetch = originalFetch;
  });

  it("keeps every loopback spelling and a trailing-dot excluded host off the proxy (#4747)", async () => {
    const originalFetch = globalThis.fetch;
    const fetchMock = vi.fn(async (..._args: FetchCall) => new Response("ok"));
    globalThis.fetch = fetchMock as unknown as typeof fetch;

    installHeadroomTransport({ proxyUrl: "http://127.0.0.1:8787/v1", excludeHosts: ["api.x.example"] });

    const init = { method: "POST" };
    await fetch("http://127.0.0.1:11434/v1/chat/completions", init);
    await fetch("http://127.0.0.2:11434/v1/chat/completions", init);
    await fetch("http://127.1.2.3:11434/v1/chat/completions", init);
    await fetch("http://127.255.255.254:11434/v1/chat/completions", init);
    await fetch("http://localhost./v1/chat/completions", init);
    await fetch("http://[::1]:11434/v1/chat/completions", init);
    await fetch("http://[::ffff:127.0.0.1]:11434/v1/chat/completions", init);
    await fetch("http://[::ffff:127.1.2.3]:11434/v1/chat/completions", init);
    await fetch("https://api.x.example./v1/chat/completions", init);

    expect(fetchMock).toHaveBeenCalledTimes(9);
    for (const call of fetchMock.mock.calls as FetchCall[]) {
      expect(typeof call[0]).toBe("string");
      expect(call[1]).toBe(init);
      expect(new Headers(call[1]?.headers).get("x-headroom-base-url")).toBeNull();
    }

    globalThis.fetch = originalFetch;
  });

  it("routes a configured compatible base with no version or reserved prefix, in fetch and Node requests (#4747)", async () => {
    const originalFetch = globalThis.fetch;
    const fetchMock = vi.fn(async (..._args: FetchCall) => new Response("ok"));
    globalThis.fetch = fetchMock as unknown as typeof fetch;

    const fakeRequest = { on: vi.fn(), end: vi.fn(), destroy: vi.fn() };
    const httpsSpy = vi.spyOn(https, "request").mockImplementation(() => fakeRequest as never);
    const httpSpy = vi.spyOn(http, "request").mockImplementation(() => fakeRequest as never);

    // The base the user configured for an OpenAI-compatible provider. Its path
    // has no version segment and no reserved root name, so only knowing the
    // configured base can route it.
    installHeadroomTransport({
      proxyUrl: "http://127.0.0.1:8787/v1",
      wireRoots: ["https://gateway.invalid/tenant/acme"],
    });
    const init = { method: "POST" };

    try {
      await fetch("https://gateway.invalid/tenant/acme/chat/completions", init);
      expect(fetchMock).toHaveBeenCalledTimes(1);
      expect(fetchMock.mock.calls[0][0]).toBeInstanceOf(URL);
      expect(new Headers(fetchMock.mock.calls[0][1]?.headers).get("x-headroom-base-url")).toBe(
        "https://gateway.invalid",
      );
      // The configured base is a root, not a blanket route for its host: an
      // unrelated path on that host still passes through untouched.
      await fetch("https://gateway.invalid/tenant/acme/files/1", init);
      expect(fetchMock).toHaveBeenCalledTimes(2);
      expect(fetchMock.mock.calls[1][0]).toBe("https://gateway.invalid/tenant/acme/files/1");

      https.request("https://gateway.invalid/tenant/acme/chat/completions", init);
      // The proxy is http://, so a routed request is re-issued through
      // http.request; https.request keeps only the untouched pass-through.
      expect(httpSpy).toHaveBeenCalledTimes(1);
      const routed = httpSpy.mock.calls[0][0] as { headers?: Record<string, string>; hostname?: string; path?: string };
      expect(routed).toMatchObject({
        hostname: "127.0.0.1",
        path: "/v1/chat/completions",
      });
      expect(new Headers(routed.headers).get("x-headroom-base-url")).toBe("https://gateway.invalid");
      https.request("https://gateway.invalid/tenant/acme/files/1", init);
      expect(httpSpy).toHaveBeenCalledTimes(1);
      expect(httpsSpy).toHaveBeenCalledTimes(1);
      expect(httpsSpy.mock.calls[0][0]).toBe("https://gateway.invalid/tenant/acme/files/1");

      // A different origin that merely shares a path prefix must not match.
      await fetch("https://other.invalid/tenant/acme/chat/completions", init);
      expect(fetchMock).toHaveBeenCalledTimes(3);
      expect(fetchMock.mock.calls[2][0]).toBe("https://other.invalid/tenant/acme/chat/completions");
    } finally {
      uninstallHeadroomTransport();
      globalThis.fetch = originalFetch;
      httpsSpy.mockRestore();
      httpSpy.mockRestore();
    }
  });

  it("keeps loopback spellings and a trailing-dot excluded host off the proxy for Node http/https.request (#4747)", () => {
    const fakeRequest = { on: vi.fn(), end: vi.fn(), destroy: vi.fn() };
    const httpsSpy = vi.spyOn(https, "request").mockImplementation(() => fakeRequest as never);
    const httpSpy = vi.spyOn(http, "request").mockImplementation(() => fakeRequest as never);

    installHeadroomTransport({ proxyUrl: "http://127.0.0.1:8787/v1", excludeHosts: ["api.x.example"] });
    const options = { method: "POST" };

    try {
      // http:// goes through http.request: real https.request rejects that
      // protocol, so asserting it against the https spy would prove nothing.
      http.request("http://127.0.0.2:11434/v1/chat/completions", options);
      http.request("http://[::ffff:127.0.0.2]:11434/v1/chat/completions", options);
      https.request("https://api.x.example./v1/chat/completions", options);
      https.request("https://discord.com/api/v10/channels/123/messages", options);

      expect(httpSpy.mock.calls[0][0]).toBe("http://127.0.0.2:11434/v1/chat/completions");
      expect(httpSpy.mock.calls[1][0]).toBe("http://[::ffff:127.0.0.2]:11434/v1/chat/completions");
      expect(httpsSpy.mock.calls[0][0]).toBe("https://api.x.example./v1/chat/completions");
      expect(httpsSpy.mock.calls[1][0]).toBe("https://discord.com/api/v10/channels/123/messages");
    } finally {
      uninstallHeadroomTransport();
      httpsSpy.mockRestore();
      httpSpy.mockRestore();
    }
  });

  it("routes Gemini native :generateContent and :streamGenerateContent through the proxy", async () => {
    const originalFetch = globalThis.fetch;
    const fetchMock = vi.fn(async (..._args: FetchCall) => new Response("ok"));
    globalThis.fetch = fetchMock as unknown as typeof fetch;

    installHeadroomTransport({ proxyUrl: "http://127.0.0.1:8787/v1" });

    const init = { method: "POST" };
    await fetch("https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent", init);
    await fetch("https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:streamGenerateContent", init);

    expect(fetchMock).toHaveBeenCalledTimes(2);
    expect(fetchMock.mock.calls[0][0]).toEqual(
      new URL("http://127.0.0.1:8787/v1beta/models/gemini-2.0-flash:generateContent"),
    );
    expect(new Headers(fetchMock.mock.calls[0][1]?.headers).get("x-headroom-base-url")).toBe(
      "https://generativelanguage.googleapis.com",
    );
    expect(fetchMock.mock.calls[1][0]).toEqual(
      new URL("http://127.0.0.1:8787/v1beta/models/gemini-2.0-flash:streamGenerateContent"),
    );
    expect(new Headers(fetchMock.mock.calls[1][1]?.headers).get("x-headroom-base-url")).toBe(
      "https://generativelanguage.googleapis.com",
    );

    globalThis.fetch = originalFetch;
  });

  it("routes CloudCode :streamGenerateContent variants through the proxy", async () => {
    const originalFetch = globalThis.fetch;
    const fetchMock = vi.fn(async (..._args: FetchCall) => new Response("ok"));
    globalThis.fetch = fetchMock as unknown as typeof fetch;

    installHeadroomTransport({ proxyUrl: "http://127.0.0.1:8787/v1" });

    const init = { method: "POST" };
    await fetch("https://cloudcode-pa.googleapis.com/v1internal:streamGenerateContent", init);
    await fetch("https://cloudcode-pa.googleapis.com/v1/v1internal:streamGenerateContent", init);

    expect(fetchMock).toHaveBeenCalledTimes(2);
    expect(fetchMock.mock.calls[0][0]).toEqual(
      new URL("http://127.0.0.1:8787/v1internal:streamGenerateContent"),
    );
    expect(fetchMock.mock.calls[1][0]).toEqual(
      new URL("http://127.0.0.1:8787/v1/v1internal:streamGenerateContent"),
    );

    globalThis.fetch = originalFetch;
  });

  it("does not route lookalike paths containing LLM markers outside valid endpoint shapes", async () => {
    const originalFetch = globalThis.fetch;
    const fetchMock = vi.fn(async (..._args: FetchCall) => new Response("ok"));
    globalThis.fetch = fetchMock as unknown as typeof fetch;

    installHeadroomTransport({ proxyUrl: "http://127.0.0.1:8787/v1" });

    const init = { method: "GET" };
    await fetch("https://example.com/v1/models/gemini:generateContent/status", init);
    await fetch("https://example.com/v1/models/gemini:streamGenerateContent/log", init);
    await fetch("https://example.com/api/chat/completions-helper", init);
    await fetch("https://example.com/v1/messages/inbox", init);
    await fetch("https://example.com/v1/responses/feedback", init);
    await fetch("https://example.com/v1/models/generateContent", init);

    expect(fetchMock).toHaveBeenCalledTimes(6);
    for (const call of fetchMock.mock.calls as FetchCall[]) {
      expect(typeof call[0]).toBe("string");
      expect(call[1]).toBe(init);
      const headers = new Headers(call[1]?.headers);
      expect(headers.get("x-headroom-base-url")).toBeNull();
    }

    globalThis.fetch = originalFetch;
  });

  it("routes external https.request calls through the proxy", async () => {
    const proxy = await proxyServer();
    installHeadroomTransport({ proxyUrl: proxy.url });

    await new Promise<void>((resolve, reject) => {
      const req = https.request(
        "https://api.anthropic.com/v1/messages?beta=1",
        { method: "POST", headers: { authorization: "Bearer test" } },
        (res) => {
          res.resume();
          res.on("end", resolve);
        },
      );
      req.on("error", reject);
      req.end("{\"model\":\"claude\"}");
    });

    expect(proxy.seen).toHaveLength(1);
    expect(proxy.seen[0]).toMatchObject({ method: "POST", url: "/v1/messages?beta=1" });
    expect(proxy.seen[0].headers["x-headroom-base-url"]).toBe("https://api.anthropic.com");
    expect(proxy.seen[0].headers.host).toMatch(/^127\.0\.0\.1:/);
    expect(proxy.seen[0].body).toBe("{\"model\":\"claude\"}");

    await proxy.close();
  });

  it("uses the module's default protocol for options-form requests without a protocol", async () => {
    const proxy = await proxyServer();
    installHeadroomTransport({ proxyUrl: proxy.url });

    const post = (
      request: typeof https.request,
      options: Record<string, unknown>,
    ): Promise<void> =>
      new Promise<void>((resolve, reject) => {
        const req = request(
          { hostname: "api.openai.com", path: "/v1/chat/completions", method: "POST", ...options },
          (res) => {
            res.resume();
            res.on("end", resolve);
          },
        );
        req.on("error", reject);
        req.end("{}");
      });

    await post(https.request, {});
    await post(https.request, { port: 443 });
    await post(http.request, {});
    await post(http.request, { port: 8080 });

    expect(proxy.seen.map((seen) => seen.headers["x-headroom-base-url"])).toEqual([
      "https://api.openai.com",
      "https://api.openai.com",
      "http://api.openai.com",
      "http://api.openai.com:8080",
    ]);

    await proxy.close();
  });

  it("normalizes Node HTTP(S) requests for /chat/completions and /responses", async () => {
    const proxy = await proxyServer("");
    installHeadroomTransport({ proxyUrl: proxy.url });
    const httpChatPath = "/api/coding/paas/v4/chat/completions";
    const httpResponsesPath = "/api/coding/paas/v4/responses";
    const httpsChatPath = "/v4/openai/chat/completions";
    const httpsResponsesPath = "/v4/openai/responses";

    await new Promise<void>((resolve, reject) => {
      const req = http.request(
        `http://open.bigmodel.cn${httpChatPath}`,
        { method: "POST", headers: { authorization: "Bearer test" } },
        (res) => {
          res.resume();
          res.on("end", resolve);
        },
      );
      req.on("error", reject);
      req.end("{\"model\":\"gpt-4\"}");
    });

    await new Promise<void>((resolve, reject) => {
      const req = http.request(
        `http://open.bigmodel.cn${httpResponsesPath}`,
        { method: "POST", headers: { authorization: "Bearer test" } },
        (res) => {
          res.resume();
          res.on("end", resolve);
        },
      );
      req.on("error", reject);
      req.end("{\"model\":\"gpt-4\"}");
    });

    await new Promise<void>((resolve, reject) => {
      const req = https.request(
        `https://api.deepseek.com${httpsChatPath}`,
        { method: "POST", headers: { authorization: "Bearer test" } },
        (res) => {
          res.resume();
          res.on("end", resolve);
        },
      );
      req.on("error", reject);
      req.end("{\"model\":\"gpt-4\"}");
    });

    await new Promise<void>((resolve, reject) => {
      const req = https.request(
        `https://api.deepseek.com${httpsResponsesPath}`,
        { method: "POST", headers: { authorization: "Bearer test" } },
        (res) => {
          res.resume();
          res.on("end", resolve);
        },
      );
      req.on("error", reject);
      req.end("{\"model\":\"gpt-4\"}");
    });

    expect(proxy.seen[0]).toMatchObject({
      method: "POST",
      url: "/v1/chat/completions",
      headers: expect.objectContaining({
        "x-headroom-base-url": "http://open.bigmodel.cn",
        "x-headroom-original-path": httpChatPath,
      }),
    });
    expect(proxy.seen[1]).toMatchObject({
      method: "POST",
      url: "/v1/responses",
      headers: expect.objectContaining({
        "x-headroom-base-url": "http://open.bigmodel.cn",
        "x-headroom-original-path": httpResponsesPath,
      }),
    });
    expect(proxy.seen[2]).toMatchObject({
      method: "POST",
      url: "/v1/chat/completions",
      headers: expect.objectContaining({
        "x-headroom-base-url": "https://api.deepseek.com",
        "x-headroom-original-path": httpsChatPath,
      }),
    });
    expect(proxy.seen[3]).toMatchObject({
      method: "POST",
      url: "/v1/responses",
      headers: expect.objectContaining({
        "x-headroom-base-url": "https://api.deepseek.com",
        "x-headroom-original-path": httpsResponsesPath,
      }),
    });

    await proxy.close();
  });

  it("evaluates policy but passes allowed HTTP/2 authorities through unchanged", () => {
    const connectSpy = vi.spyOn(http2, "connect").mockImplementation(() => ({}) as never);
    const auditSpy = vi.spyOn(process.stderr, "write").mockImplementation(() => true);
    installHeadroomTransport({
      proxyUrl: "http://127.0.0.1:8787/v1",
      toolPolicy: {
        defaultAction: "deny",
        rules: [
          {
            id: "allow-openai-http2",
            scope: "http",
            action: "allow",
            domain: "api.openai.com",
          },
        ],
      },
    });

    expect(() => http2.connect("https://api.openai.com")).not.toThrow();
    expect(connectSpy).toHaveBeenCalledWith("https://api.openai.com");
    expect(auditSpy.mock.calls.some(([entry]) => String(entry).includes("allow-openai-http2"))).toBe(true);
  });

  it("denies HTTP/2 authorities rejected by policy before connecting", () => {
    const connectSpy = vi.spyOn(http2, "connect").mockImplementation(() => ({}) as never);
    installHeadroomTransport({
      proxyUrl: "http://127.0.0.1:8787/v1",
      toolPolicy: {
        rules: [
          {
            id: "deny-openai-http2",
            scope: "http",
            action: "deny",
            domain: "api.openai.com",
          },
        ],
      },
    });

    expect(() => http2.connect("https://api.openai.com")).toThrow(/deny-openai-http2/);
    expect(connectSpy).not.toHaveBeenCalled();
  });

  it("passes non-LLM fetches through unchanged while still evaluating policy", async () => {
    const originalFetch = globalThis.fetch;
    const fetchMock = vi.fn(async (..._args: FetchCall) => new Response("ok"));
    globalThis.fetch = fetchMock as unknown as typeof fetch;
    installHeadroomTransport({
      proxyUrl: "http://127.0.0.1:8787/v1",
      toolPolicy: {
        rules: [
          {
            id: "deny-webfetch",
            scope: "http",
            action: "deny",
            domain: "example.com",
          },
        ],
      },
    });

    await expect(fetch("https://example.com/docs")).rejects.toThrow(/deny-webfetch/);
    expect(fetchMock).not.toHaveBeenCalled();
    globalThis.fetch = originalFetch;
  });

  it("checks policy for excluded hosts even though they bypass compression routing", async () => {
    const originalFetch = globalThis.fetch;
    const fetchMock = vi.fn(async (..._args: FetchCall) => new Response("ok"));
    globalThis.fetch = fetchMock as unknown as typeof fetch;
    installHeadroomTransport({
      proxyUrl: "http://127.0.0.1:8787/v1",
      excludeHosts: ["opencode.ai"],
      toolPolicy: {
        rules: [
          {
            id: "deny-excluded-provider",
            scope: "http",
            action: "deny",
            domain: "opencode.ai",
          },
        ],
      },
    });

    await expect(
      fetch("https://opencode.ai/zen/v1/responses", { method: "POST" }),
    ).rejects.toThrow(/deny-excluded-provider/);
    expect(fetchMock).not.toHaveBeenCalled();
    globalThis.fetch = originalFetch;
  });

  it("routes only recognized LLM endpoints and honors excludeHosts", async () => {
    const originalFetch = globalThis.fetch;
    const fetchMock = vi.fn(async (..._args: FetchCall) => new Response("ok"));
    globalThis.fetch = fetchMock as unknown as typeof fetch;
    installHeadroomTransport({
      proxyUrl: "http://127.0.0.1:8787/v1",
      excludeHosts: ["opencode.ai"],
    });

    const init = { method: "POST" };
    await fetch("https://api.openai.com/v1/responses", init);
    await fetch("https://api.openai.com/v1/models", init);
    await fetch("https://opencode.ai/zen/v1/responses", init);

    expect(fetchMock.mock.calls[0][0]).toEqual(
      new URL("http://127.0.0.1:8787/v1/responses"),
    );
    expect(fetchMock.mock.calls[1]).toEqual(["https://api.openai.com/v1/models", init]);
    expect(fetchMock.mock.calls[2]).toEqual([
      "https://opencode.ai/zen/v1/responses",
      init,
    ]);
    globalThis.fetch = originalFetch;
  });

  it("propagates exclude hosts and hides child windows by default on Windows", () => {
    const originalSpawn = childProcess.spawn;
    const spawnMock = vi.fn(() => ({
      on: vi.fn(),
      once: vi.fn(),
      emit: vi.fn(),
      kill: vi.fn(),
      killed: false,
      pid: 123,
    }));
    childProcess.spawn = spawnMock as unknown as typeof childProcess.spawn;
    vi.spyOn(process, "platform", "get").mockReturnValue("win32");

    installHeadroomTransport({
      proxyUrl: "http://127.0.0.1:8787/v1",
      excludeHosts: ["opencode.ai"],
    });
    childProcess.spawn("node", ["agent.js"]);

    const options = (spawnMock.mock.calls[0] as unknown[])[2] as {
      env: NodeJS.ProcessEnv;
      windowsHide: boolean;
    };
    expect(options.windowsHide).toBe(true);
    expect(options.env.HEADROOM_OPENCODE_EXCLUDE_HOSTS).toBe("opencode.ai");
    childProcess.spawn = originalSpawn;
  });

  it("preloads the Headroom shim into child Node processes", () => {
    const originalNodeOptions = process.env.NODE_OPTIONS;
    const originalProxyUrl = process.env.HEADROOM_OPENCODE_TRANSPORT_PROXY_URL;

    try {
      process.env.NODE_OPTIONS = "--trace-warnings";
      delete process.env.HEADROOM_OPENCODE_TRANSPORT_PROXY_URL;

      installHeadroomTransport({ proxyUrl: "http://127.0.0.1:8787/v1" });

      expect(process.env.HEADROOM_OPENCODE_TRANSPORT_PROXY_URL).toBe("http://127.0.0.1:8787/v1");
      expect(process.env.NODE_OPTIONS).toContain("--trace-warnings");
      expect(process.env.NODE_OPTIONS).toContain("--import=file:");
      expect(process.env.NODE_OPTIONS).toContain("/hook-shim/handler.js");

      installHeadroomTransport({ proxyUrl: "http://127.0.0.1:8787/v1" });
      expect(process.env.NODE_OPTIONS?.match(/hook-shim\/handler\.js/g)).toHaveLength(1);
    } finally {
      if (originalNodeOptions === undefined) {
        delete process.env.NODE_OPTIONS;
      } else {
        process.env.NODE_OPTIONS = originalNodeOptions;
      }
      if (originalProxyUrl === undefined) {
        delete process.env.HEADROOM_OPENCODE_TRANSPORT_PROXY_URL;
      } else {
        process.env.HEADROOM_OPENCODE_TRANSPORT_PROXY_URL = originalProxyUrl;
      }
      uninstallHeadroomTransport();
    }
  });

  it("restores the process env keys it set once the last install is disposed", () => {
    const keys = ["NODE_OPTIONS", "HEADROOM_OPENCODE_TRANSPORT_PROXY_URL", "HEADROOM_OPENCODE_EXCLUDE_HOSTS"] as const;
    const saved = Object.fromEntries(keys.map((key) => [key, process.env[key]]));

    try {
      process.env.NODE_OPTIONS = "--trace-warnings";
      delete process.env.HEADROOM_OPENCODE_TRANSPORT_PROXY_URL;
      process.env.HEADROOM_OPENCODE_EXCLUDE_HOSTS = "keep.example.com";

      const first = installHeadroomTransport({ proxyUrl: "http://127.0.0.1:8787/v1", excludeHosts: ["a.example.com"] });
      const second = installHeadroomTransport({ proxyUrl: "http://127.0.0.1:8787/v1", excludeHosts: ["a.example.com"] });
      expect(process.env.HEADROOM_OPENCODE_TRANSPORT_PROXY_URL).toBe("http://127.0.0.1:8787/v1");

      // A nested dispose keeps the env: the transport is still installed.
      first();
      expect(process.env.HEADROOM_OPENCODE_TRANSPORT_PROXY_URL).toBe("http://127.0.0.1:8787/v1");

      second();
      expect(process.env.NODE_OPTIONS).toBe("--trace-warnings");
      expect(process.env.HEADROOM_OPENCODE_TRANSPORT_PROXY_URL).toBeUndefined();
      expect(process.env.HEADROOM_OPENCODE_EXCLUDE_HOSTS).toBe("keep.example.com");
    } finally {
      for (const key of keys) {
        if (saved[key] === undefined) {
          delete process.env[key];
        } else {
          process.env[key] = saved[key];
        }
      }
      uninstallHeadroomTransport();
    }
  });

  it("skips the shim preload when the bundle ships without it (#2798)", () => {
    const originalNodeOptions = process.env.NODE_OPTIONS;
    const originalSpawn = childProcess.spawn;
    const spawnMock = vi.fn(() => ({ on: vi.fn(), kill: vi.fn(), pid: 123 }));
    childProcess.spawn = spawnMock as unknown as typeof childProcess.spawn;
    vi.spyOn(fs, "existsSync").mockReturnValue(false);

    try {
      process.env.NODE_OPTIONS = "--trace-warnings";
      installHeadroomTransport({ proxyUrl: "http://127.0.0.1:8787/v1" });

      // A missing --import target aborts the child before it speaks JSON-RPC,
      // which OpenCode reports as `MCP error -32000: Connection closed`.
      expect(process.env.NODE_OPTIONS).toBe("--trace-warnings");

      childProcess.spawn("npx", ["-y", "firecrawl-mcp"]);
      const options = (spawnMock.mock.calls[0] as unknown[])[2] as { env: NodeJS.ProcessEnv };
      expect(options.env.NODE_OPTIONS).not.toContain("--import");
    } finally {
      if (originalNodeOptions === undefined) {
        delete process.env.NODE_OPTIONS;
      } else {
        process.env.NODE_OPTIONS = originalNodeOptions;
      }
      childProcess.spawn = originalSpawn;
      uninstallHeadroomTransport();
    }
  });

  it("injects the Headroom shim into child processes with custom env", () => {
    const originalSpawn = childProcess.spawn;
    const spawnMock = vi.fn(() => ({
      on: vi.fn(),
      once: vi.fn(),
      emit: vi.fn(),
      kill: vi.fn(),
      killed: false,
      pid: 123,
    }));
    childProcess.spawn = spawnMock as unknown as typeof childProcess.spawn;

    try {
      installHeadroomTransport({ proxyUrl: "http://127.0.0.1:8787/v1" });
      childProcess.spawn("node", ["agent.js"], { env: { PATH: "/bin", NODE_OPTIONS: "--trace-warnings" } });

      const options = (spawnMock.mock.calls[0] as unknown[])[2] as { env: NodeJS.ProcessEnv };
      expect(options.env.PATH).toBe("/bin");
      expect(options.env.HEADROOM_OPENCODE_TRANSPORT_PROXY_URL).toBe("http://127.0.0.1:8787/v1");
      expect(options.env.NODE_OPTIONS).toContain("--trace-warnings");
      expect(options.env.NODE_OPTIONS).toContain("--import=file:");
      expect(options.env.NODE_OPTIONS).toContain("/hook-shim/handler.js");
    } finally {
      uninstallHeadroomTransport();
      childProcess.spawn = originalSpawn;
    }
  });

  it("strips remote policy credentials and URLs from wrapped child environments", () => {
    const originalSpawn = childProcess.spawn;
    const previousToken = process.env.HEADROOM_TOOL_POLICY_TOKEN;
    const previousUrl = process.env.HEADROOM_TOOL_POLICY_URL;
    const spawnMock = vi.fn(() => ({
      on: vi.fn(),
      once: vi.fn(),
      emit: vi.fn(),
      kill: vi.fn(),
      killed: false,
      pid: 123,
    }));
    childProcess.spawn = spawnMock as unknown as typeof childProcess.spawn;
    process.env.HEADROOM_TOOL_POLICY_TOKEN = "private-policy-token";
    process.env.HEADROOM_TOOL_POLICY_URL =
      "https://policy.example.test/v1?signature=private-signature";

    try {
      installHeadroomTransport({
        proxyUrl: "http://127.0.0.1:8787/v1",
        toolPolicy: { rules: [] },
      });

      childProcess.spawn("node", ["agent.js"]);
      childProcess.spawn("node", ["agent.js"], {
        env: {
          PATH: "/bin",
          HEADROOM_TOOL_POLICY_TOKEN: "custom-private-token",
          HEADROOM_TOOL_POLICY_URL: "https://custom.example.test/?token=secret",
        },
      });

      for (const call of spawnMock.mock.calls) {
        const options = (call as unknown[])[2] as { env: NodeJS.ProcessEnv };
        expect(options.env).not.toHaveProperty("HEADROOM_TOOL_POLICY_TOKEN");
        expect(options.env).not.toHaveProperty("HEADROOM_TOOL_POLICY_URL");
      }
    } finally {
      uninstallHeadroomTransport();
      childProcess.spawn = originalSpawn;
      if (previousToken === undefined) delete process.env.HEADROOM_TOOL_POLICY_TOKEN;
      else process.env.HEADROOM_TOOL_POLICY_TOKEN = previousToken;
      if (previousUrl === undefined) delete process.env.HEADROOM_TOOL_POLICY_URL;
      else process.env.HEADROOM_TOOL_POLICY_URL = previousUrl;
    }
  });

  it("sends x-headroom-project header on routed fetch calls when project is set", async () => {
    const originalFetch = globalThis.fetch;
    const fetchMock = vi.fn(async (..._args: FetchCall) => new Response("ok"));
    globalThis.fetch = fetchMock as unknown as typeof fetch;

    installHeadroomTransport({ proxyUrl: "http://127.0.0.1:8787/v1", project: "my-project" });

    await fetch("https://api.anthropic.com/v1/messages", { method: "POST" });

    const headers = new Headers(fetchMock.mock.calls[0][1]?.headers);
    expect(headers.get("x-headroom-project")).toBe("my-project");

    globalThis.fetch = originalFetch;
  });

  it("percent-encodes a non-Latin-1 project name instead of throwing on routed fetch calls", async () => {
    const originalFetch = globalThis.fetch;
    const fetchMock = vi.fn(async (..._args: FetchCall) => new Response("ok"));
    globalThis.fetch = fetchMock as unknown as typeof fetch;

    // Thai, Chinese, Cyrillic and a Polish letter above U+00FF, plus a path separator.
    const project = "/home/u/โลลิ/中文/Привет/ł";
    installHeadroomTransport({ proxyUrl: "http://127.0.0.1:8787/v1", project });

    await fetch("https://api.anthropic.com/v1/messages", { method: "POST" });

    expect(fetchMock).toHaveBeenCalledTimes(1);
    const value = new Headers(fetchMock.mock.calls[0][1]?.headers).get("x-headroom-project");
    expect(value).toBe(encodeURIComponent(project));
    expect(value).toMatch(/^[\x20-\x7e]*$/);
    expect(decodeURIComponent(value ?? "")).toBe(project);

    globalThis.fetch = originalFetch;
  });

  it("percent-encodes a non-Latin-1 project name on routed Node https.request calls", async () => {
    const proxy = await proxyServer();
    const project = "โลลิปืนดุ";
    installHeadroomTransport({ proxyUrl: proxy.url, project });

    await new Promise<void>((resolve, reject) => {
      const req = https.request(
        "https://api.anthropic.com/v1/messages",
        { method: "POST" },
        (res) => {
          res.resume();
          res.on("end", resolve);
        },
      );
      req.on("error", reject);
      req.end("{}");
    });

    const value = proxy.seen[0].headers["x-headroom-project"] as string;
    expect(value).toBe(encodeURIComponent(project));
    expect(decodeURIComponent(value)).toBe(project);

    await proxy.close();
  });

  it("omits x-headroom-project header when project is not set", async () => {
    const originalFetch = globalThis.fetch;
    const fetchMock = vi.fn(async (..._args: FetchCall) => new Response("ok"));
    globalThis.fetch = fetchMock as unknown as typeof fetch;

    installHeadroomTransport({ proxyUrl: "http://127.0.0.1:8787/v1" });

    await fetch("https://api.anthropic.com/v1/messages", { method: "POST" });

    const headers = new Headers(fetchMock.mock.calls[0][1]?.headers);
    expect(headers.get("x-headroom-project")).toBeNull();

    globalThis.fetch = originalFetch;
  });

  it("denies fetch calls that match an http policy rule before routing", async () => {
    const originalFetch = globalThis.fetch;
    const fetchMock = vi.fn(async (..._args: FetchCall) => new Response("ok"));
    globalThis.fetch = fetchMock as unknown as typeof fetch;
    try {
      installHeadroomTransport({
        proxyUrl: "http://127.0.0.1:8787/v1",
        toolPolicy: {
          rules: [
            {
              id: "deny-openai",
              scope: "http",
              action: "deny",
              domain: "api.openai.com",
              reason: "direct egress not approved",
            },
          ],
        },
      });

      await expect(fetch("https://api.openai.com/v1/responses", { method: "POST" })).rejects.toThrow(
        /rule=deny-openai/,
      );
      expect(fetchMock).not.toHaveBeenCalled();
    } finally {
      uninstallHeadroomTransport();
      globalThis.fetch = originalFetch;
    }
  });

  it("surfaces require_approval shell decisions as a hard block", () => {
    const originalSpawn = childProcess.spawn;
    const spawnMock = vi.fn(() => ({ on: vi.fn(), kill: vi.fn(), pid: 123 }));
    childProcess.spawn = spawnMock as unknown as typeof childProcess.spawn;

    try {
      installHeadroomTransport({
        proxyUrl: "http://127.0.0.1:8787/v1",
        toolPolicy: {
          rules: [
            {
              id: "approve-curl",
              scope: "shell",
              action: "require_approval",
              command: "curl",
            },
          ],
        },
      });

      expect(() => childProcess.spawn("curl", ["https://example.com"])).toThrow(
        /requires approval/,
      );
      expect(spawnMock).not.toHaveBeenCalled();
    } finally {
      uninstallHeadroomTransport();
      childProcess.spawn = originalSpawn;
    }
  });

  it("matches child-process rules against Node's effective cwd", () => {
    const nested = path.resolve("nested");
    const cwdPattern = `^${nested.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")}$`;
    try {
      installHeadroomTransport({
        proxyUrl: "http://127.0.0.1:8787/v1",
        toolPolicy: {
          rules: [
            {
              id: "deny-relative",
              scope: "shell",
              action: "deny",
              command: "node",
              cwdPattern,
            },
            {
              id: "deny-default",
              scope: "shell",
              action: "deny",
              command: "python",
              cwdPattern: `^${process.cwd().replace(/[.*+?^${}()|[\]\\]/g, "\\$&")}$`,
            },
            {
              id: "deny-file-url",
              scope: "shell",
              action: "deny",
              command: "deno",
              cwdPattern,
            },
          ],
        },
      });

      expect(() => childProcess.spawn("node", ["agent.js"], { cwd: "nested" })).toThrow(
        /deny-relative/,
      );
      expect(() => childProcess.exec("python agent.py")).toThrow(/deny-default/);
      expect(() =>
        childProcess.execFile("deno", ["run", "agent.ts"], { cwd: pathToFileURL(nested) }),
      ).toThrow(/deny-file-url/);
    } finally {
      uninstallHeadroomTransport();
    }
  });

  it("treats direct child-process executable paths containing spaces atomically", () => {
    const executable = path.join(process.cwd(), "Program Files", "curl.exe");
    const modulePath = path.join(process.cwd(), "Program Files", "worker file.js");
    const originalSpawn = childProcess.spawn;
    const originalExecFile = childProcess.execFile;
    const originalFork = childProcess.fork;
    const spawnMock = vi.fn(() => ({ on: vi.fn(), kill: vi.fn(), pid: 123 }));
    const execFileMock = vi.fn(() => ({ on: vi.fn(), kill: vi.fn(), pid: 124 }));
    const forkMock = vi.fn(() => ({ on: vi.fn(), kill: vi.fn(), pid: 125 }));
    childProcess.spawn = spawnMock as unknown as typeof childProcess.spawn;
    childProcess.execFile = execFileMock as unknown as typeof childProcess.execFile;
    childProcess.fork = forkMock as unknown as typeof childProcess.fork;

    try {
      installHeadroomTransport({
        proxyUrl: "http://127.0.0.1:8787/v1",
        toolPolicy: {
          rules: [
            {
              id: "deny-direct-path",
              scope: "shell",
              action: "deny",
              command: executable,
            },
            {
              id: "deny-fork-module",
              scope: "shell",
              action: "deny",
              command: path.basename(modulePath),
            },
          ],
        },
      });

      expect(() => childProcess.spawn(executable, ["https://example.test"])).toThrow(
        /deny-direct-path/,
      );
      expect(() => childProcess.execFile(executable, ["https://example.test"])).toThrow(
        /deny-direct-path/,
      );
      expect(() => childProcess.fork(modulePath)).toThrow(/deny-fork-module/);
      expect(spawnMock).not.toHaveBeenCalled();
      expect(execFileMock).not.toHaveBeenCalled();
      expect(forkMock).not.toHaveBeenCalled();
    } finally {
      uninstallHeadroomTransport();
      childProcess.spawn = originalSpawn;
      childProcess.execFile = originalExecFile;
      childProcess.fork = originalFork;
    }
  });

  it("logs report-only policy decisions but still allows the request", async () => {
    const originalFetch = globalThis.fetch;
    const fetchMock = vi.fn(async (..._args: FetchCall) => new Response("ok"));
    globalThis.fetch = fetchMock as unknown as typeof fetch;
    const stderrSpy = vi.spyOn(process.stderr, "write").mockReturnValue(true);
    try {
      installHeadroomTransport({
        proxyUrl: "http://127.0.0.1:8787/v1",
        toolPolicy: {
          mode: "report_only",
          rules: [
            {
              id: "report-http",
              scope: "http",
              action: "deny",
              domain: "*.example.com",
              reason: "dry run",
            },
          ],
        },
      });

      await fetch("https://api.example.com/v1/messages", { method: "POST" });

      expect(fetchMock).toHaveBeenCalledTimes(1);
      expect(stderrSpy).toHaveBeenCalledWith(
        expect.stringContaining('"effective_action":"allow"'),
      );
      expect(stderrSpy).toHaveBeenCalledWith(expect.stringContaining('"matched_rule":"report-http"'));
    } finally {
      uninstallHeadroomTransport();
      stderrSpy.mockRestore();
      globalThis.fetch = originalFetch;
    }
  });

  it("logs only safe resource summaries while hashing the full resource", () => {
    const root = fs.mkdtempSync(path.join(os.tmpdir(), "headroom-policy-safe-audit-"));
    const previousWorkspace = process.env.HEADROOM_WORKSPACE_DIR;
    const stderrSpy = vi.spyOn(process.stderr, "write").mockReturnValue(true);
    process.env.HEADROOM_WORKSPACE_DIR = root;
    try {
      installHeadroomTransport({
        proxyUrl: "http://127.0.0.1:8787",
        toolPolicy: {
          rules: [{ id: "deny-curl", scope: "shell", action: "deny", command: "curl" }],
        },
      });

      expect(() =>
        childProcess.spawn("curl", ["https://example.test/?token=super-secret"]),
      ).toThrow(/target curl/);

      const stderr = stderrSpy.mock.calls.map(([value]) => String(value)).join("");
      const audit = fs.readFileSync(path.join(root, "tool_policy_audit.jsonl"), "utf8");
      for (const output of [stderr, audit]) {
        expect(output).toContain('"resource":"curl"');
        expect(output).not.toContain("super-secret");
        expect(output).not.toContain("https://example.test");
        expect(output).toMatch(/"request_hash":"[a-f0-9]{16}"/);
      }
    } finally {
      uninstallHeadroomTransport();
      stderrSpy.mockRestore();
      if (previousWorkspace === undefined) delete process.env.HEADROOM_WORKSPACE_DIR;
      else process.env.HEADROOM_WORKSPACE_DIR = previousWorkspace;
      fs.rmSync(root, { recursive: true, force: true });
    }
  });

  it("loads policy from the nearest repo-local .headroom/tool_policy.json", () => {
    const originalSpawn = childProcess.spawn;
    const spawnMock = vi.fn(() => ({ on: vi.fn(), kill: vi.fn(), pid: 123 }));
    childProcess.spawn = spawnMock as unknown as typeof childProcess.spawn;
    const tmpRoot = fs.mkdtempSync(path.join(os.tmpdir(), "headroom-policy-local-"));
    const previousConfigDir = process.env.HEADROOM_CONFIG_DIR;
    process.env.HEADROOM_CONFIG_DIR = path.join(tmpRoot, "empty-global-config");
    const repoRoot = path.join(tmpRoot, "repo");
    const nested = path.join(repoRoot, "src", "app");
    fs.mkdirSync(path.join(repoRoot, ".headroom"), { recursive: true });
    fs.mkdirSync(nested, { recursive: true });
    fs.writeFileSync(
      path.join(repoRoot, ".headroom", "tool_policy.json"),
      JSON.stringify({
        rules: [{ id: "deny-curl", scope: "shell", action: "deny", command: "curl" }],
      }),
    );
    try {
      installHeadroomTransport({ proxyUrl: "http://127.0.0.1:8787/v1", project: nested });
      expect(() => childProcess.spawn("curl", ["https://example.com"])).toThrow(/deny-curl/);
      expect(spawnMock).not.toHaveBeenCalled();
    } finally {
      if (previousConfigDir === undefined) delete process.env.HEADROOM_CONFIG_DIR;
      else process.env.HEADROOM_CONFIG_DIR = previousConfigDir;
      uninstallHeadroomTransport();
      childProcess.spawn = originalSpawn;
      fs.rmSync(tmpRoot, { recursive: true, force: true });
    }
  });

  it("loads policy from HEADROOM_TOOL_POLICY_PATH", () => {
    const originalSpawn = childProcess.spawn;
    const spawnMock = vi.fn(() => ({ on: vi.fn(), kill: vi.fn(), pid: 123 }));
    const tmpRoot = fs.mkdtempSync(path.join(os.tmpdir(), "headroom-policy-env-"));
    const policyPath = path.join(tmpRoot, "tool-policy.json");
    fs.writeFileSync(
      policyPath,
      JSON.stringify({
        rules: [{ id: "deny-node", scope: "shell", action: "deny", command: "node" }],
      }),
    );
    childProcess.spawn = spawnMock as unknown as typeof childProcess.spawn;
    process.env.HEADROOM_TOOL_POLICY_PATH = policyPath;
    try {
      installHeadroomTransport({ proxyUrl: "http://127.0.0.1:8787/v1" });
      expect(() => childProcess.spawn("node", ["script.js"])).toThrow(/deny-node/);
      expect(spawnMock).not.toHaveBeenCalled();
    } finally {
      uninstallHeadroomTransport();
      delete process.env.HEADROOM_TOOL_POLICY_PATH;
      childProcess.spawn = originalSpawn;
      fs.rmSync(tmpRoot, { recursive: true, force: true });
    }
  });

  it("selects a higher-precedence local source once and never refreshes a remote URL", async () => {
    const originalFetch = globalThis.fetch;
    const previousJson = process.env.HEADROOM_TOOL_POLICY_JSON;
    const previousUrl = process.env.HEADROOM_TOOL_POLICY_URL;
    const fetchMock = vi.fn<typeof fetch>();
    globalThis.fetch = fetchMock;
    process.env.HEADROOM_TOOL_POLICY_JSON = JSON.stringify({
      rules: [{ id: "json-deny", scope: "shell", action: "deny", command: "curl" }],
    });
    process.env.HEADROOM_TOOL_POLICY_URL = "https://policy.example.test/ignored";
    try {
      installHeadroomTransport({ proxyUrl: "http://127.0.0.1:8787" });
      await refreshHeadroomToolPolicy();
      expect(() => childProcess.spawn("curl", ["example.test"])).toThrow(/json-deny/);
      expect(fetchMock).not.toHaveBeenCalled();
    } finally {
      uninstallHeadroomTransport();
      globalThis.fetch = originalFetch;
      if (previousJson === undefined) delete process.env.HEADROOM_TOOL_POLICY_JSON;
      else process.env.HEADROOM_TOOL_POLICY_JSON = previousJson;
      if (previousUrl === undefined) delete process.env.HEADROOM_TOOL_POLICY_URL;
      else process.env.HEADROOM_TOOL_POLICY_URL = previousUrl;
    }
  });

  it("gives the machine-global policy precedence and persists audit decisions", () => {
    const root = fs.mkdtempSync(path.join(os.tmpdir(), "headroom-policy-precedence-"));
    const repo = path.join(root, "repo");
    fs.mkdirSync(path.join(root, "config"), { recursive: true });
    fs.mkdirSync(path.join(repo, ".headroom"), { recursive: true });
    fs.writeFileSync(
      path.join(root, "config", "tool_policy.json"),
      JSON.stringify({
        rules: [{ id: "global-deny", scope: "shell", action: "deny", command: "curl" }],
      }),
    );
    fs.writeFileSync(
      path.join(repo, ".headroom", "tool_policy.json"),
      JSON.stringify({ rules: [] }),
    );
    const originalWorkspace = process.env.HEADROOM_WORKSPACE_DIR;
    process.env.HEADROOM_WORKSPACE_DIR = root;
    try {
      installHeadroomTransport({ proxyUrl: "http://127.0.0.1:8787", project: repo });
      expect(() => childProcess.spawn("curl", ["example.test"])).toThrow(/global-deny/);
      const audit = fs.readFileSync(path.join(root, "tool_policy_audit.jsonl"), "utf8");
      expect(audit).toContain('"matched_rule":"global-deny"');
    } finally {
      uninstallHeadroomTransport();
      if (originalWorkspace === undefined) delete process.env.HEADROOM_WORKSPACE_DIR;
      else process.env.HEADROOM_WORKSPACE_DIR = originalWorkspace;
      fs.rmSync(root, { recursive: true, force: true });
    }
  });

  it("uses remote auth, cache, ETag revalidation, and fails closed after an expired outage", async () => {
    const root = fs.mkdtempSync(path.join(os.tmpdir(), "headroom-policy-remote-"));
    const url = "https://policy.example/tool-policy";
    const originalFetch = globalThis.fetch;
    const originalEnv = {
      workspace: process.env.HEADROOM_WORKSPACE_DIR,
      url: process.env.HEADROOM_TOOL_POLICY_URL,
      token: process.env.HEADROOM_TOOL_POLICY_TOKEN,
      refresh: process.env.HEADROOM_TOOL_POLICY_REFRESH_SECONDS,
    };
    const fetchMock = vi
      .fn<typeof fetch>()
      .mockResolvedValueOnce(
        new Response(
          JSON.stringify({
            version: 1,
            rules: [{ id: "remote-deny", scope: "shell", action: "deny", command: "curl" }],
          }),
          { status: 200, headers: { etag: '"v1"' } },
        ),
      )
      .mockResolvedValueOnce(new Response(null, { status: 304 }))
      .mockRejectedValueOnce(new Error("offline"));
    globalThis.fetch = fetchMock;
    process.env.HEADROOM_WORKSPACE_DIR = root;
    process.env.HEADROOM_TOOL_POLICY_URL = url;
    process.env.HEADROOM_TOOL_POLICY_TOKEN = "secret-token";
    process.env.HEADROOM_TOOL_POLICY_REFRESH_SECONDS = "300";
    try {
      const initialCachePath = remoteToolPolicyCachePath(url, "secret-token");
      fs.mkdirSync(path.dirname(initialCachePath), { recursive: true });
      fs.writeFileSync(initialCachePath, "{corrupt");
      installHeadroomTransport({ proxyUrl: "http://127.0.0.1:8787" });
      await refreshHeadroomToolPolicy(1_000);
      expect(fetchMock.mock.calls[0][1]?.redirect).toBe("manual");
      expect(new Headers(fetchMock.mock.calls[0][1]?.headers).get("authorization")).toBe(
        "Bearer secret-token",
      );
      expect(fs.existsSync(initialCachePath)).toBe(true);
      expect(JSON.parse(fs.readFileSync(initialCachePath, "utf8")).cache_version).toBe(2);
      expect(fs.readdirSync(path.dirname(initialCachePath)).some((name) => name.endsWith(".tmp"))).toBe(
        false,
      );
      await refreshHeadroomToolPolicy(1_100);
      expect(fetchMock).toHaveBeenCalledTimes(1);
      const cachePath = remoteToolPolicyCachePath(url, "secret-token");
      const futureCache = JSON.parse(fs.readFileSync(cachePath, "utf8")) as {
        fetched_at: number;
      };
      futureCache.fetched_at = 9_999;
      fs.writeFileSync(cachePath, JSON.stringify(futureCache));
      await refreshHeadroomToolPolicy(1_200);
      expect(new Headers(fetchMock.mock.calls[1][1]?.headers).get("if-none-match")).toBe('"v1"');
      await refreshHeadroomToolPolicy(1_501);
      await expect(
        import("./transport.js").then(({ enforceNativeToolExecution }) =>
          enforceNativeToolExecution("bash", { command: "echo allowed" }),
        ),
      ).rejects.toThrow(/failing closed/);
    } finally {
      uninstallHeadroomTransport();
      globalThis.fetch = originalFetch;
      const restore = (name: string, value: string | undefined) => {
        if (value === undefined) delete process.env[name];
        else process.env[name] = value;
      };
      restore("HEADROOM_WORKSPACE_DIR", originalEnv.workspace);
      restore("HEADROOM_TOOL_POLICY_URL", originalEnv.url);
      restore("HEADROOM_TOOL_POLICY_TOKEN", originalEnv.token);
      restore("HEADROOM_TOOL_POLICY_REFRESH_SECONDS", originalEnv.refresh);
      fs.rmSync(root, { recursive: true, force: true });
    }
  });

  it("shares loaded state between compatible installs using the same remote context", async () => {
    const root = fs.mkdtempSync(path.join(os.tmpdir(), "headroom-policy-shared-"));
    const originalFetch = globalThis.fetch;
    const previousWorkspace = process.env.HEADROOM_WORKSPACE_DIR;
    const previousUrl = process.env.HEADROOM_TOOL_POLICY_URL;
    const previousToken = process.env.HEADROOM_TOOL_POLICY_TOKEN;
    const fetchMock = vi.fn<typeof fetch>().mockResolvedValue(
      new Response(
        JSON.stringify({
          version: 1,
          rules: [{ id: "shared-deny", scope: "shell", action: "deny", command: "curl" }],
        }),
      ),
    );
    globalThis.fetch = fetchMock;
    process.env.HEADROOM_WORKSPACE_DIR = root;
    process.env.HEADROOM_TOOL_POLICY_URL = "https://policy.example.test/shared";
    process.env.HEADROOM_TOOL_POLICY_TOKEN = "shared-token";
    let firstDispose: (() => void) | undefined;
    let secondDispose: (() => void) | undefined;
    try {
      firstDispose = installHeadroomTransport({ proxyUrl: "http://127.0.0.1:8787" });
      await refreshHeadroomToolPolicy();
      expect(() => {
        secondDispose = installHeadroomTransport({ proxyUrl: "http://127.0.0.1:8787" });
      }).not.toThrow();
      expect(() => childProcess.spawn("curl", ["example.test"])).toThrow(/shared-deny/);
      expect(fetchMock).toHaveBeenCalledTimes(1);
    } finally {
      secondDispose?.();
      firstDispose?.();
      globalThis.fetch = originalFetch;
      if (previousWorkspace === undefined) delete process.env.HEADROOM_WORKSPACE_DIR;
      else process.env.HEADROOM_WORKSPACE_DIR = previousWorkspace;
      if (previousUrl === undefined) delete process.env.HEADROOM_TOOL_POLICY_URL;
      else process.env.HEADROOM_TOOL_POLICY_URL = previousUrl;
      if (previousToken === undefined) delete process.env.HEADROOM_TOOL_POLICY_TOKEN;
      else process.env.HEADROOM_TOOL_POLICY_TOKEN = previousToken;
      fs.rmSync(root, { recursive: true, force: true });
    }
  });

  it("does not renew an invalid cached policy after a 304 response", async () => {
    const root = fs.mkdtempSync(path.join(os.tmpdir(), "headroom-policy-invalid-304-"));
    const url = "https://policy.example.test/invalid-cache";
    const token = "cache-token";
    const originalFetch = globalThis.fetch;
    const previousWorkspace = process.env.HEADROOM_WORKSPACE_DIR;
    const previousUrl = process.env.HEADROOM_TOOL_POLICY_URL;
    const previousToken = process.env.HEADROOM_TOOL_POLICY_TOKEN;
    const previousRefresh = process.env.HEADROOM_TOOL_POLICY_REFRESH_SECONDS;
    process.env.HEADROOM_WORKSPACE_DIR = root;
    process.env.HEADROOM_TOOL_POLICY_URL = url;
    process.env.HEADROOM_TOOL_POLICY_TOKEN = token;
    process.env.HEADROOM_TOOL_POLICY_REFRESH_SECONDS = "300";
    const cachePath = remoteToolPolicyCachePath(url, token);
    fs.mkdirSync(path.dirname(cachePath), { recursive: true });
    fs.writeFileSync(
      cachePath,
      JSON.stringify({
        cache_version: 2,
        url_hash: createHash("sha256").update(url).digest("hex"),
        etag: '"bad"',
        fetched_at: 100,
        policy: { version: 1, rules: "invalid" },
      }),
    );
    globalThis.fetch = vi.fn(async () => new Response(null, { status: 304 }));
    try {
      installHeadroomTransport({ proxyUrl: "http://127.0.0.1:8787" });
      await refreshHeadroomToolPolicy(1_000);
      expect(JSON.parse(fs.readFileSync(cachePath, "utf8")).fetched_at).toBe(100);
      await expect(
        enforceNativeToolExecution("bash", { command: "echo blocked" }),
      ).rejects.toThrow(/requires a rules array/);
    } finally {
      uninstallHeadroomTransport();
      globalThis.fetch = originalFetch;
      if (previousWorkspace === undefined) delete process.env.HEADROOM_WORKSPACE_DIR;
      else process.env.HEADROOM_WORKSPACE_DIR = previousWorkspace;
      if (previousUrl === undefined) delete process.env.HEADROOM_TOOL_POLICY_URL;
      else process.env.HEADROOM_TOOL_POLICY_URL = previousUrl;
      if (previousToken === undefined) delete process.env.HEADROOM_TOOL_POLICY_TOKEN;
      else process.env.HEADROOM_TOOL_POLICY_TOKEN = previousToken;
      if (previousRefresh === undefined) delete process.env.HEADROOM_TOOL_POLICY_REFRESH_SECONDS;
      else process.env.HEADROOM_TOOL_POLICY_REFRESH_SECONDS = previousRefresh;
      fs.rmSync(root, { recursive: true, force: true });
    }
  });

  it("propagates policy expiry without remote credentials and child shims fail closed", async () => {
    const originalSpawn = childProcess.spawn;
    const previousJson = process.env.HEADROOM_TOOL_POLICY_JSON;
    const previousExpiry = process.env.HEADROOM_INTERNAL_TOOL_POLICY_VALID_UNTIL;
    const previousUrl = process.env.HEADROOM_TOOL_POLICY_URL;
    const previousToken = process.env.HEADROOM_TOOL_POLICY_TOKEN;
    const spawnMock = vi.fn(() => ({ on: vi.fn(), kill: vi.fn(), pid: 123 }));
    childProcess.spawn = spawnMock as unknown as typeof childProcess.spawn;
    process.env.HEADROOM_TOOL_POLICY_JSON = JSON.stringify({ version: 1, rules: [] });
    process.env.HEADROOM_INTERNAL_TOOL_POLICY_VALID_UNTIL = String(Date.now() / 1000 + 60);
    process.env.HEADROOM_TOOL_POLICY_URL = "https://policy.example.test/?signature=secret";
    process.env.HEADROOM_TOOL_POLICY_TOKEN = "private-token";
    try {
      const disposeParent = installHeadroomTransport({
        proxyUrl: "http://127.0.0.1:8787",
      });

      childProcess.spawn("node", ["agent.js"]);
      const childEnv = (spawnMock.mock.calls[0] as unknown[])[2] as {
        env: NodeJS.ProcessEnv;
      };
      expect(childEnv.env.HEADROOM_INTERNAL_TOOL_POLICY_VALID_UNTIL).toBeDefined();
      expect(childEnv.env).not.toHaveProperty("HEADROOM_TOOL_POLICY_URL");
      expect(childEnv.env).not.toHaveProperty("HEADROOM_TOOL_POLICY_TOKEN");
      disposeParent();

      process.env.HEADROOM_TOOL_POLICY_JSON = childEnv.env.HEADROOM_TOOL_POLICY_JSON;
      process.env.HEADROOM_INTERNAL_TOOL_POLICY_VALID_UNTIL = String(Date.now() / 1000 - 1);
      delete process.env.HEADROOM_TOOL_POLICY_URL;
      delete process.env.HEADROOM_TOOL_POLICY_TOKEN;
      installHeadroomTransport({ proxyUrl: "http://127.0.0.1:8787" });
      expect(() => childProcess.spawn("node", ["agent.js"])).toThrow(/policy expired/);
      expect(spawnMock).toHaveBeenCalledTimes(1);
    } finally {
      uninstallHeadroomTransport();
      childProcess.spawn = originalSpawn;
      if (previousJson === undefined) delete process.env.HEADROOM_TOOL_POLICY_JSON;
      else process.env.HEADROOM_TOOL_POLICY_JSON = previousJson;
      if (previousExpiry === undefined) {
        delete process.env.HEADROOM_INTERNAL_TOOL_POLICY_VALID_UNTIL;
      } else {
        process.env.HEADROOM_INTERNAL_TOOL_POLICY_VALID_UNTIL = previousExpiry;
      }
      if (previousUrl === undefined) delete process.env.HEADROOM_TOOL_POLICY_URL;
      else process.env.HEADROOM_TOOL_POLICY_URL = previousUrl;
      if (previousToken === undefined) delete process.env.HEADROOM_TOOL_POLICY_TOKEN;
      else process.env.HEADROOM_TOOL_POLICY_TOKEN = previousToken;
    }
  });

  it("clears startup repository policy for a policy-free session repository", async () => {
    const root = fs.mkdtempSync(path.join(os.tmpdir(), "headroom-policy-transition-"));
    const startupRepo = path.join(root, "startup");
    const sessionRepo = path.join(root, "session");
    const previousConfigDir = process.env.HEADROOM_CONFIG_DIR;
    fs.mkdirSync(path.join(startupRepo, ".headroom"), { recursive: true });
    fs.mkdirSync(sessionRepo);
    process.env.HEADROOM_CONFIG_DIR = path.join(root, "empty-global-config");
    fs.writeFileSync(
      path.join(startupRepo, ".headroom", "tool_policy.json"),
      JSON.stringify({
        rules: [
          {
            id: "deny-startup-curl",
            scope: "shell",
            action: "deny",
            command: "curl",
          },
        ],
      }),
    );

    try {
      installHeadroomTransport({
        proxyUrl: "http://127.0.0.1:8787",
        policyProject: startupRepo,
      });
      await expect(
        enforceNativeToolExecution(
          "shell",
          { command: "curl example.test" },
          startupRepo,
          { sessionID: "startup", callID: "startup-call" },
        ),
      ).rejects.toThrow(/deny-startup-curl/);

      activateHeadroomRepositoryPolicy(sessionRepo);

      await expect(
        enforceNativeToolExecution(
          "shell",
          { command: "curl example.test" },
          sessionRepo,
          { sessionID: "session", callID: "session-call" },
        ),
      ).resolves.toBeUndefined();
      expect(process.env.HEADROOM_TOOL_POLICY_JSON).toBeUndefined();
    } finally {
      uninstallHeadroomTransport();
      if (previousConfigDir === undefined) delete process.env.HEADROOM_CONFIG_DIR;
      else process.env.HEADROOM_CONFIG_DIR = previousConfigDir;
      fs.rmSync(root, { recursive: true, force: true });
    }
  });

  it("rejects remote policy responses larger than 1 MiB", async () => {
    const root = fs.mkdtempSync(path.join(os.tmpdir(), "headroom-policy-oversize-"));
    const originalFetch = globalThis.fetch;
    const previousWorkspace = process.env.HEADROOM_WORKSPACE_DIR;
    const previousUrl = process.env.HEADROOM_TOOL_POLICY_URL;
    globalThis.fetch = vi.fn(async () => new Response("x".repeat(1024 * 1024 + 1)));
    process.env.HEADROOM_WORKSPACE_DIR = root;
    process.env.HEADROOM_TOOL_POLICY_URL = "http://127.0.0.1/policy";
    try {
      installHeadroomTransport({ proxyUrl: "http://127.0.0.1:8787" });
      await refreshHeadroomToolPolicy();
      await expect(
        import("./transport.js").then(({ enforceNativeToolExecution }) =>
          enforceNativeToolExecution("read_file", { path: "secret.txt" }),
        ),
      ).rejects.toThrow(/exceeds 1 MiB/);
    } finally {
      uninstallHeadroomTransport();
      globalThis.fetch = originalFetch;
      if (previousWorkspace === undefined) delete process.env.HEADROOM_WORKSPACE_DIR;
      else process.env.HEADROOM_WORKSPACE_DIR = previousWorkspace;
      if (previousUrl === undefined) delete process.env.HEADROOM_TOOL_POLICY_URL;
      else process.env.HEADROOM_TOOL_POLICY_URL = previousUrl;
      fs.rmSync(root, { recursive: true, force: true });
    }
  });

  it("isolates remote policy caches by bearer token", async () => {
    const root = fs.mkdtempSync(path.join(os.tmpdir(), "headroom-policy-token-cache-"));
    const url = "https://policy.example/tool-policy";
    const originalFetch = globalThis.fetch;
    const previousWorkspace = process.env.HEADROOM_WORKSPACE_DIR;
    const previousUrl = process.env.HEADROOM_TOOL_POLICY_URL;
    const previousToken = process.env.HEADROOM_TOOL_POLICY_TOKEN;
    const fetchMock = vi
      .fn<typeof fetch>()
      .mockResolvedValueOnce(new Response(JSON.stringify({ version: 1, rules: [] })))
      .mockResolvedValueOnce(new Response(JSON.stringify({ version: 1, rules: [] })));
    globalThis.fetch = fetchMock;
    process.env.HEADROOM_WORKSPACE_DIR = root;
    process.env.HEADROOM_TOOL_POLICY_URL = url;
    process.env.HEADROOM_TOOL_POLICY_TOKEN = "tenant-alpha";
    let dispose: (() => void) | undefined;
    try {
      dispose = installHeadroomTransport({ proxyUrl: "http://127.0.0.1:8787" });
      await refreshHeadroomToolPolicy(1_000);
      dispose();
      dispose = undefined;
      process.env.HEADROOM_TOOL_POLICY_TOKEN = "tenant-bravo";
      dispose = installHeadroomTransport({ proxyUrl: "http://127.0.0.1:8787" });
      await refreshHeadroomToolPolicy(1_001);

      const alphaCache = remoteToolPolicyCachePath(url, "tenant-alpha");
      const bravoCache = remoteToolPolicyCachePath(url, "tenant-bravo");
      expect(alphaCache).not.toBe(bravoCache);
      expect(fs.existsSync(alphaCache)).toBe(true);
      expect(fs.existsSync(bravoCache)).toBe(true);
      expect(fetchMock).toHaveBeenCalledTimes(2);
      expect(new Headers(fetchMock.mock.calls[1][1]?.headers).has("if-none-match")).toBe(false);
      expect(fs.readFileSync(alphaCache, "utf8")).not.toContain("tenant-alpha");
      expect(fs.readFileSync(bravoCache, "utf8")).not.toContain("tenant-bravo");
    } finally {
      dispose?.();
      globalThis.fetch = originalFetch;
      if (previousWorkspace === undefined) delete process.env.HEADROOM_WORKSPACE_DIR;
      else process.env.HEADROOM_WORKSPACE_DIR = previousWorkspace;
      if (previousUrl === undefined) delete process.env.HEADROOM_TOOL_POLICY_URL;
      else process.env.HEADROOM_TOOL_POLICY_URL = previousUrl;
      if (previousToken === undefined) delete process.env.HEADROOM_TOOL_POLICY_TOKEN;
      else process.env.HEADROOM_TOOL_POLICY_TOKEN = previousToken;
      fs.rmSync(root, { recursive: true, force: true });
    }
  });

  it("sends x-headroom-project header on routed Node https.request calls when project is set", async () => {
    const proxy = await proxyServer();
    installHeadroomTransport({ proxyUrl: proxy.url, project: "my-project" });

    await new Promise<void>((resolve, reject) => {
      const req = https.request(
        "https://api.anthropic.com/v1/messages",
        { method: "POST" },
        (res) => {
          res.resume();
          res.on("end", resolve);
        },
      );
      req.on("error", reject);
      req.end("{}");
    });

    expect(proxy.seen[0].headers["x-headroom-project"]).toBe("my-project");

    await proxy.close();
  });

  it("keeps HEADROOM_OPENCODE_EXCLUDE_HOSTS hosts and their subdomains off the proxy (#3657)", async () => {
    const originalExclude = process.env.HEADROOM_OPENCODE_EXCLUDE_HOSTS;
    const originalFetch = globalThis.fetch;
    const fetchMock = vi.fn(async (..._args: FetchCall) => new Response("ok"));
    globalThis.fetch = fetchMock as unknown as typeof fetch;

    try {
      process.env.HEADROOM_OPENCODE_EXCLUDE_HOSTS = " opencode.ai, .corp.internal ,, *.Gateway.Example ";
      installHeadroomTransport({ proxyUrl: "http://127.0.0.1:8787/v1" });

      const init = { method: "POST", headers: { authorization: "Bearer test" } };
      await fetch("https://opencode.ai/zen/v1/responses", init);
      await fetch("https://API.OpenCode.AI/zen/v1/chat/completions", init);
      await fetch("https://llm.corp.internal/v1/chat/completions", init);
      await fetch("https://gateway.example/v1/responses", init);
      await fetch("https://notopencode.ai/v1/responses", init);
      await fetch("https://api.anthropic.com/v1/messages", init);

      for (const index of [0, 1, 2, 3]) {
        expect(typeof fetchMock.mock.calls[index][0]).toBe("string");
        expect(fetchMock.mock.calls[index][1]).toBe(init);
      }
      expect(fetchMock.mock.calls[0][0]).toBe("https://opencode.ai/zen/v1/responses");
      expect(fetchMock.mock.calls[4][0]).toEqual(new URL("http://127.0.0.1:8787/v1/responses"));
      expect(new Headers(fetchMock.mock.calls[4][1]?.headers).get("x-headroom-base-url")).toBe(
        "https://notopencode.ai",
      );
      expect(fetchMock.mock.calls[5][0]).toEqual(new URL("http://127.0.0.1:8787/v1/messages"));
      expect(new Headers(fetchMock.mock.calls[5][1]?.headers).get("x-headroom-base-url")).toBe(
        "https://api.anthropic.com",
      );
    } finally {
      globalThis.fetch = originalFetch;
      if (originalExclude === undefined) {
        delete process.env.HEADROOM_OPENCODE_EXCLUDE_HOSTS;
      } else {
        process.env.HEADROOM_OPENCODE_EXCLUDE_HOSTS = originalExclude;
      }
    }
  });

  it("prefers the excludeHosts option over HEADROOM_OPENCODE_EXCLUDE_HOSTS", async () => {
    const originalExclude = process.env.HEADROOM_OPENCODE_EXCLUDE_HOSTS;
    const originalFetch = globalThis.fetch;
    const fetchMock = vi.fn(async (..._args: FetchCall) => new Response("ok"));
    globalThis.fetch = fetchMock as unknown as typeof fetch;

    try {
      process.env.HEADROOM_OPENCODE_EXCLUDE_HOSTS = "api.anthropic.com";
      installHeadroomTransport({ proxyUrl: "http://127.0.0.1:8787/v1", excludeHosts: ["opencode.ai"] });

      await fetch("https://opencode.ai/zen/v1/responses", { method: "POST" });
      await fetch("https://api.anthropic.com/v1/messages", { method: "POST" });

      expect(fetchMock.mock.calls[0][0]).toBe("https://opencode.ai/zen/v1/responses");
      expect(fetchMock.mock.calls[1][0]).toEqual(new URL("http://127.0.0.1:8787/v1/messages"));
      expect(process.env.HEADROOM_OPENCODE_EXCLUDE_HOSTS).toBe("opencode.ai");
    } finally {
      uninstallHeadroomTransport();
      globalThis.fetch = originalFetch;
      if (originalExclude === undefined) {
        delete process.env.HEADROOM_OPENCODE_EXCLUDE_HOSTS;
      } else {
        process.env.HEADROOM_OPENCODE_EXCLUDE_HOSTS = originalExclude;
      }
    }
  });

  it("leaves excluded hosts alone for Node https.request and http2.connect", () => {
    const originalExclude = process.env.HEADROOM_OPENCODE_EXCLUDE_HOSTS;
    const fakeRequest = { on: vi.fn(), end: vi.fn() };
    const requestSpy = vi.spyOn(https, "request").mockImplementation(() => fakeRequest as never);
    const connectSpy = vi.spyOn(http2, "connect").mockImplementation(() => ({}) as never);

    try {
      installHeadroomTransport({ proxyUrl: "http://127.0.0.1:8787/v1", excludeHosts: ["opencode.ai"] });

      const options = { method: "POST", headers: { authorization: "Bearer test" } };
      expect(https.request("https://opencode.ai/zen/v1/responses", options)).toBe(fakeRequest);
      expect(requestSpy).toHaveBeenCalledWith("https://opencode.ai/zen/v1/responses", options);

      expect(() => http2.connect("https://opencode.ai")).not.toThrow();
      expect(connectSpy).toHaveBeenCalledWith("https://opencode.ai");
      expect(() => http2.connect("https://api.openai.com")).not.toThrow();
      expect(connectSpy).toHaveBeenCalledWith("https://api.openai.com");
    } finally {
      uninstallHeadroomTransport();
      if (originalExclude === undefined) {
        delete process.env.HEADROOM_OPENCODE_EXCLUDE_HOSTS;
      } else {
        process.env.HEADROOM_OPENCODE_EXCLUDE_HOSTS = originalExclude;
      }
    }
  });

  it("propagates excluded hosts into child process env so the hook-shim matches", () => {
    const originalExclude = process.env.HEADROOM_OPENCODE_EXCLUDE_HOSTS;
    const originalSpawn = childProcess.spawn;
    const spawnMock = vi.fn(() => ({ on: vi.fn(), kill: vi.fn(), pid: 123 }));
    childProcess.spawn = spawnMock as unknown as typeof childProcess.spawn;

    try {
      delete process.env.HEADROOM_OPENCODE_EXCLUDE_HOSTS;
      installHeadroomTransport({ proxyUrl: "http://127.0.0.1:8787/v1" });
      childProcess.spawn("node", ["agent.js"], { env: { PATH: "/bin" } });
      uninstallHeadroomTransport();

      installHeadroomTransport({ proxyUrl: "http://127.0.0.1:8787/v1", excludeHosts: ["*.opencode.ai", "corp.internal"] });
      childProcess.spawn("node", ["agent.js"], { env: { PATH: "/bin" } });

      const plain = (spawnMock.mock.calls[0] as unknown[])[2] as { env: NodeJS.ProcessEnv };
      const excluded = (spawnMock.mock.calls[1] as unknown[])[2] as { env: NodeJS.ProcessEnv };
      expect(plain.env.HEADROOM_OPENCODE_EXCLUDE_HOSTS).toBeUndefined();
      expect(excluded.env.HEADROOM_OPENCODE_EXCLUDE_HOSTS).toBe("opencode.ai,corp.internal");
      expect(excluded.env.HEADROOM_OPENCODE_TRANSPORT_PROXY_URL).toBe("http://127.0.0.1:8787/v1");
    } finally {
      uninstallHeadroomTransport();
      childProcess.spawn = originalSpawn;
      if (originalExclude === undefined) {
        delete process.env.HEADROOM_OPENCODE_EXCLUDE_HOSTS;
      } else {
        process.env.HEADROOM_OPENCODE_EXCLUDE_HOSTS = originalExclude;
      }
    }
  });

  it("clears a pre-existing HEADROOM_OPENCODE_EXCLUDE_HOSTS when excludeHosts is explicitly empty", async () => {
    const originalExclude = process.env.HEADROOM_OPENCODE_EXCLUDE_HOSTS;
    const originalFetch = globalThis.fetch;
    const originalSpawn = childProcess.spawn;
    const fetchMock = vi.fn(async (..._args: FetchCall) => new Response("ok"));
    globalThis.fetch = fetchMock as unknown as typeof fetch;
    const spawnMock = vi.fn(() => ({ on: vi.fn(), kill: vi.fn(), pid: 123 }));
    childProcess.spawn = spawnMock as unknown as typeof childProcess.spawn;

    try {
      process.env.HEADROOM_OPENCODE_EXCLUDE_HOSTS = "api.anthropic.com";
      installHeadroomTransport({ proxyUrl: "http://127.0.0.1:8787/v1", excludeHosts: [] });

      // The option wins in-process: the host is routed, not excluded.
      await fetch("https://api.anthropic.com/v1/messages", { method: "POST" });
      expect(fetchMock.mock.calls[0][0]).toEqual(new URL("http://127.0.0.1:8787/v1/messages"));

      // ...and the exported variable mirrors that, so the hook-shim in a child
      // (default env, custom env, or a custom env carrying the stale value)
      // resolves the same empty list instead of the pre-existing one.
      expect(process.env.HEADROOM_OPENCODE_EXCLUDE_HOSTS).toBeUndefined();
      childProcess.spawn("node", ["agent.js"]);
      childProcess.spawn("node", ["agent.js"], { env: { PATH: "/bin" } });
      childProcess.spawn("node", ["agent.js"], {
        env: { PATH: "/bin", HEADROOM_OPENCODE_EXCLUDE_HOSTS: "api.anthropic.com" },
      });
      for (const call of spawnMock.mock.calls as unknown[][]) {
        const options = call[2] as { env: NodeJS.ProcessEnv };
        expect(options.env.HEADROOM_OPENCODE_EXCLUDE_HOSTS).toBeUndefined();
        expect(options.env.HEADROOM_OPENCODE_TRANSPORT_PROXY_URL).toBe("http://127.0.0.1:8787/v1");
      }
    } finally {
      // Dispose first: uninstall now restores the env it saved at install time.
      uninstallHeadroomTransport();
      globalThis.fetch = originalFetch;
      childProcess.spawn = originalSpawn;
      if (originalExclude === undefined) {
        delete process.env.HEADROOM_OPENCODE_EXCLUDE_HOSTS;
      } else {
        process.env.HEADROOM_OPENCODE_EXCLUDE_HOSTS = originalExclude;
      }
    }
  });

  it("keeps the exported variable in step with each re-install's resolved list", () => {
    const originalExclude = process.env.HEADROOM_OPENCODE_EXCLUDE_HOSTS;

    try {
      delete process.env.HEADROOM_OPENCODE_EXCLUDE_HOSTS;
      const first = installHeadroomTransport({ proxyUrl: "http://127.0.0.1:8787/v1", excludeHosts: ["opencode.ai"] });
      expect(process.env.HEADROOM_OPENCODE_EXCLUDE_HOSTS).toBe("opencode.ai");

      // A second install (refs = 2) that resolves to an empty list clears it...
      const second = installHeadroomTransport({ proxyUrl: "http://127.0.0.1:8787/v1", excludeHosts: [] });
      expect(process.env.HEADROOM_OPENCODE_EXCLUDE_HOSTS).toBeUndefined();

      // ...and one that resolves to a list sets it again.
      const third = installHeadroomTransport({ proxyUrl: "http://127.0.0.1:8787/v1", excludeHosts: ["corp.internal"] });
      expect(process.env.HEADROOM_OPENCODE_EXCLUDE_HOSTS).toBe("corp.internal");

      third();
      second();
      first();
    } finally {
      uninstallHeadroomTransport();
      if (originalExclude === undefined) {
        delete process.env.HEADROOM_OPENCODE_EXCLUDE_HOSTS;
      } else {
        process.env.HEADROOM_OPENCODE_EXCLUDE_HOSTS = originalExclude;
      }
    }
  });

  it("restores patched transports only after the final disposer", () => {
    const originalFetch = globalThis.fetch;
    const originalHttpRequest = http.request;
    const originalHttpsRequest = https.request;
    const firstDispose = installHeadroomTransport({ proxyUrl: "http://127.0.0.1:8787/v1" });
    const secondDispose = installHeadroomTransport({ proxyUrl: "http://127.0.0.1:8788/v1" });

    expect(globalThis.fetch).not.toBe(originalFetch);
    expect(http.request).not.toBe(originalHttpRequest);
    expect(https.request).not.toBe(originalHttpsRequest);

    firstDispose();
    expect(globalThis.fetch).not.toBe(originalFetch);
    expect(http.request).not.toBe(originalHttpRequest);

    secondDispose();
    expect(globalThis.fetch).toBe(originalFetch);
    expect(http.request).toBe(originalHttpRequest);
    expect(https.request).toBe(originalHttpsRequest);
  });

  it("sends x-headroom-session-token header on routed fetch calls when sessionToken is set", async () => {
    const originalFetch = globalThis.fetch;
    const fetchMock = vi.fn(async (..._args: FetchCall) => new Response("ok"));
    globalThis.fetch = fetchMock as unknown as typeof fetch;

    installHeadroomTransport({ proxyUrl: "http://127.0.0.1:8787/v1", sessionToken: "tok-abc" });

    await fetch("https://api.anthropic.com/v1/messages", { method: "POST" });

    const headers = new Headers(fetchMock.mock.calls[0][1]?.headers);
    expect(headers.get("x-headroom-session-token")).toBe("tok-abc");

    globalThis.fetch = originalFetch;
  });

  it("omits x-headroom-session-token header when sessionToken is not set", async () => {
    const originalFetch = globalThis.fetch;
    const fetchMock = vi.fn(async (..._args: FetchCall) => new Response("ok"));
    globalThis.fetch = fetchMock as unknown as typeof fetch;

    installHeadroomTransport({ proxyUrl: "http://127.0.0.1:8787/v1" });

    await fetch("https://api.anthropic.com/v1/messages", { method: "POST" });

    const headers = new Headers(fetchMock.mock.calls[0][1]?.headers);
    expect(headers.get("x-headroom-session-token")).toBeNull();

    globalThis.fetch = originalFetch;
  });

  it("omits x-headroom-session-token header when sessionToken is an empty string", async () => {
    const originalFetch = globalThis.fetch;
    const fetchMock = vi.fn(async (..._args: FetchCall) => new Response("ok"));
    globalThis.fetch = fetchMock as unknown as typeof fetch;

    installHeadroomTransport({ proxyUrl: "http://127.0.0.1:8787/v1", sessionToken: "" });

    await fetch("https://api.anthropic.com/v1/messages", { method: "POST" });

    const headers = new Headers(fetchMock.mock.calls[0][1]?.headers);
    expect(headers.get("x-headroom-session-token")).toBeNull();

    globalThis.fetch = originalFetch;
  });

  it("sends x-headroom-session-token header on routed Node https.request calls when sessionToken is set", async () => {
    const proxy = await proxyServer();
    installHeadroomTransport({ proxyUrl: proxy.url, sessionToken: "tok-abc" });

    await new Promise<void>((resolve, reject) => {
      const req = https.request(
        "https://api.anthropic.com/v1/messages",
        { method: "POST" },
        (res) => {
          res.resume();
          res.on("end", resolve);
        },
      );
      req.on("error", reject);
      req.end("{}");
    });

    expect(proxy.seen[0].headers["x-headroom-session-token"]).toBe("tok-abc");

    await proxy.close();
  });

  it("updates the session token on reinstall rather than keeping the first value", async () => {
    const originalFetch = globalThis.fetch;
    const fetchMock = vi.fn(async (..._args: FetchCall) => new Response("ok"));
    globalThis.fetch = fetchMock as unknown as typeof fetch;

    const firstDispose = installHeadroomTransport({
      proxyUrl: "http://127.0.0.1:8787/v1",
      sessionToken: "tok-A",
    });
    installHeadroomTransport({ proxyUrl: "http://127.0.0.1:8787/v1", sessionToken: "tok-B" });

    await fetch("https://api.anthropic.com/v1/messages", { method: "POST" });

    const headers = new Headers(fetchMock.mock.calls[0][1]?.headers);
    expect(headers.get("x-headroom-session-token")).toBe("tok-B");

    firstDispose();
    globalThis.fetch = originalFetch;
  });

  it("clears the session token on reinstall when the new options omit it", async () => {
    // Deliberate: reinstall is last-write-wins for every InstallOptions
    // field (proxyUrl, project, debug already work this way) -- an install
    // without a token must not leave a stale one active.
    const originalFetch = globalThis.fetch;
    const fetchMock = vi.fn(async (..._args: FetchCall) => new Response("ok"));
    globalThis.fetch = fetchMock as unknown as typeof fetch;

    const firstDispose = installHeadroomTransport({
      proxyUrl: "http://127.0.0.1:8787/v1",
      sessionToken: "tok-A",
    });
    installHeadroomTransport({ proxyUrl: "http://127.0.0.1:8787/v1" });

    await fetch("https://api.anthropic.com/v1/messages", { method: "POST" });

    const headers = new Headers(fetchMock.mock.calls[0][1]?.headers);
    expect(headers.get("x-headroom-session-token")).toBeNull();

    firstDispose();
    globalThis.fetch = originalFetch;
  });

  describe("child_process wrappers with omitted or null args/options slots", () => {
    // Real children, not mocks: the bug is that Node reads the injected env
    // object from the wrong parameter, so only a real call shows it. The shim
    // preload is switched off because the checkout has no dist/ for it to load.
    function probeScript(): { dir: string; script: string; run: string } {
      const dir = fs.realpathSync(fs.mkdtempSync(path.join(os.tmpdir(), "headroom-cp-")));
      const script = path.join(dir, "probe.cjs");
      fs.writeFileSync(
        script,
        "process.stdout.write(JSON.stringify({cwd: process.cwd(), proxy: process.env.HEADROOM_OPENCODE_TRANSPORT_PROXY_URL}));",
      );
      return { dir, script, run: `"${process.execPath}" "${script}"` };
    }

    function collect(child: ChildProcess): Promise<{ cwd: string; proxy?: string }> {
      return new Promise((resolve, reject) => {
        const timer = setTimeout(() => reject(new Error("child did not finish")), 5000);
        let out = "";
        child.stdout?.on("data", (chunk) => {
          out += chunk;
        });
        child.on("error", (err) => {
          clearTimeout(timer);
          reject(err);
        });
        child.on("close", () => {
          clearTimeout(timer);
          try {
            resolve(JSON.parse(out));
          } catch {
            reject(new Error(`child printed ${JSON.stringify(out)}`));
          }
        });
      });
    }

    function viaCallback(
      call: (cb: (err: Error | null, stdout: string | Buffer) => void) => unknown,
    ): Promise<{ cwd: string; proxy?: string }> {
      return new Promise((resolve, reject) => {
        const timer = setTimeout(() => reject(new Error("callback was never called")), 3000);
        call((err, stdout) => {
          clearTimeout(timer);
          if (err) reject(err);
          else resolve(JSON.parse(String(stdout)));
        });
      });
    }

    const cp = childProcess as unknown as Record<string, (...args: unknown[]) => any>;

    // Earlier tests in this file can leave a mock behind in child_process, so
    // start from the functions Node loaded.
    const real = {
      spawn: childProcess.spawn,
      exec: childProcess.exec,
      execFile: childProcess.execFile,
      fork: childProcess.fork,
    };
    beforeEach(() => {
      Object.assign(childProcess, real);
    });

    function install(): void {
      vi.spyOn(fs, "existsSync").mockReturnValue(false);
      installHeadroomTransport({ proxyUrl: "http://127.0.0.1:8787/v1" });
    }

    const plainShapes: Array<[string, (p: { dir: string; script: string; run: string }) => Promise<{ cwd: string }>]> = [
      ["spawn(file, [args], { cwd })", (p) => collect(cp.spawn(process.execPath, [p.script], { cwd: p.dir }))],
      ["spawn(cmd, { cwd, shell })", (p) => collect(cp.spawn(p.run, { cwd: p.dir, shell: true }))],
      ["exec(cmd, { cwd }, cb)", (p) => viaCallback((cb) => cp.exec(p.run, { cwd: p.dir }, cb))],
      ["exec(cmd, cb)", (p) => viaCallback((cb) => cp.exec(p.run, cb)).then((r) => ({ ...r, cwd: p.dir }))],
      ["execFile(file, [args], { cwd }, cb)", (p) => viaCallback((cb) => cp.execFile(process.execPath, [p.script], { cwd: p.dir }, cb))],
      ["execFile(file, [args], cb)", (p) => viaCallback((cb) => cp.execFile(process.execPath, [p.script], cb)).then((r) => ({ ...r, cwd: p.dir }))],
      ["fork(module, [args], { cwd })", (p) => collect(cp.fork(p.script, [], { cwd: p.dir, silent: true }))],
      ["fork(module, { cwd })", (p) => collect(cp.fork(p.script, { cwd: p.dir, silent: true }))],
    ];

    it.each(plainShapes)("plain shape %s is unchanged", async (_name, call) => {
      const probe = probeScript();
      install();
      const seen = await call(probe);
      expect(seen.cwd).toBe(probe.dir);
    });

    it.each([undefined, null])("spawn(cmd, %s, { cwd }) keeps cwd and still injects the env", async (args) => {
      const { dir, run } = probeScript();
      install();
      const seen = await collect(cp.spawn(run, args, { cwd: dir, shell: true }));
      expect(seen.cwd).toBe(dir);
      expect(seen.proxy).toBe("http://127.0.0.1:8787/v1");
    });

    it.each([undefined, null])("fork(module, %s, { cwd }) keeps cwd and still injects the env", async (args) => {
      const { dir, script } = probeScript();
      install();
      const seen = await collect(cp.fork(script, args, { cwd: dir, silent: true }));
      expect(seen.cwd).toBe(dir);
      expect(seen.proxy).toBe("http://127.0.0.1:8787/v1");
    });

    it.each([undefined, null])("exec(cmd, %s, callback) still calls the callback", async (options) => {
      const { run } = probeScript();
      install();
      const seen = await viaCallback((cb) => cp.exec(run, options, cb));
      expect(seen.proxy).toBe("http://127.0.0.1:8787/v1");
    });

    it.each([undefined, null])("execFile(file, args, %s, callback) still calls the callback", async (options) => {
      const { script } = probeScript();
      install();
      const seen = await viaCallback((cb) => cp.execFile(process.execPath, [script], options, cb));
      expect(seen.proxy).toBe("http://127.0.0.1:8787/v1");
    });

    it("execFile(file, null, callback) and execFile(file, callback) still call the callback", async () => {
      install();
      for (const call of [
        (cb: any) => cp.execFile(process.execPath, null, cb),
        (cb: any) => cp.execFile(process.execPath, cb),
      ]) {
        // no script: node starts, reads the closed stdin and exits; only the callback matters
        await new Promise<void>((resolve, reject) => {
          const timer = setTimeout(() => reject(new Error("callback was never called")), 3000);
          const child = call(() => {
            clearTimeout(timer);
            resolve();
          });
          child.stdin?.end();
        });
      }
    });
  });

  it.each(["", "127.0.0.1:8787"])("leaves all traffic alone when the proxy URL is unusable (%j)", async (proxyUrl) => {
    const originalFetch = globalThis.fetch;
    const fetchMock = vi.fn(async (..._args: FetchCall) => new Response("ok"));
    globalThis.fetch = fetchMock as unknown as typeof fetch;
    const warn = vi.spyOn(console, "warn").mockImplementation(() => {});

    try {
      installHeadroomTransport({ proxyUrl });

      // Neither an LLM endpoint nor a plain download may throw or be rewritten.
      await fetch("https://api.anthropic.com/v1/messages", { method: "POST" });
      await fetch("https://registry.npmjs.org/left-pad");
      expect(fetchMock).toHaveBeenCalledTimes(2);
      expect(fetchMock.mock.calls[0][0]).toBe("https://api.anthropic.com/v1/messages");
      expect(fetchMock.mock.calls[1][0]).toBe("https://registry.npmjs.org/left-pad");
      expect(new Headers(fetchMock.mock.calls[0][1]?.headers).get("x-headroom-base-url")).toBeNull();

      for (const [request, scheme] of [[http.request, "http"], [https.request, "https"]] as const) {
        const req = request(`${scheme}://registry.npmjs.org/left-pad`, { method: "POST" });
        req.on("error", () => {});
        expect(req.host).toBe("registry.npmjs.org");
        req.destroy();
      }

      // The problem is reported once, not once per request.
      expect(warn).toHaveBeenCalledTimes(1);
      expect(warn).toHaveBeenCalledWith(
        `headroom: the configured proxy URL is ${proxyUrl === "" ? "empty" : "not a valid http(s) URL"}; requests are not routed`,
      );
    } finally {
      globalThis.fetch = originalFetch;
      warn.mockRestore();
    }
  });

  it("does not log any part of an unusable proxy URL that carries credentials", async () => {
    const originalFetch = globalThis.fetch;
    globalThis.fetch = vi.fn(async (..._args: FetchCall) => new Response("ok")) as unknown as typeof fetch;
    const warn = vi.spyOn(console, "warn").mockImplementation(() => {});

    try {
      installHeadroomTransport({ proxyUrl: "user:secretpass@host:8787" });
      await fetch("https://api.anthropic.com/v1/messages", { method: "POST" });

      expect(warn).toHaveBeenCalledTimes(1);
      const logged = warn.mock.calls.flat().join(" ");
      expect(logged).not.toContain("secretpass");
      expect(logged).not.toContain("host:8787");
    } finally {
      globalThis.fetch = originalFetch;
      warn.mockRestore();
    }
  });
});
