import childProcess from "node:child_process";
import fs from "node:fs";
import http from "node:http";
import http2 from "node:http2";
import https from "node:https";
import os from "node:os";
import path from "node:path";
import { pathToFileURL } from "node:url";
import { afterEach, describe, expect, it, vi } from "vitest";

import {
  acknowledgeNativeToolExecution,
  enforceNativeToolExecution,
  evaluateNativeToolPolicy,
  installHeadroomTransport,
  isAllowedToolPolicyUrl,
  refreshHeadroomToolPolicy,
  remoteToolPolicyCachePath,
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

  it("blocks external http2 connections instead of leaking them", () => {
    installHeadroomTransport({ proxyUrl: "http://127.0.0.1:8787/v1" });

    expect(() => http2.connect("https://api.openai.com")).toThrow(
      /blocked direct HTTP\/2 connection to api\.openai\.com/,
    );
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
});
