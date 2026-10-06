import childProcess from "node:child_process";
import fs from "node:fs";
import http from "node:http";
import http2 from "node:http2";
import https from "node:https";
import { afterEach, describe, expect, it, vi } from "vitest";

import { installHeadroomTransport, uninstallHeadroomTransport } from "./transport.js";

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

  it("passes external, loopback, and excluded http2.connect authorities through unchanged", () => {
    const connectSpy = vi.spyOn(http2, "connect").mockImplementation(() => ({}) as never);
    installHeadroomTransport({ proxyUrl: "http://127.0.0.1:8787/v1", excludeHosts: ["opencode.ai"] });

    expect(() => http2.connect("https://api.openai.com")).not.toThrow();
    expect(() => http2.connect("http://127.0.0.1:4096")).not.toThrow();
    expect(() => http2.connect("https://opencode.ai")).not.toThrow();

    expect(connectSpy).toHaveBeenCalledWith("https://api.openai.com");
    expect(connectSpy).toHaveBeenCalledWith("http://127.0.0.1:4096");
    expect(connectSpy).toHaveBeenCalledWith("https://opencode.ai");
  });

  it("injects the hook-shim --import into process.env.NODE_OPTIONS when the shim exists", () => {
    const originalNodeOptions = process.env.NODE_OPTIONS;
    const originalProxyUrl = process.env.HEADROOM_OPENCODE_TRANSPORT_PROXY_URL;

    try {
      process.env.NODE_OPTIONS = "--trace-warnings";
      delete process.env.HEADROOM_OPENCODE_TRANSPORT_PROXY_URL;

      installHeadroomTransport({ proxyUrl: "http://127.0.0.1:8787/v1" });

      expect(process.env.HEADROOM_OPENCODE_TRANSPORT_PROXY_URL).toBe("http://127.0.0.1:8787/v1");
      expect(process.env.NODE_OPTIONS).toContain("--trace-warnings");
      expect(process.env.NODE_OPTIONS).toContain("--import=");

      installHeadroomTransport({ proxyUrl: "http://127.0.0.1:8787/v1" });
      const importCount = (process.env.NODE_OPTIONS?.match(/--import=/g) ?? []).length;
      expect(importCount).toBe(1);
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
      expect(options.env.NODE_OPTIONS).toBe("--trace-warnings");
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

  it("passes custom child NODE_OPTIONS through byte-for-byte and propagates HEADROOM env", () => {
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
      expect(options.env.NODE_OPTIONS).toContain("--import=");
    } finally {
      uninstallHeadroomTransport();
      childProcess.spawn = originalSpawn;
    }
  });

  it("hides child process windows on Windows", () => {
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

    try {
      installHeadroomTransport({ proxyUrl: "http://127.0.0.1:8787/v1" });
      childProcess.spawn("node", ["agent.js"]);

      const options = (spawnMock.mock.calls[0] as unknown[])[2] as { windowsHide: boolean };
      expect(options.windowsHide).toBe(true);
    } finally {
      uninstallHeadroomTransport();
      childProcess.spawn = originalSpawn;
    }
  });

  it("preserves an explicit request to show a child process window on Windows", () => {
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

    try {
      installHeadroomTransport({ proxyUrl: "http://127.0.0.1:8787/v1" });
      childProcess.spawn("node", ["agent.js"], { windowsHide: false });

      const options = (spawnMock.mock.calls[0] as unknown[])[2] as { windowsHide: boolean };
      expect(options.windowsHide).toBe(false);
    } finally {
      uninstallHeadroomTransport();
      childProcess.spawn = originalSpawn;
    }
  });

  it("injects one idempotent hook-shim --import into spawn/exec/execFile/fork children", () => {
    const originalSpawn = childProcess.spawn;
    const originalExec = childProcess.exec;
    const originalExecFile = childProcess.execFile;
    const originalFork = childProcess.fork;
    const originalNodeOptions = process.env.NODE_OPTIONS;
    const originalExclude = process.env.HEADROOM_OPENCODE_EXCLUDE_HOSTS;
    const childStub = { on: vi.fn(), once: vi.fn(), emit: vi.fn(), kill: vi.fn(), killed: false, pid: 123 };
    const spawnMock = vi.fn(() => childStub);
    const execMock = vi.fn(() => childStub);
    const execFileMock = vi.fn(() => childStub);
    const forkMock = vi.fn(() => childStub);
    childProcess.spawn = spawnMock as unknown as typeof childProcess.spawn;
    childProcess.exec = execMock as unknown as typeof childProcess.exec;
    childProcess.execFile = execFileMock as unknown as typeof childProcess.execFile;
    childProcess.fork = forkMock as unknown as typeof childProcess.fork;

    try {
      process.env.NODE_OPTIONS = "--max-old-space-size=4096 --trace-warnings";
      delete process.env.HEADROOM_OPENCODE_EXCLUDE_HOSTS;
      installHeadroomTransport({ proxyUrl: "http://127.0.0.1:8787/v1", excludeHosts: ["opencode.ai"] });

      const custom = { PATH: "/bin", NODE_OPTIONS: "--trace-warnings --experimental-vm-modules" };
      childProcess.spawn("npm", ["install"], { env: custom });
      childProcess.exec("npx -y firecrawl-mcp", { env: custom });
      childProcess.execFile("node", ["mcp-server.js"], { env: custom });
      childProcess.fork("agent.js", [], { env: custom });
      childProcess.spawn("bash", ["-lc", "echo hi"]);
      childProcess.spawn("node", ["mcp.js"]);
      childProcess.exec("bash -lc 'echo hi'");
      childProcess.execFile("npx", ["-y", "firecrawl-mcp"]);

      const spawnCalls = spawnMock.mock.calls as unknown[][];
      const execCalls = execMock.mock.calls as unknown[][];
      const execFileCalls = execFileMock.mock.calls as unknown[][];
      const forkCalls = forkMock.mock.calls as unknown[][];
      const customNodeOptions = "--trace-warnings --experimental-vm-modules";
      const defaultNodeOptions = "--max-old-space-size=4096 --trace-warnings";
      const pairs: Array<[unknown[], number, string]> = [
        [spawnCalls[0], 2, customNodeOptions],
        [execCalls[0], 1, customNodeOptions],
        [execFileCalls[0], 2, customNodeOptions],
        [forkCalls[0], 2, customNodeOptions],
        [spawnCalls[1], 2, defaultNodeOptions],
        [spawnCalls[2], 2, defaultNodeOptions],
        [execCalls[1], 1, defaultNodeOptions],
        [execFileCalls[1], 2, defaultNodeOptions],
      ];
      for (const [call, index, expectedNodeOptions] of pairs) {
        const env = (call[index] as { env: NodeJS.ProcessEnv }).env;
        const nodeOptions = env.NODE_OPTIONS ?? "";
        expect(nodeOptions).toContain(expectedNodeOptions);
        expect(nodeOptions).toContain("--import=");
        expect((nodeOptions.match(/--import=/g) ?? []).length).toBe(1);
        expect(env.HEADROOM_OPENCODE_TRANSPORT_PROXY_URL).toBe("http://127.0.0.1:8787/v1");
        expect(env.HEADROOM_OPENCODE_EXCLUDE_HOSTS).toBe("opencode.ai");
      }
      const processNodeOptions = process.env.NODE_OPTIONS ?? "";
      expect(processNodeOptions).toContain(defaultNodeOptions);
      expect((processNodeOptions.match(/--import=/g) ?? []).length).toBe(1);
    } finally {
      if (originalNodeOptions === undefined) {
        delete process.env.NODE_OPTIONS;
      } else {
        process.env.NODE_OPTIONS = originalNodeOptions;
      }
      if (originalExclude === undefined) {
        delete process.env.HEADROOM_OPENCODE_EXCLUDE_HOSTS;
      } else {
        process.env.HEADROOM_OPENCODE_EXCLUDE_HOSTS = originalExclude;
      }
      uninstallHeadroomTransport();
      childProcess.spawn = originalSpawn;
      childProcess.exec = originalExec;
      childProcess.execFile = originalExecFile;
      childProcess.fork = originalFork;
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
});
