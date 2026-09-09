import { chmod, mkdtemp, readFile, readdir, rm, stat, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, describe, expect, it, vi } from "vitest";

const mocked = vi.hoisted(() => ({
  compress: vi.fn(),
  start: vi.fn(async () => "http://127.0.0.1:8787"),
  stop: vi.fn(async () => undefined),
  logger: {
    debug: vi.fn(),
    error: vi.fn(),
    info: vi.fn(),
    warn: vi.fn(),
  },
}));

vi.mock("headroom-ai", () => ({
  compress: mocked.compress,
}));

vi.mock("openclaw/plugin-sdk/core", () => ({
  delegateCompactionToRuntime: vi.fn(async () => ({
    ok: true,
    compacted: true,
    reason: "delegated",
  })),
}), { virtual: true });

vi.mock("../src/proxy-manager.js", () => ({
  ProxyManager: class {
    start = mocked.start;
    stop = mocked.stop;
  },
  defaultLogger: mocked.logger,
}));

import { HeadroomContextEngine } from "../src/engine.js";
import { compress } from "headroom-ai";

afterEach(() => {
  vi.mocked(compress).mockReset();
  mocked.start.mockReset();
  mocked.start.mockResolvedValue("http://127.0.0.1:8787");
  mocked.stop.mockClear();
  mocked.logger.debug.mockClear();
  mocked.logger.error.mockClear();
  mocked.logger.info.mockClear();
  mocked.logger.warn.mockClear();
});

async function createSessionFile(records: unknown[]): Promise<{ directory: string; path: string }> {
  const directory = await mkdtemp(join(tmpdir(), "headroom-openclaw-"));
  const path = join(directory, "session.jsonl");
  await writeFile(path, `${records.map((record) => JSON.stringify(record)).join("\n")}\n`, "utf8");
  return { directory, path };
}

function setProxyUrl(engine: HeadroomContextEngine): void {
  (engine as { proxyUrl: string | null }).proxyUrl = "http://127.0.0.1:8787";
}

function createSerializedTranscriptRewrite(sessionPath: string) {
  let previous = Promise.resolve();

  return vi.fn(async (request: { replacements: Array<{ entryId: string; message: unknown }> }) => {
    const operation = previous.then(async () => {
      const replacements = new Map(
        request.replacements.map(({ entryId, message }) => [entryId, message]),
      );
      const content = await readFile(sessionPath, "utf8");
      const lines = content.split(/(?<=\n)/);
      let rewrittenEntries = 0;
      let bytesFreed = 0;
      const output = lines.map((line) => {
        if (!line.trim()) return line;
        const lineEnding = line.endsWith("\r\n") ? "\r\n" : line.endsWith("\n") ? "\n" : "";
        const record = JSON.parse(line) as Record<string, unknown>;
        const replacement = typeof record.id === "string" ? replacements.get(record.id) : undefined;
        if (record.type !== "message" || replacement === undefined) return line;
        rewrittenEntries++;
        const next = JSON.stringify({ ...record, message: replacement });
        bytesFreed += Math.max(0, Buffer.byteLength(line) - Buffer.byteLength(next + lineEnding));
        return next + lineEnding;
      });
      if (rewrittenEntries > 0) {
        await writeFile(sessionPath, output.join(""), "utf8");
      }
      return {
        changed: rewrittenEntries > 0,
        bytesFreed,
        rewrittenEntries,
        reason: rewrittenEntries > 0 ? undefined : "no matching message entries",
      };
    });
    previous = operation.then(
      () => undefined,
      () => undefined,
    );
    return operation;
  });
}

describe("HeadroomContextEngine proxy startup helpers", () => {
  it("bootstraps by scheduling proxy startup when enabled", async () => {
    const engine = new HeadroomContextEngine();

    await expect(
      engine.bootstrap({
        sessionId: "session-1",
        sessionFile: "session.jsonl",
      }),
    ).resolves.toEqual({
      bootstrapped: true,
      reason: "proxy startup scheduled",
    });
    expect(mocked.start).toHaveBeenCalledTimes(1);
  });

  it("removes unsubscribed proxy listeners before notifying readiness", async () => {
    const engine = new HeadroomContextEngine();
    const first = vi.fn();
    const second = vi.fn();

    const unsubscribeFirst = engine.onProxyReady(first);
    engine.onProxyReady(second);
    unsubscribeFirst();

    engine.ensureProxyStarted();
    await engine.ensureProxyUrl();

    expect(first).not.toHaveBeenCalled();
    expect(second).toHaveBeenCalledWith("http://127.0.0.1:8787");
  });

  it("returns the existing proxy URL without starting again", async () => {
    const engine = new HeadroomContextEngine();

    (engine as { proxyUrl: string | null }).proxyUrl = "http://127.0.0.1:8787";

    await expect(engine.ensureProxyUrl()).resolves.toBe("http://127.0.0.1:8787");
    expect(mocked.start).not.toHaveBeenCalled();
  });

  it("throws when proxy startup is disabled", async () => {
    const engine = new HeadroomContextEngine({ enabled: false });

    await expect(engine.ensureProxyUrl()).rejects.toThrow("Headroom proxy startup is disabled");
    expect(mocked.start).not.toHaveBeenCalled();
  });

  it("does not emit an unhandledRejection when fire-and-forget startup fails", async () => {
    mocked.start.mockReset();
    mocked.start.mockRejectedValue(new Error("proxy boom"));

    const engine = new HeadroomContextEngine();
    const unhandled: unknown[] = [];
    const onUnhandled = (reason: unknown) => unhandled.push(reason);
    process.on("unhandledRejection", onUnhandled);

    try {
      // Fire-and-forget: caller intentionally does not await.
      engine.ensureProxyStarted();
      // Let the startup promise settle and any microtasks/macrotasks flush.
      await new Promise((resolve) => setTimeout(resolve, 0));

      expect(unhandled).toEqual([]);
      expect(mocked.logger.warn).toHaveBeenCalledWith(
        expect.stringContaining("Headroom proxy unavailable"),
      );
    } finally {
      process.off("unhandledRejection", onUnhandled);
    }
  });

  it("stores the startup failure in getProxyStartupError()", async () => {
    const failure = new Error("proxy boom");
    mocked.start.mockReset();
    mocked.start.mockRejectedValue(failure);

    const engine = new HeadroomContextEngine();
    expect(engine.getProxyStartupError()).toBeNull();

    engine.ensureProxyStarted();
    await new Promise((resolve) => setTimeout(resolve, 0));

    expect(engine.getProxyStartupError()).toBe(failure);
  });

  it("allows retrying startup after a failure", async () => {
    mocked.start.mockReset();
    mocked.start
      .mockRejectedValueOnce(new Error("proxy boom"))
      .mockResolvedValueOnce("http://127.0.0.1:8787");

    const engine = new HeadroomContextEngine();

    engine.ensureProxyStarted();
    await new Promise((resolve) => setTimeout(resolve, 0));
    expect(engine.getProxyStartupError()).toBeInstanceOf(Error);

    // A second attempt is possible once the failed promise has cleared.
    const url = await engine.ensureProxyUrl();
    expect(url).toBe("http://127.0.0.1:8787");
    expect(engine.getProxyStartupError()).toBeNull();
    expect(mocked.start).toHaveBeenCalledTimes(2);
  });

  it("ensureProxyUrl rejects cleanly on startup failure without unhandledRejection", async () => {
    const failure = new Error("proxy boom");
    mocked.start.mockReset();
    mocked.start.mockRejectedValue(failure);

    const engine = new HeadroomContextEngine();
    const unhandled: unknown[] = [];
    const onUnhandled = (reason: unknown) => unhandled.push(reason);
    process.on("unhandledRejection", onUnhandled);

    try {
      await expect(engine.ensureProxyUrl()).rejects.toBe(failure);
      await new Promise((resolve) => setTimeout(resolve, 0));
      expect(unhandled).toEqual([]);
    } finally {
      process.off("unhandledRejection", onUnhandled);
    }
  });

  it("isolates and logs proxy-ready listener rejections", async () => {
    const engine = new HeadroomContextEngine();
    const failing = vi.fn(async () => {
      throw new Error("listener boom");
    });
    const healthy = vi.fn();

    engine.onProxyReady(failing);
    engine.onProxyReady(healthy);

    engine.ensureProxyStarted();
    // ensureProxyUrl must still resolve despite the listener throwing.
    await expect(engine.ensureProxyUrl()).resolves.toBe("http://127.0.0.1:8787");

    expect(failing).toHaveBeenCalled();
    expect(healthy).toHaveBeenCalledWith("http://127.0.0.1:8787");
    expect(mocked.logger.warn).toHaveBeenCalledWith(
      expect.stringContaining("Headroom proxy ready listener failed"),
    );
    expect(engine.getProxyStartupError()).toBeNull();
  });

  it("schedules startup and returns original messages when assembling before proxy readiness", async () => {
    const engine = new HeadroomContextEngine();
    const messages = [{ role: "user", content: "hello" }];

    await expect(
      engine.assemble({
        sessionId: "session-1",
        messages,
      }),
    ).resolves.toEqual({
      messages,
      estimatedTokens: 0,
    });
    expect(mocked.start).toHaveBeenCalledTimes(1);
  });

  it("clears the request timeout after successful compression", async () => {
    vi.useFakeTimers();
    try {
      vi.mocked(compress).mockResolvedValue({
        compressed: false,
        messages: [{ role: "user", content: "hello" }],
        tokensBefore: 5,
        tokensAfter: 5,
        tokensSaved: 0,
      });

      const engine = new HeadroomContextEngine({ requestTimeoutMs: 30_000 });
      (engine as { proxyUrl: string | null }).proxyUrl = "http://127.0.0.1:8787";

      await expect(
        engine.assemble({
          sessionId: "session-1",
          messages: [{ role: "user", content: "hello" }],
        }),
      ).resolves.toEqual({
        messages: [{ role: "user", content: "hello" }],
        estimatedTokens: 5,
      });

      expect(vi.getTimerCount()).toBe(0);
    } finally {
      vi.useRealTimers();
    }
  });

  it("opens the circuit after consecutive compression failures", async () => {
    vi.mocked(compress).mockRejectedValue(new Error("proxy stalled"));
    const messages = [{ role: "user", content: "hello" }];
    const engine = new HeadroomContextEngine({
      circuitBreakerThreshold: 2,
      circuitBreakerCooldownMs: 60_000,
    });
    (engine as { proxyUrl: string | null }).proxyUrl = "http://127.0.0.1:8787";

    await engine.assemble({ sessionId: "session-1", messages });
    await engine.assemble({ sessionId: "session-1", messages });
    await expect(engine.assemble({ sessionId: "session-1", messages })).resolves.toEqual({
      messages,
      estimatedTokens: 0,
    });

    expect(compress).toHaveBeenCalledTimes(2);
    expect(mocked.logger.warn).toHaveBeenCalledWith(
      expect.stringContaining("Circuit breaker opened"),
    );
  });
});

describe("HeadroomContextEngine compaction", () => {
  it("compresses and persists message records without dropping transcript metadata", async () => {
    const session = await createSessionFile([
      { type: "session", version: 1, id: "session-1" },
      {
        type: "message",
        id: "message-1",
        parentId: null,
        message: { role: "user", content: "old ask", timestamp: 1 },
      },
      {
        type: "model_change",
        id: "model-1",
        parentId: "message-1",
        provider: "openai-codex",
        model: "gpt-5",
      },
      {
        type: "message",
        id: "message-2",
        parentId: "model-1",
        message: {
          role: "assistant",
          content: [{ type: "text", text: "old answer" }],
          timestamp: 2,
        },
      },
    ]);
    const engine = new HeadroomContextEngine();
    setProxyUrl(engine);
    const rewriteTranscriptEntries = createSerializedTranscriptRewrite(session.path);
    mocked.compress.mockResolvedValue({
      messages: [
        { role: "user", content: "compressed ask", _headroomMeta: { timestamp: 1 } },
        { role: "assistant", content: "compressed answer", _headroomMeta: { timestamp: 2 } },
      ],
      tokensBefore: 100,
      tokensAfter: 50,
      tokensSaved: 50,
      compressionRatio: 0.5,
      transformsApplied: ["smart_crusher"],
      ccrHashes: [],
      compressed: true,
    });

    try {
      await expect(
        engine.compact({
          sessionId: "session-1",
          sessionFile: session.path,
          tokenBudget: 60,
          force: true,
          runtimeContext: { rewriteTranscriptEntries },
        }),
      ).resolves.toEqual({
        ok: true,
        compacted: true,
        reason: "Compacted session with Headroom",
        result: { tokensBefore: 100, tokensAfter: 50 },
      });

      const output = (await readFile(session.path, "utf8"))
        .trim()
        .split("\n")
        .map((line) => JSON.parse(line));
      expect(output[0]).toEqual({ type: "session", version: 1, id: "session-1" });
      expect(output[1]).toEqual({
        type: "message",
        id: "message-1",
        parentId: null,
        message: { role: "user", content: "compressed ask", timestamp: 1 },
      });
      expect(output[2]).toEqual({
        type: "model_change",
        id: "model-1",
        parentId: "message-1",
        provider: "openai-codex",
        model: "gpt-5",
      });
      expect(output[3]).toMatchObject({
        type: "message",
        id: "message-2",
        parentId: "model-1",
        message: {
          role: "assistant",
          content: [{ type: "text", text: "compressed answer" }],
          timestamp: 2,
        },
      });
      expect(mocked.compress).toHaveBeenCalledWith(
        expect.arrayContaining([
          expect.objectContaining({ role: "user", content: "old ask" }),
          expect.objectContaining({ role: "assistant", content: "old answer" }),
        ]),
        {
          model: "claude-sonnet-4-5",
          baseUrl: "http://127.0.0.1:8787",
          fallback: true,
          tokenBudget: 60,
        },
      );
      expect(engine.getStats().compactions).toBe(1);
      expect(rewriteTranscriptEntries).toHaveBeenCalledWith({
        replacements: [
          {
            entryId: "message-1",
            message: { role: "user", content: "compressed ask", timestamp: 1 },
          },
          {
            entryId: "message-2",
            message: expect.objectContaining({
              role: "assistant",
              content: [{ type: "text", text: "compressed answer" }],
              timestamp: 2,
            }),
          },
        ],
      });
    } finally {
      await rm(session.directory, { recursive: true, force: true });
    }
  });

  it("fails closed for malformed transcripts without modifying the file", async () => {
    const session = await createSessionFile([{ type: "session", version: 1, id: "session-1" }]);
    await writeFile(session.path, '{"type":"session","version":1,"id":"session-1"}\nnot-json\n', "utf8");
    const original = await readFile(session.path, "utf8");
    const engine = new HeadroomContextEngine();
    setProxyUrl(engine);
    const rewriteTranscriptEntries = createSerializedTranscriptRewrite(session.path);

    try {
      await expect(
        engine.compact({
          sessionId: "session-1",
          sessionFile: session.path,
          runtimeContext: { rewriteTranscriptEntries },
        }),
      ).resolves.toMatchObject({
        ok: false,
        compacted: false,
        reason: expect.stringContaining("Invalid JSONL record"),
      });
      expect(await readFile(session.path, "utf8")).toBe(original);
      expect(mocked.compress).not.toHaveBeenCalled();
      expect(rewriteTranscriptEntries).not.toHaveBeenCalled();
      expect(engine.getStats().compactions).toBe(0);
    } finally {
      await rm(session.directory, { recursive: true, force: true });
    }
  });

  it("does not rewrite a session when compression reports no change", async () => {
    const session = await createSessionFile([
      { type: "session", version: 1, id: "session-1" },
      {
        type: "message",
        id: "message-1",
        parentId: null,
        message: { role: "user", content: "already compact" },
      },
    ]);
    const original = await readFile(session.path, "utf8");
    const engine = new HeadroomContextEngine();
    setProxyUrl(engine);
    const rewriteTranscriptEntries = createSerializedTranscriptRewrite(session.path);
    mocked.compress.mockResolvedValue({
      messages: [{ role: "user", content: "already compact" }],
      tokensBefore: 10,
      tokensAfter: 10,
      tokensSaved: 0,
      compressionRatio: 1,
      transformsApplied: [],
      ccrHashes: [],
      compressed: false,
    });

    try {
      await expect(
        engine.compact({
          sessionId: "session-1",
          sessionFile: session.path,
          runtimeContext: { rewriteTranscriptEntries },
        }),
      ).resolves.toEqual({
        ok: true,
        compacted: false,
        reason: "Session did not need compression",
        result: { tokensBefore: 10, tokensAfter: 10 },
      });
      expect(await readFile(session.path, "utf8")).toBe(original);
      expect(engine.getStats().compactions).toBe(0);
      expect(rewriteTranscriptEntries).not.toHaveBeenCalled();
    } finally {
      await rm(session.directory, { recursive: true, force: true });
    }
  });

  it("does not rewrite a session when compression changes the message count", async () => {
    const session = await createSessionFile([
      { type: "session", version: 1, id: "session-1" },
      {
        type: "message",
        id: "message-1",
        parentId: null,
        message: { role: "user", content: "keep me" },
      },
    ]);
    const original = await readFile(session.path, "utf8");
    const engine = new HeadroomContextEngine();
    setProxyUrl(engine);
    const rewriteTranscriptEntries = createSerializedTranscriptRewrite(session.path);
    mocked.compress.mockResolvedValue({
      messages: [],
      tokensBefore: 10,
      tokensAfter: 0,
      tokensSaved: 10,
      compressionRatio: 0,
      transformsApplied: ["history_drop"],
      ccrHashes: [],
      compressed: true,
    });

    try {
      await expect(
        engine.compact({
          sessionId: "session-1",
          sessionFile: session.path,
          runtimeContext: { rewriteTranscriptEntries },
        }),
      ).resolves.toEqual({
        ok: false,
        compacted: false,
        reason: "Compression changed the session message count",
        result: { tokensBefore: 10, tokensAfter: 0 },
      });
      expect(await readFile(session.path, "utf8")).toBe(original);
      expect(engine.getStats().compactions).toBe(0);
      expect(rewriteTranscriptEntries).not.toHaveBeenCalled();
    } finally {
      await rm(session.directory, { recursive: true, force: true });
    }
  });

  it("delegates to OpenClaw when compact has no maintenance rewrite capability", async () => {
    const session = await createSessionFile([
      { type: "session", version: 1, id: "session-1" },
      {
        type: "message",
        id: "message-1",
        parentId: null,
        message: { role: "user", content: "private transcript" },
      },
    ]);
    const original = await readFile(session.path, "utf8");
    const collidingTemporaryPath = `${session.path}.headroom.tmp`;
    await writeFile(collidingTemporaryPath, "do not touch", "utf8");
    await chmod(session.path, 0o600);
    const engine = new HeadroomContextEngine();
    setProxyUrl(engine);

    try {
      await expect(
        engine.compact({ sessionId: "session-1", sessionFile: session.path }),
      ).resolves.toEqual({ ok: true, compacted: true, reason: "delegated" });
      expect(await readFile(session.path, "utf8")).toBe(original);
      if (process.platform !== "win32") {
        expect((await stat(session.path)).mode & 0o777).toBe(0o600);
      }
      expect(await readFile(collidingTemporaryPath, "utf8")).toBe("do not touch");
      expect(mocked.compress).not.toHaveBeenCalled();
      expect(engine.getStats().compactions).toBe(0);
    } finally {
      await rm(session.directory, { recursive: true, force: true });
    }
  });

  it("preserves an entry appended while compression is pending", async () => {
    const session = await createSessionFile([
      { type: "session", version: 1, id: "session-1" },
      {
        type: "message",
        id: "message-1",
        parentId: null,
        message: { role: "user", content: "old ask" },
      },
    ]);
    const engine = new HeadroomContextEngine();
    setProxyUrl(engine);
    const rewriteTranscriptEntries = createSerializedTranscriptRewrite(session.path);
    let finishCompression!: (value: any) => void;
    mocked.compress.mockImplementation(
      () => new Promise((resolve) => (finishCompression = resolve)),
    );

    try {
      const compacting = engine.compact({
        sessionId: "session-1",
        sessionFile: session.path,
        runtimeContext: { rewriteTranscriptEntries },
      });
      await vi.waitFor(() => expect(mocked.compress).toHaveBeenCalledTimes(1));
      await writeFile(
        session.path,
        `${JSON.stringify({
          type: "message",
          id: "message-2",
          parentId: "message-1",
          message: { role: "assistant", content: "concurrent append" },
        })}\n`,
        { encoding: "utf8", flag: "a" },
      );
      finishCompression({
        messages: [{ role: "user", content: "compressed ask" }],
        tokensBefore: 100,
        tokensAfter: 50,
        tokensSaved: 50,
        compressionRatio: 0.5,
        transformsApplied: ["smart_crusher"],
        ccrHashes: [],
        compressed: true,
      });

      await expect(compacting).resolves.toMatchObject({ ok: true, compacted: true });
      const output = (await readFile(session.path, "utf8"))
        .trim()
        .split("\n")
        .map((line) => JSON.parse(line));
      expect(output).toHaveLength(3);
      expect(output[1].message.content).toBe("compressed ask");
      expect(output[2]).toMatchObject({
        id: "message-2",
        message: { content: "concurrent append" },
      });
    } finally {
      await rm(session.directory, { recursive: true, force: true });
    }
  });

  it("preserves 0600 permissions and creates no plugin-owned temporary file", async () => {
    const session = await createSessionFile([
      { type: "session", version: 1, id: "session-1" },
      {
        type: "message",
        id: "message-1",
        parentId: null,
        message: { role: "user", content: "old ask" },
      },
    ]);
    await chmod(session.path, 0o600);
    const engine = new HeadroomContextEngine();
    setProxyUrl(engine);
    const rewriteTranscriptEntries = createSerializedTranscriptRewrite(session.path);
    mocked.compress.mockResolvedValue({
      messages: [{ role: "user", content: "compressed ask" }],
      tokensBefore: 100,
      tokensAfter: 50,
      tokensSaved: 50,
      compressionRatio: 0.5,
      transformsApplied: ["smart_crusher"],
      ccrHashes: [],
      compressed: true,
    });

    try {
      await expect(
        engine.compact({
          sessionId: "session-1",
          sessionFile: session.path,
          runtimeContext: { rewriteTranscriptEntries },
        }),
      ).resolves.toMatchObject({ ok: true, compacted: true });
      if (process.platform !== "win32") {
        expect((await stat(session.path)).mode & 0o777).toBe(0o600);
      }
      expect(await readdir(session.directory)).toEqual(["session.jsonl"]);
    } finally {
      await rm(session.directory, { recursive: true, force: true });
    }
  });

  it("leaves the transcript byte-identical when the runtime rewrite fails", async () => {
    const session = await createSessionFile([
      { type: "session", version: 1, id: "session-1" },
      {
        type: "message",
        id: "message-1",
        parentId: null,
        message: { role: "user", content: "old ask" },
      },
    ]);
    const original = await readFile(session.path);
    const engine = new HeadroomContextEngine();
    setProxyUrl(engine);
    const rewriteTranscriptEntries = vi.fn(async () => {
      throw new Error("runtime lock unavailable");
    });
    mocked.compress.mockResolvedValue({
      messages: [{ role: "user", content: "compressed ask" }],
      tokensBefore: 100,
      tokensAfter: 50,
      tokensSaved: 50,
      compressionRatio: 0.5,
      transformsApplied: ["smart_crusher"],
      ccrHashes: [],
      compressed: true,
    });

    try {
      await expect(
        engine.compact({
          sessionId: "session-1",
          sessionFile: session.path,
          runtimeContext: { rewriteTranscriptEntries },
        }),
      ).resolves.toMatchObject({
        ok: false,
        compacted: false,
        reason: expect.stringContaining("runtime lock unavailable"),
      });
      expect(await readFile(session.path)).toEqual(original);
      expect(engine.getStats().compactions).toBe(0);
    } finally {
      await rm(session.directory, { recursive: true, force: true });
    }
  });

  it("fails the stale contender closed when two compactions race", async () => {
    const session = await createSessionFile([
      { type: "session", version: 1, id: "session-1" },
      {
        type: "message",
        id: "message-1",
        parentId: null,
        message: { role: "user", content: "old ask" },
      },
    ]);
    const engine = new HeadroomContextEngine();
    setProxyUrl(engine);
    let releaseCompression!: () => void;
    const compressionGate = new Promise<void>((resolve) => (releaseCompression = resolve));
    mocked.compress.mockImplementation(async () => {
      await compressionGate;
      return {
        messages: [{ role: "user", content: "compressed ask" }],
        tokensBefore: 100,
        tokensAfter: 50,
        tokensSaved: 50,
        compressionRatio: 0.5,
        transformsApplied: ["smart_crusher"],
        ccrHashes: [],
        compressed: true,
      };
    });
    let previousRewrite = Promise.resolve();
    let rewrittenId = 0;
    const rewriteTranscriptEntries = vi.fn(
      async (request: { replacements: Array<{ entryId: string; message: unknown }> }) => {
        const operation = previousRewrite.then(async () => {
          const content = await readFile(session.path, "utf8");
          const lines = content.trim().split("\n");
          const records = lines.map((line) => JSON.parse(line));
          const replacement = request.replacements.find(
            ({ entryId }) => entryId === records[1]?.id,
          );
          if (!replacement) {
            return {
              changed: false,
              bytesFreed: 0,
              rewrittenEntries: 0,
              reason: "no matching message entries",
            };
          }
          rewrittenId++;
          records[1] = {
            ...records[1],
            id: `rewritten-${rewrittenId}`,
            message: replacement.message,
          };
          await writeFile(
            session.path,
            `${records.map((record) => JSON.stringify(record)).join("\n")}\n`,
            "utf8",
          );
          return { changed: true, bytesFreed: 1, rewrittenEntries: 1 };
        });
        previousRewrite = operation.then(
          () => undefined,
          () => undefined,
        );
        return operation;
      },
    );

    try {
      const first = engine.compact({
        sessionId: "session-1",
        sessionFile: session.path,
        runtimeContext: { rewriteTranscriptEntries },
      });
      const second = engine.compact({
        sessionId: "session-1",
        sessionFile: session.path,
        runtimeContext: { rewriteTranscriptEntries },
      });
      await vi.waitFor(() => expect(mocked.compress).toHaveBeenCalledTimes(2));
      releaseCompression();
      const results = await Promise.all([first, second]);

      expect(results.filter((result) => result.ok)).toHaveLength(1);
      expect(results.filter((result) => !result.ok)).toEqual([
        expect.objectContaining({
          compacted: false,
          reason: "no matching message entries",
        }),
      ]);
      const output = (await readFile(session.path, "utf8"))
        .trim()
        .split("\n")
        .map((line) => JSON.parse(line));
      expect(output).toHaveLength(2);
      expect(output[1]).toMatchObject({
        id: "rewritten-1",
        message: { role: "user", content: "compressed ask" },
      });
      expect(engine.getStats().compactions).toBe(1);
    } finally {
      await rm(session.directory, { recursive: true, force: true });
    }
  });
});
