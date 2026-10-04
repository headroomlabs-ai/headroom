import * as fs from "node:fs/promises";
import * as os from "node:os";
import * as path from "node:path";

import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

const mocked = vi.hoisted(() => ({
  compress: vi.fn(),
  delegateCompactionToRuntime: vi.fn(),
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

vi.mock("../src/openclaw-compaction.js", () => ({
  delegateCompactionToRuntime: mocked.delegateCompactionToRuntime,
}));

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
    mocked.compress.mockReset();
    mocked.delegateCompactionToRuntime.mockReset();
  mocked.start.mockReset();
  mocked.start.mockResolvedValue("http://127.0.0.1:8787");
  mocked.stop.mockClear();
  mocked.logger.debug.mockClear();
  mocked.logger.error.mockClear();
  mocked.logger.info.mockClear();
  mocked.logger.warn.mockClear();
});

describe("HeadroomContextEngine compaction", () => {
  it("delegates persistent compaction to OpenClaw without claiming ownership", async () => {
    const engine = new HeadroomContextEngine();
    const params = {
      sessionId: "session-1",
      sessionKey: "agent:main:session-1",
      tokenBudget: 12_000,
      force: true,
      runtimeContext: { workspaceDir: "/tmp/workspace" },
    };
    const delegatedResult = {
      ok: true,
      compacted: true,
      result: {
        tokensBefore: 20_000,
        tokensAfter: 8_000,
      },
    };
    mocked.delegateCompactionToRuntime.mockResolvedValueOnce(delegatedResult);

    expect(engine.info.ownsCompaction).toBe(false);
    await expect(engine.compact(params)).resolves.toEqual(delegatedResult);

    expect(mocked.delegateCompactionToRuntime).toHaveBeenCalledWith(params);
    expect(mocked.compress).not.toHaveBeenCalled();
    expect(engine.getStats().compactions).toBe(1);
  });

  it("does not count a delegated no-op as a compaction", async () => {
    const engine = new HeadroomContextEngine();
    mocked.delegateCompactionToRuntime.mockResolvedValueOnce({
      ok: true,
      compacted: false,
      reason: "Below compaction threshold",
    });

    await expect(
      engine.compact({
        sessionId: "session-1",
        sessionKey: "agent:main:session-1",
      }),
    ).resolves.toEqual({
      ok: true,
      compacted: false,
      reason: "Below compaction threshold",
    });

    expect(engine.getStats().compactions).toBe(0);
  });

  it("propagates delegated compaction failures without reporting success", async () => {
    const engine = new HeadroomContextEngine();
    const failure = new Error("native compaction failed");
    mocked.delegateCompactionToRuntime.mockRejectedValueOnce(failure);

    await expect(
      engine.compact({
        sessionId: "session-1",
        sessionKey: "agent:main:session-1",
      }),
    ).rejects.toBe(failure);

    expect(engine.getStats().compactions).toBe(0);
    expect(mocked.logger.info).not.toHaveBeenCalled();
  });
});

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

describe("HeadroomContextEngine transcriptSemantics contract", () => {
  let commitLogDir: string;
  let commitLogPath: string;

  beforeEach(async () => {
    commitLogDir = await fs.mkdtemp(path.join(os.tmpdir(), "headroom-commit-log-"));
    commitLogPath = path.join(commitLogDir, "commit-log.json");
  });

  afterEach(async () => {
    await fs.rm(commitLogDir, { recursive: true, force: true });
  });

  it("declares the durable-commit transcript semantics OpenClaw requires", () => {
    const engine = new HeadroomContextEngine({ commitLogPath });

    expect(engine.info.transcriptSemantics).toEqual({
      currentTurnFence: "before-current-turn-entry-v1",
      turnAdvancementIdempotency: "atomic-idempotent-v1",
    });
  });

  it("commits a new advancement key", async () => {
    const engine = new HeadroomContextEngine({ commitLogPath });

    await expect(
      engine.commitTurn({ advancementKey: "turn-1", messages: [] }),
    ).resolves.toEqual({ status: "committed" });
  });

  it("reports duplicate on a retried advancement key", async () => {
    const engine = new HeadroomContextEngine({ commitLogPath });

    await expect(
      engine.commitTurn({ advancementKey: "turn-1", messages: [] }),
    ).resolves.toEqual({ status: "committed" });
    await expect(
      engine.commitTurn({ advancementKey: "turn-1", messages: [] }),
    ).resolves.toEqual({ status: "duplicate" });
  });

  it("treats distinct advancement keys independently", async () => {
    const engine = new HeadroomContextEngine({ commitLogPath });

    await expect(
      engine.commitTurn({ advancementKey: "turn-1", messages: [] }),
    ).resolves.toEqual({ status: "committed" });
    await expect(
      engine.commitTurn({ advancementKey: "turn-2", messages: [] }),
    ).resolves.toEqual({ status: "committed" });
  });

  it("reports duplicate for a key committed before a process restart", async () => {
    // Simulate a restart: a brand new engine instance (no shared in-memory
    // state) pointed at the same durable commit-log path.
    const before = new HeadroomContextEngine({ commitLogPath });
    await expect(
      before.commitTurn({ advancementKey: "turn-restart", messages: [] }),
    ).resolves.toEqual({ status: "committed" });

    const after = new HeadroomContextEngine({ commitLogPath });
    await expect(
      after.commitTurn({ advancementKey: "turn-restart", messages: [] }),
    ).resolves.toEqual({ status: "duplicate" });
  });

  it(
    "never forgets a key regardless of how many other keys were committed since",
    async () => {
      // Regression: the old implementation evicted the oldest key past a
      // 512-entry cap, so a retry of an early key was wrongly re-accepted as
      // new instead of reported as a duplicate. There is no such cap now.
      const engine = new HeadroomContextEngine({ commitLogPath });

      for (let i = 0; i < 600; i++) {
        await engine.commitTurn({ advancementKey: `turn-${i}`, messages: [] });
      }

      await expect(
        engine.commitTurn({ advancementKey: "turn-0", messages: [] }),
      ).resolves.toEqual({ status: "duplicate" });
    },
    20_000,
  );

  it("persists the accepted messages together with the advancement key", async () => {
    const engine = new HeadroomContextEngine({ commitLogPath });
    const messages = [{ role: "user", content: "hello" }];

    await expect(
      engine.commitTurn({ advancementKey: "turn-1", messages }),
    ).resolves.toEqual({ status: "committed" });

    const raw = await fs.readFile(commitLogPath, "utf8");
    const entries = JSON.parse(raw) as Record<string, { messages: unknown }>;
    expect(entries["turn-1"].messages).toEqual(messages);
  });
});

describe("HeadroomContextEngine assemble() compression notice", () => {
  const HEADROOM_COMPRESSION_NOTICE =
    "[Headroom is compressing tool outputs in this session. Use headroom_retrieve if you need the original, uncompressed content.]";
  const messages = [{ role: "user", content: "hello" }];

  function mockCompressResult(overrides: Partial<{
    compressed: boolean;
    tokensSaved: number;
    tokensBefore: number;
    tokensAfter: number;
  }>) {
    const tokensSaved = overrides.tokensSaved ?? 0;
    return {
      compressed: overrides.compressed ?? tokensSaved > 0,
      messages: [{ role: "user", content: "hello" }],
      tokensBefore: overrides.tokensBefore ?? 1000,
      tokensAfter: overrides.tokensAfter ?? 1000 - tokensSaved,
      tokensSaved,
    };
  }

  function readyEngine(config?: { announceCompression?: boolean }) {
    const engine = new HeadroomContextEngine(config);
    (engine as unknown as { proxyUrl: string | null }).proxyUrl = "http://127.0.0.1:8787";
    return engine;
  }

  it("returns the static notice, with no interpolated count, once tokensSaved crosses the threshold", async () => {
    vi.mocked(compress).mockResolvedValueOnce(mockCompressResult({ tokensSaved: 150 }));

    const engine = readyEngine();
    const result = await engine.assemble({ sessionId: "s1", messages });

    expect(result.systemPromptAddition).toBe(HEADROOM_COMPRESSION_NOTICE);
  });

  it("returns byte-identical notices across turns with different tokensSaved amounts", async () => {
    vi.mocked(compress)
      .mockResolvedValueOnce(mockCompressResult({ tokensSaved: 150 }))
      .mockResolvedValueOnce(mockCompressResult({ tokensSaved: 9000 }));

    const engine = readyEngine();
    const first = await engine.assemble({ sessionId: "s1", messages });
    const second = await engine.assemble({ sessionId: "s1", messages });

    expect(first.systemPromptAddition).toBe(HEADROOM_COMPRESSION_NOTICE);
    expect(second.systemPromptAddition).toBe(first.systemPromptAddition);
  });

  it("keeps the notice present on a later turn that has nothing to compress", async () => {
    vi.mocked(compress)
      .mockResolvedValueOnce(mockCompressResult({ tokensSaved: 150 }))
      .mockResolvedValueOnce(mockCompressResult({ compressed: false, tokensSaved: 0 }));

    const engine = readyEngine();
    const first = await engine.assemble({ sessionId: "s1", messages });
    const second = await engine.assemble({ sessionId: "s1", messages });

    expect(first.systemPromptAddition).toBe(HEADROOM_COMPRESSION_NOTICE);
    expect(second.systemPromptAddition).toBe(HEADROOM_COMPRESSION_NOTICE);
  });

  it("never returns a notice when announceCompression is false", async () => {
    vi.mocked(compress).mockResolvedValueOnce(mockCompressResult({ tokensSaved: 500 }));

    const engine = readyEngine({ announceCompression: false });
    const result = await engine.assemble({ sessionId: "s1", messages });

    expect(result.systemPromptAddition).toBeUndefined();
  });

  it("does not announce for compression below the noise threshold", async () => {
    vi.mocked(compress).mockResolvedValueOnce(mockCompressResult({ tokensSaved: 50 }));

    const engine = readyEngine();
    const result = await engine.assemble({ sessionId: "s1", messages });

    expect(result.systemPromptAddition).toBeUndefined();
  });

  it("does not leak a session's announcement to a different session on the same engine", async () => {
    vi.mocked(compress)
      .mockResolvedValueOnce(mockCompressResult({ tokensSaved: 150 }))
      .mockResolvedValueOnce(mockCompressResult({ compressed: false, tokensSaved: 0 }));

    const engine = readyEngine();
    const sessionA = await engine.assemble({ sessionId: "a", messages });
    const sessionB = await engine.assemble({ sessionId: "b", messages });

    expect(sessionA.systemPromptAddition).toBe(HEADROOM_COMPRESSION_NOTICE);
    expect(sessionB.systemPromptAddition).toBeUndefined();
  });

  it("keeps announcing for the session that earned the notice even after another session is seen", async () => {
    vi.mocked(compress)
      .mockResolvedValueOnce(mockCompressResult({ tokensSaved: 150 }))
      .mockResolvedValueOnce(mockCompressResult({ compressed: false, tokensSaved: 0 }))
      .mockResolvedValueOnce(mockCompressResult({ compressed: false, tokensSaved: 0 }));

    const engine = readyEngine();
    await engine.assemble({ sessionId: "a", messages });
    await engine.assemble({ sessionId: "b", messages });
    const sessionAAgain = await engine.assemble({ sessionId: "a", messages });

    expect(sessionAAgain.systemPromptAddition).toBe(HEADROOM_COMPRESSION_NOTICE);
  });

  it("evicts the oldest tracked session once the announced-session bound is exceeded", async () => {
    vi.mocked(compress).mockResolvedValue(mockCompressResult({ tokensSaved: 150 }));

    const engine = readyEngine();
    for (let i = 0; i < 1000; i++) {
      await engine.assemble({ sessionId: `session-${i}`, messages });
    }
    const overflow = await engine.assemble({ sessionId: "session-overflow", messages });

    vi.mocked(compress).mockResolvedValueOnce(mockCompressResult({ compressed: false, tokensSaved: 0 }));
    const oldest = await engine.assemble({ sessionId: "session-0", messages });
    vi.mocked(compress).mockResolvedValueOnce(mockCompressResult({ compressed: false, tokensSaved: 0 }));
    const newest = await engine.assemble({ sessionId: "session-overflow", messages });

    expect(overflow.systemPromptAddition).toBe(HEADROOM_COMPRESSION_NOTICE);
    expect(oldest.systemPromptAddition).toBeUndefined();
    expect(newest.systemPromptAddition).toBe(HEADROOM_COMPRESSION_NOTICE);
  });
});
