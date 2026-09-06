import { afterEach, describe, expect, it, vi } from "vitest";

const mocked = vi.hoisted(() => ({
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
  compress: vi.fn(),
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
  vi.mocked(compress).mockReset();
  mocked.start.mockReset();
  mocked.start.mockResolvedValue("http://127.0.0.1:8787");
  mocked.stop.mockClear();
  mocked.logger.debug.mockClear();
  mocked.logger.error.mockClear();
  mocked.logger.info.mockClear();
  mocked.logger.warn.mockClear();
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
  it("declares the durable-commit transcript semantics OpenClaw requires", () => {
    const engine = new HeadroomContextEngine();

    expect(engine.info.transcriptSemantics).toEqual({
      currentTurnFence: "before-current-turn-entry-v1",
      turnAdvancementIdempotency: "atomic-idempotent-v1",
    });
  });

  it("commits a new advancement key", async () => {
    const engine = new HeadroomContextEngine();

    await expect(
      engine.commitTurn({ advancementKey: "turn-1", messages: [] }),
    ).resolves.toEqual({ status: "committed" });
  });

  it("reports duplicate on a retried advancement key", async () => {
    const engine = new HeadroomContextEngine();

    await expect(
      engine.commitTurn({ advancementKey: "turn-1", messages: [] }),
    ).resolves.toEqual({ status: "committed" });
    await expect(
      engine.commitTurn({ advancementKey: "turn-1", messages: [] }),
    ).resolves.toEqual({ status: "duplicate" });
  });

  it("treats distinct advancement keys independently", async () => {
    const engine = new HeadroomContextEngine();

    await expect(
      engine.commitTurn({ advancementKey: "turn-1", messages: [] }),
    ).resolves.toEqual({ status: "committed" });
    await expect(
      engine.commitTurn({ advancementKey: "turn-2", messages: [] }),
    ).resolves.toEqual({ status: "committed" });
  });

  it("bounds tracked advancement keys so long sessions cannot grow it unbounded", async () => {
    const engine = new HeadroomContextEngine();
    const cap = 512;

    for (let i = 0; i < cap + 1; i++) {
      await engine.commitTurn({ advancementKey: `turn-${i}`, messages: [] });
    }

    expect((engine as { committedAdvancementKeys: Set<string> }).committedAdvancementKeys.size).toBe(
      cap,
    );
    // The oldest key was evicted, so it is treated as new (committed) again rather than duplicate.
    await expect(
      engine.commitTurn({ advancementKey: "turn-0", messages: [] }),
    ).resolves.toEqual({ status: "committed" });
  });
});
