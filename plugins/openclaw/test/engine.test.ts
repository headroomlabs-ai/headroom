import { mkdtempSync, rmSync } from "node:fs";
import { join } from "node:path";
import { tmpdir } from "node:os";
import { afterEach, describe, expect, it, vi } from "vitest";

const mocked = vi.hoisted(() => ({
  delegateCompactionToRuntime: vi.fn(),
  awaitTranscriptProjectionSettle: vi.fn(async () => ({ waited: true })),
  runTranscriptReplaceHygiene: vi.fn(async () => ({
    changed: true,
    bytesFreed: 512,
    rewrittenEntries: 2,
    tokensBefore: 50_000,
    tokensAfter: 44_000,
  })),
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

vi.mock("../src/openclaw-compaction.js", () => ({
  delegateCompactionToRuntime: mocked.delegateCompactionToRuntime,
}));

vi.mock("../src/transcript-hygiene.js", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../src/transcript-hygiene.js")>();
  return {
    ...actual,
    runTranscriptReplaceHygiene: mocked.runTranscriptReplaceHygiene,
  };
});

vi.mock("../src/transcript-projection.js", () => ({
  awaitTranscriptProjectionSettle: mocked.awaitTranscriptProjectionSettle,
}));

vi.mock("../src/proxy-manager.js", () => ({
  ProxyManager: class {
    start = mocked.start;
    stop = mocked.stop;
  },
  defaultLogger: mocked.logger,
}));

import {
  DEFAULT_ASSEMBLE_RESERVE_TOKENS,
  DEFAULT_ASSEMBLE_SKIP_BUDGET_RATIO,
  HeadroomContextEngine,
  resolveAssembleReserveTokens,
  resolveAssembleSkipBudgetRatio,
  resolveAssembleSkipThreshold,
} from "../src/engine.js";
import { compress } from "headroom-ai";

afterEach(() => {
  vi.mocked(compress).mockReset();
  mocked.delegateCompactionToRuntime.mockReset();
  mocked.awaitTranscriptProjectionSettle.mockReset();
  mocked.awaitTranscriptProjectionSettle.mockResolvedValue({ waited: true });
  mocked.runTranscriptReplaceHygiene.mockReset();
  mocked.runTranscriptReplaceHygiene.mockResolvedValue({
    changed: true,
    bytesFreed: 512,
    rewrittenEntries: 2,
    tokensBefore: 50_000,
    tokensAfter: 44_000,
  });
  mocked.start.mockReset();
  mocked.start.mockResolvedValue("http://127.0.0.1:8787");
  mocked.stop.mockClear();
  mocked.logger.debug.mockClear();
  mocked.logger.error.mockClear();
  mocked.logger.info.mockClear();
  mocked.logger.warn.mockClear();
});

describe("HeadroomContextEngine persistent compaction mode", () => {
  it("defaults to openclaw compaction (OpenClaw owns durable compact)", () => {
    const engine = new HeadroomContextEngine();
    expect(engine.info.ownsCompaction).toBe(false);
    expect(engine.info.transcriptSemantics).toEqual({
      currentTurnFence: "before-current-turn-entry-v1",
      turnAdvancementIdempotency: "atomic-idempotent-v1",
    });
  });

  it("skips hybrid compact pre-pass when transcript hygiene is disabled", async () => {
    const engine = new HeadroomContextEngine({
      persistentCompaction: "hybrid",
      transcriptHygiene: { enabled: false },
    });
    (engine as { proxyUrl: string | null }).proxyUrl = "http://127.0.0.1:8787";
    const params = {
      sessionId: "session-1",
      sessionKey: "agent:main:session-1",
      sessionFile: "session.jsonl",
      tokenBudget: 120_000,
      force: true,
    };
    const delegatedResult = {
      ok: true,
      compacted: true,
      result: { tokensBefore: 20_000, tokensAfter: 8_000 },
    };
    mocked.delegateCompactionToRuntime.mockResolvedValueOnce(delegatedResult);

    await expect(engine.compact(params)).resolves.toEqual(delegatedResult);
    expect(mocked.runTranscriptReplaceHygiene).not.toHaveBeenCalled();
    expect(mocked.delegateCompactionToRuntime).toHaveBeenCalledWith(params);
    expect(engine.getStats().hygieneRuns).toBe(0);
  });

  it("runs hygiene pre-pass then delegates when hybrid compact is requested", async () => {
    const engine = new HeadroomContextEngine({ persistentCompaction: "hybrid" });
    (engine as { proxyUrl: string | null }).proxyUrl = "http://127.0.0.1:8787";
    const params = {
      sessionId: "session-1",
      sessionKey: "agent:main:session-1",
      sessionFile: "session.jsonl",
      tokenBudget: 120_000,
      force: true,
    };
    const delegatedResult = {
      ok: true,
      compacted: true,
      result: { tokensBefore: 20_000, tokensAfter: 8_000 },
    };
    mocked.delegateCompactionToRuntime.mockResolvedValueOnce(delegatedResult);

    await expect(engine.compact(params)).resolves.toEqual(delegatedResult);
    expect(mocked.runTranscriptReplaceHygiene).toHaveBeenCalledWith(
      expect.objectContaining({ force: true, proxyUrl: "http://127.0.0.1:8787" }),
    );
    expect(mocked.delegateCompactionToRuntime).toHaveBeenCalledWith(params);
    expect(engine.getStats().compactions).toBe(1);
    expect(engine.getStats().hygieneRuns).toBe(1);
  });

  it("waits for transcript projection after a successful hygiene rewrite", async () => {
    const engine = new HeadroomContextEngine({ persistentCompaction: "hybrid" });
    (engine as { proxyUrl: string | null }).proxyUrl = "http://127.0.0.1:8787";

    await engine.maintain({
      sessionId: "session-1",
      sessionKey: "agent:main:session-1",
      sessionFile: "session.jsonl",
    });

    expect(mocked.awaitTranscriptProjectionSettle).toHaveBeenCalledTimes(1);
  });

  it("debounces turn-end hygiene when a rewrite just ran", async () => {
    const engine = new HeadroomContextEngine({
      persistentCompaction: "hybrid",
      transcriptHygiene: { debounceMs: 60_000 },
    });
    (engine as { proxyUrl: string | null }).proxyUrl = "http://127.0.0.1:8787";

    await engine.maintain({
      sessionId: "session-1",
      sessionKey: "agent:main:session-1",
      sessionFile: "session.jsonl",
    });
    await expect(
      engine.maintain({
        sessionId: "session-1",
        sessionKey: "agent:main:session-1",
        sessionFile: "session.jsonl",
      }),
    ).resolves.toEqual({
      changed: false,
      bytesFreed: 0,
      rewrittenEntries: 0,
      reason: "debounced",
    });

    expect(mocked.runTranscriptReplaceHygiene).toHaveBeenCalledTimes(1);
  });

  it("debounces hybrid compact pre-pass after recent turn-end hygiene", async () => {
    const engine = new HeadroomContextEngine({
      persistentCompaction: "hybrid",
      transcriptHygiene: { debounceMs: 60_000 },
    });
    (engine as { proxyUrl: string | null }).proxyUrl = "http://127.0.0.1:8787";
    mocked.delegateCompactionToRuntime.mockResolvedValueOnce({
      ok: true,
      compacted: true,
      result: { tokensBefore: 20_000, tokensAfter: 8_000 },
    });

    await engine.maintain({
      sessionId: "session-1",
      sessionKey: "agent:main:session-1",
      sessionFile: "session.jsonl",
    });
    await engine.compact({
      sessionId: "session-1",
      sessionKey: "agent:main:session-1",
      sessionFile: "session.jsonl",
      force: true,
    });

    expect(mocked.runTranscriptReplaceHygiene).toHaveBeenCalledTimes(1);
    expect(mocked.delegateCompactionToRuntime).toHaveBeenCalledTimes(1);
  });

  it("skips turn-end hygiene in maintain() when transcript hygiene is disabled", async () => {
    const engine = new HeadroomContextEngine({
      persistentCompaction: "hybrid",
      transcriptHygiene: { enabled: false },
    });
    (engine as { proxyUrl: string | null }).proxyUrl = "http://127.0.0.1:8787";

    await expect(
      engine.maintain({
        sessionId: "session-1",
        sessionKey: "agent:main:session-1",
        sessionFile: "session.jsonl",
      }),
    ).resolves.toEqual({
      changed: false,
      bytesFreed: 0,
      rewrittenEntries: 0,
      reason: "transcript hygiene disabled",
    });

    expect(mocked.runTranscriptReplaceHygiene).not.toHaveBeenCalled();
  });

  it("runs turn-end hygiene in maintain() for hybrid mode", async () => {
    const engine = new HeadroomContextEngine({ persistentCompaction: "hybrid" });
    (engine as { proxyUrl: string | null }).proxyUrl = "http://127.0.0.1:8787";

    await expect(
      engine.maintain({
        sessionId: "session-1",
        sessionKey: "agent:main:session-1",
        sessionFile: "session.jsonl",
      }),
    ).resolves.toEqual({
      changed: true,
      bytesFreed: 512,
      rewrittenEntries: 2,
      tokensBefore: 50_000,
      tokensAfter: 44_000,
    });

    expect(mocked.runTranscriptReplaceHygiene).toHaveBeenCalledWith(
      expect.objectContaining({ force: false }),
    );
  });

  it("delegates persistent compaction to OpenClaw when configured", async () => {
    const engine = new HeadroomContextEngine({ persistentCompaction: "openclaw" });
    const params = {
      sessionId: "session-1",
      sessionKey: "agent:main:session-1",
      sessionFile: "session.jsonl",
      tokenBudget: 120_000,
      force: false,
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
    expect(engine.info.transcriptSemantics?.currentTurnFence).toBe(
      "before-current-turn-entry-v1",
    );
    await expect(engine.compact(params)).resolves.toEqual(delegatedResult);
    expect(mocked.delegateCompactionToRuntime).toHaveBeenCalledWith(params);
    expect(compress).not.toHaveBeenCalled();
    expect(engine.getStats().compactions).toBe(1);
  });

  it("does not count a delegated no-op as a compaction", async () => {
    const engine = new HeadroomContextEngine({ persistentCompaction: "openclaw" });
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
    const engine = new HeadroomContextEngine({ persistentCompaction: "openclaw" });
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

  it("skips proxy compression when context is clearly under token budget", async () => {
    const engine = new HeadroomContextEngine();
    (engine as { proxyUrl: string | null }).proxyUrl = "http://127.0.0.1:8787";
    const messages = [{ role: "user", content: "hello" }];

    await expect(
      engine.assemble({
        sessionId: "session-1",
        messages,
        tokenBudget: 1_000_000,
      }),
    ).resolves.toMatchObject({
      messages,
      estimatedTokens: expect.any(Number),
    });

    expect(compress).not.toHaveBeenCalled();
  });

  it("compresses once history crosses the default skip threshold (below OpenClaw's overflow line)", async () => {
    vi.mocked(compress).mockResolvedValue({
      compressed: false,
      messages: [],
      tokensBefore: 0,
      tokensAfter: 0,
      tokensSaved: 0,
    });
    const engine = new HeadroomContextEngine();
    (engine as { proxyUrl: string | null }).proxyUrl = "http://127.0.0.1:8787";
    // Default threshold = (200k − 20k reserve) × 0.7 = 126k tokens.
    // ~4 chars/token → 520k chars ≈ 130k rough tokens. The old flat 85 % rule
    // (170k) skipped this even though OpenClaw — which also counts the system
    // prompt and its reserve — would already be compacting.
    const messages = [{ role: "user", content: "x".repeat(520_000) }];

    await engine.assemble({ sessionId: "session-1", messages, tokenBudget: 200_000 });

    expect(compress).toHaveBeenCalledTimes(1);
  });

  it("honours configured assembleSkipBudgetRatio / assembleReserveTokens and clamps invalid values", async () => {
    vi.mocked(compress).mockResolvedValue({
      compressed: false,
      messages: [],
      tokensBefore: 0,
      tokensAfter: 0,
      tokensSaved: 0,
    });
    const messages = [{ role: "user", content: "x".repeat(300_000) }]; // ≈75k tokens

    // (100k − 20k) × 0.9 = 72k → 75k compresses.
    const strict = new HeadroomContextEngine({ assembleSkipBudgetRatio: 0.9, assembleReserveTokens: 0 });
    (strict as { proxyUrl: string | null }).proxyUrl = "http://127.0.0.1:8787";
    await strict.assemble({ sessionId: "s", messages, tokenBudget: 100_000 });
    expect(compress).not.toHaveBeenCalled(); // reserve 0 → 90k threshold → skip

    // A large system prompt is modelled via the reserve: (100k − 60k) × 0.7 = 28k → compress.
    const bigSystemPrompt = new HeadroomContextEngine({ assembleReserveTokens: 60_000 });
    (bigSystemPrompt as { proxyUrl: string | null }).proxyUrl = "http://127.0.0.1:8787";
    await bigSystemPrompt.assemble({ sessionId: "s", messages, tokenBudget: 100_000 });
    expect(compress).toHaveBeenCalledTimes(1);

    expect(resolveAssembleSkipThreshold({ tokenBudget: 200_000 })).toBe(126_000);
    expect(resolveAssembleSkipThreshold({ tokenBudget: 100_000, ratio: 0.5, reserveTokens: 0 })).toBe(50_000);
    // Reserve larger than the budget never disables compression entirely.
    expect(resolveAssembleSkipThreshold({ tokenBudget: 10_000, reserveTokens: 50_000 })).toBe(1);

    expect(resolveAssembleSkipBudgetRatio(undefined)).toBe(DEFAULT_ASSEMBLE_SKIP_BUDGET_RATIO);
    expect(resolveAssembleSkipBudgetRatio(0)).toBe(DEFAULT_ASSEMBLE_SKIP_BUDGET_RATIO);
    expect(resolveAssembleSkipBudgetRatio(1.5)).toBe(DEFAULT_ASSEMBLE_SKIP_BUDGET_RATIO);
    expect(resolveAssembleSkipBudgetRatio(Number.NaN)).toBe(DEFAULT_ASSEMBLE_SKIP_BUDGET_RATIO);
    expect(resolveAssembleSkipBudgetRatio(1)).toBe(1);
    expect(resolveAssembleSkipBudgetRatio(0.55)).toBe(0.55);

    expect(resolveAssembleReserveTokens(undefined)).toBe(DEFAULT_ASSEMBLE_RESERVE_TOKENS);
    expect(resolveAssembleReserveTokens(-1)).toBe(DEFAULT_ASSEMBLE_RESERVE_TOKENS);
    expect(resolveAssembleReserveTokens(Number.NaN)).toBe(DEFAULT_ASSEMBLE_RESERVE_TOKENS);
    expect(resolveAssembleReserveTokens(0)).toBe(0);
    expect(resolveAssembleReserveTokens(12_345.9)).toBe(12_345);
  });

  it("passes assembleCompressConfig to compress via headroom-ai SDK", async () => {
    vi.mocked(compress).mockResolvedValue({
      compressed: false,
      messages: [{ role: "user", content: "hello" }],
      tokensBefore: 5000,
      tokensAfter: 5000,
      tokensSaved: 0,
    });

    const engine = new HeadroomContextEngine({
      assembleCompressConfig: { protect_recent: 4, mode: "ccr" },
    });
    (engine as { proxyUrl: string | null }).proxyUrl = "http://127.0.0.1:8787";

    await engine.assemble({
      sessionId: "session-1",
      messages: [{ role: "user", content: "hello ".repeat(500) }],
    });

    expect(compress).toHaveBeenCalledWith(
      expect.any(Array),
      expect.objectContaining({
        config: expect.objectContaining({
          protect_recent: 4,
          mode: "ccr",
        }),
      }),
    );
  });

  it("skips assemble compression when skipAssembleWhenGatewayRouted is enabled", async () => {
    const engine = new HeadroomContextEngine({
      skipAssembleWhenGatewayRouted: true,
      gatewayProviderIds: ["openrouter", "opencode-go"],
    });
    (engine as { proxyUrl: string | null }).proxyUrl = "http://127.0.0.1:8787";

    await expect(
      engine.assemble({
        sessionId: "session-1",
        messages: [{ role: "user", content: "hello ".repeat(500) }],
      }),
    ).resolves.toMatchObject({
      messages: [{ role: "user", content: expect.stringContaining("hello") }],
      estimatedTokens: expect.any(Number),
    });

    expect(compress).not.toHaveBeenCalled();
  });

  it("omits CCR retrieve hint when compression saved tokens but produced no ccrHashes", async () => {
    vi.mocked(compress).mockResolvedValue({
      compressed: true,
      messages: [{ role: "user", content: "compressed hello" }],
      tokensBefore: 5000,
      tokensAfter: 2000,
      tokensSaved: 3000,
      ccrHashes: [],
    });

    const engine = new HeadroomContextEngine();
    (engine as { proxyUrl: string | null }).proxyUrl = "http://127.0.0.1:8787";

    await expect(
      engine.assemble({
        sessionId: "session-1",
        messages: [{ role: "user", content: "hello ".repeat(500) }],
      }),
    ).resolves.toMatchObject({
      estimatedTokens: 2000,
      systemPromptAddition: undefined,
    });
  });

  it("includes CCR retrieve hint only when compression saved tokens and ccrHashes exist", async () => {
    vi.mocked(compress).mockResolvedValue({
      compressed: true,
      messages: [{ role: "user", content: "compressed hello" }],
      tokensBefore: 5000,
      tokensAfter: 2000,
      tokensSaved: 3000,
      ccrHashes: ["abc123"],
    });

    const engine = new HeadroomContextEngine();
    (engine as { proxyUrl: string | null }).proxyUrl = "http://127.0.0.1:8787";

    await expect(
      engine.assemble({
        sessionId: "session-1",
        messages: [{ role: "user", content: "hello ".repeat(500) }],
      }),
    ).resolves.toMatchObject({
      estimatedTokens: 2000,
      systemPromptAddition: expect.stringContaining("headroom_retrieve"),
    });
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

  describe("commitTurn durable advancement", () => {
    const tempDirs: string[] = [];

    afterEach(() => {
      for (const dir of tempDirs.splice(0)) {
        rmSync(dir, { recursive: true, force: true });
      }
    });

    function makeEngine() {
      const dir = mkdtempSync(join(tmpdir(), "headroom-engine-commit-"));
      tempDirs.push(dir);
      return new HeadroomContextEngine({
        turnAdvancementStorePath: join(dir, "turn-advancements.json"),
      });
    }

    const commitParams = {
      sessionId: "session-1",
      advancementKey: "turn-1",
      messages: [{ role: "user", content: "hello" }],
    };

    it("returns committed on first write and duplicate on retry", async () => {
      const engine = makeEngine();

      await expect(engine.commitTurn(commitParams)).resolves.toEqual({
        status: "committed",
      });
      await expect(engine.commitTurn(commitParams)).resolves.toEqual({
        status: "duplicate",
      });
    });

    it("persists across new engine instances after restart", async () => {
      const dir = mkdtempSync(join(tmpdir(), "headroom-engine-restart-"));
      tempDirs.push(dir);
      const storePath = join(dir, "turn-advancements.json");

      const first = new HeadroomContextEngine({ turnAdvancementStorePath: storePath });
      await expect(first.commitTurn(commitParams)).resolves.toEqual({
        status: "committed",
      });

      const second = new HeadroomContextEngine({ turnAdvancementStorePath: storePath });
      await expect(second.commitTurn(commitParams)).resolves.toEqual({
        status: "duplicate",
      });
    });
  });
});
