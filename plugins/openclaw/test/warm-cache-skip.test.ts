import { afterEach, describe, expect, it, vi } from "vitest";

const mocked = vi.hoisted(() => ({
  compress: vi.fn(),
  start: vi.fn(async () => "http://127.0.0.1:8787"),
  stop: vi.fn(async () => undefined),
  logger: { debug: vi.fn(), error: vi.fn(), info: vi.fn(), warn: vi.fn() },
}));

vi.mock("headroom-ai", () => ({ compress: mocked.compress }));

vi.mock("../src/proxy-manager.js", () => ({
  ProxyManager: class {
    start = mocked.start;
    stop = mocked.stop;
  },
  defaultLogger: mocked.logger,
}));

import { normalizeAgentMessages } from "../src/convert.js";
import { HeadroomContextEngine } from "../src/engine.js";

afterEach(() => {
  mocked.compress.mockReset();
});

const NOTICE =
  "[Headroom is compressing tool outputs in this session. Use headroom_retrieve if you need the original, uncompressed content.]";

// A history whose newest message is `ageMs` old: a user turn, a tool call, and a large tool result that came back
// `toolMs` after the call.
function history(ageMs: number, toolMs = 1000) {
  const last = Date.now() - ageMs;
  return [
    { role: "user", content: [{ type: "text", text: "list the pods" }], timestamp: last - toolMs - 1000 },
    {
      role: "assistant",
      content: [{ type: "toolCall", id: "t1", name: "exec", arguments: { command: "kubectl get pods" } }],
      api: "bedrock-converse-stream",
      provider: "amazon-bedrock",
      model: "claude",
      stopReason: "toolUse",
      timestamp: last - toolMs,
    },
    { role: "toolResult", toolCallId: "t1", toolName: "exec", content: [{ type: "text", text: "x".repeat(4000) }], isError: false, timestamp: last },
  ];
}

// The proxy compresses the tool result and reports the sizes it saw.
function compressing(tokensBefore: number) {
  return async (messages: any[]) => ({
    compressed: true,
    tokensBefore,
    tokensAfter: tokensBefore - 800,
    tokensSaved: 800,
    messages: messages.map((m) => (m.role === "tool" ? { ...m, content: "[compressed]" } : m)),
  });
}

async function nothingToCompress(messages: any[]) {
  return { compressed: false, tokensBefore: 1000, tokensAfter: 1000, tokensSaved: 0, messages };
}

function engine(config: Record<string, unknown>) {
  const e = new HeadroomContextEngine(config as any);
  (e as { proxyUrl: string | null }).proxyUrl = "http://127.0.0.1:8787";
  return e;
}

describe("skipCompressionWhenCacheWarmMs", () => {
  it("returns the history unchanged while the cache is warm", async () => {
    mocked.compress.mockImplementation(compressing(1000));
    const input = history(60_000);

    const result = await engine({ skipCompressionWhenCacheWarmMs: 300_000 }).assemble({ sessionId: "s", messages: input });

    expect(result.messages).toEqual(normalizeAgentMessages(input));
    // The proxy is not called at all, so it never counts savings that are not applied.
    expect(mocked.compress).not.toHaveBeenCalled();
  });

  it("asks the proxy when a warm history may be near the budget, and keeps it unchanged if it fits", async () => {
    mocked.compress.mockImplementation(compressing(1000));
    const input = history(60_000); // ~4.5KB, so it can't be shown to fit 1200 tokens without the proxy's count

    const result = await engine({ skipCompressionWhenCacheWarmMs: 300_000 }).assemble({ sessionId: "s", messages: input, tokenBudget: 1200 });

    expect(mocked.compress).toHaveBeenCalledTimes(1);
    expect(result.messages).toEqual(normalizeAgentMessages(input)); // the proxy says 1000 tokens: fits 1200
  });

  it("applies the compression once the cache has gone cold", async () => {
    mocked.compress.mockImplementation(compressing(1000));

    const result = await engine({ skipCompressionWhenCacheWarmMs: 300_000 }).assemble({ sessionId: "s", messages: history(600_000) });

    expect(result.messages[2].content).toEqual([{ type: "text", text: "[compressed]" }]);
  });

  it("still compresses a warm history that has to shrink to fit the token budget", async () => {
    mocked.compress.mockImplementation(compressing(1000));

    const result = await engine({ skipCompressionWhenCacheWarmMs: 300_000 }).assemble({
      sessionId: "s",
      messages: history(60_000),
      tokenBudget: 500,
    });

    expect(result.messages[2].content).toEqual([{ type: "text", text: "[compressed]" }]);
  });

  it("changes nothing when the option is unset", async () => {
    mocked.compress.mockImplementation(compressing(1000));

    const result = await engine({}).assemble({ sessionId: "s", messages: history(60_000) });

    expect(result.messages[2].content).toEqual([{ type: "text", text: "[compressed]" }]);
  });

  it("asks the proxy when a warm history is short in characters but could still exceed the budget", async () => {
    // 3,000 CJK characters: under 1.2K tokens at 3 characters per token, but each is 3 bytes of UTF-8, and a BPE
    // tokenizer can spend a token on each.
    mocked.compress.mockImplementation(compressing(3000));
    const input = history(60_000);
    input[2] = { ...input[2], content: [{ type: "text", text: "漢".repeat(3000) }] };

    const result = await engine({ skipCompressionWhenCacheWarmMs: 300_000 }).assemble({ sessionId: "s", messages: input, tokenBudget: 1400 });

    expect(mocked.compress).toHaveBeenCalledTimes(1);
    expect(result.messages[2].content).toEqual([{ type: "text", text: "[compressed]" }]); // 3000 tokens do not fit 1400
  });

  it("treats the cache as cold once the last response is older than the window, however new the messages after it", async () => {
    mocked.compress.mockImplementation(compressing(1000));
    // The tool ran for 10 minutes, so the provider last read this prefix 10 minutes ago, when it returned the call.
    const input = history(0, 600_000);

    const result = await engine({ skipCompressionWhenCacheWarmMs: 300_000 }).assemble({ sessionId: "s", messages: input });

    expect(result.messages[2].content).toEqual([{ type: "text", text: "[compressed]" }]);
  });

  it("keeps the compression notice on warm turns once the session has earned it", async () => {
    mocked.compress.mockImplementation(compressing(1000));
    const e = engine({ skipCompressionWhenCacheWarmMs: 300_000 });
    const cold = await e.assemble({ sessionId: "s", messages: history(600_000) });
    expect(cold.systemPromptAddition).toBe(NOTICE);

    // The notice is part of the cached system prompt, so a warm turn sends it too: without asking the proxy ...
    const warm = await e.assemble({ sessionId: "s", messages: history(60_000) });
    // ... and after asking it and skipping its compression.
    const nearBudget = await e.assemble({ sessionId: "s", messages: history(60_000), tokenBudget: 1200 });

    expect(warm.systemPromptAddition).toBe(NOTICE);
    expect(nearBudget.systemPromptAddition).toBe(NOTICE);
  });

  it("does not announce a compression that a warm turn skipped", async () => {
    mocked.compress.mockImplementation(compressing(1000)); // saves 800 tokens
    const e = engine({ skipCompressionWhenCacheWarmMs: 300_000 });
    // Near the budget the proxy is asked, but its compression is not applied while the cache is warm ...
    const warm = await e.assemble({ sessionId: "s", messages: history(60_000), tokenBudget: 1200 });

    // ... so a later turn has nothing to announce either.
    mocked.compress.mockImplementation(nothingToCompress);
    const cold = await e.assemble({ sessionId: "s", messages: history(600_000) });

    expect(warm.systemPromptAddition).toBeUndefined();
    expect(cold.systemPromptAddition).toBeUndefined();
  });
});
