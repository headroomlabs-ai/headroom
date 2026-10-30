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

const MIN = 60_000;
const COMPRESSED = [{ type: "text", text: "[compressed]" }];

// One turn: a user message, a tool call, and a large tool result that is `ageMs` old.
function turn(ageMs: number, id: string) {
  const last = Date.now() - ageMs;
  return [
    { role: "user", content: [{ type: "text", text: `question ${id}` }], timestamp: last - 2000 },
    {
      role: "assistant",
      content: [{ type: "toolCall", id, name: "exec", arguments: { command: `kubectl get pods ${id}` } }],
      api: "bedrock-converse-stream",
      provider: "amazon-bedrock",
      model: "claude",
      stopReason: "toolUse",
      timestamp: last - 1000,
    },
    { role: "toolResult", toolCallId: id, toolName: "exec", content: [{ type: "text", text: `${id} ` + "x".repeat(4000) }], isError: false, timestamp: last },
  ];
}

// The proxy compresses every tool result it is sent.
async function compressing(messages: any[]) {
  return {
    compressed: true,
    tokensBefore: 2000,
    tokensAfter: 1200,
    tokensSaved: 800,
    messages: messages.map((m) => (m.role === "tool" ? { ...m, content: "[compressed]" } : m)),
  };
}

async function nothingToCompress(messages: any[]) {
  return { compressed: false, tokensBefore: 2000, tokensAfter: 2000, tokensSaved: 0, messages };
}

// The memo only exists with the warm-cache window on, so every engine here sets it (Anthropic/Bedrock's TTL).
function engine(config: Record<string, unknown> = {}) {
  const e = new HeadroomContextEngine({ skipCompressionWhenCacheWarmMs: 5 * MIN, ...config } as any);
  (e as { proxyUrl: string | null }).proxyUrl = "http://127.0.0.1:8787";
  return e;
}

const bytes = (messages: any[]) => JSON.stringify(messages);

describe("the compressed history stays put while the prompt cache is warm", () => {
  it("keeps no compressed view when skipCompressionWhenCacheWarmMs is unset", async () => {
    mocked.compress.mockImplementation(compressing);
    const e = engine({ skipCompressionWhenCacheWarmMs: undefined });
    const t1 = turn(10 * MIN, "t1");
    await e.assemble({ sessionId: "s", messages: t1 });

    const h2 = [...t1, ...turn(60_000, "t2")];
    await e.assemble({ sessionId: "s", messages: h2 });
    // Off means stock behaviour: every assembly asks the proxy, nothing is replayed from a memo.
    expect(mocked.compress).toHaveBeenCalledTimes(2);
    expect((e as unknown as { compressedViews: Map<string, unknown> }).compressedViews.size).toBe(0);
  });

  it("sends the same compressed messages on the warm turns after a cold turn compressed them", async () => {
    mocked.compress.mockImplementation(compressing);
    const e = engine();
    const t1 = turn(10 * MIN, "t1");

    const cold = await e.assemble({ sessionId: "s", messages: t1 });
    expect(cold.messages[2].content).toEqual(COMPRESSED);

    const h2 = [...t1, ...turn(60_000, "t2")];
    const warm = await e.assemble({ sessionId: "s", messages: h2 });
    const h3 = [...h2, ...turn(10_000, "t3")];
    const warmer = await e.assemble({ sessionId: "s", messages: h3 });

    // The provider cached the compressed messages, so both warm turns start with exactly those bytes.
    expect(bytes(warm.messages.slice(0, 3))).toBe(bytes(cold.messages));
    expect(bytes(warmer.messages.slice(0, 3))).toBe(bytes(cold.messages));
    // Newer messages go out as they came in, and nothing asked the proxy again.
    expect(warm.messages.slice(3)).toEqual(normalizeAgentMessages(h2.slice(3)));
    expect(warmer.messages.slice(3)).toEqual(normalizeAgentMessages(h3.slice(3)));
    expect(mocked.compress).toHaveBeenCalledTimes(1);
  });

  it("passes the history through when it changed under the compressed messages", async () => {
    mocked.compress.mockImplementation(compressing);
    const e = engine();
    const t1 = turn(10 * MIN, "t1");
    await e.assemble({ sessionId: "s", messages: t1 });

    const rewritten = [{ ...t1[0], content: [{ type: "text", text: "summary of the earlier turns" }] }, ...t1.slice(1), ...turn(60_000, "t2")];
    const result = await e.assemble({ sessionId: "s", messages: rewritten });

    expect(result.messages).toEqual(normalizeAgentMessages(rewritten));
  });

  it("recognises the compressed messages when OpenClaw passes them back itself, and keeps them", async () => {
    mocked.compress.mockImplementation(compressing);
    const e = engine();
    const t1 = turn(10 * MIN, "t1");
    const cold = await e.assemble({ sessionId: "s", messages: t1 });
    const t2 = turn(60_000, "t2");

    const passedBack = await e.assemble({ sessionId: "s", messages: [...structuredClone(cold.messages), ...t2] });
    expect(bytes(passedBack.messages.slice(0, 3))).toBe(bytes(cold.messages));
    expect(passedBack.messages.slice(3)).toEqual(normalizeAgentMessages(t2));

    // A later turn rebuilt from the transcript (the originals) still gets the compressed messages.
    const rebuilt = await e.assemble({ sessionId: "s", messages: [...t1, ...t2, ...turn(10_000, "t3")] });
    expect(bytes(rebuilt.messages.slice(0, 3))).toBe(bytes(cold.messages));
  });

  it("compresses afresh, newer messages included, once the cache has gone cold", async () => {
    mocked.compress.mockImplementation(compressing);
    const e = engine();
    const t1 = turn(20 * MIN, "t1");
    await e.assemble({ sessionId: "s", messages: t1 });

    const result = await e.assemble({ sessionId: "s", messages: [...t1, ...turn(10 * MIN, "t2")] });

    expect(mocked.compress).toHaveBeenCalledTimes(2);
    expect(result.messages[2].content).toEqual(COMPRESSED);
    expect(result.messages[5].content).toEqual(COMPRESSED);
  });

  it("does not bring an old view back after a cold turn that sent the originals", async () => {
    mocked.compress.mockImplementation(compressing);
    const e = engine();
    const t1 = turn(20 * MIN, "t1");
    await e.assemble({ sessionId: "s", messages: t1 });

    // The cache went cold and this time the proxy found nothing to compress, so the originals went out.
    mocked.compress.mockImplementation(nothingToCompress);
    const h2 = [...t1, ...turn(10 * MIN, "t2")];
    expect((await e.assemble({ sessionId: "s", messages: h2 })).messages).toEqual(normalizeAgentMessages(h2));

    const h3 = [...h2, ...turn(60_000, "t3")];
    expect((await e.assemble({ sessionId: "s", messages: h3 })).messages).toEqual(normalizeAgentMessages(h3));
  });

  it("keeps the compressed messages while the circuit breaker is open", async () => {
    mocked.compress.mockImplementation(compressing);
    const e = engine();
    const t1 = turn(10 * MIN, "t1");
    const cold = await e.assemble({ sessionId: "s", messages: t1 });

    (e as any).circuit = { errors: 3, openUntilMs: Date.now() + MIN };
    const result = await e.assemble({ sessionId: "s", messages: [...t1, ...turn(60_000, "t2")] });

    expect(bytes(result.messages.slice(0, 3))).toBe(bytes(cold.messages));
  });

  it("replaces the compressed messages when a warm history has to shrink to fit the budget", async () => {
    mocked.compress.mockImplementation(compressing); // reports 2000 tokens before
    const e = engine();
    const t1 = turn(10 * MIN, "t1");
    await e.assemble({ sessionId: "s", messages: t1 });

    const h2 = [...t1, ...turn(60_000, "t2")];
    const shrunk = await e.assemble({ sessionId: "s", messages: h2, tokenBudget: 1500 });
    expect(mocked.compress).toHaveBeenCalledTimes(2);
    expect(shrunk.messages[5].content).toEqual(COMPRESSED);

    const next = await e.assemble({ sessionId: "s", messages: [...h2, ...turn(10_000, "t3")] });
    expect(bytes(next.messages.slice(0, 6))).toBe(bytes(shrunk.messages));
  });

  it("keeps the compressed messages when a warm history near the budget still fits", async () => {
    mocked.compress.mockImplementation(compressing); // reports 2000 tokens before
    const e = engine();
    const t1 = turn(10 * MIN, "t1");
    const cold = await e.assemble({ sessionId: "s", messages: t1 });

    const result = await e.assemble({ sessionId: "s", messages: [...t1, ...turn(60_000, "t2")], tokenBudget: 2500 });

    expect(mocked.compress).toHaveBeenCalledTimes(2); // near the budget, so the proxy was asked
    expect(bytes(result.messages.slice(0, 3))).toBe(bytes(cold.messages));
    expect(result.messages[5].content).not.toEqual(COMPRESSED);
  });

  it("is not changed by what the caller does with the messages it gets back", async () => {
    mocked.compress.mockImplementation(compressing);
    const e = engine();
    const t1 = turn(10 * MIN, "t1");
    const cold = await e.assemble({ sessionId: "s", messages: t1 });

    // OpenClaw keeps the returned list as the turn's working state and appends to it.
    cold.messages.push({ role: "user", content: [{ type: "text", text: "appended" }], timestamp: Date.now() });
    cold.messages[2].content[0].text = "mutated";
    const result = await e.assemble({ sessionId: "s", messages: [...t1, ...turn(60_000, "t2")] });

    expect(result.messages).toHaveLength(6);
    expect(result.messages[2].content).toEqual(COMPRESSED);
  });

  it("keeps views per session", async () => {
    mocked.compress.mockImplementation(compressing);
    const e = engine();
    const t1 = turn(10 * MIN, "t1");
    await e.assemble({ sessionId: "a", messages: t1 });

    const h2 = [...t1, ...turn(60_000, "t2")];
    expect((await e.assemble({ sessionId: "b", messages: h2 })).messages).toEqual(normalizeAgentMessages(h2));
  });

  it("keeps nothing when the option is 0, compressing every assembly as before", async () => {
    mocked.compress.mockImplementation(compressing);
    const e = engine({ skipCompressionWhenCacheWarmMs: 0 });
    const t1 = turn(10 * MIN, "t1");
    await e.assemble({ sessionId: "s", messages: t1 });

    const result = await e.assemble({ sessionId: "s", messages: [...t1, ...turn(60_000, "t2")] });

    expect(mocked.compress).toHaveBeenCalledTimes(2);
    expect(result.messages[5].content).toEqual(COMPRESSED);
  });

  it("bounds a warm history by the proxy's last count plus what was appended, without asking it again", async () => {
    mocked.compress.mockImplementation(compressing); // counts 2000 tokens
    const e = engine();
    const t1 = turn(10 * MIN, "t1");
    await e.assemble({ sessionId: "s", messages: t1 });

    // ~9KB in all, too big to fit 7,500 tokens by size alone, but the proxy counted the first ~4.5KB as 2,000 tokens.
    await e.assemble({ sessionId: "s", messages: [...t1, ...turn(60_000, "t2")], tokenBudget: 7500 });

    expect(mocked.compress).toHaveBeenCalledTimes(1);
  });

  it("does not bound a warm history by a count the proxy made for another model", async () => {
    mocked.compress.mockImplementation(compressing); // counts 2000 tokens
    const e = engine();
    const t1 = turn(10 * MIN, "t1");
    await e.assemble({ sessionId: "s", messages: t1, model: "model-a" });

    // The proxy's tokenizer depends on the model, so model-a's count says nothing about model-b's.
    await e.assemble({ sessionId: "s", messages: [...t1, ...turn(60_000, "t2")], tokenBudget: 7500, model: "model-b" });

    expect(mocked.compress).toHaveBeenCalledTimes(2);
  });

  it("asks the proxy again when the history changed under its last count", async () => {
    mocked.compress.mockImplementation(compressing);
    const e = engine();
    const t1 = turn(10 * MIN, "t1");
    await e.assemble({ sessionId: "s", messages: t1 });

    const rewritten = [{ ...t1[0], content: [{ type: "text", text: "summary of the earlier turns" }] }, ...t1.slice(1), ...turn(60_000, "t2")];
    await e.assemble({ sessionId: "s", messages: rewritten, tokenBudget: 7500 });

    expect(mocked.compress).toHaveBeenCalledTimes(2);
  });

  it("drops an idle session's compressed view on any later assembly", async () => {
    mocked.compress.mockImplementation(compressing);
    const e = engine();
    await e.assemble({ sessionId: "idle", messages: turn(10 * MIN, "t1") });
    const views = (e as unknown as { compressedViews: Map<string, { usedAt: number }> }).compressedViews;
    views.get("idle")!.usedAt -= 6 * MIN; // unused for longer than the window

    // A warm turn in another session, which never reaches the proxy, still clears it out.
    await e.assemble({ sessionId: "busy", messages: turn(60_000, "t2") });

    expect(mocked.compress).toHaveBeenCalledTimes(1);
    expect(views.has("idle")).toBe(false);
  });
});
