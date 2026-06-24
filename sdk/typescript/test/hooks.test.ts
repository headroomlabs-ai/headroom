/**
 * Tests for compression hooks — matches Python test_hooks.py patterns.
 */
import { describe, it, expect } from "vitest";
import {
  CompressionHooks,
  extractUserQuery,
  countTurns,
  extractToolCalls,
} from "../src/hooks.js";
import type { CompressContext, CompressEvent } from "../src/hooks.js";

describe("CompressionHooks defaults", () => {
  it("preCompress returns messages unchanged", () => {
    const hooks = new CompressionHooks();
    const messages = [{ role: "user", content: "hello" }];
    const ctx: CompressContext = {
      model: "",
      userQuery: "",
      turnNumber: 0,
      toolCalls: [],
      provider: "",
    };
    const result = hooks.preCompress(messages, ctx);
    expect(result).toBe(messages);
  });

  it("computeBiases returns empty object", () => {
    const hooks = new CompressionHooks();
    const messages = [{ role: "user", content: "hello" }];
    const ctx: CompressContext = {
      model: "",
      userQuery: "",
      turnNumber: 0,
      toolCalls: [],
      provider: "",
    };
    const result = hooks.computeBiases(messages, ctx);
    expect(result).toEqual({});
  });

  it("postCompress is a no-op", () => {
    const hooks = new CompressionHooks();
    const event: CompressEvent = {
      tokensBefore: 100,
      tokensAfter: 50,
      tokensSaved: 50,
      compressionRatio: 0.5,
      transformsApplied: [],
      ccrHashes: [],
      model: "",
      userQuery: "",
      provider: "",
    };
    // Should not throw
    expect(() => hooks.postCompress(event)).not.toThrow();
  });
});

describe("Custom hooks", () => {
  it("preCompress can modify messages", () => {
    class PrependHooks extends CompressionHooks {
      preCompress(messages: any[], _ctx: CompressContext) {
        return [
          { role: "system", content: "Custom prefix" },
          ...messages,
        ];
      }
    }

    const hooks = new PrependHooks();
    const messages = [{ role: "user", content: "hello" }];
    const ctx: CompressContext = {
      model: "",
      userQuery: "",
      turnNumber: 0,
      toolCalls: [],
      provider: "",
    };
    const result = hooks.preCompress(messages, ctx);
    expect(result).toHaveLength(2);
    expect(result[0].role).toBe("system");
  });

  it("computeBiases can set position-aware biases", () => {
    class BiasHooks extends CompressionHooks {
      computeBiases(messages: any[], _ctx: CompressContext) {
        const biases: Record<number, number> = {};
        biases[0] = 2.0; // preserve first message
        biases[messages.length - 1] = 1.5; // preserve last
        return biases;
      }
    }

    const hooks = new BiasHooks();
    const messages = [
      { role: "user", content: "first" },
      { role: "assistant", content: "middle" },
      { role: "user", content: "last" },
    ];
    const ctx: CompressContext = {
      model: "",
      userQuery: "",
      turnNumber: 0,
      toolCalls: [],
      provider: "",
    };
    const biases = hooks.computeBiases(messages, ctx);
    expect(biases[0]).toBe(2.0);
    expect(biases[2]).toBe(1.5);
  });

  it("postCompress receives event data", () => {
    let captured: CompressEvent | null = null;

    class LogHooks extends CompressionHooks {
      postCompress(event: CompressEvent) {
        captured = event;
      }
    }

    const hooks = new LogHooks();
    const event: CompressEvent = {
      tokensBefore: 1000,
      tokensAfter: 200,
      tokensSaved: 800,
      compressionRatio: 0.2,
      transformsApplied: ["smart_crusher"],
      ccrHashes: ["abc123"],
      model: "gpt-4o",
      userQuery: "test query",
      provider: "openai",
    };

    hooks.postCompress(event);
    expect(captured).toEqual(event);
    expect(captured!.tokensSaved).toBe(800);
  });

  it("hooks receive correct context", () => {
    let capturedCtx: CompressContext | null = null;

    class CtxHooks extends CompressionHooks {
      preCompress(messages: any[], ctx: CompressContext) {
        capturedCtx = ctx;
        return messages;
      }
    }

    const hooks = new CtxHooks();
    const ctx: CompressContext = {
      model: "gpt-4o",
      userQuery: "test query",
      turnNumber: 3,
      toolCalls: ["tool_a", "tool_b"],
      provider: "openai",
    };
    hooks.preCompress([], ctx);
    expect(capturedCtx!.model).toBe("gpt-4o");
    expect(capturedCtx!.turnNumber).toBe(3);
    expect(capturedCtx!.toolCalls).toEqual(["tool_a", "tool_b"]);
  });
});

describe("CompressEvent fields", () => {
  it("has all expected fields", () => {
    const event: CompressEvent = {
      tokensBefore: 0,
      tokensAfter: 0,
      tokensSaved: 0,
      compressionRatio: 0,
      transformsApplied: [],
      ccrHashes: [],
      model: "",
      userQuery: "",
      provider: "",
    };
    expect(event.tokensBefore).toBe(0);
    expect(event.transformsApplied).toEqual([]);
    expect(event.ccrHashes).toEqual([]);
  });
});

describe("CompressContext fields", () => {
  it("has all expected fields with defaults", () => {
    const ctx: CompressContext = {
      model: "",
      userQuery: "",
      turnNumber: 0,
      toolCalls: [],
      provider: "",
    };
    expect(ctx.model).toBe("");
    expect(ctx.turnNumber).toBe(0);
    expect(ctx.toolCalls).toEqual([]);
  });
});

describe("extractUserQuery", () => {
  it("extracts last user message", () => {
    const messages = [
      { role: "user", content: "first" },
      { role: "assistant", content: "response" },
      { role: "user", content: "second" },
    ];
    expect(extractUserQuery(messages)).toBe("second");
  });

  it("returns empty for no user messages", () => {
    expect(extractUserQuery([{ role: "system", content: "sys" }])).toBe("");
  });

  it("handles array content (text part)", () => {
    const messages = [
      { role: "user", content: [{ type: "text", text: "from part" }] },
    ];
    expect(extractUserQuery(messages)).toBe("from part");
  });
});

describe("countTurns", () => {
  it("counts user messages", () => {
    const messages = [
      { role: "user", content: "1" },
      { role: "assistant", content: "r" },
      { role: "user", content: "2" },
    ];
    expect(countTurns(messages)).toBe(2);
  });
});

describe("extractToolCalls", () => {
  it("extracts tool names from OpenAI format", () => {
    const messages = [
      {
        role: "assistant",
        content: null,
        tool_calls: [
          { id: "1", type: "function", function: { name: "search", arguments: "{}" } },
        ],
      },
    ];
    expect(extractToolCalls(messages)).toEqual(["search"]);
  });

  it("extracts from Anthropic format", () => {
    const messages = [
      {
        role: "assistant",
        content: [{ type: "tool_use", name: "lookup" }],
      },
    ];
    expect(extractToolCalls(messages)).toEqual(["lookup"]);
  });

  it("extracts from Vercel format", () => {
    const messages = [
      {
        role: "assistant",
        content: [{ type: "tool-call", toolName: "fetch" }],
      },
    ];
    expect(extractToolCalls(messages)).toEqual(["fetch"]);
  });
});
