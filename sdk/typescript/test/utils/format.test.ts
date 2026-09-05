import { describe, it, expect } from "vitest";
import { vercelToOpenAI, openAIToVercel, anthropicToOpenAI } from "../../src/utils/format.js";
import type { OpenAIMessage } from "../../src/types.js";

describe("vercelToOpenAI", () => {
  it("converts system message (passthrough)", () => {
    const result = vercelToOpenAI([
      { role: "system", content: "You are helpful" },
    ]);
    expect(result).toEqual([{ role: "system", content: "You are helpful" }]);
  });

  it("converts user text-only message to flat string", () => {
    const result = vercelToOpenAI([
      {
        role: "user",
        content: [{ type: "text", text: "hello" }],
      },
    ]);
    expect(result).toEqual([{ role: "user", content: "hello" }]);
  });

  it("converts user message with multiple text parts", () => {
    const result = vercelToOpenAI([
      {
        role: "user",
        content: [
          { type: "text", text: "hello " },
          { type: "text", text: "world" },
        ],
      },
    ]);
    expect(result).toEqual([{ role: "user", content: "hello world" }]);
  });

  it("converts user message with image to content parts", () => {
    const result = vercelToOpenAI([
      {
        role: "user",
        content: [
          { type: "text", text: "describe this" },
          {
            type: "image",
            image: new URL("https://example.com/img.png"),
          },
        ],
      },
    ]);
    expect(result).toEqual([
      {
        role: "user",
        content: [
          { type: "text", text: "describe this" },
          {
            type: "image_url",
            image_url: { url: "https://example.com/img.png" },
          },
        ],
      },
    ]);
  });

  it("converts assistant text-only message", () => {
    const result = vercelToOpenAI([
      {
        role: "assistant",
        content: [{ type: "text", text: "Here is the answer" }],
      },
    ]);
    expect(result).toEqual([
      { role: "assistant", content: "Here is the answer" },
    ]);
  });

  it("converts assistant message with tool calls", () => {
    const result = vercelToOpenAI([
      {
        role: "assistant",
        content: [
          { type: "text", text: "Let me search" },
          {
            type: "tool-call",
            toolCallId: "tc_1",
            toolName: "search",
            args: { query: "test" },
          },
        ],
      },
    ]);
    expect(result).toEqual([
      {
        role: "assistant",
        content: "Let me search",
        tool_calls: [
          {
            id: "tc_1",
            type: "function",
            function: {
              name: "search",
              arguments: '{"query":"test"}',
            },
          },
        ],
      },
    ]);
  });

  it("converts assistant with only tool calls (no text)", () => {
    const result = vercelToOpenAI([
      {
        role: "assistant",
        content: [
          {
            type: "tool-call",
            toolCallId: "tc_1",
            toolName: "search",
            args: {},
          },
        ],
      },
    ]);
    expect(result).toEqual([
      {
        role: "assistant",
        content: null,
        tool_calls: [
          {
            id: "tc_1",
            type: "function",
            function: { name: "search", arguments: "{}" },
          },
        ],
      },
    ]);
  });

  it("converts tool result message", () => {
    const result = vercelToOpenAI([
      {
        role: "tool",
        content: [
          {
            type: "tool-result",
            toolCallId: "tc_1",
            toolName: "search",
            result: { data: [1, 2, 3] },
          },
        ],
      },
    ]);
    expect(result).toEqual([
      {
        role: "tool",
        content: '{"data":[1,2,3]}',
        tool_call_id: "tc_1",
      },
    ]);
  });

  it("converts tool result with string result", () => {
    const result = vercelToOpenAI([
      {
        role: "tool",
        content: [
          {
            type: "tool-result",
            toolCallId: "tc_1",
            toolName: "echo",
            result: "hello world",
          },
        ],
      },
    ]);
    expect(result).toEqual([
      {
        role: "tool",
        content: "hello world",
        tool_call_id: "tc_1",
      },
    ]);
  });

  it("handles multiple tool results in one tool message", () => {
    const result = vercelToOpenAI([
      {
        role: "tool",
        content: [
          {
            type: "tool-result",
            toolCallId: "tc_1",
            toolName: "a",
            result: "result_a",
          },
          {
            type: "tool-result",
            toolCallId: "tc_2",
            toolName: "b",
            result: "result_b",
          },
        ],
      },
    ]);
    expect(result).toHaveLength(2);
    expect(result[0]).toEqual({
      role: "tool",
      content: "result_a",
      tool_call_id: "tc_1",
    });
    expect(result[1]).toEqual({
      role: "tool",
      content: "result_b",
      tool_call_id: "tc_2",
    });
  });

  it("skips reasoning parts in assistant messages", () => {
    const result = vercelToOpenAI([
      {
        role: "assistant",
        content: [
          { type: "reasoning", text: "thinking..." },
          { type: "text", text: "answer" },
        ],
      },
    ]);
    expect(result).toEqual([{ role: "assistant", content: "answer" }]);
  });

  it("handles full multi-turn conversation", () => {
    const result = vercelToOpenAI([
      { role: "system", content: "Be helpful" },
      { role: "user", content: [{ type: "text", text: "Hi" }] },
      {
        role: "assistant",
        content: [
          { type: "text", text: "Searching..." },
          {
            type: "tool-call",
            toolCallId: "tc_1",
            toolName: "web_search",
            args: { q: "test" },
          },
        ],
      },
      {
        role: "tool",
        content: [
          {
            type: "tool-result",
            toolCallId: "tc_1",
            toolName: "web_search",
            result: { results: ["a", "b"] },
          },
        ],
      },
      {
        role: "assistant",
        content: [{ type: "text", text: "Found results" }],
      },
    ]);

    expect(result).toHaveLength(5);
    expect(result[0].role).toBe("system");
    expect(result[1].role).toBe("user");
    expect(result[2].role).toBe("assistant");
    expect(result[2].tool_calls).toHaveLength(1);
    expect(result[3].role).toBe("tool");
    expect(result[4].role).toBe("assistant");
  });
});

describe("anthropicToOpenAI", () => {
  it("converts image-only user message to a content-parts array (not dropped)", () => {
    const result = anthropicToOpenAI([
      {
        role: "user",
        content: [
          {
            type: "image",
            source: { type: "base64", media_type: "image/png", data: "AAAA" },
          },
        ],
      },
    ]);
    expect(result).toHaveLength(1);
    expect(result[0]).toEqual({
      role: "user",
      content: [
        {
          type: "image_url",
          image_url: { url: "data:image/png;base64,AAAA" },
        },
      ],
    });
  });

  it("converts a url-sourced image as a passthrough url", () => {
    const result = anthropicToOpenAI([
      {
        role: "user",
        content: [
          { type: "image", source: { type: "url", url: "https://example.com/cat.png" } },
        ],
      },
    ]);
    expect(result[0].content).toEqual([
      { type: "image_url", image_url: { url: "https://example.com/cat.png" } },
    ]);
  });

  it("preserves order of text + image blocks", () => {
    const result = anthropicToOpenAI([
      {
        role: "user",
        content: [
          { type: "text", text: "what is this?" },
          {
            type: "image",
            source: { type: "base64", media_type: "image/jpeg", data: "BBBB" },
          },
        ],
      },
    ]);
    expect(result[0].content).toEqual([
      { type: "text", text: "what is this?" },
      { type: "image_url", image_url: { url: "data:image/jpeg;base64,BBBB" } },
    ]);
  });

  it("leaves text-only user messages as a flat string (backward compat)", () => {
    const result = anthropicToOpenAI([
      { role: "user", content: [{ type: "text", text: "hello" }] },
    ]);
    expect(result).toEqual([{ role: "user", content: "hello" }]);
  });

  it("emits a separate tool message alongside an image in the same user turn", () => {
    const result = anthropicToOpenAI([
      {
        role: "user",
        content: [
          {
            type: "image",
            source: { type: "base64", media_type: "image/png", data: "CCCC" },
          },
          { type: "tool_result", tool_use_id: "tu_1", content: "ok" },
        ],
      },
    ]);
    expect(result).toHaveLength(2);
    expect(result[0]).toEqual({
      role: "user",
      content: [
        { type: "image_url", image_url: { url: "data:image/png;base64,CCCC" } },
      ],
    });
    expect(result[1]).toEqual({
      role: "tool",
      content: "ok",
      tool_call_id: "tu_1",
    });
  });

  it("drops an unconvertible lone image block (no usable url) rather than pushing a broken part", () => {
    const result = anthropicToOpenAI([
      { role: "user", content: [{ type: "image", source: { type: "unknown" } }] },
    ]);
    expect(result).toEqual([]);
  });
});

describe("openAIToVercel", () => {
  it("converts system message (passthrough)", () => {
    const result = openAIToVercel([
      { role: "system", content: "You are helpful" },
    ]);
    expect(result).toEqual([{ role: "system", content: "You are helpful" }]);
  });

  it("converts user string to text part array", () => {
    const result = openAIToVercel([{ role: "user", content: "hello" }]);
    expect(result).toEqual([
      { role: "user", content: [{ type: "text", text: "hello" }] },
    ]);
  });

  it("converts user content parts", () => {
    const msgs: OpenAIMessage[] = [
      {
        role: "user",
        content: [
          { type: "text", text: "look" },
          {
            type: "image_url",
            image_url: { url: "https://example.com/img.png" },
          },
        ],
      },
    ];
    const result = openAIToVercel(msgs);
    expect(result[0].content[0]).toEqual({ type: "text", text: "look" });
    expect(result[0].content[1].type).toBe("image");
    expect(result[0].content[1].image.toString()).toBe(
      "https://example.com/img.png",
    );
  });

  it("converts assistant with text and tool calls", () => {
    const msgs: OpenAIMessage[] = [
      {
        role: "assistant",
        content: "searching",
        tool_calls: [
          {
            id: "tc_1",
            type: "function",
            function: { name: "search", arguments: '{"q":"test"}' },
          },
        ],
      },
    ];
    const result = openAIToVercel(msgs);
    expect(result[0].role).toBe("assistant");
    const content = result[0].content;
    expect(content).toContainEqual({ type: "text", text: "searching" });
    expect(content).toContainEqual({
      type: "tool-call",
      toolCallId: "tc_1",
      toolName: "search",
      input: { q: "test" },
    });
  });

  it("converts assistant with null content (tool calls only)", () => {
    const msgs: OpenAIMessage[] = [
      {
        role: "assistant",
        content: null,
        tool_calls: [
          {
            id: "tc_1",
            type: "function",
            function: { name: "fn", arguments: "{}" },
          },
        ],
      },
    ];
    const result = openAIToVercel(msgs);
    expect(result[0].content).toEqual([
      { type: "tool-call", toolCallId: "tc_1", toolName: "fn", input: {} },
    ]);
  });

  it("converts tool message to tool-result", () => {
    const msgs: OpenAIMessage[] = [
      { role: "tool", content: '{"data":true}', tool_call_id: "tc_1" },
    ];
    const result = openAIToVercel(msgs);
    expect(result).toEqual([
      {
        role: "tool",
        content: [
          {
            type: "tool-result",
            toolCallId: "tc_1",
            toolName: "unknown",
            output: { type: "json", value: { data: true } },
          },
        ],
      },
    ]);
  });

  it("handles non-JSON tool content gracefully", () => {
    const msgs: OpenAIMessage[] = [
      {
        role: "tool",
        content: "plain text result",
        tool_call_id: "tc_1",
      },
    ];
    const result = openAIToVercel(msgs);
    expect(result[0].content[0].output).toEqual({ type: "text", value: "plain text result" });
  });
});

describe("round-trip conversion", () => {
  it("preserves system message through round-trip", () => {
    const original = [{ role: "system", content: "Be helpful" }];
    const roundTripped = openAIToVercel(vercelToOpenAI(original));
    expect(roundTripped).toEqual(original);
  });

  it("preserves user text through round-trip", () => {
    const vercel = [
      { role: "user", content: [{ type: "text", text: "hello" }] },
    ];
    const openai = vercelToOpenAI(vercel);
    expect(openai).toEqual([{ role: "user", content: "hello" }]);
    const back = openAIToVercel(openai);
    expect(back).toEqual(vercel);
  });

  it("preserves tool call flow through round-trip", () => {
    const vercel = [
      {
        role: "assistant",
        content: [
          { type: "text", text: "Let me search" },
          {
            type: "tool-call",
            toolCallId: "tc_1",
            toolName: "search",
            args: { q: "test" },
          },
        ],
      },
    ];
    const openai = vercelToOpenAI(vercel);
    const back = openAIToVercel(openai);

    expect(back[0].role).toBe("assistant");
    expect(back[0].content).toContainEqual({
      type: "text",
      text: "Let me search",
    });
    expect(back[0].content).toContainEqual({
      type: "tool-call",
      toolCallId: "tc_1",
      toolName: "search",
      input: { q: "test" },
    });
  });
});
