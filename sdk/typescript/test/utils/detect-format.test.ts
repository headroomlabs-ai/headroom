import { describe, it, expect } from "vitest";
import { detectFormat } from "../../src/utils/format.js";

describe("detectFormat", () => {
  // ---- OpenAI ----
  it("detects OpenAI: assistant with tool_calls field", () => {
    const messages = [
      { role: "user", content: "hello" },
      {
        role: "assistant",
        content: null,
        tool_calls: [{ id: "call_1", type: "function", function: { name: "search", arguments: "{}" } }],
      },
    ];
    expect(detectFormat(messages)).toBe("openai");
  });

  it("detects OpenAI: tool role with tool_call_id and string content", () => {
    const messages = [
      { role: "tool", content: '{"result": true}', tool_call_id: "call_1" },
    ];
    expect(detectFormat(messages)).toBe("openai");
  });

  it("defaults to OpenAI for simple string messages", () => {
    const messages = [
      { role: "user", content: "hello" },
      { role: "assistant", content: "hi" },
    ];
    expect(detectFormat(messages)).toBe("openai");
  });

  it("defaults to OpenAI for empty array", () => {
    expect(detectFormat([])).toBe("openai");
  });

  // ---- Anthropic ----
  it("detects Anthropic: tool_use content block", () => {
    const messages = [
      {
        role: "assistant",
        content: [{ type: "tool_use", id: "toolu_01", name: "search", input: {} }],
      },
    ];
    expect(detectFormat(messages)).toBe("anthropic");
  });

  it("detects Anthropic: tool_result content block", () => {
    const messages = [
      {
        role: "user",
        content: [{ type: "tool_result", tool_use_id: "toolu_01", content: "result" }],
      },
    ];
    expect(detectFormat(messages)).toBe("anthropic");
  });

  it("detects Anthropic: image with source.type", () => {
    const messages = [
      {
        role: "user",
        content: [
          { type: "text", text: "What's this?" },
          { type: "image", source: { type: "base64", media_type: "image/jpeg", data: "..." } },
        ],
      },
    ];
    expect(detectFormat(messages)).toBe("anthropic");
  });

  // ---- Vercel AI SDK ----
  it("detects Vercel: tool-call part (hyphenated)", () => {
    const messages = [
      {
        role: "assistant",
        content: [{ type: "tool-call", toolCallId: "tc_1", toolName: "search", args: {} }],
      },
    ];
    expect(detectFormat(messages)).toBe("vercel");
  });

  it("detects Vercel: tool-result part (hyphenated)", () => {
    const messages = [
      {
        role: "tool",
        content: [{ type: "tool-result", toolCallId: "tc_1", toolName: "search", result: {} }],
      },
    ];
    expect(detectFormat(messages)).toBe("vercel");
  });

  // ---- Google Gemini ----
  it("detects Gemini: parts field instead of content", () => {
    const messages = [
      { role: "user", parts: [{ text: "hello" }] },
    ];
    expect(detectFormat(messages)).toBe("gemini");
  });

  it("detects Gemini: model role", () => {
    const messages = [
      { role: "model", parts: [{ text: "hi" }] },
    ];
    expect(detectFormat(messages)).toBe("gemini");
  });

  it("detects Gemini: functionCall in parts", () => {
    const messages = [
      { role: "model", parts: [{ functionCall: { name: "search", args: {} } }] },
    ];
    expect(detectFormat(messages)).toBe("gemini");
  });

  it("detects Gemini: functionResponse in parts", () => {
    const messages = [
      { role: "user", parts: [{ functionResponse: { name: "search", response: {} } }] },
    ];
    expect(detectFormat(messages)).toBe("gemini");
  });

  // ---- Mixed/edge cases ----
  it("detects format from first distinguishing message", () => {
    // First message is ambiguous (plain text), second has Vercel marker
    const messages = [
      { role: "user", content: "hello" },
      {
        role: "assistant",
        content: [{ type: "tool-call", toolCallId: "tc_1", toolName: "fn", args: {} }],
      },
    ];
    expect(detectFormat(messages)).toBe("vercel");
  });
});
