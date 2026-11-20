import { describe, expect, it } from "vitest";
import {
  blockToWirePlaceholder,
  mergeCompressedTextIntoBlocks,
  messageHasProtectedToolPayload,
  serializeAssistantTextForWire,
  serializeBlocksForWire,
  stripWirePlaceholders,
} from "../src/content-blocks";

const IMAGE = { type: "image", data: "aGVsbG8=", mimeType: "image/png" };

describe("serializeBlocksForWire", () => {
  it("emits text verbatim and placeholders for non-text blocks", () => {
    expect(serializeBlocksForWire([{ type: "text", text: "a" }, IMAGE, { type: "text", text: "b" }], "\n")).toBe(
      "a\n[headroom-omitted image image/png 6 bytes]\nb",
    );
  });

  it("produces an empty string for an empty block list", () => {
    expect(serializeBlocksForWire([], "\n")).toBe("");
  });
});

describe("blockToWirePlaceholder", () => {
  it("never includes the payload", () => {
    const placeholder = blockToWirePlaceholder({ type: "custom", payload: "SECRET" });
    expect(placeholder).toBe("[headroom-omitted custom]");
    expect(placeholder).not.toContain("SECRET");
  });
});

describe("stripWirePlaceholders", () => {
  it("removes placeholder lines and trims", () => {
    expect(stripWirePlaceholders("keep\n[headroom-omitted image image/png 6 bytes]\nalso")).toBe(
      "keep\nalso",
    );
  });

  it("is a no-op when no placeholder is present", () => {
    expect(stripWirePlaceholders("plain")).toBe("plain");
  });
});

describe("mergeCompressedTextIntoBlocks", () => {
  const original = [
    { type: "text", text: "first" },
    IMAGE,
    { type: "text", text: "second" },
  ];

  const originalWireText = serializeBlocksForWire(original, "\n");

  it("returns the identical array when text is unchanged", () => {
    expect(
      mergeCompressedTextIntoBlocks({ originalBlocks: original, originalWireText, compressedText: originalWireText }),
    ).toBe(original);
  });

  it("folds compressed text into the first text slot and keeps non-text blocks in place", () => {
    expect(
      mergeCompressedTextIntoBlocks({ originalBlocks: original, originalWireText, compressedText: "[c]" }),
    ).toEqual([{ type: "text", text: "[c]" }, IMAGE]);
  });

  it("prepends text when the original had none", () => {
    expect(
      mergeCompressedTextIntoBlocks({
        originalBlocks: [IMAGE],
        originalWireText: serializeBlocksForWire([IMAGE], "\n"),
        compressedText: "caption",
      }),
    ).toEqual([{ type: "text", text: "caption" }, IMAGE]);
  });

  it("drops text entirely when the proxy returned only placeholders", () => {
    expect(
      mergeCompressedTextIntoBlocks({
        originalBlocks: original,
        originalWireText,
        compressedText: "[headroom-omitted image image/png 6 bytes]",
      }),
    ).toEqual([IMAGE]);
  });
});

describe("serializeAssistantTextForWire", () => {
  it("emits only text — no placeholders for thinking or tool calls", () => {
    expect(
      serializeAssistantTextForWire([
        { type: "thinking", thinking: "t" },
        { type: "text", text: "a" },
        { type: "toolCall", id: "c", name: "exec", arguments: {} },
        { type: "text", text: "b" },
      ]),
    ).toBe("ab");
  });
});

describe("messageHasProtectedToolPayload", () => {
  it("detects image tool results", () => {
    expect(messageHasProtectedToolPayload({ role: "toolResult", content: [IMAGE] })).toBe(true);
  });

  it("ignores plain text tool results", () => {
    expect(
      messageHasProtectedToolPayload({ role: "toolResult", content: [{ type: "text", text: "ok" }] }),
    ).toBe(false);
  });

  it("detects assistant tool calls", () => {
    expect(
      messageHasProtectedToolPayload({
        role: "assistant",
        content: [{ type: "toolCall", id: "c", name: "exec", arguments: {} }],
      }),
    ).toBe(true);
  });

  it("detects user messages with embedded tool_result blocks", () => {
    expect(
      messageHasProtectedToolPayload({
        role: "user",
        content: [{ type: "tool_result", tool_use_id: "call_1", content: [{ type: "text", text: "ok" }] }],
      }),
    ).toBe(true);
  });

  it("ignores string content and system messages", () => {
    expect(messageHasProtectedToolPayload({ role: "user", content: "text" })).toBe(false);
    expect(messageHasProtectedToolPayload({ role: "system", content: [IMAGE] })).toBe(false);
  });
});
