import { describe, expect, it } from "vitest";
import { normalizeAgentMessages } from "../src/convert";
import { buildOriginalLookup, findOriginal, wireRoleOf } from "../src/original-lookup";

const originals = normalizeAgentMessages([
  { role: "user", content: "q" },
  { role: "assistant", content: [{ type: "toolCall", id: "call_a", name: "exec", arguments: {} }] },
  { role: "toolResult", toolCallId: "call_a", toolName: "exec", content: "out-a" },
  { role: "assistant", content: [{ type: "toolCall", id: "dup", name: "read", arguments: {} }] },
  { role: "toolResult", toolCallId: "dup", toolName: "read", content: "out-dup-1" },
  { role: "toolResult", toolCallId: "dup", toolName: "read", content: "out-dup-2" },
]);
const lookup = buildOriginalLookup(originals);

describe("wireRoleOf", () => {
  it("maps AgentMessage roles onto wire roles", () => {
    expect(wireRoleOf({ role: "toolResult" })).toBe("tool");
    expect(wireRoleOf({ role: "tool_result" })).toBe("tool");
    expect(wireRoleOf({ role: "assistant" })).toBe("assistant");
    expect(wireRoleOf({ role: "system" })).toBe("system");
    expect(wireRoleOf({ role: "anything-else" })).toBe("user");
  });
});

describe("buildOriginalLookup", () => {
  it("indexes unique tool call ids and excludes duplicates", () => {
    expect(lookup.toolByCallId.has("call_a")).toBe(true);
    expect(lookup.toolByCallId.has("dup")).toBe(false);
  });
});

describe("findOriginal", () => {
  it("matches tool messages by tool_call_id regardless of position", () => {
    const found = findOriginal({
      compressed: { role: "tool", content: "x", tool_call_id: "call_a" },
      index: 0,
      compressedCount: 1,
      lookup,
    });
    expect(found).toBe(originals[2]);
  });

  it("uses the echoed hrIndex hint when role-consistent", () => {
    const found = findOriginal({
      compressed: { role: "assistant", content: "x", _headroomMeta: { hrIndex: 3 } },
      index: 0,
      compressedCount: 1,
      lookup,
    });
    expect(found).toBe(originals[3]);
  });

  it("rejects an hrIndex hint that points at a different role", () => {
    const found = findOriginal({
      compressed: { role: "assistant", content: "x", _headroomMeta: { hrIndex: 0 } },
      index: 0,
      compressedCount: 1,
      lookup,
    });
    expect(found).toBeNull();
  });

  it("falls back to positional alignment when the message count is unchanged", () => {
    const found = findOriginal({
      compressed: { role: "tool", content: "x", tool_call_id: "dup" },
      index: 5,
      compressedCount: originals.length,
      lookup,
    });
    expect(found).toBe(originals[5]);
  });

  it("returns null when nothing can be trusted", () => {
    const found = findOriginal({
      compressed: { role: "tool", content: "x", tool_call_id: "dup" },
      index: 0,
      compressedCount: 2,
      lookup,
    });
    expect(found).toBeNull();
  });
});
