import { describe, expect, it } from "vitest";
import {
  dropOrphanToolResults,
  isTurnStartMessage,
  selectTruncateStart,
} from "../src/truncate-boundary";

const user = (i: number) => ({ role: "user", content: `u${i}` });
const assistantCall = (id: string) => ({
  role: "assistant",
  content: [{ type: "toolCall", id, name: "read", arguments: {} }],
});
const toolResult = (id: string) => ({ role: "toolResult", toolCallId: id, content: "r" });

describe("isTurnStartMessage (mirrors OpenClaw compaction rules)", () => {
  it("accepts user, bashExecution and summary roles", () => {
    expect(isTurnStartMessage({ role: "user" })).toBe(true);
    expect(isTurnStartMessage({ role: "bashExecution" })).toBe(true);
    expect(isTurnStartMessage({ role: "branchSummary" })).toBe(true);
    expect(isTurnStartMessage({ role: "compactionSummary" })).toBe(true);
  });

  it("rejects assistant and toolResult", () => {
    expect(isTurnStartMessage({ role: "assistant" })).toBe(false);
    expect(isTurnStartMessage({ role: "toolResult" })).toBe(false);
  });

  it("accepts custom messages unless they are runtime-context carriers", () => {
    expect(isTurnStartMessage({ role: "custom", customType: "note" })).toBe(true);
    expect(
      isTurnStartMessage({ role: "custom", details: { runtimeContextCarrier: true } }),
    ).toBe(false);
  });

  it("rejects malformed input", () => {
    expect(isTurnStartMessage(null)).toBe(false);
    expect(isTurnStartMessage("user")).toBe(false);
    expect(isTurnStartMessage({})).toBe(false);
  });
});

describe("selectTruncateStart", () => {
  const transcript = [assistantCall("a"), toolResult("a"), user(0), assistantCall("b"), toolResult("b"), user(1)];

  it("moves a cut that lands on a toolResult forward to the next turn start", () => {
    expect(selectTruncateStart(transcript, 1)).toEqual({ startIndex: 2, shiftedBy: 1 });
  });

  it("keeps a cut that already sits on a turn start", () => {
    expect(selectTruncateStart(transcript, 2)).toEqual({ startIndex: 2, shiftedBy: 0 });
  });

  it("moves a mid-turn cut forward, never leaving an assistant continuation first", () => {
    expect(selectTruncateStart(transcript, 3)).toEqual({ startIndex: 5, shiftedBy: 2 });
    expect(selectTruncateStart(transcript, 4)).toEqual({ startIndex: 5, shiftedBy: 1 });
  });

  it("falls back to the previous turn start when nothing follows", () => {
    const tail = [user(0), assistantCall("x"), toolResult("x")];
    expect(selectTruncateStart(tail, 2)).toEqual({ startIndex: 0, shiftedBy: -2 });
  });

  it("returns null when there is no turn boundary at all", () => {
    expect(selectTruncateStart([assistantCall("a"), toolResult("a")], 1)).toBeNull();
    expect(selectTruncateStart([], 0)).toBeNull();
  });

  it("clamps out-of-range requests", () => {
    expect(selectTruncateStart(transcript, -5)).toEqual({ startIndex: 2, shiftedBy: 2 });
    expect(selectTruncateStart(transcript, 99)).toEqual({ startIndex: 5, shiftedBy: 0 });
  });
});

describe("dropOrphanToolResults", () => {
  it("removes tool results whose call is not in the list", () => {
    expect(dropOrphanToolResults([toolResult("ghost"), user(0)])).toEqual([user(0)]);
  });

  it("keeps tool results that follow their assistant call", () => {
    const paired = [user(0), assistantCall("a"), toolResult("a")];
    expect(dropOrphanToolResults(paired)).toEqual(paired);
  });

  it("removes a result that precedes its call (out of order)", () => {
    expect(dropOrphanToolResults([toolResult("a"), assistantCall("a")])).toEqual([assistantCall("a")]);
  });

  it("recognises legacy tool_use_id and tool_use blocks", () => {
    const legacy = [
      { role: "assistant", content: [{ type: "tool_use", id: "L", name: "x", input: {} }] },
      { role: "tool_result", tool_use_id: "L", content: "ok" },
    ];
    expect(dropOrphanToolResults(legacy)).toEqual(legacy);
  });
});
