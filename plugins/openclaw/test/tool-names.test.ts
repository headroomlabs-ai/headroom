import { describe, expect, it } from "vitest";
import {
  TOOL_CALL_WRAPPER,
  buildToolCallNameMap,
  isProtectedToolName,
  normalizeProtectedToolNames,
  resolveOpenClawToolId,
  resolveToolName,
  toolNameAliases,
} from "../src/tool-names";

const OPENCLAW_ARGS = {
  id: "mcp:openrouter-video-vision:openrouter-video-vision__analyze_video",
  args: { video_path: "/tmp/clip.mp4", prompt: "what happens?" },
};

describe("resolveOpenClawToolId", () => {
  it("maps mcp:<server>:<server>__<tool> to the canonical mcp__ form", () => {
    expect(resolveOpenClawToolId(OPENCLAW_ARGS.id)).toBe(
      "mcp__openrouter-video-vision__analyze_video",
    );
  });

  it("handles ids whose tool segment lacks the server prefix", () => {
    expect(resolveOpenClawToolId("mcp:pixellab:resources_read")).toBe(
      "mcp__pixellab__resources_read",
    );
  });

  it("resolves non-MCP ids to their last segment", () => {
    expect(resolveOpenClawToolId("openclaw:web_fetch")).toBe("web_fetch");
    expect(resolveOpenClawToolId("plain_tool")).toBe("plain_tool");
  });

  it("returns null for malformed ids", () => {
    expect(resolveOpenClawToolId("")).toBeNull();
    expect(resolveOpenClawToolId("   ")).toBeNull();
    expect(resolveOpenClawToolId("mcp:")).toBeNull();
    expect(resolveOpenClawToolId("mcp:server:")).toBeNull();
    expect(resolveOpenClawToolId("mcp::tool")).toBeNull();
    expect(resolveOpenClawToolId("mcp:server:server__")).toBeNull();
    expect(resolveOpenClawToolId("source:")).toBeNull();
  });
});

describe("resolveToolName", () => {
  it("passes non-wrapper names through unchanged", () => {
    expect(resolveToolName("exec", '{"command":"ls"}')).toBe("exec");
    expect(resolveToolName("read", { path: "/x" })).toBe("read");
  });

  it("unwraps OpenClaw wrappers from object and JSON-string arguments", () => {
    expect(resolveToolName(TOOL_CALL_WRAPPER, OPENCLAW_ARGS)).toBe(
      "mcp__openrouter-video-vision__analyze_video",
    );
    expect(resolveToolName(TOOL_CALL_WRAPPER, JSON.stringify(OPENCLAW_ARGS))).toBe(
      "mcp__openrouter-video-vision__analyze_video",
    );
  });

  it("unwraps Hermes wrappers and prefers `name` over `id`", () => {
    expect(resolveToolName(TOOL_CALL_WRAPPER, { name: "read_file", arguments: {} })).toBe(
      "read_file",
    );
    expect(resolveToolName(TOOL_CALL_WRAPPER, { name: " web_search ", id: "mcp:s:s__t" })).toBe(
      "web_search",
    );
  });

  it("fails open to the wrapper name on malformed payloads", () => {
    expect(resolveToolName(TOOL_CALL_WRAPPER, undefined)).toBe(TOOL_CALL_WRAPPER);
    expect(resolveToolName(TOOL_CALL_WRAPPER, "not json")).toBe(TOOL_CALL_WRAPPER);
    expect(resolveToolName(TOOL_CALL_WRAPPER, "[1,2]")).toBe(TOOL_CALL_WRAPPER);
    expect(resolveToolName(TOOL_CALL_WRAPPER, { id: 42 })).toBe(TOOL_CALL_WRAPPER);
    expect(resolveToolName(TOOL_CALL_WRAPPER, { id: "mcp:", name: "" })).toBe(TOOL_CALL_WRAPPER);
  });
});

describe("toolNameAliases", () => {
  it("expands canonical MCP names to server-prefixed and bare spellings", () => {
    expect(toolNameAliases("mcp__video-vision__analyze_video")).toEqual([
      "mcp__video-vision__analyze_video",
      "mcp_video-vision_analyze_video",
      "video-vision__analyze_video",
      "analyze_video",
    ]);
  });

  it("leaves plain and malformed names alone", () => {
    expect(toolNameAliases("exec")).toEqual(["exec"]);
    expect(toolNameAliases("mcp__server__")).toEqual(["mcp__server__"]);
    expect(toolNameAliases("mcp____tool")).toEqual(["mcp____tool"]);
  });
});

describe("normalizeProtectedToolNames", () => {
  it("lowercases, trims and drops non-string or empty entries", () => {
    const names = normalizeProtectedToolNames([" Analyze_Video ", "", 7, null, "mcp__*"]);
    expect([...names]).toEqual(["analyze_video", "mcp__*"]);
  });

  it("returns an empty set for non-array input", () => {
    expect(normalizeProtectedToolNames(undefined).size).toBe(0);
    expect(normalizeProtectedToolNames("analyze_video").size).toBe(0);
  });
});

describe("isProtectedToolName", () => {
  const canonical = "mcp__openrouter-video-vision__analyze_video";

  it("matches the bare tool, server-prefixed, canonical and glob spellings", () => {
    for (const entry of [
      "analyze_video",
      "ANALYZE_VIDEO",
      "openrouter-video-vision__analyze_video",
      canonical,
      "mcp__*",
      "*analyze_video",
      "mcp__openrouter-video-vision__*",
    ]) {
      expect(isProtectedToolName(canonical, normalizeProtectedToolNames([entry]))).toBe(true);
    }
  });

  it("does not match other tools or an empty set", () => {
    expect(isProtectedToolName(canonical, normalizeProtectedToolNames(["exec", "read"]))).toBe(
      false,
    );
    expect(isProtectedToolName("mcp__pixellab__list_characters", new Set(["analyze_video"]))).toBe(
      false,
    );
    expect(isProtectedToolName(canonical, new Set())).toBe(false);
  });

  it("treats regex metacharacters in patterns literally", () => {
    expect(isProtectedToolName("a.b", new Set(["a.b"]))).toBe(true);
    expect(isProtectedToolName("axb", new Set(["a.b"]))).toBe(false);
  });
});

describe("buildToolCallNameMap", () => {
  it("maps tool call ids to resolved names from assistant toolCall blocks", () => {
    const map = buildToolCallNameMap([
      { role: "user", content: "go" },
      {
        role: "assistant",
        content: [
          { type: "text", text: "Running." },
          { type: "toolCall", id: "call_1", name: "exec", arguments: { command: "ls" } },
          { type: "toolCall", id: "call_2", name: TOOL_CALL_WRAPPER, arguments: OPENCLAW_ARGS },
          { type: "toolCall", id: 3, name: "ignored" },
        ],
      },
      { role: "assistant", content: "plain string, no calls" },
    ]);

    expect(map.get("call_1")).toBe("exec");
    expect(map.get("call_2")).toBe("mcp__openrouter-video-vision__analyze_video");
    expect(map.size).toBe(2);
  });
});
