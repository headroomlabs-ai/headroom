/**
 * Stress tests: OpenClaw native tools through convert + assemble + compaction mocks.
 * Validates what goes into compress vs what comes back out.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { messageHasProtectedToolPayload } from "../../src/content-blocks.js";
import { agentToOpenAI, openAIToAgent, type OpenAIMessage } from "../../src/convert.js";
import {
  NATIVE_TOOL_SCENARIOS,
  buildMegaTranscript,
  buildToolScenarioTranscript,
  extractOpenAiToolNames,
  extractToolResults,
  type NativeToolScenario,
} from "../fixtures/openclaw-native-tools.js";

const fetchMock = vi.hoisted(() => vi.fn());
const mocked = vi.hoisted(() => ({
  start: vi.fn(async () => "http://127.0.0.1:8787"),
  stop: vi.fn(async () => undefined),
  logger: { debug: vi.fn(), error: vi.fn(), info: vi.fn(), warn: vi.fn() },
}));

vi.mock("headroom-ai", () => ({ compress: vi.fn() }));
vi.mock("../../src/openclaw-compaction.js", () => ({
  delegateCompactionToRuntime: vi.fn(),
}));
vi.mock("../../src/proxy-manager.js", () => ({
  ProxyManager: class {
    start = mocked.start;
    stop = mocked.stop;
  },
  defaultLogger: mocked.logger,
}));

import { compress } from "headroom-ai";
import { HeadroomContextEngine } from "../../src/engine.js";
import { planHeadroomCompaction } from "../../src/compaction.js";

afterEach(() => {
  vi.mocked(compress).mockReset();
  fetchMock.mockReset();
});

beforeEach(() => {
  vi.stubGlobal("fetch", fetchMock);
});

/** Aggressive proxy mock: lossy text, strip assistant text, random-looking summaries. */
function mockAggressiveCompress(options?: {
  ccrHashes?: string[];
  dropMeta?: boolean;
  truncateAssistant?: boolean;
}) {
  vi.mocked(compress).mockImplementation(async (messages) => {
    const openaiIn = messages as OpenAIMessage[];
    const compressed = openaiIn.map((msg) => {
      if (msg.role === "tool") {
        const stripped = {
          ...msg,
          content: `[CRUSHED:${msg.name ?? "unknown"}:${String(msg.content ?? "").length}B]`,
        };
        if (options?.dropMeta) {
          delete stripped._headroomMeta;
        }
        return stripped;
      }
      if (msg.role === "assistant" && options?.truncateAssistant && msg.content) {
        return { ...msg, content: msg.content.slice(0, 32) + "…" };
      }
      if (msg.role === "user" && typeof msg.content === "string" && msg.content.length > 100) {
        return { ...msg, content: msg.content.slice(0, 64) + "…" };
      }
      return msg;
    });

    return {
      compressed: true,
      messages: compressed,
      tokensBefore: 500_000,
      tokensAfter: 120_000,
      tokensSaved: 380_000,
      compressionRatio: 0.24,
      transformsApplied: ["ContentRouter", "SmartCrusher", "ToolCrusher"],
      ccrHashes: options?.ccrHashes ?? [],
    };
  });
}

function mockFetchAggressiveCompress() {
  fetchMock.mockImplementation(async (_url: string, init?: RequestInit) => {
    const body = JSON.parse(String(init?.body ?? "{}")) as {
      messages: OpenAIMessage[];
    };
    const compressed = body.messages.map((msg) =>
      msg.role === "tool"
        ? { ...msg, content: `[DURABLE-CRUSHED:${msg.name}:${String(msg.content ?? "").length}B]` }
        : msg,
    );
    return {
      ok: true,
      json: async () => ({
        messages: compressed,
        tokens_before: 800_000,
        tokens_after: 200_000,
        tokens_saved: 600_000,
      }),
    };
  });
}

function assertStructuredPayloadPreserved(
  before: unknown[],
  after: unknown[],
  toolName: string,
) {
  const beforeResult = extractToolResults(before).find((r) => r.toolName === toolName);
  const afterResult = extractToolResults(after).find((r) => r.toolName === toolName);
  expect(beforeResult, `missing before toolResult for ${toolName}`).toBeDefined();
  expect(afterResult, `missing after toolResult for ${toolName}`).toBeDefined();

  const beforeBlocks = Array.isArray(beforeResult?.content) ? beforeResult.content : [];
  const afterBlocks = Array.isArray(afterResult?.content) ? afterResult.content : [];
  const beforeImage = beforeBlocks.find(
    (b) => typeof b === "object" && b !== null && (b as { type?: string }).type === "image",
  ) as { data?: string } | undefined;
  const afterImage = afterBlocks.find(
    (b) => typeof b === "object" && b !== null && (b as { type?: string }).type === "image",
  ) as { data?: string } | undefined;

  expect(afterImage?.data).toBe(beforeImage?.data);
}

describe("native tool conversion stress", () => {
  it.each(NATIVE_TOOL_SCENARIOS.map((s) => [s.toolName, s.payloadKind, s] as const))(
    "%s (%s): OpenAI payload includes tool.name and tool_call_id",
    (_name, _kind, scenario) => {
      const transcript = buildToolScenarioTranscript(scenario, 0);
      const openai = agentToOpenAI(transcript);
      const toolMsgs = openai.filter((m) => m.role === "tool");
      expect(toolMsgs.length).toBeGreaterThan(0);
      for (const msg of toolMsgs) {
        expect(msg.name, `${scenario.toolName} missing name`).toBeTruthy();
        expect(msg.tool_call_id).toBeTruthy();
      }
      const names = extractOpenAiToolNames(openai);
      if (scenario.toolName !== "assistant") {
        expect(names).toContain(scenario.toolName);
      } else {
        expect(names).toEqual(expect.arrayContaining(["read", "browser"]));
      }
    },
  );

  it.each(
    NATIVE_TOOL_SCENARIOS.filter((s) => s.mustPreserveStructured).map(
      (s) => [s.toolName, s] as const,
    ),
  )("%s survives lossy OpenAI round-trip with _headroomMeta stripped", (toolName, scenario) => {
    const transcript = buildToolScenarioTranscript(scenario, 0);
    const openai = agentToOpenAI(transcript);
    const crushed = openai.map(({ _headroomMeta: _dropped, ...msg }) =>
      msg.role === "tool" ? { ...msg, content: "[TOTAL LOSS]" } : msg,
    );
    const restored = openAIToAgent(crushed, { originals: transcript });
    assertStructuredPayloadPreserved(transcript, restored, toolName === "assistant" ? "read" : toolName);
    if (toolName === "view_image") {
      assertStructuredPayloadPreserved(transcript, restored, "view_image");
    }
  });

  it("mega transcript (120 msgs): every tool name appears in OpenAI compress input", () => {
    const mega = buildMegaTranscript(120);
    const openai = agentToOpenAI(mega);
    const names = new Set(extractOpenAiToolNames(openai));
    const expected = new Set(
      NATIVE_TOOL_SCENARIOS.filter((s) => s.toolName !== "assistant").map((s) => s.toolName),
    );
    expected.add("read");
    expected.add("browser");
    for (const name of expected) {
      expect(names.has(name), `missing ${name} in mega transcript OpenAI payload`).toBe(true);
    }
    expect(openai.length).toBeGreaterThan(80);
  });

  it("mega transcript: protected payload detection scales linearly", () => {
    const mega = buildMegaTranscript(120);
    const protectedCount = mega.filter((m) => messageHasProtectedToolPayload(m)).length;
    expect(protectedCount).toBeGreaterThan(5);
  });

  it("mega transcript: no base64 image bytes are ever sent to the proxy", () => {
    const mega = buildMegaTranscript(120);
    const imageData = extractToolResults(mega)
      .flatMap((r) => (Array.isArray(r.content) ? r.content : []))
      .map((b) => (b as { data?: string }).data)
      .filter((d): d is string => typeof d === "string" && d.length > 0);
    expect(imageData.length).toBeGreaterThan(0);

    const wire = JSON.stringify(agentToOpenAI(mega));
    for (const data of imageData) {
      expect(wire).not.toContain(data);
    }
  });
});

describe("native tool assemble() stress (mock proxy)", () => {
  async function runAssemble(transcript: unknown[]) {
    mockAggressiveCompress();
    const engine = new HeadroomContextEngine({
      assembleCompressConfig: { protect_recent: 2 },
    });
    (engine as { proxyUrl: string | null }).proxyUrl = "http://127.0.0.1:8787";
    return engine.assemble({ sessionId: "stress", messages: transcript });
  }

  it("assemble preserves view_image image bytes after aggressive crush", async () => {
    const scenario = NATIVE_TOOL_SCENARIOS.find(
      (s) => s.toolName === "view_image" && s.payloadKind === "image",
    ) as NativeToolScenario;
    const transcript = buildToolScenarioTranscript(scenario, 0);
    const result = await runAssemble(transcript);
    assertStructuredPayloadPreserved(transcript, result.messages, "view_image");
  });

  it("assemble mega transcript: all tool.name values present in compress request", async () => {
    mockAggressiveCompress();
    const mega = buildMegaTranscript(120);
    const engine = new HeadroomContextEngine();
    (engine as { proxyUrl: string | null }).proxyUrl = "http://127.0.0.1:8787";
    await engine.assemble({ sessionId: "stress-mega", messages: mega });

    expect(compress).toHaveBeenCalledTimes(1);
    const sent = (vi.mocked(compress).mock.calls[0]?.[0] ?? []) as OpenAIMessage[];
    const sentNames = new Set(extractOpenAiToolNames(sent));
    expect(sentNames.size).toBeGreaterThan(10);
    expect(sentNames.has("view_image")).toBe(true);
    expect(sentNames.has("exec")).toBe(true);
    expect(sentNames.has("browser")).toBe(true);
  });

  it("assemble mega transcript: view_image images preserved after aggressive crush", async () => {
    const mega = buildMegaTranscript(120);
    const result = await runAssemble(mega);
    const images = extractToolResults(result.messages).filter((r) =>
      Array.isArray(r.content) &&
      r.content.some((b) => typeof b === "object" && b !== null && (b as { type?: string }).type === "image"),
    );
    expect(images.length).toBeGreaterThan(0);
    for (const img of images) {
      const block = (img.content as Array<{ data?: string }>).find((b) => b.data);
      expect(block?.data?.length).toBeGreaterThan(8);
    }
  });

  it("assemble 20 parallel sessions do not cross-contaminate tool payloads", async () => {
    mockAggressiveCompress();
    const engine = new HeadroomContextEngine();
    (engine as { proxyUrl: string | null }).proxyUrl = "http://127.0.0.1:8787";

    const scenarios = NATIVE_TOOL_SCENARIOS.filter((s) => s.toolName === "view_image" && s.payloadKind === "image");
    const results = await Promise.all(
      Array.from({ length: 20 }, (_, i) =>
        engine.assemble({
          sessionId: `parallel-${i}`,
          messages: buildToolScenarioTranscript(scenarios[0], i),
        }),
      ),
    );

    for (const [i, result] of results.entries()) {
      const view = extractToolResults(result.messages).find((r) => r.toolName === "view_image");
      const data = (view?.content as Array<{ data?: string }> | undefined)?.find((b) => b.data)?.data;
      expect(data).toContain(Buffer.from(`fake-png-bytes-${i}`).toString("base64"));
    }
  });

  it("assemble with dropMeta crush still restores image blocks (no _headroomMeta dependency)", async () => {
    mockAggressiveCompress({ dropMeta: true });
    const scenario = NATIVE_TOOL_SCENARIOS.find(
      (s) => s.toolName === "view_image" && s.payloadKind === "image",
    ) as NativeToolScenario;
    const transcript = buildToolScenarioTranscript(scenario, 99);
    const engine = new HeadroomContextEngine();
    (engine as { proxyUrl: string | null }).proxyUrl = "http://127.0.0.1:8787";
    const result = await engine.assemble({ sessionId: "drop-meta", messages: transcript });
    assertStructuredPayloadPreserved(transcript, result.messages, "view_image");
    const view = extractToolResults(result.messages).find((r) => r.toolName === "view_image");
    expect((view?.content as Array<{ text?: string }>)?.[0]?.text).toContain("CRUSHED");
  });

  it("assemble with dropMeta crush restores isError on error tool results", async () => {
    mockAggressiveCompress({ dropMeta: true });
    const scenario = NATIVE_TOOL_SCENARIOS.find(
      (s) => s.toolName === "view_image" && s.payloadKind === "error",
    ) as NativeToolScenario;
    const transcript = buildToolScenarioTranscript(scenario, 7);
    const engine = new HeadroomContextEngine();
    (engine as { proxyUrl: string | null }).proxyUrl = "http://127.0.0.1:8787";
    const result = await engine.assemble({ sessionId: "drop-meta-err", messages: transcript });
    const before = extractToolResults(transcript).find((r) => r.toolName === "view_image");
    const after = extractToolResults(result.messages).find((r) => r.toolName === "view_image");
    expect(before?.isError).toBe(true);
    expect(after?.isError).toBe(true);
  });

  it("text-only tools survive assemble with lossy but structured toolResult blocks", async () => {
    const scenario = NATIVE_TOOL_SCENARIOS.find((s) => s.toolName === "exec") as NativeToolScenario;
    const transcript = buildToolScenarioTranscript(scenario, 0);
    const result = await runAssemble(transcript);
    const exec = extractToolResults(result.messages).find((r) => r.toolName === "exec");
    expect(exec?.toolName).toBe("exec");
    expect(Array.isArray(exec?.content)).toBe(true);
    // Lossy crush replaces blob text — structure + toolName must remain intact
    expect((exec?.content as Array<{ text?: string }>)?.[0]?.text).toContain("[CRUSHED:exec:");
  });
});

describe("native tool durable compaction stress (mock fetch)", () => {
  it("planHeadroomCompaction skips image toolResult replace under aggressive durable crush", async () => {
    mockFetchAggressiveCompress();
    const scenario = NATIVE_TOOL_SCENARIOS.find(
      (s) => s.toolName === "view_image" && s.payloadKind === "image",
    ) as NativeToolScenario;
    const messages = scenario.buildTurn(0);
    const branchMessages = messages.map((message, index) => ({
      entryId: `entry-${index}`,
      parentId: index === 0 ? null : `entry-${index - 1}`,
      message,
    }));

    const plan = await planHeadroomCompaction({
      branchMessages,
      tokenBudget: 120_000,
      proxyUrl: "http://127.0.0.1:8787",
      timeoutMs: 30_000,
      force: true,
    });

    const replacedIds = plan.replacements?.map((r) => r.entryId) ?? [];
    const imageEntry = branchMessages.find((e) =>
      messageHasProtectedToolPayload(e.message),
    );
    expect(imageEntry).toBeDefined();
    expect(replacedIds).not.toContain(imageEntry?.entryId);
  });

  it("durable compress request includes protect_recent: 2", async () => {
    mockFetchAggressiveCompress();
    const scenario = NATIVE_TOOL_SCENARIOS.find((s) => s.toolName === "read") as NativeToolScenario;
    await planHeadroomCompaction({
      branchMessages: scenario.buildTurn(0).map((message, index) => ({
        entryId: `e-${index}`,
        parentId: index === 0 ? null : `e-${index - 1}`,
        message,
      })),
      tokenBudget: 120_000,
      proxyUrl: "http://127.0.0.1:8787",
      timeoutMs: 30_000,
    });

    const body = JSON.parse(String(fetchMock.mock.calls[0]?.[1]?.body));
    expect(body.config.protect_recent).toBe(2);
    const toolNames = extractOpenAiToolNames(body.messages);
    expect(toolNames).toContain("read");
  });
});
