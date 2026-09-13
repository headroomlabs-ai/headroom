import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { tmpdir } from "node:os";

const fetchMock = vi.hoisted(() => vi.fn());
const sessionManagerMock = vi.hoisted(() => ({
  getBranch: vi.fn(() => []),
  branch: vi.fn(),
  resetLeaf: vi.fn(),
  appendMessage: vi.fn(),
}));

vi.mock("openclaw/plugin-sdk/agent-sessions", () => ({
  SessionManager: {
    open: vi.fn(() => sessionManagerMock),
  },
}));

vi.mock("../src/proxy-manager.js", () => ({
  ProxyManager: class {
    start = vi.fn(async () => "http://127.0.0.1:8787");
    stop = vi.fn(async () => undefined);
  },
  defaultLogger: {
    debug: vi.fn(),
    error: vi.fn(),
    info: vi.fn(),
    warn: vi.fn(),
  },
}));

import { SessionManager } from "openclaw/plugin-sdk/agent-sessions";
import { HeadroomContextEngine } from "../src/engine.js";
import {
  applyCompactionPlan,
  planHeadroomCompaction,
  type BranchMessageEntry,
} from "../src/compaction.js";

beforeEach(() => {
  vi.stubGlobal("fetch", fetchMock);
});

afterEach(() => {
  fetchMock.mockReset();
  sessionManagerMock.getBranch.mockReset();
  sessionManagerMock.getBranch.mockReturnValue([]);
  sessionManagerMock.branch.mockReset();
  sessionManagerMock.resetLeaf.mockReset();
  sessionManagerMock.appendMessage.mockReset();
  vi.unstubAllGlobals();
});

function mockCompressResponse(payload: {
  messages: unknown[];
  tokens_before: number;
  tokens_after: number;
  tokens_saved: number;
}) {
  fetchMock.mockResolvedValue({
    ok: true,
    json: async () => payload,
  });
}

describe("planHeadroomCompaction", () => {
  it("forces truncate when headroom returns noop on a huge session", async () => {
    mockCompressResponse({
      messages: [{ role: "user", content: "hello" }],
      tokens_before: 900_000,
      tokens_after: 900_000,
      tokens_saved: 0,
    });

    const branchMessages = Array.from({ length: 100 }, (_, index) => ({
      entryId: `entry-${index}`,
      parentId: index === 0 ? null : `entry-${index - 1}`,
      message: { role: "user", content: `msg-${index}`, timestamp: index },
    }));

    const plan = await planHeadroomCompaction({
      branchMessages,
      tokenBudget: 120_000,
      proxyUrl: "http://127.0.0.1:8787",
      timeoutMs: 30_000,
      force: true,
    });

    expect(plan.mode).toBe("truncate");
    expect(plan.appendMessages?.length).toBeGreaterThan(0);
    expect(plan.appendMessages?.length).toBeLessThan(branchMessages.length);
  });

  it("does not force truncate for a single-message branch", async () => {
    mockCompressResponse({
      messages: [{ role: "user", content: "only" }],
      tokens_before: 900_000,
      tokens_after: 900_000,
      tokens_saved: 0,
    });

    const plan = await planHeadroomCompaction({
      branchMessages: [
        {
          entryId: "entry-0",
          parentId: null,
          message: { role: "user", content: "only", timestamp: 0 },
        },
      ],
      tokenBudget: 120_000,
      proxyUrl: "http://127.0.0.1:8787",
      timeoutMs: 30_000,
      force: true,
    });

    expect(plan.mode).toBe("none");
  });

  it("requests safer durable hygiene defaults from the proxy", async () => {
    mockCompressResponse({
      messages: [{ role: "user", content: "hello" }],
      tokens_before: 1000,
      tokens_after: 1000,
      tokens_saved: 0,
    });

    await planHeadroomCompaction({
      branchMessages: [
        {
          entryId: "entry-0",
          parentId: null,
          message: { role: "user", content: "hello", timestamp: 0 },
        },
      ],
      tokenBudget: 120_000,
      proxyUrl: "http://127.0.0.1:8787",
      timeoutMs: 30_000,
    });

    const requestBody = JSON.parse(String(fetchMock.mock.calls[0]?.[1]?.body));
    expect(requestBody.config).toMatchObject({
      protect_recent: 2,
    });
  });

  it("forced truncate drops prefix messages, including protected tool payloads that fall before the cut", async () => {
    const imageMessage = {
      role: "toolResult",
      toolCallId: "call_img",
      toolName: "view_image",
      content: [{ type: "image", data: "abc123", mimeType: "image/png" }],
      timestamp: 1,
    };
    const branchMessages = [
      {
        entryId: "entry-0",
        parentId: null,
        message: imageMessage,
      },
      ...Array.from({ length: 99 }, (_, index) => ({
        entryId: `entry-${index + 1}`,
        parentId: index === 0 ? "entry-0" : `entry-${index}`,
        message: { role: "user", content: `msg-${index}`, timestamp: index + 2 },
      })),
    ];

    mockCompressResponse({
      messages: branchMessages.map((entry) => ({
        role: "user",
        content:
          typeof entry.message.content === "string"
            ? entry.message.content
            : JSON.stringify(entry.message.content),
      })),
      tokens_before: 900_000,
      tokens_after: 900_000,
      tokens_saved: 0,
    });

    const plan = await planHeadroomCompaction({
      branchMessages,
      tokenBudget: 120_000,
      proxyUrl: "http://127.0.0.1:8787",
      timeoutMs: 30_000,
      force: true,
    });

    expect(plan.mode).toBe("truncate");
    expect(plan.appendMessages?.some((message) => message.role === "toolResult")).toBe(false);
    expect(plan.truncateParentId).not.toBeNull();
  });

  it("skips replace compaction for image tool results", async () => {
    const imageMessage = {
      role: "toolResult",
      toolCallId: "call_img",
      toolName: "view_image",
      content: [{ type: "image", data: "abc123", mimeType: "image/png" }],
      timestamp: 2,
    };
    const branchMessages = [
      {
        entryId: "entry-0",
        parentId: null,
        message: { role: "user", content: "show screenshot", timestamp: 1 },
      },
      {
        entryId: "entry-1",
        parentId: "entry-0",
        message: imageMessage,
      },
    ];

    mockCompressResponse({
      messages: [
        { role: "user", content: "[compressed user turn]" },
        {
          role: "tool",
          content: "[compressed image summary]",
          tool_call_id: "call_img",
          name: "view_image",
        },
      ],
      tokens_before: 5000,
      tokens_after: 1200,
      tokens_saved: 3800,
    });

    const plan = await planHeadroomCompaction({
      branchMessages,
      tokenBudget: 120_000,
      proxyUrl: "http://127.0.0.1:8787",
      timeoutMs: 30_000,
      force: true,
    });

    expect(plan.mode).toBe("replace");
    expect(plan.replacements?.map((entry) => entry.entryId)).toEqual(["entry-0"]);
    expect(plan.replacements?.[0]?.message.content).toBe("[compressed user turn]");
  });

  it("plans truncate when compress returns fewer messages than the branch", async () => {
    const branchMessages = Array.from({ length: 50 }, (_, index) => ({
      entryId: `entry-${index}`,
      parentId: index === 0 ? null : `entry-${index - 1}`,
      message: { role: "user", content: `msg-${index}`, timestamp: index },
    }));

    mockCompressResponse({
      messages: branchMessages.slice(30).map((entry) => ({
        role: "user",
        content: entry.message.content,
      })),
      tokens_before: 50_000,
      tokens_after: 12_000,
      tokens_saved: 38_000,
    });

    const plan = await planHeadroomCompaction({
      branchMessages,
      tokenBudget: 120_000,
      proxyUrl: "http://127.0.0.1:8787",
      timeoutMs: 30_000,
    });

    expect(plan.mode).toBe("truncate");
    expect(plan.appendMessages?.length).toBe(20);
    expect(plan.truncateParentId).toBe("entry-29");
  });
});

describe("HeadroomContextEngine durable compaction", () => {
  it("returns compacted:false for small noop sessions without force", async () => {
    mockCompressResponse({
      messages: [{ role: "user", content: "hello" }],
      tokens_before: 100,
      tokens_after: 100,
      tokens_saved: 0,
    });
    sessionManagerMock.getBranch.mockReturnValue([
      {
        type: "message",
        id: "entry-1",
        parentId: null,
        message: { role: "user", content: "hello", timestamp: 1 },
      },
    ]);

    const engine = new HeadroomContextEngine({ persistentCompaction: "headroom" });
    (engine as { proxyUrl: string | null }).proxyUrl = "http://127.0.0.1:8787";

    await expect(
      engine.compact({
        sessionId: "session-1",
        sessionKey: "agent:main:main",
        sessionFile: "ignored",
        tokenBudget: 50_000,
      }),
    ).resolves.toMatchObject({
      ok: true,
      compacted: false,
      reason: "No durable compaction needed",
    });
  });

  it("plans replace compaction and persists via maintain()", async () => {
    mockCompressResponse({
      messages: [{ role: "user", content: "hi" }],
      tokens_before: 1000,
      tokens_after: 200,
      tokens_saved: 800,
    });
    sessionManagerMock.getBranch.mockReturnValue([
      {
        type: "message",
        id: "entry-1",
        parentId: null,
        message: { role: "user", content: "hello ".repeat(200), timestamp: 1 },
      },
    ]);

    const rewriteTranscriptEntries = vi.fn(async () => ({
      changed: true,
      bytesFreed: 500,
      rewrittenEntries: 1,
    }));

    const engine = new HeadroomContextEngine({ persistentCompaction: "headroom" });
    (engine as { proxyUrl: string | null }).proxyUrl = "http://127.0.0.1:8787";

    const compactResult = await engine.compact({
      sessionId: "session-1",
      sessionKey: "agent:main:main",
      sessionFile: "ignored",
      tokenBudget: 50_000,
      force: true,
    });

    expect(compactResult.compacted).toBe(true);
    expect(compactResult.result?.tokensBefore).toBe(1000);
    expect(compactResult.result?.tokensAfter).toBe(200);

    const maintainResult = await engine.maintain({
      sessionId: "session-1",
      sessionKey: "agent:main:main",
      sessionFile: "ignored",
      runtimeContext: { rewriteTranscriptEntries },
    });

    expect(maintainResult.changed).toBe(true);
    expect(rewriteTranscriptEntries).toHaveBeenCalledTimes(1);
  });
});

function createBranchingSessionManager(initialMessages: BranchMessageEntry[]) {
    const entries = new Map<
      string,
      {
        type: "message";
        id: string;
        parentId: string | null;
        message: unknown;
      }
    >();
    for (const entry of initialMessages) {
      entries.set(entry.entryId, {
        type: "message",
        id: entry.entryId,
        parentId: entry.parentId,
        message: entry.message,
      });
    }

    let leafId =
      initialMessages.length > 0
        ? initialMessages[initialMessages.length - 1]!.entryId
        : null;
    let appendParentId = leafId;

    return {
      getBranch() {
        const path: Array<{
          type: "message";
          id: string;
          parentId: string | null;
          message: unknown;
        }> = [];
        const seen = new Set<string>();
        let currentId = leafId;
        while (currentId && !seen.has(currentId)) {
          seen.add(currentId);
          const entry = entries.get(currentId);
          if (!entry) {
            break;
          }
          path.unshift(entry);
          currentId = entry.parentId;
        }
        return path;
      },
      branch(parentId: string) {
        leafId = parentId;
        appendParentId = parentId;
      },
      resetLeaf() {
        leafId = null;
        appendParentId = null;
      },
      appendMessage(message: unknown) {
        const id = `new-${entries.size}`;
        const entry = {
          type: "message" as const,
          id,
          parentId: appendParentId,
          message,
        };
        entries.set(id, entry);
        leafId = id;
        appendParentId = id;
        return entry;
      },
    };
}

describe("applyCompactionPlan truncate", () => {
  it("drops the prefix and keeps only appendMessages on the active branch", async () => {
    mockCompressResponse({
      messages: [{ role: "user", content: "hello" }],
      tokens_before: 900_000,
      tokens_after: 900_000,
      tokens_saved: 0,
    });

    const branchMessages = Array.from({ length: 100 }, (_, index) => ({
      entryId: `entry-${index}`,
      parentId: index === 0 ? null : `entry-${index - 1}`,
      message: {
        role: "user",
        content: `msg-${index}-${"x".repeat(120)}`,
        timestamp: index,
      },
    }));

    const plan = await planHeadroomCompaction({
      branchMessages,
      tokenBudget: 120_000,
      proxyUrl: "http://127.0.0.1:8787",
      timeoutMs: 30_000,
      force: true,
    });

    expect(plan.mode).toBe("truncate");
    expect(plan.appendMessages?.length).toBeLessThan(branchMessages.length);

    const sessionManager = createBranchingSessionManager(branchMessages);
    vi.mocked(SessionManager.open).mockReturnValue(sessionManager);
    const beforeCount = sessionManager.getBranch().length;
    expect(beforeCount).toBe(100);

    const result = await applyCompactionPlan({
      sessionTarget: { sessionId: "session-1" },
      plan,
      cwd: tmpdir(),
    });

    const afterBranch = sessionManager.getBranch();
    expect(afterBranch.length).toBe(plan.appendMessages?.length);
    expect(afterBranch.length).toBeLessThan(beforeCount);
    expect(result.changed).toBe(true);
    expect(result.bytesFreed).toBeGreaterThan(0);
    expect(sessionManager.getBranch().map((entry) => entry.message)).toEqual(
      plan.appendMessages,
    );
  });

  it("uses resetLeaf instead of branching from an ancestor", async () => {
    mockCompressResponse({
      messages: [{ role: "user", content: "hello" }],
      tokens_before: 900_000,
      tokens_after: 900_000,
      tokens_saved: 0,
    });

    // Above the 40-message keep floor so forced trimming actually drops a prefix.
    const branchMessages = Array.from({ length: 100 }, (_, index) => ({
      entryId: `entry-${index}`,
      parentId: index === 0 ? null : `entry-${index - 1}`,
      message: { role: "user", content: `msg-${index}`, timestamp: index },
    }));

    const plan = await planHeadroomCompaction({
      branchMessages,
      tokenBudget: 10_000,
      proxyUrl: "http://127.0.0.1:8787",
      timeoutMs: 30_000,
      force: true,
    });
    expect(plan.mode).toBe("truncate");

    const sessionManager = createBranchingSessionManager(branchMessages);
    const resetLeaf = vi.spyOn(sessionManager, "resetLeaf");
    const branch = vi.spyOn(sessionManager, "branch");
    vi.mocked(SessionManager.open).mockReturnValue(sessionManager);

    await applyCompactionPlan({
      sessionTarget: { sessionId: "session-1" },
      plan,
      cwd: tmpdir(),
    });

    expect(resetLeaf).toHaveBeenCalledTimes(1);
    expect(branch).not.toHaveBeenCalled();
  });

  it("applies compress-result truncate with resetLeaf and drops the prefix", async () => {
    const branchMessages = Array.from({ length: 50 }, (_, index) => ({
      entryId: `entry-${index}`,
      parentId: index === 0 ? null : `entry-${index - 1}`,
      message: {
        role: "user",
        content: `msg-${index}-${"y".repeat(100)}`,
        timestamp: index,
      },
    }));

    mockCompressResponse({
      messages: branchMessages.slice(30).map((entry) => ({
        role: "user",
        content: entry.message.content,
      })),
      tokens_before: 50_000,
      tokens_after: 12_000,
      tokens_saved: 38_000,
    });

    const plan = await planHeadroomCompaction({
      branchMessages,
      tokenBudget: 120_000,
      proxyUrl: "http://127.0.0.1:8787",
      timeoutMs: 30_000,
    });

    const sessionManager = createBranchingSessionManager(branchMessages);
    vi.mocked(SessionManager.open).mockReturnValue(sessionManager);

    const result = await applyCompactionPlan({
      sessionTarget: { sessionId: "session-1" },
      plan,
      cwd: tmpdir(),
    });

    expect(sessionManager.getBranch().length).toBe(20);
    expect(result.changed).toBe(true);
    expect(result.bytesFreed).toBeGreaterThan(0);
  });

  it("returns no-op when truncate plan has no appendMessages", async () => {
    const sessionManager = createBranchingSessionManager([
      {
        entryId: "entry-0",
        parentId: null,
        message: { role: "user", content: "hello", timestamp: 0 },
      },
    ]);
    vi.mocked(SessionManager.open).mockReturnValue(sessionManager);

    const result = await applyCompactionPlan({
      sessionTarget: { sessionId: "session-1" },
      plan: {
        mode: "truncate",
        tokensBefore: 100,
        tokensAfter: 50,
        appendMessages: [],
      },
      cwd: tmpdir(),
    });

    expect(result).toEqual({
      changed: false,
      bytesFreed: 0,
      rewrittenEntries: 0,
    });
    expect(sessionManager.getBranch()).toHaveLength(1);
  });

  it("truncates correctly when the first kept message is at the branch root", async () => {
    mockCompressResponse({
      messages: [{ role: "user", content: "hello" }],
      tokens_before: 900_000,
      tokens_after: 900_000,
      tokens_saved: 0,
    });

    const branchMessages = Array.from({ length: 100 }, (_, index) => ({
      entryId: `entry-${index}`,
      parentId: index === 0 ? null : `entry-${index - 1}`,
      message: { role: "user", content: `msg-${index}`, timestamp: index },
    }));

    const plan = await planHeadroomCompaction({
      branchMessages,
      tokenBudget: 120_000,
      proxyUrl: "http://127.0.0.1:8787",
      timeoutMs: 30_000,
      force: true,
    });

    expect(plan.truncateParentId).toBe("entry-59");

    const sessionManager = createBranchingSessionManager(branchMessages);
    vi.mocked(SessionManager.open).mockReturnValue(sessionManager);

    await applyCompactionPlan({
      sessionTarget: { sessionId: "session-1" },
      plan,
      cwd: tmpdir(),
    });

    expect(sessionManager.getBranch()[0]?.parentId).toBeNull();
    expect(sessionManager.getBranch().length).toBe(plan.appendMessages?.length);
  });

  it("would retain the prefix if truncate used branch(parentId) instead of resetLeaf", async () => {
    mockCompressResponse({
      messages: [{ role: "user", content: "hello" }],
      tokens_before: 900_000,
      tokens_after: 900_000,
      tokens_saved: 0,
    });

    const branchMessages = Array.from({ length: 100 }, (_, index) => ({
      entryId: `entry-${index}`,
      parentId: index === 0 ? null : `entry-${index - 1}`,
      message: { role: "user", content: `msg-${index}`, timestamp: index },
    }));

    const plan = await planHeadroomCompaction({
      branchMessages,
      tokenBudget: 120_000,
      proxyUrl: "http://127.0.0.1:8787",
      timeoutMs: 30_000,
      force: true,
    });

    const brokenManager = createBranchingSessionManager(branchMessages);
    brokenManager.branch(plan.truncateParentId!);
    for (const message of plan.appendMessages ?? []) {
      brokenManager.appendMessage(message);
    }
    expect(brokenManager.getBranch().length).toBeGreaterThan(
      plan.appendMessages?.length ?? 0,
    );

    const fixedManager = createBranchingSessionManager(branchMessages);
    fixedManager.resetLeaf();
    for (const message of plan.appendMessages ?? []) {
      fixedManager.appendMessage(message);
    }
    expect(fixedManager.getBranch().length).toBe(plan.appendMessages?.length);
  });
});

describe("truncate turn-boundary selection", () => {
  const ASSISTANT_META = {
    api: "anthropic-messages",
    provider: "anthropic",
    model: "claude-sonnet-4-5",
    stopReason: "toolUse",
  };

  /**
   * Reviewer fixture: an assistant toolCall + its toolResult followed by 39 user
   * messages (41 total). A raw suffix of 40 would start at the orphan toolResult.
   */
  function reviewerFixture(): BranchMessageEntry[] {
    const messages: unknown[] = [
      {
        role: "assistant",
        content: [{ type: "toolCall", id: "cut-call", name: "read", arguments: { path: "a" } }],
        ...ASSISTANT_META,
        timestamp: 0,
      },
      {
        role: "toolResult",
        toolCallId: "cut-call",
        toolName: "read",
        content: [{ type: "text", text: "file body" }],
        isError: false,
        timestamp: 1,
      },
      ...Array.from({ length: 39 }, (_, index) => ({
        role: "user",
        content: `user-${index}`,
        timestamp: index + 2,
      })),
    ];
    return messages.map((message, index) => ({
      entryId: `entry-${index}`,
      parentId: index === 0 ? null : `entry-${index - 1}`,
      message,
    }));
  }

  function reload(entries: Array<{ message: unknown }>): Array<Record<string, unknown>> {
    return entries.map((entry) => entry.message as Record<string, unknown>);
  }

  it("forced trimming never starts the kept tail at an orphan toolResult (41-message fixture)", async () => {
    mockCompressResponse({
      messages: [{ role: "user", content: "hello" }],
      tokens_before: 900_000,
      tokens_after: 900_000,
      tokens_saved: 0,
    });
    const branchMessages = reviewerFixture();

    const plan = await planHeadroomCompaction({
      branchMessages,
      tokenBudget: 120_000,
      proxyUrl: "http://127.0.0.1:8787",
      timeoutMs: 30_000,
      force: true,
    });

    expect(plan.mode).toBe("truncate");
    const kept = plan.appendMessages as Array<Record<string, unknown>>;
    // The raw 40-suffix would begin at entry-1 (the toolResult); the aligned cut
    // begins at the first user turn instead.
    expect(kept).toHaveLength(39);
    expect(kept[0]).toMatchObject({ role: "user", content: "user-0" });
    expect(kept.some((m) => m.role === "toolResult")).toBe(false);
    expect(plan.truncateParentId).toBe("entry-1");

    const sessionManager = createBranchingSessionManager(branchMessages);
    vi.mocked(SessionManager.open).mockReturnValue(sessionManager);
    await applyCompactionPlan({ sessionTarget: { sessionId: "session-1" }, plan, cwd: tmpdir() });

    const reloaded = reload(sessionManager.getBranch());
    expect(reloaded).toHaveLength(39);
    expect(reloaded[0]).toMatchObject({ role: "user", content: "user-0" });
    expect(reloaded.at(-1)).toMatchObject({ role: "user", content: "user-38" });
    expect(reloaded.every((m) => m.role === "user")).toBe(true);
  });

  it("keeps whole tool groups when the cut falls inside a multi-call turn", async () => {
    mockCompressResponse({
      messages: [{ role: "user", content: "hello" }],
      tokens_before: 900_000,
      tokens_after: 900_000,
      tokens_saved: 0,
    });
    // 30 user turns, each: user -> assistant(2 toolCalls) -> toolResult -> toolResult (120 msgs).
    const messages: unknown[] = [];
    for (let turn = 0; turn < 30; turn += 1) {
      messages.push({ role: "user", content: `ask-${turn}`, timestamp: turn * 4 });
      messages.push({
        role: "assistant",
        content: [
          { type: "toolCall", id: `c${turn}a`, name: "read", arguments: {} },
          { type: "toolCall", id: `c${turn}b`, name: "exec", arguments: {} },
        ],
        ...ASSISTANT_META,
        timestamp: turn * 4 + 1,
      });
      messages.push({ role: "toolResult", toolCallId: `c${turn}a`, toolName: "read", content: "r", isError: false, timestamp: turn * 4 + 2 });
      messages.push({ role: "toolResult", toolCallId: `c${turn}b`, toolName: "exec", content: "e", isError: false, timestamp: turn * 4 + 3 });
    }
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

    expect(plan.mode).toBe("truncate");
    const kept = plan.appendMessages as Array<Record<string, unknown>>;
    expect(kept[0]?.role).toBe("user");
    expect(kept.length % 4).toBe(0);

    const calls = new Set<string>();
    for (const m of kept) {
      if (m.role === "assistant") {
        for (const b of m.content as Array<{ id: string }>) calls.add(b.id);
      }
      if (m.role === "toolResult") {
        expect(calls.has(m.toolCallId as string)).toBe(true);
      }
    }
  });

  it("re-aligns a compress-result truncate whose proxy cut landed on a toolResult", async () => {
    const branchMessages = reviewerFixture();
    // Proxy dropped exactly the first message: its window starts at the toolResult.
    mockCompressResponse({
      messages: branchMessages.slice(1).map((entry) => {
        const m = entry.message as Record<string, unknown>;
        return m.role === "toolResult"
          ? { role: "tool", tool_call_id: "cut-call", name: "read", content: "file body" }
          : { role: "user", content: m.content };
      }),
      tokens_before: 900_000,
      tokens_after: 300_000,
      tokens_saved: 600_000,
    });

    const plan = await planHeadroomCompaction({
      branchMessages,
      tokenBudget: 120_000,
      proxyUrl: "http://127.0.0.1:8787",
      timeoutMs: 30_000,
    });

    expect(plan.mode).toBe("truncate");
    const kept = plan.appendMessages as Array<Record<string, unknown>>;
    expect(kept[0]).toMatchObject({ role: "user", content: "user-0" });
    expect(kept.some((m) => m.role === "toolResult")).toBe(false);
  });

  it("refuses to truncate when the transcript has no turn boundary", async () => {
    mockCompressResponse({
      messages: [{ role: "user", content: "hello" }],
      tokens_before: 900_000,
      tokens_after: 900_000,
      tokens_saved: 0,
    });
    const branchMessages = Array.from({ length: 60 }, (_, index) => ({
      entryId: `entry-${index}`,
      parentId: index === 0 ? null : `entry-${index - 1}`,
      message:
        index % 2 === 0
          ? {
              role: "assistant",
              content: [{ type: "toolCall", id: `c${index}`, name: "read", arguments: {} }],
              ...ASSISTANT_META,
              timestamp: index,
            }
          : { role: "toolResult", toolCallId: `c${index - 1}`, toolName: "read", content: "r", isError: false, timestamp: index },
    }));

    const plan = await planHeadroomCompaction({
      branchMessages,
      tokenBudget: 120_000,
      proxyUrl: "http://127.0.0.1:8787",
      timeoutMs: 30_000,
      force: true,
    });
    expect(plan.mode).toBe("none");
  });
});
