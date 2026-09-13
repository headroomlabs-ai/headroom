/**
 * Durable transcript compaction for Headroom — load branch, compress, persist.
 */

/* eslint-disable @typescript-eslint/no-explicit-any */

import { resolveDurableCompressConfig } from "./compress-request-config.js";
import { messageHasProtectedToolPayload } from "./content-blocks.js";
import { agentToOpenAI, openAIToAgent } from "./convert.js";
import { dropOrphanToolResults, selectTruncateStart } from "./truncate-boundary.js";

type SessionManagerLike = {
  getBranch(): unknown[];
  branch(parentId: string): void;
  resetLeaf(): void;
  appendMessage(message: unknown): unknown;
};

type SessionManagerConstructor = {
  open(target: ReturnType<typeof resolveSessionTarget>, cwd?: string): SessionManagerLike;
};

async function loadOpenClawSessionManager(): Promise<SessionManagerConstructor> {
  const mod = await import("openclaw/plugin-sdk/agent-sessions");
  return mod.SessionManager as SessionManagerConstructor;
}

export async function loadBranchMessagesFromSession(
  sessionTarget: ReturnType<typeof resolveSessionTarget>,
  cwd?: string,
): Promise<BranchMessageEntry[]> {
  const SessionManager = await loadOpenClawSessionManager();
  const sessionManager = SessionManager.open(sessionTarget, cwd);
  return extractBranchMessages(sessionManager);
}

export interface BranchMessageEntry {
  entryId: string;
  parentId: string | null;
  message: any;
}

export interface CompactionPlan {
  tokensBefore: number;
  tokensAfter: number;
  mode: "none" | "replace" | "truncate";
  replacements?: Array<{ entryId: string; message: any }>;
  truncateParentId?: string | null;
  appendMessages?: any[];
}

export interface PendingCompaction extends CompactionPlan {
  sessionId: string;
}

function withTimeout<T>(
  promise: Promise<T>,
  ms: number,
  abortSignal?: AbortSignal,
): Promise<T> {
  if (abortSignal?.aborted) {
    return Promise.reject(new Error("compaction aborted"));
  }
  let timerId: ReturnType<typeof setTimeout> | undefined;
  const onAbort = () => {
    if (timerId !== undefined) clearTimeout(timerId);
  };
  abortSignal?.addEventListener("abort", onAbort, { once: true });
  const timer = new Promise<never>((_, reject) => {
    timerId = setTimeout(
      () => reject(new Error(`headroom compress() timed out after ${ms}ms`)),
      ms,
    );
  });
  return Promise.race([promise, timer]).finally(() => {
    if (timerId !== undefined) clearTimeout(timerId);
    abortSignal?.removeEventListener("abort", onAbort);
  });
}

function estimateMessageBytes(message: any): number {
  return Buffer.byteLength(JSON.stringify(message), "utf8");
}

function mergeCompressedMessage(original: any, compressed: any): any {
  return {
    ...original,
    ...compressed,
    role: original.role ?? compressed.role,
    timestamp: original.timestamp ?? compressed.timestamp,
  };
}

function messagePayloadChanged(before: any, after: any): boolean {
  return estimateMessageBytes(before) !== estimateMessageBytes(after);
}

export function extractBranchMessages(sessionManager: SessionManagerLike): BranchMessageEntry[] {
  return sessionManager
    .getBranch()
    .flatMap((entry: any) =>
      entry.type === "message"
        ? [
            {
              entryId: entry.id as string,
              parentId: (entry.parentId as string | null | undefined) ?? null,
              message: entry.message,
            },
          ]
        : [],
    );
}

export function resolveSessionTarget(params: {
  sessionId: string;
  sessionKey?: string;
  sessionTarget?: {
    agentId?: string;
    sessionId?: string;
    sessionKey?: string;
    storePath?: string;
  };
}): {
  agentId?: string;
  sessionId: string;
  sessionKey?: string;
  storePath?: string;
} {
  const target = params.sessionTarget;
  return {
    agentId: target?.agentId,
    sessionId: target?.sessionId ?? params.sessionId,
    sessionKey: target?.sessionKey ?? params.sessionKey,
    storePath: target?.storePath,
  };
}

export async function planHeadroomCompaction(options: {
  branchMessages: BranchMessageEntry[];
  tokenBudget: number | undefined;
  proxyUrl: string;
  model?: string;
  timeoutMs: number;
  abortSignal?: AbortSignal;
  force?: boolean;
  /** Tool names whose results are restored verbatim after compression. */
  protectedToolNames?: ReadonlySet<string>;
}): Promise<CompactionPlan> {
  const originals = options.branchMessages.map((entry) => entry.message);
  if (originals.length === 0) {
    return { mode: "none", tokensBefore: 0, tokensAfter: 0 };
  }

  const openaiMessages = agentToOpenAI(originals);
  const result = await withTimeout(
    compressDurableTranscript(openaiMessages, {
      model: options.model ?? "claude-sonnet-4-5",
      proxyUrl: options.proxyUrl,
      tokenBudget: options.tokenBudget,
      abortSignal: options.abortSignal,
    }),
    options.timeoutMs,
    options.abortSignal,
  );

  const tokensBefore = result.tokensBefore;
  const tokensAfter = result.tokensAfter;

  let plan = buildCompactionPlanFromCompressResult({
    branchMessages: options.branchMessages,
    originals,
    result,
    protectedToolNames: options.protectedToolNames,
  });

  if (
    plan.mode === "none" &&
    (options.force === true ||
      (options.tokenBudget !== undefined &&
        tokensBefore > Math.max(options.tokenBudget, 1) * 1.05) ||
      tokensBefore > 400_000)
  ) {
    plan = buildForcedTruncatePlan({
      branchMessages: options.branchMessages,
      tokenBudget: options.tokenBudget,
      tokensBefore,
    });
  }

  return plan;
}

interface DurableCompressResult {
  messages: ReturnType<typeof agentToOpenAI>;
  tokensBefore: number;
  tokensAfter: number;
  tokensSaved: number;
  compressed: boolean;
}

async function compressDurableTranscript(
  messages: ReturnType<typeof agentToOpenAI>,
  options: {
    proxyUrl: string;
    model: string;
    tokenBudget?: number;
    abortSignal?: AbortSignal;
  },
): Promise<DurableCompressResult> {
  const body: Record<string, unknown> = {
    messages,
    model: options.model,
    config: resolveDurableCompressConfig(),
  };
  if (options.tokenBudget !== undefined) {
    body.token_budget = options.tokenBudget;
  }

  const response = await fetch(`${options.proxyUrl.replace(/\/$/, "")}/v1/compress`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "x-headroom-mode": "token",
    },
    body: JSON.stringify(body),
    signal: options.abortSignal,
  });

  if (!response.ok) {
    const detail = await response.text().catch(() => "");
    throw new Error(`headroom /v1/compress failed (${response.status}): ${detail.slice(0, 300)}`);
  }

  const data = (await response.json()) as {
    messages: ReturnType<typeof agentToOpenAI>;
    tokens_before: number;
    tokens_after: number;
    tokens_saved: number;
  };

  return {
    messages: data.messages,
    tokensBefore: data.tokens_before,
    tokensAfter: data.tokens_after,
    tokensSaved: data.tokens_saved,
    compressed: data.tokens_saved > 0 || data.tokens_after < data.tokens_before,
  };
}

function buildCompactionPlanFromCompressResult(options: {
  branchMessages: BranchMessageEntry[];
  originals: any[];
  result: DurableCompressResult;
  protectedToolNames?: ReadonlySet<string>;
}): CompactionPlan {
  const { branchMessages, originals, result, protectedToolNames } = options;
  const tokensBefore = result.tokensBefore;
  const tokensAfter = result.tokensAfter;

  if (!result.compressed || result.tokensSaved <= 0) {
    return { mode: "none", tokensBefore, tokensAfter: tokensBefore };
  }

  const compressedAgent = openAIToAgent(result.messages, { originals, protectedToolNames });

  if (compressedAgent.length === originals.length) {
    const replacements = branchMessages.flatMap((entry, index) => {
      if (messageHasProtectedToolPayload(entry.message)) {
        return [];
      }
      const merged = mergeCompressedMessage(entry.message, compressedAgent[index]);
      if (!messagePayloadChanged(entry.message, merged)) {
        return [];
      }
      return [{ entryId: entry.entryId, message: merged }];
    });

    if (replacements.length === 0) {
      return { mode: "none", tokensBefore, tokensAfter };
    }

    return {
      mode: "replace",
      tokensBefore,
      tokensAfter,
      replacements,
    };
  }

  if (compressedAgent.length < originals.length) {
    // The proxy's rolling window drops a prefix. Re-align the cut to a turn
    // boundary so the durable tail never starts mid-turn or at an orphan
    // toolResult; messages between the aligned start and the proxy's cut are
    // kept verbatim from the originals.
    const dropped = originals.length - compressedAgent.length;
    const selection = selectTruncateStart(originals, dropped);
    if (!selection || selection.startIndex === 0) {
      return { mode: "none", tokensBefore, tokensAfter: tokensBefore };
    }
    const { startIndex } = selection;
    const tail =
      startIndex >= dropped
        ? compressedAgent.slice(startIndex - dropped)
        : [...originals.slice(startIndex, dropped), ...compressedAgent];
    const appendMessages = dropOrphanToolResults(tail);
    if (appendMessages.length === 0) {
      return { mode: "none", tokensBefore, tokensAfter: tokensBefore };
    }

    return {
      mode: "truncate",
      tokensBefore,
      tokensAfter,
      truncateParentId: branchMessages[startIndex]!.parentId,
      appendMessages,
    };
  }

  return { mode: "none", tokensBefore, tokensAfter: tokensBefore };
}

function buildForcedTruncatePlan(options: {
  branchMessages: BranchMessageEntry[];
  tokenBudget?: number;
  tokensBefore: number;
}): CompactionPlan {
  const { branchMessages, tokenBudget, tokensBefore } = options;
  if (branchMessages.length <= 1) {
    return { mode: "none", tokensBefore, tokensAfter: tokensBefore };
  }

  const ratio =
    tokenBudget !== undefined && tokenBudget > 0
      ? Math.min(0.85, tokenBudget / Math.max(tokensBefore, 1))
      : 0.35;
  const keepCount = Math.max(40, Math.min(branchMessages.length, Math.ceil(branchMessages.length * ratio)));
  const desiredStart = branchMessages.length - keepCount;

  // Never cut mid-turn: move the cut to the nearest turn start (OpenClaw's own
  // compaction rule) so the tail never begins with an orphan toolResult or an
  // assistant continuation, and always retains the latest user turn.
  const messages = branchMessages.map((entry) => entry.message);
  const selection = selectTruncateStart(messages, desiredStart);
  if (!selection || selection.startIndex === 0) {
    return { mode: "none", tokensBefore, tokensAfter: tokensBefore };
  }
  const { startIndex } = selection;
  const appendMessages = dropOrphanToolResults(messages.slice(startIndex));
  if (appendMessages.length === 0) {
    return { mode: "none", tokensBefore, tokensAfter: tokensBefore };
  }

  const estimatedAfter = Math.max(
    Math.floor(tokensBefore * (appendMessages.length / branchMessages.length)),
    Math.floor(tokensBefore * 0.15),
  );

  return {
    mode: "truncate",
    tokensBefore,
    tokensAfter: estimatedAfter,
    truncateParentId: branchMessages[startIndex]!.parentId,
    appendMessages,
  };
}

export async function applyCompactionPlan(options: {
  sessionTarget: ReturnType<typeof resolveSessionTarget>;
  plan: CompactionPlan;
  rewriteTranscriptEntries?: (request: {
    replacements: Array<{ entryId: string; message: any }>;
  }) => Promise<{ changed: boolean; bytesFreed: number; rewrittenEntries: number }>;
  cwd?: string;
}): Promise<{ changed: boolean; bytesFreed: number; rewrittenEntries: number }> {
  if (options.plan.mode === "none") {
    return { changed: false, bytesFreed: 0, rewrittenEntries: 0 };
  }

  if (options.plan.mode === "replace") {
    if (!options.rewriteTranscriptEntries || !options.plan.replacements?.length) {
      return { changed: false, bytesFreed: 0, rewrittenEntries: 0 };
    }
    return await options.rewriteTranscriptEntries({
      replacements: options.plan.replacements,
    });
  }

  const appendMessages = options.plan.appendMessages ?? [];
  if (appendMessages.length === 0) {
    return { changed: false, bytesFreed: 0, rewrittenEntries: 0 };
  }

  const SessionManager = await loadOpenClawSessionManager();
  const sessionManager = SessionManager.open(options.sessionTarget, options.cwd);
  const beforeBytes = sessionManager
    .getBranch()
    .reduce((total: number, entry: any) => {
      if (entry.type !== "message") return total;
      return total + estimateMessageBytes(entry.message);
    }, 0);

  // Truncate must drop the prefix, not branch from an ancestor. OpenClaw's
  // branch(parentId) keeps the full path to that parent; resetLeaf() starts a
  // fresh tail so only appendMessages remain on the active branch.
  sessionManager.resetLeaf();

  for (const message of appendMessages) {
    sessionManager.appendMessage(message);
  }

  const afterBytes = sessionManager
    .getBranch()
    .reduce((total: number, entry: any) => {
      if (entry.type !== "message") return total;
      return total + estimateMessageBytes(entry.message);
    }, 0);

  return {
    changed: true,
    bytesFreed: Math.max(0, beforeBytes - afterBytes),
    rewrittenEntries: appendMessages.length,
  };
}
