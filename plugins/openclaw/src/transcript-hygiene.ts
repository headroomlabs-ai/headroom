/**
 * Progressive in-place transcript compression (replace mode only).
 *
 * Used by hybrid compaction (pre-pass before OpenClaw LLM compact) and turn-end
 * maintenance when transcriptHygiene is enabled. Never truncates — OpenClaw owns
 * durable history pruning.
 */

import {
  applyCompactionPlan,
  loadBranchMessagesFromSession,
  planHeadroomCompaction,
  resolveSessionTarget,
  type CompactionPlan,
} from "./compaction.js";
import { estimateRoughTokens } from "./convert.js";

export interface TranscriptHygieneConfig {
  /** Turn-end replace hygiene (hybrid default: true). */
  transcriptHygiene?: boolean | TranscriptHygieneSettings;
}

export interface TranscriptHygieneSettings {
  enabled?: boolean;
  /** Run replace hygiene when branch tokens exceed this (default: half of token budget or 400k). */
  softThresholdTokens?: number;
  /** Skip another hygiene rewrite on the same session within this window (default: 30s). */
  debounceMs?: number;
}

export interface TranscriptHygieneRuntimeParams {
  sessionId: string;
  sessionKey?: string;
  sessionTarget?: {
    agentId?: string;
    sessionId?: string;
    sessionKey?: string;
    storePath?: string;
  };
  tokenBudget?: number;
  runtimeSettings?: { resolvedModel?: string | null; promptTokenBudget?: number };
  runtimeContext?: {
    tokenBudget?: number;
    cwd?: string;
    workspaceDir?: string;
    rewriteTranscriptEntries?: (request: {
      replacements: Array<{ entryId: string; message: unknown }>;
    }) => Promise<{ changed: boolean; bytesFreed: number; rewrittenEntries: number }>;
  };
  abortSignal?: AbortSignal;
}

const DEFAULT_HYGIENE_DEBOUNCE_MS = 30_000;

export function resolveTranscriptHygieneSettings(
  config: TranscriptHygieneConfig,
  persistentCompactionMode: "headroom" | "openclaw" | "hybrid",
): Required<TranscriptHygieneSettings> {
  const raw = config.transcriptHygiene;
  let enabled: boolean;
  let softThresholdTokens: number | undefined;
  let debounceMs: number | undefined;

  if (typeof raw === "boolean") {
    enabled = raw;
  } else if (raw && typeof raw === "object") {
    enabled = raw.enabled ?? persistentCompactionMode === "hybrid";
    softThresholdTokens = raw.softThresholdTokens;
    debounceMs = raw.debounceMs;
  } else {
    enabled = persistentCompactionMode === "hybrid";
  }

  return {
    enabled,
    softThresholdTokens: softThresholdTokens ?? 0,
    debounceMs: debounceMs ?? DEFAULT_HYGIENE_DEBOUNCE_MS,
  };
}

export function resolveSoftThresholdTokens(
  settings: TranscriptHygieneSettings,
  tokenBudget: number | undefined,
): number {
  if (settings.softThresholdTokens !== undefined && settings.softThresholdTokens > 0) {
    return Math.floor(settings.softThresholdTokens);
  }
  if (tokenBudget !== undefined && tokenBudget > 0) {
    return Math.max(100_000, Math.floor(tokenBudget * 0.5));
  }
  return 400_000;
}

function isReplaceHygienePlan(plan: CompactionPlan): plan is CompactionPlan & {
  mode: "replace";
  replacements: Array<{ entryId: string; message: unknown }>;
} {
  return plan.mode === "replace" && (plan.replacements?.length ?? 0) > 0;
}

/** Algorithmic in-place tool/message shrink. Replace mode only — no truncate. */
export async function runTranscriptReplaceHygiene(options: {
  params: TranscriptHygieneRuntimeParams;
  proxyUrl: string;
  timeoutMs: number;
  softThresholdTokens: number;
  force?: boolean;
  model?: string;
}): Promise<{
  changed: boolean;
  bytesFreed: number;
  rewrittenEntries: number;
  tokensBefore?: number;
  tokensAfter?: number;
  reason?: string;
}> {
  options.params.abortSignal?.throwIfAborted();

  const sessionTarget = resolveSessionTarget(options.params);
  const tokenBudget =
    options.params.tokenBudget ??
    options.params.runtimeContext?.tokenBudget ??
    options.params.runtimeSettings?.promptTokenBudget;

  const branchMessages = await loadBranchMessagesFromSession(
    sessionTarget,
    options.params.runtimeContext?.cwd ?? options.params.runtimeContext?.workspaceDir,
  );
  if (branchMessages.length === 0) {
    return { changed: false, bytesFreed: 0, rewrittenEntries: 0, reason: "empty transcript" };
  }

  const roughTokens = estimateRoughTokens(branchMessages.map((entry) => entry.message));
  if (!options.force && roughTokens < options.softThresholdTokens) {
    return {
      changed: false,
      bytesFreed: 0,
      rewrittenEntries: 0,
      tokensBefore: roughTokens,
      tokensAfter: roughTokens,
      reason: "below soft threshold",
    };
  }

  const plan = await planHeadroomCompaction({
    branchMessages,
    tokenBudget,
    proxyUrl: options.proxyUrl,
    model: options.model ?? options.params.runtimeSettings?.resolvedModel ?? undefined,
    timeoutMs: options.timeoutMs,
    abortSignal: options.params.abortSignal,
    force: false,
  });

  if (!isReplaceHygienePlan(plan)) {
    return {
      changed: false,
      bytesFreed: 0,
      rewrittenEntries: 0,
      tokensBefore: plan.tokensBefore,
      tokensAfter: plan.tokensAfter,
      reason: plan.mode === "none" ? "no replace candidates" : "truncate skipped for hygiene",
    };
  }

  const result = await applyCompactionPlan({
    sessionTarget,
    plan,
    rewriteTranscriptEntries: options.params.runtimeContext?.rewriteTranscriptEntries,
    cwd: options.params.runtimeContext?.cwd ?? options.params.runtimeContext?.workspaceDir,
  });

  return {
    ...result,
    tokensBefore: plan.tokensBefore,
    tokensAfter: plan.tokensAfter,
  };
}
