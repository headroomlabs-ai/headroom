/**
 * HeadroomContextEngine — ContextEngine implementation for OpenClaw.
 *
 * Compresses tool outputs and conversation context using the Headroom proxy.
 * Zero LLM calls — all compression is algorithmic (SmartCrusher, ContentRouter, etc.)
 */

/* eslint-disable @typescript-eslint/no-explicit-any */

import { compress } from "headroom-ai";
import {
  applyCompactionPlan,
  loadBranchMessagesFromSession,
  planHeadroomCompaction,
  resolveSessionTarget,
  type PendingCompaction,
} from "./compaction.js";
import { ProxyManager, defaultLogger, type ProxyManagerConfig, type ProxyManagerLogger } from "./proxy-manager.js";
import {
  TurnAdvancementStore,
  resolveTurnAdvancementStorePath,
} from "./turn-advancement-store.js";
import {
  agentToOpenAI,
  estimateRoughTokens,
  normalizeAgentMessages,
  openAIToAgent,
} from "./convert.js";
import {
  delegatesPersistentCompaction,
  ownsPersistentCompaction,
  resolvePersistentCompactionMode,
  type PersistentCompactionConfig,
  type PersistentCompactionMode,
} from "./compaction-mode.js";
import {
  delegateCompactionToRuntime,
  type OpenClawCompactParams,
  type OpenClawCompactResult,
} from "./openclaw-compaction.js";
import { HygieneDebounceTracker } from "./hygiene-debounce.js";
import {
  resolveSoftThresholdTokens,
  resolveTranscriptHygieneSettings,
  runTranscriptReplaceHygiene,
  type TranscriptHygieneConfig,
  type TranscriptHygieneRuntimeParams,
} from "./transcript-hygiene.js";
import { awaitTranscriptProjectionSettle } from "./transcript-projection.js";
import {
  resolveAssembleCompressConfig,
  type CompressRequestConfig,
} from "./compress-request-config.js";
import { decideAssembleSkip, providerIdFromRuntimeSettings } from "./assemble-skip.js";
import { normalizeProtectedToolNames } from "./tool-names.js";

type HeadroomCompactParams = OpenClawCompactParams & {
  sessionTarget?: {
    agentId?: string;
    sessionId?: string;
    sessionKey?: string;
    storePath?: string;
  };
  runtimeContext?: {
    tokenBudget?: number;
    cwd?: string;
    workspaceDir?: string;
    rewriteTranscriptEntries?: (request: {
      replacements: Array<{ entryId: string; message: unknown }>;
    }) => Promise<{ changed: boolean; bytesFreed: number; rewrittenEntries: number }>;
  };
  runtimeSettings?: { resolvedModel?: string | null; promptTokenBudget?: number };
};

/** Race a promise against a timeout and always release the timer. */
function withTimeout<T>(promise: Promise<T>, ms: number): Promise<T> {
  let timerId: ReturnType<typeof setTimeout> | undefined;
  const timer = new Promise<never>((_, reject) => {
    timerId = setTimeout(() => reject(new Error(`headroom compress() timed out after ${ms}ms`)), ms);
  });
  return Promise.race([promise, timer]).finally(() => {
    if (timerId !== undefined) clearTimeout(timerId);
  });
}

export const DEFAULT_ASSEMBLE_SKIP_BUDGET_RATIO = 0.7;
export const DEFAULT_ASSEMBLE_RESERVE_TOKENS = 20_000;

/** Non-negative integer reserve; invalid values fall back to the default. */
export function resolveAssembleReserveTokens(value: number | undefined): number {
  if (typeof value !== "number" || !Number.isFinite(value) || value < 0) {
    return DEFAULT_ASSEMBLE_RESERVE_TOKENS;
  }
  return Math.floor(value);
}

/**
 * Token count below which `assemble()` skips proxy compression:
 * `(tokenBudget − reserve) × ratio`, never below 1.
 */
export function resolveAssembleSkipThreshold(params: {
  tokenBudget: number;
  ratio?: number;
  reserveTokens?: number;
}): number {
  const ratio = resolveAssembleSkipBudgetRatio(params.ratio);
  const reserve = resolveAssembleReserveTokens(params.reserveTokens);
  return Math.max(1, Math.round((params.tokenBudget - reserve) * ratio));
}

/** Clamp a configured skip ratio into (0, 1]; invalid values fall back to the default. */
export function resolveAssembleSkipBudgetRatio(value: number | undefined): number {
  if (typeof value !== "number" || !Number.isFinite(value) || value <= 0 || value > 1) {
    return DEFAULT_ASSEMBLE_SKIP_BUDGET_RATIO;
  }
  return value;
}

export interface HeadroomEngineConfig
  extends ProxyManagerConfig, PersistentCompactionConfig, TranscriptHygieneConfig {
  enabled?: boolean;
  requestTimeoutMs?: number;
  circuitBreakerThreshold?: number;
  circuitBreakerCooldownMs?: number;
  /**
   * Max milliseconds to wait for OpenClaw transcript projection to settle after
   * hygiene rewrites. Set 0 to disable. Default: 120s.
   */
  transcriptProjectionWaitMs?: number;
  /** Override durable turn-advancement store path (primarily for tests). */
  turnAdvancementStorePath?: string;
  /** Per-turn `/v1/compress` config (default: `{ protect_recent: 2 }`). */
  assembleCompressConfig?: CompressRequestConfig;
  /**
   * `assemble()` skips proxy compression while the rough token estimate of the
   * history is below `tokenBudget * assembleSkipBudgetRatio`. OpenClaw's own
   * overflow precheck fires at roughly `budget - reserve - systemPrompt`, so
   * this must sit low enough that compression runs before native compaction on
   * ~200k windows, yet high enough that 100–200k sessions in 1M windows are not
   * compressed on every turn. Range (0, 1]; default 0.7.
   */
  assembleSkipBudgetRatio?: number;
  /**
   * Tokens subtracted from `tokenBudget` before the skip ratio is applied.
   * OpenClaw hands `assemble()` the full model window as `tokenBudget`, but
   * the prompt it actually builds also carries the system prompt, tool schemas
   * and a compaction reserve (≥ 20 000 tokens). Set this to roughly
   * `reserve + system prompt tokens` so compression starts before the native
   * overflow precheck. Default 20 000 (OpenClaw's reserve floor).
   */
  assembleReserveTokens?: number;
  /**
   * Tool names (or `*` globs) whose results are restored verbatim after every
   * `/v1/compress` round trip (assemble and durable compaction), whatever the
   * proxy returned for them. Deferred `tool_call` wrappers are resolved to the
   * real tool first, so an MCP tool can be named as `analyze_video`,
   * `<server>__analyze_video` or `mcp__<server>__analyze_video`. Default: none.
   */
  protectToolResults?: string[];
  /**
   * Skip `assemble()` compression when the current model's provider is routed
   * through the proxy via `gatewayProviderIds` (avoids double compression — see
   * BUG-6). Providers that are not routed still get `assemble()` compression.
   * Default: false.
   */
  skipAssembleWhenGatewayRouted?: boolean;
  gatewayProviderIds?: string[];
  routeCodexViaProxy?: boolean;
}

/** OpenClaw 2026.9.x turn contract — required even when OpenClaw owns compaction (hybrid/openclaw). */
const HEADROOM_TRANSCRIPT_SEMANTICS = {
  currentTurnFence: "before-current-turn-entry-v1" as const,
  turnAdvancementIdempotency: "atomic-idempotent-v1" as const,
};

export class HeadroomContextEngine {
  get info() {
    const ownsCompaction = ownsPersistentCompaction(this.persistentCompactionMode);
    return {
      id: "headroom",
      name: "Headroom Context Compression",
      version: "0.1.0",
      ownsCompaction,
      transcriptSemantics: HEADROOM_TRANSCRIPT_SEMANTICS,
    };
  }

  private proxyManager: ProxyManager;
  private proxyUrl: string | null = null;
  private config: HeadroomEngineConfig;
  private readonly persistentCompactionMode: PersistentCompactionMode;
  private logger: ProxyManagerLogger;
  private proxyReadyListeners = new Set<(proxyUrl: string) => void | Promise<void>>();
  private proxyStartupPromise: Promise<string> | null = null;
  private proxyStartupError: unknown = null;
  private stats = {
    totalCompressions: 0,
    totalTokensSaved: 0,
    totalTokensBefore: 0,
    compactions: 0,
    hygieneRuns: 0,
    hygieneBytesFreed: 0,
  };
  private readonly transcriptHygieneSettings: ReturnType<typeof resolveTranscriptHygieneSettings>;
  private circuit = { errors: 0, openUntilMs: 0 };
  private pendingCompactions = new Map<string, PendingCompaction>();
  private turnAdvancementStores = new Map<string, TurnAdvancementStore>();
  private readonly hygieneDebounce = new HygieneDebounceTracker();
  private readonly transcriptProjectionWaitMs: number;
  private readonly protectedToolNames: ReadonlySet<string>;

  constructor(config: HeadroomEngineConfig = {}, logger?: ProxyManagerLogger) {
    this.config = config;
    this.persistentCompactionMode = resolvePersistentCompactionMode(config);
    this.transcriptHygieneSettings = resolveTranscriptHygieneSettings(
      config,
      this.persistentCompactionMode,
    );
    this.transcriptProjectionWaitMs = config.transcriptProjectionWaitMs ?? 120_000;
    this.protectedToolNames = normalizeProtectedToolNames(config.protectToolResults);
    this.logger = logger ?? defaultLogger;
    this.proxyManager = new ProxyManager(config, this.logger);
  }

  // === ContextEngine Lifecycle ===

  async bootstrap(params: {
    sessionId: string;
    sessionKey?: string;
    sessionFile: string;
  }): Promise<{ bootstrapped: boolean; reason?: string }> {
    if (this.config.enabled === false) {
      return { bootstrapped: false, reason: "disabled" };
    }

    this.ensureProxyStarted();
    return { bootstrapped: true, reason: "proxy startup scheduled" };
  }

  async ingest(params: {
    sessionId: string;
    message: any;
    isHeartbeat?: boolean;
  }): Promise<{ ingested: boolean }> {
    // No-op: OpenClaw's runtime stores messages. We don't need a separate store.
    return { ingested: true };
  }

  async ingestBatch?(params: {
    sessionId: string;
    messages: any[];
    isHeartbeat?: boolean;
  }): Promise<{ ingestedCount: number }> {
    return { ingestedCount: params.messages.length };
  }

  /**
   * Assemble context for the model — THE CORE HOOK.
   *
   * Converts AgentMessage[] → OpenAI format → compress() → AgentMessage[]
   */
  async assemble(params: {
    sessionId: string;
    messages: any[];
    tokenBudget?: number;
    model?: string;
    prompt?: string;
    /** OpenClaw runtime settings; `model.provider` drives the gateway-routed skip. */
    runtimeSettings?: unknown;
  }): Promise<{
    messages: any[];
    estimatedTokens: number;
    systemPromptAddition?: string;
  }> {
    if (!this.proxyUrl || this.config.enabled === false) {
      this.ensureProxyStarted();
      // Fallback: return messages unchanged
      return { messages: normalizeAgentMessages(params.messages), estimatedTokens: 0 };
    }

    if (this.isCircuitOpen()) {
      this.logger.warn("[headroom] Circuit open — using uncompressed messages");
      return { messages: normalizeAgentMessages(params.messages), estimatedTokens: 0 };
    }

    try {
      const skipDecision = decideAssembleSkip({
        config: this.config as unknown as Record<string, unknown>,
        providerId: providerIdFromRuntimeSettings(params.runtimeSettings),
      });
      if (skipDecision.skip) {
        const roughTokens = estimateRoughTokens(params.messages);
        this.logger.debug(`Assemble skip: ${skipDecision.reason}`);
        return {
          messages: normalizeAgentMessages(params.messages),
          estimatedTokens: roughTokens,
        };
      }

      const budget = params.tokenBudget;
      if (budget != null && budget > 0) {
        const roughTokens = estimateRoughTokens(params.messages);
        // Skip proxy compression when context is clearly under budget — avoids
        // multi-minute Kompress/tokenizer work on 100–200k sessions with 1M windows.
        const skipThreshold = resolveAssembleSkipThreshold({
          tokenBudget: budget,
          ratio: this.config.assembleSkipBudgetRatio,
          reserveTokens: this.config.assembleReserveTokens,
        });
        if (roughTokens < skipThreshold) {
          this.logger.debug(
            `Assemble skip: ~${roughTokens} tokens under threshold ${Math.round(skipThreshold)} (budget ${budget})`,
          );
          return {
            messages: normalizeAgentMessages(params.messages),
            estimatedTokens: roughTokens,
          };
        }
      }

      // Convert AgentMessage → OpenAI format
      const openaiMessages = agentToOpenAI(params.messages);
      const assembleCompressConfig = resolveAssembleCompressConfig(
        this.config.assembleCompressConfig,
      );

      // Compress via proxy — pass tokenBudget so RollingWindow enforces it
      const result = await withTimeout(
        compress(openaiMessages, {
          model: params.model ?? "claude-sonnet-4-5",
          baseUrl: this.proxyUrl,
          fallback: true,
          tokenBudget: params.tokenBudget,
          config: assembleCompressConfig,
        } as any),
        this.config.requestTimeoutMs ?? 30_000,
      );

      if (!result.compressed || result.tokensSaved === 0) {
        this.resetCircuit();
        return {
          messages: normalizeAgentMessages(params.messages),
          estimatedTokens: result.tokensBefore,
        };
      }

      // Convert back to AgentMessage format, restoring images/tool blocks/metadata
      // from the originals (the proxy does not reliably echo `_headroomMeta`).
      const compressedAgentMessages = openAIToAgent(result.messages, {
        originals: params.messages,
        protectedToolNames: this.protectedToolNames,
      });
      this.resetCircuit();

      // Track stats
      this.stats.totalCompressions++;
      this.stats.totalTokensSaved += result.tokensSaved;
      this.stats.totalTokensBefore += result.tokensBefore;

      this.logger.debug(
        `Assembled: ${result.tokensBefore} → ${result.tokensAfter} tokens (saved ${result.tokensSaved})`,
      );

      const ccrHashes = Array.isArray(result.ccrHashes) ? result.ccrHashes : [];
      return {
        messages: compressedAgentMessages,
        estimatedTokens: result.tokensAfter,
        systemPromptAddition:
          result.tokensSaved > 100 && ccrHashes.length > 0
            ? `[Context compressed by Headroom: ${result.tokensSaved} tokens saved. Use headroom_retrieve with the hash to get full details.]`
            : undefined,
      };
    } catch (error) {
      this.logger.error(`Assemble failed: ${error}`);
      this.tripCircuit(error);
      // Graceful fallback: return original messages
      return { messages: normalizeAgentMessages(params.messages), estimatedTokens: 0 };
    }
  }

  /**
   * Durable compaction (`/compact`, overflow recovery).
   *
   * - `persistentCompaction: "hybrid"` (default): Headroom replace pre-pass, then OpenClaw LLM compact.
   * - `persistentCompaction: "headroom"`: rewrite SQLite via Headroom `/v1/compress` (zero LLM).
   * - `persistentCompaction: "openclaw"`: delegate to OpenClaw native compaction only.
   */
  async compact(params: OpenClawCompactParams): Promise<OpenClawCompactResult> {
    params.abortSignal?.throwIfAborted();

    if (delegatesPersistentCompaction(this.persistentCompactionMode)) {
      // Hybrid pre-pass rewrites SQLite in-place — only when transcript hygiene
      // is explicitly enabled. Never bypass enabled:false via force:true.
      if (
        this.persistentCompactionMode === "hybrid" &&
        this.transcriptHygieneSettings.enabled
      ) {
        await this.runTranscriptHygienePass(params as HeadroomCompactParams, {
          force: true,
          compactPrepass: true,
        });
      }
      return this.delegateOpenClawCompaction(params);
    }

    if (!this.proxyUrl) {
      await this.ensureProxyUrl().catch(() => undefined);
    }
    if (!this.proxyUrl) {
      return { ok: false, compacted: false, reason: "Proxy not available" };
    }

    const headroomParams = params as HeadroomCompactParams;
    const sessionTarget = resolveSessionTarget(headroomParams);
    const tokenBudget =
      headroomParams.tokenBudget ??
      headroomParams.runtimeContext?.tokenBudget ??
      headroomParams.runtimeSettings?.promptTokenBudget;

    this.stats.compactions++;
    this.logger.info(
      `Compact started (budget: ${tokenBudget ?? "none"}, force: ${headroomParams.force ?? false}, session: ${headroomParams.sessionId})`,
    );

    try {
      const branchMessages = await loadBranchMessagesFromSession(
        sessionTarget,
        headroomParams.runtimeContext?.cwd ?? headroomParams.runtimeContext?.workspaceDir,
      );
      if (branchMessages.length === 0) {
        return { ok: true, compacted: false, reason: "empty transcript" };
      }

      const plan = await planHeadroomCompaction({
        branchMessages,
        tokenBudget,
        proxyUrl: this.proxyUrl,
        model: headroomParams.runtimeSettings?.resolvedModel ?? undefined,
        timeoutMs: this.config.requestTimeoutMs ?? 30_000,
        abortSignal: headroomParams.abortSignal,
        force: headroomParams.force === true,
        protectedToolNames: this.protectedToolNames,
      });

      if (plan.mode === "none") {
        return {
          ok: true,
          compacted: false,
          reason: "No durable compaction needed",
          result: { tokensBefore: plan.tokensBefore, tokensAfter: plan.tokensAfter },
        };
      }

      this.pendingCompactions.set(headroomParams.sessionId, {
        sessionId: headroomParams.sessionId,
        ...plan,
      });

      this.logger.info(
        `Compact planned (${plan.mode}): ${plan.tokensBefore} → ${plan.tokensAfter} tokens`,
      );

      return {
        ok: true,
        compacted: true,
        result: {
          tokensBefore: plan.tokensBefore,
          tokensAfter: plan.tokensAfter,
        },
      };
    } catch (error) {
      this.pendingCompactions.delete(headroomParams.sessionId);
      this.logger.error(`Compact failed: ${error}`);
      return {
        ok: false,
        compacted: false,
        reason: String(error),
      };
    }
  }

  async maintain(params: {
    sessionId: string;
    sessionKey?: string;
    sessionTarget?: {
      agentId?: string;
      sessionId?: string;
      sessionKey?: string;
      storePath?: string;
    };
    sessionFile: string;
    runtimeContext?: any;
    abortSignal?: AbortSignal;
  }): Promise<{
    changed: boolean;
    bytesFreed: number;
    rewrittenEntries: number;
    reason?: string;
  }> {
    params.abortSignal?.throwIfAborted();

    if (this.persistentCompactionMode !== "headroom") {
      if (!this.transcriptHygieneSettings.enabled) {
        return {
          changed: false,
          bytesFreed: 0,
          rewrittenEntries: 0,
          reason:
            this.persistentCompactionMode === "hybrid"
              ? "transcript hygiene disabled"
              : "persistent compaction delegated to OpenClaw",
        };
      }
      return this.runTranscriptHygienePass(params, { force: false });
    }

    const pending = this.pendingCompactions.get(params.sessionId);
    if (!pending) {
      return { changed: false, bytesFreed: 0, rewrittenEntries: 0, reason: "no pending compaction" };
    }

    this.pendingCompactions.delete(params.sessionId);

    try {
      const sessionTarget = resolveSessionTarget(params);
      const result = await applyCompactionPlan({
        sessionTarget,
        plan: pending,
        rewriteTranscriptEntries: params.runtimeContext?.rewriteTranscriptEntries,
        cwd: params.runtimeContext?.cwd ?? params.runtimeContext?.workspaceDir,
      });

      if (result.changed) {
        this.logger.info(
          `Compact persisted: freed ~${result.bytesFreed} bytes across ${result.rewrittenEntries} entries`,
        );
        await this.awaitTranscriptProjectionAfterRewrite(params, params.abortSignal);
      }

      return result;
    } catch (error) {
      this.logger.error(`Compact maintenance failed: ${error}`);
      return {
        changed: false,
        bytesFreed: 0,
        rewrittenEntries: 0,
        reason: String(error),
      };
    }
  }

  async afterTurn?(params: {
    sessionId: string;
    messages: any[];
    prePromptMessageCount: number;
    isHeartbeat?: boolean;
  }): Promise<void> {
    // Optional: could log stats or trigger learning
  }

  async prepareSubagentSpawn?(params: {
    parentSessionKey: string;
    childSessionKey: string;
    ttlMs?: number;
  }): Promise<{ rollback: () => Promise<void> } | undefined> {
    // Subagent context is compressed naturally via assemble()
    return undefined;
  }

  async onSubagentEnded?(params: {
    childSessionKey: string;
    reason: string;
  }): Promise<void> {
    // No-op
  }

  /**
   * Atomic + idempotent turn commit. OpenClaw calls this once the run for
   * a logical turn completes successfully. We persist the accepted messages
   * keyed by advancementKey so host retries and gateway restarts collapse to
   * the same record even though OpenClaw owns the canonical transcript.
   */
  async commitTurn(params: {
    sessionId: string;
    sessionKey?: string;
    advancementKey: string;
    messages: unknown[];
    admission?: unknown;
    terminal?: unknown;
    sessionTarget?: {
      agentId?: string;
      sessionId?: string;
      sessionKey?: string;
      storePath?: string;
    };
    runtimeSettings?: unknown;
    runtimeContext?: unknown;
    isHeartbeat?: boolean;
  }): Promise<{ status: "committed" | "duplicate" }> {
    const storePath = resolveTurnAdvancementStorePath({
      sessionTarget: params.sessionTarget,
      turnAdvancementStorePath: this.config.turnAdvancementStorePath,
    });
    const store = this.getTurnAdvancementStore(storePath);
    const status = store.commit({
      advancementKey: params.advancementKey,
      sessionId: params.sessionId,
      messages: params.messages,
    });
    return { status };
  }

  async dispose(): Promise<void> {
    await this.proxyManager.stop();
    this.logger.info(
      `Engine disposed. Stats: ${this.stats.totalCompressions} compressions, ` +
        `${this.stats.totalTokensSaved} tokens saved`,
    );
  }

  // --- Public API ---

  getStats() {
    return { ...this.stats };
  }

  getProxyUrl(): string | null {
    return this.proxyUrl;
  }

  getProxyStartupError(): unknown {
    return this.proxyStartupError;
  }

  private isCircuitOpen(): boolean {
    const threshold = this.config.circuitBreakerThreshold ?? 3;
    if (this.circuit.errors < threshold) return false;
    if (Date.now() < this.circuit.openUntilMs) return true;
    this.circuit = { errors: 0, openUntilMs: 0 };
    return false;
  }

  private tripCircuit(error: unknown): void {
    this.circuit.errors += 1;
    const threshold = this.config.circuitBreakerThreshold ?? 3;
    if (this.circuit.errors < threshold) return;
    const cooldownMs = this.config.circuitBreakerCooldownMs ?? 60_000;
    this.circuit.openUntilMs = Date.now() + cooldownMs;
    this.logger.warn(
      `[headroom] Circuit breaker opened after ${this.circuit.errors} errors ` +
        `(last: ${String(error)}); bypassing compression for ${cooldownMs}ms`,
    );
  }

  private resetCircuit(): void {
    this.circuit = { errors: 0, openUntilMs: 0 };
  }

  ensureProxyStarted(): void {
    if (this.config.enabled === false || this.proxyUrl || this.proxyStartupPromise) {
      return;
    }

    this.proxyStartupError = null;
    this.proxyStartupPromise = this.proxyManager
      .start()
      .then(async (proxyUrl) => {
        this.proxyUrl = proxyUrl;
        this.proxyStartupError = null;
        await this.notifyProxyReady(proxyUrl);
        this.logger.info(`Headroom proxy ready at ${proxyUrl}`);
        return proxyUrl;
      })
      .catch((error) => {
        this.proxyStartupError = error;
        this.logger.warn(`Headroom proxy unavailable: ${error}`);
        throw error;
      })
      .finally(() => {
        this.proxyStartupPromise = null;
      });

    // Fire-and-forget lifecycle callers intentionally do not await this promise.
    // Keep the promise rejectable for ensureProxyUrl(), but mark it observed so
    // a missing proxy cannot become a process-level unhandled rejection.
    void this.proxyStartupPromise.catch(() => {});
  }

  onProxyReady(listener: (proxyUrl: string) => void | Promise<void>): () => void {
    this.proxyReadyListeners.add(listener);
    return () => {
      this.proxyReadyListeners.delete(listener);
    };
  }

  async ensureProxyUrl(): Promise<string> {
    if (this.proxyUrl) {
      return this.proxyUrl;
    }

    this.ensureProxyStarted();
    if (!this.proxyStartupPromise) {
      throw new Error("Headroom proxy startup is disabled");
    }
    return this.proxyStartupPromise;
  }

  private getTurnAdvancementStore(storePath: string): TurnAdvancementStore {
    let store = this.turnAdvancementStores.get(storePath);
    if (!store) {
      store = new TurnAdvancementStore({ storePath });
      this.turnAdvancementStores.set(storePath, store);
    }
    return store;
  }

  private async notifyProxyReady(proxyUrl: string): Promise<void> {
    for (const listener of this.proxyReadyListeners) {
      try {
        await listener(proxyUrl);
      } catch (error) {
        this.logger.warn(`Headroom proxy ready listener failed: ${error}`);
      }
    }
  }

  private async delegateOpenClawCompaction(
    params: OpenClawCompactParams,
  ): Promise<OpenClawCompactResult> {
    const result = await delegateCompactionToRuntime(params);
    if (result.compacted) {
      this.stats.compactions++;
    }
    this.logger.info(
      `Compaction ${result.compacted ? "completed" : "skipped"} ` +
        `(delegated to OpenClaw, mode: ${this.persistentCompactionMode}, budget: ${params.tokenBudget ?? "none"}, force: ${params.force ?? false})`,
    );
    return result;
  }

  private async runTranscriptHygienePass(
    params: HeadroomCompactParams & {
      sessionKey?: string;
      sessionTarget?: HeadroomCompactParams["sessionTarget"];
      sessionFile?: string;
      abortSignal?: AbortSignal;
    },
    options: { force: boolean; compactPrepass?: boolean },
  ): Promise<{ changed: boolean; bytesFreed: number; rewrittenEntries: number; reason?: string }> {
    if (
      this.shouldDebounceHygiene(params.sessionId, {
        force: options.force,
        compactPrepass: options.compactPrepass,
      })
    ) {
      return {
        changed: false,
        bytesFreed: 0,
        rewrittenEntries: 0,
        reason: "debounced",
      };
    }

    if (!this.transcriptHygieneSettings.enabled) {
      return {
        changed: false,
        bytesFreed: 0,
        rewrittenEntries: 0,
        reason: "transcript hygiene disabled",
      };
    }

    if (!this.proxyUrl) {
      await this.ensureProxyUrl().catch(() => undefined);
    }
    if (!this.proxyUrl) {
      return {
        changed: false,
        bytesFreed: 0,
        rewrittenEntries: 0,
        reason: "Proxy not available",
      };
    }

    const tokenBudget =
      params.tokenBudget ??
      params.runtimeContext?.tokenBudget ??
      params.runtimeSettings?.promptTokenBudget;
    const softThresholdTokens = resolveSoftThresholdTokens(
      this.transcriptHygieneSettings,
      tokenBudget,
    );

    try {
      const result = await runTranscriptReplaceHygiene({
        params: params as TranscriptHygieneRuntimeParams,
        proxyUrl: this.proxyUrl,
        timeoutMs: this.config.requestTimeoutMs ?? 30_000,
        softThresholdTokens,
        force: options.force,
        model: params.runtimeSettings?.resolvedModel ?? undefined,
      });

      if (result.changed) {
        this.stats.hygieneRuns++;
        this.stats.hygieneBytesFreed += result.bytesFreed;
        this.hygieneDebounce.recordChanged(params.sessionId);
        this.logger.info(
          `Transcript hygiene: freed ~${result.bytesFreed} bytes across ${result.rewrittenEntries} entries` +
            (result.tokensBefore !== undefined
              ? ` (${result.tokensBefore} → ${result.tokensAfter ?? result.tokensBefore} tokens)`
              : ""),
        );
        await this.awaitTranscriptProjectionAfterRewrite(params, params.abortSignal);
      }

      return result;
    } catch (error) {
      this.logger.warn(`Transcript hygiene failed: ${error}`);
      return {
        changed: false,
        bytesFreed: 0,
        rewrittenEntries: 0,
        reason: String(error),
      };
    }
  }

  private shouldDebounceHygiene(
    sessionId: string,
    options: { force: boolean; compactPrepass?: boolean },
  ): boolean {
    if (!options.force || options.compactPrepass) {
      return this.hygieneDebounce.shouldSkip(
        sessionId,
        this.transcriptHygieneSettings.debounceMs,
      );
    }
    return false;
  }

  private async awaitTranscriptProjectionAfterRewrite(
    params: {
      sessionId: string;
      sessionKey?: string;
      sessionTarget?: HeadroomCompactParams["sessionTarget"];
      runtimeContext?: HeadroomCompactParams["runtimeContext"];
    },
    abortSignal?: AbortSignal,
  ): Promise<void> {
    const settle = await awaitTranscriptProjectionSettle({
      params,
      runtimeContext: params.runtimeContext,
      abortSignal,
      timeoutMs: this.transcriptProjectionWaitMs,
      logger: this.logger,
    });
    if (!settle.waited && settle.reason && settle.reason !== "projection wait disabled") {
      this.logger.debug?.(
        `[headroom] Transcript projection settle skipped: ${settle.reason}`,
      );
    }
  }
}
