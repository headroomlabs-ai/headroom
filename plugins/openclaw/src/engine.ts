/**
 * HeadroomContextEngine — ContextEngine implementation for OpenClaw.
 *
 * Compresses tool outputs and conversation context using the Headroom proxy.
 * Zero LLM calls — all compression is algorithmic (SmartCrusher, ContentRouter, etc.)
 */

/* eslint-disable @typescript-eslint/no-explicit-any */

import { createHash } from "node:crypto";
import { compress } from "headroom-ai";
import { ProxyManager, defaultLogger, type ProxyManagerConfig, type ProxyManagerLogger } from "./proxy-manager.js";
import { agentToOpenAIIndexed, normalizeAgentMessages, restoreAgentMessages } from "./convert.js";
import { DurableAdvancementKeyStore, defaultCommitLogPath } from "./advancement-key-store.js";
import {
  delegateCompactionToRuntime,
  type OpenClawCompactParams,
  type OpenClawCompactResult,
} from "./openclaw-compaction.js";

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

// Must stay static — OpenClaw prepends it to the cached system prompt.
const HEADROOM_COMPRESSION_NOTICE =
  "[Headroom is compressing tool outputs in this session. Use headroom_retrieve if you need the original, uncompressed content.]";

export interface HeadroomEngineConfig extends ProxyManagerConfig {
  enabled?: boolean;
  requestTimeoutMs?: number;
  circuitBreakerThreshold?: number;
  circuitBreakerCooldownMs?: number;
  /** Where to durably record committed turn-advancement keys (see
   * `DurableAdvancementKeyStore`). Defaults to `defaultCommitLogPath()`. */
  commitLogPath?: string;
  announceCompression?: boolean;
  /**
   * Skip applying a compression while the provider's prompt cache for this session is still warm, i.e. the
   * newest assistant message in the history is younger than this many ms (0 = never skip). Compressing history the
   * provider already cached makes it re-write that history at the cache-write rate, to save tokens that
   * would have been billed at the cache-read rate, so on a warm cache it almost never pays. A cold cache
   * is re-written anyway, so compressing then is free and shrinks the write. The history is still
   * compressed when it has to shrink to fit `tokenBudget`. Set it to the provider's cache TTL
   * (Anthropic/Bedrock default: 300000).
   *
   * While the cache is warm, a session keeps receiving the compressed messages its last compressing
   * assembly returned (the provider cached those, not the originals), with any newer messages after them
   * as they came in. Only a cold assembly compresses the newer messages too.
   */
  skipCompressionWhenCacheWarmMs?: number;
}

/**
 * UTF-8 size of `messages` as JSON: an upper bound on the proxy's token count for them. Its tokenizers spend at
 * least one byte of text per token, and the JSON also carries each message's framing. (An image is counted at a
 * fixed estimate, which its base64 data exceeds unless the image is tiny.)
 */
function jsonBytes(messages: any[]): number {
  return Buffer.byteLength(JSON.stringify(messages), "utf8");
}

/** The proxy's token count for a session's history, so a warm turn can bound it without asking again. */
interface TokenCount {
  /** Fingerprints of the messages counted, in order. */
  keys: string[];
  /** The model counted for: the proxy picks its tokenizer by model. */
  model: string;
  tokens: number;
  usedAt: number;
}

/** What a compressing assembly returned, so later assemblies can send the same bytes while the cache is warm. */
interface CompressedView {
  /** Fingerprints of the original messages the view replaces, in order. */
  sourceKeys: string[];
  /** Fingerprints of the view's own messages, for a caller that passes the view back in place of the originals. */
  viewKeys: string[];
  messages: any[];
  usedAt: number;
}

/** Content fingerprint of one message. */
function messageKey(message: unknown): string {
  return createHash("sha1")
    .update(JSON.stringify(message) ?? "")
    .digest("base64");
}

/** True when `messages` starts with messages whose fingerprints are `keys`. */
function startsWithKeys(messages: any[], keys: string[]): boolean {
  return messages.length >= keys.length && keys.every((key, i) => key === messageKey(messages[i]));
}

export class HeadroomContextEngine {
  readonly info = {
    id: "headroom",
    name: "Headroom Context Compression",
    version: "0.1.0",
    ownsCompaction: false,
    transcriptSemantics: {
      currentTurnFence: "before-current-turn-entry-v1",
      turnAdvancementIdempotency: "atomic-idempotent-v1",
    },
  };

  // Durable, restart-safe record of committed advancement keys, for
  // commitTurn's idempotent-retry check. See advancement-key-store.ts for
  // why this must survive a process restart and must not evict entries.
  private advancementKeyStore: DurableAdvancementKeyStore;

  private proxyManager: ProxyManager;
  private proxyUrl: string | null = null;
  private config: HeadroomEngineConfig;
  private logger: ProxyManagerLogger;
  private proxyReadyListeners = new Set<(proxyUrl: string) => void | Promise<void>>();
  private proxyStartupPromise: Promise<string> | null = null;
  private proxyStartupError: unknown = null;
  private stats = {
    totalCompressions: 0,
    totalTokensSaved: 0,
    totalTokensBefore: 0,
    compactions: 0,
  };
  private circuit = { errors: 0, openUntilMs: 0 };
  // Sessions shown the compression notice. Bounded via insertion-order
  // eviction — no session lifecycle callback exists here to bound it otherwise.
  private announcedSessions = new Set<string>();
  private static readonly MAX_ANNOUNCED_SESSIONS = 1000;
  /** Per session: the compressed view the provider's prompt cache holds (see `unchanged`). */
  private compressedViews = new Map<string, CompressedView>();
  /** Per session: the proxy's last token count for the history (see `fitsBudget`). */
  private tokenCounts = new Map<string, TokenCount>();

  constructor(config: HeadroomEngineConfig = {}, logger?: ProxyManagerLogger) {
    this.config = config;
    this.logger = logger ?? defaultLogger;
    this.proxyManager = new ProxyManager(config, this.logger);
    this.advancementKeyStore = new DurableAdvancementKeyStore(
      config.commitLogPath ?? defaultCommitLogPath(),
    );
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
   * Converts AgentMessage[] → OpenAI format → compress() → AgentMessage[]. Only messages the proxy
   * actually changed are rebuilt; the rest are returned exactly as OpenClaw passed them, so the
   * provider prompt cache survives up to the first compressed message (see restoreAgentMessages).
   */
  async assemble(params: {
    sessionId: string;
    messages: any[];
    tokenBudget?: number;
    model?: string;
    prompt?: string;
  }): Promise<{
    messages: any[];
    estimatedTokens: number;
    systemPromptAddition?: string;
  }> {
    this.evictIdle(params.sessionId);
    const warm = this.isCacheWarm(params.messages);
    const model = params.model ?? "claude-sonnet-4-5";

    if (!this.proxyUrl || this.config.enabled === false) {
      this.ensureProxyStarted();
      // Fallback: return messages unchanged
      return {
        messages: this.unchanged(params, warm),
        estimatedTokens: 0,
        systemPromptAddition: this.sessionCompressionNotice(params.sessionId),
      };
    }

    if (this.isCircuitOpen()) {
      this.logger.warn("[headroom] Circuit open — using uncompressed messages");
      return {
        messages: this.unchanged(params, warm),
        estimatedTokens: 0,
        systemPromptAddition: this.sessionCompressionNotice(params.sessionId),
      };
    }

    // Warm cache and provably within budget: skip without calling the proxy, so it neither counts savings that
    // are never applied nor adds its latency. Otherwise ask the proxy and decide on its token count below.
    if (warm && this.fitsBudget(params, model)) {
      return {
        messages: this.unchanged(params, warm),
        estimatedTokens: 0,
        systemPromptAddition: this.sessionCompressionNotice(params.sessionId),
      };
    }

    try {
      // Convert AgentMessage → OpenAI format
      const openaiMessages = agentToOpenAIIndexed(params.messages);

      // Compress via proxy — pass tokenBudget so RollingWindow enforces it
      const result = await withTimeout(
        compress(openaiMessages, {
          model,
          baseUrl: this.proxyUrl,
          fallback: true,
          tokenBudget: params.tokenBudget,
        } as any),
        this.config.requestTimeoutMs ?? 30_000,
      );
      this.rememberTokenCount(params, model, result.tokensBefore);

      if (!result.compressed || result.tokensSaved === 0) {
        this.resetCircuit();
        return {
          messages: this.unchanged(params, warm),
          estimatedTokens: result.tokensBefore,
          systemPromptAddition: this.sessionCompressionNotice(params.sessionId),
        };
      }

      if (warm && !(params.tokenBudget && result.tokensBefore > params.tokenBudget)) {
        this.resetCircuit();
        this.logger.debug(`Skipped compression on a warm prompt cache (would have saved ${result.tokensSaved})`);
        return {
          messages: this.unchanged(params, warm),
          estimatedTokens: result.tokensBefore,
          systemPromptAddition: this.sessionCompressionNotice(params.sessionId),
        };
      }

      // Only a compression that is applied earns the notice, so this comes after the warm skip.
      const shouldAnnounceCompression =
        (this.hasAnnouncedCompression(params.sessionId) || result.tokensSaved > 100) &&
        this.config.announceCompression !== false;
      if (shouldAnnounceCompression) {
        this.markCompressionAnnounced(params.sessionId);
      }

      // Convert back to AgentMessage format
      const compressedAgentMessages = restoreAgentMessages(params.messages, openaiMessages, result.messages);
      this.resetCircuit();
      this.remember(params, compressedAgentMessages);

      // Track stats
      this.stats.totalCompressions++;
      this.stats.totalTokensSaved += result.tokensSaved;
      this.stats.totalTokensBefore += result.tokensBefore;

      this.logger.debug(
        `Assembled: ${result.tokensBefore} → ${result.tokensAfter} tokens (saved ${result.tokensSaved})`,
      );

      return {
        messages: compressedAgentMessages,
        estimatedTokens: result.tokensAfter,
        systemPromptAddition: this.sessionCompressionNotice(params.sessionId),
      };
    } catch (error) {
      this.logger.error(`Assemble failed: ${error}`);
      this.tripCircuit(error);
      // Graceful fallback: return original messages
      return {
        messages: this.unchanged(params, warm),
        estimatedTokens: 0,
        systemPromptAddition: this.sessionCompressionNotice(params.sessionId),
      };
    }
  }

  /**
   * The history to send when this assembly applies no new compression.
   *
   * While the cache is warm it has to be what the provider already holds. That is the compressed view the
   * session's last compressing assembly returned, for the messages it covered, followed by the newer messages
   * as they came in. Sending the originals instead would undo that compression and make the provider re-write
   * everything after the first message it touched. On a cold cache everything is written anyway, so the
   * originals go out and the view is dropped.
   */
  private unchanged(params: { sessionId: string; messages: any[] }, warm: boolean): any[] {
    const view = this.compressedViews.get(params.sessionId);
    if (view && warm) {
      const covered = startsWithKeys(params.messages, view.sourceKeys)
        ? view.sourceKeys.length
        : startsWithKeys(params.messages, view.viewKeys)
          ? view.viewKeys.length
          : -1;
      if (covered >= 0) {
        view.usedAt = Date.now();
        this.logger.debug(`Kept the compressed view of ${covered} messages (prompt cache warm)`);
        // A copy: OpenClaw keeps the returned list as the turn's working state and appends to it.
        return [...structuredClone(view.messages), ...normalizeAgentMessages(params.messages.slice(covered))];
      }
    }
    // No view, a cold cache, or a history that changed under the view (e.g. a compaction rewrote it), so the
    // provider cannot be holding the view any more.
    this.compressedViews.delete(params.sessionId);
    return normalizeAgentMessages(params.messages);
  }

  /** Record what a compressing assembly returned in place of `params.messages`, for `unchanged`. */
  private remember(params: { sessionId: string; messages: any[] }, assembled: any[]): void {
    if (!(this.cacheWindowMs() > 0)) return;
    try {
      this.compressedViews.set(params.sessionId, {
        sourceKeys: params.messages.map(messageKey),
        viewKeys: assembled.map(messageKey),
        messages: structuredClone(assembled),
        usedAt: Date.now(),
      });
    } catch (error) {
      // Not fatal: without a view the next warm assembly sends the originals, which costs one cache re-write.
      this.compressedViews.delete(params.sessionId);
      this.logger.warn(`[headroom] Could not keep the compressed view: ${error}`);
    }
  }

  /**
   * True when the history can be shown to fit `tokenBudget` (or there is none) without asking the proxy. The
   * bound is the proxy's own count for `model` from the last time it saw this history plus the size of any
   * messages appended since, or failing that the size of the whole history (see `jsonBytes`).
   */
  private fitsBudget(params: { sessionId: string; messages: any[]; tokenBudget?: number }, model: string): boolean {
    if (!params.tokenBudget) return true;
    const count = this.tokenCounts.get(params.sessionId);
    if (count && count.model === model && startsWithKeys(params.messages, count.keys)) {
      count.usedAt = Date.now();
      return count.tokens + jsonBytes(params.messages.slice(count.keys.length)) <= params.tokenBudget;
    }
    return jsonBytes(params.messages) <= params.tokenBudget;
  }

  /** Record the proxy's token count for `params.messages` under `model`, for `fitsBudget`. */
  private rememberTokenCount(params: { sessionId: string; messages: any[] }, model: string, tokens: number): void {
    // The SDK's fallback result (proxy unreachable) reports 0 tokens, which is not a count.
    if (!(this.cacheWindowMs() > 0) || !(tokens > 0)) return;
    this.tokenCounts.set(params.sessionId, { keys: params.messages.map(messageKey), model, tokens, usedAt: Date.now() });
  }

  /**
   * Drop the views and counts of other sessions that have gone unused for longer than the window: their next
   * assembly is cold and cannot use them. The current session's are settled by the assembly in progress.
   */
  private evictIdle(sessionId: string): void {
    const windowMs = this.cacheWindowMs();
    if (!(windowMs > 0)) return;
    const now = Date.now();
    for (const entries of [this.compressedViews, this.tokenCounts] as Map<string, { usedAt: number }>[]) {
      for (const [id, entry] of entries) {
        if (id !== sessionId && now - entry.usedAt > windowMs) entries.delete(id);
      }
    }
  }

  private cacheWindowMs(): number {
    return this.config.skipCompressionWhenCacheWarmMs ?? 0;
  }

  /**
   * True while the newest assistant message in the history is younger than `skipCompressionWhenCacheWarmMs`. An
   * assistant message is stamped when the provider is asked for it, which is the last time the provider read this
   * prefix. Messages after it, such as the result of a tool that ran for a long time, do not keep the cache warm.
   */
  private isCacheWarm(messages: any[]): boolean {
    const windowMs = this.cacheWindowMs();
    if (!(windowMs > 0)) return false;
    let newest = 0;
    for (const message of messages) {
      const ts = message?.timestamp;
      if (message?.role === "assistant" && typeof ts === "number" && ts > newest) newest = ts;
    }
    return newest > 0 && Date.now() - newest < windowMs;
  }

  /** Delegate persistent compaction to OpenClaw's built-in runtime. */
  async compact(params: OpenClawCompactParams): Promise<OpenClawCompactResult> {
    const result = await delegateCompactionToRuntime(params);

    if (result.compacted) {
      this.stats.compactions++;
    }
    this.logger.info(
      `Compaction ${result.compacted ? "completed" : "skipped"} ` +
        `(budget: ${params.tokenBudget ?? "none"}, force: ${params.force ?? false})`,
    );

    return result;
  }

  /**
   * Durable turn-advancement commit — required by OpenClaw's transcriptSemantics
   * contract. Called only for the accepted, successful turn; failed or aborted
   * turns never reach here. Must be an atomic, idempotent write of the
   * accepted `messages` keyed by `advancementKey` so a host retry with the
   * same key reports "duplicate" instead of re-applying the advancement —
   * including a retry that arrives after this process restarted, which is
   * why the record lives on disk (see `DurableAdvancementKeyStore`) rather
   * than in memory, and includes the messages rather than just the key.
   */
  async commitTurn(params: { advancementKey: string; messages: any[] }): Promise<{
    status: "committed" | "duplicate";
  }> {
    const status = await this.advancementKeyStore.tryCommit(params.advancementKey, params.messages);
    return { status };
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

  private hasAnnouncedCompression(sessionId: string): boolean {
    return this.announcedSessions.has(sessionId);
  }

  private markCompressionAnnounced(sessionId: string): void {
    if (this.announcedSessions.has(sessionId)) return;
    if (this.announcedSessions.size >= HeadroomContextEngine.MAX_ANNOUNCED_SESSIONS) {
      const oldest = this.announcedSessions.values().next().value;
      if (oldest !== undefined) this.announcedSessions.delete(oldest);
    }
    this.announcedSessions.add(sessionId);
  }

  /** The sticky notice if this session already earned it and announcements aren't opted out. */
  private sessionCompressionNotice(sessionId: string): string | undefined {
    return this.hasAnnouncedCompression(sessionId) && this.config.announceCompression !== false
      ? HEADROOM_COMPRESSION_NOTICE
      : undefined;
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

  private async notifyProxyReady(proxyUrl: string): Promise<void> {
    for (const listener of this.proxyReadyListeners) {
      try {
        await listener(proxyUrl);
      } catch (error) {
        this.logger.warn(`Headroom proxy ready listener failed: ${error}`);
      }
    }
  }
}
