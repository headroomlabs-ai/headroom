/**
 * SharedContext — compressed inter-agent context sharing.
 * Matches Python headroom.shared_context.SharedContext.
 */

import { HeadroomClient } from "./client.js";
import type { HeadroomClientOptions, CompressResult } from "./types.js";

export interface ContextEntry {
  key: string;
  original: string;
  compressed: string;
  originalTokens: number;
  compressedTokens: number;
  agent: string | null;
  timestamp: number;
  transforms: string[];
  savingsPercent: number;
}

export interface SharedContextStats {
  entries: number;
  totalOriginalTokens: number;
  totalCompressedTokens: number;
  totalTokensSaved: number;
  savingsPercent: number;
}

export interface SharedContextOptions extends HeadroomClientOptions {
  model?: string;
  ttl?: number;
  maxEntries?: number;
}

export class SharedContext {
  private entries = new Map<string, ContextEntry>();
  private client: HeadroomClient;
  private model: string;
  private ttl: number;
  private maxEntries: number;

  constructor(options: SharedContextOptions = {}) {
    const { model, ttl, maxEntries, ...clientOptions } = options;
    this.client = new HeadroomClient(clientOptions);
    this.model = model ?? "claude-sonnet-4-5-20250929";
    this.ttl = ttl ?? 3600;
    this.maxEntries = maxEntries ?? 100;
  }

  /**
   * Store content with compression.
   */
  async put(
    key: string,
    content: string,
    options?: { agent?: string },
  ): Promise<ContextEntry> {
    this.evictExpired();
    this.evictIfFull();

    const messages = [{ role: "user" as const, content }];
    let result: CompressResult;
    try {
      result = await this.client.compress(messages, { model: this.model });
    } catch {
      // If proxy unavailable, store uncompressed
      result = {
        messages,
        tokensBefore: content.length / 4, // rough estimate
        tokensAfter: content.length / 4,
        tokensSaved: 0,
        compressionRatio: 1.0,
        transformsApplied: [],
        ccrHashes: [],
        compressed: false,
      };
    }

    const compressed = result.compressed
      ? typeof result.messages[0]?.content === "string"
        ? result.messages[0].content
        : JSON.stringify(result.messages[0]?.content ?? content)
      : content;

    const entry: ContextEntry = {
      key,
      original: content,
      compressed,
      originalTokens: result.tokensBefore,
      compressedTokens: result.tokensAfter,
      agent: options?.agent ?? null,
      timestamp: Date.now() / 1000,
      transforms: result.transformsApplied,
      savingsPercent:
        result.tokensBefore > 0
          ? ((result.tokensBefore - result.tokensAfter) / result.tokensBefore) *
            100
          : 0,
    };

    this.entries.set(key, entry);
    return entry;
  }

  /**
   * Get content by key. Returns compressed by default, full original if full=true.
   */
  get(key: string, options?: { full?: boolean }): string | null {
    const entry = this.entries.get(key);
    if (!entry) return null;

    if (Date.now() / 1000 - entry.timestamp > this.ttl) {
      this.entries.delete(key);
      return null;
    }

    return options?.full ? entry.original : entry.compressed;
  }

  /**
   * Get full entry metadata.
   */
  getEntry(key: string): ContextEntry | null {
    const entry = this.entries.get(key);
    if (!entry) return null;

    if (Date.now() / 1000 - entry.timestamp > this.ttl) {
      this.entries.delete(key);
      return null;
    }

    return entry;
  }

  /**
   * List all stored keys (excluding expired).
   */
  keys(): string[] {
    this.evictExpired();
    return Array.from(this.entries.keys());
  }

  /**
   * Get aggregated statistics.
   */
  stats(): SharedContextStats {
    this.evictExpired();
    let totalOriginal = 0;
    let totalCompressed = 0;

    for (const entry of this.entries.values()) {
      totalOriginal += entry.originalTokens;
      totalCompressed += entry.compressedTokens;
    }

    const saved = totalOriginal - totalCompressed;
    return {
      entries: this.entries.size,
      totalOriginalTokens: totalOriginal,
      totalCompressedTokens: totalCompressed,
      totalTokensSaved: saved,
      savingsPercent: totalOriginal > 0 ? (saved / totalOriginal) * 100 : 0,
    };
  }

  /**
   * Clear all entries.
   */
  clear(): void {
    this.entries.clear();
  }

  private evictExpired(): void {
    const now = Date.now() / 1000;
    for (const [key, entry] of this.entries) {
      if (now - entry.timestamp > this.ttl) {
        this.entries.delete(key);
      }
    }
  }

  private evictIfFull(): void {
    while (this.entries.size >= this.maxEntries) {
      const oldest = this.entries.keys().next().value;
      if (oldest !== undefined) this.entries.delete(oldest);
    }
  }
}
