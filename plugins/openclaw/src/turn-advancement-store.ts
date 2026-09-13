/**
 * Durable, idempotent turn advancement store for OpenClaw's commitTurn contract.
 *
 * Keys turns by advancementKey and persists to disk so host retries and gateway
 * restarts collapse to the same committed record.
 *
 * OpenClaw owns the canonical transcript, so the store only keeps what the
 * idempotency contract needs: the key, a digest of the accepted messages (for
 * conflict detection), and bookkeeping metadata. Message bodies are never
 * written to disk, and old records are pruned so the file stays small enough
 * to rewrite synchronously on every commit.
 */

import { createHash } from "node:crypto";
import { mkdirSync, readFileSync, renameSync, writeFileSync } from "node:fs";
import { homedir, tmpdir } from "node:os";
import { dirname, join } from "node:path";
import { type StoreLockOptions, withStoreLock } from "./store-lock.js";

export interface TurnAdvancementRecord {
  advancementKey: string;
  messagesDigest: string;
  messageCount: number;
  sessionId: string;
  committedAtMs: number;
}

export interface TurnAdvancementRetention {
  /** Records older than this are pruned on the next commit. */
  maxAgeMs: number;
  /** Hard cap on persisted records; the oldest committed are pruned first. */
  maxRecords: number;
}

export const DEFAULT_TURN_ADVANCEMENT_RETENTION: TurnAdvancementRetention = {
  maxAgeMs: 14 * 24 * 60 * 60 * 1000,
  maxRecords: 5_000,
};

export interface TurnAdvancementStoreOptions {
  storePath: string;
  /** Test hook invoked immediately before the atomic persist. */
  injectBeforePersist?: () => void;
  /** Lock tuning (timeouts / stale thresholds); defaults suit production. */
  lockOptions?: StoreLockOptions;
  /** Retention tuning; defaults suit production. */
  retention?: Partial<TurnAdvancementRetention>;
  /** Clock override for deterministic retention tests. */
  now?: () => number;
}

/** Legacy on-disk shape (v1) that embedded full message bodies. */
interface PersistedTurnAdvancementV1 {
  advancementKey: string;
  messages?: unknown[];
  messagesDigest: string;
  sessionId: string;
  committedAtMs: number;
}

interface PersistedTurnAdvancementsV1 {
  version: 1;
  records: Record<string, PersistedTurnAdvancementV1>;
}

interface PersistedTurnAdvancementsV2 {
  version: 2;
  records: Record<string, TurnAdvancementRecord>;
}

type PersistedTurnAdvancements =
  | PersistedTurnAdvancementsV1
  | PersistedTurnAdvancementsV2;

export function digestTurnMessages(messages: unknown[]): string {
  return createHash("sha256").update(JSON.stringify(messages)).digest("hex");
}

export function resolveTurnAdvancementStorePath(params: {
  sessionTarget?: { storePath?: string };
  turnAdvancementStorePath?: string;
}): string {
  if (params.turnAdvancementStorePath) {
    return params.turnAdvancementStorePath;
  }

  const sessionStorePath = params.sessionTarget?.storePath;
  if (sessionStorePath) {
    return join(dirname(sessionStorePath), "headroom-turn-advancements.json");
  }

  const homeDir = (() => {
    try {
      return homedir();
    } catch {
      return tmpdir();
    }
  })();
  const stateDir = process.env.OPENCLAW_STATE_DIR ?? join(homeDir, ".openclaw");
  return join(stateDir, "headroom-turn-advancements.json");
}

function migrateV1Record(record: PersistedTurnAdvancementV1): TurnAdvancementRecord {
  return {
    advancementKey: record.advancementKey,
    messagesDigest: record.messagesDigest,
    messageCount: Array.isArray(record.messages) ? record.messages.length : 0,
    sessionId: record.sessionId,
    committedAtMs: record.committedAtMs,
  };
}

function loadPersistedRecords(storePath: string): Map<string, TurnAdvancementRecord> {
  const records = new Map<string, TurnAdvancementRecord>();
  try {
    const raw = readFileSync(storePath, "utf8");
    const parsed = JSON.parse(raw) as PersistedTurnAdvancements;
    if (!parsed.records) {
      return records;
    }
    if (parsed.version === 1) {
      for (const [key, record] of Object.entries(parsed.records)) {
        records.set(key, migrateV1Record(record));
      }
    } else if (parsed.version === 2) {
      for (const [key, record] of Object.entries(parsed.records)) {
        records.set(key, record);
      }
    }
  } catch (error) {
    const err = error as NodeJS.ErrnoException;
    if (err.code !== "ENOENT") {
      throw error;
    }
  }
  return records;
}

/**
 * Drop records that fall outside the retention window. The record being
 * committed (`keepKey`) is always retained regardless of the cap.
 */
export function pruneTurnAdvancementRecords(
  records: Map<string, TurnAdvancementRecord>,
  params: { retention: TurnAdvancementRetention; nowMs: number; keepKey: string },
): Map<string, TurnAdvancementRecord> {
  const cutoffMs = params.nowMs - params.retention.maxAgeMs;
  const kept = [...records.values()].filter(
    (record) => record.advancementKey === params.keepKey || record.committedAtMs >= cutoffMs,
  );
  kept.sort((a, b) => a.committedAtMs - b.committedAtMs);

  const maxRecords = Math.max(1, params.retention.maxRecords);
  while (kept.length > maxRecords) {
    const index = kept.findIndex((record) => record.advancementKey !== params.keepKey);
    if (index === -1) {
      break;
    }
    kept.splice(index, 1);
  }

  return new Map(kept.map((record) => [record.advancementKey, record]));
}

function persistRecords(
  storePath: string,
  records: Map<string, TurnAdvancementRecord>,
  injectBeforePersist?: () => void,
): void {
  mkdirSync(dirname(storePath), { recursive: true });
  injectBeforePersist?.();
  const payload: PersistedTurnAdvancementsV2 = {
    version: 2,
    records: Object.fromEntries(records),
  };
  const tmpPath = `${storePath}.tmp`;
  writeFileSync(tmpPath, JSON.stringify(payload), "utf8");
  renameSync(tmpPath, storePath);
}

export class TurnAdvancementStore {
  private records = new Map<string, TurnAdvancementRecord>();
  private loaded = false;
  private readonly retention: TurnAdvancementRetention;
  private readonly now: () => number;

  constructor(private readonly options: TurnAdvancementStoreOptions) {
    this.retention = { ...DEFAULT_TURN_ADVANCEMENT_RETENTION, ...options.retention };
    this.now = options.now ?? Date.now;
  }

  commit(params: {
    advancementKey: string;
    sessionId: string;
    messages: unknown[];
  }): "committed" | "duplicate" {
    const lockPath = `${this.options.storePath}.lock`;
    return withStoreLock(lockPath, () => {
      const records = loadPersistedRecords(this.options.storePath);
      const digest = digestTurnMessages(params.messages);
      const existing = records.get(params.advancementKey);
      if (existing) {
        if (existing.messagesDigest === digest) {
          this.syncMemoryFromDisk(records);
          return "duplicate";
        }
        throw new Error(
          `turn advancement key conflict for ${params.advancementKey}`,
        );
      }

      const nowMs = this.now();
      const record: TurnAdvancementRecord = {
        advancementKey: params.advancementKey,
        messagesDigest: digest,
        messageCount: params.messages.length,
        sessionId: params.sessionId,
        committedAtMs: nowMs,
      };

      const withRecord = new Map(records);
      withRecord.set(params.advancementKey, record);
      const nextRecords = pruneTurnAdvancementRecords(withRecord, {
        retention: this.retention,
        nowMs,
        keepKey: params.advancementKey,
      });
      // If persist throws, the key is not cached as committed and a retry
      // returns "committed" (not "duplicate") after re-reading disk under lock.
      persistRecords(this.options.storePath, nextRecords, this.options.injectBeforePersist);

      this.syncMemoryFromDisk(nextRecords);
      return "committed";
    }, this.options.lockOptions);
  }

  has(advancementKey: string): boolean {
    this.ensureLoaded();
    return this.records.has(advancementKey);
  }

  private ensureLoaded(): void {
    if (this.loaded) {
      return;
    }
    this.syncMemoryFromDisk(loadPersistedRecords(this.options.storePath));
  }

  private syncMemoryFromDisk(records: Map<string, TurnAdvancementRecord>): void {
    this.records = new Map(records);
    this.loaded = true;
  }
}
