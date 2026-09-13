/** How Headroom handles durable `/compact` and overflow-recovery compaction. */
export type PersistentCompactionMode = "headroom" | "openclaw" | "hybrid";

export interface PersistentCompactionConfig {
  /** @deprecated Prefer `persistentCompaction`. */
  durableCompaction?: boolean;
  persistentCompaction?: PersistentCompactionMode | boolean;
}

/** Resolve configured durable compaction ownership (default: openclaw — matches upstream delegation). */
export function resolvePersistentCompactionMode(
  config: PersistentCompactionConfig = {},
): PersistentCompactionMode {
  const { persistentCompaction, durableCompaction } = config;

  if (persistentCompaction === "openclaw" || persistentCompaction === false) {
    return "openclaw";
  }
  if (persistentCompaction === "headroom" || persistentCompaction === true) {
    return "headroom";
  }
  if (persistentCompaction === "hybrid") {
    return "hybrid";
  }
  if (persistentCompaction !== undefined) {
    throw new Error(
      `Invalid headroom persistentCompaction value: ${String(persistentCompaction)} (expected "headroom", "openclaw", or "hybrid")`,
    );
  }

  if (durableCompaction === false) {
    return "openclaw";
  }
  if (durableCompaction === true) {
    return "headroom";
  }

  return "openclaw";
}

export function ownsPersistentCompaction(mode: PersistentCompactionMode): boolean {
  return mode === "headroom";
}

export function delegatesPersistentCompaction(mode: PersistentCompactionMode): boolean {
  return mode === "openclaw" || mode === "hybrid";
}
