import type { PreparedEntry } from "./types.js";

interface RetrievalBinding {
  text: string;
  owners: Set<string>;
}

export interface CacheStats {
  entries: number;
  bytes: number;
  maxBytes: number;
}

export class PreparedCache {
  readonly #maxBytes: number;
  readonly #entries = new Map<string, PreparedEntry>();
  readonly #retrievals = new Map<string, RetrievalBinding>();
  #bytes = 0;

  constructor(maxBytes: number) {
    if (!Number.isInteger(maxBytes) || maxBytes <= 0) {
      throw new Error("maxBytes must be a positive integer");
    }
    this.#maxBytes = maxBytes;
  }

  get(key: string): PreparedEntry | undefined {
    const entry = this.#entries.get(key);
    if (!entry) return undefined;
    entry.lastAccessedAt = Date.now();
    this.#entries.delete(key);
    this.#entries.set(key, entry);
    return entry;
  }

  getRetrieval(hash: string): string | undefined {
    const binding = this.#retrievals.get(hash);
    if (!binding) return undefined;
    const owner = binding.owners.values().next().value;
    if (typeof owner === "string") this.get(owner);
    return binding.text;
  }

  set(entry: PreparedEntry): boolean {
    if (entry.sizeBytes > this.#maxBytes || entry.sizeBytes < 0) return false;

    for (const [hash, text] of entry.retrievals) {
      const binding = this.#retrievals.get(hash);
      const replacingSoleOwner =
        binding?.owners.size === 1 && binding.owners.has(entry.key);
      if (binding && binding.text !== text && !replacingSoleOwner) return false;
    }

    this.#delete(entry.key);
    while (
      this.#bytes + entry.sizeBytes > this.#maxBytes &&
      this.#entries.size > 0
    ) {
      const oldestKey = this.#entries.keys().next().value;
      if (typeof oldestKey !== "string") break;
      this.#delete(oldestKey);
    }

    this.#entries.set(entry.key, entry);
    this.#bytes += entry.sizeBytes;
    for (const [hash, text] of entry.retrievals) {
      const binding = this.#retrievals.get(hash);
      if (binding) binding.owners.add(entry.key);
      else {
        this.#retrievals.set(hash, {
          text,
          owners: new Set([entry.key]),
        });
      }
    }
    return true;
  }

  clear(): void {
    this.#entries.clear();
    this.#retrievals.clear();
    this.#bytes = 0;
  }

  stats(): CacheStats {
    return {
      entries: this.#entries.size,
      bytes: this.#bytes,
      maxBytes: this.#maxBytes,
    };
  }

  #delete(key: string): void {
    const entry = this.#entries.get(key);
    if (!entry) return;

    this.#entries.delete(key);
    this.#bytes -= entry.sizeBytes;
    for (const hash of entry.retrievals.keys()) {
      const binding = this.#retrievals.get(hash);
      if (!binding) continue;
      binding.owners.delete(key);
      if (binding.owners.size === 0) this.#retrievals.delete(hash);
    }
  }
}
