/**
 * Cross-process exclusive file lock for the turn advancement store.
 *
 * Protocol:
 *   1. Ownership is published atomically. The owner record is written to a
 *      private temp file and hard-linked to the lock path (`link(2)` fails with
 *      EEXIST when a lock exists). The lock file is therefore never observable
 *      in an empty / half-written state. Filesystems without hard links fall
 *      back to `open(O_EXCL)`; the staleness rules below keep that safe too.
 *   2. A lock is only considered stale when it is *old* (mtime older than
 *      `staleAfterMs`) **and** its owner is provably gone (ESRCH on probe, or an
 *      unreadable owner record). A fresh lock, an empty lock, or an EPERM probe
 *      result never qualifies. A hard ceiling (`hardStaleAfterMs`) recovers
 *      locks abandoned by a reused PID.
 *   3. Stale recovery is claimed with an atomic `rename` to a unique path before
 *      unlinking, so two waiters cannot both "recover" and one of them delete a
 *      lock that a third writer just acquired.
 */

import { randomBytes } from "node:crypto";
import {
  closeSync,
  linkSync,
  mkdirSync,
  openSync,
  readFileSync,
  renameSync,
  statSync,
  unlinkSync,
  writeFileSync,
  writeSync,
} from "node:fs";
import { dirname } from "node:path";

export interface StoreLockOwner {
  pid: number;
  token: string;
  createdAtMs: number;
}

export interface StoreLockOptions {
  /** Poll interval while waiting for a held lock. */
  retryIntervalMs?: number;
  /** Give up and throw after this long. */
  timeoutMs?: number;
  /** Minimum lock age before a dead-owner lock may be recovered. */
  staleAfterMs?: number;
  /** Lock age after which recovery happens even if the PID probe says alive. */
  hardStaleAfterMs?: number;
}

export const DEFAULT_STORE_LOCK_OPTIONS: Required<StoreLockOptions> = {
  retryIntervalMs: 10,
  timeoutMs: 5_000,
  staleAfterMs: 30_000,
  hardStaleAfterMs: 10 * 60_000,
};

const LINK_UNSUPPORTED_CODES = new Set(["EPERM", "ENOTSUP", "EOPNOTSUPP", "EXDEV", "EINVAL", "ENOSYS"]);

type OwnerLiveness = "alive" | "dead" | "unknown";

function sleepSync(ms: number): void {
  // Real sleep without burning CPU; keeps commit() synchronous for callers.
  Atomics.wait(new Int32Array(new SharedArrayBuffer(4)), 0, 0, ms);
}

function readOwner(lockPath: string): StoreLockOwner | null {
  try {
    const parsed = JSON.parse(readFileSync(lockPath, "utf8")) as Partial<StoreLockOwner>;
    if (
      typeof parsed.pid === "number" &&
      Number.isInteger(parsed.pid) &&
      parsed.pid > 0 &&
      typeof parsed.token === "string"
    ) {
      return {
        pid: parsed.pid,
        token: parsed.token,
        createdAtMs: typeof parsed.createdAtMs === "number" ? parsed.createdAtMs : 0,
      };
    }
  } catch {
    // unreadable / partial / not JSON
  }
  return null;
}

/** EPERM means "exists but not ours" — that is proof of life, not death. */
export function probeOwnerLiveness(owner: StoreLockOwner | null): OwnerLiveness {
  if (!owner) return "unknown";
  try {
    process.kill(owner.pid, 0);
    return "alive";
  } catch (error) {
    const code = (error as NodeJS.ErrnoException).code;
    if (code === "ESRCH") return "dead";
    return "alive";
  }
}

export function isLockStale(params: {
  ageMs: number;
  liveness: OwnerLiveness;
  staleAfterMs: number;
  hardStaleAfterMs: number;
}): boolean {
  if (params.ageMs >= params.hardStaleAfterMs) return true;
  if (params.ageMs < params.staleAfterMs) return false;
  return params.liveness !== "alive";
}

/** Atomically claim a stale lock for removal. Returns false if someone else got there first. */
function recoverStaleLock(lockPath: string): boolean {
  const claimPath = `${lockPath}.stale.${process.pid}.${randomBytes(4).toString("hex")}`;
  try {
    renameSync(lockPath, claimPath);
  } catch {
    return false;
  }
  try {
    unlinkSync(claimPath);
  } catch {
    // The claim file is private to us; leaving it behind is harmless.
  }
  return true;
}

/** Try once to publish ownership. Returns true on acquisition, false if the lock is held. */
function tryAcquire(lockPath: string, owner: StoreLockOwner): boolean {
  const payload = `${JSON.stringify(owner)}\n`;
  const tmpPath = `${lockPath}.${owner.pid}.${owner.token}`;
  writeFileSync(tmpPath, payload, "utf8");
  try {
    linkSync(tmpPath, lockPath);
    return true;
  } catch (error) {
    const code = (error as NodeJS.ErrnoException).code;
    if (code === "EEXIST") return false;
    if (!code || !LINK_UNSUPPORTED_CODES.has(code)) throw error;
  } finally {
    try {
      unlinkSync(tmpPath);
    } catch {
      // already gone
    }
  }

  // Hard links unsupported: exclusive create + immediate owner write. The
  // window where the file is empty is covered by the age-based stale rule.
  let fd: number | undefined;
  try {
    fd = openSync(lockPath, "wx");
    writeSync(fd, payload);
    return true;
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code === "EEXIST") return false;
    throw error;
  } finally {
    if (fd !== undefined) closeSync(fd);
  }
}

export function withStoreLock<T>(lockPath: string, fn: () => T, options: StoreLockOptions = {}): T {
  const settings = { ...DEFAULT_STORE_LOCK_OPTIONS, ...options };
  mkdirSync(dirname(lockPath), { recursive: true });

  const owner: StoreLockOwner = {
    pid: process.pid,
    token: randomBytes(8).toString("hex"),
    createdAtMs: Date.now(),
  };
  const deadline = Date.now() + settings.timeoutMs;
  let acquired = false;

  while (!acquired) {
    acquired = tryAcquire(lockPath, owner);
    if (acquired) break;

    let ageMs = 0;
    try {
      ageMs = Math.max(0, Date.now() - statSync(lockPath).mtimeMs);
    } catch {
      continue; // released between our attempt and the stat — retry immediately
    }
    const liveness = probeOwnerLiveness(readOwner(lockPath));
    if (
      isLockStale({
        ageMs,
        liveness,
        staleAfterMs: settings.staleAfterMs,
        hardStaleAfterMs: settings.hardStaleAfterMs,
      })
    ) {
      recoverStaleLock(lockPath);
      continue;
    }

    if (Date.now() >= deadline) {
      throw new Error(
        `timed out acquiring turn advancement lock at ${lockPath} (held by pid ${readOwner(lockPath)?.pid ?? "unknown"}, age ${ageMs}ms)`,
      );
    }
    sleepSync(settings.retryIntervalMs);
  }

  try {
    return fn();
  } finally {
    // Only release if the lock is still ours (token match) — never unlink a
    // lock another writer acquired after a recovery.
    if (readOwner(lockPath)?.token === owner.token) {
      try {
        unlinkSync(lockPath);
      } catch {
        // Best effort — a leftover lock is recovered by the stale rules.
      }
    }
  }
}
