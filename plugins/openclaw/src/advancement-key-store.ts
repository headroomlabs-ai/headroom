/**
 * Durable, restart-safe, cross-process-safe record of committed turn
 * advancements.
 *
 * OpenClaw's `atomic-idempotent-v1` transcriptSemantics contract asks for one
 * atomic, idempotent write keyed by `advancementKey`: "committed" the first
 * time a key is presented, and "duplicate" for every retry of it -- including
 * a retry that arrives after this process (or the host) restarted, or one made
 * by another gateway process sharing the same Headroom workspace.
 *
 * Each committed key is one small marker file, created with exclusive creation
 * (`open(..., "wx")`, i.e. O_CREAT | O_EXCL), which is atomic at the OS level
 * on POSIX and Windows alike. Whichever caller creates the marker first -- in
 * this process or any other -- gets "committed"; every later caller gets
 * EEXIST, i.e. "duplicate". There is no lock to hold and no shared file to
 * rewrite, so a process that dies mid-commit can't block the commits after it,
 * and a commit costs the same on the first turn as on the hundred-thousandth.
 *
 * Only the key is recorded. This engine keeps no turn state of its own:
 * `assemble()` works from the messages OpenClaw passes in, and OpenClaw's own
 * transcript already persists every accepted turn, so for this engine the
 * advancement *is* the key. Earlier versions also kept a copy of each turn's
 * messages (OpenClaw allows up to 8 MiB per turn) in one JSON file that was
 * re-read and rewritten in full on every commit, which made the file, and the
 * time each commit took, grow without bound.
 *
 * Keys are never evicted. OpenClaw may retry at any point after a restart or a
 * long stall, so pruning by count or age would risk re-accepting an
 * already-committed turn -- exactly the failure this store exists to avoid. A
 * marker is a few dozen bytes.
 *
 * No commit is cached in memory, and only an existing marker file counts as a
 * commit: a commit that fails before its marker exists (an unwritable or
 * blocked directory) is retried for real, never answered with a phantom
 * "duplicate". A marker is created before its contents are written, so
 * a failure after creation (a full disk) still leaves the key committed: the
 * caller sees the error, and its retry sees "duplicate" -- never a second
 * "committed".
 */

import { createHash } from "node:crypto";
import { lstat, mkdir, open, readFile, stat, writeFile, type FileHandle } from "node:fs/promises";
import { dirname, join } from "node:path";
import * as os from "node:os";

/** Default commit-log location, honoring the same workspace override the
 * rest of the Headroom workspace convention uses (`HEADROOM_WORKSPACE_DIR`).
 * The markers live beside it, in `commit-log.d/`. */
export function defaultCommitLogPath(): string {
  const workspaceDir = process.env.HEADROOM_WORKSPACE_DIR?.trim() || join(os.homedir(), ".headroom");
  return join(workspaceDir, "openclaw", "commit-log.json");
}

/** The marker directory for a commit-log path: `commit-log.json` -> `commit-log.d`. */
export function markerDirFor(commitLogPath: string): string {
  return `${commitLogPath.replace(/\.json$/i, "")}.d`;
}

function errorCode(error: unknown): string | undefined {
  return (error as NodeJS.ErrnoException)?.code;
}

/**
 * Whether something other than a directory -- a regular file, say -- sits on
 * `dir`'s path, so that nothing beneath it can exist. POSIX reports such a
 * path as ENOTDIR, but Windows reports ENOENT, which reads exactly like an
 * absent marker; the nearest ancestor that exists tells the two apart.
 */
async function isBlockedPath(dir: string): Promise<boolean> {
  for (let current = dir; ; current = dirname(current)) {
    try {
      return !(await stat(current)).isDirectory();
    } catch (error) {
      const code = errorCode(error);
      if (code === "ENOTDIR") {
        return true;
      }
      if (code !== "ENOENT" || dirname(current) === current) {
        throw error;
      }
    }
  }
}

/** Identifies one version of a legacy log file: any rewrite changes at least one of these. */
type LegacyLogVersion = { size: number; mtimeMs: number; ino: number };

function sameVersion(a: LegacyLogVersion | undefined, b: LegacyLogVersion): boolean {
  return a?.size === b.size && a.mtimeMs === b.mtimeMs && a.ino === b.ino;
}

export class DurableAdvancementKeyStore {
  private readonly markerDir: string;
  /** The version of the legacy log whose keys all have markers. */
  private migratedVersion: LegacyLogVersion | undefined;
  /** Legacy keys known to have markers, so a rewrite of the log costs only the keys it adds. */
  private readonly migratedKeys = new Set<string>();
  private migration: Promise<void> = Promise.resolve();

  /**
   * `path` names the commit log; the markers live in `markerDirFor(path)`.
   * Versions that kept every key in the log file itself may have left one at
   * `path`. Before every commit and lookup, a log there that is new or has
   * changed has its keys migrated to markers. The log itself is never moved or
   * changed, so an older version still running against the same workspace
   * keeps its own record, and every key it commits, even while this store is
   * migrating the log, is migrated before this store's next commit.
   */
  constructor(private readonly path: string) {
    this.markerDir = markerDirFor(path);
  }

  private markerPath(key: string): string {
    const digest = createHash("sha256").update(key, "utf8").digest("hex");
    // Two-character fan-out keeps any one directory small.
    return join(this.markerDir, digest.slice(0, 2), digest);
  }

  /** Records which version of the legacy log was migrated last. */
  private migratedLogRecordPath(): string {
    return join(this.markerDir, "migrated-legacy-log.json");
  }

  private storageError(error: unknown): Error {
    // Node reports a directory it can't create on a read-only filesystem as
    // ENOENT, which reads like a bug rather than a setting -- say which one.
    return new Error(
      `Cannot record turn advancements under ${this.markerDir}: ${(error as Error)?.message ?? error}. ` +
        "Point HEADROOM_WORKSPACE_DIR (or the plugin's commitLogPath) at a writable directory.",
      { cause: error },
    );
  }

  /**
   * Whether a marker file -- not a directory or anything else -- exists at
   * `marker`. Storage that can't hold one (a file where a directory belongs)
   * is an error, never an absent marker.
   */
  private async isMarker(marker: string): Promise<boolean> {
    try {
      return (await lstat(marker)).isFile();
    } catch (error) {
      if (errorCode(error) === "ENOENT" && !(await isBlockedPath(dirname(marker)).catch(() => true))) {
        return false;
      }
      throw this.storageError(error);
    }
  }

  /** Create `key`'s marker; false if it already existed. */
  private async createMarker(key: string, committedAt: string): Promise<boolean> {
    const marker = this.markerPath(key);
    try {
      await mkdir(dirname(marker), { recursive: true });
    } catch (error) {
      // An EEXIST here means a file sits where the marker's directory belongs,
      // so no marker can exist beneath it: broken storage, not a duplicate.
      throw this.storageError(error);
    }
    let handle: FileHandle;
    try {
      handle = await open(marker, "wx");
    } catch (error) {
      // Only an existing marker file makes this a retry of a committed key.
      if (errorCode(error) === "EEXIST" && (await this.isMarker(marker).catch(() => false))) {
        return false;
      }
      throw this.storageError(error);
    }
    try {
      await handle.writeFile(`${JSON.stringify({ advancementKey: key, committedAt })}\n`, "utf8");
    } catch (error) {
      throw this.storageError(error);
    } finally {
      await handle.close();
    }
    return true;
  }

  private ensureMigrated(): Promise<void> {
    // One check at a time, each starting after its caller arrived: it sees every
    // change made before then, and a large log is never parsed twice at once. A
    // failed check isn't remembered, so the next call retries it.
    const check = this.migration.catch(() => undefined).then(() => this.migrateLegacyLog());
    this.migration = check;
    return check;
  }

  /**
   * Move the keys of a legacy single-file log into markers. Without a log this
   * is one failed open(), and with an unchanged one an open() and an fstat(), so
   * it runs before every commit. Safe to run in several processes at once:
   * creating a marker that already exists is a no-op.
   */
  private async migrateLegacyLog(): Promise<void> {
    let log: FileHandle;
    try {
      log = await open(this.path, "r");
    } catch (error) {
      const code = errorCode(error);
      if (code === "ENOENT" || code === "ENOTDIR") {
        return; // No legacy log. A blocked path fails loudly at the first commit.
      }
      if ((code === "EACCES" || code === "EPERM") && (await this.isMigratedLog())) {
        return; // Unreadable now, but every key it held has a marker, and it hasn't changed since.
      }
      throw error;
    }
    let version: LegacyLogVersion;
    let raw: string;
    try {
      // Version and contents come from the same open file, so they always match.
      const { size, mtimeMs, ino } = await log.stat();
      version = { size, mtimeMs, ino };
      this.migratedVersion ??= await this.readMigratedLogRecord();
      if (sameVersion(this.migratedVersion, version)) {
        return; // Migrated already, and unchanged since.
      }
      raw = await log.readFile("utf8");
    } finally {
      await log.close();
    }
    // A corrupt log throws here rather than being treated as empty, which would
    // silently re-accept every key it held.
    const parsed = JSON.parse(raw) as unknown;
    let entries: Array<[string, string]> = [];
    if (Array.isArray(parsed)) {
      entries = parsed.filter((key): key is string => typeof key === "string").map((key) => [key, ""]);
    } else if (parsed && typeof parsed === "object") {
      entries = Object.entries(parsed as Record<string, { committedAt?: unknown }>).map(([key, entry]) => [
        key,
        typeof entry?.committedAt === "string" ? entry.committedAt : "",
      ]);
    }
    for (const [key, committedAt] of entries) {
      if (this.migratedKeys.has(key)) {
        continue;
      }
      await this.createMarker(key, committedAt);
      this.migratedKeys.add(key);
    }
    // Every key in the version that was read has a marker now. Record which
    // version that was, so checks -- here and after a restart -- skip the log
    // while it is unchanged, and migrate it again once anything (an older
    // version still writing to it) changes it, including in the moments since
    // it was read. Moving the log aside instead would carry off whatever it
    // held by then, and leave an older version without its record.
    this.migratedVersion = version;
    try {
      await mkdir(this.markerDir, { recursive: true });
      await writeFile(this.migratedLogRecordPath(), `${JSON.stringify(version)}\n`, "utf8");
    } catch {
      // The record only saves work. Without it, a store that starts later
      // reads the log again, and commits carry on meanwhile.
    }
  }

  /** Whether the log's metadata alone identifies it as the version already migrated. */
  private async isMigratedLog(): Promise<boolean> {
    try {
      const { size, mtimeMs, ino } = await stat(this.path);
      this.migratedVersion ??= await this.readMigratedLogRecord();
      return sameVersion(this.migratedVersion, { size, mtimeMs, ino });
    } catch {
      return false;
    }
  }

  private async readMigratedLogRecord(): Promise<LegacyLogVersion | undefined> {
    try {
      return JSON.parse(await readFile(this.migratedLogRecordPath(), "utf8")) as LegacyLogVersion;
    } catch {
      return undefined; // None yet, or unreadable: migrating again is always safe.
    }
  }

  /**
   * Atomically record `key`: "committed" if this call created its marker,
   * "duplicate" if any earlier call -- in this process or another, before or
   * after a restart -- already had. Throws if the marker can't be created, so
   * the caller sees a rejected commit rather than a false answer, and a retry
   * attempts the commit again.
   */
  async tryCommit(key: string): Promise<"committed" | "duplicate"> {
    await this.ensureMigrated();
    return (await this.createMarker(key, new Date().toISOString())) ? "committed" : "duplicate";
  }

  /** Test-only: check membership without recording. */
  async has(key: string): Promise<boolean> {
    await this.ensureMigrated();
    return this.isMarker(this.markerPath(key));
  }
}
