import { execFile } from "node:child_process";
import { createHash } from "node:crypto";
import * as fs from "node:fs/promises";
import * as os from "node:os";
import * as path from "node:path";
import { fileURLToPath } from "node:url";
import { promisify } from "node:util";

import { afterEach, beforeEach, describe, expect, it } from "vitest";

import {
  DurableAdvancementKeyStore,
  defaultCommitLogPath,
  markerDirFor,
} from "../src/advancement-key-store.js";

const execFileAsync = promisify(execFile);

// Node's native TypeScript support (needed to run the worker fixture, which
// imports the real DurableAdvancementKeyStore, as a plain `node` child
// process with no build step) requires `--experimental-transform-types`,
// added in Node 22.6. Skip the cross-process tests rather than fail on an
// older Node -- the failure mode being tested (a lost write across two OS
// processes) is orthogonal to which Node version happens to run the test.
const NODE_SUPPORTS_TRANSFORM_TYPES = (() => {
  const [major, minor] = process.versions.node.split(".").map(Number);
  return major > 22 || (major === 22 && minor >= 6);
})();

const WORKER_PATH = fileURLToPath(new URL("./fixtures/commit-turn-worker.ts", import.meta.url));

/** Run the commit-turn worker as a real child process; returns its parsed JSON result. */
async function runCommitWorker(
  storePath: string,
  key: string,
  goPath: string,
  resultPath: string,
): Promise<{ status?: string; error?: string }> {
  await execFileAsync(process.execPath, [
    "--experimental-transform-types",
    WORKER_PATH,
    storePath,
    key,
    goPath,
    resultPath,
  ]);
  return JSON.parse(await fs.readFile(resultPath, "utf8"));
}

/** Every file under `dir`, recursively. */
async function filesUnder(dir: string): Promise<string[]> {
  const entries = await fs.readdir(dir, { recursive: true, withFileTypes: true });
  return entries.filter((e) => e.isFile()).map((e) => path.join(e.parentPath, e.name));
}

/** Where `key`'s marker lives: `<marker dir>/<first two hex digits of its sha256>/<sha256>`. */
function markerPathFor(storePath: string, key: string): string {
  const digest = createHash("sha256").update(key, "utf8").digest("hex");
  return path.join(markerDirFor(storePath), digest.slice(0, 2), digest);
}

describe("DurableAdvancementKeyStore", () => {
  let dir: string;
  let storePath: string;

  beforeEach(async () => {
    dir = await fs.mkdtemp(path.join(os.tmpdir(), "headroom-advancement-key-store-"));
    storePath = path.join(dir, "nested", "commit-log.json");
  });

  afterEach(async () => {
    await fs.rm(dir, { recursive: true, force: true });
  });

  it("commits a new key and creates the marker directory", async () => {
    const store = new DurableAdvancementKeyStore(storePath);

    await expect(store.tryCommit("turn-1")).resolves.toBe("committed");
    await expect(fs.stat(markerDirFor(storePath))).resolves.toBeTruthy();
  });

  it("reports duplicate for an already-committed key", async () => {
    const store = new DurableAdvancementKeyStore(storePath);

    await expect(store.tryCommit("turn-1")).resolves.toBe("committed");
    await expect(store.tryCommit("turn-1")).resolves.toBe("duplicate");
  });

  it("persists across a simulated restart (a fresh store instance, same path)", async () => {
    const before = new DurableAdvancementKeyStore(storePath);
    await expect(before.tryCommit("turn-1")).resolves.toBe("committed");

    const after = new DurableAdvancementKeyStore(storePath);
    await expect(after.tryCommit("turn-1")).resolves.toBe("duplicate");
    // A key that was never committed is still accepted normally.
    await expect(after.tryCommit("turn-2")).resolves.toBe("committed");
  });

  it("treats a store that doesn't exist yet as empty rather than an error", async () => {
    const store = new DurableAdvancementKeyStore(storePath);

    await expect(store.has("turn-1")).resolves.toBe(false);
    await expect(store.tryCommit("turn-1")).resolves.toBe("committed");
  });

  it("serializes concurrent commits of the same key so exactly one wins", async () => {
    const store = new DurableAdvancementKeyStore(storePath);

    const results = await Promise.all([
      store.tryCommit("turn-race"),
      store.tryCommit("turn-race"),
      store.tryCommit("turn-race"),
    ]);

    expect(results.filter((r) => r === "committed")).toHaveLength(1);
    expect(results.filter((r) => r === "duplicate")).toHaveLength(2);
  });

  it(
    "never evicts a key regardless of how many others were committed since",
    async () => {
      const store = new DurableAdvancementKeyStore(storePath);

      for (let i = 0; i < 600; i++) {
        await store.tryCommit(`turn-${i}`);
      }

      await expect(store.tryCommit("turn-0")).resolves.toBe("duplicate");
    },
    20_000,
  );

  it("records each commit as one small marker holding only the key", async () => {
    const store = new DurableAdvancementKeyStore(storePath);

    await store.tryCommit("turn-1");

    const files = await filesUnder(markerDirFor(storePath));
    expect(files).toHaveLength(1);
    const record = JSON.parse(await fs.readFile(files[0], "utf8"));
    expect(record.advancementKey).toBe("turn-1");
    expect(typeof record.committedAt).toBe("string");
    expect((await fs.stat(files[0])).size).toBeLessThan(200);
  });

  it("never rewrites an earlier commit's record when later keys are committed", async () => {
    // Regression: every commit used to re-read and rewrite one file holding every
    // key so far, so a commit's cost grew with the number of turns ever committed.
    const store = new DurableAdvancementKeyStore(storePath);
    await store.tryCommit("turn-0");
    const [first] = await filesUnder(markerDirFor(storePath));
    const before = await fs.stat(first);

    for (let i = 1; i <= 50; i++) {
      await store.tryCommit(`turn-${i}`);
    }

    const after = await fs.stat(first);
    expect(after.ino).toBe(before.ino);
    expect(after.mtimeMs).toBe(before.mtimeMs);
  });

  it("does not remember a key when storage is inaccessible, so a retry can still succeed", async () => {
    // Block the parent directory with a file. Both a commit and a membership
    // lookup must propagate the filesystem error.
    const blockedParent = path.join(dir, "blocked");
    await fs.writeFile(blockedParent, "not a directory", "utf8");
    const blockedPath = path.join(blockedParent, "commit-log.json");
    const store = new DurableAdvancementKeyStore(blockedPath);

    // The error names the setting that fixes it.
    await expect(store.tryCommit("turn-1")).rejects.toThrow(/HEADROOM_WORKSPACE_DIR/);
    // The failed write must not be remembered: retrying the same key must
    // attempt the commit again (and fail again, for the same reason) rather
    // than report "duplicate" for a commit that never reached disk.
    await expect(store.tryCommit("turn-1")).rejects.toThrow();
    await expect(store.has("turn-1")).rejects.toThrow();

    // Once the underlying problem is fixed, the key is still absent and the
    // same key commits normally.
    await fs.rm(blockedParent, { force: true });
    await expect(store.has("turn-1")).resolves.toBe(false);
    await expect(store.tryCommit("turn-1")).resolves.toBe("committed");
    await expect(store.has("turn-1")).resolves.toBe(true);
  });

  // Regression: creating a marker's directory and creating the marker itself
  // both fail with EEXIST, and both were read as "duplicate". A file where the
  // shard directory belongs acknowledged a key that was never recorded.
  it.each([
    ["the marker directory", (storePath: string) => markerDirFor(storePath)],
    ["the key's shard directory", (storePath: string) => path.dirname(markerPathFor(storePath, "turn-1"))],
  ])("reports a file in place of %s as an error, never as a duplicate", async (_where, blockedPath) => {
    const blocker = blockedPath(storePath);
    await fs.mkdir(path.dirname(blocker), { recursive: true });
    await fs.writeFile(blocker, "not a directory", "utf8");
    const store = new DurableAdvancementKeyStore(storePath);

    await expect(store.tryCommit("turn-1")).rejects.toThrow(/HEADROOM_WORKSPACE_DIR/);
    await expect(store.tryCommit("turn-1")).rejects.toThrow(/HEADROOM_WORKSPACE_DIR/);

    await fs.rm(blocker);
    await expect(store.tryCommit("turn-1")).resolves.toBe("committed");
    await expect(store.tryCommit("turn-1")).resolves.toBe("duplicate");
  });

  it("reports anything but a marker file at a marker's path as an error, never as a duplicate", async () => {
    const marker = markerPathFor(storePath, "turn-1");
    await fs.mkdir(marker, { recursive: true });
    const store = new DurableAdvancementKeyStore(storePath);

    await expect(store.tryCommit("turn-1")).rejects.toThrow(/HEADROOM_WORKSPACE_DIR/);
    await expect(store.has("turn-1")).resolves.toBe(false);

    await fs.rm(marker, { recursive: true });
    await expect(store.tryCommit("turn-1")).resolves.toBe("committed");
    await expect(store.has("turn-1")).resolves.toBe(true);
  });

  it("is never blocked by a lock file a crashed process left behind", async () => {
    // Regression: the previous store serialized commits with `<path>.lock` and,
    // by design, never reclaimed it, so a gateway killed mid-commit left every
    // later commit waiting 60 seconds and then failing until someone deleted it.
    await fs.mkdir(path.dirname(storePath), { recursive: true });
    await fs.writeFile(`${storePath}.lock`, "4242-1-abc\n4242\n2026-10-14T00:00:00.000Z\n", "utf8");
    const store = new DurableAdvancementKeyStore(storePath);

    await expect(store.tryCommit("turn-1")).resolves.toBe("committed");
  }, 5_000);

  it("commits through store instances sharing the same directory, not only within one object", async () => {
    const storeA = new DurableAdvancementKeyStore(storePath);
    const storeB = new DurableAdvancementKeyStore(storePath);

    const [resultA, resultB] = await Promise.all([storeA.tryCommit("turn-a"), storeB.tryCommit("turn-b")]);

    expect(resultA).toBe("committed");
    expect(resultB).toBe("committed");

    // A third, fresh instance (simulating a restart) must see both.
    const after = new DurableAdvancementKeyStore(storePath);
    await expect(after.tryCommit("turn-a")).resolves.toBe("duplicate");
    await expect(after.tryCommit("turn-b")).resolves.toBe("duplicate");
  });

  describe("a commit log written by an earlier version", () => {
    async function writeLegacyLog(contents: unknown): Promise<void> {
      await fs.mkdir(path.dirname(storePath), { recursive: true });
      await fs.writeFile(storePath, JSON.stringify(contents), "utf8");
    }

    /** Rewrite the log the way an older version commits a key: a temp file renamed over it. */
    async function olderVersionWrites(contents: unknown): Promise<void> {
      const temp = `${storePath}.tmp`;
      await fs.writeFile(temp, JSON.stringify(contents), "utf8");
      await fs.rename(temp, storePath);
    }

    it("keeps its keys committed and leaves the file as it was, for any older version still using it", async () => {
      await writeLegacyLog({
        "turn-old": { messages: [{ role: "user", content: "hi" }], committedAt: "2026-10-01T00:00:00.000Z" },
      });
      const log = await fs.readFile(storePath, "utf8");
      const store = new DurableAdvancementKeyStore(storePath);

      await expect(store.tryCommit("turn-old")).resolves.toBe("duplicate");
      await expect(store.tryCommit("turn-new")).resolves.toBe("committed");
      await expect(fs.readFile(storePath, "utf8")).resolves.toBe(log);
    });

    it("reads the key-only array format too", async () => {
      await writeLegacyLog(["turn-a", "turn-b"]);
      const store = new DurableAdvancementKeyStore(storePath);

      await expect(store.tryCommit("turn-a")).resolves.toBe("duplicate");
      await expect(store.tryCommit("turn-b")).resolves.toBe("duplicate");
    });

    it("is migrated before the first commit of every store, so no store re-accepts a legacy key", async () => {
      await writeLegacyLog({ "turn-old": { messages: [], committedAt: "" } });

      const results = await Promise.all([
        new DurableAdvancementKeyStore(storePath).tryCommit("turn-old"),
        new DurableAdvancementKeyStore(storePath).tryCommit("turn-old"),
      ]);

      expect(results).toEqual(["duplicate", "duplicate"]);
    });

    it("never re-accepts a key an older version commits while the log is being migrated", async () => {
      await writeLegacyLog({ "turn-a": { messages: [], committedAt: "" }, "turn-b": { messages: [], committedAt: "" } });
      const store = new DurableAdvancementKeyStore(storePath);
      // The older version commits turn-late after this store has read the log,
      // while it is creating the markers for what it read.
      const internals = store as unknown as { createMarker(key: string, committedAt: string): Promise<boolean> };
      const createMarker = internals.createMarker.bind(store);
      let olderVersionCommitted = false;
      internals.createMarker = async (key, committedAt) => {
        if (!olderVersionCommitted) {
          olderVersionCommitted = true;
          await olderVersionWrites({
            "turn-a": { committedAt: "" },
            "turn-b": { committedAt: "" },
            "turn-late": { committedAt: "" },
          });
        }
        return createMarker(key, committedAt);
      };

      await expect(store.tryCommit("turn-new")).resolves.toBe("committed");
      expect(olderVersionCommitted).toBe(true);
      await expect(store.has("turn-late")).resolves.toBe(true);
      await expect(store.tryCommit("turn-late")).resolves.toBe("duplicate");
      await expect(new DurableAdvancementKeyStore(storePath).tryCommit("turn-late")).resolves.toBe("duplicate");
    });

    it("creates markers only for the keys an older version adds when it rewrites the log", async () => {
      await writeLegacyLog({ "turn-a": { messages: [], committedAt: "" }, "turn-b": { messages: [], committedAt: "" } });
      const store = new DurableAdvancementKeyStore(storePath);
      await expect(store.tryCommit("turn-new")).resolves.toBe("committed");

      const internals = store as unknown as { createMarker(key: string, committedAt: string): Promise<boolean> };
      const createMarker = internals.createMarker.bind(store);
      const markerAttempts: string[] = [];
      internals.createMarker = async (key, committedAt) => {
        markerAttempts.push(key);
        return createMarker(key, committedAt);
      };
      await olderVersionWrites({
        "turn-a": { committedAt: "" },
        "turn-b": { committedAt: "" },
        "turn-c": { committedAt: "" },
      });

      await expect(store.tryCommit("turn-next")).resolves.toBe("committed");
      expect(markerAttempts).toEqual(["turn-c", "turn-next"]);
      await expect(store.tryCommit("turn-c")).resolves.toBe("duplicate");
    });

    // Permission bits don't stop root, and Windows doesn't apply them the same way.
    it.skipIf(process.platform === "win32" || process.getuid?.() === 0)(
      "keeps committing when the migration record can't be written",
      async () => {
        await writeLegacyLog({ "turn-old": { messages: [], committedAt: "" } });
        // A record this process may not overwrite, say one another user wrote.
        const record = path.join(markerDirFor(storePath), "migrated-legacy-log.json");
        await fs.mkdir(markerDirFor(storePath), { recursive: true });
        await fs.writeFile(record, "{}\n", "utf8");
        await fs.chmod(record, 0o444);

        const store = new DurableAdvancementKeyStore(storePath);
        await expect(store.tryCommit("turn-old")).resolves.toBe("duplicate");
        await expect(store.tryCommit("turn-new")).resolves.toBe("committed");
        // A store that starts later reads the log again, and still knows its keys.
        await expect(new DurableAdvancementKeyStore(storePath).tryCommit("turn-old")).resolves.toBe("duplicate");
      },
    );

    // Permission bits don't stop root, and Windows doesn't apply them to directories.
    it.skipIf(process.platform === "win32" || process.getuid?.() === 0)(
      "keeps commits working while the log's directory is read-only, and migrates what an older version adds later",
      async () => {
        // A writable marker volume mounted beneath a read-only directory that
        // holds the legacy log: its keys migrate without writing to that directory.
        await writeLegacyLog({ "turn-a": { messages: [], committedAt: "" } });
        const written = new Date("2026-10-01T00:00:00.000Z");
        await fs.utimes(storePath, written, written);
        const logDir = path.dirname(storePath);
        await fs.mkdir(markerDirFor(storePath));
        await fs.chmod(logDir, 0o555);
        try {
          const store = new DurableAdvancementKeyStore(storePath);
          await expect(store.tryCommit("turn-a")).resolves.toBe("duplicate");
          await expect(store.tryCommit("turn-new")).resolves.toBe("committed");
          const log = await fs.readFile(storePath, "utf8");
          expect(log).toContain("turn-a");

          // A restart doesn't read the unchanged log again. Probe: contents that
          // can't be parsed, at the same size and mtime.
          await fs.writeFile(storePath, "x".repeat(log.length), "utf8");
          await fs.utimes(storePath, written, written);
          const restarted = new DurableAdvancementKeyStore(storePath);
          await expect(restarted.tryCommit("turn-next")).resolves.toBe("committed");

          // Nor does a migrated log that can no longer be read block commits.
          await fs.chmod(storePath, 0o000);
          await expect(new DurableAdvancementKeyStore(storePath).tryCommit("turn-c")).resolves.toBe("committed");

          // The directory becomes writable, and an older version that is still
          // running commits turn-b the way it always did (temp file + rename).
          // The running store migrates it before its next commit.
          await fs.chmod(logDir, 0o755);
          await olderVersionWrites({ "turn-a": { committedAt: "" }, "turn-b": { committedAt: "" } });
          await expect(restarted.tryCommit("turn-b")).resolves.toBe("duplicate");
        } finally {
          await fs.chmod(logDir, 0o755);
        }
      },
    );

    it("fails loudly when corrupt instead of silently forgetting its keys", async () => {
      await fs.mkdir(path.dirname(storePath), { recursive: true });
      await fs.writeFile(storePath, "{not valid json", "utf8");
      const store = new DurableAdvancementKeyStore(storePath);

      await expect(store.tryCommit("turn-1")).rejects.toThrow();
      // Fixing the file lets the same store recover without a restart.
      await fs.writeFile(storePath, JSON.stringify(["turn-1"]), "utf8");
      await expect(store.tryCommit("turn-1")).resolves.toBe("duplicate");
    });
  });

  // -------------------------------------------------------------------
  // Two separate OS processes -- e.g. two gateway processes sharing one
  // Headroom workspace -- committing at the same moment. These spawn the
  // real DurableAdvancementKeyStore in real child processes (not Worker
  // threads, which share the parent's memory) and release them via a shared
  // "go" file so their commits race as closely as possible.
  // -------------------------------------------------------------------

  it.skipIf(!NODE_SUPPORTS_TRANSFORM_TYPES)(
    "keeps commits from two separate OS processes sharing the same directory",
    async () => {
      const goPath = path.join(dir, "go");
      const resultAPath = path.join(dir, "result-a.json");
      const resultBPath = path.join(dir, "result-b.json");

      const workerA = runCommitWorker(storePath, "turn-a", goPath, resultAPath);
      const workerB = runCommitWorker(storePath, "turn-b", goPath, resultBPath);
      // Give both worker processes time to spawn and reach their poll loop
      // before releasing them together.
      await new Promise((r) => setTimeout(r, 300));
      await fs.writeFile(goPath, "go", "utf8");

      const [resultA, resultB] = await Promise.all([workerA, workerB]);

      expect(resultA).toEqual({ status: "committed" });
      expect(resultB).toEqual({ status: "committed" });

      const after = new DurableAdvancementKeyStore(storePath);
      await expect(after.has("turn-a")).resolves.toBe(true);
      await expect(after.has("turn-b")).resolves.toBe(true);
    },
    30_000,
  );

  it.skipIf(!NODE_SUPPORTS_TRANSFORM_TYPES)(
    "reports duplicate for the same key committed from two separate OS processes",
    async () => {
      const goPath = path.join(dir, "go");
      const resultAPath = path.join(dir, "result-a.json");
      const resultBPath = path.join(dir, "result-b.json");

      const workerA = runCommitWorker(storePath, "turn-same", goPath, resultAPath);
      const workerB = runCommitWorker(storePath, "turn-same", goPath, resultBPath);
      await new Promise((r) => setTimeout(r, 300));
      await fs.writeFile(goPath, "go", "utf8");

      const [resultA, resultB] = await Promise.all([workerA, workerB]);

      // Exactly one process creates the marker and sees "committed"; the
      // other must see "duplicate", never both "committed".
      const statuses = [resultA.status, resultB.status].sort();
      expect(statuses).toEqual(["committed", "duplicate"]);
    },
    30_000,
  );
});

describe("defaultCommitLogPath", () => {
  const originalWorkspaceDir = process.env.HEADROOM_WORKSPACE_DIR;

  afterEach(() => {
    if (originalWorkspaceDir === undefined) {
      delete process.env.HEADROOM_WORKSPACE_DIR;
    } else {
      process.env.HEADROOM_WORKSPACE_DIR = originalWorkspaceDir;
    }
  });

  it("honors HEADROOM_WORKSPACE_DIR when set", () => {
    process.env.HEADROOM_WORKSPACE_DIR = "/custom/workspace";

    expect(defaultCommitLogPath()).toBe(
      path.join("/custom/workspace", "openclaw", "commit-log.json"),
    );
  });

  it("falls back to ~/.headroom when unset", () => {
    delete process.env.HEADROOM_WORKSPACE_DIR;

    expect(defaultCommitLogPath()).toBe(
      path.join(os.homedir(), ".headroom", "openclaw", "commit-log.json"),
    );
  });

  it("keeps the markers beside the commit log", () => {
    expect(markerDirFor(path.join("/w", "openclaw", "commit-log.json"))).toBe(
      path.join("/w", "openclaw", "commit-log.d"),
    );
  });
});
