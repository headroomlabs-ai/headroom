import { execFile } from "node:child_process";
import * as fs from "node:fs/promises";
import * as os from "node:os";
import * as path from "node:path";
import { fileURLToPath } from "node:url";
import { promisify } from "node:util";

import { afterEach, beforeEach, describe, expect, it } from "vitest";

import { DurableAdvancementKeyStore, defaultCommitLogPath } from "../src/advancement-key-store.js";

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
const HOLD_LOCK_WORKER_PATH = fileURLToPath(
  new URL("./fixtures/hold-lock-worker.ts", import.meta.url),
);

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

async function waitForFile(p: string): Promise<void> {
  for (;;) {
    try {
      await fs.access(p);
      return;
    } catch {
      await new Promise((r) => setTimeout(r, 5));
    }
  }
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

  it("commits a new key and creates the parent directory", async () => {
    const store = new DurableAdvancementKeyStore(storePath);

    await expect(store.tryCommit("turn-1", [{ role: "user", content: "hi" }])).resolves.toBe(
      "committed",
    );
    await expect(fs.stat(storePath)).resolves.toBeTruthy();
  });

  it("reports duplicate for an already-committed key", async () => {
    const store = new DurableAdvancementKeyStore(storePath);

    await expect(store.tryCommit("turn-1", [])).resolves.toBe("committed");
    await expect(store.tryCommit("turn-1", [])).resolves.toBe("duplicate");
  });

  it("persists across a simulated restart (a fresh store instance, same path)", async () => {
    const before = new DurableAdvancementKeyStore(storePath);
    await expect(before.tryCommit("turn-1", [])).resolves.toBe("committed");

    const after = new DurableAdvancementKeyStore(storePath);
    await expect(after.tryCommit("turn-1", [])).resolves.toBe("duplicate");
    // A key that was never committed is still accepted normally.
    await expect(after.tryCommit("turn-2", [])).resolves.toBe("committed");
  });

  it("treats a missing commit-log file as an empty store rather than an error", async () => {
    const store = new DurableAdvancementKeyStore(storePath);

    await expect(store.has("turn-1")).resolves.toBe(false);
    await expect(store.tryCommit("turn-1", [])).resolves.toBe("committed");
  });

  it("fails loudly on a corrupt commit-log file instead of silently forgetting its contents", async () => {
    await fs.mkdir(path.dirname(storePath), { recursive: true });
    await fs.writeFile(storePath, "{not valid json", "utf8");
    const store = new DurableAdvancementKeyStore(storePath);

    await expect(store.tryCommit("turn-1", [])).rejects.toThrow();
  });

  it("serializes concurrent commits of the same key so exactly one wins", async () => {
    const store = new DurableAdvancementKeyStore(storePath);

    const results = await Promise.all([
      store.tryCommit("turn-race", []),
      store.tryCommit("turn-race", []),
      store.tryCommit("turn-race", []),
    ]);

    expect(results.filter((r) => r === "committed")).toHaveLength(1);
    expect(results.filter((r) => r === "duplicate")).toHaveLength(2);
  });

  it(
    "never evicts a key regardless of how many others were committed since",
    async () => {
      const store = new DurableAdvancementKeyStore(storePath);

      for (let i = 0; i < 600; i++) {
        await store.tryCommit(`turn-${i}`, []);
      }

      await expect(store.tryCommit("turn-0", [])).resolves.toBe("duplicate");
    },
    20_000,
  );

  // ---------------------------------------------------------------------
  // PR #3442 review, round 2: three additional atomic-commit failures.
  // ---------------------------------------------------------------------

  it("persists the accepted messages together with the key, not just the key", async () => {
    const store = new DurableAdvancementKeyStore(storePath);
    const messages = [
      { role: "user", content: "what's the weather?" },
      { role: "assistant", content: "Let me check." },
    ];

    await store.tryCommit("turn-1", messages);

    const entry = await store.get("turn-1");
    expect(entry).toBeDefined();
    expect(entry?.messages).toEqual(messages);
    expect(typeof entry?.committedAt).toBe("string");
    expect(entry?.committedAt.length).toBeGreaterThan(0);

    // Survives a restart too -- the messages aren't an in-memory-only echo.
    const after = new DurableAdvancementKeyStore(storePath);
    const reloaded = await after.get("turn-1");
    expect(reloaded?.messages).toEqual(messages);
  });

  it("does not remember a key when storage is inaccessible, so a retry can still succeed", async () => {
    // Block the parent directory with a file. Both the commit's initial
    // read and a membership lookup must propagate the filesystem error.
    const blockedParent = path.join(dir, "blocked");
    await fs.writeFile(blockedParent, "not a directory", "utf8");
    const blockedPath = path.join(blockedParent, "commit-log.json");
    const store = new DurableAdvancementKeyStore(blockedPath);

    await expect(store.tryCommit("turn-1", [])).rejects.toThrow();
    // The failed write must not be remembered in memory: retrying the same
    // key must attempt the commit again (and fail again, for the same
    // reason) rather than silently reporting "duplicate" for a commit that
    // never reached disk.
    await expect(store.tryCommit("turn-1", [])).rejects.toThrow();
    await expect(store.has("turn-1")).rejects.toThrow();

    // Once the underlying problem is fixed, the key is still absent and
    // the same key commits normally.
    await fs.rm(blockedParent, { force: true });
    await expect(store.has("turn-1")).resolves.toBe(false);
    await expect(store.tryCommit("turn-1", [])).resolves.toBe("committed");
    await expect(store.has("turn-1")).resolves.toBe(true);
  });

  it("serializes across store instances sharing the same file, not only within one object", async () => {
    const storeA = new DurableAdvancementKeyStore(storePath);
    const storeB = new DurableAdvancementKeyStore(storePath);

    // Interleaved commits through two independent instances against the
    // same file must not lose either one to a read-modify-write race.
    const [resultA, resultB] = await Promise.all([
      storeA.tryCommit("turn-a", ["from a"]),
      storeB.tryCommit("turn-b", ["from b"]),
    ]);

    expect(resultA).toBe("committed");
    expect(resultB).toBe("committed");

    // A third, fresh instance (simulating a restart) must see both.
    const after = new DurableAdvancementKeyStore(storePath);
    await expect(after.has("turn-a")).resolves.toBe(true);
    await expect(after.has("turn-b")).resolves.toBe(true);
    await expect(after.tryCommit("turn-a", [])).resolves.toBe("duplicate");
    await expect(after.tryCommit("turn-b", [])).resolves.toBe("duplicate");
  });

  // -------------------------------------------------------------------
  // PR #3442 review, round 3: an in-process lock (a JS Map/promise queue)
  // does nothing to stop two separate OS processes -- e.g. two gateway
  // processes sharing one Headroom workspace -- from racing the same
  // commit-log file. These spawn the real DurableAdvancementKeyStore in
  // real child processes (not Worker threads, which share the parent's
  // memory and wouldn't exercise inter-process file locking) and release
  // them via a shared "go" file so their read/check/write transactions
  // contend for the file lock as close to simultaneously as possible.
  // -------------------------------------------------------------------

  it.skipIf(!NODE_SUPPORTS_TRANSFORM_TYPES)(
    "serializes commits from two separate OS processes sharing the same file",
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

      // Neither commit was lost to a cross-process race.
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

      // Exactly one process's transaction should win the race and see
      // "committed"; the other must see "duplicate", never both
      // "committed" (which would mean the file lock let them interleave).
      const statuses = [resultA.status, resultB.status].sort();
      expect(statuses).toEqual(["committed", "duplicate"]);
    },
    30_000,
  );

  it.skipIf(!NODE_SUPPORTS_TRANSFORM_TYPES)(
    "never lets a second writer commit while a live holder still holds the lock",
    async () => {
      // Regression for PR #3442 review, round 4: a lock reclaimed on an
      // age-based staleness guess let a second process commit and release
      // while the first (merely slow, not dead) holder still believed it
      // owned the lock -- the first then resumed and overwrote the
      // second's accepted commit with its own stale snapshot. This proves
      // the invariant that replaces that heuristic: a second writer must
      // never proceed while ANY holder -- however long it takes -- still
      // holds the lock. Deterministic and fast: the "holder" is a worker
      // that occupies the real lock file and only releases it when told
      // to, rather than relying on a real multi-second sleep.
      const readyPath = path.join(dir, "holder-ready");
      const releasePath = path.join(dir, "holder-release");

      const holder = execFileAsync(process.execPath, [
        "--experimental-transform-types",
        HOLD_LOCK_WORKER_PATH,
        storePath,
        readyPath,
        releasePath,
      ]);

      // Wait until the holder has actually acquired the lock before racing
      // a commit against it.
      await waitForFile(readyPath);

      const store = new DurableAdvancementKeyStore(storePath);
      let secondCommitSettled = false;
      const secondCommit = store.tryCommit("turn-b", ["b"]).finally(() => {
        secondCommitSettled = true;
      });

      // Give the second commit ample opportunity to (wrongly) proceed
      // while the holder is still alive and has not released.
      try {
        await new Promise((r) => setTimeout(r, 500));
        expect(secondCommitSettled).toBe(false);
      } finally {
        // Release even if the assertion fails, so the child cannot leak.
        await fs.writeFile(releasePath, "go", "utf8");
        await holder;
      }
      // The second commit must now complete successfully, with nothing lost.
      await expect(secondCommit).resolves.toBe("committed");
      await expect(store.has("turn-b")).resolves.toBe(true);
    },
    15_000,
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
});
