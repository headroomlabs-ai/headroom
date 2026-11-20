import { spawn } from "node:child_process";
import {
  closeSync,
  existsSync,
  mkdtempSync,
  openSync,
  readFileSync,
  rmSync,
  utimesSync,
  writeFileSync,
} from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, describe, expect, it } from "vitest";
import { isLockStale, probeOwnerLiveness, withStoreLock } from "../src/store-lock";
import { TurnAdvancementStore } from "../src/turn-advancement-store";

const dirs: string[] = [];
function makeLockPath(): string {
  const dir = mkdtempSync(join(tmpdir(), "headroom-store-lock-"));
  dirs.push(dir);
  return join(dir, "store.json.lock");
}
afterEach(() => {
  for (const dir of dirs.splice(0)) rmSync(dir, { recursive: true, force: true });
});

function backdate(path: string, ageMs: number): void {
  const then = new Date(Date.now() - ageMs);
  utimesSync(path, then, then);
}

const FAST = { retryIntervalMs: 5, timeoutMs: 250, staleAfterMs: 1_000, hardStaleAfterMs: 5_000 };

describe("isLockStale", () => {
  it("never treats a fresh lock as stale, whatever the owner probe says", () => {
    for (const liveness of ["alive", "dead", "unknown"] as const) {
      expect(isLockStale({ ageMs: 10, liveness, staleAfterMs: 1_000, hardStaleAfterMs: 5_000 })).toBe(false);
    }
  });

  it("requires both age and a non-alive owner", () => {
    expect(isLockStale({ ageMs: 2_000, liveness: "alive", staleAfterMs: 1_000, hardStaleAfterMs: 5_000 })).toBe(false);
    expect(isLockStale({ ageMs: 2_000, liveness: "dead", staleAfterMs: 1_000, hardStaleAfterMs: 5_000 })).toBe(true);
    expect(isLockStale({ ageMs: 2_000, liveness: "unknown", staleAfterMs: 1_000, hardStaleAfterMs: 5_000 })).toBe(true);
  });

  it("recovers past the hard ceiling even when the pid probe says alive (pid reuse)", () => {
    expect(isLockStale({ ageMs: 6_000, liveness: "alive", staleAfterMs: 1_000, hardStaleAfterMs: 5_000 })).toBe(true);
  });
});

describe("probeOwnerLiveness", () => {
  it("reports the current process as alive and an unreadable owner as unknown", () => {
    expect(probeOwnerLiveness({ pid: process.pid, token: "t", createdAtMs: 0 })).toBe("alive");
    expect(probeOwnerLiveness(null)).toBe("unknown");
  });

  it("reports a definitely-dead pid as dead", () => {
    // PIDs near the max are effectively never live on Linux/macOS/Windows test hosts.
    expect(probeOwnerLiveness({ pid: 2_147_483_000, token: "t", createdAtMs: 0 })).not.toBe("alive");
  });
});

describe("withStoreLock", () => {
  it("publishes a complete owner record atomically and removes it on release", () => {
    const lockPath = makeLockPath();
    let sawOwner: unknown;
    withStoreLock(lockPath, () => {
      sawOwner = JSON.parse(readFileSync(lockPath, "utf8"));
    }, FAST);
    expect(sawOwner).toMatchObject({ pid: process.pid, token: expect.any(String) });
    expect(existsSync(lockPath)).toBe(false);
  });

  it("does NOT steal a freshly created empty lock (reviewer boundary: open(wx) before pid write)", () => {
    const lockPath = makeLockPath();
    const fd = openSync(lockPath, "wx"); // another writer mid-acquire: file exists, no owner yet
    try {
      const started = Date.now();
      expect(() => withStoreLock(lockPath, () => "ran", FAST)).toThrow(/timed out acquiring/);
      expect(Date.now() - started).toBeGreaterThanOrEqual(FAST.timeoutMs - 20);
      expect(existsSync(lockPath)).toBe(true); // the live lock was left alone
    } finally {
      closeSync(fd);
    }
  });

  it("does NOT steal an old lock whose owner is alive", () => {
    const lockPath = makeLockPath();
    writeFileSync(lockPath, JSON.stringify({ pid: process.pid, token: "live", createdAtMs: 0 }));
    backdate(lockPath, 2_000);
    expect(() => withStoreLock(lockPath, () => "ran", FAST)).toThrow(/timed out acquiring/);
    expect(JSON.parse(readFileSync(lockPath, "utf8")).token).toBe("live");
  });

  it("recovers an old lock whose owner is dead", () => {
    const lockPath = makeLockPath();
    writeFileSync(lockPath, JSON.stringify({ pid: 2_147_483_000, token: "dead", createdAtMs: 0 }));
    backdate(lockPath, 2_000);
    expect(withStoreLock(lockPath, () => "ran", FAST)).toBe("ran");
    expect(existsSync(lockPath)).toBe(false);
  });

  it("recovers an old empty/corrupt lock (owner never published)", () => {
    const lockPath = makeLockPath();
    writeFileSync(lockPath, "");
    backdate(lockPath, 2_000);
    expect(withStoreLock(lockPath, () => "ran", FAST)).toBe("ran");
  });

  it("recovers past the hard ceiling even if the pid is alive", () => {
    const lockPath = makeLockPath();
    writeFileSync(lockPath, JSON.stringify({ pid: process.pid, token: "zombie", createdAtMs: 0 }));
    backdate(lockPath, 6_000);
    expect(withStoreLock(lockPath, () => "ran", FAST)).toBe("ran");
  });

  it("never unlinks a lock that belongs to someone else on release", () => {
    const lockPath = makeLockPath();
    withStoreLock(lockPath, () => {
      // Simulate a recovery race: while we hold the lock, another writer replaced it.
      writeFileSync(lockPath, JSON.stringify({ pid: process.pid, token: "other", createdAtMs: 0 }));
    }, FAST);
    expect(existsSync(lockPath)).toBe(true);
    expect(JSON.parse(readFileSync(lockPath, "utf8")).token).toBe("other");
  });
});

describe("two-process regression at the acquire boundary", () => {
  it("waits for a concurrent writer that has created the lock but not yet published its pid", async () => {
    const lockPath = makeLockPath();
    const storePath = lockPath.replace(/\.lock$/, "");
    const HOLD_MS = 400;

    // Child reproduces the old-protocol window: open(wx) creates an EMPTY lock,
    // holds it, then finally writes an owner and releases.
    const childScript = `
      import { openSync, writeSync, closeSync, unlinkSync, existsSync } from "node:fs";
      const lockPath = process.env.LOCK_PATH;
      const fd = openSync(lockPath, "wx");
      process.stdout.write("LOCKED\\n");
      await new Promise((r) => setTimeout(r, ${HOLD_MS}));
      const stillHeld = existsSync(lockPath);
      writeSync(fd, JSON.stringify({ pid: process.pid, token: "child", createdAtMs: Date.now() }));
      closeSync(fd);
      unlinkSync(lockPath);
      process.stdout.write(stillHeld ? "STILL_HELD\\n" : "STOLEN\\n");
    `;
    const child = spawn(process.execPath, ["--input-type=module", "-e", childScript], {
      env: { ...process.env, LOCK_PATH: lockPath },
      stdio: ["ignore", "pipe", "pipe"],
    });
    let stdout = "";
    let stderr = "";
    child.stdout.on("data", (chunk: Buffer) => (stdout += chunk.toString()));
    child.stderr.on("data", (chunk: Buffer) => (stderr += chunk.toString()));
    await new Promise<void>((resolve) => {
      const check = () => (stdout.includes("LOCKED") ? resolve() : setTimeout(check, 5));
      check();
    });

    const store = new TurnAdvancementStore({
      storePath,
      lockOptions: { retryIntervalMs: 5, timeoutMs: 5_000, staleAfterMs: 30_000, hardStaleAfterMs: 60_000 },
    });
    const started = Date.now();
    const status = store.commit({
      advancementKey: "parent",
      sessionId: "s",
      messages: [{ role: "user", content: "parent" }],
    });
    const waitedMs = Date.now() - started;

    const exit = await new Promise<number | null>((resolve) => child.on("exit", resolve));

    expect(stderr).toBe("");
    expect(exit).toBe(0);
    expect(status).toBe("committed");
    // We waited for the child instead of deleting its live lock.
    expect(waitedMs).toBeGreaterThanOrEqual(HOLD_MS - 50);
    expect(stdout).toContain("STILL_HELD");
    expect(existsSync(lockPath)).toBe(false);
    expect(Object.keys(JSON.parse(readFileSync(storePath, "utf8")).records)).toEqual(["parent"]);
  });
});
