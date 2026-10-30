import * as fs from "node:fs/promises";
import * as os from "node:os";
import * as path from "node:path";

import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { DurableAdvancementKeyStore } from "../src/advancement-key-store.js";

// Where a regular file blocks a path, Windows reports ENOENT and POSIX reports
// ENOTDIR. Report it the Windows way on every platform, so the store's handling
// of it runs in every test environment, not only on Windows.
vi.mock("node:fs/promises", async (importOriginal) => {
  const actual = await importOriginal<typeof import("node:fs/promises")>();
  function asOnWindows<F extends (...args: never[]) => Promise<unknown>>(fn: F): F {
    return (async (...args: Parameters<F>) => {
      try {
        return await fn(...args);
      } catch (error) {
        if ((error as NodeJS.ErrnoException)?.code === "ENOTDIR") {
          (error as NodeJS.ErrnoException).code = "ENOENT";
        }
        throw error;
      }
    }) as F;
  }
  return {
    ...actual,
    lstat: asOnWindows(actual.lstat),
    mkdir: asOnWindows(actual.mkdir),
    open: asOnWindows(actual.open),
    readFile: asOnWindows(actual.readFile),
    stat: asOnWindows(actual.stat),
    writeFile: asOnWindows(actual.writeFile),
  };
});

describe("DurableAdvancementKeyStore, with a blocked path reported the Windows way (ENOENT)", () => {
  let dir: string;

  beforeEach(async () => {
    dir = await fs.mkdtemp(path.join(os.tmpdir(), "headroom-advancement-key-store-win-"));
  });

  afterEach(async () => {
    await fs.rm(dir, { recursive: true, force: true });
  });

  it.each([
    ["the store's directory", (storePath: string) => path.dirname(storePath)],
    ["the marker directory", (storePath: string) => `${storePath.replace(/\.json$/, "")}.d`],
  ])("reports a file in place of %s as a storage error, not as an absent key", async (_where, blocked) => {
    const storePath = path.join(dir, "nested", "commit-log.json");
    await fs.mkdir(path.dirname(blocked(storePath)), { recursive: true });
    await fs.writeFile(blocked(storePath), "not a directory", "utf8");
    const store = new DurableAdvancementKeyStore(storePath);

    await expect(store.tryCommit("turn-1")).rejects.toThrow(/HEADROOM_WORKSPACE_DIR/);
    await expect(store.has("turn-1")).rejects.toThrow(/HEADROOM_WORKSPACE_DIR/);

    // Clearing the path makes the key absent, and committable.
    await fs.rm(blocked(storePath));
    await expect(store.has("turn-1")).resolves.toBe(false);
    await expect(store.tryCommit("turn-1")).resolves.toBe("committed");
    await expect(store.has("turn-1")).resolves.toBe(true);
  });
});
