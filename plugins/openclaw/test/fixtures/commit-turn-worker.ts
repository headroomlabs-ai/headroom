/**
 * Child-process worker for the cross-process DurableAdvancementKeyStore
 * regression tests (PR #3442 review, round 3).
 *
 * Run as a real, separate OS process (not a Worker thread, which shares the
 * parent's memory and wouldn't exercise inter-process file locking) via
 * `node --experimental-transform-types commit-turn-worker.ts <storePath>
 * <key> <goPath> <resultPath>`.
 *
 * Blocks until `goPath` exists (the parent test creates it only after every
 * worker has been spawned and had time to reach this poll loop), then
 * commits `key` through the real `DurableAdvancementKeyStore` and writes
 * the outcome to `resultPath` as JSON -- this is what lets two independently
 * launched processes race their read/check/write transactions against the
 * same commit-log file as closely to simultaneously as possible.
 */

import { access, writeFile } from "node:fs/promises";

// Explicit `.ts` extension: this file runs directly under Node's native
// TypeScript support (`--experimental-transform-types`), which -- unlike
// the TypeScript compiler / bundler used to build the plugin itself --
// resolves imports literally and does not map a `.js` specifier onto a
// sibling `.ts` file.
import { DurableAdvancementKeyStore } from "../../src/advancement-key-store.ts";

async function waitForGo(goPath: string): Promise<void> {
  for (;;) {
    try {
      await access(goPath);
      return;
    } catch {
      await new Promise((resolveDelay) => setTimeout(resolveDelay, 5));
    }
  }
}

async function main(): Promise<void> {
  const [, , storePath, key, goPath, resultPath] = process.argv;
  if (!storePath || !key || !goPath || !resultPath) {
    throw new Error("usage: commit-turn-worker.ts <storePath> <key> <goPath> <resultPath>");
  }

  await waitForGo(goPath);

  const store = new DurableAdvancementKeyStore(storePath);
  try {
    const status = await store.tryCommit(key, [key]);
    await writeFile(resultPath, JSON.stringify({ status }), "utf8");
  } catch (error) {
    await writeFile(
      resultPath,
      JSON.stringify({ error: error instanceof Error ? error.message : String(error) }),
      "utf8",
    );
  }
}

main();
