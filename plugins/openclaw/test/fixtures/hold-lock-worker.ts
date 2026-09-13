/**
 * Simulates a live-but-slow lock holder for the DurableAdvancementKeyStore
 * cross-process regression tests (PR #3442 review, round 4): acquires the
 * exact same lock file the real store uses, signals readiness, then holds
 * it until told to release -- proving a real, still-alive holder is never
 * force-preempted no matter how long its critical section runs, unlike the
 * round-3 age-based staleness heuristic this replaces.
 *
 * Deliberately does not import `DurableAdvancementKeyStore`: it only needs
 * to occupy the lock file at the path the real class would use, using the
 * same `open(path, "wx")` protocol, so the *other* worker's real
 * `tryCommit` call is what gets exercised and observed.
 *
 * Usage: node --experimental-transform-types hold-lock-worker.ts
 *   <storePath> <readyPath> <releasePath>
 */

import { access, mkdir, open, unlink, utimes, writeFile } from "node:fs/promises";
import { dirname } from "node:path";

async function waitFor(p: string): Promise<void> {
  for (;;) {
    try {
      await access(p);
      return;
    } catch {
      await new Promise((resolveDelay) => setTimeout(resolveDelay, 5));
    }
  }
}

async function main(): Promise<void> {
  const [, , storePath, readyPath, releasePath] = process.argv;
  if (!storePath || !readyPath || !releasePath) {
    throw new Error("usage: hold-lock-worker.ts <storePath> <readyPath> <releasePath>");
  }

  const lockPath = `${storePath}.lock`;
  await mkdir(dirname(storePath), { recursive: true });
  const handle = await open(lockPath, "wx");
  await handle.close();
  // Exercise the former 30-second reclaim threshold without a slow test.
  const oldTime = new Date(Date.now() - 60_000);
  await utimes(lockPath, oldTime, oldTime);

  await writeFile(readyPath, "ready", "utf8");

  await waitFor(releasePath);
  await unlink(lockPath).catch(() => undefined);
}

main();
