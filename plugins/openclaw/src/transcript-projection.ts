/**
 * Wait for OpenClaw SQLite transcript projection to settle after in-place rewrites.
 *
 * Headroom hygiene calls rewriteTranscriptEntries, which publishes a projection
 * rebuild. Without waiting, the next turn can hit "Session transcript projection
 * is rebuilding".
 */

import { createRequire } from "node:module";
import { readdirSync } from "node:fs";
import { dirname, join } from "node:path";

export type TranscriptProjectionScope = {
  agentId?: string;
  sessionId?: string;
  sessionKey?: string;
  storePath?: string;
  threadId?: string | number;
};

export type WaitForSessionTranscriptProjectionFn = (
  scope: TranscriptProjectionScope,
  abortSignal?: AbortSignal,
) => Promise<void>;

type RuntimeContextWithProjectionWait = Record<string, unknown> & {
  waitForSessionTranscriptProjection?: WaitForSessionTranscriptProjectionFn;
  sessionTarget?: TranscriptProjectionScope;
};

let cachedOpenClawWaitFn: WaitForSessionTranscriptProjectionFn | null | undefined;

/** Race a promise against a timeout and always release the timer. */
function withTimeout<T>(promise: Promise<T>, ms: number, label: string): Promise<T> {
  if (ms <= 0) return promise;
  let timerId: ReturnType<typeof setTimeout> | undefined;
  const timer = new Promise<never>((_, reject) => {
    timerId = setTimeout(
      () => reject(new Error(`${label} timed out after ${ms}ms`)),
      ms,
    );
  });
  return Promise.race([promise, timer]).finally(() => {
    if (timerId !== undefined) clearTimeout(timerId);
  });
}

/** Discover OpenClaw's stable re-export barrel for waitForSessionTranscriptProjection. */
async function loadOpenClawProjectionWaitFn(): Promise<WaitForSessionTranscriptProjectionFn | null> {
  if (cachedOpenClawWaitFn !== undefined) return cachedOpenClawWaitFn;

  try {
    const require = createRequire(import.meta.url);
    const pkgPath = require.resolve("openclaw/package.json");
    const distDir = join(dirname(pkgPath), "dist");
    const barrels = readdirSync(distDir).filter(
      (name) => name.startsWith("session-transcript-reconcile-") && name.endsWith(".mjs"),
    );

    for (const name of barrels) {
      try {
        const mod = (await import(/* @vite-ignore */ join(distDir, name))) as {
          waitForSessionTranscriptProjection?: WaitForSessionTranscriptProjectionFn;
        };
        if (typeof mod.waitForSessionTranscriptProjection === "function") {
          cachedOpenClawWaitFn = mod.waitForSessionTranscriptProjection;
          return cachedOpenClawWaitFn;
        }
      } catch {
        continue;
      }
    }
  } catch {
    // openclaw not installed in this process (tests, standalone tooling)
  }

  cachedOpenClawWaitFn = null;
  return null;
}

function resolveProjectionScope(
  params: {
    sessionId: string;
    sessionKey?: string;
    sessionTarget?: TranscriptProjectionScope;
  },
  runtimeContext?: RuntimeContextWithProjectionWait,
): TranscriptProjectionScope | undefined {
  const sessionTarget = params.sessionTarget ?? runtimeContext?.sessionTarget;
  const agentId = sessionTarget?.agentId;
  const sessionId = sessionTarget?.sessionId ?? params.sessionId;
  const sessionKey = sessionTarget?.sessionKey ?? params.sessionKey;
  const storePath = sessionTarget?.storePath;
  const threadId = sessionTarget?.threadId;

  if (!agentId && !sessionId && !sessionKey && !storePath && threadId === undefined) {
    return undefined;
  }

  return {
    ...agentId ? { agentId } : {},
    ...sessionId ? { sessionId } : {},
    ...sessionKey ? { sessionKey } : {},
    ...storePath ? { storePath } : {},
    ...threadId !== undefined ? { threadId } : {},
  };
}

/**
 * Block until OpenClaw finishes rebuilding the transcript projection for this session.
 * No-op when wait is disabled, scope is missing, or no wait hook is available.
 */
export async function awaitTranscriptProjectionSettle(options: {
  params: {
    sessionId: string;
    sessionKey?: string;
    sessionTarget?: TranscriptProjectionScope;
  };
  runtimeContext?: RuntimeContextWithProjectionWait;
  abortSignal?: AbortSignal;
  timeoutMs: number;
  logger?: { debug?: (msg: string) => void; warn?: (msg: string) => void };
}): Promise<{ waited: boolean; reason?: string }> {
  if (options.timeoutMs === 0) {
    return { waited: false, reason: "projection wait disabled" };
  }

  const scope = resolveProjectionScope(options.params, options.runtimeContext);
  if (!scope?.sessionId && !scope?.sessionKey) {
    return { waited: false, reason: "missing projection scope" };
  }

  const hostWait = options.runtimeContext?.waitForSessionTranscriptProjection;
  const waitFn = hostWait ?? (await loadOpenClawProjectionWaitFn());
  if (!waitFn) {
    return { waited: false, reason: "projection wait hook unavailable" };
  }

  options.abortSignal?.throwIfAborted();

  try {
    await withTimeout(
      waitFn(scope, options.abortSignal),
      options.timeoutMs,
      "Transcript projection settle",
    );
    options.logger?.debug?.(
      `[headroom] Transcript projection settled for session ${scope.sessionId ?? scope.sessionKey}`,
    );
    return { waited: true };
  } catch (error) {
    options.logger?.warn?.(`[headroom] Transcript projection settle failed: ${String(error)}`);
    return { waited: false, reason: String(error) };
  }
}

/** Test hook to reset cached OpenClaw discovery between vitest cases. */
export function resetTranscriptProjectionWaitCacheForTest(): void {
  cachedOpenClawWaitFn = undefined;
}
