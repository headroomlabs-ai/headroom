/**
 * Per-session debounce for Headroom transcript hygiene rewrites.
 *
 * Prevents turn-end maintain() and hybrid compact() pre-passes from stacking
 * SQLite rewrites within a short window.
 */

export interface HygieneDebounceState {
  /** Timestamp of the last hygiene pass that changed the transcript. */
  lastChangedAtMs: number;
}

export class HygieneDebounceTracker {
  private readonly sessions = new Map<string, HygieneDebounceState>();

  shouldSkip(sessionId: string, debounceMs: number, nowMs = Date.now()): boolean {
    if (debounceMs <= 0) return false;
    const state = this.sessions.get(sessionId);
    if (!state) return false;
    return nowMs - state.lastChangedAtMs < debounceMs;
  }

  recordChanged(sessionId: string, nowMs = Date.now()): void {
    this.sessions.set(sessionId, { lastChangedAtMs: nowMs });
  }

  clear(sessionId?: string): void {
    if (sessionId) {
      this.sessions.delete(sessionId);
      return;
    }
    this.sessions.clear();
  }
}
