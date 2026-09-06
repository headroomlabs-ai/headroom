declare module "openclaw/plugin-sdk/core" {
  export function delegateCompactionToRuntime(params: {
    sessionId: string;
    sessionKey?: string;
    sessionFile: string;
    tokenBudget?: number;
    force?: boolean;
    runtimeContext?: unknown;
    [key: string]: unknown;
  }): Promise<{
    ok: boolean;
    compacted: boolean;
    reason?: string;
    result?: { tokensBefore: number; tokensAfter?: number };
  }>;
}
