export interface OpenClawCompactParams {
  sessionId: string;
  sessionKey?: string;
  agentId?: string;
  sessionTarget?: unknown;
  sessionFile?: string;
  tokenBudget?: number;
  force?: boolean;
  currentTokenCount?: number;
  compactionTarget?: "budget" | "threshold";
  customInstructions?: string;
  runtimeSettings?: unknown;
  runtimeContext?: unknown;
  abortSignal?: AbortSignal;
}

export interface OpenClawCompactResult {
  ok: boolean;
  compacted: boolean;
  reason?: string;
  result?: {
    summary?: string;
    firstKeptEntryId?: string;
    tokensBefore: number;
    tokensAfter?: number;
    details?: unknown;
    sessionId?: string;
    sessionTarget?: unknown;
    sessionFile?: string;
  };
}

const OPENCLAW_PLUGIN_SDK_CORE = "openclaw/plugin-sdk/core";

/** Delegate compaction without bundling the optional OpenClaw peer dependency. */
export async function delegateCompactionToRuntime(
  params: OpenClawCompactParams,
): Promise<OpenClawCompactResult> {
  const pluginSdk = (await import(OPENCLAW_PLUGIN_SDK_CORE)) as {
    delegateCompactionToRuntime?: (
      compactParams: OpenClawCompactParams,
    ) => Promise<OpenClawCompactResult>;
  };

  if (!pluginSdk.delegateCompactionToRuntime) {
    throw new Error("OpenClaw runtime does not expose delegateCompactionToRuntime");
  }

  return pluginSdk.delegateCompactionToRuntime(params);
}
