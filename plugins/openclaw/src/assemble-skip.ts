/**
 * Decide whether per-turn `assemble()` compression should be skipped because the
 * live provider request is already routed through the Headroom proxy.
 *
 * Compressing in `assemble()` *and* on the proxy's live path compresses the same
 * transcript twice (see docs/tool-call-preservation.md, BUG-6). The skip is
 * provider-aware: a provider that is not gateway-routed still gets `assemble()`
 * compression, otherwise enabling the flag would silently disable compression
 * for direct providers.
 */

import { resolveGatewayProviderIds } from "./gateway-config.js";

export interface AssembleSkipDecision {
  skip: boolean;
  routedProviderIds: string[];
  reason: string;
}

export interface AssembleSkipInput {
  /** Plugin config; only `skipAssembleWhenGatewayRouted`, `gatewayProviderIds`, `routeCodexViaProxy` are read. */
  config: Record<string, unknown> | undefined;
  /** Provider id of the model being called, from `runtimeSettings.model.provider` (may be unknown). */
  providerId: string | null | undefined;
}

export function decideAssembleSkip(input: AssembleSkipInput): AssembleSkipDecision {
  const routedProviderIds = resolveGatewayProviderIds(input.config);

  if (input.config?.skipAssembleWhenGatewayRouted !== true) {
    return { skip: false, routedProviderIds, reason: "flag disabled" };
  }
  if (routedProviderIds.length === 0) {
    return { skip: false, routedProviderIds, reason: "no gateway-routed providers" };
  }

  const providerId = typeof input.providerId === "string" ? input.providerId.trim() : "";
  if (!providerId) {
    return {
      skip: true,
      routedProviderIds,
      reason: `provider unknown; gateway-routed providers: ${routedProviderIds.join(", ")}`,
    };
  }
  if (routedProviderIds.includes(providerId)) {
    return { skip: true, routedProviderIds, reason: `provider ${providerId} is gateway-routed` };
  }
  return {
    skip: false,
    routedProviderIds,
    reason: `provider ${providerId} is not gateway-routed`,
  };
}

/** Extract the provider id OpenClaw passes in `assemble()` runtime settings. */
export function providerIdFromRuntimeSettings(runtimeSettings: unknown): string | null {
  if (typeof runtimeSettings !== "object" || runtimeSettings === null) return null;
  const model = (runtimeSettings as { model?: unknown }).model;
  if (typeof model !== "object" || model === null) return null;
  const provider = (model as { provider?: unknown }).provider;
  return typeof provider === "string" && provider.trim().length > 0 ? provider.trim() : null;
}
