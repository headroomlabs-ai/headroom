export { default, registerHeadroomPlugin } from "./plugin/index.js";
export { HeadroomContextEngine } from "./engine.js";
export {
  ownsPersistentCompaction,
  resolvePersistentCompactionMode,
  type PersistentCompactionMode,
} from "./compaction-mode.js";
export { ProxyManager, normalizeAndValidateProxyUrl, isLocalProxyUrl, defaultLogger, probeHeadroomProxy } from "./proxy-manager.js";
export { agentToOpenAI, normalizeAgentMessages, openAIToAgent } from "./convert.js";
export { createHeadroomRetrieveTool } from "./tools/headroom-retrieve.js";
export {
  DEFAULT_GATEWAY_PROVIDER_IDS,
  applyGatewayProviderBaseUrls,
  applyGatewayProviderBaseUrlsInPlace,
  resolveGatewayProviderIds,
  type GatewayRoutingOverrides,
} from "./gateway-config.js";
export {
  readProviderSessionHeaders,
  ensureSessionId,
  type SessionHeaderMap,
} from "./session-headers.js";
