/**
 * Headroom OpenClaw Plugin — register ContextEngine + CCR retrieval tool.
 *
 * Usage:
 *   openclaw plugins install headroom-ai/openclaw
 *
 * Configuration (in ~/.openclaw/config.json or ~/.clawdbot/clawdbot.json):
 *   {
 *     "plugins": {
 *       "slots": { "contextEngine": "headroom" },
 *       "entries": { "headroom": { "enabled": true } }
 *     }
 *   }
 */

/* eslint-disable @typescript-eslint/no-explicit-any */

import { HeadroomContextEngine } from "../engine.js";
import {
  applyGatewayProviderBaseUrlsInPlace,
  resolveGatewayProviderIds,
} from "../gateway-config.js";
import { normalizeAndValidateProxyUrl } from "../proxy-manager.js";
import { createHeadroomRetrieveTool } from "../tools/headroom-retrieve.js";

/**
 * OpenClaw 2026.x plugin API requires a `{ register(api) }` object export.
 * The previous bare-function default export was silently skipped by the loader.
 * See: https://github.com/chopratejas/headroom/issues/XXX
 */
export default {
  register: headroomPlugin,
};

function headroomPlugin(api: any) {
  const config = api.config?.plugins?.entries?.headroom?.config ?? {};
  const logger = api.logger ?? console;
  const rawProxyUrl = config.proxyUrl;
  const proxyUrl =
    typeof rawProxyUrl === "string" && rawProxyUrl.trim().length > 0
      ? normalizeAndValidateProxyUrl(rawProxyUrl)
      : undefined;

  const engine = new HeadroomContextEngine({ ...config, proxyUrl }, {
    info: (m: string) => logger.info(m),
    warn: (m: string) => logger.warn(m),
    error: (m: string) => logger.error(m),
    debug: (m: string) => logger.debug?.(m),
  });
  const gatewayProviderIds = resolveGatewayProviderIds(config);

  const applyGatewayRouting = async (activeProxyUrl: string) => {
    if (gatewayProviderIds.length === 0) {
      return;
    }

    try {
      const changed = applyGatewayProviderBaseUrlsInPlace(api.config, activeProxyUrl, gatewayProviderIds);

      if (changed) {
        logger.info(
          `[headroom] Routed ${gatewayProviderIds.join(", ")} through Headroom proxy in memory at ${activeProxyUrl}`,
        );
      } else {
        logger.info(
          `[headroom] Upstream gateway already routed in memory for ${gatewayProviderIds.join(", ")} at ${activeProxyUrl}`,
        );
      }
    } catch (error) {
      logger.warn(`[headroom] Failed to configure upstream gateway routing: ${error}`);
    }
  };

  const ensureGatewayRouting = async () => {
    const activeProxyUrl = engine.getProxyUrl();
    if (!activeProxyUrl) {
      logger.debug?.("[headroom] Deferring upstream gateway routing until proxy is available");
      engine.ensureProxyStarted();
      return;
    }
    await applyGatewayRouting(activeProxyUrl);
  };

  engine.onProxyReady(async (activeProxyUrl) => {
    await applyGatewayRouting(activeProxyUrl);
  });

  // Register as context engine
  api.registerContextEngine("headroom", () => engine);

  // Register CCR retrieval tool (active once proxy is running)
  api.registerTool((ctx: any) => {
    const activeProxyUrl = engine.getProxyUrl() ?? proxyUrl;
    if (!activeProxyUrl) return null;
    return createHeadroomRetrieveTool({ proxyUrl: activeProxyUrl });
  });

  api.on("gateway_start", async () => {
    await ensureGatewayRouting();
  });

  void ensureGatewayRouting();

  logger.info("[headroom] Plugin registered");
}
