import { describe, expect, it } from "vitest";
import { decideAssembleSkip, providerIdFromRuntimeSettings } from "../src/assemble-skip";

describe("decideAssembleSkip", () => {
  const routed = {
    skipAssembleWhenGatewayRouted: true,
    gatewayProviderIds: ["openrouter", "minimax-portal"],
  };

  it("never skips when the flag is off (stock behaviour)", () => {
    expect(
      decideAssembleSkip({ config: { gatewayProviderIds: ["openrouter"] }, providerId: "openrouter" })
        .skip,
    ).toBe(false);
  });

  it("never skips when no provider is gateway-routed", () => {
    expect(
      decideAssembleSkip({
        config: { skipAssembleWhenGatewayRouted: true, routeCodexViaProxy: false },
        providerId: "openai-codex",
      }).skip,
    ).toBe(false);
  });

  it("skips for a routed provider", () => {
    expect(decideAssembleSkip({ config: routed, providerId: "openrouter" }).skip).toBe(true);
  });

  it("still compresses for a provider that is NOT routed (no silent compression loss)", () => {
    const decision = decideAssembleSkip({ config: routed, providerId: "anthropic" });
    expect(decision.skip).toBe(false);
    expect(decision.reason).toContain("not gateway-routed");
  });

  it("skips when the provider is unknown (honours operator intent)", () => {
    expect(decideAssembleSkip({ config: routed, providerId: null }).skip).toBe(true);
    expect(decideAssembleSkip({ config: routed, providerId: "  " }).skip).toBe(true);
  });

  it("uses the default codex routing when gatewayProviderIds is unset", () => {
    const config = { skipAssembleWhenGatewayRouted: true };
    expect(decideAssembleSkip({ config, providerId: "openai-codex" }).skip).toBe(true);
    expect(decideAssembleSkip({ config, providerId: "anthropic" }).skip).toBe(false);
  });
});

describe("providerIdFromRuntimeSettings", () => {
  it("reads runtimeSettings.model.provider", () => {
    expect(providerIdFromRuntimeSettings({ model: { provider: " openrouter " } })).toBe("openrouter");
  });

  it("returns null for missing or malformed settings", () => {
    expect(providerIdFromRuntimeSettings(undefined)).toBeNull();
    expect(providerIdFromRuntimeSettings({})).toBeNull();
    expect(providerIdFromRuntimeSettings({ model: { provider: 3 } })).toBeNull();
    expect(providerIdFromRuntimeSettings({ model: { provider: "" } })).toBeNull();
  });
});
