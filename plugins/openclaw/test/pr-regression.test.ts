/**
 * PR regression suite — one describe block per PR pillar.
 *
 * Each test encodes a behavior that stock upstream `main` lacks or gets wrong.
 * See docs/TEST_MATRIX.md for the full mapping.
 */
import { mkdtempSync, rmSync } from "node:fs";
import { join } from "node:path";
import { tmpdir } from "node:os";
import { afterEach, describe, expect, it } from "vitest";
import { HeadroomContextEngine } from "../src/engine.js";
import { agentToOpenAI, openAIToAgent } from "../src/convert.js";
import {
  applyGatewayProviderBaseUrls,
  resolveGatewayProviderIds,
} from "../src/gateway-config.js";
import { resolveProxyPathPrefix } from "../src/proxy-routing.js";
import {
  ownsPersistentCompaction,
  resolvePersistentCompactionMode,
} from "../src/compaction-mode.js";
import { resolveAssembleCompressConfig, resolveDurableCompressConfig } from "../src/compress-request-config.js";
import { TurnAdvancementStore } from "../src/turn-advancement-store.js";
import { buildToolScenarioTranscript, NATIVE_TOOL_SCENARIOS } from "./fixtures/openclaw-native-tools.js";

const tempDirs: string[] = [];

afterEach(() => {
  for (const dir of tempDirs.splice(0)) {
    rmSync(dir, { recursive: true, force: true });
  }
});

function makeTurnStorePath(): string {
  const dir = mkdtempSync(join(tmpdir(), "pr-regression-turn-"));
  tempDirs.push(dir);
  return join(dir, "turn-advancements.json");
}

describe("PR pillar 1 — durable turn contract", () => {
  it("declares transcriptSemantics so OpenClaw 2026.9.x does not degrade to legacy", () => {
    const engine = new HeadroomContextEngine();
    expect(engine.info.transcriptSemantics?.currentTurnFence).toBe("before-current-turn-entry-v1");
    expect(engine.info.transcriptSemantics?.turnAdvancementIdempotency).toBe("atomic-idempotent-v1");
  });

  it("commitTurn returns status contract (not legacy committed boolean)", async () => {
    const engine = new HeadroomContextEngine({
      turnAdvancementStorePath: makeTurnStorePath(),
    });
    const result = await engine.commitTurn({
      sessionId: "s1",
      advancementKey: "key-1",
      messages: [{ role: "user", content: "hi" }],
    });
    expect(result.status).toMatch(/^(committed|duplicate)$/);
  });

  it("turn advancement store deduplicates same advancementKey", () => {
    const store = new TurnAdvancementStore({ storePath: makeTurnStorePath() });
    const messages = [{ role: "user", content: "x" }];
    expect(store.commit({ advancementKey: "k1", sessionId: "s1", messages })).toBe(
      "committed",
    );
    expect(store.commit({ advancementKey: "k1", sessionId: "s1", messages })).toBe(
      "duplicate",
    );
  });
});

describe("PR pillar 2 — multi-upstream gateway routing", () => {
  it("resolveGatewayProviderIds defaults to openai-codex when not disabled", () => {
    expect(resolveGatewayProviderIds(undefined)).toEqual(["openai-codex"]);
  });

  it("providerUpstreams injects x-headroom-base-url on rewrite", () => {
    const result = applyGatewayProviderBaseUrls(
      {
        models: {
          providers: {
            openrouter: { baseUrl: "https://openrouter.ai/api/v1", headers: {} },
          },
        },
      },
      "http://127.0.0.1:8787",
      ["openrouter"],
      { providerUpstreams: { openrouter: "https://openrouter.ai/api" } },
    );
    expect(result.changed).toBe(true);
    const cfg = result.config as {
      models: { providers: Record<string, { headers: Record<string, string> }> };
    };
    expect(cfg.models.providers.openrouter.headers["x-headroom-base-url"]).toBe(
      "https://openrouter.ai/api",
    );
  });

  it("Gemini providers use /v1beta proxy prefix (not blind /v1)", () => {
    expect(resolveProxyPathPrefix({ providerId: "google" })).toBe("/v1beta");
    expect(resolveProxyPathPrefix({ providerId: "openrouter" })).toBe("/v1");
  });
});

describe("PR pillar 3 — durable compaction defaults", () => {
  it("default persistentCompaction matches upstream delegation (openclaw)", () => {
    expect(resolvePersistentCompactionMode({})).toBe("openclaw");
    expect(ownsPersistentCompaction("openclaw")).toBe(false);
  });

  it("headroom mode owns compaction for zero-LLM durable rewrite", () => {
    expect(ownsPersistentCompaction("headroom")).toBe(true);
  });

  it("durable compress config uses protect_recent 2 (not 0)", () => {
    expect(resolveDurableCompressConfig().protect_recent).toBe(2);
  });
});

describe("PR pillar 4 — assemble behavior", () => {
  it("default assemble compress config protects recent tool turns", () => {
    expect(resolveAssembleCompressConfig().protect_recent).toBe(2);
  });

  it("engine exposes skipAssembleWhenGatewayRouted config surface", () => {
    const engine = new HeadroomContextEngine({
      skipAssembleWhenGatewayRouted: true,
      gatewayProviderIds: ["openrouter"],
    });
    expect(engine).toBeDefined();
  });
});

describe("PR pillar 5 — tool-call preservation vs stock convert", () => {
  const imageScenario = NATIVE_TOOL_SCENARIOS.find(
    (s) => s.toolName === "view_image" && s.payloadKind === "image",
  )!;

  it("view_image emits tool.name and a placeholder — image bytes never reach the proxy", () => {
    const transcript = buildToolScenarioTranscript(imageScenario, 0);
    const openai = agentToOpenAI(transcript);
    const tool = openai.find((m) => m.role === "tool" && m.name === "view_image");
    expect(tool?.content).toContain("[headroom-omitted image image/png");
    expect(JSON.stringify(openai)).not.toContain(Buffer.from("fake-png-bytes-0").toString("base64"));
  });

  it("lossy proxy text with _headroomMeta stripped still restores image bytes from originals", () => {
    const transcript = buildToolScenarioTranscript(imageScenario, 0);
    const openai = agentToOpenAI(transcript);
    const crushed = openai.map(({ _headroomMeta: _dropped, ...m }) =>
      m.role === "tool" ? { ...m, content: "[lossy]" } : m,
    );
    const restored = openAIToAgent(crushed, { originals: transcript });
    const view = restored.find((m) => m.role === "toolResult" && m.toolName === "view_image");
    const image = Array.isArray(view?.content)
      ? view.content.find((b: { type?: string }) => b.type === "image")
      : undefined;
    expect(image?.data).toBe(Buffer.from("fake-png-bytes-0").toString("base64"));
  });

  it("stock-style restore without originals degrades to text only (documents why originals matter)", () => {
    const transcript = buildToolScenarioTranscript(imageScenario, 0);
    const crushed = agentToOpenAI(transcript).map(({ _headroomMeta: _dropped, ...m }) =>
      m.role === "tool" ? { ...m, content: "[lossy]" } : m,
    );
    const view = openAIToAgent(crushed).find((m) => m.toolName === "view_image");
    expect(view?.content).toEqual([{ type: "text", text: "[lossy]" }]);
  });

  it("infers isError from JSON error envelope when meta stripped", () => {
    const restored = openAIToAgent([
      {
        role: "tool",
        name: "view_image",
        tool_call_id: "call_e",
        content: JSON.stringify({ status: "error", tool: "view_image", error: "denied" }),
      },
    ]);
    expect(restored[0].isError).toBe(true);
  });
});

describe("PR pillar 6 — value vs stock main (smoke)", () => {
  it("exports HeadroomContextEngine as the context engine entry point", async () => {
    const mod = await import("../src/index.js");
    expect(mod.HeadroomContextEngine).toBeDefined();
  });

  it("native tool fixture catalog covers exec browser read view_image and sessions tools", () => {
    const names = new Set(NATIVE_TOOL_SCENARIOS.map((s) => s.toolName));
    for (const required of ["exec", "read", "browser", "view_image", "memory_search"]) {
      expect(names.has(required)).toBe(true);
    }
  });
});
