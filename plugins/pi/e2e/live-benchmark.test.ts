import { createHash, randomUUID } from "node:crypto";
import { arch, platform } from "node:os";
import process from "node:process";

import type {
  ContextEvent,
  ExtensionContext,
  ToolResultEvent,
} from "@earendil-works/pi-coding-agent";
import type { ToolResultMessage } from "@earendil-works/pi-ai";
import { describe, expect, it } from "vitest";

import { HeadroomClient } from "../src/client.js";
import type { HeadroomConfig } from "../src/config.js";
import { HeadroomRuntime } from "../src/runtime.js";

const baseUrl = process.env.HEADROOM_LIVE_BASE_URL ?? "http://127.0.0.1:8787";
const fixtureSeed =
  process.env.HEADROOM_BENCHMARK_SEED ?? "headroom-pi-extension-v1";
const runNonce = process.env.HEADROOM_BENCHMARK_NONCE ?? randomUUID();
const resultCount = 8;
const transformRuns = 100;

const config: HeadroomConfig = {
  enabled: true,
  baseUrl,
  allowRemote: false,
  remoteHosts: [],
  minContextTokens: 20_000,
  minResultChars: 4_000,
  protectRecentToolResults: 2,
  protectedTools: ["read", "edit", "write", "ask", "todo", "headroom_retrieve"],
  maxCacheBytes: 67_108_864,
};

function originalFixture(seed: string): string {
  return JSON.stringify(
    Array.from({ length: 160 }, (_, index) => ({
      id: index,
      blob: (
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/=" +
        seed +
        String(index).padStart(4, "0")
      ).repeat(10),
      checksum: createHash("sha256")
        .update(`${seed}-${index}`)
        .digest("hex"),
    })),
  );
}

function context(): ExtensionContext {
  return {
    mode: "tui",
    hasUI: false,
    model: {
      id: "gpt-5.6-sol",
      provider: "openai-codex",
      contextWindow: 128_000,
    },
    ui: {
      notify() {},
      setStatus() {},
    },
    getContextUsage: () => undefined,
  } as unknown as ExtensionContext;
}

function textContent(message: ToolResultMessage): string | undefined {
  const block = message.content[0];
  return block?.type === "text" ? block.text : undefined;
}

function percentile(samples: readonly number[], fraction: number): number {
  const sorted = [...samples].sort((left, right) => left - right);
  return sorted[Math.floor(sorted.length * fraction)] ?? 0;
}

describe("live extension benchmark", () => {
  it("emits seeded preparation and latest-transform metrics as JSON", async () => {
    const client = new HeadroomClient({ baseUrl, timeoutMs: 30_000 });
    const runtime = new HeadroomRuntime({ config, client });
    const ctx = context();

    try {
      expect(await runtime.health()).toBe(true);

      const rawResults = Array.from({ length: resultCount }, (_, index) =>
        originalFixture(`${fixtureSeed}-${runNonce}-${index}`),
      );
      const preparationStarted = performance.now();
      rawResults.forEach((text, index) => {
        runtime.observeToolResult(
          {
            type: "tool_result",
            toolCallId: `benchmark-${index}`,
            toolName: "bash",
            input: { command: `fixture-${index}` },
            content: [{ type: "text", text }],
            details: { fixtureSeed, index },
            isError: false,
          } as ToolResultEvent,
          ctx,
        );
      });

      await expect
        .poll(
          () => {
            const snapshot = runtime.snapshot();
            return {
              completed:
                snapshot.worker.accepted +
                snapshot.worker.rejected +
                snapshot.worker.dropped,
              queued: snapshot.worker.queued,
              active: snapshot.worker.active,
            };
          },
          { timeout: 70_000, interval: 25 },
        )
        .toEqual({ completed: resultCount, queued: 0, active: 0 });
      const preparationWallMs = performance.now() - preparationStarted;

      const messages = rawResults.map(
        (text, index): ToolResultMessage => ({
          role: "toolResult",
          toolCallId: `benchmark-${index}`,
          toolName: "bash",
          content: [{ type: "text", text }],
          details: { fixtureSeed, index },
          isError: false,
          timestamp: index + 1,
        }),
      );
      const samples: number[] = [];
      let transformed: ContextEvent["messages"] | undefined;
      for (let index = 0; index < transformRuns; index += 1) {
        const started = performance.now();
        transformed = runtime.transform(
          messages as ContextEvent["messages"],
          ctx,
        );
        samples.push(performance.now() - started);
      }

      const snapshot = runtime.snapshot();
      const substitutedResults = transformed
        ? transformed.filter(
            (message, index) =>
              message.role === "toolResult" &&
              textContent(message as ToolResultMessage) !== rawResults[index],
          ).length
        : 0;
      const rawTranscriptUnchanged = messages.every(
        (message, index) => textContent(message) === rawResults[index],
      );
      const originalBytes = rawResults.reduce(
        (total, text) => total + Buffer.byteLength(text),
        0,
      );
      const p95Ms = percentile(samples, 0.95);
      const report = {
        schemaVersion: 1,
        environment: {
          platform: platform(),
          architecture: arch(),
          node: process.version,
          baseUrl,
          modelLabel: "openai-codex/gpt-5.6-sol",
          modelInferenceRequests: 0,
        },
        fixture: {
          seed: fixtureSeed,
          runNonce,
          results: resultCount,
          recordsPerResult: 160,
          originalBytes,
          averageResultBytes: Math.round(originalBytes / resultCount),
        },
        policy: {
          minContextTokens: config.minContextTokens,
          minResultChars: config.minResultChars,
          protectRecentToolResults: config.protectRecentToolResults,
          maxCacheBytes: config.maxCacheBytes,
        },
        preparation: {
          wallMs: Number(preparationWallMs.toFixed(3)),
          candidates: snapshot.worker.candidates,
          accepted: snapshot.worker.accepted,
          rejected: snapshot.worker.rejected,
          dropped: snapshot.worker.dropped,
          state: snapshot.worker.state,
          lastError: snapshot.worker.lastError,
          tokensBefore: snapshot.tokensBefore,
          tokensAfter: snapshot.tokensAfter,
          tokensSaved: snapshot.tokensSaved,
          bytesSaved: snapshot.bytesSaved,
        },
        latestTransform: {
          ...snapshot.lastTransform,
          substitutedResults,
          protectedRecentRaw: resultCount - substitutedResults,
          rawTranscriptUnchanged,
        },
        hotPath: {
          runs: transformRuns,
          medianMs: Number(percentile(samples, 0.5).toFixed(3)),
          p95Ms: Number(p95Ms.toFixed(3)),
          maxMs: Number(Math.max(...samples).toFixed(3)),
        },
      };

      console.log(`HEADROOM_BENCHMARK_JSON=${JSON.stringify(report)}`);

      expect(snapshot.worker.accepted).toBe(resultCount);
      expect(snapshot.worker.rejected).toBe(0);
      expect(snapshot.worker.dropped).toBe(0);
      expect(snapshot.lastTransform.substitutions).toBe(
        resultCount - config.protectRecentToolResults,
      );
      expect(substitutedResults).toBe(
        resultCount - config.protectRecentToolResults,
      );
      expect(rawTranscriptUnchanged).toBe(true);
      expect(p95Ms).toBeLessThan(10);

    } finally {
      runtime.stop();
    }
  });
});
