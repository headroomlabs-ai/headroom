import type {
	ContextEvent,
	ExtensionContext,
	ToolResultEvent,
} from "@earendil-works/pi-coding-agent";
import type { ToolResultMessage } from "@earendil-works/pi-ai";
import { describe, expect, it, vi } from "vitest";

import { mkdtemp, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { PreparedCache } from "../src/cache.js";
import { DEFAULT_CONFIG, type HeadroomConfig } from "../src/config.js";
import { candidateFromToolResult } from "../src/policy.js";
import { HeadroomRuntime } from "../src/runtime.js";
import {
	formatStats,
	formatStatus,
	type RuntimeSnapshot,
} from "../src/status.js";
import type { PreparedEntry } from "../src/types.js";
import type { WorkerClient } from "../src/worker.js";

function config(overrides: Partial<HeadroomConfig> = {}): HeadroomConfig {
	return {
		...DEFAULT_CONFIG,
		protectedTools: [...DEFAULT_CONFIG.protectedTools],
		minResultChars: 10,
		protectRecentToolResults: 1,
		...overrides,
	};
}

function context(
	contextWindow: number | null = 100_000,
	sessionFile?: string,
): ExtensionContext {
	return {
		mode: "tui",
		hasUI: true,
		model:
			contextWindow === null
				? undefined
				: ({
						id: "model",
						provider: "provider",
						contextWindow,
					} as ExtensionContext["model"]),
		ui: {
			notify: vi.fn(),
			setStatus: vi.fn(),
		},
		sessionManager: {
			getSessionFile: () => sessionFile,
		},
		getContextUsage: () => undefined,
	} as unknown as ExtensionContext;
}

function event(id = "1", text = "x".repeat(20)): ToolResultEvent {
	return {
		type: "tool_result",
		toolCallId: id,
		toolName: "bash",
		input: { command: "echo" },
		content: [{ type: "text", text }],
		details: { command: "echo" },
		isError: false,
	} as ToolResultEvent;
}

function message(id: string, text: string): ToolResultMessage {
	return {
		role: "toolResult",
		toolCallId: id,
		toolName: "bash",
		content: [{ type: "text", text }],
		details: { stable: true },
		isError: false,
		timestamp: 1,
	};
}

function entry(source: ToolResultMessage): PreparedEntry {
	const candidate = candidateFromToolResult(source, config());
	if (!candidate) throw new Error("test fixture must be eligible");
	return {
		...candidate,
		compressedText: "summary Retrieve more: hash=abcdef1234567890abcdef12",
		ccrHashes: ["abcdef1234567890abcdef12"],
		retrievals: new Map([["abcdef1234567890abcdef12", candidate.originalText]]),
		tokensBefore: 1_000,
		tokensAfter: 500,
		tokensSaved: 500,
		originalBytes: Buffer.byteLength(candidate.originalText),
		compressedBytes: Buffer.byteLength(
			"summary Retrieve more: hash=abcdef1234567890abcdef12",
		),
		sizeBytes: 100,
		originalSha256: "sha",
		policyVersion: "v1",
		createdAt: 1,
		lastAccessedAt: 1,
	};
}

function client(overrides: Partial<WorkerClient> = {}): WorkerClient {
	return {
		health: async () => true,
		compress: async () => ({ messages: [] }),
		retrieve: async () => undefined,
		...overrides,
	};
}

async function flushMicrotasks(): Promise<void> {
	await Promise.resolve();
	await Promise.resolve();
	await Promise.resolve();
}

describe("HeadroomRuntime", () => {
	it("stays quiet on startup failure until compression becomes relevant", async () => {
		const runtime = new HeadroomRuntime({
			config: config(),
			client: client({ health: async () => false }),
		});
		const ctx = context();

		runtime.start(ctx);
		await vi.waitFor(() =>
			expect(runtime.snapshot().worker.state).toBe("offline"),
		);
		expect(ctx.ui.setStatus).not.toHaveBeenCalled();

		runtime.observeToolResult(event(), ctx);
		expect(ctx.ui.setStatus).toHaveBeenCalledWith(
			"headroom",
			"Headroom offline",
		);
		runtime.stop();
	});

	it("bypasses known context windows below the configured threshold", async () => {
		const compress = vi.fn<WorkerClient["compress"]>(async () => ({
			messages: [],
		}));
		const runtime = new HeadroomRuntime({
			config: config({ minContextTokens: 20_000 }),
			client: client({ compress }),
		});
		const ctx = context(19_999);
		runtime.start(ctx);

		runtime.observeToolResult(event(), ctx);
		await flushMicrotasks();

		expect(compress).not.toHaveBeenCalled();
		runtime.stop();
	});

	it("keeps unknown models eligible and labels them unknown", async () => {
		const compress = vi.fn<WorkerClient["compress"]>(async () => ({
			messages: [],
		}));
		const runtime = new HeadroomRuntime({
			config: config(),
			client: client({ compress }),
		});
		const ctx = context(null);
		runtime.start(ctx);

		runtime.observeToolResult(event(), ctx);
		await vi.waitFor(() => expect(compress).toHaveBeenCalledTimes(1));

		expect(compress.mock.calls[0]?.[1]).toBe("unknown");
		runtime.stop();
	});

	it("queues tool observations without returning a replacement", () => {
		const runtime = new HeadroomRuntime({ config: config(), client: client() });
		const ctx = context();

		expect(runtime.observeToolResult(event(), ctx)).toBeUndefined();
		runtime.stop();
	});

	it("queues resume candidates without synchronous UI work", () => {
		const runtime = new HeadroomRuntime({ config: config(), client: client() });
		const ctx = context();
		const old = message("1", "x".repeat(20));
		const recent = message("2", "recent".repeat(4));

		expect(
			runtime.transform([old, recent] as ContextEvent["messages"], ctx),
		).toBeUndefined();
		expect(ctx.ui.setStatus).not.toHaveBeenCalled();
		runtime.stop();
	});

	it("substitutes only a prepared cold result", () => {
		const cache = new PreparedCache(1_000_000);
		const old = message("1", "x".repeat(20));
		const recent = message("2", "recent".repeat(4));
		const prepared = entry(old);
		cache.set(prepared);
		const runtime = new HeadroomRuntime({
			config: config(),
			client: client(),
			cache,
		});
		const ctx = context();

		const result = runtime.transform(
			[old, recent] as ContextEvent["messages"],
			ctx,
		);

		expect(result).toBeDefined();
		expect((result?.[0] as ToolResultMessage).content[0]).toEqual({
			type: "text",
			text: prepared.compressedText,
		});
		expect(result?.[1]).toBe(recent);
		runtime.stop();
	});

	it("reports only substitutions applied by the latest context transform", () => {
		const cache = new PreparedCache(1_000_000);
		const old = message("1", "x".repeat(20));
		const recent = message("2", "recent".repeat(4));
		const prepared = {
			...entry(old),
			compressedText: "short",
			compressedBytes: 5,
		};
		cache.set(prepared);
		const runtime = new HeadroomRuntime({
			config: config(),
			client: client(),
			cache,
		});
		const ctx = context();

		runtime.transform([old, recent] as ContextEvent["messages"], ctx);

		expect(runtime.snapshot().lastTransform).toEqual({
			substitutions: 1,
			tokensBefore: 1_000,
			tokensAfter: 500,
			tokensSaved: 500,
			bytesBefore: 20,
			bytesAfter: 5,
			bytesSaved: 15,
		});

		runtime.transform([recent] as ContextEvent["messages"], ctx);
		expect(runtime.snapshot().lastTransform).toEqual({
			substitutions: 0,
			tokensBefore: 0,
			tokensAfter: 0,
			tokensSaved: 0,
			bytesBefore: 0,
			bytesAfter: 0,
			bytesSaved: 0,
		});
		runtime.stop();
	});

	it("restores session savings after reload or resume", async () => {
		const directory = await mkdtemp(join(tmpdir(), "headroom-pi-runtime-"));
		const sessionFile = join(directory, "chat.jsonl");
		await writeFile(sessionFile, "{}\n");
		const first = new HeadroomRuntime({
			config: config(),
			client: client(),
		});
		first.start(context(100_000, sessionFile));
		first.restoreSessionSavings({
			tokensSaved: 8_593,
			tokensBefore: 40_000,
			tokensAfter: 31_407,
			bytesSaved: 12_000,
			retrievals: 2,
		});
		first.stop();

		const resumed = new HeadroomRuntime({
			config: config(),
			client: client(),
		});
		const ctx = context(100_000, sessionFile);
		resumed.start(ctx);
		await vi.waitFor(() => expect(resumed.snapshot().tokensSaved).toBe(8_593));
		expect(resumed.snapshot().retrievals).toBe(2);
		expect(ctx.ui.setStatus).toHaveBeenCalledWith(
			"headroom",
			"Headroom saved 8,593 tokens this session",
		);
		resumed.stop();
	});

	it("turns off substitution without clearing prepared state", () => {
		const cache = new PreparedCache(1_000_000);
		const old = message("1", "x".repeat(20));
		const recent = message("2", "recent".repeat(4));
		cache.set(entry(old));
		const runtime = new HeadroomRuntime({
			config: config(),
			client: client(),
			cache,
		});

		runtime.setSessionEnabled(false);
		expect(
			runtime.transform([old, recent] as ContextEvent["messages"], context()),
		).toBeUndefined();
		expect(cache.get(entry(old).key)).toBeDefined();
		runtime.stop();
	});

	it("lets the session command enable a disabled default", () => {
		const cache = new PreparedCache(1_000_000);
		const old = message("1", "x".repeat(20));
		const recent = message("2", "recent".repeat(4));
		cache.set(entry(old));
		const runtime = new HeadroomRuntime({
			config: config({ enabled: false }),
			client: client(),
			cache,
		});

		runtime.setSessionEnabled(true);
		expect(
			runtime.transform([old, recent] as ContextEvent["messages"], context()),
		).toBeDefined();
		runtime.stop();
	});

	it("retrieves locally, then remotely, then returns an actionable miss", async () => {
		const cache = new PreparedCache(1_000_000);
		const old = message("1", "x".repeat(20));
		cache.set(entry(old));
		const retrieve = vi
			.fn<WorkerClient["retrieve"]>()
			.mockResolvedValueOnce("remote original")
			.mockResolvedValueOnce(undefined)
			.mockRejectedValueOnce(new Error("offline"));
		const runtime = new HeadroomRuntime({
			config: config(),
			client: client({ retrieve }),
			cache,
		});

		await expect(runtime.retrieve("abcdef1234567890abcdef12")).resolves.toBe(
			old.content[0]?.type === "text" ? old.content[0].text : "",
		);
		await expect(runtime.retrieve("remote-hash")).resolves.toBe(
			"remote original",
		);
		await expect(runtime.retrieve("missing-hash")).resolves.toBe(
			"Headroom retrieval miss for missing-hash. Rerun the originating tool.",
		);
		await expect(runtime.retrieve("offline-hash")).resolves.toBe(
			"Headroom retrieval miss for offline-hash. Rerun the originating tool.",
		);
		runtime.stop();
	});
});

describe("status output", () => {
	const snapshot: RuntimeSnapshot = {
		enabled: true,
		modelId: "provider/model",
		tokensBefore: 2_000,
		tokensAfter: 1_000,
		tokensSaved: 1_000,
		bytesSaved: 4_096,
		retrievals: 4,
		lastTransform: {
			substitutions: 1,
			tokensBefore: 1_000,
			tokensAfter: 500,
			tokensSaved: 500,
			bytesBefore: 4_096,
			bytesAfter: 2_048,
			bytesSaved: 2_048,
		},
		config: config(),
		worker: {
			candidates: 9,
			queued: 2,
			active: 1,
			accepted: 3,
			rejected: 2,
			skipped: 0,
			dropped: 1,
			state: "offline",
			lastError: "compression request failed",
		},
		cache: { entries: 3, bytes: 1_024, maxBytes: 64 * 1_024 },
		warnings: ["config.minResultChars is invalid"],
	};

	it("reports effective configuration and current health", () => {
		const output = formatStatus(snapshot, true);

		expect(output.startsWith("Headroom offline\n")).toBe(true);
		expect(output).toContain("health offline");
		expect(output).toContain("endpoint http://127.0.0.1:8787");
		expect(output).toContain("remote blocked · hosts none");
		expect(output).toContain("queue 2 queued · 1 active");
		expect(output).toContain("cache 1,024/65,536 bytes");
		expect(output).toContain("last error compression request failed");
		expect(output).toContain("config.minResultChars is invalid");
	});

	it("separates prepared savings from the latest applied transform", () => {
		const output = formatStats(snapshot);
		const lines = output.split("\n");

		expect(lines).toContain("prepared candidates 9");
		expect(lines).toContain("prepared accepted 3");
		expect(lines).toContain("prepared rejected 2");
		expect(lines).toContain("prepared skipped 0");
		expect(lines).toContain("prepared bytes saved 4,096");
		expect(lines).toContain("prepared tokens saved 1,000");
		expect(lines).toContain("last transform substitutions 1");
		expect(lines).toContain("last transform bytes saved 2,048");
		expect(lines).toContain("last transform tokens saved 500");
		expect(lines).toContain("retrievals 4");
	});

	it("uses session savings in compact status", () => {
		expect(
			formatStatus({
				...snapshot,
				worker: { ...snapshot.worker, state: "online" },
			}),
		).toBe("Headroom saved 1,000 tokens this session");
	});

	it("stays online without claiming savings before any session total", () => {
		expect(
			formatStatus({
				...snapshot,
				tokensSaved: 0,
				lastTransform: {
					...snapshot.lastTransform,
					substitutions: 0,
					tokensSaved: 0,
				},
				worker: { ...snapshot.worker, state: "online", accepted: 2 },
			}),
		).toBe("Headroom online");
	});
});
