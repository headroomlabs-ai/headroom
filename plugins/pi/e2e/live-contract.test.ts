import { createHash, randomUUID } from "node:crypto";
import { afterEach, beforeAll, describe, expect, it } from "vitest";
import { isPreparedEntry, validateCompression } from "../src/bridge.js";
import { PreparedCache } from "../src/cache.js";
import { HeadroomClient } from "../src/client.js";
import { transformContext } from "../src/policy.js";
import { HeadroomRuntime } from "../src/runtime.js";
import type { HeadroomConfig } from "../src/config.js";
import type { ContextMessage } from "../src/types.js";

const baseUrl = process.env.HEADROOM_LIVE_BASE_URL ?? "http://127.0.0.1:8787";
const modelA = "openai-codex/gpt-5.6-sol";
const modelB = "anthropic/claude-fable-5";
const runtimes: HeadroomRuntime[] = [];
const runNonce = randomUUID();

const config: HeadroomConfig = {
	enabled: true,
	baseUrl,
	allowRemote: false,
	remoteHosts: [],
	minContextTokens: 20_000,
	minResultChars: 4_000,
	protectRecentToolResults: 1,
	protectedTools: ["read", "edit", "write", "ask", "todo", "headroom_retrieve"],
	maxCacheBytes: 67_108_864,
};

function originalFixture(seed: string): string {
	const alphabet =
		"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/=";
	return JSON.stringify(
		Array.from({ length: 160 }, (_, index) => ({
			id: index,
			blob: (alphabet + seed + String(index).padStart(4, "0")).repeat(10),
			checksum: createHash("sha256").update(`${seed}-${index}`).digest("hex"),
		})),
	);
}

function candidate(originalText: string, toolCallId: string) {
	return {
		key: createHash("sha256")
			.update("v1\0bash\0")
			.update(originalText)
			.digest("hex"),
		toolCallId,
		toolName: "bash",
		originalText,
	};
}

function context(modelId: string) {
	const [provider, ...idParts] = modelId.split("/");
	return {
		model: { id: idParts.join("/"), provider, contextWindow: 128_000 },
		ui: { setStatus() {} },
	} as never;
}

function toolResult(toolCallId: string, text: string) {
	return {
		role: "toolResult",
		toolCallId,
		toolName: "bash",
		content: [{ type: "text", text, provenance: "host" }],
		details: { source: "live-contract" },
		isError: false,
		timestamp: 1,
	};
}

async function waitFor(
	predicate: () => boolean,
	timeoutMs = 30_000,
): Promise<void> {
	// This integration gate waits on real proxy I/O and the worker's real backoff timer.
	const deadline = Date.now() + timeoutMs;
	while (!predicate()) {
		if (Date.now() >= deadline)
			throw new Error("timed out waiting for condition");
		await new Promise((resolve) => setTimeout(resolve, 25));
	}
}

beforeAll(async () => {
	const response = await fetch(`${baseUrl}/readyz`, { redirect: "error" });
	expect(response.ok).toBe(true);
});

afterEach(() => {
	for (const runtime of runtimes.splice(0)) runtime.stop();
});

describe("live Headroom contract", () => {
	it("validates real compression and retrieval before model-independent substitution", async () => {
		const originalText = originalFixture(`contract-${runNonce}`);
		const preparedCandidate = candidate(originalText, "live-contract");
		const requests: Array<{ url: string; body?: Record<string, unknown> }> = [];
		const fetchImpl: typeof fetch = async (input, init) => {
			const request: { url: string; body?: Record<string, unknown> } = {
				url: String(input),
			};
			if (typeof init?.body === "string") {
				request.body = JSON.parse(init.body) as Record<string, unknown>;
			}
			requests.push(request);
			return fetch(input, init);
		};
		const client = new HeadroomClient({
			baseUrl,
			timeoutMs: 30_000,
			fetchImpl,
		});
		const response = await client.compress(preparedCandidate, modelA);
		const retrieved = new Map<string, string>();
		const entry = await validateCompression(
			preparedCandidate,
			response,
			async (hash) => {
				const original = await client.retrieve(hash);
				if (original !== undefined) retrieved.set(hash, original);
				return original;
			},
		);

		expect(isPreparedEntry(entry)).toBe(true);
		if (!isPreparedEntry(entry)) return;
		expect(entry.ccrHashes.length).toBeGreaterThan(0);
		expect(retrieved.size).toBe(entry.ccrHashes.length);
		expect(
			[...retrieved.values()].every(
				(value) => value.length > 0 && originalText.includes(value),
			),
		).toBe(true);
		expect(requests[0]).toMatchObject({
			url: `${baseUrl}/v1/compress`,
			body: { model: modelA },
		});
		expect(
			requests
				.filter((request) => request.url.includes("/v1/retrieve/"))
				.every((request) => request.url.startsWith(baseUrl)),
		).toBe(true);

		const cache = new PreparedCache(config.maxCacheBytes);
		expect(cache.set(entry)).toBe(true);
		const runtime = new HeadroomRuntime({ config, client, cache });
		runtimes.push(runtime);
		const old = toolResult("live-contract", originalText);
		const recent = toolResult("recent", "recent".repeat(10));
		const rawBefore = structuredClone([old, recent]);
		const transformed = runtime.transform(
			[old, recent] as never,
			context(modelB),
		);

		expect(transformed).toBeDefined();
		expect((transformed?.[0] as typeof old).content[0]?.text).toBe(
			entry.compressedText,
		);
		expect(transformed?.[1]).toBe(recent);
		expect([old, recent]).toEqual(rawBefore);
		expect(runtime.snapshot().modelId).toBe(modelB);
		expect(
			requests.filter((request) => request.url.endsWith("/v1/compress")),
		).toHaveLength(1);
	}, 60_000);

	it("fails open while offline and resumes prepared substitution after recovery", async () => {
		let online = false;
		const fetchImpl: typeof fetch = (input, init) => {
			if (!online) return Promise.reject(new Error("simulated outage"));
			return fetch(input, init);
		};
		const client = new HeadroomClient({
			baseUrl,
			timeoutMs: 30_000,
			fetchImpl,
		});
		const runtime = new HeadroomRuntime({ config, client });
		runtimes.push(runtime);
		runtime.start(context(modelA));
		const originalText = originalFixture(`recovery-${runNonce}`);
		const event = {
			toolCallId: "live-recovery",
			toolName: "bash",
			input: {},
			content: [{ type: "text", text: originalText }],
			details: { source: "live-contract" },
			isError: false,
		} as never;
		const old = toolResult("live-recovery", originalText);
		const recent = toolResult("recent-recovery", "recent".repeat(10));

		runtime.observeToolResult(event, context(modelA));
		await waitFor(() => runtime.snapshot().worker.state === "offline");
		expect(
			runtime.transform([old, recent] as never, context(modelA)),
		).toBeUndefined();

		online = true;
		await waitFor(() => runtime.snapshot().worker.state === "online", 10_000);
		runtime.observeToolResult(event, context(modelB));
		await waitFor(() => runtime.snapshot().worker.accepted === 1, 60_000);
		expect(
			runtime.transform([old, recent] as never, context(modelB)),
		).toBeDefined();
		expect(runtime.snapshot().modelId).toBe(modelB);
	}, 75_000);

	it("keeps the 1 MiB context hot path below 10 ms p95", () => {
		const messages = Array.from({ length: 20 }, (_, index) => ({
			role: "toolResult",
			toolCallId: `perf-${index}`,
			toolName: "bash",
			content: [{ type: "text", text: `${index}:` + "x".repeat(52_428) }],
			details: {},
			isError: false,
			timestamp: index,
		})) as ContextMessage[];
		const sourceBytes = messages.reduce(
			(total, message) =>
				total +
				Buffer.byteLength(
					(message as ReturnType<typeof toolResult>).content[0]?.text ?? "",
				),
			0,
		);
		const samples: number[] = [];

		for (let index = 0; index < 10; index += 1) {
			transformContext(
				messages,
				config,
				() => undefined,
				() => undefined,
			);
		}
		for (let index = 0; index < 100; index += 1) {
			const started = performance.now();
			transformContext(
				messages,
				config,
				() => undefined,
				() => undefined,
			);
			samples.push(performance.now() - started);
		}
		samples.sort((left, right) => left - right);

		expect(sourceBytes).toBeGreaterThanOrEqual(1_048_576);
		expect(samples[94]).toBeLessThan(10);
	});
});
