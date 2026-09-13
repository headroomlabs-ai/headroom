import { afterEach, describe, expect, it, vi } from "vitest";

import { PreparedCache } from "../src/cache.js";
import { CompressionWorker, type WorkerClient } from "../src/worker.js";
import type { Candidate } from "../src/types.js";

function candidate(key: string): Candidate {
	return {
		key: key.padEnd(32, "0"),
		toolCallId: `call-${key}`,
		toolName: "bash",
		originalText: `original-${key}-`.repeat(1_000),
	};
}

function validResponse(item: Candidate): Record<string, unknown> {
	const callId = `call_${item.key.slice(0, 12)}`;
	return {
		messages: [
			{
				role: "assistant",
				tool_calls: [
					{
						id: callId,
						type: "function",
						function: { name: item.toolName, arguments: "{}" },
					},
				],
			},
			{ role: "tool", tool_call_id: callId, content: `summary-${item.key}` },
		],
		tokens_before: 2_000,
		tokens_after: 1_000,
		tokens_saved: 1_000,
		compression_ratio: 0.5,
		transforms_applied: ["test"],
		ccr_hashes: [],
	};
}

interface Deferred<T> {
	promise: Promise<T>;
	resolve: (value: T) => void;
	reject: (reason?: unknown) => void;
}

function deferred<T>(): Deferred<T> {
	let resolve!: (value: T) => void;
	let reject!: (reason?: unknown) => void;
	const promise = new Promise<T>((resolvePromise, rejectPromise) => {
		resolve = resolvePromise;
		reject = rejectPromise;
	});
	return { promise, resolve, reject };
}

function client(
	compress: WorkerClient["compress"],
	health: WorkerClient["health"] = async () => true,
): WorkerClient {
	return {
		compress,
		health,
		retrieve: async () => undefined,
	};
}

async function flushMicrotasks(): Promise<void> {
	await Promise.resolve();
	await Promise.resolve();
	await Promise.resolve();
}

afterEach(() => {
	vi.useRealTimers();
});

describe("CompressionWorker", () => {
	it("deduplicates queued and active keys", async () => {
		const pending = deferred<unknown>();
		const compress = vi.fn(() => pending.promise);
		const worker = new CompressionWorker({
			client: client(compress),
			cache: new PreparedCache(1_000_000),
		});
		const item = candidate("a");

		expect(worker.enqueue(item, "model")).toBe(true);
		expect(worker.enqueue(item, "model")).toBe(false);
		await flushMicrotasks();
		expect(worker.enqueue(item, "model")).toBe(false);
		expect(compress).toHaveBeenCalledTimes(1);

		pending.resolve(validResponse(item));
		await vi.waitFor(() => expect(worker.stats().accepted).toBe(1));
		expect(worker.enqueue(item, "model")).toBe(false);
		worker.stop();
	});

	it("bounds concurrency with deferred work", async () => {
		const pending = new Map<string, Deferred<unknown>>();
		let active = 0;
		let peak = 0;
		const compress = vi.fn(async (item: Candidate) => {
			active += 1;
			peak = Math.max(peak, active);
			const gate = deferred<unknown>();
			pending.set(item.key, gate);
			const result = await gate.promise;
			active -= 1;
			return result;
		});
		const worker = new CompressionWorker({
			client: client(compress),
			cache: new PreparedCache(1_000_000),
			concurrency: 2,
		});
		const items = [candidate("a"), candidate("b"), candidate("c")];

		items.forEach((item) => worker.enqueue(item, "model"));
		await flushMicrotasks();
		expect(peak).toBe(2);
		expect(compress).toHaveBeenCalledTimes(2);

		pending.get(items[0]!.key)?.resolve(validResponse(items[0]!));
		await vi.waitFor(() => expect(compress).toHaveBeenCalledTimes(3));
		expect(peak).toBe(2);
		pending.get(items[1]!.key)?.resolve(validResponse(items[1]!));
		pending.get(items[2]!.key)?.resolve(validResponse(items[2]!));
		await vi.waitFor(() => expect(worker.stats().accepted).toBe(3));
		worker.stop();
	});

	it("drops the oldest work that has not started when the queue is full", async () => {
		const firstGate = deferred<unknown>();
		const calls: string[] = [];
		const compress = vi.fn(async (item: Candidate) => {
			calls.push(item.key[0]!);
			if (item.key.startsWith("a")) return firstGate.promise;
			return validResponse(item);
		});
		const worker = new CompressionWorker({
			client: client(compress),
			cache: new PreparedCache(1_000_000),
			concurrency: 1,
			maxQueue: 2,
		});
		const a = candidate("a");
		const b = candidate("b");
		const c = candidate("c");
		const d = candidate("d");

		worker.enqueue(a, "model");
		await flushMicrotasks();
		worker.enqueue(b, "model");
		worker.enqueue(c, "model");
		worker.enqueue(d, "model");
		expect(worker.stats().dropped).toBe(1);

		firstGate.resolve(validResponse(a));
		await vi.waitFor(() => expect(worker.stats().accepted).toBe(3));
		expect(calls).toEqual(["a", "c", "d"]);
		worker.stop();
	});

	it("aborts active work on stop", async () => {
		let observedSignal: AbortSignal | undefined;
		const compress = vi.fn(
			(_item: Candidate, _model: string, signal?: AbortSignal) =>
				new Promise<unknown>((_resolve, reject) => {
					observedSignal = signal;
					signal?.addEventListener("abort", () => reject(signal.reason), {
						once: true,
					});
				}),
		);
		const worker = new CompressionWorker({
			client: client(compress),
			cache: new PreparedCache(1_000_000),
		});

		worker.enqueue(candidate("a"), "model");
		await flushMicrotasks();
		worker.stop();
		await flushMicrotasks();

		expect(observedSignal?.aborted).toBe(true);
		expect(worker.stats().state).toBe("stopped");
	});

	it("counts rejected validation without caching it", async () => {
		const cache = new PreparedCache(1_000_000);
		const item = candidate("a");
		const worker = new CompressionWorker({
			client: client(async () => ({ messages: [] })),
			cache,
		});

		worker.enqueue(item, "model");
		await vi.waitFor(() => expect(worker.stats().rejected).toBe(1));
		expect(cache.get(item.key)).toBeUndefined();
		expect(worker.stats().lastError).toBe("invalid_response");
		worker.stop();
	});

	it("does not count a valid no-op as a rejection", async () => {
		const cache = new PreparedCache(1_000_000);
		const item = candidate("a");
		const worker = new CompressionWorker({
			client: client(async () => ({
				messages: [
					{
						role: "assistant",
						tool_calls: [
							{
								id: `call_${item.key.slice(0, 12)}`,
								type: "function",
								function: { name: item.toolName, arguments: "{}" },
							},
						],
					},
					{
						role: "tool",
						tool_call_id: `call_${item.key.slice(0, 12)}`,
						content: item.originalText,
					},
				],
				tokens_before: 43,
				tokens_after: 43,
				tokens_saved: 0,
				compression_ratio: 1,
				transforms_applied: ["router:noop"],
				ccr_hashes: [],
			})),
			cache,
		});

		worker.enqueue(item, "model");
		await vi.waitFor(() => expect(worker.stats().skipped).toBe(1));
		expect(worker.stats().rejected).toBe(0);
		expect(worker.stats().lastError).toBeUndefined();
		expect(cache.get(item.key)).toBeUndefined();
		worker.stop();
	});

	it("records cache_full separately from validation rejection", async () => {
		const item = candidate("a");
		const worker = new CompressionWorker({
			client: client(async () => validResponse(item)),
			cache: new PreparedCache(1),
		});

		worker.enqueue(item, "model");
		await vi.waitFor(() => expect(worker.stats().rejected).toBe(1));
		expect(worker.stats().lastError).toBe("cache_full");
		worker.stop();
	});

	it("clears lastError after a later accepted compression", async () => {
		const first = candidate("a");
		const second = candidate("b");
		const compress = vi
			.fn<WorkerClient["compress"]>()
			.mockResolvedValueOnce({ messages: [] })
			.mockResolvedValueOnce(validResponse(second));
		const worker = new CompressionWorker({
			client: client(compress),
			cache: new PreparedCache(1_000_000),
		});

		worker.enqueue(first, "model");
		await vi.waitFor(() => expect(worker.stats().rejected).toBe(1));
		expect(worker.stats().lastError).toBe("invalid_response");

		worker.enqueue(second, "model");
		await vi.waitFor(() => expect(worker.stats().accepted).toBe(1));
		expect(worker.stats().lastError).toBeUndefined();
		worker.stop();
	});

	it("suppresses compression attempts during failure backoff", async () => {
		vi.useFakeTimers();
		vi.spyOn(Math, "random").mockReturnValue(0);
		const compress = vi.fn(async (item: Candidate) => {
			if (item.key.startsWith("a")) throw new Error("offline");
			return validResponse(item);
		});
		const worker = new CompressionWorker({
			client: client(compress),
			cache: new PreparedCache(1_000_000),
		});

		expect(worker.enqueue(candidate("a"), "model")).toBe(true);
		await flushMicrotasks();
		expect(worker.stats().state).toBe("offline");
		expect(worker.enqueue(candidate("b"), "model")).toBe(false);
		expect(compress).toHaveBeenCalledTimes(1);

		await vi.advanceTimersByTimeAsync(1_000);
		await flushMicrotasks();
		expect(worker.stats().state).toBe("online");
		expect(worker.enqueue(candidate("c"), "model")).toBe(true);
		await flushMicrotasks();
		expect(compress).toHaveBeenCalledTimes(2);
		worker.stop();
	});

	it("holds queued work until failure backoff recovers", async () => {
		vi.useFakeTimers();
		vi.spyOn(Math, "random").mockReturnValue(0);
		const compress = vi
			.fn<WorkerClient["compress"]>()
			.mockRejectedValueOnce(new Error("offline"))
			.mockImplementation(async (item) => validResponse(item));
		const worker = new CompressionWorker({
			client: client(compress),
			cache: new PreparedCache(1_000_000),
			concurrency: 1,
		});

		expect(worker.enqueue(candidate("a"), "model")).toBe(true);
		expect(worker.enqueue(candidate("b"), "model")).toBe(true);
		await vi.advanceTimersByTimeAsync(0);
		await flushMicrotasks();
		expect(worker.stats().state).toBe("offline");
		expect(compress).toHaveBeenCalledTimes(1);

		await vi.advanceTimersByTimeAsync(1_000);
		await flushMicrotasks();
		expect(compress).toHaveBeenCalledTimes(2);
		expect(worker.stats().accepted).toBe(1);
		worker.stop();
	});

	it("retries health with bounded backoff and emits state changes", async () => {
		vi.useFakeTimers();
		vi.spyOn(Math, "random").mockReturnValue(0);
		const health = vi
			.fn<WorkerClient["health"]>()
			.mockResolvedValueOnce(false)
			.mockResolvedValueOnce(true);
		const states: string[] = [];
		const worker = new CompressionWorker({
			client: client(async (item) => validResponse(item), health),
			cache: new PreparedCache(1_000_000),
			onStateChange: (state) => states.push(state),
		});

		await expect(worker.checkHealth()).resolves.toBe(false);
		expect(worker.stats().state).toBe("offline");
		await vi.advanceTimersByTimeAsync(1_000);
		await flushMicrotasks();

		expect(health).toHaveBeenCalledTimes(2);
		expect(worker.stats().state).toBe("online");
		expect(states).toEqual(["offline", "warming", "online"]);
		worker.stop();
	});
});
