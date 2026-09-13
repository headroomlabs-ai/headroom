import { describe, expect, it, vi } from "vitest";

import { HeadroomClient } from "../src/client.js";
import type { Candidate } from "../src/types.js";

const candidate: Candidate = {
	key: "a".repeat(32),
	toolCallId: "host-call-id",
	toolName: "bash",
	originalText: "candidate text",
};

describe("HeadroomClient", () => {
	it("sends only the synthetic tool pair to the compression endpoint", async () => {
		const fetchImpl = vi.fn<typeof fetch>(
			async () =>
				new Response(JSON.stringify({ messages: [] }), {
					status: 200,
					headers: { "content-type": "application/json" },
				}),
		);
		const client = new HeadroomClient({
			baseUrl: "http://127.0.0.1:8787",
			timeoutMs: 1_000,
			fetchImpl,
		});

		await client.compress(candidate, "openai-codex/gpt-5.6-sol");

		expect(fetchImpl).toHaveBeenCalledTimes(1);
		const [url, init] = fetchImpl.mock.calls[0] ?? [];
		expect(url).toBe("http://127.0.0.1:8787/v1/compress");
		expect(init?.method).toBe("POST");
		expect(init?.redirect).toBe("error");
		const body = JSON.parse(String(init?.body)) as Record<string, unknown>;
		expect(body).toEqual({
			model: "openai-codex/gpt-5.6-sol",
			messages: [
				{
					role: "assistant",
					tool_calls: [
						{
							id: "call_aaaaaaaaaaaa",
							type: "function",
							function: { name: "bash", arguments: "{}" },
						},
					],
				},
				{
					role: "tool",
					tool_call_id: "call_aaaaaaaaaaaa",
					content: "candidate text",
				},
			],
			config: { compress_user_messages: false, mode: "ccr" },
		});
		expect(Object.keys(body)).toEqual(["model", "messages", "config"]);
	});

	it("aborts a stalled request at its deadline", async () => {
		const fetchImpl = vi.fn(
			(_url: string | URL | Request, init?: RequestInit) =>
				new Promise<Response>((_resolve, reject) => {
					init?.signal?.addEventListener(
						"abort",
						() => reject(init.signal?.reason),
						{ once: true },
					);
				}),
		);
		const client = new HeadroomClient({
			baseUrl: "http://127.0.0.1:8787",
			timeoutMs: 5,
			fetchImpl,
		});

		await expect(client.health()).rejects.toBeInstanceOf(Error);
		expect(fetchImpl.mock.calls[0]?.[1]?.redirect).toBe("error");
	});

	it("rejects non-success responses without exposing their body", async () => {
		const client = new HeadroomClient({
			baseUrl: "http://127.0.0.1:8787",
			timeoutMs: 1_000,
			fetchImpl: async () => new Response("secret body", { status: 503 }),
		});

		await expect(client.compress(candidate, "unknown")).rejects.toThrow(
			"Headroom compression failed with HTTP 503",
		);
		await expect(client.compress(candidate, "unknown")).rejects.not.toThrow(
			"secret body",
		);
	});

	it("rejects malformed JSON", async () => {
		const client = new HeadroomClient({
			baseUrl: "http://127.0.0.1:8787",
			timeoutMs: 1_000,
			fetchImpl: async () => new Response("not-json", { status: 200 }),
		});

		await expect(client.compress(candidate, "unknown")).rejects.toThrow(
			"Headroom compression returned malformed JSON",
		);
	});

	it("retrieves original content and treats 404 as a miss", async () => {
		const fetchImpl = vi
			.fn<typeof fetch>()
			.mockResolvedValueOnce(
				new Response(
					JSON.stringify({ hash: "abc", original_content: "original" }),
					{ status: 200 },
				),
			)
			.mockResolvedValueOnce(new Response("missing", { status: 404 }));
		const client = new HeadroomClient({
			baseUrl: "http://127.0.0.1:8787",
			timeoutMs: 1_000,
			fetchImpl,
		});

		await expect(client.retrieve("abc")).resolves.toBe("original");
		await expect(client.retrieve("missing")).resolves.toBeUndefined();
		const body = JSON.parse(
			String(fetchImpl.mock.calls[0]?.[1]?.body),
		) as Record<string, unknown>;
		expect(body).toEqual({ hash: "abc" });
		expect(Object.keys(body)).toEqual(["hash"]);
		expect(fetchImpl.mock.calls[0]?.[1]?.redirect).toBe("error");
	});

	it("rejects retrieval content bound to a different hash", async () => {
		const client = new HeadroomClient({
			baseUrl: "http://127.0.0.1:8787",
			timeoutMs: 1_000,
			fetchImpl: async () =>
				new Response(
					JSON.stringify({ hash: "other", original_content: "wrong" }),
					{ status: 200 },
				),
		});

		await expect(client.retrieve("requested")).rejects.toThrow(
			"Headroom retrieval returned an invalid response",
		);
	});
});
