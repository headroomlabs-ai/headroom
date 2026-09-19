import { createHash } from "node:crypto";

import { describe, expect, it, vi } from "vitest";

import {
	buildCompressPayload,
	isPreparedEntry,
	validateCompression,
} from "../src/bridge.js";
import type { Candidate } from "../src/types.js";

const HASH_A = "abcdef1234567890abcdef12";
const HASH_B = "1234567890abcdef12345678";
const SOURCE_A = `retrievable-a:${HASH_A}`;
const SOURCE_B = `retrievable-b:${HASH_B}`;
const candidate: Candidate = {
	key: "a".repeat(32),
	toolCallId: "host-call-id",
	toolName: "bash",
	originalText: `${SOURCE_A}\n${SOURCE_B}\n${"original ".repeat(1_000)}`,
};

function response(
	compressedText = `summary Retrieve more: hash=${HASH_A}`,
): Record<string, unknown> {
	return {
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
				content: compressedText,
			},
		],
		tokens_before: 2_000,
		tokens_after: 1_000,
		tokens_saved: 1_000,
		compression_ratio: 0.5,
		transforms_applied: ["test"],
		ccr_hashes: [HASH_A],
	};
}

describe("buildCompressPayload", () => {
	it("uses a deterministic synthetic call and empty arguments", () => {
		expect(buildCompressPayload(candidate, "test/model")).toEqual({
			model: "test/model",
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
					content: candidate.originalText,
				},
			],
			config: { compress_user_messages: false, mode: "ccr" },
		});
	});
});

describe("validateCompression", () => {
	it("accepts validated output and snapshots every retrieval", async () => {
		const compressedText = [
			`first Retrieve more: hash=${HASH_A}`,
			`second <<ccr:${HASH_B} 20_rows_offloaded>>`,
		].join("\n");
		const raw = response(compressedText);
		raw.ccr_hashes = [HASH_A, HASH_B, HASH_A];
		const retrieve = vi.fn(async (hash: string) =>
			hash === HASH_A ? SOURCE_A : SOURCE_B,
		);

		const entry = await validateCompression(candidate, raw, retrieve);
		expect(isPreparedEntry(entry)).toBe(true);
		if (!isPreparedEntry(entry)) return;

		expect(entry).toMatchObject({
			...candidate,
			compressedText,
			ccrHashes: [HASH_A, HASH_B],
			tokensBefore: 2_000,
			tokensAfter: 1_000,
			tokensSaved: 1_000,
			originalSha256: createHash("sha256")
				.update(candidate.originalText)
				.digest("hex"),
			policyVersion: "v1",
			createdAt: expect.any(Number),
			lastAccessedAt: expect.any(Number),
		});
		expect(entry.lastAccessedAt).toBe(entry.createdAt);
		expect(entry.retrievals).toEqual(
			new Map([
				[HASH_A, SOURCE_A],
				[HASH_B, SOURCE_B],
			]),
		);
		expect(entry.sizeBytes).toBe(
			Buffer.byteLength(compressedText) +
				Buffer.byteLength(candidate.originalText) +
				Buffer.byteLength(SOURCE_A) +
				Buffer.byteLength(SOURCE_B),
		);
		expect(retrieve).toHaveBeenCalledTimes(2);
	});

	it("discovers and validates CCR markers omitted from ccr_hashes", async () => {
		const compressedText = `opaque <<ccr:${HASH_A},base64,690B>>`;
		const raw = response(compressedText);
		raw.ccr_hashes = [];
		const retrieve = vi.fn(async () => SOURCE_A);

		const entry = await validateCompression(candidate, raw, retrieve);
		expect(isPreparedEntry(entry)).toBe(true);
		if (!isPreparedEntry(entry)) return;

		expect(entry.ccrHashes).toEqual([HASH_A]);
		expect(entry.retrievals.get(HASH_A)).toBe(SOURCE_A);
		expect(retrieve).toHaveBeenCalledWith(HASH_A);
	});

	it("ignores canonical marker metadata that is not a retrieval hash", async () => {
		const digestMarker = `<headroom:tool_digest sha256="${"a".repeat(64)}">`;
		const raw = response(`summary ${digestMarker}`);
		raw.ccr_hashes = [digestMarker];
		const retrieve = vi.fn(async () => undefined);

		const entry = await validateCompression(candidate, raw, retrieve);
		expect(isPreparedEntry(entry)).toBe(true);
		if (!isPreparedEntry(entry)) return;

		expect(entry.ccrHashes).toEqual([]);
		expect(entry.retrievals).toEqual(new Map());
		expect(retrieve).not.toHaveBeenCalled();
	});

	it("rejects a response for a different synthetic tool", async () => {
		const raw = response();
		const assistant = (
			raw.messages as Array<{
				tool_calls?: Array<{ function?: { name?: string } }>;
			}>
		)[0];
		const syntheticFunction = assistant?.tool_calls?.[0]?.function;
		if (syntheticFunction) syntheticFunction.name = "read";

		await expect(
			validateCompression(candidate, raw, async () => "original"),
		).resolves.toEqual({
			status: "rejected",
			reason: "invalid_response",
		});
	});

	it.each([
		{
			label: "missing messages",
			mutate: (raw: Record<string, unknown>) => delete raw.messages,
		},
		{
			label: "wrong message count",
			mutate: (raw: Record<string, unknown>) => (raw.messages = []),
		},
		{
			label: "wrong message order",
			mutate: (raw: Record<string, unknown>) =>
				(raw.messages as unknown[]).reverse(),
		},
		{
			label: "wrong tool id",
			mutate: (raw: Record<string, unknown>) =>
				(((raw.messages as Record<string, unknown>[])[1] ?? {}).tool_call_id =
					"wrong"),
		},
		{
			label: "empty output",
			mutate: (raw: Record<string, unknown>) =>
				(((raw.messages as Record<string, unknown>[])[1] ?? {}).content = ""),
		},
		{
			label: "growing output",
			mutate: (raw: Record<string, unknown>) =>
				(((raw.messages as Record<string, unknown>[])[1] ?? {}).content =
					"x".repeat(candidate.originalText.length)),
		},
		{
			label: "derived ratio too high",
			mutate: (raw: Record<string, unknown>) => (raw.tokens_after = 1_900),
		},
		{
			label: "reported ratio too high",
			mutate: (raw: Record<string, unknown>) => (raw.compression_ratio = 0.91),
		},
		{
			label: "hash absent from output",
			mutate: (raw: Record<string, unknown>) => (raw.ccr_hashes = [HASH_B]),
		},
	])("rejects $label", async ({ mutate }) => {
		const raw = response();
		mutate(raw);

		await expect(
			validateCompression(candidate, raw, async () => "original"),
		).resolves.toEqual({
			status: "rejected",
			reason: expect.any(String),
		});
	});

	it("rejects when any CCR retrieval misses", async () => {
		const raw = response(
			`first Retrieve more: hash=${HASH_A}\nsecond Retrieve more: hash=${HASH_B}`,
		);
		raw.ccr_hashes = [HASH_A, HASH_B];

		await expect(
			validateCompression(candidate, raw, async (hash) =>
				hash === HASH_A ? "original" : undefined,
			),
		).resolves.toEqual({
			status: "rejected",
			reason: "retrieval_miss",
		});
	});

	it("rejects a nonempty CCR retrieval outside the source result", async () => {
		await expect(
			validateCompression(
				candidate,
				response(),
				async () => "unrelated payload",
			),
		).resolves.toEqual({
			status: "rejected",
			reason: "retrieval_mismatch",
		});
	});

	it("names a no-op as skipped instead of rejected", async () => {
		const raw = response(candidate.originalText);
		raw.tokens_before = 43;
		raw.tokens_after = 43;
		raw.tokens_saved = 0;
		raw.compression_ratio = 1;
		raw.transforms_applied = ["router:noop"];
		raw.ccr_hashes = [];

		await expect(
			validateCompression(candidate, raw, async () => undefined),
		).resolves.toEqual({
			status: "skipped",
			reason: "noop",
		});
	});

	it("returns a typed reason for insufficient savings", async () => {
		const raw = response();
		raw.tokens_saved = 255;

		await expect(
			validateCompression(candidate, raw, async () => SOURCE_A),
		).resolves.toEqual({
			status: "rejected",
			reason: "insufficient_savings",
		});
	});
});
