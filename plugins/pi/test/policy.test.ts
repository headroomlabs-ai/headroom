import type { ToolResultMessage } from "@earendil-works/pi-ai";
import { describe, expect, it, vi } from "vitest";

import { DEFAULT_CONFIG, type HeadroomConfig } from "../src/config.js";
import {
	candidateFromToolResult,
	protectedToolResultIndexes,
	transformContext,
} from "../src/policy.js";
import type { ContextMessage, PreparedEntry } from "../src/types.js";

function config(overrides: Partial<HeadroomConfig> = {}): HeadroomConfig {
	return {
		...DEFAULT_CONFIG,
		protectedTools: [...DEFAULT_CONFIG.protectedTools],
		minResultChars: 10,
		...overrides,
	};
}

function toolResult(
	toolCallId: string,
	toolName: string,
	text: string,
	overrides: Partial<ToolResultMessage> = {},
): ToolResultMessage {
	return {
		role: "toolResult",
		toolCallId,
		toolName,
		content: [{ type: "text", text }],
		isError: false,
		timestamp: 1,
		details: { stable: true },
		...overrides,
	};
}

function prepared(
	message: ToolResultMessage,
	compressedText = "compressed Retrieve more: hash=abcdef1234567890abcdef12",
): PreparedEntry {
	const candidate = candidateFromToolResult(message, config());
	if (!candidate) throw new Error("test fixture must be eligible");
	return {
		...candidate,
		compressedText,
		ccrHashes: ["abcdef1234567890abcdef12"],
		retrievals: new Map([["abcdef1234567890abcdef12", candidate.originalText]]),
		tokensBefore: 1_000,
		tokensAfter: 500,
		tokensSaved: 500,
		originalBytes: Buffer.byteLength(candidate.originalText),
		compressedBytes: Buffer.byteLength(compressedText),
		sizeBytes: 100,
		originalSha256: "sha",
		policyVersion: "v1",
		createdAt: 1,
		lastAccessedAt: 1,
	};
}

describe("candidateFromToolResult", () => {
	it("protects configured tools and error results", () => {
		expect(
			candidateFromToolResult(
				toolResult("1", "READ", "x".repeat(20)),
				config(),
			),
		).toBeUndefined();
		expect(
			candidateFromToolResult(
				toolResult("2", "bash", "x".repeat(20), { isError: true }),
				config(),
			),
		).toBeUndefined();
	});

	it("rejects multiple text blocks and images", () => {
		expect(
			candidateFromToolResult(
				toolResult("1", "bash", "x".repeat(20), {
					content: [
						{ type: "text", text: "x".repeat(20) },
						{ type: "text", text: "second" },
					],
				}),
				config(),
			),
		).toBeUndefined();
		expect(
			candidateFromToolResult(
				toolResult("2", "bash", "x".repeat(20), {
					content: [
						{ type: "text", text: "x".repeat(20) },
						{ type: "image", data: "AA==", mimeType: "image/png" },
					],
				}),
				config(),
			),
		).toBeUndefined();
	});

	it("rejects short and already-compressed output", () => {
		expect(
			candidateFromToolResult(toolResult("1", "bash", "short"), config()),
		).toBeUndefined();
		expect(
			candidateFromToolResult(
				toolResult(
					"2",
					"bash",
					"compressed Retrieve more: hash=abcdef1234567890abcdef12",
				),
				config(),
			),
		).toBeUndefined();
		expect(
			candidateFromToolResult(
				toolResult("3", "bash", "x".repeat(20), {
					details: { prunedAt: 123 },
				}),
				config(),
			),
		).toBeUndefined();
		expect(
			candidateFromToolResult(
				toolResult("4", "bash", "x".repeat(20), {
					details: { truncation: { truncated: true } },
				}),
				config(),
			),
		).toBeUndefined();
		expect(
			candidateFromToolResult(
				toolResult("5", "bash", "x".repeat(20), {
					details: { truncation: { truncated: false } },
				}),
				config(),
			),
		).toBeDefined();
		expect(
			candidateFromToolResult(
				toolResult(
					"6",
					"bash",
					`summary <headroom:tool_digest sha256="${"a".repeat(64)}">`,
				),
				config(),
			),
		).toBeUndefined();
	});

	it.each([
		["direct", { rtkCompaction: { truncated: true } }],
		["nested", { metadata: { rtkCompaction: { truncated: true } } }],
		[
			"emitted direct and nested",
			{
				rtkCompaction: { truncated: true },
				metadata: { rtkCompaction: { truncated: true } },
			},
		],
		[
			"nested after direct non-truncated",
			{
				rtkCompaction: { truncated: false },
				metadata: { rtkCompaction: { truncated: true } },
			},
		],
		[
			"direct before nested non-truncated",
			{
				rtkCompaction: { truncated: true },
				metadata: { rtkCompaction: { truncated: false } },
			},
		],
	] as const)("rejects RTK-truncated output from %s metadata", (_shape, details) => {
		expect(
			candidateFromToolResult(
				toolResult("rtk", "bash", "x".repeat(20), { details }),
				config(),
			),
		).toBeUndefined();
	});

	it.each([
		["direct non-truncated", { rtkCompaction: { truncated: false } }],
		[
			"nested non-truncated",
			{ metadata: { rtkCompaction: { truncated: false } } },
		],
		[
			"direct and nested non-truncated",
			{
				rtkCompaction: { truncated: false },
				metadata: { rtkCompaction: { truncated: false } },
			},
		],
		["malformed direct", { rtkCompaction: "invalid" }],
		["malformed metadata", { metadata: "invalid" }],
		[
			"malformed nested compaction",
			{ metadata: { rtkCompaction: "invalid" } },
		],
		["non-boolean truncated flag", { rtkCompaction: { truncated: "true" } }],
		["null shapes", { rtkCompaction: null, metadata: null }],
	] as const)("keeps %s RTK metadata eligible", (_shape, details) => {
		expect(
			candidateFromToolResult(
				toolResult("rtk", "bash", "x".repeat(20), { details }),
				config(),
			),
		).toBeDefined();
	});

	it("rejects repeated incomplete CCR prefixes without pathological backtracking", () => {
		const started = performance.now();

		expect(
			candidateFromToolResult(
				toolResult("adversarial", "bash", "<<ccr:".repeat(50_000)),
				config(),
			),
		).toBeUndefined();
		expect(performance.now() - started).toBeLessThan(100);
	});

	it("invalidates the cached identity when source text changes", () => {
		const message = toolResult("mutable", "bash", "x".repeat(20));
		const initial = candidateFromToolResult(message, config());
		const textBlock = message.content[0];
		if (textBlock?.type !== "text") throw new Error("expected text fixture");
		textBlock.text = "y".repeat(20);

		const changed = candidateFromToolResult(message, config());

		expect(changed?.originalText).toBe("y".repeat(20));
		expect(changed?.key).not.toBe(initial?.key);
	});
});

describe("protectedToolResultIndexes", () => {
	it("protects the newest results by message position", () => {
		const messages = [
			toolResult("1", "bash", "first result"),
			{
				role: "custom",
				customType: "gap",
				content: "gap",
				display: false,
				timestamp: 2,
			},
			toolResult("2", "bash", "second result"),
			toolResult("3", "bash", "third result"),
		] as ContextMessage[];

		expect(protectedToolResultIndexes(messages, 2)).toEqual(new Set([2, 3]));
	});
});

describe("transformContext", () => {
	it("queues an uncached cold candidate without changing the current context", () => {
		const message = toolResult("1", "bash", "x".repeat(20));
		const enqueue = vi.fn();

		const result = transformContext(
			[message],
			config({ protectRecentToolResults: 1 }),
			() => undefined,
			enqueue,
		);

		expect(result).toBeUndefined();
		expect(enqueue).not.toHaveBeenCalled();

		transformContext(
			[message, toolResult("2", "bash", "recent".repeat(3))],
			config({ protectRecentToolResults: 1 }),
			() => undefined,
			enqueue,
		);
		expect(enqueue).toHaveBeenCalledTimes(1);
		expect(enqueue.mock.calls[0]?.[0].originalText).toBe("x".repeat(20));
	});

	it("copies only the substituted message and text block", () => {
		const old = toolResult("1", "bash", "x".repeat(20));
		Object.assign(old.content[0]!, { provenance: "host" });
		const recent = toolResult("2", "bash", "recent".repeat(3));
		const entry = prepared(old);
		const messages = [old, recent] as ContextMessage[];

		const result = transformContext(
			messages,
			config({ protectRecentToolResults: 1 }),
			() => entry,
			vi.fn(),
		);

		expect(result).toBeDefined();
		expect(result).not.toBe(messages);
		expect(result?.[0]).not.toBe(old);
		expect(result?.[1]).toBe(recent);
		expect((result?.[0] as ToolResultMessage).content).not.toBe(old.content);
		expect((result?.[0] as ToolResultMessage).content[0]).not.toBe(
			old.content[0],
		);
		expect((result?.[0] as ToolResultMessage).details).toBe(old.details);
		expect((result?.[0] as ToolResultMessage).toolCallId).toBe(old.toolCallId);
		expect((result?.[0] as ToolResultMessage).content[0]).toEqual({
			type: "text",
			text: entry.compressedText,
			provenance: "host",
		});
	});

	it("requires byte-exact source text before substitution", () => {
		const message = toolResult("1", "bash", "x".repeat(20));
		const entry = prepared(message);
		entry.originalText = `${entry.originalText}changed`;

		expect(
			transformContext(
				[message, toolResult("2", "bash", "recent".repeat(3))],
				config({ protectRecentToolResults: 1 }),
				() => entry,
				vi.fn(),
			),
		).toBeUndefined();
	});

	it("preserves unknown message roles and ordering", () => {
		const custom = {
			role: "custom",
			customType: "evidence",
			content: "untouched",
			display: false,
			timestamp: 1,
		} as ContextMessage;
		const old = toolResult("1", "bash", "x".repeat(20));
		const recent = toolResult("2", "bash", "recent".repeat(3));
		const result = transformContext(
			[custom, old, recent],
			config({ protectRecentToolResults: 1 }),
			() => prepared(old),
			vi.fn(),
		);

		expect(result?.map((message) => message.role)).toEqual([
			"custom",
			"toolResult",
			"toolResult",
		]);
		expect(result?.[0]).toBe(custom);
		expect(result?.[2]).toBe(recent);
	});
});
