import { createHash } from "node:crypto";

import { POLICY_VERSION, type Candidate, type PreparedEntry } from "./types.js";

export interface CompressPayload {
	model: string;
	messages: [
		{
			role: "assistant";
			tool_calls: [
				{
					id: string;
					type: "function";
					function: { name: string; arguments: "{}" };
				},
			];
		},
		{ role: "tool"; tool_call_id: string; content: string },
	];
	config: { compress_user_messages: false; mode: "ccr" };
}

interface ParsedCompression {
	compressedText: string;
	ccrHashes: string[];
	tokensBefore: number;
	tokensAfter: number;
	tokensSaved: number;
	compressionRatio: number;
}

export type RetrieveOriginal = (hash: string) => Promise<string | undefined>;

export type CompressionOutcome =
	| PreparedEntry
	| { status: "skipped"; reason: "noop" }
	| { status: "rejected"; reason: string };

export function isPreparedEntry(
	value: CompressionOutcome | undefined,
): value is PreparedEntry {
	return value !== undefined && !("status" in value);
}

export function buildCompressPayload(
	candidate: Candidate,
	modelId: string,
): CompressPayload {
	const syntheticCallId = `call_${candidate.key.slice(0, 12)}`;
	return {
		model: modelId,
		messages: [
			{
				role: "assistant",
				tool_calls: [
					{
						id: syntheticCallId,
						type: "function",
						function: { name: candidate.toolName, arguments: "{}" },
					},
				],
			},
			{
				role: "tool",
				tool_call_id: syntheticCallId,
				content: candidate.originalText,
			},
		],
		config: { compress_user_messages: false, mode: "ccr" },
	};
}

function nonnegativeNumber(value: unknown): value is number {
	return typeof value === "number" && Number.isFinite(value) && value >= 0;
}
const CCR_PREFIX_RE = /Retrieve (?:more|original): hash=|<<ccr:/gi;
const CCR_HASH_RE =
	/(?:Retrieve (?:more|original): hash=|<<ccr:)([a-f0-9]{12,24})(?=$|[\s,>\].])/gi;

function discoverCcrHashes(text: string): string[] | undefined {
	const prefixCount = text.match(CCR_PREFIX_RE)?.length ?? 0;
	const hashes: string[] = [];
	let markerCount = 0;
	for (const match of text.matchAll(CCR_HASH_RE)) {
		markerCount += 1;
		const hash = match[1];
		if (hash) hashes.push(hash.toLowerCase());
	}
	if (markerCount !== prefixCount) return undefined;
	return hashes;
}

function extractRetrievalHashes(markers: readonly string[]): string[] {
	const hashes: string[] = [];
	for (const marker of markers) {
		if (/^[a-f0-9]{12,24}$/i.test(marker)) {
			hashes.push(marker.toLowerCase());
			continue;
		}
		for (const match of marker.matchAll(CCR_HASH_RE)) {
			const hash = match[1];
			if (hash) hashes.push(hash.toLowerCase());
		}
	}
	return hashes;
}

function parseCompressionResponse(
	value: unknown,
	expectedCallId: string,
	expectedToolName: string,
): ParsedCompression | undefined {
	if (typeof value !== "object" || value === null || Array.isArray(value)) {
		return undefined;
	}
	if (!("messages" in value) || !Array.isArray(value.messages)) {
		return undefined;
	}
	if (value.messages.length !== 2) return undefined;

	const assistant = value.messages[0];
	const tool = value.messages[1];
	if (
		typeof assistant !== "object" ||
		assistant === null ||
		Array.isArray(assistant) ||
		!("role" in assistant) ||
		assistant.role !== "assistant" ||
		!("tool_calls" in assistant) ||
		!Array.isArray(assistant.tool_calls) ||
		assistant.tool_calls.length !== 1
	) {
		return undefined;
	}

	const toolCall = assistant.tool_calls[0];
	if (
		typeof toolCall !== "object" ||
		toolCall === null ||
		Array.isArray(toolCall) ||
		!("id" in toolCall) ||
		toolCall.id !== expectedCallId ||
		!("type" in toolCall) ||
		toolCall.type !== "function" ||
		!("function" in toolCall) ||
		typeof toolCall.function !== "object" ||
		toolCall.function === null ||
		Array.isArray(toolCall.function) ||
		!("name" in toolCall.function) ||
		toolCall.function.name !== expectedToolName ||
		!("arguments" in toolCall.function) ||
		toolCall.function.arguments !== "{}"
	) {
		return undefined;
	}
	if (
		typeof tool !== "object" ||
		tool === null ||
		Array.isArray(tool) ||
		!("role" in tool) ||
		tool.role !== "tool" ||
		!("tool_call_id" in tool) ||
		tool.tool_call_id !== expectedCallId ||
		!("content" in tool) ||
		typeof tool.content !== "string"
	) {
		return undefined;
	}

	const tokensBefore =
		"tokens_before" in value ? value.tokens_before : undefined;
	const tokensAfter = "tokens_after" in value ? value.tokens_after : undefined;
	const tokensSaved = "tokens_saved" in value ? value.tokens_saved : undefined;
	const compressionRatio =
		"compression_ratio" in value ? value.compression_ratio : undefined;
	const rawHashes = "ccr_hashes" in value ? value.ccr_hashes : undefined;
	if (
		!nonnegativeNumber(tokensBefore) ||
		!nonnegativeNumber(tokensAfter) ||
		!nonnegativeNumber(tokensSaved) ||
		!nonnegativeNumber(compressionRatio) ||
		!Array.isArray(rawHashes) ||
		!rawHashes.every(
			(hash): hash is string => typeof hash === "string" && hash.length > 0,
		)
	) {
		return undefined;
	}

	const markerHashes = discoverCcrHashes(tool.content);
	if (!markerHashes) return undefined;

	return {
		compressedText: tool.content,
		ccrHashes: [
			...new Set([...extractRetrievalHashes(rawHashes), ...markerHashes]),
		],
		tokensBefore,
		tokensAfter,
		tokensSaved,
		compressionRatio,
	};
}

export async function validateCompression(
	candidate: Candidate,
	response: unknown,
	retrieve: RetrieveOriginal,
): Promise<CompressionOutcome | undefined> {
	const expectedCallId = `call_${candidate.key.slice(0, 12)}`;
	const parsed = parseCompressionResponse(
		response,
		expectedCallId,
		candidate.toolName,
	);
	if (!parsed) return { status: "rejected", reason: "invalid_response" };
	if (
		parsed.tokensSaved === 0 &&
		parsed.compressedText === candidate.originalText
	) {
		return { status: "skipped", reason: "noop" };
	}
	if (parsed.compressedText.length === 0) {
		return { status: "rejected", reason: "empty_output" };
	}
	if (
		Buffer.byteLength(parsed.compressedText) >=
		Buffer.byteLength(candidate.originalText)
	) {
		return { status: "rejected", reason: "grew" };
	}
	if (parsed.tokensBefore === 0) {
		return { status: "rejected", reason: "zero_tokens" };
	}
	if (parsed.tokensSaved < 256) {
		return { status: "rejected", reason: "insufficient_savings" };
	}
	if (
		parsed.tokensAfter / parsed.tokensBefore > 0.9 ||
		parsed.compressionRatio > 0.9
	) {
		return { status: "rejected", reason: "ratio_too_high" };
	}
	if (parsed.ccrHashes.some((hash) => !parsed.compressedText.includes(hash))) {
		return { status: "rejected", reason: "hash_missing" };
	}

	const retrievals = new Map<string, string>();
	const originalBytes = Buffer.byteLength(candidate.originalText);
	const compressedBytes = Buffer.byteLength(parsed.compressedText);
	let sizeBytes = originalBytes + compressedBytes;
	for (const hash of parsed.ccrHashes) {
		let original: string | undefined;
		try {
			original = await retrieve(hash);
		} catch {
			return { status: "rejected", reason: "retrieval_failed" };
		}
		if (original === undefined || original.length === 0) {
			return { status: "rejected", reason: "retrieval_miss" };
		}
		if (!candidate.originalText.includes(original)) {
			return { status: "rejected", reason: "retrieval_mismatch" };
		}
		retrievals.set(hash, original);
		sizeBytes += Buffer.byteLength(original);
	}

	const now = Date.now();
	return {
		...candidate,
		compressedText: parsed.compressedText,
		ccrHashes: parsed.ccrHashes,
		retrievals,
		tokensBefore: parsed.tokensBefore,
		tokensAfter: parsed.tokensAfter,
		tokensSaved: parsed.tokensSaved,
		originalBytes,
		compressedBytes,
		sizeBytes,
		originalSha256: createHash("sha256")
			.update(candidate.originalText)
			.digest("hex"),
		policyVersion: POLICY_VERSION,
		createdAt: now,
		lastAccessedAt: now,
	};
}
