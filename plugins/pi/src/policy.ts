import { createHash } from "node:crypto";

import type { ToolResultMessage } from "@earendil-works/pi-ai";

import type { HeadroomConfig } from "./config.js";
import {
	POLICY_VERSION,
	type Candidate,
	type ContextMessage,
	type PreparedEntry,
} from "./types.js";

const CCR_MARKER_PREFIXES = [
	"Retrieve more: hash=",
	"Retrieve original: hash=",
	"<<ccr:",
	"<headroom:",
] as const;

const candidateIdentityCache = new WeakMap<ToolResultMessage, Candidate>();

export type PreparedLookup = (key: string) => PreparedEntry | undefined;
export type CandidateEnqueue = (candidate: Candidate) => void;
export type PreparedSubstitution = (entry: PreparedEntry) => void;

function hasTruncatedFlag(value: unknown): boolean {
	return (
		typeof value === "object" &&
		value !== null &&
		"truncated" in value &&
		value.truncated === true
	);
}

function nestedRtkCompaction(details: object): unknown {
	if (!("metadata" in details)) return undefined;
	const { metadata } = details;
	if (typeof metadata !== "object" || metadata === null) return undefined;
	return "rtkCompaction" in metadata ? metadata.rtkCompaction : undefined;
}

function hasPruningMetadata(details: unknown): boolean {
	if (typeof details !== "object" || details === null) return false;
	if (
		"prunedAt" in details ||
		("pruned" in details && details.pruned === true)
	) {
		return true;
	}

	return (
		hasTruncatedFlag("truncation" in details ? details.truncation : undefined) ||
		hasTruncatedFlag(
			"rtkCompaction" in details ? details.rtkCompaction : undefined,
		) ||
		hasTruncatedFlag(nestedRtkCompaction(details))
	);
}
function isToolResultMessage(
	message: ContextMessage,
): message is ToolResultMessage {
	return (
		message.role === "toolResult" &&
		"toolCallId" in message &&
		typeof message.toolCallId === "string" &&
		"toolName" in message &&
		typeof message.toolName === "string" &&
		"content" in message &&
		Array.isArray(message.content) &&
		"isError" in message &&
		typeof message.isError === "boolean"
	);
}

export function candidateFromToolResult(
	message: ToolResultMessage,
	config: Readonly<HeadroomConfig>,
): Candidate | undefined {
	const toolName = message.toolName.trim().toLowerCase();
	if (message.isError || config.protectedTools.includes(toolName)) {
		return undefined;
	}
	if (message.content.length !== 1 || message.content[0]?.type !== "text") {
		return undefined;
	}

	const originalText = message.content[0].text;
	if (
		originalText.length < config.minResultChars ||
		hasPruningMetadata(message.details)
	) {
		return undefined;
	}

	const cachedCandidate = candidateIdentityCache.get(message);
	if (
		cachedCandidate &&
		cachedCandidate.toolCallId === message.toolCallId &&
		cachedCandidate.toolName === toolName &&
		cachedCandidate.originalText === originalText
	) {
		return cachedCandidate;
	}
	if (CCR_MARKER_PREFIXES.some((prefix) => originalText.includes(prefix))) {
		return undefined;
	}

	const key = createHash("sha256")
		.update(POLICY_VERSION)
		.update("\0")
		.update(toolName)
		.update("\0")
		.update(originalText)
		.digest("hex");
	const candidate = {
		key,
		toolCallId: message.toolCallId,
		toolName,
		originalText,
	};
	candidateIdentityCache.set(message, candidate);

	return candidate;
}

export function protectedToolResultIndexes(
	messages: readonly ContextMessage[],
	count: number,
): Set<number> {
	const protectedIndexes = new Set<number>();
	for (let index = messages.length - 1; index >= 0; index -= 1) {
		if (messages[index]?.role !== "toolResult") continue;
		protectedIndexes.add(index);
		if (protectedIndexes.size === count) break;
	}
	return protectedIndexes;
}

export function transformContext<TMessage extends ContextMessage>(
	messages: readonly TMessage[],
	config: Readonly<HeadroomConfig>,
	lookup: PreparedLookup,
	enqueue: CandidateEnqueue,
	onSubstituted?: PreparedSubstitution,
): TMessage[] | undefined {
	const protectedIndexes = protectedToolResultIndexes(
		messages,
		config.protectRecentToolResults,
	);
	let transformed: TMessage[] | undefined;

	for (let index = 0; index < messages.length; index += 1) {
		const message = messages[index];
		if (
			!message ||
			!isToolResultMessage(message) ||
			protectedIndexes.has(index)
		) {
			continue;
		}

		const candidate = candidateFromToolResult(message, config);
		if (!candidate) continue;

		const entry = lookup(candidate.key);
		if (!entry) {
			enqueue(candidate);
			continue;
		}
		if (entry.originalText !== candidate.originalText) continue;

		transformed ??= [...messages];
		transformed[index] = {
			...message,
			content: [{ ...message.content[0], text: entry.compressedText }],
		} as unknown as TMessage;
		onSubstituted?.(entry);
	}

	return transformed;
}
