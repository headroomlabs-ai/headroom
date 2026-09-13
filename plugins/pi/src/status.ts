import type { CacheStats } from "./cache.js";
import type { HeadroomConfig } from "./config.js";
import type { WorkerStats } from "./worker.js";

export interface TransformStats {
	substitutions: number;
	tokensBefore: number;
	tokensAfter: number;
	tokensSaved: number;
	bytesBefore: number;
	bytesAfter: number;
	bytesSaved: number;
}

export interface RuntimeSnapshot {
	enabled: boolean;
	modelId: string;
	tokensBefore: number;
	tokensAfter: number;
	tokensSaved: number;
	bytesSaved: number;
	retrievals: number;
	lastTransform: TransformStats;
	config: HeadroomConfig;
	worker: WorkerStats;
	cache: CacheStats;
	warnings: string[];
}

const NUMBER_FORMAT = new Intl.NumberFormat("en-US", {
	maximumFractionDigits: 1,
});

function formatCount(value: number): string {
	return NUMBER_FORMAT.format(value);
}

function formatEndpoint(baseUrl: string): string {
	try {
		const url = new URL(baseUrl);
		url.username = "";
		url.password = "";
		return url.toString().replace(/\/$/, "");
	} catch {
		return "<invalid>";
	}
}

function compactStatus(snapshot: RuntimeSnapshot): string {
	if (!snapshot.enabled) return "Headroom off";
	if (snapshot.worker.state !== "online") {
		return `Headroom ${snapshot.worker.state}`;
	}
	if (snapshot.tokensSaved <= 0) {
		return "Headroom online";
	}
	return `Headroom saved ${formatCount(snapshot.tokensSaved)} tokens this session`;
}

export function formatStatus(
	snapshot: RuntimeSnapshot,
	detailed = false,
): string {
	const compact = compactStatus(snapshot);
	if (!detailed) return compact;

	return [
		compact,
		`health ${snapshot.worker.state}`,
		`model ${snapshot.modelId}`,
		`endpoint ${formatEndpoint(snapshot.config.baseUrl)}`,
		`remote ${snapshot.config.allowRemote ? "allowed" : "blocked"} · hosts ${snapshot.config.remoteHosts.length > 0 ? snapshot.config.remoteHosts.join(", ") : "none"}`,
		`thresholds context ${formatCount(snapshot.config.minContextTokens)} tokens · result ${formatCount(snapshot.config.minResultChars)} chars · recent ${formatCount(snapshot.config.protectRecentToolResults)}`,
		`protected tools ${snapshot.config.protectedTools.join(", ")}`,
		`queue ${formatCount(snapshot.worker.queued)} queued · ${formatCount(snapshot.worker.active)} active`,
		`cache ${formatCount(snapshot.cache.bytes)}/${formatCount(snapshot.cache.maxBytes)} bytes`,
		`last transform ${formatCount(snapshot.lastTransform.substitutions)} substitutions · ${formatCount(snapshot.lastTransform.tokensSaved)} tokens saved · ${formatCount(snapshot.lastTransform.bytesSaved)} bytes saved`,
		`last error ${snapshot.worker.lastError ?? "none"}`,
		`config warnings ${snapshot.warnings.length > 0 ? snapshot.warnings.join(" | ") : "none"}`,
	].join("\n");
}

export function formatStats(snapshot: RuntimeSnapshot): string {
	return [
		compactStatus(snapshot),
		`prepared candidates ${formatCount(snapshot.worker.candidates)}`,
		`prepared accepted ${formatCount(snapshot.worker.accepted)}`,
		`prepared rejected ${formatCount(snapshot.worker.rejected)}`,
		`prepared skipped ${formatCount(snapshot.worker.skipped)}`,
		`prepared dropped ${formatCount(snapshot.worker.dropped)}`,
		`prepared bytes saved ${formatCount(snapshot.bytesSaved)}`,
		`prepared tokens saved ${formatCount(snapshot.tokensSaved)}`,
		`last transform substitutions ${formatCount(snapshot.lastTransform.substitutions)}`,
		`last transform bytes saved ${formatCount(snapshot.lastTransform.bytesSaved)}`,
		`last transform tokens saved ${formatCount(snapshot.lastTransform.tokensSaved)}`,
		`retrievals ${formatCount(snapshot.retrievals)}`,
	].join("\n");
}
