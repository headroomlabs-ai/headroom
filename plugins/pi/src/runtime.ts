import type { ToolResultMessage } from "@earendil-works/pi-ai";
import type {
	ContextEvent,
	ExtensionContext,
	ToolResultEvent,
} from "@earendil-works/pi-coding-agent";

import { PreparedCache } from "./cache.js";
import type { HeadroomConfig } from "./config.js";
import { candidateFromToolResult, transformContext } from "./policy.js";
import type { RuntimeSnapshot, TransformStats } from "./status.js";
import {
	loadSessionSavings,
	saveSessionSavings,
	type SessionSavings,
} from "./session-savings.js";
import { formatStatus } from "./status.js";
import {
	CompressionWorker,
	type WorkerClient,
	type WorkerState,
} from "./worker.js";

export interface HeadroomRuntimeOptions {
	config: HeadroomConfig;
	client: WorkerClient;
	cache?: PreparedCache;
	warnings?: string[];
}

interface ModelEligibility {
	eligible: boolean;
	modelId: string;
}

function emptyTransformStats(): TransformStats {
	return {
		substitutions: 0,
		tokensBefore: 0,
		tokensAfter: 0,
		tokensSaved: 0,
		bytesBefore: 0,
		bytesAfter: 0,
		bytesSaved: 0,
	};
}

export class HeadroomRuntime {
	readonly #config: HeadroomConfig;
	readonly #client: WorkerClient;
	readonly #cache: PreparedCache;
	readonly #worker: CompressionWorker;
	readonly #warnings: string[];
	#sessionEnabled: boolean;
	#context: ExtensionContext | undefined;
	#activity = false;
	#warningsShown = false;
	#tokensSaved = 0;
	#tokensBefore = 0;
	#tokensAfter = 0;
	#bytesSaved = 0;
	#retrievals = 0;
	#lastTransform = emptyTransformStats();
	#modelId = "unknown";
	#sessionFile: string | undefined;
	#restoreGeneration = 0;

	constructor(options: HeadroomRuntimeOptions) {
		this.#config = {
			...options.config,
			protectedTools: [...options.config.protectedTools],
		};
		this.#client = options.client;
		this.#cache =
			options.cache ?? new PreparedCache(options.config.maxCacheBytes);
		this.#warnings = [...(options.warnings ?? [])];
		this.#sessionEnabled = options.config.enabled;
		this.#worker = new CompressionWorker({
			client: options.client,
			cache: this.#cache,
			onAccepted: (entry) => {
				this.#tokensBefore += entry.tokensBefore;
				this.#tokensAfter += entry.tokensAfter;
				this.#bytesSaved += Math.max(
					0,
					entry.originalBytes - entry.compressedBytes,
				);
				this.#restoreGeneration += 1;
				this.#tokensSaved += entry.tokensSaved;
				this.#activity = true;
				this.#updateStatus();
				this.persistSessionSavings();
			},
			onRejected: () => {
				this.#activity = true;
				this.#updateStatus();
			},
			onStateChange: (state) => this.#onStateChange(state),
		});
	}

	start(ctx: ExtensionContext): void {
		this.#context = ctx;
		this.#sessionFile = ctx.sessionManager?.getSessionFile() ?? undefined;
		if (!this.#warningsShown && this.#warnings.length > 0 && ctx.hasUI) {
			this.#warningsShown = true;
			try {
				ctx.ui.notify(this.#warnings.join("\n"), "warning");
			} catch {
				// Host UI failures cannot affect context handling.
			}
		}
		void this.#restoreAndStart();
	}

	persistSessionSavings(): void {
		const sessionFile = this.#sessionFile;
		if (!sessionFile) return;
		void saveSessionSavings(sessionFile, this.#sessionSavings()).catch(
			() => undefined,
		);
	}

	restoreSessionSavings(savings: SessionSavings): void {
		this.#restoreGeneration += 1;
		this.#tokensSaved = savings.tokensSaved;
		this.#tokensBefore = savings.tokensBefore;
		this.#tokensAfter = savings.tokensAfter;
		this.#bytesSaved = savings.bytesSaved;
		this.#retrievals = savings.retrievals;
	}

	observeToolResult(event: ToolResultEvent, ctx: ExtensionContext): undefined {
		if (!this.#sessionEnabled) return undefined;
		this.#context = ctx;
		const model = this.#modelEligibility(ctx);
		this.#modelId = model.modelId;
		if (!model.eligible) return undefined;

		const message: ToolResultMessage = {
			role: "toolResult",
			toolCallId: event.toolCallId,
			toolName: event.toolName,
			content: event.content,
			details: event.details,
			isError: event.isError,
			timestamp: Date.now(),
		};
		const candidate = candidateFromToolResult(message, this.#config);
		if (candidate) {
			this.#activity = true;
			this.#worker.enqueue(candidate, model.modelId);
			this.#updateStatus();
		}
		return undefined;
	}

	transform(
		messages: ContextEvent["messages"],
		ctx: ExtensionContext,
	): ContextEvent["messages"] | undefined {
		const lastTransform = emptyTransformStats();
		this.#lastTransform = lastTransform;
		if (!this.#sessionEnabled) return undefined;
		this.#context = ctx;
		const model = this.#modelEligibility(ctx);
		this.#modelId = model.modelId;
		if (!model.eligible) return undefined;

		return transformContext(
			messages,
			this.#config,
			(key) => this.#cache.get(key),
			(candidate) => {
				this.#activity = true;
				this.#worker.enqueue(candidate, model.modelId);
			},
			(entry) => {
				lastTransform.substitutions += 1;
				lastTransform.tokensBefore += entry.tokensBefore;
				lastTransform.tokensAfter += entry.tokensAfter;
				lastTransform.tokensSaved += entry.tokensSaved;
				lastTransform.bytesBefore += entry.originalBytes;
				lastTransform.bytesAfter += entry.compressedBytes;
				lastTransform.bytesSaved += Math.max(
					0,
					entry.originalBytes - entry.compressedBytes,
				);
			},
		);
	}

	async retrieve(hash: string, signal?: AbortSignal): Promise<string> {
		this.#restoreGeneration += 1;
		this.#retrievals += 1;
		this.#activity = true;
		this.#updateStatus();
		this.persistSessionSavings();
		const local = this.#cache.getRetrieval(hash);
		if (local !== undefined) return local;

		try {
			const remote = await this.#client.retrieve(hash, signal);
			if (remote !== undefined) return remote;
		} catch {
			// Retrieval failure is an explicit miss, never a host failure.
		}
		return `Headroom retrieval miss for ${hash}. Rerun the originating tool.`;
	}

	setSessionEnabled(enabled: boolean): void {
		this.#sessionEnabled = enabled;
		this.#activity = true;
		this.#updateStatus();
	}

	async health(): Promise<boolean> {
		const healthy = await this.#worker.checkHealth(true);
		this.#activity = true;
		this.#updateStatus();
		return healthy;
	}

	stop(): void {
		this.persistSessionSavings();
		this.#worker.stop();
		try {
			this.#context?.ui.setStatus("headroom", undefined);
		} catch {
			// Host UI failures cannot affect shutdown.
		}
		this.#context = undefined;
	}

	#sessionSavings(): SessionSavings {
		return {
			tokensSaved: this.#tokensSaved,
			tokensBefore: this.#tokensBefore,
			tokensAfter: this.#tokensAfter,
			bytesSaved: this.#bytesSaved,
			retrievals: this.#retrievals,
		};
	}

	async #restoreAndStart(): Promise<void> {
		const generation = this.#restoreGeneration;
		const savings = await loadSessionSavings(this.#sessionFile);
		if (generation === this.#restoreGeneration) {
			this.restoreSessionSavings(savings);
			if (savings.tokensSaved > 0) {
				this.#activity = true;
				this.#updateStatus();
			}
		}
		try {
			await this.#worker.checkHealth();
		} catch {
			// Health failure is reflected in worker state, never thrown to the host.
		}
	}

	snapshot(): RuntimeSnapshot {
		return {
			enabled: this.#sessionEnabled,
			modelId: this.#modelId,
			tokensBefore: this.#tokensBefore,
			tokensAfter: this.#tokensAfter,
			bytesSaved: this.#bytesSaved,
			retrievals: this.#retrievals,
			lastTransform: { ...this.#lastTransform },
			config: {
				...this.#config,
				protectedTools: [...this.#config.protectedTools],
			},
			tokensSaved: this.#tokensSaved,
			worker: this.#worker.stats(),
			cache: this.#cache.stats(),
			warnings: [...this.#warnings],
		};
	}

	#modelEligibility(ctx: ExtensionContext): ModelEligibility {
		const model = ctx.model;
		if (!model) return { eligible: true, modelId: "unknown" };

		const modelId = model.provider
			? `${model.provider}/${model.id}`
			: model.id || "unknown";
		const contextWindow = model.contextWindow;
		return {
			eligible:
				!Number.isFinite(contextWindow) ||
				contextWindow >= this.#config.minContextTokens,
			modelId,
		};
	}

	#onStateChange(_state: WorkerState): void {
		this.#updateStatus();
	}

	#updateStatus(): void {
		if (!this.#activity || !this.#context) return;
		try {
			this.#context.ui.setStatus("headroom", formatStatus(this.snapshot()));
		} catch {
			// Host UI failures cannot affect context handling.
		}
	}
}
