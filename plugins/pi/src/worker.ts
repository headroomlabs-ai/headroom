import { isPreparedEntry, validateCompression } from "./bridge.js";
import type { PreparedCache } from "./cache.js";
import type { Candidate, PreparedEntry } from "./types.js";

export type WorkerState = "warming" | "online" | "offline" | "stopped";

export interface WorkerClient {
	health(signal?: AbortSignal): Promise<boolean>;
	compress(
		candidate: Candidate,
		modelId: string,
		signal?: AbortSignal,
	): Promise<unknown>;
	retrieve(hash: string, signal?: AbortSignal): Promise<string | undefined>;
}

export interface WorkerOptions {
	client: WorkerClient;
	cache: PreparedCache;
	maxQueue?: number;
	concurrency?: number;
	onAccepted?: (entry: PreparedEntry) => void;
	onRejected?: (candidate: Candidate, reason: string) => void;
	onStateChange?: (state: WorkerState) => void;
}

export interface WorkerStats {
	candidates: number;
	queued: number;
	active: number;
	accepted: number;
	rejected: number;
	skipped: number;
	dropped: number;
	state: WorkerState;
	lastError: string | undefined;
}

interface QueueItem {
	candidate: Candidate;
	modelId: string;
}

export class CompressionWorker {
	readonly #client: WorkerClient;
	readonly #cache: PreparedCache;
	readonly #maxQueue: number;
	readonly #concurrency: number;
	readonly #onAccepted?: WorkerOptions["onAccepted"];
	readonly #onRejected?: WorkerOptions["onRejected"];
	readonly #onStateChange?: WorkerOptions["onStateChange"];
	readonly #queue: QueueItem[] = [];
	readonly #knownKeys = new Set<string>();
	readonly #active = new Map<string, AbortController>();
	#state: WorkerState = "warming";
	#candidates = 0;
	#accepted = 0;
	#rejected = 0;
	#skipped = 0;
	#dropped = 0;
	#lastError: string | undefined;
	#drainScheduled = false;
	#stopped = false;
	#healthFailures = 0;
	#nextHealthAt = 0;
	#healthTimer: ReturnType<typeof setTimeout> | undefined;
	#healthController: AbortController | undefined;
	#healthPromise: Promise<boolean> | undefined;

	constructor(options: WorkerOptions) {
		const maxQueue = options.maxQueue ?? 32;
		const concurrency = options.concurrency ?? 2;
		if (!Number.isInteger(maxQueue) || maxQueue <= 0) {
			throw new Error("maxQueue must be a positive integer");
		}
		if (!Number.isInteger(concurrency) || concurrency <= 0) {
			throw new Error("concurrency must be a positive integer");
		}

		this.#client = options.client;
		this.#cache = options.cache;
		this.#maxQueue = maxQueue;
		this.#concurrency = concurrency;
		this.#onAccepted = options.onAccepted;
		this.#onRejected = options.onRejected;
		this.#onStateChange = options.onStateChange;
	}

	enqueue(candidate: Candidate, modelId: string): boolean {
		if (
			this.#stopped ||
			this.#knownKeys.has(candidate.key) ||
			this.#cache.get(candidate.key) !== undefined
		) {
			return false;
		}
		this.#candidates += 1;
		if (this.#state === "offline" && Date.now() < this.#nextHealthAt) {
			this.#dropped += 1;
			return false;
		}

		if (this.#queue.length >= this.#maxQueue) {
			const dropped = this.#queue.shift();
			if (dropped) {
				this.#knownKeys.delete(dropped.candidate.key);
				this.#dropped += 1;
			}
		}

		this.#queue.push({ candidate, modelId });
		this.#knownKeys.add(candidate.key);
		this.#scheduleDrain();
		return true;
	}

	async checkHealth(force = false): Promise<boolean> {
		if (this.#stopped) return false;
		if (this.#healthPromise) return this.#healthPromise;
		if (!force && Date.now() < this.#nextHealthAt) return false;

		this.#setState("warming");
		const controller = new AbortController();
		this.#healthController = controller;
		const pending = this.#runHealthCheck(controller);
		this.#healthPromise = pending;
		try {
			return await pending;
		} finally {
			if (this.#healthPromise === pending) this.#healthPromise = undefined;
			if (this.#healthController === controller) {
				this.#healthController = undefined;
			}
		}
	}

	stop(): void {
		if (this.#stopped) return;
		this.#stopped = true;
		this.#queue.length = 0;
		this.#knownKeys.clear();
		for (const controller of this.#active.values()) controller.abort();
		this.#healthController?.abort();
		clearTimeout(this.#healthTimer);
		this.#healthTimer = undefined;
		this.#setState("stopped");
	}

	stats(): WorkerStats {
		return {
			candidates: this.#candidates,
			queued: this.#queue.length,
			active: this.#active.size,
			accepted: this.#accepted,
			rejected: this.#rejected,
			skipped: this.#skipped,
			dropped: this.#dropped,
			state: this.#state,
			lastError: this.#lastError,
		};
	}

	#scheduleDrain(): void {
		if (this.#drainScheduled || this.#stopped) return;
		this.#drainScheduled = true;
		queueMicrotask(() => {
			this.#drainScheduled = false;
			this.#drain();
		});
	}

	#drain(): void {
		if (this.#state === "offline" && Date.now() < this.#nextHealthAt) {
			return;
		}
		while (
			!this.#stopped &&
			this.#active.size < this.#concurrency &&
			this.#queue.length > 0
		) {
			const item = this.#queue.shift();
			if (!item) break;
			const controller = new AbortController();
			this.#active.set(item.candidate.key, controller);
			void this.#process(item, controller).finally(() => {
				this.#active.delete(item.candidate.key);
				this.#knownKeys.delete(item.candidate.key);
				this.#scheduleDrain();
			});
		}
	}

	async #process(item: QueueItem, controller: AbortController): Promise<void> {
		try {
			const response = await this.#client.compress(
				item.candidate,
				item.modelId,
				controller.signal,
			);
			const outcome = await validateCompression(
				item.candidate,
				response,
				(hash) => this.#client.retrieve(hash, controller.signal),
			);
			if (!isPreparedEntry(outcome)) {
				if (outcome?.status === "skipped") {
					this.#skipped += 1;
					return;
				}
				this.#rejected += 1;
				this.#lastError = outcome?.reason ?? "invalid_response";
				this.#notifyRejected(item.candidate, this.#lastError);
				return;
			}
			if (!this.#cache.set(outcome)) {
				this.#rejected += 1;
				this.#lastError = "cache_full";
				this.#notifyRejected(item.candidate, "cache_full");
				return;
			}

			this.#accepted += 1;
			this.#lastError = undefined;
			this.#markOnline();
			const entry = outcome;
			try {
				this.#onAccepted?.(entry);
			} catch {
				// Host status callbacks cannot affect compression.
			}
		} catch {
			if (this.#stopped && controller.signal.aborted) return;
			this.#rejected += 1;
			this.#lastError = "compression request failed";
			this.#notifyRejected(item.candidate, "request failed");
			this.#markOffline();
		}
	}

	async #runHealthCheck(controller: AbortController): Promise<boolean> {
		let healthy = false;
		try {
			healthy = await this.#client.health(controller.signal);
		} catch {
			healthy = false;
		}
		if (this.#stopped) return false;
		if (healthy) {
			this.#markOnline();
			return true;
		}

		this.#lastError = "health check failed";
		this.#markOffline();
		return false;
	}

	#markOnline(): void {
		this.#healthFailures = 0;
		this.#nextHealthAt = 0;
		clearTimeout(this.#healthTimer);
		this.#healthTimer = undefined;
		this.#setState("online");
		this.#scheduleDrain();
	}

	#markOffline(): void {
		this.#healthFailures += 1;
		const baseDelay = Math.min(
			1_000 * 2 ** Math.min(this.#healthFailures - 1, 5),
			30_000,
		);
		const delay = baseDelay + Math.floor(baseDelay * 0.1 * Math.random());
		this.#nextHealthAt = Date.now() + delay;
		this.#setState("offline");
		if (this.#healthTimer || this.#stopped) return;
		this.#healthTimer = setTimeout(() => {
			this.#healthTimer = undefined;
			void this.checkHealth(true);
		}, delay);
	}

	#setState(state: WorkerState): void {
		if (this.#state === state) return;
		this.#state = state;
		try {
			this.#onStateChange?.(state);
		} catch {
			// Host status callbacks cannot affect worker state.
		}
	}

	#notifyRejected(candidate: Candidate, reason: string): void {
		try {
			this.#onRejected?.(candidate, reason);
		} catch {
			// Host status callbacks cannot affect worker state.
		}
	}
}
