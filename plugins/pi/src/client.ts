import { buildCompressPayload } from "./bridge.js";
import type { Candidate } from "./types.js";

export interface HeadroomClientOptions {
  baseUrl: string;
  timeoutMs: number;
  fetchImpl?: typeof fetch;
}

export class HeadroomClient {
  readonly #baseUrl: string;
  readonly #timeoutMs: number;
  readonly #fetch: typeof fetch;

  constructor(options: HeadroomClientOptions) {
    this.#baseUrl = options.baseUrl.replace(/\/$/, "");
    this.#timeoutMs = options.timeoutMs;
    this.#fetch = options.fetchImpl ?? fetch;
  }

  #signal(external?: AbortSignal): AbortSignal {
    const timeout = AbortSignal.timeout(this.#timeoutMs);
    return external ? AbortSignal.any([external, timeout]) : timeout;
  }

  async health(signal?: AbortSignal): Promise<boolean> {
    const response = await this.#fetch(`${this.#baseUrl}/readyz`, {
      method: "GET",
      redirect: "error",
      signal: this.#signal(signal),
    });
    return response.ok;
  }

  async compress(
    candidate: Candidate,
    modelId: string,
    signal?: AbortSignal,
  ): Promise<unknown> {
    const response = await this.#fetch(`${this.#baseUrl}/v1/compress`, {
      method: "POST",
      redirect: "error",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(buildCompressPayload(candidate, modelId)),
      signal: this.#signal(signal),
    });
    if (!response.ok) {
      throw new Error(
        `Headroom compression failed with HTTP ${response.status}`,
      );
    }

    try {
      return await response.json();
    } catch {
      throw new Error("Headroom compression returned malformed JSON");
    }
  }

  async retrieve(
    hash: string,
    signal?: AbortSignal,
  ): Promise<string | undefined> {
    const response = await this.#fetch(`${this.#baseUrl}/v1/retrieve`, {
      method: "POST",
      redirect: "error",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ hash }),
      signal: this.#signal(signal),
    });
    if (response.status === 404) return undefined;
    if (!response.ok) {
      throw new Error(`Headroom retrieval failed with HTTP ${response.status}`);
    }

    let body: unknown;
    try {
      body = await response.json();
    } catch {
      throw new Error("Headroom retrieval returned malformed JSON");
    }
    if (
      typeof body !== "object" ||
      body === null ||
      Array.isArray(body) ||
      !("hash" in body) ||
      body.hash !== hash ||
      !("original_content" in body) ||
      typeof body.original_content !== "string"
    ) {
      throw new Error("Headroom retrieval returned an invalid response");
    }
    return body.original_content;
  }
}
