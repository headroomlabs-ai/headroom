/**
 * Simulation API — dry-run compression to see what would happen.
 * Matches Python client.simulate() behavior.
 */

import { HeadroomClient } from "./client.js";
import type { HeadroomClientOptions } from "./types.js";
import type { SimulationResult } from "./types/models.js";
import type { HeadroomConfig } from "./types/config.js";
import { deepCamelCase, deepSnakeCase } from "./utils/case.js";
import { detectFormat, toOpenAI } from "./utils/format.js";

export interface SimulateOptions extends HeadroomClientOptions {
  model?: string;
  config?: HeadroomConfig;
  client?: HeadroomClient;
}

/**
 * Simulate compression without calling the LLM.
 * Shows what compression would do: token savings, transforms, waste signals.
 *
 * @example
 * ```typescript
 * const sim = await simulate(messages, { model: 'gpt-4o' });
 * console.log(`Would save ${sim.tokensSaved} tokens (${sim.estimatedSavings})`);
 * console.log('Transforms:', sim.transforms);
 * ```
 */
export async function simulate(
  messages: any[],
  options: SimulateOptions = {},
): Promise<SimulationResult> {
  const { client: providedClient, model, config, ...clientOptions } = options;

  const openaiMessages = toOpenAI(messages);
  const client = providedClient ?? new HeadroomClient(clientOptions);

  const body: Record<string, any> = {
    messages: openaiMessages,
    model: model ?? "gpt-4o",
    config: {
      default_mode: "simulate",
      generate_diff_artifact: true,
      ...(config ? deepSnakeCase(config) : {}),
    },
  };

  // Use the client's internal fetch to hit /v1/compress with simulation config
  const result = await client.compressRaw(body);
  return deepCamelCase<SimulationResult>(result);
}
