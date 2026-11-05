import { createRequire } from "node:module";
import { compress } from "../compress.js";
import type { CompressOptions, CompressResult } from "../types.js";

/**
 * Minimal structural type for Vercel AI SDK language models.
 * Compatible with LanguageModelV1, LanguageModelV3, and LanguageModelV4.
 */
interface LanguageModel {
  readonly specificationVersion: string;
  readonly provider: string;
  readonly modelId: string;
  doGenerate: (...args: any[]) => any;
  doStream: (...args: any[]) => any;
  [key: string]: any;
}

/* eslint-disable @typescript-eslint/no-explicit-any */
type VercelMessage = any;

/**
 * Vercel AI SDK language model middleware that compresses messages
 * before they reach the LLM.
 *
 * @example
 * ```typescript
 * import { headroomMiddleware } from 'headroom-ai/vercel-ai';
 * import { wrapLanguageModel } from 'ai';
 * import { openai } from '@ai-sdk/openai';
 *
 * const model = wrapLanguageModel({
 *   model: openai('gpt-4o'),
 *   middleware: headroomMiddleware(),
 * });
 * ```
 */
export function headroomMiddleware(options: CompressOptions = {}) {
  return {
    transformParams: async ({
      params,
    }: {
      params: any;
      model: any;
      type: string;
    }) => {
      const prompt: VercelMessage[] = params.prompt;
      if (!prompt || prompt.length === 0) return params;

      const model = options.model ?? params.modelId ?? "gpt-4o";

      // compress() converts the prompt to OpenAI shape for the proxy and back. The format is
      // declared rather than detected: a text-only prompt is indistinguishable from OpenAI
      // messages, and the converted messages carry AI SDK parts (reasoning, providerOptions)
      // that detection would otherwise read as Vercel input again.
      const result = await compress(prompt, {
        stack: "adapter_ts_vercel_ai",
        ...options,
        model,
        format: "vercel",
      });

      if (!result.compressed) return params;

      return { ...params, prompt: result.messages };
    },
  };
}

/**
 * Standalone: compress Vercel AI SDK ModelMessage[] directly.
 * Returns compressed messages in Vercel format + compression stats.
 */
export async function compressVercelMessages(
  messages: VercelMessage[],
  options: CompressOptions = {},
): Promise<CompressResult & { messages: VercelMessage[] }> {
  const result = await compress(messages, {
    stack: "adapter_ts_vercel_ai",
    ...options,
    format: "vercel",
  });

  return {
    ...result,
    messages: result.messages,
  };
}

/**
 * Wrap a Vercel AI SDK language model with Headroom compression.
 * Convenience wrapper around `wrapLanguageModel` + `headroomMiddleware`.
 *
 * @example
 * ```typescript
 * import { withHeadroom } from 'headroom-ai/vercel-ai';
 * import { openai } from '@ai-sdk/openai';
 * import { generateText } from 'ai';
 *
 * const model = withHeadroom(openai('gpt-4o'));
 * const { text } = await generateText({ model, messages });
 * ```
 */
export function withHeadroom<T extends LanguageModel>(
  model: T,
  options: CompressOptions = {},
): T {
  let wrapLanguageModel: (opts: {
    model: T;
    middleware: any;
  }) => T;

  try {
    const require = createRequire(import.meta.url);
    const ai = require("ai");
    wrapLanguageModel = ai.wrapLanguageModel;
  } catch {
    throw new Error(
      'withHeadroom() requires the "ai" package. Install it with: npm install ai',
    );
  }

  return wrapLanguageModel({
    model,
    middleware: headroomMiddleware(options),
  });
}
