import { createRequire } from "node:module";
import { compress } from "../compress.js";
import type { CompressOptions, CompressResult } from "../types.js";
import { vercelToOpenAI, openAIToVercel } from "../utils/format.js";

/**
 * Minimal structural type for Vercel AI SDK language models.
 * Compatible with both LanguageModelV1 (@ai-sdk/provider <=1.x)
 * and LanguageModelV3 (@ai-sdk/provider >=2.x).
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
 * Vercel AI SDK LanguageModelV3Middleware that compresses messages
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

      // Convert Vercel format → OpenAI format
      const openaiMessages = vercelToOpenAI(prompt);

      // Compress via Headroom
      const result = await compress(openaiMessages, {
        stack: "adapter_ts_vercel_ai",
        ...options,
        model,
      });

      if (!result.compressed) return params;

      // Convert back to Vercel format
      const compressedPrompt = openAIToVercel(result.messages);

      return { ...params, prompt: compressedPrompt };
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
  const openaiMessages = vercelToOpenAI(messages);
  const result = await compress(openaiMessages, {
    stack: "adapter_ts_vercel_ai",
    ...options,
  });
  const vercelMessages = openAIToVercel(result.messages);

  return {
    ...result,
    messages: vercelMessages,
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
