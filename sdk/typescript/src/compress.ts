/**
 * Universal compress function with hooks support.
 * Accepts messages in any format: OpenAI, Anthropic, Vercel AI SDK, or Google Gemini.
 */

import { HeadroomClient } from "./client.js";
import type { CompressResult, CompressOptions } from "./types.js";
import { detectFormat, toOpenAI, fromOpenAI } from "./utils/format.js";
import type { CompressionHooks, CompressContext, CompressEvent } from "./hooks.js";
import { extractUserQuery, countTurns, extractToolCalls } from "./hooks.js";

/**
 * Compress an array of messages using the Headroom proxy.
 *
 * Accepts messages in any format: OpenAI, Anthropic, Vercel AI SDK, or Google Gemini.
 * Detects the format automatically, compresses via the proxy, and returns
 * compressed messages in the same format as the input.
 */
export async function compress(
  messages: any[],
  options: CompressOptions = {},
): Promise<CompressResult> {
  const {
    client: providedClient,
    model,
    tokenBudget,
    hooks,
    ...clientOptions
  } = options;

  // Build hook context
  const ctx: CompressContext = {
    model: model ?? "gpt-4o",
    userQuery: extractUserQuery(messages),
    turnNumber: countTurns(messages),
    toolCalls: extractToolCalls(messages),
    provider: "",
  };

  // 1. Pre-compress hook
  let processedMessages = messages;
  if (hooks) {
    processedMessages = await hooks.preCompress(messages, ctx);
  }

  // 2. Detect input format
  const inputFormat = detectFormat(processedMessages);

  // 3. Convert to OpenAI format (the proxy's lingua franca)
  const openaiMessages = toOpenAI(processedMessages);

  // 4. Compute biases
  let biases: Record<number, number> = {};
  if (hooks) {
    biases = await hooks.computeBiases(openaiMessages, ctx);
  }

  // 5. Compress via proxy
  const client = providedClient ?? new HeadroomClient(clientOptions);
  const result = await client.compress(openaiMessages, { model, tokenBudget });

  // 6. Convert compressed messages back to original format
  const outputMessages = fromOpenAI(result.messages, inputFormat);

  const finalResult: CompressResult = {
    ...result,
    messages: outputMessages,
  };

  // 7. Post-compress hook
  if (hooks) {
    const event: CompressEvent = {
      tokensBefore: result.tokensBefore,
      tokensAfter: result.tokensAfter,
      tokensSaved: result.tokensSaved,
      compressionRatio: result.compressionRatio,
      transformsApplied: result.transformsApplied,
      ccrHashes: result.ccrHashes,
      model: ctx.model,
      userQuery: ctx.userQuery,
      provider: ctx.provider,
    };
    await hooks.postCompress(event);
  }

  return finalResult;
}
