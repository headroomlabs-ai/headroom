/**
 * Gemini adapter — wraps Google Generative AI model with auto-compression.
 * Matches the pattern of openai.ts and anthropic.ts adapters.
 */

import { compress } from "../compress.js";
import type { CompressOptions } from "../types.js";

/* eslint-disable @typescript-eslint/no-explicit-any */

interface GeminiModelLike {
  generateContent: (params: any) => any;
  generateContentStream?: (params: any) => any;
  [key: string]: any;
}

/**
 * Wrap a Google Generative AI model to auto-compress before each request.
 *
 * Intercepts `model.generateContent()` and `model.generateContentStream()`.
 *
 * @example
 * ```typescript
 * import { withHeadroom } from 'headroom-ai/gemini';
 * import { GoogleGenerativeAI } from '@google/generative-ai';
 *
 * const genAI = new GoogleGenerativeAI(process.env.GEMINI_API_KEY!);
 * const model = withHeadroom(genAI.getGenerativeModel({ model: 'gemini-2.0-flash' }));
 *
 * const result = await model.generateContent({ contents: longConversation });
 * ```
 */
export function withHeadroom<T extends GeminiModelLike>(
  model: T,
  options: CompressOptions = {},
): T {
  const originalGenerate = model.generateContent.bind(model);
  const originalStream = model.generateContentStream?.bind(model);

  const compressedGenerate = async (params: any) => {
    const contents = params.contents ?? params;
    const modelName = options.model ?? "gemini-2.0-flash";

    // compress() auto-detects Gemini format
    const result = await compress(
      Array.isArray(contents) ? contents : [contents],
      { stack: "adapter_ts_gemini", ...options, model: modelName },
    );

    const newParams = Array.isArray(params)
      ? result.messages
      : { ...params, contents: result.messages };

    return originalGenerate(newParams);
  };

  const compressedStream = originalStream
    ? async (params: any) => {
        const contents = params.contents ?? params;
        const modelName = options.model ?? "gemini-2.0-flash";

        const result = await compress(
          Array.isArray(contents) ? contents : [contents],
          { stack: "adapter_ts_gemini", ...options, model: modelName },
        );

        const newParams = Array.isArray(params)
          ? result.messages
          : { ...params, contents: result.messages };

        return originalStream(newParams);
      }
    : undefined;

  return new Proxy(model, {
    get(target, prop) {
      if (prop === "generateContent") return compressedGenerate;
      if (prop === "generateContentStream" && compressedStream)
        return compressedStream;
      return (target as any)[prop];
    },
  }) as T;
}
