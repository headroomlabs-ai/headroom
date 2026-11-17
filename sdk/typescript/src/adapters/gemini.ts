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
 * Gemini accepts a bare string, or an array of strings/Parts, as shorthand for
 * the parts of a single user turn. Either way that is one Content, not a
 * conversation, so it has to be wrapped before format detection — which only
 * understands `{ role, parts }` objects and would throw on a primitive.
 */
function toContents(params: any): any[] {
  if (typeof params === "string" || Array.isArray(params)) {
    const parts = (Array.isArray(params) ? params : [params]).map((p: any) =>
      typeof p === "string" ? { text: p } : p,
    );
    return [{ role: "user", parts }];
  }
  return params?.contents ?? [params];
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

  const compressParams = async (params: any): Promise<any> => {
    const modelName = options.model ?? "gemini-2.0-flash";

    // compress() auto-detects Gemini format
    const result = await compress(toContents(params), {
      stack: "adapter_ts_gemini",
      ...options,
      model: modelName,
    });

    // Unchanged request (proxy down / nothing to compress): hand the original
    // params through untouched, as the Anthropic and Vercel adapters do.
    if (!result.compressed) return params;

    // A bare string or part array is not a request object: rebuild one, or the
    // SDK would read the returned Contents as the parts of a single turn.
    return Array.isArray(params) || typeof params === "string"
      ? { contents: result.messages }
      : { ...params, contents: result.messages };
  };

  const compressedGenerate = async (params: any) =>
    originalGenerate(await compressParams(params));

  const compressedStream = originalStream
    ? async (params: any) => originalStream(await compressParams(params))
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
