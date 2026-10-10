/**
 * Universal message format detection and conversion.
 *
 * Supports 4 formats:
 *   - OpenAI:   { role, content, tool_calls?, tool_call_id? }
 *   - Anthropic: { role, content (string | ContentBlock[]) } with tool_use/tool_result blocks
 *   - Vercel AI SDK: { role, content (string | Part[]) } with tool-call/tool-result parts
 *   - Google Gemini: { role, parts[] } with functionCall/functionResponse
 *
 * Detection is structural (unique keys/values per format), not heuristic.
 * Conversion always targets OpenAI format (the proxy's lingua franca).
 */

import type {
  OpenAIMessage,
  AssistantMessage,
  ToolCall,
} from "../types.js";

/* eslint-disable @typescript-eslint/no-explicit-any */

export type MessageFormat = "openai" | "anthropic" | "vercel" | "gemini";

/**
 * Detect which format an array of messages is in.
 *
 * Uses structural markers unique to each format:
 *   - Gemini: messages have `parts` field (not `content`)
 *   - Vercel: content parts use hyphenated types: `tool-call`, `tool-result`
 *   - Anthropic: content blocks use underscored types: `tool_use`, `tool_result`
 *   - OpenAI: assistant messages have `tool_calls` field; tool messages have `tool_call_id`
 */
export function detectFormat(messages: any[]): MessageFormat {
  for (const msg of messages) {
    // Gemini: uses `parts` instead of `content`
    if ("parts" in msg && !("content" in msg)) return "gemini";

    // Gemini: role "model" instead of "assistant"
    if (msg.role === "model") return "gemini";

    // OpenAI: tool_calls on assistant messages
    if (msg.tool_calls && msg.role === "assistant") return "openai";

    // OpenAI: tool role with tool_call_id
    if (msg.role === "tool" && "tool_call_id" in msg && typeof msg.content === "string") return "openai";

    // Check content blocks/parts for format-specific markers
    if (Array.isArray(msg.content)) {
      for (const part of msg.content) {
        // Vercel: hyphenated types
        if (part.type === "tool-call" || part.type === "tool-result") return "vercel";
        // Anthropic: underscored types
        if (part.type === "tool_use" || part.type === "tool_result") return "anthropic";
        // Anthropic: image with source.type
        if (part.type === "image" && part.source?.type) return "anthropic";
        // Vercel: `file` parts carry data/mediaType (OpenAI's `file` part carries a `file` object);
        // `image` parts carry `image` (Anthropic's carry `source`)
        if (part && typeof part === "object") {
          if (part.type === "file" && ("data" in part || "mediaType" in part)) return "vercel";
          if (part.type === "image" && "image" in part) return "vercel";
        }
      }
    }
  }

  // Default: OpenAI (simple {role, content: string} is valid OpenAI)
  return "openai";
}

// ============================================================
// OpenAI ↔ OpenAI (passthrough)
// ============================================================

// No conversion needed — this is the canonical format.

// ============================================================
// Anthropic → OpenAI
// ============================================================

// Anthropic image block `source` -> OpenAI image_url url string, or null if unconvertible.
function anthropicImageUrl(source: any): string | null {
  if (!source || typeof source !== "object") return null;
  if (source.type === "base64" && source.media_type && source.data) {
    return `data:${source.media_type};base64,${source.data}`;
  }
  if (source.type === "url" && typeof source.url === "string") return source.url;
  return null;
}

// OpenAI image_url url -> Anthropic image block. data: URIs become a base64 source; other urls a url source.
function openAIImageBlock(url: string): any {
  const m = /^data:([^;,]+);base64,(.*)$/s.exec(url);
  if (m) return { type: "image", source: { type: "base64", media_type: m[1], data: m[2] } };
  return { type: "image", source: { type: "url", url } };
}

// Anthropic media block -> OpenAI image_url object, or null if unconvertible: image blocks via
// anthropicImageUrl; document blocks (PDF) as a data: URI, or a url with the MIME type carried on
// the private `mime_type` key.
function anthropicMediaImageUrl(block: any): { url: string; mime_type?: string } | null {
  if (block.type === "image") {
    const url = anthropicImageUrl(block.source);
    return url ? { url } : null;
  }
  if (block.type === "document" && block.source && typeof block.source === "object") {
    const src = block.source;
    if (src.type === "base64" && src.media_type && src.data) {
      return { url: `data:${src.media_type};base64,${src.data}` };
    }
    if (src.type === "url" && typeof src.url === "string") return { url: src.url, mime_type: "application/pdf" };
  }
  return null;
}

// OpenAI image_url -> Anthropic media block, or null when Anthropic has no block for that media:
// image/* (or unknown) -> image block; application/pdf -> document block (base64 or url source).
function anthropicMediaBlock(imageUrl: any): any {
  const url: string = imageUrl.url;
  const m = /^data:([^;,]*);base64,(.*)$/s.exec(url);
  const mediaType: string | undefined =
    (m && m[1]) || (typeof imageUrl.mime_type === "string" ? imageUrl.mime_type : undefined);
  if (mediaType === "application/pdf") {
    return m
      ? { type: "document", source: { type: "base64", media_type: "application/pdf", data: m[2] } }
      : { type: "document", source: { type: "url", url } };
  }
  if (mediaType === undefined || mediaType.startsWith("image/")) return openAIImageBlock(url);
  return null;
}

export function anthropicToOpenAI(messages: any[]): OpenAIMessage[] {
  const result: OpenAIMessage[] = [];

  for (const msg of messages) {
    if (msg.role === "user") {
      if (typeof msg.content === "string") {
        result.push({ role: "user", content: msg.content });
        continue;
      }
      if (Array.isArray(msg.content)) {
        const toolResults = msg.content.filter((b: any) => b.type === "tool_result");
        const hasImages = msg.content.some((b: any) => b.type === "image");

        if (hasImages) {
          const parts = msg.content
            .filter((b: any) => b.type === "text" || b.type === "image")
            .map((b: any) => {
              if (b.type === "text") return { type: "text" as const, text: b.text };
              const url = anthropicImageUrl(b.source);
              return url ? { type: "image_url" as const, image_url: { url } } : null;
            })
            .filter((p: any): p is NonNullable<typeof p> => p !== null);

          if (parts.length > 0) {
            result.push({ role: "user", content: parts });
          }
        } else {
          const textBlocks = msg.content.filter((b: any) => b.type === "text");
          if (textBlocks.length > 0) {
            result.push({
              role: "user",
              content: textBlocks.map((b: any) => b.text).join("\n"),
            });
          }
        }
        for (const tr of toolResults) {
          const content = typeof tr.content === "string"
            ? tr.content
            : Array.isArray(tr.content)
              ? tr.content.map((b: any) => b.text ?? JSON.stringify(b)).join("\n")
              : JSON.stringify(tr.content);
          result.push({
            role: "tool",
            content,
            tool_call_id: tr.tool_use_id,
          });
        }
      }
      continue;
    }

    if (msg.role === "assistant") {
      if (typeof msg.content === "string") {
        result.push({ role: "assistant", content: msg.content });
        continue;
      }
      if (Array.isArray(msg.content)) {
        const textBlocks = msg.content.filter((b: any) => b.type === "text");
        const toolUseBlocks = msg.content.filter((b: any) => b.type === "tool_use");
        const hasMedia = msg.content.some((b: any) => b.type === "image" || b.type === "document");

        let content: AssistantMessage["content"];
        if (hasMedia) {
          // Keep generated media as ordered parts (see AssistantMessage.content).
          content = msg.content
            .filter((b: any) => b.type === "text" || b.type === "image" || b.type === "document")
            .map((b: any) => {
              if (b.type === "text") return { type: "text" as const, text: b.text };
              const imageUrl = anthropicMediaImageUrl(b);
              return imageUrl ? { type: "image_url" as const, image_url: imageUrl } : null;
            })
            .filter((p: any): p is NonNullable<typeof p> => p !== null);
        } else {
          content = textBlocks.length > 0 ? textBlocks.map((b: any) => b.text).join("\n") : null;
        }

        const openaiMsg: AssistantMessage = { role: "assistant", content };
        if (toolUseBlocks.length > 0) {
          openaiMsg.tool_calls = toolUseBlocks.map((b: any): ToolCall => ({
            id: b.id,
            type: "function",
            function: {
              name: b.name,
              arguments: typeof b.input === "string" ? b.input : JSON.stringify(b.input),
            },
          }));
        }
        result.push(openaiMsg);
      }
      continue;
    }
  }

  return result;
}

export function openAIToAnthropic(messages: OpenAIMessage[]): any[] {
  const result: any[] = [];

  for (const msg of messages) {
    if (msg.role === "system") {
      // Anthropic system is top-level, but if it appears in messages, convert to user
      result.push({ role: "user", content: msg.content });
      continue;
    }

    if (msg.role === "user") {
      if (typeof msg.content === "string") {
        result.push({ role: "user", content: msg.content });
      } else if (Array.isArray(msg.content)) {
        result.push({
          role: "user",
          content: msg.content.map((p) => {
            if (p.type === "text") return { type: "text", text: p.text };
            if (p.type === "image_url") return anthropicMediaBlock(p.image_url) ?? { type: "text", text: "" };
            return { type: "text", text: "" };
          }),
        });
      }
      continue;
    }

    if (msg.role === "assistant") {
      const blocks: any[] = [];
      if (typeof msg.content === "string") {
        if (msg.content) blocks.push({ type: "text", text: msg.content });
      } else if (Array.isArray(msg.content)) {
        for (const p of msg.content) {
          if (p.type === "text") blocks.push({ type: "text", text: p.text });
          else if (p.type === "image_url") {
            const block = anthropicMediaBlock(p.image_url);
            if (block) blocks.push(block);
          }
        }
      }
      if (msg.tool_calls) {
        for (const tc of msg.tool_calls) {
          blocks.push({
            type: "tool_use",
            id: tc.id,
            name: tc.function.name,
            input: JSON.parse(tc.function.arguments),
          });
        }
      }
      result.push({
        role: "assistant",
        content: blocks.length === 1 && blocks[0].type === "text" ? blocks[0].text : blocks,
      });
      continue;
    }

    if (msg.role === "tool") {
      result.push({
        role: "user",
        content: [
          { type: "tool_result", tool_use_id: msg.tool_call_id, content: msg.content },
        ],
      });
      continue;
    }
  }

  return result;
}

// ============================================================
// Vercel AI SDK → OpenAI
// ============================================================

function bytesToBase64(bytes: Uint8Array): string {
  if (typeof Buffer !== "undefined") return Buffer.from(bytes).toString("base64");
  let binary = "";
  for (let i = 0; i < bytes.length; i += 0x8000) {
    binary += String.fromCharCode(...bytes.subarray(i, i + 0x8000));
  }
  return btoa(binary);
}

function parseUrl(value: string): URL | null {
  try {
    return new URL(value);
  } catch {
    return null;
  }
}

type VercelMediaImageUrl = { url: string; mime_type?: string; part?: "file"; filename?: string };

// Vercel `image` / `file` part -> OpenAI image_url object, or null if the data is unconvertible
// (e.g. a provider file reference). Bytes and base64 travel as a data: URI, which carries the
// media type; URLs stay URLs with the media type on a private `mime_type` key. `part: "file"`
// and `filename` are carried so the reverse conversion rebuilds the right part. The proxy
// passes non-text parts through unchanged, so the private keys survive /v1/compress.
function vercelMediaImageUrl(p: any): VercelMediaImageUrl | null {
  const data = p.type === "file" ? p.data : p.image;
  const mediaType: string | undefined = typeof p.mediaType === "string" ? p.mediaType : p.mimeType;
  let url: string | null = null;
  let carryType = false;
  if (data instanceof Uint8Array || data instanceof ArrayBuffer) {
    const bytes = data instanceof Uint8Array ? data : new Uint8Array(data);
    url = `data:${mediaType ?? ""};base64,${bytesToBase64(bytes)}`;
  } else if (data instanceof URL) {
    url = data.toString();
    carryType = true;
  } else if (typeof data === "string") {
    if (data.startsWith("data:")) url = data;
    else if (parseUrl(data)) {
      url = data;
      carryType = true;
    } else url = `data:${mediaType ?? ""};base64,${data}`;
  }
  if (url === null) return null;
  const out: VercelMediaImageUrl = { url };
  if (carryType && mediaType) out.mime_type = mediaType;
  if (p.type === "file") out.part = "file";
  if (typeof p.filename === "string") out.filename = p.filename;
  return out;
}

// OpenAI image_url -> Vercel `image` or `file` part. A data: URI yields base64 data plus its
// media type; an http(s) url yields a URL object; anything else stays a string. Never throws.
function vercelMediaPart(imageUrl: any): any {
  const url: string = imageUrl.url;
  const m = /^data:([^;,]*);base64,(.*)$/s.exec(url);
  let data: any;
  let mediaType: string | undefined = typeof imageUrl.mime_type === "string" ? imageUrl.mime_type : undefined;
  if (m) {
    data = m[2];
    if (m[1]) mediaType = m[1];
  } else {
    data = parseUrl(url) ?? url;
  }
  const isFile = imageUrl.part === "file" || (mediaType !== undefined && !mediaType.startsWith("image/"));
  if (isFile) {
    const part: any = { type: "file", mediaType: mediaType ?? "application/octet-stream", data };
    if (typeof imageUrl.filename === "string") part.filename = imageUrl.filename;
    return part;
  }
  const part: any = { type: "image", image: data };
  if (mediaType) part.mediaType = mediaType;
  return part;
}

export function vercelToOpenAI(messages: any[]): OpenAIMessage[] {
  const result: OpenAIMessage[] = [];

  for (const msg of messages) {
    if (msg.role === "system") {
      result.push({ role: "system", content: typeof msg.content === "string" ? msg.content : String(msg.content) });
      continue;
    }

    if (msg.role === "user") {
      if (typeof msg.content === "string") {
        result.push({ role: "user", content: msg.content });
        continue;
      }
      const parts = Array.isArray(msg.content) ? msg.content : [];
      const textParts = parts.filter((p: any) => p.type === "text");
      const mediaParts = parts.filter((p: any) => p.type === "image" || p.type === "file");

      if (mediaParts.length === 0 && textParts.length > 0) {
        result.push({ role: "user", content: textParts.map((p: any) => p.text).join("") });
      } else {
        const openaiParts = parts
          .filter((p: any) => p.type === "text" || p.type === "image" || p.type === "file")
          .map((p: any) => {
            if (p.type === "text") return { type: "text" as const, text: p.text };
            const imageUrl = vercelMediaImageUrl(p);
            return imageUrl ? { type: "image_url" as const, image_url: imageUrl } : null;
          })
          .filter((p: any): p is NonNullable<typeof p> => p !== null);
        result.push({ role: "user", content: openaiParts });
      }
      continue;
    }

    if (msg.role === "assistant") {
      if (typeof msg.content === "string") {
        result.push({ role: "assistant", content: msg.content });
        continue;
      }
      const parts = Array.isArray(msg.content) ? msg.content : [];
      const textParts = parts.filter((p: any) => p.type === "text");
      const toolCallParts = parts.filter((p: any) => p.type === "tool-call");
      const hasFiles = parts.some((p: any) => p.type === "file");

      let content: AssistantMessage["content"];
      if (hasFiles) {
        // AI SDK assistant content may carry generated files; keep them as ordered parts.
        content = parts
          .filter((p: any) => p.type === "text" || p.type === "file")
          .map((p: any) => {
            if (p.type === "text") return { type: "text" as const, text: p.text };
            const imageUrl = vercelMediaImageUrl(p);
            return imageUrl ? { type: "image_url" as const, image_url: imageUrl } : null;
          })
          .filter((p: any): p is NonNullable<typeof p> => p !== null);
      } else {
        content = textParts.length > 0 ? textParts.map((p: any) => p.text).join("") : null;
      }
      const openaiMsg: AssistantMessage = { role: "assistant", content };

      if (toolCallParts.length > 0) {
        openaiMsg.tool_calls = toolCallParts.map((p: any): ToolCall => ({
          id: p.toolCallId,
          type: "function",
          // AI SDK v6 uses `input`, earlier versions used `args`
          function: { name: p.toolName, arguments: JSON.stringify(p.input ?? p.args) },
        }));
      }
      result.push(openaiMsg);
      continue;
    }

    if (msg.role === "tool") {
      const parts = Array.isArray(msg.content) ? msg.content : [];
      for (const part of parts) {
        if (part.type === "tool-result") {
          // AI SDK v6 uses `output: { type, value }`, earlier versions used `result`
          let contentStr: string;
          if (part.output !== undefined) {
            // v6 format: output is { type: 'json', value } or { type: 'text', value }
            const val = part.output?.value ?? part.output;
            contentStr = typeof val === "string" ? val : JSON.stringify(val);
          } else if (part.result !== undefined) {
            // Legacy format: result field directly
            contentStr = typeof part.result === "string" ? part.result : JSON.stringify(part.result);
          } else {
            contentStr = "";
          }
          result.push({
            role: "tool",
            content: contentStr,
            tool_call_id: part.toolCallId,
          });
        }
      }
      continue;
    }
  }

  return result;
}

export function openAIToVercel(messages: OpenAIMessage[]): any[] {
  const result: any[] = [];
  // tool_call id -> function name, so tool-result parts carry the name the AI SDK expects.
  const toolNames = new Map<string, string>();

  for (const msg of messages) {
    if (msg.role === "system") {
      result.push({ role: "system", content: msg.content });
      continue;
    }

    if (msg.role === "user") {
      if (typeof msg.content === "string") {
        result.push({ role: "user", content: [{ type: "text", text: msg.content }] });
      } else if (Array.isArray(msg.content)) {
        const parts = msg.content.map((p) => {
          if (p.type === "text") return { type: "text", text: p.text };
          if (p.type === "image_url") return vercelMediaPart(p.image_url);
          return { type: "text", text: "" };
        });
        result.push({ role: "user", content: parts });
      }
      continue;
    }

    if (msg.role === "assistant") {
      const parts: any[] = [];
      if (typeof msg.content === "string") {
        if (msg.content) parts.push({ type: "text", text: msg.content });
      } else if (Array.isArray(msg.content)) {
        for (const p of msg.content) {
          if (p.type === "text") parts.push({ type: "text", text: p.text });
          else if (p.type === "image_url") parts.push(vercelMediaPart(p.image_url));
        }
      }
      if (msg.tool_calls) {
        for (const tc of msg.tool_calls) {
          toolNames.set(tc.id, tc.function.name);
          let input: any;
          try { input = JSON.parse(tc.function.arguments); } catch { input = tc.function.arguments ?? {}; }
          parts.push({
            type: "tool-call",
            toolCallId: tc.id,
            toolName: tc.function.name,
            input, // AI SDK v6 uses `input`, not `args`
          });
        }
      }
      result.push({ role: "assistant", content: parts });
      continue;
    }

    if (msg.role === "tool") {
      let parsed: any;
      try { parsed = JSON.parse(msg.content); } catch { parsed = msg.content; }
      // AI SDK v6 requires output: { type: 'json' | 'text', value }
      const output = typeof parsed === "string"
        ? { type: "text" as const, value: parsed }
        : { type: "json" as const, value: parsed };
      result.push({
        role: "tool",
        content: [{
          type: "tool-result",
          toolCallId: msg.tool_call_id,
          toolName: toolNames.get(msg.tool_call_id) ?? "unknown",
          output,
        }],
      });
      continue;
    }
  }

  return result;
}

// ============================================================
// Google Gemini → OpenAI
// ============================================================

// Gemini inlineData / fileData part -> OpenAI image_url object, or null if unconvertible.
// inlineData travels as a data: URI (MIME type included). fileData keeps its MIME type on a
// private `mime_type` key so the reverse conversion can rebuild the complete
// { mimeType, fileUri } that @google/generative-ai requires; the proxy passes non-text
// parts through unchanged, so the key survives /v1/compress.
function geminiMediaImageUrl(part: any): { url: string; mime_type?: string } | null {
  const blob = part.inlineData;
  if (blob && typeof blob === "object" && blob.mimeType && blob.data) {
    return { url: `data:${blob.mimeType};base64,${blob.data}` };
  }
  const file = part.fileData;
  if (file && typeof file === "object" && typeof file.fileUri === "string") {
    return typeof file.mimeType === "string"
      ? { url: file.fileUri, mime_type: file.mimeType }
      : { url: file.fileUri };
  }
  return null;
}

const MIME_BY_EXTENSION: Record<string, string> = {
  png: "image/png",
  jpg: "image/jpeg",
  jpeg: "image/jpeg",
  gif: "image/gif",
  webp: "image/webp",
  pdf: "application/pdf",
};

// OpenAI image_url -> Gemini part. data: URIs become inlineData. Other urls become fileData
// with the carried MIME type, else one inferred from a common extension, else fileUri only.
function geminiMediaPart(imageUrl: { url: string; mime_type?: string }): any {
  const m = /^data:([^;,]+);base64,(.*)$/s.exec(imageUrl.url);
  if (m) return { inlineData: { mimeType: m[1], data: m[2] } };
  const ext = /\.([a-z0-9]+)(?:[?#]|$)/i.exec(imageUrl.url)?.[1]?.toLowerCase();
  const mimeType = imageUrl.mime_type ?? (ext ? MIME_BY_EXTENSION[ext] : undefined);
  return { fileData: mimeType ? { mimeType, fileUri: imageUrl.url } : { fileUri: imageUrl.url } };
}

export function geminiToOpenAI(messages: any[]): OpenAIMessage[] {
  const result: OpenAIMessage[] = [];

  for (const msg of messages) {
    const role = msg.role === "model" ? "assistant" : "user";
    const parts: any[] = msg.parts ?? [];

    if (role === "user") {
      // Check for functionResponse parts
      const funcResponses = parts.filter((p: any) => p.functionResponse);
      const hasMedia = parts.some((p: any) => p.inlineData || p.fileData);

      if (hasMedia) {
        const content = parts
          .filter((p: any) => p.text !== undefined || p.inlineData || p.fileData)
          .map((p: any) => {
            if (p.text !== undefined) return { type: "text" as const, text: p.text };
            const imageUrl = geminiMediaImageUrl(p);
            return imageUrl ? { type: "image_url" as const, image_url: imageUrl } : null;
          })
          .filter((p: any): p is NonNullable<typeof p> => p !== null);

        if (content.length > 0) {
          result.push({ role: "user", content });
        }
      } else {
        const textParts = parts.filter((p: any) => p.text !== undefined);
        if (textParts.length > 0) {
          result.push({ role: "user", content: textParts.map((p: any) => p.text).join("\n") });
        }
      }
      for (const fr of funcResponses) {
        result.push({
          role: "tool",
          content: JSON.stringify(fr.functionResponse.response),
          tool_call_id: `gemini_${fr.functionResponse.name}`,
        });
      }
      continue;
    }

    if (role === "assistant") {
      const textParts = parts.filter((p: any) => p.text !== undefined);
      const funcCalls = parts.filter((p: any) => p.functionCall);
      const hasMedia = parts.some((p: any) => p.inlineData || p.fileData);

      let content: AssistantMessage["content"];
      if (hasMedia) {
        // Keep generated media (e.g. image output) as ordered parts.
        content = parts
          .filter((p: any) => p.text !== undefined || p.inlineData || p.fileData)
          .map((p: any) => {
            if (p.text !== undefined) return { type: "text" as const, text: p.text };
            const imageUrl = geminiMediaImageUrl(p);
            return imageUrl ? { type: "image_url" as const, image_url: imageUrl } : null;
          })
          .filter((p: any): p is NonNullable<typeof p> => p !== null);
      } else {
        content = textParts.length > 0 ? textParts.map((p: any) => p.text).join("\n") : null;
      }
      const openaiMsg: AssistantMessage = { role: "assistant", content };

      if (funcCalls.length > 0) {
        openaiMsg.tool_calls = funcCalls.map((p: any): ToolCall => ({
          id: `gemini_${p.functionCall.name}`,
          type: "function",
          function: {
            name: p.functionCall.name,
            arguments: JSON.stringify(p.functionCall.args),
          },
        }));
      }
      result.push(openaiMsg);
      continue;
    }
  }

  return result;
}

export function openAIToGemini(messages: OpenAIMessage[]): any[] {
  const result: any[] = [];

  for (const msg of messages) {
    if (msg.role === "system") {
      // Gemini system is top-level; if in messages, convert to user
      result.push({ role: "user", parts: [{ text: msg.content }] });
      continue;
    }

    if (msg.role === "user") {
      if (Array.isArray(msg.content) && msg.content.some((p) => p.type === "image_url")) {
        const geminiParts = msg.content
          .map((p) => {
            if (p.type === "text") return { text: p.text };
            if (p.type === "image_url") return geminiMediaPart(p.image_url);
            return null;
          })
          .filter((p) => p !== null);
        result.push({ role: "user", parts: geminiParts });
        continue;
      }
      const text = typeof msg.content === "string"
        ? msg.content
        : (msg.content ?? []).filter((p) => p.type === "text").map((p) => (p as any).text).join("\n");
      result.push({ role: "user", parts: [{ text }] });
      continue;
    }

    if (msg.role === "assistant") {
      const parts: any[] = [];
      if (typeof msg.content === "string") {
        if (msg.content) parts.push({ text: msg.content });
      } else if (Array.isArray(msg.content)) {
        for (const p of msg.content) {
          if (p.type === "text") parts.push({ text: p.text });
          else if (p.type === "image_url") parts.push(geminiMediaPart(p.image_url));
        }
      }
      if (msg.tool_calls) {
        for (const tc of msg.tool_calls) {
          parts.push({
            functionCall: { name: tc.function.name, args: JSON.parse(tc.function.arguments) },
          });
        }
      }
      result.push({ role: "model", parts });
      continue;
    }

    if (msg.role === "tool") {
      let response: any;
      try { response = JSON.parse(msg.content); } catch { response = { result: msg.content }; }
      result.push({
        role: "user",
        parts: [{ functionResponse: { name: msg.tool_call_id?.replace("gemini_", "") ?? "unknown", response } }],
      });
      continue;
    }
  }

  return result;
}

// ============================================================
// Universal: any format → OpenAI, and OpenAI → original format
// ============================================================

export function toOpenAI(messages: any[]): OpenAIMessage[] {
  const format = detectFormat(messages);
  switch (format) {
    case "openai": return messages as OpenAIMessage[];
    case "anthropic": return anthropicToOpenAI(messages);
    case "vercel": return vercelToOpenAI(messages);
    case "gemini": return geminiToOpenAI(messages);
  }
}

export function fromOpenAI(messages: OpenAIMessage[], targetFormat: MessageFormat): any[] {
  switch (targetFormat) {
    case "openai": return messages;
    case "anthropic": return openAIToAnthropic(messages);
    case "vercel": return openAIToVercel(messages);
    case "gemini": return openAIToGemini(messages);
  }
}
