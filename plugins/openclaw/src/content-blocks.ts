/**
 * Content-block helpers shared by the AgentMessage <-> OpenAI conversion.
 *
 * Design: the Headroom proxy only understands string content. Anything that is
 * not text (images, tool_use/tool_result envelopes, unknown blocks) is replaced by
 * a short placeholder on the wire and restored from the locally held original
 * message after compression. Base64 payloads never leave the process.
 */

/** Minimal shape shared by all OpenClaw content blocks. */
export interface ContentBlock {
  type: string;
  [key: string]: unknown;
}

export interface TextBlock extends ContentBlock {
  type: "text";
  text: string;
}

export interface ImageBlock extends ContentBlock {
  type: "image";
  data: string;
  mimeType: string;
}

/** Rough token cost of one image for budget estimation (vision pricing, not base64 length). */
export const IMAGE_TOKEN_ESTIMATE = 1_500;

const PLACEHOLDER_OPEN = "[headroom-omitted ";
const PLACEHOLDER_LINE = /^\[headroom-omitted [^\]]*\]$/;

export function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

export function isTextBlock(block: unknown): block is TextBlock {
  return isRecord(block) && block.type === "text" && typeof block.text === "string";
}

export function isImageBlock(block: unknown): block is ImageBlock {
  return (
    isRecord(block) &&
    block.type === "image" &&
    typeof block.data === "string" &&
    typeof block.mimeType === "string"
  );
}

/** True for block types that must never be lossy-rewritten or dropped. */
export function blockIsProtectedPayload(block: unknown): boolean {
  if (!isRecord(block) || typeof block.type !== "string") return false;
  return (
    block.type === "image" ||
    block.type === "toolCall" ||
    block.type === "tool_use" ||
    block.type === "tool_result"
  );
}

/** True when a message carries multimodal or tool payloads that must not be lossy-rewritten. */
export function messageHasProtectedToolPayload(message: unknown): boolean {
  if (!isRecord(message)) return false;
  const role = message.role;
  if (
    role !== "toolResult" &&
    role !== "tool_result" &&
    role !== "assistant" &&
    role !== "user"
  ) {
    return false;
  }
  const content = message.content;
  if (!Array.isArray(content)) return false;
  return content.some((block) => blockIsProtectedPayload(block));
}

/** Short wire placeholder for a non-text block. Never contains the payload itself. */
export function blockToWirePlaceholder(block: ContentBlock): string {
  if (isImageBlock(block)) {
    const approxBytes = Math.floor((block.data.length * 3) / 4);
    return `${PLACEHOLDER_OPEN}image ${block.mimeType} ${approxBytes} bytes]`;
  }
  return `${PLACEHOLDER_OPEN}${block.type}]`;
}

/**
 * Serialize tool-result / user blocks to the string the proxy sees.
 * Text blocks are emitted verbatim; other blocks become placeholders.
 */
export function serializeBlocksForWire(blocks: ContentBlock[], joiner: string): string {
  return blocks
    .map((block) => (isTextBlock(block) ? block.text : blockToWirePlaceholder(block)))
    .join(joiner);
}

/**
 * Serialize assistant blocks to the string the proxy sees. Only text goes on
 * the wire: tool calls travel as `tool_calls`, thinking/unknown blocks stay local.
 */
export function serializeAssistantTextForWire(blocks: ContentBlock[]): string {
  return blocks
    .filter((block): block is TextBlock => isTextBlock(block))
    .map((block) => block.text)
    .join("");
}

/** Remove placeholder lines the proxy may have echoed back unchanged. */
export function stripWirePlaceholders(text: string): string {
  if (!text.includes(PLACEHOLDER_OPEN)) return text;
  return text
    .split("\n")
    .filter((line) => !PLACEHOLDER_LINE.test(line.trim()))
    .join("\n")
    .trim();
}

/**
 * Merge proxy-compressed text back into the original block array.
 *
 * `originalWireText` must be the exact string that was sent for these blocks so
 * that "unchanged" is detected reliably.
 *
 * - Unchanged text: the original blocks are returned untouched (structure preserved).
 * - Changed text: the first text block receives the compressed text, remaining text
 *   blocks are dropped (their content is already folded into the compressed string),
 *   and every non-text block keeps its position.
 */
export function mergeCompressedTextIntoBlocks(params: {
  originalBlocks: ContentBlock[];
  originalWireText: string;
  compressedText: string;
}): ContentBlock[] {
  const { originalBlocks, originalWireText, compressedText } = params;
  if (compressedText === originalWireText) return originalBlocks;

  const cleaned = stripWirePlaceholders(compressedText);
  const merged: ContentBlock[] = [];
  let textInserted = false;

  for (const block of originalBlocks) {
    if (isTextBlock(block)) {
      if (!textInserted) {
        if (cleaned.length > 0) merged.push({ type: "text", text: cleaned });
        textInserted = true;
      }
      continue;
    }
    merged.push(block);
  }

  if (!textInserted && cleaned.length > 0) {
    merged.unshift({ type: "text", text: cleaned });
  }

  return merged;
}
