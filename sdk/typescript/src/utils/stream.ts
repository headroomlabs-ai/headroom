/**
 * SSE (Server-Sent Events) stream parser for proxy streaming responses.
 */

/**
 * Parse an SSE response body into an async generator of parsed JSON events.
 */
export async function* parseSSE<T = any>(
  response: Response,
): AsyncGenerator<T> {
  const reader = response.body!.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split("\n");
      buffer = lines.pop()!;

      for (const line of lines) {
        if (line.startsWith("data: ")) {
          const data = line.slice(6).trim();
          if (data === "[DONE]") return;
          try {
            yield JSON.parse(data) as T;
          } catch {
            // skip non-JSON data lines (e.g. Anthropic event metadata)
          }
        }
      }
    }
  } finally {
    reader.releaseLock();
  }
}

/**
 * Collect all chunks from an async iterable into an array.
 */
export async function collectStream<T>(stream: AsyncIterable<T>): Promise<T[]> {
  const chunks: T[] = [];
  for await (const chunk of stream) {
    chunks.push(chunk);
  }
  return chunks;
}
