/**
 * Tests for the SSE stream parser.
 */
import { describe, it, expect } from "vitest";
import { parseSSE, collectStream } from "../../src/utils/stream.js";

/** Build a Response whose body streams the given raw string chunks verbatim. */
function sseResponse(chunks: string[]): Response {
  const encoder = new TextEncoder();
  const stream = new ReadableStream<Uint8Array>({
    start(controller) {
      for (const chunk of chunks) controller.enqueue(encoder.encode(chunk));
      controller.close();
    },
  });
  return new Response(stream);
}

describe("parseSSE", () => {
  it("yields the final event when the stream has no trailing newline", async () => {
    // Regression: the last `data:` line was held back in the buffer and dropped
    // when the read loop exited on `done`.
    const events = await collectStream(
      parseSSE(sseResponse(['data: {"a":1}\n', 'data: {"b":2}'])),
    );
    expect(events).toEqual([{ a: 1 }, { b: 2 }]);
  });

  it("parses multiple newline-terminated events", async () => {
    const events = await collectStream(
      parseSSE(sseResponse(['data: {"a":1}\n', 'data: {"b":2}\n'])),
    );
    expect(events).toEqual([{ a: 1 }, { b: 2 }]);
  });

  it("reassembles an event split across chunk boundaries", async () => {
    const events = await collectStream(
      parseSSE(sseResponse(['data: {"a"', ":1}\n"])),
    );
    expect(events).toEqual([{ a: 1 }]);
  });

  it("stops at the [DONE] sentinel and ignores anything after it", async () => {
    const events = await collectStream(
      parseSSE(sseResponse(['data: {"a":1}\n', "data: [DONE]\n", 'data: {"b":2}\n'])),
    );
    expect(events).toEqual([{ a: 1 }]);
  });

  it("skips non-JSON data lines", async () => {
    const events = await collectStream(
      parseSSE(sseResponse(["event: ping\n", "data: not-json\n", 'data: {"a":1}\n'])),
    );
    expect(events).toEqual([{ a: 1 }]);
  });

  it("decodes a multi-byte character split across chunk boundaries", async () => {
    // "é" is 0xC3 0xA9; split the two bytes across reads so the flush matters.
    const encoder = new TextEncoder();
    const bytes = encoder.encode('data: {"s":"é"}');
    const stream = new ReadableStream<Uint8Array>({
      start(controller) {
        const split = bytes.indexOf(0xa9);
        controller.enqueue(bytes.slice(0, split));
        controller.enqueue(bytes.slice(split));
        controller.close();
      },
    });
    const events = await collectStream(parseSSE(new Response(stream)));
    expect(events).toEqual([{ s: "é" }]);
  });
});
