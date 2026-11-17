import { describe, it, expect, afterEach, vi } from "vitest";
import { createHeadroomRetrieveTool } from "../src/tools/headroom-retrieve.js";

const HASH = "a".repeat(24);

afterEach(() => {
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
});

function stubFetch() {
  const mock = vi.fn().mockResolvedValue({
    ok: true,
    status: 200,
    json: () => Promise.resolve({ original_content: "offline fixture" }),
    text: () => Promise.resolve(""),
  });
  vi.stubGlobal("fetch", mock);
  return mock;
}

describe("headroom_retrieve execute arguments (#4638)", () => {
  it("reads the hash from params when called the way OpenClaw calls tools", async () => {
    const fetchMock = stubFetch();
    const tool = createHeadroomRetrieveTool({ proxyUrl: "http://127.0.0.1:8080" });

    const result = await tool.execute("call_fixture", { hash: HASH });

    expect(fetchMock).toHaveBeenCalledTimes(1);
    expect(String(fetchMock.mock.calls[0][0])).toBe(`http://127.0.0.1:8080/v1/retrieve/${HASH}`);
    expect(result).toEqual({
      content: [{ type: "text", text: '{"original_content":"offline fixture"}' }],
      details: { original_content: "offline fixture" },
    });
  });

  it("still accepts a direct execute({ hash }) call", async () => {
    const fetchMock = stubFetch();
    const tool = createHeadroomRetrieveTool({ proxyUrl: "http://127.0.0.1:8080" });

    const result = await tool.execute({ hash: HASH });

    expect(fetchMock).toHaveBeenCalledTimes(1);
    expect(result.details).toEqual({ original_content: "offline fixture" });
  });

  it("rejects a bad hash in params without sending a request", async () => {
    const fetchMock = stubFetch();
    const tool = createHeadroomRetrieveTool({ proxyUrl: "http://127.0.0.1:8080" });

    const result = await tool.execute("call_fixture", { hash: "not-a-hash" });

    expect(fetchMock).not.toHaveBeenCalled();
    expect(result.details).toEqual({
      status: "error",
      error: "Invalid hash format. Expected 12 or 24 hex characters.",
    });
  });

  it("rejects a missing params object without sending a request", async () => {
    const fetchMock = stubFetch();
    const tool = createHeadroomRetrieveTool({ proxyUrl: "http://127.0.0.1:8080" });

    const result = await tool.execute("call_fixture");

    expect(fetchMock).not.toHaveBeenCalled();
    expect(result.details).toMatchObject({
      status: "error",
      error: expect.stringContaining("Invalid hash format"),
    });
  });

  it("preserves a string response as model-visible text", async () => {
    stubFetch().mockResolvedValueOnce(Response.json("original text"));
    const tool = createHeadroomRetrieveTool({ proxyUrl: "http://127.0.0.1:8080" });

    expect(await tool.execute("call_fixture", { hash: HASH })).toEqual({
      content: [{ type: "text", text: "original text" }],
      details: "original text",
    });
  });

  it("returns an error result when the proxy returns HTTP 404", async () => {
    stubFetch().mockResolvedValueOnce(new Response("not found", { status: 404 }));
    const tool = createHeadroomRetrieveTool({ proxyUrl: "http://127.0.0.1:8080" });

    const result = await tool.execute("call_fixture", { hash: HASH });

    expect(result.details).toEqual({
      status: "error",
      error: "Retrieval failed: HTTP 404",
      details: "not found",
    });
    expect(result.content).toEqual([{ type: "text", text: JSON.stringify(result.details) }]);
  });

  it("returns an error result when the request rejects", async () => {
    stubFetch().mockRejectedValueOnce(new Error("connection reset"));
    const tool = createHeadroomRetrieveTool({ proxyUrl: "http://127.0.0.1:8080" });

    const result = await tool.execute("call_fixture", { hash: HASH });

    expect(result.details).toMatchObject({
      status: "error",
      error: "Retrieval failed: Error: connection reset",
      hint: expect.stringContaining("expired"),
    });
    expect(result.content).toEqual([{ type: "text", text: JSON.stringify(result.details) }]);
  });
});
