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
    expect(JSON.parse(result)).toEqual({ original_content: "offline fixture" });
  });

  it("still accepts a direct execute({ hash }) call", async () => {
    const fetchMock = stubFetch();
    const tool = createHeadroomRetrieveTool({ proxyUrl: "http://127.0.0.1:8080" });

    const result = await tool.execute({ hash: HASH });

    expect(fetchMock).toHaveBeenCalledTimes(1);
    expect(JSON.parse(result)).toEqual({ original_content: "offline fixture" });
  });

  it("rejects a bad hash in params without sending a request", async () => {
    const fetchMock = stubFetch();
    const tool = createHeadroomRetrieveTool({ proxyUrl: "http://127.0.0.1:8080" });

    const result = await tool.execute("call_fixture", { hash: "not-a-hash" });

    expect(fetchMock).not.toHaveBeenCalled();
    expect(result).toContain("Invalid hash format");
  });

  it("rejects a missing params object without sending a request", async () => {
    const fetchMock = stubFetch();
    const tool = createHeadroomRetrieveTool({ proxyUrl: "http://127.0.0.1:8080" });

    const result = await tool.execute("call_fixture");

    expect(fetchMock).not.toHaveBeenCalled();
    expect(result).toContain("Invalid hash format");
  });
});
