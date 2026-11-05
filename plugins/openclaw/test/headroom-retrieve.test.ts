/**
 * Hash format accepted by the headroom_retrieve tool.
 *
 * The Rust SmartCrusher row-drop path keys its CCR marker with a 12-char
 * SHA-256 hex prefix (`hash_canonical` in
 * crates/headroom-core/src/transforms/smart_crusher/crusher.rs), mirrored into
 * the compression store through `explicit_hash`, so a 12-char hash is a valid
 * retrieval key just like a 24-char one.
 */

import { describe, it, expect, afterEach, vi } from "vitest";
import { createHeadroomRetrieveTool } from "../src/tools/headroom-retrieve.js";

const PROXY_URL = "http://127.0.0.1:8788";

function stubRetrieveOk() {
  const mock = vi.fn().mockResolvedValue({
    ok: true,
    status: 200,
    json: () => Promise.resolve({ original_content: "the full original text" }),
  });
  vi.stubGlobal("fetch", mock);
  return mock;
}

function retrieve(hash: string) {
  const tool = createHeadroomRetrieveTool({ proxyUrl: PROXY_URL });
  return tool.execute({ hash });
}

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("headroom_retrieve hash validation", () => {
  it("accepts a 12-character hex hash and queries the store", async () => {
    const fetchMock = stubRetrieveOk();

    const result = await retrieve("a1b2c3d4e5f6");

    expect(fetchMock).toHaveBeenCalledTimes(1);
    expect(fetchMock.mock.calls[0][0]).toBe(`${PROXY_URL}/v1/retrieve/a1b2c3d4e5f6`);
    expect(JSON.parse(result)).toEqual({ original_content: "the full original text" });
  });

  it("queries the store with the lowercased key for an uppercase 12-character hash", async () => {
    const fetchMock = stubRetrieveOk();

    await retrieve("A1B2C3D4E5F6");

    expect(fetchMock).toHaveBeenCalledTimes(1);
    expect(fetchMock.mock.calls[0][0]).toBe(`${PROXY_URL}/v1/retrieve/a1b2c3d4e5f6`);
  });

  it("still accepts a 24-character hex hash", async () => {
    const fetchMock = stubRetrieveOk();

    const result = await retrieve("a1b2c3d4e5f6a1b2c3d4e5f6");

    expect(fetchMock).toHaveBeenCalledTimes(1);
    expect(fetchMock.mock.calls[0][0]).toBe(`${PROXY_URL}/v1/retrieve/a1b2c3d4e5f6a1b2c3d4e5f6`);
    expect(JSON.parse(result)).toEqual({ original_content: "the full original text" });
  });

  it("queries the store with the lowercased key for an uppercase 24-character hash", async () => {
    const fetchMock = stubRetrieveOk();

    await retrieve("A1B2C3D4E5F6A1B2C3D4E5F6");

    expect(fetchMock).toHaveBeenCalledTimes(1);
    expect(fetchMock.mock.calls[0][0]).toBe(`${PROXY_URL}/v1/retrieve/a1b2c3d4e5f6a1b2c3d4e5f6`);
  });

  it("rejects a non-hex 12-character hash without querying the store", async () => {
    const fetchMock = stubRetrieveOk();

    const result = await retrieve("zzzzzzzzzzzz");

    expect(fetchMock).not.toHaveBeenCalled();
    expect(JSON.parse(result).error).toBe("Invalid hash format. Expected 12 or 24 hex characters.");
  });

  it("rejects a 13-character hash without querying the store", async () => {
    const fetchMock = stubRetrieveOk();

    const result = await retrieve("a1b2c3d4e5f67");

    expect(fetchMock).not.toHaveBeenCalled();
    expect(JSON.parse(result).error).toBe("Invalid hash format. Expected 12 or 24 hex characters.");
  });

  it("rejects an empty hash without querying the store", async () => {
    const fetchMock = stubRetrieveOk();

    const result = await retrieve("");

    expect(fetchMock).not.toHaveBeenCalled();
    expect(JSON.parse(result).error).toBe("Invalid hash format. Expected 12 or 24 hex characters.");
  });
});

describe("headroom_retrieve hash documentation", () => {
  it("documents both supported hash lengths", () => {
    const tool = createHeadroomRetrieveTool({ proxyUrl: PROXY_URL });

    const mentionsBothLengths = /\b12\b[^\n]*\b24\b/;

    expect(tool.description).toMatch(mentionsBothLengths);
    expect(tool.parameters.properties.hash.description).toMatch(mentionsBothLengths);
  });
});
