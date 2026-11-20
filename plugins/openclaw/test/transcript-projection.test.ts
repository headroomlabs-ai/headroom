import { afterEach, describe, expect, it, vi } from "vitest";
import {
  awaitTranscriptProjectionSettle,
  resetTranscriptProjectionWaitCacheForTest,
} from "../src/transcript-projection.js";

afterEach(() => {
  resetTranscriptProjectionWaitCacheForTest();
});

describe("awaitTranscriptProjectionSettle", () => {
  it("uses runtimeContext.waitForSessionTranscriptProjection when provided", async () => {
    const waitForSessionTranscriptProjection = vi.fn(async () => undefined);

    await expect(
      awaitTranscriptProjectionSettle({
        params: { sessionId: "session-1", sessionKey: "agent:main:session-1" },
        runtimeContext: { waitForSessionTranscriptProjection },
        timeoutMs: 5_000,
      }),
    ).resolves.toEqual({ waited: true });

    expect(waitForSessionTranscriptProjection).toHaveBeenCalledWith(
      { sessionId: "session-1", sessionKey: "agent:main:session-1" },
      undefined,
    );
  });

  it("returns disabled when timeout is zero", async () => {
    const waitForSessionTranscriptProjection = vi.fn(async () => undefined);

    await expect(
      awaitTranscriptProjectionSettle({
        params: { sessionId: "session-1" },
        runtimeContext: { waitForSessionTranscriptProjection },
        timeoutMs: 0,
      }),
    ).resolves.toEqual({ waited: false, reason: "projection wait disabled" });

    expect(waitForSessionTranscriptProjection).not.toHaveBeenCalled();
  });

  it("passes abortSignal through to the host wait hook", async () => {
    const controller = new AbortController();
    const waitForSessionTranscriptProjection = vi.fn(async () => undefined);

    await awaitTranscriptProjectionSettle({
      params: { sessionId: "session-1" },
      runtimeContext: { waitForSessionTranscriptProjection },
      abortSignal: controller.signal,
      timeoutMs: 5_000,
    });

    expect(waitForSessionTranscriptProjection).toHaveBeenCalledWith(
      { sessionId: "session-1" },
      controller.signal,
    );
  });
});
