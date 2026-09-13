import { describe, expect, it } from "vitest";
import {
  DEFAULT_ASSEMBLE_COMPRESS_CONFIG,
  DEFAULT_DURABLE_COMPRESS_CONFIG,
  resolveAssembleCompressConfig,
  resolveDurableCompressConfig,
} from "../src/compress-request-config.js";

describe("compress-request-config", () => {
  it("defaults assemble config to protect_recent 2", () => {
    expect(DEFAULT_ASSEMBLE_COMPRESS_CONFIG).toEqual({ protect_recent: 2 });
    expect(resolveAssembleCompressConfig()).toEqual({ protect_recent: 2 });
  });

  it("merges operator overrides into assemble config", () => {
    expect(resolveAssembleCompressConfig({ protect_recent: 5, mode: "ccr" })).toEqual({
      protect_recent: 5,
      mode: "ccr",
    });
  });

  it("defaults durable config to safer hygiene settings", () => {
    expect(DEFAULT_DURABLE_COMPRESS_CONFIG).toMatchObject({
      mode: "lossy_inline",
      protect_recent: 2,
      compress_user_messages: true,
      frozen_message_count: 0,
    });
    expect(resolveDurableCompressConfig()).toEqual(DEFAULT_DURABLE_COMPRESS_CONFIG);
  });
});
