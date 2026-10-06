import { describe, expect, it } from "vitest";

import { stripTrailingSlashes } from "./strings.js";

describe("stripTrailingSlashes", () => {
  it("removes every trailing slash", () => {
    expect(stripTrailingSlashes("http://127.0.0.1:8787/")).toBe("http://127.0.0.1:8787");
    expect(stripTrailingSlashes("http://127.0.0.1:8787///")).toBe("http://127.0.0.1:8787");
  });

  it("leaves values without trailing slashes untouched", () => {
    expect(stripTrailingSlashes("https://gateway.example/api/v1")).toBe("https://gateway.example/api/v1");
    expect(stripTrailingSlashes("")).toBe("");
    expect(stripTrailingSlashes("///")).toBe("");
  });

  it("handles a long slash run ending in a non-slash in bounded time", () => {
    const adversarial = `https://example.com/a${"/".repeat(60000)}x`;
    expect(stripTrailingSlashes(adversarial)).toBe(adversarial);
    expect(stripTrailingSlashes(`https://example.com/a${"/".repeat(60000)}`)).toBe("https://example.com/a");
  });
});
