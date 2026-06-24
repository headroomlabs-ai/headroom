import { describe, expect, it } from "vitest";
import { HeadroomContextEngine } from "../src/engine.js";

describe("HeadroomContextEngine", () => {
  it("normalizes pass-through assistant messages when no proxy is available", async () => {
    const engine = new HeadroomContextEngine({ enabled: false });

    const result = await engine.assemble({
      sessionId: "test-session",
      messages: [
        { role: "user", content: "hi", timestamp: Date.now() },
        { role: "assistant", content: "hello there", timestamp: Date.now() },
      ],
    });

    expect(result.messages[1]).toMatchObject({
      role: "assistant",
      content: [{ type: "text", text: "hello there" }],
    });
  });
});
