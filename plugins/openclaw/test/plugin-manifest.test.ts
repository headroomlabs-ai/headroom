import { readFile } from "node:fs/promises";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";

const rootDir = dirname(dirname(fileURLToPath(import.meta.url)));

describe("openclaw plugin manifest", () => {
  it("declares the retrieval tool contract", async () => {
    const manifest = JSON.parse(
      await readFile(join(rootDir, "openclaw.plugin.json"), "utf8"),
    );

    expect(manifest.contracts?.tools ?? []).toContain("headroom_retrieve");
  });
});
