import { defineConfig } from "tsup";

export default defineConfig({
  entry: { index: "src/index.ts", "entry.opencode": "src/entry.opencode.ts" },
  format: ["esm"],
  dts: true,
  sourcemap: true,
  clean: true,
  external: ["headroom-ai"],
});
