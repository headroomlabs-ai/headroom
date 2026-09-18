import { defineConfig } from "tsup";

export default defineConfig({
  entry: {
    index: "src/index.ts",
    "turn-advancement-store": "src/turn-advancement-store.ts",
  },
  format: ["esm"],
  dts: true,
  sourcemap: true,
  clean: true,
});
