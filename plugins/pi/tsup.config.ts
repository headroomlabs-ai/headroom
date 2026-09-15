import { defineConfig } from "tsup";

export default defineConfig({
  entry: ["src/index.ts"],
  format: ["esm"],
  dts: true,
  sourcemap: true,
  clean: true,
  external: [
    "@earendil-works/pi-ai",
    "@earendil-works/pi-coding-agent",
    "typebox",
  ],
});
