import { defineConfig } from "tsup";

// Self-contained build of the transport-plugin entry for distribution inside
// the Python wheel (headroom/providers/opencode/_dist/). The regular build
// (tsup.config.ts) leaves npm deps external and only runs from a checkout
// with node_modules present; pip installs have no node_modules, so this
// variant bundles every dependency into a single loadable file.
export default defineConfig({
  // `hook-shim/handler` is the self-contained Node `--import` loader shipped
  // alongside the entry so spawned Node children route their traffic too (#2850).
  entry: {
    "entry.opencode": "src/entry.opencode.ts",
    "hook-shim/handler": "src/hook-shim.ts",
  },
  outDir: "dist-standalone",
  format: ["esm"],
  splitting: false,
  dts: false,
  sourcemap: false,
  clean: true,
  noExternal: [/.*/],
});
