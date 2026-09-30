// Self-contained Node `--import` loader for the OpenCode transport, built
// standalone (see tsup.standalone.config.ts) and shipped inside the Python wheel
// at headroom/providers/opencode/hook-shim/handler.js.
//
// transport.ts wraps `fetch`/`http`/`https` in the plugin's own process only,
// and nothing auto-injects this loader into spawned children (#3633): child
// processes stay untouched so npm/npx, WebFetch and Node MCP servers keep
// working. Load it manually (`node --import <path-to-this-file> ...`) while
// HEADROOM_OPENCODE_TRANSPORT_PROXY_URL is set when a specific child Node
// process should route its traffic through Headroom; loading it installs the
// transport in that process.
//
// The checkout uses plugins/opencode/hook-shim/handler.js instead, which imports
// the non-bundled `../dist/index.js`; pip installs have no node_modules, so this
// variant inlines the transport. Without it shipped, the loader path did not
// exist, so a manual `--import` of it was unavailable for wheel installs
// (before #2806 a missing `--import` target crashed every Node child with
// ERR_MODULE_NOT_FOUND) (#2850).
import { installHeadroomTransport } from "./transport.js";

const proxyUrl = process.env.HEADROOM_OPENCODE_TRANSPORT_PROXY_URL;
if (!proxyUrl) {
  throw new Error(
    "Headroom OpenCode transport shim loaded without HEADROOM_OPENCODE_TRANSPORT_PROXY_URL",
  );
}

installHeadroomTransport({ proxyUrl });
