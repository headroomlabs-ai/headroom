// Self-contained Node `--import` loader for the OpenCode transport, built
// standalone (see tsup.standalone.config.ts) and shipped inside the Python wheel
// at headroom/providers/opencode/hook-shim/handler.js.
//
// transport.ts wraps `fetch`/`http`/`https` in the plugin's own process, but a
// spawned Node child (an `npx` MCP server, `tokensave serve`, ...) is a fresh
// process, so its traffic is only routed if this loader runs at that child's
// startup via NODE_OPTIONS=--import. `shimImportSpecifier()` in transport.ts
// resolves `../hook-shim/handler.js` next to the loaded entry, which is this
// file in a wheel install.
//
// The checkout uses plugins/opencode/hook-shim/handler.js instead, which imports
// the non-bundled `../dist/index.js`; pip installs have no node_modules, so this
// variant inlines the transport. Without it shipped, the loader path did not
// exist, so child-process routing was silently disabled for wheel installs
// (before #2806 it crashed every Node child with ERR_MODULE_NOT_FOUND) (#2850).
import { installHeadroomTransport } from "./transport.js";

const proxyUrl = process.env.HEADROOM_OPENCODE_TRANSPORT_PROXY_URL;
if (!proxyUrl) {
  throw new Error(
    "Headroom OpenCode transport shim loaded without HEADROOM_OPENCODE_TRANSPORT_PROXY_URL",
  );
}

installHeadroomTransport({ proxyUrl });
