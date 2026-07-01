// Dedicated entry for OpenCode's plugin loader.
//
// OpenCode loads a plugin module and treats its exports as plugin factories —
// it rejects the module if a non-function export is present ("Plugin export is
// not a function"). The library barrel (index.ts) re-exports helpers/constants,
// so it cannot be loaded directly. This entry exports ONLY the plugin function.
export { HeadroomPlugin as default } from "./plugin.js";
