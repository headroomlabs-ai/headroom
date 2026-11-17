// Dedicated entry for OpenCode's plugin loader.
//
// Both OpenCode loaders only read the default export: 1.x takes `{ id, server }`
// and 2.x takes `{ id, setup }`, so the dual-shape plugin object loads on
// either. The library barrel (index.ts) re-exports helpers the plugin does not
// need, so this entry keeps the standalone wheel bundle to the plugin alone.
export { default } from "./plugin.js";
