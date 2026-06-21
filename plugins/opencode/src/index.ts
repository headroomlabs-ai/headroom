export {
  DEFAULT_MODEL,
  DEFAULT_MODELS,
  buildOpencodeConfigContent,
  buildOpencodeConfigContentJson,
  createHeadroomProvider,
} from "./provider.js";
export type {
  HeadroomModelMapping,
  HeadroomProvider,
  HeadroomProviderOptions,
} from "./provider.js";
export {
  compressWithHeadroom,
  createHeadroomRetrieveTool,
  getDefaultProxyUrl,
  setDefaultProxyUrl,
} from "./retrieve.js";
export type { RetrieveToolConfig } from "./retrieve.js";
export { HeadroomPlugin, default } from "./plugin.js";
export type { HeadroomOpenCodePluginOptions } from "./plugin.js";

export { installHeadroomTransport } from "./transport.js";
