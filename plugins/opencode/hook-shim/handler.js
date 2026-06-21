import { installHeadroomTransport } from "../dist/index.js";

const proxyUrl = process.env.HEADROOM_OPENCODE_TRANSPORT_PROXY_URL;
if (!proxyUrl) {
  throw new Error("Headroom OpenCode transport shim loaded without HEADROOM_OPENCODE_TRANSPORT_PROXY_URL");
}

installHeadroomTransport({ proxyUrl });
