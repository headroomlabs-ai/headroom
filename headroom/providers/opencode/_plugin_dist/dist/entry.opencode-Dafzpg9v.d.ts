import { Plugin } from '@opencode-ai/plugin';

type HeadroomOpenCodeMode = "native-fetch" | "transport";
interface HeadroomOpenCodePluginOptions {
    proxyUrl?: string;
    project?: string;
    backend?: string;
    debug?: boolean;
    mode?: HeadroomOpenCodeMode;
}
declare const HeadroomPlugin: Plugin;

export { type HeadroomOpenCodePluginOptions as H, HeadroomPlugin as a };
