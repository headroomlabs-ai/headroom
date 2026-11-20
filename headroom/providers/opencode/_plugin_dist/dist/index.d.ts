import * as _opencode_ai_plugin from '@opencode-ai/plugin';
export { H as HeadroomOpenCodePluginOptions } from './entry.opencode-Dafzpg9v.js';
import { CompressResult } from 'headroom-ai';

interface HeadroomModelMapping {
    name: string;
    limit: {
        context: number;
        output: number;
    };
}
interface HeadroomProviderOptions {
    proxyBaseUrl?: string;
    proxyPort?: number;
    defaultModel?: string;
    models?: Record<string, HeadroomModelMapping>;
}
declare const DEFAULT_MODELS: Record<string, HeadroomModelMapping>;
declare const DEFAULT_MODEL = "claude-sonnet-4-6";
interface HeadroomProvider {
    npm: string;
    name: string;
    options: {
        baseURL: string;
        apiKey?: string;
    };
    models: Record<string, HeadroomModelMapping>;
}
declare function createHeadroomProvider(options?: HeadroomProviderOptions): HeadroomProvider;
declare function buildOpencodeConfigContent(options?: HeadroomProviderOptions): Record<string, unknown>;
declare function buildOpencodeConfigContentJson(options?: HeadroomProviderOptions): string;

declare function setDefaultProxyUrl(url: string): void;
declare function getDefaultProxyUrl(): string;
interface RetrieveToolConfig {
    proxyBaseUrl: string;
}
declare function createHeadroomRetrieveTool(config: RetrieveToolConfig): {
    name: string;
    description: string;
    parameters: {
        type: "object";
        properties: {
            hash: {
                type: string;
                description: string;
            };
        };
        required: string[];
    };
    execute: (args: {
        hash: string;
    }) => Promise<string>;
};
declare function compressWithHeadroom(messages: unknown[], options?: {
    model?: string;
    tokenBudget?: number;
    proxyUrl?: string;
}): Promise<CompressResult>;

interface InstallOptions {
    proxyUrl: string;
    debug?: boolean;
}
declare function installHeadroomTransport(options: InstallOptions): () => void;

declare const _default: {
    id: string;
    server: _opencode_ai_plugin.Plugin;
};

export { DEFAULT_MODEL, DEFAULT_MODELS, type HeadroomModelMapping, type HeadroomProvider, type HeadroomProviderOptions, type RetrieveToolConfig, buildOpencodeConfigContent, buildOpencodeConfigContentJson, compressWithHeadroom, createHeadroomProvider, createHeadroomRetrieveTool, _default as default, getDefaultProxyUrl, installHeadroomTransport, setDefaultProxyUrl };
