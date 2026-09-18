/**
 * Shared /v1/compress request `config` fields used by assemble() and durable compaction.
 * Values are passed through to the Headroom proxy as-is (snake_case where required).
 */

export const DEFAULT_PROTECT_RECENT = 2;

/** Proxy `/v1/compress` config subset — extra keys allowed for forward compatibility. */
export interface CompressRequestConfig {
  protect_recent?: number;
  mode?: string;
  compress_user_messages?: boolean;
  frozen_message_count?: number;
  [key: string]: unknown;
}

export const DEFAULT_ASSEMBLE_COMPRESS_CONFIG: CompressRequestConfig = {
  protect_recent: DEFAULT_PROTECT_RECENT,
};

export const DEFAULT_DURABLE_COMPRESS_CONFIG: CompressRequestConfig = {
  frozen_message_count: 0,
  mode: "lossy_inline",
  compress_user_messages: true,
  protect_recent: DEFAULT_PROTECT_RECENT,
};

export function resolveAssembleCompressConfig(
  configured?: CompressRequestConfig | null,
): CompressRequestConfig {
  return {
    ...DEFAULT_ASSEMBLE_COMPRESS_CONFIG,
    ...(configured ?? {}),
  };
}

export function resolveDurableCompressConfig(
  configured?: CompressRequestConfig | null,
): CompressRequestConfig {
  return {
    ...DEFAULT_DURABLE_COMPRESS_CONFIG,
    ...(configured ?? {}),
  };
}
