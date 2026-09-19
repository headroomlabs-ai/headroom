import { readFile } from "node:fs/promises";
import { homedir } from "node:os";
import { join } from "node:path";

export interface HeadroomConfig {
  enabled: boolean;
  baseUrl: string;
  allowRemote: boolean;
  remoteHosts: readonly string[];
  minContextTokens: number;
  minResultChars: number;
  protectRecentToolResults: number;
  protectedTools: string[];
  maxCacheBytes: number;
}

export interface ConfigResult {
  config: HeadroomConfig;
  warnings: string[];
}

type ReadonlyHeadroomConfig = Readonly<
  Omit<HeadroomConfig, "protectedTools" | "remoteHosts"> & {
    readonly protectedTools: readonly string[];
    readonly remoteHosts: readonly string[];
  }
>;

export const DEFAULT_CONFIG: ReadonlyHeadroomConfig = Object.freeze({
  enabled: true,
  baseUrl: "http://127.0.0.1:8787",
  allowRemote: false,
  remoteHosts: Object.freeze([]),
  minContextTokens: 20_000,
  minResultChars: 4_000,
  protectRecentToolResults: 2,
  protectedTools: Object.freeze([
    "read",
    "edit",
    "write",
    "ask",
    "todo",
    "headroom_retrieve",
  ]),
  maxCacheBytes: 64 * 1024 * 1024,
});

const LOOPBACK_HOSTS: Readonly<Record<string, true>> = Object.freeze({
  "127.0.0.1": true,
  localhost: true,
  "[::1]": true,
  "::1": true,
});

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function normalizeTools(value: unknown): string[] | undefined {
  if (!Array.isArray(value) || value.length === 0) return undefined;

  const tools = value
    .filter((item): item is string => typeof item === "string")
    .map((item) => item.trim().toLowerCase())
    .filter(Boolean);
  if (tools.length !== value.length || tools.length === 0) return undefined;
  return [...new Set(tools)];
}

function normalizeHosts(value: unknown): string[] | undefined {
  if (!Array.isArray(value)) return undefined;
  const hosts = value
    .filter((item): item is string => typeof item === "string")
    .map((item) => item.trim().toLowerCase())
    .map((item) =>
      item.startsWith("[") && item.endsWith("]") ? item.slice(1, -1) : item,
    )
    .filter(Boolean);
  if (
    hosts.length !== value.length ||
    hosts.some((host) => /[/?#@*\\s]/.test(host))
  ) {
    return undefined;
  }
  return [...new Set(hosts)];
}

function normalizeUrl(
  value: unknown,
  allowRemote: boolean,
  remoteHosts: readonly string[],
): string | undefined {
  if (typeof value !== "string") return undefined;

  try {
    const url = new URL(value);
    if (url.protocol !== "http:" && url.protocol !== "https:") return undefined;
    const hostname = url.hostname.toLowerCase().replace(/^\[|\]$/g, "");
    if (
      LOOPBACK_HOSTS[url.hostname.toLowerCase()] !== true &&
      LOOPBACK_HOSTS[hostname] !== true &&
      (!allowRemote || !remoteHosts.includes(hostname))
    ) {
      return undefined;
    }
    return url.toString().replace(/\/$/, "");
  } catch {
    return undefined;
  }
}

function positiveInteger(value: unknown): value is number {
  return Number.isInteger(value) && (value as number) > 0;
}

export function validateConfig(
  raw: unknown,
  source = "config",
  fallback: ReadonlyHeadroomConfig = DEFAULT_CONFIG,
): ConfigResult {
  const warnings: string[] = [];
  const input = isRecord(raw) ? raw : {};
  if (!isRecord(raw)) warnings.push(`${source} must be a JSON object; using defaults`);

  const booleanField = (
    key: "enabled" | "allowRemote",
    defaultValue: boolean,
  ): boolean => {
    const value = input[key];
    if (value === undefined) return defaultValue;
    if (typeof value === "boolean") return value;
    warnings.push(`${source}.${key} must be a boolean; using ${defaultValue}`);
    return defaultValue;
  };

  const integerField = (
    key:
      | "minContextTokens"
      | "minResultChars"
      | "protectRecentToolResults"
      | "maxCacheBytes",
    defaultValue: number,
  ): number => {
    const value = input[key];
    if (value === undefined) return defaultValue;
    if (positiveInteger(value)) return value;
    warnings.push(
      `${source}.${key} must be a positive integer; using ${defaultValue}`,
    );
    return defaultValue;
  };

  const enabled = booleanField("enabled", fallback.enabled);
  const allowRemote = booleanField("allowRemote", fallback.allowRemote);
  const rawRemoteHosts = input.remoteHosts;
  const normalizedRemoteHosts =
    rawRemoteHosts === undefined
      ? [...fallback.remoteHosts]
      : normalizeHosts(rawRemoteHosts);
  const remoteHosts = normalizedRemoteHosts ?? [...fallback.remoteHosts];
  if (normalizedRemoteHosts === undefined) {
    warnings.push(
      `${source}.remoteHosts must be an array of exact hostnames; using defaults`,
    );
  }

  const fallbackBaseUrl =
    normalizeUrl(fallback.baseUrl, allowRemote, remoteHosts) ??
    DEFAULT_CONFIG.baseUrl;
  const rawBaseUrl = input.baseUrl;
  const baseUrl = normalizeUrl(
    rawBaseUrl === undefined ? fallback.baseUrl : rawBaseUrl,
    allowRemote,
    remoteHosts,
  );
  if (baseUrl === undefined) {
    let rejectedHost: string | undefined;
    try {
      const candidate = new URL(String(rawBaseUrl));
      const candidateHost = candidate.hostname
        .toLowerCase()
        .replace(/^\[|\]$/g, "");
      if (
        allowRemote &&
        LOOPBACK_HOSTS[candidate.hostname.toLowerCase()] !== true &&
        LOOPBACK_HOSTS[candidateHost] !== true
      ) {
        rejectedHost = candidateHost;
      }
    } catch {
      // The generic URL warning below covers malformed values.
    }
    warnings.push(
      rejectedHost
        ? `${source}.baseUrl host ${rejectedHost} is not in remoteHosts; using ${fallbackBaseUrl}`
        : `${source}.baseUrl must be an HTTP loopback URL unless allowRemote and remoteHosts explicitly permit it; using ${fallbackBaseUrl}`,
    );
  }

  const baseProtectedTools = [
    ...new Set([...DEFAULT_CONFIG.protectedTools, ...fallback.protectedTools]),
  ];
  const rawTools = input.protectedTools;
  const normalizedTools =
    rawTools === undefined ? [] : normalizeTools(rawTools);
  const protectedTools =
    normalizedTools === undefined
      ? undefined
      : [...new Set([...baseProtectedTools, ...normalizedTools])];
  if (protectedTools === undefined) {
    warnings.push(
      `${source}.protectedTools must be a nonempty string array; using defaults`,
    );
  }

  return {
    config: {
      enabled,
      baseUrl: baseUrl ?? fallbackBaseUrl,
      allowRemote,
      remoteHosts,
      minContextTokens: integerField(
        "minContextTokens",
        fallback.minContextTokens,
      ),
      minResultChars: integerField("minResultChars", fallback.minResultChars),
      protectRecentToolResults: integerField(
        "protectRecentToolResults",
        fallback.protectRecentToolResults,
      ),
      protectedTools: protectedTools ?? baseProtectedTools,
      maxCacheBytes: integerField("maxCacheBytes", fallback.maxCacheBytes),
    },
    warnings,
  };
}

const ENV_FIELDS = {
  HEADROOM_PI_ENABLED: "enabled",
  HEADROOM_PI_BASE_URL: "baseUrl",
  HEADROOM_PI_ALLOW_REMOTE: "allowRemote",
  HEADROOM_PI_REMOTE_HOSTS: "remoteHosts",
  HEADROOM_PI_MIN_CONTEXT_TOKENS: "minContextTokens",
  HEADROOM_PI_MIN_RESULT_CHARS: "minResultChars",
  HEADROOM_PI_PROTECT_RECENT_TOOL_RESULTS: "protectRecentToolResults",
  HEADROOM_PI_PROTECTED_TOOLS: "protectedTools",
  HEADROOM_PI_MAX_CACHE_BYTES: "maxCacheBytes",
} as const;

function parseEnvironment(
  env: NodeJS.ProcessEnv,
): { values: Record<string, unknown>; warnings: string[] } {
  const values: Record<string, unknown> = {};
  const warnings: string[] = [];

  for (const [envName, field] of Object.entries(ENV_FIELDS)) {
    const value = env[envName];
    if (value === undefined) continue;

    if (field === "enabled" || field === "allowRemote") {
      const normalized = value.trim().toLowerCase();
      if (normalized === "true" || normalized === "1") values[field] = true;
      else if (normalized === "false" || normalized === "0") {
        values[field] = false;
      } else {
        warnings.push(`${envName} must be true, false, 1, or 0; ignoring it`);
      }
      continue;
    }

    if (field === "baseUrl") {
      values[field] = value;
      continue;
    }

    if (field === "protectedTools" || field === "remoteHosts") {
      values[field] = value.split(",");
      continue;
    }

    const parsed = Number(value);
    if (positiveInteger(parsed)) values[field] = parsed;
    else warnings.push(`${envName} must be a positive integer; ignoring it`);
  }

  return { values, warnings };
}

export async function loadConfig(
  env: NodeJS.ProcessEnv = process.env,
  homeDir = homedir(),
): Promise<ConfigResult> {
  const path = join(homeDir, ".headroom", "integrations", "pi-extension.json");
  let fileResult: ConfigResult = {
    config: {
      ...DEFAULT_CONFIG,
      protectedTools: [...DEFAULT_CONFIG.protectedTools],
      remoteHosts: [...DEFAULT_CONFIG.remoteHosts],
    },
    warnings: [],
  };

  try {
    const contents = await readFile(path, "utf8");
    try {
      fileResult = validateConfig(JSON.parse(contents), path);
    } catch {
      fileResult.warnings.push(`${path} could not be parsed; using defaults`);
    }
  } catch (error) {
    if (!(isRecord(error) && error.code === "ENOENT")) {
      fileResult.warnings.push(`${path} could not be read; using defaults`);
    }
  }

  const parsedEnv = parseEnvironment(env);
  const envResult = validateConfig(
    { ...fileResult.config, ...parsedEnv.values },
    "environment",
    fileResult.config,
  );

  return {
    config: envResult.config,
    warnings: [
      ...fileResult.warnings,
      ...parsedEnv.warnings,
      ...envResult.warnings,
    ],
  };
}
