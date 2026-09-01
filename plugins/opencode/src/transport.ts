import { createRequire, syncBuiltinESMExports } from "node:module";
import os from "node:os";
import path from "node:path";
import { createHash, randomUUID } from "node:crypto";
import { fileURLToPath } from "node:url";

const nodeRequire = createRequire(import.meta.url);
const http = nodeRequire("node:http") as typeof import("node:http");
const https = nodeRequire("node:https") as typeof import("node:https");
const http2 = nodeRequire("node:http2") as typeof import("node:http2");
const childProcess = nodeRequire("node:child_process") as typeof import("node:child_process");
const fs = nodeRequire("node:fs") as typeof import("node:fs");

const BASE_URL_HEADER = "x-headroom-base-url";
const ORIGINAL_PATH_HEADER = "x-headroom-original-path";
const PROJECT_HEADER = "x-headroom-project";
const PROXY_ENV = "HEADROOM_OPENCODE_TRANSPORT_PROXY_URL";
export const TOOL_POLICY_ENV = "HEADROOM_TOOL_POLICY_JSON";
export const TOOL_POLICY_PATH_ENV = "HEADROOM_TOOL_POLICY_PATH";
export const TOOL_POLICY_URL_ENV = "HEADROOM_TOOL_POLICY_URL";
export const TOOL_POLICY_TOKEN_ENV = "HEADROOM_TOOL_POLICY_TOKEN";
export const TOOL_POLICY_REFRESH_SECONDS_ENV = "HEADROOM_TOOL_POLICY_REFRESH_SECONDS";
const TOOL_POLICY_FILE_NAME = "tool_policy.json";
const POLICY_VERSION = 1;
const DEFAULT_REFRESH_SECONDS = 300;
const MAX_REFRESH_SECONDS = 3600;
const REMOTE_TIMEOUT_MS = 5_000;
const MAX_REMOTE_POLICY_BYTES = 1024 * 1024;
const STATE_KEY = Symbol.for("headroom.opencode.transport");

type FetchArgs = Parameters<typeof fetch>;
type HttpRequest = typeof http.request;
type HttpGet = typeof http.get;
type HttpsRequest = typeof https.request;
type HttpsGet = typeof https.get;
type Http2Connect = typeof http2.connect;
type ChildSpawn = typeof childProcess.spawn;
type ChildExec = typeof childProcess.exec;
type ChildExecFile = typeof childProcess.execFile;
type ChildFork = typeof childProcess.fork;
export type ToolPolicyAction = "allow" | "deny" | "require_approval";
export type ToolPolicyMode = "enforce" | "report_only";
export type ToolPolicyScope = "tool_call" | "shell" | "http";

export interface HeadroomToolPolicyRule {
  id?: string;
  scope: ToolPolicyScope;
  action: ToolPolicyAction;
  reason?: string;
  tool?: string | string[];
  command?: string | string[];
  argsPattern?: string;
  cwdPattern?: string;
  envKeys?: string[];
  domain?: string | string[];
  urlPattern?: string;
}

export interface HeadroomToolPolicyConfig {
  version?: 1;
  mode?: ToolPolicyMode;
  defaultAction?: Extract<ToolPolicyAction, "allow" | "deny">;
  rules: HeadroomToolPolicyRule[];
}

type HeadroomToolPolicyInput = HeadroomToolPolicyConfig | string;

interface InstallOptions {
  proxyUrl: string;
  project?: string;
  policyProject?: string;
  debug?: boolean;
  toolPolicy?: HeadroomToolPolicyInput;
}

interface CompiledToolPolicyRule {
  id: string;
  scope: ToolPolicyScope;
  action: ToolPolicyAction;
  reason?: string;
  tools?: string[];
  commands?: string[];
  argsPattern?: RegExp;
  cwdPattern?: RegExp;
  envKeys?: string[];
  domains?: string[];
  urlPattern?: RegExp;
}

interface CompiledToolPolicy {
  version: 1;
  mode: ToolPolicyMode;
  defaultAction: "allow" | "deny";
  rules: CompiledToolPolicyRule[];
  serialized: string;
  source: string;
  validUntil?: number;
}

interface TransportState {
  refs: number;
  policyContextKey: string;
  proxyUrl: string;
  project: string | undefined;
  debug: boolean;
  toolPolicy?: CompiledToolPolicy;
  toolPolicyInput?: HeadroomToolPolicyInput;
  remotePolicyUrl?: string;
  remotePolicyToken: string;
  policyUnavailable?: string;
  previousNodeOptions?: string;
  previousProxyUrlEnv?: string;
  previousToolPolicyEnv?: string;
  originalFetch: typeof fetch;
  originalHttpRequest: HttpRequest;
  originalHttpGet: HttpGet;
  originalHttpsRequest: HttpsRequest;
  originalHttpsGet: HttpsGet;
  originalHttp2Connect: Http2Connect;
  originalChildSpawn: ChildSpawn;
  originalChildExec: ChildExec;
  originalChildExecFile: ChildExecFile;
  originalChildFork: ChildFork;
}

interface GlobalWithHeadroomTransport {
  [STATE_KEY]?: TransportState;
}

interface NodeRequestParts {
  url?: URL;
  options: Record<string, unknown>;
  callback?: (...args: unknown[]) => unknown;
}

export interface HeadroomToolPolicyDecision {
  version: 1;
  decisionId: string;
  authority: "authoritative" | "advisory";
  scope: ToolPolicyScope;
  action: ToolPolicyAction;
  effectiveAction: ToolPolicyAction;
  mode: ToolPolicyMode;
  matchedRuleId?: string;
  reason?: string;
  resource: string;
  requestHash: string;
  source: string;
  binding?: HeadroomToolPolicyBinding;
}

export interface HeadroomToolPolicyBinding {
  caller: "opencode";
  adapter: "tool.execute.before";
  sessionID: string;
  taskID: string;
  callID: string;
  toolName: string;
  cwd?: string;
  canonicalArgsHash: string;
}

export interface HeadroomToolPolicyAcknowledgement {
  version: 1;
  event: "headroom_tool_policy_enforcement_acknowledgement";
  decisionId: string;
  authority: "authoritative";
  effect: "allowed" | "blocked" | "unknown";
  reason?:
    | "postflight_timeout"
    | "postflight_mismatch"
    | "ambiguous_reused_call"
    | "capacity_evicted"
    | "plugin_disposed"
    | "call_replaced";
  requestHash: string;
  binding: HeadroomToolPolicyBinding;
  timestamp: string;
}

export interface HeadroomToolPolicyPreflight {
  decision: HeadroomToolPolicyDecision;
}

export class ToolPolicyEnforcementError extends Error {
  readonly decision: HeadroomToolPolicyDecision;
  readonly acknowledgement: HeadroomToolPolicyAcknowledgement;

  constructor(
    message: string,
    decision: HeadroomToolPolicyDecision,
    acknowledgement: HeadroomToolPolicyAcknowledgement,
  ) {
    super(message);
    this.name = "ToolPolicyEnforcementError";
    this.decision = decision;
    this.acknowledgement = acknowledgement;
  }
}

interface ShellPolicyInput {
  scope: "shell";
  resource: string;
  command: string;
  argsText: string;
  cwd?: string;
  env?: NodeJS.ProcessEnv | Record<string, unknown>;
  toolName?: string;
}

interface HttpPolicyInput {
  scope: "http";
  resource: string;
  url: URL;
}

interface ToolCallPolicyInput {
  scope: "tool_call";
  resource: string;
  toolName: string;
  argsText: string;
  cwd?: string;
  env?: NodeJS.ProcessEnv | Record<string, unknown>;
}

function getState(): TransportState | undefined {
  return (globalThis as GlobalWithHeadroomTransport)[STATE_KEY];
}

function setState(state: TransportState | undefined): void {
  (globalThis as GlobalWithHeadroomTransport)[STATE_KEY] = state;
}

// ponytail: the shim only exists next to the checkout build
// (plugins/opencode/dist/). The wheel ships entry.opencode.js alone, so
// `--import=<missing file>` killed every Node child at startup — including
// OpenCode's stdio MCP servers (issue #2798). No shim on disk, no injection:
// children go direct instead of dying. Upgrade path is bundling the shim into
// _dist/ so wheel installs get child-process routing back.
function shimImportSpecifier(): string | undefined {
  const shim = new URL("../hook-shim/handler.js", import.meta.url);
  return fs.existsSync(shim) ? shim.href : undefined;
}

function withNodeImportOption(existing: string | undefined, shim: string): string {
  const parts = existing?.trim() ? existing.trim().split(/\s+/) : [];
  const alreadyPresent = parts.some((part, index) => {
    return part === `--import=${shim}` || (part === "--import" && parts[index + 1] === shim);
  });
  if (!alreadyPresent) {
    parts.push(`--import=${shim}`);
  }
  return parts.join(" ");
}

function parseToolPolicyJson(raw: string, source: string): HeadroomToolPolicyConfig {
  try {
    const parsed = JSON.parse(raw) as HeadroomToolPolicyConfig;
    if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) {
      throw new Error("expected a JSON object");
    }
    if (parsed.version !== undefined && parsed.version !== POLICY_VERSION) {
      throw new Error(`unsupported version ${String(parsed.version)}; expected ${POLICY_VERSION}`);
    }
    return parsed;
  } catch {
    throw new Error(`Invalid Headroom tool policy JSON in ${source}`);
  }
}

function readToolPolicyFile(filePath: string, source: string): HeadroomToolPolicyConfig {
  try {
    return parseToolPolicyJson(fs.readFileSync(filePath, "utf8"), source);
  } catch (error) {
    if (error instanceof Error && error.message.startsWith("Invalid Headroom tool policy JSON")) {
      throw error;
    }
    throw new Error(`Invalid Headroom tool policy file ${filePath} (${source}): ${String(error)}`);
  }
}

export function defaultGlobalToolPolicyPath(): string {
  const explicitConfigDir = process.env.HEADROOM_CONFIG_DIR?.trim();
  if (explicitConfigDir) {
    return path.join(explicitConfigDir, TOOL_POLICY_FILE_NAME);
  }
  const explicitWorkspaceDir = process.env.HEADROOM_WORKSPACE_DIR?.trim();
  if (explicitWorkspaceDir) {
    return path.join(explicitWorkspaceDir, "config", TOOL_POLICY_FILE_NAME);
  }
  return path.join(os.homedir(), ".headroom", "config", TOOL_POLICY_FILE_NAME);
}

export function findLocalToolPolicyPath(project: string | undefined): string | undefined {
  let start = path.resolve(project || process.cwd());
  try {
    if (fs.statSync(start).isFile()) {
      start = path.dirname(start);
    }
  } catch {
    // Nonexistent project paths are still useful as discovery starting points.
  }
  let current = start;
  while (true) {
    const candidate = path.join(current, ".headroom", TOOL_POLICY_FILE_NAME);
    if (fs.existsSync(candidate)) {
      return candidate;
    }
    const parent = path.dirname(current);
    if (parent === current) {
      return undefined;
    }
    current = parent;
  }
}

function loadToolPolicyConfig(
  policy: HeadroomToolPolicyInput | undefined,
  project: string | undefined,
): HeadroomToolPolicyConfig | undefined {
  if (policy === undefined) {
    const raw = process.env[TOOL_POLICY_ENV]?.trim();
    if (raw) {
      return parseToolPolicyJson(raw, TOOL_POLICY_ENV);
    }
    const rawPath = process.env[TOOL_POLICY_PATH_ENV]?.trim();
    if (rawPath) {
      return readToolPolicyFile(rawPath, TOOL_POLICY_PATH_ENV);
    }
    if (process.env[TOOL_POLICY_URL_ENV]?.trim()) {
      return undefined;
    }
    const globalPath = defaultGlobalToolPolicyPath();
    if (fs.existsSync(globalPath)) {
      return readToolPolicyFile(globalPath, globalPath);
    }
    const localPath = findLocalToolPolicyPath(project);
    if (localPath) {
      return readToolPolicyFile(localPath, localPath);
    }
    return undefined;
  }
  if (typeof policy !== "string") {
    return policy;
  }
  const trimmed = policy.trim();
  if (!trimmed) {
    return undefined;
  }
  if (trimmed.startsWith("{")) {
    return parseToolPolicyJson(trimmed, "inline string");
  }
  return readToolPolicyFile(trimmed, trimmed);
}

function compileRegex(source: string | undefined, field: string, ruleId: string): RegExp | undefined {
  if (!source) {
    return undefined;
  }
  try {
    return new RegExp(source);
  } catch {
    throw new Error(
      `Invalid Headroom tool policy regex for ${field} in rule ${ruleId}`,
    );
  }
}

function asArray(value: string | string[] | undefined): string[] | undefined {
  if (value === undefined) {
    return undefined;
  }
  return Array.isArray(value) ? value : [value];
}

function compileToolPolicy(
  policy: HeadroomToolPolicyInput | undefined,
  project: string | undefined,
  source = "configured",
): CompiledToolPolicy | undefined {
  const loaded = loadToolPolicyConfig(policy, project);
  if (!loaded) {
    return undefined;
  }
  if (!Array.isArray(loaded.rules)) {
    throw new Error("Headroom tool policy requires a rules array");
  }
  if (loaded.version !== undefined && loaded.version !== POLICY_VERSION) {
    throw new Error(`Unsupported Headroom tool policy version: ${String(loaded.version)}`);
  }
  const compiledRules = loaded.rules.map((rule, index): CompiledToolPolicyRule => {
    const id = rule.id?.trim() || `rule_${index + 1}`;
    if (rule.scope !== "tool_call" && rule.scope !== "shell" && rule.scope !== "http") {
      throw new Error(`Invalid Headroom tool policy scope in rule ${id}: ${String(rule.scope)}`);
    }
    if (rule.action !== "allow" && rule.action !== "deny" && rule.action !== "require_approval") {
      throw new Error(`Invalid Headroom tool policy action in rule ${id}: ${String(rule.action)}`);
    }
    return {
      id,
      scope: rule.scope,
      action: rule.action,
      reason: rule.reason,
      tools: asArray(rule.tool)?.map((entry) => entry.toLowerCase()),
      commands: asArray(rule.command)?.map((entry) => entry.toLowerCase()),
      argsPattern: compileRegex(rule.argsPattern, "argsPattern", id),
      cwdPattern: compileRegex(rule.cwdPattern, "cwdPattern", id),
      envKeys: rule.envKeys?.map((entry) => entry.toLowerCase()),
      domains: asArray(rule.domain)?.map((entry) => entry.toLowerCase()),
      urlPattern: compileRegex(rule.urlPattern, "urlPattern", id),
    };
  });
  const defaultAction = loaded.defaultAction ?? "allow";
  if (defaultAction !== "allow" && defaultAction !== "deny") {
    throw new Error(`Invalid Headroom tool policy defaultAction: ${String(defaultAction)}`);
  }
  const mode = loaded.mode ?? "enforce";
  if (mode !== "enforce" && mode !== "report_only") {
    throw new Error(`Invalid Headroom tool policy mode: ${String(mode)}`);
  }
  return {
    version: POLICY_VERSION,
    mode,
    defaultAction,
    rules: compiledRules,
    serialized: JSON.stringify({
      version: POLICY_VERSION,
      mode,
      defaultAction,
      rules: loaded.rules.map((rule, index) => ({
        ...rule,
        id: rule.id?.trim() || `rule_${index + 1}`,
      })),
    }),
    source,
  };
}

function workspaceDir(): string {
  return process.env.HEADROOM_WORKSPACE_DIR?.trim() || path.join(os.homedir(), ".headroom");
}

function processIsStateless(): boolean {
  return ["1", "true", "yes", "on"].includes(
    process.env.HEADROOM_STATELESS?.trim().toLowerCase() ?? "",
  );
}

export function toolPolicyRefreshSeconds(env: NodeJS.ProcessEnv = process.env): number {
  const raw = env[TOOL_POLICY_REFRESH_SECONDS_ENV]?.trim();
  if (!raw || !/^\d+$/.test(raw)) {
    return DEFAULT_REFRESH_SECONDS;
  }
  const value = Number(raw);
  return Number.isSafeInteger(value) && value >= DEFAULT_REFRESH_SECONDS && value <= MAX_REFRESH_SECONDS
    ? value
    : DEFAULT_REFRESH_SECONDS;
}

export function remoteToolPolicyCachePath(url: string, token = ""): string {
  const digest = createHash("sha256").update(`${url}\0${token}`).digest("hex");
  return path.join(workspaceDir(), "policy-cache", `${digest}.json`);
}

export function isAllowedToolPolicyUrl(value: string): boolean {
  let url: URL;
  try {
    url = new URL(value);
  } catch {
    return false;
  }
  if (url.protocol === "https:") {
    return true;
  }
  if (url.protocol !== "http:") {
    return false;
  }
  const hostname = url.hostname.toLowerCase().replace(/^\[|\]$/g, "");
  return (
    hostname === "localhost" ||
    hostname === "::1" ||
    /^127(?:\.\d{1,3}){3}$/.test(hostname)
  );
}

interface RemotePolicyCache {
  cache_version: 2;
  url_hash: string;
  etag: string;
  fetched_at: number;
  policy: HeadroomToolPolicyConfig;
}

function readRemotePolicyCache(url: string, token: string): RemotePolicyCache | undefined {
  if (processIsStateless()) {
    return undefined;
  }
  const cachePath = remoteToolPolicyCachePath(url, token);
  if (!fs.existsSync(cachePath)) {
    return undefined;
  }
  try {
    const parsed = JSON.parse(fs.readFileSync(cachePath, "utf8")) as Partial<RemotePolicyCache>;
    if (
      parsed.cache_version !== 2 ||
      parsed.url_hash !== createHash("sha256").update(url).digest("hex") ||
      typeof parsed.fetched_at !== "number" ||
      !parsed.policy ||
      typeof parsed.policy !== "object" ||
      Array.isArray(parsed.policy)
    ) {
      return undefined;
    }
    return parsed as RemotePolicyCache;
  } catch {
    return undefined;
  }
}

function writeRemotePolicyCache(url: string, cache: RemotePolicyCache, token: string): void {
  if (processIsStateless()) {
    return;
  }
  const cachePath = remoteToolPolicyCachePath(url, token);
  const directory = path.dirname(cachePath);
  const temporaryPath = path.join(
    directory,
    `.${path.basename(cachePath)}.${process.pid}.${randomUUID()}.tmp`,
  );
  fs.mkdirSync(directory, { recursive: true });
  try {
    fs.writeFileSync(temporaryPath, `${JSON.stringify(cache)}\n`, {
      encoding: "utf8",
      flag: "wx",
    });
    fs.renameSync(temporaryPath, cachePath);
  } finally {
    try {
      fs.rmSync(temporaryPath, { force: true });
    } catch {
      // Cache persistence is best effort; never mask the original failure.
    }
  }
}

async function readLimitedResponseText(response: Response): Promise<string> {
  const declaredLength = Number(response.headers.get("content-length"));
  if (Number.isFinite(declaredLength) && declaredLength > MAX_REMOTE_POLICY_BYTES) {
    throw new Error("remote Headroom tool policy exceeds 1 MiB");
  }
  if (!response.body) {
    return "";
  }
  const reader = response.body.getReader();
  const decoder = new TextDecoder("utf-8", { fatal: true });
  let size = 0;
  let text = "";
  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      size += value.byteLength;
      if (size > MAX_REMOTE_POLICY_BYTES) {
        await reader.cancel();
        throw new Error("remote Headroom tool policy exceeds 1 MiB");
      }
      text += decoder.decode(value, { stream: true });
    }
    return text + decoder.decode();
  } finally {
    reader.releaseLock();
  }
}

async function loadRemoteToolPolicy(
  url: string,
  token: string,
  originalFetch: typeof fetch,
  now = Date.now() / 1000,
): Promise<CompiledToolPolicy> {
  const remoteLabel = new URL(url).hostname.toLowerCase().replace(/^\[|\]$/g, "");
  const cache = readRemotePolicyCache(url, token);
  const cacheAge = cache ? now - cache.fetched_at : undefined;
  if (
    cache &&
    cacheAge !== undefined &&
    cacheAge >= 0 &&
    cacheAge < toolPolicyRefreshSeconds()
  ) {
    const compiled = compileToolPolicy(cache.policy, undefined, `remote-cache:${remoteLabel}`)!;
    compiled.validUntil = cache.fetched_at + toolPolicyRefreshSeconds();
    return compiled;
  }

  const headers = new Headers({ accept: "application/json" });
  if (token) {
    headers.set("authorization", `Bearer ${token}`);
  }
  if (cache?.etag) {
    headers.set("if-none-match", cache.etag);
  }

  let response: Response;
  try {
    response = await originalFetch(url, {
      method: "GET",
      headers,
      redirect: "manual",
      signal: AbortSignal.timeout(REMOTE_TIMEOUT_MS),
    });
  } catch {
    throw new Error(`Headroom tool policy service ${remoteLabel} is unavailable`);
  }
  if (response.status === 304 && cache) {
    const refreshed = { ...cache, fetched_at: now };
    writeRemotePolicyCache(url, refreshed, token);
    const compiled = compileToolPolicy(
      refreshed.policy,
      undefined,
      `remote-cache:${remoteLabel}`,
    )!;
    compiled.validUntil = now + toolPolicyRefreshSeconds();
    return compiled;
  }
  if (!response.ok) {
    throw new Error(`Headroom tool policy service ${remoteLabel} returned HTTP ${response.status}`);
  }

  const text = await readLimitedResponseText(response);
  const payload = parseToolPolicyJson(text, remoteLabel);
  const compiled = compileToolPolicy(payload, undefined, `remote:${remoteLabel}`)!;
  compiled.validUntil = now + toolPolicyRefreshSeconds();
  writeRemotePolicyCache(
    url,
    {
      cache_version: 2,
      url_hash: createHash("sha256").update(url).digest("hex"),
      etag: response.headers.get("etag") ?? "",
      fetched_at: now,
      policy: JSON.parse(compiled.serialized) as HeadroomToolPolicyConfig,
    },
    token,
  );
  return compiled;
}

export async function refreshHeadroomToolPolicy(now = Date.now() / 1000): Promise<void> {
  const state = getState();
  if (!state || state.toolPolicyInput !== undefined) {
    return;
  }
  const url = state.remotePolicyUrl;
  if (!url) {
    return;
  }
  if (!isAllowedToolPolicyUrl(url)) {
    state.toolPolicy = undefined;
    state.policyUnavailable =
      `${TOOL_POLICY_URL_ENV} must use HTTPS; HTTP is allowed only for loopback hosts`;
    return;
  }
  try {
    state.toolPolicy = await loadRemoteToolPolicy(
      url,
      state.remotePolicyToken,
      state.originalFetch,
      now,
    );
    state.policyUnavailable = undefined;
    installProcessEnv(state.proxyUrl, state.toolPolicy);
  } catch (error) {
    state.toolPolicy = undefined;
    state.policyUnavailable = error instanceof Error ? error.message : String(error);
  }
}

function withShimEnv(
  env: NodeJS.ProcessEnv | Record<string, unknown> | undefined,
  proxyUrl: string,
  toolPolicy: CompiledToolPolicy | undefined,
): NodeJS.ProcessEnv {
  const nextEnv = { ...(env ?? process.env) } as NodeJS.ProcessEnv;
  delete nextEnv[TOOL_POLICY_TOKEN_ENV];
  delete nextEnv[TOOL_POLICY_URL_ENV];
  nextEnv[PROXY_ENV] = proxyUrl;
  if (toolPolicy) {
    nextEnv[TOOL_POLICY_ENV] = toolPolicy.serialized;
  } else {
    delete nextEnv[TOOL_POLICY_ENV];
  }
  const shim = shimImportSpecifier();
  if (shim) {
    nextEnv.NODE_OPTIONS = withNodeImportOption(nextEnv.NODE_OPTIONS, shim);
  }
  return nextEnv;
}

function installProcessEnv(proxyUrl: string, toolPolicy: CompiledToolPolicy | undefined): void {
  process.env[PROXY_ENV] = proxyUrl;
  if (toolPolicy) {
    process.env[TOOL_POLICY_ENV] = toolPolicy.serialized;
  } else {
    delete process.env[TOOL_POLICY_ENV];
  }
  const shim = shimImportSpecifier();
  if (shim) {
    process.env.NODE_OPTIONS = withNodeImportOption(process.env.NODE_OPTIONS, shim);
  }
}

function isOptions(value: unknown): value is Record<string, unknown> {
  return Boolean(value) && typeof value === "object" && !Array.isArray(value) && !(value instanceof URL);
}

function injectOptionsEnv(args: unknown[], optionIndex: number, proxyUrl: string): unknown[] {
  const state = getState();
  const nextArgs = [...args];
  const callback = typeof nextArgs.at(-1) === "function" ? nextArgs.pop() : undefined;
  const existing = isOptions(nextArgs[optionIndex]) ? { ...(nextArgs[optionIndex] as Record<string, unknown>) } : {};
  existing.env = withShimEnv(existing.env as NodeJS.ProcessEnv | undefined, proxyUrl, state?.toolPolicy);

  if (isOptions(nextArgs[optionIndex])) {
    nextArgs[optionIndex] = existing;
  } else {
    nextArgs.splice(optionIndex, 0, existing);
  }

  if (callback) {
    nextArgs.push(callback);
  }
  return nextArgs;
}

function normalizedCommandName(command: string): string {
  const trimmed = command.trim();
  if (!trimmed) {
    return "";
  }
  return path.basename(trimmed).toLowerCase();
}

function commandMatches(command: string, patterns: string[] | undefined): boolean {
  if (!patterns?.length) {
    return true;
  }
  const normalized = normalizedCommandName(command);
  const lowered = command.trim().toLowerCase();
  return patterns.some((pattern) => {
    const candidate = pattern.toLowerCase();
    return candidate === lowered || candidate === normalized || path.basename(candidate) === normalized;
  });
}

function shellTokens(commandLine: string): string[] {
  const tokens: string[] = [];
  let current = "";
  let quote = "";
  for (let index = 0; index < commandLine.length; index += 1) {
    const char = commandLine[index];
    if (quote) {
      if (char === quote) quote = "";
      else if (char === "\\" && quote === '"' && index + 1 < commandLine.length) {
        current += commandLine[++index];
      } else current += char;
    } else if (char === "'" || char === '"') {
      quote = char;
    } else if (char === "\r" || char === "\n") {
      if (current) tokens.push(current);
      current = "";
      tokens.push(";");
      if (char === "\r" && commandLine[index + 1] === "\n") index += 1;
    } else if (/\s/.test(char)) {
      if (current) tokens.push(current);
      current = "";
    } else if (";&|".includes(char)) {
      if (current) tokens.push(current);
      current = "";
      if (commandLine[index + 1] === char) tokens.push(char + commandLine[++index]);
      else tokens.push(char);
    } else {
      current += char;
    }
  }
  if (current) tokens.push(current);
  return tokens;
}

const SHELL_OPERATORS = new Set([";", "&&", "||", "|", "&"]);
const COMMAND_WRAPPERS = new Set(["command", "env", "nohup", "sudo", "time"]);
const SHELL_WRAPPERS = new Set(["bash", "cmd", "dash", "ksh", "powershell", "pwsh", "sh", "zsh"]);
const ENV_ASSIGNMENT = /^[A-Za-z_][A-Za-z0-9_]*=/;
const WRAPPER_OPTIONS_WITH_VALUE: Record<string, Set<string>> = {
  env: new Set(["-C", "--chdir", "-S", "--split-string", "-u", "--unset"]),
  sudo: new Set([
    "-C",
    "--close-from",
    "-g",
    "--group",
    "-h",
    "--host",
    "-p",
    "--prompt",
    "-R",
    "--chroot",
    "-r",
    "--role",
    "-T",
    "--command-timeout",
    "-t",
    "--type",
    "-u",
    "--user",
  ]),
  time: new Set(["-f", "--format", "-o", "--output"]),
};

function shellCommandSubstitutions(commandLine: string): string[] {
  const substitutions: string[] = [];
  let quote = "";
  for (let index = 0; index < commandLine.length; index += 1) {
    const char = commandLine[index];
    if (char === "\\") {
      index += 1;
      continue;
    }
    if (quote === "'") {
      if (char === "'") quote = "";
      continue;
    }
    if (char === "'" && !quote) {
      quote = char;
      continue;
    }
    if (char === '"') {
      quote = quote === '"' ? "" : '"';
      continue;
    }
    if (char === "`") {
      let end = index + 1;
      for (; end < commandLine.length; end += 1) {
        if (commandLine[end] === "\\") {
          end += 1;
        } else if (commandLine[end] === "`") {
          break;
        }
      }
      if (end < commandLine.length) {
        substitutions.push(commandLine.slice(index + 1, end));
        index = end;
      }
      continue;
    }
    if (char !== "$" || commandLine[index + 1] !== "(") {
      continue;
    }
    if (commandLine[index + 2] === "(") {
      index += 2;
      continue;
    }
    let depth = 1;
    let nestedQuote = "";
    let end = index + 2;
    for (; end < commandLine.length; end += 1) {
      const nestedChar = commandLine[end];
      if (nestedChar === "\\") {
        end += 1;
        continue;
      }
      if (nestedQuote) {
        if (nestedChar === nestedQuote) nestedQuote = "";
        continue;
      }
      if (nestedChar === "'" || nestedChar === '"') {
        nestedQuote = nestedChar;
      } else if (nestedChar === "(") {
        depth += 1;
      } else if (nestedChar === ")" && --depth === 0) {
        break;
      }
    }
    if (depth === 0) {
      substitutions.push(commandLine.slice(index + 2, end));
      index = end;
    }
  }
  return substitutions;
}

export function shellCommandBinaries(commandLine: string): string[] {
  const segments: string[][] = [[]];
  for (const token of shellTokens(commandLine)) {
    if (SHELL_OPERATORS.has(token)) {
      if (segments.at(-1)?.length) segments.push([]);
    } else {
      segments.at(-1)!.push(token);
    }
  }
  const binaries: string[] = [];
  for (const segment of segments) {
    let index = 0;
    while (index < segment.length && ENV_ASSIGNMENT.test(segment[index])) index += 1;
    while (index < segment.length && COMMAND_WRAPPERS.has(normalizedCommandName(segment[index]))) {
      const wrapper = normalizedCommandName(segment[index]);
      index += 1;
      while (index < segment.length) {
        const token = segment[index];
        if (token === "--") {
          index += 1;
          break;
        }
        if (ENV_ASSIGNMENT.test(token)) {
          index += 1;
          continue;
        }
        const optionName = token.split("=", 1)[0];
        if (token.startsWith("-")) {
          index += 1;
          if (
            !token.includes("=") &&
            WRAPPER_OPTIONS_WITH_VALUE[wrapper]?.has(optionName) &&
            index < segment.length
          ) {
            if (wrapper === "env" && ["-S", "--split-string"].includes(optionName)) {
              binaries.push(...shellCommandBinaries(segment[index]));
            }
            index += 1;
          }
          continue;
        }
        break;
      }
    }
    if (index >= segment.length) continue;
    const command = segment[index];
    binaries.push(command);
    if (SHELL_WRAPPERS.has(normalizedCommandName(command))) {
      for (let flagIndex = index + 1; flagIndex < segment.length - 1; flagIndex += 1) {
        if (["-c", "/c", "-command"].includes(segment[flagIndex].toLowerCase())) {
          binaries.push(...shellCommandBinaries(segment[flagIndex + 1]));
          break;
        }
      }
    }
  }
  for (const substitution of shellCommandSubstitutions(commandLine)) {
    binaries.push(...shellCommandBinaries(substitution));
  }
  return [...new Set(binaries)];
}

function matchesDomain(hostname: string, patterns: string[] | undefined): boolean {
  if (!patterns?.length) {
    return true;
  }
  const normalized = hostname.toLowerCase().replace(/^\[|\]$/g, "");
  return patterns.some((pattern) => {
    const candidate = pattern.toLowerCase();
    if (candidate.startsWith("*.")) {
      const suffix = candidate.slice(2);
      return normalized === suffix || normalized.endsWith(`.${suffix}`);
    }
    return normalized === candidate;
  });
}

function hashPolicyResource(resource: string): string {
  return createHash("sha256").update(resource).digest("hex").slice(0, 16);
}

function safePolicyResource(
  input: ShellPolicyInput | HttpPolicyInput | ToolCallPolicyInput,
): string {
  if (input.scope === "shell") {
    return shellCommandBinaries(input.resource)
      .map((command) => normalizedCommandName(command))
      .filter(Boolean)
      .join(",");
  }
  if (input.scope === "tool_call") {
    return input.toolName;
  }
  return input.url.hostname.toLowerCase().replace(/^\[|\]$/g, "");
}

function emitPolicyDecision(decision: HeadroomToolPolicyDecision): void {
  const record = {
    event: "headroom_tool_policy_decision",
    version: decision.version,
    decision_id: decision.decisionId,
    authority: decision.authority,
    timestamp: new Date().toISOString(),
    agent: "opencode",
    tool_name: decision.scope,
    scope: decision.scope,
    action: decision.action,
    effective_action: decision.effectiveAction,
    mode: decision.mode,
    matched_rule: decision.matchedRuleId ?? "",
    reason: decision.reason ?? "",
    request_hash: decision.requestHash,
    resource: decision.resource,
    source: decision.source,
    binding: decision.binding,
  };
  try {
    process.stderr.write(`${JSON.stringify(record)}\n`);
  } catch {
    // Never let logging break the transport.
  }

  try {
    if (processIsStateless()) {
      return;
    }
    const target = path.join(workspaceDir(), "tool_policy_audit.jsonl");
    fs.mkdirSync(path.dirname(target), { recursive: true });
    fs.appendFileSync(target, `${JSON.stringify(record)}\n`, "utf8");
  } catch {
    // Auditing is best effort and must not affect enforcement.
  }
}

function emitPolicyAcknowledgement(
  acknowledgement: HeadroomToolPolicyAcknowledgement,
): void {
  const record = {
    ...acknowledgement,
    decision_id: acknowledgement.decisionId,
    request_hash: acknowledgement.requestHash,
    decisionId: undefined,
    requestHash: undefined,
  };
  try {
    process.stderr.write(`${JSON.stringify(record)}\n`);
  } catch {
    // Enforcement remains authoritative if diagnostic output is unavailable.
  }
  try {
    if (processIsStateless()) return;
    const target = path.join(workspaceDir(), "tool_policy_audit.jsonl");
    fs.mkdirSync(path.dirname(target), { recursive: true });
    fs.appendFileSync(target, `${JSON.stringify(record)}\n`, "utf8");
  } catch {
    // The adapter has already enforced the decision; auditing is best effort.
  }
}

function policyDecisionId(
  requestHash: string,
  action: ToolPolicyAction,
  authority: "authoritative" | "advisory",
  binding?: HeadroomToolPolicyBinding,
): string {
  return createHash("sha256")
    .update(stableJson({ version: 1, requestHash, action, authority, binding }))
    .update(binding ? randomUUID() : "")
    .digest("hex");
}

function evaluatePolicy(
  policy: CompiledToolPolicy | undefined,
  input: ShellPolicyInput | HttpPolicyInput | ToolCallPolicyInput,
  authority: "authoritative" | "advisory" = "advisory",
  binding?: HeadroomToolPolicyBinding,
): HeadroomToolPolicyDecision | undefined {
  if (!policy) {
    return undefined;
  }
  const hasDynamicShellExecution =
    input.scope === "shell" &&
    policy.rules.some(
      (rule) =>
        rule.action !== "allow" && (rule.scope === "shell" || rule.scope === "tool_call"),
    ) &&
    /(?:\\[A-Za-z0-9]|\$(?:\{|[A-Za-z_('" ])|`|%[^%\r\n]+%|(?:^|[\s;&|])%[A-Za-z]|![A-Za-z_][A-Za-z0-9_]*!|\b(?:eval|exec|source|invoke-expression|iex|get-command|call|if|then|else|elif|fi|for|while|until|do|done|case|esac|select|function|coproc)\b|[<>]\s*\(|(?:^|[;&|]\s*)&\s*\(|[{}])/i.test(
      input.resource,
    );
  const matchedRule = policy.rules.find((rule) => {
        if (
          (input.scope === "tool_call" && rule.scope !== "tool_call") ||
          (input.scope !== "tool_call" &&
            rule.scope !== "tool_call" &&
            rule.scope !== input.scope)
        ) {
          return false;
        }
        if (rule.tools?.length) {
          if (
            !("toolName" in input) ||
            !rule.tools.includes((input.toolName ?? "").trim().toLowerCase())
          ) {
            return false;
          }
        }
        if (input.scope === "shell") {
          const commands = shellCommandBinaries(input.resource);
          if (rule.commands?.length) {
            const commandsMatch =
              rule.action === "allow"
                ? commands.length > 0 &&
                  commands.every((candidate) => commandMatches(candidate, rule.commands))
                : commands.some((candidate) => commandMatches(candidate, rule.commands));
            if (!commandsMatch) {
              return false;
            }
          }
          if (rule.argsPattern && !rule.argsPattern.test(input.argsText)) {
            return false;
          }
          if (rule.cwdPattern && !rule.cwdPattern.test(input.cwd ?? "")) {
            return false;
          }
          if (
            rule.envKeys?.length &&
            !rule.envKeys.every((entry) =>
              Object.keys(input.env ?? process.env).some(
                (key) => key.toLowerCase() === entry,
              ),
            )
          ) {
            return false;
          }
          return true;
        }
        if (input.scope === "tool_call") {
          if (rule.commands?.length) {
            return false;
          }
          if (rule.argsPattern && !rule.argsPattern.test(input.argsText)) {
            return false;
          }
          if (rule.cwdPattern && !rule.cwdPattern.test(input.cwd ?? "")) {
            return false;
          }
          if (
            rule.envKeys?.length &&
            !rule.envKeys.every((entry) =>
              Object.keys(input.env ?? process.env).some(
                (key) => key.toLowerCase() === entry,
              ),
            )
          ) {
            return false;
          }
          return true;
        }
        if (rule.tools?.length) {
          return false;
        }
        if (!matchesDomain(input.url.hostname, rule.domains)) {
          return false;
        }
        if (rule.urlPattern && !rule.urlPattern.test(input.url.href)) {
          return false;
        }
        return true;
    });
  const dynamicShellDenied =
    hasDynamicShellExecution && (!matchedRule || matchedRule.action === "allow");
  const action = dynamicShellDenied
    ? "deny"
    : (matchedRule?.action ?? policy.defaultAction);
  const effectiveAction = policy.mode === "report_only" && action !== "allow" ? "allow" : action;
  const requestHash = hashPolicyResource(input.resource);
  return {
    version: 1,
    decisionId: policyDecisionId(requestHash, action, authority, binding),
    authority,
    scope: input.scope,
    action,
    effectiveAction,
    mode: policy.mode,
    matchedRuleId: matchedRule?.id,
    reason: dynamicShellDenied
      ? "dynamic or escaped shell execution cannot be safely authorized"
      : matchedRule?.reason,
    resource: safePolicyResource(input),
    requestHash,
    source: policy.source,
    binding,
  };
}

function enforcePolicy(
  policy: CompiledToolPolicy | undefined,
  input: ShellPolicyInput | HttpPolicyInput | ToolCallPolicyInput,
  authority: "authoritative" | "advisory" = "advisory",
  binding?: HeadroomToolPolicyBinding,
): HeadroomToolPolicyPreflight | undefined {
  const state = getState();
  if (state?.policyUnavailable) {
    throw new Error(`[headroom] Tool policy unavailable; failing closed: ${state.policyUnavailable}`);
  }
  if (policy?.validUntil !== undefined && Date.now() / 1000 >= policy.validUntil) {
    throw new Error("[headroom] Remote tool policy expired; failing closed until it is refreshed");
  }
  const decision = evaluatePolicy(policy, input, authority, binding);
  if (!decision) {
    return;
  }

  emitPolicyDecision(decision);
  const blockedAcknowledgement =
    decision.authority === "authoritative" &&
    decision.binding &&
    decision.effectiveAction !== "allow"
      ? {
          version: 1 as const,
          event: "headroom_tool_policy_enforcement_acknowledgement" as const,
          decisionId: decision.decisionId,
          authority: "authoritative" as const,
          effect: "blocked" as const,
          requestHash: decision.requestHash,
          binding: decision.binding,
          timestamp: new Date().toISOString(),
        }
      : undefined;
  if (decision.effectiveAction === "allow") {
    return { decision };
  }
  if (blockedAcknowledgement) emitPolicyAcknowledgement(blockedAcknowledgement);
  const suffix =
    (decision.matchedRuleId ? ` (rule=${decision.matchedRuleId})` : "") +
    (decision.reason ? `: ${decision.reason}` : "");
  if (decision.effectiveAction === "require_approval") {
    const message =
      `[headroom] Tool policy requires approval for ${decision.scope} target ${decision.resource}` +
      suffix +
      ". No approval handler is installed in the OpenCode transport yet.";
    if (blockedAcknowledgement) {
      throw new ToolPolicyEnforcementError(message, decision, blockedAcknowledgement);
    }
    throw new Error(message);
  }

  const message =
    `[headroom] Tool policy denied ${decision.scope} target ${decision.resource}` +
    suffix;
  if (blockedAcknowledgement) {
    throw new ToolPolicyEnforcementError(message, decision, blockedAcknowledgement);
  }
  throw new Error(message);
}

export async function enforceNativeToolExecution(
  toolName: string,
  args: Record<string, unknown>,
  cwd?: string,
  execution?: { sessionID: string; callID: string },
): Promise<HeadroomToolPolicyPreflight | undefined> {
  await refreshHeadroomToolPolicy();
  const state = getState();
  const canonicalArgs = stableJson(args);
  const binding = execution
    ? {
        caller: "opencode" as const,
        adapter: "tool.execute.before" as const,
        sessionID: execution.sessionID,
        taskID: execution.callID,
        callID: execution.callID,
        toolName,
        cwd,
        canonicalArgsHash: hashPolicyResource(canonicalArgs),
      }
    : undefined;
  return enforcePolicy(
    state?.toolPolicy,
    nativePolicyInput(toolName, args, cwd),
    binding ? "authoritative" : "advisory",
    binding,
  );
}

export function acknowledgeNativeToolExecution(
  preflight: HeadroomToolPolicyPreflight,
  toolName: string,
  args: Record<string, unknown>,
  cwd: string | undefined,
  execution: { sessionID: string; callID: string },
): HeadroomToolPolicyAcknowledgement {
  const binding = preflight.decision.binding;
  const canonicalArgsHash = hashPolicyResource(stableJson(args));
  if (
    preflight.decision.authority !== "authoritative" ||
    !binding ||
    binding.caller !== "opencode" ||
    binding.adapter !== "tool.execute.before" ||
    binding.sessionID !== execution.sessionID ||
    binding.callID !== execution.callID ||
    binding.taskID !== execution.callID ||
    binding.toolName !== toolName ||
    binding.cwd !== cwd ||
    binding.canonicalArgsHash !== canonicalArgsHash
  ) {
    throw new Error(
      "[headroom] OpenCode execution acknowledgement did not match the bound preflight decision",
    );
  }
  const acknowledgement: HeadroomToolPolicyAcknowledgement = {
    version: 1,
    event: "headroom_tool_policy_enforcement_acknowledgement",
    decisionId: preflight.decision.decisionId,
    authority: "authoritative",
    effect: "allowed",
    requestHash: preflight.decision.requestHash,
    binding,
    timestamp: new Date().toISOString(),
  };
  emitPolicyAcknowledgement(acknowledgement);
  return acknowledgement;
}

export function acknowledgeUnknownNativeToolExecution(
  preflight: HeadroomToolPolicyPreflight,
  reason: NonNullable<HeadroomToolPolicyAcknowledgement["reason"]>,
): HeadroomToolPolicyAcknowledgement {
  const binding = preflight.decision.binding;
  if (
    preflight.decision.authority !== "authoritative" ||
    preflight.decision.effectiveAction !== "allow" ||
    !binding
  ) {
    throw new Error(
      "[headroom] Cannot record an unknown outcome for an unbound or blocked preflight",
    );
  }
  const acknowledgement: HeadroomToolPolicyAcknowledgement = {
    version: 1,
    event: "headroom_tool_policy_enforcement_acknowledgement",
    decisionId: preflight.decision.decisionId,
    authority: "authoritative",
    effect: "unknown",
    reason,
    requestHash: preflight.decision.requestHash,
    binding,
    timestamp: new Date().toISOString(),
  };
  emitPolicyAcknowledgement(acknowledgement);
  return acknowledgement;
}

export function evaluateNativeToolPolicy(
  policy: HeadroomToolPolicyConfig,
  toolName: string,
  args: Record<string, unknown>,
  cwd?: string,
): HeadroomToolPolicyDecision {
  const compiled = compileToolPolicy(policy, cwd, "explicit")!;
  return evaluatePolicy(compiled, nativePolicyInput(toolName, args, cwd))!;
}

function nativePolicyInput(
  toolName: string,
  args: Record<string, unknown>,
  cwd?: string,
): ShellPolicyInput | ToolCallPolicyInput {
  const stableArgs = stableJson(args);
  if (!["bash", "shell", "powershell", "sh"].includes(toolName.toLowerCase())) {
    return {
      scope: "tool_call",
      resource: `${toolName} ${stableArgs}`,
      toolName,
      argsText: stableArgs,
      cwd,
      env: isOptions(args.env) ? args.env : undefined,
    };
  }
  const commandLine = typeof args.command === "string" ? args.command : "";
  return {
    scope: "shell",
    resource: commandLine,
    command: shellCommandBinaries(commandLine)[0] ?? "",
    argsText: commandLine,
    cwd,
    env: isOptions(args.env) ? args.env : undefined,
    toolName,
  };
}

function stableJson(value: unknown): string {
  if (Array.isArray(value)) {
    return `[${value.map((entry) => stableJson(entry)).join(",")}]`;
  }
  if (value && typeof value === "object") {
    return `{${Object.entries(value as Record<string, unknown>)
      .sort(([left], [right]) => left.localeCompare(right))
      .map(([key, entry]) => `${JSON.stringify(key)}:${stableJson(entry)}`)
      .join(",")}}`;
  }
  return JSON.stringify(value) ?? "null";
}

function effectiveChildCwd(value: unknown): string {
  if (typeof value === "string") {
    return path.resolve(process.cwd(), value);
  }
  if (value instanceof URL && value.protocol === "file:") {
    return path.resolve(fileURLToPath(value));
  }
  return process.cwd();
}

function wrapSpawn(originalSpawn: ChildSpawn): ChildSpawn {
  return function headroomSpawn(this: unknown, ...args: unknown[]) {
    const state = getState();
    if (!state) {
      return Reflect.apply(originalSpawn, this, args);
    }
    const command = String(args[0] ?? "");
    const commandArgs = Array.isArray(args[1]) ? args[1].map((entry) => String(entry)) : [];
    const resource = [command, ...commandArgs].join(" ").trim() || command;
    const options = isOptions(args[Array.isArray(args[1]) ? 2 : 1])
      ? (args[Array.isArray(args[1]) ? 2 : 1] as Record<string, unknown>)
      : undefined;
    enforcePolicy(state.toolPolicy, {
      scope: "shell",
      resource,
      command,
      argsText: resource,
      cwd: effectiveChildCwd(options?.cwd),
      env: options?.env as NodeJS.ProcessEnv | Record<string, unknown> | undefined,
    });
    const optionIndex = Array.isArray(args[1]) ? 2 : 1;
    return Reflect.apply(originalSpawn, this, injectOptionsEnv(args, optionIndex, state.proxyUrl));
  } as ChildSpawn;
}

function wrapExec(originalExec: ChildExec): ChildExec {
  return function headroomExec(this: unknown, ...args: unknown[]) {
    const state = getState();
    if (!state) {
      return Reflect.apply(originalExec, this, args);
    }
    const commandLine = String(args[0] ?? "");
    const options = isOptions(args[1]) ? (args[1] as Record<string, unknown>) : undefined;
    enforcePolicy(state.toolPolicy, {
      scope: "shell",
      resource: commandLine,
      command: shellCommandBinaries(commandLine)[0] ?? "",
      argsText: commandLine,
      cwd: effectiveChildCwd(options?.cwd),
      env: options?.env as NodeJS.ProcessEnv | Record<string, unknown> | undefined,
    });
    return Reflect.apply(originalExec, this, injectOptionsEnv(args, 1, state.proxyUrl));
  } as ChildExec;
}

function wrapExecFile(originalExecFile: ChildExecFile): ChildExecFile {
  return function headroomExecFile(this: unknown, ...args: unknown[]) {
    const state = getState();
    if (!state) {
      return Reflect.apply(originalExecFile, this, args);
    }
    const command = String(args[0] ?? "");
    const commandArgs = Array.isArray(args[1]) ? args[1].map((entry) => String(entry)) : [];
    const resource = [command, ...commandArgs].join(" ").trim() || command;
    const options = isOptions(args[Array.isArray(args[1]) ? 2 : 1])
      ? (args[Array.isArray(args[1]) ? 2 : 1] as Record<string, unknown>)
      : undefined;
    enforcePolicy(state.toolPolicy, {
      scope: "shell",
      resource,
      command,
      argsText: resource,
      cwd: effectiveChildCwd(options?.cwd),
      env: options?.env as NodeJS.ProcessEnv | Record<string, unknown> | undefined,
    });
    const optionIndex = Array.isArray(args[1]) ? 2 : 1;
    return Reflect.apply(originalExecFile, this, injectOptionsEnv(args, optionIndex, state.proxyUrl));
  } as ChildExecFile;
}

function wrapFork(originalFork: ChildFork): ChildFork {
  return function headroomFork(this: unknown, ...args: unknown[]) {
    const state = getState();
    if (!state) {
      return Reflect.apply(originalFork, this, args);
    }
    const command = String(args[0] ?? "");
    const commandArgs = Array.isArray(args[1]) ? args[1].map((entry) => String(entry)) : [];
    const resource = [command, ...commandArgs].join(" ").trim() || command;
    const options = isOptions(args[Array.isArray(args[1]) ? 2 : 1])
      ? (args[Array.isArray(args[1]) ? 2 : 1] as Record<string, unknown>)
      : undefined;
    enforcePolicy(state.toolPolicy, {
      scope: "shell",
      resource,
      command,
      argsText: resource,
      cwd: effectiveChildCwd(options?.cwd),
      env: options?.env as NodeJS.ProcessEnv | Record<string, unknown> | undefined,
    });
    const optionIndex = Array.isArray(args[1]) ? 2 : 1;
    return Reflect.apply(originalFork, this, injectOptionsEnv(args, optionIndex, state.proxyUrl));
  } as ChildFork;
}

function normalizeProxyUrl(proxyUrl: string): URL {
  return new URL(proxyUrl);
}

function isLoopback(hostname: string): boolean {
  const normalized = hostname.toLowerCase().replace(/^\[|\]$/g, "");
  return normalized === "localhost" || normalized === "127.0.0.1" || normalized === "::1";
}

function shouldRoute(url: URL, proxy: URL): boolean {
  if (url.protocol !== "http:" && url.protocol !== "https:") {
    return false;
  }
  if (isLoopback(url.hostname)) {
    return false;
  }
  if (url.origin === proxy.origin) {
    return false;
  }
  return true;
}

function routedUrl(upstream: URL, proxy: URL): URL {
  return new URL(`${upstream.pathname}${upstream.search}`, proxy.origin);
}

function normalizedOpenAiProxyPath(pathname: string): string | undefined {
  if (pathname.endsWith("/chat/completions")) {
    return "/v1/chat/completions";
  }
  if (pathname.endsWith("/responses")) {
    return "/v1/responses";
  }
  return undefined;
}

function routedUrlForOpenCode(upstream: URL, proxy: URL): { url: URL; originalPath: string | undefined } {
  const normalizedPath = normalizedOpenAiProxyPath(upstream.pathname);
  if (!normalizedPath) {
    return {
      url: routedUrl(upstream, proxy),
      originalPath: undefined,
    };
  }

  return {
    url: new URL(`${normalizedPath}${upstream.search}`, proxy.origin),
    originalPath: upstream.pathname,
  };
}

function requestUrl(input: RequestInfo | URL): URL {
  if (input instanceof Request) {
    return new URL(input.url);
  }
  if (input instanceof URL) {
    return input;
  }
  return new URL(String(input));
}

function mergeFetchHeaders(
  input: RequestInfo | URL,
  init: RequestInit | undefined,
  upstream: URL | undefined,
  originalPath: string | undefined = undefined,
  project: string | undefined = undefined,
): Headers {
  const headers = new Headers(input instanceof Request ? input.headers : undefined);
  if (init?.headers) {
    new Headers(init.headers).forEach((value, key) => headers.set(key, value));
  }
  if (upstream) {
    headers.set(BASE_URL_HEADER, upstream.origin);
    headers.delete("host");
  }
  if (originalPath) {
    headers.set(ORIGINAL_PATH_HEADER, originalPath);
  }
  if (project) {
    headers.set(PROJECT_HEADER, project);
  }
  return headers;
}

function withRoutedFetchInput(input: RequestInfo | URL, init: RequestInit | undefined, proxy: URL, project: string | undefined): FetchArgs {
  const upstream = requestUrl(input);
  if (!shouldRoute(upstream, proxy)) {
    return [input, init];
  }

  const { url: nextUrl, originalPath } = routedUrlForOpenCode(upstream, proxy);
  const nextInit = {
    ...init,
    headers: mergeFetchHeaders(input, init, upstream, originalPath, project),
  };

  if (input instanceof Request) {
    return [new Request(nextUrl, input), nextInit];
  }
  return [nextUrl, nextInit];
}

function splitNodeArgs(args: unknown[]): NodeRequestParts {
  const callback = typeof args.at(-1) === "function" ? (args.at(-1) as (...args: unknown[]) => unknown) : undefined;
  const withoutCallback = callback ? args.slice(0, -1) : args;
  const [first, second] = withoutCallback;
  const options = typeof second === "object" && second !== null ? { ...(second as Record<string, unknown>) } : {};

  if (first instanceof URL) {
    return { url: first, options, callback };
  }
  if (typeof first === "string") {
    try {
      return { url: new URL(first), options, callback };
    } catch {
      return { options, callback };
    }
  }
  if (typeof first === "object" && first !== null) {
    const requestOptions = { ...(first as Record<string, unknown>), ...options };
    return { url: urlFromRequestOptions(requestOptions), options: requestOptions, callback };
  }
  return { options, callback };
}

function urlFromRequestOptions(options: Record<string, unknown>): URL | undefined {
  const protocol = String(options.protocol ?? "http:");
  if (protocol !== "http:" && protocol !== "https:") {
    return undefined;
  }

  const hostValue = options.hostname ?? options.host;
  if (!hostValue) {
    return undefined;
  }

  const hostname = String(hostValue).replace(/:\d+$/, "");
  const port = options.port ? `:${String(options.port)}` : "";
  const path = String(options.path ?? "/");
  try {
    return new URL(`${protocol}//${hostname}${port}${path}`);
  } catch {
    return undefined;
  }
}

function headersForNodeRequest(
  options: Record<string, unknown>,
  upstream: URL,
  originalPath: string | undefined,
  project: string | undefined,
): Record<string, string> {
  const headers = new Headers(options.headers as HeadersInit | undefined);
  headers.set(BASE_URL_HEADER, upstream.origin);
  if (originalPath) {
    headers.set(ORIGINAL_PATH_HEADER, originalPath);
  }
  if (project) {
    headers.set(PROJECT_HEADER, project);
  }
  headers.delete("host");

  const result: Record<string, string> = {};
  headers.forEach((value, key) => {
    result[key] = value;
  });
  return result;
}

function routedNodeOptions(parts: NodeRequestParts, proxy: URL, project: string | undefined): Record<string, unknown> | undefined {
  if (!parts.url || !shouldRoute(parts.url, proxy)) {
    return undefined;
  }

  const { url: nextUrl, originalPath } = routedUrlForOpenCode(parts.url, proxy);
  const {
    agent: _agent,
    auth: _auth,
    createConnection: _createConnection,
    defaultPort: _defaultPort,
    family: _family,
    headers: _headers,
    host: _host,
    hostname: _hostname,
    href: _href,
    lookup: _lookup,
    path: _path,
    pathname: _pathname,
    port: _port,
    protocol: _protocol,
    search: _search,
    servername: _servername,
    setHost: _setHost,
    ...rest
  } = parts.options;

  return {
    ...rest,
    protocol: nextUrl.protocol,
    hostname: nextUrl.hostname,
    port: nextUrl.port || undefined,
    path: `${nextUrl.pathname}${nextUrl.search}`,
    headers: headersForNodeRequest(parts.options, parts.url, originalPath, project),
  };
}

function wrapRequest(
  originalHttpRequest: HttpRequest,
  originalHttpsRequest: HttpsRequest,
  originalRequest: HttpRequest | HttpsRequest,
): HttpRequest | HttpsRequest {
  return function headroomRequest(this: unknown, ...args: unknown[]) {
    const state = getState();
    if (!state) {
      return Reflect.apply(originalRequest, this, args);
    }

    const proxy = normalizeProxyUrl(state.proxyUrl);
    const parts = splitNodeArgs(args);
    if (parts.url) {
      enforcePolicy(state.toolPolicy, {
        scope: "http",
        resource: parts.url.href,
        url: parts.url,
      });
    }
    const nextOptions = routedNodeOptions(parts, proxy, state.project);
    if (!nextOptions) {
      return Reflect.apply(originalRequest, this, args);
    }

    const targetRequest = proxy.protocol === "https:" ? originalHttpsRequest : originalHttpRequest;
    const nextArgs = parts.callback ? [nextOptions, parts.callback] : [nextOptions];
    return Reflect.apply(targetRequest, this, nextArgs);
  } as HttpRequest | HttpsRequest;
}

function wrapGet(request: HttpRequest | HttpsRequest): HttpGet | HttpsGet {
  return function headroomGet(this: unknown, ...args: unknown[]) {
    const req = Reflect.apply(request, this, args);
    req.end();
    return req;
  } as HttpGet | HttpsGet;
}

function wrapHttp2Connect(originalConnect: Http2Connect): Http2Connect {
  return function headroomHttp2Connect(this: unknown, authority: string | URL, ...args: unknown[]) {
    const state = getState();
    if (state) {
      const proxy = normalizeProxyUrl(state.proxyUrl);
      const upstream = authority instanceof URL ? authority : new URL(String(authority));
      enforcePolicy(state.toolPolicy, {
        scope: "http",
        resource: upstream.href,
        url: upstream,
      });
      if (shouldRoute(upstream, proxy)) {
        throw new Error(
          `Headroom OpenCode wrap blocked direct HTTP/2 connection to ${upstream.hostname}. ` +
            "Use fetch, http, or https so traffic can be routed through Headroom.",
        );
      }
    }
    return Reflect.apply(originalConnect, this, [authority, ...args]);
  } as Http2Connect;
}

export function installHeadroomTransport(options: InstallOptions): () => void {
  const existing = getState();
  const remotePolicyUrl =
    options.toolPolicy === undefined
      ? process.env[TOOL_POLICY_URL_ENV]?.trim() || undefined
      : undefined;
  const remotePolicyToken =
    options.toolPolicy === undefined
      ? process.env[TOOL_POLICY_TOKEN_ENV]?.trim() ?? ""
      : "";
  const policyContextKey = createHash("sha256")
    .update(
      stableJson({
        policyProject: options.policyProject ?? options.project,
        remotePolicyUrl,
        remotePolicyToken,
      }),
    )
    .digest("hex");
  let toolPolicy: CompiledToolPolicy | undefined;
  let policyUnavailable: string | undefined;
  try {
    toolPolicy = compileToolPolicy(options.toolPolicy, options.policyProject ?? options.project);
  } catch (error) {
    policyUnavailable = error instanceof Error ? error.message : String(error);
  }
  if (remotePolicyUrl && !toolPolicy) {
    policyUnavailable ??= "remote Headroom tool policy has not been loaded";
  }
  if (existing) {
    const existingPolicy = existing.toolPolicy?.serialized;
    const nextPolicy = toolPolicy?.serialized;
    if (
      existingPolicy !== nextPolicy ||
      existing.policyUnavailable !== policyUnavailable ||
      existing.policyContextKey !== policyContextKey
    ) {
      throw new Error(
        "[headroom] Multiple OpenCode workspaces with different tool policies share one " +
          "process; refusing to replace the active policy",
      );
    }
    existing.refs += 1;
    existing.proxyUrl = options.proxyUrl;
    existing.project = options.project;
    existing.debug = Boolean(options.debug);
    installProcessEnv(options.proxyUrl, toolPolicy);
    return () => uninstallHeadroomTransport();
  }

  const state: TransportState = {
    refs: 1,
    policyContextKey,
    proxyUrl: options.proxyUrl,
    project: options.project,
    debug: Boolean(options.debug),
    toolPolicy,
    toolPolicyInput: options.toolPolicy,
    remotePolicyUrl,
    remotePolicyToken,
    policyUnavailable,
    previousNodeOptions: process.env.NODE_OPTIONS,
    previousProxyUrlEnv: process.env[PROXY_ENV],
    previousToolPolicyEnv: process.env[TOOL_POLICY_ENV],
    originalFetch: globalThis.fetch,
    originalHttpRequest: http.request,
    originalHttpGet: http.get,
    originalHttpsRequest: https.request,
    originalHttpsGet: https.get,
    originalHttp2Connect: http2.connect,
    originalChildSpawn: childProcess.spawn,
    originalChildExec: childProcess.exec,
    originalChildExecFile: childProcess.execFile,
    originalChildFork: childProcess.fork,
  };

  setState(state);
  installProcessEnv(options.proxyUrl, toolPolicy);
  globalThis.fetch = async (...args: FetchArgs) => {
    const current = getState();
    if (!current) {
      return state.originalFetch(...args);
    }
    await refreshHeadroomToolPolicy();
    const upstream = requestUrl(args[0]);
    enforcePolicy(current.toolPolicy, {
      scope: "http",
      resource: upstream.href,
      url: upstream,
    });
    const proxy = normalizeProxyUrl(current.proxyUrl);
    const [nextInput, nextInit] = withRoutedFetchInput(args[0], args[1], proxy, current.project);
    return state.originalFetch(nextInput, nextInit);
  };

  http.request = wrapRequest(state.originalHttpRequest, state.originalHttpsRequest, state.originalHttpRequest) as HttpRequest;
  https.request = wrapRequest(state.originalHttpRequest, state.originalHttpsRequest, state.originalHttpsRequest) as HttpsRequest;
  http.get = wrapGet(http.request) as HttpGet;
  https.get = wrapGet(https.request) as HttpsGet;
  http2.connect = wrapHttp2Connect(state.originalHttp2Connect);
  childProcess.spawn = wrapSpawn(state.originalChildSpawn);
  childProcess.exec = wrapExec(state.originalChildExec);
  childProcess.execFile = wrapExecFile(state.originalChildExecFile);
  childProcess.fork = wrapFork(state.originalChildFork);
  syncBuiltinESMExports();

  return () => uninstallHeadroomTransport();
}

export function uninstallHeadroomTransport(): void {
  const state = getState();
  if (!state) {
    return;
  }

  state.refs -= 1;
  if (state.refs > 0) {
    return;
  }

  globalThis.fetch = state.originalFetch;
  http.request = state.originalHttpRequest;
  http.get = state.originalHttpGet;
  https.request = state.originalHttpsRequest;
  https.get = state.originalHttpsGet;
  http2.connect = state.originalHttp2Connect;
  childProcess.spawn = state.originalChildSpawn;
  childProcess.exec = state.originalChildExec;
  childProcess.execFile = state.originalChildExecFile;
  childProcess.fork = state.originalChildFork;
  syncBuiltinESMExports();
  if (state.previousNodeOptions === undefined) {
    delete process.env.NODE_OPTIONS;
  } else {
    process.env.NODE_OPTIONS = state.previousNodeOptions;
  }
  if (state.previousProxyUrlEnv === undefined) {
    delete process.env[PROXY_ENV];
  } else {
    process.env[PROXY_ENV] = state.previousProxyUrlEnv;
  }
  if (state.previousToolPolicyEnv === undefined) {
    delete process.env[TOOL_POLICY_ENV];
  } else {
    process.env[TOOL_POLICY_ENV] = state.previousToolPolicyEnv;
  }
  setState(undefined);
}
