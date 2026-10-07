// src/transport.ts
import { createRequire, syncBuiltinESMExports } from "module";
import os from "os";
import path from "path";
import { createHash, randomUUID } from "crypto";
import { fileURLToPath } from "url";
var nodeRequire = createRequire(import.meta.url);
var http = nodeRequire("node:http");
var https = nodeRequire("node:https");
var http2 = nodeRequire("node:http2");
var childProcess = nodeRequire("node:child_process");
var fs = nodeRequire("node:fs");
var BASE_URL_HEADER = "x-headroom-base-url";
var ORIGINAL_PATH_HEADER = "x-headroom-original-path";
var PROJECT_HEADER = "x-headroom-project";
var PROXY_ENV = "HEADROOM_OPENCODE_TRANSPORT_PROXY_URL";
var EXCLUDE_HOSTS_ENV = "HEADROOM_OPENCODE_EXCLUDE_HOSTS";
var TOOL_POLICY_ENV = "HEADROOM_TOOL_POLICY_JSON";
var TOOL_POLICY_PATH_ENV = "HEADROOM_TOOL_POLICY_PATH";
var TOOL_POLICY_URL_ENV = "HEADROOM_TOOL_POLICY_URL";
var TOOL_POLICY_TOKEN_ENV = "HEADROOM_TOOL_POLICY_TOKEN";
var TOOL_POLICY_REFRESH_SECONDS_ENV = "HEADROOM_TOOL_POLICY_REFRESH_SECONDS";
var TOOL_POLICY_VALID_UNTIL_ENV = "HEADROOM_INTERNAL_TOOL_POLICY_VALID_UNTIL";
var TOOL_POLICY_FILE_NAME = "tool_policy.json";
var POLICY_VERSION = 1;
var DEFAULT_REFRESH_SECONDS = 300;
var MAX_REFRESH_SECONDS = 3600;
var REMOTE_TIMEOUT_MS = 5e3;
var MAX_REMOTE_POLICY_BYTES = 1024 * 1024;
var STATE_KEY = /* @__PURE__ */ Symbol.for("headroom.opencode.transport");
var ToolPolicyEnforcementError = class extends Error {
  decision;
  acknowledgement;
  constructor(message, decision, acknowledgement) {
    super(message);
    this.name = "ToolPolicyEnforcementError";
    this.decision = decision;
    this.acknowledgement = acknowledgement;
  }
};
function getState() {
  return globalThis[STATE_KEY];
}
function setState(state) {
  globalThis[STATE_KEY] = state;
}
function shimImportSpecifier() {
  const shim = new URL("../hook-shim/handler.js", import.meta.url);
  return fs.existsSync(shim) ? shim.href : void 0;
}
function withNodeImportOption(existing, shim) {
  const parts = existing?.trim() ? existing.trim().split(/\s+/) : [];
  const alreadyPresent = parts.some((part, index) => {
    return part === `--import=${shim}` || part === "--import" && parts[index + 1] === shim;
  });
  if (!alreadyPresent) {
    parts.push(`--import=${shim}`);
  }
  return parts.join(" ");
}
function parseToolPolicyJson(raw, source) {
  try {
    const parsed = JSON.parse(raw);
    if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) {
      throw new Error("expected a JSON object");
    }
    if (parsed.version !== void 0 && parsed.version !== POLICY_VERSION) {
      throw new Error(`unsupported version ${String(parsed.version)}; expected ${POLICY_VERSION}`);
    }
    return parsed;
  } catch {
    throw new Error(`Invalid Headroom tool policy JSON in ${source}`);
  }
}
function readToolPolicyFile(filePath, source) {
  try {
    return parseToolPolicyJson(fs.readFileSync(filePath, "utf8"), source);
  } catch (error) {
    if (error instanceof Error && error.message.startsWith("Invalid Headroom tool policy JSON")) {
      throw error;
    }
    throw new Error(`Invalid Headroom tool policy file ${filePath} (${source}): ${String(error)}`);
  }
}
function defaultGlobalToolPolicyPath() {
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
function findLocalToolPolicyPath(project) {
  let start = path.resolve(project || process.cwd());
  try {
    if (fs.statSync(start).isFile()) {
      start = path.dirname(start);
    }
  } catch {
  }
  let current = start;
  while (true) {
    const candidate = path.join(current, ".headroom", TOOL_POLICY_FILE_NAME);
    if (fs.existsSync(candidate)) {
      return candidate;
    }
    const parent = path.dirname(current);
    if (parent === current) {
      return void 0;
    }
    current = parent;
  }
}
function loadToolPolicyConfig(policy, project) {
  if (policy === void 0) {
    const raw = process.env[TOOL_POLICY_ENV]?.trim();
    if (raw) {
      return parseToolPolicyJson(raw, TOOL_POLICY_ENV);
    }
    const rawPath = process.env[TOOL_POLICY_PATH_ENV]?.trim();
    if (rawPath) {
      return readToolPolicyFile(rawPath, TOOL_POLICY_PATH_ENV);
    }
    if (process.env[TOOL_POLICY_URL_ENV]?.trim()) {
      return void 0;
    }
    const globalPath = defaultGlobalToolPolicyPath();
    if (fs.existsSync(globalPath)) {
      return readToolPolicyFile(globalPath, globalPath);
    }
    const localPath = findLocalToolPolicyPath(project);
    if (localPath) {
      return readToolPolicyFile(localPath, localPath);
    }
    return void 0;
  }
  if (typeof policy !== "string") {
    return policy;
  }
  const trimmed = policy.trim();
  if (!trimmed) {
    return void 0;
  }
  if (trimmed.startsWith("{")) {
    return parseToolPolicyJson(trimmed, "inline string");
  }
  return readToolPolicyFile(trimmed, trimmed);
}
function compileRegex(source, field, ruleId) {
  if (!source) {
    return void 0;
  }
  try {
    return new RegExp(source);
  } catch {
    throw new Error(
      `Invalid Headroom tool policy regex for ${field} in rule ${ruleId}`
    );
  }
}
function asArray(value) {
  if (value === void 0) {
    return void 0;
  }
  return Array.isArray(value) ? value : [value];
}
function compileToolPolicy(policy, project, source = "configured") {
  const loaded = loadToolPolicyConfig(policy, project);
  if (!loaded) {
    return void 0;
  }
  if (!Array.isArray(loaded.rules)) {
    throw new Error("Headroom tool policy requires a rules array");
  }
  if (loaded.version !== void 0 && loaded.version !== POLICY_VERSION) {
    throw new Error(`Unsupported Headroom tool policy version: ${String(loaded.version)}`);
  }
  const compiledRules = loaded.rules.map((rule, index) => {
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
      urlPattern: compileRegex(rule.urlPattern, "urlPattern", id)
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
        id: rule.id?.trim() || `rule_${index + 1}`
      }))
    }),
    source
  };
}
function parsePropagatedValidUntil(value) {
  const parsed = Number(value);
  return Number.isFinite(parsed) && parsed > 0 ? parsed : void 0;
}
function policySourceFingerprint(value) {
  const serialized = typeof value === "string" ? value : stableJson(value);
  return createHash("sha256").update(serialized).digest("hex");
}
function resolveToolPolicySource(options, existing) {
  if (options.toolPolicy !== void 0) {
    return {
      input: options.toolPolicy,
      source: "configured",
      identity: `configured:${policySourceFingerprint(options.toolPolicy)}`,
      remoteToken: ""
    };
  }
  const installedPolicy = existing?.toolPolicy?.serialized;
  const currentPolicy = process.env[TOOL_POLICY_ENV]?.trim();
  const originalPolicy = existing?.previousToolPolicyEnv?.trim();
  const policyJson = existing && currentPolicy === installedPolicy && currentPolicy !== originalPolicy ? originalPolicy : currentPolicy;
  if (policyJson) {
    const propagatedExpiry = existing && currentPolicy === installedPolicy && currentPolicy !== originalPolicy ? existing.previousToolPolicyValidUntilEnv : process.env[TOOL_POLICY_VALID_UNTIL_ENV];
    return {
      input: policyJson,
      source: TOOL_POLICY_ENV,
      identity: `${TOOL_POLICY_ENV}:${policySourceFingerprint(policyJson)}`,
      remoteToken: "",
      validUntil: parsePropagatedValidUntil(propagatedExpiry)
    };
  }
  const configuredPath = process.env[TOOL_POLICY_PATH_ENV]?.trim();
  if (configuredPath) {
    return {
      input: configuredPath,
      source: TOOL_POLICY_PATH_ENV,
      identity: `${TOOL_POLICY_PATH_ENV}:${path.resolve(configuredPath)}`,
      remoteToken: ""
    };
  }
  const remoteUrl = process.env[TOOL_POLICY_URL_ENV]?.trim() || void 0;
  if (remoteUrl) {
    const remoteToken = process.env[TOOL_POLICY_TOKEN_ENV]?.trim() ?? "";
    return {
      source: TOOL_POLICY_URL_ENV,
      identity: `${TOOL_POLICY_URL_ENV}:${remoteUrl}`,
      remoteUrl,
      remoteToken
    };
  }
  const globalPath = defaultGlobalToolPolicyPath();
  if (fs.existsSync(globalPath)) {
    return {
      input: globalPath,
      source: globalPath,
      identity: `global:${path.resolve(globalPath)}`,
      remoteToken: ""
    };
  }
  const localPath = findLocalToolPolicyPath(options.policyProject ?? options.project);
  if (localPath) {
    return {
      input: localPath,
      source: localPath,
      identity: `repository:${path.resolve(localPath)}`,
      remoteToken: ""
    };
  }
  return { source: "unconfigured", identity: "unconfigured", remoteToken: "" };
}
function workspaceDir() {
  return process.env.HEADROOM_WORKSPACE_DIR?.trim() || path.join(os.homedir(), ".headroom");
}
function processIsStateless() {
  return ["1", "true", "yes", "on"].includes(
    process.env.HEADROOM_STATELESS?.trim().toLowerCase() ?? ""
  );
}
function toolPolicyRefreshSeconds(env = process.env) {
  const raw = env[TOOL_POLICY_REFRESH_SECONDS_ENV]?.trim();
  if (!raw || !/^\d+$/.test(raw)) {
    return DEFAULT_REFRESH_SECONDS;
  }
  const value = Number(raw);
  return Number.isSafeInteger(value) && value >= DEFAULT_REFRESH_SECONDS && value <= MAX_REFRESH_SECONDS ? value : DEFAULT_REFRESH_SECONDS;
}
function remoteToolPolicyCachePath(url, token = "") {
  const digest = createHash("sha256").update(`${url}\0${token}`).digest("hex");
  return path.join(workspaceDir(), "policy-cache", `${digest}.json`);
}
function isAllowedToolPolicyUrl(value) {
  let url;
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
  return hostname === "localhost" || hostname === "::1" || /^127(?:\.\d{1,3}){3}$/.test(hostname);
}
function readRemotePolicyCache(url, token) {
  if (processIsStateless()) {
    return void 0;
  }
  const cachePath = remoteToolPolicyCachePath(url, token);
  if (!fs.existsSync(cachePath)) {
    return void 0;
  }
  try {
    const parsed = JSON.parse(fs.readFileSync(cachePath, "utf8"));
    if (parsed.cache_version !== 2 || parsed.url_hash !== createHash("sha256").update(url).digest("hex") || typeof parsed.fetched_at !== "number" || !parsed.policy || typeof parsed.policy !== "object" || Array.isArray(parsed.policy)) {
      return void 0;
    }
    return parsed;
  } catch {
    return void 0;
  }
}
function writeRemotePolicyCache(url, cache, token) {
  if (processIsStateless()) {
    return;
  }
  const cachePath = remoteToolPolicyCachePath(url, token);
  const directory = path.dirname(cachePath);
  const temporaryPath = path.join(
    directory,
    `.${path.basename(cachePath)}.${process.pid}.${randomUUID()}.tmp`
  );
  fs.mkdirSync(directory, { recursive: true });
  try {
    fs.writeFileSync(temporaryPath, `${JSON.stringify(cache)}
`, {
      encoding: "utf8",
      flag: "wx"
    });
    fs.renameSync(temporaryPath, cachePath);
  } finally {
    try {
      fs.rmSync(temporaryPath, { force: true });
    } catch {
    }
  }
}
async function readLimitedResponseText(response) {
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
async function loadRemoteToolPolicy(url, token, originalFetch, now = Date.now() / 1e3) {
  const remoteLabel = new URL(url).hostname.toLowerCase().replace(/^\[|\]$/g, "");
  const cache = readRemotePolicyCache(url, token);
  const cacheAge = cache ? now - cache.fetched_at : void 0;
  if (cache && cacheAge !== void 0 && cacheAge >= 0 && cacheAge < toolPolicyRefreshSeconds()) {
    const compiled2 = compileToolPolicy(cache.policy, void 0, `remote-cache:${remoteLabel}`);
    compiled2.validUntil = cache.fetched_at + toolPolicyRefreshSeconds();
    return compiled2;
  }
  const headers = new Headers({ accept: "application/json" });
  if (token) {
    headers.set("authorization", `Bearer ${token}`);
  }
  if (cache?.etag) {
    headers.set("if-none-match", cache.etag);
  }
  let response;
  try {
    response = await originalFetch(url, {
      method: "GET",
      headers,
      redirect: "manual",
      signal: AbortSignal.timeout(REMOTE_TIMEOUT_MS)
    });
  } catch {
    throw new Error(`Headroom tool policy service ${remoteLabel} is unavailable`);
  }
  if (response.status === 304 && cache) {
    const compiled2 = compileToolPolicy(
      cache.policy,
      void 0,
      `remote-cache:${remoteLabel}`
    );
    const refreshed = { ...cache, fetched_at: now };
    writeRemotePolicyCache(url, refreshed, token);
    compiled2.validUntil = now + toolPolicyRefreshSeconds();
    return compiled2;
  }
  if (!response.ok) {
    throw new Error(`Headroom tool policy service ${remoteLabel} returned HTTP ${response.status}`);
  }
  const text = await readLimitedResponseText(response);
  const payload = parseToolPolicyJson(text, remoteLabel);
  const compiled = compileToolPolicy(payload, void 0, `remote:${remoteLabel}`);
  compiled.validUntil = now + toolPolicyRefreshSeconds();
  writeRemotePolicyCache(
    url,
    {
      cache_version: 2,
      url_hash: createHash("sha256").update(url).digest("hex"),
      etag: response.headers.get("etag") ?? "",
      fetched_at: now,
      policy: JSON.parse(compiled.serialized)
    },
    token
  );
  return compiled;
}
async function refreshHeadroomToolPolicy(now = Date.now() / 1e3) {
  const state = getState();
  if (!state || state.toolPolicyInput !== void 0) {
    return;
  }
  const url = state.remotePolicyUrl;
  if (!url) {
    return;
  }
  if (!isAllowedToolPolicyUrl(url)) {
    state.toolPolicy = void 0;
    state.policyUnavailable = `${TOOL_POLICY_URL_ENV} must use HTTPS; HTTP is allowed only for loopback hosts`;
    return;
  }
  try {
    state.toolPolicy = await loadRemoteToolPolicy(
      url,
      state.remotePolicyToken,
      state.originalFetch,
      now
    );
    state.policyUnavailable = void 0;
    installProcessEnv(state.proxyUrl, state.excludeHosts, state.toolPolicy);
  } catch (error) {
    state.toolPolicy = void 0;
    state.policyUnavailable = error instanceof Error ? error.message : String(error);
  }
}
function withShimEnv(env, proxyUrl2, excludeHosts, toolPolicy) {
  const nextEnv = { ...env ?? process.env };
  delete nextEnv[TOOL_POLICY_TOKEN_ENV];
  delete nextEnv[TOOL_POLICY_URL_ENV];
  nextEnv[PROXY_ENV] = proxyUrl2;
  withExcludeHostsEnv(nextEnv, excludeHosts);
  if (toolPolicy) {
    nextEnv[TOOL_POLICY_ENV] = toolPolicy.serialized;
    if (toolPolicy.validUntil !== void 0) {
      nextEnv[TOOL_POLICY_VALID_UNTIL_ENV] = String(toolPolicy.validUntil);
    } else {
      delete nextEnv[TOOL_POLICY_VALID_UNTIL_ENV];
    }
  } else {
    delete nextEnv[TOOL_POLICY_ENV];
    delete nextEnv[TOOL_POLICY_VALID_UNTIL_ENV];
  }
  const shim = shimImportSpecifier();
  if (shim) {
    nextEnv.NODE_OPTIONS = withNodeImportOption(nextEnv.NODE_OPTIONS, shim);
  }
  return nextEnv;
}
function withExcludeHostsEnv(env, excludeHosts) {
  if (excludeHosts.length > 0) {
    env[EXCLUDE_HOSTS_ENV] = excludeHosts.join(",");
  } else {
    delete env[EXCLUDE_HOSTS_ENV];
  }
}
function installProcessEnv(proxyUrl2, excludeHosts, toolPolicy) {
  process.env[PROXY_ENV] = proxyUrl2;
  withExcludeHostsEnv(process.env, excludeHosts);
  if (toolPolicy) {
    process.env[TOOL_POLICY_ENV] = toolPolicy.serialized;
    if (toolPolicy.validUntil !== void 0) {
      process.env[TOOL_POLICY_VALID_UNTIL_ENV] = String(toolPolicy.validUntil);
    } else {
      delete process.env[TOOL_POLICY_VALID_UNTIL_ENV];
    }
  } else {
    delete process.env[TOOL_POLICY_ENV];
    delete process.env[TOOL_POLICY_VALID_UNTIL_ENV];
  }
  const shim = shimImportSpecifier();
  if (shim) {
    process.env.NODE_OPTIONS = withNodeImportOption(process.env.NODE_OPTIONS, shim);
  }
}
function isOptions(value) {
  return Boolean(value) && typeof value === "object" && !Array.isArray(value) && !(value instanceof URL);
}
function injectOptionsEnv(args, optionIndex, proxyUrl2) {
  const state = getState();
  const nextArgs = [...args];
  const callback = typeof nextArgs.at(-1) === "function" ? nextArgs.pop() : void 0;
  const existing = isOptions(nextArgs[optionIndex]) ? { ...nextArgs[optionIndex] } : {};
  existing.env = withShimEnv(
    existing.env,
    proxyUrl2,
    state?.excludeHosts ?? [],
    state?.toolPolicy
  );
  if (process.platform === "win32" && existing.windowsHide === void 0) {
    existing.windowsHide = true;
  }
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
function normalizedCommandName(command) {
  const trimmed = command.trim();
  if (!trimmed) {
    return "";
  }
  return path.basename(trimmed).toLowerCase();
}
function commandMatches(command, patterns) {
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
function shellTokens(commandLine) {
  const tokens = [];
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
var SHELL_OPERATORS = /* @__PURE__ */ new Set([";", "&&", "||", "|", "&"]);
var COMMAND_WRAPPERS = /* @__PURE__ */ new Set(["command", "env", "nohup", "sudo", "time"]);
var SHELL_WRAPPERS = /* @__PURE__ */ new Set(["bash", "cmd", "dash", "ksh", "powershell", "pwsh", "sh", "zsh"]);
var ENV_ASSIGNMENT = /^[A-Za-z_][A-Za-z0-9_]*=/;
var WRAPPER_OPTIONS_WITH_VALUE = {
  env: /* @__PURE__ */ new Set(["-C", "--chdir", "-S", "--split-string", "-u", "--unset"]),
  sudo: /* @__PURE__ */ new Set([
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
    "--user"
  ]),
  time: /* @__PURE__ */ new Set(["-f", "--format", "-o", "--output"])
};
function shellCommandSubstitutions(commandLine) {
  const substitutions = [];
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
      let end2 = index + 1;
      for (; end2 < commandLine.length; end2 += 1) {
        if (commandLine[end2] === "\\") {
          end2 += 1;
        } else if (commandLine[end2] === "`") {
          break;
        }
      }
      if (end2 < commandLine.length) {
        substitutions.push(commandLine.slice(index + 1, end2));
        index = end2;
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
var DYNAMIC_COMMANDS = /* @__PURE__ */ new Set([
  "call",
  "case",
  "coproc",
  "do",
  "done",
  "elif",
  "else",
  "esac",
  "eval",
  "exec",
  "fi",
  "for",
  "function",
  "get-command",
  "iex",
  "if",
  "invoke-expression",
  "select",
  "source",
  "then",
  "until",
  "while"
]);
function hasDynamicShellExecution(commandLine, toolName) {
  const normalizedTool = normalizedCommandName(toolName ?? "");
  const cmdDelayedExpansion = normalizedTool === "cmd" || normalizedTool === "shell" && process.platform === "win32";
  let quote = "";
  let cmdDelayedExpansionStart = -1;
  let delayedExpansionState = 0;
  let percentStart = -1;
  let visible = "";
  for (let index = 0; index < commandLine.length; index += 1) {
    const char = commandLine[index];
    if (char === "\r" || char === "\n") {
      cmdDelayedExpansionStart = -1;
      delayedExpansionState = 0;
      percentStart = -1;
    }
    if (cmdDelayedExpansion && char === "!") {
      if (cmdDelayedExpansionStart >= 0 && index > cmdDelayedExpansionStart + 1) return true;
      cmdDelayedExpansionStart = index;
    }
    if (quote === "'") {
      delayedExpansionState = 0;
      visible += " ";
      if (char === "'") quote = "";
      continue;
    }
    if (char === "'") {
      delayedExpansionState = 0;
      quote = "'";
      visible += " ";
      continue;
    }
    if (char === '"') {
      delayedExpansionState = 0;
      quote = quote === '"' ? "" : '"';
      visible += " ";
      continue;
    }
    if (char === "\\") {
      delayedExpansionState = 0;
      const next = commandLine[index + 1];
      if (quote !== '"' && next && /[A-Za-z0-9]/.test(next)) return true;
      visible += "  ";
      index += 1;
      continue;
    }
    if (char === "%") {
      if (percentStart >= 0 && index > percentStart + 1) return true;
      percentStart = index;
    }
    if (char === "!") {
      if (delayedExpansionState === 2) return true;
      delayedExpansionState = 1;
    } else if (delayedExpansionState === 1) {
      delayedExpansionState = /[A-Za-z_]/.test(char) ? 2 : 0;
    } else if (delayedExpansionState === 2 && !/[A-Za-z0-9_]/.test(char)) {
      delayedExpansionState = 0;
    }
    if (char === "`" || char === "$") return true;
    visible += quote ? " " : char;
  }
  if (quote) return true;
  if (/[<>]\s*\(|(?:^|[;&|]\s*)&\s*\(|[{}]/.test(visible)) return true;
  return shellCommandBinaries(commandLine).some(
    (command) => DYNAMIC_COMMANDS.has(normalizedCommandName(command))
  );
}
function shellCommandBinaries(commandLine) {
  const segments = [[]];
  for (const token of shellTokens(commandLine)) {
    if (SHELL_OPERATORS.has(token)) {
      if (segments.at(-1)?.length) segments.push([]);
    } else {
      segments.at(-1).push(token);
    }
  }
  const binaries = [];
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
          if (!token.includes("=") && WRAPPER_OPTIONS_WITH_VALUE[wrapper]?.has(optionName) && index < segment.length) {
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
function matchesDomain(hostname, patterns) {
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
function hashPolicyResource(resource) {
  return createHash("sha256").update(resource).digest("hex").slice(0, 16);
}
function safePolicyResource(input) {
  if (input.scope === "shell") {
    return (input.atomicCommand ? [input.command] : shellCommandBinaries(input.resource)).map((command) => normalizedCommandName(command)).filter(Boolean).join(",");
  }
  if (input.scope === "tool_call") {
    return input.toolName;
  }
  return input.url.hostname.toLowerCase().replace(/^\[|\]$/g, "");
}
function emitPolicyDecision(decision) {
  const record = {
    event: "headroom_tool_policy_decision",
    version: decision.version,
    decision_id: decision.decisionId,
    authority: decision.authority,
    timestamp: (/* @__PURE__ */ new Date()).toISOString(),
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
    binding: decision.binding
  };
  try {
    process.stderr.write(`${JSON.stringify(record)}
`);
  } catch {
  }
  try {
    if (processIsStateless()) {
      return;
    }
    const target = path.join(workspaceDir(), "tool_policy_audit.jsonl");
    fs.mkdirSync(path.dirname(target), { recursive: true });
    fs.appendFileSync(target, `${JSON.stringify(record)}
`, "utf8");
  } catch {
  }
}
function emitPolicyAcknowledgement(acknowledgement) {
  const record = {
    ...acknowledgement,
    decision_id: acknowledgement.decisionId,
    request_hash: acknowledgement.requestHash,
    decisionId: void 0,
    requestHash: void 0
  };
  try {
    process.stderr.write(`${JSON.stringify(record)}
`);
  } catch {
  }
  try {
    if (processIsStateless()) return;
    const target = path.join(workspaceDir(), "tool_policy_audit.jsonl");
    fs.mkdirSync(path.dirname(target), { recursive: true });
    fs.appendFileSync(target, `${JSON.stringify(record)}
`, "utf8");
  } catch {
  }
}
function policyDecisionId(requestHash, action, authority, binding) {
  return createHash("sha256").update(stableJson({ version: 1, requestHash, action, authority, binding })).update(binding ? randomUUID() : "").digest("hex");
}
function evaluatePolicy(policy, input, authority = "advisory", binding) {
  if (!policy) {
    return void 0;
  }
  const dynamicShellExecution = input.scope === "shell" && !input.atomicCommand && (policy.defaultAction === "deny" || policy.rules.some(
    (rule) => rule.action !== "allow" && (rule.scope === "shell" || rule.scope === "tool_call")
  )) && hasDynamicShellExecution(input.resource, input.toolName);
  const matchedRule = policy.rules.find((rule) => {
    if (input.scope === "tool_call" && rule.scope !== "tool_call" || input.scope !== "tool_call" && rule.scope !== "tool_call" && rule.scope !== input.scope) {
      return false;
    }
    if (rule.tools?.length) {
      if (!("toolName" in input) || !rule.tools.includes((input.toolName ?? "").trim().toLowerCase())) {
        return false;
      }
    }
    if (input.scope === "shell") {
      const commands = input.atomicCommand ? [input.command] : shellCommandBinaries(input.resource);
      if (rule.commands?.length) {
        const commandsMatch = rule.action === "allow" ? commands.length > 0 && commands.every((candidate) => commandMatches(candidate, rule.commands)) : commands.some((candidate) => commandMatches(candidate, rule.commands));
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
      if (rule.envKeys?.length && !rule.envKeys.every(
        (entry) => Object.keys(input.env ?? process.env).some(
          (key) => key.toLowerCase() === entry
        )
      )) {
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
      if (rule.envKeys?.length && !rule.envKeys.every(
        (entry) => Object.keys(input.env ?? process.env).some(
          (key) => key.toLowerCase() === entry
        )
      )) {
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
  const dynamicShellDenied = dynamicShellExecution && (!matchedRule || matchedRule.action === "allow");
  const action = dynamicShellDenied ? "deny" : matchedRule?.action ?? policy.defaultAction;
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
    reason: dynamicShellDenied ? "dynamic or escaped shell execution cannot be safely authorized" : matchedRule?.reason,
    resource: safePolicyResource(input),
    requestHash,
    source: policy.source,
    binding
  };
}
function enforcePolicy(policy, input, authority = "advisory", binding) {
  const state = getState();
  if (state?.policyUnavailable) {
    throw new Error(`[headroom] Tool policy unavailable; failing closed: ${state.policyUnavailable}`);
  }
  if (policy?.validUntil !== void 0 && Date.now() / 1e3 >= policy.validUntil) {
    throw new Error("[headroom] Remote tool policy expired; failing closed until it is refreshed");
  }
  const decision = evaluatePolicy(policy, input, authority, binding);
  if (!decision) {
    return;
  }
  emitPolicyDecision(decision);
  const blockedAcknowledgement = decision.authority === "authoritative" && decision.binding && decision.effectiveAction !== "allow" ? {
    version: 1,
    event: "headroom_tool_policy_enforcement_acknowledgement",
    decisionId: decision.decisionId,
    authority: "authoritative",
    effect: "blocked",
    requestHash: decision.requestHash,
    binding: decision.binding,
    timestamp: (/* @__PURE__ */ new Date()).toISOString()
  } : void 0;
  if (decision.effectiveAction === "allow") {
    return { decision };
  }
  if (blockedAcknowledgement) emitPolicyAcknowledgement(blockedAcknowledgement);
  const suffix = (decision.matchedRuleId ? ` (rule=${decision.matchedRuleId})` : "") + (decision.reason ? `: ${decision.reason}` : "");
  if (decision.effectiveAction === "require_approval") {
    const message2 = `[headroom] Tool policy requires approval for ${decision.scope} target ${decision.resource}` + suffix + ". No approval handler is installed in the OpenCode transport yet.";
    if (blockedAcknowledgement) {
      throw new ToolPolicyEnforcementError(message2, decision, blockedAcknowledgement);
    }
    throw new Error(message2);
  }
  const message = `[headroom] Tool policy denied ${decision.scope} target ${decision.resource}` + suffix;
  if (blockedAcknowledgement) {
    throw new ToolPolicyEnforcementError(message, decision, blockedAcknowledgement);
  }
  throw new Error(message);
}
function stableJson(value) {
  if (Array.isArray(value)) {
    return `[${value.map((entry) => stableJson(entry)).join(",")}]`;
  }
  if (value && typeof value === "object") {
    return `{${Object.entries(value).sort(([left], [right]) => left.localeCompare(right)).map(([key, entry]) => `${JSON.stringify(key)}:${stableJson(entry)}`).join(",")}}`;
  }
  return JSON.stringify(value) ?? "null";
}
function effectiveChildCwd(value) {
  if (typeof value === "string") {
    return path.resolve(process.cwd(), value);
  }
  if (value instanceof URL && value.protocol === "file:") {
    return path.resolve(fileURLToPath(value));
  }
  return process.cwd();
}
function wrapSpawn(originalSpawn) {
  return function headroomSpawn(...args) {
    const state = getState();
    if (!state) {
      return Reflect.apply(originalSpawn, this, args);
    }
    const command = String(args[0] ?? "");
    const commandArgs = Array.isArray(args[1]) ? args[1].map((entry) => String(entry)) : [];
    const resource = [command, ...commandArgs].join(" ").trim() || command;
    const options = isOptions(args[Array.isArray(args[1]) ? 2 : 1]) ? args[Array.isArray(args[1]) ? 2 : 1] : void 0;
    enforcePolicy(state.toolPolicy, {
      scope: "shell",
      resource,
      command,
      argsText: resource,
      cwd: effectiveChildCwd(options?.cwd),
      env: options?.env,
      atomicCommand: options?.shell !== true && typeof options?.shell !== "string"
    });
    const optionIndex = Array.isArray(args[1]) ? 2 : 1;
    return Reflect.apply(originalSpawn, this, injectOptionsEnv(args, optionIndex, state.proxyUrl));
  };
}
function wrapExec(originalExec) {
  return function headroomExec(...args) {
    const state = getState();
    if (!state) {
      return Reflect.apply(originalExec, this, args);
    }
    const commandLine = String(args[0] ?? "");
    const options = isOptions(args[1]) ? args[1] : void 0;
    enforcePolicy(state.toolPolicy, {
      scope: "shell",
      resource: commandLine,
      command: shellCommandBinaries(commandLine)[0] ?? "",
      argsText: commandLine,
      cwd: effectiveChildCwd(options?.cwd),
      env: options?.env
    });
    return Reflect.apply(originalExec, this, injectOptionsEnv(args, 1, state.proxyUrl));
  };
}
function wrapExecFile(originalExecFile) {
  return function headroomExecFile(...args) {
    const state = getState();
    if (!state) {
      return Reflect.apply(originalExecFile, this, args);
    }
    const command = String(args[0] ?? "");
    const commandArgs = Array.isArray(args[1]) ? args[1].map((entry) => String(entry)) : [];
    const resource = [command, ...commandArgs].join(" ").trim() || command;
    const options = isOptions(args[Array.isArray(args[1]) ? 2 : 1]) ? args[Array.isArray(args[1]) ? 2 : 1] : void 0;
    enforcePolicy(state.toolPolicy, {
      scope: "shell",
      resource,
      command,
      argsText: resource,
      cwd: effectiveChildCwd(options?.cwd),
      env: options?.env,
      atomicCommand: options?.shell !== true && typeof options?.shell !== "string"
    });
    const optionIndex = Array.isArray(args[1]) ? 2 : 1;
    return Reflect.apply(originalExecFile, this, injectOptionsEnv(args, optionIndex, state.proxyUrl));
  };
}
function wrapFork(originalFork) {
  return function headroomFork(...args) {
    const state = getState();
    if (!state) {
      return Reflect.apply(originalFork, this, args);
    }
    const command = String(args[0] ?? "");
    const commandArgs = Array.isArray(args[1]) ? args[1].map((entry) => String(entry)) : [];
    const resource = [command, ...commandArgs].join(" ").trim() || command;
    const options = isOptions(args[Array.isArray(args[1]) ? 2 : 1]) ? args[Array.isArray(args[1]) ? 2 : 1] : void 0;
    enforcePolicy(state.toolPolicy, {
      scope: "shell",
      resource,
      command,
      argsText: resource,
      cwd: effectiveChildCwd(options?.cwd),
      env: options?.env,
      atomicCommand: true
    });
    const optionIndex = Array.isArray(args[1]) ? 2 : 1;
    return Reflect.apply(originalFork, this, injectOptionsEnv(args, optionIndex, state.proxyUrl));
  };
}
function normalizeProxyUrl(proxyUrl2) {
  return new URL(proxyUrl2);
}
function isLoopback(hostname) {
  const normalized = hostname.toLowerCase().replace(/^\[|\]$/g, "");
  return normalized === "localhost" || normalized === "127.0.0.1" || normalized === "::1";
}
function normalizeExcludeHosts(entries) {
  const hosts = /* @__PURE__ */ new Set();
  for (const entry of typeof entries === "string" ? entries.split(",") : entries) {
    const host = String(entry).trim().toLowerCase().replace(/^(\*\.|\.)/, "");
    if (host) {
      hosts.add(host);
    }
  }
  return [...hosts];
}
function isExcludedHost(hostname, excludeHosts) {
  const normalized = hostname.toLowerCase().replace(/^\[|\]$/g, "");
  return excludeHosts.some((host) => normalized === host || normalized.endsWith(`.${host}`));
}
function isLlmEndpointPath(pathname) {
  return pathname.endsWith("/chat/completions") || pathname.endsWith("/responses") || pathname.endsWith("/messages") || pathname.endsWith(":generateContent") || pathname.endsWith(":streamGenerateContent");
}
function shouldRoute(url, proxy, excludeHosts) {
  if (url.protocol !== "http:" && url.protocol !== "https:") {
    return false;
  }
  if (isLoopback(url.hostname)) {
    return false;
  }
  if (url.origin === proxy.origin) {
    return false;
  }
  if (isExcludedHost(url.hostname, excludeHosts)) {
    return false;
  }
  return isLlmEndpointPath(url.pathname);
}
function routedUrl(upstream, proxy) {
  return new URL(`${upstream.pathname}${upstream.search}`, proxy.origin);
}
function normalizedOpenAiProxyPath(pathname) {
  if (pathname.endsWith("/chat/completions")) {
    return "/v1/chat/completions";
  }
  if (pathname.endsWith("/responses")) {
    return "/v1/responses";
  }
  return void 0;
}
function routedUrlForOpenCode(upstream, proxy) {
  const normalizedPath = normalizedOpenAiProxyPath(upstream.pathname);
  if (!normalizedPath) {
    return {
      url: routedUrl(upstream, proxy),
      originalPath: void 0
    };
  }
  return {
    url: new URL(`${normalizedPath}${upstream.search}`, proxy.origin),
    originalPath: upstream.pathname
  };
}
function requestUrl(input) {
  if (input instanceof Request) {
    return new URL(input.url);
  }
  if (input instanceof URL) {
    return input;
  }
  return new URL(String(input));
}
function mergeFetchHeaders(input, init, upstream, originalPath = void 0, project = void 0) {
  const headers = new Headers(input instanceof Request ? input.headers : void 0);
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
function withRoutedFetchInput(input, init, proxy, project, excludeHosts) {
  const upstream = requestUrl(input);
  if (!shouldRoute(upstream, proxy, excludeHosts)) {
    return [input, init];
  }
  const { url: nextUrl, originalPath } = routedUrlForOpenCode(upstream, proxy);
  const nextInit = {
    ...init,
    headers: mergeFetchHeaders(input, init, upstream, originalPath, project)
  };
  if (input instanceof Request) {
    return [new Request(nextUrl, input), nextInit];
  }
  return [nextUrl, nextInit];
}
function splitNodeArgs(args) {
  const callback = typeof args.at(-1) === "function" ? args.at(-1) : void 0;
  const withoutCallback = callback ? args.slice(0, -1) : args;
  const [first, second] = withoutCallback;
  const options = typeof second === "object" && second !== null ? { ...second } : {};
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
    const requestOptions = { ...first, ...options };
    return { url: urlFromRequestOptions(requestOptions), options: requestOptions, callback };
  }
  return { options, callback };
}
function urlFromRequestOptions(options) {
  const protocol = String(options.protocol ?? "http:");
  if (protocol !== "http:" && protocol !== "https:") {
    return void 0;
  }
  const hostValue = options.hostname ?? options.host;
  if (!hostValue) {
    return void 0;
  }
  const hostname = String(hostValue).replace(/:\d+$/, "");
  const port = options.port ? `:${String(options.port)}` : "";
  const path2 = String(options.path ?? "/");
  try {
    return new URL(`${protocol}//${hostname}${port}${path2}`);
  } catch {
    return void 0;
  }
}
function headersForNodeRequest(options, upstream, originalPath, project) {
  const headers = new Headers(options.headers);
  headers.set(BASE_URL_HEADER, upstream.origin);
  if (originalPath) {
    headers.set(ORIGINAL_PATH_HEADER, originalPath);
  }
  if (project) {
    headers.set(PROJECT_HEADER, project);
  }
  headers.delete("host");
  const result = {};
  headers.forEach((value, key) => {
    result[key] = value;
  });
  return result;
}
function routedNodeOptions(parts, proxy, project, excludeHosts) {
  if (!parts.url || !shouldRoute(parts.url, proxy, excludeHosts)) {
    return void 0;
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
    port: nextUrl.port || void 0,
    path: `${nextUrl.pathname}${nextUrl.search}`,
    headers: headersForNodeRequest(parts.options, parts.url, originalPath, project)
  };
}
function wrapRequest(originalHttpRequest, originalHttpsRequest, originalRequest) {
  return function headroomRequest(...args) {
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
        url: parts.url
      });
    }
    const nextOptions = routedNodeOptions(parts, proxy, state.project, state.excludeHosts);
    if (!nextOptions) {
      return Reflect.apply(originalRequest, this, args);
    }
    const targetRequest = proxy.protocol === "https:" ? originalHttpsRequest : originalHttpRequest;
    const nextArgs = parts.callback ? [nextOptions, parts.callback] : [nextOptions];
    return Reflect.apply(targetRequest, this, nextArgs);
  };
}
function wrapGet(request) {
  return function headroomGet(...args) {
    const req = Reflect.apply(request, this, args);
    req.end();
    return req;
  };
}
function wrapHttp2Connect(originalConnect) {
  return function headroomHttp2Connect(authority, ...args) {
    const state = getState();
    if (state) {
      const upstream = authority instanceof URL ? authority : new URL(String(authority));
      enforcePolicy(state.toolPolicy, {
        scope: "http",
        resource: upstream.href,
        url: upstream
      });
    }
    return Reflect.apply(originalConnect, this, [authority, ...args]);
  };
}
function installHeadroomTransport(options) {
  const excludeHosts = normalizeExcludeHosts(
    options.excludeHosts ?? process.env[EXCLUDE_HOSTS_ENV] ?? ""
  );
  const existing = getState();
  const selectedSource = resolveToolPolicySource(options, existing);
  let toolPolicy;
  let policyUnavailable;
  try {
    toolPolicy = compileToolPolicy(
      selectedSource.input,
      options.policyProject ?? options.project,
      selectedSource.source
    );
    if (toolPolicy && selectedSource.validUntil !== void 0) {
      toolPolicy.validUntil = selectedSource.validUntil;
    }
  } catch (error) {
    policyUnavailable = error instanceof Error ? error.message : String(error);
  }
  if (selectedSource.remoteUrl && !toolPolicy) {
    policyUnavailable ??= "remote Headroom tool policy has not been loaded";
  }
  const policyContextKey = createHash("sha256").update(
    stableJson({
      identity: selectedSource.identity,
      remoteToken: selectedSource.remoteToken,
      policy: selectedSource.remoteUrl ? void 0 : toolPolicy?.serialized,
      validUntil: selectedSource.validUntil
    })
  ).digest("hex");
  if (existing) {
    if (existing.policyContextKey !== policyContextKey) {
      throw new Error(
        "[headroom] Multiple OpenCode workspaces with different tool policies share one process; refusing to replace the active policy"
      );
    }
    existing.refs += 1;
    existing.proxyUrl = options.proxyUrl;
    existing.project = options.project;
    existing.excludeHosts = excludeHosts;
    existing.debug = Boolean(options.debug);
    installProcessEnv(options.proxyUrl, excludeHosts, existing.toolPolicy);
    return () => uninstallHeadroomTransport();
  }
  const state = {
    refs: 1,
    policyContextKey,
    proxyUrl: options.proxyUrl,
    project: options.project,
    excludeHosts,
    debug: Boolean(options.debug),
    toolPolicy,
    toolPolicyInput: options.toolPolicy,
    remotePolicyUrl: selectedSource.remoteUrl,
    remotePolicyToken: selectedSource.remoteToken,
    policyUnavailable,
    previousNodeOptions: process.env.NODE_OPTIONS,
    previousProxyUrlEnv: process.env[PROXY_ENV],
    previousExcludeHostsEnv: process.env[EXCLUDE_HOSTS_ENV],
    previousToolPolicyEnv: process.env[TOOL_POLICY_ENV],
    previousToolPolicyValidUntilEnv: process.env[TOOL_POLICY_VALID_UNTIL_ENV],
    originalFetch: globalThis.fetch,
    originalHttpRequest: http.request,
    originalHttpGet: http.get,
    originalHttpsRequest: https.request,
    originalHttpsGet: https.get,
    originalHttp2Connect: http2.connect,
    originalChildSpawn: childProcess.spawn,
    originalChildExec: childProcess.exec,
    originalChildExecFile: childProcess.execFile,
    originalChildFork: childProcess.fork
  };
  setState(state);
  installProcessEnv(options.proxyUrl, excludeHosts, toolPolicy);
  globalThis.fetch = async (...args) => {
    const current = getState();
    if (!current) {
      return state.originalFetch(...args);
    }
    await refreshHeadroomToolPolicy();
    const upstream = requestUrl(args[0]);
    enforcePolicy(current.toolPolicy, {
      scope: "http",
      resource: upstream.href,
      url: upstream
    });
    const proxy = normalizeProxyUrl(current.proxyUrl);
    const [nextInput, nextInit] = withRoutedFetchInput(
      args[0],
      args[1],
      proxy,
      current.project,
      current.excludeHosts
    );
    return state.originalFetch(nextInput, nextInit);
  };
  http.request = wrapRequest(state.originalHttpRequest, state.originalHttpsRequest, state.originalHttpRequest);
  https.request = wrapRequest(state.originalHttpRequest, state.originalHttpsRequest, state.originalHttpsRequest);
  http.get = wrapGet(http.request);
  https.get = wrapGet(https.request);
  http2.connect = wrapHttp2Connect(state.originalHttp2Connect);
  childProcess.spawn = wrapSpawn(state.originalChildSpawn);
  childProcess.exec = wrapExec(state.originalChildExec);
  childProcess.execFile = wrapExecFile(state.originalChildExecFile);
  childProcess.fork = wrapFork(state.originalChildFork);
  syncBuiltinESMExports();
  return () => uninstallHeadroomTransport();
}
function uninstallHeadroomTransport() {
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
  if (state.previousNodeOptions === void 0) {
    delete process.env.NODE_OPTIONS;
  } else {
    process.env.NODE_OPTIONS = state.previousNodeOptions;
  }
  if (state.previousProxyUrlEnv === void 0) {
    delete process.env[PROXY_ENV];
  } else {
    process.env[PROXY_ENV] = state.previousProxyUrlEnv;
  }
  if (state.previousExcludeHostsEnv === void 0) {
    delete process.env[EXCLUDE_HOSTS_ENV];
  } else {
    process.env[EXCLUDE_HOSTS_ENV] = state.previousExcludeHostsEnv;
  }
  if (state.previousToolPolicyEnv === void 0) {
    delete process.env[TOOL_POLICY_ENV];
  } else {
    process.env[TOOL_POLICY_ENV] = state.previousToolPolicyEnv;
  }
  if (state.previousToolPolicyValidUntilEnv === void 0) {
    delete process.env[TOOL_POLICY_VALID_UNTIL_ENV];
  } else {
    process.env[TOOL_POLICY_VALID_UNTIL_ENV] = state.previousToolPolicyValidUntilEnv;
  }
  setState(void 0);
}

// src/hook-shim.ts
var proxyUrl = process.env.HEADROOM_OPENCODE_TRANSPORT_PROXY_URL;
if (!proxyUrl) {
  throw new Error(
    "Headroom OpenCode transport shim loaded without HEADROOM_OPENCODE_TRANSPORT_PROXY_URL"
  );
}
installHeadroomTransport({ proxyUrl });
