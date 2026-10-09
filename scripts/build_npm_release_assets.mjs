#!/usr/bin/env node

import {
  existsSync,
  mkdirSync,
  readFileSync,
  readdirSync,
  rmSync,
  writeFileSync,
} from "node:fs";
import path from "node:path";
import { spawnSync } from "node:child_process";
import { fileURLToPath } from "node:url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const rootDir = path.resolve(__dirname, "..");
const sdkDir = path.join(rootDir, "sdk", "typescript");
const openClawDir = path.join(rootDir, "plugins", "openclaw");

const rawArgs = process.argv.slice(2);
const flags = new Set(rawArgs.filter((arg) => arg.startsWith("--")));
const positional = rawArgs.filter((arg) => !arg.startsWith("--"));
const [version, assetsDirArg] = positional;

if (!version || flags.has("--help") || flags.has("-h")) {
  console.error(
    [
      "Usage: node scripts/build_npm_release_assets.mjs <version> [assets-dir] [--skip-install] [--no-verify]",
      "",
      "Builds the TypeScript SDK and OpenClaw npm release tarballs, rewrites",
      "OpenClaw release metadata to depend on the just-built SDK version,",
      "regenerates dist/package.json, and verifies the resulting assets.",
    ].join("\n"),
  );
  process.exit(flags.has("--help") || flags.has("-h") ? 0 : 2);
}

if (!/^\d+\.\d+\.\d+(?:[-+][0-9A-Za-z.-]+)?$/.test(version)) {
  console.error(`Invalid version: ${version}`);
  process.exit(2);
}

const timestamp = new Date().toISOString().replace(/\D/g, "").slice(0, 14);
const assetsDir = path.resolve(
  rootDir,
  assetsDirArg || path.join("release-assets-local", `${version}-${timestamp}`),
);

const trackedFiles = [
  path.join(sdkDir, "package.json"),
  path.join(sdkDir, "package-lock.json"),
  path.join(openClawDir, "package.json"),
  path.join(openClawDir, "package-lock.json"),
  path.join(openClawDir, "dist", "package.json"),
];

const snapshots = new Map(
  trackedFiles.map((filePath) => [
    filePath,
    existsSync(filePath) ? readFileSync(filePath, "utf8") : null,
  ]),
);

function quoteCmdArg(value) {
  const arg = String(value);
  if (/^[A-Za-z0-9_./:=\\-]+$/.test(arg)) {
    return arg;
  }
  return `"${arg.replace(/"/g, '""')}"`;
}

function run(command, args, cwd) {
  console.log(`\n> ${command} ${args.map(quoteCmdArg).join(" ")}`);
  const result = spawnSync(command, args, {
    cwd,
    encoding: "utf8",
    stdio: "inherit",
  });

  if (result.error) {
    throw new Error(`${command} failed: ${result.error.message}`);
  }

  if (result.status !== 0) {
    throw new Error(`${command} failed with exit code ${result.status ?? "unknown"}`);
  }
}

// npm on Windows is a .cmd shim, which only runs through cmd.exe, and cmd.exe
// reinterprets characters such as & and | inside arguments (paths included).
// Run npm's own CLI script under this Node binary instead, so no shell is
// involved on any platform.
function npmCommand(args) {
  if (process.platform !== "win32") {
    return ["npm", args];
  }
  return [process.execPath, [findNpmCli(), ...args]];
}

// Node and npm can live in different directories (a node shim, or npm
// upgraded into the global prefix), so look where npm itself says it is, then
// next to node, then next to every npm.cmd on PATH. Every npm.cmd install
// keeps its CLI at node_modules/npm/bin/npm-cli.js beside the shim.
function findNpmCli() {
  const cliPath = path.join("node_modules", "npm", "bin", "npm-cli.js");
  const candidates = [];
  if (process.env.npm_execpath?.endsWith("npm-cli.js")) {
    candidates.push(process.env.npm_execpath);
  }
  candidates.push(path.join(path.dirname(process.execPath), cliPath));
  for (const dir of (process.env.PATH || "").split(path.delimiter)) {
    if (dir && existsSync(path.join(dir, "npm.cmd"))) {
      candidates.push(path.join(dir, cliPath));
    }
  }
  const found = candidates.find((candidate) => existsSync(candidate));
  if (!found) {
    throw new Error(`npm CLI (npm-cli.js) not found. Looked in: ${candidates.join(", ")}`);
  }
  return found;
}

function runNpm(args, cwd) {
  const [command, commandArgs] = npmCommand(args);
  run(command, commandArgs, cwd);
}

function runNode(args, cwd) {
  run(process.execPath, args, cwd);
}

function readJson(filePath) {
  return JSON.parse(readFileSync(filePath, "utf8"));
}

function writeJson(filePath, data) {
  writeFileSync(filePath, `${JSON.stringify(data, null, 2)}\n`, "utf8");
}

function ensureEmptyAssetsDir() {
  mkdirSync(assetsDir, { recursive: true });
  const existing = readdirSync(assetsDir);
  if (existing.length > 0) {
    throw new Error(
      `Assets directory must be empty to avoid stale tarballs: ${assetsDir}`,
    );
  }
}

function restoreTrackedFiles() {
  for (const [filePath, contents] of snapshots.entries()) {
    if (contents === null) {
      rmSync(filePath, { force: true });
    } else {
      mkdirSync(path.dirname(filePath), { recursive: true });
      writeFileSync(filePath, contents, "utf8");
    }
  }
}

function relativeFileSpec(fromDir, targetPath) {
  let relativePath = path.relative(fromDir, targetPath).split(path.sep).join("/");
  if (!relativePath.startsWith(".")) {
    relativePath = `./${relativePath}`;
  }
  return `file:${relativePath}`;
}

function rewriteOpenClawDependency(spec) {
  const packageJsonPath = path.join(openClawDir, "package.json");
  const pkg = readJson(packageJsonPath);
  pkg.dependencies = pkg.dependencies || {};
  pkg.dependencies["headroom-ai"] = spec;
  writeJson(packageJsonPath, pkg);
}

function rewriteOpenClawLocalDependency(sdkTarballPath) {
  rewriteOpenClawDependency(relativeFileSpec(openClawDir, sdkTarballPath));
}

function rewriteOpenClawReleaseDependency() {
  rewriteOpenClawDependency(`^${version}`);
}

function assertTarballBuilt(name) {
  const tarballPath = path.join(assetsDir, `${name}-${version}.tgz`);
  if (!existsSync(tarballPath)) {
    throw new Error(`Expected npm pack to produce ${tarballPath}`);
  }
  return tarballPath;
}

try {
  ensureEmptyAssetsDir();

  if (!flags.has("--skip-install")) {
    runNpm(["ci"], sdkDir);
  }
  runNpm(["run", "build"], sdkDir);
  runNpm(["version", version, "--no-git-tag-version", "--allow-same-version"], sdkDir);
  runNpm(["pack", "--pack-destination", assetsDir], sdkDir);
  const sdkTarballPath = assertTarballBuilt("headroom-ai");

  rewriteOpenClawLocalDependency(sdkTarballPath);
  if (!flags.has("--skip-install")) {
    runNpm(
      ["install", "--package-lock=false", "--no-audit", "--no-fund", "--ignore-scripts"],
      openClawDir,
    );
  } else {
    runNpm(["install", "--no-save", "--package-lock=false", sdkTarballPath], openClawDir);
  }
  runNpm(["run", "build"], openClawDir);
  runNpm(["version", version, "--no-git-tag-version", "--allow-same-version"], openClawDir);
  rewriteOpenClawReleaseDependency();
  runNode(["prepare-dist.mjs"], openClawDir);
  runNpm(["pack", "--pack-destination", assetsDir], openClawDir);
  assertTarballBuilt("headroom-openclaw");

  if (!flags.has("--no-verify")) {
    runNode(["scripts/verify_npm_release_assets.mjs", assetsDir, version], rootDir);
  }

  console.log(`\nBuilt and verified npm release assets in ${assetsDir}`);
} finally {
  restoreTrackedFiles();
  runNode(["prepare-dist.mjs"], openClawDir);
}
