#!/usr/bin/env node

import { copyFileSync, existsSync, mkdtempSync, readFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import path from "node:path";
import { spawnSync } from "node:child_process";

const [assetsDirArg, version] = process.argv.slice(2);

if (!assetsDirArg || !version) {
  console.error("Usage: node scripts/verify_npm_release_assets.mjs <assets-dir> <version>");
  process.exit(2);
}

const assetsDir = path.resolve(assetsDirArg);

const packages = [
  {
    name: "headroom-ai",
    tarball: `headroom-ai-${version}.tgz`,
  },
  {
    name: "headroom-openclaw",
    tarball: `headroom-openclaw-${version}.tgz`,
    dependencies: {
      "headroom-ai": `^${version}`,
    },
  },
];

const tarballPaths = new Map();

function extractPackageJson(tarballPath) {
  return extractJsonFromTarball(tarballPath, "package/package.json");
}

function extractDistPackageJson(tarballPath) {
  return extractJsonFromTarball(tarballPath, "package/dist/package.json");
}

function extractJsonFromTarball(tarballPath, packageJsonPath) {
  const workdir = mkdtempSync(path.join(tmpdir(), "headroom-npm-asset-"));
  try {
    const result = spawnSync("tar", ["-xzf", tarballPath, "-C", workdir], {
      encoding: "utf8",
    });
    if (result.status !== 0) {
      throw new Error(
        `tar failed for ${tarballPath}: ${result.stderr || result.stdout || "unknown error"}`,
      );
    }
    return JSON.parse(readFileSync(path.join(workdir, packageJsonPath), "utf8"));
  } finally {
    rmSync(workdir, { recursive: true, force: true });
  }
}

function assertNoFileDependencies(pkg) {
  for (const field of ["dependencies", "peerDependencies", "optionalDependencies"]) {
    for (const [name, spec] of Object.entries(pkg[field] || {})) {
      if (typeof spec === "string" && (spec.startsWith("file:") || spec.includes("release-assets"))) {
        throw new Error(`${pkg.name} has non-portable ${field}.${name} spec: ${spec}`);
      }
    }
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
  return spawnSync(command, commandArgs, {
    cwd,
    encoding: "utf8",
  });
}

function assertOpenClawExtensionContract(cwd) {
  const smoke = `
    const mod = await import("headroom-openclaw");
    if (typeof mod.default?.register !== "function") {
      throw new Error("headroom-openclaw default export must expose register(api)");
    }
    if (typeof mod.registerHeadroomPlugin !== "function") {
      throw new Error("headroom-openclaw must export registerHeadroomPlugin(api)");
    }
    if (mod.default.register !== mod.registerHeadroomPlugin) {
      throw new Error("headroom-openclaw default.register must match registerHeadroomPlugin");
    }
  `;
  const result = spawnSync(process.execPath, ["--input-type=module", "-e", smoke], {
    cwd,
    encoding: "utf8",
  });
  if (result.status !== 0) {
    throw new Error(
      `headroom-openclaw import smoke failed: ${
        result.error?.message || result.stderr || result.stdout || "unknown error"
      }`,
    );
  }
}

for (const expected of packages) {
  const tarballPath = path.join(assetsDir, expected.tarball);
  tarballPaths.set(expected.name, tarballPath);
  const pkg = extractPackageJson(tarballPath);

  if (pkg.name !== expected.name) {
    throw new Error(`${expected.tarball} package name mismatch: expected ${expected.name}, got ${pkg.name}`);
  }
  if (pkg.version !== version) {
    throw new Error(`${expected.tarball} version mismatch: expected ${version}, got ${pkg.version}`);
  }

  assertNoFileDependencies(pkg);

  for (const [name, spec] of Object.entries(expected.dependencies || {})) {
    const actual = pkg.dependencies?.[name];
    if (actual !== spec) {
      throw new Error(`${pkg.name} dependency ${name} mismatch: expected ${spec}, got ${actual}`);
    }
  }

  if (expected.name === "headroom-openclaw") {
    const distPkg = extractDistPackageJson(tarballPath);
    if (distPkg.name !== expected.name) {
      throw new Error(`${expected.tarball} dist package name mismatch: expected ${expected.name}, got ${distPkg.name}`);
    }
    if (distPkg.version !== version) {
      throw new Error(`${expected.tarball} dist package version mismatch: expected ${version}, got ${distPkg.version}`);
    }
    assertNoFileDependencies(distPkg);
    for (const [name, spec] of Object.entries(expected.dependencies || {})) {
      const actual = distPkg.dependencies?.[name];
      if (actual !== spec) {
        throw new Error(`${distPkg.name} dist dependency ${name} mismatch: expected ${spec}, got ${actual}`);
      }
    }
  }
}

const installDir = mkdtempSync(path.join(tmpdir(), "headroom-npm-install-"));
try {
  for (const expected of packages) {
    copyFileSync(tarballPaths.get(expected.name), path.join(installDir, expected.tarball));
  }
  const result = runNpm(
    [
      "install",
      "--ignore-scripts",
      "--no-audit",
      "--no-fund",
      `./${packages[0].tarball}`,
      `./${packages[1].tarball}`,
    ],
    installDir,
  );
  if (result.status !== 0) {
    throw new Error(
      `clean npm install failed: ${
        result.error?.message || result.stderr || result.stdout || "unknown error"
      }`,
    );
  }
  assertOpenClawExtensionContract(installDir);
} finally {
  rmSync(installDir, { recursive: true, force: true });
}

console.log(`Verified npm release assets for ${version}`);
