#!/usr/bin/env node

import { copyFileSync, mkdtempSync, readFileSync, rmSync } from "node:fs";
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

function runNpm(args, cwd) {
  if (process.platform === "win32") {
    return spawnSync("cmd.exe", ["/d", "/s", "/c", "npm.cmd", ...args], {
      cwd,
      encoding: "utf8",
    });
  }

  return spawnSync("npm", args, {
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
