import fs from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const rootDir = __dirname;
const distDir = path.join(rootDir, "dist");

const rootPackage = JSON.parse(
  await fs.readFile(path.join(rootDir, "package.json"), "utf8"),
);

const distPackage = {
  name: rootPackage.name,
  version: rootPackage.version,
  description: rootPackage.description,
  type: rootPackage.type,
  main: "./index.js",
  types: "./index.d.ts",
  license: rootPackage.license,
  dependencies: rootPackage.dependencies,
  peerDependencies: rootPackage.peerDependencies,
  peerDependenciesMeta: rootPackage.peerDependenciesMeta,
  openclaw: {
    hooks: ["./hook-shim"],
    extensions: ["./index.js"],
    capabilities: rootPackage.openclaw?.capabilities ?? {},
  },
};

await fs.mkdir(distDir, { recursive: true });
await fs.writeFile(
  path.join(distDir, "package.json"),
  `${JSON.stringify(distPackage, null, 2)}\n`,
  "utf8",
);

await Promise.all([
  fs.copyFile(
    path.join(rootDir, "openclaw.plugin.json"),
    path.join(distDir, "openclaw.plugin.json"),
  ),
  fs.copyFile(path.join(rootDir, "README.md"), path.join(distDir, "README.md")),
  fs.mkdir(path.join(distDir, "hook-shim"), { recursive: true }),
]);

await Promise.all([
  fs.copyFile(
    path.join(rootDir, "hook-shim", "HOOK.md"),
    path.join(distDir, "hook-shim", "HOOK.md"),
  ),
  fs.copyFile(
    path.join(rootDir, "hook-shim", "handler.js"),
    path.join(distDir, "hook-shim", "handler.js"),
  ),
]);
