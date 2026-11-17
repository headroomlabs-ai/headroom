# Wrap E2E CLI dependencies

The Docker harness installs Codex and OpenClaw here using npm 12.2.2 and
`npm ci`. Keep the manifest and lockfile together: pinning only the CLI versions
still lets transitive dependency ranges change on every image build. That broke
main during an AWS SDK publication on 2026-09-09 (ETARGET and tarball 404).

The image installs into `/opt/wrap-tools` and exposes `node_modules/.bin` on
PATH. `npm ci` does not support global installs, so diagnostics must also use
`npm list --prefix /opt/wrap-tools`.

After editing a CLI version in `package.json`, regenerate the lockfile using
Node 24.22.1+ within the Node 24 line and npm 12.2.2 on Linux (the pinned
OpenClaw version no longer supports Node 22):

```sh
cd e2e/wrap/tools
npm install --package-lock-only --no-fund --no-audit
npm ci --no-fund --no-audit
npm test
cd ../../..
docker build -f e2e/wrap/Dockerfile -t headroom-wrap-e2e .
docker run --rm headroom-wrap-e2e
```

`npm test` runs the Ajv URI, minimatch pattern-expansion, Express body-parser
and npm bundled-copy regression checks against the installed locked
dependencies. The Dockerfile runs the same command from `/opt/wrap-tools` after
`npm ci`, so failures stop the image build in both the existing Wrap E2E
workflow and CI's Docker-native wrap E2E job. Their path filters include
`e2e/**`, covering dependency and regression-test updates.

To intentionally refresh transitive versions without changing the pinned CLI
versions, use `npm update --package-lock-only --no-fund --no-audit` in this
directory instead of `npm install --package-lock-only`, then run the same
installation and Docker checks above.

Changing the `npm` override is a separate procedure. `npm install
--package-lock-only` does not refresh npm's `inBundle` children, so the lock
keeps the old bundled versions and the audit keeps reporting them. Bump the
override, then run a real install in a scratch copy of this directory (with the
`package.json`, `.npmrc` and old lock), and copy its `package-lock.json` back:

```sh
cp package.json .npmrc package-lock.json /tmp/npm-bump/
cd /tmp/npm-bump && npm install --ignore-scripts --no-fund --no-audit
cp package-lock.json /path/to/e2e/wrap/tools/
```

Review that diff: only the bundled copies npm changed between the two releases
should move (`node_modules/npm/node_modules/*`), plus the npm entry itself. Then
update the `npm@` pin in the Dockerfile, recheck `npm-audit-exceptions.json`
against the new bundled versions, and run the checks above.

Review the lockfile diff and justify dependency changes in the PR. A lockfile
fixes resolution and checks package integrity; it cannot make the npm registry
available during an outage. Do not delete the lockfile or add fallback installs
to make a build pass.

These are existing test tools, not new product dependencies. Codex is maintained
by OpenAI; OpenClaw and its transitive dependencies retain their upstream
maintainers. Their install surface includes registry downloads, native optional
packages and lifecycle scripts, as before. OpenClaw 2026.10.10 and its fixed
Undici override follow the upstream #4453 security repair.

The raw audit still reports npm 12.2.2's bundled glob and minimatch copies.
`npm-audit-exceptions.json` records the upstream-approved E2E-only risk
acceptance, restricted to exact advisory/path/version combinations and expiring
on 2026-12-15. Security CI rejects other findings and expired or stale entries;
these exceptions are not a claim that the contained packages are patched.

The root `allowScripts` policy approves only the exact locked install-script
identities reviewed for this harness: `@google/genai@2.27.0` (no-op preinstall),
`esbuild@0.28.2` (native binary validation/fallback acquisition),
`protobufjs@7.6.6` (dependency version-scheme warning), and
`openclaw@2026.10.10` (package-manager/runtime validation and bundled-plugin/native
prebuild maintenance). These scripts execute package code and may spawn helpers;
esbuild may acquire its platform binary. No wildcard or blanket approval is used.
The local `.npmrc` enables strict review: a new install-script identity fails
installation until its script and version are reviewed. npm12 otherwise blocks
unreviewed scripts while merely warning, which is insufficient proof that the
normal lifecycle ran. Bundled npm dependency copies do not run separate scripts.
