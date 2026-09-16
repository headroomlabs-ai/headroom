# Wrap E2E CLI dependencies

The Docker harness installs Codex and OpenClaw here using npm 11.16.0 and
`npm ci`. Keep the manifest and lockfile together: pinning only the CLI versions
still lets transitive dependency ranges change on every image build. That broke
main during an AWS SDK publication on 2026-09-09 (ETARGET and tarball 404).

The image installs into `/opt/wrap-tools` and exposes `node_modules/.bin` on
PATH. `npm ci` does not support global installs, so diagnostics must also use
`npm list --prefix /opt/wrap-tools`.

After editing a CLI version in `package.json`, regenerate the lockfile using
Node 22 and npm 11.16.0 on Linux:

```sh
cd e2e/wrap/tools
npm install --package-lock-only --no-fund --no-audit
npm ci --no-fund --no-audit
cd ../../..
docker build -f e2e/wrap/Dockerfile -t headroom-wrap-e2e .
docker run --rm headroom-wrap-e2e
```

To intentionally refresh transitive versions without changing the pinned CLI
versions, use `npm update --package-lock-only --no-fund --no-audit` in this
directory instead of `npm install --package-lock-only`, then run the same
installation and Docker checks above.

Review the lockfile diff and justify dependency changes in the PR. A lockfile
fixes resolution and checks package integrity; it cannot make the npm registry
available during an outage. Do not delete the lockfile or add fallback installs
to make a build pass.

These are existing test tools, not new product dependencies. Codex is maintained
by OpenAI; OpenClaw and its transitive dependencies retain their upstream
maintainers. Their install surface includes registry downloads, native optional
packages and lifecycle scripts, as before. Top-level CLI versions are unchanged.
