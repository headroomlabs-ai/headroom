# Dependency review

The companion is installed into the environment that already runs Headroom's
FastAPI proxy. It does not start another server or install `fastapi[standard]`.

| Requirement | Reason and version floor | Maintainer and install surface |
| --- | --- | --- |
| `fastapi>=0.100.0` | Reuses Headroom's existing proxy/dev dependency and version floor. Required for `APIRouter`, typed query validation, dependency guards and `JSONResponse`; rebuilding routing/security outside the host framework would create a second policy implementation. No newer FastAPI feature is required. | FastAPI is maintained by Sebastián Ramírez and the FastAPI team, with public releases and a [security policy](https://github.com/fastapi/fastapi/security/policy). Its normal dependency graph includes Starlette, Pydantic and their dependencies; installed Pydantic versions can include the native `pydantic-core` wheel. These packages are already in the proxy environment. |
| `pydantic>=2.0.0` | Validates strict boolean control payloads and rejects extra fields. Matches Headroom core's existing minimum; no new dependency family. | Maintained by the Pydantic team; uses the same Pydantic/pydantic-core installation already required by Headroom. |
| `setuptools>=77.0.3` (build only) | PEP 639 support is needed for the project's SPDX `license = "MIT"` metadata. This is a packaging requirement, not a runtime dependency. | Maintained by PyPA, with public releases and a [security policy](https://github.com/pypa/setuptools/security/policy). Build isolation can download a compatible setuptools release from the configured package index. The companion package itself contains Python source and has no install scripts or native compilation. |

Installing into an already compatible environment introduces no additional
runtime dependency family. An environment using FastAPI below Headroom's
existing minimum can be upgraded by pip; review the resolver output before
deployment. Headroom's lockfile and dependency-audit workflow remain the
source of exact repository versions and vulnerability qualification.
Upstream security policies and existing usage do not establish that every
future allowed version is vulnerability-free; human dependency review still
applies before merge.

The companion's runtime network surface is local HTTP with explicit conversation-scoped POST controls to the explicitly
configured local proxy. These dependencies do not require a package-index,
model-provider or telemetry request at runtime. The launcher separately
starts the user's existing Headroom and Claude executables; those retain
their own provider network behavior.
