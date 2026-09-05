"""Workflow regression tests for release publishing behavior."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from packaging.requirements import Requirement

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 fallback
    import tomli as tomllib  # type: ignore[no-redef]

ROOT = Path(__file__).resolve().parent.parent


def test_every_published_docker_variant_includes_bedrock_auth_dependencies() -> None:
    """Every image that advertises ``--backend bedrock`` must ship botocore.

    Temporary AWS credentials take LiteLLM's botocore-backed authentication
    path.  The default images previously installed only ``proxy``/``code``, so
    the documented Docker Bedrock command failed at runtime with
    ``No module named 'botocore'`` (#1551).  Keep the standalone Dockerfile and
    every bake target on the existing ``bedrock`` package extra.
    """
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    assert "ARG HEADROOM_EXTRAS=proxy,code,bedrock" in dockerfile

    bake = (ROOT / "docker-bake.hcl").read_text(encoding="utf-8")
    extras_lines = [
        line.strip() for line in bake.splitlines() if line.strip().startswith("HEADROOM_EXTRAS =")
    ]
    assert len(extras_lines) == 9
    assert all("bedrock" in line for line in extras_lines), extras_lines

    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    bedrock_names = {
        Requirement(requirement).name
        for requirement in project["project"]["optional-dependencies"]["bedrock"]
    }
    assert {"boto3", "botocore"} <= bedrock_names


def test_public_docker_instructions_use_the_current_organization_package() -> None:
    """Do not send users back to the personal GHCR package frozen at 0.27.0."""
    public_docs = (
        "README.md",
        "llms.txt",
        "docker-compose.yml",
        "TESTING-copilot-subscription.md",
        "wiki/cli.md",
        "wiki/docker-install.md",
    )
    deprecated = "ghcr.io/chopratejas/headroom"
    current = "ghcr.io/headroomlabs-ai/headroom"

    for relative_path in public_docs:
        content = (ROOT / relative_path).read_text(encoding="utf-8")
        assert deprecated not in content, relative_path
        assert current in content, relative_path


def test_docker_workflow_normalizes_repository_name_for_signing() -> None:
    content = (ROOT / ".github" / "workflows" / "docker.yml").read_text(encoding="utf-8")

    assert "id: image-name" in content
    assert "tr '[:upper:]' '[:lower:]'" in content
    assert "steps.image-name.outputs.image_name" in content


def test_docker_latest_promotion_is_owned_by_root_manifest_cell() -> None:
    workflow = yaml.safe_load((ROOT / ".github" / "workflows" / "docker.yml").read_text())
    jobs = workflow["jobs"]
    build = jobs["docker-build"]
    manifest = jobs["docker-manifest"]
    variants = manifest["strategy"]["matrix"]["variant"]
    build_variants = build["strategy"]["matrix"]["variant"]
    architectures = build["strategy"]["matrix"]["arch"]
    root = next(entry for entry in variants if entry["name"] == "")
    nonroot = next(entry for entry in variants if entry["name"] == "nonroot")
    promotion = next(
        step for step in manifest["steps"] if step["name"] == "Re-tag root image as :latest"
    )
    command = promotion["run"]

    assert len(variants) == 8
    assert [entry["name"] for entry in build_variants] == [entry["name"] for entry in variants]
    assert len(architectures) == 2
    assert {entry["platform"] for entry in architectures} == {"linux/amd64", "linux/arm64"}
    assert root["name"] == ""
    assert nonroot["name"] == "nonroot"
    assert "matrix.variant.name == ''" in promotion["if"]
    assert "steps.manifest.outputs.index_digest != ''" in promotion["if"]
    assert "steps.version.outputs.version != ''" in promotion["if"]
    assert (
        promotion["if"]
        == "steps.manifest.outputs.index_digest != '' && matrix.variant.name == '' && steps.version.outputs.version != ''"
    )
    assert '"${IMAGE}:latest"' in command
    assert '"${IMAGE}:${VERSION}"' in command
    assert "promote-latest" not in jobs
    assert manifest["needs"] == "docker-build"
    assert manifest["if"] == "${{ always() }}"
    step_names = [step["name"] for step in manifest["steps"]]
    assert step_names.index("Sign multi-arch index manifest with cosign") < step_names.index(
        "Re-tag root image as :latest"
    )
    manifest_script = next(
        step["run"] for step in manifest["steps"] if step["name"] == "Create multi-arch manifest"
    )
    assert 'digest_count="$(find "${DIGEST_DIR}" -maxdepth 1 -type f | wc -l)"' in manifest_script
    assert '"${digest_count}" -ne 2' in manifest_script
    assert manifest_script.index('"${digest_count}" -ne 2') < manifest_script.index(
        "docker buildx imagetools create"
    )
    guard_start = manifest_script.index('"${digest_count}" -ne 2')
    create_start = manifest_script.index("docker buildx imagetools create")
    assert guard_start < manifest_script.index("exit 1", guard_start) < create_start


def test_only_the_root_cell_can_ever_publish_a_bare_latest_tag() -> None:
    """`:latest` must have exactly one writer (#3150).

    The sibling test above proves the *promotion step* is owned by the root
    cell. That was never the whole contract: it checked the intended writer
    and not the absence of unintended ones.

    ``docker/metadata-action`` defaults to ``latest=auto``, which appends a
    bare ``latest`` for any semver release. Its own log line reads
    ``suffixLatest=false`` — the per-tag ``suffix=`` that keeps every other
    tag variant-scoped does not reach it. So all eight variant cells emitted
    ``ghcr.io/.../headroom:latest`` and the last to finish won the tag. At
    0.36.0 that was ``code-slim``: ``:latest`` resolved to the distroless
    build, whose ``import onnxruntime`` segfaults on arm64, and
    ``headroom deploy`` crash-looped on Apple Silicon with SIGSEGV.

    Two things hold the line, and both are asserted here: ``latest=false``
    stops the tag being generated at all, and a runtime guard refuses to
    publish if a suffixed variant ever carries one anyway.
    """
    workflow = yaml.safe_load((ROOT / ".github" / "workflows" / "docker.yml").read_text())
    manifest = workflow["jobs"]["docker-manifest"]

    meta = next(step for step in manifest["steps"] if step.get("id") == "meta")
    flavor = meta["with"].get("flavor", "")
    assert "latest=false" in flavor, (
        "docker/metadata-action defaults to latest=auto; without latest=false "
        "every variant cell publishes a bare :latest and the last one wins"
    )

    # No tag rule may reintroduce it explicitly either.
    assert "value=latest" not in meta["with"]["tags"]

    create = next(
        step for step in manifest["steps"] if step["name"] == "Create multi-arch manifest"
    )
    script = create["run"]
    # The guard reads the variant name from env, never spliced inline.
    assert create["env"].get("VARIANT_NAME") == "${{ matrix.variant.name }}"
    assert 'endswith(":latest")' in script
    assert script.index('endswith(":latest")') < script.index("docker buildx imagetools create"), (
        "the bare-latest guard must run before anything is pushed"
    )


def test_docker_manifest_downloads_exactly_one_artifact_per_architecture() -> None:
    """Each manifest cell must download exactly its two architecture digests.

    Keeping the variant before a trailing wildcard makes prefix-related names
    overlap: ``digests-code-*`` also selects code-nonroot, code-slim, and
    code-slim-nonroot. The 0.35.0 Docker release exposed this by downloading
    eight markers into the code manifest job instead of two.
    """
    workflow = yaml.safe_load((ROOT / ".github" / "workflows" / "docker.yml").read_text())
    jobs = workflow["jobs"]
    build = jobs["docker-build"]
    manifest = jobs["docker-manifest"]
    upload = next(step for step in build["steps"] if step.get("name") == "Upload digest marker")
    downloads = [
        step
        for step in manifest["steps"]
        if step.get("name")
        in {
            "Download amd64 digest for this variant",
            "Download arm64 digest for this variant",
        }
    ]

    assert upload["with"]["name"] == (
        "digests-${{ matrix.variant.name || 'root' }}-${{ matrix.arch.name }}"
    )
    assert [step["with"]["name"] for step in downloads] == [
        "digests-${{ matrix.variant.name || 'root' }}-amd64",
        "digests-${{ matrix.variant.name || 'root' }}-arm64",
    ]
    assert all("pattern" not in step["with"] for step in downloads)
    assert all(step["with"]["path"] == "${{ runner.temp }}/digests" for step in downloads)


def test_release_workflow_publishes_both_node_packages_to_github_packages() -> None:
    content = (ROOT / ".github" / "workflows" / "release.yml").read_text(encoding="utf-8")

    assert "Publish ${{ env.NPM_SDK_PACKAGE }} to GitHub Package Registry" in content
    assert "Publish ${{ env.NPM_OPENCLAW_PACKAGE }} to GitHub Package Registry" in content
    assert "pkg.name = `@${process.env.GITHUB_PACKAGES_SCOPE}/${pkg.name}`;" in content
    assert (
        'unscoped_sdk_tarball="$(npm pack --pack-destination "$assets_dir" | tail -n 1)"' in content
    )
    assert "SDK_TARBALL: ${{ steps.gpr-sdk-publish.outputs.unscoped_sdk_tarball }}" in content


def test_release_workflow_publishes_python_distributions_to_github_release() -> None:
    content = (ROOT / ".github" / "workflows" / "release.yml").read_text(encoding="utf-8")

    assert "Publish ${{ env.PYPI_PACKAGE }} Python distributions to GitHub Release" in content
    assert (
        'gh release upload "$TAG" release-assets/*.whl release-assets/*.tar.gz --clobber' in content
    )
    assert "Publish Node package tarballs to GitHub Release" in content
    assert 'gh release upload "$TAG" release-assets/*.tgz --clobber' in content


def test_create_release_requires_successful_build_and_pypi_publish() -> None:
    content = (ROOT / ".github" / "workflows" / "release.yml").read_text(encoding="utf-8")

    # Single-wheel maturin refactor (PR #360) added `build-wheels` (the
    # cross-platform matrix that produces the linux/macos/aarch64 wheels)
    # and `collect-dist` (aggregator that merges wheel artifacts + npm
    # release-assets) between `build` and the publish jobs. create-release
    # must wait for all of them.
    # PR #387 (X1) added `smoke-import-wheels` — the runtime gate that
    # actually loads the wheel on a customer-representative environment
    # before publish. create-release must wait for it AND require its
    # success in the `if:` block (otherwise `always()` would let the
    # release proceed even when the smoke gate failed).
    assert (
        "needs: [detect-version, build, build-wheels, collect-dist, smoke-import-wheels, publish-pypi, publish-npm, publish-github-packages, publish-docker]"
        in content
    )
    assert "always()" in content
    assert "needs.build.result == 'success'" in content
    assert "needs.build-wheels.result == 'success'" in content
    assert "needs.collect-dist.result == 'success'" in content
    assert "needs.smoke-import-wheels.result == 'success'" in content
    assert "(vars.PYPI_SKIP == 'true' || needs.publish-pypi.result == 'success')" in content


def test_macos_native_wrapper_dependency_install_retries_pypi_downloads() -> None:
    content = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")

    assert "python -m pip install --retries 10 --timeout 60 pytest" in content


def test_ci_commitlint_runs_only_for_pull_requests() -> None:
    content = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")

    assert "github.event_name == 'pull_request'" in content


def test_no_openssl_sys_in_wheel_build_tree() -> None:
    """STRUCTURAL INVARIANT: openssl-sys must NOT appear in the wheel
    build's resolved dependency graph.

    This is the load-bearing assertion for the entire build pipeline.
    If openssl-sys is in the wheel-build resolution graph, every
    Linux/macOS surface that builds from source needs system OpenSSL
    + perl modules + pkg-config — and we've spent five hot-fixes
    chasing whichever combination of perl modules / OpenSSL versions
    / pkg-config paths was missing in each manylinux/Dockerfile/
    devcontainer surface. The cleanest fix is to NOT depend on
    OpenSSL at all.

    fastembed exposes `hf-hub-rustls-tls` and
    `ort-download-binaries-rustls-tls` features that replace its
    default `native-tls` path. With `default-features = false` plus
    those rustls features enabled in headroom-core, our entire build
    tree uses rustls and no crate pulls openssl-sys.

    This test runs `cargo tree` (so it actually exercises the
    resolved feature graph, not just declared Cargo.toml features).
    A future refactor that adds a transitive native-tls user will
    fail here, surfaced at PR time rather than 5 minutes into a CI
    wheel-build error.
    """
    import subprocess

    for crate in ("headroom-py", "headroom-proxy", "headroom-core"):
        try:
            result = subprocess.run(
                [
                    "cargo",
                    "tree",
                    "--target",
                    "x86_64-unknown-linux-gnu",
                    "-p",
                    crate,
                    "-i",
                    "openssl-sys",
                ],
                cwd=str(ROOT),
                capture_output=True,
                text=True,
                check=False,
            )
        except FileNotFoundError:
            pytest.skip("cargo is unavailable in this environment")
        # `cargo tree -i <pkg>` exits 101 with "did not match any
        # packages" when the package is NOT in the tree — the GREEN
        # case. Exit 0 with a tree of consumers means it IS pulled.
        not_in_tree = result.returncode != 0 and "did not match any packages" in result.stderr
        if (
            result.returncode != 0
            and "package ID specification `openssl-sys` did not match"
            not in (result.stderr + result.stdout)
        ):
            pytest.skip(
                "cargo dependency tree for the Linux wheel target is unavailable in this environment"
            )
        assert not_in_tree, (
            f"openssl-sys is back in {crate}'s build tree:\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}\n"
            "Find the new native-tls user (likely a default-features=true "
            "on a transitive crate) and disable it. Switching every "
            "transitive HTTP+TLS consumer to rustls is the load-bearing "
            "invariant that keeps wheel builds working without system "
            "OpenSSL or perl modules."
        )


def test_no_native_tls_in_wheel_build_tree() -> None:
    """The dual of the openssl-sys gate: native-tls is the proximate
    cause of openssl-sys being pulled. Catch it earlier with a more
    specific error message so future debugging starts at the right
    place.
    """
    import subprocess

    for crate in ("headroom-py", "headroom-proxy", "headroom-core"):
        try:
            result = subprocess.run(
                [
                    "cargo",
                    "tree",
                    "--target",
                    "x86_64-unknown-linux-gnu",
                    "-p",
                    crate,
                    "-i",
                    "native-tls",
                ],
                cwd=str(ROOT),
                capture_output=True,
                text=True,
                check=False,
            )
        except FileNotFoundError:
            pytest.skip("cargo is unavailable in this environment")
        # Same environment gates as the openssl-sys dual above: a cargo
        # failure that is not "package did not match" means the Linux wheel
        # target is unavailable here, not that native-tls came back.
        not_in_tree = result.returncode != 0 and "did not match any packages" in result.stderr
        if result.returncode != 0 and "package ID specification `native-tls` did not match" not in (
            result.stderr + result.stdout
        ):
            pytest.skip(
                "cargo dependency tree for the Linux wheel target is unavailable in this environment"
            )
        assert not_in_tree, (
            f"native-tls is back in {crate}'s build tree — likely some "
            f"crate's `default-features = true` re-enabled native-tls "
            f"transitively:\n{result.stdout}"
        )


def test_fastembed_uses_rustls_features() -> None:
    """The mechanism that keeps openssl-sys out of the build is
    fastembed's explicit rustls feature selection in headroom-core.
    fastembed's default features include `hf-hub-native-tls` (pulls
    openssl-sys). Disabling defaults and enabling the rustls
    equivalent removes the OpenSSL surface entirely.
    """
    cargo = (ROOT / "crates" / "headroom-core" / "Cargo.toml").read_text(encoding="utf-8")

    assert "default-features = false" in cargo
    assert '"hf-hub-rustls-tls"' in cargo
    # `image-models` is in default; we re-enable it explicitly so we
    # don't lose the image-embedding capability when defaults are off.
    assert '"image-models"' in cargo


def test_fastembed_uses_dynamic_ort_everywhere() -> None:
    """No build may statically link Pyke's prebuilt ORT binaries.

    `ort-download-binaries-*` emits platform SDK link libs (DirectML on
    Windows; no prebuilts for `x86_64-apple-darwin`) and its Linux/macOS
    binaries require AVX2 at load time, SIGILLing `import headroom._core`
    on pre-AVX2 x86-64 CPUs (#1278). Every platform loads ORT dynamically
    (`ort-load-dynamic`), resolved at runtime from the pip `onnxruntime`
    package by `headroom/_ort.py` / the crate's loader guard.
    """

    cargo = (ROOT / "crates" / "headroom-core" / "Cargo.toml").read_text(encoding="utf-8")
    dependency_lines = "\n".join(
        line for line in cargo.splitlines() if not line.lstrip().startswith("#")
    )
    assert '"ort-load-dynamic"' in dependency_lines
    assert "ort-download-binaries" not in dependency_lines


def test_dockerfiles_no_longer_install_openssl_devel() -> None:
    """Once openssl-sys is out of the build tree, every Dockerfile
    that used to install `openssl-devel` / `libssl-dev` for the Rust
    build can drop those packages. This test enforces the cleanup so
    a future refactor doesn't carry the old packages forward "just
    in case".

    The check looks only at non-comment lines so explanatory comments
    that mention the historical packages don't false-positive.
    """
    targets = [
        ROOT / "e2e" / "wrap" / "Dockerfile",
        ROOT / "e2e" / "init" / "Dockerfile",
        ROOT / "Dockerfile",
        ROOT / ".devcontainer" / "Dockerfile",
    ]

    forbidden = ["openssl-devel", "libssl-dev"]

    for target in targets:
        content = target.read_text(encoding="utf-8")
        non_comment = "\n".join(
            line for line in content.splitlines() if not line.lstrip().startswith("#")
        )
        for pkg in forbidden:
            assert pkg not in non_comment, (
                f"{target.relative_to(ROOT)} still installs {pkg!r} on a "
                f"non-comment line. The rustls-everywhere refactor removed "
                f"openssl-sys from the build tree; this package is no "
                f"longer needed."
            )


def test_release_yml_does_not_install_openssl_or_perl_for_wheels() -> None:
    """With openssl-sys out of the build tree (verified by
    test_no_openssl_sys_in_wheel_build_tree), the previous
    before-script-linux that installed perl-IPC-Cmd / perl /
    perl-utils for the openssl-src vendored Configure script is
    obsolete. Removing it speeds the wheel build and keeps the
    Linux entry honest — every package install we keep here
    represents a hidden assumption about the manylinux container.
    """
    content = (ROOT / ".github" / "workflows" / "release.yml").read_text(encoding="utf-8")

    bw_start = content.index("\n  build-wheels:")
    bw_end = content.index("\n  collect-dist:")
    body = content[bw_start:bw_end]

    non_comment = "\n".join(line for line in body.splitlines() if not line.lstrip().startswith("#"))

    # No legacy install commands or env vars must appear on a non-comment
    # line. Each forbidden token represents an assumption about system
    # OpenSSL that the rustls refactor removed.
    forbidden = [
        "openssl-devel",
        "libssl-dev",
        "perl-IPC-Cmd",
        "libipc-cmd-perl",
        "OPENSSL_DIR",
    ]
    for token in forbidden:
        assert token not in non_comment, (
            f"release.yml build-wheels job still references {token!r} on "
            f"a non-comment line. The rustls-everywhere refactor removed "
            f"openssl-sys from the build tree; this command/env is now "
            f"obsolete."
        )


def test_build_wheels_matrix_includes_intel_macos_with_dynamic_ort() -> None:
    """Intel macOS wheels use `ort-load-dynamic` because `ort-sys 2.0.0-rc.12`
    has no prebuilt ONNX Runtime binaries for `x86_64-apple-darwin`.

    We assert against the actual matrix entry shape (`target: <triple>`
    on a non-comment line) so explanatory comments mentioning other
    triples don't false-positive.
    """
    content = (ROOT / ".github" / "workflows" / "release.yml").read_text(encoding="utf-8")

    bw_start = content.index("\n  build-wheels:")
    bw_end = content.index("\n  collect-dist:")
    body = content[bw_start:bw_end]

    matrix_targets: list[str] = []
    for raw in body.splitlines():
        stripped = raw.lstrip()
        # Skip YAML comments — only look at real matrix-entry lines.
        if stripped.startswith("#"):
            continue
        if stripped.startswith("target:"):
            # `target: x86_64-apple-darwin` → `x86_64-apple-darwin`
            matrix_targets.append(stripped.split(":", 1)[1].strip())

    assert "aarch64-apple-darwin" in matrix_targets, "Apple Silicon must stay in the matrix"
    assert "x86_64-unknown-linux-gnu" in matrix_targets
    assert "aarch64-unknown-linux-gnu" in matrix_targets
    assert "x86_64-apple-darwin" in matrix_targets, (
        f"x86_64-apple-darwin must be a wheel-matrix target; got {matrix_targets}"
    )

    matrix_os: list[str] = []
    for raw in body.splitlines():
        stripped = raw.lstrip()
        if stripped.startswith("#"):
            continue
        if stripped.startswith("os:"):
            matrix_os.append(stripped.split(":", 1)[1].strip())
        elif stripped.startswith("- os:"):
            matrix_os.append(stripped.split(":", 1)[1].strip())
    assert "macos-15-intel" in matrix_os


def test_smoke_import_macos_selects_wheel_arch_from_target() -> None:
    """The macOS smoke-import step must pick the wheel tag from the matrix
    target (arm64 for Apple Silicon, x86_64 for Intel) instead of
    hardcoding `_arm64` for every macOS row."""
    content = (ROOT / ".github" / "workflows" / "release.yml").read_text(encoding="utf-8")

    step_start = content.index("- name: Smoke-import wheel on macOS host")
    step_end = content.index("- name: Smoke-import wheel on Windows host", step_start)
    macos_block = content[step_start:step_end]

    assert "WHEEL_TARGET: ${{ matrix.wheel_target }}" in macos_block
    assert "aarch64-apple-darwin) mac_arch=arm64" in macos_block
    assert "x86_64-apple-darwin) mac_arch=x86_64" in macos_block
    assert "macosx_*_${mac_arch}.whl" in macos_block
    assert "headroom_ai-*-${py_tag}-${py_tag}-macosx_*_arm64.whl" not in macos_block
    assert "headroom_ai-*-abi3-macosx_*_arm64.whl" not in macos_block


def test_aarch64_wheel_uses_native_arm64_runner() -> None:
    """STRUCTURAL INVARIANT: the aarch64 wheel matrix row must run on a
    native arm64 runner (`ubuntu-24.04-arm`), NOT a QEMU-emulated x64
    runner (`ubuntu-latest`).

    Pre-#377 we built the aarch64 wheel on `ubuntu-latest` (x86_64) inside
    `manylinux_2_28_aarch64` via QEMU emulation, taking ~50–60 min. Native
    arm64 GitHub-hosted runners (GA Jan 2025, free for public repos) drop
    QEMU and complete the same build in ~10 min.

    A future "let me unify all wheel rows on `ubuntu-latest`" refactor
    would silently re-introduce QEMU and slow CI back down — this test
    pins the runner.
    """
    content = (ROOT / ".github" / "workflows" / "release.yml").read_text(encoding="utf-8")

    bw_start = content.index("\n  build-wheels:")
    bw_end = content.index("\n  collect-dist:")
    body = content[bw_start:bw_end]

    # Walk the matrix.include rows. Each row is a contiguous block of
    # `key: value` lines starting with `os:` (the first key in our
    # convention). Pair `os:` with the immediately-following `target:`
    # so we can assert per-row.
    rows: list[dict[str, str]] = []
    current: dict[str, str] = {}
    for raw in body.splitlines():
        stripped = raw.lstrip()
        if stripped.startswith("#"):
            continue
        if stripped.startswith("- os:"):
            if current:
                rows.append(current)
            current = {"os": stripped.split(":", 1)[1].strip()}
        elif stripped.startswith("target:") and current:
            current["target"] = stripped.split(":", 1)[1].strip()
        elif stripped.startswith("manylinux:") and current:
            current["manylinux"] = stripped.split(":", 1)[1].strip()
    if current:
        rows.append(current)

    aarch64_linux = [r for r in rows if r.get("target") == "aarch64-unknown-linux-gnu"]
    assert len(aarch64_linux) == 1, f"expected exactly one aarch64-linux row; got {aarch64_linux}"
    assert aarch64_linux[0]["os"] == "ubuntu-24.04-arm", (
        f"aarch64-unknown-linux-gnu must run on native arm64 runner "
        f"`ubuntu-24.04-arm`, not {aarch64_linux[0]['os']!r}. Reverting "
        f"to `ubuntu-latest` re-introduces QEMU emulation and ~6× slower "
        f"wheel builds."
    )

    # The amd64 Linux row should also be pinned to ubuntu-24.04 (not
    # `ubuntu-latest`, which is a moving target). Pinning keeps the
    # wheel-build environment reproducible across runner image rolls.
    amd64_linux = [r for r in rows if r.get("target") == "x86_64-unknown-linux-gnu"]
    assert len(amd64_linux) == 1
    assert amd64_linux[0]["os"] == "ubuntu-24.04", (
        f"x86_64-unknown-linux-gnu should pin `ubuntu-24.04`, not "
        f"{amd64_linux[0]['os']!r} — `ubuntu-latest` is a moving alias "
        f"and reproducibility benefits from explicit pinning."
    )


def test_docker_workflow_builds_on_native_arch_runners() -> None:
    """STRUCTURAL INVARIANT: the docker variant build must fan out per
    arch onto native runners — `linux/amd64` on `ubuntu-24.04`,
    `linux/arm64` on `ubuntu-24.04-arm`. No QEMU.

    Pre-#377 each variant ran `docker bake` with
    `platforms = ["linux/amd64","linux/arm64"]` on a single x64 runner
    using QEMU for arm64 emulation — ~1h per variant. Splitting into
    16 native single-arch builds (8 variants × 2 arches) + a manifest
    merge job per variant cuts wall-clock to ~10 min and removes the
    QEMU surface that contributed to transient build failures.
    """
    content = (ROOT / ".github" / "workflows" / "docker.yml").read_text(encoding="utf-8")

    # The fan-out job must exist with both runners in its arch matrix.
    assert "docker-build:" in content, "docker-build fan-out job missing"
    assert "runs_on: ubuntu-24.04, platform: linux/amd64" in content, (
        "amd64 arch matrix entry must bind ubuntu-24.04 (native x86_64)"
    )
    assert "runs_on: ubuntu-24.04-arm, platform: linux/arm64" in content, (
        "arm64 arch matrix entry must bind ubuntu-24.04-arm (native aarch64)"
    )

    # Per-arch builds must push by digest only — tags belong on the
    # multi-arch manifest, applied later by docker-manifest.
    assert "push-by-digest=true,name-canonical=true,push=true" in content, (
        "per-arch builds must push by digest only; tags applied at manifest merge step"
    )

    # The QEMU action must NOT be invoked anywhere — its presence would
    # mean someone re-introduced an emulated build path.
    non_comment = "\n".join(
        line for line in content.splitlines() if not line.lstrip().startswith("#")
    )
    assert "docker/setup-qemu-action" not in non_comment, (
        "docker.yml must not invoke `docker/setup-qemu-action` — native "
        "arm64 runners replaced QEMU. A new reference here means someone "
        "re-emulated arm64 on an x64 runner."
    )

    # Manifest merge job must exist and depend on docker-build.
    assert "docker-manifest:" in content
    assert "needs: docker-build" in content
    assert "docker buildx imagetools create" in content


def test_docker_per_arch_build_specifies_image_name_in_output() -> None:
    """STRUCTURAL INVARIANT: the per-arch bake's `*.output` spec must
    include `name=<registry>/<image>` — without it, buildx fails with
    the misleading `ERROR: tag is needed when pushing to registry`.

    Background: pre-#377 each docker variant ran with bake-file-tags
    (multi-arch tagged push), which gave bake the registry/image name
    via the tag strings. PR #376 split into per-arch fan-out and
    correctly removed bake-file-tags from the per-arch step (tags
    belong on the multi-arch manifest, not on per-arch images). But
    that left bake without ANY reference for the push target — no
    tags AND no explicit `name=` in the output spec.

    The first release after #376 merged failed every docker-build job
    with "ERROR: tag is needed when pushing to registry". The fix is
    to explicitly pass `name=<registry>/<image>` in the output spec
    so bake knows the push target without needing tags.

    A future refactor that removes the explicit name (e.g., "we
    already have labels, surely buildx can figure it out") will
    silently re-break this. This test pins it.
    """
    content = (ROOT / ".github" / "workflows" / "docker.yml").read_text(encoding="utf-8")

    # Find the per-arch build's *.output set line. Must contain
    # `name=` with the registry+image-name expression.
    output_line_present = (
        "*.output=type=image,name=${{ env.REGISTRY }}/${{ steps.image-name.outputs.image_name }},push-by-digest=true,name-canonical=true,push=true"
        in content
    )
    assert output_line_present, (
        "per-arch bake `*.output` must include `name=<registry>/<image>`. "
        "Without it, buildx fails the push with 'tag is needed when pushing "
        "to registry' because no tags AND no explicit name = no push target. "
        "This is a regression of the docker-build break right after PR #376."
    )


def test_sdist_build_conditional_keyed_on_target_not_os() -> None:
    """STRUCTURAL INVARIANT: the sdist build's `if` conditional must
    key on `matrix.target`, not `matrix.os`.

    Background: PR #376 changed the wheel matrix from `os: ubuntu-latest`
    to `os: ubuntu-24.04` (explicit pinning, no semantic change in
    practice). It silently broke the sdist build, whose `if` was
    `matrix.os == 'ubuntu-latest' && matrix.target == 'x86_64-unknown-linux-gnu'`
    — the literal `'ubuntu-latest'` no longer matched. Sdist never
    built, `release-assets/*.tar.gz` was empty, and the create-release
    job failed `gh release upload release-assets/*.tar.gz` with
    "no matches found".

    The fix is to key the conditional on `matrix.target` only — sdist
    is platform-independent, so any single matrix row is a fine host.
    `target` is more semantically meaningful than `os` here AND is
    decoupled from any future host-runner rename.

    This test pins the `target`-only conditional so a future "let's
    add `os` back to the conditional for clarity" refactor will fail
    at PR time, not 8 minutes into a release.
    """
    content = (ROOT / ".github" / "workflows" / "release.yml").read_text(encoding="utf-8")

    # Locate the "Build sdist" step.
    sdist_marker = "name: Build sdist"
    assert sdist_marker in content, "sdist build step missing from release.yml"

    # Walk forward to the next `if:` line — that's the conditional.
    sdist_idx = content.index(sdist_marker)
    if_idx = content.index("if:", sdist_idx)
    if_line_end = content.index("\n", if_idx)
    if_line = content[if_idx:if_line_end]

    # Must reference `matrix.target`. Must NOT reference `matrix.os`.
    assert "matrix.target == 'x86_64-unknown-linux-gnu'" in if_line, (
        f"sdist build conditional must check `matrix.target`; got: {if_line!r}"
    )
    assert "matrix.os" not in if_line, (
        f"sdist build conditional must NOT depend on `matrix.os` — that's "
        f"how PR #376 silently disabled the sdist build. Got: {if_line!r}"
    )


def test_release_workflow_verifies_versions_before_build_outputs() -> None:
    """Release sync must be followed by an explicit cross-package version gate."""
    content = (ROOT / ".github" / "workflows" / "release.yml").read_text(encoding="utf-8")

    assert "scripts/verify-versions.py" in content
    assert "scripts/version-sync.py" in content
    assert content.count("python scripts/verify-versions.py") >= 2

    first_sync = content.index("python scripts/version-sync.py --version")
    first_verify = content.index("python scripts/verify-versions.py", first_sync)
    changelog = content.index("name: Run changelog generation", first_verify)
    assert first_sync < first_verify < changelog

    second_sync = content.index("python scripts/version-sync.py --version", first_verify)
    second_verify = content.index("python scripts/verify-versions.py", second_sync)
    build_wheels = content.index("name: Build wheels", second_verify)
    assert second_sync < second_verify < build_wheels


def test_release_workflow_uses_local_npm_asset_builder() -> None:
    """npm tarball metadata must be built and verified by the reusable local gate."""
    content = (ROOT / ".github" / "workflows" / "release.yml").read_text(encoding="utf-8")
    builder = (ROOT / "scripts" / "build_npm_release_assets.mjs").read_text(encoding="utf-8")
    verifier = (ROOT / "scripts" / "verify_npm_release_assets.mjs").read_text(encoding="utf-8")

    assert (
        'node scripts/build_npm_release_assets.mjs "${{ needs.detect-version.outputs.npm_version }}" release-assets'
        in content
    )
    build_start = content.index("name: Build npm release packages")
    upload_start = content.index("name: Upload release assets artifact", build_start)
    build_block = content[build_start:upload_start]
    assert "npm pack" not in build_block, (
        "release.yml must not reimplement npm packing inline; the script "
        "regenerates OpenClaw dist metadata and runs install/import smoke checks."
    )

    assert "scripts/build_npm_release_assets.mjs" in content
    assert "scripts/verify_npm_release_assets.mjs" in content
    assert "scripts/verify_npm_release_assets.mjs" in builder
    assert "registerHeadroomPlugin" in verifier


def test_npm_release_builder_regenerates_openclaw_dist_metadata_after_rewrite() -> None:
    """OpenClaw's packed dist/package.json must see the release dependency."""
    builder = (ROOT / "scripts" / "build_npm_release_assets.mjs").read_text(encoding="utf-8")

    rewrite = builder.index("rewriteOpenClawReleaseDependency();")
    prepare_dist = builder.index('runNode(["prepare-dist.mjs"], openClawDir);', rewrite)
    pack = builder.index(
        'runNpm(["pack", "--pack-destination", assetsDir], openClawDir);', prepare_dist
    )
    verify = builder.index(
        'runNode(["scripts/verify_npm_release_assets.mjs", assetsDir, version], rootDir)', pack
    )

    assert rewrite < prepare_dist < pack < verify


def test_npm_release_builder_installs_openclaw_against_local_sdk_tarball() -> None:
    """The OpenClaw build must not require the release SDK to exist on npm."""
    builder = (ROOT / "scripts" / "build_npm_release_assets.mjs").read_text(encoding="utf-8")

    local_dependency = builder.index("rewriteOpenClawLocalDependency(sdkTarballPath);")
    install = builder.index(
        '["install", "--package-lock=false", "--no-audit", "--no-fund", "--ignore-scripts"]',
        local_dependency,
    )
    build = builder.index('runNpm(["run", "build"], openClawDir);', install)
    release_dependency = builder.index("rewriteOpenClawReleaseDependency();", build)

    assert local_dependency < install < build < release_dependency
    assert 'runNpm(["ci"], openClawDir)' not in builder


def test_openclaw_source_dependency_matches_lockfile_registry_range() -> None:
    """The source checkout must remain npm-ci installable before a release exists."""
    import json

    package_json = json.loads((ROOT / "plugins" / "openclaw" / "package.json").read_text())
    package_lock = json.loads((ROOT / "plugins" / "openclaw" / "package-lock.json").read_text())

    source_range = package_json["dependencies"]["headroom-ai"]
    lock_range = package_lock["packages"][""]["dependencies"]["headroom-ai"]

    assert source_range == lock_range == "^0.22.3"


def test_opencode_source_dependency_matches_lockfile_registry_range() -> None:
    """The source checkout must remain npm-ci installable before a release exists."""
    import json

    package_json = json.loads((ROOT / "plugins" / "opencode" / "package.json").read_text())
    package_lock = json.loads((ROOT / "plugins" / "opencode" / "package-lock.json").read_text())

    source_range = package_json["dependencies"]["headroom-ai"]
    lock_range = package_lock["packages"][""]["dependencies"]["headroom-ai"]

    assert source_range == lock_range == "^0.22.3"


def test_python_release_smoke_imports_installed_wheel_outside_source_tree() -> None:
    """The wheel smoke must not import the checkout package by accident."""
    script = (ROOT / "scripts" / "build_python_release_smoke.py").read_text(encoding="utf-8")

    assert "cwd: Path = ROOT" in script
    assert 'import_cwd = Path(tmp) / "import-cwd"' in script
    assert 'run([smoke_python, "-c", smoke_code], cwd=import_cwd)' in script


def test_publish_npm_regenerates_openclaw_dist_metadata_after_version_and_dependency() -> None:
    """The direct npm publish path must not ship stale OpenClaw dist metadata."""
    content = (ROOT / ".github" / "workflows" / "release.yml").read_text(encoding="utf-8")

    start = content.index("name: Publish ${{ env.NPM_OPENCLAW_PACKAGE }} to npmjs.org")
    end = content.index("continue-on-error: true", start)
    block = content[start:end]

    version = block.index('npm version "$version"')
    dependency = block.index('pkg.dependencies["headroom-ai"]')
    prepare_dist = block.index("node prepare-dist.mjs")
    publish = block.index("npm publish --access public")

    assert version < dependency < prepare_dist < publish


def test_publish_npm_rewrites_opencode_dependency_after_version_and_before_publish() -> None:
    """The direct npm publish path must version and retarget the Opencode dependency."""
    content = (ROOT / ".github" / "workflows" / "release.yml").read_text(encoding="utf-8")

    start = content.index("name: Publish ${{ env.NPM_OPENCODE_PACKAGE }} to npmjs.org")
    end = content.index("continue-on-error: true", start)
    block = content[start:end]

    version = block.index('npm version "$version"')
    dependency = block.index('pkg.dependencies["headroom-ai"]')
    publish = block.index("npm publish --access public")

    assert version < dependency < publish


def test_sdist_license_is_packaged_and_verified_before_upload() -> None:
    """STRUCTURAL INVARIANT: the sdist tarball must physically contain
    every license file PEP 639 declares in PKG-INFO, and the release
    workflow must verify that match before upload.

    PyPI rejects sdists whose `License-File:` metadata entries
    reference files missing from the tarball with `400 License-File X
    does not exist in distribution file ...`. Maturin's PEP 639
    auto-discovery emits both `LICENSE` and `NOTICE` into PKG-INFO
    because both files exist at the project root and match the default
    glob — but maturin sdists don't get the package-directory
    treatment wheels do, so each file must be explicitly listed in
    `[tool.maturin].include` with `format = "sdist"`. Issue trail:
    sdist publish broke at v0.20.16 (the hatch -> maturin migration
    in 2a91cbb dropped NOTICE from the include list), masked for ~22
    releases by an earlier twine `400 File already exists` failure on
    duplicate wheels, surfaced once PR #412 added skip-existing.
    """
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    release_yml = (ROOT / ".github" / "workflows" / "release.yml").read_text(encoding="utf-8")

    assert '{ path = "LICENSE", format = "sdist" }' in pyproject, (
        "pyproject.toml [tool.maturin].include must list LICENSE for sdist format"
    )
    assert '{ path = "NOTICE", format = "sdist" }' in pyproject, (
        "pyproject.toml [tool.maturin].include must list NOTICE for sdist format. "
        "Maturin's PEP 639 auto-discovery emits `License-File: NOTICE` into "
        "PKG-INFO because NOTICE exists at the project root, so the file MUST "
        "ship in the tarball or PyPI rejects the sdist with a 400."
    )
    assert "name: Verify sdist license-file metadata matches tarball contents" in release_yml, (
        "release.yml must run the License-File / tarball-contents cross-check before publish"
    )
    assert 'if line.startswith("License-File:")' in release_yml, (
        "release.yml verifier must parse PKG-INFO License-File entries — "
        "not just a hardcoded LICENSE check — so any future PEP 639-discoverable "
        "file (COPYING, AUTHORS, ...) is also gated."
    )
    assert "declares License-File entries that are missing from the tarball" in release_yml, (
        "release.yml verifier must fail loudly when declared license files "
        "are missing — silent passes would let the same regression resurface."
    )


def test_pypi_publish_failure_blocks_github_release() -> None:
    content = (ROOT / ".github" / "workflows" / "release.yml").read_text(encoding="utf-8")

    pypi_job_start = content.index("publish-pypi:")
    npm_job_start = content.index("publish-npm:", pypi_job_start)
    pypi_job = content[pypi_job_start:npm_job_start]

    assert "uses: pypa/gh-action-pypi-publish@v1.13.0" in pypi_job
    assert "continue-on-error: true" not in pypi_job
    assert "(vars.PYPI_SKIP == 'true' || needs.publish-pypi.result == 'success')" in content


def test_glibc_compat_shim_present_in_headroom_py() -> None:
    """STRUCTURAL INVARIANT: the headroom-py crate ships a glibc-2.38
    compatibility shim that defines weak `__isoc23_*` aliases.

    Issue #355 (https://github.com/headroomlabs-ai/headroom/issues/355) —
    the published wheel's `_core.so` references `__isoc23_strtoll`
    (glibc 2.38+) because we statically link prebuilt ONNX Runtime
    artifacts compiled with gcc 14. Users with libc < 2.38 (Ubuntu
    22.04, most Conda envs, Debian 11/12) hit:

        ImportError: undefined symbol: __isoc23_strtoll

    The fix is `crates/headroom-py/glibc_compat.c` which provides
    weak-alias definitions for the four `__isoc23_*` symbols,
    delegating to the older `strtol*` family. `build.rs` compiles
    the shim into `_core.so` on Linux/glibc only.

    A future "let me drop this weird C file, surely it's dead code"
    refactor would silently re-introduce the import failure for
    every user on glibc < 2.38. This test pins all three load-bearing
    pieces (the .c file, the build.rs trigger, the [build-dependencies]
    cc dep).
    """
    headroom_py_dir = ROOT / "crates" / "headroom-py"

    shim = headroom_py_dir / "glibc_compat.c"
    assert shim.exists(), (
        "crates/headroom-py/glibc_compat.c is missing — without it, "
        "`_core.so` fails to import on every glibc < 2.38 host. See "
        "issue #355 for the full bug class. NEVER delete this file "
        "without confirming via `scripts/audit_wheel_glibc_symbols.py` "
        "that the wheel no longer references __isoc23_* symbols."
    )
    shim_content = shim.read_text(encoding="utf-8")
    for sym in ("__isoc23_strtol", "__isoc23_strtoll", "__isoc23_strtoul", "__isoc23_strtoull"):
        assert sym in shim_content, f"shim missing alias for {sym}"

    build_rs = headroom_py_dir / "build.rs"
    assert build_rs.exists(), "crates/headroom-py/build.rs is missing"
    build_rs_content = build_rs.read_text(encoding="utf-8")
    assert "glibc_compat.c" in build_rs_content, (
        "build.rs must reference glibc_compat.c — otherwise Cargo "
        "skips the shim and the wheel's `_core.so` ships without it."
    )

    cargo_toml = (headroom_py_dir / "Cargo.toml").read_text(encoding="utf-8")
    assert 'build = "build.rs"' in cargo_toml, (
        'headroom-py/Cargo.toml must declare `build = "build.rs"` — '
        "Cargo only auto-detects build.rs when this is set; without "
        "it, the shim never compiles."
    )
    assert "[build-dependencies]" in cargo_toml and 'cc = "1"' in cargo_toml, (
        'headroom-py/Cargo.toml must declare `cc = "1"` in '
        "[build-dependencies] for build.rs to compile the C shim."
    )


def test_release_workflow_audits_wheel_glibc_symbols() -> None:
    """STRUCTURAL INVARIANT: the release workflow audits each Linux
    wheel for symbol references that exceed its manylinux glibc floor.

    Companion to `test_glibc_compat_shim_present_in_headroom_py` —
    the shim is the FIX, this audit is the GATE. Without the audit,
    a future toolchain bump in the prebuilt ORT artifacts (or any
    other statically-linked C/C++ dep) could re-introduce a
    post-floor symbol that our current shim doesn't cover. The audit
    catches that at release time, before publish-pypi.
    """
    content = (ROOT / ".github" / "workflows" / "release.yml").read_text(encoding="utf-8")

    assert "audit_wheel_glibc_symbols.py" in content, (
        "release.yml must invoke `scripts/audit_wheel_glibc_symbols.py` "
        "on every Linux wheel before publish. Without it, regressions "
        "of issue #355's bug class ship to PyPI silently."
    )
    assert "Audit wheel glibc symbols (Linux only)" in content, (
        "audit step name has been renamed; update both this test and the workflow"
    )


def test_release_workflow_has_smoke_import_wheel_gate() -> None:
    """STRUCTURAL INVARIANT: release.yml runs the just-built wheels
    through `import headroom._core` on a matrix of representative
    customer environments BEFORE publishing to PyPI / pushing to
    GHCR / cutting a GitHub Release.

    This is the X1 gate from the post-#355 hardening plan. Issue #355
    plus its three follow-on hotfixes (#384/#385/#386) all share a
    pattern: the wheel is technically valid (clippy passes, tests
    pass, auditwheel is happy) but fails to import on a customer's
    box because of a runtime symbol mismatch. Static gates can't
    catch that — only actually loading the .so does.

    Required matrix coverage:
    - manylinux floor we promise (`manylinux_2_28_x86_64` and
      `manylinux_2_28_aarch64`). If these fail, our manylinux tag
      is a lie.
    - At least one customer-representative glibc per arch (Ubuntu
      LTS, the issue #355 reporter's environment).
    - macOS native (Apple Silicon).

    Required gating: `publish-pypi`, `publish-docker`, AND
    `create-release` must all `needs:` smoke-import-wheels. A
    smoke failure has to BLOCK publish, not just produce a
    notification.

    A future "remove this slow CI step that always passes anyway"
    refactor — exactly the impulse that landed us PR #382's sdist
    gap and PR #386's link-order surprise — fails this test at
    PR time.
    """
    content = (ROOT / ".github" / "workflows" / "release.yml").read_text(encoding="utf-8")

    # The job itself must exist.
    assert "\n  smoke-import-wheels:" in content, (
        "release.yml must define a `smoke-import-wheels` job. This is the "
        "X1 gate that catches runtime symbol mismatches in the published "
        "wheel before it hits PyPI. Issue #355 + #384/#385/#386 are the "
        "canonical reason this gate exists."
    )

    # Required matrix entries — pin both the floor (manylinux_2_28)
    # and at least one customer environment per arch.
    required_matrix_substrings = [
        # manylinux floor for x86_64 — pins what we promise customers.
        'image: "quay.io/pypa/manylinux_2_28_x86_64"',
        # manylinux floor for aarch64 — would have caught PR #386.
        'image: "quay.io/pypa/manylinux_2_28_aarch64"',
        # At least one Ubuntu LTS — issue #355's environment was
        # ubuntu:22.04 + Python 3.12.
        'image: "ubuntu:22.04"',
        # macOS native (no container) — Apple Silicon wheel.
        "runner: macos-14",
    ]
    for sub in required_matrix_substrings:
        assert sub in content, (
            f"smoke matrix missing required entry: {sub!r}. The matrix "
            f"must cover the manylinux floor + at least one customer-"
            f"representative environment per arch + macOS native."
        )

    # Gating: publish-pypi must wait for the smoke job.
    publish_pypi_idx = content.index("\n  publish-pypi:")
    next_job_idx = content.index("\n  publish-npm:", publish_pypi_idx)
    publish_pypi_block = content[publish_pypi_idx:next_job_idx]
    assert "smoke-import-wheels" in publish_pypi_block, (
        "publish-pypi must `needs: [..., smoke-import-wheels]` — without "
        "the dependency, a broken wheel can be published before the "
        "smoke job has even finished. The whole point of X1 is that it "
        "BLOCKS publish."
    )

    # Same for publish-docker.
    publish_docker_idx = content.index("\n  publish-docker:")
    next_idx = content.index("\n  create-release:", publish_docker_idx)
    publish_docker_block = content[publish_docker_idx:next_idx]
    assert "smoke-import-wheels" in publish_docker_block, (
        "publish-docker must `needs: [..., smoke-import-wheels]` — the "
        "docker image bundles the same wheels; a broken wheel will fail "
        "the docker build's `pip install` 3 minutes later anyway. "
        "Failing fast in smoke saves matrix budget."
    )

    # And create-release.
    create_release_idx = content.index("\n  create-release:")
    create_release_block = content[create_release_idx:]
    assert "smoke-import-wheels" in create_release_block, (
        "create-release must `needs: [..., smoke-import-wheels]` and gate on its success"
    )
    assert "needs.smoke-import-wheels.result == 'success'" in create_release_block, (
        "create-release's `if:` must explicitly require "
        "`needs.smoke-import-wheels.result == 'success'` — without "
        "this, `always()` would let the release proceed even if the "
        "smoke gate failed."
    )

    # The actual import command must hit `from headroom._core import hello`
    # — this is the same call the proxy's `_check_rust_core` makes on
    # startup (per `headroom/proxy/server.py` and the issue #355 backtrace).
    # Anything else (e.g. just `import headroom`) fails to exercise the
    # Rust _core.so binary.
    assert "from headroom._core import hello" in content, (
        "smoke-import command must call `from headroom._core import hello` "
        "— that's what the proxy does at startup. A weaker check (e.g. "
        "`import headroom`) wouldn't exercise the .so and wouldn't catch "
        "the bugs the gate exists for."
    )


def test_npm_publish_jobs_do_not_download_dist_artifact() -> None:
    """`publish-npm` and `publish-github-packages` `npm pack`+`npm publish`
    directly from the checked-out source tree; they never read the
    Python `dist` artifact. The earlier speculative download was failing
    "Artifact not found" because neither job is gated on `collect-dist`.
    Ensure no future refactor re-adds the dead step.
    """
    content = (ROOT / ".github" / "workflows" / "release.yml").read_text(encoding="utf-8")

    # Locate publish-npm + publish-github-packages bodies and assert
    # neither contains a download-artifact step that pulls `name: dist`.
    npm_start = content.index("\n  publish-npm:")
    npm_end = content.index("\n  publish-github-packages:")
    publish_npm_body = content[npm_start:npm_end]

    gpr_start = content.index("\n  publish-github-packages:")
    gpr_end = content.index("\n  publish-docker:")
    publish_gpr_body = content[gpr_start:gpr_end]

    for body, label in (
        (publish_npm_body, "publish-npm"),
        (publish_gpr_body, "publish-github-packages"),
    ):
        assert "download-artifact" not in body, (
            f"{label} must not download the `dist` artifact — it `npm pack`s "
            f"its own tarball and the speculative download fails when "
            f"collect-dist hasn't run."
        )


def test_smoke_import_ubuntu_apt_installs_are_retried() -> None:
    """Ubuntu smoke-import containers must tolerate stale package mirrors.

    The ARM Ubuntu ports mirror can briefly serve indexes that point at a
    package version which has just been removed, causing apt install to fail
    with a 404 even after an update. Keep the smoke gate strict, but retry the
    package operations and use --fix-missing so transient mirror skew does not
    make unrelated PRs red.
    """
    content = (ROOT / ".github" / "workflows" / "release.yml").read_text(encoding="utf-8")
    smoke_start = content.index("\n  smoke-import-wheels:")
    smoke_end = content.index("\n  publish-pypi:", smoke_start)
    smoke_body = content[smoke_start:smoke_end]

    assert "apt_retry()" in smoke_body
    assert "apt_retry update -qq" in smoke_body
    assert "apt_retry install -y -qq --fix-missing --no-install-recommends" in smoke_body


def test_release_workflow_runs_dry_run_on_pull_request() -> None:
    """X2: the release workflow MUST trigger on `pull_request` for paths
    that change wheel-layout / release pipeline so the wheel matrix +
    smoke-import gate run BEFORE merge.

    Issues this gate would have caught at PR time instead of after-merge:
    - #379 (docker bake `name=` regression in PR #376)
    - #382 (sdist os-mismatch — `ubuntu-latest` → `ubuntu-24.04` rename)
    - #384 / #385 / #386 (glibc shim iterations — alias, link-order)
    - #387's heredoc-indent regression that broke main on first release
      run after merge

    Required:
    1. `pull_request:` trigger present.
    2. Path filter is narrow enough to skip source-only PRs to
       `crates/headroom-core` / `crates/headroom-proxy` (where wheel
       layout doesn't change), but wide enough to cover release.yml,
       docker.yml, headroom-py crate, pyproject.toml, root Cargo.
    3. publish-pypi / publish-npm / publish-github-packages /
       publish-docker / create-release ALL gate on
       `github.event_name != 'pull_request'` so a PR run never
       publishes anything — the dry-run is build+smoke only.
    4. concurrency.group is namespaced by PR number for PR runs and by
       ref_name for main runs, AND cancel-in-progress is true for PR
       runs (rapid PR pushes cancel stale dry-runs) and false for main
       runs (a tag-push release should never be cancelled mid-flight).

    A future "lighten CI by dropping the dry-run" refactor — exactly
    the impulse that gave us PR #382 and PR #387 — fails this test
    at PR time.
    """
    content = (ROOT / ".github" / "workflows" / "release.yml").read_text(encoding="utf-8")

    # 1. pull_request trigger present.
    on_block_end = content.index("\nconcurrency:")
    on_block = content[:on_block_end]
    assert "\n  pull_request:" in on_block, (
        "release.yml must trigger on `pull_request` so the wheel matrix + "
        "smoke-import gate run BEFORE merge. Without this, wheel-layout / "
        "release-pipeline regressions are only caught after the tag is "
        "pushed and main is broken (see #382, #387 for canonical examples)."
    )

    # 2. Path filter covers wheel-layout-affecting paths.
    pr_idx = content.index("\n  pull_request:")
    pr_end = content.index("\n  workflow_dispatch:", pr_idx)
    pr_block = content[pr_idx:pr_end]
    required_paths = [
        ".github/workflows/release.yml",
        ".github/workflows/docker.yml",
        "crates/headroom-py/**",
        "pyproject.toml",
        "Cargo.toml",
        "Cargo.lock",
    ]
    for path in required_paths:
        assert f'"{path}"' in pr_block, (
            f"pull_request path filter missing {path!r}. The dry-run must "
            f"trigger when this path changes — otherwise a regression in "
            f"that file lands on main without exercising the wheel matrix."
        )

    # 3. Each publish job + create-release gates on event_name != pull_request.
    publish_jobs = [
        ("publish-pypi", "\n  publish-npm:"),
        ("publish-npm", "\n  publish-github-packages:"),
        ("publish-github-packages", "\n  publish-docker:"),
        ("publish-docker", "\n  create-release:"),
    ]
    for job_name, next_marker in publish_jobs:
        start = content.index(f"\n  {job_name}:")
        end = content.index(next_marker, start)
        body = content[start:end]
        assert "github.event_name != 'pull_request'" in body, (
            f"{job_name} must gate on `github.event_name != 'pull_request'`. "
            f"Without this gate, a PR dry-run would attempt to publish — "
            f"in the best case the publish credentials are missing and the "
            f"job fails noisily; in the worst case it succeeds and a "
            f"non-merged PR ships to PyPI / npm / GHCR."
        )

    create_release_idx = content.index("\n  create-release:")
    create_release_block = content[create_release_idx:]
    assert "github.event_name != 'pull_request'" in create_release_block, (
        "create-release must gate on `github.event_name != 'pull_request'`. "
        "Without it, a PR dry-run would cut a GitHub Release for an unmerged "
        "branch."
    )

    # 4. Concurrency: PR runs use a per-PR group and DO cancel-in-progress;
    #    main runs use ref_name and DO NOT cancel.
    concurrency_idx = content.index("\nconcurrency:")
    jobs_idx = content.index("\njobs:", concurrency_idx)
    concurrency_block = content[concurrency_idx:jobs_idx]

    assert "github.event.pull_request.number" in concurrency_block, (
        "concurrency.group must include the PR number for pull_request runs "
        "(via `format('pr-{0}', github.event.pull_request.number)`) — "
        "otherwise PR runs collide with each other or with main."
    )
    assert "cancel-in-progress: ${{ github.event_name == 'pull_request' }}" in concurrency_block, (
        "concurrency.cancel-in-progress must be conditional: TRUE for "
        "pull_request (rapid PR pushes shouldn't queue N parallel wheel "
        "builds) and FALSE for main (a tag-push release that's mid-flight "
        "must not be cancelled — partial PyPI/Docker state is worse than "
        "a slow CI queue)."
    )


def test_release_yml_triggers_on_release_published_not_every_push_to_main() -> None:
    """release.yml fires when release-please publishes a release, not per main push.

    The prior trigger (`push: branches: [main]`) caused a fresh wheel
    matrix to be uploaded to PyPI for every merged `fix:`/`feat:` PR.
    PyPI enforces a 10 GiB per-project storage quota and the project
    breached it in May 2026 (publish-pypi failing on every main merge
    from PR #482 forward). The fix routes releases through
    release-please's release-PR pattern: bot opens/maintains a
    `chore: release vX.Y.Z` PR aggregating conventional-commit traffic;
    merging that PR creates the tag + GitHub Release; THAT release
    event is what triggers this workflow.

    Reverting to a per-push trigger would re-create the quota
    blowup. This test fails any refactor that does so silently.
    """
    content = (ROOT / ".github" / "workflows" / "release.yml").read_text(encoding="utf-8")
    on_block_end = content.index("\nconcurrency:")
    on_block = content[:on_block_end]

    assert "\n  release:\n    types: [published]" in on_block, (
        "release.yml must trigger on the `release: published` event so "
        "release-please's release-PR merge is the only way to publish — "
        "see .github/workflows/release-please.yml."
    )
    assert "\n  push:\n    branches: [main]" not in on_block, (
        "release.yml MUST NOT trigger on every push to main. That pattern "
        "burned PyPI's 10 GiB storage quota (one fresh wheel matrix per "
        "merged PR). Route releases through release-please instead."
    )


def test_release_yml_resolves_manual_ver_from_release_tag() -> None:
    """When fired by release event, MANUAL_VER must come from the release tag.

    release_version.py defaults to deriving the next version from git
    log + canonical pyproject.toml version. On a release-published
    run, that derivation would re-bump past the version the bot just
    tagged, producing wheels for the wrong version. The detect-version
    job must read `github.event.release.tag_name` and strip the leading
    `v` so the SemVer parser accepts it.
    """
    content = (ROOT / ".github" / "workflows" / "release.yml").read_text(encoding="utf-8")

    assert "Resolve MANUAL_VER from trigger" in content, (
        "detect-version must include a step that resolves MANUAL_VER from "
        "the trigger context (release.tag_name on release events; "
        "inputs.version on workflow_dispatch)."
    )
    assert "RELEASE_TAG: ${{ github.event.release.tag_name }}" in content, (
        "Resolver must read the tag from github.event.release.tag_name."
    )
    assert "${RELEASE_TAG#v}" in content, (
        "Resolver must strip the leading 'v' from the release tag — "
        "release_version.py's SemVer regex rejects 'v0.9.2'."
    )
    assert "MANUAL_VER: ${{ steps.manualver.outputs.value }}" in content, (
        "Compute-version step must consume the resolver's output."
    )


def test_release_yml_preserves_release_please_notes_when_release_exists() -> None:
    """create-release must not clobber release-please's auto-generated notes.

    release-please creates the GitHub Release with an auto-generated
    changelog body when its release PR merges. If create-release then
    runs `gh release edit --notes-file .changelog.md`, the bot's
    changelog gets overwritten with this workflow's full-history
    fallback (which has no `--since` bound when MANUAL_VER is set
    and previous_tag comes back empty). Keep the bot's notes intact;
    only update title.
    """
    content = (ROOT / ".github" / "workflows" / "release.yml").read_text(encoding="utf-8")
    create_release_idx = content.index("\n  create-release:")
    create_release_block = content[create_release_idx:]

    assert 'gh release edit "$TAG" --title "$TITLE"\n' in create_release_block, (
        "When the release already exists (release-please case), the edit "
        "must only sync title — NOT pass --notes-file, which would "
        "clobber the bot's auto-generated changelog."
    )


def test_release_please_workflow_exists_and_targets_main() -> None:
    """The release-please bot workflow must be present and watch main."""
    rp_path = ROOT / ".github" / "workflows" / "release-please.yml"
    assert rp_path.exists(), (
        "release-please.yml is the bot that opens/maintains the release "
        "PR. Without it, no release ever fires (release.yml now only "
        "triggers on the release event the bot emits)."
    )

    content = rp_path.read_text(encoding="utf-8")
    assert any(f"googleapis/release-please-action@v{v}" in content for v in (4, 5)), (
        "release-please.yml must use the v4 or v5 action — earlier versions "
        "have different manifest semantics."
    )
    assert "branches: [main]" in content, (
        "release-please.yml must watch main; that's where the bot reads "
        "conventional-commit traffic to compute version bumps."
    )
    assert "config-file: .release-please-config.json" in content
    assert "manifest-file: .release-please-manifest.json" in content
    assert "pull-requests: write" in content, (
        "Bot needs write permission to open/update its release PR."
    )
    assert "contents: write" in content, (
        "Bot needs contents write to tag the release commit on merge."
    )


def test_release_please_config_and_manifest_are_present_and_consistent() -> None:
    """Config and manifest must agree with pyproject.toml's version."""
    import json

    # tomllib is stdlib on 3.11+; tomli is the backport for 3.10 (which
    # the project still supports per pyproject.toml `requires-python`).
    # Matches the same fallback pattern in headroom/release_version.py.
    try:
        import tomllib
    except ModuleNotFoundError:  # pragma: no cover - Python 3.10 only
        import tomli as tomllib  # type: ignore[no-redef]

    manifest = json.loads((ROOT / ".release-please-manifest.json").read_text(encoding="utf-8"))
    config = json.loads((ROOT / ".release-please-config.json").read_text(encoding="utf-8"))
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))

    # Manifest tracks current version per package; the root package must
    # match pyproject.toml exactly. A drift here means the bot will
    # propose a version bump from the wrong base.
    assert manifest["."] == pyproject["project"]["version"], (
        f"manifest['.'] ({manifest['.']}) must match "
        f"pyproject.toml version ({pyproject['project']['version']}). "
        "Update the manifest when you bump pyproject.toml manually, or "
        "let release-please own both."
    )

    # Config: the root package must declare python release-type so the
    # bot updates pyproject.toml.
    root_pkg = config["packages"]["."]
    assert root_pkg["release-type"] == "python"
    assert root_pkg["package-name"] == "headroom-ai"

    # Tag format: existing tags in this repo are `vX.Y.Z`, NOT
    # `headroom-ai-vX.Y.Z`. release-please's default for manifest
    # configs prepends the component name; that would produce
    # `headroom-ai-v0.22.4` and the bot would never find the existing
    # `v0.22.3` baseline tag. include-component-in-tag MUST be false
    # to keep tag format consistent with the project's pre-bot tags.
    assert config.get("include-component-in-tag") is False, (
        "include-component-in-tag must be false — existing tags are "
        "`vX.Y.Z`, not `headroom-ai-vX.Y.Z`. Reverting this setting "
        "would orphan every prior tag and produce a months-long "
        "changelog because the bot can't find its baseline."
    )

    # This manifest has one package. Sending it through the merge plugin
    # produces the group title `chore: release main`, which contains neither
    # the package component nor its version. On merge, release-please cannot
    # associate that title with `headroom-ai`, leaves the PR tagged
    # `autorelease: pending`, and never emits the release event that publishes
    # to PyPI. Keep the single package on the normal, versioned PR path and
    # preserve the component in the title used to match the merged PR.
    assert config.get("separate-pull-requests") is True, (
        "The single root package must bypass release-please's merge plugin; "
        "its grouped PR title is `chore: release main` and cannot be tagged."
    )
    assert config.get("pull-request-title-pattern") == ("chore: release${component} ${version}"), (
        "Release PR titles must include both component and version so "
        "release-please can match the merged PR back to headroom-ai."
    )

    # extra-files: TypeScript SDK and npm plugin package.json files
    # files must be in lockstep with pyproject.toml.
    extra_paths = {ef["path"] for ef in root_pkg.get("extra-files", [])}
    assert "sdk/typescript/package.json" in extra_paths, (
        "release-please must bump sdk/typescript/package.json so the npm "
        "publish in release.yml ships the same version as the wheel."
    )
    assert "plugins/openclaw/package.json" in extra_paths, (
        "release-please must bump plugins/openclaw/package.json so the "
        "openclaw npm publish stays in sync."
    )
    assert "plugins/opencode/package.json" in extra_paths, (
        "release-please must bump plugins/opencode/package.json so the "
        "opencode npm publish stays in sync."
    )


def test_release_metadata_sync_runs_on_release_please_branch() -> None:
    """The release branch must self-heal the versions release-please does not bump.

    release-please rewrites `pyproject.toml` plus its configured `extra-files` only.
    `server.json` is asserted byte-for-byte against `render_server_json()`, which
    reads the version from `pyproject.toml`, so a bump without a sync fails
    `test_root_server_json_matches_builder` on the release PR — that is what blocked
    v0.33.0 (#2339). `release.yml` syncs in-workspace before its own gate, but the
    regular CI test job does not, so the sync has to be committed to the branch.
    """
    content = (ROOT / ".github" / "workflows" / "release-metadata-sync.yml").read_text(
        encoding="utf-8"
    )

    # Keyed off a push to the release branch: release-please force-regenerates that
    # branch on every merge to main, which is what wiped the hand-pushed fixes.
    assert '"release-please--branches--**"' in content
    assert "contents: write" in content

    # Sync, then gate on the verifier, then commit — in that order.
    sync = content.index("python scripts/version-sync.py")
    verify = content.index("python scripts/verify-versions.py", sync)
    commit = content.index("git commit", verify)
    assert sync < verify < commit

    # Must no-op rather than loop when the branch is already in sync.
    assert "git diff --quiet" in content

    # A GITHUB_TOKEN push would not re-trigger the release PR's checks.
    assert "RELEASE_PLEASE_TOKEN" in content


def test_version_sync_covers_every_file_the_verifier_gates() -> None:
    """version-sync.py must write every version location verify-versions.py checks.

    These two scripts drifting apart is the root cause of the stuck release: the
    verifier gated files nothing propagated a version to.
    """
    sync = (ROOT / "scripts" / "version-sync.py").read_text(encoding="utf-8")
    verify = (ROOT / "scripts" / "verify-versions.py").read_text(encoding="utf-8")

    gated = [
        "pyproject.toml",
        "plugins/openclaw/package.json",
        "sdk/typescript/package.json",
        "plugins/headroom-agent-hooks/.claude-plugin/plugin.json",
        "plugins/headroom-agent-hooks/.github/plugin/plugin.json",
        "marketplace.json",
    ]
    for path in gated:
        assert path in verify, f"{path} unexpectedly no longer gated by verify-versions.py"

    # version-sync builds paths piecewise, so match on the distinctive components.
    for fragment in [
        "openclaw",
        "typescript",
        "headroom-agent-hooks",
        "marketplace.json",
        "server.json",
    ]:
        assert fragment in sync, f"version-sync.py no longer propagates a version to {fragment}"


def test_metadata_sync_does_not_persist_credentials_for_branch_supplied_code() -> None:
    """The release credential must not be readable by the synced branch's code.

    ``release-metadata-sync`` triggers on a push to the ``release-please--branches--**``
    glob, which is not a protected namespace, and then runs
    ``scripts/version-sync.py`` *from the checked-out branch*. With
    ``actions/checkout``'s default ``persist-credentials: true`` the token is
    written to ``.git/config`` before that script runs, so anyone able to push a
    matching branch could read it. The credential reaches PyPI, npm and GHCR via
    the ``release: published`` publishes, so this is not a theoretical leak.
    """
    workflow = yaml.safe_load(
        (ROOT / ".github/workflows/release-metadata-sync.yml").read_text(encoding="utf-8")
    )
    steps = workflow["jobs"]["sync"]["steps"]

    checkouts = [s for s in steps if str(s.get("uses", "")).startswith("actions/checkout")]
    assert checkouts, "expected a checkout step"
    for step in checkouts:
        assert step.get("with", {}).get("persist-credentials") is False, step
        # A token passed to checkout is exactly what persist-credentials would
        # write to disk; the push step supplies it via env instead.
        assert "token" not in step.get("with", {}), step


@pytest.mark.parametrize(
    "workflow_path,job",
    [
        (".github/workflows/release-please.yml", "release-please"),
        (".github/workflows/release-metadata-sync.yml", "sync"),
    ],
)
def test_release_workflows_prefer_scoped_app_token(workflow_path: str, job: str) -> None:
    """A repo-scoped, short-lived app token must be preferred over the PAT.

    The PAT carries a maintainer's entire account and bypasses branch and tag
    protection (#2955). The app-token step is gated on ``vars.RELEASE_APP_ID``
    and marked ``continue-on-error`` so an unconfigured app falls back to the
    existing chain rather than blocking a release.
    """
    workflow = yaml.safe_load((ROOT / workflow_path).read_text(encoding="utf-8"))
    steps = workflow["jobs"][job]["steps"]

    minters = [
        s for s in steps if str(s.get("uses", "")).startswith("actions/create-github-app-token")
    ]
    assert len(minters) == 1, steps
    minter = minters[0]
    assert minter["id"] == "app-token"
    assert minter["continue-on-error"] is True
    assert "vars.RELEASE_APP_ID" in str(minter["if"])

    # Whatever consumes the credential must try the app token first.
    consumers = [
        value
        for step in steps
        for value in list(step.get("with", {}).values()) + list(step.get("env", {}).values())
        if "RELEASE_PLEASE_TOKEN" in str(value)
    ]
    assert consumers, "expected a step consuming the release credential"
    for value in consumers:
        assert str(value).index("steps.app-token.outputs.token") < str(value).index(
            "secrets.RELEASE_PLEASE_TOKEN"
        ), value
