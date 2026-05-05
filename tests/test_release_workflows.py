"""Workflow regression tests for release publishing behavior."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def test_docker_workflow_normalizes_repository_name_for_signing() -> None:
    content = (ROOT / ".github" / "workflows" / "docker.yml").read_text(encoding="utf-8")

    assert "id: image-name" in content
    assert "tr '[:upper:]' '[:lower:]'" in content
    assert "steps.image-name.outputs.image_name" in content


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


def test_create_release_runs_after_successful_build_even_if_other_publishes_fail() -> None:
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


def test_macos_native_wrapper_dependency_install_retries_pypi_downloads() -> None:
    content = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")

    assert "python -m pip install --retries 10 --timeout 60 pytest" in content


def test_ci_commitlint_skips_default_github_merge_commits() -> None:
    content = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")

    assert "github.event_name != 'push'" in content
    assert "!startsWith(github.event.head_commit.message, 'Merge pull request ')" in content


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
        # `cargo tree -i <pkg>` exits 101 with "did not match any
        # packages" when the package is NOT in the tree — the GREEN
        # case. Exit 0 with a tree of consumers means it IS pulled.
        not_in_tree = result.returncode != 0 and "did not match any packages" in result.stderr
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
        not_in_tree = result.returncode != 0 and "did not match any packages" in result.stderr
        assert not_in_tree, (
            f"native-tls is back in {crate}'s build tree — likely some "
            f"crate's `default-features = true` re-enabled native-tls "
            f"transitively:\n{result.stdout}"
        )


def test_fastembed_uses_rustls_features() -> None:
    """The mechanism that keeps openssl-sys out of the build is
    fastembed's explicit rustls feature selection in headroom-core.
    fastembed's default features include `hf-hub-native-tls` and
    `ort-download-binaries-native-tls` — both pull openssl-sys.
    Disabling defaults and enabling the rustls equivalents removes
    the OpenSSL surface entirely.
    """
    cargo = (ROOT / "crates" / "headroom-core" / "Cargo.toml").read_text(encoding="utf-8")

    assert "default-features = false" in cargo
    assert '"hf-hub-rustls-tls"' in cargo
    assert '"ort-download-binaries-rustls-tls"' in cargo
    # `image-models` is in default; we re-enable it explicitly so we
    # don't lose the image-embedding capability when defaults are off.
    assert '"image-models"' in cargo


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


def test_build_wheels_matrix_excludes_intel_macos() -> None:
    """`ort-sys 2.0.0-rc.12` (transitive via the ML compression backend)
    has no prebuilt ONNX Runtime binaries for `x86_64-apple-darwin`.
    Building ORT from source would add CMake + ~5 minutes per build.
    Apple Silicon macOS is fully covered; Intel-mac users install from
    the platform-independent sdist this matrix also produces.

    We assert against the actual matrix entry shape (`target: <triple>`
    on a non-comment line) so explanatory comments mentioning the
    excluded triple don't false-positive.
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

    # Intel macOS must NOT be a matrix entry — re-add only after switching
    # off ort-sys (e.g., to ort-tract) or adding a CMake-from-source step.
    assert "x86_64-apple-darwin" not in matrix_targets, (
        f"x86_64-apple-darwin must not be a wheel-matrix target; got {matrix_targets}"
    )

    # The runner OS itself shouldn't appear as a configured `os:` either.
    matrix_os: list[str] = []
    for raw in body.splitlines():
        stripped = raw.lstrip()
        if stripped.startswith("#"):
            continue
        if stripped.startswith("os:"):
            matrix_os.append(stripped.split(":", 1)[1].strip())
    assert "macos-15-intel" not in matrix_os


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


def test_glibc_compat_shim_present_in_headroom_py() -> None:
    """STRUCTURAL INVARIANT: the headroom-py crate ships a glibc-2.38
    compatibility shim that defines weak `__isoc23_*` aliases.

    Issue #355 (https://github.com/chopratejas/headroom/issues/355) —
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
