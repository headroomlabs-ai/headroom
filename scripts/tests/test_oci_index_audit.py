from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from oci_index_audit import (  # noqa: E402
    ATTESTATION_REFERENCE_TYPE,
    ATTESTATION_REFERENCE_TYPE_ANNOTATION,
    OCIIndexAuditError,
    audit_oci_index,
    count_runnable_linux_descriptors,
)


def descriptor(
    os_name: str | None,
    architecture: str | None,
    digest: str,
    *,
    variant: str | None = None,
    attestation: bool = False,
) -> dict[str, object]:
    entry: dict[str, object] = {
        "digest": digest if digest.startswith("sha256:") else f"sha256:{digest}",
        "mediaType": "application/vnd.oci.image.manifest.v1+json",
        "size": 123,
    }
    if os_name is not None and architecture is not None:
        platform: dict[str, object] = {"os": os_name, "architecture": architecture}
        if variant:
            platform["variant"] = variant
        entry["platform"] = platform
    if attestation:
        entry["annotations"] = {
            ATTESTATION_REFERENCE_TYPE_ANNOTATION: ATTESTATION_REFERENCE_TYPE,
            "vnd.docker.reference.digest": "sha256:0000",
        }
    return entry


def index(*descriptors: dict[str, object]) -> dict[str, object]:
    return {
        "schemaVersion": 2,
        "mediaType": "application/vnd.oci.image.index.v1+json",
        "manifests": list(descriptors),
    }


def valid_index() -> dict[str, object]:
    return index(
        descriptor("linux", "amd64", "amd64-image"),
        descriptor("linux", "arm64", "arm64-image"),
        descriptor("unknown", "unknown", "amd64-attestation", attestation=True),
        descriptor("unknown", "unknown", "arm64-attestation", attestation=True),
    )


def reported_corrupted_index() -> dict[str, object]:
    """The index shape issue #2673 reports for :0.33.0-code and :0.33-slim.

    Sixteen descriptors: four runnable per architecture, alternating distroless
    and alpine variants, plus eight `unknown/unknown` attestation manifests. The
    digests are the ones the reporter read off the published tag, so a change
    that stops recognising this exact shape fails here rather than against a
    fixture written to match the implementation.
    """
    amd64 = [
        "sha256:6dad521558ec7cf702dcfa8acee7582a5a20bd3ac2ba6a8593a184f67b9faea1",
        "sha256:dbb23bc7102956480a3a8af72d539a66d8770f2a710f827ae9b8c15f9fe9ed12",
        "sha256:8b3e9776ebd41488e2da081a0417764ac401819daae37135a3fc25a9f85c9e76",
        "sha256:a733d8f5d689f33f01128739a1cc6523195f22846d4e0c1b3f4422d99afc4814",
    ]
    arm64 = [
        "sha256:d64cb51d981b8ebcaf19a3aa6f1d732de6fb5867a4a6057304928df19a7ae860",
        "sha256:7dee3581199114ea682e46b53a3555a0e028c662d18f08545df1c8e575fe08af",
        "sha256:ed1398992b19eb744217ec016ce9b9a46e3ba9e99c399d1c5aa5a0440c9f5a9b",
        "sha256:17ff460602124b28e6d3ebc2ef7211a212d1e6ea8c2c8bfc50ca526e330e17d4",
    ]
    descriptors = [descriptor("linux", "amd64", digest) for digest in amd64]
    descriptors += [descriptor("linux", "arm64", digest) for digest in arm64]
    descriptors += [
        descriptor("unknown", "unknown", f"attestation-{position}", attestation=True)
        for position in range(8)
    ]
    return index(*descriptors)


def test_counts_one_runnable_linux_descriptor_per_architecture() -> None:
    result = audit_oci_index(valid_index())

    assert result.counts == {"linux/amd64": 1, "linux/arm64": 1}
    assert result.ignored_attestations == 2


def test_reported_corrupted_index_is_rejected_and_names_the_collisions() -> None:
    corrupted = reported_corrupted_index()

    assert len(corrupted["manifests"]) == 16
    assert count_runnable_linux_descriptors(corrupted) == {"linux/amd64": 4, "linux/arm64": 4}

    with pytest.raises(OCIIndexAuditError) as failure:
        audit_oci_index(corrupted)

    message = str(failure.value)
    assert "linux/amd64=4" in message
    assert "linux/arm64=4" in message
    # The distroless manifest amd64 resolves to first, which is the descriptor
    # the reporter's shell-form RUN actually failed on.
    assert "sha256:6dad521558ec7cf702dcfa8acee7582a5a20bd3ac2ba6a8593a184f67b9faea1" in message


def test_attestation_annotation_excludes_a_descriptor_carrying_a_real_platform() -> None:
    """The annotation is authoritative; `unknown/unknown` is only the convention."""
    observed = index(
        descriptor("linux", "amd64", "amd64-image"),
        descriptor("linux", "arm64", "arm64-image"),
        descriptor("linux", "amd64", "amd64-provenance", attestation=True),
    )

    result = audit_oci_index(observed)
    assert result.counts == {"linux/amd64": 1, "linux/arm64": 1}
    assert result.ignored_attestations == 1
    assert result.ignored_unknown_platforms == 0


def test_an_unannotated_unknown_descriptor_is_not_reported_as_an_attestation() -> None:
    """It stays out of the runnable count either way, but it is not proof of one."""
    observed = index(
        descriptor("linux", "amd64", "amd64-image"),
        descriptor("linux", "arm64", "arm64-image"),
        descriptor("unknown", "unknown", "annotated", attestation=True),
        descriptor("unknown", "unknown", "bare-unknown"),
    )

    result = audit_oci_index(observed)
    assert result.counts == {"linux/amd64": 1, "linux/arm64": 1}
    assert result.ignored_attestations == 1
    assert result.ignored_unknown_platforms == 1


def test_unknown_attestations_do_not_mask_duplicate_runnable_descriptors() -> None:
    corrupted = index(
        descriptor("linux", "amd64", "amd64-image-1"),
        descriptor("linux", "amd64", "amd64-image-2"),
        descriptor("linux", "arm64", "arm64-image"),
        descriptor("unknown", "unknown", "attestation-1", attestation=True),
        descriptor("unknown", "unknown", "attestation-2", attestation=True),
    )

    assert count_runnable_linux_descriptors(corrupted) == {"linux/amd64": 2, "linux/arm64": 1}
    with pytest.raises(OCIIndexAuditError, match="linux/amd64=2"):
        audit_oci_index(corrupted)


def test_missing_runnable_architecture_fails() -> None:
    with pytest.raises(OCIIndexAuditError, match="linux/arm64=0"):
        audit_oci_index(index(descriptor("linux", "amd64", "amd64-image")))


def test_non_linux_and_unknown_descriptors_are_not_runnable() -> None:
    observed = index(
        descriptor("linux", "amd64", "amd64-image"),
        descriptor("linux", "arm64", "arm64-image"),
        descriptor("windows", "amd64", "windows-image"),
        descriptor("linux", "386", "linux-386-image"),
        descriptor("linux", "arm", "linux-armv7-image", variant="v7"),
        descriptor("linux", "arm", "linux-armv6-image", variant="v6"),
        descriptor(None, None, "no-platform-key"),
        descriptor("unknown", "unknown", "attestation", attestation=True),
    )

    assert count_runnable_linux_descriptors(observed) == {"linux/amd64": 1, "linux/arm64": 1}
    assert audit_oci_index(observed).ignored_attestations == 1


def test_malformed_index_fails() -> None:
    with pytest.raises(OCIIndexAuditError, match="manifests array"):
        audit_oci_index({})

    with pytest.raises(OCIIndexAuditError, match="manifests array"):
        audit_oci_index({"manifests": {"linux/amd64": 1}})


def _run_cli(stdin_text: str, tag: str = "example:code") -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "oci_index_audit.py"), "--tag", tag],
        input=stdin_text,
        capture_output=True,
        text=True,
        check=False,
    )


def test_cli_reads_raw_index_and_reports_attestations() -> None:
    completed = _run_cli(json.dumps(valid_index()))

    assert completed.returncode == 0
    assert "example:code" in completed.stdout
    assert "linux/amd64=1" in completed.stdout
    assert "ignored attestation manifests=2" in completed.stdout
    assert "ignored unknown-platform descriptors=0" in completed.stdout


def test_cli_fails_on_the_reported_corrupted_index_and_says_what_to_do() -> None:
    completed = _run_cli(json.dumps(reported_corrupted_index()), tag="headroom:0.33.0-code")

    assert completed.returncode == 1
    assert "OCI index audit failed for headroom:0.33.0-code" in completed.stderr
    assert "linux/amd64=4" in completed.stderr
    assert "Delete the headroom:0.33.0-code package version" in completed.stderr
    assert ".github/workflows/docker.yml" in completed.stderr
    assert "publish-docker job" in completed.stderr
    assert "Do not promote or sign this index." in completed.stderr


@pytest.mark.parametrize(
    ("stdin_text", "expected"),
    [
        ("", "audit failed"),
        ("not json at all", "audit failed"),
        ("[]", "must be a JSON object"),
        ("7", "must be a JSON object"),
        (json.dumps({"schemaVersion": 2, "mediaType": "…image.manifest.v1+json"}), "manifests"),
        (json.dumps({"manifests": "linux/amd64"}), "manifests"),
    ],
)
def test_cli_rejects_unusable_stdin(stdin_text: str, expected: str) -> None:
    """A producer that exits zero with unusable output must still fail the release."""
    completed = _run_cli(stdin_text)

    assert completed.returncode == 1
    assert expected in completed.stderr
    assert completed.stdout == ""


def _workflow() -> dict[str, object]:
    return yaml.safe_load((ROOT / ".github" / "workflows" / "docker.yml").read_text("utf-8"))


# `Re-tag root image as :latest` re-creates :latest from a version tag that the
# manifest job already audited, and its job needs the manifest job, so it is
# skipped when the audit fails. Any other publisher must audit its own output.
# `test_the_latest_exemption_rests_on_properties_the_workflow_still_has` pins both
# of those properties, so the exemption cannot quietly become unsound.
AUDIT_EXEMPT_PUBLISHERS = frozenset({"Re-tag root image as :latest"})


def _manifest_job() -> dict[str, object]:
    for job in _workflow()["jobs"].values():
        steps = job.get("steps") or []
        if any(
            "imagetools create" in str(step.get("run", ""))
            and str(step.get("name", "")) not in AUDIT_EXEMPT_PUBLISHERS
            for step in steps
        ):
            return job
    raise AssertionError("no job creates and audits a multi-arch manifest")


def _manifest_job_steps() -> list[dict[str, object]]:
    return _manifest_job()["steps"]


def test_every_publisher_either_audits_or_is_named_exempt() -> None:
    """Every step running `imagetools create` audits its output or is named exempt.

    Scoped to that command, which is how this workflow publishes indexes today. A
    publisher introduced with `docker manifest create`, `crane`, or `skopeo` would
    not be seen here.
    """
    unaudited: list[str] = []
    for job_name, job in _workflow()["jobs"].items():
        for step in job.get("steps") or []:
            run = str(step.get("run", ""))
            if "imagetools create" not in run:
                continue
            name = str(step.get("name", ""))
            if name in AUDIT_EXEMPT_PUBLISHERS:
                continue
            if "oci_index_audit.py" not in run:
                unaudited.append(f"{job_name} / {name}")

    assert unaudited == []


def test_manifest_job_checks_out_the_repo_and_installs_python() -> None:
    """The audit runs a checked-in script, so this job needs the repo on disk."""
    steps = _manifest_job_steps()
    uses = [str(step.get("uses", "")) for step in steps]

    assert any(entry.startswith("actions/checkout@") for entry in uses)
    assert any(entry.startswith("actions/setup-python@") for entry in uses)


def test_manifest_step_audits_every_published_tag_before_signing() -> None:
    steps = _manifest_job_steps()
    manifest_step = next(step for step in steps if "imagetools create" in str(step.get("run", "")))
    run = str(manifest_step["run"])

    # Fail-closed: a producer failure inside the pipeline must not be masked.
    assert manifest_step.get("shell") == "bash"
    assert "set -o pipefail" in run

    create_at = run.index("docker buildx imagetools create")
    loop_at = run.index('for tag in "${published_tags[@]}"')
    audit_at = run.index("python scripts/oci_index_audit.py")
    sign_at = run.index("index_digest=")

    # The audit sits inside the published-tags loop, after the push, before the
    # digest resolution that cosign signs.
    assert create_at < loop_at < audit_at < sign_at

    loop_body = run[loop_at:sign_at]
    assert 'docker buildx imagetools inspect "${tag}" --raw' in loop_body
    assert '--tag "${tag}"' in loop_body

    # Every tag the job pushes is also the set it audits.
    assert 'published_tags+=("$tag")' in run
    assert 'tag_args+=("--tag" "$tag")' in run


def test_the_latest_exemption_rests_on_properties_the_workflow_still_has() -> None:
    """The `:latest` re-tag is exempt only because of two facts. Pin both.

    It is skipped when the audit fails, and it copies a tag the audited job
    published. Lose either and the exemption is unsound while every other
    assertion here still passes.
    """
    workflow = _workflow()
    audited_job_id = next(
        job_id
        for job_id, job in workflow["jobs"].items()
        if any(
            "oci_index_audit.py" in str(step.get("run", "")) for step in (job.get("steps") or [])
        )
    )

    exempt_job_id, exempt_job = next(
        (job_id, job)
        for job_id, job in workflow["jobs"].items()
        if any(
            str(step.get("name", "")) in AUDIT_EXEMPT_PUBLISHERS
            for step in (job.get("steps") or [])
        )
    )
    if exempt_job_id != audited_job_id:
        needs = exempt_job.get("needs")
        needs = [needs] if isinstance(needs, str) else list(needs or [])
        assert audited_job_id in needs

    exempt_step = next(
        step for step in exempt_job["steps"] if str(step.get("name", "")) in AUDIT_EXEMPT_PUBLISHERS
    )
    run = str(exempt_step["run"])
    # The source of the re-tag is a version tag the audited job published.
    assert '"${IMAGE}:${VERSION}"' in run
    assert "imagetools create" in run

    # When promotion shares the audited manifest job, step ordering is the
    # dependency: GitHub Actions stops before this step if the audit fails.
    if exempt_job_id == audited_job_id:
        steps = exempt_job["steps"]
        audit_at = next(
            index
            for index, step in enumerate(steps)
            if "oci_index_audit.py" in str(step.get("run", ""))
        )
        promote_at = steps.index(exempt_step)
        assert audit_at < promote_at


def test_every_manifest_matrix_variant_runs_the_audit() -> None:
    """One unconditional loop covers all eight bake variants; none opts out."""
    job = _manifest_job()
    variants = job["strategy"]["matrix"]["variant"]

    assert len(variants) == 8
    names = {str(variant.get("name", "")) for variant in variants}
    assert {"", "nonroot", "code", "code-nonroot", "slim", "slim-nonroot"} <= names

    manifest_step = next(
        step for step in job["steps"] if "imagetools create" in str(step.get("run", ""))
    )
    run = str(manifest_step["run"])
    # The audit runs for every leg: it is in the step body, not behind a
    # variant-keyed step condition and not inside a variant branch.
    assert "oci_index_audit.py" in run
    assert "variant" not in str(manifest_step.get("if", ""))
    assert "matrix.variant" not in run[run.index('for tag in "${published_tags[@]}"') :]
