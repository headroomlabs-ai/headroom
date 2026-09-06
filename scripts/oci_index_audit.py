"""Validate the runnable platform cardinality of a published OCI index."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

EXPECTED_PLATFORMS = ("linux/amd64", "linux/arm64")

# BuildKit stores provenance and SBOM manifests in the same index as the
# runnable images and marks them with this annotation. Their platform is
# conventionally `unknown/unknown`, but the annotation is what the descriptor
# is actually stamped with, so an attestation carrying a real platform still
# has to stay out of the runnable count.
ATTESTATION_REFERENCE_TYPE_ANNOTATION = "vnd.docker.reference.type"
ATTESTATION_REFERENCE_TYPE = "attestation-manifest"


class OCIIndexAuditError(ValueError):
    """Raised when an OCI index does not have one runnable descriptor per platform."""


@dataclass(frozen=True)
class OCIIndexAuditResult:
    """Cardinality observed in a published OCI index."""

    counts: dict[str, int]
    ignored_attestations: int
    ignored_unknown_platforms: int


def _is_annotated_attestation(descriptor: Any) -> bool:
    if not isinstance(descriptor, dict):
        return False
    annotations = descriptor.get("annotations")
    if not isinstance(annotations, dict):
        return False
    reference_type = annotations.get(ATTESTATION_REFERENCE_TYPE_ANNOTATION)
    return isinstance(reference_type, str) and reference_type == ATTESTATION_REFERENCE_TYPE


def _platform_name(descriptor: Any) -> str | None:
    platform = descriptor.get("platform") if isinstance(descriptor, dict) else None
    if not isinstance(platform, dict):
        return None
    operating_system = platform.get("os")
    architecture = platform.get("architecture")
    if not isinstance(operating_system, str) or not isinstance(architecture, str):
        return None
    return f"{operating_system}/{architecture}"


def _descriptor_label(descriptor: Any) -> str:
    """Digest plus variant, so a failure names which manifests collided."""
    digest = descriptor.get("digest") if isinstance(descriptor, dict) else None
    platform = descriptor.get("platform") if isinstance(descriptor, dict) else None
    variant = platform.get("variant") if isinstance(platform, dict) else None
    label = str(digest) if isinstance(digest, str) else "<no digest>"
    return f"{label} ({variant})" if isinstance(variant, str) and variant else label


def _runnable_descriptors_by_platform(
    index: dict[str, Any], expected_platforms: tuple[str, ...]
) -> dict[str, list[Any]]:
    manifests = index.get("manifests")
    if not isinstance(manifests, list):
        raise OCIIndexAuditError("OCI index must contain a manifests array")

    grouped: dict[str, list[Any]] = defaultdict(list)
    for descriptor in manifests:
        if _is_annotated_attestation(descriptor):
            continue
        platform = _platform_name(descriptor)
        if platform in expected_platforms:
            grouped[platform].append(descriptor)
    return {platform: grouped.get(platform, []) for platform in expected_platforms}


def count_runnable_linux_descriptors(
    index: dict[str, Any], expected_platforms: tuple[str, ...] = EXPECTED_PLATFORMS
) -> dict[str, int]:
    """Count known Linux platform descriptors, excluding attestations and other platforms."""
    grouped = _runnable_descriptors_by_platform(index, expected_platforms)
    return {platform: len(descriptors) for platform, descriptors in grouped.items()}


def audit_oci_index(
    index: dict[str, Any], expected_platforms: tuple[str, ...] = EXPECTED_PLATFORMS
) -> OCIIndexAuditResult:
    """Require exactly one runnable Linux descriptor for every expected platform."""
    grouped = _runnable_descriptors_by_platform(index, expected_platforms)
    counts = {platform: len(descriptors) for platform, descriptors in grouped.items()}
    manifests = index["manifests"]
    ignored_attestations = sum(_is_annotated_attestation(d) for d in manifests)
    # An unannotated `unknown/unknown` descriptor is excluded from the runnable
    # count either way, but it is not proof of an attestation, so it is counted
    # separately rather than inflating the attestation total.
    ignored_unknown_platforms = sum(
        not _is_annotated_attestation(d) and _platform_name(d) == "unknown/unknown"
        for d in manifests
    )

    invalid = {platform: count for platform, count in counts.items() if count != 1}
    if invalid:
        details = ", ".join(f"{platform}={count}" for platform, count in counts.items())
        collisions = "; ".join(
            f"{platform}: " + ", ".join(_descriptor_label(d) for d in grouped[platform])
            for platform in invalid
            if grouped[platform]
        )
        message = f"expected one runnable descriptor per platform, observed {details}"
        if collisions:
            message += f" [{collisions}]"
        raise OCIIndexAuditError(message)
    return OCIIndexAuditResult(counts, ignored_attestations, ignored_unknown_platforms)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tag",
        default="published tag",
        help="Tag being audited, used only to label the audit output",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Read a raw OCI index on stdin and exit non-zero when it is not one-per-platform."""
    args = _parse_args(argv)
    try:
        index = json.load(sys.stdin)
        if not isinstance(index, dict):
            raise OCIIndexAuditError("OCI index must be a JSON object")
        result = audit_oci_index(index)
    except (json.JSONDecodeError, OCIIndexAuditError) as error:
        print(f"OCI index audit failed for {args.tag}: {error}", file=sys.stderr)
        print(
            f"The tag is already published. Delete the {args.tag} package version in the "
            "registry, then re-run .github/workflows/docker.yml (the release calls it as its "
            "publish-docker job) so the manifest is rebuilt with one image per architecture "
            "and audited again before signing. Do not promote or sign this index.",
            file=sys.stderr,
        )
        return 1

    counts = ", ".join(f"{platform}={count}" for platform, count in result.counts.items())
    print(
        f"OCI index audit passed for {args.tag}: {counts}; "
        f"ignored attestation manifests={result.ignored_attestations}; "
        f"ignored unknown-platform descriptors={result.ignored_unknown_platforms}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
