#!/usr/bin/env python3
"""Verify that documented environment variables exist in implementation source.

The check is deliberately text-based and dependency-free so it can run before
either the Python package or the documentation application is installed.  It
guards against documentation for a misspelled or removed setting silently
surviving after the implementation changes.
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

# Match uppercase snake-case names without assuming who owns them.  Wildcard
# families may be written as ``VENDOR_*`` or ``VENDOR_<FAMILY>``; both are
# normalized to ``VENDOR_*`` before comparison with concrete source names.
# Lowercase letters are part of the boundary so fragments such as ``DS_S`` in
# ``.DS_Store`` are not mistaken for environment variables.
ENV_VAR_PATTERN = re.compile(
    r"(?<![A-Za-z0-9_])"
    r"(?P<name>"
    r"(?:[A-Z][A-Z0-9]*(?:_[A-Z0-9]+)*_(?:\*|<[A-Z][A-Z0-9_]*>))"
    r"|(?:[A-Z][A-Z0-9]*(?:_[A-Z0-9]+)+)"
    r")"
    r"(?![A-Za-z0-9_])"
)

# Single-word names are too ambiguous to extract from prose, but shell syntax is
# unambiguous.  This covers POSIX expansion (``$HOME`` / ``${HOME}``),
# PowerShell (``$env:PATH``), and Windows expansion (``%USERPROFILE%``).
CONTEXTUAL_SINGLE_ENV_VAR_PATTERN = re.compile(
    r"(?:\$env:|\$\{?|%)"
    r"(?P<name>[A-Z][A-Z0-9]*)"
    r"(?:\}|%|(?![A-Za-z0-9_]))"
)

# Source can read a single-word variable through language APIs without shell
# expansion.  Keep the accessors explicit so arbitrary quoted constants do not
# become implementation evidence.
SOURCE_SINGLE_ENV_ACCESS_PATTERN = re.compile(
    r"(?:"
    r"(?:os\.)?environ\.get|(?:os\.)?getenv|env\.get|getEnv|env_path|"
    r"std::env::(?:var|var_os)"
    r")\(\s*[\"'](?P<name>[A-Z][A-Z0-9]*)[\"']"
    r"|(?:os\.)?environ\[\s*[\"'](?P<subscript_name>[A-Z][A-Z0-9]*)[\"']\s*\]"
)

# Uppercase snake-case is also used for enum members, error identifiers, and
# explanatory path aliases.  These reviewed names occur in the scanned docs but
# are not configuration.  Keep this list narrow: adding an entry opts that name
# out of the documentation/source consistency guarantee.
DOCUMENTED_NON_ENV_IDENTIFIERS = frozenset(
    {
        # Content types and pipeline stages.
        "BUILD_OUTPUT",
        "FIRST_LINE",
        "INPUT_CACHED",
        "INPUT_COMPRESSED",
        "INPUT_RECEIVED",
        "INPUT_REMEMBERED",
        "INPUT_ROUTED",
        "PLAIN_TEXT",
        "POST_SEND",
        "POST_START",
        "PRE_SEND",
        "PRE_START",
        "RESPONSE_RECEIVED",
        "SEARCH_RESULTS",
        # Errors and library constants shown in troubleshooting guidance.
        "CERTIFICATE_VERIFY_FAILED",
        "MALFORMED_FUNCTION_CALL",
        "UNSUPPORTED_METRIC_TYPE_MONOTONIC_CUMULATIVE_SUM",
        "VERIFY_X509_STRICT",
        # Symbolic bucket names used only in filesystem diagrams.
        "CONFIG_DIR",
        "WORKSPACE_DIR",
    }
)

# Some documented variables are read by libraries or tools that Headroom invokes
# rather than by a literal read site in this repository.  Listing them here keeps
# typo detection strict while making that external ownership explicit and
# reviewable.  The verifier itself is excluded from source scanning so this
# policy cannot accidentally count as an implementation reference.
EXTERNALLY_CONSUMED_VARIABLES = frozenset(
    {
        # Standard proxy and certificate variables read by HTTP clients.
        "ALL_PROXY",
        "CURL_CA_BUNDLE",
        "HF_ENDPOINT",
        "HTTPS_PROXY",
        "HTTP_PROXY",
        "NO_PROXY",
        "REQUESTS_CA_BUNDLE",
        "SSL_CERT_FILE",
        # Standard shell / operating-system environment.
        "HOME",
        "PATH",
        "PWD",
        "TMPDIR",
        "USERPROFILE",
        # AWS SDK / Bedrock credential and endpoint discovery.
        "AWS_ENDPOINT_URL_BEDROCK_RUNTIME",
        # LiteLLM provider configuration.
        "DEEPSEEK_API_KEY",
        "GOOGLE_CLOUD_LOCATION",
        "GOOGLE_CLOUD_PROJECT",
        "GROK_CODE_XAI_API_KEY",
        "VERTEXAI_LOCATION",
        "VERTEXAI_PROJECT",
        "XAI_API_KEY",
        # ONNX Runtime build and dynamic-link controls.
        "ORT_LIB_LOCATION",
        "ORT_PREFER_DYNAMIC_LINK",
        "ORT_STRATEGY",
        # OpenTelemetry SDK autoconfiguration.
        "OTEL_EXPORTER_OTLP_ENDPOINT",
        "OTEL_EXPORTER_OTLP_HEADERS",
        "OTEL_EXPORTER_OTLP_METRICS_TEMPORALITY_PREFERENCE",
        "OTEL_EXPORTER_OTLP_PROTOCOL",
    }
)

DOCUMENTATION_FILES = (Path("README.md"), Path("SECURITY.md"))
DOCUMENTATION_GLOB = "docs/content/docs/**/*.mdx"

# These are implementation surfaces, not tests or examples.  Docker and install
# sources are included because a few documented host-side variables are consumed
# before the Python or Rust process starts.
SOURCE_PATHS = (
    Path(".github"),
    Path("headroom"),
    Path("crates"),
    Path("sdk"),
    Path("plugins"),
    Path("deploy"),
    Path("docker"),
    Path("scripts"),
)
EXCLUDED_SOURCE_FILES = {Path("scripts/ci/verify_documented_env_vars.py")}
SOURCE_SUFFIXES = {
    ".cjs",
    ".js",
    ".json",
    ".mjs",
    ".ps1",
    ".py",
    ".rs",
    ".sh",
    ".toml",
    ".ts",
    ".tsx",
    ".yaml",
    ".yml",
}
EXCLUDED_SOURCE_PARTS = {
    "__pycache__",
    "examples",
    "fixtures",
    "node_modules",
    "target",
    "tests",
}


@dataclass(frozen=True)
class Occurrence:
    """One documented environment-variable reference."""

    path: Path
    line: int


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def _documentation_paths(root: Path) -> list[Path]:
    paths = [root / relative_path for relative_path in DOCUMENTATION_FILES]
    paths.extend(root.glob(DOCUMENTATION_GLOB))
    return sorted(path for path in paths if path.is_file())


def _source_files(root: Path) -> list[Path]:
    files: list[Path] = []
    for relative_path in SOURCE_PATHS:
        path = root / relative_path
        if path.is_file():
            candidates = [path]
        elif path.is_dir():
            candidates = path.rglob("*")
        else:
            continue

        for candidate in candidates:
            if not candidate.is_file() or candidate.suffix not in SOURCE_SUFFIXES:
                continue
            relative_path = candidate.relative_to(root)
            if relative_path in EXCLUDED_SOURCE_FILES:
                continue
            relative_parts = relative_path.parts
            if any(part in EXCLUDED_SOURCE_PARTS for part in relative_parts):
                continue
            files.append(candidate)
    return sorted(files)


def _normalized_name(match: re.Match[str]) -> str:
    """Return a concrete name or a normalized ``PREFIX_*`` family."""

    name = match.group("name")
    if "_<" in name:
        return f"{name.split('_<', maxsplit=1)[0]}_*"
    return name


def _is_dotted_member(line: str, start: int) -> bool:
    """Return whether a candidate is a code member such as ``Stage.PRE_SEND``."""

    return re.search(r"[A-Za-z_][A-Za-z0-9_]*\.$", line[:start]) is not None


def _documented_names_in_line(line: str) -> set[str]:
    """Extract environment-variable references from one documentation line."""

    names: set[str] = set()
    for match in ENV_VAR_PATTERN.finditer(line):
        name = _normalized_name(match)
        if name in DOCUMENTED_NON_ENV_IDENTIFIERS:
            continue
        # Dotted uppercase members are code constants, not process settings.
        # Explicit exclusions above still cover prose/table mentions of the same
        # constants where the owning type is not present.
        if _is_dotted_member(line, match.start()):
            continue
        names.add(name)
    names.update(match.group("name") for match in CONTEXTUAL_SINGLE_ENV_VAR_PATTERN.finditer(line))
    return names


def documented_variables(root: Path) -> dict[str, list[Occurrence]]:
    """Return documented variables and the locations that mention them."""

    variables: dict[str, list[Occurrence]] = defaultdict(list)
    for path in _documentation_paths(root):
        for line_number, line in enumerate(_read_text(path).splitlines(), start=1):
            for name in _documented_names_in_line(line):
                variables[name].append(Occurrence(path=path.relative_to(root), line=line_number))
    return dict(variables)


def source_variables(root: Path) -> set[str]:
    """Return environment-variable-shaped identifiers found in implementation source."""

    variables: set[str] = set()
    for path in _source_files(root):
        text = _read_text(path)
        variables.update(_normalized_name(match) for match in ENV_VAR_PATTERN.finditer(text))
        variables.update(
            match.group("name") for match in CONTEXTUAL_SINGLE_ENV_VAR_PATTERN.finditer(text)
        )
        variables.update(
            match.group("name") or match.group("subscript_name")
            for match in SOURCE_SINGLE_ENV_ACCESS_PATTERN.finditer(text)
        )
    return variables


def _is_resolved(name: str, available: set[str] | frozenset[str]) -> bool:
    """Return whether a concrete name or wildcard family is available."""

    if name.endswith("*"):
        prefix = name[:-1]
        return any(
            candidate.startswith(prefix) and not candidate.endswith("*") for candidate in available
        )
    return name in available


def missing_variables(root: Path) -> dict[str, list[Occurrence]]:
    """Return documented variables that have no implementation reference."""

    documented = documented_variables(root)
    implemented = source_variables(root)
    available = implemented | EXTERNALLY_CONSUMED_VARIABLES
    missing: dict[str, list[Occurrence]] = {}

    for name, occurrences in documented.items():
        if not _is_resolved(name, available):
            missing[name] = occurrences

    return missing


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
        help="Repository root (defaults to the checkout containing this script)",
    )
    args = parser.parse_args(argv)
    root = args.root.resolve()

    documented = documented_variables(root)
    missing = missing_variables(root)
    if missing:
        for name, occurrences in sorted(missing.items()):
            first = occurrences[0]
            print(
                f"::error file={first.path},line={first.line}::"
                f"Documented environment variable {name} has no implementation reference",
                file=sys.stderr,
            )
        print(
            f"Found {len(missing)} documented environment variable(s) with no source reference.",
            file=sys.stderr,
        )
        return 1

    implemented = source_variables(root)
    external_count = sum(
        not _is_resolved(name, implemented) and _is_resolved(name, EXTERNALLY_CONSUMED_VARIABLES)
        for name in documented
    )
    print(
        f"Verified {len(documented)} documented environment variables "
        f"({len(documented) - external_count} against repository source; "
        f"{external_count} against the approved external-consumer policy)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
