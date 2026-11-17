"""Setup-text lines shared by the provider ``render_setup_lines`` helpers."""

from __future__ import annotations


def project_attribution_lines(project: str | None) -> list[str]:
    """Setup-text lines telling the user which project the proxy URL attributes savings to."""
    if not project:
        return []
    return [
        "",
        f"  Dashboard savings will be attributed to project '{project}'",
        "  (the directory this command was run from). Re-run from another",
        "  project directory to get that project's URL.",
    ]
