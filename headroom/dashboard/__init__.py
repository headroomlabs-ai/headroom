"""Headroom Dashboard - Real-time proxy monitoring UI."""

from pathlib import Path

DASHBOARD_DIR = Path(__file__).parent
TEMPLATES_DIR = DASHBOARD_DIR / "templates"


def get_dashboard_html() -> str:
    """Load the dashboard HTML template."""
    template_path = TEMPLATES_DIR / "dashboard.html"
    return template_path.read_text(encoding="utf-8")
