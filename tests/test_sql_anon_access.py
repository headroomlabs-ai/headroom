"""Static checks on the Supabase setup scripts and the all-plugins dev script.

There is no SQL harness in the repo, so these read the script text.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _statements(sql: str) -> list[str]:
    sql = re.sub(r"--[^\n]*", "", sql)  # comments may quote the statements to undo
    return [" ".join(s.split()) for s in sql.split(";") if s.strip()]


def test_raw_telemetry_table_is_not_readable_by_anon() -> None:
    sql = (ROOT / "sql" / "create_proxy_telemetry_v2.sql").read_text(encoding="utf-8")
    statements = _statements(sql)
    for stmt in statements:
        assert not re.match(r"CREATE POLICY .* FOR (SELECT|ALL)\b.* TO anon\b", stmt, re.I), stmt
        grant = re.match(r"GRANT (.*?) ON .* TO .*\banon\b", stmt, re.I)
        assert not (grant and re.search(r"\b(SELECT|ALL)\b", grant.group(1), re.I)), stmt
    assert any(re.match(r"GRANT INSERT ON proxy_telemetry_v2 TO anon", s) for s in statements)


def test_all_plugins_script_does_not_turn_off_licence_checks() -> None:
    text = (ROOT / "run-all-plugins.sh").read_text(encoding="utf-8")
    assert not re.search(r"^\s*export\s+HEADROOM_LICENSE_DEV", text, re.M)
