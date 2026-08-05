from __future__ import annotations

from headroom.proxy.project_name_policy import PROJECT_NAME_MAX_LENGTH, sanitize_project_name
from headroom.proxy.savings_tracker import sanitize_project_name as savings_sanitize_project_name


def test_project_name_policy_normalizes_and_caps() -> None:
    assert sanitize_project_name("  api-server  ") == "api-server"
    assert sanitize_project_name("a" * 300) == "a" * PROJECT_NAME_MAX_LENGTH
    assert sanitize_project_name("x\x00\x1by") == "xy"


def test_project_name_policy_decodes_percent_encoded_unicode() -> None:
    assert sanitize_project_name("%E9%A1%B9%E7%9B%AE") == "\u9879\u76ee"
    assert sanitize_project_name("my%20repo") == "my repo"


def test_project_name_policy_rejects_unusable_values() -> None:
    assert sanitize_project_name("") is None
    assert sanitize_project_name("   ") is None
    assert sanitize_project_name(None) is None
    assert sanitize_project_name(42) is None


def test_savings_tracker_reexports_project_name_policy() -> None:
    assert savings_sanitize_project_name is sanitize_project_name
