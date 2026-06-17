from __future__ import annotations

from pathlib import Path

from headroom.providers.codex.install import build_provider_section, codex_uses_chatgpt_auth


def test_codex_provider_section_omits_requires_openai_auth_by_default() -> None:
    """#406: the flag must default off (API-key users), and only on for OAuth.

    Setting requires_openai_auth on a custom [model_providers.headroom] block
    forces codex to demand OpenAI OAuth login for every headroom-routed request,
    which breaks API-key users; so callers opt in explicitly for ChatGPT users.
    """
    section = build_provider_section(port=8787, name="OpenAI via Headroom proxy")

    assert 'name = "OpenAI via Headroom proxy"' in section
    assert 'base_url = "http://127.0.0.1:8787/v1"' in section
    assert "requires_openai_auth" not in section, (
        f"requires_openai_auth must be absent by default; got:\n{section}"
    )
    assert "supports_websockets = true" in section
    assert 'env_key = "OPENAI_API_KEY"' not in section


def test_codex_provider_section_emits_requires_openai_auth_when_flagged() -> None:
    section = build_provider_section(
        port=8787, name="OpenAI via Headroom proxy", requires_openai_auth=True
    )

    assert "requires_openai_auth = true" in section


def test_codex_uses_chatgpt_auth_true_for_chatgpt_mode(tmp_path: Path) -> None:
    auth = tmp_path / "auth.json"
    auth.write_text('{"auth_mode": "chatgpt"}', encoding="utf-8")

    assert codex_uses_chatgpt_auth(auth) is True


def test_codex_uses_chatgpt_auth_true_for_account_id_without_mode(tmp_path: Path) -> None:
    auth = tmp_path / "auth.json"
    auth.write_text('{"tokens": {"account_id": "acct_1"}}', encoding="utf-8")

    assert codex_uses_chatgpt_auth(auth) is True


def test_codex_uses_chatgpt_auth_false_for_api_key(tmp_path: Path) -> None:
    auth = tmp_path / "auth.json"
    auth.write_text('{"auth_mode": "apikey", "OPENAI_API_KEY": "sk-x"}', encoding="utf-8")

    assert codex_uses_chatgpt_auth(auth) is False


def test_codex_uses_chatgpt_auth_false_for_missing_or_malformed(tmp_path: Path) -> None:
    assert codex_uses_chatgpt_auth(tmp_path / "absent.json") is False
    bad = tmp_path / "auth.json"
    bad.write_text("not json", encoding="utf-8")
    assert codex_uses_chatgpt_auth(bad) is False


def test_codex_uses_chatgpt_auth_false_for_non_dict_json(tmp_path: Path) -> None:
    auth = tmp_path / "auth.json"
    auth.write_text("[]", encoding="utf-8")

    assert codex_uses_chatgpt_auth(auth) is False


def test_codex_uses_chatgpt_auth_false_for_empty_object(tmp_path: Path) -> None:
    auth = tmp_path / "auth.json"
    auth.write_text("{}", encoding="utf-8")

    assert codex_uses_chatgpt_auth(auth) is False


def test_codex_provider_section_supports_custom_markers() -> None:
    section = build_provider_section(
        port=9100,
        name="Headroom init proxy",
        marker_start="# --- start ---",
        marker_end="# --- end ---",
    )

    assert section.startswith("# --- start ---\n")
    assert section.endswith("# --- end ---\n")
    assert 'base_url = "http://127.0.0.1:9100/v1"' in section
    assert 'env_key = "OPENAI_API_KEY"' not in section
