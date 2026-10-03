from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover - exercised only on Python 3.10
    import tomli as tomllib  # type: ignore[no-redef]

from headroom.providers.codex.install import (
    build_codex_auth_config,
    build_provider_section,
    cleanup_codex_auth_helper,
    codex_auth_helper_is_referenced,
    codex_uses_api_key_auth,
    codex_uses_chatgpt_auth,
)


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


def test_codex_uses_api_key_auth_detects_file_backed_key(tmp_path: Path) -> None:
    auth = tmp_path / "auth.json"
    auth.write_text('{"OPENAI_API_KEY": "sk-x"}', encoding="utf-8")

    assert codex_uses_api_key_auth(auth) is True


def test_codex_uses_api_key_auth_rejects_missing_or_blank_key(tmp_path: Path) -> None:
    auth = tmp_path / "auth.json"
    for document in ("{}", '{"OPENAI_API_KEY": "  "}', "not json"):
        auth.write_text(document, encoding="utf-8")
        assert codex_uses_api_key_auth(auth) is False


def test_build_codex_auth_config_generates_helper_without_copying_key(tmp_path: Path) -> None:
    auth = tmp_path / "auth.json"
    auth.write_text('{"OPENAI_API_KEY": "sk-test-only"}', encoding="utf-8")

    config = build_codex_auth_config(auth)
    helper = tmp_path / ".headroom-codex-auth.py"

    assert "auth = { command =" in config
    assert tomllib.loads(config)["auth"]["args"] == [str(helper.resolve())]
    assert "sk-test-only" not in config
    assert helper.read_text(encoding="utf-8").count("OPENAI_API_KEY") == 1
    if sys.platform != "win32":
        assert helper.stat().st_mode & 0o777 == 0o600
    result = subprocess.run(
        [sys.executable, str(helper)], capture_output=True, text=True, check=True
    )
    assert result.stdout == "sk-test-only"


def test_codex_auth_helper_reference_parses_toml_escaped_windows_path() -> None:
    helper = r"C:\Users\example\.codex\.headroom-codex-auth.py"
    config = (
        "[model_providers.headroom]\n"
        f"auth = {{ command = \"python\", args = [{json.dumps(helper)}] }}\n"
    )

    assert codex_auth_helper_is_referenced(config, helper) is True


def test_build_codex_auth_config_does_not_overwrite_existing_helper(
    tmp_path: Path,
) -> None:
    auth = tmp_path / "auth.json"
    auth.write_text('{"OPENAI_API_KEY": "sk-test-only"}', encoding="utf-8")
    helper = tmp_path / ".headroom-codex-auth.py"
    helper.write_text("user content", encoding="utf-8")

    assert build_codex_auth_config(auth) == ""
    assert helper.read_text(encoding="utf-8") == "user content"


def test_build_codex_auth_config_rejects_helper_symlink(tmp_path: Path) -> None:
    auth = tmp_path / "auth.json"
    auth.write_text('{"OPENAI_API_KEY": "sk-test-only"}', encoding="utf-8")
    target = tmp_path / "user-file.txt"
    target.write_text("user content", encoding="utf-8")
    helper = tmp_path / ".headroom-codex-auth.py"
    try:
        helper.symlink_to(target)
    except OSError:
        pytest.skip("symlinks are unavailable")

    assert build_codex_auth_config(auth) == ""
    assert target.read_text(encoding="utf-8") == "user content"
    assert helper.is_symlink()


def test_build_codex_auth_config_handles_invalid_existing_helper(tmp_path: Path) -> None:
    auth = tmp_path / "auth.json"
    auth.write_text('{"OPENAI_API_KEY": "sk-test-only"}', encoding="utf-8")
    helper = tmp_path / ".headroom-codex-auth.py"
    helper.write_bytes(b"\xff\xfe")

    assert build_codex_auth_config(auth) == ""
    assert helper.read_bytes() == b"\xff\xfe"


def test_cleanup_codex_auth_helper_only_removes_headroom_file(tmp_path: Path) -> None:
    auth = tmp_path / "auth.json"
    auth.write_text('{"OPENAI_API_KEY": "sk-test-only"}', encoding="utf-8")
    build_codex_auth_config(auth)
    helper = tmp_path / ".headroom-codex-auth.py"

    cleanup_codex_auth_helper(auth)

    assert not helper.exists()


def test_build_codex_auth_config_skips_chatgpt_auth(tmp_path: Path) -> None:
    auth = tmp_path / "auth.json"
    auth.write_text('{"auth_mode": "chatgpt"}', encoding="utf-8")

    assert build_codex_auth_config(auth) == ""


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


# ---------------------------------------------------------------------------
# ChatGPT-auth detection from the id_token claims (#3206)
#
# Newer Codex releases can write an auth.json with neither `auth_mode` nor a
# top-level `tokens.account_id`; the account identity lives only in the
# id_token claims. Those configs read as API-key mode, so requires_openai_auth
# is omitted, Codex attaches no Authorization header, and every request 401s
# with "Missing bearer" -- silently, with doctor reporting green.
# ---------------------------------------------------------------------------


def _unsigned_jwt(claims: dict[str, object]) -> str:
    import base64
    import json as _json

    def seg(raw: bytes) -> str:
        return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")

    header = seg(b'{"alg":"none"}')
    payload = seg(_json.dumps(claims).encode("utf-8"))
    return ".".join((header, payload, "sig"))


_CHATGPT_CLAIMS: dict[str, object] = {
    "https://api.openai.com/auth": {
        "chatgpt_account_id": "1a155430-5551-47f4-9c7b-aeab7983f24a",
        "chatgpt_plan_type": "pro",
    }
}


def _write_auth(tmp_path, document: dict[str, object]):  # noqa: ANN001, ANN202
    import json as _json

    path = tmp_path / "auth.json"
    path.write_text(_json.dumps(document), encoding="utf-8")
    return path


def test_chatgpt_auth_detected_from_id_token_claims_alone(tmp_path) -> None:
    """The #3206 shape: no auth_mode, no tokens.account_id, only the JWT."""
    path = _write_auth(tmp_path, {"tokens": {"id_token": _unsigned_jwt(_CHATGPT_CLAIMS)}})

    assert codex_uses_chatgpt_auth(path) is True


def test_explicit_api_key_mode_still_wins_over_a_chatgpt_id_token(tmp_path) -> None:
    """Guards the #406 regression: API-key users must not get forced OAuth."""
    path = _write_auth(
        tmp_path,
        {"auth_mode": "apikey", "tokens": {"id_token": _unsigned_jwt(_CHATGPT_CLAIMS)}},
    )

    assert codex_uses_chatgpt_auth(path) is False


def test_api_key_config_without_tokens_is_not_chatgpt(tmp_path) -> None:
    path = _write_auth(tmp_path, {"OPENAI_API_KEY": "sk-test"})

    assert codex_uses_chatgpt_auth(path) is False


def test_id_token_without_the_chatgpt_claim_is_not_chatgpt(tmp_path) -> None:
    path = _write_auth(tmp_path, {"tokens": {"id_token": _unsigned_jwt({"sub": "user"})}})

    assert codex_uses_chatgpt_auth(path) is False


def test_malformed_id_token_is_not_chatgpt(tmp_path) -> None:
    for bogus in ("not-a-jwt", "a.b", "a.!!!not-base64!!!.c", ""):
        path = _write_auth(tmp_path, {"tokens": {"id_token": bogus}})
        assert codex_uses_chatgpt_auth(path) is False, bogus


def test_blank_chatgpt_account_id_is_not_chatgpt(tmp_path) -> None:
    claims = {"https://api.openai.com/auth": {"chatgpt_account_id": "   "}}
    path = _write_auth(tmp_path, {"tokens": {"id_token": _unsigned_jwt(claims)}})

    assert codex_uses_chatgpt_auth(path) is False


def test_legacy_account_id_still_detected(tmp_path) -> None:
    path = _write_auth(tmp_path, {"tokens": {"account_id": "acct-123"}})

    assert codex_uses_chatgpt_auth(path) is True


def test_provider_block_emits_requires_openai_auth_for_the_new_shape(tmp_path) -> None:
    """End of the chain: the JWT-only shape must produce the key Codex needs."""
    path = _write_auth(tmp_path, {"tokens": {"id_token": _unsigned_jwt(_CHATGPT_CLAIMS)}})

    block = build_provider_section(
        port=8787,
        name="Headroom",
        include_markers=False,
        requires_openai_auth=codex_uses_chatgpt_auth(path),
    )

    assert "requires_openai_auth = true" in block
