from __future__ import annotations

from headroom.providers.cortex_code import build_install_env, proxy_base_url, render_setup_lines
from headroom.providers.cortex_code.runtime import build_launch_env, default_api_url


def test_cortex_code_proxy_base_url_is_openai_compatible() -> None:
    assert proxy_base_url(8787) == "http://127.0.0.1:8787/v1"


def test_cortex_code_proxy_base_url_uses_given_port() -> None:
    assert proxy_base_url(9999) == "http://127.0.0.1:9999/v1"


def test_cortex_code_build_install_env_sets_openai_base_url() -> None:
    env = build_install_env(port=8787, backend="ignored")
    assert env == {"OPENAI_BASE_URL": "http://127.0.0.1:8787/v1"}


def test_cortex_code_build_launch_env_does_not_mutate_input() -> None:
    source = {"EXISTING": "val"}
    env, lines = build_launch_env(port=9999, environ=source)
    assert source == {"EXISTING": "val"}
    assert env["OPENAI_BASE_URL"] == "http://127.0.0.1:9999/v1"
    assert lines == ["OPENAI_BASE_URL=http://127.0.0.1:9999/v1"]


def test_cortex_code_build_launch_env_applies_project_prefix() -> None:
    env, lines = build_launch_env(port=9999, environ={}, project="myrepo")
    assert env["OPENAI_BASE_URL"] == "http://127.0.0.1:9999/p/myrepo/v1"
    assert lines == ["OPENAI_BASE_URL=http://127.0.0.1:9999/p/myrepo/v1"]


def test_cortex_code_build_launch_env_ignores_blank_project() -> None:
    env, lines = build_launch_env(port=9999, environ={}, project="   ")
    assert env["OPENAI_BASE_URL"] == "http://127.0.0.1:9999/v1"
    assert lines == ["OPENAI_BASE_URL=http://127.0.0.1:9999/v1"]


def test_cortex_code_render_setup_lines_contains_proxy_url() -> None:
    lines = render_setup_lines(8787)
    joined = "\n".join(lines)
    assert "http://127.0.0.1:8787/v1" in joined
    assert "Cortex Code" in joined


def test_cortex_code_render_setup_lines_project_attribution() -> None:
    lines = render_setup_lines(8787, project="my-sf-project")
    joined = "\n".join(lines)
    assert "my-sf-project" in joined
    plain = "\n".join(render_setup_lines(8787))
    assert "attributed" not in plain


def test_cortex_code_default_api_url_reads_snowflake_host_env() -> None:
    url = default_api_url({"SNOWFLAKE_HOST": "myaccount.snowflakecomputing.com"})
    assert url == "https://myaccount.snowflakecomputing.com"


def test_cortex_code_default_api_url_constructs_url_from_account_name() -> None:
    url = default_api_url({"SNOWFLAKE_ACCOUNT": "myaccount"})
    assert url == "https://myaccount.snowflakecomputing.com"


def test_cortex_code_default_api_url_host_takes_priority_over_account() -> None:
    url = default_api_url(
        {
            "SNOWFLAKE_HOST": "host.snowflakecomputing.com",
            "SNOWFLAKE_ACCOUNT": "account",
        }
    )
    assert url == "https://host.snowflakecomputing.com"


def test_cortex_code_default_api_url_falls_back_when_no_env() -> None:
    url = default_api_url({})
    assert url == "https://app.snowflake.com"


def test_cortex_code_default_api_url_preserves_https_prefix() -> None:
    url = default_api_url({"SNOWFLAKE_HOST": "https://already.snowflakecomputing.com"})
    assert url == "https://already.snowflakecomputing.com"


def test_cortex_code_install_registry_includes_cortex_code() -> None:
    from headroom.providers.install_registry import build_install_target_envs

    result = build_install_target_envs(port=1234, backend="ignored", targets=["cortex-code"])
    assert result["cortex-code"]["OPENAI_BASE_URL"] == "http://127.0.0.1:1234/v1"


def test_cortex_code_install_registry_unknown_target_skipped() -> None:
    from headroom.providers.install_registry import build_install_target_envs

    result = build_install_target_envs(
        port=1234, backend="ignored", targets=["cortex-code", "unknown-tool"]
    )
    assert "unknown-tool" not in result
    assert "cortex-code" in result
