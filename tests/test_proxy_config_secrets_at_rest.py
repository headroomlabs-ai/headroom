"""ProxyConfig credentials stay out of manifests, the worker environment and TOIN."""

from __future__ import annotations

import dataclasses
import json
import os
import stat
from pathlib import Path
from unittest.mock import patch

import pytest

pytest.importorskip("fastapi")

from headroom.proxy.models import ProxyConfig  # noqa: E402
from headroom.telemetry.toin import TOINConfig, ToolIntelligenceNetwork  # noqa: E402
from headroom.testing import Headroom  # noqa: E402

SECRETS = {
    "memory_qdrant_api_key": "FAKE-QDRANT",
    "memory_neo4j_password": "FAKE-NEO4J",
    "license_key": "FAKE-LICENSE",
    "proxy_token": "FAKE-PROXYTOKEN",
    "openai_extra_headers": {"x-gateway-auth": "FAKE-GATEWAY"},
}
SECRET_VALUES = ("FAKE-QDRANT", "FAKE-NEO4J", "FAKE-LICENSE", "FAKE-PROXYTOKEN", "FAKE-GATEWAY")


def test_bench_manifest_does_not_serialise_proxy_secrets(tmp_path: Path) -> None:
    scenario = Headroom.WithBedrock(region="us-east-2", profile="bench").Build()
    scenario = dataclasses.replace(scenario, proxy_config=ProxyConfig(**SECRETS))

    text = scenario.write_manifest_fragment(tmp_path / "manifest.json").read_text(encoding="utf-8")

    for value in SECRET_VALUES:
        assert value not in text
    payload = json.loads(text)
    assert payload["proxy_config"]["license_key"] == "[REDACTED]"
    assert payload["proxy_config"]["openai_extra_headers"] == {"x-gateway-auth": "[REDACTED]"}
    # The live plan still carries the real values for the runner.
    plan = scenario.deployment_plan(port=18888)
    assert json.loads(plan.env["HEADROOM_PROXY_CONFIG_JSON"])["proxy_token"] == "FAKE-PROXYTOKEN"


def test_multi_worker_config_keeps_secrets_out_of_the_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from headroom.proxy import server

    # setenv first so monkeypatch restores both after run_server sets them.
    for name in ("HEADROOM_PROXY_CONFIG_JSON", "HEADROOM_PROXY_CONFIG_SECRETS_FILE"):
        monkeypatch.setenv(name, "")
        monkeypatch.delenv(name)
    seen: dict[str, object] = {}

    def fake_run_uvicorn(*_args: object) -> None:
        env_json = os.environ["HEADROOM_PROXY_CONFIG_JSON"]
        secrets_file = Path(os.environ["HEADROOM_PROXY_CONFIG_SECRETS_FILE"])
        seen["env_json"] = env_json
        seen["file_mode"] = stat.S_IMODE(secrets_file.stat().st_mode)
        seen["secrets_file"] = secrets_file
        # What a worker rebuilds from the environment and the file.
        seen["worker_config"] = server._proxy_config_from_env()

    with patch.object(server, "_run_uvicorn", side_effect=fake_run_uvicorn):
        server.run_server(ProxyConfig(**SECRETS), workers=2, print_banner=False)

    for value in SECRET_VALUES:
        assert value not in seen["env_json"]
    assert seen["file_mode"] == 0o600
    worker_config = seen["worker_config"]
    assert isinstance(worker_config, ProxyConfig)
    assert worker_config.proxy_token == "FAKE-PROXYTOKEN"
    assert worker_config.openai_extra_headers == {"x-gateway-auth": "FAKE-GATEWAY"}
    assert not Path(seen["secrets_file"]).exists()  # removed when the server stops


def test_credential_free_restart_after_a_credentialed_run_still_starts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from headroom.proxy import server

    for name in ("HEADROOM_PROXY_CONFIG_JSON", "HEADROOM_PROXY_CONFIG_SECRETS_FILE"):
        monkeypatch.setenv(name, "")
        monkeypatch.delenv(name)
    configs: list[ProxyConfig] = []

    def fake_run_uvicorn(*_args: object) -> None:
        configs.append(server._proxy_config_from_env())

    with patch.object(server, "_run_uvicorn", side_effect=fake_run_uvicorn):
        server.run_server(ProxyConfig(**SECRETS), workers=2, print_banner=False)
        # The worker hand-off variables are back to their pre-run state (unset).
        assert "HEADROOM_PROXY_CONFIG_SECRETS_FILE" not in os.environ
        assert "HEADROOM_PROXY_CONFIG_JSON" not in os.environ
        server.run_server(ProxyConfig(), workers=2, print_banner=False)

    assert configs[0].proxy_token == "FAKE-PROXYTOKEN"
    assert configs[1].proxy_token is None


@pytest.mark.skipif(os.name == "nt", reason="Unix sockets")
def test_failed_startup_removes_the_secrets_file_and_restores_the_environment(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import tempfile

    from headroom.proxy import server
    from headroom.proxy.uds import UdsError

    secrets_dir = tmp_path / "tmp"
    secrets_dir.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(secrets_dir))
    monkeypatch.setenv("HEADROOM_PROXY_CONFIG_JSON", "prior-value")
    monkeypatch.delenv("HEADROOM_PROXY_CONFIG_SECRETS_FILE", raising=False)
    not_a_socket = tmp_path / "proxy.sock"
    not_a_socket.write_text("regular file", encoding="utf-8")

    with pytest.raises(UdsError):
        server.run_server(
            ProxyConfig(proxy_token="FAKE-PROXYTOKEN", uds=str(not_a_socket)),
            workers=2,
            print_banner=False,
        )

    assert list(secrets_dir.iterdir()) == []
    assert os.environ["HEADROOM_PROXY_CONFIG_JSON"] == "prior-value"
    assert "HEADROOM_PROXY_CONFIG_SECRETS_FILE" not in os.environ


def test_worker_without_its_secrets_file_refuses_to_start(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from headroom.exceptions import ConfigurationError
    from headroom.proxy import server

    # Falling back to env defaults here would drop the proxy token.
    monkeypatch.setenv("HEADROOM_PROXY_CONFIG_JSON", json.dumps({"host": "0.0.0.0"}))
    monkeypatch.setenv("HEADROOM_PROXY_CONFIG_SECRETS_FILE", str(tmp_path / "gone.json"))

    with pytest.raises(ConfigurationError, match="restart the proxy"):
        server._proxy_config_from_env()


def test_toin_query_pattern_does_not_keep_the_text_before_a_colon() -> None:
    toin = ToolIntelligenceNetwork(TOINConfig(storage_path=""))

    pattern = toin._anonymize_query_pattern("AKIAABCDEFGHIJKLMNOP:secretvalue")

    assert pattern is not None
    assert "AKIA" not in pattern
    assert "secretvalue" not in pattern
