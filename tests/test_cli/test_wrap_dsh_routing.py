"""DeepSeek target compatibility when wrapping against a shared proxy."""

from headroom.cli import wrap as wrap_cli


def test_deepseek_target_mismatch_starts_dedicated_proxy(monkeypatch):
    target = "https://deepseek.gateway.test/v1"
    health = {
        "version": wrap_cli._HEADROOM_VERSION,
        "config": {
            "backend": "anthropic",
            "memory": False,
            "learn": False,
            "code_graph": False,
        },
    }
    started = []
    monkeypatch.delenv("HEADROOM_BACKEND", raising=False)
    monkeypatch.setattr(wrap_cli, "_find_persistent_manifest", lambda port: None)
    monkeypatch.setattr(wrap_cli, "_check_proxy", lambda port: True)
    monkeypatch.setattr(wrap_cli, "_query_proxy_health", lambda port: health)
    monkeypatch.setattr(wrap_cli, "_live_proxy_clients", lambda *a, **kw: [999])
    monkeypatch.setattr(wrap_cli, "_find_available_port", lambda port: 8788)
    monkeypatch.setattr(
        wrap_cli, "_kill_proxy_by_pid", lambda *a: (_ for _ in ()).throw(AssertionError())
    )
    monkeypatch.setattr(
        wrap_cli, "_start_proxy", lambda port, **kw: started.append((port, kw)) or "proxy"
    )

    proc, port = wrap_cli._ensure_proxy_unlocked(
        8787, False, deepseek_api_url=target, agent_type="dsh"
    )

    assert (proc, port) == ("proxy", 8788)
    assert started[0][1]["deepseek_api_url"] == target


def test_deepseek_environment_target_and_explicit_precedence(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_TARGET_API_URL", "https://ambient.deepseek.test")
    args = {
        "backend": None,
        "openai_api_url": None,
        "anthropic_api_url": None,
        "vertex_api_url": None,
        "clear_vertex_api_url": False,
    }
    _, ambient = wrap_cli._effective_requested_proxy_routing(**args)
    _, explicit = wrap_cli._effective_requested_proxy_routing(
        **args, deepseek_api_url="https://explicit.deepseek.test/v1"
    )
    assert ambient["deepseek_api_url"] == "https://ambient.deepseek.test"
    assert explicit["deepseek_api_url"] == "https://explicit.deepseek.test/v1"
    assert (
        wrap_cli._proxy_routing_mismatches(
            {"deepseek_api_url": "https://explicit.deepseek.test"},
            backend=None,
            requested_api_urls=explicit,
        )
        == []
    )
