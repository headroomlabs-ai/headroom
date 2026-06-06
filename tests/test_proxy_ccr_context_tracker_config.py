from headroom.proxy.server import HeadroomProxy, ProxyConfig


def test_proxy_threads_ccr_max_turn_distance_into_context_tracker() -> None:
    config = ProxyConfig(ccr_max_turn_distance=3)

    proxy = HeadroomProxy(config)

    assert proxy.ccr_context_tracker is not None
    assert proxy.ccr_context_tracker.config.max_turn_distance == 3
