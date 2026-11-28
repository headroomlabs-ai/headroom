"""Readiness and control status do not turn into source probes."""

from tests.unified_gateway.process.test_runtime_catalog import catalog_process  # noqa: F401
from tests.unified_gateway.test_gateway_tls import local_pki  # noqa: F401


def test_liveness_readiness_and_status_do_not_acquire_expired_source(catalog_process):  # noqa: F811
    client, admin, state, *_ = catalog_process
    assert client.get("/admin/gateway/status").status_code == 403
    assert client.get("/livez").status_code == 200
    assert client.get("/readyz").status_code == 200
    assert client.get("/admin/gateway/status", headers=admin).status_code == 200
    assert client.get("/__test/probe").json()["identity"] == 0
    assert client.post("/admin/gateway/catalog/refresh", headers=admin).json()["refreshed"] == 1
    assert client.get("/__test/probe").json()["identity"] == 1
    assert client.post("/__test/expire-source").status_code == 200
    client.post("/__test/clock/1016")
    assert client.get("/livez").status_code == 200
    assert client.get("/readyz").json()["profile"] == "gateway"
    assert client.get("/admin/gateway/status", headers=admin).status_code == 200
    assert client.get("/__test/probe").json()["identity"] == 1
    assert state["metadata_calls"] == 1 and state["generation_calls"] == 0
