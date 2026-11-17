from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap

import pytest


@pytest.mark.parametrize(
    ("deadline", "expected"),
    [("0.001", {"timed_out": True}), ("1.0", {"value": 42})],
)
def test_startup_deadline_bounds_actual_compression_work(deadline, expected, tmp_path) -> None:
    program = textwrap.dedent(
        """
        import asyncio
        import json
        import time
        from headroom.proxy import helpers
        from headroom.proxy.server import HeadroomProxy, ProxyConfig

        proxy = HeadroomProxy(ProxyConfig(
            optimize=False, cache_enabled=False, rate_limit_enabled=False,
            cost_tracking_enabled=False, log_requests=False, image_optimize=False,
        ))

        def work():
            time.sleep(0.05)
            return 42

        async def main():
            try:
                value = await proxy._run_compression_in_executor(
                    work, timeout=helpers.COMPRESSION_TIMEOUT_SECONDS
                )
                print(json.dumps({"value": value}))
            except asyncio.TimeoutError:
                print(json.dumps({"timed_out": True}))
            finally:
                proxy._compression_executor.shutdown(wait=True)

        asyncio.run(main())
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", program],
        env={
            **os.environ,
            "HEADROOM_COMPRESSION_TIMEOUT_SECONDS": deadline,
            "HEADROOM_WORKSPACE_DIR": str(tmp_path),
        },
        capture_output=True,
        text=True,
        check=True,
        timeout=120,
    )
    assert json.loads(result.stdout) == expected
