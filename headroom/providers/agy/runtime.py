"""Runtime env builder for agy-specific MITM proxy wiring.

Pure data transform: given a terminator URL and a CA trust bundle path,
produce the child environment dict that routes agy through the CONNECT
terminator while trusting the minted CA bundle.

No side effects; no I/O; no subprocess.
"""

from __future__ import annotations

from pathlib import Path


def build_agy_env(
    *,
    terminator_url: str,
    bundle_path: Path,
    base_env: dict[str, str],
) -> dict[str, str]:
    """Return a new env dict suitable for launching agy through the MITM terminator.

    Parameters
    ----------
    terminator_url:
        Full HTTP URL of the AgyCONNECTTerminator (e.g. ``http://127.0.0.1:<port>``).
    bundle_path:
        Path to the combined CA trust bundle produced by
        ``headroom.proxy.agy_ca.build_combined_bundle``.  Set in all three
        trust-bundle env vars so Python, Node.js, and curl all see it.
    base_env:
        Base environment (typically ``os.environ.copy()``).  A fresh copy is
        returned — ``base_env`` is never mutated.

    Returns
    -------
    dict[str, str]
        New environment dict with proxy and CA vars wired for agy.

    Notes
    -----
    Corporate proxy chaining works without any extra plumbing here: this
    function returns a COPY and never mutates ``base_env`` or
    ``os.environ``.  The CONNECT terminator runs in the PARENT process and
    therefore still reads the original corporate ``os.environ["HTTPS_PROXY"]``
    when it chains non-allowlisted CONNECTs upstream
    (see ``agy_terminator.py:_handle_blind_tunnel``).  Only the CHILD agy
    process receives ``HTTPS_PROXY=terminator_url`` so that all of its
    traffic is routed into the terminator first.
    """
    bundle_str = str(bundle_path)
    env = dict(base_env)  # copy — never mutate caller's dict

    # Drop the wrapper's session-scoped savings redirection. Those vars belong to
    # THIS process's in-proxy funnel (savings written to a temp dir that is
    # deleted on exit, inbox-emit marker set); the agy child — and the
    # `headroom mcp serve` grandchild it spawns, which is headroom code — must
    # not inherit them and write into a sink that disappears.
    for leaked in (
        "HEADROOM_AGY_INBOX_EMIT",
        "HEADROOM_SAVINGS_PATH",
        "HEADROOM_SAVINGS_EVENTS_PATH",
        "HEADROOM_OTEL_METRICS_ENABLED",
    ):
        env.pop(leaked, None)

    # Route all traffic through the CONNECT terminator.
    env["HTTPS_PROXY"] = terminator_url
    env["HTTP_PROXY"] = terminator_url
    # Extend, never replace: on a corporate machine the inherited NO_PROXY names
    # hosts that MUST bypass the proxy, and dropping them would tunnel them
    # through the terminator.
    inherited_no_proxy = (env.get("NO_PROXY") or env.get("no_proxy") or "").strip().strip(",")
    env["NO_PROXY"] = (
        f"127.0.0.1,localhost,{inherited_no_proxy}" if inherited_no_proxy else "127.0.0.1,localhost"
    )

    # Trust our minted CA bundle — blanking these would break MITM.
    env["SSL_CERT_FILE"] = bundle_str
    env["CACERT_PATH"] = bundle_str
    env["NODE_EXTRA_CA_CERTS"] = bundle_str

    return env
