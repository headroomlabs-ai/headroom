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

    # Route all traffic through the CONNECT terminator.
    env["HTTPS_PROXY"] = terminator_url
    env["HTTP_PROXY"] = terminator_url
    env["NO_PROXY"] = "127.0.0.1,localhost"

    # Trust our minted CA bundle — blanking these would break MITM.
    env["SSL_CERT_FILE"] = bundle_str
    env["CACERT_PATH"] = bundle_str
    env["NODE_EXTRA_CA_CERTS"] = bundle_str

    return env
