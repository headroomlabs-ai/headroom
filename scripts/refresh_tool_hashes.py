#!/usr/bin/env python3
"""Refresh SHA-256 pins for externally fetched tool binaries (WEB-03).

Fetches every asset URL in ``headroom/tools.json``, computes its SHA-256, and
writes the digests back into the registry. Run it locally after bumping a tool
version, or let the ``tools-hash-refresh`` CI workflow run it.

    python scripts/refresh_tool_hashes.py            # populate/update pins
    python scripts/refresh_tool_hashes.py --check     # exit 1 if any pin drifts

Only ``https://`` URLs are accepted; a plaintext URL is a hard error.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.request
from pathlib import Path

REGISTRY = Path(__file__).resolve().parent.parent / "headroom" / "tools.json"


def _fetch_sha256(url: str) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": "headroom-tools-refresh/1"})
    digest = hashlib.sha256()
    with urllib.request.urlopen(req, timeout=120) as resp:  # noqa: S310 - https enforced below
        for chunk in iter(lambda: resp.read(1024 * 64), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="fail if any pin is missing/stale")
    args = parser.parse_args()

    data = json.loads(REGISTRY.read_text())
    seen: dict[str, str] = {}
    drift: list[str] = []

    for tool_name, tool in data.get("tools", {}).items():
        for platform, asset in tool.get("assets", {}).items():
            url = asset.get("url")
            if not url:
                continue
            if not url.startswith("https://"):
                print(f"ERROR: {tool_name}/{platform}: non-https url {url!r}", file=sys.stderr)
                return 2
            if url not in seen:
                print(f"fetching {tool_name}/{platform} …", file=sys.stderr)
                seen[url] = _fetch_sha256(url)
            digest = seen[url]
            if asset.get("sha256") != digest:
                drift.append(f"{tool_name}/{platform}")
                if not args.check:
                    asset["sha256"] = digest

    if args.check:
        if drift:
            print("stale pins: " + ", ".join(drift), file=sys.stderr)
            return 1
        print("all tool pins up to date")
        return 0

    REGISTRY.write_text(json.dumps(data, indent=2) + "\n")
    print(f"updated {len(drift)} pin(s) in {REGISTRY}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
