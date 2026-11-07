"""Am I running the catalog-driven Copilot routing, or the old name heuristic?

Prints a single verdict. Discriminates by asking for the wire API of
``mai-code-1-flash-picker``: the old name heuristic answers ``completions``
(which the upstream rejects with 400), the catalog answers ``responses``.
A model like ``gpt-5.4`` cannot tell the two apart -- both answer ``responses``
-- which is why an eyeball test on gpt-5.4 looks unchanged either way.
"""

from __future__ import annotations

import sys

DISCRIMINATOR = "mai-code-1-flash-picker"


def main() -> int:
    print(f"python      : {sys.executable}")
    try:
        import headroom

        print(f"headroom    : {headroom.__file__}")
    except Exception as exc:  # noqa: BLE001
        print(f"headroom    : IMPORT FAILED: {exc}")
        return 2

    have_planner = have_catalog = False
    try:
        import headroom.proxy.transport_planner  # noqa: F401

        have_planner = True
    except Exception:  # noqa: BLE001
        pass
    try:
        import headroom.models.copilot_catalog  # noqa: F401

        have_catalog = True
    except Exception:  # noqa: BLE001
        pass

    print(f"new modules : transport_planner={have_planner} copilot_catalog={have_catalog}")

    resolver = None
    try:
        from headroom.providers.copilot.wrap import resolve_wire_api_for_model

        resolver = resolve_wire_api_for_model
    except Exception:  # noqa: BLE001
        pass
    print(f"launcher fix: resolve_wire_api_for_model={'present' if resolver else 'MISSING'}")

    if not (have_planner and have_catalog and resolver):
        print()
        print("VERDICT: OLD CODE — these changes are NOT active.")
        print("         You are almost certainly running the separately installed")
        print("         `headroom` binary instead of the repo venv.")
        return 1

    # Live discriminator: ask the real upstream which wire serves the model.
    try:
        from headroom.copilot_auth import resolve_subscription_bearer_token_details

        res = resolve_subscription_bearer_token_details()
    except Exception as exc:  # noqa: BLE001
        print(f"\nWARNING: could not resolve a Copilot token ({exc}).")
        print("VERDICT: NEW CODE loaded, but could not confirm live discovery.")
        return 0

    if res is None:
        print("\nWARNING: no Copilot subscription token resolved.")
        print("VERDICT: NEW CODE loaded, but could not confirm live discovery.")
        return 0

    chosen = resolver(DISCRIMINATOR, api_url=res.api_url, token=res.token)
    print(f"live check  : wire API for {DISCRIMINATOR!r} -> {chosen!r}")
    print()
    if chosen == "responses":
        print("VERDICT: NEW CODE ACTIVE and live discovery works.")
        print("         The old heuristic would have said 'completions' for")
        print(f"         {DISCRIMINATOR}, which the upstream rejects with 400.")
        return 0
    print("VERDICT: new modules are present but discovery did not take effect")
    print("         (it fell back to the name heuristic). Check network access to")
    print(f"         {res.api_url}/models.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
