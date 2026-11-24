"""Keep LiteLLM's price table current in a running proxy.

LiteLLM prices from a table bundled with each of its releases, and only
downloads the current one during ``import litellm`` when
``LITELLM_LOCAL_MODEL_COST_MAP`` is unset. Enterprise launchers set it, because
that import-time download stalled proxy startup behind TLS-inspecting proxies
for up to 25 seconds. The bundled table then goes stale: litellm 1.101.6 has no
``claude-opus-5-5``, so its savings fell back to a flat default rate.

The proxy's lifespan runs :func:`refresh_periodically`, which fetches the current
table once the proxy is serving and again every day, and merges it into
``litellm.model_cost``. Startup never waits on it, and a failed fetch leaves the
bundled table in place. Each ``--workers`` process has its own table and fetches
its own copy. A proxy without the launcher pin has already downloaded the table
during import, so its first refresh changes nothing. Disabled by
``HEADROOM_PRICE_REFRESH=off``, by ``HEADROOM_OFFLINE``, and without litellm.
"""

from __future__ import annotations

import asyncio
import http.client
import json
import logging
import os
import threading
import urllib.request
from typing import Any

logger = logging.getLogger(__name__)

LITELLM_PRICES_URL = (
    "https://raw.githubusercontent.com/BerriAI/litellm/main/model_prices_and_context_window.json"
)
REFRESH_INTERVAL_SECONDS = 24 * 3600
#: Seconds after a refresh before litellm's lowercase index is invalidated
#: again; far longer than a rebuild of the index takes.
INDEX_SETTLE_SECONDS = 5.0
_FETCH_TIMEOUT_SECONDS = 15.0
#: A real table lists thousands of models. Anything far smaller is an error
#: page or a truncated download, and must not be merged.
_MIN_MODELS = 1000


def is_price_refresh_enabled() -> bool:
    from headroom.offline import is_offline
    from headroom.pricing.litellm_pricing import LITELLM_AVAILABLE

    if is_offline() or not LITELLM_AVAILABLE:
        return False
    return os.environ.get("HEADROOM_PRICE_REFRESH", "").strip().lower() not in {
        "0",
        "off",
        "false",
        "no",
    }


def fetch_litellm_prices(timeout: float = _FETCH_TIMEOUT_SECONDS) -> dict[str, Any]:
    """Download and validate LiteLLM's current price table.

    Trusts the same roots as the proxy (OS store, ``HEADROOM_CA_BUNDLE``,
    ``SSL_CERT_FILE``), so corporate TLS inspection works. Raises on any
    failure; :func:`refresh_litellm_prices` decides what a failure means.
    """
    from headroom.offline import guard_egress
    from headroom.proxy.ssl_context import build_urlopen_context

    guard_egress("LiteLLM price table refresh", LITELLM_PRICES_URL)
    request = urllib.request.Request(
        LITELLM_PRICES_URL,
        headers={"Accept": "application/json", "User-Agent": "headroom-price-refresh"},
    )
    context = build_urlopen_context()
    with urllib.request.urlopen(request, timeout=timeout, context=context) as response:  # noqa: S310 - fixed https URL
        table = json.loads(response.read().decode("utf-8"))
    models = (
        {k: v for k, v in table.items() if isinstance(v, dict)} if isinstance(table, dict) else {}
    )
    if len(models) < _MIN_MODELS:
        raise ValueError(
            f"LiteLLM price table from {LITELLM_PRICES_URL} lists {len(models)} models, "
            f"expected at least {_MIN_MODELS}"
        )
    return models


def _clear_price_caches() -> None:
    """Forget every model resolution and rate memoised against the old table."""
    from headroom.pricing import litellm_pricing
    from headroom.pricing.counterfactual import long_context_threshold, resolve_rates
    from headroom.proxy.savings_tracker import _resolve_litellm_model

    litellm_pricing._resolved_model_cache.clear()
    resolve_rates.cache_clear()
    long_context_threshold.cache_clear()
    _resolve_litellm_model.cache_clear()


#: Longest ``alias_of`` chain followed; deeper (or cyclic) aliases are dropped.
_MAX_ALIAS_DEPTH = 4


def _resolve_aliases(models: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Rows with each ``alias_of`` row replaced by its target's prices.

    Current tables publish aliases such as ``bedrock/claude-opus-latest`` as a
    row holding only ``alias_of``; older litellm releases cannot follow it, so
    the row would carry no price. An alias whose target is missing is left out,
    keeping whatever the bundled table had for that name.
    """
    resolved: dict[str, dict[str, Any]] = {}
    for key, row in models.items():
        target, depth = row, 0
        while "alias_of" in target and depth < _MAX_ALIAS_DEPTH:
            nxt = models.get(target["alias_of"])
            if not isinstance(nxt, dict):
                break
            target, depth = nxt, depth + 1
        if "alias_of" in target:
            continue
        own = {k: v for k, v in row.items() if k != "alias_of"}
        resolved[key] = {**target, **own} if target is not row else dict(row)
    return resolved


def _invalidate_lowercase_index() -> None:
    """Drop litellm's lowercase model-name index, so its next lookup rebuilds it."""
    from headroom.pricing.litellm_pricing import litellm

    invalidate = getattr(
        getattr(litellm, "utils", None), "_invalidate_model_cost_lowercase_map", None
    )
    if invalidate is not None:
        invalidate()


def merge_litellm_prices(models: dict[str, Any]) -> int:
    """Merge ``models`` into LiteLLM's table and drop stale caches.

    Each fetched row replaces the current one whole, so a rate upstream has
    retired (a dropped long-context tier) goes with it. Alias rows are resolved
    to their target's prices, after Headroom's own rows (DeepSeek off-peak
    rates) are laid on, so aliases inherit those too. Returns how many rows
    changed.

    The live table is never changed in place. A request thread may be
    iterating it (litellm rebuilds its lowercase name index that way), and
    growing the dict, even in one ``dict.update``, fails that iteration, so
    the model reads as unmapped. The merged table is built aside and bound to
    ``litellm.model_cost`` in one assignment; a reader already holding the old
    table finishes on it.
    """
    from headroom.pricing.litellm_pricing import deepseek_rows, litellm, mirrored_rows

    if litellm is None:
        return 0
    current = litellm.model_cost
    # Headroom's rows go on before aliases resolve, so an alias such as
    # deepseek/deepseek-flash-latest takes the off-peak rate of its target,
    # and again after, for any of them the fetched table does not list.
    ours = {key: row for key, row in deepseek_rows(models).items() if key in models}
    rows = _resolve_aliases({**models, **ours})
    rows.update(deepseek_rows({**current, **rows}))
    # A bare row Headroom copied from a prefixed one follows that row, unless
    # the fetched table now lists the bare name itself.
    rows.update({k: v for k, v in mirrored_rows(rows).items() if k not in models})
    changed = {key: row for key, row in rows.items() if current.get(key) != row}
    if not changed:
        return 0
    try:
        litellm.model_cost = {**current, **changed}
        # litellm infers a bare model's provider from per-provider name sets
        # built at import, and caches a lowercase index of the keys; new models
        # need both, or `cost_per_token` cannot place them.
        litellm.add_known_models(changed)
        _invalidate_lowercase_index()
    finally:
        _clear_price_caches()
    return len(changed)


def refresh_litellm_prices() -> int | None:
    """Fetch and merge the current table. Returns rows changed, or None on failure."""
    from headroom.offline import OfflineEgressBlocked, note_refusal

    try:
        changed = merge_litellm_prices(fetch_litellm_prices())
    except OfflineEgressBlocked as blocked:
        note_refusal(blocked, logger)
        return None
    # OSError covers URLError, TLS and timeouts; HTTPException a truncated body;
    # ValueError bad JSON or a table too small to be real.
    except (OSError, http.client.HTTPException, ValueError) as exc:
        logger.warning(
            "event=price_table_refresh_failed url=%s error=%s; savings keep using the "
            "price table bundled with litellm. Set HEADROOM_PRICE_REFRESH=off to stop trying.",
            LITELLM_PRICES_URL,
            type(exc).__name__,
        )
        return None
    logger.info("event=price_table_refreshed changed=%d url=%s", changed, LITELLM_PRICES_URL)
    return changed


async def _refresh_in_daemon_thread() -> None:
    """Run one refresh on a daemon thread and wait for it.

    Not asyncio.to_thread: the interpreter joins executor threads at exit,
    so a download in flight at shutdown would hold the process up to the fetch
    timeout after the task was cancelled. A daemon thread is simply abandoned.
    """
    loop = asyncio.get_running_loop()
    finished = loop.create_future()

    def finish() -> None:
        if not finished.done():
            finished.set_result(None)

    def run() -> None:
        try:
            refresh_litellm_prices()
        except Exception:  # a thread boundary: report it, the next run tries again
            logger.exception("event=price_table_refresh_crashed url=%s", LITELLM_PRICES_URL)
        finally:
            try:
                loop.call_soon_threadsafe(finish)
            except RuntimeError:  # the loop closed during shutdown; nobody is waiting
                pass

    threading.Thread(target=run, name="headroom-price-refresh", daemon=True).start()
    await finished


async def refresh_periodically(interval: float = REFRESH_INTERVAL_SECONDS) -> None:
    """Refresh now and then every interval seconds until cancelled."""
    if not is_price_refresh_enabled():
        return
    while True:
        await _refresh_in_daemon_thread()
        # A lookup that was rebuilding litellm's lowercase index when the
        # refresh swapped the table can store an index of the old table after
        # the refresh invalidated it, and litellm rebuilds only when an indexed
        # name disappears. Invalidating once more shortly after discards that
        # index: until then a refreshed model may miss by a mixed-case name,
        # never by its exact name.
        await asyncio.sleep(INDEX_SETTLE_SECONDS)
        _invalidate_lowercase_index()
        await asyncio.sleep(interval)


__all__ = [
    "LITELLM_PRICES_URL",
    "fetch_litellm_prices",
    "is_price_refresh_enabled",
    "merge_litellm_prices",
    "refresh_litellm_prices",
    "refresh_periodically",
]
