"""One dashboard poll runs at a time.

A five-second ``setInterval`` and the ``R`` shortcut both call
``pollDashboard()``, which never waited for a previous invocation to finish.
Each invocation fetches ``/stats`` and ``/health``, so a slow endpoint or a
repeated manual refresh put two full sequences in flight at once.

These tests drive the real Alpine component in a browser. They gate
``window.fetch`` on a promise the test resolves, which is what makes "while the
first request is still pending" a deterministic state rather than a race.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from headroom.dashboard import get_dashboard_html, get_settings_html
from tests.test_dashboard_cache_net_playwright import _open_dashboard
from tests.test_dashboard_cache_ttl_playwright import _sample_stats

playwright = pytest.importorskip("playwright.sync_api")
Page = playwright.Page
sync_playwright = playwright.sync_playwright

# Replaces fetch with a call recorder that hangs until the test releases it,
# and stops the interval so the test drives pollDashboard() itself.
_GATE_FETCH = """
() => {
    const component = Alpine.$data(document.body);
    clearInterval(component.pollInterval);
    window.__urls = [];
    const gate = new Promise((resolve) => { window.__release = resolve; });
    window.fetch = (url) => {
        window.__urls.push(String(url));
        return gate.then(() => new Response('{}', {
            status: 200,
            headers: { 'content-type': 'application/json' },
        }));
    };
}
"""

# One poll fetches /stats and /health together.
FETCHES_PER_POLL = 2


@pytest.fixture
def page() -> Iterator[Page]:  # type: ignore[valid-type]
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        _open_dashboard(page, _sample_stats())
        page.evaluate(_GATE_FETCH)
        yield page
        browser.close()


@pytest.mark.parametrize(
    "html",
    [get_dashboard_html(), get_settings_html()],
    ids=["dashboard", "settings"],
)
def test_no_template_asks_alpine_to_run_init_again(html: str) -> None:
    """Both templates had it; the settings page fetched its schema twice."""
    assert 'x-init="init()"' not in html


def test_the_component_initializes_once() -> None:
    """Alpine calls the x-data object's ``init()`` itself.

    Naming it in ``x-init`` as well ran it twice: two 5s poll timers, two R
    shortcut listeners and two opening fetch sequences, for the lifetime of the
    page. Counting the opening sequence is enough to pin it — the duplicate
    timer and listener can only exist if ``init()`` ran more than once.
    """
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        polled: list[str] = []
        page.on(
            "request",
            lambda request: (
                polled.append(request.url)
                if "/stats" in request.url or "/health" in request.url
                else None
            ),
        )

        _open_dashboard(page, _sample_stats())

        assert len(polled) == FETCHES_PER_POLL, polled
        browser.close()


def _url_count(page: Page) -> int:  # type: ignore[valid-type]
    return int(page.evaluate("() => window.__urls.length"))


def test_a_second_poll_during_the_first_issues_no_further_requests(page: Page) -> None:  # type: ignore[valid-type]
    """The timer's entry point: two unforced polls, one fetch sequence."""
    counts = page.evaluate(
        """async () => {
            const component = Alpine.$data(document.body);
            const first = component.pollDashboard();
            const second = component.pollDashboard();
            const inFlight = window.__urls.length;
            window.__release();
            await Promise.all([first, second]);
            return { inFlight, settled: window.__urls.length };
        }"""
    )

    assert counts == {"inFlight": FETCHES_PER_POLL, "settled": FETCHES_PER_POLL}


def test_a_poll_after_the_first_completes_still_runs(page: Page) -> None:  # type: ignore[valid-type]
    """The guard suppresses overlap, not the next scheduled refresh."""
    page.evaluate(
        """async () => {
            const component = Alpine.$data(document.body);
            const first = component.pollDashboard();
            window.__release();
            await first;
            await component.pollDashboard();
        }"""
    )

    assert _url_count(page) == 2 * FETCHES_PER_POLL


def test_manual_refresh_during_a_poll_does_not_duplicate_it(page: Page) -> None:  # type: ignore[valid-type]
    """The R shortcut shares the guard, and works again once the poll lands."""
    # Deliberately not returned: evaluate() awaits a returned promise, and this
    # one cannot settle until the gate is released further down.
    page.evaluate("() => { Alpine.$data(document.body).pollDashboard(); }")
    assert _url_count(page) == FETCHES_PER_POLL

    page.keyboard.press("r")
    page.wait_for_timeout(100)
    assert _url_count(page) == FETCHES_PER_POLL

    page.evaluate("() => window.__release()")
    page.wait_for_timeout(100)
    page.keyboard.press("r")
    page.wait_for_timeout(100)
    assert _url_count(page) == 2 * FETCHES_PER_POLL


def test_a_failed_poll_releases_the_guard(page: Page) -> None:  # type: ignore[valid-type]
    """A rejection must not wedge the dashboard on stale numbers forever."""
    recovered = page.evaluate(
        """async () => {
            const component = Alpine.$data(document.body);
            component.fetchStats = () => Promise.reject(new Error('boom'));
            await component.pollDashboard().catch(() => {});

            let calls = 0;
            component.fetchStats = () => { calls += 1; return Promise.resolve(); };
            await component.pollDashboard();
            return calls;
        }"""
    )

    assert recovered == 1


def test_a_hidden_tab_still_skips_unforced_polls(page: Page) -> None:  # type: ignore[valid-type]
    """Guarding must not cost the existing background-tab saving."""
    counts = page.evaluate(
        """async () => {
            Object.defineProperty(document, 'hidden', {
                configurable: true,
                get: () => true,
            });
            const component = Alpine.$data(document.body);

            component.pollDashboard();
            const unforced = window.__urls.length;
            component.pollDashboard(true);
            return { unforced, forced: window.__urls.length };
        }"""
    )

    assert counts == {"unforced": 0, "forced": FETCHES_PER_POLL}
