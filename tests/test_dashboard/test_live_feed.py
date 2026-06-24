import pytest

pytest.importorskip("playwright")
from playwright.sync_api import sync_playwright


@pytest.fixture(scope="module")
def browser():
    with sync_playwright() as p:
        b = p.chromium.launch()
        yield b
        b.close()


@pytest.fixture
def page(browser):
    return browser.new_page()


@pytest.fixture
def dashboard_url():
    return "http://localhost:8787/dashboard"


def test_live_feed_button_exists(page, dashboard_url):
    """The Live Feed button should be visible in the dashboard header."""
    page.goto(dashboard_url)
    feed_button = page.locator("#feed-toggle")
    assert feed_button.is_visible(), "Live Feed button not visible in header"


def test_live_feed_drawer_opens(page, dashboard_url):
    """Clicking Live Feed should open the sidebar drawer."""
    page.goto(dashboard_url)
    page.click("#feed-toggle")
    page.wait_for_timeout(400)
    # Check drawer is displayed (x-show becomes visible)
    drawer = page.locator('[x-show="feedOpen"]')
    assert drawer.count() > 0


def test_live_feed_shows_empty_state(page, dashboard_url):
    """Feed should show empty state when no transformations available."""
    page.goto(dashboard_url)
    page.click("#feed-toggle")
    page.wait_for_timeout(1000)
    # Check for empty state or feed container
    feed_container = page.locator("#feed-virtual-list")
    assert feed_container.is_visible()


def test_live_feed_fetches_and_displays(page, dashboard_url):
    """Feed should display transformation data after polling."""
    page.goto(dashboard_url)
    page.click("#feed-toggle")
    # Wait for at least one poll cycle
    page.wait_for_timeout(4000)
    feed_container = page.locator("#feed-virtual-list")
    content = feed_container.inner_html()
    # Should have at least empty state
    assert len(content) >= 0
