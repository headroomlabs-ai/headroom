import json

from headroom.pricing.opencode_prices import (
    OPENCODE_GO_SURFACE,
    OPENCODE_ZEN_SURFACE,
    OpenCodeReferencePricing,
    get_opencode_reference_pricing,
    pricing_surface_from_tags,
)


def test_pricing_surface_from_opencode_tags_detects_zen_and_go():
    assert (
        pricing_surface_from_tags(
            {
                "base-url": "https://opencode.ai",
                "original-path": "/zen/v1/chat/completions",
            }
        )
        == OPENCODE_ZEN_SURFACE
    )
    assert (
        pricing_surface_from_tags(
            {
                "base-url": "http://127.0.0.1:8799",
                "original-path": "/zen/go/v1/chat/completions",
            }
        )
        == OPENCODE_GO_SURFACE
    )
    assert (
        pricing_surface_from_tags(
            {
                "base-url": "https://api.openai.com",
                "original-path": "/v1/chat/completions",
            }
        )
        is None
    )


def test_opencode_reference_pricing_uses_scraped_prices_first(monkeypatch, tmp_path):
    import headroom.pricing.opencode_prices as opencode_prices

    monkeypatch.setenv("HEADROOM_OPENCODE_PRICING_CACHE", str(tmp_path / "cache.json"))
    opencode_prices._SCRAPE_CACHE.clear()
    opencode_prices._SCRAPE_CACHE_DATE.clear()
    monkeypatch.setattr(opencode_prices, "_write_surface_disk_cache", lambda *a, **k: None)
    monkeypatch.setattr(
        opencode_prices,
        "_fetch_docs_pricing",
        lambda surface: {
            "glm-5.2": OpenCodeReferencePricing(
                "glm-5.2",
                9.99 / 1_000_000,
                8.88 / 1_000_000,
                source="scraped_docs",
                source_url="https://opencode.ai/docs/test/",
            )
        },
    )

    pricing = get_opencode_reference_pricing("glm-5.2", OPENCODE_ZEN_SURFACE)
    assert pricing.input_per_1m == 9.99
    assert pricing.output_per_1m == 8.88
    assert pricing.source == "scraped_docs"


def test_opencode_reference_pricing_falls_back_when_scrape_has_no_model(monkeypatch, tmp_path):
    import headroom.pricing.opencode_prices as opencode_prices

    monkeypatch.setenv("HEADROOM_OPENCODE_PRICING_CACHE", str(tmp_path / "cache.json"))
    opencode_prices._SCRAPE_CACHE.clear()
    opencode_prices._SCRAPE_CACHE_DATE.clear()
    monkeypatch.setattr(opencode_prices, "_write_surface_disk_cache", lambda *a, **k: None)
    monkeypatch.setattr(opencode_prices, "_fetch_docs_pricing", lambda surface: {})

    assert get_opencode_reference_pricing("glm-5.2", OPENCODE_ZEN_SURFACE).input_per_1m == 1.4
    assert get_opencode_reference_pricing("glm-5.2", OPENCODE_GO_SURFACE).output_per_1m == 4.4
    assert get_opencode_reference_pricing("glm-5.2", None) is None


def test_opencode_pricing_parser_reads_docs_table():
    import headroom.pricing.opencode_prices as opencode_prices

    prices = opencode_prices._parse_pricing_table(
        """
        <table><tbody>
        <tr><td>GLM-5.2</td><td>$1.40</td><td>$4.40</td><td>$0.26</td><td>-</td></tr>
        <tr><td>Qwen3.7 Max</td><td>$2.50</td><td>$7.50</td><td>$0.50</td><td>$3.125</td></tr>
        </tbody></table>
        """,
        source_url="https://opencode.ai/docs/go/",
    )

    assert prices["glm-5.2"].input_per_1m == 1.4
    assert prices["qwen3.7-max"].cache_write_per_1m == 3.125
    assert prices["glm-5.2"].source == "scraped_docs"


def test_opencode_pricing_disk_cache_round_trip_and_hit(monkeypatch, tmp_path):
    import headroom.pricing.opencode_prices as opencode_prices

    monkeypatch.setenv("HEADROOM_OPENCODE_PRICING_CACHE", str(tmp_path / "cache.json"))
    monkeypatch.setattr(opencode_prices.paths, "process_is_stateless", lambda: False)
    opencode_prices._SCRAPE_CACHE.clear()
    opencode_prices._SCRAPE_CACHE_DATE.clear()
    calls: list[str] = []

    def _fake_fetch(surface: str):
        calls.append(surface)
        return {
            "glm-5.2": OpenCodeReferencePricing(
                "glm-5.2",
                9.99 / 1_000_000,
                8.88 / 1_000_000,
                source="scraped_docs",
                source_url="https://opencode.ai/docs/test/",
            )
        }

    monkeypatch.setattr(opencode_prices, "_fetch_docs_pricing", _fake_fetch)

    # First call fetches and persists the disk cache.
    pricing = get_opencode_reference_pricing("glm-5.2", OPENCODE_ZEN_SURFACE)
    assert pricing.input_per_1m == 9.99
    assert calls == [OPENCODE_ZEN_SURFACE]

    # Reset in-memory cache; the on-disk entry (today) must satisfy the next
    # call WITHOUT triggering a new fetch.
    opencode_prices._SCRAPE_CACHE.clear()
    opencode_prices._SCRAPE_CACHE_DATE.clear()
    pricing = get_opencode_reference_pricing("glm-5.2", OPENCODE_ZEN_SURFACE)
    assert pricing.input_per_1m == 9.99
    assert pricing.source == "scraped_docs"
    assert calls == [OPENCODE_ZEN_SURFACE]


def test_opencode_pricing_uses_stale_disk_cache_when_fetch_fails(monkeypatch, tmp_path):
    from datetime import date, timedelta

    import headroom.pricing.opencode_prices as opencode_prices

    cache_path = tmp_path / "cache.json"
    yesterday = (date.today() - timedelta(days=1)).isoformat()
    stale_payload = {
        "version": 1,
        "surfaces": {
            OPENCODE_ZEN_SURFACE: {
                "date": yesterday,
                "prices": {
                    "glm-5.2": {
                        "model": "glm-5.2",
                        "input_cost_per_token": 9.99 / 1_000_000,
                        "output_cost_per_token": 8.88 / 1_000_000,
                        "cache_read_input_token_cost": None,
                        "cache_write_input_token_cost": None,
                        "source": "scraped_docs",
                        "source_url": "https://opencode.ai/docs/test/",
                        "note": "",
                    }
                },
            }
        },
    }
    cache_path.write_text(json.dumps(stale_payload), encoding="utf-8")
    monkeypatch.setenv("HEADROOM_OPENCODE_PRICING_CACHE", str(cache_path))
    opencode_prices._SCRAPE_CACHE.clear()
    opencode_prices._SCRAPE_CACHE_DATE.clear()
    monkeypatch.setattr(opencode_prices, "_write_surface_disk_cache", lambda *a, **k: None)
    monkeypatch.setattr(
        opencode_prices,
        "_fetch_docs_pricing",
        lambda surface: (_ for _ in ()).throw(RuntimeError("network down")),
    )

    # Fetch fails -> the stale on-disk entry (yesterday) must win over the
    # hardcoded fallback table (which would report 1.4 instead of 9.99).
    pricing = get_opencode_reference_pricing("glm-5.2", OPENCODE_ZEN_SURFACE)
    assert pricing.input_per_1m == 9.99
    assert pricing.source == "scraped_docs"


def test_opencode_pricing_fetch_failure_with_no_disk_falls_to_hardcoded(monkeypatch, tmp_path):
    import headroom.pricing.opencode_prices as opencode_prices

    monkeypatch.setenv("HEADROOM_OPENCODE_PRICING_CACHE", str(tmp_path / "absent.json"))
    opencode_prices._SCRAPE_CACHE.clear()
    opencode_prices._SCRAPE_CACHE_DATE.clear()
    monkeypatch.setattr(opencode_prices, "_write_surface_disk_cache", lambda *a, **k: None)
    monkeypatch.setattr(
        opencode_prices,
        "_fetch_docs_pricing",
        lambda surface: (_ for _ in ()).throw(RuntimeError("network down")),
    )

    # No disk cache + fetch failure -> hardcoded fallback table.
    assert get_opencode_reference_pricing("glm-5.2", OPENCODE_ZEN_SURFACE).input_per_1m == 1.4
