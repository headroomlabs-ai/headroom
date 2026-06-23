from __future__ import annotations

import builtins
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta
from types import SimpleNamespace

import pytest

import headroom.reporting as reporting
from headroom.reporting import generator


@dataclass
class FakeMetrics:
    request_id: str
    model: str
    mode: str
    timestamp: datetime
    tokens_input_before: int
    tokens_input_after: int
    cache_alignment_score: float
    waste_signals: dict[str, int]


class FakeStorage:
    def __init__(self, stats: dict, items: list[FakeMetrics]) -> None:
        self._stats = stats
        self._items = items
        self.closed = False

    def get_summary_stats(self, start_time, end_time):
        return dict(self._stats)

    def iter_all(self):
        return iter(self._items)

    def close(self) -> None:
        self.closed = True


def test_reporting_public_export() -> None:
    assert reporting.generate_report is generator.generate_report
    assert reporting.__all__ == ["generate_report"]


def test_get_jinja2_template_success_with_stub(monkeypatch) -> None:
    class FakeTemplate:
        def __init__(self, template_str: str) -> None:
            self.template_str = template_str

        def render(self, **kwargs) -> str:
            return f"{self.template_str}:{kwargs['name']}"

    monkeypatch.setitem(sys.modules, "jinja2", SimpleNamespace(Template=FakeTemplate))
    template = generator._get_jinja2_template("hello")
    assert template.render(name="world") == "hello:world"


def test_get_jinja2_template_raises_helpful_error(monkeypatch) -> None:
    real_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "jinja2":
            raise ImportError("missing")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(ImportError, match="jinja2 is required for report generation"):
        generator._get_jinja2_template("ignored")


def test_build_waste_histogram_empty_and_filtered_data() -> None:
    now = datetime(2026, 4, 23, 12, 0, 0)
    metrics = [
        FakeMetrics(
            request_id="before",
            model="gpt-4o",
            mode="audit",
            timestamp=now - timedelta(days=2),
            tokens_input_before=100,
            tokens_input_after=90,
            cache_alignment_score=10,
            waste_signals={"json_bloat": 5},
        ),
        FakeMetrics(
            request_id="inside",
            model="gpt-4o",
            mode="optimize",
            timestamp=now,
            tokens_input_before=200,
            tokens_input_after=100,
            cache_alignment_score=70,
            waste_signals={"json_bloat": 30, "html_noise": 10, "dynamic_date": 5, "reread": 20},
        ),
        FakeMetrics(
            request_id="flat",
            model="gpt-4o",
            mode="audit",
            timestamp=now,
            tokens_input_before=50,
            tokens_input_after=50,
            cache_alignment_score=50,
            waste_signals={"whitespace": 4},
        ),
        FakeMetrics(
            request_id="after",
            model="gpt-4o",
            mode="audit",
            timestamp=now + timedelta(days=2),
            tokens_input_before=100,
            tokens_input_after=20,
            cache_alignment_score=20,
            waste_signals={"base64": 50},
        ),
    ]
    histogram = generator._build_waste_histogram(
        FakeStorage({}, metrics),
        start_time=now - timedelta(hours=1),
        end_time=now + timedelta(hours=1),
    )

    assert histogram[0] == {"label": "History Bloat", "tokens": 55, "percentage": 100.0}
    assert histogram[1] == pytest.approx(
        {"label": "Tool JSON Bloat", "tokens": 30, "percentage": 54.54545454545454}
    )
    # "reread" surfaces in the histogram but is excluded from known_waste,
    # so History Bloat above stays 100 - 45 = 55.
    assert any(
        item["label"] == "Re-served Tool Results" and item["tokens"] == 20 for item in histogram
    )
    assert any(item["label"] == "HTML Noise" and item["tokens"] == 10 for item in histogram)
    assert any(item["label"] == "Dynamic Dates" and item["tokens"] == 5 for item in histogram)
    assert any(item["label"] == "Base64 Blobs" and item["tokens"] == 0 for item in histogram)

    empty = generator._build_waste_histogram(FakeStorage({}, []), None, None)
    assert all(item["tokens"] == 0 and item["percentage"] == 0 for item in empty)


def test_get_top_waste_requests_sorts_filters_and_limits() -> None:
    now = datetime(2026, 4, 23, 12, 0, 0)
    metrics = [
        FakeMetrics("one", "gpt-4o", "audit", now, 400, 100, 80, {}),
        FakeMetrics("two", "gpt-4o-mini", "optimize", now, 350, 330, 70, {}),
        FakeMetrics("three", "claude", "audit", now - timedelta(days=3), 1000, 10, 50, {}),
        FakeMetrics("four", "claude", "audit", now + timedelta(days=3), 1000, 200, 40, {}),
    ]
    top_requests = generator._get_top_waste_requests(
        FakeStorage({}, metrics),
        start_time=now - timedelta(hours=1),
        end_time=now + timedelta(hours=1),
        limit=1,
    )
    assert top_requests == [
        {
            "request_id": "one",
            "model": "gpt-4o",
            "mode": "audit",
            "tokens_before": 400,
            "tokens_saved": 300,
            "cache_alignment": 80,
        }
    ]


def test_generate_recommendations_for_heavy_waste_and_for_getting_started() -> None:
    stats = {
        "avg_cache_alignment": 40,
        "audit_count": 7,
        "optimize_count": 3,
        "total_tokens_saved": 120000,
        "estimated_savings": "$1.23",
    }
    histogram = [
        {"label": "Tool JSON Bloat", "tokens": 15000, "percentage": 100},
        {"label": "History Bloat", "tokens": 60000, "percentage": 50},
    ]
    recommendations = generator._generate_recommendations(stats, histogram, top_requests=[{}])
    titles = [item["title"] for item in recommendations]
    assert titles == [
        "Improve Cache Alignment",
        "Enable Tool Output Compression",
        "Review Rolling Window Settings",
        "Switch to Optimize Mode",
        "Continue Monitoring",
    ]
    assert "15,000" in recommendations[1]["description"]
    assert "60,000" in recommendations[2]["description"]

    starter = generator._generate_recommendations(
        {
            "avg_cache_alignment": 90,
            "audit_count": 1,
            "optimize_count": 1,
            "total_tokens_saved": 0,
            "estimated_savings": "$0.00",
        },
        [{"label": "Tool JSON Bloat", "tokens": 1, "percentage": 100}],
        top_requests=[],
    )
    assert starter == [
        {
            "title": "Get Started",
            "description": "No optimizations applied yet. Try setting headroom_mode='optimize' "
            "on your next request to start seeing token savings.",
        }
    ]


@pytest.mark.parametrize(
    ("start_time", "end_time", "expected_period"),
    [
        (
            datetime(2026, 4, 20, 8, 0, 0),
            datetime(2026, 4, 23, 18, 0, 0),
            "2026-04-20 to 2026-04-23",
        ),
        (datetime(2026, 4, 20, 8, 0, 0), None, "Since 2026-04-20"),
        (None, datetime(2026, 4, 23, 18, 0, 0), "Until 2026-04-23"),
        (None, None, "All time"),
    ],
)
def test_generate_report_writes_output_and_closes_storage(
    monkeypatch, tmp_path, start_time, end_time, expected_period
) -> None:
    storage = FakeStorage(
        {
            "total_requests": 3,
            "total_tokens_saved": 50,
            "avg_tokens_saved": 16.6,
            "total_tokens_before": 100,
            "total_tokens_after": 0,
            "avg_cache_alignment": 82,
            "audit_count": 1,
            "optimize_count": 2,
        },
        [],
    )
    render_calls: list[dict] = []

    class FakeTemplate:
        def render(self, **kwargs) -> str:
            render_calls.append(kwargs)
            return "<html>report</html>"

    monkeypatch.setattr(generator, "create_storage", lambda store_url: storage)
    monkeypatch.setattr(
        generator, "_build_waste_histogram", lambda *args: [{"label": "x", "tokens": 1}]
    )
    monkeypatch.setattr(
        generator, "_get_top_waste_requests", lambda *args, **kwargs: [{"request_id": "abc"}]
    )
    monkeypatch.setattr(
        generator, "_generate_recommendations", lambda *args: [{"title": "Keep going"}]
    )
    monkeypatch.setattr(generator, "_get_jinja2_template", lambda template_str: FakeTemplate())
    monkeypatch.setattr(
        generator,
        "estimate_cost",
        lambda tokens, output_tokens, model: {100: 2.0, 0: None}[tokens],
    )
    monkeypatch.setattr(generator, "format_cost", lambda cost: f"${cost:.2f}")

    output_path = tmp_path / "report.html"
    result = generator.generate_report(
        "sqlite:///demo.db",
        output_path=str(output_path),
        start_time=start_time,
        end_time=end_time,
    )

    assert result == str(output_path)
    assert output_path.read_text() == "<html>report</html>"
    assert render_calls[0]["period"] == expected_period
    assert render_calls[0]["stats"]["tpm_multiplier"] == 100.0
    assert render_calls[0]["stats"]["estimated_savings"] == "$2.00"
    assert storage.closed is True


def test_generate_report_closes_storage_when_render_fails(monkeypatch, tmp_path) -> None:
    storage = FakeStorage(
        {
            "total_requests": 0,
            "total_tokens_saved": 0,
            "avg_tokens_saved": 0,
            "total_tokens_before": 0,
            "total_tokens_after": 0,
            "avg_cache_alignment": 0,
            "audit_count": 0,
            "optimize_count": 0,
        },
        [],
    )

    class FakeTemplate:
        def render(self, **kwargs) -> str:
            raise RuntimeError("boom")

    monkeypatch.setattr(generator, "create_storage", lambda store_url: storage)
    monkeypatch.setattr(generator, "_build_waste_histogram", lambda *args: [])
    monkeypatch.setattr(generator, "_get_top_waste_requests", lambda *args, **kwargs: [])
    monkeypatch.setattr(generator, "_generate_recommendations", lambda *args: [])
    monkeypatch.setattr(generator, "_get_jinja2_template", lambda template_str: FakeTemplate())
    monkeypatch.setattr(generator, "estimate_cost", lambda *args: 0.0)
    monkeypatch.setattr(generator, "format_cost", lambda cost: "$0.00")

    with pytest.raises(RuntimeError, match="boom"):
        generator.generate_report("sqlite:///demo.db", output_path=str(tmp_path / "report.html"))
    assert storage.closed is True
