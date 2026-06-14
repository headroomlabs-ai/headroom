"""HTML report generator for Headroom SDK."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..storage import create_storage
from ..utils import estimate_cost, format_cost

if TYPE_CHECKING:
    pass


def _get_jinja2_template(template_str: str):
    """Lazily import jinja2 and create template."""
    try:
        from jinja2 import Template

        return Template(template_str)
    except ImportError as e:
        raise ImportError(
            "jinja2 is required for report generation. Install with: pip install headroom[reports]"
        ) from e


# HTML template embedded as string
REPORT_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Headroom Report - {{ generated_at }}</title>
    <style>
        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            line-height: 1.6;
            color: #333;
            background: #f5f5f5;
            padding: 20px;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
        }
        header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 20px;
        }
        header h1 {
            font-size: 2em;
            margin-bottom: 10px;
        }
        header p {
            opacity: 0.9;
        }
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }
        .stat-card {
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        .stat-card h3 {
            font-size: 0.9em;
            color: #666;
            margin-bottom: 5px;
        }
        .stat-card .value {
            font-size: 2em;
            font-weight: bold;
            color: #333;
        }
        .stat-card .value.positive {
            color: #22c55e;
        }
        .stat-card .value.warning {
            color: #f59e0b;
        }
        .section {
            background: white;
            padding: 25px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            margin-bottom: 20px;
        }
        .section h2 {
            font-size: 1.3em;
            margin-bottom: 20px;
            padding-bottom: 10px;
            border-bottom: 2px solid #eee;
        }
        .histogram {
            display: flex;
            flex-direction: column;
            gap: 10px;
        }
        .bar-row {
            display: flex;
            align-items: center;
            gap: 10px;
        }
        .bar-label {
            width: 120px;
            font-size: 0.9em;
            color: #666;
        }
        .bar-container {
            flex: 1;
            background: #eee;
            border-radius: 4px;
            height: 24px;
            overflow: hidden;
        }
        .bar {
            height: 100%;
            background: linear-gradient(90deg, #667eea, #764ba2);
            border-radius: 4px;
            display: flex;
            align-items: center;
            padding: 0 10px;
            color: white;
            font-size: 0.8em;
            min-width: fit-content;
        }
        table {
            width: 100%;
            border-collapse: collapse;
        }
        th, td {
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #eee;
        }
        th {
            background: #f9f9f9;
            font-weight: 600;
        }
        tr:hover {
            background: #f9f9f9;
        }
        .tag {
            display: inline-block;
            padding: 2px 8px;
            border-radius: 4px;
            font-size: 0.8em;
            font-weight: 500;
        }
        .tag.audit {
            background: #dbeafe;
            color: #1d4ed8;
        }
        .tag.optimize {
            background: #dcfce7;
            color: #16a34a;
        }
        .recommendations {
            list-style: none;
        }
        .recommendations li {
            padding: 15px;
            background: #f9f9f9;
            border-radius: 8px;
            margin-bottom: 10px;
            border-left: 4px solid #667eea;
        }
        .recommendations li strong {
            display: block;
            margin-bottom: 5px;
        }
        footer {
            text-align: center;
            padding: 20px;
            color: #666;
            font-size: 0.9em;
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>Headroom Context Analysis Report</h1>
            <p>Generated: {{ generated_at }} | Period: {{ period }}</p>
        </header>

        <div class="stats-grid">
            <div class="stat-card">
                <h3>Total Requests</h3>
                <div class="value">{{ stats.total_requests }}</div>
            </div>
            <div class="stat-card">
                <h3>Tokens Saved</h3>
                <div class="value positive">{{ "{:,}".format(stats.total_tokens_saved) }}</div>
            </div>
            <div class="stat-card">
                <h3>Avg Saved/Request</h3>
                <div class="value positive">{{ "{:,.0f}".format(stats.avg_tokens_saved) }}</div>
            </div>
            <div class="stat-card">
                <h3>Est. Cost Savings</h3>
                <div class="value positive">{{ stats.estimated_savings }}</div>
            </div>
            <div class="stat-card">
                <h3>Cache Alignment</h3>
                <div class="value {% if stats.avg_cache_alignment > 80 %}positive{% elif stats.avg_cache_alignment > 50 %}warning{% endif %}">{{ "{:.0f}%".format(stats.avg_cache_alignment) }}</div>
            </div>
            <div class="stat-card">
                <h3>TPM Headroom</h3>
                <div class="value positive">{{ "{:.1f}x".format(stats.tpm_multiplier) }}</div>
            </div>
        </div>

        <div class="section">
            <h2>Waste Histogram</h2>
            <div class="histogram">
                {% for item in waste_histogram %}
                <div class="bar-row">
                    <span class="bar-label">{{ item.label }}</span>
                    <div class="bar-container">
                        <div class="bar" style="width: {{ item.percentage }}%;">
                            {{ "{:,}".format(item.tokens) }} tokens
                        </div>
                    </div>
                </div>
                {% endfor %}
            </div>
        </div>

        <div class="section">
            <h2>Top 10 High-Waste Requests</h2>
            <table>
                <thead>
                    <tr>
                        <th>Request ID</th>
                        <th>Model</th>
                        <th>Mode</th>
                        <th>Tokens Before</th>
                        <th>Tokens Saved</th>
                        <th>Cache Align</th>
                    </tr>
                </thead>
                <tbody>
                    {% for req in top_requests %}
                    <tr>
                        <td><code>{{ req.request_id[:8] }}...</code></td>
                        <td>{{ req.model }}</td>
                        <td><span class="tag {{ req.mode }}">{{ req.mode }}</span></td>
                        <td>{{ "{:,}".format(req.tokens_before) }}</td>
                        <td>{{ "{:,}".format(req.tokens_saved) }}</td>
                        <td>{{ "{:.0f}%".format(req.cache_alignment) }}</td>
                    </tr>
                    {% endfor %}
                </tbody>
            </table>
        </div>

        <div class="section">
            <h2>Cache Alignment Analysis</h2>
            <p style="margin-bottom: 15px;">
                Average cache alignment score: <strong>{{ "{:.1f}%".format(stats.avg_cache_alignment) }}</strong>
            </p>
            <p style="color: #666;">
                {% if stats.avg_cache_alignment > 80 %}
                Excellent! Your prompts are well-aligned for provider caching.
                {% elif stats.avg_cache_alignment > 50 %}
                Good alignment, but there's room for improvement. Consider stabilizing dynamic content in system prompts.
                {% else %}
                Low cache alignment detected. Review system prompts for dynamic content (dates, timestamps, variable data).
                {% endif %}
            </p>
        </div>

        <div class="section">
            <h2>Recommendations</h2>
            <ul class="recommendations">
                {% for rec in recommendations %}
                <li>
                    <strong>{{ rec.title }}</strong>
                    {{ rec.description }}
                </li>
                {% endfor %}
            </ul>
        </div>

        <footer>
            Generated by Headroom SDK v0.1.0
        </footer>
    </div>
</body>
</html>
"""


def generate_report(
    store_url: str,
    output_path: str = "report.html",
    start_time: datetime | None = None,
    end_time: datetime | None = None,
) -> str:
    """
    Generate HTML report from stored metrics.

    Args:
        store_url: Storage URL (sqlite:// or jsonl://).
        output_path: Path for output HTML file.
        start_time: Filter by timestamp >= start_time.
        end_time: Filter by timestamp <= end_time.

    Returns:
        Path to generated report.
    """
    storage = create_storage(store_url)

    try:
        # Get summary stats
        stats = storage.get_summary_stats(start_time, end_time)

        # Calculate additional metrics
        if stats["total_tokens_before"] > 0:
            tpm_multiplier = stats["total_tokens_before"] / max(stats["total_tokens_after"], 1)
        else:
            tpm_multiplier = 1.0

        # Estimate cost savings (using gpt-4o pricing)
        cost_before = estimate_cost(stats["total_tokens_before"], 0, "gpt-4o") or 0.0
        cost_after = estimate_cost(stats["total_tokens_after"], 0, "gpt-4o") or 0.0
        estimated_savings = format_cost(cost_before - cost_after)

        stats["tpm_multiplier"] = tpm_multiplier
        stats["estimated_savings"] = estimated_savings

        # Build waste histogram
        waste_histogram = _build_waste_histogram(storage, start_time, end_time)

        # Get top requests by waste
        top_requests = _get_top_waste_requests(storage, start_time, end_time, limit=10)

        # Generate recommendations
        recommendations = _generate_recommendations(stats, waste_histogram, top_requests)

        # Format period string
        if start_time and end_time:
            period = f"{start_time.date()} to {end_time.date()}"
        elif start_time:
            period = f"Since {start_time.date()}"
        elif end_time:
            period = f"Until {end_time.date()}"
        else:
            period = "All time"

        # Render template
        template = _get_jinja2_template(REPORT_TEMPLATE)
        html = template.render(
            generated_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            period=period,
            stats=stats,
            waste_histogram=waste_histogram,
            top_requests=top_requests,
            recommendations=recommendations,
        )

        # Write output
        Path(output_path).write_text(html)

        return output_path

    finally:
        storage.close()


def _build_waste_histogram(
    storage: Any,
    start_time: datetime | None,
    end_time: datetime | None,
) -> list[dict[str, Any]]:
    """Build waste histogram data."""
    totals: dict[str, int] = {
        "json_bloat": 0,
        "html_noise": 0,
        "base64": 0,
        "whitespace": 0,
        "dynamic_date": 0,
        "reread": 0,
        "reread_compressed": 0,
        "history_bloat": 0,
    }

    for metrics in storage.iter_all():
        if start_time and metrics.timestamp < start_time:
            continue
        if end_time and metrics.timestamp > end_time:
            continue

        waste = metrics.waste_signals
        for key in totals:
            totals[key] += waste.get(key, 0)

        # Estimate history bloat from tokens saved
        if metrics.tokens_input_before > metrics.tokens_input_after:
            tokens_saved = metrics.tokens_input_before - metrics.tokens_input_after
            # Subtract known waste types. "reread" is excluded: it measures
            # over-compression cost (content the agent re-fetched), not
            # waste removed by compression, so it doesn't explain any part
            # of tokens_saved. "reread_compressed" is a subset of "reread"
            # (#899) and is excluded for the same reason — counting it would
            # also double-subtract.
            known_waste = sum(
                v for k, v in waste.items() if k not in ("reread", "reread_compressed")
            )
            history_bloat = max(0, tokens_saved - known_waste)
            totals["history_bloat"] += history_bloat

    # Find max for percentage calculation
    max_val = max(totals.values()) if totals.values() else 1

    labels = {
        "json_bloat": "Tool JSON Bloat",
        "html_noise": "HTML Noise",
        "base64": "Base64 Blobs",
        "whitespace": "Whitespace",
        "dynamic_date": "Dynamic Dates",
        "reread": "Re-served Tool Results",
        "reread_compressed": "Re-served After Compression",
        "history_bloat": "History Bloat",
    }

    histogram = []
    for key, tokens in sorted(totals.items(), key=lambda x: x[1], reverse=True):
        percentage = (tokens / max_val * 100) if max_val > 0 else 0
        histogram.append(
            {
                "label": labels.get(key, key),
                "tokens": tokens,
                "percentage": percentage,
            }
        )

    return histogram


def _get_top_waste_requests(
    storage: Any,
    start_time: datetime | None,
    end_time: datetime | None,
    limit: int = 10,
) -> list[dict[str, Any]]:
    """Get top requests by waste."""
    requests: list[dict[str, Any]] = []

    for metrics in storage.iter_all():
        if start_time and metrics.timestamp < start_time:
            continue
        if end_time and metrics.timestamp > end_time:
            continue

        tokens_saved = metrics.tokens_input_before - metrics.tokens_input_after

        requests.append(
            {
                "request_id": metrics.request_id,
                "model": metrics.model,
                "mode": metrics.mode,
                "tokens_before": metrics.tokens_input_before,
                "tokens_saved": tokens_saved,
                "cache_alignment": metrics.cache_alignment_score,
            }
        )

    # Sort by tokens saved (waste potential)
    requests.sort(key=lambda x: x["tokens_saved"], reverse=True)

    return requests[:limit]


def _generate_recommendations(
    stats: dict[str, Any],
    waste_histogram: list[dict[str, Any]],
    top_requests: list[dict[str, Any]],
) -> list[dict[str, str]]:
    """Generate actionable recommendations."""
    recommendations = []

    # Check cache alignment
    if stats["avg_cache_alignment"] < 50:
        recommendations.append(
            {
                "title": "Improve Cache Alignment",
                "description": "Your cache alignment score is low. Consider moving dynamic content "
                "(dates, timestamps, session IDs) out of system prompts into user messages.",
            }
        )

    # Check for tool JSON bloat
    for item in waste_histogram:
        if item["label"] == "Tool JSON Bloat" and item["tokens"] > 10000:
            recommendations.append(
                {
                    "title": "Enable Tool Output Compression",
                    "description": f"Detected {item['tokens']:,} tokens of tool JSON bloat. "
                    "Switch to 'optimize' mode and configure tool profiles to compress large tool outputs.",
                }
            )
            break

    # Check for history bloat
    for item in waste_histogram:
        if item["label"] == "History Bloat" and item["tokens"] > 50000:
            recommendations.append(
                {
                    "title": "Review Rolling Window Settings",
                    "description": f"Detected {item['tokens']:,} tokens of history bloat. "
                    "Consider reducing keep_last_turns or increasing output_buffer_tokens.",
                }
            )
            break

    # Check audit vs optimize ratio
    if stats["audit_count"] > stats["optimize_count"] * 2:
        recommendations.append(
            {
                "title": "Switch to Optimize Mode",
                "description": f"{stats['audit_count']} requests in audit mode vs {stats['optimize_count']} in optimize. "
                "Consider switching default_mode to 'optimize' to realize token savings.",
            }
        )

    # General recommendation
    if stats["total_tokens_saved"] > 0:
        recommendations.append(
            {
                "title": "Continue Monitoring",
                "description": f"You've saved {stats['total_tokens_saved']:,} tokens so far. "
                f"Estimated cost savings: {stats['estimated_savings']}. Keep up the good work!",
            }
        )
    else:
        recommendations.append(
            {
                "title": "Get Started",
                "description": "No optimizations applied yet. Try setting headroom_mode='optimize' "
                "on your next request to start seeing token savings.",
            }
        )

    return recommendations
