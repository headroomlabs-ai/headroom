"""Tests for deterministic retention probes over recorded compression events."""

import json

from headroom.evals.session_probes import (
    DIMENSIONS,
    DimensionTally,
    extract_probe_targets,
    probe_event,
    render_report,
    run_probes,
)

ORIGINAL_TOOL_TEXT = (
    "Deploy summary\n"
    "retry_limit: 3\n"
    "port=8787\n"
    "see headroom/proxy/server.py and https://example.com/build/42\n"
    "commit d293b77ab12\n"
    "ModuleNotFoundError: No module named 'left_pad'\n"
)


def _record(compressed_content, tokens_before=100, tokens_after=40, transforms=None):
    return {
        "request_id": "req-1",
        "tokens_before": tokens_before,
        "tokens_after": tokens_after,
        "transforms_applied": transforms or ["smart_crusher"],
        "original_messages": [
            {
                "role": "user",
                "content": [{"type": "tool_result", "content": ORIGINAL_TOOL_TEXT}],
            }
        ],
        "compressed_messages": [
            {
                "role": "user",
                "content": [{"type": "tool_result", "content": compressed_content}],
            }
        ],
    }


class TestExtractProbeTargets:
    def test_extracts_contextual_numerics(self):
        targets = extract_probe_targets("retry_limit: 3 and port=8787")

        assert "retry_limit: 3" in targets["numerics"]
        assert "port=8787" in targets["numerics"]

    def test_extracts_json_quoted_numerics(self):
        targets = extract_probe_targets('{"latency_ms": 12, "status": 200}')

        assert 'latency_ms": 12' in targets["numerics"]
        assert 'status": 200' in targets["numerics"]

    def test_extracts_artifacts(self):
        text = (
            "path headroom/proxy/server.py url https://example.com/build/42 "
            "hash d293b77ab12 id 123e4567-e89b-42d3-a456-426614174000"
        )
        targets = extract_probe_targets(text)

        assert "headroom/proxy/server.py" in targets["artifacts"]
        assert "https://example.com/build/42" in targets["artifacts"]
        assert "d293b77ab12" in targets["artifacts"]
        assert "123e4567-e89b-42d3-a456-426614174000" in targets["artifacts"]

    def test_bare_decimal_runs_are_not_artifacts(self):
        targets = extract_probe_targets("run id 27344471690 at ts 1765449600")

        assert "27344471690" not in targets["artifacts"]
        assert "1765449600" not in targets["artifacts"]

    def test_extracts_error_lines(self):
        targets = extract_probe_targets("all good\nModuleNotFoundError: No module named 'x'\n")

        assert any("ModuleNotFoundError" in value for value in targets["errors"])
        assert all("all good" not in value for value in targets["errors"])

    def test_openai_role_tool_messages_supported(self):
        record = _record("anything")
        record["original_messages"] = [{"role": "tool", "content": ORIGINAL_TOOL_TEXT}]

        result = probe_event(record)

        assert result is not None
        assert result.dims["numerics"].total > 0


class TestProbeEvent:
    def test_everything_retained_when_content_survives(self):
        result = probe_event(_record(ORIGINAL_TOOL_TEXT))

        assert result is not None
        for name in DIMENSIONS:
            tally = result.dims[name]
            assert tally.total > 0
            assert tally.retained == tally.total
            assert tally.lost == 0

    def test_recoverable_when_ccr_marker_present(self):
        result = probe_event(_record("[60 items compressed to 5. Retrieve more: hash=abc123def]"))

        assert result is not None
        numerics = result.dims["numerics"]
        assert numerics.total > 0
        assert numerics.retained == 0
        assert numerics.recoverable == numerics.total
        assert numerics.lost == 0

    def test_lost_when_dropped_without_marker(self):
        result = probe_event(_record("everything went fine"))

        assert result is not None
        for name in DIMENSIONS:
            tally = result.dims[name]
            assert tally.retained == 0
            assert tally.recoverable == 0
            assert tally.lost == tally.total

    def test_ratio_and_transforms(self):
        result = probe_event(_record("x", tokens_before=200, tokens_after=50))

        assert result is not None
        assert result.ratio == 0.25
        assert result.transforms == ["smart_crusher"]

    def test_numerics_retained_across_format_change(self):
        record = _record("| latency_ms | status |\n| 12 | 200 |")
        record["original_messages"] = [
            {"role": "tool", "content": '{"latency_ms": 12, "status": 200}'}
        ]

        result = probe_event(record)

        assert result is not None
        numerics = result.dims["numerics"]
        assert numerics.total > 0
        assert numerics.retained == numerics.total

    def test_numerics_lost_when_value_dropped_after_format_change(self):
        record = _record("| latency_ms |\n| 99 |")
        record["original_messages"] = [{"role": "tool", "content": '{"latency_ms": 12}'}]

        result = probe_event(record)

        assert result is not None
        assert result.dims["numerics"].lost == result.dims["numerics"].total

    def test_error_line_survives_punctuation_rewrite(self):
        record = _record("msg=ModuleNotFoundError: No module named 'left_pad'")
        record["original_messages"] = [
            {
                "role": "tool",
                "content": '{"msg": "ModuleNotFoundError: No module named \'left_pad\'"}',
            }
        ]

        result = probe_event(record)

        assert result is not None
        errors = result.dims["errors"]
        assert errors.total > 0
        assert errors.retained == errors.total

    def test_error_line_survives_json_to_csv_compaction(self):
        record = _record("error,ModuleNotFoundError: No module named 'left_pad',src/imports.py")
        record["original_messages"] = [
            {
                "role": "tool",
                "content": '{"msg": "ModuleNotFoundError: No module named \'left_pad\'"}',
            }
        ]

        result = probe_event(record)

        assert result is not None
        errors = result.dims["errors"]
        assert errors.total > 0
        assert errors.retained == errors.total

    def test_rejects_unscorable_records(self):
        assert probe_event({"tokens_before": 0, "tokens_after": 0}) is None
        assert probe_event({"tokens_before": "x", "tokens_after": 5}) is None
        assert probe_event({}) is None


class TestRunProbesAndReport:
    def test_run_probes_reads_jsonl_and_skips_garbage(self, tmp_path):
        records = [
            _record(ORIGINAL_TOOL_TEXT),
            _record("gone", tokens_before=100, tokens_after=80),
        ]
        lines = [json.dumps(record) for record in records]
        lines.insert(1, "{not valid json")
        lines.append(json.dumps(["not", "a", "dict"]))
        (tmp_path / "compression-events-1.jsonl").write_text(
            "\n".join(lines) + "\n", encoding="utf-8"
        )

        report = run_probes(tmp_path)

        assert len(report.events) == 2
        assert report.skipped_lines == 2

    def test_aggregate_sums_dimensions(self, tmp_path):
        path = tmp_path / "compression-events-1.jsonl"
        path.write_text(
            json.dumps(_record(ORIGINAL_TOOL_TEXT)) + "\n" + json.dumps(_record("gone")) + "\n",
            encoding="utf-8",
        )

        report = run_probes(tmp_path)
        aggregate = report.aggregate()

        for name in DIMENSIONS:
            single = report.events[0].dims[name]
            assert aggregate[name].total == single.total * 2
            assert aggregate[name].retained == single.total
            assert aggregate[name].lost == single.total

    def test_bucketing_and_transform_grouping(self, tmp_path):
        path = tmp_path / "compression-events-1.jsonl"
        path.write_text(
            json.dumps(_record("gone", tokens_before=100, tokens_after=10)) + "\n",
            encoding="utf-8",
        )

        report = run_probes(tmp_path)

        buckets = report.by_ratio_bucket()
        assert buckets["0.00-0.25"]["numerics"].total > 0
        assert buckets["0.75-1.00"]["numerics"].total == 0
        assert "smart_crusher" in report.by_transform()

    def test_inflated_events_land_in_inflation_bucket(self, tmp_path):
        record = _record("gone", tokens_before=100, tokens_after=130)
        path = tmp_path / "compression-events-1.jsonl"
        path.write_text(json.dumps(record) + "\n", encoding="utf-8")

        report = run_probes(tmp_path)

        buckets = report.by_ratio_bucket()
        assert buckets["1.00+ (inflated)"]["numerics"].total > 0
        assert all(
            dims["numerics"].total == 0
            for label, dims in buckets.items()
            if label != "1.00+ (inflated)"
        )

    def test_transform_grouping_dedupes_repeated_markers(self, tmp_path):
        record = _record("gone", transforms=["smart_crusher", "smart_crusher"])
        path = tmp_path / "compression-events-1.jsonl"
        path.write_text(json.dumps(record) + "\n", encoding="utf-8")

        report = run_probes(tmp_path)

        per_dim = report.by_transform()["smart_crusher"]
        assert per_dim["numerics"].total == report.events[0].dims["numerics"].total

    def test_to_dict_and_render(self, tmp_path):
        path = tmp_path / "compression-events-1.jsonl"
        path.write_text(json.dumps(_record("gone")) + "\n", encoding="utf-8")

        report = run_probes(tmp_path)
        payload = report.to_dict()
        rendered = render_report(report)

        assert payload["aggregate"]["numerics"]["lost"] > 0
        assert payload["events"][0]["ratio"] == 0.4
        assert "Aggregate retention" in rendered
        assert "smart_crusher" in rendered

    def test_dimension_tally_lost_property(self):
        tally = DimensionTally(total=5, retained=2, recoverable=1)

        assert tally.lost == 2
        assert tally.to_dict() == {"total": 5, "retained": 2, "recoverable": 1, "lost": 2}
