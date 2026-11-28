"""Deterministic generator for the compression-ratio golden corpus + baseline.

Run from the repo root to (re)create ``cases.json`` and ``baseline.json``::

    python tests/fixtures/compression_ratio_golden/_generate.py

Every case is a fixed string built from seedless generators — no randomness,
no clock, no network — so the corpus is identical on every machine and the
measured ratios are a pure function of compressor code. Commit the baseline;
the gate (``tests/test_compression_ratio_gate.py``) then fails any changeset
that drifts a compressor's measured percentage outside its tolerance band,
which is exactly the "compression changed with no indicator" signal the gate
exists for. When a drift is intended, re-run this script and review the
baseline diff in the PR.

Compressors that need an optional backend (tree-sitter, trafilatura, the
Kompress ONNX model) are measured when available and recorded as
``"unavailable"`` otherwise — regenerate on a fully-provisioned host
(``pip install headroom-ai[all]``) to refresh their numbers.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO_ROOT))

from headroom.evals.runners.compression_ratio import (  # noqa: E402
    RATIO_TOLERANCE,
    run_cases,
)

OUT_DIR = Path(__file__).parent
CASES_OUT = OUT_DIR / "cases.json"
BASELINE_OUT = OUT_DIR / "baseline.json"


# ─── Corpus builders (pure functions; identical output on every host) ───────


def _json_cases() -> list[tuple[str, str]]:
    rows = [
        {
            "id": i,
            "level": "ERROR" if i % 9 == 0 else "INFO",
            "status": "failed" if i % 9 == 0 else "ok",
            "latency_ms": 40 + (i % 13),
            "msg": f"request {i} handled by worker-{i % 4}",
        }
        for i in range(200)
    ]
    dict_array = json.dumps(rows)

    dupes = json.dumps([{"event": "heartbeat", "ok": True} for _ in range(60)])

    nested = json.dumps(
        {
            "request_id": "req-9f31",
            "events": [
                {"step": i, "kind": "trace", "detail": f"stage {i} complete"} for i in range(30)
            ],
        }
    )

    small = json.dumps({"a": 1, "b": "hi"})
    return [
        ("dict_array_200", dict_array),
        ("duplicate_rows_60", dupes),
        ("nested_events_30", nested),
        ("small_object_passthrough", small),
    ]


def _log_cases() -> list[tuple[str, str]]:
    pytest_log = (
        "============================= test session starts ==============================\n"
        "platform linux -- Python 3.12.4, pytest-8.2.0\n"
        "collected 40 items\n"
        + "".join(
            f"tests/test_mod{i}.py::test_case_{j} PASSED [ {i * 10 + j}%]\n"
            for i in range(4)
            for j in range(10)
        )
        + "\n=================================== FAILURES ===================================\n"
        "tests/test_mod2.py::test_case_3 - AssertionError: expected 42, got 41\n"
        "=========================== short test summary info ============================\n"
        "FAILED tests/test_mod2.py::test_case_3\n"
        "========================= 1 failed, 39 passed in 3.12s =========================\n"
    )

    cargo_log = (
        "".join(
            f"   Compiling dep{i} v0.{i}.0 (/registry/dep{i})\n"
            f"   Compiling dep{i}-sys v0.{i}.1 (/registry/dep{i}-sys)\n"
            for i in range(60)
        )
        + "".join(
            f"warning: unused variable: `x{i}`\n"
            f"  --> src/module{i}.rs:{i + 2}:9\n"
            f"   |\n"
            f"{i + 2} |     let x{i} = {i};\n"
            f"   |         ^^^ help: if this is intentional, prefix it with an underscore: `_x{i}`\n"
            for i in range(20)
        )
        + (
            "error[E0308]: mismatched types\n"
            "  --> src/lib.rs:44:18\n"
            "    Finished dev [unoptimized + debuginfo] target(s) in 4.20s\n"
        )
    )

    app_log = "".join(
        f"2026-11-01T08:{i // 60:02d}:{i % 60:02d}.000Z INFO worker-2 heartbeat ok\n"
        for i in range(80)
    ) + ("2026-11-01T08:59:59.000Z ERROR database connection refused\n" * 5)

    short = "INFO starting\nINFO done\n"
    return [
        ("pytest_output", pytest_log),
        ("cargo_build", cargo_log),
        ("repetitive_app_log", app_log),
        ("short_log_passthrough", short),
    ]


def _search_cases() -> list[tuple[str, str]]:
    grep = "".join(
        f"src/pkg{g}/file{i}.py:{10 + i}:{' ' * (i % 3)}def handle_{i}(ctx): return ctx.{i}\n"
        for g in range(4)
        for i in range(12)
    )
    rg_ctx = "".join(
        f"src/auth/session.py-{40 + i}-some context line {i}\n"
        f"src/auth/session.py:{50 + i}:    token = refresh(token)\n"
        for i in range(20)
    )
    small = "src/main.py:42:def main():\n"
    return [
        ("grep_multifile_48", grep),
        ("ripgrep_context_20", rg_ctx),
        ("small_search_passthrough", small),
    ]


def _diff_cases() -> list[tuple[str, str]]:
    def file_block(idx: int) -> str:
        lines = [f"@@ -{10 + idx},20 +{10 + idx},20 @@"]
        lines += [f" unchanged_context_{idx}_{k}" for k in range(8)]
        lines += [f"-removed_line_{idx}_{k}" for k in range(4)]
        lines += [f"+added_line_{idx}_{k}" for k in range(4)]
        lines += [f" unchanged_context_{idx}_{k + 8}" for k in range(8)]
        return (
            f"diff --git a/src/module{idx}.py b/src/module{idx}.py\n"
            f"index {idx:06x}..{idx + 1:06x} 100644\n"
            f"--- a/src/module{idx}.py\n"
            f"+++ b/src/module{idx}.py\n" + "\n".join(lines) + "\n"
        )

    multi = "".join(file_block(i) for i in range(5))
    small = (
        "diff --git a/a.py b/a.py\n"
        "index 111..222 100644\n"
        "--- a/a.py\n"
        "+++ b/a.py\n"
        "@@ -1,3 +1,3 @@\n"
        " keep\n"
        "-old\n"
        "+new\n"
        " keep2\n"
    )
    return [
        ("multi_file_diff_5", multi),
        ("single_hunk_small", small),
    ]


def _tabular_cases() -> list[tuple[str, str]]:
    csv_rows = "".join(
        f"{i},user{i % 7},team-{i % 4},{'active' if i % 3 else 'inactive'},"
        f"{i * 7 % 100},us-east-{(i % 2) + 1},standard processing notes apply\n"
        for i in range(40)
    )
    # Two identical row blocks: denormalized exports repeat like this, and the
    # JSON-bridge dedup in SmartCrusher is the savings lever being measured.
    csv = "id,name,team,status,score,region,notes\n" + csv_rows + csv_rows
    md = "| id | name | status |\n| --- | --- | --- |\n" + "".join(
        f"| {i} | row{i} | {'ok' if i % 2 else 'bad'} |\n" for i in range(40)
    )
    tiny = "a,b\n1,2\n"
    return [
        ("csv_80_rows", csv),
        ("markdown_table_40", md),
        ("tiny_csv_passthrough", tiny),
    ]


def _config_cases() -> list[tuple[str, str]]:
    yaml = (
        "# service configuration\n"
        "# maintained by the platform team — do not edit by hand\n\n"
        "service:\n"
        "  name: api-gateway\n"
        "  replicas: 3\n\n"
        "# database settings\n"
        "# every host entry maps one logical shard\n\n"
        "database:\n"
        + "".join(
            f"  # shard {i} endpoint\n  host_{i}: db-{i}.internal\n  port_{i}: {5432 + i}\n\n"
            for i in range(12)
        )
        + "# observability\nlogging:\n  level: info\n  format: json\n"
    )
    toml = "".join(
        f'[[package]]\nname = "crate-{i}"\nversion = "1.{i}.0"\n'
        f'source = "registry+https://github.com/rust-lang/crates.io-index"\n\n'
        for i in range(15)
    )
    ini = "".join(
        f"; settings block {i}\n[section_{i}]\nkey_a_{i} = value_{i}\nkey_b_{i} = {i * 3}\n"
        for i in range(12)
    )
    return [
        ("yaml_service_config", yaml),
        ("toml_lockfile_packages", toml),
        ("ini_sections", ini),
    ]


def _prose_cases() -> list[tuple[str, str]]:
    sentences = [
        "The deployment completed successfully at 03:00 UTC.",
        "All health checks passed within the expected window.",
        "The deployment completed successfully at 03:00 UTC.",
        "No customer-facing errors were detected during the rollout.",
        "The deployment completed successfully at 03:00 UTC.",
        "Cache hit rates remained above 98 percent throughout.",
        "All health checks passed within the expected window.",
        "The on-call engineer acknowledged the completion notice.",
        "The deployment completed successfully at 03:00 UTC.",
        "Database migration 0042 applied cleanly in 1.2 seconds.",
        "All health checks passed within the expected window.",
        "The deployment completed successfully at 03:00 UTC.",
        "Rollback procedures were not required at any stage.",
        "Cache hit rates remained above 98 percent throughout.",
        "The deployment completed successfully at 03:00 UTC.",
        "Post-deploy latency p99 measured 142ms, within SLO.",
        "All health checks passed within the expected window.",
        "The deployment completed successfully at 03:00 UTC.",
        "The incident channel was closed after thirty minutes of quiet.",
        "Cache hit rates remained above 98 percent throughout.",
    ]
    prose = "\n".join(sentences)
    short = "Compression reduces tokens. That is all."
    return [
        ("prose_redundant_20", prose),
        ("short_prose_passthrough", short),
    ]


def _code_cases() -> list[tuple[str, str]]:
    py = (
        '"""Module docstring describing the data pipeline utilities."""\n'
        "import os\nimport sys\nfrom dataclasses import dataclass\nfrom typing import Any\n\n"
        "GLOBAL_TIMEOUT = 30\n\n"
        + "".join(
            f"def transform_{i}(payload: dict[str, Any]) -> dict[str, Any]:\n"
            f'    """Transform payload stage {i} of the pipeline."""\n'
            f"    result = {{}}\n"
            f"    for key, value in payload.items():\n"
            f"        if isinstance(value, str):\n"
            f"            result[key] = value.strip().lower()\n"
            f"        elif isinstance(value, (int, float)):\n"
            f"            result[key] = value * {i + 1}\n"
            f"        else:\n"
            f"            result[key] = value\n"
            f"    return result\n\n"
            for i in range(8)
        )
        + "@dataclass\nclass Pipeline:\n"
        '    """Pipeline orchestrator."""\n'
        "    name: str\n    stages: int = 8\n\n"
        "    def run(self, payload: dict[str, Any]) -> dict[str, Any]:\n"
        "        for i in range(self.stages):\n"
        "            payload = transform_0(payload)\n"
        "        return payload\n"
    )
    ts = (
        "interface Record { id: number; name: string; tags: string[]; }\n\n"
        + "".join(
            f"export function process{i}(rec: Record): Record {{\n"
            f"  const updated = {{ ...rec, name: rec.name.trim() }};\n"
            f"  updated.tags = updated.tags.filter((t) => t.length > {i});\n"
            f"  return updated;\n"
            f"}}\n\n"
            for i in range(8)
        )
        + "export const registry = [process0, process1, process2];\n"
    )
    return [
        ("python_module", py),
        ("typescript_module", ts),
    ]


def _html_cases() -> list[tuple[str, str]]:
    article = (
        "<!DOCTYPE html>\n<html>\n<head>\n<title>Quarterly Platform Report</title>\n"
        "<script>analytics.track('pageview');</script>\n"
        "<style>body { margin: 0; }</style>\n</head>\n<body>\n"
        "<nav><a href='/'>Home</a><a href='/about'>About</a><a href='/contact'>Contact</a></nav>\n"
        "<aside class='ads'><div>Buy our product!</div></aside>\n"
        "<article>\n<h1>Quarterly Platform Report</h1>\n"
        + "".join(
            f"<p>Section {i} summary: request volume grew {i * 3} percent while "
            f"error rates held flat at 0.0{i} percent. The team shipped {2 + i} "
            f"improvements to the compression pipeline this quarter.</p>\n"
            for i in range(12)
        )
        + "</article>\n"
        "<footer><p>Copyright 2026 Example Corp. All rights reserved.</p>"
        "<a href='/privacy'>Privacy</a></footer>\n"
        "<script>consent.init();</script>\n</body>\n</html>\n"
    )
    docs_page = (
        "<html><head><title>API Reference</title></head><body>\n"
        "<header>Docs | Guides | API | Blog</header>\n"
        "<main>\n<h1>compress()</h1>\n"
        + "".join(
            f"<h2>Parameter {i}</h2>\n<p>Parameter {i} controls behavior "
            f"{i}: it accepts values between 0 and {i * 10} and defaults "
            f"to {i}. See the guide for examples.</p>\n"
            f"<pre><code>compress(messages, param_{i}={i})</code></pre>\n"
            for i in range(8)
        )
        + "</main>\n<footer>Powered by docs framework</footer>\n"
        "</body></html>\n"
    )
    return [
        ("article_page", article),
        ("docs_page", docs_page),
    ]


def _kompress_cases() -> list[tuple[str, str]]:
    # Long-form prose (>= min_input_words) exercising the ML keep-mask.
    para = (
        "The compression pipeline evaluates each inbound payload to decide "
        "whether contextual information can be removed without harming the "
        "model's ability to answer correctly. Structured data such as JSON "
        "arrays and log files compresses extremely well because repeated "
        "fields carry little new information per row. Prose is harder: every "
        "sentence can carry a distinct fact, so aggressive removal risks "
        "dropping the one detail the user actually needed. The pipeline "
        "therefore routes content by type and applies the strongest "
        "compressor that still preserves answer-critical evidence.\n\n"
    )
    prose_doc = para * 4
    meeting = (
        "Action items from the platform review: Priya will draft the cache "
        "invalidation proposal by Thursday. Marcus owns the load-test plan "
        "for the proxy rewrite and will share results next sprint. The team "
        "agreed the rollout gates need explicit compression metrics before "
        "the next release, since savings drifted twice this quarter without "
        "anyone noticing until dashboards were audited manually. Dana will "
        "file the tracking issue and seed it with the baseline numbers from "
        "the October benchmark run so reviewers can see the expected bands.\n\n"
    ) * 3
    return [
        ("prose_pipeline_doc", prose_doc),
        ("meeting_notes", meeting),
    ]


def _cases() -> list[dict]:
    cases: list[dict] = []

    for case_id, content in _json_cases():
        cases.append({"id": case_id, "compressor": "smart_crusher", "content": content})
    for case_id, content in _log_cases():
        cases.append({"id": case_id, "compressor": "log", "content": content})
    for case_id, content in _search_cases():
        cases.append({"id": case_id, "compressor": "search", "content": content})
    for case_id, content in _diff_cases():
        cases.append({"id": case_id, "compressor": "diff", "content": content})
    for case_id, content in _tabular_cases():
        cases.append({"id": case_id, "compressor": "tabular", "content": content})
    for case_id, content in _config_cases():
        cases.append({"id": case_id, "compressor": "config", "content": content})
    for case_id, content in _prose_cases():
        cases.append({"id": case_id, "compressor": "text_crusher", "content": content})
    for case_id, content in _code_cases():
        cases.append(
            {
                "id": case_id,
                "compressor": "code_aware",
                "content": content,
                "params": {"language": "python" if case_id.startswith("python") else "typescript"},
            }
        )
    for case_id, content in _html_cases():
        cases.append({"id": case_id, "compressor": "html", "content": content})
    for case_id, content in _kompress_cases():
        cases.append({"id": case_id, "compressor": "kompress", "content": content})

    return cases


def main() -> int:
    cases = _cases()
    measurements = run_cases(cases)

    baseline: dict = {
        "version": 1,
        "generated_by": "tests/fixtures/compression_ratio_golden/_generate.py",
        "metric_notes": {
            "byte_ratio": "1 - bytes_after/bytes_before (UTF-8)",
            "token_ratio": "1 - tokens_after/tokens_before (EstimatingTokenCounter)",
        },
        "tolerances": RATIO_TOLERANCE,
        "cases": {},
        "aggregates": {},
    }

    per_compressor: dict[str, list[float]] = {}
    unavailable = 0
    for case in cases:
        key = f"{case['compressor']}/{case['id']}"
        m = measurements[key]
        if "unavailable" in m:
            unavailable += 1
            baseline["cases"][key] = {
                "unavailable": m["unavailable"],
                "tolerance": RATIO_TOLERANCE.get(case["compressor"], 0.02),
            }
            continue
        baseline["cases"][key] = {
            "bytes_before": m["bytes_before"],
            "bytes_after": m["bytes_after"],
            "tokens_before": m["tokens_before"],
            "tokens_after": m["tokens_after"],
            "byte_ratio": round(m["byte_ratio"], 6),
            "token_ratio": round(m["token_ratio"], 6),
            "modified": m["modified"],
            "tolerance": RATIO_TOLERANCE.get(case["compressor"], 0.02),
        }
        per_compressor.setdefault(case["compressor"], []).append(m["byte_ratio"])

    baseline["aggregates"] = {
        name: round(sum(ratios) / len(ratios), 6) for name, ratios in sorted(per_compressor.items())
    }

    CASES_OUT.write_text(json.dumps(cases, indent=1) + "\n")
    BASELINE_OUT.write_text(json.dumps(baseline, indent=1, sort_keys=False) + "\n")

    measured = sum(1 for v in baseline["cases"].values() if "unavailable" not in v)
    print(f"wrote {len(cases)} cases -> {CASES_OUT.relative_to(_REPO_ROOT)}")
    print(f"wrote baseline -> {BASELINE_OUT.relative_to(_REPO_ROOT)}")
    print(f"  measured: {measured} case(s); unavailable: {unavailable}")
    for name, agg in baseline["aggregates"].items():
        print(f"  {name}: mean byte_ratio {agg:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
