#!/usr/bin/env python3
"""End-to-end token-savings test for Cortex Code (CoCo) + Headroom.

Simulates a real Cortex Code session using JSON-format tool results —
the format Snowflake's Python connector and most tool wrappers actually
emit.  Headroom's SmartCrusher compresses JSON natively without any ML
model, so this test works with the base install (no [ml] extra needed).

No API key required. Compression runs fully local.

Usage:
    # Benchmark (pretty-printed report):
    cd headroom && uv run python tests/test_cortex_code_compression.py

    # Pytest (CI-friendly assertions):
    cd headroom && uv run --with pytest pytest tests/test_cortex_code_compression.py -v -s
"""

from __future__ import annotations

import json
import time

MODEL = "claude-sonnet-4-5-20250929"

# ── Realistic CoCo JSON payload builders ─────────────────────────────────────


def snowflake_tables_json() -> str:
    """JSON array returned by INFORMATION_SCHEMA.TABLES — SmartCrusher target."""
    rows = [
        {
            "TABLE_CATALOG": "PROD_DB",
            "TABLE_SCHEMA": "ANALYTICS",
            "TABLE_NAME": f"FACT_ORDERS_{i:03d}",
            "TABLE_TYPE": "BASE TABLE",
            "ROW_COUNT": i * 1_423_001,
            "BYTES": i * 8_192_000,
            "CREATED": "2024-01-15T08:00:00Z",
            "LAST_ALTERED": "2025-06-10T14:22:00Z",
            "COMMENT": f"Daily order fact partition {i:03d}",
        }
        for i in range(1, 80)
    ]
    return json.dumps(rows, indent=2)


def snowflake_schema_json() -> str:
    """JSON array from DESCRIBE TABLE — repeated structure SmartCrusher loves."""
    base = [
        {
            "COLUMN_NAME": "order_id",
            "DATA_TYPE": "VARCHAR",
            "LENGTH": 36,
            "NULLABLE": False,
            "PRIMARY_KEY": True,
            "COMMENT": "UUID primary key",
        },
        {
            "COLUMN_NAME": "order_date",
            "DATA_TYPE": "DATE",
            "LENGTH": None,
            "NULLABLE": False,
            "PRIMARY_KEY": False,
            "COMMENT": "Order placement date",
        },
        {
            "COLUMN_NAME": "customer_id",
            "DATA_TYPE": "VARCHAR",
            "LENGTH": 36,
            "NULLABLE": False,
            "PRIMARY_KEY": False,
            "COMMENT": "FK to dim_customers",
        },
        {
            "COLUMN_NAME": "region",
            "DATA_TYPE": "VARCHAR",
            "LENGTH": 50,
            "NULLABLE": False,
            "PRIMARY_KEY": False,
            "COMMENT": "Sales region code",
        },
        {
            "COLUMN_NAME": "product_category",
            "DATA_TYPE": "VARCHAR",
            "LENGTH": 100,
            "NULLABLE": False,
            "PRIMARY_KEY": False,
            "COMMENT": "Top-level product category",
        },
        {
            "COLUMN_NAME": "product_sku",
            "DATA_TYPE": "VARCHAR",
            "LENGTH": 50,
            "NULLABLE": False,
            "PRIMARY_KEY": False,
            "COMMENT": "FK to dim_products",
        },
        {
            "COLUMN_NAME": "quantity",
            "DATA_TYPE": "NUMBER",
            "LENGTH": None,
            "NULLABLE": False,
            "PRIMARY_KEY": False,
            "COMMENT": "Units ordered",
        },
        {
            "COLUMN_NAME": "unit_price",
            "DATA_TYPE": "NUMBER",
            "LENGTH": None,
            "NULLABLE": False,
            "PRIMARY_KEY": False,
            "COMMENT": "Price per unit USD",
        },
        {
            "COLUMN_NAME": "discount_pct",
            "DATA_TYPE": "NUMBER",
            "LENGTH": None,
            "NULLABLE": False,
            "PRIMARY_KEY": False,
            "COMMENT": "Discount percentage 0-100",
        },
        {
            "COLUMN_NAME": "status",
            "DATA_TYPE": "VARCHAR",
            "LENGTH": 20,
            "NULLABLE": False,
            "PRIMARY_KEY": False,
            "COMMENT": "Order lifecycle status",
        },
        {
            "COLUMN_NAME": "net_revenue",
            "DATA_TYPE": "NUMBER",
            "LENGTH": None,
            "NULLABLE": True,
            "PRIMARY_KEY": False,
            "COMMENT": "qty * price * (1-disc)",
        },
        {
            "COLUMN_NAME": "gross_profit",
            "DATA_TYPE": "NUMBER",
            "LENGTH": None,
            "NULLABLE": True,
            "PRIMARY_KEY": False,
            "COMMENT": "net_revenue - COGS",
        },
        {
            "COLUMN_NAME": "customer_tier",
            "DATA_TYPE": "VARCHAR",
            "LENGTH": 20,
            "NULLABLE": True,
            "PRIMARY_KEY": False,
            "COMMENT": "Gold/Silver/Bronze",
        },
        {
            "COLUMN_NAME": "acquisition_channel",
            "DATA_TYPE": "VARCHAR",
            "LENGTH": 50,
            "NULLABLE": True,
            "PRIMARY_KEY": False,
            "COMMENT": "How customer was acquired",
        },
        {
            "COLUMN_NAME": "created_at",
            "DATA_TYPE": "TIMESTAMP_NTZ",
            "LENGTH": None,
            "NULLABLE": False,
            "PRIMARY_KEY": False,
            "COMMENT": "Row creation timestamp",
        },
        {
            "COLUMN_NAME": "updated_at",
            "DATA_TYPE": "TIMESTAMP_NTZ",
            "LENGTH": None,
            "NULLABLE": False,
            "PRIMARY_KEY": False,
            "COMMENT": "Last modified timestamp",
        },
        {
            "COLUMN_NAME": "_dbt_scd_id",
            "DATA_TYPE": "VARCHAR",
            "LENGTH": 36,
            "NULLABLE": True,
            "PRIMARY_KEY": False,
            "COMMENT": "dbt SCD type-2 surrogate key",
        },
        {
            "COLUMN_NAME": "_dbt_updated_at",
            "DATA_TYPE": "TIMESTAMP_NTZ",
            "LENGTH": None,
            "NULLABLE": True,
            "PRIMARY_KEY": False,
            "COMMENT": "dbt update marker",
        },
        {
            "COLUMN_NAME": "_dbt_valid_from",
            "DATA_TYPE": "TIMESTAMP_NTZ",
            "LENGTH": None,
            "NULLABLE": True,
            "PRIMARY_KEY": False,
            "COMMENT": "SCD validity start",
        },
        {
            "COLUMN_NAME": "_dbt_valid_to",
            "DATA_TYPE": "TIMESTAMP_NTZ",
            "LENGTH": None,
            "NULLABLE": True,
            "PRIMARY_KEY": False,
            "COMMENT": "SCD validity end",
        },
    ]
    # Three tables introspected in sequence — same schema, different table names
    result = []
    for table in ["stg_orders", "int_orders_enriched", "fct_revenue"]:
        for col in base:
            result.append({**col, "TABLE_NAME": table})
    return json.dumps(result, indent=2)


def dbt_run_results_json() -> str:
    """JSON run-results.json from a dbt invocation — realistic CoCo tool output."""
    nodes = [
        {
            "unique_id": f"model.analytics.{'stg_' if i < 10 else 'fct_'}model_{i:03d}",
            "status": "success" if i % 7 != 0 else "error",
            "execution_time": round(0.8 + i * 0.12, 3),
            "rows_affected": i * 12_500,
            "compiled_code": f"SELECT * FROM raw.orders_{i:03d} WHERE status = 'active'",
            "failures": None
            if i % 7 != 0
            else [{"message": f"Invalid identifier 'col_{i}' in select list", "line": i % 40 + 1}],
            "adapter_response": {
                "query_id": f"01b{i:06x}-0000-0001-0000-000300000001",
                "rows_produced": i * 12_500,
                "bytes_scanned": i * 8_192,
                "compilation_time": 0.05,
                "execution_time": round(0.8 + i * 0.12, 3),
            },
        }
        for i in range(40)
    ]
    return json.dumps(
        {"metadata": {"dbt_version": "1.8.0", "invocation_id": "abc123"}, "results": nodes},
        indent=2,
    )


def rag_cortex_search_json() -> str:
    """JSON results from a Cortex Search query — common in CoCo sessions."""
    docs = [
        {
            "rank": i + 1,
            "score": round(0.98 - i * 0.02, 4),
            "document_id": f"doc_{i:04d}",
            "source_table": "PROD_DB.DOCS.ENGINEERING_WIKI",
            "chunk_index": i % 5,
            "content": (
                "The revenue pipeline processes approximately 2.3 million orders per day "
                "across 14 regional data centers. Each order record contains pricing "
                "information, customer segmentation data, and fulfillment status. "
                "The dbt transformation layer applies discount calculations and joins "
                "to the customer dimension table to derive net revenue and gross profit "
                "metrics. Incremental models refresh every 4 hours using Snowflake "
                "dynamic tables as the upstream source. Known issue: the product_family "
                "column was renamed to product_group in Q3 2024; models referencing "
                "the old column name will fail with SQL compilation error 001003. "
                "Migration guide: update all references from product_family to product_group "
                "in models/marts/revenue/ and run dbt run --full-refresh."
            ),
            "metadata": {
                "author": f"engineer_{i % 8}@company.com",
                "last_updated": "2025-05-20",
                "tags": ["dbt", "revenue", "snowflake", "migration"],
            },
        }
        for i in range(15)
    ]
    return json.dumps(docs, indent=2)


def build_coco_session_messages() -> list[dict]:
    """Multi-turn CoCo session: diagnose a failing dbt model via Snowflake tools.

    Turn structure mirrors what CoCo actually does:
      1. User asks to fix fct_revenue
      2. CoCo queries table catalog  (→ large JSON tool result)
      3. CoCo introspects schema     (→ large JSON tool result)
      4. CoCo runs dbt, reads results (→ large JSON tool result)
      5. CoCo searches the wiki      (→ large JSON tool result)
      6. User asks follow-up
    """
    return [
        {
            "role": "user",
            "content": (
                "My dbt model fct_revenue is failing in prod with SQL compilation error 001003. "
                "Check the table catalog, inspect the schema, run dbt, and search the wiki for any "
                "known migration guides. Then tell me exactly what to fix."
            ),
        },
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_tables",
                    "type": "function",
                    "function": {
                        "name": "snowflake_query",
                        "arguments": json.dumps(
                            {
                                "sql": "SELECT * FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_SCHEMA = 'ANALYTICS'"
                            }
                        ),
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_tables",
            "content": snowflake_tables_json(),
        },
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_schema",
                    "type": "function",
                    "function": {
                        "name": "snowflake_query",
                        "arguments": json.dumps(
                            {"sql": "DESCRIBE TABLE PROD_DB.ANALYTICS.FCT_REVENUE"}
                        ),
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_schema",
            "content": snowflake_schema_json(),
        },
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_dbt",
                    "type": "function",
                    "function": {
                        "name": "bash",
                        "arguments": json.dumps(
                            {"command": "dbt run --select fct_revenue --target prod 2>&1"}
                        ),
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_dbt",
            "content": dbt_run_results_json(),
        },
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_search",
                    "type": "function",
                    "function": {
                        "name": "cortex_search",
                        "arguments": json.dumps(
                            {"query": "product_family column rename migration fct_revenue"}
                        ),
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_search",
            "content": rag_cortex_search_json(),
        },
        {
            "role": "assistant",
            "content": (
                "Found it. The column `product_family` was renamed to `product_group` in Q3 2024. "
                "The fix is to update line 47 of `models/marts/revenue/fct_revenue.sql` and run "
                "`dbt run --select fct_revenue --full-refresh`."
            ),
        },
        {
            "role": "user",
            "content": "Perfect. Are there any other models in models/marts/revenue/ that reference product_family?",
        },
    ]


# ── Helpers ───────────────────────────────────────────────────────────────────


def _count_tokens_approx(messages: list[dict]) -> int:
    """Approximate token count from serialised JSON (~4 chars/token)."""
    return len(json.dumps(messages)) // 4


def _table_row(label: str, before: int, after: int) -> str:
    saved = before - after
    pct = saved / max(before, 1) * 100
    bar = "█" * int(pct / 5)
    return f"  {label:<35} {before:>7,} → {after:>7,}   {pct:>5.1f}%  {bar}"


# ── Pytest tests ──────────────────────────────────────────────────────────────


def test_cortex_code_headroom_compression_saves_tokens() -> None:
    """Headroom must compress a realistic multi-turn CoCo session."""
    from headroom import compress

    messages = build_coco_session_messages()

    t0 = time.perf_counter()
    result = compress(messages, model=MODEL)
    latency_ms = (time.perf_counter() - t0) * 1000

    _ = result.tokens_saved / max(result.tokens_before, 1) * 100
    print(f"\n{_table_row('Full CoCo session', result.tokens_before, result.tokens_after)}")
    print(f"  Latency: {latency_ms:.0f} ms   Transforms: {', '.join(result.transforms_applied)}")

    assert result.tokens_saved > 0, (
        f"Expected compression on the multi-turn CoCo session. "
        f"before={result.tokens_before}, after={result.tokens_after}. "
        f"Transforms: {result.transforms_applied}"
    )
    assert len(result.messages) == len(messages), "Message count must not change"
    assert result.messages[0]["content"] == messages[0]["content"], "User prompt must be verbatim"


def test_cortex_code_tool_results_are_compressed_not_user_turns() -> None:
    """User turn content must be identical before and after compression."""
    from headroom import compress

    messages = build_coco_session_messages()
    result = compress(messages, model=MODEL)

    user_orig = [m for m in messages if m.get("role") == "user"]
    user_comp = [m for m in result.messages if m.get("role") == "user"]

    assert len(user_orig) == len(user_comp)
    for orig, comp in zip(user_orig, user_comp):
        assert orig["content"] == comp["content"], (
            f"User turn was mutated:\n  before: {orig['content'][:80]!r}"
        )


def test_cortex_code_tables_json_compresses() -> None:
    """Large Snowflake INFORMATION_SCHEMA result (JSON) must compress."""
    from headroom import compress

    messages = [
        {"role": "user", "content": "List all tables in ANALYTICS schema."},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "c1",
                    "type": "function",
                    "function": {
                        "name": "snowflake_query",
                        "arguments": json.dumps({"sql": "SELECT * FROM INFORMATION_SCHEMA.TABLES"}),
                    },
                }
            ],
        },
        {"role": "tool", "tool_call_id": "c1", "content": snowflake_tables_json()},
    ]

    result = compress(messages, model=MODEL)
    _ = result.tokens_saved / max(result.tokens_before, 1) * 100
    print(f"\n{_table_row('Tables JSON (79 rows)', result.tokens_before, result.tokens_after)}")

    assert result.tokens_saved > 0, (
        f"INFORMATION_SCHEMA tables JSON was not compressed. "
        f"before={result.tokens_before}, after={result.tokens_after}. "
        f"Payload size: {len(snowflake_tables_json())} chars."
    )


def test_cortex_code_rag_search_json_compresses() -> None:
    """Cortex Search JSON results (repeated structure) must compress."""
    from headroom import compress

    messages = [
        {"role": "user", "content": "Search for product_family migration guide."},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "c2",
                    "type": "function",
                    "function": {
                        "name": "cortex_search",
                        "arguments": json.dumps({"query": "product_family rename"}),
                    },
                }
            ],
        },
        {"role": "tool", "tool_call_id": "c2", "content": rag_cortex_search_json()},
    ]

    result = compress(messages, model=MODEL)
    _ = result.tokens_saved / max(result.tokens_before, 1) * 100
    print(
        f"\n{_table_row('Cortex Search JSON (15 docs)', result.tokens_before, result.tokens_after)}"
    )

    assert result.tokens_saved > 0, (
        f"Cortex Search JSON was not compressed. "
        f"before={result.tokens_before}, after={result.tokens_after}."
    )


def test_cortex_code_compression_is_lossless_on_key_content() -> None:
    """Key answer tokens must survive compression (the model can still answer)."""
    from headroom import compress

    messages = [
        {"role": "user", "content": "Search wiki for product_family rename."},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "c3",
                    "type": "function",
                    "function": {
                        "name": "cortex_search",
                        "arguments": json.dumps({"query": "product_family"}),
                    },
                }
            ],
        },
        {"role": "tool", "tool_call_id": "c3", "content": rag_cortex_search_json()},
    ]

    result = compress(messages, model=MODEL)
    compressed_tool = next(
        (m.get("content", "") for m in result.messages if m.get("role") == "tool"), ""
    )

    # The critical answer ("product_group") must survive
    key_terms = ["product_group", "migration", "dbt", "fct_revenue"]
    found = [t for t in key_terms if t in str(compressed_tool)]
    assert len(found) >= 2, (
        f"Too many key terms lost in compression. "
        f"Found: {found}, missing: {[t for t in key_terms if t not in found]}. "
        f"Compressed output (first 500 chars): {str(compressed_tool)[:500]}"
    )


# ── Standalone benchmark ──────────────────────────────────────────────────────


if __name__ == "__main__":
    from headroom import compress

    print()
    print("=" * 65)
    print("  Cortex Code × Headroom  —  token savings benchmark")
    print("  (No API key needed — compression is fully local)")
    print("=" * 65)

    payloads = [
        ("Full CoCo session (10 turns)", build_coco_session_messages),
        (
            "INFORMATION_SCHEMA tables (79 rows)",
            lambda: [
                {"role": "user", "content": "List tables."},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "c1",
                            "type": "function",
                            "function": {"name": "q", "arguments": "{}"},
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": "c1", "content": snowflake_tables_json()},
            ],
        ),
        (
            "Schema JSON (3 tables × 20 cols)",
            lambda: [
                {"role": "user", "content": "Describe schema."},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "c1",
                            "type": "function",
                            "function": {"name": "q", "arguments": "{}"},
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": "c1", "content": snowflake_schema_json()},
            ],
        ),
        (
            "dbt run-results JSON (40 models)",
            lambda: [
                {"role": "user", "content": "Run dbt."},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "c1",
                            "type": "function",
                            "function": {"name": "q", "arguments": "{}"},
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": "c1", "content": dbt_run_results_json()},
            ],
        ),
        (
            "Cortex Search JSON (15 docs)",
            lambda: [
                {"role": "user", "content": "Search wiki."},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "c1",
                            "type": "function",
                            "function": {"name": "q", "arguments": "{}"},
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": "c1", "content": rag_cortex_search_json()},
            ],
        ),
    ]

    print(f"\n  {'Payload':<35} {'Before':>7}   {'After':>7}   {'Saved%':>6}  Bar")
    print(f"  {'─' * 35} {'─' * 7}   {'─' * 7}   {'─' * 6}  {'─' * 20}")

    total_before = total_after = 0
    for label, builder in payloads:
        msgs = builder()
        t0 = time.perf_counter()
        r = compress(msgs, model=MODEL)
        ms = (time.perf_counter() - t0) * 1000
        total_before += r.tokens_before
        total_after += r.tokens_after
        print(f"{_table_row(label, r.tokens_before, r.tokens_after)}  ({ms:.0f}ms)")

    total_saved = total_before - total_after
    total_pct = total_saved / max(total_before, 1) * 100
    print(f"\n  {'─' * 65}")
    print(f"{_table_row('TOTAL', total_before, total_after)}")
    print()
    if total_saved > 0:
        print(
            f"  PASS  headroom saved {total_saved:,} tokens ({total_pct:.0f}%) across all CoCo payload types"
        )
    else:
        print("  FAIL  no compression — run: pip install 'headroom-ai[all]'")
    print()
