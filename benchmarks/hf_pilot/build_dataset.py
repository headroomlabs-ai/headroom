#!/usr/bin/env python
"""Build the **Headroom pilot dataset** — curated Anthropic Messages API payloads
that showcase context-compression value across the workloads Headroom is best at.

Each row is a complete ``/v1/messages`` request (``system`` + ``tools`` +
``messages``) whose final turn is a **bloated ``tool_result``** — exactly the
content Headroom compresses. Every row embeds a verifiable **needle** (a precise
id / field / function name) and the question is answerable *only* from the tool
output, so a harness can prove the answer survives compression.

Categories (what Headroom is great at):
  - ``code_agent``   : Claude-Code-style loop (read big file, grep, run tests)
  - ``json_data``    : large JSON / API / DB tool outputs to analyze
  - ``logs``         : multi-thousand-line log dumps; find the error
  - ``rag_search``   : many retrieved documents stuffed into context
  - ``agentic_loop`` : full multi-tool end-to-end trace

Output: ``benchmarks/hf_pilot/data/headroom_pilot.jsonl`` (one JSON row per line).
The complex fields (system/tools/messages) are stored as a single ``request_json``
string so the schema stays portable across the HF ``datasets`` Arrow writer.

Run:  python benchmarks/hf_pilot/build_dataset.py
"""

from __future__ import annotations

import hashlib
import json
import random
import sys
from pathlib import Path
from typing import Any

# --- repo imports -----------------------------------------------------------
_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from benchmarks.scenarios.tool_outputs import (  # noqa: E402
    generate_api_responses,
    generate_database_rows,
    generate_log_entries,
    generate_search_results,
)

OUT = Path(__file__).resolve().parent / "data" / "headroom_pilot.jsonl"
DEFAULT_MODEL = "claude-haiku-4-5-20251001"
LICENSE = "CC-BY-4.0"

# --- token counting (best-effort) -------------------------------------------
try:
    from headroom.tokenizers import get_tokenizer

    _TOK = get_tokenizer(DEFAULT_MODEL)
except Exception:  # pragma: no cover - fallback when tokenizer unavailable
    _TOK = None


def _count(system: str, messages: list[dict[str, Any]]) -> int:
    if _TOK is not None:
        try:
            return int(_TOK.count_messages(messages)) + int(_TOK.count_text(system))
        except Exception:
            pass
    blob = system + json.dumps(messages)
    return len(blob) // 4


def _tuid(rid: str, idx: int) -> str:
    """Deterministic, Anthropic-valid tool_use id."""
    return "toolu_" + hashlib.md5(f"{rid}:{idx}".encode()).hexdigest()[:20]  # noqa: S324


def _dumps(obj: Any) -> str:
    return json.dumps(obj, indent=2, ensure_ascii=False)


# --- conversation assembly --------------------------------------------------
def _single_tool_convo(
    rid: str,
    user_task: str,
    tool_name: str,
    tool_input: dict[str, Any],
    tool_result: str,
    answer_instruction: str,
) -> list[dict[str, Any]]:
    """One tool round, ending in a user turn (tool_result + explicit question)."""
    tid = _tuid(rid, 0)
    return [
        {"role": "user", "content": user_task},
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": f"Let me {tool_name.replace('_', ' ')} to investigate."},
                {"type": "tool_use", "id": tid, "name": tool_name, "input": tool_input},
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": tid, "content": tool_result},
                {"type": "text", "text": answer_instruction},
            ],
        },
    ]


def _multi_tool_convo(
    rid: str,
    user_task: str,
    rounds: list[tuple[str, dict[str, Any], str]],
    answer_instruction: str,
) -> list[dict[str, Any]]:
    """Several tool rounds; the last user turn appends the explicit question."""
    msgs: list[dict[str, Any]] = [{"role": "user", "content": user_task}]
    for i, (tool_name, tool_input, tool_result) in enumerate(rounds):
        tid = _tuid(rid, i)
        msgs.append(
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": f"Running {tool_name} (step {i + 1})."},
                    {"type": "tool_use", "id": tid, "name": tool_name, "input": tool_input},
                ],
            }
        )
        user_content: list[dict[str, Any]] = [
            {"type": "tool_result", "tool_use_id": tid, "content": tool_result}
        ]
        if i == len(rounds) - 1:
            user_content.append({"type": "text", "text": answer_instruction})
        msgs.append({"role": "user", "content": user_content})
    return msgs


def _row(
    rid: str,
    category: str,
    title: str,
    task: str,
    system: str,
    tools: list[dict[str, Any]],
    messages: list[dict[str, Any]],
    needles: list[str],
    model: str = DEFAULT_MODEL,
) -> dict[str, Any]:
    request = {"system": system, "tools": tools, "messages": messages}
    return {
        "id": rid,
        "category": category,
        "title": title,
        "task": task,
        "model": model,
        "request_json": json.dumps(request, ensure_ascii=False),
        "expected_answer_contains": needles,
        # Ground truth for the LLM-as-judge: a correct answer must convey every
        # one of these facts. Derived from the planted needles so it is exact.
        "reference_answer": "A correct answer must state these exact value(s): "
        + "; ".join(needles)
        + ".",
        "n_messages": len(messages),
        "approx_input_tokens": _count(system, messages),
        "source": "synthetic",
        "license": LICENSE,
    }


SYSTEM_AGENT = (
    "You are a precise engineering assistant working through tool outputs. "
    "Answer concisely. When asked for a specific value (an id, status, field, "
    "or function name), state that exact value explicitly in your answer."
)

# --- tool schemas (lightweight; content is what matters) --------------------
# NOTE: deliberately NOT named read_file/grep — Headroom protects the Claude-Code
# tool names {Read, Grep, Glob, Write, Edit} from compression by default (for
# prefix-cache safety). Generic names let CodeCompressor actually engage, which
# is the realistic case for agents that use custom tool names.
_T_READ = {
    "name": "open_source_file",
    "description": "Open and return the contents of a source file.",
    "input_schema": {"type": "object", "properties": {"path": {"type": "string"}}},
}
_T_GREP = {
    "name": "search_repository",
    "description": "Search the repository for a pattern.",
    "input_schema": {"type": "object", "properties": {"pattern": {"type": "string"}}},
}
_T_TESTS = {
    "name": "run_test_suite",
    "description": "Run the test suite and return output.",
    "input_schema": {"type": "object", "properties": {"path": {"type": "string"}}},
}
_T_QUERY = {
    "name": "query_records",
    "description": "Query records from a data source.",
    "input_schema": {"type": "object", "properties": {"q": {"type": "string"}}},
}
_T_LOGS = {
    "name": "fetch_logs",
    "description": "Fetch recent service logs.",
    "input_schema": {"type": "object", "properties": {"service": {"type": "string"}}},
}
_T_SEARCH = {
    "name": "search_docs",
    "description": "Search the documentation / knowledge base.",
    "input_schema": {"type": "object", "properties": {"query": {"type": "string"}}},
}


# --- synthetic code file (for code_agent) -----------------------------------
def _python_module(n_funcs: int, bug_fn: str, needle_const: str) -> str:
    lines = [
        '"""Auto-generated service module (synthetic).',
        "",
        "Implements a slice of an order-processing pipeline with pagination,",
        "validation, retry, and reporting helpers.",
        '"""',
        "",
        "import math",
        "from dataclasses import dataclass",
        "",
        f'SERVICE_TAG = "{needle_const}"',
        "",
    ]
    verbs = ["load", "validate", "transform", "aggregate", "format", "retry", "normalize", "score"]
    nouns = ["order", "invoice", "shipment", "customer", "payment", "ledger", "batch", "quote"]
    for i in range(n_funcs):
        name = f"{verbs[i % len(verbs)]}_{nouns[(i * 3) % len(nouns)]}_{i}"
        lines += [
            f"def {name}(items, limit={10 + i}):",
            f'    """Handle {name.replace("_", " ")} for a page of items."""',
            "    out = []",
            "    total = 0",
            "    for idx, item in enumerate(items):",
            "        weight = math.sqrt(abs(item.get('amount', 1)) + 1)",
            "        total += weight",
            f"        out.append({{'idx': idx, 'w': round(weight, 4), 'fn': '{name}'}})",
            "    return out[:limit], total",
            "",
        ]
    # The planted bug: off-by-one in the page slice (uses limit+1).
    lines += [
        f"def {bug_fn}(items, page=0, page_size=20):",
        '    """Return one page of items. NOTE: contains an off-by-one slice bug."""',
        "    start = page * page_size",
        "    end = start + page_size + 1  # BUG: +1 overruns into the next page",
        "    return items[start:end]",
        "",
    ]
    return "\n".join(lines)


# --- synthetic prose corpus (for rag_search -> Kompress text path) ----------
_PROSE_SUBJECTS = [
    "the ingestion pipeline", "the auth gateway", "the billing reconciler",
    "the search indexer", "the notification fan-out", "the schema registry",
    "the rate limiter", "the cache warmer", "the export worker", "the webhook relay",
]
_PROSE_CLAUSES = [
    "buffers events in a bounded queue before flushing them downstream",
    "retries idempotently with exponential backoff and a jittered ceiling",
    "emits structured spans so latency can be attributed per stage",
    "degrades to a read-only path when the primary store is unreachable",
    "validates payloads against a versioned schema before accepting them",
    "shards work by tenant id to keep noisy neighbors isolated",
    "checkpoints offsets so a restart resumes without duplicating work",
    "applies a circuit breaker that opens after consecutive timeouts",
    "compacts its write-ahead log on a cadence tuned to disk pressure",
    "reconciles drift against the source of truth every few minutes",
]
_PROSE_REASONS = [
    "because tail latency dominates the user-perceived experience",
    "so that a single slow dependency cannot stall the whole request",
    "to keep the blast radius of a bad deploy small and recoverable",
    "since the holiday surge multiplies traffic by roughly six times",
    "given that downstream quotas are enforced per minute, not per second",
    "to satisfy the data-residency constraints for EU tenants",
    "because partial failures are far more common than total outages",
    "so on-call can reason about the system from dashboards alone",
]


def _prose_doc(title: str, n_paras: int, rng: random.Random) -> str:
    paras = [f"# {title}", ""]
    for _ in range(n_paras):
        sents = []
        for _ in range(rng.randint(4, 7)):
            subj = rng.choice(_PROSE_SUBJECTS)
            clause = rng.choice(_PROSE_CLAUSES)
            reason = rng.choice(_PROSE_REASONS)
            sents.append(f"{subj.capitalize()} {clause} {reason}.")
        paras.append(" ".join(sents))
        paras.append("")
    return "\n".join(paras)


def _prose_corpus(n_docs: int, needle_doc: str, rng: random.Random) -> str:
    titles = [
        "Architecture overview", "Operational runbook", "Capacity planning notes",
        "Incident retrospective", "API design rationale", "Data model guide",
        "Deployment checklist", "Security posture", "Observability handbook",
        "Migration playbook",
    ]
    docs = []
    insert_at = rng.randint(1, max(1, n_docs - 1))
    for i in range(n_docs):
        if i == insert_at:
            docs.append(needle_doc)
        docs.append(_prose_doc(f"{rng.choice(titles)} ({i})", rng.randint(3, 6), rng))
    return "\n\n---\n\n".join(docs)


# --- category builders ------------------------------------------------------
def build_code_agent(k: int) -> list[dict[str, Any]]:
    rows = []
    for i in range(k):
        seed = 1000 + i
        random.seed(seed)
        rid = f"code_agent_{i + 1:03d}"
        bug_fn = f"paginate_{['orders', 'invoices', 'ledger', 'quotes', 'batches', 'shipments'][i % 6]}"
        needle_const = f"svc-{seed:04x}-tag"
        n_funcs = 60 + i * 18  # scale size
        code = _python_module(n_funcs, bug_fn, needle_const)
        grep_hits = "\n".join(
            f"src/pipeline.py:{40 + j * 7}:    return items[start:end]  # candidate slice"
            for j in range(30 + i * 6)
        )
        pytest_out = (
            "============================= test session starts ==============================\n"
            + "".join(f"tests/test_pipeline.py::test_case_{j} PASSED\n" for j in range(25 + i * 4))
            + f"tests/test_pipeline.py::test_{bug_fn}_boundary FAILED\n\n"
            + "=================================== FAILURES ===================================\n"
            + f"____________________________ test_{bug_fn}_boundary ____________________________\n"
            + f"    assert len(page) == 20\nE   assert 21 == 20\nE    +  where 21 = len({bug_fn}(items, page=0))\n"
            + f"\nsrc/pipeline.py: in {bug_fn}\n    end = start + page_size + 1  # BUG\n"
            + "=========================== short test summary info ============================\n"
            + f"FAILED tests/test_pipeline.py::test_{bug_fn}_boundary - assert 21 == 20\n"
            + "1 failed, %d passed\n" % (25 + i * 4)
        )
        task = (
            "A pagination test is failing — pages return one too many items. "
            "Read the module, grep for the slice, and run the tests. "
            "Which function contains the off-by-one bug, and what is SERVICE_TAG?"
        )
        rounds = [
            ("open_source_file", {"path": "src/pipeline.py"}, code),
            ("search_repository", {"pattern": "items[start:end]"}, grep_hits),
            ("run_test_suite", {"path": "tests/test_pipeline.py"}, pytest_out),
        ]
        msgs = _multi_tool_convo(
            rid,
            task,
            rounds,
            f"Name the buggy function exactly and give the SERVICE_TAG value. "
            f"(Bug is the off-by-one slice; tag is the SERVICE_TAG constant.)",
        )
        rows.append(
            _row(
                rid,
                "code_agent",
                f"Off-by-one in {bug_fn}",
                task,
                SYSTEM_AGENT,
                [_T_READ, _T_GREP, _T_TESTS],
                msgs,
                [bug_fn, needle_const],
            )
        )
    return rows


def build_json_data(k: int) -> list[dict[str, Any]]:
    rows = []
    for i in range(k):
        seed = 2000 + i
        random.seed(seed)
        rid = f"json_data_{i + 1:03d}"
        n = 120 + i * 60
        if i % 2 == 0:
            items = generate_database_rows(n, table_type="users")
            needle_id = f"U_{seed % 9000 + 1000}"
            needle_rec = {
                "user_id": needle_id,
                "email": f"{needle_id.lower()}@acme.io",
                "status": "suspended",
                "plan": "enterprise",
                "mrr_usd": 4200,
                "seats": 87,
            }
            items.insert(n // 3, needle_rec)
            task = f"Across these account records, what are the status and plan for {needle_id}?"
            needles = [needle_id, "suspended", "enterprise"]
        else:
            items = generate_api_responses(n)
            needle_id = f"ord_{seed:05d}"
            needle_rec = {
                "id": needle_id,
                "type": "order",
                "state": "refunded",
                "amount_cents": 998877,
                "currency": "USD",
                "flag": "chargeback_dispute",
            }
            items.insert(n // 4, needle_rec)
            task = f"In this API dump, what is the state and amount_cents for {needle_id}?"
            needles = [needle_id, "refunded", "998877"]
        # Multi-turn: the big needle dump lands in round 1, then two smaller
        # follow-up queries push it past the proxy's protected-recent window.
        small1 = _dumps(generate_database_rows(15, table_type="metrics"))
        small2 = _dumps(generate_api_responses(12))
        rounds = [
            ("query_records", {"q": "select * from records limit %d" % n}, _dumps(items)),
            ("query_records", {"q": "select metric, p50, p99 from service_metrics"}, small1),
            ("query_records", {"q": "select * from recent_events limit 12"}, small2),
        ]
        msgs = _multi_tool_convo(rid, task, rounds, "Answer with the exact field values requested.")
        rows.append(
            _row(
                rid,
                "json_data",
                f"Find record in {n}-row JSON dump",
                task,
                SYSTEM_AGENT,
                [_T_QUERY],
                msgs,
                needles,
            )
        )
    return rows


def build_logs(k: int) -> list[dict[str, Any]]:
    rows = []
    for i in range(k):
        seed = 3000 + i
        random.seed(seed)
        rid = f"logs_{i + 1:03d}"
        n = 400 + i * 250
        logs = generate_log_entries(n, include_errors=8, include_critical=1)
        req_id = f"req_{seed:x}f{i}a"
        crit = {
            "timestamp": "2025-01-06T04:21:09Z",
            "level": "CRITICAL",
            "service": "payments",
            "message": (
                f"Unhandled exception in payment processor; request_id={req_id}; "
                "downstream=stripe; cause=ConnectTimeout after 30s"
            ),
        }
        logs.insert(n // 2, crit)
        task = (
            f"Investigate a payment incident. Pull the payments logs ({n} lines), "
            "then check the gateway and auth services. Exactly one CRITICAL line is a "
            "payment processor exception — what is its request_id and downstream service?"
        )
        # Big needle log dump in round 1; two smaller service pulls follow it.
        small_gw = _dumps(generate_log_entries(20, include_errors=2, include_critical=0))
        small_auth = _dumps(generate_log_entries(15, include_errors=1, include_critical=0))
        rounds = [
            ("fetch_logs", {"service": "payments"}, _dumps(logs)),
            ("fetch_logs", {"service": "gateway"}, small_gw),
            ("fetch_logs", {"service": "auth"}, small_auth),
        ]
        msgs = _multi_tool_convo(
            rid, task, rounds,
            "Give the exact request_id from the CRITICAL payments line and the downstream service.",
        )
        rows.append(
            _row(
                rid,
                "logs",
                f"Find CRITICAL in {n} log lines",
                task,
                SYSTEM_AGENT,
                [_T_LOGS],
                msgs,
                [req_id, "stripe"],
            )
        )
    return rows


def build_rag_search(k: int) -> list[dict[str, Any]]:
    """RAG over PROSE documents (concatenated text) -> routes to Kompress (text path)."""
    rows = []
    for i in range(k):
        seed = 4000 + i
        rng = random.Random(seed)
        rid = f"rag_search_{i + 1:03d}"
        n_docs = 8 + i * 4
        token_val = f"HDRM-{seed:06d}-KEY"
        needle_doc = (
            "# Internal API: rotation policy for service tokens\n\n"
            "Service tokens must be rotated every 90 days without exception. The "
            f"current production signing token is {token_val}. Rotation runs "
            "automatically through the vault-rotate cron job; any manual override "
            "requires explicit SRE approval recorded in the change log. Tokens that "
            "miss a rotation window are revoked automatically by the gateway."
        )
        corpus = _prose_corpus(n_docs, needle_doc, rng)
        task = (
            "Research our token-rotation policy. Search the internal docs, then the "
            "runbooks and the onboarding guide. According to the rotation-policy doc, "
            "what is the current production signing token, and how often must service "
            "tokens be rotated?"
        )
        # Big needle corpus in round 1; two smaller searches push it out of the
        # protected-recent window so it compresses through the proxy.
        small_a = _prose_doc("Runbook: incident escalation", 2, rng)
        small_b = _prose_doc("Onboarding: developer setup", 2, rng)
        rounds = [
            ("search_docs", {"query": "service token rotation policy"}, corpus),
            ("search_docs", {"query": "incident escalation runbook"}, small_a),
            ("search_docs", {"query": "developer onboarding setup"}, small_b),
        ]
        msgs = _multi_tool_convo(
            rid, task, rounds,
            "Quote the exact token value and the rotation interval in days.",
        )
        rows.append(
            _row(
                rid,
                "rag_search",
                f"Needle in {n_docs} prose documents",
                task,
                SYSTEM_AGENT,
                [_T_SEARCH],
                msgs,
                [token_val, "90"],
            )
        )
    return rows


def build_agentic_loop(k: int) -> list[dict[str, Any]]:
    rows = []
    for i in range(k):
        seed = 5000 + i
        random.seed(seed)
        rng = random.Random(seed)
        rid = f"agentic_loop_{i + 1:03d}"
        # incident triage: logs -> records -> docs, synthesize one answer
        n_logs = 300 + i * 150
        logs = generate_log_entries(n_logs, include_errors=6, include_critical=1)
        cust = f"cust_{seed:05d}"
        logs.insert(
            n_logs // 2,
            {
                "timestamp": "2025-01-06T05:00:00Z",
                "level": "ERROR",
                "service": "billing",
                "message": f"Payment retry exhausted for customer_id={cust}; code=card_declined",
            },
        )
        rows_db = generate_database_rows(120 + i * 40, table_type="transactions")
        rows_db.insert(
            10,
            {
                "customer_id": cust,
                "last_charge": "2025-01-06T04:59:31Z",
                "status": "past_due",
                "amount_cents": 21900,
                "retries": 4,
            },
        )
        remedy = f"PLAYBOOK-{seed % 900 + 100}"
        runbook = (
            "# Runbook: card_declined past_due recovery\n\n"
            f"For accounts in card_declined and past_due state, apply {remedy}. Start "
            "the email dunning sequence immediately, set a grace period of 7 days, and "
            "do not suspend the account before the grace window expires. Escalate to "
            "the billing on-call only if a second charge attempt fails after grace."
        )
        docs_text = _prose_corpus(6 + i * 3, runbook, rng)
        task = (
            "Triage a billing incident end-to-end. Find the customer whose payment "
            "retries were exhausted in the logs, look up their transaction status, "
            "and find the matching runbook. What is the customer_id, their account "
            "status, and which playbook should we apply?"
        )
        rounds = [
            ("fetch_logs", {"service": "billing"}, _dumps(logs)),
            ("query_records", {"q": "transactions where status='past_due'"}, _dumps(rows_db)),
            ("search_docs", {"query": "card_declined past_due recovery runbook"}, docs_text),
        ]
        msgs = _multi_tool_convo(
            rid,
            task,
            rounds,
            "State the exact customer_id, their account status, and the playbook id to apply.",
        )
        rows.append(
            _row(
                rid,
                "agentic_loop",
                "Billing incident triage (logs+records+docs)",
                task,
                SYSTEM_AGENT,
                [_T_LOGS, _T_QUERY, _T_SEARCH],
                msgs,
                [cust, "past_due", remedy],
            )
        )
    return rows


def main() -> None:
    builders = [
        build_code_agent,
        build_json_data,
        build_logs,
        build_rag_search,
        build_agentic_loop,
    ]
    rows: list[dict[str, Any]] = []
    for b in builders:
        rows.extend(b(6))  # 6 rows per category -> 30 rows

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # summary
    by_cat: dict[str, list[int]] = {}
    for r in rows:
        by_cat.setdefault(r["category"], []).append(r["approx_input_tokens"])
    print(f"Wrote {len(rows)} rows -> {OUT}")
    print(f"{'category':<14} {'rows':>4} {'min_tok':>9} {'med_tok':>9} {'max_tok':>9}")
    total = 0
    for cat, toks in by_cat.items():
        toks_sorted = sorted(toks)
        med = toks_sorted[len(toks_sorted) // 2]
        total += sum(toks)
        print(f"{cat:<14} {len(toks):>4} {min(toks):>9,} {med:>9,} {max(toks):>9,}")
    print(f"{'TOTAL':<14} {len(rows):>4} input tokens across dataset: {total:,}")


if __name__ == "__main__":
    main()
