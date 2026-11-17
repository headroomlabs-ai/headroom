"""Realistic agent scenarios with verbose tool outputs for benchmarking Gemini on Vertex AI.

Simulates actual tool-use patterns in production developer and SRE agents:
1. SRE Incident Investigation (Microservice logs, metrics, k8s events)
2. Codebase Security & PR Audit (File tree, code search matches, git diff)
3. Enterprise Analytics & Anomaly Detection (500 database records with anomalies)
4. Multi-turn RAG & Architecture Synthesis (Complex system docs and API schemas)
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass, field
from typing import Any


@dataclass
class BenchmarkScenario:
    """A realistic multi-turn agent scenario."""

    name: str
    category: str
    description: str
    system_prompt: str
    user_query: str
    tool_outputs: list[dict[str, Any]]
    expected_facts: list[str] = field(default_factory=list)
    expected_anomalies: list[str] = field(default_factory=list)

    def total_raw_chars(self) -> int:
        return sum(len(json.dumps(t["result"])) for t in self.tool_outputs)


# ----------------------------------------------------------------------------
# 1. SRE Incident Investigation
# ----------------------------------------------------------------------------


def generate_sre_logs(num_entries: int = 350) -> dict[str, Any]:
    """Generate realistic microservice logs with a buried critical cascade failure."""
    random.seed(42)
    services = [
        "api-gateway",
        "auth-service",
        "payment-service",
        "order-service",
        "notification-service",
    ]
    log_levels = ["INFO", "INFO", "INFO", "INFO", "WARN", "DEBUG"]

    entries: list[dict[str, Any]] = []

    for i in range(num_entries):
        svc = random.choice(services)
        lvl = random.choice(log_levels)
        ts = f"2026-08-31T01:{i // 60:02d}:{i % 60:02d}.{random.randint(100, 999)}Z"
        trace_id = hashlib.md5(f"trace-{i}".encode()).hexdigest()[:16]

        if i == 142:
            lvl = "ERROR"
            svc = "payment-service"
            msg = "FATAL: Connection pool exhausted for Postgres cluster pg-primary.db.internal:5432. Active: 100/100, Queue: 580 waiters, Timeout: 30000ms."
        elif i == 143:
            lvl = "ERROR"
            svc = "payment-service"
            msg = "Unhandled exception: ConnectionPoolExhaustedError: Unable to acquire connection within 30000ms at PaymentProcessor.execute (payment_gateway.py:314)"
        elif i == 188:
            lvl = "WARN"
            svc = "api-gateway"
            msg = "Upstream 504 Gateway Timeout from http://payment-service:8080/v2/charge (latency: 30004ms). Circuit breaker trip threshold reached (failure_rate=82%)."
        elif lvl == "WARN":
            msg = f"High thread pool utilization on {svc} (88% threshold reached)"
        else:
            msg = f"Handled HTTP request 200 OK for /{svc}/v1/healthcheck [latency={random.randint(2, 45)}ms]"

        entries.append(
            {
                "timestamp": ts,
                "level": lvl,
                "service": svc,
                "message": msg,
                "trace_id": trace_id,
                "host": f"k8s-pod-{svc}-prod-{random.randint(1, 4)}",
                "region": "us-central1",
            }
        )

    return {
        "tool": "kubernetes_log_query",
        "result": {
            "query": "namespace:prod status:>=400 OR level:WARN OR level:ERROR",
            "total_lines_scanned": num_entries * 12,
            "returned_entries_count": len(entries),
            "entries": entries,
        },
    }


def generate_sre_metrics() -> dict[str, Any]:
    """Generate timeseries metric query results."""
    return {
        "tool": "prometheus_metrics_query",
        "result": {
            "query": "sum by (service, status_code) (rate(http_requests_total[5m]))",
            "status": "success",
            "data": {
                "resultType": "matrix",
                "result": [
                    {
                        "metric": {"service": "payment-service", "status_code": "500"},
                        "values": [
                            [1725062400 + i * 60, str(random.randint(45, 120))] for i in range(30)
                        ],
                    },
                    {
                        "metric": {"service": "payment-service", "status_code": "200"},
                        "values": [
                            [1725062400 + i * 60, str(random.randint(10, 25))] for i in range(30)
                        ],
                    },
                    {
                        "metric": {"service": "api-gateway", "status_code": "504"},
                        "values": [
                            [1725062400 + i * 60, str(random.randint(80, 200))] for i in range(30)
                        ],
                    },
                ],
            },
        },
    }


def create_sre_scenario() -> BenchmarkScenario:
    return BenchmarkScenario(
        name="SRE Incident Root Cause Analysis",
        category="DevOps & Reliability",
        description="Analyze 350+ multi-service container logs and Prometheus metrics to find cascading incident root cause.",
        system_prompt="You are an expert Google Cloud SRE agent diagnosing a production outage. Identify the root cause service, exact error condition, and recommend immediate remediation steps.",
        user_query="We are experiencing a surge of 500/504 errors on user checkout. Inspect the attached Kubernetes logs and Prometheus metrics to determine the exact root cause and affected database cluster.",
        tool_outputs=[generate_sre_logs(350), generate_sre_metrics()],
        expected_facts=[
            "payment-service",
            "pg-primary.db.internal",
            "connection pool",
            "5432",
        ],
        expected_anomalies=["connectionpoolexhaustederror", "circuit breaker"],
    )


# ----------------------------------------------------------------------------
# 2. Codebase Security & PR Audit
# ----------------------------------------------------------------------------


def generate_code_search_results(num_files: int = 40) -> dict[str, Any]:
    """Generate realistic codebase search output with embedded JWT auth vulnerability."""
    random.seed(42)
    files = []
    for i in range(num_files):
        if i == 7:
            files.append(
                {
                    "path": "services/auth/jwt_validator.py",
                    "language": "python",
                    "size_bytes": 4210,
                    "matches": [
                        {
                            "line_number": 42,
                            "line_content": "def verify_token(token: str, secret_key: str):",
                            "context": [
                                "    # Insecure fallback: skip verification when verify=False is passed in header",
                                "    header = jwt.get_unverified_header(token)",
                                "    if header.get('alg') == 'none' or header.get('bypass_auth'):",
                                "        return jwt.decode(token, options={'verify_signature': False})",
                                "    return jwt.decode(token, secret_key, algorithms=['HS256', 'RS256'])",
                            ],
                        }
                    ],
                    "score": 0.98,
                }
            )
        else:
            files.append(
                {
                    "path": f"services/core/handler_module_{i}.py",
                    "language": "python",
                    "size_bytes": random.randint(1500, 9000),
                    "matches": [
                        {
                            "line_number": random.randint(10, 200),
                            "line_content": f"    user_ctx = request.state.get('user_context_{i}')",
                            "context": [
                                "    # Standard authorization check",
                                "    if not user_ctx or not user_ctx.is_authenticated:",
                                "        raise PermissionDeniedError('Unauthorized')",
                            ],
                        }
                    ],
                    "score": round(random.uniform(0.3, 0.7), 2),
                }
            )

    return {
        "tool": "ast_code_search",
        "result": {
            "query": "jwt.decode OR verify_signature: False",
            "repository": "enterprise-agent-platform/api-gateway",
            "total_matches": len(files),
            "files": files,
        },
    }


def generate_git_diff() -> dict[str, Any]:
    """Generate git diff showing an authentication bypass regression."""
    diff_text = """
diff --git a/services/auth/jwt_validator.py b/services/auth/jwt_validator.py
index a1b2c3d..e4f5g6h 100644
--- a/services/auth/jwt_validator.py
+++ b/services/auth/jwt_validator.py
@@ -39,7 +39,10 @@ class JWTValidator:
     def verify_token(self, token: str) -> dict:
-        return jwt.decode(token, self.secret_key, algorithms=['RS256'])
+        header = jwt.get_unverified_header(token)
+        if header.get('alg') == 'none' or header.get('bypass_auth'):
+            # TODO: Remove testing bypass before production release
+            return jwt.decode(token, options={'verify_signature': False})
+        return jwt.decode(token, self.secret_key, algorithms=['HS256', 'RS256'])
"""
    return {
        "tool": "git_diff",
        "result": {
            "base_commit": "main",
            "target_commit": "feature/auth-speedup",
            "files_changed": 1,
            "insertions": 5,
            "deletions": 1,
            "diff": diff_text.strip(),
        },
    }


def create_security_audit_scenario() -> BenchmarkScenario:
    return BenchmarkScenario(
        name="Codebase Security Audit & PR Review",
        category="Application Security",
        description="Inspect code search results across 40 files and a git diff to detect an authentication signature bypass vulnerability.",
        system_prompt="You are a principal software engineer conducting a defensive code review on a pull request. Identify security flaws, cite the exact file and lines, describe the security risk, and propose a secure remediation.",
        user_query="Please audit the proposed changes in PR #482 against our codebase search results. Are there any critical security vulnerabilities or auth bypasses introduced? Propose a secure remediation.",
        tool_outputs=[generate_code_search_results(40), generate_git_diff()],
        expected_facts=["jwt_validator.py", "verify_signature", "alg", "none"],
        expected_anomalies=["bypass", "signature"],
    )


# ----------------------------------------------------------------------------
# 3. Enterprise Database Analytics & Outlier Detection
# ----------------------------------------------------------------------------


def generate_database_records(num_rows: int = 500) -> dict[str, Any]:
    """Generate 500 rows of user financial/usage metrics with high-value anomalies."""
    random.seed(42)
    tiers = ["free", "starter", "pro", "enterprise"]
    statuses = ["active", "active", "active", "pending", "churned"]
    countries = ["US", "DE", "GB", "JP", "FR", "AU", "SG"]

    rows: list[dict[str, Any]] = []
    for i in range(num_rows):
        user_id = f"usr_{10000 + i}"
        tier = random.choice(tiers)
        status = random.choice(statuses)
        country = random.choice(countries)
        api_calls = random.randint(100, 25000)
        spend = round(random.uniform(10.0, 850.0), 2)

        if i == 87:
            user_id = "usr_SUSPICIOUS_WHALE_87"
            tier = "enterprise"
            status = "flagged_review"
            spend = 148500.00
            api_calls = 9800000
        elif i == 312:
            user_id = "usr_CREDIT_FRAUD_312"
            tier = "starter"
            status = "suspended_chargeback"
            spend = 92450.75
            api_calls = 4500000

        rows.append(
            {
                "user_id": user_id,
                "tier": tier,
                "status": status,
                "country": country,
                "monthly_api_calls": api_calls,
                "total_spend_usd": spend,
                "last_active": f"2026-08-{random.randint(1, 30):02d}",
            }
        )

    return {
        "tool": "bigquery_analytics_query",
        "result": {
            "query": "SELECT user_id, tier, status, country, monthly_api_calls, total_spend_usd, last_active FROM `billing_db.monthly_usage`",
            "total_rows_scanned": num_rows * 50,
            "row_count": len(rows),
            "rows": rows,
        },
    }


def create_analytics_scenario() -> BenchmarkScenario:
    return BenchmarkScenario(
        name="Enterprise BigQuery Analytics & Outlier Detection",
        category="Data Analytics",
        description="Process 500 tabular user metrics rows to compute summary aggregations and detect anomalous/fraudulent billing records.",
        system_prompt="You are an enterprise data analyst assistant on Google Cloud. Summarize the dataset and explicitly pinpoint any extreme anomalies, billing outliers, or suspicious accounts.",
        user_query="Analyze the query output for our billing database. Find the top anomalous spenders, identify suspicious accounts, and provide summary insights.",
        tool_outputs=[generate_database_records(500)],
        expected_facts=[
            "usr_SUSPICIOUS_WHALE_87",
            "usr_CREDIT_FRAUD_312",
            "148500",
            "92450",
        ],
        expected_anomalies=["flagged_review", "suspended_chargeback"],
    )


# ----------------------------------------------------------------------------
# 4. Multi-Turn RAG & System Architecture Synthesis
# ----------------------------------------------------------------------------


def generate_rag_documents(num_chunks: int = 25) -> dict[str, Any]:
    """Generate dense technical architecture documents and API contracts."""
    random.seed(42)
    docs = []

    topics = [
        (
            "Agent Runtime Memory Bus",
            "The Agent Platform Runtime maintains session state across persistent memory buses. For models including Gemini 3.8 Flash, context tokens are billed at regional rates. Long-horizon agent sessions that execute >50 tool calls frequently accumulate over 150k context tokens, causing quadratic latency growth in multi-turn reasoning loops. Automatic prompt compression at the proxy layer reduces token footprint before entering the inference queue.",
        ),
        (
            "SmartCrusher Compression Contract",
            "SmartCrusher evaluates input JSON arrays by preserving the first N elements (schema anchor), the last N elements (temporal recency), all error/warning records, and generating compact statistical distributions for numerical columns. It achieves 70-90% token reduction with zero loss of semantic grounding.",
        ),
        (
            "Vertex AI Regional Routing",
            "Vertex AI serves Gemini 3.8 Flash on global and regional endpoints. Application Default Credentials authenticate requests via Google OAuth bearer tokens. Headroom preserves client ADC credentials and transparently forwards streaming tokens.",
        ),
        (
            "Tool Execution Idempotency",
            "All sidecar tools implementing MCP (Model Context Protocol) must declare idempotency keys. When tool outputs contain tabular payloads or JSON arrays, structured compression engines must preserve primary keys, status codes, and statistical distributions while stripping redundant schema metadata.",
        ),
    ]

    for i in range(num_chunks):
        title, content = topics[i % len(topics)]
        docs.append(
            {
                "doc_id": f"doc_platform_spec_{i + 1:03d}",
                "title": f"{title} (Revision {i + 1}.0)",
                "section": f"Section {i % 5 + 1}.{i % 3 + 1}",
                "author": "Google Cloud AI Architecture Team",
                "content": content
                + f" Additional metadata specification chunk #{i + 1} covering protocol validation rules and telemetry hooks.",
                "relevance_score": round(1.0 - (i * 0.02), 3),
            }
        )

    return {
        "tool": "enterprise_vector_search",
        "result": {
            "query": "Gemini Agent Platform context compression and SmartCrusher specification",
            "total_documents": len(docs),
            "documents": docs,
        },
    }


def create_rag_scenario() -> BenchmarkScenario:
    return BenchmarkScenario(
        name="Multi-Turn RAG & Architecture Synthesis",
        category="Knowledge Retrieval (RAG)",
        description="Synthesize architecture specifications from 25 dense vector retrieval chunks regarding agent context management.",
        system_prompt="You are an enterprise AI solutions architect. Synthesize the provided technical documentation to answer the user's architectural questions accurately.",
        user_query="Explain how context compression optimizes agent performance on Gemini 3.8 Flash and explain how the SmartCrusher algorithm operates.",
        tool_outputs=[generate_rag_documents(25)],
        expected_facts=[
            "SmartCrusher",
            "Gemini 3.8 Flash",
            "first N",
            "last N",
        ],
        expected_anomalies=["quadratic latency", "token reduction"],
    )


def get_all_scenarios() -> list[BenchmarkScenario]:
    """Return all benchmark scenarios."""
    return [
        create_sre_scenario(),
        create_security_audit_scenario(),
        create_analytics_scenario(),
        create_rag_scenario(),
    ]
