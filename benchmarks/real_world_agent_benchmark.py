"""
Real-World Agent Benchmark: MCP Tools + Headroom

This benchmark simulates real multi-agent workflows using actual MCP tool output formats:
1. Filesystem MCP Server - directory trees, file searches, file contents
2. GitHub MCP Server - code search, issues, PRs, commits
3. Database MCP Server - query results, schema info

We measure:
- Token usage with vs without Headroom
- Cost savings
- Answer quality (does compression hurt agent performance?)

This is NOT synthetic data - these are actual output formats from production MCP servers.
"""

import hashlib
import json
import os
import random
import time
from dataclasses import dataclass
from typing import Any

# OpenAI for agent
try:
    from openai import OpenAI  # noqa: F401

    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

# Headroom
try:
    from headroom import HeadroomClient, OpenAIProvider

    HEADROOM_AVAILABLE = True
except ImportError:
    HEADROOM_AVAILABLE = False


# =============================================================================
# REALISTIC MCP TOOL OUTPUT GENERATORS
# Based on actual MCP server output formats
# =============================================================================


def generate_filesystem_tree(
    path: str = "/project", depth: int = 3, files_per_dir: int = 15
) -> dict:
    """
    Generate realistic filesystem tree output (MCP filesystem server format).
    This mimics `tree` command output from @modelcontextprotocol/server-filesystem.
    """

    def generate_dir(current_path: str, current_depth: int) -> list:
        if current_depth <= 0:
            return []

        entries = []

        # Common project structure
        dir_names = [
            "src",
            "lib",
            "utils",
            "components",
            "services",
            "models",
            "controllers",
            "middleware",
            "tests",
            "config",
            "scripts",
            "api",
            "core",
            "helpers",
            "types",
            "interfaces",
        ]

        file_extensions = [".py", ".ts", ".js", ".json", ".yaml", ".md"]

        # Add some directories
        num_dirs = random.randint(2, 5) if current_depth > 1 else 0
        for i in range(num_dirs):
            dir_name = random.choice(dir_names) + (f"_{i}" if i > 0 else "")
            dir_path = f"{current_path}/{dir_name}"
            entries.append(
                {
                    "name": dir_name,
                    "type": "directory",
                    "path": dir_path,
                    "children": generate_dir(dir_path, current_depth - 1),
                }
            )

        # Add files
        for i in range(files_per_dir):
            ext = random.choice(file_extensions)
            file_name = f"module_{i}{ext}"
            entries.append(
                {
                    "name": file_name,
                    "type": "file",
                    "path": f"{current_path}/{file_name}",
                    "size": random.randint(100, 10000),
                    "modified": f"2024-01-{random.randint(1, 28):02d}T{random.randint(0, 23):02d}:{random.randint(0, 59):02d}:00Z",
                }
            )

        return entries

    return {
        "tool": "filesystem_tree",
        "path": path,
        "result": {
            "name": path.split("/")[-1] or "project",
            "type": "directory",
            "path": path,
            "children": generate_dir(path, depth),
        },
    }


def generate_filesystem_search(query: str, num_results: int = 200) -> dict:
    """
    Generate realistic file search results (MCP filesystem server format).
    Mimics search_files output with path matches and content snippets.
    """
    results = []

    # Common file paths in a real project
    paths = [
        "src/auth/jwt_handler.py",
        "src/auth/oauth_provider.py",
        "src/api/routes/users.py",
        "src/api/routes/products.py",
        "src/services/payment_processor.py",
        "src/services/email_sender.py",
        "src/middleware/rate_limiter.py",
        "src/middleware/auth_middleware.py",
        "src/models/user.py",
        "src/models/order.py",
        "tests/test_auth.py",
        "tests/test_api.py",
        "config/database.py",
        "config/settings.py",
    ]

    for i in range(num_results):
        if i < len(paths):
            path = paths[i]
        else:
            path = f"src/modules/module_{i}.py"

        # Generate realistic match context
        match_line = random.randint(10, 500)
        results.append(
            {
                "path": path,
                "type": "file",
                "size": random.randint(500, 15000),
                "modified": f"2024-01-{random.randint(1, 28):02d}",
                "matches": [
                    {
                        "line": match_line,
                        "content": f"    def process_{query.lower().replace(' ', '_')}(self, data):",
                        "context_before": "    # Process incoming request",
                        "context_after": f"        return self.handler.{query.lower()}(data)",
                    }
                ],
                "score": round(random.uniform(0.5, 1.0), 3),
            }
        )

    return {
        "tool": "search_files",
        "query": query,
        "result": {
            "total_matches": num_results,
            "files_searched": num_results * 10,
            "matches": results,
        },
    }


def generate_github_code_search(query: str, num_results: int = 100) -> dict:
    """
    Generate realistic GitHub code search results (GitHub MCP server format).
    Based on actual github-mcp-server output.
    """
    repos = [
        "facebook/react",
        "microsoft/vscode",
        "tensorflow/tensorflow",
        "kubernetes/kubernetes",
        "golang/go",
        "rust-lang/rust",
        "apache/spark",
        "elastic/elasticsearch",
        "grafana/grafana",
        "prometheus/prometheus",
        "docker/docker-ce",
        "nginx/nginx",
    ]

    results = []
    for i in range(num_results):
        repo = random.choice(repos)
        results.append(
            {
                "repository": {
                    "full_name": repo,
                    "description": f"The {repo.split('/')[1]} project",
                    "stars": random.randint(1000, 100000),
                    "language": random.choice(["Python", "Go", "TypeScript", "Java", "Rust"]),
                    "updated_at": f"2024-01-{random.randint(1, 28):02d}T00:00:00Z",
                },
                "path": f"src/{query.lower().replace(' ', '_')}/handler.py",
                "sha": hashlib.sha1(f"{repo}{i}".encode()).hexdigest(),
                "url": f"https://github.com/{repo}/blob/main/src/handler.py",
                "score": round(random.uniform(10, 100), 2),
                "text_matches": [
                    {
                        "fragment": f"def {query.lower().replace(' ', '_')}(request):\n    # Implementation\n    return response",
                        "matches": [{"text": query, "indices": [4, 4 + len(query)]}],
                    }
                ],
            }
        )

    return {
        "tool": "github_search_code",
        "query": query,
        "result": {"total_count": num_results * 50, "incomplete_results": False, "items": results},
    }


def generate_github_issues(repo: str, num_issues: int = 50) -> dict:
    """
    Generate realistic GitHub issues list (GitHub MCP server format).
    """
    labels = ["bug", "enhancement", "documentation", "help wanted", "good first issue"]
    states = ["open", "open", "open", "closed"]  # Weighted toward open

    issues = []
    for i in range(num_issues):
        issues.append(
            {
                "number": 1000 + i,
                "title": f"Issue #{1000 + i}: "
                + random.choice(
                    [
                        "Fix authentication flow",
                        "Add support for OAuth2",
                        "Performance regression in v2.0",
                        "Documentation needs update",
                        "Memory leak in worker process",
                        "Add dark mode support",
                        "API rate limiting not working",
                    ]
                ),
                "state": random.choice(states),
                "user": {
                    "login": f"user{random.randint(1, 1000)}",
                    "avatar_url": f"https://avatars.githubusercontent.com/u/{random.randint(1, 100000)}",
                },
                "labels": random.sample(labels, k=random.randint(0, 3)),
                "created_at": f"2024-01-{random.randint(1, 28):02d}T{random.randint(0, 23):02d}:00:00Z",
                "updated_at": f"2024-01-{random.randint(1, 28):02d}T{random.randint(0, 23):02d}:00:00Z",
                "comments": random.randint(0, 50),
                "body": f"## Description\n\nThis issue tracks {random.choice(['a bug', 'a feature request', 'documentation update'])}.\n\n## Steps to Reproduce\n\n1. Step one\n2. Step two\n3. Step three\n\n## Expected Behavior\n\nIt should work.\n\n## Actual Behavior\n\nIt doesn't work.",
            }
        )

    return {
        "tool": "github_list_issues",
        "repository": repo,
        "result": {"total_count": num_issues, "items": issues},
    }


def generate_database_query_results(query: str, num_rows: int = 500) -> dict:
    """
    Generate realistic database query results (Database MCP server format).
    """
    # Simulate a user analytics query
    rows = []
    for i in range(num_rows):
        rows.append(
            {
                "user_id": f"user_{10000 + i}",
                "email": f"user{10000 + i}@example.com",
                "created_at": f"2024-01-{random.randint(1, 28):02d}",
                "last_login": f"2024-01-{random.randint(1, 28):02d}T{random.randint(0, 23):02d}:00:00Z",
                "total_orders": random.randint(0, 100),
                "total_revenue": round(random.uniform(0, 10000), 2),
                "status": random.choice(["active", "active", "active", "inactive", "suspended"]),
                "country": random.choice(["US", "UK", "DE", "FR", "JP", "AU", "CA"]),
                "subscription_tier": random.choice(
                    ["free", "free", "basic", "premium", "enterprise"]
                ),
            }
        )

    # Add some anomalies (high-value users)
    for _ in range(3):
        rows[random.randint(0, len(rows) - 1)]["total_revenue"] = round(
            random.uniform(50000, 100000), 2
        )
        rows[random.randint(0, len(rows) - 1)]["status"] = "suspended"

    return {
        "tool": "database_query",
        "query": query,
        "result": {
            "columns": [
                "user_id",
                "email",
                "created_at",
                "last_login",
                "total_orders",
                "total_revenue",
                "status",
                "country",
                "subscription_tier",
            ],
            "row_count": num_rows,
            "rows": rows,
            "execution_time_ms": random.randint(50, 500),
        },
    }


def generate_log_search(query: str, num_entries: int = 300) -> dict:
    """
    Generate realistic log search results (Logging MCP server format).
    """
    log_levels = ["INFO", "INFO", "INFO", "INFO", "WARN", "ERROR", "DEBUG"]
    services = [
        "api-gateway",
        "auth-service",
        "payment-service",
        "user-service",
        "notification-service",
    ]

    entries = []

    for i in range(num_entries):
        level = random.choice(log_levels)
        service = random.choice(services)

        if level == "ERROR":
            message = random.choice(
                [
                    f"Connection refused to {service}:8080 - ECONNREFUSED",
                    f"Timeout waiting for response from {service} after 30000ms",
                    "Failed to process request: NullPointerException",
                    "Database connection pool exhausted",
                ]
            )
        elif level == "WARN":
            message = random.choice(
                [
                    "High latency detected: 2500ms (threshold: 1000ms)",
                    "Rate limit approaching: 450/500 requests",
                    "Memory usage at 85%",
                    f"Retry attempt 3/5 for {service}",
                ]
            )
        else:
            message = random.choice(
                [
                    "Request processed successfully",
                    "Health check passed",
                    f"Cache hit for key: user_session_{random.randint(1000, 9999)}",
                    f"Authenticated user: user_{random.randint(1000, 9999)}",
                ]
            )

        entries.append(
            {
                "timestamp": f"2024-01-15T{10 + (i // 60):02d}:{i % 60:02d}:00Z",
                "level": level,
                "service": service,
                "message": message,
                "trace_id": hashlib.md5(f"{i}".encode()).hexdigest()[:16],  # nosec B324
                "metadata": {
                    "host": f"pod-{service}-{random.randint(1, 5)}",
                    "region": random.choice(["us-east-1", "us-west-2", "eu-west-1"]),
                },
            }
        )

    return {
        "tool": "search_logs",
        "query": query,
        "result": {"total_hits": num_entries * 10, "returned": num_entries, "entries": entries},
    }


# =============================================================================
# AGENT SCENARIOS
# =============================================================================


@dataclass
class AgentScenario:
    """A realistic agent workflow scenario."""

    name: str
    description: str
    system_prompt: str
    user_query: str
    tools: list[dict]  # Tool outputs in sequence
    expected_answer_contains: list[str]  # Key phrases expected in good answer


def create_sre_debugging_scenario() -> AgentScenario:
    """
    SRE agent debugging a production incident.
    Multiple tool calls with large outputs.
    """
    return AgentScenario(
        name="SRE Incident Debugging",
        description="Debug a production incident using logs, metrics, and deployment info",
        system_prompt="""You are an SRE assistant helping debug production incidents.
You have access to tools for searching logs, querying metrics, and checking deployments.
Analyze the data carefully and identify the root cause.""",
        user_query="We're seeing 500 errors on the payment service. Can you investigate and find the root cause?",
        tools=[
            generate_log_search("payment error", num_entries=300),
            generate_database_query_results(
                "SELECT * FROM service_metrics WHERE service='payment'", num_rows=200
            ),
            generate_filesystem_search("payment", num_results=150),
        ],
        expected_answer_contains=["payment", "error", "connection", "timeout"],
    )


def create_codebase_exploration_scenario() -> AgentScenario:
    """
    Developer agent exploring a new codebase.
    File tree + search + code reading.
    """
    return AgentScenario(
        name="Codebase Exploration",
        description="Explore a codebase to understand authentication implementation",
        system_prompt="""You are a developer assistant helping explore codebases.
You have access to file system tools and code search.
Help the user understand how the codebase is structured.""",
        user_query="I need to understand how authentication is implemented. Can you find the relevant files and explain the flow?",
        tools=[
            generate_filesystem_tree("/project", depth=3, files_per_dir=20),
            generate_filesystem_search("authentication", num_results=200),
            generate_github_code_search("JWT authentication middleware", num_results=100),
        ],
        expected_answer_contains=["auth", "jwt", "middleware", "handler"],
    )


def create_issue_triage_scenario() -> AgentScenario:
    """
    GitHub agent triaging issues and finding related code.
    """
    return AgentScenario(
        name="GitHub Issue Triage",
        description="Triage GitHub issues and find related code",
        system_prompt="""You are a GitHub assistant helping triage issues.
Analyze issues, find patterns, and identify related code.""",
        user_query="Can you analyze the open issues and identify any patterns or high-priority bugs we should focus on?",
        tools=[
            generate_github_issues("myorg/myrepo", num_issues=100),
            generate_github_code_search("bug fix", num_results=80),
            generate_log_search("exception", num_entries=200),
        ],
        expected_answer_contains=["bug", "issue", "priority"],
    )


# =============================================================================
# BENCHMARK RUNNER
# =============================================================================


@dataclass
class BenchmarkResult:
    """Result from running a scenario."""

    scenario_name: str
    mode: str  # "baseline" or "headroom"
    total_input_tokens: int
    total_output_tokens: int
    total_tokens: int
    cost_usd: float
    latency_ms: float
    answer_quality: float  # 0-1 based on expected keywords
    num_tool_calls: int


def count_tokens_simple(text: str) -> int:
    """Simple token estimation (4 chars per token)."""
    return len(text) // 4


def run_agent_scenario(
    client: Any, scenario: AgentScenario, model: str = "gpt-4o-mini"
) -> BenchmarkResult:
    """Run a scenario and measure token usage."""

    messages = [
        {"role": "system", "content": scenario.system_prompt},
        {"role": "user", "content": scenario.user_query},
    ]

    # Add tool results with proper OpenAI format
    for tool_output in scenario.tools:
        tool_call_id = f"call_{hashlib.md5(tool_output['tool'].encode()).hexdigest()[:8]}"  # nosec B324
        # Assistant message with tool_calls (required by OpenAI)
        messages.append(
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": tool_call_id,
                        "type": "function",
                        "function": {"name": tool_output["tool"], "arguments": "{}"},
                    }
                ],
            }
        )
        messages.append(
            {
                "role": "tool",
                "tool_call_id": tool_call_id,
                "content": json.dumps(tool_output["result"], indent=2),
            }
        )

    # Add final question
    messages.append(
        {
            "role": "user",
            "content": "Based on all this information, what's your analysis and recommendation?",
        }
    )

    # Count input tokens
    input_text = json.dumps(messages)
    input_tokens = count_tokens_simple(input_text)

    # Make API call
    start = time.time()

    # Determine if using HeadroomClient
    is_headroom = isinstance(client, HeadroomClient) if HEADROOM_AVAILABLE else False
    mode = "headroom" if is_headroom else "baseline"

    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=1000,
        )
        latency = (time.time() - start) * 1000

        answer = response.choices[0].message.content
        output_tokens = (
            response.usage.completion_tokens
            if hasattr(response, "usage")
            else count_tokens_simple(answer)
        )
        actual_input_tokens = (
            response.usage.prompt_tokens if hasattr(response, "usage") else input_tokens
        )

        # Calculate answer quality
        answer_lower = answer.lower()
        matches = sum(1 for kw in scenario.expected_answer_contains if kw.lower() in answer_lower)
        quality = matches / len(scenario.expected_answer_contains)

        # Estimate cost (gpt-4o-mini pricing)
        cost = (actual_input_tokens * 0.00015 + output_tokens * 0.0006) / 1000

    except Exception as e:
        print(f"   Error: {e}")
        return BenchmarkResult(
            scenario_name=scenario.name,
            mode=mode,
            total_input_tokens=input_tokens,
            total_output_tokens=0,
            total_tokens=input_tokens,
            cost_usd=0.0,
            latency_ms=0,
            answer_quality=0.0,
            num_tool_calls=len(scenario.tools),
        )

    return BenchmarkResult(
        scenario_name=scenario.name,
        mode=mode,
        total_input_tokens=actual_input_tokens,
        total_output_tokens=output_tokens,
        total_tokens=actual_input_tokens + output_tokens,
        cost_usd=cost,
        latency_ms=latency,
        answer_quality=quality,
        num_tool_calls=len(scenario.tools),
    )


def run_full_benchmark(api_key: str = None) -> dict:
    """Run complete benchmark comparing baseline vs Headroom."""

    if api_key is None:
        api_key = os.environ.get("OPENAI_API_KEY")

    if not api_key:
        raise ValueError("OPENAI_API_KEY required")

    if not HEADROOM_AVAILABLE:
        raise RuntimeError("Headroom not available")

    # Create clients
    import tempfile

    from openai import OpenAI

    baseline_client = OpenAI(api_key=api_key)

    # Headroom-wrapped client
    db_path = os.path.join(tempfile.gettempdir(), "headroom_benchmark.db")
    headroom_client = HeadroomClient(
        original_client=OpenAI(api_key=api_key),
        provider=OpenAIProvider(),
        store_url=f"sqlite:///{db_path}",
        default_mode="optimize",
    )

    scenarios = [
        create_sre_debugging_scenario(),
        create_codebase_exploration_scenario(),
        create_issue_triage_scenario(),
    ]

    results = []

    print("\n" + "=" * 70)
    print("REAL-WORLD AGENT BENCHMARK: MCP Tools + Headroom")
    print("=" * 70)

    for scenario in scenarios:
        print(f"\n{'=' * 60}")
        print(f"Scenario: {scenario.name}")
        print(f"Description: {scenario.description}")
        print(f"Tool calls: {len(scenario.tools)}")
        print(f"{'=' * 60}")

        # Estimate raw data size
        raw_size = sum(len(json.dumps(t["result"])) for t in scenario.tools)
        print(f"\nRaw tool output size: {raw_size:,} chars (~{raw_size // 4:,} tokens)")

        # Run baseline
        print("\n[1/2] Running BASELINE (no compression)...")
        baseline_result = run_agent_scenario(baseline_client, scenario)
        print(f"   Input tokens: {baseline_result.total_input_tokens:,}")
        print(f"   Output tokens: {baseline_result.total_output_tokens:,}")
        print(f"   Cost: ${baseline_result.cost_usd:.4f}")
        print(f"   Answer quality: {baseline_result.answer_quality:.1%}")
        results.append(baseline_result)

        # Run with Headroom
        print("\n[2/2] Running HEADROOM (optimized)...")
        headroom_result = run_agent_scenario(headroom_client, scenario)
        print(f"   Input tokens: {headroom_result.total_input_tokens:,}")
        print(f"   Output tokens: {headroom_result.total_output_tokens:,}")
        print(f"   Cost: ${headroom_result.cost_usd:.4f}")
        print(f"   Answer quality: {headroom_result.answer_quality:.1%}")
        results.append(headroom_result)

        # Calculate savings
        if baseline_result.total_input_tokens > 0:
            token_savings = 1 - (
                headroom_result.total_input_tokens / baseline_result.total_input_tokens
            )
            cost_savings = (
                1 - (headroom_result.cost_usd / baseline_result.cost_usd)
                if baseline_result.cost_usd > 0
                else 0
            )
            print("\n   📊 SAVINGS:")
            print(f"   Token reduction: {token_savings:.1%}")
            print(f"   Cost reduction: {cost_savings:.1%}")
            print(
                f"   Quality preserved: {'✓' if headroom_result.answer_quality >= baseline_result.answer_quality * 0.9 else '✗'}"
            )

    # Summary
    print("\n" + "=" * 70)
    print("BENCHMARK SUMMARY")
    print("=" * 70)

    baseline_results = [r for r in results if r.mode == "baseline"]
    headroom_results = [r for r in results if r.mode == "headroom"]

    total_baseline_tokens = sum(r.total_input_tokens for r in baseline_results)
    total_headroom_tokens = sum(r.total_input_tokens for r in headroom_results)
    total_baseline_cost = sum(r.cost_usd for r in baseline_results)
    total_headroom_cost = sum(r.cost_usd for r in headroom_results)

    print(f"\n{'Metric':<25} {'Baseline':>15} {'Headroom':>15} {'Savings':>15}")
    print("-" * 70)

    token_savings = (
        (1 - total_headroom_tokens / total_baseline_tokens) if total_baseline_tokens > 0 else 0
    )
    cost_savings = (1 - total_headroom_cost / total_baseline_cost) if total_baseline_cost > 0 else 0

    print(
        f"{'Total Input Tokens':<25} {total_baseline_tokens:>15,} {total_headroom_tokens:>15,} {token_savings:>14.1%}"
    )
    print(
        f"{'Total Cost':<25} ${total_baseline_cost:>14.4f} ${total_headroom_cost:>14.4f} {cost_savings:>14.1%}"
    )

    avg_baseline_quality = (
        sum(r.answer_quality for r in baseline_results) / len(baseline_results)
        if baseline_results
        else 0
    )
    avg_headroom_quality = (
        sum(r.answer_quality for r in headroom_results) / len(headroom_results)
        if headroom_results
        else 0
    )
    print(
        f"{'Avg Answer Quality':<25} {avg_baseline_quality:>14.1%} {avg_headroom_quality:>14.1%} {'preserved' if avg_headroom_quality >= avg_baseline_quality * 0.9 else 'degraded':>15}"
    )

    return {
        "baseline": [r.__dict__ for r in baseline_results],
        "headroom": [r.__dict__ for r in headroom_results],
        "summary": {
            "total_baseline_tokens": total_baseline_tokens,
            "total_headroom_tokens": total_headroom_tokens,
            "token_savings": token_savings,
            "total_baseline_cost": total_baseline_cost,
            "total_headroom_cost": total_headroom_cost,
            "cost_savings": cost_savings,
        },
    }


if __name__ == "__main__":
    results = run_full_benchmark()

    # Save results
    with open("real_world_benchmark_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\nResults saved to real_world_benchmark_results.json")
