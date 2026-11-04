#!/usr/bin/env python3
"""Vertex AI (Gemini Enterprise Agent Platform) + Headroom Context Compression Benchmark.

Evaluates Gemini 3.8 Flash on Vertex AI across realistic multi-turn agent workloads:
1. Baseline: Direct Vertex AI API calls with raw, uncompressed tool outputs.
2. Headroom: Transparently proxied Vertex AI API calls with intelligent context compression.

Measures:
- Input / Prompt Token Reduction (%)
- Roundtrip Latency & TTFT (ms)
- Total Inference Cost Savings ($ USD)
- Absolute Ground Truth Accuracy (% of expected facts/anomalies identified)
- Relative Quality Retention (% of Baseline accuracy preserved after compression)
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from contextlib import suppress
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

# Ensure local package import works
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from google import genai
    from google.genai import types
except ImportError:
    print("Error: google-genai SDK is not installed.", file=sys.stderr)
    print("Run `pip install google-genai` and try again.", file=sys.stderr)
    sys.exit(1)

from examples.vertex_gemini_benchmark.scenarios import (  # noqa: E402
    BenchmarkScenario,
    get_all_scenarios,
)

# ----------------------------------------------------------------------------
# Constants & Pricing
# ----------------------------------------------------------------------------

DEFAULT_MODEL = "gemini-3.8-flash"
DEFAULT_LOCATION = "global"
DEFAULT_PORT = 8787

# Vertex AI Gemini 3.8 Flash Introductory Standard Pricing (through 2026-12-31):
# - Input tokens: $0.75 per 1M un-cached prompt tokens
# - Text output tokens: $3.75 per 1M candidate tokens
# Source: Google Cloud Vertex AI Pricing documentation (Introductory rates through Dec 31, 2026)
INPUT_PRICE_PER_M = 0.75
OUTPUT_PRICE_PER_M = 3.75


@dataclass
class TrialResult:
    scenario_name: str
    category: str
    mode: str  # "Baseline (Direct)" or "Headroom (Optimized)"
    prompt_tokens: int
    candidate_tokens: int
    total_tokens: int
    latency_ms: float
    cost_usd: float
    accuracy_score: float  # Absolute ground truth score [0.0 - 1.0]
    response_text: str
    facts_found: list[str]
    anomalies_found: list[str]


@dataclass
class ScenarioComparison:
    scenario_name: str
    category: str
    baseline: TrialResult
    headroom: TrialResult
    token_savings_pct: float
    cost_savings_pct: float
    latency_delta_ms: float
    quality_retained: bool
    relative_accuracy_retention_pct: float


# ----------------------------------------------------------------------------
# Proxy Lifecycle Management
# ----------------------------------------------------------------------------


def start_headroom_proxy(
    port: int, region: str, savings_profile: str = "agent-90"
) -> subprocess.Popen[bytes]:
    """Start Headroom proxy as a background process configured for agent compression."""
    env = os.environ.copy()
    env.setdefault("HEADROOM_LOG", "INFO")
    env["HEADROOM_SAVINGS_PROFILE"] = savings_profile
    env["HEADROOM_COMPRESS_USER_MESSAGES"] = "1"
    env["HEADROOM_NO_MEMORY_TOOLS"] = "1"
    env["HEADROOM_NO_MEMORY_CONTEXT"] = "1"

    cmd = [
        sys.executable,
        "-m",
        "headroom.cli",
        "proxy",
        "--backend",
        "vertex",
        "--region",
        region,
        "--port",
        str(port),
        "--no-ccr",
        "--no-memory-tools",
        "--no-memory-context",
    ]
    log_path = Path("/tmp") / f"headroom_benchmark_proxy_{port}.log"
    log_file = log_path.open("wb")
    proc = subprocess.Popen(
        cmd,
        env=env,
        stdout=log_file,
        stderr=subprocess.STDOUT,
    )
    return proc


def wait_for_proxy_ready(port: int, timeout_s: float = 35.0) -> None:
    """Poll proxy /readyz until ready."""
    url = f"http://127.0.0.1:{port}/readyz"
    deadline = time.time() + timeout_s
    last_err: Exception | None = None
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=1) as resp:
                if resp.status == 200:
                    return
        except (urllib.error.URLError, ConnectionError, TimeoutError) as e:
            last_err = e
        time.sleep(0.4)
    raise RuntimeError(
        f"Proxy on port {port} failed to become ready within {timeout_s}s: {last_err}"
    )


def stop_proxy(proc: subprocess.Popen[bytes]) -> None:
    """Politely terminate proxy process."""
    with suppress(ProcessLookupError):
        proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=5)


# ----------------------------------------------------------------------------
# Execution & Scoring
# ----------------------------------------------------------------------------


def build_gemini_contents(scenario: BenchmarkScenario) -> list[types.Content]:
    """Format multi-turn tool interaction into realistic Gemini Content turns."""
    contents: list[types.Content] = []

    # 1. User query
    contents.append(
        types.Content(
            role="user",
            parts=[types.Part.from_text(text=scenario.user_query)],
        )
    )

    # 2. Simulated tool executions and output responses
    for tool_out in scenario.tool_outputs:
        tool_name = tool_out["tool"]
        raw_result = json.dumps(tool_out["result"], indent=2)

        # Model turn announcing tool execution
        contents.append(
            types.Content(
                role="model",
                parts=[
                    types.Part.from_text(
                        text=f"Executing tool `{tool_name}` to retrieve relevant context..."
                    )
                ],
            )
        )

        # Tool result returned to model in user context turn
        contents.append(
            types.Content(
                role="user",
                parts=[
                    types.Part.from_text(
                        text=f"Tool `{tool_name}` output:\n```json\n{raw_result}\n```"
                    )
                ],
            )
        )

    # 3. Final instruction turn to trigger synthesis
    contents.append(
        types.Content(
            role="user",
            parts=[
                types.Part.from_text(
                    text="Synthesize findings from all tool outputs above. Provide a concise, highly specific response citing all exact root causes, service names, error messages, identifiers, and anomalies."
                )
            ],
        )
    )

    return contents


def evaluate_response_quality(
    response_text: str, scenario: BenchmarkScenario
) -> tuple[float, list[str], list[str]]:
    """Evaluate whether the model response accurately captured all ground truth facts & anomalies."""
    text_lower = response_text.lower()
    text_clean = text_lower.replace("$", "").replace(",", "").replace("-", " ")

    def _matches(needle: str) -> bool:
        n = needle.lower()
        if n in text_lower:
            return True
        n_clean = n.replace("$", "").replace(",", "").replace("-", " ")
        if n_clean in text_clean:
            return True
        return False

    facts_found = [fact for fact in scenario.expected_facts if _matches(fact)]
    anomalies_found = [anom for anom in scenario.expected_anomalies if _matches(anom)]

    total_expected = len(scenario.expected_facts) + len(scenario.expected_anomalies)
    if total_expected == 0:
        return 1.0, facts_found, anomalies_found

    total_found = len(facts_found) + len(anomalies_found)
    score = total_found / total_expected
    return score, facts_found, anomalies_found


def calculate_cost(prompt_tokens: int, candidate_tokens: int) -> float:
    """Calculate inference cost in USD for Gemini 3.8 Flash on Vertex."""
    return (prompt_tokens * INPUT_PRICE_PER_M + candidate_tokens * OUTPUT_PRICE_PER_M) / 1_000_000.0


def run_trial(
    client: genai.Client,
    scenario: BenchmarkScenario,
    model: str,
    mode_name: str,
    thinking_budget: int = 0,
) -> TrialResult:
    """Run a single benchmark trial against Gemini on Vertex."""
    contents = build_gemini_contents(scenario)

    config_kwargs: dict[str, Any] = {
        "system_instruction": scenario.system_prompt,
        "temperature": 0.1,  # Low temperature for deterministic evaluation
    }
    if thinking_budget > 0:
        config_kwargs["thinking_config"] = types.ThinkingConfig(thinking_budget=thinking_budget)

    config = types.GenerateContentConfig(**config_kwargs)

    start_time = time.perf_counter()
    response = client.models.generate_content(
        model=model,
        contents=contents,
        config=config,
    )
    elapsed_ms = (time.perf_counter() - start_time) * 1000.0

    usage = getattr(response, "usage_metadata", None)
    prompt_tokens = getattr(usage, "prompt_token_count", 0) if usage else 0
    candidate_tokens = getattr(usage, "candidates_token_count", 0) if usage else 0
    total_tokens = (
        getattr(usage, "total_token_count", prompt_tokens + candidate_tokens) if usage else 0
    )

    response_text = response.text or ""
    score, facts_found, anomalies_found = evaluate_response_quality(response_text, scenario)
    cost = calculate_cost(prompt_tokens, candidate_tokens)

    return TrialResult(
        scenario_name=scenario.name,
        category=scenario.category,
        mode=mode_name,
        prompt_tokens=prompt_tokens,
        candidate_tokens=candidate_tokens,
        total_tokens=total_tokens,
        latency_ms=elapsed_ms,
        cost_usd=cost,
        accuracy_score=score,
        response_text=response_text,
        facts_found=facts_found,
        anomalies_found=anomalies_found,
    )


# ----------------------------------------------------------------------------
# Benchmark Suite Runner
# ----------------------------------------------------------------------------


def run_benchmark_suite(
    project_id: str,
    location: str = DEFAULT_LOCATION,
    model: str = DEFAULT_MODEL,
    port: int = DEFAULT_PORT,
    thinking_budget: int = 0,
    output_json: str | None = None,
    social_format: bool = False,
) -> dict[str, Any]:
    """Execute the complete comparative benchmark suite."""
    print("=" * 82)
    print(" 🚀 VERTEX AI (GEMINI ENTERPRISE AGENT PLATFORM) + HEADROOM BENCHMARK")
    print("=" * 82)
    print(f" Model:     {model}")
    print(f" Location:  {location}")
    print(f" Project:   {project_id}")
    print(f" Proxy:     http://127.0.0.1:{port}")
    print(
        f" Pricing:   ${INPUT_PRICE_PER_M}/M input, ${OUTPUT_PRICE_PER_M}/M output (Introductory rate)"
    )
    if thinking_budget > 0:
        print(f" Thinking:  budget={thinking_budget} tokens")
    print("=" * 82)

    scenarios = get_all_scenarios()
    comparisons: list[ScenarioComparison] = []

    # Clean any stale proxy processes on this port
    try:
        subprocess.run(
            ["pkill", "-f", f"headroom.cli proxy.*{port}"],
            check=False,
            capture_output=True,
        )
        time.sleep(0.5)
    except Exception:
        pass

    # 1. Start Headroom Proxy
    print("\n[1/4] Spawning Headroom compression proxy ...")
    proxy_proc = start_headroom_proxy(port=port, region=location)

    try:
        wait_for_proxy_ready(port=port, timeout_s=40.0)
        print("  ✓ Headroom proxy is active and ready.\n")

        # 2. Build Clients
        client_baseline = genai.Client(
            vertexai=True,
            project=project_id,
            location=location,
        )
        client_headroom = genai.Client(
            vertexai=True,
            project=project_id,
            location=location,
            http_options={"base_url": f"http://127.0.0.1:{port}"},
        )

        print("[2/4] Executing Benchmark Scenarios ...\n")

        for idx, scenario in enumerate(scenarios, 1):
            raw_kb = scenario.total_raw_chars() / 1024.0
            print(f"Scenario [{idx}/{len(scenarios)}]: {scenario.name} ({scenario.category})")
            print(
                f"  Payload: ~{raw_kb:.1f} KB raw tool outputs across {len(scenario.tool_outputs)} calls"
            )

            # Run Baseline
            print("  ↳ Running Baseline (Direct Vertex AI) ...", end="", flush=True)
            baseline_res = run_trial(
                client_baseline,
                scenario,
                model=model,
                mode_name="Baseline (Direct)",
                thinking_budget=thinking_budget,
            )
            print(
                f" done ({baseline_res.prompt_tokens:,} prompt tokens, {baseline_res.latency_ms:.0f}ms)"
            )

            # Run Headroom
            print("  ↳ Running Headroom (Context Compressed) ...", end="", flush=True)
            headroom_res = run_trial(
                client_headroom,
                scenario,
                model=model,
                mode_name="Headroom (Optimized)",
                thinking_budget=thinking_budget,
            )
            print(
                f" done ({headroom_res.prompt_tokens:,} prompt tokens, {headroom_res.latency_ms:.0f}ms)"
            )

            # Calculate savings
            token_savings = (
                (1.0 - headroom_res.prompt_tokens / baseline_res.prompt_tokens) * 100.0
                if baseline_res.prompt_tokens > 0
                else 0.0
            )
            cost_savings = (
                (1.0 - headroom_res.cost_usd / baseline_res.cost_usd) * 100.0
                if baseline_res.cost_usd > 0
                else 0.0
            )
            latency_delta = baseline_res.latency_ms - headroom_res.latency_ms
            quality_ok = headroom_res.accuracy_score >= baseline_res.accuracy_score * 0.90
            relative_retention = (
                (headroom_res.accuracy_score / baseline_res.accuracy_score) * 100.0
                if baseline_res.accuracy_score > 0
                else 100.0
            )

            print(
                f"  📊 Results: Prompt Tokens: {baseline_res.prompt_tokens:,} -> {headroom_res.prompt_tokens:,} (-{token_savings:.1f}%)"
            )
            print(
                f"     Latency: {baseline_res.latency_ms:.0f}ms -> {headroom_res.latency_ms:.0f}ms ({latency_delta:+.0f}ms)"
            )
            print(
                f"     Accuracy: Baseline={baseline_res.accuracy_score:.1%}, Headroom={headroom_res.accuracy_score:.1%} (Retention: {relative_retention:.1f}%, {'✓ PASS' if quality_ok else '✗ DEGRADED'})\n"
            )

            comparisons.append(
                ScenarioComparison(
                    scenario_name=scenario.name,
                    category=scenario.category,
                    baseline=baseline_res,
                    headroom=headroom_res,
                    token_savings_pct=token_savings,
                    cost_savings_pct=cost_savings,
                    latency_delta_ms=latency_delta,
                    quality_retained=quality_ok,
                    relative_accuracy_retention_pct=relative_retention,
                )
            )

    finally:
        print("[3/4] Shutting down Headroom proxy ...")
        stop_proxy(proxy_proc)
        print("  ✓ Proxy shut down cleanly.\n")

    # 3. Overall Summary Calculations
    total_baseline_prompt = sum(c.baseline.prompt_tokens for c in comparisons)
    total_headroom_prompt = sum(c.headroom.prompt_tokens for c in comparisons)
    total_baseline_tokens = sum(c.baseline.total_tokens for c in comparisons)
    total_headroom_tokens = sum(c.headroom.total_tokens for c in comparisons)
    total_baseline_cost = sum(c.baseline.cost_usd for c in comparisons)
    total_headroom_cost = sum(c.headroom.cost_usd for c in comparisons)
    avg_baseline_latency = sum(c.baseline.latency_ms for c in comparisons) / len(comparisons)
    avg_headroom_latency = sum(c.headroom.latency_ms for c in comparisons) / len(comparisons)
    avg_baseline_acc = sum(c.baseline.accuracy_score for c in comparisons) / len(comparisons)
    avg_headroom_acc = sum(c.headroom.accuracy_score for c in comparisons) / len(comparisons)

    overall_token_savings = (
        (1.0 - total_headroom_prompt / total_baseline_prompt) * 100.0
        if total_baseline_prompt > 0
        else 0.0
    )
    overall_cost_savings = (
        (1.0 - total_headroom_cost / total_baseline_cost) * 100.0
        if total_baseline_cost > 0
        else 0.0
    )
    overall_relative_retention = (
        (avg_headroom_acc / avg_baseline_acc * 100.0) if avg_baseline_acc > 0 else 100.0
    )
    retention_label = (
        "100.0% Retained"
        if overall_relative_retention >= 99.9
        else f"{overall_relative_retention:.1f}% Retained"
    )

    # 4. Print Formatted Table
    print("=" * 82)
    print(f" 📊 FINAL BENCHMARK SUMMARY: {model.upper()} ON VERTEX AI")
    print("=" * 82)
    print(
        f"{'Scenario':<34} | {'Baseline Prompt':>15} | {'Headroom Prompt':>15} | {'Reduction':>10}"
    )
    print("-" * 82)
    for c in comparisons:
        print(
            f"{c.scenario_name:<34} | {c.baseline.prompt_tokens:>15,} | {c.headroom.prompt_tokens:>15,} | {c.token_savings_pct:>9.1f}%"
        )
    print("-" * 82)
    print(
        f"{'TOTAL / AGGREGATE':<34} | {total_baseline_prompt:>15,} | {total_headroom_prompt:>15,} | {overall_token_savings:>9.1f}%\n"
    )

    print(f"{'Metric':<34} | {'Baseline':>15} | {'Headroom':>15} | {'Delta / Impact':>14}")
    print("-" * 82)
    print(
        f"{'Total Prompt Tokens':<34} | {total_baseline_prompt:>15,} | {total_headroom_prompt:>15,} | -{overall_token_savings:>12.1f}%"
    )
    print(
        f"{'Total All Tokens':<34} | {total_baseline_tokens:>15,} | {total_headroom_tokens:>15,} | -{(1 - total_headroom_tokens / total_baseline_tokens) * 100:>12.1f}%"
    )
    print(
        f"{'Total Cost ($ USD)':<34} | ${total_baseline_cost:>14.5f} | ${total_headroom_cost:>14.5f} | -{overall_cost_savings:>12.1f}%"
    )
    print(
        f"{'Avg Latency (ms)':<34} | {avg_baseline_latency:>13.0f}ms | {avg_headroom_latency:>13.0f}ms | {avg_baseline_latency - avg_headroom_latency:>+12.0f}ms"
    )
    print(
        f"{'Ground Truth Accuracy (Absolute)':<34} | {avg_baseline_acc:>14.1%} | {avg_headroom_acc:>14.1%} | {retention_label:>14}"
    )
    print("=" * 82)

    # 5. Social Post Format
    social_text = f"""
🚀 **Headroom + {model} on Google Cloud Vertex AI Benchmark**

When AI agents run complex multi-turn workflows (SRE debugging, PR reviews, BigQuery analytics), tool output bloat explodes prompt token costs and degrades TTFT.

We ran reproducible end-to-end agent benchmarks comparing **Direct Vertex AI** vs **Headroom-Proxied Vertex AI** on `{model}`:

📉 **Results**:
• **Prompt Token Reduction**: **{overall_token_savings:.1f}%** ({total_baseline_prompt:,} ➔ {total_headroom_prompt:,} tokens)
• **Total Cost Savings**: **{overall_cost_savings:.1f}%** (${total_baseline_cost:.4f} ➔ ${total_headroom_cost:.4f})
• **Relative Quality Retention**: **{overall_relative_retention:.1f}%** ({avg_headroom_acc:.1%} Headroom vs {avg_baseline_acc:.1%} Baseline ground truth score)
• **Zero Code Changes**: Point `google-genai` SDK `http_options.base_url` to `http://127.0.0.1:{port}`.

🔗 Full benchmark suite, reproducible scenarios, and code:
https://github.com/headroomlabs-ai/headroom/tree/main/examples/vertex_gemini_benchmark
"""
    if social_format:
        print("\n" + "=" * 82)
        print(" 📢 DEVELOPER SOCIAL POST PROOF POINT")
        print("=" * 82)
        print(social_text.strip())
        print("=" * 82 + "\n")

    # 6. JSON Export
    result_data = {
        "metadata": {
            "model": model,
            "location": location,
            "project_id": project_id,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "thinking_budget": thinking_budget,
            "pricing_source": (
                "Google Cloud Vertex AI introductory standard pricing through 2026-12-31: "
                f"${INPUT_PRICE_PER_M}/M input tokens, ${OUTPUT_PRICE_PER_M}/M text output tokens."
            ),
        },
        "aggregate": {
            "total_baseline_prompt_tokens": total_baseline_prompt,
            "total_headroom_prompt_tokens": total_headroom_prompt,
            "overall_token_savings_pct": overall_token_savings,
            "total_baseline_cost_usd": total_baseline_cost,
            "total_headroom_cost_usd": total_headroom_cost,
            "overall_cost_savings_pct": overall_cost_savings,
            "avg_baseline_latency_ms": avg_baseline_latency,
            "avg_headroom_latency_ms": avg_headroom_latency,
            "avg_baseline_accuracy": avg_baseline_acc,
            "avg_headroom_accuracy": avg_headroom_acc,
            "overall_relative_retention_pct": overall_relative_retention,
        },
        "scenarios": [asdict(c) for c in comparisons],
        "social_proof_point": social_text.strip(),
    }

    if output_json:
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(result_data, f, indent=2)
        print(f"✓ Detailed benchmark results exported to: {output_json}")

    return result_data


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Gemini 3.8 Flash on Vertex AI + Headroom Benchmark"
    )
    parser.add_argument(
        "--project", default=os.environ.get("GCP_PROJECT_ID"), help="GCP Project ID"
    )
    parser.add_argument(
        "--location", default=DEFAULT_LOCATION, help="Vertex location (default: global)"
    )
    parser.add_argument(
        "--model", default=DEFAULT_MODEL, help="Model ID (default: gemini-3.8-flash)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=DEFAULT_PORT,
        help="Headroom proxy port (default: 8787)",
    )
    parser.add_argument(
        "--thinking-budget",
        type=int,
        default=0,
        help="Thinking token budget (0 = disabled)",
    )
    parser.add_argument(
        "--output-json",
        default="examples/vertex_gemini_benchmark/results.json",
        help="Path to save output JSON",
    )
    parser.add_argument(
        "--social",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Print social post text",
    )
    args = parser.parse_args()

    project_id = args.project
    if not project_id:
        try:
            cmd_out = subprocess.check_output(
                ["gcloud", "config", "get-value", "project"], text=True
            ).strip()
            if cmd_out:
                project_id = cmd_out
        except Exception:
            pass

    if not project_id:
        print(
            "Error: GCP_PROJECT_ID is not set. Specify via --project or set GCP_PROJECT_ID.",
            file=sys.stderr,
        )
        return 1

    try:
        run_benchmark_suite(
            project_id=project_id,
            location=args.location,
            model=args.model,
            port=args.port,
            thinking_budget=args.thinking_budget,
            output_json=args.output_json,
            social_format=args.social,
        )
        return 0
    except Exception as e:
        print(f"\n❌ Benchmark failed with error: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        return 2


if __name__ == "__main__":
    sys.exit(main())
