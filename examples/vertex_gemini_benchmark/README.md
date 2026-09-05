# Gemini 3.8 Flash on Google Cloud Vertex AI + Headroom Benchmark

A reproducible, real-world benchmark evaluating **Headroom context compression** with **Gemini 3.8 Flash** on **Google Cloud Vertex AI (Gemini Enterprise Agent Platform)**.

---

## 🎯 Why This Matters

When building production agents (coding assistants, SRE incident responders, data analysts, multi-agent frameworks) on Vertex AI, multi-turn tool loops cause **rapid context explosion**:

* **Container & Kubernetes logs** dump hundreds of lines of noise for a single stack trace.
* **Code search & file trees** inflate prompts with repetitive schema structures.
* **Database queries** return large tabular results where only outliers and aggregations matter.

Even with Gemini 3.8 Flash's massive 1M-token context window and fast inference, bloated tool returns:
1. **Drive up inference spend** as conversation histories compound across turns.
2. **Increase Time to First Token (TTFT)** due to large prompt prefill processing.
3. **Dilute attention**, making needle-in-a-haystack reasoning and anomaly isolation harder.

**Headroom** acts as an intelligent, transparent proxy (or in-process SDK layer) that compresses JSON arrays, structured logs, and tables by **40–85%** while strictly preserving schema anchors, recent turns, anomalies, error traces, and ground truth accuracy.

---

## 📊 Live Benchmark Results

Tested live on **Google Cloud Vertex AI** (`global` endpoint) with **`gemini-3.8-flash`**.

Pricing basis: Google Cloud Vertex AI introductory standard rates through 2026-12-31 ($0.75 per 1M prompt tokens, $3.75 per 1M text output tokens; cached input rate is $0.075/M).

| Scenario | Workload Category | Baseline Prompt | Headroom Prompt | Token Reduction | Latency Delta | Baseline Accuracy | Headroom Accuracy | Relative Retention |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **SRE Incident Root Cause** | Kubernetes & Microservice Logs | 51,775 | 13,842 | **-73.3%** | +9.2s faster | 100.0% | 100.0% | **100.0%** (✓ PASS) |
| **Security Audit & PR Review** | Code Search & Git Diffs | 6,821 | 4,632 | **-32.1%** | +1.9s faster | 100.0% | 100.0% | **100.0%** (✓ PASS) |
| **BigQuery Table Analytics** | 500 Tabular Transaction Rows | 49,020 | 6,120 | **-87.5%** | +2.3s faster | 100.0% | 100.0% | **100.0%** (✓ PASS) |
| **Multi-Turn RAG Synthesis** | 25 Dense Specification Chunks | 4,071 | 1,437 | **-64.7%** | -0.7s | 100.0% | 100.0% | **100.0%** (✓ PASS) |
| **TOTAL / AGGREGATE** | **Real-World Agent Trajectory** | **111,687** | **26,031** | **-76.7%** | **+3.2s avg faster** | **100.0%** | **100.0%** | **100.0% Retained** |

### Key Metrics Summary

* **Prompt Tokens Saved**: **85,656 tokens** (**76.7% net reduction**, 111,687 down to 26,031)
* **Total Inference Cost**: **$0.0941 down to $0.0285** (**69.7% cost savings** at standard Vertex rates)
* **Average Latency**: **8.9s down to 5.7s** (**+3.2s faster roundtrip** due to reduced prompt prefill load)
* **Relative Quality Retention**: **100.0%** (zero degradation; Headroom matched baseline extraction of all root causes, database connection pool exhaustion, JWT `alg: none` auth bypass, fraud outliers, and architectural contracts)

---

## 🏗️ Architecture

```
                                          ┌───────────────────────────────────────┐
                                          │ Google Cloud Vertex AI                │
                                          │ (Gemini Enterprise Agent Platform)    │
                                          │                                       │
┌───────────────────────┐                 │  ┌─────────────────────────────────┐  │
│  google-genai Python  │                 │  │        gemini-3.8-flash         │  │
│  SDK Agent / Script   │                 │  └─────────────────────────────────┘  │
└───────────┬───────────┘                 └───────────────────▲───────────────────┘
            │                                                 │
            │  POST /v1/projects/.../publishers/...           │ Compressed
            │  (base_url = http://127.0.0.1:8787)             │ Payload
            ▼                                                 │
┌─────────────────────────────────────────────────────────────┴───────────────────┐
│ Headroom Proxy (:8787)                                                          │
│                                                                                 │
│  ┌───────────────────────┐   ┌────────────────────────┐   ┌──────────────────┐  │
│  │ ContentRouter         │──▶│ SmartCrusher           │──▶│ LogCompressor    │  │
│  │ (Format & Role Sieve) │   │ (JSON Array Compactor) │   │ (Error Anchor)   │  │
│  └───────────────────────┘   └────────────────────────┘   └──────────────────┘  │
│                                                                                 │
│  • Preserves Google ADC Bearer Auth Tokens                                      │
│  • Compresses verbose tool arrays & tables                                      │
│  • Preserves 100% of anomalies, errors, and schema anchors                      │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## 🚀 How to Run the Benchmark

### 1. Prerequisites

Ensure you have Google Cloud Application Default Credentials (ADC) configured:

```bash
gcloud auth application-default login
export GCP_PROJECT_ID=$(gcloud config get-value project)
```

Install required dependencies:

```bash
pip install "headroom-ai[proxy]" google-genai
```

### 2. Execute Benchmark

Run the full comparative suite:

```bash
python examples/vertex_gemini_benchmark/benchmark.py --model gemini-3.8-flash
```

#### CLI Options

```text
--project         GCP Project ID (defaults to $GCP_PROJECT_ID or gcloud default)
--location        Vertex AI location (default: global)
--model           Vertex model ID (default: gemini-3.8-flash)
--port            Headroom local proxy port (default: 8787)
--thinking-budget Thinking token budget in tokens (default: 0 = standard inference)
--output-json     Output file for JSON metrics (default: examples/vertex_gemini_benchmark/results.json)
--social / --no-social  Print formatted social media proof point summary (default: on)
```

---

## 💡 Using Headroom with Vertex AI in Your Agent Code

Connecting your `google-genai` agent to Headroom requires **one line** (`http_options`):

```python
from google import genai

# Point the standard SDK at the Headroom proxy
client = genai.Client(
    vertexai=True,
    project="your-gcp-project-id",
    location="global",
    http_options={"base_url": "http://127.0.0.1:8787"},
)

response = client.models.generate_content(
    model="gemini-3.8-flash",
    contents=[
        "You are an SRE agent.",
        f"Analyze these Kubernetes logs:\n{verbose_json_logs}",
    ],
)
print(response.text)
```

---

## 📢 Social Post / Proof Point Card

```markdown
🚀 Headroom + Gemini 3.8 Flash on Google Cloud Vertex AI Benchmark

When AI agents run complex multi-turn workflows (SRE debugging, PR reviews, BigQuery analytics), tool output bloat explodes prompt token costs and degrades TTFT.

We ran reproducible end-to-end agent benchmarks comparing Direct Vertex AI vs Headroom-Proxied Vertex AI on gemini-3.8-flash:

📉 Results:
• Prompt Token Reduction: 76.7% (111,687 ➔ 26,031 tokens)
• SRE Log Scenario Reduction: 73.3% (51.7k ➔ 13.8k tokens)
• BigQuery Analytics Scenario: 87.5% (49.0k ➔ 6.1k tokens)
• Total Cost Savings: 69.7% ($0.0941 ➔ $0.0285 at standard Vertex rates)
• Relative Quality Retention: 100.0% (Zero reasoning degradation; 100.0% ground truth retention in both arms)
• Zero Code Changes: Point google-genai SDK http_options.base_url to http://127.0.0.1:8787.

🔗 Full benchmark suite, reproducible scenarios, and code:
https://github.com/headroomlabs-ai/headroom/tree/main/examples/vertex_gemini_benchmark
```
