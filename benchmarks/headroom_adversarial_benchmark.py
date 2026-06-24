"""
Headroom ADVERSARIAL Benchmark: True Worst Cases

The previous "worst case" scenarios still had JSON structure.
This benchmark tests TRUE adversarial cases:

1. Dense prose - research papers, no structure
2. Code diffs - every line matters, minimal redundancy
3. Encrypted/random data - no patterns possible
4. Tiny datasets - not enough data for statistics
5. High-entropy unique content - no repeated patterns
"""

import hashlib
import json
import os
import random
import string
from dataclasses import dataclass

try:
    from openai import OpenAI  # noqa: F401

    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    from headroom import HeadroomClient, OpenAIProvider

    HEADROOM_AVAILABLE = True
except ImportError:
    HEADROOM_AVAILABLE = False


# =============================================================================
# ADVERSARIAL DATA GENERATORS
# =============================================================================


def generate_research_paper_excerpts(num_papers: int = 10) -> dict:
    """
    Dense academic text - every word carries meaning.
    No JSON structure, no repetition, pure prose.
    """
    # Simulated research paper abstracts - dense, unique content
    papers = []

    topics = [
        ("quantum computing", "qubit coherence", "error correction", "topological"),
        ("machine learning", "transformer architecture", "attention mechanism", "gradient"),
        ("climate science", "carbon sequestration", "permafrost", "albedo effect"),
        ("neuroscience", "synaptic plasticity", "hippocampal", "neurogenesis"),
        ("economics", "monetary policy", "inflation targeting", "yield curve"),
        ("genetics", "CRISPR-Cas9", "gene expression", "epigenetic"),
        ("astrophysics", "gravitational waves", "neutron star", "black hole merger"),
        ("materials science", "graphene", "superconductivity", "metamaterial"),
        ("cryptography", "post-quantum", "lattice-based", "homomorphic encryption"),
        ("pharmacology", "receptor binding", "pharmacokinetics", "bioavailability"),
    ]

    for i in range(num_papers):
        topic = topics[i % len(topics)]

        # Generate unique, dense academic prose
        abstract = f"""
This paper presents novel findings in {topic[0]} research, specifically addressing the challenge of {topic[1]} optimization.
Our methodology employs a combination of {topic[2]} analysis and {topic[3]} modeling approaches that have not been
previously explored in the literature. Through rigorous experimentation with {random.randint(50, 500)} samples
across {random.randint(3, 12)} controlled conditions, we demonstrate a {random.randint(15, 45)}% improvement
over baseline methods (p < 0.{random.randint(1, 5):02d}).

The theoretical framework builds upon the seminal work of {random.choice(["Smith et al.", "Johnson & Lee", "Chen group", "Williams lab"])} (20{random.randint(15, 23)}),
extending their {random.choice(["analytical", "computational", "experimental", "theoretical"])} approach to address
{random.choice(["scalability concerns", "edge cases", "real-world constraints", "noise sensitivity"])}.
Our key contribution is the development of a {random.choice(["novel algorithm", "unified framework", "hybrid methodology", "robust protocol"])}
that achieves {random.choice(["state-of-the-art", "competitive", "superior", "breakthrough"])} performance while
maintaining {random.choice(["computational efficiency", "interpretability", "generalizability", "reproducibility"])}.

Implications of this work extend to {random.choice(["industrial applications", "clinical settings", "policy decisions", "fundamental understanding"])}
in the domain of {topic[0]}. We identify {random.randint(3, 7)} key factors that influence {topic[1]} behavior,
with {random.choice(["temperature", "pressure", "concentration", "frequency", "duration"])} being the most significant
(correlation coefficient r = 0.{random.randint(70, 95)}). Future work will focus on {random.choice(["scaling", "optimizing", "validating", "extending"])}
these findings to {random.choice(["larger systems", "different domains", "real-world deployment", "clinical trials"])}.
""".strip()

        papers.append(
            {
                "paper_id": f"arxiv:{random.randint(2000, 2400)}.{random.randint(10000, 99999)}",
                "title": f"Advances in {topic[0].title()}: A {random.choice(['Novel', 'Comprehensive', 'Systematic', 'Rigorous'])} Approach to {topic[1].title()}",
                "authors": [f"Author{j}" for j in range(random.randint(2, 6))],
                "abstract": abstract,
                "year": random.randint(2022, 2024),
                "citations": random.randint(0, 150),
            }
        )

    # Return as plain text, not JSON structure
    output = "RESEARCH PAPER SEARCH RESULTS\n" + "=" * 50 + "\n\n"
    for p in papers:
        output += f"[{p['paper_id']}] {p['title']}\n"
        output += f"Authors: {', '.join(p['authors'])} ({p['year']})\n"
        output += f"Citations: {p['citations']}\n\n"
        output += p["abstract"] + "\n\n"
        output += "-" * 50 + "\n\n"

    return {
        "tool": "research_search",
        "result": output,  # Plain text, not JSON!
    }


def generate_code_diff(num_files: int = 15, changes_per_file: int = 20) -> dict:
    """
    Git diff output - every line is unique and important.
    Can't summarize code changes - need exact lines.
    """
    languages = {
        "py": (
            "def ",
            "class ",
            "import ",
            "return ",
            "if ",
            "for ",
            "while ",
            "try:",
            "except:",
            "with ",
        ),
        "ts": (
            "function ",
            "const ",
            "interface ",
            "import ",
            "export ",
            "return ",
            "if ",
            "for ",
            "async ",
            "await ",
        ),
        "go": (
            "func ",
            "type ",
            "import ",
            "return ",
            "if ",
            "for ",
            "defer ",
            "go ",
            "chan ",
            "struct ",
        ),
        "rs": (
            "fn ",
            "struct ",
            "impl ",
            "use ",
            "let ",
            "match ",
            "if ",
            "for ",
            "pub ",
            "async ",
        ),
    }

    diff_output = ""

    for file_idx in range(num_files):
        ext = random.choice(list(languages.keys()))
        keywords = languages[ext]
        filename = f"src/module_{file_idx}/handler.{ext}"

        diff_output += f"diff --git a/{filename} b/{filename}\n"
        diff_output += f"index {hashlib.md5(f'{file_idx}a'.encode()).hexdigest()[:7]}..{hashlib.md5(f'{file_idx}b'.encode()).hexdigest()[:7]} 100644\n"  # nosec B324
        diff_output += f"--- a/{filename}\n"
        diff_output += f"+++ b/{filename}\n"

        line_num = random.randint(10, 50)
        for change_idx in range(changes_per_file):
            # Generate realistic code changes
            keyword = random.choice(keywords)
            var_name = f"{''.join(random.choices(string.ascii_lowercase, k=random.randint(4, 10)))}"
            value = random.randint(1, 1000)

            diff_output += (
                f"@@ -{line_num},{random.randint(3, 7)} +{line_num},{random.randint(3, 7)} @@\n"
            )

            # Context line
            diff_output += f" {random.choice(keywords)}{var_name}_{change_idx}()\n"

            # Removed line
            old_impl = f"{keyword}{var_name} = {value}"
            diff_output += f"-    {old_impl}\n"

            # Added line (different)
            new_impl = f"{keyword}{var_name} = {value + random.randint(1, 100)}"
            diff_output += f"+    {new_impl}\n"

            # More context
            diff_output += f" {random.choice(keywords)}{var_name}_next()\n"

            line_num += random.randint(10, 30)

        diff_output += "\n"

    return {
        "tool": "git_diff",
        "result": diff_output,  # Plain text diff
    }


def generate_encrypted_data(size_kb: int = 20) -> dict:
    """
    Base64 encoded / encrypted content - NO patterns possible.
    This is the ultimate adversarial case for compression.
    """
    # Generate random bytes and base64 encode
    random_bytes = bytes([random.randint(0, 255) for _ in range(size_kb * 1024)])
    import base64

    encoded = base64.b64encode(random_bytes).decode("ascii")

    return {
        "tool": "encrypted_blob",
        "result": {
            "blob_id": f"enc_{hashlib.md5(encoded[:100].encode()).hexdigest()[:16]}",  # nosec B324
            "encryption": "AES-256-GCM",
            "content": encoded,
            "size_bytes": len(random_bytes),
        },
    }


def generate_tiny_dataset(num_items: int = 5) -> dict:
    """
    Very small dataset - not enough data for statistical patterns.
    """
    items = []
    for i in range(num_items):
        items.append(
            {
                "id": i + 1,
                "name": f"Item {chr(65 + i)}",
                "value": random.randint(100, 999),
                "note": f"Unique note for item {i + 1}: {hashlib.md5(str(i).encode()).hexdigest()[:20]}",  # nosec B324
            }
        )

    return {"tool": "tiny_query", "result": {"count": num_items, "items": items}}


def generate_conversation_history(num_messages: int = 50) -> dict:
    """
    Chat conversation - context and flow matter, not just content.
    Each message builds on previous, can't remove context.
    """
    participants = ["Alice", "Bob", "Charlie", "Diana"]

    messages = []
    topics = [
        "the quarterly review",
        "the product launch",
        "the customer feedback",
        "the technical debt",
        "the team restructuring",
    ]
    current_topic = random.choice(topics)

    for i in range(num_messages):
        sender = participants[i % len(participants)]

        # Change topic occasionally
        if random.random() < 0.1:
            current_topic = random.choice(topics)

        # Generate contextual message
        message_templates = [
            f"I think we need to reconsider {current_topic}. The data shows {random.choice(['promising', 'concerning', 'mixed'])} results.",
            f"Building on what {participants[(i - 1) % len(participants)]} said, I'd add that {random.choice(['timing', 'resources', 'alignment'])} is crucial here.",
            f"Let me share some context: when we discussed {current_topic} last month, we agreed on {random.choice(['three priorities', 'a phased approach', 'immediate action'])}.",
            f"I disagree with the previous point. {current_topic.title()} requires {random.choice(['more analysis', 'quick action', 'stakeholder buy-in'])} first.",
            f"To summarize so far: we've covered {random.choice(['the risks', 'the opportunities', 'the constraints'])} of {current_topic}. Next steps?",
            f"Quick question about {current_topic}: have we considered {random.choice(['the budget impact', 'customer perception', 'timeline feasibility'])}?",
            f"I can take the action item on {current_topic}. Will need input from {random.choice(participants)} by {random.choice(['EOD', 'tomorrow', 'Friday'])}.",
        ]

        messages.append(
            {
                "timestamp": f"2024-01-17T{10 + (i // 10):02d}:{(i * 2) % 60:02d}:00Z",
                "sender": sender,
                "message": random.choice(message_templates),
            }
        )

    # Format as conversation transcript
    transcript = "MEETING TRANSCRIPT\n" + "=" * 50 + "\n\n"
    for msg in messages:
        transcript += f"[{msg['timestamp']}] {msg['sender']}:\n"
        transcript += f"  {msg['message']}\n\n"

    return {"tool": "meeting_transcript", "result": transcript}


# =============================================================================
# ADVERSARIAL SCENARIOS
# =============================================================================


@dataclass
class AdversarialScenario:
    name: str
    description: str
    why_adversarial: str
    system_prompt: str
    user_query: str
    tools: list[dict]
    expected_behavior: str  # What we expect to happen


def create_research_synthesis_scenario() -> AdversarialScenario:
    return AdversarialScenario(
        name="Research Paper Synthesis",
        description="Synthesize findings from 10 research papers",
        why_adversarial="Dense academic prose with no structural repetition. Every sentence carries unique meaning. No JSON overhead to compress.",
        system_prompt="""You are a research assistant synthesizing academic papers.
Each paper's findings are important. Don't skip any paper.
Focus on methodology differences and key findings.""",
        user_query="Synthesize these research papers. For each paper, summarize the key methodology and findings. Then identify common themes and contradictions across papers.",
        tools=[generate_research_paper_excerpts(num_papers=10)],
        expected_behavior="Headroom should have minimal compression - prose has no structural redundancy",
    )


def create_code_review_scenario() -> AdversarialScenario:
    return AdversarialScenario(
        name="Code Diff Review",
        description="Review a large code diff across 15 files",
        why_adversarial="Git diffs have minimal redundancy. Each +/- line is unique code. Can't summarize - reviewer needs exact changes.",
        system_prompt="""You are a senior engineer reviewing a pull request.
Every changed line matters. Look for bugs, style issues, and potential problems.
Don't skip any file or change.""",
        user_query="Review this diff carefully. For each file, identify: 1) What changed, 2) Any bugs or issues, 3) Style concerns. Be thorough.",
        tools=[generate_code_diff(num_files=15, changes_per_file=20)],
        expected_behavior="Headroom should struggle - code changes are unique and can't be summarized",
    )


def create_encrypted_analysis_scenario() -> AdversarialScenario:
    return AdversarialScenario(
        name="Encrypted Data Analysis",
        description="Analyze encrypted/encoded data blob",
        why_adversarial="Random/encrypted data has maximum entropy. No patterns exist to compress. This is mathematically incompressible.",
        system_prompt="""You are a data analyst examining an encrypted data blob.
Describe what you observe about the data format and structure.""",
        user_query="Examine this encrypted data blob. What can you tell about its format? Is there any visible structure? What's the encoding?",
        tools=[generate_encrypted_data(size_kb=20)],
        expected_behavior="Headroom CANNOT compress this - random data has no patterns",
    )


def create_small_data_scenario() -> AdversarialScenario:
    return AdversarialScenario(
        name="Tiny Dataset Analysis",
        description="Analyze a very small dataset (5 items)",
        why_adversarial="Too little data for statistical analysis. No patterns emerge with only 5 samples.",
        system_prompt="""You are a data analyst. Analyze this small dataset.""",
        user_query="What patterns do you see in this data? Provide summary statistics and insights.",
        tools=[generate_tiny_dataset(num_items=5)],
        expected_behavior="Headroom has no opportunity - data is already minimal",
    )


def create_conversation_context_scenario() -> AdversarialScenario:
    return AdversarialScenario(
        name="Meeting Context Analysis",
        description="Summarize a 50-message meeting transcript",
        why_adversarial="Conversation requires context. Each message builds on previous ones. Removing messages loses the thread.",
        system_prompt="""You are a meeting analyst. The conversation flow and context matters.
Pay attention to who said what and how opinions evolved.""",
        user_query="Summarize this meeting. Who took which positions? How did the discussion evolve? What were the action items and who owns them?",
        tools=[generate_conversation_history(num_messages=50)],
        expected_behavior="Headroom should preserve conversation flow - context matters",
    )


# =============================================================================
# BENCHMARK RUNNER
# =============================================================================


@dataclass
class BenchmarkResult:
    scenario_name: str
    mode: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    raw_tool_size: int
    compression_ratio: float


def run_scenario(
    client, scenario: AdversarialScenario, mode: str, model: str = "gpt-4o-mini"
) -> BenchmarkResult:
    messages = [
        {"role": "system", "content": scenario.system_prompt},
        {"role": "user", "content": scenario.user_query},
    ]

    # Calculate raw tool output size
    raw_size = 0
    for tool_output in scenario.tools:
        result = tool_output["result"]
        if isinstance(result, str):
            raw_size += len(result)
        else:
            raw_size += len(json.dumps(result))

    # Add tool results
    for tool_output in scenario.tools:
        tool_call_id = f"call_{hashlib.md5(tool_output['tool'].encode()).hexdigest()[:8]}"  # nosec B324
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

        content = tool_output["result"]
        if not isinstance(content, str):
            content = json.dumps(content, indent=2)

        messages.append({"role": "tool", "tool_call_id": tool_call_id, "content": content})

    messages.append({"role": "user", "content": "Please provide your analysis."})

    try:
        response = client.chat.completions.create(model=model, messages=messages, max_tokens=2000)
        input_tokens = response.usage.prompt_tokens
        output_tokens = response.usage.completion_tokens
        cost = (input_tokens * 0.00015 + output_tokens * 0.0006) / 1000
        compression_ratio = 1 - (input_tokens / (raw_size / 4)) if raw_size > 0 else 0

    except Exception as e:
        print(f"   Error: {e}")
        return BenchmarkResult(scenario.name, mode, 0, 0, 0, raw_size, 0)

    return BenchmarkResult(
        scenario.name, mode, input_tokens, output_tokens, cost, raw_size, compression_ratio
    )


def run_adversarial_benchmark(api_key: str = None) -> dict:
    if api_key is None:
        api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY required")

    print("=" * 70)
    print("HEADROOM ADVERSARIAL BENCHMARK")
    print("Testing TRUE worst cases for compression")
    print("=" * 70)

    import tempfile

    from openai import OpenAI

    baseline_client = OpenAI(api_key=api_key)

    if HEADROOM_AVAILABLE:
        db_path = os.path.join(tempfile.gettempdir(), "headroom_adversarial.db")
        headroom_client = HeadroomClient(
            original_client=OpenAI(api_key=api_key),
            provider=OpenAIProvider(),
            store_url=f"sqlite:///{db_path}",
            default_mode="optimize",
        )
    else:
        headroom_client = None

    scenarios = [
        create_research_synthesis_scenario(),
        create_code_review_scenario(),
        create_encrypted_analysis_scenario(),
        create_small_data_scenario(),
        create_conversation_context_scenario(),
    ]

    results = []

    for scenario in scenarios:
        print(f"\n{'=' * 60}")
        print(f"Scenario: {scenario.name}")
        print(f"WHY ADVERSARIAL: {scenario.why_adversarial}")
        print(f"Expected: {scenario.expected_behavior}")
        print("=" * 60)

        # Baseline
        print("\n[1/2] BASELINE...")
        baseline = run_scenario(baseline_client, scenario, "baseline")
        print(
            f"   Raw data: ~{baseline.raw_tool_size:,} chars ({baseline.raw_tool_size // 4:,} est. tokens)"
        )
        print(f"   Input tokens: {baseline.input_tokens:,}")
        print(f"   Cost: ${baseline.cost_usd:.4f}")
        results.append(baseline)

        # Headroom
        if headroom_client:
            print("\n[2/2] HEADROOM...")
            headroom = run_scenario(headroom_client, scenario, "headroom")
            print(f"   Input tokens: {headroom.input_tokens:,}")
            print(f"   Cost: ${headroom.cost_usd:.4f}")
            results.append(headroom)

            if baseline.input_tokens > 0:
                change = (headroom.input_tokens - baseline.input_tokens) / baseline.input_tokens
                print(f"\n   📊 Token change: {change:+.1%}")
                if change > 0:
                    print("   ⚠️  HEADROOM INCREASED TOKENS (overhead > savings)")
                elif change > -0.1:
                    print("   ⚡ Minimal compression (as expected for adversarial data)")
                else:
                    print("   ✓ Still found patterns to compress")

    # Summary
    print("\n" + "=" * 70)
    print("ADVERSARIAL BENCHMARK SUMMARY")
    print("=" * 70)

    print(f"\n{'Scenario':<30} {'Baseline':>12} {'Headroom':>12} {'Change':>12}")
    print("-" * 66)

    baseline_results = [r for r in results if r.mode == "baseline"]
    headroom_results = [r for r in results if r.mode == "headroom"]

    for br in baseline_results:
        hr = next((r for r in headroom_results if r.scenario_name == br.scenario_name), None)
        if hr and br.input_tokens > 0:
            change = (hr.input_tokens - br.input_tokens) / br.input_tokens
            print(
                f"{br.scenario_name:<30} {br.input_tokens:>12,} {hr.input_tokens:>12,} {change:>+11.1%}"
            )

    return {"results": [r.__dict__ for r in results]}


if __name__ == "__main__":
    results = run_adversarial_benchmark()
    with open("adversarial_benchmark_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nResults saved to adversarial_benchmark_results.json")
