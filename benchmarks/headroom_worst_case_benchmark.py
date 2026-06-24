"""
Headroom Worst-Case Benchmark: Where Compression Hurts

This benchmark tests scenarios where Headroom's statistical compression
may NOT be beneficial - to understand the limits of the approach.

Worst cases for Headroom:
1. Highly unique data (no patterns to compress)
2. Data where every item is equally important
3. Data where subtle differences matter
4. Small datasets (not enough data for statistics)
5. Data where you need EXACT recall (audit/legal)
"""

import hashlib
import json
import os
import random
import time
from dataclasses import dataclass
from typing import Any

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
# WORST-CASE DATA GENERATORS
# =============================================================================


def generate_unique_support_tickets(num_tickets: int = 50) -> dict:
    """
    Customer support tickets where EVERY ticket is unique and important.
    No redundancy - each customer has a different problem.

    This is hard for Headroom because:
    - No repeated patterns to compress
    - Every ticket needs attention
    - Can't safely remove any ticket
    """
    products = ["Pro Plan", "Enterprise", "Starter", "Team", "Individual"]
    issues = [
        "billing discrepancy of ${amount} on invoice #{inv}",
        "cannot access feature '{feature}' despite paying for it",
        "data export failing with error code {code}",
        "SSO integration with {provider} not working",
        "API rate limits hitting at {rate}/min instead of promised {expected}/min",
        "webhook deliveries delayed by {hours} hours",
        "user {user} locked out after password reset",
        "mobile app crashing on {device} with iOS {version}",
        "search returning wrong results for query '{query}'",
        "file upload stuck at {percent}% for files over {size}MB",
        "notification emails going to spam for domain {domain}",
        "timezone showing {wrong_tz} instead of {correct_tz}",
        "dashboard metrics {days} days out of date",
        "cannot downgrade from {from_plan} to {to_plan}",
        "GDPR data deletion request not completing for user {user_id}",
    ]

    severities = ["critical", "high", "medium", "low"]

    tickets = []
    for i in range(num_tickets):
        # Each ticket is genuinely unique
        issue_template = issues[i % len(issues)]
        issue = issue_template.format(
            amount=random.randint(50, 5000),
            inv=random.randint(10000, 99999),
            feature=random.choice(
                ["advanced analytics", "custom domains", "API access", "SSO", "audit logs"]
            ),
            code=f"ERR_{random.randint(1000, 9999)}",
            provider=random.choice(["Okta", "Azure AD", "Google Workspace", "OneLogin"]),
            rate=random.randint(100, 500),
            expected=random.randint(1000, 5000),
            hours=random.randint(1, 48),
            user=f"user_{random.randint(1000, 9999)}@company{random.randint(1, 100)}.com",
            device=random.choice(["iPhone 15", "iPhone 14", "iPad Pro", "iPhone 13"]),
            version=random.choice(["17.2", "17.1", "16.5", "16.4"]),
            query=random.choice(
                ["quarterly report", "user metrics", "revenue data", "team performance"]
            ),
            percent=random.randint(45, 95),
            size=random.randint(10, 500),
            domain=f"company{random.randint(1, 500)}.com",
            wrong_tz=random.choice(["UTC", "PST", "EST"]),
            correct_tz=random.choice(["CET", "JST", "IST"]),
            days=random.randint(2, 14),
            from_plan=random.choice(["Enterprise", "Pro"]),
            to_plan=random.choice(["Starter", "Team"]),
            user_id=f"usr_{hashlib.md5(str(i).encode()).hexdigest()[:8]}",  # nosec B324
        )

        tickets.append(
            {
                "ticket_id": f"TKT-{20000 + i}",
                "customer": {
                    "id": f"cust_{hashlib.md5(f'customer{i}'.encode()).hexdigest()[:8]}",  # nosec B324
                    "name": f"Customer {i + 1}",
                    "company": f"Company {chr(65 + (i % 26))}{i // 26 + 1} Inc.",
                    "plan": random.choice(products),
                    "mrr": random.randint(99, 9999),
                    "account_age_days": random.randint(30, 1500),
                },
                "issue": issue,
                "severity": random.choice(severities),
                "created_at": f"2024-01-{random.randint(10, 17):02d}T{random.randint(0, 23):02d}:{random.randint(0, 59):02d}:00Z",
                "last_response": f"2024-01-{random.randint(15, 17):02d}T{random.randint(0, 23):02d}:{random.randint(0, 59):02d}:00Z",
                "response_count": random.randint(1, 8),
                "tags": random.sample(
                    ["billing", "technical", "feature-request", "bug", "urgent", "escalated"],
                    k=random.randint(1, 3),
                ),
                "assignee": None,  # Unassigned - needs triage
            }
        )

    return {
        "tool": "support_queue",
        "result": {"queue": "unassigned", "total_tickets": num_tickets, "tickets": tickets},
    }


def generate_unique_error_traces(num_traces: int = 30) -> dict:
    """
    Unique stack traces where each error is different.

    This is hard for Headroom because:
    - Each stack trace has different functions, line numbers
    - Each error message is unique
    - All errors need investigation
    """
    languages = ["python", "javascript", "go", "java"]

    traces = []
    for i in range(num_traces):
        lang = random.choice(languages)

        if lang == "python":
            trace = generate_python_trace(i)
        elif lang == "javascript":
            trace = generate_js_trace(i)
        elif lang == "go":
            trace = generate_go_trace(i)
        else:
            trace = generate_java_trace(i)

        traces.append(
            {
                "error_id": f"err_{hashlib.md5(str(i).encode()).hexdigest()[:12]}",  # nosec B324
                "timestamp": f"2024-01-17T{10 + (i % 12):02d}:{(i * 7) % 60:02d}:00Z",
                "service": random.choice(["api", "worker", "scheduler", "gateway"]),
                "environment": "production",
                "language": lang,
                "error_type": trace["error_type"],
                "message": trace["message"],
                "stack_trace": trace["stack"],
                "context": {
                    "user_id": f"user_{random.randint(10000, 99999)}",
                    "request_id": hashlib.md5(f"req{i}".encode()).hexdigest()[:16],  # nosec B324
                    "endpoint": trace.get("endpoint", "/api/unknown"),
                },
                "occurrence_count": random.randint(1, 5),  # Low count - each is unique
            }
        )

    return {
        "tool": "error_tracker",
        "result": {"time_range": "last_24h", "total_unique_errors": num_traces, "errors": traces},
    }


def generate_python_trace(seed: int) -> dict:
    """Generate a unique Python stack trace."""
    error_types = [
        ("ValueError", f"Invalid value for parameter 'config_{seed}': expected int, got str"),
        ("KeyError", f"'{random.choice(['user', 'account', 'session', 'token'])}_{seed}'"),
        ("TypeError", f"unsupported operand type(s) for +: 'NoneType' and 'str' in field_{seed}"),
        ("AttributeError", f"'NoneType' object has no attribute 'process_{seed}'"),
        ("RuntimeError", f"Maximum recursion depth exceeded in handler_{seed}"),
        ("ConnectionError", f"Connection refused to service_{seed}:8080"),
        (
            "TimeoutError",
            f"Operation timed out after {random.randint(30, 120)}s waiting for resource_{seed}",
        ),
    ]

    error_type, message = random.choice(error_types)

    functions = [
        f"process_request_{seed}",
        f"validate_input_{seed % 10}",
        f"transform_data_{seed}",
        f"save_to_db_{seed % 5}",
        f"send_notification_{seed}",
    ]

    stack_lines = []
    for j, func in enumerate(random.sample(functions, k=random.randint(3, 5))):
        line_no = random.randint(50, 500)
        file_path = f"/app/services/module_{seed % 20}/{func.split('_')[0]}.py"
        stack_lines.append(f'  File "{file_path}", line {line_no}, in {func}')
        stack_lines.append(f"    result = self.handler_{j}(data)")

    return {
        "error_type": error_type,
        "message": message,
        "stack": "\n".join(stack_lines),
        "endpoint": f"/api/v{random.randint(1, 3)}/{random.choice(['users', 'orders', 'products'])}/{seed}",
    }


def generate_js_trace(seed: int) -> dict:
    """Generate a unique JavaScript stack trace."""
    error_types = [
        (
            "TypeError",
            f"Cannot read property '{random.choice(['map', 'filter', 'length', 'data'])}' of undefined",
        ),
        ("ReferenceError", f"config_{seed} is not defined"),
        ("SyntaxError", f"Unexpected token in JSON at position {random.randint(100, 1000)}"),
        ("RangeError", f"Maximum call stack size exceeded in recursive_{seed}"),
    ]

    error_type, message = random.choice(error_types)

    stack = f"""    at processData_{seed} (/app/src/handlers/processor_{seed % 10}.js:{random.randint(50, 200)}:15)
    at async handleRequest_{seed} (/app/src/routes/api_{seed % 5}.js:{random.randint(20, 100)}:23)
    at async Router.dispatch (/app/node_modules/express/router.js:142:12)
    at async Layer.handle (/app/node_modules/express/layer.js:95:5)"""

    return {
        "error_type": error_type,
        "message": message,
        "stack": stack,
        "endpoint": f"/api/{random.choice(['graphql', 'rest', 'webhook'])}/{seed}",
    }


def generate_go_trace(seed: int) -> dict:
    """Generate a unique Go stack trace."""
    error_types = [
        ("panic", f"runtime error: index out of range [{seed}] with length {seed - 1}"),
        ("panic", "runtime error: invalid memory address or nil pointer dereference"),
        ("error", f"context deadline exceeded after {random.randint(5, 30)}s"),
        ("error", f"connection refused to database_{seed % 3}:5432"),
    ]

    error_type, message = random.choice(error_types)

    stack = f"""goroutine {random.randint(1, 100)} [running]:
main.processHandler_{seed}(0xc0001{seed:04x}, 0x{random.randint(1000, 9999):x})
	/app/internal/handlers/handler_{seed % 10}.go:{random.randint(50, 200)} +0x{random.randint(100, 999):x}
main.(*Server).ServeHTTP_{seed}(0xc000{seed:04x}, 0x7f{random.randint(1000, 9999):x})
	/app/internal/server/server.go:{random.randint(80, 150)} +0x{random.randint(100, 500):x}"""

    return {
        "error_type": error_type,
        "message": message,
        "stack": stack,
    }


def generate_java_trace(seed: int) -> dict:
    """Generate a unique Java stack trace."""
    error_types = [
        ("NullPointerException", f"Cannot invoke method on null object in Service_{seed}"),
        ("IllegalArgumentException", f"Parameter 'id_{seed}' cannot be negative"),
        ("SQLException", f"Connection to database_{seed % 3} timed out"),
        ("OutOfMemoryError", f"Java heap space exhausted processing batch_{seed}"),
    ]

    error_type, message = random.choice(error_types)

    stack = f"""java.lang.{error_type}: {message}
	at com.app.services.Handler{seed}.process(Handler{seed}.java:{random.randint(50, 200)})
	at com.app.controllers.Api{seed % 10}Controller.handle(Api{seed % 10}Controller.java:{random.randint(30, 100)})
	at org.springframework.web.servlet.FrameworkServlet.service(FrameworkServlet.java:897)
	at javax.servlet.http.HttpServlet.service(HttpServlet.java:750)"""

    return {
        "error_type": error_type,
        "message": message,
        "stack": stack,
    }


def generate_medical_records(num_patients: int = 25) -> dict:
    """
    Medical records where EVERY detail matters.

    This is hard for Headroom because:
    - Similar symptoms can have different diagnoses
    - Missing any detail could be dangerous
    - "Repetitive" info (vitals) is actually critical data
    """
    conditions = [
        "Type 2 Diabetes",
        "Hypertension",
        "Asthma",
        "GERD",
        "Anxiety Disorder",
        "Hypothyroidism",
        "Chronic Back Pain",
        "Migraine",
        "Allergic Rhinitis",
        "Depression",
    ]

    medications = [
        "Metformin 500mg",
        "Lisinopril 10mg",
        "Omeprazole 20mg",
        "Albuterol inhaler",
        "Sertraline 50mg",
        "Levothyroxine 50mcg",
        "Ibuprofen 400mg PRN",
        "Sumatriptan 50mg PRN",
        "Loratadine 10mg",
    ]

    records = []
    for i in range(num_patients):
        # Each patient has a unique combination of conditions, meds, vitals
        patient_conditions = random.sample(conditions, k=random.randint(1, 4))
        patient_meds = random.sample(medications, k=random.randint(1, 5))

        # Vitals - these look "similar" but each patient's baseline is different
        systolic = random.randint(110, 160)
        diastolic = random.randint(70, 100)

        records.append(
            {
                "patient_id": f"PT-{100000 + i}",
                "name": f"Patient {chr(65 + (i % 26))}{chr(65 + ((i // 26) % 26))}",
                "age": random.randint(25, 85),
                "sex": random.choice(["M", "F"]),
                "visit_date": f"2024-01-{random.randint(15, 17):02d}",
                "chief_complaint": random.choice(
                    [
                        f"Chest pain radiating to left arm for {random.randint(1, 6)} hours",
                        f"Shortness of breath worsening over {random.randint(1, 14)} days",
                        "Severe headache, worst of life, sudden onset",
                        f"Abdominal pain, {random.choice(['RLQ', 'LLQ', 'epigastric'])}, {random.randint(1, 72)} hours",
                        f"Dizziness and {random.choice(['syncope', 'near-syncope'])} today",
                        f"Fever {random.randint(100, 104)}°F for {random.randint(1, 5)} days",
                        "Medication refill - stable on current regimen",
                        f"Follow-up for recent {random.choice(['hospitalization', 'procedure', 'diagnosis'])}",
                    ]
                ),
                "vitals": {
                    "bp": f"{systolic}/{diastolic}",
                    "hr": random.randint(60, 110),
                    "temp": round(random.uniform(97.5, 100.5), 1),
                    "resp": random.randint(12, 22),
                    "spo2": random.randint(94, 100),
                },
                "conditions": patient_conditions,
                "medications": patient_meds,
                "allergies": random.sample(
                    ["Penicillin", "Sulfa", "NSAIDs", "Latex", "None"], k=random.randint(1, 2)
                ),
                "notes": f"Patient presents with {random.choice(['acute', 'chronic', 'worsening', 'stable'])} symptoms. "
                f"Last seen {random.randint(1, 12)} months ago. "
                f"Compliance with medications: {random.choice(['good', 'fair', 'poor'])}. "
                f"Social history: {random.choice(['non-smoker', 'former smoker', 'current smoker'])}, "
                f"{random.choice(['no alcohol', 'occasional alcohol', 'daily alcohol'])}.",
            }
        )

    return {
        "tool": "ehr_query",
        "result": {
            "query": "today's patients",
            "total_patients": num_patients,
            "patients": records,
        },
    }


def generate_legal_discovery_docs(num_docs: int = 40) -> dict:
    """
    Legal discovery documents where EVERY document must be reviewed.

    This is hard for Headroom because:
    - Can't skip any document - legal requirement
    - "Similar" emails might have crucial differences
    - Need exact quotes, not summaries
    """
    senders = [f"person{i}@company.com" for i in range(1, 20)]
    subjects = [
        "Re: Q4 projections discussion",
        "Fw: Board meeting notes",
        "Re: Re: Customer complaint handling",
        "Meeting tomorrow",
        "Urgent: Need your input",
        "Re: Project timeline update",
        "Fw: Legal review needed",
        "Re: Re: Re: Budget approval",
        "Quick question",
        "Following up",
    ]

    docs = []
    for i in range(num_docs):
        sender = random.choice(senders)
        recipient = random.choice([s for s in senders if s != sender])

        # Each email has unique content that could be relevant
        body_templates = [
            f"As we discussed in the meeting on {random.randint(1, 28)}/{random.randint(1, 12)}, the numbers for Q{random.randint(1, 4)} show {random.choice(['concerning', 'promising', 'unexpected'])} trends. I think we should {random.choice(['proceed', 'hold off', 'reconsider'])} with the {random.choice(['merger', 'acquisition', 'expansion', 'restructuring'])} plan.",
            f"I'm forwarding this because I think you should be aware. The customer in region {random.choice(['APAC', 'EMEA', 'Americas'])} has raised {random.choice(['serious', 'minor', 'recurring'])} concerns about our {random.choice(['pricing', 'service', 'product quality'])}. Can we discuss {random.choice(['today', 'tomorrow', 'this week'])}?",
            f"Following up on your question - the {random.choice(['contract', 'agreement', 'terms'])} with {random.choice(['Vendor A', 'Vendor B', 'the client'])} does {random.choice(['', 'not '])}allow for {random.choice(['early termination', 'price adjustment', 'scope changes'])}. See clause {random.randint(1, 20)}.{random.randint(1, 9)}.",
            f"Quick update: the {random.choice(['audit', 'review', 'investigation'])} team found {random.choice(['no issues', 'minor discrepancies', 'significant concerns'])} in the {random.choice(['financial records', 'compliance documents', 'HR files'])} for {random.choice(['Q1', 'Q2', 'Q3', 'Q4'])} {random.randint(2021, 2023)}.",
            f"I need to flag something - the {random.choice(['employee', 'manager', 'director'])} in {random.choice(['sales', 'marketing', 'engineering'])} mentioned that {random.choice(['deadlines were missed', 'budgets were exceeded', 'protocols were bypassed'])}. Not sure if this is relevant to the case but wanted you to know.",
        ]

        docs.append(
            {
                "doc_id": f"DOC-{30000 + i}",
                "type": "email",
                "date": f"2023-{random.randint(1, 12):02d}-{random.randint(1, 28):02d}T{random.randint(8, 18):02d}:{random.randint(0, 59):02d}:00Z",
                "from": sender,
                "to": [recipient],
                "cc": random.sample(senders, k=random.randint(0, 3)),
                "subject": random.choice(subjects),
                "body": random.choice(body_templates),
                "attachments": [
                    f"document_{random.randint(1, 100)}.{random.choice(['pdf', 'xlsx', 'docx'])}"
                ]
                if random.random() > 0.6
                else [],
                "flags": random.sample(
                    ["privileged", "responsive", "hot", "needs_review"], k=random.randint(0, 2)
                ),
                "reviewed": False,
            }
        )

    return {
        "tool": "discovery_search",
        "result": {"case": "Matter 2024-CV-1234", "total_documents": num_docs, "documents": docs},
    }


# =============================================================================
# WORST-CASE SCENARIOS
# =============================================================================


@dataclass
class WorstCaseScenario:
    """A scenario where Headroom might struggle."""

    name: str
    description: str
    why_hard: str
    system_prompt: str
    user_query: str
    tools: list[dict]
    validation_questions: list[str]  # Specific questions to test recall


def create_support_triage_scenario() -> WorstCaseScenario:
    """
    Support queue where every ticket is unique and important.
    """
    return WorstCaseScenario(
        name="Support Ticket Triage",
        description="Triage 50 unique customer support tickets",
        why_hard="Every ticket is unique - no patterns to compress. Each customer's problem is different. Missing any ticket means a customer gets ignored.",
        system_prompt="""You are a support team lead triaging tickets.
Every ticket represents a real customer with a real problem.
You must acknowledge ALL tickets and prioritize them appropriately.
Do not skip or summarize away any customer's issue.""",
        user_query="Please review all tickets in the queue and give me a prioritized action plan. I need to know about EVERY ticket - which ones need immediate attention, which can wait, and which need escalation.",
        tools=[
            generate_unique_support_tickets(num_tickets=50),
        ],
        validation_questions=[
            "How many critical severity tickets are there?",
            "Which Enterprise customers have open tickets?",
            "List all tickets related to billing issues",
            "Which tickets mention SSO or authentication problems?",
        ],
    )


def create_error_investigation_scenario() -> WorstCaseScenario:
    """
    Unique errors where each needs individual investigation.
    """
    return WorstCaseScenario(
        name="Production Error Investigation",
        description="Investigate 30 unique production errors",
        why_hard="Each error has a different stack trace, different service, different root cause. Can't group them - each needs individual attention.",
        system_prompt="""You are an on-call engineer investigating production errors.
Each error is unique and may indicate a different underlying issue.
Do not group or summarize - each error needs specific investigation.""",
        user_query="Review all errors from the last 24 hours. For EACH error, tell me: what service, what type, and what you think the root cause might be. Don't group them - I need to know about each one individually.",
        tools=[
            generate_unique_error_traces(num_traces=30),
        ],
        validation_questions=[
            "How many Python errors vs JavaScript errors?",
            "Which services have the most errors?",
            "List all NullPointerException or nil pointer errors",
            "Which errors are related to database connections?",
        ],
    )


def create_medical_review_scenario() -> WorstCaseScenario:
    """
    Medical records where every detail matters.
    """
    return WorstCaseScenario(
        name="Medical Record Review",
        description="Review 25 patients for today's clinic",
        why_hard="Every patient's vitals, conditions, and medications are unique. 'Similar' symptoms could mean very different things. Can't summarize - details save lives.",
        system_prompt="""You are a physician reviewing today's patient list.
Every patient's details matter - similar symptoms may need different treatment.
Pay attention to vital signs, medication lists, and allergies.
Never assume two patients with similar complaints have the same issue.""",
        user_query="Review all patients on today's schedule. Flag any concerning vitals, potential drug interactions, or high-acuity complaints. Give me a brief on EACH patient.",
        tools=[
            generate_medical_records(num_patients=25),
        ],
        validation_questions=[
            "Which patients have BP over 140 systolic?",
            "Which patients are on Metformin?",
            "List patients with chest pain or cardiac symptoms",
            "Which patients have drug allergies we should note?",
        ],
    )


def create_legal_discovery_scenario() -> WorstCaseScenario:
    """
    Legal documents where completeness is required.
    """
    return WorstCaseScenario(
        name="Legal Discovery Review",
        description="Review 40 documents for legal discovery",
        why_hard="Legal requirement to review EVERY document. Similar-looking emails may have crucial differences. Need exact recall - summaries aren't acceptable in court.",
        system_prompt="""You are a legal assistant reviewing discovery documents.
EVERY document must be accounted for - missing one could be sanctions.
Pay attention to dates, senders, and specific language used.
Similar documents may have legally significant differences.""",
        user_query="Review all documents and categorize them. For each document, note: the date, sender, key topics, and whether it seems relevant to the case. I need a complete accounting.",
        tools=[
            generate_legal_discovery_docs(num_docs=40),
        ],
        validation_questions=[
            "How many documents mention 'audit' or 'investigation'?",
            "List all documents with attachments",
            "Which documents are flagged as 'privileged'?",
            "How many documents were sent in Q4 2023?",
        ],
    )


# =============================================================================
# BENCHMARK RUNNER
# =============================================================================


@dataclass
class BenchmarkResult:
    """Result from running a scenario."""

    scenario_name: str
    mode: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    latency_ms: float
    answer: str
    validation_scores: dict  # Scores for each validation question


def count_tokens(text: str) -> int:
    """Simple token estimation."""
    return len(text) // 4


def validate_answer(answer: str, scenario: WorstCaseScenario) -> dict:
    """
    Check if the answer addresses all validation questions.
    Returns dict of question -> (found keywords, score).
    """
    scores = {}
    answer_lower = answer.lower()

    for question in scenario.validation_questions:
        # Extract key terms from question
        key_terms = [w for w in question.lower().split() if len(w) > 4]
        found = sum(1 for term in key_terms if term in answer_lower)
        score = found / len(key_terms) if key_terms else 0
        scores[question] = {
            "terms_found": found,
            "terms_total": len(key_terms),
            "score": round(score, 2),
        }

    return scores


def run_scenario(
    client: Any, scenario: WorstCaseScenario, mode: str, model: str = "gpt-4o-mini"
) -> BenchmarkResult:
    """Run a single scenario."""

    messages = [
        {"role": "system", "content": scenario.system_prompt},
        {"role": "user", "content": scenario.user_query},
    ]

    # Add tool results with proper format
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
        messages.append(
            {
                "role": "tool",
                "tool_call_id": tool_call_id,
                "content": json.dumps(tool_output["result"], indent=2),
            }
        )

    messages.append({"role": "user", "content": "Please provide your complete analysis now."})

    start = time.time()

    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=4000,  # Allow longer responses
        )
        latency = (time.time() - start) * 1000

        answer = response.choices[0].message.content
        input_tokens = response.usage.prompt_tokens
        output_tokens = response.usage.completion_tokens

        # GPT-4o-mini pricing
        cost = (input_tokens * 0.00015 + output_tokens * 0.0006) / 1000

        validation_scores = validate_answer(answer, scenario)

    except Exception as e:
        print(f"   Error: {e}")
        return BenchmarkResult(
            scenario_name=scenario.name,
            mode=mode,
            input_tokens=count_tokens(json.dumps(messages)),
            output_tokens=0,
            cost_usd=0,
            latency_ms=0,
            answer=f"Error: {e}",
            validation_scores={},
        )

    return BenchmarkResult(
        scenario_name=scenario.name,
        mode=mode,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cost_usd=cost,
        latency_ms=latency,
        answer=answer,
        validation_scores=validation_scores,
    )


def run_worst_case_benchmark(api_key: str = None) -> dict:
    """Run the complete worst-case benchmark."""

    if api_key is None:
        api_key = os.environ.get("OPENAI_API_KEY")

    if not api_key:
        raise ValueError("OPENAI_API_KEY required")

    print("=" * 70)
    print("HEADROOM WORST-CASE BENCHMARK")
    print("Testing scenarios where compression may hurt performance")
    print("=" * 70)

    # Create clients
    import tempfile

    from openai import OpenAI

    baseline_client = OpenAI(api_key=api_key)

    if HEADROOM_AVAILABLE:
        db_path = os.path.join(tempfile.gettempdir(), "headroom_worst_case.db")
        headroom_client = HeadroomClient(
            original_client=OpenAI(api_key=api_key),
            provider=OpenAIProvider(),
            store_url=f"sqlite:///{db_path}",
            default_mode="optimize",
        )
    else:
        print("WARNING: Headroom not available, running baseline only")
        headroom_client = None

    scenarios = [
        create_support_triage_scenario(),
        create_error_investigation_scenario(),
        create_medical_review_scenario(),
        create_legal_discovery_scenario(),
    ]

    results = []

    for scenario in scenarios:
        print(f"\n{'=' * 60}")
        print(f"Scenario: {scenario.name}")
        print(f"Description: {scenario.description}")
        print(f"WHY THIS IS HARD: {scenario.why_hard}")
        print("=" * 60)

        # Calculate raw size
        raw_size = sum(len(json.dumps(t["result"], indent=2)) for t in scenario.tools)
        print(f"\nRaw tool output size: {raw_size:,} chars (~{raw_size // 4:,} tokens)")

        # Run baseline
        print("\n[1/2] Running BASELINE...")
        baseline_result = run_scenario(baseline_client, scenario, "baseline")
        print(f"   Input tokens: {baseline_result.input_tokens:,}")
        print(f"   Output tokens: {baseline_result.output_tokens:,}")
        print(f"   Cost: ${baseline_result.cost_usd:.4f}")

        avg_baseline_score = (
            sum(v["score"] for v in baseline_result.validation_scores.values())
            / len(baseline_result.validation_scores)
            if baseline_result.validation_scores
            else 0
        )
        print(f"   Validation score: {avg_baseline_score:.1%}")

        results.append(baseline_result)

        # Run Headroom
        if headroom_client:
            print("\n[2/2] Running HEADROOM...")
            headroom_result = run_scenario(headroom_client, scenario, "headroom")
            print(f"   Input tokens: {headroom_result.input_tokens:,}")
            print(f"   Output tokens: {headroom_result.output_tokens:,}")
            print(f"   Cost: ${headroom_result.cost_usd:.4f}")

            avg_headroom_score = (
                sum(v["score"] for v in headroom_result.validation_scores.values())
                / len(headroom_result.validation_scores)
                if headroom_result.validation_scores
                else 0
            )
            print(f"   Validation score: {avg_headroom_score:.1%}")

            results.append(headroom_result)

            # Compare
            if baseline_result.input_tokens > 0:
                token_change = (
                    headroom_result.input_tokens - baseline_result.input_tokens
                ) / baseline_result.input_tokens
                quality_change = avg_headroom_score - avg_baseline_score

                print("\n   📊 COMPARISON:")
                print(
                    f"   Token change: {token_change:+.1%} ({'saved' if token_change < 0 else 'INCREASED'})"
                )
                print(
                    f"   Quality change: {quality_change:+.1%} ({'preserved' if quality_change >= -0.1 else 'DEGRADED'})"
                )

                if quality_change < -0.1:
                    print("   ⚠️  WARNING: Quality degraded significantly!")

    # Summary
    print("\n" + "=" * 70)
    print("WORST-CASE BENCHMARK SUMMARY")
    print("=" * 70)

    baseline_results = [r for r in results if r.mode == "baseline"]
    headroom_results = [r for r in results if r.mode == "headroom"]

    print(f"\n{'Scenario':<30} {'Baseline Tokens':>15} {'Headroom Tokens':>15} {'Quality Δ':>12}")
    print("-" * 72)

    for br in baseline_results:
        hr = next((r for r in headroom_results if r.scenario_name == br.scenario_name), None)
        if hr:
            b_score = (
                sum(v["score"] for v in br.validation_scores.values()) / len(br.validation_scores)
                if br.validation_scores
                else 0
            )
            h_score = (
                sum(v["score"] for v in hr.validation_scores.values()) / len(hr.validation_scores)
                if hr.validation_scores
                else 0
            )
            quality_delta = h_score - b_score
            print(
                f"{br.scenario_name:<30} {br.input_tokens:>15,} {hr.input_tokens:>15,} {quality_delta:>+11.1%}"
            )

    return {
        "baseline": [
            {
                "scenario": r.scenario_name,
                "tokens": r.input_tokens,
                "cost": r.cost_usd,
                "validation": r.validation_scores,
            }
            for r in baseline_results
        ],
        "headroom": [
            {
                "scenario": r.scenario_name,
                "tokens": r.input_tokens,
                "cost": r.cost_usd,
                "validation": r.validation_scores,
            }
            for r in headroom_results
        ],
    }


if __name__ == "__main__":
    results = run_worst_case_benchmark()

    with open("worst_case_benchmark_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\nResults saved to worst_case_benchmark_results.json")
