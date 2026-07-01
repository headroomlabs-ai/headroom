from __future__ import annotations

import re
from dataclasses import dataclass

CANONICAL_RULE = "Trust kept rows unless you have a concrete gap."

SKILL_RELATIVE_PATH = "headroom/skills/ccr-literacy/SKILL.md"
SKILL_GITHUB_URL = (
    "https://github.com/headroomlabs-ai/headroom/blob/main/headroom/skills/ccr-literacy/SKILL.md"
)

LEARN_SECTION = "CCR Retrieve Literacy"

_THOROUGHNESS_PATTERNS = (
    re.compile(r"\bbe sure\b"),
    re.compile(r"\bmake sure\b"),
    re.compile(r"\bdouble[- ]check\b"),
    re.compile(r"\bthorough(?:ly)?\b"),
    re.compile(r"\bcareful(?:ly)?\b"),
    re.compile(r"\bjust to be safe\b"),
    re.compile(r"\bverify\b"),
)

_CONCRETE_GAP_PATTERNS = (
    re.compile(r"\braw\b"),
    re.compile(r"\boriginal\b"),
    re.compile(r"\bfull\b"),
    re.compile(r"\bentire\b"),
    re.compile(r"\bcomplete\b"),
    re.compile(r"\bexact\b"),
    re.compile(r"\bverbatim\b"),
    re.compile(r"\bomitted\b"),
    re.compile(r"\bquote(?:d)?\b"),
    re.compile(r"\b(?:line|row|record|entry|item|file)\s+\d+\b"),
    re.compile(r"\bquoted?\s+(?:passage|text|section)\b"),
    re.compile(r"\bspecific (?:line|row|record|entry|item|file)\b"),
)


@dataclass(frozen=True)
class RetrieveNeedAssessment:
    should_retrieve: bool
    is_redundant: bool
    reason: str


def render_retrieve_tool_description() -> str:
    return (
        "Retrieve original uncompressed content that was compressed to save tokens. "
        "Trust kept rows unless you have a concrete gap. Retrieve when you need raw, "
        "original, or complete content, or when you need to inspect the original payload "
        "for a specific follow-up. The hash is provided in compression markers like "
        "[N items compressed... hash=abc123]."
    )


def render_retrieve_query_description() -> str:
    return (
        "Optional context hint for the concrete gap you are checking. The hint is recorded "
        "for feedback and stats; retrieval still returns the full original content."
    )


def render_retrieve_cli_guidance() -> str:
    return (
        "Trust kept rows unless you have a concrete gap. Use headroom_retrieve for raw, "
        "original, or complete content, or to inspect the original payload for a specific "
        "follow-up."
    )


def render_retrieve_cli_workflow_steps() -> str:
    return (
        "    4. Claude answers from kept rows unless it has a concrete gap\n"
        "    5. When raw, original, complete, or specific follow-up access is needed, "
        "it calls headroom_retrieve"
    )


def render_retrieve_runtime_prompt_hint() -> str:
    return (
        "Trust kept rows unless you have a concrete gap. Use headroom_retrieve when "
        "you need raw, original, complete, or specific follow-up access to the original payload."
    )


def render_retrieve_system_instructions(hashes: list[str], tool_name: str) -> str:
    hash_list = ", ".join(hashes) if len(hashes) <= 5 else f"{', '.join(hashes[:5])} ..."
    return f"""
## Compressed Context Available

Some tool outputs have been compressed to reduce context size. {CANONICAL_RULE}

Use `{tool_name}` when:
- the user asks for raw, original, full, or exact content
- you need to inspect the original payload for a specific follow-up

Do not retrieve just because the user asked you to be thorough, careful, or to double-check.

**How to retrieve:**
- Call `{tool_name}(hash="<hash>")` to get all original items
- Call `{tool_name}(hash="<hash>", query="concrete gap")` to record the context for feedback. Retrieval still returns the full original content.

**Available hashes:** {hash_list}

Look for markers like `[N items compressed to M. Retrieve more: hash=abc123]`
in tool results to find the hash for each compressed output.
"""


def render_retrieve_skill_markdown() -> str:
    return """# CCR Retrieve Literacy

Trust kept rows unless you have a concrete gap.

## Use `headroom_retrieve` when

- The user explicitly asks for raw, original, full, exact, or omitted content.
- You need to inspect the original payload for a specific follow-up the kept summary cannot answer.
- You need to inspect or quote a specific row, record, line, or file that was compressed away.

## Do not use `headroom_retrieve` when

- The kept summary already answers the question.
- The only reason to retrieve is to be thorough, careful, or to double-check.
- You can answer from the kept rows without looking at the full payload.

## Retrieval style

- Use `query` only as a note about the concrete gap you are checking.
- Current retrieval still returns the full original payload, even when `query` is present.
"""


def render_skill_markdown() -> str:
    return render_retrieve_skill_markdown()


def render_learn_recommendation() -> str:
    return (
        f"- {CANONICAL_RULE}\n"
        "- Use `headroom_retrieve` for raw, original, or complete-content requests, or for "
        "specific follow-up access to the original payload.\n"
        "- Do not retrieve the full payload just because the user asked you to be thorough, "
        "careful, or to double-check."
    )


def classify_retrieve_need(user_text: str, query: str | None = None) -> RetrieveNeedAssessment:
    normalized = _normalize(user_text)
    query_text = (query or "").strip()

    if _matches(_CONCRETE_GAP_PATTERNS, normalized):
        return RetrieveNeedAssessment(
            should_retrieve=True,
            is_redundant=False,
            reason="explicit_raw_or_exact_request",
        )

    if _matches(_THOROUGHNESS_PATTERNS, normalized):
        return RetrieveNeedAssessment(
            should_retrieve=False,
            is_redundant=True,
            reason="thoroughness_without_gap",
        )

    if query_text:
        return RetrieveNeedAssessment(
            should_retrieve=True,
            is_redundant=False,
            reason="specific_followup_with_query_hint",
        )

    return RetrieveNeedAssessment(
        should_retrieve=False,
        is_redundant=False,
        reason="no_clear_gap",
    )


def _normalize(text: str) -> str:
    return " ".join(text.lower().split())


def _matches(patterns: tuple[re.Pattern[str], ...], text: str) -> bool:
    return any(pattern.search(text) for pattern in patterns)
