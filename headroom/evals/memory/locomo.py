"""LoCoMo dataset loader for memory evaluation.

LoCoMo (Long-term Conversational Memory) is a benchmark for evaluating
very long-term conversational memory of LLM agents.

Paper: https://arxiv.org/abs/2402.17753
GitHub: https://github.com/snap-research/locomo

The dataset contains 10 multi-session conversations with:
- ~300 turns per conversation
- ~9K tokens per conversation
- Up to 35 sessions spanning weeks/months
- QA pairs across 5 categories

Categories:
- 1: Single-hop (simple fact recall)
- 2: Temporal (time-based questions)
- 3: Multi-hop (reasoning across memories)
- 4: Open-domain (interpretation required)
- 5: Adversarial (unanswerable) - typically skipped
"""

from __future__ import annotations

import json
import logging
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# LoCoMo dataset URL
LOCOMO_URL = "https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json"

# Category definitions
LOCOMO_CATEGORIES = {
    1: "single_hop",
    2: "temporal",
    3: "multi_hop",
    4: "open_domain",
    5: "adversarial",
}

CATEGORY_DESCRIPTIONS = {
    1: "Simple fact recall from a single evidence source",
    2: "Questions about when something happened",
    3: "Reasoning across multiple evidence sources",
    4: "Interpretation and inference required",
    5: "Questions that cannot be answered (typically skipped)",
}


@dataclass
class DialogueTurn:
    """A single dialogue turn in a conversation."""

    speaker: str
    text: str
    dia_id: str  # e.g., "D1:3" = Session 1, Turn 3
    image_url: str | None = None
    image_caption: str | None = None

    @classmethod
    def from_dict(cls, d: dict) -> DialogueTurn:
        return cls(
            speaker=d["speaker"],
            text=d["text"],
            dia_id=d["dia_id"],
            image_url=d.get("img_file"),
            image_caption=d.get("blip_caption"),
        )

    def to_message_format(self) -> str:
        """Convert to a format suitable for memory storage."""
        msg = f"{self.speaker}: {self.text}"
        if self.image_caption:
            msg += f" [shares image: {self.image_caption}]"
        return msg


@dataclass
class Session:
    """A single session in a conversation."""

    session_num: int
    datetime: str
    dialogues: list[DialogueTurn]

    @property
    def text(self) -> str:
        """Get full session text."""
        return "\n".join(d.to_message_format() for d in self.dialogues)

    @property
    def num_turns(self) -> int:
        return len(self.dialogues)


@dataclass
class LoCoMoCase:
    """A single QA case from LoCoMo."""

    question: str
    answer: str | int | None
    category: int
    evidence: list[str]  # List of dia_ids
    conversation_id: str

    @property
    def category_name(self) -> str:
        return LOCOMO_CATEGORIES.get(self.category, "unknown")

    @property
    def is_answerable(self) -> bool:
        """Check if this question has an answer."""
        return self.answer is not None and self.answer != "N/A"

    @classmethod
    def from_dict(cls, d: dict, conversation_id: str) -> LoCoMoCase:
        return cls(
            question=d["question"],
            answer=d.get("answer"),
            category=d["category"],
            evidence=d.get("evidence", []),
            conversation_id=conversation_id,
        )


@dataclass
class LoCoMoConversation:
    """A complete LoCoMo conversation with multiple sessions."""

    sample_id: str
    speaker_a: str
    speaker_b: str
    sessions: list[Session]
    qa_cases: list[LoCoMoCase]
    event_summaries: dict[str, Any] = field(default_factory=dict)

    @property
    def total_turns(self) -> int:
        return sum(s.num_turns for s in self.sessions)

    @property
    def total_tokens_approx(self) -> int:
        """Approximate token count (chars / 4)."""
        total_chars = sum(len(s.text) for s in self.sessions)
        return total_chars // 4

    @classmethod
    def from_dict(cls, d: dict) -> LoCoMoConversation:
        sample_id = d["sample_id"]
        speaker_a = d["conversation"]["speaker_a"]
        speaker_b = d["conversation"]["speaker_b"]

        # Parse sessions
        sessions = []
        for i in range(1, 100):
            session_key = f"session_{i}"
            datetime_key = f"session_{i}_date_time"

            if session_key not in d["conversation"]:
                break
            if not d["conversation"][session_key]:  # Empty session
                continue

            dialogues = [DialogueTurn.from_dict(turn) for turn in d["conversation"][session_key]]
            sessions.append(
                Session(
                    session_num=i,
                    datetime=d["conversation"].get(datetime_key, ""),
                    dialogues=dialogues,
                )
            )

        # Parse QA cases
        qa_cases = [LoCoMoCase.from_dict(qa, sample_id) for qa in d.get("qa", [])]

        return cls(
            sample_id=sample_id,
            speaker_a=speaker_a,
            speaker_b=speaker_b,
            sessions=sessions,
            qa_cases=qa_cases,
            event_summaries=d.get("event_summary", {}),
        )


@dataclass
class LoCoMoResult:
    """Result of evaluating a single LoCoMo case."""

    case: LoCoMoCase
    predicted_answer: str
    is_correct: bool
    f1_score: float
    exact_match: bool
    llm_judge_score: float | None = None

    def to_dict(self) -> dict:
        return {
            "question": self.case.question,
            "ground_truth": self.case.answer,
            "predicted": self.predicted_answer,
            "category": self.case.category_name,
            "is_correct": self.is_correct,
            "f1_score": self.f1_score,
            "exact_match": self.exact_match,
            "llm_judge_score": self.llm_judge_score,
        }


def download_locomo(cache_dir: Path | None = None) -> Path:
    """Download LoCoMo dataset if not cached.

    Args:
        cache_dir: Directory to cache the dataset. Defaults to ~/.cache/headroom/

    Returns:
        Path to the downloaded JSON file
    """
    if cache_dir is None:
        cache_dir = Path.home() / ".cache" / "headroom"

    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / "locomo10.json"

    if cache_path.exists():
        logger.info(f"Using cached LoCoMo dataset: {cache_path}")
        return cache_path

    logger.info(f"Downloading LoCoMo dataset from {LOCOMO_URL}...")
    urllib.request.urlretrieve(LOCOMO_URL, cache_path)  # nosec B310
    logger.info(f"Downloaded to {cache_path}")

    return cache_path


def load_locomo(
    n_conversations: int | None = None,
    categories: list[int] | None = None,
    skip_adversarial: bool = True,
    cache_dir: Path | None = None,
) -> list[LoCoMoConversation]:
    """Load LoCoMo dataset for memory evaluation.

    Args:
        n_conversations: Number of conversations to load (default: all 10)
        categories: Filter to specific categories (1-5). Default: [1,2,3,4]
        skip_adversarial: Skip category 5 (unanswerable questions). Default: True
        cache_dir: Directory to cache the dataset

    Returns:
        List of LoCoMoConversation objects
    """
    # Default categories (skip adversarial)
    if categories is None:
        categories = [1, 2, 3, 4] if skip_adversarial else [1, 2, 3, 4, 5]

    # Download/load dataset
    cache_path = download_locomo(cache_dir)

    with open(cache_path) as f:
        raw_data = json.load(f)

    # Parse conversations
    conversations = []
    for i, conv_data in enumerate(raw_data):
        if n_conversations is not None and i >= n_conversations:
            break

        conv = LoCoMoConversation.from_dict(conv_data)

        # Filter QA cases by category
        conv.qa_cases = [qa for qa in conv.qa_cases if qa.category in categories]

        conversations.append(conv)

    # Log stats
    total_qa = sum(len(c.qa_cases) for c in conversations)
    total_sessions = sum(len(c.sessions) for c in conversations)
    total_turns = sum(c.total_turns for c in conversations)

    logger.info(
        f"Loaded LoCoMo: {len(conversations)} conversations, "
        f"{total_sessions} sessions, {total_turns} turns, {total_qa} QA pairs"
    )

    # Category breakdown
    cat_counts: dict[int, int] = {}
    for conv in conversations:
        for qa in conv.qa_cases:
            cat_counts[qa.category] = cat_counts.get(qa.category, 0) + 1

    for cat, count in sorted(cat_counts.items()):
        logger.info(f"  Category {cat} ({LOCOMO_CATEGORIES[cat]}): {count} questions")

    return conversations


def get_locomo_stats(conversations: list[LoCoMoConversation]) -> dict:
    """Get statistics about the loaded LoCoMo dataset."""
    total_qa = sum(len(c.qa_cases) for c in conversations)
    total_sessions = sum(len(c.sessions) for c in conversations)
    total_turns = sum(c.total_turns for c in conversations)
    total_tokens = sum(c.total_tokens_approx for c in conversations)

    cat_counts: dict[str, int] = {}
    for conv in conversations:
        for qa in conv.qa_cases:
            cat_name = qa.category_name
            cat_counts[cat_name] = cat_counts.get(cat_name, 0) + 1

    return {
        "num_conversations": len(conversations),
        "num_sessions": total_sessions,
        "num_turns": total_turns,
        "num_qa_pairs": total_qa,
        "approx_tokens": total_tokens,
        "questions_by_category": cat_counts,
    }
