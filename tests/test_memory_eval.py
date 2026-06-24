"""Tests for the memory evaluation framework."""

from headroom.evals.memory.judge import _parse_judge_response, simple_judge
from headroom.evals.memory.locomo import (
    LOCOMO_CATEGORIES,
    DialogueTurn,
    LoCoMoCase,
    LoCoMoConversation,
    Session,
    get_locomo_stats,
)


class TestLoCoMoDataStructures:
    """Test LoCoMo data structures."""

    def test_dialogue_turn_from_dict(self):
        """Test DialogueTurn parsing."""
        data = {
            "speaker": "Alice",
            "text": "Hello Bob!",
            "dia_id": "D1:1",
        }
        turn = DialogueTurn.from_dict(data)

        assert turn.speaker == "Alice"
        assert turn.text == "Hello Bob!"
        assert turn.dia_id == "D1:1"
        assert turn.image_url is None

    def test_dialogue_turn_with_image(self):
        """Test DialogueTurn with image."""
        data = {
            "speaker": "Bob",
            "text": "Check this out",
            "dia_id": "D1:2",
            "img_file": "http://example.com/img.jpg",
            "blip_caption": "A beautiful sunset",
        }
        turn = DialogueTurn.from_dict(data)

        assert turn.image_url == "http://example.com/img.jpg"
        assert turn.image_caption == "A beautiful sunset"

    def test_dialogue_turn_to_message_format(self):
        """Test message format conversion."""
        turn = DialogueTurn(
            speaker="Alice",
            text="I love Python",
            dia_id="D1:1",
        )
        msg = turn.to_message_format()
        assert msg == "Alice: I love Python"

        # With image
        turn_img = DialogueTurn(
            speaker="Bob",
            text="Look at this",
            dia_id="D1:2",
            image_url="http://example.com/img.jpg",
            image_caption="A dog playing",
        )
        msg_img = turn_img.to_message_format()
        assert "[shares image: A dog playing]" in msg_img

    def test_session_properties(self):
        """Test Session properties."""
        dialogues = [
            DialogueTurn(speaker="Alice", text="Hi", dia_id="D1:1"),
            DialogueTurn(speaker="Bob", text="Hello", dia_id="D1:2"),
        ]
        session = Session(session_num=1, datetime="2024-01-15", dialogues=dialogues)

        assert session.num_turns == 2
        assert "Alice: Hi" in session.text
        assert "Bob: Hello" in session.text

    def test_locomo_case_properties(self):
        """Test LoCoMoCase properties."""
        case = LoCoMoCase(
            question="What is Alice's favorite color?",
            answer="Blue",
            category=1,
            evidence=["D1:5", "D2:3"],
            conversation_id="sample_1",
        )

        assert case.category_name == "single_hop"
        assert case.is_answerable is True

        # Test unanswerable case
        case_na = LoCoMoCase(
            question="What is unknown?",
            answer="N/A",
            category=5,
            evidence=[],
            conversation_id="sample_1",
        )
        assert case_na.is_answerable is False

    def test_locomo_categories(self):
        """Test category definitions."""
        assert LOCOMO_CATEGORIES[1] == "single_hop"
        assert LOCOMO_CATEGORIES[2] == "temporal"
        assert LOCOMO_CATEGORIES[3] == "multi_hop"
        assert LOCOMO_CATEGORIES[4] == "open_domain"
        assert LOCOMO_CATEGORIES[5] == "adversarial"


class TestLoCoMoStats:
    """Test LoCoMo statistics."""

    def test_get_stats_empty(self):
        """Test stats with empty list."""
        stats = get_locomo_stats([])
        assert stats["num_conversations"] == 0
        assert stats["num_qa_pairs"] == 0

    def test_get_stats_with_data(self):
        """Test stats calculation."""
        # Create mock conversation
        dialogues = [
            DialogueTurn(speaker="A", text="Hello", dia_id="D1:1"),
            DialogueTurn(speaker="B", text="Hi there", dia_id="D1:2"),
        ]
        session = Session(session_num=1, datetime="2024-01-15", dialogues=dialogues)

        qa_cases = [
            LoCoMoCase(question="Q1", answer="A1", category=1, evidence=[], conversation_id="s1"),
            LoCoMoCase(question="Q2", answer="A2", category=2, evidence=[], conversation_id="s1"),
        ]

        conv = LoCoMoConversation(
            sample_id="s1",
            speaker_a="Alice",
            speaker_b="Bob",
            sessions=[session],
            qa_cases=qa_cases,
        )

        stats = get_locomo_stats([conv])

        assert stats["num_conversations"] == 1
        assert stats["num_sessions"] == 1
        assert stats["num_turns"] == 2
        assert stats["num_qa_pairs"] == 2
        assert "single_hop" in stats["questions_by_category"]
        assert "temporal" in stats["questions_by_category"]


class TestJudge:
    """Test LLM judge functions."""

    def test_parse_judge_response_standard(self):
        """Test parsing standard judge response."""
        response = """Reasoning: The prediction captures the main point.
Score: 4"""

        score, reasoning = _parse_judge_response(response)
        assert score == 4.0
        assert "main point" in reasoning

    def test_parse_judge_response_with_decimal(self):
        """Test parsing score with decimal."""
        response = """Reasoning: Partially correct.
Score: 3.5"""

        score, reasoning = _parse_judge_response(response)
        assert score == 3.5

    def test_parse_judge_response_clamping(self):
        """Test score clamping to valid range."""
        # Score too high
        response = "Reasoning: Perfect\nScore: 10"
        score, _ = _parse_judge_response(response)
        assert score == 5.0

        # Score too low
        response = "Reasoning: Terrible\nScore: 0"
        score, _ = _parse_judge_response(response)
        assert score == 1.0

    def test_simple_judge_exact_match(self):
        """Test simple judge with exact match."""
        score, reasoning = simple_judge(
            "What color?",
            "Blue",
            "Blue",
        )
        assert score == 5.0
        assert "Exact match" in reasoning

    def test_simple_judge_high_overlap(self):
        """Test simple judge with high F1."""
        score, reasoning = simple_judge(
            "What happened?",
            "Alice went to the store to buy groceries",
            "Alice went to the store for groceries",
        )
        assert score >= 4.0
        assert "F1" in reasoning

    def test_simple_judge_no_overlap(self):
        """Test simple judge with no overlap."""
        score, reasoning = simple_judge(
            "What color?",
            "Blue",
            "The weather is nice",
        )
        assert score == 1.0
        assert "Very low" in reasoning


class TestMemoryEvalConfig:
    """Test MemoryEvalConfig."""

    def test_default_config(self):
        """Test default configuration."""
        from headroom.evals.memory import MemoryEvalConfig

        config = MemoryEvalConfig()

        assert config.n_conversations is None
        assert config.skip_adversarial is True
        assert config.top_k_memories == 10
        assert config.llm_judge_enabled is False
        assert config.f1_threshold == 0.5

    def test_custom_config(self):
        """Test custom configuration."""
        from headroom.evals.memory import MemoryEvalConfig

        config = MemoryEvalConfig(
            n_conversations=5,
            categories=[1, 2],
            top_k_memories=20,
            llm_judge_enabled=True,
            f1_threshold=0.7,
        )

        assert config.n_conversations == 5
        assert config.categories == [1, 2]
        assert config.top_k_memories == 20
        assert config.llm_judge_enabled is True
        assert config.f1_threshold == 0.7


class TestMemoryEvalResult:
    """Test MemoryEvalResult and MemoryEvalSuiteResult."""

    def test_eval_result_to_dict(self):
        """Test result serialization."""
        from headroom.evals.memory.runner import MemoryEvalResult

        case = LoCoMoCase(
            question="What color?",
            answer="Blue",
            category=1,
            evidence=[],
            conversation_id="s1",
        )

        result = MemoryEvalResult(
            case=case,
            predicted_answer="Blue",
            retrieved_memories=["Memory 1", "Memory 2"],
            retrieval_scores=[0.9, 0.8],
            f1_score=1.0,
            exact_match=True,
            is_correct=True,
        )

        d = result.to_dict()
        assert d["question"] == "What color?"
        assert d["ground_truth"] == "Blue"
        assert d["predicted"] == "Blue"
        assert d["f1_score"] == 1.0
        assert d["is_correct"] is True

    def test_suite_result_summary(self):
        """Test suite result summary generation."""
        from headroom.evals.memory.runner import MemoryEvalSuiteResult

        suite_result = MemoryEvalSuiteResult(
            total_cases=100,
            correct_cases=75,
            accuracy=0.75,
            avg_f1_score=0.82,
            exact_match_rate=0.5,
            avg_llm_judge_score=4.2,
            metrics_by_category={
                "single_hop": {"count": 30, "accuracy": 0.9, "avg_f1": 0.88, "correct": 27},
                "temporal": {"count": 25, "accuracy": 0.7, "avg_f1": 0.75, "correct": 18},
            },
            total_duration_seconds=120.5,
            avg_retrieval_latency_ms=15.3,
            avg_generation_latency_ms=250.0,
        )

        summary = suite_result.summary()
        assert "100" in summary
        assert "75" in summary  # Accuracy percentage
        assert "0.820" in summary  # F1 score
        assert "single_hop" in summary
        assert "temporal" in summary
