"""Real-world LLM evaluation tests for compression efficacy.

These tests use actual LLM calls to validate that:
1. Compressed content is still understandable
2. LLM can identify what data exists (for CCR retrieval)
3. Structure preservation enables meaningful reasoning

Run with: pytest tests/test_compression/test_llm_eval.py -v -s

Requires OPENAI_API_KEY environment variable.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass

import pytest

from headroom.compression.detector import ContentType
from headroom.compression.universal import (
    UniversalCompressor,
    UniversalCompressorConfig,
)

# Skip all tests if no API key
pytestmark = pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY not set - skipping LLM eval tests",
)


# =============================================================================
# Test Fixtures
# =============================================================================

PRODUCT_CATALOG = json.dumps(
    {
        "catalog": {
            "products": [
                {
                    "id": "prod_001",
                    "sku": "LAPTOP-PRO-15",
                    "name": "ProBook Laptop 15-inch",
                    "category": "electronics",
                    "price": 1299.99,
                    "currency": "USD",
                    "description": "High-performance laptop with 16GB RAM, 512GB SSD, Intel i7 processor. "
                    "Perfect for professionals and power users who need reliable computing power "
                    "for demanding tasks like video editing, software development, and data analysis. "
                    "Features include backlit keyboard, fingerprint reader, and Thunderbolt 4 ports.",
                    "specs": {
                        "processor": "Intel Core i7-1260P",
                        "ram": "16GB DDR5",
                        "storage": "512GB NVMe SSD",
                        "display": "15.6-inch FHD IPS",
                        "battery": "72Wh",
                        "weight": "1.8kg",
                    },
                    "stock": 45,
                    "rating": 4.7,
                    "reviews_count": 234,
                },
                {
                    "id": "prod_002",
                    "sku": "HEADPHONES-NC-100",
                    "name": "NoiseCanceller Pro Headphones",
                    "category": "audio",
                    "price": 349.99,
                    "currency": "USD",
                    "description": "Premium wireless headphones with industry-leading active noise cancellation. "
                    "Immerse yourself in crystal-clear audio with 30-hour battery life and quick charge "
                    "capability. Comfortable memory foam ear cushions make these perfect for long listening "
                    "sessions, flights, or focused work environments.",
                    "specs": {
                        "driver_size": "40mm",
                        "frequency_response": "20Hz-20kHz",
                        "battery_life": "30 hours",
                        "bluetooth": "5.2",
                        "weight": "250g",
                    },
                    "stock": 128,
                    "rating": 4.8,
                    "reviews_count": 567,
                },
                {
                    "id": "prod_003",
                    "sku": "MONITOR-4K-27",
                    "name": "UltraView 4K Monitor 27-inch",
                    "category": "electronics",
                    "price": 599.99,
                    "currency": "USD",
                    "description": "Professional-grade 4K monitor with exceptional color accuracy for creative "
                    "professionals. Features HDR400 support, USB-C connectivity with 65W power delivery, "
                    "and an ergonomic stand with height, tilt, and swivel adjustments.",
                    "specs": {
                        "resolution": "3840x2160",
                        "panel_type": "IPS",
                        "refresh_rate": "60Hz",
                        "response_time": "5ms",
                        "color_gamut": "99% sRGB",
                    },
                    "stock": 72,
                    "rating": 4.5,
                    "reviews_count": 189,
                },
            ],
            "total_products": 3,
            "last_updated": "2024-06-20T15:30:00Z",
        },
        "metadata": {
            "api_version": "v2",
            "request_id": "req_abc123xyz789",
        },
    },
    indent=2,
)

CODE_FILE = '''"""User authentication service with JWT tokens."""

from datetime import datetime, timezone, timedelta
from typing import Optional
import jwt
from pydantic import BaseModel

SECRET_KEY = "your-secret-key-here"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30


class TokenData(BaseModel):
    """Data stored in JWT token."""
    username: Optional[str] = None
    scopes: list[str] = []


class User(BaseModel):
    """User model."""
    username: str
    email: str
    full_name: Optional[str] = None
    disabled: bool = False


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """Create a new JWT access token.

    Args:
        data: Payload data to encode in the token.
        expires_delta: Custom expiration time.

    Returns:
        Encoded JWT token string.
    """
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.now(timezone.utc).replace(tzinfo=None) + expires_delta
    else:
        expire = datetime.now(timezone.utc).replace(tzinfo=None) + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


def verify_token(token: str) -> Optional[TokenData]:
    """Verify and decode a JWT token.

    Args:
        token: The JWT token to verify.

    Returns:
        TokenData if valid, None otherwise.
    """
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            return None
        scopes = payload.get("scopes", [])
        return TokenData(username=username, scopes=scopes)
    except jwt.JWTError:
        return None


def authenticate_user(username: str, password: str) -> Optional[User]:
    """Authenticate a user by username and password.

    Args:
        username: The username to authenticate.
        password: The password to verify.

    Returns:
        User object if authenticated, None otherwise.
    """
    # In production, this would check against a database
    # This is a placeholder implementation
    if username == "admin" and password == "secret":
        return User(
            username="admin",
            email="admin@example.com",
            full_name="Admin User",
            disabled=False,
        )
    return None


class RateLimiter:
    """Simple rate limiter for API endpoints."""

    def __init__(self, max_requests: int = 100, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._requests: dict[str, list[datetime]] = {}

    def is_allowed(self, client_id: str) -> bool:
        """Check if a request from client_id is allowed."""
        now = datetime.now(timezone.utc).replace(tzinfo=None)
        cutoff = now - timedelta(seconds=self.window_seconds)

        if client_id not in self._requests:
            self._requests[client_id] = []

        # Clean old requests
        self._requests[client_id] = [
            t for t in self._requests[client_id] if t > cutoff
        ]

        if len(self._requests[client_id]) >= self.max_requests:
            return False

        self._requests[client_id].append(now)
        return True
'''


@dataclass
class LLMEvalResult:
    """Result from an LLM evaluation."""

    test_name: str
    passed: bool
    expected: str
    actual: str
    tokens_original: int
    tokens_compressed: int
    compression_ratio: float
    details: str = ""

    def __str__(self) -> str:
        status = "✓ PASS" if self.passed else "✗ FAIL"
        return (
            f"{status}: {self.test_name}\n"
            f"  Compression: {self.tokens_original} → {self.tokens_compressed} "
            f"({self.compression_ratio:.1%})\n"
            f"  Expected: {self.expected}\n"
            f"  Actual: {self.actual}\n"
            f"  {self.details}"
        )


def call_openai(prompt: str, system: str = "You are a helpful assistant.") -> str:
    """Call OpenAI API with given prompt.

    Args:
        prompt: User prompt.
        system: System prompt.

    Returns:
        Model response text.
    """
    try:
        from openai import OpenAI

        client = OpenAI()
        response = client.chat.completions.create(
            model="gpt-4o-mini",  # Cost-effective for evals
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            max_tokens=500,
            temperature=0,  # Deterministic for evals
        )
        return response.choices[0].message.content or ""
    except Exception as e:
        pytest.skip(f"OpenAI API error: {e}")
        return ""


# =============================================================================
# LLM Evaluation Tests
# =============================================================================


class TestJSONDiscoverability:
    """Test that LLM can discover structure in compressed JSON."""

    @pytest.fixture
    def compressor(self):
        """Create compressor."""
        config = UniversalCompressorConfig(
            use_magika=False,
            use_kompress=False,
            ccr_enabled=False,
        )
        return UniversalCompressor(config=config)

    def test_llm_can_list_product_fields(self, compressor):
        """Test that LLM can identify available fields from compressed JSON."""
        result = compressor.compress(PRODUCT_CATALOG)

        prompt = f"""Here is a product catalog (may be compressed):

{result.compressed}

List ALL the field names/keys that are available for each product.
Format your answer as a comma-separated list of field names only."""

        response = call_openai(prompt)

        # Check that key fields are mentioned
        expected_fields = [
            "id",
            "sku",
            "name",
            "category",
            "price",
            "description",
            "specs",
            "stock",
            "rating",
        ]
        found_fields = [f for f in expected_fields if f.lower() in response.lower()]

        eval_result = LLMEvalResult(
            test_name="JSON Field Discoverability",
            passed=len(found_fields) >= 7,  # At least 7 of 9 fields
            expected=", ".join(expected_fields),
            actual=response[:200],
            tokens_original=result.tokens_before,
            tokens_compressed=result.tokens_after,
            compression_ratio=result.compression_ratio,
            details=f"Found {len(found_fields)}/9 fields: {found_fields}",
        )

        print(f"\n{eval_result}")
        assert eval_result.passed, f"LLM could not discover enough fields: {found_fields}"

    def test_llm_can_answer_specific_question(self, compressor):
        """Test that LLM can answer questions about compressed data."""
        result = compressor.compress(PRODUCT_CATALOG)

        prompt = f"""Here is a product catalog (may be compressed):

{result.compressed}

What is the price of the laptop? Just answer with the number."""

        response = call_openai(prompt)

        # The price should be visible (1299.99)
        passed = "1299" in response or "1,299" in response

        eval_result = LLMEvalResult(
            test_name="JSON Specific Query",
            passed=passed,
            expected="1299.99",
            actual=response[:100],
            tokens_original=result.tokens_before,
            tokens_compressed=result.tokens_after,
            compression_ratio=result.compression_ratio,
        )

        print(f"\n{eval_result}")
        assert eval_result.passed, "LLM could not find laptop price"

    def test_llm_knows_what_to_retrieve(self, compressor):
        """Test that LLM can identify what additional info might be needed."""
        result = compressor.compress(PRODUCT_CATALOG)

        prompt = f"""Here is a product catalog (may be compressed):

{result.compressed}

I want to write a detailed product comparison. Looking at the compressed data,
which specific product fields or details would you need me to retrieve in full
to write a good comparison? List the field names."""

        response = call_openai(prompt)

        # LLM should identify description and specs as needing full retrieval
        wants_description = "description" in response.lower()
        wants_specs = "spec" in response.lower()

        passed = wants_description or wants_specs

        eval_result = LLMEvalResult(
            test_name="CCR Retrieval Identification",
            passed=passed,
            expected="description, specs (compressed fields)",
            actual=response[:200],
            tokens_original=result.tokens_before,
            tokens_compressed=result.tokens_after,
            compression_ratio=result.compression_ratio,
            details=f"Identified description: {wants_description}, specs: {wants_specs}",
        )

        print(f"\n{eval_result}")
        assert eval_result.passed, "LLM could not identify what to retrieve"


class TestCodeUnderstanding:
    """Test that LLM can understand compressed code."""

    @pytest.fixture
    def compressor(self):
        """Create compressor."""
        config = UniversalCompressorConfig(
            use_magika=False,
            use_kompress=False,
            ccr_enabled=False,
        )
        return UniversalCompressor(config=config)

    def test_llm_can_list_functions(self, compressor):
        """Test that LLM can identify functions from compressed code."""
        result = compressor.compress(CODE_FILE)

        prompt = f"""Here is a Python file (may be compressed):

{result.compressed}

List all the function names defined in this file.
Format: one function name per line."""

        response = call_openai(prompt)

        expected_functions = [
            "create_access_token",
            "verify_token",
            "authenticate_user",
        ]
        found = [f for f in expected_functions if f in response]

        eval_result = LLMEvalResult(
            test_name="Code Function Discovery",
            passed=len(found) >= 2,
            expected=", ".join(expected_functions),
            actual=response[:200],
            tokens_original=result.tokens_before,
            tokens_compressed=result.tokens_after,
            compression_ratio=result.compression_ratio,
            details=f"Found {len(found)}/3 functions: {found}",
        )

        print(f"\n{eval_result}")
        assert eval_result.passed, "LLM could not find enough functions"

    def test_llm_can_describe_function_purpose(self, compressor):
        """Test that LLM can describe what a function does from signature."""
        result = compressor.compress(CODE_FILE)

        prompt = f"""Here is a Python file (may be compressed):

{result.compressed}

What does the `create_access_token` function do?
Answer in one sentence based on the function signature and any visible docstring."""

        response = call_openai(prompt)

        # Should mention JWT, token, or access in description
        keywords = ["jwt", "token", "access", "create"]
        found_keywords = [k for k in keywords if k.lower() in response.lower()]

        passed = len(found_keywords) >= 2

        eval_result = LLMEvalResult(
            test_name="Code Function Understanding",
            passed=passed,
            expected="Creates a JWT access token",
            actual=response[:200],
            tokens_original=result.tokens_before,
            tokens_compressed=result.tokens_after,
            compression_ratio=result.compression_ratio,
            details=f"Keywords found: {found_keywords}",
        )

        print(f"\n{eval_result}")
        assert eval_result.passed, "LLM could not understand function purpose"

    def test_llm_can_identify_classes(self, compressor):
        """Test that LLM can identify classes from compressed code."""
        result = compressor.compress(CODE_FILE)

        prompt = f"""Here is a Python file (may be compressed):

{result.compressed}

List all class names defined in this file."""

        response = call_openai(prompt)

        expected_classes = ["TokenData", "User", "RateLimiter"]
        found = [c for c in expected_classes if c in response]

        eval_result = LLMEvalResult(
            test_name="Code Class Discovery",
            passed=len(found) >= 2,
            expected=", ".join(expected_classes),
            actual=response[:200],
            tokens_original=result.tokens_before,
            tokens_compressed=result.tokens_after,
            compression_ratio=result.compression_ratio,
            details=f"Found {len(found)}/3 classes: {found}",
        )

        print(f"\n{eval_result}")
        assert eval_result.passed, "LLM could not find enough classes"


class TestMultiContentAgent:
    """Test multi-content scenario simulating an agent."""

    @pytest.fixture
    def compressor(self):
        """Create compressor."""
        config = UniversalCompressorConfig(
            use_magika=False,
            use_kompress=False,
            ccr_enabled=False,
        )
        return UniversalCompressor(config=config)

    def test_agent_mixed_content_understanding(self, compressor):
        """Test that LLM can work with mixed compressed content."""
        # Compress both
        json_result = compressor.compress(PRODUCT_CATALOG)
        code_result = compressor.compress(CODE_FILE)

        prompt = f"""You are an agent with access to two data sources.

## Data Source 1: Product Catalog (JSON)
{json_result.compressed}

## Data Source 2: Authentication Code (Python)
{code_result.compressed}

Based on the available data, answer these questions:
1. What is the most expensive product?
2. What function would I use to create a login token?
3. What product categories are available?

Answer each question briefly."""

        response = call_openai(prompt)

        # Check answers
        checks = {
            "expensive_product": any(x in response.lower() for x in ["laptop", "probook", "1299"]),
            "token_function": "create_access_token" in response,
            "categories": any(x in response.lower() for x in ["electronics", "audio"]),
        }

        passed = sum(checks.values()) >= 2

        total_original = json_result.tokens_before + code_result.tokens_before
        total_compressed = json_result.tokens_after + code_result.tokens_after

        eval_result = LLMEvalResult(
            test_name="Multi-Content Agent Understanding",
            passed=passed,
            expected="Laptop ($1299), create_access_token, electronics/audio",
            actual=response[:300],
            tokens_original=total_original,
            tokens_compressed=total_compressed,
            compression_ratio=total_compressed / total_original,
            details=f"Checks: {checks}",
        )

        print(f"\n{eval_result}")
        assert eval_result.passed, "Agent could not understand mixed content"


class TestCompressionEfficacy:
    """Test overall compression efficacy with real metrics."""

    @pytest.fixture
    def compressor(self):
        """Create compressor."""
        config = UniversalCompressorConfig(
            use_magika=False,
            use_kompress=False,
            ccr_enabled=False,
        )
        return UniversalCompressor(config=config)

    def test_compression_summary(self, compressor):
        """Generate summary of compression efficacy."""
        test_cases = [
            ("Product Catalog (JSON)", PRODUCT_CATALOG, ContentType.JSON),
            ("Auth Service (Python)", CODE_FILE, ContentType.CODE),
        ]

        print("\n" + "=" * 70)
        print("COMPRESSION EFFICACY SUMMARY (with LLM Validation)")
        print("=" * 70)

        all_passed = True

        for name, content, expected_type in test_cases:
            result = compressor.compress(content)

            # Test LLM can extract basic info
            if expected_type == ContentType.JSON:
                prompt = f"What are the top-level keys in this JSON?\n\n{result.compressed}"
                test_query = "JSON keys"
            else:
                prompt = f"What functions are defined in this code?\n\n{result.compressed}"
                test_query = "Function names"

            response = call_openai(prompt)

            # Basic validation
            llm_understood = len(response) > 20 and "error" not in response.lower()

            status = "✓" if llm_understood else "✗"
            all_passed = all_passed and llm_understood

            print(f"\n{name}:")
            print(f"  Type: {result.content_type.name}")
            print(
                f"  Tokens: {result.tokens_before} → {result.tokens_after} ({result.compression_ratio:.1%})"
            )
            print(f"  Savings: {result.tokens_before - result.tokens_after} tokens")
            print(f"  LLM Test ({test_query}): {status}")
            print(f"  LLM Response: {response[:100]}...")

        print("\n" + "=" * 70)
        print(f"Overall: {'✓ ALL TESTS PASSED' if all_passed else '✗ SOME TESTS FAILED'}")
        print("=" * 70)

        assert all_passed, "Some LLM validation tests failed"
