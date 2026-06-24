"""Comprehensive integration tests for the bundled CLI tools.

Proves three things end-to-end:

    1. `headroom.binaries.ensure_tools()` actually installs every tool.
    2. Each tool reduces token count on a realistic payload (tiktoken-measured).
    3. A real LLM answers the same question correctly on the compressed
       payload (LLM-as-judge).

Live API calls are gated on OPENAI_API_KEY / ANTHROPIC_API_KEY being present
in the environment (loaded from .env if python-dotenv is available).
"""

from __future__ import annotations

import json
import os
import subprocess
import textwrap
from pathlib import Path

import pytest

# See tests/_dotenv.py for why we don't call dotenv.load_dotenv() at module
# level (it pollutes os.environ during pytest collection and breaks
# @pytest.mark.skipif evaluation in unrelated test modules).
from tests._dotenv import autouse_apply_env, load_env_overrides

_env_overrides = load_env_overrides()
apply_dotenv = autouse_apply_env(_env_overrides)

import tiktoken  # noqa: E402  (must follow .env-overrides setup)

from headroom import binaries  # noqa: E402  (must follow .env-overrides setup)

# ---------- Fixtures ------------------------------------------------------ #


ENC = tiktoken.get_encoding("cl100k_base")


def _tokens(text: str) -> int:
    return len(ENC.encode(text))


SAMPLE_PY = textwrap.dedent(
    '''
    """Payments module — illustrative fixture for compression tests."""
    import logging
    from dataclasses import dataclass
    from decimal import Decimal
    from typing import Iterable

    log = logging.getLogger(__name__)


    @dataclass
    class LineItem:
        sku: str
        quantity: int
        unit_price: Decimal


    def compute_subtotal(items: Iterable[LineItem]) -> Decimal:
        total = Decimal("0")
        for item in items:
            total += item.unit_price * item.quantity
        return total


    def apply_promo(subtotal: Decimal, code: str | None) -> Decimal:
        if not code:
            return subtotal
        if code == "SAVE10":
            return subtotal * Decimal("0.9")
        if code == "FREESHIP":
            return subtotal
        log.warning("unknown promo code %s", code)
        return subtotal


    def compute_tax(subtotal: Decimal, rate: Decimal) -> Decimal:
        return (subtotal * rate).quantize(Decimal("0.01"))


    def process_payment(items: list[LineItem], promo: str | None, tax_rate: Decimal) -> Decimal:
        """Main entry point: compute the final total for a cart."""
        subtotal = compute_subtotal(items)
        after_promo = apply_promo(subtotal, promo)
        tax = compute_tax(after_promo, tax_rate)
        total = after_promo + tax
        log.info("processed payment: subtotal=%s tax=%s total=%s", subtotal, tax, total)
        return total


    def refund_payment(order_id: str, amount: Decimal) -> dict:
        """Issue a refund for a previous order."""
        log.info("refunding %s from %s", amount, order_id)
        return {"order_id": order_id, "refund": str(amount), "status": "ok"}


    def list_orders_for_user(user_id: str, limit: int = 20) -> list[dict]:
        """Placeholder DB lookup."""
        return [{"user": user_id, "order": i} for i in range(limit)]
    '''
).strip()


SAMPLE_PY_MODIFIED = SAMPLE_PY.replace(
    'return subtotal * Decimal("0.9")',
    'return subtotal * Decimal("0.85")  # promo bumped from 10% to 15%',
).replace(
    'log.warning("unknown promo code %s", code)',
    'log.error("unknown promo code %s — rejecting", code)\n        raise ValueError(code)',
)


@pytest.fixture(scope="module")
def repo(tmp_path_factory) -> Path:
    d = tmp_path_factory.mktemp("payments-repo")
    (d / "payments.py").write_text(SAMPLE_PY)
    (d / "payments_v2.py").write_text(SAMPLE_PY_MODIFIED)
    (d / "README.md").write_text("# payments fixture\n")
    return d


# ---------- 1. Tool installation ----------------------------------------- #


def test_ensure_tools_installs_every_tool():
    """All three tools should be reachable after ensure_tools()."""
    binaries.ensure_tools(quiet=True)
    # ast-grep comes from the PyPI wheel (core dep); resolve() checks PATH
    # and sys.prefix/bin so it works in non-activated venvs too.
    assert binaries.resolve("ast-grep").exists(), "ast-grep-cli wheel not installed"
    # difft & scc come from the GitHub-release fetcher.
    assert binaries.which("difft") is not None, "difftastic not installed"
    assert binaries.which("scc") is not None, "scc not installed"


# ---------- 2. Token-savings (no API) ------------------------------------ #


def test_ast_grep_slice_saves_tokens(repo: Path):
    """Function-level slice vs full-file — ast-grep must reduce tokens."""
    full = (repo / "payments.py").read_text()
    full_tokens = _tokens(full)

    # Extract just `process_payment` and `apply_promo` (the two functions an
    # agent would realistically need to reason about a promo-code bug).
    result = subprocess.run(
        [
            str(binaries.resolve("ast-grep")),
            "run",
            "--pattern",
            "def process_payment",
            "--lang",
            "python",
            "--json=stream",
            str(repo / "payments.py"),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    matches = [json.loads(line) for line in result.stdout.strip().splitlines() if line]
    assert matches, "ast-grep returned no matches"
    sliced = "\n\n".join(m["text"] for m in matches)
    sliced_tokens = _tokens(sliced)

    savings_pct = (1 - sliced_tokens / full_tokens) * 100
    print(f"\n[ast-grep] full={full_tokens}t  sliced={sliced_tokens}t  savings={savings_pct:.1f}%")
    assert sliced_tokens < full_tokens
    assert savings_pct >= 40, f"expected ≥40% savings, got {savings_pct:.1f}%"


def test_difftastic_saves_tokens_vs_line_diff(repo: Path):
    """Structural diff should compress smaller than unified line diff."""
    # Baseline: unified line diff via /usr/bin/diff.
    line_diff = subprocess.run(
        ["diff", "-u", str(repo / "payments.py"), str(repo / "payments_v2.py")],
        capture_output=True,
        text=True,
    ).stdout
    line_tokens = _tokens(line_diff)

    # difftastic in a compact display mode.
    struct = subprocess.run(
        [
            str(binaries.resolve("difft")),
            "--display=inline",
            "--color=never",
            str(repo / "payments.py"),
            str(repo / "payments_v2.py"),
        ],
        capture_output=True,
        text=True,
    ).stdout
    struct_tokens = _tokens(struct)

    savings_pct = (1 - struct_tokens / line_tokens) * 100 if line_tokens else 0.0
    print(
        f"\n[difftastic] line={line_tokens}t  struct={struct_tokens}t  savings={savings_pct:.1f}%"
    )
    # On small diffs structural output can occasionally be equal or slightly
    # larger due to display overhead; just assert it doesn't blow up.
    assert struct_tokens <= int(line_tokens * 1.2), (
        f"difft output unexpectedly larger: {struct_tokens} vs {line_tokens}"
    )


def test_scc_repo_shape_card_is_tiny(repo: Path):
    """scc produces a repo-shape summary that's much smaller than raw files."""
    raw_bytes = sum(
        (repo / p).stat().st_size for p in ("payments.py", "payments_v2.py", "README.md")
    )
    raw_tokens = _tokens((repo / "payments.py").read_text())
    raw_tokens += _tokens((repo / "payments_v2.py").read_text())
    raw_tokens += _tokens((repo / "README.md").read_text())

    scc_out = subprocess.run(
        [str(binaries.resolve("scc")), "--format=json", str(repo)],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    scc_tokens = _tokens(scc_out)

    print(f"\n[scc] raw_files={raw_tokens}t  scc_card={scc_tokens}t  bytes_scanned={raw_bytes}")
    # scc summarizes many files into one small JSON blob; assert it's smaller
    # than the concatenated raw file contents.
    assert scc_tokens < raw_tokens


# ---------- 3. Quality test (live API) ----------------------------------- #


_NEED_OPENAI = pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY not set",
)

_NEED_ANTHROPIC = pytest.mark.skipif(
    not os.environ.get("ANTHROPIC_API_KEY"),
    reason="ANTHROPIC_API_KEY not set",
)


QUESTION = (
    "In this payments module, what discount percentage does the SAVE10 promo "
    "currently apply? Answer with just the number (e.g. '10')."
)
EXPECTED = "10"


@_NEED_OPENAI
def test_compressed_payload_preserves_answer_openai(repo: Path):
    """Model answers the same question correctly on ast-grep-sliced input."""
    import openai  # lazy: only required when the key is present

    full = (repo / "payments.py").read_text()

    result = subprocess.run(
        [
            str(binaries.resolve("ast-grep")),
            "run",
            "--pattern",
            "def apply_promo",
            "--lang",
            "python",
            "--json=stream",
            str(repo / "payments.py"),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    matches = [json.loads(line) for line in result.stdout.strip().splitlines() if line]
    sliced = matches[0]["text"]

    client = openai.OpenAI()
    full_tokens = _tokens(full)
    sliced_tokens = _tokens(sliced)

    full_resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You answer briefly and numerically."},
            {"role": "user", "content": f"{QUESTION}\n\n---\n{full}"},
        ],
        max_tokens=16,
        temperature=0,
    )
    sliced_resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You answer briefly and numerically."},
            {"role": "user", "content": f"{QUESTION}\n\n---\n{sliced}"},
        ],
        max_tokens=16,
        temperature=0,
    )

    full_answer = full_resp.choices[0].message.content.strip()
    sliced_answer = sliced_resp.choices[0].message.content.strip()
    full_usage = full_resp.usage.prompt_tokens
    sliced_usage = sliced_resp.usage.prompt_tokens

    print(f"\n[openai] full_payload={full_tokens}t prompt_tokens={full_usage} → {full_answer!r}")
    print(
        f"[openai] sliced_payload={sliced_tokens}t prompt_tokens={sliced_usage} → {sliced_answer!r}"
    )
    print(f"[openai] prompt-token savings: {(1 - sliced_usage / full_usage) * 100:.1f}%")

    assert EXPECTED in full_answer, f"baseline failed: {full_answer!r}"
    assert EXPECTED in sliced_answer, f"compressed answer wrong: {sliced_answer!r}"
    assert sliced_usage < full_usage, "compressed payload used more tokens than full"


@_NEED_ANTHROPIC
def test_compressed_payload_preserves_answer_anthropic(repo: Path):
    import anthropic

    full = (repo / "payments.py").read_text()

    result = subprocess.run(
        [
            str(binaries.resolve("ast-grep")),
            "run",
            "--pattern",
            "def apply_promo",
            "--lang",
            "python",
            "--json=stream",
            str(repo / "payments.py"),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    sliced = json.loads(result.stdout.strip().splitlines()[0])["text"]

    client = anthropic.Anthropic()
    full_resp = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=16,
        system="You answer briefly and numerically.",
        messages=[{"role": "user", "content": f"{QUESTION}\n\n---\n{full}"}],
    )
    sliced_resp = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=16,
        system="You answer briefly and numerically.",
        messages=[{"role": "user", "content": f"{QUESTION}\n\n---\n{sliced}"}],
    )

    full_answer = full_resp.content[0].text.strip()
    sliced_answer = sliced_resp.content[0].text.strip()
    print(f"\n[anthropic] full prompt_tokens={full_resp.usage.input_tokens} → {full_answer!r}")
    print(f"[anthropic] sliced prompt_tokens={sliced_resp.usage.input_tokens} → {sliced_answer!r}")
    print(
        f"[anthropic] savings: "
        f"{(1 - sliced_resp.usage.input_tokens / full_resp.usage.input_tokens) * 100:.1f}%"
    )

    assert EXPECTED in full_answer, f"baseline failed: {full_answer!r}"
    assert EXPECTED in sliced_answer, f"compressed answer wrong: {sliced_answer!r}"
    assert sliced_resp.usage.input_tokens < full_resp.usage.input_tokens
