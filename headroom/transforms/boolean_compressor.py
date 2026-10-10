"""Boolean algebra compressor — lossless token reduction for boolean logic content.

Two compression paths share this module:

1. BooleanCompressor  — zero LLM calls, always lossless.
   Handles structured boolean content: truth tables (markdown or plain-text) and
   expressions written with single uppercase variables (A, B, C …) in any notation
   (symbolic, English operators, mixed). Uses Quine-McCluskey via boolean-algebra-engine
   to produce the minimal SOP form.

2. NLBooleanCompressor  — optional and probabilistic; requires an API key.
   Handles natural-language logic descriptions such as
   "output is high when door is open AND motion is detected but not both".
   The NL layer (boolean-algebra-engine[nl]) calls an LLM once to extract the
   boolean function, then synthesizes the minimal SOP. The synthesis is exact for
   the inferred function, but the LLM extraction is not lossless.
   Activated only when ANTHROPIC_API_KEY or OPENAI_API_KEY is set in the environment.

Compression examples:
  - Truth table (8 rows, ~80 tokens)      → "B.C+A.C+A.B"  (4 tokens,  ~91% savings)
  - English expression (17 tokens)        → "B.C+A.C+A.B"  (4 tokens,  ~71% savings)
  - NL description (15 tokens)            → "A^B"          (2 tokens,  ~87% savings)

This integration does not send telemetry to boolean-algebra-engine or any other
third-party analytics service.
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    pass


def _disable_engine_telemetry() -> None:
    """Suppress the optional engine's install analytics before its cold import."""
    os.environ["BOOLCALC_NO_TELEMETRY"] = "1"


# ── Notation normalisation ────────────────────────────────────────────────────

# Order matters: longer patterns before shorter ones
_NORMALISE_RULES: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\bXOR\b", re.I), "^"),
    (re.compile(r"\bNOT\s+", re.I), "!"),
    (re.compile(r"\bAND\b", re.I), "."),
    (re.compile(r"\bOR\b", re.I), "+"),
    (re.compile(r"\|\|"), "+"),
    (re.compile(r"&&"), "."),
    (re.compile(r"[~¬]"), "!"),
    (re.compile(r"\|(?!\|)"), "+"),
    (re.compile(r"&(?!&)"), "."),
    (re.compile(r"\s+"), ""),  # strip all whitespace last
]


def _normalise(expr: str) -> str:
    """Normalise various boolean notations to boolcalc syntax (A.B+!C)."""
    for pattern, replacement in _NORMALISE_RULES:
        expr = pattern.sub(replacement, expr)
    return expr.strip()


# ── Truth table parsing ───────────────────────────────────────────────────────


@dataclass
class _ParsedTable:
    variables: list[str]
    minterms: list[int]


def _try_parse_truth_table(content: str) -> _ParsedTable | None:
    """Parse a markdown or space-separated truth table.

    Returns _ParsedTable if confident, None otherwise.
    """
    lines = [ln.strip() for ln in content.strip().splitlines() if ln.strip()]
    # Need at least a header + one data row
    if len(lines) < 2:
        return None

    # Find header line (contains variable names, no binary data)
    header_line = None
    data_start = 0
    for i, line in enumerate(lines):
        clean = re.sub(r"[|\-:+]", " ", line).strip()
        words = clean.split()
        if words and all(re.match(r"^[A-Za-z_]\w*$", w) for w in words):
            header_line = words
            data_start = i + 1
            # Skip markdown separator row (---|---...)
            if data_start < len(lines) and re.match(r"^[\s|:\-]+$", lines[data_start]):
                data_start += 1
            break

    if header_line is None or len(header_line) < 2:
        return None

    variables = header_line[:-1]  # all but last column
    # output_col = header_line[-1]  # not used — last column is output

    if not (1 <= len(variables) <= 8):
        return None

    # Parse data rows
    minterms: list[int] = []
    row_index = 0
    for line in lines[data_start:]:
        clean = re.sub(r"[|\s]+", " ", line).strip()
        values = clean.split()
        if not values:
            continue
        if not all(v in ("0", "1") for v in values):
            return None  # non-binary cell → not a truth table
        if len(values) != len(header_line):
            return None  # column count mismatch
        if int(values[-1]) == 1:
            minterms.append(row_index)
        row_index += 1

    expected_rows = 2 ** len(variables)
    if row_index != expected_rows:
        return None  # incomplete table

    return _ParsedTable(variables=variables, minterms=minterms)


# ── Compression result ────────────────────────────────────────────────────────


@dataclass
class BooleanCompressionResult:
    compressed: str
    original: str
    original_tokens: int
    compressed_tokens: int
    variable_count: int
    strategy: str  # "expression" | "truth_table"

    @property
    def tokens_saved(self) -> int:
        return max(0, self.original_tokens - self.compressed_tokens)

    @property
    def savings_pct(self) -> float:
        if self.original_tokens == 0:
            return 0.0
        return 100.0 * self.tokens_saved / self.original_tokens


# ── Core compressor ───────────────────────────────────────────────────────────


class BooleanCompressor:
    """Lossless boolean algebra compressor.

    Requires: pip install boolean-algebra-engine
    Optional dependency — fails gracefully if not installed.
    """

    def compress(self, content: str) -> BooleanCompressionResult | None:
        """Compress boolean content. Returns None on failure (caller should passthrough)."""
        try:
            result = self._compress_truth_table(content)
            if result is None:
                result = self._compress_expression(content)
            return result
        except Exception as exc:
            logger.debug("BooleanCompressor: compression failed: %s", exc)
            return None

    def _compress_truth_table(self, content: str) -> BooleanCompressionResult | None:
        parsed = _try_parse_truth_table(content)
        if parsed is None:
            return None
        if not parsed.minterms and len(parsed.variables) > 0:
            # Contradiction — all outputs 0
            minimal = "0"
        elif len(parsed.minterms) == 2 ** len(parsed.variables):
            # Tautology — all outputs 1
            minimal = "1"
        else:
            try:
                _disable_engine_telemetry()
                from boolean_algebra_engine import synthesize
                from boolean_algebra_engine.core.models import TruthTable, TruthTableRow

                rows: list[TruthTableRow] = []
                n = len(parsed.variables)
                for i in range(2**n):
                    bits = [(i >> (n - 1 - j)) & 1 for j in range(n)]
                    inputs = dict(zip(parsed.variables, bits))
                    output = 1 if i in parsed.minterms else 0
                    rows.append(TruthTableRow(inputs=inputs, output=output))

                table = TruthTable(
                    expression="(truth table)",
                    variables=list(parsed.variables),
                    rows=rows,
                )
                minimal, _ = synthesize(table)
            except Exception as exc:
                logger.debug("BooleanCompressor: synthesize failed: %s", exc)
                return None

        original_tokens = len(content.split())
        compressed_tokens = len(minimal.split())
        if compressed_tokens >= original_tokens:
            return None  # no savings

        header = f"[boolean-simplified: {', '.join(parsed.variables)} → {minimal}]\n"
        compressed = header + minimal

        return BooleanCompressionResult(
            compressed=compressed,
            original=content,
            original_tokens=original_tokens,
            compressed_tokens=len(compressed.split()),
            variable_count=len(parsed.variables),
            strategy="truth_table",
        )

    def _compress_expression(self, content: str) -> BooleanCompressionResult | None:
        content_stripped = content.strip()
        # These operators require grouping-aware rewrites. Never approximate
        # them with character substitution on a path advertised as exact.
        if re.search(r"\b(?:NAND|NOR|XNOR)\b", content_stripped, re.I):
            return None
        # Only compress if it looks like a single expression (not prose)
        if "\n" in content_stripped and not re.search(
            r"\b(AND|OR|NOT|XOR)\b", content_stripped, re.I
        ):
            return None
        # Collapse multi-line expressions (each line is one expression or one term)
        lines = [ln.strip() for ln in content_stripped.splitlines() if ln.strip()]
        expr_line = " ".join(lines) if len(lines) > 1 else content_stripped

        normalised = _normalise(expr_line)

        # Must contain at least one operator to be worth compressing
        if not re.search(r"[.+!^]", normalised):
            return None
        # Sanity: only boolcalc-legal characters
        if not re.match(r"^[A-Za-z0-9_.+!^()]+$", normalised):
            return None

        try:
            _disable_engine_telemetry()
            from boolean_algebra_engine import evaluate, synthesize

            table, _ = evaluate(normalised)
            minimal, m = synthesize(table)
        except Exception as exc:
            logger.debug("BooleanCompressor: expression evaluate/synthesize failed: %s", exc)
            return None

        original_tokens = len(content.split())
        compressed_tokens = len(minimal.split())
        if compressed_tokens >= original_tokens:
            return None

        var_count = len(table.variables)
        compressed = f"[boolean-simplified: {normalised} → {minimal}]\n{minimal}"

        return BooleanCompressionResult(
            compressed=compressed,
            original=content,
            original_tokens=original_tokens,
            compressed_tokens=len(compressed.split()),
            variable_count=var_count,
            strategy="expression",
        )


# ── NL Boolean Compressor ─────────────────────────────────────────────────────

# Natural-language logic description markers
_NL_LOGIC_PATTERNS = [
    re.compile(
        r"\b(output|signal|result|flag|value|state)\s+(is\s+)?(high|low|true|false|1|0|on|off)\s+(when|if|whenever)\b",
        re.I,
    ),
    re.compile(r"\b(true|on|active|high|enabled)\s+(only\s+)?(if|when|whenever|iff)\b", re.I),
    re.compile(r"\b(if|when)\s+\w+\s+(and|or|not|xor|nor|nand)\s+\w+\b", re.I),
    re.compile(
        r"\b(lights?|motor|alarm|gate|switch|relay)\s+(turns?\s+)?(on|off)\s+(when|if)\b", re.I
    ),
]

_NL_OPERATOR_WORDS = re.compile(
    r"\b(and|or|not|xor|nor|nand|both|neither|either|unless|except|only if|but not)\b", re.I
)


def _looks_like_nl_logic(content: str) -> bool:
    """Return True if content reads like a natural-language logic description."""
    for pat in _NL_LOGIC_PATTERNS:
        if pat.search(content):
            return True
    # Heuristic: short prose with at least 2 operator words
    hits = _NL_OPERATOR_WORDS.findall(content)
    return len(hits) >= 2 and len(content.split()) < 80


def _detect_provider() -> Any | None:
    """Return an NL provider if a supported API key is set, else None."""
    try:
        _disable_engine_telemetry()
        from boolean_algebra_engine.nl.nl import AnthropicProvider, OpenAIProvider

        if os.environ.get("ANTHROPIC_API_KEY"):
            return AnthropicProvider()
        if os.environ.get("OPENAI_API_KEY"):
            return OpenAIProvider()
    except ImportError:
        pass
    return None


class NLBooleanCompressor:
    """Natural-language → minimal boolean expression compressor.

    Requires:
      - pip install boolean-algebra-engine[nl-anthropic]  (or nl-openai)
      - ANTHROPIC_API_KEY or OPENAI_API_KEY in environment

    Uses one LLM call per compression to extract the boolean function from
    natural-language text; the Quine-McCluskey synthesis step is always
    deterministic for the inferred function. Because the inference uses an LLM,
    this path is probabilistic and must not be described as lossless.
    """

    def compress(self, content: str) -> BooleanCompressionResult | None:
        if not _looks_like_nl_logic(content):
            return None
        provider = _detect_provider()
        if provider is None:
            logger.debug("NLBooleanCompressor: no API key configured — skipping")
            return None
        try:
            _disable_engine_telemetry()
            from boolean_algebra_engine.nl.nl import ask

            result = ask(content.strip(), provider=provider)
        except Exception as exc:
            logger.debug("NLBooleanCompressor: ask() failed: %s", exc)
            return None

        minimal = result.minimal or result.expression
        if not minimal:
            return None

        var_map = "  ".join(f"{k}={v}" for k, v in result.variables.items())
        compressed = f"[nl-boolean: {var_map}]\n{minimal}"

        original_tokens = len(content.split())
        compressed_tokens = len(compressed.split())
        if compressed_tokens >= original_tokens:
            return None

        return BooleanCompressionResult(
            compressed=compressed,
            original=content,
            original_tokens=original_tokens,
            compressed_tokens=compressed_tokens,
            variable_count=len(result.variables),
            strategy="nl_expression",
        )
