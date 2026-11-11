"""Pure output-savings stratification and holdout policy helpers."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from typing import Any, cast

# Coarse input-token buckets. Coarse on purpose: too many strata make
# per-stratum baselines sparse and noisy. Boundaries in tokens.
_INPUT_BUCKETS = (2_000, 8_000, 32_000, 128_000)

_STRATUM_LABEL = "output_shaper:stratum:"
_CONTROL_LABEL = "output_shaper:control:"
_CONVERSATION_LABEL = "output_shaper:conv:"

# Enough of the conversation digest to separate conversations without
# carrying a full key around: 48 bits, so a machine would need ~16M
# conversations before two collided, and a collision only ever merges two
# clusters (it cannot invent one).
_CONVERSATION_LABEL_CHARS = 12


def input_bucket(input_tokens: int) -> str:
    """Map an input-token count to a coarse bucket label."""
    if input_tokens < _INPUT_BUCKETS[0]:
        return "xs"
    if input_tokens < _INPUT_BUCKETS[1]:
        return "s"
    if input_tokens < _INPUT_BUCKETS[2]:
        return "m"
    if input_tokens < _INPUT_BUCKETS[3]:
        return "l"
    return "xl"


def model_family(model: str) -> str:
    """Collapse a model id to a coarse family for stratification.

    Token-spend behaviour clusters by family far more than by point release,
    so we bucket (e.g.) every ``claude-opus-*`` together.
    """
    m = model.lower()
    for fam in ("opus", "sonnet", "haiku", "fable", "mythos", "gpt", "gemini"):
        if fam in m:
            return fam
    return "other"


def stratum_key(
    *,
    turn_kind: str,
    input_tokens: int,
    model: str,
    has_tools: bool,
) -> str:
    """Build a stratum key from request features observable before the response.

    Order is most-to-least specific so baseline lookup can back off by trimming
    trailing fields.
    """
    return "|".join(
        (
            model_family(model),
            turn_kind,
            input_bucket(input_tokens),
            "tools" if has_tools else "notools",
        )
    )


def _absorb(digest: Any, text: str) -> None:
    """Fold one seed field into ``digest``.

    Incremental so no length cap is needed: the seed is consumed a field at a
    time instead of being concatenated into one string first. The NUL is the
    field separator, and feeding it separately is byte-identical to hashing
    ``"\x00" + text``.
    """
    digest.update(b"\x00")
    digest.update(text.encode("utf-8", "ignore"))


def _absorb_text_blocks(digest: Any, blocks: Iterable[str]) -> None:
    """Fold every text block of one message into ``digest`` as one field.

    Byte-identical to absorbing ``"\x00".join(blocks)``, including the empty
    case: a message with no text blocks still contributes its separator.
    """
    absorbed = False
    for text in blocks:
        digest.update(b"\x00")
        digest.update(text.encode("utf-8", "ignore"))
        absorbed = True
    if not absorbed:
        digest.update(b"\x00")


def _unwrap_response_create_body(body: dict[str, Any]) -> dict[str, Any]:
    response = body.get("response")
    if body.get("type") == "response.create" and isinstance(response, dict):
        return cast("dict[str, Any]", response)
    return body


def _stable_response_identifier(body: dict[str, Any]) -> str:
    def _string_value(value: Any) -> str:
        if isinstance(value, str):
            return value
        if isinstance(value, dict):
            for key in ("id", "conversation_id", "session_id", "thread_id"):
                nested = value.get(key)
                if isinstance(nested, str) and nested:
                    return nested
        return ""

    for key in ("conversation", "conversation_id", "session_id", "thread_id"):
        value = _string_value(body.get(key))
        if value and value.lower() != "auto":
            return f"{key}:{value}"

    for container_key in ("client_metadata", "metadata"):
        container = body.get(container_key)
        if not isinstance(container, dict):
            continue
        for key in (
            "conversation_id",
            "conversation_key",
            "session_id",
            "thread_id",
            "codex_session_id",
        ):
            value = _string_value(container.get(key))
            if value and value.lower() != "auto":
                return f"{container_key}.{key}:{value}"

    instructions = body.get("instructions")
    if isinstance(instructions, str) and instructions:
        return f"instructions:{instructions[:512]}"
    return ""


def conversation_key_from_body(body: dict[str, Any]) -> str:
    """Derive a conversation-stable key for holdout assignment.

    Every text block of the first user message feeds the seed, in full: the
    digest is built incrementally, field by field, so there is no length cap to
    collapse behind.

    Seeding on the first 512 characters of the *first* block collapsed whole
    populations of conversations onto one key, because agent clients open a
    conversation with injected context -- CLAUDE.md, IDE selection, memory
    digests -- that is byte-identical for every conversation in a project and
    far longer than 512 characters. The user's own words, the part that makes
    one conversation different from the next, sat past the cut.

    One key means one arm, permanently, since the assignment is deterministic:
    on a real 78k-request ledger that left fable at 22,222 treatment requests
    against 1 control and sonnet at 15,401 against 0, while opus accumulated
    3,532 control samples -- a holdout nominally set to 3% that had in fact
    frozen each client into a single arm. Reading the whole message restores
    the intended unit of randomization without weakening stability: user turns
    are never compressed, so the first message is immutable for the life of
    the conversation.
    """
    body = _unwrap_response_create_body(body)
    digest = hashlib.sha256()
    digest.update(str(body.get("model", "")).encode("utf-8", "ignore"))
    for msg in body.get("messages", []):
        if isinstance(msg, dict) and msg.get("role") == "user":
            content = msg.get("content")
            if isinstance(content, str):
                _absorb(digest, content)
            elif isinstance(content, list):
                _absorb_text_blocks(
                    digest,
                    (
                        str(block.get("text", ""))
                        for block in content
                        if isinstance(block, dict) and block.get("type") == "text"
                    ),
                )
            break
    if "input" in body:
        stable_response_key = _stable_response_identifier(body)
        if stable_response_key:
            _absorb(digest, stable_response_key)
        elif not body.get("messages"):
            _absorb(digest, "responses")
    return digest.hexdigest()


def conversation_key_from_responses_body(body: dict[str, Any]) -> str:
    """Conversation-stable key for an OpenAI Responses payload.

    Same seeding rule as :func:`conversation_key_from_body`, and for the same
    reason: this path carried the identical 512-character cut, on the first
    text part alone, so a Codex-style client whose opener begins with injected
    context keyed every conversation the same way and never left one arm.
    """
    body = _unwrap_response_create_body(body)
    digest = hashlib.sha256()
    digest.update(str(body.get("model", "")).encode("utf-8", "ignore"))
    input_data = body.get("input")
    if isinstance(input_data, str):
        _absorb(digest, input_data)
    elif isinstance(input_data, list):
        for item in input_data:
            if not isinstance(item, dict) or item.get("role") != "user":
                continue
            content = item.get("content")
            if isinstance(content, str):
                _absorb(digest, content)
            elif isinstance(content, list):
                _absorb_text_blocks(
                    digest,
                    (
                        part["text"]
                        for part in content
                        if isinstance(part, dict) and isinstance(part.get("text"), str)
                    ),
                )
            break
    return digest.hexdigest()


def assign_arm(conversation_key: str, holdout_fraction: float) -> str:
    """Deterministically assign a conversation to ``treatment`` or ``control``."""
    if holdout_fraction <= 0.0:
        return "treatment"
    if holdout_fraction >= 1.0:
        return "control"
    digest = hashlib.sha256(("arm:" + conversation_key).encode()).hexdigest()
    frac = int(digest[:8], 16) / 0xFFFFFFFF
    return "control" if frac < holdout_fraction else "treatment"


def stratum_label(arm: str, key: str) -> str:
    """Encode (arm, stratum) as a transforms_applied label."""
    prefix = _STRATUM_LABEL if arm == "treatment" else _CONTROL_LABEL
    return prefix + key


def conversation_label(conversation_key: str) -> str:
    """Encode the conversation a request belongs to as a label.

    Rides the same ``transforms_applied`` channel as the stratum so the
    ledger can count DISTINCT conversations per arm, not just requests.
    Randomization is per conversation (see :func:`assign_arm`), so the
    conversation is the independent unit; a single long agent session
    otherwise enters the estimate as thousands of independent draws.

    ``conversation_key`` is already a digest, so this only truncates it --
    no request content reaches the label or the ledger.
    """
    return _CONVERSATION_LABEL + conversation_key[:_CONVERSATION_LABEL_CHARS]


def parse_conversation_label(label: str) -> str | None:
    """Decode a conversation label, or None if not one of ours."""
    if label.startswith(_CONVERSATION_LABEL):
        return label[len(_CONVERSATION_LABEL) :]
    return None


def parse_stratum_label(label: str) -> tuple[str, str] | None:
    """Decode a label into ``(arm, stratum)``, or None if not one of ours."""
    if label.startswith(_STRATUM_LABEL):
        return "treatment", label[len(_STRATUM_LABEL) :]
    if label.startswith(_CONTROL_LABEL):
        return "control", label[len(_CONTROL_LABEL) :]
    return None
