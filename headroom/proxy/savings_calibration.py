"""Per-request savings in the provider's own token units.

Headroom measures what compression removed with its local tokenizer. For
OpenAI models that IS the provider's tokenizer (``tiktoken``; measured against
OpenAI's billed ``prompt_tokens`` it matches to within a token), so the saving
is already exact and nothing is adjusted. Claude's tokenizer is private and is
counted with an ``o200k_base`` stand-in that runs 15-25% under Claude 4.x and
35-38% under Claude 5.x; Gemini, Cohere and Kimi use estimators. For those,
the saving is converted per request:

    saved_provider = saved_local x (billed / local_forwarded) x correction

``billed / local_forwarded`` is THIS request's own exchange rate: what the
provider says it received, over what Headroom counts for the same forwarded
request (system prompt and tools included). It needs no side calls and no
credentials beyond the user's own, so it works on-prem with API keys and
subscriptions alike, and it follows a new provider tokenizer the moment one
ships.

``correction`` exists because what Headroom removes is not average text. It
is mostly JSON punctuation, whitespace and repeated lines, which the provider
tokenizes more cheaply than the code and prose it received. The removed text
itself never reaches the provider, so its count can only be measured offline:
``benchmarks/savings_calibration_accuracy.py`` replays real agent transcripts
and compares against Anthropic ``count_tokens``. The measured values are below
per tokenizer family; a family without one uses 1.0 and says so.

Every request records which basis it got (``source``), so no number is
presented as more exact than it is:

``native``          local tokenizer is the provider's; saving is exact.
``request``         this request's billed/local ratio x the family correction.
``model_average``   the request could not be measured (no provider usage, or
                    images/documents whose local counts are rough), so the
                    model's recent average ratio is used.
``uncalibrated``    nothing to calibrate against yet; local units.

Savings are also split into ``novel`` (removed from content new this turn) and
``carried`` (history compressed on an earlier turn, still smaller as it is
re-sent). Carried savings are real on every turn but are re-sent as cache
reads, which is why savings appear on every turn and why they price
differently.
"""

from __future__ import annotations

import re
import threading
from collections import OrderedDict
from dataclasses import dataclass

__all__ = [
    "CLIENT_REQUEST_TOKENS_TAG",
    "CLIENT_TOOL_TOKENS_TAG",
    "CalibratedSavings",
    "SavingsCalibrator",
    "get_savings_calibrator",
    "local_request_counts",
    "local_request_overhead",
    "local_request_tokens",
    "removed_content_correction",
    "reset_savings_calibrator",
]

# Internal tag carrying Headroom's local count of the WHOLE request as the
# client sent it (system + tools + messages), measured before any Headroom
# mutation. Compared with the same count of the forwarded request it gives the
# request's net saving: message compression, tool-schema compaction and
# deferral, minus anything Headroom added (its own tool definitions), each
# counted exactly once regardless of which handler path ran.
CLIENT_REQUEST_TOKENS_TAG = "_headroom_client_request_tokens"
# Same, for the billed tool definitions alone, so the change in tools (deferral,
# compaction, tools Headroom adds) is converted at the tool-definition rate.
CLIENT_TOOL_TOKENS_TAG = "_headroom_client_tool_tokens"

SOURCE_NATIVE = "native"
SOURCE_REQUEST = "request"
SOURCE_MODEL_AVERAGE = "model_average"
SOURCE_UNCALIBRATED = "uncalibrated"

# A per-request ratio outside this band is a measurement artefact (a response
# without usage, a half-counted request), not a tokenizer difference.
MIN_RATIO = 0.5
MAX_RATIO = 3.0
# Below this many locally counted tokens, fixed framing overhead dominates the
# ratio.
MIN_REQUEST_TOKENS = 1_000
EWMA_ALPHA = 0.1
MAX_CONVERSATIONS = 4_096

# Removed-content correction per tokenizer family: provider tokens of the text
# Headroom removed, relative to the request-level exchange rate. Measured with
# benchmarks/savings_calibration_accuracy.py on 110 requests from 7 real Claude
# Code transcripts (default pipeline: text/code/mixed/log compressors), scored
# leave-one-conversation-out against Anthropic count_tokens:
#
#                         median |err|   total bias
#   claude-sonnet-4-6  raw 7.7%  -> 4.5%    -10.4% -> +0.3%
#   claude-sonnet-5-5  raw 28.1% -> 5.3%    -29.5% -> +0.5%
#
# Re-run with ~22K tokens of agent tool definitions in every request: fitted
# 0.918 / 0.883, LOO median |err| 4.5% / 4.4%, total bias +0.2% / +0.5%.
#
# Re-measure when a family's tokenizer changes. Families not listed use 1.0
# (the request ratio alone).
REMOVED_CONTENT_CORRECTION: dict[str, float] = {
    "claude-4": 0.91,
    "claude-5": 0.89,
}

# Any tier name (opus, sonnet, haiku, fable, ...) followed by the major
# version. Measured with count_tokens: every tier of a generation counts text
# identically (Opus/Sonnet 4.6 and Haiku 4.5; Opus/Sonnet/Haiku 5.5 and
# Fable 5.1), so the family is the generation, not the tier.
_CLAUDE_MAJOR = re.compile(r"claude-(?:[a-z]+-)?(\d+)")


# Tool definitions removed from a request (tool-search deferral, schema
# compaction) are converted at their own rate: provider tokens per local token
# of a tool definition. Unlike message content this does not track the rest of
# the request (a request dominated by a compressed table has a low ratio while
# its deferred tool prose is still counted heavily), so it is an absolute rate,
# measured with count_tokens on prose-heavy agent tool definitions:
#   claude-sonnet-4-6 1.198, claude-sonnet-5-5 1.623
# (schema-heavy definitions run higher, 1.44 / 1.81, so this is conservative).
# Families not listed use the request's own ratio.
TOOL_DEFINITION_RATE: dict[str, float] = {
    "claude-4": 1.20,
    "claude-5": 1.62,
}


def tokenizer_family(model: str) -> str | None:
    """Tokenizer family for the correction table, or None when unknown."""
    match = _CLAUDE_MAJOR.search((model or "").lower())
    if match:
        return "claude-5" if int(match.group(1)) >= 5 else "claude-4"
    return None


def removed_content_correction(model: str) -> float:
    family = tokenizer_family(model)
    return REMOVED_CONTENT_CORRECTION.get(family, 1.0) if family else 1.0


@dataclass(frozen=True)
class CalibratedSavings:
    """One request's savings, in the provider's units where measurable."""

    # saved_provider / saved_local: the request ratio times the correction.
    factor: float
    # billed / local for this request (0.0 when not measured).
    request_ratio: float
    source: str
    # Net saving: everything Headroom removed minus anything it added. Can be
    # negative on a request where Headroom added more than it removed.
    tokens_saved: int
    novel_tokens_saved: int
    carried_tokens_saved: int
    # The part of tokens_saved that is tool definitions removed from the
    # request (deferral/compaction), converted at TOOL_DEFINITION_RATE.
    tool_tokens_saved: int
    # What this request would have been without Headroom, in provider tokens:
    # billed input (or its estimate, see baseline_estimated) plus the saving.
    baseline_input_tokens: int
    # True when the provider reported no usage and the baseline is built from
    # Headroom's own estimate of the forwarded request.
    baseline_estimated: bool
    reduction_percent: float


@dataclass
class _Turn:
    # Previous request's savings per bucket, local and provider units.
    message_local: int = 0
    message_provider: int = 0
    tool_local: int = 0
    tool_provider: int = 0


def _carry(current: int, prev_local: int, prev_provider: int, rate: float) -> tuple[int, int]:
    """Split one bucket's saving into (carried, novel) provider tokens.

    The part also present last request (same sign, up to its size) is carried
    and keeps the provider value it was booked at; the rest is novel and is
    converted at this request's rate. A bucket that first appears, grows, or
    flips sign contributes novel tokens only for the change.
    """
    if prev_local and current and (current > 0) == (prev_local > 0):
        overlap = min(abs(current), abs(prev_local)) * (1 if current > 0 else -1)
        carried = round(prev_provider * overlap / prev_local)
    else:
        overlap, carried = 0, 0
    return carried, round((current - overlap) * rate)


class SavingsCalibrator:
    """Per-request calibration. Keeps each conversation's previous request (for
    the novel/carried split) and each model's recent ratio (for unmeasurable
    requests)."""

    def __init__(self, max_conversations: int = MAX_CONVERSATIONS) -> None:
        self._lock = threading.Lock()
        self._turns: OrderedDict[str, _Turn] = OrderedDict()
        self._model_ratio: dict[str, float] = {}
        self._max_conversations = max_conversations

    def calibrate(
        self,
        *,
        model: str,
        conversation_key: str | None,
        billed_input_tokens: int,
        local_forwarded_tokens: int,
        local_covers_request: bool,
        tokens_saved: int,
        native_tokenizer: bool,
        tool_definition_tokens_saved: int = 0,
        local_forwarded_tool_tokens: int = 0,
    ) -> CalibratedSavings:
        """Calibrate one request.

        ``billed_input_tokens``: the provider's own count of this request (0 if
        it reported none). ``local_forwarded_tokens``: Headroom's local count of
        the forwarded request, same tokenizer as ``tokens_saved``.
        ``local_covers_request``: that count includes the whole request (system
        prompt and tools, no images/documents), so billed/local is like for
        like. ``native_tokenizer``: the local tokenizer is the provider's own.
        ``tokens_saved``: the net local saving (may be negative).
        ``tool_definition_tokens_saved``: the part of it that is the change in
        billed tool definitions (deferral and compaction minus tools Headroom
        added; may be negative). The rest is message content.
        ``local_forwarded_tool_tokens``: the billed tool definitions within
        ``local_forwarded_tokens``, used only to estimate the forwarded size
        when the provider reported no usage.
        """
        billed = max(int(billed_input_tokens or 0), 0)
        local = max(int(local_forwarded_tokens or 0), 0)
        net = int(tokens_saved or 0)
        tools_local = int(tool_definition_tokens_saved or 0)
        message_local = net - tools_local

        with self._lock:
            ratio = 0.0
            if native_tokenizer:
                factor, source = 1.0, SOURCE_NATIVE
            else:
                measured = None
                if billed > 0 and local_covers_request and local >= MIN_REQUEST_TOKENS:
                    candidate = billed / local
                    if MIN_RATIO <= candidate <= MAX_RATIO:
                        measured = candidate
                if measured is not None:
                    ratio, source = measured, SOURCE_REQUEST
                    avg = self._model_ratio.get(model)
                    self._model_ratio[model] = (
                        measured if avg is None else avg + EWMA_ALPHA * (measured - avg)
                    )
                elif model in self._model_ratio:
                    ratio, source = self._model_ratio[model], SOURCE_MODEL_AVERAGE
                else:
                    source = SOURCE_UNCALIBRATED
                factor = ratio * removed_content_correction(model) if ratio else 1.0
            if native_tokenizer:
                tool_rate = 1.0
            else:
                family = tokenizer_family(model)
                tool_rate = TOOL_DEFINITION_RATE.get(family or "", ratio or 1.0)

            # Carried = removed on an earlier request and still removed (history
            # compressed earlier, a tool deferred since an earlier turn). It
            # would have been re-sent as cached prefix, so it is priced as cache
            # reads; novel is what changed on this request.
            prev = (self._turns.get(conversation_key) if conversation_key else None) or _Turn()
            msg_carried, msg_novel = _carry(
                message_local, prev.message_local, prev.message_provider, factor
            )
            tool_carried, tool_novel = _carry(
                tools_local, prev.tool_local, prev.tool_provider, tool_rate
            )
            carried = msg_carried + tool_carried
            novel = msg_novel + tool_novel
            tools = tool_carried + tool_novel
            total = carried + novel

            if conversation_key:
                self._turns[conversation_key] = _Turn(
                    message_local=message_local,
                    message_provider=msg_carried + msg_novel,
                    tool_local=tools_local,
                    tool_provider=tools,
                )
                self._turns.move_to_end(conversation_key)
                while len(self._turns) > self._max_conversations:
                    self._turns.popitem(last=False)

        # Without provider usage the forwarded size is an estimate in provider
        # units (local count x the ratio in use), and the baseline says so.
        # Tool definitions are estimated at the same rate their saving was
        # converted at, so a tool Headroom added cannot push the baseline below
        # the client's own request.
        if billed > 0:
            sent, estimated = billed, False
        else:
            fwd_tools = min(max(int(local_forwarded_tool_tokens or 0), 0), local)
            sent = round((local - fwd_tools) * (ratio or 1.0) + fwd_tools * tool_rate)
            estimated = True
        baseline = max(sent + total, 0) if sent > 0 else 0
        return CalibratedSavings(
            factor=round(factor, 4),
            request_ratio=round(ratio, 4),
            source=source,
            tokens_saved=total,
            novel_tokens_saved=novel,
            carried_tokens_saved=carried,
            tool_tokens_saved=tools,
            baseline_input_tokens=baseline,
            baseline_estimated=estimated,
            reduction_percent=round(total / baseline * 100, 2) if baseline > 0 else 0.0,
        )

    def model_ratios(self) -> dict[str, float]:
        with self._lock:
            return {model: round(r, 4) for model, r in self._model_ratio.items()}


_calibrator: SavingsCalibrator | None = None
_calibrator_lock = threading.Lock()


def get_savings_calibrator() -> SavingsCalibrator:
    global _calibrator
    with _calibrator_lock:
        if _calibrator is None:
            _calibrator = SavingsCalibrator()
        return _calibrator


def reset_savings_calibrator() -> None:
    global _calibrator
    with _calibrator_lock:
        _calibrator = None


_MEDIA_BLOCK_TYPES = frozenset({"image", "document", "image_url", "input_image", "input_file"})
_overhead_cache: OrderedDict[tuple[str, str], int] = OrderedDict()
_overhead_lock = threading.Lock()
_OVERHEAD_CACHE_MAX = 256


def _has_media(messages: object) -> bool:
    if not isinstance(messages, list):
        return False
    for message in messages:
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, list):
            continue
        for block in content:
            if not isinstance(block, dict):
                continue
            if block.get("type") in _MEDIA_BLOCK_TYPES:
                return True
            inner = block.get("content")
            if isinstance(inner, list) and any(
                isinstance(b, dict) and b.get("type") in _MEDIA_BLOCK_TYPES for b in inner
            ):
                return True
    return False


def _count_json(model: str, value: object) -> int:
    """Local tokens of a JSON-serialisable value, cached by content (system
    prompts and tool lists repeat on every turn of a session)."""
    import hashlib
    import json

    text = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    key = (model, hashlib.sha256(text.encode("utf-8", "ignore")).hexdigest())
    with _overhead_lock:
        cached = _overhead_cache.get(key)
        if cached is not None:
            _overhead_cache.move_to_end(key)
            return cached
    from headroom.tokenizers import get_tokenizer

    tokens = int(get_tokenizer(model).count_text(text))
    with _overhead_lock:
        _overhead_cache[key] = tokens
        while len(_overhead_cache) > _OVERHEAD_CACHE_MAX:
            _overhead_cache.popitem(last=False)
    return tokens


def _billed_tools(tools: object) -> list[object]:
    # A deferred tool (``defer_loading``, tool search) is not billed until the
    # model loads it, so it is not part of what the provider counted.
    if not isinstance(tools, list):
        return []
    return [t for t in tools if not (isinstance(t, dict) and t.get("defer_loading"))]


_message_cache: OrderedDict[tuple[str, str], int] = OrderedDict()
_MESSAGE_CACHE_MAX = 8_192


def _message_tokens(model: str, counter: object, message: object) -> int:
    """Local tokens of one message, cached by content. Agent conversations
    re-send the same history every turn, so only new messages are tokenized."""
    import hashlib
    import json

    text = json.dumps(message, sort_keys=True, separators=(",", ":"), default=str)
    key = (model, hashlib.sha256(text.encode("utf-8", "ignore")).hexdigest())
    with _overhead_lock:
        cached = _message_cache.get(key)
        if cached is not None:
            _message_cache.move_to_end(key)
            return cached
    tokens = int(counter.count_messages([message])) - int(counter.count_messages([]))  # type: ignore[attr-defined]
    with _overhead_lock:
        _message_cache[key] = tokens
        while len(_message_cache) > _MESSAGE_CACHE_MAX:
            _message_cache.popitem(last=False)
    return tokens


def local_request_counts(model: str, body: dict | None) -> tuple[int, int, bool]:
    """Headroom's local count of a whole request as the provider bills it.

    Returns ``(total, tools, covers_request)``: system + billed tools +
    messages; the billed tool definitions alone; and whether the count is like
    for like with the provider's (False with images/documents, whose local
    counts are rough). Counted the same way for the client's request and the
    forwarded one, so their difference is the request's net saving and the
    difference in ``tools`` is the tool-definition part of it.
    """
    if not isinstance(body, dict):
        return 0, 0, False
    covers = not _has_media(body.get("messages"))
    try:
        tools_list = _billed_tools(body.get("tools"))
        tools = _count_json(model, tools_list) if tools_list else 0
        system = body.get("system")
        total = tools + (_count_json(model, system) if system else 0)
        messages = body.get("messages")
        if isinstance(messages, list):
            from headroom.tokenizers import get_tokenizer

            counter = get_tokenizer(model)
            total += int(counter.count_messages([]))
            for message in messages:
                total += _message_tokens(model, counter, message)
    except Exception:  # pragma: no cover - never let accounting break a request
        return 0, 0, False
    return total, tools, covers


def local_request_tokens(model: str, body: dict | None) -> tuple[int, bool]:
    """``(total, covers_request)`` from :func:`local_request_counts`."""
    total, _tools, covers = local_request_counts(model, body)
    return total, covers


def local_request_overhead(model: str, body: dict | None) -> tuple[int, bool]:
    """Local count of the non-message parts (system + billed tools), and
    whether the request is like for like (see :func:`local_request_counts`)."""
    if not isinstance(body, dict):
        return 0, False
    total, _tools, covers = local_request_counts(
        model, {"system": body.get("system"), "tools": body.get("tools")}
    )
    return total, covers and not _has_media(body.get("messages"))
