"""The `headroom.pipeline_extension` entry point: dispatches INPUT_RECEIVED
tool_result blocks to the right edge by tool name, and applies the file-read
edge over the whole message list first (it needs cross-message context the
per-block edges don't).

Registered via `pyproject.toml`'s entry-points table as the class itself
(`headroom_edge_layer:EdgeLayerExtension`) — `discover_pipeline_extensions()`
does a plain attribute lookup and only instantiates the result when it's a
`type`, so a factory function would silently be treated as an inert object
instead of being called.
"""

from __future__ import annotations

import copy
import dataclasses
from typing import Any

from headroom.pipeline import PipelineEvent, PipelineStage

from .buckets import BUCKET_GREP, BUCKET_SHELL_LOG, BUCKET_WEB, classify_tool_name
from .cache_safety import SavingsLedgerEntry, log_savings
from .config import EdgeLayerConfig
from .file_read_edge import compress_file_reads
from .grep_edge import compress_grep_result
from .intent_query import extract_intent_query
from .message_utils import collect_tool_use_index, get_block_text, iter_tool_result_blocks, set_block_text
from .rerank_edge import compress_log_or_web
from .scoring import Scorer, build_scorer


def _default_store() -> Any:
    # Imported lazily so importing this module (e.g. from tests, without the
    # full Headroom package on the path) doesn't hard-fail.
    from headroom.cache.compression_store import get_compression_store

    return get_compression_store()


class EdgeLayerExtension:
    """`PipelineExtension` implementation — see `headroom/pipeline.py`'s
    `PipelineExtension` protocol: `on_pipeline_event(event) -> event | None`.
    """

    def __init__(
        self,
        config: EdgeLayerConfig | None = None,
        *,
        store: Any = None,
        scorer: Scorer | None = None,
    ) -> None:
        self.config = config or EdgeLayerConfig.from_env()
        self._store = store
        self._scorer = scorer or build_scorer(
            reranker_url=self.config.reranker_url,
            timeout_seconds=self.config.reranker_timeout_seconds,
        )

    @property
    def store(self) -> Any:
        if self._store is not None:
            return self._store
        return _default_store()

    def on_pipeline_event(self, event: PipelineEvent) -> PipelineEvent | None:
        if event.stage != PipelineStage.INPUT_RECEIVED:
            return None
        if not event.messages:
            return None

        messages = copy.deepcopy(event.messages)
        ledger: list[SavingsLedgerEntry] = []

        if self.config.enable_file_read:
            messages, file_read_ledger = compress_file_reads(
                messages, config=self.config, store=self.store, request_id=event.request_id
            )
            ledger.extend(file_read_ledger)

        intent_query = extract_intent_query(messages) if self.config.enable_intent_query else ""
        tool_use_index = collect_tool_use_index(messages)

        for _message_index, _block_index, block, tool_use_id in list(
            iter_tool_result_blocks(messages)
        ):
            if not tool_use_id or tool_use_id not in tool_use_index:
                continue
            tool_name, tool_input = tool_use_index[tool_use_id]
            bucket = classify_tool_name(tool_name)
            text = get_block_text(block)
            if text is None:
                continue

            new_text: str | None = None
            entry: SavingsLedgerEntry | None = None

            if bucket == BUCKET_GREP and self.config.enable_grep:
                new_text, entry = compress_grep_result(
                    text,
                    tool_name=tool_name,
                    tool_call_id=tool_use_id,
                    tool_input=tool_input if isinstance(tool_input, dict) else {},
                    intent_query=intent_query,
                    config=self.config,
                    store=self.store,
                    scorer=self._scorer,
                    request_id=event.request_id,
                )
            elif bucket in (BUCKET_WEB, BUCKET_SHELL_LOG) and self.config.enable_rerank:
                new_text, entry = compress_log_or_web(
                    text,
                    tool_name=tool_name,
                    tool_call_id=tool_use_id,
                    intent_query=intent_query,
                    config=self.config,
                    store=self.store,
                    scorer=self._scorer,
                    request_id=event.request_id,
                )

            if entry is not None and new_text is not None and new_text != text:
                set_block_text(block, new_text)
                log_savings(entry)

        for entry in ledger:
            log_savings(entry)

        return dataclasses.replace(event, messages=messages)


def build_extension(config: EdgeLayerConfig | None = None) -> EdgeLayerExtension:
    """Convenience constructor for tests/harness code — NOT the entry-point
    target (see the module docstring for why the entry point names the class
    directly instead).
    """
    return EdgeLayerExtension(config)
