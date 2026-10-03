"""Batch result post-processor for CCR tool call handling.

When batch results are retrieved, this processor:
1. Detects CCR tool calls in each result
2. Executes the retrieval locally (from compression store)
3. Makes continuation API calls to get final responses
4. Returns the processed results with complete answers

This module works with all three providers:
- Anthropic: Batch Message API
- OpenAI: Batch API
- Google/Gemini: Batch API

Each provider has different result formats, but the logic is the same:
1. Parse result to detect CCR tool calls
2. Execute retrieval
3. Make continuation call with tool result
4. Replace partial result with complete result
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Protocol

import httpx

from headroom.providers.ccr import CCR_API_URLS, GOOGLE_CCR_ADAPTER, get_ccr_adapter

from .batch_store import BatchContext, BatchRequestContext, get_batch_context_store
from .response_handler import CCRResponseHandler, ResponseHandlerConfig

logger = logging.getLogger(__name__)


class APIClient(Protocol):
    """Protocol for making API calls."""

    async def post(
        self,
        url: str,
        headers: dict[str, str],
        json: dict[str, Any],
    ) -> httpx.Response:
        """Make a POST request."""
        ...


@dataclass
class BatchResultProcessorConfig:
    """Configuration for batch result processing."""

    # Whether to process CCR tool calls automatically
    enabled: bool = True

    # Timeout for continuation API calls (seconds)
    continuation_timeout: int = 120

    # Maximum continuation rounds per result
    max_continuation_rounds: int = 3


@dataclass
class ProcessedBatchResult:
    """A processed batch result."""

    custom_id: str
    result: dict[str, Any]
    was_processed: bool = False  # True if CCR tool calls were handled
    continuation_rounds: int = 0
    error: str | None = None


class BatchResultProcessor:
    """Processes batch results to handle CCR tool calls.

    When a batch result contains a CCR tool call (headroom_retrieve),
    this processor:
    1. Looks up the original request context
    2. Executes the retrieval from the compression store
    3. Makes a continuation API call with the tool result
    4. Returns the final (complete) response

    Usage:
        processor = BatchResultProcessor(http_client)

        # Process results as they come in
        processed = await processor.process_results(
            batch_id="batch_123",
            results=raw_results,
            provider="anthropic"
        )

        # Results now have complete responses (CCR handled)
    """

    def __init__(
        self,
        http_client: httpx.AsyncClient,
        config: BatchResultProcessorConfig | None = None,
    ) -> None:
        self.http_client = http_client
        self.config = config or BatchResultProcessorConfig()
        self.ccr_handler = CCRResponseHandler(
            ResponseHandlerConfig(
                enabled=True,
                max_retrieval_rounds=self.config.max_continuation_rounds,
            )
        )

        self.api_urls = CCR_API_URLS

    async def process_results(
        self,
        batch_id: str,
        results: list[dict[str, Any]],
        provider: str,
    ) -> list[ProcessedBatchResult]:
        """Process batch results, handling CCR tool calls.

        Args:
            batch_id: The batch ID (to look up context).
            results: Raw batch results from the provider.
            provider: The provider type.

        Returns:
            List of processed results (with CCR handled).
        """
        if not self.config.enabled:
            return [
                ProcessedBatchResult(
                    custom_id=self._get_custom_id(r, provider),
                    result=r,
                )
                for r in results
            ]

        # Get batch context
        store = get_batch_context_store()
        batch_context = await store.get(batch_id)

        if batch_context is None:
            logger.warning(
                f"Batch context not found for {batch_id}, returning results without CCR processing"
            )
            return [
                ProcessedBatchResult(
                    custom_id=self._get_custom_id(r, provider),
                    result=r,
                )
                for r in results
            ]

        # Process each result
        processed = []
        for result in results:
            custom_id = self._get_custom_id(result, provider)
            request_context = batch_context.get_request(custom_id)

            if request_context is None:
                logger.warning(f"Request context not found for {custom_id} in batch {batch_id}")
                processed.append(ProcessedBatchResult(custom_id=custom_id, result=result))
                continue

            # Check if result contains CCR tool calls
            response = self._extract_response(result, provider)

            if response and self.ccr_handler.has_ccr_tool_calls(response, provider):
                # Process the CCR tool calls
                try:
                    final_result = await self._process_single_result(
                        result,
                        response,
                        request_context,
                        batch_context,
                        provider,
                    )
                    processed.append(final_result)
                except Exception as e:
                    logger.error(f"Failed to process CCR for {custom_id}: {e}")
                    processed.append(
                        ProcessedBatchResult(
                            custom_id=custom_id,
                            result=result,
                            error=str(e),
                        )
                    )
            else:
                # No CCR tool calls, pass through
                processed.append(ProcessedBatchResult(custom_id=custom_id, result=result))

        return processed

    def _get_custom_id(self, result: dict[str, Any], provider: str) -> str:
        """Extract the custom ID from a result."""
        return get_ccr_adapter(provider).batch_custom_id(result)

    def _extract_response(
        self,
        result: dict[str, Any],
        provider: str,
    ) -> dict[str, Any] | None:
        """Extract the actual response from a batch result."""
        return get_ccr_adapter(provider).batch_response(result)

    async def _process_single_result(
        self,
        original_result: dict[str, Any],
        response: dict[str, Any],
        request_context: BatchRequestContext,
        batch_context: BatchContext,
        provider: str,
    ) -> ProcessedBatchResult:
        """Process a single result with CCR tool calls.

        Args:
            original_result: The original batch result.
            response: The extracted response (with CCR tool calls).
            request_context: The original request context.
            batch_context: The batch context.
            provider: The provider type.

        Returns:
            Processed result with complete response.
        """
        custom_id = request_context.custom_id

        # Create API call function for continuations
        async def api_call_fn(
            messages: list[dict[str, Any]],
            tools: list[dict[str, Any]] | None,
        ) -> dict[str, Any]:
            return await self._make_continuation_call(
                messages,
                tools,
                request_context,
                batch_context,
                provider,
            )

        # Use CCR handler to process the response
        final_response = await self.ccr_handler.handle_response(
            response,
            request_context.messages,
            request_context.tools,
            api_call_fn,
            provider,
        )

        # Update the result with the final response
        updated_result = self._update_result(
            original_result,
            final_response,
            provider,
        )

        return ProcessedBatchResult(
            custom_id=custom_id,
            result=updated_result,
            was_processed=True,
            continuation_rounds=self.ccr_handler._retrieval_count,
        )

    async def _make_continuation_call(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        request_context: BatchRequestContext,
        batch_context: BatchContext,
        provider: str,
    ) -> dict[str, Any]:
        """Make a continuation API call.

        Args:
            messages: The messages including tool results.
            tools: The tools list.
            request_context: The request context.
            batch_context: The batch context.
            provider: The provider type.

        Returns:
            The API response.
        """
        adapter = get_ccr_adapter(provider)
        response = await self.http_client.post(
            adapter.continuation_url(self.api_urls, request_context, batch_context),
            headers=adapter.continuation_headers(batch_context),
            json=adapter.continuation_body(messages, tools, request_context),
            timeout=self.config.continuation_timeout,
        )
        response.raise_for_status()
        result: dict[str, Any] = response.json()
        return result

    async def _anthropic_continuation(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        request_context: BatchRequestContext,
        batch_context: BatchContext,
    ) -> dict[str, Any]:
        """Make Anthropic continuation call."""
        return await self._make_continuation_call(
            messages,
            tools,
            request_context,
            batch_context,
            "anthropic",
        )

    async def _openai_continuation(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        request_context: BatchRequestContext,
        batch_context: BatchContext,
    ) -> dict[str, Any]:
        """Make OpenAI continuation call."""
        return await self._make_continuation_call(
            messages,
            tools,
            request_context,
            batch_context,
            "openai",
        )

    async def _google_continuation(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        request_context: BatchRequestContext,
        batch_context: BatchContext,
    ) -> dict[str, Any]:
        """Make Google/Gemini continuation call.

        Note: Google format uses 'contents' not 'messages',
        and 'parts' format for messages.
        """
        return await self._make_continuation_call(
            messages,
            tools,
            request_context,
            batch_context,
            "google",
        )

    def _messages_to_google_contents(
        self,
        messages: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Convert standard messages to Google contents format."""
        return GOOGLE_CCR_ADAPTER.messages_to_contents(messages)

    def _update_result(
        self,
        original_result: dict[str, Any],
        final_response: dict[str, Any],
        provider: str,
    ) -> dict[str, Any]:
        """Update a batch result with the final processed response."""
        return get_ccr_adapter(provider).update_batch_result(original_result, final_response)


# Convenience function
async def process_batch_results(
    batch_id: str,
    results: list[dict[str, Any]],
    provider: str,
    http_client: httpx.AsyncClient,
) -> list[ProcessedBatchResult]:
    """Process batch results with CCR handling.

    This is a convenience function for one-off processing.

    Args:
        batch_id: The batch ID.
        results: Raw batch results.
        provider: The provider type.
        http_client: HTTP client for API calls.

    Returns:
        Processed results.
    """
    processor = BatchResultProcessor(http_client)
    return await processor.process_results(batch_id, results, provider)
