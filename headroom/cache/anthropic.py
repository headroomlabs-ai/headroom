"""
Anthropic Cache Optimizer.

Implements cache optimization for Anthropic's explicit cache_control mechanism.
Anthropic uses ephemeral cache breakpoints to mark content that should be cached.

Anthropic Caching Characteristics:
- Explicit cache_control: {"type": "ephemeral"} blocks
- Minimum 1024 tokens for caching to be effective
- Maximum 4 cache breakpoints per request
- 5-minute TTL (extended on cache hit)
- Cost: 25% MORE to write to cache, 90% LESS to read

Usage:
    from headroom.cache import AnthropicCacheOptimizer, OptimizationContext

    optimizer = AnthropicCacheOptimizer()
    context = OptimizationContext(provider="anthropic", model="claude-3-opus")

    result = optimizer.optimize(messages, context)
    # result.messages now contains cache_control blocks
"""

from __future__ import annotations

import copy
import re
from dataclasses import dataclass, field
from typing import Any

from .base import (
    BaseCacheOptimizer,
    BreakpointLocation,
    CacheBreakpoint,
    CacheConfig,
    CacheMetrics,
    CacheResult,
    CacheStrategy,
    OptimizationContext,
)

# Anthropic-specific constants
ANTHROPIC_MIN_CACHEABLE_TOKENS = 1024
ANTHROPIC_MAX_BREAKPOINTS = 4
ANTHROPIC_CACHE_TTL_SECONDS = 300  # 5 minutes
ANTHROPIC_WRITE_COST_MULTIPLIER = 1.25  # 25% more to write
ANTHROPIC_READ_COST_MULTIPLIER = 0.10  # 90% less to read


@dataclass
class ContentSection:
    """Represents a section of content that may be cacheable."""

    content: str | list[dict[str, Any]]
    section_type: str  # "system", "tools", "examples", "user", "assistant"
    message_index: int
    content_index: int | None = None
    token_count: int = 0
    is_cacheable: bool = False
    reason: str = ""


@dataclass
class BreakpointPlan:
    """Plan for where to insert cache breakpoints."""

    breakpoints: list[CacheBreakpoint] = field(default_factory=list)
    total_cacheable_tokens: int = 0
    estimated_savings_percent: float = 0.0
    warnings: list[str] = field(default_factory=list)


class AnthropicCacheOptimizer(BaseCacheOptimizer):
    """
    Cache optimizer for Anthropic's explicit cache_control mechanism.

    This optimizer analyzes messages and inserts cache_control blocks at
    optimal positions to maximize cache hit rates and minimize costs.

    Key features:
    - Detects cacheable sections (system prompt, tools, few-shot examples)
    - Respects Anthropic's 1024 token minimum and 4 breakpoint maximum
    - Stabilizes prefixes by moving dates and normalizing whitespace
    - Tracks metrics for monitoring and debugging
    """

    def __init__(self, config: CacheConfig | None = None):
        super().__init__(config)
        if self.config.min_cacheable_tokens < ANTHROPIC_MIN_CACHEABLE_TOKENS:
            self.config.min_cacheable_tokens = ANTHROPIC_MIN_CACHEABLE_TOKENS
        if self.config.max_breakpoints > ANTHROPIC_MAX_BREAKPOINTS:
            self.config.max_breakpoints = ANTHROPIC_MAX_BREAKPOINTS

    @property
    def name(self) -> str:
        return "anthropic-cache-optimizer"

    @property
    def provider(self) -> str:
        return "anthropic"

    @property
    def strategy(self) -> CacheStrategy:
        return CacheStrategy.EXPLICIT_BREAKPOINTS

    def optimize(
        self,
        messages: list[dict[str, Any]],
        context: OptimizationContext,
        config: CacheConfig | None = None,
    ) -> CacheResult:
        """
        Optimize messages for Anthropic's cache.

        Steps:
        1. Analyze messages to identify cacheable sections
        2. Stabilize the prefix (moves dates, normalizes whitespace)
        3. Plan breakpoint placement
        4. Insert cache_control blocks at optimal positions
        5. Record metrics for monitoring
        """
        effective_config = config or self.config

        if not effective_config.enabled:
            return CacheResult(
                messages=messages,
                metrics=CacheMetrics(),
                transforms_applied=[],
            )

        optimized_messages = copy.deepcopy(messages)
        transforms_applied: list[str] = []
        warnings: list[str] = []

        # Step 1: Analyze content sections
        sections = self._analyze_sections(optimized_messages)

        # Step 2: Stabilize prefix
        optimized_messages, stabilization_applied = self._stabilize_prefix(
            optimized_messages, effective_config
        )
        transforms_applied.extend(stabilization_applied)

        # Step 3: Plan breakpoint placement
        plan = self._plan_breakpoints(sections, effective_config)
        warnings.extend(plan.warnings)

        # Step 4: Insert cache_control blocks
        optimized_messages = self._insert_breakpoints(optimized_messages, plan.breakpoints)
        if plan.breakpoints:
            transforms_applied.append(f"inserted_{len(plan.breakpoints)}_cache_breakpoints")

        # Step 5: Compute metrics
        prefix_content = self._extract_cacheable_content(optimized_messages)
        prefix_hash = self._compute_prefix_hash(prefix_content)

        cache_hit = False
        if context.previous_prefix_hash:
            cache_hit = prefix_hash == context.previous_prefix_hash
        elif self._previous_prefix_hash:
            cache_hit = prefix_hash == self._previous_prefix_hash

        total_tokens = sum(s.token_count for s in sections)
        cacheable_tokens = plan.total_cacheable_tokens

        metrics = CacheMetrics(
            stable_prefix_tokens=cacheable_tokens,
            stable_prefix_hash=prefix_hash,
            breakpoints_inserted=len(plan.breakpoints),
            breakpoint_locations=plan.breakpoints,
            prefix_changed_from_previous=not cache_hit,
            previous_prefix_hash=self._previous_prefix_hash,
            estimated_cache_hit=cache_hit,
            estimated_savings_percent=plan.estimated_savings_percent if cache_hit else 0.0,
            cacheable_tokens=cacheable_tokens,
            non_cacheable_tokens=total_tokens - cacheable_tokens,
            cache_ttl_remaining_seconds=ANTHROPIC_CACHE_TTL_SECONDS if cache_hit else None,
        )

        self._previous_prefix_hash = prefix_hash
        self._record_metrics(metrics)

        return CacheResult(
            messages=optimized_messages,
            metrics=metrics,
            tokens_before=total_tokens,
            tokens_after=total_tokens,
            transforms_applied=transforms_applied,
            warnings=warnings,
        )

    def _analyze_sections(self, messages: list[dict[str, Any]]) -> list[ContentSection]:
        """Analyze messages to identify distinct content sections."""
        sections: list[ContentSection] = []

        for idx, message in enumerate(messages):
            role = message.get("role", "")
            content = message.get("content", "")

            if role == "system":
                section_type = "system"
            elif role == "user":
                section_type = (
                    "examples" if self._looks_like_example(message, messages, idx) else "user"
                )
            elif role == "assistant":
                section_type = (
                    "examples" if self._looks_like_example(message, messages, idx) else "assistant"
                )
            else:
                section_type = role

            # Handle tools
            if "tools" in message:
                tool_section = ContentSection(
                    content=message["tools"],
                    section_type="tools",
                    message_index=idx,
                    token_count=self._estimate_tools_tokens(message["tools"]),
                    is_cacheable=True,
                    reason="Tool definitions are static and cacheable",
                )
                sections.append(tool_section)

            if isinstance(content, str):
                token_count = self._count_tokens_estimate(content)
                is_cacheable, reason = self._assess_cacheability(section_type, token_count, content)
                sections.append(
                    ContentSection(
                        content=content,
                        section_type=section_type,
                        message_index=idx,
                        token_count=token_count,
                        is_cacheable=is_cacheable,
                        reason=reason,
                    )
                )

            elif isinstance(content, list):
                for block_idx, block in enumerate(content):
                    if isinstance(block, dict) and block.get("type") == "text":
                        text = block.get("text", "")
                        token_count = self._count_tokens_estimate(text)
                        is_cacheable, reason = self._assess_cacheability(
                            section_type, token_count, text
                        )
                        sections.append(
                            ContentSection(
                                content=block,  # type: ignore[arg-type]
                                section_type=section_type,
                                message_index=idx,
                                content_index=block_idx,
                                token_count=token_count,
                                is_cacheable=is_cacheable,
                                reason=reason,
                            )
                        )

        return sections

    def _assess_cacheability(
        self, section_type: str, token_count: int, content: str
    ) -> tuple[bool, str]:
        """Assess whether a section is cacheable."""
        if token_count < self.config.min_cacheable_tokens:
            return (
                False,
                f"Below minimum tokens ({token_count} < {self.config.min_cacheable_tokens})",
            )

        if section_type == "system":
            return True, "System prompts are highly cacheable"
        if section_type == "tools":
            return True, "Tool definitions are static and cacheable"
        if section_type == "examples":
            return True, "Few-shot examples are typically static"
        if self._has_dynamic_content(content):
            return False, "Contains dynamic content (dates, times, etc.)"
        if section_type == "user":
            return False, "User messages are typically dynamic"

        return True, "Content is large enough for caching"

    def _has_dynamic_content(self, content: str) -> bool:
        """Check if content has dynamic elements."""
        for pattern in self.config.date_patterns:
            if re.search(pattern, content):
                return True
        return False

    def _looks_like_example(
        self,
        message: dict[str, Any],
        messages: list[dict[str, Any]],
        idx: int,
    ) -> bool:
        """Determine if a message looks like a few-shot example."""
        system_idx = -1
        for i, msg in enumerate(messages):
            if msg.get("role") == "system":
                system_idx = i
                break

        if system_idx >= 0 and idx <= system_idx + 4:
            role = message.get("role")
            if role == "user" and idx + 1 < len(messages):
                if messages[idx + 1].get("role") == "assistant":
                    return True
            elif role == "assistant" and idx > 0:
                if messages[idx - 1].get("role") == "user":
                    return True

        content = message.get("content", "")
        if isinstance(content, str):
            example_markers = ["example:", "for example", "e.g.", "sample:"]
            return any(marker in content.lower() for marker in example_markers)

        return False

    def _estimate_tools_tokens(self, tools: Any) -> int:
        """Estimate token count for tool definitions."""
        import json

        try:
            return self._count_tokens_estimate(json.dumps(tools))
        except (TypeError, ValueError):
            return 0

    def _stabilize_prefix(
        self,
        messages: list[dict[str, Any]],
        config: CacheConfig,
    ) -> tuple[list[dict[str, Any]], list[str]]:
        """Stabilize the prefix by moving dynamic content."""
        transforms: list[str] = []

        for message in messages:
            if message.get("role") != "system":
                continue

            content = message.get("content", "")
            if isinstance(content, str):
                new_content, applied = self._stabilize_text(content, config)
                if new_content != content:
                    message["content"] = new_content
                    transforms.extend(applied)

            elif isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        text = block.get("text", "")
                        new_text, applied = self._stabilize_text(text, config)
                        if new_text != text:
                            block["text"] = new_text
                            transforms.extend(applied)

        return messages, transforms

    def _stabilize_text(self, text: str, config: CacheConfig) -> tuple[str, list[str]]:
        """Stabilize a text string."""
        transforms: list[str] = []
        result = text

        extracted_dates: list[str] = []
        for pattern in config.date_patterns:
            matches = re.findall(pattern, result)
            if matches:
                extracted_dates.extend(matches)
                result = re.sub(pattern, "", result)
                transforms.append("extracted_dates")

        if config.normalize_whitespace:
            new_result = re.sub(r"[ \t]+", " ", result)
            if new_result != result:
                result = new_result
                transforms.append("normalized_spaces")

        if config.collapse_blank_lines:
            new_result = re.sub(r"\n{3,}", "\n\n", result)
            if new_result != result:
                result = new_result
                transforms.append("collapsed_blank_lines")

        result = result.strip()

        if extracted_dates:
            result = result + config.dynamic_separator + " ".join(extracted_dates)

        return result, list(set(transforms))

    def _plan_breakpoints(
        self,
        sections: list[ContentSection],
        config: CacheConfig,
    ) -> BreakpointPlan:
        """Plan where to place cache breakpoints."""
        plan = BreakpointPlan()

        cacheable = [s for s in sections if s.is_cacheable]
        if not cacheable:
            plan.warnings.append("No sections meet caching requirements")
            return plan

        priority_order = {"system": 0, "tools": 1, "examples": 2}
        cacheable.sort(key=lambda s: priority_order.get(s.section_type, 3))

        selected: list[ContentSection] = []
        accumulated_tokens = 0

        for section in cacheable:
            if len(selected) >= config.max_breakpoints:
                plan.warnings.append(f"Reached maximum breakpoints ({config.max_breakpoints})")
                break

            selected.append(section)
            accumulated_tokens += section.token_count

        for section in selected:
            location = self._section_type_to_location(section.section_type)
            breakpoint = CacheBreakpoint(
                message_index=section.message_index,
                location=location,
                content_index=section.content_index,
                tokens_at_breakpoint=section.token_count,
                reason=section.reason,
            )
            plan.breakpoints.append(breakpoint)

        plan.total_cacheable_tokens = accumulated_tokens
        if accumulated_tokens > 0:
            plan.estimated_savings_percent = 90.0

        return plan

    def _section_type_to_location(self, section_type: str) -> BreakpointLocation:
        """Convert section type to breakpoint location enum."""
        mapping = {
            "system": BreakpointLocation.AFTER_SYSTEM,
            "tools": BreakpointLocation.AFTER_TOOLS,
            "examples": BreakpointLocation.AFTER_EXAMPLES,
        }
        return mapping.get(section_type, BreakpointLocation.CUSTOM)

    def _insert_breakpoints(
        self,
        messages: list[dict[str, Any]],
        breakpoints: list[CacheBreakpoint],
    ) -> list[dict[str, Any]]:
        """Insert cache_control blocks at specified positions."""
        for bp in breakpoints:
            if bp.message_index >= len(messages):
                continue

            message = messages[bp.message_index]
            content = message.get("content", "")

            if isinstance(content, str):
                message["content"] = [
                    {
                        "type": "text",
                        "text": content,
                        "cache_control": {"type": "ephemeral"},
                    }
                ]
            elif isinstance(content, list):
                if bp.content_index is not None and bp.content_index < len(content):
                    block = content[bp.content_index]
                    if isinstance(block, dict):
                        block["cache_control"] = {"type": "ephemeral"}
                elif content:
                    last_block = content[-1]
                    if isinstance(last_block, dict):
                        last_block["cache_control"] = {"type": "ephemeral"}

        return messages

    def _extract_cacheable_content(self, messages: list[dict[str, Any]]) -> str:
        """Extract content that has cache_control markers for hashing."""
        parts: list[str] = []

        for message in messages:
            content = message.get("content", "")
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and "cache_control" in block:
                        text = block.get("text", "")
                        if text:
                            parts.append(text)
            elif isinstance(content, str) and message.get("role") == "system":
                parts.append(content)

        return "\n".join(parts)

    def estimate_savings(
        self,
        messages: list[dict[str, Any]],
        context: OptimizationContext,
    ) -> float:
        """Estimate potential savings from caching."""
        sections = self._analyze_sections(messages)
        plan = self._plan_breakpoints(sections, self.config)

        if plan.total_cacheable_tokens == 0:
            return 0.0

        total_tokens = sum(s.token_count for s in sections)
        cacheable_ratio = plan.total_cacheable_tokens / total_tokens
        return 90.0 * cacheable_ratio

    def get_cache_write_cost_multiplier(self) -> float:
        return ANTHROPIC_WRITE_COST_MULTIPLIER

    def get_cache_read_cost_multiplier(self) -> float:
        return ANTHROPIC_READ_COST_MULTIPLIER

    def get_cache_ttl_seconds(self) -> int:
        return ANTHROPIC_CACHE_TTL_SECONDS
