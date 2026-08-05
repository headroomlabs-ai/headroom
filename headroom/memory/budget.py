"""Memory file budget manager — token-optimized memory file maintenance.

Manages the token budget for agent memory files:
- Ranks memories by importance × recency × access frequency
- Prunes memories that fall below budget threshold
- Detects stale memories referencing deleted/renamed entities
- Merges similar memories to save tokens

This is Headroom's superpower applied to itself — the proxy that optimizes
LLM context also optimizes its own memory files for minimum token consumption.
"""

from __future__ import annotations

import logging
import re
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

from headroom._subprocess import run
from headroom.memory.writers.base import MemoryEntry, _estimate_tokens

logger = logging.getLogger(__name__)


@dataclass
class BudgetConfig:
    """Configuration for memory budget management."""

    # Token budgets per agent type
    agent_budgets: dict[str, int] = field(
        default_factory=lambda: {
            "claude": 2000,  # Claude Code: 200-line MEMORY.md limit
            "cursor": 3000,  # Cursor: .mdc rules loaded on match
            "codex": 3000,  # Codex: AGENTS.md merged
            "aider": 2000,  # Aider: read files
            "gemini": 3000,  # Gemini: GEMINI.md
            "generic": 3000,  # Default
        }
    )

    # Staleness detection
    staleness_decay_rate: float = 0.1  # Importance decays ~10% per day
    staleness_min_importance: float = 0.15  # Below this = prunable
    staleness_check_git: bool = True  # Check git for deleted files

    # Deduplication
    similarity_merge_threshold: float = 0.85  # Cosine sim above = merge


@dataclass
class BudgetReport:
    """Report from a budget optimization pass."""

    total_memories: int = 0
    kept: int = 0
    pruned_staleness: int = 0
    pruned_budget: int = 0
    merged: int = 0
    tokens_before: int = 0
    tokens_after: int = 0

    @property
    def tokens_saved(self) -> int:
        return self.tokens_before - self.tokens_after


class MemoryBudgetManager:
    """Manages token budgets for agent memory files.

    Responsible for deciding which memories to keep, prune, or merge
    to stay within token budgets while maximizing information density.
    """

    def __init__(
        self,
        project_path: Path | None = None,
        config: BudgetConfig | None = None,
    ) -> None:
        self._project_path = project_path or Path.cwd()
        self._config = config or BudgetConfig()
        self._git_files_cache: set[str] | None = None

    def optimize(
        self,
        memories: list[MemoryEntry],
        agent_type: str = "generic",
    ) -> tuple[list[MemoryEntry], BudgetReport]:
        """Optimize a set of memories for an agent's token budget.

        Returns the optimized list and a report of what changed.

        Args:
            memories: All candidate memories.
            agent_type: Target agent type (determines budget).

        Returns:
            Tuple of (optimized memories, report).
        """
        report = BudgetReport(total_memories=len(memories))
        report.tokens_before = sum(_estimate_tokens(m.content) for m in memories)

        budget = self._config.agent_budgets.get(agent_type, 3000)

        # Step 1: Apply temporal decay to importance scores
        decayed = self._apply_decay(memories)

        # Step 2: Detect and flag stale memories
        fresh, stale = self._detect_staleness(decayed)
        report.pruned_staleness = len(stale)

        # Step 3: Merge similar memories
        merged = self._merge_similar(fresh)
        report.merged = len(fresh) - len(merged)

        # Step 4: Rank by score and apply budget
        ranked = sorted(merged, key=lambda m: m.score, reverse=True)
        budgeted: list[MemoryEntry] = []
        tokens_used = 0
        for m in ranked:
            entry_tokens = _estimate_tokens(m.content) + 10
            if tokens_used + entry_tokens > budget:
                report.pruned_budget += 1
                continue
            tokens_used += entry_tokens
            budgeted.append(m)

        report.kept = len(budgeted)
        report.tokens_after = tokens_used

        return budgeted, report

    def _apply_decay(self, memories: list[MemoryEntry]) -> list[MemoryEntry]:
        """Apply temporal decay to importance scores."""
        now = time.time()
        rate = self._config.staleness_decay_rate
        min_importance = self._config.staleness_min_importance

        result: list[MemoryEntry] = []
        for m in memories:
            age_days = (now - m.created_at) / 86400
            # Exponential decay: importance × e^(-rate × age)
            import math

            decayed_importance = m.importance * math.exp(-rate * age_days)

            # Access count counteracts decay
            if m.access_count > 0:
                access_boost = min(0.3, m.access_count * 0.05)
                decayed_importance = min(1.0, decayed_importance + access_boost)

            if decayed_importance >= min_importance:
                m.importance = decayed_importance
                result.append(m)

        return result

    def _detect_staleness(
        self,
        memories: list[MemoryEntry],
    ) -> tuple[list[MemoryEntry], list[MemoryEntry]]:
        """Separate fresh memories from stale ones.

        A memory is stale if it references files/paths that no longer exist.
        Uses git ls-files when available, falls back to filesystem checks.
        """
        git_files = self._get_git_files() if self._config.staleness_check_git else set()

        fresh: list[MemoryEntry] = []
        stale: list[MemoryEntry] = []

        for m in memories:
            if self._is_stale(m, git_files):
                stale.append(m)
            else:
                fresh.append(m)

        return fresh, stale

    def _is_stale(self, memory: MemoryEntry, git_files: set[str]) -> bool:
        """Check if a memory references entities that no longer exist."""
        # Check entity_refs for file paths
        for ref in memory.entity_refs:
            if ref.startswith("/") or ref.startswith("./"):
                # Absolute or relative path — check if file exists
                if ref.startswith("./"):
                    ref = ref[2:]
                # Normalize to relative
                try:
                    rel = str(Path(ref).relative_to(self._project_path))
                except ValueError:
                    rel = ref

                if rel not in git_files and not Path(ref).exists():
                    return True

        # Check content for backtick-quoted file paths
        path_refs = re.findall(r"`([/\w.]+(?:/[\w.]+)+)`", memory.content)
        for path_ref in path_refs:
            # Only flag if it looks like a full path and is missing
            if "/" in path_ref:
                try:
                    rel = str(Path(path_ref).relative_to(self._project_path))
                except ValueError:
                    rel = path_ref
                if rel not in git_files and not Path(path_ref).exists():
                    return True

        return False

    def _get_git_files(self) -> set[str]:
        """Get set of tracked files in the git repo."""
        if self._git_files_cache is not None:
            return self._git_files_cache

        try:
            result = run(
                ["git", "ls-files"],
                capture_output=True,
                text=True,
                cwd=self._project_path,
                timeout=5,
            )
            if result.returncode == 0:
                self._git_files_cache = set(result.stdout.strip().split("\n"))
            else:
                self._git_files_cache = set()
        except (subprocess.TimeoutExpired, FileNotFoundError):
            self._git_files_cache = set()

        return self._git_files_cache

    def _merge_similar(self, memories: list[MemoryEntry]) -> list[MemoryEntry]:
        """Merge memories with very similar content.

        Uses simple text overlap heuristic (no embeddings required).
        """
        if len(memories) <= 1:
            return memories

        merged: list[MemoryEntry] = []
        used: set[int] = set()

        for i, m1 in enumerate(memories):
            if i in used:
                continue

            # Find similar memories
            group = [m1]
            for j, m2 in enumerate(memories[i + 1 :], start=i + 1):
                if j in used:
                    continue
                if (
                    self._text_similarity(m1.content, m2.content)
                    > self._config.similarity_merge_threshold
                ):
                    group.append(m2)
                    used.add(j)

            if len(group) == 1:
                merged.append(m1)
            else:
                # Merge: keep highest-importance content, combine entity refs
                best = max(group, key=lambda m: m.importance)
                all_entities = set()
                for m in group:
                    all_entities.update(m.entity_refs)
                best.entity_refs = list(all_entities)
                best.access_count = max(m.access_count for m in group)
                merged.append(best)

        return merged

    @staticmethod
    def _text_similarity(a: str, b: str) -> float:
        """Simple Jaccard similarity on word sets."""
        words_a = set(a.lower().split())
        words_b = set(b.lower().split())
        if not words_a or not words_b:
            return 0.0
        intersection = words_a & words_b
        union = words_a | words_b
        return len(intersection) / len(union)
