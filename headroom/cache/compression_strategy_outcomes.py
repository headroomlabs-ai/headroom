"""Strategy outcome accounting for local compression feedback."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class CompressionStrategyOutcomes:
    """Track compression and retrieval outcomes by compression strategy."""

    compressions: dict[str, int] = field(default_factory=dict)
    retrievals: dict[str, int] = field(default_factory=dict)
    max_strategies: int = 50
    top_strategies_per_counter: int = 40
    minimum_samples_for_recommendation: int = 3

    def record_compression(self, strategy: str) -> None:
        """Record one compression for a strategy."""
        self.compressions[strategy] = self.compressions.get(strategy, 0) + 1
        self.prune()

    def record_retrieval(self, strategy: str) -> None:
        """Record one retrieval for a strategy."""
        self.retrievals[strategy] = self.retrievals.get(strategy, 0) + 1
        self.prune()

    def retrieval_rate(self, strategy: str) -> float:
        """Return the retrievals-per-compression rate for one strategy."""
        compressions = self.compressions.get(strategy, 0)
        if compressions == 0:
            return 0.0
        return self.retrievals.get(strategy, 0) / compressions

    def best_strategy(self) -> str | None:
        """Return the sampled strategy with the lowest retrieval rate."""
        best = None
        best_rate = 1.0

        for strategy, compression_count in self.compressions.items():
            if compression_count < self.minimum_samples_for_recommendation:
                continue

            rate = self.retrieval_rate(strategy)
            if rate < best_rate:
                best = strategy
                best_rate = rate

        return best

    def prune(self) -> None:
        """Bound counters while preserving the highest-signal strategies."""
        if (
            len(self.compressions) <= self.max_strategies
            and len(self.retrievals) <= self.max_strategies
        ):
            return

        keys_to_keep = self._keys_to_keep()
        self.compressions = {
            strategy: count
            for strategy, count in self.compressions.items()
            if strategy in keys_to_keep
        }
        self.retrievals = {
            strategy: count
            for strategy, count in self.retrievals.items()
            if strategy in keys_to_keep
        }

    def _keys_to_keep(self) -> set[str]:
        top_compressions = self._top_keys(self.compressions)
        top_retrievals = self._top_keys(self.retrievals)
        candidate_keys = top_compressions | top_retrievals

        if len(candidate_keys) <= self.max_strategies:
            return candidate_keys

        ranked_keys = sorted(
            candidate_keys,
            key=lambda strategy: (
                self.compressions.get(strategy, 0) + self.retrievals.get(strategy, 0),
                self.compressions.get(strategy, 0),
                self.retrievals.get(strategy, 0),
                strategy,
            ),
            reverse=True,
        )
        return set(ranked_keys[: self.max_strategies])

    def _top_keys(self, counts: dict[str, int]) -> set[str]:
        return {
            strategy
            for strategy, _ in sorted(
                counts.items(),
                key=lambda item: (item[1], item[0]),
                reverse=True,
            )[: self.top_strategies_per_counter]
        }
