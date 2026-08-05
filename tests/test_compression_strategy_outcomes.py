from headroom.cache.compression_strategy_outcomes import CompressionStrategyOutcomes


def test_retrieval_rate_is_zero_without_strategy_compressions():
    outcomes = CompressionStrategyOutcomes(retrievals={"sample": 2})

    assert outcomes.retrieval_rate("sample") == 0.0


def test_best_strategy_requires_minimum_samples():
    outcomes = CompressionStrategyOutcomes(
        compressions={"under_sampled": 2, "sampled": 3},
        retrievals={"under_sampled": 0, "sampled": 1},
    )

    assert outcomes.best_strategy() == "sampled"


def test_best_strategy_uses_lowest_retrieval_rate():
    outcomes = CompressionStrategyOutcomes(
        compressions={"top_n": 10, "smart_sample": 10},
        retrievals={"top_n": 7, "smart_sample": 2},
    )

    assert outcomes.retrieval_rate("smart_sample") == 0.2
    assert outcomes.best_strategy() == "smart_sample"


def test_recording_prunes_strategy_counters_to_bounded_high_signal_set():
    outcomes = CompressionStrategyOutcomes(max_strategies=10, top_strategies_per_counter=8)

    for index in range(30):
        strategy = f"strategy_{index:02d}"
        for _ in range(index + 1):
            outcomes.record_compression(strategy)

    for index in range(30):
        strategy = f"strategy_{index:02d}"
        for _ in range(30 - index):
            outcomes.record_retrieval(strategy)

    assert len(outcomes.compressions) <= 10
    assert len(outcomes.retrievals) <= 10
    assert "strategy_29" in outcomes.compressions
    assert "strategy_00" in outcomes.retrievals
