"""An unchanged numeric prefix is not evidence that the original value survived."""

import pytest

from headroom.evals.session_probes import probe_event


@pytest.mark.parametrize(
    "compressed",
    [
        "port: 8080",
        '"port": 80.5',
        "port: -80",
        "port: 80e3",
        "| port |\n| 8080 |",
        "| port |\n| 80.5 |",
        "| port |\n| -80 |",
    ],
)
def test_changed_numeric_value_is_lost(compressed: str) -> None:
    result = probe_event(
        {
            "tokens_before": 100,
            "tokens_after": 10,
            "original_messages": [{"role": "tool", "content": "port: 80"}],
            "compressed_messages": [{"role": "tool", "content": compressed}],
        }
    )
    assert result is not None
    assert result.dims["numerics"].to_dict() == {
        "total": 1,
        "retained": 0,
        "recoverable": 0,
        "lost": 1,
    }


@pytest.mark.parametrize(
    "compressed", ["port: 80", '"port": 80', "| port |\n| 80 |", "port is 80."]
)
def test_exact_numeric_value_remains_retained(compressed: str) -> None:
    result = probe_event(
        {
            "tokens_before": 100,
            "tokens_after": 10,
            "original_messages": [{"role": "tool", "content": "port: 80"}],
            "compressed_messages": [{"role": "tool", "content": compressed}],
        }
    )
    assert result is not None
    assert result.dims["numerics"].retained == 1


def test_changed_value_with_marker_is_recoverable_not_retained() -> None:
    result = probe_event(
        {
            "tokens_before": 100,
            "tokens_after": 10,
            "original_messages": [{"role": "tool", "content": "port: 80"}],
            "compressed_messages": [{"role": "tool", "content": "port: 8080 <<ccr:abc123>>"}],
        }
    )
    assert result is not None
    assert result.dims["numerics"].retained == 0
    assert result.dims["numerics"].recoverable == 1


@pytest.mark.parametrize("number", ["80e3", "80e-3", "80e+3", "0.001e2", "0.001e-2", "80E3"])
def test_unsupported_exponent_does_not_generate_partial_probe(number: str) -> None:
    from headroom.evals.session_probes import extract_probe_targets

    text = f"port: {number}"
    assert extract_probe_targets(text)["numerics"] == set()
    result = probe_event(
        {
            "tokens_before": 100,
            "tokens_after": 10,
            "original_messages": [{"role": "tool", "content": text}],
            "compressed_messages": [{"role": "tool", "content": text}],
        }
    )
    assert result is not None
    assert result.dims["numerics"].total == 0


@pytest.mark.parametrize(
    "original,compressed,retained",
    [
        ("port: 8080", "port: 8081", 0),
        ("latency: 80.5", "latency: 80.50", 0),
        ("latency: 80.5", "latency: 80.5", 1),
        ("port: 8080", "port: 8080", 1),
    ],
)
def test_complete_integer_and_decimal_values(original: str, compressed: str, retained: int) -> None:
    result = probe_event(
        {
            "tokens_before": 100,
            "tokens_after": 10,
            "original_messages": [{"role": "tool", "content": original}],
            "compressed_messages": [{"role": "tool", "content": compressed}],
        }
    )
    assert result is not None
    assert result.dims["numerics"].total == 1
    assert result.dims["numerics"].retained == retained
