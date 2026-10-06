"""The OpenAI Responses memory tool-call loops must not crash on a null
``arguments``.

A ``function_call`` item with ``"arguments": null`` makes ``fc.get("arguments",
"{}")`` return ``None``, and ``json.loads(None)`` raises ``TypeError`` — which the
``except json.JSONDecodeError`` around it does not catch. The two memory
tool-execution loops in ``handlers/openai.py`` are deep inside streaming request
handlers, so this guards the fix at the source level (reading the file, not
importing the ML stack) plus a behavioural proof in the standalone script.
"""

from __future__ import annotations

from pathlib import Path

_OPENAI = Path(__file__).resolve().parents[1] / "headroom" / "proxy" / "handlers" / "openai.py"


def test_memory_tool_argument_parsing_is_null_safe():
    src = _OPENAI.read_text(encoding="utf-8")

    # The vulnerable form (bare default + json.loads that can receive None) is gone.
    assert 'fc.get("arguments", "{}")' not in src

    # Both memory tool-call loops now coalesce the arguments string and catch
    # TypeError alongside JSONDecodeError.
    assert src.count('fc.get("arguments") or "{}"') >= 2
    assert src.count("except (json.JSONDecodeError, TypeError):") >= 2
