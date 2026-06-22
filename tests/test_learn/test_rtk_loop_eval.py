"""CI wrapper for the RTK-loop eval (benchmarks/rtk_loop_learn_eval.py).

The deterministic path runs everywhere and gates the loop-weighting behavior
end-to-end. The real-LLM path is opt-in via the repo's ``real_llm`` marker and
only runs when an API key is present.
"""

import os

import pytest

from benchmarks.rtk_loop_learn_eval import run_eval


def test_rtk_loop_eval_deterministic():
    card = run_eval(use_real_llm=False)
    assert card.passed, "RTK-loop eval failed:\n" + card.render()


@pytest.mark.real_llm
@pytest.mark.skipif(
    not os.environ.get("ANTHROPIC_API_KEY"),
    reason="real_llm eval needs ANTHROPIC_API_KEY",
)
def test_rtk_loop_eval_real_llm():
    card = run_eval(use_real_llm=True)
    assert card.passed, "RTK-loop eval (real LLM) failed:\n" + card.render()
