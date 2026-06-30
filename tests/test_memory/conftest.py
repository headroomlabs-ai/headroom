"""Test fixtures for Headroom Memory."""

from __future__ import annotations

# CRITICAL: Must be set before ANY imports that could trigger sentence_transformers
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import pytest

from tests._skip_helpers import external_model_skip_reason


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_call(item):
    """Wrap test execution to skip transient or offline external model failures.

    This handles model-loading failures that occur when:
    - HuggingFace Hub is slow during model downloads (sentence-transformers)
    - Required HuggingFace model files were not restored into the offline CI cache
    - External embedding APIs timeout
    - Network connectivity issues in CI
    """
    outcome = yield

    if outcome.excinfo is not None:
        exc_type, exc_value, exc_tb = outcome.excinfo
        reason = external_model_skip_reason(exc_value)
        if reason is not None:
            pytest.skip(reason)
