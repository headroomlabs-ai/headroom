from __future__ import annotations

import httpx
from huggingface_hub.errors import LocalEntryNotFoundError

from tests._skip_helpers import external_model_skip_reason


def test_external_model_skip_reason_handles_httpx_timeout() -> None:
    reason = external_model_skip_reason(httpx.ReadTimeout("slow network"))
    assert reason == "Skipped due to network timeout (flaky CI)"


def test_external_model_skip_reason_handles_hf_local_cache_miss() -> None:
    reason = external_model_skip_reason(LocalEntryNotFoundError("not cached"))
    assert reason == "Skipped because required Hugging Face model files are unavailable offline"


def test_external_model_skip_reason_handles_transformers_offline_oserror() -> None:
    exc = OSError(
        "We couldn't connect to 'https://huggingface.co' to load the files, "
        "and couldn't find them in the cached files."
    )
    reason = external_model_skip_reason(exc)
    assert reason == "Skipped because required Hugging Face model files are unavailable offline"


def test_external_model_skip_reason_ignores_unrelated_errors() -> None:
    assert external_model_skip_reason(RuntimeError("boom")) is None
