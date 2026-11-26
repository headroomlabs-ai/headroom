"""HuggingFace tokenizer wrapper for open models.

Supports Llama, Mistral, Falcon, and other models with HuggingFace
tokenizers. Requires the `transformers` library.
"""

from __future__ import annotations

import logging
import os
import threading
from functools import lru_cache
from typing import Any

from .base import BaseTokenizer

logger = logging.getLogger(__name__)


# Model name to HuggingFace tokenizer mapping
# Maps common model names to their HuggingFace tokenizer identifiers
MODEL_TO_TOKENIZER: dict[str, str] = {
    # Llama 3 family
    "llama-3": "meta-llama/Meta-Llama-3-8B",
    "llama-3-8b": "meta-llama/Meta-Llama-3-8B",
    "llama-3-70b": "meta-llama/Meta-Llama-3-70B",
    "llama-3.1-8b": "meta-llama/Llama-3.1-8B",
    "llama-3.1-70b": "meta-llama/Llama-3.1-70B",
    "llama-3.1-405b": "meta-llama/Llama-3.1-405B",
    "llama-3.2-1b": "meta-llama/Llama-3.2-1B",
    "llama-3.2-3b": "meta-llama/Llama-3.2-3B",
    "llama-3.3-70b": "meta-llama/Llama-3.3-70B-Instruct",
    # Llama 2 family
    "llama-2": "meta-llama/Llama-2-7b-hf",
    "llama-2-7b": "meta-llama/Llama-2-7b-hf",
    "llama-2-13b": "meta-llama/Llama-2-13b-hf",
    "llama-2-70b": "meta-llama/Llama-2-70b-hf",
    # CodeLlama
    "codellama": "codellama/CodeLlama-7b-hf",
    "codellama-7b": "codellama/CodeLlama-7b-hf",
    "codellama-13b": "codellama/CodeLlama-13b-hf",
    "codellama-34b": "codellama/CodeLlama-34b-hf",
    # Mistral family
    "mistral": "mistralai/Mistral-7B-v0.1",
    "mistral-7b": "mistralai/Mistral-7B-v0.1",
    "mistral-7b-v0.2": "mistralai/Mistral-7B-Instruct-v0.2",
    "mistral-7b-v0.3": "mistralai/Mistral-7B-Instruct-v0.3",
    "mistral-nemo": "mistralai/Mistral-Nemo-Base-2407",
    "mistral-small": "mistralai/Mistral-Small-Instruct-2409",
    "mistral-large": "mistralai/Mistral-Large-Instruct-2407",
    # Mixtral
    "mixtral": "mistralai/Mixtral-8x7B-v0.1",
    "mixtral-8x7b": "mistralai/Mixtral-8x7B-v0.1",
    "mixtral-8x22b": "mistralai/Mixtral-8x22B-v0.1",
    # Qwen family
    "qwen": "Qwen/Qwen-7B",
    "qwen-7b": "Qwen/Qwen-7B",
    "qwen-14b": "Qwen/Qwen-14B",
    "qwen-72b": "Qwen/Qwen-72B",
    "qwen2": "Qwen/Qwen2-7B",
    "qwen2-7b": "Qwen/Qwen2-7B",
    "qwen2-72b": "Qwen/Qwen2-72B",
    "qwen2.5": "Qwen/Qwen2.5-7B",
    "qwen2.5-7b": "Qwen/Qwen2.5-7B",
    "qwen2.5-72b": "Qwen/Qwen2.5-72B",
    # DeepSeek V1 / Coder (legacy, 2023-2024)
    "deepseek": "deepseek-ai/deepseek-llm-7b-base",
    "deepseek-7b": "deepseek-ai/deepseek-llm-7b-base",
    "deepseek-67b": "deepseek-ai/deepseek-llm-67b-base",
    "deepseek-coder": "deepseek-ai/deepseek-coder-6.7b-base",
    "deepseek-coder-v2": "deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct",
    "deepseek-coder-v2-lite": "deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct",
    # DeepSeek V2/V3 family (2024)
    "deepseek-v2": "deepseek-ai/DeepSeek-V2",
    "deepseek-v2-lite": "deepseek-ai/DeepSeek-V2-Lite",
    "deepseek-v3": "deepseek-ai/DeepSeek-V3",
    "deepseek-v3-0324": "deepseek-ai/DeepSeek-V3-0324",
    "deepseek-v3.2": "deepseek-ai/DeepSeek-V3.2",
    # DeepSeek R1 reasoning family (2025)
    "deepseek-r1": "deepseek-ai/DeepSeek-R1",
    "deepseek-r1-0528": "deepseek-ai/DeepSeek-R1-0528",
    "deepseek-reasoner": "deepseek-ai/DeepSeek-R1",
    # DeepSeek V4 family (2025-2026). The retired v4-flash ids are still
    # accepted on the wire but served by V4.1-Flash, so they resolve there.
    "deepseek-flash": "deepseek-ai/DeepSeek-V4.1-Flash",
    "deepseek-v4-pro": "deepseek-ai/DeepSeek-V4-Pro",
    "deepseek-v4-flash": "deepseek-ai/DeepSeek-V4.1-Flash",
    "deepseek-v4-flash-vision-exp": "deepseek-ai/DeepSeek-V4.1-Flash",
    # DeepSeek API aliases (routed through the proxy)
    "deepseek-chat": "deepseek-ai/DeepSeek-V3",
    "deepseek-r1-distill-qwen": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    "deepseek-r1-distill-llama": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    # Yi family
    "yi": "01-ai/Yi-6B",
    "yi-6b": "01-ai/Yi-6B",
    "yi-34b": "01-ai/Yi-34B",
    "yi-1.5": "01-ai/Yi-1.5-6B",
    # Phi family
    "phi-2": "microsoft/phi-2",
    "phi-3": "microsoft/Phi-3-mini-4k-instruct",
    "phi-3-mini": "microsoft/Phi-3-mini-4k-instruct",
    "phi-3-small": "microsoft/Phi-3-small-8k-instruct",
    "phi-3-medium": "microsoft/Phi-3-medium-4k-instruct",
    # Falcon
    "falcon": "tiiuae/falcon-7b",
    "falcon-7b": "tiiuae/falcon-7b",
    "falcon-40b": "tiiuae/falcon-40b",
    "falcon-180b": "tiiuae/falcon-180B",
    # StarCoder
    "starcoder": "bigcode/starcoder",
    "starcoder2": "bigcode/starcoder2-15b",
    "starcoder2-3b": "bigcode/starcoder2-3b",
    "starcoder2-7b": "bigcode/starcoder2-7b",
    "starcoder2-15b": "bigcode/starcoder2-15b",
    # MPT
    "mpt-7b": "mosaicml/mpt-7b",
    "mpt-30b": "mosaicml/mpt-30b",
    # Gemma
    "gemma": "google/gemma-7b",
    "gemma-2b": "google/gemma-2b",
    "gemma-7b": "google/gemma-7b",
    "gemma-2": "google/gemma-2-9b",
    "gemma-2-9b": "google/gemma-2-9b",
    "gemma-2-27b": "google/gemma-2-27b",
}


# Every repository Headroom is willing to hand to the HuggingFace loader.
#
# The loader resolves a name against the Hub and, historically, ran repository
# code while doing it (``trust_remote_code``). The model id reaching this module
# can come straight off a client request body — the proxy routes ``model`` to a
# tokenizer — so an unconstrained identifier is remote code execution in the
# proxy process by way of "publish a repo, then ask for it by name". The
# allowlist is derived from the mappings we actually ship, so adding a family to
# MODEL_TO_TOKENIZER allows it and nothing else drifts in.
ALLOWED_TOKENIZER_REPOS: frozenset[str] = frozenset(MODEL_TO_TOKENIZER.values())

# The tokenizer an unrecognised model resolves to. Llama 3 is the canonical open
# model vocabulary and is what the registry already routes bare ``llama*`` ids
# to. The repository is gated on the Hub, so a deployment without HF credentials
# simply fails the load and falls back to character estimation — which is what
# an unknown model got before this allowlist existed.
DEFAULT_TOKENIZER: str = MODEL_TO_TOKENIZER["llama-3"]

# Immutable commit SHAs for the allowlisted repositories (GH #4939).
#
# Without these, a cache miss resolved ``main`` and the same repo id could hand
# back different ``tokenizer.json`` bytes a month later — the vocabulary behind
# every token count in the proxy would drift with no change on our side. Pinned
# here rather than in ``onnx_runtime._PINNED_REVISIONS`` because this table is
# keyed by the allowlist above: a repo is pinned exactly when we ship it, so
# adding a family to MODEL_TO_TOKENIZER without a SHA fails
# ``test_every_shipped_tokenizer_repo_is_pinned`` instead of silently floating.
#
# Refresh deliberately: bump the SHA, re-run the tests, ship it. ``HEADROOM_HF_PIN=off``
# floats every repository again (same switch the ONNX/model pins use), for
# evaluating a newer upstream revision before adopting it.
#
# ``mosaicml/mpt-7b`` and ``mosaicml/mpt-30b`` are absent on purpose: both repos
# are gone from the Hub (401/404), so there is no commit to pin and no download
# to make reproducible. They keep failing closed to estimation, as before.
_TOKENIZER_REVISIONS: dict[str, str] = {
    "meta-llama/Meta-Llama-3-8B": "8cde5ca8380496c9a6cc7ef3a8b46a0372a1d920",
    "meta-llama/Meta-Llama-3-70B": "c82494877ce7f6d7d317c56ec081328e382c72fe",
    "meta-llama/Llama-3.1-8B": "d04e592bb4f6aa9cfee91e2e20afa771667e1d4b",
    "meta-llama/Llama-3.1-70B": "349b2ddb53ce8f2849a6c168a81980ab25258dac",
    "meta-llama/Llama-3.1-405B": "b906e4dc842aa489c962f9db26554dcfdde901fe",
    "meta-llama/Llama-3.2-1B": "4e20de362430cd3b72f300e6b0f18e50e7166e08",
    "meta-llama/Llama-3.2-3B": "13afe5124825b4f3751f836b40dafda64c1ed062",
    "meta-llama/Llama-3.3-70B-Instruct": "6f6073b423013f6a7d4d9f39144961bfbfbc386b",
    "meta-llama/Llama-2-7b-hf": "01c7f73d771dfac7d292323805ebc428287df4f9",
    "meta-llama/Llama-2-13b-hf": "5c31dfb671ce7cfe2d7bb7c04375e44c55e815b1",
    "meta-llama/Llama-2-70b-hf": "3aba440b59558f995867ba6e1f58f21d0336b5bb",
    "codellama/CodeLlama-7b-hf": "6c284d1468fe6c413cf56183e69b194dcfa27fe6",
    "codellama/CodeLlama-13b-hf": "8da65ff4ee20f74ecd107ca9d54f9f121b279860",
    "codellama/CodeLlama-34b-hf": "6008b9656730b71c7d19a15370c7ff6d2902f4ef",
    "mistralai/Mistral-7B-v0.1": "27d67f1b5f57dc0953326b2601d68371d40ea8da",
    "mistralai/Mistral-7B-Instruct-v0.2": "63a8b081895390a26e140280378bc85ec8bce07a",
    "mistralai/Mistral-7B-Instruct-v0.3": "c170c708c41dac9275d15a8fff4eca08d52bab71",
    "mistralai/Mistral-Nemo-Base-2407": "a4477a2f977929a969745b69bbd62e03043551a5",
    "mistralai/Mistral-Small-Instruct-2409": "4600506f6b13c7ef89e61a54263f4c9bf483de30",
    "mistralai/Mistral-Large-Instruct-2407": "a286006d554cb37a61d13c7ae61bc90cc1d372fc",
    "mistralai/Mixtral-8x7B-v0.1": "fc7ac94680e38d7348cfa806e51218e6273104b0",
    "mistralai/Mixtral-8x22B-v0.1": "e1cd34ff1747406fb2277635ed25242803009bc2",
    "Qwen/Qwen-7B": "ef3c5c9c57b252f3149c1408daf4d649ec8b6c85",
    "Qwen/Qwen-14B": "c4051215126d906ac22bb67fe5edb39a921cd831",
    "Qwen/Qwen-72B": "b8e18ac61df64d35308695769ff46b976b6a00f4",
    "Qwen/Qwen2-7B": "453ed1575b739b5b03ce3758b23befdb0967f40e",
    "Qwen/Qwen2-72B": "5f149cae83c11c408a953dec64b12e7e662e8cb9",
    "Qwen/Qwen2.5-7B": "d149729398750b98c0af14eb82c78cfe92750796",
    "Qwen/Qwen2.5-72B": "efba10c8e54e91e0d9570ab5f7b51a958474d4cb",
    "deepseek-ai/deepseek-llm-7b-base": "7683fea62db869066ddaff6a41d032262c490d4f",
    "deepseek-ai/deepseek-llm-67b-base": "c3f813a1121c95488a20132d3a4da89f4a46452f",
    "deepseek-ai/deepseek-coder-6.7b-base": "ce2207a8bfef3ee92bd7dd4cc31c52cfa0046912",
    "deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct": "e434a23f91ba5b4923cf6c9d9a238eb4a08e3a11",
    "deepseek-ai/DeepSeek-V2": "4461458f186c35188585855f28f77af5661ad489",
    "deepseek-ai/DeepSeek-V2-Lite": "604d5664dddd88a0433dbae533b7fe9472482de0",
    "deepseek-ai/DeepSeek-V3": "e815299b0bcbac849fa540c768ef21845365c9eb",
    "deepseek-ai/DeepSeek-V3-0324": "e9b33add76883f293d6bf61f6bd89b497e80e335",
    "deepseek-ai/DeepSeek-V3.2": "a7e62ac04ecb2c0a54d736dc46601c5606cf10a6",
    "deepseek-ai/DeepSeek-R1": "56d4cbbb4d29f4355bab4b9a39ccb717a14ad5ad",
    "deepseek-ai/DeepSeek-R1-0528": "4236a6af538feda4548eca9ab308586007567f52",
    "deepseek-ai/DeepSeek-V4.1-Flash": "314113f8a7dadb388215396ba5fbee2c610a04d6",
    "deepseek-ai/DeepSeek-V4-Pro": "b5968e9190ef611bbf34a7229255be88a0e937c1",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B": "916b56a44061fd5cd7d6a8fb632557ed4f724f60",
    "deepseek-ai/DeepSeek-R1-Distill-Llama-8B": "6a6f4aa4197940add57724a7707d069478df56b1",
    "01-ai/Yi-6B": "80080be87ec5a0103f643195f2d9003b8068941b",
    "01-ai/Yi-34B": "e1fa7c83283f2e1d5a12cb4adea37c3f829e28f8",
    "01-ai/Yi-1.5-6B": "157a3d77ab5a8a8f079510eaa25169b547481f06",
    "microsoft/phi-2": "810d367871c1d460086d9f82db8696f2e0a0fcd0",
    "microsoft/Phi-3-mini-4k-instruct": "f39ac1d28e925b323eae81227eaba4464caced4e",
    "microsoft/Phi-3-small-8k-instruct": "188b876e997ee4055df01afe46274c14c400fe40",
    "microsoft/Phi-3-medium-4k-instruct": "48d87cd5e0523b430d77e93becd3655cd6897230",
    "tiiuae/falcon-7b": "ec89142b67d748a1865ea4451372db8313ada0d8",
    "tiiuae/falcon-40b": "05ab2ee8d6b593bdbab17d728de5c028a7a94d83",
    "tiiuae/falcon-180B": "d2ea5531862d4fe907280234990e6380d2befd97",
    "bigcode/starcoder": "827cd7721231872c152e3faef5cae1b22964ed58",
    "bigcode/starcoder2-15b": "46d44742909c03ac8cee08eb03fdebce02e193ec",
    "bigcode/starcoder2-3b": "733247c55e3f73af49ce8e9c7949bf14af205928",
    "bigcode/starcoder2-7b": "bb9afde76d7945da5745592525db122d4d729eb1",
    "google/gemma-7b": "ff6768d9368919a1f025a54f9f5aa0ee591730bb",
    "google/gemma-2b": "9cf48e52b224239de00d483ec8eb84fb8d0f3a3a",
    "google/gemma-2-9b": "33c193028431c2fde6c6e51f29e6f17b60cbfac6",
    "google/gemma-2-27b": "938270f5272feb02779b55c2bb2fffdd0f53ff0c",
}

# Operator escape hatch for self-hosted or private tokenizer repositories:
# a comma-separated list of repo ids. This is deliberately environment-only —
# server-side configuration an operator sets, never anything a client can
# influence through a request body. Repositories added here are still loaded
# with ``trust_remote_code=False``, and float on the default branch: a pin is a
# decision about a repository we ship, and we cannot choose one for an id we
# only learned about from the environment. Set ``HEADROOM_HF_TOKENIZER_REVISIONS``
# (``repo=sha,repo=sha``) to pin an operator repo as well.
_ALLOWLIST_ENV = "HEADROOM_HF_TOKENIZER_ALLOWLIST"
_REVISIONS_ENV = "HEADROOM_HF_TOKENIZER_REVISIONS"


def _operator_revisions(extra: str) -> dict[str, str]:
    """Parse ``repo=sha,...`` from the operator env knob, ignoring malformed pairs."""
    out: dict[str, str] = {}
    for part in extra.split(","):
        repo, sep, sha = part.partition("=")
        if sep and repo.strip() and sha.strip():
            out[repo.strip()] = sha.strip()
    return out


def _revision_for(repo: str) -> str | None:
    """The commit SHA to fetch ``repo`` at, or ``None`` for the default branch.

    Precedence: the operator's ``HEADROOM_HF_TOKENIZER_REVISIONS`` entry, then
    our shipped pin, and no pin at all when ``HEADROOM_HF_PIN`` is off. Raises
    nothing — an unpinned repository loads exactly as it did before.
    """
    from headroom.onnx_runtime import _resolve_revision

    operator = _operator_revisions(os.environ.get(_REVISIONS_ENV, ""))
    return _resolve_revision(repo, operator.get(repo), _TOKENIZER_REVISIONS)


@lru_cache(maxsize=8)
def _allowlist_index(extra: str) -> dict[str, str]:
    """Lowercased repo id -> canonical repo id, for the shipped list plus ``extra``.

    Keyed on the raw environment value so the (tiny) index is built once per
    distinct configuration rather than on every model resolution.
    """
    repos = set(ALLOWED_TOKENIZER_REPOS)
    repos.update(part.strip() for part in extra.split(",") if part.strip())
    return {repo.lower(): repo for repo in repos}


def _resolve_allowed_repo(name: str) -> str | None:
    """Return the canonical allowlisted repo id for ``name``, or None.

    Matching is case-insensitive because Hub ids are commonly retyped with
    different capitalisation, but the value returned is always *our* spelling
    from the allowlist — the caller's string is never propagated to the loader.
    """
    if not name:
        return None
    return _allowlist_index(os.environ.get(_ALLOWLIST_ENV, "")).get(name.strip().lower())


# Bound the first (network) load of a HuggingFace tokenizer. Without a bound,
# huggingface_hub download retries can block for many minutes (GH #1701: 610s
# on a restricted Windows network). 0 disables network loads entirely.
_LOAD_TIMEOUT_ENV = "HEADROOM_HF_TOKENIZER_LOAD_TIMEOUT_SECS"
_LOAD_TIMEOUT_DEFAULT = 10.0


def _load_timeout_secs() -> float:
    try:
        return float(os.environ.get(_LOAD_TIMEOUT_ENV, _LOAD_TIMEOUT_DEFAULT))
    except (TypeError, ValueError):
        return _LOAD_TIMEOUT_DEFAULT


@lru_cache(maxsize=16)
def _load_tokenizer(tokenizer_name: str):
    """Load and cache HuggingFace tokenizer.

    The first attempt is cache-only (``local_files_only=True``) so a warm
    HF cache never touches the network. A cache miss falls through to a
    network download bounded by ``HEADROOM_HF_TOKENIZER_LOAD_TIMEOUT_SECS``
    (default 10s) on a daemon thread — the download itself cannot be
    cancelled, but the caller unblocks and falls back to estimation.
    Failures are cached by ``lru_cache`` (returns ``None``), so a slow or
    offline hub is probed at most once per process per tokenizer.

    Only repositories on the allowlist reach ``from_pretrained`` at all. This is
    the chokepoint, not a second opinion: ``get_tokenizer_name`` already resolves
    unrecognised models to DEFAULT_TOKENIZER, but this function is importable and
    callable with an arbitrary string, and a name that is not on the allowlist
    must never turn into a Hub lookup. Refusing here fails closed to estimation
    rather than substituting a different vocabulary behind the caller's back.

    Args:
        tokenizer_name: HuggingFace model/tokenizer name.

    Returns:
        Loaded tokenizer, or None if unavailable.
    """
    repo = _resolve_allowed_repo(tokenizer_name)
    if repo is None:
        logger.warning(
            f"Refusing to load unallowlisted tokenizer {tokenizer_name!r}; using "
            f"estimation (add it to {_ALLOWLIST_ENV} if this repository is trusted)"
        )
        return None
    # Load the allowlist's own spelling, never the argument: matching is
    # case-insensitive, so the string that reaches the Hub must be the one we
    # vetted, not a variant the caller chose.
    tokenizer_name = repo
    # Immutable revision for this repository (GH #4939). Both attempts use it,
    # including the cache-only one: a cache seeded at a floating `main` is a
    # different snapshot, and loading it would defeat the pin.
    revision = _revision_for(tokenizer_name)

    from headroom.offline import apply_offline_env, guard_egress, is_offline

    offline = is_offline()
    if offline:
        apply_offline_env()

    from transformers import AutoTokenizer

    try:
        # trust_remote_code=False, always. A tokenizer repository can ship its own
        # Python, and executing it is equivalent to running whatever the repo owner
        # publishes inside the proxy. Every tokenizer we map is a plain vocabulary
        # that loads fine without it; a repo that genuinely needs custom code is a
        # repo we are not willing to execute.
        return AutoTokenizer.from_pretrained(
            tokenizer_name,
            revision=revision,
            trust_remote_code=False,
            local_files_only=True,
        )
    except Exception:
        pass  # Not in the local cache — try the network below, bounded.

    if offline:
        logger.warning(
            f"Tokenizer {tokenizer_name} not in local HF cache and network "
            f"loading is disabled (HEADROOM_OFFLINE); using estimation"
        )
        return None

    timeout = _load_timeout_secs()
    if timeout <= 0:
        logger.warning(
            f"Tokenizer {tokenizer_name} not in local HF cache and network "
            f"loading is disabled ({_LOAD_TIMEOUT_ENV}=0); using estimation"
        )
        return None

    result: list[Any] = []
    error: list[BaseException] = []

    def _download() -> None:
        try:
            # Defence in depth behind the is_offline() return above: the switch
            # could flip between that check and this thread, and the guard is
            # what the egress meta-test recognises as dominating the fetch.
            guard_egress(f"HuggingFace tokenizer download ({tokenizer_name})", "huggingface.co")
            result.append(
                AutoTokenizer.from_pretrained(
                    tokenizer_name,
                    revision=revision,
                    trust_remote_code=False,  # see the cache-only attempt above
                )
            )
        except BaseException as e:  # noqa: BLE001 — report any failure to the waiter
            error.append(e)

    thread = threading.Thread(
        target=_download,
        name=f"headroom-hf-tokenizer-load-{tokenizer_name}",
        daemon=True,
    )
    thread.start()
    thread.join(timeout)
    if thread.is_alive():
        logger.warning(
            f"Timed out loading tokenizer {tokenizer_name} after {timeout}s "
            f"(set {_LOAD_TIMEOUT_ENV} to adjust); using estimation"
        )
        return None
    if error:
        logger.warning(f"Failed to load tokenizer {tokenizer_name}: {error[0]}")
        return None
    return result[0] if result else None


def get_tokenizer_name(model: str) -> str:
    """Get HuggingFace tokenizer name for a model.

    Always returns an allowlisted repository. ``model`` is attacker-reachable —
    it is the ``model`` field of a proxied request body — so this resolver must
    never return a caller-controlled string for the loader to look up on the Hub.

    Args:
        model: Model name.

    Returns:
        HuggingFace tokenizer identifier, from ALLOWED_TOKENIZER_REPOS.
    """
    model_lower = model.lower()

    # Direct lookup
    if model_lower in MODEL_TO_TOKENIZER:
        return MODEL_TO_TOKENIZER[model_lower]

    # Try prefix matching, longest (most specific) key first. Scanning in
    # dict-insertion order is wrong: a short family key like "qwen" precedes
    # "qwen2"/"qwen2.5", so "qwen2-7b-instruct" would match "qwen" first and
    # resolve to the Qwen1 tokenizer (a different vocabulary -> wrong counts).
    # The sibling tiktoken resolver (get_encoding_for_model) documents and
    # guards this exact order-dependent pitfall.
    for key in sorted(MODEL_TO_TOKENIZER, key=len, reverse=True):
        if model_lower.startswith(key):
            return MODEL_TO_TOKENIZER[key]

    # A caller may legitimately name a shipped repository outright
    # ("meta-llama/Llama-3.1-8B") instead of using the short alias, and that has
    # to keep counting exactly. Accepting it via the allowlist returns our own
    # canonical spelling, so the caller's string still never reaches the loader.
    allowed = _resolve_allowed_repo(model)
    if allowed is not None:
        return allowed

    # Fail closed. This branch used to be "assume the model name is the
    # tokenizer name", which handed an arbitrary client-supplied string to
    # AutoTokenizer.from_pretrained -> a Hub fetch of whatever repository the
    # caller named. An unrecognised model now resolves to the default vocabulary;
    # the count is an approximation, which is what an unknown model already got
    # when the made-up repo id failed to resolve.
    logger.debug(
        "No tokenizer mapping for model %r; using default tokenizer %s",
        model,
        DEFAULT_TOKENIZER,
    )
    return DEFAULT_TOKENIZER


class HuggingFaceTokenizer(BaseTokenizer):
    """Token counter using HuggingFace tokenizers.

    Supports any model with a HuggingFace tokenizer, including:
    - Llama family (Llama 2, Llama 3, CodeLlama)
    - Mistral family (Mistral, Mixtral)
    - Qwen family
    - DeepSeek family
    - Phi family
    - Falcon, StarCoder, MPT, Gemma, etc.

    Requires the `transformers` library:
        pip install transformers

    Some models may require authentication:
        huggingface-cli login

    Example:
        counter = HuggingFaceTokenizer("llama-3-8b")
        tokens = counter.count_text("Hello, world!")
    """

    # Overhead per message (varies by model, this is a reasonable default)
    MESSAGE_OVERHEAD = 4
    REPLY_OVERHEAD = 3

    def __init__(self, model: str):
        """Initialize HuggingFace tokenizer.

        Args:
            model: Model name (e.g., 'llama-3-8b', 'mistral-7b').
        """
        self.model = model
        self.tokenizer_name = get_tokenizer_name(model)
        self._tokenizer = None  # Lazy load

    @property
    def tokenizer(self):
        """Lazy-load the tokenizer."""
        if self._tokenizer is None:
            loaded = _load_tokenizer(self.tokenizer_name)
            if loaded is not None:
                self._tokenizer = loaded
            else:
                # Mark as unavailable
                self._tokenizer = False
        return self._tokenizer if self._tokenizer is not False else None

    def _use_fallback(self) -> bool:
        """Check if we need to use fallback estimation."""
        return self.tokenizer is None

    def count_text(self, text: str) -> int:
        """Count tokens in text.

        Falls back to estimation if tokenizer unavailable.

        Args:
            text: Text to tokenize.

        Returns:
            Number of tokens.
        """
        if not text:
            return 0
        if self._use_fallback():
            # Fall back to ~4 chars per token estimation
            return max(1, int(len(text) / 4 + 0.5))
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        return len(tokens)

    def count_messages(self, messages: list[dict[str, Any]]) -> int:
        """Count tokens in chat messages.

        Uses the model's chat template if available, otherwise
        falls back to base class implementation.

        Args:
            messages: List of chat messages.

        Returns:
            Total token count.
        """
        if self._use_fallback():
            # Use base class implementation with estimation
            return super().count_messages(messages)

        # Try to use chat template for accurate counting
        templated = self._count_with_chat_template(messages, add_generation_prompt=True)
        if templated is not None:
            return templated

        return super().count_messages(messages)

    def count_message(self, message: dict[str, Any]) -> int:
        """Count tokens in a single message.

        The base implementation subtracts a fixed reply overhead from
        ``count_messages([message])``, but a chat template defines its own
        generation-prompt overhead, so that subtraction can go below zero for a
        short message (GH #4274). With a chat template, render the message alone
        without the generation prompt instead.

        Args:
            message: A message dict with 'role' and 'content'.

        Returns:
            Token count for this message, never negative.
        """
        if not self._use_fallback():
            templated = self._count_with_chat_template([message], add_generation_prompt=False)
            if templated is not None:
                return templated
        # Estimate without the template. ``super().count_message`` would go
        # through ``self.count_messages``, which retries the template with a
        # generation prompt and can still subtract a short message down to 0.
        return max(0, BaseTokenizer.count_messages(self, [message]) - self.REPLY_OVERHEAD)

    def _count_with_chat_template(
        self, messages: list[dict[str, Any]], *, add_generation_prompt: bool
    ) -> int | None:
        """Count ``messages`` through the tokenizer's chat template.

        Returns None when the tokenizer has no usable chat template, so the
        caller can fall back to the per-message estimate.
        """
        if not hasattr(self.tokenizer, "apply_chat_template"):
            return None
        try:
            # ``return_dict=False`` is load-bearing. transformers >= 5 defaults
            # ``apply_chat_template(tokenize=True)`` to ``return_dict=True``,
            # which hands back a BatchEncoding — so ``len(formatted)`` counted
            # DICT KEYS (2: input_ids, attention_mask) instead of tokens.
            # Measured on Qwen2.5-72B, a 6,000-char message: count_messages
            # returned 2 and count_message returned -1 (base subtracts a
            # 3-token reply overhead), against a true 1,003 tokens. That is a
            # ~99.8% undercount on every HF-routed family whose resolved
            # tokenizer carries a chat template — llama, qwen, deepseek, phi,
            # yi, falcon, starcoder. pyproject pins transformers>=5.5.0,<6.0,
            # so the affected version is the only installable one.
            formatted = self.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=add_generation_prompt,
                return_dict=False,
            )
        except Exception:
            return None
        # Recent transformers return a BatchEncoding (dict-like with
        # input_ids/attention_mask) here; len() of it is the key count
        # (2), not the token count.
        if hasattr(formatted, "keys") and "input_ids" in formatted:
            return len(formatted["input_ids"])
        return len(formatted)

    def encode(self, text: str) -> list[int]:
        """Encode text to token IDs.

        Args:
            text: Text to encode.

        Returns:
            List of token IDs.

        Raises:
            NotImplementedError: If tokenizer not available.
        """
        if self._use_fallback():
            raise NotImplementedError(
                f"Encoding not available for {self.model} - "
                f"tokenizer {self.tokenizer_name} could not be loaded"
            )
        return self.tokenizer.encode(text, add_special_tokens=False)

    def decode(self, tokens: list[int]) -> str:
        """Decode token IDs to text.

        Args:
            tokens: List of token IDs.

        Returns:
            Decoded text.

        Raises:
            NotImplementedError: If tokenizer not available.
        """
        if self._use_fallback():
            raise NotImplementedError(
                f"Decoding not available for {self.model} - "
                f"tokenizer {self.tokenizer_name} could not be loaded"
            )
        return self.tokenizer.decode(tokens)

    @classmethod
    def is_available(cls) -> bool:
        """Check if HuggingFace tokenizers are available.

        Returns:
            True if transformers is installed.
        """
        try:
            import transformers  # noqa: F401

            return True
        except ImportError:
            return False

    @classmethod
    def list_supported_models(cls) -> list[str]:
        """List models with known tokenizer mappings.

        Returns:
            List of supported model names.
        """
        return list(MODEL_TO_TOKENIZER.keys())

    def __repr__(self) -> str:
        return f"HuggingFaceTokenizer(model={self.model!r}, tokenizer={self.tokenizer_name!r})"
