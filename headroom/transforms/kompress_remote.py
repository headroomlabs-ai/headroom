"""Remote Kompress: offload ML compression to a hosted ``/compress`` endpoint.

Lets a sandboxed proxy — installed WITHOUT the ``[ml]`` extra (no torch/onnx) —
still run Kompress by calling a remote endpoint over HTTP. The class mirrors
:class:`~headroom.transforms.kompress_compressor.KompressCompressor`'s public
surface (``is_ready`` / ``preload`` / ``ensure_background_load`` / ``compress``),
so it is a drop-in at the ContentRouter seam.

Only the model inference is remote. The CCR store + retrieval marker stay
proxy-local (the endpoint is stateless, ``enable_ccr=False``), so
``headroom_retrieve`` keeps working and original content never persists off-box.

Enabled by ``HEADROOM_KOMPRESS_ENDPOINT`` — see ``ContentRouter._get_kompress``.

# Bring-your-own deployment

The endpoint does not have to be Headroom Labs'. An org can pull the Kompress
weights from HuggingFace, serve them on its own stack (vLLM, TorchServe,
SageMaker, KServe, a bare FastAPI box) and point Headroom at it. Nothing about
this class is Modal-specific, and no credential is required — auth is whatever
the operator's own infrastructure expects, including none at all:

    HEADROOM_KOMPRESS_ENDPOINT          https://ml.internal.acme.com
    HEADROOM_KOMPRESS_ENDPOINT_PATH     /compress   (default; set empty to use
                                        the endpoint URL verbatim)
    HEADROOM_KOMPRESS_ENDPOINT_TOKEN    optional; sent as `Authorization: Bearer`
    HEADROOM_KOMPRESS_ENDPOINT_HEADERS  optional; `k=v,k2=v2`, applied last so it
                                        can replace the Authorization header for
                                        stacks that want `x-api-key` or similar

Both new knobs default to today's behaviour: with only
``HEADROOM_KOMPRESS_ENDPOINT`` set, the request is byte-identical to before —
``POST <endpoint>/compress`` with an optional Bearer token. Existing Modal
deployments need no change.

# The HTTP contract

Deliberately small, so a shim in front of an existing inference server is a few
lines. ``POST <endpoint><path>``:

    request   {"content": "<text>", "target_ratio": 0.5 | null}
    response  {"compressed": "<text>",          # REQUIRED, must be a string
               "original_tokens": int,          # optional, defaults to word count
               "compressed_tokens": int,        # optional, defaults to word count
               "compression_ratio": float,      # optional, defaults to 1.0
               "model_used": str}               # optional

``compressed`` is the only required field; every other value is derived if
absent. Any non-2xx, timeout, malformed field, or missing ``compressed`` makes
this pass the content through verbatim — a broken endpoint costs compression,
never correctness.
"""

from __future__ import annotations

import concurrent.futures
import json
import logging
import threading
import time
from typing import Any

import httpx

from ..offline import guard_egress
from .kompress_compressor import (
    KompressConfig,
    KompressResult,
    ccr_retrieval_marker,
    payload_tokens,
    store_kompress_in_ccr,
)

logger = logging.getLogger(__name__)

# Appended to the configured endpoint unless overridden. An operator whose stack
# already serves a full path (``/v1/models/kompress:predict``) sets
# HEADROOM_KOMPRESS_ENDPOINT_PATH="" and gives the complete URL instead.
DEFAULT_ENDPOINT_PATH = "/compress"


def parse_endpoint_headers(raw: str | None) -> dict[str, str]:
    """Parse ``k=v,k2=v2`` into a header dict.

    Same format as ``HEADROOM_OTEL_METRICS_HEADERS`` so operators meet one
    convention. Reimplemented rather than imported from
    :mod:`headroom.observability.metrics`, which pulls in opentelemetry at module
    scope — remote Kompress exists precisely so a proxy can run without heavy
    optional deps, so it must not drag one in through a parsing helper.
    """
    pairs: dict[str, str] = {}
    for item in (raw or "").split(","):
        part = item.strip()
        if not part or "=" not in part:
            continue
        key, _, value = part.partition("=")
        key, value = key.strip(), value.strip()
        if key and value:
            pairs[key] = value
    return pairs


# Below this word count local Kompress passes through verbatim (KompressCompressor
# .compress); mirror it so we never pay a round-trip on a trivially small block.
_MIN_WORDS = 10

# Accept-any-shrink CCR gate, identical to KompressCompressor.compress: only
# store + mark when the shrink is worth the retrieval marker's own cost.

# A remote call is declined outright when the caller's request-scoped ML budget
# has this little left: no HTTP round trip (connect + write + server inference +
# read) fits in it, so starting one only burns the budget and leaves a worker
# behind. The block passes through unchanged instead.
_MIN_REMOTE_BUDGET_SECONDS = 0.05

# Workers for budget-capped calls, and the number of admission permits. A call
# whose caller times out is abandoned, not killed (a sync httpx request cannot be
# interrupted from another thread), so the worker keeps its permit until the
# request itself ends -- and every request ends at an absolute deadline (see
# ``_capped_request``). With permits == workers nothing ever waits in the
# executor's (unbounded) queue: a saturated caller waits for a permit, not with
# its payload enqueued, and fails open when its own deadline passes.
_CAPPED_CALL_WORKERS = 16


class RemoteKompressCompressor:
    """Drop-in for KompressCompressor that POSTs to a hosted ``/compress`` endpoint.

    Fails OPEN: any network/HTTP error returns the content verbatim so a flaky
    endpoint degrades compression rather than breaking the proxy.
    """

    name = "kompress_compressor"

    # ``compress()`` accepts ``_time_budget_cap_seconds``: what the router's
    # request-scoped ML budget has left. Unlike local Kompress this class does
    # not take ``_deadline_started_at``, so it does not set
    # ``shares_request_deadline``; the router checks this flag separately.
    accepts_time_budget_cap: bool = True

    def __init__(
        self,
        endpoint: str,
        token: str | None = None,
        config: KompressConfig | None = None,
        timeout: float = 20.0,
        path: str | None = DEFAULT_ENDPOINT_PATH,
        headers: dict[str, str] | None = None,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.config = config or KompressConfig()
        # Default keeps the pre-existing behaviour exactly: <endpoint>/compress.
        # An empty (or None) path means the caller has supplied a complete URL —
        # needed because real inference servers do not serve at /compress
        # (TorchServe /predictions/<model>, KServe /v1/models/<name>:predict,
        # SageMaker /invocations), and appending to those yields a 404.
        if path:
            suffix = path if path.startswith("/") else "/" + path
            self._url = endpoint.rstrip("/") + suffix
        else:
            self._url = endpoint
        self._headers = {"content-type": "application/json"}
        if token:
            self._headers["authorization"] = f"Bearer {token}"
        # Applied last on purpose: lets an operator replace `authorization` with
        # whatever their gateway wants (x-api-key, a signed header, a tenant id)
        # without needing a separate auth-scheme setting.
        if headers:
            self._headers.update(headers)
        # Air-gap chokepoint. HEADROOM_KOMPRESS_ENDPOINT points at a box that is
        # by definition off this host, so under HEADROOM_OFFLINE this class must
        # not exist at all. Guarding at construction (not at the POST) means the
        # connection pool is never created and the operator hears about the
        # contradictory configuration the first time the router reaches for
        # Kompress, rather than discovering it in a packet capture.
        guard_egress("remote Kompress inference", self._url)
        # httpx.Client is safe to share across the proxy's worker threads.
        # ``transport`` exists for tests (httpx.MockTransport); None is the
        # normal network transport.
        self._timeout = timeout
        self._client = httpx.Client(timeout=timeout, transport=transport)
        self._capped_pool: concurrent.futures.ThreadPoolExecutor | None = None
        self._capped_pool_lock = threading.Lock()
        self._capped_slots = threading.BoundedSemaphore(_CAPPED_CALL_WORKERS)

    @property
    def url(self) -> str:
        """The resolved POST target.

        Logged when the router builds this, so a wrong ``_PATH`` shows up as a
        visibly odd URL at startup rather than as silent pass-through later —
        the fail-open contract means a 404 never surfaces as an error.
        """
        return self._url

    # Nothing to load locally; short-circuit the router straight to compress().
    def is_ready(self) -> bool:
        return True

    def ready_backend(self) -> str | None:
        return "remote"

    def preload(self, *, allow_download: bool = True) -> str:
        return "remote"

    def ensure_background_load(self) -> None:
        return None

    def _passthrough(self, content: str, n_words: int) -> KompressResult:
        return KompressResult(
            compressed=content,
            original=content,
            original_tokens=n_words,
            compressed_tokens=n_words,
            compression_ratio=1.0,
            model_used=self.config.model_id,
        )

    def compress(
        self,
        content: str,
        context: str = "",
        content_type: str | None = None,
        question: str | None = None,
        target_ratio: float | None = None,
        *,
        allow_download: bool = True,
        ccr_original: str | None = None,
        _time_budget_cap_seconds: float | None = None,
    ) -> KompressResult:
        """Compress via the remote endpoint.

        ``_time_budget_cap_seconds`` is what is left of the caller's
        request-scoped ML budget (ContentRouter passes it). It bounds the HTTP
        request AS A WHOLE -- connect, write, server time and read together,
        not each phase -- and a call that runs out passes the block through
        unchanged. When the remainder is at or below
        :data:`_MIN_REMOTE_BUDGET_SECONDS` no request is made at all. ``None``
        (direct callers) keeps the configured per-phase timeout, 20s by default.

        ``ccr_original`` mirrors :meth:`KompressCompressor.compress`: text to
        store in CCR instead of ``content``, used when ``content`` is a
        tag-protected placeholder intermediate ({{HEADROOM_TAG_N}}). It is
        accepted here because this class promises to be a DROP-IN for the local
        compressor at the ContentRouter seam (see the module docstring) — and it
        was not. ContentRouter passes the kwarg whenever custom tags are
        protected, so on any deployment with HEADROOM_KOMPRESS_ENDPOINT set,
        every such request raised

            TypeError: RemoteKompressCompressor.compress() got an unexpected
            keyword argument 'ccr_original'

        which ContentRouter caught with a broad ``except Exception`` and logged
        as ``Kompress failed: ...``. Compression silently degraded to zero on the
        whole deployment while the proxy kept reporting success.
        """
        n_words = len(content.split())
        # Same floor contract as the in-process compressor: lossy
        # word-dropping below config.min_input_words is a net loss (the
        # retrieval marker alone is ~20 words) and garbles short
        # instruction-like blocks. _MIN_WORDS stays the hard clamp.
        if n_words < max(_MIN_WORDS, self.config.min_input_words):
            return self._passthrough(content, n_words)

        # Re-check outside the fail-open try. __init__ already refused if the
        # flag was set then, but the env can be flipped after this object was
        # cached on the ContentRouter (the router builds it once per instance
        # and reuses it for the process lifetime), and the security property we
        # owe is "no socket", not "no construction". It is outside the try on
        # purpose: the except below turns every exception into a silent
        # pass-through, and a policy refusal that degrades silently is exactly
        # the hole this guard closes.
        guard_egress("remote Kompress inference", self._url)

        payload = {"content": content, "target_ratio": target_ratio}
        if _time_budget_cap_seconds is not None and (
            _time_budget_cap_seconds <= _MIN_REMOTE_BUDGET_SECONDS
        ):
            logger.info(
                "Remote Kompress declined: %.3fs of request ML budget left (floor %.3fs); "
                "passing through unchanged",
                max(0.0, _time_budget_cap_seconds),
                _MIN_REMOTE_BUDGET_SECONDS,
            )
            return self._passthrough(content, n_words)

        try:
            if _time_budget_cap_seconds is None:
                data = self._post(payload)
            else:
                data = self._post_within(payload, _time_budget_cap_seconds)
            compressed = data["compressed"]
            if not isinstance(compressed, str):
                raise TypeError("remote Kompress response field 'compressed' must be a string")
            # Coerce the numeric/string metadata fields inside the fail-open guard.
            # A 200 response with a malformed field (e.g. a non-numeric string, or
            # an explicit JSON null: data.get returns None for a present key, and
            # float(None)/int(None) raise) would otherwise escape uncaught and break
            # the proxy request, defeating the fail-open contract this class promises.
            result = KompressResult(
                compressed=compressed,
                original=content,
                original_tokens=int(data.get("original_tokens", n_words)),
                compressed_tokens=int(data.get("compressed_tokens", len(compressed.split()))),
                compression_ratio=float(data.get("compression_ratio", 1.0)),
                model_used=str(data.get("model_used", self.config.model_id)),
            )
        except Exception as e:  # fail OPEN — never break the proxy on a bad endpoint
            logger.warning("Remote Kompress failed (%s); passing through", e)
            return self._passthrough(content, n_words)

        # CCR stays PROXY-LOCAL: endpoint is stateless (enable_ccr=False), so we
        # store the mapping + append the retrieval marker here — same policy and
        # marker format as KompressCompressor.compress.
        if self.config.enable_ccr and compressed != content:
            # Same gate as KompressCompressor: the marked payload must save
            # tokens, and a result that cannot pay for the marker passes through.
            # Store the PRE-protection text when the caller supplied it. ``content``
            # may be the tag-protected placeholder intermediate, and storing that
            # makes a later full retrieval hand back {{HEADROOM_TAG_N}} instead of
            # the real block — the exact loss ``ccr_original`` exists to prevent.
            # Same resolution order as KompressCompressor.compress.
            ccr_source = ccr_original if ccr_original is not None else content
            cache_key = store_kompress_in_ccr(ccr_source, compressed)
            if cache_key:
                # Report the source line span so a reader can tell content was
                # compressed away rather than absent (#2586).
                marked = compressed + ccr_retrieval_marker(
                    result.original_tokens, result.compressed_tokens, ccr_source, cache_key
                )
                # Whole original against whole marked candidate, one unit,
                # and the accounting reports that measurement.
                original_tokens = payload_tokens(content)
                compressed_tokens = payload_tokens(marked)
                if compressed_tokens >= original_tokens:
                    return self._passthrough(content, n_words)
                result.cache_key = cache_key
                result.compressed = marked
                result.original_tokens = original_tokens
                result.compressed_tokens = compressed_tokens
                result.compression_ratio = (
                    compressed_tokens / original_tokens if original_tokens else 1.0
                )

        return result

    def _post(self, payload: dict[str, Any], timeout: float | None = None) -> Any:
        # No ``timeout`` kwarg on the default path: the client's own (20s)
        # applies, and the call is exactly what it was before the cap existed.
        extra: dict[str, Any] = {} if timeout is None else {"timeout": timeout}
        resp = self._client.post(self._url, headers=self._headers, json=payload, **extra)
        resp.raise_for_status()
        return resp.json()

    def _post_within(self, payload: dict[str, Any], cap: float) -> Any:
        """POST and read the response within *cap* seconds in total.

        httpx timeouts are per phase (connect, write, pool, and each read
        between chunks), so ``timeout=cap`` alone neither bounds the call nor
        stops a slow-drip body. Three things together do:

        * **Admission.** A permit (one per worker) is taken before submitting,
          waiting at most until the caller's deadline. Saturated means fail
          open, with the payload never enqueued -- ``future.cancel()`` does not
          free a queued payload, so an unbounded queue would retain every
          timed-out prompt.
        * **Caller deadline.** The caller waits for the worker with one overall
          deadline and passes the block through unchanged when it runs out.
        * **Worker deadline.** The worker keeps its permit until it actually
          finishes (released in its own ``finally``, never on caller timeout),
          and it finishes by an absolute deadline: per-phase timeouts are capped
          to what is left, and the body is streamed and cut off at the
          deadline, so a dripping response cannot hold a permit indefinitely.
        """
        deadline = time.monotonic() + cap
        if not self._capped_slots.acquire(timeout=max(0.0, cap)):
            raise TimeoutError(
                f"all {_CAPPED_CALL_WORKERS} remote Kompress workers busy for the "
                f"request ML budget ({cap:.3f}s)"
            )
        try:
            future = self._get_capped_pool().submit(self._capped_request, payload, deadline)
        except BaseException:
            self._capped_slots.release()  # never reached the worker that would
            raise
        try:
            return future.result(timeout=max(0.0, deadline - time.monotonic()))
        except concurrent.futures.TimeoutError:
            raise TimeoutError(
                f"exceeded the request ML budget ({cap:.3f}s) before the endpoint answered"
            ) from None

    def _capped_request(self, payload: dict[str, Any], deadline: float) -> Any:
        """The worker side of :meth:`_post_within`. Owns one admission permit.

        Lifetime is bounded by *deadline* plus at most one per-phase timeout
        (itself no longer than what was left at start): every httpx phase is
        capped to the time remaining, and the body is read chunk by chunk with
        the absolute deadline checked after each one.
        """
        try:
            left = deadline - time.monotonic()
            if left <= 0.0:
                raise TimeoutError("request ML budget spent before the request started")
            with self._client.stream(
                "POST",
                self._url,
                headers=self._headers,
                json=payload,
                timeout=min(left, self._timeout),
            ) as resp:
                resp.raise_for_status()
                body = bytearray()
                for chunk in resp.iter_bytes():
                    body += chunk
                    if time.monotonic() >= deadline:
                        raise TimeoutError(
                            "remote Kompress response outlived the request ML budget"
                        )
            return json.loads(bytes(body))
        finally:
            self._capped_slots.release()

    def _get_capped_pool(self) -> concurrent.futures.ThreadPoolExecutor:
        with self._capped_pool_lock:
            if self._capped_pool is None:
                self._capped_pool = concurrent.futures.ThreadPoolExecutor(
                    max_workers=_CAPPED_CALL_WORKERS,
                    thread_name_prefix="remote-kompress",
                )
            return self._capped_pool

    def close(self) -> None:
        with self._capped_pool_lock:
            pool, self._capped_pool = self._capped_pool, None
        if pool is not None:
            pool.shutdown(wait=False, cancel_futures=True)
        self._client.close()
