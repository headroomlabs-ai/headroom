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

import logging

import httpx

from .kompress_compressor import KompressConfig, KompressResult, store_kompress_in_ccr

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
_CCR_RATIO_GATE = 0.8


class RemoteKompressCompressor:
    """Drop-in for KompressCompressor that POSTs to a hosted ``/compress`` endpoint.

    Fails OPEN: any network/HTTP error returns the content verbatim so a flaky
    endpoint degrades compression rather than breaking the proxy.
    """

    name = "kompress_compressor"

    def __init__(
        self,
        endpoint: str,
        token: str | None = None,
        config: KompressConfig | None = None,
        timeout: float = 20.0,
        path: str | None = DEFAULT_ENDPOINT_PATH,
        headers: dict[str, str] | None = None,
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
        # httpx.Client is safe to share across the proxy's worker threads.
        self._client = httpx.Client(timeout=timeout)

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
    ) -> KompressResult:
        """Compress via the remote endpoint.

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

        try:
            resp = self._client.post(
                self._url,
                headers=self._headers,
                json={"content": content, "target_ratio": target_ratio},
            )
            resp.raise_for_status()
            data = resp.json()
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
        if self.config.enable_ccr and result.compression_ratio < _CCR_RATIO_GATE:
            # Store the PRE-protection text when the caller supplied it. ``content``
            # may be the tag-protected placeholder intermediate, and storing that
            # makes a later full retrieval hand back {{HEADROOM_TAG_N}} instead of
            # the real block — the exact loss ``ccr_original`` exists to prevent.
            # Same resolution order as KompressCompressor.compress.
            ccr_source = ccr_original if ccr_original is not None else content
            # The endpoint's ``original_tokens`` describes ``content``, so it does
            # not describe a different ``ccr_source``; count that one locally.
            # Unchanged on the common path where no override was passed.
            ccr_source_tokens = (
                len(ccr_source.split()) if ccr_original is not None else result.original_tokens
            )
            cache_key = store_kompress_in_ccr(ccr_source, compressed, ccr_source_tokens)
            if cache_key:
                result.cache_key = cache_key
                # Report the source line span so a reader can tell content was
                # compressed away rather than absent (#2586).
                source_lines = ccr_source.count("\n") + 1
                line_word = "line" if source_lines == 1 else "lines"
                result.compressed += (
                    f"\n[{result.original_tokens} words compressed to "
                    f"{result.compressed_tokens} (from {source_lines} source {line_word})."
                    f" Original content preserved."
                    f" Retrieve more: hash={cache_key}]"
                )

        return result

    def close(self) -> None:
        self._client.close()
