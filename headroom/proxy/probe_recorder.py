"""Opt-in JSONL recorder for compression events (probe-based replay evals).

Records (original, compressed) message pairs at ``INPUT_COMPRESSED`` so that
``headroom evals probes`` can measure what compression removed from real
proxied sessions. Activated only when ``HEADROOM_PROBE_RECORD_DIR`` is set;
recordings contain full conversation content in plaintext, are written with
directory mode 0700, and never leave the machine.

Writes happen synchronously on the request path, so this is a diagnostic
tool for bounded recording sessions, not an always-on production setting.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from pathlib import Path

from headroom.pipeline import PipelineEvent, PipelineStage

logger = logging.getLogger(__name__)

RECORD_DIR_ENV = "HEADROOM_PROBE_RECORD_DIR"


class CompressionEventRecorder:
    """Pipeline extension appending one JSONL line per compression event.

    Only ``INPUT_COMPRESSED`` events that carry ``original_messages`` in their
    metadata and actually changed the token count are recorded. The extension
    never mutates the event; ``PipelineExtensionManager.emit`` already swallows
    extension exceptions, so a broken recorder cannot break a request.
    """

    def __init__(self, record_dir: Path) -> None:
        self._dir = record_dir
        self._dir.mkdir(parents=True, exist_ok=True)
        os.chmod(self._dir, 0o700)
        # One file per process so concurrent proxy workers never interleave
        # partial lines.
        self._path = self._dir / f"compression-events-{os.getpid()}.jsonl"
        self._lock = threading.Lock()

    @property
    def path(self) -> Path:
        return self._path

    def on_pipeline_event(self, event: PipelineEvent) -> None:
        if event.stage is not PipelineStage.INPUT_COMPRESSED:
            return None
        metadata = event.metadata or {}
        original = metadata.get("original_messages")
        tokens_before = metadata.get("tokens_before")
        tokens_after = metadata.get("tokens_after")
        if original is None or event.messages is None:
            return None
        if tokens_before is None or tokens_after is None or tokens_before == tokens_after:
            return None
        record = {
            "ts": time.time(),
            "request_id": event.request_id,
            "provider": event.provider,
            "model": event.model,
            "tokens_before": tokens_before,
            "tokens_after": tokens_after,
            "transforms_applied": metadata.get("transforms_applied") or [],
            "original_messages": original,
            "compressed_messages": event.messages,
        }
        line = json.dumps(record, ensure_ascii=False, default=str)
        with self._lock:
            with self._path.open("a", encoding="utf-8") as fh:
                fh.write(line + "\n")
        return None


def probe_recorder_from_env() -> CompressionEventRecorder | None:
    """Build a recorder when ``HEADROOM_PROBE_RECORD_DIR`` is set, else None.

    Fail-open: any error constructing the recorder (unwritable path, etc.)
    disables recording with a warning instead of breaking proxy startup.
    """

    record_dir = os.environ.get(RECORD_DIR_ENV, "").strip()
    if not record_dir:
        return None
    try:
        return CompressionEventRecorder(Path(record_dir).expanduser())
    except Exception as exc:  # noqa: BLE001 - recorder must never break proxy startup
        logger.warning("probe recorder disabled (%s): %s", RECORD_DIR_ENV, exc)
        return None
