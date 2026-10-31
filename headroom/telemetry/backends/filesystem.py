"""Filesystem storage backend for TOIN.

Stores TOIN patterns as a JSON file with atomic writes.
This is the default backend, matching the original toin.py behavior.
"""

from __future__ import annotations

import json
import logging
import tempfile
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class FileSystemTOINBackend:
    """Filesystem-backed TOIN storage using atomic JSON writes.

    Characteristics:
    - Persists to a single JSON file
    - Atomic writes via temp file + rename (POSIX)
    - Creates parent directories on first save
    - Returns empty dict if file doesn't exist or is corrupted

    Args:
        path: Path to the JSON storage file.
        max_load_bytes: Maximum file size in bytes for loading and saving.
            None disables the limit.
    """

    def __init__(self, path: str, *, max_load_bytes: int | None = None) -> None:
        self._path = Path(path)
        self._max_load_bytes = max_load_bytes

    def load(self) -> dict[str, Any]:
        """Load TOIN data from the JSON file.

        Returns:
            Parsed JSON data, or empty dict if file doesn't exist or is corrupt.
        """
        if not self._path.exists():
            return {}

        try:
            if (
                self._max_load_bytes is not None
                and self._path.stat().st_size > self._max_load_bytes
            ):
                logger.warning(
                    "TOIN data file %s exceeds the configured load limit (%d bytes); "
                    "leaving it unchanged",
                    self._path,
                    self._max_load_bytes,
                )
                return {}
            with open(self._path) as f:
                data: dict[str, Any] = json.load(f)
                return data
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("Failed to load TOIN data from %s: %s", self._path, e)
            return {}

    def save(self, data: dict[str, Any]) -> None:
        """Save TOIN data to the JSON file with atomic write.

        Uses a temporary file and rename to ensure atomicity.
        If the write fails, the original file is preserved.

        Args:
            data: Serialized TOIN data to persist.

        Raises:
            ValueError: The existing file or serialized data exceeds the size limit.
        """
        try:
            # A file skipped by load() must not be replaced with partial state.
            if (
                self._max_load_bytes is not None
                and self._path.exists()
                and self._path.stat().st_size > self._max_load_bytes
            ):
                raise ValueError(
                    f"Existing TOIN data file {self._path} exceeds the configured "
                    f"size limit ({self._max_load_bytes} bytes); refusing to overwrite it"
                )
            self._path.parent.mkdir(parents=True, exist_ok=True)

            fd, tmp_path = tempfile.mkstemp(dir=self._path.parent, prefix=".toin_", suffix=".tmp")
            try:
                # Stream compact JSON directly to the temp file. Building a
                # second multi-gigabyte string was the primary autosave RSS
                # spike reported in #2886.
                with open(fd, "w", newline="\n") as f:
                    json.dump(data, f, separators=(",", ":"))
                # Check actual bytes after flushing, before replacing a valid store.
                if (
                    self._max_load_bytes is not None
                    and Path(tmp_path).stat().st_size > self._max_load_bytes
                ):
                    raise ValueError(
                        f"Serialized TOIN data exceeds the configured size limit "
                        f"({self._max_load_bytes} bytes)"
                    )
                Path(tmp_path).replace(self._path)
            except Exception:
                try:
                    Path(tmp_path).unlink()
                except OSError:
                    pass
                raise

        except OSError as e:
            logger.warning("Failed to save TOIN data to %s: %s", self._path, e)
