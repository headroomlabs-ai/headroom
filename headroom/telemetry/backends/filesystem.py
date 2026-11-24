"""Filesystem storage backend for TOIN.

Stores TOIN patterns as a JSON file with atomic writes.
This is the default backend, matching the original toin.py behavior.

When ``store_version`` is given, the store is written to a
version-suffixed sibling (``toin.json`` -> ``toin.v3.json``) and the
unversioned path is only read, as the pre-upgrade layout. A released
reader that does not know the new version therefore never opens the new
file: rolling back loads the last pre-upgrade store instead of
reinterpreting keys under its old tenant resolver. A path that already
carries the version is written in place, so ``toin.json`` and
``toin.v3.json`` stay independent stores.
"""

from __future__ import annotations

import json
import logging
import tempfile
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def _version_paths(path: Path, store_version: str) -> tuple[Path, Path | None]:
    """Return ``(versioned_write_path, base_rollback_path)``.

    ``toin.json`` + ``3.0`` -> write ``toin.v3.json``, rollback ``toin.json``.
    A path that already carries the version (``toin.v3.json``) is written in
    place with no pre-upgrade source, so a caller who names the v3 file
    explicitly never doubles the suffix or reads another store's data.
    """
    major = store_version.split(".", 1)[0]
    versioned_suffix = f".v{major}{path.suffix}"
    if path.name.endswith(versioned_suffix):
        # The caller already named the versioned file (e.g. ``toin.v3.json``).
        # Write it in place and do not treat the unversioned sibling as a
        # pre-upgrade source: ``toin.json`` and ``toin.v3.json`` must stay
        # independent stores, otherwise opening ``toin.v3.json`` would import
        # the legacy ``toin.json`` store across namespaces.
        return path, None
    return path.with_name(f"{path.stem}{versioned_suffix}"), path


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
        store_version: On-disk layout version. When set on an unversioned
            path, writes go to a version-suffixed sibling and ``path`` is read
            only as the pre-upgrade store. When the path already carries that
            version (``toin.v3.json``), it is written in place with no
            pre-upgrade source, keeping the two files independent. None keeps
            the unversioned single-file layout.
    """

    def __init__(
        self,
        path: str,
        *,
        max_load_bytes: int | None = None,
        store_version: str | None = None,
    ) -> None:
        self._path = Path(path)
        self._max_load_bytes = max_load_bytes
        self._pre_upgrade_path: Path | None = None
        self._loaded_pre_upgrade = False
        if store_version:
            self._path, self._pre_upgrade_path = _version_paths(self._path, store_version)

    def _read_path(self) -> Path:
        """Path to load: this version's store, else the pre-upgrade one."""
        if not self._path.exists() and self._pre_upgrade_path is not None:
            return self._pre_upgrade_path
        return self._path

    @property
    def pre_upgrade_path(self) -> Path | None:
        """Read-only pre-upgrade store path, or None when the layout is unversioned."""
        return self._pre_upgrade_path

    @property
    def loaded_pre_upgrade(self) -> bool:
        """True when the last load read the pre-upgrade file, not this layout's."""
        return self._loaded_pre_upgrade

    def load(self) -> dict[str, Any]:
        """Load TOIN data from the JSON file.

        Returns:
            Parsed JSON data, or empty dict if file doesn't exist or is corrupt.
        """
        read_path = self._read_path()
        self._loaded_pre_upgrade = (
            self._pre_upgrade_path is not None and read_path == self._pre_upgrade_path
        )
        if not read_path.exists():
            return {}

        try:
            if self._max_load_bytes is not None and read_path.stat().st_size > self._max_load_bytes:
                logger.warning(
                    "TOIN data file %s exceeds the configured load limit (%d bytes); "
                    "leaving it unchanged",
                    read_path,
                    self._max_load_bytes,
                )
                return {}
            with open(read_path) as f:
                data: dict[str, Any] = json.load(f)
                return data
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("Failed to load TOIN data from %s: %s", read_path, e)
            return {}

    def save(self, data: dict[str, Any]) -> None:
        """Save TOIN data to the JSON file with atomic write.

        Uses a temporary file and rename to ensure atomicity.
        If the write fails, the original file is preserved.

        Args:
            data: Serialized TOIN data to persist.

        Raises:
            ValueError: The existing file or serialized data exceeds the size limit.
            OSError: If persistence fails, so the caller can retain pending changes.
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
            raise
