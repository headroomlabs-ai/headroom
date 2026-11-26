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

import hashlib
import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Case-sensitivity of the volume holding a TOIN store's parent directory,
# keyed by resolved parent. Measured once per directory per process; only
# measured results are cached, so an unclassifiable directory is retried.
_case_insensitive_volumes: dict[str, bool] = {}


def _case_variant(name: str) -> str | None:
    """``name`` with one ASCII letter's case flipped, or None if it has none.

    Only an ASCII swap is safe: filesystems fold case one-to-one, while
    Python's ``str.upper()`` expands characters (``ß`` -> ``SS``), so a
    missing expanded spelling would look like a case-sensitive volume and
    a case-insensitive store would be disowned on restart.
    """
    for index, char in enumerate(name):
        if "a" <= char <= "z":
            return name[:index] + char.upper() + name[index + 1 :]
        if "A" <= char <= "Z":
            return name[:index] + char.lower() + name[index + 1 :]
    return None


def _volume_is_case_insensitive(parent: Path) -> bool | None:
    """Whether ``parent``'s volume treats case-variant spellings as one file.

    Returns True/False when measured and None when the directory could not
    be classified (the caller then keeps the exact spelling and the next
    call retries). No standard-library call reports this directly
    (``os.path.normcase`` is the identity on macOS, whose default volume is
    case-insensitive), so it is measured in two steps:

    1. Read-only: the names listed in the directory answer by themselves —
       two listed spellings are two entries (no folding), one listed
       spelling whose variant also resolves is one entry folded (insensitive)
       — no write, so a readable-but-unwritable store directory is still
       classified.
    2. Otherwise a probe file is created with ``tempfile.mkstemp``
       (exclusively, unpredictable name) and a case-variant of its name is
       checked. The unpredictable name matters: a predictable one could be
       pre-planted as a symlink by a local writer of the directory and the
       probe's write would follow it.

    Step 1 compares the *names listed in this directory*, never file
    identity. An insensitive volume lists one spelling per entry, so two
    listed spellings are two entries and the volume does not fold case.
    Identity cannot answer this: a local writer of the directory can create
    a hardlink under the variant spelling, a second entry that shares the
    inode, and ``os.path.samefile`` would call a case-sensitive volume
    insensitive — folding two proxies' owner identities into one store.
    """
    key = str(parent)
    if key in _case_insensitive_volumes:
        return _case_insensitive_volumes[key]
    try:
        names: set[str] | None = {child.name for child in parent.iterdir()}
    except OSError:
        # Unreadable directory: no read-only evidence, so the probe decides.
        names = None
    if names:
        for name in names:
            variant_name = _case_variant(name)
            if variant_name is None:
                continue  # spelling has no ASCII case to compare
            if variant_name in names:
                # Both spellings are listed, so they are two directory
                # entries: the volume does not fold case.
                result = False
            else:
                # One spelling listed. It is the same entry on an insensitive
                # volume exactly when the other spelling resolves.
                result = os.path.lexists(parent / variant_name)
            _case_insensitive_volumes[key] = result
            return result
    try:
        parent.mkdir(parents=True, exist_ok=True)
        fd, probe_name = tempfile.mkstemp(dir=parent, prefix=".headroom_case_probe_")
        os.close(fd)
        try:
            probe = parent / probe_name
            variant_name = _case_variant(probe.name)
            if variant_name is None:
                return None  # mkstemp name carries no case to compare
            result = os.path.lexists(probe.with_name(variant_name))
        finally:
            (parent / probe_name).unlink(missing_ok=True)
    except OSError:
        return None
    _case_insensitive_volumes[key] = result
    return result


def _version_paths(path: Path, store_version: str) -> tuple[Path, Path | None]:
    """Return ``(versioned_write_path, base_rollback_path)``.

    ``toin.json`` + ``3.0`` -> write ``toin.v3.json``, rollback ``toin.json``.
    A path that already carries the version (``toin.v3.json``, or the
    extensionless ``toin.v3``) is written in place with no pre-upgrade
    source, so a caller who names the v3 file explicitly never doubles the
    version marker or reads another store's data.
    """
    major = store_version.split(".", 1)[0]
    versioned_marker = f".v{major}"
    versioned_suffix = f"{versioned_marker}{path.suffix}"
    # The version check is case-insensitive on every platform: Windows and
    # the default macOS volume spell one directory entry many ways, and
    # ``os.path.normcase`` is the identity on macOS, so it cannot be relied
    # on there. A POSIX file that merely happens to contain ``.v3`` in
    # another case is unaffected in practice — its owner identity folds the
    # same way, so owner and write target stay consistent per spelling.
    name = path.name.lower()
    if name.endswith(versioned_marker) or name.endswith(versioned_suffix.lower()):
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
        # Owner identity: the path the caller configured, before versioning.
        # Two backends configured at different paths (``toin.json`` and
        # ``toin.v3.json``) can derive to the same versioned write target;
        # stamping the owner lets save() refuse to clobber another proxy's
        # store instead of silently overwriting it. The parent directory is
        # canonicalized but the final filename is kept verbatim: an atomic
        # save replaces the file *at the configured name* (a symlink is
        # replaced, not followed), so the token must identify that name,
        # not the link target a previous store may have lived at. Whether the
        # identity folds case is decided per volume (see
        # ``_volume_is_case_insensitive``): on a case-insensitive volume a
        # case-only respelling of HEADROOM_TOIN_PATH is the same directory
        # entry and must stay one owner, while on a case-sensitive volume the
        # spellings are different files with different write targets and must
        # keep distinct owners. Computed lazily so construction stays
        # side-effect free.
        self._configured_path = self._path
        self._owner: Path | None = None
        self._store_version: str | None = None
        if store_version:
            self.configure_version(store_version)

    def configure_version(self, store_version: str) -> None:
        """Apply TOIN's storage version to this backend without replacing it."""
        if self._store_version == store_version:
            return
        if self._store_version is not None:
            raise ValueError("TOIN backend already configured for a different version")
        self._store_version = store_version
        self._path, self._pre_upgrade_path = _version_paths(self._configured_path, store_version)
        if self._pre_upgrade_path is not None:
            return
        # The configured name already carries the version (``toin.v3.json``).
        # It may still hold a pre-upgrade layout — an operator who pointed
        # HEADROOM_TOIN_PATH at a legacy file named like the new one — so the
        # write target is decided from what is on disk. The decision must not
        # depend on the configured file's presence: an operator who archives
        # the rollback copy after the first migration must land on the same
        # migrated store on restart, not open a fresh store at the configured
        # name.
        sibling = self._path.with_name(self._path.name + f".v{store_version.split('.', 1)[0]}")
        if self._migrated_store_exists(sibling):
            self._pre_upgrade_path = self._path
            self._path = sibling
            return
        # Never overwrite an existing legacy store merely because of its name,
        # and never parse an oversized one at startup. Both redirect writes to
        # the sibling: the configured file is preserved untouched as the
        # rollback view (or an oversized archive), and fresh learning lands in
        # an ownership-checked sibling the next start rediscovers.
        #
        # An oversized store may in fact already be at this layout version —
        # its version cannot be checked without the parse the size limit
        # forbids. Redirecting then forks the history: the oversized original
        # keeps the older records and the sibling receives the new ones. That
        # is still strictly better than the alternative (every write refused,
        # all new learning lost on restart), and it is recoverable: once the
        # load limit is raised, moving the sibling aside lets the next start
        # read the original store again.
        if self._oversized(self._path) or self._probe_version(self._path) in ("1.0", "2.0"):
            self._pre_upgrade_path = self._path
            self._path = sibling

    def _oversized(self, path: Path) -> bool:
        """True when ``path`` exists but exceeds the configured load limit."""
        try:
            return self._max_load_bytes is not None and path.stat().st_size > self._max_load_bytes
        except OSError:
            return False

    def _probe_version(self, path: Path) -> str | None:
        """Read a store's layout version, or None when unreadable or oversized."""
        try:
            if self._oversized(path):
                return None
            with open(path, encoding="utf-8") as f:
                version: Any = json.load(f).get("version", "1.0")
                return version if isinstance(version, str) else None
        except (OSError, ValueError, AttributeError):
            return None

    def _migrated_store_exists(self, path: Path) -> bool:
        """True when ``path`` holds a store this configured path migrated.

        Positive evidence only: the expected layout version and this path's
        owner stamp, both written by every save the network performs. An
        ownerless sibling is a legacy store from a differently configured
        proxy and must be left alone, not adopted and rewritten.
        """
        try:
            if self._oversized(path):
                return False
            with open(path, encoding="utf-8") as f:
                existing = json.load(f)
        except (OSError, ValueError, AttributeError):
            return False
        return (
            isinstance(existing, dict)
            and existing.get("version") == self._store_version
            and existing.get("owner") == self._owner_token()
        )

    def _read_path(self) -> Path:
        """Path to load: this version's store, else the pre-upgrade one."""
        if not self._path.exists() and self._pre_upgrade_path is not None:
            return self._pre_upgrade_path
        return self._path

    @property
    def path(self) -> Path:
        """Active write path."""
        return self._path

    @property
    def pre_upgrade_path(self) -> Path | None:
        """Read-only pre-upgrade store path, or None when the layout is unversioned."""
        return self._pre_upgrade_path

    @property
    def loaded_pre_upgrade(self) -> bool:
        """True when the last load read the pre-upgrade file, not this layout's."""
        return self._loaded_pre_upgrade

    def _owner_token(self) -> str:
        """Stable token identifying the configured path this backend owns.

        Derived from the caller-supplied path (before versioning), so every
        process pointed at the same configured path agrees, and two processes
        pointed at different configured paths disagree — except where the
        volume itself treats two spellings as one directory entry.
        """
        if self._owner is None:
            parent = self._configured_path.parent.resolve()
            identity = str(parent / self._configured_path.name)
            folded = _volume_is_case_insensitive(parent)
            if folded:
                identity = identity.lower()
                self._owner = Path(identity)
            elif folded is False:
                self._owner = Path(identity)
            else:
                # Directory could not be classified (e.g. blocked parent):
                # hash the exact spelling for this call only and leave the
                # identity unmemoized so a later call re-measures — a stamp
                # written under "unknown" must not outlive the measurement.
                return hashlib.sha256(identity.encode("utf-8", "surrogatepass")).hexdigest()
        return hashlib.sha256(str(self._owner).encode("utf-8", "surrogatepass")).hexdigest()

    def _refuse_foreign_store(self) -> None:
        """Refuse to overwrite a store a different configured path owns.

        ``toin.json`` and an explicitly named ``toin.v3.json`` both derive to
        the same versioned write target. Whoever gets there first owns it; the
        second must not silently replace that history with its own. A file with
        no ``owner`` key predates this check and stays writable, so upgrades
        and the pre-upgrade rewrite path are unaffected.
        """
        if not self._path.exists():
            return
        try:
            with open(self._path) as f:
                existing = json.load(f)
        except (json.JSONDecodeError, UnicodeDecodeError, OSError):
            return
        if not isinstance(existing, dict):
            return
        owner = existing.get("owner")
        if isinstance(owner, str) and owner and owner != self._owner_token():
            raise ValueError(
                f"TOIN store {self._path} is owned by a differently configured "
                "path; refusing to overwrite another instance's history"
            )

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
                if read_path == self._path and isinstance(data, dict):
                    owner = data.get("owner")
                    if isinstance(owner, str) and owner and owner != self._owner_token():
                        raise ValueError(
                            f"TOIN store {read_path} belongs to another configured path"
                        )
                return data
        except (json.JSONDecodeError, UnicodeDecodeError, OSError) as e:
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
            self._refuse_foreign_store()
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
