"""File watcher for live code graph reindexing.

Monitors the project directory for source file changes and triggers
incremental reindexing via codebase-memory-mcp. Runs as a background
thread in the proxy process.

Cross-platform: uses watchdog (FSEvents on macOS, inotify on Linux,
ReadDirectoryChangesW on Windows).

Usage:
    watcher = CodeGraphWatcher(project_dir="/path/to/project")
    watcher.start()   # Non-blocking, runs in background thread
    ...
    watcher.stop()    # Clean shutdown
"""

from __future__ import annotations

import json
import logging
import subprocess
import threading
import time
from pathlib import Path

from headroom._subprocess import run

logger = logging.getLogger(__name__)

# Source file extensions worth reindexing for
_SOURCE_EXTENSIONS = frozenset(
    {
        ".py",
        ".js",
        ".jsx",
        ".ts",
        ".tsx",
        ".go",
        ".rs",
        ".java",
        ".c",
        ".cpp",
        ".cc",
        ".h",
        ".hpp",
        ".cs",
        ".kt",
        ".scala",
        ".rb",
        ".php",
        ".swift",
        ".lua",
        ".zig",
        ".ex",
        ".exs",
        ".m",
        ".mm",
        ".jl",
        ".r",
        ".R",
        ".sql",
        # Config/docs that affect graph
        ".md",
        ".rst",
        ".toml",
        ".yaml",
        ".yml",
        ".json",
    }
)

# Directories to ignore
_IGNORE_DIRS = frozenset(
    {
        ".git",
        ".hg",
        ".svn",
        "node_modules",
        "__pycache__",
        ".venv",
        "venv",
        ".tox",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        "dist",
        "build",
        ".eggs",
        "*.egg-info",
        ".headroom",
        "target",  # Rust/Java
    }
)


class CodeGraphWatcher:
    """Watches project files and triggers incremental graph reindex.

    Uses a debounce strategy: after a file change, waits for a quiet
    period (no more changes) before triggering reindex. This handles
    rapid multi-file edits efficiently.

    Args:
        project_dir: Root directory to watch.
        debounce_seconds: Wait this long after last change before reindexing.
        cbm_binary: Path to codebase-memory-mcp binary. Auto-detected if None.
    """

    def __init__(
        self,
        project_dir: str | Path,
        debounce_seconds: float = 2.0,
        cbm_binary: str | None = None,
    ) -> None:
        self.project_dir = str(project_dir)
        self.debounce_seconds = debounce_seconds
        self.cbm_binary: str | None = None
        if cbm_binary:
            self.cbm_binary = cbm_binary
        else:
            from headroom.graph.installer import get_cbm_path

            path = get_cbm_path()
            self.cbm_binary = str(path) if path else None
        self._observer: object | None = None
        self._debounce_timer: threading.Timer | None = None
        self._lock = threading.Lock()
        self._running = False
        self._reindex_count = 0
        self._last_reindex: float = 0

    def start(self) -> bool:
        """Start watching in a background thread. Returns True if started."""
        if not self.cbm_binary:
            logger.debug("Code graph watcher: codebase-memory-mcp not found, not starting")
            return False

        try:
            from watchdog.events import FileSystemEventHandler
            from watchdog.observers import Observer
        except ImportError:
            logger.debug("Code graph watcher: watchdog not installed, not starting")
            return False

        class _Handler(FileSystemEventHandler):
            def __init__(self, watcher: CodeGraphWatcher):
                self._watcher = watcher

            def on_any_event(self, event: object) -> None:
                # watchdog event has src_path attribute
                src_path = getattr(event, "src_path", "")
                if not src_path:
                    return

                path = Path(src_path)

                # Skip ignored directories
                for part in path.parts:
                    if part in _IGNORE_DIRS:
                        return

                # Only react to source file changes
                if path.suffix.lower() not in _SOURCE_EXTENSIONS:
                    return

                # Skip temporary/swap files
                if path.name.startswith(".") or path.name.endswith("~"):
                    return

                self._watcher._schedule_reindex()

        observer = Observer()
        observer.schedule(_Handler(self), self.project_dir, recursive=True)
        observer.daemon = True
        observer.start()
        self._observer = observer
        self._running = True

        logger.info(
            "Code graph watcher: monitoring %s (debounce=%.1fs)",
            self.project_dir,
            self.debounce_seconds,
        )
        return True

    def stop(self) -> None:
        """Stop watching and clean up."""
        self._running = False
        with self._lock:
            if self._debounce_timer:
                self._debounce_timer.cancel()
                self._debounce_timer = None

        if self._observer:
            observer = self._observer
            self._observer = None
            # watchdog Observer has stop() and join()
            if hasattr(observer, "stop"):
                observer.stop()  # type: ignore[union-attr]
            if hasattr(observer, "join"):
                observer.join(timeout=3)  # type: ignore[union-attr]

        if self._reindex_count > 0:
            logger.info("Code graph watcher: stopped after %d reindexes", self._reindex_count)

    def _schedule_reindex(self) -> None:
        """Schedule a debounced reindex. Resets timer on each call."""
        with self._lock:
            if self._debounce_timer:
                self._debounce_timer.cancel()
            self._debounce_timer = threading.Timer(self.debounce_seconds, self._do_reindex)
            self._debounce_timer.daemon = True
            self._debounce_timer.start()

    def _do_reindex(self) -> None:
        """Trigger incremental reindex via codebase-memory-mcp."""
        if not self._running or not self.cbm_binary:
            return

        try:
            start = time.monotonic()
            result = run(
                [
                    self.cbm_binary,
                    "cli",
                    "index_repository",
                    json.dumps({"repo_path": self.project_dir, "mode": "fast"}),
                ],
                capture_output=True,
                text=True,
                timeout=30,
            )

            elapsed = time.monotonic() - start
            self._reindex_count += 1
            self._last_reindex = time.time()

            if result.returncode == 0:
                # Check if incremental (look for "changed=N" in output)
                changed = "?"
                for line in result.stderr.splitlines():
                    if "changed=" in line:
                        import re

                        m = re.search(r"changed=(\d+)", line)
                        if m:
                            changed = m.group(1)
                            break

                logger.info(
                    "Code graph: reindexed (%s files changed, %.1fs)",
                    changed,
                    elapsed,
                )
            else:
                logger.debug(
                    "Code graph: reindex failed (exit=%d, %.1fs)",
                    result.returncode,
                    elapsed,
                )

        except subprocess.TimeoutExpired:
            logger.warning("Code graph: reindex timed out after 30s")
        except Exception as e:
            logger.debug("Code graph: reindex error: %s", e)

    @property
    def stats(self) -> dict:
        """Return watcher statistics."""
        return {
            "running": self._running,
            "project_dir": self.project_dir,
            "reindex_count": self._reindex_count,
            "last_reindex": self._last_reindex,
            "debounce_seconds": self.debounce_seconds,
        }
