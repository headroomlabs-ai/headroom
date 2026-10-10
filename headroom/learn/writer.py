"""Context writers — write learned patterns to agent-specific context files.

Writers take Recommendations and write them to the appropriate context
injection mechanism for each agent system (CLAUDE.md, .cursorrules, etc.).
"""

from __future__ import annotations

import html
import re
import subprocess
from abc import ABC, abstractmethod
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

from headroom._subprocess import run
from headroom.ignore import IgnorePolicy

from ..managed_block import block_pattern, sanitize_block_text
from ._shared import claude_config_dir
from .models import (
    ProjectInfo,
    Recommendation,
    RecommendationTarget,
)

# Marker delimiters for Headroom-managed sections. Everything written between
# them goes through sanitize_block_text() first, so transcript-derived content
# cannot close the block early (see headroom.managed_block).
_MARKER_START = "<!-- headroom:learn:start -->"
_MARKER_END = "<!-- headroom:learn:end -->"
_MARKER_PATTERN = block_pattern(_MARKER_START, _MARKER_END)


def _read_text_tolerant(file_path: Path) -> str:
    """Read an existing context file that we are about to rewrite as UTF-8.

    These files are predominantly valid UTF-8 but may carry a stray legacy
    byte (e.g. a cp1252 em-dash ``0x97``). Strict UTF-8 decoding aborts the
    whole ``--apply`` on a single such byte, so fall back to UTF-8 with
    replacement: this preserves the valid UTF-8 content — a full-file cp1252
    fallback would instead turn every genuine UTF-8 em-dash into mojibake —
    and the subsequent ``write_text(encoding="utf-8")`` self-heals the file.
    """
    raw = file_path.read_bytes()
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        text = raw.decode("utf-8", errors="replace")
    return text.replace("\r\n", "\n").replace("\r", "\n")


# =============================================================================
# Abstract Writer
# =============================================================================


class ContextWriter(ABC):
    """Base class for writing recommendations to context/memory files."""

    @abstractmethod
    def write(
        self,
        recommendations: list[Recommendation],
        project: ProjectInfo,
        dry_run: bool = True,
        config: object | None = None,
    ) -> WriteResult: ...


# =============================================================================
# Write Result
# =============================================================================


class WriteResult:
    """Result of a write operation."""

    def __init__(self) -> None:
        self.files_written: list[Path] = []
        self.content_by_file: dict[Path, str] = {}
        self.dry_run: bool = True
        # Human-readable notices (e.g. legacy CLAUDE.md migration) surfaced by the CLI.
        self.warnings: list[str] = []

    def add(self, path: Path, content: str) -> None:
        self.files_written.append(path)
        self.content_by_file[path] = content


def _mutation_blocked(
    target_path: Path,
    project: ProjectInfo,
    result: WriteResult,
    config: object | None = None,
    *,
    action: str = "writing",
    policy_path: Path | None = None,
) -> bool:
    """Return True and record a warning if ``target_path`` is ignored for mutation.

    Central enforcement point for the ``.headroomignore`` / ``ignore.mutate``
    policy (see :mod:`headroom.ignore`): repositories can list generated
    agent-harness files (e.g. cARL-managed ``CLAUDE.md``/``AGENTS.md``) so
    Headroom's ``learn`` writers never overwrite them.

    ``config``, if given, should be a ``headroom.config.HeadroomConfig`` (or
    any object with an ``ignore`` attribute shaped like ``IgnoreConfig``) so
    programmatic callers (SDK integrations that already hold a
    ``HeadroomConfig``) get ``ignore.mutate`` enforcement too. The
    ``headroom learn`` CLI itself has no config-file loader today, so it
    calls writers without ``config`` and only ``.headroomignore`` applies —
    see README "Ignoring governed / generated files".
    """
    ignore_config = getattr(config, "ignore", None) if config is not None else None
    policy = IgnorePolicy.load(project.project_path, ignore_config)
    rule = policy.matching_rule(target_path, "mutate")
    if rule is None and policy_path is not None:
        rule = policy.matching_rule(policy_path, "mutate")
    if rule is None:
        return False
    result.warnings.append(
        f"Skipped {action} {target_path}: ignored for mutation by rule "
        f"'{rule.pattern}' (from {rule.source})."
    )
    return True


# =============================================================================
# Shared section builder
# =============================================================================


def _build_section(recommendations: list[Recommendation]) -> str:
    """Build the marker-delimited section content from recommendations."""
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    lines = [
        _MARKER_START,
        "## Headroom Learned Patterns",
        f"*Auto-generated by `headroom learn` on {now} — do not edit manually*",
        "",
    ]

    for rec in recommendations:
        # Section names and bodies come from transcript-derived analysis (tool
        # output, error text, user messages): they must not close our markers.
        lines.append(f"### {sanitize_block_text(rec.section)}")
        if rec.estimated_tokens_saved > 0:
            lines.append(f"*~{rec.estimated_tokens_saved:,} tokens/session saved*")
        lines.append(sanitize_block_text(rec.content))
        lines.append("")

    lines.append(_MARKER_END)
    return "\n".join(lines)


# Matches the "*~N tokens/session saved*" annotation emitted by _build_section.
_TOKENS_ANNOTATION_PATTERN = re.compile(r"\*~([\d,]+) tokens/session saved\*\n?")
# New recommendations contain raw annotations; saved blocks contain the
# escaped delimiters emitted by sanitize_block_text. Recognize both without
# unescaping transcript-derived content or weakening managed-block boundaries.
_PATTERN_ID_PATTERN = re.compile(
    r"(?:<!--|&lt;!--)\s*headroom:pattern-id:([^\s>]+)\s*(?:-->|--&gt;)\s*$"
)


def _tagged_pattern_id(line: str) -> str | None:
    """Return the id inside a bullet's pattern-id comment, if any.

    ``_build_section`` sanitizes block content, so on disk the comment reads
    ``&lt;!-- ... --&gt;``; match against the decoded line so carried-forward
    sections are still recognized as lifecycle-tracked. Callers keep the
    original (still-sanitized) line when the bullet survives.
    """
    match = _PATTERN_ID_PATTERN.search(html.unescape(line))
    return match.group(1) if match else None


def _merge_markdown_items(
    new_content: str,
    prior_content: str,
    active_item_ids: frozenset[str] | None = None,
) -> str | None:
    """Merge simple markdown bullets, preferring new text for stable IDs.

    ``active_item_ids`` is the producing learner's authoritative set of ids
    that are still alive — it must include the items that this batch omitted
    only because of ranking or top-N capping, otherwise still-active advice
    is deleted. When it is supplied, a prior item is carried forward only
    while its `headroom:pattern-id` is in the set, so an expired or
    tombstoned item is genuinely removed instead of being pinned forever.
    Prior items with no id predate id tagging: a still-active one is
    re-emitted by the current run with an id and collapses into that line by
    visible text, so dropping the untagged leftovers removes only items the
    learner no longer considers active.

    Passing ``None`` means the producer exposes no lifecycle signal, and the
    merge degrades to a plain union of new and prior items.
    """

    def _items(content: str) -> list[tuple[str | None, str, str]] | None:
        lines = [line.strip() for line in content.splitlines() if line.strip()]
        if any(not line.startswith("- ") for line in lines):
            return None
        items: list[tuple[str | None, str, str]] = []
        for line in lines:
            pattern_id = _tagged_pattern_id(line)
            visible = _PATTERN_ID_PATTERN.sub("", line).strip().casefold()
            items.append((pattern_id, visible, line))
        return items

    new_items = _items(new_content)
    prior_items = _items(prior_content)
    if new_items is None or prior_items is None:
        return None

    if active_item_ids is not None:
        prior_items = [
            item for item in prior_items if item[0] is not None and item[0] in active_item_ids
        ]

    merged: list[str] = []
    seen_ids: set[str] = set()
    seen_content: set[str] = set()
    for pattern_id, visible, line in (*new_items, *prior_items):
        if (pattern_id is not None and pattern_id in seen_ids) or visible in seen_content:
            continue
        if pattern_id is not None:
            seen_ids.add(pattern_id)
        seen_content.add(visible)
        merged.append(line)
    return "\n".join(merged)


def _authoritative_item_ids(
    recommendations: list[Recommendation],
) -> frozenset[str] | None:
    """Union every lifecycle signal the current run carries, or None.

    A section the new run did not re-emit cannot be judged by its own
    recommendation — that recommendation is exactly what is missing. The
    sets the run *does* carry stand in for it: a producer publishes one set
    per run covering all of its live items, so an id absent from every set
    in the run is one no producer still claims. Returns ``None`` when no
    recommendation carries a signal, which keeps the historical
    carry-everything behaviour for runs that cannot speak to lifecycle.
    """
    signals = [r.active_item_ids for r in recommendations if r.active_item_ids is not None]
    if not signals:
        return None
    return frozenset().union(*signals)


def _prune_carried_section(
    content: str,
    active_item_ids: frozenset[str],
) -> str | None:
    """Drop expired id-tagged bullets from a section the new run did not re-emit.

    This is the deletion path for a heading whose last item expired: the
    producer stops emitting the section entirely, so the same-section merge
    never runs and the heading would otherwise be carried forward forever.

    Only bullets carrying a `headroom:pattern-id` are removable — they come
    from a lifecycle-tracked producer, so their absence from
    ``active_item_ids`` means that producer dropped them. Untagged bullets
    are kept: unlike the same-section merge, where a still-active legacy
    item is re-emitted with an id by the same run and collapses by visible
    text, nothing in this path would bring an untagged bullet back, so
    deleting it would discard content on no evidence.

    Returns the pruned content, or ``None`` when the section carries no
    tagged bullets at all and is therefore not a tracked section to prune.
    """
    lines = [line.strip() for line in content.splitlines() if line.strip()]
    if not lines or any(not line.startswith("- ") for line in lines):
        return None
    kept: list[str] = []
    saw_tracked_item = False
    for line in lines:
        pattern_id = _tagged_pattern_id(line)
        if pattern_id is None:
            kept.append(line)
            continue
        saw_tracked_item = True
        if pattern_id in active_item_ids:
            kept.append(line)
    if not saw_tracked_item:
        return None
    return "\n".join(kept)


def extract_marker_block(file_content: str) -> str | None:
    """Return the raw text of the headroom:learn marker block, or None.

    Unlike _parse_prior_recommendations, this returns the block verbatim
    (including the start/end markers) so it can be fed back to an LLM as
    context without losing formatting. Returns None if no block is present.
    """
    match = _MARKER_PATTERN.search(file_content)
    return match.group(0) if match else None


def _parse_prior_recommendations(existing: str) -> list[Recommendation]:
    """Parse recommendations out of a prior marker block.

    Returns [] if no marker block is present or it contains no sections.
    The returned Recommendation objects are round-trip compatible with
    _build_section — target is set to a placeholder since the marker block
    itself doesn't record it (blocks are always per-file and per-target).
    """
    match = _MARKER_PATTERN.search(existing)
    if not match:
        return []
    inner = match.group(0)[len(_MARKER_START) : -len(_MARKER_END)]

    recs: list[Recommendation] = []
    for part in re.split(r"\n### ", "\n" + inner)[1:]:
        heading_line, _, body = part.partition("\n")
        heading = heading_line.strip()
        if not heading:
            continue

        tokens_saved = 0
        tokens_match = _TOKENS_ANNOTATION_PATTERN.match(body)
        if tokens_match:
            tokens_saved = int(tokens_match.group(1).replace(",", ""))
            body = body[tokens_match.end() :]

        recs.append(
            Recommendation(
                target=RecommendationTarget.CONTEXT_FILE,
                section=heading,
                content=body.rstrip(),
                estimated_tokens_saved=tokens_saved,
            )
        )
    return recs


def _merge_recommendations(
    file_path: Path,
    new_recommendations: list[Recommendation],
) -> list[Recommendation]:
    """Union new recommendations with prior ones whose section is not re-surfaced.

    Sections produced by the current run take precedence over same-named
    prior sections — the latest analysis is authoritative. Prior sections
    whose headings do not reappear in the new run are carried forward so
    a re-run doesn't silently drop accumulated learnings. To fully rebuild
    the block, delete it manually and re-run.

    Recommendations that opt into ``preserve_prior_items`` are merged at the
    item level instead, bounded by their ``active_item_ids`` lifecycle signal
    so prior items can still expire out of the file.

    That signal also reaches the carried-forward sections. A category whose
    last item expires stops producing a recommendation at all, so its
    heading never enters the same-section merge; without pruning the carry
    path too, those bullets would be pinned in the file forever. Sections
    holding no id-tagged items are carried untouched, as before.
    """
    if not file_path.exists():
        return new_recommendations
    prior = _parse_prior_recommendations(_read_text_tolerant(file_path))
    if not prior:
        return new_recommendations
    prior_by_section = {r.section: r for r in prior}
    merged_new: list[Recommendation] = []
    for recommendation in new_recommendations:
        prior_recommendation = prior_by_section.get(recommendation.section)
        if recommendation.preserve_prior_items and prior_recommendation is not None:
            merged_content = _merge_markdown_items(
                recommendation.content,
                prior_recommendation.content,
                recommendation.active_item_ids,
            )
            if merged_content is not None:
                recommendation = replace(recommendation, content=merged_content)
        merged_new.append(recommendation)

    new_sections = {r.section for r in merged_new}
    run_active_item_ids = _authoritative_item_ids(merged_new)
    carried: list[Recommendation] = []
    for prior_recommendation in prior:
        if prior_recommendation.section in new_sections:
            continue
        if run_active_item_ids is not None:
            pruned = _prune_carried_section(prior_recommendation.content, run_active_item_ids)
            if pruned is not None:
                if not pruned:
                    # Every tracked item under this heading is gone; the
                    # heading goes with them rather than outliving them.
                    continue
                prior_recommendation = replace(prior_recommendation, content=pruned)
        carried.append(prior_recommendation)
    return merged_new + carried


def _merge_into_file(file_path: Path, new_recommendations: list[Recommendation]) -> str:
    """Merge new recommendations with any existing marker block and rebuild the file."""
    merged = _merge_recommendations(file_path, new_recommendations)
    section = _build_section(merged)
    if file_path.exists():
        existing = _read_text_tolerant(file_path)
        if _MARKER_START in existing:
            return _MARKER_PATTERN.sub(lambda _match: section, existing)
        return existing.rstrip() + "\n\n" + section + "\n"
    return section + "\n"


def _strip_marker_block(content: str) -> str:
    """Remove the headroom:learn marker block from text, tidying blank lines.

    Used when migrating a stale block out of the team-shared CLAUDE.md into the
    personal CLAUDE.local.md. Returns "" if nothing but the block remained.
    """
    cleaned = _MARKER_PATTERN.sub("", content)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
    return cleaned + "\n" if cleaned else ""


def _git(repo: Path, *argv: str) -> subprocess.CompletedProcess | None:
    """Run a git command in ``repo``; None when git is absent or hangs."""
    try:
        return run(
            ["git", *argv],
            capture_output=True,
            text=True,
            cwd=repo,
            timeout=5,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return None


def _ensure_git_ignored(
    target_path: Path,
    dry_run: bool,
    project: ProjectInfo,
    result: WriteResult,
    config: object | None = None,
) -> str | None:
    """Keep a personal context file out of git via ``.git/info/exclude``.

    ``CLAUDE.local.md`` is only personal if git actually ignores it, and nothing
    makes that true by default: git ships no rule for the name and neither does
    Claude Code, so the file lands in the next ``git add -A`` and the machine-
    specific absolute paths inside it reach teammates anyway -- the exact
    outcome issue #1072 set out to prevent.

    Writes to the per-clone exclude file rather than the repo's ``.gitignore``
    because the latter is team-shared and committed: appending to it would leave
    an unexpected diff in someone else's repo, trading one kind of pollution for
    another. Returns a warning instead of acting when the file is already
    tracked -- git honors no ignore rule for tracked files, so only
    ``git rm --cached`` can fix that, and running it here would silently stage a
    deletion in the user's repo.
    """
    repo = target_path.parent
    name = target_path.name

    common_dir = _git(repo, "rev-parse", "--git-common-dir")
    if common_dir is None or common_dir.returncode != 0:
        return None  # not a git repo, or no git on PATH: nothing to ignore

    tracked = _git(repo, "ls-files", "--error-unmatch", "--", name)
    if tracked is not None and tracked.returncode == 0:
        return (
            f"{target_path} is tracked in git, so learned patterns (including "
            f"absolute paths from this machine) are committed and shared with "
            f"your team. Run `git rm --cached {name}` to untrack it; the file "
            f"itself stays on disk."
        )

    ignored = _git(repo, "check-ignore", "-q", "--", name)
    if ignored is not None and ignored.returncode == 0:
        return None  # already covered by .gitignore or a previous run

    if dry_run:
        return None

    # Deliberately unanchored: a CLAUDE.local.md at any depth is personal, and
    # anchoring would need the path relative to the repo root, which differs
    # when the project is a subdirectory of a larger repo.
    # git shares info/exclude across linked worktrees via the common dir.
    exclude = Path(common_dir.stdout.strip() or ".git")
    exclude = (repo / exclude / "info" / "exclude").resolve()
    # Linked worktrees share metadata outside the policy root, but rules for
    # the repository-relative administrative path must still protect it.
    if _mutation_blocked(
        exclude,
        project,
        result,
        config,
        action="updating",
        policy_path=Path(".git/info/exclude"),
    ):
        return None
    try:
        exclude.parent.mkdir(parents=True, exist_ok=True)
        prior = exclude.read_text(encoding="utf-8") if exclude.exists() else ""
        prefix = "" if not prior or prior.endswith("\n") else "\n"
        exclude.write_text(
            f"{prior}{prefix}\n# Personal `headroom learn` output, not team-shared\n{name}\n",
            encoding="utf-8",
        )
    except OSError:
        return None  # read-only .git, exotic setup: the write is best-effort
    return None


# =============================================================================
# Claude Code Writer
# =============================================================================


class ClaudeCodeWriter(ContextWriter):
    """Writes learned patterns to CLAUDE.local.md and MEMORY.md for Claude Code.

    Project-level learnings default to ``CLAUDE.local.md`` rather than
    ``CLAUDE.md``: per Claude Code's memory convention ``CLAUDE.md`` is
    team-shared and checked into git, while ``CLAUDE.local.md`` is personal and
    gitignored by default. Learned patterns are personal-by-default (they hold
    machine-specific absolute paths and tool-discovery byproducts), so writing
    them to the shared file pollutes it for teammates (issue #1072).

    Pass an explicit target via :meth:`set_context_target` (CLI ``--target``) to
    override -- e.g. ``CLAUDE.md`` to opt back into the shared file.
    """

    def __init__(self, context_target: str | None = None) -> None:
        # Explicit write target for CONTEXT_FILE recs (overrides the default).
        self._context_target = context_target

    def set_context_target(self, context_target: str | None) -> None:
        """Override where CONTEXT_FILE recommendations are written.

        Accepts a path relative to the project root or an absolute path.
        """
        self._context_target = context_target

    def write(
        self,
        recommendations: list[Recommendation],
        project: ProjectInfo,
        dry_run: bool = True,
        config: object | None = None,
    ) -> WriteResult:
        result = WriteResult()
        result.dry_run = dry_run

        context_recs = [r for r in recommendations if r.target == RecommendationTarget.CONTEXT_FILE]
        memory_recs = [r for r in recommendations if r.target == RecommendationTarget.MEMORY_FILE]

        if context_recs:
            target_path = self._resolve_context_path(project)
            if not _mutation_blocked(target_path, project, result, config):
                # Only ever auto-ignore a file whose name marks it personal: an
                # explicit --target CLAUDE.md is a deliberate opt-in to the shared
                # file, and ~/.claude/CLAUDE.md may sit in a dotfiles repo.
                if target_path.name.endswith(".local.md"):
                    tracked_warning = _ensure_git_ignored(
                        target_path, dry_run, project, result, config
                    )
                    if tracked_warning:
                        result.warnings.append(tracked_warning)
                # Migrate any stale block left in the team-shared CLAUDE.md by
                # older headroom versions into the new target, then strip it
                # from CLAUDE.md so the shared file is no longer polluted.
                migrated = self._migrate_legacy_block(project, target_path, result, dry_run, config)
                new_sections = {r.section for r in context_recs}
                merged_recs = context_recs + [r for r in migrated if r.section not in new_sections]
                full_content = _merge_into_file(target_path, merged_recs)
                result.add(target_path, full_content)
                if not dry_run:
                    target_path.parent.mkdir(parents=True, exist_ok=True)
                    target_path.write_text(full_content, encoding="utf-8", newline="\n")

        if memory_recs:
            memory_path = self._resolve_memory_path(project)
            if not _mutation_blocked(memory_path, project, result, config):
                full_content = _merge_into_file(memory_path, memory_recs)
                result.add(memory_path, full_content)
                if not dry_run:
                    memory_path.parent.mkdir(parents=True, exist_ok=True)
                    memory_path.write_text(full_content, encoding="utf-8", newline="\n")

        return result

    def _resolve_context_path(self, project: ProjectInfo) -> Path:
        # Explicit --target wins over every default.
        if self._context_target is not None:
            target = Path(self._context_target).expanduser()
            return target if target.is_absolute() else project.project_path / target
        # The home directory's CLAUDE.md (~/.claude/CLAUDE.md, or
        # $CLAUDE_CONFIG_DIR/CLAUDE.md) is the user's personal global memory,
        # not a team-shared file, so keep writing there.
        if project.project_path == Path.home():
            return claude_config_dir() / "CLAUDE.md"
        # Project level: default to the gitignored, personal CLAUDE.local.md so
        # we never pollute the team-shared CLAUDE.md (issue #1072).
        return project.project_path / "CLAUDE.local.md"

    def _migrate_legacy_block(
        self,
        project: ProjectInfo,
        target_path: Path,
        result: WriteResult,
        dry_run: bool,
        config: object | None = None,
    ) -> list[Recommendation]:
        """Move a stale headroom block out of CLAUDE.md into the new target.

        Only fires for the default project-level case (no explicit --target, not
        the home directory) when CLAUDE.md still carries a marker block and the
        new target doesn't yet own one. Returns the migrated recommendations so
        the caller can carry them forward; records the cleaned CLAUDE.md and a
        warning on ``result``. Honors ``dry_run`` (no writes, warning still set).
        """
        legacy_path = project.project_path / "CLAUDE.md"
        if self._context_target is not None or project.project_path == Path.home():
            return []
        if target_path == legacy_path or not legacy_path.exists():
            return []
        legacy_text = _read_text_tolerant(legacy_path)
        if _MARKER_START not in legacy_text:
            return []
        # If the target already owns a block, it is the source of truth -- don't
        # double-migrate or clobber accumulated learnings.
        if target_path.exists() and _MARKER_START in _read_text_tolerant(target_path):
            return []

        if _mutation_blocked(
            legacy_path,
            project,
            result,
            config,
            action="legacy migration from",
        ):
            result.warnings.append(
                f"Left Headroom learnings in {legacy_path}: legacy migration would "
                f"rewrite or delete that protected file."
            )
            return []

        migrated = _parse_prior_recommendations(legacy_text)
        cleaned = _strip_marker_block(legacy_text)
        gitignore_hint = f" Ensure {target_path.name} is in your .gitignore so it stays personal."
        if cleaned:
            # CLAUDE.md has hand-written content too — keep it, drop only the block.
            result.add(legacy_path, cleaned)
            result.warnings.append(
                f"Moved Headroom learnings out of {legacy_path} into {target_path}: "
                f"CLAUDE.md is team-shared, so personal learnings now live in "
                f"{target_path.name}. Review the diff before committing.{gitignore_hint}"
            )
            if not dry_run:
                legacy_path.write_text(cleaned, encoding="utf-8", newline="\n")
        else:
            # CLAUDE.md held nothing but the Headroom block — remove the husk.
            result.warnings.append(
                f"Removed {legacy_path} (it contained only Headroom learnings) and "
                f"moved them into {target_path}.{gitignore_hint}"
            )
            if not dry_run:
                legacy_path.unlink()
        return migrated

    def _resolve_memory_path(self, project: ProjectInfo) -> Path:
        if project.memory_file:
            return project.memory_file
        return project.data_path / "memory" / "MEMORY.md"


# =============================================================================
# Codex Writer (OpenAI Codex CLI)
# =============================================================================


class CodexWriter(ContextWriter):
    """Writes learned patterns to AGENTS.md and instructions.md for Codex CLI."""

    def write(
        self,
        recommendations: list[Recommendation],
        project: ProjectInfo,
        dry_run: bool = True,
        config: object | None = None,
    ) -> WriteResult:
        result = WriteResult()
        result.dry_run = dry_run

        context_recs = [r for r in recommendations if r.target == RecommendationTarget.CONTEXT_FILE]
        memory_recs = [r for r in recommendations if r.target == RecommendationTarget.MEMORY_FILE]

        if context_recs:
            agents_md = project.context_file or (project.project_path / "AGENTS.md")
            if not _mutation_blocked(agents_md, project, result, config):
                full_content = _merge_into_file(agents_md, context_recs)
                result.add(agents_md, full_content)
                if not dry_run:
                    agents_md.parent.mkdir(parents=True, exist_ok=True)
                    agents_md.write_text(full_content, encoding="utf-8", newline="\n")

        if memory_recs:
            instructions_md = project.memory_file or (project.data_path.parent / "instructions.md")
            if not _mutation_blocked(instructions_md, project, result, config):
                full_content = _merge_into_file(instructions_md, memory_recs)
                result.add(instructions_md, full_content)
                if not dry_run:
                    instructions_md.parent.mkdir(parents=True, exist_ok=True)
                    instructions_md.write_text(full_content, encoding="utf-8", newline="\n")

        return result


# =============================================================================
# Gemini Writer (Google Gemini CLI)
# =============================================================================


class GeminiWriter(ContextWriter):
    """Writes learned patterns to GEMINI.md for Gemini CLI."""

    def write(
        self,
        recommendations: list[Recommendation],
        project: ProjectInfo,
        dry_run: bool = True,
        config: object | None = None,
    ) -> WriteResult:
        result = WriteResult()
        result.dry_run = dry_run

        if not recommendations:
            return result

        gemini_md = project.context_file or (project.project_path / "GEMINI.md")
        if _mutation_blocked(gemini_md, project, result, config):
            return result
        full_content = _merge_into_file(gemini_md, recommendations)
        result.add(gemini_md, full_content)
        if not dry_run:
            gemini_md.parent.mkdir(parents=True, exist_ok=True)
            gemini_md.write_text(full_content, encoding="utf-8", newline="\n")

        return result


# =============================================================================
# Grok Writer (Grok CLI)
# =============================================================================


class GrokWriter(ContextWriter):
    """Writes learned patterns to GROK.md for Grok CLI."""

    def write(
        self,
        recommendations: list[Recommendation],
        project: ProjectInfo,
        dry_run: bool = True,
        config: object | None = None,
    ) -> WriteResult:
        result = WriteResult()
        result.dry_run = dry_run

        if not recommendations:
            return result

        grok_md = project.context_file or (project.project_path / "GROK.md")
        if _mutation_blocked(grok_md, project, result, config):
            return result
        full_content = _merge_into_file(grok_md, recommendations)
        result.add(grok_md, full_content)
        if not dry_run:
            grok_md.parent.mkdir(parents=True, exist_ok=True)
            grok_md.write_text(full_content, encoding="utf-8", newline="\n")

        return result
