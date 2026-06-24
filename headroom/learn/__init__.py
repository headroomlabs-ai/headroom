"""Headroom Learn — offline session learning for coding agents.

Analyzes conversation logs using an LLM to extract actionable patterns
and generates context (CLAUDE.md, AGENTS.md, GEMINI.md, etc.) that
prevents future token waste.

Plugin architecture:
    plugins/claude.py  ─┐
    plugins/codex.py   ─┤→  Analyzer (LLM)  →  Writer (adapter)
    plugins/gemini.py  ─┘

Built-in plugins are auto-discovered from headroom.learn.plugins.*.
External plugins register via the ``headroom.learn_plugin`` entry point.
"""
