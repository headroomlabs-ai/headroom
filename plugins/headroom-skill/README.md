# headroom-skill

An [Agent Skill](https://agentskills.io) that teaches a coding agent when and how
to add `headroom-ai` to an app: the job words it fires on (cut input tokens,
trim tool outputs, fit a context window), the one `compress()` call, the
zero-network check, and what not to do. The skill body is `skills/headroom/SKILL.md`;
the repo-root copy at `/skills/headroom/SKILL.md` is the same file, kept in step
by `tests/test_plugin_manifests.py`.

Install:

- Claude Code: `/plugin marketplace add headroomlabs-ai/headroom` then `/plugin install headroom-skill@headroom-marketplace`
- Copilot CLI: `copilot plugin marketplace add headroomlabs-ai/headroom` then `copilot plugin install headroom-skill@headroom-marketplace`
- Any agent that reads `skills/`: `npx skills add headroomlabs-ai/headroom`
