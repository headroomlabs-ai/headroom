# Agent Prompt: LeanContext Followup

You are a loop engineering agent. Verify and complete the LeanContext integration into Headroom.

**PR:** #1460 | **Issue:** #1461

## Checklist

1. Check CI is green: https://github.com/headroomlabs-ai/headroom/actions/workflows/leanctx-ci.yml
2. Merge main, resolve conflicts: `git fetch upstream main && git merge upstream/main`
3. Ensure Python LeanContext coexists with existing lean-ctx Rust binary
4. Run `python -m pytest tests/ -k "rtk or context or lean" -v`
5. Update PR description with CI badge and findings

**If all pass:** comment "✅ Ready for review. @tejas pure-Python RTK alternative."
**If fail:** fix simple ones, document complex ones on the PR.

Loop engineering in action — this is iteration N of "make context filtering better."
