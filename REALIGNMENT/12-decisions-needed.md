# 12 — Decisions Needed

Open questions the realignment can't resolve unilaterally. Greenlight or alternative each before the corresponding PR lands.

---

## Q1. Phase A timing — land tonight or wait?

**Recommendation:** Land **PR-A1 tonight**. It's a small diff (-180/+30) that eliminates the worst cache-killer cluster (P0-3, P0-4, P0-5 stop firing immediately). The proxy goes to passthrough on `/v1/messages`; compression returns in Phase B. Net positive because today's compression is actively destroying cache hit rate.

PR-A2 through PR-A8 land over the rest of the week.

**Alternative:** Hold all of Phase A until the synthesis is "perfect." Risk: cache hit rate stays poor.

---

## Q2. ICM removal scope — Tier 1+2, or include Tier 3?

**Recommendation: Tier 1 + Tier 2 in Phase B PR-B1** (~10K LOC).

- **Tier 1** (ICM proper): `intelligent_context.py`, `manager.rs`, `icm.rs`, the proxy call site.
- **Tier 2** (subsystems whose only consumer is ICM): `RollingWindow`, `ProgressiveSummarizer`, `scoring.py`, `tool_crusher.py`, `MessageScorer`, all of `crates/headroom-core/src/scoring/` and `relevance/`, most of `context/` (keep `safety.rs`).
- **Tier 3** (separable cleanup): `CacheAligner` rewrite path is in Phase A PR-A2 (already scheduled). Memory `_inject_system_context` paths in Phase A PR-A2 + Phase B PR-B6 (already scheduled).

So "Tier 1 + Tier 2" is the right scope for the Phase B big-delete PR; Tier 3 is already covered by Phase A and Phase B's other PRs.

**Alternative:** Stop at Tier 1 (just ICM proper). Risk: ~6 K LOC of dead-but-still-imported scoring/relevance machinery; future contributors won't know it's dead.

---

## Q3. MessageScorer Rust port — delete?

**Recommendation: Delete.**

The PR #338 / #343 port (April 2026) was investment in the wrong abstraction (per Agent G's audit: scoring's only consumer is `DropByScoreStrategy::try_fit`, which Phase B retires). Keeping it as a dead crate creates maintenance debt and confusion. Sunk cost stays sunk; the parity-harness scaffolding learnings carry forward to live-zone work where they actually matter.

Folded into Phase B PR-B1.

**Alternative:** Keep the crate around as off-path "in case scoring is needed later." Risk: dead-code review burden every PR.

---

## Q4. Stage 3g (lossless-first compression pipeline, issue #315) — re-scope or close?

**Context:** Per project memory `~/.claude/projects/-Users-tchopra-claude-projects-headroom/memory/project_lossless_first_pipeline.md`, Stage 3g was queued to formalize "lossless-then-lossy-then-CCR ordering as a `CompressionPipeline` orchestrator + `LosslessTransform`/`LossyTransform` traits." The plan assumed an ICM-style orchestrator over the messages array.

**Recommendation:** **Re-scope** issue #315 to "live-zone-only pipeline orchestrator." The traits stay (`LosslessTransform`/`LossyTransform`); the scope changes from "history compactor" to "live-zone block dispatcher." This is what Phase B PR-B2 builds. Update issue #315's body to reflect the realignment.

**Alternative:** Close issue #315 and treat Phase B PR-B2 as fulfilling its intent. Risk: history of the decision is lost.

---

## Q5. Headroom Loop / AWS Marketplace BYOC — affected scope?

**Context:** Per project memory `project_headroom_loop.md` (enterprise paid product) and `project_headroom_aws_marketplace.md` (BYOC CFN stack in customer VPC). Both depend on the OSS proxy.

**Recommendation:** The realignment **strengthens** both:
- Headroom Loop's value proposition is "trace stream + enterprise compression policy"; Phase F's auth-mode policy is exactly the surface Loop wants to gate on.
- AWS Marketplace BYOC's pitch is "context compression in front of Bedrock"; Phase D's native Bedrock support makes that pitch real (today's LiteLLM-converted Bedrock path was fake; Phase D fixes it).

No re-scoping needed; revisit after Phase D lands.

**Alternative:** Pause Headroom Loop / Marketplace work until Phase D completes. Recommended if their roadmap conflicts with Phase D timing.

---

## Q6. `make test-parity` per-PR gate — enable now or wait?

**Recommendation:** Enable now (Phase I PR-I6) with the existing stubs. `Skipped` is permitted; `Diff` fails the build. As Phase I PR-I5 promotes stubs to real comparators, the per-PR gate gradually tightens.

**Alternative:** Wait until all stubs are real. Risk: parity divergence merges silently for the next month.

---

## Q7. Operator config switch — explicit `HEADROOM_PROXY_BACKEND` env var, or implicit?

**Context:** During Phase H rollout, operators need a way to choose Python vs Rust proxy.

**Recommendation:** Add `HEADROOM_PROXY_BACKEND={python|rust}` env var in Phase H PR-H1; default to `rust` once the canary in Phase I PR-I4 confirms ≥99.9% byte-equality. Keep the Python proxy alive in the codebase for 30 days post-Phase-H as an explicit rollback target. After 30 days of stable Rust operation, run Phase H PR-H2/H3 to delete Python.

**Alternative:** Cut over implicitly (`headroom proxy start` always uses Rust after Phase H). Riskier; no clean rollback path.

---

## Q8. Container image strategy — single binary or multi-stage?

**Recommendation:** Single binary (`headroom-proxy` Rust). Container is `FROM scratch` or `FROM gcr.io/distroless/static`. Image size drops from ~500 MB (with Python + LiteLLM + ONNX models) to ~50 MB.

**Alternative:** Multi-stage Docker with Rust binary + Python sidecar (for evals/learn/memory writers). Recommended only if those subsystems become production-relevant; today they're CLI tools.

---

## Q9. RTK proxy-side invocation — ever revisit?

**Recommendation:** **No, document the decision in `docs/rtk-architecture.md`** (Phase G PR-G3). The argument:
1. Cache hot zone risk: shell-out + buffer per tool result is correctness-fragile.
2. Parallel implementation: `crates/headroom-core/src/transforms/log_compressor.rs` covers post-hoc log/output compression; RTK rewrites *commands* (different value).
3. RTK itself is a third-party binary the team doesn't control; an upstream version change silently busts cache.

If a future requirement emerges (e.g., "Headroom must compress shell output for users who don't run wrap"), reconsider with explicit cache-safety design.

**Alternative:** Build proxy-side RTK as a feature-flagged opt-in. Recommended only if the wrap-CLI breadth (PR-G1) doesn't cover enough surface.

---

## Q10. Bedrock/Vertex priority — parallel with proxy port (Phase D in calendar) or after Phase H?

**Recommendation:** **Parallel.** Phase D blocks H2 (Python LiteLLM retirement) but not H1 (Python proxy retirement). Run Phase D and Phase C/E/F/G concurrently.

**Alternative:** Sequential, Phase D after Phase H. Risk: Bedrock/Vertex users stay on the broken Python LiteLLM path for an extra month.

---

## Q11. Memory subsystem — auto-tail mode default, or tool-only?

**Recommendation:** Auto-tail mode default in Phase B PR-B6, with tool-only mode behind a flag. Migrate users to tool-only over the next 6 months once docs and tooling are mature. Auto-tail is byte-deterministic (per the cache-safety invariant) and matches existing UX.

**Alternative:** Force tool-only immediately. Risk: breaks customers' existing memory-augmented prompts.

---

## Q12. Parity harness post-Phase-H — keep or delete?

**Context:** After Phase H deletes Python, `crates/headroom-parity/` no longer has a Python side to compare against. Per Phase H PR-H3, this is a decision point.

**Recommendation:** **Repurpose**, don't delete. Rename to `crates/headroom-version-parity/` and use it to compare current-Rust-version vs previous-Rust-version on the recorded fixtures. Catches Rust-vs-Rust regressions during future ML compressor variants (e.g., when Kompress is ported to Rust via `ort`).

**Alternative:** Delete entirely. Save ~2K LOC. Risk: no automated regression test for compressor changes.

---

## Q13. Auth-mode UA detection list — which CLIs to recognize?

**Phase F PR-F1 starts with this list:**
- `claude-cli/` (Anthropic CLI)
- `claude-code/` (Claude Code)
- `codex-cli/` (Codex CLI)
- `cursor/` (Cursor IDE)
- `claude-vscode/`
- `github-copilot/`
- `anthropic-cli/`
- `antigravity/` (Cloudcode Antigravity)

**Recommendation:** Extend over time as new CLIs emerge. Alphabetic sort for determinism. Document in `docs/auth-modes.md`.

**Alternative:** Start with a smaller list; expand reactively. Risk: subscription users mis-classified as PAYG and fingerprint-leaked.

---

## Q14. The ICM removal blast radius — confirm acceptable

**Counts:**
- Lines deleted (Python): ~3,300
- Lines deleted (Rust): ~4,500
- Files deleted: ~30
- Tests deleted: ~50
- PRs that recently merged but become wasted work: PR #338, PR #343 (MessageScorer Rust port)
- Project memory updates needed: 1 (the "53270 lines" content_router.py figure was wrong by 25× — already corrected in `MEMORY.md`).

**Recommendation:** Acceptable. The cache-killer bugs cost more than the deleted code's hypothetical future value.

---

## Q15. Calendar + capacity — sequential or parallel?

**Sequential calendar:** ~13 weeks. One contributor working through phases A→I.
**Parallel calendar:** ~8 weeks with 2-3 contributors splitting along these natural boundaries:
- Lead: Phase A (lockdown), Phase B (live-zone), Phase H (retirement) — the critical path.
- Contributor 2: Phase C (Rust proxy paths), Phase D (Bedrock/Vertex). Self-contained.
- Contributor 3 (optional): Phase E (cache stabilization), Phase F (auth-mode), Phase G (RTK + obs), Phase I (test infra). Mostly independent.

**Recommendation:** Parallel. The bug list is real and the cache hit rate is hemorrhaging in production today.

---

## Quick answer template

For decision sign-off, fill in this block:

```
Q1 (Phase A timing): [ ] tonight  [ ] wait
Q2 (ICM scope):       [ ] Tier 1+2  [ ] Tier 1 only  [ ] all 3 tiers
Q3 (MessageScorer):   [ ] delete  [ ] keep
Q4 (issue #315):      [ ] re-scope  [ ] close
Q5 (Loop/Marketplace):[ ] proceed unchanged  [ ] pause until D
Q6 (parity gate):     [ ] enable now  [ ] wait
Q7 (operator switch): [ ] env var w/ default rust  [ ] implicit cutover
Q8 (container):       [ ] single binary  [ ] multi-stage
Q9 (RTK proxy-side):  [ ] document never  [ ] feature-flag for future
Q10 (Bedrock priority):[ ] parallel  [ ] sequential after H
Q11 (memory mode):    [ ] auto-tail default  [ ] tool-only force
Q12 (parity harness): [ ] repurpose  [ ] delete
Q13 (UA list):        [ ] approve list  [ ] revise: ___________
Q14 (ICM blast radius): [ ] accept  [ ] reduce scope
Q15 (calendar):       [ ] parallel (2-3 contributors)  [ ] sequential
```
