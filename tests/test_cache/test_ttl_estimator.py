"""Tests for the offline cache-TTL estimator (`headroom-cache-ttl`)."""

from __future__ import annotations

import json

import pytest

from headroom.cache.ttl_estimator import (
    estimate_ttls,
    main,
    write_learned,
)
from headroom.cache.ttl_observations import resolve_learned_ttl


def _row(
    provider: str = "openai",
    model: str = "gpt-5.5",
    *,
    idle: float,
    hit: bool,
    reason: str | None = None,
) -> dict:
    return {
        "ts": 1000.0,
        "provider": provider,
        "model": model,
        "reason": reason if reason is not None else ("hit" if hit else "ttl_expiry"),
        "idle_seconds": idle,
        "ttl_assumed": 300,
        "is_miss": not hit,
        "cache_read": 1000 if hit else 0,
        "expected_cached": 1000,
    }


def _corpus(hit_idles: list[float], expiry_idles: list[float]) -> list[dict]:
    return [_row(idle=i, hit=True) for i in hit_idles] + [
        _row(idle=i, hit=False) for i in expiry_idles
    ]


def _legacy_aggregate(ttl: int) -> dict:
    # The exact record shape the earlier per-provider estimator wrote under a
    # bare provider key; `max_hit_idle` is what marks it as estimator output.
    return {
        "ttl_seconds": ttl,
        "max_hit_idle": ttl // 2,
        "hits": 5,
        "ttl_expiry_misses": 4,
        "updated_at": 1000.0,
    }


class TestEstimateTtls:
    def test_upper_bound_of_interval_is_emitted(self):
        # Alive at up to 480s, first observed death beyond that at 600s
        # -> TTL estimate is the safe upper end: 600.
        table = estimate_ttls(_corpus([120, 300, 480], [600, 900, 1200]))
        assert table["openai/gpt-5.5"]["ttl_seconds"] == 600
        assert table["openai/gpt-5.5"]["max_hit_idle"] == 480
        assert "openai" not in table  # provider aggregates are never emitted

    def test_death_at_or_below_max_hit_idle_is_not_an_upper_bound(self):
        # Deaths at 400/450 sit below a hit at 480 (variable eviction);
        # only the 700s death is beyond every observed life.
        table = estimate_ttls(_corpus([120, 300, 480], [400, 450, 700]))
        assert table["openai/gpt-5.5"]["ttl_seconds"] == 700

    def test_no_death_beyond_life_skips_key(self):
        # Every observed death overlaps the life range: no safe upper end.
        table = estimate_ttls(_corpus([120, 480, 900], [400, 450, 600]))
        assert table == {}

    def test_insufficient_samples_skip_key(self):
        assert estimate_ttls(_corpus([480], [600, 700, 800])) == {}  # 1 hit < 3
        assert estimate_ttls(_corpus([100, 200, 480], [600])) == {}  # 1 expiry < 3
        # Thresholds are tunable.
        table = estimate_ttls(_corpus([480], [600]), min_hits=1, min_expiry_misses=1)
        assert table["openai/gpt-5.5"]["ttl_seconds"] == 600

    def test_sub_one_floors_do_not_disable_the_evidence_requirement(self):
        # An interval needs one life and one death sample to exist; floors below
        # 1 must not let a key through to max()/min() on an empty list.
        rows = [_row(idle=i, hit=False) for i in (600, 700, 800)]  # deaths only
        assert estimate_ttls(rows, min_hits=0, min_expiry_misses=0) == {}
        assert estimate_ttls(rows, min_hits=-5, min_expiry_misses=-5) == {}

    def test_non_ttl_expiry_misses_are_ignored(self):
        rows = _corpus([100, 200, 480], [600, 700]) + [
            _row(idle=550, hit=False, reason="prefix_change"),
            _row(idle=560, hit=False, reason="cold_start"),
        ]
        # prefix_change/cold_start misses say nothing about TTL: still only
        # 2 ttl_expiry rows -> below the min_expiry_misses=3 floor.
        assert estimate_ttls(rows) == {}

    def test_zero_idle_and_malformed_rows_are_ignored(self):
        rows = _corpus([100, 200, 480], [600, 700, 800]) + [
            _row(idle=0.0, hit=True),
            {"provider": "", "idle_seconds": 50, "is_miss": False},
            {"provider": "openai", "model": "gpt-5.5", "idle_seconds": "nan-ish"},
            {"provider": "openai", "idle_seconds": 50, "is_miss": False},  # no model
            "not a dict",  # type: ignore[list-item]
        ]
        assert estimate_ttls(rows)["openai/gpt-5.5"]["ttl_seconds"] == 600

    def test_non_finite_idles_are_ignored(self):
        # NaN/±Inf pass a bare `idle <= 0` (NaN comparisons are all False), then
        # poison max()/min() and raise ValueError/OverflowError in int() — one
        # such row must not take the whole batch down with it.
        rows = _corpus([100, 200, 480], [600, 700, 800]) + [
            _row(idle=float(v), hit=h) for v in ("nan", "inf", "-inf") for h in (True, False)
        ]
        assert estimate_ttls(rows)["openai/gpt-5.5"]["ttl_seconds"] == 600

    def test_one_models_death_cannot_close_anothers_life_interval(self):
        # gpt-5.5 is alive at up to 1000s and never observed dead; mini dies at
        # 1200s+. Pooled per-provider these would emit ttl=1200, which the
        # consumer would then apply to gpt-5.5 (real TTL far higher) and to any
        # unseen model, recompacting a still-warm prefix. Only mini is estimable.
        rows = (
            [_row(idle=i, hit=True) for i in (900, 950, 1000)]
            + [_row(model="gpt-5.5-mini", idle=i, hit=True) for i in (100, 200, 300)]
            + [_row(model="gpt-5.5-mini", idle=i, hit=False) for i in (1200, 1300, 1400)]
        )
        table = estimate_ttls(rows)
        assert table["openai/gpt-5.5-mini"]["ttl_seconds"] == 1200
        assert "openai/gpt-5.5" not in table  # no death evidence of its own
        assert "openai" not in table


class TestWriteLearned:
    def test_merge_preserves_existing_keys(self, tmp_path):
        out = tmp_path / "learned.json"
        out.write_text(json.dumps({"kimi/k2": {"ttl_seconds": 900}}))
        write_learned({"openai/gpt-5.5": {"ttl_seconds": 600}}, str(out))
        data = json.loads(out.read_text())
        assert data["kimi/k2"]["ttl_seconds"] == 900
        assert data["openai/gpt-5.5"]["ttl_seconds"] == 600

    def test_legacy_bare_provider_keys_are_purged_on_write(self, tmp_path):
        # Files written by the earlier per-provider version of this tool carry
        # exactly the unsound cross-model aggregates the estimator no longer
        # emits; a merge that preserves them would keep feeding the consumer's
        # provider fallback forever. They are recognized by the estimator's own
        # metadata shape — a bare-provider key withOUT it is an operator-written
        # fallback and must survive the purge.
        out = tmp_path / "learned.json"
        out.write_text(
            json.dumps({"openai": _legacy_aggregate(1200), "kimi/k2": {"ttl_seconds": 900}})
        )
        write_learned({"openai/gpt-5.5": {"ttl_seconds": 600}}, str(out))
        data = json.loads(out.read_text())
        assert "openai" not in data
        assert data["kimi/k2"]["ttl_seconds"] == 900
        assert data["openai/gpt-5.5"]["ttl_seconds"] == 600

    def test_manual_bare_provider_fallbacks_survive_the_purge(self, tmp_path):
        # Both hand-written forms resolve_learned_ttl accepts for a bare
        # provider key: a {"ttl_seconds": N} dict and a plain number.
        out = tmp_path / "learned.json"
        out.write_text(
            json.dumps(
                {"openai": {"ttl_seconds": 3600}, "kimi": 900, "codex": _legacy_aggregate(1200)}
            )
        )
        write_learned({"openai/gpt-5.5": {"ttl_seconds": 600}}, str(out))
        data = json.loads(out.read_text())
        assert data["openai"] == {"ttl_seconds": 3600}
        assert data["kimi"] == 900
        assert "codex" not in data

    def test_corrupt_existing_file_is_replaced_not_fatal(self, tmp_path):
        out = tmp_path / "learned.json"
        out.write_text("{not json")
        write_learned({"openai/gpt-5.5": {"ttl_seconds": 600}}, str(out))
        assert json.loads(out.read_text())["openai/gpt-5.5"]["ttl_seconds"] == 600

    def test_non_dict_existing_file_is_replaced_not_merged(self, tmp_path):
        out = tmp_path / "learned.json"
        out.write_text("[1, 2]")
        write_learned({"openai/gpt-5.5": {"ttl_seconds": 600}}, str(out))
        assert json.loads(out.read_text()) == {"openai/gpt-5.5": {"ttl_seconds": 600}}


class TestMain:
    @pytest.fixture()
    def paths(self, tmp_path, monkeypatch):
        obs = tmp_path / "obs.jsonl"
        out = tmp_path / "learned.json"
        monkeypatch.setenv("HEADROOM_CACHE_TTL_OBS_PATH", str(obs))
        monkeypatch.setenv("HEADROOM_CACHE_TTL_LEARNED_PATH", str(out))
        return obs, out

    def test_end_to_end_resolve_learned_ttl_reads_output(self, paths):
        obs, out = paths
        rows = _corpus([120, 300, 480], [600, 900])
        rows += [_row(idle=650, hit=False)]
        # Trailing junk the parser must skip: invalid JSON, a blank line,
        # and valid JSON that is not an object.
        obs.write_text("\n".join(json.dumps(r) for r in rows) + '\nnot-json\n\n[1, 2]\n"scalar"\n')

        assert main([]) == 0
        assert resolve_learned_ttl("openai", "gpt-5.5") == 600
        # An unseen model gets nothing rather than gpt-5.5's bound: the consumer
        # falls back to its own static default instead of a TTL learned from a
        # cache population it has no evidence about.
        assert resolve_learned_ttl("openai", "other-model") is None

    def test_upgrade_purges_legacy_provider_aggregate(self, paths):
        # Upgrade regression: a learned file from the earlier per-provider
        # version must not keep serving its aggregate to unseen models after
        # a run of the model-scoped estimator — while an operator-written
        # provider fallback in the same file keeps resolving.
        obs, out = paths
        out.write_text(
            json.dumps({"openai": _legacy_aggregate(1200), "kimi": {"ttl_seconds": 3600}})
        )
        obs.write_text("\n".join(json.dumps(r) for r in _corpus([120, 300, 480], [600, 900, 1200])))

        assert main([]) == 0
        data = json.loads(out.read_text())
        assert "openai" not in data
        assert data["openai/gpt-5.5"]["ttl_seconds"] == 600
        assert resolve_learned_ttl("openai", "unseen-model") is None
        assert resolve_learned_ttl("kimi", "unseen-model") == 3600

    def test_upgrade_purge_runs_even_with_no_estimable_data(self, paths, capsys):
        # The purge must not depend on today's window producing an estimate.
        obs, out = paths
        out.write_text(json.dumps({"openai": _legacy_aggregate(1200)}))
        obs.write_text(json.dumps(_row(idle=120, hit=True)) + "\n")

        assert main([]) == 0
        assert json.loads(out.read_text()) == {}
        assert resolve_learned_ttl("openai", "unseen-model") is None

    def test_non_finite_jsonl_row_does_not_abort_the_run(self, paths):
        obs, out = paths
        # json.dumps emits bare NaN/Infinity and json.loads reads them back, so
        # the recorder can put one in the log without any hand-editing.
        # First row on purpose: max() keeps a leading NaN (every later `x > nan`
        # is False), so this is the ordering that takes the run down.
        rows = [_row(idle=float("nan"), hit=True)] + _corpus([120, 300, 480], [600, 900, 1200])
        obs.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
        assert "NaN" in obs.read_text()

        assert main([]) == 0
        assert json.loads(out.read_text())["openai/gpt-5.5"]["ttl_seconds"] == 600

    def test_sub_one_sample_floors_are_rejected(self, paths):
        with pytest.raises(SystemExit):
            main(["--min-hits", "0"])
        with pytest.raises(SystemExit):
            main(["--min-expiry-misses", "-1"])

    def test_no_estimable_data_writes_nothing(self, paths, capsys):
        obs, out = paths
        obs.write_text(json.dumps(_row(idle=120, hit=True)) + "\n")
        assert main([]) == 0
        assert not out.exists()
        assert "nothing written" in capsys.readouterr().out

    def test_missing_obs_file_is_not_an_error(self, paths):
        assert main([]) == 0

    def test_dry_run_prints_without_writing(self, paths, capsys):
        obs, out = paths
        obs.write_text("\n".join(json.dumps(r) for r in _corpus([120, 300, 480], [600, 900, 1200])))
        assert main(["--dry-run"]) == 0
        assert not out.exists()
        assert json.loads(capsys.readouterr().out)["openai/gpt-5.5"]["ttl_seconds"] == 600

    def test_explicit_paths_override_env(self, paths, tmp_path):
        obs2 = tmp_path / "other.jsonl"
        out2 = tmp_path / "other.json"
        obs2.write_text(
            "\n".join(json.dumps(r) for r in _corpus([120, 300, 480], [600, 900, 1200]))
        )
        assert main(["--obs", str(obs2), "--out", str(out2)]) == 0
        assert json.loads(out2.read_text())["openai/gpt-5.5"]["ttl_seconds"] == 600
