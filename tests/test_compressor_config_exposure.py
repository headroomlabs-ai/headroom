"""Tests for the newly exposed Rust compressor knobs.

Covers:
- lossless_min_savings_ratio plumbing (Python dataclass → PyO3 → Rust)
  and the 0.15 lockstep default on both sides.
- CompactConfig heuristics plumbing.
- SearchCompressor group_by_file output mode (`rg --heading` style).
- factor_out_constants config acceptance end-to-end.
- ContentRouter plumbing for both knobs.

Requires the rebuilt `headroom._core` extension — these tests fail loudly
(not skip) if the installed extension predates the new fields, because a
silent version skew here is exactly the parity drift the lockstep rule
exists to prevent.
"""

from __future__ import annotations

import pytest

from headroom.transforms.search_compressor import SearchCompressor, SearchCompressorConfig
from headroom.transforms.smart_crusher import SmartCrusher, SmartCrusherConfig


class TestSmartCrusherConfigExposure:
    def test_python_defaults(self):
        cfg = SmartCrusherConfig()
        assert cfg.lossless_min_savings_ratio == 0.15
        assert cfg.compaction_core_field_fraction == 0.8
        assert cfg.compaction_heterogeneous_core_ratio == 0.6
        assert cfg.compaction_max_flatten_inner_keys == 6
        assert cfg.compaction_min_buckets == 2
        assert cfg.compaction_max_buckets == 8

    def test_rust_default_lockstep(self):
        """Rust PyO3 default must equal the Python dataclass default."""
        from headroom._core import SmartCrusherConfig as RustConfig

        assert RustConfig().lossless_min_savings_ratio == 0.15
        assert RustConfig().compaction_core_field_fraction == 0.8
        assert RustConfig().compaction_max_buckets == 8

    def test_values_reach_rust(self):
        from headroom._core import SmartCrusherConfig as RustConfig

        rust_cfg = RustConfig(
            lossless_min_savings_ratio=0.42,
            compaction_core_field_fraction=0.7,
            compaction_heterogeneous_core_ratio=0.5,
            compaction_max_flatten_inner_keys=10,
            compaction_min_buckets=3,
            compaction_max_buckets=12,
        )
        assert rust_cfg.lossless_min_savings_ratio == 0.42
        assert rust_cfg.compaction_core_field_fraction == 0.7
        assert rust_cfg.compaction_heterogeneous_core_ratio == 0.5
        assert rust_cfg.compaction_max_flatten_inner_keys == 10
        assert rust_cfg.compaction_min_buckets == 3
        assert rust_cfg.compaction_max_buckets == 12

    def test_crusher_accepts_new_fields(self):
        crusher = SmartCrusher(
            config=SmartCrusherConfig(
                lossless_min_savings_ratio=0.5,
                factor_out_constants=True,
            )
        )
        # Construction succeeded and compresses without error.
        items = ",".join(f'{{"id": {i}, "status": "ok"}}' for i in range(40))
        result = crusher.crush(f"[{items}]")
        assert result.compressed

    def test_foreign_config_object_tolerated(self):
        """headroom.config.SmartCrusherConfig (the SDK-surface class) is
        structurally similar and flows through getattr fallbacks."""
        from headroom.config import SmartCrusherConfig as SdkConfig

        crusher = SmartCrusher(config=SdkConfig())  # type: ignore[arg-type]
        assert crusher is not None


class TestSearchGroupedOutput:
    INPUT = "\n".join(
        [
            "src/very/long/path/to/module.py:10:def alpha():",
            "src/very/long/path/to/module.py:20:def beta():",
            "src/very/long/path/to/module.py:30:def gamma():",
            "src/other.py:5:class Other:",
        ]
    )

    def test_standard_format_default(self):
        result = SearchCompressor(SearchCompressorConfig()).compress(self.INPUT)
        # Classic file:line:content — path on every match line.
        assert "src/very/long/path/to/module.py:10:" in result.compressed

    def test_grouped_format(self):
        result = SearchCompressor(SearchCompressorConfig(group_by_file=True)).compress(self.INPUT)
        lines = result.compressed.splitlines()
        # Path appears as a heading...
        assert "src/very/long/path/to/module.py" in lines
        # ...and match lines carry only line:content.
        assert "10:def alpha():" in lines
        # No classic-format line remains.
        assert not any(line.startswith("src/very/long/path/to/module.py:10:") for line in lines)

    def test_grouped_is_smaller(self):
        std = SearchCompressor(SearchCompressorConfig()).compress(self.INPUT)
        grp = SearchCompressor(SearchCompressorConfig(group_by_file=True)).compress(self.INPUT)
        assert len(grp.compressed) < len(std.compressed)

    def test_grouped_deterministic(self):
        c = SearchCompressor(SearchCompressorConfig(group_by_file=True))
        assert c.compress(self.INPUT).compressed == c.compress(self.INPUT).compressed


class TestContentRouterPlumbing:
    def test_router_passes_smart_crusher_config(self):
        from headroom.transforms.content_router import ContentRouter, ContentRouterConfig

        router = ContentRouter(
            ContentRouterConfig(smart_crusher=SmartCrusherConfig(lossless_min_savings_ratio=0.33))
        )
        crusher = router._get_smart_crusher()
        assert crusher.config.lossless_min_savings_ratio == 0.33

    def test_router_passes_search_grouping(self):
        from headroom.transforms.content_router import ContentRouter, ContentRouterConfig

        router = ContentRouter(ContentRouterConfig(search_group_by_file=True))
        compressor = router._get_search_compressor()
        assert compressor.config.group_by_file is True

    def test_router_defaults_off(self):
        from headroom.transforms.content_router import ContentRouter, ContentRouterConfig

        router = ContentRouter(ContentRouterConfig())
        compressor = router._get_search_compressor()
        assert compressor.config.group_by_file is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
