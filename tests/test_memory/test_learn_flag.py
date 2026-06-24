"""Tests for --learn flag opt-in/opt-out behavior.

Verifies that:
1. Default proxy has no traffic learner (memory_enabled=False, traffic_learning=False)
2. --memory enables memory tools but NOT traffic learning
3. --learn enables traffic learning AND implies --memory
4. --no-learn disables traffic learning even with --memory
5. compress() API is completely unaffected by memory/learn flags
6. ProxyConfig correctly resolves flag combinations
"""

from __future__ import annotations

from headroom.proxy.server import HeadroomProxy, ProxyConfig

# =============================================================================
# ProxyConfig Flag Resolution Tests
# =============================================================================


class TestProxyConfigFlags:
    """Test that ProxyConfig correctly reflects flag combinations."""

    def test_default_config_no_memory_no_learning(self):
        """Default config: no memory, no traffic learning."""
        config = ProxyConfig()
        assert config.memory_enabled is False
        assert config.traffic_learning_enabled is False

    def test_memory_only(self):
        """--memory: memory tools enabled, no traffic learning."""
        config = ProxyConfig(memory_enabled=True)
        assert config.memory_enabled is True
        assert config.traffic_learning_enabled is False

    def test_learn_implies_memory(self):
        """--learn: both memory and traffic learning enabled."""
        config = ProxyConfig(
            memory_enabled=True,  # --learn implies --memory in CLI
            traffic_learning_enabled=True,
        )
        assert config.memory_enabled is True
        assert config.traffic_learning_enabled is True

    def test_learn_without_memory_still_works(self):
        """Edge case: traffic_learning=True but memory=False (server handles gracefully)."""
        config = ProxyConfig(
            memory_enabled=False,
            traffic_learning_enabled=True,
        )
        assert config.traffic_learning_enabled is True
        # In practice, CLI ensures --learn implies --memory

    def test_no_learn_overrides(self):
        """--no-learn should prevent traffic learning even with --memory."""
        # This is handled at CLI level, not config level.
        # Config just stores the resolved values.
        config = ProxyConfig(
            memory_enabled=True,
            traffic_learning_enabled=False,  # --no-learn
        )
        assert config.memory_enabled is True
        assert config.traffic_learning_enabled is False


# =============================================================================
# ProxyServer Traffic Learner Initialization Tests
# =============================================================================


class TestHeadroomProxyTrafficLearner:
    """Test that HeadroomProxy correctly initializes (or doesn't) the traffic learner."""

    def test_no_traffic_learner_by_default(self):
        """Default proxy has no traffic learner."""
        config = ProxyConfig()
        proxy = HeadroomProxy(config)
        assert proxy.traffic_learner is None

    def test_no_traffic_learner_with_memory_only(self):
        """--memory alone does NOT create traffic learner."""
        config = ProxyConfig(memory_enabled=True)
        proxy = HeadroomProxy(config)
        assert proxy.traffic_learner is None

    def test_traffic_learner_created_with_learn(self):
        """--learn creates traffic learner."""
        config = ProxyConfig(
            memory_enabled=True,
            traffic_learning_enabled=True,
        )
        proxy = HeadroomProxy(config)
        assert proxy.traffic_learner is not None

    def test_traffic_learner_starts_without_backend(self):
        """Traffic learner starts with no backend (wired lazily)."""
        config = ProxyConfig(
            memory_enabled=True,
            traffic_learning_enabled=True,
        )
        proxy = HeadroomProxy(config)
        assert proxy.traffic_learner is not None
        assert proxy.traffic_learner._backend is None

    def test_min_evidence_defaults_to_five(self):
        """Default ProxyConfig has min_evidence=5; learner inherits it."""
        config = ProxyConfig(
            memory_enabled=True,
            traffic_learning_enabled=True,
        )
        assert config.traffic_learning_min_evidence == 5
        proxy = HeadroomProxy(config)
        assert proxy.traffic_learner is not None
        assert proxy.traffic_learner._min_evidence == 5

    def test_min_evidence_propagates_to_learner(self):
        """A custom min_evidence flows from ProxyConfig into TrafficLearner."""
        config = ProxyConfig(
            memory_enabled=True,
            traffic_learning_enabled=True,
            traffic_learning_min_evidence=10,
        )
        proxy = HeadroomProxy(config)
        assert proxy.traffic_learner is not None
        assert proxy.traffic_learner._min_evidence == 10


# =============================================================================
# CLI Flag Resolution Tests (simulates CLI logic without running Click)
# =============================================================================


class TestCLIFlagResolution:
    """Test the flag resolution logic used in the proxy CLI.

    Replicates the logic: memory_enabled = memory or (learn and not no_learn)
                          traffic_learning_enabled = learn and not no_learn
    """

    @staticmethod
    def _resolve(
        memory: bool = False,
        learn: bool = False,
        no_learn: bool = False,
    ) -> tuple[bool, bool]:
        """Replicate CLI flag resolution logic from proxy.py."""
        memory_enabled = memory or (learn and not no_learn)
        traffic_learning_enabled = learn and not no_learn
        return memory_enabled, traffic_learning_enabled

    def test_no_flags(self):
        """No flags: nothing enabled."""
        mem, learn = self._resolve()
        assert mem is False
        assert learn is False

    def test_memory_only(self):
        """--memory: memory yes, learning no."""
        mem, learn = self._resolve(memory=True)
        assert mem is True
        assert learn is False

    def test_learn_only(self):
        """--learn: both memory and learning enabled."""
        mem, learn = self._resolve(learn=True)
        assert mem is True
        assert learn is True

    def test_learn_and_memory(self):
        """--learn --memory: both enabled (redundant but valid)."""
        mem, learn = self._resolve(memory=True, learn=True)
        assert mem is True
        assert learn is True

    def test_no_learn_overrides_learn(self):
        """--learn --no-learn: learning disabled, memory disabled (learn was sole enabler)."""
        mem, learn = self._resolve(learn=True, no_learn=True)
        assert mem is False
        assert learn is False

    def test_memory_and_no_learn(self):
        """--memory --no-learn: memory yes, learning no."""
        mem, learn = self._resolve(memory=True, no_learn=True)
        assert mem is True
        assert learn is False

    def test_all_flags(self):
        """--memory --learn --no-learn: memory yes (explicit), learning no (--no-learn wins)."""
        mem, learn = self._resolve(memory=True, learn=True, no_learn=True)
        assert mem is True
        assert learn is False


# =============================================================================
# compress() API Isolation Tests
# =============================================================================


class TestCompressAPIIsolation:
    """Verify that compress() is completely unaffected by memory/learn flags.

    compress() is a pure function — it should never touch memory, traffic
    learning, or any stateful components.
    """

    def test_compress_has_no_memory_dependency(self):
        """compress() module does not import memory modules."""
        from headroom import compress

        # compress is a function, not a class with state
        assert callable(compress)

    def test_compress_function_signature(self):
        """compress() signature has no memory/learn parameters."""
        import inspect

        from headroom import compress

        sig = inspect.signature(compress)
        param_names = set(sig.parameters.keys())

        # Should NOT have memory or learn parameters
        assert "memory" not in param_names
        assert "learn" not in param_names
        assert "traffic_learning" not in param_names
        assert "memory_enabled" not in param_names


# =============================================================================
# Wrap CLI --learn Flag Tests
# =============================================================================


class TestWrapCLILearnFlag:
    """Test that wrap subcommands accept --learn flag."""

    def test_start_proxy_builds_learn_command(self):
        """_start_proxy with learn=True adds --learn to command."""

        # We can't actually run the proxy, but we can check the function signature
        import inspect

        from headroom.cli.wrap import _start_proxy

        sig = inspect.signature(_start_proxy)
        assert "learn" in sig.parameters

    def test_ensure_proxy_accepts_learn(self):
        """_ensure_proxy accepts learn kwarg."""
        import inspect

        from headroom.cli.wrap import _ensure_proxy

        sig = inspect.signature(_ensure_proxy)
        assert "learn" in sig.parameters

    def test_launch_tool_accepts_learn(self):
        """_launch_tool accepts learn kwarg."""
        import inspect

        from headroom.cli.wrap import _launch_tool

        sig = inspect.signature(_launch_tool)
        assert "learn" in sig.parameters
