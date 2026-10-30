import os
import sys

import pytest

from headroom.onnx_runtime import (
    ONNX_ALLOW_SPINNING_ENV,
    ONNX_CPU_ARENA_ENV,
    cpu_arena_enabled,
    create_cpu_session_options,
    hf_entry_known_absent,
    onnx_thread_spinning_enabled,
)

_THREAD_ENV_VARS = (
    "HEADROOM_ONNX_INTRA_OP_THREADS",
    "HEADROOM_ONNX_INTER_OP_THREADS",
)


@pytest.fixture(autouse=True)
def _clear_thread_env(monkeypatch):
    """Keep thread-count tests independent of the ambient environment."""
    for name in _THREAD_ENV_VARS:
        monkeypatch.delenv(name, raising=False)


class _FakeSessionOptions:
    def __init__(self):
        self.intra_op_num_threads = None
        self.inter_op_num_threads = None
        self.enable_cpu_mem_arena = True
        self.enable_mem_pattern = True
        self.config_entries: dict[str, str] = {}

    def add_session_config_entry(self, key: str, value: str) -> None:
        self.config_entries[key] = value


class _FakeOrt:
    SessionOptions = _FakeSessionOptions


class _FakeSessionOptionsWithoutToggles:
    def __init__(self):
        self.intra_op_num_threads = None
        self.inter_op_num_threads = None

    def add_session_config_entry(self, key: str, value: str) -> None:
        # No config storage on this stand-in; ORT here just accepts the call.
        return None


class _FakeOrtWithoutToggles:
    SessionOptions = _FakeSessionOptionsWithoutToggles


def test_create_cpu_session_options_disables_retention_features(monkeypatch):
    """Non-Windows keeps the legacy low-RSS behavior: arena + mem pattern off."""
    monkeypatch.delenv(ONNX_CPU_ARENA_ENV, raising=False)
    monkeypatch.setattr(sys, "platform", "linux")

    options = create_cpu_session_options(
        _FakeOrt,
        intra_op_num_threads=1,
        inter_op_num_threads=2,
    )

    assert options.intra_op_num_threads == 1
    assert options.inter_op_num_threads == 2
    assert options.enable_cpu_mem_arena is False
    assert options.enable_mem_pattern is False


def test_create_cpu_session_options_darwin_unchanged(monkeypatch):
    monkeypatch.delenv(ONNX_CPU_ARENA_ENV, raising=False)
    monkeypatch.setattr(sys, "platform", "darwin")

    options = create_cpu_session_options(_FakeOrt)

    assert options.enable_cpu_mem_arena is False
    assert options.enable_mem_pattern is False


def test_create_cpu_session_options_keeps_arena_on_windows(monkeypatch):
    """Disabling the arena on Windows degrades inference by orders of
    magnitude (onnxruntime#11627) — ORT defaults must stay untouched there."""
    monkeypatch.delenv(ONNX_CPU_ARENA_ENV, raising=False)
    monkeypatch.setattr(sys, "platform", "win32")

    options = create_cpu_session_options(_FakeOrt, intra_op_num_threads=3)

    assert options.enable_cpu_mem_arena is True
    assert options.enable_mem_pattern is True
    assert options.intra_op_num_threads == 3


def test_arena_env_override_forces_on(monkeypatch):
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setenv(ONNX_CPU_ARENA_ENV, "1")

    assert cpu_arena_enabled() is True
    options = create_cpu_session_options(_FakeOrt)
    assert options.enable_cpu_mem_arena is True


def test_arena_env_override_forces_off(monkeypatch):
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setenv(ONNX_CPU_ARENA_ENV, "0")

    assert cpu_arena_enabled() is False
    options = create_cpu_session_options(_FakeOrt)
    assert options.enable_cpu_mem_arena is False


def test_arena_env_invalid_falls_back_to_platform_default(monkeypatch):
    monkeypatch.setenv(ONNX_CPU_ARENA_ENV, "bananas")

    monkeypatch.setattr(sys, "platform", "win32")
    assert cpu_arena_enabled() is True
    monkeypatch.setattr(sys, "platform", "linux")
    assert cpu_arena_enabled() is False


def test_create_cpu_session_options_handles_older_session_options(monkeypatch):
    monkeypatch.delenv(ONNX_CPU_ARENA_ENV, raising=False)
    monkeypatch.setattr(sys, "platform", "linux")

    options = create_cpu_session_options(_FakeOrtWithoutToggles)

    # Older SessionOptions without the toggle attributes must not raise.
    assert isinstance(options.intra_op_num_threads, int)
    assert options.intra_op_num_threads > 0
    assert options.inter_op_num_threads == 1


def test_create_cpu_session_options_defaults_to_positive_cpuset():
    """With no explicit values and no env override, intra is sized to the cpuset."""
    options = create_cpu_session_options(_FakeOrt)

    assert isinstance(options.intra_op_num_threads, int)
    assert options.intra_op_num_threads > 0
    assert options.inter_op_num_threads == 1


def test_create_cpu_session_options_honors_thread_env_overrides(monkeypatch):
    monkeypatch.setenv("HEADROOM_ONNX_INTRA_OP_THREADS", "3")
    monkeypatch.setenv("HEADROOM_ONNX_INTER_OP_THREADS", "2")

    options = create_cpu_session_options(_FakeOrt)

    assert options.intra_op_num_threads == 3
    assert options.inter_op_num_threads == 2


def test_create_cpu_session_options_ignores_invalid_thread_env(monkeypatch):
    monkeypatch.setenv("HEADROOM_ONNX_INTRA_OP_THREADS", "not-a-number")

    options = create_cpu_session_options(_FakeOrt)

    # Invalid override falls back to the cpuset-sized default.
    assert isinstance(options.intra_op_num_threads, int)
    assert options.intra_op_num_threads > 0


def test_create_cpu_session_options_explicit_values_ignore_env(monkeypatch):
    monkeypatch.setenv("HEADROOM_ONNX_INTRA_OP_THREADS", "9")

    options = create_cpu_session_options(_FakeOrt, intra_op_num_threads=1)

    # Explicit caller value wins over the env override.
    assert options.intra_op_num_threads == 1


def test_thread_spinning_disabled_by_default(monkeypatch):
    # #2495: ORT thread pools spin-wait on all cores between inferences, so a
    # long-lived proxy pegs every core while idle. Disable spinning by default.
    monkeypatch.delenv(ONNX_ALLOW_SPINNING_ENV, raising=False)
    monkeypatch.delenv(ONNX_CPU_ARENA_ENV, raising=False)

    assert onnx_thread_spinning_enabled() is False
    options = create_cpu_session_options(_FakeOrt)
    assert options.config_entries.get("session.intra_op.allow_spinning") == "0"
    assert options.config_entries.get("session.inter_op.allow_spinning") == "0"


def test_thread_spinning_env_can_reenable(monkeypatch):
    monkeypatch.setenv(ONNX_ALLOW_SPINNING_ENV, "1")
    monkeypatch.delenv(ONNX_CPU_ARENA_ENV, raising=False)

    assert onnx_thread_spinning_enabled() is True
    options = create_cpu_session_options(_FakeOrt)
    assert "session.intra_op.allow_spinning" not in options.config_entries
    assert "session.inter_op.allow_spinning" not in options.config_entries


def test_thread_spinning_env_explicit_off(monkeypatch):
    monkeypatch.setenv(ONNX_ALLOW_SPINNING_ENV, "0")

    assert onnx_thread_spinning_enabled() is False
    options = create_cpu_session_options(_FakeOrt)
    assert options.config_entries.get("session.intra_op.allow_spinning") == "0"


def test_spinning_disable_is_best_effort_on_older_ort(monkeypatch):
    # An ORT build that rejects the config key must not break session creation.
    monkeypatch.delenv(ONNX_ALLOW_SPINNING_ENV, raising=False)
    monkeypatch.setattr(sys, "platform", "linux")

    class _RejectingSessionOptions(_FakeSessionOptions):
        def add_session_config_entry(self, key: str, value: str) -> None:
            raise RuntimeError(f"unknown config key: {key}")

    class _RejectingOrt:
        SessionOptions = _RejectingSessionOptions

    # Must not raise.
    options = create_cpu_session_options(_RejectingOrt)
    assert options.enable_cpu_mem_arena is False


def _write_fake_hf_cache(
    root: str, repo_id: str, revision: str, *, no_exist_files: list[str]
) -> None:
    """Build a minimal on-disk HF hub cache layout for a single repo/revision.

    Mirrors the real cache structure closely enough for
    ``huggingface_hub.try_to_load_from_cache`` to read it: a ``refs/<name>``
    pointer file, a ``snapshots/<hash>`` directory, and a
    ``.no_exist/<hash>/<filename>`` marker per file whose absence is cached.
    """
    from huggingface_hub.file_download import repo_folder_name

    repo_folder = os.path.join(root, repo_folder_name(repo_id=repo_id, repo_type="model"))
    os.makedirs(os.path.join(repo_folder, "refs"), exist_ok=True)
    with open(os.path.join(repo_folder, "refs", revision), "w") as f:
        f.write("abc123")
    os.makedirs(os.path.join(repo_folder, "snapshots", "abc123"), exist_ok=True)
    no_exist_dir = os.path.join(repo_folder, ".no_exist", "abc123")
    os.makedirs(no_exist_dir, exist_ok=True)
    for filename in no_exist_files:
        open(os.path.join(no_exist_dir, filename), "w").close()


def test_hf_entry_known_absent_true_when_404_was_cached(tmp_path, monkeypatch):
    from huggingface_hub import constants

    _write_fake_hf_cache(str(tmp_path), "acme/widget", "main", no_exist_files=["merged.pt"])
    monkeypatch.setattr(constants, "HF_HUB_CACHE", str(tmp_path))
    monkeypatch.delenv("HEADROOM_HF_PIN", raising=False)

    assert hf_entry_known_absent("acme/widget", "merged.pt") is True


def test_hf_entry_known_absent_false_when_never_checked(tmp_path, monkeypatch):
    from huggingface_hub import constants

    _write_fake_hf_cache(str(tmp_path), "acme/widget", "main", no_exist_files=[])
    monkeypatch.setattr(constants, "HF_HUB_CACHE", str(tmp_path))
    monkeypatch.delenv("HEADROOM_HF_PIN", raising=False)

    assert hf_entry_known_absent("acme/widget", "merged.pt") is False


def test_hf_entry_known_absent_false_when_repo_not_cached_at_all(tmp_path, monkeypatch):
    from huggingface_hub import constants

    monkeypatch.setattr(constants, "HF_HUB_CACHE", str(tmp_path))
    monkeypatch.delenv("HEADROOM_HF_PIN", raising=False)

    assert hf_entry_known_absent("nobody/nothing", "merged.pt") is False
