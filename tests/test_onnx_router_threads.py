"""The technique-router session must cap its ONNX intra-op thread pool.

The classifier runs a MiniLM INT8 graph over a fixed [1, 64] input, so its
thread pool saturates almost immediately. Built with a bare
``create_cpu_session_options(ort)`` the pool instead scales with the host core
count, which costs CPU without buying latency on a many-core box.

``headroom/memory/adapters/embedders.py`` already pins its own session, so this
keeps the two ONNX call sites consistent.
"""

from __future__ import annotations

import sys
import types

import pytest

from headroom.image.onnx_router import (
    _CLASSIFIER_MAX_INTRA_OP_THREADS,
    OnnxTechniqueRouter,
    _available_cpu_count,
    _classifier_intra_op_threads,
)


class _FakeSessionOptions:
    def __init__(self) -> None:
        self.intra_op_num_threads: int | None = None
        self.inter_op_num_threads: int | None = None
        self.enable_cpu_mem_arena = True
        self.enable_mem_pattern = True
        self.config_entries: dict[str, str] = {}

    def add_session_config_entry(self, key: str, value: str) -> None:
        self.config_entries[key] = value


class _FakeInferenceSession:
    def __init__(self, model_path, sess_options, providers=None):
        self.model_path = model_path
        self.sess_options = sess_options
        self.providers = providers


def _install_fake_ort(monkeypatch) -> list[_FakeInferenceSession]:
    """Stand in for onnxruntime, which onnx_router imports inside the method."""
    created: list[_FakeInferenceSession] = []

    def _make(model_path, sess_options, providers=None):
        session = _FakeInferenceSession(model_path, sess_options, providers)
        created.append(session)
        return session

    fake_ort = types.SimpleNamespace(
        SessionOptions=_FakeSessionOptions,
        InferenceSession=_make,
    )
    monkeypatch.setitem(sys.modules, "onnxruntime", fake_ort)
    return created


def _install_fake_tokenizers(monkeypatch) -> None:
    class _Tokenizer:
        @staticmethod
        def from_file(path):
            return types.SimpleNamespace(
                enable_truncation=lambda **_: None,
                enable_padding=lambda **_: None,
            )

    monkeypatch.setitem(sys.modules, "tokenizers", types.SimpleNamespace(Tokenizer=_Tokenizer))


def _load_classifier(monkeypatch, tmp_path):
    created = _install_fake_ort(monkeypatch)
    _install_fake_tokenizers(monkeypatch)

    config_path = tmp_path / "config.json"
    config_path.write_text('{"id2label": {"0": "preserve"}}')
    model_path = tmp_path / "model_quantized.onnx"
    model_path.write_bytes(b"")
    tokenizer_path = tmp_path / "tokenizer.json"
    tokenizer_path.write_text("{}")

    def _download(repo, filename):
        return str(tmp_path / filename)

    monkeypatch.setattr("headroom.image.onnx_router.hf_hub_download_local_first", _download)

    router = OnnxTechniqueRouter()
    router._load_classifier()
    return created


@pytest.fixture
def loaded_classifier(monkeypatch, tmp_path):
    return _load_classifier(monkeypatch, tmp_path)


def test_classifier_session_caps_intra_op_threads(loaded_classifier):
    assert len(loaded_classifier) == 1
    options = loaded_classifier[0].sess_options

    assert options.intra_op_num_threads == _classifier_intra_op_threads()
    assert options.intra_op_num_threads <= _CLASSIFIER_MAX_INTRA_OP_THREADS
    assert options.inter_op_num_threads == 1


def _fake_cpu_topology(monkeypatch, *, host, affinity=None, process="absent"):
    """Model a host/container CPU topology for the count helpers.

    ``affinity`` is the CPU set ``sched_getaffinity`` reports (``None`` removes
    the call, as on macOS/Windows); ``process`` is what ``process_cpu_count``
    returns, or ``"absent"`` for pre-3.13 interpreters.
    """
    monkeypatch.setattr("headroom.image.onnx_router.os.cpu_count", lambda: host)

    if affinity is None:
        monkeypatch.delattr("os.sched_getaffinity", raising=False)
    else:
        monkeypatch.setattr("os.sched_getaffinity", lambda _pid: set(affinity), raising=False)

    if process == "absent":
        monkeypatch.delattr("os.process_cpu_count", raising=False)
    else:
        monkeypatch.setattr("os.process_cpu_count", lambda: process, raising=False)


def test_classifier_thread_cap_never_exceeds_available_cores(monkeypatch):
    """A 2-core host must not be told to run 4 intra-op threads."""
    _fake_cpu_topology(monkeypatch, host=2)
    assert _classifier_intra_op_threads() == 2

    _fake_cpu_topology(monkeypatch, host=64)
    assert _classifier_intra_op_threads() == _CLASSIFIER_MAX_INTRA_OP_THREADS

    _fake_cpu_topology(monkeypatch, host=None)
    assert _classifier_intra_op_threads() == 1


def test_thread_cap_follows_process_affinity_not_host_cores(monkeypatch):
    """A cpuset-restricted container on a big host must not get 4 threads.

    ``os.cpu_count()`` still reports the host's 64 cores inside such a cgroup,
    so counting cores that way would reintroduce the oversubscription the cap
    is meant to prevent.
    """
    _fake_cpu_topology(monkeypatch, host=64, affinity={3, 11})

    assert _available_cpu_count() == 2
    assert _classifier_intra_op_threads() == 2


def test_thread_cap_uses_process_cpu_count_without_affinity_api(monkeypatch):
    """Without ``sched_getaffinity`` the 3.13 process count still wins."""
    _fake_cpu_topology(monkeypatch, host=64, affinity=None, process=2)

    assert _available_cpu_count() == 2
    assert _classifier_intra_op_threads() == 2


def test_thread_cap_falls_back_to_host_cores(monkeypatch):
    """With no usable affinity-aware answer, the host count is all we have."""
    _fake_cpu_topology(monkeypatch, host=2, affinity=None)
    assert _available_cpu_count() == 2

    # ``process_cpu_count()`` returns None when it cannot determine the count.
    _fake_cpu_topology(monkeypatch, host=2, affinity=None, process=None)
    assert _available_cpu_count() == 2
    assert _classifier_intra_op_threads() == 2


def test_classifier_session_honors_affinity_on_many_core_host(monkeypatch, tmp_path):
    """The session itself, not just the helper, respects the process affinity."""
    _fake_cpu_topology(monkeypatch, host=64, affinity={0, 5})

    created = _load_classifier(monkeypatch, tmp_path)

    assert created[0].sess_options.intra_op_num_threads == 2


def test_classifier_session_keeps_retention_defaults(loaded_classifier):
    """The cap must not bypass create_cpu_session_options' other settings."""
    options = loaded_classifier[0].sess_options

    assert options.config_entries.get("session.intra_op.allow_spinning") == "0"
    assert options.config_entries.get("session.inter_op.allow_spinning") == "0"
