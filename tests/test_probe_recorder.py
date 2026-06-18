"""Tests for the opt-in compression event probe recorder."""

import json
import stat

from headroom.pipeline import PipelineEvent, PipelineStage
from headroom.proxy.probe_recorder import (
    RECORD_DIR_ENV,
    CompressionEventRecorder,
    probe_recorder_from_env,
)

ORIGINAL = [
    {
        "role": "user",
        "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "retry_limit: 3"}],
    }
]
COMPRESSED = [
    {
        "role": "user",
        "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "[compressed]"}],
    }
]


def _metadata(**overrides):
    metadata = {
        "tokens_before": 100,
        "tokens_after": 40,
        "transforms_applied": ["smart_crusher"],
        "original_messages": ORIGINAL,
    }
    metadata.update(overrides)
    return metadata


def _event(stage=PipelineStage.INPUT_COMPRESSED, messages=COMPRESSED, metadata=None):
    return PipelineEvent(
        stage=stage,
        operation="proxy.request",
        request_id="req-1",
        provider="anthropic",
        model="claude-test",
        messages=messages,
        metadata=_metadata() if metadata is None else metadata,
    )


class TestCompressionEventRecorder:
    def test_records_compression_event(self, tmp_path):
        recorder = CompressionEventRecorder(tmp_path)

        recorder.on_pipeline_event(_event())

        lines = recorder.path.read_text(encoding="utf-8").splitlines()
        assert len(lines) == 1
        record = json.loads(lines[0])
        assert record["request_id"] == "req-1"
        assert record["provider"] == "anthropic"
        assert record["model"] == "claude-test"
        assert record["tokens_before"] == 100
        assert record["tokens_after"] == 40
        assert record["transforms_applied"] == ["smart_crusher"]
        assert record["original_messages"] == ORIGINAL
        assert record["compressed_messages"] == COMPRESSED
        assert record["ts"] > 0

    def test_appends_one_line_per_event(self, tmp_path):
        recorder = CompressionEventRecorder(tmp_path)

        recorder.on_pipeline_event(_event())
        recorder.on_pipeline_event(_event())

        assert len(recorder.path.read_text(encoding="utf-8").splitlines()) == 2

    def test_ignores_other_stages(self, tmp_path):
        recorder = CompressionEventRecorder(tmp_path)

        recorder.on_pipeline_event(_event(stage=PipelineStage.INPUT_ROUTED))

        assert not recorder.path.exists()

    def test_skips_without_original_messages(self, tmp_path):
        recorder = CompressionEventRecorder(tmp_path)

        recorder.on_pipeline_event(_event(metadata=_metadata(original_messages=None)))

        assert not recorder.path.exists()

    def test_skips_zero_token_delta(self, tmp_path):
        recorder = CompressionEventRecorder(tmp_path)

        recorder.on_pipeline_event(_event(metadata=_metadata(tokens_after=100)))

        assert not recorder.path.exists()

    def test_skips_without_compressed_messages(self, tmp_path):
        recorder = CompressionEventRecorder(tmp_path)

        recorder.on_pipeline_event(_event(messages=None))

        assert not recorder.path.exists()

    def test_record_dir_is_private(self, tmp_path):
        record_dir = tmp_path / "recordings"

        CompressionEventRecorder(record_dir)

        mode = stat.S_IMODE(record_dir.stat().st_mode)
        assert mode == 0o700


class TestProbeRecorderFromEnv:
    def test_disabled_without_env(self, monkeypatch):
        monkeypatch.delenv(RECORD_DIR_ENV, raising=False)

        assert probe_recorder_from_env() is None

    def test_disabled_with_blank_env(self, monkeypatch):
        monkeypatch.setenv(RECORD_DIR_ENV, "   ")

        assert probe_recorder_from_env() is None

    def test_enabled_with_env(self, tmp_path, monkeypatch):
        record_dir = tmp_path / "recordings"
        monkeypatch.setenv(RECORD_DIR_ENV, str(record_dir))

        recorder = probe_recorder_from_env()

        assert isinstance(recorder, CompressionEventRecorder)
        assert record_dir.is_dir()

    def test_fail_open_on_unusable_path(self, tmp_path, monkeypatch):
        blocker = tmp_path / "not-a-dir"
        blocker.write_text("file", encoding="utf-8")
        monkeypatch.setenv(RECORD_DIR_ENV, str(blocker / "recordings"))

        assert probe_recorder_from_env() is None
