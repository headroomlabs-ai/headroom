class TestMidTurnSteering:
    def test_mid_turn_queue_exists_on_streaming_mixin(self):
        """StreamingMixin has _mid_turn_queues class attribute after the fix."""
        from headroom.proxy.handlers.streaming import StreamingMixin

        assert hasattr(StreamingMixin, "_mid_turn_queues")
        assert hasattr(StreamingMixin, "_active_streams")

    def test_mid_turn_message_queued_when_stream_active(self):
        """When a session has an active stream, mid-turn messages are queued."""
        from headroom.proxy.handlers.streaming import StreamingMixin

        mixin = StreamingMixin()
        session_key = "test-session-123"
        mixin._active_streams.add(session_key)
        body = {"messages": [{"role": "user", "content": "follow-up"}]}
        result = mixin._queue_mid_turn_message(session_key, body)
        assert result["status"] == 202
        assert result["event"] == "headroom_queued"
        assert not mixin._mid_turn_queues[session_key].empty()
        queued = mixin._mid_turn_queues[session_key].get_nowait()
        assert queued == body
        # Cleanup
        mixin._active_streams.discard(session_key)
        del mixin._mid_turn_queues[session_key]

    def test_no_queue_when_no_prior_stream(self):
        """When no stream is active, _mid_turn_queues stays empty for the session."""
        from headroom.proxy.handlers.streaming import StreamingMixin

        mixin = StreamingMixin()
        session_key = "inactive-session"
        assert session_key not in mixin._active_streams
        assert session_key not in mixin._mid_turn_queues
