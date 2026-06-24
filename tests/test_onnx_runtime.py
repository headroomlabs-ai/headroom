from headroom.onnx_runtime import create_cpu_session_options


class _FakeSessionOptions:
    def __init__(self):
        self.intra_op_num_threads = None
        self.inter_op_num_threads = None
        self.enable_cpu_mem_arena = True
        self.enable_mem_pattern = True


class _FakeOrt:
    SessionOptions = _FakeSessionOptions


class _FakeSessionOptionsWithoutToggles:
    def __init__(self):
        self.intra_op_num_threads = None
        self.inter_op_num_threads = None


class _FakeOrtWithoutToggles:
    SessionOptions = _FakeSessionOptionsWithoutToggles


def test_create_cpu_session_options_disables_retention_features():
    options = create_cpu_session_options(
        _FakeOrt,
        intra_op_num_threads=1,
        inter_op_num_threads=2,
    )

    assert options.intra_op_num_threads == 1
    assert options.inter_op_num_threads == 2
    assert options.enable_cpu_mem_arena is False
    assert options.enable_mem_pattern is False


def test_create_cpu_session_options_handles_older_session_options():
    options = create_cpu_session_options(_FakeOrtWithoutToggles)

    assert options.intra_op_num_threads is None
    assert options.inter_op_num_threads is None
