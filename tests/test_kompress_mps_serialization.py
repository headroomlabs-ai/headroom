"""Device transfers must share Kompress's execution slot with inference."""

import threading

import headroom.transforms.kompress_compressor as kc


def test_validation_device_copy_holds_execution_slot(monkeypatch):
    semaphore = threading.BoundedSemaphore(1)
    copy_saw_slot: list[bool] = []

    class FakeTensor:
        def to(self, _device):
            acquired = semaphore.acquire(blocking=False)
            if acquired:
                semaphore.release()
            copy_saw_slot.append(not acquired)
            return self

    class FakeEncoding(dict):
        def __init__(self):
            super().__init__(input_ids=FakeTensor(), attention_mask=FakeTensor())

    class FakeTokenizer:
        def __call__(self, *_args, **_kwargs):
            return FakeEncoding()

    class FakeScore:
        def detach(self):
            return self

        def cpu(self):
            return self

    class FakeModel:
        def get_scores(self, input_ids, attention_mask):
            return [FakeScore()]

    monkeypatch.setattr(kc, "_execution_semaphore", lambda *_args, **_kwargs: semaphore)

    kc._validate_pytorch_device(FakeModel(), FakeTokenizer(), "mps")

    assert copy_saw_slot == [True, True]


def test_single_and_batch_device_copies_hold_execution_slot(monkeypatch):
    semaphore = threading.BoundedSemaphore(1)
    copy_saw_slot: list[bool] = []

    class FakeTensor:
        def __init__(self, row_lengths):
            self.row_lengths = row_lengths

        def to(self, _device):
            acquired = semaphore.acquire(blocking=False)
            if acquired:
                semaphore.release()
            copy_saw_slot.append(not acquired)
            return self

    class FakeEncoding(dict):
        def __init__(self, row_lengths):
            super().__init__(
                input_ids=FakeTensor(row_lengths),
                attention_mask=FakeTensor(row_lengths),
            )
            self.row_lengths = row_lengths

        def word_ids(self, batch_index=0):
            return list(range(self.row_lengths[batch_index]))

    class FakeTokenizer:
        def __call__(self, words, **_kwargs):
            rows = words if words and isinstance(words[0], list) else [words]
            return FakeEncoding([len(row) for row in rows])

    class FakeScores:
        def __init__(self, row_length):
            self.values = [float(index) for index in range(row_length)]

        def cpu(self):
            return self.values

    class FakeParameter:
        device = "mps"

    class FakeModel:
        def parameters(self):
            return iter([FakeParameter()])

        def get_scores(self, input_ids, _attention_mask):
            return [FakeScores(row_length) for row_length in input_ids.row_lengths]

    model = FakeModel()
    monkeypatch.setattr(kc, "_execution_semaphore", lambda *_args, **_kwargs: semaphore)
    monkeypatch.setattr(
        kc,
        "_load_kompress",
        lambda *args, **kwargs: (model, FakeTokenizer(), "pytorch"),
    )

    compressor = kc.KompressCompressor(kc.KompressConfig(min_input_words=10, enable_ccr=False))
    compressor.compress(" ".join(["word"] * 20), target_ratio=0.5)
    assert copy_saw_slot == [True, True]

    copy_saw_slot.clear()
    monkeypatch.setattr(
        kc.KompressCompressor,
        "_should_use_sequential_fallback",
        lambda self: False,
    )
    compressor.compress_batch(
        [" ".join(["word"] * 20), " ".join(["token"] * 20)],
        target_ratio=0.5,
    )
    assert copy_saw_slot == [True, True]


def test_canary_device_copy_holds_execution_slot(monkeypatch):
    semaphore = threading.BoundedSemaphore(1)
    copy_saw_slot: list[bool] = []

    class FakeTensor:
        def to(self, _device):
            acquired = semaphore.acquire(blocking=False)
            if acquired:
                semaphore.release()
            copy_saw_slot.append(not acquired)
            return self

    class FakeTokenizer:
        def __call__(self, *_args, **_kwargs):
            return {"input_ids": FakeTensor(), "attention_mask": FakeTensor()}

    class FakeParameter:
        device = "mps"

    class FakeModel:
        def parameters(self):
            return iter([FakeParameter()])

        def get_keep_mask(self, input_ids, attention_mask):
            return []

    monkeypatch.setattr(kc, "_execution_semaphore", lambda *_args, **_kwargs: semaphore)

    kc.KompressCompressor()._timed_canary(FakeModel(), FakeTokenizer(), "pytorch")

    assert copy_saw_slot == [True, True]
