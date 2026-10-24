from collections import deque
from types import SimpleNamespace

import pytest
from conftest import OTHER, SID, Log
from headroom_claude_mod.telemetry import (
    IncompatibleLogger,
    belongs,
    inspect,
    number,
    preview,
    record,
    snapshot,
    summarize,
)


@pytest.mark.parametrize("value", [None, True, False, "42", float("nan"), float("inf"), -1, 2**70])
def test_unknown_tokens_never_become_zero(value):
    assert number(value) is None


def test_signed_expansion_is_not_hidden():
    r = record(Log(input_tokens_original=100, input_tokens_optimized=120, tokens_saved=-20))
    assert r["percent"] == -20
    assert summarize([r])["saved"] == -20


@pytest.mark.parametrize("before,after,saved", [(100, 120, 0), (1000, 800, 150)])
def test_proxy_clamping_and_replay_debt_remain_accounted(before, after, saved):
    r = record(Log(input_tokens_original=before, input_tokens_optimized=after, tokens_saved=saved))
    assert r["accounting"] == "complete"
    assert r["percent"] == round(saved / before * 100, 2)
    totals = summarize([r])
    assert totals["accounted_requests"] == 1
    assert (totals["before"], totals["after"], totals["saved"]) == (before, after, saved)


def test_consistent_weighted_not_average_percent():
    a = record(Log(input_tokens_original=1000, input_tokens_optimized=900, tokens_saved=100))
    b = record(Log(input_tokens_original=100, input_tokens_optimized=10, tokens_saved=90))
    s = summarize([a, b])
    assert s["saved"] == 190
    assert s["percent"] == 17.27
    assert s["percent"] != 50


def test_errors_and_inconsistent_records_excluded():
    valid = record(Log())
    failed = record(Log(error="secret error"))
    wrong = record(Log(tokens_saved=999))
    missing = record(Log(input_tokens_original=None))
    s = summarize([valid, failed, wrong, missing])
    assert s["requests"] == 4
    assert s["accounted_requests"] == 1
    assert s["unaccounted_requests"] == 3
    assert s["saved"] == 400
    assert summarize([])["saved"] is None


def test_metadata_excludes_payload_credentials_and_arbitrary_tags():
    entry = Log(tags={"mod-session": SID, "authorization": "TOP_SECRET"})
    r = record(entry)
    assert "TOP_SECRET" not in str(r)
    assert "original tool output" not in str(r)
    assert "MUST_NOT_BE_SERVED" not in str(r)
    assert not {"request_messages", "compressed_messages", "tags", "error"} & r.keys()


def test_transform_counts_not_double_added():
    r = record(Log(transforms_applied=["smart_crusher", "smart_crusher", "router"]))
    s = summarize([r])
    assert s["transforms"] == {"router": 1, "smart_crusher": 1}
    assert s["saved"] == 400


def test_only_dedicated_correlation_tag_matches():
    assert belongs(Log(), SID)
    assert not belongs(Log(), OTHER)
    assert not belongs(Log(tags={"session-id": SID}), SID)


def test_snapshot_copies_references_not_heavy_payloads():
    class Uncopyable(list):
        def __deepcopy__(self, memo):
            raise AssertionError("Message bodies must not be copied during polling")

    log = Log(request_messages=Uncopyable([{"content": "x" * 1_000_000}]))
    logger = SimpleNamespace(_logs=deque([log], maxlen=10000))
    entries = snapshot(logger)
    assert entries[0] is log
    assert record(entries[0])["saved"] == 400


@pytest.mark.parametrize("logs", [[], deque(), deque(maxlen=10001), deque([{}], maxlen=100)])
def test_adapter_fail_closed_for_unknown_logger(logs):
    with pytest.raises(IncompatibleLogger):
        snapshot(SimpleNamespace(_logs=logs))


def test_inspection_paginates_without_mutation():
    log = Log(compressed_messages=[{"content": "x" * 5000}])
    first = inspect(log, side="compressed", message=0, page=0)
    last = inspect(log, side="compressed", message=0, page=99999)
    assert len(first["text"]) <= 1400
    assert first["pages"] > 1
    assert last["page"] == last["pages"] - 1
    assert log.compressed_messages[0]["content"] == "x" * 5000


def test_deleted_messages_not_falsely_index_paired():
    log = Log(
        request_messages=[{"content": "removed"}, {"content": "kept"}],
        compressed_messages=[{"content": "kept"}],
    )
    d = inspect(log, side="diff", message=0, page=0)
    assert '-    "content": "removed"' in d["text"]
    assert d["side"] == "diff"
    assert d["counts"] == {"original": 2, "compressed": 1}


def test_expired_capture_and_wrong_message_index_are_explicit():
    assert not inspect(Log(request_messages=None), side="original", message=0, page=0)["available"]
    assert not inspect(Log(), side="compressed", message=100, page=0)["available"]


def test_huge_recursive_and_terminal_payloads_bounded():
    cyclic = []
    cyclic.append(cyclic)
    body, truncated = preview(
        {"cyclic": cyclic, "huge": "x" * 1_000_000, "control": "\x1b[2J\u202e"}
    )
    assert truncated
    assert len(body) <= 32800
    assert "\x1b" not in body and "\u202e" not in body
