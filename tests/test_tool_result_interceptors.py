"""Tests for the tool_result interceptor framework + ast-grep Read outliner."""

from __future__ import annotations

import asyncio
import os
import sys
import textwrap

import pytest

from headroom.proxy.interceptors import (
    INTERCEPTORS,
    ToolResultInterceptor,
    ToolResultInterceptorTransform,
    apply_to_messages,
    interceptor_failure_counts,
    register,
)
from headroom.proxy.interceptors.astgrep import (
    AstGrepReadOutline,
    ReadVerificationResult,
    _open_regular_file_under_root,
    _verify_read_against_disk,
)
from headroom.proxy.interceptors.base import reset_interceptor_failure_counts
from headroom.proxy.project_context import get_registered_cwd, set_registered_cwd
from headroom.tokenizer import Tokenizer


class _FakeTokenCounter:
    """Deterministic 4-chars-per-token counter for unit tests."""

    def count_text(self, text: str) -> int:
        return max(1, len(text) // 4)

    def count_messages(self, messages) -> int:
        total = 0
        for m in messages:
            c = m.get("content")
            if isinstance(c, str):
                total += self.count_text(c)
            elif isinstance(c, list):
                for b in c:
                    if isinstance(b, dict):
                        inner = b.get("content") or b.get("text") or ""
                        if isinstance(inner, str):
                            total += self.count_text(inner)
        return total


@pytest.fixture
def tokenizer() -> Tokenizer:
    # Real Tokenizer wrapping the fake counter; mirrors production construction.
    return Tokenizer(_FakeTokenCounter())  # type: ignore[arg-type]


# -------- Framework basics ----------------------------------------------- #


def test_astgrep_interceptor_registered_by_default():
    assert any(i.name == "ast-grep" for i in INTERCEPTORS)


def test_register_is_idempotent_on_name():
    before = len(INTERCEPTORS)
    register(AstGrepReadOutline())  # same name
    assert len(INTERCEPTORS) == before


def test_custom_interceptor_plugs_in(tokenizer):
    class UpperCase:
        name = "uppercase-test"

        def matches(self, tool_name, tool_input, tool_output):
            return tool_name == "Echo"

        def transform(self, tool_name, tool_input, tool_output):
            # Must REDUCE tokens — use a single short marker.
            return "X"

    dummy: ToolResultInterceptor = UpperCase()  # type: ignore[assignment]
    register(dummy)
    try:
        messages = [
            {
                "role": "assistant",
                "content": [{"type": "tool_use", "id": "1", "name": "Echo", "input": {}}],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "1",
                        "content": "hello " * 100,
                    }
                ],
            },
        ]
        result = apply_to_messages(messages, tokenizer)
        assert any(s.tool == "uppercase-test" for s in result.spans)
        swapped = result.messages[1]["content"][0]["content"]
        assert swapped == "X"
    finally:
        INTERCEPTORS[:] = [i for i in INTERCEPTORS if i.name != "uppercase-test"]


def test_pass_through_when_no_interceptor_matches(tokenizer):
    messages = [
        {
            "role": "assistant",
            "content": [{"type": "tool_use", "id": "1", "name": "Unknown", "input": {}}],
        },
        {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": "1", "content": "x" * 5000}],
        },
    ]
    result = apply_to_messages(messages, tokenizer)
    assert result.spans == []
    assert result.messages[1] is messages[1]  # untouched identity


# -------- ast-grep interceptor ------------------------------------------- #


_PY_FIXTURE = textwrap.dedent(
    '''
    """Payments module fixture."""
    from decimal import Decimal

    def compute_subtotal(items):
        total = Decimal("0")
        for item in items:
            total += item.price * item.qty
        return total


    def apply_promo(subtotal, code):
        if not code:
            return subtotal
        if code == "SAVE10":
            return subtotal * Decimal("0.9")
        return subtotal


    def compute_tax(subtotal, rate):
        return (subtotal * rate).quantize(Decimal("0.01"))


    def process_payment(items, promo, tax_rate):
        """Main entry point."""
        subtotal = compute_subtotal(items)
        after = apply_promo(subtotal, promo)
        tax = compute_tax(after, tax_rate)
        return after + tax


    def refund(order_id, amount):
        """Issue a refund."""
        return {"order": order_id, "refund": str(amount)}


    def list_orders_for_user(user_id, limit=20):
        """Placeholder DB lookup for a user's orders."""
        return [{"user": user_id, "order": i} for i in range(limit)]


    def cancel_order(order_id, reason=None):
        """Cancel an order, logging the reason if provided."""
        return {"order": order_id, "cancelled": True, "reason": reason or "unspecified"}


    def summarize_cart(items):
        """Return a one-line summary of cart contents."""
        skus = [i.sku for i in items]
        total_qty = sum(i.qty for i in items)
        return f"{len(items)} line items ({total_qty} units): {', '.join(skus)}"


    def format_receipt(order_id, items, total):
        """Render a textual receipt."""
        lines = [f"Order {order_id}"]
        for i in items:
            lines.append(f"  {i.sku} x {i.qty} @ {i.unit_price} = {i.qty * i.unit_price}")
        lines.append(f"Total: {total}")
        return "\\n".join(lines)
    '''
).strip()


def test_astgrep_outlines_large_python_read(tokenizer):
    messages = [
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": "abc",
                    "name": "Read",
                    "input": {"file_path": "/repo/payments.py"},
                }
            ],
        },
        {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": "abc", "content": _PY_FIXTURE}],
        },
    ]
    result = apply_to_messages(messages, tokenizer)
    assert len(result.spans) == 1
    span = result.spans[0]
    assert span.tool == "ast-grep"
    assert span.tokens_after < span.tokens_before
    new_content = result.messages[1]["content"][0]["content"]
    assert "outlined by ast-grep" in new_content
    assert "body elided" in new_content
    assert "def process_payment" in new_content
    assert "def apply_promo" in new_content
    # Bodies should NOT leak through unchanged.
    assert "total += item.price * item.qty" not in new_content
    # Complete-file control: no truncation banner in the input -> no truncation marker.
    assert "truncated upstream" not in new_content


def test_astgrep_flags_truncated_read(tokenizer):
    truncated_source = (
        _PY_FIXTURE + "\n\n[Truncated: PARTIAL view — /repo/payments.py: "
        "showing lines 1-42 of 90 total (26031 tokens, cap 25000). "
        "Call Read with offset=43 to see more.]\n"
    )
    messages = [
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": "abc",
                    "name": "Read",
                    "input": {"file_path": "/repo/payments.py"},
                }
            ],
        },
        {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": "abc", "content": truncated_source}],
        },
    ]
    result = apply_to_messages(messages, tokenizer)
    assert len(result.spans) == 1
    new_content = result.messages[1]["content"][0]["content"]
    assert "truncated upstream" in new_content
    assert "showing through line 42 of 90 total" in new_content
    # Still lists the definitions actually present in the visible portion.
    assert "def process_payment" in new_content
    assert "def apply_promo" in new_content


def _read_result_messages(content: str, file_path: str = "/repo/payments.py"):
    return [
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": "abc",
                    "name": "Read",
                    "input": {"file_path": file_path},
                }
            ],
        },
        {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": "abc", "content": content}],
        },
    ]


def test_astgrep_flags_truncated_read_wording_variants(tokenizer):
    """The signature tolerates wording/casing/dash variation, but only inside
    the recognized envelope — not as a synonym match over arbitrary prose."""
    truncated_source = (
        _PY_FIXTURE + "\n\n[TRUNCATED:   PARTIAL VIEW — /repo/payments.py: "
        "SHOWING  LINES 1–42  of 90  TOTAL. Call Read with offset=43 to see more.]\n"
    )
    messages = _read_result_messages(truncated_source)
    result = apply_to_messages(messages, tokenizer)
    assert len(result.spans) == 1
    new_content = result.messages[1]["content"][0]["content"]
    assert "truncated upstream" in new_content
    assert "showing through line 42 of 90 total" in new_content


def test_astgrep_ignores_truncation_phrase_in_comment(tokenizer):
    """A count-shaped phrase in a plain comment, with no bracketed envelope,
    must not be read as an upstream truncation claim."""
    source_with_comment = (
        _PY_FIXTURE + "\n\n# API pagination showing lines 10-20 of 30 total records\n"
    )
    messages = _read_result_messages(source_with_comment)
    result = apply_to_messages(messages, tokenizer)
    new_content = result.messages[1]["content"][0]["content"]
    assert "truncated upstream" not in new_content


def test_astgrep_ignores_bracketed_phrase_without_signature(tokenizer):
    """Brackets plus a count-shaped phrase aren't enough on their own — the
    exact recognized signature phrase must also be present."""
    source_with_bracket = (
        _PY_FIXTURE + "\n\n[Truncation happened; showing lines 1-42 of 90 total]\n"
    )
    messages = _read_result_messages(source_with_bracket)
    result = apply_to_messages(messages, tokenizer)
    new_content = result.messages[1]["content"][0]["content"]
    assert "truncated upstream" not in new_content


@pytest.mark.parametrize(
    "banner_numbers",
    [
        pytest.param("50-90 of 90", id="end_equals_total"),
        pytest.param("42-10 of 90", id="end_less_than_start"),
        pytest.param("0-42 of 90", id="start_is_zero"),
        pytest.param("1-5000 of 9000", id="end_exceeds_visible_payload"),
    ],
)
def test_astgrep_ignores_malformed_truncation_counts(tokenizer, banner_numbers):
    truncated_source = (
        _PY_FIXTURE + "\n\n[Truncated: PARTIAL view — /repo/payments.py: "
        f"showing lines {banner_numbers} total (26031 tokens, cap 25000). "
        "Call Read with offset=43 to see more.]\n"
    )
    messages = _read_result_messages(truncated_source)
    result = apply_to_messages(messages, tokenizer)
    new_content = result.messages[1]["content"][0]["content"]
    assert "outlined by ast-grep" in new_content
    assert "truncated upstream" not in new_content


def test_astgrep_accepts_truncation_at_start_equals_end(tokenizer):
    """`end >= start` is inclusive — a single-line visible window is valid."""
    truncated_source = (
        _PY_FIXTURE + "\n\n[Truncated: PARTIAL view — /repo/payments.py: "
        "showing lines 42-42 of 90 total (26031 tokens, cap 25000). "
        "Call Read with offset=43 to see more.]\n"
    )
    messages = _read_result_messages(truncated_source)
    result = apply_to_messages(messages, tokenizer)
    new_content = result.messages[1]["content"][0]["content"]
    assert "truncated upstream" in new_content


# -------- Disk verification: client-independent truncation fallback ----- #


def test_set_get_registered_cwd_round_trips():
    set_registered_cwd("/repo/project")
    try:
        assert get_registered_cwd() == "/repo/project"
    finally:
        set_registered_cwd(None)


class TestVerifyReadAgainstDisk:
    def test_missing_file_path_is_unknown(self):
        verdict, info = _verify_read_against_disk(None, "abc", "/repo")
        assert verdict is ReadVerificationResult.UNKNOWN
        assert info is None

    def test_no_registered_cwd_is_unknown(self):
        verdict, info = _verify_read_against_disk("payments.py", "abc", None)
        assert verdict is ReadVerificationResult.UNKNOWN
        assert info is None

    def test_untrusted_request_never_touches_disk(self, tmp_path, monkeypatch):
        """A valid file/content combo must short-circuit to UNKNOWN -- and
        never call os.open -- when no root is registered."""
        f = tmp_path / "payments.py"
        f.write_text(_PY_FIXTURE, encoding="utf-8")
        opened: list[object] = []
        real_open = os.open

        def _tracking_open(*args, **kwargs):
            opened.append(args)
            return real_open(*args, **kwargs)

        monkeypatch.setattr(os, "open", _tracking_open)
        verdict, info = _verify_read_against_disk(str(f), _PY_FIXTURE, None)
        assert verdict is ReadVerificationResult.UNKNOWN
        assert info is None
        assert opened == []

    def test_missing_file_under_workspace_root_is_unknown(self, tmp_path):
        verdict, info = _verify_read_against_disk(
            str(tmp_path / "missing.py"), "abc", str(tmp_path)
        )
        assert verdict is ReadVerificationResult.UNKNOWN
        assert info is None

    @pytest.mark.skipif(
        sys.platform == "win32",
        reason="dir_fd disk verification unavailable on Windows; falls back to UNKNOWN"
        " (see test_open_regular_file_under_root_falls_back_unknown_when_dir_fd_unsupported)",
    )
    def test_exact_match_is_complete(self, tmp_path):
        f = tmp_path / "payments.py"
        f.write_text(_PY_FIXTURE, encoding="utf-8")
        verdict, info = _verify_read_against_disk(str(f), _PY_FIXTURE, str(tmp_path))
        assert verdict is ReadVerificationResult.COMPLETE
        assert info is None

    @pytest.mark.skipif(
        sys.platform == "win32",
        reason="dir_fd disk verification unavailable on Windows; falls back to UNKNOWN"
        " (see test_open_regular_file_under_root_falls_back_unknown_when_dir_fd_unsupported)",
    )
    def test_strict_prefix_is_truncated(self, tmp_path):
        f = tmp_path / "payments.py"
        f.write_text(_PY_FIXTURE, encoding="utf-8")
        partial = _PY_FIXTURE[:200]
        verdict, info = _verify_read_against_disk(str(f), partial, str(tmp_path))
        assert verdict is ReadVerificationResult.TRUNCATED
        assert info == (len(partial.splitlines()), len(_PY_FIXTURE.splitlines()))

    @pytest.mark.skipif(
        sys.platform == "win32",
        reason="dir_fd disk verification unavailable on Windows; falls back to UNKNOWN"
        " (see test_open_regular_file_under_root_falls_back_unknown_when_dir_fd_unsupported)",
    )
    def test_relative_path_resolves_against_registered_cwd(self, tmp_path):
        f = tmp_path / "payments.py"
        f.write_text(_PY_FIXTURE, encoding="utf-8")
        partial = _PY_FIXTURE[:200]
        verdict, info = _verify_read_against_disk("payments.py", partial, str(tmp_path))
        assert verdict is ReadVerificationResult.TRUNCATED
        assert info is not None

    def test_content_mismatch_is_unknown_not_truncated(self, tmp_path):
        # File diverged since the client read it -- not a clean prefix.
        f = tmp_path / "payments.py"
        f.write_text(_PY_FIXTURE.replace("compute_subtotal", "compute_total"), encoding="utf-8")
        verdict, info = _verify_read_against_disk(str(f), _PY_FIXTURE, str(tmp_path))
        assert verdict is ReadVerificationResult.UNKNOWN
        assert info is None

    def test_absolute_path_outside_workspace_root_is_unknown(self, tmp_path):
        root = tmp_path / "project"
        root.mkdir()
        sibling = tmp_path / "other"
        sibling.mkdir()
        f = sibling / "secret.py"
        f.write_text(_PY_FIXTURE, encoding="utf-8")
        verdict, info = _verify_read_against_disk(str(f), _PY_FIXTURE, str(root))
        assert verdict is ReadVerificationResult.UNKNOWN
        assert info is None

    def test_relative_traversal_escapes_workspace_is_unknown(self, tmp_path):
        root = tmp_path / "project"
        root.mkdir()
        f = tmp_path / "secret.py"
        f.write_text(_PY_FIXTURE, encoding="utf-8")
        verdict, info = _verify_read_against_disk("../secret.py", _PY_FIXTURE, str(root))
        assert verdict is ReadVerificationResult.UNKNOWN
        assert info is None

    def test_sibling_directory_sharing_string_prefix_is_not_inside_workspace(self, tmp_path):
        """Containment is a path-segment check (relpath/segment split), not
        a string-prefix check -- a target that merely starts with the
        root's string must not be treated as inside it."""
        root = tmp_path / "project"
        root.mkdir()
        other = tmp_path / "project-other"
        other.mkdir()
        f = other / "secret.py"
        f.write_text(_PY_FIXTURE, encoding="utf-8")
        verdict, info = _verify_read_against_disk(str(f), _PY_FIXTURE, str(root))
        assert verdict is ReadVerificationResult.UNKNOWN
        assert info is None

    def test_directory_passed_as_file_path_is_unknown(self, tmp_path):
        root = tmp_path / "project"
        subdir = root / "subdir"
        subdir.mkdir(parents=True)
        verdict, info = _verify_read_against_disk("subdir", "abc", str(root))
        assert verdict is ReadVerificationResult.UNKNOWN
        assert info is None

    def test_oversized_file_exceeding_byte_cap_is_unknown(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HEADROOM_VERIFY_TRUNCATION_MAX_BYTES", "10")
        f = tmp_path / "payments.py"
        f.write_text(_PY_FIXTURE, encoding="utf-8")
        assert len(_PY_FIXTURE.encode("utf-8")) > 10
        # Content matches exactly -- would be COMPLETE without the cap.
        verdict, info = _verify_read_against_disk(str(f), _PY_FIXTURE, str(tmp_path))
        assert verdict is ReadVerificationResult.UNKNOWN
        assert info is None


# -------- _open_regular_file_under_root: race-safety + off-loop (review round 2) --------- #


@pytest.mark.skipif(sys.platform == "win32", reason="symlinks need elevated privilege on Windows")
def test_open_regular_file_under_root_rejects_symlinked_final_file(tmp_path):
    """A symlinked leaf file must be rejected by O_NOFOLLOW, not followed.

    A static symlink is sufficient race evidence here: O_NOFOLLOW rejects it
    the same way whether it's always been there or appeared 2ms ago -- a
    concurrent-race harness would prove nothing more and would be flaky.
    """
    root = tmp_path / "project"
    root.mkdir()
    outside = tmp_path / "outside.py"
    outside.write_text(_PY_FIXTURE, encoding="utf-8")
    link = root / "link.py"
    link.symlink_to(outside)
    assert _open_regular_file_under_root("link.py", root, 10_000) is None


@pytest.mark.skipif(sys.platform == "win32", reason="symlinks need elevated privilege on Windows")
def test_open_regular_file_under_root_rejects_symlinked_intermediate_directory(tmp_path):
    """A symlinked directory component must be rejected at that hop, not
    walked into."""
    root = tmp_path / "project"
    root.mkdir()
    real_dir = tmp_path / "realdir"
    real_dir.mkdir()
    (real_dir / "secret.py").write_text(_PY_FIXTURE, encoding="utf-8")
    link_dir = root / "linkdir"
    link_dir.symlink_to(real_dir)
    assert _open_regular_file_under_root("linkdir/secret.py", root, 10_000) is None


@pytest.mark.skipif(sys.platform == "win32", reason="symlinks need elevated privilege on Windows")
def test_open_regular_file_under_root_rejects_symlinked_root_anchor(tmp_path):
    """The root anchor itself must be opened O_NOFOLLOW too, not just the
    segments below it -- otherwise a symlink swapped in at the resolved root
    path would be followed, the exact TOCTOU shape review round 2 flagged,
    just moved from the leaf to the anchor."""
    real_root = tmp_path / "real-project"
    real_root.mkdir()
    (real_root / "x.py").write_text("content", encoding="utf-8")
    root_link = tmp_path / "project-link"
    root_link.symlink_to(real_root)
    assert _open_regular_file_under_root("x.py", root_link, 10_000) is None


def test_open_regular_file_under_root_refuses_on_event_loop_thread(tmp_path):
    """Calling from inside a running event loop must refuse immediately
    rather than perform the read."""
    f = tmp_path / "x.py"
    f.write_text("content", encoding="utf-8")

    async def _call_from_loop():
        return _open_regular_file_under_root(str(f), tmp_path, 10_000)

    assert asyncio.run(_call_from_loop()) is None


@pytest.mark.skipif(
    sys.platform == "win32",
    reason="dir_fd disk verification unavailable on Windows; falls back to UNKNOWN"
    " (see test_open_regular_file_under_root_falls_back_unknown_when_dir_fd_unsupported)",
)
def test_open_regular_file_under_root_proceeds_from_plain_sync_context(tmp_path):
    """Sanity check: with no running loop, the read must still succeed."""
    f = tmp_path / "x.py"
    f.write_text("content", encoding="utf-8")
    assert _open_regular_file_under_root(str(f), tmp_path, 10_000) == "content"


def test_open_regular_file_under_root_falls_back_unknown_when_dir_fd_unsupported(
    tmp_path, monkeypatch
):
    """Fails closed (None) rather than falling back to a less-safe pattern
    when dir_fd-relative opens aren't supported (notably Windows)."""
    import headroom.proxy.interceptors.astgrep as astgrep_module

    f = tmp_path / "x.py"
    f.write_text("content", encoding="utf-8")
    monkeypatch.setattr(astgrep_module, "_dir_fd_walk_supported", lambda: False)
    assert _open_regular_file_under_root(str(f), tmp_path, 10_000) is None


def test_open_regular_file_under_root_directory_passed_as_target_is_unknown(tmp_path):
    """A directory at the leaf position must be rejected -- the target must
    be a file."""
    (tmp_path / "subdir").mkdir()
    assert _open_regular_file_under_root("subdir", tmp_path, 10_000) is None


@pytest.mark.skipif(sys.platform == "win32", reason="os.mkfifo unavailable on Windows")
def test_open_regular_file_under_root_rejects_fifo_without_hanging(tmp_path):
    """A FIFO must be rejected by the S_ISREG check, and -- because the open
    uses O_NONBLOCK -- must return promptly rather than blocking on a
    writer that will never arrive. If this test hangs, O_NONBLOCK regressed."""
    fifo_path = tmp_path / "pipe"
    os.mkfifo(fifo_path)
    assert _open_regular_file_under_root("pipe", tmp_path, 10_000) is None


# -------- End-to-end: apply_to_messages, real registered_cwd --------- #


def _disk_verification_messages(partial: str) -> list[dict]:
    return [
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": "abc",
                    "name": "Read",
                    "input": {"file_path": "payments.py"},
                }
            ],
        },
        {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": "abc", "content": partial}],
        },
    ]


@pytest.mark.skipif(
    sys.platform == "win32",
    reason="dir_fd disk verification unavailable on Windows; falls back to UNKNOWN"
    " (see test_astgrep_disk_verification_stays_unknown_when_dir_fd_unsupported)",
)
def test_astgrep_disk_verification_flags_truncation_when_no_banner(
    tokenizer, tmp_path, monkeypatch
):
    """Opted in, root registered, real on-disk prefix, no banner -- disk
    verification alone qualifies the header."""
    monkeypatch.setenv("HEADROOM_VERIFY_TRUNCATION_ON_DISK", "1")
    monkeypatch.setenv("HEADROOM_INTERCEPT_READ_MIN_CHARS", "50")
    f = tmp_path / "payments.py"
    f.write_text(_PY_FIXTURE, encoding="utf-8")
    marker = "\n\ndef format_receipt"
    partial = _PY_FIXTURE[: _PY_FIXTURE.index(marker)]  # no banner text anywhere
    assert "truncated" not in partial.lower()

    set_registered_cwd(str(tmp_path))
    try:
        result = apply_to_messages(_disk_verification_messages(partial), tokenizer)
    finally:
        set_registered_cwd(None)

    assert len(result.spans) == 1
    new_content = result.messages[1]["content"][0]["content"]
    assert "truncated upstream" in new_content
    visible_lines = len(partial.splitlines())
    total_lines = len(_PY_FIXTURE.splitlines())
    assert f"showing through line {visible_lines} of {total_lines} total" in new_content
    assert "def compute_subtotal" in new_content
    assert "def apply_promo" in new_content
    assert "def format_receipt" not in new_content


def test_astgrep_disk_verification_stays_unknown_when_dir_fd_unsupported(
    tokenizer, tmp_path, monkeypatch
):
    """Fail-closed contract, simulated cross-platform: dir_fd unsupported ->
    UNKNOWN -> no false "truncated upstream" claim. Stands in for the real
    Windows behavior the skipif'd tests above can't exercise here."""
    import headroom.proxy.interceptors.astgrep as astgrep_module

    monkeypatch.setattr(astgrep_module, "_dir_fd_walk_supported", lambda: False)
    monkeypatch.setenv("HEADROOM_VERIFY_TRUNCATION_ON_DISK", "1")
    monkeypatch.setenv("HEADROOM_INTERCEPT_READ_MIN_CHARS", "50")
    f = tmp_path / "payments.py"
    f.write_text(_PY_FIXTURE, encoding="utf-8")
    marker = "\n\ndef format_receipt"
    partial = _PY_FIXTURE[: _PY_FIXTURE.index(marker)]  # no banner text anywhere
    assert "truncated" not in partial.lower()

    set_registered_cwd(str(tmp_path))
    try:
        result = apply_to_messages(_disk_verification_messages(partial), tokenizer)
    finally:
        set_registered_cwd(None)

    # The outline rewrite is independent of the disk-verify verdict.
    assert len(result.spans) == 1
    new_content = result.messages[1]["content"][0]["content"]
    assert "truncated upstream" not in new_content


def test_astgrep_disk_verification_skips_when_no_root_registered(tokenizer, tmp_path, monkeypatch):
    """No registered root at all (the default) -- opted in, real on-disk
    prefix, but never flags. Spoofed-header-doesn't-work is covered
    end-to-end in test_proxy_workspace_registration_middleware.py."""
    monkeypatch.setenv("HEADROOM_VERIFY_TRUNCATION_ON_DISK", "1")
    monkeypatch.setenv("HEADROOM_INTERCEPT_READ_MIN_CHARS", "50")
    f = tmp_path / "payments.py"
    f.write_text(_PY_FIXTURE, encoding="utf-8")
    marker = "\n\ndef format_receipt"
    partial = _PY_FIXTURE[: _PY_FIXTURE.index(marker)]

    messages = [
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": "abc",
                    "name": "Read",
                    "input": {"file_path": "payments.py"},
                }
            ],
        },
        {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": "abc", "content": partial}],
        },
    ]
    result = apply_to_messages(messages, tokenizer)

    new_content = result.messages[1]["content"][0]["content"]
    assert "truncated upstream" not in new_content


def test_astgrep_banner_detection_takes_priority_over_disk_verification(tokenizer, monkeypatch):
    """When a banner is already present, disk verification must not run at
    all (no cwd bound here -- if it ran, resolution would fail anyway), and
    the banner's own numbers must be what the header reports."""
    monkeypatch.setenv("HEADROOM_VERIFY_TRUNCATION_ON_DISK", "1")
    truncated_source = (
        _PY_FIXTURE + "\n\n[Truncated: PARTIAL view — /repo/payments.py: "
        "showing lines 1-42 of 90 total (26031 tokens, cap 25000). "
        "Call Read with offset=43 to see more.]\n"
    )
    messages = [
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": "abc",
                    "name": "Read",
                    "input": {"file_path": "/repo/payments.py"},
                }
            ],
        },
        {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": "abc", "content": truncated_source}],
        },
    ]
    result = apply_to_messages(messages, tokenizer)
    new_content = result.messages[1]["content"][0]["content"]
    assert "showing through line 42 of 90 total" in new_content


def test_astgrep_skips_small_files(tokenizer):
    small = "def foo(): return 1\n"
    messages = [
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": "x",
                    "name": "Read",
                    "input": {"file_path": "/a.py"},
                }
            ],
        },
        {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": "x", "content": small}],
        },
    ]
    result = apply_to_messages(messages, tokenizer)
    assert result.spans == []


def test_astgrep_skips_non_code_extensions(tokenizer):
    messages = [
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": "r",
                    "name": "Read",
                    "input": {"file_path": "/notes.txt"},
                }
            ],
        },
        {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": "r", "content": "x" * 3000}],
        },
    ]
    result = apply_to_messages(messages, tokenizer)
    assert result.spans == []


# -------- OpenAI-format tool_result -------------------------------------- #


def test_astgrep_skips_when_line_range_requested(tokenizer):
    """If the tool_input specifies a line range, the model wants those lines — pass through."""
    messages = [
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": "r",
                    "name": "Read",
                    "input": {
                        "file_path": "/repo/payments.py",
                        "offset": 30,
                        "limit": 20,
                    },
                }
            ],
        },
        {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": "r", "content": _PY_FIXTURE}],
        },
    ]
    result = apply_to_messages(messages, tokenizer)
    assert result.spans == []


def test_progressive_disclosure_second_read_passes_through(tokenizer):
    """First Read of a file gets outlined; second Read of the same path is untouched."""
    messages = [
        # Turn 1: Read foo.py → outlined
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": "t1",
                    "name": "Read",
                    "input": {"file_path": "/repo/payments.py"},
                }
            ],
        },
        {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": "t1", "content": _PY_FIXTURE}],
        },
        # Turn 2: Read foo.py again (model came back for more) → pass through
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": "t2",
                    "name": "Read",
                    "input": {"file_path": "/repo/payments.py"},
                }
            ],
        },
        {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": "t2", "content": _PY_FIXTURE}],
        },
    ]
    result = apply_to_messages(messages, tokenizer)
    # Only the first Read is rewritten; the second keeps its full body.
    assert len(result.spans) == 1
    first_tr = result.messages[1]["content"][0]["content"]
    second_tr = result.messages[3]["content"][0]["content"]
    assert "outlined by ast-grep" in first_tr
    assert "outlined by ast-grep" not in second_tr
    assert "def process_payment" in second_tr
    # Second Read preserves the bodies.
    assert "subtotal = compute_subtotal(items)" in second_tr


def test_progressive_disclosure_different_file_still_outlined(tokenizer):
    """Reading a DIFFERENT file after the first outline should still outline."""
    messages = [
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": "t1",
                    "name": "Read",
                    "input": {"file_path": "/repo/payments.py"},
                }
            ],
        },
        {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": "t1", "content": _PY_FIXTURE}],
        },
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": "t2",
                    "name": "Read",
                    "input": {"file_path": "/repo/other.py"},
                }
            ],
        },
        {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": "t2", "content": _PY_FIXTURE}],
        },
    ]
    result = apply_to_messages(messages, tokenizer)
    # Both files get outlined — different keys.
    assert len(result.spans) == 2


def test_openai_format_tool_result_is_rewritten(tokenizer):
    messages = [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "Read",
                        "arguments": '{"file_path": "/x/payments.py"}',
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_1",
            "content": _PY_FIXTURE,
        },
    ]
    result = apply_to_messages(messages, tokenizer)
    assert len(result.spans) == 1
    new_content = result.messages[1]["content"]
    assert "outlined by ast-grep" in new_content


# -------- Failure isolation & safety guarantees -------------------------- #


def test_failing_interceptor_does_not_crash_request(tokenizer):
    """If transform() raises, the request still succeeds unchanged."""
    reset_interceptor_failure_counts()

    class BoomInterceptor:
        name = "boom"

        def matches(self, tool_name, tool_input, tool_output):
            return tool_name == "Read"

        def transform(self, tool_name, tool_input, tool_output):
            raise RuntimeError("simulated interceptor bug")

    register(BoomInterceptor())
    try:
        messages = [
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "b",
                        "name": "Read",
                        "input": {"file_path": "/repo/payments.py"},
                    }
                ],
            },
            {
                "role": "user",
                "content": [{"type": "tool_result", "tool_use_id": "b", "content": _PY_FIXTURE}],
            },
        ]
        result = apply_to_messages(messages, tokenizer)
        # No span recorded for boom; request survives.
        assert not any(s.tool == "boom" for s in result.spans)
        # The failure counter incremented.
        assert interceptor_failure_counts().get("boom") == 1
    finally:
        INTERCEPTORS[:] = [i for i in INTERCEPTORS if i.name != "boom"]


def test_failing_key_skips_interceptor_entirely(tokenizer):
    """Broken progressive_disclosure_key() must skip, not fire without a key."""
    reset_interceptor_failure_counts()
    fire_count = {"n": 0}

    class BadKey:
        name = "bad-key"

        def matches(self, tool_name, tool_input, tool_output):
            return tool_name == "Read"

        def transform(self, tool_name, tool_input, tool_output):
            fire_count["n"] += 1
            return "X"  # reduces tokens

        def progressive_disclosure_key(self, tool_name, tool_input):
            raise RuntimeError("cannot compute key")

    register(BadKey())
    try:
        messages = [
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "k",
                        "name": "Read",
                        "input": {"file_path": "/repo/payments.py"},
                    }
                ],
            },
            {
                "role": "user",
                "content": [{"type": "tool_result", "tool_use_id": "k", "content": _PY_FIXTURE}],
            },
        ]
        apply_to_messages(messages, tokenizer)
        assert fire_count["n"] == 0  # transform never ran
        assert interceptor_failure_counts().get("bad-key") == 1
    finally:
        INTERCEPTORS[:] = [i for i in INTERCEPTORS if i.name != "bad-key"]


def test_refuses_to_enlarge(tokenizer):
    """If rewrite has MORE tokens than original, pass through unchanged.

    Uses a non-code tool path so only the Inflater runs (ast-grep passes
    through on non-Read tools).
    """
    original_content = "some data " * 200

    class Inflater:
        name = "inflater"

        def matches(self, tool_name, tool_input, tool_output):
            return tool_name == "FetchPage"

        def transform(self, tool_name, tool_input, tool_output):
            return tool_output + (" padding" * 200)

    register(Inflater())
    try:
        messages = [
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "i",
                        "name": "FetchPage",
                        "input": {"url": "https://example.com"},
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "i", "content": original_content}
                ],
            },
        ]
        result = apply_to_messages(messages, tokenizer)
        assert not any(s.tool == "inflater" for s in result.spans)
        # Original content preserved.
        assert result.messages[1]["content"][0]["content"] == original_content
    finally:
        INTERCEPTORS[:] = [i for i in INTERCEPTORS if i.name != "inflater"]


def test_orphaned_tool_result_does_not_crash(tokenizer):
    """A tool_result with no matching tool_use still runs safely (no tool_name)."""
    messages = [
        # No tool_use block — the model's prior turn is missing.
        {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "orphan-id", "content": _PY_FIXTURE}
            ],
        },
    ]
    result = apply_to_messages(messages, tokenizer)
    # ast-grep.matches() returns False when tool_name is None, so no span.
    assert result.spans == []
    # The orphan message is preserved.
    assert result.messages[0]["content"][0]["content"] == _PY_FIXTURE


# -------- Transform adapter tests ---------------------------------------- #


def test_transform_adapter_applies_interceptors(tokenizer):
    """ToolResultInterceptorTransform.apply() runs interceptors + records tokens."""
    transform = ToolResultInterceptorTransform()
    messages = [
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": "a",
                    "name": "Read",
                    "input": {"file_path": "/repo/payments.py"},
                }
            ],
        },
        {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": "a", "content": _PY_FIXTURE}],
        },
    ]
    result = transform.apply(messages, tokenizer)
    assert result.tokens_after < result.tokens_before
    assert "interceptor:ast-grep" in result.transforms_applied


def test_transform_adapter_respects_frozen_message_count(tokenizer):
    """Messages in the frozen prefix must be untouched to preserve prefix caches."""
    transform = ToolResultInterceptorTransform()
    messages = [
        # Frozen prefix (first tool_result) — MUST pass through unchanged.
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": "t1",
                    "name": "Read",
                    "input": {"file_path": "/repo/a.py"},
                }
            ],
        },
        {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": "t1", "content": _PY_FIXTURE}],
        },
        # Mutable tail (second Read of a different file) — free to outline.
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": "t2",
                    "name": "Read",
                    "input": {"file_path": "/repo/b.py"},
                }
            ],
        },
        {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": "t2", "content": _PY_FIXTURE}],
        },
    ]
    result = transform.apply(messages, tokenizer, frozen_message_count=2)
    # Frozen prefix identity preserved (exact same list refs).
    assert result.messages[0] is messages[0]
    assert result.messages[1] is messages[1]
    # Tail got outlined.
    assert "outlined by ast-grep" in result.messages[3]["content"][0]["content"]


def test_progressive_disclosure_respects_frozen_prefix_history(tokenizer):
    """If a file was Read in the frozen prefix, re-reading it in the mutable
    tail passes through — even though apply_to_messages only sees the tail
    for rewriting, it pre-scans the frozen prefix to seed `fired` keys.
    """
    transform = ToolResultInterceptorTransform()
    messages = [
        # Frozen prefix: first Read of payments.py. This is cached, so we
        # don't outline it; but it counts as "already disclosed."
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": "frozen-read",
                    "name": "Read",
                    "input": {"file_path": "/repo/payments.py"},
                }
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "frozen-read",
                    "content": _PY_FIXTURE,
                }
            ],
        },
        # Mutable tail: model reads payments.py again — should pass through
        # because the frozen prefix already served it.
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": "tail-read",
                    "name": "Read",
                    "input": {"file_path": "/repo/payments.py"},
                }
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "tail-read",
                    "content": _PY_FIXTURE,
                }
            ],
        },
    ]
    result = transform.apply(messages, tokenizer, frozen_message_count=2)
    # Tail re-read preserved (not outlined) because the frozen prefix
    # already exposed the file.
    tail_content = result.messages[3]["content"][0]["content"]
    assert "outlined by ast-grep" not in tail_content
    assert "def process_payment" in tail_content
    assert "subtotal = compute_subtotal(items)" in tail_content


def test_transform_adapter_tokens_before_is_baseline_not_reconstruction(tokenizer):
    """tokens_before must reflect the real original messages, not back-calc."""
    transform = ToolResultInterceptorTransform()
    messages = [
        {"role": "user", "content": [{"type": "text", "text": "plain non-tool message"}]},
    ]
    result = transform.apply(messages, tokenizer)
    # No spans, no change.
    assert result.tokens_before == result.tokens_after
    assert result.transforms_applied == []


def test_proxy_pipeline_includes_interceptor_when_env_enabled(monkeypatch):
    """An eligible legacy request installs the interceptor in both pipelines."""
    monkeypatch.setenv("HEADROOM_INTERCEPT_ENABLED", "1")
    monkeypatch.setenv("HEADROOM_ROLLOUT_CHANNEL", "canary")
    from headroom.proxy.interceptors import ToolResultInterceptorTransform
    from headroom.proxy.models import ProxyConfig
    from headroom.proxy.server import HeadroomProxy

    proxy = HeadroomProxy(ProxyConfig())
    for pipeline in (proxy.anthropic_pipeline, proxy.openai_pipeline):
        transforms = pipeline.transforms
        assert len(transforms) > 0
        assert isinstance(transforms[0], ToolResultInterceptorTransform)


def test_proxy_pipeline_blocks_interceptor_below_rollout_channel(monkeypatch):
    """A legacy request cannot bypass the stable rollout-channel boundary."""
    monkeypatch.setenv("HEADROOM_INTERCEPT_ENABLED", "1")
    monkeypatch.setenv("HEADROOM_ROLLOUT_CHANNEL", "stable")
    from headroom.proxy.interceptors import ToolResultInterceptorTransform
    from headroom.proxy.models import ProxyConfig
    from headroom.proxy.server import HeadroomProxy

    proxy = HeadroomProxy(ProxyConfig())
    for pipeline in (proxy.anthropic_pipeline, proxy.openai_pipeline):
        assert not any(isinstance(t, ToolResultInterceptorTransform) for t in pipeline.transforms)


def test_proxy_pipeline_excludes_interceptor_when_env_not_set(monkeypatch):
    """When HEADROOM_INTERCEPT_ENABLED is unset, no interceptor in either pipeline."""
    monkeypatch.delenv("HEADROOM_INTERCEPT_ENABLED", raising=False)
    from headroom.proxy.interceptors import ToolResultInterceptorTransform
    from headroom.proxy.models import ProxyConfig
    from headroom.proxy.server import HeadroomProxy

    proxy = HeadroomProxy(ProxyConfig())
    for pipeline in (proxy.anthropic_pipeline, proxy.openai_pipeline):
        transforms = pipeline.transforms
        assert not any(isinstance(t, ToolResultInterceptorTransform) for t in transforms)
