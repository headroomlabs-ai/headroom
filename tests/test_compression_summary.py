"""Tests for compression summary generation."""

from headroom.transforms.compression_summary import (
    _extract_name_from_signature,
    summarize_compressed_code,
    summarize_dropped_items,
)


class TestSummarizeDroppedItems:
    def test_items_with_status_field(self):
        all_items = [{"id": i, "name": f"item-{i}", "status": "active"} for i in range(50)] + [
            {"id": i, "name": f"err-{i}", "status": "error"} for i in range(5)
        ]
        kept = all_items[:3]
        summary = summarize_dropped_items(all_items, kept, kept_indices={0, 1, 2})
        assert "active" in summary
        assert summary

    def test_items_with_type_field(self):
        all_items = [{"type": "log", "message": f"entry {i}"} for i in range(30)] + [
            {"type": "metric", "value": i} for i in range(20)
        ]
        kept = all_items[:5]
        summary = summarize_dropped_items(all_items, kept, kept_indices={0, 1, 2, 3, 4})
        assert "log" in summary or "metric" in summary

    def test_notable_items_with_errors(self):
        all_items = [{"name": f"test-{i}", "result": "pass"} for i in range(40)] + [
            {"name": "test-auth", "result": "fail", "error": "authentication failed"},
        ]
        kept = all_items[:5]
        summary = summarize_dropped_items(all_items, kept, kept_indices=set(range(5)))
        assert summary

    def test_no_dropped_items(self):
        items = [{"id": 1}, {"id": 2}]
        summary = summarize_dropped_items(items, items)
        assert summary == ""

    def test_empty_input(self):
        summary = summarize_dropped_items([], [])
        assert summary == ""

    def test_all_items_dropped(self):
        items = [{"status": "active", "name": f"item-{i}"} for i in range(20)]
        summary = summarize_dropped_items(items, [], kept_indices=set())
        assert "active" in summary

    def test_mixed_category_fields(self):
        all_items = [{"level": "info", "msg": "something"} for _ in range(10)] + [
            {"level": "error", "msg": "bad thing"} for _ in range(3)
        ]
        kept = all_items[:2]
        summary = summarize_dropped_items(all_items, kept, kept_indices={0, 1})
        assert summary

    def test_items_without_category_fields(self):
        all_items = [{"code": 200, "count": i} for i in range(30)]
        kept = all_items[:3]
        summary = summarize_dropped_items(all_items, kept, kept_indices={0, 1, 2})
        assert summary  # Should produce field-based fallback

    def test_summary_not_too_long(self):
        all_items = [{"type": f"type_{i % 20}", "data": "x"} for i in range(100)]
        kept = all_items[:5]
        summary = summarize_dropped_items(all_items, kept, kept_indices=set(range(5)))
        assert len(summary) < 500

    def test_url_values_excluded_from_categories(self):
        """URL-like values should not be used as category labels."""
        all_items = [
            {"url": f"https://api.example.com/v1/items/{i}", "method": "GET"} for i in range(30)
        ]
        kept = all_items[:3]
        summary = summarize_dropped_items(all_items, kept, kept_indices={0, 1, 2})
        assert "https://" not in summary

    def test_fallback_without_kept_indices(self):
        """Works correctly without kept_indices (uses _item_key fallback)."""
        all_items = [{"status": "active", "id": i} for i in range(20)]
        kept = [all_items[0], all_items[1]]  # Copies from same list
        summary = summarize_dropped_items(all_items, kept)
        assert summary


class TestSummarizeCompressedCode:
    def test_python_function_bodies(self):
        bodies = [
            ("def authenticate(username, password):", "    db = get_db()\n    return True", 10),
            ("def validate_token(token):", "    return jwt.decode(token)", 20),
            ("def refresh_session(user):", "    session.extend()", 30),
        ]
        summary = summarize_compressed_code(bodies, 3)
        assert "3 bodies compressed" in summary
        assert "authenticate()" in summary
        assert "validate_token()" in summary

    def test_javascript_function_bodies(self):
        bodies = [
            ("function handleRequest(req, res) {", "  res.send('ok');", 5),
            ("async function fetchData(url) {", "  return await fetch(url);", 15),
        ]
        summary = summarize_compressed_code(bodies, 2)
        assert "handleRequest()" in summary
        assert "fetchData()" in summary

    def test_go_function_bodies(self):
        bodies = [
            (
                "func (s *Server) HandleRequest(w http.ResponseWriter, r *http.Request) {",
                '  w.Write([]byte("ok"))',
                10,
            ),
            ("func main() {", "  server.Start()", 1),
        ]
        summary = summarize_compressed_code(bodies, 2)
        assert "HandleRequest()" in summary
        assert "main()" in summary

    def test_rust_function_bodies(self):
        bodies = [
            ("fn authenticate(token: &str) -> Result<User, Error> {", "  Ok(User::new())", 10),
        ]
        summary = summarize_compressed_code(bodies, 1)
        assert "authenticate()" in summary

    def test_empty_bodies(self):
        summary = summarize_compressed_code([], 0)
        assert summary == ""

    def test_many_bodies_truncated(self):
        bodies = [(f"def func_{i}(x):", f"    return {i}", i * 10) for i in range(20)]
        summary = summarize_compressed_code(bodies, 20)
        assert "+14 more" in summary  # 20 - 6 shown


class TestExtractNameFromSignature:
    def test_python_def(self):
        assert _extract_name_from_signature("def authenticate(username):") == "authenticate()"

    def test_python_async_def(self):
        assert _extract_name_from_signature("async def fetch_data(url):") == "fetch_data()"

    def test_javascript_function(self):
        assert _extract_name_from_signature("function handleClick(event) {") == "handleClick()"

    def test_go_func(self):
        assert (
            _extract_name_from_signature("func HandleRequest(w http.ResponseWriter) {")
            == "HandleRequest()"
        )

    def test_go_method(self):
        assert _extract_name_from_signature("func (s *Server) Start() {") == "Start()"

    def test_rust_fn(self):
        assert (
            _extract_name_from_signature("fn authenticate(token: &str) -> Result<User> {")
            == "authenticate()"
        )

    def test_java_method(self):
        assert (
            _extract_name_from_signature("public void processPayment(Payment p) {")
            == "processPayment()"
        )

    def test_class(self):
        assert _extract_name_from_signature("class TokenValidator:") == "TokenValidator"

    def test_empty(self):
        assert _extract_name_from_signature("") == ""

    def test_export_async(self):
        assert (
            _extract_name_from_signature("export async function fetchUsers() {") == "fetchUsers()"
        )
