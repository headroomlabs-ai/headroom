"""Tests for compression summary generation."""

from headroom.transforms.compression_summary import (
    _extract_name_from_signature,
    summarize_compressed_code,
)


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

    def test_real_python_module(self):
        bodies = [
            ("def __init__(self, url: str, pool_size: int = 10):", "...", 8),
            ("def connect(self) -> Any:", "...", 15),
            ("def _create_new(self) -> Any:", "...", 22),
            ("def release(self, conn: Any) -> None:", "...", 28),
            ("def close_all(self) -> None:", "...", 33),
            ("def create_engine(url: str) -> DatabaseConnection:", "...", 38),
        ]
        summary = summarize_compressed_code(bodies, 6)
        assert "6 bodies compressed" in summary
        has_names = any(
            name in summary for name in ["connect()", "release()", "close_all()", "create_engine()"]
        )
        assert has_names, f"Summary missing function names: {summary}"


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
