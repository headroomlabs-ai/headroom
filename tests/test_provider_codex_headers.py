from headroom.providers.codex.headers import drop_header, header_name


def test_header_name_matches_case_insensitively() -> None:
    headers = {
        "Authorization": "Bearer token",
        "ChatGPT-Account-ID": "acct",
    }

    assert header_name(headers, "authorization") == "Authorization"
    assert header_name(headers, "chatgpt-account-id") == "ChatGPT-Account-ID"
    assert header_name(headers, "missing") is None


def test_drop_header_removes_case_insensitive_match() -> None:
    headers = {
        "Host": "localhost:8787",
        "Accept-Encoding": "gzip",
        "authorization": "Bearer token",
    }

    drop_header(headers, "host")
    drop_header(headers, "accept-encoding")
    drop_header(headers, "missing")

    assert headers == {"authorization": "Bearer token"}
