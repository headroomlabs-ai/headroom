from __future__ import annotations

import headroom.proxy.handlers.openai as openai_mod


def test_openai_chat_routes_copilot_requests_per_model() -> None:

    copilot_base = "https://api.githubcopilot.com"
    gpt54_mini_url = openai_mod.build_copilot_upstream_url(
        copilot_base,
        openai_mod._resolve_openai_handler_path(
            {},
            handler_path=openai_mod._resolve_openai_chat_handler_path(copilot_base, "gpt-5.4-mini"),
        ),
    )
    claude_url = openai_mod.build_copilot_upstream_url(
        copilot_base,
        openai_mod._resolve_openai_handler_path(
            {},
            handler_path=openai_mod._resolve_openai_chat_handler_path(
                copilot_base, "claude-sonnet-5"
            ),
        ),
    )
    openai_url = openai_mod.build_copilot_upstream_url(
        "https://api.openai.com",
        openai_mod._resolve_openai_handler_path(
            {},
            handler_path=openai_mod._resolve_openai_chat_handler_path(
                "https://api.openai.com", "gpt-5.4-mini"
            ),
        ),
    )

    assert gpt54_mini_url == "https://api.githubcopilot.com/responses"
    assert claude_url == "https://api.githubcopilot.com/chat/completions"
    assert openai_url == "https://api.openai.com/v1/chat/completions"
