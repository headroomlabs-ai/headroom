from headroom.providers.vertex import resolve_vertex_route


def test_vertex_route_matrix_constrains_google_actions_to_gemini_handlers() -> None:
    generate = resolve_vertex_route("google", "generateContent")
    stream = resolve_vertex_route("google", "streamGenerateContent")
    count = resolve_vertex_route("google", "countTokens")

    assert generate.kind == "gemini_generate_content"
    assert generate.provider == "vertex:google"
    assert stream.kind == "gemini_generate_content"
    assert stream.provider == "vertex:google"
    assert count.kind == "gemini_count_tokens"
    assert count.provider == "vertex:google"


def test_vertex_route_matrix_constrains_anthropic_actions_to_messages_handler() -> None:
    raw = resolve_vertex_route("anthropic", "rawPredict")
    stream = resolve_vertex_route("anthropic", "streamRawPredict")

    assert raw.kind == "anthropic_messages"
    assert raw.provider == "vertex:anthropic"
    assert raw.force_stream is False
    assert stream.kind == "anthropic_messages"
    assert stream.provider == "vertex:anthropic"
    assert stream.force_stream is True


def test_vertex_route_matrix_falls_back_to_publisher_passthrough() -> None:
    route = resolve_vertex_route("mistral", "rawPredict")

    assert route.kind == "passthrough"
    assert route.provider == "vertex:mistral"
    assert route.action == "rawPredict"
