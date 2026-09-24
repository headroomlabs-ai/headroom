"""Offline adapter contracts and content-free request feature classification."""

from typing import Any

QUALIFIED_TRANSLATIONS = frozenset(
    {
        ("openai-chat", "anthropic-messages"),
        ("anthropic-messages", "openai-chat"),
        ("gemini-generate", "openai-chat"),
    }
)


def implemented_features(
    protocol: str, transport: str, native: bool, target: str | None = None
) -> frozenset[str]:
    if transport == "websocket" and (protocol != "openai-responses" or not native):
        return frozenset()
    if protocol == "bedrock-invoke" and transport != "http-json":
        return frozenset()
    if not native:
        return (
            frozenset({"text", "inline_images"})
            if transport in {"http-json", "http-stream"}
            and (protocol, target) in QUALIFIED_TRANSLATIONS
            else frozenset()
        )
    return frozenset(
        {"text", "tools", "parallel_tools", "inline_images", "structured_output", "signed_state"}
    )


def requested_features(protocol: str, payload: dict[str, Any]) -> frozenset[str]:
    """Classify native semantic fields, never arbitrary tool arguments or schemas."""
    features = {"text"}

    def content(value: Any) -> None:
        if isinstance(value, list):
            for item in value:
                content(item)
            return
        if not isinstance(value, dict):
            return
        kind = value.get("type")
        if kind in {"image", "image_url", "input_image"} or "image" in value:
            features.add("inline_images")
        if kind in {"input_audio", "audio", "video", "file", "input_file", "document"} or any(
            key in value for key in ("audio", "video", "document", "fileData")
        ):
            features.add("unsupported_media")
        inline = value.get("inlineData")
        if isinstance(inline, dict):
            mime = inline.get("mimeType")
            features.add(
                "inline_images"
                if isinstance(mime, str) and mime.startswith("image/")
                else "unsupported_media"
            )
        if kind in {"thinking", "redacted_thinking", "reasoning"} or any(
            key in value for key in ("signature", "encrypted_content", "thoughtSignature")
        ):
            features.add("signed_state")
        if protocol == "openai-chat" and (
            value.get("role") in {"tool", "function"}
            or "tool_calls" in value
            or "function_call" in value
        ):
            features.add("tools")
        if protocol == "openai-responses" and kind in {
            "function_call",
            "function_call_output",
            "custom_tool_call",
            "custom_tool_call_output",
        }:
            features.add("tools")
            if kind == "function_call_output" and isinstance(value.get("output"), list):
                content(value["output"])
        if protocol in {"anthropic-messages", "bedrock-invoke", "bedrock-converse"} and (
            kind in {"tool_use", "tool_result"} or "toolUse" in value or "toolResult" in value
        ):
            features.add("tools")
        if protocol in {"gemini-generate", "vertex-generate"} and any(
            key in value for key in ("functionCall", "functionResponse")
        ):
            features.add("tools")
        if isinstance(kind, str) and (
            kind.startswith(("web_search", "code_interpreter", "file_search", "computer"))
            or kind
            in {"server_tool_use", "code_execution_tool_result", "bash_code_execution_tool_result"}
        ):
            features.add("hosted_tools")
        for key in ("content", "messages", "input", "contents", "parts", "system"):
            content(value.get(key))

    tools = payload.get("tools")
    if tools or payload.get("tool_choice") or payload.get("toolConfig") or payload.get("functions"):
        features.add("tools")
    if isinstance(tools, list):
        for tool in tools:
            if not isinstance(tool, dict):
                continue
            kind = tool.get("type")
            if (
                isinstance(kind, str)
                and kind not in {"function", "custom"}
                and not kind.startswith("function_")
            ) or any(
                key in tool
                for key in (
                    "googleSearch",
                    "googleSearchRetrieval",
                    "codeExecution",
                    "retrieval",
                    "urlContext",
                    "computerUse",
                )
            ):
                features.add("hosted_tools")
    if payload.get("parallel_tool_calls"):
        features.add("parallel_tools")
    text_options = payload.get("text")
    output_config = payload.get("output_config")
    generation_config = payload.get("generationConfig")
    if (
        payload.get("response_format")
        or isinstance(text_options, dict)
        and text_options.get("format")
        or isinstance(output_config, dict)
        and output_config.get("format")
        or isinstance(generation_config, dict)
        and any(key in generation_config for key in ("responseSchema", "responseJsonSchema"))
        or isinstance(generation_config, dict)
        and generation_config.get("responseMimeType") not in {None, "text/plain"}
    ):
        features.add("structured_output")
    content(payload)
    return frozenset(features)
