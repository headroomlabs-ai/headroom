"""Recognize original evidence when a delta omits its tool-call history."""

import json


def _is_original(value: object) -> bool:
    return (
        isinstance(value, dict)
        and isinstance(value.get("hash"), str)
        and bool(value["hash"])
        and isinstance(value.get("original_content"), str)
    )


def is_retrieval_result(text: str) -> bool:
    if "original_content" not in text:
        return False
    try:
        value = json.loads(text)
    except (ValueError, TypeError):
        return False
    if _is_original(value):
        return True
    if not isinstance(value, dict) or not isinstance(value.get("content"), list):
        return False
    for block in value["content"]:
        if not isinstance(block, dict) or block.get("type") != "text":
            continue
        try:
            if _is_original(json.loads(block.get("text", ""))):
                return True
        except (ValueError, TypeError):
            continue
    return False
