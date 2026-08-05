from __future__ import annotations

from headroom.proxy.request_log_redaction_policy import (
    IMAGE_BASE64_REDACT_THRESHOLD_BYTES,
    IMAGE_BASE64_REPLACEMENT_TEMPLATE,
    is_base64_image_payload,
    redact_image_base64_value,
)


def test_policy_reports_redaction_count_without_side_effects() -> None:
    image_payload = "x" * IMAGE_BASE64_REDACT_THRESHOLD_BYTES
    non_image_payload = "y" * IMAGE_BASE64_REDACT_THRESHOLD_BYTES

    result = redact_image_base64_value(
        {
            "source": {"data": image_payload},
            "signature": non_image_payload,
        }
    )

    assert result.redactions == 1
    assert result.value["source"]["data"] == IMAGE_BASE64_REPLACEMENT_TEMPLATE.format(
        n=len(image_payload)
    )
    assert result.value["signature"] == non_image_payload


def test_policy_counts_nested_list_redactions() -> None:
    data_url = "data:image/png;base64," + ("A" * IMAGE_BASE64_REDACT_THRESHOLD_BYTES)
    direct_payload = "B" * IMAGE_BASE64_REDACT_THRESHOLD_BYTES

    result = redact_image_base64_value(
        [{"content": [{"image_url": {"url": data_url}}, {"image": direct_payload}]}]
    )

    assert result.redactions == 2
    assert result.value[0]["content"][0]["image_url"]["url"] == (
        IMAGE_BASE64_REPLACEMENT_TEMPLATE.format(n=len(data_url))
    )
    assert result.value[0]["content"][1]["image"] == IMAGE_BASE64_REPLACEMENT_TEMPLATE.format(
        n=len(direct_payload)
    )


def test_explicit_image_data_url_requires_threshold() -> None:
    short_data_url = "data:image/png;base64,abc"
    long_data_url = "data:image/png;base64," + ("A" * IMAGE_BASE64_REDACT_THRESHOLD_BYTES)

    assert not is_base64_image_payload(short_data_url)
    assert is_base64_image_payload(long_data_url)
