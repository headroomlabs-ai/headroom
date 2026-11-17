"""YAML scalar contents are data even when a line starts with a hash."""

import pytest

from headroom.transforms.config_compressor import ConfigCompressor


@pytest.mark.parametrize("newline", ["\n", "\r\n"])
@pytest.mark.parametrize(
    "header",
    [
        "  - |-",
        "  - >+",
        "  - | # literal script",
        "  - >2- # folded script",
        "  value: | # literal script",
        "  value: >- # folded script",
        "  value: |2-",
        "  value: >2+",
        "  value: |-2",
        "  value: !script |",
    ],
)
def test_yaml_scalar_hash_lines_survive_compression(header: str, newline: str) -> None:
    content = (
        "apiVersion: v1\nkind: ConfigMap\nmetadata:\n  name: demo\nscripts:\n"
        + header
        + "\n"
        + "".join(
            f"    # note {i}: this is scalar content, not a removable YAML comment\n"
            for i in range(12)
        )
        + "    echo ready\n"
    )
    content = content.replace("\n", newline)
    result = ConfigCompressor().compress(content)
    assert result.flavor == "yaml"
    assert result.lines_elided == 0
    assert result.ccr_hash is None
    assert result.compressed == content


def test_plain_yaml_comments_remain_eligible_for_elision() -> None:
    content = "metadata:\n  name: demo\n  value: plain\n# ordinary comment\n"
    assert ConfigCompressor._elision_safe(content, "yaml")
