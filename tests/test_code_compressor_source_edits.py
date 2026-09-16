import pytest

from headroom.transforms.code_compressor import SourceEdit, apply_source_edits


def test_source_edits_copy_gaps_and_allow_adjacency():
    assert (
        apply_source_edits(
            "a\r\nbc\n",
            [
                SourceEdit(0, 1, "A"),
                SourceEdit(3, 5, "BC"),
            ],
        )
        == "A\r\nBC\n"
    )


def test_source_edits_empty_and_unsorted_plans():
    assert apply_source_edits("unchanged", []) == "unchanged"
    assert (
        apply_source_edits(
            "abcdef",
            [SourceEdit(3, 4, "D"), SourceEdit(0, 1, "A")],
        )
        == "AbcDef"
    )


def test_source_edits_accept_valid_utf8_boundary():
    assert apply_source_edits("éclair", [SourceEdit(2, 3, "X")]) == "éXlair"


def test_source_edits_rejects_a_split_at_the_start_of_a_multibyte_character():
    assert apply_source_edits("éclair", [SourceEdit(1, 2, "X")]) is None


@pytest.mark.parametrize(
    "edits",
    [
        [SourceEdit(2, 1, "")],
        [SourceEdit(0, 2, ""), SourceEdit(1, 3, "")],
        [SourceEdit(0, 1, ""), SourceEdit(0, 1, "")],
        [SourceEdit(0, 99, "")],
        [SourceEdit(1, 2, "")],
        [SourceEdit(0, 1, "")],
    ],
)
def test_source_edits_reject_invalid_plans(edits):
    assert apply_source_edits("éclair", edits) is None
