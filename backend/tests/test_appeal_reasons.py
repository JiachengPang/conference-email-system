"""Tests for the appeal-reason registry (reject-appeal Phase 2, Step 2).

Pure module, so pure tests: no DB, no model calls, no labels file.

The expected names, codes and escalation flags are pinned as LITERALS, never
derived from the registry under test — a test parametrized off the set it
checks would lose a case when a member is dropped, instead of failing (the
same reasoning as test_taxonomy.py's `_EXPECTED_APPEAL_INTENTS`).
"""

from __future__ import annotations

import ast
import dataclasses
from pathlib import Path

import pytest

from app.pipeline import appeal_reasons as ar

_EXPECTED_NAMES = [
    "wrong_paper_review",
    "score_outcome_mismatch",
    "reviewer_misunderstanding",
    "llm_generated_review",
    "general_dissatisfaction",
    "other",
]
_EXPECTED_CODES = ["a", "b", "c", "d", "e", "o"]
_EXPECTED_ESCALATING = ["wrong_paper_review", "score_outcome_mismatch"]


# ---------------------------------------------------------------------------
# Registry shape
# ---------------------------------------------------------------------------
def test_registry_names_in_order():
    assert list(ar.REASON_NAMES) == _EXPECTED_NAMES
    assert [r.name for r in ar.APPEAL_REASONS] == _EXPECTED_NAMES


def test_registry_label_codes_pinned_in_registry_order():
    """CONTAINER-SIDE PIN for the label codes.

    Runs everywhere. Its CI-only twin below checks the same codes against
    scripts/labeling/label_appeals.py, which is not in the backend image.
    Together they catch drift on either side: this one when the registry
    changes, that one when the labeling tool does.
    """
    assert [r.label_code for r in ar.APPEAL_REASONS] == _EXPECTED_CODES


def test_escalates_is_true_exactly_for_the_first_two():
    assert [r.name for r in ar.APPEAL_REASONS if r.escalates] == _EXPECTED_ESCALATING
    assert [r.escalates for r in ar.APPEAL_REASONS] == [True, True, False, False, False, False]


def test_r_and_n_are_not_reasons():
    """D60: `r` belongs to is_reciprocal_dispute. D67: `n` is `[]`, not a reason."""
    codes = {r.label_code for r in ar.APPEAL_REASONS}
    assert "r" not in codes and "n" not in codes
    assert not any("reciprocal" in name for name in ar.REASON_NAMES)


def test_names_and_codes_are_unique():
    assert len(set(ar.REASON_NAMES)) == len(ar.REASON_NAMES)
    codes = [r.label_code for r in ar.APPEAL_REASONS]
    assert len(set(codes)) == len(codes)


def test_no_letter_code_doubles_as_a_wire_name():
    """Label codes must never be accepted on the wire."""
    assert not set(ar.LABEL_CODE_TO_NAME) & set(ar.REASON_NAMES)


def test_descriptions_are_one_nonempty_line():
    for r in ar.APPEAL_REASONS:
        assert r.description.strip(), r.name
        assert "\n" not in r.description, r.name


def test_descriptions_state_the_claim_without_internal_references():
    """Descriptions may later be shown to a model — no D-numbers or codes."""
    for r in ar.APPEAL_REASONS:
        assert "D6" not in r.description and "D5" not in r.description, r.name
        assert r.description.startswith("The author "), r.name


def test_label_code_to_name_mapping():
    assert dict(ar.LABEL_CODE_TO_NAME) == dict(zip(_EXPECTED_CODES, _EXPECTED_NAMES))


def test_registry_is_immutable():
    assert isinstance(ar.APPEAL_REASONS, tuple)
    with pytest.raises(dataclasses.FrozenInstanceError):
        ar.APPEAL_REASONS[0].escalates = False  # type: ignore[misc]
    with pytest.raises(TypeError):
        ar.LABEL_CODE_TO_NAME["x"] = "other"  # type: ignore[index]


# ---------------------------------------------------------------------------
# normalize_reasons
# ---------------------------------------------------------------------------
def test_normalize_empty_list_is_asked_none_apply():
    assert ar.normalize_reasons([]) == []


def test_normalize_returns_registry_order():
    assert ar.normalize_reasons(["other", "wrong_paper_review", "llm_generated_review"]) == [
        "wrong_paper_review",
        "llm_generated_review",
        "other",
    ]


def test_normalize_removes_duplicates():
    assert ar.normalize_reasons(["other", "other", "score_outcome_mismatch", "other"]) == [
        "score_outcome_mismatch",
        "other",
    ]


@pytest.mark.parametrize(
    "values",
    [
        ["not_a_reason"],
        ["wrong_paper_review", "not_a_reason"],  # one unknown poisons the answer
        ["a"],  # letter codes are never accepted on the wire
        ["Other"],  # no case folding
        [" other"],  # no trimming
        ["reciprocal_review_duty_dispute"],  # r is not a reason (D60)
    ],
)
def test_normalize_unknown_value_fails_the_whole_answer(values):
    assert ar.normalize_reasons(values) is None


@pytest.mark.parametrize(
    "values",
    [None, "other", ("other",), {"other"}, {"other": True}, 0],
)
def test_normalize_non_list_is_none(values):
    assert ar.normalize_reasons(values) is None


@pytest.mark.parametrize(
    "values",
    [[1], [None], [True], ["other", 3], [["other"]]],
)
def test_normalize_non_str_item_is_none(values):
    assert ar.normalize_reasons(values) is None


def test_normalize_returns_a_new_list_and_leaves_input_alone():
    original = ["other", "wrong_paper_review", "other"]
    snapshot = list(original)
    result = ar.normalize_reasons(original)
    assert result is not original
    assert original == snapshot


# ---------------------------------------------------------------------------
# is_valid_stored (the D66 preserve rule)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "value",
    [[], ["other"], ["wrong_paper_review", "score_outcome_mismatch"], list(_EXPECTED_NAMES)],
)
def test_canonical_stored_lists_are_valid(value):
    assert ar.is_valid_stored(value) is True


@pytest.mark.parametrize(
    "value",
    [
        None,  # not an answer (D59)
        ["other", "wrong_paper_review"],  # not registry order
        ["other", "other"],  # duplicates
        ["b"],  # letter code
        ["not_a_reason"],
        "other",
        ("other",),
        [1],
    ],
)
def test_non_canonical_stored_values_are_not_a_prior(value):
    assert ar.is_valid_stored(value) is False


# ---------------------------------------------------------------------------
# escalation_reasons
# ---------------------------------------------------------------------------
def test_escalation_reasons_keeps_only_escalating_in_order():
    assert ar.escalation_reasons(
        ["other", "score_outcome_mismatch", "reviewer_misunderstanding", "wrong_paper_review"]
    ) == ["wrong_paper_review", "score_outcome_mismatch"]


def test_escalation_reasons_empty_cases():
    assert ar.escalation_reasons([]) == []
    assert ar.escalation_reasons(None) == []
    assert ar.escalation_reasons(["other", "general_dissatisfaction"]) == []


def test_escalation_reasons_rejects_an_invalid_list_loudly():
    """`[]` here would silently read as "no escalation ground"."""
    with pytest.raises(ValueError):
        ar.escalation_reasons(["wrong_paper_review", "not_a_reason"])


# ---------------------------------------------------------------------------
# Drift vs the labeling tool — CI-only
# ---------------------------------------------------------------------------
_LABEL_TOOL = Path(__file__).resolve().parents[2] / "scripts" / "labeling" / "label_appeals.py"

needs_label_tool = pytest.mark.skipif(
    not _LABEL_TOOL.exists(),
    reason="scripts/labeling/label_appeals.py is not in the backend image "
    "(the Dockerfile copies only backend/ and data/); this drift test runs in "
    "CI, where the whole repo is checked out",
)


def _label_tool_reason_codes() -> list[str]:
    """`REASONS` keys from the labeling tool, read by AST, never executed."""
    tree = ast.parse(_LABEL_TOOL.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "REASONS" for t in node.targets
        ):
            return list(ast.literal_eval(node.value))
    raise AssertionError("REASONS not found in label_appeals.py")


@needs_label_tool
def test_registry_codes_match_the_labeling_tool():
    """Registry label codes == label_appeals.REASONS keys minus {'r'}.

    Codes only: the tool's display names ("wrong-paper review") are labeling-UI
    text, not wire names. `r` is required to be present there and absent here
    (D60), so the subtraction is checked rather than assumed.
    """
    tool_codes = _label_tool_reason_codes()
    assert "r" in tool_codes
    assert {r.label_code for r in ar.APPEAL_REASONS} == set(tool_codes) - {"r"}
