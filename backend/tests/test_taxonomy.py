import pytest

from app.pipeline import taxonomy as tx


def test_fourteen_intents_five_families():
    assert len(tx.VALID_INTENTS) == 14
    assert len(set(tx.VALID_INTENTS)) == 14
    assert set(tx.INTENT_FAMILIES) == set(tx.VALID_INTENTS)
    assert set(tx.INTENT_DEFS) == set(tx.VALID_INTENTS)
    assert len(tx.FAMILIES) == 5
    assert set(tx.INTENT_FAMILIES.values()) == set(tx.FAMILIES)


def test_fallback_intent_is_valid():
    assert tx.FALLBACK_INTENT in tx.VALID_INTENTS


def test_definitions_nonempty():
    assert all(tx.INTENT_DEFS[i].strip() for i in tx.VALID_INTENTS)


# --- reject-appeal derived helper -------------------------------------------


# Pinned literally, NOT derived from tx.REJECT_APPEAL_INTENTS: parametrizing off
# the set under test means a dropped member makes its case DISAPPEAR rather than
# fail, so the test could never catch the set shrinking.
_EXPECTED_APPEAL_INTENTS = ["desk_reject_appeal", "review_decision_appeal"]


def test_reject_appeal_intents_are_exactly_the_two_appeals():
    assert tx.REJECT_APPEAL_INTENTS == frozenset(_EXPECTED_APPEAL_INTENTS)


def test_reject_appeal_intents_are_all_valid():
    # The frozenset is the single source of truth, so a typo in it would
    # silently make is_reject_appeal() always False for that name.
    assert tx.REJECT_APPEAL_INTENTS <= set(tx.VALID_INTENTS)


@pytest.mark.parametrize("intent", _EXPECTED_APPEAL_INTENTS)
def test_is_reject_appeal_true_for_each_appeal_intent(intent):
    assert tx.is_reject_appeal(intent) is True


def test_is_reject_appeal_false_for_non_appeal_intent():
    assert tx.is_reject_appeal("submission_upload_help") is False


def test_is_reject_appeal_none_for_none():
    # None means "not classified" and must stay distinguishable from False.
    result = tx.is_reject_appeal(None)
    assert result is None
    assert result is not False


# --- desk_reject_appeal definition text -------------------------------------
#
# ⚠️ INVERTED. Two tests here used to REQUIRE the tokens "reciprocal-review" and
# "waive", which `beb5cf6` added. That amendment was REVERTED for a zero-change
# guarantee: this definition is interpolated verbatim into the distiller's
# intent menu, so editing it edits the live production prompt — the one thing
# the detector rebuild exists to avoid. The tokens now live in
# `reciprocal_detector.py`'s own prompt, where changing them cannot perturb
# retrieval.
#
# These assert ABSENCE rather than presence, so a well-meaning re-amendment has
# to argue with a test instead of silently reopening the prompt.


def test_desk_reject_appeal_definition_is_the_original_wording():
    """The definition must stay byte-identical to the pre-series original.

    Not a style preference: `INTENT_DEFS[...]` is interpolated VERBATIM into
    `distiller._INTENT_MENU`, which is interpolated into `_SYSTEM_PROMPT`. Any
    edit here changes the production prompt, and the same prompt also emits the
    retrieval QUERY lines — a 798-char addition to it measurably moved
    retrieval (Jaccard 0.445 against a 0.710 noise floor, reject_appeal.md
    D27/D35). The whole point of moving the reciprocal question into its own
    call was to stop paying that cost.
    """
    assert tx.INTENT_DEFS["desk_reject_appeal"] == (
        "Requests to explain, reconsider, or reverse a desk rejection "
        "(formatting, page-limit, appendix, checklist, or compliance grounds)."
    )


def test_the_reciprocal_tokens_are_NOT_in_the_taxonomy():
    """The coverage those tokens bought now lives in the detector's prompt.

    ⚠️ This has a real, accepted cost: on half 1 the amended definition
    classified 54 of 56 r tickets as `desk_reject_appeal` versus 50 of 56
    without it — about 7pp of reach, which the detector's intent gate then
    inherits. Traded for a guarantee that the existing pipeline does not change
    at all. The follow-up, if that cost bites, is to widen the DETECTOR's gate
    to the neighbouring intents where r tickets land — a change confined to the
    new code, touching no prompt.
    """
    definition = tx.INTENT_DEFS["desk_reject_appeal"].lower()
    assert "reciprocal" not in definition
    assert "waive" not in definition
