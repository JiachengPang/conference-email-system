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
