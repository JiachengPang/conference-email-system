"""Wording check: the ethics-form URL allow-list (reject-appeal Step 2.5, step 3).

Marc's misconduct point 3 links the AAAI ethics report form. That one exact URL
is allowed; every other URL is still refused under ``structure``. The same
holds for the AI draft's check 6 (2a), which runs the same wording check on the
rendered middle. Every expected value is a hand-written literal.
"""

from __future__ import annotations

import pytest

from app.pipeline.appeal_ai_suggestion import build_bank, check_answer
from app.pipeline.appeal_reply_lint import ALLOWED_URLS, lint_template_body
from app.pipeline.appeal_reply_templates import ApprovedTemplate

ETHICS_URL = (
    "https://docs.google.com/forms/d/e/"
    "1FAIpQLSdIs72RunUy5wKsOv7SdBma6A6riv3jp8lifUxlLcwhcdXMxw/viewform"
)
# Marc's approved point 3 (2026-10-05), exactly as written.
POINT_3 = (
    "You can also report unethical behavior through the ethics report form at "
    f"{ETHICS_URL}. This may impact our future relationship with this reviewer, but it "
    "will not change the outcome for this specific paper."
)

REFUSED_URLS = {
    "different-url": "https://docs.google.com/forms/d/e/1FAIpQLSc0therF0rmId000000000000000000/viewform",
    "different-host": "https://example.org/ethics",
    "one-char-changed-id": ETHICS_URL.replace("1FAIpQLSd", "1FAIpQLSe"),
    "one-char-changed-case": ETHICS_URL.replace("1FAIp", "1fAIp"),
    "one-char-changed-end": ETHICS_URL[:-1] + "n",
    "http-not-https": ETHICS_URL.replace("https://", "http://"),
    "extra-path": ETHICS_URL + "/extra",
    "extra-query": ETHICS_URL + "?usp=sf_link",
    "extra-fragment": ETHICS_URL + "#section",
    "extra-letters": ETHICS_URL + "x",
}


def _rules(body: str) -> set[str]:
    return {name for name, _ in lint_template_body(body)}


# --- the wording check ------------------------------------------------------
def test_only_the_ethics_form_url_is_allowed():
    assert ALLOWED_URLS == frozenset({ETHICS_URL})


def test_the_url_alone_passes():
    assert lint_template_body(ETHICS_URL) == []


@pytest.mark.parametrize("text", [
    f"{ETHICS_URL}.",
    f"Report it at {ETHICS_URL}.",
    f"Report it at {ETHICS_URL}. Then wait for the outcome.",
], ids=["url-then-stop", "sentence-end", "mid-paragraph"])
def test_the_url_followed_by_a_full_stop_passes(text):
    assert lint_template_body(text) == []


def test_marcs_point_3_is_clean():
    assert lint_template_body(POINT_3) == []


@pytest.mark.parametrize("url", REFUSED_URLS.values(), ids=REFUSED_URLS.keys())
@pytest.mark.parametrize("stop", ["", "."], ids=["bare", "with-stop"])
def test_every_other_url_is_still_refused_under_structure_only(url, stop):
    assert _rules(f"Report it at {url}{stop}") == {"structure"}


@pytest.mark.parametrize("url", REFUSED_URLS.values(), ids=REFUSED_URLS.keys())
def test_point_3_with_another_url_is_refused(url):
    assert _rules(POINT_3.replace(ETHICS_URL, url)) == {"structure"}


# --- check 6 of the AI draft (2a) -------------------------------------------
INTRO = "Thank you for your message."
OUTRO = "The decision is final."
SENT_1, SENT_2 = POINT_3.split(". ", 1)
SENT_1 += "."


def _block(block_id: str, body: str) -> ApprovedTemplate:
    return ApprovedTemplate(
        id=block_id, title=block_id, kind="point", order=1, optional=False,
        reasons=(), when_used="", body=body, approved_by="test", approved_at="2026-10-05",
        approved_sha256="x", cycle="AAAI-27", scope="phase1_reject", basis=(),
    )


def _check(point_body: str, sentence_1: str):
    bank = build_bank([_block("intro", INTRO), _block("ethics", point_body), _block("outro", OUTRO)])
    answer = f"INTRO: {INTRO}\nPOINT: {sentence_1} {SENT_2}\nOUTRO: {OUTRO}"
    return check_answer(answer, bank)


def test_check_6_passes_the_ethics_form_sentence():
    result = _check(POINT_3, SENT_1)
    assert result.failure is None
    assert result.middle == f"{INTRO}\n\n(1) {POINT_3}\n\n{OUTRO}"
    assert result.block_ids == ("intro", "ethics", "outro")


@pytest.mark.parametrize("url", REFUSED_URLS.values(), ids=REFUSED_URLS.keys())
def test_check_6_refuses_any_other_url_even_from_an_approved_sentence(url):
    result = _check(POINT_3.replace(ETHICS_URL, url), SENT_1.replace(ETHICS_URL, url))
    assert (result.middle, result.failure) == (None, "lint:structure")


@pytest.mark.parametrize("url", REFUSED_URLS.values(), ids=REFUSED_URLS.keys())
def test_check_6_refuses_a_model_that_alters_the_approved_url(url):
    """A changed URL is not an approved sentence: dropped before the wording check."""
    result = _check(POINT_3, SENT_1.replace(ETHICS_URL, url))
    assert (result.middle, result.failure) == (None, "foreign_sentence")
