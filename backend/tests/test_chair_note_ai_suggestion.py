"""Chair note for a flagged AI suggestion (reject-appeal Phase 4, 2c).

Mode ``ai_suggestion``: the banner "AI-written suggestion, not approved wording"
is always the second paragraph (directly under the fixed marker line), the
suggestion follows between the usual rules WITHOUT its [CHAIR: ...] flag line,
and the note can never pass for an approved copy block (no "COPY BELOW" / "END").
The chair notes (verify-before-sending checks included) are kept. Every other
mode is unchanged (test_chair_note.py). Nothing here touches Zendesk.
"""

from __future__ import annotations

import re

import pytest

from app.integrations.zendesk import chair_note
from app.integrations.zendesk.chair_note import (
    BANNER_DO_NOT_SEND,
    COPY_END,
    COPY_START,
    FOOTER,
    MARKER,
    _is_copyable,
    _paragraphs,
    build_chair_note_html,
)
from app.pipeline import appeal_ai_draft
from app.pipeline.paper_apc_resolver import ApcResolution

FLAG = "[CHAIR: AI-written suggestion, not approved wording; review and edit before sending]"
BANNER = "AI-written suggestion, not approved wording"
V_MISC = ("Before sending, check for harassment or an undisclosed conflict of interest. "
          "If either is present, do not send; forward the ticket to the Ethics Chairs.")
SUGGESTION = ("Dear Jane Doe,\n\nWe would like to respond to your concerns:\n\n"
              "(1) Decisions are not based on any single review; all assessments are weighed "
              "together.\n\nBest Regards,\nAAAI 2027 PC Team")


def _draft(text, mode="ai_suggestion", notes=V_MISC):
    return {"draft_text": text, "notes_for_chair": notes,
            "appeal_reply": {"mode": mode, "reasons": None, "block_ids": ["point_all_assessments"],
                             "source": "phase1",
                             "ai_suggestion": {"base_mode": "chair_writes"}}}


def _paras(html: str) -> list[str]:
    return re.findall(r"<p>(.*?)</p>", html)


RESOLUTIONS = {
    "nothing-found": ApcResolution(),
    "chair-and-link": ApcResolution(apc_names=("Grace Hopper",), forum_ids_matched=("Ab3xY9kLm2",)),
    "two-chairs-two-links": ApcResolution(apc_names=("A", "B"), forum_ids_matched=("Ab3xY9kLm2",),
                                          forum_ids_unmatched=("Zz9yX8wV7u",)),
}
EXTRACTIONS = {"no-numbers": {}, "numbers": {"submission_numbers": ["12345", "67890"]}}


def test_the_constants_match_the_ai_draft_module():
    assert chair_note.MODE_AI_SUGGESTION == appeal_ai_draft.MODE_AI_SUGGESTION
    assert chair_note.AI_FLAG_LINE == appeal_ai_draft.AI_FLAG_LINE == FLAG
    assert chair_note.BANNER_AI_SUGGESTION == BANNER


@pytest.mark.parametrize("resolution", RESOLUTIONS.values(), ids=RESOLUTIONS.keys())
@pytest.mark.parametrize("extraction", EXTRACTIONS.values(), ids=EXTRACTIONS.keys())
def test_the_banner_is_always_the_second_paragraph(resolution, extraction):
    html = build_chair_note_html(_draft(f"{FLAG}\n\n{SUGGESTION}"), extraction, resolution)
    assert html.startswith(f"<p><strong>{MARKER}</strong></p><p><strong>{BANNER}</strong></p>")
    assert _paras(html)[1] == f"<strong>{BANNER}</strong>"
    assert html.count(BANNER) == 1


def test_the_note_layout_exactly():
    html = build_chair_note_html(_draft(f"{FLAG}\n\n{SUGGESTION}"), {}, ApcResolution())
    assert html == (
        f"<p><strong>{MARKER}</strong></p>"
        f"<p><strong>{BANNER}</strong></p>"
        "<p>Chair: not identified from this email</p>"
        "<p>Paper number(s): none found in the email</p>"
        "<hr>"
        + _paragraphs(SUGGESTION)
        + "<hr>"
        "<p>Notes for the chair:</p>"
        f"<p>{V_MISC}</p>"
        f"<p><em>{FOOTER}</em></p>"
    )


def test_the_flag_line_never_appears_in_the_note():
    html = build_chair_note_html(_draft(f"{FLAG}\n\n{SUGGESTION}"), {}, ApcResolution())
    assert "[CHAIR" not in html
    assert "review and edit before sending" not in html


def test_it_can_never_pass_for_an_approved_copy_block():
    html = build_chair_note_html(_draft(f"{FLAG}\n\n{SUGGESTION}"), {}, ApcResolution())
    assert COPY_START not in html and f"<strong>{COPY_END}</strong>" not in html
    assert BANNER_DO_NOT_SEND not in html
    # Even after a chair deleted the flag line (no placeholder left), the mode is
    # never copyable and the banner stays.
    assert _is_copyable("ai_suggestion", SUGGESTION) is False
    edited = build_chair_note_html(_draft(SUGGESTION), {}, ApcResolution())
    assert edited == html
    assert COPY_START not in edited


def test_the_verify_notes_are_kept_after_the_rules():
    html = build_chair_note_html(_draft(f"{FLAG}\n\n{SUGGESTION}"), {}, ApcResolution())
    assert html.index("</p><hr>") < html.rindex("<hr>") < html.index(V_MISC) < html.index(FOOTER)


@pytest.mark.parametrize("text", [FLAG, "", f"{FLAG}\n\n[CHAIR: write reply]"],
                         ids=["flag-only", "empty", "another-placeholder"])
def test_nothing_usable_left_says_do_not_send_under_the_banner(text):
    html = build_chair_note_html(_draft(text), {}, ApcResolution())
    assert _paras(html)[1] == f"<strong>{BANNER}</strong>"
    assert f"<p><strong>{BANNER_DO_NOT_SEND}</strong></p>" in html
    assert "<hr>" not in html


def test_the_suggestion_is_escaped():
    html = build_chair_note_html(_draft(f"{FLAG}\n\n<script>x</script> & Co"), {}, ApcResolution())
    assert "<script>" not in html and "&lt;script&gt;x&lt;/script&gt; &amp; Co" in html


@pytest.mark.parametrize("mode", ["merged", "standalone", "no_draft", "reciprocal_review",
                                  "chair_writes", "reason_unknown", None])
def test_no_other_mode_gets_the_banner(mode):
    html = build_chair_note_html(_draft("Dear Author,\n\nReply.", mode=mode), {}, ApcResolution())
    assert BANNER not in html
