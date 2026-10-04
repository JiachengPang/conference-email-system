"""Chair notes (Z2a): settings, eligibility, the note HTML, and import isolation.

Every expected note is a hand-written literal, so a change to the wording or the
layout fails here rather than drifting silently. Nothing here touches Zendesk:
the builder and eligibility check are pure, and conftest makes
ZendeskSender.add_comment raise if anything calls it.
"""

import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from app.core.config import (
    Settings,
    parse_chair_note_intents,
    parse_chair_note_ticket_ids,
    settings,
)
from app.integrations.zendesk import chair_note
from app.integrations.zendesk.chair_note import (
    Eligibility,
    build_chair_note_html,
    check_eligibility,
    note_body_sha256,
)
from app.pipeline.paper_apc_resolver import ApcResolution

BACKEND_ROOT = Path(__file__).resolve().parents[1]

HEADER = "<p><strong>ConfMail draft (not sent to the author)</strong></p>"
FOOTER = "<p><em>Posted by ConfMail. Review and edit before sending.</em></p>"
NO_CHAIR = "<p>Chair: not identified from this email</p>"
NO_NUMBERS = "<p>Paper number(s): none found in the email</p>"
DO_NOT_SEND = "<p><strong>Do not send: the chair writes this reply.</strong></p>"

COMPOSED_TEXT = (
    "Dear Ada Lovelace,\n\n"
    "Thank you for your message.\nWe have reviewed it.\n\n"
    "Best Regards,\nAAAI 2027 PC Team"
)


def _draft(mode, text, notes=None):
    return {
        "draft_text": text,
        "notes_for_chair": notes,
        "appeal_reply": {"mode": mode, "reasons": ["x"], "block_ids": ["b1"]},
    }


# --- settings ---------------------------------------------------------------


def test_setting_defaults_are_off_and_scoped():
    # Field defaults, not an instantiated Settings: an operator's .env override
    # must not be able to fail this test.
    fields = Settings.model_fields
    assert fields["CHAIR_NOTE_ENABLED"].default is False
    assert fields["CHAIR_NOTE_INTENTS"].default == "review_decision_appeal,desk_reject_appeal"
    assert fields["CHAIR_NOTE_TICKET_IDS"].default == ""
    assert fields["CHAIR_NOTE_MAX_PER_CYCLE"].default == 20


def test_conftest_forces_the_flag_off():
    assert settings.CHAIR_NOTE_ENABLED is False


async def test_conftest_guard_makes_add_comment_raise():
    from app.integrations.zendesk.sender import ZendeskSender

    with pytest.raises(AssertionError, match="add_comment was called in a test"):
        await ZendeskSender().add_comment(None, 21567, html_body="<p>x</p>", public=False)


@pytest.mark.zendesk_transport
def test_the_transport_marker_opts_out_of_the_guard():
    from app.integrations.zendesk.sender import ZendeskSender

    assert ZendeskSender.add_comment.__name__ == "add_comment"


def test_intents_parse_like_statuses():
    assert parse_chair_note_intents(
        " review_decision_appeal, DESK_REJECT_APPEAL ,bogus_intent,review_decision_appeal"
    ) == ["review_decision_appeal", "desk_reject_appeal"]


@pytest.mark.parametrize("raw", ["", "   ", None, "bogus_intent", ",,"])
def test_intents_fail_closed_when_nothing_valid(raw):
    assert parse_chair_note_intents(raw) == []


def test_default_intents_parse_to_the_two_appeal_intents():
    assert parse_chair_note_intents(
        Settings.model_fields["CHAIR_NOTE_INTENTS"].default
    ) == ["review_decision_appeal", "desk_reject_appeal"]


@pytest.mark.parametrize("raw", ["", "   ", None])
def test_blank_ticket_allow_list_means_every_ticket(raw):
    assert parse_chair_note_ticket_ids(raw) is None


def test_ticket_allow_list_keeps_positive_ascii_integers_only():
    assert parse_chair_note_ticket_ids("21567, 42,abc,-3,0,²,4.5, 42") == frozenset({21567, 42})


@pytest.mark.parametrize("raw", ["abc", "0", "-1", "²"])
def test_ticket_allow_list_with_no_valid_id_allows_none(raw):
    assert parse_chair_note_ticket_ids(raw) == frozenset()


# --- eligibility --------------------------------------------------------------


def _email(**overrides):
    base = dict(
        zendesk_ticket_id=21567,
        zendesk_status="open",
        classification={"intent": "review_decision_appeal"},
        draft={"draft_text": "x", "appeal_reply": {"mode": "merged"}},
    )
    base.update(overrides)
    return SimpleNamespace(**base)


@pytest.fixture
def notes_on(monkeypatch):
    monkeypatch.setattr(settings, "CHAIR_NOTE_ENABLED", True)
    monkeypatch.setattr(settings, "CHAIR_NOTE_INTENTS", "review_decision_appeal,desk_reject_appeal")
    monkeypatch.setattr(settings, "CHAIR_NOTE_TICKET_IDS", "")


def test_flag_off_refuses_even_a_fully_eligible_email():
    assert check_eligibility(_email()) == Eligibility(False, "flag_off")


def test_eligible_email(notes_on):
    assert check_eligibility(_email()) == Eligibility(True, "eligible")


def test_desk_reject_appeal_is_in_scope(notes_on):
    email = _email(classification={"intent": "desk_reject_appeal"})
    assert check_eligibility(email) == Eligibility(True, "eligible")


@pytest.mark.parametrize("status", ["new", "open", "pending", "hold", " Open ", "PENDING"])
def test_actionable_statuses_are_eligible(notes_on, status):
    assert check_eligibility(_email(zendesk_status=status)).eligible is True


def test_no_ticket_id(notes_on):
    assert check_eligibility(_email(zendesk_ticket_id=None)) == Eligibility(False, "no_ticket_id")


@pytest.mark.parametrize("status", ["solved", "closed", None, "", "deleted"])
def test_resolved_or_unknown_status_is_refused(notes_on, status):
    assert check_eligibility(_email(zendesk_status=status)) == Eligibility(
        False, "ticket_status_not_allowed"
    )


def test_allow_list_excludes_other_tickets(notes_on, monkeypatch):
    monkeypatch.setattr(settings, "CHAIR_NOTE_TICKET_IDS", "99999")
    assert check_eligibility(_email()) == Eligibility(False, "ticket_not_allowlisted")


def test_allow_list_includes_its_ticket(notes_on, monkeypatch):
    monkeypatch.setattr(settings, "CHAIR_NOTE_TICKET_IDS", "99999, 21567")
    assert check_eligibility(_email()) == Eligibility(True, "eligible")


def test_allow_list_typo_allows_no_ticket(notes_on, monkeypatch):
    monkeypatch.setattr(settings, "CHAIR_NOTE_TICKET_IDS", "2l567")
    assert check_eligibility(_email()) == Eligibility(False, "ticket_not_allowlisted")


@pytest.mark.parametrize(
    "classification",
    [{"intent": "cms_support"}, {"intent": None}, {}, None, "review_decision_appeal"],
)
def test_intent_out_of_scope(notes_on, classification):
    assert check_eligibility(_email(classification=classification)) == Eligibility(
        False, "intent_out_of_scope"
    )


def test_intent_scope_follows_the_setting(notes_on, monkeypatch):
    monkeypatch.setattr(settings, "CHAIR_NOTE_INTENTS", "desk_reject_appeal")
    assert check_eligibility(_email()) == Eligibility(False, "intent_out_of_scope")


def test_intent_scope_typo_puts_nothing_in_scope(notes_on, monkeypatch):
    monkeypatch.setattr(settings, "CHAIR_NOTE_INTENTS", "review_decision_apeal")
    assert check_eligibility(_email()) == Eligibility(False, "intent_out_of_scope")


@pytest.mark.parametrize(
    "draft",
    [
        {"draft_text": "A model-written appeal draft."},
        {"draft_text": "x", "appeal_reply": None},
        {"draft_text": "x", "appeal_reply": "merged"},
        None,
        {},
    ],
)
def test_draft_without_the_appeal_reply_record_is_refused(notes_on, draft):
    assert check_eligibility(_email(draft=draft)) == Eligibility(False, "no_appeal_reply")


# --- the note HTML: one test per mode ----------------------------------------


def test_merged_note_offers_the_stored_reply_between_rules():
    html = build_chair_note_html(
        _draft("merged", COMPOSED_TEXT),
        {"submission_numbers": ["12345"], "openreview_forum_ids": ["Ab3xY9kLm2"]},
        ApcResolution(apc_names=("Grace Hopper",), forum_ids_matched=("Ab3xY9kLm2",)),
    )
    assert html == (
        "<p><strong>ConfMail draft (not sent to the author)</strong></p>"
        "<p>Chair: Grace Hopper</p>"
        "<p>Paper number(s) as written in the email (unverified): 12345</p>"
        '<p>OpenReview: <a href="https://openreview.net/forum?id=Ab3xY9kLm2">'
        "https://openreview.net/forum?id=Ab3xY9kLm2</a></p>"
        "<p><strong>COPY BELOW</strong></p>"
        "<hr>"
        "<p>Dear Ada Lovelace,</p>"
        "<p>Thank you for your message.<br>We have reviewed it.</p>"
        "<p>Best Regards,<br>AAAI 2027 PC Team</p>"
        "<hr>"
        "<p><strong>END</strong></p>"
        "<p><em>Posted by ConfMail. Review and edit before sending.</em></p>"
    )


def test_standalone_note_keeps_notes_outside_the_copy_block():
    html = build_chair_note_html(
        _draft("standalone", "Dear Author,\n\nShort reply.\n\nBest Regards,\nAAAI 2027 PC Team",
               notes="Check the review dates."),
        {"submission_numbers": [], "openreview_forum_ids": []},
        ApcResolution(),
    )
    assert html == (
        "<p><strong>ConfMail draft (not sent to the author)</strong></p>"
        "<p>Chair: not identified from this email</p>"
        "<p>Paper number(s): none found in the email</p>"
        "<p><strong>COPY BELOW</strong></p>"
        "<hr>"
        "<p>Dear Author,</p>"
        "<p>Short reply.</p>"
        "<p>Best Regards,<br>AAAI 2027 PC Team</p>"
        "<hr>"
        "<p><strong>END</strong></p>"
        "<p>Notes for the chair:</p>"
        "<p>Check the review dates.</p>"
        "<p><em>Posted by ConfMail. Review and edit before sending.</em></p>"
    )


def test_no_draft_note_says_investigate_first():
    html = build_chair_note_html(
        _draft("no_draft", "[CHAIR: do not reply yet; see note]",
               notes="The review may belong to a different paper."),
        {"submission_numbers": ["777"]},
        ApcResolution(),
    )
    assert html == (
        "<p><strong>ConfMail draft (not sent to the author)</strong></p>"
        "<p>Chair: not identified from this email</p>"
        "<p>Paper number(s) as written in the email (unverified): 777</p>"
        "<p><strong>Investigate first. Do not reply to or close the ticket yet.</strong></p>"
        "<p>Notes for the chair:</p>"
        "<p>The review may belong to a different paper.</p>"
        "<p><em>Posted by ConfMail. Review and edit before sending.</em></p>"
    )


def test_reciprocal_review_note_is_for_marc():
    html = build_chair_note_html(
        _draft("reciprocal_review", "[CHAIR: reciprocal complaint; see note]",
               notes="Reciprocal reviewer complaint."),
        None,
        ApcResolution(),
    )
    assert html == (
        "<p><strong>ConfMail draft (not sent to the author)</strong></p>"
        "<p>Chair: not identified from this email</p>"
        "<p>Paper number(s): none found in the email</p>"
        "<p><strong>Reciprocal complaint: for Marc to review.</strong></p>"
        "<p>Notes for the chair:</p>"
        "<p>Reciprocal reviewer complaint.</p>"
        "<p><em>Posted by ConfMail. Review and edit before sending.</em></p>"
    )


DO_NOT_SEND_EXPECTED = (
    "<p><strong>ConfMail draft (not sent to the author)</strong></p>"
    "<p>Chair: not identified from this email</p>"
    "<p>Paper number(s): none found in the email</p>"
    "<p><strong>Do not send: the chair writes this reply.</strong></p>"
    "<p>Notes for the chair:</p>"
    "<p>Chair writes.</p>"
    "<p><em>Posted by ConfMail. Review and edit before sending.</em></p>"
)


@pytest.mark.parametrize(
    "mode",
    ["chair_writes", "refused", "reason_unknown", "desk_reject", "window", "failed",
     "a_mode_added_later", None],
)
def test_every_other_mode_says_do_not_send(mode):
    html = build_chair_note_html(
        _draft(mode, "[CHAIR: write reply]", notes="Chair writes."), {}, ApcResolution()
    )
    assert html == DO_NOT_SEND_EXPECTED


@pytest.mark.parametrize(
    "mode", ["chair_writes", "refused", "window", "failed", "a_mode_added_later", None]
)
def test_only_composed_modes_offer_text_even_when_it_looks_sendable(mode):
    # Clean text with no placeholder: only the MODE may decide it is not a reply.
    html = build_chair_note_html(
        _draft(mode, "Dear Author,\n\nThis text has no placeholder."), {}, ApcResolution()
    )
    assert html == (
        "<p><strong>ConfMail draft (not sent to the author)</strong></p>"
        "<p>Chair: not identified from this email</p>"
        "<p>Paper number(s): none found in the email</p>"
        "<p><strong>Do not send: the chair writes this reply.</strong></p>"
        "<p><em>Posted by ConfMail. Review and edit before sending.</em></p>"
    )


@pytest.mark.parametrize("text", ["[CHAIR: write reply]", "Dear X,\n\n[CHAIR: add the date]", "", "   "])
def test_composed_mode_without_sendable_text_is_never_offered_for_copying(text):
    html = build_chair_note_html(
        _draft("merged", text, notes="Chair writes."), {}, ApcResolution()
    )
    assert html == DO_NOT_SEND_EXPECTED


def test_several_chairs_and_unmatched_forum_ids_are_all_listed():
    html = build_chair_note_html(
        _draft("chair_writes", "[CHAIR: write reply]"),
        {"submission_numbers": ["101", "202", "101"]},
        ApcResolution(
            apc_names=("APC North", "APC South"),
            forum_ids_matched=("Ab3xY9kLm2", "Zz9yY8xX77"),
            forum_ids_unmatched=("NotInSheet",),
        ),
    )
    assert html == (
        "<p><strong>ConfMail draft (not sent to the author)</strong></p>"
        "<p>Chair: APC North, APC South</p>"
        "<p>Paper number(s) as written in the email (unverified): 101, 202</p>"
        "<p>OpenReview: "
        '<a href="https://openreview.net/forum?id=Ab3xY9kLm2">https://openreview.net/forum?id=Ab3xY9kLm2</a>, '
        '<a href="https://openreview.net/forum?id=Zz9yY8xX77">https://openreview.net/forum?id=Zz9yY8xX77</a>, '
        '<a href="https://openreview.net/forum?id=NotInSheet">https://openreview.net/forum?id=NotInSheet</a></p>'
        "<p><strong>Do not send: the chair writes this reply.</strong></p>"
        "<p><em>Posted by ConfMail. Review and edit before sending.</em></p>"
    )


def test_every_outside_value_is_escaped():
    html = build_chair_note_html(
        _draft(
            "merged",
            'Dear <script>alert("x")</script>,\n\nBody & more.\n\nBest Regards,\nAAAI 2027 PC Team',
            notes="<b>bold</b> note",
        ),
        {"submission_numbers": ["12<3"]},
        ApcResolution(apc_names=("<script>steal()</script> & Co",)),
    )
    assert html == (
        "<p><strong>ConfMail draft (not sent to the author)</strong></p>"
        "<p>Chair: &lt;script&gt;steal()&lt;/script&gt; &amp; Co</p>"
        "<p>Paper number(s) as written in the email (unverified): 12&lt;3</p>"
        "<p><strong>COPY BELOW</strong></p>"
        "<hr>"
        "<p>Dear &lt;script&gt;alert(&quot;x&quot;)&lt;/script&gt;,</p>"
        "<p>Body &amp; more.</p>"
        "<p>Best Regards,<br>AAAI 2027 PC Team</p>"
        "<hr>"
        "<p><strong>END</strong></p>"
        "<p>Notes for the chair:</p>"
        "<p>&lt;b&gt;bold&lt;/b&gt; note</p>"
        "<p><em>Posted by ConfMail. Review and edit before sending.</em></p>"
    )
    assert "<script" not in html


def test_marker_is_the_first_line():
    html = build_chair_note_html(_draft("failed", "[CHAIR: write reply]"), {}, ApcResolution())
    assert html.startswith(HEADER)
    assert html.endswith(FOOTER)


def test_body_sha256_is_of_the_exact_utf8_bytes():
    # The standard SHA-256 test vectors (FIPS 180-2) for "abc" and "".
    assert note_body_sha256("abc") == (
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    )
    assert note_body_sha256("") == (
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    )


# --- agreement with the appeal reply hook -----------------------------------


def test_composed_modes_match_the_hook():
    from app.pipeline.appeal_reply_hook import COMPOSED_MODES

    assert chair_note.COMPOSED_MODES == COMPOSED_MODES


def test_mode_names_match_the_hooks_placeholders():
    from app.pipeline.appeal_reply_hook import (
        NO_DRAFT_PLACEHOLDER,
        RECIPROCAL_PLACEHOLDER,
        AppealReplyDecision,
        build_appeal_draft,
    )

    def text_for(mode):
        return build_appeal_draft(AppealReplyDecision(mode, (), (), None, ()), None).draft_text

    assert text_for(chair_note.MODE_NO_DRAFT) == NO_DRAFT_PLACEHOLDER
    assert text_for(chair_note.MODE_RECIPROCAL) == RECIPROCAL_PLACEHOLDER


# --- import isolation ----------------------------------------------------------

_NEW_MODULES = (
    "app.integrations.zendesk.chair_note",
    "app.pipeline.paper_apc_resolver",
    "app.repositories.chair_note_repository",
)


def test_startup_does_not_import_the_new_modules(tmp_path):
    """With the flag off, importing the app loads the model only."""
    probe = (
        "import json, sys\n"
        "import main\n"
        f"new = {list(_NEW_MODULES)!r}\n"
        "models = sys.modules.get('app.db.models')\n"
        "print(json.dumps({\n"
        "    'loaded': [m for m in new if m in sys.modules],\n"
        "    'model': bool(models) and hasattr(models, 'ZendeskChairNote'),\n"
        "}))\n"
    )
    env = {
        **os.environ,
        "DATABASE_URL": f"sqlite:///{(tmp_path / 'probe.db').as_posix()}",
        "CHAIR_NOTE_ENABLED": "False",
    }
    result = subprocess.run(
        [sys.executable, "-c", probe],
        cwd=str(BACKEND_ROOT),
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout.strip().splitlines()[-1])
    assert report == {"loaded": [], "model": True}
