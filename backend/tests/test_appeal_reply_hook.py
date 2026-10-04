"""Tests for the appeal reply hook (reject-appeal Phase 4, Step 4).

Decision + draft build, run against the REAL approved template file. Every
expected text is a hand-written literal. Placeholder drafts are also pushed
through the approve endpoint (must answer 409) and the send gate (must refuse).
No network, no model call.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from types import SimpleNamespace

import httpx
import pytest
import pytest_asyncio
from httpx import ASGITransport
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool

import main
from app.core.send_gate import authorize_send
from app.db.database import Base, get_db
from app.db.models import Email
from app.pipeline import appeal_reply_hook as hook
from app.pipeline.appeal_reply_hook import (
    SIGN_OFF,
    build_appeal_draft,
    decide_appeal_reply,
    prepare_appeal_draft,
    ticket_created_at,
)
from app.pipeline.drafter import find_placeholders

REVIEW, DESK = "review_decision_appeal", "desk_reject_appeal"
SCORE, REVIEWER, LLM = "score_outcome_mismatch", "reviewer_misunderstanding", "llm_generated_review"
WRONG, GENERAL, OTHER = "wrong_paper_review", "general_dissatisfaction", "other"
INSIDE = "2026-09-15T10:00:00Z"
WINDOW_END = datetime(2026, 11, 1, tzinfo=timezone.utc)

# --- the four composed drafts, written out in full -----------------------------------------
OPENING = (
    "We understand that this outcome may be disappointing, and we appreciate the effort you invested "
    "in preparing your submission. We would like to respond to your concerns:"
)
P_SCORES = (
    "(1) Decisions are not based solely on the visible reviewer scores. Senior program committee "
    "members evaluated both the paper and the reviews, including whether the raised concerns can be "
    "addressed with minor clarifications or require substantial revision."
)
P_REBUTTAL = (
    "AAAI's two-phase process forgoes rebuttal for Phase 1 papers in favor of a quicker decision. We "
    "understand this can be frustrating, but Phase 1 decisions are final and will not be revisited in "
    "response to author objections."
)
P_REVIEWERS = "Decisions are not based on any single review; all assessments are weighed together."
P_THANKS = (
    "Thank you for sharing your view of the review process. We will consider your input when studying "
    "possible changes for future editions."
)
CLOSING = (
    "The decision is final, but we hope the feedback will be useful in further strengthening your work "
    "and helping you secure publication in another leading venue or future AAAI edition."
)
SIGN = "Best Regards,\nAAAI 2027 PC Team"

DRAFT_T1 = f"Dear Jane Doe,\n\n{OPENING}\n\n{P_SCORES}\n\n(2) {P_REBUTTAL}\n\n{CLOSING}\n\n{SIGN}"
DRAFT_T2 = (f"Dear Author,\n\n{OPENING}\n\n(1) {P_REVIEWERS}\n\n(2) {P_REBUTTAL}\n\n(3) {P_THANKS}\n\n"
            f"{CLOSING}\n\n{SIGN}")
DRAFT_T1_T2 = (f"Dear Author,\n\n{OPENING}\n\n{P_SCORES}\n\n(2) {P_REVIEWERS}\n\n(3) {P_REBUTTAL}\n\n"
               f"(4) {P_THANKS}\n\n{CLOSING}\n\n{SIGN}")
DRAFT_YAN_A = (
    "Dear Wei Zhang,\n\n"
    "Thank you for taking the time to share your concerns regarding the review process for your "
    "submission. We take concerns about fairness in the review process seriously and have carefully "
    "considered the issues you raised.\n\n"
    "We understand that aspects of the reviews may have led you to feel that your submission did not "
    "receive fair consideration. We would like to emphasize that the final decision on a submission is "
    "not determined by any single review or reviewer.\n\n"
    "To further strengthen the consistency and fairness of the decision-making process, we have "
    "introduced a new Senior Program Chair (SPC) buddy system. Under this system, each SPC is paired "
    "with another SPC who serves as a \"buddy\" and independently reviews the SPC's recommendations. "
    "This additional layer of cross-checking is intended to reduce the influence of any individual "
    "assessment and promote greater consistency and fairness across the review process. In addition, "
    "the Area Chair is asked to review the recommendations by SPCs. When concerns arise regarding the "
    "quality of a particular review, these concerns are taken into account when the reviews are "
    "weighed as part of the overall assessment.\n\n"
    "Your feedback is valuable to us, not only in reviewing the process surrounding your submission "
    "but also in identifying areas where the review process can be improved. We will document the "
    "concerns raised and share relevant information with future AAAI Program Chairs to support "
    "continued improvements in the quality, fairness, and integrity of the review process. We "
    "appreciate your engagement with the process and your effort in bringing these concerns to our "
    "attention.\n\n"
    f"{SIGN}"
)
# Yan's AI-review reply (approved 2026-10-05 exactly as written): greeting, her four
# paragraphs, then the common sign-off.
YAN_B_PARAGRAPHS = (
    "Thank you for providing the detailed information regarding your concerns about the reviews of "
    "your submission. We take concerns about the integrity and quality of the review process seriously "
    "and have carefully considered the issues you raised.",
    "We recognize that some characteristics of a review may raise concerns about the possible use of "
    "AI tools. But rest assured that the decision on your submission does not rely on any single "
    "review. The Senior Program Chair and/or Area Chair have also reviewed the paper, considered the "
    "reviews and the authors' responses, and formed their own assessment of the submission. The final "
    "decision is made based on this broader evaluation rather than on the assessment or "
    "recommendation of any individual reviewer.",
    "In addition, we ask Senior Program Chairs to assess the quality of the reviews and provide "
    "feedback on the reviewers, including identifying reviews that exhibit characteristics associated "
    "with AI-generated content. Your feedback is also very valuable to us. We will document these "
    "concerns and share the relevant information with future AAAI Program Chairs to help further "
    "improve the quality and integrity of the review process.",
    "Thank you again for raising your concerns and for providing the supporting details. We "
    "appreciate your engagement with the review process.",
)
DRAFT_YAN_B = (
    "Dear Ana Silva,\n\n"
    "Thank you for providing the detailed information regarding your concerns about the reviews of "
    "your submission. We take concerns about the integrity and quality of the review process seriously "
    "and have carefully considered the issues you raised.\n\n"
    "We recognize that some characteristics of a review may raise concerns about the possible use of "
    "AI tools. But rest assured that the decision on your submission does not rely on any single "
    "review. The Senior Program Chair and/or Area Chair have also reviewed the paper, considered the "
    "reviews and the authors' responses, and formed their own assessment of the submission. The final "
    "decision is made based on this broader evaluation rather than on the assessment or "
    "recommendation of any individual reviewer.\n\n"
    "In addition, we ask Senior Program Chairs to assess the quality of the reviews and provide "
    "feedback on the reviewers, including identifying reviews that exhibit characteristics associated "
    "with AI-generated content. Your feedback is also very valuable to us. We will document these "
    "concerns and share the relevant information with future AAAI Program Chairs to help further "
    "improve the quality and integrity of the review process.\n\n"
    "Thank you again for raising your concerns and for providing the supporting details. We "
    "appreciate your engagement with the review process.\n\n"
    "Best Regards,\nAAAI 2027 PC Team"
)

NOTE_NO_DRAFT = ("Investigate first: the author says a review is about a different paper. "
                 "Do not reply to or close the ticket yet.")
NOTE_RECIP = "Reciprocal-review complaint: tagged for Marc to review himself. No reply is drafted."
NOTE_REASON = "Chair writes: the appeal reason was not determined, so no approved reply could be chosen."
NOTE_DESK = ("Chair writes: desk-rejection appeals get no composed reply, because the approved wording "
             "assumes the paper was reviewed.")
NOTE_WINDOW = "Appeal reply wording is for Phase 1 rejections only"


def prepare(intent, reasons, *, reciprocal=None, created=INSIDE, name=None, window_end=None):
    data = {"subject": "s", "body": "b"}
    if created is not None:
        data["timestamp"] = created
    if name is not None:
        data["sender_name"] = name
    return prepare_appeal_draft(intent, reasons, reciprocal, data, window_end=window_end)


# --- composed drafts --------------------------------------------------------------------------
@pytest.mark.parametrize("reasons, name, expected, record", [
    ([SCORE], "Jane Doe", DRAFT_T1,
     {"mode": "merged", "reasons": [SCORE],
      "block_ids": ["opening_warm", "lead_in_concerns", "point_scores", "point_rebuttal", "closing_reviewed"]}),
    ([REVIEWER], None, DRAFT_T2,
     {"mode": "merged", "reasons": [REVIEWER],
      "block_ids": ["opening_warm", "lead_in_concerns", "point_all_assessments", "point_rebuttal",
                    "point_consider_input", "closing_reviewed"]}),
    ([SCORE, REVIEWER], "   ", DRAFT_T1_T2,
     {"mode": "merged", "reasons": [SCORE, REVIEWER],
      "block_ids": ["opening_warm", "lead_in_concerns", "point_scores", "point_all_assessments",
                    "point_rebuttal", "point_consider_input", "closing_reviewed"]}),
    ([GENERAL], "Wei Zhang", DRAFT_YAN_A,
     {"mode": "standalone", "reasons": [GENERAL], "block_ids": ["standalone_general_stage1"]}),
    ([LLM], "Ana Silva", DRAFT_YAN_B,
     {"mode": "standalone", "reasons": [LLM], "block_ids": ["standalone_ai_review"]}),
], ids=["T1-scores", "T2-reviewer", "T1+T2", "yan-a", "yan-b"])
def test_composed_drafts_through_the_real_approved_file(reasons, name, expected, record):
    draft, rec = prepare(REVIEW, reasons, name=name)
    assert draft.draft_text == expected
    assert rec == record
    assert (draft.notes_for_chair, draft.placeholders, draft.citations) == (None, [], [])
    assert draft.model_used == "none"
    assert draft.generation_metadata == {"provider": "appeal_reply_composer", "appeal_mode": record["mode"]}


def test_the_sign_off_is_exactly_the_common_one():
    assert SIGN_OFF == "Best Regards,\nAAAI 2027 PC Team"
    draft, _ = prepare(REVIEW, [SCORE])
    assert draft.draft_text.endswith("\n\nBest Regards,\nAAAI 2027 PC Team")
    assert "[Sender name]" not in draft.draft_text and "Marc" not in draft.draft_text


@pytest.mark.parametrize("name, greeting", [
    ("Jane Doe", "Dear Jane Doe,"), (None, "Dear Author,"), ("", "Dear Author,"),
    ("   ", "Dear Author,"), ("  Ana \n  Silva ", "Dear Ana Silva,"),
])
def test_the_greeting_uses_the_requester_name_or_author(name, greeting):
    draft, _ = prepare(REVIEW, [SCORE], name=name)
    assert draft.draft_text.split("\n\n")[0] == greeting


def test_a_merged_reply_with_other_keeps_the_chair_line_and_cannot_be_sent():
    draft, rec = prepare(REVIEW, [SCORE, OTHER])
    assert "\n\n[CHAIR: write reply]\n\n" + CLOSING in draft.draft_text
    assert draft.placeholders == ["write reply"]
    assert rec["mode"] == "merged"


# --- placeholder cases --------------------------------------------------------------------------
PLACEHOLDER_CASES = [
    # id, intent, reasons, reciprocal, created, window_end, text, notes, mode
    ("wrong-paper", REVIEW, [WRONG], None, INSIDE, None,
     "[CHAIR: do not reply yet; see note]", NOTE_NO_DRAFT, "no_draft"),
    ("wrong-paper+score", REVIEW, [WRONG, SCORE], None, INSIDE, None,
     "[CHAIR: do not reply yet; see note]", f"{NOTE_NO_DRAFT}\nAlso raised: score_outcome_mismatch.", "no_draft"),
    ("reciprocal", DESK, None, True, INSIDE, None,
     "[CHAIR: reciprocal complaint; see note]", NOTE_RECIP, "reciprocal_review"),
    ("reciprocal-with-reason", REVIEW, [SCORE], True, INSIDE, None,
     "[CHAIR: reciprocal complaint; see note]", f"{NOTE_RECIP}\nAlso raised: score_outcome_mismatch.",
     "reciprocal_review"),
    ("other", REVIEW, [OTHER], None, INSIDE, None, "[CHAIR: write reply]", None, "chair_writes"),
    ("reason-none", REVIEW, None, None, INSIDE, None, "[CHAIR: write reply]", NOTE_REASON, "reason_unknown"),
    ("reason-empty", REVIEW, [], None, INSIDE, None, "[CHAIR: write reply]", NOTE_REASON, "reason_unknown"),
    ("desk-reject", DESK, [OTHER], False, INSIDE, None, "[CHAIR: write reply]", NOTE_DESK, "desk_reject"),
    ("desk-reject-reciprocal-unknown", DESK, [SCORE], None, INSIDE, None,
     "[CHAIR: write reply]", NOTE_DESK, "desk_reject"),
    # Yan's AI-review reply is approved (2026-10-05), so an AI-review reason alone is
    # composed (see the composed drafts above); mixed with another reason the chair
    # still writes (D105, unchanged).
    ("yan-b+reviewer", REVIEW, [LLM, REVIEWER], None, INSIDE, None, "[CHAIR: write reply]",
     "Chair writes: no approved reply covers these reasons together: "
     "reviewer_misunderstanding, llm_generated_review.", "chair_writes"),
    ("yan-a+score", REVIEW, [GENERAL, SCORE], None, INSIDE, None, "[CHAIR: write reply]",
     "Chair writes: no approved reply covers these reasons together: "
     "score_outcome_mismatch, general_dissatisfaction.", "chair_writes"),
    ("outside-window", REVIEW, [SCORE], None, "2026-12-01T10:00:00Z", WINDOW_END,
     "[CHAIR: write reply]", NOTE_WINDOW, "window"),
    ("unknown-created-with-window", REVIEW, [SCORE], None, None, WINDOW_END,
     "[CHAIR: write reply]", NOTE_WINDOW, "window"),
    ("yan-a-outside-window", REVIEW, [GENERAL], None, "2026-12-01T10:00:00Z", WINDOW_END,
     "[CHAIR: write reply]", NOTE_WINDOW, "window"),
    ("yan-b-outside-window", REVIEW, [LLM], None, "2026-12-01T10:00:00Z", WINDOW_END,
     "[CHAIR: write reply]", NOTE_WINDOW, "window"),
    # The window replaces ONLY composed text; these keep their own notes.
    ("wrong-paper-outside-window", REVIEW, [WRONG], None, "2026-12-01T10:00:00Z", WINDOW_END,
     "[CHAIR: do not reply yet; see note]", NOTE_NO_DRAFT, "no_draft"),
    ("reciprocal-outside-window", DESK, None, True, None, WINDOW_END,
     "[CHAIR: reciprocal complaint; see note]", NOTE_RECIP, "reciprocal_review"),
    ("other-outside-window", REVIEW, [OTHER], None, "2026-12-01T10:00:00Z", WINDOW_END,
     "[CHAIR: write reply]", None, "chair_writes"),
]


@pytest.mark.parametrize("intent, reasons, reciprocal, created, window_end, text, notes, mode",
                         [c[1:] for c in PLACEHOLDER_CASES], ids=[c[0] for c in PLACEHOLDER_CASES])
def test_placeholder_cases(intent, reasons, reciprocal, created, window_end, text, notes, mode):
    draft, rec = prepare(intent, reasons, reciprocal=reciprocal, created=created, window_end=window_end)
    assert draft.draft_text == text
    assert draft.notes_for_chair == notes
    assert rec["mode"] == mode
    assert draft.placeholders == find_placeholders(text) and draft.placeholders
    assert draft.model_used == "none"
    assert not (draft.notes_for_chair or "").startswith("WARNING (automated check):")
    # The send gate refuses it, even once approved.
    for status in ("DRAFT_GENERATED", "approved"):
        decision = authorize_send(SimpleNamespace(status=status, routing={"lane": "human_review"},
                                                  draft={"draft_text": draft.draft_text}))
        assert decision.authorized is False and "placeholder" in decision.reason


def test_inside_the_window_the_composed_text_is_used():
    draft, rec = prepare(REVIEW, [SCORE], name="Jane Doe", window_end=WINDOW_END)
    assert (draft.draft_text, rec["mode"]) == (DRAFT_T1, "merged")


def test_exactly_at_the_window_end_the_composed_text_is_used():
    draft, rec = prepare(REVIEW, [SCORE], created="2026-11-01T00:00:00Z", window_end=WINDOW_END)
    assert rec["mode"] == "merged"


def test_a_naive_window_end_is_read_as_utc():
    naive = datetime(2026, 11, 1)
    assert prepare(REVIEW, [SCORE], created="2026-10-31T23:00:00Z", window_end=naive)[1]["mode"] == "merged"
    assert prepare(REVIEW, [SCORE], created="2026-11-01T01:00:00Z", window_end=naive)[1]["mode"] == "window"


def test_the_record_for_a_placeholder():
    assert prepare(REVIEW, None)[1] == {"mode": "reason_unknown", "reasons": None, "block_ids": []}
    assert prepare(REVIEW, [WRONG])[1] == {"mode": "no_draft", "reasons": [WRONG], "block_ids": []}
    assert prepare(DESK, None, reciprocal=True)[1] == {
        "mode": "reciprocal_review", "reasons": ["reciprocal_dispute"], "block_ids": []}


# --- scope and failure ---------------------------------------------------------------------------
@pytest.mark.parametrize("intent", ["submission_requirements", "cms_support", None, ""])
def test_a_non_appeal_intent_is_not_handled(intent):
    assert prepare(intent, [SCORE]) is None
    assert decide_appeal_reply(intent, [SCORE], None, None) is None


def test_any_failure_inside_gives_the_chair_writes_placeholder(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("synthetic failure")

    monkeypatch.setattr(hook, "compose_reply", boom)
    draft, rec = prepare(REVIEW, [SCORE])
    assert draft.draft_text == "[CHAIR: write reply]"
    assert draft.notes_for_chair == "Chair writes: the appeal reply could not be prepared automatically."
    assert rec == {"mode": "failed", "reasons": [SCORE], "block_ids": []}


def test_a_failure_for_a_non_appeal_intent_is_still_not_handled(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("synthetic failure")

    monkeypatch.setattr(hook, "decide_appeal_reply", boom)
    assert prepare("submission_requirements", [SCORE]) is None


@pytest.mark.parametrize("data, expected", [
    ({"timestamp": "2026-09-15T10:00:00Z"}, datetime(2026, 9, 15, 10, tzinfo=timezone.utc)),
    ({"ticket_created_at": datetime(2026, 9, 1)}, datetime(2026, 9, 1, tzinfo=timezone.utc)),
    ({"ticket_created_at": datetime(2026, 9, 1, tzinfo=timezone.utc), "timestamp": "2026-12-01T00:00:00Z"},
     datetime(2026, 9, 1, tzinfo=timezone.utc)),
    ({"timestamp": "not a date"}, None),
    ({"timestamp": ""}, None),
    ({}, None),
])
def test_ticket_created_at_reads_only_email_data(data, expected):
    assert ticket_created_at(data) == expected


def test_build_appeal_draft_never_uses_the_middle_of_a_non_composed_mode():
    d = hook.AppealReplyDecision("chair_writes", ("other",), ("line_chair_writes",), "IGNORED", ())
    assert build_appeal_draft(d, "Jane").draft_text == "[CHAIR: write reply]"


# --- Yan's AI-review reply, end to end (approved 2026-10-05) ---------------------------------------
def test_the_ai_review_draft_is_greeting_then_yans_four_paragraphs_then_the_sign_off():
    draft, rec = prepare(REVIEW, [LLM], name="Ana Silva")
    parts = draft.draft_text.split("\n\n")
    assert parts == ["Dear Ana Silva,", *YAN_B_PARAGRAPHS, "Best Regards,\nAAAI 2027 PC Team"]
    assert draft.draft_text.endswith("\n\nBest Regards,\nAAAI 2027 PC Team")
    assert (draft.placeholders, draft.notes_for_chair, draft.citations, draft.model_used) == ([], None, [], "none")
    assert rec == {"mode": "standalone", "reasons": [LLM], "block_ids": ["standalone_ai_review"]}


def test_a_refused_composition_still_gives_the_chair_writes_placeholder_with_its_reason(tmp_path):
    """The hook's "refused" branch, which only Yan B's blocker used to reach in the
    real file: a copy whose AI-review block is back to draft."""
    data = json.loads(hook.DEFAULT_PATH.read_text(encoding="utf-8"))
    for e in data["templates"]:
        if e["id"] == "standalone_ai_review":
            e.update(status="draft", approved_by=None, approved_at=None, approved_sha256=None)
    path = tmp_path / "templates.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    draft, rec = prepare_appeal_draft(REVIEW, [LLM], None, {"timestamp": INSIDE}, path=path)
    assert draft.draft_text == "[CHAIR: write reply]"
    assert draft.notes_for_chair == (
        "Chair writes: no approved reply could be composed (missing_approved_block:standalone_ai_review).")
    assert rec == {"mode": "refused", "reasons": [LLM], "block_ids": []}


# --- approve returns 409 on every placeholder draft ---------------------------------------------
@pytest_asyncio.fixture
async def client():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:",
                                 connect_args={"check_same_thread": False}, poolclass=StaticPool)
    factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    async def _override_get_db():
        async with factory() as s:
            yield s

    main.app.dependency_overrides[get_db] = _override_get_db
    async with httpx.AsyncClient(transport=ASGITransport(app=main.app), base_url="http://test") as c:
        c.factory = factory
        yield c
    main.app.dependency_overrides.clear()
    await engine.dispose()


@pytest.mark.parametrize("intent, reasons, reciprocal, created, window_end, text, notes, mode",
                         [c[1:] for c in PLACEHOLDER_CASES], ids=[c[0] for c in PLACEHOLDER_CASES])
async def test_approve_returns_409_for_every_placeholder_draft(
    client, intent, reasons, reciprocal, created, window_end, text, notes, mode
):
    draft, rec = prepare(intent, reasons, reciprocal=reciprocal, created=created, window_end=window_end)
    stored = {**draft.model_dump(), "appeal_reply": rec}
    async with client.factory() as s:
        email = Email(sender="author@example.org", subject="Appeal", body="b", status="DRAFT_GENERATED",
                      routing={"lane": "human_review", "reason": "x"},
                      classification={"intent": intent, "confidence": 0.9}, draft=stored)
        s.add(email)
        await s.commit()
        await s.refresh(email)
        email_id = email.id
    resp = await client.patch(f"/api/v1/emails/{email_id}/approve", json={"approved_by": "chair"})
    assert resp.status_code == 409
    assert resp.json()["detail"]["placeholders"] == find_placeholders(text)
