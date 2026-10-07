"""The flagged AI suggestion in the appeal reply hook (reject-appeal Phase 4, 2b).

``appeal_ai_draft.apply_ai_suggestion`` after ``prepare_appeal_draft``: when it
triggers, what the draft looks like, that it can be neither approved nor sent
unedited, that learning skips it, that every failure leaves the plain hook
result, and that the author's text never reaches a log. The model is always a
mock: either ``suggest_appeal_middle`` itself or its ``_call_model``.
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone
from types import SimpleNamespace

import httpx
import pytest
import pytest_asyncio
from httpx import ASGITransport
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool

import main
from app.api.v1 import emails as emails_api
from app.core.config import Settings, settings
from app.core.send_gate import authorize_send
from app.db.database import Base, get_db
from app.db.models import Email
from app.pipeline import appeal_ai_draft as aid
from app.pipeline import appeal_ai_suggestion as ais
from app.pipeline import appeal_reply_hook as hook
from app.pipeline import orchestrator as orch
from app.pipeline.appeal_ai_suggestion import SuggestionOutcome
from app.pipeline.appeal_reply_hook import SIGN_OFF, prepare_appeal_draft
from app.pipeline.drafter import find_placeholders
from app.pipeline.phase1_appeal_classifier import AppealReason, Phase1AppealResult
from app.pipeline.phase1_appeal_outcome import (
    CLASSIFIED,
    FAILED,
    GATE_NOT_MET,
    NOT_APPEAL,
    Phase1Outcome,
)
from app.pipeline.phase1_reply_mapping import map_phase1
from tests.test_appeal_reply_hook_wiring import (  # noqa: F401 - session is a fixture
    _Phase1Stub,
    _pipeline,
    _run_all_four,
    _SpyDrafter,
    phase1_result,
    session,
)

REVIEW, DESK = "review_decision_appeal", "desk_reject_appeal"
SENTINEL = "EMAIL-SENTINEL-2b-71c4"
FLAG = "[CHAIR: AI-written suggestion, not approved wording; review and edit before sending]"
FLAG_PLACEHOLDER = "AI-written suggestion, not approved wording; review and edit before sending"
PROMPT_SHA = "f6c217cefd5f49254fb000b9a98d944dc1f920316f0a6a55d979e652feee211f"
V_MISC = ("Before sending, check for harassment or an undisclosed conflict of interest. "
          "If either is present, do not send; forward the ticket to the Ethics Chairs.")
MIDDLE = ("We understand that this outcome may be disappointing, and we appreciate the effort you "
          "invested in preparing your submission. We would like to respond to your concerns:\n\n"
          "(1) Decisions are not based on any single review or on the visible scores alone. SPCs "
          "evaluated both the paper and the reviews, and all assessments were weighed together.\n\n"
          "The decision is final, but we hope the feedback will be useful in further strengthening "
          "your work and helping you secure publication in another leading venue or future AAAI "
          "edition. Thank you for raising your concerns; we will document them and help improve the "
          "future AAAI editions.")
SOURCE_BLOCKS = ("opening_warm", "lead_in_concerns", "point_review_process", "closing_reviewed")
EMAIL_DATA = {"subject": "Appeal", "body": f"Please reconsider. {SENTINEL}", "sender_name": "Jane Doe",
              "timestamp": "2026-09-15T10:00:00Z"}


def classified(reasons, relation="appeal", papers=("12345",)) -> Phase1Outcome:
    return Phase1Outcome(CLASSIFIED, Phase1AppealResult(
        relation=relation, papers=list(papers), dropped_unquoted=[],
        reasons=[AppealReason(reason=r, quote=f"{SENTINEL} quote") for r in reasons]))


def hook_result(outcome, *, intent=REVIEW, reciprocal=False, created="2026-09-15T10:00:00Z",
                window_end=None, phase1=True, path=None):
    data = {**EMAIL_DATA, "timestamp": created}
    kwargs = {} if path is None else {"path": path}
    return prepare_appeal_draft(intent, None, reciprocal, data, window_end=window_end,
                                mapped=map_phase1(outcome) if phase1 else None, **kwargs)


class FakeSuggest:
    """Stands in for suggest_appeal_middle: records calls, returns ``outcome``."""

    def __init__(self, outcome=SuggestionOutcome(MIDDLE, SOURCE_BLOCKS)):
        self.outcome = outcome
        self.calls = []

    async def __call__(self, email_data, reasons, **kwargs):
        self.calls.append(list(reasons))
        return self.outcome


@pytest.fixture
def flag_on(monkeypatch):
    monkeypatch.setattr(settings, "APPEAL_AI_SUGGESTION_ENABLED", True)


# --- the flag -------------------------------------------------------------------------------
def test_the_flag_defaults_to_off():
    assert Settings.model_fields["APPEAL_AI_SUGGESTION_ENABLED"].default is False


def test_the_trigger_modes_are_exactly_three():
    assert aid.TRIGGER_MODES == frozenset({"chair_writes", "refused", "reason_unknown"})


# --- trigger matrix: every hook mode x flag on/off, on real hook outcomes ---------------------
def _refused_path(tmp_path):
    data = json.loads(hook.DEFAULT_PATH.read_text(encoding="utf-8"))
    for e in data["templates"]:
        if e["id"] == "standalone_ai_review":
            e.update(status="draft", approved_by=None, approved_at=None, approved_sha256=None)
    p = tmp_path / "t.json"
    p.write_text(json.dumps(data), encoding="utf-8")
    return p


MATRIX = [
    # id, kwargs for hook_result, expected mode, triggers
    ("merged", dict(outcome=classified(["decision_vs_reviews"])), "merged", False),
    ("standalone", dict(outcome=classified(["llm_generated_review"])), "standalone", False),
    ("chair-writes-yan-mix", dict(outcome=classified(["llm_generated_review", "decision_vs_reviews"])),
     "chair_writes", True),
    ("chair-writes-cap", dict(outcome=classified(["record_error", "decision_vs_reviews",
                                                  "missing_material_claim", "reviewer_misconduct"])),
     "chair_writes", True),
    ("refused", dict(outcome=classified(["llm_generated_review"]), path="REFUSED"), "refused", True),
    ("reason-unknown-classifier-failed", dict(outcome=Phase1Outcome(FAILED)), "reason_unknown", True),
    ("reason-unknown-no-verified-reason", dict(outcome=classified([])), "reason_unknown", True),
    ("reason-unknown-flag-off", dict(outcome=None), "reason_unknown", False),
    ("reason-unknown-gate-not-met", dict(outcome=Phase1Outcome(GATE_NOT_MET)), "reason_unknown", False),
    ("reason-unknown-rollback-source", dict(outcome=None, phase1=False), "reason_unknown", False),
    ("no-draft", dict(outcome=classified(["wrong_paper_review"])), "no_draft", False),
    ("reciprocal", dict(outcome=classified(["decision_vs_reviews"]), reciprocal=True),
     "reciprocal_review", False),
    ("desk-reject", dict(outcome=Phase1Outcome(GATE_NOT_MET), intent=DESK), "desk_reject", False),
    ("not-appeal", dict(outcome=Phase1Outcome(NOT_APPEAL)), "not_appeal", False),
    ("window", dict(outcome=classified(["decision_vs_reviews"]), created="2026-12-01T10:00:00Z",
                    window_end=datetime(2026, 11, 1, tzinfo=timezone.utc)), "window", False),
    # the exclusions (all chair_writes holds)
    ("excluded-feedback-only", dict(outcome=classified(["reviewer_misjudgment"], relation="feedback_only")),
     "chair_writes", False),
    ("excluded-other", dict(outcome=classified(["other"])), "chair_writes", False),
    ("excluded-other-with-composable", dict(outcome=classified(["decision_vs_reviews", "other"])),
     "chair_writes", False),
    ("excluded-unknown-name", dict(outcome=classified(["future_reason"])), "chair_writes", False),
    ("excluded-two-papers", dict(outcome=classified(["decision_vs_reviews"], papers=("1111", "2222"))),
     "chair_writes", False),
    ("excluded-two-papers-no-reason", dict(outcome=classified([], papers=("1111", "2222"))),
     "chair_writes", False),
]


@pytest.mark.parametrize("flag", [False, True], ids=["flag-off", "flag-on"])
@pytest.mark.parametrize("kwargs, mode, triggers", [c[1:] for c in MATRIX], ids=[c[0] for c in MATRIX])
def test_trigger_matrix(monkeypatch, tmp_path, flag, kwargs, mode, triggers):
    monkeypatch.setattr(settings, "APPEAL_AI_SUGGESTION_ENABLED", flag)
    kwargs = dict(kwargs)
    if kwargs.get("path") == "REFUSED":
        kwargs["path"] = _refused_path(tmp_path)
    intent = kwargs.get("intent", REVIEW)
    draft, record = hook_result(**kwargs)
    assert record["mode"] == mode
    assert aid.should_suggest(intent, record) is triggers
    fake = FakeSuggest()
    monkeypatch.setattr(aid, "suggest_appeal_middle", fake)
    new_draft, new_record = asyncio.run(aid.apply_ai_suggestion(intent, draft, record, EMAIL_DATA))
    if flag and triggers:
        assert len(fake.calls) == 1
        assert new_record["mode"] == "ai_suggestion"
        assert new_record["ai_suggestion"]["base_mode"] == mode
    else:
        assert fake.calls == []
        assert (new_draft, new_record) == (draft, record)
        assert new_draft is draft and new_record is record


@pytest.mark.parametrize("mode", ["merged", "standalone", "no_draft", "reciprocal_review",
                                  "desk_reject", "not_appeal", "window", "failed", "ai_suggestion"])
def test_no_other_mode_triggers_even_with_a_perfect_snapshot(mode):
    snapshot = {"state": "classified", "relation": "appeal",
                "reasons": ["llm_generated_review", "decision_vs_reviews"],
                "must_verify": False, "papers": ["12345"]}
    record = {"mode": mode, "reasons": None, "block_ids": [], "source": "phase1", "phase1": snapshot}
    assert aid.should_suggest(REVIEW, record) is False
    assert aid.should_suggest(REVIEW, {**record, "mode": "chair_writes"}) is True


@pytest.mark.parametrize("source", [None, "appeal_reason", "PHASE1"], ids=["no-source", "rollback", "case"])
def test_never_without_the_phase1_source_even_with_a_perfect_snapshot(source):
    snapshot = {"state": "classified", "relation": "appeal",
                "reasons": ["llm_generated_review", "decision_vs_reviews"],
                "must_verify": False, "papers": ["12345"]}
    record = {"mode": "chair_writes", "reasons": None, "block_ids": [], "phase1": snapshot}
    if source is not None:
        record["source"] = source
    assert aid.should_suggest(REVIEW, record) is False
    assert aid.should_suggest(REVIEW, {**record, "source": "phase1"}) is True


@pytest.mark.parametrize("intent", [DESK, "submission_requirements", None])
def test_only_review_decision_appeals(intent):
    _, record = hook_result(classified(["llm_generated_review", "decision_vs_reviews"]))
    assert aid.should_suggest(REVIEW, record) is True
    assert aid.should_suggest(intent, record) is False


# --- the draft --------------------------------------------------------------------------------
def test_the_suggestion_draft_record_and_notes(monkeypatch, flag_on):
    draft, record = hook_result(classified(["reviewer_misconduct", "llm_generated_review"]))
    assert record["mode"] == "chair_writes"
    fake = FakeSuggest()
    monkeypatch.setattr(aid, "suggest_appeal_middle", fake)
    new_draft, new_record = asyncio.run(aid.apply_ai_suggestion(REVIEW, draft, record, EMAIL_DATA))

    assert new_draft.draft_text == f"{FLAG}\n\nDear Jane Doe,\n\n{MIDDLE}\n\n{SIGN_OFF}"
    assert aid.AI_FLAG_LINE == FLAG
    assert new_draft.placeholders == [FLAG_PLACEHOLDER] == find_placeholders(new_draft.draft_text)
    assert new_draft.notes_for_chair == draft.notes_for_chair
    assert V_MISC in new_draft.notes_for_chair, "the verify notes are kept"
    assert new_draft.answer_confidence is None and new_draft.citations == []
    assert new_record == {
        **record, "mode": "ai_suggestion", "block_ids": list(SOURCE_BLOCKS),
        "ai_suggestion": {"base_mode": "chair_writes", "prompt_sha256": PROMPT_SHA,
                          "model": aid.active_model_id()},
    }
    # The model sees his reason NAMES from the snapshot, never a quote.
    assert fake.calls == [["reviewer_misconduct", "llm_generated_review"]]


@pytest.mark.parametrize("outcome", [Phase1Outcome(FAILED), classified([])],
                         ids=["classifier-failed", "no-verified-reason"])
def test_no_reasons_reach_the_model_as_an_empty_list(monkeypatch, flag_on, outcome):
    draft, record = hook_result(outcome)
    fake = FakeSuggest()
    monkeypatch.setattr(aid, "suggest_appeal_middle", fake)
    asyncio.run(aid.apply_ai_suggestion(REVIEW, draft, record, EMAIL_DATA))
    assert fake.calls == [[]]


def test_an_empty_reason_list_reads_not_determined_in_the_user_message():
    from app.pipeline.appeal_ai_suggestion_prompt import build_user_message

    assert "not determined" in build_user_message("s", "b", None, [], ())


# --- approve and send are blocked; learning is skipped ----------------------------------------
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


async def _store(client, draft, record) -> int:
    async with client.factory() as s:
        email = Email(sender="author@example.org", subject="Appeal", body="b", status="DRAFT_GENERATED",
                      routing={"lane": "human_review", "reason": "x"},
                      classification={"intent": REVIEW, "confidence": 0.9},
                      draft={**draft.model_dump(), "appeal_reply": record})
        s.add(email)
        await s.commit()
        await s.refresh(email)
        return email.id


async def _ai_draft(monkeypatch):
    monkeypatch.setattr(settings, "APPEAL_AI_SUGGESTION_ENABLED", True)
    monkeypatch.setattr(aid, "suggest_appeal_middle", FakeSuggest())
    draft, record = hook_result(classified(["llm_generated_review", "decision_vs_reviews"]))
    return await aid.apply_ai_suggestion(REVIEW, draft, record, EMAIL_DATA)


async def test_approve_answers_409_on_the_flag_line(client, monkeypatch):
    draft, record = await _ai_draft(monkeypatch)
    email_id = await _store(client, draft, record)
    resp = await client.patch(f"/api/v1/emails/{email_id}/approve", json={"approved_by": "chair"})
    assert resp.status_code == 409
    assert resp.json()["detail"]["placeholders"] == [FLAG_PLACEHOLDER]


async def test_the_send_gate_refuses_the_flagged_draft_even_once_approved(monkeypatch):
    draft, _ = await _ai_draft(monkeypatch)
    for status in ("DRAFT_GENERATED", "approved"):
        decision = authorize_send(SimpleNamespace(status=status, routing={"lane": "human_review"},
                                                  draft={"draft_text": draft.draft_text}))
        assert decision.authorized is False and "placeholder" in decision.reason


@pytest.fixture
def learner(monkeypatch):
    calls = []

    async def record(email_id):
        calls.append(email_id)

    monkeypatch.setattr(emails_api, "_learn_from_edit_bg", record)
    return calls


async def test_approving_an_edited_ai_suggestion_never_schedules_learning(client, monkeypatch, learner):
    draft, record = await _ai_draft(monkeypatch)
    email_id = await _store(client, draft, record)
    edited = draft.draft_text.replace(f"{FLAG}\n\n", "")
    resp = await client.patch(f"/api/v1/emails/{email_id}/approve",
                              json={"approved_by": "chair", "final_text": edited})
    assert resp.status_code == 200
    assert learner == []


async def test_other_modes_still_schedule_learning_after_a_gap_is_filled(client, learner):
    draft, record = hook_result(classified(["llm_generated_review", "decision_vs_reviews"]))
    assert record["mode"] == "chair_writes"
    email_id = await _store(client, draft, record)
    resp = await client.patch(f"/api/v1/emails/{email_id}/approve",
                              json={"approved_by": "chair", "final_text": "A real reply."})
    assert resp.status_code == 200
    assert learner == [str(email_id)]


# --- failures leave the plain hook result; nothing from the email reaches a log -----------------
async def _slow(user):
    await asyncio.sleep(5)


async def _boom(user):
    raise RuntimeError(f"transport failed {SENTINEL}")


@pytest.mark.parametrize("model, failure", [
    (lambda: "NONE", "none_answer"),
    (lambda: f"INTRO: {SENTINEL} said this.\nPOINT: Something new.\nOUTRO: Bye.", "foreign_sentence"),
    ("boom", "error"),
    ("slow", "timeout"),
], ids=["none", "foreign-text", "raising-model", "timeout"])
def test_every_failure_keeps_the_placeholder_and_logs_no_email_text(
        monkeypatch, caplog, flag_on, model, failure):
    if model == "boom":
        call = _boom
    elif model == "slow":
        call = _slow
    else:
        async def call(user, _model=model):
            return _model()
    monkeypatch.setattr(ais, "_call_model", call)
    draft, record = hook_result(classified(["llm_generated_review", "decision_vs_reviews"]))
    caplog.set_level(logging.DEBUG)
    new_draft, new_record = asyncio.run(
        aid.apply_ai_suggestion(REVIEW, draft, record, EMAIL_DATA, timeout=0.2))
    assert (new_draft, new_record) == (draft, record)
    assert new_draft.draft_text == "[CHAIR: write reply]"
    assert SENTINEL not in caplog.text
    assert f"Appeal AI suggestion dropped: {failure}" in caplog.text
    assert "Appeal AI suggestion model calls: 1" in caplog.text


def test_a_raising_suggestion_keeps_the_placeholder(monkeypatch, caplog, flag_on):
    async def boom(email_data, reasons, **kwargs):
        raise RuntimeError(SENTINEL)

    monkeypatch.setattr(aid, "suggest_appeal_middle", boom)
    draft, record = hook_result(Phase1Outcome(FAILED))
    caplog.set_level(logging.DEBUG)
    assert asyncio.run(aid.apply_ai_suggestion(REVIEW, draft, record, EMAIL_DATA)) == (draft, record)
    assert SENTINEL not in caplog.text
    assert "Appeal AI suggestion dropped: error (RuntimeError)" in caplog.text


def test_success_logs_one_count_line_and_no_email_text(monkeypatch, caplog, flag_on):
    good = MIDDLE.replace("(1) ", "").split("\n\n")
    answer = f"INTRO: {good[0]}\nPOINT: {good[1]}\nOUTRO: {good[2]}"

    async def call(user):
        assert SENTINEL in user, "the email does reach the model (fenced as data)"
        return answer

    monkeypatch.setattr(ais, "_call_model", call)
    draft, record = hook_result(classified(["llm_generated_review", "decision_vs_reviews"]))
    caplog.set_level(logging.DEBUG)
    new_draft, new_record = asyncio.run(aid.apply_ai_suggestion(REVIEW, draft, record, EMAIL_DATA))
    assert new_record["mode"] == "ai_suggestion"
    assert new_draft.draft_text == f"{FLAG}\n\nDear Jane Doe,\n\n{MIDDLE}\n\n{SIGN_OFF}"
    assert new_record["block_ids"] == list(SOURCE_BLOCKS)
    assert SENTINEL not in caplog.text
    assert caplog.text.count("Appeal AI suggestion model calls:") == 1
    assert "Appeal AI suggestion model calls: 1" in caplog.text


def test_no_model_configured_counts_zero_calls(monkeypatch, caplog, flag_on):
    monkeypatch.setattr(settings, "MODEL_PROVIDER", "fallback")
    draft, record = hook_result(Phase1Outcome(FAILED))
    caplog.set_level(logging.DEBUG)
    assert asyncio.run(aid.apply_ai_suggestion(REVIEW, draft, record, EMAIL_DATA)) == (draft, record)
    assert "Appeal AI suggestion model calls: 0" in caplog.text


def test_not_triggered_logs_nothing(monkeypatch, caplog, flag_on):
    monkeypatch.setattr(aid, "suggest_appeal_middle", FakeSuggest())
    draft, record = hook_result(classified(["decision_vs_reviews"]))
    caplog.set_level(logging.DEBUG)
    asyncio.run(aid.apply_ai_suggestion(REVIEW, draft, record, EMAIL_DATA))
    assert "Appeal AI suggestion" not in caplog.text


# --- wired into the orchestrator ----------------------------------------------------------------
@pytest.fixture
def wiring(monkeypatch):
    monkeypatch.setattr(settings, "QUERY_STRATEGY", "distill")
    monkeypatch.setattr(settings, "RECIPROCAL_DETECTOR_ENABLED", False)
    monkeypatch.setattr(settings, "APPEAL_REPLY_WINDOW_END", None)
    monkeypatch.setattr(settings, "PHASE1_APPEAL_START", None)
    monkeypatch.setattr(settings, "APPEAL_REPLY_REASON_SOURCE", "phase1")
    monkeypatch.setattr(settings, "PHASE1_APPEAL_ENABLED", True)
    monkeypatch.setattr(settings, "APPEAL_REPLY_COMPOSER_ENABLED", True)
    monkeypatch.setattr(settings, "APPEAL_REASON_CLASSIFIER_ENABLED", False)
    monkeypatch.setattr(orch, "classify_phase1_appeal", _Phase1Stub(phase1_result(
        reasons=("llm_generated_review", "decision_vs_reviews"))))


async def test_flag_off_never_calls_it_and_the_stored_drafts_are_the_hooks(session, monkeypatch, wiring):
    monkeypatch.setattr(settings, "APPEAL_AI_SUGGESTION_ENABLED", False)

    async def forbidden(*a, **k):
        raise AssertionError("apply_ai_suggestion must not be called with the flag off")

    monkeypatch.setattr(orch, "apply_ai_suggestion", forbidden)
    drafts = await _run_all_four(session, _pipeline(REVIEW, _SpyDrafter(forbid=True)))
    for entry_point, draft in drafts.items():
        assert draft["appeal_reply"]["mode"] == "chair_writes", entry_point
        assert draft["draft_text"] == "[CHAIR: write reply]", entry_point


async def test_flag_on_with_a_failed_suggestion_stores_exactly_the_flag_off_drafts(
        session, monkeypatch, wiring):
    monkeypatch.setattr(settings, "APPEAL_AI_SUGGESTION_ENABLED", False)
    off = await _run_all_four(session, _pipeline(REVIEW, _SpyDrafter(forbid=True)))
    monkeypatch.setattr(settings, "APPEAL_AI_SUGGESTION_ENABLED", True)
    monkeypatch.setattr(aid, "suggest_appeal_middle", FakeSuggest(SuggestionOutcome(None, failure="none_answer")))
    on = await _run_all_four(session, _pipeline(REVIEW, _SpyDrafter(forbid=True)))
    strip = lambda d: {k: v for k, v in d.items() if k != "history"}  # noqa: E731
    assert {k: strip(v) for k, v in on.items()} == {k: strip(v) for k, v in off.items()}


async def test_flag_on_every_entry_point_stores_the_flagged_suggestion(session, monkeypatch, wiring):
    monkeypatch.setattr(settings, "APPEAL_AI_SUGGESTION_ENABLED", True)
    fake = FakeSuggest()
    monkeypatch.setattr(aid, "suggest_appeal_middle", fake)
    drafts = await _run_all_four(session, _pipeline(REVIEW, _SpyDrafter(forbid=True)))
    for entry_point, draft in drafts.items():
        assert draft["draft_text"] == f"{FLAG}\n\nDear Jane Doe,\n\n{MIDDLE}\n\n{SIGN_OFF}", entry_point
        assert draft["appeal_reply"]["mode"] == "ai_suggestion", entry_point
        assert draft["appeal_reply"]["ai_suggestion"]["base_mode"] == "chair_writes", entry_point
    assert len(fake.calls) == 4
