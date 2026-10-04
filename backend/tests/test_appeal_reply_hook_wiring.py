"""The appeal reply hook wired into orchestrator._compute (reject-appeal Phase 4, Step 4).

Exercises all four entry points — process_email, reprocess_email,
reprocess_email_with_thread and the compute seam — with stubbed classifier
(distiller), retriever, model-call sites and drafter. No network, no model call.

Flag OFF: the hook never runs, the drafter is called exactly as before, and the
stored draft is exactly the drafter's output (no ``appeal_reply`` key).
Flag ON: for an appeal email the drafter is NEVER called; other intents still
use it; ``appeal_reply`` is present only when the hook ran.

Reasons (P3): by default they come from the phase-1 classifier
(APPEAL_REPLY_REASON_SOURCE="phase1"), driven here by a ``classify_phase1_appeal``
stub with PHASE1_APPEAL_ENABLED on; our own appeal-reason classifier is a stub
that must never be called. The rollback source ("appeal_reason") keeps the
pre-P3 behaviour, pinned by its own test below.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool

from app.core.config import settings
from app.db.models import Base
from app.pipeline import appeal_reply_hook as hook
from app.pipeline import orchestrator as orch
from app.pipeline.distiller import DistillResult
from app.pipeline.drafter import DraftResponse
from app.pipeline.orchestrator import EmailPipeline
from app.pipeline.phase1_appeal_classifier import AppealReason, Phase1AppealResult

REVIEW = "review_decision_appeal"
NON_APPEAL = "submission_requirements"
SCORE = "score_outcome_mismatch"
T1_BLOCKS = ["opening_warm", "lead_in_concerns", "point_scores", "point_rebuttal", "closing_reviewed"]


def phase1_result(relation="appeal", reasons=("decision_vs_reviews",), papers=("12345",)):
    return Phase1AppealResult(
        relation=relation, papers=list(papers),
        reasons=[AppealReason(reason=r, quote="the author's words") for r in reasons],
        dropped_unquoted=[])


def snapshot(state="classified", relation="appeal", reasons=("decision_vs_reviews",),
             must_verify=False, papers=("12345",)):
    return {"state": state, "relation": relation,
            "reasons": None if reasons is None else list(reasons),
            "must_verify": must_verify, "papers": None if papers is None else list(papers)}

T1 = (
    "Dear Jane Doe,\n\n"
    "We understand that this outcome may be disappointing, and we appreciate the effort you invested "
    "in preparing your submission. We would like to respond to your concerns:\n\n"
    "(1) Decisions are not based solely on the visible reviewer scores. Senior program committee "
    "members evaluated both the paper and the reviews, including whether the raised concerns can be "
    "addressed with minor clarifications or require substantial revision.\n\n"
    "(2) AAAI's two-phase process forgoes rebuttal for Phase 1 papers in favor of a quicker decision. We "
    "understand this can be frustrating, but Phase 1 decisions are final and will not be revisited in "
    "response to author objections.\n\n"
    "The decision is final, but we hope the feedback will be useful in further strengthening your work "
    "and helping you secure publication in another leading venue or future AAAI edition.\n\n"
    "Best Regards,\nAAAI 2027 PC Team"
)
MODEL_DRAFT = DraftResponse(
    draft_text="A model-written reply.", notes_for_chair=None, placeholders=[],
    citations=["policy_102"], answer_confidence=0.5, model_used="synthetic-model",
    generation_metadata={"provider": "local"},
)


@pytest_asyncio.fixture
async def session():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:",
                                 connect_args={"check_same_thread": False}, poolclass=StaticPool)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    maker = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    async with maker() as s:
        yield s
    await engine.dispose()


class _StubDistiller:
    def __init__(self, intent):
        self.intent = intent

    async def distill(self, subject, body, *, transcript=None):
        return DistillResult(queries=["appeal of review decision"], intent=self.intent, confidence=0.9)


class _StubRetriever:
    async def retrieve(self, query, intent, top_k=3, *, prior_intent=""):
        return []


class _SpyDrafter:
    """Records every call; returns MODEL_DRAFT, or raises when ``forbid`` is set."""

    provider = "local"

    def __init__(self, forbid=False):
        self.forbid = forbid
        self.calls = []

    async def draft(self, email, classification, retrieved_chunks, forced_policy_key=None):
        self.calls.append((dict(email), classification.intent, list(retrieved_chunks), forced_policy_key))
        if self.forbid:
            raise AssertionError("drafter.draft must not be called for an appeal email")
        return MODEL_DRAFT.model_copy(deep=True)


class _Phase1Stub:
    """Stands in for classify_phase1_appeal: records calls, returns ``result``."""

    def __init__(self, result):
        self.result = result
        self.calls = 0

    async def __call__(self, email_data):
        self.calls += 1
        return self.result


@pytest.fixture(autouse=True)
def _settings(monkeypatch):
    monkeypatch.setattr(settings, "QUERY_STRATEGY", "distill")
    monkeypatch.setattr(settings, "RECIPROCAL_DETECTOR_ENABLED", False)
    monkeypatch.setattr(settings, "APPEAL_REPLY_WINDOW_END", None)
    monkeypatch.setattr(settings, "PHASE1_APPEAL_START", None)
    # P3: reasons come from the phase-1 classifier (the default source).
    monkeypatch.setattr(settings, "APPEAL_REPLY_REASON_SOURCE", "phase1")
    monkeypatch.setattr(settings, "PHASE1_APPEAL_ENABLED", True)
    monkeypatch.setattr(orch, "classify_phase1_appeal", _Phase1Stub(phase1_result()))
    # Our own appeal-reason classifier is dormant: off, and loud if ever called.
    monkeypatch.setattr(settings, "APPEAL_REASON_CLASSIFIER_ENABLED", False)

    async def must_not_be_called(email_data, prior):
        raise AssertionError("the dormant appeal-reason classifier was called")

    monkeypatch.setattr(orch, "classify_appeal_reason", must_not_be_called)


def _pipeline(intent, drafter):
    p = EmailPipeline()
    p.distiller = _StubDistiller(intent)
    p.retriever = _StubRetriever()
    p.drafter = drafter
    return p


def _capture(pipeline):
    """Wrap create/update so the exact record handed to persistence is kept."""
    seen = []
    real_create, real_update = pipeline.email_repo.create_email, pipeline.email_repo.update_email_outputs

    async def create(db, record):
        seen.append(json.loads(json.dumps(record, default=str, sort_keys=True)))
        return await real_create(db, record)

    async def update(db, email_id, record):
        seen.append(json.loads(json.dumps(record, default=str, sort_keys=True)))
        return await real_update(db, email_id, record)

    pipeline.email_repo.create_email, pipeline.email_repo.update_email_outputs = create, update
    return seen


EMAIL = {"from": "author@example.org", "sender_name": "Jane Doe", "subject": "Appeal",
         "body": "We appeal the decision.", "timestamp": "2026-09-15T10:00:00Z"}
THREAD = [{"public": True, "plain_body": "We appeal the decision.", "author_role": "end-user",
           "created_at": datetime(2026, 9, 15, 10, tzinfo=timezone.utc)}]


async def _run_all_four(session, pipeline):
    """The draft dict produced by each of the four entry points."""
    seen = _capture(pipeline)
    res = await pipeline.process_email(dict(EMAIL), session)
    created = seen[-1]["draft"]
    email = await pipeline.email_repo.get_email_by_id(session, res.email_id)
    await pipeline.reprocess_email(session, email)
    reprocessed = seen[-1]["draft"]
    email = await pipeline.email_repo.get_email_by_id(session, res.email_id)
    await pipeline.reprocess_email_with_thread(session, email, THREAD, triggering_comment_ids=[1])
    threaded = seen[-1]["draft"]
    computed = json.loads(json.dumps((await pipeline.compute(dict(EMAIL), session)).record["draft"],
                                     default=str, sort_keys=True))
    return {"process_email": created, "reprocess_email": reprocessed,
            "reprocess_email_with_thread": threaded, "compute": computed}


def _without_history(draft):
    return {k: v for k, v in draft.items() if k != "history"}


# --- flag OFF ----------------------------------------------------------------------------------
@pytest.mark.parametrize("intent", [REVIEW, "desk_reject_appeal", NON_APPEAL])
async def test_flag_off_the_hook_never_runs_and_the_drafter_output_is_stored_unchanged(
    session, monkeypatch, intent
):
    monkeypatch.setattr(settings, "APPEAL_REPLY_COMPOSER_ENABLED", False)

    def forbidden(*a, **k):
        raise AssertionError("the appeal hook must not run with the flag off")

    monkeypatch.setattr(orch, "prepare_appeal_draft", forbidden)
    drafter = _SpyDrafter()
    drafts = await _run_all_four(session, _pipeline(intent, drafter))

    expected = json.loads(json.dumps(MODEL_DRAFT.model_dump(), sort_keys=True))
    for entry_point, draft in drafts.items():
        assert _without_history(draft) == expected, entry_point
        assert "appeal_reply" not in draft, entry_point
    assert len(drafter.calls) == 4
    for email, call_intent, chunks, forced in drafter.calls:
        assert (call_intent, chunks, forced) == (intent, [], None)


# --- flag ON ------------------------------------------------------------------------------------
async def test_flag_on_an_appeal_email_never_reaches_the_drafter(session, monkeypatch):
    monkeypatch.setattr(settings, "APPEAL_REPLY_COMPOSER_ENABLED", True)
    drafter = _SpyDrafter(forbid=True)
    drafts = await _run_all_four(session, _pipeline(REVIEW, drafter))
    assert drafter.calls == []
    for entry_point, draft in drafts.items():
        assert draft["draft_text"] == T1, entry_point
        assert draft["appeal_reply"] == {
            "mode": "merged", "reasons": [SCORE], "block_ids": T1_BLOCKS,
            "source": "phase1", "phase1": snapshot(),
        }, entry_point
        assert (draft["model_used"], draft["generation_metadata"]["provider"]) == ("none", "appeal_reply_composer")
    assert orch.classify_phase1_appeal.calls == 4, "the hook read the outcome of the same run, each time"


async def test_flag_on_a_non_appeal_email_still_uses_the_drafter_and_has_no_appeal_key(session, monkeypatch):
    monkeypatch.setattr(settings, "APPEAL_REPLY_COMPOSER_ENABLED", True)
    drafter = _SpyDrafter()
    drafts = await _run_all_four(session, _pipeline(NON_APPEAL, drafter))
    assert len(drafter.calls) == 4
    expected = json.loads(json.dumps(MODEL_DRAFT.model_dump(), sort_keys=True))
    for entry_point, draft in drafts.items():
        assert _without_history(draft) == expected, entry_point
        assert "appeal_reply" not in draft, entry_point


async def test_flag_on_a_desk_reject_appeal_gets_the_placeholder_not_the_drafter(session, monkeypatch):
    monkeypatch.setattr(settings, "APPEAL_REPLY_COMPOSER_ENABLED", True)
    drafter = _SpyDrafter(forbid=True)
    drafts = await _run_all_four(session, _pipeline("desk_reject_appeal", drafter))
    assert drafter.calls == []
    for entry_point, draft in drafts.items():
        assert draft["draft_text"] == "[CHAIR: write reply]", entry_point
        assert draft["placeholders"] == ["write reply"], entry_point
        # The phase-1 gate asks only review-decision appeals, so it is not met here.
        assert draft["appeal_reply"] == {
            "mode": "desk_reject", "reasons": None, "block_ids": [], "source": "phase1",
            "phase1": snapshot("gate_not_met", None, None, None, None),
        }, entry_point
    assert orch.classify_phase1_appeal.calls == 0


async def test_flag_on_the_window_reaches_the_hook_from_settings(session, monkeypatch):
    monkeypatch.setattr(settings, "APPEAL_REPLY_COMPOSER_ENABLED", True)
    monkeypatch.setattr(settings, "APPEAL_REPLY_WINDOW_END", datetime(2026, 9, 1, tzinfo=timezone.utc))
    p = _pipeline(REVIEW, _SpyDrafter(forbid=True))
    seen = _capture(p)
    await p.process_email(dict(EMAIL), session)
    draft = seen[-1]["draft"]
    assert (draft["draft_text"], draft["notes_for_chair"], draft["appeal_reply"]["mode"]) == (
        "[CHAIR: write reply]", "Appeal reply wording is for Phase 1 rejections only", "window")


async def test_flag_on_the_reciprocal_flag_reaches_the_hook(session, monkeypatch):
    monkeypatch.setattr(settings, "APPEAL_REPLY_COMPOSER_ENABLED", True)
    monkeypatch.setattr(settings, "RECIPROCAL_DETECTOR_ENABLED", True)

    async def reciprocal(*, subject, body, transcript=None):
        return True

    monkeypatch.setattr(orch, "detect_reciprocal_dispute", reciprocal)
    p = _pipeline("desk_reject_appeal", _SpyDrafter(forbid=True))
    seen = _capture(p)
    await p.process_email(dict(EMAIL), session)
    draft = seen[-1]["draft"]
    assert draft["draft_text"] == "[CHAIR: reciprocal complaint; see note]"
    assert draft["appeal_reply"] == {"mode": "reciprocal_review", "reasons": ["reciprocal_dispute"],
                                     "block_ids": [], "source": "phase1",
                                     "phase1": snapshot("gate_not_met", None, None, None, None)}


async def test_a_failure_inside_the_hook_never_breaks_the_pipeline(session, monkeypatch):
    monkeypatch.setattr(settings, "APPEAL_REPLY_COMPOSER_ENABLED", True)

    def boom(*a, **k):
        raise RuntimeError("synthetic failure")

    monkeypatch.setattr(hook, "decide_appeal_reply", boom)
    drafter = _SpyDrafter(forbid=True)
    drafts = await _run_all_four(session, _pipeline(REVIEW, drafter))
    assert drafter.calls == []
    for entry_point, draft in drafts.items():
        assert draft["draft_text"] == "[CHAIR: write reply]", entry_point
        assert draft["notes_for_chair"] == "Chair writes: the appeal reply could not be prepared automatically."
        assert draft["appeal_reply"]["mode"] == "failed", entry_point


async def test_the_routing_of_a_composed_appeal_draft_is_still_human_review(session, monkeypatch):
    monkeypatch.setattr(settings, "APPEAL_REPLY_COMPOSER_ENABLED", True)
    p = _pipeline(REVIEW, _SpyDrafter(forbid=True))
    seen = _capture(p)
    await p.process_email(dict(EMAIL), session)
    assert seen[-1]["routing"]["lane"] == "human_review"


# --- P3: the reason source ---------------------------------------------------------------------
async def test_the_rollback_source_composes_from_appeal_reason_exactly_as_before(session, monkeypatch):
    """APPEAL_REPLY_REASON_SOURCE="appeal_reason": our own classifier's answer drives the
    draft and the record carries no source/phase1 keys — the pre-P3 record exactly. The
    phase-1 classifier still runs (its rows are its own business) but is not read."""
    monkeypatch.setattr(settings, "APPEAL_REPLY_COMPOSER_ENABLED", True)
    monkeypatch.setattr(settings, "APPEAL_REPLY_REASON_SOURCE", "appeal_reason")
    monkeypatch.setattr(settings, "APPEAL_REASON_CLASSIFIER_ENABLED", True)
    monkeypatch.setattr(orch, "classify_phase1_appeal", _Phase1Stub(phase1_result(reasons=["record_error"])))

    async def reasons(email_data, prior):
        return [SCORE]

    monkeypatch.setattr(orch, "classify_appeal_reason", reasons)
    drafts = await _run_all_four(session, _pipeline(REVIEW, _SpyDrafter(forbid=True)))
    for entry_point, draft in drafts.items():
        assert draft["draft_text"] == T1, entry_point
        assert draft["appeal_reply"] == {"mode": "merged", "reasons": [SCORE], "block_ids": T1_BLOCKS}, entry_point


async def test_the_phase1_source_ignores_appeal_reason(session, monkeypatch):
    """Even with our classifier switched on and saying "wrong paper", the phase-1 outcome
    (T1) decides the draft."""
    monkeypatch.setattr(settings, "APPEAL_REPLY_COMPOSER_ENABLED", True)
    monkeypatch.setattr(settings, "APPEAL_REASON_CLASSIFIER_ENABLED", True)

    async def reasons(email_data, prior):
        return ["wrong_paper_review"]

    monkeypatch.setattr(orch, "classify_appeal_reason", reasons)
    p = _pipeline(REVIEW, _SpyDrafter(forbid=True))
    seen = _capture(p)
    await p.process_email(dict(EMAIL), session)
    assert seen[-1]["draft"]["draft_text"] == T1
    assert seen[-1]["extraction"]["appeal_reason"] == ["wrong_paper_review"], "still stored, D66 untouched"


async def test_with_the_phase1_flag_off_every_review_appeal_gets_reason_unknown(session, monkeypatch):
    monkeypatch.setattr(settings, "APPEAL_REPLY_COMPOSER_ENABLED", True)
    monkeypatch.setattr(settings, "PHASE1_APPEAL_ENABLED", False)
    drafts = await _run_all_four(session, _pipeline(REVIEW, _SpyDrafter(forbid=True)))
    assert orch.classify_phase1_appeal.calls == 0
    for entry_point, draft in drafts.items():
        assert draft["draft_text"] == "[CHAIR: write reply]", entry_point
        assert draft["notes_for_chair"] == (
            "Chair writes: the phase-1 appeal classifier is turned off, so the appeal reasons were "
            "not determined."), entry_point
        assert draft["appeal_reply"] == {
            "mode": "reason_unknown", "reasons": None, "block_ids": [], "source": "phase1",
            "phase1": snapshot("flag_off", None, None, None, None)}, entry_point


async def test_a_failed_phase1_call_gives_reason_unknown(session, monkeypatch):
    monkeypatch.setattr(settings, "APPEAL_REPLY_COMPOSER_ENABLED", True)
    monkeypatch.setattr(orch, "classify_phase1_appeal", _Phase1Stub(None))
    drafts = await _run_all_four(session, _pipeline(REVIEW, _SpyDrafter(forbid=True)))
    for entry_point, draft in drafts.items():
        assert (draft["draft_text"], draft["notes_for_chair"], draft["appeal_reply"]["mode"]) == (
            "[CHAIR: write reply]", "Chair writes: the phase-1 appeal classifier did not return an answer.",
            "reason_unknown"), entry_point
        assert draft["appeal_reply"]["phase1"] == snapshot("failed", None, None, None, None)


async def test_not_appeal_gives_the_chair_writes_placeholder(session, monkeypatch):
    monkeypatch.setattr(settings, "APPEAL_REPLY_COMPOSER_ENABLED", True)
    monkeypatch.setattr(orch, "classify_phase1_appeal", _Phase1Stub(phase1_result("not_appeal", reasons=())))
    p = _pipeline(REVIEW, _SpyDrafter(forbid=True))
    seen = _capture(p)
    await p.process_email(dict(EMAIL), session)
    draft = seen[-1]["draft"]
    assert draft["draft_text"] == "[CHAIR: write reply]"
    assert draft["appeal_reply"]["mode"] == "not_appeal"
    assert draft["appeal_reply"]["phase1"] == snapshot("not_appeal", "not_appeal", None, None, None)


async def test_the_hook_reads_the_outcome_of_the_run_never_the_stored_rows(session, monkeypatch):
    """S2: the first run classifies T1 and stores phase1_appeals rows; on the redraft the
    call fails, which KEEPS those rows (his rule), yet the draft is the placeholder —
    the hook never falls back to stored rows."""
    from sqlalchemy import func, select

    from app.db.models import Phase1Appeal

    monkeypatch.setattr(settings, "APPEAL_REPLY_COMPOSER_ENABLED", True)
    p = _pipeline(REVIEW, _SpyDrafter(forbid=True))
    seen = _capture(p)
    res = await p.process_email(dict(EMAIL), session)
    assert seen[-1]["draft"]["draft_text"] == T1
    assert (await session.execute(select(func.count()).select_from(Phase1Appeal))).scalar_one() == 1

    monkeypatch.setattr(orch, "classify_phase1_appeal", _Phase1Stub(None))
    email = await p.email_repo.get_email_by_id(session, res.email_id)
    await p.reprocess_email(session, email)
    assert (await session.execute(select(func.count()).select_from(Phase1Appeal))).scalar_one() == 1
    assert seen[-1]["draft"]["draft_text"] == "[CHAIR: write reply]"
    assert seen[-1]["draft"]["appeal_reply"]["mode"] == "reason_unknown"
