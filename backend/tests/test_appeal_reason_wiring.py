"""Appeal-reason classifier wiring in `orchestrator._compute` (reject_appeal.md D76).

Covers the rules the CALLER owns: the config flag (D65), the intent gate (D58),
the reciprocal skip on the EFFECTIVE flag (D68), handing the STORED value in as
the prior (D66), and storing the result inside the extraction JSON. The
classifier's own behaviour is tested in test_appeal_reason_classifier.py; here
it is stubbed, so no test in this file makes a model call.

⚠️ Stubs RECORD their calls instead of raising (D48): code that catches broadly
would swallow a raised AssertionError into the same outcome a passing case has.
"""

from __future__ import annotations

import inspect
from datetime import datetime, timezone

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool

from app.core.config import settings
from app.db.models import Base
from app.pipeline import appeal_reason_classifier as arc
from app.pipeline import orchestrator as orch
from app.pipeline.distiller import DistillResult
from app.pipeline.orchestrator import EmailPipeline
from app.repositories.email_repository import EmailRepository

GATE_INTENTS = ("desk_reject_appeal", "review_decision_appeal")


@pytest.fixture(autouse=True)
def _use_the_distiller(monkeypatch):
    """⚠️ Required (D49): under the conftest's `prefix` strategy the orchestrator
    skips the distiller and the KEYWORD classifier picks the intent, so the
    stubbed intent below would never reach the gate."""
    monkeypatch.setattr(settings, "QUERY_STRATEGY", "distill")


@pytest.fixture(autouse=True)
def _detector_answers_nothing(monkeypatch):
    """Default: the reciprocal detector gives no answer, so the effective flag is
    whatever was stored. Tests that need a flag install their own detector."""
    monkeypatch.setattr(orch, "detect_reciprocal_dispute", _RecordingDetector(None))


@pytest_asyncio.fixture
async def session():
    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    maker = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    async with maker() as s:
        yield s
    await engine.dispose()


class _StubDistiller:
    def __init__(self, intent="review_decision_appeal"):
        self.intent = intent

    async def distill(self, subject, body, *, transcript=None):
        return DistillResult(
            queries=["appeal of review decision"],
            intent=self.intent,
            confidence=0.9,
            submission_numbers_raw=["22336"],
        )


class _StubRetriever:
    async def retrieve(self, query, intent, top_k=3, *, prior_intent=""):
        return []


class _RecordingDetector:
    def __init__(self, verdict=None):
        self.verdict = verdict
        self.calls: list[dict] = []

    async def __call__(self, *, subject, body, transcript=None):
        self.calls.append({"subject": subject, "body": body, "transcript": transcript})
        return self.verdict


class _RecordingClassifier:
    """Records (email_data, prior) for every call and returns a fixed answer."""

    def __init__(self, answer=None):
        self.answer = answer
        self.calls: list[dict] = []

    async def __call__(self, email_data, prior_appeal_reason=None):
        self.calls.append({
            "email_data": dict(email_data),
            "prior": None if prior_appeal_reason is None else list(prior_appeal_reason),
        })
        return self.answer


def _pipeline(intent="review_decision_appeal"):
    p = EmailPipeline()
    p.distiller = _StubDistiller(intent)
    p.retriever = _StubRetriever()
    return p


_EMAIL = {
    "from": "jane@example.edu",
    "subject": "Appeal of the review decision",
    "body": "One review is about a different paper.",
}

_FOLLOWUP = [{
    "plain_body": "UNIQUE-FOLLOWUP-MARKER any update on my appeal?",
    "is_requester": True,
    "public": True,
    # A real datetime: build_transcript reads `.tzinfo` off this value.
    "created_at": datetime(2026, 9, 28, tzinfo=timezone.utc),
}]


def _install(monkeypatch, *, flag, classifier):
    monkeypatch.setattr(settings, "APPEAL_REASON_CLASSIFIER_ENABLED", flag)
    monkeypatch.setattr(orch, "classify_appeal_reason", classifier)


async def _new_row(session, pipeline):
    result = await pipeline.process_email(dict(_EMAIL), session)
    return await EmailRepository().get_email_by_id(session, str(result.email_id))


async def _refresh(session, email):
    return await EmailRepository().get_email_by_id(session, str(email.id))


async def _row_with_stored_reasons(session, monkeypatch, reasons, *, intent="review_decision_appeal"):
    """A row whose stored appeal_reason is `reasons`, written through the pipeline."""
    _install(monkeypatch, flag=True, classifier=_RecordingClassifier(reasons))
    pipeline = _pipeline(intent)
    email = await _new_row(session, pipeline)
    assert email.extraction["appeal_reason"] == reasons, "setup"
    return pipeline, email


# ---------------------------------------------------------------------------
# The flag (D65) — off means zero behaviour change
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("flag", [False, True])
async def test_the_call_happens_ONLY_when_the_flag_is_on(session, monkeypatch, flag):
    classifier = _RecordingClassifier(["wrong_paper_review"])
    _install(monkeypatch, flag=flag, classifier=classifier)
    email = await _new_row(session, _pipeline())
    if flag:
        assert len(classifier.calls) == 1
        assert email.extraction["appeal_reason"] == ["wrong_paper_review"]
    else:
        assert classifier.calls == [], "flag off: the classifier must never be called"
        assert email.extraction["appeal_reason"] is None


async def test_flag_off_leaves_a_stored_answer_untouched(session, monkeypatch):
    """Turning the feature off stops the CALL only — a stored answer survives a
    re-run (D65/D66), so the switch can never destroy data."""
    pipeline, email = await _row_with_stored_reasons(
        session, monkeypatch, ["score_outcome_mismatch"]
    )
    classifier = _RecordingClassifier(["other"])
    _install(monkeypatch, flag=False, classifier=classifier)
    await pipeline.reprocess_email(session, email)
    refreshed = await _refresh(session, email)
    assert classifier.calls == []
    assert refreshed.extraction["appeal_reason"] == ["score_outcome_mismatch"]


# ---------------------------------------------------------------------------
# The gate (D58)
# ---------------------------------------------------------------------------
def test_the_gate_is_exactly_the_two_appeal_intents():
    """One constant, shared with taxonomy.REJECT_APPEAL_INTENTS — pinned so a
    silent widening of the taxonomy set cannot widen the gate unnoticed."""
    assert orch._APPEAL_REASON_GATE_INTENTS == frozenset(GATE_INTENTS)


@pytest.mark.parametrize("intent", GATE_INTENTS)
async def test_gate_intents_call_the_classifier_once(session, monkeypatch, intent):
    classifier = _RecordingClassifier(["other"])
    _install(monkeypatch, flag=True, classifier=classifier)
    email = await _new_row(session, _pipeline(intent))
    assert len(classifier.calls) == 1
    assert email.extraction["appeal_reason"] == ["other"]


@pytest.mark.parametrize(
    "intent", ["cms_support", "reviewer_assignment", "submission_requirements"]
)
async def test_other_intents_do_not_call_the_classifier(session, monkeypatch, intent):
    classifier = _RecordingClassifier(["other"])
    _install(monkeypatch, flag=True, classifier=classifier)
    email = await _new_row(session, _pipeline(intent))
    assert classifier.calls == []
    assert email.extraction["appeal_reason"] is None


async def test_gate_not_met_preserves_the_stored_answer(session, monkeypatch):
    """A follow-up whose intent drifts off the gate ("any update?") keeps reasons."""
    pipeline, email = await _row_with_stored_reasons(session, monkeypatch, ["other"])
    classifier = _RecordingClassifier(["wrong_paper_review"])
    _install(monkeypatch, flag=True, classifier=classifier)
    pipeline.distiller = _StubDistiller("reviewer_assignment")
    await pipeline.reprocess_email_with_thread(session, email, _FOLLOWUP)
    refreshed = await _refresh(session, email)
    assert classifier.calls == []
    assert refreshed.extraction["appeal_reason"] == ["other"]


# ---------------------------------------------------------------------------
# The reciprocal skip (D68) — on the EFFECTIVE flag
# ---------------------------------------------------------------------------
async def test_a_fresh_reciprocal_True_skips_the_call(session, monkeypatch):
    monkeypatch.setattr(settings, "RECIPROCAL_DETECTOR_ENABLED", True)
    monkeypatch.setattr(orch, "detect_reciprocal_dispute", _RecordingDetector(True))
    classifier = _RecordingClassifier(["other"])
    _install(monkeypatch, flag=True, classifier=classifier)
    email = await _new_row(session, _pipeline("desk_reject_appeal"))
    assert email.extraction["is_reciprocal_dispute"] is True, "setup"
    assert classifier.calls == []
    assert email.extraction["appeal_reason"] is None


async def test_a_STORED_reciprocal_True_skips_even_when_the_detector_does_not_run(
    session, monkeypatch
):
    """D68: the skip reads the flag AFTER D42's preserve rule. Here the follow-up
    classifies as `review_decision_appeal`, outside the detector's own gate, so
    the detector does not run — only the stored True can trigger the skip."""
    monkeypatch.setattr(settings, "RECIPROCAL_DETECTOR_ENABLED", True)
    detector = _RecordingDetector(True)
    monkeypatch.setattr(orch, "detect_reciprocal_dispute", detector)
    _install(monkeypatch, flag=False, classifier=_RecordingClassifier(None))
    pipeline = _pipeline("desk_reject_appeal")
    email = await _new_row(session, pipeline)
    assert email.extraction["is_reciprocal_dispute"] is True, "setup: stored True"

    detector.calls.clear()
    classifier = _RecordingClassifier(["other"])
    _install(monkeypatch, flag=True, classifier=classifier)
    pipeline.distiller = _StubDistiller("review_decision_appeal")
    await pipeline.reprocess_email(session, email)
    refreshed = await _refresh(session, email)
    assert detector.calls == [], "setup: the detector must not have run"
    assert refreshed.extraction["is_reciprocal_dispute"] is True
    assert classifier.calls == [], "effective True must skip the reason call"


async def test_the_skip_leaves_a_stored_reason_list_untouched(session, monkeypatch):
    """Skipped means NOT touched — never overwritten to None or []."""
    pipeline, email = await _row_with_stored_reasons(
        session, monkeypatch, ["wrong_paper_review"], intent="desk_reject_appeal"
    )
    monkeypatch.setattr(settings, "RECIPROCAL_DETECTOR_ENABLED", True)
    monkeypatch.setattr(orch, "detect_reciprocal_dispute", _RecordingDetector(True))
    classifier = _RecordingClassifier([])
    _install(monkeypatch, flag=True, classifier=classifier)
    await pipeline.reprocess_email(session, email)
    refreshed = await _refresh(session, email)
    assert classifier.calls == []
    assert refreshed.extraction["appeal_reason"] == ["wrong_paper_review"]


@pytest.mark.parametrize("verdict", [False, None])
async def test_a_reciprocal_False_or_None_still_asks(session, monkeypatch, verdict):
    """Skip only on True (D58)."""
    monkeypatch.setattr(settings, "RECIPROCAL_DETECTOR_ENABLED", True)
    monkeypatch.setattr(orch, "detect_reciprocal_dispute", _RecordingDetector(verdict))
    classifier = _RecordingClassifier(["other"])
    _install(monkeypatch, flag=True, classifier=classifier)
    await _new_row(session, _pipeline("desk_reject_appeal"))
    assert len(classifier.calls) == 1


# ---------------------------------------------------------------------------
# The prior handed in (D66) — asserted on the call args, not only the outcome
# ---------------------------------------------------------------------------
async def test_a_new_row_passes_no_prior(session, monkeypatch):
    classifier = _RecordingClassifier(["other"])
    _install(monkeypatch, flag=True, classifier=classifier)
    await _new_row(session, _pipeline())
    assert classifier.calls[0]["prior"] is None


@pytest.mark.parametrize("stored", [["wrong_paper_review", "reviewer_misunderstanding"], []])
async def test_manual_redraft_passes_the_STORED_value_as_prior(session, monkeypatch, stored):
    """`[]` counts as a stored answer too — it must reach the classifier as [],
    not decay to None on the way in."""
    pipeline, email = await _row_with_stored_reasons(session, monkeypatch, stored)
    classifier = _RecordingClassifier(["other"])
    _install(monkeypatch, flag=True, classifier=classifier)
    await pipeline.reprocess_email(session, email)
    assert len(classifier.calls) == 1
    assert classifier.calls[0]["prior"] == stored
    assert classifier.calls[0]["prior"] is not None


async def test_followup_passes_the_stored_value_and_the_transcript(session, monkeypatch):
    pipeline, email = await _row_with_stored_reasons(session, monkeypatch, ["other"])
    classifier = _RecordingClassifier(["other"])
    _install(monkeypatch, flag=True, classifier=classifier)
    await pipeline.reprocess_email_with_thread(session, email, _FOLLOWUP)
    (call,) = classifier.calls
    assert call["prior"] == ["other"]
    assert "UNIQUE-FOLLOWUP-MARKER" in (call["email_data"].get("thread_transcript") or ""), (
        "the whole-conversation input must reach the classifier"
    )


async def test_an_invalid_stored_value_is_passed_as_no_prior(session, monkeypatch):
    """A letter code (or any non-canonical list) in storage is not ours (D66)."""
    classifier = _RecordingClassifier(None)
    _install(monkeypatch, flag=False, classifier=classifier)
    pipeline = _pipeline()
    email = await _new_row(session, pipeline)
    await EmailRepository().update_email_outputs(
        session, str(email.id), {"extraction": {**email.extraction, "appeal_reason": ["b"]}}
    )
    email = await _refresh(session, email)
    _install(monkeypatch, flag=True, classifier=classifier)
    await pipeline.reprocess_email(session, email)
    assert classifier.calls[0]["prior"] is None


@pytest.mark.parametrize(
    "supplied, expected",
    [(["other"], ["other"]), ([], []), (["b"], None), ("other", None), (None, None)],
)
async def test_the_compute_seam_validates_a_caller_supplied_prior(
    session, monkeypatch, supplied, expected
):
    """`compute`'s caller fills `prior_appeal_reason` itself, so `_compute` is the
    one place a bad value can be stopped before it reaches the classifier."""
    classifier = _RecordingClassifier(["other"])
    _install(monkeypatch, flag=True, classifier=classifier)
    await _pipeline().compute({**_EMAIL, "prior_appeal_reason": supplied}, session)
    assert classifier.calls[0]["prior"] == expected


# ---------------------------------------------------------------------------
# Storage — the classifier's return value lands in the extraction JSON
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("answer", [["score_outcome_mismatch"], [], None])
async def test_the_result_is_stored_in_the_extraction_json(session, monkeypatch, answer):
    _install(monkeypatch, flag=True, classifier=_RecordingClassifier(answer))
    email = await _new_row(session, _pipeline())
    assert email.extraction["appeal_reason"] == answer


async def test_end_to_end_a_failed_answer_preserves_the_stored_value(session, monkeypatch):
    """The REAL classifier, with only its transport mocked: a garbage answer on a
    re-run keeps the stored reasons, which proves the prior handed in is the one
    D66 then preserves."""
    pipeline, email = await _row_with_stored_reasons(
        session, monkeypatch, ["llm_generated_review"]
    )

    async def garbage(user):  # noqa: ANN001
        return "not the contract"

    monkeypatch.setattr(settings, "APPEAL_REASON_CLASSIFIER_ENABLED", True)
    monkeypatch.setattr(orch, "classify_appeal_reason", arc.classify_appeal_reason)
    # Patch the dispatcher itself, NOT MODEL_PROVIDER: setting a real provider
    # would also send the drafter to a live endpoint.
    monkeypatch.setattr(arc, "_call_model", garbage)
    await pipeline.reprocess_email(session, email)
    refreshed = await _refresh(session, email)
    assert refreshed.extraction["appeal_reason"] == ["llm_generated_review"]


def test_every_row_updating_call_site_threads_the_prior():
    """Source-level guard, as for the reciprocal flag: a behavioural test only
    covers the paths someone remembered to write."""
    for method in (EmailPipeline.reprocess_email, EmailPipeline.reprocess_email_with_thread):
        src = inspect.getsource(method)
        assert '"prior_appeal_reason": _prior_appeal_reason(email)' in src, (
            f"{method.__name__} does not thread the stored appeal_reason (D66)"
        )
