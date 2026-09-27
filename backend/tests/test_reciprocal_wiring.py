"""Detector wiring in `orchestrator._compute` (commit 4 of the rebuild).

Covers the three rules the CALLER owns — the config flag, the intent gate, and
the preserve-prior rule (reject_appeal.md D40/D42/D43/D44). The detector's own
behaviour is tested in test_reciprocal_detector.py; here it is always stubbed,
so no test in this file makes a model call.

⚠️ Every stub RECORDS its calls instead of raising (D48). A stub that raises
cannot prove "this must not be called" against code that catches broadly — an
AssertionError would be swallowed and turn into the same `None` the passing
case returns, which is exactly how a mutation survived once already.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool

from app.core.config import settings
from app.db.models import Base
from app.pipeline import orchestrator as orch
from app.pipeline.distiller import DistillResult
from app.pipeline.orchestrator import EmailPipeline
from app.repositories.email_repository import EmailRepository


@pytest.fixture(autouse=True)
def _use_the_distiller(monkeypatch):
    """⚠️ Required, and its absence fails in a MISLEADING way.

    The hermetic conftest pins `QUERY_STRATEGY="prefix"` so the suite never
    hits a hosted model. Under `prefix` the orchestrator skips the distiller
    entirely and classifies with the KEYWORD backend — which labels this
    module's fixture email `reviewer_assignment`, so the intent gate correctly
    does not fire and the stubbed detector is never called.

    Without this fixture every "the detector should run" test here fails with
    an empty call list, which reads exactly like broken wiring rather than a
    test-environment artefact. Diagnosed the hard way.
    """
    monkeypatch.setattr(settings, "QUERY_STRATEGY", "distill")


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
    def __init__(self, intent="desk_reject_appeal"):
        self.intent = intent

    async def distill(self, subject, body, *, transcript=None):
        return DistillResult(
            queries=["desk rejection appeal"],
            intent=self.intent,
            confidence=0.9,
            submission_numbers_raw=["22336"],
        )


class _StubRetriever:
    async def retrieve(self, query, intent, top_k=3, *, prior_intent=""):
        return []


class _RecordingDetector:
    """Records every call and returns a fixed verdict. Never raises (D48)."""

    def __init__(self, verdict=None):
        self.verdict = verdict
        self.calls: list[dict] = []

    async def __call__(self, *, subject, body, transcript=None):
        self.calls.append(
            {"subject": subject, "body": body, "transcript": transcript}
        )
        return self.verdict


def _pipeline(intent="desk_reject_appeal"):
    p = EmailPipeline()
    p.distiller = _StubDistiller(intent)
    p.retriever = _StubRetriever()
    return p


_EMAIL = {
    "from": "jane@example.edu",
    "subject": "Desk rejection appeal",
    "body": "My reciprocal reviewers never submitted their reviews.",
}


async def _run(session, monkeypatch, *, flag, detector, intent="desk_reject_appeal"):
    monkeypatch.setattr(settings, "RECIPROCAL_DETECTOR_ENABLED", flag)
    monkeypatch.setattr(orch, "detect_reciprocal_dispute", detector)
    result = await _pipeline(intent).process_email(dict(_EMAIL), session)
    email = await EmailRepository().get_email_by_id(session, str(result.email_id))
    return email


# ---------------------------------------------------------------------------
# The config flag (D44)
# ---------------------------------------------------------------------------
async def test_flag_off_means_the_detector_is_never_called(session, monkeypatch):
    detector = _RecordingDetector(verdict=True)
    email = await _run(session, monkeypatch, flag=False, detector=detector)
    assert detector.calls == [], "no call may be made while the flag is off"
    assert email.extraction["is_reciprocal_dispute"] is None


async def test_flag_on_and_matching_intent_calls_once(session, monkeypatch):
    detector = _RecordingDetector(verdict=True)
    email = await _run(session, monkeypatch, flag=True, detector=detector)
    assert len(detector.calls) == 1
    assert email.extraction["is_reciprocal_dispute"] is True


# ---------------------------------------------------------------------------
# The intent gate (D40)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "intent",
    ["review_decision_appeal", "reviewer_assignment", "cms_support", "paper_bidding"],
)
async def test_other_intents_do_not_call_the_detector(session, monkeypatch, intent):
    """Narrow gate: `review_decision_appeal` is included here deliberately.

    It is the neighbouring appeal intent and the tempting one to let through;
    D40 measured that widening reaches no additional r tickets.
    """
    detector = _RecordingDetector(verdict=True)
    email = await _run(session, monkeypatch, flag=True, detector=detector, intent=intent)
    assert detector.calls == []
    assert email.extraction["is_reciprocal_dispute"] is None


async def test_the_intent_is_never_passed_to_the_detector(session, monkeypatch):
    """D40/D47: naming the intent would lend authority to a judgment that is
    wrong in exactly the cases where the detector most needs to disagree."""
    detector = _RecordingDetector(verdict=True)
    await _run(session, monkeypatch, flag=True, detector=detector)
    call = detector.calls[0]
    assert set(call) == {"subject", "body", "transcript"}
    assert "desk_reject_appeal" not in f"{call['subject']} {call['body']}"


# ---------------------------------------------------------------------------
# The preserve rule (D42) — the core of this commit
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("verdict", [True, False])
@pytest.mark.parametrize("prior", [True, False, None])
async def test_a_real_verdict_always_overwrites_the_prior(
    session, monkeypatch, prior, verdict
):
    """Both directions, including False over True — a real answer wins."""
    monkeypatch.setattr(settings, "RECIPROCAL_DETECTOR_ENABLED", True)
    monkeypatch.setattr(orch, "detect_reciprocal_dispute", _RecordingDetector(verdict))
    c = await _pipeline().compute(
        dict(_EMAIL, prior_is_reciprocal_dispute=prior), session
    )
    assert c.record["extraction"]["is_reciprocal_dispute"] is verdict


@pytest.mark.parametrize("prior", [True, False])
async def test_detector_returning_none_preserves_a_real_prior(
    session, monkeypatch, prior
):
    """A failed/unparseable call must not downgrade a stored answer.

    Preserving a prior FALSE matters as much as a True: under the tri-state,
    letting False decay to None turns "ruled out" into "never asked" (D7).
    """
    monkeypatch.setattr(settings, "RECIPROCAL_DETECTOR_ENABLED", True)
    detector = _RecordingDetector(verdict=None)
    monkeypatch.setattr(orch, "detect_reciprocal_dispute", detector)
    c = await _pipeline().compute(
        dict(_EMAIL, prior_is_reciprocal_dispute=prior), session
    )
    assert len(detector.calls) == 1, "the detector did run — it just had no answer"
    assert c.record["extraction"]["is_reciprocal_dispute"] is prior


async def test_no_prior_and_no_answer_is_none(session, monkeypatch):
    monkeypatch.setattr(settings, "RECIPROCAL_DETECTOR_ENABLED", True)
    monkeypatch.setattr(orch, "detect_reciprocal_dispute", _RecordingDetector(None))
    c = await _pipeline().compute(dict(_EMAIL), session)
    assert c.record["extraction"]["is_reciprocal_dispute"] is None


@pytest.mark.parametrize("prior", [True, False])
async def test_the_preserve_rule_applies_WITH_THE_FLAG_OFF(session, monkeypatch, prior):
    """⚠️ The flag gates the CALL, not the preservation.

    Turning the feature off must never become a way to wipe stored answers —
    otherwise flipping the kill switch would quietly destroy data on every
    subsequent re-run.
    """
    monkeypatch.setattr(settings, "RECIPROCAL_DETECTOR_ENABLED", False)
    detector = _RecordingDetector(verdict=True)
    monkeypatch.setattr(orch, "detect_reciprocal_dispute", detector)
    c = await _pipeline().compute(
        dict(_EMAIL, prior_is_reciprocal_dispute=prior), session
    )
    assert detector.calls == []
    assert c.record["extraction"]["is_reciprocal_dispute"] is prior


async def test_gate_not_met_preserves_the_prior(session, monkeypatch):
    """Same rule on the other non-running branch."""
    monkeypatch.setattr(settings, "RECIPROCAL_DETECTOR_ENABLED", True)
    detector = _RecordingDetector(verdict=False)
    monkeypatch.setattr(orch, "detect_reciprocal_dispute", detector)
    c = await _pipeline(intent="reviewer_assignment").compute(
        dict(_EMAIL, prior_is_reciprocal_dispute=True), session
    )
    assert detector.calls == []
    assert c.record["extraction"]["is_reciprocal_dispute"] is True


@pytest.mark.parametrize("junk", ["yes", 1, 0, "", {}])
async def test_a_non_boolean_prior_is_not_treated_as_an_answer(
    session, monkeypatch, junk
):
    """Legacy/garbage values read as "not answered", never as True/False.

    `1`/`0` matter most: they are truthy/falsy but not booleans, and accepting
    them would invent an answer from a value nothing ever wrote as one.
    """
    monkeypatch.setattr(settings, "RECIPROCAL_DETECTOR_ENABLED", False)
    monkeypatch.setattr(orch, "detect_reciprocal_dispute", _RecordingDetector(None))
    c = await _pipeline().compute(
        dict(_EMAIL, prior_is_reciprocal_dispute=junk), session
    )
    assert c.record["extraction"]["is_reciprocal_dispute"] is None


# ---------------------------------------------------------------------------
# The follow-up path — the scenario the preserve rule exists for (D42)
# ---------------------------------------------------------------------------
async def test_followup_keeps_a_true_when_the_intent_drifts_off_the_gate(
    session, monkeypatch
):
    """The motivating case, end to end.

    Ticket is a reciprocal dispute; the requester then sends "any update?", so
    the intent re-classifies OFF the appeal family and the detector never runs.
    Before the preserve rule this overwrote the stored True with None.
    """
    monkeypatch.setattr(settings, "RECIPROCAL_DETECTOR_ENABLED", True)
    detector = _RecordingDetector(verdict=True)
    monkeypatch.setattr(orch, "detect_reciprocal_dispute", detector)

    pipeline = _pipeline()
    result = await pipeline.process_email(dict(_EMAIL), session)
    email = await EmailRepository().get_email_by_id(session, str(result.email_id))
    assert email.extraction["is_reciprocal_dispute"] is True, "setup: first run stores True"

    # Follow-up: the latest turn is a status chase, so the intent drifts.
    pipeline.distiller = _StubDistiller("reviewer_assignment")
    detector.calls.clear()
    await pipeline.reprocess_email_with_thread(
        session,
        email,
        [
            {
                "plain_body": "Any update on this?",
                "is_requester": True,
                "public": True,
                # A real datetime, not an ISO string — build_transcript reads
                # `.tzinfo` off this value.
                "created_at": datetime(2026, 9, 26, tzinfo=timezone.utc),
            }
        ],
    )
    refreshed = await EmailRepository().get_email_by_id(session, str(email.id))
    assert detector.calls == [], "gate not met — the detector must not run"
    assert refreshed.extraction["is_reciprocal_dispute"] is True, (
        "the stored answer was wiped by a follow-up — the D42 preserve rule "
        "is not wired on reprocess_email_with_thread"
    )


async def test_the_thread_transcript_reaches_the_detector(session, monkeypatch):
    """The detector must see the SAME input the distiller gets (D40/D9).

    ⚠️ Added because a mutation replacing `transcript=transcript_text` with
    `transcript=None` SURVIVED the first mutation pass — nothing asserted the
    transcript was threaded at all. Without it the detector would judge only
    the latest turn on a follow-up, silently contradicting the
    whole-conversation rule its own prompt states.

    Distinct from the drift test above: here the intent STAYS on the gate, so
    the detector actually runs and its input can be inspected.
    """
    monkeypatch.setattr(settings, "RECIPROCAL_DETECTOR_ENABLED", True)
    detector = _RecordingDetector(verdict=True)
    monkeypatch.setattr(orch, "detect_reciprocal_dispute", detector)

    pipeline = _pipeline()
    result = await pipeline.process_email(dict(_EMAIL), session)
    email = await EmailRepository().get_email_by_id(session, str(result.email_id))

    # First call is the single-message path: no thread exists yet.
    assert detector.calls[0]["transcript"] is None
    detector.calls.clear()

    await pipeline.reprocess_email_with_thread(
        session,
        email,
        [
            {
                "plain_body": "UNIQUE-FOLLOWUP-MARKER still disputing this.",
                "is_requester": True,
                "public": True,
                "created_at": datetime(2026, 9, 26, tzinfo=timezone.utc),
            }
        ],
    )
    assert len(detector.calls) == 1
    transcript = detector.calls[0]["transcript"]
    assert transcript is not None, "the transcript was not threaded through"
    assert "UNIQUE-FOLLOWUP-MARKER" in transcript


async def test_manual_redraft_also_preserves_the_prior(session, monkeypatch):
    """`reprocess_email` is the other row-updating call site (D42)."""
    monkeypatch.setattr(settings, "RECIPROCAL_DETECTOR_ENABLED", True)
    detector = _RecordingDetector(verdict=True)
    monkeypatch.setattr(orch, "detect_reciprocal_dispute", detector)

    pipeline = _pipeline()
    result = await pipeline.process_email(dict(_EMAIL), session)
    email = await EmailRepository().get_email_by_id(session, str(result.email_id))

    pipeline.distiller = _StubDistiller("cms_support")
    detector.calls.clear()
    await pipeline.reprocess_email(session, email)
    refreshed = await EmailRepository().get_email_by_id(session, str(email.id))
    assert detector.calls == []
    assert refreshed.extraction["is_reciprocal_dispute"] is True


def test_every_row_updating_call_site_threads_the_prior():
    """Source-level guard over all four `_compute` call sites.

    `process_email` and the public `compute` seam legitimately do NOT thread it
    — a new row has no prior, and `compute` has no Email row to read (its caller
    owns the key). The two that UPDATE an existing row must, or a re-run wipes
    the stored answer. Asserted on the source because a behavioural test can
    only cover the paths someone remembered to write.
    """
    import inspect

    for method in (EmailPipeline.reprocess_email, EmailPipeline.reprocess_email_with_thread):
        src = inspect.getsource(method)
        assert '"prior_is_reciprocal_dispute": _prior_dispute_flag(email)' in src, (
            f"{method.__name__} does not thread the prior value (D42)"
        )
