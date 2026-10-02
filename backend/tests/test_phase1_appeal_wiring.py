"""Phase-1 appeal wiring in the orchestrator: the gate, the outcome rules, and
persistence on the create path and every update path.

The classifier's own behaviour is tested in test_phase1_appeal_classifier.py;
here it is stubbed with a recorder (or its transport is), so no test makes a
model call. All text below is synthetic.
"""

from __future__ import annotations

import inspect
import logging
from datetime import datetime, timezone

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool

from app.core.config import settings
from app.db.models import Base
from app.pipeline import orchestrator as orch
from app.pipeline import phase1_appeal_classifier as p1c
from app.pipeline import phase1_appeal_outcome as outcome_rules
from app.pipeline.distiller import DistillResult
from app.pipeline.orchestrator import EmailPipeline
from app.pipeline.phase1_appeal_classifier import (
    PROMPT_SHA256,
    AppealReason,
    Phase1AppealResult,
)
from app.repositories.email_repository import EmailRepository
from app.repositories.phase1_appeal_repository import (
    PaperAssignmentRepository,
    Phase1AppealRepository,
)

GATE_INTENT = "review_decision_appeal"
SUBJECT_MARKER = "SYNTHETIC-SUBJECT-MARKER"
BODY_MARKER = "SYNTHETIC-BODY-MARKER"
QUOTE = "the second review discusses a completely different paper"

_EMAIL = {
    "from": "author@example.edu",
    "subject": f"Appeal of rejection {SUBJECT_MARKER}",
    "body": f"I believe {QUOTE}. {BODY_MARKER}",
}

_FOLLOWUP = [{
    "plain_body": "UNIQUE-FOLLOWUP-MARKER is there any update on my appeal?",
    "public": True,
    "author_role": "end-user",
    "created_at": datetime(2026, 9, 28, tzinfo=timezone.utc),
}]


@pytest.fixture(autouse=True)
def _settings(monkeypatch):
    # The distiller is what reports the stubbed intent; under the conftest's
    # `prefix` strategy the keyword classifier would pick it instead.
    monkeypatch.setattr(settings, "QUERY_STRATEGY", "distill")
    monkeypatch.setattr(settings, "RECIPROCAL_DETECTOR_ENABLED", False)
    monkeypatch.setattr(settings, "APPEAL_REASON_CLASSIFIER_ENABLED", False)
    monkeypatch.setattr(settings, "PHASE1_APPEAL_ENABLED", True)
    monkeypatch.setattr(settings, "PHASE1_APPEAL_START", None)
    monkeypatch.setattr(settings, "PHASE1_APPEAL_INTENT", GATE_INTENT)


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
    def __init__(self, intent=GATE_INTENT, numbers=(), forum_ids=()):
        self.intent = intent
        self.numbers = list(numbers)
        self.forum_ids = list(forum_ids)

    async def distill(self, subject, body, *, transcript=None):
        return DistillResult(
            queries=["appeal of review decision"],
            intent=self.intent,
            confidence=0.9,
            submission_numbers_raw=self.numbers,
            openreview_ids_raw=self.forum_ids,
        )


class _StubRetriever:
    async def retrieve(self, query, intent, top_k=3, *, prior_intent=""):
        return []


class _Recorder:
    """Records each call's email_data and returns a fixed answer (never raises)."""

    def __init__(self, answer=None):
        self.answer = answer
        self.calls: list[dict] = []

    async def __call__(self, email_data):
        self.calls.append(dict(email_data))
        return self.answer


def _result(relation="appeal", papers=("12345",), reasons=("wrong_paper_review",)):
    return Phase1AppealResult(
        relation=relation,
        papers=list(papers),
        reasons=[AppealReason(reason=r, quote=QUOTE) for r in reasons],
        dropped_unquoted=[],
    )


def _pipeline(intent=GATE_INTENT, numbers=(), forum_ids=()):
    p = EmailPipeline()
    p.distiller = _StubDistiller(intent, numbers, forum_ids)
    p.retriever = _StubRetriever()
    return p


def _install(monkeypatch, answer=None):
    recorder = _Recorder(answer)
    monkeypatch.setattr(orch, "classify_phase1_appeal", recorder)
    return recorder


async def _rows(session, email_id):
    return [r for r in await Phase1AppealRepository().list_all(session) if r.email_id == email_id]


async def _create(session, pipeline, **extra):
    result = await pipeline.process_email({**_EMAIL, **extra}, session)
    return await EmailRepository().get_email_by_id(session, str(result.email_id))


_OLD_ROW = {
    "submission_number": "11111",
    "relation": "appeal",
    "reasons": [{"reason": "other", "quote": "an earlier quote"}],
    "must_verify": False,
    "prompt_sha256": "0" * 64,
    "model": "earlier-model",
}


async def _email_with_old_row(session, monkeypatch, pipeline):
    """An email created with the feature off, plus one stored appeal row."""
    monkeypatch.setattr(settings, "PHASE1_APPEAL_ENABLED", False)
    email = await _create(session, pipeline)
    await Phase1AppealRepository().replace_for_email(session, email.id, [_OLD_ROW])
    monkeypatch.setattr(settings, "PHASE1_APPEAL_ENABLED", True)
    return email


async def _reprocess(path, pipeline, session, email):
    if path == "reprocess_email":
        await pipeline.reprocess_email(session, email)
    else:
        await pipeline.reprocess_email_with_thread(session, email, _FOLLOWUP)


UPDATE_PATHS = ("reprocess_email", "reprocess_email_with_thread")


# ---------------------------------------------------------------------------
# The flag: off means no call and no row changes
# ---------------------------------------------------------------------------
async def test_flag_off_create_makes_no_call_and_no_rows(session, monkeypatch):
    monkeypatch.setattr(settings, "PHASE1_APPEAL_ENABLED", False)
    recorder = _install(monkeypatch, _result())
    email = await _create(session, _pipeline())
    assert recorder.calls == []
    assert await _rows(session, email.id) == []


@pytest.mark.parametrize("path", UPDATE_PATHS)
async def test_flag_off_reprocess_makes_no_call_and_keeps_rows(session, monkeypatch, path):
    pipeline = _pipeline()
    email = await _email_with_old_row(session, monkeypatch, pipeline)
    monkeypatch.setattr(settings, "PHASE1_APPEAL_ENABLED", False)
    recorder = _install(monkeypatch, _result(relation="not_appeal"))
    await _reprocess(path, pipeline, session, email)
    assert recorder.calls == []
    assert [r.submission_number for r in await _rows(session, email.id)] == ["11111"]


@pytest.mark.parametrize("flag", [False, True])
async def test_the_model_transport_is_reached_only_with_the_flag_on(session, monkeypatch, flag):
    """The REAL classifier with only its transport replaced: flag off never
    reaches it, flag on reaches it exactly once."""
    calls: list[str] = []

    async def transport(user):  # noqa: ANN001
        calls.append(user)
        return "not the contract"

    monkeypatch.setattr(p1c, "_call_model", transport)
    monkeypatch.setattr(settings, "PHASE1_APPEAL_ENABLED", flag)
    email = await _create(session, _pipeline())
    assert len(calls) == (1 if flag else 0)
    assert await _rows(session, email.id) == []


# ---------------------------------------------------------------------------
# Create path
# ---------------------------------------------------------------------------
async def test_create_writes_one_row_per_paper_with_assignment_fields(session, monkeypatch):
    await PaperAssignmentRepository().upsert_many(session, [{
        "paper_number": "12345",
        "apc_name": "Synthetic APC",
        "openreview_url": "https://openreview.net/forum?id=Ab3xY9kLm2",
        "openreview_forum_id": "Ab3xY9kLm2",
    }])
    recorder = _install(monkeypatch, _result(papers=("12345", "67890")))
    email = await _create(session, _pipeline(), zendesk_ticket_id=4242)
    assert len(recorder.calls) == 1
    rows = await _rows(session, email.id)
    assert [r.submission_number for r in rows] == ["12345", "67890"]
    first, second = rows
    assert (first.apc_name, first.openreview_url) == (
        "Synthetic APC", "https://openreview.net/forum?id=Ab3xY9kLm2"
    )
    assert (second.apc_name, second.openreview_url) == (None, None)
    for row in rows:
        assert row.zendesk_ticket_id == 4242
        assert row.relation == "appeal"
        assert row.reasons == [{"reason": "wrong_paper_review", "quote": QUOTE}]
        assert row.must_verify is True
        assert row.prompt_sha256 == PROMPT_SHA256


async def test_feedback_only_is_stored(session, monkeypatch):
    _install(monkeypatch, _result(relation="feedback_only", reasons=("reviewer_misconduct",)))
    email = await _create(session, _pipeline())
    (row,) = await _rows(session, email.id)
    assert row.relation == "feedback_only"
    assert row.must_verify is False


@pytest.mark.parametrize(
    "provider, field, value",
    [("local", "LOCAL_MODEL_NAME", "synthetic-local-model"),
     ("anthropic", "DRAFT_MODEL", "synthetic-hosted-model")],
)
async def test_model_is_the_configured_id_of_the_active_provider(
    session, monkeypatch, provider, field, value
):
    pipeline = _pipeline()  # built under `fallback`, so the drafter stays offline
    monkeypatch.setattr(settings, "MODEL_PROVIDER", provider)
    monkeypatch.setattr(settings, field, value)
    _install(monkeypatch, _result())
    email = await _create(session, pipeline)
    (row,) = await _rows(session, email.id)
    assert row.model == value


def test_active_model_id_per_provider(monkeypatch):
    monkeypatch.setattr(settings, "LOCAL_MODEL_NAME", "m-local")
    monkeypatch.setattr(settings, "DRAFT_MODEL", "m-hosted")
    for provider, expected in [
        ("local", "m-local"), ("anthropic", "m-hosted"), ("anthropic_api", "m-hosted"),
        ("template", None), ("fallback", None),
    ]:
        monkeypatch.setattr(settings, "MODEL_PROVIDER", provider)
        assert outcome_rules.active_model_id() == expected


async def test_create_not_appeal_writes_no_rows(session, monkeypatch):
    recorder = _install(monkeypatch, _result(relation="not_appeal"))
    email = await _create(session, _pipeline())
    assert len(recorder.calls) == 1
    assert await _rows(session, email.id) == []


# ---------------------------------------------------------------------------
# Update paths: each outcome rule
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("path", UPDATE_PATHS)
async def test_appeal_replaces_the_existing_rows(session, monkeypatch, path):
    pipeline = _pipeline()
    email = await _email_with_old_row(session, monkeypatch, pipeline)
    _install(monkeypatch, _result(papers=("12345",)))
    await _reprocess(path, pipeline, session, email)
    rows = await _rows(session, email.id)
    assert [r.submission_number for r in rows] == ["12345"]
    assert rows[0].prompt_sha256 == PROMPT_SHA256


@pytest.mark.parametrize("path", UPDATE_PATHS)
async def test_failure_keeps_the_existing_rows(session, monkeypatch, path):
    pipeline = _pipeline()
    email = await _email_with_old_row(session, monkeypatch, pipeline)
    recorder = _install(monkeypatch, None)
    await _reprocess(path, pipeline, session, email)
    assert len(recorder.calls) == 1
    rows = await _rows(session, email.id)
    assert [(r.submission_number, r.model) for r in rows] == [("11111", "earlier-model")]


@pytest.mark.parametrize("path", UPDATE_PATHS)
async def test_not_appeal_deletes_the_rows(session, monkeypatch, path):
    pipeline = _pipeline()
    email = await _email_with_old_row(session, monkeypatch, pipeline)
    recorder = _install(monkeypatch, _result(relation="not_appeal"))
    await _reprocess(path, pipeline, session, email)
    assert len(recorder.calls) == 1
    assert await _rows(session, email.id) == []


@pytest.mark.parametrize("path", UPDATE_PATHS)
async def test_gate_not_met_deletes_the_rows_without_a_call(session, monkeypatch, path):
    """The intent moved off the gate on reprocess: earlier rows must not linger."""
    pipeline = _pipeline()
    email = await _email_with_old_row(session, monkeypatch, pipeline)
    recorder = _install(monkeypatch, _result())
    pipeline.distiller = _StubDistiller("reviewer_assignment")
    await _reprocess(path, pipeline, session, email)
    assert recorder.calls == []
    assert await _rows(session, email.id) == []


async def test_followup_hands_the_transcript_to_the_classifier(session, monkeypatch):
    pipeline = _pipeline()
    email = await _email_with_old_row(session, monkeypatch, pipeline)
    recorder = _install(monkeypatch, _result())
    await pipeline.reprocess_email_with_thread(session, email, _FOLLOWUP)
    (call,) = recorder.calls
    assert "UNIQUE-FOLLOWUP-MARKER" in (call.get("thread_transcript") or "")


async def test_reprocess_keeps_the_ticket_id_of_the_row(session, monkeypatch):
    pipeline = _pipeline()
    email = await _email_with_old_row(session, monkeypatch, pipeline)
    await EmailRepository().apply_zendesk_fields(
        session, str(email.id), {"zendesk_ticket_id": 777}
    )
    email = await EmailRepository().get_email_by_id(session, str(email.id))
    _install(monkeypatch, _result())
    await pipeline.reprocess_email(session, email)
    (row,) = await _rows(session, email.id)
    assert row.zendesk_ticket_id == 777


# ---------------------------------------------------------------------------
# The date gate
# ---------------------------------------------------------------------------
START = datetime(2026, 9, 1, tzinfo=timezone.utc)


@pytest.mark.parametrize(
    "timestamp, called",
    [("2026-08-31T23:59:59Z", False), ("2026-09-01T00:00:00Z", True),
     ("2026-09-15T10:00:00Z", True)],
)
async def test_ingest_date_gate_reads_the_email_timestamp(session, monkeypatch, timestamp, called):
    monkeypatch.setattr(settings, "PHASE1_APPEAL_START", START)
    recorder = _install(monkeypatch, _result())
    email = await _create(session, _pipeline(), timestamp=timestamp)
    assert len(recorder.calls) == (1 if called else 0)
    assert len(await _rows(session, email.id)) == (1 if called else 0)


async def test_unknown_creation_time_meets_the_date_gate(session, monkeypatch):
    monkeypatch.setattr(settings, "PHASE1_APPEAL_START", START)
    recorder = _install(monkeypatch, _result())
    await _create(session, _pipeline())  # no timestamp at all
    assert len(recorder.calls) == 1


async def test_a_naive_start_is_read_as_utc(session, monkeypatch):
    monkeypatch.setattr(settings, "PHASE1_APPEAL_START", datetime(2026, 9, 1, 12, 0))
    recorder = _install(monkeypatch, _result())
    await _create(session, _pipeline(), timestamp="2026-09-01T07:00:00-05:00")  # 12:00Z
    assert len(recorder.calls) == 1


@pytest.mark.parametrize("path", UPDATE_PATHS)
async def test_reprocess_date_gate_prefers_zendesk_created_at(session, monkeypatch, path):
    """received_at is after the start but the ticket was created before it: the
    ticket's own creation time decides, so the call is skipped and rows go."""
    pipeline = _pipeline()
    monkeypatch.setattr(settings, "PHASE1_APPEAL_ENABLED", False)
    email = await _create(session, pipeline, timestamp="2026-09-20T00:00:00Z")
    await Phase1AppealRepository().replace_for_email(session, email.id, [_OLD_ROW])
    await EmailRepository().apply_zendesk_fields(
        session, str(email.id),
        {"zendesk_created_at": datetime(2026, 8, 20, tzinfo=timezone.utc)},
    )
    email = await EmailRepository().get_email_by_id(session, str(email.id))
    monkeypatch.setattr(settings, "PHASE1_APPEAL_ENABLED", True)
    monkeypatch.setattr(settings, "PHASE1_APPEAL_START", START)
    recorder = _install(monkeypatch, _result())
    await _reprocess(path, pipeline, session, email)
    assert recorder.calls == []
    assert await _rows(session, email.id) == []


@pytest.mark.parametrize(
    "received, called", [("2026-08-20T00:00:00Z", False), ("2026-09-20T00:00:00Z", True)]
)
async def test_reprocess_date_gate_falls_back_to_received_at(session, monkeypatch, received, called):
    pipeline = _pipeline()
    monkeypatch.setattr(settings, "PHASE1_APPEAL_ENABLED", False)
    email = await _create(session, pipeline, timestamp=received)
    monkeypatch.setattr(settings, "PHASE1_APPEAL_ENABLED", True)
    monkeypatch.setattr(settings, "PHASE1_APPEAL_START", START)
    recorder = _install(monkeypatch, _result())
    await pipeline.reprocess_email(session, email)
    assert len(recorder.calls) == (1 if called else 0)


# ---------------------------------------------------------------------------
# Paper resolution order
# ---------------------------------------------------------------------------
FORUM_ID = "Ab3xY9kLm2"


async def _seed_forum_paper(session):
    await PaperAssignmentRepository().upsert_many(session, [{
        "paper_number": "55555",
        "apc_name": "Forum APC",
        "openreview_url": f"https://openreview.net/forum?id={FORUM_ID}",
        "openreview_forum_id": FORUM_ID,
    }])


async def test_classifier_papers_come_first(session, monkeypatch):
    await _seed_forum_paper(session)
    _install(monkeypatch, _result(papers=("12345",)))
    email = await _create(session, _pipeline(numbers=["22336"], forum_ids=[FORUM_ID]))
    assert [r.submission_number for r in await _rows(session, email.id)] == ["12345"]


async def test_extraction_numbers_when_the_classifier_names_none(session, monkeypatch):
    await _seed_forum_paper(session)
    _install(monkeypatch, _result(papers=()))
    email = await _create(session, _pipeline(numbers=["22336", "33447"], forum_ids=[FORUM_ID]))
    assert [r.submission_number for r in await _rows(session, email.id)] == ["22336", "33447"]


async def test_forum_ids_map_through_the_sheet_when_no_number_is_found(session, monkeypatch):
    await _seed_forum_paper(session)
    _install(monkeypatch, _result(papers=()))
    email = await _create(session, _pipeline(forum_ids=[FORUM_ID]))
    (row,) = await _rows(session, email.id)
    assert (row.submission_number, row.apc_name) == ("55555", "Forum APC")


@pytest.mark.parametrize("forum_ids", [[], ["Zz9yX8wV7u"]])
async def test_no_paper_at_all_gives_one_row_without_a_number(session, monkeypatch, forum_ids):
    await _seed_forum_paper(session)
    _install(monkeypatch, _result(papers=()))
    email = await _create(session, _pipeline(forum_ids=forum_ids))
    (row,) = await _rows(session, email.id)
    assert (row.submission_number, row.apc_name, row.openreview_url) == (None, None, None)


# ---------------------------------------------------------------------------
# The public compute seam is compute-only
# ---------------------------------------------------------------------------
async def test_the_compute_seam_writes_no_rows(session, monkeypatch):
    """The seam carries the outcome like `_compute` does, but never persists it,
    not even a delete."""
    pipeline = _pipeline()
    email = await _email_with_old_row(session, monkeypatch, pipeline)
    _install(monkeypatch, _result(relation="not_appeal"))
    c = await pipeline.compute(dict(_EMAIL), session)
    assert c.phase1.state == outcome_rules.NOT_APPEAL
    assert [r.submission_number for r in await _rows(session, email.id)] == ["11111"]


async def test_compute_carries_the_outcome_without_writing(session, monkeypatch):
    """`_compute` itself decides but never persists."""
    recorder = _install(monkeypatch, _result())
    c = await _pipeline()._compute(dict(_EMAIL), session)
    assert len(recorder.calls) == 1
    assert c.phase1.state == outcome_rules.CLASSIFIED
    assert await Phase1AppealRepository().list_all(session) == []


# ---------------------------------------------------------------------------
# Best-effort persistence
# ---------------------------------------------------------------------------
class _Boom(Exception):
    pass


async def _raise(*args, **kwargs):
    raise _Boom("synthetic failure")


@pytest.mark.parametrize("method", ["replace_for_email", "delete_for_email"])
async def test_a_persistence_failure_never_breaks_the_pipeline(session, monkeypatch, method):
    monkeypatch.setattr(Phase1AppealRepository, method, _raise)
    relation = "appeal" if method == "replace_for_email" else "not_appeal"
    _install(monkeypatch, _result(relation=relation))
    pipeline = _pipeline()
    result = await pipeline.process_email(dict(_EMAIL), session)
    email = await EmailRepository().get_email_by_id(session, result.email_id)
    assert email is not None and email.draft is not None
    # The update path survives too, and the row stays updated.
    await pipeline.reprocess_email(session, email)
    assert (await EmailRepository().get_email_by_id(session, result.email_id)).redrafting is False


async def test_an_assignment_lookup_failure_never_breaks_the_pipeline(session, monkeypatch):
    monkeypatch.setattr(PaperAssignmentRepository, "get_by_numbers", _raise)
    _install(monkeypatch, _result())
    result = await _pipeline().process_email(dict(_EMAIL), session)
    assert result.status in ("complete", "draft_failed", "draft_truncated")
    assert await Phase1AppealRepository().list_all(session) == []


# ---------------------------------------------------------------------------
# Logging: ids and states only
# ---------------------------------------------------------------------------
async def test_logs_carry_no_email_text(session, monkeypatch, caplog):
    _install(monkeypatch, _result())
    with caplog.at_level(logging.DEBUG, logger="app"):
        email = await _create(session, _pipeline())
    # The app's own loggers (the database driver's debug log echoes SQL params).
    ours = [r for r in caplog.records if r.name.startswith("app.")]
    phase1_lines = [r.getMessage() for r in ours if "Phase-1 appeal" in r.getMessage()]
    assert any(f"email={email.id}" in line for line in phase1_lines)
    for record in ours:
        text = record.getMessage()
        assert SUBJECT_MARKER not in text
        assert BODY_MARKER not in text
        assert QUOTE not in text


# ---------------------------------------------------------------------------
# Every row-writing path persists
# ---------------------------------------------------------------------------
def test_every_row_writing_path_persists_the_outcome():
    """Source-level guard: a behavioural test only covers the paths someone
    remembered to write."""
    for method in (
        EmailPipeline.process_email,
        EmailPipeline.reprocess_email,
        EmailPipeline.reprocess_email_with_thread,
    ):
        assert "await self._persist_phase1(" in inspect.getsource(method), method.__name__
    assert "_persist_phase1" not in inspect.getsource(EmailPipeline.compute)
