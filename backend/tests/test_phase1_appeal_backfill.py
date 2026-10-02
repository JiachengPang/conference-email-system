"""Phase-1 appeal backfill script: selection, input shape, online and batch modes.

No network: the classifier is replaced in online mode and the Batch API is an
``httpx.MockTransport``. All data is synthetic.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

import httpx
import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool

from app.core.config import settings
from app.db.models import Base
from app.pipeline.phase1_appeal_classifier import (
    SYSTEM_PROMPT,
    AppealReason,
    Phase1AppealResult,
)
from app.repositories.email_repository import EmailRepository
from app.repositories.phase1_appeal_repository import Phase1AppealRepository
from scripts import backfill_phase1_appeals as backfill

GATE_INTENT = "review_decision_appeal"
REQUESTER = 9001
START = datetime(2026, 9, 1, tzinfo=timezone.utc)
QUOTE = "the second review discusses a completely different paper"


@pytest.fixture(autouse=True)
def _settings(monkeypatch):
    monkeypatch.setattr(settings, "PHASE1_APPEAL_INTENT", GATE_INTENT)
    monkeypatch.setattr(settings, "PHASE1_APPEAL_START", START)
    monkeypatch.setattr(settings, "LOCAL_MODEL_BASE_URL", "https://batch.example.test/v1")
    monkeypatch.setattr(settings, "LOCAL_MODEL_NAME", "synthetic-batch-model")
    monkeypatch.setattr(settings, "LOCAL_MODEL_API_KEY", "synthetic-key")


@pytest_asyncio.fixture
async def factory():
    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    await engine.dispose()


async def _email(factory, *, intent=GATE_INTENT, created=datetime(2026, 9, 10, tzinfo=timezone.utc),
                 body=f"I believe {QUOTE}.", messages=(), ticket=None):
    async with factory() as db:
        email = await EmailRepository().create_email(db, {
            "sender": "author@example.edu",
            "subject": "Synthetic appeal",
            "body": body,
            "status": "draft_generated",
            "classification": {"intent": intent},
            "extraction": {"submission_numbers": ["22336"], "openreview_forum_ids": []},
            "zendesk_ticket_id": ticket,
            "zendesk_requester_id": REQUESTER,
            "zendesk_created_at": created,
        })
        if messages:
            await EmailRepository().add_thread_messages(db, str(email.id), list(messages))
        return email.id


def _msg(comment_id, text, minute, author=REQUESTER):
    return {
        "zendesk_comment_id": comment_id,
        "public": True,
        "author_id": author,
        "author_role": "end-user",
        "plain_body": text,
        "created_at": datetime(2026, 9, 10, 8, minute, tzinfo=timezone.utc),
    }


async def _rows(factory, email_id):
    async with factory() as db:
        return [r for r in await Phase1AppealRepository().list_all(db) if r.email_id == email_id]


def _result(relation="appeal", papers=("12345",)):
    return Phase1AppealResult(
        relation=relation,
        papers=list(papers),
        reasons=[AppealReason(reason="wrong_paper_review", quote=QUOTE)],
        dropped_unquoted=[],
    )


# ---------------------------------------------------------------------------
# Selection and input shape
# ---------------------------------------------------------------------------
async def test_selection_is_the_pipelines_gate(factory):
    keep = await _email(factory)
    await _email(factory, intent="reviewer_assignment")
    await _email(factory, created=datetime(2026, 8, 31, tzinfo=timezone.utc))
    on_start = await _email(factory, created=START)
    async with factory() as db:
        candidates = await backfill.load_candidates(db)
        limited = await backfill.load_candidates(db, limit=1)
    assert [c.email_id for c in candidates] == [keep, on_start]
    assert [c.email_id for c in limited] == [keep]


async def test_one_requester_message_uses_the_single_message_shape(factory):
    await _email(factory, body="Stored body.", messages=[
        _msg(1, "The requester's only message.", 1),
        _msg(2, "An agent reply.", 2, author=9002),
    ])
    async with factory() as db:
        (c,) = await backfill.load_candidates(db)
    assert c.email_data == {"subject": "Synthetic appeal", "body": "Stored body."}


async def test_a_followup_uses_the_pipelines_transcript(factory):
    await _email(factory, messages=[
        _msg(1, "FIRST-MARKER my original appeal.", 1),
        _msg(2, "LATEST-MARKER any update?", 5),
    ])
    async with factory() as db:
        (c,) = await backfill.load_candidates(db)
    assert c.email_data["body"] == "LATEST-MARKER any update?"
    transcript = c.email_data["thread_transcript"]
    assert transcript.index("FIRST-MARKER") < transcript.index("LATEST-MARKER")


# ---------------------------------------------------------------------------
# Online mode
# ---------------------------------------------------------------------------
class _Classify:
    def __init__(self, answer):
        self.answer = answer
        self.calls = 0

    async def __call__(self, email_data):
        self.calls += 1
        return self.answer


async def test_online_persists_through_the_pipelines_code_path(factory, monkeypatch):
    email_id = await _email(factory, ticket=4242)
    monkeypatch.setattr(settings, "MODEL_PROVIDER", "local")
    classify = _Classify(_result())
    counts = await backfill.run_online(factory, classify=classify)
    assert classify.calls == 1
    assert counts["classified"] == 1 and counts["replaced"] == 1
    (row,) = await _rows(factory, email_id)
    assert (row.submission_number, row.zendesk_ticket_id, row.model) == (
        "12345", 4242, "synthetic-batch-model"
    )


async def test_online_dry_run_writes_nothing(factory):
    email_id = await _email(factory)
    counts = await backfill.run_online(factory, dry_run=True, classify=_Classify(_result()))
    assert counts["classified"] == 1 and "replaced" not in counts
    assert await _rows(factory, email_id) == []


async def test_online_limit_and_failure(factory):
    first = await _email(factory)
    await _email(factory)
    classify = _Classify(None)
    counts = await backfill.run_online(factory, limit=1, classify=classify)
    assert classify.calls == 1
    assert counts["failed"] == 1 and counts["kept"] == 1
    assert await _rows(factory, first) == []


# ---------------------------------------------------------------------------
# Batch mode
# ---------------------------------------------------------------------------
def test_request_body_matches_the_online_call_minus_what_a_batch_cannot_adapt():
    body = backfill.request_body("Subject: s\nBody:\nb")
    assert set(body) == {"model", "messages", "max_completion_tokens", "seed"}
    assert body["model"] == "synthetic-batch-model"
    assert body["messages"] == [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "Subject: s\nBody:\nb"},
    ]
    assert body["max_completion_tokens"] == 4000
    assert body["seed"] == settings.DRAFTER_SEED
    assert "temperature" not in body and "max_tokens" not in body


async def test_batch_build_writes_one_request_per_candidate(factory, tmp_path):
    email_id = await _email(factory)
    await _email(factory, intent="reviewer_assignment")
    assert await backfill.batch_build(factory, tmp_path) == 1
    (line,) = (tmp_path / "input.jsonl").read_text().splitlines()
    request = json.loads(line)
    assert request["custom_id"] == f"email-{email_id}"
    assert (request["method"], request["url"]) == ("POST", "/v1/chat/completions")
    assert QUOTE in request["body"]["messages"][1]["content"]


def test_batch_submit_uploads_and_records_the_batch_id(tmp_path):
    (tmp_path / "input.jsonl").write_text('{"custom_id": "email-1"}\n')
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append((request.method, request.url.path, request.headers["authorization"]))
        if request.url.path.endswith("/files"):
            return httpx.Response(200, json={"id": "file-1"})
        assert json.loads(request.content)["input_file_id"] == "file-1"
        return httpx.Response(200, json={"id": "batch-1", "status": "validating"})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    assert backfill.batch_submit(tmp_path, client=client) == "batch-1"
    assert (tmp_path / "batch_id").read_text() == "batch-1"
    assert [s[:2] for s in seen] == [("POST", "/v1/files"), ("POST", "/v1/batches")]
    assert all(s[2] == "Bearer synthetic-key" for s in seen)


def test_batch_download_polls_until_done(tmp_path):
    (tmp_path / "batch_id").write_text("batch-1")
    statuses = iter(["in_progress", "completed"])
    sleeps = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/content"):
            return httpx.Response(200, text='{"custom_id": "email-1"}\n')
        status = next(statuses)
        return httpx.Response(200, json={
            "status": status, "output_file_id": "out-1" if status == "completed" else None,
        })

    client = httpx.Client(transport=httpx.MockTransport(handler))
    raw = backfill.batch_download(tmp_path, client=client, sleep=sleeps.append, poll_seconds=5)
    assert raw == '{"custom_id": "email-1"}\n'
    assert (tmp_path / "output_raw.jsonl").read_text() == raw
    assert sleeps == [5]


def _answer(email_id, content):
    return json.dumps({
        "custom_id": f"email-{email_id}",
        "response": {"body": {"choices": [{"message": {"content": content}}]}},
    })


async def test_batch_apply_parses_and_persists(factory, tmp_path):
    appeal = await _email(factory, ticket=1)
    garbage = await _email(factory, ticket=2)
    missing = await _email(factory, ticket=3)
    not_appeal = await _email(factory, ticket=4)
    for email_id in (garbage, missing, not_appeal):
        async with factory() as db:
            await Phase1AppealRepository().replace_for_email(db, email_id, [{
                "submission_number": "11111", "relation": "appeal", "reasons": [],
                "must_verify": False, "prompt_sha256": "0" * 64, "model": "earlier",
            }])
    await backfill.batch_build(factory, tmp_path)
    answer = json.dumps({"relation": "appeal", "papers": ["12345"],
                         "reasons": [{"reason": "wrong_paper_review", "quote": QUOTE}]})
    raw = "\n".join([
        _answer(appeal, answer),
        _answer(garbage, "not the contract"),
        _answer(not_appeal, json.dumps({"relation": "not_appeal", "papers": [], "reasons": []})),
    ])
    counts = await backfill.batch_apply(factory, tmp_path, raw)
    assert counts["classified"] == 1 and counts["failed"] == 2 and counts["not_appeal"] == 1
    (row,) = await _rows(factory, appeal)
    assert (row.submission_number, row.must_verify, row.model) == (
        "12345", True, "synthetic-batch-model"
    )
    assert [r.model for r in await _rows(factory, garbage)] == ["earlier"]
    assert [r.model for r in await _rows(factory, missing)] == ["earlier"]
    assert await _rows(factory, not_appeal) == []


async def test_batch_apply_dry_run_writes_nothing(factory, tmp_path):
    email_id = await _email(factory)
    await backfill.batch_build(factory, tmp_path)
    answer = json.dumps({"relation": "appeal", "papers": [],
                         "reasons": [{"reason": "wrong_paper_review", "quote": QUOTE}]})
    counts = await backfill.batch_apply(
        factory, tmp_path, _answer(email_id, answer), dry_run=True
    )
    assert counts["classified"] == 1
    assert await _rows(factory, email_id) == []


def test_batch_mode_needs_a_work_dir():
    with pytest.raises(SystemExit):
        backfill.main(["--batch", "build"])
