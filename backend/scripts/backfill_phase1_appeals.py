"""Backfill phase-1 appeal rows for emails already in the database.

Selects emails whose stored intent is ``PHASE1_APPEAL_INTENT`` and whose ticket
was created on or after ``PHASE1_APPEAL_START`` (the pipeline's own gate),
classifies each one and persists the outcome through
``app.pipeline.phase1_appeal_outcome.persist_phase1_outcome``, the same code
path the pipeline uses. The model input matches the pipeline's: subject + body
for a single message, and the pipeline's thread transcript once the requester
has written more than once.

Modes:
  (default)               online: one classifier call per email, 3 at a time.
                          --limit N caps the number of emails; --dry-run
                          classifies and reports counts but writes nothing.
  --batch build           write <work-dir>/input.jsonl for the OpenAI Batch API.
  --batch submit          upload it to LOCAL_MODEL_BASE_URL and create the
                          batch; the batch id goes to <work-dir>/batch_id.
  --batch collect         wait for the batch, download the answers, parse them
                          with the classifier's parser and persist (--dry-run:
                          parse and report only).

Batch requests use the online call's body except for two parameters a batch
cannot adapt at run time: ``max_completion_tokens`` instead of ``max_tokens``,
and no ``temperature``. The work directory holds email text; keep it out of the
repository. Output names email ids and states only.

Run with:  cd backend && python scripts/backfill_phase1_appeals.py [--limit N] [--dry-run]
           cd backend && python scripts/backfill_phase1_appeals.py --batch build --work-dir DIR
"""

import argparse
import asyncio
import json
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import httpx

# scripts/backfill_phase1_appeals.py -> parents[1] is backend/ (put it on
# sys.path so `app` imports work when run as a script).
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.config import settings  # noqa: E402
from app.pipeline import phase1_appeal_outcome as outcome_rules  # noqa: E402
from app.pipeline.phase1_appeal_classifier import (  # noqa: E402
    _MAX_TOKENS,
    SYSTEM_PROMPT,
    build_user_prompt,
    classify_phase1_appeal,
    parse_answer,
)
from app.pipeline.thread_transcript import build_transcript  # noqa: E402
from app.repositories.email_repository import EmailRepository  # noqa: E402

DEFAULT_CONCURRENCY = 3
BATCH_ENDPOINT = "/v1/chat/completions"
BATCH_TERMINAL = ("completed", "failed", "expired", "cancelled")
POLL_SECONDS = 60
INPUT_FILE = "input.jsonl"
OUTPUT_FILE = "output_raw.jsonl"
BATCH_ID_FILE = "batch_id"
_CUSTOM_ID_PREFIX = "email-"


@dataclass
class Candidate:
    """One email to classify, with everything the run needs after the read."""

    email_id: int
    zendesk_ticket_id: int | None
    extraction: dict | None
    email_data: dict


def _requester_turns(email, messages: list[dict]) -> int:
    requester_id = email.zendesk_requester_id
    if requester_id is None:
        return 0
    return sum(
        1
        for m in messages
        if m.get("public")
        and (m.get("plain_body") or "").strip()
        and m.get("author_id") == requester_id
    )


async def email_data_for(db, email, email_repo: EmailRepository) -> dict:
    """The classifier input the pipeline would build for this email.

    One requester message: the ingest shape (subject + stored body). More than
    one: the follow-up shape, built exactly as ``reprocess_email_with_thread``
    builds it.
    """
    messages = await email_repo.get_thread_messages(db, str(email.id))
    if _requester_turns(email, messages) <= 1:
        return {"subject": email.subject, "body": email.body}
    transcript = build_transcript(
        messages,
        char_budget=settings.THREAD_TRANSCRIPT_MAX_CHARS,
        requester_id=email.zendesk_requester_id,
    )
    return {
        "subject": email.subject,
        "body": transcript.latest_requester_message or email.body,
        "thread_transcript": transcript.text,
    }


async def load_candidates(
    db, *, limit: int | None = None, email_repo: EmailRepository | None = None
) -> list[Candidate]:
    """Emails that meet the pipeline's phase-1 gate, oldest id first."""
    email_repo = email_repo or EmailRepository()
    emails = await email_repo.get_emails_by_intent(db, settings.PHASE1_APPEAL_INTENT)
    candidates = []
    for email in emails:
        intent = (email.classification or {}).get("intent")
        if not outcome_rules.gate_met(intent, outcome_rules.ticket_created_at(email)):
            continue
        candidates.append(
            Candidate(
                email_id=email.id,
                zendesk_ticket_id=email.zendesk_ticket_id,
                extraction=email.extraction,
                email_data=await email_data_for(db, email, email_repo),
            )
        )
        if limit is not None and len(candidates) >= limit:
            break
    return candidates


async def run_online(
    session_factory,
    *,
    limit: int | None = None,
    dry_run: bool = False,
    concurrency: int = DEFAULT_CONCURRENCY,
    classify=classify_phase1_appeal,
) -> Counter:
    """Classify each candidate and persist its outcome; returns state/action counts."""
    async with session_factory() as db:
        candidates = await load_candidates(db, limit=limit)
    counts: Counter = Counter(candidates=len(candidates))
    semaphore = asyncio.Semaphore(concurrency)
    # Classification runs concurrently; writes go one at a time, each in its own
    # session (a session is not safe to share across tasks).
    write_lock = asyncio.Lock()
    model = outcome_rules.active_model_id()

    async def one(candidate: Candidate) -> None:
        async with semaphore:
            outcome = outcome_rules.outcome_for(await classify(candidate.email_data))
        counts[outcome.state] += 1
        if dry_run:
            return
        async with write_lock, session_factory() as db:
            action = await outcome_rules.persist_phase1_outcome(
                db,
                email_id=candidate.email_id,
                zendesk_ticket_id=candidate.zendesk_ticket_id,
                extraction=candidate.extraction,
                outcome=outcome,
                model=model,
            )
        counts[action] += 1

    await asyncio.gather(*(one(c) for c in candidates))
    return counts


# --- Batch API --------------------------------------------------------------
def request_body(user: str) -> dict:
    """The online call's request body, minus what a batch cannot adapt."""
    return {
        "model": settings.LOCAL_MODEL_NAME,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user},
        ],
        "max_completion_tokens": _MAX_TOKENS,
        "seed": settings.DRAFTER_SEED,
    }


def _custom_id(email_id: int) -> str:
    return f"{_CUSTOM_ID_PREFIX}{email_id}"


def _email_id(custom_id: str) -> int:
    return int(custom_id.removeprefix(_CUSTOM_ID_PREFIX))


def _api_base() -> str:
    return settings.LOCAL_MODEL_BASE_URL.rstrip("/")


def _headers() -> dict:
    if not settings.LOCAL_MODEL_API_KEY:
        raise SystemExit("LOCAL_MODEL_API_KEY is not set; the Batch API needs it.")
    return {"Authorization": f"Bearer {settings.LOCAL_MODEL_API_KEY}"}


async def batch_build(session_factory, work_dir: Path, *, limit: int | None = None) -> int:
    """Write one batch request per candidate; returns the request count."""
    async with session_factory() as db:
        candidates = await load_candidates(db, limit=limit)
    work_dir.mkdir(parents=True, exist_ok=True)
    with open(work_dir / INPUT_FILE, "w", encoding="utf-8") as fh:
        for c in candidates:
            user = build_user_prompt(
                c.email_data.get("subject") or "",
                c.email_data.get("body") or "",
                c.email_data.get("thread_transcript"),
            )
            request = {
                "custom_id": _custom_id(c.email_id),
                "method": "POST",
                "url": BATCH_ENDPOINT,
                "body": request_body(user),
            }
            fh.write(json.dumps(request, ensure_ascii=False) + "\n")
    return len(candidates)


def batch_submit(work_dir: Path, *, client: httpx.Client | None = None) -> str:
    """Upload the input file and create the batch; returns the batch id."""
    client = client or httpx.Client(timeout=120)
    with open(work_dir / INPUT_FILE, "rb") as fh:
        upload = client.post(
            f"{_api_base()}/files",
            headers=_headers(),
            files={"file": (INPUT_FILE, fh)},
            data={"purpose": "batch"},
        )
    upload.raise_for_status()
    batch = client.post(
        f"{_api_base()}/batches",
        headers=_headers(),
        json={
            "input_file_id": upload.json()["id"],
            "endpoint": BATCH_ENDPOINT,
            "completion_window": "24h",
        },
    )
    batch.raise_for_status()
    batch_id = batch.json()["id"]
    (work_dir / BATCH_ID_FILE).write_text(batch_id)
    return batch_id


def _answer_text(record: dict) -> str | None:
    try:
        return record["response"]["body"]["choices"][0]["message"]["content"]
    except (KeyError, TypeError, IndexError):
        return None


def batch_download(
    work_dir: Path,
    *,
    client: httpx.Client | None = None,
    sleep=time.sleep,
    poll_seconds: int = POLL_SECONDS,
) -> str:
    """Wait for the batch to finish and save its raw output; returns that text."""
    client = client or httpx.Client(timeout=120)
    batch_id = (work_dir / BATCH_ID_FILE).read_text().strip()
    while True:
        batch = client.get(f"{_api_base()}/batches/{batch_id}", headers=_headers()).json()
        print(f"batch {batch_id}: {batch.get('status')} {batch.get('request_counts')}")
        if batch.get("status") in BATCH_TERMINAL:
            break
        sleep(poll_seconds)
    if not batch.get("output_file_id"):
        raise SystemExit(f"batch {batch_id} has no output (status {batch.get('status')}).")
    raw = client.get(
        f"{_api_base()}/files/{batch['output_file_id']}/content", headers=_headers()
    ).text
    (work_dir / OUTPUT_FILE).write_text(raw, encoding="utf-8")
    return raw


async def batch_apply(
    session_factory, work_dir: Path, raw: str, *, dry_run: bool = False
) -> Counter:
    """Parse the batch answers and persist each outcome; returns counts.

    Quotes are checked against the user prompt that was sent, read back from the
    input file. An email whose request has no usable answer is a failure (its
    rows are kept). Each email's gate is re-checked against its current row, so
    an email whose intent changed since the build is treated as the pipeline
    would treat it now.
    """
    sent: dict[int, str] = {}
    with open(work_dir / INPUT_FILE, encoding="utf-8") as fh:
        for line in fh:
            request = json.loads(line)
            sent[_email_id(request["custom_id"])] = request["body"]["messages"][1]["content"]
    answers: dict[int, str | None] = {}
    for line in raw.splitlines():
        if line.strip():
            record = json.loads(line)
            answers[_email_id(record["custom_id"])] = _answer_text(record)

    counts: Counter = Counter(requests=len(sent))
    email_repo = EmailRepository()
    for email_id, user in sent.items():
        async with session_factory() as db:
            email = await email_repo.get_email_by_id(db, str(email_id))
            if email is None:
                counts["email_missing"] += 1
                continue
            intent = (email.classification or {}).get("intent")
            if not outcome_rules.gate_met(intent, outcome_rules.ticket_created_at(email)):
                outcome = outcome_rules.Phase1Outcome(outcome_rules.GATE_NOT_MET)
            else:
                text = answers.get(email_id)
                result = parse_answer(text, user) if text else None
                outcome = outcome_rules.outcome_for(result)
            counts[outcome.state] += 1
            if dry_run:
                continue
            action = await outcome_rules.persist_phase1_outcome(
                db,
                email_id=email.id,
                zendesk_ticket_id=email.zendesk_ticket_id,
                extraction=email.extraction,
                outcome=outcome,
                # A batch always runs on the Batch API's model.
                model=settings.LOCAL_MODEL_NAME,
            )
            counts[action] += 1
    return counts


def _print_counts(counts: Counter) -> None:
    for key in sorted(counts):
        print(f"{key + ':':<16}{counts[key]}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--batch", choices=("build", "submit", "collect"))
    parser.add_argument("--work-dir", type=Path)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY)
    args = parser.parse_args(argv)
    if args.batch and args.work_dir is None:
        parser.error("--batch needs --work-dir")

    from app.db.database import async_session_factory

    if args.batch is None:
        if outcome_rules.active_model_id() is None:
            print(
                f"MODEL_PROVIDER={settings.MODEL_PROVIDER} has no model; every "
                "email will count as failed and no rows will change.",
                file=sys.stderr,
            )
        _print_counts(
            asyncio.run(
                run_online(
                    async_session_factory,
                    limit=args.limit,
                    dry_run=args.dry_run,
                    concurrency=args.concurrency,
                )
            )
        )
    elif args.batch == "build":
        count = asyncio.run(batch_build(async_session_factory, args.work_dir, limit=args.limit))
        print(f"Wrote {count} request(s) to {args.work_dir / INPUT_FILE}")
    elif args.batch == "submit":
        print(f"Submitted batch {batch_submit(args.work_dir)}")
    else:
        raw = batch_download(args.work_dir)
        _print_counts(
            asyncio.run(
                batch_apply(async_session_factory, args.work_dir, raw, dry_run=args.dry_run)
            )
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
