"""Build labeling records in the exact Phase 0 shape from data/tickets/.

The script that produced data/labeling/phase1_appeals_2025-09-20_to_30.jsonl was
never committed. This module rebuilds it (reject_appeal.md, Step 3b-1). It is
verified by regenerating that file's window and comparing it field by field
(verify_phase0_extraction.py), so these rules are the ones that reproduce the
original, not guesses about it.

THE RULES (each confirmed against all 530 Phase 0 records):
  * Selection: tickets whose ``created_at`` DATE (UTC, first 10 chars) falls in
    the window, both ends inclusive. Output order: ``created_at`` ascending,
    ties kept in tickets.jsonl order.
  * ``channel`` = ``via.channel``; ``status``, ``subject``, ``created_at`` copied
    as-is from the ticket.
  * ``thread[]`` = EVERY comment event of the ticket (public and not), ordered by
    ``created_at``, ties kept in comment_events.jsonl order. ``body`` verbatim,
    NOT stripped.
  * ``sender_type``, decided in this order:
      - ``requester``      author_id == ticket.requester_id (even if that user
                           carries an agent role)
      - ``unknown``        author_id is not in users.jsonl
      - ``agent``          user role is ``agent`` or ``admin``
      - ``other_end_user`` anyone else (role ``end-user``)
  * ``sender_email_or_null`` = the user's email, or None when the author is not
    in users.jsonl (or has no email).
  * ``initial_message_body`` = the FIRST comment's body in thread order — not the
    ticket ``description``, which differs on 491 of 530 records.
  * ``marc_replies`` = Marc's PUBLIC comments ({body, created_at}), thread order;
    ``marc_reply_body`` = the first one's body, else None. Marc's user id is read
    from manifest.json. In the Phase 0 window "public only" and "all of Marc's
    comments" give the same result; public-only follows the manifest's own
    definition ("tickets where Marc authored >=1 public reply").
  * ``submission_numbers_mentioned`` = the PRODUCTION extractor's
    ``submission_numbers`` over (subject, initial_message_body), or None when it
    finds none. Reused, not copied — a copied regex would drift from the
    extractor. ⚠️ Consequence: a later change to the extractor's acceptance rules
    changes what this field regenerates to.

Pure builders take already-loaded data; ``load_archive`` is the only I/O.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

REPO_ROOT = Path(__file__).resolve().parents[2]
TICKETS_DIR = REPO_ROOT / "data" / "tickets"

AGENT_ROLES = frozenset({"agent", "admin"})

# Phase 0's label fields sit in the MIDDLE of the record, between
# submission_numbers_mentioned and thread; order is part of the shape.
PHASE0_LABEL_FIELDS: dict = {"is_reject_appeal": None, "appeal_reason": None}

FindNumbers = Callable[[str, str], list]


@dataclass
class Archive:
    """The three archive files, loaded. File order is kept for tie-breaks."""

    tickets: dict[int, dict]
    ticket_order: list[int]
    users: dict[int, dict]
    comments: dict[int, list[dict]] = field(default_factory=dict)  # file order
    marc_user_id: int | None = None


# --------------------------------------------------------------------------- I/O
def load_archive(tickets_dir: Path = TICKETS_DIR) -> Archive:
    def rows(name: str):
        with open(tickets_dir / name, encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    yield json.loads(line)

    tickets, order = {}, []
    for t in rows("tickets.jsonl"):
        tickets[t["id"]] = t
        order.append(t["id"])
    users = {u["id"]: u for u in rows("users.jsonl")}
    comments: dict[int, list[dict]] = {}
    for c in rows("comment_events.jsonl"):
        comments.setdefault(c["ticket_id"], []).append(c)
    manifest = json.loads((tickets_dir / "manifest.json").read_text(encoding="utf-8"))
    marc = (manifest.get("marc_threads") or {}).get("user_id")
    return Archive(tickets, order, users, comments, marc)


# --------------------------------------------------------------------------- pure
def select_window(archive: Archive, start: str, end: str) -> list[int]:
    """Ticket ids created on dates start..end (``YYYY-MM-DD``, inclusive)."""
    ids = [t for t in archive.ticket_order
           if start <= archive.tickets[t]["created_at"][:10] <= end]
    # sorted() is stable, so equal timestamps keep tickets.jsonl order.
    return sorted(ids, key=lambda t: archive.tickets[t]["created_at"])


def ordered_comments(comments: list[dict]) -> list[dict]:
    """By created_at; stable, so ties keep file order."""
    return sorted(comments, key=lambda c: c["created_at"])


def sender_type(comment: dict, ticket: dict, users: dict[int, dict]) -> str:
    if comment["author_id"] == ticket["requester_id"]:
        return "requester"
    user = users.get(comment["author_id"])
    if user is None:
        return "unknown"
    return "agent" if user.get("role") in AGENT_ROLES else "other_end_user"


def build_thread(ticket: dict, comments: list[dict], users: dict[int, dict]) -> list[dict]:
    thread = []
    for c in ordered_comments(comments):
        user = users.get(c["author_id"])
        thread.append({
            "sender_type": sender_type(c, ticket, users),
            "sender_email_or_null": user.get("email") if user else None,
            "body": c["body"],
            "created_at": c["created_at"],
            "is_public": c["public"],
        })
    return thread


def marc_replies(comments: list[dict], marc_user_id: int | None) -> list[dict]:
    if marc_user_id is None:
        return []
    return [{"body": c["body"], "created_at": c["created_at"]}
            for c in ordered_comments(comments)
            if c["author_id"] == marc_user_id and c["public"]]


def production_submission_numbers(subject: str, body: str) -> list:
    """The live extractor's answer (lazy import: needs backend/ on sys.path)."""
    backend = str(REPO_ROOT / "backend")
    if backend not in sys.path:
        sys.path.insert(0, backend)
    from app.pipeline.extractor import EmailExtractor

    return EmailExtractor().extract(subject, body, "", "", None).submission_numbers


def build_record(
    ticket: dict,
    comments: list[dict],
    users: dict[int, dict],
    marc_user_id: int | None,
    *,
    label_fields: dict | None = None,
    find_numbers: FindNumbers = production_submission_numbers,
) -> dict:
    """One record, keys in the exact Phase 0 order.

    ``label_fields`` go where Phase 0's did (after submission_numbers_mentioned).
    Pass ``PHASE0_LABEL_FIELDS`` to reproduce the Phase 0 file; omit it for a
    Phase 2 pool, whose label fields the labeling tool adds itself.
    """
    thread = build_thread(ticket, comments, users)
    initial = thread[0]["body"] if thread else ""
    replies = marc_replies(comments, marc_user_id)
    numbers = list(find_numbers(ticket["subject"], initial))
    record = {
        "ticket_id": ticket["id"],
        "created_at": ticket["created_at"],
        "channel": (ticket.get("via") or {}).get("channel"),
        "status": ticket["status"],
        "subject": ticket["subject"],
        "initial_message_body": initial,
        "marc_reply_body": replies[0]["body"] if replies else None,
        "submission_numbers_mentioned": numbers or None,
    }
    record.update(dict(label_fields or {}))
    record["thread"] = thread
    record["marc_replies"] = replies
    return record


def build_records(
    archive: Archive,
    ids: list[int],
    *,
    label_fields: dict | None = None,
    find_numbers: FindNumbers = production_submission_numbers,
) -> list[dict]:
    return [
        build_record(archive.tickets[t], archive.comments.get(t, []), archive.users,
                     archive.marc_user_id, label_fields=label_fields,
                     find_numbers=find_numbers)
        for t in ids
    ]
