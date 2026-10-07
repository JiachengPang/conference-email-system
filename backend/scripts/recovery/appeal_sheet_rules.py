"""Pure rules for importing the APC-approved Reject Appeals sheet (Part 2a).

The APCs fill a Google Sheet that we exported (13 columns, Jiacheng's eight then
five extras). Sahil exports it back to CSV by hand, and a later import script
(Part 2b) posts each approved row to its Zendesk ticket as a PUBLIC reply and
sets the ticket to solved, once per ticket. This module holds everything that
can be decided WITHOUT Zendesk and WITHOUT the database:

  * ``parse_sheet_csv``     the strict CSV parser (exact header, 13 cells per row);
  * ``decide_rows``         the row rules (which rows post, skip or are refused);
  * ``text_guards``         what the posted text may never contain;
  * ``status_refusal`` / ``ticket_changed``   the live-ticket rules;
  * ``build_manifest`` / ``load_manifest``    the export-side record (ids + hashes);
  * ``build_plan`` / ``load_plan`` / ``confirm_count``   the dry-run plan file;
  * ``find_posted_comment`` the reconcile matcher (no Zendesk tag needed).

Decisions recorded in reject_appeal.md (Part 2a): a ticked box AND a written
modified_draft is refused; wrong-paper rows post only a written modified_draft;
allowed starting statuses are new, open and pending.

PII: nothing here prints. Errors carry row numbers, column names, reason names
and ticket ids only, never cell text; the text fields of the dataclasses are
excluded from ``repr`` so a stray log line or traceback cannot show them.
"""

from __future__ import annotations

import csv
import hashlib
import io
import re
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Iterable, Mapping

# --- the sheet --------------------------------------------------------------------

COLUMNS: tuple[str, ...] = (
    "ticket_id", "paper_id", "(A)PC", "appeal_reason", "request_body", "reply_draft",
    "use_provided_draft", "modified_draft",
    "draft_type", "verify_before_sending", "openreview_link", "chair_warnings", "mode",
)

# Which export file a ticket came from (recorded in the manifest, never read
# from the CSV, so an edited cell cannot move a row between rule sets).
SHEET_APPEAL = "appeal"
SHEET_WRONG_PAPER = "wrong_paper"
SHEET_NOT_APPEAL = "not_appeal"
SHEETS: frozenset[str] = frozenset({SHEET_APPEAL, SHEET_WRONG_PAPER, SHEET_NOT_APPEAL})

# The export's formula guard prefixed these with an apostrophe.
FORMULA_START: tuple[str, ...] = ("=", "+", "-", "@", "\t", "\r")

# Live-ticket rule: the statuses a ticket may have when we post (answer 4).
ALLOWED_STATUSES: frozenset[str] = frozenset({"new", "open", "pending"})

# The flagged AI draft's first line (appeal_ai_draft.AI_FLAG_LINE, D191). Kept as
# a literal so this module stays import-light; a test pins it to the real constant.
AI_FLAG_PHRASE = "ai-written suggestion"

MIN_TEXT_CHARS = 20          # a reply shorter than this (non-space) is refused
CSV_FIELD_LIMIT = 10_000_000  # request_body can hold a long thread

# Outcomes and reasons (names only; these are what the reports count).
POST = "post"
SKIP = "skip"
REFUSE = "refuse"

R_BAD_TICKET_ID = "bad_ticket_id"
R_DUPLICATE = "duplicate_ticket_id"
R_UNKNOWN = "unknown_ticket_id"
R_NOT_APPEAL = "not_appeal_sheet"
R_BAD_CHECKBOX = "bad_checkbox"
R_REPLY_EDITED = "reply_draft_edited"
R_BOTH = "ticked_and_modified"
R_NOT_APPROVED = "not_approved"
R_WRONG_PAPER_NEEDS_MODIFIED = "wrong_paper_needs_modified_draft"
R_NOTHING_TO_SEND = "nothing_to_send"
R_READY = "ready"

G_EMPTY = "empty_text"
G_CHAIR = "chair_placeholder"
G_AI_FLAG = "ai_flag_line"
G_SENDER_NAME = "sender_name_placeholder"
G_TEMPLATE = "template_placeholder"
G_FORMULA = "formula_like_start"
G_TOO_SHORT = "too_short"
G_ENCODING = "encoding_damage"
G_CONTROL = "control_characters"

SOURCE_REPLY = "reply_draft"
SOURCE_MODIFIED = "modified_draft"

MANIFEST_SCHEMA = 1
PLAN_SCHEMA = 1


class SheetFormatError(ValueError):
    """The CSV is not the sheet we exported. The message never holds cell text."""


class ManifestError(ValueError):
    """The manifest is missing, malformed, or does not cover the sheet."""


class PlanError(ValueError):
    """The plan file is malformed."""


# --- text helpers -------------------------------------------------------------------

def normalize_newlines(text: str) -> str:
    """CRLF and lone CR become LF (Sheets and Windows tools disagree on these)."""
    return text.replace("\r\n", "\n").replace("\r", "\n")


def unguard(text: str) -> str:
    """Drop ONE leading apostrophe that our export's formula guard added."""
    if len(text) >= 2 and text[0] == "'" and text[1:].startswith(FORMULA_START):
        return text[1:]
    return text


def canonical_text(cell: str) -> str:
    """The text a cell stands for: guard removed, newlines normalised.

    Applied to every text cell before hashing, guarding and posting, so the
    manifest hash, the plan hash and the posted text are of the same string.
    """
    return normalize_newlines(unguard(cell))


def text_sha256(text: str) -> str:
    return hashlib.sha256(canonical_text(text).encode("utf-8")).hexdigest()


def fingerprint_sha256(text: str) -> str:
    """Whitespace-insensitive hash, for finding our reply among a ticket's comments.

    Zendesk turns the HTML we send back into ``plain_body`` with its own line
    breaks, so reconcile compares the words in order, not the exact layout.
    """
    return hashlib.sha256(" ".join(canonical_text(text).split()).encode("utf-8")).hexdigest()


# --- CSV parser -----------------------------------------------------------------------

@dataclass(frozen=True)
class SheetRow:
    """One data row. ``line`` is the 1-based CSV record number (header = 1)."""

    line: int
    cells: Mapping[str, str] = field(repr=False)


def parse_sheet_csv(data: bytes) -> list[SheetRow]:
    """Parse one exported sheet. Strict: exact header, 13 cells in every row.

    Accepts UTF-8 with or without a BOM and any line ending; keeps line breaks
    inside cells. Blank records (no cells at all) are ignored. Raises
    :class:`SheetFormatError` naming only a record number or a column count.
    """
    try:
        text = data.decode("utf-8-sig")
    except UnicodeDecodeError:
        text = None
    if text is None:   # raised outside the handler: the error keeps no input bytes
        raise SheetFormatError("the file is not UTF-8")

    csv.field_size_limit(max(csv.field_size_limit(), CSV_FIELD_LIMIT))
    reader = csv.reader(io.StringIO(text, newline=""), strict=True)
    records: list[list[str]] = []
    bad_record: int | None = None
    try:
        for record in reader:
            records.append(record)
    except csv.Error:
        bad_record = len(records) + 1
    if bad_record is not None:
        raise SheetFormatError(f"malformed CSV at record {bad_record}")

    if not records or tuple(records[0]) != COLUMNS:
        raise SheetFormatError("the header is not the exported 13 columns in order")
    rows: list[SheetRow] = []
    for number, record in enumerate(records[1:], start=2):
        if not record:
            continue
        if len(record) != len(COLUMNS):
            raise SheetFormatError(
                f"record {number} has {len(record)} cells, expected {len(COLUMNS)}")
        rows.append(SheetRow(number, dict(zip(COLUMNS, record))))
    return rows


def parse_ticket_id(value: str) -> int | None:
    value = value.strip()
    if not value.isdigit() or not value.isascii():
        return None
    number = int(value)
    return number if number > 0 else None


def parse_checkbox(value: str) -> bool | None:
    """Google Sheets exports a checkbox as TRUE / FALSE. Anything else is None."""
    value = value.strip().upper()
    if value == "TRUE":
        return True
    if value == "FALSE":
        return False
    return None


# --- text guards ------------------------------------------------------------------------

_CHAIR_RE = re.compile(r"\[\s*chair", re.IGNORECASE)
_SENDER_RE = re.compile(r"\[\s*sender\s*name\s*\]", re.IGNORECASE)
_TEMPLATE_RE = re.compile(r"\{\s*[A-Za-z_][A-Za-z0-9_]*\s*\}")
_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def text_guards(text: str) -> tuple[str, ...]:
    """Names of every rule the text to be posted breaks (empty tuple = clean).

    ``text`` is the cell as exported (``canonical_text`` is applied here).
    """
    body = canonical_text(text)
    if not body.strip():
        return (G_EMPTY,)
    failed: list[str] = []
    if _CHAIR_RE.search(body):
        failed.append(G_CHAIR)
    if AI_FLAG_PHRASE in body.casefold():
        failed.append(G_AI_FLAG)
    if _SENDER_RE.search(body):
        failed.append(G_SENDER_NAME)
    if _TEMPLATE_RE.search(body):
        failed.append(G_TEMPLATE)
    if body.lstrip(" ").startswith(FORMULA_START):
        failed.append(G_FORMULA)
    if len("".join(body.split())) < MIN_TEXT_CHARS:
        failed.append(G_TOO_SHORT)
    if "�" in body:
        failed.append(G_ENCODING)
    if _CONTROL_RE.search(body):
        failed.append(G_CONTROL)
    return tuple(failed)


# --- manifest -------------------------------------------------------------------------

@dataclass(frozen=True)
class ManifestEntry:
    ticket_id: int
    email_id: int
    zendesk_updated_at: str | None   # ISO UTC "...Z": the snapshot the draft was built on
    reply_draft_sha256: str          # text_sha256 of the exported reply_draft cell
    sheet: str                       # SHEET_*


def iso_utc(value: datetime | None) -> str | None:
    if value is None:
        return None
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def build_manifest(
    rows_by_sheet: Mapping[str, Iterable[SheetRow]],
    snapshot: Mapping[int, tuple[int, datetime | None]],
    source_files: Mapping[str, str],
    created_at: datetime,
) -> dict:
    """The export-side record: ids and hashes only, never text.

    ``snapshot`` maps ticket id -> (email id, stored ``zendesk_updated_at``), read
    from the same database the drafts were built from. Every exported ticket must
    appear exactly once across the sheets and exactly once in the snapshot.
    """
    entries: list[dict] = []
    seen: set[int] = set()
    bad: list[int] = []
    for sheet, rows in rows_by_sheet.items():
        if sheet not in SHEETS:
            raise ManifestError(f"unknown sheet name {sheet!r}")
        for row in rows:
            ticket = parse_ticket_id(row.cells["ticket_id"])
            if ticket is None:
                raise ManifestError(f"{sheet}: record {row.line} has no valid ticket id")
            if ticket in seen:
                raise ManifestError(f"ticket {ticket} appears more than once")
            seen.add(ticket)
            if ticket not in snapshot:
                bad.append(ticket)
                continue
            email_id, updated = snapshot[ticket]
            entries.append({
                "ticket_id": ticket,
                "email_id": int(email_id),
                "zendesk_updated_at": iso_utc(updated),
                "reply_draft_sha256": text_sha256(row.cells["reply_draft"]),
                "sheet": sheet,
            })
    if bad:
        raise ManifestError(f"tickets missing from the database snapshot: {sorted(bad)}")
    entries.sort(key=lambda e: e["ticket_id"])
    return {
        "schema_version": MANIFEST_SCHEMA,
        "created_at": iso_utc(created_at),
        "source_files": dict(sorted(source_files.items())),
        "counts": dict(sorted(Counter(e["sheet"] for e in entries).items())),
        "entries": entries,
    }


def load_manifest(data: Mapping) -> dict[int, ManifestEntry]:
    if not isinstance(data, Mapping) or data.get("schema_version") != MANIFEST_SCHEMA:
        raise ManifestError("not a schema-1 manifest")
    out: dict[int, ManifestEntry] = {}
    for item in data.get("entries") or []:
        try:
            entry = ManifestEntry(
                ticket_id=int(item["ticket_id"]),
                email_id=int(item["email_id"]),
                zendesk_updated_at=item["zendesk_updated_at"],
                reply_draft_sha256=str(item["reply_draft_sha256"]),
                sheet=str(item["sheet"]),
            )
        except (KeyError, TypeError, ValueError):
            entry = None
        if entry is None or entry.sheet not in SHEETS or len(entry.reply_draft_sha256) != 64:
            raise ManifestError("a manifest entry is malformed")
        if entry.ticket_id in out:
            raise ManifestError(f"ticket {entry.ticket_id} appears twice in the manifest")
        out[entry.ticket_id] = entry
    if not out:
        raise ManifestError("the manifest has no entries")
    return out


# --- row rules -----------------------------------------------------------------------------

@dataclass(frozen=True)
class RowDecision:
    """What to do with one sheet row. ``text`` is never in ``repr``."""

    line: int
    ticket_id: int | None
    outcome: str                       # POST / SKIP / REFUSE
    reason: str                        # R_* or a "+"-joined list of G_* guard names
    source: str | None = None          # SOURCE_* when a text was chosen
    text: str | None = field(default=None, repr=False)

    @property
    def text_sha256(self) -> str | None:
        return None if self.text is None else text_sha256(self.text)

    @property
    def fingerprint_sha256(self) -> str | None:
        return None if self.text is None else fingerprint_sha256(self.text)


def decide_row(row: SheetRow, manifest: Mapping[int, ManifestEntry]) -> RowDecision:
    """The row rules, first match wins (duplicates are handled by ``decide_rows``)."""
    cells = row.cells
    ticket = parse_ticket_id(cells["ticket_id"])
    if ticket is None:
        return RowDecision(row.line, None, REFUSE, R_BAD_TICKET_ID)
    entry = manifest.get(ticket)
    if entry is None:
        return RowDecision(row.line, ticket, REFUSE, R_UNKNOWN)
    if entry.sheet == SHEET_NOT_APPEAL:
        return RowDecision(row.line, ticket, REFUSE, R_NOT_APPEAL)
    ticked = parse_checkbox(cells["use_provided_draft"])
    if ticked is None:
        return RowDecision(row.line, ticket, REFUSE, R_BAD_CHECKBOX)
    if text_sha256(cells["reply_draft"]) != entry.reply_draft_sha256:
        return RowDecision(row.line, ticket, REFUSE, R_REPLY_EDITED)
    modified = canonical_text(cells["modified_draft"])
    has_modified = bool(modified.strip())
    if ticked and has_modified:
        return RowDecision(row.line, ticket, REFUSE, R_BOTH)
    if not ticked and not has_modified:
        return RowDecision(row.line, ticket, SKIP, R_NOT_APPROVED)
    if ticked:
        if entry.sheet == SHEET_WRONG_PAPER:
            return RowDecision(row.line, ticket, REFUSE, R_WRONG_PAPER_NEEDS_MODIFIED)
        text, source = canonical_text(cells["reply_draft"]), SOURCE_REPLY
        if not text.strip():
            return RowDecision(row.line, ticket, REFUSE, R_NOTHING_TO_SEND)
    else:
        text, source = modified, SOURCE_MODIFIED
    failed = text_guards(text)
    if failed:
        return RowDecision(row.line, ticket, REFUSE, "+".join(failed), source)
    return RowDecision(row.line, ticket, POST, R_READY, source, text)


def decide_rows(
    rows: Iterable[SheetRow], manifest: Mapping[int, ManifestEntry]
) -> list[RowDecision]:
    """Every row's decision. A ticket id on more than one row refuses ALL of them."""
    rows = list(rows)
    ids = Counter(parse_ticket_id(r.cells["ticket_id"]) for r in rows)
    out: list[RowDecision] = []
    for row in rows:
        ticket = parse_ticket_id(row.cells["ticket_id"])
        if ticket is not None and ids[ticket] > 1:
            out.append(RowDecision(row.line, ticket, REFUSE, R_DUPLICATE))
        else:
            out.append(decide_row(row, manifest))
    return out


# --- live-ticket rules (the values come from Zendesk in Part 2b) ---------------------------

def status_refusal(status: str | None) -> str | None:
    """None when a ticket in this status may be answered, else a reason name."""
    normalized = (status or "").strip().lower()
    if normalized in ALLOWED_STATUSES:
        return None
    return f"status_{normalized or 'unknown'}"


def _parse_iso(value: str | None) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def ticket_changed(live_updated_at: str | None, snapshot_updated_at: str | None) -> bool:
    """True unless both timestamps are known and equal. Unknown counts as changed."""
    live, snap = _parse_iso(live_updated_at), _parse_iso(snapshot_updated_at)
    return live is None or snap is None or live != snap


def find_posted_comment(
    comments: Iterable[Mapping], fingerprint: str, *, author_id: int | None = None
) -> int | None:
    """Reconcile without a tag: the id of a PUBLIC comment whose words match ours.

    Compares ``fingerprint_sha256`` of the comment's ``plain_body`` (else
    ``body``) with the plan's fingerprint. With ``author_id`` only that author's
    comments count. Returns the first match's id, or None.
    """
    for comment in comments:
        if comment.get("public") is not True:
            continue
        if author_id is not None and comment.get("author_id") != author_id:
            continue
        body = comment.get("plain_body") or comment.get("body") or ""
        if body and fingerprint_sha256(body) == fingerprint:
            return comment.get("id")
    return None


# --- plan file -------------------------------------------------------------------------------

def build_plan(
    decisions: Iterable[RowDecision],
    manifest: Mapping[int, ManifestEntry],
    *,
    manifest_sha256: str,
    sheet_sha256: Mapping[str, str],
    created_at: datetime,
) -> dict:
    """The dry run's record: ids, names and hashes only, never text."""
    decisions = list(decisions)
    rows = []
    for d in decisions:
        entry = manifest.get(d.ticket_id) if d.ticket_id is not None else None
        rows.append({
            "line": d.line,
            "ticket_id": d.ticket_id,
            "email_id": entry.email_id if entry else None,
            "outcome": d.outcome,
            "reason": d.reason,
            "source": d.source,
            "text_sha256": d.text_sha256,
            "fingerprint_sha256": d.fingerprint_sha256,
            "snapshot_updated_at": entry.zendesk_updated_at if entry else None,
        })
    return {
        "schema_version": PLAN_SCHEMA,
        "created_at": iso_utc(created_at),
        "manifest_sha256": manifest_sha256,
        "sheet_sha256": dict(sorted(sheet_sha256.items())),
        "counts": {
            "outcome": dict(sorted(Counter(d.outcome for d in decisions).items())),
            "reason": dict(sorted(Counter(d.reason for d in decisions).items())),
        },
        "to_post": sum(1 for d in decisions if d.outcome == POST),
        "rows": rows,
    }


_PLAN_ROW_KEYS = frozenset({"line", "ticket_id", "email_id", "outcome", "reason", "source",
                            "text_sha256", "fingerprint_sha256", "snapshot_updated_at"})


def load_plan(data: Mapping) -> dict:
    """Validate a plan file read back from disk (shape only; no text is allowed in it)."""
    if not isinstance(data, Mapping) or data.get("schema_version") != PLAN_SCHEMA:
        raise PlanError("not a schema-1 plan")
    rows = data.get("rows")
    if not isinstance(rows, list):
        raise PlanError("the plan has no rows")
    for row in rows:
        if not isinstance(row, Mapping) or set(row) != _PLAN_ROW_KEYS:
            raise PlanError("a plan row has unexpected keys")
        if row["outcome"] == POST and (
            not isinstance(row["ticket_id"], int)
            or not isinstance(row["text_sha256"], str)
            or len(row["text_sha256"]) != 64
        ):
            raise PlanError("a post row lacks its ticket id or text hash")
    posts = [r["ticket_id"] for r in rows if r["outcome"] == POST]
    if len(posts) != len(set(posts)):
        raise PlanError("a ticket is planned to post twice")
    if data.get("to_post") != len(posts):
        raise PlanError("to_post does not match the post rows")
    return dict(data)


def decision_matches_plan(decision: RowDecision, plan_row: Mapping) -> bool:
    """At execution time: the row still decides POST with exactly the planned text."""
    return (
        decision.outcome == POST
        and plan_row.get("outcome") == POST
        and decision.ticket_id == plan_row.get("ticket_id")
        and decision.text_sha256 == plan_row.get("text_sha256")
    )


def confirm_count(typed: str, expected: int) -> bool:
    """The typed confirmation: exactly the number of replies about to be posted."""
    return expected > 0 and typed.strip() == str(expected)
