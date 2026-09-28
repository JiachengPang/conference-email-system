"""Phase 2 labeling mode for label_appeals.py (``--phase2``). Stdlib only.

Labels the Phase 2 reason pool (reject_appeal.md D63): multi-label appeal
reasons, a separate reciprocal checkbox, and an AUTHOR-ONLY view.

Kept in its own module on purpose: ``label_appeals.py`` gains only a flag and a
dispatch, so the Phase 0 path and its file format stay exactly as they were.
Shared machinery (atomic save, SIGINT deferral, the fixed walk order, the reason
display names) is imported from there, not copied.

WHAT THE LABELER SEES — and what it deliberately does not:
  * Author-side messages only (``requester`` and ``other_end_user``). Agent
    replies are hidden and only counted, so a label can never rest on
    information that exists only in the chair's reply (D63, fixing D22 for
    Phase 2). Messages whose sender type is ``unknown`` are hidden too and
    counted separately: we cannot tell which side wrote them.
  * PUBLIC messages only, matching production exactly. The model never sees a
    non-public comment: the follow-up transcript keeps only public turns
    (backend/app/pipeline/thread_transcript.py, ``build_transcript``), and
    first ingest classifies the first PUBLIC requester message
    (backend/app/integrations/zendesk/adapter.py, the initial-inquiry rule).
    Non-public author messages are therefore hidden and counted like agent
    replies. No visibility marker is ever shown on a message — every shown
    message is public — so none can bias the labeler.
  * Nothing about WHY a ticket is in the pool. Bucket membership lives in a
    separate sidecar file and is never read here, so it cannot nudge a label.

STORED FIELDS PER ROW (Phase 0 rows carry ``appeal_reason`` instead, and the two
formats are refused in each other's mode):
  * ``is_reject_appeal``      bool | null   — null = unlabeled
  * ``appeal_reason_codes``   list | null   — label codes in registry order;
                                              [] = no listed reason (incl. n)
  * ``is_reciprocal_label``   bool | null   — the ``r`` checkbox (D60)
  * ``deferred``              bool

The codes are LABEL codes, used only for scoring. The wire names live in
backend/app/pipeline/appeal_reasons.py, and a CI drift test keeps the two
aligned.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable

from label_appeals import (
    DEFERRED_KEY,
    NOT_APPEAL,
    REASONS,
    REPO_ROOT,
    RULE,
    THIN,
    _DeferInterrupt,
    save,
    walk_order,
)

PHASE2_PATH = REPO_ROOT / "data" / "labeling" / "phase2_reasons_pool.jsonl"

RECIPROCAL_KEY = "r"
# Registry order (a, b, c, d, e, o): REASONS minus the reciprocal code, which is
# a checkbox here, not a reason (D60).
REASON_CODES: tuple[str, ...] = tuple(k for k in REASONS if k != RECIPROCAL_KEY)

APPEAL_KEY = "is_reject_appeal"
CODES_KEY = "appeal_reason_codes"
RECIPROCAL_LABEL_KEY = "is_reciprocal_label"
PHASE2_FIELDS = (APPEAL_KEY, CODES_KEY, RECIPROCAL_LABEL_KEY, DEFERRED_KEY)
PHASE0_ONLY_KEY = "appeal_reason"

AUTHOR_SENDER_TYPES = frozenset({"requester", "other_end_user"})
AGENT_SENDER_TYPE = "agent"

SAVE_KEY = ""  # a bare Enter
DEFER_KEY = "s"
QUIT_KEY = "q"


class WrongFileFormat(ValueError):
    """A Phase 0 file handed to --phase2 (or the reverse)."""


# --------------------------------------------------------------------------- data
def load_phase2(path: Path) -> tuple[list[dict], int]:
    """Read the pool, defaulting the four label fields IN MEMORY only.

    Same lazy discipline as Phase 0's `deferred`: defaulting is not a reason
    to rewrite the file, so it never marks the session dirty. Refuses a Phase 0
    file outright; mixing `appeal_reason` and `appeal_reason_codes` rows in one
    file would leave two label shapes that nothing scores consistently.
    """
    import json

    with open(path, encoding="utf-8") as fh:
        records = [json.loads(line) for line in fh if line.strip()]
    if any(PHASE0_ONLY_KEY in r for r in records):
        raise WrongFileFormat(
            f"{path.name} is a Phase 0 file (rows carry '{PHASE0_ONLY_KEY}'); "
            "run it without --phase2."
        )
    migrated = 0
    for record in records:
        missing = False
        for key in PHASE2_FIELDS:
            if key not in record:
                record[key] = False if key == DEFERRED_KEY else None
                missing = True
        migrated += missing
    return records, migrated


def author_view(record: dict) -> tuple[list[dict], int, int, int]:
    """(shown, agent hidden, non-public author hidden, unknown hidden).

    Shown = PUBLIC author-side messages, in thread order. `is_public` is
    matched with `is True`: a missing or non-boolean flag is hidden rather
    than guessed public, since the model would not see it either.
    """
    shown, agents, internal, unknown = [], 0, 0, 0
    for entry in record.get("thread") or []:
        sender = entry.get("sender_type")
        if sender == AGENT_SENDER_TYPE:
            agents += 1
        elif sender not in AUTHOR_SENDER_TYPES:
            unknown += 1
        elif entry.get("is_public") is not True:
            internal += 1
        else:
            shown.append(entry)
    return shown, agents, internal, unknown


class Selection:
    """The in-progress label for one ticket, before Enter saves it."""

    def __init__(self) -> None:
        self.not_appeal = False
        self.codes: set[str] = set()
        self.reciprocal = False

    def toggle(self, key: str) -> None:
        # Any reason or the reciprocal box means "this IS an appeal".
        self.not_appeal = False
        if key == RECIPROCAL_KEY:
            self.reciprocal = not self.reciprocal
        else:
            self.codes ^= {key}

    def mark_not_appeal(self) -> None:
        self.not_appeal = True
        self.codes.clear()
        self.reciprocal = False

    @property
    def is_empty_appeal(self) -> bool:
        """An appeal with no listed reason and no reciprocal box — the [] case."""
        return not self.not_appeal and not self.codes and not self.reciprocal

    def ordered_codes(self) -> list[str]:
        return [c for c in REASON_CODES if c in self.codes]

    def describe(self) -> str:
        if self.not_appeal:
            return "NOT a reject appeal"
        reasons = ", ".join(
            "[{}] {}".format(c, REASONS[c]) for c in self.ordered_codes()
        ) or "(none)"
        return "appeal - reasons: {}   reciprocal: {}".format(
            reasons, "YES" if self.reciprocal else "no"
        )


def apply_selection(record: dict, selection: Selection) -> None:
    """Write a finished selection onto the record. Clears any defer."""
    if selection.not_appeal:
        record[APPEAL_KEY] = False
        record[CODES_KEY] = []  # D67: n scores as []
        record[RECIPROCAL_LABEL_KEY] = False
    else:
        record[APPEAL_KEY] = True
        record[CODES_KEY] = selection.ordered_codes()
        record[RECIPROCAL_LABEL_KEY] = selection.reciprocal
    record[DEFERRED_KEY] = False


def progress_phase2(records: list[dict]) -> dict:
    """Disjoint labeled / deferred / untouched, plus a breakdown of the labeled."""
    labeled = [r for r in records if r.get(APPEAL_KEY) is not None]
    deferred = sum(
        1 for r in records if r.get(APPEAL_KEY) is None and r.get(DEFERRED_KEY)
    )
    appeals = [r for r in labeled if r[APPEAL_KEY]]
    per_code = {c: sum(1 for r in appeals if c in (r.get(CODES_KEY) or [])) for c in REASON_CODES}
    return {
        "labeled": len(labeled),
        "deferred": deferred,
        "untouched": len(records) - len(labeled) - deferred,
        "not_appeal": len(labeled) - len(appeals),
        "appeals": len(appeals),
        "reciprocal": sum(1 for r in appeals if r.get(RECIPROCAL_LABEL_KEY)),
        "no_listed_reason": sum(
            1 for r in appeals
            if not r.get(CODES_KEY) and not r.get(RECIPROCAL_LABEL_KEY)
        ),
        "per_code": per_code,
    }


# --------------------------------------------------------------------------- display
def print_progress_phase2(records: list[dict], out: Callable[[str], None]) -> None:
    p = progress_phase2(records)
    total = len(records)
    out("\nLabeled   {} / {}".format(p["labeled"], total))
    out("    [n] not a reject appeal        {}".format(p["not_appeal"]))
    out("    appeals                        {}".format(p["appeals"]))
    for code in REASON_CODES:
        if p["per_code"][code]:
            out("      [{}] {:<28} {}".format(code, REASONS[code], p["per_code"][code]))
    out("      [r] reciprocal box ticked    {}".format(p["reciprocal"]))
    out("      no listed reason (the [] case) {}".format(p["no_listed_reason"]))
    out("Deferred  {} / {}".format(p["deferred"], total))
    out("Untouched {} / {}".format(p["untouched"], total))


def show_ticket_phase2(record: dict, position: int, remaining: int,
                       out: Callable[[str], None]) -> None:
    """Ticket id, date, subject, and the author-side messages. Nothing else.

    Deliberately omitted: `marc_reply_body`, `marc_replies`, `status`, and any
    pool/bucket information.
    """
    shown, agents, internal, unknown = author_view(record)
    out("\n" + RULE)
    out("ticket {}   {}   [{}/{} this run]".format(
        record.get("ticket_id"), record.get("created_at"), position, remaining))
    out("SUBJECT: {}".format(record.get("subject")))
    for index, entry in enumerate(shown, start=1):
        out(THIN)
        # No visibility marker: everything shown is public (see module doc).
        out("[{}] {}   {}".format(index, entry.get("sender_type"), entry.get("created_at")))
        out(entry.get("body") or "(empty)")
    if not shown:
        out(THIN)
        out("(no author-side messages in this thread)")
    out(THIN)
    hidden = "{} agent replies hidden".format(agents)
    if internal:
        hidden += "; {} non-public author messages hidden".format(internal)
    if unknown:
        hidden += "; {} messages of unknown sender hidden".format(unknown)
    out(hidden)
    out(RULE)


def print_menu_phase2(selection: Selection, out: Callable[[str], None]) -> None:
    out("CURRENT: " + selection.describe())
    out("  toggle a reason:")
    out("  [a] {:<28} [b] {}".format(REASONS["a"], REASONS["b"]))
    out("  [c] {:<28} [d] {}".format(REASONS["c"], REASONS["d"]))
    out("  [e] {:<28} [o] {}".format(REASONS["e"], REASONS["o"]))
    out("  [r] toggle the RECIPROCAL box (separate from reasons)")
    out("  [n] not a reject appeal (clears everything)")
    out("  [Enter] save    [s] defer    [q] save and quit")


# --------------------------------------------------------------------------- loop
def run_phase2(
    records: list[dict],
    persist: Callable[[], None],
    *,
    review_deferred: bool = False,
    input_fn: Callable[[str], str] | None = None,
    out: Callable[[str], None] = print,
) -> int:
    """The labeling loop. Injected I/O so it can be driven by tests.

    `persist` is called after every change (rewrites the whole file
    atomically), and once more on quit/interrupt if anything changed —
    mirroring Phase 0.
    """
    # Resolved at CALL time, not as a default argument: a default would bind
    # whatever `input` was when the module was imported.
    if input_fn is None:
        input_fn = input
    by_id = {r["ticket_id"]: r for r in records}
    order = walk_order(records)
    unlabeled = [t for t in order if by_id[t].get(APPEAL_KEY) is None]
    queue = [
        t for t in unlabeled
        if bool(by_id[t].get(DEFERRED_KEY)) == review_deferred
    ]
    dirty = False

    def finish(message: str) -> int:
        if dirty:
            persist()
        out(message)
        print_progress_phase2(records, out)
        return 0

    if review_deferred:
        out("MODE: --phase2 --review-deferred (deferred + still unlabeled only)")
    print_progress_phase2(records, out)
    if not queue:
        out("\nNothing deferred to review." if review_deferred else "\nNothing left to label.")
        return 0
    out("\n{} to go. Ctrl+C or [q] saves and exits safely.".format(len(queue)))

    try:
        for position, ticket_id in enumerate(queue, start=1):
            record = by_id[ticket_id]
            show_ticket_phase2(record, position, len(queue), out)
            selection = Selection()
            while True:
                print_menu_phase2(selection, out)
                try:
                    raw = input_fn("> ")
                except EOFError:
                    out("\n(end of input)")
                    raise KeyboardInterrupt from None
                choice = raw.strip().lower()

                if choice in REASON_CODES or choice == RECIPROCAL_KEY:
                    selection.toggle(choice)
                    continue
                if choice == NOT_APPEAL:
                    selection.mark_not_appeal()
                    continue
                if choice == DEFER_KEY:
                    record[DEFERRED_KEY] = True
                    dirty = True
                    persist()
                    out("deferred - labels left null (revisit with --review-deferred)  (saved)")
                    break
                if choice == QUIT_KEY:
                    return finish("\nSaved." if dirty else "\nNothing changed - file left untouched.")
                if choice == SAVE_KEY:
                    if selection.is_empty_appeal:
                        answer = input_fn(
                            "Save as an appeal with NO listed reason and NOT reciprocal? [y/N] "
                        ).strip().lower()
                        if answer != "y":
                            out("not saved - keep choosing")
                            continue
                    apply_selection(record, selection)
                    dirty = True
                    persist()
                    out("recorded: {}  (saved)".format(selection.describe()))
                    break

                valid = ", ".join(REASON_CODES + (RECIPROCAL_KEY, NOT_APPEAL, DEFER_KEY, QUIT_KEY))
                out("'{}' is not one of: {} (or Enter to save)".format(choice, valid))
    except KeyboardInterrupt:
        return finish("\n\nInterrupted - saved." if dirty else
                      "\n\nInterrupted - nothing changed, file left untouched.")

    return finish("\nEnd of queue.")


def main_phase2(path: Path | None, review_deferred: bool) -> int:
    """Entry point from label_appeals.main when --phase2 is given."""
    path = path or PHASE2_PATH
    if not path.exists():
        print("ERROR: data file not found: {}".format(path), file=sys.stderr)
        return 2
    try:
        records, migrated = load_phase2(path)
    except WrongFileFormat as exc:
        print("ERROR: {}".format(exc), file=sys.stderr)
        return 2
    print("Loaded {} tickets from {} (Phase 2, author-only view)".format(len(records), path.name))
    if migrated:
        print("({} rows defaulted in memory; written on the next real change)".format(migrated))

    def persist() -> None:
        with _DeferInterrupt():
            save(records, path)

    return run_phase2(records, persist, review_deferred=review_deferred)
