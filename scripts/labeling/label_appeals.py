#!/usr/bin/env python
"""Hand-labeling CLI for the Phase 1 reject-appeal set.

(``--phase2`` switches to the Phase 2 reason-labeling mode in label_phase2.py;
without it, everything below is unchanged.)

Walks data/labeling/phase1_appeals_2025-09-20_to_30.jsonl one ticket at a
time and records a human judgment into the two label fields that extraction
left null (``is_reject_appeal`` / ``appeal_reason``). Stdlib only.

This tool is DELIBERATELY UNASSISTED. It shows the raw ticket text and the
taxonomy, and nothing else: no keyword highlighting, no scoring, no
suggested answer, no ordering by "likely appeal". Anything that nudged the
labeler would leak into the labels and quietly bias the very ground truth
this set exists to provide.

Order is a fixed shuffle (SEED), computed over ALL ticket ids rather than
over the unlabeled ones, so the sequence is identical on every run and
stopping/restarting resumes the same walk instead of reshuffling. Already
labeled tickets are skipped, which is what makes the tool resumable.

Three states, not two. A ticket is LABELED (``is_reject_appeal`` non-null),
DEFERRED (``deferred`` true, labels still null -- parked by [s] as too
unclear to call now), or UNTOUCHED. The default run shows only untouched
tickets, so a deferred one stops resurfacing every pass; ``--review-deferred``
shows only the deferred ones, for a dedicated revisit. Labeling a ticket in
either mode clears its deferred flag, so the three states stay disjoint and
always sum to the file's row count.

``deferred`` is migrated in LAZILY. It is defaulted in memory at load, but
that alone never triggers a write: a run that changes nothing leaves the file
untouched on disk. The field reaches disk with the first real change, which
rewrites every row anyway.

Durability: after every single label the WHOLE file is rewritten atomically
(temp file in the same directory -> fsync -> os.replace). A crash, a Ctrl+C
or a power cut can therefore lose at most the answer being typed, never the
file: the original is replaced only once the replacement is complete on
disk. SIGINT is held for the duration of that rename and re-raised after,
so Ctrl+C lands between writes and never during one.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import signal
import sys
import tempfile
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = REPO_ROOT / "data" / "labeling" / "phase1_appeals_2025-09-20_to_30.jsonl"

# Fixed so the walk order is stable across runs. Changing it reshuffles the
# remaining queue, which is harmless but breaks "resume where I left off".
SEED = 42

REASONS = {
    "a": "wrong-paper review",
    "b": "score/outcome mismatch",
    "c": "reviewer misunderstood",
    "d": '"LLM-generated review" claim',
    "e": "general dissatisfaction",
    "r": "reciprocal-review duty dispute",
    "o": "other/unclear",
}
NOT_APPEAL = "n"
ACTIONS = ("t", "s", "q")
# Appended to each record rather than slotted next to the other two label
# fields: inserting it mid-record would shift `thread` and `marc_replies`, and
# every untouched row's existing field order must survive a rewrite untouched.
DEFERRED_KEY = "deferred"
RULE = "=" * 78
THIN = "-" * 78


class _DeferInterrupt:
    """Hold SIGINT for a critical section, re-raising it once the block ends.

    Without this a Ctrl+C landing between the temp file's fsync and the
    os.replace would abort the rename and orphan the temp file. The data file
    would survive (it is only ever replaced whole), but the just-entered label
    would be silently lost even though the user saw it accepted.
    """

    def __enter__(self) -> "_DeferInterrupt":
        self._caught = False
        try:
            self._previous = signal.signal(signal.SIGINT, self._handle)
        except ValueError:  # not the main thread; nothing to defer
            self._previous = None
        return self

    def _handle(self, signum, frame) -> None:  # noqa: ANN001 - signal signature
        self._caught = True
        print("\n(finishing save before exiting...)", flush=True)

    def __exit__(self, exc_type, exc, tb) -> bool:
        if self._previous is not None:
            signal.signal(signal.SIGINT, self._previous)
        if self._caught and exc_type is None:
            raise KeyboardInterrupt
        return False


def load(path: Path) -> tuple[list[dict], int]:
    """Read the file and default `deferred` in memory only.

    Returns the records and how many needed the default. That count is
    reported but deliberately does NOT mark the session dirty: migrating a
    field is not a reason to rewrite 530 rows. The default reaches disk with
    the first real label or defer, which rewrites the whole file regardless.
    """
    with open(path, encoding="utf-8") as fh:
        records = [json.loads(line) for line in fh if line.strip()]
    migrated = 0
    for record in records:
        if DEFERRED_KEY not in record:
            record[DEFERRED_KEY] = False
            migrated += 1
    return records, migrated


def save(records: list[dict], path: Path) -> None:
    """Rewrite the file atomically: temp -> fsync -> replace.

    The temp file is created in the SAME directory so os.replace is a true
    atomic rename rather than a cross-filesystem copy. On any failure the temp
    file is removed and the original is left exactly as it was.
    """
    fd, tmp = tempfile.mkstemp(
        dir=str(path.parent), prefix=path.name + ".", suffix=".tmp"
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as fh:
            for record in records:
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def walk_order(records: list[dict]) -> list[int]:
    """The fixed walk order: a seeded shuffle over ALL ticket ids present.

    The single definition of the ordering, so anything that needs to agree
    with the labeler's sequence (the batch splitter, for one) reuses this
    rather than re-deriving it and silently drifting. Sorting first makes the
    result independent of the order the rows happen to sit in the file.

    NOTE the order is a function of the ids PRESENT in `records`. Over a batch
    file it reproduces that batch's internal order, not the batch's positions
    within the full set -- which is what a labeler working a batch wants.
    """
    order = sorted(r["ticket_id"] for r in records)
    random.Random(SEED).shuffle(order)
    return order


def progress(records: list[dict]) -> tuple[int, int, int, Counter]:
    """(labeled, deferred, untouched, reason breakdown).

    The three buckets are disjoint by construction -- `deferred` counts only
    rows that are ALSO unlabeled -- so they always sum to len(records) and the
    summary can never double-count a ticket that was deferred and later
    labeled.
    """
    done = [r for r in records if r.get("is_reject_appeal") is not None]
    deferred = sum(
        1
        for r in records
        if r.get("is_reject_appeal") is None and r.get(DEFERRED_KEY)
    )
    reasons = Counter(
        r.get("appeal_reason") if r.get("is_reject_appeal") else NOT_APPEAL
        for r in done
    )
    return len(done), deferred, len(records) - len(done) - deferred, reasons


def print_progress(records: list[dict]) -> None:
    done, deferred, untouched, reasons = progress(records)
    total = len(records)
    print("\nLabeled   {} / {}".format(done, total))
    for key in (NOT_APPEAL,) + tuple(REASONS):
        if reasons.get(key):
            label = "not a reject appeal" if key == NOT_APPEAL else REASONS[key]
            print("    [{}] {:<28} {}".format(key, label, reasons[key]))
    if not reasons:
        print("    (no labels yet)")
    print("Deferred  {} / {}   (unclear; --review-deferred to revisit)".format(
        deferred, total))
    print("Untouched {} / {}".format(untouched, total))


def show_ticket(record: dict, position: int, remaining: int, records: list[dict]) -> None:
    done, _, _, _ = progress(records)
    print("\n" + RULE)
    print(
        "ticket {}   {}   [{}/{} this run | {}/{} labeled]".format(
            record.get("ticket_id"),
            record.get("created_at"),
            position,
            remaining,
            done,
            len(records),
        )
    )
    print("SUBJECT: {}".format(record.get("subject")))
    print(THIN)
    print("INITIAL MESSAGE:")
    print(record.get("initial_message_body") or "(empty)")
    print(THIN)
    print("MARC REPLY:")
    print(record.get("marc_reply_body") or "(no reply)")
    print(RULE)


def show_thread(record: dict) -> None:
    thread = record.get("thread") or []
    print("\n" + THIN)
    print("FULL THREAD ({} messages)".format(len(thread)))
    for index, entry in enumerate(thread, start=1):
        print(THIN)
        visibility = "" if entry.get("is_public") else "  [INTERNAL]"
        print(
            "[{}] {}   {}{}".format(
                index, entry.get("sender_type"), entry.get("created_at"), visibility
            )
        )
        print(entry.get("body") or "(empty)")
    print(THIN)


def print_menu() -> None:
    print("  [n] not a reject appeal")
    print("  [a] wrong-paper review        [b] score/outcome mismatch")
    print('  [c] reviewer misunderstood    [d] "LLM-generated review" claim')
    print("  [e] general dissatisfaction   [r] reciprocal-review duty dispute")
    print("  [o] other/unclear")
    print("  [t] show full thread first, then ask again")
    print("  [s] defer - unclear, come back to this later")
    print("  [q] save and quit")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Hand-label the Phase 1 reject-appeal set (manual, unassisted).",
    )
    parser.add_argument(
        "path",
        nargs="?",
        default=None,
        type=Path,
        help="JSONL file to label. Defaults to the full Phase 1 set; point it "
             "at a batch file (data/labeling/batch_*.jsonl) to work a slice "
             "independently of anyone else.",
    )
    parser.add_argument(
        "--review-deferred",
        action="store_true",
        help="Revisit pass: show ONLY tickets deferred with [s] that are still "
             "unlabeled. Labeling one clears its deferred flag.",
    )
    parser.add_argument(
        "--phase2",
        action="store_true",
        help="Phase 2 reason labeling (label_phase2.py): multi-label reasons, "
             "a separate reciprocal box, author-only view. Defaults to "
             "data/labeling/phase2_reasons_pool.jsonl.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    # Ticket bodies are real email text and carry non-ASCII; the default
    # Windows console codec would raise UnicodeEncodeError mid-ticket.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass

    if args.phase2:
        # Lazy: label_phase2 imports from this module, and the Phase 0 path
        # below must not depend on it.
        from label_phase2 import main_phase2

        return main_phase2(args.path, args.review_deferred)

    path = args.path or DATA_PATH
    if not path.exists():
        print("ERROR: data file not found: {}".format(path), file=sys.stderr)
        return 2

    records, migrated = load(path)
    # A Phase 2 file labeled here would get single-code Phase 0 fields written
    # beside its multi-label ones. Refuse rather than mix the two shapes.
    if any("appeal_reason_codes" in r for r in records):
        print("ERROR: {} is a Phase 2 file; run it with --phase2.".format(path.name),
              file=sys.stderr)
        return 2
    by_id = {r["ticket_id"]: r for r in records}

    # Nothing is written until a real label or defer happens. A migrated-only
    # session must leave the file byte-identical on disk.
    dirty = False

    def persist() -> None:
        with _DeferInterrupt():
            save(records, path)

    # Shuffle ALL ids, then filter, so the walk order is identical every run
    # regardless of how much is already labeled or deferred.
    order = walk_order(records)
    unlabeled = [tid for tid in order if by_id[tid].get("is_reject_appeal") is None]
    if args.review_deferred:
        queue = [tid for tid in unlabeled if by_id[tid].get(DEFERRED_KEY)]
    else:
        queue = [tid for tid in unlabeled if not by_id[tid].get(DEFERRED_KEY)]

    print("Loaded {} tickets from {}".format(len(records), path.name))
    if migrated:
        print("({} rows defaulted to deferred=false in memory; "
              "written on the next real change)".format(migrated))
    if args.review_deferred:
        print("MODE: --review-deferred (deferred + still unlabeled only)")
    print_progress(records)
    if not queue:
        print(
            "\nNothing deferred to review."
            if args.review_deferred
            else "\nNothing left to label."
        )
        return 0
    print("\n{} to go. Ctrl+C or [q] saves and exits safely.".format(len(queue)))

    try:
        for position, ticket_id in enumerate(queue, start=1):
            record = by_id[ticket_id]
            show_ticket(record, position, len(queue), records)
            while True:
                print_menu()
                try:
                    choice = input("> ").strip().lower()
                except EOFError:
                    print("\n(end of input)")
                    raise KeyboardInterrupt from None

                if choice == "t":
                    show_thread(record)
                    continue
                if choice == "s":
                    record[DEFERRED_KEY] = True
                    dirty = True
                    persist()
                    print("deferred - labels left null "
                          "(revisit with --review-deferred)  (saved)")
                    break
                if choice == "q":
                    if dirty:
                        persist()
                        print("\nSaved.")
                    else:
                        print("\nNothing changed - file left untouched.")
                    print_progress(records)
                    return 0
                if choice == NOT_APPEAL or choice in REASONS:
                    record["is_reject_appeal"] = choice != NOT_APPEAL
                    record["appeal_reason"] = None if choice == NOT_APPEAL else choice
                    # A decision supersedes a defer, so the three states stay
                    # disjoint whichever mode this was labeled from.
                    record[DEFERRED_KEY] = False
                    dirty = True
                    persist()
                    shown = (
                        "not a reject appeal"
                        if choice == NOT_APPEAL
                        else REASONS[choice]
                    )
                    print("recorded: {}  (saved)".format(shown))
                    break

                valid = ", ".join((NOT_APPEAL,) + tuple(REASONS) + ACTIONS)
                print("'{}' is not one of: {}".format(choice, valid))
    except KeyboardInterrupt:
        if dirty:
            persist()
            print("\n\nInterrupted - saved.")
        else:
            print("\n\nInterrupted - nothing changed, file left untouched.")
        print_progress(records)
        return 0

    print("\nEnd of queue.")
    if dirty:
        persist()
    print_progress(records)
    return 0


if __name__ == "__main__":
    sys.exit(main())
