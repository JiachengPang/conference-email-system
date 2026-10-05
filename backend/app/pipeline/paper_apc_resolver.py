"""Which APC(s) an email's paper belongs to (Zendesk chair notes, Reject Appeals queue).

The ONE place that decides how an email maps to rows of the APC assignment
sheet (``paper_assignments``). Callers use :func:`resolve_paper_apcs` (one
email) or :func:`resolve_paper_apcs_many` (a whole page); the rule itself is the
pure :func:`combine`, so changing it later happens here and nowhere else.

THE RULE (Z3a, from the Z1d check on 184 real September appeals: the number an
author writes equalled the sheet's ``paper_number`` in 17 of 17 cases where a
link could confirm it, with 0 disagreements; the sheet lists only papers that
went to review, so desk-rejected papers are almost never in it):

- Number route (primary): the email's paper numbers matched on
  ``paper_number``, normalised on both sides (trim, a leading ``#``, leading
  zeros).
- Link route (cross-check): the email's OpenReview forum ids, shape-checked,
  matched exactly on ``openreview_forum_id``.
- Combining: both routes find rows with the same chair set → ``both``; both
  find rows but the chair sets differ → ``conflict`` and NO chair is
  suggested; one route only → ``number`` or ``link``; neither → ``none``.
- Desk-reject guard: for ``desk_reject_appeal`` a chair is suggested only when
  the link route matches. A number-only match gives ``none`` with the warning
  ``desk_reject_not_in_sheet``, because such a number most likely points at a
  different, reviewed paper.

Warnings: ``conflict``, ``desk_reject_not_in_sheet``, ``short_number`` (a
matched number with fewer than 3 digits) and ``several_papers`` (more than one
distinct sheet paper behind the result). With only a handful of APCs, two
routes can name the same chair for different papers; ``several_papers``
surfaces that case. ``classifier_papers_differ`` (P4) is added afterwards by
:func:`with_classifier_papers` when the phase-1 classifier's papers differ from
the sheet papers matched through the extraction; those papers only ever raise
this warning and never choose a chair.

Number normalisation (both sides, identical in Python and SQL): strip the
whitespace in ``appeal_queue.PAPER_NUMBER_WHITESPACE`` (spaces, tabs, newlines,
non-breaking spaces...) from both ends, then a leading ``#`` and any whitespace
after it, then leading zeros. Whitespace inside a number is kept, so "12 345"
never becomes a number nobody wrote.

Both stored extraction shapes are read (see ``app.models.appeal_queue``): the
lists ``submission_numbers`` / ``openreview_forum_ids`` and the older single
values ``submission_number`` / ``openreview_forum_id``.

Chair names are personal data from the sheet: nothing here logs them.
"""

import re
from collections.abc import Hashable, Iterable
from dataclasses import dataclass, replace

from sqlalchemy.ext.asyncio import AsyncSession

from app.models.appeal_queue import PAPER_NUMBER_WHITESPACE, forum_ids, submission_numbers
from app.repositories.phase1_appeal_repository import PaperAssignmentRepository

# The shape the regex extractor accepts for a forum id. The model path passes
# its forum-id answers through unchecked, so anything else is set aside here
# rather than looked up or turned into a link.
FORUM_ID_RE = re.compile(r"^[A-Za-z0-9]{10}$")

DESK_REJECT_INTENT = "desk_reject_appeal"

SOURCE_BOTH = "both"
SOURCE_NUMBER = "number"
SOURCE_LINK = "link"
SOURCE_CONFLICT = "conflict"
SOURCE_NONE = "none"

WARN_CONFLICT = "conflict"
WARN_DESK_REJECT = "desk_reject_not_in_sheet"
WARN_SHORT_NUMBER = "short_number"
WARN_SEVERAL_PAPERS = "several_papers"
WARN_CLASSIFIER_PAPERS = "classifier_papers_differ"
# The order warnings are reported in.
WARNINGS: tuple[str, ...] = (
    WARN_CONFLICT,
    WARN_DESK_REJECT,
    WARN_SHORT_NUMBER,
    WARN_SEVERAL_PAPERS,
    WARN_CLASSIFIER_PAPERS,
)

# A matched number shorter than this is flagged: short numbers are more often
# counts or typos than paper numbers.
SHORT_NUMBER_DIGITS = 3


@dataclass(frozen=True)
class SheetRow:
    """The two sheet columns the rule needs."""

    paper_number: str
    apc_name: str


@dataclass(frozen=True)
class ApcResolution:
    """The outcome for one email.

    ``apc_names``: the suggested chairs (empty on ``conflict``, ``none`` and
    the desk-reject guard). ``source``: which route produced them.
    ``paper_numbers``: the sheet papers behind the result (both routes on a
    conflict). ``warnings``: in :data:`WARNINGS` order. ``forum_ids_matched`` /
    ``forum_ids_unmatched``: the well-formed forum ids that did / did not match
    a sheet row (used for links in the chair note).
    """

    apc_names: tuple[str, ...] = ()
    source: str = SOURCE_NONE
    paper_numbers: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    forum_ids_matched: tuple[str, ...] = ()
    forum_ids_unmatched: tuple[str, ...] = ()


def normalize_paper_number(value: str | None) -> str | None:
    """Strip edge whitespace, a leading ``#`` and the whitespace after it, then
    leading zeros. Whitespace = ``PAPER_NUMBER_WHITESPACE`` (incl. tab, newline and
    the non-breaking space).

    Mirrors, step for step, the SQL applied to ``paper_number`` in
    ``PaperAssignmentRepository.get_by_normalized_numbers``. Empty → None.
    """
    if not isinstance(value, str):
        return None
    ws = PAPER_NUMBER_WHITESPACE
    text = value.strip(ws).lstrip("#").lstrip(ws).lstrip("0")
    return text or None


def with_classifier_papers(resolution: ApcResolution, papers: Iterable[str]) -> ApcResolution:
    """Add ``classifier_papers_differ`` when the phase-1 classifier's papers differ
    from the sheet papers behind ``resolution`` (both normalised). Pure.

    Never changes the suggested chairs, the source or the papers: the classifier's
    papers only raise this warning. No classifier papers → no comparison, no
    warning; classifier papers but nothing matched through the extraction → a
    difference.
    """
    theirs = {n for n in (normalize_paper_number(p) for p in papers or ()) if n is not None}
    if not theirs:
        return resolution
    matched = {n for n in (normalize_paper_number(p) for p in resolution.paper_numbers) if n is not None}
    if theirs == matched:
        return resolution
    warnings = set(resolution.warnings) | {WARN_CLASSIFIER_PAPERS}
    return replace(resolution, warnings=tuple(w for w in WARNINGS if w in warnings))


def email_numbers(extraction) -> list[str]:
    """The email's paper numbers (either stored shape), normalised, first-seen order."""
    out: list[str] = []
    for value in submission_numbers(extraction):
        number = normalize_paper_number(value)
        if number is not None and number not in out:
            out.append(number)
    return out


def email_forum_ids(extraction) -> list[str]:
    """The email's well-formed forum ids (either stored shape), first-seen order.

    Exact and case-sensitive (forum ids are case-sensitive tokens).
    """
    return [f for f in forum_ids(extraction) if FORUM_ID_RE.fullmatch(f)]


def _distinct_names(rows: list[SheetRow]) -> list[str]:
    names: list[str] = []
    for row in rows:
        name = (row.apc_name or "").strip()
        if name and name not in names:
            names.append(name)
    return names


def _distinct_papers(rows: list[SheetRow]) -> list[str]:
    papers: list[str] = []
    for row in rows:
        if row.paper_number not in papers:
            papers.append(row.paper_number)
    return papers


def combine(
    intent: str | None,
    numbers: list[str],
    number_hits: dict[str, list[SheetRow]],
    forum_id_list: list[str],
    forum_hits: dict[str, SheetRow],
) -> ApcResolution:
    """Apply the rule to one email. Pure.

    ``numbers`` are the email's normalised numbers and ``number_hits`` maps a
    normalised number to its sheet rows; ``forum_id_list`` are the email's
    well-formed forum ids and ``forum_hits`` maps a forum id to its sheet row.
    """
    number_rows = [row for n in numbers for row in number_hits.get(n, [])]
    link_rows = [forum_hits[f] for f in forum_id_list if f in forum_hits]
    matched = tuple(f for f in forum_id_list if f in forum_hits)
    unmatched = tuple(f for f in forum_id_list if f not in forum_hits)

    number_names = _distinct_names(number_rows)
    link_names = _distinct_names(link_rows)
    warnings: set[str] = set()
    if any(len(n) < SHORT_NUMBER_DIGITS for n in numbers if number_hits.get(n)):
        warnings.add(WARN_SHORT_NUMBER)

    if number_rows and link_rows:
        papers = _distinct_papers(number_rows + link_rows)
        if set(number_names) == set(link_names):
            source, names = SOURCE_BOTH, number_names
        else:
            source, names = SOURCE_CONFLICT, []
            warnings.add(WARN_CONFLICT)
    elif number_rows:
        if intent == DESK_REJECT_INTENT:
            source, names, papers = SOURCE_NONE, [], []
            warnings.add(WARN_DESK_REJECT)
        else:
            source, names, papers = SOURCE_NUMBER, number_names, _distinct_papers(number_rows)
    elif link_rows:
        source, names, papers = SOURCE_LINK, link_names, _distinct_papers(link_rows)
    else:
        source, names, papers = SOURCE_NONE, [], []

    if len(papers) > 1:
        warnings.add(WARN_SEVERAL_PAPERS)
    return ApcResolution(
        apc_names=tuple(names),
        source=source,
        paper_numbers=tuple(papers),
        warnings=tuple(w for w in WARNINGS if w in warnings),
        forum_ids_matched=matched,
        forum_ids_unmatched=unmatched,
    )


async def resolve_paper_apcs_many(
    db: AsyncSession,
    items: dict[Hashable, tuple[str | None, dict | None]],
    *,
    assignments: PaperAssignmentRepository | None = None,
) -> dict[Hashable, ApcResolution]:
    """Resolve a whole page: ``{key: (intent, extraction)}`` → ``{key: resolution}``.

    One batched lookup per route for every email together (no query at all for
    a route nothing needs). Database errors are not caught: the caller decides.
    """
    assignments = assignments or PaperAssignmentRepository()
    parsed = {
        key: (intent, email_numbers(extraction), email_forum_ids(extraction))
        for key, (intent, extraction) in items.items()
    }
    all_numbers = list(dict.fromkeys(n for _, nums, _ in parsed.values() for n in nums))
    all_forum_ids = list(dict.fromkeys(f for _, _, fids in parsed.values() for f in fids))

    number_hits: dict[str, list[SheetRow]] = {}
    if all_numbers:
        found = await assignments.get_by_normalized_numbers(db, all_numbers)
        number_hits = {
            n: [SheetRow(r.paper_number, r.apc_name) for r in rows] for n, rows in found.items()
        }
    forum_hits: dict[str, SheetRow] = {}
    if all_forum_ids:
        found_f = await assignments.get_by_forum_ids(db, all_forum_ids)
        forum_hits = {f: SheetRow(r.paper_number, r.apc_name) for f, r in found_f.items()}

    return {
        key: combine(intent, nums, number_hits, fids, forum_hits)
        for key, (intent, nums, fids) in parsed.items()
    }


async def resolve_paper_apcs(
    db: AsyncSession,
    extraction: dict | None,
    *,
    intent: str | None = None,
    assignments: PaperAssignmentRepository | None = None,
) -> ApcResolution:
    """Resolve one email (a one-item :func:`resolve_paper_apcs_many`)."""
    result = await resolve_paper_apcs_many(
        db, {0: (intent, extraction)}, assignments=assignments
    )
    return result[0]
