"""Which APC(s) an email's paper belongs to, for the Zendesk chair note (Z2).

The ONE place that decides how an email maps to a row of the APC assignment
sheet (``paper_assignments``). Callers use :func:`resolve_paper_apcs`; changing
the lookup rule later (for example adding a paper-number route) happens here
and nowhere else.

The rule today is the FORUM-ID ROUTE ONLY: the OpenReview forum ids the
extractor found in the email (``extraction["openreview_forum_ids"]``), checked
for shape, then matched exactly on ``paper_assignments.openreview_forum_id``.

Paper numbers are deliberately NOT used. Whether the number an author writes
equals the sheet's ``paper_number`` is still being checked on real emails, and
a number that happens to equal an unrelated sheet row would name the wrong
APC with nothing to show it. This is also why this module does not reuse
``phase1_appeal_outcome._resolve_papers``, which tries numbers first.
"""

import re
from dataclasses import dataclass

from sqlalchemy.ext.asyncio import AsyncSession

from app.repositories.phase1_appeal_repository import PaperAssignmentRepository

# The shape the regex extractor accepts for a forum id. The model path passes
# its forum-id answers through unchecked, so anything else is set aside here
# rather than looked up or turned into a link.
FORUM_ID_RE = re.compile(r"^[A-Za-z0-9]{10}$")

ROUTE_FORUM_ID = "forum_id"


@dataclass(frozen=True)
class ApcResolution:
    """The outcome of one lookup.

    ``apc_names``: the APCs found, in the order their forum ids appear in the
    email, without repeats. ``forum_ids_matched`` / ``forum_ids_unmatched``:
    the well-formed forum ids that did / did not match a sheet row. Malformed
    values are in neither list.
    """

    apc_names: tuple[str, ...] = ()
    forum_ids_matched: tuple[str, ...] = ()
    forum_ids_unmatched: tuple[str, ...] = ()
    route: str = ROUTE_FORUM_ID


def usable_forum_ids(extraction: dict | None) -> list[str]:
    """The well-formed forum ids in ``extraction``, trimmed, first-seen order.

    Exact and case-sensitive (forum ids are case-sensitive tokens). Anything
    that is not a 10-character alphanumeric string is dropped.
    """
    if not isinstance(extraction, dict):
        return []
    raw = extraction.get("openreview_forum_ids")
    if not isinstance(raw, list):
        return []
    seen: list[str] = []
    for value in raw:
        if not isinstance(value, str):
            continue
        cleaned = value.strip()
        if FORUM_ID_RE.fullmatch(cleaned) and cleaned not in seen:
            seen.append(cleaned)
    return seen


async def resolve_apcs_by_forum_id(
    db: AsyncSession,
    extraction: dict | None,
    *,
    assignments: PaperAssignmentRepository | None = None,
) -> ApcResolution:
    """Resolve APCs through the email's forum ids. Never reads paper numbers.

    No well-formed forum id means no database query and an empty result.
    Database errors are not caught: the caller is the best-effort layer.
    """
    forum_ids = usable_forum_ids(extraction)
    if not forum_ids:
        return ApcResolution()
    assignments = assignments or PaperAssignmentRepository()
    by_forum = await assignments.get_by_forum_ids(db, forum_ids)

    names: list[str] = []
    matched: list[str] = []
    unmatched: list[str] = []
    for forum_id in forum_ids:
        row = by_forum.get(forum_id)
        if row is None:
            unmatched.append(forum_id)
            continue
        matched.append(forum_id)
        name = (row.apc_name or "").strip()
        if name and name not in names:
            names.append(name)
    return ApcResolution(
        apc_names=tuple(names),
        forum_ids_matched=tuple(matched),
        forum_ids_unmatched=tuple(unmatched),
    )


# The single entry point callers use. Swap the rule by pointing this elsewhere.
resolve_paper_apcs = resolve_apcs_by_forum_id
