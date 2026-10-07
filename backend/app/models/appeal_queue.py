"""Reject Appeals queue (Z3): mode groups and stored-extraction readers. Pure.

Shared by the repository (which filters on the same groups in SQL), the
resolver and the API, so the three can never disagree about what a group or a
stored identifier means.

MODE GROUPS come from ``draft["appeal_reply"]["mode"]``, written only by the
appeal reply hook:

- ``composed``: an approved reply was composed (merged, standalone).
- ``investigate``: possible wrong-paper review, do not reply yet (no_draft).
- ``reciprocal``: a reciprocal-review complaint for Marc (reciprocal_review).
- ``ai_suggestion``: a flagged AI suggestion (2b/2c), never approved wording. Its
  own group: it is NOT counted inside ``chair_writes``.
- ``chair_writes``: any other mode the hook wrote, including modes added later.
- ``not_drafted``: no mode at all — the draft was not made by the approved
  rules (the switches were off, or the draft predates the hook).

STORED SHAPES: rows processed by older code store single values
``submission_number`` / ``openreview_forum_id``; current code stores the lists
``submission_numbers`` / ``openreview_forum_ids``. :func:`identifier_list` reads
either.

PHASE-1 REASONS (P4): :func:`phase1_block` builds a row's ``appeal.phase1`` from
Jiacheng's stored ``phase1_appeals`` rows when the email has any (quotes cut to
:data:`QUOTE_MAX_CHARS`), else from the names-only snapshot the appeal reply hook
stored in ``draft["appeal_reply"]["phase1"]``, else None.
"""

from __future__ import annotations

# The whitespace a paper number may carry around it, stripped identically by the
# Python normaliser (paper_apc_resolver.normalize_paper_number) and the SQL one
# (PaperAssignmentRepository.get_by_normalized_numbers). Explicit rather than
# str.strip()'s default, so the two sides can never disagree: space, tab, LF, CR,
# vertical tab, form feed and the non-breaking space.
PAPER_NUMBER_WHITESPACE = " \t\n\r\x0b\x0c\xa0"

# Longest author quote served per reason in the queue (never logged).
QUOTE_MAX_CHARS = 240

PHASE1_SOURCE_ROWS = "rows"
PHASE1_SOURCE_SNAPSHOT = "snapshot"

MODE_GROUP_COMPOSED = "composed"
MODE_GROUP_CHAIR_WRITES = "chair_writes"
MODE_GROUP_INVESTIGATE = "investigate"
MODE_GROUP_RECIPROCAL = "reciprocal"
MODE_GROUP_NOT_DRAFTED = "not_drafted"
MODE_GROUP_AI_SUGGESTION = "ai_suggestion"
MODE_GROUPS: tuple[str, ...] = (
    MODE_GROUP_COMPOSED,
    MODE_GROUP_CHAIR_WRITES,
    MODE_GROUP_INVESTIGATE,
    MODE_GROUP_RECIPROCAL,
    MODE_GROUP_NOT_DRAFTED,
    MODE_GROUP_AI_SUGGESTION,
)

COMPOSED_MODES: tuple[str, ...] = ("merged", "standalone")
INVESTIGATE_MODES: tuple[str, ...] = ("no_draft",)
RECIPROCAL_MODES: tuple[str, ...] = ("reciprocal_review",)
AI_SUGGESTION_MODES: tuple[str, ...] = ("ai_suggestion",)

NOTE_STATE_NONE = "none"
NOTE_STATES: tuple[str, ...] = (NOTE_STATE_NONE, "pending", "posting", "posted", "failed")


def appeal_mode(draft) -> str | None:
    """``draft["appeal_reply"]["mode"]``, or None when there is none.

    Mirrors the SQL path read used for filtering: anything other than a missing
    or null mode counts as a mode (a non-string becomes its string form).
    """
    if not isinstance(draft, dict):
        return None
    reply = draft.get("appeal_reply")
    if not isinstance(reply, dict):
        return None
    mode = reply.get("mode")
    if mode is None:
        return None
    return mode if isinstance(mode, str) else str(mode)


def mode_group(mode: str | None) -> str:
    """The queue group of an appeal mode (see the module docstring)."""
    if mode is None:
        return MODE_GROUP_NOT_DRAFTED
    if mode in COMPOSED_MODES:
        return MODE_GROUP_COMPOSED
    if mode in INVESTIGATE_MODES:
        return MODE_GROUP_INVESTIGATE
    if mode in RECIPROCAL_MODES:
        return MODE_GROUP_RECIPROCAL
    if mode in AI_SUGGESTION_MODES:
        return MODE_GROUP_AI_SUGGESTION
    return MODE_GROUP_CHAIR_WRITES


def _as_text(value) -> str | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return str(int(value)) if value.is_integer() else None
    if isinstance(value, str):
        return value.strip() or None
    return None


def identifier_list(extraction, list_key: str, scalar_key: str) -> list[str]:
    """Identifiers from either stored shape, trimmed, blanks dropped, first-seen order.

    The list (``list_key``) is used when the key holds a list; otherwise the old
    single value (``scalar_key``) becomes a one-element list. A string or a
    number is accepted as a value; anything else is skipped.
    """
    if not isinstance(extraction, dict):
        return []
    raw = extraction.get(list_key)
    values = raw if isinstance(raw, list) else [extraction.get(scalar_key)]
    out: list[str] = []
    for value in values:
        text = _as_text(value)
        if text is not None and text not in out:
            out.append(text)
    return out


def submission_numbers(extraction) -> list[str]:
    """The paper numbers an email names, from either stored shape."""
    return identifier_list(extraction, "submission_numbers", "submission_number")


def forum_ids(extraction) -> list[str]:
    """The OpenReview forum ids an email names, from either stored shape."""
    return identifier_list(extraction, "openreview_forum_ids", "openreview_forum_id")


def composer_reasons(draft):
    """``appeal.reasons``: the composer input the appeal reply hook actually used,
    read from ``draft["appeal_reply"]["reasons"]``. None when the hook never ran on
    this draft or recorded no input (every hold records None)."""
    reply = draft.get("appeal_reply") if isinstance(draft, dict) else None
    if not isinstance(reply, dict):
        return None
    reasons = reply.get("reasons")
    if not isinstance(reasons, list):
        return None
    return [r for r in reasons if isinstance(r, str)]


def _quote(value) -> str | None:
    return value[:QUOTE_MAX_CHARS] if isinstance(value, str) else None


def _phase1_from_rows(rows) -> dict:
    """One email's phase-1 block from its stored rows (one row per paper; relation,
    reasons and must_verify are the same on every row of one run)."""
    first = rows[0]
    reasons = []
    for item in first.reasons if isinstance(first.reasons, list) else []:
        if isinstance(item, dict) and isinstance(item.get("reason"), str):
            reasons.append({"reason": item["reason"], "quote": _quote(item.get("quote"))})
    papers: list[str] = []
    for row in rows:
        number = row.submission_number
        if isinstance(number, str) and number and number not in papers:
            papers.append(number)
    return {
        "source": PHASE1_SOURCE_ROWS,
        "relation": first.relation,
        "must_verify": bool(first.must_verify),
        "papers": papers,
        "reasons": reasons,
    }


def _phase1_from_snapshot(draft) -> dict | None:
    """The names-only snapshot the hook stored (source "phase1"), or None."""
    reply = draft.get("appeal_reply") if isinstance(draft, dict) else None
    if not isinstance(reply, dict) or reply.get("source") != "phase1":
        return None
    snap = reply.get("phase1")
    if not isinstance(snap, dict):
        return None
    names = snap.get("reasons") if isinstance(snap.get("reasons"), list) else []
    papers = snap.get("papers") if isinstance(snap.get("papers"), list) else []
    must_verify = snap.get("must_verify")
    return {
        "source": PHASE1_SOURCE_SNAPSHOT,
        "relation": snap.get("relation") if isinstance(snap.get("relation"), str) else None,
        "must_verify": must_verify if isinstance(must_verify, bool) else None,
        "papers": [p for p in papers if isinstance(p, str) and p],
        "reasons": [{"reason": n, "quote": None} for n in names if isinstance(n, str)],
    }


def phase1_block(draft, rows) -> dict | None:
    """``appeal.phase1`` for one email: his stored rows first, else the draft's
    snapshot, else None. ``rows`` are this email's ``phase1_appeals`` rows."""
    if rows:
        return _phase1_from_rows(rows)
    return _phase1_from_snapshot(draft)
