"""Reject Appeals queue (Z3): mode groups and stored-extraction readers. Pure.

Shared by the repository (which filters on the same groups in SQL), the
resolver and the API, so the three can never disagree about what a group or a
stored identifier means.

MODE GROUPS come from ``draft["appeal_reply"]["mode"]``, written only by the
appeal reply hook:

- ``composed``: an approved reply was composed (merged, standalone).
- ``investigate``: possible wrong-paper review, do not reply yet (no_draft).
- ``reciprocal``: a reciprocal-review complaint for Marc (reciprocal_review).
- ``chair_writes``: any other mode the hook wrote, including modes added later.
- ``not_drafted``: no mode at all — the draft was not made by the approved
  rules (the switches were off, or the draft predates the hook).

STORED SHAPES: rows processed by older code store single values
``submission_number`` / ``openreview_forum_id``; current code stores the lists
``submission_numbers`` / ``openreview_forum_ids``. :func:`identifier_list` reads
either.
"""

from __future__ import annotations

MODE_GROUP_COMPOSED = "composed"
MODE_GROUP_CHAIR_WRITES = "chair_writes"
MODE_GROUP_INVESTIGATE = "investigate"
MODE_GROUP_RECIPROCAL = "reciprocal"
MODE_GROUP_NOT_DRAFTED = "not_drafted"
MODE_GROUPS: tuple[str, ...] = (
    MODE_GROUP_COMPOSED,
    MODE_GROUP_CHAIR_WRITES,
    MODE_GROUP_INVESTIGATE,
    MODE_GROUP_RECIPROCAL,
    MODE_GROUP_NOT_DRAFTED,
)

COMPOSED_MODES: tuple[str, ...] = ("merged", "standalone")
INVESTIGATE_MODES: tuple[str, ...] = ("no_draft",)
RECIPROCAL_MODES: tuple[str, ...] = ("reciprocal_review",)

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
