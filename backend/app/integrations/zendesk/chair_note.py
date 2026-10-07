"""Zendesk chair notes (Z2): which emails get one, and what the note says.

A chair who works in Zendesk rather than ConfMail should find ConfMail's draft
for a reject-appeal email on the ticket itself, as an INTERNAL note
(``public: false``), and copy it into their own reply.

This module holds the two PURE pieces only:

- :func:`check_eligibility`: whether an email may get a note.
- :func:`build_chair_note_html`: the note's HTML.

Nothing here posts to Zendesk, touches the database or is called by the
pipeline yet (Z2a). Posting, the claim protocol and the hooks come later; the
claim state lives in ``zendesk_chair_notes`` (see ``ChairNoteRepository``).

Every value that comes from outside ConfMail's own code (the draft, the notes,
APC names, paper numbers, forum ids) is HTML-escaped. The note never carries
tags, an assignee, a group, CCs or a status change; that is the poster's
contract, recorded here so the builder is not mistaken for the place to add
them.
"""

from __future__ import annotations

import hashlib
import html
from dataclasses import dataclass
from urllib.parse import quote

from app.core.config import settings
from app.pipeline.drafter import find_placeholders
from app.pipeline.paper_apc_resolver import ApcResolution

# The fixed first line. It doubles as the marker that recognises ConfMail's own
# notes, so it must not change once notes are live.
MARKER = "ConfMail draft (not sent to the author)"
FOOTER = "Posted by ConfMail. Review and edit before sending."

COPY_START = "COPY BELOW"
COPY_END = "END"
BANNER_DO_NOT_SEND = "Do not send: the chair writes this reply."
BANNER_NO_DRAFT = "Investigate first. Do not reply to or close the ticket yet."
BANNER_RECIPROCAL = "Reciprocal complaint: for Marc to review."
# 2c: the AI suggestion (mode ai_suggestion) is never approved wording. Its banner
# always sits directly under the marker line, the same place in every note.
BANNER_AI_SUGGESTION = "AI-written suggestion, not approved wording"

CHAIR_LABEL = "Chair"
CHAIR_NOT_IDENTIFIED = "not identified from this email"
NUMBERS_LABEL = "Paper number(s) as written in the email (unverified)"
NUMBERS_NONE = "Paper number(s): none found in the email"
OPENREVIEW_LABEL = "OpenReview"
NOTES_LABEL = "Notes for the chair:"

OPENREVIEW_FORUM_URL = "https://openreview.net/forum?id="

# The appeal reply hook's composed modes (appeal_reply_hook.COMPOSED_MODES).
# Repeated rather than imported so this module does not load the composer and
# its template files; a test pins the two sets equal.
COMPOSED_MODES = frozenset({"merged", "standalone"})
MODE_NO_DRAFT = "no_draft"
MODE_RECIPROCAL = "reciprocal_review"
# The flagged AI suggestion (appeal_ai_draft.MODE_AI_SUGGESTION / AI_FLAG_LINE).
# Repeated rather than imported for the same reason; a test pins them equal.
MODE_AI_SUGGESTION = "ai_suggestion"
AI_FLAG_LINE = (
    "[CHAIR: AI-written suggestion, not approved wording; review and edit before sending]"
)

# Zendesk statuses a note may be posted to (D4). Solved and closed are out:
# a closed ticket cannot be written to, and a note on a solved one is not
# needed and its effect is unverified.
ALLOWED_TICKET_STATUSES = frozenset({"new", "open", "pending", "hold"})

# Eligibility outcomes, in the order they are checked.
ELIGIBLE = "eligible"
FLAG_OFF = "flag_off"
NO_TICKET = "no_ticket_id"
NOT_ALLOWLISTED = "ticket_not_allowlisted"
TICKET_STATUS = "ticket_status_not_allowed"
INTENT_OUT_OF_SCOPE = "intent_out_of_scope"
NO_APPEAL_REPLY = "no_appeal_reply"


@dataclass(frozen=True)
class Eligibility:
    """Whether an email may get a chair note, and why not when it may not."""

    eligible: bool
    reason: str


def check_eligibility(email) -> Eligibility:
    """Decide whether ``email`` may get a chair note. Pure, no I/O.

    All of these must hold, checked in this order: the flag is on; the email
    has a Zendesk ticket id; the ticket passes the allow-list; its Zendesk
    status is new / open / pending / hold; its intent is in scope; and its
    stored draft carries the ``appeal_reply`` record, so only approved wording
    or a chair placeholder is ever posted, never a model-written appeal draft.
    """
    if not settings.CHAIR_NOTE_ENABLED:
        return Eligibility(False, FLAG_OFF)

    ticket_id = getattr(email, "zendesk_ticket_id", None)
    if ticket_id is None:
        return Eligibility(False, NO_TICKET)

    allowed_ids = settings.chair_note_ticket_ids
    if allowed_ids is not None and int(ticket_id) not in allowed_ids:
        return Eligibility(False, NOT_ALLOWLISTED)

    status = (getattr(email, "zendesk_status", None) or "").strip().lower()
    if status not in ALLOWED_TICKET_STATUSES:
        return Eligibility(False, TICKET_STATUS)

    classification = getattr(email, "classification", None)
    intent = classification.get("intent") if isinstance(classification, dict) else None
    if intent not in settings.chair_note_intents:
        return Eligibility(False, INTENT_OUT_OF_SCOPE)

    draft = getattr(email, "draft", None)
    appeal_reply = draft.get("appeal_reply") if isinstance(draft, dict) else None
    if not isinstance(appeal_reply, dict):
        return Eligibility(False, NO_APPEAL_REPLY)

    return Eligibility(True, ELIGIBLE)


def _esc(value: str) -> str:
    return html.escape(value, quote=True)


def _paragraphs(text: str | None) -> str:
    """Plain text as escaped HTML: a blank line starts a new <p>, a newline is <br>.

    The same rendering as ``api.v1.emails._text_to_html``, repeated here so the
    integration layer does not import the API layer. Empty text gives "".
    """
    escaped = _esc(text or "").strip()
    if not escaped:
        return ""
    return "".join(
        f"<p>{part.replace(chr(10), '<br>')}</p>" for part in escaped.split("\n\n")
    )


def _paper_numbers(extraction: dict | None) -> list[str]:
    if not isinstance(extraction, dict):
        return []
    raw = extraction.get("submission_numbers")
    if not isinstance(raw, list):
        return []
    numbers: list[str] = []
    for value in raw:
        if not isinstance(value, str):
            continue
        cleaned = value.strip()
        if cleaned and cleaned not in numbers:
            numbers.append(cleaned)
    return numbers


def _without_flag_line(text: str) -> str:
    """The AI suggestion's text with every AI flag line removed, ends stripped."""
    return "\n".join(
        line for line in text.split("\n") if line.strip() != AI_FLAG_LINE
    ).strip()


def _is_copyable(mode: str | None, draft_text: str) -> bool:
    """Only a composed reply with real text and no placeholder may be offered for copying."""
    return (
        mode in COMPOSED_MODES
        and bool(draft_text.strip())
        and not find_placeholders(draft_text)
    )


def build_chair_note_html(
    draft: dict,
    extraction: dict | None,
    resolution: ApcResolution,
) -> str:
    """The chair note's HTML for a stored draft. Pure, no I/O.

    ``draft`` is the email's stored draft dict (``draft_text``,
    ``notes_for_chair``, ``appeal_reply``); ``extraction`` the stored
    extraction; ``resolution`` the APC lookup for this email.

    Layout: the marker line; the chair line; the paper numbers as written; the
    OpenReview links; then, by ``appeal_reply.mode``:

    - ``merged`` / ``standalone``: "COPY BELOW", the stored draft text exactly
      (greeting and sign-off included) between two rules, then "END";
    - ``no_draft``: "Investigate first. Do not reply to or close the ticket yet.";
    - ``reciprocal_review``: "Reciprocal complaint: for Marc to review.";
    - ``ai_suggestion`` (2c): the banner "AI-written suggestion, not approved
      wording" directly under the marker line (so always the second line), and
      in the mode slot the suggestion WITHOUT its [CHAIR: ...] flag line between
      two rules — never "COPY BELOW" / "END", so it can never pass for an
      approved copy block. If nothing usable is left: "Do not send";
    - every other mode, an unknown mode, or a composed mode whose text is empty
      or still holds a [CHAIR: ...] placeholder: "Do not send".

    The chair notes, then the footer, always come after the copy block, never
    inside it.
    """
    draft = draft if isinstance(draft, dict) else {}
    appeal_reply = draft.get("appeal_reply")
    mode = appeal_reply.get("mode") if isinstance(appeal_reply, dict) else None
    draft_text = draft.get("draft_text") if isinstance(draft.get("draft_text"), str) else ""
    notes = draft.get("notes_for_chair") if isinstance(draft.get("notes_for_chair"), str) else ""
    suggestion = _without_flag_line(draft_text) if mode == MODE_AI_SUGGESTION else ""

    parts: list[str] = [f"<p><strong>{_esc(MARKER)}</strong></p>"]
    if mode == MODE_AI_SUGGESTION:
        parts.append(f"<p><strong>{_esc(BANNER_AI_SUGGESTION)}</strong></p>")

    if resolution.apc_names:
        chairs = ", ".join(_esc(name) for name in resolution.apc_names)
    else:
        chairs = _esc(CHAIR_NOT_IDENTIFIED)
    parts.append(f"<p>{CHAIR_LABEL}: {chairs}</p>")

    numbers = _paper_numbers(extraction)
    if numbers:
        parts.append(f"<p>{_esc(NUMBERS_LABEL)}: {', '.join(_esc(n) for n in numbers)}</p>")
    else:
        parts.append(f"<p>{_esc(NUMBERS_NONE)}</p>")

    forum_ids = [*resolution.forum_ids_matched, *resolution.forum_ids_unmatched]
    if forum_ids:
        links = ", ".join(
            f'<a href="{_esc(OPENREVIEW_FORUM_URL + quote(fid, safe=""))}">'
            f"{_esc(OPENREVIEW_FORUM_URL + fid)}</a>"
            for fid in forum_ids
        )
        parts.append(f"<p>{OPENREVIEW_LABEL}: {links}</p>")

    if _is_copyable(mode, draft_text):
        parts.append(f"<p><strong>{_esc(COPY_START)}</strong></p>")
        parts.append("<hr>")
        parts.append(_paragraphs(draft_text))
        parts.append("<hr>")
        parts.append(f"<p><strong>{_esc(COPY_END)}</strong></p>")
    elif mode == MODE_NO_DRAFT:
        parts.append(f"<p><strong>{_esc(BANNER_NO_DRAFT)}</strong></p>")
    elif mode == MODE_RECIPROCAL:
        parts.append(f"<p><strong>{_esc(BANNER_RECIPROCAL)}</strong></p>")
    elif mode == MODE_AI_SUGGESTION and suggestion and not find_placeholders(suggestion):
        parts.append("<hr>")
        parts.append(_paragraphs(suggestion))
        parts.append("<hr>")
    else:
        parts.append(f"<p><strong>{_esc(BANNER_DO_NOT_SEND)}</strong></p>")

    notes_html = _paragraphs(notes)
    if notes_html:
        parts.append(f"<p>{_esc(NOTES_LABEL)}</p>")
        parts.append(notes_html)

    parts.append(f"<p><em>{_esc(FOOTER)}</em></p>")
    return "".join(parts)


def note_body_sha256(body_html: str) -> str:
    """sha256 hex of the note body exactly as it would be sent."""
    return hashlib.sha256(body_html.encode("utf-8")).hexdigest()
