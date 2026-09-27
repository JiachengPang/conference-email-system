"""Extractor (which submission an email is about, and who it names).

Turns the distiller's RAW identification lines into validated, deduplicated
structures. The split is deliberate: :mod:`app.pipeline.distiller` is the
transport parser and keeps whatever the model emitted verbatim, so a malformed
value survives the wire; this module is the only place that decides what a
value MEANS. Keeping validation here means a bad value can be inspected (and
later counted) with the full picture, instead of vanishing at parse time where
it would look identical to the model never emitting the line at all.

Failure policy: best-effort, like the distiller. ``extract`` never raises — a
normalization bug degrades to an empty result rather than breaking the pipeline
that called it, since knowing which paper an email is about is an enrichment,
never a precondition for drafting a reply.

Two paths, and they are mutually exclusive by design. When the distiller ran,
its answer is used outright; only when it did not run at all does a regex
fallback read the raw subject/body. Regex is deliberately NOT a supplement that
tops up a present LLM result — that would need per-field provenance to stay
honest, and ``method`` is a clean three-value record of which path produced the
whole result.

TWO fields sit deliberately OUTSIDE that split — ``openreview_note_id`` and
``openreview_notification_sender`` — and both are read from the raw text on BOTH
paths. The test for membership is exactly one question: DOES THE DISTILLER'S
PROMPT CONTRACT ASK FOR THIS EXACT VALUE?

* ``openreview_forum_ids`` — YES. The prompt asks for ``OPENREVIEW_ID`` lines,
  so the model genuinely looked, and an empty answer is a real answer. Regex
  must never top it up; doing so would override the model with a weaker tool and
  leave ``method`` unable to say which field came from where. UNTOUCHED.
* ``openreview_note_id`` — NO. The prompt has no note-id line at all, and its
  ``OPENREVIEW_ID`` line carries a BARE forum id with the link, and therefore
  the ``noteId`` parameter, already discarded. Nothing was asked, so nothing was
  answered.
* ``openreview_notification_sender`` — NO, for the same reason.

For the latter two an unset value on the LLM path would not record "the model
looked and found none"; it would record that nobody looked, on the path
production actually runs. Both are structural facts about the raw text with one
exact, machine-checkable answer and no judgment to make, so both are read by
regex either way. This is bounded to signals of that KIND and is NOT licence to
regex-fill a value the prompt does ask for.

One field is NOT SET BY THIS MODULE AT ALL. ``is_reciprocal_dispute`` is a
judgment about what an email is ARGUING, not a structural fact about its text,
so there is no exact, machine-checkable answer for regex to supply. It used to
be copied here from the distiller, which asked the question inside its main
prompt; that question now lives in its own conditional call
(``app.pipeline.reciprocal_detector``) because embedding it cost real retrieval
quality — it perturbed the QUERY lines the same prompt produces
(reject_appeal.md D27/D35).

So ``extract()`` leaves the field at ``None`` on BOTH paths, always, and
``orchestrator._compute`` attaches the detector's answer afterwards (it cannot
be done in here: ``extract`` is synchronous, pure and never-raises, and a
network call would end all three). A ``None`` coming out of this module
therefore means only "not answered here" — never "ruled out".

The fallback is tuned for PRECISION over recall. Roughly half of real threads
carry no submission reference at all, so returning nothing is the ordinary
outcome, not a failure; attaching the WRONG paper to a ticket is far more
costly to a chair than attaching none.
"""

import logging
import re
from typing import Literal

from pydantic import BaseModel, Field, computed_field

from app.pipeline.distiller import DistillResult
from app.pipeline.quoted_reply import extract_reply_text

logger = logging.getLogger(__name__)

_AUTHOR_FIELD_SEPARATOR = "|"
# The model is told to emit exactly `name | email | affiliation`.
_AUTHOR_FIELD_COUNT = 3

ExtractionMethod = Literal["llm_distiller", "regex_fallback", "none"]

# --- regex-fallback patterns ------------------------------------------------
# Real submission numbers are 4-5 digits and are only trustworthy next to a cue
# word: an ungated \d{4,5} sweep pulls in years, dates, counts and phone
# fragments. `filler` captures at most ONE intervening word so the cue stays
# close, and is inspected afterwards — a number the cue introduces directly
# ("Submission 2026") is believed, one reached across a word ("paper due 2026")
# is not.
_SUBMISSION_CUE_RE = re.compile(
    r"(?<![A-Za-z0-9])"
    r"(?:paper|submission)s?"
    r"(?:\s*(?:id|no\.?|number))?"
    r"(?P<filler>(?:\s*[A-Za-z][A-Za-z.\-]{0,11}){0,1})"
    r"\s*[#:\-]?\s*"
    r"(?P<number>\d{4,5})"
    r"(?!\d)",
    re.IGNORECASE,
)
# A bare "#12345" is its own cue — no preceding word needed. The lookbehind
# keeps it out of URL fragments and anchors.
_HASH_NUMBER_RE = re.compile(r"(?<![\w/])#(?P<number>\d{4,5})(?!\d)")

# Anchored to the two link shapes that actually carry a forum id. `?id=` alone
# is NOT enough: AAAI's committee/group URLs (openreview.net/group?id=...) share
# that shape and are confirmed false positives in real traffic. The trailing
# lookahead rejects longer ids rather than silently truncating them to 10.
_OPENREVIEW_FORUM_ID_RE = re.compile(
    r"openreview\.net/(?:forum|pdf)\?id=(?P<forum_id>[A-Za-z0-9]{10})"
    r"(?![A-Za-z0-9])",
    re.IGNORECASE,
)

# --- OpenReview note id (one comment inside a forum) ------------------------
# A forum link may name the specific note under discussion in a `noteId` query
# parameter, as in openreview.net/forum?id=ll0avn6ylq&noteId=jnHgRMHgrm.
#
# A note id is meaningful ONLY alongside the forum id it travelled with, so it
# must never be matched on its own: a forum id taken from one link and a noteId
# taken from another would together name a comment that does not exist in that
# forum, and would look exactly like a real pair. The link is therefore matched
# WHOLE and its query string read as a single unit; the two parameter patterns
# below are only ever applied to one such query at a time.
#
# The forum-id pattern above is deliberately NOT reused or modified here — this
# is a strictly additive scan, and `openreview_forum_ids` keeps precisely the
# behaviour it had.
_OPENREVIEW_LINK_RE = re.compile(
    r"openreview\.net/(?:forum|pdf)\?(?P<query>[^\s\"'<>]*)",
    re.IGNORECASE,
)
# Anchoring each parameter to a separator (start-of-query, `&`, or an
# HTML-escaped `&amp;`) is load-bearing twice over. It stops `id=` matching the
# tail of `noteId=` — under IGNORECASE those three characters are identical —
# and it lets EITHER parameter come first, so both query orderings parse.
_OPENREVIEW_PARAM_FORUM_ID_RE = re.compile(
    r"(?:^|&amp;|&)id=(?P<forum_id>[A-Za-z0-9]{10})(?![A-Za-z0-9])",
    re.IGNORECASE,
)
# The value shape is looser than the forum id's fixed 10 on purpose: OpenReview
# API v1 note ids are numeric while v2 ids are 10-character tokens. A named
# `noteId` parameter inside an already-validated forum link is specific enough
# that a permissive value carries no false-positive risk — the precision comes
# from the surrounding link, not from the token. The trailing lookahead still
# rejects an over-long token outright rather than truncating it into a match.
_OPENREVIEW_PARAM_NOTE_ID_RE = re.compile(
    r"(?:^|&amp;|&)noteId=(?P<note_id>[A-Za-z0-9]{1,32})(?![A-Za-z0-9])",
    re.IGNORECASE,
)

# --- OpenReview notification sender ----------------------------------------
# OpenReview mails each venue from its own notification address, shaped
# <venue>-notifications@openreview.net (aaai2027-notifications@openreview.net
# today). The venue prefix changes every conference and every year, so it is
# matched as a SHAPE and never hardcoded.
#
# Anchored on the ADDRESS ALONE, deliberately. The address turns up inside a
# quoted header whose field labels are written by the replier's mail client and
# may be in any language — "From:", "De:", "Von:", "发件人:" — so any pattern
# reaching for the label would work for one locale and silently fail for the
# rest. The address is the part that is identical everywhere.
#
# Precision comes from three guards:
#  * The venue prefix is REQUIRED. A bare notifications@openreview.net has no
#    venue and is not this signal; the leading lookbehind is what makes that
#    rejection airtight rather than incidental, since without it the engine
#    would happily retry one character in and match a suffix of a longer local
#    part (foo.bar-notifications@... must not read as bar-notifications@...).
#  * The domain is literal and exact.
#  * The trailing lookahead rejects a lookalike that merely STARTS with the real
#    domain — openreview.net.evil.com, openreview.network — instead of matching
#    its prefix and reporting a bogus address.
#
# SCOPE: this reports that the address APPEARS in the text, not that it appeared
# as a header. Prose ("I got mail from aaai2027-notifications@openreview.net")
# matches too. That is intended here — the stronger "this is a reply to a
# notification" inference is a separate, later decision that combines this
# signal with the forum/note pairing, and it should be the thing that weighs
# them, not this detector second-guessing itself.
_OPENREVIEW_NOTIFICATION_SENDER_RE = re.compile(
    r"(?<![A-Za-z0-9._%+\-])"
    r"(?P<address>"
    r"[A-Za-z0-9](?:[A-Za-z0-9\-]*[A-Za-z0-9])?"
    r"-notifications@openreview\.net"
    r")"
    r"(?![A-Za-z0-9.\-])",
    re.IGNORECASE,
)

# A number sitting directly after a conference designator is that conference's
# year (AAAI-26, AAAI 2026, IAAI-2027), never a submission number.
_CONFERENCE_PREFIX_RE = re.compile(r"(?:AAAI|IAAI|EAAI)[-\s]*$", re.IGNORECASE)
# Plausible conference years. Only consulted when a filler word separated the
# cue from the number, so a genuine 4-digit submission number in this range
# (they exist) still survives when the cue introduces it directly.
_CONFERENCE_YEAR_MIN = 2020
_CONFERENCE_YEAR_MAX = 2035

# NOTE: the quoted-notification tie-break (a reply-marker regex plus a mined
# list of AAAI notification phrases) lived here and has been REMOVED, not
# disabled. It existed solely to pick between a subject number and a
# conflicting body number; now that both are reported there is nothing to pick,
# so it was dead weight rather than a guard. Its acceptance logic was never
# involved — that lives in _accept_submission_match below and is untouched.


def _reads_as_conference_year(value: str) -> bool:
    return len(value) == 4 and _CONFERENCE_YEAR_MIN <= int(value) <= _CONFERENCE_YEAR_MAX


def _accept_submission_match(match: re.Match[str], text: str) -> bool:
    """Reject the two false-positive shapes real ticket traffic produces."""
    if _CONFERENCE_PREFIX_RE.search(text[: match.start("number")]):
        return False
    filler = (match.groupdict().get("filler") or "").strip()
    return not (filler and _reads_as_conference_year(match.group("number")))


def _find_submission_numbers(subject: str, body: str) -> list[str]:
    """EVERY trustworthy submission number, subject first, deduplicated.

    Scanning no longer stops at the first hit: an email may name several
    submissions, and returning one silently discarded the rest. WHAT counts as
    a match is unchanged — every candidate still passes
    :func:`_accept_submission_match`, so the cue-word gate, the
    conference-designator rejection and the year rule all still apply.

    Order is the existing traversal, kept deliberately: subject before body
    (discovery found the subject usually carries the conference's own,
    highest-value id), and within each, cue-worded matches before the bare
    ``#NNNNN`` form (the stronger signal first). That was the old precedence
    chain; it now decides list ORDER rather than which single value survives.

    Note this replaces the quoted-notification tie-break outright. That existed
    only to choose between a subject number and a conflicting body number —
    with both reported there is nothing left to disambiguate.
    """
    found: list[str] = []
    for text in (subject or "", body or ""):
        for pattern in (_SUBMISSION_CUE_RE, _HASH_NUMBER_RE):
            for match in pattern.finditer(text):
                if _accept_submission_match(match, text):
                    found.append(match.group("number"))
    return _dedupe_identifiers(found)


def _find_openreview_forum_ids(subject: str, body: str) -> list[str]:
    """EVERY valid forum id, subject first, deduplicated.

    Dedup is case-SENSITIVE (via :func:`_dedupe_identifiers`) because forum ids
    are case-sensitive tokens — two ids differing only in case are two
    different papers.
    """
    found: list[str] = []
    for text in (subject or "", body or ""):
        for match in _OPENREVIEW_FORUM_ID_RE.finditer(text):
            found.append(match.group("forum_id"))
    return _dedupe_identifiers(found)


def _find_openreview_note_pairs(subject: str, body: str) -> list[tuple[str, str]]:
    """Every ``(forum_id, note_id)`` read from a SINGLE link's query string.

    Pairing is the whole point. Each link is matched whole and its query read
    once, so the two values in a pair always came from the same URL; a forum id
    in one link and a ``noteId`` in another are never combined. A link carrying
    only one of the two contributes nothing — a note id with no forum beside it
    names a comment we cannot place, and a forum id with no note is already the
    job of :func:`_find_openreview_forum_ids`.

    Either parameter ordering parses (``?id=...&noteId=...`` and
    ``?noteId=...&id=...``); see the pattern comments for why that is anchored
    rather than assumed. Order of results is the existing traversal, subject
    before body. Not deduplicated: the caller takes the first usable pair.
    """
    pairs: list[tuple[str, str]] = []
    for text in (subject or "", body or ""):
        for link in _OPENREVIEW_LINK_RE.finditer(text):
            query = link.group("query")
            forum = _OPENREVIEW_PARAM_FORUM_ID_RE.search(query)
            note = _OPENREVIEW_PARAM_NOTE_ID_RE.search(query)
            if forum is not None and note is not None:
                pairs.append((forum.group("forum_id"), note.group("note_id")))
    return pairs


def _first_note_id_for(
    pairs: list[tuple[str, str]], forum_ids: list[str]
) -> str | None:
    """The first note id whose forum id this result actually reports.

    A coherence gate, not a second filter on validity. It keeps the scalar
    consistent with the list beside it: a note id whose forum is missing from
    ``openreview_forum_ids`` would point into a discussion the result never
    mentions, which reads as a bug to anything downstream.

    Concretely it suppresses the ``?noteId=...&id=...`` ordering today. The pair
    scan above reads that shape, but the untouched forum-id pattern is anchored
    to a literal ``?id=`` and does not, so the forum id is absent from the list
    and the note id is withheld rather than orphaned. That asymmetry is a
    consequence of leaving the existing pattern exactly as it was, and it is
    self-correcting: widen that pattern and this gate opens with it, no second
    edit needed.
    """
    reported = set(forum_ids)
    for forum_id, note_id in pairs:
        if forum_id in reported:
            return note_id
    return None


def _find_openreview_notification_sender(subject: str, body: str) -> str | None:
    """The first OpenReview notification address in the text, or None.

    Returned VERBATIM rather than lowercased. The value's job is to show what
    the email actually said, and normalizing it would quietly discard that; it
    also matches :func:`_dedupe_identifiers`, which does not casefold either.
    An email address is case-insensitive in practice, so anything COMPARING
    this value must casefold it — the field description says so.

    First-seen wins, subject before body, the same traversal every other finder
    here uses. A quoted chain can carry the same address several times and, in
    principle, two different venues; one value is what the field holds, and the
    first is the least arbitrary choice available without inventing a ranking.
    """
    for text in (subject or "", body or ""):
        match = _OPENREVIEW_NOTIFICATION_SENDER_RE.search(text)
        if match is not None:
            return match.group("address")
    return None


class AuthorMention(BaseModel):
    """One person an email identifies. Every field is independently optional.

    A mention with only a name is still useful (it is who wrote in), so a
    missing piece never invalidates the rest of the entry. An entry with no
    populated field at all carries no information and is dropped upstream in
    :func:`_parse_author`.
    """

    name: str | None = Field(default=None, description="Display name, if given.")
    email: str | None = Field(default=None, description="Email address, if given.")
    affiliation: str | None = Field(
        default=None, description="Institution/affiliation, if given."
    )


class ExtractionResult(BaseModel):
    """Which submissions an email refers to, and who it names.

    The identifier and author fields are LISTS: an email may legitimately name
    several submissions (an appeal covering two desk rejections, a reviewer
    asking to be unassigned from four papers), and reporting only one silently
    discarded the rest.

    ``openreview_note_id`` is the deliberate exception and is a SCALAR. It names
    one comment inside one forum, so unlike a submission reference it is only
    meaningful attached to a single forum id — a set of note ids with no record
    of which forum each belongs to would not be usable.

    ``openreview_notification_sender`` is a scalar too, and is the one field
    ``method`` does NOT describe — see its own note and the module docstring.

    ``openreview_reply_candidate`` is DERIVED from two of the fields above and
    stored nowhere. It is a plain boolean AND, so it introduces no new detection
    and no new judgment — see its own note.

    ``extracted_reply_text`` is the one field that is not about identity at all.
    It is STORED rather than derived, unlike the flag above: it is a function of
    the raw ``body``, which this model does not carry, so there is nothing on
    the object to recompute it from.

    ``method`` records HOW the values were obtained, not how well: an
    ``llm_distiller`` result with everything empty means the model looked and
    found nothing, which is a real finding and distinct from ``none`` (nothing
    looked at all). That distinction lives in ``method`` and in whether the
    whole result is stored at all — an EMPTY LIST means "examined, found none",
    never "not examined".
    """

    submission_numbers: list[str] = Field(
        default_factory=list,
        description="Submission/paper numbers as identified, deduplicated, in "
        "first-seen order. Empty when the email named none.",
    )
    openreview_forum_ids: list[str] = Field(
        default_factory=list,
        description="OpenReview forum ids as identified, deduplicated, in "
        "first-seen order. Empty when the email named none.",
    )
    openreview_note_id: str | None = Field(
        default=None,
        description="OpenReview note (Official Comment) id, when a forum link "
        "carried one as its `noteId` parameter. Read from the SAME link as its "
        "forum id and never paired across links, so it always names a comment "
        "inside a forum this result also reports. None when no link carried "
        "one — which, like an empty list, means 'looked, found none'. Read from "
        "the raw text on BOTH paths and therefore NOT described by `method`: "
        "the distiller is never asked for a note id (its OPENREVIEW_ID line "
        "reports a bare forum id with the link discarded), so there is no model "
        "answer here to preserve or override.",
    )
    openreview_notification_sender: str | None = Field(
        default=None,
        description="The OpenReview per-venue notification address "
        "(<venue>-notifications@openreview.net) found in the text, verbatim; "
        "None when none appears. Reported as the matched ADDRESS rather than a "
        "bare boolean: it answers 'why was this flagged' without re-running the "
        "scan, and its venue prefix names the conference and year the quoted "
        "notification came from, which a True could not. The boolean is always "
        "recoverable as `is not None`; the address is not recoverable from a "
        "boolean. Case is preserved as found, so COMPARE CASE-INSENSITIVELY. "
        "Unlike every other field here, this one is read from the raw body on "
        "BOTH paths and is therefore NOT described by `method`.",
    )
    extracted_reply_text: str = Field(
        default="",
        description="What the person actually wrote, with quoted history "
        "removed and the ends trimmed (app.pipeline.quoted_reply). Read from "
        "the raw body on BOTH paths and therefore NOT described by `method`: "
        "this is a string operation on the text, not an answer the distiller "
        "was ever asked for. Empty string when the body was entirely quoted "
        "material, or absent. TEST EMPTINESS ON THIS STRING — it is a more "
        "reliable 'wrote nothing new' signal than any boundary check, because a "
        "body whose quote is preceded by a blank line has a non-zero boundary "
        "yet still contains no reply.",
    )
    authors: list[AuthorMention] = Field(
        default_factory=list,
        description="People the email identifies, deduplicated, in first-seen "
        "order.",
    )
    method: ExtractionMethod = Field(
        default="none",
        description="Which path produced the IDENTIFIER and AUTHOR fields. It "
        "does not describe `openreview_note_id`, "
        "`openreview_notification_sender`, or `extracted_reply_text`, all of "
        "which are read from the raw text on either path.",
    )
    is_reciprocal_dispute: bool | None = Field(
        default=None,
        description="True when the email disputes or concedes a desk rejection "
        "grounded in reciprocal-review duties, False when the model says it "
        "does not. TRI-STATE: None means unanswered, which is NOT False — an "
        "unanswered flag is no evidence either way, while False is a positive "
        "'ruled out'. Never infer one from the other.\n\n"
        "NOT SET BY THE EXTRACTOR, on either path. `extract()` always leaves "
        "it None; `orchestrator._compute` attaches the answer from the "
        "separate `reciprocal_detector` call, which runs only when the intent "
        "is `desk_reject_appeal`. So `method` does NOT describe this field the "
        "way it describes the identifier and author fields — a None here "
        "usually means the detector was never asked (a non-appeal intent, or "
        "the feature gated off), not 'looked and found nothing'.\n\n"
        "On a thread the detector judges the WHOLE conversation, not just the "
        "latest turn, so this does not flip to False when a dispute thread's "
        "newest message is only a follow-up.",
    )

    @computed_field
    @property
    def openreview_reply_candidate(self) -> bool:
        """Both OpenReview signals present: a reply to a notification.

        True IFF this result carries BOTH a ``openreview_note_id`` and a
        ``openreview_notification_sender``. Neither alone is trusted — a forum
        link can be pasted into any question, and the notification address can
        be quoted in an email that is not a reply to it — so the combination is
        the signal, not either half.

        The note id, not the forum-id list, is the left operand ON PURPOSE. A
        bare forum id says only "some OpenReview paper is mentioned"; a note id
        says "a specific comment", and by the coherence gate in
        :func:`_first_note_id_for` it can only exist alongside a forum id this
        result also reports. So ``openreview_note_id is not None`` already means
        "a real forum+note PAIR", and adding a forum-id check beside it would be
        redundant rather than stricter.

        DERIVED, never stored or passed in — the same rule the API's
        ``_forced_policy_applied`` follows for the same reason. Both operands
        already live on this object, so recomputing costs nothing and the flag
        cannot drift from the two fields it summarizes. That also settles the
        "same email" question structurally rather than by discipline: there is
        no construction site to keep in step, no argument to thread through, and
        no way for a caller to hand in a value contradicting the fields beside
        it. A stale or corrupt persisted value is ignored and recomputed on
        load, so it self-heals rather than lying.

        Reachable on BOTH paths, production included. It briefly was not: when
        this flag was introduced, ``openreview_note_id`` was hardcoded None on
        the LLM path, so the left operand was unreachable exactly where
        ``QUERY_STRATEGY=distill`` runs. That was a misclassification of the
        note id, not a flaw in this AND, and it has since been corrected — both
        operands are now read from the raw text on either path. The logic here
        is unchanged from the day it was written.
        """
        return (
            self.openreview_note_id is not None
            and self.openreview_notification_sender is not None
        )


def _dedupe_identifiers(values: list[str]) -> list[str]:
    """Strip blanks and exact repeats from raw identifiers, first-seen order.

    EXACT match, deliberately — not the ``casefold`` used for author names.
    These are opaque tokens rather than free text, so there is no casing or
    spacing variation to fold, and folding case would be actively WRONG for an
    OpenReview forum id: ``Ab3xY9kLm2`` and ``ab3xy9klm2`` are different ids.

    Blanks are dropped defensively. The distiller already filters them, but a
    model's output shape is never fully trusted here, and one stray empty line
    would otherwise render as a blank row in the panel.
    """
    seen: set[str] = set()
    unique: list[str] = []
    for value in values:
        cleaned = value.strip()
        if not cleaned or cleaned in seen:
            continue
        seen.add(cleaned)
        unique.append(cleaned)
    return unique


def _field_or_none(value: str) -> str | None:
    """Normalize one raw author field: blank or the ``NONE`` sentinel → None."""
    cleaned = value.strip()
    if not cleaned or cleaned.upper() == "NONE":
        return None
    return cleaned


def _parse_author(raw: str) -> AuthorMention | None:
    """Parse one raw ``name | email | affiliation`` string; None if empty.

    Malformed input is parsed BEST-EFFORT rather than skipped, because the
    distiller deliberately forwards malformed lines here to be salvaged, and a
    dropped line silently loses a real person:

    * Fewer than three fields — the missing trailing fields become None, so a
      bare ``John Doe`` reads as a name-only mention. This is the same shape as
      an obedient ``John Doe | NONE | NONE``, which is the point: the model
      omitting separators should not change the meaning.
    * More than three fields — the surplus is rejoined into the affiliation
      rather than truncated away. A stray separator is far likelier to sit
      inside an affiliation (``Dept | Univ``) than to signal a fourth value, and
      dropping the tail would silently discard real text.
    * Every field empty — returns None; an entry that identifies nobody is not
      a mention.
    """
    parts = raw.split(_AUTHOR_FIELD_SEPARATOR)
    if len(parts) > _AUTHOR_FIELD_COUNT:
        head = parts[: _AUTHOR_FIELD_COUNT - 1]
        tail = _AUTHOR_FIELD_SEPARATOR.join(parts[_AUTHOR_FIELD_COUNT - 1 :])
        parts = [*head, tail]
    parts += [""] * (_AUTHOR_FIELD_COUNT - len(parts))

    mention = AuthorMention(
        name=_field_or_none(parts[0]),
        email=_field_or_none(parts[1]),
        affiliation=_field_or_none(parts[2]),
    )
    if mention.name is None and mention.email is None and mention.affiliation is None:
        return None
    return mention


def _dedupe_authors(mentions: list[AuthorMention]) -> list[AuthorMention]:
    """Drop repeats on (name, email), case-insensitively, keeping first-seen.

    ``casefold`` rather than ``lower``: real ticket traffic carries non-ASCII
    author names, and only casefold folds them correctly.

    Affiliation is NOT part of the key — the same person written once with and
    once without their institution is one person. First-seen wins so the order
    the model reported (sender first, per the prompt) is preserved.
    """
    seen: set[tuple[str | None, str | None]] = set()
    unique: list[AuthorMention] = []
    for mention in mentions:
        key = (
            mention.name.casefold() if mention.name else None,
            mention.email.casefold() if mention.email else None,
        )
        if key in seen:
            continue
        seen.add(key)
        unique.append(mention)
    return unique


class EmailExtractor:
    """Resolves which submission an email is about and who it names."""

    def extract(
        self,
        subject: str,
        body: str,
        sender: str,
        sender_name: str | None,
        distilled: DistillResult | None,
    ) -> ExtractionResult:
        """Best-effort extraction. Never raises.

        ``sender`` / ``sender_name`` feed the regex fallback only, and on the
        LLM path ``subject`` / ``body`` are unused for identifier extraction on
        purpose — the model already read them itself.

        The exceptions are ``openreview_note_id``,
        ``openreview_notification_sender`` and ``extracted_reply_text``, all
        read from the raw text on either path. None is a supplement to a present
        LLM result, because the distiller's prompt asks for none of them — see
        the module docstring for the one question that decides which fields may
        be read this way. ``extracted_reply_text`` is the least arguable of the
        three: it is a pure string operation on the body, and there is no shape
        in which a model could be asked to return it as a structured field.

        When ``distilled`` is present it is trusted outright, INCLUDING when
        ALL THREE of its lists are empty: the prompt directs the model to read
        both subject and body and to emit no line when there is genuinely no
        identifier, so an empty LLM result is an answer, not a failure to
        re-litigate with a strictly weaker tool. It stays ``llm_distiller``
        with empty lists rather than falling through to regex.
        """
        try:
            # Computed BEFORE the branch, so the two paths cannot drift on a
            # field that has exactly one correct answer either way.
            notification_sender = _find_openreview_notification_sender(
                subject, body
            )

            if distilled is None:
                return self._extract_by_regex(
                    subject, body, sender, sender_name, notification_sender
                )

            authors = [
                mention
                for raw in distilled.authors_raw
                if (mention := _parse_author(raw)) is not None
            ]
            # The MODEL's forum-id answer, untouched — the prompt asks for
            # these, so they are its to report. Hoisted to a local only because
            # the note-id gate below needs to compare against the very list this
            # result will carry.
            forum_ids = _dedupe_identifiers(distilled.openreview_ids_raw)

            return ExtractionResult(
                submission_numbers=_dedupe_identifiers(
                    distilled.submission_numbers_raw
                ),
                openreview_forum_ids=forum_ids,
                # Read by regex here exactly as on the regex path, because
                # the distiller is never asked for a note id: its prompt has no
                # note-id line, and its OPENREVIEW_ID line reports a BARE forum
                # id with the link — and so the `noteId` parameter — already
                # discarded. There is therefore no model answer being overridden
                # here; leaving it None would record that nobody looked, on the
                # path production actually runs (QUERY_STRATEGY=distill).
                #
                # Note the asymmetry with `openreview_forum_ids` directly above,
                # which the prompt DOES ask for and which stays the model's
                # answer untouched. "Is this exact value in the prompt contract"
                # is the whole test; see the module docstring.
                #
                # Gated against `forum_ids` — the MODEL's list, i.e. the very
                # one this result carries — so the scalar can still never name a
                # forum the result omits. That is the same coherence rule the
                # regex path applies, stated once and applied to whichever list
                # the path reports. Consequence, accepted deliberately: if the
                # model misses a forum id the regex found, its note id is
                # withheld. That fails toward reporting less, matching this
                # module's precision-over-recall stance, and is far better than
                # emitting a note id pointing into a forum the result never
                # mentions.
                openreview_note_id=_first_note_id_for(
                    _find_openreview_note_pairs(subject, body), forum_ids
                ),
                # Same rule, same reason — the prompt asks for no sender either.
                openreview_notification_sender=notification_sender,
                # Identical call on both paths, deliberately UNCONDITIONAL: it
                # is not gated on `openreview_reply_candidate` or on anything
                # else, because quote-stripped body text is general-purpose and
                # every email has some. Gating it on the OpenReview signals
                # would make a general utility silently unavailable to whatever
                # asks next.
                #
                # Called here rather than hoisted above the branch like
                # `notification_sender`: that hoist exists so ONE value feeds a
                # coherence gate, and threading this through would add a second
                # defaulted parameter to `_extract_by_regex` that silently
                # yields "" if a direct caller omits it. Two identical calls of
                # a pure single-argument function cannot disagree, and a
                # both-paths-agree test pins that they do not.
                extracted_reply_text=extract_reply_text(body),
                authors=_dedupe_authors(authors),
                # NOTE: `is_reciprocal_dispute` is deliberately NOT set here.
                # It is left at its `None` default on this path exactly as on
                # the regex path, and `orchestrator._compute` attaches the
                # answer from `reciprocal_detector` afterwards. The distiller no
                # longer carries the value — it no longer parses one, because
                # the question moved out of its prompt (see the module
                # docstring). Re-adding a copy here would resurrect the coupling
                # that change exists to remove.
                method="llm_distiller",
            )
        except Exception as exc:  # noqa: BLE001 - extraction must never raise
            logger.warning(
                "Extraction failed (%s: %s); returning an empty result.",
                type(exc).__name__,
                exc,
            )
            return ExtractionResult(method="none")

    def _extract_by_regex(
        self,
        subject: str,
        body: str,
        sender: str,
        sender_name: str | None,
        notification_sender: str | None = None,
    ) -> ExtractionResult:
        """Fallback used only when the distiller did not run.

        The author here is the SENDER, constructed directly rather than parsed:
        the envelope already gives a clean name and address, so none of the
        malformed-string salvage the LLM path needs applies. Signature-block
        parsing is deliberately out of scope — co-authors named in the body are
        exactly what the LLM path is for.

        The "usable" bar for ``method``, restated for lists: at least one
        populated field, i.e. a NON-EMPTY submission-number list, OR a non-empty
        forum-id list, OR a sender mention with a name or an address. Same rule
        as before, with "non-empty list" where "non-None scalar" used to be.
        ``none`` is therefore still reserved for input carrying no identifying
        trace whatsoever — in practice a malformed or synthetic ingest, since a
        real email always has a sender. That is the intended reading: ``none``
        means "nothing to go on", not "regex found no number", which is an
        ordinary outcome recorded as ``regex_fallback`` with empty lists.
        """
        submission_numbers = _find_submission_numbers(subject, body)
        forum_ids = _find_openreview_forum_ids(subject, body)
        # Gated on `forum_ids` so the scalar can never name a forum the list
        # omits. Deliberately NOT part of `found_anything` below: a note id only
        # ever accompanies a forum id that already counted, so including it
        # could not change the outcome, and adding it would imply it can.
        note_id = _first_note_id_for(
            _find_openreview_note_pairs(subject, body), forum_ids
        )

        authors: list[AuthorMention] = []
        name = _field_or_none(sender_name or "")
        email = _field_or_none(sender or "")
        if name is not None or email is not None:
            authors.append(AuthorMention(name=name, email=email))

        # `notification_sender` is deliberately absent from `found_anything`:
        # `method` describes the identifier extraction, and this field is
        # explicitly outside it, so letting it flip `none` to `regex_fallback`
        # would make `method` claim something it does not mean. The corner that
        # exposes is a text carrying a notification address and NOTHING else,
        # not even a sender — which reports `method="none"` beside a populated
        # address. Pinned by test rather than smoothed over, and unreachable in
        # production, where every real email has a sender.
        # `extracted_reply_text` is likewise absent from `found_anything`, and
        # the case for excluding it is STRONGER than for the two scalars above:
        # it is non-empty for essentially every real body, so counting it would
        # make `method="none"` nearly unreachable rather than merely exposing a
        # corner. `method` describes identifier extraction; a body having text
        # in it is not an identifier finding.
        found_anything = bool(submission_numbers or forum_ids or authors)
        return ExtractionResult(
            submission_numbers=submission_numbers,
            openreview_forum_ids=forum_ids,
            openreview_note_id=note_id,
            openreview_notification_sender=notification_sender,
            extracted_reply_text=extract_reply_text(body),
            authors=authors,
            method="regex_fallback" if found_anything else "none",
        )
