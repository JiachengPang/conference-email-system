"""Pydantic v2 API / domain schemas.

These models define the contracts passed between modules and over the API.
They are intentionally logic-free — validation only. The ORM models in
`app.db.models` map to/from these (Piece 5).
"""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, EmailStr, Field, computed_field

from app.models.enums import (
    EmailStatus,
    RoutingLane,
    SensitivityLevel,
)


# ---------------------------------------------------------------------------
# Inbound payloads
# ---------------------------------------------------------------------------
class EmailIn(BaseModel):
    """An inbound conference email as received by the system."""

    sender: EmailStr = Field(..., description="Email address of the sender.")
    sender_name: str | None = Field(
        default=None, description="Display name of the sender, if known."
    )
    subject: str = Field(..., description="Email subject line.")
    body: str = Field(..., description="Plain-text body of the email.")
    received_at: datetime | None = Field(
        default=None, description="When the email was received (server time if omitted)."
    )


# ---------------------------------------------------------------------------
# Pipeline result sub-objects
# ---------------------------------------------------------------------------
class IntentMatch(BaseModel):
    """A single candidate intent and its score from the classifier."""

    intent: str
    score: float = Field(..., ge=0.0, le=1.0, description="Match score in [0, 1].")


class ClassificationResult(BaseModel):
    """Output of the Classifier module."""

    intent: str = Field(..., description="Best-guess intent.")
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Confidence of the chosen intent."
    )
    reasoning: str = Field(
        default="", description="Human-readable rationale for the classification."
    )
    top_matches: list[IntentMatch] = Field(
        default_factory=list, description="Ranked alternative intents with scores."
    )


class RoutingDecision(BaseModel):
    """Output of the Router module."""

    lane: RoutingLane = Field(..., description="Destination lane.")
    sensitivity: SensitivityLevel = Field(
        ..., description="Assessed content sensitivity."
    )
    reason: str = Field(default="", description="Why this routing decision was made.")
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Confidence in the routing decision."
    )


class PolicyCitation(BaseModel):
    """A citation to a policy / FAQ document grounding a draft."""

    policy_id: str = Field(..., description="Identifier of the cited policy document.")
    title: str = Field(..., description="Human-readable title of the policy.")
    snippet: str = Field(..., description="The exact text excerpt cited.")
    score: float | None = Field(
        default=None, ge=0.0, le=1.0, description="Relevance score, if available."
    )


class RetrievalContextItem(BaseModel):
    """A single retrieved passage used as grounding context for drafting."""

    policy_id: str = Field(..., description="Source policy document id.")
    title: str = Field(..., description="Source policy document title.")
    content: str = Field(..., description="Retrieved passage content.")
    score: float = Field(..., ge=0.0, le=1.0, description="Retrieval relevance score.")


class DraftResponse(BaseModel):
    """Output of the Drafter module — a grounded reply draft."""

    draft_body: str = Field(..., description="Generated reply text.")
    policy_citations: list[PolicyCitation] = Field(
        default_factory=list, description="Policies cited in the draft."
    )
    retrieval_context: list[RetrievalContextItem] = Field(
        default_factory=list,
        description="Passages retrieved as grounding for the draft.",
    )


class AuthorMention(BaseModel):
    """One person an email identifies. Every field is independently optional.

    Mirrors ``app.pipeline.extractor.AuthorMention``; see the note on
    :class:`ExtractionResult` for why this is a mirror and not a re-export.
    """

    name: str | None = Field(default=None, description="Display name, if given.")
    email: str | None = Field(default=None, description="Email address, if given.")
    affiliation: str | None = Field(
        default=None, description="Institution/affiliation, if given."
    )


class ExtractionResult(BaseModel):
    """Which submissions an email refers to, and who it names.

    Mirrors ``app.pipeline.extractor.ExtractionResult``, following the same
    local-mirror convention the other pipeline sub-objects in this module use
    (``ClassificationResult`` / ``DraftResponse``) rather than importing the
    pipeline types: these are the wire contract, the pipeline's are the internal
    one, and the two are free to move independently.

    The submission/forum identifier fields are LISTS — an email may legitimately
    name several submissions, and reporting one silently discarded the rest. The
    two OpenReview reply signals are SCALARS: each names one specific thing (a
    single comment; a single sender address), so a list of them would carry no
    usable meaning.

    ``method`` is part of the contract, not decoration: it says WHICH path
    produced these values, so a consumer can tell the model's answer from a
    weaker regex guess. It does NOT describe ``openreview_note_id``,
    ``openreview_notification_sender`` or ``extracted_reply_text``, all of which
    the pipeline reads from the raw text whichever path ran. Note the
    distinction a null ``extraction`` carries — absent means the row was never
    examined, whereas a present result with EMPTY lists means it was examined
    and nothing was found.

    ``openreview_reply_candidate`` is DERIVED here exactly as it is on the
    pipeline model — see its own note for why mirroring it as a stored field
    would have reintroduced, at the wire layer, the very drift it exists to
    prevent.

    ``extracted_reply_text`` is STORED on both sides, and that is not an
    inconsistency with the flag above. The flag is a function of two fields this
    model carries, so it can be recomputed here; the reply text is a function of
    the raw email body, which this model does not carry and should not, so there
    is nothing to recompute it from. Mirroring it as a plain field is the only
    option, not a weaker choice.

    Field ORDER matches the pipeline model deliberately, so the two can be read
    side by side; nothing asserts key order, so this is for humans.
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
        "carried one as its `noteId` parameter. Always names a comment inside a "
        "forum this result also reports — it is read from the SAME link as its "
        "forum id and never paired across links. None when no link carried one, "
        "which means 'looked, found none'.",
    )
    openreview_notification_sender: str | None = Field(
        default=None,
        description="The OpenReview per-venue notification address "
        "(<venue>-notifications@openreview.net) found in the text, verbatim; "
        "None when none appears. Sent as the matched ADDRESS rather than a bare "
        "boolean so a consumer can see WHICH venue and year the quoted "
        "notification came from. Case is preserved as found, so COMPARE "
        "CASE-INSENSITIVELY.",
    )
    extracted_reply_text: str = Field(
        default="",
        description="What the person actually wrote, with quoted reply history "
        "removed and the ends trimmed. Empty string when the body was entirely "
        "quoted material, or absent — TEST EMPTINESS ON THIS STRING rather than "
        "re-deriving it, since a body whose quote is preceded by a blank line "
        "still contains no reply. Always a string, never null, so a consumer "
        "never has to handle two kinds of empty.",
    )
    authors: list[AuthorMention] = Field(
        default_factory=list,
        description="People the email identifies, deduplicated, in first-seen order.",
    )
    method: Literal["llm_distiller", "regex_fallback", "none"] = Field(
        default="none",
        description="Which path produced the IDENTIFIER and AUTHOR fields. It "
        "does not describe `openreview_note_id` or "
        "`openreview_notification_sender`, which are read from the raw text on "
        "either path.",
    )
    is_reciprocal_dispute: bool | None = Field(
        default=None,
        description="True when the email disputes or concedes a desk rejection "
        "grounded in reciprocal-review duties, False when the model says it "
        "does not. TRI-STATE: None means the model did not answer, which is not "
        "the same as False — an unanswered flag is no evidence either way, "
        "while False is a positive 'ruled out'. Null for every row persisted "
        "before this field existed, so treat null as unknown, never as no.",
    )
    appeal_reason: list[str] | None = Field(
        default=None,
        description="The grounds an appeal argues, as full reason names, in a "
        "fixed order. TRI-STATE: null means not asked, skipped, or failed — "
        "never 'no reason applies'; [] means asked, and no listed reason "
        "applies; a non-empty list is the reasons argued. Null for every row "
        "persisted before this field existed.",
    )

    @computed_field
    @property
    def openreview_reply_candidate(self) -> bool:
        """Both OpenReview signals present: a reply to a notification.

        DERIVED, exactly as on the pipeline model — and that choice is the whole
        point rather than a stylistic echo. Mirrored as a stored
        ``bool = False`` field, this wire model could be handed a value
        contradicting the two fields beside it, and would then be able to
        represent a state the pipeline can NEVER produce. A mirror that can say
        something its source cannot has drifted, which is the exact failure this
        class's docstring warns about. Deriving keeps the two identical by
        construction instead of by anyone remembering.

        Practical consequence: a stale or corrupt persisted value is ignored and
        recomputed on validation, so BOTH models independently reach the same
        answer from the same operands and cannot disagree.

        The expression is duplicated rather than imported, following this
        module's mirror convention. That duplication is itself a drift surface,
        so it is covered by a test that compares the derived value across both
        models over the full truth table — not merely by inspection.
        """
        return (
            self.openreview_note_id is not None
            and self.openreview_notification_sender is not None
        )


# ---------------------------------------------------------------------------
# Persisted record
# ---------------------------------------------------------------------------
class EmailRecord(BaseModel):
    """Full lifecycle record for an email, including pipeline outputs."""

    model_config = ConfigDict(from_attributes=True)

    id: int = Field(..., description="Primary key.")
    sender: EmailStr
    sender_name: str | None = None
    subject: str
    body: str
    received_at: datetime
    status: EmailStatus = Field(default=EmailStatus.PENDING)

    classification: ClassificationResult | None = None
    routing: RoutingDecision | None = None
    draft: DraftResponse | None = None
    # None means the row was never examined (it predates extraction), which is
    # NOT the same as a present result whose lists are EMPTY (examined, found
    # none) — hence optional with no default factory.
    extraction: ExtractionResult | None = None

    created_at: datetime
    updated_at: datetime


# ---------------------------------------------------------------------------
# Human-review actions
# ---------------------------------------------------------------------------
class ApprovalAction(BaseModel):
    """A chair's action on a human-review email."""

    action: Literal["approve", "edit", "reroute"] = Field(
        ..., description="The action taken by the chair."
    )
    edited_body: str | None = Field(
        default=None, description="Edited reply body (required when action='edit')."
    )
    reroute_reason: str | None = Field(
        default=None,
        description="Reason for rerouting (required when action='reroute').",
    )
