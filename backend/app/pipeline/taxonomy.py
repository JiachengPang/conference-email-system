"""Intent taxonomy — the single source of truth for classification intents.

Data-mined from ~18.5k inbound tickets (blind LLM induction, chair-tag validated,
year-persistence filtered — see docs/local/CLASSIFICATION_REWORK.md §4). Replaces
the hand-crafted 11 intents. Every other module (distiller, keyword fallback,
router, template drafter, KB labeler) imports from here — do not hardcode intents
anywhere else.
"""

# (canonical_name, family, definition)
_TAXONOMY: list[tuple[str, str, str]] = [
    ("reviewer_assignment", "review_workflow",
     "Requests to add, remove, replace, validate, or locate reviewer or "
     "emergency-reviewer assignments for papers."),
    ("review_submission_help", "review_workflow",
     "Help submitting reviews or meta-reviews, or reports of late/missing reviews, "
     "reviewer delays, or review-system access problems including outages."),
    ("paper_bidding", "review_workflow",
     "Requests to access, reopen, extend, or correct the paper-bidding / "
     "reviewer-preference process."),
    ("author_profile_compliance", "submission_compliance",
     "Missing, invalid, or ambiguous Google Scholar / Semantic Scholar / DBLP "
     "profile IDs, or required user-info / conflict / subject-area completion."),
    ("submission_upload_help", "submission_compliance",
     "Help uploading, replacing, restoring, or correcting a paper, camera-ready, or "
     "supplementary file, or restoring an accidentally-withdrawn submission."),
    ("submission_requirements", "submission_compliance",
     "Questions about submission eligibility, deadlines, required steps, portal "
     "access, tracks, or next steps for accepted papers."),
    ("submission_format_policy", "submission_compliance",
     "Clarification of formatting / submission-policy rules: page limits, "
     "appendices, checklists, supplementary placement, anonymized code links."),
    ("author_list_change", "submission_compliance",
     "Requests to add, remove, reorder, or correct authors or submission metadata "
     "after submission."),
    ("review_decision_appeal", "appeals_integrity",
     "Concerns or appeals about review quality, rebuttal handling, mismatched "
     "reviews, scores, or the final decision."),
    ("desk_reject_appeal", "appeals_integrity",
     "Requests to explain, reconsider, reverse, or waive a desk rejection "
     "(formatting, page-limit, appendix, checklist, compliance, or "
     "reciprocal-review duty grounds)."),
    ("anonymity_violation", "appeals_integrity",
     "Reports that a submission may violate double-blind / anonymity rules via "
     "identifying information, public materials, or disclosures."),
    ("reviewer_workload_role", "committee",
     "Requests to adjust review workload, volunteer as a reviewer, or be considered "
     "for an elevated role (SPC / area chair)."),
    ("committee_invitation", "committee",
     "Responses to reviewer / PC / session-chair invitations: accept, decline, "
     "availability, or resend/reactivate an invitation link."),
    ("cms_support", "systems",
     "CMT / OpenReview account, email-linking, duplicate-account, site-access, or "
     "general workflow support not tied to a single submission action."),
]

VALID_INTENTS: list[str] = [name for name, _, _ in _TAXONOMY]
INTENT_FAMILIES: dict[str, str] = {name: fam for name, fam, _ in _TAXONOMY}
INTENT_DEFS: dict[str, str] = {name: d for name, _, d in _TAXONOMY}
FAMILIES: list[str] = list(dict.fromkeys(fam for _, fam, _ in _TAXONOMY))
FALLBACK_INTENT: str = "cms_support"


# The appeals against a rejection — the two intents where a requester is
# contesting an outcome rather than asking how something works. Single source of
# truth: anything that needs "is this a reject appeal?" derives it from here
# rather than re-listing the names.
REJECT_APPEAL_INTENTS: frozenset[str] = frozenset(
    {"desk_reject_appeal", "review_decision_appeal"}
)


def is_reject_appeal(intent: str | None) -> bool | None:
    """Whether ``intent`` is one of the reject-appeal intents.

    Returns ``None`` for a ``None`` intent — "not classified" is distinct from
    "classified as not an appeal", and the two must not collapse.
    """
    if intent is None:
        return None
    return intent in REJECT_APPEAL_INTENTS
