# Intent Taxonomy

The classification taxonomy the system assigns to every incoming email: **14 intents grouped into 5 families**.

**Source of truth:** `backend/app/pipeline/taxonomy.py` (`VALID_INTENTS`, `INTENT_FAMILIES`, `INTENT_DEFS`, `FAMILIES`, `FALLBACK_INTENT`). This document mirrors that module; if they ever disagree, the code wins. Every consumer — the distiller (production classifier), the keyword fallback, the router, the template drafter, and the KB intent labeler — imports from `taxonomy.py`; no intent strings are hardcoded elsewhere.

**Provenance:** data-mined from ~18.5k historical inbound tickets by blind LLM induction (the model was not shown the previous hand-crafted labels), validated against chair-applied ticket tags, and filtered for year-persistence so that categories recur across conference cycles rather than encoding one year's incidents. This replaced an earlier hand-crafted 11-intent set that mapped poorly to real inbox structure. Full methodology: `docs/local/CLASSIFICATION_REWORK.md` §4 (working doc).

**Fallback intent:** `cms_support` — used when classification is unknown or low-confidence.

---

## The 14 intents

### Family: `review_workflow` — running the review process
| Intent | Definition |
|---|---|
| `reviewer_assignment` | Requests to add, remove, replace, validate, or locate reviewer or emergency-reviewer assignments for papers. |
| `review_submission_help` | Help submitting reviews or meta-reviews, or reports of late/missing reviews, reviewer delays, or review-system access problems including outages. |
| `paper_bidding` | Requests to access, reopen, extend, or correct the paper-bidding / reviewer-preference process. |

### Family: `submission_compliance` — getting a submission correct and complete
| Intent | Definition |
|---|---|
| `author_profile_compliance` | Missing, invalid, or ambiguous Google Scholar / Semantic Scholar / DBLP profile IDs, or required user-info / conflict / subject-area completion. |
| `submission_upload_help` | Help uploading, replacing, restoring, or correcting a paper, camera-ready, or supplementary file, or restoring an accidentally-withdrawn submission. |
| `submission_requirements` | Questions about submission eligibility, deadlines, required steps, portal access, tracks, or next steps for accepted papers. |
| `submission_format_policy` | Clarification of formatting / submission-policy rules: page limits, appendices, checklists, supplementary placement, anonymized code links. |
| `author_list_change` | Requests to add, remove, reorder, or correct authors or submission metadata after submission. |

### Family: `appeals_integrity` — contested outcomes and integrity reports
| Intent | Definition |
|---|---|
| `review_decision_appeal` | Concerns or appeals about review quality, rebuttal handling, mismatched reviews, scores, or the final decision. |
| `desk_reject_appeal` | Requests to explain, reconsider, or reverse a desk rejection (formatting, page-limit, appendix, checklist, or compliance grounds). |
| `anonymity_violation` | Reports that a submission may violate double-blind / anonymity rules via identifying information, public materials, or disclosures. |

### Family: `committee` — reviewer/committee roles and invitations
| Intent | Definition |
|---|---|
| `reviewer_workload_role` | Requests to adjust review workload, volunteer as a reviewer, or be considered for an elevated role (SPC / area chair). |
| `committee_invitation` | Responses to reviewer / PC / session-chair invitations: accept, decline, availability, or resend/reactivate an invitation link. |

### Family: `systems` — accounts and platform support
| Intent | Definition |
|---|---|
| `cms_support` | CMT / OpenReview account, email-linking, duplicate-account, site-access, or general workflow support not tied to a single submission action. (Also the fallback intent.) |

---

## How the intent is used

- **Classification.** The distiller (one model call) assigns one of the 14 intents plus a confidence; on failure a keyword classifier falls back to the same 14, defaulting to `cms_support`.
- **Routing.** The lane router (FAQ auto-answer vs. human review) and the chair-assignment router key off the intent and its family. (The precise FAQ-eligibility rule is defined by the router, not this document, and is evolving — see the routing design docs/plans.)
- **Drafting.** The template drafter selects a per-intent opening line (with a generic default for any miss).
- **KB coverage map.** Each policy chunk carries an `intents` list (the intents it can answer). The intent→chunk coverage report (`backend/reports/kb_intent_coverage.json`) shows which intents the knowledge base can actually answer — a KB-authoring signal, since the corpus is submission-policy-scoped while the inbox skews toward reviewing operations.
