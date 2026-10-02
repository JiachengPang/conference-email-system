"""Tests for the paper/author extractor — both paths.

The LLM path drives ``extract`` through a constructed ``DistillResult``; the
regex path drives it with ``distilled=None``. No model call, no HTTP.

Both paths now report LISTS: every submission number and forum id an email
names, not just the first. Regex examples are synthetic, modelled on shapes
found in the real corpus.
"""

from app.pipeline.distiller import DistillResult
from app.pipeline.extractor import (
    AuthorMention,
    EmailExtractor,
    ExtractionResult,
    _dedupe_authors,
    _find_openreview_note_pairs,
    _find_openreview_notification_sender,
    _first_note_id_for,
    _parse_author,
)

_SUBJECT = "Re: Regarding the Desk Rejection of Your AAAI-2026 Submission 12345"
_BODY = "Dear Chairs, we are writing about our submission."
_SENDER = "jane@example.edu"
_SENDER_NAME = "Jane Roe"


def _distilled(**kwargs) -> DistillResult:
    """A DistillResult with the pipeline-critical fields already filled in."""
    return DistillResult(
        queries=["desk rejection appeal procedure"],
        intent="desk_reject_appeal",
        confidence=0.9,
        **kwargs,
    )


def _extract(distilled: DistillResult | None) -> ExtractionResult:
    return EmailExtractor().extract(
        _SUBJECT, _BODY, _SENDER, _SENDER_NAME, distilled
    )


# ---------------------------------------------------------------------------
# Full and partial data
# ---------------------------------------------------------------------------
def test_extract_full_data():
    result = _extract(
        _distilled(
            submission_numbers_raw=["12345"],
            openreview_ids_raw=["Ab3xY9kLm2"],
            authors_raw=[
                "Jane Roe | jane@example.edu | Example University",
                "John Doe | john@example.org | Other Institute",
            ],
        )
    )
    assert result.method == "llm_distiller"
    assert result.submission_numbers == ["12345"]
    assert result.openreview_forum_ids == ["Ab3xY9kLm2"]
    assert result.authors == [
        AuthorMention(
            name="Jane Roe",
            email="jane@example.edu",
            affiliation="Example University",
        ),
        AuthorMention(
            name="John Doe",
            email="john@example.org",
            affiliation="Other Institute",
        ),
    ]


def test_extract_partial_author_fields_kept_as_partial_mention():
    """A NONE piece nulls only that field — it never drops the whole entry."""
    result = _extract(
        _distilled(authors_raw=["Jane Roe | NONE | Example University"])
    )
    assert result.authors == [
        AuthorMention(
            name="Jane Roe", email=None, affiliation="Example University"
        )
    ]


def test_extract_none_sentinel_is_case_insensitive_per_field():
    result = _extract(_distilled(authors_raw=["Jane Roe | none | None"]))
    assert result.authors == [
        AuthorMention(name="Jane Roe", email=None, affiliation=None)
    ]


def test_extract_submission_number_and_forum_id_pass_through_independently():
    """Either identifier may be present without the other."""
    only_number = _extract(_distilled(submission_numbers_raw=["12345"]))
    assert only_number.submission_numbers == ["12345"]
    assert only_number.openreview_forum_ids == []

    only_forum = _extract(_distilled(openreview_ids_raw=["Ab3xY9kLm2"]))
    assert only_forum.submission_numbers == []
    assert only_forum.openreview_forum_ids == ["Ab3xY9kLm2"]


def test_extract_drops_a_model_submission_number_that_is_not_a_number():
    """Submission numbers key (A)PC routing and the appeal report, so a value
    that is not a paper number must not reach them. Forum ids still pass
    through verbatim."""
    result = _extract(
        _distilled(
            submission_numbers_raw=["AAAI-2026"], openreview_ids_raw=["short"]
        )
    )
    assert result.submission_numbers == []
    assert result.openreview_forum_ids == ["short"]


def test_extract_strips_the_label_a_model_wraps_around_a_submission_number():
    for raw in ["615", "#615", "Submission 615", "submission #615", "Paper ID: 615",
                "paper no. 615", "615.", " 615 "]:
        result = _extract(_distilled(submission_numbers_raw=[raw]))
        assert result.submission_numbers == ["615"], raw


def test_extract_accepts_any_length_model_submission_number():
    result = _extract(_distilled(submission_numbers_raw=["7", "42", "123456", "1234567"]))
    assert result.submission_numbers == ["7", "42", "123456", "1234567"]


def test_extract_rejects_unusable_model_submission_numbers():
    for raw in ["0615", "12345678", "615-620", "615, 620", "about 615",
                "AAAI-27", "615a", "N/A", "unknown"]:
        result = _extract(_distilled(submission_numbers_raw=[raw]))
        assert result.submission_numbers == [], raw


def test_extract_dedupes_labelled_and_bare_forms_of_one_number():
    result = _extract(_distilled(submission_numbers_raw=["#615", "615", "Submission 9"]))
    assert result.submission_numbers == ["615", "9"]


# ---------------------------------------------------------------------------
# Multiple identifiers — the point of the list shape
# ---------------------------------------------------------------------------
def test_extract_multiple_submission_numbers():
    """An appeal covering two desk rejections names both; both are reported."""
    result = _extract(_distilled(submission_numbers_raw=["11111", "22222"]))
    assert result.submission_numbers == ["11111", "22222"]


def test_extract_multiple_openreview_forum_ids():
    result = _extract(
        _distilled(openreview_ids_raw=["Ab3xY9kLm2", "Zz9QwErTy1"])
    )
    assert result.openreview_forum_ids == ["Ab3xY9kLm2", "Zz9QwErTy1"]


def test_extract_multiple_of_both_kinds_at_once():
    result = _extract(
        _distilled(
            submission_numbers_raw=["11111", "22222", "33333"],
            openreview_ids_raw=["Ab3xY9kLm2", "Zz9QwErTy1"],
        )
    )
    assert result.submission_numbers == ["11111", "22222", "33333"]
    assert result.openreview_forum_ids == ["Ab3xY9kLm2", "Zz9QwErTy1"]


def test_extract_preserves_the_order_the_model_reported():
    """Order is the model's; re-sorting would discard which it named first."""
    result = _extract(_distilled(submission_numbers_raw=["99999", "11111"]))
    assert result.submission_numbers == ["99999", "11111"]


def test_extract_dedupes_exact_duplicate_identifiers():
    """The distiller forwards duplicates verbatim; dedup happens here."""
    result = _extract(
        _distilled(
            submission_numbers_raw=["12345", "12345", "67890"],
            openreview_ids_raw=["Ab3xY9kLm2", "Ab3xY9kLm2"],
        )
    )
    assert result.submission_numbers == ["12345", "67890"]
    assert result.openreview_forum_ids == ["Ab3xY9kLm2"]


def test_extract_dedupe_keeps_the_first_occurrence():
    result = _extract(_distilled(submission_numbers_raw=["11111", "22222", "11111"]))
    assert result.submission_numbers == ["11111", "22222"]


def test_extract_forum_id_dedupe_is_case_SENSITIVE():
    """Casefolding would be WRONG here — forum ids are case-sensitive tokens,
    so two ids differing only in case are two different papers."""
    result = _extract(
        _distilled(openreview_ids_raw=["Ab3xY9kLm2", "ab3xy9klm2"])
    )
    assert result.openreview_forum_ids == ["Ab3xY9kLm2", "ab3xy9klm2"]


def test_extract_strips_blank_identifier_entries():
    """Defensive: the distiller filters blanks, but model shape is not trusted.

    An unfiltered blank would otherwise render as an empty row in the panel.
    """
    result = _extract(
        _distilled(
            submission_numbers_raw=["", "  ", "12345"],
            openreview_ids_raw=["   ", "Ab3xY9kLm2"],
        )
    )
    assert result.submission_numbers == ["12345"]
    assert result.openreview_forum_ids == ["Ab3xY9kLm2"]


def test_extract_trims_surrounding_whitespace_on_identifiers():
    result = _extract(_distilled(submission_numbers_raw=["  12345  "]))
    assert result.submission_numbers == ["12345"]


def test_extract_all_blank_identifiers_collapse_to_empty_lists():
    result = _extract(
        _distilled(submission_numbers_raw=["", " "], openreview_ids_raw=[""])
    )
    assert result.submission_numbers == []
    assert result.openreview_forum_ids == []
    # Still the model's answer, not a reason to fall through to regex.
    assert result.method == "llm_distiller"


# ---------------------------------------------------------------------------
# Malformed author strings — best-effort parse, never skipped
# ---------------------------------------------------------------------------
def test_extract_author_missing_separators_parsed_as_name_only():
    """A bare name means the same thing as `John Doe | NONE | NONE`."""
    result = _extract(_distilled(authors_raw=["John Doe"]))
    assert result.authors == [
        AuthorMention(name="John Doe", email=None, affiliation=None)
    ]


def test_extract_author_with_one_separator_fills_name_and_email():
    result = _extract(_distilled(authors_raw=["Ann Poe | ann@example.edu"]))
    assert result.authors == [
        AuthorMention(name="Ann Poe", email="ann@example.edu", affiliation=None)
    ]


def test_extract_author_with_extra_separators_rejoins_affiliation():
    """A surplus separator is salvaged into the affiliation, not truncated."""
    result = _extract(
        _distilled(authors_raw=["Jane Roe | jane@example.edu | Dept | Example U"])
    )
    assert result.authors == [
        AuthorMention(
            name="Jane Roe",
            email="jane@example.edu",
            affiliation="Dept | Example U",
        )
    ]


def test_extract_author_with_no_populated_field_is_dropped():
    """`| |` and `NONE | NONE | NONE` identify nobody, so they are not mentions."""
    result = _extract(_distilled(authors_raw=["| |", "NONE | NONE | NONE"]))
    assert result.authors == []


def test_extract_malformed_author_does_not_discard_its_siblings():
    """One bad entry must not cost the good ones."""
    result = _extract(
        _distilled(
            authors_raw=[
                "Jane Roe | jane@example.edu | Example University",
                "| |",
                "John Doe",
            ]
        )
    )
    assert [a.name for a in result.authors] == ["Jane Roe", "John Doe"]


# ---------------------------------------------------------------------------
# Dedupe
# ---------------------------------------------------------------------------
def test_extract_dedupes_authors_case_insensitively():
    result = _extract(
        _distilled(
            authors_raw=[
                "Jane Roe | jane@example.edu | Example University",
                "JANE ROE | JANE@EXAMPLE.EDU | Example University",
            ]
        )
    )
    assert len(result.authors) == 1
    assert result.authors[0].name == "Jane Roe"  # first-seen wins


def test_extract_dedupe_keeps_first_seen_order():
    result = _extract(
        _distilled(
            authors_raw=[
                "Jane Roe | jane@example.edu | NONE",
                "John Doe | john@example.org | NONE",
                "jane roe | jane@example.edu | NONE",
            ]
        )
    )
    assert [a.name for a in result.authors] == ["Jane Roe", "John Doe"]


def test_extract_dedupe_ignores_affiliation_differences():
    """Same person, one entry with an institution and one without, is one person."""
    result = _extract(
        _distilled(
            authors_raw=[
                "Jane Roe | jane@example.edu | Example University",
                "Jane Roe | jane@example.edu | NONE",
            ]
        )
    )
    assert len(result.authors) == 1
    assert result.authors[0].affiliation == "Example University"


def test_extract_dedupe_keeps_distinct_people_apart():
    """Same affiliation must not collapse two different people."""
    result = _extract(
        _distilled(
            authors_raw=[
                "Jane Roe | jane@example.edu | Example University",
                "John Doe | john@example.org | Example University",
            ]
        )
    )
    assert len(result.authors) == 2


def test_extract_dedupe_uses_casefold_for_non_ascii_names():
    """Real traffic carries non-ASCII names; `lower()` under-folds some of them."""
    assert len(_dedupe_authors([
        AuthorMention(name="STRASSE", email=None, affiliation=None),
        AuthorMention(name="strasse", email=None, affiliation=None),
    ])) == 1
    # German sharp-s: casefold maps "ß" to "ss", lower() does not.
    assert len(_dedupe_authors([
        AuthorMention(name="STRASSE", email=None, affiliation=None),
        AuthorMention(name="straße", email=None, affiliation=None),
    ])) == 1


def test_extract_same_name_different_email_are_distinct_mentions():
    """The key is (name, email) — a differing email is a differing mention."""
    result = _extract(
        _distilled(
            authors_raw=[
                "Jane Roe | jane@example.edu | NONE",
                "Jane Roe | jane.roe@other.org | NONE",
            ]
        )
    )
    assert len(result.authors) == 2


# ---------------------------------------------------------------------------
# Empty / absent input
# ---------------------------------------------------------------------------
def test_extract_empty_authors_raw():
    result = _extract(_distilled(submission_numbers_raw=["12345"]))
    assert result.authors == []
    assert result.method == "llm_distiller"


def test_extract_distilled_with_all_fields_empty_is_still_llm_distiller():
    """Point-3 contract: the model ran and found nothing.

    That is an ANSWER, not a failure — the prompt directs it to read both
    subject and body — so it must not be recorded as `none` (nothing looked)
    and must not fall through to the regex path. Restated for the list shape:
    ALL THREE lists empty is still the model's answer.
    """
    result = _extract(_distilled())
    assert result.method == "llm_distiller"
    assert result.submission_numbers == []
    assert result.openreview_forum_ids == []
    assert result.authors == []


def test_extract_without_distilled_routes_to_the_regex_fallback():
    """The absence of a distiller result is what selects the regex path.

    (Replaces the 2a stub assertion, which pinned `method="none"` only because
    the fallback did not exist yet.)
    """
    result = _extract(None)
    assert result.method == "regex_fallback"
    # `_SUBJECT` carries a cue-worded number, so the fallback really ran.
    # Single-element for now: the regex path still finds at most one of each
    # until it is widened in the next piece.
    assert result.submission_numbers == ["12345"]


def test_extract_with_distilled_never_consults_the_regex_fallback():
    """Locked decision: regex is a fallback, never a supplement.

    The subject here carries a number the regex would happily find; a present
    distiller result must suppress it entirely rather than top itself up.
    """
    result = _extract(_distilled())
    assert result.method == "llm_distiller"
    assert result.submission_numbers == []
    assert result.openreview_forum_ids == []
    assert result.authors == []


# ---------------------------------------------------------------------------
# Failure policy
# ---------------------------------------------------------------------------
def test_extract_never_raises_on_bad_input():
    """A normalization bug degrades to an empty result, never an exception.

    Identification is an enrichment; it must never break the drafting pipeline
    that calls it.
    """

    class _Exploding:
        queries = ["q"]
        submission_numbers_raw = ["12345"]
        openreview_ids_raw = []

        @property
        def authors_raw(self):
            raise RuntimeError("boom")

    result = EmailExtractor().extract(
        _SUBJECT, _BODY, _SENDER, _SENDER_NAME, _Exploding()
    )
    assert result.method == "none"
    assert result.authors == []


# ---------------------------------------------------------------------------
# Model defaults
# ---------------------------------------------------------------------------
def test_extraction_result_defaults_are_empty():
    result = ExtractionResult()
    assert result.submission_numbers == []
    assert result.openreview_forum_ids == []
    assert result.authors == []
    assert result.method == "none"


def test_author_mention_defaults_are_empty():
    mention = AuthorMention()
    assert mention.name is None
    assert mention.email is None
    assert mention.affiliation is None


def test_parse_author_returns_none_for_empty_entry():
    assert _parse_author("NONE | NONE | NONE") is None
    assert _parse_author("  ") is None


# ---------------------------------------------------------------------------
# Regex fallback (subtask 2b) — reached ONLY when distilled is None
#
# SCOPE LIMIT: every example below is synthetic. Real ticket text is PII and
# never enters this suite; the shapes are modelled on the discovery findings
# (cue-word gating, the AAAI-designator and year false positives, the
# /group?id= false positive) without reproducing any real ticket's content.
# ---------------------------------------------------------------------------
def _regex_extract(
    subject: str = "", body: str = "", sender: str = "", sender_name: str | None = None
) -> ExtractionResult:
    """Drive the fallback by passing distilled=None."""
    return EmailExtractor().extract(subject, body, sender, sender_name, None)


# --- submission number -----------------------------------------------------
def test_regex_submission_number_with_cue_word():
    result = _regex_extract(body="We are writing about submission 12345.")
    assert result.submission_numbers == ["12345"]
    assert result.method == "regex_fallback"


def test_regex_submission_number_accepts_common_cue_shapes():
    for text, expected in [
        ("Paper ID: 26681", "26681"),
        ("Paper ID 26681", "26681"),
        ("paper 9904", "9904"),
        ("submission #10824", "10824"),
        ("Submission ID:7377", "7377"),
        ("Paper number: 29546", "29546"),
        ("our submissions 13276 and others", "13276"),
        ("regarding #18898", "18898"),
    ]:
        assert _regex_extract(body=text).submission_numbers == [expected], text


def test_regex_submission_number_requires_a_cue_word():
    """A bare number with no cue is not a submission number."""
    assert _regex_extract(body="We waited 12345 seconds.").submission_numbers == []


def test_regex_submission_number_rejects_conference_designator_year():
    """`AAAI-26` / `AAAI 2026` is the conference, never the submission."""
    for text in [
        "our submission AAAI 2026 was rejected",
        "our submission AAAI-2026 was rejected",
        "the paper IAAI 2027 track",
    ]:
        assert _regex_extract(body=text).submission_numbers == [], text


def test_regex_submission_number_rejects_any_number_directly_after_designator():
    """A number adjacent to the designator is distrusted whatever its value.

    Deliberately covers a 5-digit number, which the year rule ignores — so this
    isolates the designator guard. In real traffic a designator is followed by
    its year, never by the submission number (that follows a cue like
    `Submission` / `Paper ID`), so the adjacency itself is the signal.
    """
    assert _regex_extract(body="our submission AAAI 12345 here").submission_numbers == []
    assert _regex_extract(body="the paper EAAI-31337 track").submission_numbers == []


def test_regex_submission_number_keeps_scanning_past_a_rejected_match():
    """A rejected candidate must not abort the search for a real one."""
    result = _regex_extract(body="our submission AAAI 12345 concerns paper 22336")
    assert result.submission_numbers == ["22336"]


def test_regex_submission_number_rejects_year_reached_across_a_word():
    """`paper due 2026` is a deadline; the cue does not introduce the number."""
    for text in ["the paper due 2026", "our submission deadline 2026"]:
        assert _regex_extract(body=text).submission_numbers == [], text


def test_regex_submission_number_accepts_year_like_value_adjacent_to_cue():
    """Genuine 4-digit numbers in the year range exist; a direct cue believes them."""
    assert _regex_extract(body="Submission 2026 was desk rejected.").submission_numbers == ["2026"]


def test_regex_submission_number_conference_year_not_taken_from_surrounding_text():
    """The designator year must lose to the real number later in the line."""
    result = _regex_extract(subject="Update on your AAAI 2026 Submission 22336")
    assert result.submission_numbers == ["22336"]


def test_regex_submission_number_found_in_subject_only():
    result = _regex_extract(
        subject="Re: Desk Rejection of Your Submission 15357",
        body="Dear chairs, please reconsider.",
    )
    assert result.submission_numbers == ["15357"]


def test_regex_submission_number_found_in_body_only():
    result = _regex_extract(
        subject="Appeal request", body="This concerns submission 15357."
    )
    assert result.submission_numbers == ["15357"]


# NOTE: `test_regex_submission_number_prefers_subject_over_body` lived here and
# is GONE, not adapted. It pinned that a subject number BEAT a body number —
# a claim that no longer exists now that both are reported. Its exact scenario
# is covered by `test_regex_collects_from_subject_AND_body_subject_first`
# below, which asserts the surviving property: subject-first ORDER.


# --- collecting EVERY match, not just the first -----------------------------
def test_regex_collects_two_numbers_each_with_its_own_cue():
    """Repeating the cue makes both numbers independently valid — both returned."""
    result = _regex_extract(body="See paper 11111 and paper 22222 for context.")
    assert result.submission_numbers == ["11111", "22222"]


def test_regex_collects_two_hash_numbers():
    """The bare `#` is its own cue, so each item in the list qualifies."""
    result = _regex_extract(body="Please reassign submissions #5458 and #21675.")
    assert result.submission_numbers == ["5458", "21675"]


def test_regex_collects_numbers_across_separate_sentences():
    result = _regex_extract(body="This is submission 11111. Also submission 22222.")
    assert result.submission_numbers == ["11111", "22222"]


def test_regex_collects_from_subject_AND_body_subject_first():
    """Replaces the old subject-precedence tie-break: both are reported now.

    There is no longer a winner to choose, so the subject/body disagreement the
    quoted-notification exception existed to arbitrate simply does not arise —
    which is why that exception (and its tests) are gone rather than adapted.
    """
    result = _regex_extract(
        subject="Re: Your Submission 11111", body="Please also see paper 22222."
    )
    assert result.submission_numbers == ["11111", "22222"]


def test_regex_dedupes_the_same_number_in_subject_and_body():
    result = _regex_extract(
        subject="Re: Your Submission 11111",
        body="I am writing about submission 11111 to appeal.",
    )
    assert result.submission_numbers == ["11111"]


def test_regex_keeps_rejecting_bad_candidates_alongside_good_ones():
    """Collecting all must not smuggle in candidates the guards reject."""
    result = _regex_extract(
        body="our submission AAAI 12345 concerns paper 22336 and paper due 2026"
    )
    # 12345 sits after a conference designator; 2026 is a year across a filler.
    assert result.submission_numbers == ["22336"]


def test_regex_collects_two_forum_ids_across_subject_and_body():
    result = _regex_extract(
        subject="Re: openreview.net/forum?id=Ab3xY9kLm2",
        body="and the other is https://openreview.net/pdf?id=Zz9QwErTy1",
    )
    assert result.openreview_forum_ids == ["Ab3xY9kLm2", "Zz9QwErTy1"]


def test_regex_dedupes_a_forum_id_repeated_in_the_body():
    result = _regex_extract(
        body=(
            "See https://openreview.net/forum?id=Ab3xY9kLm2 — again, "
            "https://openreview.net/forum?id=Ab3xY9kLm2"
        )
    )
    assert result.openreview_forum_ids == ["Ab3xY9kLm2"]


def test_regex_forum_id_collection_still_rejects_group_links():
    """A group link beside a real forum link must not be collected."""
    result = _regex_extract(
        body=(
            "Roster: https://openreview.net/group?id=AAAI.org/2026/Conference "
            "Paper: https://openreview.net/forum?id=Ab3xY9kLm2"
        )
    )
    assert result.openreview_forum_ids == ["Ab3xY9kLm2"]


def test_regex_collects_several_of_both_kinds_together():
    result = _regex_extract(
        subject="Re: Submissions #11111 and #22222",
        body=(
            "Forums: https://openreview.net/forum?id=Ab3xY9kLm2 and "
            "https://openreview.net/pdf?id=Zz9QwErTy1"
        ),
    )
    assert result.submission_numbers == ["11111", "22222"]
    assert result.openreview_forum_ids == ["Ab3xY9kLm2", "Zz9QwErTy1"]
    assert result.method == "regex_fallback"


def test_regex_known_limitation_bare_conjunction_second_item_still_missed():
    """KNOWN LIMITATION, pinned deliberately — not an assertion that this is right.

    "papers 11111 and 22222" yields only 11111. Collecting every match does NOT
    fix this: the second number is never a CANDIDATE, because the cue-word gate
    requires a cue adjacent to each number and "and 22222" carries none. That is
    an acceptance rule, not an early return, and acceptance was deliberately
    left untouched here.

    Measured on the real corpus: of the trailing items in "X and Y" / "X, Y"
    shapes, 76 are now captured (their second item carried its own cue, usually
    "#") and 62 remain missed like this one. Widening the cue gate to reach
    across a conjunction is a separate decision with its own false-positive
    risk.
    """
    assert _regex_extract(body="papers 11111 and 22222").submission_numbers == ["11111"]
    assert _regex_extract(body="Paper IDs 3157, 17066").submission_numbers == ["3157"]


def test_regex_submission_number_accepts_any_length_after_a_direct_cue():
    """Papers are numbered from 1 upward, so 1 to 7 digits are all real."""
    for text, expected in [
        ("Re: Desk rejection of Submission 7", ["7"]),
        ("our paper #42 was rejected", ["42"]),
        ("Paper ID: 615 was rejected", ["615"]),
        ("Submission123 decision", ["123"]),
        ("paper 123456 was rejected", ["123456"]),
        ("submission 1234567 was rejected", ["1234567"]),
    ]:
        assert _regex_extract(body=text).submission_numbers == expected, text


def test_regex_submission_number_rejects_leading_zero_and_over_long():
    """Paper numbers are never zero-padded; eight or more digits are not papers."""
    for text in ["paper 0123", "paper 007", "paper 0", "paper 12345678"]:
        assert _regex_extract(body=text).submission_numbers == [], text


def test_regex_short_number_needs_the_cue_to_introduce_it_directly():
    """A filler word before a short number is noise in real traffic."""
    for text in [
        "the submission deadline 15 August",
        "a paper with 2 reviews",
        "my paper is 3 days late",
        "submission has 4 issues",
    ]:
        assert _regex_extract(body=text).submission_numbers == [], text


def test_regex_long_number_still_crosses_a_filler_word():
    """The direct-cue rule is for short numbers only; 4+ digits keep the old rule."""
    assert _regex_extract(body="our paper titled 12345 was rejected").submission_numbers == ["12345"]


def test_regex_short_number_rejected_when_it_reads_as_a_count():
    for text in [
        "paper 3 reviewers disagreed",
        "submission 8 pages long",
        "papers 2 authors only",
        "submission 30 minutes ago",
        "paper 50% done",
        "submission 2nd round",
    ]:
        assert _regex_extract(body=text).submission_numbers == [], text


def test_regex_short_number_kept_before_ordinary_words():
    """Only count nouns reject; a singular "reviewer" may follow a real id."""
    for text, expected in [
        ("paper 12 reviewer 2 misread the method", ["12"]),
        ("Submission 9 help needed", ["9"]),
        ("Paper 615.", ["615"]),
    ]:
        assert _regex_extract(body=text).submission_numbers == expected, text


def test_regex_bare_hash_short_number_is_not_a_submission():
    """Bare short "#N" is overwhelmingly "reviewer #2", "question #3" in real traffic."""
    for text in ["Reviewer #2 misread the paper", "see question #3", "#12 below"]:
        assert _regex_extract(body=text).submission_numbers == [], text


def test_regex_bare_hash_accepts_four_to_seven_digits():
    assert _regex_extract(body="regarding #123456 please").submission_numbers == ["123456"]


def test_regex_colour_code_is_not_a_submission():
    """Digit-only colours share the bare-hash shape."""
    for text in ["color #333333", "background: #1234567", "colour #4455"]:
        assert _regex_extract(body=text).submission_numbers == [], text


# --- OpenReview forum id ---------------------------------------------------
def test_regex_forum_id_from_forum_link():
    result = _regex_extract(body="See https://openreview.net/forum?id=Ab3xY9kLm2 please.")
    assert result.openreview_forum_ids == ["Ab3xY9kLm2"]
    assert result.method == "regex_fallback"


def test_regex_forum_id_from_pdf_link():
    result = _regex_extract(body="PDF: https://openreview.net/pdf?id=Zz9QwErTy1")
    assert result.openreview_forum_ids == ["Zz9QwErTy1"]


def test_regex_forum_id_rejects_group_link():
    """AAAI committee/group URLs share the ?id= shape — confirmed false positive."""
    result = _regex_extract(
        body="Join https://openreview.net/group?id=AAAI.org/2026/Conference today."
    )
    assert result.openreview_forum_ids == []


def test_regex_forum_id_rejects_bare_id_parameter():
    """A ?id= on any other host or path is not a forum id."""
    for text in [
        "https://example.com/thing?id=Ab3xY9kLm2",
        "https://openreview.net/profile?id=Ab3xY9kLm2",
        "see ?id=Ab3xY9kLm2",
    ]:
        assert _regex_extract(body=text).openreview_forum_ids == [], text


def test_regex_forum_id_rejects_wrong_length():
    """Exactly 10 chars — a longer id must not be truncated into a false match."""
    assert _regex_extract(
        body="https://openreview.net/forum?id=Ab3xY9kLm2Extra24Chars"
    ).openreview_forum_ids == []
    assert _regex_extract(
        body="https://openreview.net/forum?id=Short1"
    ).openreview_forum_ids == []


def test_regex_forum_id_found_in_subject():
    result = _regex_extract(subject="Re: openreview.net/forum?id=Ab3xY9kLm2")
    assert result.openreview_forum_ids == ["Ab3xY9kLm2"]


def test_regex_forum_id_preserves_case():
    """The id is opaque and case-sensitive; matching must not normalize it."""
    assert (
        _regex_extract(body="openreview.net/forum?id=aB3Xy9KlM2").openreview_forum_ids
        == ["aB3Xy9KlM2"]
    )


# --- OpenReview note id (Official Comment) ---------------------------------
# The note id must come from the SAME link as its forum id. Everything here
# either proves that pairing holds or proves the field stays None; the
# `openreview_forum_ids` assertions are repeated throughout on purpose, because
# this change is required to be strictly additive to that field.
_REAL_SHAPE = "https://openreview.net/forum?id=ll0avn6ylq&noteId=jnHgRMHgrm"


def test_regex_note_id_from_forum_link_id_before_note_id():
    """The ordering OpenReview's own notification links use."""
    result = _regex_extract(body=f"Reply here: {_REAL_SHAPE}")
    assert result.openreview_note_id == "jnHgRMHgrm"
    assert result.openreview_forum_ids == ["ll0avn6ylq"]
    assert result.method == "regex_fallback"


def test_regex_note_id_from_pdf_link():
    result = _regex_extract(
        body="https://openreview.net/pdf?id=Ab3xY9kLm2&noteId=Zz9QwErTy1"
    )
    assert result.openreview_note_id == "Zz9QwErTy1"
    assert result.openreview_forum_ids == ["Ab3xY9kLm2"]


def test_regex_note_id_found_in_subject():
    result = _regex_extract(subject=f"Re: {_REAL_SHAPE}")
    assert result.openreview_note_id == "jnHgRMHgrm"


def test_regex_note_id_survives_html_escaped_ampersand():
    """A crudely de-HTML-ed body can leave `&amp;` between the parameters."""
    result = _regex_extract(
        body="https://openreview.net/forum?id=ll0avn6ylq&amp;noteId=jnHgRMHgrm"
    )
    assert result.openreview_note_id == "jnHgRMHgrm"
    assert result.openreview_forum_ids == ["ll0avn6ylq"]


def test_regex_note_id_preserves_case():
    """Opaque, case-sensitive token — IGNORECASE matching must not fold it."""
    result = _regex_extract(
        body="https://openreview.net/forum?id=ll0avn6ylq&noteId=jNhGrmHGRM"
    )
    assert result.openreview_note_id == "jNhGrmHGRM"


def test_regex_note_id_tolerates_trailing_sentence_punctuation():
    for text in [f"See {_REAL_SHAPE}.", f"See ({_REAL_SHAPE})", f"<{_REAL_SHAPE}>"]:
        assert _regex_extract(body=text).openreview_note_id == "jnHgRMHgrm", text


def test_regex_note_id_ignores_unrelated_query_parameters():
    result = _regex_extract(
        body="https://openreview.net/forum?id=ll0avn6ylq&referrer=x&noteId=jnHgRMHgrm&t=1"
    )
    assert result.openreview_note_id == "jnHgRMHgrm"
    assert result.openreview_forum_ids == ["ll0avn6ylq"]


# --- the field stays None ---------------------------------------------------
def test_regex_forum_link_without_note_id_leaves_note_id_none():
    """The forum id is reported exactly as before; only the scalar is absent."""
    result = _regex_extract(body="See https://openreview.net/forum?id=Ab3xY9kLm2 please.")
    assert result.openreview_note_id is None
    assert result.openreview_forum_ids == ["Ab3xY9kLm2"]
    assert result.method == "regex_fallback"


def test_regex_no_openreview_link_at_all_leaves_both_empty():
    result = _regex_extract(
        subject="Re: Your Submission 22336",
        body="Could you clarify the page limit?",
        sender=_SENDER,
        sender_name=_SENDER_NAME,
    )
    assert result.openreview_note_id is None
    assert result.openreview_forum_ids == []
    assert result.submission_numbers == ["22336"]
    assert result.method == "regex_fallback"


def test_regex_group_link_with_note_id_is_still_rejected():
    """`/group?id=` is a confirmed false positive — a noteId does not redeem it."""
    result = _regex_extract(
        body="https://openreview.net/group?id=AAAI.org/2026&noteId=jnHgRMHgrm"
    )
    assert result.openreview_note_id is None
    assert result.openreview_forum_ids == []


def test_regex_note_id_rejects_over_long_token():
    """Reject rather than truncate — the same rule the forum id follows."""
    result = _regex_extract(
        body="https://openreview.net/forum?id=ll0avn6ylq&noteId=" + "a" * 33
    )
    assert result.openreview_note_id is None
    assert result.openreview_forum_ids == ["ll0avn6ylq"]


def test_regex_note_id_alone_on_a_link_yields_nothing():
    """No forum id on the link, so there is nothing to attach the note to."""
    result = _regex_extract(body="https://openreview.net/forum?noteId=jnHgRMHgrm")
    assert result.openreview_note_id is None
    assert result.openreview_forum_ids == []


# --- pairing is per-link, never across links --------------------------------
def test_regex_note_id_is_never_paired_across_two_different_links():
    """THE safety property: a forum id from one link and a noteId from another
    would name a comment that does not exist in that forum, and would be
    indistinguishable from a real pair once stored."""
    result = _regex_extract(
        body=(
            "Paper: https://openreview.net/forum?id=Ab3xY9kLm2 "
            "Comment: https://openreview.net/forum?noteId=jnHgRMHgrm"
        )
    )
    assert result.openreview_note_id is None
    assert result.openreview_forum_ids == ["Ab3xY9kLm2"]


def test_regex_note_id_pairs_with_its_own_link_not_a_neighbouring_one():
    """Two complete links: the reported note must belong to the FIRST forum."""
    result = _regex_extract(
        body=(
            "https://openreview.net/forum?id=Ab3xY9kLm2&noteId=note111111 and "
            "https://openreview.net/forum?id=Zz9QwErTy1&noteId=note222222"
        )
    )
    assert result.openreview_note_id == "note111111"
    assert result.openreview_forum_ids == ["Ab3xY9kLm2", "Zz9QwErTy1"]


def test_find_note_pairs_returns_both_values_from_one_query():
    assert _find_openreview_note_pairs("", _REAL_SHAPE) == [
        ("ll0avn6ylq", "jnHgRMHgrm")
    ]


def test_find_note_pairs_reads_note_id_before_id_ordering():
    """The pair SCAN is order-agnostic even though the coherence gate currently
    withholds this shape — see the suppression test below."""
    assert _find_openreview_note_pairs(
        "", "https://openreview.net/forum?noteId=jnHgRMHgrm&id=ll0avn6ylq"
    ) == [("ll0avn6ylq", "jnHgRMHgrm")]


def test_find_note_pairs_skips_links_missing_either_parameter():
    assert (
        _find_openreview_note_pairs(
            "",
            "https://openreview.net/forum?id=Ab3xY9kLm2 "
            "https://openreview.net/forum?noteId=jnHgRMHgrm",
        )
        == []
    )


def test_first_note_id_for_withholds_a_note_whose_forum_is_unreported():
    pairs = [("Ab3xY9kLm2", "note111111")]
    assert _first_note_id_for(pairs, ["Ab3xY9kLm2"]) == "note111111"
    assert _first_note_id_for(pairs, ["Zz9QwErTy1"]) is None
    assert _first_note_id_for(pairs, []) is None


def test_regex_note_id_before_id_ordering_is_currently_suppressed():
    """PINNED CONSEQUENCE, not an endorsement.

    `?noteId=...&id=...` parses fine at the pair scan (proved above), but the
    forum-id pattern is anchored to a literal `?id=` and was deliberately left
    untouched by this change, so that ordering yields no forum id — and the
    coherence gate then withholds the note rather than orphaning it.

    OpenReview's own links put `id=` first, so this shape is not expected in
    real traffic; the scan handles it defensively because query-parameter order
    carries no guarantee once a link is forwarded or rewritten. If the forum-id
    pattern is ever widened, this test flips to a pairing assertion and no other
    change is needed.
    """
    result = _regex_extract(
        body="https://openreview.net/forum?noteId=jnHgRMHgrm&id=ll0avn6ylq"
    )
    assert result.openreview_forum_ids == []
    assert result.openreview_note_id is None


# --- the LLM path -----------------------------------------------------------
def test_llm_path_reports_a_note_id_read_from_the_body():
    """CORRECTED BEHAVIOUR. This field is regex-read on the LLM path too.

    Replaces two earlier tests that pinned the opposite. The original reasoning
    was that regex-filling would override a real model answer — but the
    distiller is never ASKED for a note id: its prompt has no note-id line, and
    its OPENREVIEW_ID line reports a bare forum id with the link (and so the
    `noteId` parameter) already discarded. There is no answer to override.
    """
    result = EmailExtractor().extract(
        "",
        f"Reply here: {_REAL_SHAPE}",
        _SENDER,
        _SENDER_NAME,
        _distilled(openreview_ids_raw=["ll0avn6ylq"]),
    )
    assert result.method == "llm_distiller"
    assert result.openreview_note_id == "jnHgRMHgrm"


def test_both_paths_agree_on_the_note_id():
    """One exact answer either way, like the notification sender.

    The pairing logic is the SAME helper on both paths, so a body cannot yield
    one note id through the distiller and another through the fallback.
    """
    body = f"Reply here: {_REAL_SHAPE}"
    via_regex = EmailExtractor().extract("", body, _SENDER, _SENDER_NAME, None)
    via_llm = EmailExtractor().extract(
        "", body, _SENDER, _SENDER_NAME, _distilled(openreview_ids_raw=["ll0avn6ylq"])
    )
    assert via_regex.openreview_note_id == via_llm.openreview_note_id == "jnHgRMHgrm"


def test_llm_path_still_never_regexes_the_forum_ids():
    """THE BOUNDARY of the correction — `openreview_forum_ids` is UNTOUCHED.

    The prompt DOES ask for OPENREVIEW_ID lines, so that list stays the model's
    answer and regex must never top it up. Here the body carries a forum link
    the regex would happily find, and the distiller reports none: the list must
    stay empty. Same call, same body, one field regex-read and the other not —
    which is exactly the "is this value in the prompt contract" test.
    """
    result = EmailExtractor().extract(
        "", f"Reply here: {_REAL_SHAPE}", _SENDER, _SENDER_NAME,
        _distilled(openreview_ids_raw=[]),
    )
    assert result.openreview_forum_ids == []


def test_llm_path_note_id_is_gated_against_the_models_forum_ids():
    """ACCEPTED CONSEQUENCE of that gate, pinned deliberately.

    The note id stays coherent with the list beside it, and on this path that
    list is the MODEL's. So when the model misses a forum id the regex found,
    its note id is withheld rather than left pointing into a forum this result
    never mentions. Fails toward reporting less, matching the module's
    precision-over-recall stance.

    Note this is the same body as the test above; only the model's answer
    differs, which is what makes the gate — not the regex — the deciding factor.
    """
    result = EmailExtractor().extract(
        "", f"Reply here: {_REAL_SHAPE}", _SENDER, _SENDER_NAME,
        _distilled(openreview_ids_raw=[]),
    )
    assert result.openreview_note_id is None


def test_extraction_result_note_id_defaults_to_none():
    assert ExtractionResult().openreview_note_id is None


# --- OpenReview notification sender ----------------------------------------
# Detects that the QUOTED original message came from OpenReview's per-venue
# notification address. The address is the only stable part of a quoted header:
# the field labels around it are written by the replier's mail client and vary
# by locale, so nothing here may depend on them.
_ADDR = "aaai2027-notifications@openreview.net"

# The real reported example, verbatim in shape: a Chinese mail client's quote
# format with Chinese field labels. Values are what matter, labels are noise.
_CHINESE_QUOTE = f"""\
老师您好，请见下方邮件。

发件人:"AAAI 2027" <{_ADDR}>
发送时间:2026-08-27 15:28:28 (星期四)
收件人: pengshaohui@iscas.ac.cn
主题: [AAAI 2027] Senior Program Committee 6UDQ commented on a paper you are
reviewing. Paper Number: 1030, Paper Title: "BlindTune:..."
"""

_ENGLISH_QUOTE = f"""\
Hi, see below.

From: "AAAI 2027" <{_ADDR}>
Sent: 2026-08-27 15:28:28
To: pengshaohui@iscas.ac.cn
Subject: [AAAI 2027] Senior Program Committee 6UDQ commented on a paper you
are reviewing. Paper Number: 1030
"""


def test_notification_sender_detected_in_chinese_labelled_quote():
    """The reported real-world case."""
    result = _regex_extract(body=_CHINESE_QUOTE)
    assert result.openreview_notification_sender == _ADDR


def test_notification_sender_detected_in_english_labelled_quote():
    """Same address, English labels — the label must not be load-bearing."""
    result = _regex_extract(body=_ENGLISH_QUOTE)
    assert result.openreview_notification_sender == _ADDR


def test_notification_sender_is_label_independent_across_locales():
    """Every one of these must behave identically, including no label at all.

    This is THE guard against reintroducing a `From:`-style anchor: such a
    pattern passes the English case and silently fails every other locale, which
    is exactly the bug shape the reported example arrived as.
    """
    for label in ["From:", "发件人:", "De:", "Von:", "Da:", "Från:", "差出人:", ""]:
        body = f'{label}"AAAI 2027" <{_ADDR}>'
        assert _regex_extract(body=body).openreview_notification_sender == _ADDR, label


def test_notification_sender_venue_prefix_is_not_hardcoded():
    """The prefix changes per conference and per year."""
    for prefix in ["aaai2027", "aaai2026", "neurips2025", "iclr2026", "icml2027"]:
        addr = f"{prefix}-notifications@openreview.net"
        assert _regex_extract(body=f"From: <{addr}>").openreview_notification_sender == addr


def test_notification_sender_accepts_the_full_allowed_prefix_shape():
    """Alphanumeric-or-hyphen: digits-only, single char, and internal hyphens."""
    for prefix in ["2027", "a", "no-reply", "aaai-2027-main"]:
        addr = f"{prefix}-notifications@openreview.net"
        assert _regex_extract(body=addr).openreview_notification_sender == addr, prefix


def test_notification_sender_found_in_subject_too():
    result = _regex_extract(subject=f"Fwd: mail from {_ADDR}")
    assert result.openreview_notification_sender == _ADDR


def test_notification_sender_matches_inside_a_mailto_link():
    result = _regex_extract(body=f"<a href='mailto:{_ADDR}'>reply</a>")
    assert result.openreview_notification_sender == _ADDR


def test_notification_sender_preserves_case_verbatim():
    """Returned as written, NOT lowercased — the field's job is to show what the
    email said. Consumers comparing it must casefold; that is documented on the
    field and pinned here so the behaviour is deliberate rather than incidental.
    """
    mixed = "AAAI2027-Notifications@OpenReview.NET"
    assert _regex_extract(body=f"From: <{mixed}>").openreview_notification_sender == mixed


def test_notification_sender_takes_the_first_of_several():
    """A quoted chain can carry the address more than once, or two venues."""
    result = _regex_extract(
        body=(
            f"From: <{_ADDR}>\n"
            "quoted...\n"
            "From: <iclr2026-notifications@openreview.net>"
        )
    )
    assert result.openreview_notification_sender == _ADDR


# --- what must NOT match ----------------------------------------------------
# "Fits the pattern" means, precisely: a local part of `<venue>-notifications`
# where <venue> is one or more alphanumerics/hyphens starting AND ending with an
# alphanumeric, at the exact domain openreview.net. Each rejection below breaks
# exactly one of those clauses.
def test_notification_sender_rejects_bare_notifications_address():
    """No venue prefix at all — the required `<venue>-` is simply absent.

    The boundary the task calls out: a generic notifications@openreview.net is
    not this signal, and must not be smuggled in by a pattern that treats the
    prefix as optional.
    """
    result = _regex_extract(body="Please write to notifications@openreview.net for help.")
    assert result.openreview_notification_sender is None


def test_notification_sender_rejects_prefix_that_is_only_a_hyphen():
    """The venue must START with an alphanumeric, so a bare `-notifications`
    (the one-character-away neighbour of the case above) is still not a venue."""
    assert (
        _regex_extract(body="write to -notifications@openreview.net").openreview_notification_sender
        is None
    )


def test_notification_sender_rejects_other_openreview_addresses():
    for addr in [
        "noreply@openreview.net",
        "info@openreview.net",
        "notifications-aaai2027@openreview.net",  # prefix on the wrong side
        "aaai2027-notification@openreview.net",  # singular: not the real mailbox
    ]:
        assert _regex_extract(body=addr).openreview_notification_sender is None, addr


def test_notification_sender_rejects_a_suffix_of_a_longer_local_part():
    """`foo.bar-notifications@...` must not be read as `bar-notifications@...`.

    Without the leading lookbehind the engine simply retries one character in
    and reports a substring that was never an address — a fabricated value, not
    a missed one.
    """
    assert (
        _regex_extract(body="foo.bar-notifications@openreview.net").openreview_notification_sender
        is None
    )


def test_notification_sender_rejects_wrong_and_lookalike_domains():
    """A lookalike must be REJECTED, not truncated into the real domain."""
    for addr in [
        "aaai2027-notifications@example.com",
        "aaai2027-notifications@openreview.net.evil.com",
        "aaai2027-notifications@openreview.network",
        "aaai2027-notifications@notopenreview.net",
    ]:
        assert _regex_extract(body=addr).openreview_notification_sender is None, addr


def test_notification_sender_ignores_an_unrelated_openreview_mention():
    """Talking ABOUT OpenReview is not a quoted notification."""
    for text in [
        "Please check your OpenReview account settings.",
        "I cannot log in to openreview.net at all.",
        "See https://openreview.net/forum?id=Ab3xY9kLm2 for the paper.",
    ]:
        assert _regex_extract(body=text).openreview_notification_sender is None, text


def test_notification_sender_none_when_no_openreview_mention_at_all():
    result = _regex_extract(
        subject="Re: Your Submission 22336",
        body="Could you clarify the page limit?",
        sender=_SENDER,
        sender_name=_SENDER_NAME,
    )
    assert result.openreview_notification_sender is None
    assert result.method == "regex_fallback"


# --- interaction with the rest of the result --------------------------------
def test_notification_sender_leaves_the_other_fields_untouched():
    """Strictly additive: the quote's own number and link still extract."""
    result = _regex_extract(
        body=(
            f"From: <{_ADDR}>\n"
            "Subject: commented on submission 1030\n"
            "https://openreview.net/forum?id=ll0avn6ylq&noteId=jnHgRMHgrm"
        ),
        sender=_SENDER,
        sender_name=_SENDER_NAME,
    )
    assert result.openreview_notification_sender == _ADDR
    assert result.submission_numbers == ["1030"]
    assert result.openreview_forum_ids == ["ll0avn6ylq"]
    assert result.openreview_note_id == "jnHgRMHgrm"
    assert len(result.authors) == 1
    assert result.method == "regex_fallback"


def test_notification_sender_alone_does_not_flip_method_off_none():
    """PINNED CONSEQUENCE, deliberate — not an oversight.

    `method` describes the IDENTIFIER extraction, and this field is explicitly
    outside it, so it is kept out of `found_anything`. The corner that exposes
    is a text carrying the address and nothing else — no number, no link, and
    no sender — which reports `method="none"` beside a populated address.

    Kept because letting this field flip `method` would make `method` claim
    something it does not mean, and because the corner needs a senderless email
    to reach: every real email has a sender, so production never sees it.
    """
    result = _regex_extract(body=f"From: <{_ADDR}>", sender="", sender_name=None)
    assert result.openreview_notification_sender == _ADDR
    assert result.method == "none"
    assert result.submission_numbers == []


# --- the LLM path ----------------------------------------------------------
def test_llm_path_reports_the_notification_sender():
    """Populated on the distiller path, because the distiller is never asked
    about it in any form — so leaving it unset would record nothing at all on
    the path production actually runs, not "looked, found none".

    `openreview_note_id` is now read on this path for the identical reason; the
    contrast this docstring used to draw between the two no longer exists.
    """
    result = EmailExtractor().extract(
        "", _CHINESE_QUOTE, _SENDER, _SENDER_NAME,
        _distilled(submission_numbers_raw=["1030"]),
    )
    assert result.method == "llm_distiller"
    assert result.openreview_notification_sender == _ADDR
    # None here only because THIS body carries no forum link at all — not
    # because the path withholds it. See the note-id LLM tests above.
    assert result.openreview_note_id is None


def test_llm_path_notification_sender_is_none_when_absent():
    """Populated on this path does not mean always-truthy on this path."""
    result = EmailExtractor().extract(
        "", "Could you clarify the page limit?", _SENDER, _SENDER_NAME,
        _distilled(submission_numbers_raw=["1030"]),
    )
    assert result.method == "llm_distiller"
    assert result.openreview_notification_sender is None


def test_both_paths_agree_on_the_same_body():
    """One exact answer either way — the whole reason it is computed before the
    paths diverge rather than once per branch."""
    body = _ENGLISH_QUOTE
    via_regex = EmailExtractor().extract("", body, _SENDER, _SENDER_NAME, None)
    via_llm = EmailExtractor().extract("", body, _SENDER, _SENDER_NAME, _distilled())
    assert (
        via_regex.openreview_notification_sender
        == via_llm.openreview_notification_sender
        == _ADDR
    )


# --- the finder in isolation ------------------------------------------------
def test_find_notification_sender_prefers_subject_over_body():
    assert (
        _find_openreview_notification_sender(f"subj {_ADDR}", "body iclr2026-notifications@openreview.net")
        == _ADDR
    )


def test_find_notification_sender_returns_none_for_empty_input():
    assert _find_openreview_notification_sender("", "") is None


def test_extraction_result_notification_sender_defaults_to_none():
    assert ExtractionResult().openreview_notification_sender is None


# --- openreview_reply_candidate (the combined signal) -----------------------
# A pure boolean AND over two fields already tested above, so nothing here
# re-tests detection. What IS worth pinning: that neither half alone is enough,
# that the flag cannot be set or drift independently of those two fields, and
# the one consequence the combination inherits from the LLM path.
_FORUM_NOTE_LINK = "https://openreview.net/forum?id=ll0avn6ylq&noteId=jnHgRMHgrm"
_BARE_FORUM_LINK = "https://openreview.net/forum?id=ll0avn6ylq"

_BOTH_SIGNALS_BODY = f"""\
请见下方邮件。

发件人:"AAAI 2027" <{_ADDR}>
发送时间:2026-08-27 15:28:28 (星期四)
收件人: pengshaohui@iscas.ac.cn
主题: [AAAI 2027] Senior Program Committee 6UDQ commented on a paper you are
reviewing. Paper Number: 1030
{_FORUM_NOTE_LINK}
"""


def test_reply_candidate_true_when_both_signals_present():
    """The real reported example, end to end: Chinese-quoted notification whose
    body also carries the forum+note link."""
    result = _regex_extract(body=_BOTH_SIGNALS_BODY, sender=_SENDER, sender_name=_SENDER_NAME)
    assert result.openreview_note_id == "jnHgRMHgrm"
    assert result.openreview_notification_sender == _ADDR
    assert result.openreview_reply_candidate is True


def test_reply_candidate_false_with_note_id_but_no_sender():
    """A forum+note link can be pasted into any ordinary question."""
    result = _regex_extract(body=f"About this comment: {_FORUM_NOTE_LINK}")
    assert result.openreview_note_id == "jnHgRMHgrm"
    assert result.openreview_notification_sender is None
    assert result.openreview_reply_candidate is False


def test_reply_candidate_false_with_sender_but_no_note_id():
    """The address can be quoted in an email that is not a reply to it."""
    result = _regex_extract(body=f"I get mail from {_ADDR} constantly, please stop.")
    assert result.openreview_note_id is None
    assert result.openreview_notification_sender == _ADDR
    assert result.openreview_reply_candidate is False


def test_reply_candidate_false_when_neither_signal_present():
    result = _regex_extract(
        subject="Re: Your Submission 22336",
        body="Could you clarify the page limit?",
        sender=_SENDER,
        sender_name=_SENDER_NAME,
    )
    assert result.openreview_reply_candidate is False


def test_reply_candidate_false_for_an_unrelated_openreview_mention():
    for text in [
        "Please check your OpenReview account settings.",
        "I cannot log in to openreview.net at all.",
    ]:
        assert _regex_extract(body=text).openreview_reply_candidate is False, text


def test_reply_candidate_needs_a_note_id_not_merely_a_forum_id():
    """A BARE forum id alongside the address is NOT enough.

    This is the distinction the left operand encodes: a forum id says some
    OpenReview paper is mentioned, a note id says a specific comment is. Using
    `openreview_forum_ids` here instead would flip this case to True.
    """
    result = _regex_extract(body=f"From: <{_ADDR}>\nPaper: {_BARE_FORUM_LINK}")
    assert result.openreview_forum_ids == ["ll0avn6ylq"]
    assert result.openreview_note_id is None
    assert result.openreview_notification_sender == _ADDR
    assert result.openreview_reply_candidate is False


# --- the model-level truth table, independent of any detection --------------
def test_reply_candidate_truth_table_on_the_model_itself():
    """All four combinations, constructed directly — proves it is an AND and
    not, say, an OR that the detection tests happen not to distinguish."""
    cases = [
        ("n", "s", True),
        ("n", None, False),
        (None, "s", False),
        (None, None, False),
    ]
    for note, sender, expected in cases:
        result = ExtractionResult(
            openreview_note_id=note, openreview_notification_sender=sender
        )
        assert result.openreview_reply_candidate is expected, (note, sender)


def test_reply_candidate_defaults_to_false():
    assert ExtractionResult().openreview_reply_candidate is False


def test_reply_candidate_is_serialized():
    """It must reach the wire/JSON column, not just exist in memory."""
    dumped = ExtractionResult(
        openreview_note_id="n", openreview_notification_sender="s"
    ).model_dump()
    assert dumped["openreview_reply_candidate"] is True
    assert ExtractionResult().model_dump()["openreview_reply_candidate"] is False


# --- it cannot drift from the two fields it summarizes ----------------------
def test_reply_candidate_cannot_be_set_independently_of_its_operands():
    """Derived, so a caller cannot hand in a value contradicting the fields.

    A plain stored boolean populated at each construction site could be set to
    anything; this cannot, which is what makes 'both signals came from the same
    email' structural rather than a rule someone has to remember.
    """
    result = ExtractionResult(openreview_reply_candidate=True)
    assert result.openreview_reply_candidate is False


def test_reply_candidate_self_heals_a_corrupt_persisted_value():
    """Recomputed on load, so a stale stored True cannot outlive its operands."""
    corrupt = {
        "submission_numbers": [],
        "openreview_forum_ids": [],
        "openreview_note_id": None,
        "openreview_notification_sender": None,
        "authors": [],
        "method": "regex_fallback",
        "openreview_reply_candidate": True,
    }
    assert ExtractionResult.model_validate(corrupt).openreview_reply_candidate is False


def test_reply_candidate_round_trips_through_a_dump():
    original = _regex_extract(body=_BOTH_SIGNALS_BODY)
    assert original.openreview_reply_candidate is True
    revived = ExtractionResult.model_validate(original.model_dump())
    assert revived.openreview_reply_candidate is True


# --- no state carries between calls -----------------------------------------
def test_reply_candidate_does_not_bleed_between_bodies_on_one_extractor():
    """ONE extractor instance, alternating bodies, interleaved deliberately.

    `EmailExtractor` holds no instance state and the finders are pure, so a
    signal found in one body can never be attributed to the next. Interleaving
    (not merely running them in sequence) is what would expose a cached or
    carried-over value: a True leaking forward would show on the very next
    call, and a stale operand would show on the repeat.
    """
    extractor = EmailExtractor()
    both = lambda: extractor.extract("", _BOTH_SIGNALS_BODY, _SENDER, _SENDER_NAME, None)
    neither = lambda: extractor.extract("", "No identifiers here.", _SENDER, _SENDER_NAME, None)
    note_only = lambda: extractor.extract("", _FORUM_NOTE_LINK, _SENDER, _SENDER_NAME, None)

    for _ in range(2):
        assert both().openreview_reply_candidate is True
        assert neither().openreview_reply_candidate is False
        assert note_only().openreview_reply_candidate is False
        assert neither().openreview_reply_candidate is False
        assert both().openreview_reply_candidate is True


def test_extractor_holds_no_instance_state():
    """The structural reason the test above can be relied on rather than hoped
    for: there is nowhere for a previous body's signal to be kept."""
    extractor = EmailExtractor()
    extractor.extract("", _BOTH_SIGNALS_BODY, _SENDER, _SENDER_NAME, None)
    assert extractor.__dict__ == {}


# --- reachable on the production path ---------------------------------------
def test_reply_candidate_is_reachable_on_the_llm_path():
    """INVERTED from the test that used to pin the opposite.

    It once asserted this flag could never be True on the distiller path,
    because `openreview_note_id` was hardcoded None there — which meant it could
    never fire in production, where `QUERY_STRATEGY=distill` runs. That was a
    misclassification of the note id rather than a flaw in the AND, and it is
    now fixed, so the same body reaches True either way.

    Kept and inverted rather than deleted: this is the single assertion that the
    feature actually works where it has to, and the side-by-side harness was
    already here to make the point.
    """
    extractor = EmailExtractor()
    via_regex = extractor.extract("", _BOTH_SIGNALS_BODY, _SENDER, _SENDER_NAME, None)
    via_llm = extractor.extract(
        "", _BOTH_SIGNALS_BODY, _SENDER, _SENDER_NAME,
        _distilled(openreview_ids_raw=["ll0avn6ylq"]),
    )

    assert via_regex.openreview_reply_candidate is True
    assert via_llm.openreview_reply_candidate is True
    # Both operands really are present on the production path, not just the flag.
    assert via_llm.openreview_note_id == "jnHgRMHgrm"
    assert via_llm.openreview_notification_sender == _ADDR


# --- sender-based author ---------------------------------------------------
def test_regex_sender_becomes_the_only_author():
    result = _regex_extract(sender=_SENDER, sender_name=_SENDER_NAME)
    assert result.authors == [
        AuthorMention(name="Jane Roe", email="jane@example.edu", affiliation=None)
    ]


def test_regex_sender_only_still_counts_as_found():
    """No regex hit but a real sender is still a usable result, not `none`."""
    result = _regex_extract(
        body="Please advise on my situation.", sender=_SENDER, sender_name=_SENDER_NAME
    )
    assert result.method == "regex_fallback"
    assert result.submission_numbers == []
    assert len(result.authors) == 1


def test_regex_sender_email_without_name():
    result = _regex_extract(sender=_SENDER, sender_name=None)
    assert result.authors == [
        AuthorMention(name=None, email="jane@example.edu", affiliation=None)
    ]


def test_regex_sender_blank_fields_produce_no_author():
    result = _regex_extract(body="submission 12345", sender="   ", sender_name="  ")
    assert result.authors == []
    assert result.method == "regex_fallback"  # the number is still a find


def test_regex_never_invents_affiliation():
    """Signature-block parsing is out of scope — affiliation stays unset."""
    result = _regex_extract(
        body="Best regards,\nJane Roe\nExample University\nDept of CS",
        sender=_SENDER,
        sender_name=_SENDER_NAME,
    )
    assert result.authors[0].affiliation is None


# --- nothing found ---------------------------------------------------------
def test_regex_nothing_usable_is_method_none():
    """No sender and no regex hit: nothing to go on at all."""
    result = _regex_extract(subject="Question", body="Can you help?", sender="")
    assert result.method == "none"
    assert result.submission_numbers == []
    assert result.openreview_forum_ids == []
    assert result.authors == []


def test_regex_empty_input_is_method_none():
    result = _regex_extract()
    assert result.method == "none"


def test_regex_fallback_finds_both_identifiers_together():
    result = _regex_extract(
        subject="Re: Your Submission 22336",
        body="Forum: https://openreview.net/forum?id=Ab3xY9kLm2",
        sender=_SENDER,
        sender_name=_SENDER_NAME,
    )
    assert result.submission_numbers == ["22336"]
    assert result.openreview_forum_ids == ["Ab3xY9kLm2"]
    assert len(result.authors) == 1
    assert result.method == "regex_fallback"


def test_regex_fallback_never_raises_on_odd_input():
    result = EmailExtractor().extract("", "", "", None, None)
    assert result.method == "none"


# ---------------------------------------------------------------------------
# extracted_reply_text — the body with quoted history removed
#
# The quote-boundary logic itself is tested in test_quoted_reply.py; nothing
# here re-tests detection. What IS tested here: that the field is populated
# identically on BOTH paths, that it is UNGATED, and that it stays out of the
# `method` decision.
# ---------------------------------------------------------------------------
_RT_ADDRESS = "aaai2027-notifications@openreview.net"
_RT_LINK = "https://openreview.net/forum?id=ll0avn6ylq&noteId=jnHgRMHgrm"

_RT_CHINESE_QUOTE = (
    "-----原始邮件-----\n"
    f'发件人:"AAAI 2027" <{_RT_ADDRESS}>\n'
    "发送时间:2026-08-27 15:28:28 (星期四)\n"
    "收件人: pengshaohui@iscas.ac.cn\n"
    f"主题: [AAAI 2027] SPC commented on a paper  {_RT_LINK}\n"
)
_RT_REPLY = "Dear Chairs,\n\nI will provide review before the deadline.\n\n"
_RT_CLEAN = "Dear Chairs,\n\nI will provide review before the deadline."


def _both_paths(body: str, **distilled_kwargs):
    """Run the SAME body down both paths and hand back both results."""
    extractor = EmailExtractor()
    via_regex = extractor.extract("", body, _SENDER, _SENDER_NAME, None)
    via_llm = extractor.extract(
        "", body, _SENDER, _SENDER_NAME, _distilled(**distilled_kwargs)
    )
    return via_regex, via_llm


def test_reply_text_from_the_real_chinese_quote_on_both_paths():
    """The reported case: only the person's own sentence survives, either way."""
    via_regex, via_llm = _both_paths(
        _RT_REPLY + _RT_CHINESE_QUOTE, openreview_ids_raw=["ll0avn6ylq"]
    )

    assert via_regex.extracted_reply_text == _RT_CLEAN
    assert via_llm.extracted_reply_text == _RT_CLEAN
    assert via_llm.method == "llm_distiller"
    assert via_regex.method == "regex_fallback"


def test_reply_text_with_no_quote_is_the_whole_trimmed_body_on_both_paths():
    via_regex, via_llm = _both_paths("  Could you clarify the page limit?  \n\n")

    assert via_regex.extracted_reply_text == "Could you clarify the page limit?"
    assert via_llm.extracted_reply_text == "Could you clarify the page limit?"


def test_reply_text_is_empty_when_the_body_is_entirely_quote_on_both_paths():
    """Empty string, not the quote — on the production path too."""
    via_regex, via_llm = _both_paths(
        _RT_CHINESE_QUOTE, openreview_ids_raw=["ll0avn6ylq"]
    )

    assert via_regex.extracted_reply_text == ""
    assert via_llm.extracted_reply_text == ""


def test_reply_text_both_paths_agree_across_a_range_of_bodies():
    """One pure function of one argument, so the paths cannot disagree — pinned
    rather than assumed, because it is the property that would break first if
    someone gated or hoisted the call differently on one side."""
    bodies = [
        _RT_REPLY + _RT_CHINESE_QUOTE,
        _RT_REPLY + "-----Original Message-----\nFrom: A <a@b.com>\nSent: Wed\nTo: c@d.com\n",
        _RT_REPLY + "On Thu X <x@y.com> wrote:\n> hello\n> there\n",
        "No quote anywhere in this one.",
        _RT_CHINESE_QUOTE,
        "",
    ]
    for body in bodies:
        via_regex, via_llm = _both_paths(body)
        assert via_regex.extracted_reply_text == via_llm.extracted_reply_text, body[:40]


# --- ungated: this is a general-purpose field -------------------------------
def test_reply_text_is_populated_when_reply_candidate_is_false():
    """THE anti-gating test.

    An ordinary email with no OpenReview link, no notification sender and no
    quote at all: `openreview_reply_candidate` is False, and the reply text is
    populated anyway. The field is built FOR the OpenReview case but is not
    scoped to it, and gating it there would make a general utility silently
    unavailable to whatever needs it next.
    """
    via_regex, via_llm = _both_paths(
        "Hi, when will the decisions be released?\n\n"
        "-----Original Message-----\nFrom: A <a@b.com>\nSent: Wed\nTo: c@d.com\n"
    )

    for result in (via_regex, via_llm):
        assert result.openreview_reply_candidate is False
        assert result.extracted_reply_text == "Hi, when will the decisions be released?"


def test_reply_text_populated_for_an_email_with_no_openreview_trace_at_all():
    via_regex, via_llm = _both_paths("Please confirm the camera-ready deadline.")

    for result in (via_regex, via_llm):
        assert result.openreview_forum_ids == []
        assert result.openreview_note_id is None
        assert result.openreview_notification_sender is None
        assert result.openreview_reply_candidate is False
        assert result.extracted_reply_text == "Please confirm the camera-ready deadline."


# --- it stays out of the `method` decision ----------------------------------
def test_reply_text_does_not_flip_method_off_none():
    """`found_anything` must be untouched by this field.

    The exclusion argument is stronger here than for the two OpenReview
    scalars: reply text is non-empty for essentially every real body, so
    counting it would make `method="none"` nearly unreachable rather than
    merely exposing a corner.
    """
    result = _regex_extract(subject="Question", body="Can you help?", sender="")

    assert result.method == "none"
    assert result.extracted_reply_text == "Can you help?"


def test_reply_text_present_on_a_method_none_result_does_not_imply_identifiers():
    result = _regex_extract(subject="Question", body="Can you help?", sender="")

    assert result.submission_numbers == []
    assert result.openreview_forum_ids == []
    assert result.authors == []


# --- shape ------------------------------------------------------------------
def test_extraction_result_reply_text_defaults_to_empty_string():
    """Defaults to "" rather than None — the field is `str`, never optional, so
    a consumer never has to handle two kinds of empty."""
    result = ExtractionResult()

    assert result.extracted_reply_text == ""
    assert isinstance(result.extracted_reply_text, str)


def test_reply_text_is_serialized():
    dumped = ExtractionResult(extracted_reply_text="hello").model_dump()

    assert dumped["extracted_reply_text"] == "hello"
    assert "extracted_reply_text" in ExtractionResult().model_dump()


def test_reply_text_is_a_stored_field_not_a_computed_one():
    """Unlike `openreview_reply_candidate`, this one is settable.

    It is a function of the raw body, which this model does not carry, so there
    is nothing on the object to derive it from — stored is the only option, and
    that difference is worth pinning so nobody "fixes" it into a computed field.
    """
    assert "extracted_reply_text" in ExtractionResult.model_fields
    assert ExtractionResult(extracted_reply_text="set").extracted_reply_text == "set"


def test_reply_text_empty_body_is_empty_string_on_both_paths():
    via_regex, via_llm = _both_paths("")

    assert via_regex.extracted_reply_text == ""
    assert via_llm.extracted_reply_text == ""

