"""Tests for scripts/recovery/appeal_sheet_rules.py (Part 2a: pure rules, no I/O).

Synthetic data only. SENTINEL marks text that must never reach a manifest, a
plan, a repr or an error message.
"""

from __future__ import annotations

import codecs
import csv
import io
import json
from datetime import datetime, timezone

import pytest

from scripts.recovery import appeal_sheet_rules as r

SENTINEL = "SENTINEL_PII_TEXT_q7"
NOW = datetime(2026, 10, 6, 12, 0, tzinfo=timezone.utc)
SNAP = datetime(2026, 9, 20, 8, 30, 15, tzinfo=timezone.utc)
REPLY = f"Dear Author,\n\nThank you for your message about the decision. {SENTINEL}\n\nBest Regards,\nAAAI 2027 PC Team"
MODIFIED = f"Dear Author,\n\nWe looked again at your paper and the reviews. {SENTINEL}\n\nBest regards,\nMarc"


# --- helpers -------------------------------------------------------------------------

def cells(ticket="101", reply=REPLY, ticked="FALSE", modified="", **extra) -> dict:
    row = {c: "" for c in r.COLUMNS}
    row.update({"ticket_id": ticket, "reply_draft": reply, "use_provided_draft": ticked,
                "modified_draft": modified, "request_body": f"body {SENTINEL}",
                "draft_type": "approved wording", "mode": "merged"})
    row.update(extra)
    return row


def sheet_bytes(rows: list[dict], *, bom=True, crlf=True, header=r.COLUMNS) -> bytes:
    buf = io.StringIO()
    writer = csv.writer(buf, quoting=csv.QUOTE_ALL, lineterminator="\r\n" if crlf else "\n")
    writer.writerow(header)
    for row in rows:
        writer.writerow([row[c] for c in r.COLUMNS])
    data = buf.getvalue().encode("utf-8")
    return (codecs.BOM_UTF8 + data) if bom else data


def srow(line=2, **kw) -> r.SheetRow:
    return r.SheetRow(line, cells(**kw))


def entry(ticket=101, sheet=r.SHEET_APPEAL, reply=REPLY, updated="2026-09-20T08:30:15Z"):
    return r.ManifestEntry(ticket, 5000 + ticket, updated, r.text_sha256(reply), sheet)


def manifest(*entries) -> dict:
    return {e.ticket_id: e for e in entries}


# --- parser ----------------------------------------------------------------------------

@pytest.mark.parametrize("bom", [True, False])
@pytest.mark.parametrize("crlf", [True, False])
def test_parser_reads_the_exported_format_with_or_without_bom_and_crlf(bom, crlf):
    rows = r.parse_sheet_csv(sheet_bytes([cells(), cells(ticket="102")], bom=bom, crlf=crlf))
    assert [row.cells["ticket_id"] for row in rows] == ["101", "102"]
    assert [row.line for row in rows] == [2, 3]
    assert rows[0].cells["reply_draft"].replace("\r\n", "\n") == REPLY


def test_parser_keeps_line_breaks_inside_cells():
    rows = r.parse_sheet_csv(sheet_bytes([cells()], crlf=False))
    assert rows[0].cells["reply_draft"] == REPLY


def test_parser_ignores_blank_records():
    data = sheet_bytes([cells()], crlf=False) + b"\n\n"
    assert len(r.parse_sheet_csv(data)) == 1


@pytest.mark.parametrize("header", [
    r.COLUMNS[::-1],                       # reordered
    r.COLUMNS + ("extra",),                # added column
    r.COLUMNS[:-1],                        # missing column
    tuple(c.upper() for c in r.COLUMNS),   # renamed
])
def test_parser_refuses_any_header_change(header):
    buf = io.StringIO()
    csv.writer(buf).writerow(header)
    with pytest.raises(r.SheetFormatError, match="header"):
        r.parse_sheet_csv(buf.getvalue().encode())


def test_parser_refuses_an_empty_file():
    with pytest.raises(r.SheetFormatError):
        r.parse_sheet_csv(b"")


def test_parser_refuses_a_short_row_naming_only_the_record():
    data = sheet_bytes([cells()], crlf=False) + f'"102","{SENTINEL}"\n'.encode()
    with pytest.raises(r.SheetFormatError) as err:
        r.parse_sheet_csv(data)
    assert "record 3" in str(err.value) and SENTINEL not in str(err.value)


def test_parser_refuses_broken_quoting_without_keeping_the_text():
    data = sheet_bytes([cells()], crlf=False) + f'"102","a"{SENTINEL}"\n'.encode()
    with pytest.raises(r.SheetFormatError) as err:
        r.parse_sheet_csv(data)
    assert SENTINEL not in str(err.value)
    assert err.value.__context__ is None


def test_parser_refuses_non_utf8_without_keeping_the_bytes():
    data = sheet_bytes([cells()]).replace(b"Dear", b"D\xe9ar")
    with pytest.raises(r.SheetFormatError, match="UTF-8") as err:
        r.parse_sheet_csv(data)
    assert err.value.__context__ is None


def test_parser_accepts_a_very_long_cell():
    long_body = "x" * 300_000
    rows = r.parse_sheet_csv(sheet_bytes([cells(request_body=long_body)]))
    assert len(rows[0].cells["request_body"]) == 300_000


def test_sheet_row_repr_has_no_cell_text():
    assert SENTINEL not in repr(srow())


# --- small parsers + text helpers -----------------------------------------------------

@pytest.mark.parametrize("value,expected", [
    ("TRUE", True), ("true", True), (" TRUE ", True), ("FALSE", False), ("false", False),
    ("", None), ("yes", None), ("1", None), ("x", None), ("✓", None), ("TRUE.", None),
])
def test_checkbox_accepts_only_true_or_false(value, expected):
    assert r.parse_checkbox(value) is expected


@pytest.mark.parametrize("value,expected", [
    ("123", 123), (" 123 ", 123), ("0", None), ("-5", None), ("12a", None), ("", None),
    ("'123", None), ("1.0", None), ("١٢٣", None),
])
def test_ticket_id_is_a_positive_ascii_integer(value, expected):
    assert r.parse_ticket_id(value) == expected


def test_unguard_removes_exactly_one_formula_guard_apostrophe():
    assert r.unguard("'=SUM(A1)") == "=SUM(A1)"
    assert r.unguard("'-x") == "-x"
    assert r.unguard("''=x") == "''=x"          # the apostrophe is followed by ', not a formula start
    assert r.unguard("'hello") == "'hello"      # a real apostrophe stays
    assert r.unguard("'") == "'"


def test_hash_ignores_line_ending_style_and_the_guard():
    lf = "a\nb"
    assert r.text_sha256("a\r\nb") == r.text_sha256(lf) == r.text_sha256("a\rb")
    assert r.text_sha256("'=x") == r.text_sha256("=x")
    assert r.text_sha256("a\nb") != r.text_sha256("a\n\nb")


def test_fingerprint_ignores_layout_but_not_words():
    assert r.fingerprint_sha256("Dear A,\n\nText  here.") == r.fingerprint_sha256("Dear A, Text here.")
    assert r.fingerprint_sha256("Dear A, text here.") != r.fingerprint_sha256("Dear A, Text here.")


# --- text guards ------------------------------------------------------------------------

def test_a_clean_reply_passes_every_guard():
    assert r.text_guards(REPLY) == ()


@pytest.mark.parametrize("text,guard", [
    (REPLY + "\n[CHAIR: write reply]", r.G_CHAIR),
    (REPLY + "\n[chair: x]", r.G_CHAIR),
    (REPLY + "\n[ Chair note]", r.G_CHAIR),
    (REPLY + "\nAI-written suggestion, not approved wording", r.G_AI_FLAG),
    (REPLY.replace("AAAI 2027 PC Team", "[Sender name]"), r.G_SENDER_NAME),
    (REPLY.replace("Author", "{name}"), r.G_TEMPLATE),
    ("=HYPERLINK(1) " + REPLY, r.G_FORMULA),
    ("+1 " + REPLY, r.G_FORMULA),
    ("@x " + REPLY, r.G_FORMULA),
    ("-x " + REPLY, r.G_FORMULA),
    ("Dear A, ok.", r.G_TOO_SHORT),
    (REPLY + "�", r.G_ENCODING),
    (REPLY + "\x00", r.G_CONTROL),
    (REPLY + "\x1b", r.G_CONTROL),
])
def test_each_guard_fires(text, guard):
    assert guard in r.text_guards(text)


@pytest.mark.parametrize("text", ["", "   ", "\n\n", "'"])
def test_empty_text_reports_only_empty(text):
    assert r.text_guards(text) in ((r.G_EMPTY,), (r.G_TOO_SHORT,))
    assert r.text_guards("") == (r.G_EMPTY,)


def test_the_real_ai_flag_line_is_caught():
    from app.pipeline.appeal_ai_draft import AI_FLAG_LINE
    assert r.AI_FLAG_PHRASE in AI_FLAG_LINE.casefold()
    assert r.G_AI_FLAG in r.text_guards(AI_FLAG_LINE + "\n\n" + REPLY)
    # Even with the brackets removed by an editor, the phrase is still caught.
    assert r.G_AI_FLAG in r.text_guards(AI_FLAG_LINE.strip("[]") + "\n\n" + REPLY)


def test_tabs_and_newlines_inside_text_are_not_control_characters():
    assert r.G_CONTROL not in r.text_guards(REPLY.replace(" ", "\t", 1))


# --- row rules --------------------------------------------------------------------------

def test_ticked_box_posts_the_provided_reply():
    d = r.decide_row(srow(ticked="TRUE"), manifest(entry()))
    assert (d.outcome, d.reason, d.source) == (r.POST, r.R_READY, r.SOURCE_REPLY)
    assert d.text == REPLY and d.text_sha256 == r.text_sha256(REPLY)


def test_written_modified_draft_posts_the_modified_text():
    d = r.decide_row(srow(modified=MODIFIED), manifest(entry()))
    assert (d.outcome, d.source) == (r.POST, r.SOURCE_MODIFIED)
    assert d.text == MODIFIED


def test_modified_draft_line_endings_are_normalised_before_posting():
    d = r.decide_row(srow(modified=MODIFIED.replace("\n", "\r\n")), manifest(entry()))
    assert d.text == MODIFIED


def test_ticked_and_modified_is_refused():
    d = r.decide_row(srow(ticked="TRUE", modified=MODIFIED), manifest(entry()))
    assert (d.outcome, d.reason, d.text) == (r.REFUSE, r.R_BOTH, None)


@pytest.mark.parametrize("modified", ["", "   ", "\n"])
def test_unticked_with_no_modified_draft_is_skipped(modified):
    d = r.decide_row(srow(modified=modified), manifest(entry()))
    assert (d.outcome, d.reason) == (r.SKIP, r.R_NOT_APPROVED)


def test_unknown_ticket_is_refused():
    d = r.decide_row(srow(ticket="999", ticked="TRUE"), manifest(entry()))
    assert (d.outcome, d.reason, d.ticket_id) == (r.REFUSE, r.R_UNKNOWN, 999)


def test_bad_ticket_id_is_refused():
    d = r.decide_row(srow(ticket="12a", ticked="TRUE"), manifest(entry()))
    assert (d.outcome, d.reason, d.ticket_id) == (r.REFUSE, r.R_BAD_TICKET_ID, None)


def test_not_appeal_rows_are_refused_even_when_approved():
    m = manifest(entry(sheet=r.SHEET_NOT_APPEAL))
    for kw in ({"ticked": "TRUE"}, {"modified": MODIFIED}):
        assert r.decide_row(srow(**kw), m).reason == r.R_NOT_APPEAL


@pytest.mark.parametrize("box", ["", "yes", "1"])
def test_a_box_that_is_not_true_or_false_is_refused(box):
    d = r.decide_row(srow(ticked=box, modified=MODIFIED), manifest(entry()))
    assert (d.outcome, d.reason) == (r.REFUSE, r.R_BAD_CHECKBOX)


@pytest.mark.parametrize("kw", [{"ticked": "TRUE"}, {"modified": MODIFIED}, {}])
def test_an_edited_reply_draft_cell_is_refused_whatever_else_is_set(kw):
    d = r.decide_row(srow(reply=REPLY + " edited", **kw), manifest(entry()))
    assert (d.outcome, d.reason) == (r.REFUSE, r.R_REPLY_EDITED)


def test_the_reply_draft_hash_survives_crlf_and_the_formula_guard():
    m = manifest(entry(reply="=" + REPLY))
    d = r.decide_row(srow(reply="'=" + REPLY.replace("\n", "\r\n"), modified=MODIFIED), m)
    assert d.outcome == r.POST


def test_wrong_paper_row_with_only_the_box_ticked_is_refused():
    m = manifest(entry(sheet=r.SHEET_WRONG_PAPER, reply=""))
    d = r.decide_row(srow(reply="", ticked="TRUE"), m)
    assert (d.outcome, d.reason) == (r.REFUSE, r.R_WRONG_PAPER_NEEDS_MODIFIED)


def test_wrong_paper_row_posts_a_written_modified_draft():
    m = manifest(entry(sheet=r.SHEET_WRONG_PAPER, reply=""))
    d = r.decide_row(srow(reply="", modified=MODIFIED), m)
    assert (d.outcome, d.source, d.text) == (r.POST, r.SOURCE_MODIFIED, MODIFIED)


def test_ticked_box_on_an_empty_reply_is_refused():
    m = manifest(entry(reply=""))
    d = r.decide_row(srow(reply="", ticked="TRUE"), m)
    assert (d.outcome, d.reason) == (r.REFUSE, r.R_NOTHING_TO_SEND)


def test_a_ticked_placeholder_reply_is_refused_by_the_guard():
    reply = "Dear Author,\n\n[CHAIR: write reply]\n\nBest Regards,\nAAAI 2027 PC Team"
    d = r.decide_row(srow(reply=reply, ticked="TRUE"), manifest(entry(reply=reply)))
    assert (d.outcome, d.reason, d.source, d.text) == (r.REFUSE, r.G_CHAIR, r.SOURCE_REPLY, None)


def test_a_modified_draft_with_several_problems_names_them_all():
    bad = "[Sender name] [CHAIR: x] {name} " + MODIFIED
    d = r.decide_row(srow(modified=bad), manifest(entry()))
    assert d.outcome == r.REFUSE
    assert set(d.reason.split("+")) == {r.G_CHAIR, r.G_SENDER_NAME, r.G_TEMPLATE}


def test_duplicate_ticket_ids_refuse_every_copy():
    rows = [srow(2, ticket="101", ticked="TRUE"), srow(3, ticket="102", ticked="TRUE"),
            srow(4, ticket="101", modified=MODIFIED)]
    out = r.decide_rows(rows, manifest(entry(101), entry(102)))
    assert [(d.line, d.reason) for d in out] == [
        (2, r.R_DUPLICATE), (3, r.R_READY), (4, r.R_DUPLICATE)]


def test_two_bad_ticket_ids_are_not_duplicates_of_each_other():
    out = r.decide_rows([srow(2, ticket="x"), srow(3, ticket="y")], manifest(entry()))
    assert [d.reason for d in out] == [r.R_BAD_TICKET_ID, r.R_BAD_TICKET_ID]


def test_row_decision_repr_has_no_text():
    d = r.decide_row(srow(ticked="TRUE"), manifest(entry()))
    assert d.outcome == r.POST and SENTINEL not in repr(d)


# --- live-ticket rules -----------------------------------------------------------------

@pytest.mark.parametrize("status", ["new", "open", "pending", " Open "])
def test_new_open_pending_may_be_answered(status):
    assert r.status_refusal(status) is None


@pytest.mark.parametrize("status,reason", [
    ("solved", "status_solved"), ("closed", "status_closed"), ("hold", "status_hold"),
    ("deleted", "status_deleted"), (None, "status_unknown"), ("", "status_unknown"),
])
def test_other_statuses_are_refused(status, reason):
    assert r.status_refusal(status) == reason


@pytest.mark.parametrize("live,snap,changed", [
    ("2026-09-20T08:30:15Z", "2026-09-20T08:30:15Z", False),
    ("2026-09-20T08:30:15+00:00", "2026-09-20T08:30:15Z", False),
    ("2026-09-20T10:30:15+02:00", "2026-09-20T08:30:15Z", False),
    ("2026-09-20T08:30:15", "2026-09-20T08:30:15Z", False),   # naive read as UTC
    ("2026-09-20T08:30:16Z", "2026-09-20T08:30:15Z", True),
    ("2026-10-01T00:00:00Z", "2026-09-20T08:30:15Z", True),
    (None, "2026-09-20T08:30:15Z", True),
    ("2026-09-20T08:30:15Z", None, True),
    (None, None, True),                       # both unknown is NOT "unchanged"
    ("garbage", "also garbage", True),
    ("garbage", "2026-09-20T08:30:15Z", True),
])
def test_ticket_changed_unless_both_times_are_known_and_equal(live, snap, changed):
    assert r.ticket_changed(live, snap) is changed


def test_reconcile_finds_our_reply_despite_zendesk_relayout():
    fp = r.fingerprint_sha256(MODIFIED)
    comments = [
        {"id": 1, "public": True, "author_id": 7, "plain_body": "Something else entirely here."},
        {"id": 2, "public": False, "author_id": 9, "plain_body": MODIFIED},     # internal note
        {"id": 3, "public": True, "author_id": 9,
         "plain_body": MODIFIED.replace("\n\n", "\n").replace(" ", "  ")},
    ]
    assert r.find_posted_comment(comments, fp) == 3
    assert r.find_posted_comment(comments, fp, author_id=9) == 3
    assert r.find_posted_comment(comments, fp, author_id=7) is None


def test_reconcile_falls_back_to_body_and_returns_none_without_a_match():
    fp = r.fingerprint_sha256(MODIFIED)
    assert r.find_posted_comment([{"id": 4, "public": True, "body": MODIFIED}], fp) == 4
    assert r.find_posted_comment([{"id": 5, "public": True, "body": REPLY}], fp) is None
    assert r.find_posted_comment([], fp) is None


def test_reconcile_requires_public_to_be_exactly_true():
    fp = r.fingerprint_sha256(MODIFIED)
    assert r.find_posted_comment([{"id": 6, "public": "true", "plain_body": MODIFIED}], fp) is None


# --- manifest ---------------------------------------------------------------------------

def _manifest_doc():
    rows = {
        r.SHEET_APPEAL: [srow(2, ticket="101"), srow(3, ticket="103")],
        r.SHEET_WRONG_PAPER: [srow(2, ticket="102", reply="")],
        r.SHEET_NOT_APPEAL: [srow(2, ticket="104")],
    }
    snapshot = {101: (11, SNAP), 102: (12, SNAP.replace(tzinfo=None)), 103: (13, None),
                104: (14, SNAP)}
    return r.build_manifest(rows, snapshot, {"a.csv": "0" * 64}, NOW)


def test_manifest_holds_ids_and_hashes_only():
    doc = _manifest_doc()
    assert SENTINEL not in json.dumps(doc)
    assert [e["ticket_id"] for e in doc["entries"]] == [101, 102, 103, 104]
    first = doc["entries"][0]
    assert set(first) == {"ticket_id", "email_id", "zendesk_updated_at", "reply_draft_sha256", "sheet"}
    assert first["reply_draft_sha256"] == r.text_sha256(REPLY)
    assert first["zendesk_updated_at"] == "2026-09-20T08:30:15Z"
    assert doc["entries"][1]["zendesk_updated_at"] == "2026-09-20T08:30:15Z"   # naive -> UTC
    assert doc["entries"][2]["zendesk_updated_at"] is None
    assert doc["counts"] == {"appeal": 2, "not_appeal": 1, "wrong_paper": 1}


def test_manifest_round_trips():
    loaded = r.load_manifest(json.loads(json.dumps(_manifest_doc())))
    assert loaded[102] == r.ManifestEntry(102, 12, "2026-09-20T08:30:15Z", r.text_sha256(""),
                                          r.SHEET_WRONG_PAPER)


def test_manifest_refuses_a_ticket_in_two_sheets():
    rows = {r.SHEET_APPEAL: [srow(ticket="101")], r.SHEET_WRONG_PAPER: [srow(ticket="101")]}
    with pytest.raises(r.ManifestError, match="101"):
        r.build_manifest(rows, {101: (1, SNAP)}, {}, NOW)


def test_manifest_refuses_tickets_missing_from_the_snapshot():
    rows = {r.SHEET_APPEAL: [srow(ticket="101"), srow(3, ticket="102")]}
    with pytest.raises(r.ManifestError, match=r"\[102\]"):
        r.build_manifest(rows, {101: (1, SNAP)}, {}, NOW)


def test_manifest_refuses_an_unknown_sheet_and_a_bad_ticket_id():
    with pytest.raises(r.ManifestError):
        r.build_manifest({"other": [srow()]}, {101: (1, SNAP)}, {}, NOW)
    with pytest.raises(r.ManifestError):
        r.build_manifest({r.SHEET_APPEAL: [srow(ticket="x")]}, {}, {}, NOW)


@pytest.mark.parametrize("mutate", [
    lambda d: d.update(schema_version=2),
    lambda d: d.update(entries=[]),
    lambda d: d["entries"][0].update(sheet="other"),
    lambda d: d["entries"][0].update(reply_draft_sha256="abc"),
    lambda d: d["entries"][0].pop("email_id"),
    lambda d: d["entries"].append(dict(d["entries"][0])),
])
def test_load_manifest_refuses_a_malformed_manifest(mutate):
    doc = json.loads(json.dumps(_manifest_doc()))
    mutate(doc)
    with pytest.raises(r.ManifestError):
        r.load_manifest(doc)


# --- plan file ----------------------------------------------------------------------------

def _decisions():
    m = manifest(entry(101), entry(102), entry(103), entry(104, sheet=r.SHEET_NOT_APPEAL))
    rows = [srow(2, ticket="101", ticked="TRUE"), srow(3, ticket="102", modified=MODIFIED),
            srow(4, ticket="103"), srow(5, ticket="104", ticked="TRUE"), srow(6, ticket="999")]
    return r.decide_rows(rows, m), m


def _plan():
    decisions, m = _decisions()
    return r.build_plan(decisions, m, manifest_sha256="a" * 64,
                        sheet_sha256={"s.csv": "b" * 64}, created_at=NOW)


def test_plan_holds_no_text_and_counts_every_outcome():
    plan = _plan()
    assert SENTINEL not in json.dumps(plan)
    assert plan["to_post"] == 2
    assert plan["counts"]["outcome"] == {"post": 2, "refuse": 2, "skip": 1}
    assert plan["counts"]["reason"] == {r.R_READY: 2, r.R_NOT_APPROVED: 1,
                                        r.R_NOT_APPEAL: 1, r.R_UNKNOWN: 1}
    post = plan["rows"][1]
    assert post["text_sha256"] == r.text_sha256(MODIFIED)
    assert post["fingerprint_sha256"] == r.fingerprint_sha256(MODIFIED)
    assert post["email_id"] == 5102 and post["snapshot_updated_at"] == "2026-09-20T08:30:15Z"
    unknown = plan["rows"][4]
    assert unknown["email_id"] is None and unknown["text_sha256"] is None


def test_plan_round_trips():
    assert r.load_plan(json.loads(json.dumps(_plan())))["to_post"] == 2


@pytest.mark.parametrize("mutate", [
    lambda p: p.update(schema_version=9),
    lambda p: p.update(rows=None),
    lambda p: p["rows"][0].update(text="x"),                       # no text key, ever
    lambda p: p["rows"][0].update(text_sha256=None),
    lambda p: p.update(to_post=3),
    lambda p: p["rows"].append(dict(p["rows"][0])),               # same ticket twice
])
def test_load_plan_refuses_a_malformed_plan(mutate):
    plan = json.loads(json.dumps(_plan()))
    mutate(plan)
    with pytest.raises(r.PlanError):
        r.load_plan(plan)


def test_execution_must_match_the_planned_text():
    decisions, m = _decisions()
    plan = r.build_plan(decisions, m, manifest_sha256="a" * 64, sheet_sha256={}, created_at=NOW)
    assert r.decision_matches_plan(decisions[1], plan["rows"][1])
    changed = r.decide_row(srow(3, ticket="102", modified=MODIFIED + " more"), m)
    assert not r.decision_matches_plan(changed, plan["rows"][1])
    assert not r.decision_matches_plan(decisions[1], plan["rows"][0])   # another ticket
    assert not r.decision_matches_plan(decisions[2], plan["rows"][2])   # a skip never posts


@pytest.mark.parametrize("typed,expected,ok", [
    ("2", 2, True), (" 2\n", 2, True), ("02", 2, False), ("3", 2, False),
    ("", 2, False), ("yes", 2, False), ("0", 0, False),
])
def test_typed_confirmation_is_exactly_the_count(typed, expected, ok):
    assert r.confirm_count(typed, expected) is ok


# --- end to end over the exported file format ----------------------------------------

def test_exported_file_to_plan_end_to_end():
    exported = sheet_bytes([cells("101"), cells("102", reply="")])
    rows = r.parse_sheet_csv(exported)
    doc = r.build_manifest({r.SHEET_APPEAL: rows[:1], r.SHEET_WRONG_PAPER: rows[1:]},
                           {101: (1, SNAP), 102: (2, SNAP)}, {"x.csv": "c" * 64}, NOW)
    m = r.load_manifest(doc)
    # The APC ticks 101 and writes a reply for the wrong-paper ticket 102.
    filled = sheet_bytes([cells("101", ticked="TRUE"), cells("102", reply="", modified=MODIFIED)],
                         bom=False, crlf=False)
    decisions = r.decide_rows(r.parse_sheet_csv(filled), m)
    assert [(d.ticket_id, d.outcome, d.source) for d in decisions] == [
        (101, r.POST, r.SOURCE_REPLY), (102, r.POST, r.SOURCE_MODIFIED)]
