"""Tests for scripts/labeling/build_phase2_pool.py (+ candidate_patterns.py).

SCOPE LIMIT: SYNTHETIC records only — nothing here reads data/tickets/ or
data/labeling/. `scripts/labeling/` is not in the backend image, so this file
SKIPS in the container and runs on the host and in CI.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_LABELING_DIR = Path(__file__).resolve().parents[2] / "scripts" / "labeling"

pytestmark = pytest.mark.skipif(
    not (_LABELING_DIR / "build_phase2_pool.py").exists(),
    reason="scripts/labeling/ is not in the backend image (the Dockerfile copies "
    "only backend/ and data/); these tests run on the host and in CI",
)

if (_LABELING_DIR / "build_phase2_pool.py").exists():
    sys.path.insert(0, str(_LABELING_DIR))
    import build_phase2_pool as bp  # noqa: E402
    import candidate_patterns as cp  # noqa: E402
    import ticket_extraction as te  # noqa: E402

A_TEXT = "The review is about a different paper."
B_TEXT = "Our scores were high but the decision was reject."
C_TEXT = "The reviewer misunderstood the method."
D_TEXT = "This review looks chatgpt written."
E_TEXT = "Please reconsider."
PLAIN = "Hello, a question about registration."


def _msg(sender="requester", public=True, body="x"):
    return {"sender_type": sender, "sender_email_or_null": None, "body": body,
            "created_at": "t", "is_public": public}


def _rec(tid, created, text, thread=None):
    """Minimal pool record; patterns read subject + initial_message_body."""
    return {"ticket_id": tid, "created_at": created, "channel": "email", "status": "closed",
            "subject": "", "initial_message_body": text, "marc_reply_body": None,
            "submission_numbers_mentioned": None,
            "thread": thread if thread is not None else [_msg(body=text)],
            "marc_replies": []}


S2025_15 = "2025-09-16T10:00:00Z"
S2025_20 = "2025-09-25T10:00:00Z"
S2022_20 = "2022-09-22T10:00:00Z"
S2023_15 = "2023-09-15T10:00:00Z"
OUT_OF_WINDOW = "2025-09-19T10:00:00Z"


def _bucket(name, cap=None):
    b = next(b for b in bp.BUCKETS if b.name == name)
    return b if cap is None else bp.Bucket(b.name, b.pattern, b.sources, cap, b.non_reciprocal)


def _select(records, phase0=(), buckets=None, fixed=()):
    return bp.select_pool({r["ticket_id"]: r for r in records}, set(phase0),
                          fixed_ids=tuple(fixed), buckets=buckets or bp.BUCKETS)


# ---------------------------------------------------------------------------
# shared patterns — one place, not retyped
# ---------------------------------------------------------------------------
def test_builder_reuses_the_shared_patterns_and_extraction():
    src = (_LABELING_DIR / "build_phase2_pool.py").read_text(encoding="utf-8")
    assert "re.compile" not in src  # no retyped regex
    assert "from candidate_patterns import" in src
    assert "from ticket_extraction import" in src
    assert "from label_phase2 import" in src  # the --phase2 visibility rule


def test_shared_patterns_have_the_step0_keys():
    assert list(cp.PATTERNS) == ["a_wrong_paper", "b_score_vs_decision", "c_misunderstood",
                                 "d_llm_generated", "e_generic_reconsider", "ctx_reciprocal"]


@pytest.mark.parametrize("text, key", [(A_TEXT, "a_wrong_paper"), (B_TEXT, "b_score_vs_decision"),
                                       (C_TEXT, "c_misunderstood"), (D_TEXT, "d_llm_generated"),
                                       (E_TEXT, "e_generic_reconsider")])
def test_fixture_texts_hit_their_pattern(text, key):
    assert key in cp.pattern_hits(text)


def test_september_windows():
    assert cp.september_window(S2025_15) == cp.SEP_15_18
    assert cp.september_window("2021-09-18T23:59:59Z") == cp.SEP_15_18
    assert cp.september_window("2024-09-20T00:00:00Z") == cp.SEP_20_30
    assert cp.september_window(OUT_OF_WINDOW) == cp.OTHER


# ---------------------------------------------------------------------------
# bucket sources and filters
# ---------------------------------------------------------------------------
def test_a_bucket_sources_exclude_2025_sept_20_30():
    sel = _select([_rec(1, S2025_15, A_TEXT), _rec(2, S2022_20, A_TEXT), _rec(3, S2025_20, A_TEXT)],
                  buckets=(_bucket("a_wrong_paper"),))
    assert set(sel.pool_ids) == {1, 2}


def test_b_bucket_adds_2025_sept_20_30():
    sel = _select([_rec(1, S2025_15, B_TEXT), _rec(3, S2025_20, B_TEXT)],
                  buckets=(_bucket("b_score_vs_decision"),))
    assert set(sel.pool_ids) == {1, 3}


def test_out_of_window_tickets_are_never_picked():
    sel = _select([_rec(1, OUT_OF_WINDOW, A_TEXT), _rec(2, OUT_OF_WINDOW, PLAIN)])
    assert sel.pool_ids == []


def test_a_and_b_drop_reciprocal_mentions_but_c_d_e_keep_them():
    recip = " reciprocal reviewer"
    recs = [_rec(1, S2025_15, A_TEXT + recip), _rec(2, S2025_15, B_TEXT + recip),
            _rec(3, S2025_15, C_TEXT + recip), _rec(4, S2025_15, D_TEXT + recip),
            _rec(5, S2025_15, E_TEXT + recip)]
    buckets = tuple(_bucket(n) for n in ("a_wrong_paper", "b_score_vs_decision", "c_misunderstood",
                                         "d_llm_generated", "e_generic_reconsider"))
    sel = _select(recs, buckets=buckets)
    assert set(sel.pool_ids) == {3, 4, 5}


def test_random_slice_takes_any_content():
    sel = _select([_rec(1, S2023_15, PLAIN)], buckets=(_bucket("random"),))
    assert sel.pool_ids == [1] and sel.bucket_of[1] == "random"


def test_caps_are_respected():
    recs = [_rec(i, S2025_15, A_TEXT) for i in range(1, 11)]
    sel = _select(recs, buckets=(_bucket("a_wrong_paper", cap=3),))
    assert len(sel.pool_ids) == 3 and sel.stats["a_wrong_paper"].first_draw == 3


# ---------------------------------------------------------------------------
# Phase 0 exclusion and the fixed ids
# ---------------------------------------------------------------------------
def test_phase0_labeled_tickets_are_excluded_except_the_fixed_ids():
    recs = [_rec(1, S2025_20, B_TEXT), _rec(2, S2025_20, B_TEXT), _rec(9, S2025_20, PLAIN)]
    sel = _select(recs, phase0={1, 9}, fixed=(9,))
    assert 1 not in sel.pool_ids
    assert 9 in sel.pool_ids and sel.bucket_of[9] == "fixed"
    assert 2 in sel.pool_ids


def test_fixed_ids_are_included_even_out_of_window():
    sel = _select([_rec(9, OUT_OF_WINDOW, PLAIN)], fixed=(9,))
    assert sel.pool_ids == [9]


def test_missing_fixed_id_is_reported_not_invented():
    sel = _select([], fixed=(9,))
    assert sel.pool_ids == [] and sel.fixed_missing == [9]


# ---------------------------------------------------------------------------
# dedupe, visibility, refill
# ---------------------------------------------------------------------------
def test_a_ticket_lands_in_the_first_bucket_and_the_later_one_refills():
    both = A_TEXT + " " + B_TEXT
    recs = [_rec(1, S2025_15, both), _rec(2, S2025_15, B_TEXT)]
    buckets = (_bucket("a_wrong_paper", cap=1), _bucket("b_score_vs_decision", cap=1))
    sel = _select(recs, buckets=buckets)
    assert sel.bucket_of == {1: "a_wrong_paper", 2: "b_score_vs_decision"}
    assert sel.stats["b_score_vs_decision"].picked == 1


@pytest.mark.parametrize("thread", [
    [],
    [_msg("agent", body=A_TEXT)],
    [_msg("requester", public=False, body=A_TEXT)],
    [_msg("unknown", body=A_TEXT)],
    [_msg("other_end_user", body=A_TEXT)],  # co-author/staff: hidden under D72
])
def test_nothing_visible_under_phase2_rules_is_excluded_counted_and_refilled(thread):
    recs = [_rec(1, S2025_15, A_TEXT, thread=thread), _rec(2, S2025_15, A_TEXT)]
    sel = _select(recs, buckets=(_bucket("a_wrong_paper", cap=2),))
    assert sel.pool_ids == [2]
    assert sel.excluded_no_visible == {1}
    assert sel.stats["a_wrong_paper"].skipped_no_visible == 1


def test_fixed_id_with_nothing_visible_is_excluded_and_counted():
    sel = _select([_rec(9, S2025_15, PLAIN, thread=[_msg("agent")])], fixed=(9,))
    assert sel.pool_ids == [] and sel.excluded_no_visible == {9}


def test_refill_reaches_the_cap_when_supply_allows():
    recs = [_rec(1, S2025_15, A_TEXT, thread=[_msg("agent")])] + [
        _rec(i, S2025_15, A_TEXT) for i in range(2, 6)]
    sel = _select(recs, buckets=(_bucket("a_wrong_paper", cap=4),))
    assert len(sel.pool_ids) == 4 and 1 not in sel.pool_ids


# ---------------------------------------------------------------------------
# determinism
# ---------------------------------------------------------------------------
def _many():
    return ([_rec(i, S2025_15, A_TEXT) for i in range(100, 160)]
            + [_rec(i, S2023_15, PLAIN) for i in range(200, 260)])


def test_selection_is_deterministic():
    assert _select(_many()).pool_ids == _select(_many()).pool_ids


def test_resizing_one_bucket_never_reshuffles_another():
    """Per-bucket seeded RNGs: changing the random slice leaves the a-picks alone."""
    small = (_bucket("a_wrong_paper", cap=5), _bucket("random", cap=2))
    large = (_bucket("a_wrong_paper", cap=5), _bucket("random", cap=9))
    a1 = {t for t, b in _select(_many(), buckets=small).bucket_of.items() if b == "a_wrong_paper"}
    a2 = {t for t, b in _select(_many(), buckets=large).bucket_of.items() if b == "a_wrong_paper"}
    assert a1 == a2


def test_adding_an_earlier_bucket_never_reshuffles_a_later_one():
    """The D36 hazard: with ONE RNG consumed bucket by bucket, an earlier
    bucket's shuffle shifts every later draw. The earlier bucket here has
    candidates disjoint from the later one's, so dedupe cannot explain a change;
    only shared randomness could."""
    recs = ([_rec(i, S2025_15, A_TEXT) for i in range(100, 160)]
            + [_rec(i, S2025_15, D_TEXT) for i in range(300, 340)])
    alone = _select(recs, buckets=(_bucket("a_wrong_paper", cap=5),))
    with_d_first = _select(recs, buckets=(_bucket("d_llm_generated", cap=5),
                                          _bucket("a_wrong_paper", cap=5)))
    a_alone = {t for t, b in alone.bucket_of.items() if b == "a_wrong_paper"}
    a_after = {t for t, b in with_d_first.bucket_of.items() if b == "a_wrong_paper"}
    assert a_alone == a_after


def test_final_order_is_a_shuffle_of_the_picks():
    sel = _select(_many())
    assert sorted(sel.pool_ids) != sel.pool_ids
    assert len(set(sel.pool_ids)) == len(sel.pool_ids)


# ---------------------------------------------------------------------------
# sidecar + outputs
# ---------------------------------------------------------------------------
def test_sidecar_carries_membership_the_pool_records_do_not():
    recs = {r["ticket_id"]: r for r in [_rec(1, S2025_15, A_TEXT + " Please reconsider.")]}
    sel = bp.select_pool(recs, set(), fixed_ids=())
    rows = bp.sidecar_rows(sel, recs)
    assert rows == [{"ticket_id": 1, "bucket": "a_wrong_paper",
                     "pattern_hits": ["a_wrong_paper", "e_generic_reconsider"],
                     "mentions_reciprocal": False, "year": 2025, "window": cp.SEP_15_18}]
    assert not {"bucket", "pattern_hits", "mentions_reciprocal"} & set(recs[1])


def test_refuses_to_overwrite_an_existing_pool(tmp_path):
    path = tmp_path / "pool.jsonl"
    path.write_text("{}\n", encoding="utf-8")
    with pytest.raises(SystemExit):
        bp.refuse_overwrite(path)
    assert path.read_text(encoding="utf-8") == "{}\n"


def test_refuses_a_path_git_does_not_ignore(tmp_path):
    with pytest.raises(SystemExit):
        bp.assert_gitignored(tmp_path / "pool.jsonl")


def test_phase0_labeled_ids_reads_only_labeled_rows(tmp_path):
    path = tmp_path / "p0.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in [
        {"ticket_id": 1, "is_reject_appeal": True}, {"ticket_id": 2, "is_reject_appeal": False},
        {"ticket_id": 3, "is_reject_appeal": None}]), encoding="utf-8")
    assert bp.phase0_labeled_ids(path) == {1, 2}


# ---------------------------------------------------------------------------
# D72 amendment 2 — drop tickets FILED by a high-frequency account
# ---------------------------------------------------------------------------
STAFF, OTHER_STAFFISH, AUTHOR = 700, 701, 702


def _archive_with_spread(n_staff: int, n_other: int):
    """STAFF comments as other_end_user on n_staff tickets; OTHER_STAFFISH on n_other.

    Each ticket is filed by AUTHOR, so every STAFF/OTHER comment is classified
    other_end_user by ticket_extraction.sender_type.
    """
    users = {u: {"id": u, "role": "end-user", "email": None} for u in (STAFF, OTHER_STAFFISH, AUTHOR)}
    tickets, comments = {}, {}
    for i in range(max(n_staff, n_other)):
        tid = 5000 + i
        tickets[tid] = {"id": tid, "created_at": S2025_15, "status": "closed", "subject": "",
                        "requester_id": AUTHOR, "via": {"channel": "email"}}
        cs = [{"ticket_id": tid, "created_at": "t0", "author_id": AUTHOR, "public": True, "body": "q"}]
        if i < n_staff:
            cs.append({"ticket_id": tid, "created_at": "t1", "author_id": STAFF, "public": True, "body": "a"})
        if i < n_other:
            cs.append({"ticket_id": tid, "created_at": "t1", "author_id": OTHER_STAFFISH, "public": True, "body": "a"})
        comments[tid] = cs
    return te.Archive(tickets, sorted(tickets), users, comments, None)


def test_cutoff_is_ten_distinct_tickets_inclusive():
    arc = _archive_with_spread(n_staff=10, n_other=9)
    assert bp.STAFF_TICKET_CUTOFF == 10
    assert bp.high_frequency_accounts(arc) == {STAFF}  # 10 counts, 9 does not


def test_only_other_end_user_comments_count_toward_the_spread():
    """Filing your own tickets (requester) or answering as an agent does not count."""
    arc = _archive_with_spread(n_staff=0, n_other=0)
    for i in range(12):
        tid = 6000 + i
        arc.tickets[tid] = {"id": tid, "created_at": S2025_15, "status": "closed", "subject": "",
                            "requester_id": STAFF, "via": {"channel": "email"}}
        arc.comments[tid] = [{"ticket_id": tid, "created_at": "t0", "author_id": STAFF,
                              "public": True, "body": "q"}]
    assert bp.high_frequency_accounts(arc) == set()


def test_staff_requested_tickets_are_those_filed_by_an_account():
    arc = _archive_with_spread(n_staff=10, n_other=0)
    arc.tickets[9999] = {"id": 9999, "created_at": S2025_15, "status": "closed", "subject": "",
                         "requester_id": STAFF, "via": {"channel": "email"}}
    assert bp.staff_requested_tickets(arc, {STAFF}) == frozenset({9999})


def test_staff_requested_candidate_is_dropped_counted_and_refilled():
    recs = [_rec(1, S2025_15, A_TEXT), _rec(2, S2025_15, A_TEXT), _rec(3, S2025_15, A_TEXT)]
    sel = bp.select_pool({r["ticket_id"]: r for r in recs}, set(), fixed_ids=(),
                         buckets=(_bucket("a_wrong_paper", cap=2),), staff_requested=frozenset({1}))
    assert sorted(sel.pool_ids) == [2, 3]  # the refill reaches the cap
    assert sel.excluded_staff_requester == {1}
    assert sel.stats["a_wrong_paper"].skipped_staff_requester == 1


def test_fixed_id_with_a_staff_requester_stops_before_selecting_anything():
    recs = {9: _rec(9, S2025_15, PLAIN), 1: _rec(1, S2025_15, A_TEXT)}
    with pytest.raises(bp.FixedStaffRequester) as exc:
        bp.select_pool(recs, set(), fixed_ids=(9,), staff_requested=frozenset({9}))
    assert exc.value.ids == [9]


def test_no_staff_set_leaves_the_selection_unchanged():
    base = _select(_many())
    same = bp.select_pool({r["ticket_id"]: r for r in _many()}, set(), fixed_ids=(),
                          staff_requested=frozenset())
    assert base.pool_ids == same.pool_ids
