#!/usr/bin/env python
"""Build the Phase 2 reason-labeling pool (reject_appeal.md, Step 3b-2, D63).

    python scripts/labeling/build_phase2_pool.py

Writes TWO gitignored files:
  * data/labeling/phase2_reasons_pool.jsonl         — what the labeler opens
    (``label_appeals.py --phase2``). Phase 0 record shape with NO label fields
    (the tool adds its own), and NOTHING about why a ticket was picked.
  * data/labeling/phase2_reasons_pool_buckets.jsonl — the sidecar: which bucket
    picked each ticket and which patterns it hit, for within-pool scoring
    later. Never read by the labeling tool.

Refuses to overwrite an existing pool file: once labeled, it holds hours of
manual judgment.

Records come from ticket_extraction.py (the verified Phase 0 extraction).
Patterns come from candidate_patterns.py (the Step 0 regexes, one shared copy).
"Visible" messages use label_phase2.author_view, the exact --phase2 rule
(the requester's public messages only, D72), so a ticket with nothing to label
never reaches the labeler.

Output is ids, bucket names and counts ONLY.
"""

from __future__ import annotations

import json
import random
import subprocess
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from candidate_patterns import (  # noqa: E402
    RECIPROCAL_KEY,
    SEP_15_18,
    SEP_20_30,
    match_text,
    pattern_hits,
    september_window,
)
from label_appeals import DATA_PATH as PHASE0_PATH  # noqa: E402
from label_appeals import save  # noqa: E402
from label_phase2 import PHASE2_PATH, author_view  # noqa: E402
from ticket_extraction import (  # noqa: E402
    REPO_ROOT,
    Archive,
    build_records,
    load_archive,
    sender_type,
)

SIDECAR_PATH = PHASE2_PATH.with_name("phase2_reasons_pool_buckets.jsonl")
SEED = 42
FIXED_IDS: tuple[int, ...] = (18809, 18947, 18996, 19129)  # the four Phase 0 non-r appeals
FIXED_BUCKET = "fixed"

# D72 amendment 2. An account that authored comments classified
# `other_end_user` on at least this many DISTINCT tickets across the archive is
# staff-like: measured, every such account is on >= 10 tickets and every other
# one on <= 7, with none on 8-9. Tickets FILED by one are dropped from the pool
# (their visible "requester" text is staff-written) and refilled.
STAFF_TICKET_CUTOFF = 10

_OLD_SEPT = frozenset((y, w) for y in (2021, 2022, 2023, 2024) for w in (SEP_15_18, SEP_20_30))
SOURCES_A = frozenset({(2025, SEP_15_18)}) | _OLD_SEPT
SOURCES_B = SOURCES_A | {(2025, SEP_20_30)}  # "the same September windows"


@dataclass(frozen=True)
class Bucket:
    name: str
    pattern: str | None  # None = any content (the random slice)
    sources: frozenset
    cap: int
    non_reciprocal: bool = False


# Order matters: dedupe keeps a ticket in the FIRST bucket that picks it.
BUCKETS: tuple[Bucket, ...] = (
    Bucket("a_wrong_paper", "a_wrong_paper", SOURCES_A, 35, non_reciprocal=True),
    Bucket("b_score_vs_decision", "b_score_vs_decision", SOURCES_B, 35, non_reciprocal=True),
    Bucket("c_misunderstood", "c_misunderstood", SOURCES_B, 10),
    Bucket("d_llm_generated", "d_llm_generated", SOURCES_B, 10),
    Bucket("e_generic_reconsider", "e_generic_reconsider", SOURCES_B, 10),
    Bucket("random", None, SOURCES_B, 20),
)


@dataclass
class BucketStats:
    candidates: int = 0  # eligible tickets matching the bucket rule
    first_draw: int = 0  # min(cap, candidates) — before dedupe/refill
    skipped_overlap: int = 0  # already picked by an earlier bucket
    skipped_no_visible: int = 0  # zero visible author messages
    skipped_staff_requester: int = 0  # filed by a high-frequency account
    picked: int = 0


@dataclass
class Selection:
    pool_ids: list[int]  # final, shuffled
    bucket_of: dict[int, str]
    stats: dict[str, BucketStats] = field(default_factory=dict)
    excluded_no_visible: set[int] = field(default_factory=set)
    excluded_staff_requester: set[int] = field(default_factory=set)
    fixed_missing: list[int] = field(default_factory=list)


class FixedStaffRequester(Exception):
    """A fixed id was filed by a high-frequency account. Never dropped: stop."""

    def __init__(self, ids: list[int]):
        super().__init__(f"fixed ids with a high-frequency requester: {ids}")
        self.ids = ids


def high_frequency_accounts(archive: Archive, cutoff: int = STAFF_TICKET_CUTOFF) -> set[int]:
    """Accounts that commented as `other_end_user` on >= cutoff distinct tickets.

    Counted exactly as the D72 measurement did: a comment only counts when
    `ticket_extraction.sender_type` classifies it `other_end_user` on that
    ticket. Filing your own ticket (requester) or answering as an agent does not.
    """
    spread: dict[int, set[int]] = defaultdict(set)
    for tid, comments in archive.comments.items():
        ticket = archive.tickets.get(tid)
        if ticket is None:
            continue
        for c in comments:
            if sender_type(c, ticket, archive.users) == "other_end_user":
                spread[c["author_id"]].add(tid)
    return {a for a, tickets in spread.items() if len(tickets) >= cutoff}


def staff_requested_tickets(archive: Archive, accounts: set[int]) -> frozenset[int]:
    """Ticket ids whose requester is one of `accounts`."""
    return frozenset(t for t, tk in archive.tickets.items() if tk["requester_id"] in accounts)


def ticket_year(record: dict) -> int:
    return int(record["created_at"][:4])


def source_of(record: dict) -> tuple[int, str]:
    return ticket_year(record), september_window(record["created_at"])


def hits_of(record: dict) -> set[str]:
    return pattern_hits(match_text(record.get("subject"), record.get("initial_message_body")))


def has_visible_author_message(record: dict) -> bool:
    return bool(author_view(record)[0])


def select_pool(
    records: dict[int, dict],
    phase0_labeled: set[int],
    *,
    fixed_ids: tuple[int, ...] = FIXED_IDS,
    buckets: tuple[Bucket, ...] = BUCKETS,
    seed: int = SEED,
    staff_requested: frozenset[int] = frozenset(),
) -> Selection:
    """Pure selection. ``records`` must cover every source-window ticket + the fixed ids.

    Each bucket shuffles its sorted candidates with its OWN seeded RNG
    (``f"{seed}:{name}"``), so adding or resizing one bucket never re-rolls
    another. Walking the shuffle, a ticket already picked, filed by a
    high-frequency account (``staff_requested``), or with nothing to label is
    skipped and the next one taken (refill from the same source).

    Raises ``FixedStaffRequester`` — before selecting anything — if a fixed id
    is in ``staff_requested``: the fixed ids are never silently dropped.
    """
    bad_fixed = sorted(t for t in fixed_ids if t in staff_requested)
    if bad_fixed:
        raise FixedStaffRequester(bad_fixed)

    sel = Selection([], {})
    picked: set[int] = set()

    fixed = BucketStats()
    for tid in fixed_ids:
        fixed.candidates += 1
        fixed.first_draw += 1
        rec = records.get(tid)
        if rec is None:
            sel.fixed_missing.append(tid)
        elif not has_visible_author_message(rec):
            fixed.skipped_no_visible += 1
            sel.excluded_no_visible.add(tid)
        else:
            picked.add(tid)
            sel.bucket_of[tid] = FIXED_BUCKET
            fixed.picked += 1
    sel.stats[FIXED_BUCKET] = fixed

    hits = {t: hits_of(r) for t, r in records.items()}
    for bucket in buckets:
        st = BucketStats()
        cands = sorted(
            t for t, r in records.items()
            if t not in phase0_labeled
            and source_of(r) in bucket.sources
            and (bucket.pattern is None or bucket.pattern in hits[t])
            and not (bucket.non_reciprocal and RECIPROCAL_KEY in hits[t])
        )
        st.candidates = len(cands)
        st.first_draw = min(bucket.cap, len(cands))
        random.Random(f"{seed}:{bucket.name}").shuffle(cands)
        for tid in cands:
            if st.picked == bucket.cap:
                break
            if tid in picked:
                st.skipped_overlap += 1
                continue
            if tid in staff_requested:
                st.skipped_staff_requester += 1
                sel.excluded_staff_requester.add(tid)
                continue
            if not has_visible_author_message(records[tid]):
                st.skipped_no_visible += 1
                sel.excluded_no_visible.add(tid)
                continue
            picked.add(tid)
            sel.bucket_of[tid] = bucket.name
            st.picked += 1
        sel.stats[bucket.name] = st

    sel.pool_ids = sorted(picked)
    random.Random(seed).shuffle(sel.pool_ids)
    return sel


def sidecar_rows(sel: Selection, records: dict[int, dict]) -> list[dict]:
    rows = []
    for tid in sorted(sel.pool_ids):
        r = records[tid]
        h = hits_of(r)
        year, window = source_of(r)
        rows.append({
            "ticket_id": tid,
            "bucket": sel.bucket_of[tid],
            "pattern_hits": sorted(h - {RECIPROCAL_KEY}),
            "mentions_reciprocal": RECIPROCAL_KEY in h,
            "year": year,
            "window": window,
        })
    return rows


# --------------------------------------------------------------------------- I/O
def assert_gitignored(path: Path) -> None:
    result = subprocess.run(["git", "check-ignore", "-q", str(path)], cwd=REPO_ROOT,
                            capture_output=True)
    if result.returncode != 0:
        raise SystemExit(f"REFUSING TO WRITE: {path.name} is not gitignored.")


def refuse_overwrite(path: Path) -> None:
    if path.exists():
        raise SystemExit(
            f"REFUSING TO OVERWRITE {path.name}: it may already hold labels. "
            "Move it aside deliberately if a rebuild is really intended."
        )


def phase0_labeled_ids(path: Path = PHASE0_PATH) -> set[int]:
    """Ids of the Phase 0 LABELED tickets — ids and label state only."""
    with open(path, encoding="utf-8") as fh:
        return {r["ticket_id"] for r in map(json.loads, filter(str.strip, fh))
                if r.get("is_reject_appeal") is not None}


def report(sel: Selection, records: dict[int, dict], phase0_labeled: set[int]) -> None:
    print("per bucket (candidates -> first draw -> skipped overlap / staff-requester / no-visible -> picked):")
    for name, st in sel.stats.items():
        print(f"  {name:<22} {st.candidates:>5} -> {st.first_draw:>3} -> "
              f"{st.skipped_overlap:>3} / {st.skipped_staff_requester:>3} / {st.skipped_no_visible:>3}"
              f" -> {st.picked:>3}")
    if sel.fixed_missing:
        print(f"  ⚠️ fixed ids not in the archive: {sel.fixed_missing}")
    print(f"excluded for a high-frequency REQUESTER: {len(sel.excluded_staff_requester)} distinct tickets")
    print(f"excluded for ZERO visible author messages: {len(sel.excluded_no_visible)} distinct tickets")

    pool = [records[t] for t in sel.pool_ids]
    print("\nfinal pool by year / window:")
    for (y, w), n in sorted(Counter(source_of(r) for r in pool).items()):
        print(f"  {y} {w:<9} {n}")
    print("final pool by year x bucket:")
    by = defaultdict(Counter)
    for t in sel.pool_ids:
        by[ticket_year(records[t])][sel.bucket_of[t]] += 1
    for y in sorted(by):
        print(f"  {y}: {dict(sorted(by[y].items()))}")

    print("\noverlaps inside the final pool (tickets hitting BOTH patterns):")
    hits = {t: hits_of(records[t]) for t in sel.pool_ids}
    keys = sorted({k for h in hits.values() for k in h})
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            n = sum(1 for h in hits.values() if a in h and b in h)
            if n:
                print(f"  {a} & {b}: {n}")
    multi = sum(1 for h in hits.values() if len(h - {RECIPROCAL_KEY}) > 1)
    print(f"  tickets hitting >1 reason pattern: {multi}")
    print("reciprocal mentions per bucket:",
          dict(sorted(Counter(sel.bucket_of[t] for t in sel.pool_ids if RECIPROCAL_KEY in hits[t]).items())))

    print("\nsender_type per year (all thread messages in the pool; visible = public requester):")
    st_year = defaultdict(Counter)
    vis_year = Counter()
    for r in pool:
        y = ticket_year(r)
        for e in r["thread"]:
            st_year[y][e["sender_type"]] += 1
        vis_year[y] += len(author_view(r)[0])
    for y in sorted(st_year):
        total = sum(st_year[y].values())
        unk = st_year[y]["unknown"]
        print(f"  {y}: {dict(sorted(st_year[y].items()))}  unknown share {unk / total:.1%}  "
              f"visible (public requester) msgs {vis_year[y]}")

    in_phase0 = [t for t in sel.pool_ids if t in phase0_labeled]
    print(f"\npool tickets that are in the Phase 0 200: {sorted(in_phase0)} (must be only the fixed ids)")
    print(f"FINAL TOTAL: {len(sel.pool_ids)}")


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass
    for path in (PHASE2_PATH, SIDECAR_PATH):
        assert_gitignored(path)
        refuse_overwrite(path)

    archive = load_archive()
    wanted = set(FIXED_IDS) | {
        t for t, tk in archive.tickets.items()
        if (int(tk["created_at"][:4]), september_window(tk["created_at"])) in SOURCES_B
    }
    ids = sorted(wanted & set(archive.tickets))
    records = {r["ticket_id"]: r for r in build_records(archive, ids)}  # no label fields
    phase0 = phase0_labeled_ids()
    accounts = high_frequency_accounts(archive)
    staff_requested = staff_requested_tickets(archive, accounts)
    print(f"high-frequency accounts: cutoff >= {STAFF_TICKET_CUTOFF} distinct tickets as other_end_user "
          f"-> {len(accounts)} accounts; {len(staff_requested & set(ids))} source-window tickets filed by one")
    try:
        sel = select_pool(records, phase0, staff_requested=staff_requested)
    except FixedStaffRequester as exc:
        print(f"STOPPED, nothing written: {exc}. The fixed ids are never dropped; decide by hand.")
        return 2

    save([records[t] for t in sel.pool_ids], PHASE2_PATH)  # atomic, same serializer as the tool
    save(sidecar_rows(sel, records), SIDECAR_PATH)
    print(f"source-window tickets considered: {len(ids)} (Phase 0 labeled excluded: "
          f"{len(phase0 & set(ids)) - len(set(FIXED_IDS) & phase0)} besides the fixed ids)")
    report(sel, records, phase0)
    print(f"\nwrote {PHASE2_PATH.relative_to(REPO_ROOT)} and {SIDECAR_PATH.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
