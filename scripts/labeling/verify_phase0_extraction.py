#!/usr/bin/env python
"""Verify ticket_extraction.py by regenerating the Phase 0 window.

    python scripts/labeling/verify_phase0_extraction.py

Rebuilds 2025-09-20..30 into a SCRATCH file under data/labeling/, then compares
it field by field with the existing Phase 0 file, ignoring only the label fields
(is_reject_appeal, appeal_reason, deferred). The existing file is opened
READ-ONLY and is never written.

Output is ids, field names and counts ONLY: no ticket text, names, emails or
titles. Exits 0 on zero mismatches, 1 otherwise.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from ticket_extraction import (  # noqa: E402
    PHASE0_LABEL_FIELDS,
    REPO_ROOT,
    build_records,
    load_archive,
    select_window,
)

PHASE0_PATH = REPO_ROOT / "data" / "labeling" / "phase1_appeals_2025-09-20_to_30.jsonl"
SCRATCH_PATH = REPO_ROOT / "data" / "labeling" / "_scratch_phase0_regen.jsonl"
WINDOW = ("2025-09-20", "2025-09-30")
LABEL_FIELDS = {"is_reject_appeal", "appeal_reason", "deferred"}
THREAD_SUBFIELDS = ("sender_type", "sender_email_or_null", "body", "created_at", "is_public")
MAX_IDS = 10


def assert_gitignored(path: Path) -> None:
    """Refuse to write unless git confirms the path is ignored."""
    result = subprocess.run(
        ["git", "check-ignore", "-q", str(path)], cwd=REPO_ROOT, capture_output=True
    )
    if result.returncode != 0:
        raise SystemExit(f"REFUSING TO WRITE: {path.name} is not gitignored.")


def write_jsonl(path: Path, records: list[dict]) -> None:
    if path.resolve() == PHASE0_PATH.resolve():
        raise SystemExit("REFUSING: the scratch path is the Phase 0 file.")
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        for r in records:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def compare(regen: list[dict], original: list[dict]) -> int:
    """Print the report. Returns the total number of mismatching checks."""
    total = 0

    def report(name: str, bad: list[int]) -> None:
        nonlocal total
        total += len(bad)
        print(f"  {name:<34} {len(bad):>4} mismatching records   {sorted(bad)[:MAX_IDS]}")

    ro = {r["ticket_id"]: r for r in regen}
    oo = {r["ticket_id"]: r for r in original}
    print(f"records: regenerated {len(regen)}, original {len(original)}  "
          f"-> {'MATCH' if len(regen) == len(original) else 'MISMATCH'}")
    id_ok = set(ro) == set(oo)
    print(f"id set: {'MATCH' if id_ok else 'MISMATCH'}  "
          f"(only regenerated {sorted(set(ro) - set(oo))[:MAX_IDS]}, "
          f"only original {sorted(set(oo) - set(ro))[:MAX_IDS]})")
    total += 0 if id_ok else len(set(ro) ^ set(oo))
    order_ok = [r["ticket_id"] for r in regen] == [r["ticket_id"] for r in original]
    print(f"record order: {'MATCH' if order_ok else 'MISMATCH'}")
    total += 0 if order_ok else 1

    shared = sorted(set(ro) & set(oo))
    fields = [k for k in original[0] if k not in LABEL_FIELDS] if original else []
    extra_regen = sorted({k for r in regen for k in r} - {k for r in original for k in r} - LABEL_FIELDS)
    extra_orig = sorted({k for r in original for k in r} - {k for r in regen for k in r} - LABEL_FIELDS)
    print(f"fields only in regenerated: {extra_regen}; only in original: {extra_orig}")
    total += len(extra_regen) + len(extra_orig)

    print("per field (label fields ignored):")
    report("key order (non-label keys)", [
        t for t in shared
        if [k for k in ro[t] if k not in LABEL_FIELDS] != [k for k in oo[t] if k not in LABEL_FIELDS]
    ])
    for f in fields:
        if f == "thread":
            continue
        report(f, [t for t in shared if ro[t].get(f) != oo[t].get(f)])

    print("thread[] sub-fields:")
    report("thread count", [t for t in shared if len(ro[t]["thread"]) != len(oo[t]["thread"])])
    same_len = [t for t in shared if len(ro[t]["thread"]) == len(oo[t]["thread"])]
    report("thread order (created_at sequence)", [
        t for t in same_len
        if [e["created_at"] for e in ro[t]["thread"]] != [e["created_at"] for e in oo[t]["thread"]]
    ])
    for sub in THREAD_SUBFIELDS:
        label = "public" if sub == "is_public" else sub
        report(f"thread.{label}", [
            t for t in same_len
            if [e.get(sub) for e in ro[t]["thread"]] != [e.get(sub) for e in oo[t]["thread"]]
        ])
    report("thread entry key order", [
        t for t in same_len
        if [list(e) for e in ro[t]["thread"]] != [list(e) for e in oo[t]["thread"]]
    ])

    def strip(r):
        return {k: v for k, v in r.items() if k not in LABEL_FIELDS}

    report("WHOLE RECORD (label fields removed)", [t for t in shared if strip(ro[t]) != strip(oo[t])])
    return total


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass
    assert_gitignored(SCRATCH_PATH)
    archive = load_archive()
    ids = select_window(archive, *WINDOW)
    regen = build_records(archive, ids, label_fields=PHASE0_LABEL_FIELDS)
    write_jsonl(SCRATCH_PATH, regen)
    print(f"wrote scratch file: {SCRATCH_PATH.relative_to(REPO_ROOT)} ({len(regen)} records)")
    mismatches = compare(regen, read_jsonl(PHASE0_PATH))
    print(f"\nTOTAL MISMATCHES: {mismatches}")
    return 0 if mismatches == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
