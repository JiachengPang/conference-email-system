"""Load the program committee's paper-assignment sheet into ``paper_assignments``.

The CSV is not in the repo; pass its path. Expected columns (header row):

    Assignment    the APC's name
    Paper number  the submission number
    Paper URL     the OpenReview forum link, e.g. https://openreview.net/forum?id=<id>

Values are stripped. The forum id is the URL's ``id`` query parameter (NULL when
absent). Rows missing a paper number, an APC, or a URL are skipped and counted.
Re-running is safe: rows upsert by paper number.

Run with:  cd backend && python scripts/load_paper_assignments.py <csv_path> [--cycle AAAI-27]
"""

import argparse
import asyncio
import csv
import sys
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import parse_qs, urlparse

# scripts/load_paper_assignments.py -> parents[1] is backend/ (put it on
# sys.path so `app` imports work when run as a script).
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.repositories.phase1_appeal_repository import (  # noqa: E402
    PaperAssignmentRepository,
)

DEFAULT_CYCLE = "AAAI-27"

COL_APC = "Assignment"
COL_NUMBER = "Paper number"
COL_URL = "Paper URL"

# Matches PaperAssignment.openreview_forum_id's column width.
_FORUM_ID_MAX_LEN = 32


@dataclass
class ParsedSheet:
    rows: list[dict] = field(default_factory=list)
    read: int = 0
    skipped: int = 0
    without_forum_id: int = 0


def parse_forum_id(url: str | None) -> str | None:
    """The ``id`` query parameter of an OpenReview URL, or ``None``."""
    if not url:
        return None
    values = parse_qs(urlparse(url.strip()).query).get("id")
    if not values:
        return None
    forum_id = values[0].strip()
    if not forum_id or len(forum_id) > _FORUM_ID_MAX_LEN:
        return None
    return forum_id


def parse_sheet(csv_path: Path, cycle: str = DEFAULT_CYCLE) -> ParsedSheet:
    """Read the CSV into repository rows (no database access)."""
    sheet = ParsedSheet()
    # utf-8-sig: spreadsheet exports often lead with a byte-order mark, which
    # would otherwise glue itself onto the first header name.
    with open(csv_path, newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        missing = {COL_APC, COL_NUMBER, COL_URL} - {
            (name or "").strip() for name in (reader.fieldnames or [])
        }
        if missing:
            raise ValueError(f"CSV is missing column(s): {', '.join(sorted(missing))}")
        for raw in reader:
            record = {(k or "").strip(): (v or "").strip() for k, v in raw.items()}
            sheet.read += 1
            number = record.get(COL_NUMBER, "")
            apc = record.get(COL_APC, "")
            url = record.get(COL_URL, "")
            if not number or not apc or not url:
                sheet.skipped += 1
                continue
            forum_id = parse_forum_id(url)
            if forum_id is None:
                sheet.without_forum_id += 1
            sheet.rows.append(
                {
                    "paper_number": number,
                    "apc_name": apc,
                    "openreview_url": url,
                    "openreview_forum_id": forum_id,
                    "cycle": cycle,
                }
            )
    return sheet


async def load(csv_path: Path, cycle: str, session_factory) -> tuple[ParsedSheet, int]:
    """Parse ``csv_path`` and upsert it; returns the parse stats and rows written."""
    sheet = parse_sheet(csv_path, cycle)
    async with session_factory() as db:
        written = await PaperAssignmentRepository().upsert_many(db, sheet.rows)
    return sheet, written


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("csv_path", type=Path)
    parser.add_argument("--cycle", default=DEFAULT_CYCLE)
    args = parser.parse_args(argv)

    from app.db.database import async_session_factory

    sheet, written = asyncio.run(load(args.csv_path, args.cycle, async_session_factory))
    for label, value in (
        ("Rows read", sheet.read),
        ("Rows skipped", sheet.skipped),
        ("Rows without forum id", sheet.without_forum_id),
        ("Papers upserted", written),
    ):
        print(f"{label + ':':<23}{value}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
