"""Export the ``phase1_appeals`` rows as CSV (same file as the API export).

Writes to ``--out`` when given, else to stdout. The file holds email subjects and
bodies, so keep it out of the repository.

Run with:  cd backend && python scripts/export_phase1_appeals.py [--out PATH]
"""

import argparse
import asyncio
import csv
import io
import sys
from pathlib import Path

# scripts/export_phase1_appeals.py -> parents[1] is backend/ (put it on
# sys.path so `app` imports work when run as a script).
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.exports.phase1_appeals import build_phase1_export_csv  # noqa: E402


async def export(session_factory) -> str:
    """The CSV text, read in one session."""
    async with session_factory() as db:
        return await build_phase1_export_csv(db)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    from app.db.database import async_session_factory

    text = asyncio.run(export(async_session_factory))
    if args.out is None:
        sys.stdout.write(text)
    else:
        # newline="": the csv module already wrote its own line endings.
        with open(args.out, "w", encoding="utf-8", newline="") as fh:
            fh.write(text)
        rows = max(len(list(csv.reader(io.StringIO(text)))) - 1, 0)
        print(f"Wrote {rows} row(s) to {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
