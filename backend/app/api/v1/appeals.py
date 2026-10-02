"""Appeals API (v1) — exports of the stored appeal classifications.

Mounted under ``/api/v1``, so the public path is
``/api/v1/appeals/phase1/export.csv``. The CSV is built by
``app.exports.phase1_appeals``, the same builder the CLI
(``scripts/export_phase1_appeals.py``) uses.
"""

from fastapi import APIRouter, Depends
from fastapi.responses import Response
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.database import get_db
from app.exports.phase1_appeals import build_phase1_export_csv

router = APIRouter(prefix="/appeals", tags=["appeals"])

_EXPORT_FILENAME = "phase1_appeals.csv"


@router.get("/phase1/export.csv")
async def export_phase1_appeals(db: AsyncSession = Depends(get_db)) -> Response:
    """Every phase-1 appeal row as CSV, one line per appealed paper."""
    return Response(
        content=await build_phase1_export_csv(db),
        media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="{_EXPORT_FILENAME}"'},
    )
