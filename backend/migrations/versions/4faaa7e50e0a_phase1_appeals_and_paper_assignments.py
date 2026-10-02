"""add paper_assignments and phase1_appeals tables

``paper_assignments`` holds the program committee's assignment sheet: one row
per submission number with its APC and OpenReview forum link (plus the parsed
forum id, indexed, so a forum link found in an email maps back to a number).
Loaded by ``scripts/load_paper_assignments.py``.

``phase1_appeals`` holds the phase-1 rejection appeal classifier's result, one
row per appealed paper per email. ``email_id`` cascades on delete so rows die
with their email. The set of rows for an email is replaced wholesale on every
reprocess, so there is no uniqueness constraint on (email_id, submission_number).

Both tables are new and additive; downgrade drops them. Plain create_table /
create_index (no batch mode needed for new tables), Postgres-safe types only.

Revision ID: 4faaa7e50e0a
Revises: c9f3a1b7d204
Create Date: 2026-10-02 12:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '4faaa7e50e0a'
down_revision: Union[str, Sequence[str], None] = 'c9f3a1b7d204'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema — create both tables and their indexes."""
    op.create_table(
        'paper_assignments',
        sa.Column('paper_number', sa.String(length=16), primary_key=True),
        sa.Column('apc_name', sa.String(length=255), nullable=False),
        sa.Column('openreview_url', sa.String(length=512), nullable=False),
        sa.Column('openreview_forum_id', sa.String(length=32), nullable=True),
        sa.Column(
            'cycle',
            sa.String(length=32),
            nullable=False,
            server_default='AAAI-27',
        ),
        sa.Column(
            'created_at',
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.Column(
            'updated_at',
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
    )
    op.create_index(
        'ix_paper_assignments_openreview_forum_id',
        'paper_assignments',
        ['openreview_forum_id'],
        unique=False,
    )

    op.create_table(
        'phase1_appeals',
        sa.Column('id', sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column(
            'email_id',
            sa.Integer(),
            sa.ForeignKey('emails.id', ondelete='CASCADE'),
            nullable=False,
        ),
        sa.Column('zendesk_ticket_id', sa.BigInteger(), nullable=True),
        sa.Column('submission_number', sa.String(length=16), nullable=True),
        sa.Column('apc_name', sa.String(length=255), nullable=True),
        sa.Column('openreview_url', sa.String(length=512), nullable=True),
        sa.Column('relation', sa.String(length=16), nullable=False),
        sa.Column('reasons', sa.JSON(), nullable=False),
        sa.Column('must_verify', sa.Boolean(), nullable=False),
        sa.Column('prompt_sha256', sa.String(length=64), nullable=False),
        sa.Column('model', sa.String(length=128), nullable=True),
        sa.Column(
            'classified_at',
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
    )
    op.create_index(
        'ix_phase1_appeals_email_id', 'phase1_appeals', ['email_id'], unique=False
    )
    op.create_index(
        'ix_phase1_appeals_zendesk_ticket_id',
        'phase1_appeals',
        ['zendesk_ticket_id'],
        unique=False,
    )
    op.create_index(
        'ix_phase1_appeals_submission_number',
        'phase1_appeals',
        ['submission_number'],
        unique=False,
    )


def downgrade() -> None:
    """Downgrade schema — drop both tables (indexes first)."""
    op.drop_index('ix_phase1_appeals_submission_number', table_name='phase1_appeals')
    op.drop_index('ix_phase1_appeals_zendesk_ticket_id', table_name='phase1_appeals')
    op.drop_index('ix_phase1_appeals_email_id', table_name='phase1_appeals')
    op.drop_table('phase1_appeals')
    op.drop_index(
        'ix_paper_assignments_openreview_forum_id', table_name='paper_assignments'
    )
    op.drop_table('paper_assignments')
