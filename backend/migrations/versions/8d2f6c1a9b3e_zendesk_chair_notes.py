"""add zendesk_chair_notes table

``zendesk_chair_notes`` records the one Zendesk internal note ConfMail posts for
an email (chair notes, Z2): which ticket, the rendered mode and a sha256 of the
note body, the claim state (pending / posting / posted / failed) and the number
of attempts (at most 3). ``email_id`` is UNIQUE, which is what limits an email
to a single note, and cascades on delete so a row dies with its email.

New and additive; downgrade drops it. Plain create_table / create_index, with
named constraints so SQLite and Postgres get the same schema.

Revision ID: 8d2f6c1a9b3e
Revises: 4faaa7e50e0a
Create Date: 2026-10-03 12:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '8d2f6c1a9b3e'
down_revision: Union[str, Sequence[str], None] = '4faaa7e50e0a'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema — create the table and its indexes."""
    op.create_table(
        'zendesk_chair_notes',
        sa.Column('id', sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column(
            'email_id',
            sa.Integer(),
            sa.ForeignKey('emails.id', ondelete='CASCADE'),
            nullable=False,
        ),
        sa.Column('zendesk_ticket_id', sa.BigInteger(), nullable=False),
        sa.Column('mode', sa.String(length=32), nullable=True),
        sa.Column('body_sha256', sa.String(length=64), nullable=True),
        sa.Column(
            'status',
            sa.String(length=16),
            nullable=False,
            server_default='pending',
        ),
        sa.Column(
            'attempts', sa.Integer(), nullable=False, server_default='0'
        ),
        sa.Column('last_error', sa.Text(), nullable=True),
        sa.Column('zendesk_audit_id', sa.BigInteger(), nullable=True),
        sa.Column('claimed_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('posted_at', sa.DateTime(timezone=True), nullable=True),
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
        sa.UniqueConstraint('email_id', name='uq_zendesk_chair_notes_email_id'),
        sa.CheckConstraint(
            "status IN ('pending', 'posting', 'posted', 'failed')",
            name='ck_zendesk_chair_notes_status',
        ),
        sa.CheckConstraint(
            'attempts >= 0 AND attempts <= 3',
            name='ck_zendesk_chair_notes_attempts',
        ),
    )
    op.create_index(
        'ix_zendesk_chair_notes_zendesk_ticket_id',
        'zendesk_chair_notes',
        ['zendesk_ticket_id'],
        unique=False,
    )
    op.create_index(
        'ix_zendesk_chair_notes_status',
        'zendesk_chair_notes',
        ['status'],
        unique=False,
    )


def downgrade() -> None:
    """Downgrade schema — drop the table (indexes first)."""
    op.drop_index('ix_zendesk_chair_notes_status', table_name='zendesk_chair_notes')
    op.drop_index(
        'ix_zendesk_chair_notes_zendesk_ticket_id', table_name='zendesk_chair_notes'
    )
    op.drop_table('zendesk_chair_notes')
