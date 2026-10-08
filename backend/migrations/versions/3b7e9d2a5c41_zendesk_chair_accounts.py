"""add zendesk_chair_accounts table

``zendesk_chair_accounts`` maps each chair (APC), by the name exactly as the
paper-to-chair sheet has it, to that chair's Zendesk agent user id, with an
``active`` flag. It is what the reject-appeal assignment uses to set a ticket's
assignee. ``chair_name`` is UNIQUE; ``zendesk_user_id`` must be positive.

New and additive; downgrade drops it. Plain create_table with named
constraints so SQLite and Postgres get the same schema. The rows come only from
``scripts/load_chair_accounts.py`` (a private file outside the repo); this
migration seeds nothing.

Revision ID: 3b7e9d2a5c41
Revises: 8d2f6c1a9b3e
Create Date: 2026-10-07 12:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '3b7e9d2a5c41'
down_revision: Union[str, Sequence[str], None] = '8d2f6c1a9b3e'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema — create the table."""
    op.create_table(
        'zendesk_chair_accounts',
        sa.Column('id', sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column('chair_name', sa.String(length=255), nullable=False),
        sa.Column('zendesk_user_id', sa.BigInteger(), nullable=False),
        sa.Column('active', sa.Boolean(), nullable=False, server_default=sa.true()),
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
        sa.UniqueConstraint('chair_name', name='uq_zendesk_chair_accounts_chair_name'),
        sa.CheckConstraint(
            'zendesk_user_id > 0', name='ck_zendesk_chair_accounts_user_id'
        ),
    )


def downgrade() -> None:
    """Downgrade schema — drop the table."""
    op.drop_table('zendesk_chair_accounts')
