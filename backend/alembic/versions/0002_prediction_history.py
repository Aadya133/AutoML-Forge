"""Add prediction history table."""
from alembic import op
from app.db import Base
from app import entities  # noqa: F401

revision = "0002_prediction_history"
down_revision = "0001_initial"
branch_labels = None
depends_on = None


def upgrade():
    Base.metadata.tables["predictions"].create(bind=op.get_bind(), checkfirst=True)


def downgrade():
    op.drop_table("predictions")
