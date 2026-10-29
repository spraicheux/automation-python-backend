"""
R-12 persistent price events.

Alerts (api/alerts.list_alerts) are a live snapshot — they describe the
current state of each peer group. For the acceptance test Samuel asked us
to show the *history* of trading events: when batch 02 landed, Hennessy T2
moved €25 → €24 (A01 → B01); when batch 05 later landed, it moved
€24 → €23.50 (B01 → C01). Both are facts, both need an event ID, and
"navigation and replay behaviour" has to show the chain, not just the
latest snapshot.

PriceEventDB stores one row per detected trading event. The processor
calls `record_price_events(db, …)` after each batch commits; dedup is on
(peer_group_id, current_uid, previous_uid) so re-ingesting the same batch
cannot double-count. The /api/events endpoint lists them, newest first,
with full lineage.
"""
import uuid
from datetime import datetime

from sqlalchemy import Column, String, Float, DateTime, Text, Index

from core.database import Base


class PriceEventDB(Base):
    __tablename__ = "price_events"

    id = Column(String(64), primary_key=True, default=lambda: str(uuid.uuid4()))

    # The commercial identity the event lives in.
    peer_group_id = Column(String(512), nullable=False, index=True)

    # Event type: "new_best_price" | "competitive_supplier"
    event_type = Column(String(64), nullable=False)

    # Current offer (the row that announced the event).
    current_uid           = Column(String(64), nullable=False, index=True)
    current_supplier      = Column(String(255), nullable=True)
    current_price_eur     = Column(Float,       nullable=True)
    current_offer_date    = Column(DateTime,    nullable=True)
    current_source_file   = Column(String(512), nullable=True)

    # Previous best (the row the event improved on).
    previous_uid          = Column(String(64), nullable=True, index=True)
    previous_supplier     = Column(String(255), nullable=True)
    previous_price_eur    = Column(Float,       nullable=True)
    previous_offer_date   = Column(DateTime,    nullable=True)
    previous_source_file  = Column(String(512), nullable=True)

    # Magnitude.
    delta_pct             = Column(Float, nullable=True)
    delta_abs             = Column(Float, nullable=True)

    # Narrative + full snapshot (JSON string) for audit replay.
    message               = Column(Text, nullable=True)
    peer_snapshot_json    = Column(Text, nullable=True)

    # When we observed it.
    created_at            = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)

    __table_args__ = (
        Index("ix_price_events_pg_cur_prev", "peer_group_id", "current_uid", "previous_uid", unique=True),
    )
