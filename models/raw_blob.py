import uuid
from datetime import datetime
from sqlalchemy import Column, String, DateTime, Integer, LargeBinary
from core.database import Base


class RawBlobDB(Base):
    """Durable archive of every inbound offer file's raw bytes.

    Samuel 2026-11-11: the 8 CHELSEA images and the Alder NE410 PDF were lost
    because they were only ever written to /home/offer_files, and /home on this
    App Service is EPHEMERAL (WEBSITES_ENABLE_APP_SERVICE_STORAGE is unset), so
    every restart/deploy wiped them. Text bodies already survived in
    source_files.source_text; binary attachments did not. This table keeps the
    bytes in Postgres so no offer file is ever lost again — across restarts,
    deploys, credit outages, and 360dialog media expiry.
    """
    __tablename__ = "raw_blobs"

    id              = Column(String(64), primary_key=True, default=lambda: uuid.uuid4().hex)
    job_id          = Column(String(64), nullable=True, index=True)
    source_filename = Column(String(512), nullable=True, index=True)
    content_type    = Column(String(128), nullable=True)
    size_bytes      = Column(Integer, nullable=True)
    sha256          = Column(String(64), nullable=True, index=True)
    data            = Column(LargeBinary, nullable=False)
    created_at      = Column(DateTime, default=datetime.utcnow, nullable=False)

    def to_meta(self) -> dict:
        return {
            "id": self.id,
            "job_id": self.job_id,
            "source_filename": self.source_filename,
            "content_type": self.content_type,
            "size_bytes": self.size_bytes,
            "sha256": self.sha256,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }
