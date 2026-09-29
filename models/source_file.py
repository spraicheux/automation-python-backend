import uuid
from datetime import datetime
from sqlalchemy import Column, String, DateTime, Integer, Boolean
from sqlalchemy.orm import relationship
from core.database import Base


class SourceFileDB(Base):
    __tablename__ = "source_files"

    id           = Column(String(64), primary_key=True, default=lambda: str(uuid.uuid4()))
    job_id       = Column(String(64), nullable=False, unique=True, index=True)
    source_filename  = Column(String(512), nullable=True, index=True)
    sender_name  = Column(String(255), nullable=True)
    sender_email = Column(String(255), nullable=True)
    supplier_name  = Column(String(255), nullable=True)
    supplier_email = Column(String(255), nullable=True)
    source_channel = Column(String(128), nullable=True)
    source_message_id = Column(String(255), nullable=True)
    product_count = Column(Integer, default=0)
    # Phase 3 M1: content-hash duplicate guard + source-vs-imported audit.
    content_hash          = Column(String(64), nullable=True, index=True)  # sha256 hex
    expected_row_count    = Column(Integer, nullable=True)  # source-row estimate (barcodes in PDF, df length in XLSX)
    imported_row_count    = Column(Integer, nullable=True)  # how many actually persisted
    import_incomplete     = Column(Boolean, nullable=True)  # true when expected > imported
    created_at   = Column(DateTime, default=datetime.utcnow, nullable=False)

    offer_items = relationship("OfferItemDB", back_populates="source_file", lazy="dynamic")

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "job_id": self.job_id,
            "source_filename": self.source_filename,
            "sender_name": self.sender_name,
            "sender_email": self.sender_email,
            "supplier_name": self.supplier_name,
            "supplier_email": self.supplier_email,
            "source_channel": self.source_channel,
            "source_message_id": self.source_message_id,
            "product_count": self.product_count,
            "content_hash": self.content_hash,
            "expected_row_count": self.expected_row_count,
            "imported_row_count": self.imported_row_count,
            "import_incomplete": self.import_incomplete,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }