"""
core/database.py
SQLAlchemy engine + session factory for Azure PostgreSQL.
"""
import os
import logging
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

logger = logging.getLogger(__name__)

DATABASE_URL = os.getenv("DATABASE_URL")

# Declarative base used by all ORM models
Base = declarative_base()

_engine = None
SessionLocal = None


def get_engine():
    """Lazily create and return the SQLAlchemy engine."""
    global _engine
    if _engine is None:
        if not DATABASE_URL:
            raise RuntimeError(
                "DATABASE_URL environment variable is not set. "
                "Add it to your .env file."
            )
        _engine = create_engine(
            DATABASE_URL,
            pool_pre_ping=True,       # reconnect on stale connections
            pool_size=5,
            max_overflow=10,
            echo=False,
        )
        logger.info("✓ SQLAlchemy engine created for Azure PostgreSQL")
    return _engine


def get_session_factory():
    """Return (and lazily create) the SessionLocal factory."""
    global SessionLocal
    if SessionLocal is None:
        SessionLocal = sessionmaker(
            autocommit=False,
            autoflush=False,
            bind=get_engine(),
        )
    return SessionLocal


def get_db():
    """FastAPI dependency — yields a DB session and closes it after use."""
    factory = get_session_factory()
    db = factory()
    try:
        yield db
    finally:
        db.close()


def init_db():
    """Create all tables defined via Base.metadata if they do not exist."""
    try:
        # Import models so their tables are registered on Base.metadata
        from models import offer_item,source_file
        engine = get_engine()
        Base.metadata.create_all(bind=engine)
        logger.info("✓ Database tables verified / created")
    except Exception as e:
        logger.error(f"✗ Failed to initialise database: {e}")
        raise
