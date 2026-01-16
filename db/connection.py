"""Database connection management for async PostgreSQL."""

from typing import AsyncGenerator, Optional
import asyncio
from sqlalchemy.ext.asyncio import (
    create_async_engine,
    AsyncEngine,
    async_sessionmaker,
    AsyncSession,
)
from sqlalchemy.pool import NullPool

from config.settings import Settings


class DatabaseManager:
    """Manages database connections and sessions."""

    def __init__(self):
        self._engine: Optional[AsyncEngine] = None
        self._session_factory: Optional[async_sessionmaker[AsyncSession]] = None
        self._settings = Settings()

    async def initialize(self) -> None:
        """Initialize database engine and session factory."""
        if self._engine is not None:
            return

        self._engine = create_async_engine(
            self._settings.database_url,
            echo=self._settings.debug,  # Enable SQL logging in debug mode
            poolclass=NullPool,  # Simple pool for development
            pool_pre_ping=True,  # Verify connections before use
        )

        self._session_factory = async_sessionmaker(
            self._engine,
            class_=AsyncSession,
            expire_on_commit=False,
        )

    async def close(self) -> None:
        """Close database engine."""
        if self._engine:
            await self._engine.dispose()
            self._engine = None
            self._session_factory = None

    @property
    def engine(self) -> AsyncEngine:
        """Get database engine."""
        if self._engine is None:
            raise RuntimeError("Database not initialized. Call initialize() first.")
        return self._engine

    async def get_session(self) -> AsyncGenerator[AsyncSession, None]:
        """Get database session."""
        if self._session_factory is None:
            raise RuntimeError("Database not initialized. Call initialize() first.")

        async with self._session_factory() as session:
            try:
                yield session
            except Exception:
                await session.rollback()
                raise
            finally:
                await session.close()


# Global database manager instance
_db_manager = DatabaseManager()


async def get_session_manager() -> DatabaseManager:
    """Get database manager."""
    await _db_manager.initialize()
    return _db_manager
