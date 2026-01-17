"""Simple database initialization without migrations."""

import os
from pathlib import Path
from sqlalchemy.ext.asyncio import create_async_engine
from .models import Base
from config.settings import Settings


async def init_database():
    """Initialize SQLite database and create all tables."""
    settings = Settings()

    # Ensure the data directory exists
    db_path = Path(settings.database_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    # Create async engine
    engine = create_async_engine(
        settings.database_url,
        echo=settings.debug,
    )

    # Create all tables
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    await engine.dispose()
    print(f"✅ Database initialized at: {settings.database_path}")


if __name__ == "__main__":
    import asyncio
    asyncio.run(init_database())