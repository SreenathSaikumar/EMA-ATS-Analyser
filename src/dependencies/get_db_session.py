from typing import final
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import sessionmaker

from src.common.commons_container import common_utils

session_factory = sessionmaker(common_utils.db_engine, class_=AsyncSession, expire_on_commit=False)
async def get_db_session() -> AsyncSession:
    session = session_factory()
    try:
        async with session.begin():
            yield session
    except Exception:
        await session.rollback()
    else:
        await session.commit()
    finally:
        await session.close()