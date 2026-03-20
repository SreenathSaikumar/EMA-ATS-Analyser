from typing import Self

from sqlalchemy.orm import DeclarativeBase, declarative_mixin, session, sessionmaker, declared_attr
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import Column, Integer, DateTime, BigInteger, select, func
from datetime import datetime

from src.common.commons_container import common_utils

session_factory = sessionmaker(common_utils.db_engine, class_=AsyncSession, autoflush=False, expire_on_commit=False)

class ORMBase(DeclarativeBase):
    pass

@declarative_mixin
class BaseModel:

    @declared_attr
    def __tablename__(cls) -> str:
        return cls.__name__.lower()

    id = Column(BigInteger, primary_key=True)
    created_at = Column(DateTime, default=func.now(), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime, default=func.now(), server_default=func.now(), onupdate=func.now(), nullable=False)


    @classmethod
    async def get(cls, id: int) -> Self | None:
        async with session_factory() as session:
            return await session.get(cls, id)

    @classmethod
    async def filter(cls, *where_clauses) -> list[Self]:
        query = select(cls).where(*where_clauses)
        async with session_factory() as session:
            res = await session.execute(query)
            return res.scalars().all()



