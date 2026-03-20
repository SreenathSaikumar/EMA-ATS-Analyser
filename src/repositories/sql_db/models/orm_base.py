from typing import Self

from sqlalchemy.orm import DeclarativeBase, declarative_mixin
from sqlalchemy import Column, Integer, DateTime, BigInteger
from datetime import datetime

class ORMBase(DeclarativeBase):
    pass

@declarative_mixin
class BaseModel(ORMBase):
    id = Column(BigInteger, primary_key=True)
    created_at = Column(DateTime, default=datetime.now, nullable=False)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now, nullable=False)


    @staticmethod
    def get_by_id(cls, id: int) -> Self | None:
        return cls.query.filter(cls.id == id).first()

