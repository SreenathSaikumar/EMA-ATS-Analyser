from sqlalchemy import (
    BigInteger,
    Column,
    ForeignKey,
    String,
    Text,
    DECIMAL,
)
from sqlalchemy.dialects.mysql import LONGBLOB

from src.repositories.sql_db.models.orm_base import BaseModel, ORMBase
from src.utils.enums import ProcessingStatus


class JobDescription(ORMBase, BaseModel):
    __tablename__ = "job_description"

    position = Column(String(300), nullable=False)
    description = Column(Text, nullable=True)

    def __repr__(self) -> str:
        return f"JobDescription(id={self.id}, position={self.position}, description={self.description})"


class Application(ORMBase, BaseModel):
    __tablename__ = "application"

    job_id = Column(BigInteger, ForeignKey("job_description.id"), nullable=False)
    name = Column(String(300), nullable=False)
    relevance_score = Column(DECIMAL(precision=1, scale=1), nullable=True)
    reasoning = Column(Text, nullable=True)
    resume = Column(LONGBLOB, nullable=False)
    resume_file_type = Column(String(100), nullable=False)
    resume_file_name = Column(String(100), nullable=False)
    processing_status = Column(
        String(100), nullable=False, default=ProcessingStatus.PENDING.value
    )

    def __repr__(self) -> str:
        return f"Application(id={self.id}, job_id={self.job_id}, name={self.name}, resume_file_type={self.resume_file_type}, resume_file_name={self.resume_file_name}, processing_status={self.processing_status}, relevance_score={self.relevance_score}, reasoning={self.reasoning})"
