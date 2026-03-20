from sqlalchemy import BigInteger, Column, ForeignKey, ForeignKeyConstraint, String, Text, DECIMAL, LargeBinary

from src.repositories.sql_db.models.orm_base import BaseModel, ORMBase


class JobDescription(ORMBase, BaseModel):
    __tablename__ = "job_description"

    position = Column(String(300), nullable=False)
    description = Column(Text, nullable=True)

    def __repr__(self) -> str:
        return f"JobDescription(id={self.id}, position={self.position}, description={self.description})"


class Application(ORMBase, BaseModel):
    __tablename__ = "application"

    job_id = ForeignKey("job_description.id")
    name = Column(String(300), nullable=False)
    relevance_score = Column(DECIMAL(precision=1), nullable=True)
    reasoning = Column(Text, nullable=True)
    resume = Column(LargeBinary, nullable=False)
    resume_file_type = Column(String(100), nullable=False)
    resume_file_name = Column(String(100), nullable=False)

    def __repr__(self) -> str:
        return f"Application(id={self.id}, job_id={self.job_id}, name={self.name}, resume_path={self.resume_path}, relevance_score={self.relevance_score}, reasoning={self.reasoning})"