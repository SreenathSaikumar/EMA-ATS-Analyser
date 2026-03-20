from logging import getLogger

from sqlalchemy.ext.asyncio import AsyncSession

from src.dtos.requests.create_posting_request import CreatePostingRequest
from src.repositories.sql_db.models.ats_models import JobDescription

logger = getLogger(__name__)


class JobPostingService:
    def __init__(self):
        self.__logger = logger

    async def create_job_posting(
        self, session: AsyncSession, job_posting: CreatePostingRequest
    ) -> JobDescription:
        try:
            self.__logger.info(f"Creating job posting: {job_posting}")
            posting = JobDescription(
                position=job_posting.position, description=job_posting.description
            )
            session.add(posting)
            await session.flush()
            return posting
        except Exception as e:
            self.__logger.error(f"Error creating job posting: {e}")
            raise e

    async def list_job_postings(self, session: AsyncSession) -> list[JobDescription]:
        try:
            self.__logger.info("Listing job postings...")
            postings = await JobDescription.filter()
            return postings
        except Exception as e:
            self.__logger.error(f"Error listing job postings: {e}")
            raise e
