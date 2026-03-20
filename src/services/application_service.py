from logging import getLogger
from fastapi import UploadFile, status
from fastapi.exceptions import HTTPException

from sqlalchemy.ext.asyncio import AsyncSession
from src.repositories.sql_db.models.ats_models import Application, JobDescription

logger = getLogger(__name__)

class ApplicationService:
    def __init__(self):
        self.__logger = logger

    async def create_application(self, session: AsyncSession, job_id: int, name: str, resume: UploadFile) -> Application:
        try:
            self.__logger.info(f"Creating application: {job_id}, {name}, {resume}")

            if not await self.__check_job_exists(session, job_id):
                self.__logger.warning(f"Job not found: {job_id}")
                raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Job not found")

            resume_bytes = await resume.read()
            application = Application(job_id=job_id, name=name, resume=resume_bytes, resume_file_type=resume.content_type, resume_file_name=resume.filename)
            session.add(application)
            await session.commit()
            await session.refresh(application)
            return application
        except Exception as e:
            self.__logger.error(f"Error creating application: {e}")

    async def list_applications(self, session: AsyncSession, job_id: int) -> list[Application]:
        try:
            self.__logger.info(f"Listing applications: {job_id}")
            applications = await Application.filter(Application.job_id == job_id)
            return applications
        except Exception as e:
            self.__logger.error(f"Error listing applications: {e}")
            raise e

    async def __check_job_exists(self, session: AsyncSession, job_id: int) -> bool:
        job = await JobDescription.get(job_id)
        return job is not None