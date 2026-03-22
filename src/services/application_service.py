import json

from logging import getLogger
from fastapi import UploadFile, status
from fastapi.exceptions import HTTPException

from sqlalchemy.ext.asyncio import AsyncSession

from src.common.commons_container import common_utils
from src.repositories.sql_db.models.ats_models import Application, JobDescription
from src.utils.enums import ProcessingStatus
from src.dtos.responses.list_applications_response import LiteApplication

logger = getLogger(__name__)


class ApplicationService:
    def __init__(self):
        self.__logger = logger

    async def create_application(
        self, session: AsyncSession, job_id: int, name: str, resume: UploadFile
    ) -> Application:
        self.__logger.info(f"Creating application: {job_id}, {name}, {resume}")

        if not await self.__check_job_exists(session, job_id):
            self.__logger.warning(f"Job not found: {job_id}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, detail="Job not found"
            )

        resume_bytes = await resume.read()
        application = Application(
            job_id=job_id,
            name=name,
            resume=resume_bytes,
            resume_file_type=resume.content_type,
            resume_file_name=resume.filename,
            processing_status=ProcessingStatus.PENDING.value,
        )
        session.add(application)
        await session.flush()
        await session.commit()
        await self.__dispatch_application_process_job(application)
        return application

    async def list_applications(
        self, session: AsyncSession, job_id: int
    ) -> list[LiteApplication]:
        try:
            self.__logger.info(f"Listing applications: {job_id}")
            applications = await Application.filter(Application.job_id == job_id)
            return [
                LiteApplication(
                    id=application.id,
                    name=application.name,
                    relevance_score=application.relevance_score,
                    reasoning=json.loads(application.reasoning),
                    processing_status=application.processing_status,
                    created_at=application.created_at,
                    updated_at=application.updated_at,
                )
                for application in applications
            ]
        except Exception as e:
            self.__logger.error(f"Error listing applications: {e}")
            raise e

    async def __check_job_exists(self, session: AsyncSession, job_id: int) -> bool:
        job = await JobDescription.get(job_id)
        return job is not None

    async def __dispatch_application_process_job(
        self, application: Application
    ) -> None:
        await common_utils.sqs_producer.send_message(
            {
                "application_id": application.id,
                "job_description_id": application.job_id,
            }
        )
