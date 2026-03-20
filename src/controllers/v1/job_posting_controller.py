from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
from fastapi import Depends, status
from sqlalchemy.ext.asyncio import AsyncSession

from src.dependencies.get_db_session import get_db_session
from src.dtos.requests.create_posting_request import CreatePostingRequest
from src.controllers.base_controller import BaseController
from src.services.job_posting_service import JobPostingService


class JobPostingController(BaseController):
    def __init__(self):
        super().__init__()
        self.__job_posting_service = JobPostingService()

    async def create_job_posting(
        self,
        request: CreatePostingRequest,
        session: AsyncSession = Depends(get_db_session),
    ) -> JSONResponse:
        try:
            self._logger.info(f"Creating job posting: {request}")
            await self.__job_posting_service.create_job_posting(session, request)
            return JSONResponse(
                status_code=status.HTTP_201_CREATED,
                content={"message": "Job posting created successfully"},
            )
        except Exception as e:
            self._logger.error(f"Error creating job posting: {e}")
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content={"message": "Error creating job posting"},
            )

    async def list_job_postings(
        self, session: AsyncSession = Depends(get_db_session)
    ) -> JSONResponse:
        try:
            self._logger.info("Listing job postings...")
            postings = await self.__job_posting_service.list_job_postings(session)
            posting_response_data = [
                {
                    "id": posting.id,
                    "position": posting.position,
                    "description": posting.description,
                    "posted_at": posting.created_at,
                }
                for posting in postings
            ]
            res = {"data": posting_response_data}
            return JSONResponse(
                status_code=status.HTTP_200_OK, content=jsonable_encoder(res)
            )
        except Exception as e:
            self._logger.error(f"Error listing job postings: {e}")
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content={"message": "Error listing job postings"},
            )
