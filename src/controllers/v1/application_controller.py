from fastapi import Form, UploadFile, File, Depends, Path
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
from fastapi import status

from sqlalchemy.ext.asyncio import AsyncSession
from src.dependencies.get_db_session import get_db_session
from src.controllers.base_controller import BaseController
from src.services.application_service import ApplicationService

class ApplicationController(BaseController):
    def __init__(self):
        super().__init__()
        self.__application_service = ApplicationService()

    async def create_application(self, job_id: int = Path(...), name: str = Form(...), resume: UploadFile = File(...), session: AsyncSession = Depends(get_db_session)) -> JSONResponse:
        try:
            self._logger.info(f"Creating application: {name}, {resume} {job_id}")
            await self.__application_service.create_application(session, job_id, name, resume)
            return JSONResponse(status_code=status.HTTP_201_CREATED, content={"message": "Application created successfully"})
        except Exception as e:
            self._logger.error(f"Error creating application: {e}")
            return JSONResponse(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, content={"message": "Error creating application"})

    async def list_applications(self, job_id: int = Path(...), session: AsyncSession = Depends(get_db_session)) -> JSONResponse:
        try:
            self._logger.info(f"Listing applications: {job_id}")
            applications = await self.__application_service.list_applications(session, job_id)
            res = {"data": applications}
            return JSONResponse(status_code=status.HTTP_200_OK, content=jsonable_encoder(res))
        except Exception as e:
            self._logger.error(f"Error listing applications: {e}")
            return JSONResponse(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, content={"message": "Error listing applications"})