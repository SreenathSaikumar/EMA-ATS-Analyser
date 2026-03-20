from fastapi.responses import JSONResponse
from fastapi import status

from src.controllers.base_controller import BaseController
from src.services.db_health_check_service import HealthCheckService

class HealthCheckController(BaseController):
    def __init__(self):
        super().__init__()
        self.__health_check_service = HealthCheckService()

    async def check_health(self) -> JSONResponse:
        self.__logger.info("Checking health...")
        if await self.__health_check_service.check_db_health():
            return JSONResponse(status_code=status.HTTP_200_OK, content={"message": "OK"})
        else:
            return JSONResponse(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, content={"message": "Error checking database health"})