from fastapi import APIRouter

from src.routers.health_check_router import HealthCheckRouter
from src.routers.v1.job_posting_router import JobPostingRouter
from src.routers.v1.application_router import ApplicationRouter

class ApiContainer:
    def __init__(self):
        self.__health_check_router = HealthCheckRouter()
        self.__job_posting_router = JobPostingRouter()
        self.__application_router = ApplicationRouter()
    async def get_routers(self) -> list[APIRouter]:
        await self.__check_dependencies()
        return [self.__health_check_router.get_router(), self.__job_posting_router.get_router(), self.__application_router.get_router()]

    async def __check_dependencies(self):
        pass