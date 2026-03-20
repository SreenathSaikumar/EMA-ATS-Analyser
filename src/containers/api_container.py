from fastapi import APIRouter

from src.routers.health_check_router import HealthCheckRouter

class ApiContainer:
    def __init__(self):
        self.__health_check_router = HealthCheckRouter()

    async def get_routers(self) -> list[APIRouter]:
        await self.__check_dependencies()
        return [self.__health_check_router.get_router()]

    async def __check_dependencies(self):
        pass