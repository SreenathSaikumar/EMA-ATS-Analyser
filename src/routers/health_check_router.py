from src.controllers.health_check_controller import HealthCheckController

from src.routers.router_base import RouterBase

class HealthCheckRouter(RouterBase):
    def __init__(self):
        super().__init__(prefix="/public/v1/isup", tags=["health_check"])
        self.__health_check_controller = HealthCheckController()

    def get_router(self):
        self.__router.add_api_route("/health", self.__health_check_controller.check_health, methods=["GET"])
        return self.__router