from src.routers.router_base import RouterBase
from src.controllers.v1.application_controller import ApplicationController

class ApplicationRouter(RouterBase):
    def __init__(self):
        super().__init__(prefix="/v1/{job_id}/application", tags=["application"])
        self.__application_controller = ApplicationController()

    def get_router(self):
        router = super().get_router()
        router.add_api_route("/", self.__application_controller.create_application, methods=["POST"])
        router.add_api_route("/", self.__application_controller.list_applications, methods=["GET"])
        return router