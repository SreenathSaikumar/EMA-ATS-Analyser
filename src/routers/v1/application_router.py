from src.routers.router_base import RouterBase
from src.controllers.v1.application_controller import ApplicationController

class ApplicationRouter(RouterBase):
    def __init__(self):
        super().__init__(prefix="/v1/{job_id}/application", tags=["application"])

    def get_router(self):
        self.__router.add_api_route("/", self.__application_controller.create_application, methods=["POST"])
        self.__router.add_api_route("/", self.__application_controller.list_applications, methods=["GET"])
        return self.__router