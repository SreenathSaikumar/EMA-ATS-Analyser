from src.routers.router_base import RouterBase
from src.controllers.v1.job_posting_controller import JobPostingController

class JobPostingRouter(RouterBase):
    def __init__(self):
        super().__init__(prefix="/v1/job-posting", tags=["job-posting"])
        self.__job_posting_controller = JobPostingController()

    def get_router(self):
        router = super().get_router()
        router.add_api_route("/", self.__job_posting_controller.create_job_posting, methods=["POST"])
        router.add_api_route("/", self.__job_posting_controller.list_job_postings, methods=["GET"])
        return router