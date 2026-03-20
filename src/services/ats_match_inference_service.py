from src.repositories.sql_db.models.ats_models import Application, JobDescription
from src.utils.enums import ProcessingStatus


class AtsMatchInferenceService:
    def __init__(self):
        pass

    async def infer_ats_match(
        self, application_id: int, job_description_id: int
    ) -> float:
        application = await Application.get(application_id)
        job_description = await JobDescription.get(job_description_id)

        if application is None or job_description is None:
            raise Exception("Application or job description not found")

        relevance_score, reasoning = await self.__start_ats_match_inference(
            application, job_description
        )

        application.relevance_score = relevance_score
        application.reasoning = reasoning
        application.processing_status = ProcessingStatus.COMPLETED.value
        await application.save()

        if application is None or job_description is None:
            raise Exception("Application or job description not found")
        return 0.0

    async def __start_ats_match_inference(
        self, application: Application, job_description: JobDescription
    ) -> tuple[float, str]:
        return 0.0, "dummy"
