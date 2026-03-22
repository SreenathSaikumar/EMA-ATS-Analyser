import json
import logging
from typing import Any

from src.agents.ats_processor_agent import build_ats_resume_evaluation_graph
from src.repositories.sql_db.models.ats_models import Application, JobDescription
from src.utils.enums import ProcessingStatus
from src.utils.resume_text import load_resume_text_from_bytes

logger = logging.getLogger(__name__)


class AtsMatchInferenceService:
    def __init__(self) -> None:
        self._graph = build_ats_resume_evaluation_graph()

    async def infer_ats_match(
        self, application_id: int, job_description_id: int
    ) -> float:
        application = await Application.get(application_id)
        job_description = await JobDescription.get(job_description_id)

        if application is None or job_description is None:
            raise Exception("Application or job description not found")

        relevance_score, reasoning_json = await self.__start_ats_match_inference(
            application, job_description
        )

        application.relevance_score = relevance_score
        application.reasoning = reasoning_json
        application.processing_status = ProcessingStatus.COMPLETED.value
        await application.save()

        return float(relevance_score)

    async def __start_ats_match_inference(
        self,
        application: Application,
        job_description: JobDescription,
    ) -> tuple[float, str]:
        if int(application.job_id) != int(job_description.id):
            raise ValueError(
                "Application job_id does not match the loaded job description"
            )

        resume_text = await load_resume_text_from_bytes(
            application.resume,
            application.resume_file_type,
            application.resume_file_name,
        )
        if not resume_text.strip():
            raise ValueError("Resume text is empty after loading")

        jd_parts = ["Position: " + job_description.position]
        if job_description.description:
            jd_parts.append("Description: " + job_description.description)
        jd_text = "\n\n".join(jd_parts)

        result: dict[str, Any] = await self._graph.ainvoke(
            {"resume_text": resume_text, "jd_text": jd_text}
        )

        final_score = float(result.get("final_score") or 0.0)
        reasoning_payload = {
            "explanation": result.get("explanation") or "",
            "strengths": result.get("strengths") or [],
            "weaknesses": result.get("weaknesses") or [],
            "skills_match_score": result.get("skills_match_score"),
            "experience_match_score": result.get("experience_match_score"),
            "role_match_score": result.get("role_match_score"),
            "final_score": final_score,
        }

        return final_score, json.dumps(reasoning_payload)

    async def mark_application_failed(self, application_id: int) -> None:
        application = await Application.get(application_id)
        if application is not None:
            application.processing_status = ProcessingStatus.FAILED.value
            await application.save()
