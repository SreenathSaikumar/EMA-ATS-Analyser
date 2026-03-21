from pydantic import BaseModel, Field


class AtsSqsJobs(BaseModel):
    application_id: int = Field(..., description="The ID of the application")
    job_description_id: int = Field(..., description="The ID of the job description")
    retry_count: int = Field(
        0, description="The number of times the job has been retried"
    )
