from pydantic import BaseModel
from datetime import datetime


class LiteApplication(BaseModel):
    id: int
    name: str
    relevance_score: float
    reasoning: dict
    processing_status: str
    created_at: datetime
    updated_at: datetime


class ListApplicationsResponse(BaseModel):
    data: list[LiteApplication]
