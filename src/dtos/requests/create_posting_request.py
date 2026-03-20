from pydantic import BaseModel, Field

class CreatePostingRequest(BaseModel):
    position: str = Field(..., description="The position of the job posting", min_length=1, max_length=300)
    description: str = Field(None, description="The description of the job posting")
