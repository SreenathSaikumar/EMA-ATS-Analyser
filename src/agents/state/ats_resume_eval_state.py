from __future__ import annotations

from typing import Any, Dict, List, TypedDict

from pydantic import BaseModel, Field
from typing import Required


class GraphState(TypedDict, total=False):
    # Input
    resume_text: Required[str]
    jd_text: Required[str]

    # extracted data
    resume_skills: List[str]
    resume_experience: Dict[str, Any]
    resume_education: List[str]
    jd_skills: List[str]
    jd_experience: int
    jd_role: str

    # normalized
    normalized_resume_skills: List[str]
    normalized_jd_skills: List[str]

    # matching results
    skills_match_score: float
    missing_skills: List[str]
    experience_match_score: float
    role_match_score: float

    # final
    final_score: float
    explanation: str
    strengths: List[str]
    weaknesses: List[str]


class GraphStatePartial(TypedDict, total=False):
    resume_skills: List[str]
    resume_experience: Dict[str, Any]
    resume_education: List[str]
    jd_skills: List[str]
    jd_experience: int
    jd_role: str

    normalized_resume_skills: List[str]
    normalized_jd_skills: List[str]

    skills_match_score: float
    missing_skills: List[str]
    experience_match_score: float
    role_match_score: float

    final_score: float
    explanation: str
    strengths: List[str]
    weaknesses: List[str]


class ResumeSkillsExtractionOutput(BaseModel):
    resume_skills: List[str] = Field(default_factory=list)


class ResumeExperienceModel(BaseModel):
    years_total: float = 0.0
    years_relevant: float = 0.0
    titles: List[str] = Field(default_factory=list)


class ResumeExperienceExtractionOutput(BaseModel):
    resume_experience: Dict[str, Any]


class ResumeEducationExtractionOutput(BaseModel):
    resume_education: List[str] = Field(default_factory=list)


class JDRequirementsExtractionOutput(BaseModel):
    jd_skills: List[str] = Field(default_factory=list)
    jd_experience: int = 0
    jd_role: str = ""


class ExplanationOutput(BaseModel):
    explanation: str
    strengths: List[str] = Field(default_factory=list)
    weaknesses: List[str] = Field(default_factory=list)

