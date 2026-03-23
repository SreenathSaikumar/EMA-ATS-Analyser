from __future__ import annotations

from typing import Any, Dict, List, TypedDict

from pydantic import BaseModel, ConfigDict, Field
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
    jd_education_requirements: List[str]
    requirement_constraints: List[Dict[str, Any]]

    # normalized
    normalized_resume_skills: List[str]
    normalized_jd_skills: List[str]
    normalized_jd_education: List[str]

    # matching results
    skills_match_score: float
    missing_skills: List[str]
    experience_match_score: float
    role_match_score: float
    education_match_score: float
    missing_education: List[str]
    requirement_constraint_score: float
    hard_requirement_misses: List[Dict[str, Any]]
    constraint_findings: List[Dict[str, Any]]

    # final
    final_score: float
    explanation: str
    strengths: List[str]
    weaknesses: List[str]
    judge_verdict: str
    judge_confidence: float
    judge_notes: List[str]


class GraphStatePartial(TypedDict, total=False):
    resume_skills: List[str]
    resume_experience: Dict[str, Any]
    resume_education: List[str]
    jd_skills: List[str]
    jd_experience: int
    jd_role: str
    jd_education_requirements: List[str]
    requirement_constraints: List[Dict[str, Any]]

    normalized_resume_skills: List[str]
    normalized_jd_skills: List[str]
    normalized_jd_education: List[str]

    skills_match_score: float
    missing_skills: List[str]
    experience_match_score: float
    role_match_score: float
    education_match_score: float
    missing_education: List[str]
    requirement_constraint_score: float
    hard_requirement_misses: List[Dict[str, Any]]
    constraint_findings: List[Dict[str, Any]]

    final_score: float
    explanation: str
    strengths: List[str]
    weaknesses: List[str]
    judge_verdict: str
    judge_confidence: float
    judge_notes: List[str]


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
    model_config = ConfigDict(extra="ignore")

    jd_skills: List[str] = Field(default_factory=list)
    jd_experience: int = 0
    jd_role: str = ""
    jd_education_requirements: List[str] = Field(default_factory=list)


class RequirementConstraintItem(BaseModel):
    requirement_label: str
    min_years: float = 0.0
    strictness: str = "soft"
    strictness_evidence: str = ""


class RequirementConstraintsExtractionOutput(BaseModel):
    requirement_constraints: List[RequirementConstraintItem] = Field(default_factory=list)


class RequirementConstraintEvalItem(BaseModel):
    requirement_label: str
    matched: bool = False
    estimated_years: float = 0.0
    confidence: float = 0.0
    reasoning: str = ""


class RequirementConstraintEvaluationOutput(BaseModel):
    evaluations: List[RequirementConstraintEvalItem] = Field(default_factory=list)


class ExtractionVerifyOutput(BaseModel):
    """LLM pass: drop extraction items not supported by source texts."""

    model_config = ConfigDict(extra="ignore")

    resume_skills: List[str] = Field(default_factory=list)
    jd_skills: List[str] = Field(default_factory=list)
    resume_education: List[str] = Field(default_factory=list)
    jd_education_requirements: List[str] = Field(default_factory=list)


class ExplanationOutput(BaseModel):
    explanation: str
    strengths: List[str] = Field(default_factory=list)
    weaknesses: List[str] = Field(default_factory=list)


class JudgeFinalOutput(BaseModel):
    judge_verdict: str = "pass"
    judge_confidence: float = 0.0
    judge_notes: List[str] = Field(default_factory=list)

