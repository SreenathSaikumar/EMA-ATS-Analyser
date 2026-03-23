import asyncio
import logging

from typing import Any

from langgraph.graph import END, START, StateGraph

from src.agents.nodes import (
    compute_score,
    evaluate_requirement_constraints,
    extract_jd_requirements,
    extract_requirement_constraints,
    extract_resume_education,
    extract_resume_experience,
    extract_resume_skills,
    generate_explanation,
    judge_final_evaluation,
    match_education,
    match_experience,
    match_role,
    match_skills,
    normalize_entities,
    verify_extractions,
)
from src.agents.state import GraphState

logger = logging.getLogger(__name__)


def build_ats_resume_evaluation_graph() -> Any:
    """
    Build and compile the LangGraph workflow for ATS resume evaluation.
    """

    workflow: StateGraph[GraphState] = StateGraph(GraphState)

    # Extraction (fan-out in parallel).
    workflow.add_node("extract_resume_skills", extract_resume_skills)
    workflow.add_node("extract_resume_experience", extract_resume_experience)
    workflow.add_node("extract_resume_education", extract_resume_education)
    workflow.add_node("extract_jd_requirements", extract_jd_requirements)
    workflow.add_node("extract_requirement_constraints", extract_requirement_constraints)

    # Optional verification then normalization.
    workflow.add_node("verify_extractions", verify_extractions)
    workflow.add_node("normalize_entities", normalize_entities)

    # Matching (parallel).
    workflow.add_node("match_skills", match_skills)
    workflow.add_node("match_experience", match_experience)
    workflow.add_node("match_role", match_role)
    workflow.add_node("match_education", match_education)
    workflow.add_node("evaluate_requirement_constraints", evaluate_requirement_constraints)

    # Deterministic scoring + explanation.
    workflow.add_node("compute_score", compute_score)
    workflow.add_node("generate_explanation", generate_explanation)
    workflow.add_node("judge_final_evaluation", judge_final_evaluation)

    # Fan-out from START.
    workflow.add_edge(START, "extract_resume_skills")
    workflow.add_edge(START, "extract_resume_experience")
    workflow.add_edge(START, "extract_resume_education")
    workflow.add_edge(START, "extract_jd_requirements")
    workflow.add_edge(START, "extract_requirement_constraints")

    # Join to normalization.
    workflow.add_edge("extract_resume_skills", "verify_extractions")
    workflow.add_edge("extract_resume_experience", "verify_extractions")
    workflow.add_edge("extract_resume_education", "verify_extractions")
    workflow.add_edge("extract_jd_requirements", "verify_extractions")
    workflow.add_edge("verify_extractions", "normalize_entities")

    # Fan-out from normalization to matching nodes.
    workflow.add_edge("normalize_entities", "match_skills")
    workflow.add_edge("normalize_entities", "match_experience")
    workflow.add_edge("normalize_entities", "match_role")
    workflow.add_edge("normalize_entities", "match_education")

    # Join to compute_score.
    workflow.add_edge("match_skills", "compute_score")
    workflow.add_edge("match_experience", "compute_score")
    workflow.add_edge("match_role", "compute_score")
    workflow.add_edge("match_education", "compute_score")
    workflow.add_edge("evaluate_requirement_constraints", "compute_score")

    # Requirement constraints evaluation path.
    workflow.add_edge("extract_requirement_constraints", "evaluate_requirement_constraints")
    workflow.add_edge("normalize_entities", "evaluate_requirement_constraints")

    # Final steps.
    workflow.add_edge("compute_score", "generate_explanation")
    workflow.add_edge("generate_explanation", "judge_final_evaluation")
    workflow.add_edge("judge_final_evaluation", END)

    return workflow.compile()


async def _example() -> None:
    graph = build_ats_resume_evaluation_graph()
    result = await graph.ainvoke(
        {
            "resume_text": (
                "Sample resume text with skills: Python, SQL, React. "
                "Experience: 4 years Software Engineer."
            ),
            "jd_text": (
                "We need a Software Engineer with Python, SQL, React. "
                "3 years experience. Role: Software Engineer."
            ),
        }
    )
    print(result)


if __name__ == "__main__":
    asyncio.run(_example())
