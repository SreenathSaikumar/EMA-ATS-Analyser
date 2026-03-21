import asyncio
import logging
from typing import Any

from langgraph.graph import END, START, StateGraph

from src.agents.nodes import (
    compute_score,
    extract_jd_requirements,
    extract_resume_education,
    extract_resume_experience,
    extract_resume_skills,
    generate_explanation,
    match_experience,
    match_role,
    match_skills,
    normalize_entities,
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

    # Join to normalization: wait for all extraction nodes.
    workflow.add_node("normalize_entities", normalize_entities)

    # Matching (parallel).
    workflow.add_node("match_skills", match_skills)
    workflow.add_node("match_experience", match_experience)
    workflow.add_node("match_role", match_role)

    # Deterministic scoring + explanation.
    workflow.add_node("compute_score", compute_score)
    workflow.add_node("generate_explanation", generate_explanation)

    # Fan-out from START.
    workflow.add_edge(START, "extract_resume_skills")
    workflow.add_edge(START, "extract_resume_experience")
    workflow.add_edge(START, "extract_resume_education")
    workflow.add_edge(START, "extract_jd_requirements")

    # Join to normalization.
    workflow.add_edge("extract_resume_skills", "normalize_entities")
    workflow.add_edge("extract_resume_experience", "normalize_entities")
    workflow.add_edge("extract_resume_education", "normalize_entities")
    workflow.add_edge("extract_jd_requirements", "normalize_entities")

    # Fan-out from normalization to matching nodes.
    workflow.add_edge("normalize_entities", "match_skills")
    workflow.add_edge("normalize_entities", "match_experience")
    workflow.add_edge("normalize_entities", "match_role")

    # Join to compute_score.
    workflow.add_edge("match_skills", "compute_score")
    workflow.add_edge("match_experience", "compute_score")
    workflow.add_edge("match_role", "compute_score")

    # Final steps.
    workflow.add_edge("compute_score", "generate_explanation")
    workflow.add_edge("generate_explanation", END)

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
