from __future__ import annotations

import asyncio

from src.agents.nodes.ats_resume_eval_nodes import match_role, match_skills
from src.utils.constants import Constants


def test_support_integration_semantic_skills_not_zero(monkeypatch) -> None:
    # Keep this deterministic by disabling the optional LLM alignment path.
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    state = {
        "normalized_resume_skills": [
            "automation",
            "workflow orchestration",
            "api integration",
            "customer support",
            "data engineering",
            "agent collaboration",
        ],
        "normalized_jd_skills": [
            "workflow automation",
            "technical integration",
            "saas apis",
            "customer support",
            "agent collaboration",
            "high precision outcomes",
        ],
        "resume_text": (
            "Built automation for support ticket triage and non-support inquiry resolution. "
            "Designed workflow orchestration across internal and external agent systems. "
            "Implemented API integrations with SaaS tools and CRM platforms. "
            "Improved customer support accuracy and faster resolution outcomes."
        ),
    }

    out = asyncio.run(match_skills(state))
    assert out["skills_match_score"] > 0.0
    assert "customer support" not in out["missing_skills"]
    assert "agent collaboration" not in out["missing_skills"]
    assert len(out["missing_skills"]) < len(state["normalized_jd_skills"])


def test_match_role_is_neutral_when_jd_role_missing() -> None:
    out = match_role(
        {
            "jd_role": "",
            "resume_experience": {
                "titles": ["Senior Software Engineer", "Software Engineer"],
            },
        }
    )
    assert out["role_match_score"] == Constants.EXPERIENCE_MATCH_NEUTRAL
