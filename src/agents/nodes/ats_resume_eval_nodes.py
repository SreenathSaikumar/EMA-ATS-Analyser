import json
import logging
import re
from typing import Dict, List, Optional, Set, cast

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel

from langchain_openai import ChatOpenAI

from src.agents.prompts import PromptsManager
from src.agents.state import (
    ExplanationOutput,
    GraphState,
    GraphStatePartial,
    JDRequirementsExtractionOutput,
    ResumeEducationExtractionOutput,
    ResumeExperienceExtractionOutput,
    ResumeExperienceModel,
    ResumeSkillsExtractionOutput,
)
from src.comlib.llms import LLMs
from src.utils.constants import Constants

logger = logging.getLogger(__name__)


def _build_llm() -> ChatOpenAI:
    # `temperature=0.0` makes extraction/explanation deterministic.
    return LLMs().get_gpt_4o_mini()


def _extract_json_from_text(text: str) -> str:
    """
    Best-effort JSON extraction for when the model wraps output in code blocks.
    """

    try:
        json.loads(text)
        return text
    except Exception:
        pass

    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        raise ValueError("No JSON object found in LLM response.")
    return cast(str, match.group(0))


async def _ainvoke_json_llm(
    prompt: str,
    output_model: type[BaseModel],
    *,
    system_content: Optional[str] = None,
) -> BaseModel:
    """
    Invoke an LLM to return ONLY valid JSON for `output_model`.
    """

    model = _build_llm()
    system = system_content or PromptsManager.system_json_extraction()
    messages = [
        SystemMessage(content=system),
        HumanMessage(content=prompt),
    ]

    resp = await model.ainvoke(messages)
    content = getattr(resp, "content", None)
    if not content:
        content = str(resp)

    json_text = _extract_json_from_text(content)
    return output_model.model_validate_json(json_text)


def _unique_preserve_order(items: List[str]) -> List[str]:
    seen: Set[str] = set()
    out: List[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def _normalize_skill_name(raw: str) -> str:
    s = raw.strip().lower()
    if not s:
        return s

    # Common abbreviations / aliases (basic mapping).
    alias_map: Dict[str, str] = {
        "c#": "c sharp",
        "c++": "c plus plus",
        "node.js": "node js",
        "js": "javascript",
        "ml": "machine learning",
        "nlp": "natural language processing",
        "ai": "artificial intelligence",
        "langchain": "langchain",
        "llm": "large language model",
        "llms": "large language model",
        "aws": "amazon web services",
    }

    s_for_alias = s.replace("&", " and ").replace("/", " ").replace("-", " ")
    s_for_alias = re.sub(r"\s+", " ", s_for_alias).strip()
    if s_for_alias in alias_map:
        return alias_map[s_for_alias]

    # Remove symbols (keep letters/numbers/spaces).
    s = s.replace("&", " and ")
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _tokenize_role(text: str) -> List[str]:
    s = text.strip().lower()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()

    stop = {"the", "a", "an", "and", "or", "of", "for", "to", "with", "in", "at", "on"}
    return [t for t in s.split(" ") if t and t not in stop]


def _tokenize_skill_tokens(normalized_skill: str) -> Set[str]:
    """
    Deterministic tokenization for skill similarity.
    """

    stop = {
        "and",
        "or",
        "the",
        "a",
        "an",
        "for",
        "to",
        "with",
        "of",
        "in",
        "on",
        "using",
    }
    tokens = re.split(r"\s+", normalized_skill.strip().lower())
    return {t for t in tokens if t and t not in stop}


def _jaccard(a: Set[str], b: Set[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    inter = a.intersection(b)
    union = a.union(b)
    return float(len(inter)) / float(len(union))


async def extract_resume_skills(state: GraphState) -> GraphStatePartial:
    logger.info("Node extract_resume_skills start")
    resume_text = state.get("resume_text", "") or ""

    prompt = PromptsManager.get_extract_resume_skills_prompt().format(
        resume_text=resume_text
    )

    output = cast(
        ResumeSkillsExtractionOutput,
        await _ainvoke_json_llm(
            prompt=prompt, output_model=ResumeSkillsExtractionOutput
        ),
    )
    logger.info("Node extract_resume_skills extracted=%d", len(output.resume_skills))
    return {"resume_skills": output.resume_skills}


async def extract_resume_experience(state: GraphState) -> GraphStatePartial:
    logger.info("Node extract_resume_experience start")
    resume_text = state.get("resume_text", "") or ""

    prompt = PromptsManager.get_extract_resume_experience_prompt().format(
        resume_text=resume_text
    )

    raw_output = cast(
        ResumeExperienceExtractionOutput,
        await _ainvoke_json_llm(
            prompt=prompt, output_model=ResumeExperienceExtractionOutput
        ),
    )

    exp = raw_output.resume_experience
    years_total = float(exp.get("years_total") or 0.0)
    years_relevant = float(exp.get("years_relevant") or 0.0)
    titles = exp.get("titles") or []
    if not isinstance(titles, list):
        titles = []
    titles = [str(t).strip() for t in titles if str(t).strip()]

    normalized_exp: ResumeExperienceModel = ResumeExperienceModel(
        years_total=years_total,
        years_relevant=years_relevant,
        titles=titles,
    )

    logger.info(
        "Node extract_resume_experience years_relevant=%.2f titles=%d",
        normalized_exp.years_relevant,
        len(normalized_exp.titles),
    )
    return {"resume_experience": normalized_exp.model_dump()}


async def extract_resume_education(state: GraphState) -> GraphStatePartial:
    logger.info("Node extract_resume_education start")
    resume_text = state.get("resume_text", "") or ""

    prompt = PromptsManager.get_extract_resume_education_prompt().format(
        resume_text=resume_text
    )

    output = cast(
        ResumeEducationExtractionOutput,
        await _ainvoke_json_llm(
            prompt=prompt, output_model=ResumeEducationExtractionOutput
        ),
    )
    logger.info(
        "Node extract_resume_education extracted=%d", len(output.resume_education)
    )
    return {"resume_education": output.resume_education}


async def extract_jd_requirements(state: GraphState) -> GraphStatePartial:
    logger.info("Node extract_jd_requirements start")
    jd_text = state.get("jd_text", "") or ""

    prompt = PromptsManager.get_extract_jd_requirements_prompt().format(jd_text=jd_text)

    output = cast(
        JDRequirementsExtractionOutput,
        await _ainvoke_json_llm(
            prompt=prompt, output_model=JDRequirementsExtractionOutput
        ),
    )
    logger.info(
        "Node extract_jd_requirements jd_experience=%d skills=%d",
        output.jd_experience,
        len(output.jd_skills),
    )
    return {
        "jd_skills": output.jd_skills,
        "jd_experience": output.jd_experience,
        "jd_role": output.jd_role,
    }


def normalize_entities(state: GraphState) -> GraphStatePartial:
    logger.info("Node normalize_entities start")
    resume_skills = state.get("resume_skills") or []
    jd_skills = state.get("jd_skills") or []

    normalized_resume = [
        _normalize_skill_name(s) for s in resume_skills if s and s.strip()
    ]
    normalized_jd = [_normalize_skill_name(s) for s in jd_skills if s and s.strip()]

    normalized_resume = _unique_preserve_order([s for s in normalized_resume if s])
    normalized_jd = _unique_preserve_order([s for s in normalized_jd if s])

    logger.info(
        "Node normalize_entities normalized_resume=%d normalized_jd=%d",
        len(normalized_resume),
        len(normalized_jd),
    )
    return {
        "normalized_resume_skills": normalized_resume,
        "normalized_jd_skills": normalized_jd,
    }


def match_skills(state: GraphState) -> GraphStatePartial:
    logger.info("Node match_skills start")
    normalized_resume_skills = state.get("normalized_resume_skills") or []
    normalized_jd_skills = state.get("normalized_jd_skills") or []

    if not normalized_jd_skills:
        return {"skills_match_score": 0.0, "missing_skills": []}

    resume_set = set(normalized_resume_skills)
    resume_token_map: Dict[str, Set[str]] = {
        rs: _tokenize_skill_tokens(rs) for rs in normalized_resume_skills
    }

    matched: List[str] = []
    missing: List[str] = []

    for jd_skill in normalized_jd_skills:
        jd_tokens = _tokenize_skill_tokens(jd_skill)
        if not jd_tokens:
            missing.append(jd_skill)
            continue

        if jd_skill in resume_set:
            matched.append(jd_skill)
            continue

        best = 0.0
        for rs_skill in normalized_resume_skills:
            overlap = _jaccard(jd_tokens, resume_token_map.get(rs_skill, set()))
            best = max(best, overlap)
            if best >= Constants.SKILL_MATCH_JACCARD_THRESHOLD:
                break

        if best >= Constants.SKILL_MATCH_JACCARD_THRESHOLD:
            matched.append(jd_skill)
        else:
            missing.append(jd_skill)

    score = float(len(matched)) / float(max(len(normalized_jd_skills), 1))
    score = max(0.0, min(1.0, score))

    logger.info("Node match_skills score=%.3f missing=%d", score, len(missing))
    return {"skills_match_score": score, "missing_skills": missing}


def match_experience(state: GraphState) -> GraphStatePartial:
    logger.info("Node match_experience start")
    resume_experience = state.get("resume_experience") or {}
    jd_experience = int(state.get("jd_experience") or 0)

    years_relevant = 0.0
    try:
        years_relevant = float(resume_experience.get("years_relevant") or 0.0)
    except Exception:
        years_relevant = 0.0

    if jd_experience <= 0:
        score = 1.0
    else:
        score = years_relevant / float(jd_experience)

    score = max(0.0, min(1.0, float(score)))
    logger.info(
        "Node match_experience years_relevant=%.2f jd_experience=%d score=%.3f",
        years_relevant,
        jd_experience,
        score,
    )
    return {"experience_match_score": score}


def match_role(state: GraphState) -> GraphStatePartial:
    logger.info("Node match_role start")
    jd_role = state.get("jd_role") or ""
    resume_experience = state.get("resume_experience") or {}
    titles = resume_experience.get("titles") or []
    if not isinstance(titles, list):
        titles = []

    jd_tokens = set(_tokenize_role(jd_role))
    if not jd_tokens:
        return {"role_match_score": 0.0}

    title_tokens: Set[str] = set()
    for t in titles:
        if not t:
            continue
        title_tokens.update(_tokenize_role(str(t)))

    intersection = jd_tokens.intersection(title_tokens)
    score = float(len(intersection)) / float(max(len(jd_tokens), 1))
    score = max(0.0, min(1.0, score))

    logger.info("Node match_role score=%.3f jd_tokens=%d", score, len(jd_tokens))
    return {"role_match_score": score}


def compute_score(state: GraphState) -> GraphStatePartial:
    logger.info("Node compute_score start")
    skills_score = float(state.get("skills_match_score") or 0.0)
    experience_score = float(state.get("experience_match_score") or 0.0)
    role_score = float(state.get("role_match_score") or 0.0)

    final = 0.5 * skills_score + 0.3 * experience_score + 0.2 * role_score
    final = max(0.0, min(1.0, float(final)))

    logger.info(
        "Node compute_score skills=%.3f exp=%.3f role=%.3f final=%.3f",
        skills_score,
        experience_score,
        role_score,
        final,
    )
    return {"final_score": final}


async def generate_explanation(state: GraphState) -> GraphStatePartial:
    logger.info("Node generate_explanation start")
    skills_score = float(state.get("skills_match_score") or 0.0)
    exp_score = float(state.get("experience_match_score") or 0.0)
    role_score = float(state.get("role_match_score") or 0.0)
    final_score = float(state.get("final_score") or 0.0)
    missing = state.get("missing_skills") or []
    jd_role = state.get("jd_role") or ""

    prompt = PromptsManager.get_generate_explanation_prompt().format(
        jd_role_json=json.dumps(jd_role),
        skills_score=skills_score,
        experience_match_score=exp_score,
        role_match_score=role_score,
        final_score=final_score,
        missing_skills_json=json.dumps(missing),
    )

    output = cast(
        ExplanationOutput,
        await _ainvoke_json_llm(
            prompt=prompt,
            output_model=ExplanationOutput,
            system_content=PromptsManager.system_generate_explanation(),
        ),
    )

    logger.info(
        "Node generate_explanation strengths=%d weaknesses=%d",
        len(output.strengths),
        len(output.weaknesses),
    )
    return {
        "explanation": output.explanation,
        "strengths": output.strengths,
        "weaknesses": output.weaknesses,
    }
