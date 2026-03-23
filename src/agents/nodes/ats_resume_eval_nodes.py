import asyncio
import json
import logging
import os
import re
from typing import Dict, List, Optional, Set, cast

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel

from langchain_openai import ChatOpenAI

from src.agents.prompts import PromptsManager
from src.agents.state import (
    ExplanationOutput,
    ExtractionVerifyOutput,
    GraphState,
    GraphStatePartial,
    JDRequirementsExtractionOutput,
    JudgeFinalOutput,
    RequirementConstraintEvaluationOutput,
    RequirementConstraintsExtractionOutput,
    ResumeEducationExtractionOutput,
    ResumeExperienceExtractionOutput,
    ResumeExperienceModel,
    ResumeSkillsExtractionOutput,
)
from src.comlib.llms import LLMs
from src.config.env_vars import GlobalConfig
from src.utils.constants import Constants, TECH_SIGNAL_TOKENS
from src.utils.grounding import filter_grounded, is_grounded_in_text
from src.utils.text_chunking import chunk_text

logger = logging.getLogger(__name__)


def _build_llm() -> ChatOpenAI:
    # `temperature=0.0` makes extraction/explanation deterministic.
    return LLMs().get_model(GlobalConfig.llm.model_to_use.replace("-", "_"))


def _extract_json_from_text(text: str) -> str:
    """
    Best-effort JSON extraction for when the model wraps output in code blocks.
    """

    raw = (text or "").strip()
    if not raw:
        raise ValueError("Empty LLM response.")

    # Fast path: already valid JSON object string.
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            return raw
    except Exception:
        pass

    # Remove markdown fences if present.
    fence_match = re.search(r"```(?:json)?\s*(.*?)\s*```", raw, flags=re.DOTALL | re.IGNORECASE)
    if fence_match:
        raw = cast(str, fence_match.group(1)).strip()
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                return raw
        except Exception:
            pass

    # Robust fallback: extract first balanced top-level JSON object.
    start = raw.find("{")
    if start < 0:
        raise ValueError("No JSON object found in LLM response.")

    depth = 0
    in_string = False
    escape = False
    end = -1
    for i in range(start, len(raw)):
        ch = raw[i]
        if in_string:
            if escape:
                escape = False
                continue
            if ch == "\\":
                escape = True
                continue
            if ch == '"':
                in_string = False
            continue

        if ch == '"':
            in_string = True
            continue
        if ch == "{":
            depth += 1
            continue
        if ch == "}":
            depth -= 1
            if depth == 0:
                end = i
                break

    if end < 0:
        raise ValueError("No balanced JSON object found in LLM response.")

    candidate = raw[start : end + 1]
    # Validate candidate; if this fails, surface precise parse error.
    json.loads(candidate)
    return candidate


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
        "k8s": "kubernetes",
        "k8": "kubernetes",
        "apis": "api",
        "microservices": "microservice",
        "microservice": "microservice",
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


# Narrow token canonicalization for overlap (plural/synonym), not broad domain stripping.
_TOKEN_CANON: Dict[str, str] = {
    "apis": "api",
    "api": "api",
    "microservices": "microservice",
    "microservice": "microservice",
    "k8s": "kubernetes",
    "kubernetes": "kubernetes",
    "saas": "saas",
    "llms": "llm",
    "llm": "llm",
    "integrations": "integration",
    "containers": "container",
    "pipelines": "pipeline",
}


def _canon_token(t: str) -> str:
    t = t.strip().lower()
    t = _TOKEN_CANON.get(t, t)
    # Lightweight morphology to improve cross-domain lexical variance:
    # plural/surface-form reduction without external NLP dependencies.
    if len(t) > 4 and t.endswith("ies"):
        t = f"{t[:-3]}y"
    elif len(t) > 5 and t.endswith("ing"):
        t = t[:-3]
    elif len(t) > 5 and t.endswith("ed"):
        t = t[:-2]
    elif len(t) > 4 and t.endswith("s") and not t.endswith(("ss", "us", "is")):
        t = t[:-1]
    return t


def _canon_token_set(tokens: Set[str]) -> Set[str]:
    return {_canon_token(t) for t in tokens if t}


# Hype-only tokens excluded from JD requirement token coverage denominator (keep minimal).
_JD_HYPE_TOKENS = frozenset(
    {
        "world",
        "class",
        "rockstar",
        "ninja",
        "guru",
        "superstar",
    }
)


def _jd_tokens_for_match(jd_skill: str) -> Set[str]:
    raw = _tokenize_skill_tokens(jd_skill)
    return {t for t in raw if t not in _JD_HYPE_TOKENS}


def _resume_text_tokens(raw: str) -> Set[str]:
    """
    Word-level tokens from full resume text for coverage against JD requirements.
    """

    if not raw.strip():
        return set()
    s = raw.lower()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    stopw = {
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
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "as",
        "by",
        "an",
    }
    out: Set[str] = set()
    for w in s.split():
        if len(w) < 2:
            continue
        if w in stopw:
            continue
        out.add(w)
    return out


def _build_resume_token_bag(
    normalized_resume_skills: List[str], resume_text: str
) -> Set[str]:
    bag: Set[str] = set()
    for rs in normalized_resume_skills:
        bag |= _tokenize_skill_tokens(rs)
    bag |= _resume_text_tokens(resume_text or "")
    return _canon_token_set(bag)


def _jd_skill_matched_against_bag(
    jd_skill: str,
    resume_set: Set[str],
    resume_token_map: Dict[str, Set[str]],
    normalized_resume_skills: List[str],
    resume_bag_canon: Set[str],
) -> bool:
    if jd_skill in resume_set:
        return True

    jd_tokens = _jd_tokens_for_match(jd_skill)
    if not jd_tokens:
        return False

    jd_canon = _canon_token_set(jd_tokens)
    if jd_canon and jd_canon <= resume_bag_canon:
        return True

    if jd_canon:
        threshold = (
            Constants.SKILL_TOKEN_COVERAGE_THRESHOLD_LONG
            if len(jd_canon) >= 3
            else Constants.SKILL_TOKEN_COVERAGE_THRESHOLD
        )
        overlap = len(jd_canon & resume_bag_canon) / float(len(jd_canon))
        if overlap >= threshold:
            return True

    best = 0.0
    for rs_skill in normalized_resume_skills:
        overlap = _jaccard(jd_tokens, resume_token_map.get(rs_skill, set()))
        best = max(best, overlap)
        if best >= Constants.SKILL_MATCH_JACCARD_THRESHOLD:
            return True

    return False


def _inferred_skill_is_grounded(
    jd_skill: str, resume_text: str, resume_bag_canon: Set[str]
) -> bool:
    # Require a minimal token grounding check before accepting LLM-inferred matches.
    jd_tokens = _jd_tokens_for_match(jd_skill)
    if not jd_tokens:
        return False
    jd_canon = _canon_token_set(jd_tokens)
    if not jd_canon:
        return False
    overlap = len(jd_canon & resume_bag_canon) / float(len(jd_canon))
    threshold = (
        Constants.SKILL_TOKEN_COVERAGE_THRESHOLD_LONG
        if len(jd_canon) >= 3
        else Constants.SKILL_TOKEN_COVERAGE_THRESHOLD
    )
    if overlap >= threshold:
        return True
    # Secondary guard for phrases that appear in text after normalization.
    return is_grounded_in_text(jd_skill, resume_text)


def _jd_tech_ratio(normalized_jd_skills: List[str]) -> float:
    if not normalized_jd_skills:
        return 0.0
    hits = 0
    for s in normalized_jd_skills:
        toks = _tokenize_skill_tokens(s)
        if toks & TECH_SIGNAL_TOKENS:
            hits += 1
    return float(hits) / float(len(normalized_jd_skills))


def _normalize_role_string(raw: str) -> str:
    s = raw.strip().lower()
    if not s:
        return s
    replacements = [
        (r"\bsr\.?\b", "senior"),
        (r"\bjr\.?\b", "junior"),
        (r"\bvp\b", "vice president"),
        (r"\bswe\b", "software engineer"),
        (r"\bpm\b", "product manager"),
        (r"\bds\b", "data scientist"),
    ]
    for pat, repl in replacements:
        s = re.sub(pat, repl, s, flags=re.IGNORECASE)
    return s


def _education_requirement_met(
    req: str,
    resume_education: List[str],
    resume_text: str,
) -> bool:
    blob = " ".join(resume_education) + " " + (resume_text or "")
    norm_req = _normalize_skill_name(req)
    if not norm_req:
        return False
    if is_grounded_in_text(norm_req, blob):
        return True
    req_toks = _tokenize_skill_tokens(norm_req)
    if not req_toks:
        return False
    bag = _resume_text_tokens(blob)
    bag |= _canon_token_set(_tokenize_skill_tokens(" ".join(resume_education)))
    inter = _canon_token_set(req_toks) & bag
    return len(inter) >= max(1, int(0.45 * len(_canon_token_set(req_toks))))


async def _extract_resume_skills_llm(resume_text: str) -> ResumeSkillsExtractionOutput:
    prompt = PromptsManager.get_extract_resume_skills_prompt().format(
        resume_text=resume_text
    )
    return cast(
        ResumeSkillsExtractionOutput,
        await _ainvoke_json_llm(
            prompt=prompt, output_model=ResumeSkillsExtractionOutput
        ),
    )


async def extract_resume_skills(state: GraphState) -> GraphStatePartial:
    logger.info("Node extract_resume_skills start")
    resume_text = state.get("resume_text", "") or ""
    chunks = chunk_text(
        resume_text,
        max_chars=Constants.CHUNK_MAX_CHARS,
        overlap_chars=Constants.CHUNK_OVERLAP_CHARS,
    )
    outputs = await asyncio.gather(*[_extract_resume_skills_llm(c) for c in chunks])
    merged: List[str] = []
    for o in outputs:
        merged.extend(o.resume_skills)
    merged = _unique_preserve_order([s.strip() for s in merged if s and s.strip()])[:30]
    logger.info(f"Node extract_resume_skills extracted={merged}")
    return {"resume_skills": merged}


async def _extract_resume_experience_llm(
    resume_text: str,
) -> ResumeExperienceExtractionOutput:
    prompt = PromptsManager.get_extract_resume_experience_prompt().format(
        resume_text=resume_text
    )
    return cast(
        ResumeExperienceExtractionOutput,
        await _ainvoke_json_llm(
            prompt=prompt, output_model=ResumeExperienceExtractionOutput
        ),
    )


async def extract_resume_experience(state: GraphState) -> GraphStatePartial:
    logger.info("Node extract_resume_experience start")
    resume_text = state.get("resume_text", "") or ""
    chunks = chunk_text(
        resume_text,
        max_chars=Constants.CHUNK_MAX_CHARS,
        overlap_chars=Constants.CHUNK_OVERLAP_CHARS,
    )
    raw_outputs = await asyncio.gather(
        *[_extract_resume_experience_llm(c) for c in chunks]
    )

    years_total = 0.0
    years_relevant = 0.0
    all_titles: List[str] = []
    for raw_output in raw_outputs:
        exp = raw_output.resume_experience
        try:
            years_total = max(years_total, float(exp.get("years_total") or 0.0))
        except Exception:
            pass
        try:
            years_relevant = max(
                years_relevant, float(exp.get("years_relevant") or 0.0)
            )
        except Exception:
            pass
        titles = exp.get("titles") or []
        if not isinstance(titles, list):
            titles = []
        for t in titles:
            s = str(t).strip()
            if s:
                all_titles.append(s)

    titles_merged = _unique_preserve_order(all_titles)[:8]

    normalized_exp = ResumeExperienceModel(
        years_total=years_total,
        years_relevant=years_relevant,
        titles=titles_merged,
    )

    logger.info(
        "Node extract_resume_experience years_relevant=%.2f titles=%d",
        normalized_exp.years_relevant,
        len(normalized_exp.titles),
    )
    return {"resume_experience": normalized_exp.model_dump()}


async def _extract_resume_education_llm(
    resume_text: str,
) -> ResumeEducationExtractionOutput:
    prompt = PromptsManager.get_extract_resume_education_prompt().format(
        resume_text=resume_text
    )
    return cast(
        ResumeEducationExtractionOutput,
        await _ainvoke_json_llm(
            prompt=prompt, output_model=ResumeEducationExtractionOutput
        ),
    )


async def extract_resume_education(state: GraphState) -> GraphStatePartial:
    logger.info("Node extract_resume_education start")
    resume_text = state.get("resume_text", "") or ""
    chunks = chunk_text(
        resume_text,
        max_chars=Constants.CHUNK_MAX_CHARS,
        overlap_chars=Constants.CHUNK_OVERLAP_CHARS,
    )
    outputs = await asyncio.gather(*[_extract_resume_education_llm(c) for c in chunks])
    merged: List[str] = []
    for o in outputs:
        merged.extend(o.resume_education)
    merged = _unique_preserve_order([s.strip() for s in merged if s and s.strip()])[:10]
    logger.info(f"Node extract_resume_education extracted={merged}")
    return {"resume_education": merged}


async def _extract_jd_requirements_llm(jd_text: str) -> JDRequirementsExtractionOutput:
    prompt = PromptsManager.get_extract_jd_requirements_prompt().format(jd_text=jd_text)
    return cast(
        JDRequirementsExtractionOutput,
        await _ainvoke_json_llm(
            prompt=prompt, output_model=JDRequirementsExtractionOutput
        ),
    )


async def extract_jd_requirements(state: GraphState) -> GraphStatePartial:
    logger.info("Node extract_jd_requirements start")
    jd_text = state.get("jd_text", "") or ""
    chunks = chunk_text(
        jd_text,
        max_chars=Constants.CHUNK_MAX_CHARS,
        overlap_chars=Constants.CHUNK_OVERLAP_CHARS,
    )
    outputs = await asyncio.gather(*[_extract_jd_requirements_llm(c) for c in chunks])

    merged_skills: List[str] = []
    for o in outputs:
        merged_skills.extend(o.jd_skills)
    merged_skills = _unique_preserve_order(
        [s.strip() for s in merged_skills if s and s.strip()]
    )[:30]

    merged_edu: List[str] = []
    for o in outputs:
        merged_edu.extend(getattr(o, "jd_education_requirements", None) or [])
    merged_edu = _unique_preserve_order(
        [s.strip() for s in merged_edu if s and s.strip()]
    )[:5]

    jd_experience = max((int(o.jd_experience or 0) for o in outputs), default=0)

    jd_role = ""
    for o in outputs:
        r = str(o.jd_role or "").strip()
        if r:
            jd_role = r
            break

    logger.info(
        f"Node extract_jd_requirements jd_experience={jd_experience} "
        f"skills={merged_skills} jd_education={merged_edu}"
    )
    return {
        "jd_skills": merged_skills,
        "jd_experience": jd_experience,
        "jd_role": jd_role,
        "jd_education_requirements": merged_edu,
    }


async def _extract_requirement_constraints_llm(
    jd_text: str,
) -> RequirementConstraintsExtractionOutput:
    prompt = PromptsManager.get_extract_requirement_constraints_prompt().format(
        jd_text=jd_text
    )
    return cast(
        RequirementConstraintsExtractionOutput,
        await _ainvoke_json_llm(
            prompt=prompt,
            output_model=RequirementConstraintsExtractionOutput,
        ),
    )


async def extract_requirement_constraints(state: GraphState) -> GraphStatePartial:
    logger.info("Node extract_requirement_constraints start")
    jd_text = state.get("jd_text", "") or ""
    if not jd_text.strip():
        return {"requirement_constraints": []}
    out = await _extract_requirement_constraints_llm(jd_text)
    constraints: List[Dict[str, object]] = []
    for c in out.requirement_constraints:
        label = (c.requirement_label or "").strip()
        if not label:
            continue
        strict = (c.strictness or "soft").strip().lower()
        # Default-soft policy is owned by constraint flow.
        strict = "hard" if strict == "hard" else "soft"
        constraints.append(
            {
                "requirement_label": _normalize_skill_name(label),
                "min_years": max(0.0, float(c.min_years or 0.0)),
                "strictness": strict,
                "strictness_evidence": (c.strictness_evidence or "").strip(),
            }
        )
    logger.info(
        "Node extract_requirement_constraints extracted=%d",
        len(constraints),
    )
    return {"requirement_constraints": constraints}


async def verify_extractions(state: GraphState) -> GraphStatePartial:
    """
    Second LLM pass: drop extraction items not clearly supported by source text.
    """

    logger.info("Node verify_extractions start")
    resume_text = state.get("resume_text") or ""
    jd_text = state.get("jd_text") or ""
    prompt = PromptsManager.get_verify_extractions_prompt().format(
        resume_text=resume_text,
        jd_text=jd_text,
        resume_skills_json=json.dumps(state.get("resume_skills") or []),
        jd_skills_json=json.dumps(state.get("jd_skills") or []),
        resume_education_json=json.dumps(state.get("resume_education") or []),
        jd_education_json=json.dumps(state.get("jd_education_requirements") or []),
    )
    out = cast(
        ExtractionVerifyOutput,
        await _ainvoke_json_llm(prompt=prompt, output_model=ExtractionVerifyOutput),
    )
    orig_resume_skills = [
        s for s in (state.get("resume_skills") or []) if s and s.strip()
    ]
    orig_jd_skills = [s for s in (state.get("jd_skills") or []) if s and s.strip()]
    orig_resume_edu = [
        s for s in (state.get("resume_education") or []) if s and s.strip()
    ]
    orig_jd_edu = [
        s for s in (state.get("jd_education_requirements") or []) if s and s.strip()
    ]

    def _is_over_pruned(filtered: List[str], original: List[str]) -> bool:
        if not original:
            return False
        # Keep verifier high-precision behavior, but prevent recall collapse.
        return (len(filtered) / float(len(original))) < 0.4

    resume_skills = (
        _unique_preserve_order(orig_resume_skills + out.resume_skills)
        if _is_over_pruned(out.resume_skills, orig_resume_skills)
        else out.resume_skills
    )
    jd_skills = (
        _unique_preserve_order(orig_jd_skills + out.jd_skills)
        if _is_over_pruned(out.jd_skills, orig_jd_skills)
        else out.jd_skills
    )
    resume_education = (
        _unique_preserve_order(orig_resume_edu + out.resume_education)
        if _is_over_pruned(out.resume_education, orig_resume_edu)
        else out.resume_education
    )
    jd_education = (
        _unique_preserve_order(orig_jd_edu + out.jd_education_requirements)
        if _is_over_pruned(out.jd_education_requirements, orig_jd_edu)
        else out.jd_education_requirements
    )
    logger.info(
        "Node verify_extractions filtered resume_skills=%d jd_skills=%d",
        len(resume_skills),
        len(jd_skills),
    )
    return {
        "resume_skills": resume_skills,
        "jd_skills": jd_skills,
        "resume_education": resume_education,
        "jd_education_requirements": jd_education,
    }


def normalize_entities(state: GraphState) -> GraphStatePartial:
    logger.info("Node normalize_entities start")
    resume_text = state.get("resume_text") or ""
    jd_text = state.get("jd_text") or ""

    # LLM-first flow: verifier/alignment does heavy lifting. Keep deterministic grounding
    # only as a light pass for education where precision matters more.
    resume_skills = [s for s in (state.get("resume_skills") or []) if s and s.strip()]
    jd_skills = [s for s in (state.get("jd_skills") or []) if s and s.strip()]
    resume_education = filter_grounded(
        [s for s in (state.get("resume_education") or []) if s and s.strip()],
        resume_text,
    )
    jd_education_req = filter_grounded(
        [s for s in (state.get("jd_education_requirements") or []) if s and s.strip()],
        jd_text,
    )

    normalized_resume = [
        _normalize_skill_name(s) for s in resume_skills if s and s.strip()
    ]
    normalized_jd = [_normalize_skill_name(s) for s in jd_skills if s and s.strip()]
    normalized_jd_edu = [
        _normalize_skill_name(s) for s in jd_education_req if s and s.strip()
    ]

    normalized_resume = _unique_preserve_order([s for s in normalized_resume if s])
    normalized_jd = _unique_preserve_order([s for s in normalized_jd if s])
    normalized_jd_edu = _unique_preserve_order([s for s in normalized_jd_edu if s])

    logger.info(
        f"Node normalize_entities normalized_resume={normalized_resume} normalized_jd={normalized_jd} "
        f"normalized_jd_edu={normalized_jd_edu}"
    )
    return {
        "normalized_resume_skills": normalized_resume,
        "normalized_jd_skills": normalized_jd,
        "normalized_jd_education": normalized_jd_edu,
    }


async def match_skills(state: GraphState) -> GraphStatePartial:
    logger.info("Node match_skills start")
    normalized_resume_skills = state.get("normalized_resume_skills") or []
    normalized_jd_skills = state.get("normalized_jd_skills") or []
    resume_text = state.get("resume_text") or ""

    if not normalized_jd_skills:
        return {"skills_match_score": 0.0, "missing_skills": []}

    # Deterministic matcher kept as fallback if LLM alignment fails.
    resume_set = set(normalized_resume_skills)
    resume_token_map: Dict[str, Set[str]] = {
        rs: _tokenize_skill_tokens(rs) for rs in normalized_resume_skills
    }
    resume_bag_canon = _build_resume_token_bag(normalized_resume_skills, resume_text)

    # Primary path: LLM evidence alignment for all JD skills.
    class _AlignedSkillsOutput(BaseModel):
        matched_skills: List[str] = []
        missing_skills: List[str] = []

    class _InferredSkillsOutput(BaseModel):
        inferred_matched_skills: List[str] = []

    matched_set: Set[str] = set()
    llm_used = False
    if resume_text.strip() and os.getenv("OPENAI_API_KEY"):
        prompt = PromptsManager.get_align_jd_skills_prompt().format(
            resume_text=resume_text,
            resume_skills_json=json.dumps(normalized_resume_skills),
            jd_skills_json=json.dumps(normalized_jd_skills),
        )
        try:
            out = cast(
                _AlignedSkillsOutput,
                await _ainvoke_json_llm(
                    prompt=prompt,
                    output_model=_AlignedSkillsOutput,
                    system_content=PromptsManager.system_skill_alignment(),
                ),
            )
            llm_used = True
            llm_matched = [
                s for s in out.matched_skills if s in set(normalized_jd_skills)
            ]
            matched_set.update(llm_matched)
        except Exception as e:
            logger.warning(
                "LLM skill alignment failed; falling back to deterministic matcher: %s",
                e,
            )

    # Always run deterministic matcher and union with LLM results.
    for jd_skill in normalized_jd_skills:
        if _jd_skill_matched_against_bag(
            jd_skill,
            resume_set,
            resume_token_map,
            normalized_resume_skills,
            resume_bag_canon,
        ):
            matched_set.add(jd_skill)

    # Inference pass: let LLM semantically infer unresolved requirements from resume intent.
    unresolved = [s for s in normalized_jd_skills if s not in matched_set]
    if unresolved and resume_text.strip() and os.getenv("OPENAI_API_KEY"):
        infer_prompt = PromptsManager.get_infer_unresolved_jd_skills_prompt().format(
            resume_text=resume_text,
            resume_skills_json=json.dumps(normalized_resume_skills),
            unresolved_jd_skills_json=json.dumps(unresolved),
        )
        try:
            infer_out = cast(
                _InferredSkillsOutput,
                await _ainvoke_json_llm(
                    prompt=infer_prompt,
                    output_model=_InferredSkillsOutput,
                    system_content=PromptsManager.system_skill_inference(),
                ),
            )
            inferred = [
                s
                for s in infer_out.inferred_matched_skills
                if s in set(unresolved)
                and _inferred_skill_is_grounded(s, resume_text, resume_bag_canon)
            ]
            matched_set.update(inferred)
        except Exception as e:
            logger.warning("LLM unresolved skill inference failed: %s", e)

    matched = [s for s in normalized_jd_skills if s in matched_set]
    missing = [s for s in normalized_jd_skills if s not in matched_set]

    score = float(len(matched)) / float(max(len(normalized_jd_skills), 1))
    score = max(0.0, min(1.0, score))

    logger.info("Node match_skills score=%s missing=%s", score, missing)
    return {"skills_match_score": score, "missing_skills": missing}


def match_education(state: GraphState) -> GraphStatePartial:
    logger.info("Node match_education start")
    normalized_jd_edu = state.get("normalized_jd_education") or []
    resume_education = state.get("resume_education") or []
    resume_text = state.get("resume_text") or ""

    if not normalized_jd_edu:
        return {"education_match_score": 1.0, "missing_education": []}

    matched: List[str] = []
    missing: List[str] = []
    for req in normalized_jd_edu:
        if _education_requirement_met(req, resume_education, resume_text):
            matched.append(req)
        else:
            missing.append(req)

    score = float(len(matched)) / float(max(len(normalized_jd_edu), 1))
    score = max(0.0, min(1.0, score))
    logger.info(f"Node match_education score={score} missing={missing}")
    return {"education_match_score": score, "missing_education": missing}


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
        score = Constants.EXPERIENCE_MATCH_NEUTRAL
    else:
        score = years_relevant / float(jd_experience)

    score = max(0.0, min(1.0, float(score)))
    logger.info(
        f"Node match_experience years_relevant={years_relevant} jd_experience={jd_experience} score={score}"
    )
    return {"experience_match_score": score}


async def evaluate_requirement_constraints(state: GraphState) -> GraphStatePartial:
    logger.info("Node evaluate_requirement_constraints start")
    constraints = state.get("requirement_constraints") or []
    resume_text = state.get("resume_text") or ""
    normalized_resume_skills = state.get("normalized_resume_skills") or []
    if not constraints:
        return {
            "requirement_constraint_score": 1.0,
            "hard_requirement_misses": [],
            "constraint_findings": [],
        }

    prompt = PromptsManager.get_evaluate_requirement_constraints_prompt().format(
        resume_text=resume_text,
        resume_skills_json=json.dumps(normalized_resume_skills),
        requirement_constraints_json=json.dumps(constraints),
    )
    out = cast(
        RequirementConstraintEvaluationOutput,
        await _ainvoke_json_llm(
            prompt=prompt,
            output_model=RequirementConstraintEvaluationOutput,
            system_content=PromptsManager.system_constraint_evaluation(),
        ),
    )

    eval_map = {(e.requirement_label or "").strip().lower(): e for e in out.evaluations}
    findings: List[Dict[str, object]] = []
    hard_misses: List[Dict[str, object]] = []
    score_acc = 0.0
    counted = 0
    for c in constraints:
        label = str(c.get("requirement_label") or "").strip()
        min_years = float(c.get("min_years") or 0.0)
        strictness = str(c.get("strictness") or "soft").lower()
        e = eval_map.get(label.lower())
        matched = bool(e.matched) if e else False
        est_years = max(0.0, float(e.estimated_years or 0.0)) if e else 0.0
        confidence = max(0.0, min(1.0, float(e.confidence or 0.0))) if e else 0.0
        reasoning = (e.reasoning or "").strip() if e else ""
        if min_years > 0:
            ratio = min(1.0, est_years / min_years)
            local_score = ratio if matched else 0.0
        else:
            local_score = 1.0 if matched else 0.0
        score_acc += local_score
        counted += 1
        finding = {
            "requirement_label": label,
            "strictness": strictness,
            "min_years": min_years,
            "matched": matched,
            "estimated_years": est_years,
            "confidence": confidence,
            "reasoning": reasoning,
        }
        findings.append(finding)
        if strictness == "hard" and (
            not matched or (min_years > 0 and est_years < min_years)
        ):
            hard_misses.append(finding)

    constraint_score = score_acc / float(max(counted, 1))
    logger.info(
        "Node evaluate_requirement_constraints score=%.3f hard_misses=%d",
        constraint_score,
        len(hard_misses),
    )
    return {
        "requirement_constraint_score": max(0.0, min(1.0, constraint_score)),
        "hard_requirement_misses": hard_misses,
        "constraint_findings": findings,
    }


def match_role(state: GraphState) -> GraphStatePartial:
    logger.info("Node match_role start")
    jd_role = _normalize_role_string(state.get("jd_role") or "")
    resume_experience = state.get("resume_experience") or {}
    titles = resume_experience.get("titles") or []
    if not isinstance(titles, list):
        titles = []

    jd_tokens = set(_tokenize_role(jd_role))
    if not jd_tokens:
        return {"role_match_score": Constants.EXPERIENCE_MATCH_NEUTRAL}

    title_tokens: Set[str] = set()
    for t in titles:
        if not t:
            continue
        title_tokens.update(_tokenize_role(_normalize_role_string(str(t))))

    intersection = jd_tokens.intersection(title_tokens)
    score = float(len(intersection)) / float(max(len(jd_tokens), 1))
    score = max(0.0, min(1.0, score))

    logger.info(f"Node match_role score={score} jd_tokens={jd_tokens}")
    return {"role_match_score": score}


def compute_score(state: GraphState) -> GraphStatePartial:
    logger.info("Node compute_score start")
    skills_score = float(state.get("skills_match_score") or 0.0)
    experience_score = float(state.get("experience_match_score") or 0.0)
    role_score = float(state.get("role_match_score") or 0.0)
    education_score = float(state.get("education_match_score") or 1.0)
    constraint_score = float(state.get("requirement_constraint_score") or 1.0)
    hard_requirement_misses = state.get("hard_requirement_misses") or []
    normalized_jd_skills = state.get("normalized_jd_skills") or []
    normalized_jd_edu = state.get("normalized_jd_education") or []

    tech_ratio = _jd_tech_ratio(normalized_jd_skills)
    has_edu_req = bool(normalized_jd_edu)

    if has_edu_req:
        if tech_ratio >= Constants.JD_TECH_RATIO_THRESHOLD:
            ws, we, wr, wu = Constants.WEIGHTS_TECH_HEAVY_4
        else:
            ws, we, wr, wu = Constants.WEIGHTS_BALANCED_4
        final = (
            ws * skills_score
            + we * experience_score
            + wr * role_score
            + wu * education_score
        )
        wlog = (ws, we, wr, wu)
    else:
        if tech_ratio >= Constants.JD_TECH_RATIO_THRESHOLD:
            ws, we, wr = Constants.WEIGHTS_TECH_HEAVY
        else:
            ws, we, wr = Constants.WEIGHTS_BALANCED
        final = ws * skills_score + we * experience_score + wr * role_score
        wlog = (ws, we, wr)

    final = max(0.0, min(1.0, float(final)))
    final = (1.0 - Constants.CONSTRAINT_SCORE_BLEND_WEIGHT) * final + (
        Constants.CONSTRAINT_SCORE_BLEND_WEIGHT * constraint_score
    )
    hard_penalty = min(
        Constants.HARD_REQUIREMENT_MISS_PENALTY_CAP,
        Constants.HARD_REQUIREMENT_MISS_PENALTY * float(len(hard_requirement_misses)),
    )
    final = max(0.0, min(1.0, float(final - hard_penalty)))

    logger.info(
        f"Node compute_score skills={skills_score} exp={experience_score} role={role_score} "
        f"edu={education_score} constraints={constraint_score} hard_misses={len(hard_requirement_misses)} "
        f"jd_tech_ratio={tech_ratio:.2f} has_edu_req={has_edu_req} weights={wlog} final={final}"
    )
    return {"final_score": final}


async def generate_explanation(state: GraphState) -> GraphStatePartial:
    logger.info("Node generate_explanation start")
    skills_score = float(state.get("skills_match_score") or 0.0)
    exp_score = float(state.get("experience_match_score") or 0.0)
    role_score = float(state.get("role_match_score") or 0.0)
    education_score = float(state.get("education_match_score") or 1.0)
    final_score = float(state.get("final_score") or 0.0)
    missing = state.get("missing_skills") or []
    missing_edu = state.get("missing_education") or []
    jd_role = state.get("jd_role") or ""

    prompt = PromptsManager.get_generate_explanation_prompt().format(
        jd_role_json=json.dumps(jd_role),
        skills_score=skills_score,
        experience_match_score=exp_score,
        role_match_score=role_score,
        education_match_score=education_score,
        final_score=final_score,
        missing_skills_json=json.dumps(missing),
        missing_education_json=json.dumps(missing_edu),
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
        f"Node generate_explanation strengths={output.strengths} weaknesses={output.weaknesses}"
    )
    return {
        "explanation": output.explanation,
        "strengths": output.strengths,
        "weaknesses": output.weaknesses,
    }


async def judge_final_evaluation(state: GraphState) -> GraphStatePartial:
    logger.info("Node judge_final_evaluation start")
    prompt = PromptsManager.get_judge_final_output_prompt().format(
        final_score=float(state.get("final_score") or 0.0),
        skills_match_score=float(state.get("skills_match_score") or 0.0),
        experience_match_score=float(state.get("experience_match_score") or 0.0),
        role_match_score=float(state.get("role_match_score") or 0.0),
        education_match_score=float(state.get("education_match_score") or 1.0),
        requirement_constraint_score=float(
            state.get("requirement_constraint_score") or 1.0
        ),
        hard_requirement_misses_json=json.dumps(
            state.get("hard_requirement_misses") or []
        ),
        constraint_findings_json=json.dumps(state.get("constraint_findings") or []),
        explanation_json=json.dumps(state.get("explanation") or ""),
        weaknesses_json=json.dumps(state.get("weaknesses") or []),
    )
    out = cast(
        JudgeFinalOutput,
        await _ainvoke_json_llm(
            prompt=prompt,
            output_model=JudgeFinalOutput,
            system_content=PromptsManager.system_judge_final_output(),
        ),
    )
    verdict = (out.judge_verdict or "pass").strip().lower()
    verdict = "review" if verdict == "review" else "pass"
    confidence = max(0.0, min(1.0, float(out.judge_confidence or 0.0)))
    notes = [n.strip() for n in (out.judge_notes or []) if n and n.strip()][:8]
    logger.info(
        "Node judge_final_evaluation verdict=%s confidence=%.2f notes=%d",
        verdict,
        confidence,
        len(notes),
    )
    return {
        "judge_verdict": verdict,
        "judge_confidence": confidence,
        "judge_notes": notes,
    }
