"""
Deterministic grounding: keep extracted labels only when supported by source text.
"""

from __future__ import annotations

import re
from typing import Iterable, List


def _normalize_for_match(text: str) -> str:
    s = text.strip().lower()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _significant_tokens(phrase: str) -> List[str]:
    raw = _normalize_for_match(phrase).split()
    out: List[str] = []
    for t in raw:
        if len(t) < 2:
            continue
        if t in {"the", "a", "an", "or", "of", "in", "to", "for", "and", "with"}:
            continue
        out.append(t)
    return out


def is_grounded_in_text(item: str, source_text: str) -> bool:
    """
    Return True if `item` is plausibly supported by `source_text`.
    Uses substring check on normalized text and token hit heuristics.
    """

    if not item.strip() or not source_text.strip():
        return False

    norm_item = _normalize_for_match(item)
    norm_src = _normalize_for_match(source_text)
    if not norm_item:
        return False

    if norm_item in norm_src:
        return True

    toks = _significant_tokens(item)
    if not toks:
        return False

    hits = 0
    for t in toks:
        if len(t) >= 3 and t in norm_src:
            hits += 1
        elif len(t) == 2 and re.search(rf"\b{re.escape(t)}\b", norm_src):
            hits += 1

    # Be intentionally lenient here: grounding is meant as a cheap pre-filter,
    # not a strict "must appear verbatim" constraint. A verifier/alignment step
    # can correct borderline cases.
    if len(toks) <= 2:
        return hits >= 1

    # Require at least ~1/3 of significant tokens to appear.
    needed = (len(toks) + 2) // 3  # ceil(len/3)
    return hits >= max(1, needed)


def filter_grounded(items: Iterable[str], source_text: str) -> List[str]:
    out: List[str] = []
    seen: set[str] = set()
    for raw in items:
        s = (raw or "").strip()
        if not s or s in seen:
            continue
        if is_grounded_in_text(s, source_text):
            seen.add(s)
            out.append(s)
    return out
