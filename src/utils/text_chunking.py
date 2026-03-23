"""
Split long documents for LLM extraction with overlap so boundaries do not drop bullets.
"""

from __future__ import annotations

import re
from typing import List


def chunk_text(
    text: str,
    *,
    max_chars: int = 6000,
    overlap_chars: int = 400,
) -> List[str]:
    """
    Return one or more chunks. Short text is returned as a single chunk unchanged.
    Overlap reduces the chance that a requirement or bullet is split across chunks.
    """

    t = text.strip()
    if not t:
        return [""]
    if len(t) <= max_chars:
        return [t]

    chunks: List[str] = []
    start = 0
    n = len(t)
    step = max(1, max_chars - overlap_chars)

    while start < n:
        end = min(start + max_chars, n)
        chunk = t[start:end]
        chunks.append(chunk)
        if end >= n:
            break
        start += step

    return chunks
