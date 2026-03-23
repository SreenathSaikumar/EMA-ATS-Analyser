import logging
import re
import tempfile
import unicodedata
import uuid

from pathlib import Path

from langchain_community.document_loaders import PyMuPDFLoader, TextLoader
from langchain_core.documents import Document

from src.utils.constants import (
    DOC_INJECTION_LINE_PATTERNS,
    DOC_PROMPT_MARKER_PREFIXES,
    Constants,
)

logger = logging.getLogger(__name__)

_PDF_MIME = "application/pdf"


def _is_pdf(mime: str | None, filename: str | None) -> bool:
    m = (mime or "").lower()
    if m == _PDF_MIME or "pdf" in m:
        return True
    return (filename or "").lower().endswith(".pdf")


def _is_plain_text(mime: str | None, filename: str | None) -> bool:
    m = (mime or "").lower()
    if m == "text/plain" or m.startswith("text/"):
        return True
    fn = (filename or "").lower()
    return fn.endswith(".txt") or fn.endswith(".md") or fn.endswith(".text")


def _documents_to_text(docs: list[Document]) -> str:
    return "\n\n".join(d.page_content for d in docs if getattr(d, "page_content", None))


def sanitize_untrusted_document_text(text: str) -> tuple[str, dict[str, int]]:
    if not text:
        return "", {"removed_lines": 0, "trimmed_lines": 0, "sanitized_markers": 0}

    patterns = [re.compile(p, flags=re.IGNORECASE) for p in DOC_INJECTION_LINE_PATTERNS]
    normalized = unicodedata.normalize("NFKC", text)
    # Remove control chars + zero-width/bidi chars commonly used for hidden payloads.
    normalized = "".join(
        ch
        for ch in normalized
        if (ch == "\n" or ch == "\t" or (ord(ch) >= 32 and unicodedata.category(ch) != "Cf"))
    )

    removed_lines = 0
    trimmed_lines = 0
    sanitized_markers = 0
    out_lines: list[str] = []
    blank_run = 0

    for raw_line in normalized.splitlines():
        line = re.sub(r"\s+", " ", raw_line).strip()
        if not line:
            blank_run += 1
            if blank_run <= Constants.DOC_SANITIZE_MAX_CONSECUTIVE_BLANK_LINES:
                out_lines.append("")
            continue
        blank_run = 0

        lowered = line.lower()
        if any(lowered.startswith(prefix) for prefix in DOC_PROMPT_MARKER_PREFIXES):
            line = f"[filtered-marker] {line}"
            sanitized_markers += 1

        if any(p.search(line) for p in patterns):
            removed_lines += 1
            continue

        # Cap repeated character runs and max line length.
        line = re.sub(
            rf"(.)\1{{{Constants.DOC_SANITIZE_MAX_REPEAT_CHAR_SEQ},}}",
            lambda m: m.group(1) * Constants.DOC_SANITIZE_MAX_REPEAT_CHAR_SEQ,
            line,
        )
        if len(line) > Constants.DOC_SANITIZE_MAX_LINE_LEN:
            line = line[: Constants.DOC_SANITIZE_MAX_LINE_LEN]
            trimmed_lines += 1

        out_lines.append(line)

    sanitized = "\n".join(out_lines).strip()
    meta = {
        "removed_lines": removed_lines,
        "trimmed_lines": trimmed_lines,
        "sanitized_markers": sanitized_markers,
    }
    return sanitized, meta


async def load_resume_text_from_bytes(
    resume_bytes: bytes,
    resume_file_type: str | None,
    resume_file_name: str | None,
) -> str:
    """
    Load plain text from resume bytes using LangChain document loaders (async).
    """
    if not resume_bytes:
        return ""

    if _is_pdf(resume_file_type, resume_file_name):
        suffix = ".pdf"
    elif _is_plain_text(resume_file_type, resume_file_name):
        suffix = ".txt"
    else:
        raise ValueError(
            f"Unsupported resume type (mime={resume_file_type!r}, name={resume_file_name!r})"
        )

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / f"resume_{uuid.uuid4()}{suffix}"
        path.write_bytes(resume_bytes)
        if suffix == ".pdf":
            loader = PyMuPDFLoader(str(path))
        else:
            loader = TextLoader(str(path), encoding="utf-8")
        docs = await loader.aload()
        text = _documents_to_text(docs).strip()
        logger.info("Loaded resume text chars=%d", len(text))
        return text
