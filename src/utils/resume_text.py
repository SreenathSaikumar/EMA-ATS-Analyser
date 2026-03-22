import logging
import tempfile
import uuid

from pathlib import Path

from langchain_community.document_loaders import PyMuPDFLoader, TextLoader
from langchain_core.documents import Document

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
