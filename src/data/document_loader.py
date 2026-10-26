"""Offline PDF/TXT loading and passage chunking for IntelliDesk."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class DocumentChunk:
    """A searchable text passage with traceable source information."""

    text: str
    source: str
    chunk_index: int
    page_number: int | None = None
    line_number: int | None = None


def chunk_text(
    text: str,
    chunk_size: int = 700,
    overlap: int = 100,
) -> list[str]:
    """Split text into overlapping character windows, preferring word boundaries."""
    if chunk_size < 1:
        raise ValueError("chunk_size must be at least 1")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be non-negative and smaller than chunk_size")

    normalized = " ".join(text.split())
    if not normalized:
        return []
    if len(normalized) <= chunk_size:
        return [normalized]

    chunks: list[str] = []
    start = 0
    text_length = len(normalized)

    while start < text_length:
        end = min(start + chunk_size, text_length)
        if end < text_length:
            # Prefer a word boundary if it does not make the chunk too short.
            boundary = normalized.rfind(" ", start + chunk_size // 2, end)
            if boundary > start:
                end = boundary

        piece = normalized[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= text_length:
            break

        next_start = max(0, end - overlap)
        # Ensure progress even when unusual whitespace/boundary positions occur.
        start = next_start if next_start > start else end

    return chunks


def load_document(
    path: str | Path,
    chunk_size: int = 700,
    overlap: int = 100,
) -> list[DocumentChunk]:
    """Load a UTF-8 TXT or text-based PDF and return chunks with source metadata.

    Scanned/image-only PDFs are not OCR'd. If a PDF page has no extractable text,
    it is skipped; a document with no extractable text raises ValueError.
    """
    file_path = Path(path).expanduser().resolve()
    if not file_path.is_file():
        raise FileNotFoundError(f"Document not found: {file_path}")

    suffix = file_path.suffix.lower()
    chunks: list[DocumentChunk] = []
    source = str(file_path)

    if suffix == ".txt":
        text = file_path.read_text(encoding="utf-8-sig", errors="replace")
        chunk_index = 0
        # Keep line boundaries as passage boundaries so unrelated topics do not
        # get merged into one retrieval chunk. Long lines are split with overlap.
        for line_number, line in enumerate(text.splitlines(), start=1):
            for part in chunk_text(line, chunk_size, overlap):
                chunks.append(
                    DocumentChunk(
                        text=part,
                        source=source,
                        chunk_index=chunk_index,
                        line_number=line_number,
                    )
                )
                chunk_index += 1
    elif suffix == ".pdf":
        try:
            from pypdf import PdfReader
        except ImportError as exc:
            raise RuntimeError(
                "PDF support requires pypdf. Install project dependencies with: "
                "python -m pip install -r requirements.txt"
            ) from exc

        reader = PdfReader(str(file_path))
        for page_number, page in enumerate(reader.pages, start=1):
            page_text = page.extract_text() or ""
            for local_index, part in enumerate(chunk_text(page_text, chunk_size, overlap)):
                chunks.append(
                    DocumentChunk(
                        text=part,
                        source=source,
                        page_number=page_number,
                        chunk_index=local_index,
                    )
                )
    else:
        raise ValueError(
            f"Unsupported document type '{suffix}'. IntelliDesk currently supports .txt and .pdf."
        )

    if not chunks:
        raise ValueError(
            f"No extractable text found in {file_path.name}. If this is a scanned PDF, OCR is not yet supported."
        )
    return chunks


def load_documents(
    paths: Iterable[str | Path],
    chunk_size: int = 700,
    overlap: int = 100,
) -> list[DocumentChunk]:
    """Load multiple files or all supported files in supplied directories."""
    resolved_files: list[Path] = []
    for raw_path in paths:
        path = Path(raw_path).expanduser()
        if path.is_dir():
            resolved_files.extend(
                item for item in sorted(path.rglob("*"))
                if item.is_file() and item.suffix.lower() in {".txt", ".pdf"}
            )
        else:
            resolved_files.append(path)

    if not resolved_files:
        raise ValueError("No PDF or TXT documents were found.")

    all_chunks: list[DocumentChunk] = []
    failures: list[str] = []
    for file_path in resolved_files:
        try:
            all_chunks.extend(load_document(file_path, chunk_size, overlap))
        except (FileNotFoundError, ValueError, RuntimeError) as exc:
            failures.append(f"{file_path}: {exc}")

    if not all_chunks:
        detail = "\n".join(failures)
        raise ValueError(f"No documents could be indexed. {detail}")
    return all_chunks
