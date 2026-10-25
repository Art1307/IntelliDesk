from pathlib import Path

import pytest

from src.data.document_loader import chunk_text, load_document
from src.inference.retriever import DocumentRetriever
from src.training.train import load_training_text


def test_chunk_text_returns_overlapping_chunks():
    text = " ".join(f"word{i}" for i in range(30))
    chunks = chunk_text(text, chunk_size=45, overlap=10)

    assert len(chunks) > 1
    assert all(chunks)
    assert all(len(chunk) <= 45 for chunk in chunks)
    assert chunks[0].split()[-1] in chunks[1].split()


def test_load_txt_creates_chunks_and_source_metadata(tmp_path: Path):
    file_path = tmp_path / "notes.txt"
    file_path.write_text("Alpha topic. Beta topic. Gamma topic. " * 20, encoding="utf-8")

    chunks = load_document(file_path, chunk_size=70, overlap=10)

    assert len(chunks) > 1
    assert all(chunk.source == str(file_path.resolve()) for chunk in chunks)
    assert [chunk.chunk_index for chunk in chunks] == list(range(len(chunks)))
    assert all(chunk.page_number is None for chunk in chunks)


def test_retriever_indexes_txt_and_returns_metadata(tmp_path: Path):
    file_path = tmp_path / "notes.txt"
    file_path.write_text(
        "Causal masking prevents access to future tokens.\n"
        "Offline document assistants protect data privacy.\n",
        encoding="utf-8",
    )

    retriever = DocumentRetriever.from_paths([file_path], chunk_size=200, overlap=20)
    results = retriever.retrieve_with_metadata("causal masking future tokens", top_k=1)

    assert results
    assert "Causal masking" in results[0]["text"]
    assert results[0]["score"] > 0
    assert results[0]["metadata"]["source"] == str(file_path.resolve())
    assert results[0]["metadata"]["chunk_index"] == 0
    assert results[0]["metadata"]["line_number"] == 1


def test_load_document_rejects_unsupported_extension(tmp_path: Path):
    file_path = tmp_path / "notes.docx"
    file_path.write_text("not a supported format", encoding="utf-8")

    with pytest.raises(ValueError, match="supports .txt and .pdf"):
        load_document(file_path)


def test_training_loader_reads_txt_input(tmp_path: Path):
    file_path = tmp_path / "training.txt"
    file_path.write_text("Text from a local document.\nSecond line.", encoding="utf-8")

    assert "Text from a local document." in load_training_text(file_path)


def test_training_loader_reads_directory_of_txt_files(tmp_path: Path):
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    (docs_dir / "one.txt").write_text("First local document content.", encoding="utf-8")
    (docs_dir / "two.txt").write_text("Second local document content.", encoding="utf-8")

    text = load_training_text(docs_dir)
    assert "First local document content." in text
    assert "Second local document content." in text
