
import re
import uuid
import threading
from pathlib import Path
from typing import Any

from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel, Field

from src.inference.rag_pipeline import RAGPipeline
from src.inference.retriever import DocumentRetriever


# --------------------------------------------------
# PATH CONFIGURATION
# --------------------------------------------------

ROOT_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = ROOT_DIR / "data"
UPLOAD_DIR = DATA_DIR / "uploads"

MODEL_PATH = (
    ROOT_DIR
    / "checkpoints"
    / "context256_v2"
    / "model.pt"
)

TOKENIZER_PATH = (
    ROOT_DIR
    / "checkpoints"
    / "context256_v2"
    / "tokenizer.json"
)

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_EXTENSIONS = {".pdf", ".txt"}
MAX_UPLOAD_SIZE = 20 * 1024 * 1024  # 20 MB


# --------------------------------------------------
# FASTAPI APPLICATION
# --------------------------------------------------

app = FastAPI(
    title="IntelliDesk API",
    description="Offline AI Assistant for Document Understanding",
    version="1.0.0",
)


# --------------------------------------------------
# GLOBAL PIPELINE
# --------------------------------------------------

pipeline = None
pipeline_lock = threading.RLock()


def rebuild_retriever() -> DocumentRetriever:
    """Index all supported documents inside the data directory."""

    supported_files = [
        path
        for path in DATA_DIR.rglob("*")
        if path.is_file()
        and path.suffix.lower() in ALLOWED_EXTENSIONS
    ]

    if not supported_files:
        return DocumentRetriever()

    return DocumentRetriever.from_paths(
        [DATA_DIR]
    )


def get_pipeline() -> RAGPipeline:
    """Load the model and initialize the RAG pipeline once."""

    global pipeline

    with pipeline_lock:
        if pipeline is None:
            if not MODEL_PATH.is_file():
                raise HTTPException(
                    status_code=503,
                    detail="Model checkpoint was not found.",
                )

            if not TOKENIZER_PATH.is_file():
                raise HTTPException(
                    status_code=503,
                    detail="Tokenizer checkpoint was not found.",
                )

            try:
                pipeline = RAGPipeline.from_checkpoints(
                    model_path=MODEL_PATH,
                    tokenizer_path=TOKENIZER_PATH,
                    data_path=None,
                    device="cpu",
                )

                pipeline.retriever = rebuild_retriever()

            except Exception as error:
                pipeline = None

                raise HTTPException(
                    status_code=503,
                    detail=f"Could not initialize RAG pipeline: {error}",
                )

        return pipeline


# --------------------------------------------------
# REQUEST MODEL
# --------------------------------------------------

class QuestionRequest(BaseModel):
    query: str = Field(
        min_length=1,
        max_length=2000,
    )

    top_k: int = Field(
        default=2,
        ge=1,
        le=10,
    )


# --------------------------------------------------
# HEALTH CHECK
# --------------------------------------------------

@app.get("/api/health")
def health_check():
    """Check whether the backend is running."""

    return {
        "status": "running",
        "project": "IntelliDesk",
        "model_loaded": pipeline is not None,
    }


# --------------------------------------------------
# DOCUMENT LIST
# --------------------------------------------------

@app.get("/api/documents")
def list_documents():
    """Return the documents currently indexed."""

    current_pipeline = get_pipeline()

    documents = []

    for index, text in enumerate(
        current_pipeline.retriever.documents
    ):
        metadata = (
            current_pipeline.retriever.source_metadata[index]
        )

        documents.append({
            "filename": Path(
                str(metadata.get("source", "Unknown"))
            ).name,
            "source": metadata.get("source"),
            "page_number": metadata.get("page_number"),
            "line_number": metadata.get("line_number"),
            "chunk_index": metadata.get("chunk_index"),
            "text_preview": text[:200],
        })

    return {
        "total_chunks": len(documents),
        "documents": documents,
    }


# --------------------------------------------------
# DOCUMENT UPLOAD
# --------------------------------------------------

@app.post("/api/upload")
async def upload_document(
    file: UploadFile = File(...),
):
    """Upload and index a PDF or TXT document."""

    current_pipeline = get_pipeline()

    original_name = Path(
        file.filename or ""
    ).name

    extension = Path(original_name).suffix.lower()

    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail="Only PDF and TXT files are supported.",
        )

    safe_name = re.sub(
        r"[^a-zA-Z0-9._-]+",
        "_",
        original_name,
    ).strip("._")

    if not safe_name:
        raise HTTPException(
            status_code=400,
            detail="Invalid filename.",
        )

    file_content = await file.read(
        MAX_UPLOAD_SIZE + 1
    )

    await file.close()

    if not file_content:
        raise HTTPException(
            status_code=400,
            detail="The uploaded file is empty.",
        )

    if len(file_content) > MAX_UPLOAD_SIZE:
        raise HTTPException(
            status_code=413,
            detail="File size must not exceed 20 MB.",
        )

    stored_name = (
        f"{uuid.uuid4().hex}_{safe_name}"
    )

    destination = UPLOAD_DIR / stored_name

    with pipeline_lock:
        try:
            destination.write_bytes(file_content)

            new_retriever = rebuild_retriever()

            current_pipeline.retriever = new_retriever

        except Exception as error:
            if destination.exists():
                destination.unlink()

            raise HTTPException(
                status_code=400,
                detail=f"Could not process document: {error}",
            )

    return {
        "message": "Document uploaded and indexed successfully.",
        "filename": original_name,
        "total_chunks": len(
            current_pipeline.retriever.documents
        ),
    }


# --------------------------------------------------
# ASK QUESTION
# --------------------------------------------------

@app.post("/api/ask")
def ask_question(request: QuestionRequest):
    """Answer a question using the indexed documents."""

    query = request.query.strip()

    if not query:
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty.",
        )

    current_pipeline = get_pipeline()

    try:
        with pipeline_lock:
            result: dict[str, Any] = (
                current_pipeline.answer_query(
                    query=query,
                    top_k=request.top_k,
                )
            )

        return {
            "query": result["query"],
            "answer": result["answer"],
            "answer_source": result["answer_source"],
            "retrieved_sources": result["retrieved_sources"],
            "message": "Question processed successfully.",
        }

    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error),
        )

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Could not process question: {error}",
        )
