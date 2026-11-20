
import logging
import os
import uuid
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Response, UploadFile, status

from app.core.config import settings
from app.generation.generator import generate_answer
from app.ingestion.document_registry import DocumentRegistry
from app.ingestion.indexer import index_chunks
from app.ingestion.pipeline import ingest_document
from app.retrieval.retriever import retrieve_documents
from app.schemas.answer import AnswerResponse
from app.schemas.query import QueryRequest
from app.schemas.upload import UploadResponse

logger = logging.getLogger(__name__)

document_registry = DocumentRegistry()

app = FastAPI(
    title=settings.app_name,
    description="A research assistant powered by Retrieval-Augmented Generation.",
    version="0.1.0",
)


@app.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "healthy"}


@app.post("/api/v1/query", response_model=AnswerResponse)
def query_documents(request: QueryRequest) -> AnswerResponse:
    try:
        documents = retrieve_documents(
            query=request.question,
            top_k=request.top_k,
            filters=request.filters,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception:
        logger.exception("Unexpected error during document retrieval.")
        raise HTTPException(
            status_code=500,
            detail="An internal error occurred during document retrieval.",
        ) from None

    try:
        return generate_answer(
            question=request.question,
            documents=documents,
        )
    except Exception:
        logger.exception("Unexpected error during answer generation.")
        raise HTTPException(
            status_code=500,
            detail="An internal error occurred during answer generation.",
        ) from None


@app.post(
    "/api/v1/documents/upload",
    response_model=UploadResponse,
    status_code=status.HTTP_201_CREATED,
)
def upload_document(
    response: Response,
    file: UploadFile = File(...),
) -> UploadResponse:
    """Upload, ingest, and index a PDF or TXT document."""

    temporary_path = None
    stored_path = None

    try:
        original_filename = (file.filename or "").replace("\\", "/")
        original_filename = Path(original_filename).name

        if not original_filename:
            raise HTTPException(
                status_code=400,
                detail="A valid filename is required.",
            )

        extension = Path(original_filename).suffix.lower()

        if extension not in {".pdf", ".txt"}:
            raise HTTPException(
                status_code=400,
                detail="Only PDF and TXT files are supported.",
            )

        documents_dir = settings.documents_dir
        documents_dir.mkdir(parents=True, exist_ok=True)

        # Generate an unpredictable temporary filename.
        temporary_path = documents_dir / f".{uuid.uuid4().hex}{extension}"

        # Stream the upload to disk while enforcing the size limit.
        total_size = 0

        with temporary_path.open("wb") as destination:
            while True:
                chunk = file.file.read(64 * 1024)

                if not chunk:
                    break

                total_size += len(chunk)

                if total_size > settings.max_upload_size_bytes:
                    raise HTTPException(
                        status_code=413,
                        detail="Uploaded file exceeds the maximum allowed size.",
                    )

                destination.write(chunk)

        if total_size == 0:
            raise HTTPException(
                status_code=400,
                detail="Uploaded file is empty.",
            )

        from app.ingestion.document_identity import compute_file_hash

        document_id = compute_file_hash(temporary_path)

        if document_registry.is_registered(document_id):
            response.status_code = status.HTTP_200_OK
            return UploadResponse(
                document_id=document_id,
                filename=original_filename,
                chunk_count=0,
                skipped=True,
                message="Document has already been uploaded.",
            )

        stored_path = documents_dir / f"{document_id}{extension}"
        os.replace(temporary_path, stored_path)

        try:
            result = ingest_document(
                str(stored_path),
                document_registry,
                register=False,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception:
            logger.exception("Unexpected error during document ingestion.")
            raise HTTPException(
                status_code=500,
                detail="An internal error occurred during document ingestion.",
            ) from None

        if result.skipped:
            response.status_code = status.HTTP_200_OK
            return UploadResponse(
                document_id=result.document_id,
                filename=original_filename,
                chunk_count=0,
                skipped=True,
                message="Document has already been uploaded.",
            )

        if not result.documents:
            raise HTTPException(
                status_code=422,
                detail="No extractable text was found in the uploaded document.",
            )

        for document in result.documents:
            document.metadata["filename"] = original_filename

        try:
            indexed_ids = index_chunks(result.documents)
        except Exception:
            logger.exception("Unexpected error during document indexing.")
            raise HTTPException(
                status_code=500,
                detail="An internal error occurred during document indexing.",
            ) from None

        # Register only after successful indexing.
        document_registry.register(result.document_id)

        return UploadResponse(
            document_id=result.document_id,
            filename=original_filename,
            chunk_count=len(indexed_ids),
            skipped=False,
            message="Document uploaded and indexed successfully.",
        )

    except HTTPException:
        if stored_path is not None and stored_path.exists():
            stored_path.unlink()
        raise

    except Exception:
        logger.exception("Unexpected error during document upload.")

        if stored_path is not None and stored_path.exists():
            stored_path.unlink()

        raise HTTPException(
            status_code=500,
            detail="An internal error occurred during document upload.",
        ) from None

    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()

        file.file.close()
