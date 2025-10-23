from __future__ import annotations

from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from src.core.config import get_settings
from src.services.rag_pipeline import DeepThinkingRAGService, IngestionResult

router = APIRouter()

settings = get_settings()
rag_service = DeepThinkingRAGService(settings)


class IngestRequest(BaseModel):
    file_path: Optional[str] = Field(
        default=None,
        description="Absolute path to a document to ingest.",
    )
    content: Optional[str] = Field(
        default=None,
        description="Raw text content to ingest directly.",
    )
    metadata: Optional[Dict[str, str]] = Field(
        default=None,
        description="Additional metadata to attach to each chunk.",
    )

    def model_post_init(self, __context: Any) -> None:
        if not self.file_path and not self.content:
            raise ValueError("Either 'file_path' or 'content' must be provided.")


class IngestResponse(BaseModel):
    chunks_ingested: int
    sections_found: int

    @classmethod
    def from_result(cls, result: IngestionResult) -> "IngestResponse":
        return cls(chunks_ingested=result.chunks_ingested, sections_found=result.sections_found)


class QueryRequest(BaseModel):
    query: str = Field(..., description="Natural language question to answer.")


class QueryResponse(BaseModel):
    answer: str


@router.post("/ingest", response_model=IngestResponse)
def ingest_documents(request: IngestRequest) -> IngestResponse:
    try:
        result = rag_service.ingest_document(
            file_path=request.file_path,
            content=request.content,
            metadata=request.metadata,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return IngestResponse.from_result(result)


@router.post("/query", response_model=QueryResponse)
def query_rag(request: QueryRequest) -> QueryResponse:
    try:
        answer = rag_service.query(request.query)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return QueryResponse(answer=answer)


__all__ = ["router"]
