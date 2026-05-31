from __future__ import annotations

from dependency_injector.wiring import Provide, inject
from fastapi import APIRouter, Depends

from ..containers import Container
from ..schemas import RAGQueryRequest, RAGQueryResponse
from ..services.rag_service import RagService

router = APIRouter(tags=["RAG"])


@router.post("/rag/query", response_model=RAGQueryResponse)
@inject
def rag_query(
    req: RAGQueryRequest,
    service: RagService = Depends(Provide[Container.rag_service]),
) -> RAGQueryResponse:
    return service.search(req)