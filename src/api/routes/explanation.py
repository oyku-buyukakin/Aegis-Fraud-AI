from __future__ import annotations

from dependency_injector.wiring import Provide, inject
from fastapi import APIRouter, Depends

from ..containers import Container
from ..schemas import ExplainRequest, ExplainResponse
from ..services.explanation_service import ExplanationService

router = APIRouter(tags=["Explainability"])


@router.post("/explain", response_model=ExplainResponse)
@inject
def explain(
    req: ExplainRequest,
    service: ExplanationService = Depends(Provide[Container.explanation_service]),
) -> ExplainResponse:
    return service.explain(req)
