from __future__ import annotations

from dependency_injector.wiring import Provide, inject
from fastapi import APIRouter, Depends

from ..containers import Container
from ..schemas import ScoreRequest, ScoreResponse
from ..services.scoring_service import ScoringService

router = APIRouter(tags=["Scoring"])


@router.post("/score", response_model=ScoreResponse)
@inject
def score(
    req: ScoreRequest,
    service: ScoringService = Depends(Provide[Container.scoring_service]),
) -> ScoreResponse:
    return service.to_response(req)
