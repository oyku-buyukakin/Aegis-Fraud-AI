from __future__ import annotations

from dependency_injector.wiring import Provide, inject
from fastapi import APIRouter, Depends

from ..containers import Container
from ..schemas import RuleCatalogResponse, RuleEvalRequest, RuleEvalResponse
from ..services.rule_service import RuleService

router = APIRouter(tags=["Rules"])


@router.post("/rules/evaluate", response_model=RuleEvalResponse)
@inject
def rules_evaluate(
    req: RuleEvalRequest,
    service: RuleService = Depends(Provide[Container.rule_service]),
) -> RuleEvalResponse:
    return service.evaluate(req)


@router.get("/rules/list", response_model=RuleCatalogResponse)
@inject
def rules_list(
    service: RuleService = Depends(Provide[Container.rule_service]),
) -> RuleCatalogResponse:
    return service.catalog()