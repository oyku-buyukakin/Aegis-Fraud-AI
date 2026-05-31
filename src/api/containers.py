from __future__ import annotations

from dependency_injector import containers, providers

from .services.explanation_service import ExplanationService
from .services.rag_service import RagService
from .services.rule_service import RuleService
from .services.scoring_service import ScoringService, build_anomaly_model


class Container(containers.DeclarativeContainer):
    config = providers.Configuration()

    anomaly_model = providers.Singleton(
        build_anomaly_model,
        model_path=config.model_path,
    )

    scoring_service = providers.Singleton(
        ScoringService,
        model=anomaly_model,
    )

    explanation_service = providers.Singleton(
        ExplanationService,
        scoring=scoring_service,
    )

    rule_service = providers.Singleton(
        RuleService,
        rules_path=config.rules_path,
    )

    rag_service = providers.Singleton(
        RagService,
        kb_dir=config.kb_dir,
        min_score=config.min_rag_score,
    )