from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse

from .containers import Container
from .dashboard import DASHBOARD_HTML
from .routes import explanation, knowledge_query, risk_score, rule_evaluation
from .schemas import HealthResponse

_ROOT = Path(__file__).resolve().parents[2]
_CONFIG_DIR = _ROOT / "configs"
_KB_DIR = _ROOT / "docs" / "knowledge_base"
_MODEL_PATH = _ROOT / "models" / "anomaly_model.joblib"


def create_container() -> Container:
    container = Container()
    container.config.rules_path.from_value(str(_CONFIG_DIR / "fraud_rules.yaml"))
    container.config.kb_dir.from_value(str(_KB_DIR))
    container.config.model_path.from_value(str(_MODEL_PATH))
    container.config.min_rag_score.from_value(0.8)
    container.wire(modules=[risk_score, explanation, rule_evaluation, knowledge_query])
    return container


def create_app() -> FastAPI:
    container = create_container()

    app = FastAPI(
        title="Aegis Fraud Detection API",
        description="Anomaly scoring, explainability, rule evaluation, and RAG knowledge query.",
        version="1.0.0",
    )
    app.container = container  

    app.include_router(risk_score.router)
    app.include_router(explanation.router)
    app.include_router(rule_evaluation.router)
    app.include_router(knowledge_query.router)

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    def root() -> HTMLResponse:
        return HTMLResponse(content=DASHBOARD_HTML)

    @app.get("/health", tags=["Meta"], response_model=HealthResponse)
    def health() -> HealthResponse:
        return HealthResponse(
            status="ok",
            rules_loaded=container.rule_service().rule_count(),
            kb_chunks=container.rag_service().chunk_count(),
            model=container.scoring_service().model_name(),
        )

    return app


app = create_app()