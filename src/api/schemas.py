from __future__ import annotations

from pydantic import BaseModel, Field

class ScoreRequest(BaseModel):
    transaction_id: str
    TransactionAmt: float
    hour: int = Field(ge=0, le=23)
    account_age_days: int = Field(ge=0)
    num_txn_last_1h: int = Field(ge=0)
    P_emaildomain: str = "unknown"
    card_type: str = "debit"
    country_mismatch: int = Field(ge=0, le=1)
    device_trust_score: float = Field(default=0.5, ge=0.0, le=1.0)


class ScoreResponse(BaseModel):
    transaction_id: str
    anomaly_score: float
    risk_level: str
    triggered_signals: list[str]

class ExplainRequest(ScoreRequest):
    anomaly_score: float | None = None


class ExplainResponse(BaseModel):
    transaction_id: str
    anomaly_score: float
    risk_level: str
    explanation: str

class RuleEvalRequest(BaseModel):
    transaction_id: str
    features: dict[str, float | int | str]


class FiredRule(BaseModel):
    rule_id: str
    name: str
    severity: str
    action: str
    explanation: str


class RuleEvalResponse(BaseModel):
    transaction_id: str
    fired_rules: list[FiredRule]
    final_action: str
    final_severity: str


class RuleInfo(BaseModel):
    rule_id: str
    name: str
    severity: str
    action: str
    description: str


class RuleCatalogResponse(BaseModel):
    rules: list[RuleInfo]

class RAGQueryRequest(BaseModel):
    query: str
    top_k: int = Field(default=3, ge=1, le=10)


class RAGChunk(BaseModel):
    source: str
    score: float
    text: str


class RAGQueryResponse(BaseModel):
    query: str
    results: list[RAGChunk]


class HealthResponse(BaseModel):
    status: str
    rules_loaded: int
    kb_chunks: int
    model: str