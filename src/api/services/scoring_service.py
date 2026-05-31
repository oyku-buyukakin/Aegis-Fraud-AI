from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from ..schemas import ScoreRequest, ScoreResponse

_SIGNAL_WEIGHTS: dict[str, float] = {
    "is_night_transaction":  0.10,
    "is_new_account":        0.15,
    "is_high_amount":        0.15,
    "has_velocity_spike":    0.20,
    "uses_disposable_email": 0.15,
    "uses_credit_card":      0.05,
    "country_mismatch":      0.10,
    "low_device_trust":      0.10,
}

_DISPOSABLE = {"protonmail.com", "mailinator.com", "guerrillamail.com", "tempmail.com", "throwam.com"}


def extract_signals(req: ScoreRequest) -> dict[str, int]:
    """Derive binary fraud signals from a raw transaction request."""
    return {
        "is_night_transaction":  int(req.hour < 6 or req.hour >= 22),
        "is_new_account":        int(req.account_age_days <= 7),
        "is_high_amount":        int(req.TransactionAmt >= 500),
        "has_velocity_spike":    int(req.num_txn_last_1h >= 5),
        "uses_disposable_email": int(req.P_emaildomain.lower() in _DISPOSABLE),
        "uses_credit_card":      int(req.card_type.lower() == "credit"),
        "country_mismatch":      int(req.country_mismatch),
        "low_device_trust":      int(req.device_trust_score < 0.45),
    }


def risk_level(score: float) -> str:
    if score >= 0.65:
        return "HIGH"
    if score >= 0.35:
        return "MEDIUM"
    return "LOW"


class AnomalyModel(Protocol):
    name: str

    def score(self, signals: dict[str, int]) -> float:
        ...


class SignalWeightModel:

    name = "signal-weight-heuristic"

    def score(self, signals: dict[str, int]) -> float:
        raw = sum(signals.get(k, 0) * w for k, w in _SIGNAL_WEIGHTS.items())
        return round(min(raw, 1.0), 4)


class PipelineAnomalyModel:

    name = "pipeline-anomaly-model"
    FEATURE_ORDER: tuple[str, ...] = tuple(_SIGNAL_WEIGHTS.keys())

    def __init__(self, model_path: Path) -> None:
        import joblib 

        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Anomaly model artifact not found: {model_path}")
        loaded = joblib.load(model_path)
        if isinstance(loaded, dict) and "model" in loaded:
            self._model = loaded["model"]
            self.name = loaded.get("name", self.name)
            order = loaded.get("feature_order")
            if order:
                self._feature_order = tuple(order)
            else:
                self._feature_order = self.FEATURE_ORDER
        else:
            self._model = loaded
            self._feature_order = self.FEATURE_ORDER

    def score(self, signals: dict[str, int]) -> float:
        import pandas as pd

        row = {k: signals.get(k, 0) for k in self._feature_order}
        features = pd.DataFrame([row], columns=list(self._feature_order))
        if hasattr(self._model, "predict_proba"):
            return round(float(self._model.predict_proba(features)[0][1]), 4)
        if hasattr(self._model, "decision_function"):
            df = float(self._model.decision_function(features)[0])
            return round(1.0 / (1.0 + math.exp(df)), 4)
        return round(min(max(float(self._model.predict(features)[0]), 0.0), 1.0), 4)


def build_anomaly_model(model_path: str | None) -> AnomalyModel:
    if model_path and Path(model_path).exists():
        try:
            return PipelineAnomalyModel(Path(model_path))
        except Exception: 
            pass
    return SignalWeightModel()


@dataclass
class ScoreResult:
    transaction_id: str
    anomaly_score: float
    risk_level: str
    triggered_signals: list[str]


class ScoringService:

    def __init__(self, model: AnomalyModel) -> None:
        self._model = model

    @property
    def model(self) -> AnomalyModel:
        return self._model

    def model_name(self) -> str:
        return getattr(self._model, "name", "unknown")

    def evaluate(self, req: ScoreRequest) -> ScoreResult:
        signals = extract_signals(req)
        score = self._model.score(signals)
        return ScoreResult(
            transaction_id=req.transaction_id,
            anomaly_score=score,
            risk_level=risk_level(score),
            triggered_signals=[k for k, v in signals.items() if v == 1],
        )

    def to_response(self, req: ScoreRequest) -> ScoreResponse:
        result = self.evaluate(req)
        return ScoreResponse(**result.__dict__)