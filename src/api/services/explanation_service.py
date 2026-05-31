from __future__ import annotations

from ..schemas import ExplainRequest, ExplainResponse
from .scoring_service import ScoringService, extract_signals, risk_level


class ExplanationService:

    def __init__(self, scoring: ScoringService) -> None:
        self._scoring = scoring

    def explain(self, req: ExplainRequest) -> ExplainResponse:
        signals = extract_signals(req)
        score = req.anomaly_score if req.anomaly_score is not None else self._scoring.model.score(signals)
        level = risk_level(score)
        triggered = [k for k, v in signals.items() if v == 1]
        signal_text = (
            ", ".join(k.replace("_", " ") for k in triggered)
            if triggered
            else "no major risk signals"
        )
        explanation = (
            f"This transaction has a risk score of {score:.3f}, which is considered {level} risk. "
            f"It was flagged because of the following: {signal_text}. "
            f"The transaction was for {req.TransactionAmt:.2f} at {req.hour:02d}:00, "
            f"from an account that is {req.account_age_days} day(s) old."
        )
        return ExplainResponse(
            transaction_id=req.transaction_id,
            anomaly_score=round(score, 4),
            risk_level=level,
            explanation=explanation,
        )