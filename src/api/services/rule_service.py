from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import yaml

from ..schemas import FiredRule, RuleCatalogResponse, RuleEvalRequest, RuleEvalResponse, RuleInfo

_OPS: dict[str, Callable[[float, float], bool]] = {
    "gte": lambda a, b: a >= b,
    "gt":  lambda a, b: a > b,
    "lte": lambda a, b: a <= b,
    "lt":  lambda a, b: a < b,
    "eq":  lambda a, b: a == b,
    "neq": lambda a, b: a != b,
}

_SEVERITY_RANK = {"CRITICAL": 4, "HIGH": 3, "MEDIUM": 2, "LOW": 1}


class RuleService:
    """Loads ``fraud_rules.yaml`` once and evaluates transactions against it."""

    def __init__(self, rules_path: str) -> None:
        self._rules_path = Path(rules_path)
        self._rules: list[dict] | None = None

    def _rules_cache(self) -> list[dict]:
        if self._rules is None:
            if self._rules_path.exists():
                self._rules = yaml.safe_load(self._rules_path.read_text())["rules"]
            else:
                self._rules = []
        return self._rules

    def rule_count(self) -> int:
        return len(self._rules_cache())

    def evaluate(self, req: RuleEvalRequest) -> RuleEvalResponse:
        features = req.features
        fired: list[dict] = []

        for rule in self._rules_cache():
            conditions = rule.get("conditions", [])
            logic = rule.get("logic", "AND")
            results = []
            for cond in conditions:
                val = features.get(cond["field"])
                if val is None:
                    results.append(False)
                    continue
                op_fn = _OPS.get(cond["operator"], lambda a, b: False)
                results.append(op_fn(float(val), float(cond["value"])))

            matched = all(results) if logic == "AND" else any(results)
            if not matched:
                continue

            tmpl = rule.get("explanation_template", "Rule fired.")
            try:
                explanation = tmpl.format(**{k: v for k, v in features.items() if isinstance(v, (int, float))})
            except (KeyError, ValueError):
                explanation = tmpl

            fired.append({
                "rule_id":     rule["id"],
                "name":        rule["name"],
                "severity":    rule["severity"],
                "action":      rule["action"],
                "priority":    rule["priority"],
                "explanation": explanation.strip(),
            })

        fired.sort(key=lambda r: r["priority"])
        final_action = fired[0]["action"] if fired else "APPROVE"
        final_severity = max(
            (r["severity"] for r in fired),
            key=lambda s: _SEVERITY_RANK.get(s, 0),
            default="LOW",
        )

        return RuleEvalResponse(
            transaction_id=req.transaction_id,
            fired_rules=[FiredRule(**{k: v for k, v in r.items() if k != "priority"}) for r in fired],
            final_action=final_action,
            final_severity=final_severity,
        )

    def catalog(self) -> RuleCatalogResponse:
        return RuleCatalogResponse(
            rules=[
                RuleInfo(
                    rule_id=r["id"],
                    name=r["name"],
                    severity=r["severity"],
                    action=r["action"],
                    description=" ".join(str(r.get("description", "")).split()),
                )
                for r in self._rules_cache()
            ]
        )