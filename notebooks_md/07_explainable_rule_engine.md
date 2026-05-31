# Rule Engine



```python
import pandas as pd
import numpy as np
import json, yaml
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from dataclasses import dataclass
from typing import Any, List
from IPython.display import display
pd.options.future.infer_string = False
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    fbeta_score)
import warnings
warnings.filterwarnings("ignore")
```

### Data


```python
train = pd.read_pickle('../data/interim/train_context_adjusted.pkl')
test  = pd.read_pickle('../data/interim/test_context_adjusted.pkl')

TARGET    = 'isFraud'
y = train[TARGET].astype(int)

ID_COL    = 'TransactionID'
SCORE_COL = 'adjusted_anomaly_score'

```


```python
baseline_auc = roc_auc_score(y, train[SCORE_COL])
baseline_pr  = average_precision_score(y, train[SCORE_COL])

print(f'Train: {train.shape}  |  Test: {test.shape}')
print(f'Fraud rate      : {y.mean():.2%}')
print(f'Baseline AUC    : {baseline_auc:.4f}')
print(f'Baseline PR-AUC : {baseline_pr:.4f}')
```

### Rule and Conflict Configuration


```python
fraud_rules = yaml.safe_load(open("../configs/fraud_rules.yaml", encoding="utf-8"))

conflict_resolution = yaml.safe_load(open("../configs/conflict_resolution.yaml", encoding="utf-8"))
```

### Rule Priority


```python
if isinstance(fraud_rules, dict) and "rules" in fraud_rules:
    fraud_rules = fraud_rules["rules"]

priority_table = pd.DataFrame([
    {
        "rule_id": r.get("id", r.get("rule_id")),
        "name": r.get("name"),
        "priority": r.get("priority"),
        "severity": r.get("severity"),
        "score_boost": r.get("score_boost", r.get("boost", 0)),
        "action": r.get("action", "REVIEW"),
        "logic": r.get("logic", "AND")
    }
    for r in fraud_rules
])

priority_table = priority_table.sort_values("priority").reset_index(drop=True)
```

### RuleEngine Class


```python
class RuleEngine:
    def __init__(
        self,
        rules=None,
        config_path=None,
        conflict_config=None,
        base_score_col="adjusted_anomaly_score",
        score_cap=1.0,
        default_action="PASS",
        verbose=False
    ):
        self.rules = rules
        self.config_path = config_path
        self.conflict_config = conflict_config or {}
        self.base_score_col = base_score_col
        self.score_cap = float(score_cap)
        self.default_action = default_action
        self.verbose = verbose

        if isinstance(self.conflict_config, dict) and "conflict_resolution" in self.conflict_config:
            self.conflict_config = self.conflict_config["conflict_resolution"]

        if self.rules is None and self.config_path is not None:
            loaded_config = self._load_config(self.config_path)

            if isinstance(loaded_config, dict):
                self.rules = loaded_config.get("rules", [])
                self.conflict_config = loaded_config.get(
                    "conflict_resolution",
                    self.conflict_config
                )

                if isinstance(self.conflict_config, dict) and "conflict_resolution" in self.conflict_config:
                    self.conflict_config = self.conflict_config["conflict_resolution"]

                self.base_score_col = loaded_config.get("base_score_col", self.base_score_col)
                self.score_cap = float(loaded_config.get("score_cap", self.score_cap))
                self.default_action = loaded_config.get("default_action", self.default_action)

            elif isinstance(loaded_config, list):
                self.rules = loaded_config

            else:
                raise ValueError("Config file must contain a list of rules or a dictionary with a 'rules' key.")

        self.rules = self._normalize_rules(self.rules)

        self.severity_order = {
            "NONE": 0,
            "LOW": 1,
            "MEDIUM": 2,
            "HIGH": 3,
            "CRITICAL": 4
        }

        self.action_order = {
            "PASS": 0,
            "MONITOR": 1,
            "FLAG_REVIEW": 2,
            "REVIEW": 2,
            "MANUAL_REVIEW": 2,
            "FLAG_CRITICAL_REVIEW": 3,
            "BLOCK": 3,
            "DECLINE": 3,
        }

    def _format_explanation(self, rule, row):
        template = (
            rule.get("explanation_template")
            or rule.get("explanation")
            or rule.get("description")
            or rule.get("name")
            or ""
        )
        try:
            return template.format(**row.to_dict())
        except Exception:
            return rule.get("description") or rule.get("name") or template

    def _load_config(self, path):
        with open(path, "r", encoding="utf-8") as file:
            if path.endswith(".json"):
                return json.load(file)

            if path.endswith((".yaml", ".yml")):
                return yaml.safe_load(file)

            raise ValueError("Only JSON, YAML or YML config files are supported.")

    def _normalize_rules(self, rules):
        if rules is None:
            rules = []

        if isinstance(rules, dict) and "rules" in rules:
            rules = rules["rules"]

        if not isinstance(rules, list):
            raise ValueError(
                "Rules must be a list of rule dictionaries or a dictionary containing a 'rules' key."
            )

        normalized_rules = []

        for i, rule in enumerate(rules):
            if not isinstance(rule, dict):
                raise ValueError(f"Rule at index {i} is not a dictionary. Rule value: {rule}")

            rule = dict(rule)

            rule_id = rule.get("id", rule.get("rule_id", f"RULE_{i + 1:03d}"))

            rule.setdefault("id", rule_id)
            rule.setdefault("rule_id", rule_id)
            rule.setdefault("name", rule.get("rule_name", rule_id))
            rule.setdefault("priority", i + 1)
            rule.setdefault("severity", "LOW")
            rule.setdefault("action", "REVIEW")
            rule.setdefault("enabled", True)
            rule.setdefault("logic", "AND")

            if "boost" not in rule:
                rule["boost"] = rule.get("score_boost", 0.0)

            if "score_boost" not in rule:
                rule["score_boost"] = rule.get("boost", 0.0)

            if "conditions" not in rule:
                if "if" in rule:
                    rule["conditions"] = rule["if"]
                elif "condition" in rule:
                    rule["conditions"] = rule["condition"]
                else:
                    rule["conditions"] = []

            if "explanation" not in rule:
                rule["explanation"] = rule.get("description", rule["name"])

            normalized_rules.append(rule)

        normalized_rules = sorted(
            normalized_rules,
            key=lambda x: int(x.get("priority", 999999))
        )

        return normalized_rules

    def _safe_series(self, df, col):
        if col not in df.columns:
            return pd.Series(np.nan, index=df.index)

        return df[col]

    def _to_numeric(self, series):
        return pd.to_numeric(series, errors="coerce")

    def _evaluate_leaf_condition(self, df, condition):
        if not isinstance(condition, dict):
            return pd.Series(False, index=df.index)

        field = (
            condition.get("field")
            or condition.get("column")
            or condition.get("feature")
            or condition.get("var")
        )

        operator = (
            condition.get("operator")
            or condition.get("op")
            or condition.get("condition")
        )

        value = condition.get("value")

        if field is None or operator is None:
            return pd.Series(False, index=df.index)

        series = self._safe_series(df, field)
        operator = str(operator).lower()

        if operator in [">", "gt", "greater_than"]:
            return self._to_numeric(series) > value

        if operator in [">=", "gte", "ge", "greater_equal", "greater_than_or_equal"]:
            return self._to_numeric(series) >= value

        if operator in ["<", "lt", "less_than"]:
            return self._to_numeric(series) < value

        if operator in ["<=", "lte", "le", "less_equal", "less_than_or_equal"]:
            return self._to_numeric(series) <= value

        if operator in ["==", "eq", "equals", "equal"]:
            return series == value

        if operator in ["!=", "ne", "not_equals", "not_equal"]:
            return series != value

        if operator in ["in", "isin"]:
            if not isinstance(value, (list, tuple, set)):
                value = [value]
            return series.isin(value)

        if operator in ["not_in", "notin"]:
            if not isinstance(value, (list, tuple, set)):
                value = [value]
            return ~series.isin(value)

        if operator == "between":
            if not isinstance(value, (list, tuple)) or len(value) != 2:
                return pd.Series(False, index=df.index)

            low, high = value
            numeric_series = self._to_numeric(series)
            return numeric_series.between(low, high, inclusive="both")

        if operator in ["is_null", "isna", "missing"]:
            return series.isna()

        if operator in ["not_null", "notna", "not_missing"]:
            return series.notna()

        if operator == "contains":
            return series.astype(str).str.contains(str(value), case=False, na=False)

        if operator in ["startswith", "starts_with"]:
            return series.astype(str).str.startswith(str(value), na=False)

        if operator in ["endswith", "ends_with"]:
            return series.astype(str).str.endswith(str(value), na=False)

        return pd.Series(False, index=df.index)

    def _evaluate_condition_tree(self, df, condition_tree, default_logic="AND"):
        if condition_tree is None:
            return pd.Series(False, index=df.index)

        if isinstance(condition_tree, list):
            masks = [
                self._evaluate_condition_tree(df, condition, default_logic=default_logic)
                for condition in condition_tree
            ]

            if len(masks) == 0:
                return pd.Series(False, index=df.index)

            if str(default_logic).upper() == "OR":
                final_mask = masks[0].copy()
                for mask in masks[1:]:
                    final_mask = final_mask | mask
            else:
                final_mask = masks[0].copy()
                for mask in masks[1:]:
                    final_mask = final_mask & mask

            return final_mask.fillna(False)

        if not isinstance(condition_tree, dict):
            return pd.Series(False, index=df.index)

        if "all" in condition_tree:
            masks = [
                self._evaluate_condition_tree(df, condition, default_logic="AND")
                for condition in condition_tree["all"]
            ]

            if len(masks) == 0:
                return pd.Series(False, index=df.index)

            final_mask = masks[0].copy()

            for mask in masks[1:]:
                final_mask = final_mask & mask

            return final_mask.fillna(False)

        if "any" in condition_tree:
            masks = [
                self._evaluate_condition_tree(df, condition, default_logic="OR")
                for condition in condition_tree["any"]
            ]

            if len(masks) == 0:
                return pd.Series(False, index=df.index)

            final_mask = masks[0].copy()

            for mask in masks[1:]:
                final_mask = final_mask | mask

            return final_mask.fillna(False)

        if "not" in condition_tree:
            return (~self._evaluate_condition_tree(df, condition_tree["not"])).fillna(False)

        return self._evaluate_leaf_condition(df, condition_tree).fillna(False)

    def _resolve_action(self, fired_rules):
        if len(fired_rules) == 0:
            return self.default_action

        strategy = self.conflict_config.get("action_strategy", "highest_severity")
        strategy = str(strategy).lower()

        if strategy in ["highest_priority", "priority"]:
            selected_rule = sorted(
                fired_rules,
                key=lambda rule: int(rule.get("priority", 999999))
            )[0]
            return selected_rule.get("action", self.default_action)

        if strategy in ["highest_action", "action"]:
            selected_rule = sorted(
                fired_rules,
                key=lambda rule: self.action_order.get(
                    str(rule.get("action", "PASS")).upper(),
                    0
                ),
                reverse=True
            )[0]
            return selected_rule.get("action", self.default_action)

        selected_rule = sorted(
            fired_rules,
            key=lambda rule: (
                self.severity_order.get(
                    str(rule.get("severity", "NONE")).upper(),
                    0
                ),
                -int(rule.get("priority", 999999))
            ),
            reverse=True
        )[0]

        return selected_rule.get("action", self.default_action)

    def _resolve_severity(self, fired_rules):
        if len(fired_rules) == 0:
            return "NONE"

        selected_rule = sorted(
            fired_rules,
            key=lambda rule: self.severity_order.get(
                str(rule.get("severity", "NONE")).upper(),
                0
            ),
            reverse=True
        )[0]

        return str(selected_rule.get("severity", "NONE")).upper()

    def _resolve_boost(self, fired_rules):
        if len(fired_rules) == 0:
            return 0.0

        boost_values = [
            float(rule.get("score_boost", rule.get("boost", 0.0)))
            for rule in fired_rules
        ]

        strategy = self.conflict_config.get("boost_strategy", "sum")
        strategy = str(strategy).lower()

        if strategy in ["max", "maximum", "highest"]:
            return max(boost_values)

        if strategy in ["mean", "average", "avg"]:
            return float(np.mean(boost_values))

        if strategy in ["priority", "highest_priority"]:
            selected_rule = sorted(
                fired_rules,
                key=lambda rule: int(rule.get("priority", 999999))
            )[0]
            return float(selected_rule.get("score_boost", selected_rule.get("boost", 0.0)))

        return float(np.sum(boost_values))

    def evaluate_dataframe(self, df, verbose=False):
        result = df.copy()

        if self.base_score_col in result.columns:
            base_score = pd.to_numeric(
                result[self.base_score_col],
                errors="coerce"
            ).fillna(0)
        else:
            base_score = pd.Series(0.0, index=result.index)

        base_score = base_score.clip(lower=0, upper=self.score_cap)

        fired_rule_ids = [[] for _ in range(len(result))]
        fired_rule_names = [[] for _ in range(len(result))]
        fired_rule_explanations = [[] for _ in range(len(result))]
        fired_rule_objects = [[] for _ in range(len(result))]

        rule_summary = []

        for rule in self.rules:
            if not rule.get("enabled", True):
                continue

            rule_logic = rule.get("logic", "AND")
            rule_conditions = rule.get("conditions", [])

            mask = self._evaluate_condition_tree(
                result,
                rule_conditions,
                default_logic=rule_logic
            )

            mask = mask.fillna(False).astype(bool)

            fired_positions = np.where(mask.values)[0]
            boost = float(rule.get("score_boost", rule.get("boost", 0.0)))

            rule_summary.append({
                "rule_id": rule.get("id", rule.get("rule_id")),
                "rule_name": rule.get("name"),
                "priority": rule.get("priority"),
                "severity": rule.get("severity"),
                "action": rule.get("action"),
                "logic": rule.get("logic", "AND"),
                "score_boost": boost,
                "fired_count": int(mask.sum()),
                "coverage_pct": float(mask.mean() * 100)
            })

            for pos in fired_positions:
                fired_rule_ids[pos].append(rule.get("id", rule.get("rule_id")))
                fired_rule_names[pos].append(rule.get("name"))
                fired_rule_explanations[pos].append(
                    self._format_explanation(rule, result.iloc[pos])
                )
                fired_rule_objects[pos].append(rule)

        total_boost = np.array(
            [self._resolve_boost(fired) for fired in fired_rule_objects],
            dtype=float
        )

        final_scores = (base_score.values + total_boost).clip(0, self.score_cap)

        final_actions = []
        final_severities = []

        for fired in fired_rule_objects:
            final_actions.append(self._resolve_action(fired))
            final_severities.append(self._resolve_severity(fired))

        result["rule_base_score"] = base_score.values
        result["rule_total_boost"] = total_boost
        result["rule_adjusted_score"] = final_scores
        result["rule_fired_count"] = [len(x) for x in fired_rule_ids]
        result["rule_rule_ids"] = fired_rule_ids
        result["rule_rule_names"] = fired_rule_names
        result["rule_explanations"] = fired_rule_explanations
        result["rule_final_action"] = final_actions
        result["rule_max_severity"] = final_severities
        result["rule_has_fired"] = result["rule_fired_count"] > 0

        self.rule_summary_ = (
            pd.DataFrame(rule_summary)
            .sort_values(["priority", "fired_count"], ascending=[True, False])
            .reset_index(drop=True)
        )

        if verbose or self.verbose:
            print("=" * 90)
            print("Rule Engine Evaluation Completed")
            print("=" * 90)
            print(f"Rows evaluated: {len(result):,}")
            print(f"Rules evaluated: {len(self.rules):,}")
            print(f"Rows with at least one fired rule: {result['rule_has_fired'].sum():,}")
            print(f"Rule fired ratio: {result['rule_has_fired'].mean() * 100:.2f}%")
            print(f"Base score column: {self.base_score_col}")
            print(f"Score cap: {self.score_cap}")
            print(f"Boost strategy: {self.conflict_config.get('boost_strategy', 'sum')}")
            print(f"Action strategy: {self.conflict_config.get('action_strategy', 'highest_severity')}")
            print("=" * 90)

        return result

    def get_rule_summary(self, df_with_target=None, target_col="isFraud"):
        if not hasattr(self, "rule_summary_"):
            raise ValueError("No rule summary found. Run evaluate_dataframe() first.")

        summary = self.rule_summary_.copy()

        if df_with_target is not None and target_col in df_with_target.columns:
            fraud_rates = []

            for rule_id in summary["rule_id"]:
                mask = df_with_target["rule_rule_ids"].apply(
                    lambda x: rule_id in x if isinstance(x, list) else False
                )

                if mask.sum() == 0:
                    fraud_rates.append(np.nan)
                else:
                    fraud_rates.append(
                        df_with_target.loc[mask, target_col].mean() * 100
                    )

            summary["fraud_rate_pct"] = fraud_rates

            base_fraud_rate = df_with_target[target_col].mean() * 100

            if base_fraud_rate > 0:
                summary["lift"] = summary["fraud_rate_pct"] / base_fraud_rate
            else:
                summary["lift"] = np.nan

        return summary
```

### If-Then Evaluation Demo


```python
demo_cases = {
    "high_risk_night": {
        "TransactionAmt": 850.0,
        "transaction_hour": 2,
        "is_night_transaction": 1,
        "is_weekend": 1,
        "is_business_hour": 0,
        "is_high_risk_entity": 1,
        "is_outlier": 1,
        "card6": "credit",
        "P_emaildomain": "protonmail.com",
        "ctx_amount_zscore": 5.2,
        "anomaly_high_flag_count": 12,
        "anomaly_high_flag_ratio": 0.45,
        "adjusted_anomaly_score": 0.82,
        "final_context_anomaly_score": 0.78,
        "card4_amount_vs_avg": 620.0,
        "card6_amount_ratio": 4.2,
        "ctx_amount_vs_product_avg": 420.0,
        "ctx_amount_ratio_to_global_median": 7.5,
        "time_since_first_transaction": 3600,
        "entity_trusted_score": 0.18,
        "entity_context_score": 0.82,
        "norm_multivariate_anomaly_score": 0.88,
        "norm_temporal_anomaly_score": 0.81,
        "norm_entity_anomaly_score": 0.84,
        "norm_fraud_corr_weighted_anomaly_score": 0.79,
        "card1_entity_anomaly_score": 0.83,
        "card4_entity_anomaly_score": 0.86,
        "card6_entity_anomaly_score": 0.82,
        "addr1_entity_anomaly_score": 0.77,
        "addr2_entity_anomaly_score": 0.81,
        "P_emaildomain_entity_anomaly_score": 0.85,
    },

    "low_risk_daytime": {
        "TransactionAmt": 45.0,
        "transaction_hour": 11,
        "is_night_transaction": 0,
        "is_weekend": 0,
        "is_business_hour": 1,
        "is_high_risk_entity": 0,
        "is_outlier": 0,
        "card6": "debit",
        "P_emaildomain": "gmail.com",
        "ctx_amount_zscore": 0.3,
        "anomaly_high_flag_count": 1,
        "anomaly_high_flag_ratio": 0.02,
        "adjusted_anomaly_score": 0.12,
        "final_context_anomaly_score": 0.15,
        "card4_amount_vs_avg": 10.0,
        "card6_amount_ratio": 0.8,
        "ctx_amount_vs_product_avg": 5.0,
        "ctx_amount_ratio_to_global_median": 0.9,
        "time_since_first_transaction": 8000000,
        "entity_trusted_score": 0.92,
        "entity_context_score": 0.10,
        "norm_multivariate_anomaly_score": 0.12,
        "norm_temporal_anomaly_score": 0.08,
        "norm_entity_anomaly_score": 0.10,
        "norm_fraud_corr_weighted_anomaly_score": 0.09,
        "card1_entity_anomaly_score": 0.10,
        "card4_entity_anomaly_score": 0.08,
        "card6_entity_anomaly_score": 0.07,
        "addr1_entity_anomaly_score": 0.11,
        "addr2_entity_anomaly_score": 0.10,
        "P_emaildomain_entity_anomaly_score": 0.05,
    },
}
```


```python
demo_df = pd.DataFrame(
    list(demo_cases.values()),
    index=list(demo_cases.keys()))
```


```python
demo_engine = RuleEngine(
    rules=fraud_rules,
    conflict_config=conflict_resolution,
    base_score_col="adjusted_anomaly_score",
    score_cap=1.0,
    default_action="PASS")
```


```python
demo_result = demo_engine.evaluate_dataframe(demo_df)
```


```python
for case_name in demo_cases.keys():
    row = demo_result.loc[case_name]

    base_score     = row["adjusted_anomaly_score"]
    rule_score     = row["rule_adjusted_score"]
    fired_count    = row["rule_fired_count"]
    severity       = row["rule_max_severity"]
    action         = row["rule_final_action"]
    fired_names    = row["rule_rule_names"]
    explanations   = row["rule_explanations"]

    print(f"\nCase        : {case_name}")
    print(f"  Base Score  : {base_score:.4f}  →  Rule Score: {rule_score:.4f}")
    print(f"  Rules Fired : {fired_count}  |  Max Severity: {severity}  |  Action: {action}")

    if not fired_names:
        print("  Rules       : None fired — transaction is clean")
    else:
        print("  Rules fired :")
        for rule_name, explanation in zip(fired_names, explanations):
            print(f"    ✓ {rule_name}")
            print(f"      → {explanation}")

print("\n" + "=" * 80)
print("Summary table:")
display(demo_result[[
    "adjusted_anomaly_score",
    "rule_adjusted_score",
    "rule_fired_count",
    "rule_max_severity",
    "rule_final_action"]])
```

### Evaluation Helper Functions


```python
def evaluate_rule_score_quality(
    df,
    target_col="isFraud",
    base_score_col="adjusted_anomaly_score",
    rule_score_col="rule_adjusted_score"
):
    temp = df[[target_col, base_score_col, rule_score_col]].copy()
    temp = temp.replace([np.inf, -np.inf], np.nan).dropna()

    y_true = temp[target_col].astype(int)

    output = {}

    for score_col in [base_score_col, rule_score_col]:
        y_score = pd.to_numeric(temp[score_col], errors="coerce").fillna(0)

        output[score_col] = {
            "AUC": roc_auc_score(y_true, y_score),
            "PR_AUC": average_precision_score(y_true, y_score)
        }

    result = (
        pd.DataFrame(output)
        .T
        .reset_index()
        .rename(columns={"index": "score_type"})
    )

    return result


def evaluate_top_percent_capture(
    df,
    target_col="isFraud",
    score_col="rule_adjusted_score",
    top_percent_list=None
):
    if top_percent_list is None:
        top_percent_list = [0.005, 0.01, 0.02, 0.03, 0.05, 0.10]

    temp = df[[target_col, score_col]].copy()
    temp = temp.replace([np.inf, -np.inf], np.nan).dropna()
    temp = temp.sort_values(score_col, ascending=False).reset_index(drop=True)

    total_fraud = int(temp[target_col].sum())
    rows = []

    for top_pct in top_percent_list:
        top_n = max(1, int(len(temp) * top_pct))
        selected = temp.head(top_n)

        fraud_caught = int(selected[target_col].sum())
        false_positives = int(top_n - fraud_caught)

        precision = fraud_caught / top_n if top_n > 0 else 0
        recall = fraud_caught / total_fraud if total_fraud > 0 else 0
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall > 0
            else 0
        )
        f2 = (
            5 * precision * recall / (4 * precision + recall)
            if precision + recall > 0
            else 0
        )

        rows.append({
            "top_pct": top_pct,
            "top_n": top_n,
            "fraud_caught": fraud_caught,
            "total_fraud": total_fraud,
            "false_positives": false_positives,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "f2": f2
        })

    return pd.DataFrame(rows)


def evaluate_thresholds(
    df,
    target_col="isFraud",
    score_col="rule_adjusted_score",
    thresholds=None
):
    if thresholds is None:
        thresholds = np.round(np.arange(0.10, 0.96, 0.01), 2)

    temp = df[[target_col, score_col]].copy()
    temp = temp.replace([np.inf, -np.inf], np.nan).dropna()

    y_true = temp[target_col].astype(int)
    y_score = pd.to_numeric(temp[score_col], errors="coerce").fillna(0)

    rows = []

    for threshold in thresholds:
        y_pred = (y_score >= threshold).astype(int)

        predicted_fraud_count = int(y_pred.sum())
        false_positives = int(((y_pred == 1) & (y_true == 0)).sum())

        rows.append({
            "threshold": threshold,
            "predicted_fraud_count": predicted_fraud_count,
            "false_positives": false_positives,
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1": f1_score(y_true, y_pred, zero_division=0),
            "f2": fbeta_score(y_true, y_pred, beta=2, zero_division=0)
        })

    return (
        pd.DataFrame(rows)
        .sort_values("f2", ascending=False)
        .reset_index(drop=True)
    )
```

### Evaluate on Train & Test


```python
engine = RuleEngine(
    rules=fraud_rules,
    conflict_config=conflict_resolution,
    base_score_col="adjusted_anomaly_score",
    score_cap=1.0,
    default_action="PASS",
    verbose=True)
```


```python
train_rules = engine.evaluate_dataframe(train, verbose=True)
test_rules = engine.evaluate_dataframe(test, verbose=True)
```

### Rule Boost


```python
eval_df = train_rules.copy() if "train_rules" in globals() else train.copy()

target_col = "isFraud"
baseline_score_col = "adjusted_anomaly_score"
rule_score_col = "rule_adjusted_score"

required_cols = [
    target_col,
    baseline_score_col,
    rule_score_col,
    "rule_rule_ids"]

missing_cols = [col for col in required_cols if col not in eval_df.columns]

if missing_cols:
    raise ValueError(f"Missing required columns: {missing_cols}")

y = eval_df[target_col].astype(int)

baseline_score = (
    pd.to_numeric(eval_df[baseline_score_col], errors="coerce")
    .fillna(0)
    .clip(0, 1)
)

rule_boosted_score = (
    pd.to_numeric(eval_df[rule_score_col], errors="coerce")
    .fillna(0)
    .clip(0, 1)
)

baseline_auc = roc_auc_score(y, baseline_score)
baseline_pr_auc = average_precision_score(y, baseline_score)

rule_boosted_auc = roc_auc_score(y, rule_boosted_score)
rule_boosted_pr_auc = average_precision_score(y, rule_boosted_score)

print("Score Quality: Rule Boost Impact")

print(f"{'Score Type':40s}  {'AUC':>8}  {'PR-AUC':>8}")
print(f"{'Baseline anomaly score':40s}  {baseline_auc:8.4f}  {baseline_pr_auc:8.4f}")
print(f"{'Rule-boosted anomaly score':40s}  {rule_boosted_auc:8.4f}  {rule_boosted_pr_auc:8.4f}")
print(f"{'Delta':40s}  {rule_boosted_auc - baseline_auc:+8.4f}  {rule_boosted_pr_auc - baseline_pr_auc:+8.4f}")


print("\n" + "=" * 120)
print("Per-Rule Coverage, Fraud Rate and Rule Boost")
print("=" * 120)

print(
    f"{'Rule ID':<12} "
    f"{'Name':<45} "
    f"{'#Fired':>10} "
    f"{'Coverage%':>10} "
    f"{'Fraud%':>10} "
    f"{'Lift':>8} "
    f"{'Rule Boost':>11} "
    f"{'Severity':<10} "
    f"{'Action':<10}"
)

print("-" * 130)

overall_fraud_rate = y.mean()
rule_summary_rows = []

for rule in engine.rules:
    rule_id = rule.get("id", rule.get("rule_id"))
    rule_name = rule.get("name", rule_id)
    rule_priority = rule.get("priority")
    rule_severity = rule.get("severity", "LOW")
    rule_action = rule.get("action", "REVIEW")
    rule_logic = rule.get("logic", "AND")
    rule_boost = float(rule.get("score_boost", rule.get("boost", 0.0)))

    mask = eval_df["rule_rule_ids"].apply(
        lambda fired_rules: rule_id in fired_rules if isinstance(fired_rules, list) else False
    )

    n_fired = int(mask.sum())
    coverage = n_fired / len(eval_df) if len(eval_df) > 0 else 0

    if n_fired == 0:
        fraud_rate = np.nan
        lift = np.nan
    else:
        fraud_rate = y.loc[mask].mean()
        lift = fraud_rate / overall_fraud_rate if overall_fraud_rate > 0 else np.nan

    rule_summary_rows.append({
        "rule_id": rule_id,
        "rule_name": rule_name,
        "priority": rule_priority,
        "severity": rule_severity,
        "action": rule_action,
        "logic": rule_logic,
        "n_fired": n_fired,
        "coverage": coverage,
        "fraud_rate": fraud_rate,
        "lift": lift,
        "rule_boost": rule_boost
    })

    fraud_rate_text = "n/a" if pd.isna(fraud_rate) else f"{fraud_rate:.2%}"
    lift_text = "n/a" if pd.isna(lift) else f"{lift:.2f}x"

    print(
        f"{rule_id:<12} "
        f"{rule_name[:45]:<45} "
        f"{n_fired:>10,} "
        f"{coverage:>9.2%} "
        f"{fraud_rate_text:>10} "
        f"{lift_text:>8} "
        f"{rule_boost:>11.2f} "
        f"{rule_severity:<10} "
        f"{rule_action:<10}"
    )

rule_summary_df = pd.DataFrame(rule_summary_rows)

rule_summary_df = (
    rule_summary_df
    .sort_values(
        by=["fraud_rate", "n_fired"],
        ascending=[False, False]
    )
    .reset_index(drop=True)
)

print("\nOverall train fraud rate:")
print(f"{overall_fraud_rate:.2%}")

print("\nTop rules by fraud rate:")
display(rule_summary_df.head(10))
```

### Explainability Output


```python
eval_df = train_rules if "train_rules" in globals() else train

target_col = "isFraud"
base_score_col = "adjusted_anomaly_score"
rule_score_col = "rule_adjusted_score"

required_cols = [
    target_col,
    base_score_col,
    rule_score_col,
    "rule_rule_ids",
    "rule_rule_names",
    "rule_explanations",
    "rule_fired_count",
    "rule_max_severity",
    "rule_final_action"]

missing_cols = [col for col in required_cols if col not in eval_df.columns]

if missing_cols:
    raise ValueError(f"Missing required columns for explainability output: {missing_cols}")


basic_cols = [
    "TransactionID",
    "TransactionAmt",
    "ProductCD",
    "card4",
    "card6",
    "transaction_hour",
    "is_night_transaction",
    "is_weekend",
    base_score_col,
    "rule_total_boost",
    rule_score_col,
    "rule_fired_count",
    "rule_max_severity",
    "rule_final_action",
    "rule_rule_ids",
    "rule_rule_names",
    "rule_explanations"
]

available_cols = [col for col in basic_cols if col in eval_df.columns]

fraud_view = eval_df.loc[
    eval_df[target_col].eq(1),
    available_cols
]

top_fraud = fraud_view.nlargest(5, rule_score_col)

print("=" * 90)
print("Top 5 Highest-Scoring Fraud Transactions — Explainability Output")
print("=" * 90)

for i, (idx, row) in enumerate(top_fraud.iterrows(), 1):
    transaction_id = row.get("TransactionID", "N/A")
    amount = row.get("TransactionAmt", 0)
    base_score = row.get(base_score_col, 0)
    rule_boost = row.get("rule_total_boost", 0)
    rule_score = row.get(rule_score_col, 0)
    severity = row.get("rule_max_severity", "N/A")
    action = row.get("rule_final_action", "N/A")
    fired_count = row.get("rule_fired_count", 0)

    print("\n" + "-" * 90)
    print(
        f"#{i} | "
        f"TxID={transaction_id} | "
        f"Amount={amount:.2f} | "
        f"BaseScore={base_score:.4f} | "
        f"RuleBoost={rule_boost:.4f} | "
        f"RuleScore={rule_score:.4f} | "
        f"Severity={severity} | "
        f"Action={action}"
    )

    print(f"Rules fired count: {fired_count}")

    fired_rule_ids = row.get("rule_rule_ids", [])
    fired_rule_names = row.get("rule_rule_names", [])
    explanations = row.get("rule_explanations", [])

    if not isinstance(fired_rule_ids, list):
        fired_rule_ids = [fired_rule_ids]

    if not isinstance(fired_rule_names, list):
        fired_rule_names = [fired_rule_names]

    if not isinstance(explanations, list):
        explanations = [explanations]

    if len(fired_rule_ids) == 0:
        print("Rules fired: None")
    else:
        print("Rules fired:")

        for rule_id, rule_name, explanation in zip(
            fired_rule_ids[:3],
            fired_rule_names[:3],
            explanations[:3]
        ):
            print(f"  - {rule_id}: {rule_name}")
            print(f"    Explanation: {explanation}")

        if len(fired_rule_ids) > 3:
            print(f"  ... {len(fired_rule_ids) - 3} more rules")

print("\n" + "=" * 90)
print("Compact transaction detail table:")

compact_display_cols = [
    col for col in [
        "TransactionID",
        "TransactionAmt",
        "ProductCD",
        base_score_col,
        "rule_total_boost",
        rule_score_col,
        "rule_fired_count",
        "rule_max_severity",
        "rule_final_action"
    ]
    if col in top_fraud.columns
]

display(top_fraud[compact_display_cols])
```

Rule engine is used mainly for explainability, prioritization and business decision support; score boosting was disabled because it did not improve ranking quality.
