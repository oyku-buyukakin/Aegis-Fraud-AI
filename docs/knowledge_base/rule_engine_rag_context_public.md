# Public Rule Engine Knowledge Base Context

This document summarizes the configurable fraud rule engine for RAG.
It intentionally excludes real competition transaction rows, labels, IDs, and row-level scores.

## Conflict Resolution

- Boost strategy: disabled
- Action strategy: highest_priority
- Severity strategy: highest_severity
- Score cap: 1.0

## Rule Catalog

### RULE_001 - Extreme Composite Anomaly Risk
- Priority: 1
- Severity: CRITICAL
- Action: FLAG_CRITICAL_REVIEW
- Logic: AND
- Description: Very high anomaly score combined with many high anomaly flags. This is the strongest high-precision fraud segment.

- Tags: ['anomaly', 'composite', 'high_precision']
- Conditions: [{'field': 'adjusted_anomaly_score', 'operator': 'gte', 'value': 0.78}, {'field': 'anomaly_high_flag_count', 'operator': 'gte', 'value': 10}]

### RULE_002 - High-Risk Entity with Strong Context Risk
- Priority: 2
- Severity: CRITICAL
- Action: FLAG_CRITICAL_REVIEW
- Logic: AND
- Description: Entity is high-risk and transaction context risk is also high. This captures suspicious repeat entity behavior.

- Tags: ['entity', 'context', 'anomaly']
- Conditions: [{'field': 'is_high_risk_entity', 'operator': 'eq', 'value': 1}, {'field': 'final_context_anomaly_score', 'operator': 'gte', 'value': 0.75}, {'field': 'adjusted_anomaly_score', 'operator': 'gte', 'value': 0.7}]

### RULE_003 - Outlier with Very Strong Model Risk
- Priority: 3
- Severity: CRITICAL
- Action: FLAG_CRITICAL_REVIEW
- Logic: AND
- Description: Transaction is a statistical outlier and the anomaly score is already high.

- Tags: ['outlier', 'anomaly', 'model_score']
- Conditions: [{'field': 'is_outlier', 'operator': 'eq', 'value': 1}, {'field': 'adjusted_anomaly_score', 'operator': 'gte', 'value': 0.7}]

### RULE_004 - Extreme Amount Deviation
- Priority: 4
- Severity: CRITICAL
- Action: FLAG_CRITICAL_REVIEW
- Logic: AND
- Description: Transaction amount is extremely unusual and the anomaly score is high.

- Tags: ['amount', 'zscore', 'anomaly']
- Conditions: [{'field': 'ctx_amount_zscore', 'operator': 'gt', 'value': 4.0}, {'field': 'adjusted_anomaly_score', 'operator': 'gte', 'value': 0.7}]

### RULE_005 - Off-Hours Strong Fraud Signal
- Priority: 5
- Severity: HIGH
- Action: FLAG_REVIEW
- Logic: AND
- Description: Transaction occurred outside business hours with high anomaly score and dense anomaly signals.

- Tags: ['time', 'off_hours', 'anomaly']
- Conditions: [{'field': 'is_business_hour', 'operator': 'eq', 'value': 0}, {'field': 'adjusted_anomaly_score', 'operator': 'gte', 'value': 0.75}, {'field': 'anomaly_high_flag_ratio', 'operator': 'gt', 'value': 0.25}]

### RULE_006 - Product Amount Spike with High Risk
- Priority: 6
- Severity: HIGH
- Action: FLAG_REVIEW
- Logic: AND
- Description: Transaction amount is far above product-level average and the anomaly score is high.

- Tags: ['product', 'amount', 'context']
- Conditions: [{'field': 'ctx_amount_vs_product_avg', 'operator': 'gt', 'value': 300}, {'field': 'TransactionAmt', 'operator': 'gt', 'value': 200}, {'field': 'adjusted_anomaly_score', 'operator': 'gte', 'value': 0.7}]

### RULE_007 - High Global Amount Ratio with High Risk
- Priority: 7
- Severity: HIGH
- Action: FLAG_REVIEW
- Logic: AND
- Description: Transaction amount is much higher than the global median and the anomaly score is high.

- Tags: ['amount', 'global_median', 'context']
- Conditions: [{'field': 'ctx_amount_ratio_to_global_median', 'operator': 'gt', 'value': 5.0}, {'field': 'adjusted_anomaly_score', 'operator': 'gte', 'value': 0.7}]

### RULE_008 - Dense Anomaly Pattern
- Priority: 8
- Severity: HIGH
- Action: FLAG_REVIEW
- Logic: AND
- Description: Multiple anomaly signals fired together while the anomaly score is high.

- Tags: ['anomaly', 'density']
- Conditions: [{'field': 'anomaly_high_flag_ratio', 'operator': 'gt', 'value': 0.35}, {'field': 'anomaly_high_flag_count', 'operator': 'gte', 'value': 8}, {'field': 'adjusted_anomaly_score', 'operator': 'gte', 'value': 0.68}]

### RULE_009 - New Account Velocity Risk
- Priority: 9
- Severity: HIGH
- Action: FLAG_REVIEW
- Logic: AND
- Description: Transaction originated from a very new account (first transaction within the last 2 hours) with a notable anomaly score. Mimics a velocity check: fraudsters often open accounts and transact immediately.

- Tags: ['velocity', 'new_account', 'temporal']
- Conditions: [{'field': 'time_since_first_transaction', 'operator': 'lt', 'value': 7200}, {'field': 'adjusted_anomaly_score', 'operator': 'gte', 'value': 0.65}]

### RULE_010 - Night Transaction with Untrusted Entity
- Priority: 10
- Severity: MEDIUM
- Action: FLAG_REVIEW
- Logic: AND
- Description: Transaction occurred at night and the entity trust score is very low. Combines temporal risk (off-hours activity typical of foreign / automated fraud) with a low-trust entity profile.

- Tags: ['night', 'entity', 'trust', 'foreign_proxy']
- Conditions: [{'field': 'is_night_transaction', 'operator': 'eq', 'value': 1}, {'field': 'entity_trusted_score', 'operator': 'lt', 'value': 0.3}, {'field': 'adjusted_anomaly_score', 'operator': 'gte', 'value': 0.6}]

## Synthetic Explainability Examples

### Synthetic Case: high_risk_night
- Base score: 0.8200
- Rule-adjusted score: 0.8200
- Fired rule count: 10
- Final severity: CRITICAL
- Final action: FLAG_CRITICAL_REVIEW
- Fired rules: ['Extreme Composite Anomaly Risk', 'High-Risk Entity with Strong Context Risk', 'Outlier with Very Strong Model Risk', 'Extreme Amount Deviation', 'Off-Hours Strong Fraud Signal', 'Product Amount Spike with High Risk', 'High Global Amount Ratio with High Risk', 'Dense Anomaly Pattern', 'New Account Velocity Risk', 'Night Transaction with Untrusted Entity']
- Explanations: ['Very high anomaly score combined with many high anomaly flags. This is the strongest high-precision fraud segment.\n', 'Entity is high-risk and transaction context risk is also high. This captures suspicious repeat entity behavior.\n', 'Transaction is a statistical outlier and the anomaly score is already high.\n', 'Transaction amount is extremely unusual and the anomaly score is high.\n', 'Transaction occurred outside business hours with high anomaly score and dense anomaly signals.\n', 'Transaction amount is far above product-level average and the anomaly score is high.\n', 'Transaction amount is much higher than the global median and the anomaly score is high.\n', 'Multiple anomaly signals fired together while the anomaly score is high.\n', 'Transaction originated from a very new account (first transaction within the last 2 hours) with a notable anomaly score. Mimics a velocity check: fraudsters often open accounts and transact immediately.\n', 'Transaction occurred at night and the entity trust score is very low. Combines temporal risk (off-hours activity typical of foreign / automated fraud) with a low-trust entity profile.\n']

### Synthetic Case: low_risk_daytime
- Base score: 0.1200
- Rule-adjusted score: 0.1200
- Fired rule count: 0
- Final severity: NONE
- Final action: PASS
- Fired rules: []
- Explanations: []