# Extended Test Rules — Experimental Fraud Rules 

---

## RULE_T01 — Card-Not-Present High Value Transaction

- Priority: 11
- Severity: HIGH
- Action: FLAG_REVIEW
- Logic: AND
- Tags: card_not_present, amount, cnp_fraud

### Conditions
- card6 == "credit"
- TransactionAmt > 500
- adjusted_anomaly_score >= 0.65

### Description
High-value credit card transaction flagged as card-not-present fraud.
CNP fraud occurs when card details are used without physical card presence,
common in e-commerce. Large amounts on credit cards with elevated anomaly
scores are a strong CNP fraud signal.

### Explanation Template
Credit card transaction of {TransactionAmt:.2f} with anomaly score
{adjusted_anomaly_score:.2f}. High-value CNP transactions on credit cards
are a known fraud vector.

---

## RULE_T02 — Disposable Email Domain

- Priority: 12
- Severity: MEDIUM
- Action: FLAG_REVIEW
- Logic: AND
- Tags: email, identity, synthetic_identity

### Conditions
- P_emaildomain in [protonmail.com, guerrillamail.com, mailinator.com, tempmail.com, throwam.com]
- adjusted_anomaly_score >= 0.55

### Description
Transaction associated with a known disposable or anonymous email domain.
Fraudsters frequently use disposable email services to avoid identity tracing.
Combined with an elevated anomaly score, this is a synthetic identity signal.

### Explanation Template
Transaction email domain is {P_emaildomain}, which is a known disposable
or anonymous email provider. Anomaly score is {adjusted_anomaly_score:.2f}.

---

## RULE_T03 — Multi-Entity Anomaly Cluster

- Priority: 13
- Severity: HIGH
- Action: FLAG_CRITICAL_REVIEW
- Logic: AND
- Tags: entity, cluster, multi_entity_anomaly

### Conditions
- card1_entity_anomaly_score >= 0.75
- card4_entity_anomaly_score >= 0.75
- card6_entity_anomaly_score >= 0.75
- adjusted_anomaly_score >= 0.70

### Description
Multiple entity-level anomaly scores are simultaneously elevated. This pattern
indicates coordinated fraud across card identity dimensions. When the card1,
card4, and card6 entity anomaly scores are all high together, it strongly
suggests account takeover or synthetic identity fraud at scale.

### Explanation Template
Card entity anomaly scores are all high: card1={card1_entity_anomaly_score:.3f},
card4={card4_entity_anomaly_score:.3f}, card6={card6_entity_anomaly_score:.3f}.
Adjusted anomaly score is {adjusted_anomaly_score:.2f}. Multi-entity anomaly
cluster detected.

---

## RULE_T04 — Trusted Entity Score Collapse

- Priority: 14
- Severity: CRITICAL
- Action: FLAG_CRITICAL_REVIEW
- Logic: AND
- Tags: entity, trust, ato

### Conditions
- entity_trusted_score < 0.10
- entity_context_score >= 0.80
- adjusted_anomaly_score >= 0.72

### Description
Entity trust score has collapsed near zero while context anomaly score is very
high. This combination is characteristic of an Account Takeover (ATO) event:
a formerly trusted entity now shows highly anomalous behavior. The sharp
divergence between trust and context scores is a strong ATO signal.

### Explanation Template
Entity trusted score is critically low at {entity_trusted_score:.3f} while
context anomaly score is {entity_context_score:.2f}. Adjusted anomaly score
is {adjusted_anomaly_score:.3f}. This pattern is consistent with account
takeover behavior.

---

## RULE_T05 — Weekend Night High-Value Transaction

- Priority: 15
- Severity: MEDIUM
- Action: FLAG_REVIEW
- Logic: AND
- Tags: temporal, weekend, night, amount

### Conditions
- is_night_transaction == 1
- is_weekend == 1
- TransactionAmt > 300
- adjusted_anomaly_score >= 0.60

### Description
High-value transaction initiated during weekend nights. Weekend night periods
have historically lower fraud monitoring capacity and are often exploited by
fraudsters for larger transactions. Combined with an elevated anomaly score,
this temporal pattern warrants review.

### Explanation Template
Transaction of {TransactionAmt:.2f} occurred on a weekend night. Weekend
night high-value transactions have elevated fraud risk. Anomaly score is
{adjusted_anomaly_score:.2f}.