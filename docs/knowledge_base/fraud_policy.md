# Aegis Fraud Detection Policy

## Overview

Aegis is a real-time fraud detection system that combines AI-based anomaly detection
with an explainable rule engine. All transactions are scored by an anomaly model and
then evaluated by a configurable rule engine that produces actionable decisions.

## Risk Levels

| Severity | Score Range | Action              | Description                                               |
|----------|-------------|---------------------|-----------------------------------------------------------|
| CRITICAL | 0.78 – 1.00 | FLAG_CRITICAL_REVIEW | Highest confidence fraud signal; immediate manual review  |
| HIGH     | 0.65 – 0.77 | FLAG_REVIEW          | Strong fraud indicators; standard review queue            |
| MEDIUM   | 0.50 – 0.64 | FLAG_REVIEW          | Moderate risk; monitor and review                         |
| LOW      | 0.30 – 0.49 | MONITOR              | Low-risk anomaly; passive monitoring                      |
| NONE     | 0.00 – 0.29 | PASS                 | Transaction appears normal                                |

## Fraud Categories

### 1. Account Takeover (ATO)
Fraudster gains unauthorized access to a legitimate account and initiates transactions.
- Indicators: sudden change in device/IP, off-hours activity, high-value transactions,
  low entity trust score, new account with immediate high-value transactions.
- Policy: Block if combined anomaly score > 0.80 and entity trust score < 0.20.

### 2. Card-Not-Present (CNP) Fraud
Fraudulent use of card details without physical card. Common in e-commerce.
- Indicators: high transaction amount, suspicious email domain (protonmail, tempmail),
  credit card usage, foreign country or unusual IP geolocation.
- Policy: Flag for critical review if amount > 500 and anomaly score > 0.70.

### 3. Velocity Fraud
Fraudster opens new accounts and immediately initiates many transactions.
- Indicators: time since first transaction < 2 hours, multiple transactions in short
  window, high anomaly score.
- Policy: Flag for review if time_since_first_transaction < 7200 seconds
  and anomaly score > 0.65.

### 4. Synthetic Identity Fraud
Fraudster creates a fake identity using a mix of real and fabricated information.
- Indicators: no prior transaction history, unusual identity signals, email domain
  from known disposable providers.
- Policy: Manual review for new entities with high anomaly scores.

### 5. Transaction Amount Anomaly
Transaction amount is significantly higher than normal for the entity, product, or
global average.
- Indicators: ctx_amount_zscore > 4.0, ctx_amount_ratio_to_global_median > 5.0,
  ctx_amount_vs_product_avg > 300.
- Policy: Flag for critical review if amount anomaly is combined with high model score.

### 6. Off-Hours Fraud
Transactions initiated outside normal business hours, especially at night, are higher
risk when combined with other fraud signals.
- Indicators: is_night_transaction = 1, is_business_hour = 0, high anomaly score,
  high anomaly flag ratio.
- Policy: Flag for review if off-hours with anomaly score > 0.75.

### 7. Established Account High-Value Transactions
When an old, long-standing, or established account with a positive history makes a
high-value transaction, the risk is treated very differently from a new account.
- Indicators: account age greater than 30 days, high entity trust score, consistent
  prior transaction history, no velocity spike, recognized device and location.
- Policy: A high-value transaction from a trusted, established account is generally
  allowed and only passively monitored. It is escalated to standard review only when it
  is combined with other strong fraud signals such as off-hours activity, a disposable
  email domain, a country mismatch, or an amount far above the account's normal spending
  pattern. A long account age and a strong trust history reduce the overall risk score,
  so established accounts are rarely blocked outright for amount alone.

## Conflict Resolution Policy

When multiple fraud rules fire for the same transaction:
1. The rule with the highest priority (lowest numeric priority value) determines the
   final action.
2. The highest severity across all fired rules is used for escalation.
3. All fired rule explanations are preserved for analyst review.
4. Maximum 5 rules per transaction are reported to avoid explanation overload.

## Explainability Requirements

Every fraud decision must be accompanied by:
- List of fired rules with rule IDs, names, and descriptions.
- Base anomaly score and final rule-adjusted score.
- Maximum severity level and final recommended action.
- Plain-language explanation of why each rule fired.

## Review Actions

| Action               | Description                                              |
|----------------------|----------------------------------------------------------|
| PASS                 | No fraud signal detected; allow transaction              |
| MONITOR              | Low-risk; log and passively monitor                      |
| FLAG_REVIEW          | Standard fraud review queue; analyst reviews within 4h   |
| FLAG_CRITICAL_REVIEW | Critical fraud review queue; analyst reviews within 15m  |
| BLOCK                | Immediately block transaction pending review             |
| DECLINE              | Permanently decline transaction                          |