# Context Adjust Engine


```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score, classification_report, confusion_matrix, precision_score, recall_score, fbeta_score
pd.options.future.infer_string = False
from lightgbm import LGBMClassifier
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")
```

### Data


```python
train = pd.read_pickle('../data/interim/train_score_fused.pkl')
test  = pd.read_pickle('../data/interim/test_score_fused.pkl')

TARGET   = 'isFraud'
ID_COLS  = ['TransactionID']
TIME_COL = 'TransactionDT'
SCORE_COL = 'final_raw_anomaly_score'

y = train[TARGET].astype(int)
```


```python
print(f'Fraud rate: {y.mean():.2%}')
baseline_auc    = roc_auc_score(y, train[SCORE_COL])
baseline_pr_auc = average_precision_score(y, train[SCORE_COL])
print(f'\nBaseline AUC={baseline_auc:.2f}  PR-AUC={baseline_pr_auc:.2f}')
```

### Business Hours Context & Weekend Adjustment

This section adds simple business context rules before the final fraud score is created. The goal is not to replace the model, but to make the risk score more realistic.

Context rules used here:

- Night transactions are slightly riskier because fraud often happens outside normal customer activity hours.
- Business-hour and morning transactions are slightly less risky because they are more common and easier to verify.
- Evening transactions receive a small risk increase because they are outside core business hours.
- Very new transaction timelines receive a small risk increase because there is less history to trust.
- The adjustment is clipped between `0.80` and `1.30` so context can influence the score without dominating it.


```python
TIME_COLS = [
    'transaction_hour', 'transaction_day', 'transaction_week',
    'transaction_day_of_week', 'is_weekend', 'is_night_transaction',
    'is_business_hour', 'is_morning', 'is_afternoon', 'is_evening',
    'hour_sin', 'hour_cos', 'dayofweek_sin', 'dayofweek_cos',
    'time_since_first_transaction']

def compute_time_context_score(df):
    df = df.copy()
    m = np.ones(len(df), dtype=np.float64)

    if 'is_night_transaction' in df.columns:
        m += 0.08 * df['is_night_transaction'].fillna(0).values

    if 'is_weekend' in df.columns:
        m += 0.06 * df['is_weekend'].fillna(0).values

    if 'is_business_hour' in df.columns:
        m -= 0.08 * df['is_business_hour'].fillna(0).values

    if 'is_morning' in df.columns:
        m -= 0.05 * df['is_morning'].fillna(0).values

    if 'is_evening' in df.columns:
        m += 0.05 * df['is_evening'].fillna(0).values

    if 'hour_sin' in df.columns:
        m += 0.03 * df['hour_sin'].fillna(0).values

    if 'time_since_first_transaction' in df.columns:
        is_new = (df['time_since_first_transaction'].fillna(0) == 0).values.astype(float)
        m += 0.05 * is_new

    df['time_context_score'] = np.clip(m, 0.80, 1.30).astype(np.float32)
    return df

def add_weighted_time_context_score(train, test, **_ignored):
    train = compute_time_context_score(train)
    test  = compute_time_context_score(test)
    return train, test
```


```python
train, test = add_weighted_time_context_score(
    train=train,
    test=test,
    time_col=TIME_COL,
    target_col=TARGET,
    n_splits=5,
    smoothing=100)
```

### Trusted Entity Adjustment

This section separates two ideas:

- `entity_context_score` is model-based. It learns risk from entity, relationship, amount, and time features.
- `entity_trusted_score` is rule-based. It marks entities as more trusted when they have enough transaction history, are not already high-risk, and show low context risk.

Trusted entity rule:

If an entity has frequent previous activity, is not marked as a high-risk entity, is not an outlier, and its model-based context risk is low, the notebook applies a small score reduction. This helps reduce false positives for normal repeat customers or stable entity patterns.

The reduction is intentionally conservative: trusted entities can lower the final score, but they cannot remove risk completely.


```python
CONTEXT_MODEL_COLS = [
    'M6_transaction_count', 'M6_avg_amount', 'M6_median_amount',
    'M6_std_amount', 'M6_amount_vs_avg', 'M6_amount_ratio',
    'card4_transaction_count', 'card4_avg_amount', 'card4_median_amount',
    'card4_std_amount', 'card4_amount_vs_avg', 'card4_amount_ratio',
    'card6_transaction_count', 'card6_avg_amount', 'card6_median_amount',
    'card6_std_amount', 'card6_amount_vs_avg', 'card6_amount_ratio',
    'ProductCD_transaction_count', 'ProductCD_avg_amount', 'ProductCD_median_amount',
    'ProductCD_std_amount', 'ProductCD_amount_vs_avg', 'ProductCD_amount_ratio',
    'P_emaildomain_transaction_count', 'P_emaildomain_avg_amount',
    'P_emaildomain_median_amount', 'P_emaildomain_std_amount',
    'P_emaildomain_amount_vs_avg', 'P_emaildomain_amount_ratio',
    'card3_transaction_count', 'card3_avg_amount', 'card3_median_amount',
    'card3_std_amount', 'card3_amount_vs_avg', 'card3_amount_ratio',
    'card5_transaction_count', 'card5_avg_amount', 'card5_median_amount',
    'card5_std_amount', 'card5_amount_vs_avg', 'card5_amount_ratio',
    'rel_M6_card4_count', 'rel_M6_card4_high_risk_flag',
    'rel_M6_card4_avg_amount_count', 'rel_M6_card4_avg_amount_high_risk_flag',
    'rel_M6_card4_median_amount_count', 'rel_M6_card4_median_amount_high_risk_flag',
    'rel_M6_card4_std_amount_count', 'rel_M6_card4_std_amount_high_risk_flag',
    'rel_M6_card4_transaction_count_count', 'rel_M6_card4_transaction_count_high_risk_flag',
    'rel_M6_avg_amount_card4_count', 'rel_M6_avg_amount_card4_high_risk_flag',
    'ctx_transaction_amount_log', 'ctx_amount_vs_global_median',
    'ctx_amount_ratio_to_global_median', 'ctx_amount_zscore',
    'ctx_is_high_amount', 'ctx_is_low_amount',
    'ctx_amount_vs_hour_avg', 'ctx_amount_vs_product_avg', 'ctx_high_amount_weekend',
    'is_outlier', 'is_high_risk_entity',]
```


```python
_ctx_features = [c for c in (CONTEXT_MODEL_COLS + TIME_COLS) if c in train.columns and c in test.columns]

_X_ctx_train = train[_ctx_features].replace([np.inf, -np.inf], np.nan).fillna(0).astype(np.float32)
_X_ctx_test  = test[_ctx_features].replace([np.inf, -np.inf], np.nan).fillna(0).astype(np.float32)

_y = train[TARGET].astype(int)
_imbal = (_y == 0).sum() / (_y == 1).sum()

_cv  = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
_oof = np.zeros(len(_y), dtype=np.float32)
_test_preds = []

print(f"Training context LightGBM  (3-fold OOF, imbalance_ratio={_imbal:.1f}x)")
for fold, (tr_idx, val_idx) in enumerate(_cv.split(_X_ctx_train, _y), 1):
    _m = LGBMClassifier(
        n_estimators=300, learning_rate=0.05, num_leaves=31, max_depth=5,
        subsample=0.80, colsample_bytree=0.80,
        scale_pos_weight=_imbal,
        random_state=42 + fold, n_jobs=2, verbose=-1,
    )
    _m.fit(_X_ctx_train.iloc[tr_idx], _y.iloc[tr_idx])
    _oof[val_idx] = _m.predict_proba(_X_ctx_train.iloc[val_idx])[:, 1].astype(np.float32)
    _test_preds.append(_m.predict_proba(_X_ctx_test)[:, 1].astype(np.float32))

    fold_auc = roc_auc_score(_y.iloc[val_idx], _oof[val_idx])
    fold_pr  = average_precision_score(_y.iloc[val_idx], _oof[val_idx])

train["entity_context_score"] = _oof
test["entity_context_score"]  = np.mean(_test_preds, axis=0).astype(np.float32)

# Rule-based trusted entity score: lower value means the entity looks more trusted.
def add_trusted_entity_rule(df):
    df = df.copy()

    count_cols = [
        c for c in df.columns
        if c.endswith("_transaction_count") and not c.startswith("rel_")
    ]

    if count_cols:
        activity_count = df[count_cols].replace([np.inf, -np.inf], np.nan).fillna(0).max(axis=1)
        frequent_entity = activity_count >= activity_count.quantile(0.75)
    else:
        frequent_entity = pd.Series(False, index=df.index)

    def zero_flag(col_name):
        if col_name in df.columns:
            return df[col_name].fillna(0).astype(int)
        return pd.Series(0, index=df.index)

    low_context_risk = df["entity_context_score"].fillna(1) <= 0.35
    not_high_risk = zero_flag("is_high_risk_entity").eq(0)
    not_outlier = zero_flag("is_outlier").eq(0)

    trusted_entity_flag = frequent_entity & low_context_risk & not_high_risk & not_outlier

    df["trusted_entity_flag"] = trusted_entity_flag.astype(int)
    df["entity_trusted_score"] = df["entity_context_score"].copy()
    df.loc[trusted_entity_flag, "entity_trusted_score"] = (
        df.loc[trusted_entity_flag, "entity_context_score"] * 0.85
    )
    df["entity_trusted_score"] = df["entity_trusted_score"].clip(0, 1).astype(np.float32)
    return df

train = add_trusted_entity_rule(train)
test = add_trusted_entity_rule(test)

ctx_model_auc = roc_auc_score(_y, train["entity_context_score"])
ctx_model_pr  = average_precision_score(_y, train["entity_context_score"])

trusted_rule_summary = pd.DataFrame({
    "dataset": ["train", "test"],
    "trusted_rows": [
        int(train["trusted_entity_flag"].sum()),
        int(test["trusted_entity_flag"].sum()),
    ],
    "trusted_ratio_percent": [
        round(train["trusted_entity_flag"].mean() * 100, 2),
        round(test["trusted_entity_flag"].mean() * 100, 2),
    ],
    "avg_score_reduction": [
        round(float((train["entity_context_score"] - train["entity_trusted_score"]).mean()), 5),
        round(float((test["entity_context_score"] - test["entity_trusted_score"]).mean()), 5),
    ],
})

print("Trusted entity rule summary")
display(trusted_rule_summary)
```

### Final Anomaly Score

The final score blends the original anomaly score with the context-adjusted entity score. In this version, the final blend uses `entity_trusted_score`, so trusted repeat entities can receive a small risk reduction while still keeping the original anomaly signal as the main driver.


```python
W_RAW     = 0.80   
W_CONTEXT = 0.20  # time+entity

def minmax_normalize_using_train(train, test, col):

    train = train.copy()
    test  = test.copy()

    vals_tr = train[col].replace([np.inf, -np.inf], np.nan).fillna(0)
    vals_te = test[col].replace([np.inf, -np.inf], np.nan).fillna(0)

    mn, mx = float(vals_tr.min()), float(vals_tr.max())
    norm_col = f"norm_{col}"

    if mx == mn:
        train[norm_col] = 0.0
        test[norm_col]  = 0.0
    else:
        train[norm_col] = ((vals_tr - mn) / (mx - mn)).clip(0, 1).astype(np.float32)
        test[norm_col]  = ((vals_te - mn) / (mx - mn)).clip(0, 1).astype(np.float32)

    return train, test, norm_col


def create_final_context_score(
    train,
    test,
    raw_score_col="final_raw_anomaly_score",
    context_score_col="entity_trusted_score",
    w_raw=W_RAW,
    w_context=W_CONTEXT,
):
    train = train.copy()
    test  = test.copy()

    for df in (train, test):
        raw_s = df[raw_score_col].clip(0, 1).astype(np.float32)
        ctx_s = df[context_score_col].clip(0, 1).astype(np.float32)
        df["final_context_anomaly_score"] = (
            w_raw * raw_s + w_context * ctx_s
        ).clip(0, 1).astype(np.float32)

    y = train[TARGET].astype(int)
    raw_auc  = roc_auc_score(y, train[raw_score_col])
    raw_pr   = average_precision_score(y, train[raw_score_col])
    ctx_auc  = roc_auc_score(y, train[context_score_col])
    ctx_pr   = average_precision_score(y, train[context_score_col])
    final_auc = roc_auc_score(y, train["final_context_anomaly_score"])
    final_pr  = average_precision_score(y, train["final_context_anomaly_score"])
    
    return train, test
```


```python
train, test = create_final_context_score(
    train=train,
    test=test,
    raw_score_col="final_raw_anomaly_score",
    context_score_col="entity_trusted_score",
    w_raw=W_RAW,
    w_context=W_CONTEXT)
```

### False Positive Reduction Analysis


```python
RAW_SCORE_COL = "final_raw_anomaly_score"
CONTEXT_SCORE_COL = "final_context_anomaly_score"

if RAW_SCORE_COL not in train.columns:
    raise ValueError(f"Missing {RAW_SCORE_COL}. Run notebook 05 first.")
if CONTEXT_SCORE_COL not in train.columns:
    raise ValueError(f"Missing {CONTEXT_SCORE_COL}. Run the context adjustment cell first.")

train["adjusted_anomaly_score"] = train[CONTEXT_SCORE_COL].astype(np.float32)
test["adjusted_anomaly_score"] = test[CONTEXT_SCORE_COL].astype(np.float32)

def top_percent_metrics(y_true, score, top_pct):
    y_true = np.asarray(y_true).astype(int)
    score = np.asarray(score)

    n = len(y_true)
    k = max(1, int(n * top_pct))
    top_idx = np.argsort(score)[::-1][:k]

    y_pred = np.zeros(n, dtype=int)
    y_pred[top_idx] = 1

    false_positives = int(((y_true == 0) & (y_pred == 1)).sum())
    fraud_caught = int(y_true[top_idx].sum())

    return {
        "top_%": f"{top_pct * 100:.1f}%",
        "flagged": k,
        "fraud_caught": fraud_caught,
        "false_positives": false_positives,
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "f2": fbeta_score(y_true, y_pred, beta=2, zero_division=0),
    }

pcts = [0.005, 0.01, 0.02, 0.03, 0.05, 0.075, 0.10, 0.15, 0.20]
y_np = y.values

before_df = pd.DataFrame([
    top_percent_metrics(y_np, train[RAW_SCORE_COL].values, p)
    for p in pcts
])
after_df = pd.DataFrame([
    top_percent_metrics(y_np, train[CONTEXT_SCORE_COL].values, p)
    for p in pcts
])

compare = before_df[["top_%", "flagged", "fraud_caught", "false_positives", "recall", "precision", "f2"]].copy()
compare.columns = [
    "top_%", "flagged", "fraud_caught_before", "fp_before",
    "recall_before", "precision_before", "f2_before",
]
compare["fraud_caught_after"] = after_df["fraud_caught"]
compare["fp_after"] = after_df["false_positives"]
compare["recall_after"] = after_df["recall"]
compare["precision_after"] = after_df["precision"]
compare["f2_after"] = after_df["f2"]

compare["fp_reduction"] = compare["fp_before"] - compare["fp_after"]
compare["fp_reduction_%"] = np.where(
    compare["fp_before"] > 0,
    100 * compare["fp_reduction"] / compare["fp_before"],
    0,
)
compare["precision_gain"] = compare["precision_after"] - compare["precision_before"]
compare["recall_delta"] = compare["recall_after"] - compare["recall_before"]
compare["f2_delta"] = compare["f2_after"] - compare["f2_before"]

metric_cols = [
    "recall_before", "recall_after", "precision_before", "precision_after",
    "f2_before", "f2_after", "fp_reduction_%", "precision_gain",
    "recall_delta", "f2_delta",
]
compare[metric_cols] = compare[metric_cols].round(4)

raw_auc = roc_auc_score(y, train[RAW_SCORE_COL])
raw_pr_auc = average_precision_score(y, train[RAW_SCORE_COL])
ctx_auc = roc_auc_score(y, train[CONTEXT_SCORE_COL])
ctx_pr_auc = average_precision_score(y, train[CONTEXT_SCORE_COL])

print("── Score Quality Before vs After Context Adjustment ───────────────")
print(f"Raw score      AUC={raw_auc:.4f} | PR-AUC={raw_pr_auc:.4f}")
print(f"Context score  AUC={ctx_auc:.4f} | PR-AUC={ctx_pr_auc:.4f}")
print(f"Delta          AUC={ctx_auc - raw_auc:+.4f} | PR-AUC={ctx_pr_auc - raw_pr_auc:+.4f}")

print("\n── False Positive Reduction by Top-% Risk Group ───────────────────")
display(compare)
```


```python
thresholds = np.quantile(train[CONTEXT_SCORE_COL], np.linspace(0.50, 0.995, 100))
threshold_rows = []

for threshold in thresholds:
    y_pred = (train[CONTEXT_SCORE_COL] >= threshold).astype(int)
    threshold_rows.append({
        "threshold": float(threshold),
        "predicted_fraud_count": int(y_pred.sum()),
        "false_positives": int(((y == 0) & (y_pred == 1)).sum()),
        "precision": precision_score(y, y_pred, zero_division=0),
        "recall": recall_score(y, y_pred, zero_division=0),
        "f1": precision_score(y, y_pred, zero_division=0) if y_pred.sum() == 0 else 0, 
        "f2": fbeta_score(y, y_pred, beta=2, zero_division=0),
    })

threshold_df = pd.DataFrame(threshold_rows).drop_duplicates("threshold")
threshold_df["f1"] = [
    0 if row.predicted_fraud_count == 0 else (2 * row.precision * row.recall / (row.precision + row.recall))
    if (row.precision + row.recall) > 0 else 0
    for row in threshold_df.itertuples()]
    
threshold_df = threshold_df.sort_values("f2", ascending=False).reset_index(drop=True)

best_context_threshold = float(threshold_df.iloc[0]["threshold"])
print("\n── Best Context Thresholds by F2 ───────────────────────────────────")
display(threshold_df.head(20).round(2))
```
