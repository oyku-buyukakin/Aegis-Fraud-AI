# Score Aggregation


```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score, classification_report, confusion_matrix
from xgboost import XGBClassifier as _XGB
from mrmr import mrmr_classif
from lightgbm import LGBMClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    fbeta_score,
    precision_score,
    recall_score)
pd.options.future.infer_string = False
import warnings
warnings.filterwarnings("ignore")
```

### Data


```python
train = pd.read_pickle("../data/interim/train_anomaly_scored.pkl")
test  = pd.read_pickle("../data/interim/test_anomaly_scored.pkl")

TARGET   = "isFraud"
ID_COLS  = ["TransactionID"]
TIME_COL = "TransactionDT"
```

### Normalize Anomaly Scores


```python
base_anomaly_score_cols = [
    col for col in train.columns
    if (
        col.endswith("_anomaly_score")
        and not col.startswith("norm_")
        and not col.startswith("inter_")
        and "_high_flag" not in col
    )
]

available_score_cols = [c for c in base_anomaly_score_cols if c in test.columns]
normalized_score_cols = []

for score_col in available_score_cols:
    train_vals = train[score_col].replace([np.inf, -np.inf], np.nan).fillna(0)
    test_vals  = test[score_col].replace([np.inf, -np.inf], np.nan).fillna(0)

    mn, mx = train_vals.min(), train_vals.max()
    norm_col = f"norm_{score_col}"
    normalized_score_cols.append(norm_col)

    if mx == mn:
        train[norm_col] = 0.0
        test[norm_col]  = 0.0
    else:
        train[norm_col] = ((train_vals - mn) / (mx - mn)).clip(0, 1)
        test[norm_col]  = ((test_vals  - mn) / (mx - mn)).clip(0, 1)

print(f"Normalized {len(normalized_score_cols)} anomaly score columns.")
```

### Anomaly Score Variables Feature Selection

In this section, the strongest normalized anomaly score features are selected and used to train a final LightGBM model that produces a unified fraud risk score.


The first block ranks anomaly score features with XGBoost, tests different top-K feature combinations using cross-validation, and applies mRMR to select the final non-redundant feature set.

The second block trains a LightGBM classifier with stratified cross-validation and class imbalance handling, then creates the final output score `final_raw_anomaly_score` for both train and test datasets.


```python
y = train[TARGET].astype(int)

X_pool = train[normalized_score_cols].fillna(0)
print(f"Candidate pool: {X_pool.shape[1]} normalized anomaly score columns")

_ranker = _XGB(
    n_estimators=300, learning_rate=0.05, max_depth=4,
    eval_metric="logloss", verbosity=0, random_state=42, n_jobs=-1,
)
_ranker.fit(X_pool, y)

_importance_order = list(
    pd.Series(_ranker.feature_importances_, index=normalized_score_cols)
    .sort_values(ascending=False)
    .index
)

_cv_k  = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
k_aucs, k_prs = {}, {}

print(f"\n── XGBoost K-sweep  (K = 1 … {len(_importance_order)}) ──────────────────────────")
for k in range(1, len(_importance_order) + 1):
    _feats = _importance_order[:k]
    _X_k   = X_pool[_feats]
    _fa, _fp = [], []

    for _tr, _val in _cv_k.split(_X_k, y):
        _m = _XGB(
            n_estimators=300, learning_rate=0.05, max_depth=4,
            eval_metric="logloss", verbosity=0, random_state=42, n_jobs=-1,
        )
        _m.fit(_X_k.iloc[_tr], y.iloc[_tr])
        _s = _m.predict_proba(_X_k.iloc[_val])[:, 1]
        _fa.append(roc_auc_score(y.iloc[_val], _s))
        _fp.append(average_precision_score(y.iloc[_val], _s))

    k_aucs[k] = float(np.mean(_fa))
    k_prs[k]  = float(np.mean(_fp))
    composite  = 0.6 * k_prs[k] + 0.4 * k_aucs[k]
    print(f"  K={k:2d}  AUC={k_aucs[k]:.4f}  PR-AUC={k_prs[k]:.4f}  composite={composite:.4f}")

_composites = {k: 0.6 * k_prs[k] + 0.4 * k_aucs[k] for k in k_aucs}
optimal_k   = max(_composites, key=_composites.get)

print(f"\n→ Optimal K = {optimal_k}")
print(f"  AUC={k_aucs[optimal_k]:.4f}  PR-AUC={k_prs[optimal_k]:.4f}  composite={_composites[optimal_k]:.4f}")

selected_features = mrmr_classif(X=X_pool, y=y, K=optimal_k)
print(f"\n✓ mRMR selected {len(selected_features)} feature(s):")
for f in selected_features:
    print(f"  · {f}  (AUC={roc_auc_score(y, X_pool[f]):.4f}  PR-AUC={average_precision_score(y, X_pool[f]):.4f})")

X      = X_pool[selected_features].astype(np.float32)
X_test = test[selected_features].fillna(0).astype(np.float32)
```

AUC is strong, and PR-AUC is useful for the imbalanced fraud problem. The composite score shows a good overall feature combination.


```python
imbalance_ratio = (y == 0).sum() / (y == 1).sum()
print(f"Imbalance ratio: {imbalance_ratio:.2f}x  |  Fraud rate: {y.mean():.4%}")

cv       = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
oof_prob = np.zeros(len(X), dtype=np.float32)
test_prob_folds = []

for fold, (tr_idx, val_idx) in enumerate(cv.split(X, y), 1):
    print(f"\nFold {fold}")
    X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
    y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]

    model = LGBMClassifier(
        n_estimators=700, learning_rate=0.03, num_leaves=63,
        max_depth=7, min_child_samples=80,
        subsample=0.80, colsample_bytree=0.80,
        reg_alpha=1.0, reg_lambda=5.0,
        scale_pos_weight=imbalance_ratio,
        random_state=42 + fold, n_jobs=2, verbose=-1,
    )
    model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], eval_metric="auc")

    oof_prob[val_idx] = model.predict_proba(X_val)[:, 1].astype(np.float32)
    fold_auc = roc_auc_score(y_val, oof_prob[val_idx])
    fold_pr  = average_precision_score(y_val, oof_prob[val_idx])
    print(f"  AUC={fold_auc:.4f}  PR-AUC={fold_pr:.4f}")

    test_prob_folds.append(model.predict_proba(X_test)[:, 1].astype(np.float32))
    del X_tr, X_val, y_tr, y_val, model

train["final_raw_anomaly_score"] = oof_prob
test["final_raw_anomaly_score"]  = np.mean(test_prob_folds, axis=0)

final_auc    = roc_auc_score(y, train["final_raw_anomaly_score"])
final_pr_auc = average_precision_score(y, train["final_raw_anomaly_score"])
print(f"\n{'─'*50}")
print(f"Final OOF AUC:    {final_auc:.4f}")
print(f"Final OOF PR-AUC: {final_pr_auc:.4f}")
```

### Weighted Aggregation & Final Raw Anomaly Score

Performance-weighted anomaly scores are generated by giving higher weights to signals with stronger PR-AUC. These scores are then blended with the LightGBM risk score to create the final `final_raw_anomaly_score`, while also keeping `weighted_anomaly_score` as a separate output.



```python
weight_rows = []
for col in normalized_score_cols:
    vals = train[col].replace([np.inf, -np.inf], np.nan).fillna(0)
    if vals.nunique() <= 1:
        pr_auc = 0.0
        roc_auc_val = 0.5
    else:
        pr_auc      = average_precision_score(y, vals)
        roc_auc_val = roc_auc_score(y, vals)
    weight_rows.append({"score": col, "pr_auc": pr_auc, "roc_auc": roc_auc_val})

weight_df = (
    pd.DataFrame(weight_rows)
    .sort_values("pr_auc", ascending=False)
    .reset_index(drop=True)
)

_raw_w = np.array(weight_df["pr_auc"].values, dtype=np.float64)
_raw_w = np.exp(_raw_w - _raw_w.max())         
_raw_w = _raw_w / _raw_w.sum()
weight_df["weight"] = _raw_w.round(6)

_score_matrix_train = train[normalized_score_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
_score_matrix_test  = test[normalized_score_cols].replace([np.inf, -np.inf], np.nan).fillna(0)

_weights_aligned = np.array(
    [weight_df.loc[weight_df["score"] == c, "weight"].values[0] for c in normalized_score_cols],
    dtype=np.float32,
)

weighted_score_train = (_score_matrix_train.values * _weights_aligned).sum(axis=1)
weighted_score_test  = (_score_matrix_test.values  * _weights_aligned).sum(axis=1)

_ws_min, _ws_max = weighted_score_train.min(), weighted_score_train.max()
if _ws_max > _ws_min:
    weighted_score_train = (weighted_score_train - _ws_min) / (_ws_max - _ws_min)
    weighted_score_test  = np.clip((weighted_score_test  - _ws_min) / (_ws_max - _ws_min), 0, 1)


_lgbm_pr  = average_precision_score(y, train["final_raw_anomaly_score"])
_wagg_pr  = average_precision_score(y, weighted_score_train)

_total_pr = _lgbm_pr + _wagg_pr
alpha     = _lgbm_pr / _total_pr       
beta      = _wagg_pr / _total_pr         

print(f"\nLightGBM standalone  PR-AUC : {_lgbm_pr:.4f}  → blend weight α = {alpha:.3f}")
print(f"Weighted-agg standalone PR-AUC: {_wagg_pr:.4f}  → blend weight β = {beta:.3f}")

train["final_raw_anomaly_score"] = (
    alpha * train["final_raw_anomaly_score"] + beta * weighted_score_train
).astype(np.float32)

test["final_raw_anomaly_score"] = (
    alpha * test["final_raw_anomaly_score"]  + beta * weighted_score_test
).astype(np.float32)

final_auc    = roc_auc_score(y, train["final_raw_anomaly_score"])
final_pr_auc = average_precision_score(y, train["final_raw_anomaly_score"])
print(f"\n{'─'*52}")
print(f"Final blended  AUC   : {final_auc:.4f}")
print(f"Final blended  PR-AUC: {final_pr_auc:.4f}  (was LightGBM={_lgbm_pr:.4f}, WA={_wagg_pr:.4f})")

train["weighted_anomaly_score"] = weighted_score_train.astype(np.float32)
test["weighted_anomaly_score"]  = weighted_score_test.astype(np.float32)
```

### Fraud Capture Check


```python
y_true = np.asarray(y).astype(int)
scores = train["final_raw_anomaly_score"].replace([np.inf, -np.inf], np.nan).fillna(0).to_numpy()


def top_percent_metrics(y_true, scores, top_pct):
    n = len(y_true)
    k = max(1, int(n * top_pct))
    top_idx = np.argsort(scores)[::-1][:k]

    y_pred = np.zeros(n, dtype=int)
    y_pred[top_idx] = 1

    return {
        "top_percent": f"{top_pct * 100:.1f}%",
        "flagged_transactions": k,
        "fraud_caught": int(y_true[top_idx].sum()),
        "total_fraud": int(y_true.sum()),
        "recall": round(recall_score(y_true, y_pred, zero_division=0), 4),
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "f1": round(f1_score(y_true, y_pred, zero_division=0), 4),
        "f2": round(fbeta_score(y_true, y_pred, beta=2, zero_division=0), 4),
    }


capture_levels = [0.005, 0.01, 0.02, 0.03, 0.05, 0.075, 0.10, 0.15, 0.20]
topk_df = pd.DataFrame([
    top_percent_metrics(y_true, scores, pct)
    for pct in capture_levels
])

print("Top-percent fraud capture")
display(topk_df)

plt.figure(figsize=(9, 4))
sns.barplot(data=topk_df, x="top_percent", y="recall", color="#4c72b0")
plt.title("Fraud Recall by Highest-Risk Group")
plt.xlabel("Highest-risk group")
plt.ylabel("Recall")
plt.tight_layout()
plt.show()

thresholds = np.unique(np.quantile(scores, np.linspace(0.50, 0.995, 100)))
thr_rows = []

for threshold in thresholds:
    y_pred = (scores >= threshold).astype(int)
    thr_rows.append({
        "threshold": round(float(threshold), 6),
        "predicted_fraud_count": int(y_pred.sum()),
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_true, y_pred, zero_division=0), 4),
        "f1": round(f1_score(y_true, y_pred, zero_division=0), 4),
        "f2": round(fbeta_score(y_true, y_pred, beta=2, zero_division=0), 4),
    })

thr_df = (
    pd.DataFrame(thr_rows)
    .sort_values("f2", ascending=False)
    .reset_index(drop=True)
)

print("\nThreshold sweep - best rows by F2")
display(thr_df.head(20))

best_threshold = float(thr_df.loc[0, "threshold"])
y_pred_best = (scores >= best_threshold).astype(int)

thr_plot = thr_df.sort_values("threshold")
plt.figure(figsize=(9, 4))
plt.plot(thr_plot["threshold"], thr_plot["precision"], label="Precision")
plt.plot(thr_plot["threshold"], thr_plot["recall"], label="Recall")
plt.plot(thr_plot["threshold"], thr_plot["f2"], label="F2", linestyle="--")
plt.axvline(best_threshold, color="red", linestyle=":", label=f"Best={best_threshold:.4f}")
plt.title("Precision, Recall, and F2 by Threshold")
plt.xlabel("Threshold")
plt.ylabel("Metric value")
plt.legend()
plt.tight_layout()
plt.show()

print(f"\nBest threshold by F2: {best_threshold:.6f}")
print("\nConfusion matrix")
print(confusion_matrix(y_true, y_pred_best))
print("\nClassification report")
print(classification_report(y_true, y_pred_best, zero_division=0))
```

## Summary

Different anomaly scores are combined into one final fraud risk score. The process normalizes anomaly signals, selects the strongest ones, trains a LightGBM model, blends the model score with weighted anomaly scores, and evaluates fraud capture performance. The final output is `final_raw_anomaly_score`, which can be used for fraud ranking and threshold-based risk decisions.
