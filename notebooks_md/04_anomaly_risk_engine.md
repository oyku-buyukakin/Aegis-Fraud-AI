# Multi-Layer Anomaly Detection Engine


```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
pd.options.future.infer_string = False 
import warnings
warnings.filterwarnings("ignore")
```

### Data


```python
train = pd.read_pickle("../data/interim/train_feature_engineered.pkl")
test = pd.read_pickle("../data/interim/test_feature_engineered.pkl")

TARGET = "isFraud"
ID_COLS = ["TransactionID"]
TIME_COL = "TransactionDT"
```


```python
# This function normalizes train and test values to the 0–1 range using the minimum and maximum values from the training data, while safely handling missing and infinite values.

def minmax_normalize(train_values, test_values=None):
    train_values = pd.Series(train_values).replace([np.inf, -np.inf], np.nan).fillna(0)
    train_min = train_values.min()
    train_max = train_values.max()
    denominator = train_max - train_min

    if denominator == 0:
        train_score = pd.Series(0, index=train_values.index)
        if test_values is None:
            return train_score
        return train_score, pd.Series(0, index=pd.Series(test_values).index)

    train_score = ((train_values - train_min) / denominator).clip(0, 1)

    if test_values is None:
        return train_score

    test_values = pd.Series(test_values).replace([np.inf, -np.inf], np.nan).fillna(0)
    test_score = ((test_values - train_min) / denominator).clip(0, 1)

    return train_score, test_score
```


```python
numeric_feature_cols = train.select_dtypes(include=["number"]).columns.tolist()

numeric_feature_cols = [
    col for col in numeric_feature_cols
    if col not in [TARGET] + ID_COLS + [TIME_COL]
    and col in test.columns]

print(f"Numeric features available for anomaly scoring: {len(numeric_feature_cols)}")
```

### Column Anomaly Detection


```python
# z-score and quartile based approach

mean_values = train[numeric_feature_cols].mean()
std_values = train[numeric_feature_cols].std().replace(0, np.nan)

train_z_scores = (train[numeric_feature_cols] - mean_values) / std_values
test_z_scores = (test[numeric_feature_cols] - mean_values) / std_values

train_z_flags = train_z_scores.abs() > 3
test_z_flags = test_z_scores.abs() > 3

train_z_raw_score = train_z_flags.mean(axis=1)
test_z_raw_score = test_z_flags.mean(axis=1)

train["zscore_column_anomaly_score"], test["zscore_column_anomaly_score"] = minmax_normalize(
    train_z_raw_score,
    test_z_raw_score
)

q1 = train[numeric_feature_cols].quantile(0.25)
q3 = train[numeric_feature_cols].quantile(0.75)

iqr = q3 - q1

lower_bounds = q1 - 1.5 * iqr
upper_bounds = q3 + 1.5 * iqr

train_iqr_flags = (
    (train[numeric_feature_cols] < lower_bounds)
    | (train[numeric_feature_cols] > upper_bounds)
)

test_iqr_flags = (
    (test[numeric_feature_cols] < lower_bounds)
    | (test[numeric_feature_cols] > upper_bounds)
)

train_iqr_raw_score = train_iqr_flags.mean(axis=1)
test_iqr_raw_score = test_iqr_flags.mean(axis=1)

train["iqr_column_anomaly_score"], test["iqr_column_anomaly_score"] = minmax_normalize(
    train_iqr_raw_score,
    test_iqr_raw_score
)
```

### Multivariate Anomaly Detection


```python
multivariate_cols = numeric_feature_cols.copy()

X_train_multi = train[multivariate_cols].replace([np.inf, -np.inf], np.nan)
X_test_multi = test[multivariate_cols].replace([np.inf, -np.inf], np.nan)


multivariate_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("isolation_forest", IsolationForest(
        n_estimators=200,
        contamination=0.03,
        max_samples=min(10000, len(X_train_multi)),
        random_state=42,
        n_jobs=-1))])


multivariate_pipeline.fit(X_train_multi)

train_multi_raw_score = -multivariate_pipeline.decision_function(X_train_multi)
test_multi_raw_score = -multivariate_pipeline.decision_function(X_test_multi)


train["multivariate_anomaly_score"], test["multivariate_anomaly_score"] = minmax_normalize(train_multi_raw_score, test_multi_raw_score)
```

### Entity Anomaly Detection


```python
entity_candidate_cols = [
    col for col in train.columns
    if (
        col.startswith("card")
        or col.startswith("addr")
        or col.startswith("Device")
        or col.startswith("id_")
        or col.startswith("M")
        or col in ["ProductCD", "P_emaildomain", "R_emaildomain"]
    )
    and col in test.columns
]

entity_candidate_cols = [
    col for col in entity_candidate_cols
    if col not in [TARGET] + ID_COLS + [TIME_COL]
]

entity_score_cols = []

for entity_col in entity_candidate_cols:

    entity_stats = (
        train
        .groupby(entity_col, dropna=False)
        .agg(
            entity_transaction_count=("TransactionAmt", "count"),
            entity_avg_amount=("TransactionAmt", "mean"),
            entity_std_amount=("TransactionAmt", "std")
        )
    )

    entity_stats["entity_std_amount"] = entity_stats["entity_std_amount"].fillna(0)

    global_entity_count = entity_stats["entity_transaction_count"].median()
    global_avg_amount = train["TransactionAmt"].mean()
    global_std_amount = train["TransactionAmt"].std()

    train_entity_count = train[entity_col].map(entity_stats["entity_transaction_count"]).fillna(global_entity_count)
    test_entity_count = test[entity_col].map(entity_stats["entity_transaction_count"]).fillna(global_entity_count)

    train_entity_avg = train[entity_col].map(entity_stats["entity_avg_amount"]).fillna(global_avg_amount)
    test_entity_avg = test[entity_col].map(entity_stats["entity_avg_amount"]).fillna(global_avg_amount)

    train_entity_std = train[entity_col].map(entity_stats["entity_std_amount"]).fillna(global_std_amount)
    test_entity_std = test[entity_col].map(entity_stats["entity_std_amount"]).fillna(global_std_amount)

    train_entity_rarity_raw = 1 / np.log1p(train_entity_count)
    test_entity_rarity_raw = 1 / np.log1p(test_entity_count)

    train_amount_deviation_raw = (
        (train["TransactionAmt"] - train_entity_avg).abs()
        / (train_entity_std + 1)
    )

    test_amount_deviation_raw = (
        (test["TransactionAmt"] - test_entity_avg).abs()
        / (test_entity_std + 1)
    )

    train_entity_rarity_score, test_entity_rarity_score = minmax_normalize(
        train_entity_rarity_raw,
        test_entity_rarity_raw
    )

    train_amount_deviation_score, test_amount_deviation_score = minmax_normalize(
        train_amount_deviation_raw,
        test_amount_deviation_raw
    )

    score_col = f"{entity_col}_entity_anomaly_score"

    train[score_col] = (
        0.5 * train_entity_rarity_score
        + 0.5 * train_amount_deviation_score
    )

    test[score_col] = (
        0.5 * test_entity_rarity_score
        + 0.5 * test_amount_deviation_score
    )

    entity_score_cols.append(score_col)


if len(entity_score_cols) > 0:
    train["entity_anomaly_score"] = train[entity_score_cols].mean(axis=1)
    test["entity_anomaly_score"] = test[entity_score_cols].mean(axis=1)
else:
    train["entity_anomaly_score"] = 0
    test["entity_anomaly_score"] = 0
```

### Temporal Anomaly Detection


```python
temporal_features = {}

time_group_cols = [
    col for col in [
        "transaction_hour",
        "transaction_day",
        "transaction_week",
        "transaction_day_of_week",
        "is_weekend",
        "is_night_transaction",
        "is_business_hour",
        "is_morning",
        "is_afternoon",
        "is_evening"
    ] if col in train.columns and col in test.columns]

for time_col in time_group_cols:
    train_time_counts = train[time_col].value_counts(dropna=False)

    train_time_frequency = train[time_col].map(train_time_counts).fillna(0)
    test_time_frequency = test[time_col].map(train_time_counts).fillna(0)

    train_time_rarity_raw = 1 / np.log1p(train_time_frequency)
    test_time_rarity_raw = 1 / np.log1p(test_time_frequency.replace(0, np.nan)).fillna(1)

    train_time_rarity_score, test_time_rarity_score = minmax_normalize(
        train_time_rarity_raw,
        test_time_rarity_raw
    )

    temporal_features[f"{time_col}_rarity"] = (
        train_time_rarity_score,
        test_time_rarity_score
    )


if temporal_features:
    train["temporal_anomaly_score"] = np.mean(
        [scores[0] for scores in temporal_features.values()],
        axis=0
    )

    test["temporal_anomaly_score"] = np.mean(
        [scores[1] for scores in temporal_features.values()],
        axis=0
    )
else:
    train["temporal_anomaly_score"] = 0
    test["temporal_anomaly_score"] = 0
```

### Anomaly Score Interactions

Interaction based anomaly features were generated using aggregation, weighting, and pairwise combinations of the most fraud-related anomaly scores.


```python
base_anomaly_score_cols = [
    col for col in train.columns
    if (col.endswith("_anomaly_score")
        and "_entity_anomaly_score_entity_anomaly_score" not in col
        and col in test.columns)]
```


```python
interaction_cols = []

train["anomaly_score_mean"] = train[base_anomaly_score_cols].mean(axis=1)
test["anomaly_score_mean"] = test[base_anomaly_score_cols].mean(axis=1)

train["anomaly_score_max"] = train[base_anomaly_score_cols].max(axis=1)
test["anomaly_score_max"] = test[base_anomaly_score_cols].max(axis=1)

train["anomaly_score_std"] = train[base_anomaly_score_cols].std(axis=1)
test["anomaly_score_std"] = test[base_anomaly_score_cols].std(axis=1)

high_flag_cols = []

for col in base_anomaly_score_cols:
    threshold = train[col].quantile(0.95)
    flag_col = f"{col}_high_flag"

    train[flag_col] = (train[col] >= threshold).astype(int)
    test[flag_col] = (test[col] >= threshold).astype(int)

    high_flag_cols.append(flag_col)

train["anomaly_high_flag_count"] = train[high_flag_cols].sum(axis=1)
test["anomaly_high_flag_count"] = test[high_flag_cols].sum(axis=1)

train["anomaly_high_flag_ratio"] = train[high_flag_cols].mean(axis=1)
test["anomaly_high_flag_ratio"] = test[high_flag_cols].mean(axis=1)

corr_weights = (
    train[base_anomaly_score_cols + [TARGET]]
    .corr(numeric_only=True)[TARGET]
    .drop(TARGET)
    .abs())

corr_weights = corr_weights / corr_weights.sum()

train["fraud_corr_weighted_anomaly_score"] = 0
test["fraud_corr_weighted_anomaly_score"] = 0

for col in base_anomaly_score_cols:
    train["fraud_corr_weighted_anomaly_score"] += train[col] * corr_weights[col]
    test["fraud_corr_weighted_anomaly_score"] += test[col] * corr_weights[col]


score_corr = (
    train[base_anomaly_score_cols + [TARGET]]
    .corr(numeric_only=True)[TARGET]
    .drop(TARGET)
    .abs()
    .sort_values(ascending=False))

top_score_cols = score_corr.head(10).index.tolist()

for i in range(len(top_score_cols)):
    for j in range(i + 1, len(top_score_cols)):
        col1 = top_score_cols[i]
        col2 = top_score_cols[j]

        interaction_col = f"inter_{col1}_x_{col2}"

        train[interaction_col] = train[col1] * train[col2]
        test[interaction_col] = test[col1] * test[col2]

        interaction_cols.append(interaction_col)


new_interaction_score_cols = [
    "anomaly_score_mean",
    "anomaly_score_max",
    "anomaly_score_std",
    "anomaly_high_flag_count",
    "anomaly_high_flag_ratio",
    "fraud_corr_weighted_anomaly_score"
] + interaction_cols

```

## Summary

This notebook builds anomaly risk signals for fraud detection. It creates different anomaly scores, combines them into stronger risk indicators, and adds interaction features so later models can better detect suspicious transactions.
