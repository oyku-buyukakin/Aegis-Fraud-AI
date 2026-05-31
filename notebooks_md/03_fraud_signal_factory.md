# Feature Engineering


```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge
from sklearn.ensemble import IsolationForest
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from itertools import combinations
pd.options.future.infer_string = False
import warnings
warnings.filterwarnings("ignore")
```

### Data


```python
train = pd.read_pickle("../data/interim/train_merged.pkl")
test = pd.read_pickle("../data/interim/test_merged.pkl")

TARGET = "isFraud"
ID_COLS = ["TransactionID"]
TIME_COL = "TransactionDT"
```

### High NaN Columns


```python
# Columns with excessively high missing value ratios are generally considered statistically unreliable and may reduce the robustness and interpretability of the analysis.

missing_threshold = 0.40
protected_cols = [TARGET] + ID_COLS + [TIME_COL]

train_missing_ratio = train.isna().mean()
high_missing_cols = train_missing_ratio[train_missing_ratio > missing_threshold].index.tolist()
high_missing_cols = [col for col in high_missing_cols if col not in protected_cols]

train_clean = train.drop(columns=high_missing_cols, errors="ignore")
test_clean = test.drop(columns=high_missing_cols, errors="ignore")

common_feature_cols = [
    col for col in train_clean.columns
    if col != TARGET and col in test_clean.columns
]

train_clean = train_clean[common_feature_cols + [TARGET]]
test_clean = test_clean[common_feature_cols]

missing_drop_summary = pd.DataFrame({
    "dropped_column": high_missing_cols,
    "train_missing_ratio_percent": [round(train_missing_ratio[col] * 100, 2) for col in high_missing_cols],
})

print(f"Dropped columns with more than {missing_threshold:.0%} missing values: {len(high_missing_cols)}")
```

### Expectation Maximization Imputation


```python
# Expectation Maximization (EM) Imputation estimates missing values using statistical relationships in the data, while bfill/ffill simply copies nearby values.

train_imputed = train_clean.copy()
test_imputed = test_clean.copy()

numeric_cols = train_imputed.select_dtypes(include=["number"]).columns.tolist()
numeric_feature_cols = [col for col in numeric_cols if col not in protected_cols]

numeric_cols_with_missing = [
    col for col in numeric_feature_cols
    if train_imputed[col].isna().any() or (col in test_imputed.columns and test_imputed[col].isna().any())]

em_imputer = IterativeImputer(
    estimator=BayesianRidge(),
    max_iter=10,
    initial_strategy="median",
    n_nearest_features=25,
    random_state=42,
    skip_complete=True)

if numeric_cols_with_missing:
    imputer_fit_sample = train_imputed[numeric_cols_with_missing].sample(
        n=min(50_000, len(train_imputed)),
        random_state=42)

    em_imputer.fit(imputer_fit_sample)
    train_imputed[numeric_cols_with_missing] = em_imputer.transform(train_imputed[numeric_cols_with_missing])
    test_imputed[numeric_cols_with_missing] = em_imputer.transform(test_imputed[numeric_cols_with_missing])

sort_cols = [col for col in ID_COLS + [TIME_COL] if col in train_imputed.columns]

if sort_cols:
    train_imputed = train_imputed.sort_values(sort_cols).reset_index(drop=True)
    test_imputed = test_imputed.sort_values(sort_cols).reset_index(drop=True)

train_imputed = train_imputed.ffill().bfill()
test_imputed = test_imputed.ffill().bfill()
```


```python
remaining_missing_summary = pd.DataFrame({
    "dataset": ["train", "test"],
    "remaining_missing_values": [
        int(train_imputed.isna().sum().sum()),
        int(test_imputed.isna().sum().sum()),],})

display(remaining_missing_summary)
# There are no NaN values in either the train or test datasets.
```

### Temporal Features


```python
seconds_per_minute = 60
seconds_per_hour = 60 * seconds_per_minute
seconds_per_day = 24 * seconds_per_hour
seconds_per_week = 7 * seconds_per_day

for df in [train_imputed, test_imputed]:

    df["transaction_hour"] = (
        (df[TIME_COL] // seconds_per_hour) % 24
    ).astype(int)

    df["transaction_day"] = (
        df[TIME_COL] // seconds_per_day
    ).astype(int)

    df["transaction_week"] = (
        df[TIME_COL] // seconds_per_week
    ).astype(int)

    df["transaction_day_of_week"] = (
        df["transaction_day"] % 7
    ).astype(int)

    df["is_weekend"] = (
        df["transaction_day_of_week"]
        .isin([5, 6])
        .astype(int)
    )

    df["is_night_transaction"] = (
        df["transaction_hour"]
        .isin([0, 1, 2, 3, 4, 5])
        .astype(int)
    )

    df["is_business_hour"] = (
        df["transaction_hour"]
        .between(9, 18)
        .astype(int)
    )

    df["is_morning"] = (
        df["transaction_hour"]
        .between(6, 11)
        .astype(int)
    )

    df["is_afternoon"] = (
        df["transaction_hour"]
        .between(12, 17)
        .astype(int)
    )

    df["is_evening"] = (
        df["transaction_hour"]
        .between(18, 23)
        .astype(int)
    )

    df["hour_sin"] = np.sin(
        2 * np.pi * df["transaction_hour"] / 24
    )

    df["hour_cos"] = np.cos(
        2 * np.pi * df["transaction_hour"] / 24
    )

    df["dayofweek_sin"] = np.sin(
        2 * np.pi * df["transaction_day_of_week"] / 7
    )

    df["dayofweek_cos"] = np.cos(
        2 * np.pi * df["transaction_day_of_week"] / 7
    )

    df["time_since_first_transaction"] = (
        df[TIME_COL] - df[TIME_COL].min()
    )
```

### Entity Features


```python
entity_candidate_cols = [
    col for col in train_imputed.columns
    if (
        col.startswith("card")
        or col.startswith("addr")
        or col.startswith("Device")
        or col.startswith("id_")
        or col.startswith("M")
        or col in ["ProductCD", "P_emaildomain", "R_emaildomain"]
    )
]

entity_profile = pd.DataFrame({
    "entity_col": entity_candidate_cols,
    "missing_ratio": [train_imputed[col].isna().mean() for col in entity_candidate_cols],
    "unique_count": [train_imputed[col].nunique(dropna=False) for col in entity_candidate_cols],
})

selected_entity_cols = (
    entity_profile[
        (entity_profile["missing_ratio"] < 0.40)
        & (entity_profile["unique_count"] <= 5000)
    ]
    .sort_values(["missing_ratio", "unique_count"])
    .head(12)["entity_col"]
    .tolist()
)

train_entity_features = {}
test_entity_features = {}
generated_entity_features = []

for entity_col in selected_entity_cols:
    entity_stats = train_imputed.groupby(entity_col, dropna=False)["TransactionAmt"].agg(
        transaction_count="count",
        avg_amount="mean",
        median_amount="median",
        std_amount="std",
    )

    entity_stats["std_amount"] = entity_stats["std_amount"].fillna(0)

    for stat_col in entity_stats.columns:
        feature_name = f"{entity_col}_{stat_col}"
        fallback_value = entity_stats[stat_col].median()

        train_entity_features[feature_name] = (
            train_imputed[entity_col]
            .map(entity_stats[stat_col])
            .fillna(fallback_value)
        )
        test_entity_features[feature_name] = (
            test_imputed[entity_col]
            .map(entity_stats[stat_col])
            .fillna(fallback_value)
        )
        generated_entity_features.append(feature_name)

    avg_feature_name = f"{entity_col}_avg_amount"
    amount_vs_avg_feature = f"{entity_col}_amount_vs_avg"
    amount_ratio_feature = f"{entity_col}_amount_ratio"

    train_entity_features[amount_vs_avg_feature] = (
        train_imputed["TransactionAmt"] - train_entity_features[avg_feature_name]
    )
    test_entity_features[amount_vs_avg_feature] = (
        test_imputed["TransactionAmt"] - test_entity_features[avg_feature_name]
    )

    train_entity_features[amount_ratio_feature] = (
        train_imputed["TransactionAmt"] / (train_entity_features[avg_feature_name] + 1)
    )
    test_entity_features[amount_ratio_feature] = (
        test_imputed["TransactionAmt"] / (test_entity_features[avg_feature_name] + 1)
    )

    generated_entity_features.extend([amount_vs_avg_feature, amount_ratio_feature])

train_imputed = pd.concat([train_imputed, pd.DataFrame(train_entity_features, index=train_imputed.index)], axis=1)
test_imputed = pd.concat([test_imputed, pd.DataFrame(test_entity_features, index=test_imputed.index)], axis=1)
```

### Relational Features


```python
relation_candidate_cols = [
    col for col in train_imputed.columns
    if (
        col.startswith("card")
        or col.startswith("addr")
        or col.startswith("Device")
        or col.startswith("id_")
        or col.startswith("M")
        or col in ["ProductCD", "P_emaildomain", "R_emaildomain"]
    )
    and col in test_imputed.columns
]

relation_profile = pd.DataFrame({
    "column_name": relation_candidate_cols,
    "missing_ratio": [train_imputed[col].isna().mean() for col in relation_candidate_cols],
    "unique_count": [train_imputed[col].nunique(dropna=False) for col in relation_candidate_cols],
})

selected_relation_cols = (
    relation_profile[
        (relation_profile["missing_ratio"] < 0.40)
        & (relation_profile["unique_count"].between(2, 5000))
    ]
    .sort_values(["missing_ratio", "unique_count"])
    .head(12)["column_name"]
    .tolist()
)

baseline_fraud_rate = train_imputed[TARGET].mean()
min_pair_count = 20
fraud_lift_threshold = 2.0
pair_review_frames = []

for left_col, right_col in combinations(selected_relation_cols, 2):
    pair_data = train_imputed[[left_col, right_col, TARGET]].copy()
    pair_data["pair_key"] = (
        pair_data[left_col].astype("string").fillna("Missing")
        + "__"
        + pair_data[right_col].astype("string").fillna("Missing")
    )

    pair_stats = (
        pair_data
        .groupby("pair_key", observed=False)
        .agg(
            pair_count=(TARGET, "count"),
            fraud_count=(TARGET, "sum"),
            fraud_rate=(TARGET, "mean"))
        .reset_index()
    )

    pair_stats = pair_stats[pair_stats["pair_count"] >= min_pair_count].copy()

    if pair_stats.empty:
        continue

    pair_stats["left_col"] = left_col
    pair_stats["right_col"] = right_col
    pair_stats["column_pair"] = f"{left_col} + {right_col}"
    pair_stats["fraud_rate_percent"] = (pair_stats["fraud_rate"] * 100).round(2)
    pair_stats["fraud_lift"] = (pair_stats["fraud_rate"] / baseline_fraud_rate).round(2)

    pair_review_frames.append(pair_stats[[
        "left_col",
        "right_col",
        "column_pair",
        "pair_key",
        "pair_count",
        "fraud_count",
        "fraud_rate_percent",
        "fraud_lift",
    ]])

if pair_review_frames:
    relational_pair_review = pd.concat(pair_review_frames, ignore_index=True)
else:
    relational_pair_review = pd.DataFrame(columns=[
        "left_col", "right_col", "column_pair", "pair_key",
        "pair_count", "fraud_count", "fraud_rate_percent", "fraud_lift"
    ])

top_relational_combinations = (
    relational_pair_review
    .sort_values(["fraud_lift", "pair_count"], ascending=[False, False])
    .head(30)
)

selected_pair_cols = (
    relational_pair_review
    .groupby(["left_col", "right_col", "column_pair"], as_index=False)
    .agg(max_fraud_lift=("fraud_lift", "max"), max_pair_count=("pair_count", "max"))
    .sort_values(["max_fraud_lift", "max_pair_count"], ascending=[False, False])
    .head(6)[["left_col", "right_col"]]
    .itertuples(index=False, name=None)
)

train_relational_features = {}
test_relational_features = {}
relational_feature_cols = []

for left_col, right_col in selected_pair_cols:
    train_pair_key = (
        train_imputed[left_col].astype("string").fillna("Missing")
        + "__"
        + train_imputed[right_col].astype("string").fillna("Missing")
    )
    test_pair_key = (
        test_imputed[left_col].astype("string").fillna("Missing")
        + "__"
        + test_imputed[right_col].astype("string").fillna("Missing")
    )

    pair_count_feature = f"rel_{left_col}_{right_col}_count"
    pair_risk_feature = f"rel_{left_col}_{right_col}_high_risk_flag"

    pair_counts = train_pair_key.value_counts()
    risky_pair_keys = set(
        relational_pair_review[
            (relational_pair_review["left_col"] == left_col)
            & (relational_pair_review["right_col"] == right_col)
            & (relational_pair_review["fraud_lift"] >= fraud_lift_threshold)
        ]["pair_key"]
    )

    train_relational_features[pair_count_feature] = train_pair_key.map(pair_counts).fillna(0).astype(int)
    test_relational_features[pair_count_feature] = test_pair_key.map(pair_counts).fillna(0).astype(int)
    train_relational_features[pair_risk_feature] = train_pair_key.isin(risky_pair_keys).astype(int)
    test_relational_features[pair_risk_feature] = test_pair_key.isin(risky_pair_keys).astype(int)

    relational_feature_cols.extend([pair_count_feature, pair_risk_feature])

train_imputed = train_imputed.drop(columns=relational_feature_cols, errors="ignore")
test_imputed = test_imputed.drop(columns=relational_feature_cols, errors="ignore")

train_imputed = pd.concat(
    [train_imputed, pd.DataFrame(train_relational_features, index=train_imputed.index)],
    axis=1)
    
test_imputed = pd.concat(
    [test_imputed, pd.DataFrame(test_relational_features, index=test_imputed.index)],
    axis=1)
```

### Context Features


```python
amount_median = train_imputed["TransactionAmt"].median()
amount_mean = train_imputed["TransactionAmt"].mean()
amount_std = train_imputed["TransactionAmt"].std()
amount_q05 = train_imputed["TransactionAmt"].quantile(0.05)
amount_q95 = train_imputed["TransactionAmt"].quantile(0.95)

train_context_features = {
    "ctx_transaction_amount_log": np.log1p(train_imputed["TransactionAmt"]),
    "ctx_amount_vs_global_median": train_imputed["TransactionAmt"] - amount_median,
    "ctx_amount_ratio_to_global_median": train_imputed["TransactionAmt"] / (amount_median + 1),
    "ctx_amount_zscore": (train_imputed["TransactionAmt"] - amount_mean) / (amount_std + 1e-9),
    "ctx_is_high_amount": (train_imputed["TransactionAmt"] >= amount_q95).astype(int),
    "ctx_is_low_amount": (train_imputed["TransactionAmt"] <= amount_q05).astype(int),
}

test_context_features = {
    "ctx_transaction_amount_log": np.log1p(test_imputed["TransactionAmt"]),
    "ctx_amount_vs_global_median": test_imputed["TransactionAmt"] - amount_median,
    "ctx_amount_ratio_to_global_median": test_imputed["TransactionAmt"] / (amount_median + 1),
    "ctx_amount_zscore": (test_imputed["TransactionAmt"] - amount_mean) / (amount_std + 1e-9),
    "ctx_is_high_amount": (test_imputed["TransactionAmt"] >= amount_q95).astype(int),
    "ctx_is_low_amount": (test_imputed["TransactionAmt"] <= amount_q05).astype(int),
}

if "transaction_hour" in train_imputed.columns:
    hour_amount_mean = train_imputed.groupby("transaction_hour")["TransactionAmt"].mean()
    global_hour_amount_mean = train_imputed["TransactionAmt"].mean()

    train_hour_avg = train_imputed["transaction_hour"].map(hour_amount_mean).fillna(global_hour_amount_mean)
    test_hour_avg = test_imputed["transaction_hour"].map(hour_amount_mean).fillna(global_hour_amount_mean)

    train_context_features["ctx_amount_vs_hour_avg"] = train_imputed["TransactionAmt"] - train_hour_avg
    test_context_features["ctx_amount_vs_hour_avg"] = test_imputed["TransactionAmt"] - test_hour_avg

if "ProductCD" in train_imputed.columns:
    product_amount_mean = train_imputed.groupby("ProductCD")["TransactionAmt"].mean()
    global_product_amount_mean = train_imputed["TransactionAmt"].mean()

    train_product_avg = train_imputed["ProductCD"].map(product_amount_mean).fillna(global_product_amount_mean)
    test_product_avg = test_imputed["ProductCD"].map(product_amount_mean).fillna(global_product_amount_mean)

    train_context_features["ctx_amount_vs_product_avg"] = train_imputed["TransactionAmt"] - train_product_avg
    test_context_features["ctx_amount_vs_product_avg"] = test_imputed["TransactionAmt"] - test_product_avg

if "is_weekend" in train_imputed.columns:
    train_context_features["ctx_high_amount_weekend"] = (
        train_context_features["ctx_is_high_amount"] * train_imputed["is_weekend"]
    ).astype(int)
    test_context_features["ctx_high_amount_weekend"] = (
        test_context_features["ctx_is_high_amount"] * test_imputed["is_weekend"]
    ).astype(int)

context_feature_cols = list(train_context_features.keys())

train_imputed = train_imputed.drop(columns=context_feature_cols, errors="ignore")
test_imputed = test_imputed.drop(columns=context_feature_cols, errors="ignore")

train_imputed = pd.concat(
    [train_imputed, pd.DataFrame(train_context_features, index=train_imputed.index)],
    axis=1)
    
test_imputed = pd.concat(
    [test_imputed, pd.DataFrame(test_context_features, index=test_imputed.index)],
    axis=1)
```

### Isolation Forest Outlier Flag Variable


```python
outlier_feature_cols = train_imputed.select_dtypes(include=["number"]).columns.tolist()
outlier_feature_cols = [col for col in outlier_feature_cols if col not in protected_cols]

X_train_outlier = train_imputed[outlier_feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
X_test_outlier = test_imputed[outlier_feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)

outlier_feature_cols = X_train_outlier.loc[:, X_train_outlier.nunique(dropna=True) > 1].columns.tolist()
X_train_outlier = X_train_outlier[outlier_feature_cols]
X_test_outlier = X_test_outlier[outlier_feature_cols]

outlier_pipeline = Pipeline(steps=[
    ("scaler", StandardScaler()),
    ("isolation_forest", IsolationForest(
        n_estimators=100,
        contamination=0.03,
        max_samples=10000,
        random_state=42,
        n_jobs=-1))])

train_outlier_labels = outlier_pipeline.fit_predict(X_train_outlier)
test_outlier_labels = outlier_pipeline.predict(X_test_outlier)

train_imputed["is_outlier"] = (train_outlier_labels == -1).astype(int)
test_imputed["is_outlier"] = (test_outlier_labels == -1).astype(int)
```


```python
outlier_flag_summary = pd.DataFrame({
    "dataset": ["train", "test"],
    "features_used": [len(outlier_feature_cols), len(outlier_feature_cols)],
    "outlier_count": [
        int(train_imputed["is_outlier"].sum()),
        int(test_imputed["is_outlier"].sum()),],
    "outlier_ratio_percent": [
        round(train_imputed["is_outlier"].mean() * 100, 2),
        round(test_imputed["is_outlier"].mean() * 100, 2),],})

display(outlier_flag_summary)
```

### High Risk Entity Flag


```python
entity_candidate_cols = [
    col for col in train_imputed.columns
    if (
        col.startswith("card")
        or col.startswith("addr")
        or col.startswith("Device")
        or col.startswith("id_")
        or col.startswith("M")
        or col in ["ProductCD", "P_emaildomain", "R_emaildomain"]
    )
]

min_entity_transactions = 10
fraud_rate_lift_threshold = 2.0
baseline_fraud_rate = train_imputed[TARGET].mean()
high_risk_entity_rule_frames = []

for entity_col in entity_candidate_cols:
    entity_behavior = (
        train_imputed
        .groupby(entity_col, dropna=False)
        .agg(
            transaction_count=(TARGET, "count"),
            fraud_count=(TARGET, "sum"),
            fraud_rate=(TARGET, "mean"))
        .reset_index())

    entity_behavior = entity_behavior[
        (entity_behavior["transaction_count"] >= min_entity_transactions)
        & (entity_behavior["fraud_rate"] >= baseline_fraud_rate * fraud_rate_lift_threshold)
    ].copy()

    if entity_behavior.empty:
        continue

    entity_behavior["entity_column"] = entity_col
    entity_behavior["entity_value"] = entity_behavior[entity_col].astype(str)
    entity_behavior["fraud_rate_percent"] = (entity_behavior["fraud_rate"] * 100).round(2)
    entity_behavior["fraud_lift_vs_baseline"] = (entity_behavior["fraud_rate"] / baseline_fraud_rate).round(2)

    high_risk_entity_rule_frames.append(entity_behavior[[
        "entity_column",
        "entity_value",
        "transaction_count",
        "fraud_count",
        "fraud_rate_percent",
        "fraud_lift_vs_baseline",
    ]])

if high_risk_entity_rule_frames:
    high_risk_entity_rules = pd.concat(high_risk_entity_rule_frames, ignore_index=True)
else:
    high_risk_entity_rules = pd.DataFrame(columns=[
        "entity_column",
        "entity_value",
        "transaction_count",
        "fraud_count",
        "fraud_rate_percent",
        "fraud_lift_vs_baseline",
    ])

train_imputed["is_high_risk_entity"] = 0
test_imputed["is_high_risk_entity"] = 0

for entity_col, entity_values in high_risk_entity_rules.groupby("entity_column")["entity_value"]:
    risk_values = set(entity_values.astype(str))
    train_imputed.loc[
        train_imputed[entity_col].astype(str).isin(risk_values),
        "is_high_risk_entity"
    ] = 1

    if entity_col in test_imputed.columns:
        test_imputed.loc[
            test_imputed[entity_col].astype(str).isin(risk_values),
            "is_high_risk_entity"
        ] = 1

high_risk_entity_flag_summary = pd.DataFrame({
    "dataset": ["train", "test"],
    "flagged_count": [
        int(train_imputed["is_high_risk_entity"].sum()),
        int(test_imputed["is_high_risk_entity"].sum()),
    ],
    "flagged_ratio_percent": [
        round(train_imputed["is_high_risk_entity"].mean() * 100, 2),
        round(test_imputed["is_high_risk_entity"].mean() * 100, 2),
    ],
})
```

## Summary

This script cleans the fraud transaction data and creates new features for fraud detection.

It removes columns with too many missing values, fills the remaining missing values, and prepares the train and test datasets.

Then it creates new signals from the data, such as transaction time, night transactions, weekend transactions, unusual amounts, risky entities, suspicious relationships, and outlier transactions.

At the end, the dataset becomes more useful for fraud detection models and rule-based fraud checks.
