# Exploratory Data Analysis (EDA)


```python
import pandas as pd
import numpy as np
import missingno as msno
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import seaborn as sns
from scipy.stats import chi2_contingency, ttest_ind, mannwhitneyu
pd.options.future.infer_string = False   
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
```

### Data


```python
train_transaction = pd.read_csv("../data/raw/ieee-fraud-detection/train_transaction.csv")
train_identity = pd.read_csv("../data/raw/ieee-fraud-detection/train_identity.csv")

test_transaction = pd.read_csv("../data/raw/ieee-fraud-detection/test_transaction.csv")
test_identity = pd.read_csv("../data/raw/ieee-fraud-detection/test_identity.csv")
```


```python
common_columns = train_transaction.columns.intersection(train_identity.columns).tolist()

train = train_transaction.merge(train_identity, on=common_columns, how="left")

test = test_transaction.merge(test_identity, on=common_columns, how="left")
```

### Missing Value Analysis


```python
missing_summary = pd.DataFrame({
    "missing_count": train.isnull().sum(),
    "missing_ratio": train.isnull().mean() * 100
    }).sort_values("missing_ratio", ascending=False)

# 20% NaN ratio threshold is used as a practical warning level. 
threshold = 20

columns_over_threshold = missing_summary[missing_summary["missing_ratio"] > threshold]

num_columns_over_threshold = columns_over_threshold.shape[0]
total_columns = train.shape[1]
ratio_columns_over_threshold = (num_columns_over_threshold / total_columns) * 100

print(f"Total number of columns: {total_columns}")
print(f"Number of columns with missing ratio > {threshold}%: {num_columns_over_threshold}")
print(f"Ratio of columns with missing ratio > {threshold}%: {ratio_columns_over_threshold:.2f}%")

# A high proportion of variables have substantial missing values, indicating a serious data quality issue before modeling.
```


```python
# NaNs were analyzed on the training set to check whether missing values follow a systematic pattern.
# If missing records show different target behavior, missingness may not be completely random.

# MCAR: Missing Completely At Random
# MAR: Missing At Random
# MNAR: Missing Not At Random
# Note: MNAR cannot be proven directly with this test (This analysis only checks whether missingness is related to the target).

# diff_threshold defines the minimum target rate difference used to flag missingness as potentially systematic.

def missing_pattern_analysis(df, target_col=None, threshold=0.20, diff_threshold=0.01):
    missing_summary = pd.DataFrame({
        "missing_count": df.isnull().sum(),
        "missing_ratio": df.isnull().mean()
    }).sort_values("missing_ratio", ascending=False)

    high_missing_cols = missing_summary[missing_summary["missing_ratio"] > threshold]

    if target_col is not None:
        target_missing_relation = []

        for col in df.columns:
            if col != target_col and df[col].isnull().sum() > 0:
                missing_flag = df[col].isnull().astype(int)

                target_mean_when_missing = df.loc[missing_flag == 1, target_col].mean()
                target_mean_when_not_missing = df.loc[missing_flag == 0, target_col].mean()

                difference = target_mean_when_missing - target_mean_when_not_missing

                if abs(difference) < diff_threshold:
                    missingness_type = "Likely MCAR"
                else:
                    missingness_type = "Possibly MAR/MNAR"

                target_missing_relation.append({
                    "variable": col,
                    "missing_ratio": df[col].isnull().mean(),
                    "target_mean_when_missing": target_mean_when_missing,
                    "target_mean_when_not_missing": target_mean_when_not_missing,
                    "difference": difference,
                    "missingness_type": missingness_type})

        target_missing_relation = pd.DataFrame(target_missing_relation)
        target_missing_relation = target_missing_relation.sort_values("difference", ascending=False)

        missingness_counts = target_missing_relation["missingness_type"].value_counts()

        print("\nMissingness Type Counts:")
        print(missingness_counts)

        print("\nColumns by Missingness Type:")

        for m_type in target_missing_relation["missingness_type"].unique():
            cols = target_missing_relation.loc[
                target_missing_relation["missingness_type"] == m_type,
                "variable"].tolist()

            print(f"\n{m_type}: {len(cols)} columns")
            print(cols)

        return missing_summary, target_missing_relation

    return missing_summary
```


```python
missing_summary, target_missing_relation = missing_pattern_analysis(train, target_col="isFraud", threshold=0.20, diff_threshold=0.01)

# Most missing variables were classified as Possibly MAR/MNAR, suggesting that missingness is likely systematic and may carry predictive information rather than being completely random.
```


```python
threshold = 0.2 # 20% NaN ratio threshold is used as a practical warning level.

columns_over_threshold = (missing_summary[missing_summary["missing_ratio"] > threshold].index.tolist())

columns_over_threshold = [col for col in columns_over_threshold if col != "isFraud"]

missing_flags = train[columns_over_threshold].isnull().astype("int8")

target = train["isFraud"].astype("int8")

target_corr = missing_flags.corrwith(target)

top_n = 15 # Number of variables with the strongest missingness-target correlation to display for readability.

top_missing_cols_by_target = (target_corr.abs().sort_values(ascending=False).head(top_n).index.tolist())

plot_data = missing_flags[top_missing_cols_by_target].copy()
plot_data["isFraud"] = target

plot_corr_matrix = plot_data.corr()
```


```python
plot_corr_matrix_percent = plot_corr_matrix * 100

annot_labels = plot_corr_matrix_percent.round(1).astype(str) + "%"

plt.figure(figsize=(13, 10))

ax = sns.heatmap(
    plot_corr_matrix_percent,
    cmap="coolwarm",
    center=0,
    vmin=-100,
    vmax=100,
    linewidths=0.5,
    linecolor="white",
    square=True,
    annot=annot_labels,
    fmt="",
    annot_kws={"size": 8.5},
    cbar_kws={"label": "Correlation (%)"})


cbar = ax.collections[0].colorbar
cbar.ax.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{x:.0f}%"))

ax.set_title(
    "Percentage Correlation Heatmap of Missingness Indicators and isFraud",
    fontsize=18,
    fontweight="bold",
    pad=22)

plt.xticks(rotation=90, fontsize=8)
plt.yticks(rotation=0, fontsize=8)

plt.tight_layout()
plt.show()

#T his heatmap shows both the correlations among missingness indicators and their correlation with the target variable isFraud. 
# The isFraud row/column is the key part; stronger positive or negative values there indicate that missingness may be related to fraud behavior.
```

### Automatic Column Type Detection


```python
column_types = []

for col in train.columns:
    unique_count = train[col].nunique(dropna=False)
    unique_ratio = unique_count / len(train)
    dtype = train[col].dtype

    if col == "isFraud":
        col_type = "target"

    elif col == "TransactionID" or unique_ratio > 0.95:
        col_type = "id_like"

    elif unique_count == 2:
        col_type = "binary"

    elif pd.api.types.is_numeric_dtype(train[col]):
        col_type = "numeric"

    elif pd.api.types.is_object_dtype(train[col]):
        if unique_ratio > 0.80:
            col_type = "high_cardinality_categorical"
        else:
            col_type = "categorical"

    else:
        col_type = "other"

    column_types.append({
        "column_name": col,
        "dtype": str(dtype),
        "unique_count": unique_count,
        "unique_ratio": round(unique_ratio * 100, 2),
        "detected_type": col_type
    })

column_type_summary = pd.DataFrame(column_types)

display(
    column_type_summary["detected_type"]
    .value_counts()
    .reset_index()
    .rename(columns={"index": "detected_type", "detected_type": "column_count"}))
```

### Data Quality Analysis


```python
total_rows, total_columns = train.shape
total_cells = total_rows * total_columns

missing_values = train.isnull().sum()
total_missing_values = missing_values.sum()
missing_ratio = total_missing_values / total_cells * 100

columns_with_missing = (missing_values > 0).sum()
columns_with_missing_ratio = columns_with_missing / total_columns * 100

complete_columns = (missing_values == 0).sum()
complete_columns_ratio = complete_columns / total_columns * 100

empty_columns = (missing_values == total_rows).sum()
empty_columns_ratio = empty_columns / total_columns * 100

duplicate_rows = train.duplicated().sum()
duplicate_ratio = duplicate_rows / total_rows * 100

constant_columns = (train.nunique(dropna=False) == 1).sum()
constant_columns_ratio = constant_columns / total_columns * 100

print("Dataset Shape")
print(f"Total rows: {total_rows}")
print(f"Total columns: {total_columns}")
print(f"Total cells: {total_cells}")

print("\nMissing Value Quality")
print(f"Total missing values: {total_missing_values} ({missing_ratio:.2f}% of all cells)")
print(f"Columns with missing values: {columns_with_missing} ({columns_with_missing_ratio:.2f}% of columns)")
print(f"Complete columns: {complete_columns} ({complete_columns_ratio:.2f}% of columns)")
print(f"Completely empty columns: {empty_columns} ({empty_columns_ratio:.2f}% of columns)")

print("\nDuplicate Quality")
print(f"Duplicate rows: {duplicate_rows} ({duplicate_ratio:.2f}% of rows)")

print("\nColumn Consistency Quality")
print(f"Constant columns: {constant_columns} ({constant_columns_ratio:.2f}% of columns)")
```


```python
# Missing Value Analysis by Data Type

dtype_missing_summary = pd.DataFrame({
    "data_type": train.dtypes.astype(str),
    "missing_count": train.isnull().sum(),
    "missing_ratio": train.isnull().mean() * 100})

dtype_missing_summary = (
    dtype_missing_summary
    .groupby("data_type")
    .agg(
        column_count=("data_type", "count"),
        total_missing_values=("missing_count", "sum"),
        average_missing_ratio=("missing_ratio", "mean"),
        max_missing_ratio=("missing_ratio", "max"))
    .round(2)
    .sort_values("average_missing_ratio", ascending=False))

dtype_missing_summary
```

### Column Distribution Analysis


```python
numeric_cols = train.select_dtypes(include=["number"]).columns
categorical_cols = train.select_dtypes(include=["object", "category", "bool"]).columns

print("Column Distribution Summary")
print(f"Total numeric columns: {len(numeric_cols)}")
print(f"Total categorical columns: {len(categorical_cols)}")

numeric_distribution = train[numeric_cols].describe().T

numeric_distribution["missing_ratio"] = train[numeric_cols].isnull().mean() * 100
numeric_distribution["zero_ratio"] = (train[numeric_cols] == 0).mean() * 100
numeric_distribution["unique_count"] = train[numeric_cols].nunique()

numeric_distribution = numeric_distribution.round()

numeric_distribution.head(10)
```


```python
high_zero_columns = numeric_distribution.sort_values("zero_ratio", ascending=False)   # Columns with high zero ratio

high_zero_columns.head(10)
```


```python
categorical_distribution = pd.DataFrame({
    "column_name": categorical_cols,
    "unique_count": train[categorical_cols].nunique().values,
    "missing_ratio": (train[categorical_cols].isnull().mean() * 100).round(2).values})

categorical_distribution.head(10)
```


```python
def categorical_distribution_analysis(df, categorical_cols):
    records = []

    n_rows = len(df)

    for col in categorical_cols:
        series = df[col].astype("object")

        missing_ratio = series.isna().mean()
        unique_count = series.nunique(dropna=False)
        unique_ratio = unique_count / n_rows

        value_counts = series.value_counts(dropna=False, normalize=True)

        top_category = value_counts.index[0] if len(value_counts) > 0 else None
        top_category_ratio = value_counts.iloc[0] if len(value_counts) > 0 else np.nan

        rare_category_count = (series.value_counts(dropna=False) < 10).sum()

        records.append({
            "column_name": col,
            "missing_ratio": round(missing_ratio * 100, 2),
            "unique_count": unique_count,
            "unique_ratio": round(unique_ratio * 100, 2),
            "top_category_ratio": round(top_category_ratio * 100, 2),
            "rare_category_count": rare_category_count
        })

    categorical_summary = pd.DataFrame(records)

    return categorical_summary


categorical_distribution_summary = categorical_distribution_analysis(
    train,
    categorical_cols=categorical_cols)

display(
    categorical_distribution_summary
    .sort_values("unique_ratio", ascending=False)
    .head(10))
```

### High Cardinality Field Detection


```python
low_threshold = 0.10           # Columns with unique value ratio lower than 10% are considered low-cardinality fields
high_threshold = 0.80          # Columns with unique value ratio greater than 80% are considered high-cardinality fields
continuous_unique_threshold = 50            
id_like_columns = {"TransactionID", "TransactionDT"}

unique_counts = train.nunique(dropna=False)
unique_ratios = unique_counts / len(train)

cardinality_summary = pd.DataFrame({
    "column_name": train.columns,
    "data_type": train.dtypes.astype(str).values,
    "unique_count": unique_counts.values,
    "unique_ratio": unique_ratios.values})

cardinality_summary["unique_ratio_percent"] = (cardinality_summary["unique_ratio"] * 100).round(2)
cardinality_summary["is_numeric"] = train.dtypes.apply(lambda dtype: pd.api.types.is_numeric_dtype(dtype)).values
cardinality_summary["is_categorical"] = train.dtypes.apply(
    lambda dtype: pd.api.types.is_object_dtype(dtype)
    or pd.api.types.is_string_dtype(dtype)
    or pd.api.types.is_categorical_dtype(dtype)
    or pd.api.types.is_bool_dtype(dtype)).values

cardinality_summary["is_id_like"] = cardinality_summary["column_name"].isin(id_like_columns)

cardinality_summary["cardinality_level"] = pd.cut(
    cardinality_summary["unique_ratio"],
    bins=[-0.01, low_threshold, high_threshold, 1.00],
    labels=["Low Cardinality", "Medium Cardinality", "High Cardinality"])

cardinality_summary["column_group"] = "low_or_medium_cardinality"
cardinality_summary.loc[
    cardinality_summary["is_id_like"],
    "column_group"] = "id_like_high_cardinality"
cardinality_summary.loc[
    (~cardinality_summary["is_id_like"])
    & cardinality_summary["is_categorical"]
    & (cardinality_summary["unique_ratio"] >= low_threshold),
    "column_group"] = "categorical_high_cardinality"
cardinality_summary.loc[
    (~cardinality_summary["is_id_like"])
    & cardinality_summary["is_numeric"]
    & (cardinality_summary["unique_count"] >= continuous_unique_threshold),
    "column_group"] = "numeric_continuous"     # numeric_continuous: numeric columns with many unique values

column_group_counts = cardinality_summary["column_group"].value_counts().reset_index()
column_group_counts.columns = ["column_group", "column_count"]

print("Group Summary:")
print(column_group_counts)

cardinality_summary.sort_values(["column_group", "unique_ratio_percent"], ascending=[True, False])
```

## Summary of Findings

The dataset has a high dimensional structure and includes several data quality challenges. Missing values are a major issue, and some variables show sparse or highly unique value patterns. These findings indicate that careful preprocessing, feature selection, and appropriate missing value handling are necessary before anomaly detection and modeling.
