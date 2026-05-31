from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.api.services.scoring_service import _DISPOSABLE, _SIGNAL_WEIGHTS 

FEATURE_ORDER: tuple[str, ...] = tuple(_SIGNAL_WEIGHTS.keys())
INTERIM = ROOT / "data" / "interim"
DEFAULT_OUT = ROOT / "models" / "anomaly_model.joblib"


def _disposable_flag(domain: pd.Series) -> pd.Series:
    return domain.fillna("unknown").str.lower().isin(_DISPOSABLE).astype(int)


def features_from_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)

    hour = df["transaction_hour"] if "transaction_hour" in df.columns else (
        (df["TransactionDT"] // 3600) % 24 if "TransactionDT" in df.columns else 12
    )
    out["is_night_transaction"] = ((hour < 6) | (hour >= 22)).astype(int)

    if "is_new_account" in df.columns:
        out["is_new_account"] = df["is_new_account"].astype(int)
    elif "time_since_first_transaction" in df.columns:
        out["is_new_account"] = (df["time_since_first_transaction"] < 7 * 86400).astype(int)
    else:
        out["is_new_account"] = 0

    amt = df["TransactionAmt"] if "TransactionAmt" in df.columns else 0.0
    out["is_high_amount"] = (amt >= 500).astype(int)

    if "has_velocity_spike" in df.columns:
        out["has_velocity_spike"] = df["has_velocity_spike"].astype(int)
    elif "num_txn_last_1h" in df.columns:
        out["has_velocity_spike"] = (df["num_txn_last_1h"] >= 5).astype(int)
    else:
        out["has_velocity_spike"] = 0

    if "P_emaildomain" in df.columns:
        out["uses_disposable_email"] = _disposable_flag(df["P_emaildomain"])
    else:
        out["uses_disposable_email"] = 0

    if "uses_credit_card" in df.columns:
        out["uses_credit_card"] = df["uses_credit_card"].astype(int)
    elif "card4" in df.columns:
        out["uses_credit_card"] = df["card4"].fillna("").str.lower().str.contains("credit").astype(int)
    else:
        out["uses_credit_card"] = 0

    if "country_mismatch" in df.columns:
        out["country_mismatch"] = df["country_mismatch"].astype(int)
    elif {"addr1", "P_emaildomain"}.issubset(df.columns):
        out["country_mismatch"] = (df["addr1"].isna() | (df["addr1"] == -1)).astype(int)
    else:
        out["country_mismatch"] = 0

    if "low_device_trust" in df.columns:
        out["low_device_trust"] = df["low_device_trust"].astype(int)
    elif "entity_trusted_score" in df.columns:
        out["low_device_trust"] = (df["entity_trusted_score"] < 0.45).astype(int)
    else:
        out["low_device_trust"] = 0

    return out[list(FEATURE_ORDER)]


def _load_pipeline_frame() -> pd.DataFrame | None:
    for name in ("train_anomaly_scored.pkl", "train_feature_engineered.pkl", "train_score_fused.pkl"):
        path = INTERIM / name
        if path.exists():
            print(f"Loading {path.relative_to(ROOT)}")
            return pd.read_pickle(path)
    return None


def _synthetic_frame(n: int = 8000, seed: int = 42) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(seed)
    rows = []
    labels = []
    for _ in range(n):
        signals = {
            "is_night_transaction":  int(rng.random() < 0.18),
            "is_new_account":        int(rng.random() < 0.12),
            "is_high_amount":        int(rng.random() < 0.15),
            "has_velocity_spike":    int(rng.random() < 0.10),
            "uses_disposable_email": int(rng.random() < 0.08),
            "uses_credit_card":      int(rng.random() < 0.45),
            "country_mismatch":      int(rng.random() < 0.14),
            "low_device_trust":      int(rng.random() < 0.20),
        }
        score = sum(signals[k] * _SIGNAL_WEIGHTS[k] for k in FEATURE_ORDER)
        noise = rng.normal(0, 0.05)
        prob = min(max(score + noise, 0.01), 0.99)
        label = int(rng.random() < prob)
        rows.append(signals)
        labels.append(label)
    X = pd.DataFrame(rows, columns=list(FEATURE_ORDER))
    y = pd.Series(labels, name="isFraud")
    return X, y


def train_and_export(output: Path, sample_rows: int | None) -> None:
    df = _load_pipeline_frame()
    if df is not None and "isFraud" in df.columns:
        X = features_from_pipeline(df)
        y = df["isFraud"].astype(int)
        source = "pipeline"
        if sample_rows and len(X) > sample_rows:
            idx = y[y == 1].index.tolist()
            idx += y[y == 0].sample(sample_rows - len(idx), random_state=42).index.tolist()
            X, y = X.loc[idx], y.loc[idx]
    else:
        print("Pipeline pickles not found — training on synthetic API-signal data.")
        X, y = _synthetic_frame()
        source = "synthetic"

    model = LGBMClassifier(
        n_estimators=120,
        max_depth=4,
        learning_rate=0.08,
        class_weight="balanced",
        random_state=42,
        verbose=-1,
    )
    model.fit(X, y)

    output.parent.mkdir(parents=True, exist_ok=True)
    artifact = {
        "model": model,
        "feature_order": list(FEATURE_ORDER),
        "source": source,
        "name": "pipeline-lightgbm-api-signals",
    }
    joblib.dump(artifact, output)
    print(f"Saved {output.relative_to(ROOT)}  (source={source}, rows={len(X):,})")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export API scoring model artifact")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--sample-rows", type=int, default=50_000, help="Max training rows from pipeline")
    args = parser.parse_args()
    train_and_export(args.output, args.sample_rows)


if __name__ == "__main__":
    main()