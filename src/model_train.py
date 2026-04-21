import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from xgboost import XGBClassifier, XGBRegressor
from sklearn.metrics import accuracy_score, precision_score, mean_absolute_error, mean_squared_error

from config import FEATURE_COLUMNS, MERGED_DATASET_PATH, MODEL_PATH, REG_MODEL_PATH


TRAIN_FEATURES_PATH = Path("data") / "training_features.json"
MODEL_METRICS_PATH = Path("data") / "model_metrics.json"


def normalize_base_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    rename_map = {}

    if "target" in df.columns:
        rename_map["target"] = "Target"
    if "target_return" in df.columns:
        rename_map["target_return"] = "Target_Return"
    if "close" in df.columns:
        rename_map["close"] = "Close"

    if rename_map:
        df = df.rename(columns=rename_map)

    return df


def train_model():
    start_time = time.time()

    dataset_path = Path(MERGED_DATASET_PATH)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {MERGED_DATASET_PATH}")

    print("Loading dataset...")
    df = pd.read_csv(dataset_path)
    print(f"Original shape: {df.shape}")

    df = normalize_base_columns(df)

    required_base = ["Close", "Target", "Target_Return"]
    missing_base = [c for c in required_base if c not in df.columns]
    if missing_base:
        print("Current columns:", df.columns.tolist())
        raise ValueError(f"Missing required base columns: {missing_base}")

    # Use only the features that actually exist in merged_dataset.csv
    active_features = [col for col in FEATURE_COLUMNS if col in df.columns]
    missing_features = [col for col in FEATURE_COLUMNS if col not in df.columns]

    print("Active training features:", active_features)
    print("Missing features skipped:", missing_features)

    if len(active_features) < 5:
        raise ValueError("Too few usable features found in merged_dataset.csv.")

    df = df.replace([np.inf, -np.inf], np.nan)

    for col in active_features:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["Target"] = pd.to_numeric(df["Target"], errors="coerce")
    df["Target_Return"] = pd.to_numeric(df["Target_Return"], errors="coerce")
    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")

    df = df.dropna(subset=active_features + ["Target", "Target_Return", "Close"]).reset_index(drop=True)

    if df.empty:
        raise ValueError("Dataset became empty after numeric conversion and dropna().")

    df["Target_Return"] = df["Target_Return"].clip(-0.1, 0.1)

    print(f"Clean shape: {df.shape}")

    output_dir = Path("data")
    output_dir.mkdir(parents=True, exist_ok=True)

    cleaned_path = output_dir / "cleaned_dataset.csv"
    df.to_csv(cleaned_path, index=False)
    print(f"Saved cleaned dataset -> {cleaned_path}")

    split = int(len(df) * 0.8)
    train_df = df.iloc[:split].copy()
    test_df = df.iloc[split:].copy()

    if train_df.empty or test_df.empty:
        raise ValueError("Train/test split failed. Check dataset size.")

    print(f"Train rows: {len(train_df)}")
    print(f"Test rows: {len(test_df)}")

    X_train = train_df[active_features].copy()
    X_test = test_df[active_features].copy()

    y_train_cls = train_df["Target"].astype(int).copy()
    y_test_cls = test_df["Target"].astype(int).copy()

    y_train_reg = train_df["Target_Return"].copy()
    y_test_reg = test_df["Target_Return"].copy()

    print("Training XGBoost classifier...")
    clf = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        eval_metric="logloss",
    )
    clf.fit(X_train, y_train_cls)

    print("Training XGBoost regressor...")
    reg = XGBRegressor(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
    )
    reg.fit(X_train, y_train_reg)

    print("\n===== Training Evaluation =====")

    y_pred_cls = clf.predict(X_test)
    acc = accuracy_score(y_test_cls, y_pred_cls)
    prec = precision_score(y_test_cls, y_pred_cls, zero_division=0)
    print(f"Classifier Accuracy: {acc:.4f}")
    print(f"Classifier Precision: {prec:.4f}")

    y_pred_reg = reg.predict(X_test)
    mae = mean_absolute_error(y_test_reg, y_pred_reg)
    rmse = np.sqrt(mean_squared_error(y_test_reg, y_pred_reg))
    print(f"Return MAE: {mae:.6f}")
    print(f"Return RMSE: {rmse:.6f}")

    Path(MODEL_PATH).parent.mkdir(parents=True, exist_ok=True)
    Path(REG_MODEL_PATH).parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(clf, MODEL_PATH)
    joblib.dump(reg, REG_MODEL_PATH)

    with open(TRAIN_FEATURES_PATH, "w", encoding="utf-8") as f:
        json.dump(active_features, f, indent=2)

    metrics = {
        "train_time_sec": round(time.time() - start_time, 2),
        "classifier_accuracy": float(acc),
        "classifier_precision": float(prec),
        "regression_mae": float(mae),
        "regression_rmse": float(rmse),
        "rows_used": int(len(df)),
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "trained_at": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
        "active_features": active_features,
        "missing_features_skipped": missing_features,
    }

    with open(MODEL_METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"Saved classifier -> {MODEL_PATH}")
    print(f"Saved regressor -> {REG_MODEL_PATH}")
    print(f"Saved training feature list -> {TRAIN_FEATURES_PATH}")
    print(f"Saved metrics -> {MODEL_METRICS_PATH}")
    print(f"Total time: {time.time() - start_time:.2f} sec")


if __name__ == "__main__":
    train_model()