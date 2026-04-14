import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from xgboost import XGBClassifier, XGBRegressor
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    mean_absolute_error,
    mean_squared_error,
)

from config import FEATURE_COLUMNS, MERGED_DATASET_PATH, MODEL_PATH, REG_MODEL_PATH


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    required_cols = ["Close", "Volume"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required raw columns for feature engineering: {missing}")

    # Existing features
    df["MA5"] = df["Close"].rolling(5).mean()
    df["MA20"] = df["Close"].rolling(20).mean()
    df["MA60"] = df["Close"].rolling(60).mean()

    delta = df["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss.replace(0, pd.NA)
    df["RSI14"] = 100 - (100 / (1 + rs))

    df["Return"] = df["Close"].pct_change()
    df["Vol_Change"] = df["Volume"].pct_change()
    df["Volatility20"] = df["Return"].rolling(20).std()

    df["MA20_slope"] = df["MA20"].diff()
    df["MA60_slope"] = df["MA60"].diff()
    df["Price_Trend_5d"] = df["Close"].pct_change(5)
    df["Price_Trend_10d"] = df["Close"].pct_change(10)
    df["RSI_Trend"] = df["RSI14"].diff()

    if "IndustryScore" not in df.columns:
        df["IndustryScore"] = 1.0

    # New features
    df["Return_1d"] = df["Close"].pct_change(1)
    df["Return_3d"] = df["Close"].pct_change(3)
    df["Return_5d"] = df["Close"].pct_change(5)
    df["Return_10d"] = df["Close"].pct_change(10)

    daily_ret = df["Close"].pct_change()
    df["Vol_5d"] = daily_ret.rolling(5).std()
    df["Vol_10d"] = daily_ret.rolling(10).std()

    df["MA20_gap"] = (df["Close"] - df["MA20"]) / df["MA20"].replace(0, pd.NA)
    df["MA60_gap"] = (df["Close"] - df["MA60"]) / df["MA60"].replace(0, pd.NA)

    volume_ma20 = df["Volume"].rolling(20).mean()
    df["Volume_ratio"] = df["Volume"] / volume_ma20.replace(0, pd.NA)

    return df


def train_model():
    start_time = time.time()

    dataset_path = Path(MERGED_DATASET_PATH)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {MERGED_DATASET_PATH}")

    print("Loading dataset...")
    df = pd.read_csv(dataset_path)
    print(f"Original shape: {df.shape}")

    required_base = ["Close", "Volume", "Target", "Target_Return"]
    missing_base = [c for c in required_base if c not in df.columns]
    if missing_base:
        raise ValueError(f"Missing required base columns in source dataset: {missing_base}")

    # Build all features
    df = add_features(df)

    required_cols = FEATURE_COLUMNS + ["Target", "Target_Return", "Close", "Volume"]
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns after feature engineering: {missing_cols}")

    df = df.replace([np.inf, -np.inf], np.nan)

    # Force numeric dtypes
    for col in FEATURE_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["Target"] = pd.to_numeric(df["Target"], errors="coerce")
    df["Target_Return"] = pd.to_numeric(df["Target_Return"], errors="coerce")
    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
    df["Volume"] = pd.to_numeric(df["Volume"], errors="coerce")

    df = df.dropna(subset=FEATURE_COLUMNS + ["Target", "Target_Return", "Close", "Volume"]).reset_index(drop=True)

    if df.empty:
        raise ValueError("Dataset became empty after numeric conversion and dropna().")

    df["Target_Return"] = df["Target_Return"].clip(-0.1, 0.1)

    print(f"Clean shape: {df.shape}")

    # Save cleaned dataset for evaluation
    output_dir = Path("data")
    output_dir.mkdir(parents=True, exist_ok=True)
    cleaned_path = output_dir / "cleaned_dataset.csv"
    df.to_csv(cleaned_path, index=False)
    print(f"Saved cleaned dataset -> {cleaned_path}")

    # Time-based split
    split = int(len(df) * 0.8)
    train_df = df.iloc[:split].copy()
    test_df = df.iloc[split:].copy()

    if train_df.empty or test_df.empty:
        raise ValueError("Train/test split failed. Check dataset size.")

    print(f"Train rows: {len(train_df)}")
    print(f"Test rows: {len(test_df)}")

    X_train = train_df[FEATURE_COLUMNS].copy()
    X_test = test_df[FEATURE_COLUMNS].copy()

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
    print(f"Classifier Accuracy: {acc:.4f}")
    print(classification_report(y_test_cls, y_pred_cls, zero_division=0))

    y_pred_reg = reg.predict(X_test)
    mae = mean_absolute_error(y_test_reg, y_pred_reg)
    rmse = np.sqrt(mean_squared_error(y_test_reg, y_pred_reg))
    print(f"Return MAE: {mae:.6f}")
    print(f"Return RMSE: {rmse:.6f}")

    Path(MODEL_PATH).parent.mkdir(parents=True, exist_ok=True)
    Path(REG_MODEL_PATH).parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(clf, MODEL_PATH)
    joblib.dump(reg, REG_MODEL_PATH)

    print(f"Saved classifier -> {MODEL_PATH}")
    print(f"Saved regressor -> {REG_MODEL_PATH}")
    print(f"Total time: {time.time() - start_time:.2f} sec")


if __name__ == "__main__":
    train_model()