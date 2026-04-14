from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

from config import FEATURE_COLUMNS, REG_MODEL_PATH


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


def summarize_strategy(name: str, returns: np.ndarray, cost_per_trade: float):
    print(f"\n===== {name} =====")
    print(f"Trades: {len(returns)}")

    if len(returns) == 0:
        print("Win Rate: N/A")
        print("Avg Return: N/A")
        print("Total Return: N/A")
        print("Net Avg Return: N/A")
        print("Net Total Return: N/A")
        return

    net_returns = returns - cost_per_trade

    print(f"Win Rate: {(returns > 0).mean():.4f}")
    print(f"Avg Return: {returns.mean():.6f}")
    print(f"Total Return: {returns.sum():.6f}")
    print(f"Net Avg Return: {net_returns.mean():.6f}")
    print(f"Net Total Return: {net_returns.sum():.6f}")


def evaluate_thresholds(
    y_pred_return: np.ndarray,
    y_true_return: np.ndarray,
    test_df: pd.DataFrame,
    thresholds: list[float],
    cost_per_trade: float,
):
    print("\n===== Threshold Comparison =====")
    print(
        "Threshold | Trades | WinRate | AvgReturn | TotalReturn | "
        "NetAvgReturn | NetTotalReturn"
    )

    close_vals = test_df["Close"].values
    ma20_vals = test_df["MA20"].values
    ma60_vals = test_df["MA60"].values

    for threshold in thresholds:
        mask = y_pred_return > threshold
        returns = y_true_return[mask]

        if len(returns) == 0:
            print(
                f"{threshold:8.3f} | {0:6d} | {'N/A':>7} | {'N/A':>9} | {'N/A':>11} | "
                f"{'N/A':>12} | {'N/A':>14}"
            )
            continue

        net_returns = returns - cost_per_trade

        print(
            f"{threshold:8.3f} | "
            f"{len(returns):6d} | "
            f"{(returns > 0).mean():7.4f} | "
            f"{returns.mean():9.6f} | "
            f"{returns.sum():11.6f} | "
            f"{net_returns.mean():12.6f} | "
            f"{net_returns.sum():14.6f}"
        )

    print("\n===== Threshold + Trend Filter Comparison =====")
    print(
        "Threshold | Trades | WinRate | AvgReturn | TotalReturn | "
        "NetAvgReturn | NetTotalReturn"
    )

    trend_filter = (close_vals > ma20_vals) & (ma20_vals > ma60_vals)

    for threshold in thresholds:
        mask = (y_pred_return > threshold) & trend_filter
        returns = y_true_return[mask]

        if len(returns) == 0:
            print(
                f"{threshold:8.3f} | {0:6d} | {'N/A':>7} | {'N/A':>9} | {'N/A':>11} | "
                f"{'N/A':>12} | {'N/A':>14}"
            )
            continue

        net_returns = returns - cost_per_trade

        print(
            f"{threshold:8.3f} | "
            f"{len(returns):6d} | "
            f"{(returns > 0).mean():7.4f} | "
            f"{returns.mean():9.6f} | "
            f"{returns.sum():11.6f} | "
            f"{net_returns.mean():12.6f} | "
            f"{net_returns.sum():14.6f}"
        )


def evaluate_model():
    cleaned_dataset_path = Path("data/cleaned_dataset.csv")

    if not cleaned_dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {cleaned_dataset_path}. Run model_train.py first."
        )

    if not Path(REG_MODEL_PATH).exists():
        raise FileNotFoundError(
            f"Regressor model not found: {REG_MODEL_PATH}. Run model_train.py first."
        )

    print("Loading cleaned dataset...")
    df = pd.read_csv(cleaned_dataset_path)

    required_base_cols = ["Close", "Volume", "Target", "Target_Return"]
    missing_base = [c for c in required_base_cols if c not in df.columns]
    if missing_base:
        raise ValueError(f"Missing required base columns in dataset: {missing_base}")

    df = add_features(df)

    required_cols = FEATURE_COLUMNS + ["Target", "Target_Return", "Close", "Volume", "MA20", "MA60"]
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing columns after feature engineering: {missing_cols}")

    df = df.replace([np.inf, -np.inf], np.nan)

    for col in FEATURE_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["Target"] = pd.to_numeric(df["Target"], errors="coerce")
    df["Target_Return"] = pd.to_numeric(df["Target_Return"], errors="coerce")
    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
    df["Volume"] = pd.to_numeric(df["Volume"], errors="coerce")
    df["MA20"] = pd.to_numeric(df["MA20"], errors="coerce")
    df["MA60"] = pd.to_numeric(df["MA60"], errors="coerce")

    df = df.dropna(
        subset=FEATURE_COLUMNS + ["Target", "Target_Return", "Close", "Volume", "MA20", "MA60"]
    ).reset_index(drop=True)

    if df.empty:
        raise ValueError("Dataset became empty after feature engineering and dropna().")

    split_idx = int(len(df) * 0.8)
    test_df = df.iloc[split_idx:].copy()

    if test_df.empty:
        raise ValueError("Test set is empty after split.")

    X_test = test_df[FEATURE_COLUMNS].copy()
    y_true_return = test_df["Target_Return"].values
    current_price = test_df["Close"].values
    actual_direction = test_df["Target"].astype(int).values

    print(f"Test rows: {len(test_df)}")

    print("Loading regressor...")
    reg = joblib.load(REG_MODEL_PATH)

    print("Predicting returns...")
    y_pred_return = reg.predict(X_test)

    pred_future_price = current_price * (1 + y_pred_return)
    true_future_price = current_price * (1 + y_true_return)

    mae_return = mean_absolute_error(y_true_return, y_pred_return)
    rmse_return = np.sqrt(mean_squared_error(y_true_return, y_pred_return))

    mae_price = mean_absolute_error(true_future_price, pred_future_price)
    rmse_price = np.sqrt(mean_squared_error(true_future_price, pred_future_price))

    pred_direction = (y_pred_return > 0).astype(int)
    direction_acc = (pred_direction == actual_direction).mean()

    print("\n===== Regression Metrics =====")
    print(f"Return MAE: {mae_return:.6f}")
    print(f"Return RMSE: {rmse_return:.6f}")
    print(f"Future Price MAE: {mae_price:.4f}")
    print(f"Future Price RMSE: {rmse_price:.4f}")
    print(f"Direction Accuracy: {direction_acc:.4f}")

    cost_per_trade = 0.0015  # 0.15%

    # Base strategies
    trade_mask = y_pred_return > 0
    trade_returns = y_true_return[trade_mask]

    strong_trade_mask = y_pred_return > 0.02
    strong_trade_returns = y_true_return[strong_trade_mask]

    trend_mask = (
        (y_pred_return > 0.02)
        & (test_df["Close"].values > test_df["MA20"].values)
        & (test_df["MA20"].values > test_df["MA60"].values)
    )
    trend_trade_returns = y_true_return[trend_mask]

    summarize_strategy(
        "Trading Simulation (Predicted Return > 0)",
        trade_returns,
        cost_per_trade,
    )
    summarize_strategy(
        "Filtered Trading (Predicted Return > 2%)",
        strong_trade_returns,
        cost_per_trade,
    )
    summarize_strategy(
        "Filtered Trading + Trend Filter",
        trend_trade_returns,
        cost_per_trade,
    )

    thresholds = [0.01, 0.015, 0.02, 0.025, 0.03, 0.04, 0.05]
    evaluate_thresholds(
        y_pred_return=y_pred_return,
        y_true_return=y_true_return,
        test_df=test_df,
        thresholds=thresholds,
        cost_per_trade=cost_per_trade,
    )


if __name__ == "__main__":
    evaluate_model()