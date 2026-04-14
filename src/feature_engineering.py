import numpy as np
import pandas as pd

from config import (
    PRICE_DIR,
    PROCESSED_DIR,
    MERGED_DATASET_PATH,
    INDUSTRY_SCORE_MAP,
    ensure_directories,
)
from stock_list import build_stock_list


def calculate_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def build_features_for_one_stock(
    price_df: pd.DataFrame,
    meta_row: pd.Series,
    include_targets: bool = True
) -> pd.DataFrame:
    if price_df is None or not isinstance(price_df, pd.DataFrame) or price_df.empty:
        return pd.DataFrame()

    required_input_cols = ["Date", "Close", "Volume"]
    missing_input_cols = [col for col in required_input_cols if col not in price_df.columns]
    if missing_input_cols:
        return pd.DataFrame()

    df = price_df.copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"]).copy()

    df["MA5"] = df["Close"].rolling(5).mean()
    df["MA20"] = df["Close"].rolling(20).mean()
    df["MA60"] = df["Close"].rolling(60).mean()

    df["RSI14"] = calculate_rsi(df["Close"], 14)
    df["Return"] = df["Close"].pct_change()
    df["Vol_Change"] = df["Volume"].pct_change()
    df["Volatility20"] = df["Return"].rolling(20).std()

    df["MA20_slope"] = df["MA20"].diff()
    df["MA60_slope"] = df["MA60"].diff()
    df["Price_Trend_5d"] = df["Close"].pct_change(5)
    df["Price_Trend_10d"] = df["Close"].pct_change(10)
    df["RSI_Trend"] = df["RSI14"].diff()

    industry = meta_row["industry"]
    industry_score = INDUSTRY_SCORE_MAP.get(industry, 0.5)
    df["IndustryScore"] = industry_score

    if include_targets:
        # next-day direction
        df["Target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)

        # next-day return, clipped to reduce extreme outliers
        df["Target_Return"] = (
            (df["Close"].shift(-1) - df["Close"]) / df["Close"]
        ).clip(-0.10, 0.10)

    df["code"] = str(meta_row["code"])
    df["name"] = meta_row["name"]
    df["market"] = meta_row["market"]
    df["industry"] = industry
    df["ticker"] = meta_row["ticker"]

    df = df.replace([np.inf, -np.inf], np.nan)

    required_cols = [
        "Date",
        "Close",
        "MA5",
        "MA20",
        "MA60",
        "RSI14",
        "Return",
        "Vol_Change",
        "Volatility20",
        "IndustryScore",
        "MA20_slope",
        "MA60_slope",
        "Price_Trend_5d",
        "Price_Trend_10d",
        "RSI_Trend",
    ]

    if include_targets:
        required_cols.extend(["Target", "Target_Return"])

    missing_required_cols = [col for col in required_cols if col not in df.columns]
    if missing_required_cols:
        return pd.DataFrame()

    df = df.dropna(subset=required_cols).reset_index(drop=True)
    return df


def build_all_features():
    ensure_directories()
    stock_df = build_stock_list()
    all_frames = []

    for _, row in stock_df.iterrows():
        price_path = PRICE_DIR / f'{row["code"]}.csv'

        if not price_path.exists():
            print(f"[SKIP] Missing price file: {price_path.name}")
            continue

        try:
            price_df = pd.read_csv(price_path)
            feature_df = build_features_for_one_stock(price_df, row, include_targets=True)

            if feature_df.empty:
                print(f"[SKIP] Empty features for {row['code']}")
                continue

            save_path = PROCESSED_DIR / f'features_{row["code"]}.csv'
            feature_df.to_csv(save_path, index=False, encoding="utf-8-sig")

            all_frames.append(feature_df)
            print(f"[OK] Built features for {row['code']}")

        except Exception as e:
            print(f"[ERROR] feature build {row['code']}: {e}")

    if not all_frames:
        print("[WARN] No feature data generated.")
        return None

    merged_df = pd.concat(all_frames, ignore_index=True)
    merged_df.to_csv(MERGED_DATASET_PATH, index=False, encoding="utf-8-sig")

    print(f"\nSaved merged dataset to: {MERGED_DATASET_PATH}")
    return merged_df


if __name__ == "__main__":
    build_all_features()