from pathlib import Path

import pandas as pd
import yfinance as yf

from config import PRICE_DIR, get_stock_list_path, ensure_directories


MERGED_DATASET_PATH = Path("data") / "merged_dataset.csv"


def normalize_code(value) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).replace(".0", "").strip()


def normalize_ticker(ticker: str) -> str:
    ticker = str(ticker).strip()

    if ticker.isdigit():
        return f"{ticker}.TW"

    if not (ticker.endswith(".TW") or ticker.endswith(".TWO")):
        if ticker.replace(".", "").isdigit():
            return f"{ticker}.TW"

    return ticker


def load_stock_list() -> pd.DataFrame:
    stock_list_path = get_stock_list_path()
    if not stock_list_path.exists():
        raise FileNotFoundError(f"stock_list.csv not found: {stock_list_path}")

    df = pd.read_csv(stock_list_path)
    if df.empty:
        raise ValueError("stock_list.csv is empty.")

    df["code"] = df["code"].apply(normalize_code)
    return df


def download_recent_data(ticker: str) -> pd.DataFrame:
    try:
        ticker = normalize_ticker(ticker)

        df = yf.download(
            ticker,
            period="1y",
            interval="1d",
            auto_adjust=True,
            progress=False,
            threads=False,
        )

        if df is None or df.empty:
            return pd.DataFrame()

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df = df.reset_index()

        required = ["Date", "Close", "Volume"]
        if not all(col in df.columns for col in required):
            return pd.DataFrame()

        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df.dropna(subset=["Date"]).copy()

        return df
    except Exception as e:
        print(f"[ERROR] download {ticker}: {e}")
        return pd.DataFrame()


def merge_and_save_price_data(code: str, recent_df: pd.DataFrame) -> pd.DataFrame:
    code = normalize_code(code)

    PRICE_DIR.mkdir(parents=True, exist_ok=True)
    path = PRICE_DIR / f"{code}.csv"

    recent_df = recent_df.copy()
    recent_df["Date"] = pd.to_datetime(recent_df["Date"], errors="coerce")
    recent_df = recent_df.dropna(subset=["Date"]).copy()

    if path.exists():
        try:
            old_df = pd.read_csv(path)

            if "Date" not in old_df.columns:
                old_df = pd.DataFrame()

            if not old_df.empty:
                old_df["Date"] = pd.to_datetime(old_df["Date"], errors="coerce")
                old_df = old_df.dropna(subset=["Date"]).copy()

            merged = pd.concat([old_df, recent_df], ignore_index=True)
        except Exception:
            merged = recent_df.copy()
    else:
        merged = recent_df.copy()

    merged["Date"] = pd.to_datetime(merged["Date"], errors="coerce")
    merged = merged.dropna(subset=["Date"]).copy()

    merged = (
        merged.drop_duplicates(subset=["Date"])
        .sort_values("Date")
        .reset_index(drop=True)
    )

    save_df = merged.copy()
    save_df["Date"] = save_df["Date"].dt.strftime("%Y-%m-%d")
    save_df.to_csv(path, index=False, encoding="utf-8-sig")

    return merged


def calculate_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()

    rs = avg_gain / avg_loss.replace(0, pd.NA)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def build_features_for_one_stock(
    price_df: pd.DataFrame,
    meta_row: pd.Series,
    include_targets: bool = True,
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

    industry_score = 1.0
    df["IndustryScore"] = industry_score

    # keep these extra features so training and prediction stay compatible
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

    if include_targets:
        df["Target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)
        df["Target_Return"] = (
            (df["Close"].shift(-1) - df["Close"]) / df["Close"]
        ).clip(-0.10, 0.10)

    df["code"] = str(meta_row["code"])
    df["name"] = meta_row.get("name", "")
    df["market"] = meta_row.get("market", "")
    df["industry"] = meta_row.get("industry", "")
    df["ticker"] = meta_row.get("ticker", "")

    df = df.replace([float("inf"), float("-inf")], pd.NA)

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
        "Return_1d",
        "Return_3d",
        "Return_5d",
        "Return_10d",
        "Vol_5d",
        "Vol_10d",
        "MA20_gap",
        "MA60_gap",
        "Volume_ratio",
    ]

    if include_targets:
        required_cols.extend(["Target", "Target_Return"])

    df = df.dropna(subset=required_cols).reset_index(drop=True)
    return df


def build_merged_dataset(stock_df: pd.DataFrame) -> pd.DataFrame | None:
    all_frames = []

    for _, row in stock_df.iterrows():
        code = normalize_code(row["code"])
        price_path = PRICE_DIR / f"{code}.csv"

        if not price_path.exists():
            print(f"[SKIP] Missing price file: {price_path.name}")
            continue

        try:
            price_df = pd.read_csv(price_path)
            feature_df = build_features_for_one_stock(price_df, row, include_targets=True)

            if feature_df.empty:
                print(f"[SKIP] Empty features for {code}")
                continue

            all_frames.append(feature_df)
            print(f"[OK] Built features for {code}")

        except Exception as e:
            print(f"[ERROR] feature build {code}: {e}")

    if not all_frames:
        print("[WARN] No merged dataset generated.")
        return None

    merged_df = pd.concat(all_frames, ignore_index=True)
    MERGED_DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)
    merged_df.to_csv(MERGED_DATASET_PATH, index=False, encoding="utf-8-sig")

    print(f"Saved merged dataset -> {MERGED_DATASET_PATH}")
    return merged_df


def main():
    ensure_directories()
    stock_df = load_stock_list()

    total = len(stock_df)
    print(f"Updating price data for {total} stocks...")

    for idx, (_, row) in enumerate(stock_df.iterrows(), start=1):
        code = normalize_code(row["code"])
        ticker = row["ticker"]
        name = row.get("name", "")

        print(f"[{idx}/{total}] Updating {code} {name}")

        recent_df = download_recent_data(ticker)
        if recent_df.empty:
            print(f"[SKIP] Download failed: {ticker}")
            continue

        try:
            merge_and_save_price_data(code, recent_df)
            print(f"[OK] Saved price data for {code}")
        except Exception as e:
            print(f"[ERROR] save price {code}: {e}")

    print("\nRebuilding merged dataset...")
    build_merged_dataset(stock_df)
    print("Daily update finished.")


if __name__ == "__main__":
    main()