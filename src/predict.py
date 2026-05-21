import json
from pathlib import Path
from datetime import timedelta

import joblib
import numpy as np
import pandas as pd
import yfinance as yf

from config import (
    BASE_DIR,
    PRICE_DIR,
    MODEL_PATH,
    REG_MODEL_PATH,
    STOCK_LIST_PATH,
    get_stock_list_path,
)

from stock_list import build_stock_list
from save_predictions import save_predictions


OUTPUT_DIR = BASE_DIR / "outputs"

TOP_CANDIDATES_PATH = OUTPUT_DIR / "top_candidates.csv"
DAILY_ALL_PATH = OUTPUT_DIR / "daily_all_predictions.csv"
FAILED_PATH = OUTPUT_DIR / "failed_symbols.csv"

TRAIN_FEATURES_PATH = Path("data") / "training_features.json"


def normalize_code(value) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).replace(".0", "").strip()


def load_or_build_stock_list() -> pd.DataFrame:
    stock_list_path = get_stock_list_path()

    if stock_list_path.exists():
        try:
            df = pd.read_csv(stock_list_path)

            if not df.empty:
                df["code"] = df["code"].apply(normalize_code)
                return df

        except Exception:
            pass

    if STOCK_LIST_PATH.exists():
        try:
            df = pd.read_csv(STOCK_LIST_PATH)

            if not df.empty:
                df["code"] = df["code"].apply(normalize_code)
                return df

        except Exception:
            pass

    df = build_stock_list()

    if not df.empty:
        df["code"] = df["code"].apply(normalize_code)

    return df


def normalize_ticker(ticker: str) -> str:
    ticker = str(ticker).strip()

    if ticker.isdigit():
        return f"{ticker}.TW"

    if not (ticker.endswith(".TW") or ticker.endswith(".TWO")):
        if ticker.replace(".", "").isdigit():
            return f"{ticker}.TW"

    return ticker


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

    except Exception:
        return pd.DataFrame()


def merge_and_save_price_data(code: str, recent_df: pd.DataFrame) -> pd.DataFrame:
    code = normalize_code(code)

    PRICE_DIR.mkdir(parents=True, exist_ok=True)

    path = PRICE_DIR / f"{code}.csv"

    recent_df = recent_df.copy()

    recent_df["Date"] = pd.to_datetime(
        recent_df["Date"],
        errors="coerce",
    )

    recent_df = recent_df.dropna(subset=["Date"]).copy()

    if path.exists():
        try:
            old_df = pd.read_csv(path)

            if "Date" not in old_df.columns:
                old_df = pd.DataFrame()

            if not old_df.empty:
                old_df["Date"] = pd.to_datetime(
                    old_df["Date"],
                    errors="coerce",
                )

                old_df = old_df.dropna(subset=["Date"]).copy()

            merged = pd.concat(
                [old_df, recent_df],
                ignore_index=True,
            )

        except Exception:
            merged = recent_df.copy()

    else:
        merged = recent_df.copy()

    merged["Date"] = pd.to_datetime(
        merged["Date"],
        errors="coerce",
    )

    merged = merged.dropna(subset=["Date"]).copy()

    merged = (
        merged.drop_duplicates(subset=["Date"])
        .sort_values("Date")
        .reset_index(drop=True)
    )

    save_df = merged.copy()

    save_df["Date"] = save_df["Date"].dt.strftime("%Y-%m-%d")

    save_df.to_csv(
        path,
        index=False,
        encoding="utf-8-sig",
    )

    return merged


def build_features_for_one_stock(
    price_df: pd.DataFrame,
    stock_row: pd.Series,
    active_features: list[str],
) -> pd.DataFrame:

    if price_df is None or price_df.empty:
        return pd.DataFrame()

    df = price_df.copy()

    required = ["Date", "Close", "Volume"]

    if not all(col in df.columns for col in required):
        return pd.DataFrame()

    industry_score = stock_row.get(
        "IndustryScore",
        stock_row.get("industry_score", 1.0),
    )

    try:
        industry_score = float(industry_score)

    except Exception:
        industry_score = 1.0

    # =========================
    # Moving Averages
    # =========================
    df["MA5"] = df["Close"].rolling(5).mean()
    df["MA20"] = df["Close"].rolling(20).mean()
    df["MA60"] = df["Close"].rolling(60).mean()

    # =========================
    # RSI
    # =========================
    delta = df["Close"].diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()

    rs = avg_gain / avg_loss.replace(0, pd.NA)

    df["RSI14"] = 100 - (100 / (1 + rs))

    # =========================
    # Return Features
    # =========================
    df["Return"] = df["Close"].pct_change()

    df["Vol_Change"] = df["Volume"].pct_change()

    df["Volatility20"] = (
        df["Return"]
        .rolling(20)
        .std()
    )

    # =========================
    # Trend Features
    # =========================
    df["MA20_slope"] = df["MA20"].diff()
    df["MA60_slope"] = df["MA60"].diff()

    df["Price_Trend_5d"] = df["Close"].pct_change(5)
    df["Price_Trend_10d"] = df["Close"].pct_change(10)

    df["RSI_Trend"] = df["RSI14"].diff()

    # =========================
    # Industry Score
    # =========================
    df["IndustryScore"] = industry_score

    # =========================
    # Multi-timeframe Returns
    # =========================
    df["Return_1d"] = df["Close"].pct_change(1)
    df["Return_3d"] = df["Close"].pct_change(3)
    df["Return_5d"] = df["Close"].pct_change(5)
    df["Return_10d"] = df["Close"].pct_change(10)

    # =========================
    # Volatility
    # =========================
    daily_ret = df["Close"].pct_change()

    df["Vol_5d"] = daily_ret.rolling(5).std()
    df["Vol_10d"] = daily_ret.rolling(10).std()

    # =========================
    # Gap Features
    # =========================
    df["MA20_gap"] = (
        (df["Close"] - df["MA20"])
        / df["MA20"].replace(0, pd.NA)
    )

    df["MA60_gap"] = (
        (df["Close"] - df["MA60"])
        / df["MA60"].replace(0, pd.NA)
    )

    # =========================
    # Volume Features
    # =========================
    volume_ma20 = df["Volume"].rolling(20).mean()

    df["Volume_ratio"] = (
        df["Volume"]
        / volume_ma20.replace(0, pd.NA)
    )

    # =========================
    # NEW MOMENTUM FEATURES
    # =========================
    df["Momentum_20d"] = (
        df["Close"]
        / df["Close"].shift(20)
        - 1
    )

    df["Momentum_60d"] = (
        df["Close"]
        / df["Close"].shift(60)
        - 1
    )

    df["Volume_Momentum"] = (
        df["Volume"]
        / df["Volume"].rolling(20).mean().replace(0, pd.NA)
    )

    # =========================
    # Breakout Features
    # =========================
    rolling_high_20 = df["Close"].rolling(20).max()
    rolling_high_60 = df["Close"].rolling(60).max()

    df["High_20d"] = (
        df["Close"]
        / rolling_high_20.replace(0, pd.NA)
    )

    df["High_60d"] = (
        df["Close"]
        / rolling_high_60.replace(0, pd.NA)
    )

    df["Breakout_20d"] = (
        df["Close"] >= rolling_high_20
    ).astype(int)

    df["Breakout_60d"] = (
        df["Close"] >= rolling_high_60
    ).astype(int)

    # =========================
    # Relative Strength
    # =========================
    df["MA20_MA60_Ratio"] = (
        df["MA20"]
        / df["MA60"].replace(0, pd.NA)
    )

    df["Close_MA20_Ratio"] = (
        df["Close"]
        / df["MA20"].replace(0, pd.NA)
    )

    df["Close_MA60_Ratio"] = (
        df["Close"]
        / df["MA60"].replace(0, pd.NA)
    )

    df = df.replace([np.inf, -np.inf], np.nan)

    for col in active_features:
        if col not in df.columns:
            df[col] = np.nan

        df[col] = pd.to_numeric(
            df[col],
            errors="coerce",
        )

    df["Close"] = pd.to_numeric(
        df["Close"],
        errors="coerce",
    )

    df["Volume"] = pd.to_numeric(
        df["Volume"],
        errors="coerce",
    )

    df = df.dropna(
        subset=active_features + ["Close"]
    ).reset_index(drop=True)

    return df


def trading_signal_label(
    prob_up: float,
    pred_return: float,
    close_price: float,
    ma20: float,
    ma60: float,
) -> str:

    trend_up = close_price > ma20 and ma20 > ma60

    if pred_return >= 0.05 and trend_up and prob_up >= 0.60:
        return "STRONG BUY"

    if pred_return >= 0.03 and trend_up:
        return "BUY"

    if pred_return >= 0.015 and close_price > ma20:
        return "WATCH"

    return "NO BUY"


def setup_quality_label(pred_return: float) -> str:
    if pred_return >= 0.05:
        return "High-conviction setup"

    if pred_return >= 0.03:
        return "Valid buy setup"

    if pred_return >= 0.015:
        return "Watchlist candidate"

    return "Weak setup"


def get_next_trading_day(date_value):
    d = pd.to_datetime(date_value).date()

    next_day = d + timedelta(days=1)

    while next_day.weekday() >= 5:
        next_day += timedelta(days=1)

    return next_day


def main(top_n: int = 10):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not TRAIN_FEATURES_PATH.exists():
        raise FileNotFoundError(
            f"Missing {TRAIN_FEATURES_PATH}. Run model_train.py first."
        )

    with open(TRAIN_FEATURES_PATH, "r", encoding="utf-8") as f:
        active_features = json.load(f)

    stock_df = load_or_build_stock_list()

    if stock_df.empty:
        raise ValueError(
            "stock_list is empty. Please check stock_list.csv or build_stock_list()."
        )

    clf = (
        joblib.load(MODEL_PATH)
        if Path(MODEL_PATH).exists()
        else None
    )

    reg = (
        joblib.load(REG_MODEL_PATH)
        if Path(REG_MODEL_PATH).exists()
        else None
    )

    if clf is None or reg is None:
        raise FileNotFoundError(
            "Missing trained classifier or regressor model."
        )

    results = []
    failed = []
    prediction_rows = []

    total = len(stock_df)

    for idx, (_, row) in enumerate(
        stock_df.iterrows(),
        start=1,
    ):

        code = normalize_code(row["code"])

        ticker = normalize_ticker(row["ticker"])

        name = row.get("name", "")
        market = row.get("market", "")
        industry = row.get("industry", "")

        print(f"[{idx}/{total}] {code} {name}")

        recent_df = download_recent_data(ticker)

        if recent_df.empty:
            failed.append(
                {
                    "code": code,
                    "ticker": ticker,
                    "reason": "download_failed",
                }
            )
            continue

        full_df = merge_and_save_price_data(
            code,
            recent_df,
        )

        feature_df = build_features_for_one_stock(
            full_df,
            row,
            active_features,
        )

        if feature_df.empty:
            failed.append(
                {
                    "code": code,
                    "ticker": ticker,
                    "reason": "feature_failed",
                }
            )
            continue

        latest = feature_df.iloc[-1]

        try:
            X = pd.DataFrame(
                [latest[active_features]],
                columns=active_features,
            )

            for col in active_features:
                X[col] = pd.to_numeric(
                    X[col],
                    errors="coerce",
                )

            X = X.astype(float)

            # =========================
            # Classification
            # =========================
            prob_proba = clf.predict_proba(X)

            if prob_proba.shape[1] >= 2:
                prob_up = float(prob_proba[0][1])
            else:
                prob_up = float(prob_proba[0][0])

            # =========================
            # Return Prediction
            # =========================
            raw_pred_return = float(reg.predict(X)[0])

            # wider range for momentum stocks
            pred_return = max(
                min(raw_pred_return, 0.20),
                -0.20,
            )

            # momentum boost logic
            if (
                latest["Breakout_20d"] == 1
                and latest["Momentum_20d"] > 0.10
                and prob_up > 0.60
            ):
                pred_return *= 1.35

            today_close = float(latest["Close"])

            pred_price = float(
                today_close * (1 + pred_return)
            )

            signal = trading_signal_label(
                prob_up=prob_up,
                pred_return=pred_return,
                close_price=today_close,
                ma20=float(latest["MA20"]),
                ma60=float(latest["MA60"]),
            )

            setup_quality = setup_quality_label(
                pred_return
            )

        except Exception as e:
            failed.append(
                {
                    "code": code,
                    "ticker": ticker,
                    "reason": f"predict_failed: {str(e)}",
                }
            )
            continue

        prediction_date = pd.to_datetime(
            latest["Date"]
        ).date()

        target_date = get_next_trading_day(
            prediction_date
        )

        pred_dir = (
            "UP"
            if pred_price > today_close
            else "DOWN"
        )

        results.append(
            {
                "Date": prediction_date.strftime("%Y-%m-%d"),
                "code": normalize_code(code),
                "name": name,
                "market": market,
                "industry": industry,
                "ticker": ticker,
                "Close": today_close,
                "MA20": float(latest["MA20"]),
                "MA60": float(latest["MA60"]),
                "RSI14": float(latest["RSI14"]),
                "prob_up": prob_up,
                "pred_return": pred_return,
                "pred_price": pred_price,
                "signal": signal,
                "setup_quality": setup_quality,
            }
        )

        prediction_rows.append(
            {
                "Prediction Date": prediction_date.strftime("%Y-%m-%d"),
                "Target Date": target_date.strftime("%Y-%m-%d"),
                "Symbol": ticker,
                "Today Close": today_close,
                "Pred Close": pred_price,
                "Pred Dir": pred_dir,
            }
        )

    result_df = pd.DataFrame(results)

    failed_df = pd.DataFrame(failed)

    if not result_df.empty:

        result_df["code"] = (
            result_df["code"]
            .apply(normalize_code)
        )

        result_df["signal_rank"] = (
            result_df["signal"]
            .map(
                {
                    "STRONG BUY": 3,
                    "BUY": 2,
                    "WATCH": 1,
                    "NO BUY": 0,
                }
            )
            .fillna(0)
        )

        result_df = result_df.sort_values(
            by=[
                "signal_rank",
                "pred_return",
                "prob_up",
            ],
            ascending=[
                False,
                False,
                False,
            ],
        ).reset_index(drop=True)

        result_df = result_df.drop(
            columns=["signal_rank"]
        )

        result_df.to_csv(
            DAILY_ALL_PATH,
            index=False,
            encoding="utf-8-sig",
        )

        result_df.head(top_n).to_csv(
            TOP_CANDIDATES_PATH,
            index=False,
            encoding="utf-8-sig",
        )

        save_predictions(prediction_rows)

        print(
            f"Saved prediction rows: {len(prediction_rows)}"
        )

    else:
        print("No predictions generated.")

    if not failed_df.empty:

        failed_df["code"] = (
            failed_df["code"]
            .apply(normalize_code)
        )

        failed_df.to_csv(
            FAILED_PATH,
            index=False,
            encoding="utf-8-sig",
        )

        print(f"Failed count: {len(failed_df)}")

        print(
            f"Failed reasons sample:\n{failed_df.head(20)}"
        )

    elif FAILED_PATH.exists():
        FAILED_PATH.unlink()

    print("Done.")
    print(f"Saved: {DAILY_ALL_PATH}")
    print(f"Saved: {TOP_CANDIDATES_PATH}")


if __name__ == "__main__":
    main(top_n=10)