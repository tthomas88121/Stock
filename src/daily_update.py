from pathlib import Path

import joblib
import pandas as pd
import yfinance as yf

from config import (
    BASE_DIR,
    PRICE_DIR,
    FEATURE_COLUMNS,
    MODEL_PATH,
    REG_MODEL_PATH,
    STOCK_LIST_PATH,
    get_stock_list_path,
)
from stock_list import build_stock_list
from feature_engineering import build_features_for_one_stock


OUTPUT_DIR = BASE_DIR / "outputs"
TOP_CANDIDATES_PATH = OUTPUT_DIR / "top_candidates.csv"
DAILY_ALL_PATH = OUTPUT_DIR / "daily_all_predictions.csv"
FAILED_PATH = OUTPUT_DIR / "failed_symbols.csv"


def load_or_build_stock_list() -> pd.DataFrame:
    stock_list_path = get_stock_list_path()

    if stock_list_path.exists():
        try:
            df = pd.read_csv(stock_list_path)
            if not df.empty:
                df["code"] = df["code"].astype(str)
                return df
        except Exception:
            pass

    if STOCK_LIST_PATH.exists():
        try:
            df = pd.read_csv(STOCK_LIST_PATH)
            if not df.empty:
                df["code"] = df["code"].astype(str)
                return df
        except Exception:
            pass

    df = build_stock_list()
    if not df.empty:
        df["code"] = df["code"].astype(str)
    return df


def download_recent_data(ticker: str) -> pd.DataFrame:
    try:
        df = yf.download(
            ticker,
            period="3mo",
            auto_adjust=True,
            progress=False,
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

    return save_df


def main(top_n: int = 10):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    stock_df = load_or_build_stock_list()
    if stock_df.empty:
        raise ValueError("stock_list is empty. Please check stock_list.csv or build_stock_list().")

    clf = joblib.load(MODEL_PATH) if Path(MODEL_PATH).exists() else None
    reg = joblib.load(REG_MODEL_PATH) if Path(REG_MODEL_PATH).exists() else None

    if clf is None or reg is None:
        raise FileNotFoundError("Missing trained classifier or regressor model.")

    results = []
    failed = []

    total = len(stock_df)
    for idx, (_, row) in enumerate(stock_df.iterrows(), start=1):
        code = str(row["code"])
        ticker = row["ticker"]
        name = row.get("name", "")
        market = row.get("market", "")
        industry = row.get("industry", "")

        print(f"[{idx}/{total}] {code} {name}")

        recent_df = download_recent_data(ticker)
        if recent_df.empty:
            failed.append({"code": code, "ticker": ticker, "reason": "download_failed"})
            continue

        full_df = merge_and_save_price_data(code, recent_df)

        try:
            feature_df = build_features_for_one_stock(full_df, row, include_targets=False)
        except TypeError:
            # backward compatibility if old function version is still used
            feature_df = build_features_for_one_stock(full_df, row)

        if feature_df.empty:
            failed.append({"code": code, "ticker": ticker, "reason": "feature_failed"})
            continue

        latest = feature_df.iloc[-1]

        try:
            X = pd.DataFrame([latest[FEATURE_COLUMNS]], columns=FEATURE_COLUMNS)

            prob_up = float(clf.predict_proba(X)[0][1])
            pred_return = float(reg.predict(X)[0])
            pred_price = float(latest["Close"] * (1 + pred_return))
        except Exception as e:
            failed.append(
                {
                    "code": code,
                    "ticker": ticker,
                    "reason": f"predict_failed: {str(e)}",
                }
            )
            continue

        results.append(
            {
                "Date": latest["Date"],
                "code": code,
                "name": name,
                "market": market,
                "industry": industry,
                "ticker": ticker,
                "Close": float(latest["Close"]),
                "prob_up": prob_up,
                "pred_return": pred_return,
                "pred_price": pred_price,
            }
        )

    result_df = pd.DataFrame(results)
    failed_df = pd.DataFrame(failed)

    if not result_df.empty:
        result_df = result_df.sort_values(by="prob_up", ascending=False).reset_index(drop=True)
        result_df.to_csv(DAILY_ALL_PATH, index=False, encoding="utf-8-sig")
        result_df.head(top_n).to_csv(TOP_CANDIDATES_PATH, index=False, encoding="utf-8-sig")

    if not failed_df.empty:
        failed_df.to_csv(FAILED_PATH, index=False, encoding="utf-8-sig")
    elif FAILED_PATH.exists():
        FAILED_PATH.unlink()

    print("Done.")
    print(f"Saved: {DAILY_ALL_PATH}")
    print(f"Saved: {TOP_CANDIDATES_PATH}")
    print(f"Failed count: {len(failed_df)}")


if __name__ == "__main__":
    main(top_n=10)