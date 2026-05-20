from pathlib import Path
import pandas as pd
import yfinance as yf
from config import DATA_DIR


PREDICTION_HISTORY_PATH = DATA_DIR / "predictions_history.csv"
EVALUATION_PATH = DATA_DIR / "prediction_evaluation.csv"


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Support both old lowercase format and new title-case format.
    """
    rename_map = {
        "prediction_date": "Prediction Date",
        "target_date": "Target Date",
        "symbol": "Symbol",
        "today_close": "Today Close",
        "predicted_close": "Pred Close",
        "predicted_return": "Pred Return",
        "predicted_direction": "Pred Dir",
        "actual_date": "Actual Date",
        "actual_close": "Actual Close",
        "actual_direction": "Actual Dir",
        "direction_correct": "Direction Correct",
        "abs_error": "Abs Error",
        "pct_error": "Pct Error",
    }

    df = df.rename(columns=rename_map)
    return df


def get_actual_close(symbol, target_date):
    """
    Get actual close for the target date from Yahoo Finance.

    If target date is a weekend/holiday and no row exists, this returns None.
    That row will be evaluated later after real market data exists.
    """
    target_dt = pd.to_datetime(target_date)
    start_dt = target_dt - pd.Timedelta(days=2)
    end_dt = target_dt + pd.Timedelta(days=3)

    try:
        df = yf.download(
            symbol,
            start=start_dt.strftime("%Y-%m-%d"),
            end=end_dt.strftime("%Y-%m-%d"),
            auto_adjust=True,
            progress=False,
            threads=False,
        )

        if df is None or df.empty:
            print(f"[SKIP] No Yahoo data for {symbol}")
            return None, None

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df = df.reset_index()

        if "Date" not in df.columns or "Close" not in df.columns:
            print(f"[SKIP] Missing Date/Close columns for {symbol}")
            return None, None

        df["Date"] = pd.to_datetime(df["Date"], errors="coerce").dt.date
        df = df.dropna(subset=["Date"]).copy()

        target_date_obj = target_dt.date()
        row = df[df["Date"] == target_date_obj]

        if row.empty:
            print(f"[SKIP] {symbol} has no data for target date {target_date}")
            return None, None

        actual_close = float(row["Close"].iloc[0])
        actual_date = str(row["Date"].iloc[0])

        return actual_close, actual_date

    except Exception as e:
        print(f"[SKIP] Error fetching {symbol}: {e}")
        return None, None


def evaluate_predictions():
    if not PREDICTION_HISTORY_PATH.exists():
        print("No predictions history found.")
        return

    predictions_df = pd.read_csv(PREDICTION_HISTORY_PATH)
    predictions_df = normalize_columns(predictions_df)

    if predictions_df.empty:
        print("Predictions history is empty.")
        return

    required_cols = [
        "Prediction Date",
        "Target Date",
        "Symbol",
        "Today Close",
        "Pred Close",
        "Pred Dir",
    ]

    missing_cols = [c for c in required_cols if c not in predictions_df.columns]
    if missing_cols:
        print(f"Missing columns in predictions_history.csv: {missing_cols}")
        return

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if EVALUATION_PATH.exists():
        eval_df_old = pd.read_csv(EVALUATION_PATH)
        eval_df_old = normalize_columns(eval_df_old)
    else:
        eval_df_old = pd.DataFrame()

    existing_keys = set()
    if not eval_df_old.empty:
        if "Symbol" in eval_df_old.columns and "Target Date" in eval_df_old.columns:
            eval_df_old["Target Date"] = pd.to_datetime(eval_df_old["Target Date"]).dt.strftime("%Y-%m-%d")
            existing_keys = set(zip(eval_df_old["Symbol"], eval_df_old["Target Date"]))

    predictions_df["Prediction Date"] = pd.to_datetime(predictions_df["Prediction Date"]).dt.strftime("%Y-%m-%d")
    predictions_df["Target Date"] = pd.to_datetime(predictions_df["Target Date"]).dt.strftime("%Y-%m-%d")

    pending_df = predictions_df[
        ~predictions_df.apply(
            lambda r: (r["Symbol"], r["Target Date"]) in existing_keys,
            axis=1,
        )
    ].copy()

    if pending_df.empty:
        print("No pending predictions to evaluate.")
        return

    print(f"Pending rows to evaluate: {len(pending_df)}")

    new_eval_rows = []

    for idx, (_, row) in enumerate(pending_df.iterrows(), start=1):
        symbol = row["Symbol"]
        target_date = row["Target Date"]

        print(f"[{idx}/{len(pending_df)}] Evaluating {symbol} for {target_date}")

        actual_close, actual_date = get_actual_close(symbol, target_date)

        if actual_close is None:
            continue

        today_close = float(row["Today Close"])
        pred_close = float(row["Pred Close"])

        pred_return = (pred_close - today_close) / today_close if today_close != 0 else 0.0
        actual_return = (actual_close - today_close) / today_close if today_close != 0 else 0.0

        pred_dir = row.get("Pred Dir")
        if pd.isna(pred_dir) or pred_dir == "":
            pred_dir = "UP" if pred_close >= today_close else "DOWN"

        actual_dir = "UP" if actual_close >= today_close else "DOWN"
        direction_correct = int(pred_dir == actual_dir)

        abs_error = abs(pred_close - actual_close)
        pct_error = abs_error / actual_close if actual_close != 0 else None

        new_eval_rows.append(
            {
                "Prediction Date": row["Prediction Date"],
                "Target Date": target_date,
                "Actual Date": actual_date,
                "Symbol": symbol,
                "Today Close": today_close,
                "Pred Close": pred_close,
                "Actual Close": actual_close,
                "Pred Return": pred_return,
                "Actual Return": actual_return,
                "Pred Dir": pred_dir,
                "Actual Dir": actual_dir,
                "Direction Correct": direction_correct,
                "Abs Error": abs_error,
                "Pct Error": pct_error,
            }
        )

    if not new_eval_rows:
        print("No new predictions could be evaluated.")
        return

    new_eval_df = pd.DataFrame(new_eval_rows)

    if eval_df_old.empty:
        final_eval_df = new_eval_df
    else:
        final_eval_df = pd.concat([eval_df_old, new_eval_df], ignore_index=True)

    final_eval_df["Prediction Date"] = pd.to_datetime(final_eval_df["Prediction Date"]).dt.strftime("%Y-%m-%d")
    final_eval_df["Target Date"] = pd.to_datetime(final_eval_df["Target Date"]).dt.strftime("%Y-%m-%d")

    final_eval_df = final_eval_df.drop_duplicates(
        subset=["Prediction Date", "Target Date", "Symbol"],
        keep="last",
    )

    final_eval_df = final_eval_df.sort_values(
        by=["Target Date", "Symbol"],
        ascending=[True, True],
    ).reset_index(drop=True)

    final_eval_df.to_csv(EVALUATION_PATH, index=False, encoding="utf-8-sig")

    print(f"Saved evaluation to {EVALUATION_PATH}")
    print(f"Total evaluated rows: {len(final_eval_df)}")
    print(final_eval_df.tail())


if __name__ == "__main__":
    evaluate_predictions()