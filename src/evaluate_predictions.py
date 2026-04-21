import pandas as pd
import yfinance as yf
from config import DATA_DIR

PREDICTION_HISTORY_PATH = DATA_DIR / "predictions_history.csv"
EVALUATION_PATH = DATA_DIR / "prediction_evaluation.csv"


def get_actual_close(symbol, target_date):
    target_dt = pd.to_datetime(target_date)
    start_dt = target_dt - pd.Timedelta(days=2)
    end_dt = target_dt + pd.Timedelta(days=3)

    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(
            start=start_dt.strftime("%Y-%m-%d"),
            end=end_dt.strftime("%Y-%m-%d"),
            auto_adjust=True,
            timeout=10,
        )

        if df is None or df.empty:
            print(f"[SKIP] No Yahoo data for {symbol}")
            return None, None

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

    if predictions_df.empty:
        print("Predictions history is empty.")
        return

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if EVALUATION_PATH.exists():
        eval_df_old = pd.read_csv(EVALUATION_PATH)
    else:
        eval_df_old = pd.DataFrame()

    existing_keys = set()
    if not eval_df_old.empty and "symbol" in eval_df_old.columns and "target_date" in eval_df_old.columns:
        existing_keys = set(zip(eval_df_old["symbol"], eval_df_old["target_date"]))

    pending_df = predictions_df[
        ~predictions_df.apply(lambda r: (r["symbol"], r["target_date"]) in existing_keys, axis=1)
    ].copy()

    if pending_df.empty:
        print("No pending predictions to evaluate.")
        return

    print(f"Pending rows to evaluate: {len(pending_df)}")

    new_eval_rows = []

    for idx, (_, row) in enumerate(pending_df.iterrows(), start=1):
        symbol = row["symbol"]
        target_date = row["target_date"]

        print(f"[{idx}/{len(pending_df)}] Evaluating {symbol} for {target_date}")

        actual_close, actual_date = get_actual_close(symbol, target_date)

        if actual_close is None:
            continue

        predicted_close = float(row["predicted_close"])
        today_close = float(row["today_close"])

        abs_error = abs(predicted_close - actual_close)
        pct_error = abs_error / actual_close if actual_close != 0 else None

        actual_direction = "UP" if actual_close >= today_close else "DOWN"
        predicted_direction = row["predicted_direction"]
        direction_correct = int(predicted_direction == actual_direction)

        new_eval_rows.append({
            "prediction_date": row["prediction_date"],
            "target_date": row["target_date"],
            "actual_date": actual_date,
            "symbol": symbol,
            "today_close": today_close,
            "predicted_close": predicted_close,
            "actual_close": actual_close,
            "predicted_direction": predicted_direction,
            "actual_direction": actual_direction,
            "direction_correct": direction_correct,
            "abs_error": abs_error,
            "pct_error": pct_error
        })

    if not new_eval_rows:
        print("No new predictions could be evaluated.")
        return

    new_eval_df = pd.DataFrame(new_eval_rows)

    if eval_df_old.empty:
        final_eval_df = new_eval_df
    else:
        final_eval_df = pd.concat([eval_df_old, new_eval_df], ignore_index=True)
        final_eval_df = final_eval_df.drop_duplicates(subset=["symbol", "target_date"], keep="last")

    final_eval_df.to_csv(EVALUATION_PATH, index=False)
    print(f"Saved evaluation to {EVALUATION_PATH}")
    print(final_eval_df.tail())


if __name__ == "__main__":
    evaluate_predictions()