from pathlib import Path
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
PREDICTION_HISTORY_PATH = DATA_DIR / "predictions_history.csv"


def normalize_prediction_rows(prediction_rows):
    records = []

    for row in prediction_rows:
        prediction_date = row.get("Prediction Date")
        target_date = row.get("Target Date")
        symbol = row.get("Symbol")

        today_close = float(row.get("Today Close"))
        pred_close = float(row.get("Pred Close"))

        pred_return = (pred_close - today_close) / today_close if today_close != 0 else 0.0
        pred_dir = row.get("Pred Dir")
        if not pred_dir:
            pred_dir = "UP" if pred_close >= today_close else "DOWN"

        records.append(
            {
                "Prediction Date": prediction_date,
                "Target Date": target_date,
                "Symbol": symbol,
                "Today Close": today_close,
                "Pred Close": pred_close,
                "Pred Return": pred_return,
                "Pred Dir": pred_dir,
            }
        )

    return pd.DataFrame(records)


def save_predictions(prediction_rows):
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    new_df = normalize_prediction_rows(prediction_rows)

    if new_df.empty:
        print("No new prediction rows to save.")
        return

    new_df["Prediction Date"] = pd.to_datetime(new_df["Prediction Date"]).dt.strftime("%Y-%m-%d")
    new_df["Target Date"] = pd.to_datetime(new_df["Target Date"]).dt.strftime("%Y-%m-%d")

    if PREDICTION_HISTORY_PATH.exists():
        old_df = pd.read_csv(PREDICTION_HISTORY_PATH)

        combined = pd.concat([old_df, new_df], ignore_index=True)
    else:
        combined = new_df

    combined["Prediction Date"] = pd.to_datetime(combined["Prediction Date"]).dt.strftime("%Y-%m-%d")
    combined["Target Date"] = pd.to_datetime(combined["Target Date"]).dt.strftime("%Y-%m-%d")

    combined = combined.drop_duplicates(
        subset=["Prediction Date", "Target Date", "Symbol"],
        keep="last",
    )

    combined = combined.sort_values(
        by=["Prediction Date", "Symbol"],
        ascending=[True, True],
    ).reset_index(drop=True)

    combined.to_csv(PREDICTION_HISTORY_PATH, index=False, encoding="utf-8-sig")

    print(f"Saved predictions to {PREDICTION_HISTORY_PATH}")
    print(f"Total prediction history rows: {len(combined)}")