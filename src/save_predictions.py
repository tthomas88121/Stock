from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
PREDICTION_HISTORY_PATH = DATA_DIR / "predictions_history.csv"
EVALUATION_PATH = DATA_DIR / "prediction_evaluation.csv"


def get_next_trading_day(date_obj):
    next_day = date_obj + timedelta(days=1)
    while next_day.weekday() >= 5:
        next_day += timedelta(days=1)
    return next_day


def save_predictions(prediction_rows):
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    today = datetime.today().date()
    target_date = get_next_trading_day(today)

    records = []
    for row in prediction_rows:
        today_close = float(row["today_close"])
        predicted_close = float(row["predicted_close"])
        predicted_return = (predicted_close - today_close) / today_close if today_close != 0 else 0.0
        predicted_direction = "UP" if predicted_close >= today_close else "DOWN"

        records.append({
            "prediction_date": str(today),
            "target_date": str(target_date),
            "symbol": row["symbol"],
            "today_close": today_close,
            "predicted_close": predicted_close,
            "predicted_return": predicted_return,
            "predicted_direction": predicted_direction
        })

    new_df = pd.DataFrame(records)

    if PREDICTION_HISTORY_PATH.exists():
        old_df = pd.read_csv(PREDICTION_HISTORY_PATH)
        combined = pd.concat([old_df, new_df], ignore_index=True)
        combined = combined.drop_duplicates(subset=["symbol", "target_date"], keep="last")
    else:
        combined = new_df

    combined.to_csv(PREDICTION_HISTORY_PATH, index=False)
    print(f"Saved predictions to {PREDICTION_HISTORY_PATH}")