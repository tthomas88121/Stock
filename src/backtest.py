import numpy as np
import pandas as pd
import joblib

from config import FEATURE_COLUMNS, MERGED_DATASET_PATH, MODEL_PATH, ensure_directories


def run_backtest(top_n: int = 3):
    ensure_directories()

    df = pd.read_csv(MERGED_DATASET_PATH)

    # 日期格式
    df["Date"] = pd.to_datetime(df["Date"])

    # 先清 feature 欄位，不要再額外 concat，避免重複欄位
    df[FEATURE_COLUMNS] = df[FEATURE_COLUMNS].replace([np.inf, -np.inf], np.nan)

    # 刪掉必要欄位有缺值的列
    required_cols = FEATURE_COLUMNS + ["Target", "Close", "Date", "code"]
    df = df.dropna(subset=required_cols).copy()

    # 載入模型
    model = joblib.load(MODEL_PATH)

    # 一定要用跟訓練時完全一樣的欄位順序
    X_backtest = df.loc[:, FEATURE_COLUMNS].copy()

    # 預測每列上漲機率
    df["prob_up"] = model.predict_proba(X_backtest)[:, 1]

    # 算隔天報酬
    df["next_close"] = df.groupby("code")["Close"].shift(-1)
    df["next_day_return"] = (df["next_close"] - df["Close"]) / df["Close"]

    # 去掉最後一天沒 next_close 的資料
    df = df.dropna(subset=["next_close", "next_day_return"]).copy()

    daily_results = []
    unique_dates = sorted(df["Date"].unique())

    for date in unique_dates:
        day_df = df[df["Date"] == date].copy()

        if day_df.empty:
            continue

        top_df = day_df.sort_values(by="prob_up", ascending=False).head(top_n)

        avg_return = top_df["next_day_return"].mean()
        win_rate = (top_df["next_day_return"] > 0).mean()

        daily_results.append(
            {
                "Date": date,
                "Selected_Count": len(top_df),
                "Average_Return": avg_return,
                "Win_Rate": win_rate,
            }
        )

    result_df = pd.DataFrame(daily_results)

    if result_df.empty:
        print("No backtest result.")
        return None

    result_df["Cumulative_Return"] = (1 + result_df["Average_Return"]).cumprod() - 1

    total_return = result_df["Cumulative_Return"].iloc[-1]
    avg_daily_return = result_df["Average_Return"].mean()
    avg_win_rate = result_df["Win_Rate"].mean()

    print("\n=== Backtest Result ===")
    print(f"Top N each day: {top_n}")
    print(f"Average Daily Return: {avg_daily_return:.4%}")
    print(f"Average Win Rate: {avg_win_rate:.2%}")
    print(f"Total Cumulative Return: {total_return:.2%}")

    print("\nLast 10 days:")
    print(result_df.tail(10))

    return result_df


if __name__ == "__main__":
    run_backtest(top_n=3)