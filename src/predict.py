import joblib
import pandas as pd
from config import FEATURE_COLUMNS, MODEL_PATH
from stock_list import build_stock_list
from feature_engineering import build_features_for_one_stock
from config import PRICE_DIR


def predict_all():
    model = joblib.load(MODEL_PATH)
    stock_df = build_stock_list()

    results = []

    for _, row in stock_df.iterrows():
        price_path = PRICE_DIR / f'{row["code"]}.csv'

        if not price_path.exists():
            continue

        try:
            price_df = pd.read_csv(price_path)
            feature_df = build_features_for_one_stock(price_df, row)

            if feature_df.empty:
                continue

            latest = feature_df.iloc[-1]
            X = latest[FEATURE_COLUMNS].values.reshape(1, -1)

            prob = model.predict_proba(X)[0][1]

            results.append({
                "code": row["code"],
                "name": row["name"],
                "industry": row["industry"],
                "prob_up": prob
            })

        except Exception as e:
            print(f"[ERROR] predict {row['code']}: {e}")

    result_df = pd.DataFrame(results)
    result_df = result_df.sort_values(by="prob_up", ascending=False)

    return result_df


if __name__ == "__main__":
    df = predict_all()
    print(df.head(10))