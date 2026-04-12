import pandas as pd
from predict import predict_all
from config import BASE_DIR

OUTPUT_PATH = BASE_DIR / "outputs" / "top_candidates.csv"


def scan_market(top_n=10):
    df = predict_all()

    if df.empty:
        print("No prediction results.")
        return

    top_df = df.head(top_n)

    print("\n🔥 Top Candidates 🔥")
    print(top_df)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    top_df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")

    print(f"\nSaved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    scan_market()