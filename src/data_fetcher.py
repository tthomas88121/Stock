import pandas as pd
import yfinance as yf
from config import PRICE_DIR, DOWNLOAD_PERIOD, MAX_STOCKS, ensure_directories
from stock_list import build_stock_list

def download_one_stock(ticker: str, code: str) -> bool:
    try:
        df = yf.download(
            ticker,
            period=DOWNLOAD_PERIOD,
            auto_adjust=True,
            progress=False,
        )

        if df.empty:
            print(f"[WARN] No data for {ticker}")
            return False

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df = df.reset_index()
        save_path = PRICE_DIR / f"{code}.csv"
        df.to_csv(save_path, index=False, encoding="utf-8-sig")
        print(f"[OK] Saved {ticker} -> {save_path.name}")
        return True

    except Exception as e:
        print(f"[ERROR] {ticker}: {e}")
        return False

def download_all_prices():
    ensure_directories()
    stock_df = build_stock_list()

    if MAX_STOCKS:
        stock_df = stock_df.head(MAX_STOCKS)

    success_count = 0

    for _, row in stock_df.iterrows():
        ok = download_one_stock(row["ticker"], row["code"])
        if ok:
            success_count += 1

    print(f"\nDownloaded {success_count}/{len(stock_df)} stocks successfully.")

if __name__ == "__main__":
    download_all_prices()