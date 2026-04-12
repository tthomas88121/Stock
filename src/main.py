from config import ensure_directories
from stock_list import build_stock_list
from data_fetcher import download_all_prices
from feature_engineering import build_all_features
from model_train import train_model
from backtest import run_backtest


def main():
    ensure_directories()

    print("Step 1: Build stock list")
    build_stock_list()

    print("\nStep 2: Download price data")
    download_all_prices()

    print("\nStep 3: Build features")
    build_all_features()

    print("\nStep 4: Train model")
    train_model()

    print("\nStep 5: Run backtest")
    run_backtest(top_n=3)

    print("\nDone.")


if __name__ == "__main__":
    main()