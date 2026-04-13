from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
PRICE_DIR = RAW_DIR / "price_data"

# Preferred path for deployment
STOCK_LIST_PATH = BASE_DIR / "stock_list.csv"

# Backward-compatible local fallback
STOCK_LIST_FALLBACK_PATH = RAW_DIR / "stock_list.csv"

MERGED_DATASET_PATH = PROCESSED_DIR / "merged_dataset.csv"

MODEL_PATH = BASE_DIR / "random_forest_model.pkl"
REG_MODEL_PATH = BASE_DIR / "random_forest_regressor.pkl"

DOWNLOAD_PERIOD = "2y"
MAX_STOCKS = None

FEATURE_COLUMNS = [
    "MA5",
    "MA20",
    "MA60",
    "RSI14",
    "Return",
    "Vol_Change",
    "Volatility20",
    "IndustryScore",
    "MA20_slope",
    "MA60_slope",
    "Price_Trend_5d",
    "Price_Trend_10d",
    "RSI_Trend",
]

INDUSTRY_SCORE_MAP = {
    "Semiconductor": 0.90,
    "Memory": 0.95,
    "AI Server": 1.00,
    "Electronics": 0.75,
    "Computer": 0.70,
    "Communication": 0.70,
    "Other": 0.50,
}


def get_stock_list_path() -> Path:
    """
    Prefer repo-root stock_list.csv for deployment.
    Fall back to data/raw/stock_list.csv for old local structure.
    """
    if STOCK_LIST_PATH.exists():
        return STOCK_LIST_PATH
    return STOCK_LIST_FALLBACK_PATH


def ensure_directories():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    PRICE_DIR.mkdir(parents=True, exist_ok=True)