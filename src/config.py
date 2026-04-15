from pathlib import Path

# config.py is inside src/, so parent.parent = project root
BASE_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = BASE_DIR / "src"

DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PRICE_DIR = DATA_DIR / "price"
OUTPUT_DIR = BASE_DIR / "outputs"

STOCK_LIST_PATH = BASE_DIR / "stock_list.csv"
MERGED_DATASET_PATH = DATA_DIR / "merged_dataset.csv"

MODEL_PATH = BASE_DIR / "random_forest_model.pkl"
REG_MODEL_PATH = BASE_DIR / "random_forest_regressor.pkl"


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
    "Return_1d",
    "Return_3d",
    "Return_5d",
    "Return_10d",
    "Vol_5d",
    "Vol_10d",
    "MA20_gap",
    "MA60_gap",
    "Volume_ratio",
]


def ensure_directories():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PRICE_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def get_stock_list_path() -> Path:
    candidates = [
        BASE_DIR / "stock_list.csv",
        RAW_DIR / "stock_list.csv",
        STOCK_LIST_PATH,
    ]
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]