import warnings
from io import StringIO

import pandas as pd
import requests
import urllib3

from config import STOCK_LIST_PATH, ensure_directories


# 關閉 SSL 警告
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
warnings.filterwarnings("ignore", message="Unverified HTTPS request")


TWSE_URL = "https://isin.twse.com.tw/isin/C_public.jsp?strMode=2"
TPEX_URL = "https://isin.twse.com.tw/isin/C_public.jsp?strMode=4"


def fetch_html(url: str) -> str:
    headers = {
        "User-Agent": "Mozilla/5.0",
    }

    response = requests.get(
        url,
        headers=headers,
        timeout=30,
        verify=False,   # 關鍵：跳過 SSL 驗證
    )
    response.raise_for_status()
    response.encoding = "big5"
    return response.text


def parse_isin_table(url: str, market_name: str) -> pd.DataFrame:
    html = fetch_html(url)
    tables = pd.read_html(StringIO(html))

    if not tables:
        raise ValueError(f"No table found for {market_name}")

    df = tables[0].copy()

    # 第一列通常是實際欄名
    df.columns = df.iloc[0]
    df = df.iloc[1:].copy()

    keep_cols = ["有價證券代號及名稱", "市場別", "產業別"]
    existing_cols = [c for c in keep_cols if c in df.columns]
    df = df[existing_cols].copy()

    df = df.dropna(how="all")
    df["有價證券代號及名稱"] = df["有價證券代號及名稱"].astype(str).str.strip()

    # 只抓像 2330 台積電 這種格式
    split_df = df["有價證券代號及名稱"].str.extract(r"^(\d{4,6})\s+(.+)$")
    df["code"] = split_df[0]
    df["name"] = split_df[1]

    df = df.dropna(subset=["code", "name"])
    df = df[df["code"].str.match(r"^\d{4}$")].copy()

    df["market"] = market_name
    df["industry"] = df["產業別"].fillna("Other").astype(str).str.strip()
    df["industry"] = df["industry"].replace("", "Other")

    df = df[["code", "name", "market", "industry"]].copy()
    return df


def normalize_industry(industry: str) -> str:
    industry = str(industry).strip()

    mapping = {
        "半導體業": "Semiconductor",
        "電腦及週邊設備業": "Computer",
        "電子零組件業": "Electronics",
        "通信網路業": "Communication",
        "光電業": "Electronics",
        "其他電子業": "Electronics",
        "電子通路業": "Electronics",
        "數位雲端": "AI Server",
        "航運業": "Other",
        "金融保險業": "Other",
        "鋼鐵工業": "Other",
        "建材營造": "Other",
        "食品工業": "Other",
        "塑膠工業": "Other",
        "電機機械": "Other",
        "生技醫療業": "Other",
        "貿易百貨": "Other",
        "油電燃氣業": "Other",
        "文化創意業": "Other",
        "居家生活": "Other",
        "綠能環保": "Other",
        "運動休閒": "Other",
        "存託憑證": "Other",
        "其他": "Other",
        "未分類": "Other",
    }

    return mapping.get(industry, industry if industry else "Other")


def add_yfinance_ticker(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    def convert_ticker(row):
        if row["market"] == "TWSE":
            return f'{row["code"]}.TW'
        return f'{row["code"]}.TWO'

    df["ticker"] = df.apply(convert_ticker, axis=1)
    return df


def build_stock_list() -> pd.DataFrame:
    ensure_directories()

    print("Fetching TWSE stock list...")
    twse_df = parse_isin_table(TWSE_URL, "TWSE")

    print("Fetching TPEX stock list...")
    tpex_df = parse_isin_table(TPEX_URL, "TPEX")

    df = pd.concat([twse_df, tpex_df], ignore_index=True)

    df["industry"] = df["industry"].apply(normalize_industry)
    df = df.drop_duplicates(subset=["code"]).reset_index(drop=True)
    df = add_yfinance_ticker(df)

    df.to_csv(STOCK_LIST_PATH, index=False, encoding="utf-8-sig")

    print(f"Saved stock list to: {STOCK_LIST_PATH}")
    print(f"Total stocks: {len(df)}")

    return df


if __name__ == "__main__":
    build_stock_list()