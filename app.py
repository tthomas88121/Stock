import sys
from pathlib import Path
from datetime import datetime

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

BASE_DIR = Path(__file__).resolve().parent
SRC_DIR = BASE_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from config import PRICE_DIR, STOCK_LIST_PATH, get_stock_list_path

TOP_PATH = BASE_DIR / "outputs" / "top_candidates.csv"
DAILY_ALL_PATH = BASE_DIR / "outputs" / "daily_all_predictions.csv"
FAILED_PATH = BASE_DIR / "outputs" / "failed_symbols.csv"

st.set_page_config(
    page_title="AI 台股智慧儀表板 | AI Taiwan Stock Dashboard",
    page_icon="📈",
    layout="wide",
)


def inject_css():
    st.markdown(
        """
        <style>
        #MainMenu {display: none !important;}
        header {display: none !important;}
        footer {display: none !important;}
        [data-testid="stHeader"] {display: none !important; height: 0 !important;}
        [data-testid="stToolbar"] {display: none !important;}
        [data-testid="stDecoration"] {display: none !important;}

        html, body, .stApp {
            margin-top: 0 !important;
            padding-top: 0 !important;
        }

        .block-container {
            padding-top: 0.8rem !important;
            padding-bottom: 1.2rem;
            max-width: 1450px;
        }

        .main-title {
            font-size: 2.2rem;
            font-weight: 800;
            margin-bottom: 0.2rem;
            line-height: 1.2;
        }

        .subtitle {
            color: #6b7280;
            font-size: 0.98rem;
            margin-bottom: 1.2rem;
        }

        .hero-card {
            padding: 1.1rem 1.2rem;
            border: 1px solid rgba(128,128,128,0.18);
            border-radius: 18px;
            background: linear-gradient(135deg, rgba(245,247,250,0.95), rgba(255,255,255,0.98));
            margin-bottom: 1rem;
        }

        .soft-card {
            padding: 0.9rem 1rem;
            border: 1px solid rgba(128,128,128,0.15);
            border-radius: 16px;
            background: rgba(255,255,255,0.9);
            margin-bottom: 0.9rem;
        }

        div[data-testid="stMetric"] {
            background: rgba(255,255,255,0.85);
            border: 1px solid rgba(128,128,128,0.13);
            padding: 12px 14px;
            border-radius: 16px;
        }

        div[data-testid="stDataFrame"] {
            border-radius: 14px;
            overflow: hidden;
        }

        .selected-badge {
            color: #0f766e;
            font-weight: 600;
            font-size: 0.9rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def get_stock_list_candidates():
    return [
        get_stock_list_path(),
        STOCK_LIST_PATH,
        BASE_DIR / "stock_list.csv",
        BASE_DIR / "data" / "raw" / "stock_list.csv",
    ]


@st.cache_data(ttl=3600)
def load_stock_list() -> pd.DataFrame:
    seen = set()

    for path in get_stock_list_candidates():
        path = Path(path)
        if str(path) in seen:
            continue
        seen.add(str(path))

        if path.exists():
            try:
                df = pd.read_csv(path)
                if not df.empty:
                    df["code"] = df["code"].astype(str)
                    return df
            except Exception:
                continue
    return pd.DataFrame()


@st.cache_data(ttl=900)
def load_predictions() -> pd.DataFrame:
    if DAILY_ALL_PATH.exists():
        try:
            df = pd.read_csv(DAILY_ALL_PATH)
            if not df.empty:
                df["code"] = df["code"].astype(str)
                return df
        except Exception:
            pass
    return pd.DataFrame()


@st.cache_data(ttl=900)
def load_top_candidates() -> pd.DataFrame:
    if TOP_PATH.exists():
        try:
            df = pd.read_csv(TOP_PATH)
            if not df.empty:
                df["code"] = df["code"].astype(str)
                return df
        except Exception:
            pass
    return pd.DataFrame()


@st.cache_data(ttl=900)
def load_failed_symbols() -> pd.DataFrame:
    if FAILED_PATH.exists():
        try:
            return pd.read_csv(FAILED_PATH)
        except Exception:
            pass
    return pd.DataFrame()


@st.cache_data(ttl=3600)
def load_local_price(code: str) -> pd.DataFrame:
    path = PRICE_DIR / f"{code}.csv"
    if not path.exists():
        return pd.DataFrame()

    try:
        df = pd.read_csv(path)
        if df.empty:
            return pd.DataFrame()
        return df
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=900)
def fetch_live_price(ticker: str) -> pd.DataFrame:
    try:
        df = yf.download(
            ticker,
            period="1y",
            auto_adjust=True,
            progress=False,
        )
        if df is None or df.empty:
            return pd.DataFrame()

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df = df.reset_index()
        required = ["Date", "Close", "Volume"]
        if not all(col in df.columns for col in required):
            return pd.DataFrame()

        return df
    except Exception:
        return pd.DataFrame()


def build_features(price_df: pd.DataFrame, industry_score: float) -> pd.DataFrame:
    if price_df is None or price_df.empty:
        return pd.DataFrame()

    df = price_df.copy()
    required = ["Date", "Close", "Volume"]
    if not all(col in df.columns for col in required):
        return pd.DataFrame()

    df["MA5"] = df["Close"].rolling(5).mean()
    df["MA20"] = df["Close"].rolling(20).mean()
    df["MA60"] = df["Close"].rolling(60).mean()

    delta = df["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss.replace(0, pd.NA)
    df["RSI14"] = 100 - (100 / (1 + rs))

    df["Return"] = df["Close"].pct_change()
    df["Vol_Change"] = df["Volume"].pct_change()
    df["Volatility20"] = df["Return"].rolling(20).std()

    df["MA20_slope"] = df["MA20"].diff()
    df["MA60_slope"] = df["MA60"].diff()
    df["Price_Trend_5d"] = df["Close"].pct_change(5)
    df["Price_Trend_10d"] = df["Close"].pct_change(10)
    df["RSI_Trend"] = df["RSI14"].diff()

    df["IndustryScore"] = industry_score
    df = df.replace([float("inf"), float("-inf")], pd.NA)
    df = df.dropna().reset_index(drop=True)
    return df


def get_prediction_row(pred_df: pd.DataFrame, code: str):
    if pred_df.empty or "code" not in pred_df.columns:
        return None
    row = pred_df[pred_df["code"] == str(code)]
    if row.empty:
        return None
    return row.iloc[0]


def plot_price(df: pd.DataFrame):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["Date"], y=df["Close"], name="收盤價 Close"))
    fig.add_trace(go.Scatter(x=df["Date"], y=df["MA20"], name="20日均線 MA20"))
    fig.add_trace(go.Scatter(x=df["Date"], y=df["MA60"], name="60日均線 MA60"))
    fig.update_layout(
        height=430,
        margin=dict(l=20, r=20, t=40, b=20),
        hovermode="x unified",
        title="價格走勢 | Price Trend",
        legend_title="指標 | Metrics",
    )
    return fig


def plot_growth(df: pd.DataFrame):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["Date"], y=df["Price_Trend_5d"], name="5日趨勢 5D"))
    fig.add_trace(go.Scatter(x=df["Date"], y=df["Price_Trend_10d"], name="10日趨勢 10D"))
    fig.update_layout(
        height=320,
        margin=dict(l=20, r=20, t=40, b=20),
        hovermode="x unified",
        title="短期動能 | Short-Term Momentum",
    )
    return fig


def plot_rsi(df: pd.DataFrame):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["Date"], y=df["RSI14"], name="RSI"))
    fig.add_trace(go.Scatter(x=df["Date"], y=df["RSI_Trend"], name="RSI 趨勢 RSI Trend"))
    fig.update_layout(
        height=320,
        margin=dict(l=20, r=20, t=40, b=20),
        hovermode="x unified",
        title="相對強弱指標 | RSI Analysis",
    )
    return fig


def fmt_pct(val):
    if val is None or pd.isna(val):
        return "N/A"
    return f"{val * 100:.2f}%"


def fmt_num(val):
    if val is None or pd.isna(val):
        return "N/A"
    return f"{val:.2f}"


def probability_label(prob):
    if prob is None or pd.isna(prob):
        return "無資料 | No Data"
    if prob >= 0.7:
        return "偏多 Strong Bullish"
    if prob >= 0.55:
        return "看多 Bullish"
    if prob >= 0.45:
        return "中性 Neutral"
    if prob >= 0.3:
        return "偏空 Bearish"
    return "看空 Strong Bearish"


def main():
    inject_css()

    debug_mode = st.sidebar.checkbox("Debug Mode", value=False)

    stock_df = load_stock_list()
    pred_df = load_predictions()
    top_df = load_top_candidates()
    failed_df = load_failed_symbols()

    if debug_mode:
        with st.expander("Debug Info", expanded=True):
            st.write("BASE_DIR:", str(BASE_DIR))
            st.write("PRICE_DIR:", str(PRICE_DIR))
            st.write("Configured STOCK_LIST_PATH:", str(STOCK_LIST_PATH))
            st.write("Resolved stock list path:", str(get_stock_list_path()))
            st.write("Candidate stock list paths:", [str(p) for p in get_stock_list_candidates()])
            st.write("Existing candidate paths:", [str(p) for p in get_stock_list_candidates() if Path(p).exists()])
            st.write("TOP_PATH exists:", TOP_PATH.exists())
            st.write("DAILY_ALL_PATH exists:", DAILY_ALL_PATH.exists())
            st.write("FAILED_PATH exists:", FAILED_PATH.exists())
            st.write("Stock rows loaded:", len(stock_df))
            st.write("Prediction rows loaded:", len(pred_df))
            st.write("Top candidates loaded:", len(top_df))
            st.write("Failed symbols loaded:", len(failed_df))

    if stock_df.empty:
        st.error("找不到 stock_list.csv，或檔案為空。 | stock_list.csv not found or empty.")
        st.info("Please place stock_list.csv in repo root or data/raw/stock_list.csv")
        return

    industries = sorted(stock_df["industry"].dropna().astype(str).unique().tolist())

    st.markdown(
        '<div class="main-title">📈 AI 台股智慧儀表板 | AI Taiwan Stock Dashboard</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="subtitle">以機器學習預測方向與報酬，提供雙語介面、精簡專業的投資觀察面板。 | '
        'A bilingual dashboard for direction prediction, return forecasting, and cleaner stock analysis workflow.</div>',
        unsafe_allow_html=True,
    )

    st.sidebar.header("控制面板 | Control Panel")
    st.sidebar.markdown("### 產業權重 | Industry Weights")
    industry_weights = {
        ind: st.sidebar.slider(ind, 0.0, 2.0, 1.0, 0.1)
        for ind in industries
    }

    pinned_codes = ["2330", "2454", "2408", "6669"]
    available_codes = stock_df["code"].astype(str).tolist()
    pinned_codes = [c for c in pinned_codes if c in available_codes]
    fallback_code = pinned_codes[0] if pinned_codes else available_codes[0]
    default_code = "2330" if "2330" in available_codes else fallback_code

    if "selected_code" not in st.session_state:
        st.session_state.selected_code = default_code

    if st.session_state.selected_code not in available_codes:
        st.session_state.selected_code = default_code

    last_update_text = "N/A"
    if DAILY_ALL_PATH.exists():
        ts = datetime.fromtimestamp(DAILY_ALL_PATH.stat().st_mtime)
        last_update_text = ts.strftime("%Y-%m-%d %H:%M")

    hero_left, hero_mid, hero_right = st.columns([2.2, 1.1, 1.1])

    with hero_left:
        st.markdown('<div class="hero-card">', unsafe_allow_html=True)
        st.markdown("### 今日總覽 | Daily Overview")
        st.write("使用本地歷史資料與每日預測結果，快速查看今日候選股票、個股訊號與技術指標。")
        st.write("Use your local historical data and daily predictions to review top candidates, stock signals, and technical indicators.")
        st.markdown("</div>", unsafe_allow_html=True)

    with hero_mid:
        st.metric("股票數量 | Stocks", len(stock_df))
        st.metric("預測筆數 | Predictions", len(pred_df) if not pred_df.empty else 0)

    with hero_right:
        st.metric("更新時間 | Last Update", last_update_text)
        st.metric("失敗筆數 | Failed", len(failed_df) if not failed_df.empty else 0)

    st.markdown("### 🔥 今日推薦 | Top Picks")
    if not top_df.empty:
        show_cols = [c for c in ["code", "name", "industry", "prob_up", "pred_return", "pred_price"] if c in top_df.columns]
        show_df = top_df[show_cols].copy()

        rename_map = {
            "code": "代碼 Code",
            "name": "名稱 Name",
            "industry": "產業 Industry",
            "prob_up": "上漲機率 Up Prob",
            "pred_return": "預測報酬 Pred Return",
            "pred_price": "預測價格 Pred Price",
        }
        show_df = show_df.rename(columns=rename_map)

        if "上漲機率 Up Prob" in show_df.columns:
            show_df["上漲機率 Up Prob"] = (show_df["上漲機率 Up Prob"] * 100).round(2).astype(str) + "%"
        if "預測報酬 Pred Return" in show_df.columns:
            show_df["預測報酬 Pred Return"] = (show_df["預測報酬 Pred Return"] * 100).round(2).astype(str) + "%"
        if "預測價格 Pred Price" in show_df.columns:
            show_df["預測價格 Pred Price"] = show_df["預測價格 Pred Price"].round(2)

        st.dataframe(show_df.head(10), use_container_width=True, hide_index=True)
    else:
        st.info("目前沒有 top_candidates.csv 可顯示。 | No cached top candidates found.")

    st.markdown("### ⭐ 置頂個股 | Pinned Stocks")
    cols = st.columns(max(len(pinned_codes), 1))

    for i, code in enumerate(pinned_codes):
        row = stock_df[stock_df["code"] == code].iloc[0]
        pred_row = get_prediction_row(pred_df, code)
        weight = industry_weights.get(str(row["industry"]), 1.0)

        with cols[i]:
            st.markdown('<div class="soft-card">', unsafe_allow_html=True)

            if st.button(f"{code} | {row['name']}", key=f"pin_{code}", use_container_width=True):
                st.session_state.selected_code = code

            if pred_row is not None:
                prob = float(pred_row["prob_up"]) if "prob_up" in pred_row else None
                pred_ret = float(pred_row["pred_return"]) if "pred_return" in pred_row else None
                weighted = prob * weight if prob is not None else None

                st.metric("上漲機率 | Up Prob", fmt_pct(prob))
                st.metric("加權分數 | Weighted", fmt_pct(weighted))
                st.caption(f"預測報酬 | Pred Return: {fmt_pct(pred_ret)}")
                st.caption(f"訊號 | Signal: {probability_label(prob)}")
            else:
                st.caption(f"產業 | Industry: {row['industry']}")

            if st.session_state.selected_code == code:
                st.markdown('<div class="selected-badge">已選取 | Selected</div>', unsafe_allow_html=True)

            st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("### 🔎 個股分析 | Stock Analysis")

    stock_df["label"] = (
        stock_df["code"].astype(str)
        + " - "
        + stock_df["name"].astype(str)
        + " ("
        + stock_df["industry"].astype(str)
        + ")"
    )
    code_to_label = dict(zip(stock_df["code"], stock_df["label"]))

    current_code = st.session_state.selected_code
    selected_code = st.selectbox(
        "選擇股票 | Select Stock",
        options=available_codes,
        index=available_codes.index(current_code),
        format_func=lambda x: code_to_label.get(x, x),
        key="stock_selector",
    )

    if selected_code != st.session_state.selected_code:
        st.session_state.selected_code = selected_code

    code = st.session_state.selected_code
    row = stock_df[stock_df["code"] == code].iloc[0]
    ticker = row["ticker"]
    weight = industry_weights.get(str(row["industry"]), 1.0)

    if debug_mode:
        st.caption(f"Selected code: {code} | ticker: {ticker}")

    price_df = load_local_price(code)
    data_source = "Local CSV"
    if price_df.empty:
        price_df = fetch_live_price(ticker)
        data_source = "Yahoo Finance"

    if price_df.empty:
        st.error(f"無法讀取 {code} 的股價資料。 | Could not load price data for {code}.")
        return

    feature_df = build_features(price_df, weight)
    if feature_df.empty:
        st.error(f"無法建立 {code} 的特徵。 | Could not build features for {code}.")
        return

    latest = feature_df.iloc[-1]
    pred_row = get_prediction_row(pred_df, code)

    prob_up = None
    pred_return = None
    pred_price = None

    if pred_row is not None:
        if "prob_up" in pred_row:
            prob_up = float(pred_row["prob_up"])
        if "pred_return" in pred_row:
            pred_return = float(pred_row["pred_return"])
        if "pred_price" in pred_row:
            pred_price = float(pred_row["pred_price"])

    st.markdown(
        f"#### {code} - {row['name']} | {row['industry']} | 訊號 Signal: {probability_label(prob_up)}"
    )

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("收盤價 | Close", f"{latest['Close']:.2f}")
    m2.metric("20日均線 | MA20", f"{latest['MA20']:.2f}")
    m3.metric("60日均線 | MA60", f"{latest['MA60']:.2f}")
    m4.metric("RSI 指標 | RSI", f"{latest['RSI14']:.1f}")

    m5, m6, m7, m8 = st.columns(4)
    m5.metric("上漲機率 | Up Probability", fmt_pct(prob_up))
    m6.metric("預測報酬 | Predicted Return", fmt_pct(pred_return))
    m7.metric("預測價格 | Predicted Next Price", fmt_num(pred_price))
    m8.metric(
        "加權分數 | Weighted Score",
        fmt_pct(prob_up * weight if prob_up is not None else None),
    )

    st.caption(
        f"產業權重 | Industry Weight: {weight:.2f}   •   "
        f"資料來源 | Data Source: {data_source}"
    )

    tab1, tab2, tab3, tab4 = st.tabs([
        "價格圖表 | Price",
        "成長動能 | Growth",
        "RSI 分析 | RSI",
        "原始資料 | Data",
    ])

    with tab1:
        st.plotly_chart(plot_price(feature_df), use_container_width=True)

    with tab2:
        st.plotly_chart(plot_growth(feature_df), use_container_width=True)

    with tab3:
        st.plotly_chart(plot_rsi(feature_df), use_container_width=True)

    with tab4:
        show_cols = [
            c for c in [
                "Date", "Close", "MA20", "MA60", "RSI14",
                "Price_Trend_5d", "Price_Trend_10d", "RSI_Trend"
            ] if c in feature_df.columns
        ]
        st.dataframe(feature_df[show_cols].tail(30), use_container_width=True, hide_index=True)

    if not failed_df.empty:
        with st.expander("失敗股票清單 | Failed Symbols"):
            st.dataframe(failed_df, use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()