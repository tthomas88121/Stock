import sys
from pathlib import Path
from datetime import datetime

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from config import (
    PRICE_DIR,
    STOCK_LIST_PATH,
    get_stock_list_path,
    OUTPUT_DIR,
)

TOP_PATH = OUTPUT_DIR / "top_candidates.csv"
DAILY_ALL_PATH = OUTPUT_DIR / "daily_all_predictions.csv"

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
            padding-top: 0.45rem !important;
            padding-bottom: 1.2rem;
            max-width: 1450px;
        }

        .main-title {
            font-size: 2.1rem;
            font-weight: 800;
            margin-bottom: 0.2rem;
            line-height: 1.2;
        }

        .subtitle {
            color: #6b7280;
            font-size: 0.98rem;
            margin-bottom: 1rem;
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
            background: rgba(255,255,255,0.92);
            margin-bottom: 0.9rem;
        }

        .explain-card {
            padding: 1rem 1.1rem;
            border: 1px solid rgba(128,128,128,0.15);
            border-radius: 16px;
            background: rgba(255,255,255,0.94);
            margin-top: 0.8rem;
        }

        .signal-bull {
            color: #15803d;
            font-weight: 700;
            font-size: 1rem;
        }

        .signal-neutral {
            color: #b45309;
            font-weight: 700;
            font-size: 1rem;
        }

        .signal-bear {
            color: #b91c1c;
            font-weight: 700;
            font-size: 1rem;
        }

        div[data-testid="stMetric"] {
            background: rgba(255,255,255,0.88);
            border: 1px solid rgba(128,128,128,0.12);
            padding: 12px 14px;
            border-radius: 16px;
        }

        div[data-testid="stDataFrame"] {
            border-radius: 14px;
            overflow: hidden;
        }

        .selected-badge {
            color: #0f766e;
            font-weight: 700;
            font-size: 0.92rem;
        }

        .small-muted {
            color: #6b7280;
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
        ROOT_DIR / "stock_list.csv",
        ROOT_DIR / "data" / "raw" / "stock_list.csv",
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


@st.cache_data(ttl=3600)
def load_local_price(code: str) -> pd.DataFrame:
    path = PRICE_DIR / f"{code}.csv"
    if not path.exists():
        return pd.DataFrame()

    try:
        df = pd.read_csv(path)
        if df.empty:
            return pd.DataFrame()

        required = ["Date", "Close", "Volume"]
        if not all(col in df.columns for col in required):
            return pd.DataFrame()

        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df.dropna(subset=["Date"]).copy()
        return df
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=600)
def fetch_live_price(ticker: str) -> pd.DataFrame:
    try:
        ticker = str(ticker).strip()

        if ticker.isdigit():
            ticker = f"{ticker}.TW"

        if not (ticker.endswith(".TW") or ticker.endswith(".TWO")):
            if ticker.replace(".", "").isdigit():
                ticker = f"{ticker}.TW"

        df = yf.download(
            ticker,
            period="1y",
            interval="1d",
            auto_adjust=True,
            progress=False,
            threads=False,
        )

        if df is None or df.empty:
            return pd.DataFrame()

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df = df.reset_index()

        required = ["Date", "Close", "Volume"]
        if not all(col in df.columns for col in required):
            return pd.DataFrame()

        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df.dropna(subset=["Date"]).copy()

        return df
    except Exception as e:
        print("Yahoo fetch error:", e)
        return pd.DataFrame()


def get_best_price_data(code: str, ticker: str) -> tuple[pd.DataFrame, str]:
    local_df = load_local_price(code)

    if not local_df.empty and len(local_df) >= 80:
        return local_df, "Cached CSV"

    live_df = fetch_live_price(ticker)
    if not live_df.empty and len(live_df) >= 80:
        return live_df, "Yahoo Finance"

    if not local_df.empty:
        return local_df, "Cached CSV (fallback)"

    return pd.DataFrame(), "No Data"


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

    df["Return_1d"] = df["Close"].pct_change(1)
    df["Return_3d"] = df["Close"].pct_change(3)
    df["Return_5d"] = df["Close"].pct_change(5)
    df["Return_10d"] = df["Close"].pct_change(10)

    daily_ret = df["Close"].pct_change()
    df["Vol_5d"] = daily_ret.rolling(5).std()
    df["Vol_10d"] = daily_ret.rolling(10).std()

    df["MA20_gap"] = (df["Close"] - df["MA20"]) / df["MA20"].replace(0, pd.NA)
    df["MA60_gap"] = (df["Close"] - df["MA60"]) / df["MA60"].replace(0, pd.NA)

    volume_ma20 = df["Volume"].rolling(20).mean()
    df["Volume_ratio"] = df["Volume"] / volume_ma20.replace(0, pd.NA)

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


def signal_class(prob):
    if prob is None or pd.isna(prob):
        return "signal-neutral"
    if prob >= 0.55:
        return "signal-bull"
    if prob >= 0.45:
        return "signal-neutral"
    return "signal-bear"


def trade_signal_info(prob_up, pred_return, close_price, ma20, ma60):
    if pred_return is None or pd.isna(pred_return):
        return {
            "label": "NO DATA",
            "class_name": "signal-neutral",
            "reason": "No prediction data available."
        }

    trend_up = (
        close_price is not None and ma20 is not None and ma60 is not None
        and not pd.isna(close_price)
        and not pd.isna(ma20)
        and not pd.isna(ma60)
        and close_price > ma20
        and ma20 > ma60
    )

    above_ma20 = (
        close_price is not None and ma20 is not None
        and not pd.isna(close_price)
        and not pd.isna(ma20)
        and close_price > ma20
    )

    prob_ok = (prob_up is not None and not pd.isna(prob_up) and prob_up >= 0.55)

    if pred_return >= 0.03 and trend_up and prob_ok:
        return {
            "label": "🔥 STRONG BUY",
            "class_name": "signal-bull",
            "reason": "Predicted return > 3%, trend is strong (Close > MA20 > MA60), and probability supports entry."
        }

    if pred_return >= 0.02 and trend_up:
        return {
            "label": "✅ BUY",
            "class_name": "signal-bull",
            "reason": "Predicted return > 2% and trend filter passes (Close > MA20 > MA60)."
        }

    if pred_return >= 0.015 and above_ma20:
        return {
            "label": "👀 WATCH",
            "class_name": "signal-neutral",
            "reason": "Predicted return is decent, but trend strength is not fully confirmed yet."
        }

    return {
        "label": "⛔ NO BUY",
        "class_name": "signal-bear",
        "reason": "Predicted return is too low or trend filter does not pass."
    }


def trading_bias_text(pred_return):
    if pred_return is None or pd.isna(pred_return):
        return "No Signal"
    if pred_return >= 0.03:
        return "High-conviction setup"
    if pred_return >= 0.02:
        return "Valid buy setup"
    if pred_return >= 0.015:
        return "Watchlist candidate"
    return "Weak setup"


def init_state(available_codes, default_code):
    if "selected_code" not in st.session_state:
        st.session_state.selected_code = default_code

    if st.session_state.selected_code not in available_codes:
        st.session_state.selected_code = default_code

    if "pinned_codes" not in st.session_state:
        default_pins = ["2330", "2454", "2408", "6669"]
        st.session_state.pinned_codes = [c for c in default_pins if c in available_codes]


def add_pin(code: str, available_codes: list[str]):
    if code in available_codes and code not in st.session_state.pinned_codes:
        st.session_state.pinned_codes.append(code)


def remove_pin(code: str):
    if code in st.session_state.pinned_codes:
        st.session_state.pinned_codes.remove(code)
        if not st.session_state.pinned_codes:
            st.session_state.pinned_codes = []


def main():
    inject_css()

    stock_df = load_stock_list()
    pred_df = load_predictions()
    top_df = load_top_candidates()

    if stock_df.empty:
        st.error("找不到 stock_list.csv，或檔案為空。 | stock_list.csv not found or empty.")
        st.info("Please place stock_list.csv in repo root or data/raw/stock_list.csv")
        return

    industries = sorted(stock_df["industry"].dropna().astype(str).unique().tolist())
    available_codes = stock_df["code"].astype(str).tolist()
    default_code = "2330" if "2330" in available_codes else available_codes[0]

    init_state(available_codes, default_code)

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
    debug_mode = st.sidebar.checkbox("Debug Mode", value=False)

    st.sidebar.markdown("### 產業權重 | Industry Weights")
    industry_weights = {
        ind: st.sidebar.slider(ind, 0.0, 2.0, 1.0, 0.1)
        for ind in industries
    }

    stock_df["label"] = (
        stock_df["code"].astype(str)
        + " - "
        + stock_df["name"].astype(str)
        + " ("
        + stock_df["industry"].astype(str)
        + ")"
    )
    code_to_label = dict(zip(stock_df["code"], stock_df["label"]))

    st.sidebar.markdown("### 自訂置頂股票 | Custom Pins")
    pin_to_add = st.sidebar.selectbox(
        "新增置頂股票 | Add Pin",
        options=available_codes,
        format_func=lambda x: code_to_label.get(x, x),
        key="pin_selector",
    )

    c1, c2 = st.sidebar.columns(2)
    with c1:
        if st.button("加入置頂 | Add Pin", use_container_width=True):
            add_pin(pin_to_add, available_codes)
            st.session_state.selected_code = pin_to_add
            st.rerun()

    with c2:
        if st.button("重設置頂 | Reset Pins", use_container_width=True):
            default_pins = ["2330", "2454", "2408", "6669"]
            st.session_state.pinned_codes = [c for c in default_pins if c in available_codes]
            if st.session_state.pinned_codes:
                st.session_state.selected_code = st.session_state.pinned_codes[0]
            st.rerun()

    if debug_mode:
        with st.expander("Debug Info", expanded=True):
            st.write("ROOT_DIR:", str(ROOT_DIR))
            st.write("SRC_DIR:", str(SRC_DIR))
            st.write("PRICE_DIR:", str(PRICE_DIR))
            st.write("OUTPUT_DIR:", str(OUTPUT_DIR))
            st.write("TOP_PATH:", str(TOP_PATH))
            st.write("DAILY_ALL_PATH:", str(DAILY_ALL_PATH))
            st.write("TOP_PATH exists:", TOP_PATH.exists())
            st.write("DAILY_ALL_PATH exists:", DAILY_ALL_PATH.exists())
            st.write("Configured STOCK_LIST_PATH:", str(STOCK_LIST_PATH))
            st.write("Resolved stock list path:", str(get_stock_list_path()))
            st.write("Selected code:", st.session_state.selected_code)
            st.write("Pinned codes:", st.session_state.pinned_codes)
            st.write("Stock rows loaded:", len(stock_df))
            st.write("Prediction rows loaded:", len(pred_df))
            st.write("Top candidates loaded:", len(top_df))

    last_update_text = "N/A"
    if DAILY_ALL_PATH.exists():
        ts = datetime.fromtimestamp(DAILY_ALL_PATH.stat().st_mtime)
        last_update_text = ts.strftime("%Y-%m-%d %H:%M")

    hero_left, hero_mid, hero_right = st.columns([2.2, 1.1, 1.1])

    with hero_left:
        st.markdown('<div class="hero-card">', unsafe_allow_html=True)
        st.markdown("### 今日總覽 | Daily Overview")
        st.write("使用每日預測結果與穩定股價來源，快速查看今日候選股票、個股訊號與技術指標。")
        st.write("Use daily predictions and stable price sources to review candidates, stock signals, and technical indicators.")
        st.markdown("</div>", unsafe_allow_html=True)

    with hero_mid:
        st.metric("股票數量 | Stocks", len(stock_df))
        st.metric("預測筆數 | Predictions", len(pred_df) if not pred_df.empty else 0)

    with hero_right:
        st.metric("更新時間 | Last Update", last_update_text)
        st.metric("置頂數量 | Pinned", len(st.session_state.pinned_codes))

    st.markdown("### 🔥 今日推薦 | Top Picks")
    if not top_df.empty:
        show_cols = [c for c in ["code", "name", "industry", "signal", "prob_up", "pred_return", "pred_price"] if c in top_df.columns]
        show_df = top_df[show_cols].copy()

        rename_map = {
            "code": "代碼 Code",
            "name": "名稱 Name",
            "industry": "產業 Industry",
            "signal": "交易訊號 Signal",
            "prob_up": "上漲機率 Up Prob",
            "pred_return": "預測報酬 Pred Return",
            "pred_price": "預期收盤價 Expected Next Close",
        }
        show_df = show_df.rename(columns=rename_map)

        if "上漲機率 Up Prob" in show_df.columns:
            show_df["上漲機率 Up Prob"] = (show_df["上漲機率 Up Prob"] * 100).round(2).astype(str) + "%"
        if "預測報酬 Pred Return" in show_df.columns:
            show_df["預測報酬 Pred Return"] = (show_df["預測報酬 Pred Return"] * 100).round(2).astype(str) + "%"
        if "預期收盤價 Expected Next Close" in show_df.columns:
            show_df["預期收盤價 Expected Next Close"] = show_df["預期收盤價 Expected Next Close"].round(2)

        st.dataframe(show_df.head(10), use_container_width=True, hide_index=True)
    else:
        st.info("目前沒有 top_candidates.csv 可顯示。 | No cached top candidates found.")

    st.markdown("### ⭐ 置頂個股 | Pinned Stocks")

    pinned_codes = [c for c in st.session_state.pinned_codes if c in available_codes]
    if not pinned_codes:
        pinned_codes = [default_code]
        st.session_state.pinned_codes = pinned_codes

    cols = st.columns(len(pinned_codes))

    for i, code in enumerate(pinned_codes):
        row = stock_df[stock_df["code"] == code].iloc[0]
        pred_row = get_prediction_row(pred_df, code)
        weight = industry_weights.get(str(row["industry"]), 1.0)

        with cols[i]:
            st.markdown('<div class="soft-card">', unsafe_allow_html=True)

            if st.button(f"{code} | {row['name']}", key=f"pin_btn_{code}", use_container_width=True):
                st.session_state.selected_code = code
                st.rerun()

            if pred_row is not None:
                prob = float(pred_row["prob_up"]) if "prob_up" in pred_row else None
                pred_ret = float(pred_row["pred_return"]) if "pred_return" in pred_row else None
                weighted = prob * weight if prob is not None else None
                signal = pred_row["signal"] if "signal" in pred_row else probability_label(prob)

                st.metric("上漲機率 | Up Prob", fmt_pct(prob))
                st.metric("加權分數 | Weighted", fmt_pct(weighted))
                st.caption(f"預測報酬 | Pred Return: {fmt_pct(pred_ret)}")
                st.markdown(
                    f'<div class="{signal_class(prob)}">訊號 | Signal: {signal}</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.caption(f"產業 | Industry: {row['industry']}")

            _, remove_col = st.columns([3, 2])
            with remove_col:
                if st.button("取消置頂", key=f"remove_pin_{code}", use_container_width=True):
                    remove_pin(code)
                    if st.session_state.selected_code == code and st.session_state.pinned_codes:
                        st.session_state.selected_code = st.session_state.pinned_codes[0]
                    elif st.session_state.selected_code == code:
                        st.session_state.selected_code = default_code
                    st.rerun()

            if st.session_state.selected_code == code:
                st.markdown('<div class="selected-badge">已選取 | Selected</div>', unsafe_allow_html=True)

            st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("### 🔎 個股分析 | Stock Analysis")

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
        st.rerun()

    code = st.session_state.selected_code
    row = stock_df[stock_df["code"] == code].iloc[0]
    ticker = row["ticker"]
    weight = industry_weights.get(str(row["industry"]), 1.0)

    if debug_mode:
        st.caption(f"Selected code: {code} | ticker: {ticker}")

    price_df, data_source = get_best_price_data(code, ticker)

    if debug_mode:
        st.caption(f"Data source used: {data_source}")
        st.caption(f"Price rows: {len(price_df) if not price_df.empty else 0}")

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

    trade_signal = trade_signal_info(
        prob_up=prob_up,
        pred_return=pred_return,
        close_price=latest["Close"],
        ma20=latest["MA20"],
        ma60=latest["MA60"],
    )

    st.markdown(
        f"#### {code} - {row['name']} | {row['industry']}",
        unsafe_allow_html=False,
    )
    st.markdown(
        f'<div class="{trade_signal["class_name"]}">交易訊號 | Trading Signal: {trade_signal["label"]}</div>',
        unsafe_allow_html=True,
    )
    st.caption(f"Signal Reason: {trade_signal['reason']}")

    _, pin_col = st.columns([5, 1])
    with pin_col:
        if code not in st.session_state.pinned_codes:
            if st.button("加入置頂股票", use_container_width=True):
                add_pin(code, available_codes)
                st.rerun()

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("收盤價 | Close", f"{latest['Close']:.2f}")
    m2.metric("20日均線 | MA20", f"{latest['MA20']:.2f}")
    m3.metric("60日均線 | MA60", f"{latest['MA60']:.2f}")
    m4.metric("RSI 指標 | RSI", f"{latest['RSI14']:.1f}")

    m5, m6, m7, m8 = st.columns(4)
    m5.metric("上漲機率 | Up Probability", fmt_pct(prob_up))
    m6.metric("預測報酬 | Predicted Return", fmt_pct(pred_return))
    m7.metric("預期收盤價 | Expected Next Close", fmt_num(pred_price))
    m8.metric("Setup Quality", trading_bias_text(pred_return))

    st.caption(
        f"產業權重 | Industry Weight: {weight:.2f}   •   "
        f"資料來源 | Data Source: {data_source}"
    )

    tab1, tab2, tab3, tab4 = st.tabs([
        "價格圖表 | Price",
        "成長動能 | Growth",
        "RSI 分析 | RSI",
        "指標解釋 | Indicator Guide",
    ])

    with tab1:
        st.plotly_chart(plot_price(feature_df), use_container_width=True)

    with tab2:
        st.plotly_chart(plot_growth(feature_df), use_container_width=True)

    with tab3:
        st.plotly_chart(plot_rsi(feature_df), use_container_width=True)

    with tab4:
        st.markdown('<div class="explain-card">', unsafe_allow_html=True)
        st.markdown("#### 技術指標說明 | Indicator Guide")

        st.markdown("**MA20（20日均線）**")
        st.write("代表最近 20 個交易日平均收盤價，通常用來看短中期趨勢。股價站上 MA20，通常表示短期偏強。")

        st.markdown("**MA60（60日均線）**")
        st.write("代表最近 60 個交易日平均收盤價，通常用來看中期趨勢。股價站上 MA60，通常代表中期趨勢較穩。")

        st.markdown("**RSI（Relative Strength Index）**")
        st.write("用來衡量近期買賣力道。一般來說：RSI > 70 可能偏熱，RSI < 30 可能偏弱或接近超賣。")

        st.markdown("**怎麼一起看**")
        st.write("如果股價在 MA20 和 MA60 之上，且 RSI 在 50 以上，通常代表趨勢偏強；如果跌破均線且 RSI 偏低，通常代表動能較弱。")
        st.markdown("</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()