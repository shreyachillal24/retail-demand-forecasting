import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
from datetime import date


# ---------------------------------------------------
# PATHS (FIXED FOR DEPLOYMENT)
# ---------------------------------------------------
MODEL_PATH = "model/model.pkl"
DATA_PATH  = "data/processed/train_features.csv"


# ---------------------------------------------------
# INPUT PREPARATION (CRITICAL FIX)
# ---------------------------------------------------
def prepare_input(row):
    feature_cols = joblib.load("model/features.pkl")

    # Add missing columns
    for col in feature_cols:
        if col not in row.columns:
            row[col] = 0

    # Enforce correct order
    row = row[feature_cols]

    # Fix categorical types
    cat_cols = ["Store", "Dept", "Type", "IsHoliday"]
    for col in cat_cols:
        if col in row.columns:
            row[col] = row[col].astype(str)

    return row
def prepare_input(row):
    feature_cols = joblib.load("model/features.pkl")

    for col in feature_cols:
        if col not in row.columns:
            row[col] = 0

    row = row[feature_cols]
    return row
# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------
st.set_page_config(
    page_title="Retail Forecast Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------
# CUSTOM CSS FOR PREMIUM UI
# ---------------------------------------------------
st.markdown("""
<style>
    .block-container { padding-top: 2rem; padding-bottom: 2rem; }

    .main-header {
        background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%);
        padding: 2rem 2.5rem; border-radius: 16px; margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .main-header h1 { color: white; font-size: 2.5rem; font-weight: 700; margin: 0; letter-spacing: -0.5px; }
    .main-header p  { color: #bfdbfe; font-size: 1.1rem; margin: 0.5rem 0 0 0; }

    .kpi-card {
        background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
        border-radius: 12px; padding: 1.5rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        border-left: 4px solid #3b82f6; margin-bottom: 1rem; transition: transform 0.2s;
    }
    .kpi-card:hover { transform: translateY(-2px); box-shadow: 0 4px 12px rgba(0,0,0,0.12); }
    .kpi-label  { color: #64748b; font-size: 0.875rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 0.5rem; }
    .kpi-value  { color: #1e293b; font-size: 2rem; font-weight: 700; margin: 0.25rem 0; }
    .kpi-delta  { color: #10b981; font-size: 0.875rem; font-weight: 600; }
    .kpi-delta.negative { color: #ef4444; }

    .section-header {
        color: #1e293b; font-size: 1.5rem; font-weight: 700;
        margin: 2rem 0 1rem 0; padding-bottom: 0.5rem; border-bottom: 2px solid #e2e8f0;
    }

    section[data-testid="stSidebar"] { background: linear-gradient(180deg, #f8fafc 0%, #ffffff 100%) !important; }
    section[data-testid="stSidebar"] > div { background: linear-gradient(180deg, #f8fafc 0%, #ffffff 100%) !important; }

    .sidebar-section-header {
        color: #1e3a8a !important; font-size: 0.95rem !important; font-weight: 700 !important;
        text-transform: uppercase !important; letter-spacing: 0.8px !important;
        margin: 1.75rem 0 0.75rem 0 !important; padding-bottom: 0.5rem !important;
        border-bottom: 2px solid #3b82f6 !important; display: block !important;
    }
    section[data-testid="stSidebar"] label { color: #1e293b !important; font-size: 0.875rem !important; font-weight: 600 !important; margin-bottom: 0.5rem !important; }

    section[data-testid="stSidebar"] [data-baseweb="select"] {
        background-color: #ffffff !important; border: 2px solid #cbd5e1 !important;
        border-radius: 8px !important; font-weight: 500 !important; transition: all 0.2s ease !important;
    }
    section[data-testid="stSidebar"] [data-baseweb="select"]:hover { border-color: #3b82f6 !important; box-shadow: 0 0 0 3px rgba(59,130,246,0.1) !important; }
    section[data-testid="stSidebar"] [data-baseweb="select"] > div { background-color: #ffffff !important; color: #1e293b !important; font-weight: 500 !important; }
    section[data-testid="stSidebar"] [data-baseweb="select"] > div > div { background-color: #ffffff !important; color: #1e293b !important; }
    section[data-testid="stSidebar"] [data-baseweb="select"] [data-baseweb="select-value"] { color: #1e293b !important; background-color: #ffffff !important; }
    section[data-testid="stSidebar"] [data-baseweb="select"] svg { fill: #64748b !important; }
    section[data-testid="stSidebar"] [data-baseweb="menu"] { background-color: #ffffff !important; border: 1px solid #e2e8f0 !important; border-radius: 8px !important; box-shadow: 0 4px 12px rgba(0,0,0,0.1) !important; }
    section[data-testid="stSidebar"] [data-baseweb="menu"] li { color: #1e293b !important; background-color: #ffffff !important; }
    section[data-testid="stSidebar"] [data-baseweb="menu"] li:hover { background-color: #eff6ff !important; color: #1e3a8a !important; }

    section[data-testid="stSidebar"] input[type="text"] {
        background-color: #ffffff !important; border: 2px solid #cbd5e1 !important;
        border-radius: 8px !important; padding: 0.625rem !important;
        color: #1e293b !important; font-weight: 500 !important; transition: all 0.2s ease !important;
    }
    section[data-testid="stSidebar"] input[type="text"]:focus { border-color: #3b82f6 !important; box-shadow: 0 0 0 3px rgba(59,130,246,0.1) !important; outline: none !important; }

    section[data-testid="stSidebar"] .stSlider { padding: 0.5rem 0 1rem 0 !important; }
    section[data-testid="stSidebar"] .stSlider label { color: #1e293b !important; font-weight: 600 !important; font-size: 0.875rem !important; }
    section[data-testid="stSidebar"] .stSlider [data-baseweb="slider"] [role="slider"] { background-color: #3b82f6 !important; width: 18px !important; height: 18px !important; box-shadow: 0 2px 6px rgba(59,130,246,0.4) !important; }

    section[data-testid="stSidebar"] .streamlit-expanderHeader {
        background: linear-gradient(135deg, #eff6ff 0%, #dbeafe 100%) !important;
        border: 2px solid #3b82f6 !important; border-radius: 8px !important;
        color: #1e3a8a !important; font-weight: 700 !important; font-size: 0.9rem !important;
        padding: 0.75rem 1rem !important; transition: all 0.2s ease !important;
        box-shadow: 0 2px 6px rgba(59,130,246,0.15) !important;
    }
    section[data-testid="stSidebar"] .streamlit-expanderHeader:hover { background: linear-gradient(135deg, #dbeafe 0%, #bfdbfe 100%) !important; box-shadow: 0 4px 10px rgba(59,130,246,0.25) !important; }
    section[data-testid="stSidebar"] .streamlit-expanderHeader p,
    section[data-testid="stSidebar"] .streamlit-expanderHeader span,
    section[data-testid="stSidebar"] .streamlit-expanderHeader * { color: #1e3a8a !important; font-weight: 700 !important; }
    section[data-testid="stSidebar"] .streamlit-expanderHeader svg { fill: #1e3a8a !important; color: #1e3a8a !important; }
    section[data-testid="stSidebar"] .streamlit-expanderContent {
        background-color: #f8fafc !important; border: 2px solid #e2e8f0 !important;
        border-top: none !important; border-radius: 0 0 8px 8px !important; padding: 1rem !important;
    }

    section[data-testid="stSidebar"] [data-testid="stExpander"] > details > summary {
        background: linear-gradient(135deg, #eff6ff 0%, #dbeafe 100%) !important;
        border: 2px solid #3b82f6 !important; border-radius: 8px !important;
        color: #1e3a8a !important; font-weight: 700 !important; font-size: 0.9rem !important;
        padding: 0.75rem 1rem !important; box-shadow: 0 2px 6px rgba(59,130,246,0.15) !important;
        transition: all 0.2s ease !important; list-style: none !important;
    }
    section[data-testid="stSidebar"] [data-testid="stExpander"] > details > summary:hover { background: linear-gradient(135deg, #dbeafe 0%, #bfdbfe 100%) !important; box-shadow: 0 4px 10px rgba(59,130,246,0.25) !important; }
    section[data-testid="stSidebar"] [data-testid="stExpander"] > details > summary * { color: #1e3a8a !important; font-weight: 700 !important; }
    section[data-testid="stSidebar"] [data-testid="stExpander"] > details > summary svg { fill: #1e3a8a !important; color: #1e3a8a !important; }
    section[data-testid="stSidebar"] [data-testid="stExpander"] > details[open] > summary { border-radius: 8px 8px 0 0 !important; }
    section[data-testid="stSidebar"] [data-testid="stExpander"] > details > div {
        background-color: #f8fafc !important; border: 2px solid #e2e8f0 !important;
        border-top: none !important; border-radius: 0 0 8px 8px !important; padding: 1rem !important;
    }

    .sidebar-helper  { color: #64748b !important; font-size: 0.75rem !important; font-style: italic !important; margin-top: 0.25rem !important; line-height: 1.4 !important; }
    .sidebar-divider { height: 1px; background: linear-gradient(90deg, transparent 0%, #cbd5e1 50%, transparent 100%); margin: 1.5rem 0; }

    .info-box    { background: linear-gradient(135deg, #eff6ff 0%, #dbeafe 100%); border-left: 4px solid #3b82f6; padding: 1.25rem; border-radius: 8px; margin: 1rem 0; }
    .success-box { background: linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%); border-left: 4px solid #10b981; padding: 1.25rem; border-radius: 8px; margin: 1rem 0; }

    div[data-testid="stMetricValue"] { font-size: 1.8rem; font-weight: 700; color: #1e293b; }
    div[data-testid="stMetricLabel"] { font-size: 0.875rem; font-weight: 600; color: #64748b; text-transform: uppercase; letter-spacing: 0.5px; }

    .divider { height: 2px; background: linear-gradient(90deg, transparent 0%, #e2e8f0 50%, transparent 100%); margin: 2rem 0; }
</style>
""", unsafe_allow_html=True)



# ---------------------------------------------------
# LOAD
# ---------------------------------------------------
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    df["Date"] = pd.to_datetime(df["Date"])
    return df

model = load_model()
df    = load_data()

DATA_MAX_YEAR = int(df["Date"].dt.year.max())

# ---------------------------------------------------
# READABLE NAMES
# ---------------------------------------------------
store_map = {
    1:"Bentonville Central", 2:"Springdale Market",  3:"Fayetteville Hub",
    4:"Rogers Plaza",        5:"Little Rock South",   6:"Dallas North",
    7:"Austin Central",      8:"Houston West",         9:"Tulsa Square",
   10:"Kansas City Hub"
}
for i in sorted(df["Store"].unique()):
    if i not in store_map:
        store_map[i] = f"Regional Store {i}"

dept_map = {
    1:"Grocery",        2:"Electronics",    3:"Clothing",
    4:"Home Essentials",5:"Pharmacy",        6:"Sports",
    7:"Toys",           8:"Beauty",          9:"Automotive",
   10:"Furniture"
}
for i in sorted(df["Dept"].unique()):
    if i not in dept_map:
        dept_map[i] = f"Department {i}"

SPILLOVER_MAP = {
    2: ["Grocery", "Home Essentials", "Beauty"],
    3: ["Beauty", "Grocery"],
    6: ["Grocery", "Pharmacy"],
    7: ["Grocery", "Clothing"],
    8: ["Clothing", "Pharmacy"],
    1: ["Pharmacy", "Home Essentials"],
}

type_labels = {"A":"Large Scale Store","B":"Mid Scale Store","C":"Small Scale Store"}

# ---------------------------------------------------
# HELPER — build one input row for model.predict()
# ---------------------------------------------------
def build_input_row(store, dept, target_date, hist_df,
                    md1=0, md2=0, md3=0, md4=0, md5=0):
    fd       = pd.Timestamp(target_date)
    week_num = int(fd.isocalendar().week)
    year     = fd.year

    min_year  = int(df["Date"].dt.year.min())
    ref_years = [y for y in range(year - 1, year - 4, -1)
                 if min_year <= y <= DATA_MAX_YEAR]

    ref_rows = []
    for ry in ref_years:
        candidates = hist_df[
            (hist_df["Date"].dt.isocalendar().week.astype(int) == week_num) &
            (hist_df["Date"].dt.year == ry)
        ]
        if not candidates.empty:
            ref_rows.append(candidates.iloc[-1])

    if ref_rows:
        combined = pd.concat(ref_rows, axis=1).T
        ref = combined.select_dtypes(include="number").mean()
        first = ref_rows[0]
        for col in ["Type", "Date"]:
            if col in first.index:
                ref[col] = first[col]
    else:
        prior = hist_df[hist_df["Date"] < fd]
        if prior.empty:
            prior = hist_df
        prior = prior.copy()
        prior["week_dist"] = (prior["Date"].dt.isocalendar().week.astype(int) - week_num).abs()
        ref = prior.sort_values("week_dist").iloc[0]

    latest = hist_df.iloc[-1]

    def _get(key, fallback=None):
        try:
            v = ref[key]
            if pd.isna(v):
                raise ValueError
            return float(v)
        except (KeyError, ValueError, TypeError):
            if fallback is not None:
                return float(fallback)
            if key in hist_df.columns:
                return float(hist_df[key].mean())
            return float(hist_df["Weekly_Sales"].mean())

    _fallback_sales = float(hist_df["Weekly_Sales"].mean())
    base_sales = _get("Weekly_Sales", _fallback_sales)

    lag1  = _get("Lag_1",  base_sales)
    lag2  = _get("Lag_2",  base_sales)
    lag4  = _get("Lag_4",  base_sales)
    lag13 = base_sales
    lag52 = base_sales

    rolling4 = _get("Rolling_Mean_4", hist_df["Weekly_Sales"].tail(4).mean())
    rolling8 = float(hist_df["Weekly_Sales"].tail(8).mean())

    same_week_last_year = base_sales

    week_sin = np.sin(2 * np.pi * week_num / 52)
    week_cos = np.cos(2 * np.pi * week_num / 52)

    promo_total = md1 + md2 + md3 + md4 + md5
    promo_flag  = 1 if promo_total > 0 else 0

    return pd.DataFrame({
        "Store":               [store],
        "Dept":                [dept],
        "Type":                [latest["Type"]],
        "IsHoliday":           [False],
        "Size":                [float(latest["Size"])],
        "Temperature":         [_get("Temperature", latest["Temperature"])],
        "Fuel_Price":          [_get("Fuel_Price",  latest["Fuel_Price"])],
        "CPI":                 [_get("CPI",         latest["CPI"])],
        "Unemployment":        [_get("Unemployment",latest["Unemployment"])],
        "Year":                [year],
        "Month":               [fd.month],
        "Week":                [week_num],
        "Week_sin":            [week_sin],
        "Week_cos":            [week_cos],
        "MarkDown1":           [md1],
        "MarkDown2":           [md2],
        "MarkDown3":           [md3],
        "MarkDown4":           [md4],
        "MarkDown5":           [md5],
        "Total_Markdown":      [promo_total],
        "Promo_Flag":          [promo_flag],
        "Lag_1":               [lag1],
        "Lag_2":               [lag2],
        "Lag_4":               [lag4],
        "Lag_13":              [lag13],
        "Lag_52":              [lag52],
        "Rolling_Mean_4":      [rolling4],
        "Same_Week_Last_Year": [same_week_last_year],
        "Rolling_Mean_8":      [rolling8],
    })

# ==================== SIDEBAR ====================
with st.sidebar:
    st.markdown("""
    <div style='text-align: center; padding: 1.5rem 0.5rem;
                background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%);
                border-radius: 12px; margin-bottom: 1.5rem; box-shadow: 0 4px 6px rgba(0,0,0,0.1);'>
        <div style='font-size: 2.5rem; margin-bottom: 0.5rem;'>⚙️</div>
        <h2 style='color: white; margin: 0; font-size: 1.4rem; font-weight: 700;'>Forecast Controls</h2>
        <p style='color: #bfdbfe; font-size: 0.8rem; margin: 0.5rem 0 0 0;'>Configure your demand forecast parameters</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div class='sidebar-section-header'>🏪 Store & Department</div>", unsafe_allow_html=True)
    st.markdown("<p class='sidebar-helper'>Select the retail location and product category</p>", unsafe_allow_html=True)

    store_ids  = sorted(df["Store"].unique())
    store_name = st.selectbox(
        "Retail Location",
        [store_map[i] for i in store_ids],
        help="Choose the store location for forecasting"
    )
    store = [k for k, v in store_map.items() if v == store_name][0]

    available_depts = sorted(df[df["Store"] == store]["Dept"].unique())
    dept_name       = st.selectbox(
        "Product Department",
        [dept_map[i] for i in available_depts],
        help="Select the product category to forecast"
    )
    dept = [k for k, v in dept_map.items() if v == dept_name][0]

    st.markdown("<div class='sidebar-divider'></div>", unsafe_allow_html=True)
    st.markdown("<div class='sidebar-section-header'>📅 Forecast Configuration</div>", unsafe_allow_html=True)
    st.markdown("<p class='sidebar-helper'>Set the forecast date and profit margin</p>", unsafe_allow_html=True)

    forecast_date = st.date_input(
        "Target Forecast Date",
        value=date(2013, 4, 16),
        min_value=date(2010, 3, 5),
        max_value=date(2015, 12, 31),
        help="Select the specific date you want to forecast sales for"
    )

    margin = st.slider(
        "Profit Margin (%)",
        min_value=5, max_value=40, value=18,
        help="Expected profit margin percentage for this department"
    )

    st.markdown("<div class='sidebar-divider'></div>", unsafe_allow_html=True)

    with st.expander("🎯 **Promotional Campaigns**", expanded=False):
        st.markdown("""
        <p style='color: #475569; font-size: 0.85rem; margin-bottom: 1rem; line-height: 1.5;'>
            Set markdown values to simulate promotional impact on sales forecast.
        </p>
        """, unsafe_allow_html=True)
        md1 = st.slider("Seasonal Promotion",   0, 10000, 0, step=100, help="Holiday and seasonal discount campaigns")
        md2 = st.slider("Electronics Discount",  0, 10000, 0, step=100, help="Technology product promotions")
        md3 = st.slider("Clothing Clearance",    0, 10000, 0, step=100, help="Apparel markdown and clearance sales")
        md4 = st.slider("Storewide Campaign",    0, 10000, 0, step=100, help="Cross-category promotional campaigns")
        md5 = st.slider("Holiday Promotion",     0, 10000, 0, step=100, help="Special event and holiday discounts")

# ---------------------------------------------------
# FILTER HISTORY
# ---------------------------------------------------
hist = df[(df["Store"] == store) & (df["Dept"] == dept)].sort_values("Date")
if hist.empty:
    hist = df[df["Store"] == store].sort_values("Date")
if hist.empty:
    hist = df.sort_values("Date")

latest = hist.iloc[-1]
fd     = pd.Timestamp(forecast_date)

# ---------------------------------------------------
# MONTHLY FORECAST (for KPIs + table)
# ---------------------------------------------------
selected_year  = fd.year
selected_month = fd.month

monthly_points = []
for m in range(1, selected_month + 1):
    try:
        fdate = pd.Timestamp(date(selected_year, m, 16))
    except ValueError:
        fdate = pd.Timestamp(date(selected_year, m, 15))
    if m == selected_month:
        fdate = fd
    row = build_input_row(store, dept, fdate, hist, md1, md2, md3, md4, md5)
    row = prepare_input(row)
    pred_val = float(model.predict(row)[0])
    monthly_points.append({"Date": fdate, "Forecast": pred_val})

monthly_forecast_df = pd.DataFrame(monthly_points)

pred        = monthly_forecast_df.iloc[-1]["Forecast"]
profit      = pred * margin / 100
promo_total = md1 + md2 + md3 + md4 + md5
promo_active_label = "Yes" if promo_total > 0 else "No"

week_num = int(fd.isocalendar().week)
same_week_rows = hist[
    (hist["Date"].dt.isocalendar().week.astype(int) == week_num) &
    (hist["Date"].dt.year == selected_year - 1)
]
last_year_sales = (
    float(same_week_rows["Weekly_Sales"].mean())
    if not same_week_rows.empty
    else float(hist["Weekly_Sales"].tail(8).mean())
)
rolling_mean_8 = float(hist["Weekly_Sales"].tail(8).mean())

# ---------------------------------------------------
# PROMOTION HISTORY
# ---------------------------------------------------
promo_col_map = {
    "MarkDown1": "Seasonal Promotion",
    "MarkDown2": "Electronics Discount",
    "MarkDown3": "Clothing Clearance",
    "MarkDown4": "Storewide Campaign",
    "MarkDown5": "Holiday Promotion",
}
PROMO_THRESHOLDS = {
    "MarkDown1": 5544, "MarkDown2": 538, "MarkDown3": 163,
    "MarkDown4": 1667, "MarkDown5": 3359,
}

store_hist_md = (
    df[df["Store"] == store]
    .drop_duplicates(subset=["Date"])
    .sort_values("Date")
)
ly_week_rows = store_hist_md[
    (store_hist_md["Date"].dt.isocalendar().week.astype(int).between(week_num - 1, week_num + 1)) &
    (store_hist_md["Date"].dt.year == selected_year - 1)
]

last_year_promos = []
ly_promo_md_vals = {}

if not ly_week_rows.empty:
    for col, label in promo_col_map.items():
        if col in ly_week_rows.columns:
            mean_val  = float(ly_week_rows[col].mean())
            threshold = PROMO_THRESHOLDS.get(col, 100)
            if mean_val >= threshold:
                last_year_promos.append(label)
                ly_promo_md_vals[col] = mean_val

whatif_pred   = None
whatif_profit = None
whatif_lift   = None

if last_year_promos and ly_promo_md_vals:
    wmd1 = ly_promo_md_vals.get("MarkDown1", 0)
    wmd2 = ly_promo_md_vals.get("MarkDown2", 0)
    wmd3 = ly_promo_md_vals.get("MarkDown3", 0)
    wmd4 = ly_promo_md_vals.get("MarkDown4", 0)
    wmd5 = ly_promo_md_vals.get("MarkDown5", 0)
    whatif_row    = build_input_row(store, dept, fd, hist, wmd1, wmd2, wmd3, wmd4, wmd5)
    whatif_row    = prepare_input(whatif_row)
    whatif_pred   = float(model.predict(whatif_row)[0])
    whatif_profit = whatif_pred * margin / 100
    whatif_lift   = whatif_pred - pred

# ---------------------------------------------------
# HEADER
# ---------------------------------------------------
st.markdown("""
<div class='main-header'>
    <h1>📊 Retail Demand Forecast Dashboard</h1>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------
# RECOMMENDATION BOX
# ---------------------------------------------------
active_promos = []
promo_labels  = {
    "md1": "Seasonal Promotion",  "md2": "Electronics Discount",
    "md3": "Clothing Clearance",  "md4": "Storewide Campaign",
    "md5": "Holiday Promotion"
}
promo_vals = {"md1": md1, "md2": md2, "md3": md3, "md4": md4, "md5": md5}
for k, v in promo_vals.items():
    if v > 0:
        active_promos.append(promo_labels[k])

demand_signal = "High" if pred > rolling_mean_8 else "Normal"
yoy_change    = ((pred - last_year_sales) / last_year_sales * 100) if last_year_sales > 0 else 0

spillover_depts = SPILLOVER_MAP.get(dept, [])
spillover_text  = ""
if active_promos and spillover_depts:
    spillover_text = (
        f"<div style='margin-top: 0.75rem; padding: 0.75rem; background: rgba(255,255,255,0.1); border-radius: 6px;'>"
        f"<strong>🔄 Spillover Effect:</strong> Promotion in <strong>{dept_map.get(dept)}</strong> drives foot traffic to: "
        f"<strong>{', '.join(spillover_depts)}</strong>. Ensure baseline stock availability in these departments."
        f"</div>"
    )

if active_promos:
    primary_action = (
        f"<strong>📦 Primary Action:</strong> Increase {dept_map.get(dept)} inventory — "
        f"active promotion will drive direct demand.<br>"
    )
    if spillover_text:
        primary_action += spillover_text
elif demand_signal == "High":
    primary_action = (
        f"<strong>📈 Demand Alert:</strong> Sales tracking above 8-week average. "
        f"Consider pre-stocking {dept_map.get(dept)} for week of {fd.strftime('%b %d')}."
    )
else:
    primary_action = (
        f"<strong>✓ Normal Conditions:</strong> Demand is in normal range. "
        f"Maintain standard stock levels for {dept_map.get(dept)}."
    )

if last_year_promos:
    promo_list_str   = ", ".join(last_year_promos)
    promo_history_html = (
        "<div style='margin-top: 1rem; padding-top: 1rem; border-top: 1px solid rgba(255,255,255,0.2);'>"
        f"<p style='color: #ffd97d; margin: 0 0 0.5rem 0; font-size: 0.9rem;'>"
        f"📅 <strong>Historical Context:</strong> Last year same week — <strong>{promo_list_str}</strong> was active.</p>"
    )
    if whatif_pred is not None and whatif_lift is not None:
        lift_sign  = "+" if whatif_lift >= 0 else ""
        lift_color = "#86efac" if whatif_lift >= 0 else "#fca5a5"
        promo_history_html += (
            f"<div style='background: rgba(255,255,255,0.1); padding: 0.875rem; border-radius: 8px; margin-top: 0.5rem;'>"
            f"<p style='margin: 0; color: white; font-size: 0.95rem;'>"
            f"💡 <strong>Scenario: Repeat Last Year's Campaign</strong></p>"
            f"<p style='margin: 0.5rem 0 0 0; color: #e0e7ff;'>"
            f"Projected Sales: <strong style='color: white;'>${whatif_pred:,.0f}</strong> "
            f"<span style='color: {lift_color}; font-weight: 600;'>({lift_sign}${whatif_lift:,.0f})</span><br>"
            f"Projected Profit: <strong style='color: white;'>${whatif_profit:,.0f}</strong> at {margin}% margin"
            f"</p></div>"
        )
    promo_history_html += "</div>"
else:
    promo_history_html = (
        "<div style='margin-top: 1rem; padding-top: 1rem; border-top: 1px solid rgba(255,255,255,0.2);'>"
        "<p style='color: rgba(255,255,255,0.5); margin: 0; font-size: 0.875rem;'>"
        "📅 No promotional activity during this week last year.</p></div>"
    )

if demand_signal == "High" or active_promos:
    box_bg       = "linear-gradient(135deg, #065f46 0%, #047857 100%)"
    border_color = "#10b981"
    badge_bg     = "#86efac"
    badge_text   = "#065f46"
else:
    box_bg       = "linear-gradient(135deg, #1e3a8a 0%, #2563eb 100%)"
    border_color = "#60a5fa"
    badge_bg     = "#bfdbfe"
    badge_text   = "#1e3a8a"

st.markdown(
    f"<div style='background:{box_bg}; padding:1.75rem; border-radius:14px; margin-bottom:2rem;"
    f"border:2px solid {border_color}; box-shadow:0 4px 12px rgba(0,0,0,0.15);'>"
    f"<div style='display:flex; align-items:center; justify-content:space-between; margin-bottom:1rem;'>"
    f"<h3 style='margin:0; color:white; font-size:1.4rem;'>📌 Inventory Action Plan</h3>"
    f"<span style='background:{badge_bg}; color:{badge_text}; padding:0.375rem 0.875rem;"
    f"border-radius:20px; font-size:0.875rem; font-weight:700;'>{demand_signal} Demand</span>"
    f"</div>"
    f"<div style='background:rgba(255,255,255,0.1); padding:1rem; border-radius:8px; margin-bottom:1rem;'>"
    f"<p style='margin:0; color:#e0e7ff; font-size:0.95rem;'>"
    f"<strong style='color:white;'>{store_map.get(store)}</strong> • "
    f"<strong style='color:white;'>{dept_map.get(dept)}</strong> • "
    f"{fd.strftime('%B %d, %Y')}<br>"
    f"<span style='font-size:0.875rem;'>YoY Change: "
    f"<strong style='color:{'#86efac' if yoy_change >= 0 else '#fca5a5'};'>"
    f"{'+' if yoy_change >= 0 else ''}{yoy_change:.1f}%</strong></span>"
    f"</p></div>"
    f"<p style='color:white; margin:0; font-size:1rem; line-height:1.6;'>{primary_action}</p>"
    f"{promo_history_html}"
    f"</div>",
    unsafe_allow_html=True
)

# ---------------------------------------------------
# KPI CARDS
# ---------------------------------------------------
st.markdown("<h3 class='section-header'>📊 Key Performance Indicators</h3>", unsafe_allow_html=True)

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.markdown(f"""
    <div class='kpi-card' style='border-left-color:#3b82f6;'>
        <div class='kpi-label'>💰 Forecasted Sales</div>
        <div class='kpi-value'>${pred:,.0f}</div>
        <div style='color:#64748b; font-size:0.8rem; margin-top:0.25rem;'>Weekly projection</div>
    </div>""", unsafe_allow_html=True)

with c2:
    delta_color = "#10b981" if yoy_change >= 0 else "#ef4444"
    delta_class = "" if yoy_change >= 0 else "negative"
    st.markdown(f"""
    <div class='kpi-card' style='border-left-color:{delta_color};'>
        <div class='kpi-label'>📅 Last Year Same Week</div>
        <div class='kpi-value'>${last_year_sales:,.0f}</div>
        <div class='kpi-delta {delta_class}'>
            {'▲' if yoy_change >= 0 else '▼'} {abs(yoy_change):.1f}% YoY
        </div>
    </div>""", unsafe_allow_html=True)

with c3:
    promo_color = "#f59e0b" if promo_total > 0 else "#64748b"
    st.markdown(f"""
    <div class='kpi-card' style='border-left-color:{promo_color};'>
        <div class='kpi-label'>🎯 Promotion Status</div>
        <div class='kpi-value' style='font-size:1.6rem;'>{promo_active_label}</div>
        <div style='color:#64748b; font-size:0.8rem; margin-top:0.25rem;'>
            ${promo_total:,.0f} total markdown
        </div>
    </div>""", unsafe_allow_html=True)

with c4:
    st.markdown(f"""
    <div class='kpi-card' style='border-left-color:#10b981;'>
        <div class='kpi-label'>💵 Estimated Profit</div>
        <div class='kpi-value' style='color:#10b981;'>${profit:,.0f}</div>
        <div style='color:#64748b; font-size:0.8rem; margin-top:0.25rem;'>At {margin}% margin</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

# ---------------------------------------------------
# INSIGHTS SECTION
# ---------------------------------------------------
st.markdown("<h3 class='section-header'>💡 Business Insights</h3>", unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    store_type = type_labels.get(str(latest["Type"]), str(latest["Type"]))
    st.markdown(f"""
    <div class='info-box'>
        <h4 style='margin:0 0 0.75rem 0; color:#1e40af; font-size:1.1rem;'>🏪 Location Overview</h4>
        <p style='margin:0; color:#1e293b; line-height:1.7;'>
            <strong>Store:</strong> {store_map.get(store)}<br>
            <strong>Department:</strong> {dept_map.get(dept)}<br>
            <strong>Store Type:</strong> {store_type}<br>
            <strong>Promotion Value:</strong> ${promo_total:,.0f}<br>
            <strong>YoY Variance:</strong>
            <span style='color:{"#10b981" if (pred - last_year_sales) >= 0 else "#ef4444"}; font-weight:600;'>
            ${pred - last_year_sales:+,.0f}</span>
        </p>
    </div>""", unsafe_allow_html=True)

with col2:
    store_df  = df[df["Store"] == store]
    dept_avg  = store_df.groupby("Dept")["Weekly_Sales"].mean().sort_values(ascending=False)
    top_dept  = dept_avg.index[0]
    top_sales = dept_avg.iloc[0]
    sel_sales = float(dept_avg.loc[dept]) if dept in dept_avg.index else 0.0
    rank      = list(dept_avg.index).index(dept) + 1 if dept in dept_avg.index else "N/A"
    gap       = top_sales - sel_sales

    st.markdown(f"""
    <div class='success-box'>
        <h4 style='margin:0 0 0.75rem 0; color:#065f46; font-size:1.1rem;'>📈 Department Performance</h4>
        <p style='margin:0; color:#1e293b; line-height:1.7;'>
            <strong>Department:</strong> {dept_map.get(dept)}<br>
            <strong>Store Rank:</strong> #{rank}<br>
            <strong>Avg Weekly Sales:</strong> ${sel_sales:,.0f}<br>
            <strong>Top Department:</strong> {dept_map.get(top_dept)} (${top_sales:,.0f})<br>
            <strong>Gap to Leader:</strong> <span style='color:#dc2626; font-weight:600;'>-${gap:,.0f}</span>
        </p>
    </div>""", unsafe_allow_html=True)

st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

# ---------------------------------------------------
# CHART SECTION
# ---------------------------------------------------
st.markdown("<h3 class='section-header'>📈 Sales Trend & Forecast Trajectory</h3>", unsafe_allow_html=True)

# Historical actuals: last 3 months of data available
chart_start  = fd - pd.DateOffset(months=3)
hist_chart   = hist[hist["Date"] >= chart_start].copy()

# Biweekly forecast path from 3 months back to selected date
biweekly_dates = []
current = chart_start
while current <= fd:
    biweekly_dates.append(current)
    current += pd.DateOffset(weeks=2)
if not biweekly_dates or biweekly_dates[-1] != fd:
    biweekly_dates.append(fd)

biweekly_points = []
for fdate in biweekly_dates:
    row      = build_input_row(store, dept, fdate, hist, md1, md2, md3, md4, md5)
    row      = prepare_input(row)
    pred_val = float(model.predict(row)[0])
    biweekly_points.append({"Date": fdate, "Forecast": pred_val})

chart_forecast_df = pd.DataFrame(biweekly_points)

fig = go.Figure()

# Actual sales trace
if not hist_chart.empty:
    fig.add_trace(go.Scatter(
        x=hist_chart["Date"],
        y=hist_chart["Weekly_Sales"],
        mode="lines+markers",
        name="Actual Sales",
        line=dict(color="#3b82f6", width=3, shape="spline"),
        marker=dict(size=6, color="#3b82f6")
    ))

    # Bridge connector between last actual and first forecast point
    fig.add_trace(go.Scatter(
        x=[hist_chart["Date"].iloc[-1], chart_forecast_df["Date"].iloc[0]],
        y=[hist_chart["Weekly_Sales"].iloc[-1], chart_forecast_df["Forecast"].iloc[0]],
        mode="lines",
        line=dict(color="#1e40af", width=2, dash="dot"),
        showlegend=False
    ))

# Forecast path
fig.add_trace(go.Scatter(
    x=chart_forecast_df["Date"],
    y=chart_forecast_df["Forecast"],
    mode="lines+markers",
    name="AI Forecast",
    line=dict(color="#1e40af", width=3, dash="dash", shape="spline"),
    marker=dict(size=10, symbol="diamond", color="#1e40af",
                line=dict(color="#93c5fd", width=2))
))

# Annotations on all forecast points
for _, row in chart_forecast_df.iterrows():
    fig.add_annotation(
        x=row["Date"], y=row["Forecast"],
        text=f"${row['Forecast']:,.0f}",
        showarrow=False,
        yshift=20,
        font=dict(color="#1e40af", size=10, family="Arial Black"),
        bgcolor="rgba(255,255,255,0.75)",
        borderpad=2
    )

fig.update_layout(
    template="plotly_white",
    height=480,
    margin=dict(l=20, r=20, t=40, b=20),
    xaxis_title="Timeline",
    yaxis_title="Weekly Sales ($)",
    hovermode="x unified",
    legend=dict(
        orientation="h", yanchor="bottom", y=1.02,
        xanchor="right", x=1,
        bgcolor="rgba(255,255,255,0.95)",
        bordercolor="#bfdbfe", borderwidth=1
    ),
    plot_bgcolor="#f0f7ff",
    paper_bgcolor="#ffffff",
    font=dict(family="Arial, sans-serif")
)
fig.update_yaxes(
    tickprefix="$", gridcolor="#dbeafe", zerolinecolor="#bfdbfe",
    linecolor="#1e293b", linewidth=2, tickcolor="#1e293b",
    tickfont=dict(color="#1e293b"), title_font=dict(color="#1e293b")
)
fig.update_xaxes(
    gridcolor="#dbeafe", zerolinecolor="#bfdbfe",
    linecolor="#1e293b", linewidth=2, tickcolor="#1e293b",
    tickfont=dict(color="#1e293b"), title_font=dict(color="#1e293b")
)

st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})

st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

# ---------------------------------------------------
# FORECAST TABLE
# ---------------------------------------------------
with st.expander("📋 **Monthly Forecast Breakdown**", expanded=False):
    st.markdown("<small style='color:#64748b;'>Detailed month-by-month projections with profit estimates</small>", unsafe_allow_html=True)
    st.markdown("<div style='margin:0.75rem 0;'></div>", unsafe_allow_html=True)

    display_df = monthly_forecast_df.copy()
    display_df["Month"]           = display_df["Date"].dt.strftime("%B %Y")
    display_df["Predicted Sales"] = display_df["Forecast"].apply(lambda x: f"${x:,.0f}")
    display_df["Est. Profit"]     = display_df["Forecast"].apply(lambda x: f"${x * margin / 100:,.0f}")
    st.dataframe(
        display_df[["Month", "Predicted Sales", "Est. Profit"]].reset_index(drop=True),
        use_container_width=True, hide_index=True
    )

# ---------------------------------------------------
# DEBUG SECTION
# ---------------------------------------------------
with st.expander("🔧 **Model Input Features** (Debug View)", expanded=False):
    st.markdown("<small style='color:#64748b;'>Raw feature values used for prediction model</small>", unsafe_allow_html=True)
    st.markdown("<div style='margin:0.75rem 0;'></div>", unsafe_allow_html=True)
    debug_row = build_input_row(store, dept, fd, hist, md1, md2, md3, md4, md5)
    st.dataframe(debug_row, use_container_width=True)

# Footer
st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
st.markdown("""
<div style='text-align:center; color:#94a3b8; font-size:0.875rem; padding:1rem 0;'>
    <p style='margin:0;'>Powered by AI-driven demand forecasting • Built with Streamlit</p>
</div>
""", unsafe_allow_html=True)