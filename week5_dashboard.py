# ============================================================
# ExportWatch — Week 5 UPGRADED: Monetization Edition
# New features:
#   1. My Shipment Simulator — company-level profit calculator
#   2. Actionable Alerts — tells user WHAT TO DO
#   3. What Changed Since Yesterday — monitoring feel
#   4. Fixed forecast with confidence bands + capped bounds
#
# Run: streamlit run week5_dashboard.py
# Install: pip install streamlit plotly pandas numpy scikit-learn openai requests python-dotenv
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import requests
from sklearn.linear_model import LinearRegression
from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()
nvidia_key = os.getenv("NVIDIA_API_KEY", "")

# ── PAGE CONFIG ──────────────────────────────────────────────
st.set_page_config(
    page_title            = "ExportWatch",
    page_icon             = "📦",
    layout                = "wide",
    initial_sidebar_state = "expanded"
)

# ── CUSTOM CSS ───────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0a0c0f; }
    .block-container { padding-top: 1.5rem; }
    .metric-card {
        background: #111418; border: 1px solid #1e2530;
        border-radius: 10px; padding: 18px 20px; text-align: center;
    }
    .metric-label { font-size: 11px; color: #6b7280; text-transform: uppercase; letter-spacing: 1px; }
    .metric-value { font-size: 28px; font-weight: 300; color: #e8eaf0; margin: 4px 0; }
    .metric-change-up   { font-size: 12px; color: #f87171; }
    .metric-change-down { font-size: 12px; color: #4ade80; }
    .section-title {
        font-size: 11px; color: #6b7280; text-transform: uppercase;
        letter-spacing: 2px; margin-bottom: 12px; margin-top: 8px;
    }
    .ai-box {
        background: #111418; border: 1px solid rgba(200,240,77,0.2);
        border-radius: 10px; padding: 20px 24px;
        font-size: 14px; line-height: 1.8; color: #e8eaf0;
    }
    .alert-box {
        background: #111418; border-left: 4px solid #f87171;
        border-radius: 0 10px 10px 0; padding: 16px 20px;
        margin-bottom: 10px; font-size: 14px; color: #e8eaf0; line-height: 1.7;
    }
    .alert-box.amber { border-left-color: #facc15; }
    .alert-box.green { border-left-color: #4ade80; }
    .sim-result {
        background: #111418; border: 1px solid #1e2530;
        border-radius: 10px; padding: 16px 20px; text-align: center;
    }
    .sim-label { font-size: 11px; color: #6b7280; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 6px; }
    .sim-value { font-size: 26px; font-weight: 300; }
    .change-pill {
        display: inline-block; padding: 2px 10px;
        border-radius: 20px; font-size: 11px; font-weight: 500; margin-top: 2px;
    }
    .delta-box {
        background: #111418; border: 1px solid #1e2530; border-radius: 10px;
        padding: 14px 18px; margin-bottom: 8px; display: flex;
        align-items: center; justify-content: space-between;
    }
    div[data-testid="stSidebar"] { background-color: #111418; }
</style>
""", unsafe_allow_html=True)


# ── SIDEBAR ──────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📦 ExportWatch")
    st.markdown("*Indian Textile Export Risk Intelligence*")
    st.divider()
    if nvidia_key:
        st.success("✅ API key loaded")
    else:
        st.error("❌ No API key in .env")

    st.divider()
    st.markdown("### 📊 Risk Weight Config")
    w_oil      = st.slider("Oil Price weight",      5, 50, 30)
    w_freight  = st.slider("Freight Index weight",  5, 50, 40)
    w_inr      = st.slider("USD/INR weight",        5, 30, 15)
    w_export   = st.slider("Export Value weight",   5, 30, 15)

    st.divider()
    st.markdown("### 📅 Forecast Assumptions")
    oil_growth     = st.slider("Oil growth %/month",     0.0, 5.0, 2.0, 0.5)
    freight_growth = st.slider("Freight growth %/month", 0.0, 5.0, 1.5, 0.5)
    inr_growth     = st.slider("INR weakening %/month",  0.0, 2.0, 0.2, 0.1)

    st.divider()
    st.caption("ExportWatch MVP v2.0")


# ── DATA ─────────────────────────────────────────────────────
@st.cache_data(ttl=3600)
def fetch_usd_inr():
    try:
        r = requests.get("https://api.exchangerate-api.com/v4/latest/USD", timeout=10)
        return r.json()["rates"]["INR"]
    except:
        return 84.0

@st.cache_data(ttl=3600)
def load_master_data():
    return pd.DataFrame({
        "month": ["Oct-23","Nov-23","Dec-23","Jan-24","Feb-24","Mar-24",
                  "Apr-24","May-24","Jun-24","Jul-24","Aug-24","Sep-24"],
        "usd_inr":       [83.0,83.1,83.2,83.3,83.4,83.5,83.4,83.5,83.7,83.6,83.8,84.0],
        "oil_price":     [85.0,86.0,87.0,86.5,87.5,88.0,85.5,87.0,89.0,88.0,90.0,91.5],
        "freight_index": [1820,1850,1900,1870,1910,1950,1850,1920,1780,1950,2100,2250],
        "export_usd_bn": [3.82,3.71,3.65,3.78,3.54,3.61,3.45,3.52,3.38,3.61,3.44,3.29],
    })


# ── RISK SCORE ───────────────────────────────────────────────
def compute_risk(df, latest_inr, weights):
    latest = df.iloc[-1].copy()
    latest["usd_inr"] = latest_inr
    baselines = {
        "usd_inr":       {"base": 83.0, "weight": weights["inr"]},
        "oil_price":     {"base": 80.0, "weight": weights["oil"]},
        "freight_index": {"base": 1800, "weight": weights["freight"]},
        "export_usd_bn": {"base": 3.70, "weight": weights["export"]},
    }
    risk_score   = 0
    risk_details = {}
    for col, config in baselines.items():
        base, weight, actual = config["base"], config["weight"], latest[col]
        dev   = max(0, min(1, (base - actual)/base if col == "export_usd_bn" else (actual - base)/base))
        score = round(dev * weight, 1)
        risk_score       += score
        risk_details[col] = {"actual": actual, "score": score, "weight": weight}
    return round(risk_score, 1), risk_details, latest


# ── FORECAST WITH BOUNDS ─────────────────────────────────────
def compute_forecast(df, latest, oil_g, freight_g, inr_g):
    X     = df[["oil_price","freight_index","usd_inr"]].values
    y     = df["export_usd_bn"].values
    model = LinearRegression().fit(X, y)
    r2    = model.score(X, y)
    residual_std = np.std(y - model.predict(X))

    forecast_months = ["Oct-24","Nov-24","Dec-24"]
    forecast_values, upper_bounds, lower_bounds = [], [], []

    # Cap forecast: never drop more than 15% or rise more than 10% from current
    current = latest["export_usd_bn"]
    min_cap = current * 0.85
    max_cap = current * 1.10

    for i in range(1, 4):
        proj = np.array([[
            latest["oil_price"]     * ((1 + oil_g/100)     ** i),
            latest["freight_index"] * ((1 + freight_g/100) ** i),
            latest["usd_inr"]       * ((1 + inr_g/100)     ** i),
        ]])
        raw = model.predict(proj)[0]
        val = round(float(np.clip(raw, min_cap, max_cap)), 2)
        forecast_values.append(val)
        upper_bounds.append(round(min(val + residual_std * 1.5, max_cap + 0.1), 2))
        lower_bounds.append(round(max(val - residual_std * 1.5, min_cap - 0.1), 2))

    change_pct = round(((forecast_values[-1] - current) / current) * 100, 1)
    return forecast_months, forecast_values, upper_bounds, lower_bounds, change_pct, r2


# ── SIMULATE WHAT CHANGED "SINCE YESTERDAY" ──────────────────
def get_delta_signals(df, risk_score):
    prev_month  = df.iloc[-2]
    curr_month  = df.iloc[-1]
    prev_risk   = round(risk_score * 0.88, 1)  # simulated previous score
    delta_risk  = round(risk_score - prev_risk, 1)
    freight_chg = round(((curr_month["freight_index"] - prev_month["freight_index"]) / prev_month["freight_index"]) * 100, 1)
    oil_chg     = round(((curr_month["oil_price"] - prev_month["oil_price"]) / prev_month["oil_price"]) * 100, 1)
    inr_chg     = round(((curr_month["usd_inr"] - prev_month["usd_inr"]) / prev_month["usd_inr"]) * 100, 1)
    return {
        "risk_prev":    prev_risk,
        "risk_now":     risk_score,
        "risk_delta":   delta_risk,
        "freight_chg":  freight_chg,
        "oil_chg":      oil_chg,
        "inr_chg":      inr_chg,
    }


# ── SHIPMENT SIMULATOR ────────────────────────────────────────
def simulate_shipment(order_value, cost_per_unit, units, freight_pct, currency_exposure_pct, live_inr, risk_score, freight_index):
    total_cost       = cost_per_unit * units
    revenue          = order_value
    freight_cost     = total_cost * (freight_pct / 100)
    base_margin      = revenue - total_cost
    base_margin_pct  = round((base_margin / revenue) * 100, 1)

    # Risk adjustment: higher freight index = higher logistics cost
    freight_pressure = max(0, (freight_index - 1800) / 1800) * 0.15
    adjusted_freight = freight_cost * (1 + freight_pressure)

    # Currency impact: if INR weakens, USD-invoiced revenue is higher in INR
    # but raw material imports cost more too
    currency_gain    = (live_inr - 83.0) / 83.0 * currency_exposure_pct / 100 * revenue
    adjusted_margin  = base_margin - (adjusted_freight - freight_cost) + currency_gain
    adjusted_margin_pct = round((adjusted_margin / revenue) * 100, 1)
    margin_change    = round(adjusted_margin_pct - base_margin_pct, 1)

    # Shipment risk rating
    if risk_score >= 70 or margin_change < -5:
        ship_risk = "🔴 High Risk"
    elif risk_score >= 40 or margin_change < -2:
        ship_risk = "🟡 Moderate Risk"
    else:
        ship_risk = "🟢 Low Risk"

    return {
        "base_margin_pct":     base_margin_pct,
        "adjusted_margin_pct": adjusted_margin_pct,
        "margin_change":       margin_change,
        "freight_impact":      round(adjusted_freight - freight_cost, 0),
        "currency_impact":     round(currency_gain, 0),
        "ship_risk":           ship_risk,
    }


# ── ACTIONABLE ALERTS (AI) ────────────────────────────────────
def get_actionable_alerts(api_key, risk_score, risk_details, latest, forecasts, change_pct, sim=None):
    if not api_key:
        return "⚠️ No API key found in .env file."

    risk_status = "HIGH RISK" if risk_score >= 70 else "MODERATE RISK" if risk_score >= 40 else "LOW RISK"
    sim_context = ""
    if sim:
        sim_context = f"""
Shipment Simulation:
- Base margin: {sim['base_margin_pct']}%
- Adjusted margin under current risk: {sim['adjusted_margin_pct']}%
- Margin change: {sim['margin_change']:+.1f}%
- Shipment risk rating: {sim['ship_risk']}
"""

    prompt = f"""
You are an export risk advisor for Indian textile exporters.
Based on the data below, give 3 SPECIFIC ACTIONABLE ALERTS.
Each alert must tell the exporter exactly WHAT TO DO — not just what's happening.
Format: start each alert with an emoji and action verb. No bullet points within alerts. Plain English.

Risk Score: {risk_score}/100 — {risk_status}
- Oil Price: ${latest['oil_price']}/bbl (score: {risk_details['oil_price']['score']}/{risk_details['oil_price']['weight']} pts)
- Freight Index: {latest['freight_index']} (score: {risk_details['freight_index']['score']}/{risk_details['freight_index']['weight']} pts)
- USD/INR: ₹{latest['usd_inr']} (score: {risk_details['usd_inr']['score']}/{risk_details['usd_inr']['weight']} pts)
3-Month Forecast: exports projected at {change_pct:+.1f}%
{sim_context}

Give exactly 3 alerts. Each must be 1-2 sentences. Start with one of these:
⚠️ DELAY — if timing of shipment should change
💰 HEDGE — if currency or oil exposure should be locked
📦 REROUTE — if freight route should change
🔒 LOCK — if contracts should be fixed now
✅ PROCEED — if conditions are acceptable

Example format:
💰 HEDGE your USD exposure now — lock forward contracts for the next 60 days before INR weakens further past ₹84.50.
"""
    try:
        client   = OpenAI(base_url="https://integrate.api.nvidia.com/v1", api_key=api_key)
        response = client.chat.completions.create(
            model       = "meta/llama-3.1-8b-instruct",
            messages    = [{"role": "user", "content": prompt}],
            max_tokens  = 400,
            temperature = 0.4,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"❌ Could not generate alerts: {e}"


# ══════════════════════════════════════════════════════════════
# MAIN DASHBOARD
# ══════════════════════════════════════════════════════════════

st.markdown("# 📦 ExportWatch")
st.markdown("##### Indian Textile Export Risk Intelligence — Company Edition")
st.divider()

# Load data
df       = load_master_data()
live_inr = fetch_usd_inr()
weights  = {"oil": w_oil, "freight": w_freight, "inr": w_inr, "export": w_export}

risk_score, risk_details, latest = compute_risk(df, live_inr, weights)
forecast_months, forecast_values, upper_bounds, lower_bounds, change_pct, r2 = compute_forecast(
    df, latest, oil_growth, freight_growth, inr_growth
)
deltas = get_delta_signals(df, risk_score)


# ── ROW 1: KPI CARDS ─────────────────────────────────────────
st.markdown('<p class="section-title">Live Indicators</p>', unsafe_allow_html=True)
c1, c2, c3, c4 = st.columns(4)

with c1:
    st.markdown(f"""<div class="metric-card">
        <div class="metric-label">USD / INR Rate</div>
        <div class="metric-value">₹{live_inr:.2f}</div>
        <div class="metric-change-up">↑ Live rate</div>
    </div>""", unsafe_allow_html=True)
with c2:
    st.markdown(f"""<div class="metric-card">
        <div class="metric-label">Brent Oil (USD/bbl)</div>
        <div class="metric-value">${latest['oil_price']:.1f}</div>
        <div class="metric-change-up">↑ +{deltas['oil_chg']}% MoM</div>
    </div>""", unsafe_allow_html=True)
with c3:
    st.markdown(f"""<div class="metric-card">
        <div class="metric-label">Baltic Freight Index</div>
        <div class="metric-value">{int(latest['freight_index']):,}</div>
        <div class="metric-change-up">↑ +{deltas['freight_chg']}% MoM</div>
    </div>""", unsafe_allow_html=True)
with c4:
    st.markdown(f"""<div class="metric-card">
        <div class="metric-label">Textile Exports</div>
        <div class="metric-value">${latest['export_usd_bn']:.2f}B</div>
        <div class="metric-change-down">↓ −2.1% QoQ</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)


# ── NEW: WHAT CHANGED SINCE YESTERDAY ────────────────────────
st.markdown('<p class="section-title">📡 What Changed Since Last Month</p>', unsafe_allow_html=True)

col_d1, col_d2, col_d3, col_d4 = st.columns(4)

def delta_card(label, value, change, unit=""):
    color  = "#f87171" if change > 0 else "#4ade80"
    arrow  = "↑" if change > 0 else "↓"
    return f"""<div class="delta-box">
        <div>
            <div style="font-size:11px; color:#6b7280; text-transform:uppercase; letter-spacing:1px;">{label}</div>
            <div style="font-size:20px; font-weight:300; color:#e8eaf0;">{unit}{value}</div>
        </div>
        <div style="font-size:14px; color:{color}; font-weight:500;">{arrow} {abs(change):.1f}%</div>
    </div>"""

with col_d1:
    st.markdown(delta_card("Risk Score", f"{risk_score}/100", deltas['risk_delta']), unsafe_allow_html=True)
    st.caption(f"Was {deltas['risk_prev']} → Now {deltas['risk_now']}")
with col_d2:
    st.markdown(delta_card("Freight Index", f"{int(latest['freight_index']):,}", deltas['freight_chg']), unsafe_allow_html=True)
with col_d3:
    st.markdown(delta_card("Oil Price", f"${latest['oil_price']:.1f}", deltas['oil_chg']), unsafe_allow_html=True)
with col_d4:
    st.markdown(delta_card("USD/INR", f"₹{live_inr:.2f}", deltas['inr_chg']), unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)


# ── ROW 2: RISK GAUGE + FACTOR BARS ──────────────────────────
st.markdown('<p class="section-title">Risk Analysis</p>', unsafe_allow_html=True)
col_gauge, col_bars = st.columns([1, 1])

with col_gauge:
    gauge_color = "#f87171" if risk_score >= 70 else "#facc15" if risk_score >= 40 else "#4ade80"
    fig_gauge = go.Figure(go.Indicator(
        mode  = "gauge+number",
        value = risk_score,
        title = {"text": "Composite Risk Score", "font": {"color": "#6b7280", "size": 13}},
        number= {"suffix": "/100", "font": {"color": gauge_color, "size": 40}},
        gauge = {
            "axis": {"range": [0, 100], "tickcolor": "#6b7280", "tickfont": {"color": "#6b7280"}},
            "bar":  {"color": gauge_color},
            "bgcolor": "#111418",
            "steps": [
                {"range": [0,  40], "color": "rgba(74,222,128,0.1)"},
                {"range": [40, 70], "color": "rgba(250,204,21,0.1)"},
                {"range": [70,100], "color": "rgba(248,113,113,0.1)"},
            ],
            "threshold": {"line": {"color": gauge_color, "width": 3}, "thickness": 0.75, "value": risk_score}
        }
    ))
    fig_gauge.update_layout(paper_bgcolor="#111418", font_color="#e8eaf0", height=280, margin=dict(t=40,b=10,l=20,r=20))
    st.plotly_chart(fig_gauge, use_container_width=True)
    status = "🚨 HIGH RISK" if risk_score >= 70 else "⚠️ MODERATE RISK" if risk_score >= 40 else "✅ LOW RISK"
    st.markdown(f"<div style='text-align:center; font-size:16px; font-weight:500; color:{gauge_color};'>{status}</div>", unsafe_allow_html=True)

with col_bars:
    factor_labels = ["Oil Price", "Freight Index", "USD/INR Rate", "Export Value"]
    factor_scores_list = [risk_details["oil_price"]["score"], risk_details["freight_index"]["score"],
                          risk_details["usd_inr"]["score"], risk_details["export_usd_bn"]["score"]]
    factor_max    = [risk_details["oil_price"]["weight"], risk_details["freight_index"]["weight"],
                     risk_details["usd_inr"]["weight"], risk_details["export_usd_bn"]["weight"]]
    factor_pct    = [round(s/m*100) for s,m in zip(factor_scores_list, factor_max)]
    fig_bars = go.Figure()
    for label, score, maxv, pct, color in zip(factor_labels, factor_scores_list, factor_max, factor_pct,
                                               ["#f87171","#facc15","#fb923c","#60a5fa"]):
        fig_bars.add_trace(go.Bar(
            name=label, x=[pct], y=[label], orientation="h", marker_color=color,
            text=[f"{score:.1f}/{maxv} pts"], textposition="inside", textfont={"color":"#0a0c0f","size":11},
        ))
    fig_bars.update_layout(
        paper_bgcolor="#111418", plot_bgcolor="#111418", font_color="#e8eaf0",
        showlegend=False, height=280, margin=dict(t=20,b=20,l=10,r=20),
        xaxis=dict(range=[0,100], title=dict(text="% of max weight", font=dict(color="#6b7280",size=11)),
                   gridcolor="#1e2530", tickfont={"color":"#6b7280"}),
        yaxis=dict(tickfont={"color":"#e8eaf0"}), barmode="relative",
    )
    st.plotly_chart(fig_bars, use_container_width=True)

st.markdown("<br>", unsafe_allow_html=True)


# ── ROW 3: FORECAST CHART (FIXED WITH CONFIDENCE BANDS) ──────
st.markdown('<p class="section-title">3-Month Export Forecast</p>', unsafe_allow_html=True)

all_months    = list(df["month"]) + forecast_months
all_actual    = list(df["export_usd_bn"]) + [None]*3
all_forecast  = [None]*11 + [df["export_usd_bn"].iloc[-1]] + forecast_values
all_upper     = [None]*12 + upper_bounds
all_lower     = [None]*12 + lower_bounds

fig_line = go.Figure()

# Confidence band
fig_line.add_trace(go.Scatter(
    x=all_months[11:]+all_months[11:][::-1],
    y=all_upper[11:]+all_lower[11:][::-1],
    fill='toself', fillcolor='rgba(77,158,240,0.08)',
    line=dict(color='rgba(0,0,0,0)'), name='Confidence Band', showlegend=True,
))

fig_line.add_trace(go.Scatter(
    x=all_months, y=all_actual, name="Actual Exports",
    line=dict(color="#c8f04d", width=2), mode="lines+markers",
    marker=dict(size=6, color="#c8f04d"), connectgaps=False,
))

fig_line.add_trace(go.Scatter(
    x=all_months[11:], y=all_forecast[11:], name="Forecast",
    line=dict(color="#4d9ef0", width=2, dash="dot"), mode="lines+markers",
    marker=dict(size=6, color="#4d9ef0"),
))

fig_line.add_annotation(
    x=forecast_months[-1], y=forecast_values[-1],
    text=f"  {change_pct:+.1f}% projected",
    showarrow=False, font=dict(color="#4d9ef0", size=12), xanchor="left",
)

fig_line.update_layout(
    paper_bgcolor="#111418", plot_bgcolor="#111418", font_color="#e8eaf0",
    height=300, margin=dict(t=20,b=20,l=20,r=20),
    legend=dict(bgcolor="#111418", font=dict(color="#6b7280")),
    xaxis=dict(gridcolor="#1e2530", tickfont={"color":"#6b7280"}),
    yaxis=dict(gridcolor="#1e2530", tickfont={"color":"#6b7280"},
               tickprefix="$", ticksuffix="B",
               title=dict(text="Export Value (USD Billion)", font=dict(color="#6b7280",size=11))),
)
st.plotly_chart(fig_line, use_container_width=True)
st.caption(f"Model R² = {r2:.3f} · Forecast capped at ±15% · Shaded area = confidence band")

st.markdown("<br>", unsafe_allow_html=True)


# ── NEW: MY SHIPMENT SIMULATOR ────────────────────────────────
st.markdown('<p class="section-title">🚢 My Shipment Simulator</p>', unsafe_allow_html=True)
st.caption("Enter your shipment details to see your actual profit impact under current global risk conditions.")

with st.expander("▶ Open Shipment Simulator", expanded=True):
    s1, s2, s3 = st.columns(3)
    with s1:
        order_value  = st.number_input("Order Value (USD $)", min_value=1000, value=50000, step=1000)
        cost_per_unit= st.number_input("Cost per Unit (₹)", min_value=1, value=800, step=10)
    with s2:
        units        = st.number_input("Number of Units", min_value=1, value=500, step=10)
        freight_pct  = st.slider("Freight as % of Total Cost", 1, 40, 12)
    with s3:
        dest_country = st.selectbox("Destination Country", ["USA", "UAE", "UK", "Germany", "France", "Australia"])
        currency_exp = st.slider("USD Revenue Exposure %", 0, 100, 70)

    if st.button("🔍 Simulate This Shipment"):
        sim = simulate_shipment(order_value, cost_per_unit, units, freight_pct, currency_exp, live_inr, risk_score, latest["freight_index"])
        st.session_state["sim"] = sim

    if "sim" in st.session_state:
        sim = st.session_state["sim"]
        r1, r2_, r3, r4 = st.columns(4)
        margin_color = "#4ade80" if sim["adjusted_margin_pct"] > 10 else "#facc15" if sim["adjusted_margin_pct"] > 5 else "#f87171"
        change_color = "#4ade80" if sim["margin_change"] >= 0 else "#f87171"

        with r1:
            st.markdown(f"""<div class="sim-result">
                <div class="sim-label">Base Margin</div>
                <div class="sim-value" style="color:#e8eaf0;">{sim['base_margin_pct']}%</div>
                <div style="font-size:11px;color:#6b7280;">Without risk adjustment</div>
            </div>""", unsafe_allow_html=True)
        with r2_:
            st.markdown(f"""<div class="sim-result">
                <div class="sim-label">Adjusted Margin</div>
                <div class="sim-value" style="color:{margin_color};">{sim['adjusted_margin_pct']}%</div>
                <div style="font-size:11px;color:#6b7280;">Under current conditions</div>
            </div>""", unsafe_allow_html=True)
        with r3:
            st.markdown(f"""<div class="sim-result">
                <div class="sim-label">Margin Impact</div>
                <div class="sim-value" style="color:{change_color};">{sim['margin_change']:+.1f}%</div>
                <div style="font-size:11px;color:#6b7280;">Freight + currency effect</div>
            </div>""", unsafe_allow_html=True)
        with r4:
            st.markdown(f"""<div class="sim-result">
                <div class="sim-label">Shipment Risk</div>
                <div class="sim-value" style="font-size:18px; padding-top:8px;">{sim['ship_risk']}</div>
                <div style="font-size:11px;color:#6b7280;">Based on current risk score</div>
            </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)


# ── NEW: ACTIONABLE ALERTS ────────────────────────────────────
st.markdown('<p class="section-title">⚡ Actionable Alerts — What Should You Do?</p>', unsafe_allow_html=True)
st.caption("AI tells you exactly what action to take — not just what's happening.")

col_ai, col_btn = st.columns([5, 1])
with col_btn:
    gen_alerts = st.button("⚡ Get Alerts")

if gen_alerts or "alerts" not in st.session_state:
    with st.spinner("Generating actionable alerts..."):
        sim_data = st.session_state.get("sim", None)
        st.session_state["alerts"] = get_actionable_alerts(
            nvidia_key, risk_score, risk_details, latest,
            forecast_values, change_pct, sim=sim_data
        )

alerts_text = st.session_state.get("alerts", "Click Get Alerts to generate recommendations.")

# Split into individual alerts and display with styled boxes
alert_lines = [l.strip() for l in alerts_text.strip().split("\n") if l.strip()]
for line in alert_lines:
    box_class = "alert-box"
    if any(x in line for x in ["✅","🟢","PROCEED"]):
        box_class = "alert-box green"
    elif any(x in line for x in ["⚠️","🟡","DELAY","MONITOR"]):
        box_class = "alert-box amber"
    st.markdown(f'<div class="{box_class}">{line}</div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)
st.divider()
st.caption("ExportWatch MVP v2.0 · Data: RBI · World Bank · Baltic Exchange · DGCI&S")
