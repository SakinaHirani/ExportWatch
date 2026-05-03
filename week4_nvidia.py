# ============================================================
# ExportWatch — Week 4: AI Layer using NVIDIA NIM API
# Before running:
#   1. pip install openai (NVIDIA NIM uses OpenAI-compatible SDK)
#   2. Generate a fresh key at: build.nvidia.com
#   3. Paste it in CELL 2 below — NEVER share it anywhere else
# ============================================================


# ── CELL 1: IMPORTS ─────────────────────────────────────────

from openai import OpenAI
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

print("✅ Libraries loaded!")


# ── CELL 2: YOUR API KEY ─────────────────────────────────────
# Paste your fresh NVIDIA NIM key here
# Generate one at: https://build.nvidia.com
# Looks like: nvapi-xxxxxxxxxxxxxxxxxxxx

API_KEY = "paste-your-fresh-nvapi-key-here"   # ← replace this!

client = OpenAI(
    base_url = "https://integrate.api.nvidia.com/v1",
    api_key  = API_KEY
)

print("✅ NVIDIA NIM client ready!")


# ── CELL 3: RUN THE MODEL (from Week 3) ─────────────────────

data = {
    "month": [
        "Oct-23","Nov-23","Dec-23","Jan-24","Feb-24","Mar-24",
        "Apr-24","May-24","Jun-24","Jul-24","Aug-24","Sep-24"
    ],
    "usd_inr":        [83.0, 83.1, 83.2, 83.3, 83.4, 83.5,
                       83.4, 83.5, 83.7, 83.6, 83.8, 84.0],
    "oil_price":      [85.0, 86.0, 87.0, 86.5, 87.5, 88.0,
                       85.5, 87.0, 89.0, 88.0, 90.0, 91.5],
    "freight_index":  [1820, 1850, 1900, 1870, 1910, 1950,
                       1850, 1920, 1780, 1950, 2100, 2250],
    "export_usd_bn":  [3.82, 3.71, 3.65, 3.78, 3.54, 3.61,
                       3.45, 3.52, 3.38, 3.61, 3.44, 3.29],
}

df     = pd.DataFrame(data)
latest = df.iloc[-1]

# Weighted risk score
baselines = {
    "usd_inr":       {"base": 83.0, "weight": 15},
    "oil_price":     {"base": 80.0, "weight": 30},
    "freight_index": {"base": 1800, "weight": 40},
    "export_usd_bn": {"base": 3.70, "weight": 15},
}

risk_score   = 0
risk_details = {}

for col, config in baselines.items():
    base   = config["base"]
    weight = config["weight"]
    actual = latest[col]

    deviation = max(0, min(1,
        (base - actual) / base if col == "export_usd_bn"
        else (actual - base) / base
    ))

    score              = round(deviation * weight, 1)
    risk_score        += score
    risk_details[col]  = {"actual": actual, "score": score, "weight": weight}

risk_score = round(risk_score, 1)

# Forecast
X     = df[["oil_price", "freight_index", "usd_inr"]].values
y     = df["export_usd_bn"].values
model = LinearRegression().fit(X, y)

forecasts = []
for i in range(1, 4):
    proj = np.array([[
        latest["oil_price"]     * (1.020 ** i),
        latest["freight_index"] * (1.015 ** i),
        latest["usd_inr"]       * (1.002 ** i),
    ]])
    forecasts.append(round(model.predict(proj)[0], 2))

change_pct = round(
    ((forecasts[-1] - latest["export_usd_bn"]) / latest["export_usd_bn"]) * 100, 1
)

risk_status = (
    "HIGH RISK"     if risk_score >= 70 else
    "MODERATE RISK" if risk_score >= 40 else
    "LOW RISK"
)

print("\n" + "=" * 50)
print("📊 Model outputs ready — sending to AI...")
print("=" * 50)
print(f"  Risk Score  : {risk_score}/100 — {risk_status}")
print(f"  Forecast    : ${forecasts[0]}B → ${forecasts[1]}B → ${forecasts[2]}B")
print(f"  Change      : {change_pct:+.1f}%")


# ── CELL 4: BUILD THE PROMPT ─────────────────────────────────

prompt = f"""
You are an export risk analyst specializing in Indian textile trade.
Based on the following model outputs, write a clear 2-paragraph insight
for exporters. Be specific with numbers. No bullet points. Plain business English.

Current Indicators (latest month: {latest['month']}):
- USD/INR Rate       : ₹{latest['usd_inr']}
- Oil Price          : ${latest['oil_price']} per barrel
- Freight Index (BDI): {latest['freight_index']}
- Export Value       : ${latest['export_usd_bn']}B

Risk Score Breakdown:
- Oil pressure   : {risk_details['oil_price']['score']}/{risk_details['oil_price']['weight']} pts
- Freight cost   : {risk_details['freight_index']['score']}/{risk_details['freight_index']['weight']} pts
- Exchange rate  : {risk_details['usd_inr']['score']}/{risk_details['usd_inr']['weight']} pts
- Export decline : {risk_details['export_usd_bn']['score']}/{risk_details['export_usd_bn']['weight']} pts
- TOTAL          : {risk_score}/100 — {risk_status}

3-Month Export Forecast:
- Oct-24 : ${forecasts[0]}B
- Nov-24 : ${forecasts[1]}B
- Dec-24 : ${forecasts[2]}B
- Change : {change_pct:+.1f}% from current

Paragraph 1: Current risk situation and what's driving it.
Paragraph 2: Forecast outlook and what exporters should watch.
End with one specific actionable recommendation.
"""


# ── CELL 5: CALL NVIDIA NIM API ──────────────────────────────

print("\n" + "=" * 50)
print("🤖 AI-GENERATED EXPORT RISK INSIGHT")
print("=" * 50)

try:
    response = client.chat.completions.create(
        model      = "meta/llama-3.1-8b-instruct",  # free model on NVIDIA NIM
        messages   = [{"role": "user", "content": prompt}],
        max_tokens = 500,
        temperature= 0.7,
    )

    ai_insight = response.choices[0].message.content
    print(f"\n{ai_insight}")
    print(f"\n  Model used : meta/llama-3.1-8b-instruct (via NVIDIA NIM)")

except Exception as e:
    print(f"\n  ❌ Error: {e}")
    print("  → Make sure your fresh API key is pasted in CELL 2")
    print("  → Generate one at: https://build.nvidia.com")


# ── CELL 6: FINAL SUMMARY ───────────────────────────────────

print("\n" + "=" * 50)
print("📋 FULL SYSTEM OUTPUT")
print("=" * 50)
print(f"""
  ┌─────────────────────────────────────────┐
  │  EXPORTWATCH — {latest['month']} Report           │
  ├─────────────────────────────────────────┤
  │  Risk Score : {risk_score:5.1f} / 100               │
  │  Status     : {risk_status:<28}│
  │  Forecast Δ : {change_pct:+5.1f}% next quarter         │
  │  AI Insight : ✅ Generated above          │
  └─────────────────────────────────────────┘
""")

print("=" * 50)
print("✅ Week 4 complete!")
print("   Your system now thinks AND explains 🧠")
print("   Next: Week 5 — plug everything into a dashboard")
print("=" * 50)
