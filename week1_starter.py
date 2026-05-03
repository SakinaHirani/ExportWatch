# ============================================================
# ExportWatch — Week 1 Starter Script
# What this does: loads sample trade data and finds patterns
# Libraries needed: pip install pandas
# ============================================================

import pandas as pd

# ── STEP 1: CREATE SAMPLE DATA ───────────────────────────────
# In real life, you'd download this from RBI / World Bank APIs.
# For now, we're using 12 months of realistic sample numbers.

data = {
    "month":        ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                     "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"],

    "usd_inr":      [82.5, 82.8, 83.0, 83.2, 83.5, 83.7,
                     83.4, 83.8, 84.0, 83.9, 84.1, 84.3],   # INR per 1 USD

    "oil_price":    [78,   80,   82,   85,   87,   89,
                     86,   88,   91,   90,   93,   95],       # USD per barrel

    "freight_index":[1800, 1850, 1900, 1980, 2050, 2100,
                     2000, 2080, 2200, 2180, 2300, 2400],     # Baltic Dry Index

    "export_value": [15.1, 15.0, 14.9, 14.7, 14.5, 14.3,
                     14.6, 14.4, 14.1, 14.2, 13.9, 13.6],    # USD Billion
}

# Load it into a DataFrame (like a Python spreadsheet)
df = pd.DataFrame(data)


# ── STEP 2: BASIC EXPLORATION ────────────────────────────────
print("=" * 50)
print("📊 YOUR DATASET — First Look")
print("=" * 50)
print(df.to_string(index=False))


# ── STEP 3: SUMMARY STATISTICS ───────────────────────────────
print("\n" + "=" * 50)
print("📈 SUMMARY STATISTICS")
print("=" * 50)
print(df.describe().round(2))


# ── STEP 4: FIND CORRELATIONS ────────────────────────────────
# Correlation = how strongly two variables move together
# Value close to  1.0 = they rise together
# Value close to -1.0 = one rises, other falls
# Value close to  0.0 = no relationship

print("\n" + "=" * 50)
print("🔗 CORRELATION MATRIX")
print("(How strongly each factor is linked to the others)")
print("=" * 50)

numeric_cols = ["usd_inr", "oil_price", "freight_index", "export_value"]
corr = df[numeric_cols].corr().round(2)
print(corr)


# ── STEP 5: PLAIN ENGLISH INTERPRETATION ────────────────────
print("\n" + "=" * 50)
print("💡 WHAT THE CORRELATIONS MEAN")
print("=" * 50)

export_corr = corr["export_value"]

for factor in ["usd_inr", "oil_price", "freight_index"]:
    val = export_corr[factor]
    if val <= -0.7:
        direction = "📉 Strong NEGATIVE — when this rises, exports fall"
    elif val <= -0.4:
        direction = "↘ Moderate negative relationship"
    elif val >= 0.7:
        direction = "📈 Strong POSITIVE — both rise together"
    else:
        direction = "↔ Weak relationship"
    print(f"  {factor:20s}  r = {val:+.2f}   {direction}")


# ── STEP 6: SIMPLE RISK SIGNAL ───────────────────────────────
# A very basic rule-based risk score (0 to 100)
# We'll make this smarter in Week 3 with a real model

print("\n" + "=" * 50)
print("⚠️  BASIC RISK SIGNAL — Latest Month")
print("=" * 50)

latest = df.iloc[-1]  # Get last row (December)

risk_score = 0

# Rule 1: oil above 90 → add risk
if latest["oil_price"] > 90:
    risk_score += 30
    print("  🔴 Oil price above $90 → +30 risk points")

# Rule 2: freight index above 2200 → add risk
if latest["freight_index"] > 2200:
    risk_score += 30
    print("  🔴 Freight index above 2200 → +30 risk points")

# Rule 3: exports declining (last vs first)
if latest["export_value"] < df.iloc[0]["export_value"]:
    risk_score += 25
    print("  🟡 Export value declining year-on-year → +25 risk points")

# Rule 4: INR weakening (higher number = weaker rupee)
if latest["usd_inr"] > 84:
    risk_score += 15
    print("  🟡 INR weakening past 84 → +15 risk points")

print(f"\n  COMPOSITE RISK SCORE: {risk_score} / 100")

if risk_score >= 70:
    print("  🚨 Status: HIGH RISK — exporters face serious pressure")
elif risk_score >= 40:
    print("  ⚠️  Status: MODERATE RISK — monitor closely")
else:
    print("  ✅ Status: LOW RISK — conditions are stable")

print("\n" + "=" * 50)
print("✅ Week 1 complete! Next up: pulling REAL data from APIs.")
print("=" * 50)
