import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="DynNav Research Dashboard", layout="wide")

st.title("🧭 DynNav – Research Analytics Dashboard")
st.markdown("Interactive visualization of statistical validation, t-tests and ablation studies.")

# =========================
# LOAD DATA
# =========================

@st.cache_data
def load_data():
    stats = pd.read_csv("statistical_summary.csv")
    ttests = pd.read_csv("t_test_results.csv")
    return stats, ttests

stats_df, ttest_df = load_data()

# =========================
# SIDEBAR CONTROLS
# =========================

st.sidebar.header("Controls")

metric = st.sidebar.selectbox(
    "Select Metric",
    stats_df["metric"].unique()
)

variants = stats_df["variant"].unique()
selected_variants = st.sidebar.multiselect(
    "Select Variants",
    variants,
    default=list(variants)
)

# =========================
# FILTER DATA
# =========================

filtered = stats_df[
    (stats_df["metric"] == metric) &
    (stats_df["variant"].isin(selected_variants))
]

# =========================
# PLOT: MEAN + 95% CI
# =========================

st.subheader(f"📊 Mean ± 95% CI — Metric: {metric}")

fig = px.bar(
    filtered,
    x="variant",
    y="mean",
    error_y="ci_95",
    title=f"{metric} (Mean ± 95% CI)"
)

st.plotly_chart(fig, use_container_width=True)

# =========================
# RAW STATISTICS TABLE
# =========================

st.subheader("📄 Statistical Summary Table")
st.dataframe(filtered)

# =========================
# T-TEST RESULTS
# =========================

st.subheader("🧪 t-test Results (Learned vs Classic A*)")
st.dataframe(ttest_df)

# Highlight statistically significant results
st.markdown("✅ p < 0.05 → Statistically significant")

# =========================
# INTERPRETATION PANEL
# =========================

st.subheader("🧠 Automatic Interpretation")

for _, row in ttest_df.iterrows():
    metric_name = row["metric"]
    p_val = row["p_value"]

    if p_val < 0.05:
        st.success(f"{metric_name}: statistically significant improvement (p = {p_val:.2e})")
    else:
        st.warning(f"{metric_name}: no statistically significant difference (p = {p_val:.2e})")
