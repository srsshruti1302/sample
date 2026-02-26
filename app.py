import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(page_title="AI Executive Analytics Suite", layout="wide")
st.title("🚀 AI-Powered Executive Analytics Suite")

# =====================================================
# FILE UPLOAD
# =====================================================

uploaded_files = st.file_uploader(
    "Upload Business CSV Files",
    type=["csv"],
    accept_multiple_files=True
)

if not uploaded_files:
    st.info("Upload CSV files to begin.")
    st.stop()

df_list = []

for file in uploaded_files:
    try:
        temp = pd.read_csv(file, encoding="utf-8")
    except:
        temp = pd.read_csv(file, encoding="latin1")

    temp["Source_File"] = file.name
    df_list.append(temp)

df = pd.concat(df_list, ignore_index=True)

# =====================================================
# DATA PREP
# =====================================================

numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
categorical_cols = df.select_dtypes(include="object").columns.tolist()

if not numeric_cols:
    st.error("No numeric columns found.")
    st.stop()

metric = st.sidebar.selectbox("Select KPI Metric", numeric_cols)

df[metric] = pd.to_numeric(df[metric], errors="coerce")
df = df.dropna(subset=[metric]).reset_index(drop=True)

# =====================================================
# TIMELINE SLIDER
# =====================================================

st.sidebar.subheader("🕒 Adjust Timeline")

start_idx, end_idx = st.sidebar.slider(
    "Select Data Range",
    0,
    len(df)-1,
    (0, len(df)-1)
)

df_filtered = df.iloc[start_idx:end_idx+1]

# =====================================================
# TABS
# =====================================================

tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Executive Overview",
    "🚨 Risk Intelligence",
    "🔮 Predictive Intelligence",
    "🧠 Strategic Report"
])

# =====================================================
# TAB 1 – EXECUTIVE OVERVIEW
# =====================================================

with tab1:

    total = df_filtered[metric].sum()
    avg = df_filtered[metric].mean()
    max_val = df_filtered[metric].max()
    min_val = df_filtered[metric].min()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total", f"{total:,.2f}")
    col2.metric("Average", f"{avg:,.2f}")
    col3.metric("Max", f"{max_val:,.2f}")
    col4.metric("Min", f"{min_val:,.2f}")

    # =====================================================
    # AREA CHART (NEW)
    # =====================================================

    st.subheader("📈 Area Trend View")

    fig_area = px.area(
        df_filtered,
        x=df_filtered.index,
        y=metric,
        template="plotly_dark"
    )

    st.plotly_chart(fig_area, use_container_width=True)

    
# HISTOGRAM (Replaces Column Chart)
# =====================================================

st.subheader("📊 Distribution Analysis (Histogram)")

fig_hist = px.histogram(
    df_filtered,
    x=metric,
    nbins=30,
    template="plotly_dark"
)

fig_hist.update_layout(
    xaxis_title=metric,
    yaxis_title="Frequency"
)

st.plotly_chart(fig_hist, use_container_width=True)

    # =====================================================
    # TREEMAP
    # =====================================================

    if categorical_cols:
        st.subheader("🌳 Hierarchical Contribution")

        cat = categorical_cols[0]
        tree_data = df_filtered.groupby(cat)[metric].sum().reset_index()

        fig_tree = px.treemap(
            tree_data,
            path=[cat],
            values=metric,
            template="plotly_dark"
        )

        st.plotly_chart(fig_tree, use_container_width=True)

# =====================================================
# TAB 2 – RISK INTELLIGENCE
# =====================================================

with tab2:

    if len(df_filtered) > 10:

        iso = IsolationForest(contamination=0.05, random_state=42)
        df_filtered["Anomaly"] = iso.fit_predict(df_filtered[[metric]])

        fig = px.scatter(
            df_filtered,
            x=df_filtered.index,
            y=metric,
            color=df_filtered["Anomaly"].astype(str),
            template="plotly_dark"
        )

        st.plotly_chart(fig, use_container_width=True)

# =====================================================
# TAB 3 – PREDICTIVE INTELLIGENCE
# =====================================================

with tab3:

    if len(df_filtered) > 5:

        X = np.arange(len(df_filtered)).reshape(-1,1)
        y = df_filtered[metric].values

        model = LinearRegression()
        model.fit(X,y)

        y_pred = model.predict(X)

        future_steps = 10
        future_x = np.arange(len(df_filtered),
                             len(df_filtered)+future_steps).reshape(-1,1)
        future_pred = model.predict(future_x)

        fig_forecast = go.Figure()
        fig_forecast.add_trace(go.Scatter(x=X.flatten(), y=y, mode="lines", name="Actual"))
        fig_forecast.add_trace(go.Scatter(x=X.flatten(), y=y_pred, mode="lines", name="Model Fit"))
        fig_forecast.add_trace(go.Scatter(x=future_x.flatten(), y=future_pred, mode="lines", name="Forecast"))

        fig_forecast.update_layout(template="plotly_dark")
        st.plotly_chart(fig_forecast, use_container_width=True)

# =====================================================
# TAB 4 – STRATEGIC REPORT
# =====================================================

with tab4:

    st.subheader("📋 Executive Summary")

    report_points = [
        f"• Total records analyzed: {len(df_filtered)}",
        f"• KPI selected: {metric}",
        f"• Aggregate value: {round(total,2)}",
        f"• Average performance: {round(avg,2)}",
        f"• Maximum value observed: {round(max_val,2)}",
        f"• Minimum value observed: {round(min_val,2)}",
        "• Area chart highlights overall trend behaviour",
        "• Column chart compares performance across timeline",
        "• Treemap shows categorical contribution distribution",
        "• Anomaly detection executed using Isolation Forest",
        "• Predictive modeling performed using Linear Regression",
        "• Forecast generated for next 10 periods",
        "• Strategic recommendation: Monitor growth and variability closely"
    ]

    for point in report_points:
        st.write(point)
