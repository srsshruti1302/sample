import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(page_title="AI Executive BI System", layout="wide")
st.title("🚀 AI-Powered Executive Business Intelligence System")

# =====================================================
# FILE UPLOAD
# =====================================================

uploaded_files = st.file_uploader(
    "Upload Business Data (Multiple CSV files allowed)",
    type=["csv"],
    accept_multiple_files=True
)

if not uploaded_files:
    st.info("Upload CSV files to start analysis.")
    st.stop()

df_list = []

for file in uploaded_files:
    try:
        temp_df = pd.read_csv(file, encoding="utf-8")
    except:
        temp_df = pd.read_csv(file, encoding="latin1")

    temp_df["Source_File"] = file.name
    df_list.append(temp_df)

df = pd.concat(df_list, ignore_index=True)

# =====================================================
# DATA PREPARATION
# =====================================================

numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
categorical_cols = df.select_dtypes(include="object").columns.tolist()

if not numeric_cols:
    st.error("No numeric columns found.")
    st.stop()

metric = st.sidebar.selectbox("Select KPI Metric", numeric_cols)

df[metric] = pd.to_numeric(df[metric], errors="coerce")
df = df.replace([np.inf, -np.inf], np.nan)

# =====================================================
# TABS
# =====================================================

tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Overview",
    "🚨 Risk & Anomaly",
    "🔮 Forecasting & Segmentation",
    "🧠 Executive Report"
])

# =====================================================
# TAB 1 – OVERVIEW
# =====================================================

with tab1:

    st.subheader("📌 KPI Overview")

    total = df[metric].sum(skipna=True)
    avg = df[metric].mean(skipna=True)
    max_val = df[metric].max(skipna=True)
    min_val = df[metric].min(skipna=True)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total", f"{total:,.2f}")
    col2.metric("Average", f"{avg:,.2f}")
    col3.metric("Maximum", f"{max_val:,.2f}")
    col4.metric("Minimum", f"{min_val:,.2f}")

    growth_percent = ((max_val - min_val) / abs(min_val)) * 100 if min_val != 0 else 0
    st.metric("Overall Growth %", f"{round(growth_percent,2)}%")

    st.markdown("---")

    # Trend Chart
    fig_line = px.line(df, y=metric, color="Source_File",
                       template="plotly_dark",
                       title="Trend Analysis")
    st.plotly_chart(fig_line, use_container_width=True)

    # Moving Average
    df["Moving_Avg"] = df[metric].rolling(5).mean()

    fig_ma = px.line(df, y="Moving_Avg",
                     template="plotly_dark",
                     title="5-Period Moving Average")
    st.plotly_chart(fig_ma, use_container_width=True)

    # CUMULATIVE GROWTH
    df["Cumulative_Sum"] = df[metric].cumsum()

    fig_cum = px.line(df, y="Cumulative_Sum",
                      template="plotly_dark",
                      title="Cumulative Growth Trend")
    st.plotly_chart(fig_cum, use_container_width=True)

    # VOLATILITY (Rolling Std Dev)
    df["Rolling_STD"] = df[metric].rolling(5).std()

    fig_vol = px.line(df, y="Rolling_STD",
                      template="plotly_dark",
                      title="Volatility (Rolling Standard Deviation)")
    st.plotly_chart(fig_vol, use_container_width=True)

    # Histogram
    fig_hist = px.histogram(df, x=metric,
                            color="Source_File",
                            nbins=30,
                            template="plotly_dark",
                            title="Metric Distribution")
    st.plotly_chart(fig_hist, use_container_width=True)

# =====================================================
# TAB 2 – ANOMALY DETECTION
# =====================================================

with tab2:

    st.subheader("🚨 Anomaly Detection")

    clean_df = df.dropna(subset=[metric]).copy()

    if len(clean_df) > 10:

        iso = IsolationForest(contamination=0.05, random_state=42)
        clean_df["Anomaly"] = iso.fit_predict(clean_df[[metric]])

        anomalies = clean_df[clean_df["Anomaly"] == -1]

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=clean_df.index.tolist(),
            y=clean_df[metric].tolist(),
            mode="markers",
            name="Normal"
        ))

        fig.add_trace(go.Scatter(
            x=anomalies.index.tolist(),
            y=anomalies[metric].tolist(),
            mode="markers",
            name="Anomaly"
        ))

        fig.update_layout(template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)

        st.success(f"Total anomalies detected: {len(anomalies)}")
    else:
        st.warning("Not enough data.")

# =====================================================
# TAB 3 – FORECASTING + SEGMENTATION
# =====================================================

with tab3:

    st.subheader("🔮 Forecasting")

    forecast_df = df.dropna(subset=[metric]).reset_index(drop=True)

    if len(forecast_df) > 5:

        X = np.arange(len(forecast_df)).reshape(-1,1)
        y = forecast_df[metric].values

        model = LinearRegression()
        model.fit(X,y)

        y_pred = model.predict(X)

        future_steps = 10
        future_x = np.arange(len(forecast_df),
                             len(forecast_df)+future_steps).reshape(-1,1)
        future_pred = model.predict(future_x)

        fig_forecast = go.Figure()
        fig_forecast.add_trace(go.Scatter(x=X.flatten(), y=y, mode="lines", name="Actual"))
        fig_forecast.add_trace(go.Scatter(x=X.flatten(), y=y_pred, mode="lines", name="Model Fit"))
        fig_forecast.add_trace(go.Scatter(x=future_x.flatten(), y=future_pred, mode="lines", name="Forecast"))

        fig_forecast.update_layout(template="plotly_dark")
        st.plotly_chart(fig_forecast, use_container_width=True)

        st.info(f"R² Score: {round(r2_score(y,y_pred),3)}")
        st.info(f"MSE: {round(mean_squared_error(y,y_pred),3)}")

    # Clustering
    if len(numeric_cols) >= 2:

        cluster_df = df[numeric_cols].dropna().copy()

        if len(cluster_df) > 10:

            kmeans = KMeans(n_clusters=3, random_state=42)
            cluster_df["Cluster"] = kmeans.fit_predict(cluster_df)

            fig_cluster = px.scatter(cluster_df,
                                     x=numeric_cols[0],
                                     y=numeric_cols[1],
                                     color="Cluster",
                                     template="plotly_dark",
                                     title="Business Segmentation")
            st.plotly_chart(fig_cluster, use_container_width=True)

# =====================================================
# TAB 4 – EXECUTIVE SUMMARY
# =====================================================

with tab4:

    st.subheader("🧠 Executive Summary")

    report = f"""
    Total Records: {len(df)}
    KPI Selected: {metric}
    Total Value: {round(total,2)}
    Forecast Trend: Strategic monitoring recommended.
    """

    st.success(report)
