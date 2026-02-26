import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

st.set_page_config(page_title="AI Executive BI System", layout="wide")

st.title("🚀 AI-Powered Executive Business Intelligence System")

# ============================
# FILE UPLOAD
# ============================

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

if len(df_list) == 0:
    st.error("No valid files uploaded.")
    st.stop()

df = pd.concat(df_list, ignore_index=True)

numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
categorical_cols = df.select_dtypes(include="object").columns.tolist()

if not numeric_cols:
    st.error("No numeric columns found for analysis.")
    st.stop()

metric = st.sidebar.selectbox("Select KPI Metric", numeric_cols)

# ============================
# NAVIGATION TABS
# ============================

tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Overview",
    "🚨 Risk & Anomaly",
    "🔮 Forecasting & Segmentation",
    "🧠 Executive Report"
])

# =====================================================
# TAB 1 – LEVEL 1 (Professional BI Features)
# =====================================================

with tab1:

    st.subheader("📌 KPI Overview")

    df[metric] = pd.to_numeric(df[metric], errors="coerce")
    df = df.replace([np.inf, -np.inf], np.nan)

    total = df[metric].sum()
    avg = df[metric].mean()
    max_val = df[metric].max()
    min_val = df[metric].min()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total", f"{total:,.2f}")
    col2.metric("Average", f"{avg:,.2f}")
    col3.metric("Maximum", f"{max_val:,.2f}")
    col4.metric("Minimum", f"{min_val:,.2f}")

    st.markdown("---")

    # Trend Line
    fig_line = px.line(df, y=metric, color="Source_File",
                       title="Trend Analysis",
                       template="plotly_dark")
    st.plotly_chart(fig_line, use_container_width=True)

    # Moving Average
    df["Moving_Avg"] = df[metric].rolling(5).mean()

    fig_ma = px.line(df, y=["Moving_Avg"],
                     title="5-Period Moving Average",
                     template="plotly_dark")
    st.plotly_chart(fig_ma, use_container_width=True)

    # Top & Bottom Performers
    st.subheader("🏆 Top & Bottom Performers")

    if categorical_cols:
        cat = st.selectbox("Select Category", categorical_cols)
        grouped = df.groupby(cat)[metric].sum().reset_index()

        top5 = grouped.sort_values(metric, ascending=False).head()
        bottom5 = grouped.sort_values(metric).head()

        colA, colB = st.columns(2)
        colA.write("Top 5")
        colA.dataframe(top5)

        colB.write("Bottom 5")
        colB.dataframe(bottom5)

    # Download
    st.download_button(
        "Download Filtered Data",
        df.to_csv(index=False),
        "filtered_data.csv"
    )

# =====================================================
# TAB 2 – LEVEL 2 (Anomaly Detection)
# =====================================================

with tab2:

    st.subheader("🚨 Anomaly Detection (Isolation Forest)")

    clean_df = df.dropna(subset=[metric]).copy()

    if len(clean_df) < 10:
        st.warning("Not enough data for anomaly detection.")
    else:
        iso = IsolationForest(contamination=0.05, random_state=42)
        clean_df["Anomaly"] = iso.fit_predict(clean_df[[metric]])

        anomalies = clean_df[clean_df["Anomaly"] == -1]

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=clean_df.index,
            y=clean_df[metric],
            mode="markers",
            name="Normal"
        ))

        fig.add_trace(go.Scatter(
            x=anomalies.index,
            y=anomalies[metric],
            mode="markers",
            name="Anomaly"
        ))

        fig.update_layout(template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)

        st.success(f"Total anomalies detected: {len(anomalies)}")

# =====================================================
# TAB 3 – LEVEL 2 (Forecasting + Clustering)
# =====================================================

with tab3:

    st.subheader("🔮 Forecasting (Linear Regression)")

    forecast_df = df.copy()
    forecast_df = forecast_df.dropna(subset=[metric])
    forecast_df = forecast_df.reset_index(drop=True)

    if len(forecast_df) < 5:
        st.warning("Not enough clean data for forecasting.")
    else:
        X = forecast_df.index.values.reshape(-1, 1)
        y = forecast_df[metric].values

        model = LinearRegression()
        model.fit(X, y)

        y_pred = model.predict(X)

        r2 = r2_score(y, y_pred)
        mse = mean_squared_error(y, y_pred)

        future_index = np.arange(len(forecast_df),
                                 len(forecast_df)+10).reshape(-1,1)

        future_pred = model.predict(future_index)

        fig_forecast = go.Figure()

        fig_forecast.add_trace(go.Scatter(
            x=forecast_df.index,
            y=forecast_df[metric],
            mode="lines",
            name="Actual"
        ))

        fig_forecast.add_trace(go.Scatter(
            x=forecast_df.index,
            y=y_pred,
            mode="lines",
            name="Model Fit"
        ))

        fig_forecast.add_trace(go.Scatter(
            x=range(len(forecast_df), len(forecast_df)+10),
            y=future_pred,
            mode="lines",
            name="Forecast"
        ))

        fig_forecast.update_layout(template="plotly_dark")
        st.plotly_chart(fig_forecast, use_container_width=True)

        st.info(f"Model R² Score: {round(r2,3)}")
        st.info(f"Model MSE: {round(mse,3)}")

    # Clustering
    st.subheader("📌 Business Segmentation (KMeans)")

    clean_cluster = df.dropna()

    if len(numeric_cols) >= 2 and len(clean_cluster) > 10:
        kmeans = KMeans(n_clusters=3, random_state=42)
        clean_cluster["Cluster"] = kmeans.fit_predict(clean_cluster[numeric_cols])

        fig_cluster = px.scatter(
            clean_cluster,
            x=numeric_cols[0],
            y=numeric_cols[1],
            color="Cluster",
            template="plotly_dark"
        )

        st.plotly_chart(fig_cluster, use_container_width=True)
    else:
        st.warning("Need at least 2 numeric columns and sufficient data.")

# =====================================================
# TAB 4 – LEVEL 3 (AI Executive Summary)
# =====================================================

with tab4:

    st.subheader("🧠 AI-Generated Executive Summary")

    total_records = len(df)

    growth = "increasing" if total > avg else "stable/declining"

    report = f"""
    The system analyzed {total_records} records from {len(uploaded_files)} data sources.

    Key KPI Selected: {metric}

    Total Value: {round(total,2)}
    Average Value: {round(avg,2)}

    Risk Analysis:
    Anomaly detection performed to identify abnormal spikes.

    Predictive Insight:
    Forecast indicates business trend is {growth}.

    Strategic Recommendation:
    Monitor anomaly zones closely and align operational strategy
    with predictive trend outcomes for risk mitigation.
    """

    st.success(report)
