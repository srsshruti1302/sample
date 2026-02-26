import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="AI Executive BI System", layout="wide")

st.title("🚀 AI-Powered Executive Business Intelligence System")

# ----------------------------
# FILE UPLOAD
# ----------------------------
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

numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
categorical_cols = df.select_dtypes(include="object").columns.tolist()

if not numeric_cols:
    st.error("No numeric columns found.")
    st.stop()

metric = st.sidebar.selectbox("Select KPI Metric", numeric_cols)

# ----------------------------
# NAVIGATION TABS
# ----------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Overview",
    "🚨 Risk & Anomaly",
    "🔮 Forecasting & Segmentation",
    "🧠 Executive Report"
])

# ============================
# TAB 1 – OVERVIEW (LEVEL 1)
# ============================
with tab1:

    st.subheader("📌 KPI Overview")

    total = df[metric].sum()
    avg = df[metric].mean()
    max_val = df[metric].max()
    min_val = df[metric].min()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total", f"{total:,.2f}")
    col2.metric("Average", f"{avg:,.2f}")
    col3.metric("Max", f"{max_val:,.2f}")
    col4.metric("Min", f"{min_val:,.2f}")

    st.markdown("---")

    # Trend
    fig_line = px.line(df, y=metric, color="Source_File",
                       title="Trend Analysis", template="plotly_dark")
    st.plotly_chart(fig_line, use_container_width=True)

    # Moving Average
    df["Moving_Avg"] = df[metric].rolling(5).mean()
    fig_ma = px.line(df, y=["Moving_Avg"], title="5-Period Moving Average",
                     template="plotly_dark")
    st.plotly_chart(fig_ma, use_container_width=True)

    # Top / Bottom
    st.subheader("🏆 Top & Bottom Performers")

    if categorical_cols:
        cat = st.selectbox("Select Category", categorical_cols)
        grouped = df.groupby(cat)[metric].sum().reset_index()
        top5 = grouped.sort_values(metric, ascending=False).head()
        bottom5 = grouped.sort_values(metric).head()

        colA, colB = st.columns(2)
        colA.dataframe(top5)
        colB.dataframe(bottom5)

    # Download Button
    st.download_button("Download Filtered Data",
                       df.to_csv(index=False),
                       "filtered_data.csv")

# ============================
# TAB 2 – ANOMALY DETECTION
# ============================
with tab2:

    st.subheader("🚨 Anomaly Detection")

    iso = IsolationForest(contamination=0.05, random_state=42)
    df["Anomaly"] = iso.fit_predict(df[[metric]])
    anomalies = df[df["Anomaly"] == -1]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df[metric],
                             mode="markers", name="Normal"))
    fig.add_trace(go.Scatter(x=anomalies.index, y=anomalies[metric],
                             mode="markers", name="Anomaly",
                             marker=dict(color="red", size=10)))
    fig.update_layout(template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)

    st.info(f"Total anomalies detected: {len(anomalies)}")

# ============================
# TAB 3 – FORECAST + CLUSTER
# ============================
with tab3:

    st.subheader("🔮 Forecasting (Linear Regression)")

    df = df.reset_index()
    X = df.index.values.reshape(-1,1)
    y = df[metric].values

    model = LinearRegression()
    model.fit(X,y)

    future_index = np.arange(len(df), len(df)+10).reshape(-1,1)
    future_pred = model.predict(future_index)

    fig_forecast = go.Figure()
    fig_forecast.add_trace(go.Scatter(x=df.index,
                                      y=df[metric],
                                      mode="lines",
                                      name="Actual"))
    fig_forecast.add_trace(go.Scatter(x=range(len(df), len(df)+10),
                                      y=future_pred,
                                      mode="lines",
                                      name="Forecast",
                                      line=dict(color="green")))
    fig_forecast.update_layout(template="plotly_dark")
    st.plotly_chart(fig_forecast, use_container_width=True)

    # Clustering
    st.subheader("📌 Business Segmentation (KMeans)")

    if len(numeric_cols) >= 2:
        kmeans = KMeans(n_clusters=3, random_state=42)
        df["Cluster"] = kmeans.fit_predict(df[numeric_cols])

        fig_cluster = px.scatter(df,
                                 x=numeric_cols[0],
                                 y=numeric_cols[1],
                                 color="Cluster",
                                 template="plotly_dark")
        st.plotly_chart(fig_cluster, use_container_width=True)
    else:
        st.warning("Need at least 2 numeric columns for clustering.")

# ============================
# TAB 4 – EXECUTIVE REPORT
# ============================
with tab4:

    st.subheader("🧠 AI-Generated Executive Summary")

    growth = "increasing" if future_pred[-1] > avg else "declining"
    risk_flag = "High risk spikes detected" if len(anomalies) > 0 else "No major anomalies"

    report = f"""
    The system analyzed {len(df)} records across {len(uploaded_files)} datasets.

    Total {metric}: {round(total,2)}
    Average {metric}: {round(avg,2)}

    Risk Assessment:
    {risk_flag}

    Forecast Insight:
    Business trend appears {growth} over next periods.

    Strategic Recommendation:
    Align operational planning with forecast trend while monitoring anomaly zones.
    """

    st.success(report)
