import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import IsolationForest

st.set_page_config(page_title="AI Visual Analytics Dashboard", layout="wide")

st.title("📊 AI Visual Analytics Dashboard with Anomaly Detection")

# -------------------------
# FILE UPLOAD
# -------------------------
uploaded_files = st.file_uploader(
    "Upload Multiple CSV Files",
    type=["csv"],
    accept_multiple_files=True
)

if uploaded_files:

    df_list = []
    for file in uploaded_files:
        temp_df = pd.read_csv(file)
        temp_df["Source_File"] = file.name
        df_list.append(temp_df)

    df = pd.concat(df_list, ignore_index=True)

    st.success(f"{len(uploaded_files)} file(s) merged successfully!")

    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

    if not numeric_cols:
        st.error("No numeric columns found.")
        st.stop()

    metric = st.sidebar.selectbox("Select Metric", numeric_cols)

    # -------------------------
    # KPI SECTION
    # -------------------------
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

    # -------------------------
    # 1️⃣ TREND LINE
    # -------------------------
    st.subheader("📈 Trend Analysis")

    fig_line = px.line(
        df,
        y=metric,
        color="Source_File",
        template="plotly_dark"
    )

    st.plotly_chart(fig_line, use_container_width=True)

    # -------------------------
    # 2️⃣ SCATTER PLOT
    # -------------------------
    st.subheader("🔵 Scatter Plot (Index vs Metric)")

    fig_scatter = px.scatter(
        df,
        x=df.index,
        y=metric,
        color="Source_File",
        template="plotly_dark"
    )

    st.plotly_chart(fig_scatter, use_container_width=True)

    # -------------------------
    # 3️⃣ HISTOGRAM
    # -------------------------
    st.subheader("📊 Distribution Analysis")

    fig_hist = px.histogram(
        df,
        x=metric,
        nbins=30,
        color="Source_File",
        template="plotly_dark"
    )

    st.plotly_chart(fig_hist, use_container_width=True)

    # -------------------------
    # 4️⃣ ANOMALY DETECTION
    # -------------------------
    st.subheader("🚨 Anomaly Detection (Isolation Forest)")

    iso = IsolationForest(contamination=0.05, random_state=42)
    df["Anomaly"] = iso.fit_predict(df[[metric]])

    anomalies = df[df["Anomaly"] == -1]

    fig_anomaly = go.Figure()

    # Normal points
    fig_anomaly.add_trace(go.Scatter(
        x=df.index,
        y=df[metric],
        mode="markers",
        name="Normal",
        marker=dict(color="blue")
    ))

    # Anomalies
    fig_anomaly.add_trace(go.Scatter(
        x=anomalies.index,
        y=anomalies[metric],
        mode="markers",
        name="Anomaly",
        marker=dict(color="red", size=10)
    ))

    fig_anomaly.update_layout(
        template="plotly_dark",
        title="Anomaly Visualization"
    )

    st.plotly_chart(fig_anomaly, use_container_width=True)

    st.info(f"Total Anomalies Detected: {len(anomalies)}")

    # -------------------------
    # 5️⃣ CORRELATION HEATMAP
    # -------------------------
    if len(numeric_cols) > 1:
        st.subheader("🔥 Correlation Heatmap")

        corr = df[numeric_cols].corr()

        fig_heat = go.Figure(data=go.Heatmap(
            z=corr.values,
            x=corr.columns,
            y=corr.columns
        ))

        fig_heat.update_layout(template="plotly_dark")

        st.plotly_chart(fig_heat, use_container_width=True)

else:
    st.info("Upload CSV files to begin analysis.")
