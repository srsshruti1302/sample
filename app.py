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
df = df.replace([np.inf, -np.inf], np.nan)

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

    st.subheader("📌 KPI Summary")

    total = df[metric].sum(skipna=True)
    avg = df[metric].mean(skipna=True)
    max_val = df[metric].max(skipna=True)
    min_val = df[metric].min(skipna=True)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total", f"{total:,.2f}")
    col2.metric("Average", f"{avg:,.2f}")
    col3.metric("Max", f"{max_val:,.2f}")
    col4.metric("Min", f"{min_val:,.2f}")

    growth = ((max_val - min_val) / abs(min_val))*100 if min_val != 0 else 0
    st.metric("Growth %", f"{round(growth,2)}%")

    # =================================================
    # GAUGE CHART
    # =================================================

    st.subheader("🎯 Performance Gauge")

    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=avg,
        title={'text': f"Average {metric}"},
        gauge={
            'axis': {'range': [min_val, max_val]},
            'bar': {'color': "cyan"}
        }
    ))

    fig_gauge.update_layout(template="plotly_dark")
    st.plotly_chart(fig_gauge, use_container_width=True)

    # =================================================
    # WATERFALL CHART
    # =================================================

    st.subheader("💰 Contribution Waterfall")

    sample = df[metric].dropna().head(6)

    fig_water = go.Figure(go.Waterfall(
        y=sample,
        measure=["relative"]*len(sample)
    ))

    fig_water.update_layout(template="plotly_dark")
    st.plotly_chart(fig_water, use_container_width=True)

    # =================================================
    # AREA CHART
    # =================================================

    st.subheader("📈 Area Performance Curve")

    fig_area = px.area(df, y=metric, template="plotly_dark")
    st.plotly_chart(fig_area, use_container_width=True)

    # =================================================
    # RADAR CHART
    # =================================================

    if len(numeric_cols) >= 3:

        st.subheader("🕸 Multi-Metric Radar")

        radar_vals = df[numeric_cols[:5]].mean()

        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(
            r=radar_vals.values,
            theta=radar_vals.index,
            fill='toself'
        ))

        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True)),
            template="plotly_dark"
        )

        st.plotly_chart(fig_radar, use_container_width=True)

# =====================================================
# TAB 2 – RISK INTELLIGENCE
# =====================================================

with tab2:

    st.subheader("🚨 Anomaly Detection")

    clean_df = df.dropna(subset=[metric])

    if len(clean_df) > 10:

        iso = IsolationForest(contamination=0.05, random_state=42)
        clean_df["Anomaly"] = iso.fit_predict(clean_df[[metric]])

        anomalies = clean_df[clean_df["Anomaly"] == -1]

        fig = px.scatter(
            clean_df,
            y=metric,
            color=clean_df["Anomaly"].astype(str),
            template="plotly_dark"
        )

        st.plotly_chart(fig, use_container_width=True)
        st.success(f"Anomalies detected: {len(anomalies)}")

# =====================================================
# TAB 3 – PREDICTIVE INTELLIGENCE
# =====================================================

with tab3:

    st.subheader("🔮 Forecast Model")

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

        st.info(f"R²: {round(r2_score(y,y_pred),3)}")
        st.info(f"MSE: {round(mean_squared_error(y,y_pred),3)}")

    # Clustering
    if len(numeric_cols) >= 2:

        cluster_df = df[numeric_cols].dropna()

        if len(cluster_df) > 10:

            kmeans = KMeans(n_clusters=3, random_state=42)
            cluster_df["Cluster"] = kmeans.fit_predict(cluster_df)

            fig_cluster = px.scatter(
                cluster_df,
                x=numeric_cols[0],
                y=numeric_cols[1],
                color="Cluster",
                template="plotly_dark"
            )

            st.plotly_chart(fig_cluster, use_container_width=True)

    # Correlation Heatmap
    if len(numeric_cols) > 1:

        corr = df[numeric_cols].corr()

        fig_heat = go.Figure(data=go.Heatmap(
            z=corr.values,
            x=corr.columns,
            y=corr.columns,
            colorscale="Viridis"
        ))

        fig_heat.update_layout(template="plotly_dark")
        st.plotly_chart(fig_heat, use_container_width=True)

# =====================================================
# TAB 4 – STRATEGIC REPORT
# =====================================================

with tab4:

    st.subheader("🧠 Executive Summary")

    report = f"""
    Total Records Analyzed: {len(df)}
    KPI Selected: {metric}
    Total Value: {round(total,2)}
    Forecast suggests strategic trend monitoring.
    Risk detection and segmentation completed.
    """

    st.success(report)
