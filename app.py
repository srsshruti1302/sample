import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Executive BI Dashboard", layout="wide")

# -----------------------------
# HEADER
# -----------------------------
st.markdown("""
    <style>
    .big-title {font-size:28px; font-weight:600;}
    .kpi-card {background-color:#111827; padding:20px; border-radius:10px;}
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="big-title">📊 Executive Business Intelligence Dashboard</div>', unsafe_allow_html=True)

# -----------------------------
# FILE UPLOAD
# -----------------------------
uploaded_files = st.file_uploader(
    "Upload Business Data (Multiple CSV files allowed)",
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

    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = df.select_dtypes(include='object').columns.tolist()

    # -----------------------------
    # SIDEBAR FILTERS
    # -----------------------------
    st.sidebar.header("📌 Filters")

    if len(categorical_cols) > 0:
        filter_column = st.sidebar.selectbox("Select Category Filter", categorical_cols)
        unique_values = df[filter_column].dropna().unique()
        selected_value = st.sidebar.multiselect("Choose Values", unique_values, default=unique_values)
        df = df[df[filter_column].isin(selected_value)]

    metric = st.sidebar.selectbox("Select KPI Metric", numeric_cols)

    # -----------------------------
    # KPI SECTION
    # -----------------------------
    total = df[metric].sum()
    avg = df[metric].mean()
    max_val = df[metric].max()
    min_val = df[metric].min()

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("💰 Total", f"{total:,.2f}")
    col2.metric("📈 Average", f"{avg:,.2f}")
    col3.metric("⬆ Maximum", f"{max_val:,.2f}")
    col4.metric("⬇ Minimum", f"{min_val:,.2f}")

    st.markdown("---")

    # -----------------------------
    # MAIN CHART AREA
    # -----------------------------
    colA, colB = st.columns(2)

    with colA:
        fig_trend = px.line(
            df,
            y=metric,
            color="Source_File",
            title="Performance Trend",
            template="plotly_dark"
        )
        st.plotly_chart(fig_trend, use_container_width=True)

    with colB:
        fig_dist = px.histogram(
            df,
            x=metric,
            color="Source_File",
            title="Distribution Analysis",
            template="plotly_dark"
        )
        st.plotly_chart(fig_dist, use_container_width=True)

    # -----------------------------
    # CONTRIBUTION PIE
    # -----------------------------
    summary = df.groupby("Source_File")[metric].sum().reset_index()

    fig_pie = px.pie(
        summary,
        names="Source_File",
        values=metric,
        title="Revenue Contribution by Data Source",
        template="plotly_dark"
    )

    st.plotly_chart(fig_pie, use_container_width=True)

    # -----------------------------
    # CATEGORY PERFORMANCE
    # -----------------------------
    if len(categorical_cols) > 0:
        category = st.selectbox("Select Category for Comparison", categorical_cols)

        grouped = df.groupby(category)[metric].mean().reset_index()

        fig_bar = px.bar(
            grouped,
            x=category,
            y=metric,
            title="Category Performance Comparison",
            template="plotly_dark"
        )

        st.plotly_chart(fig_bar, use_container_width=True)

    # -----------------------------
    # CORRELATION MATRIX
    # -----------------------------
    if len(numeric_cols) > 1:

        corr = df[numeric_cols].corr()

        fig_heat = go.Figure(data=go.Heatmap(
            z=corr.values,
            x=corr.columns,
            y=corr.columns
        ))

        fig_heat.update_layout(
            title="Correlation Matrix",
            template="plotly_dark"
        )

        st.plotly_chart(fig_heat, use_container_width=True)

    # -----------------------------
    # EXECUTIVE INSIGHTS SECTION
    # -----------------------------
    st.markdown("### 🧠 Executive Insights")

    if total > avg * len(df):
        insight = "Overall performance indicates strong cumulative growth."
    else:
        insight = "Performance consistency needs monitoring."

    if max_val > avg * 1.5:
        anomaly = "High positive spike detected in dataset."
    else:
        anomaly = "No extreme spikes detected."

    st.info(f"""
    • Total {metric} across uploaded files: {total:,.2f}  
    • Average {metric}: {avg:,.2f}  
    • Insight: {insight}  
    • Risk Flag: {anomaly}
    """)

else:
    st.info("Upload business datasets to generate executive dashboard.")
