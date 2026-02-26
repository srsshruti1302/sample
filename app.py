import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="GenAI BI Dashboard", layout="wide")

# ----------------------------
# DARK THEME CSS
# ----------------------------
st.markdown("""
<style>
body {
    background-color: #0E1117;
}
.metric-card {
    background-color: #1c1f26;
    padding: 20px;
    border-radius: 10px;
    color: white;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

st.title("📊 GenAI-Powered Business Intelligence Dashboard")

uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = df.select_dtypes(include='object').columns.tolist()

    if len(numeric_cols) == 0:
        st.error("No numeric columns found.")
    else:

        selected_col = st.selectbox("Select Metric", numeric_cols)

        total = df[selected_col].sum()
        avg = df[selected_col].mean()
        max_val = df[selected_col].max()
        min_val = df[selected_col].min()

        # ----------------------------
        # KPI CARDS (Top Row)
        # ----------------------------
        col1, col2, col3, col4 = st.columns(4)

        col1.markdown(f"<div class='metric-card'><h3>Total</h3><h2>{round(total,2)}</h2></div>", unsafe_allow_html=True)
        col2.markdown(f"<div class='metric-card'><h3>Average</h3><h2>{round(avg,2)}</h2></div>", unsafe_allow_html=True)
        col3.markdown(f"<div class='metric-card'><h3>Maximum</h3><h2>{round(max_val,2)}</h2></div>", unsafe_allow_html=True)
        col4.markdown(f"<div class='metric-card'><h3>Minimum</h3><h2>{round(min_val,2)}</h2></div>", unsafe_allow_html=True)

        st.markdown("---")

        # ----------------------------
        # ROW 1 - LINE & BAR
        # ----------------------------
        row1_col1, row1_col2 = st.columns(2)

        with row1_col1:
            fig_line = px.line(df, y=selected_col, title="Trend Analysis")
            fig_line.update_layout(template="plotly_dark")
            st.plotly_chart(fig_line, use_container_width=True)

        with row1_col2:
            fig_bar = px.bar(df, y=selected_col, title="Bar Analysis")
            fig_bar.update_layout(template="plotly_dark")
            st.plotly_chart(fig_bar, use_container_width=True)

        # ----------------------------
        # ROW 2 - PIE & HISTOGRAM
        # ----------------------------
        row2_col1, row2_col2 = st.columns(2)

        if len(categorical_cols) > 0:
            with row2_col1:
                pie_data = df[categorical_cols[0]].value_counts().reset_index()
                pie_data.columns = [categorical_cols[0], "Count"]

                fig_pie = px.pie(pie_data,
                                 names=categorical_cols[0],
                                 values="Count",
                                 title="Category Distribution")
                fig_pie.update_layout(template="plotly_dark")
                st.plotly_chart(fig_pie, use_container_width=True)

        with row2_col2:
            fig_hist = px.histogram(df, x=selected_col, nbins=20, title="Distribution")
            fig_hist.update_layout(template="plotly_dark")
            st.plotly_chart(fig_hist, use_container_width=True)

        # ----------------------------
        # CORRELATION HEATMAP
        # ----------------------------
        if len(numeric_cols) > 1:
            st.markdown("### 🔥 Correlation Heatmap")
            corr = df[numeric_cols].corr()

            fig_heat = go.Figure(data=go.Heatmap(
                z=corr.values,
                x=corr.columns,
                y=corr.columns
            ))
            fig_heat.update_layout(template="plotly_dark")
            st.plotly_chart(fig_heat, use_container_width=True)
