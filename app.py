import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="GenAI BI Dashboard", layout="wide")

st.title("📊 GenAI-Powered Business Intelligence Dashboard")

# ----------------------------
# MULTIPLE FILE UPLOAD
# ----------------------------
uploaded_files = st.file_uploader(
    "Upload one or more CSV files",
    type=["csv"],
    accept_multiple_files=True
)

if uploaded_files:

    df_list = []

    for file in uploaded_files:
        temp_df = pd.read_csv(file)
        temp_df["Source_File"] = file.name   # Track file origin
        df_list.append(temp_df)

    df = pd.concat(df_list, ignore_index=True)

    st.success(f"{len(uploaded_files)} files merged successfully!")

    st.subheader("📌 Combined Dataset Preview")
    st.dataframe(df.head())

    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = df.select_dtypes(include='object').columns.tolist()

    if len(numeric_cols) == 0:
        st.error("No numeric columns found.")
    else:

        selected_col = st.selectbox("Select Metric for Analysis", numeric_cols)

        # ----------------------------
        # KPI SECTION
        # ----------------------------
        total = df[selected_col].sum()
        avg = df[selected_col].mean()
        max_val = df[selected_col].max()
        min_val = df[selected_col].min()

        col1, col2, col3, col4 = st.columns(4)

        col1.metric("Total", round(total, 2))
        col2.metric("Average", round(avg, 2))
        col3.metric("Maximum", round(max_val, 2))
        col4.metric("Minimum", round(min_val, 2))

        st.markdown("---")

        # ----------------------------
        # TREND CHART
        # ----------------------------
        colA, colB = st.columns(2)

        with colA:
            fig_line = px.line(df, y=selected_col, color="Source_File",
                               title="Trend by File Source")
            fig_line.update_layout(template="plotly_dark")
            st.plotly_chart(fig_line, use_container_width=True)

        with colB:
            fig_hist = px.histogram(df, x=selected_col,
                                    title="Distribution Across Files",
                                    color="Source_File")
            fig_hist.update_layout(template="plotly_dark")
            st.plotly_chart(fig_hist, use_container_width=True)

        # ----------------------------
        # PIE CHART (File Contribution)
        # ----------------------------
        st.subheader("🥧 File Contribution Analysis")

        file_summary = df.groupby("Source_File")[selected_col].sum().reset_index()

        fig_pie = px.pie(file_summary,
                         names="Source_File",
                         values=selected_col,
                         title="Contribution by Uploaded File")
        fig_pie.update_layout(template="plotly_dark")
        st.plotly_chart(fig_pie, use_container_width=True)

        # ----------------------------
        # CATEGORY ANALYSIS (if available)
        # ----------------------------
        if len(categorical_cols) > 1:
            st.subheader("📊 Category Comparison")

            category_col = st.selectbox(
                "Select Category Column",
                [col for col in categorical_cols if col != "Source_File"]
            )

            grouped = df.groupby(category_col)[selected_col].mean().reset_index()

            fig_bar = px.bar(grouped,
                             x=category_col,
                             y=selected_col,
                             title="Category vs Metric")
            fig_bar.update_layout(template="plotly_dark")
            st.plotly_chart(fig_bar, use_container_width=True)

        # ----------------------------
        # CORRELATION HEATMAP
        # ----------------------------
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
    st.info("Upload one or more CSV files to begin analysis.")
