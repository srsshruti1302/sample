import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import IsolationForest
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

st.set_page_config(page_title="GenAI BI Dashboard", layout="wide")

st.title("📊 GenAI-Powered Business Intelligence Dashboard")

# -----------------------------
# FILE UPLOAD
# -----------------------------
uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    st.subheader("📌 Dataset Preview")
    st.dataframe(df.head())

    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = df.select_dtypes(include='object').columns.tolist()

    if len(numeric_cols) == 0:
        st.error("No numeric columns found for analysis.")
    else:

        selected_col = st.selectbox("Select Metric Column", numeric_cols)

        # -----------------------------
        # KPI SECTION
        # -----------------------------
        st.subheader("📈 Key Performance Indicators")

        col1, col2, col3 = st.columns(3)

        total_val = df[selected_col].sum()
        avg_val = df[selected_col].mean()
        max_val = df[selected_col].max()
        min_val = df[selected_col].min()

        col1.metric("Total", round(total_val, 2))
        col2.metric("Average", round(avg_val, 2))
        col3.metric("Maximum", round(max_val, 2))

        # -----------------------------
        # VISUALIZATIONS
        # -----------------------------
        st.subheader("📊 Smart Visualizations")

        # 1️⃣ Trend Line
        st.subheader("📈 Trend Analysis")
        fig1, ax1 = plt.subplots()
        ax1.plot(df[selected_col])
        ax1.set_title("Trend Over Time")
        st.pyplot(fig1)

        # 2️⃣ Correlation Heatmap
        if len(numeric_cols) > 1:
            st.subheader("🔥 Correlation Heatmap")
            corr = df[numeric_cols].corr()

            fig2, ax2 = plt.subplots()
            cax = ax2.matshow(corr)
            plt.xticks(range(len(numeric_cols)), numeric_cols, rotation=90)
            plt.yticks(range(len(numeric_cols)), numeric_cols)
            fig2.colorbar(cax)
            st.pyplot(fig2)

        # 3️⃣ Category Comparison
        if len(categorical_cols) > 0:
            cat_col = st.selectbox("Select Category for Comparison", categorical_cols)
            grouped = df.groupby(cat_col)[selected_col].mean()

            st.subheader("📊 Category Comparison")
            fig3, ax3 = plt.subplots()
            grouped.plot(kind='bar', ax=ax3)
            st.pyplot(fig3)

        # 4️⃣ Histogram
        st.subheader("📉 Distribution")
        fig4, ax4 = plt.subplots()
        ax4.hist(df[selected_col], bins=20)
        st.pyplot(fig4)

        # 5️⃣ Box Plot
        st.subheader("📦 Box Plot")
        fig5, ax5 = plt.subplots()
        ax5.boxplot(df[selected_col])
        st.pyplot(fig5)

        # -----------------------------
        # ANOMALY DETECTION
        # -----------------------------
        st.subheader("🚨 Anomaly Detection")

        model_iso = IsolationForest(contamination=0.05, random_state=42)
        df["Anomaly"] = model_iso.fit_predict(df[[selected_col]])

        anomaly_count = len(df[df["Anomaly"] == -1])

        fig6, ax6 = plt.subplots()
        ax6.scatter(range(len(df)), df[selected_col], c=df["Anomaly"])
        ax6.set_title("Anomaly Detection")
        st.pyplot(fig6)

        # -----------------------------
        # LSTM FORECASTING
        # -----------------------------
        st.subheader("🔮 Forecasting (Deep Learning LSTM)")

        data = df[selected_col].values.reshape(-1, 1)

        if len(data) > 15:

            scaler = MinMaxScaler()
            data_scaled = scaler.fit_transform(data)

            X = []
            y = []

            for i in range(10, len(data_scaled)):
                X.append(data_scaled[i-10:i])
                y.append(data_scaled[i])

            X, y = np.array(X), np.array(y)

            model_lstm = Sequential()
            model_lstm.add(LSTM(50, input_shape=(X.shape[1], 1)))
            model_lstm.add(Dense(1))
            model_lstm.compile(optimizer='adam', loss='mse')

            model_lstm.fit(X, y, epochs=5, verbose=0)

            predictions = model_lstm.predict(X)
            predictions = scaler.inverse_transform(predictions)

            fig7, ax7 = plt.subplots()
            ax7.plot(df[selected_col].values, label="Original")
            ax7.plot(range(10, len(predictions)+10), predictions, label="Predicted")
            ax7.legend()
            st.pyplot(fig7)

        else:
            st.warning("Not enough data for forecasting (need at least 15 rows).")

        # -----------------------------
        # AI GENERATED INSIGHT SUMMARY
        # -----------------------------
        st.subheader("🤖 AI Generated Insights")

        insight = f"""
        The dataset shows a total of {round(total_val,2)}.
        The average value is {round(avg_val,2)}.
        The minimum recorded value is {round(min_val,2)}.
        The maximum recorded value is {round(max_val,2)}.
        {anomaly_count} anomalies were detected in the dataset.
        The forecasting model indicates continuation of recent patterns.
        """

        st.success(insight)

        # -----------------------------
        # SMART QUESTION ANSWERING
        # -----------------------------
        st.subheader("💬 Ask Questions About Data")

        question = st.text_input("Type your business question:")

        if question:

            q = question.lower()

            if "minimum" in q or "lowest" in q:
                answer = f"The minimum value of {selected_col} is {round(min_val,2)}."

            elif "maximum" in q or "highest" in q:
                answer = f"The maximum value of {selected_col} is {round(max_val,2)}."

            elif "average" in q or "mean" in q:
                answer = f"The average value of {selected_col} is {round(avg_val,2)}."

            elif "total" in q or "sum" in q:
                answer = f"The total value of {selected_col} is {round(total_val,2)}."

            elif "anomaly" in q:
                answer = f"There are {anomaly_count} anomalies detected."

            elif "trend" in q:
                answer = "The trend can be observed in the line chart above showing historical progression."

            else:
                answer = "Please ask about total, average, minimum, maximum, trend, or anomalies."

            st.success(answer)

else:
    st.info("Please upload a CSV file to begin analysis.")
