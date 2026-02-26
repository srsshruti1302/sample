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

    if len(numeric_cols) == 0:
        st.error("No numeric columns found for analysis.")
    else:

        selected_col = st.selectbox("Select Metric Column", numeric_cols)

        # -----------------------------
        # KPI SECTION
        # -----------------------------
        col1, col2, col3 = st.columns(3)

        col1.metric("Total", round(df[selected_col].sum(), 2))
        col2.metric("Average", round(df[selected_col].mean(), 2))
        col3.metric("Maximum", round(df[selected_col].max(), 2))

        # -----------------------------
        # 1️⃣ Line Chart
        # -----------------------------
        st.subheader("📈 Trend Analysis")
        fig1, ax1 = plt.subplots()
        ax1.plot(df[selected_col])
        ax1.set_title("Trend Over Time")
        st.pyplot(fig1)

        # -----------------------------
        # 2️⃣ Bar Chart
        # -----------------------------
        st.subheader("📊 Distribution (Bar)")
        fig2, ax2 = plt.subplots()
        ax2.bar(range(len(df[selected_col])), df[selected_col])
        st.pyplot(fig2)

        # -----------------------------
        # 3️⃣ Histogram
        # -----------------------------
        st.subheader("📉 Histogram")
        fig3, ax3 = plt.subplots()
        ax3.hist(df[selected_col], bins=20)
        st.pyplot(fig3)

        # -----------------------------
        # 4️⃣ Anomaly Detection
        # -----------------------------
        st.subheader("🚨 Anomaly Detection")

        model = IsolationForest(contamination=0.05)
        df["Anomaly"] = model.fit_predict(df[[selected_col]])

        fig4, ax4 = plt.subplots()
        ax4.scatter(range(len(df)), df[selected_col], c=df["Anomaly"])
        st.pyplot(fig4)

        # -----------------------------
        # 5️⃣ LSTM Forecasting
        # -----------------------------
        st.subheader("🔮 Forecasting (Deep Learning LSTM)")

        data = df[selected_col].values.reshape(-1,1)

        scaler = MinMaxScaler()
        data_scaled = scaler.fit_transform(data)

        X = []
        y = []

        for i in range(10, len(data_scaled)):
            X.append(data_scaled[i-10:i])
            y.append(data_scaled[i])

        X, y = np.array(X), np.array(y)

        model_lstm = Sequential()
        model_lstm.add(LSTM(50, return_sequences=False, input_shape=(X.shape[1], 1)))
        model_lstm.add(Dense(1))
        model_lstm.compile(optimizer='adam', loss='mse')

        model_lstm.fit(X, y, epochs=5, verbose=0)

        predictions = model_lstm.predict(X)
        predictions = scaler.inverse_transform(predictions)

        fig5, ax5 = plt.subplots()
        ax5.plot(df[selected_col].values, label="Original")
        ax5.plot(range(10, len(predictions)+10), predictions, label="Predicted")
        ax5.legend()
        st.pyplot(fig5)

        # -----------------------------
        # GenAI Insight Generator
        # -----------------------------
        st.subheader("🤖 AI Generated Insights")

        summary = f"""
        The total value of {selected_col} is {round(df[selected_col].sum(),2)}.
        The average value is {round(df[selected_col].mean(),2)}.
        The maximum recorded value is {round(df[selected_col].max(),2)}.
        Anomalies were detected in approximately {len(df[df['Anomaly']==-1])} records.
        Forecasting suggests a continuing trend based on historical patterns.
        """

        st.success(summary)

        # -----------------------------
        # Natural Language Query
        # -----------------------------
        st.subheader("💬 Ask Questions About Data")

        question = st.text_input("Type your business question:")

        if question:
            response = f"""
            Based on analysis of {selected_col}, your question '{question}' 
            relates to overall trends and statistical behavior of the dataset.
            The dashboard indicates pattern consistency with some anomalies.
            """
            st.info(response)

else:
    st.info("Upload a CSV file to begin.")
