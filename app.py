import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

st.set_page_config(page_title="AI Predictive BI Dashboard", layout="wide")

st.title("🚀 AI-Powered Predictive Business Intelligence Dashboard")

# -----------------------------------------
# FILE UPLOAD
# -----------------------------------------
uploaded_files = st.file_uploader(
    "Upload Multiple CSV Files",
    type=["csv"],
    accept_multiple_files=True
)

if not uploaded_files:
    st.info("Please upload one or more CSV files.")
    st.stop()

df_list = []

for file in uploaded_files:
    try:
        temp_df = pd.read_csv(file, encoding="utf-8")
    except:
        try:
            temp_df = pd.read_csv(file, encoding="latin1")
        except:
            try:
                temp_df = pd.read_csv(file, encoding="ISO-8859-1", encoding_errors="ignore")
            except:
                st.warning(f"Could not read file: {file.name}")
                continue

    if temp_df.empty:
        st.warning(f"File {file.name} is empty. Skipping.")
        continue

    temp_df["Source_File"] = file.name
    df_list.append(temp_df)

# Safety check before concat
if len(df_list) == 0:
    st.error("No valid CSV files were loaded.")
    st.stop()

df = pd.concat(df_list, ignore_index=True)

# -----------------------------------------
# NUMERIC COLUMN CHECK
# -----------------------------------------
numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

if not numeric_cols:
    st.error("No numeric columns found in uploaded data.")
    st.stop()

metric = st.sidebar.selectbox("Select KPI Metric", numeric_cols)

# -----------------------------------------
# KPI SECTION
# -----------------------------------------
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

# -----------------------------------------
# TREND CHART
# -----------------------------------------
st.subheader("📈 Trend Analysis")
fig_line = px.line(df, y=metric, color="Source_File", template="plotly_dark")
st.plotly_chart(fig_line, use_container_width=True)

# -----------------------------------------
# SCATTER PLOT
# -----------------------------------------
st.subheader("🔵 Scatter Plot")
fig_scatter = px.scatter(df, x=df.index, y=metric,
                         color="Source_File",
                         template="plotly_dark")
st.plotly_chart(fig_scatter, use_container_width=True)

# -----------------------------------------
# HISTOGRAM
# -----------------------------------------
st.subheader("📊 Distribution")
fig_hist = px.histogram(df, x=metric, nbins=30,
                        color="Source_File",
                        template="plotly_dark")
st.plotly_chart(fig_hist, use_container_width=True)

# -----------------------------------------
# ANOMALY DETECTION
# -----------------------------------------
st.subheader("🚨 Anomaly Detection (Isolation Forest)")

iso = IsolationForest(contamination=0.05, random_state=42)
df["Anomaly"] = iso.fit_predict(df[[metric]])
anomalies = df[df["Anomaly"] == -1]

fig_anomaly = go.Figure()

fig_anomaly.add_trace(go.Scatter(
    x=df.index,
    y=df[metric],
    mode="markers",
    name="Normal",
    marker=dict(color="blue")
))

fig_anomaly.add_trace(go.Scatter(
    x=anomalies.index,
    y=anomalies[metric],
    mode="markers",
    name="Anomaly",
    marker=dict(color="red", size=10)
))

fig_anomaly.update_layout(template="plotly_dark")
st.plotly_chart(fig_anomaly, use_container_width=True)

st.info(f"Total Anomalies Detected: {len(anomalies)}")

# -----------------------------------------
# LSTM FORECASTING
# -----------------------------------------
st.subheader("🔮 LSTM Deep Learning Forecast")

data = df[metric].values.reshape(-1, 1)

scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data)

sequence_length = 10

if len(scaled_data) > sequence_length:

    X, y = [], []

    for i in range(sequence_length, len(scaled_data)):
        X.append(scaled_data[i-sequence_length:i])
        y.append(scaled_data[i])

    X = np.array(X)
    y = np.array(y)

    model = Sequential()
    model.add(LSTM(50, input_shape=(X.shape[1], 1)))
    model.add(Dense(1))
    model.compile(optimizer="adam", loss="mse")

    model.fit(X, y, epochs=10, batch_size=16, verbose=0)

    last_sequence = scaled_data[-sequence_length:]
    future_predictions = []
    current_sequence = last_sequence

    for _ in range(10):
        prediction = model.predict(current_sequence.reshape(1, sequence_length, 1), verbose=0)
        future_predictions.append(prediction[0, 0])
        current_sequence = np.append(current_sequence[1:], prediction)

    future_predictions = scaler.inverse_transform(
        np.array(future_predictions).reshape(-1, 1)
    )

    fig_forecast = go.Figure()

    fig_forecast.add_trace(go.Scatter(
        y=df[metric],
        mode='lines',
        name="Actual"
    ))

    future_index = list(range(len(df), len(df)+10))

    fig_forecast.add_trace(go.Scatter(
        x=future_index,
        y=future_predictions.flatten(),
        mode='lines',
        name="LSTM Forecast",
        line=dict(color="green")
    ))

    fig_forecast.update_layout(template="plotly_dark")
    st.plotly_chart(fig_forecast, use_container_width=True)

else:
    st.warning("Not enough data for LSTM forecasting.")

# -----------------------------------------
# CORRELATION HEATMAP
# -----------------------------------------
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

# -----------------------------------------
# AI GENERATED REPORT
# -----------------------------------------
st.markdown("## 🧠 AI Generated Executive Report")

trend_direction = "increasing"
if 'future_predictions' in locals():
    if future_predictions[-1] < avg:
        trend_direction = "declining"

report = f"""
The system analyzed {len(df)} records across {len(uploaded_files)} datasets.

Total {metric}: {round(total,2)}
Average {metric}: {round(avg,2)}
Maximum: {round(max_val,2)}
Minimum: {round(min_val,2)}

Isolation Forest detected {len(anomalies)} anomalies.

The LSTM model predicts a {trend_direction} trend
for the next 10 periods.

Strategic planning should align with projected movement
while monitoring anomaly spikes.
"""

st.success(report)
