import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

st.set_page_config(page_title="GenAI BI Dashboard", layout="wide")
st.title("📊 GenAI-Powered Business Intelligence Dashboard")

# ---------------- USER INPUT ----------------
st.sidebar.header("User Inputs")

uploaded_file = st.sidebar.file_uploader("Upload Business CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
else:
    df = pd.DataFrame({
        "date": pd.date_range(start="2024-01-01", periods=20),
        "sales": np.random.randint(100, 200, 20),
        "revenue": np.random.randint(20000, 40000, 20),
        "customers": np.random.randint(30, 70, 20)
    })

df['date'] = pd.to_datetime(df['date'])
df = df.sort_values("date")

kpi = st.sidebar.selectbox("Select KPI", ["sales", "revenue", "customers"])
alert_threshold = st.sidebar.slider("Alert Threshold", 0.8, 1.2, 1.0)

st.subheader("📄 Business Dataset")
st.dataframe(df)

# ---------------- VISUALIZATION 1 ----------------
st.subheader("1️⃣ KPI Time Series")
fig1, ax1 = plt.subplots()
ax1.plot(df['date'], df[kpi])
ax1.set_ylabel(kpi)
st.pyplot(fig1)

# ---------------- VISUALIZATION 2 ----------------
st.subheader("2️⃣ KPI Distribution")
fig2, ax2 = plt.subplots()
ax2.hist(df[kpi], bins=10)
st.pyplot(fig2)

# ---------------- VISUALIZATION 3 ----------------
st.subheader("3️⃣ KPI Correlation Heatmap")
fig3, ax3 = plt.subplots()
sns.heatmap(df.drop(columns=['date']).corr(), annot=True, ax=ax3)
st.pyplot(fig3)

# ---------------- VISUALIZATION 4 ----------------
st.subheader("4️⃣ Rolling Average Trend")
df['rolling_avg'] = df[kpi].rolling(window=3).mean()
fig4, ax4 = plt.subplots()
ax4.plot(df['date'], df[kpi], label="Actual")
ax4.plot(df['date'], df['rolling_avg'], label="Rolling Avg")
ax4.legend()
st.pyplot(fig4)

# ---------------- DEEP LEARNING MODEL ----------------
scaler = MinMaxScaler()
scaled = scaler.fit_transform(df[[kpi]])

X, y = [], []
for i in range(1, len(scaled)):
    X.append(scaled[i-1])
    y.append(scaled[i])

X = np.array(X).reshape(-1, 1, 1)
y = np.array(y)

model = Sequential([
    LSTM(50, activation='relu', input_shape=(1,1)),
    Dense(1)
])

model.compile(optimizer='adam', loss='mse')
model.fit(X, y, epochs=20, verbose=0)

# ---------------- PREDICTION ----------------
st.subheader("5️⃣ KPI Forecast & AI Insight")

if st.button("🔮 Predict Next Value"):
    last_val = scaled[-1].reshape(1,1,1)
    pred_scaled = model.predict(last_val)
    prediction = scaler.inverse_transform(pred_scaled)[0][0]

    st.success(f"Predicted Next {kpi.upper()}: {round(prediction,2)}")

    mean_val = df[kpi].mean()

    if prediction > mean_val * alert_threshold:
        st.info("📈 AI Insight: Strong positive business trend detected.")
    else:
        st.warning("⚠ AI Insight: Potential slowdown. Strategic action recommended.")

    # ---------------- VISUALIZATION 5 ----------------
    fig5, ax5 = plt.subplots()
    ax5.plot(df[kpi].values, label="Actual")
    ax5.plot(len(df), prediction, 'ro', label="Predicted")
    ax5.legend()
    st.pyplot(fig5)
