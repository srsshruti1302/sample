# ===================== ONE CELL STREAMLIT BI DASHBOARD =====================

# Install required libraries
!pip -q install streamlit tensorflow scikit-learn matplotlib pandas localtunnel

# Create dataset
import pandas as pd

df = pd.DataFrame({
    "date": ["2024-01-01","2024-01-02","2024-01-03","2024-01-04","2024-01-05","2024-01-06"],
    "sales": [120,135,128,150,160,170],
    "revenue": [24000,26000,25500,30000,32000,34000],
    "customers": [40,45,42,50,55,60]
})
df.to_csv("business_data.csv", index=False)

# Write Streamlit app
with open("app.py", "w") as f:
    f.write("""
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

st.set_page_config(page_title="GenAI BI Dashboard")

st.title("📊 GenAI-Powered Business Intelligence Dashboard")

df = pd.read_csv("business_data.csv")
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values("date")

st.subheader("Business Dataset")
st.dataframe(df)

st.subheader("Sales Trend")
fig, ax = plt.subplots()
ax.plot(df['date'], df['sales'])
ax.set_xlabel("Date")
ax.set_ylabel("Sales")
st.pyplot(fig)

scaler = MinMaxScaler()
sales_scaled = scaler.fit_transform(df[['sales']])

X, y = [], []
for i in range(1, len(sales_scaled)):
    X.append(sales_scaled[i-1])
    y.append(sales_scaled[i])

X = np.array(X).reshape(-1,1,1)
y = np.array(y)

model = Sequential([
    LSTM(50, activation='relu', input_shape=(1,1)),
    Dense(1)
])
model.compile(optimizer='adam', loss='mse')
model.fit(X, y, epochs=30, verbose=0)

if st.button("🔮 Predict Next Sales"):
    last = sales_scaled[-1].reshape(1,1,1)
    pred_scaled = model.predict(last)
    prediction = scaler.inverse_transform(pred_scaled)[0][0]

    st.success(f"Predicted Next Sales: {round(prediction,2)}")

    if prediction > df['sales'].mean():
        st.info("📈 AI Insight: Sales trend is positive. Business growth expected.")
    else:
        st.warning("⚠ AI Insight: Possible decline detected. Strategy review needed.")
""")

# Run Streamlit with public URL
!streamlit run app.py & npx localtunnel --port 8501
