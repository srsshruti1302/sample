import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import glob
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LinearRegression

# -----------------------------------
# 1️⃣ LOAD MULTIPLE CSV FILES
# -----------------------------------

file_paths = glob.glob("*.csv")

df_list = []

for file in file_paths:
    temp_df = pd.read_csv(file)
    temp_df["Source_File"] = file
    df_list.append(temp_df)

df = pd.concat(df_list, ignore_index=True)

print("Files Loaded:", len(file_paths))
print("Combined Shape:", df.shape)

# -----------------------------------
# 2️⃣ SELECT NUMERIC COLUMN
# -----------------------------------

numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

if not numeric_cols:
    print("No numeric columns found.")
    exit()

metric = numeric_cols[0]
print("\nSelected KPI Metric:", metric)

# -----------------------------------
# 3️⃣ KPI CALCULATIONS
# -----------------------------------

total = df[metric].sum()
avg = df[metric].mean()
max_val = df[metric].max()
min_val = df[metric].min()

print("\n--- KPI SUMMARY ---")
print("Total:", round(total,2))
print("Average:", round(avg,2))
print("Maximum:", round(max_val,2))
print("Minimum:", round(min_val,2))

# -----------------------------------
# 4️⃣ ANOMALY DETECTION
# -----------------------------------

iso = IsolationForest(contamination=0.05, random_state=42)
df["Anomaly"] = iso.fit_predict(df[[metric]])

anomalies = df[df["Anomaly"] == -1]

print("\nAnomalies Detected:", len(anomalies))

# -----------------------------------
# 5️⃣ FORECASTING (Linear Regression)
# -----------------------------------

df = df.reset_index()
X = df.index.values.reshape(-1,1)
y = df[metric].values

model = LinearRegression()
model.fit(X,y)

future_steps = 10
future_index = np.arange(len(df), len(df)+future_steps).reshape(-1,1)
future_pred = model.predict(future_index)

# -----------------------------------
# 6️⃣ VISUALIZATIONS
# -----------------------------------

plt.figure(figsize=(18,12))

# 1. Trend Line
plt.subplot(2,3,1)
plt.plot(df[metric])
plt.title("Trend Analysis")

# 2. Histogram
plt.subplot(2,3,2)
plt.hist(df[metric], bins=20)
plt.title("Distribution")

# 3. Scatter Plot
plt.subplot(2,3,3)
plt.scatter(df.index, df[metric])
plt.title("Scatter Plot (Index vs Metric)")

# 4. Anomaly Plot
plt.subplot(2,3,4)
plt.scatter(df.index, df[metric], label="Normal")
plt.scatter(anomalies.index, anomalies[metric], color="red", label="Anomaly")
plt.legend()
plt.title("Anomaly Detection (Isolation Forest)")

# 5. Forecast Plot
plt.subplot(2,3,5)
plt.plot(df.index, df[metric], label="Actual")
plt.plot(range(len(df), len(df)+future_steps), future_pred, color="green", label="Forecast")
plt.legend()
plt.title("Forecasting (Linear Regression)")

# 6. Correlation Heatmap
plt.subplot(2,3,6)
if len(numeric_cols) > 1:
    corr = df[numeric_cols].corr()
    sns.heatmap(corr, annot=True)
    plt.title("Correlation Matrix")
else:
    plt.text(0.3,0.5,"Not enough numeric columns")
    plt.title("Correlation Matrix")

plt.tight_layout()
plt.show()

# -----------------------------------
# 7️⃣ EXECUTIVE INSIGHTS
# -----------------------------------

print("\n--- AI INSIGHTS ---")

if len(anomalies) > 0:
    print("⚠ Risk Alert: Abnormal activity detected.")
else:
    print("✔ No major anomalies found.")

print("📈 Forecast for next", future_steps, "steps:")
print(future_pred)
